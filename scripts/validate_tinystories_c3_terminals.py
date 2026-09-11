#!/usr/bin/env python3
"""Audit C3 saved results, retaining the explicit missing-clipping failure."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

BASE = Path('/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/campaigns/concat-gmc-lmc-v1')


def read(path):
    return json.loads(Path(path).read_text())


def main():
    plan = read(BASE / 'launchers/plan.json')
    sys.path.insert(0, plan['source_dir'])
    from src.evaluation import optimizer_ownership as oo
    from scripts.run_optimizer_ownership_corrections import verify_sources
    verify_sources(plan)
    manifest = oo._read_preflight_manifest(BASE / 'campaign/campaign_manifest.json')
    original = read(plan['reference_manifest'])
    oo._check_content_hash(original, 'content_hash', 'original frozen manifest')
    reference = next(r for r in original['runs'] if r['arm_id'] == 'C3')
    oo._check_sources(reference['sources'])
    output = BASE / 'diagnostics/c3-terminal-validation.json'
    if output.exists():
        raise RuntimeError(f'Occupied validation output: {output}')
    report = {'created_at': time.time(), 'scope': 'C3 saved endpoint, checkpoint and trace audit',
        'strict_artifact_validation': 'failed', 'holdout_evaluated': False,
        'limitation': 'Required C3 clipping sidecars were not recorded because the metrics writer only recognized original arm names. Endpoint and trace validation is separate; strict campaign validation remains failed.',
        'reference_manifest': plan['reference_manifest'], 'reference_endpoints': reference['endpoints'],
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'runs': []}
    for arm in ('C3-GMC', 'C3-LMC'):
        print('Validating:', arm, flush=True)
        run = next(r for r in manifest['runs'] if r['arm_id'] == arm)
        root = BASE / 'runs' / arm
        summary = read(root / 'run_summary.json')
        audit = summary['optimizer_ownership']
        terminal = read(root / 'terminal_validation_results.json')
        endpoints = oo._terminal_endpoints(terminal, run, allow_partial=False)
        try:
            oo._inspect_terminal_run(root, run, manifest['expected_traces'][arm], allow_partial=False)
        except FileNotFoundError as error:
            assert Path(error.filename) == root / 'optimizer_ownership_clipping.jsonl'
            strict_error = str(error)
        else:
            raise AssertionError('Expected strict validation to retain the absent clipping-file failure')
        sources = [oo._source_record(p) for p in (root / 'run_summary.json', root / 'terminal_validation_results.json',
                   Path(terminal['checkpoint_path']), root / 'metrics.csv', root / 'optimizer_ownership_trace.jsonl',
                   root / 'resource_attempts.json')]
        assert sources[2]['sha256'] == terminal['checkpoint_sha256'] == summary['terminal_checkpoint_sha256']
        assert Path(terminal['checkpoint_path']).stat().st_size == terminal['checkpoint_bytes']
        saved = torch.load(terminal['checkpoint_path'], map_location='cpu', weights_only=False)
        expected = {'checkpoint_kind': 'resumable_training', 'checkpoint_schema_version': 1,
            'optimizer_ownership_checkpoint_schema_version': 1, 'run_id': run['run_id'],
            'optimizer_ownership_contract': run['optimizer_ownership_contract'],
            'optimizer_ownership_contract_hash': run['contract_hash'], 'step': 348528, 'tokens_seen': 2855141376,
            'epoch': 4, 'batch_index': 0, 'global_scheduler_position': 348528,
            'optimizer_total_successful_updates': 348528, 'resume_count': 0,
            'optimizer_width_selection_counts': audit['width_selection_counts'],
            'optimizer_quarter_activation_counts': audit['quarter_activation_counts'],
            'optimizer_update_counts': audit['owner_call_counts']}
        for key, value in expected.items(): assert saved[key] == value, key
        assert saved['sampler_state']['total_cursor'] == 22305792
        assert saved['sampler_state']['within_epoch_cursor'] == 0
        assert saved['scheduler_state_dict']['position'] == 348528
        assert saved['scheduler_state_dict']['scheduler_state_dict']['last_epoch'] == 348528
        assert saved['scheduler_state_dict']['scheduler_state_dict']['_last_lr'] == [0.0]
        collection = saved['optimizer_state_collection']
        assert collection['state_scope'] == 'per_ffn_block' and not collection['diagnostic_global_clip']
        assert collection['total_successful_updates'] == 348528
        assert collection['successful_update_counts'] == audit['owner_call_counts']
        assert collection['width_selection_counts'] == audit['width_selection_counts']
        assert collection['current_learning_rates'] == [0.0]
        owners = collection['ordered_owners']
        assert [o['owner_id'] for o in owners] == ['O-A', 'O-B', 'O-C', 'O-D', 'O-common']
        finite_count = 0
        def finite(value):
            nonlocal finite_count
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all().item()
                finite_count += 1
            elif isinstance(value, dict):
                for item in value.values(): finite(item)
            elif isinstance(value, (list, tuple)):
                for item in value: finite(item)
        finite(saved['model_state_dict']); finite(collection)
        descriptors = {d['canonical_name']: d for d in saved['optimizer_parameter_descriptors']}
        seen, history_counts = set(), {}
        for owner in owners:
            state = owner['state_dict']
            ids = [i for group in state['param_groups'] for i in group['params']]
            assert len(ids) == len(owner['descriptors']) == len(state['state'])
            assert set(ids) == set(state['state'])
            for idx, descriptor in zip(ids, owner['descriptors'], strict=True):
                name = descriptor['canonical_name']
                assert name not in seen and descriptor == descriptors[name]
                seen.add(name)
                expected_owner = 'O-' + descriptor['quarter_id'] if descriptor['quarter_id'] else 'O-common'
                assert owner['owner_id'] == expected_owner
                assert owner['active_widths'] == descriptor['gradient_support']
                history = state['state'][idx]
                assert int(history['step'].item()) == audit['owner_call_counts'][expected_owner]
                for key in ('exp_avg', 'exp_avg_sq'):
                    assert list(history[key].shape) == descriptor['shape']
                assert (history['exp_avg_sq'] >= 0).all().item()
            history_counts[owner['owner_id']] = len(ids)
        assert seen == set(descriptors) and len(seen) == 75
        # This supplementary trace-only inspection changes only an in-memory
        # copy. Saved evidence and the strict missing-clipping failure stay intact.
        trace_summary = copy.deepcopy(summary)
        trace_summary['optimizer_ownership']['clipping_path'] = None
        observations = oo.inspect_run_observations(root, trace_summary)
        traces = manifest['expected_traces'][arm]
        assert observations['committed_updates'] == 348528
        assert observations['action_sha256'] == traces['actions']['sha256'] == reference['observations']['action_sha256']
        assert observations['epoch_order_sha256'] == {str(e['epoch_index']): e['sha256'] for e in traces['epochs']}
        assert observations['epoch_order_sha256'] == reference['observations']['epoch_order_sha256']
        assert summary['failed_optimizer_attempts'] == 0 and not summary['unresolved_artifact_failures']
        ledger = read(root / 'resource_attempts.json')
        assert len(ledger['attempts']) == 1
        attempt = next(iter(ledger['attempts'].values()))
        assert attempt['status'] == 'completed' and attempt['failure'] is None
        assert attempt['source_checkpoint'] is None and attempt['attempted_steps'] == 348528
        assert attempt['measurement_complete']
        job = next(j for j in read(BASE / 'launchers/submissions.json')['jobs'] if j['arm_id'] == arm)
        assert job['slurm_accounting']['state'] == 'COMPLETED' and job['slurm_accounting']['exit_code'] == '0:0'
        assert read(BASE / 'launchers' / f"worker-{job['job_id']}.json")['worker_exit_code'] == 0
        losses = 0
        for chunk in pd.read_csv(root / 'metrics.csv', usecols=['loss'], chunksize=50000):
            values = chunk['loss'].dropna().to_numpy(dtype=float)
            assert np.isfinite(values).all()
            losses += len(values)
        assert losses > 0 and not list(root.glob('final_holdout_results*'))
        oo._check_sources(sources)
        report['runs'].append({'arm_id': arm, 'job_id': job['job_id'], 'sources': sources,
            'checkpoint_validation': 'passed', 'strict_artifact_validation': 'failed', 'strict_error': strict_error,
            'trace_validation': 'passed (separate from missing clipping evidence)', 'paired_with_original_C3': True,
            'checkpoint_sha256': terminal['checkpoint_sha256'], 'terminal_source': sources[1], 'endpoints': endpoints,
            'observations': observations, 'actual_epochs': 4, 'actual_updates': 348528, 'actual_tokens': 2855141376,
            'finite_model_and_optimizer_tensors': finite_count, 'optimizer_histories_per_owner': history_counts,
            'owner_call_counts': audit['owner_call_counts'], 'width_selection_counts': audit['width_selection_counts'],
            'finite_metric_loss_rows': losses, 'resource_attempt': attempt,
            'slurm_elapsed_seconds': job['slurm_accounting']['elapsed_seconds']})
        print(arm, 'endpoint/checkpoint/trace audit passed; clipping artifact absent', flush=True)
    with output.open('x') as stream: json.dump(report, stream, indent=2, allow_nan=False)
    print('Evidence:', output, flush=True)


if __name__ == '__main__':
    main()
