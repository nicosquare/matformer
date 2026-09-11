#!/usr/bin/env python3
"""Validate completed C2 correction runs using saved CPU-readable artifacts."""
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
    original = read(plan['reference_manifest'])
    oo._check_content_hash(original, 'content_hash', 'original frozen manifest')
    reference = next(r for r in original['runs'] if r['arm_id'] == 'C2')
    oo._check_sources(reference['sources'])
    arms = ('C2-GMC', 'C2-LMC')
    output = BASE / 'diagnostics/c2-terminal-validation.json'
    if output.exists():
        raise RuntimeError(f'Occupied validation output: {output}')
    print('Validating and freezing both complete C2 runs; snapshot covers only two of six campaign arms.', flush=True)
    frozen = oo.freeze_campaign(
        campaign_manifest=BASE / 'campaign/campaign_manifest.json',
        run_dirs=[BASE / 'runs' / arm for arm in arms],
        output_dir=BASE / 'diagnostics/c2-terminal-freeze-20260911', allow_partial=True)
    assert len(frozen['runs']) == 2
    report = {'created_at': time.time(), 'scope': 'C2-GMC and C2-LMC terminal validation',
              'strict_artifact_validation': 'passed', 'full_six_arm_campaign_validation': 'incomplete',
              'holdout_evaluated': False, 'limitation': None,
              'reference_manifest': plan['reference_manifest'],
              'reference_endpoints': reference['endpoints'],
              'frozen_manifest': str(BASE / 'diagnostics/c2-terminal-freeze-20260911/frozen_manifest.json'),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'runs': []}
    for arm in arms:
        print('Checking finite states and per-width histories:', arm, flush=True)
        validated = next(r for r in frozen['runs'] if r['arm_id'] == arm)
        root = BASE / 'runs' / arm
        terminal = read(root / 'terminal_validation_results.json')
        summary = read(root / 'run_summary.json')
        assert {e['width'] for e in validated['endpoints']} == set(oo.WIDTH_LABELS)
        assert terminal['actual_epochs'] == 4 and terminal['actual_updates'] == 348528
        assert terminal['actual_tokens'] == 2855141376
        for key in ('action_sha256', 'epoch_order_sha256'):
            assert validated['observations'][key] == reference['observations'][key]
        saved = torch.load(terminal['checkpoint_path'], map_location='cpu', weights_only=False)
        assert saved['resume_count'] == 0
        assert saved['sampler_state']['total_cursor'] == 22305792
        assert saved['sampler_state']['within_epoch_cursor'] == 0
        assert saved['scheduler_state_dict']['scheduler_state_dict']['last_epoch'] == 348528
        assert saved['scheduler_state_dict']['scheduler_state_dict']['_last_lr'] == [0.0]
        collection = saved['optimizer_state_collection']
        counts = saved['optimizer_width_selection_counts']
        assert collection['successful_update_counts'] == counts
        assert collection['current_learning_rates'] == [0.0]
        assert [e['granularity'] for e in collection['ordered_entries']] == list(oo.WIDTH_LABELS)
        descriptors = saved['optimizer_parameter_descriptors']
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
        history_counts = {}
        for entry in collection['ordered_entries']:
            width, state = entry['granularity'], entry['state_dict']
            ids = [i for group in state['param_groups'] for i in group['params']]
            assert len(ids) == len(descriptors) == 75
            expected_ids = set()
            for idx, descriptor in zip(ids, descriptors, strict=True):
                if width not in descriptor['gradient_support']:
                    assert idx not in state['state'], 'Inactive block acquired a width-specific history'
                    continue
                expected_ids.add(idx)
                history = state['state'][idx]
                assert int(history['step'].item()) == counts[width]
                for key in ('exp_avg', 'exp_avg_sq'):
                    assert list(history[key].shape) == descriptor['shape']
                assert (history['exp_avg_sq'] >= 0).all().item()
            assert set(state['state']) == expected_ids
            history_counts[width] = len(expected_ids)
        assert summary['failed_optimizer_attempts'] == 0 and not summary['unresolved_artifact_failures']
        ledger = read(root / 'resource_attempts.json')
        assert len(ledger['attempts']) == 1
        attempt = next(iter(ledger['attempts'].values()))
        assert attempt['status'] == 'completed' and attempt['failure'] is None
        assert attempt['source_checkpoint'] is None and attempt['attempted_steps'] == 348528
        assert attempt['measurement_complete']
        job = next(j for j in read(BASE / 'launchers/submissions.json')['jobs'] if j['arm_id'] == arm)
        assert job['slurm_accounting']['state'] == 'COMPLETED'
        assert job['slurm_accounting']['exit_code'] == '0:0'
        worker = read(BASE / 'launchers' / f"worker-{job['job_id']}.json")
        assert worker['worker_exit_code'] == 0
        losses = 0
        for chunk in pd.read_csv(root / 'metrics.csv', usecols=['loss'], chunksize=50000):
            values = chunk['loss'].dropna().to_numpy(dtype=float)
            assert np.isfinite(values).all()
            losses += len(values)
        assert losses > 0 and not list(root.glob('final_holdout_results*'))
        oo._check_sources(validated['sources'])
        report['runs'].append({**validated, 'job_id': job['job_id'], 'scheduler_status': 'COMPLETED',
            'exit_code': '0:0', 'strict_artifact_validation': 'passed', 'checkpoint_validation': 'passed',
            'trace_validation': 'passed', 'paired_with_original_C2': True,
            'finite_model_and_optimizer_tensors': finite_count, 'optimizer_histories_per_width': history_counts,
            'width_selection_counts': counts, 'quarter_activation_counts': saved['optimizer_quarter_activation_counts'],
            'finite_metric_loss_rows': losses, 'actual_epochs': 4, 'actual_updates': 348528,
            'actual_tokens': 2855141376, 'resource_attempt': attempt,
            'slurm_elapsed_seconds': job['slurm_accounting']['elapsed_seconds'],
            'terminal_source': oo._source_record(root / 'terminal_validation_results.json')})
        print(arm, 'PASS', history_counts, flush=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('Evidence:', output, flush=True)


if __name__ == '__main__':
    main()
