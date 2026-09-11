#!/usr/bin/env python3
"""GPU smoke/resume and overhead evidence at the real campaign shape/batch/data.

Keep the full scientific horizon. Deliberately interrupt at committed boundaries,
then restore each diagnostic's own durable checkpoint. No terminal/holdout eval.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from src.evaluation.optimizer_ownership import CORRECTION_ARM_IDS
from src.training import run, steps, checkpointing
from src.utils.config import resolve_run_config
from src.utils.reproducibility import stable_hash

BASE = Path('/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/campaigns/concat-gmc-lmc-v1')


class DiagnosticBoundary(KeyboardInterrupt):
    pass


def main():
    job = os.environ.get('SLURM_JOB_ID')
    if not job: raise RuntimeError('Submit every GPU workload through sbatch')
    # run_training configures strict determinism before the first CUDA context.
    output = BASE/'diagnostics'/f'gpu-runtime-{job}'
    output.mkdir(exist_ok=False)
    manifest = json.loads((BASE/'campaign/campaign_manifest.json').read_text())
    original = json.loads((BASE.parent.parent/'campaign/campaign_manifest.json').read_text())
    definitions = [(r, r['arm_id']) for r in original['runs'] if r['arm_id'] in ('C1','C2','C3')]
    definitions += [(r, r['arm_id']) for r in manifest['runs']]
    plan = json.loads((BASE/'launchers/plan.json').read_text())
    results = []
    # Diagnostic bundles share a process; production still requires configuration
    # before CUDA initialization. On reuse verify the settings established first.
    original_determinism = run.configure_strict_determinism
    expected_determinism = None
    def configure_once(config):
        nonlocal expected_determinism
        from src.utils.reproducibility import deterministic_runtime_settings
        if not torch.cuda.is_initialized():
            expected_determinism = original_determinism(config)
        else:
            assert deterministic_runtime_settings() == expected_determinism
        return expected_determinism
    run.configure_strict_determinism = configure_once
    original_train = steps.train_for_steps
    original_apply = steps._apply_concat_lmc_corrections
    original_save = checkpointing.maybe_write_latest_checkpoint
    for definition, name in definitions:
        config_file = output/f'{name}.yaml'
        raw = copy.deepcopy(definition['executable_config'])
        diagnostic_id = f'tinystories-optimizer-ownership-gpu-{job}'
        raw['run']['campaign_id'] = diagnostic_id
        raw['run']['run_id'] = f'{diagnostic_id}-{name}-s42'
        raw['run']['output_dir'] = str(output/'runs'/name)
        contract = raw['optimizer_ownership_contract']
        contract['campaign_id'] = diagnostic_id
        contract['run_id'] = raw['run']['run_id']
        # Preserve scientific initialization method, record actual diagnostic code.
        contract['initialization'].update(manifest['runs'][0]['initialization'])
        raw['optimizer_ownership_contract_hash'] = stable_hash(contract)
        config_file.write_text(yaml.safe_dump(raw, sort_keys=False))
        mode = raw['model']['correction_mode']
        timing, corrections, progress = [], [], []
        for stop in (192, 256):
            config = resolve_run_config(config_file)
            def train_window(config, model, train_loader, eval_loader, optimizer, scheduler, device, **kw):
                state = kw['run_state']
                start_step = state['last_completed_step']
                assert start_step == (0 if stop == 192 else 192), (name, start_step)
                assert not state.get('optimizer_poisoned') and not state.get('update_in_flight')
                for layer in model.model.layers:
                    assert list(layer.mlp.gradient_membership_counts) == [4,3,2,1]
                    assert list(layer.mlp.gradient_membership_correction_scales) == [1.,4/3,2.,4.]
                callback = kw.get('successful_step_callback')
                previous_time = time.perf_counter()
                application_count = 0
                def apply(snapshots):
                    nonlocal application_count
                    assert mode == 'lmc'
                    assert all(p.grad is not None for p, _, _ in snapshots)
                    assert len({id(p) for p, _, _ in snapshots}) == len(snapshots)
                    original_apply(snapshots)
                    application_count += 1
                    corrections.append(sorted(set(scale for _, _, scale in snapshots)))
                steps._apply_concat_lmc_corrections = apply
                def committed(*, step, tokens_seen):
                    nonlocal previous_time, application_count
                    torch.cuda.synchronize()
                    now = time.perf_counter()
                    width = state['last_optimizer_action']['granularities'][0] if 'last_optimizer_action' in state else state['global_sampling_state']['held_granularity']
                    # C1/C2 call the helper for empty snapshots; C3 skips an empty list.
                    expected_calls = int(mode == 'lmc' and (name[:2] != 'C3' or width != 'g250'))
                    assert application_count == expected_calls, (name, width, application_count, expected_calls)
                    application_count = 0
                    # Exclude startup, warmup, and iterations immediately after validation.
                    if step > start_step + 16 and step > 80 and (step-1) % 64 != 0:
                        timing.append(now-previous_time)
                    previous_time = now
                    if step == stop:
                        assert all(torch.isfinite(p).all() for p in model.parameters())
                    if callback: callback(step=step, tokens_seen=tokens_seen)
                    if step == stop:
                        progress.append({'restored_step': start_step, 'stopped_step': step, 'tokens': tokens_seen})
                        # Interrupt only after the normal validation checkpoint below.
                        # The successful-step callback precedes metric publication.
                kw['successful_step_callback'] = committed
                return original_train(config, model, train_loader, eval_loader, optimizer, scheduler, device, **kw)
            def save_boundary(*args, **kwargs):
                value = original_save(*args, **kwargs)
                if kwargs.get('reason') == 'validation' and kwargs.get('step') == stop:
                    raise DiagnosticBoundary('intentional durable diagnostic boundary')
                return value
            checkpointing.maybe_write_latest_checkpoint = save_boundary
            steps.train_for_steps = train_window
            try:
                run.run_training(config)
            except DiagnosticBoundary:
                pass
            finally:
                steps.train_for_steps = original_train
                steps._apply_concat_lmc_corrections = original_apply
                checkpointing.maybe_write_latest_checkpoint = original_save
            checkpoint = Path(raw['run']['output_dir'])/'checkpoints/latest.pt'
            payload = torch.load(checkpoint, weights_only=False, map_location='cpu')
            assert payload['step'] == stop
            assert payload['optimizer_ownership_contract_hash'] == raw['optimizer_ownership_contract_hash']
            assert not payload.get('optimizer_poisoned') and not payload.get('update_in_flight')
        root = Path(raw['run']['output_dir'])
        import csv, math
        with (root/'metrics.csv').open() as handle:
            metrics = list(csv.DictReader(handle))
        losses = [float(r['loss']) for r in metrics if r.get('loss') and r['split'] in ('train','validation') and not r.get('optimizer_failure_stage')]
        assert losses and all(math.isfinite(v) for v in losses)
        assert len(timing) >= 100
        result = {'arm_id': name, 'mode': mode, 'status': 'passed', 'progress': progress,
                  'membership_counts': [4,3,2,1], 'factors': [1.,4/3,2.,4.],
                  'applied_lmc_scales': sorted({s for values in corrections for s in values}),
                  'recent_update_seconds_median': statistics.median(timing[-40:]),
                  'steady_update_seconds_median': statistics.median(timing), 'timed_updates': len(timing),
                  'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  'checkpoint_bytes': checkpoint.stat().st_size, 'diagnostic_config': str(config_file),
                  'resource_attempts': json.loads((root/'resource_attempts.json').read_text()),
                  'timing_scope': 'committed update intervals; excludes startup, warmup and post-validation intervals; finite-state checks occur only after the stop-boundary timing'}
        if mode == 'lmc': assert result['applied_lmc_scales'] == [4/3,2.,4.]
        results.append(result)
        (output/'measurements.json').write_text(json.dumps(results, indent=2))
        print('PREFLIGHT', name, result['recent_update_seconds_median'], flush=True)
    baseline = {r['arm_id']: r['recent_update_seconds_median'] for r in results[:3]}
    for result in results[3:]:
        result['relative_to_same_job_none'] = result['recent_update_seconds_median']/baseline[result['arm_id'][:2]]
    assert {r['arm_id'] for r in results[3:]} == set(CORRECTION_ARM_IDS)
    record = {'status': 'passed', 'job_id': job, 'device': torch.cuda.get_device_name(), 'results': results,
              'runtime_source_sha256': plan['runtime_source_sha256'], 'holdout_evaluated': False,
              'kind': 'real_shape_batch_corpus_bf16_interruption_resume', 'output_dir': str(output)}
    (output/'gpu-gate.json').write_text(json.dumps(record, indent=2))
    (BASE/'diagnostics/gpu-gate.json').write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
