#!/usr/bin/env python3
"""Real-control five-arm bf16 diagnostic; submit through sbatch only."""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from src.evaluation.optimizer_ownership import INVERSE_MEMBERSHIP_ARM_IDS, expected_action_trace
from src.training import run, steps, checkpointing
from src.utils.config import resolve_run_config
from src.utils.reproducibility import stable_hash, deterministic_runtime_settings
from scripts import run_tinystories_inverse_membership as operations


class DiagnosticBoundary(KeyboardInterrupt):
    pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root',type=Path,default=operations.BASE)
    args=parser.parse_args()
    base=args.campaign_root.resolve()
    job=os.environ.get('SLURM_JOB_ID')
    if not job: raise RuntimeError('Submit every GPU workload through sbatch')
    operations.BASE=base
    plan=json.loads((base/'launchers/plan.json').read_text())
    operations.verify_sources(plan)
    output=base/'diagnostics'/f'gpu-runtime-{job}'
    output.mkdir(exist_ok=False)
    manifest=json.loads((base/'campaign/campaign_manifest.json').read_text())
    original_determinism=run.configure_strict_determinism
    established=None
    def configure_once(config):
        nonlocal established
        if not torch.cuda.is_initialized(): established=original_determinism(config)
        else: assert deterministic_runtime_settings()==established
        return established
    run.configure_strict_determinism=configure_once
    original_train=steps.train_for_steps
    original_save=checkpointing.maybe_write_latest_checkpoint
    results=[]
    for definition in manifest['runs']:
        name=definition['arm_id']
        raw=copy.deepcopy(definition['executable_config'])
        identity=f'tinystories-optimizer-ownership-im-gpu-{job}'
        raw['run'].update(campaign_id=identity,run_id=f'{identity}-{name}-s42',output_dir=str(output/'runs'/name))
        raw['optimizer_ownership_contract'].update(campaign_id=identity,run_id=raw['run']['run_id'])
        raw['optimizer_ownership_contract_hash']=stable_hash(raw['optimizer_ownership_contract'])
        config_file=output/f'{name}.yaml'
        config_file.write_text(yaml.safe_dump(raw,sort_keys=False))
        timing,progress=[],[]
        for stop in (192,256):
            config=resolve_run_config(config_file)
            def train_window(config,model,train_loader,eval_loader,optimizer,scheduler,device,**kw):
                state=kw['run_state']
                start=state['last_completed_step']
                assert start==(0 if stop==192 else 192)
                assert not state.get('optimizer_poisoned') and not state.get('update_in_flight')
                callback=kw.get('successful_step_callback')
                previous=time.perf_counter()
                def committed(*,step,tokens_seen):
                    nonlocal previous
                    torch.cuda.synchronize()
                    now=time.perf_counter()
                    if step>start+16 and step>80 and (step-1)%64!=0: timing.append(now-previous)
                    previous=now
                    sampling=state['global_sampling_state']
                    assert sampling['sampling_policy']=='fixed_inverse_membership'
                    assert sampling['total_successful_updates']==step
                    assert sum(sampling['exposure_counts'].values())==step
                    assert state['global_scheduler_position']==step
                    if name=='C3-IM': assert state['optimizer_update_counts']['O-common']==step
                    if step==stop:
                        assert all(torch.isfinite(p).all() for p in model.parameters())
                        progress.append({'restored_step':start,'stopped_step':step,'tokens':tokens_seen})
                    if callback: callback(step=step,tokens_seen=tokens_seen)
                kw['successful_step_callback']=committed
                return original_train(config,model,train_loader,eval_loader,optimizer,scheduler,device,**kw)
            def save_boundary(*a,**kw):
                result=original_save(*a,**kw)
                if kw.get('reason')=='validation' and kw.get('step')==stop:
                    raise DiagnosticBoundary('intentional durable diagnostic boundary')
                return result
            steps.train_for_steps=train_window
            checkpointing.maybe_write_latest_checkpoint=save_boundary
            try: run.run_training(config)
            except DiagnosticBoundary: pass
            finally:
                steps.train_for_steps=original_train
                checkpointing.maybe_write_latest_checkpoint=original_save
            checkpoint=Path(raw['run']['output_dir'])/'checkpoints/latest.pt'
            payload=torch.load(checkpoint,weights_only=False,map_location='cpu')
            assert payload['step']==stop
            assert payload['optimizer_ownership_contract_hash']==raw['optimizer_ownership_contract_hash']
            assert not payload.get('optimizer_poisoned') and not payload.get('update_in_flight')
        root=Path(raw['run']['output_dir'])
        with (root/'metrics.csv').open() as handle: metrics=list(csv.DictReader(handle))
        losses=[float(r['loss']) for r in metrics if r.get('loss') and r['split'] in ('train','validation') and not r.get('optimizer_failure_stage')]
        assert losses and all(math.isfinite(v) for v in losses)
        trace=[json.loads(line) for line in (root/'optimizer_ownership_trace.jsonl').read_text().splitlines()]
        assert len(trace)==256
        # Hash actual production selections after restore against the seeded policy.
        selected=[r['width'] if 'width' in r else r['selected_width'] for r in trace]
        expected_config=copy.deepcopy(config); expected_config['training']['max_steps']=256
        expected=expected_action_trace(expected_config)
        actual_hash=hashlib.sha256(''.join(w+'\n' for w in selected).encode()).hexdigest()
        assert actual_hash==expected['sha256']
        assert payload['optimizer_width_selection_counts']==expected['counts']
        assert len(timing)>=100
        result={'arm_id':name,'status':'passed','progress':progress,'action_sha256':actual_hash,
                'width_selection_counts':expected['counts'],'owner_counts':payload['optimizer_update_counts'],
                'recent_update_seconds_median':statistics.median(timing[-40:]),
                'steady_update_seconds_median':statistics.median(timing),'timed_updates':len(timing),
                'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'checkpoint_bytes':checkpoint.stat().st_size,'diagnostic_config':str(config_file),
                'resource_attempts':json.loads((root/'resource_attempts.json').read_text()),
                'timing_scope':'committed update intervals excluding startup, warmup and post-validation intervals'}
        results.append(result)
        operations.save(output/'measurements.json',results)
        print('PREFLIGHT',name,result['recent_update_seconds_median'],flush=True)
    assert tuple(r['arm_id'] for r in results)==INVERSE_MEMBERSHIP_ARM_IDS
    assert len({r['action_sha256'] for r in results})==1
    operations.verify_sources(plan)
    record={'status':'passed','job_id':job,'device':torch.cuda.get_device_name(),'results':results,
            'runtime_source_sha256':plan['runtime_source_sha256'],'holdout_evaluated':False,
            'kind':'real_shape_batch_corpus_bf16_interruption_resume','output_dir':str(output)}
    operations.save(output/'gpu-gate.json',record)
    operations.save(base/'diagnostics/gpu-gate.json',record)


if __name__=='__main__': main()
