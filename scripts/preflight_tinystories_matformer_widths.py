#!/usr/bin/env python3
"""Run snapshot-bound CPU checks or separately authorized sbatch GPU probes."""
from __future__ import annotations
import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from scripts import run_tinystories_matformer_widths as ops

CPU_SUITES = (
    'test_matformer_widths_campaign.py', 'test_matformer_widths_reporting.py', 'test_matformer_widths_queue.py',
    'test_optimizer_ownership.py', 'test_optimizer_ownership_resume.py', 'test_optimizer_ownership_campaign.py',
    'test_optimizer_ownership_reporting.py', 'test_optimizer_ownership_corrections.py',
    'test_inverse_membership_sampling.py', 'test_inverse_membership_reporting.py', 'test_inverse_membership_queue.py',
    'test_config.py', 'test_per_granularity_optimizer.py', 'test_per_granularity_optimizer_resume.py',
    'test_metrics_compact_accounting.py', 'test_metrics_history_performance.py', 'test_global_sampling_windows.py',
    'test_reproducibility.py', 'test_model_size.py', 'test_packed_corpus.py', 'test_train_cli.py', 'test_training_smoke.py')


def snapshot(root):
    root = Path(root)
    source = root/'source'
    manifest = ops.campaign._read_preflight_manifest(root/'campaign/campaign_manifest.json')
    for run in manifest['runs']:
        for relative, expected in run['initialization']['source_files_sha256'].items():
            if ops.digest(REPO/relative) != expected:
                raise ops.ConfigError(f'Preflight source changed: {relative}')
    if source.exists():
        saved = ops.read(root/'diagnostics/source-manifest.json')
        ops.check_seal(saved,'source snapshot')
        if ops.source_files(source) != saved['files']:
            raise ops.ConfigError('Existing immutable snapshot changed')
        return source
    stage = root/f'.source-{uuid.uuid4().hex}'
    stage.mkdir()
    try:
        for name in ('src','scripts','configs','tests'):
            shutil.copytree(REPO/name,stage/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
        for name in ('train.py','pyproject.toml','requirements.txt','build_backend.py'):
            if (REPO/name).exists(): shutil.copy2(REPO/name,stage/name)
        ops.save(stage/'source-provenance.json', ops.campaign._provenance())
        files = ops.source_files(stage)
        for path in stage.rglob('*'):
            path.chmod(0o555 if path.is_dir() else 0o444)
        stage.chmod(0o555)
        stage.rename(source)
        ops.save(root/'diagnostics/source-manifest.json',ops.sealed(dict(files=files,source_sha256=ops.stable_hash(files))))
    finally:
        if stage.exists(): shutil.rmtree(stage)
    return source


def pytest_check(source, output, *, gpu=False):
    # Fixtures resolve relative configs and create default outputs/. Keep their
    # working files outside the immutable source, importing the tested modules
    # through symlinks rather than copying or making the snapshot writable.
    workspace = output/'pytest-workspace'
    workspace.mkdir()
    for path in source.iterdir():
        (workspace/path.name).symlink_to(path, target_is_directory=path.is_dir())
    junit = output/'pytest.xml'
    cmd = [ops.PYTHON,'-m','pytest', '-q','-rs','--tb=short','-p','no:cacheprovider','--junitxml='+str(junit)]
    if gpu:
        cmd += ['tests/test_matformer_widths_campaign.py','-k','mw_resume_at_epoch_boundaries or mw_gpu_semantic or mw_partial_update or mw_corrupt_bundle or mw_physical_identity']
    else: cmd += ['tests/'+name for name in CPU_SUITES]
    env = dict(os.environ, OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
        MPLCONFIGDIR=str(output/'matplotlib'), PYTHONPATH=str(source),
        MATFORMER_DIAGNOSTIC_DEVICE='cuda' if gpu else 'cpu')
    if not gpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    started = time.time()
    with (output/'pytest.log').open('w') as log:
        if gpu:
            # The same sbatch process runs probes and pytest; no second CUDA
            # process competes with the diagnostic runtime's allocator/context.
            import pytest
            previous_device = os.environ.get('MATFORMER_DIAGNOSTIC_DEVICE')
            previous_cwd = Path.cwd()
            os.environ['MATFORMER_DIAGNOSTIC_DEVICE'] = 'cuda'
            try:
                os.chdir(workspace)
                with redirect_stdout(log), redirect_stderr(log):
                    returncode = int(pytest.main(cmd[3:]))
            finally:
                os.chdir(previous_cwd)
                if previous_device is None: os.environ.pop('MATFORMER_DIAGNOSTIC_DEVICE', None)
                else: os.environ['MATFORMER_DIAGNOSTIC_DEVICE'] = previous_device
        else:
            returncode = subprocess.run(cmd,cwd=workspace,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    record = dict(command=cmd,cwd=str(workspace),source=str(source),execution='in_process_pytest' if gpu else 'subprocess',returncode=returncode, started_at=started,finished_at=time.time())
    suites = ET.parse(junit).getroot().iter('testsuite')
    counts = dict(tests=0,failures=0,errors=0,skipped=0)
    for suite in suites:
        for key in counts: counts[key] += int(suite.get(key,0))
    record.update(counts)
    record['artifacts'] = [dict(path=str(p),sha256=ops.digest(p)) for p in (junit,output/'pytest.log')]
    return record


@contextmanager
def diagnostic_identity(definition, output, run_id):
    """Allow only this probe's run/path substitution; validate all real controls.

    Production validators stay strict. This single-process diagnostic context
    validates a canonical copy before the shared trainer sees the distinct ID.
    """
    original = ops.campaign.validate_materialized_config
    def validate(config):
        if config['run']['run_id'] != run_id or Path(config['run']['output_dir']) != output:
            return original(config)
        normalized = copy.deepcopy(config)
        normalized['run']['run_id'] = definition['run_id']
        normalized['optimizer_ownership_contract']['run_id'] = definition['run_id']
        normalized['optimizer_ownership_contract_hash'] = ops.stable_hash(normalized['optimizer_ownership_contract'])
        if normalized['optimizer_ownership_contract_hash'] != definition['contract_hash']:
            raise ops.ConfigError('Diagnostic changed more than run identity')
        original(normalized)
    ops.campaign.validate_materialized_config = validate
    try:
        yield
    finally:
        ops.campaign.validate_materialized_config = original


def gpu_real_shapes(root, output):
    """Eight real-corpus updates + own resume; full production horizons retained."""
    import torch
    import yaml
    from src.training import run, steps, checkpointing as cp
    from src.utils.config import resolve_run_config
    from src.utils.reproducibility import deterministic_runtime_settings
    if not torch.cuda.is_available():
        raise ops.ConfigError('Actual bf16 CUDA required; a skip cannot pass')
    original_train, original_determinism = steps.train_for_steps, run.configure_strict_determinism
    established = None
    def configure(config):
        nonlocal established
        if not torch.cuda.is_initialized(): established = original_determinism(config)
        else:
            if established is None or deterministic_runtime_settings()!=established:
                raise ops.ConfigError('Diagnostic deterministic runtime changed')
        return established
    # Configure strict CUDA before the first actual device operation.
    first = resolve_run_config(root/'campaign/configs'/f'{ops.ARMS[0]}.yaml')
    established = original_determinism(first)
    if not torch.cuda.is_bf16_supported():
        raise ops.ConfigError('Actual bf16 CUDA required; a skip cannot pass')
    run.configure_strict_determinism = configure
    measurements=[]
    class Boundary(KeyboardInterrupt): pass
    try:
        manifest=ops.read(root/'campaign/campaign_manifest.json')
        for definition in manifest['runs']:
            arm=definition['arm_id']; raw=copy.deepcopy(definition['executable_config'])
            identity=f"mw-gpu-diagnostic-{os.environ['SLURM_JOB_ID']}"
            probe_root=output/'runs'/f'{identity}-{arm}'
            raw['run'].update(run_id=f'{identity}-{arm}',output_dir=str(probe_root))
            raw['optimizer_ownership_contract'].update(run_id=raw['run']['run_id'])
            raw['optimizer_ownership_contract_hash']=ops.stable_hash(raw['optimizer_ownership_contract'])
            config_file=output/f'{arm}.yaml'; config_file.write_text(yaml.safe_dump(raw,sort_keys=False))
            for boundary in (4,8):
                def window(config,model,loader,evaluation,optimizer,scheduler,device,**kwargs):
                    state=kwargs['run_state']
                    assert state['last_completed_step']==boundary-4
                    assert config['training']['max_steps']==definition['assigned_updates']
                    assert config['training']['batch_size_per_process']==64 and config['model']['context_length']==128
                    assert config['training']['mixed_precision']=='bf16' and device.type=='cuda'
                    callback=kwargs.get('successful_step_callback')
                    def committed(*,step,tokens_seen):
                        if callback: callback(step=step,tokens_seen=tokens_seen)
                        if step==boundary:
                            assert all(torch.isfinite(p).all() for p in model.parameters())
                            cp.save_model_checkpoint(config,model,optimizer,scheduler,Path(config['run']['output_dir'])/'checkpoints/latest.pt',
                                {'checkpoint_status':'latest','checkpoint_metric':None,'checkpoint_metric_value':None,'checkpoint_selection_step':None},state)
                            raise Boundary('durable real-shape diagnostic boundary')
                    kwargs['successful_step_callback']=committed
                    return original_train(config,model,loader,evaluation,optimizer,scheduler,device,**kwargs)
                steps.train_for_steps=window
                with diagnostic_identity(definition, probe_root, raw['run']['run_id']):
                    try: run.run_training(resolve_run_config(config_file))
                    except Boundary: pass
                checkpoint=probe_root/'checkpoints/latest.pt'
                saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
                assert saved['step']==boundary
            measurements.append(dict(arm_id=arm,status='passed',steps=8,horizon=definition['assigned_updates'],
                config_path=str(config_file),config_sha256=ops.digest(config_file),checkpoint_path=str(checkpoint),checkpoint_sha256=ops.digest(checkpoint),
                artifacts=[dict(path=str(p),sha256=ops.digest(p)) for p in [config_file,checkpoint,*probe_root.glob('*.jsonl'),probe_root/'resource_attempts.json']],
                overrides={'run_id':raw['run']['run_id'],'output_dir':raw['run']['output_dir'],'stops':[4,8]}))
            ops.save(output/'real-shapes.json',measurements)
    finally:
        steps.train_for_steps=original_train; run.configure_strict_determinism=original_determinism
    del saved, window
    import gc
    gc.collect(); torch.cuda.empty_cache()
    return measurements


def execute(root, mode):
    root=Path(root).resolve()
    with ops.lock(root/'diagnostics'/f'{mode}.lock'):
        if mode=='cpu': source=snapshot(root)
        else:
            if not os.environ.get('SLURM_JOB_ID'): raise ops.ConfigError('GPU diagnostics require sbatch')
            limits=ops.scheduler_limits()
            active=ops.queue_state()
            if (any(r.get('qos')!=ops.QOS for r in active) or len(active)>limits['max_submitted']
                    or sum(r.get('state')=='RUNNING' for r in active)>limits['max_running']):
                raise ops.ConfigError('Live user-wide diagnostic limits are not satisfied')
            allocation=ops.command(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'-o'])
            source=root/'source'
            if REPO != source: raise ops.ConfigError('GPU diagnostics must execute the tested snapshot')
            os.chdir(source)
        expected=ops.bindings(root)
        cpu=ops.verify_gate(root,'cpu',expected) if mode=='gpu' else None
        output=root/'diagnostics'/f'{mode}-{uuid.uuid4().hex}'; output.mkdir()
        gate=dict(schema_version=1,mode=mode,status='failed',bindings=expected,checks=[],holdout_evaluated=False)
        import platform
        import importlib.metadata
        gate['environment']=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
            packages={name:importlib.metadata.version(name) for name in ('torch','transformers','numpy','pytest','PyYAML')})
        try:
            if mode=='gpu':
                measurements=gpu_real_shapes(root,output)
                import torch
                gate.update(job_id=os.environ['SLURM_JOB_ID'],hardware=torch.cuda.get_device_name(),allocation=allocation,
                    cpu_gate_hash=cpu['content_hash'],real_shape_arms=[r['arm_id'] for r in measurements],probes=measurements)
                gate['checks'].append(dict(command=[sys.executable,__file__,'--mode','gpu','--campaign-root',str(root)],
                    returncode=0,tests=9,failures=0,errors=0,skipped=0,
                    artifacts=[dict(path=str(output/'real-shapes.json'),sha256=ops.digest(output/'real-shapes.json')), *[a for probe in measurements for a in probe['artifacts']]]))
            check=pytest_check(source,output,gpu=mode=='gpu'); gate['checks'].append(check)
            if check['returncode'] or check['tests']==0 or check['failures'] or check['errors'] or (mode=='gpu' and check['skipped']):
                raise ops.ConfigError(f'{mode} acceptance failed: {output}/pytest.log')
            if ops.bindings(root)!=expected: raise ops.ConfigError('Diagnostic inputs changed during execution')
            gate['status']='passed'
        except BaseException as error:
            gate['error']=str(error)
            raise
        finally:
            gate=ops.sealed(gate)
            ops.save(output/f'{mode}-gate.json',gate)
            ops.save(root/'diagnostics'/f'{mode}-gate.json',gate)
        return gate


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('cpu','gpu'),required=True)
    parser.add_argument('--campaign-root',type=Path,required=True)
    args=parser.parse_args(argv)
    execute(args.campaign_root,args.mode)


if __name__=='__main__': main()
