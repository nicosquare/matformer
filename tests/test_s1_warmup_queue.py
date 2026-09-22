"""Fixture-only readiness/admission checks. Never contacts Slurm or submits jobs."""
import copy
import os
from pathlib import Path
import subprocess

import pytest
from scripts import run_tinystories_s1_warmup as ops
from scripts import preflight_tinystories_s1_warmup as diagnostics
from src.utils.config import ConfigError


@pytest.fixture
def queue_setup(tmp_path, monkeypatch):
    for name in ('launchers', 'diagnostics', 'logs', 'runs'): (tmp_path/name).mkdir()
    plan = dict(campaign_id=ops.campaign.WARMUP_CAMPAIGN_ID, campaign_root=str(tmp_path),
        arms=list(ops.ARMS), bindings={'fixture': 'source'}, cpu_gate_hash='cpu',
        assigned_updates=dict.fromkeys(ops.ARMS, 348528))
    ops.save(tmp_path/'diagnostics/gpu-gate.json', {'content_hash': 'gpu'})
    monkeypatch.setattr(ops, 'verify_plan', lambda *a, **k: plan)
    monkeypatch.setattr(ops, 'scheduler_limits', lambda: dict(max_running=2, max_submitted=4))
    monkeypatch.setattr(ops, 'continuation', lambda *a: dict(mode='fresh', step=0, checkpoint=None))
    monkeypatch.setattr(ops, 'scheduler_history', lambda *a: None)
    active, submitted = [], []
    monkeypatch.setattr(ops, 'queue_state', lambda: copy.deepcopy(active))
    def command(cmd):
        assert cmd[0] == 'sbatch'
        assert {'--exclude=gpu-[05,50,51,54]', '--gres=gpu:1', '--ntasks=1', '--no-requeue'} <= set(cmd)
        submitted.append(cmd)
        job = str(100+len(submitted)); name = next(a.split('=',1)[1] for a in cmd if a.startswith('--job-name='))
        active.append(dict(job_id=job, name=name, state='PENDING', qos=ops.QOS))
        return job
    monkeypatch.setattr(ops, 'command', command)
    return tmp_path, plan, active, submitted


def test_exact_two_arms_restart_no_duplicates(queue_setup):
    root, plan, active, submitted = queue_setup
    for _ in range(2): ops.queue(root, once=True)
    jobs = ops.read(root/'launchers/submissions.json')['jobs']
    assert [j['arm_id'] for j in jobs] == list(ops.ARMS)
    assert len(submitted) == 2 and all(j['attempt_id']==1 and j['intent_uuid'] for j in jobs)
    assert all(str(root/'source/scripts/run_tinystories_s1_warmup.py') in j['command'][-1] for j in jobs)


@pytest.mark.parametrize('count,running,submitted,expected', [(0,2,4,2), (1,2,4,1), (2,2,4,0), (0,1,4,1), (0,2,1,1)])
def test_user_wide_pending_reservations_include_foreign_jobs(queue_setup, monkeypatch, count, running, submitted, expected):
    root, plan, active, calls = queue_setup
    active.extend(dict(job_id=str(n), name='unrelated', state='PENDING', qos='other') for n in range(count))
    monkeypatch.setattr(ops, 'scheduler_limits', lambda: dict(max_running=running, max_submitted=submitted))
    ops.queue(root, once=True)
    assert len(calls) == expected


def test_delayed_queue_visibility_counts_intents(queue_setup, monkeypatch):
    root, _, _, submitted = queue_setup
    monkeypatch.setattr(ops, 'queue_state', lambda: [])
    ops.queue(root, once=True)
    assert len(submitted) == 2
    status = ops.queue(root, once=True)
    assert status['production'] == 'pending' and len(submitted)==2
    assert all(j['status']=='uncertain' for j in status['jobs'])


def test_uncertain_submission_blocks_duplicate(queue_setup, monkeypatch):
    root, _, _, submitted = queue_setup
    def timeout(cmd): submitted.append(cmd); raise subprocess.TimeoutExpired(cmd, 45)
    monkeypatch.setattr(ops, 'command', timeout)
    with pytest.raises(subprocess.TimeoutExpired): ops.queue(root, once=True)
    ops.queue(root, once=True)
    assert len(submitted)==1


def test_accounting_delay_cannot_be_replaced_by_worker_success(queue_setup, monkeypatch):
    root, _, active, submitted = queue_setup
    ops.queue(root, once=True); active.clear()
    jobs = ops.read(root/'launchers/submissions.json')['jobs']
    for j in jobs:
        ops.save(root/'launchers'/f"worker-{j['arm_id']}-1.json", dict(job_id=j['job_id'], status='completed', returncode=0))
        ops.save(root/'runs'/j['arm_id']/'terminal_validation_results.json', {})
    status = ops.queue(root, once=True)
    assert status['production']=='pending' and len(submitted)==2
    assert all(j['status']=='uncertain' for j in status['jobs'])


def test_failed_attempt_own_checkpoint_retries_monotonically(queue_setup, monkeypatch):
    root, _, active, submitted = queue_setup
    ops.queue(root, once=True); active.pop(0)
    monkeypatch.setattr(ops, 'scheduler_history', lambda root,j: dict(job_id=j['job_id'], name=j['name'], state='FAILED', exit_code='1:0', allocation_seconds=12))
    monkeypatch.setattr(ops, 'continuation', lambda *a: dict(mode='resume', step=3, checkpoint={'sha256':'own'}))
    ops.queue(root, once=True)
    jobs = ops.read(root/'launchers/submissions.json')['jobs']
    assert len(submitted)==3 and jobs[-1]['attempt_id']==2
    assert jobs[-1]['arm_id']==ops.ARMS[0] and jobs[0]['scheduler_accounting']['allocation_seconds']==12


def test_terminal_recovery_submits_completion_only(queue_setup, monkeypatch):
    root, _, active, submitted = queue_setup
    ops.queue(root, once=True); active.pop(0)
    monkeypatch.setattr(ops, 'scheduler_history', lambda root,j: dict(job_id=j['job_id'], name=j['name'], state='FAILED', exit_code='1:0', allocation_seconds=12))
    monkeypatch.setattr(ops, 'continuation', lambda *a: dict(mode='completion_only', step=348528, checkpoint={'sha256':'own'}))
    ops.queue(root, once=True)
    assert ops.read(root/'launchers/submissions.json')['jobs'][-1]['continuation']['mode']=='completion_only'


def test_gate_failure_prevents_any_submission(queue_setup, monkeypatch):
    root, _, _, submitted = queue_setup
    def reject(*a, **k): raise ConfigError('stale gate')
    monkeypatch.setattr(ops, 'verify_plan', reject)
    with pytest.raises(ConfigError, match='stale'): ops.queue(root, once=True)
    assert not submitted


def test_lock_and_occupied_identity(queue_setup, monkeypatch):
    root, _, _, submitted = queue_setup
    with ops.lock(root/'launchers/queue.lock'):
        with pytest.raises(ConfigError, match='locked'): ops.queue(root, once=True)
    def occupied(*a): raise ConfigError('Occupied run has no durable own checkpoint')
    monkeypatch.setattr(ops, 'continuation', occupied)
    with pytest.raises(ConfigError, match='Occupied'): ops.queue(root, once=True)
    assert not submitted


@pytest.mark.parametrize('mode,damage', [('cpu','status'),('cpu','bindings'),('cpu','checks'),('gpu','skipped'),('gpu','cpu'),('gpu','precision'),('gpu','arms')])
def test_gate_rejects_failed_stale_skipped_incomplete(tmp_path, monkeypatch, mode, damage):
    binding={'source':'frozen'}; monkeypatch.setattr(ops,'bindings',lambda root:binding)
    log=tmp_path/'executed.log';log.write_text('fixture command passed')
    check=dict(command=['fixture'],returncode=0, tests=1, failures=0, errors=0, skipped=0,
        artifacts=[ops.campaign._source_record(log)])
    cpu=ops.sealed(dict(mode='cpu',status='passed',bindings=binding,checks=[check]))
    ops.save(tmp_path/'diagnostics/cpu-gate.json',cpu)
    gate=dict(mode=mode,status='passed',bindings=binding,checks=[copy.deepcopy(check)],cpu_gate_hash=cpu['content_hash'],
        job_id='12', hardware='GPU', precision='bf16', real_shape_arms=list(ops.ARMS),
        real_shape=dict(d_model=64,num_layers=4,num_attention_heads=4,batch_size=64,context_length=128),
        coverage=list(diagnostics.GPU_COVERAGE))
    if damage=='status': gate['status']='failed'
    elif damage=='bindings': gate['bindings']={}
    elif damage=='checks': gate['checks']=[]
    elif damage=='skipped': gate['checks'][0]['skipped']=1
    elif damage=='cpu': gate['cpu_gate_hash']='old'
    elif damage=='precision': gate['precision']='fp32'
    else: gate['real_shape_arms']=['S1']
    ops.save(tmp_path/'diagnostics'/f'{mode}-gate.json',ops.sealed(gate))
    with pytest.raises(ConfigError): ops.verify_gate(tmp_path,mode)


def test_scheduler_history_never_uses_worker_as_accounting(tmp_path, monkeypatch):
    intent=dict(arm_id=ops.ARMS[0],attempt_id=1,name='job',created_date='2026-09-22',job_id='12')
    ops.save(tmp_path/'launchers'/f'worker-{ops.ARMS[0]}-1.json', dict(job_id='12',status='completed',returncode=0))
    monkeypatch.setattr(ops,'command',lambda args:'')
    assert ops.scheduler_history(tmp_path,intent) is None


def test_worker_requires_sbatch_and_declared_arm(tmp_path,monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(ConfigError,match='sbatch'): ops.worker(tmp_path,ops.ARMS[0],1)
    monkeypatch.setenv('SLURM_JOB_ID','12')
    with pytest.raises(ConfigError,match='arm'): ops.worker(tmp_path,'S1',1)


def test_cuda_entry_rejects_absence_and_wrong_precision(monkeypatch):
    import torch
    from scripts.train_cuda_required import required_training
    monkeypatch.setenv('SLURM_JOB_ID','12')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(ConfigError,match='CUDA'): required_training({'training':{'mixed_precision':'bf16'}})
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    with pytest.raises(ConfigError,match='BF16'): required_training({'training':{'mixed_precision':'none'}})


def test_resource_attempts_retain_replayed_costs(tmp_path):
    from src.training.run import ResourceAttemptLedger
    ledger=ResourceAttemptLedger(tmp_path,run_id='run')
    for name,steps,status,seconds in [('failed',4,'failed',5.),('replay',3,'completed',7.),('killed',None,'running',None)]:
        ledger.observe(name,sequence=1,elapsed_seconds=seconds,attempted_steps=steps,status=status,
            peak_allocated_bytes=None,peak_reserved_bytes=None,source_checkpoint=None)
    summary=ledger.summary()
    assert summary['attempted_steps']==7 and summary['elapsed_seconds']==12
    assert not summary['measurement_complete'] and summary['unobserved_attempts']==['killed']


def test_cpu_snapshot_test_command_uses_frozen_sources(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();(source/'tests').mkdir()
    output=tmp_path/'checks';output.mkdir()
    calls=[]
    def execute(cmd,**kwargs):
        calls.append((cmd,kwargs))
        xml=next(x.split('=',1)[1] for x in cmd if x.startswith('--junitxml='))
        Path(xml).write_text('<testsuites><testsuite tests="3" failures="0" errors="0" skipped="0"/></testsuites>')
        return subprocess.CompletedProcess(cmd,0)
    monkeypatch.setattr(diagnostics.subprocess,'run',execute)
    result=diagnostics.pytest_check(source,output,gpu=False)
    assert result['tests']==3 and result['returncode']==0
    assert calls[0][1]['env']['PYTHONPATH']==str(source)
    assert calls[0][1]['env']['CUDA_VISIBLE_DEVICES']==''
    assert (Path(calls[0][1]['cwd'])/'tests').resolve()==source/'tests'


@pytest.fixture
def bound_root(tmp_path, monkeypatch):
    """Real bytes/hashes/reservation; stub only scientific preflight parsing."""
    import yaml
    source = tmp_path/'source'; source.mkdir()
    for name in ('run_tinystories_s1_warmup.py','preflight_tinystories_s1_warmup.py','train_cuda_required.py'):
        path=source/'scripts'/name;path.parent.mkdir(exist_ok=True);path.write_text('# frozen '+name)
    recipe=tmp_path/'recipe.yaml';recipe.write_text('fixed recipe')
    reference=tmp_path/'reference.json';reference.write_text('immutable reference')
    runs=[];artifacts={}
    for arm in ops.ARMS:
        raw={'run':{'arm_id':arm}}
        path=tmp_path/'campaign/configs'/f'{arm}.yaml';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(yaml.safe_dump(raw))
        artifacts[f'configs/{arm}.yaml']=ops.digest(path)
        runs.append(dict(arm_id=arm,run_id='campaign-'+arm,executable_config=raw,contract_hash='contract-'+arm,
                         initialization={'source_files_sha256':{}},assigned_updates=348528))
    manifest=dict(schema_version=5,campaign_id=ops.campaign.WARMUP_CAMPAIGN_ID,runs=runs,
        run_output_root=str(tmp_path/'runs'),manifest_hash='manifest',artifact_sha256=artifacts,
        recipe_source=ops.campaign._source_record(recipe),control_audits={a:{'status':'passed'} for a in ops.ARMS},
        reference_selection=dict(status='passed',sources=[ops.campaign._source_record(reference)],
            grids={'linear':{'root':'/reference-linear'},'geometric':{'root':'/reference-geometric'}}))
    ops.save(tmp_path/'campaign/campaign_manifest.json',manifest)
    monkeypatch.setattr(ops.campaign,'_read_preflight_manifest',lambda p:ops.read(p))
    ops.save(tmp_path/'campaign/preflight.json',dict(status='passed',manifest_hash='manifest'))
    files=ops.source_files(source)
    ops.save(tmp_path/'diagnostics/source-manifest.json',ops.sealed(dict(files=files,source_sha256=ops.stable_hash(files))))
    ops.save(ops.campaign._reservation_path(tmp_path/'runs'),dict(campaign_id=manifest['campaign_id'],
        manifest_path=str(tmp_path/'campaign/campaign_manifest.json'),manifest_hash='manifest',run_ids=[r['run_id'] for r in runs]))
    log=tmp_path/'diagnostics/check.log';log.write_text('fixture executed')
    check=dict(command=['fixture'],returncode=0,tests=1,failures=0,errors=0,skipped=0,artifacts=[ops.campaign._source_record(log)])
    ops.save(tmp_path/'diagnostics/cpu-gate.json',ops.sealed(dict(mode='cpu',status='passed',bindings=ops.bindings(tmp_path),checks=[check])))
    return tmp_path


def test_prepare_is_exact_and_idempotent(bound_root):
    root=bound_root
    first=ops.prepare(root,root/'diagnostics/cpu-gate.json')
    assert first['arms']==list(ops.ARMS) and first['assigned_updates']==dict.fromkeys(ops.ARMS,348528)
    assert ops.prepare(root,root/'diagnostics/cpu-gate.json')==first
    assert not (root/'launchers/submissions.json').exists()


@pytest.mark.parametrize('damage',['source','config','reference','recipe','log','reservation','occupied'])
def test_prepare_rejects_changed_bindings(bound_root,damage):
    root=bound_root
    if damage=='source': path=root/'source/scripts/train_cuda_required.py'
    elif damage=='config': path=root/'campaign/configs'/f'{ops.ARMS[0]}.yaml'
    elif damage=='reference':path=root/'reference.json'
    elif damage=='recipe':path=root/'recipe.yaml'
    elif damage=='log':path=root/'diagnostics/check.log'
    elif damage=='reservation':
        ops.save(ops.campaign._reservation_path(root/'runs'),{});path=None
    else:
        (root/'runs'/ops.ARMS[0]).mkdir(parents=True);path=None
    if path:path.write_text('changed')
    with pytest.raises(ConfigError):ops.prepare(root,root/'diagnostics/cpu-gate.json')
    assert not (root/'launchers/plan.json').exists()


def test_snapshot_is_immutable_and_refuses_tampering(tmp_path,monkeypatch):
    repo=tmp_path/'repo';repo.mkdir()
    for name in ('src','scripts','tests','configs'):(repo/name).mkdir()
    (repo/'src/core.py').write_text('value = 1\n')
    root=tmp_path/'campaign-root';root.mkdir()
    manifest={'runs':[{'initialization':{'source_files_sha256':{'src/core.py':ops.digest(repo/'src/core.py')}}}]}
    monkeypatch.setattr(diagnostics,'REPO',repo)
    monkeypatch.setattr(ops.campaign,'_read_preflight_manifest',lambda p:manifest)
    monkeypatch.setattr(ops.campaign,'_provenance',lambda:{'fixture':True})
    source=diagnostics.snapshot(root)
    try:
        assert not (source/'src/core.py').stat().st_mode & 0o222
        assert diagnostics.snapshot(root)==source
        (source/'src/core.py').chmod(0o644);(source/'src/core.py').write_text('tampered')
        with pytest.raises(ConfigError,match='changed'):diagnostics.snapshot(root)
    finally:
        source.chmod(0o755)
        for path in source.rglob('*'):path.chmod(0o755 if path.is_dir() else 0o644)


def install_worker_intent(setup,monkeypatch):
    root,plan,active,submitted=setup
    ops.queue(root,once=True)
    intent=ops.read(root/'launchers/submissions.json')['jobs'][0]
    monkeypatch.setenv('SLURM_JOB_ID',intent['job_id']);monkeypatch.setenv('SLURM_JOB_NAME',intent['name'])
    ops.save(root/'campaign/campaign_manifest.json',{'runs':[{'arm_id':a,'run_id':'run-'+a} for a in ops.ARMS]})
    return intent


@pytest.mark.parametrize('damage',['job','name','checkpoint','gate','duplicate'])
def test_worker_revalidates_before_trainer(queue_setup,monkeypatch,damage):
    root,plan,_,_=queue_setup;intent=install_worker_intent(queue_setup,monkeypatch)
    if damage=='job':monkeypatch.setenv('SLURM_JOB_ID','999')
    elif damage=='name':monkeypatch.setenv('SLURM_JOB_NAME','other')
    elif damage=='checkpoint':monkeypatch.setattr(ops,'continuation',lambda *a:dict(mode='resume',step=1,checkpoint={}))
    elif damage=='gate':ops.save(root/'diagnostics/gpu-gate.json',{'content_hash':'changed'})
    else:ops.save(root/'launchers'/f"worker-{ops.ARMS[0]}-1.json",{})
    monkeypatch.setattr(ops.subprocess,'run',lambda *a,**k:pytest.fail('invalid worker must not execute trainer'))
    with pytest.raises(ConfigError):ops.worker(root,ops.ARMS[0],1)


def test_worker_invokes_frozen_cuda_and_discloses_missing_measurements(queue_setup,monkeypatch):
    root,plan,_,_=queue_setup;install_worker_intent(queue_setup,monkeypatch)
    def execute(cmd,**kwargs):
        assert cmd[1]==str(root/'source/scripts/train_cuda_required.py')
        assert kwargs['env']['MATFORMER_LAUNCH_ATTEMPT_ID']=='1'
        assert kwargs['cwd']==root/'source'
        return subprocess.CompletedProcess(cmd,1)
    monkeypatch.setattr(ops.subprocess,'run',execute)
    assert ops.worker(root,ops.ARMS[0],1)==1
    result=ops.read(root/'launchers'/f'worker-{ops.ARMS[0]}-1.json')
    assert result['status']=='failed' and not result['measurement_complete']
    assert result['resources']['unobserved_attempts']==[result['process_uuid']]
    with pytest.raises(ConfigError):ops.worker(root,ops.ARMS[0],1)


def test_successful_accounting_requires_matching_execution_evidence(queue_setup,monkeypatch):
    root,plan,active,submitted=queue_setup
    ops.queue(root,once=True);active.clear()
    monkeypatch.setattr(ops,'scheduler_history',lambda root,j:dict(job_id=j['job_id'],state='COMPLETED',exit_code='0:0',allocation_seconds=10))
    for arm in ops.ARMS:
        ops.save(root/'runs'/arm/'terminal_validation_results.json',{})
        ops.save(root/'runs'/arm/'run_summary.json',{})
        ops.save(root/'launchers'/f'worker-{arm}-1.json',{})
    def reject(*a):raise ConfigError('missing matching GPU worker')
    monkeypatch.setattr(ops,'execution_evidence',reject)
    with pytest.raises(ConfigError,match='matching GPU'):ops.queue(root,once=True)
    assert len(submitted)==2
    monkeypatch.setattr(ops,'execution_evidence',lambda *a:dict(validated=True))
    status=ops.queue(root,once=True)
    assert status['production']=='complete' and status['resources']['scheduler_allocation_seconds']==20


@pytest.mark.parametrize('damage',[None,'worker','job','precision','device','entry','checkpoint'])
def test_execution_evidence_binds_worker_device_and_terminal(queue_setup,monkeypatch,damage):
    from src.training.run import ResourceAttemptLedger
    root,plan,_,_=queue_setup;intent=install_worker_intent(queue_setup,monkeypatch);arm=intent['arm_id']
    intent['scheduler_accounting']=dict(state='COMPLETED',exit_code='0:0')
    worker=dict(arm_id=arm,attempt_id=1,job_id=intent['job_id'],name=intent['name'],intent_uuid=intent['intent_uuid'],
        bindings=plan['bindings'],cpu_gate_hash='cpu',gpu_gate_hash='gpu',process_uuid='process',returncode=0,status='completed')
    entry=dict(job_id=intent['job_id'],requested_device='cuda:0',required_precision='bf16',status='starting',
        config=str(root/'campaign/configs'/f'{arm}.yaml'),nvidia_smi={'returncode':0})
    config=dict(run={'run_id':'run-'+arm},training={'resolved_mixed_precision':'bf16'})
    measured=dict(sequence=1,run_id='run-'+arm,launch_attempt_id=1,slurm_job_id=intent['job_id'],process_uuid='process',
        status='completed',elapsed_seconds=1.,attempted_steps=348528,peak_allocated_bytes=100,peak_reserved_bytes=200,source_checkpoint=None)
    if damage=='worker':worker['returncode']=1
    elif damage=='job':worker['job_id']='other'
    elif damage=='precision':config['training']['resolved_mixed_precision']='fp32'
    elif damage=='device':measured['peak_allocated_bytes']=None
    elif damage=='entry':entry['status']='failed_or_interrupted'
    ops.save(root/'launchers'/f'worker-{arm}-1.json',worker)
    ops.save(root/'launchers'/f'cuda-entry-{arm}-1.json',entry)
    ops.save(root/'runs'/arm/'config.json',config)
    ResourceAttemptLedger(root/'runs'/arm,run_id='run-'+arm).observe('process',**measured)
    monkeypatch.setattr(ops.campaign,'inspect_selected_terminals',lambda *a,**k:[{'checkpoint_sha256':'terminal'}])
    monkeypatch.setattr(ops,'continuation',lambda *a:dict(mode='completion_only',checkpoint={'sha256':'other' if damage=='checkpoint' else 'terminal'}))
    if damage:
        with pytest.raises(ConfigError):ops.execution_evidence(root,intent,plan)
    else:
        result=ops.execution_evidence(root,intent,plan)
        assert result['resources']['measurement_complete'] and len(result['sources'])==4


def test_diagnostic_submission_uses_shared_limits_and_durable_identity(queue_setup,monkeypatch):
    root,plan,active,submitted=queue_setup
    monkeypatch.setattr(ops,'bindings',lambda root:plan['bindings'])
    monkeypatch.setattr(ops,'verify_gate',lambda *a,**k:{'content_hash':'cpu'})
    active.append(dict(job_id='foreign',name='foreign',state='PENDING',qos='other'))
    first=diagnostics.submit_gpu(root)
    assert first['status']=='submitted' and len(submitted)==1
    assert 'preflight_tinystories_s1_warmup.py' in submitted[0][-1]
    assert diagnostics.submit_gpu(root)['status']=='pending' and len(submitted)==1
    active.clear()
    assert diagnostics.submit_gpu(root)['status']=='pending' and len(submitted)==1


def test_diagnostic_submission_capacity_waits(queue_setup,monkeypatch):
    root,plan,active,submitted=queue_setup
    monkeypatch.setattr(ops,'bindings',lambda root:plan['bindings'])
    monkeypatch.setattr(ops,'verify_gate',lambda *a,**k:{'content_hash':'cpu'})
    active.extend(dict(job_id=str(i),name='foreign',state='PENDING') for i in range(2))
    assert diagnostics.submit_gpu(root)['status']=='pending_capacity' and not submitted


@pytest.mark.parametrize('passed',[True,False])
def test_cpu_gate_publishes_executed_result_and_never_claims_readiness(bound_root,monkeypatch,passed):
    root=bound_root;monkeypatch.setattr(diagnostics,'REPO',root/'source')
    def check(source,output,**kwargs):
        log=output/'pytest.log';log.write_text('passed' if passed else 'failed')
        return dict(command=['fixture'],returncode=0 if passed else 1,tests=2,failures=0 if passed else 1,errors=0,skipped=0,
            artifacts=[ops.campaign._source_record(log)])
    monkeypatch.setattr(diagnostics,'pytest_check',check)
    if passed:
        result=diagnostics.execute(root,'cpu')
        assert result['status']=='passed' and result['reporting_fixture_status']=='pending'
        assert ops.verify_gate(root,'cpu')==result
    else:
        with pytest.raises(ConfigError,match='acceptance failed'):diagnostics.execute(root,'cpu')
        result=ops.read(root/'diagnostics/cpu-gate.json')
        assert result['status']=='failed' and result['checks'][0]['failures']==1
    assert not (root/'diagnostics/gpu-gate.json').exists()
    assert not (root/'launchers/submissions.json').exists()


def test_gpu_gate_waits_for_actual_diagnostic_accounting(bound_root,monkeypatch):
    root=bound_root;cpu=ops.read(root/'diagnostics/cpu-gate.json')
    gate=ops.sealed(dict(mode='gpu',status='passed',bindings=ops.bindings(root),checks=cpu['checks'],
        cpu_gate_hash=cpu['content_hash'],job_id='12',intent_uuid='diagnostic',hardware='GPU',precision='bf16',
        real_shape_arms=list(ops.ARMS),real_shape=ops.REAL_SHAPE,coverage=list(ops.GPU_COVERAGE)))
    ops.save(root/'diagnostics/gpu-gate.json',gate)
    ops.save(root/'diagnostics/submissions.json',{'jobs':[dict(intent_uuid='diagnostic',job_id='12')]})
    ops.save(root/'diagnostics/worker-diagnostic.json',dict(job_id='12',returncode=0,gate_hash=gate['content_hash']))
    monkeypatch.setattr(ops,'scheduler_history',lambda *a:None)
    with pytest.raises(ConfigError,match='accounting'):ops.verify_gate(root,'gpu')
    monkeypatch.setattr(ops,'scheduler_history',lambda *a:dict(job_id='12',state='COMPLETED',exit_code='0:0'))
    assert ops.verify_gate(root,'gpu')==gate


def test_restart_recovers_missing_outputs_after_recorded_completion(queue_setup,monkeypatch):
    root,plan,active,submitted=queue_setup
    ops.queue(root,once=True);active.clear()
    record=ops.read(root/'launchers/submissions.json')
    for job in record['jobs']:job['status']='completed'
    ops.save(root/'launchers/submissions.json',record)
    # Full checkpoints are inspected by continuation; neither old status nor
    # a missing sidecar grants permission for a fresh restart.
    monkeypatch.setattr(ops,'continuation',lambda *a:dict(mode='completion_only',step=348528,checkpoint={'sha256':'own'}))
    status=ops.queue(root,once=True)
    assert status['production']=='pending' and len(submitted)==4
    retries=ops.read(root/'launchers/submissions.json')['jobs'][2:]
    assert all(j['attempt_id']==2 and j['continuation']['mode']=='completion_only' for j in retries)
