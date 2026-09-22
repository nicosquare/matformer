"""No scheduler calls: gate provenance, stage admission and restart fixtures."""
import copy
import json
import subprocess
from pathlib import Path

import pytest
from scripts import run_tinystories_matformer_widths as queue
from scripts import preflight_tinystories_matformer_widths as diagnostics
from src.utils.config import ConfigError


@pytest.fixture
def setup(tmp_path, monkeypatch):
    for folder in ('launchers','diagnostics','runs','logs'):
        (tmp_path/folder).mkdir()
    plan=dict(campaign_id=queue.campaign.MATFORMER_CAMPAIGN_ID,bindings={'fixture':'source'},cpu_gate_hash='cpu',
        assigned_updates={a:87132 if a in queue.STANDALONES else 348528 for a in queue.ARMS},created_date='2026-09-21')
    queue.save(tmp_path/'diagnostics/gpu-gate.json',{'content_hash':'gpu'})
    monkeypatch.setattr(queue,'verify_plan',lambda *args,**kwargs:plan)
    monkeypatch.setattr(queue,'scheduler_limits',lambda:dict(max_running=2,max_submitted=4))
    monkeypatch.setattr(queue,'continuation',lambda *args:dict(mode='fresh',step=0,checkpoint=None))
    monkeypatch.setattr(queue.campaign,'inspect_selected_terminals',lambda *args,**kwargs:[])
    active=[]; submissions=[]
    monkeypatch.setattr(queue,'queue_state',lambda:copy.deepcopy(active))
    monkeypatch.setattr(queue,'scheduler_history',lambda *args:None)
    def submit(cmd):
        assert cmd[0]=='sbatch' and '--exclude=gpu-[05,50,51,54]' in cmd and '--gres=gpu:1' in cmd
        submissions.append(cmd)
        name=next(v.split('=',1)[1] for v in cmd if v.startswith('--job-name='))
        job=str(100+len(submissions)); active.append(dict(job_id=job,name=name,state='PENDING',qos=queue.QOS))
        return job
    monkeypatch.setattr(queue,'command',submit)
    return tmp_path,plan,active,submissions


def test_standalones_only_and_no_restart_duplicates(setup):
    root,plan,active,submitted=setup
    queue.queue(root,once=True); queue.queue(root,once=True)
    jobs=queue.read(root/'launchers/submissions.json')['jobs']
    assert len(submitted)==4
    assert [j['arm_id'] for j in jobs]==list(queue.STANDALONES)
    assert all(j['attempt_id']==1 for j in jobs)


def test_barrier_failure_blocks_all_elastic_admission(setup,monkeypatch):
    root,plan,active,submitted=setup
    queue.queue(root,once=True); active.clear()
    jobs=queue.read(root/'launchers/submissions.json')['jobs']
    for job in jobs:
        path=root/'runs'/job['arm_id']/'terminal_validation_results.json';queue.save(path,{})
    monkeypatch.setattr(queue,'scheduler_history',lambda root,j:dict(job_id=j['job_id'],state='COMPLETED'))
    def reject(*args):raise ConfigError('tampered standalone')
    monkeypatch.setattr(queue,'standalone_barrier',reject)
    with pytest.raises(ConfigError,match='tampered'):queue.queue(root,once=True)
    assert len(submitted)==4


def test_elastic_admission_requires_barrier_each_time(setup,monkeypatch):
    root,plan,active,submitted=setup
    queue.queue(root,once=True); active.clear()
    jobs=queue.read(root/'launchers/submissions.json')['jobs']
    for job in jobs:queue.save(root/'runs'/job['arm_id']/'terminal_validation_results.json',{})
    monkeypatch.setattr(queue,'scheduler_history',lambda root,j:dict(job_id=j['job_id'],state='COMPLETED'))
    barriers=[]
    monkeypatch.setattr(queue,'standalone_barrier',lambda *args:barriers.append(True))
    queue.queue(root,once=True)
    assert len(submitted)==8 and len(barriers)==5


def test_uncertain_sbatch_never_resubmits(setup,monkeypatch):
    root,plan,active,submitted=setup
    calls=[]
    def uncertain(cmd):calls.append(cmd);raise subprocess.TimeoutExpired(cmd,45)
    monkeypatch.setattr(queue,'command',uncertain)
    with pytest.raises(subprocess.TimeoutExpired):queue.queue(root,once=True)
    with pytest.raises(ConfigError,match='Uncertain'):queue.queue(root,once=True)
    assert len(calls)==1
    assert queue.read(root/'launchers/submissions.json')['jobs'][0]['status']=='uncertain'


def test_delayed_squeue_visibility_respects_durable_submissions(setup,monkeypatch):
    root,plan,active,submitted=setup
    monkeypatch.setattr(queue,'queue_state',lambda:[])
    queue.queue(root,once=True)
    assert len(submitted)==4
    with pytest.raises(ConfigError,match='Uncertain'):queue.queue(root,once=True)
    assert len(submitted)==4


@pytest.mark.parametrize('active_count,foreign,limit,expected',[(3,False,4,1),(4,False,4,0),(1,True,4,0),(0,False,1,1)])
def test_unrelated_jobs_and_stricter_submission_limits(setup,monkeypatch,active_count,foreign,limit,expected):
    root,plan,active,submitted=setup
    active.extend(dict(job_id=str(n),name='unrelated',state='PENDING',qos='other' if foreign else queue.QOS) for n in range(active_count))
    monkeypatch.setattr(queue,'scheduler_limits',lambda:dict(max_running=1,max_submitted=limit))
    queue.queue(root,once=True)
    assert len(submitted)==expected


def test_failed_own_checkpoint_gets_monotonic_attempt(setup,monkeypatch):
    root,plan,active,submitted=setup
    queue.queue(root,once=True); gone=active.pop(0)
    monkeypatch.setattr(queue,'scheduler_history',lambda root,j:dict(job_id=j['job_id'],state='FAILED',allocation_seconds='12'))
    monkeypatch.setattr(queue,'continuation',lambda *args:dict(mode='resume',step=3,checkpoint={'sha256':'durable'}))
    queue.queue(root,once=True)
    jobs=queue.read(root/'launchers/submissions.json')['jobs']
    assert len(submitted)==5 and jobs[-1]['attempt_id']==2 and jobs[-1]['arm_id']=='ST-g125'
    assert jobs[0]['scheduler_accounting']['allocation_seconds']=='12'


def test_lock_prevents_duplicate_queue(setup):
    root,*_=setup
    with queue.lock(root/'launchers/queue.lock'):
        with pytest.raises(ConfigError,match='locked'):queue.queue(root,once=True)


def test_worker_requires_sbatch(monkeypatch,tmp_path):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(ConfigError,match='sbatch'):queue.worker(tmp_path,'S1',1)


def test_worker_revalidates_barrier_before_trainer(setup,monkeypatch):
    root,plan,active,submitted=setup
    intent=dict(arm_id='S1',attempt_id=1,name='mw-v1-S1-a1',job_id='1',status='submitted',bindings=plan['bindings'],gpu_gate_hash='gpu')
    queue.save(root/'launchers/submissions.json',{'jobs':[intent]})
    monkeypatch.setenv('SLURM_JOB_ID','1');monkeypatch.setenv('SLURM_JOB_NAME',intent['name'])
    def reject(*args):raise ConfigError('standalone changed')
    monkeypatch.setattr(queue,'standalone_barrier',reject)
    monkeypatch.setattr(queue.subprocess,'run',lambda *args,**kwargs:pytest.fail('must not train'))
    with pytest.raises(ConfigError,match='standalone changed'):queue.worker(root,'S1',1)


@pytest.mark.parametrize('damage',['source','config','status','results','artifact'])
def test_gate_rejects_stale_or_incomplete_evidence(tmp_path,damage):
    artifact=tmp_path/'result';artifact.write_text('passed')
    binding={'source':'new','config':'new'}
    gate=dict(mode='cpu',status='passed',bindings=copy.deepcopy(binding),checks=[dict(returncode=0,tests=1,failures=0,errors=0,skipped=0,
        artifacts=[dict(path=str(artifact),sha256=queue.digest(artifact))])])
    if damage in ('source','config'):gate['bindings'][damage]='old'
    elif damage=='status':gate['status']='failed'
    elif damage=='results':gate['checks']=[]
    else:artifact.write_text('changed')
    path=tmp_path/'gate.json';queue.save(path,queue.sealed(gate))
    with pytest.raises(ConfigError):queue.verify_gate(tmp_path,'cpu',binding,path)


def test_gpu_skips_cannot_pass(tmp_path,monkeypatch):
    gate=queue.sealed(dict(mode='gpu',status='passed',bindings={},checks=[dict(returncode=0,tests=1,skipped=1)],job_id='1',hardware='GPU',cpu_gate_hash='cpu',real_shape_arms=list(queue.ARMS)))
    path=tmp_path/'gpu.json';queue.save(path,gate)
    with pytest.raises((ConfigError,FileNotFoundError)):queue.verify_gate(tmp_path,'gpu',{},path)


def test_progress_uses_assigned_horizon(tmp_path):
    path=tmp_path/'runs/ST-g125/heartbeats.jsonl';path.parent.mkdir(parents=True)
    path.write_text('\n'.join(json.dumps(dict(step=s,elapsed_seconds=t)) for s,t in [(10,10),(20,30)]))
    result=queue.progress(tmp_path,'ST-g125',87132)
    assert result['eta_seconds']==(87132-20)*2


@pytest.mark.parametrize('running,submitted,association,valid',[(2,4,'1|2',True),(3,4,'|',False),(0,4,'|',False),(1,2,'|',True)])
def test_live_scheduler_limits(monkeypatch,running,submitted,association,valid):
    monkeypatch.setattr(queue.shutil, 'which', lambda name: '/fixture/sacctmgr')
    def command(cmd):
        if cmd[-1]=='config':return 'AccountingStorageEnforce = limits,qos'
        if cmd[0]=='sacctmgr':return association
        return f'QOS={queue.QOS}(1)\n      {queue.getpass.getuser()}(123)\n        MaxJobsPU={running}(0) MaxSubmitJobsPU={submitted}(0)\n'
    monkeypatch.setattr(queue,'command',command)
    if valid:
        result=queue.scheduler_limits()
        assert result['max_running']==1 and result['max_submitted']==2
    else:
        with pytest.raises(ConfigError):queue.scheduler_limits()


@pytest.mark.parametrize('damage', [None, 'qos', 'user', 'parent', 'cycle'])
def test_idle_user_controller_limits_and_stricter_ancestors(monkeypatch, damage):
    user = queue.getpass.getuser()
    raw = f'QOS|2|4|4294967295|4294967295|4294967295|4294967295\nASSOC|1|0||1|3|4294967295|4294967295\nASSOC|2|1|{user}|4294967295|4294967295|4294967295|4294967295'
    if damage == 'qos': raw = raw.replace('QOS|2|4', 'QOS|3|4')
    if damage == 'user': raw = raw.replace(user, 'unrelated')
    if damage == 'parent': raw = raw.replace('ASSOC|2|1|', 'ASSOC|2|9|')
    if damage == 'cycle': raw = raw.replace('ASSOC|1|0|', 'ASSOC|1|2|')
    monkeypatch.setattr(queue, 'command', lambda cmd: '' if cmd[0] == 'gcc' else raw)
    if damage:
        with pytest.raises(ConfigError): queue.controller_limits('limits,qos')
    else:
        result = queue.controller_limits('limits,qos')
        assert (result['max_running'], result['max_submitted']) == (1, 3)


def test_hard_kill_before_first_measurement_is_unknown(tmp_path):
    from src.training.run import ResourceAttemptLedger
    ledger=ResourceAttemptLedger(tmp_path,run_id='run')
    ledger.observe('process',sequence=1,run_id='run',launch_attempt_id=1,slurm_job_id='1',process_uuid='process',
        status='running',elapsed_seconds=None,attempted_steps=None,peak_allocated_bytes=None,peak_reserved_bytes=None,source_checkpoint=None)
    summary=ResourceAttemptLedger(tmp_path,run_id='run').summary()
    assert not summary['measurement_complete']
    assert summary['unobserved_attempts']==['process']
    assert summary['elapsed_seconds_is_lower_bound']


def test_gpu_requires_tested_snapshot_before_any_work(tmp_path,monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(ConfigError,match='sbatch'):diagnostics.execute(tmp_path,'gpu')


# These use complete terminal fixtures and the real schema/preflight reader.
from test_matformer_widths_reporting import terminal_campaign
from test_optimizer_ownership_campaign import audited_inputs


@pytest.fixture
def prepared(tmp_path,terminal_campaign):
    manifest_path,runs=terminal_campaign
    # Rehome the fixture preflight to the operational layout before sealing gates.
    (tmp_path/'preflight').rename(tmp_path/'campaign')
    manifest=queue.read(tmp_path/'campaign/campaign_manifest.json')
    reservation=queue.campaign._reservation_path(runs)
    record=queue.read(reservation);record['manifest_path']=str(tmp_path/'campaign/campaign_manifest.json');queue.save(reservation,record)
    source=tmp_path/'source';source.mkdir();(source/'runtime.py').write_text('fixture source\n')
    for relative in manifest['runs'][0]['initialization']['source_files_sha256']:
        target=source/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((queue.REPO/relative).read_bytes())
    files=queue.source_files(source)
    queue.save(tmp_path/'diagnostics/source-manifest.json',queue.sealed(dict(files=files,source_sha256=queue.stable_hash(files))))
    binding=queue.bindings(tmp_path)
    gate=queue.sealed(dict(mode='cpu',status='passed',bindings=binding,
        checks=[dict(returncode=0,tests=1,failures=0,errors=0,skipped=0,artifacts=[])]))
    queue.save(tmp_path/'diagnostics/cpu-gate.json',gate)
    return tmp_path,manifest,gate


def test_prepare_adopts_reservation_and_is_idempotent(prepared):
    import shutil
    root,manifest,gate=prepared
    shutil.rmtree(root/'runs')
    reservation=queue.campaign._reservation_path(root/'runs');before=reservation.read_bytes()
    plan=queue.prepare(root,root/'diagnostics/cpu-gate.json')
    assert reservation.read_bytes()==before and plan['cpu_gate_hash']==gate['content_hash']
    assert plan['reference_status']=='outstanding'
    assert queue.prepare(root,root/'diagnostics/cpu-gate.json')==plan


@pytest.mark.parametrize('damage',['source','config','preflight','reservation'])
def test_prepare_rejects_changed_bound_inputs(prepared,damage):
    import shutil
    root,manifest,gate=prepared
    shutil.rmtree(root/'runs')
    paths=dict(source=root/'source/runtime.py',config=root/'campaign/configs/C3.yaml',preflight=root/'campaign/preflight.json',reservation=queue.campaign._reservation_path(root/'runs'))
    path=paths[damage]
    if damage in ('source','config'):path.write_text(path.read_text()+'# changed\n')
    else:
        data=queue.read(path);data['manifest_hash']='changed';queue.save(path,data)
    with pytest.raises(ConfigError):queue.prepare(root,root/'diagnostics/cpu-gate.json')
    assert not (root/'launchers/plan.json').exists()


def test_strict_barrier_binds_all_four_sources_and_detects_tamper(prepared):
    root,manifest,gate=prepared
    plan=dict(bindings=queue.bindings(root))
    barrier=queue.standalone_barrier(root,plan)
    assert len(barrier['terminals'])==4
    assert queue.standalone_barrier(root,plan)==barrier
    sidecar=root/'runs/ST-g125/terminal_validation_results.json'
    data=queue.read(sidecar);data['actual_updates']=4
    data['content_hash']=queue.stable_hash({k:v for k,v in data.items() if k!='content_hash'})
    queue.save(sidecar,data)
    with pytest.raises(ConfigError):queue.standalone_barrier(root,plan)
    assert queue.read(root/'launchers/standalone-barrier.json')==barrier


def test_restart_report_preserves_new_success_when_history_missing(prepared,monkeypatch):
    root,manifest,gate=prepared
    monkeypatch.setattr(queue,'verify_plan',lambda *args,**kwargs:{'reference_manifest':None})
    with pytest.raises(ConfigError,match='Historical reference unavailable'):queue.report(root)
    status=queue.read(root/'launchers/completion.json')
    assert status['production']==status['new_report']=='complete' and status['combined_report']=='outstanding'
    first=(root/'reports/new/comparison_report.json').read_bytes()
    with pytest.raises(ConfigError,match='Historical reference unavailable'):queue.report(root)
    assert (root/'reports/new/comparison_report.json').read_bytes()==first
    (root/'reports/new/endpoints.csv').write_text('corrupt')
    with pytest.raises(ConfigError,match='output changed'):queue.report(root)
    assert queue.read(root/'launchers/completion.json')['new_report']=='pending'


def test_cpu_snapshot_is_frozen_before_checks(prepared,monkeypatch):
    import shutil
    root,manifest,gate=prepared
    shutil.rmtree(root/'source');(root/'diagnostics/source-manifest.json').unlink()
    calls=[]
    def check(source,output,*,gpu=False):
        assert source==root/'source' and not gpu
        files=queue.source_files(source)
        assert files==queue.read(root/'diagnostics/source-manifest.json')['files']
        assert not ((source/'src/training/checkpointing.py').stat().st_mode & 0o222)
        # Snapshot provenance must work without a Git checkout.
        result=subprocess.run([queue.PYTHON,'-c','from src.evaluation.optimizer_ownership import _provenance; assert _provenance()["source_files_sha256"]'],cwd=source,
            env={**__import__('os').environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        calls.append(str(source))
        return dict(command=['fixture-check'],returncode=0,tests=1,failures=0,errors=0,skipped=0,artifacts=[])
    monkeypatch.setattr(diagnostics,'pytest_check',check)
    result=diagnostics.execute(root,'cpu')
    assert calls==[str(root/'source')]
    assert result['bindings']==queue.bindings(root)
    assert queue.verify_gate(root,'cpu')==result
    # Make fixture cleanup possible; operational snapshots remain read-only.
    for p in (root/'source').rglob('*'):p.chmod(0o755 if p.is_dir() else 0o644)
    (root/'source').chmod(0o755)


def test_pytest_evidence_records_actual_result_and_command(tmp_path,monkeypatch):
    output=tmp_path/'evidence';output.mkdir()
    source=tmp_path/'source';source.mkdir()
    (source/'tests').mkdir()
    def run(cmd,**kwargs):
        assert kwargs['cwd']==output/'pytest-workspace'
        assert (kwargs['cwd']/'tests').resolve()==source/'tests'
        assert kwargs['env']['PYTHONPATH']==str(source)
        assert set('tests/'+s for s in diagnostics.CPU_SUITES)<=set(cmd)
        assert kwargs['env']['MATFORMER_DIAGNOSTIC_DEVICE']=='cpu'
        (output/'pytest.xml').write_text('<testsuites><testsuite tests="12" failures="1" errors="0" skipped="2"/></testsuites>')
        kwargs['stdout'].write('actual failure')
        return subprocess.CompletedProcess(cmd,1)
    monkeypatch.setattr(diagnostics.subprocess,'run',run)
    result=diagnostics.pytest_check(tmp_path/'source',output)
    assert (result['returncode'],result['tests'],result['failures'],result['skipped'])==(1,12,1,2)
    assert len(result['artifacts'])==2


def test_worker_launch_records_process_and_never_reuses_attempt(setup,monkeypatch):
    root,plan,active,submitted=setup
    queue.queue(root,once=True)
    intent=queue.read(root/'launchers/submissions.json')['jobs'][0]
    monkeypatch.setenv('SLURM_JOB_ID',intent['job_id']);monkeypatch.setenv('SLURM_JOB_NAME',intent['name'])
    queue.save(root/'campaign/campaign_manifest.json',{'runs':[dict(arm_id='ST-g125',run_id='run')]})
    calls=[]
    def run(cmd,**kwargs):
        calls.append(kwargs['env'])
        assert kwargs['env']['MATFORMER_LAUNCH_ATTEMPT_ID']=='1'
        return subprocess.CompletedProcess(cmd,0)
    monkeypatch.setattr(queue.subprocess,'run',run)
    assert queue.worker(root,'ST-g125',1)==0
    worker=queue.read(root/'launchers/worker-ST-g125-1.json')
    assert worker['process_uuid']==calls[0]['MATFORMER_PROCESS_UUID']
    with pytest.raises(ConfigError,match='already has a process'):queue.worker(root,'ST-g125',1)
    assert len(calls)==1


def test_ambiguous_scheduler_name_cannot_admit_duplicate(setup):
    root,plan,active,submitted=setup
    queue.queue(root,once=True)
    active.append({**active[0],'job_id':'different'})
    with pytest.raises(ConfigError,match='Duplicate'):queue.queue(root,once=True)
    assert len(submitted)==4


def test_diagnostic_identity_retains_strict_production_controls(prepared,tmp_path):
    import yaml
    from src.utils.config import resolve_run_config
    root,manifest,gate=prepared
    definition=manifest['runs'][-1]
    raw=copy.deepcopy(definition['executable_config'])
    name='mw-gpu-diagnostic-123-C3';output=root/'diagnostics/runs'/name
    raw['run'].update(run_id=name,output_dir=str(output))
    raw['optimizer_ownership_contract']['run_id']=name
    raw['optimizer_ownership_contract_hash']=queue.stable_hash(raw['optimizer_ownership_contract'])
    path=tmp_path/'probe.yaml';path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError):resolve_run_config(path)
    with diagnostics.diagnostic_identity(definition,output,name):
        config=resolve_run_config(path)
        assert config['training']['max_steps']==348528
        config['training']['token_budget']-=8192
        with pytest.raises(ConfigError):queue.campaign.validate_materialized_config(config)
    with pytest.raises(ConfigError):resolve_run_config(path)


@pytest.mark.parametrize('damage',[None,'identity','moment','action_rng','cursor'])
def test_own_checkpoint_continuation_inspects_real_bundle(tmp_path,monkeypatch,damage):
    import torch
    from test_matformer_widths_campaign import mw_resume_fixture
    from test_optimizer_ownership_resume import train,save
    from src.training import modeling,data,run
    from src.utils import config as configs
    bundle=mw_resume_fixture(tmp_path,'C3')
    config=bundle[0];output=tmp_path/'runs/C3';config['run']['output_dir']=str(output)
    train(bundle,stop=3)
    checkpoint=output/'checkpoints/latest.pt';save(bundle,checkpoint)
    payload=torch.load(checkpoint,weights_only=False)
    if damage=='identity':payload['run_id']='old-run'
    elif damage=='moment':next(iter(payload['optimizer_state_collection']['ordered_owners'][0]['state_dict']['state'].values()))['exp_avg'].fill_(float('nan'))
    elif damage=='action_rng':
        import random
        for rng in [payload['reproducibility']['rng_state'],*payload['reproducibility']['rng_states_by_rank']]:rng['dedicated']['granularity_selection']=random.Random(1).getstate()
    elif damage=='cursor':payload['sampler_state']['total_cursor']+=1
    torch.save(payload,checkpoint)
    inspection_model=mw_resume_fixture(tmp_path/'model','C3')[1]
    monkeypatch.setattr(configs,'resolve_run_config',lambda *args:copy.deepcopy(config))
    monkeypatch.setattr(modeling,'build_model',lambda config:inspection_model)
    monkeypatch.setattr(data,'build_packed_mmap_dataloaders',lambda *args:(bundle[4],[],[],{}))
    monkeypatch.setattr(run,'_packed_role_partition',lambda m:{})
    monkeypatch.setattr(run,'_attach_probabilistic_role_provenance',lambda *args:None)
    before=checkpoint.read_bytes()
    if damage:
        with pytest.raises(ConfigError):queue.continuation(tmp_path,'C3')
    else:
        result=queue.continuation(tmp_path,'C3')
        assert result['mode']=='resume' and result['step']==3 and result['checkpoint']['sha256']==queue.digest(checkpoint)
    assert checkpoint.read_bytes()==before


def test_unknown_scheduler_state_cannot_grant_retry(setup,monkeypatch):
    root,plan,active,submitted=setup
    queue.queue(root,once=True);active.clear()
    monkeypatch.setattr(queue,'scheduler_history',lambda root,j:dict(job_id=j['job_id'],state='UNKNOWN'))
    with pytest.raises(ConfigError,match='Unrecognized'):queue.queue(root,once=True)
    assert len(submitted)==4


def test_worker_exception_without_child_exit_is_not_terminal(tmp_path,monkeypatch):
    intent=dict(arm_id='C3',attempt_id=2,name='name',created_date='2026-09-21',job_id='1')
    queue.save(tmp_path/'launchers/worker-C3-2.json',dict(job_id='1',status='failed',error='interrupted'))
    monkeypatch.setattr(queue,'command',lambda args:'')
    assert queue.scheduler_history(tmp_path,intent) is None


def test_gpu_acceptance_uses_one_process(tmp_path,monkeypatch):
    output=tmp_path/'result';output.mkdir()
    def main(args):
        assert __import__('os').environ['MATFORMER_DIAGNOSTIC_DEVICE']=='cuda'
        assert 'mw_gpu_semantic' in args[-1]
        (output/'pytest.xml').write_text('<testsuites><testsuite tests="37" failures="0" errors="0" skipped="0"/></testsuites>')
        return 0
    monkeypatch.setattr(pytest,'main',main)
    monkeypatch.setattr(diagnostics.subprocess,'run',lambda *args,**kwargs:pytest.fail('must not spawn another GPU process'))
    result=diagnostics.pytest_check(tmp_path,output,gpu=True)
    assert result['execution']=='in_process_pytest' and result['tests']==37


def test_operational_cli_requires_tested_snapshot(tmp_path,capsys):
    with pytest.raises(SystemExit) as error:
        queue.main(['queue','--campaign-root',str(tmp_path),'--once'])
    assert error.value.code==2
    assert 'tested campaign source snapshot' in capsys.readouterr().err
