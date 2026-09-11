"""Submission recovery and admission limits without contacting Slurm."""
import json
from pathlib import Path
import pytest
from scripts import run_optimizer_ownership_corrections as queue


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, 'BASE', tmp_path)
    monkeypatch.setattr(queue, 'verify_scheduler_limits', lambda: {'max_running':2,'max_submitted':4})
    for name in ('launchers', 'diagnostics', 'logs'):
        (tmp_path/name).mkdir()
    for gate in ('cpu','gpu'):
        (tmp_path/'diagnostics'/f'{gate}-gate.json').write_text(json.dumps({'status':'passed','runtime_source_sha256':'source'}))
    plan = {'campaign_id':'test','source_files_sha256':{},'config_sha256':{},
            'source_dir':str(tmp_path),'runtime_source_sha256':'source','created_date':'2026-09-10'}
    return tmp_path, plan


def test_queue_submits_four_and_restart_does_not_duplicate(setup, monkeypatch):
    root, plan = setup
    active, submissions = [], []
    monkeypatch.setattr(queue, 'queue_state', lambda: active.copy())
    def command(cmd):
        if cmd[0] == 'scontrol': return 'GRES=gpu:a100:1(IDX:0)'
        assert cmd[0] == 'sbatch'
        assert '--exclude=gpu-[05,50,51]' in cmd
        assert '--gres=gpu:1' in cmd
        assert '--no-requeue' in cmd
        job_id = str(100+len(submissions))
        submissions.append(cmd)
        active.append({'job_id':job_id,'qos':queue.QOS,'name':next(c.split('=',1)[1] for c in cmd if c.startswith('--job-name='))})
        return job_id
    monkeypatch.setattr(queue, 'command', command)
    queue.queue(plan, once=True)
    queue.queue(plan, once=True)
    assert len(submissions) == 4
    record = json.loads((root/'launchers/submissions.json').read_text())
    assert [r['arm_id'] for r in record['jobs']] == ['C1-GMC','C1-LMC','C2-GMC','C2-LMC']
    # When one terminal frees a submission slot, queue C3 immediately, even
    # while two jobs remain running and another is pending.
    done = active.pop(0)
    terminal = root/'runs/C1-GMC/terminal_validation_results.json'
    terminal.parent.mkdir(parents=True)
    terminal.write_text('{}')
    monkeypatch.setattr(queue, 'accounting', lambda job: {'state':'COMPLETED'})
    queue.queue(plan, once=True)
    assert len(submissions) == 5 and len(active) == 4
    record = json.loads((root/'launchers/submissions.json').read_text())
    assert record['jobs'][-1]['arm_id'] == 'C3-GMC'
    assert record['jobs'][0]['job_id'] == done['job_id']


@pytest.mark.parametrize('active', [
    [{'job_id':str(i),'qos':queue.QOS} for i in range(4)],
    [{'job_id':'1','qos':'another-qos'}],
])
def test_other_user_jobs_consume_admission_slots(setup, monkeypatch, active):
    root, plan = setup
    monkeypatch.setattr(queue, 'queue_state', lambda: active)
    monkeypatch.setattr(queue, 'command', lambda cmd: pytest.fail('must not submit'))
    queue.queue(plan, once=True)
    assert not (root/'launchers/submissions.json').exists()


@pytest.mark.parametrize('running,submitted,enforcement,valid', [
    ('2','4','associations,limits,nosteps,qos',True),
    ('3','4','limits,qos',False),
    ('N','4','limits,qos',False),
    ('2','2','limits,qos',False),
    ('2','4','associations',False),
])
def test_scheduler_limit_verification(monkeypatch, running, submitted, enforcement, valid):
    def command(cmd):
        if cmd[-1] == 'config': return 'AccountingStorageEnforce = '+enforcement
        return (f'QOS={queue.QOS}(64)\n    User Limits\n      ivo.navarrete(2196)\n'
                f'        MaxJobsPU={running}(2) MaxJobsAccruePU=N(0) MaxSubmitJobsPU={submitted}(2)\n'
                '      someone.else(1234)\n        MaxJobsPU=2(0) MaxSubmitJobsPU=4(0)\n')
    monkeypatch.setattr(queue, 'command', command)
    if valid:
        assert queue.verify_scheduler_limits()['max_running'] == 2
    else:
        with pytest.raises(RuntimeError): queue.verify_scheduler_limits()


def test_unverified_scheduler_blocks_before_submission_intent(setup, monkeypatch):
    root, plan = setup
    monkeypatch.setattr(queue, 'queue_state', lambda: [])
    def fail(): raise RuntimeError('unverified')
    monkeypatch.setattr(queue, 'verify_scheduler_limits', fail)
    with pytest.raises(RuntimeError, match='unverified'): queue.queue(plan, once=True)
    assert not (root/'launchers/submissions.json').exists()


def test_submission_limit_survives_delayed_squeue_visibility(setup, monkeypatch):
    root, plan = setup
    monkeypatch.setattr(queue, 'queue_state', lambda: [])
    calls = []
    def submit(cmd):
        assert cmd[0] == 'sbatch'
        calls.append(cmd)
        return str(100+len(calls))
    monkeypatch.setattr(queue, 'command', submit)
    queue.queue(plan, once=True)
    assert len(calls) == 4


def test_restart_reconciles_uncertain_submission_by_unique_name(setup, monkeypatch):
    root, plan = setup
    name = 'oo-corr-v1-C1-GMC-a1'
    (root/'launchers/submissions.json').write_text(json.dumps({'jobs':[{'arm_id':'C1-GMC','name':name,'status':'submitting','gpu_index':0}]}))
    monkeypatch.setattr(queue, 'queue_state', lambda: [{'job_id':'10','name':name}, {'job_id':'11','name':'other'}])
    monkeypatch.setattr(queue, 'command', lambda cmd: pytest.fail('must not submit'))
    queue.queue(plan, once=True)
    assert json.loads((root/'launchers/submissions.json').read_text())['jobs'][0]['job_id'] == '10'


def test_stale_or_failed_gate_blocks_launch(setup, monkeypatch):
    root, plan = setup
    (root/'diagnostics/gpu-gate.json').write_text(json.dumps({'status':'passed','runtime_source_sha256':'old'}))
    monkeypatch.setattr(queue, 'queue_state', lambda: pytest.fail('must check gates first'))
    with pytest.raises(RuntimeError, match='gpu gate'):
        queue.queue(plan, once=True)


def test_accounting_uses_controller_when_database_unavailable(setup, monkeypatch):
    root, plan = setup
    def command(cmd):
        assert cmd[0] == 'scontrol'
        return 'JobId=10 JobName=test JobState=COMPLETED ExitCode=0:0 RunTime=01:02:03 StartTime=start EndTime=end AllocTRES=cpu=4,gres/gpu=1 NodeList=gpu-03'
    monkeypatch.setattr(queue, 'command', command)
    result = queue.accounting('10')
    assert result['elapsed_seconds'] == 3723
    assert result['state'] == 'COMPLETED' and result['source'] == 'scontrol'


def test_accounting_retains_worker_evidence_if_scheduler_forgets_job(setup, monkeypatch):
    root, plan = setup
    import subprocess
    def unavailable(cmd):
        raise subprocess.CalledProcessError(1,cmd)
    monkeypatch.setattr(queue, 'command', unavailable)
    (root/'launchers/worker-10.json').write_text(json.dumps({'job_id':'10','worker_exit_code':0,'allocation':'recorded'}))
    result = queue.accounting('10')
    assert result['scheduler_history_unavailable']
    assert result['worker']['allocation'] == 'recorded'


def test_gpu_monitor_selects_allocated_device_not_last_inventory_row(setup):
    root, plan = setup
    (root/'logs/C1-GMC-10-gpu.csv').write_text('timestamp, uuid, utilization.gpu, memory.used\nnow, GPU-a, 20 %, 1000 MiB\nnow, GPU-b, 99 %, 30000 MiB\n')
    assert 'GPU-a' in queue.progress('C1-GMC', 0)['gpu_observation']
    assert queue.progress('C1-GMC')['gpu_observation'] is None
    (root/'logs/C1-GMC-10-gpu.csv').write_text('index, timestamp, uuid, utilization.gpu, memory.used\n0, now, GPU-a, 20 %, 1000 MiB\n1, now, GPU-b, 99 %, 30000 MiB\n')
    assert 'GPU-a' in queue.progress('C1-GMC', 0)['gpu_observation']
