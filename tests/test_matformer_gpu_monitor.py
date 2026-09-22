"""GPU-use evidence must belong to the current attempt; CPU fallback cancels it."""
import copy
import pytest
from scripts import monitor_matformer_cuda_runs as monitor
from scripts import run_tinystories_matformer_widths as ops


@pytest.fixture
def observed(tmp_path, monkeypatch):
    intent=dict(arm_id='S1', attempt_id=2, job_id='123')
    ops.save(tmp_path/'launchers/cuda-entry-S1-2.json', dict(job_id='123',host='gpu-52',process_id=42))
    ops.save(tmp_path/'launchers/worker-S1-2.json', dict(job_id='123',process_uuid='process'))
    ops.save(tmp_path/'runs/S1/config.json', {'training': {'resolved_mixed_precision':'bf16'}})
    ledger={'attempts':{'process':dict(slurm_job_id='123',peak_allocated_bytes=100,peak_reserved_bytes=200,attempted_steps=10,elapsed_seconds=1)}}
    ops.save(tmp_path/'runs/S1/resource_attempts.json', ledger)
    cancelled=[]
    monkeypatch.setattr(ops,'command',lambda cmd:cancelled.append(cmd))
    return tmp_path,intent,ledger,cancelled


def test_actual_gpu_observation(observed):
    root,intent,_,cancelled=observed
    result=monitor.observe_gpu(root,ops,intent)
    assert result['job_id']=='123' and result['peak_allocated_bytes']==100
    assert not cancelled


def test_cpu_fallback_cancelled(observed):
    root,intent,_,cancelled=observed
    ops.save(root/'runs/S1/config.json', {'training': {'resolved_mixed_precision':'none'}})
    with pytest.raises(RuntimeError,match='cancelled unexpected CPU'):
        monitor.observe_gpu(root,ops,intent)
    assert cancelled==[['scancel','123']]


@pytest.mark.parametrize('damage',['no_gpu','no_steps','other_attempt','other_job'])
def test_missing_or_wrong_gpu_evidence_rejected(observed,damage):
    root,intent,ledger,_=observed
    ledger=copy.deepcopy(ledger)
    measurement=ledger['attempts']['process']
    if damage=='no_gpu':measurement['peak_allocated_bytes']=None
    if damage=='no_steps':measurement['attempted_steps']=0
    if damage=='other_attempt':ledger['attempts']={'old':measurement}
    if damage=='other_job':measurement['slurm_job_id']='old'
    ops.save(root/'runs/S1/resource_attempts.json',ledger)
    if damage=='other_job':
        with pytest.raises(RuntimeError,match='job identity'):monitor.observe_gpu(root,ops,intent)
    else:
        assert monitor.observe_gpu(root,ops,intent) is None
