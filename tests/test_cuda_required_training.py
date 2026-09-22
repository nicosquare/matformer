"""Production CUDA failures must stop before invoking the trainer."""
from unittest.mock import Mock
import pytest
import torch
from scripts import train_cuda_required as guard
from scripts import run_tinystories_matformer_widths as ops
from src.utils.config import ConfigError


@pytest.mark.parametrize('job,host,available,count,precision,message', [
    (None, 'gpu-52', True, 1, 'bf16', 'sbatch'),
    ('1', 'gpu-54', True, 1, 'bf16', 'excluded'),
    ('1', 'gpu-52', False, 0, 'bf16', 'CUDA'),
    ('1', 'gpu-52', True, 2, 'bf16', 'exactly one'),
    ('1', 'gpu-52', True, 1, 'none', 'BF16'),
])
def test_reject_before_training(monkeypatch, job, host, available, count, precision, message):
    if job: monkeypatch.setenv('SLURM_JOB_ID', job)
    else: monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.setattr(guard.socket, 'gethostname', lambda: host)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: available)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: count)
    trainer = Mock()
    with pytest.raises(ConfigError, match=message):
        guard.required_training({'training': {'mixed_precision': precision}}, training_function=trainer)
    trainer.assert_not_called()


def test_explicit_cuda_without_early_context_initialization(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    monkeypatch.setattr(guard.socket, 'gethostname', lambda: 'gpu-52')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'init', lambda: pytest.fail('CUDA initialized before strict determinism'))
    trainer = Mock(return_value='trained')
    config = {'training': {'mixed_precision': 'bf16'}}
    assert guard.required_training(config, training_function=trainer) == 'trained'
    trainer.assert_called_once_with(config, device='cuda:0')


def test_cpu_checkpoint_is_never_a_production_resume(tmp_path):
    ops.save(tmp_path/'runs/S1/config.json', {'training': {'resolved_mixed_precision': 'none'}})
    with pytest.raises(ConfigError, match='not a valid production continuation'):
        ops.continuation(tmp_path, 'S1')
