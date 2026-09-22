#!/usr/bin/env python3
"""Run the unchanged trainer on an explicit CUDA device; never fall back to CPU."""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def required_training(config, *, training_function=None):
    import torch
    from src.utils.config import ConfigError
    if not os.environ.get('SLURM_JOB_ID'):
        raise ConfigError('CUDA-required training must run under sbatch')
    if socket.gethostname().split('.')[0] in {'gpu-05', 'gpu-50', 'gpu-51', 'gpu-54'}:
        raise ConfigError('CUDA-required training was assigned an excluded node')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ConfigError('Production requires exactly one visible usable CUDA GPU; CPU fallback is forbidden')
    if config['training']['mixed_precision'] != 'bf16':
        raise ConfigError('Production requires BF16')
    # Do not initialize CUDA here: the trainer must configure strict determinism
    # first. Its explicit CUDA context then checks native BF16 support and fails
    # before model construction if CUDA or BF16 cannot be used.
    if training_function is None:
        from src.training.run import run_training
        training_function = run_training
    return training_function(config, device='cuda:0')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--entry-evidence', required=True, type=Path)
    args = parser.parse_args()
    from src.utils.config import resolve_run_config
    from src.utils.metrics import write_json_artifact
    config = resolve_run_config(args.config, create_output_dirs=False)
    evidence = dict(status='starting', host=socket.gethostname(), job_id=os.environ.get('SLURM_JOB_ID'),
                    process_id=os.getpid(), requested_device='cuda:0', required_precision='bf16',
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    slurm_job_gpus=os.environ.get('SLURM_JOB_GPUS'), config=str(args.config))
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=15)
        evidence['nvidia_smi'] = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        evidence['nvidia_smi'] = dict(error=str(error))
    write_json_artifact(args.entry_evidence, evidence)
    print('CUDA-required trainer: explicit cuda:0, BF16, CPU fallback forbidden', flush=True)
    try:
        required_training(config)
    except BaseException as error:
        evidence.update(status='failed_or_interrupted', error=str(error))
        write_json_artifact(args.entry_evidence, evidence)
        raise


if __name__ == '__main__':
    main()
