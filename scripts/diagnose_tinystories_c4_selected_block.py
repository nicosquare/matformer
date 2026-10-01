#!/usr/bin/env python3
"""Real-shape BF16 CUDA gate for both C4 selected-block grids."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from src.training import steps
from src.training.modeling import build_model
from src.utils.config import ConfigError, resolve_run_config
from src.utils.metrics import write_json_artifact
from src.utils.reproducibility import configure_strict_determinism, deterministic_runtime_settings, seed_model_initialization


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnose(grid):
    path = ROOT / f"configs/controlled_exps/tinystories_instruct_c4_selected_block_{grid}.yaml"
    config = resolve_run_config(path, create_output_dirs=False)
    if not deterministic_runtime_settings().get('deterministic_algorithms'):
        raise ConfigError('C4 strict deterministic runtime was not established')
    seed_model_initialization(config)
    model = build_model(config).to('cuda:0')
    optimizer, clock = steps.build_optimizer_and_scheduler(model, config['training'])
    widths = config['model']['granularities']
    tokens = (torch.arange(64 * 128, device='cuda:0').reshape(64, 128) % (config['model']['vocab_size'] - 1)) + 1
    observations = []
    for width in [*widths, *widths]:
        model.zero_grad(set_to_none=True)
        model.configure_subnetwork(width)
        selected = optimizer.active_owner_ids(width)
        before = {owner.owner_id: [p.detach().clone() for p in owner.parameters]
                  for owner in optimizer.owners if owner.owner_id not in selected}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(input_ids=tokens, labels=tokens).loss
        if not torch.isfinite(loss):
            raise RuntimeError('C4 BF16 loss is not finite')
        loss.backward()
        earlier_gradients = sum(p.grad is not None for owner in optimizer.owners
                                if owner.owner_id not in selected for p in owner.parameters)
        if width != widths[0] and earlier_gradients == 0:
            raise RuntimeError('Earlier blocks were detached from the selected-width graph')
        for owner in optimizer.owners:
            if owner.owner_id not in selected:
                for parameter in owner.parameters:
                    parameter.grad = None
        clipping = steps.clip_optimizer_gradients(
            model, config['training'], width, owners=optimizer.owners,
            selected_owner_ids=selected,
        )
        if clipping['mode'] != 'global' or clipping['combined_post_norm'] > 1.00001:
            raise RuntimeError('C4 selected owner plus common joint clipping failed')
        applied_lr = optimizer.current_learning_rates[0]
        for owner_id in selected:
            optimizer.optimizer_for(owner_id).step()
        clock.step()
        clock.synchronize(optimizer)
        optimizer.record_successful_update(width, returned_owners=selected)
        optimizer.validate_accounting(step=clock.position)
        for owner in optimizer.owners:
            if owner.owner_id in selected:
                continue
            if any(not torch.equal(parameter, saved) for parameter, saved in zip(owner.parameters, before[owner.owner_id])):
                raise RuntimeError(f'Unselected owner {owner.owner_id} changed on {width}')
        observations.append(dict(width=width, selected_owners=list(selected),
                                 earlier_gradient_tensors=earlier_gradients,
                                 clipping=clipping, applied_lr=applied_lr,
                                 committed_step=clock.position, loss=float(loss.detach().float().cpu())))
    clone_model = build_model(config).to('cuda:0')
    clone, clone_clock = steps.build_optimizer_and_scheduler(clone_model, config['training'])
    clone.load_state_dict(optimizer.state_dict())
    clone_clock.load_state_dict(clock.state_dict())
    clone.validate_accounting(step=8)
    if clone_clock.position != 8 or clone.successful_update_counts != optimizer.successful_update_counts:
        raise RuntimeError('C4 checkpoint restore changed owner or scheduler counters')
    return dict(grid=grid, config_sha256=sha256(path), run_id=config['run']['run_id'],
                width_selection_counts=optimizer.width_selection_counts,
                owner_call_counts=optimizer.successful_update_counts,
                observations=observations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    first = resolve_run_config(ROOT / 'configs/controlled_exps/tinystories_instruct_c4_selected_block_linear.yaml', create_output_dirs=False)
    configure_strict_determinism(first)
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ConfigError('C4 diagnostic requires one BF16 CUDA device under sbatch')
    started = time.time()
    hardware = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True).stdout.strip()
    result = dict(status='passed', job_id=os.environ['SLURM_JOB_ID'], hardware=hardware,
                  torch_version=torch.__version__, started_at=started,
                  grids=[diagnose(grid) for grid in ('linear', 'geometric')],
                  finished_at=time.time())
    write_json_artifact(args.output, result)
    print(json.dumps(dict(status=result['status'], job_id=result['job_id'],
                          grids=[row['grid'] for row in result['grids']])))


if __name__ == '__main__':
    main()
