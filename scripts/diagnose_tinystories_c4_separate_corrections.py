#!/usr/bin/env python3
"""Real-shape BF16 CUDA gate for all four separate-correction C4 arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import sys

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
    path = ROOT / f"configs/controlled_exps/tinystories_instruct_c4_separate_{grid}.yaml"
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
        applied_rates = [optimizer.step_owner(owner_id) for owner_id in selected]
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
                                 clipping=clipping, applied_lr=applied_lr, owner_step_learning_rates=applied_rates,
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


def real_corpus_resume(arm, output):
    """Full horizon retained; actual trainer/data resume at steps 63, 64 and 65."""
    import copy
    from src.training import checkpointing as cp, data
    from src.utils.reproducibility import seed_training_randomness
    path = ROOT / f'configs/controlled_exps/tinystories_instruct_c4_separate_{arm}.yaml'
    config = resolve_run_config(path, create_output_dirs=False)
    config['run']['output_dir'] = str(output / arm)
    config['outputs']['save_checkpoints'] = False
    config['evaluation']['validation']['enabled'] = False
    config['evaluation']['validation']['run_at_completion'] = False
    config['evaluation']['validation']['interval_steps'] = 0
    config['training']['eval_interval'] = 0
    config['evaluation']['validation']['interval_tokens'] = 0
    loader, _, _, _ = data.build_packed_mmap_dataloaders(config, torch.device('cuda'))
    seed_model_initialization(config)
    model = build_model(config).to('cuda')
    opt, clock = steps.build_optimizer_and_scheduler(model, config['training'])
    seed_training_randomness(config)
    state = cp.build_initial_continuation_state(config)
    def advance(model, opt, clock, loader, state, stop):
        if state['last_completed_step'] == stop:
            return
        if state.get('sampler_state'):
            data.restore_packed_sampler_state(loader, state['sampler_state'])
        def committed(**kw):
            if kw['step'] == stop:
                raise StopIteration('GPU committed probe boundary')
        try:
            steps.train_for_steps(config, model, loader, [], opt, clock, torch.device('cuda'), run_state=state, successful_step_callback=committed)
        except StopIteration:
            pass
        assert state['last_completed_step'] == stop and clock.position == stop
    observations = []
    checkpoint = output / f'{arm}-probe.pt'
    for boundary in (63, 64, 65):
        advance(model, opt, clock, loader, state, boundary)
        cp.save_model_checkpoint(config, model, opt, clock, checkpoint,
            {'checkpoint_status':'latest', 'checkpoint_metric':None, 'checkpoint_metric_value':None, 'checkpoint_selection_step':None}, state)
        advance(model, opt, clock, loader, state, boundary + 1)
        expected_model = copy.deepcopy(model.state_dict())
        expected_opt = copy.deepcopy(opt.state_dict())
        expected_clock = copy.deepcopy(clock.state_dict())
        clone = build_model(config).to('cuda')
        clone_opt, clone_clock = steps.build_optimizer_and_scheduler(clone, config['training'])
        clone_loader, _, _, _ = data.build_packed_mmap_dataloaders(config, torch.device('cuda'))
        clone_state = cp.load_checkpoint_state(checkpoint, clone, clone_opt, clone_clock, config=config, train_dataloader=clone_loader)
        data.restore_packed_sampler_state(clone_loader, clone_state['sampler_state'])
        advance(clone, clone_opt, clone_clock, clone_loader, clone_state, boundary + 1)
        def equal(a, b):
            if torch.is_tensor(a):
                assert torch.equal(a, b)
            elif isinstance(a, dict):
                assert a.keys() == b.keys()
                for key in a: equal(a[key], b[key])
            elif isinstance(a, (tuple, list)):
                assert len(a) == len(b)
                for left, right in zip(a, b): equal(left, right)
            else: assert a == b
        equal(expected_model, clone.state_dict())
        equal(expected_opt, clone_opt.state_dict())
        equal(expected_clock, clone_clock.state_dict())
        model, opt, clock, loader, state = clone, clone_opt, clone_clock, clone_loader, clone_state
        observations.append(dict(boundary=boundary, resumed_step=boundary+1, nominal_lr=clock.last_committed_learning_rates,
                                 checkpoint_sha256=sha256(checkpoint), action_state_sha256=hashlib.sha256(repr(state['global_sampling_state']).encode()).hexdigest()))
    return dict(arm=arm, batch=64, context=128, precision='bf16', retained_horizon=config['training']['max_steps'], observations=observations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    first = resolve_run_config(ROOT / 'configs/controlled_exps/tinystories_instruct_c4_separate_linear-GMC.yaml', create_output_dirs=False)
    configure_strict_determinism(first)
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ConfigError('C4 diagnostic requires one BF16 CUDA device under sbatch')
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_c4_separate_corrections.py', '--override-ini=addopts=', '-k', 'cuda'], check=True)
    started = time.time()
    hardware = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True).stdout.strip()
    result = dict(status='passed', job_id=os.environ['SLURM_JOB_ID'], hardware=hardware,
                  torch_version=torch.__version__, started_at=started,
                  source_manifest_sha256=sha256(args.output.parent / 'source-manifest.json'),
                  grids=[diagnose(f'{grid}-{kind}') for grid in ('linear', 'geometric') for kind in ('GMC', 'LMC-only')],
                  real_corpus_resume=[real_corpus_resume(f'{grid}-{kind}', args.output.parent) for grid in ('linear','geometric') for kind in ('GMC','LMC-only')],
                  finished_at=time.time())
    write_json_artifact(args.output, result)
    print(json.dumps(dict(status=result['status'], job_id=result['job_id'],
                          grids=[row['grid'] for row in result['grids']])))


if __name__ == '__main__':
    main()
