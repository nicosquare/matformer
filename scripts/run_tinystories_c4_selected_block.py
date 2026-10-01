#!/usr/bin/env python3
"""Prepare the immutable C4 source, gate CUDA, and submit two resumable runs."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import run_tinystories_matformer_widths as shared
from src.utils.config import ConfigError, resolve_run_config


CAMPAIGN = 'tinystories-optimizer-ownership-c4-selected-block-v1'
ROOT = Path('/nfs-stor/ivo.navarrete/results/elasticnn') / CAMPAIGN
GRIDS = ('linear', 'geometric')
ARMS = tuple(f'C4-{grid}' for grid in GRIDS)
PYTHON = shared.PYTHON


def path_config(root, grid):
    return root / 'source/configs/controlled_exps' / f'tinystories_instruct_c4_selected_block_{grid}.yaml'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def source_hashes(root):
    return {str(p.relative_to(root / 'source')): digest(p) for p in sorted((root / 'source').rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def verify(root):
    manifest = read(root / 'diagnostics/source-manifest.json')
    if manifest.get('campaign_id') != CAMPAIGN or source_hashes(root) != manifest['files']:
        raise ConfigError('C4 immutable source differs from prepared manifest')
    for grid in GRIDS:
        config = resolve_run_config(path_config(root, grid), create_output_dirs=False)
        if (config['run']['output_dir'] != str(root / 'runs' / config['run']['run_id'])
                or config['run']['campaign_id'] != CAMPAIGN
                or digest(path_config(root, grid)) != manifest['configs'][grid]):
            raise ConfigError(f'C4 {grid} executable config differs from prepared identity')
    return manifest


def prepare(root):
    if root.exists():
        verify(root)
        return
    root.mkdir(parents=True)
    for name in ('logs', 'diagnostics', 'runs'):
        (root / name).mkdir()
    source = root / 'source'
    source.mkdir()
    for name in ('src', 'scripts', 'configs', 'tests'):
        shutil.copytree(REPO / name, source / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
    for name in ('train.py', 'pyproject.toml', 'requirements.txt', 'build_backend.py'):
        if (REPO / name).exists():
            shutil.copy2(REPO / name, source / name)
    configs = {grid: digest(path_config(root, grid)) for grid in GRIDS}
    files = source_hashes(root)
    save(root / 'diagnostics/source-manifest.json', dict(campaign_id=CAMPAIGN,
        created_at=time.time(), files=files, configs=configs))
    for item in source.rglob('*'):
        item.chmod(0o555 if item.is_dir() else 0o444)
    source.chmod(0o555)
    verify(root)


def refresh_diagnostic(root):
    """Record failed gate source and publish a corrected diagnostic revision."""
    old = verify(root)
    if (root / 'diagnostics/production-intents.json').exists():
        raise ConfigError('Cannot revise the C4 tested source after production admission')
    revision = int(old.get('revision', 1)) + 1
    archive = root / 'diagnostics/revisions' / f'revision-{revision - 1}'
    if archive.exists():
        raise ConfigError('C4 diagnostic revision archive already exists')
    archive.mkdir(parents=True)
    shutil.copy2(root / 'diagnostics/source-manifest.json', archive / 'source-manifest.json')
    for relative in ('scripts/diagnose_tinystories_c4_selected_block.py',
                     'scripts/run_tinystories_c4_selected_block.py', 'src/utils/metrics.py',
                     'tests/test_c4_selected_block.py'):
        target = root / 'source' / relative
        (archive / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, archive / relative)
        target.chmod(0o644)
        shutil.copy2(REPO / relative, target)
        target.chmod(0o444)
    for name in ('gpu-gate.json', 'gpu-intent.json'):
        artifact = root / 'diagnostics' / name
        if artifact.exists():
            shutil.copy2(artifact, archive / name)
            artifact.unlink()
    old['previous_manifest_sha256'] = digest(archive / 'source-manifest.json')
    old['revision'] = revision
    old['revised_at'] = time.time()
    old['files'] = source_hashes(root)
    save(root / 'diagnostics/source-manifest.json', old)
    verify(root)


def admission(root):
    verify(root)
    limits = shared.scheduler_limits()
    active = shared.queue_state()
    if any(row['qos'] != shared.QOS and row['name'] != 'c4-selected-block-report' for row in active):
        raise ConfigError('An active user job has a different QoS; C4 admission is uncertain')
    gpu = [row for row in active if row['qos'] == shared.QOS]
    if sum(row['state'] == 'RUNNING' for row in gpu) > limits['max_running']:
        raise ConfigError('Current user running jobs exceed the verified limit')
    if len(gpu) >= limits['max_submitted']:
        raise ConfigError('No verified user submission capacity')
    return limits, gpu


def sbatch(root, name, entry, label, *, minutes=1440):
    command = ['sbatch', '--parsable', '--job-name=' + name,
        '--partition=cscc-gpu-p', '--qos=' + shared.QOS, '--nodes=1', '--ntasks=1',
        '--cpus-per-task=4', '--gres=gpu:1', '--mem=16G', f'--time={minutes}:00',
        '--exclude=' + shared.EXCLUDED_NODES, '--no-requeue', '--chdir=' + str(root / 'source'),
        '--output=' + str(root / 'logs' / f'{label}-%j.out'),
        '--error=' + str(root / 'logs' / f'{label}-%j.err'), '--wrap=' + shlex.join(entry)]
    response = shared.command(command).split(';')[0]
    if not response.isdigit():
        raise ConfigError(f'Uncertain sbatch response for {name}: {response}')
    return response


def diagnostic(root):
    verify(root)
    intent_path = root / 'diagnostics/gpu-intent.json'
    if intent_path.exists():
        previous = read(intent_path)
        if (root / 'diagnostics/gpu-gate.json').exists():
            print(json.dumps(previous))
            return
        if previous.get('status') != 'submitted' or not previous.get('job_id'):
            raise ConfigError('Uncertain prior C4 GPU gate submission')
        if controller_job(previous['job_id'], previous['name']) not in ('FAILED', 'CANCELLED', 'TIMEOUT'):
            raise ConfigError('Previous C4 GPU gate is live or has uncertain status')
        history_path = root / 'diagnostics/gpu-attempts.json'
        history = read(history_path) if history_path.exists() else []
        history.append(previous)
        save(history_path, history)
    admission(root)
    name = 'c4-selected-block-gpu-gate'
    if any(row['name'] == name for row in shared.queue_state()):
        raise ConfigError('C4 GPU gate already exists in Slurm')
    entry = ['env', 'PYTHONPATH=' + str(root / 'source'), PYTHON,
             str(root / 'source/scripts/diagnose_tinystories_c4_selected_block.py'),
             '--output', str(root / 'diagnostics/gpu-gate.json')]
    save(intent_path, dict(status='submitting', name=name, created_at=time.time()))
    job_id = sbatch(root, name, entry, 'gpu-gate', minutes=60)
    save(intent_path, dict(status='submitted', name=name, job_id=job_id, created_at=time.time()))
    print(job_id)


def check_gate(root):
    manifest = verify(root)
    intent = read(root / 'diagnostics/gpu-intent.json')
    gate = read(root / 'diagnostics/gpu-gate.json')
    if gate.get('status') != 'passed' or gate.get('job_id') != intent.get('job_id') or len(gate.get('grids', [])) != 2:
        raise ConfigError('Matching two-grid BF16 CUDA gate has not passed')
    for row, grid in zip(gate['grids'], GRIDS, strict=True):
        if (row.get('grid') != grid or row.get('config_sha256') != manifest['configs'][grid]
                or row.get('owner_call_counts', {}).get('O-common') != 8):
            raise ConfigError(f'Incomplete CUDA evidence for C4 {grid}')
    try:
        completed = controller_job(intent['job_id'], intent['name']) == 'COMPLETED'
    except subprocess.CalledProcessError:
        # slurmctld retains completed jobs only briefly. A recorded successful
        # admission tied to this exact gate and source survives that expiry.
        admission_path = root / 'diagnostics/gpu-gate-admission.json'
        if not admission_path.exists():
            raise ConfigError('C4 CUDA gate controller record expired without durable admission evidence')
        admitted = read(admission_path)
        submitted = read(root / 'diagnostics/production-intents.json')
        completed = (admitted.get('job_id') == intent['job_id']
            and admitted.get('gate_sha256') == digest(root / 'diagnostics/gpu-gate.json')
            and admitted.get('source_manifest_sha256') == digest(root / 'diagnostics/source-manifest.json')
            and admitted.get('production_job_ids') == [row['job_id'] for row in submitted['jobs'][:2]]
            and all(row['gate_job_id'] == intent['job_id'] for row in submitted['jobs'][:2]))
    if not completed:
        raise ConfigError('C4 CUDA gate lacks matching successful Slurm admission evidence')
    return gate


def controller_job(job_id, name):
    """Use the controller's retained job state when the accounting DB is down."""
    import re
    detail = shared.command(['scontrol', 'show', 'job', str(job_id)])
    if not re.search(r'\bJobId=' + re.escape(str(job_id)) + r'\b', detail) or not re.search(r'\bJobName=' + re.escape(name) + r'\b', detail):
        raise ConfigError('Slurm controller job identity differs from C4 intent')
    state = re.search(r'\bJobState=([A-Z_]+)', detail)
    exit_code = re.search(r'\bExitCode=([^\s]+)', detail)
    if state is None or exit_code is None:
        raise ConfigError('Slurm controller job record is incomplete')
    if state[1] == 'COMPLETED' and exit_code[1] != '0:0':
        raise ConfigError('Slurm controller reports nonzero exit for completed job')
    return state[1]


def submit(root):
    gate = check_gate(root)
    path = root / 'diagnostics/production-intents.json'
    with shared.lock(root / 'diagnostics/queue.lock'):
        intents = read(path) if path.exists() else {'jobs': []}
        for grid in GRIDS:
            arm = f'C4-{grid}'
            existing = [row for row in intents['jobs'] if row['arm'] == arm]
            if existing:
                continue
            limits, active = admission(root)
            if len(active) >= limits['max_submitted']:
                break
            name = 'c4-selected-' + grid
            if any(row['name'] == name for row in active):
                raise ConfigError(f'Existing Slurm job has C4 name {name}')
            config = resolve_run_config(path_config(root, grid), create_output_dirs=False)
            output = Path(config['run']['output_dir'])
            if output.exists():
                raise ConfigError(f'C4 output identity already exists without a recorded job: {output}')
            entry = [PYTHON, str(root / 'source/scripts/train_cuda_required.py'),
                     '--config', str(path_config(root, grid)), '--entry-evidence',
                     str(root / 'diagnostics' / f'cuda-entry-{grid}.json')]
            intent = dict(arm=arm, name=name, status='submitting', created_at=time.time(),
                          config_sha256=digest(path_config(root, grid)), gate_job_id=gate['job_id'])
            intents['jobs'].append(intent)
            save(path, intents)
            job_id = sbatch(root, name, entry, grid)
            intent.update(status='submitted', job_id=job_id)
            save(path, intents)
        print(json.dumps(intents, indent=2))


def validated_own_checkpoint(root, grid):
    """Inspect a stopped C4 run on CPU before admitting its continuation."""
    import torch
    from src.training import checkpointing as cp
    from src.training.modeling import build_model
    from src.training.steps import build_optimizer_and_scheduler
    source_config = resolve_run_config(path_config(root, grid), create_output_dirs=False)
    output = Path(source_config['run']['output_dir'])
    saved_config = read(output / 'config.json')
    if (saved_config['run']['run_id'] != source_config['run']['run_id']
            or saved_config['training']['resolved_mixed_precision'] != 'bf16'
            or saved_config['training']['block_update_policy'] != 'selected_block'
            or saved_config['model']['granularity_prefixes'] != source_config['model']['granularity_prefixes']
            or saved_config['training']['gradient_clipping'] != source_config['training']['gradient_clipping']):
        raise ConfigError('C4 stopped run has the wrong identity or precision')
    config = saved_config
    checkpoint = output / 'checkpoints/latest.pt'
    if not checkpoint.is_file():
        raise ConfigError(f'C4 occupied output has no durable own checkpoint: {output}')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = build_model(config)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config['training'])
    cp._validate_optimizer_resume_contract(payload, config=config, optimizer=optimizer, scheduler=scheduler)
    cp._validate_model_state_before_load(model, payload['model_state_dict'])
    cp._validate_reproducibility_payload(payload, config=config, checkpoint_path=checkpoint)
    cp._validate_rng_locally(payload['reproducibility']['rng_state'], inspect_only=True)
    collection = optimizer.validate_state_dict(payload['optimizer_state_collection'])
    clock = scheduler.validate_state_dict(payload['scheduler_state_dict'])
    step = payload['step']
    if (not 0 < step <= config['training']['max_steps']
            or payload['tokens_seen'] != step * config['training']['expected_tokens_per_step']
            or collection['total_successful_updates'] != clock['position'] or clock['position'] != step):
        raise ConfigError('C4 stopped checkpoint is not a complete committed update')
    return dict(path=str(checkpoint), step=step, sha256=digest(checkpoint))


def resume(root):
    check_gate(root)
    path = root / 'diagnostics/production-intents.json'
    with shared.lock(root / 'diagnostics/queue.lock'):
        intents = read(path)
        for grid in GRIDS:
            arm = f'C4-{grid}'
            records = [record for record in intents['jobs'] if record['arm'] == arm]
            if not records:
                raise ConfigError(f'C4 {grid} was never submitted')
            previous = records[-1]
            state = controller_job(previous['job_id'], previous['name'])
            if state in ('RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING'):
                continue
            if state == 'COMPLETED':
                summary = read(Path(resolve_run_config(path_config(root, grid), create_output_dirs=False)['run']['output_dir']) / 'run_summary.json')
                if summary.get('status') != 'completed' or summary.get('steps_completed') != 348_528:
                    raise ConfigError(f'C4 {grid} job completed without a valid terminal summary')
                previous['status'] = 'completed'
                save(path, intents)
                continue
            if state not in ('FAILED', 'TIMEOUT', 'CANCELLED'):
                raise ConfigError(f'C4 {grid} has uncertain Slurm state: {state}')
            checkpoint = validated_own_checkpoint(root, grid)
            limits, active = admission(root)
            if len(active) >= limits['max_submitted']:
                continue
            if any(row['name'].startswith('c4-selected-' + grid) for row in active):
                raise ConfigError(f'C4 {grid} already has a live job')
            attempt = len(records) + 1
            name = f'c4-selected-{grid}-a{attempt}'
            entry = [PYTHON, str(root / 'source/scripts/train_cuda_required.py'),
                     '--config', str(path_config(root, grid)), '--entry-evidence',
                     str(root / 'diagnostics' / f'cuda-entry-{grid}-a{attempt}.json')]
            record = dict(arm=arm, name=name, status='submitting', attempt=attempt,
                          created_at=time.time(), continuation=checkpoint,
                          config_sha256=digest(path_config(root, grid)), gate_job_id=read(root / 'diagnostics/gpu-intent.json')['job_id'])
            intents['jobs'].append(record)
            save(path, intents)
            job_id = sbatch(root, name, entry, f'{grid}-a{attempt}')
            record.update(status='submitted', job_id=job_id)
            previous['status'] = state.lower()
            save(path, intents)
        print(json.dumps(intents, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'refresh-diagnostic', 'diagnostic', 'submit', 'resume', 'status'))
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == 'prepare':
        prepare(root)
    elif args.mode == 'refresh-diagnostic':
        refresh_diagnostic(root)
    elif args.mode == 'diagnostic':
        diagnostic(root)
    elif args.mode == 'submit':
        submit(root)
    elif args.mode == 'resume':
        resume(root)
    else:
        print(json.dumps(dict(gpu_intent=read(root / 'diagnostics/gpu-intent.json') if (root / 'diagnostics/gpu-intent.json').exists() else None,
                              production=read(root / 'diagnostics/production-intents.json') if (root / 'diagnostics/production-intents.json').exists() else None,
                              queue=shared.queue_state()), indent=2))


if __name__ == '__main__':
    main()
