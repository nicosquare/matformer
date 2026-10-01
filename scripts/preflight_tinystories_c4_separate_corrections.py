#!/usr/bin/env python3
"""Bind CPU regressions, unchanged controls, streams and read-only terminals."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from src.training.data import build_packed_mmap_dataloaders
from src.utils.config import resolve_run_config
from src.utils.reproducibility import seed_for, stable_hash
from scripts import report_tinystories_c4_separate_corrections as report


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, data):
    Path(p).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def audit(root):
    controls, actions, references = {}, {}, {}
    for grid, (reference, widths) in report.GRIDS.items():
        _, _, _, _, refs = report.collect(report.REFERENCE_ROOT / report.UNCORRECTED, grid, reference, widths, campaign=report.UNCORRECTED)
        from src.evaluation.optimizer_ownership import inspect_selected_terminals
        selected = inspect_selected_terminals(report.REFERENCE_ROOT / reference / 'campaign/campaign_manifest.json', [f'ST-{w}' for w in widths])
        references[grid] = dict(c4=refs, standalones=selected)
        old_path = ROOT / f'configs/controlled_exps/tinystories_instruct_c4_selected_block_{grid}.yaml'
        original = yaml.safe_load(old_path.read_text())
        saved = report.read(report.REFERENCE_ROOT / report.UNCORRECTED / 'runs' / f'{report.UNCORRECTED}-C4-{grid}-s42' / 'config.json')
        for kind in ('GMC', 'LMC-only'):
            arm = f'{grid}-{kind}'
            path = ROOT / f'configs/controlled_exps/tinystories_instruct_c4_separate_{arm}.yaml'
            raw = yaml.safe_load(path.read_text())
            normalized = json.loads(json.dumps(raw))
            for field in ('campaign_id', 'arm_id', 'run_id', 'output_dir'):
                normalized['run'][field] = original['run'][field]
            normalized['run'].pop('c4_separate_corrections_protocol')
            normalized['model']['correction_mode'] = 'none'
            normalized['training'].pop('c4_correction')
            assert normalized == original, f'Unexpected scientific control change: {arm}'
            config = resolve_run_config(path, create_output_dirs=False)
            assert config['training']['max_steps'] == 348528 and config['training']['token_budget'] == 2855141376
            assert config['training']['resolved_learning_rate'] == .008 and config['training']['resolved_warmup_steps'] == 64
            streams = {name: [seed_for(config, name), seed_for(saved, name)] for name in ('model_initialization', 'granularity_selection', 'python_training', 'numpy_training', 'torch_training')}
            assert all(a == b for a, b in streams.values())
            assert config['dataset']['optimizer_iteration'] == saved['dataset']['optimizer_iteration']
            loader, _, _, _ = build_packed_mmap_dataloaders(config, torch.device('cpu'))
            old_loader, _, _, _ = build_packed_mmap_dataloaders(saved, torch.device('cpu'))
            new_iter, old_iter = iter(loader.batch_sampler), iter(old_loader.batch_sampler)
            hashes = []
            for _ in range(64):
                a, b = next(new_iter), next(old_iter)
                assert a == b
                hashes.append(stable_hash(a))
            generator = random.Random(streams['granularity_selection'][0])
            trace = [generator.randrange(4) for _ in range(348528)]
            controls[arm] = dict(new_config_sha256=sha(path), control_config_sha256=sha(old_path), only_changes=['fresh identity', 'versioned isolated correction'])
            actions[arm] = dict(seed_streams=streams, first_64_batch_hashes=hashes, full_horizon_width_position_sha256=stable_hash(trace), width_position_counts={i:trace.count(i) for i in range(4)}, checked_batch_updates=64, full_action_updates=348528, sampler_contract_hash=stable_hash(config['dataset']['optimizer_iteration']))
    save(root / 'diagnostics/control-difference-audit.json', controls)
    save(root / 'diagnostics/action-data-audit.json', actions)
    save(root / 'diagnostics/read-only-reference-audit.json', references)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    workspace = root / 'diagnostics/pytest-workspace'; workspace.mkdir(exist_ok=True)
    for p in ROOT.iterdir():
        target = workspace / p.name
        if not target.exists():
            target.symlink_to(p, target_is_directory=p.is_dir())
    suites = ['test_c4_separate_reporting_queue.py', 'test_c4_separate_corrections.py', 'test_c4_selected_block.py', 'test_config.py', 'test_reproducibility.py', 'test_optimizer_ownership.py', 'test_optimizer_ownership_resume.py', 'test_optimizer_ownership_corrections.py', 'test_optimizer_ownership_campaign.py', 'test_optimizer_ownership_reporting.py', 'test_per_granularity_optimizer_resume.py', 'test_optimizer_ownership_correction_queue.py']
    junit = root / 'diagnostics/cpu-pytest.xml'
    cmd = [sys.executable, '-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider', '--override-ini=addopts=', '--junitxml='+str(junit)] + ['tests/'+s for s in suites]
    started = time.time()
    with (root / 'diagnostics/cpu-pytest.log').open('w') as log:
        result = subprocess.run(cmd, cwd=workspace, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONPATH':str(ROOT), 'OMP_NUM_THREADS':'1', 'CUDA_VISIBLE_DEVICES':'', 'PYTHONDONTWRITEBYTECODE':'1'})
    counts = {k:sum(int(s.get(k,0)) for s in ET.parse(junit).getroot().iter('testsuite')) for k in ('tests','failures','errors','skipped')}
    evidence = dict(status='passed' if result.returncode == 0 else 'failed', command=cmd, counts=counts, started_at=started, finished_at=time.time(), source_manifest_sha256=sha(root / 'diagnostics/source-manifest.json'))
    if result.returncode:
        save(root / 'diagnostics/cpu-gate.json', evidence)
        raise RuntimeError('CPU readiness failed; inspect cpu-pytest.log')
    audit(root)
    save(root / 'diagnostics/cpu-gate.json', evidence)
    print(json.dumps(evidence))


if __name__ == '__main__':
    main()
