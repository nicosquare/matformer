#!/usr/bin/env python3
"""Publish C4 versus standalone ordinary-validation endpoints after both runs finish."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matformer-c4-matplotlib')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.metrics import write_json_artifact


CAMPAIGN = 'tinystories-optimizer-ownership-c4-selected-block-v1'
ROOT = Path('/nfs-stor/ivo.navarrete/results/elasticnn') / CAMPAIGN
REFERENCE_ROOT = Path('/nfs-stor/ivo.navarrete/results/elasticnn')
GRIDS = {
    'linear': ('optimizer-ownership-v1', ('g250', 'g500', 'g750', 'g1000')),
    'geometric': ('optimizer-ownership-matformer-widths-v1', ('g125', 'g250', 'g500', 'g1000')),
}
MANIFEST = '0c1beea552f54941e397d2442de736b1586e0292f6b1271b62d27ad782627856'
AGGREGATION = 'target_token_weighted_causal_shift_float64'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(newline='') as stream:
        yield from csv.DictReader(stream)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def endpoint_from_metric(row, width, count):
    require(row['split'] == 'validation' and row['granularity'] == width, 'Wrong C4 validation width')
    require(row['validation_manifest_hash'] == MANIFEST and row['validation_loss_aggregation'] == AGGREGATION,
            'C4 validation role or manifest changed')
    loss, perplexity = float(row['loss']), float(row['perplexity'])
    require(math.isfinite(loss) and math.isfinite(perplexity) and abs(math.exp(loss) - perplexity) < 1e-5,
            'C4 validation metric is invalid')
    return dict(width=width, non_embedding_parameters=count, loss=loss, perplexity=perplexity,
                update=int(row['step']), evaluation_role='ordinary_validation', validation_manifest_hash=MANIFEST)


def collect(root, grid, reference_campaign, widths):
    run_id = f'{CAMPAIGN}-C4-{grid}-s42'
    run = root / 'runs' / run_id
    config = read(run / 'config.json')
    summary = read(run / 'run_summary.json')
    require(config['run']['run_id'] == summary['run_id'] == run_id, 'C4 run identity mismatch')
    require(config['run']['c4_selected_block_protocol'] == 1 and config['training']['block_update_policy'] == 'selected_block',
            'C4 selected-block policy mismatch')
    require(config['training']['gradient_clipping']['mode'] == 'global' and config['training']['gradient_clipping']['max_norm'] == 1.,
            'C4 joint global clip mismatch')
    require(summary['status'] == 'completed' and summary['steps_completed'] == 348_528
            and summary['tokens_seen'] == 2_855_141_376 and summary['optimizer_accounting_reconciled'] is True,
            'C4 terminal is incomplete or unreconciled')
    require(summary['validation_manifest_hash'] == MANIFEST and config['model']['granularities'] == list(widths),
            'C4 ordinary validation or width grid mismatch')
    counts = summary['optimizer_exposure_counts']
    calls = summary['optimizer_successful_update_counts']
    expected = {**{f'O-{letter}': counts[width] for letter, width in zip('ABCD', widths, strict=True)},
                'O-common': 348_528}
    require(calls == expected and sum(counts.values()) == 348_528,
            'C4 selected owner calls do not match width draws')
    checkpoint = Path(summary['latest_checkpoint_path'])
    require(checkpoint.is_file(), 'C4 terminal checkpoint missing')
    checkpoint_sha = sha(checkpoint)
    import torch
    from src.training import checkpointing as cp
    from src.training.modeling import build_model
    from src.training.steps import build_optimizer_and_scheduler
    model = build_model(config)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config['training'])
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require(cp._validate_optimizer_resume_contract(payload, config=config, optimizer=optimizer, scheduler=scheduler) == 'per_ffn_block',
            'C4 terminal optimizer scope or checkpoint contract mismatch')
    cp._validate_model_state_before_load(model, payload['model_state_dict'])
    cp._validate_reproducibility_payload(payload, config=config, checkpoint_path=checkpoint)
    cp._validate_rng_locally(payload['reproducibility']['rng_state'], inspect_only=True)
    collection = optimizer.validate_state_dict(payload['optimizer_state_collection'])
    clock_state = scheduler.validate_state_dict(payload['scheduler_state_dict'])
    require(payload['step'] == 348_528 and payload['tokens_seen'] == 2_855_141_376,
            'C4 terminal checkpoint budget mismatch')
    require(clock_state['position'] == 348_528 and collection['width_selection_counts'] == counts
            and collection['successful_update_counts'] == expected,
            'C4 restored scheduler or owner counters mismatch')
    evidence = dict(run_id=run_id, config_sha256=sha(run / 'config.json'),
                    summary_sha256=sha(run / 'run_summary.json'), checkpoint_sha256=checkpoint_sha,
                    width_selection_counts=counts, owner_call_counts=calls,
                    metrics_sha256=sha(run / 'metrics.csv'))

    terminal_rows, progress = {}, []
    for row in rows(run / 'metrics.csv'):
        if row['split'] != 'validation':
            continue
        step = int(row['step'])
        width = row['granularity']
        if width not in widths:
            continue
        require(row['validation_manifest_hash'] == MANIFEST, 'C4 validation history manifest changed')
        progress.append((step, width, float(row['loss'])))
        if step == 348_528:
            require(width not in terminal_rows, 'Duplicate C4 terminal validation width')
            terminal_rows[width] = row
    require(set(terminal_rows) == set(widths), 'Missing C4 terminal ordinary validation endpoints')

    c4, standalones, differences = [], [], []
    for width in widths:
        count = summary['parameter_counts_by_granularity'][width]['non_embedding_parameters']
        c4_row = endpoint_from_metric(terminal_rows[width], width, count)
        reference_path = REFERENCE_ROOT / reference_campaign / 'runs' / f'ST-{width}' / 'terminal_validation_results.json'
        reference = read(reference_path)
        require(reference['actual_updates'] == 87_132 and reference['evaluation_role'] == 'ordinary_validation'
                and reference['validation_manifest_hash'] == MANIFEST,
                f'Standalone {grid}/{width} is not a matching ordinary-validation terminal')
        require(len(reference['endpoints']) == 1, 'Standalone terminal endpoint count changed')
        baseline = reference['endpoints'][0]
        require(baseline['width'] == width and baseline['non_embedding_parameters'] == count
                and baseline['validation_manifest_hash'] == MANIFEST,
                f'Standalone {grid}/{width} width, count, or manifest mismatch')
        standalone_row = dict(width=width, non_embedding_parameters=count,
                              loss=baseline['loss'], perplexity=baseline['perplexity'],
                              update=87_132, evaluation_role='ordinary_validation',
                              validation_manifest_hash=MANIFEST)
        c4.append(c4_row)
        standalones.append(standalone_row)
        differences.append(dict(grid=grid, width=width, non_embedding_parameters=count,
                                c4_minus_standalone_loss=c4_row['loss'] - standalone_row['loss'],
                                c4_minus_standalone_perplexity=c4_row['perplexity'] - standalone_row['perplexity']))
        evidence[f'standalone_{width}_terminal_sha256'] = sha(reference_path)
    return c4, standalones, differences, progress, evidence


def collect_s1(reference_campaign, widths, c4):
    run = REFERENCE_ROOT / reference_campaign / 'runs/S1'
    terminal = read(run / 'terminal_validation_results.json')
    config = read(run / 'config.json')
    require(terminal['actual_updates'] == 348_528 and terminal['evaluation_role'] == 'ordinary_validation'
            and terminal['validation_manifest_hash'] == MANIFEST
            and config['training']['resolved_warmup_steps'] == 64, 'S1 reference protocol mismatch')
    require(sha(terminal['checkpoint_path']) == terminal['checkpoint_sha256'], 'S1 terminal checkpoint changed')
    by_width = {row['width']: row for row in terminal['endpoints']}
    require(set(by_width) == set(widths), 'S1 endpoint widths mismatch')
    endpoints = []
    for baseline in c4:
        row = by_width[baseline['width']]
        require(row['non_embedding_parameters'] == baseline['non_embedding_parameters']
                and row['validation_manifest_hash'] == MANIFEST
                and row['evaluation_role'] == 'ordinary_validation'
                and row['validation_loss_aggregation'] == AGGREGATION, 'S1 endpoint identity mismatch')
        endpoints.append({key: row[key] for key in ('width', 'non_embedding_parameters', 'loss',
                         'perplexity', 'evaluation_role', 'validation_manifest_hash')} | {'update': 348_528})
    progress = []
    for row in rows(run / 'metrics.csv'):
        if row['split'] == 'validation' and row['granularity'] in widths:
            require(row['validation_manifest_hash'] == MANIFEST, 'S1 history manifest mismatch')
            progress.append((int(row['step']), row['granularity'], float(row['loss'])))
    return endpoints, progress, dict(terminal_sha256=sha(run / 'terminal_validation_results.json'),
        checkpoint_sha256=terminal['checkpoint_sha256'], metrics_sha256=sha(run / 'metrics.csv'),
        config_sha256=sha(run / 'config.json'), warmup_steps=64, run_id=terminal['run_id'])


def plot_endpoints(out, grid, c4, standalones, field, s1=None):
    fig, axis = plt.subplots(figsize=(7, 4.5))
    x = [row['non_embedding_parameters'] for row in c4]
    axis.plot(x, [row[field] for row in c4], 'o-', label='C4 selected block')
    if s1 is not None:
        axis.plot(x, [row[field] for row in s1], 's--', color='tab:orange', label='S1 (64-update warmup)')
    axis.scatter(x, [row[field] for row in standalones], marker='x', s=64, color='black', label='Standalone terminal')
    for row in c4:
        axis.annotate(row['width'], (row['non_embedding_parameters'], row[field]), xytext=(3, 5), textcoords='offset points')
    axis.set(xlabel='Active non-embedding parameters', ylabel=f'Ordinary-validation {field}',
             title=f'{grid.title()}: C4, S1 and standalones' if s1 is not None else f'C4 {grid} versus matching standalones')
    axis.margins(x=0.10, y=0.12)
    axis.legend(); fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'{grid}_{field}_vs_parameters.{suffix}', dpi=180)
    plt.close(fig)


def plot_progress(out, grid, standalones, progress, s1_progress=None):
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for width in GRIDS[grid][1]:
        trajectory = sorted((step, loss) for step, label, loss in progress if label == width)
        axis.plot([point[0] for point in trajectory], [point[1] for point in trajectory], label=f'C4 {width}')
        color = axis.lines[-1].get_color()
        if s1_progress is not None:
            s1 = sorted((step, loss) for step, label, loss in s1_progress if label == width)
            axis.plot([p[0] for p in s1], [p[1] for p in s1], '--', color=color, label=f'S1 {width}')
        baseline = next(row for row in standalones if row['width'] == width)
        axis.scatter([87_132], [baseline['loss']], marker='x', s=55, color=color)
    axis.set(xlabel='Global optimizer updates', ylabel='Ordinary-validation loss',
             title=f'{grid.title()}: C4 solid, S1 dashed; standalone points at 87,132' if s1_progress is not None else f'C4 {grid} validation progress; standalone terminal points at 87,132')
    axis.legend(ncol=2); fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'{grid}_validation_loss_vs_updates.{suffix}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--include-s1', action='store_true', help='Add the matching original S1 runs with 64-update warmup')
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / ('report-with-s1' if args.include_s1 else 'report')
    output.mkdir(parents=True, exist_ok=True)
    endpoints, differences, sources = [], [], {}
    s1_differences = []
    for grid, (reference, widths) in GRIDS.items():
        c4, standalones, delta, progress, evidence = collect(root, grid, reference, widths)
        endpoints.extend([dict(grid=grid, arm='C4', **row) for row in c4])
        endpoints.extend([dict(grid=grid, arm='standalone', **row) for row in standalones])
        differences.extend(delta)
        sources[grid] = evidence
        s1, s1_progress = None, None
        if args.include_s1:
            s1, s1_progress, s1_evidence = collect_s1(reference, widths, c4)
            sources[grid]['s1'] = s1_evidence
            endpoints.extend(dict(grid=grid, arm='S1', **row) for row in s1)
            s1_differences.extend(dict(grid=grid, width=a['width'],
                c4_minus_s1_loss=a['loss'] - b['loss'],
                c4_minus_s1_perplexity=a['perplexity'] - b['perplexity'])
                for a, b in zip(c4, s1, strict=True))
        for field in ('loss', 'perplexity'):
            plot_endpoints(output, grid, c4, standalones, field, s1=s1)
        plot_progress(output, grid, standalones, progress, s1_progress=s1_progress)
    write_json_artifact(output / 'endpoints.json', endpoints)
    write_json_artifact(output / 'differences.json', differences)
    if s1_differences:
        write_json_artifact(output / 'c4_minus_s1.json', s1_differences)
        with (output / 'c4_minus_s1.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(s1_differences[0]))
            writer.writeheader(); writer.writerows(s1_differences)
    for name, records in (('endpoints', endpoints), ('differences', differences)):
        with (output / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    write_json_artifact(output / 'sources.json', dict(report_script_sha256=sha(__file__), grids=sources,
        source_manifest_sha256=sha(root / 'diagnostics/source-manifest.json'),
        gpu_gate_sha256=sha(root / 'diagnostics/gpu-gate.json'),
        interpretation='One seed 42; expected FFN block owner calls follow width draws, while training-example coverage is width-dependent.'))
    print(json.dumps(dict(status='complete', endpoints=len(endpoints), differences=len(differences), output=str(output))))


if __name__ == '__main__':
    main()
