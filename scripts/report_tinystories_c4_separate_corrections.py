#!/usr/bin/env python3
"""Publish completed C4 grid endpoints with standalone and optional original S1 references."""

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


CAMPAIGN = 'tinystories-optimizer-ownership-c4-separate-corrections-v1'
UNCORRECTED = 'tinystories-optimizer-ownership-c4-selected-block-v1'
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


def collect(root, grid, reference_campaign, widths, *, kind=None, campaign=CAMPAIGN):
    arm = f"C4-{grid}" + (f"-{kind}" if kind else "")
    run_id = f"{campaign}-{arm}-s42"
    run = root / 'runs' / run_id
    config = read(run / 'config.json')
    summary = read(run / 'run_summary.json')
    require(config['training']['resolved_mixed_precision'] == 'bf16', 'C4 terminal lacks BF16 execution')
    if kind:
        from src.utils.config import resolve_run_config
        tested = resolve_run_config(root / 'source/configs/controlled_exps' / f'tinystories_instruct_c4_separate_{grid}-{kind}.yaml', create_output_dirs=False)
        require(config['training']['c4_correction'] == tested['training']['c4_correction'] and config['model']['membership_correction'] == (kind == 'GMC'), 'Correction metadata mismatch')
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
        require(sha(reference['checkpoint_path']) == reference['checkpoint_sha256'], 'Standalone terminal checkpoint changed')
        require(reference['validation_loss_aggregation'] == AGGREGATION, 'Standalone aggregation changed')
        require(len(reference['endpoints']) == 1, 'Standalone terminal endpoint count changed')
        baseline = reference['endpoints'][0]
        require(baseline['width'] == width and baseline['non_embedding_parameters'] == count
                and baseline['validation_manifest_hash'] == MANIFEST
                and baseline['validation_loss_aggregation'] == AGGREGATION
                and baseline['evaluation_role'] == 'ordinary_validation',
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



def publish_tables(output, name, records):
    write_json_artifact(output / f'{name}.json', records)
    with (output / f'{name}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def plot(output, grid, curves, standalones, histories):
    labels = {'GMC': 'C4 GMC only', 'LMC-only': 'C4 LMC only, no GMC', 'uncorrected': 'Uncorrected C4', 'S1': 'S1 (64-update warmup)'}
    for field in ('loss', 'perplexity'):
        fig, axis = plt.subplots(figsize=(8, 5))
        for kind, points in curves.items():
            axis.plot([r['non_embedding_parameters'] for r in points], [r[field] for r in points], 's--' if kind == 'S1' else 'o-', label=labels[kind])
        axis.scatter([r['non_embedding_parameters'] for r in standalones], [r[field] for r in standalones], marker='x', color='black', s=60, zorder=5, label='Standalone terminal')
        for row in standalones:
            axis.annotate(row['width'], (row['non_embedding_parameters'], row[field]), xytext=(4, -12), textcoords='offset points', fontsize=9)
        axis.set(xlabel='Active non-embedding parameters', ylabel=f'Ordinary-validation {field}', title=f'C4 {grid}, seed 42')
        axis.margins(x=0.08, y=0.12)
        axis.legend(); fig.tight_layout()
        for suffix in ('png', 'pdf'):
            fig.savefig(output / f'{grid}_{field}_vs_parameters.{suffix}', dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, baseline in zip(axes.flat, standalones, strict=True):
        for kind, history in histories.items():
            trajectory = sorted((step, loss) for step, width, loss in history if width == baseline['width'])
            axis.plot([s for s, _ in trajectory], [v for _, v in trajectory], '--' if kind == 'S1' else '-', label=labels[kind])
        axis.scatter([87132], [baseline['loss']], marker='x', color='black', s=60, zorder=5, label='Standalone terminal')
        axis.set(title=baseline['width'], xlabel='Global updates', ylabel='Ordinary-validation loss')
        axis.legend(fontsize=8)
    fig.suptitle(f'C4 {grid}, seed 42; standalones only at 87,132 updates'); fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'{grid}_validation_loss_vs_updates.{suffix}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--grid', choices=('linear', 'geometric', 'all'), default='all')
    parser.add_argument('--include-s1', action='store_true', help='Include matching original S1 with 64-update warmup')
    args = parser.parse_args()
    root = args.root.resolve()
    # Collect and validate all evidence before publishing comparison outputs.
    endpoints, differences, sources, plot_inputs = [], [], {}, []
    grids = GRIDS if args.grid == 'all' else {args.grid: GRIDS[args.grid]}
    for grid, (reference, widths) in grids.items():
        original, standalones, _, progress, evidence = collect(REFERENCE_ROOT / UNCORRECTED, grid, reference, widths, campaign=UNCORRECTED)
        curves, histories = {'uncorrected': original}, {'uncorrected': progress}
        sources[grid] = {'uncorrected': evidence}
        endpoints.extend(dict(grid=grid, arm='uncorrected-C4', **r) for r in original)
        endpoints.extend(dict(grid=grid, arm='standalone', **r) for r in standalones)
        references = [('uncorrected-C4', original), ('standalone', standalones)]
        if args.include_s1:
            from scripts.report_tinystories_c4_selected_block import collect_s1
            s1, s1_history, s1_evidence = collect_s1(reference, widths, original)
            curves['S1'], histories['S1'] = s1, s1_history
            sources[grid]['S1'] = s1_evidence
            endpoints.extend(dict(grid=grid, arm='S1', **r) for r in s1)
            references.append(('S1', s1))
        for kind in ('GMC', 'LMC-only'):
            corrected, matching, _, history, evidence = collect(root, grid, reference, widths, kind=kind)
            require(matching == standalones, 'Standalone reference differs across corrections')
            curves[kind], histories[kind] = corrected, history
            sources[grid][kind] = evidence
            endpoints.extend(dict(grid=grid, arm=f'C4-{grid}-{kind}', **r) for r in corrected)
            for target, baseline in references:
                for a, b in zip(corrected, baseline, strict=True):
                    require(a['width'] == b['width'] and a['non_embedding_parameters'] == b['non_embedding_parameters'], 'Comparison identity differs')
                    differences.append(dict(grid=grid, arm=f'C4-{grid}-{kind}', width=a['width'], reference=target,
                        loss_difference=a['loss'] - b['loss'], perplexity_difference=a['perplexity'] - b['perplexity'],
                        reaches_or_improves_reference=a['loss'] <= b['loss']))
        plot_inputs.append((grid, curves, standalones, histories))
    expected_endpoints = len(grids) * (20 if args.include_s1 else 16)
    expected_differences = len(grids) * (24 if args.include_s1 else 16)
    require(len(endpoints) == expected_endpoints and len(differences) == expected_differences, 'Endpoint or difference count differs')
    name = 'report' + (f'-{args.grid}' if args.grid != 'all' else '') + ('-with-s1' if args.include_s1 else '')
    output = root / name; output.mkdir(exist_ok=True)
    publish_tables(output, 'endpoints', endpoints)
    publish_tables(output, 'differences', differences)
    for target in ('uncorrected-C4', 'standalone', *(['S1'] if args.include_s1 else [])):
        publish_tables(output, f'corrected_minus_{target}', [r for r in differences if r['reference'] == target])
    for inputs in plot_inputs:
        plot(output, *inputs)
    write_json_artifact(output / 'sources.json', dict(status='complete', selected_grids=list(grids), includes_original_s1=args.include_s1, grids=sources,
        source_manifest_sha256=sha(root / 'diagnostics/source-manifest.json'), gpu_gate_sha256=sha(root / 'diagnostics/gpu-gate.json'),
        report_script_sha256=sha(__file__), interpretation='Descriptive seed-42 results. Equal expected C4 block update counts do not imply equal forward/example coverage. GMC and LMC test configured forward membership factors; LMC only has no GMC and scales effective block LR, while the global clock remains nominal.'))
    print(json.dumps(dict(status='complete', endpoints=len(endpoints), differences=len(differences), output=str(output))))


if __name__ == '__main__':
    main()
