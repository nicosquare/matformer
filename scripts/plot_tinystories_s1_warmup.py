#!/usr/bin/env python3
"""Plot saved S1 warmup results without updating campaign or phase status."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import optimizer_ownership as oo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=Path('/nfs-stor/ivo.navarrete/results/elasticnn'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/tinystories-s1-warmup-comparison'))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    roots = {g: args.results_root / name for g, name in (
        ('new', 'optimizer-ownership-s1-warmup-v1'),
        ('linear', 'optimizer-ownership-v1'),
        ('geometric', 'optimizer-ownership-matformer-widths-v1'))}
    manifests = {g: oo._read_preflight_manifest(p / 'campaign/campaign_manifest.json') for g, p in roots.items()}
    rows, early, notes, sources, curves = [], [], [], [], {}
    for grid in ('linear', 'geometric'):
        selected = [(grid, r) for r in manifests[grid]['runs'] if r['arm_id'] == 'S1' or r['arm_id'].startswith('ST-')]
        selected += [('new', r) for r in manifests['new']['runs'] if r['grid_id'] == grid]
        for origin, run in selected:
            root = roots[origin] / 'runs' / run['arm_id']
            print(f'Reading {grid}: {run["arm_id"]}', flush=True)
            terminal = root / 'terminal_validation_results.json'
            sidecar = json.loads(terminal.read_text())
            endpoints = oo._terminal_endpoints(sidecar, run, allow_partial=False)
            checkpoint = oo._source_record(sidecar['checkpoint_path'])
            if checkpoint['sha256'] != sidecar['checkpoint_sha256']:
                raise ValueError(f'Checkpoint changed: {root}')
            sources.extend([oo._source_record(terminal), checkpoint])
            role = 'standalone' if run['source_width'] else ('s1_warmup256' if origin == 'new' else 's1_warmup64')
            widths = {w['label']: w for w in oo.campaign_widths(manifests[origin]['schema_version'], run['arm_id'])}
            for endpoint in endpoints:
                rows.append(dict(endpoint, grid_id=grid, grid_label=grid.title(), role=role,
                    ffn_dimension=widths[endpoint['width']]['active_ffn_dimension'],
                    endpoint_identity=[grid, run['campaign_id'], run['run_id'], widths[endpoint['width']]['active_ffn_dimension']]))
            if role == 'standalone':
                continue
            observed, note = oo.extract_warmup_early(run, root)
            early.extend(observed)
            notes.append(note)
            frames = []
            for chunk in pd.read_csv(root / 'metrics.csv', usecols=['split', 'step', 'granularity', 'loss'], chunksize=50000):
                frames.append(chunk.loc[chunk['split'] == 'validation'])
            frame = pd.concat(frames, ignore_index=True)
            for endpoint in endpoints:
                values = frame.loc[frame['granularity'] == endpoint['width'], ['step', 'loss']].sort_values('step')
                x, y = values['step'].to_numpy(), values['loss'].to_numpy()
                if not len(x) or not np.isfinite(y).all() or not np.all(np.diff(x) > 0) or x[-1] != run['assigned_updates'] or not np.isclose(y[-1], endpoint['loss'], atol=1e-12, rtol=0):
                    raise ValueError(f'Invalid validation trajectory: {root}/{endpoint["width"]}')
                curves[grid, role, widths[endpoint['width']]['active_ffn_dimension']] = (x, y)
    oo.pair_warmup_endpoints(rows)
    oo._check_sources(sources)
    (output / 'endpoints.json').write_text(json.dumps(rows, indent=2) + '\n')
    (output / 'early_metrics.json').write_text(json.dumps(early, indent=2) + '\n')
    (output / 'plot_sources.json').write_text(json.dumps(dict(sources=sources, early_notes=notes), indent=2) + '\n')

    def save(figure, name):
        for suffix in ('png', 'pdf'):
            figure.savefig(output / f'{name}.{suffix}', dpi=180, bbox_inches='tight')
        plt.close(figure)

    for grid in ('linear', 'geometric'):
        selected = [r for r in rows if r['grid_id'] == grid]
        for metric in ('loss', 'perplexity'):
            save(oo.warmup_endpoint_figure(selected, metric=metric), f'{grid}_{metric}_vs_parameters')
        for metric, label in [('learning_rate', 'lr'), ('loss', 'loss')]:
            save(oo.warmup_early_figure(early, grid=grid, metric=metric, end_step=1024), f'{grid}_early_{label}')
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for ax, dim in zip(axes.flat, sorted({r['ffn_dimension'] for r in selected})):
            for role, label, color in [('s1_warmup64', '64-update warmup', '#0072B2'), ('s1_warmup256', '256-update warmup', '#D55E00')]:
                x, y = curves[grid, role, dim]
                ax.plot(x, y, label=label, color=color, linewidth=1)
            st = next(r for r in selected if r['role'] == 'standalone' and r['ffn_dimension'] == dim)
            ax.axhline(st['loss'], color='#555555', linestyle=':', label='Standalone terminal (1 epoch)')
            ax.set(title=f'FFN {dim}', xlabel='Optimizer update', ylabel='Ordinary-validation loss', xlim=(0, 348528))
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle(f'{grid.title()} · TinyStories-Instruct · seed 42\nRaw validation curves · S1: 4 epochs; standalone reference: 1 epoch')
        fig.tight_layout()
        save(fig, f'{grid}_validation_loss_vs_updates')
    print(f'Saved 10 plots in PNG and PDF: {output}', flush=True)


if __name__ == '__main__':
    main()
