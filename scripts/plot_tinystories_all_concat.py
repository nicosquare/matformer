#!/usr/bin/env python3
"""Combine validated C1/C2/C3 result tables and unsmoothed validation curves."""
import argparse
import hashlib
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

ROOT = Path('/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/campaigns/concat-gmc-lmc-v1')
FAMILIES = ('C1', 'C2', 'C3')
ARMS = tuple(arm for family in FAMILIES for arm in (family, family + '-GMC', family + '-LMC'))
COLORS = dict(zip(FAMILIES, ('#009E73', '#0072B2', '#CC79A7')))
STYLES = {'none': '-', 'GMC': '--', 'LMC': ':'}
MARKERS = {'none': 'o', 'GMC': 's', 'LMC': 'D'}
STEPS = 348528


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', action='append', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / 'reports') or output.exists():
        parser.error('Output must be a fresh directory under campaign reports')
    sources, endpoints, progress, families, limitations = {}, {}, [], set(), {}
    def add_source(source):
        previous = sources.setdefault(source['path'], source)
        assert previous == source, 'Conflicting source hashes'
    for directory in args.report_dir:
        report_path = directory / 'report.json'
        report = read(report_path)
        validation_path = directory / 'output-validation.json'
        validation = read(validation_path)
        family_set = {arm.split('-')[0] for arm in report['progress']}
        assert len(family_set) == 1
        family = family_set.pop()
        assert family in FAMILIES and family not in families
        families.add(family)
        prefix = family.lower()
        for path in (report_path, directory / f'{prefix}_endpoints.json', directory / f'{prefix}_validation_progress.csv'):
            source = oo._source_record(path)
            assert source['sha256'] == validation['files_sha256'][path.name], f'Changed report artifact: {path}'
            add_source(source)
        add_source(oo._source_record(validation_path))
        for source in report['sources']: add_source(source)
        for row in read(directory / f'{prefix}_endpoints.json'):
            key = (row['arm_id'], row['width'])
            previous = endpoints.setdefault(key, row)
            assert previous == row, 'Standalone references disagree across reports'
        frame = pd.read_csv(directory / f'{prefix}_validation_progress.csv')
        assert set(frame['arm_id']) == {family, family + '-GMC', family + '-LMC'}
        for (arm, width), curve in frame.groupby(['arm_id', 'width'], sort=False):
            x, y = curve['step'].to_numpy(), curve['loss'].to_numpy()
            assert width in oo.WIDTH_LABELS and len(x) == 5446
            assert x[0] == 64 and x[-1] == STEPS and np.all(np.diff(x) >= 0)
            assert np.isfinite(y).all()
            assert np.isclose(y[-1], endpoints[(arm, width)]['loss'], rtol=0, atol=1e-12)
        progress.append(frame)
        if report['clipping_limitation']: limitations[family] = report['clipping_limitation']
        print('Loaded verified tables:', family, flush=True)
    assert families == set(FAMILIES)
    expected = {(arm, width) for arm in ARMS for width in oo.WIDTH_LABELS}
    expected |= {('ST-' + width, width) for width in oo.WIDTH_LABELS}
    assert set(endpoints) == expected and len(endpoints) == 40
    rows = [endpoints[key] for key in sorted(endpoints)]
    data = pd.concat(progress, ignore_index=True)
    source_records = list(sources.values())
    print('Verifying original source hashes before plotting.', flush=True)
    oo._check_sources(source_records)

    def publish(stage, final):
        plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
        figures = []
        def save(fig, stem):
            for suffix in ('png', 'pdf'):
                name = stem + '.' + suffix
                fig.savefig(stage / name, dpi=180, bbox_inches='tight')
                figures.append(str(final / name))
            plt.close(fig)
        def appearance(arm):
            family, _, mode = arm.partition('-')
            mode = mode or 'none'
            return COLORS[family], STYLES[mode], MARKERS[mode], f'{family} ({mode})'
        for metric in ('loss', 'perplexity'):
            fig, ax = plt.subplots(figsize=(12, 8))
            for arm in ARMS:
                values = [endpoints[(arm, width)] for width in oo.WIDTH_LABELS]
                color, style, marker, label = appearance(arm)
                ax.plot([r['non_embedding_parameters'] for r in values], [r[metric] for r in values],
                        color=color, linestyle=style, marker=marker, markersize=5, fillstyle='none',
                        linewidth=1.7, label=label)
            standalone = [endpoints[('ST-' + w, w)] for w in oo.WIDTH_LABELS]
            ax.plot([r['non_embedding_parameters'] for r in standalone], [r[metric] for r in standalone],
                    color='#8B4513', marker='^', linestyle='None', markersize=9, label='Standalone')
            ax.set(xlabel='Active non-embedding parameters', ylabel=metric.capitalize(),
                   title='All concat variants · terminal ordinary validation · seed 42')
            ax.set_xticks([w['non_embedding_parameters'] for w in oo.WIDTHS])
            ax.ticklabel_format(axis='x', style='plain'); ax.grid(alpha=.2)
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .035), ncol=5, frameon=False)
            fig.text(.5, .005, 'Standalone: 1 epoch / 713,785,344 tokens per run; elastic: 4 epochs / 2,855,141,376 tokens per run.',
                     ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .15, 1, 1))
            save(fig, f'all_c_{metric}_vs_non_embedding_parameters')
        for zoom in (False, True):
            fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
            start = STEPS // 2 if zoom else 0
            for ax, width in zip(axes.flat, oo.WIDTH_LABELS):
                for arm in ARMS:
                    curve = data.loc[(data['arm_id'] == arm) & (data['width'] == width) & (data['step'] >= start)]
                    color, style, _, label = appearance(arm)
                    ax.plot(curve['step'], curve['loss'], color=color, linestyle=style, linewidth=1, label=label)
                ax.set(title=width, xlabel='Committed optimizer updates', ylabel='Loss', xlim=(start, STEPS))
                ax.grid(alpha=.2); ax.ticklabel_format(axis='x', style='plain')
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
            detail = 'Epochs 3–4 detail · unsmoothed' if zoom else 'Full four-epoch trajectory · unsmoothed'
            fig.suptitle('All concat variants · ordinary-validation loss · seed 42\n' + detail)
            fig.tight_layout(rect=(0, .12, 1, .94))
            save(fig, 'all_c_validation_loss_progress' + ('_epochs_3_4' if zoom else ''))
        pd.DataFrame(rows).to_csv(stage / 'all_c_endpoints.csv', index=False)
        (stage / 'all_c_endpoints.json').write_text(json.dumps(rows, indent=2))
        data.to_csv(stage / 'all_c_validation_progress.csv', index=False)
        report = {'scope': 'All nine concat variants and four standalone terminal references',
                  'endpoint_count': 40, 'progress_curves': 36, 'observations_per_curve': 5446,
                  'status': 'plots_generated_with_documented_clipping_gaps',
                  'strict_campaign_validation': 'not passed', 'clipping_limitations': limitations,
                  'holdout_evaluated': False, 'sources': source_records, 'figures': figures,
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'encoding': 'Color: ownership family C1/C2/C3. Line style: none/GMC/LMC.',
                  'lmc_semantics': 'GMC hooks plus completed AdamW parameter-change scaling, including decay'}
        (stage / 'report.json').write_text(json.dumps(report, indent=2))
        (stage / 'README.md').write_text(
            '# All concat variants\n\n'
            'C1/C2/C3, each with none/GMC/LMC: nine four-epoch runs, plus four original one-epoch standalone references. '
            'The endpoint tables contain 40 rows. Colors identify ownership family; solid/dashed/dotted lines identify '
            'none/GMC/LMC. Standalones are unconnected brown triangles with one legend entry.\n\n'
            'Validation progress contains 36 unsmoothed curves, each with 5,446 observations from update 64 through 348,528. '
            'The separate epochs 3–4 view exposes later differences. LMC combines GMC hooks and full AdamW-delta scaling, '
            'including weight decay. These are descriptive paired seed-42 results, without multi-seed significance claims.\n\n'
            'The existing endpoint/progress audits and source hashes support these plots. Missing clipping logs in corrected '
            'C1/C3 runs remain documented; strict full-campaign artifact validation has not passed. Original artifacts and '
            'previous reports are unchanged. No training or holdout evaluation was performed.\n')
        oo._check_sources(source_records)
        return report
    result = oo._publish_directory(output, publish)
    print(json.dumps({'output_dir': str(output), 'figures': result['figures'], 'endpoints': 40, 'curves': 36}, indent=2))


if __name__ == '__main__':
    main()
