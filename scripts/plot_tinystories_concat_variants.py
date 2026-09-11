#!/usr/bin/env python3
"""Plot validated concat variants and standalone references from saved results."""
import argparse
import csv
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

BASE = Path('/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1')
CAMPAIGN = BASE / 'campaigns/concat-gmc-lmc-v1'
STANDALONES = tuple('ST-' + width for width in oo.WIDTH_LABELS)
COLORS = ('#009E73', '#0072B2', '#CC79A7')
STYLES = ('-', '--', ':')
STEPS = 348528


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('C1', 'C2', 'C3'), default='C1')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    family = args.family
    prefix = family.lower()
    arms = (family, family + '-GMC', family + '-LMC')
    labels = (family + ' (none)', family + ' (GMC)', family + ' (LMC)')
    output = args.output_dir.resolve()
    if not output.is_relative_to(CAMPAIGN / 'reports'):
        parser.error('Output must be beneath the correction campaign reports directory')
    plan = read(CAMPAIGN / 'launchers/plan.json')
    audit_path = CAMPAIGN / 'diagnostics' / f'{prefix}-terminal-validation.json'
    audit = read(audit_path)
    corrected = oo._read_preflight_manifest(CAMPAIGN / 'campaign/campaign_manifest.json')
    frozen = read(plan['reference_manifest'])
    oo._check_content_hash(frozen, 'content_hash', 'original frozen manifest')
    original = oo._read_preflight_manifest(frozen['preflight_source']['path'])
    references = {r['arm_id']: r for r in frozen['runs'] if r['arm_id'] in (family, *STANDALONES)}
    roots = {a: Path(r['run_dir']) for a, r in references.items()}
    roots.update({a: CAMPAIGN / 'runs' / a for a in arms[1:]})
    sources = [oo._source_record(audit_path), oo._source_record(plan['reference_manifest']),
               frozen['preflight_source'], oo._source_record(CAMPAIGN / 'campaign/campaign_manifest.json')]
    rows, progress = [], {}
    for arm in (*arms, *STANDALONES):
        print(f'Reading validated results: {arm}', flush=True)
        root = roots[arm]
        manifest = original if arm in references else corrected
        run = next(r for r in manifest['runs'] if r['arm_id'] == arm)
        sidecar_path = root / 'terminal_validation_results.json'
        sidecar = read(sidecar_path)
        endpoints = oo._terminal_endpoints(sidecar, run, allow_partial=False)
        current_sources = [oo._source_record(p) for p in
                           (sidecar_path, Path(sidecar['checkpoint_path']), root / 'metrics.csv')]
        assert current_sources[1]['sha256'] == sidecar['checkpoint_sha256']
        if arm in references:
            reference = references[arm]
            for source in current_sources:
                assert source in reference['sources'], 'Original frozen source changed'
            assert endpoints == reference['endpoints']
        else:
            validated = next(r for r in audit['runs'] if r['arm_id'] == arm)
            assert validated['checkpoint_validation'] == 'passed'
            assert validated['checkpoint_sha256'] == sidecar['checkpoint_sha256']
            assert validated['terminal_source'] == current_sources[0]
            assert validated['endpoints'] == endpoints
            if family == 'C2':
                assert validated['strict_artifact_validation'] == 'passed'
            if family in ('C2', 'C3'):
                for source in current_sources:
                    assert source in validated['sources'], f'Validated {family} source changed'
        sources.extend(current_sources)
        for endpoint in endpoints:
            rows.append({'arm_id': arm, 'width': endpoint['width'],
                         'non_embedding_parameters': endpoint['non_embedding_parameters'],
                         'loss': endpoint['loss'], 'perplexity': endpoint['perplexity'],
                         'updates': run['assigned_updates'], 'epochs': run['assigned_epochs'],
                         'training_tokens': run['assigned_tokens'],
                         'evaluation_role': 'ordinary_validation',
                         'checkpoint_sha256': sidecar['checkpoint_sha256']})
        if arm in STANDALONES:
            continue
        chunks = []
        for chunk in pd.read_csv(root / 'metrics.csv', usecols=['split', 'step', 'granularity', 'loss'],
                                 chunksize=50000):
            chunks.append(chunk.loc[chunk['split'] == 'validation'].copy())
        frame = pd.concat(chunks, ignore_index=True)
        assert set(frame['granularity']) == set(oo.WIDTH_LABELS)
        frame['step'] = frame['step'].astype(int)
        frame['loss'] = frame['loss'].astype(float)
        assert np.isfinite(frame['loss']).all()
        curves = {}
        for endpoint in endpoints:
            values = frame.loc[frame['granularity'] == endpoint['width'], ['step', 'loss']]
            x, y = values['step'].to_numpy(), values['loss'].to_numpy()
            assert len(x) and np.all(np.diff(x) >= 0) and x[0] >= 0 and x[-1] == STEPS
            assert np.isclose(y[-1], endpoint['loss'], rtol=0, atol=1e-12)
            curves[endpoint['width']] = (x, y)
        progress[arm] = curves

    def publish(stage, final):
        plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
        figures = []
        def save(figure, stem):
            for suffix in ('png', 'pdf'):
                name = f'{stem}.{suffix}'
                figure.savefig(stage / name, dpi=180, bbox_inches='tight')
                figures.append(str(final / name))
            plt.close(figure)
        for metric in ('loss', 'perplexity'):
            fig, ax = plt.subplots(figsize=(10, 6))
            for arm, color, style, label in zip(arms, COLORS, STYLES, labels):
                points = sorted((r for r in rows if r['arm_id'] == arm), key=lambda r: r['non_embedding_parameters'])
                ax.plot([r['non_embedding_parameters'] for r in points], [r[metric] for r in points],
                        color=color, linestyle=style, label=label, marker='o', linewidth=2)
            standalone_points = sorted((r for r in rows if r['arm_id'] in STANDALONES),
                                       key=lambda r: r['non_embedding_parameters'])
            ax.plot([r['non_embedding_parameters'] for r in standalone_points],
                    [r[metric] for r in standalone_points], linestyle='None', marker='^',
                    markersize=9, color='#8B4513', label='Standalone')
            ax.set(xlabel='Active non-embedding parameters', ylabel=metric.capitalize(),
                   title=f'{family} variants · terminal ordinary validation · seed 42')
            ax.set_xticks([w['non_embedding_parameters'] for w in oo.WIDTHS])
            ax.ticklabel_format(axis='x', style='plain')
            ax.grid(alpha=.2); ax.legend()
            fig.text(.5, .015, 'Standalone: 1 epoch / 713,785,344 tokens per run; elastic: 4 epochs / 2,855,141,376 tokens per run.',
                     ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .045, 1, 1))
            save(fig, f'{prefix}_{metric}_vs_non_embedding_parameters')
        for zoom in (False, True):
            fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
            start = STEPS // 2 if zoom else 0
            for ax, width in zip(axes.flat, oo.WIDTH_LABELS):
                for arm, color, style, label in zip(arms, COLORS, STYLES, labels):
                    x, y = progress[arm][width]
                    mask = x >= start
                    ax.plot(x[mask], y[mask], color=color, linestyle=style, linewidth=1.2, label=label)
                ax.set(title=width, xlabel='Committed optimizer updates', ylabel='Loss', xlim=(start, STEPS))
                ax.grid(alpha=.2)
                ax.ticklabel_format(axis='x', style='plain')
            handles, legend_labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc='lower center', ncol=3, frameon=False)
            title = f'{family} variants · ordinary-validation loss · seed 42'
            fig.suptitle(title + ('\nEpochs 3–4 detail · unsmoothed' if zoom else '\nFull four-epoch trajectory · unsmoothed'))
            fig.tight_layout(rect=(0, .055, 1, .93))
            save(fig, f'{prefix}_validation_loss_progress' + ('_epochs_3_4' if zoom else ''))
        with (stage / f'{prefix}_endpoints.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        (stage / f'{prefix}_endpoints.json').write_text(json.dumps(rows, indent=2))
        with (stage / f'{prefix}_validation_progress.csv').open('w') as stream:
            writer = csv.writer(stream); writer.writerow(['arm_id', 'width', 'step', 'loss'])
            for arm, curves in progress.items():
                for width, (x, y) in curves.items():
                    writer.writerows((arm, width, int(step), float(loss)) for step, loss in zip(x, y))
        report = {'scope': f'{family} endpoints with standalone references; {family} validation-progress comparison',
                  'status': 'plots_generated_with_documented_clipping_gap' if audit['limitation'] else 'plots_generated',
                  'family_artifact_validation': audit['strict_artifact_validation'], 'figures': figures,
                  'holdout_evaluated': False, 'sources': sources,
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'clipping_limitation': audit['limitation'], 'strict_campaign_validation': 'not passed',
                  'lmc_semantics': 'GMC gradient hooks plus completed AdamW parameter-change correction, including decay',
                  'progress': {a: {w: {'observations': len(x), 'first_update': int(x[0]), 'last_update': int(x[-1])}
                                   for w, (x, y) in curves.items()} for a, curves in progress.items()}}
        (stage / 'report.json').write_text(json.dumps(report, indent=2))
        clipping_note = (
            f'Clipping logs for corrected {family} runs were not recorded because of the original-arm-name-only logging gate. '
            'These plots do not need clipping statistics; full campaign artifact validation remains incomplete. '
            if audit['limitation'] else
            'Both corrected C2 runs passed their strict terminal artifact validation. The C2 contract does not require '
            'a separate clipping sidecar. Full six-arm campaign validation is separate and remains incomplete. ')
        (stage / 'README.md').write_text(
            f'# {family} variant comparison\n\n'
            'Saved terminal ordinary-validation results and unsmoothed validation progress. '
            f'All three {family} runs completed four epochs, 348,528 updates and 2,855,141,376 training tokens. '
            'Four completed standalone references each used one epoch, 87,132 updates and 713,785,344 training tokens. '
            'Their terminal points appear as unconnected brown triangles with one Standalone legend entry. '
            f'Progress panels compare only the three {family} trajectories.\n\n'
            'LMC combines GMC gradient hooks with correction of the entire AdamW parameter change, including weight decay. '
            'These are paired seed-42 observations, without multi-seed significance claims.\n\n'
            + clipping_note +
            'Original artifacts and the strict campaign reporting gate were preserved. No holdout was evaluated.\n')
        oo._check_sources(sources)
        return report
    report = oo._publish_directory(output, publish)
    print(json.dumps({'output_dir': str(output), 'figures': report['figures'], 'progress': report['progress']}, indent=2))


if __name__ == '__main__':
    main()
