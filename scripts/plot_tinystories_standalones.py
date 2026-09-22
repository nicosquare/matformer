#!/usr/bin/env python3
"""Plot the two standalone grids using strict readers from the campaign snapshot."""
import argparse
import csv
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    sys.path.insert(0, str(root / 'source'))
    from src.evaluation import optimizer_ownership as c
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    plan_path = root / 'launchers/plan.json'
    barrier_path = root / 'launchers/standalone-barrier.json'
    manifest_path = root / 'campaign/campaign_manifest.json'
    plan = c._read_json(plan_path)
    barrier = c._read_json(barrier_path)
    c._check_content_hash(plan, 'content_hash', 'plan')
    c._check_content_hash(barrier, 'content_hash', 'standalone barrier')
    c._require_equal(barrier['status'], 'complete', 'barrier status')
    c._require_equal(barrier['bindings'], plan['bindings'], 'barrier bindings')
    preflight = c._read_preflight_manifest(manifest_path)
    c._require_equal(preflight['schema_version'], 4, 'fresh campaign schema')
    c._require_equal(preflight['manifest_hash'], plan['bindings']['manifest_hash'], 'fresh manifest')
    fresh_labels = ['ST-g125', 'ST-g250', 'ST-g500', 'ST-g1000']
    print('Strictly validating fresh standalones...', flush=True)
    terminals = c.inspect_selected_terminals(manifest_path, fresh_labels)
    c._require_equal(terminals, barrier['terminals'], 'saved standalone barrier terminals')
    rows = c._endpoint_table(dict(runs=terminals, status='complete',
                                 campaign_id=preflight['campaign_id']), preflight)
    sources = [c._source_record(p) for p in (plan_path, barrier_path, manifest_path, Path(__file__))]
    for saved in terminals:
        sources.extend(saved['sources'])

    reference = Path(plan['reference_manifest'])
    old = c._read_json(reference)
    c._check_content_hash(old, 'content_hash', 'historical frozen')
    c._require_equal(old['schema_version'], c.FROZEN_MANIFEST_SCHEMA_VERSION, 'historical frozen schema')
    c._require_equal(old['status'], 'complete', 'historical status')
    c._require_equal(old['holdout_evaluated'], False, 'historical holdout')
    sources.extend([c._source_record(reference), old['preflight_source']])
    c._check_sources(sources)
    old_preflight = c._read_preflight_manifest(old['preflight_source']['path'])
    c._require_equal(old_preflight['schema_version'], 1, 'historical schema')
    c._require_equal(old['campaign_id'], old_preflight['campaign_id'], 'historical campaign')
    c._require_equal(old['preflight_manifest_hash'], old_preflight['manifest_hash'], 'historical preflight')
    c._require_equal([r['arm_id'] for r in old['runs']], [a['arm_id'] for a in c.ARMS], 'historical frozen arms')
    labels = [a['arm_id'] for a in c.STANDALONE_ARMS]
    selected = [r for r in old['runs'] if r['arm_id'] in labels]
    c._require_equal([r['arm_id'] for r in selected], labels, 'historical selected standalones')
    definitions = {r['arm_id']: r for r in old_preflight['runs']}
    for saved in selected:
        arm = saved['arm_id']
        print(f'Strictly validating historical {arm}...', flush=True)
        c._check_sources(saved['sources'])
        for path, present in saved['optional_sources'].items():
            c._require_equal(Path(path).exists(), present, f'Historical source presence: {path}')
        actual = c._inspect_terminal_run(saved['run_dir'], definitions[arm],
                                         old_preflight['expected_traces'][arm], allow_partial=False)
        c._require_equal(actual, saved, f'{arm}.historical terminal')
        sources.extend(saved['sources'])
    history = c._endpoint_table({**old, 'runs': selected}, old_preflight)
    for row in history:
        row.update(historical_reference=True, group='historical', canonical_arm=row['arm_id'],
                   endpoint_identity=[row['campaign_id'], row['run_id'], row['width_fraction'], row['ffn_dimension']],
                   source_records=next(r['sources'] for r in selected if r['arm_id'] == row['arm_id']),
                   clipping_observations=None)
    rows += history
    c._require_equal(len(rows), 8, 'standalone endpoint count')
    c._require_equal(len({tuple(r['endpoint_identity']) for r in rows}), 8, 'unique endpoint identities')
    for row in rows:
        for field, value in dict(actual_epochs=1, assigned_epochs=1, actual_tokens=713785344,
                                 assigned_tokens=713785344, seed=42).items():
            c._require_equal(row[field], value, f"{row['run_id']}.{field}")

    def publish(stage, output):
        c.write_json_artifact(stage / 'standalone_endpoints.json', dict(
            schema_version=1, status='standalone_only', holdout_evaluated=False, endpoints=rows))
        fields = ['group', 'campaign_id', 'run_id', 'arm_id', 'ffn_dimension',
                  'non_embedding_parameters', 'loss', 'perplexity', 'seed',
                  'actual_epochs', 'actual_tokens', 'checkpoint_sha256']
        with (stage / 'standalone_endpoints.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        ticks = sorted({r['non_embedding_parameters'] for r in rows})
        dimensions = {r['non_embedding_parameters']: r['ffn_dimension'] for r in rows}
        colors = {True: '#0072B2', False: '#E69F00'}
        for metric in ('loss', 'perplexity'):
            figure = Figure(figsize=(7.2, 5.4))
            ax = figure.subplots()
            for historical in (True, False):
                values = [r for r in rows if r['historical_reference'] == historical]
                for row in values:
                    shared = any(r['historical_reference'] != historical
                                 and r['non_embedding_parameters'] == row['non_embedding_parameters']
                                 and r[metric] == row[metric] for r in rows)
                    # One triangle at coincident coordinates: blue left, orange right.
                    fill = ('left' if historical else 'right') if shared else 'full'
                    ax.plot(row['non_embedding_parameters'], row[metric], linestyle='None',
                            marker='^', markersize=10, fillstyle=fill,
                            markerfacecolor=colors[historical], markerfacecoloralt='none',
                            markeredgewidth=0, color=colors[historical], zorder=4)
            ax.set_ylabel('Validation loss (nats)' if metric == 'loss' else 'Validation perplexity')
            ax.set_xlabel('Active non-embedding parameters (thousands)')
            ax.set_xticks(ticks, [f'{v / 1000:.1f}' for v in ticks])
            ax.grid(alpha=.15)
            ax.set_axisbelow(True)
            ax.margins(x=.08, y=.12)
            ax.spines['right'].set_visible(False)
            top = ax.secondary_xaxis('top')
            top.set_xticks(ticks, [str(dimensions[v]) for v in ticks])
            top.set_xlabel('FFN dimension')
            handles = [Line2D([], [], linestyle='None', marker='^', markersize=9,
                              markeredgewidth=0, color=colors[historical], label=label)
                       for historical, label in ((True, 'Linear'), (False, 'Geometric'))]
            ax.legend(handles=handles, frameon=False, loc='upper right')
            figure.suptitle('TinyStories-Instruct · standalone models', fontsize=14, y=.97)
            figure.text(.5, .025, 'Terminal ordinary validation · seed 42 · 1 epoch',
                        ha='center', fontsize=9, color='#555555')
            figure.tight_layout(rect=(0, .055, 1, .91))
            for suffix in ('png', 'pdf'):
                figure.savefig(stage / f'{metric}_vs_parameters.{suffix}', dpi=180)
        report = dict(schema_version=1, status='standalone_only', endpoint_count=8,
                      holdout_evaluated=False, input_sources=sources,
                      plot_style=dict(linear='blue triangle', geometric='orange triangle',
                                      coincident_points='single triangle, blue left half and orange right half'),
                      interpretation_scope='Descriptive seed-42 terminal standalone comparison; no across-seed inference. '
                      'Does not complete T045/T052 or the 24/28-endpoint campaign reports.',
                      output_sha256={p.name: c._source_record(p)['sha256'] for p in sorted(stage.iterdir())})
        report['content_hash'] = c.stable_hash(report)
        c._check_sources(sources)
        c.write_json_artifact(stage / 'plot_manifest.json', report)
        return report

    c._publish_directory(args.output_dir.resolve(), publish)
    print(f'Published eight strictly validated endpoints to {args.output_dir}', flush=True)
    for r in rows:
        print(r['group'], r['ffn_dimension'], r['loss'], r['perplexity'])


if __name__ == '__main__':
    main()
