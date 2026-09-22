#!/usr/bin/env python3
"""Plot validated standalone grids and optional completed elastic endpoints."""
import argparse
import csv
from pathlib import Path
import sys
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--elastic-arms', nargs='+', choices=('S1', 'S2', 'C1', 'C2', 'C3'), default=[])
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
    selected_terminals = []
    scheduler_evidence = []
    if args.elastic_arms:
        c._require_equal(len(set(args.elastic_arms)), len(args.elastic_arms), 'unique selected arms')
        submissions = c._read_json(root / 'launchers/submissions.json')['jobs']
        for arm in args.elastic_arms:
            intent = max((j for j in submissions if j['arm_id'] == arm), key=lambda j: j['attempt_id'])
            job, attempt = intent['job_id'], intent['attempt_id']
            accounting = subprocess.run(['sacct', '-X', '--noheader', '--parsable2', '--jobs=' + job,
                '--format=JobIDRaw,State,ExitCode,ElapsedRaw,NodeList'], check=True, capture_output=True,
                text=True, timeout=45).stdout.strip()
            c._require_equal(len(accounting.splitlines()), 1, f'{arm}.scheduler rows')
            c._require_equal(accounting.split('|')[:3], [job, 'COMPLETED', '0:0'], f'{arm}.scheduler success')
            worker_path = root / 'launchers' / f'worker-{arm}-{attempt}.json'
            entry_path = root / 'launchers' / f'cuda-entry-{arm}-{attempt}.json'
            worker, entry = c._read_json(worker_path), c._read_json(entry_path)
            c._require_equal(worker['job_id'], job, f'{arm}.worker job')
            c._require_equal(worker['status'], 'completed', f'{arm}.worker status')
            c._require_equal(worker['returncode'], 0, f'{arm}.worker exit')
            c._require_equal(entry['job_id'], job, f'{arm}.CUDA entry job')
            config = c._read_json(root / 'runs' / arm / 'config.json')
            c._require_equal(config['training']['resolved_mixed_precision'], 'bf16', f'{arm}.actual precision')
            print(f'Strictly validating completed {arm} (job {job})...', flush=True)
            terminal = c.inspect_selected_terminals(manifest_path, [arm])[0]
            ledger = c._read_json(root / 'runs' / arm / 'resource_attempts.json')['attempts']
            measured = ledger[worker['process_uuid']]
            c._require_equal(measured['slurm_job_id'], job, f'{arm}.resource job')
            c._require_equal(measured['status'], 'completed', f'{arm}.resource status')
            c._require_equal(measured['measurement_complete'], True, f'{arm}.complete measurement')
            if not measured['peak_allocated_bytes'] or not measured['peak_reserved_bytes']:
                raise ValueError(f'{arm}: no actual CUDA allocation')
            selected_terminals.append(terminal)
            sources.extend(terminal['sources'])
            sources.extend(c._source_record(p) for p in (worker_path, entry_path))
            scheduler_evidence.append(dict(arm_id=arm, intent=intent, accounting=accounting,
                                           worker=worker, cuda_entry=entry, resources=measured))
        rows += c._endpoint_table(dict(runs=selected_terminals, status='complete',
                                      campaign_id=preflight['campaign_id']), preflight)
    endpoint_count = 8 + 4 * len(args.elastic_arms)
    c._require_equal(len(rows), endpoint_count, 'selected endpoint count')
    c._require_equal(len({tuple(r['endpoint_identity']) for r in rows}), endpoint_count, 'unique endpoint identities')
    for row in rows:
        epochs = 1 if row['arm_id'].startswith('ST-') else 4
        for field, value in dict(actual_epochs=epochs, assigned_epochs=epochs, actual_tokens=713785344 * epochs,
                                 assigned_tokens=713785344 * epochs, seed=42).items():
            c._require_equal(row[field], value, f"{row['run_id']}.{field}")

    def publish(stage, output):
        status = 'selected_completed_runs' if args.elastic_arms else 'standalone_only'
        stem = 'endpoints' if args.elastic_arms else 'standalone_endpoints'
        c.write_json_artifact(stage / f'{stem}.json', dict(
            schema_version=1, status=status, holdout_evaluated=False, endpoints=rows))
        if args.elastic_arms:
            c.write_json_artifact(stage / 'selected_terminal_validation.json', dict(
                schema_version=1, status='passed', arms=args.elastic_arms,
                terminals=selected_terminals, scheduler_evidence=scheduler_evidence))
        fields = ['group', 'campaign_id', 'run_id', 'arm_id', 'ffn_dimension',
                  'non_embedding_parameters', 'loss', 'perplexity', 'seed',
                  'actual_epochs', 'actual_tokens', 'checkpoint_sha256']
        with (stage / f'{stem}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        ticks = sorted({r['non_embedding_parameters'] for r in rows})
        dimensions = {r['non_embedding_parameters']: r['ffn_dimension'] for r in rows}
        colors = {True: '#0072B2', False: '#E69F00'}
        for metric in ('loss', 'perplexity'):
            figure = Figure(figsize=(7.2, 5.4))
            ax = figure.subplots()
            standalone_rows = [r for r in rows if r['arm_id'].startswith('ST-')]
            for historical in (True, False):
                values = [r for r in standalone_rows if r['historical_reference'] == historical]
                for row in values:
                    shared = any(r['historical_reference'] != historical
                                 and r['non_embedding_parameters'] == row['non_embedding_parameters']
                                 and r[metric] == row[metric] for r in standalone_rows)
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
            elastic_styles = dict(S1=('#009E73', 'o', '-'), S2=('#8C56A2', 's', '--'),
                                  C1=('#CC79A7', 'D', '-'), C2=('#555555', 'v', '--'),
                                  C3=('#D55E00', 'P', ':'))
            for arm in args.elastic_arms:
                values = sorted((r for r in rows if r['arm_id'] == arm), key=lambda r: r['non_embedding_parameters'])
                color, marker, style = elastic_styles[arm]
                line, = ax.plot([r['non_embedding_parameters'] for r in values], [r[metric] for r in values],
                                color=color, marker=marker, linestyle=style, linewidth=1.6,
                                markersize=6, markerfacecolor='white', markeredgewidth=1.3, label=arm)
                handles.append(line)
            ax.legend(handles=handles, frameon=False, loc='upper right')
            figure.suptitle('TinyStories-Instruct' if args.elastic_arms else 'TinyStories-Instruct · standalone models',
                            fontsize=14, y=.97)
            footer = ('Terminal ordinary validation · seed 42\nStandalone: 1 epoch · Elastic: 4 epochs'
                      if args.elastic_arms else 'Terminal ordinary validation · seed 42 · 1 epoch')
            figure.text(.5, .025, footer,
                        ha='center', fontsize=9, color='#555555')
            figure.tight_layout(rect=(0, .055, 1, .91))
            for suffix in ('png', 'pdf'):
                figure.savefig(stage / f'{metric}_vs_parameters.{suffix}', dpi=180)
        report = dict(schema_version=1, status=status, endpoint_count=endpoint_count,
                      selected_elastic_arms=args.elastic_arms,
                      holdout_evaluated=False, input_sources=sources,
                      plot_style=dict(linear='blue triangle', geometric='orange triangle',
                                      coincident_points='single triangle, blue left half and orange right half'),
                      interpretation_scope='Descriptive seed-42 terminal comparison; no across-seed inference. '
                      'Standalones receive one epoch; elastics receive four epochs with sampled widths, not equal compute or direct exposure. '
                      'Does not complete T045/T052 or the 24/28-endpoint campaign reports.',
                      output_sha256={p.name: c._source_record(p)['sha256'] for p in sorted(stage.iterdir())})
        report['content_hash'] = c.stable_hash(report)
        c._check_sources(sources)
        c.write_json_artifact(stage / 'plot_manifest.json', report)
        return report

    c._publish_directory(args.output_dir.resolve(), publish)
    print(f'Published {endpoint_count} strictly validated endpoints to {args.output_dir}', flush=True)
    for r in rows:
        print(r['group'], r['arm_id'], r['ffn_dimension'], r['loss'], r['perplexity'])


if __name__ == '__main__':
    main()
