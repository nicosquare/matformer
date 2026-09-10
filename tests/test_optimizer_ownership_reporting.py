"""Saved phase-6 diagnostics, measured histories, and ordinary-validation evidence."""
import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from test_optimizer_ownership_resume import ARMS, packed_fixture, train
from test_optimizer_ownership import assert_state_equal
from src.training import optimizer_state
from src.utils.config import ConfigError


@pytest.mark.parametrize('arm', ARMS)
def test_storage_reads_actual_tensors_without_allocating_lazy_state(tmp_path, arm):
    bundle = packed_fixture(tmp_path, arm)
    opt = bundle[2]
    before = copy.deepcopy(opt.state_dict())
    empty = optimizer_state.measure_optimizer_storage(opt, step=0)
    assert empty['moment_elements'] == empty['counter_elements'] == 0
    assert_state_equal(before, opt.state_dict())
    train(bundle, stop=1)
    before = copy.deepcopy(opt.state_dict())
    measured = optimizer_state.measure_optimizer_storage(opt, step=1)
    optimizers = [entry.optimizer for entry in opt.entries] if hasattr(opt, 'entries') else [opt]
    tensors = [value for item in optimizers for state in item.state.values() for value in state.values() if torch.is_tensor(value)]
    assert measured['total_bytes'] == sum(t.numel() * t.element_size() for t in tensors)
    assert measured['moment_elements'] == sum(t.numel() for item in optimizers for state in item.state.values() for key, t in state.items() if key != 'step' and torch.is_tensor(t))
    assert sum(row['bytes'] for row in measured['components']) == measured['total_bytes']
    assert all(row['dtype'] == 'torch.float32' for row in measured['components'])
    if arm == 'C3':
        assert len(measured['owners']) == 5
        assert all(row['allocated_histories'] == 0 for row in empty['owners'])
        active = opt.active_owner_ids(bundle[-1]['optimizer_last_active_granularity'])
        assert all((row['allocated_histories'] > 0) == (row['owner_id'] in active) for row in measured['owners'])
    assert_state_equal(before, opt.state_dict())


@pytest.mark.parametrize('arm', ARMS)
def test_committed_saved_trace_and_summary_reconcile(tmp_path, arm):
    from src.utils.metrics import append_optimizer_ownership_observation
    from src.training.run import build_ownership_run_summary
    bundle = packed_fixture(tmp_path, arm)
    config, model, opt, clock, batches, state = bundle
    opt._ownership_observer = lambda: append_optimizer_ownership_observation(config, state, train_dataloader=batches)
    train(bundle)
    root = Path(config['run']['output_dir'])
    rows = [json.loads(line) for line in (root / 'optimizer_ownership_trace.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == list(range(1, 9))
    for row in rows:
        assert row['schema_version'] == 1
        assert row['contract_hash'] == config['optimizer_ownership_contract_hash']
        assert row['scheduler_position'] == row['step']
        assert sum(row['width_selection_counts'].values()) == row['step']
        assert row['tokens_seen'] == row['step'] * 8
        assert row['batch_sha256'] == hashlib.sha256(__import__('numpy').asarray(row['sample_ids'], dtype='<u8').tobytes()).hexdigest()
    assert rows[-1]['epoch'] == 4 and rows[-1]['batch_index'] == 0
    fields = build_ownership_run_summary(config, model, opt, state)
    assert fields['optimizer_ownership_schema_version'] == 1
    assert fields['optimizer_ownership']['accounting_reconciled']
    assert fields['optimizer_ownership']['expected_exposure']['label'] == 'uniform replacement expectation; realized counts are random'
    assert fields['optimizer_ownership']['storage']['total_bytes'] > 0
    assert fields['optimizer_ownership']['resources']['peak_allocated_bytes'] is None
    assert fields['optimizer_ownership']['resources']['measurement_complete'] is False
    assert (root / 'optimizer_ownership_clipping.jsonl').exists() == (arm in ('C1', 'C3'))


@pytest.mark.parametrize('arm', ['C1', 'C3'])
def test_clipping_active_denominators_and_saved_plot_series(tmp_path, arm):
    from src.utils.metrics import append_optimizer_ownership_observation, write_json_artifact
    from src.training.run import build_ownership_run_summary
    from src.evaluation.optimizer_ownership import report_run_artifacts
    bundle = packed_fixture(tmp_path, arm)
    config, model, opt, clock, batches, state = bundle
    opt._ownership_observer = lambda: append_optimizer_ownership_observation(config, state, train_dataloader=batches)
    train(bundle)
    root = Path(config['run']['output_dir'])
    summary = {'run_id': config['run']['run_id'], **build_ownership_run_summary(config, model, opt, state)}
    write_json_artifact(root / 'run_summary.json', summary)
    import csv
    with (root / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'split', 'granularity', 'loss', 'perplexity'])
        writer.writeheader()
        for step, loss in ((4, 2.), (8, 1.)):
            writer.writerow(dict(step=step, split='validation', granularity='g250', loss=loss, perplexity=math.exp(loss)))
    report = report_run_artifacts(root, tmp_path / 'plots')
    assert report['trajectories']['validation:g250']['loss'] == [2., 1.]
    assert report['resources']['measurement_complete'] is False
    assert report['resource_label'] == 'incomplete measurements'
    clipping = [json.loads(line) for line in (root / 'optimizer_ownership_clipping.jsonl').read_text().splitlines()]
    for width, groups in report['clipping_by_width'].items():
        for owner, result in groups.items():
            values = [r['groups'][owner]['coefficient'] for r in clipping if r['width'] == width and r['groups'][owner]['active']]
            assert result['active_observations'] == len(values)
            assert result['frequency'] == (sum(v < 1 for v in values) / len(values) if values else None)
    assert len(report['figures']) == 6
    assert all(Path(path).stat().st_size > 100 for path in report['figures'])
    rows = (root / 'optimizer_ownership_trace.jsonl').read_text().splitlines()
    bad = json.loads(rows[-1]); bad['tokens_seen'] += 8
    rows[-1] = json.dumps(bad)
    (root / 'optimizer_ownership_trace.jsonl').write_text('\n'.join(rows) + '\n')
    with pytest.raises(ConfigError, match='tokens'):
        report_run_artifacts(root, tmp_path / 'bad-plots')


@pytest.mark.parametrize('arm,expected', [('S1', 1049728), ('S2', 4198912), ('C1', 1049728), ('C2', 3609088), ('C3', 1049728)])
def test_post_exposure_pinned_model_moment_elements(arm, expected):
    from test_optimizer_ownership import _config, training, WIDTHS
    from src.models.ffn import CatLlamaMLP, ModifiedLlamaMLP
    from src.models.wiring import ModifiedLlamaForCausalLM
    from src.training.steps import build_optimizer_and_scheduler
    config = _config(intermediate_size=256, hidden_size=64)
    config.vocab_size = 2048
    config.num_hidden_layers = 4
    model = ModifiedLlamaForCausalLM(config, mlp_cls=ModifiedLlamaMLP if arm.startswith('S') else CatLlamaMLP,
                                   mlp_kwargs={'gradient_membership_correction_enabled': False})
    assert sum(p.numel() for p in model.parameters()) == 524864
    scope = 'per_ffn_block' if arm == 'C3' else 'per_granularity' if arm in ('S2', 'C2') else 'shared'
    opt, clock = build_optimizer_and_scheduler(model, training(scope))
    for width in WIDTHS:
        opt.zero_grad(set_to_none=True)
        model.configure_subnetwork(width)
        tokens = torch.arange(1, 9).reshape(1, 8)
        model(input_ids=tokens, labels=tokens).loss.backward()
        if scope == 'per_ffn_block':
            for owner in opt.active_owner_ids(width): opt.optimizer_for(owner).step()
        elif scope == 'per_granularity': opt.optimizer_for(width).step()
        else: opt.step()
    measurement = optimizer_state.measure_optimizer_storage(opt, step=4)
    assert measurement['moment_elements'] == expected
    assert measurement['moment_bytes'] == expected * 4


from test_optimizer_ownership_campaign import audited_inputs
import src.evaluation.optimizer_ownership as campaign


@pytest.fixture
def terminal_campaign(tmp_path, audited_inputs, monkeypatch):
    """Pinned controls with synthetic terminal metadata; bulk trace IO is stubbed.

    Real streamed trace accounting is exercised above. No full-budget training or
    multi-million-row trace generation is implied by these comparison fixtures.
    """
    import yaml
    recipe, _, _ = audited_inputs
    source = tmp_path / 'recipe.yaml'
    source.write_text(yaml.safe_dump(recipe))
    expected = {a['arm_id']: {'epochs': [dict(epoch_index=i, sha256=f'epoch-{i}') for i in range(a['assigned_epochs'])],
                'actions': None if a['source_width'] else {'sha256': 'elastic-actions'}} for a in campaign.ARMS}
    monkeypatch.setattr(campaign, 'build_expected_traces', lambda *a: expected)
    campaign.preflight_campaign(campaign_path=source, prepared_corpus_dir=tmp_path / 'corpus',
        tokenizer_dir=tmp_path / 'tokenizer', output_dir=tmp_path / 'preflight', run_output_root=tmp_path / 'runs')
    manifest_path = tmp_path / 'preflight/campaign_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for run in manifest['runs']:
        root = Path(run['output_path']); root.mkdir(parents=True)
        arm, steps = run['arm_id'], run['assigned_updates']
        contract = run['optimizer_ownership_contract']
        counts = dict.fromkeys(campaign.WIDTH_LABELS, steps // 4) if not run['source_width'] else {'g1000': steps}
        quarters = {f'O-{q}': steps * (4-i)//4 for i,q in enumerate('ABCD')} if not run['source_width'] else {}
        calls = {**quarters, 'O-common': steps} if arm == 'C3' else counts if run['state_scope'] == 'per_granularity' else {'shared': steps}
        checkpoint = root / 'latest.pt'
        payload = dict(checkpoint_kind='resumable_training', checkpoint_schema_version=1,
            optimizer_ownership_checkpoint_schema_version=1, run_id=run['run_id'],
            optimizer_ownership_contract=contract, optimizer_ownership_contract_hash=run['contract_hash'],
            step=steps, tokens_seen=run['assigned_tokens'], epoch=run['assigned_epochs'], batch_index=0,
            global_scheduler_position=steps, optimizer_width_selection_counts=counts,
            optimizer_quarter_activation_counts=quarters, optimizer_update_counts=calls,
            model_state_dict={'fixture': torch.ones(1)}, optimizer_state_dict={'fixture': {}}, scheduler_state_dict={'fixture': {}})
        if run['state_scope'] != 'shared':
            payload['optimizer_state_dict'] = None
            payload['optimizer_state_collection'] = {'total_successful_updates': steps}
            payload['scheduler_state_dict'] = {'position': steps}
        torch.save(payload, checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        protocol = contract['evaluation']['validation']
        common = dict(evaluation_role=campaign.EVALUATION_ROLE, validation_manifest_hash=campaign.PINNED_DATA['ordinary_validation_manifest_hash'],
            evaluation_protocol_hash=campaign.stable_hash(protocol), validation_loss_aggregation=campaign.VALIDATION_AGGREGATION,
            count_convention=campaign.PARAMETER_COUNT_CONVENTION, evaluation_examples=285, evaluation_target_tokens=36195)
        sidecar = dict(schema_version=1, campaign_id=run['campaign_id'], arm_id=arm, run_id=run['run_id'],
            contract_hash=run['contract_hash'], contract=contract, checkpoint_path=str(checkpoint), checkpoint_sha256=digest,
            checkpoint_bytes=checkpoint.stat().st_size, global_step=steps, representation=run['representation'],
            state_scope=run['state_scope'], clipping=run['clipping'], initialization=run['initialization'],
            evaluation_protocol=protocol, **common)
        for unit in ('updates', 'tokens', 'epochs'):
            sidecar['assigned_'+unit] = sidecar['actual_'+unit] = run['assigned_'+unit]
        sidecar['endpoints'] = [dict(width=w['label'], loss=2.0, perplexity=math.exp(2),
            non_embedding_parameters=w['non_embedding_parameters'], **common) for w in campaign.WIDTHS if w['label'] in run['endpoint_widths']]
        sidecar['content_hash'] = campaign.stable_hash(sidecar)
        (root / 'terminal_validation_results.json').write_text(json.dumps(sidecar))
        audit = dict(schema_version=1, run_id=run['run_id'], campaign_id=run['campaign_id'], arm_id=arm,
            contract=contract, contract_hash=run['contract_hash'], state_scope=run['state_scope'], clipping_contract=run['resolved_config']['training']['gradient_clipping'],
            steps=steps, tokens_seen=run['assigned_tokens'], packed_tokens_per_update=8192, epoch=run['assigned_epochs'], batch_index=0,
            scheduler_position=steps, width_selection_counts=counts, quarter_activation_counts=quarters, owner_call_counts=calls,
            accounting_reconciled=True, sampler_state={'fixture': True}, trace_path='optimizer_ownership_trace.jsonl',
            clipping_path='optimizer_ownership_clipping.jsonl' if arm in ('C1','C3') else None,
            expected_exposure={'label': 'uniform expectation', 'width_selections': counts if quarters else {}, 'quarter_activations': quarters},
            storage={'owners': [], 'components': [], 'moment_elements': 0, 'total_bytes': 0}, temporary_concat_storage={},
            resources={'elapsed_seconds': 10., 'attempted_steps': steps, 'measurement_complete': False,
                       'peak_allocated_bytes': None, 'peak_reserved_bytes': None},
            checkpoint={'terminal_checkpoint_path': str(checkpoint), 'terminal_checkpoint_sha256': digest,
                        'terminal_checkpoint_bytes': checkpoint.stat().st_size, 'terminal_checkpoint_purpose': 'resumable_training'})
        (root / 'run_summary.json').write_text(json.dumps(dict(run_id=run['run_id'], status='completed', optimizer_ownership_schema_version=1, optimizer_ownership=audit)))
        (root / 'optimizer_ownership_trace.jsonl').write_text('{}\n')
        if audit['clipping_path']: (root / audit['clipping_path']).write_text('{}\n')
        (root / 'metrics.csv').write_text('step,split,granularity,loss,perplexity\n1,validation,g250,2.0,7.38905609893065\n')
        # A poisoned holdout file proves that readers never open it.
        (root / 'final_holdout_results.json').write_text('DO NOT READ')
    def observations(root, summary):
        a = summary['optimizer_ownership']; trace = expected[a['arm_id']]
        return dict(committed_updates=a['steps'], clipping_by_width={}, action_sha256='elastic-actions',
                    epoch_order_sha256={str(e['epoch_index']): e['sha256'] for e in trace['epochs']})
    monkeypatch.setattr(campaign, 'inspect_run_observations', observations)
    return manifest_path, tmp_path / 'runs'


def change_terminal(root, arm, mutate):
    path = root / arm / 'terminal_validation_results.json'
    data = json.loads(path.read_text()); mutate(data)
    data['content_hash'] = campaign.stable_hash({k:v for k,v in data.items() if k != 'content_hash'})
    path.write_text(json.dumps(data))


def test_complete_freeze_tables_and_figures(tmp_path, terminal_campaign):
    import csv
    manifest, root = terminal_campaign
    frozen = campaign.freeze_campaign(campaign_manifest=manifest, run_root=root, output_dir=tmp_path / 'frozen')
    assert frozen['status'] == 'complete' and len(frozen['runs']) == 9
    report = campaign.report_campaign(manifest=tmp_path / 'frozen/frozen_manifest.json', output_dir=tmp_path / 'report')
    rows = json.loads((tmp_path / 'report/optimizer_ownership_endpoints.json').read_text())['endpoints']
    with (tmp_path / 'report/optimizer_ownership_endpoints.csv').open() as f: csv_rows = list(csv.DictReader(f))
    assert len(rows) == len(csv_rows) == 24
    for row, exported in zip(rows, csv_rows):
        assert exported == {k: campaign.endpoint_csv_value(v) for k,v in row.items()}
    assert len(report['figures']) == 4
    assert all(Path(p).stat().st_size > 100 for p in report['figures'])
    assert len(report['individual_reports']) == 9
    assert len(report['comparisons']) == 6
    for metric in ('loss', 'perplexity'):
        fig = campaign.endpoint_figure(rows, metric=metric, partial=False)
        ax = fig.axes[0]
        assert len(ax.lines) == 9 and not ax.containers
        assert [list(line.get_xdata()) for line in ax.lines[:5]] == [[w['non_embedding_parameters'] for w in campaign.WIDTHS]] * 5
        assert all(line.get_linestyle() == 'None' for line in ax.lines[5:])
        assert ax.get_legend_handles_labels()[1] == ['S1', 'S2', 'C1', 'C2', 'C3', 'Standalone']
        assert 'seed 42' in fig.texts[0].get_text()


@pytest.mark.parametrize('field,value', [
    ('global_step', 87132), ('evaluation_role', 'final_holdout'), ('evaluation_target_tokens', 7),
    ('actual_tokens', 713785344), ('assigned_epochs', 1), ('representation', 'slicing'),
    ('evaluation_protocol_hash', 'trailing-five-mean'), ('contract_hash', 'historical'),
    ('checkpoint_path', 'best.pt'), ('validation_manifest_hash', 'mixed'),
])
def test_freeze_rejects_terminal_mismatch(tmp_path, terminal_campaign, field, value):
    manifest, root = terminal_campaign
    change_terminal(root, 'C3', lambda d: d.update({field:value}))
    with pytest.raises((ConfigError, OSError), match=field+'|checkpoint'):
        campaign.freeze_campaign(campaign_manifest=manifest, run_root=root, output_dir=tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'nonfinite', 'count', 'targets', 'exp', 'role'])
def test_freeze_rejects_bad_endpoints_even_partial(tmp_path, terminal_campaign, mutation):
    manifest, root = terminal_campaign
    def mutate(d):
        row = d['endpoints'][0]
        if mutation == 'duplicate': d['endpoints'].append(copy.deepcopy(row))
        elif mutation == 'missing': d['endpoints'].pop()
        elif mutation == 'nonfinite': row['loss'] = float('nan')
        elif mutation == 'count': row['non_embedding_parameters'] += 1
        elif mutation == 'targets': row['evaluation_target_tokens'] += 1
        elif mutation == 'exp': row['perplexity'] = 42
        else: row['evaluation_role'] = 'final_holdout'
    change_terminal(root, 'C3', mutate)
    with pytest.raises(ConfigError):
        campaign.freeze_campaign(campaign_manifest=manifest, run_root=root, output_dir=tmp_path / 'bad', allow_partial=mutation != 'missing')
    assert not (tmp_path / 'bad').exists()


def test_partial_requires_two_opt_ins_and_labels(tmp_path, terminal_campaign):
    manifest, root = terminal_campaign
    change_terminal(root, 'C3', lambda d: d['endpoints'].pop())
    dirs = [root / 'C3']
    with pytest.raises(ConfigError, match='missing'):
        campaign.freeze_campaign(campaign_manifest=manifest, run_dirs=dirs, output_dir=tmp_path / 'bad')
    result = campaign.freeze_campaign(campaign_manifest=manifest, run_dirs=dirs, output_dir=tmp_path / 'frozen', allow_partial=True)
    assert result['status'] == 'partial' and len(result['missing_endpoints']) == 21
    path = tmp_path / 'frozen/frozen_manifest.json'
    with pytest.raises(ConfigError, match='allow-partial'):
        campaign.report_campaign(manifest=path, output_dir=tmp_path / 'bad-report')
    result = campaign.report_campaign(manifest=path, output_dir=tmp_path / 'report', allow_partial=True)
    assert result['status'] == 'partial'
    assert all(r['status'] == 'partial' for r in result['endpoints'])
    assert 'PARTIAL' in campaign.endpoint_figure(result['endpoints'], metric='loss', partial=True).texts[0].get_text()


@pytest.mark.parametrize('source', ['latest.pt', 'run_summary.json', 'terminal_validation_results.json', 'optimizer_ownership_trace.jsonl', 'metrics.csv'])
def test_report_rejects_replaced_frozen_source(tmp_path, terminal_campaign, source):
    manifest, root = terminal_campaign
    campaign.freeze_campaign(campaign_manifest=manifest, run_root=root, output_dir=tmp_path / 'frozen')
    with (root / 'C3' / source).open('ab') as f: f.write(b' ')
    with pytest.raises(ConfigError, match='source.*hash'):
        campaign.report_campaign(manifest=tmp_path / 'frozen/frozen_manifest.json', output_dir=tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


@pytest.mark.parametrize('mutation', ['summary_control', 'summary_tokens', 'checkpoint_step', 'checkpoint_purpose', 'actions', 'epochs'])
def test_freeze_rejects_controls_checkpoint_and_trace(tmp_path, terminal_campaign, monkeypatch, mutation):
    manifest, root = terminal_campaign
    if mutation.startswith('summary'):
        path = root / 'C3/run_summary.json'; value = json.loads(path.read_text())
        if mutation == 'summary_control': value['optimizer_ownership']['contract']['optimizer']['learning_rate'] = .1
        else: value['optimizer_ownership']['tokens_seen'] -= 8192
        path.write_text(json.dumps(value))
    elif mutation.startswith('checkpoint'):
        path = root / 'C3/latest.pt'; value = torch.load(path, weights_only=False)
        value['step' if mutation == 'checkpoint_step' else 'checkpoint_kind'] = 87132 if mutation == 'checkpoint_step' else 'model_only'
        torch.save(value, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        change_terminal(root, 'C3', lambda d: d.update(checkpoint_sha256=digest, checkpoint_bytes=path.stat().st_size))
        summary = root / 'C3/run_summary.json'; value = json.loads(summary.read_text())
        value['optimizer_ownership']['checkpoint'].update(terminal_checkpoint_sha256=digest, terminal_checkpoint_bytes=path.stat().st_size)
        summary.write_text(json.dumps(value))
    else:
        original = campaign.inspect_run_observations
        def bad_trace(path, summary):
            result = original(path, summary)
            if summary['optimizer_ownership']['arm_id'] == 'C3':
                if mutation == 'actions': result['action_sha256'] = 'changed'
                else: result['epoch_order_sha256']['3'] = 'changed'
            return result
        monkeypatch.setattr(campaign, 'inspect_run_observations', bad_trace)
    with pytest.raises(ConfigError):
        campaign.freeze_campaign(campaign_manifest=manifest, run_root=root, output_dir=tmp_path / 'bad', allow_partial=True)
    assert not (tmp_path / 'bad').exists()


def test_freeze_duplicate_run_dirs_and_publication_failure(tmp_path, terminal_campaign, monkeypatch):
    manifest, root = terminal_campaign
    with pytest.raises(ConfigError, match='Duplicate'):
        campaign.freeze_campaign(campaign_manifest=manifest, run_dirs=[root / 'C3']*2, output_dir=tmp_path / 'bad', allow_partial=True)
    rename = Path.rename
    def fail(path, target):
        if target == tmp_path / 'frozen': raise OSError('injected publication failure')
        return rename(path, target)
    monkeypatch.setattr(Path, 'rename', fail)
    with pytest.raises(OSError, match='publication failure'):
        campaign.freeze_campaign(campaign_manifest=manifest, run_root=root, output_dir=tmp_path / 'frozen')
    assert not (tmp_path / 'frozen').exists() and not list(tmp_path.glob('.frozen-*'))


def test_report_stages_exports_and_preserves_sources(tmp_path, terminal_campaign, monkeypatch):
    from matplotlib.figure import Figure
    manifest, root = terminal_campaign
    campaign.freeze_campaign(campaign_manifest=manifest, run_root=root, output_dir=tmp_path / 'frozen')
    path = tmp_path / 'frozen/frozen_manifest.json'
    before = path.read_bytes()
    def fail(*args, **kwargs): raise OSError('injected figure export failure')
    monkeypatch.setattr(Figure, 'savefig', fail)
    with pytest.raises(OSError, match='figure export'):
        campaign.report_campaign(manifest=path, output_dir=tmp_path / 'bad')
    assert path.read_bytes() == before
    assert not (tmp_path / 'bad').exists() and not list(tmp_path.glob('.bad-*'))


def test_report_detects_new_resource_attempt_after_freeze(tmp_path, terminal_campaign):
    manifest, root = terminal_campaign
    campaign.freeze_campaign(campaign_manifest=manifest, run_root=root, output_dir=tmp_path / 'frozen')
    (root / 'C3/resource_attempts.json').write_text('{}')
    with pytest.raises(ConfigError, match='source presence'):
        campaign.report_campaign(manifest=tmp_path / 'frozen/frozen_manifest.json', output_dir=tmp_path / 'bad')


@pytest.mark.parametrize('arm', ARMS)
def test_terminal_reader_consumes_real_short_run_artifacts(tmp_path, monkeypatch, arm):
    """Exercise reader/runtime schema integration with real traces and checkpoints.

    Only fixed campaign scale/count expectations are reduced for this diagnostic;
    no terminal reader, trace reader, checkpoint loader or evaluator is stubbed.
    """
    from src.training.run import complete_ownership_terminal, build_ownership_run_summary
    from src.utils.metrics import append_optimizer_ownership_observation, write_json_artifact
    from src.utils.model_size import model_parameter_counts
    bundle = packed_fixture(tmp_path, arm)
    config, model, opt, clock, batches, state = bundle
    source_width = arm[3:] if arm.startswith('ST-') else None
    contract = config['optimizer_ownership_contract']
    contract.update(representation='dense' if source_width else config['model']['variant'],
        evaluation=config['evaluation'], initialization=None, clipping=config['training']['gradient_clipping'])
    config['optimizer_ownership_contract_hash'] = campaign.stable_hash(contract)
    config['validation_manifest_hash'] = 'ordinary-fixture'
    config['run']['continuation']['enabled'] = True
    opt._ownership_observer = lambda: append_optimizer_ownership_observation(config, state, train_dataloader=batches)
    expected = {'epochs': campaign.expected_epoch_traces(batches.batch_sampler, epochs=4),
                'actions': None if source_width else campaign.expected_action_trace(config)}
    train(bundle)
    evaluation = [{'input_ids': torch.arange(1,9).reshape(1,8), 'labels': torch.arange(1,9).reshape(1,8)}]
    terminal = complete_ownership_terminal(config, model, opt, clock, state, evaluation, torch.device('cpu'))
    root = Path(config['run']['output_dir'])
    write_json_artifact(root / 'run_summary.json', {'run_id': config['run']['run_id'], 'status': 'completed',
        **build_ownership_run_summary(config, model, opt, state)})
    (root / 'metrics.csv').write_text('step,split,granularity,loss,perplexity\n')
    widths = [source_width] if source_width else list(campaign.WIDTH_LABELS)
    monkeypatch.setattr(campaign, 'WIDTHS', tuple({**w, 'non_embedding_parameters': model_parameter_counts(model, granularity='g1000' if source_width else w['label'])['non_embedding_parameters']} for w in campaign.WIDTHS))
    monkeypatch.setattr(campaign, 'TOKENS_PER_UPDATE', 8)
    monkeypatch.setattr(campaign, 'VALIDATION_SEQUENCES', 1)
    monkeypatch.setattr(campaign, 'VALIDATION_TARGET_TOKENS', 7)
    monkeypatch.setitem(campaign.PINNED_DATA, 'ordinary_validation_manifest_hash', 'ordinary-fixture')
    run = {k: terminal[k] for k in ('campaign_id', 'arm_id', 'run_id', 'contract_hash', 'representation', 'state_scope', 'clipping', 'initialization', 'assigned_updates', 'assigned_tokens', 'assigned_epochs')}
    run.update(optimizer_ownership_contract=contract, resolved_config=config, endpoint_widths=widths, source_width=source_width)
    inspected = campaign._inspect_terminal_run(root, run, expected, allow_partial=False)
    assert len(inspected['endpoints']) == len(widths)
    assert inspected['observations']['committed_updates'] == 8


def test_rehashed_resolved_control_cannot_override_pinned_recipe(tmp_path, terminal_campaign):
    manifest, root = terminal_campaign
    value = json.loads(manifest.read_text())
    run = value['runs'][-1]
    contract = run['optimizer_ownership_contract']
    contract['optimizer']['learning_rate'] = .1
    digest = campaign.stable_hash(contract)
    run['contract_hash'] = digest
    value['run_contract_hashes']['C3'] = digest
    for field in ('resolved_config', 'executable_config'):
        run[field]['optimizer_ownership_contract'] = copy.deepcopy(contract)
        run[field]['optimizer_ownership_contract_hash'] = digest
    run['resolved_config']['training']['learning_rate'] = .1
    value['manifest_hash'] = campaign.stable_hash({k:v for k,v in value.items() if k != 'manifest_hash'})
    manifest.write_text(json.dumps(value))
    with pytest.raises(ConfigError, match='learning_rate'):
        campaign.freeze_campaign(campaign_manifest=manifest, run_root=root, output_dir=tmp_path / 'bad')
