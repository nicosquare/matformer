"""Pre-schema-5 compatibility baseline; external data reads use fixtures only."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from src.evaluation import optimizer_ownership as campaign
from src.utils.config import ConfigError, resolve_run_config
from src.utils.reproducibility import build_optimizer_ownership_signature, stable_hash
from test_optimizer_ownership_campaign import audited_inputs, expand


FIXTURES = Path(__file__).parent / "fixtures"
LEGACY = json.loads((FIXTURES / "optimizer_ownership_legacy_signatures.json").read_text())
SCHEMA4 = json.loads((FIXTURES / "s1_warmup_schema4_signatures.json").read_text())
RECIPES = {
    1: "optimizer_ownership",
    2: "optimizer_ownership_corrections",
    3: "inverse_membership",
    4: "matformer_widths",
}
CONTRACT_FIELDS = {
    "schema_version", "campaign_id", "run_id", "arm_id", "representation",
    "state_scope", "clipping", "initialization", "model", "optimizer",
    "sampling", "data", "budget", "evaluation", "count_convention",
}
TOPOLOGY_FIELDS = {"campaign_schema_version", "width_grid", "block_boundaries"}


@pytest.fixture(params=RECIPES, ids=lambda schema: f"schema{schema}")
def historical_campaign(request, tmp_path, monkeypatch, audited_inputs):
    schema = request.param
    baseline = SCHEMA4 if schema == 4 else LEGACY
    # Pin only volatile provenance. Keep resolution and serialization real.
    monkeypatch.setattr(campaign, "_provenance", lambda: copy.deepcopy(baseline["provenance"]))
    recipe = yaml.safe_load(Path(
        f"configs/controlled_exps/tinystories_instruct_{RECIPES[schema]}.yaml"
    ).read_text())
    return schema, recipe, expand(tmp_path, recipe)


def test_historical_resolved_signatures_and_serialized_fields(historical_campaign, tmp_path):
    schema, _, runs = historical_campaign
    expected_hashes = (
        {arm: values["contract_hash"] for arm, values in SCHEMA4["arms"].items()}
        if schema == 4 else LEGACY["contracts"][str(schema)]
    )
    assert {run["arm_id"]: run["contract_hash"] for run in runs} == expected_hashes
    for run in runs:
        config = run["resolved_config"]
        contract = run["optimizer_ownership_contract"]
        campaign.validate_materialized_config(config)
        assert config["training"]["warmup_steps"] == 64
        assert config["training"]["resolved_warmup_steps"] == 64
        expected_fields = CONTRACT_FIELDS.copy()
        if schema == 4:
            expected_fields |= TOPOLOGY_FIELDS
        elif schema == 2:
            expected_fields.add("correction")
        assert set(contract) == expected_fields
        assert ("campaign_schema_version" in config["run"]) is (schema == 4)
        before = copy.deepcopy(contract)
        digest, serialized = build_optimizer_ownership_signature(contract)
        assert (digest, serialized) == (expected_hashes[run["arm_id"]], before)
        assert contract == before
        assert serialized is not contract
        assert serialized["optimizer"] is not contract["optimizer"]
        # JSON/YAML round trips cannot introduce fields or alter identities.
        for restored in (json.loads(json.dumps(contract)), yaml.safe_load(yaml.safe_dump(contract))):
            assert build_optimizer_ownership_signature(restored) == (digest, before)
        if schema == 4:
            expected = SCHEMA4["arms"][run["arm_id"]]
            normalized = copy.deepcopy(config)
            for field, value in SCHEMA4["normalized_paths"].items():
                section, key = field.split(".")
                # Verify actual paths before replacing exactly four volatile locations.
                assert normalized[section][key] == value.replace("<fixture>", str(tmp_path)).replace(
                    "<arm_id>", run["arm_id"]
                )
                normalized[section][key] = value.replace("<arm_id>", run["arm_id"])
            assert stable_hash(normalized) == expected["resolved_config_hash"]
            assert sorted(config) == expected["resolved_top_level_fields"]
            assert sorted(contract) == expected["contract_fields"]
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("schema", RECIPES)
def test_historical_selectors_without_arm_keep_their_defaults(schema):
    # Schema 5 will require an arm; these calls must remain valid and unchanged.
    widths = campaign.campaign_widths(schema)
    common = campaign.campaign_common(schema)
    topology = campaign.campaign_topology(schema)
    if schema == 4:
        assert stable_hash(widths) == SCHEMA4["widths_hash"]
        assert stable_hash(common) == SCHEMA4["common_hash"]
        assert stable_hash(topology) == SCHEMA4["topology_hash"]
    else:
        assert widths == campaign.WIDTHS
        assert stable_hash(common) == LEGACY["common_hash"]
        assert topology == {}
    assert campaign.campaign_arms() is campaign.ARMS
    assert campaign.campaign_widths() is campaign.WIDTHS
    assert stable_hash(campaign.campaign_common()) == LEGACY["common_hash"]
    assert campaign.campaign_topology() == {}


@pytest.mark.parametrize("location", ["common", "arm"])
def test_historical_recipe_rejects_warmup256(historical_campaign, tmp_path, location):
    _, recipe, runs = historical_campaign
    # Exercise every arm override, including standalone and corrected variants.
    arm_ids = [run["arm_id"] for run in runs] if location == "arm" else [None]
    for arm_id in arm_ids:
        changed = copy.deepcopy(recipe)
        target = changed["common"] if arm_id is None else changed["arms"][arm_id]
        target.setdefault("training", {})["warmup_steps"] = 256
        with pytest.raises(ConfigError, match="warmup_steps"):
            expand(tmp_path, changed)
    assert not (tmp_path / "runs").exists()


def test_historical_materialized_config_rejects_rehashed_warmup256(historical_campaign, tmp_path):
    schema, _, runs = historical_campaign
    for run in runs:
        changed = copy.deepcopy(run["resolved_config"])
        changed["training"].update(warmup_steps=256, resolved_warmup_steps=256)
        # Rehash the changed contract so rejection proves the fixed budget rule,
        # rather than merely detecting a stale hash or optimizer projection.
        changed["optimizer_ownership_contract"]["optimizer"] = copy.deepcopy(changed["training"])
        changed["optimizer_ownership_contract_hash"] = build_optimizer_ownership_signature(
            changed["optimizer_ownership_contract"]
        )[0]
        arm = campaign.campaign_arm(schema, run["arm_id"])
        with pytest.raises(ConfigError, match="resolved_warmup_steps"):
            campaign.validate_run_budget(changed, arm)
        with pytest.raises(ConfigError, match="resolved_warmup_steps"):
            campaign.validate_materialized_config(changed)

        raw = copy.deepcopy(run["executable_config"])
        raw["training"]["warmup_steps"] = 256
        raw["optimizer_ownership_contract"] = changed["optimizer_ownership_contract"]
        raw["optimizer_ownership_contract_hash"] = changed["optimizer_ownership_contract_hash"]
        path = tmp_path / f"warmup256-{run['arm_id']}.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ConfigError, match="resolved_warmup_steps"):
            resolve_run_config(path, create_output_dirs=False)
    assert not (tmp_path / "runs").exists()

WARMUP_RECIPE = Path('configs/controlled_exps/tinystories_instruct_s1_warmup.yaml')
NEW_ARMS = ('S1-linear-w256', 'S1-geometric-w256')


@pytest.fixture
def warmup_runs(tmp_path, audited_inputs):
    return expand(tmp_path, yaml.safe_load(WARMUP_RECIPE.read_text()))


def test_two_fixed_protocols_and_physical_models(warmup_runs):
    assert tuple(r['arm_id'] for r in warmup_runs) == NEW_ARMS
    checks = campaign.inspect_campaign_models(warmup_runs)
    for run, check, dimensions, counts in zip(warmup_runs, checks,
            ([64, 128, 192, 256], [32, 64, 128, 256]),
            ([115264, 164416, 213568, 262720], [90688, 115264, 164416, 262720])):
        arm = run['arm_id']; config = run['resolved_config']
        assert [w['active_ffn_dimension'] for w in campaign.campaign_widths(5, arm)] == dimensions
        assert list(check['counts'].values()) == counts
        assert config['training']['resolved_warmup_steps'] == 256
        assert run['assigned_updates'] == 348528 and run['assigned_tokens'] == 2855141376
        assert run['initialization']['method'] == 'fresh_normal_constructor'
        assert run['initialization']['seed'] == 42
        assert config['dataset']['optimizer_iteration']['excluded_tail_samples'] == 43
        assert run['run_id'] == f'tinystories-optimizer-ownership-s1-warmup-v1-{arm}-s42'
        campaign.validate_materialized_config(config)
    assert warmup_runs[0]['initialization']['derived_seed'] == warmup_runs[1]['initialization']['derived_seed']


@pytest.mark.parametrize('arm', NEW_ARMS)
def test_gpu_diagnostic_identity_resolves_without_relaxing_protocol(tmp_path, warmup_runs, arm):
    from scripts.preflight_tinystories_s1_warmup import diagnostic_identity
    definition = next(run for run in warmup_runs if run['arm_id'] == arm)
    run_id = 's1w-gpu-diagnostic-test-' + arm
    output = tmp_path/'diagnostics'/run_id
    raw = copy.deepcopy(definition['executable_config'])
    raw['run'].update(run_id=run_id, output_dir=str(output))
    raw['optimizer_ownership_contract']['run_id'] = run_id
    raw['optimizer_ownership_contract_hash'] = stable_hash(raw['optimizer_ownership_contract'])
    path = tmp_path/'diagnostic.yaml'
    path.write_text(yaml.safe_dump(raw))
    original_budget = campaign.validate_run_budget
    original_materialized = campaign.validate_materialized_config
    with pytest.raises(ConfigError):
        resolve_run_config(path, create_output_dirs=False)
    with diagnostic_identity(definition, output, run_id):
        resolved = resolve_run_config(path, create_output_dirs=False)
        assert resolved['run']['run_id'] == run_id
        assert Path(resolved['run']['output_dir']) == output
        campaign.validate_run_budget(resolved, campaign.campaign_arm(5, arm))
        campaign.validate_materialized_config(resolved)
        for field, value in [('warmup_steps', 64), ('learning_rate', .004)]:
            changed = copy.deepcopy(raw); changed['training'][field] = value
            path.write_text(yaml.safe_dump(changed))
            with pytest.raises(ConfigError):
                resolve_run_config(path, create_output_dirs=False)
        for field, value in [('run_id', 'unrelated'), ('output_dir', str(tmp_path/'unrelated'))]:
            changed = copy.deepcopy(raw); changed['run'][field] = value
            path.write_text(yaml.safe_dump(changed))
            with pytest.raises(ConfigError):
                resolve_run_config(path, create_output_dirs=False)
        changed = copy.deepcopy(raw); changed['optimizer_ownership_contract_hash'] = 'stale'
        path.write_text(yaml.safe_dump(changed))
        with pytest.raises(ConfigError):
            resolve_run_config(path, create_output_dirs=False)
    assert campaign.validate_run_budget is original_budget
    assert campaign.validate_materialized_config is original_materialized
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError):
        resolve_run_config(path, create_output_dirs=False)


@pytest.mark.parametrize('selector', ['campaign_widths', 'campaign_common', 'campaign_topology'])
@pytest.mark.parametrize('arm', [None, 'S1', 'unknown'])
def test_schema5_requires_qualified_grid(selector, arm):
    with pytest.raises(ConfigError):
        getattr(campaign, selector)(5, arm)


@pytest.mark.parametrize('path,value', [
    ('training.warmup_steps', 64), ('training.learning_rate', .004),
    ('training.token_budget', 713785344), ('training.mixed_precision', 'fp32'),
    ('training.gradient_clip_norm', 2), ('training.pre_nested_warmup.enabled', True),
    ('run.seed', 43), ('run.reproducibility.seed_stream_version', 2),
    ('dataset.data_seed', 43), ('model.global_sampling_interval_steps', 2),
    ('model.correction_mode', 'gmc'), ('model.initializer_range', .01),
    ('model.checkpoint_path', '/reference/weights.pt'),
])
def test_schema5_rejects_changed_controls(tmp_path, audited_inputs, path, value):
    recipe = yaml.safe_load(WARMUP_RECIPE.read_text())
    target = recipe['arms'][NEW_ARMS[0]]
    keys = path.split('.')
    for key in keys[:-1]: target = target.setdefault(key, {})
    target[keys[-1]] = value
    with pytest.raises(ConfigError): expand(tmp_path, recipe)
    assert not (tmp_path/'runs').exists()


@pytest.mark.parametrize('mutation', ['extra', 'identity', 'reference', 'occupied'])
def test_schema5_rejects_other_identities(tmp_path, audited_inputs, mutation):
    recipe = yaml.safe_load(WARMUP_RECIPE.read_text())
    if mutation == 'extra': recipe['arms']['S2'] = recipe['arms'][NEW_ARMS[0]]
    if mutation == 'identity': recipe['campaign_id'] += '-other'
    if mutation == 'reference': recipe['references']['linear']['s1_config_sha256'] = 'bad'
    if mutation == 'occupied': (tmp_path/'runs'/NEW_ARMS[0]).mkdir(parents=True)
    with pytest.raises(ConfigError): expand(tmp_path, recipe)


def test_closed_counterpart_audits_and_complete_actions(tmp_path, audited_inputs, warmup_runs):
    from src.utils.reproducibility import seed_for
    for run, suffix in zip(warmup_runs, ('optimizer_ownership', 'matformer_widths')):
        recipe = yaml.safe_load(Path(f'configs/controlled_exps/tinystories_instruct_{suffix}.yaml').read_text())
        old = next(r for r in expand(tmp_path, recipe) if r['arm_id'] == 'S1')
        audit = campaign.audit_warmup_counterpart(run['resolved_config'], old['resolved_config'])
        assert audit['status'] == 'passed'
        assert 'training.warmup_steps' in {d['path'] for d in audit['differences']}
        assert audit['old_projection'] and audit['new_projection']
        actions = campaign.expected_action_trace(run['resolved_config'])
        assert actions == campaign.expected_action_trace(old['resolved_config'])
        assert actions['updates'] == 348528
        assert set(actions['expected_counts'].values()) == {87132}
        assert set(actions['counts'].values()) != {87132}
        for stream in ('model_initialization', 'granularity_selection'):
            assert seed_for(run['resolved_config'], stream) == seed_for(old['resolved_config'], stream)
        for section, key in [('training', 'resolved_learning_rate'), ('model', 'initializer_range'),
                             ('dataset', 'data_seed'), ('evaluation', 'test')]:
            changed = copy.deepcopy(old['resolved_config']); changed[section][key] = 'tampered'
            with pytest.raises(ConfigError, match=key):
                campaign.audit_warmup_counterpart(run['resolved_config'], changed)
        changed = copy.deepcopy(old['resolved_config']); changed['model']['unknown_science'] = 1
        with pytest.raises(ConfigError, match='unknown_science'):
            campaign.audit_warmup_counterpart(run['resolved_config'], changed)


def test_schema5_rehashed_materialized_controls_rejected(warmup_runs):
    for run in warmup_runs:
        for key, value in [('warmup_steps', 64), ('resolved_warmup_steps', 64), ('learning_rate', .001)]:
            config = copy.deepcopy(run['resolved_config'])
            config['training'][key] = value
            config['optimizer_ownership_contract']['optimizer'] = copy.deepcopy(config['training'])
            config['optimizer_ownership_contract_hash'] = stable_hash(config['optimizer_ownership_contract'])
            with pytest.raises(ConfigError, match=key): campaign.validate_materialized_config(config)


def test_bounded_full_schedule_export(tmp_path, warmup_runs):
    import csv
    import math
    hashes = []
    for run in warmup_runs:
        path = tmp_path/f"{run['arm_id']}.csv"
        evidence = campaign.export_expected_schedule(run['resolved_config'], path)
        hashes.append(evidence['array_sha256'])
        assert evidence['positions'] == 348529
        assert evidence['dependency_versions'] and evidence['scheduler_source_sha256']
        with path.open() as stream:
            for p, row in enumerate(csv.DictReader(stream)):
                expected = .008*p/256 if p < 256 else .004*(1+math.cos(math.pi*(p-256)/(348528-256)))
                assert int(row['scheduler_position']) == p
                assert float(row['learning_rate']) == expected
                assert row['applied_update'] == (str(p+1) if p < 348528 else '')
            assert p == 348528
    assert hashes[0] == hashes[1]


def test_full_four_epoch_batch_and_action_pairing(tmp_path, warmup_runs, monkeypatch):
    import numpy as np
    from src.training import packed_corpus
    # Full cardinality, synthetic initial order: this is not the real corpus audit.
    order = np.arange(5576491, dtype='<u8'); (tmp_path/'corpus').mkdir()
    order.tofile(tmp_path/'corpus/order.bin')
    import hashlib
    digest = hashlib.sha256(order.tobytes()).hexdigest()
    runs = copy.deepcopy(warmup_runs)
    for run in runs:
        run['resolved_config']['dataset']['optimizer_iteration']['permutation_hash'] = digest
    traces = campaign.build_expected_traces(runs, tmp_path/'corpus', {'training_order': {'path': 'order.bin'}})
    assert traces[NEW_ARMS[0]]['epochs'] == traces[NEW_ARMS[1]]['epochs']
    for run in runs:
        epochs = traces[run['arm_id']]['epochs']
        assert len(epochs) == 4
        assert all(e['sequences'] == 5576448 and e['updates'] == 87132 for e in epochs)
        assert len({e['sha256'] for e in epochs}) == 4
        assert len({e['fixed_epoch_set_hash'] for e in epochs}) == 1
        reference = copy.deepcopy(run)
        reference['arm_id'] = 'S1'
        reference['resolved_config']['training']['warmup_steps'] = 64
        own = campaign.build_expected_traces([reference], tmp_path/'corpus', {'training_order': {'path': 'order.bin'}})
        assert traces[run['arm_id']] == own['S1']


def test_materialized_yaml_without_contract_cannot_bypass_protocol(tmp_path, audited_inputs, warmup_runs):
    raw=copy.deepcopy(warmup_runs[0]['executable_config'])
    raw.pop('optimizer_ownership_contract');raw.pop('optimizer_ownership_contract_hash')
    raw['training']['warmup_steps']=64
    path=tmp_path/'bypass.yaml';path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError,match='warmup_steps'): resolve_run_config(path,create_output_dirs=False)


def test_older_linear_absent_disabled_diagnostics_are_explicit(tmp_path, audited_inputs, warmup_runs):
    recipe=yaml.safe_load(Path('configs/controlled_exps/tinystories_instruct_optimizer_ownership.yaml').read_text())
    old=next(r['resolved_config'] for r in expand(tmp_path,recipe) if r['arm_id']=='S1')
    old['evaluation'].pop('sign_dynamics')
    old['optimizer_ownership_contract']['evaluation'].pop('sign_dynamics')
    old['optimizer_ownership_contract_hash']=stable_hash(old['optimizer_ownership_contract'])
    audit=campaign.audit_warmup_counterpart(warmup_runs[0]['resolved_config'],old)
    assert 'evaluation.sign_dynamics.enabled' in audit['allowed_paths']
    changed=copy.deepcopy(warmup_runs[0]['resolved_config'])
    changed['evaluation']['sign_dynamics']['enabled']=True
    changed['optimizer_ownership_contract']['evaluation']=copy.deepcopy(changed['evaluation'])
    changed['optimizer_ownership_contract_hash']=stable_hash(changed['optimizer_ownership_contract'])
    with pytest.raises(ConfigError,match='sign_dynamics.enabled'): campaign.audit_warmup_counterpart(changed,old)


# These runtime fixtures retain the real horizon and epoch boundaries. CPU uses
# short synthetic batches; GPU readiness uses batch64/context128/vocab2048.
def warmup_runtime(tmp_path, arm, *, device=None):
    import os
    import torch
    from src.training import checkpointing as cp, steps
    from src.training.modeling import build_model
    from src.training.packed_corpus import RepeatingNoPaddingDistributedBatchSampler, REPEATED_EPOCH_ORDER_VERSION
    from src.utils.reproducibility import seed_training_randomness
    from test_optimizer_ownership import runtime_fixture
    device = device or os.environ.get('S1_WARMUP_DIAGNOSTIC_DEVICE', 'cpu')
    config, _, _, _, _, _ = runtime_fixture(tmp_path, scope='shared', max_steps=348528, device=device)
    batch, context, vocab = (64, 128, 2048) if device == 'cuda' else (1, 8, 32)
    widths = campaign.campaign_widths(5, arm)
    labels = [w['label'] for w in widths]
    config['run'].update(run_id=f'warmup-diagnostic-{arm}', arm_id=arm,
        campaign_id=campaign.WARMUP_CAMPAIGN_ID, campaign_schema_version=5,
        output_dir=str(tmp_path/f'warmup-diagnostic-{arm}'))
    config['model'].update(variant='slicing', d_model=64, num_layers=4, num_attention_heads=4,
        intermediate_size=256, vocab_size=vocab, context_length=context,
        granularities=labels, granularity_prefixes={w['label']: w['source_fraction'] for w in widths})
    for key in ('ffn_prefix_metadata', 'ffn_concat_block_metadata', 'matformer_source_granularity_prefixes'):
        config['model'].pop(key, None)
    training = config['training']
    training.update(warmup_steps=256, resolved_warmup_steps=256, learning_rate=.008,
        resolved_learning_rate=.008, batch_size_per_process=batch,
        expected_tokens_per_step=batch*context, token_budget=348528*batch*context,
        optimizer_state_topology=campaign.campaign_topology(5, arm))
    training['optimizer_state_contract']['ordered_granularities'] = labels
    # Retain all resolver fields, changing only the explicit diagnostic schedule.
    for schedule in (training['scheduler'], training['optimizer_state_contract']['scheduler_contract'],
                     training.get('scheduler_contract', {})):
        if schedule:
            schedule['resolved_warmup_steps'] = 256
            schedule['kwargs']['warmup_steps'] = 256
    config['optimizer_ownership_contract'] = dict(schema_version=1, campaign_id=campaign.WARMUP_CAMPAIGN_ID,
        run_id=config['run']['run_id'], arm_id=arm, representation='slicing', state_scope='shared',
        **campaign.campaign_topology(5, arm), intervention=campaign.warmup_intervention(arm))
    config['optimizer_ownership_contract'] = json.loads(json.dumps(config['optimizer_ownership_contract']))
    config['optimizer_ownership_contract_hash'] = stable_hash(config['optimizer_ownership_contract'])
    config['dataset'].update(mode='packed_mmap', data_seed=42, optimizer_iteration=dict(
        mode='repeat_epochs', epoch_order='deterministic_per_epoch', ordering_policy_version=REPEATED_EPOCH_ORDER_VERSION,
        aligned_epoch_samples=87132*batch, aligned_epoch_tokens=87132*batch*context,
        excluded_tail_samples=43, excluded_tail_tokens=43*context))
    class SyntheticTokens:
        def __len__(self): return 87132*batch+43
        def __getitem__(self, index):
            tokens = (torch.arange(context) + index) % vocab
            return {'input_ids': tokens, 'labels': tokens.clone()}
    sampler = RepeatingNoPaddingDistributedBatchSampler(len(SyntheticTokens()), batch, 0, 1,
        planned_sample_count=348528*batch, epoch_sample_count=87132*batch,
        corpus_hash='synthetic-history-diagnostic', optimizer_training_manifest_hash='diagnostic-role')
    loader = torch.utils.data.DataLoader(SyntheticTokens(), batch_sampler=sampler)
    from src.utils.config import resolve_training_length_for_world_size
    resolve_training_length_for_world_size(config, effective_world_size=1, world_size_source='single_process')
    seed_training_randomness(config)
    model = build_model(config).to(device)
    optimizer, scheduler = steps.build_optimizer_and_scheduler(model, training)
    return config, model, optimizer, scheduler, loader, cp.build_initial_continuation_state(config)


def seed_synthetic_history(bundle, position):
    """State-seeded diagnostic, never evidence of executing the preceding steps."""
    import random
    from src.utils.reproducibility import dedicated_random, seed_for
    from test_optimizer_ownership_resume import train
    config, model, optimizer, scheduler, loader, state = bundle
    train(bundle, stop=1)  # Allocate real full-tensor shared history.
    widths = config['model']['granularities']
    rng = random.Random(seed_for(config, 'granularity_selection'))
    counts = dict.fromkeys(widths, 0)
    for _ in range(position):
        width = widths[rng.randrange(4)]; counts[width] += 1
    dedicated_random(config, 'granularity_selection').setstate(rng.getstate())
    sampling = state['global_sampling_state']
    sampling.update(held_granularity=width, window_index=position-1, successful_updates_in_window=1,
        total_successful_updates=position, exposure_counts=counts)
    rate = scheduler.base_lrs[0] * scheduler.lr_lambdas[0](position)
    saved_clock = scheduler.state_dict()
    saved_clock.update(last_epoch=position, _step_count=position+1, _last_lr=[rate])
    scheduler.load_state_dict(saved_clock)
    for group in optimizer.param_groups: group['lr'] = rate
    for history in optimizer.state.values(): history['step'].fill_(position)
    loader.batch_sampler.set_cursor(position*config['training']['batch_size_per_process'])
    sampler = loader.batch_sampler.state_dict()
    tokens = position*config['training']['expected_tokens_per_step']
    state.update(step=position, last_completed_step=position, microstep=position,
        tokens_seen=tokens, content_tokens_seen=tokens, epoch=sampler['epoch'],
        batch_index=sampler['within_epoch_cursor']//config['training']['batch_size_per_process'],
        sampler_state=sampler, optimizer_width_selection_counts=counts,
        optimizer_quarter_activation_counts={f'O-{q}': sum(counts[w] for w in widths[i:]) for i,q in enumerate('ABCD')},
        optimizer_update_counts={'shared': position}, optimizer_total_successful_updates=position,
        global_scheduler_position=position, optimizer_last_active_granularity=width)
    from src.utils.metrics import StreamingMetricsAccumulator
    accounting = StreamingMetricsAccumulator(ordered_attempts=True, campaign_contract=config['optimizer_ownership_contract']).state_dict()
    accounting.update(last_training_step=position, tokens_seen=tokens, content_tokens_seen=tokens,
        training_row_count=position, selection_counts=counts.copy(), attempted_optimizer_steps=position,
        committed_optimizer_steps=position, optimizer_last_attempt_id=f'global:{position-1}:-:-:{width}')
    state['metrics_accumulator_state'] = accounting


def warmup_window(bundle, stop, trace=None):
    from src.training import steps, data
    from src.utils.metrics import MetricsJournal
    config, model, optimizer, scheduler, loader, state = bundle
    if state.get('sampler_state') is not None:
        data.restore_packed_sampler_state(loader, state['sampler_state'])
    journal = MetricsJournal(config['run']['output_dir'], artifact_state=state, artifact_io_config=config,
                             checkpoint_step=state['last_completed_step'])
    def committed(**kw):
        if trace is not None:
            trace.append((kw['step'], copy.deepcopy(state['global_sampling_state']), copy.deepcopy(state['sampler_state'])))
        if kw['step'] == stop: raise StopIteration('diagnostic boundary')
    try:
        steps.train_for_steps(config, model, loader, [], optimizer, scheduler, next(model.parameters()).device,
            run_state=state, metrics_journal=journal, successful_step_callback=committed)
    except StopIteration:
        pass
    journal.flush()
    return journal


@pytest.mark.parametrize('arm', NEW_ARMS)
def test_warmup_runtime_applies_full_horizon_schedule(tmp_path, monkeypatch, arm):
    import math
    from src.training import steps
    from test_optimizer_ownership_resume import train
    bundle = warmup_runtime(tmp_path, arm)
    config, _, opt, scheduler, _, state = bundle
    for p in range(348529):
        expected = .008*p/256 if p < 256 else .004*(1+math.cos(math.pi*(p-256)/(348528-256)))
        assert math.isclose(scheduler.base_lrs[0]*scheduler.lr_lambdas[0](p), expected, rel_tol=1e-12, abs_tol=1e-18)
    applied, advances = [], []
    original_step, original_tick = opt.step, scheduler.step
    def step(*args, **kwargs):
        applied.append((state['last_completed_step']+1, opt.param_groups[0]['lr']))
        return original_step(*args, **kwargs)
    def tick(*args, **kwargs):
        assert len(applied) == len(advances)+1
        original_tick(*args, **kwargs); advances.append(scheduler.last_epoch)
    step._wrapped_by_lr_sched = True  # The spy delegates to the already wrapped optimizer.
    monkeypatch.setattr(opt, 'step', step); monkeypatch.setattr(scheduler, 'step', tick)
    warmup_window(bundle, 258)
    assert advances == list(range(1, 259))
    values = dict(applied)
    assert values[1] == 0 and values[64] == .00196875 and values[65] == .002
    assert values[256] == .00796875 and values[257] == .008
    assert values[258] < .008
    import csv
    with (Path(config['run']['output_dir'])/'metrics.csv').open() as stream:
        rows = [row for row in csv.DictReader(stream) if row['split']=='train']
    assert {int(row['step']):float(row['learning_rate']) for row in rows} == values
    assert all(row['optimizer_step_committed']=='True' for row in rows)
    assert state['last_clipping_observation']['combined_post_norm'] <= 1.000001
    assert all(v > 0 for v in state['optimizer_width_selection_counts'].values())


@pytest.mark.parametrize('arm', NEW_ARMS)
@pytest.mark.parametrize('boundary', [255, 256, 257, 87131, 87132, 87133, 174263, 174264, 174265, 261395, 261396, 261397, 348527])
def test_warmup_runtime_seeded_resume_boundaries(tmp_path, arm, boundary):
    from test_optimizer_ownership_resume import train, save, load
    from test_optimizer_ownership import assert_state_equal
    from src.utils.reproducibility import capture_rng_state
    source = warmup_runtime(tmp_path, arm)
    seed_synthetic_history(source, boundary)
    path = tmp_path/'synthetic-history.pt'; save(source, path)
    trace = []; stop = min(boundary+2, 348528)
    batches, resumed_batches = [], []
    source[1].register_forward_pre_hook(lambda m,a,k: batches.append(k['input_ids'].clone()), with_kwargs=True)
    warmup_window(source, stop, trace); rng = capture_rng_state()
    target = warmup_runtime(tmp_path, arm); load(target, path)
    target[1].register_forward_pre_hook(lambda m,a,k: resumed_batches.append(k['input_ids'].clone()), with_kwargs=True)
    resumed = []; warmup_window(target, stop, resumed)
    assert resumed == trace
    assert_state_equal(batches, resumed_batches)
    for left, right in zip(source[1:4], target[1:4]): assert_state_equal(left.state_dict(), right.state_dict())
    for key in ('global_sampling_state', 'sampler_state', 'tokens_seen', 'epoch', 'batch_index',
                'optimizer_width_selection_counts', 'optimizer_update_counts', 'metrics_accumulator_state'):
        assert_state_equal(source[-1].get(key), target[-1].get(key))
    assert_state_equal(capture_rng_state(), rng)
    if stop == 348528:
        assert source[3].get_last_lr() == [0.]
        assert source[-1]['epoch'] == 4 and source[-1]['batch_index'] == 0
        import csv
        with (Path(source[0]['run']['output_dir'])/'metrics.csv').open() as stream:
            last = list(csv.DictReader(stream))[-1]
        assert int(last['step']) == 348528
        assert float(last['learning_rate']) == source[3].base_lrs[0]*source[3].lr_lambdas[0](348527)
        assert float(last['learning_rate']) > 0


@pytest.mark.parametrize('arm', NEW_ARMS)
def test_warmup_runtime_shared_tail_and_grid_summary(tmp_path, arm):
    import torch
    from src.training import steps
    from src.training.run import build_ownership_run_summary
    from src.utils.metrics import StreamingMetricsAccumulator
    from test_optimizer_ownership_resume import train
    bundle = warmup_runtime(tmp_path, arm)
    config, model, optimizer, scheduler, loader, state = bundle
    train(bundle, stop=8)
    summary = build_ownership_run_summary(config, model, optimizer, state)['optimizer_ownership']
    assert summary['grid_id'] == campaign.campaign_arm(5, arm)['grid_id']
    assert set(summary['expected_exposure']['width_selections']) == set(config['model']['granularities'])
    accumulator = StreamingMetricsAccumulator(ordered_attempts=True, campaign_contract=config['optimizer_ownership_contract'])
    assert accumulator.attempt_widths == tuple(config['model']['granularities'])
    parameter = model.model.layers[0].mlp.gate_proj.weight
    labels = config['model']['granularities']
    for width in reversed(labels):
        optimizer.zero_grad(set_to_none=True)
        from src.training.steps import configure_model_layer_granularities
        configure_model_layer_granularities(model, [width]*4)
        batch = next(iter(loader)); batch = {k:v.to(parameter.device) for k,v in batch.items()}
        with torch.autocast(parameter.device.type, dtype=torch.bfloat16, enabled=parameter.is_cuda):
            model(**batch).loss.backward()
        limit = next(w['active_ffn_dimension'] for w in campaign.campaign_widths(5,arm) if w['label']==width)
        before, momentum = parameter.detach().clone(), optimizer.state[parameter]['exp_avg'].clone()
        assert torch.count_nonzero(parameter.grad[limit:]) == 0
        clipping = steps.clip_optimizer_gradients(model, config['training'], width)
        assert clipping['combined_post_norm'] <= 1.000001
        rate = optimizer.param_groups[0]['lr']
        optimizer.step(); scheduler.step()
        if limit < 256:
            torch.testing.assert_close(optimizer.state[parameter]['exp_avg'][limit:], momentum[limit:]*.9, rtol=1e-6, atol=1e-7)
            assert not torch.equal(parameter[limit:], before[limit:])
            assert not torch.equal(parameter[limit:], before[limit:]*(1-rate*.1))


@pytest.mark.parametrize('arm', NEW_ARMS)
@pytest.mark.parametrize('damage', ['cross_run', 'grid', 'warmup64', 'horizon', 'model_only', 'nan', 'clock', 'cursor', 'rng', 'metrics'])
def test_warmup_runtime_rejects_before_mutation(tmp_path, arm, damage):
    import torch
    from test_optimizer_ownership_resume import train, save, load
    from test_optimizer_ownership import assert_state_equal
    from src.utils.reproducibility import capture_rng_state
    source = warmup_runtime(tmp_path, arm); train(source, stop=2)
    path = tmp_path/'bad.pt'; save(source, path); payload = torch.load(path, weights_only=False)
    if damage == 'cross_run': payload['run_id'] += '-other'
    elif damage == 'grid': payload['optimizer_ownership_contract']['grid_id'] = 'wrong'
    elif damage == 'warmup64': payload['optimizer_ownership_contract']['intervention']['warmup_steps'] = 64
    elif damage == 'horizon': payload['ownership_budget']['max_steps'] += 1
    elif damage == 'model_only': payload['checkpoint_kind'] = 'model_only_evaluation'
    elif damage == 'nan': next(iter(payload['optimizer_state_dict']['state'].values()))['exp_avg'].fill_(float('nan'))
    elif damage == 'clock': payload['scheduler_state_dict']['_last_lr'] = [.008]
    elif damage == 'cursor': payload['sampler_state']['total_cursor'] += 1
    elif damage == 'rng': payload['reproducibility']['rng_states_by_rank'][0]['python'] = ()
    else: payload['metrics_accumulator_state'] = {'bad': True}
    torch.save(payload, path)
    target = warmup_runtime(tmp_path, arm)
    before = [copy.deepcopy(x.state_dict()) for x in target[1:4]]
    rng, sampler = capture_rng_state(), target[4].batch_sampler.state_dict()
    with pytest.raises(ConfigError): load(target, path)
    for value, saved in zip(target[1:4], before): assert_state_equal(value.state_dict(), saved)
    assert target[4].batch_sampler.state_dict() == sampler
    assert_state_equal(capture_rng_state(), rng)


@pytest.mark.parametrize('arm', NEW_ARMS)
@pytest.mark.parametrize('failure', ['optimizer', 'scheduler', 'accounting'])
def test_warmup_runtime_partial_failure_preserves_durable(tmp_path, monkeypatch, arm, failure):
    import hashlib
    from src.training import checkpointing as cp, steps
    from test_optimizer_ownership_resume import train, save
    bundle = warmup_runtime(tmp_path, arm); train(bundle, stop=1)
    path = tmp_path/'latest.pt'; save(bundle, path); digest = hashlib.sha256(path.read_bytes()).hexdigest()
    target, method = (bundle[2], 'step') if failure == 'optimizer' else (bundle[3], 'step') if failure == 'scheduler' else (steps, '_commit_global_sampling_window_action')
    original = getattr(target, method)
    def fail(*args, **kwargs):
        original(*args, **kwargs); raise RuntimeError('injected partial update')
    monkeypatch.setattr(target, method, fail)
    with pytest.raises(RuntimeError, match='injected'): train(bundle, stop=2)
    assert bundle[-1]['optimizer_poisoned'] and bundle[-1]['update_in_flight']
    for reason in ('periodic', 'failure', 'signal', 'finalization'):
        with pytest.raises(ConfigError, match='unsafe|poison'):
            cp.maybe_write_latest_checkpoint(*bundle[:4], None, bundle[-1], reason=reason, step=2)
    with pytest.raises(ConfigError, match='unsafe|poison'): save(bundle, path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('arm', NEW_ARMS)
def test_warmup_runtime_terminal_only_recovery(tmp_path, monkeypatch, arm):
    import torch
    from src.training import run, checkpointing as cp
    from test_optimizer_ownership_resume import save, load
    from test_optimizer_ownership import assert_state_equal
    bundle = warmup_runtime(tmp_path, arm); seed_synthetic_history(bundle, 348528)
    config, model, optimizer, scheduler, loader, state = bundle
    config['run']['continuation']['enabled'] = True
    config['validation_manifest_hash'] = 'synthetic-ordinary-validation'
    checkpoint = Path(config['run']['output_dir'])/'checkpoints/latest.pt'
    save(bundle, checkpoint)
    state.update(latest_checkpoint_path=str(checkpoint), latest_checkpoint_step=348528)
    def forbidden(*args, **kwargs): raise AssertionError('terminal recovery cannot train')
    monkeypatch.setattr(optimizer, 'step', forbidden)
    monkeypatch.setattr(scheduler, 'step', forbidden)
    before = [copy.deepcopy(x.state_dict()) for x in bundle[1:4]]
    batch = loader.dataset[0]; evaluation = [{k:v.unsqueeze(0) for k,v in batch.items()}]
    result = run.complete_ownership_terminal(config, model, optimizer, scheduler, state, evaluation, next(model.parameters()).device)
    assert result['global_step'] == 348528 and len(result['endpoints']) == 4
    assert result['grid_id'] == campaign.campaign_arm(5,arm)['grid_id']
    assert result['evaluation_role'] == 'ordinary_validation'
    for value, expected in zip(bundle[1:4], before): assert_state_equal(value.state_dict(), expected)
    assert run.complete_ownership_terminal(config, model, optimizer, scheduler, state, evaluation, next(model.parameters()).device) == result


@pytest.mark.parametrize('arm', NEW_ARMS)
def test_warmup_runtime_trainer_reentry_zero_extra_updates(tmp_path, monkeypatch, arm):
    """Real orchestration from a synthetic-history full-horizon checkpoint."""
    import hashlib
    import torch
    from src.training import run
    from test_optimizer_ownership_resume import save
    bundle = warmup_runtime(tmp_path, arm); seed_synthetic_history(bundle, 348528)
    config, model, _, _, loader, state = bundle
    config['run']['continuation']['enabled'] = True
    config['_validation_manifest'] = {'fixture': True}
    config['validation_manifest_hash'] = 'fixture-validation'
    path = Path(config['run']['output_dir'])/'checkpoints/latest.pt'
    save(bundle, path); digest = hashlib.sha256(path.read_bytes()).hexdigest()
    batch = loader.dataset[0]
    evaluation = [{k:v.unsqueeze(0).repeat(config['training']['batch_size_per_process'],1) for k,v in batch.items()}]
    monkeypatch.setattr(campaign, 'validate_materialized_config', lambda c: None)
    # Explicit synthetic data/identity: production resolution is covered by
    # the protocol tests. Whole-bundle checkpoint validation stays enabled.
    monkeypatch.setattr(run, 'validate_run_config', lambda c: None)
    monkeypatch.setattr(run, '_uses_packed_mmap_corpus', lambda c: False)
    def dataloaders(current, *args, **kwargs):
        current['_validation_manifest'] = {'fixture': True}
        return loader, evaluation
    monkeypatch.setattr(run.training_data, 'build_dataloaders', dataloaders)
    def forbidden(*args, **kwargs): raise AssertionError('terminal-only reentry cannot train')
    monkeypatch.setattr(run.training_steps, 'train_for_steps', forbidden)
    device = next(model.parameters()).device
    # Multiple GPU diagnostics share one process; its strict setup is already verified.
    if device.type == 'cuda':
        from src.utils.reproducibility import deterministic_runtime_settings
        monkeypatch.setattr(run, 'configure_strict_determinism', lambda c: deterministic_runtime_settings())
    for _ in range(2):
        run.run_training(config, model=model, tokenizer=object(), tokenized_dataset=[{}], device=device)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    resources = run.ResourceAttemptLedger(Path(config['run']['output_dir']),run_id=config['run']['run_id']).summary()
    assert resources['attempt_count'] == 2 and resources['attempted_steps'] == 0
    result = json.loads((Path(config['run']['output_dir'])/'terminal_validation_results.json').read_text())
    assert result['global_step'] == 348528 and len(result['endpoints']) == 4
