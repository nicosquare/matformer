"""Schema/grid foundations; actual schema-4 model and runtime checks come later."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from src.evaluation import optimizer_ownership as campaign
from src.utils.config import ConfigError
from src.utils.reproducibility import build_optimizer_ownership_signature, stable_hash
from test_optimizer_ownership_campaign import audited_inputs, expand


LEGACY = json.loads(
    (Path(__file__).parent / "fixtures/optimizer_ownership_legacy_signatures.json").read_text()
)
NEW_FIELDS = {"campaign_schema_version", "width_grid", "block_boundaries"}
LABELS = ("g125", "g250", "g500", "g1000")


@pytest.mark.parametrize("schema,suffix", [
    (1, "optimizer_ownership"),
    (2, "optimizer_ownership_corrections"),
    (3, "inverse_membership"),
])
def test_legacy_resolutions_and_signatures_are_unchanged(
    tmp_path, monkeypatch, audited_inputs, schema, suffix
):
    # Only external corpus/tokenizer IO and volatile source provenance are stubbed.
    # Hashes were captured from the real resolver before the foundation edits.
    monkeypatch.setattr(campaign, "_provenance", lambda: copy.deepcopy(LEGACY["provenance"]))
    recipe = yaml.safe_load(Path(
        f"configs/controlled_exps/tinystories_instruct_{suffix}.yaml"
    ).read_text())
    runs = expand(tmp_path, recipe)
    assert {run["arm_id"]: run["contract_hash"] for run in runs} == LEGACY["contracts"][str(schema)]
    assert stable_hash(campaign.campaign_common(schema)) == LEGACY["common_hash"]
    assert campaign.campaign_widths(schema) == campaign.WIDTHS
    assert campaign.campaign_topology(schema) == {}
    assert [run["arm_id"] for run in runs] == [
        arm["arm_id"] for arm in campaign.campaign_arms(schema)
    ]
    for run in runs:
        config = run["resolved_config"]
        campaign.validate_materialized_config(config)
        contract = run["optimizer_ownership_contract"]
        assert NEW_FIELDS.isdisjoint(contract)
        assert "campaign_schema_version" not in config["run"]
        assert config["model"]["intermediate_size"] == run["physical_ffn_dimension"]
        assert config["training"]["max_steps"] == run["assigned_updates"]
        assert config["model"]["correction_mode"] == run.get("correction_mode", "none")
        if run["source_width"]:
            assert config["model"]["granularity_prefixes"] == {run["source_width"]: 1.0}
        else:
            assert tuple(config["model"]["granularities"]) == campaign.WIDTH_LABELS
            assert config["model"]["granularity_sampling_mode"] == (
                "fixed_global" if schema == 3 else "global"
            )
        assert build_optimizer_ownership_signature(contract) == (run["contract_hash"], contract)


def test_matformer_grid_matrix_and_inherited_controls():
    widths = campaign.campaign_widths(4)
    assert tuple(width["label"] for width in widths) == LABELS
    assert [width["source_fraction"] for width in widths] == [.125, .25, .5, 1.]
    assert [width["active_ffn_dimension"] for width in widths] == [32, 64, 128, 256]
    assert [width["non_embedding_parameters"] for width in widths] == [90688, 115264, 164416, 262720]
    assert [width["active_quarters"] for width in widths] == [
        ("A",), ("A", "B"), ("A", "B", "C"), ("A", "B", "C", "D"),
    ]
    arms = campaign.campaign_arms(4)
    assert [arm["arm_id"] for arm in arms] == [
        "ST-g125", "ST-g250", "ST-g500", "ST-g1000", "S1", "S2", "C1", "C2", "C3",
    ]
    assert [arm["physical_ffn_dimension"] for arm in arms[:4]] == [32, 64, 128, 256]
    assert [(arm["representation"], arm["state_scope"], arm["clipping_mode"]) for arm in arms] == [
        *([("dense", "shared", "global")] * 4),
        ("slicing", "shared", "global"), ("slicing", "per_granularity", "global"),
        ("concat", "shared", "global"), ("concat", "per_granularity", "global"),
        ("concat", "per_ffn_block", "per_owner"),
    ]
    for index, arm in enumerate(arms):
        epochs = 1 if index < 4 else 4
        assert arm["assigned_epochs"] == epochs
        assert arm["assigned_updates"] == epochs * 87132
        assert arm["assigned_tokens"] == epochs * 713785344
        assert arm["endpoint_widths"] == ((LABELS[index],) if index < 4 else LABELS)
        assert arm["source_width"] == (LABELS[index] if index < 4 else None)
    assert sum(arm["assigned_tokens"] for arm in arms) == 17130848256
    common = campaign.campaign_common(4)
    assert common["model"]["granularities"] == list(LABELS)
    assert common["model"]["granularity_prefixes"] == dict(zip(LABELS, [.125, .25, .5, 1.]))
    for field in ("granularities", "granularity_prefixes"):
        common["model"][field] = copy.deepcopy(campaign.PINNED_COMMON["model"][field])
    assert common == campaign.PINNED_COMMON
    common["training"]["optimizer"]["kwargs"]["betas"][0] = 0
    assert stable_hash(campaign.PINNED_COMMON) == LEGACY["common_hash"]


def test_schema_qualified_repeated_labels_and_legacy_defaults():
    assert campaign.campaign_arms() is campaign.ARMS
    assert campaign.campaign_widths() is campaign.WIDTHS
    assert campaign.campaign_common() == campaign.PINNED_COMMON
    assert campaign.campaign_topology() == {}
    assert campaign.campaign_arm(1, "S1")["endpoint_widths"] == campaign.WIDTH_LABELS
    assert campaign.campaign_arm(4, "S1")["endpoint_widths"] == LABELS
    # Same physical standalone, different block membership within the two grids.
    assert campaign.campaign_arm(1, "ST-g250")["physical_ffn_dimension"] == 64
    assert campaign.campaign_arm(4, "ST-g250")["physical_ffn_dimension"] == 64
    old = next(w for w in campaign.campaign_widths(1) if w["label"] == "g250")
    new = next(w for w in campaign.campaign_widths(4) if w["label"] == "g250")
    assert old["active_quarters"] == ("A",)
    assert new["active_quarters"] == ("A", "B")
    for schema, arm in ((1, "ST-g125"), (4, "ST-g750"), (4, "C3-IM"), (2, "C3")):
        with pytest.raises(ConfigError, match="Unknown arm_id"):
            campaign.campaign_arm(schema, arm)


@pytest.mark.parametrize("version", [None, True, False, 0, -1, 5, "4", 4.0, 1.0])
def test_unknown_or_noninteger_schemas_fail_all_selectors(version):
    for selector in (campaign.campaign_arms, campaign.campaign_widths,
                     campaign.campaign_common, campaign.campaign_topology):
        with pytest.raises(ConfigError, match="Unsupported campaign schema_version"):
            selector(version)
    with pytest.raises(ConfigError, match="Unsupported campaign schema_version"):
        campaign.campaign_arm(version, "S1")


def test_topology_records_contiguous_physical_blocks_and_support():
    topology = campaign.campaign_topology(4)
    assert set(topology) == NEW_FIELDS
    assert topology["campaign_schema_version"] == 4
    assert topology["width_grid"] == list(campaign.campaign_widths(4))
    assert topology["block_boundaries"] == [
        {"id": "A", "start": 0, "end": 32, "dimension": 32, "supported_widths": LABELS},
        {"id": "B", "start": 32, "end": 64, "dimension": 32, "supported_widths": LABELS[1:]},
        {"id": "C", "start": 64, "end": 128, "dimension": 64, "supported_widths": LABELS[2:]},
        {"id": "D", "start": 128, "end": 256, "dimension": 128, "supported_widths": LABELS[3:]},
    ]
    topology["width_grid"][0]["active_ffn_dimension"] = 999
    topology["block_boundaries"][0]["end"] = 999
    assert campaign.campaign_topology(4)["block_boundaries"][0]["end"] == 32
    assert campaign.campaign_widths(4)[0]["active_ffn_dimension"] == 32


@pytest.mark.parametrize("arm_id", [
    "ST-g125", "ST-g250", "ST-g500", "ST-g1000", "S1", "S2", "C1", "C2", "C3",
])
def test_new_contract_topology_uses_existing_serializer(arm_id):
    # Declared-control fixture exercises construction only. It does not claim a
    # resolved schema-4 config, actual model count, or topology eligibility pass.
    arm = campaign.campaign_arm(4, arm_id)
    config = campaign._merge(campaign.campaign_common(4), campaign._arm_overrides(arm))
    config["run"].update(
        campaign_id=campaign.MATFORMER_CAMPAIGN_ID, arm_id=arm_id,
        run_id=f"{campaign.MATFORMER_CAMPAIGN_ID}-{arm_id}-s42",
        campaign_schema_version=4,
    )
    config["run"].setdefault("sampling_mode", "standalone")
    before = copy.deepcopy(config)
    digest, contract = campaign.build_run_scientific_contract(
        config, {"seed": 42, "method": "fixture"}, campaign_schema_version=4
    )
    assert config == before
    assert contract["schema_version"] == 1
    assert contract["campaign_schema_version"] == 4
    assert contract["arm_id"] == arm_id
    assert contract["run_id"] == config["run"]["run_id"]
    assert contract["width_grid"][0]["active_ffn_dimension"] == 32
    assert contract["block_boundaries"][3]["supported_widths"] == ["g1000"]
    assert contract["sampling"]["probabilities"] == (None if arm["source_width"] else [.25] * 4)
    assert build_optimizer_ownership_signature(contract) == (digest, contract)
    assert "correction" not in contract
    historical_shape = {key: value for key, value in contract.items() if key not in NEW_FIELDS}
    historical_before = copy.deepcopy(historical_shape)
    historical_hash, historical_inputs = build_optimizer_ownership_signature(historical_shape)
    assert historical_hash != digest
    assert historical_inputs == historical_before == historical_shape
    for field in NEW_FIELDS:
        changed = copy.deepcopy(contract)
        if field == "campaign_schema_version":
            changed[field] = 3
        elif field == "width_grid":
            changed[field][0]["active_ffn_dimension"] = 64
        else:
            changed[field][0]["end"] = 64
        assert build_optimizer_ownership_signature(changed)[0] != digest
    config["run"]["campaign_id"] = "tinystories-optimizer-ownership-v1"
    with pytest.raises(ConfigError, match="campaign_id"):
        campaign.build_run_scientific_contract(config, {}, campaign_schema_version=4)


@pytest.fixture
def mw_recipe(audited_inputs):
    return yaml.safe_load(Path(
        'configs/controlled_exps/tinystories_instruct_matformer_widths.yaml'
    ).read_text())


def test_new_nine_models_and_materialized_configs(tmp_path, mw_recipe):
    from src.utils.config import resolve_run_config
    runs = expand(tmp_path, mw_recipe)
    checks = campaign.inspect_campaign_models(runs)
    assert len(runs) == len(checks) == 9
    for run, check in zip(runs, checks):
        config = run['resolved_config']
        assert config['run']['campaign_schema_version'] == 4
        assert run['run_id'] == f"{campaign.MATFORMER_CAMPAIGN_ID}-{run['arm_id']}-s42"
        assert check['counts'] == {w['label']: w['non_embedding_parameters']
            for w in campaign.campaign_widths(4) if w['label'] in run['endpoint_widths']}
        if run['source_width']:
            assert config['model']['granularity_prefixes'] == {run['source_width']: 1.}
            assert config['model']['matformer_source_granularity_prefixes'] == dict(zip(LABELS, [.125, .25, .5, 1.]))
        path = tmp_path / f"{run['arm_id']}.yaml"
        path.write_text(yaml.safe_dump(run['executable_config']))
        campaign.validate_materialized_config(resolve_run_config(path, create_output_dirs=False))
    assert [o['parameter_elements'] for o in checks[-1]['owners'][:4]] == [24576, 24576, 49152, 98304]
    assert len(checks[-1]['owners']) == 5


@pytest.mark.parametrize('section,key,value', [
    ('training', 'learning_rate', .007), ('training', 'max_steps_cap', 100),
    ('training', 'warmup_steps', 0), ('training', 'mixed_precision', 'none'),
    ('model', 'correction_mode', 'inverse_membership'),
    ('model', 'granularity_prefixes', dict(zip(LABELS, [.25, .5, .75, 1.]))),
    ('run', 'seed', 43), ('run', 'resume_from', '/diagnostic/checkpoint.pt'),
    ('dataset', 'data_seed', 43),
])
def test_new_control_mutations_fail(tmp_path, mw_recipe, section, key, value):
    mw_recipe['common'][section][key] = value
    with pytest.raises(ConfigError):
        expand(tmp_path, mw_recipe)


@pytest.mark.parametrize('mutation', ['marker', 'grid', 'boundary', 'source', 'initialization', 'sampling'])
def test_new_materialized_metadata_rejection(tmp_path, mw_recipe, mutation):
    from src.utils.config import resolve_run_config
    run = expand(tmp_path, mw_recipe)[0 if mutation == 'source' else -1]
    raw = run['executable_config']
    contract = raw['optimizer_ownership_contract']
    if mutation == 'marker':
        del raw['run']['campaign_schema_version']
    elif mutation == 'grid':
        contract['width_grid'][0]['source_fraction'] = .25
    elif mutation == 'boundary':
        contract['block_boundaries'][0]['end'] = 64
    elif mutation == 'source':
        raw['run']['granularity'] = 'g250'
    elif mutation == 'initialization':
        contract['initialization']['method'] = 'diagnostic_checkpoint'
    else:
        raw['model']['granularity_sampling_mode'] = 'fixed_global'
        raw['model']['global_sampling_distribution'] = dict.fromkeys(LABELS, .25)
    raw['optimizer_ownership_contract_hash'] = build_optimizer_ownership_signature(contract)[0]
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError):
        resolve_run_config(path, create_output_dirs=False)


def test_new_preflight_publication_and_data_rejections(tmp_path, mw_recipe, audited_inputs, monkeypatch):
    import test_optimizer_ownership_campaign as legacy
    inputs = (mw_recipe, *audited_inputs[1:])
    legacy.test_occupied_and_pinned_identity_rejections(tmp_path, copy.deepcopy(inputs))
    (tmp_path / 'runs').rmdir()
    legacy.test_preflight_publishes_only_after_all_checks(tmp_path, inputs, monkeypatch)
    assert campaign._reservation_path(tmp_path / 'runs').exists()
    with pytest.raises(ConfigError, match='occupied'):
        expand(tmp_path, mw_recipe)


def test_new_full_action_and_epoch_traces(tmp_path, mw_recipe, audited_inputs):
    import hashlib
    import random
    import numpy as np
    from src.training.packed_corpus import RepeatingNoPaddingDistributedBatchSampler
    from src.utils.reproducibility import seed_for
    runs = expand(tmp_path, mw_recipe)
    manifest = copy.deepcopy(audited_inputs[1])
    # Real full-size order/sampler; synthetic corpus identity, no document reads.
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    order = np.arange(5576491, dtype='<u8')
    path = corpus / manifest['training_order']['path']
    path.write_bytes(order.tobytes())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for run in runs:
        run['resolved_config']['dataset']['optimizer_iteration']['permutation_hash'] = digest
    traces = campaign.build_expected_traces(runs, corpus, manifest)
    assert len(traces) == 9
    elastic = traces['C3']
    assert [e['sequences'] for e in elastic['epochs']] == [5576448] * 4
    assert [e['updates'] for e in elastic['epochs']] == [87132] * 4
    assert len({e['fixed_epoch_set_hash'] for e in elastic['epochs']}) == 1
    rng = random.Random(seed_for(runs[-1]['resolved_config'], 'granularity_selection'))
    expected = [LABELS[rng.randrange(4)] for _ in range(348528)]
    assert elastic['actions']['sha256'] == hashlib.sha256(''.join(w+'\n' for w in expected).encode()).hexdigest()
    assert elastic['actions']['counts'] == {w: expected.count(w) for w in LABELS}
    assert len(set(elastic['actions']['counts'].values())) > 1
    sampler = RepeatingNoPaddingDistributedBatchSampler(19, 4, 0, 1, 64, 16, corpus_hash='fixture', optimizer_training_manifest_hash='fixture')
    epochs = campaign.expected_epoch_traces(sampler, epochs=4)
    batches = list(sampler)
    for i, epoch in enumerate(epochs):
        values = np.asarray(batches[i*4:(i+1)*4], dtype='<u8').reshape(-1)
        assert epoch['sha256'] == hashlib.sha256(values.tobytes()).hexdigest()
        assert set(values) == set(np.asarray(batches[:4]).reshape(-1))


def mw_model(variant='concat', *, bias=False, tied=False):
    from transformers import LlamaConfig
    from src.models.ffn import CatLlamaMLP, ModifiedLlamaMLP
    from src.models.wiring import ModifiedLlamaForCausalLM
    import torch
    torch.manual_seed(42)
    config = LlamaConfig(vocab_size=2048, hidden_size=64, intermediate_size=256,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, mlp_bias=bias, tie_word_embeddings=tied)
    config.granularities = list(LABELS)
    config.granularity_prefixes = dict(zip(LABELS, [.125, .25, .5, 1.]))
    return ModifiedLlamaForCausalLM(config,
        mlp_cls=CatLlamaMLP if variant == 'concat' else ModifiedLlamaMLP,
        mlp_kwargs={'gradient_membership_correction_enabled': False})


def mw_training(scope='shared'):
    from test_optimizer_ownership import training
    result = training(scope)
    result['optimizer_state_contract']['ordered_granularities'] = list(LABELS)
    result['optimizer_state_topology'] = campaign.campaign_topology(4)
    return result


def mw_backward(model, optimizer, width):
    import torch
    optimizer.zero_grad(set_to_none=True)
    model.configure_subnetwork(width)
    tokens = torch.arange(1, 9).reshape(1, 8)
    model(input_ids=tokens, labels=tokens).loss.backward()


@pytest.mark.parametrize('arm', ['S1', 'S2', 'C1', 'C2', 'C3'])
def test_all_width_histories_and_wide_then_narrow_updates(arm):
    import torch
    from src.training import steps
    from src.training.optimizer_state import build_parameter_descriptors, measure_optimizer_storage
    from test_optimizer_ownership import commit, assert_state_equal
    variant = 'slicing' if arm.startswith('S') else 'concat'
    scope = 'per_granularity' if arm.endswith('2') else 'per_ffn_block' if arm == 'C3' else 'shared'
    model = mw_model(variant)
    optimizer, scheduler = steps.build_optimizer_and_scheduler(model, mw_training(scope))
    params = dict(model.named_parameters())
    descriptors = build_parameter_descriptors(model, ordered_widths=LABELS)
    mw_backward(model, optimizer, 'g1000')
    commit(optimizer, scheduler, 'g1000')
    wide_opt = optimizer.optimizer_for('g1000') if scope == 'per_granularity' else optimizer
    wide_state = copy.deepcopy(wide_opt.state_dict())
    before = {n: p.detach().clone() for n, p in params.items()}
    inactive_histories = {}
    if variant == 'concat':
        for d in descriptors:
            if d['quarter_id'] == 'D':
                owner_optimizer = optimizer.optimizer_for('O-D') if arm == 'C3' else wide_opt
                inactive_histories[d['canonical_name']] = copy.deepcopy(owner_optimizer.state[params[d['canonical_name']]])
    mw_backward(model, optimizer, 'g125')
    for d in descriptors:
        p = params[d['canonical_name']]
        assert (p.grad is not None) == ('g125' in d['gradient_support'])
    p = model.model.layers[0].mlp.gate_proj.weight if variant == 'slicing' else None
    if p is not None:
        assert torch.count_nonzero(p.grad[32:]) == 0
        old_moment = wide_opt.state[p]['exp_avg'].clone()
    commit(optimizer, scheduler, 'g125')
    if arm == 'S1':
        assert not torch.equal(p[32:], before['model.layers.0.mlp.gate_proj.weight'][32:])
        assert not torch.equal(p[32:], before['model.layers.0.mlp.gate_proj.weight'][32:] * (1 - .01*.1))
        torch.testing.assert_close(optimizer.state[p]['exp_avg'][32:], old_moment[32:] * .9)
        assert optimizer.state[p]['step'] == 2
    if scope == 'per_granularity':
        assert_state_equal(wide_state, wide_opt.state_dict())
        if arm == 'S2':
            for layer in model.model.layers:
                for parameter, tail in ((layer.mlp.gate_proj.weight, (slice(32, None),)),
                                        (layer.mlp.up_proj.weight, (slice(32, None),)),
                                        (layer.mlp.down_proj.weight, (slice(None), slice(32, None)))):
                    for key in ('exp_avg', 'exp_avg_sq'):
                        moment = optimizer.optimizer_for('g125').state[parameter][key]
                        assert moment.shape == parameter.shape
                        assert torch.count_nonzero(moment[tail]) == 0
    if variant == 'concat':
        for d in descriptors:
            if 'g125' not in d['gradient_support']:
                assert torch.equal(params[d['canonical_name']], before[d['canonical_name']])
        for name, saved in inactive_histories.items():
            owner_optimizer = optimizer.optimizer_for('O-D') if arm == 'C3' else wide_opt
            assert_state_equal(owner_optimizer.state[params[name]], saved)
    for width in LABELS[1:3]:
        mw_backward(model, optimizer, width)
        commit(optimizer, scheduler, width)
    measured = measure_optimizer_storage(optimizer, step=4)
    total = sum(p.numel() for p in model.parameters())
    expected = 2 * total * (4 if arm == 'S2' else 1)
    if arm == 'C2':
        expected = 2 * sum(d['scalar_count'] * len(d['gradient_support']) for d in descriptors)
        assert expected == 2 * (4*24576 + 3*24576 + 2*49152 + 98304 + 4*328256)
        for d in descriptors:
            parameter = params[d['canonical_name']]
            histories = [entry.optimizer.state[parameter] for entry in optimizer.entries if parameter in entry.optimizer.state]
            assert len(histories) == len(d['gradient_support'])
            assert all(history['step'] == 1 for history in histories)
    assert measured['moment_elements'] == expected
    assert measured['moment_bytes'] == expected * 4
    assert measured['counter_elements'] > 0
    if arm == 'C3':
        assert optimizer.successful_update_counts == dict(zip(campaign.OWNER_IDS, [4, 3, 2, 1, 4]))
        assert scheduler.position == 4
    # A present zero gradient still advances AdamW; absent concat blocks did not.
    mw_backward(model, optimizer, 'g1000')
    for parameter in model.parameters():
        parameter.grad.zero_()
    before = next(model.parameters()).detach().clone()
    commit(optimizer, scheduler, 'g1000')
    assert not torch.equal(next(model.parameters()), before)


@pytest.mark.parametrize('bias,tied', [(False, False), (True, False), (True, True)])
def test_unequal_partition_and_independent_clipping(bias, tied):
    import math
    import torch
    from src.training import steps
    from src.training.optimizer_state import build_concat_parameter_partition
    model = mw_model(bias=bias, tied=tied)
    with pytest.raises(ConfigError, match='equal quarters'):
        build_concat_parameter_partition(model, ordered_widths=LABELS)
    owners = build_concat_parameter_partition(model, ordered_widths=LABELS, topology=campaign.campaign_topology(4))
    assert [o.owner_id for o in owners] == list(campaign.OWNER_IDS)
    assert len({id(p) for o in owners for p in o.parameters}) == sum(len(o.parameters) for o in owners) == len(list(model.parameters()))
    if bias:
        assert sum(d['component'] == 'down_bias' for d in owners[-1].descriptors) == 4
    if tied:
        assert any('lm_head.weight' in d['tied_aliases'] for d in owners[-1].descriptors)
    for i, width in enumerate(LABELS):
        for o in owners:
            for p in o.parameters:
                p.grad = torch.full_like(p, 10.) if width in o.active_widths else None
        observation = steps.clip_optimizer_gradients(model, mw_training('per_ffn_block'), width, owners=owners)
        assert observation['combined_post_norm'] == pytest.approx(math.sqrt(i+2), rel=1e-4)
        assert observation['global_coefficient'] is None
        for o in owners:
            group = observation['groups'][o.owner_id]
            if width in o.active_widths:
                assert group['max_norm'] == 1.
                assert group['post_norm'] == pytest.approx(1., rel=1e-4)
            else:
                assert group == dict(active=False, pre_norm=None, post_norm=None, coefficient=None, max_norm=None)
    model.model.layers[-1].mlp.down_weight_blocks[3] = torch.nn.Parameter(torch.ones(64, 64))
    with pytest.raises(ConfigError, match='shape'):
        build_concat_parameter_partition(model, ordered_widths=LABELS, topology=campaign.campaign_topology(4))


def mw_runtime(tmp_path, arm):
    """Real campaign geometry with explicit small CPU diagnostic controls."""
    import torch
    from test_optimizer_ownership import runtime_fixture
    from src.training import steps, checkpointing
    scope = 'per_ffn_block' if arm == 'C3' else 'per_granularity' if arm in ('S2', 'C2') else 'shared'
    config, _, _, _, batches, _ = runtime_fixture(tmp_path, scope='shared')
    config['run'].update(run_id=f'{campaign.MATFORMER_CAMPAIGN_ID}-{arm}-s42',
        campaign_id=campaign.MATFORMER_CAMPAIGN_ID, arm_id=arm, campaign_schema_version=4)
    config['model'].update(d_model=64, num_layers=4, intermediate_size=256,
        granularities=list(LABELS), granularity_prefixes=dict(zip(LABELS, [.125,.25,.5,1.])))
    config['training'].update(optimizer_state_scope=scope,
        optimizer_state_topology=campaign.campaign_topology(4),
        gradient_clipping={**mw_training(scope)['gradient_clipping'], 'norm_type': 2.})
    config['training']['optimizer_state_contract'].update(state_scope=scope, ordered_granularities=list(LABELS))
    contract = dict(schema_version=1, arm_id=arm, campaign_id=campaign.MATFORMER_CAMPAIGN_ID,
        representation='concat', state_scope=scope, **campaign.campaign_topology(4))
    config['optimizer_ownership_contract'] = contract
    config['optimizer_ownership_contract_hash'] = stable_hash(contract)
    config['model']['variant'] = 'slicing' if arm.startswith('S') else 'concat'
    contract['representation'] = config['model']['variant']
    config['optimizer_ownership_contract_hash'] = stable_hash(contract)
    model = mw_model(config['model']['variant'])
    optimizer, clock = steps.build_optimizer_and_scheduler(model, config['training'])
    state = checkpointing.build_initial_continuation_state(config)
    return config, model, optimizer, clock, batches, state


@pytest.mark.parametrize('arm', ['C1', 'C3'])
def test_new_real_trainer_clipping_sidecars_and_rejection(tmp_path, monkeypatch, arm):
    import csv
    import math
    import torch
    from src.training import steps
    from src.training.run import build_ownership_run_summary
    from src.utils.metrics import MetricsJournal, append_optimizer_ownership_observation
    config, model, optimizer, clock, batches, state = mw_runtime(tmp_path, arm)
    # A deterministic seed whose first eight draws cover all four widths also
    # exercises the ordinary production selector, without patching its internals.
    import random
    rng = random.Random(4)
    monkeypatch.setattr(steps, 'dedicated_random', lambda *args: rng)
    events = []
    if arm == 'C3':
        for entry in optimizer.entries:
            original_step = entry.optimizer.step
            def tracked(*args, _entry=entry, _step=original_step, **kwargs):
                assert state['update_in_flight']
                assert _entry.optimizer.param_groups[0]['lr'] == clock.current_learning_rates[0]
                events.append(_entry.owner_id)
                return _step(*args, **kwargs)
            monkeypatch.setattr(entry.optimizer, 'step', tracked)
        original_clock = clock.step
        def tick():
            events.append('clock')
            original_clock()
        monkeypatch.setattr(clock, 'step', tick)
    journal = MetricsJournal(config['run']['output_dir'], artifact_state=state, artifact_io_config=config)
    def committed(**kwargs):
        if arm == 'C3':
            assert events == [*optimizer.active_owner_ids(state['optimizer_last_active_granularity']), 'clock']
            events.clear()
            assert clock.position == kwargs['step']
        append_optimizer_ownership_observation(config, state, train_dataloader=batches)
    steps.train_for_steps(config, model, batches, [], optimizer, clock, torch.device('cpu'),
        run_state=state, metrics_journal=journal, successful_step_callback=committed)
    journal.flush()
    assert all(n > 0 for n in state['optimizer_width_selection_counts'].values())
    summary = {'run_id': config['run']['run_id'], **build_ownership_run_summary(config, model, optimizer, state)}
    audit = campaign.inspect_run_observations(config['run']['output_dir'], summary)
    assert audit['committed_updates'] == 8
    assert set(audit['clipping_by_width']) == set(LABELS)
    assert set(summary['optimizer_ownership']['expected_exposure']['width_selections']) == set(LABELS)
    root = Path(config['run']['output_dir'])
    rows = [json.loads(line) for line in (root / 'optimizer_ownership_clipping.jsonl').read_text().splitlines()]
    assert len(rows) == 8
    for row in rows:
        assert set(row['groups']) == set(campaign.OWNER_IDS)
        assert row['combined_post_norm'] <= (1 if arm == 'C1' else math.sqrt(LABELS.index(row['width']) + 2)) + 3e-5
    metrics = list(csv.DictReader((root / 'metrics.csv').open()))
    assert all(r['optimizer_ownership_clipping_path'] == 'optimizer_ownership_clipping.jsonl' for r in metrics if r['split'] == 'train')
    for damage in ('omit', 'active', 'coefficient', 'combined', 'count', 'metric'):
        changed = copy.deepcopy(summary)
        broken = copy.deepcopy(rows)
        if damage == 'omit': changed['optimizer_ownership']['clipping_path'] = None
        elif damage == 'active': broken[0]['groups']['O-D']['active'] = not broken[0]['groups']['O-D']['active']
        elif damage == 'coefficient': broken[0]['groups']['O-A']['coefficient'] = 0.
        elif damage == 'combined': broken[0]['combined_post_norm'] += 1.
        elif damage == 'count': broken.pop()
        else:
            text = (root / 'metrics.csv').read_text()
            (root / 'metrics.csv').write_text(text.replace('optimizer_ownership_clipping.jsonl', 'missing.jsonl'))
        path = root / 'optimizer_ownership_clipping.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in broken))
        with pytest.raises(ConfigError):
            campaign.inspect_run_observations(root, changed)
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


@pytest.mark.parametrize('arm', ['S1', 'S2', 'C1', 'C2', 'C3'])
def test_runtime_summary_measured_and_support_expected_storage(tmp_path, arm):
    from src.training import steps
    from src.training.run import build_ownership_run_summary
    from src.utils.reproducibility import seed_training_randomness
    import torch
    config, model, optimizer, clock, batches, state = mw_runtime(tmp_path, arm)
    seed_training_randomness(config)
    steps.train_for_steps(config, model, batches, [], optimizer, clock, torch.device('cpu'), run_state=state)
    audit = build_ownership_run_summary(config, model, optimizer, state)['optimizer_ownership']
    assert audit['steps'] == audit['scheduler_position'] == 8
    assert sum(audit['width_selection_counts'].values()) == 8
    assert audit['storage']['moment_elements'] == audit['storage']['expected_observed_moment_elements']
    expected = 3363328 if arm == 'C2' else 4198912 if arm == 'S2' else 1049728
    assert audit['storage']['expected_fully_exposed_moment_elements'] == expected
    assert audit['expected_exposure']['width_selections'] == dict.fromkeys(LABELS, 2.)
    assert audit['expected_exposure']['quarter_activations'] == dict(zip(campaign.OWNER_IDS[:4], [8., 6., 4., 2.]))
    assert audit['tokens_seen'] == 64
    if arm.startswith('C'):
        assert audit['temporary_concat_storage']['bytes_by_width'] == dict(zip(LABELS, [98304, 196608, 393216, 786432]))


def test_dense_summary_preserves_physical_source_fraction(tmp_path, mw_recipe):
    from src.training import checkpointing, steps
    from src.training.modeling import build_model
    from src.training.run import build_ownership_run_summary
    for run in expand(tmp_path, mw_recipe)[:4]:
        config = run['resolved_config']
        model = build_model(config)
        optimizer, _ = steps.build_optimizer_and_scheduler(model, config['training'])
        state = checkpointing.build_initial_continuation_state(config)
        audit = build_ownership_run_summary(config, model, optimizer, state)['optimizer_ownership']
        assert audit['source_width']['label'] == run['source_width']
        assert audit['source_width']['active_ffn_dimension'] == run['physical_ffn_dimension']
        assert audit['source_width']['source_fraction'] == run['physical_ffn_dimension'] / 256
        assert audit['storage']['moment_elements'] == 0


@pytest.mark.parametrize('arm', ['S2', 'C3'])
def test_new_compact_metrics_checkpoint_roundtrip(tmp_path, arm):
    from test_metrics_compact_accounting import train_with_journal
    from test_optimizer_ownership_resume import save, load
    from src.utils.reproducibility import seed_training_randomness
    source = mw_runtime(tmp_path, arm)
    seed_training_randomness(source[0])
    journal = train_with_journal(source, stop=4)
    journal.flush()
    path = tmp_path / 'diagnostic.pt'
    save(source, path)
    restored = mw_runtime(tmp_path, arm)
    load(restored, path)
    assert restored[-1]['metrics_accumulator_state'] == source[-1]['metrics_accumulator_state']
    resumed = train_with_journal(restored)
    resumed.flush()
    assert restored[-1]['metrics_accumulator_state']['committed_optimizer_steps'] == 8


@pytest.mark.parametrize('damage', ['prefix', 'layer', 'bias', 'boundary', 'support', 'mixed'])
def test_new_partition_rejects_inconsistent_physical_layout(damage):
    import torch
    from src.training.optimizer_state import build_concat_parameter_partition
    from transformers.models.llama.modeling_llama import LlamaMLP
    model = mw_model(bias=True)
    topology = campaign.campaign_topology(4)
    mlp = model.model.layers[2].mlp
    if damage == 'prefix':
        mlp.ffn_concat_block_metadata[2]['cumulative_prefix_width'] = 192
    elif damage == 'layer':
        model.model.layers = torch.nn.ModuleList(list(model.model.layers)[:3])
    elif damage == 'bias':
        mlp.up_bias_blocks[2] = torch.nn.Parameter(torch.zeros(32))
    elif damage == 'boundary':
        topology['block_boundaries'][1]['start'] = 31
    elif damage == 'support':
        mlp.granularity_prefixes['g500'] = .75
    else:
        model.model.layers[1].mlp = LlamaMLP(model.config)
    with pytest.raises(ConfigError):
        build_concat_parameter_partition(model, ordered_widths=LABELS, topology=topology)


def test_new_owner_coefficient_is_independent_of_common_gradient():
    import torch
    from src.training import steps
    from src.training.optimizer_state import build_concat_parameter_partition
    model = mw_model()
    owners = build_concat_parameter_partition(model, ordered_widths=LABELS, topology=campaign.campaign_topology(4))
    results = []
    for scale in (0., 100.):
        model.zero_grad(set_to_none=True)
        for owner in (owners[0], owners[-1]):
            for p in owner.parameters:
                p.grad = torch.full_like(p, 5. if owner.owner_id == 'O-A' else scale)
        result = steps.clip_optimizer_gradients(model, mw_training('per_ffn_block'), 'g125', owners=owners)
        results.append(result)
    assert results[0]['groups']['O-A'] == results[1]['groups']['O-A']
    assert results[0]['groups']['O-common']['coefficient'] == 1.
    assert results[0]['groups']['O-common']['active'] is True
