"""Real-model checks for static optimizer ownership and physical support."""

import copy
import json
import math
import random

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP

from src.models.ffn import CatLlamaMLP, ModifiedLlamaMLP
from src.models.wiring import ModifiedLlamaForCausalLM
from src.training import steps
from src.training.optimizer_state import (
    BlockOptimizerCollection,
    PerGranularityOptimizerCollection,
    build_concat_parameter_partition,
    build_parameter_descriptors,
)
from src.utils.config import ConfigError
from src.utils.reproducibility import stable_hash


WIDTHS = ("g250", "g500", "g750", "g1000")


def _config(*, bias=False, tied=False, intermediate_size=64, hidden_size=16):
    config = LlamaConfig(
        vocab_size=32, hidden_size=hidden_size, intermediate_size=intermediate_size,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=16, mlp_bias=bias, tie_word_embeddings=tied,
    )
    config.granularities = list(WIDTHS)
    config.granularity_prefixes = dict(zip(WIDTHS, (0.25, 0.5, 0.75, 1.0)))
    return config


def _model(*, variant="concat", bias=False, tied=False):
    return ModifiedLlamaForCausalLM(
        _config(bias=bias, tied=tied),
        mlp_cls=CatLlamaMLP if variant == "concat" else ModifiedLlamaMLP,
        mlp_kwargs={"gradient_membership_correction_enabled": False},
    )


@pytest.mark.parametrize("mlp_type", [CatLlamaMLP, ModifiedLlamaMLP])
@pytest.mark.parametrize("bias", [False, True])
def test_ffn_metadata_matches_physical_gradients_without_changing_forward(mlp_type, bias):
    mlp = mlp_type(_config(bias=bias), gradient_membership_correction_enabled=False)
    baseline = copy.deepcopy(mlp)
    rng = torch.get_rng_state().clone()
    entries = mlp.physical_parameter_metadata()
    assert torch.equal(torch.get_rng_state(), rng)
    assert {entry.parameter_name for entry in entries} == dict(mlp.named_parameters()).keys()
    x = torch.randn(2, 3, 16)
    for index, width in enumerate(WIDTHS):
        mlp.zero_grad(set_to_none=True)
        baseline.zero_grad(set_to_none=True)
        mlp.configure_subnetwork(width)
        baseline.configure_subnetwork(width)
        actual, expected = mlp(x), baseline(x)
        assert torch.equal(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        for entry in entries:
            grad = entry.parameter.grad
            assert (grad is not None) == (width in entry.gradient_support)
            reference = dict(baseline.named_parameters())[entry.parameter_name].grad
            if grad is not None:
                assert grad.shape == entry.parameter.shape
                assert torch.equal(grad, reference)
            if mlp_type is ModifiedLlamaMLP and entry.component == "gate_weight":
                assert torch.count_nonzero(grad[(index + 1) * 16:]) == 0
    if bias:
        down_bias = next(entry for entry in entries if entry.component == "down_bias")
        assert down_bias.block_index is None
        assert down_bias.gradient_support == WIDTHS


@pytest.mark.parametrize("bias,tied", [(False, False), (True, False), (True, True)])
def test_concat_partition_is_complete_disjoint_stable_and_allocation_free(bias, tied, monkeypatch):
    model = _model(bias=bias, tied=tied)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()

    def no_optimizer(*args, **kwargs):
        pytest.fail("Static partition must not construct an optimizer")

    monkeypatch.setattr(torch.optim.AdamW, "__init__", no_optimizer)
    owners = build_concat_parameter_partition(model, ordered_widths=WIDTHS)
    descriptors = build_parameter_descriptors(model, ordered_widths=WIDTHS)
    assert [owner.owner_id for owner in owners] == ["O-A", "O-B", "O-C", "O-D", "O-common"]
    ids = [id(parameter) for owner in owners for parameter in owner.parameters]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    assert [d["canonical_name"] for d in descriptors] == [name for name, _ in model.named_parameters()]
    assert descriptors == build_parameter_descriptors(copy.deepcopy(model), ordered_widths=WIDTHS)
    json.dumps(descriptors, allow_nan=False)
    for i, owner in enumerate(owners):
        assert owner.active_widths == (WIDTHS[i:] if i < 4 else WIDTHS)
        assert len(owner.parameters) == len(owner.descriptors)
        for parameter, descriptor in zip(owner.parameters, owner.descriptors, strict=True):
            assert descriptor["shape"] == list(parameter.shape)
            assert descriptor["dtype"] == str(parameter.dtype)
            assert descriptor["scalar_count"] == parameter.numel()
            assert descriptor["quarter_id"] == ("ABCD"[i] if i < 4 else None)
    if bias:
        assert sum(d["component"] == "down_bias" for d in owners[-1].descriptors) == 2
        for owner in owners[:4]:
            assert sum(d["component"] in {"gate_bias", "up_bias"} for d in owner.descriptors) == 4
    if tied:
        embedding = next(d for d in owners[-1].descriptors if d["canonical_name"] == "model.embed_tokens.weight")
        assert "lm_head.weight" in embedding["tied_aliases"]
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


@pytest.mark.parametrize("variant", ["slicing", "concat", "dense"])
def test_descriptors_predict_real_model_gradient_presence(variant):
    model = LlamaForCausalLM(_config()) if variant == "dense" else _model(variant=variant)
    widths = ("g1000",) if variant == "dense" else WIDTHS
    descriptors = build_parameter_descriptors(model, ordered_widths=widths)
    parameters = dict(model.named_parameters())
    tokens = torch.tensor([[1, 2, 3, 4]])
    for width in widths:
        model.zero_grad(set_to_none=True)
        if variant != "dense":
            model.configure_subnetwork(width)
        model(input_ids=tokens, labels=tokens).loss.backward()
        for descriptor in descriptors:
            parameter = parameters[descriptor["canonical_name"]]
            assert (parameter.grad is not None) == (width in descriptor["gradient_support"])


def test_frozen_parameters_are_described_but_excluded_from_partition():
    model = _model()
    model.model.layers[0].mlp.gate_weight_blocks[0].requires_grad_(False)
    descriptors = build_parameter_descriptors(model, ordered_widths=WIDTHS)
    frozen = [d for d in descriptors if not d["trainable"]]
    assert len(frozen) == 1
    assert frozen[0]["gradient_support"] == []
    owners = build_concat_parameter_partition(model, ordered_widths=WIDTHS)
    assert all(p.requires_grad for owner in owners for p in owner.parameters)


@pytest.mark.parametrize("damage,match", [
    ("unequal", "equal quarters"), ("overlap", "overlap"),
    ("missing", "gate_weight"), ("extra", "unclassified"),
    ("common_alias", "overlap"), ("shape", "shape"),
])
def test_partition_rejects_malformed_ffn_topology(damage, match):
    model = _model()
    mlp = model.model.layers[0].mlp
    if damage == "unequal":
        mlp.ffn_concat_block_metadata[0]["block_width"] += 1
    elif damage == "overlap":
        mlp.gate_weight_blocks[1] = mlp.gate_weight_blocks[0]
    elif damage == "missing":
        mlp.gate_weight_blocks = torch.nn.ParameterList(list(mlp.gate_weight_blocks)[:3])
    elif damage == "extra":
        mlp.unrecognized = torch.nn.Parameter(torch.ones(16))
    elif damage == "common_alias":
        model.extra_common = mlp.gate_weight_blocks[0]
    elif damage == "shape":
        mlp.down_weight_blocks[2] = torch.nn.Parameter(torch.ones(15, 16))
    with pytest.raises(ConfigError, match=match):
        build_concat_parameter_partition(model, ordered_widths=WIDTHS)


def test_partition_rejects_non_concat_and_wrong_width_order():
    with pytest.raises(ConfigError, match="concat"):
        build_concat_parameter_partition(_model(variant="slicing"), ordered_widths=WIDTHS)
    with pytest.raises(ConfigError, match="width"):
        build_concat_parameter_partition(_model(), ordered_widths=WIDTHS[::-1])


def test_partition_rejects_mixed_dense_concat_layers():
    model = _model()
    model.model.layers[1].mlp = LlamaMLP(_config())
    with pytest.raises(ConfigError, match="concat FFNs throughout"):
        build_concat_parameter_partition(model, ordered_widths=WIDTHS)


def test_campaign_dimensions_have_exact_physical_owner_counts():
    config = _config(intermediate_size=256, hidden_size=64)
    config.num_hidden_layers = 4
    config.vocab_size = 2048
    model = ModifiedLlamaForCausalLM(
        config, mlp_cls=CatLlamaMLP,
        mlp_kwargs={"gradient_membership_correction_enabled": False},
    )
    owners = build_concat_parameter_partition(model, ordered_widths=WIDTHS)
    counts = [sum(p.numel() for p in owner.parameters) for owner in owners]
    assert counts == [49152, 49152, 49152, 49152, 328256]
    assert sum(counts) == 524864


def test_descriptor_identity_includes_dtype_ties_and_trainability():
    model = _model(tied=True)
    original = stable_hash(build_parameter_descriptors(model, ordered_widths=WIDTHS))
    changed = copy.deepcopy(model).to(dtype=torch.float64)
    assert stable_hash(build_parameter_descriptors(changed, ordered_widths=WIDTHS)) != original
    changed = copy.deepcopy(model)
    changed.lm_head.weight = torch.nn.Parameter(changed.lm_head.weight.detach().clone())
    assert stable_hash(build_parameter_descriptors(changed, ordered_widths=WIDTHS)) != original
    changed = copy.deepcopy(model)
    changed.lm_head.weight.requires_grad_(False)
    assert stable_hash(build_parameter_descriptors(changed, ordered_widths=WIDTHS)) != original


OWNERS = ("O-A", "O-B", "O-C", "O-D", "O-common")


class RealFFNModel(torch.nn.Module):
    def __init__(self, variant='concat'):
        super().__init__()
        config = LlamaConfig(hidden_size=8, intermediate_size=16, num_attention_heads=2, num_key_value_heads=2, mlp_bias=True)
        config.granularities = list(WIDTHS)
        config.granularity_prefixes = {w: (i + 1) / 4 for i, w in enumerate(WIDTHS)}
        cls = CatLlamaMLP if variant == 'concat' else ModifiedLlamaMLP
        self.ffns = torch.nn.ModuleList([
            cls(config, gradient_membership_correction_enabled=False) for _ in range(2)
        ])
        self.embedding = torch.nn.Parameter(torch.ones(8))
        self.tied_head = self.embedding

    def forward(self, x, width):
        x = x * self.embedding
        for ffn in self.ffns:
            ffn.configure_subnetwork(width)
            x = x + ffn(x)
        return x.square().mean()


def training(scope='shared', scheduler='constant'):
    return {
        'optimizer_name': 'adamw',
        'optimizer_kwargs': {'betas': [0.9, 0.95], 'eps': 1e-8, 'weight_decay': 0.1},
        'optimizer_state_scope': scope,
        'optimizer_state_contract': {'ordered_granularities': list(WIDTHS)},
        'resolved_learning_rate': 0.01,
        'scheduler_name': scheduler, 'scheduler_kwargs': {},
        'resolved_warmup_steps': 0, 'max_steps': 8,
        'gradient_clip_norm': 1.0,
        'gradient_clipping': (
            {'mode': 'per_owner', 'owner_max_norms': dict.fromkeys(OWNERS, 1.0)}
            if scope == 'per_ffn_block' else {'mode': 'global', 'max_norm': 1.0}
        ),
    }


def backward(model, optimizer, width, x=None):
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(2, 3, 8) if x is None else x, width).backward()


def commit(optimizer, scheduler, width):
    if isinstance(optimizer, BlockOptimizerCollection):
        active = optimizer.active_owner_ids(width)
        for owner in active:
            optimizer.optimizer_for(owner).step()
        scheduler.step()
        scheduler.synchronize(optimizer)
        optimizer.record_successful_update(width, returned_owners=active)
    elif isinstance(optimizer, PerGranularityOptimizerCollection):
        optimizer.optimizer_for(width).step()
        scheduler.step()
        scheduler.synchronize(optimizer)
        optimizer.record_successful_update(width)
    else:
        optimizer.step()
        scheduler.step()


def assert_state_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            assert_state_equal(left, right)
    else:
        assert a == b


def test_s1_full_shaped_zero_tail_retains_momentum_decay_and_step():
    torch.manual_seed(42)
    model = RealFFNModel('slicing')
    optimizer, clock = steps.build_optimizer_and_scheduler(model, training())
    p = model.ffns[0].gate_proj.weight
    backward(model, optimizer, 'g1000')
    commit(optimizer, clock, 'g1000')
    before = p.detach().clone()
    moment = optimizer.state[p]['exp_avg'].clone()
    backward(model, optimizer, 'g250')
    assert p.grad.shape == p.shape
    assert torch.count_nonzero(p.grad[4:]) == 0
    commit(optimizer, clock, 'g250')
    assert not torch.equal(p[4:], before[4:])
    assert not torch.equal(p[4:], before[4:] * (1 - 0.01 * 0.1))
    torch.testing.assert_close(optimizer.state[p]['exp_avg'][4:], moment[4:] * 0.9)
    assert optimizer.state[p]['step'].item() == 2


def test_s2_selected_history_isolation_full_allocation_and_never_used_zero_tails():
    model = RealFFNModel('slicing')
    optimizer, clock = steps.build_optimizer_and_scheduler(model, training('per_granularity'))
    backward(model, optimizer, 'g1000')
    commit(optimizer, clock, 'g1000')
    wide = copy.deepcopy(optimizer.optimizer_for('g1000').state_dict())
    backward(model, optimizer, 'g250')
    commit(optimizer, clock, 'g250')
    assert_state_equal(wide, optimizer.optimizer_for('g1000').state_dict())
    for ffn in model.ffns:
        for p, tail in ((ffn.gate_proj.weight, (slice(4, None),)),
                        (ffn.up_proj.weight, (slice(4, None),)),
                        (ffn.down_proj.weight, (slice(None), slice(4, None)))):
            state = optimizer.optimizer_for('g250').state[p]
            for key in ('exp_avg', 'exp_avg_sq'):
                assert state[key].shape == p.shape
                assert torch.count_nonzero(state[key][tail]) == 0
            assert state['step'].item() == 1
    assert not optimizer.optimizer_for('g500').state


@pytest.mark.parametrize('scope', ['shared', 'per_granularity', 'per_ffn_block'])
def test_concat_absent_quarters_are_bitwise_frozen_but_present_zero_updates(scope):
    model = RealFFNModel()
    optimizer, clock = steps.build_optimizer_and_scheduler(model, training(scope))
    partition = build_concat_parameter_partition(model, ordered_widths=WIDTHS)
    backward(model, optimizer, 'g1000')
    commit(optimizer, clock, 'g1000')
    owner_optimizer = (optimizer.optimizer_for('O-D') if scope == 'per_ffn_block' else
                       optimizer.optimizer_for('g1000') if scope == 'per_granularity' else optimizer)
    before = [(p.detach().clone(), copy.deepcopy(owner_optimizer.state[p])) for p in partition[3].parameters]
    backward(model, optimizer, 'g250')
    assert all(p.grad is None for owner in partition[1:4] for p in owner.parameters)
    commit(optimizer, clock, 'g250')
    for p, (weight, state) in zip(partition[3].parameters, before, strict=True):
        assert torch.equal(p, weight)
        assert_state_equal(owner_optimizer.state[p], state)
    backward(model, optimizer, 'g1000')
    for p in model.parameters():
        assert p.grad is not None
        p.grad.zero_()
    p = partition[3].parameters[0]
    before = p.detach().clone()
    commit(optimizer, clock, 'g1000')
    assert not torch.equal(p, before)
    assert owner_optimizer.state[p]['step'].item() == 2


def test_c2_lazy_quarter_history_multiplicities():
    model = RealFFNModel()
    optimizer, clock = steps.build_optimizer_and_scheduler(model, training('per_granularity'))
    assert all(not entry.optimizer.state for entry in optimizer.entries)
    for width in WIDTHS:
        backward(model, optimizer, width)
        commit(optimizer, clock, width)
    for owner, expected in zip(build_concat_parameter_partition(model, ordered_widths=WIDTHS), (4, 3, 2, 1, 4), strict=True):
        for p in owner.parameters:
            assert sum(p in entry.optimizer.state for entry in optimizer.entries) == expected
            for entry in optimizer.entries:
                if p in entry.optimizer.state:
                    assert entry.optimizer.state[p]['step'].item() == 1


def test_c3_tied_bias_partition_and_ordered_successful_accounting():
    model = RealFFNModel()
    optimizer, clock = steps.build_optimizer_and_scheduler(model, training('per_ffn_block'))
    assert isinstance(optimizer, BlockOptimizerCollection)
    assert optimizer.ordered_owner_ids == OWNERS
    parameters = [p for owner in optimizer.owners for p in owner.parameters]
    assert len(parameters) == len({id(p) for p in parameters}) == len(list(model.parameters()))
    common = optimizer.owners[-1]
    assert any(d['tied_aliases'] == ['tied_head'] for d in common.descriptors)
    assert all(any(p is ffn.down_bias for p in common.parameters) for ffn in model.ffns)
    for width in WIDTHS:
        backward(model, optimizer, width)
        commit(optimizer, clock, width)
    assert optimizer.successful_update_counts == dict(zip(OWNERS, (4, 3, 2, 1, 4)))
    assert optimizer.width_selection_counts == dict.fromkeys(WIDTHS, 1)
    assert optimizer.total_successful_updates == clock.position == 4
    assert optimizer.state_dict()['ordered_owners'][0]['descriptors'] == list(optimizer.owners[0].descriptors)


@pytest.mark.parametrize('width,n', zip(WIDTHS, (2, 3, 4, 5)))
def test_independent_clipping_bounds_and_inactive_nulls(width, n):
    model = RealFFNModel()
    optimizer, _ = steps.build_optimizer_and_scheduler(model, training('per_ffn_block'))
    backward(model, optimizer, width)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.fill_(10)
    rng = (random.getstate(), torch.get_rng_state().clone())
    observation = steps.clip_optimizer_gradients(model, training('per_ffn_block'), width, owners=optimizer.owners)
    assert random.getstate() == rng[0]
    assert torch.equal(torch.get_rng_state(), rng[1])
    assert observation['combined_post_norm'] == pytest.approx(math.sqrt(n), abs=2e-6)
    assert observation['global_coefficient'] is None
    for owner in optimizer.owners:
        item = observation['groups'][owner.owner_id]
        if width in owner.active_widths:
            assert item['post_norm'] == pytest.approx(1.0, abs=2e-6)
            assert 0 < item['coefficient'] < 1
        else:
            assert item == {'active': False, 'pre_norm': None, 'post_norm': None, 'coefficient': None, 'max_norm': None}


def test_owner_clipping_is_independent_and_active_zero_coefficient_is_one():
    model = RealFFNModel()
    owners = build_concat_parameter_partition(model, ordered_widths=WIDTHS)
    results = []
    for common_scale in (0, 100):
        model.zero_grad(set_to_none=True)
        for owner in (owners[0], owners[-1]):
            for p in owner.parameters:
                p.grad = torch.full_like(p, 5 if owner.owner_id == 'O-A' else common_scale)
        observation = steps.clip_optimizer_gradients(model, training('per_ffn_block'), 'g250', owners=owners)
        results.append(observation)
        if common_scale == 0:
            assert observation['groups']['O-common']['coefficient'] == 1
            assert observation['groups']['O-common']['active']
    assert results[0]['groups']['O-A'] == results[1]['groups']['O-A']


def test_c1_matches_diagnostic_global_c3_parameters_and_every_history_tensor():
    torch.manual_seed(42)
    shared = RealFFNModel()
    block = copy.deepcopy(shared)
    c1, s1 = steps.build_optimizer_and_scheduler(shared, training(scheduler='cosine'))
    c3, s3 = steps.build_optimizer_and_scheduler(block, training('per_ffn_block', 'cosine'), _diagnostic_global_clip=True)
    owners = build_concat_parameter_partition(shared, ordered_widths=WIDTHS)
    for width in ('g1000', 'g250', 'g750', 'g500') * 2:
        x = torch.randn(2, 3, 8) * 10
        backward(shared, c1, width, x)
        backward(block, c3, width, x)
        obs = steps.clip_optimizer_gradients(shared, training(), width, owners=owners)
        steps.clip_optimizer_gradients(block, training('per_ffn_block'), width, owners=c3.owners, diagnostic_global_clip=c3.diagnostic_global_clip)
        for group in obs['groups'].values():
            if group['active']:
                assert group['coefficient'] == obs['global_coefficient']
        commit(c1, s1, width)
        commit(c3, s3, width)
        for left, right in zip(shared.parameters(), block.parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
            owner = next(o for o in c3.owners if any(p is right for p in o.parameters))
            state = c3.optimizer_for(owner.owner_id).state.get(right, {})
            for key, value in c1.state.get(left, {}).items():
                torch.testing.assert_close(value, state[key], rtol=1e-6, atol=1e-7)


def runtime_fixture(tmp_path, *, scope='per_ffn_block', max_steps=8, widths=WIDTHS, device='cpu'):
    from src.training.checkpointing import build_initial_continuation_state
    from src.training.modeling import build_model
    from src.utils.config import resolve_run_config

    if device == 'cuda':
        if not torch.cuda.is_available():
            pytest.skip('CUDA unavailable; bf16 diagnostic requires one GPU')

    config = resolve_run_config(
        'tests/fixtures/per_granularity_optimizer_smoke.yaml',
        output_dir=tmp_path / 'per-granularity-optimizer-smoke-001',
        overrides={
            'model.variant': 'concat',
            'model.granularities': list(widths),
            'model.granularity_prefixes': {w: (i + 1) / len(widths) for i, w in enumerate(widths)},
            'model.global_sampling_schedule': 'random_with_replacement',
            'model.d_model': 16, 'model.num_layers': 2, 'model.vocab_size': 32,
            'model.context_length': 8,
            'training.max_steps': max_steps, 'training.token_budget': max_steps * 8,
            'training.batch_size_per_process': 1,
            'training.mixed_precision': 'bf16' if device == 'cuda' else 'none',
            'training.optimizer.state_scope': scope,
            'training.gradient_clipping': training(scope)['gradient_clipping'],
            'training.warmup_steps': 2,
            'run.continuation.enabled': False,
            'outputs.save_checkpoints': False,
            'evaluation.validation.enabled': False,
            'evaluation.validation.run_at_completion': False,
            'evaluation.validation.interval_steps': 0,
            'training.eval_interval': 0,
        },
    )
    from src.utils.reproducibility import (
        configure_strict_determinism, deterministic_runtime_settings,
        STRICT_CUBLAS_WORKSPACE_CONFIG,
    )
    # Multiple diagnostic bundles share one pytest process/CUDA context. Set the
    # strict recipe before the first device operation, then verify it on reuse.
    if not torch.cuda.is_initialized():
        configure_strict_determinism(config)
    assert deterministic_runtime_settings() == {
        'mode': 'strict',
        'cublas_workspace_config': STRICT_CUBLAS_WORKSPACE_CONFIG,
        'deterministic_algorithms': True,
        'cudnn_benchmark': False,
        'cudnn_deterministic': True,
        'cudnn_allow_tf32': False,
        'cuda_matmul_allow_tf32': False,
    }
    if device == 'cuda':
        if not torch.cuda.is_bf16_supported():
            pytest.skip('CUDA device does not support bf16')
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42)
    from src.training.distributed import resolve_runtime_settings
    resolve_runtime_settings(config['training'], device, single_process=True)
    model = build_model(config).to(device)
    optimizer, scheduler = steps.build_optimizer_and_scheduler(model, config['training'])
    batches = [{'input_ids': torch.arange(1, 9).reshape(1, 8),
                'labels': torch.arange(1, 9).reshape(1, 8)} for _ in range(2)]
    return config, model, optimizer, scheduler, batches, build_initial_continuation_state(config)


def test_runtime_fixture_sets_determinism_before_device_operations(tmp_path, monkeypatch):
    from src.utils.reproducibility import deterministic_runtime_settings, STRICT_CUBLAS_WORKSPACE_CONFIG

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    monkeypatch.setattr(torch.backends.cudnn, 'deterministic', False)

    def check_bf16():
        settings = deterministic_runtime_settings()
        assert settings['deterministic_algorithms'] is True
        assert settings['cudnn_deterministic'] is True
        assert settings['cublas_workspace_config'] == STRICT_CUBLAS_WORKSPACE_CONFIG
        # End the test before requiring real GPU hardware.
        raise RuntimeError('reached CUDA with strict determinism configured')

    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', check_bf16)
    with pytest.raises(RuntimeError, match='reached CUDA with strict determinism configured'):
        runtime_fixture(tmp_path, device='cuda')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_trainer_orders_owner_calls_then_clock_and_publishes_complete_updates(tmp_path, monkeypatch, device):
    from src.training.run import _validate_restored_optimizer_ownership_runtime, _resource_attempt_observer
    import time

    started_at = time.perf_counter()
    config, model, optimizer, clock, batches, state = runtime_fixture(tmp_path, device=device)
    observer = _resource_attempt_observer(config, torch.device(device), started_at=started_at, source_checkpoint=None)
    optimizer._resource_observer = observer
    events, observations, forwards = [], [], []
    model.register_forward_hook(lambda *args: forwards.append(1))
    compute_dtypes = []
    model.lm_head.register_forward_hook(lambda module, args, output: compute_dtypes.append(output.dtype))
    for entry in optimizer.entries:
        original = entry.optimizer.step
        def tracked_step(*args, _owner=entry.owner_id, _step=original, **kwargs):
            assert state['update_in_flight'] is True
            assert optimizer.total_successful_updates == clock.position
            assert state['optimizer_update_counts'] == optimizer.successful_update_counts
            events.append(_owner)
            return _step(*args, **kwargs)
        monkeypatch.setattr(entry.optimizer, 'step', tracked_step)
    original_clock_step = clock.step
    def clock_step():
        assert state['update_in_flight'] is True
        events.append('clock')
        original_clock_step()
    monkeypatch.setattr(clock, 'step', clock_step)

    def committed(*, step, tokens_seen):
        width = optimizer.last_active_granularity
        assert events == [*optimizer.active_owner_ids(width), 'clock']
        events.clear()
        assert state['update_in_flight'] is False
        assert optimizer.total_successful_updates == clock.position == step
        assert tokens_seen == state['content_tokens_seen'] == step * 8
        assert state['microstep'] == step
        assert state['batch_index'] == (step - 1) % 2 + 1
        assert state['epoch'] == (step - 1) // 2
        assert state['global_sampling_state']['exposure_counts'] == optimizer.width_selection_counts
        assert state['last_clipping_observation']['step'] == step
        observation = state['last_clipping_observation']
        assert math.isfinite(observation['combined_pre_norm'])
        assert math.isfinite(observation['combined_post_norm'])
        assert observation['combined_post_norm'] <= math.sqrt(len(optimizer.active_owner_ids(width))) + 2e-6
        for group in observation['groups'].values():
            if group['active']:
                assert math.isfinite(group['pre_norm'])
                assert 0 <= group['post_norm'] <= 1.0 + 2e-6
                assert 0 < group['coefficient'] <= 1
        _validate_restored_optimizer_ownership_runtime(config, state, optimizer, clock)
        observations.append(copy.deepcopy(state['last_clipping_observation']))

    steps.train_for_steps(config, model, batches, [], optimizer, clock, torch.device(device),
                          run_state=state, successful_step_callback=committed)
    assert len(forwards) == len(observations) == 8
    assert sum(optimizer.width_selection_counts.values()) == 8
    assert all(entry.optimizer.param_groups[0]['lr'] == 0 for entry in optimizer.entries)
    observer(run_state=state, boundary='completed')
    peaks = state['resource_summary']
    assert peaks['attempted_steps'] == 8
    assert peaks['measurement_complete'] is True
    assert all(torch.isfinite(p).all() for p in model.parameters())
    for entry in optimizer.entries:
        for history in entry.optimizer.state.values():
            assert all(torch.isfinite(value).all() for value in history.values() if torch.is_tensor(value))
    if device == 'cuda':
        assert compute_dtypes == [torch.bfloat16] * 8
        assert 0 < peaks['peak_allocated_bytes'] <= peaks['peak_reserved_bytes']
        (tmp_path / 'cuda_bf16_diagnostic.json').write_text(json.dumps({
            'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
            'compute_dtype': str(compute_dtypes[0]), 'updates': 8, 'resources': peaks,
        }, indent=2))
    state['optimizer_width_selection_counts']['g250'] += 1
    with pytest.raises(ValueError, match='unreconciled'):
        _validate_restored_optimizer_ownership_runtime(config, state, optimizer, clock)


@pytest.mark.parametrize('failure', ['loss', 'gradient', 'rate'])
def test_nonfinite_or_unsynchronized_update_is_rejected_before_any_owner_step(tmp_path, monkeypatch, failure):
    config, model, optimizer, clock, batches, state = runtime_fixture(tmp_path)
    before = copy.deepcopy(model.state_dict())
    for entry in optimizer.entries:
        monkeypatch.setattr(entry.optimizer, 'step', lambda: pytest.fail('Owner step reached'))
    if failure == 'loss':
        def bad_loss(module, args, output):
            output.loss = output.loss * float('nan')
        model.register_forward_hook(bad_loss)
    elif failure == 'gradient':
        next(model.parameters()).register_hook(lambda grad: grad * float('inf'))
    else:
        # Corrupt an inactive owner's rate after the initial boundary check.
        def bad_rate(module, args, output):
            optimizer.optimizer_for('O-D').param_groups[0]['lr'] = 0.7
        model.register_forward_hook(bad_rate)
    with pytest.raises(RuntimeError):
        steps.train_for_steps(config, model, batches, [], optimizer, clock, torch.device('cpu'), run_state=state)
    assert clock.position == optimizer.total_successful_updates == 0
    assert not state.get('update_in_flight')
    assert_state_equal(before, model.state_dict())


def test_explicit_global_clipping_preserves_other_concat_layouts(tmp_path):
    config, model, optimizer, clock, batches, state = runtime_fixture(
        tmp_path, scope="shared", max_steps=2, widths=("narrow", "full"),
    )
    steps.train_for_steps(
        config, model, batches, [], optimizer, clock, torch.device("cpu"), run_state=state,
    )
    assert state["last_completed_step"] == 2
    assert state["last_clipping_observation"]["mode"] == "global"
    assert state["last_clipping_observation"]["combined_post_norm"] <= 1.000001


@pytest.mark.parametrize('failure', ['O-A', 'O-C', 'O-common', 'scheduler', 'accounting'])
def test_mutating_failure_poison_and_all_save_paths_preserve_durable_checkpoint(tmp_path, monkeypatch, failure):
    from test_optimizer_ownership_resume import fixture, train, save
    from src.training import checkpointing as cp
    import hashlib

    bundle = fixture(tmp_path)
    config, model, optimizer, clock, batches, state = bundle
    train(bundle, stop=1)
    path = tmp_path / 'latest.pt'
    save(bundle, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    original_select = steps._select_optimizer_window_action
    def select(*args, **kwargs):
        action = original_select(*args, **kwargs)
        action['granularities'] = ['g1000']
        kwargs['run_state']['global_sampling_state']['held_granularity'] = 'g1000'
        return action
    monkeypatch.setattr(steps, '_select_optimizer_window_action', select)
    target = clock if failure == 'scheduler' else optimizer if failure == 'accounting' else optimizer.optimizer_for(failure)
    method = 'record_successful_update' if failure == 'accounting' else 'step'
    original = getattr(target, method)
    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('injected mutation failure')
    monkeypatch.setattr(target, method, fail)
    with pytest.raises(RuntimeError, match='injected'): train(bundle)
    assert state['optimizer_poisoned'] and state['update_in_flight']
    record = state['optimizer_failure']
    assert record['pending_step'] == 2
    expected = list(OWNERS) if failure in ('scheduler', 'accounting') else list(OWNERS[:OWNERS.index(failure)])
    assert record['returned_owners'] == expected
    for reason in ('periodic', 'failure', 'signal', 'finalization'):
        with pytest.raises(ConfigError, match='unsafe|poison'):
            cp.maybe_write_latest_checkpoint(config, model, optimizer, clock, None, state, reason=reason, step=2)
    with pytest.raises(ConfigError, match='unsafe|poison'): save(bundle, path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
