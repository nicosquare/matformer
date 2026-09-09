"""Real-model checks for static optimizer ownership and physical support."""

import copy
import json

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP

from src.models.ffn import CatLlamaMLP, ModifiedLlamaMLP
from src.models.wiring import ModifiedLlamaForCausalLM
from src.training.optimizer_state import (
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
