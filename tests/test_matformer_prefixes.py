import pytest
import torch
from transformers import LlamaConfig

from models.ffn import (
    CatLlamaMLP,
    ModifiedLlamaMLP,
    get_concat_layout_diagnostic,
    get_ffn_prefix_metadata,
    get_prefix_membership_segment_metadata,
    get_concat_block_metadata,
    get_concat_block_membership_counts,
    get_concat_block_membership_counts_from_metadata,
    get_concat_gradient_membership_correction_scales,
    get_concat_gradient_membership_correction_scales_from_metadata,
    granularity_prefix_width,
)
from models.granularity import (
    MATFORMER_GRANULARITY_ORDER,
    build_granularity_pattern,
    expand_layer_granularity_pattern,
    get_block_membership_counts,
    get_gradient_membership_correction_scales,
    get_granularity_metadata,
    summarize_granularity_pattern,
)
from models.wiring import ModifiedLlamaForCausalLM
from models.wiring import build_global_granularity_pattern
from utils.config import resolve_run_config


def tiny_llama_config(
    num_hidden_layers=2,
    intermediate_size=64,
    granularity_prefixes=None,
    granularities=None,
):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=16,
        tie_word_embeddings=False,
    )
    if granularity_prefixes is not None:
        config.granularity_prefixes = granularity_prefixes
    if granularities is not None:
        config.granularities = granularities
    return config


def test_granularity_metadata_matches_paper_ratios_and_prefix_widths():
    metadata = get_ffn_prefix_metadata(intermediate_size=64)

    assert [entry["name"] for entry in metadata] == ["s", "m", "l", "xl"]
    assert [entry["display_name"] for entry in metadata] == ["S", "M", "L", "XL"]
    assert [entry["ffn_ratio"] for entry in metadata] == [0.5, 1.0, 2.0, 4.0]
    assert [entry["full_intermediate_fraction"] for entry in metadata] == [
        0.125,
        0.25,
        0.5,
        1.0,
    ]
    assert [entry["prefix_width"] for entry in metadata] == [8, 16, 32, 64]


def test_granularity_metadata_follows_configured_prefix_map():
    metadata = get_ffn_prefix_metadata(
        intermediate_size=100,
        granularity_prefixes={
            "s": 0.1,
            "m": 0.2,
            "l": 0.4,
            "xl": 1.0,
        },
    )

    assert [entry["full_intermediate_fraction"] for entry in metadata] == [
        0.1,
        0.2,
        0.4,
        1.0,
    ]
    assert [entry["prefix_width"] for entry in metadata] == [10, 20, 40, 100]


def test_concat_block_metadata_preserves_granularity_boundaries():
    metadata = get_concat_block_metadata(intermediate_size=64)

    assert [entry["block_width"] for entry in metadata] == [8, 8, 16, 32]
    assert [entry["prefix_width"] for entry in metadata] == [8, 16, 32, 64]
    assert [entry["cumulative_prefix_width"] for entry in metadata] == [
        8,
        16,
        32,
        64,
    ]


def test_concat_block_metadata_can_be_derived_from_config_prefix_map():
    metadata = get_concat_block_metadata(
        intermediate_size=100,
        granularity_prefixes={
            "s": 0.1,
            "m": 0.2,
            "l": 0.4,
            "xl": 1.0,
        },
    )

    assert [entry["block_width"] for entry in metadata] == [10, 10, 20, 60]
    assert [entry["prefix_width"] for entry in metadata] == [10, 20, 40, 100]


def test_granularity_metadata_helpers_build_stable_pattern_summaries():
    metadata = get_granularity_metadata("l")
    pattern = build_granularity_pattern(
        pattern_type="per_layer",
        selected_granularities=("s", "m", "l"),
        layer_count=3,
        repeatable_source=(
            "debug-nested-001",
            "model.granularity_sampling_mode=per_layer",
        ),
    )

    assert metadata == {
        "display_name": "L",
        "ffn_ratio": 2.0,
        "full_intermediate_fraction": 0.5,
    }
    assert pattern.pattern_type == "per_layer"
    assert pattern.selected_granularities == ("s", "m", "l")
    assert pattern.layer_count == 3
    assert pattern.repeatable_source == (
        "debug-nested-001",
        "model.granularity_sampling_mode=per_layer",
    )
    assert summarize_granularity_pattern(pattern) == {
        "pattern_type": "per_layer",
        "selected_granularities": ("s", "m", "l"),
        "layer_count": 3,
        "repeatable_source": (
            "debug-nested-001",
            "model.granularity_sampling_mode=per_layer",
        ),
    }


def test_concat_membership_counts_follow_concat_boundaries():
    metadata = get_concat_block_metadata(64)

    assert get_concat_block_membership_counts(
        64,
        ["s", "m", "l", "xl"],
    ) == [4, 3, 2, 1]
    assert get_concat_block_membership_counts_from_metadata(
        metadata,
        ["s", "m", "l", "xl"],
    ) == [4, 3, 2, 1]
    assert get_concat_gradient_membership_correction_scales(
        64,
        ["s", "m", "l", "xl"],
    ) == [1.0, 4 / 3, 2.0, 4.0]
    assert get_concat_gradient_membership_correction_scales_from_metadata(
        metadata,
        ["s", "m", "l", "xl"],
    ) == [1.0, 4 / 3, 2.0, 4.0]


def test_concat_layout_diagnostic_matches_instantiated_blocks():
    diagnostic = get_concat_layout_diagnostic(64, ["s", "m", "l", "xl"])

    assert diagnostic["intermediate_size"] == 64
    assert diagnostic["granularities"] == ["s", "m", "l", "xl"]
    assert diagnostic["block_widths"] == [8, 8, 16, 32]
    assert diagnostic["prefix_widths"] == [8, 16, 32, 64]
    assert diagnostic["gradient_membership_counts"] == [4, 3, 2, 1]
    assert diagnostic["gradient_membership_correction_scales"] == [
        1.0,
        4 / 3,
        2.0,
        4.0,
    ]


def test_prefix_widths_are_strictly_ordered():
    widths = [
        granularity_prefix_width(64, granularity)
        for granularity in MATFORMER_GRANULARITY_ORDER
    ]

    assert widths == sorted(widths)
    assert len(set(widths)) == len(widths)


def test_mlp_configures_prefix_tensor_shapes_and_forward_output():
    config = tiny_llama_config(num_hidden_layers=1)
    mlp = ModifiedLlamaMLP(config)
    x = torch.randn(2, 3, config.hidden_size)

    for granularity in MATFORMER_GRANULARITY_ORDER:
        mlp.configure_subnetwork(granularity)
        prefix_width = granularity_prefix_width(
            config.intermediate_size,
            granularity,
        )

        assert mlp.current_granularity == granularity
        assert mlp.current_subset_hd == prefix_width
        assert mlp.gate_proj.weight[:prefix_width].shape == (
            prefix_width,
            config.hidden_size,
        )
        assert mlp.up_proj.weight[:prefix_width].shape == (
            prefix_width,
            config.hidden_size,
        )
        assert mlp.down_proj.weight[:, :prefix_width].shape == (
            config.hidden_size,
            prefix_width,
        )
        assert mlp(x).shape == x.shape


def test_modified_mlp_uses_configured_prefix_metadata():
    config = tiny_llama_config(
        num_hidden_layers=1,
        intermediate_size=100,
        granularity_prefixes={
            "s": 0.1,
            "m": 0.2,
            "l": 0.4,
            "xl": 1.0,
        },
        granularities=["s", "m", "l", "xl"],
    )
    mlp = ModifiedLlamaMLP(config)

    assert [entry["prefix_width"] for entry in mlp.ffn_prefix_metadata] == [
        10,
        20,
        40,
        100,
    ]
    mlp.configure_subnetwork("l")
    assert mlp.current_subset_hd == 40


def test_cat_mlp_uses_concat_block_counts_for_subnetwork_configuration():
    config = tiny_llama_config(num_hidden_layers=1)
    mlp = CatLlamaMLP(config)

    for granularity, expected_blocks in [("s", 1), ("m", 2), ("l", 3), ("xl", 4)]:
        mlp.configure_subnetwork(granularity)
        assert mlp.current_subset_blocks == expected_blocks
    assert mlp.gradient_membership_counts == [4, 3, 2, 1]
    assert mlp.gradient_membership_correction_scales == [1.0, 4 / 3, 2.0, 4.0]


def test_cat_mlp_uses_configured_concat_block_metadata():
    config = tiny_llama_config(
        num_hidden_layers=1,
        intermediate_size=100,
        granularity_prefixes={
            "s": 0.1,
            "m": 0.2,
            "l": 0.4,
            "xl": 1.0,
        },
        granularities=["s", "m", "l", "xl"],
    )
    mlp = CatLlamaMLP(config)

    assert [entry["prefix_width"] for entry in mlp.ffn_prefix_metadata] == [
        10,
        20,
        40,
        100,
    ]
    assert [entry["block_width"] for entry in mlp.ffn_concat_block_metadata] == [
        10,
        10,
        20,
        60,
    ]
    mlp.configure_subnetwork("m")
    assert mlp.current_subset_hd == 20


def test_modified_mlp_defaults_to_gradient_membership_correction_enabled():
    mlp = ModifiedLlamaMLP(tiny_llama_config(num_hidden_layers=1))

    assert mlp.gradient_membership_correction_enabled is True


def test_invalid_granularity_is_rejected():
    mlp = ModifiedLlamaMLP(tiny_llama_config(num_hidden_layers=1))

    with pytest.raises(ValueError, match="Unknown MatFormer granularity"):
        mlp.configure_subnetwork("xs")


def test_model_configures_all_layer_prefixes():
    config = tiny_llama_config(num_hidden_layers=2)
    model = ModifiedLlamaForCausalLM(config)

    model.configure_subnetwork("m")

    layer_widths = [
        layer.mlp.current_subset_hd
        for layer in model.model.layers
        if isinstance(layer.mlp, ModifiedLlamaMLP)
    ]
    assert layer_widths == [16, 16]
    assert model.ffn_prefix_metadata[-1]["prefix_width"] == config.intermediate_size


def test_explicit_global_sampling_path_uses_all_configured_granularities():
    resolved = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        overrides=["model.granularity_sampling_mode=global"],
    )

    pattern = build_global_granularity_pattern(resolved)

    assert pattern.pattern_type == "single"
    assert pattern.selected_granularities == ("s", "m", "l", "xl")
    assert pattern.layer_count == resolved["model"]["num_layers"]
    assert pattern.repeatable_source == (
        "debug-nested-001",
        "model.granularity_sampling_mode=global",
    )


def test_per_layer_sampling_path_repeats_layer_choices_across_blocks():
    resolved = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        overrides=["model.granularity_sampling_mode=per_layer"],
    )

    assert resolved["model"]["granularity_sampling_mode"] == "per_layer"
    assert resolved["run"]["sampling_mode"] == "nested-random"
    assert resolved["model"]["granularity_pattern_provenance"] == {
        "pattern_type": "per_layer",
        "scope": "model",
        "source": "model.granularity_sampling_mode",
        "requested_alias": None,
        "layer_count": resolved["model"]["num_layers"],
        "available_granularities": ["s", "m", "l", "xl"],
    }

    pattern = expand_layer_granularity_pattern(["s", "m"], num_layers=4)

    assert pattern == ["s", "m", "s", "m"]
    assert len(pattern) == 4


def test_layer_granularity_pattern_repeats_across_model_layers():
    config = tiny_llama_config(num_hidden_layers=4)
    model = ModifiedLlamaForCausalLM(config)

    model.configure_layer_granularities(["xl", "s"])

    assert model.current_layer_granularities == ["xl", "s", "xl", "s"]
    assert [
        layer.current_granularity for layer in model.matformer_layers
    ] == ["xl", "s", "xl", "s"]
    assert [
        layer.current_subset_hd for layer in model.matformer_layers
    ] == [64, 8, 64, 8]


def test_layer_granularity_pattern_rejects_unknown_granularity():
    with pytest.raises(ValueError, match="Unknown MatFormer granularity"):
        expand_layer_granularity_pattern(["xl", "tiny"], num_layers=4)


def test_layer_granularity_pattern_must_be_non_empty():
    with pytest.raises(ValueError, match="non-empty"):
        expand_layer_granularity_pattern([], num_layers=4)


def test_gradient_membership_correction_scales_match_configured_granularities():
    total_blocks = 8

    assert get_block_membership_counts(
        ["s", "m", "l", "xl"],
        total_blocks=total_blocks,
    ) == [4, 3, 2, 2, 1, 1, 1, 1]
    assert get_gradient_membership_correction_scales(
        ["s", "m", "l", "xl"],
        total_blocks=total_blocks,
    ) == [1.0, 4 / 3, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0]

    assert get_block_membership_counts(
        ["m", "xl"],
        total_blocks=total_blocks,
    ) == [2, 2, 1, 1, 1, 1, 1, 1]
    assert get_gradient_membership_correction_scales(
        ["m", "xl"],
        total_blocks=total_blocks,
    ) == [1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]


def test_prefix_membership_segment_metadata_matches_sliced_prefix_boundaries():
    metadata = get_prefix_membership_segment_metadata(64, ["s", "m", "l", "xl"])

    assert [(segment["start"], segment["end"]) for segment in metadata] == [
        (0, 8),
        (8, 16),
        (16, 32),
        (32, 64),
    ]
    assert [segment["membership_count"] for segment in metadata] == [4, 3, 2, 1]
    assert [segment["scale"] for segment in metadata] == [1.0, 4 / 3, 2.0, 4.0]


def test_cat_llama_scales_active_block_gradients_by_inverse_membership():
    torch.manual_seed(0)
    config = tiny_llama_config(num_hidden_layers=1)
    corrected = CatLlamaMLP(
        config,
        trained_granularities=["s", "m", "l", "xl"],
    )
    baseline = CatLlamaMLP(config, trained_granularities=["xl"])
    baseline.load_state_dict(corrected.state_dict())

    x = torch.randn(2, 3, config.hidden_size)
    corrected.configure_subnetwork("m")
    baseline.configure_subnetwork("m")

    corrected(x).sum().backward()
    baseline(x).sum().backward()

    assert torch.allclose(
        corrected.gate_weight_blocks[0].grad,
        baseline.gate_weight_blocks[0].grad,
    )
    assert torch.allclose(
        corrected.gate_weight_blocks[1].grad,
        baseline.gate_weight_blocks[1].grad * (4 / 3),
    )
    assert corrected.gate_weight_blocks[2].grad is None
    assert baseline.gate_weight_blocks[2].grad is None


def test_cat_llama_can_disable_gradient_membership_correction():
    torch.manual_seed(0)
    config = tiny_llama_config(num_hidden_layers=1)
    corrected = CatLlamaMLP(
        config,
        trained_granularities=["s", "m", "l", "xl"],
        gradient_membership_correction_enabled=True,
    )
    uncorrected = CatLlamaMLP(
        config,
        trained_granularities=["s", "m", "l", "xl"],
        gradient_membership_correction_enabled=False,
    )
    uncorrected.load_state_dict(corrected.state_dict())

    x = torch.randn(2, 3, config.hidden_size)
    corrected.configure_subnetwork("m")
    uncorrected.configure_subnetwork("m")

    corrected(x).sum().backward()
    uncorrected(x).sum().backward()

    assert uncorrected.gradient_membership_correction_enabled is False
    assert torch.allclose(
        corrected.gate_weight_blocks[1].grad,
        uncorrected.gate_weight_blocks[1].grad * (4 / 3),
    )


def test_modified_llama_scales_active_prefix_gradients_by_inverse_membership():
    torch.manual_seed(0)
    config = tiny_llama_config(num_hidden_layers=1)
    corrected = ModifiedLlamaMLP(
        config,
        trained_granularities=["s", "m", "l", "xl"],
        gradient_membership_correction_enabled=True,
    )
    baseline = ModifiedLlamaMLP(
        config,
        trained_granularities=["s", "m", "l", "xl"],
        gradient_membership_correction_enabled=False,
    )
    baseline.load_state_dict(corrected.state_dict())

    x = torch.randn(2, 3, config.hidden_size)
    corrected.configure_subnetwork("m")
    baseline.configure_subnetwork("m")

    corrected(x).sum().backward()
    baseline(x).sum().backward()

    assert torch.allclose(
        corrected.gate_proj.weight.grad[:8],
        baseline.gate_proj.weight.grad[:8],
    )
    assert torch.allclose(
        corrected.gate_proj.weight.grad[8:16],
        baseline.gate_proj.weight.grad[8:16] * (4 / 3),
    )
    assert torch.allclose(
        corrected.down_proj.weight.grad[:, 8:16],
        baseline.down_proj.weight.grad[:, 8:16] * (4 / 3),
    )
    assert torch.count_nonzero(corrected.gate_proj.weight.grad[16:]) == 0
    assert torch.count_nonzero(baseline.gate_proj.weight.grad[16:]) == 0
