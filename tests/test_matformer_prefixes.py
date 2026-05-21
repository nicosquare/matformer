import pytest
import torch
from transformers import LlamaConfig

from modified_llama import (
    MATFORMER_GRANULARITY_ORDER,
    ModifiedLlamaForCausalLM,
    ModifiedLlamaMLP,
    expand_layer_granularity_pattern,
    get_ffn_prefix_metadata,
    granularity_prefix_width,
)


def tiny_llama_config(num_hidden_layers=2):
    return LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=64,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=16,
        tie_word_embeddings=False,
    )


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
