import torch.nn as nn
import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from models.ffn import CatLlamaMLP, ModifiedLlamaMLP
from models.wiring import ModifiedLlamaForCausalLM
from training.run import build_model
from utils.config import resolve_run_config
from utils.model_size import (
    count_embedding_parameters,
    count_lm_head_parameters,
    count_non_embedding_parameters,
    count_parameters,
    model_parameter_counts,
)


class TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(10, 4)
        self.projection = nn.Linear(4, 8)
        self.lm_head = nn.Linear(8, 10, bias=False)


class TinyDisaggregatedLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(10, 4)
        self.model.layers = nn.ModuleList([TinyDecoderLayer()])
        self.lm_head = nn.Linear(4, 10, bias=False)


class TinyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(4, 4, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(4, 8, bias=False)
        self.mlp.up_proj = nn.Linear(4, 8, bias=False)
        self.mlp.down_proj = nn.Linear(8, 4, bias=False)
        self.input_layernorm = nn.LayerNorm(4)


def tiny_llama_config():
    return LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=16,
        tie_word_embeddings=False,
    )


def test_non_embedding_counts_exclude_embeddings_and_lm_head():
    model = TinyLanguageModel()

    counts = model_parameter_counts(model)

    assert counts["total_parameters"] == 160
    assert counts["embedding_parameters"] == 40
    assert counts["lm_head_parameters"] == 80
    assert counts["non_embedding_parameters"] == 40
    assert counts["lm_head_counting"] == "separately_counted"
    assert count_parameters(model) == 160
    assert count_embedding_parameters(model) == 40
    assert count_lm_head_parameters(model) == 80
    assert count_non_embedding_parameters(model) == 40


def test_trainable_only_counts_ignore_frozen_parameters():
    model = TinyLanguageModel()
    model.model.embed_tokens.weight.requires_grad = False
    model.lm_head.weight.requires_grad = False

    counts = model_parameter_counts(model, trainable_only=True)

    assert counts["total_parameters"] == 40
    assert counts["embedding_parameters"] == 0
    assert counts["lm_head_parameters"] == 0
    assert counts["non_embedding_parameters"] == 40
    assert counts["lm_head_counting"] == "separately_counted"


def test_disaggregated_parameter_counts_include_ffn_attention_and_other_buckets():
    model = TinyDisaggregatedLanguageModel()

    counts = model_parameter_counts(model)

    assert counts["total_parameters"] == 200
    assert counts["embedding_parameters"] == 40
    assert counts["lm_head_parameters"] == 40
    assert counts["non_embedding_parameters"] == 120
    assert counts["ffn_parameters"] == 96
    assert counts["attention_parameters"] == 16
    assert counts["other_non_embedding_parameters"] == 8
    assert counts["lm_head_counting"] == "separately_counted"
    assert counts["unavailable_component_reasons"] == {}


def test_unavailable_optional_component_counts_include_reasons():
    model = TinyLanguageModel()

    counts = model_parameter_counts(model)

    assert counts["ffn_parameters"] is None
    assert counts["attention_parameters"] is None
    assert counts["other_non_embedding_parameters"] is None
    reasons = counts["unavailable_component_reasons"]
    assert "ffn_parameters" in reasons
    assert "attention_parameters" in reasons
    assert "other_non_embedding_parameters" in reasons
    assert "ffn" in reasons["ffn_parameters"].lower()
    assert "attention" in reasons["attention_parameters"].lower()
    assert "requires" in reasons["other_non_embedding_parameters"].lower()


def test_matformer_active_prefix_counts_are_granularity_specific():
    model = ModifiedLlamaForCausalLM(tiny_llama_config())

    full_counts = model_parameter_counts(model)
    s_counts = model_parameter_counts(model, granularity="s")
    xl_counts = model_parameter_counts(model, granularity="xl")

    assert xl_counts == full_counts
    assert s_counts["embedding_parameters"] == full_counts["embedding_parameters"]
    assert s_counts["lm_head_parameters"] == full_counts["lm_head_parameters"]
    assert s_counts["non_embedding_parameters"] < xl_counts["non_embedding_parameters"]


@pytest.mark.parametrize(
    "run_id, granularity, expected_intermediate_size",
    [
        ("debug-standalone-s-001", "s", 64),
        ("debug-standalone-m-001", "m", 128),
        ("debug-standalone-l-001", "l", 256),
        ("debug-standalone-xl-001", "xl", 512),
    ],
)
def test_standalone_model_builds_fixed_width_llama_baselines(
    tmp_path,
    run_id,
    granularity,
    expected_intermediate_size,
):
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id=run_id,
        output_dir=tmp_path / run_id,
    )

    assert config["run"]["model_family"] == "standalone"
    assert config["run"]["granularity"] == granularity
    assert config["model"]["granularities"] == [granularity]

    model = build_model(config)

    assert isinstance(model, LlamaForCausalLM)
    assert not isinstance(model, ModifiedLlamaForCausalLM)
    assert model.config.intermediate_size == expected_intermediate_size
    assert not hasattr(model, "configure_subnetwork")


def test_cat_llama_build_passes_configured_granularities_to_membership_correction():
    config = {
        "run": {"model_family": "nested"},
        "model": {
            "variant": "cat_llama",
            "granularities": ["m", "xl"],
            "membership_correction": False,
            "vocab_size_assumption": 32,
            "hidden_size": 16,
            "intermediate_size": 64,
            "num_layers": 1,
            "num_attention_heads": 4,
            "context_length": 16,
        },
    }

    model = build_model(config)
    layer_mlp = model.model.layers[0].mlp

    assert isinstance(layer_mlp, CatLlamaMLP)
    assert layer_mlp.trained_granularities == ("m", "xl")
    assert layer_mlp.gradient_membership_correction_enabled is False
    assert layer_mlp.gradient_membership_counts == [2, 2, 1, 1]


def test_modified_llama_build_passes_membership_correction_configuration():
    config = {
        "run": {"model_family": "nested"},
        "model": {
            "variant": "matformer_llama",
            "granularities": ["m", "xl"],
            "membership_correction": True,
            "vocab_size_assumption": 32,
            "hidden_size": 16,
            "intermediate_size": 64,
            "num_layers": 1,
            "num_attention_heads": 4,
            "context_length": 16,
        },
    }

    model = build_model(config)
    layer_mlp = model.model.layers[0].mlp

    assert isinstance(layer_mlp, ModifiedLlamaMLP)
    assert layer_mlp.trained_granularities == ("m", "xl")
    assert layer_mlp.gradient_membership_correction_enabled is True
    assert layer_mlp.gradient_membership_counts == [2, 1]
    assert [
        (segment["start"], segment["end"])
        for segment in layer_mlp.gradient_membership_segment_metadata
    ] == [(0, 16), (16, 64)]
