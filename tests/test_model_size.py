import torch.nn as nn
from transformers import LlamaConfig

from modified_llama import ModifiedLlamaForCausalLM
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
        self.ff = nn.Linear(4, 8)
        self.lm_head = nn.Linear(8, 10, bias=False)


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

    assert counts == {
        "total_parameters": 160,
        "embedding_parameters": 40,
        "lm_head_parameters": 80,
        "non_embedding_parameters": 40,
    }
    assert count_parameters(model) == 160
    assert count_embedding_parameters(model) == 40
    assert count_lm_head_parameters(model) == 80
    assert count_non_embedding_parameters(model) == 40


def test_trainable_only_counts_ignore_frozen_parameters():
    model = TinyLanguageModel()
    model.model.embed_tokens.weight.requires_grad = False
    model.lm_head.weight.requires_grad = False

    counts = model_parameter_counts(model, trainable_only=True)

    assert counts == {
        "total_parameters": 40,
        "embedding_parameters": 0,
        "lm_head_parameters": 0,
        "non_embedding_parameters": 40,
    }


def test_matformer_active_prefix_counts_are_granularity_specific():
    model = ModifiedLlamaForCausalLM(tiny_llama_config())

    full_counts = model_parameter_counts(model)
    s_counts = model_parameter_counts(model, granularity="s")
    xl_counts = model_parameter_counts(model, granularity="xl")

    assert xl_counts == full_counts
    assert s_counts["embedding_parameters"] == full_counts["embedding_parameters"]
    assert s_counts["lm_head_parameters"] == full_counts["lm_head_parameters"]
    assert s_counts["non_embedding_parameters"] < xl_counts["non_embedding_parameters"]
