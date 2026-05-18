"""Parameter counting helpers for scaling reports."""

from __future__ import annotations

from typing import Any


EMBEDDING_NAME_MARKERS = ("embed_tokens", "word_embeddings", "wte")
LM_HEAD_NAME_MARKERS = ("lm_head", "output_head")
FFN_NAME_MARKERS = (
    ".mlp.",
    ".feed_forward.",
    ".ffn.",
    ".ff.",
    ".gate_proj.",
    ".up_proj.",
    ".down_proj.",
)
ATTENTION_NAME_MARKERS = (
    ".self_attn.",
    ".attention.",
    ".attn.",
    ".q_proj.",
    ".k_proj.",
    ".v_proj.",
    ".o_proj.",
)


def model_parameter_counts(
    model: Any,
    trainable_only: bool = False,
    granularity: str | None = None,
) -> dict[str, Any]:
    matformer_parameter_ids = set()
    active_matformer_parameters = 0

    if granularity is not None:
        for module in model.modules():
            if hasattr(module, "prefix_parameter_count"):
                active_matformer_parameters += module.prefix_parameter_count(
                    granularity,
                    trainable_only=trainable_only,
                )
                for parameter in module.parameters(recurse=True):
                    matformer_parameter_ids.add(id(parameter))

    named_parameters = list(model.named_parameters())
    total_parameters = active_matformer_parameters
    embedding_parameters = 0
    lm_head_parameters = 0
    ffn_parameters = active_matformer_parameters
    attention_parameters = 0
    other_non_embedding_parameters = 0

    for name, parameter in named_parameters:
        if id(parameter) in matformer_parameter_ids:
            continue
        if trainable_only and not parameter.requires_grad:
            continue

        parameter_count = parameter.numel()
        total_parameters += parameter_count

        if _is_lm_head_parameter(name):
            lm_head_parameters += parameter_count
        elif _is_embedding_parameter(name):
            embedding_parameters += parameter_count
        elif _is_ffn_parameter(name):
            ffn_parameters += parameter_count
        elif _is_attention_parameter(name):
            attention_parameters += parameter_count
        else:
            other_non_embedding_parameters += parameter_count

    non_embedding_parameters = (
        total_parameters - embedding_parameters - lm_head_parameters
    )
    unavailable_component_reasons: dict[str, str] = {}
    ffn_parameters_value: int | None = ffn_parameters
    attention_parameters_value: int | None = attention_parameters
    other_non_embedding_parameters_value: int | None = other_non_embedding_parameters

    if ffn_parameters == 0:
        ffn_parameters_value = None
        unavailable_component_reasons["ffn_parameters"] = (
            "No FFN parameters could be identified by known module/name markers."
        )

    if attention_parameters == 0:
        attention_parameters_value = None
        unavailable_component_reasons["attention_parameters"] = (
            "No attention parameters could be identified by known module/name markers."
        )

    if ffn_parameters_value is None or attention_parameters_value is None:
        other_non_embedding_parameters_value = None
        unavailable_component_reasons["other_non_embedding_parameters"] = (
            "Counting other non-embedding parameters requires both FFN and attention "
            "parameter buckets to be available."
        )

    return {
        "total_parameters": total_parameters,
        "embedding_parameters": embedding_parameters,
        "lm_head_parameters": lm_head_parameters,
        "non_embedding_parameters": non_embedding_parameters,
        "ffn_parameters": ffn_parameters_value,
        "attention_parameters": attention_parameters_value,
        "other_non_embedding_parameters": other_non_embedding_parameters_value,
        "lm_head_counting": _lm_head_counting_convention(model, named_parameters),
        "unavailable_component_reasons": unavailable_component_reasons,
    }


def count_parameters(
    model: Any,
    trainable_only: bool = False,
    granularity: str | None = None,
) -> int:
    return model_parameter_counts(
        model,
        trainable_only=trainable_only,
        granularity=granularity,
    )["total_parameters"]


def count_embedding_parameters(model: Any, trainable_only: bool = False) -> int:
    return model_parameter_counts(model, trainable_only=trainable_only)[
        "embedding_parameters"
    ]


def count_lm_head_parameters(model: Any, trainable_only: bool = False) -> int:
    return model_parameter_counts(model, trainable_only=trainable_only)[
        "lm_head_parameters"
    ]


def count_non_embedding_parameters(
    model: Any,
    trainable_only: bool = False,
    granularity: str | None = None,
) -> int:
    return model_parameter_counts(
        model,
        trainable_only=trainable_only,
        granularity=granularity,
    )["non_embedding_parameters"]


def _is_embedding_parameter(name: str) -> bool:
    return any(marker in name for marker in EMBEDDING_NAME_MARKERS)


def _is_lm_head_parameter(name: str) -> bool:
    return any(marker in name for marker in LM_HEAD_NAME_MARKERS)


def _is_ffn_parameter(name: str) -> bool:
    dotted_name = f".{name}."
    return any(marker in dotted_name for marker in FFN_NAME_MARKERS)


def _is_attention_parameter(name: str) -> bool:
    dotted_name = f".{name}."
    return any(marker in dotted_name for marker in ATTENTION_NAME_MARKERS)


def _lm_head_counting_convention(
    model: Any,
    named_parameters: list[tuple[str, Any]],
) -> str:
    if any(_is_lm_head_parameter(name) for name, _ in named_parameters):
        return "separately_counted"

    config = getattr(model, "config", None)
    if getattr(config, "tie_word_embeddings", False):
        return "tied_with_embeddings"

    return "unavailable"
