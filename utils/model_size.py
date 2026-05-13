"""Parameter counting helpers for scaling reports."""

from __future__ import annotations

from typing import Any


EMBEDDING_NAME_MARKERS = ("embed_tokens", "word_embeddings", "wte")
LM_HEAD_NAME_MARKERS = ("lm_head", "output_head")


def model_parameter_counts(
    model: Any,
    trainable_only: bool = False,
    granularity: str | None = None,
) -> dict[str, int]:
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

    total_parameters = active_matformer_parameters
    embedding_parameters = 0
    lm_head_parameters = 0

    for name, parameter in model.named_parameters():
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

    non_embedding_parameters = (
        total_parameters - embedding_parameters - lm_head_parameters
    )

    return {
        "total_parameters": total_parameters,
        "embedding_parameters": embedding_parameters,
        "lm_head_parameters": lm_head_parameters,
        "non_embedding_parameters": non_embedding_parameters,
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
