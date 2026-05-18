"""Consistency metrics for nested and standalone comparison runs."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import torch


TOKEN_LEVEL_AGREEMENT = "token_level_agreement"
TOP_K_OVERLAP = "top_k_overlap"
KL_DIVERGENCE = "kl_divergence"
DEFAULT_TOP_K_VALUES = (5,)
KL_DIVERGENCE_DEFERRED_REASON = (
    "KL divergence is deferred for a later phase because it requires more "
    "careful probability-space handling and numerical stabilization."
)


def token_level_agreement(
    small_logits: torch.Tensor,
    large_logits: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    small_tokens, large_tokens = _masked_argmax_predictions(
        small_logits,
        large_logits,
        attention_mask=attention_mask,
    )
    sample_count = int(small_tokens.numel())
    if sample_count == 0:
        agreement = 0.0
    else:
        agreement = float((small_tokens == large_tokens).float().mean().item())
    return {
        "metric_name": TOKEN_LEVEL_AGREEMENT,
        "metric_value": agreement,
        "sample_count": sample_count,
    }


def top_k_overlap(
    small_logits: torch.Tensor,
    large_logits: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    k: int = 5,
) -> dict[str, Any]:
    _validate_logits_shape(small_logits, large_logits)
    if k <= 0:
        raise ValueError("top-k overlap requires k > 0")

    active_indices = _active_token_indices(small_logits, attention_mask=attention_mask)
    sample_count = int(active_indices.numel())
    if sample_count == 0:
        return {
            "metric_name": TOP_K_OVERLAP,
            "metric_value": 0.0,
            "sample_count": 0,
            "top_k": k,
        }

    vocab_size = int(small_logits.shape[-1])
    effective_k = min(k, vocab_size)
    small_topk = torch.topk(
        small_logits.reshape(-1, vocab_size).index_select(0, active_indices),
        k=effective_k,
        dim=-1,
    ).indices
    large_topk = torch.topk(
        large_logits.reshape(-1, vocab_size).index_select(0, active_indices),
        k=effective_k,
        dim=-1,
    ).indices

    overlaps = []
    for small_ids, large_ids in zip(small_topk, large_topk):
        overlap_count = len(set(small_ids.tolist()) & set(large_ids.tolist()))
        overlaps.append(overlap_count / effective_k)

    return {
        "metric_name": TOP_K_OVERLAP,
        "metric_value": float(sum(overlaps) / len(overlaps)),
        "sample_count": sample_count,
        "top_k": effective_k,
    }


def deferred_kl_divergence_note(
    sample_count: int,
) -> dict[str, Any]:
    return {
        "metric_name": KL_DIVERGENCE,
        "metric_value": None,
        "sample_count": int(sample_count),
        "deferred": True,
        "deferred_reason": KL_DIVERGENCE_DEFERRED_REASON,
    }


def build_consistency_rows(
    comparison_id: str,
    small_run_id: str,
    large_run_id: str,
    small_granularity: str,
    large_granularity: str,
    small_logits: torch.Tensor,
    large_logits: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    top_k_values: Sequence[int] = DEFAULT_TOP_K_VALUES,
    include_deferred_kl_note: bool = True,
) -> list[dict[str, Any]]:
    base_fields = {
        "comparison_id": comparison_id,
        "small_run_id": small_run_id,
        "large_run_id": large_run_id,
        "small_granularity": small_granularity,
        "large_granularity": large_granularity,
    }

    rows = [base_fields | token_level_agreement(small_logits, large_logits, attention_mask)]
    sample_count = rows[0]["sample_count"]

    for k in top_k_values:
        rows.append(base_fields | top_k_overlap(small_logits, large_logits, attention_mask, k=k))

    if include_deferred_kl_note:
        rows.append(base_fields | deferred_kl_divergence_note(sample_count))

    return rows


def summarize_consistency_suite(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(row["metric_name"]): dict(row)
        for row in rows
    }


def _masked_argmax_predictions(
    small_logits: torch.Tensor,
    large_logits: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_logits_shape(small_logits, large_logits)
    active_indices = _active_token_indices(small_logits, attention_mask=attention_mask)
    vocab_size = int(small_logits.shape[-1])
    flat_small = small_logits.reshape(-1, vocab_size).index_select(0, active_indices)
    flat_large = large_logits.reshape(-1, vocab_size).index_select(0, active_indices)
    return flat_small.argmax(dim=-1), flat_large.argmax(dim=-1)


def _active_token_indices(
    logits: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("consistency logits must have shape [batch, seq, vocab]")

    batch_size, sequence_length, _ = logits.shape
    total_positions = batch_size * sequence_length
    if attention_mask is None:
        return torch.arange(total_positions, device=logits.device)

    if attention_mask.shape != logits.shape[:2]:
        raise ValueError(
            "attention_mask must match logits batch and sequence dimensions"
        )
    flat_mask = attention_mask.reshape(-1).to(dtype=torch.bool, device=logits.device)
    return flat_mask.nonzero(as_tuple=False).flatten()


def _validate_logits_shape(
    small_logits: torch.Tensor,
    large_logits: torch.Tensor,
) -> None:
    if small_logits.shape != large_logits.shape:
        raise ValueError(
            f"consistency logits must match exactly: {small_logits.shape} vs {large_logits.shape}"
        )
    if small_logits.ndim != 3:
        raise ValueError("consistency logits must have shape [batch, seq, vocab]")
