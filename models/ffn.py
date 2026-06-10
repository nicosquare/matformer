"""FFN helpers and metadata for MatFormer and CatLlama variants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from models.granularity import (
    MATFORMER_GRANULARITY_ORDER,
    get_concat_block_metadata,
    get_concat_block_membership_counts,
    get_concat_block_membership_counts_from_metadata,
    get_concat_gradient_membership_correction_scales,
    get_concat_gradient_membership_correction_scales_from_metadata,
    get_concat_layout_diagnostic,
    get_ffn_prefix_metadata,
    get_gradient_membership_correction_scales,
    get_prefix_membership_segment_metadata,
    granularity_concat_block_count,
    granularity_prefix_width,
)


def build_ffn_prefix_metadata(
    intermediate_size: int,
    granularity_prefixes: Mapping[str, object] | None = None,
    granularities: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    return get_ffn_prefix_metadata(
        intermediate_size,
        granularity_prefixes=granularity_prefixes,
        granularities=granularities,
    )


def build_concat_layout_diagnostic(
    intermediate_size: int,
    granularities: Sequence[str],
    granularity_prefixes: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    return get_concat_layout_diagnostic(
        intermediate_size,
        granularities,
        granularity_prefixes=granularity_prefixes,
    )


def build_prefix_membership_segment_metadata(
    intermediate_size: int,
    granularities: Sequence[str],
    granularity_prefixes: Mapping[str, object] | None = None,
) -> list[dict[str, Any]]:
    return get_prefix_membership_segment_metadata(
        intermediate_size,
        granularities,
        granularity_prefixes=granularity_prefixes,
    )


__all__ = [
    "MATFORMER_GRANULARITY_ORDER",
    "build_concat_layout_diagnostic",
    "build_ffn_prefix_metadata",
    "build_prefix_membership_segment_metadata",
    "get_concat_block_metadata",
    "get_concat_block_membership_counts",
    "get_concat_block_membership_counts_from_metadata",
    "get_concat_gradient_membership_correction_scales",
    "get_concat_gradient_membership_correction_scales_from_metadata",
    "get_ffn_prefix_metadata",
    "get_gradient_membership_correction_scales",
    "get_prefix_membership_segment_metadata",
    "granularity_concat_block_count",
    "granularity_prefix_width",
]
