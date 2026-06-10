"""Compatibility façade for legacy MatFormer imports.

The concrete granularity helpers now live under ``models/``. This module keeps
the historical import surface stable for training, diagnostics, and tests while
the repository migrates call sites over to the shallow package layout.
"""

from __future__ import annotations

from models.correction import (
    CorrectionContext,
    VALID_CORRECTION_MODES,
    VALID_SAMPLING_MODES,
    build_correction_context,
    build_correction_context_from_pattern,
    build_local_correction_context_from_pattern,
    correction_context_from_config,
    derive_local_membership_pattern,
    should_activate_local_correction,
    summarize_correction_context,
)
from models.ffn import (
    CatLlamaMLP,
    MATFORMER_GRANULARITY_ORDER,
    ModifiedLlamaMLP,
    build_concat_layout_diagnostic,
    build_ffn_prefix_metadata,
    build_prefix_membership_segment_metadata,
    get_concat_block_membership_counts,
    get_concat_block_membership_counts_from_metadata,
    get_concat_block_metadata,
    get_concat_gradient_membership_correction_scales,
    get_concat_gradient_membership_correction_scales_from_metadata,
    get_ffn_prefix_metadata,
    get_prefix_membership_segment_metadata,
    granularity_concat_block_count,
    granularity_prefix_width,
)
from models.granularity import (
    MATFORMER_GRANULARITY_DISPLAY_NAMES,
    GranularityMetadata,
    GranularityPattern,
    build_granularity_pattern,
    canonical_prefix_fraction,
    expand_layer_granularity_pattern,
    get_block_membership_counts,
    get_gradient_membership_correction_scales,
    get_granularity_metadata,
    granularity_block_count,
    summarize_granularity_pattern,
)
from models.wiring import ModifiedLlamaForCausalLM, apply_granularity_pattern_to_model

get_concat_layout_diagnostic = build_concat_layout_diagnostic

__all__ = [
    "CorrectionContext",
    "VALID_CORRECTION_MODES",
    "VALID_SAMPLING_MODES",
    "CatLlamaMLP",
    "ModifiedLlamaMLP",
    "ModifiedLlamaForCausalLM",
    "MATFORMER_GRANULARITY_ORDER",
    "MATFORMER_GRANULARITY_DISPLAY_NAMES",
    "GranularityMetadata",
    "GranularityPattern",
    "build_correction_context",
    "build_correction_context_from_pattern",
    "build_local_correction_context_from_pattern",
    "correction_context_from_config",
    "derive_local_membership_pattern",
    "should_activate_local_correction",
    "summarize_correction_context",
    "build_concat_layout_diagnostic",
    "get_concat_layout_diagnostic",
    "build_ffn_prefix_metadata",
    "build_prefix_membership_segment_metadata",
    "get_concat_block_membership_counts",
    "get_concat_block_membership_counts_from_metadata",
    "get_concat_block_metadata",
    "get_concat_gradient_membership_correction_scales",
    "get_concat_gradient_membership_correction_scales_from_metadata",
    "get_ffn_prefix_metadata",
    "get_prefix_membership_segment_metadata",
    "granularity_concat_block_count",
    "granularity_prefix_width",
    "build_granularity_pattern",
    "canonical_prefix_fraction",
    "expand_layer_granularity_pattern",
    "get_block_membership_counts",
    "get_gradient_membership_correction_scales",
    "get_granularity_metadata",
    "granularity_block_count",
    "summarize_granularity_pattern",
    "apply_granularity_pattern_to_model",
]
