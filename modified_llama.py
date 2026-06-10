"""Compatibility façade for legacy MatFormer imports.

The concrete granularity helpers now live under ``models/``. This module keeps
the historical import surface stable for training, diagnostics, and tests while
the repository migrates call sites over to the shallow package layout.
"""

from __future__ import annotations

from transformers import LlamaForCausalLM

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
from models.wiring import apply_granularity_pattern_to_model

get_concat_layout_diagnostic = build_concat_layout_diagnostic


class ModifiedLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, mlp_cls=ModifiedLlamaMLP, mlp_kwargs=None):
        super().__init__(config)
        self.granularity_order = MATFORMER_GRANULARITY_ORDER
        self.ffn_prefix_metadata = (
            [dict(entry) for entry in getattr(config, "ffn_prefix_metadata", [])]
            if getattr(config, "ffn_prefix_metadata", None)
            else get_ffn_prefix_metadata(
                config.intermediate_size,
                granularity_prefixes=getattr(config, "granularity_prefixes", None),
                granularities=getattr(config, "granularities", None),
            )
        )
        self.mlp_cls = mlp_cls
        self.mlp_kwargs = dict(mlp_kwargs or {})
        self.matformer_layers = []
        self.current_layer_granularities = None
        self.current_granularity_pattern = None
        self.current_sampling_mode = "global"

        for layer_idx in range(config.num_hidden_layers):
            mlp = self.mlp_cls(config, **self.mlp_kwargs)
            self.model.layers[layer_idx].mlp = mlp
            self.matformer_layers.append(mlp)

    def configure_subnetwork(self, flag):
        """Configure the subnetwork for all layers based on the flag."""
        apply_granularity_pattern_to_model(
            self,
            flag,
            sampling_mode="global",
        )

    def configure_layer_granularities(self, layer_granularities):
        """Configure a repeating or explicit granularity pattern across layers."""
        apply_granularity_pattern_to_model(
            self,
            layer_granularities,
            sampling_mode="per_layer",
        )


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
]
