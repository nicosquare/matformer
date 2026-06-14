"""FFN helpers and metadata for the slicing and concat variants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.llama.modeling_llama import LlamaMLP

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


class ModifiedLlamaMLP(LlamaMLP):
    def __init__(
        self,
        config,
        trained_granularities=None,
        gradient_membership_correction_enabled=True,
    ):
        super().__init__(config)
        self.intermediate_size = config.intermediate_size
        config_granularities = tuple(
            getattr(config, "granularities", MATFORMER_GRANULARITY_ORDER)
        )
        config_prefixes = getattr(config, "granularity_prefixes", None)
        self.ffn_prefix_metadata = (
            [dict(entry) for entry in getattr(config, "ffn_prefix_metadata", [])]
            if getattr(config, "ffn_prefix_metadata", None)
            else get_ffn_prefix_metadata(
                self.intermediate_size,
                granularity_prefixes=config_prefixes,
                granularities=config_granularities,
            )
        )
        self.granularity_prefixes = {
            entry["name"]: entry["full_intermediate_fraction"]
            for entry in self.ffn_prefix_metadata
        }
        self.trained_granularities = tuple(
            trained_granularities or config_granularities
        )
        self.gradient_membership_segment_metadata = (
            get_prefix_membership_segment_metadata(
                self.intermediate_size,
                self.trained_granularities,
                granularity_prefixes=self.granularity_prefixes,
            )
        )
        self.gradient_membership_correction_enabled = (
            gradient_membership_correction_enabled
        )
        self.gradient_membership_counts = [
            segment["membership_count"]
            for segment in self.gradient_membership_segment_metadata
        ]
        self.gradient_membership_correction_scales = [
            segment["scale"] for segment in self.gradient_membership_segment_metadata
        ]
        self.register_buffer(
            "gradient_membership_correction_scale_vector",
            self._build_gradient_membership_scale_vector(),
            persistent=False,
        )
        self.current_granularity = None
        self.current_subset_hd = None
        if self.gradient_membership_correction_enabled:
            self._register_gradient_membership_correction_hooks()

    def _build_gradient_membership_scale_vector(self):
        scale_vector = torch.ones(self.intermediate_size, dtype=torch.float32)
        for segment in self.gradient_membership_segment_metadata:
            scale_vector[segment["start"] : segment["end"]] = segment["scale"]
        return scale_vector

    def _register_gradient_membership_correction_hooks(self):
        self.gate_proj.weight.register_hook(self._scale_gate_or_up_weight_grad)
        self.up_proj.weight.register_hook(self._scale_gate_or_up_weight_grad)
        self.down_proj.weight.register_hook(self._scale_down_weight_grad)
        if self.gate_proj.bias is not None:
            self.gate_proj.bias.register_hook(self._scale_bias_grad)
        if self.up_proj.bias is not None:
            self.up_proj.bias.register_hook(self._scale_bias_grad)

    def _scale_bias_grad(self, grad):
        scale = self.gradient_membership_correction_scale_vector.to(dtype=grad.dtype)
        return grad * scale

    def _scale_gate_or_up_weight_grad(self, grad):
        scale = self.gradient_membership_correction_scale_vector.to(dtype=grad.dtype)
        return grad * scale.unsqueeze(1)

    def _scale_down_weight_grad(self, grad):
        scale = self.gradient_membership_correction_scale_vector.to(dtype=grad.dtype)
        return grad * scale.unsqueeze(0)

    def configure_subnetwork(self, flag):
        """Configure subnetwork size based on flag."""
        self.current_granularity = flag
        self.current_subset_hd = granularity_prefix_width(
            self.intermediate_size,
            flag,
            granularity_prefixes=self.granularity_prefixes,
        )

    def prefix_parameter_count(self, flag, trainable_only=False):
        prefix_width = granularity_prefix_width(
            self.intermediate_size,
            flag,
            granularity_prefixes=self.granularity_prefixes,
        )
        parameter_count = 0

        if not trainable_only or self.gate_proj.weight.requires_grad:
            parameter_count += self.gate_proj.weight.shape[1] * prefix_width
        if self.gate_proj.bias is not None:
            if not trainable_only or self.gate_proj.bias.requires_grad:
                parameter_count += prefix_width

        if not trainable_only or self.up_proj.weight.requires_grad:
            parameter_count += self.up_proj.weight.shape[1] * prefix_width
        if self.up_proj.bias is not None:
            if not trainable_only or self.up_proj.bias.requires_grad:
                parameter_count += prefix_width

        if not trainable_only or self.down_proj.weight.requires_grad:
            parameter_count += self.down_proj.weight.shape[0] * prefix_width
        if self.down_proj.bias is not None:
            if not trainable_only or self.down_proj.bias.requires_grad:
                parameter_count += self.down_proj.bias.numel()

        return parameter_count
    
    def forward(self, x):
        if self.current_subset_hd is None:
            raise ValueError("Subnetwork size not configured. Call `configure_subnetwork` first.")
        gate_proj = self.gate_proj.weight[:self.current_subset_hd]
        up_proj = self.up_proj.weight[:self.current_subset_hd]
        down_proj = self.down_proj.weight[:, :self.current_subset_hd]
        gate_bias = None
        up_bias = None
        down_bias = self.down_proj.bias

        if self.gate_proj.bias is not None:
            gate_bias = self.gate_proj.bias[:self.current_subset_hd]
        if self.up_proj.bias is not None:
            up_bias = self.up_proj.bias[:self.current_subset_hd]

        down_proj = F.linear(
            self.act_fn(F.linear(x, gate_proj, gate_bias))
            * F.linear(x, up_proj, up_bias),
            down_proj,
            down_bias,
        )

        return down_proj


class CatLlamaMLP(LlamaMLP):
    def __init__(
        self,
        config,
        trained_granularities=None,
        gradient_membership_correction_enabled=True,
    ):
        super().__init__(config)
        self.intermediate_size = config.intermediate_size
        config_granularities = tuple(
            getattr(config, "granularities", MATFORMER_GRANULARITY_ORDER)
        )
        config_prefixes = getattr(config, "granularity_prefixes", None)
        self.ffn_prefix_metadata = (
            [dict(entry) for entry in getattr(config, "ffn_prefix_metadata", [])]
            if getattr(config, "ffn_prefix_metadata", None)
            else get_ffn_prefix_metadata(
                self.intermediate_size,
                granularity_prefixes=config_prefixes,
                granularities=config_granularities,
            )
        )
        self.granularity_prefixes = {
            entry["name"]: entry["full_intermediate_fraction"]
            for entry in self.ffn_prefix_metadata
        }
        block_granularities = config_granularities
        block_prefixes = self.granularity_prefixes
        if config_prefixes is None and len(config_granularities) < len(
            MATFORMER_GRANULARITY_ORDER
        ):
            block_granularities = MATFORMER_GRANULARITY_ORDER
            block_prefixes = None

        self.ffn_concat_block_metadata = (
            [dict(entry) for entry in getattr(config, "ffn_concat_block_metadata", [])]
            if getattr(config, "ffn_concat_block_metadata", None)
            else get_concat_block_metadata(
                self.intermediate_size,
                granularity_prefixes=block_prefixes,
                granularities=block_granularities,
            )
        )
        self.trained_granularities = tuple(
            trained_granularities or config_granularities
        )
        max_trained_block_index = max(
            MATFORMER_GRANULARITY_ORDER.index(granularity)
            for granularity in self.trained_granularities
        )
        if max_trained_block_index >= len(self.ffn_concat_block_metadata):
            self.ffn_concat_block_metadata = get_concat_block_metadata(
                self.intermediate_size,
                granularities=MATFORMER_GRANULARITY_ORDER,
            )
        self.gradient_membership_correction_enabled = (
            gradient_membership_correction_enabled
        )
        self.current_granularity = None
        self.current_subset_hd = None
        self.current_subset_blocks = None

        self.gate_weight_blocks = nn.ParameterList()
        self.up_weight_blocks = nn.ParameterList()
        self.down_weight_blocks = nn.ParameterList()
        self.gate_bias_blocks = nn.ParameterList()
        self.up_bias_blocks = nn.ParameterList()

        # Copy each dense projection into block parameters independently so we
        # can drop the source tensor before materializing the next one.
        offset = 0
        gate_weight = self.gate_proj.weight
        gate_bias = self.gate_proj.bias
        for block_metadata in self.ffn_concat_block_metadata:
            block_width = block_metadata["block_width"]
            next_offset = offset + block_width

            self.gate_weight_blocks.append(
                nn.Parameter(gate_weight[offset:next_offset].detach().clone())
            )
            if gate_bias is not None:
                self.gate_bias_blocks.append(
                    nn.Parameter(gate_bias[offset:next_offset].detach().clone())
            )

            offset = next_offset
        # Release the dense gate projection before moving on to the next tensor.
        del self.gate_proj
        del gate_weight
        del gate_bias

        # Repeat the same pattern for the up projection to keep peak overlap low.
        offset = 0
        up_weight = self.up_proj.weight
        up_bias = self.up_proj.bias
        for block_metadata in self.ffn_concat_block_metadata:
            block_width = block_metadata["block_width"]
            next_offset = offset + block_width

            self.up_weight_blocks.append(
                nn.Parameter(up_weight[offset:next_offset].detach().clone())
            )
            if up_bias is not None:
                self.up_bias_blocks.append(
                    nn.Parameter(up_bias[offset:next_offset].detach().clone())
            )

            offset = next_offset
        # Release the dense up projection before building the down blocks.
        del self.up_proj
        del up_weight
        del up_bias

        # The down projection is split across input columns, so copy its blocks
        # last and then discard the dense tensor.
        offset = 0
        down_weight = self.down_proj.weight
        down_bias = self.down_proj.bias
        for block_metadata in self.ffn_concat_block_metadata:
            block_width = block_metadata["block_width"]
            next_offset = offset + block_width

            self.down_weight_blocks.append(
                nn.Parameter(down_weight[:, offset:next_offset].detach().clone())
            )

            offset = next_offset

        self.down_bias = (
            None if down_bias is None else nn.Parameter(down_bias.detach().clone())
        )
        # Drop the dense projections once their block parameters are in place.
        del self.down_proj
        del down_weight
        del down_bias
        self.gradient_membership_counts = get_concat_block_membership_counts_from_metadata(
            self.ffn_concat_block_metadata,
            self.trained_granularities,
        )
        self.gradient_membership_correction_scales = (
            get_concat_gradient_membership_correction_scales_from_metadata(
                self.ffn_concat_block_metadata,
                self.trained_granularities,
            )
        )
        if self.gradient_membership_correction_enabled:
            self._register_gradient_membership_correction_hooks()

    def _register_gradient_membership_correction_hooks(self):
        block_groups = [
            self.gate_weight_blocks,
            self.up_weight_blocks,
            self.down_weight_blocks,
            self.gate_bias_blocks,
            self.up_bias_blocks,
        ]

        for block_index, scale in enumerate(self.gradient_membership_correction_scales):
            if scale == 1.0:
                continue
            for blocks in block_groups:
                if block_index >= len(blocks):
                    continue
                param = blocks[block_index]
                if not param.requires_grad:
                    continue
                param.register_hook(lambda grad, scale=scale: grad * scale)

    def configure_subnetwork(self, flag):
        self.current_granularity = flag
        self.current_subset_blocks = granularity_concat_block_count(flag)
        self.current_subset_hd = granularity_prefix_width(
            self.intermediate_size,
            flag,
            granularity_prefixes=self.granularity_prefixes,
        )

    def prefix_parameter_count(self, flag, trainable_only=False):
        block_count = granularity_concat_block_count(flag)
        parameter_count = 0

        for block_index in range(block_count):
            if not trainable_only or self.gate_weight_blocks[block_index].requires_grad:
                parameter_count += self.gate_weight_blocks[block_index].numel()
            if self.gate_bias_blocks and (
                not trainable_only or self.gate_bias_blocks[block_index].requires_grad
            ):
                parameter_count += self.gate_bias_blocks[block_index].numel()

            if not trainable_only or self.up_weight_blocks[block_index].requires_grad:
                parameter_count += self.up_weight_blocks[block_index].numel()
            if self.up_bias_blocks and (
                not trainable_only or self.up_bias_blocks[block_index].requires_grad
            ):
                parameter_count += self.up_bias_blocks[block_index].numel()

            if not trainable_only or self.down_weight_blocks[block_index].requires_grad:
                parameter_count += self.down_weight_blocks[block_index].numel()

        if self.down_bias is not None and (
            not trainable_only or self.down_bias.requires_grad
        ):
            parameter_count += self.down_bias.numel()

        return parameter_count
    
    def _assemble_prefix(self, blocks, dim):
        if self.current_subset_blocks is None:
            raise ValueError("Subnetwork size not configured. Call `configure_subnetwork` first.")
        selected_blocks = list(blocks[: self.current_subset_blocks])
        if not selected_blocks:
            raise ValueError("Configured subnetwork produced no blocks")
        return torch.cat(selected_blocks, dim=dim)

    def forward(self, x):
        if self.current_subset_blocks is None:
            raise ValueError("Subnetwork size not configured. Call `configure_subnetwork` first.")

        gate_proj = self._assemble_prefix(self.gate_weight_blocks, dim=0)
        up_proj = self._assemble_prefix(self.up_weight_blocks, dim=0)
        down_proj = self._assemble_prefix(self.down_weight_blocks, dim=1)

        gate_bias = None
        up_bias = None
        if self.gate_bias_blocks:
            gate_bias = self._assemble_prefix(self.gate_bias_blocks, dim=0)
        if self.up_bias_blocks:
            up_bias = self._assemble_prefix(self.up_bias_blocks, dim=0)

        return F.linear(
            self.act_fn(F.linear(x, gate_proj, gate_bias))
            * F.linear(x, up_proj, up_bias),
            down_proj,
            self.down_bias,
        )


__all__ = [
    "MATFORMER_GRANULARITY_ORDER",
    "ModifiedLlamaMLP",
    "CatLlamaMLP",
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
