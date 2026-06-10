from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP

from models.ffn import (
    build_concat_layout_diagnostic as _build_concat_layout_diagnostic,
    build_ffn_prefix_metadata as _build_ffn_prefix_metadata,
    build_prefix_membership_segment_metadata as _build_prefix_membership_segment_metadata,
)
from models.wiring import (
    build_global_granularity_pattern,
    build_per_layer_granularity_pattern,
)
from utils.config import CANONICAL_GRANULARITY_PREFIX_FRACTIONS


MATFORMER_GRANULARITY_ORDER = ("s", "m", "l", "xl")
MATFORMER_GRANULARITY_DISPLAY_NAMES = {
    "s": "S",
    "m": "M",
    "l": "L",
    "xl": "XL",
}


def get_granularity_metadata(granularity):
    if granularity not in MATFORMER_GRANULARITY_DISPLAY_NAMES:
        raise ValueError(f"Unknown MatFormer granularity: {granularity}")
    fraction = _canonical_prefix_fraction(granularity)
    return {
        "display_name": MATFORMER_GRANULARITY_DISPLAY_NAMES[granularity],
        "ffn_ratio": fraction / _canonical_prefix_fraction("m"),
        "full_intermediate_fraction": fraction,
    }


def granularity_prefix_width(
    intermediate_size,
    granularity,
    granularity_prefixes: Mapping[str, object] | None = None,
):
    get_granularity_metadata(granularity)
    fraction = _granularity_prefix_fraction(
        granularity,
        granularity_prefixes=granularity_prefixes,
    )
    prefix_width = int(intermediate_size * fraction)
    if prefix_width <= 0:
        raise ValueError(
            f"Granularity {granularity} produced empty FFN prefix for "
            f"intermediate_size={intermediate_size}"
        )
    return prefix_width


def get_ffn_prefix_metadata(
    intermediate_size,
    granularity_prefixes: Mapping[str, object] | None = None,
    granularities: list[str] | tuple[str, ...] | None = None,
):
    return _build_ffn_prefix_metadata(
        intermediate_size,
        granularity_prefixes=granularity_prefixes,
        granularities=granularities,
    )


def _canonical_prefix_fraction(granularity):
    numerator, denominator = CANONICAL_GRANULARITY_PREFIX_FRACTIONS[granularity]
    return numerator / denominator


def _granularity_prefix_fraction(
    granularity,
    granularity_prefixes: Mapping[str, object] | None = None,
):
    if granularity_prefixes is None:
        return _canonical_prefix_fraction(granularity)
    if granularity not in granularity_prefixes:
        raise ValueError(f"Missing MatFormer granularity prefix: {granularity}")
    return float(granularity_prefixes[granularity])


def _resolve_granularity_prefixes(
    granularities: tuple[str, ...] | list[str],
    granularity_prefixes: Mapping[str, object] | None = None,
) -> dict[str, float]:
    prefixes = granularity_prefixes or {
        granularity: _canonical_prefix_fraction(granularity)
        for granularity in granularities
    }
    resolved = {}
    previous_fraction = 0.0
    for granularity in granularities:
        fraction = _granularity_prefix_fraction(
            granularity,
            granularity_prefixes=prefixes,
        )
        if fraction <= 0:
            raise ValueError(
                f"Granularity {granularity} must resolve to a positive prefix fraction"
            )
        if fraction <= previous_fraction:
            raise ValueError(
                "MatFormer granularity prefixes must be strictly increasing "
                "in the configured order"
            )
        resolved[granularity] = fraction
        previous_fraction = fraction
    return resolved


def expand_layer_granularity_pattern(layer_granularities, num_layers):
    if not isinstance(layer_granularities, (list, tuple)) or not layer_granularities:
        raise ValueError("layer_granularities must be a non-empty list or tuple")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    expanded = []
    for layer_index in range(num_layers):
        granularity = layer_granularities[layer_index % len(layer_granularities)]
        get_granularity_metadata(granularity)
        expanded.append(granularity)
    return expanded


def granularity_block_count(granularity):
    metadata = get_granularity_metadata(granularity)
    smallest_granularity = MATFORMER_GRANULARITY_ORDER[0]
    smallest_ratio = get_granularity_metadata(smallest_granularity)["ffn_ratio"]
    ratio = metadata["ffn_ratio"]
    block_count = ratio / smallest_ratio
    if int(block_count) != block_count:
        raise ValueError(
            f"Granularity {granularity} has incompatible ffn_ratio={ratio} "
            f"for smallest_ratio={smallest_ratio}"
        )
    return int(block_count)


def get_concat_block_metadata(
    intermediate_size,
    granularity_prefixes: Mapping[str, object] | None = None,
    granularities: list[str] | tuple[str, ...] | None = None,
):
    granularities = tuple(granularities or MATFORMER_GRANULARITY_ORDER)
    prefix_metadata = get_ffn_prefix_metadata(
        intermediate_size,
        granularity_prefixes=granularity_prefixes,
        granularities=granularities,
    )
    base_block_width = prefix_metadata[0]["prefix_width"]
    block_metadata = []
    previous_prefix_width = 0

    for block_index, prefix_entry in enumerate(prefix_metadata):
        prefix_width = prefix_entry["prefix_width"]
        if prefix_width <= previous_prefix_width:
            raise ValueError(
                "Granularity order must expand to strictly larger prefix blocks"
            )
        block_width = prefix_width - previous_prefix_width
        block_metadata.append(
            {
                "name": f"block_{block_index + 1}",
                "display_name": f"B{block_index + 1}",
                "ffn_ratio": block_width / base_block_width,
                "full_intermediate_fraction": prefix_width / intermediate_size,
                "prefix_width": prefix_width,
                "block_width": block_width,
                "cumulative_prefix_width": prefix_width,
            }
        )
        previous_prefix_width = prefix_width

    return block_metadata


def granularity_concat_block_count(granularity):
    get_granularity_metadata(granularity)
    return MATFORMER_GRANULARITY_ORDER.index(granularity) + 1


def get_block_membership_counts(granularities, total_blocks=None):
    if not granularities:
        raise ValueError("granularities must be a non-empty sequence")

    block_counts = [granularity_block_count(granularity) for granularity in granularities]
    max_blocks = max(block_counts)
    if total_blocks is None:
        total_blocks = max_blocks
    elif total_blocks < max_blocks:
        raise ValueError(
            "total_blocks cannot be smaller than the widest configured granularity"
        )

    return [
        sum(1 for block_count in block_counts if block_index < block_count)
        for block_index in range(total_blocks)
    ]


def get_gradient_membership_correction_scales(granularities, total_blocks=None):
    membership_counts = get_block_membership_counts(
        granularities,
        total_blocks=total_blocks,
    )
    total_losses = len(granularities)
    scales = []
    for membership_count in membership_counts:
        if membership_count == 0:
            scales.append(1.0)
            continue
        scales.append(total_losses / membership_count)
    return scales


def get_concat_block_membership_counts(
    intermediate_size,
    granularities,
    granularity_prefixes: Mapping[str, object] | None = None,
):
    if not granularities:
        raise ValueError("granularities must be a non-empty sequence")

    return get_concat_block_membership_counts_from_metadata(
        get_concat_block_metadata(
            intermediate_size,
            granularity_prefixes=granularity_prefixes,
            granularities=granularities,
        ),
        granularities,
    )


def get_concat_block_membership_counts_from_metadata(block_metadata, granularities):
    if not block_metadata:
        raise ValueError("block_metadata must be a non-empty sequence")
    if not granularities:
        raise ValueError("granularities must be a non-empty sequence")

    prefix_widths = []
    for granularity in granularities:
        block_index = MATFORMER_GRANULARITY_ORDER.index(granularity)
        if block_index >= len(block_metadata):
            raise ValueError(
                "Configured granularities exceed the available concat blocks"
            )
        prefix_widths.append(block_metadata[block_index]["cumulative_prefix_width"])
    return [
        sum(
            1
            for prefix_width in prefix_widths
            if prefix_width >= block["cumulative_prefix_width"]
        )
        for block in block_metadata
    ]


def get_concat_gradient_membership_correction_scales_from_metadata(
    block_metadata,
    granularities,
):
    membership_counts = get_concat_block_membership_counts_from_metadata(
        block_metadata,
        granularities,
    )
    total_losses = len(granularities)
    scales = []
    for membership_count in membership_counts:
        if membership_count == 0:
            scales.append(1.0)
            continue
        scales.append(total_losses / membership_count)
    return scales


def get_concat_gradient_membership_correction_scales(
    intermediate_size,
    granularities,
    granularity_prefixes: Mapping[str, object] | None = None,
):
    return get_concat_gradient_membership_correction_scales_from_metadata(
        get_concat_block_metadata(
            intermediate_size,
            granularity_prefixes=granularity_prefixes,
            granularities=granularities,
        ),
        granularities,
    )


def get_concat_layout_diagnostic(
    intermediate_size,
    granularities,
    granularity_prefixes: Mapping[str, object] | None = None,
):
    return _build_concat_layout_diagnostic(
        intermediate_size,
        granularities,
        granularity_prefixes=granularity_prefixes,
    )


def get_prefix_membership_segment_metadata(
    intermediate_size,
    granularities,
    granularity_prefixes: Mapping[str, object] | None = None,
):
    return _build_prefix_membership_segment_metadata(
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

        # Replace FFN in each layer with the selected MatFormer FFN variant
        for layer_idx in range(config.num_hidden_layers):
            mlp = self.mlp_cls(config, **self.mlp_kwargs)
            self.model.layers[layer_idx].mlp = mlp
            self.matformer_layers.append(mlp)

    def configure_subnetwork(self, flag):
        """Configure the subnetwork for all layers based on the flag."""
        self.current_layer_granularities = [flag] * len(self.matformer_layers)
        self.current_granularity_pattern = build_global_granularity_pattern(
            {
                "model": {
                    "granularity_sampling_mode": "global",
                    "granularities": [flag],
                    "num_layers": len(self.matformer_layers),
                },
                "run": {"run_id": ""},
            },
            granularities=[flag],
        )
        for layer in self.matformer_layers:
            layer.configure_subnetwork(flag)

    def configure_layer_granularities(self, layer_granularities):
        """Configure a repeating or explicit granularity pattern across layers."""
        self.current_granularity_pattern = build_per_layer_granularity_pattern(
            {
                "model": {
                    "granularity_sampling_mode": "per_layer",
                    "granularities": list(layer_granularities),
                    "num_layers": len(self.matformer_layers),
                },
                "run": {"run_id": ""},
            },
            layer_granularities=layer_granularities,
        )
        self.current_layer_granularities = list(
            self.current_granularity_pattern.selected_granularities
        )
        for layer, granularity in zip(
            self.matformer_layers,
            self.current_layer_granularities,
        ):
            layer.configure_subnetwork(granularity)
