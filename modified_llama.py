import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP


MATFORMER_GRANULARITY_ORDER = ("s", "m", "l", "xl")
MATFORMER_GRANULARITIES = {
    "s": {
        "display_name": "S",
        "ffn_ratio": 0.5,
        "full_intermediate_fraction": 0.125,
    },
    "m": {
        "display_name": "M",
        "ffn_ratio": 1.0,
        "full_intermediate_fraction": 0.25,
    },
    "l": {
        "display_name": "L",
        "ffn_ratio": 2.0,
        "full_intermediate_fraction": 0.5,
    },
    "xl": {
        "display_name": "XL",
        "ffn_ratio": 4.0,
        "full_intermediate_fraction": 1.0,
    },
}


def get_granularity_metadata(granularity):
    if granularity not in MATFORMER_GRANULARITIES:
        raise ValueError(f"Unknown MatFormer granularity: {granularity}")
    return MATFORMER_GRANULARITIES[granularity]


def granularity_prefix_width(intermediate_size, granularity):
    metadata = get_granularity_metadata(granularity)
    prefix_width = int(intermediate_size * metadata["full_intermediate_fraction"])
    if prefix_width <= 0:
        raise ValueError(
            f"Granularity {granularity} produced empty FFN prefix for "
            f"intermediate_size={intermediate_size}"
        )
    return prefix_width


def get_ffn_prefix_metadata(intermediate_size):
    metadata = []
    for granularity in MATFORMER_GRANULARITY_ORDER:
        granularity_metadata = get_granularity_metadata(granularity)
        metadata.append(
            {
                "name": granularity,
                "display_name": granularity_metadata["display_name"],
                "ffn_ratio": granularity_metadata["ffn_ratio"],
                "full_intermediate_fraction": granularity_metadata[
                    "full_intermediate_fraction"
                ],
                "prefix_width": granularity_prefix_width(
                    intermediate_size,
                    granularity,
                ),
            }
        )
    return metadata


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


def get_concat_block_metadata(intermediate_size):
    base_block_width = granularity_prefix_width(
        intermediate_size,
        MATFORMER_GRANULARITY_ORDER[0],
    )
    total_blocks = granularity_block_count(MATFORMER_GRANULARITY_ORDER[-1])
    block_metadata = []
    for block_index in range(total_blocks):
        prefix_width = (block_index + 1) * base_block_width
        block_metadata.append(
            {
                "name": f"block_{block_index + 1}",
                "display_name": f"B{block_index + 1}",
                "ffn_ratio": MATFORMER_GRANULARITIES[MATFORMER_GRANULARITY_ORDER[0]][
                    "ffn_ratio"
                ],
                "full_intermediate_fraction": prefix_width / intermediate_size,
                "prefix_width": prefix_width,
                "block_width": base_block_width,
                "cumulative_prefix_width": prefix_width,
            }
        )

    return block_metadata


class ModifiedLlamaMLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.intermediate_size = config.intermediate_size
        self.ffn_prefix_metadata = get_ffn_prefix_metadata(self.intermediate_size)
        self.current_granularity = None
        self.current_subset_hd = None

    def configure_subnetwork(self, flag):
        """Configure subnetwork size based on flag."""
        self.current_granularity = flag
        self.current_subset_hd = granularity_prefix_width(self.intermediate_size, flag)

    def prefix_parameter_count(self, flag, trainable_only=False):
        prefix_width = granularity_prefix_width(self.intermediate_size, flag)
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
    def __init__(self, config):
        super().__init__(config)
        self.intermediate_size = config.intermediate_size
        self.ffn_prefix_metadata = get_ffn_prefix_metadata(self.intermediate_size)
        self.ffn_concat_block_metadata = get_concat_block_metadata(
            self.intermediate_size
        )
        self.current_granularity = None
        self.current_subset_hd = None
        self.current_subset_blocks = None

        gate_weight = self.gate_proj.weight.detach().clone()
        gate_bias = (
            None
            if self.gate_proj.bias is None
            else self.gate_proj.bias.detach().clone()
        )
        up_weight = self.up_proj.weight.detach().clone()
        up_bias = None if self.up_proj.bias is None else self.up_proj.bias.detach().clone()
        down_weight = self.down_proj.weight.detach().clone()
        down_bias = (
            None
            if self.down_proj.bias is None
            else self.down_proj.bias.detach().clone()
        )

        del self.gate_proj
        del self.up_proj
        del self.down_proj

        self.gate_weight_blocks = nn.ParameterList()
        self.up_weight_blocks = nn.ParameterList()
        self.down_weight_blocks = nn.ParameterList()
        self.gate_bias_blocks = nn.ParameterList()
        self.up_bias_blocks = nn.ParameterList()

        offset = 0
        for block_metadata in self.ffn_concat_block_metadata:
            block_width = block_metadata["block_width"]
            next_offset = offset + block_width

            self.gate_weight_blocks.append(
                nn.Parameter(gate_weight[offset:next_offset].contiguous())
            )
            self.up_weight_blocks.append(
                nn.Parameter(up_weight[offset:next_offset].contiguous())
            )
            self.down_weight_blocks.append(
                nn.Parameter(down_weight[:, offset:next_offset].contiguous())
            )

            if gate_bias is not None:
                self.gate_bias_blocks.append(
                    nn.Parameter(gate_bias[offset:next_offset].contiguous())
                )
            if up_bias is not None:
                self.up_bias_blocks.append(
                    nn.Parameter(up_bias[offset:next_offset].contiguous())
                )

            offset = next_offset

        self.down_bias = None if down_bias is None else nn.Parameter(down_bias.contiguous())

    def configure_subnetwork(self, flag):
        self.current_granularity = flag
        self.current_subset_blocks = granularity_block_count(flag)
        self.current_subset_hd = granularity_prefix_width(self.intermediate_size, flag)

    def prefix_parameter_count(self, flag, trainable_only=False):
        block_count = granularity_block_count(flag)
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
        self.ffn_prefix_metadata = get_ffn_prefix_metadata(config.intermediate_size)
        self.mlp_cls = mlp_cls
        self.mlp_kwargs = dict(mlp_kwargs or {})
        self.matformer_layers = []
        self.current_layer_granularities = None

        # Replace FFN in each layer with the selected MatFormer FFN variant
        for layer_idx in range(config.num_hidden_layers):
            mlp = self.mlp_cls(config, **self.mlp_kwargs)
            self.model.layers[layer_idx].mlp = mlp
            self.matformer_layers.append(mlp)

    def configure_subnetwork(self, flag):
        """Configure the subnetwork for all layers based on the flag."""
        self.current_layer_granularities = [flag] * len(self.matformer_layers)
        for layer in self.matformer_layers:
            layer.configure_subnetwork(flag)

    def configure_layer_granularities(self, layer_granularities):
        """Configure a repeating or explicit granularity pattern across layers."""
        expanded_pattern = expand_layer_granularity_pattern(
            layer_granularities,
            len(self.matformer_layers),
        )
        self.current_layer_granularities = expanded_pattern
        for layer, granularity in zip(self.matformer_layers, expanded_pattern):
            layer.configure_subnetwork(granularity)
