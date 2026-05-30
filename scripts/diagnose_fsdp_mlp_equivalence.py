import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import LlamaConfig

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modified_llama import CatLlamaMLP, MATFORMER_GRANULARITY_ORDER, ModifiedLlamaMLP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--intermediate-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "sgd"],
        default="adamw",
    )
    parser.add_argument(
        "--use-fsdp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-orig-params",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def init_distributed_if_needed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, device


def destroy_distributed_if_needed():
    if dist.is_initialized():
        dist.destroy_process_group()


def tiny_llama_config(hidden_size, intermediate_size):
    return LlamaConfig(
        vocab_size=32,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=16,
        tie_word_embeddings=False,
    )


def build_paired_mlps(config, seed):
    torch.manual_seed(seed)
    dense = ModifiedLlamaMLP(config)
    torch.manual_seed(seed)
    cat = CatLlamaMLP(config)
    return dense, cat


def build_optimizer(module, optimizer_name, lr):
    params = module.parameters()
    if optimizer_name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    return torch.optim.AdamW(params, lr=lr)


def maybe_wrap(module, use_fsdp, use_orig_params, device):
    if not use_fsdp:
        return module.to(device)
    module = module.to(device)
    return FSDP(module, device_id=device, use_orig_params=use_orig_params)


def configure_granularity(module, granularity):
    target = module.module if hasattr(module, "module") else module
    target.configure_subnetwork(granularity)


def nested_all_loss(module, x):
    losses = []
    outputs = {}
    for granularity in MATFORMER_GRANULARITY_ORDER:
        configure_granularity(module, granularity)
        out = module(x)
        outputs[granularity] = out.detach().cpu()
        losses.append(out.mean())
    return torch.stack(losses).mean(), outputs


def full_state_dict(module):
    if not isinstance(module, FSDP):
        return {name: param.detach().cpu().clone() for name, param in module.named_parameters()}

    with FSDP.summon_full_params(module, writeback=False, recurse=True):
        target = module.module
        return {
            name: param.detach().cpu().clone()
            for name, param in target.named_parameters()
        }


def full_grad_dict(module):
    if not isinstance(module, FSDP):
        return {
            name: param.grad.detach().cpu().clone()
            for name, param in module.named_parameters()
            if param.grad is not None
        }

    with FSDP.summon_full_params(module, writeback=False, recurse=True):
        target = module.module
        return {
            name: param.grad.detach().cpu().clone()
            for name, param in target.named_parameters()
            if param.grad is not None
        }


def diff_equivalent_states(dense_state, cat_state):
    gate_block_count = len(
        [name for name in cat_state if name.startswith("gate_weight_blocks.")]
    )
    diffs = {}
    for dense_name, dense_tensor in dense_state.items():
        cat_tensor = None
        if dense_name.startswith("gate_proj.weight"):
            tensors = [
                cat_state[f"gate_weight_blocks.{index}"]
                for index in range(gate_block_count)
                if f"gate_weight_blocks.{index}" in cat_state
            ]
            if tensors:
                cat_tensor = torch.cat(tensors, dim=0)
        elif dense_name.startswith("up_proj.weight"):
            tensors = [
                cat_state[f"up_weight_blocks.{index}"]
                for index in range(gate_block_count)
                if f"up_weight_blocks.{index}" in cat_state
            ]
            if tensors:
                cat_tensor = torch.cat(tensors, dim=0)
        elif dense_name.startswith("down_proj.weight"):
            tensors = [
                cat_state[f"down_weight_blocks.{index}"]
                for index in range(gate_block_count)
                if f"down_weight_blocks.{index}" in cat_state
            ]
            if tensors:
                cat_tensor = torch.cat(tensors, dim=1)
        elif dense_name.startswith("gate_proj.bias"):
            tensors = [
                cat_state[f"gate_bias_blocks.{index}"]
                for index in range(gate_block_count)
                if f"gate_bias_blocks.{index}" in cat_state
            ]
            if tensors:
                cat_tensor = torch.cat(tensors, dim=0)
        elif dense_name.startswith("up_proj.bias"):
            tensors = [
                cat_state[f"up_bias_blocks.{index}"]
                for index in range(gate_block_count)
                if f"up_bias_blocks.{index}" in cat_state
            ]
            if tensors:
                cat_tensor = torch.cat(tensors, dim=0)
        elif dense_name == "down_proj.bias":
            cat_tensor = cat_state.get("down_bias")
        else:
            continue
        if cat_tensor is None:
            diffs[dense_name] = None
            continue
        diffs[dense_name] = float((dense_tensor - cat_tensor).abs().max().item())
    return diffs


def main():
    args = parse_args()
    rank, world_size, device = init_distributed_if_needed()
    try:
        config = tiny_llama_config(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
        )
        dense, cat = build_paired_mlps(config, seed=args.seed)
        dense = maybe_wrap(dense, args.use_fsdp, args.use_orig_params, device)
        cat = maybe_wrap(cat, args.use_fsdp, args.use_orig_params, device)

        x = torch.randn(
            args.batch_size,
            args.seq_len,
            args.hidden_size,
            device=device,
        )

        dense_opt = build_optimizer(dense, args.optimizer, args.lr)
        cat_opt = build_optimizer(cat, args.optimizer, args.lr)

        dense_opt.zero_grad(set_to_none=True)
        cat_opt.zero_grad(set_to_none=True)

        dense_loss, dense_outputs = nested_all_loss(dense, x)
        cat_loss, cat_outputs = nested_all_loss(cat, x)

        dense_loss.backward()
        cat_loss.backward()

        dense_grads = full_grad_dict(dense)
        cat_grads = full_grad_dict(cat)

        dense_opt.step()
        cat_opt.step()

        dense_state = full_state_dict(dense)
        cat_state = full_state_dict(cat)

        output_diffs = {
            granularity: float(
                (dense_outputs[granularity] - cat_outputs[granularity]).abs().max().item()
            )
            for granularity in MATFORMER_GRANULARITY_ORDER
        }
        grad_diffs = diff_equivalent_states(dense_grads, cat_grads)
        state_diffs = diff_equivalent_states(dense_state, cat_state)
        result = {
            "rank": rank,
            "world_size": world_size,
            "optimizer": args.optimizer,
            "use_fsdp": args.use_fsdp,
            "use_orig_params": args.use_orig_params,
            "dense_loss": float(dense_loss.detach().cpu().item()),
            "cat_loss": float(cat_loss.detach().cpu().item()),
            "loss_diff": float(abs(dense_loss.detach().cpu().item() - cat_loss.detach().cpu().item())),
            "output_max_abs_diff": output_diffs,
            "grad_max_abs_diff": grad_diffs,
            "state_max_abs_diff": state_diffs,
        }
        if rank == 0:
            print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        destroy_distributed_if_needed()


if __name__ == "__main__":
    main()
