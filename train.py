import argparse
import os
import random
import functools

import torch
import torch.distributed as dist
from datasets import load_dataset
from modified_llama import ModifiedLlamaForCausalLM, CatLlamaMLP
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import yaml

from training.run import build_optimizer_and_scheduler
from utils.config import (
    DEFAULT_MODEL_VARIANT,
    VALID_MODEL_VARIANTS,
    VALID_OPTIMIZER_NAMES,
)


FLAGS = ['s', 'm', 'l', 'xl']


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="YAML experiment config for the Spec Kit flow.")
    parser.add_argument("--run-id", help="Run id to select from a matrix config.")
    parser.add_argument(
        "--output-root",
        help="Root directory for config-driven run artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        help="Explicit output directory for one config-driven run.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted config override, for example training.max_steps=10.",
    )
    parser.add_argument("--model-name", default="NousResearch/Llama-3.2-1B")
    parser.add_argument(
        "--model-variant",
        choices=sorted(VALID_MODEL_VARIANTS),
        default=DEFAULT_MODEL_VARIANT,
        help="Select the MatFormer family variant for the legacy direct path.",
    )
    parser.add_argument("--dataset-name", default="vilm/RedPajama-v2-small")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-size", type=int, default=10000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8, help="Per-process batch size.")
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--num-training-steps", type=int, default=10000)
    parser.add_argument("--num-warmup-steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer-name",
        choices=sorted(VALID_OPTIMIZER_NAMES),
        default="adamw",
        help="Select the optimizer for the legacy direct path.",
    )
    parser.add_argument(
        "--optimizer-kwargs",
        default="{}",
        help="YAML mapping of optimizer kwargs for the legacy direct path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocess-num-proc", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=["none", "bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Checkpoint Llama decoder layers when running under torchrun/FSDP.",
    )
    return parser.parse_args(argv)


def parse_optimizer_kwargs(raw_value):
    parsed = yaml.safe_load(raw_value)
    if parsed in (None, ""):
        return {}
    if not isinstance(parsed, dict):
        raise ValueError("--optimizer-kwargs must parse to a mapping")
    return parsed


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("FSDP distributed training requires CUDA devices.")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    return distributed, rank, local_rank, world_size, device


def print_rank0(rank, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_data(example, tokenizer, max_length):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_length)


def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(b["input_ids"]) for b in batch])
    attention_mask = torch.stack([torch.tensor(b["attention_mask"]) for b in batch])
    labels = input_ids.clone()

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_processed_dataset(args, tokenizer, distributed, rank):
    def build_dataset():
        print_rank0(rank, "loading dataset", flush=True)
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
        dataset = dataset.shuffle(seed=args.seed)

        if args.dataset_size is not None:
            dataset = dataset.select(range(min(args.dataset_size, len(dataset))))

        print_rank0(rank, "preprocessing dataset", flush=True)
        return dataset.map(
            functools.partial(preprocess_data, tokenizer=tokenizer, max_length=args.max_length),
            num_proc=max(1, args.preprocess_num_proc),
        )

    if not distributed:
        return build_dataset()

    if rank == 0:
        dataset = build_dataset()
    dist.barrier()

    if rank != 0:
        dataset = build_dataset()
    dist.barrier()

    return dataset


def build_mixed_precision(choice, rank):
    if choice == "none":
        return None

    if choice == "bf16":
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            print_rank0(rank, "bf16 is not supported on this GPU; falling back to fp16.", flush=True)
            dtype = torch.float16
    else:
        dtype = torch.float16

    return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)


def wrap_with_fsdp(model, args, device, rank):
    if args.activation_checkpointing:
        checkpoint_fn = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_fn,
            check_fn=lambda module: isinstance(module, LlamaDecoderLayer),
        )

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=build_mixed_precision(args.mixed_precision, rank),
        device_id=device,
        sync_module_states=True,
        use_orig_params=True,
    )


def configure_subnetwork(model, flag):
    target = model.module if hasattr(model, "module") else model
    target.configure_subnetwork(flag)


def select_training_flag(device, distributed):
    if not distributed:
        return random.choice(FLAGS)

    flag_idx = torch.empty((), dtype=torch.long, device=device)
    if dist.get_rank() == 0:
        flag_idx.fill_(random.randrange(len(FLAGS)))
    dist.broadcast(flag_idx, src=0)
    return FLAGS[int(flag_idx.item())]


def move_batch_to_device(batch, device):
    return {
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
    }


def reduce_mean(value, distributed):
    if not distributed:
        return value.detach().float().item()

    reduced = value.detach().float().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return reduced.item()


def evaluate_model(model, eval_dataloader, flags, device, distributed):
    """Evaluate the model on the eval dataset for each flag and return losses."""
    model.eval()
    eval_losses = {flag: 0.0 for flag in flags}

    with torch.no_grad():
        for flag in flags:
            total_loss = 0.0
            num_batches = 0
            for batch in eval_dataloader:
                batch = move_batch_to_device(batch, device)

                configure_subnetwork(model, flag)

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                total_loss += outputs.loss.item()
                num_batches += 1

            stats = torch.tensor([total_loss, float(num_batches)], device=device)
            if distributed:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)

            eval_losses[flag] = (stats[0] / stats[1]).item()

    model.train()
    return eval_losses


def build_legacy_model(config, model_variant):
    if model_variant == "cat_llama":
        return ModifiedLlamaForCausalLM(config=config, mlp_cls=CatLlamaMLP)
    return ModifiedLlamaForCausalLM(config=config)


def main():
    args = parse_args()
    if args.config:
        from training.run import run_from_config_path

        overrides = list(args.override)
        output_root = args.output_root or os.environ.get("OUTPUT_ROOT")
        if output_root:
            overrides.append(f"run.output_root={output_root}")

        run_from_config_path(
            args.config,
            run_id=args.run_id,
            overrides=overrides,
            output_dir=args.output_dir,
        )
        return

    distributed, rank, _, world_size, device = setup_distributed()
    set_random_seed(args.seed)

    print_rank0(
        rank,
        f"using device={device}, world_size={world_size}, per-process batch_size={args.batch_size}",
        flush=True,
    )

    print_rank0(rank, "loading tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_processed_dataset(args, tokenizer, distributed, rank)

    print_rank0(rank, "loading config", flush=True)
    config = LlamaConfig.from_pretrained(args.model_name)
    config.use_cache = False

    print_rank0(
        rank,
        f"initializing model for variant={args.model_variant}. This may take a while... ",
        end="",
        flush=True,
    )
    model = build_legacy_model(config=config, model_variant=args.model_variant)
    if distributed:
        model = wrap_with_fsdp(model, args, device, rank)
    else:
        model = model.to(device)
    print_rank0(rank, "Done!", flush=True)

    eval_size = min(args.eval_batches * args.batch_size, len(dataset))
    eval_dataset = dataset.select(range(eval_size))
    train_dataset = dataset.select(range(eval_size, len(dataset)))

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False) if distributed else None
    pin_memory = device.type == "cuda"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=pin_memory,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        sampler=eval_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=pin_memory,
    )

    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        {
            "learning_rate": args.learning_rate,
            "resolved_learning_rate": args.learning_rate,
            "max_steps": args.num_training_steps,
            "warmup_steps": args.num_warmup_steps,
            "resolved_warmup_steps": args.num_warmup_steps,
            "optimizer_name": args.optimizer_name,
            "optimizer_kwargs": parse_optimizer_kwargs(args.optimizer_kwargs),
        },
    )
    model.train()

    step = 0
    epoch = 0

    while step < args.num_training_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch in train_dataloader:
            batch = move_batch_to_device(batch, device)
            flag = select_training_flag(device, distributed)

            configure_subnetwork(model, flag)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            loss_value = reduce_mean(loss, distributed)
            print_rank0(rank, f"Step {step}, Flag: {flag}, Loss: {loss_value}", flush=True)

            if args.eval_interval > 0 and step % args.eval_interval == 0:
                eval_losses = evaluate_model(model, eval_dataloader, FLAGS, device, distributed)
                print_rank0(rank, f"Step {step}, Eval Losses: {eval_losses}", flush=True)

            if step >= args.num_training_steps:
                break

        epoch += 1


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
