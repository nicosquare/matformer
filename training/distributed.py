"""Distributed runtime helpers for config-driven training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import torch


T = TypeVar("T")


@dataclass
class DistributedContext:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    strategy: str = "none"
    device: torch.device | str | None = None

    def __post_init__(self) -> None:
        self.rank = int(self.rank)
        self.local_rank = int(self.local_rank)
        self.world_size = int(self.world_size)
        self.enabled = bool(self.enabled and self.world_size > 1)
        self.strategy = self.strategy or "none"
        if self.device is not None:
            self.device = torch.device(self.device)

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0


def prepare_distributed_context(
    config: dict[str, Any] | None = None,
    device: torch.device | str | None = None,
    initialize_process_group: bool = True,
    backend: str | None = None,
) -> DistributedContext:
    distributed_config = _distributed_config(config)
    env_world_size = env_int("WORLD_SIZE", 1)
    requested = bool(distributed_config.get("enabled", False)) or env_world_size > 1
    strategy = distributed_config.get("strategy") or ("fsdp" if requested else "none")
    local_rank = get_local_rank(default=env_int("LOCAL_RANK", 0))
    resolved_device = resolve_device(device=device, local_rank=local_rank)

    if resolved_device.type == "cuda":
        torch.cuda.set_device(resolved_device)

    if requested and env_world_size > 1:
        backend = backend or ("nccl" if resolved_device.type == "cuda" else "gloo")
        if initialize_process_group and not distributed_is_initialized():
            init_kwargs = {"backend": backend}
            if backend == "nccl" and resolved_device.type == "cuda":
                init_kwargs["device_id"] = resolved_device
            torch.distributed.init_process_group(**init_kwargs)

    rank = get_rank(default=env_int("RANK", 0))
    world_size = get_world_size(default=env_world_size)

    return DistributedContext(
        enabled=requested and world_size > 1,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        strategy=strategy,
        device=resolved_device,
    )


def wrap_model_for_distributed(model, context: DistributedContext):
    if not context.enabled:
        return model
    if context.strategy != "fsdp":
        raise ValueError(f"Unsupported distributed strategy: {context.strategy}")

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if context.device is not None and torch.device(context.device).type == "cuda":
        return FSDP(
            model,
            device_id=torch.device(context.device),
            use_orig_params=True,
        )
    return FSDP(model, use_orig_params=True)


def rank_zero_only(
    context: DistributedContext | None,
    action: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T | None:
    if should_write_shared_artifact(context):
        return action(*args, **kwargs)
    return None


def should_write_shared_artifact(context: DistributedContext | None = None) -> bool:
    if context is None:
        return get_rank(default=0) == 0
    if not context.enabled:
        return True
    return context.is_rank_zero


def barrier(context: DistributedContext | None = None) -> None:
    if context is not None and not context.enabled:
        return
    if distributed_is_initialized():
        torch.distributed.barrier()


def broadcast_object(
    value: T | None,
    context: DistributedContext | None = None,
    src: int = 0,
) -> T | None:
    if context is not None and not context.enabled:
        return value
    if not distributed_is_initialized():
        return value

    object_list: list[T | None] = [value if get_rank(default=0) == src else None]
    torch.distributed.broadcast_object_list(object_list, src=src)
    return object_list[0]


def sum_int(
    value: int,
    device: torch.device | str,
    context: DistributedContext | None = None,
) -> int:
    if context is not None and not context.enabled:
        return int(value)
    if not distributed_is_initialized():
        return int(value)

    tensor = torch.tensor(int(value), dtype=torch.long, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return int(tensor.item())


def destroy_distributed_process_group(
    context: DistributedContext | None = None,
) -> None:
    if context is not None and not context.enabled:
        return
    if distributed_is_initialized():
        torch.distributed.destroy_process_group()


def get_rank(default: int = 0) -> int:
    if distributed_is_initialized():
        return int(torch.distributed.get_rank())
    return env_int("RANK", default)


def get_local_rank(default: int = 0) -> int:
    return env_int("LOCAL_RANK", default)


def get_world_size(default: int = 1) -> int:
    if distributed_is_initialized():
        return int(torch.distributed.get_world_size())
    return env_int("WORLD_SIZE", default)


def distributed_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def resolve_device(
    device: torch.device | str | None = None,
    local_rank: int = 0,
) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value in (None, ""):
        return default
    try:
        parsed = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from error
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, got {raw_value!r}")
    return parsed


def _distributed_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    training = config.get("training", {})
    if not isinstance(training, dict):
        return {}
    distributed = training.get("distributed", {})
    if not isinstance(distributed, dict):
        return {}
    return distributed
