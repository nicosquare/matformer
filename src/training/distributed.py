"""Distributed runtime helpers for config-driven training."""

from __future__ import annotations

import copy
import functools
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from src.utils.config import ConfigError

T = TypeVar("T")


@dataclass
class DistributedContext:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    strategy: str = "none"
    device: torch.device | str | None = None
    mixed_precision: str = "none"
    activation_checkpointing: bool = False
    fsdp_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rank = int(self.rank)
        self.local_rank = int(self.local_rank)
        self.world_size = int(self.world_size)
        self.enabled = bool(self.enabled and self.world_size > 1)
        self.strategy = self.strategy or "none"
        self.mixed_precision = str(self.mixed_precision or "none")
        self.activation_checkpointing = bool(self.activation_checkpointing)
        self.fsdp_config = dict(self.fsdp_config or {})
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
    training_config = _training_config(config)
    env_world_size = env_int("WORLD_SIZE", 1)
    expected_world_size = int(
        distributed_config.get("expected_world_size", env_world_size)
    )
    if expected_world_size not in {1, 2, 3, 4}:
        raise ConfigError(
            "training.distributed.expected_world_size must be between 1 and 4"
        )
    if env_world_size != expected_world_size:
        raise ConfigError(
            "Distributed runtime world size does not match preflight: "
            f"expected={expected_world_size}, runtime={env_world_size}"
        )
    requested = env_world_size > 1
    strategy = distributed_config.get("strategy") or ("fsdp" if requested else "none")
    fsdp_config = _fsdp_config(distributed_config)
    local_rank = get_local_rank(default=env_int("LOCAL_RANK", 0))
    resolved_device = resolve_device(device=device, local_rank=local_rank)
    runtime_settings = resolve_runtime_settings(
        training_config,
        resolved_device,
        single_process=env_world_size <= 1,
    )

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
        enabled=world_size > 1,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        strategy=strategy,
        device=resolved_device,
        mixed_precision=runtime_settings["resolved_mixed_precision"],
        activation_checkpointing=runtime_settings[
            "resolved_activation_checkpointing"
        ],
        fsdp_config=fsdp_config,
    )


def wrap_model_for_distributed(model, context: DistributedContext):
    if context.activation_checkpointing:
        apply_model_activation_checkpointing(model)
    if not context.enabled:
        return model
    if context.strategy != "fsdp":
        raise ValueError(f"Unsupported distributed strategy: {context.strategy}")

    fsdp_kwargs = build_fsdp_kwargs(context)
    return FSDP(model, **fsdp_kwargs)


def apply_model_activation_checkpointing(model) -> None:
    """Checkpoint Llama decoder layers for both local and FSDP execution."""

    checkpoint_fn = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=checkpoint_fn,
        check_fn=lambda module: isinstance(module, LlamaDecoderLayer),
    )


def resolve_runtime_settings(
    training_config: dict[str, Any],
    device: torch.device | str,
    *,
    single_process: bool,
) -> dict[str, Any]:
    """Resolve requested precision/checkpointing against the actual device."""

    resolved_device = torch.device(device)
    requested_precision = str(
        training_config.get(
            "requested_mixed_precision",
            training_config.get("mixed_precision", "none"),
        )
        or "none"
    ).lower()
    requested_checkpointing = bool(
        training_config.get(
            "requested_activation_checkpointing",
            training_config.get("activation_checkpointing", False),
        )
    )

    if requested_precision == "fp16" and single_process:
        raise ConfigError(
            "training.mixed_precision=fp16 is unsupported for single-process "
            "training; use bf16 on a CUDA device with native BF16 support"
        )
    if requested_precision not in {"none", "bf16", "fp16"}:
        raise ConfigError(
            f"Unsupported training.mixed_precision={requested_precision!r}"
        )

    if resolved_device.type != "cuda":
        resolved_precision = "none"
        resolved_checkpointing = False
    else:
        if requested_precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ConfigError(
                "training.mixed_precision=bf16 requires native CUDA BF16 support"
            )
        resolved_precision = requested_precision
        resolved_checkpointing = requested_checkpointing

    training_config["requested_mixed_precision"] = requested_precision
    training_config["resolved_mixed_precision"] = resolved_precision
    training_config["requested_activation_checkpointing"] = requested_checkpointing
    training_config["resolved_activation_checkpointing"] = resolved_checkpointing
    return {
        "requested_mixed_precision": requested_precision,
        "resolved_mixed_precision": resolved_precision,
        "requested_activation_checkpointing": requested_checkpointing,
        "resolved_activation_checkpointing": resolved_checkpointing,
    }


def autocast_context(
    config: dict[str, Any],
    device: torch.device | str,
):
    training = config.get("training", {})
    if (
        torch.device(device).type == "cuda"
        and training.get("resolved_mixed_precision") == "bf16"
    ):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def build_fsdp_kwargs(context: DistributedContext) -> dict[str, Any]:
    fsdp_config = context.fsdp_config
    kwargs: dict[str, Any] = {
        "use_orig_params": bool(fsdp_config.get("use_orig_params", True)),
    }

    if context.device is not None and torch.device(context.device).type == "cuda":
        kwargs["device_id"] = torch.device(context.device)
        kwargs["sync_module_states"] = bool(fsdp_config.get("sync_module_states", True))
    else:
        kwargs["sync_module_states"] = False

    sharding_strategy = _resolve_sharding_strategy(
        fsdp_config.get("sharding_strategy", "full_shard")
    )
    if sharding_strategy is not None:
        kwargs["sharding_strategy"] = sharding_strategy

    backward_prefetch = _resolve_backward_prefetch(
        fsdp_config.get("backward_prefetch", "backward_pre")
    )
    if backward_prefetch is not None:
        kwargs["backward_prefetch"] = backward_prefetch

    mixed_precision = build_mixed_precision(context.mixed_precision, context.device)
    if mixed_precision is not None:
        kwargs["mixed_precision"] = mixed_precision

    auto_wrap_policy = _build_auto_wrap_policy(fsdp_config)
    if auto_wrap_policy is not None:
        kwargs["auto_wrap_policy"] = auto_wrap_policy

    forward_prefetch = fsdp_config.get("forward_prefetch", False)
    if forward_prefetch is not None:
        kwargs["forward_prefetch"] = bool(forward_prefetch)

    if bool(fsdp_config.get("cpu_offload", False)):
        kwargs["cpu_offload"] = CPUOffload(offload_params=True)

    if "limit_all_gathers" in fsdp_config:
        kwargs["limit_all_gathers"] = bool(fsdp_config["limit_all_gathers"])

    return kwargs


def build_mixed_precision(
    choice: str | None,
    device: torch.device | str | None,
) -> MixedPrecision | None:
    if device is None or torch.device(device).type != "cuda":
        return None

    normalized = str(choice or "none")
    if normalized == "none":
        return None

    if normalized == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ConfigError("BF16 requires native CUDA BF16 support")
        dtype = torch.bfloat16
    elif normalized == "fp16":
        dtype = torch.float16
    else:
        raise ValueError(f"Unsupported mixed precision choice: {choice}")

    return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)


def _build_auto_wrap_policy(fsdp_config: dict[str, Any]):
    policy_name = str(fsdp_config.get("auto_wrap_policy", "transformer_based_wrap"))
    if policy_name in ("", "none", "no_wrap"):
        return None
    if policy_name != "transformer_based_wrap":
        raise ValueError(f"Unsupported FSDP auto_wrap_policy: {policy_name}")

    layer_cls_names = fsdp_config.get("transformer_layer_cls_to_wrap", ["LlamaDecoderLayer"])
    if isinstance(layer_cls_names, str):
        layer_cls_names = [layer_cls_names]

    layer_classes = set()
    for class_name in layer_cls_names:
        layer_classes.add(_resolve_transformer_layer_class(str(class_name)))

    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=layer_classes,
    )


def _resolve_transformer_layer_class(class_name: str):
    if class_name == "LlamaDecoderLayer":
        return LlamaDecoderLayer
    raise ValueError(f"Unsupported transformer layer class for FSDP wrapping: {class_name}")


def _resolve_sharding_strategy(strategy: Any) -> ShardingStrategy | None:
    if strategy in (None, "", "none"):
        return None

    strategy_name = str(strategy).upper()
    try:
        return ShardingStrategy[strategy_name]
    except KeyError as error:
        raise ValueError(f"Unsupported FSDP sharding_strategy: {strategy}") from error


def _resolve_backward_prefetch(choice: Any) -> BackwardPrefetch | None:
    if choice in (None, "", "none"):
        return None

    prefetch_name = str(choice).upper()
    try:
        return BackwardPrefetch[prefetch_name]
    except KeyError as error:
        raise ValueError(f"Unsupported FSDP backward_prefetch: {choice}") from error


def _training_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    training = config.get("training", {})
    if not isinstance(training, dict):
        return {}
    return training


def _fsdp_config(distributed_config: dict[str, Any]) -> dict[str, Any]:
    fsdp_config = distributed_config.get("fsdp", {})
    if not isinstance(fsdp_config, dict):
        return {}
    return copy.deepcopy(fsdp_config)


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


def _validate_probabilistic_controller_broadcast_payload(
    payload: Any,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ConfigError("Broadcast Bayesian controller payload is missing")
    controller_state = payload.get("controller_state")
    action = payload.get("action")
    if not isinstance(controller_state, dict):
        raise ConfigError("Broadcast Bayesian controller state is missing")
    if (
        isinstance(controller_state.get("method_version"), bool)
        or not isinstance(controller_state.get("method_version"), int)
        or controller_state["method_version"] <= 0
        or controller_state.get("scope") not in {"global", "per_block"}
        or not isinstance(controller_state.get("belief"), dict)
        or not isinstance(controller_state.get("sampling"), dict)
    ):
        raise ConfigError("Broadcast Bayesian controller state is incomplete")
    round_index = controller_state["belief"].get("round_index")
    sample_count = controller_state["sampling"].get("sample_count")
    if (
        isinstance(round_index, bool)
        or not isinstance(round_index, int)
        or round_index < 0
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 0
    ):
        raise ConfigError("Broadcast Bayesian controller state is incomplete")
    if not isinstance(action, dict):
        raise ConfigError("Broadcast Bayesian controller action is missing")
    scope = controller_state["scope"]
    if scope == "global" and not isinstance(action.get("global_granularity"), str):
        raise ConfigError("Broadcast Bayesian global action is incomplete")
    if scope == "per_block" and not isinstance(action.get("block_granularities"), list):
        raise ConfigError("Broadcast Bayesian per-block action is incomplete")
    return copy.deepcopy(payload)


def broadcast_probabilistic_controller_state(
    *,
    controller_state: dict[str, Any] | None,
    action: dict[str, Any] | None,
    context: DistributedContext | None = None,
) -> dict[str, Any]:
    """Broadcast rank zero's validated controller state and selected action."""

    is_rank_zero = (
        context.is_rank_zero
        if context is not None
        else get_rank(default=0) == 0
    )
    payload = None
    if is_rank_zero:
        if controller_state is None:
            raise ConfigError(
                "rank zero must provide Bayesian controller state for broadcast"
            )
        payload = _validate_probabilistic_controller_broadcast_payload(
            {
                "controller_state": controller_state,
                "action": action,
            }
        )
    received = broadcast_object(payload, context=context, src=0)
    return _validate_probabilistic_controller_broadcast_payload(received)


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


def sum_float(
    value: float,
    device: torch.device | str,
    context: DistributedContext | None = None,
) -> float:
    if context is not None and not context.enabled:
        return float(value)
    if not distributed_is_initialized():
        return float(value)
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor.item())


def gather_objects(
    value: T,
    context: DistributedContext | None = None,
) -> list[T]:
    if context is not None and not context.enabled:
        return [value]
    if not distributed_is_initialized():
        return [value]
    gathered: list[T | None] = [None] * get_world_size(default=1)
    torch.distributed.all_gather_object(gathered, value)
    return [item for item in gathered if item is not None]


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
