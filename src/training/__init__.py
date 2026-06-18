"""Training helpers for MatFormer reproduction experiments."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_MODULE_EXPORTS = {
    "baselines",
    "checkpointing",
    "data",
    "distributed",
    "run",
    "steps",
    "warmup",
}

_SYMBOL_EXPORTS: dict[str, tuple[str, str]] = {
    "run_from_config_path": ("run", "run_from_config_path"),
    "run_training": ("run", "run_training"),
    "build_artifact_parameter_counts": ("run", "build_artifact_parameter_counts"),
    "build_model": ("run", "build_model"),
    "build_llama_config": ("run", "build_llama_config"),
    "load_tokenizer": ("run", "load_tokenizer"),
    "build_dataloaders": ("steps", "build_dataloaders"),
    "build_optimizer_and_scheduler": ("steps", "build_optimizer_and_scheduler"),
    "train_for_steps": ("steps", "train_for_steps"),
    "build_initial_continuation_state": (
        "checkpointing",
        "build_initial_continuation_state",
    ),
    "load_run_continuation_state": ("checkpointing", "load_run_continuation_state"),
    "load_checkpoint_state": ("checkpointing", "load_checkpoint_state"),
    "update_run_continuation_state": (
        "checkpointing",
        "update_run_continuation_state",
    ),
    "run_pre_nested_warmup_phase": ("warmup", "run_pre_nested_warmup_phase"),
    "should_run_pre_nested_warmup": ("warmup", "should_run_pre_nested_warmup"),
    "build_pre_nested_warmup_state": ("warmup", "build_pre_nested_warmup_state"),
    "resolve_pre_nested_warmup_target_steps": (
        "warmup",
        "resolve_pre_nested_warmup_target_steps",
    ),
    "DistributedContext": ("distributed", "DistributedContext"),
    "prepare_distributed_context": ("distributed", "prepare_distributed_context"),
    "wrap_model_for_distributed": ("distributed", "wrap_model_for_distributed"),
    "should_write_shared_artifact": (
        "distributed",
        "should_write_shared_artifact",
    ),
}

__all__ = sorted(_MODULE_EXPORTS | set(_SYMBOL_EXPORTS))


def __getattr__(name: str) -> Any:
    if name in _MODULE_EXPORTS:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    export = _SYMBOL_EXPORTS.get(name)
    if export is not None:
        module_name, attribute_name = export
        module = import_module(f"{__name__}.{module_name}")
        value = getattr(module, attribute_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _MODULE_EXPORTS | set(_SYMBOL_EXPORTS))
