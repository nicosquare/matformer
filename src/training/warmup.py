"""Config-driven training flow for MatFormer reproduction runs."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import copy
from contextlib import contextmanager
import random
import os
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, get_scheduler

from src.evaluation.validation import (
    configure_model_granularity,
    evaluate_validation_per_granularity,
    move_batch_to_device,
    perplexity_from_loss,
    validation_results_to_metric_rows,
)
from src.models.ffn import (
    CatLlamaMLP,
    build_concat_layout_diagnostic,
    get_ffn_prefix_metadata,
)
from src.models.adaptive_sampler import (
    build_adaptive_reward_record,
    build_adaptive_sampler_artifact_fields,
    build_adaptive_sampler_state,
    AdaptiveSamplerState,
    coerce_adaptive_sampler_state,
    normalize_adaptive_sampler_state,
    select_adaptive_sampler_layer_granularities,
    summarize_adaptive_sampler_state,
    update_adaptive_sampler_state,
)
from src.models.correction import summarize_correction_context_from_config
from src.models.granularity import summarize_granularity_pattern_from_config
from src.models.wiring import (
    ModifiedLlamaForCausalLM,
    prime_standalone_granularity_state,
    record_runtime_sampling_provenance,
)
from src.training.checkpointing import NoopHeartbeatWriter
from src.training.checkpointing import (
    build_initial_continuation_state,
    continuation_latest_checkpoint_policy,
    maybe_write_latest_checkpoint,
)
from src.training.steps import train_for_steps
from src.utils.config import (
    ConfigError,
    attach_parameter_counts_to_config,
    resolve_run_config,
    resolve_optimizer_kwargs,
    resolve_sampling_mode_from_config_sections,
    resolve_training_length_for_world_size,
    validate_run_config,
)
from src.utils.heartbeats import HeartbeatCadence, HeartbeatWriter
from src.utils.metrics import (
    build_checkpoint_summary_fields,
    build_monitoring_summary_fields,
    build_parameter_counts_by_granularity,
    build_run_summary,
    build_scaling_result_rows,
    write_config_artifact,
    write_failed_run_summary,
    json_artifact_value,
    write_json_artifact,
    write_metrics_csv,
    write_run_summary,
    write_scaling_results_csv,
    summarize_runtime_granularity_pattern_from_config,
)
from src.utils.monitoring import group_loss_rows_by_series
def update_pre_nested_warmup_state(
    config: dict[str, Any],
    state: Mapping[str, Any],
) -> None:
    training = config.setdefault("training", {})
    warmup = training.setdefault("pre_nested_warmup", {})
    if not isinstance(warmup, dict):
        raise ConfigError("training.pre_nested_warmup must be a mapping when provided")

    for key in [
        "enabled",
        "active",
        "duration",
        "unit",
        "completed",
        "completion_step",
        "transition_reason",
    ]:
        if key in state:
            warmup[key] = state[key]


def build_pre_nested_warmup_state(
    config: Mapping[str, Any],
    *,
    completed: bool,
    completion_step: int | None,
    transition_reason: str | None,
) -> dict[str, Any]:
    training = config.get("training", {})
    warmup = training.get("pre_nested_warmup", {})
    if not isinstance(warmup, Mapping):
        warmup = {}

    return {
        "enabled": bool(warmup.get("enabled", False)),
        "active": bool(warmup.get("active", False)),
        "duration": int(warmup.get("duration", 0)),
        "unit": str(warmup.get("unit", "epochs")),
        "completed": completed,
        "completion_step": completion_step,
        "transition_reason": transition_reason,
    }


def resolve_pre_nested_warmup_target_steps(
    config: Mapping[str, Any],
    train_dataloader,
) -> int:
    warmup = config.get("training", {}).get("pre_nested_warmup", {})
    if not isinstance(warmup, Mapping):
        warmup = {}

    duration = int(warmup.get("duration", 0))
    unit = str(warmup.get("unit", "epochs"))
    if unit == "steps":
        return duration
    if unit == "epochs":
        return duration * max(1, len(train_dataloader))
    raise ConfigError(f"Unsupported pre_nested_warmup unit: {unit}")


def should_run_pre_nested_warmup(
    config: Mapping[str, Any],
    run_state: Mapping[str, Any],
) -> bool:
    warmup = config.get("training", {}).get("pre_nested_warmup", {})
    if not isinstance(warmup, Mapping):
        warmup = {}
    if not bool(warmup.get("enabled", False)):
        return False
    if config.get("run", {}).get("model_family") != "nested":
        return False
    return not bool(run_state.get("warmup_completed", False))


def run_pre_nested_warmup_phase(
    config: dict[str, Any],
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    scheduler,
    device: torch.device,
    heartbeat_writer=None,
    distributed_context=None,
    checkpoint_state: dict[str, Any] | None = None,
    run_state: dict[str, Any] | None = None,
    monitoring_session=None,
    stage_name: str = "training",
) -> list[dict[str, Any]]:
    run_state = run_state if run_state is not None else build_initial_continuation_state(config)
    warmup = config["training"].get("pre_nested_warmup", {})
    if not isinstance(warmup, Mapping):
        warmup = {}

    warmup_state = build_pre_nested_warmup_state(
        config,
        completed=bool(run_state.get("warmup_completed", False)),
        completion_step=(
            int(run_state["warmup_completion_step"])
            if run_state.get("warmup_completion_step") is not None
            else None
        ),
        transition_reason=run_state.get("warmup_transition_reason"),
    )

    if not should_run_pre_nested_warmup(config, run_state):
        update_pre_nested_warmup_state(config, warmup_state)
        return []

    warmup_target_steps = resolve_pre_nested_warmup_target_steps(config, train_dataloader)
    current_step = int(run_state.get("last_completed_step", 0))
    if current_step >= warmup_target_steps:
        warmup_state.update(
            {
                "completed": True,
                "completion_step": current_step,
                "transition_reason": "warmup_duration_reached",
            }
        )
        run_state.update(
            {
                "warmup_completed": True,
                "warmup_completion_step": current_step,
                "warmup_transition_reason": "warmup_duration_reached",
            }
        )
        update_pre_nested_warmup_state(config, warmup_state)
        if checkpoint_state is not None:
            checkpoint_state.update(
                {
                    "warmup_completed": True,
                    "warmup_completion_step": current_step,
                    "warmup_transition_reason": "warmup_duration_reached",
                }
            )
        if continuation_latest_checkpoint_policy(config)["enabled"]:
            maybe_write_latest_checkpoint(
                config,
                model,
                optimizer,
                scheduler,
                heartbeat_writer or NoopHeartbeatWriter(),
                run_state,
                reason="warmup_completion",
                step=current_step,
                distributed_context=distributed_context,
                force=True,
            )
        return []

    warmup_config = copy.deepcopy(config)
    warmup_config["training"]["max_steps"] = min(
        int(config["training"]["max_steps"]),
        warmup_target_steps,
    )
    warmup_config["model"]["granularities"] = [
        config["model"]["granularities"][-1]
    ]
    warmup_config["training"]["granularity_sampling"] = "all"

    warmup_metrics_rows = train_for_steps(
        warmup_config,
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        scheduler,
        device,
        heartbeat_writer=heartbeat_writer,
        distributed_context=distributed_context,
        checkpoint_state=checkpoint_state,
        run_state=run_state,
        monitoring_session=monitoring_session,
        stage_name="warmup",
    )

    current_step = int(run_state.get("last_completed_step", 0))
    warmup_completed = current_step >= warmup_target_steps
    transition_reason = (
        "warmup_duration_reached"
        if warmup_completed
        else "budget_exhausted_before_nested_phase"
    )
    completion_step = current_step if warmup_completed else None
    warmup_state.update(
        {
            "completed": warmup_completed,
            "completion_step": completion_step,
            "transition_reason": transition_reason,
        }
    )
    run_state.update(
        {
            "warmup_completed": warmup_completed,
            "warmup_completion_step": completion_step,
            "warmup_transition_reason": transition_reason,
        }
    )
    update_pre_nested_warmup_state(config, warmup_state)

    if checkpoint_state is not None:
        checkpoint_state.update(
            {
                "warmup_completed": warmup_completed,
                "warmup_completion_step": completion_step,
                "warmup_transition_reason": transition_reason,
            }
        )

    if continuation_latest_checkpoint_policy(config)["enabled"]:
        maybe_write_latest_checkpoint(
            config,
            model,
            optimizer,
            scheduler,
            heartbeat_writer or NoopHeartbeatWriter(),
            run_state,
            reason="warmup_completion",
            step=current_step,
            distributed_context=distributed_context,
            force=True,
        )

    return warmup_metrics_rows
