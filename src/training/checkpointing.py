"""Config-driven training flow for MatFormer reproduction runs."""

from __future__ import annotations

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*args, **kwargs):
        return None

load_dotenv()

import copy
from contextlib import contextmanager
import random
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from src.training.data import (
    build_language_model_dataloader,
    load_and_tokenize_dataset,
    split_train_eval_dataset,
)
from src.training.distributed import (
    broadcast_object,
    destroy_distributed_process_group,
    prepare_distributed_context,
    should_write_shared_artifact,
    sum_int,
    wrap_model_for_distributed,
)
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


class NoopHeartbeatWriter:
    path = None

    def stage_start(self, stage: str, **fields: Any):
        return None

    def stage_complete(self, stage: str, **fields: Any):
        return None

    def heartbeat(self, stage: str, **fields: Any):
        return None
def continuation_latest_checkpoint_policy(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    run = config.get("run", {})
    continuation = run.get("continuation", {})
    if not isinstance(continuation, Mapping):
        continuation = {}

    enabled = bool(continuation.get("enabled", False))
    interval_steps = continuation.get("latest_checkpoint_save_interval_steps", 1)
    if interval_steps is None:
        interval_steps = 0
    interval_steps = int(interval_steps)
    if interval_steps < 0:
        raise ConfigError(
            "run.continuation.latest_checkpoint_save_interval_steps must be non-negative"
        )

    return {
        "enabled": enabled,
        "save_interval_steps": interval_steps,
        "save_on_validation": bool(
            continuation.get("latest_checkpoint_save_on_validation", False)
        ),
        "save_on_completion": bool(
            continuation.get("latest_checkpoint_save_on_completion", True)
        ),
    }


def should_save_latest_checkpoint(
    config: Mapping[str, Any],
    step: int,
    reason: str,
) -> bool:
    policy = continuation_latest_checkpoint_policy(config)
    if not policy["enabled"]:
        return False

    if reason == "validation":
        return policy["save_on_validation"]
    if reason == "completion":
        return policy["save_on_completion"]
    if reason == "failure":
        return True

    interval_steps = policy["save_interval_steps"]
    return interval_steps > 0 and step > 0 and step % interval_steps == 0


def maybe_write_latest_checkpoint(
    config: dict[str, Any],
    model,
    optimizer,
    scheduler,
    heartbeat_writer,
    run_state: dict[str, Any],
    reason: str,
    step: int,
    distributed_context=None,
    force: bool = False,
) -> None:
    if not force and not should_save_latest_checkpoint(config, step, reason):
        return
    if not force and int(run_state.get("latest_checkpoint_step", 0)) == int(step):
        return

    latest_checkpoint_path = Path(
        run_state.get("latest_checkpoint_path")
        or Path(config["run"]["output_dir"]) / "checkpoints" / "latest.pt"
    )
    checkpoint_fields = {
        "checkpoint_status": "latest",
        "checkpoint_metric": None,
        "checkpoint_metric_value": None,
        "checkpoint_selection_step": None,
        "checkpoint_unavailable_reason": None,
    }

    with heartbeat_stage(
        heartbeat_writer,
        "checkpointing",
        checkpoint_status="latest",
        checkpoint_reason=reason,
    ):
        save_model_checkpoint(
            config,
            model,
            optimizer,
            scheduler,
            latest_checkpoint_path,
            checkpoint_fields,
            run_state,
            distributed_context=distributed_context,
        )

    run_state["latest_checkpoint_path"] = str(latest_checkpoint_path)
    run_state["latest_checkpoint_step"] = step


def write_checkpoint_if_needed(
    config: dict[str, Any],
    model,
    optimizer,
    scheduler,
    metrics_rows: list[dict[str, Any]],
    heartbeat_writer,
    run_state: dict[str, Any],
    distributed_context=None,
) -> dict[str, Any]:
    if should_write_shared_artifact(distributed_context):
        checkpoint_fields = build_checkpoint_summary_fields(config, metrics_rows)
        checkpoint_path = checkpoint_fields.get("best_checkpoint_path")
        if checkpoint_path is None:
            checkpoint_path = checkpoint_fields.get("final_checkpoint_path")

        should_save = False
        if checkpoint_path is not None:
            output_path = Path(str(checkpoint_path))
            should_save = not output_path.exists()

        payload = {
            "checkpoint_fields": checkpoint_fields,
            "checkpoint_path": checkpoint_path,
            "should_save": should_save,
        }
    else:
        payload = None

    payload = broadcast_object(payload, distributed_context)
    if payload is None:
        return build_checkpoint_summary_fields(config, metrics_rows)

    checkpoint_fields = payload["checkpoint_fields"]
    checkpoint_path = payload["checkpoint_path"]
    if checkpoint_path is None or not payload["should_save"]:
        return checkpoint_fields

    output_path = Path(str(checkpoint_path))

    with heartbeat_stage(
        heartbeat_writer,
        "checkpointing",
        checkpoint_status=checkpoint_fields["checkpoint_status"],
    ):
        save_model_checkpoint(
            config,
            model,
            optimizer,
            scheduler,
            output_path,
            checkpoint_fields,
            run_state,
            distributed_context=distributed_context,
        )

    return checkpoint_fields


def maybe_write_best_eval_checkpoint(
    config: dict[str, Any],
    model,
    validation_results: list[dict[str, Any]],
    step: int,
    heartbeat_writer,
    checkpoint_state: dict[str, Any],
    run_state: dict[str, Any],
    distributed_context=None,
) -> None:
    if not config.get("outputs", {}).get("save_checkpoints", False):
        return
    if not config.get("evaluation", {}).get("validation", False):
        return

    if should_write_shared_artifact(distributed_context):
        payload = build_best_eval_checkpoint_payload(
            config,
            validation_results,
            step,
            checkpoint_state,
        )
    else:
        payload = None

    payload = broadcast_object(payload, distributed_context)
    if payload is None or not payload["should_save"]:
        return

    checkpoint_fields = payload["checkpoint_fields"]
    checkpoint_path = Path(str(checkpoint_fields["best_checkpoint_path"]))

    with heartbeat_stage(
        heartbeat_writer,
        "checkpointing",
        checkpoint_status=checkpoint_fields["checkpoint_status"],
    ):
        save_model_checkpoint(
            config,
            model,
            None,
            None,
            checkpoint_path,
            checkpoint_fields,
            run_state,
            distributed_context=distributed_context,
        )

    checkpoint_state.update(checkpoint_fields)


def build_best_eval_checkpoint_payload(
    config: dict[str, Any],
    validation_results: list[dict[str, Any]],
    step: int,
    checkpoint_state: dict[str, Any],
) -> dict[str, Any]:
    metric_name, metric_value = best_validation_metric_value(validation_results)
    if metric_name is None or metric_value is None:
        return {"should_save": False, "checkpoint_fields": None}

    previous_metric_value = checkpoint_state.get("checkpoint_metric_value")
    if previous_metric_value is not None and metric_value >= previous_metric_value:
        return {"should_save": False, "checkpoint_fields": None}

    metric_field = "loss" if metric_name == "validation_loss" else "perplexity"
    best_result = min(
        (
            result
            for result in validation_results
            if result.get(metric_field) is not None
        ),
        key=lambda result: float(result[metric_field]),
    )
    metric_row = {
        "split": "validation",
        "step": step,
        "granularity": best_result.get("granularity"),
        metric_field: metric_value,
    }
    checkpoint_fields = build_checkpoint_summary_fields(
        config,
        [metric_row],
        validation_enabled=True,
        save_checkpoints=True,
    )
    return {"should_save": True, "checkpoint_fields": checkpoint_fields}


def save_model_checkpoint(
    config: dict[str, Any],
    model,
    optimizer,
    scheduler,
    output_path: Path,
    checkpoint_fields: dict[str, Any],
    run_state: dict[str, Any],
    distributed_context=None,
) -> None:
    if not should_write_shared_artifact(distributed_context):
        return

    model_state_dict, optimizer_state_dict = checkpoint_state_dicts(
        model,
        optimizer,
        distributed_context,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "run_id": config["run"]["run_id"],
            "checkpoint_status": checkpoint_fields["checkpoint_status"],
            "checkpoint_metric": checkpoint_fields["checkpoint_metric"],
            "checkpoint_metric_value": checkpoint_fields[
                "checkpoint_metric_value"
            ],
            "checkpoint_selection_step": checkpoint_fields[
                "checkpoint_selection_step"
            ],
            "step": run_state.get("step", run_state.get("last_completed_step", 0)),
            "epoch": run_state.get("epoch", 0),
            "batch_index": run_state.get("batch_index", 0),
            "tokens_seen": run_state.get("tokens_seen", 0),
            "content_tokens_seen": run_state.get("content_tokens_seen", 0),
            "resume_count": run_state.get("resume_count", 0),
            "warmup_completed": run_state.get("warmup_completed", False),
            "warmup_completion_step": run_state.get("warmup_completion_step"),
            "warmup_transition_reason": run_state.get("warmup_transition_reason"),
            "resolved_run_mode": run_state.get("resolved_run_mode"),
            "resolved_sampling_mode": run_state.get("resolved_sampling_mode"),
            "granularity_pattern_provenance": run_state.get(
                "granularity_pattern_provenance"
            ),
            "adaptive_sampler_state": run_state.get("adaptive_sampler_state"),
            "adaptive_sampler_previous_loss": run_state.get(
                "adaptive_sampler_previous_loss"
            ),
            "adaptive_sampler_previous_pattern": run_state.get(
                "adaptive_sampler_previous_pattern"
            ),
            "adaptive_reward_summary": run_state.get("adaptive_reward_summary"),
            "adaptive_correction_penalty_summary": run_state.get(
                "adaptive_correction_penalty_summary"
            ),
            "adaptive_sampler_strategy": run_state.get(
                "adaptive_sampler_strategy"
            ),
            "adaptive_sampler_exploration_scale": run_state.get(
                "adaptive_sampler_exploration_scale"
            ),
            "adaptive_sampler_decay_rate": run_state.get(
                "adaptive_sampler_decay_rate"
            ),
            "adaptive_sampler_reward_penalty_weight": run_state.get(
                "adaptive_sampler_reward_penalty_weight"
            ),
            "latest_checkpoint_path": run_state.get("latest_checkpoint_path"),
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        },
        output_path,
    )


def checkpoint_state_dicts(
    model,
    optimizer=None,
    distributed_context=None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if (
        distributed_context is None
        or not distributed_context.enabled
        or distributed_context.strategy != "fsdp"
    ):
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict() if optimizer is not None else None
        return model_state_dict, optimizer_state_dict

    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    model_state_dict, optimizer_state_dict = get_state_dict(
        model,
        optimizer if optimizer is not None else [],
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    return model_state_dict, optimizer_state_dict


def checkpoint_state_dict(model, distributed_context=None) -> dict[str, Any]:
    model_state_dict, _ = checkpoint_state_dicts(model, distributed_context=distributed_context)
    return model_state_dict


def build_initial_continuation_state(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["run"]["output_dir"])
    run = config.get("run", {})
    model = config.get("model", {})
    if not isinstance(run, Mapping):
        run = {}
    if not isinstance(model, Mapping):
        model = {}
    return {
        "status": "fresh",
        "latest_checkpoint_path": None,
        "latest_checkpoint_step": 0,
        "last_completed_step": 0,
        "resume_count": 0,
        "tokens_seen": 0,
        "content_tokens_seen": 0,
        "step": 0,
        "epoch": 0,
        "batch_index": 0,
        "warmup_completed": False,
        "warmup_completion_step": None,
        "warmup_transition_reason": None,
        "output_dir": str(output_dir),
        "resolved_run_mode": str(
            run.get(
                "resolved_run_mode",
                resolve_sampling_mode_from_config_sections(
                    run,
                    config.get("training", {}),
                ),
            )
        ),
        "resolved_sampling_mode": str(
            model.get(
                "resolved_sampling_mode",
                model.get("granularity_sampling_mode", "global"),
            )
        ),
        "granularity_pattern_provenance": copy.deepcopy(
            model.get("granularity_pattern_provenance")
        ),
        "adaptive_sampler_state": _build_initial_adaptive_sampler_state(config),
        "adaptive_sampler_previous_loss": None,
        "adaptive_sampler_previous_pattern": None,
        "adaptive_reward_summary": None,
        "adaptive_correction_penalty_summary": None,
        "adaptive_sampler_strategy": model.get(
            "adaptive_sampler_strategy",
            None,
        ),
        "adaptive_sampler_exploration_scale": model.get(
            "adaptive_sampler_exploration_scale",
            None,
        ),
        "adaptive_sampler_decay_rate": model.get(
            "adaptive_sampler_decay_rate",
            None,
        ),
        "adaptive_sampler_reward_penalty_weight": model.get(
            "adaptive_sampler_reward_penalty_weight",
            None,
        ),
    }


def update_run_continuation_state(
    config: dict[str, Any],
    state: Mapping[str, Any],
) -> None:
    continuation = config["run"].setdefault("continuation", {})
    if not isinstance(continuation, dict):
        raise ConfigError("run.continuation must be a mapping when provided")
    for key in [
        "status",
        "latest_checkpoint_path",
        "latest_checkpoint_step",
        "last_completed_step",
        "resume_count",
        "tokens_seen",
        "content_tokens_seen",
        "step",
        "epoch",
        "batch_index",
    ]:
        if key in state and state[key] is not None:
            continuation[key] = state[key]


def _build_initial_adaptive_sampler_state(
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        model = {}
    if model.get("granularity_sampling_mode") != "adaptive_per_block":
        return None

    state = build_adaptive_sampler_state(
        strategy_name=str(model.get("adaptive_sampler_strategy", "thompson")),
        exploration_scale=float(model.get("adaptive_sampler_exploration_scale", 1.0)),
        decay_rate=float(model.get("adaptive_sampler_decay_rate", 0.0)),
    )
    normalized_state = normalize_adaptive_sampler_state(
        state,
        block_count=int(model["num_layers"]),
        granularities=model.get("granularities"),
    )
    return summarize_adaptive_sampler_state(normalized_state)


def _populate_adaptive_sampler_state_metadata(
    state: dict[str, Any],
    config: Mapping[str, Any],
) -> None:
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        model = {}

    if model.get("granularity_sampling_mode") == "adaptive_per_block":
        state["adaptive_sampler_state"] = _build_initial_adaptive_sampler_state(
            config
        )
    else:
        state.setdefault("adaptive_sampler_state", None)
    state["adaptive_sampler_previous_loss"] = None
    state["adaptive_sampler_previous_pattern"] = None
    state["adaptive_reward_summary"] = None
    state["adaptive_correction_penalty_summary"] = None
    state["adaptive_sampler_strategy"] = model.get(
        "adaptive_sampler_strategy"
    )
    state["adaptive_sampler_exploration_scale"] = model.get(
        "adaptive_sampler_exploration_scale"
    )
    state["adaptive_sampler_decay_rate"] = model.get(
        "adaptive_sampler_decay_rate"
    )
    state["adaptive_sampler_reward_penalty_weight"] = model.get(
        "adaptive_sampler_reward_penalty_weight"
    )


def _validate_loaded_adaptive_sampler_state(
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    checkpoint_path: Path,
) -> None:
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        model = {}
    if model.get("granularity_sampling_mode") != "adaptive_per_block":
        return

    expected_strategy = str(model.get("adaptive_sampler_strategy", "thompson"))
    expected_exploration_scale = float(
        model.get("adaptive_sampler_exploration_scale", 1.0)
    )
    expected_decay_rate = float(model.get("adaptive_sampler_decay_rate", 0.0))
    expected_reward_penalty_weight = float(
        model.get("adaptive_sampler_reward_penalty_weight", 1.0)
    )

    if state.get("adaptive_sampler_strategy") not in (None, expected_strategy):
        raise ConfigError(
            "Checkpoint adaptive sampler strategy does not match the current config "
            f"for {checkpoint_path}"
        )
    if state.get("adaptive_sampler_exploration_scale") not in (
        None,
        expected_exploration_scale,
    ):
        raise ConfigError(
            "Checkpoint adaptive sampler exploration scale does not match the current config "
            f"for {checkpoint_path}"
        )
    if state.get("adaptive_sampler_decay_rate") not in (None, expected_decay_rate):
        raise ConfigError(
            "Checkpoint adaptive sampler decay rate does not match the current config "
            f"for {checkpoint_path}"
        )
    if state.get("adaptive_sampler_reward_penalty_weight") not in (
        None,
        expected_reward_penalty_weight,
    ):
        raise ConfigError(
            "Checkpoint adaptive sampler reward penalty weight does not match the current config "
            f"for {checkpoint_path}"
        )

    adaptive_state = coerce_adaptive_sampler_state(state.get("adaptive_sampler_state"))
    if adaptive_state is None:
        raise ConfigError(
            "Checkpoint is missing adaptive sampler state required for resume "
            f"at {checkpoint_path}"
        )

    normalize_adaptive_sampler_state(
        adaptive_state,
        block_count=int(model["num_layers"]),
        granularities=model.get("granularities"),
    )


def _prepare_adaptive_sampler_runtime_state(
    config: Mapping[str, Any],
    run_state: dict[str, Any],
) -> AdaptiveSamplerState | None:
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        model = {}
    if model.get("granularity_sampling_mode") != "adaptive_per_block":
        run_state.pop("adaptive_sampler_state", None)
        return None

    adaptive_state = coerce_adaptive_sampler_state(
        run_state.get("adaptive_sampler_state")
    )
    if adaptive_state is None:
        adaptive_state = build_adaptive_sampler_state(
            strategy_name=str(model.get("adaptive_sampler_strategy", "thompson")),
            exploration_scale=float(model.get("adaptive_sampler_exploration_scale", 1.0)),
            decay_rate=float(model.get("adaptive_sampler_decay_rate", 0.0)),
        )

    normalized_state = normalize_adaptive_sampler_state(
        adaptive_state,
        block_count=int(model["num_layers"]),
        granularities=model.get("granularities"),
    )
    run_state["adaptive_sampler_state"] = summarize_adaptive_sampler_state(
        normalized_state
    )
    run_state["adaptive_sampler_strategy"] = normalized_state.strategy_name
    run_state["adaptive_sampler_exploration_scale"] = (
        normalized_state.exploration_scale
    )
    run_state["adaptive_sampler_decay_rate"] = normalized_state.decay_rate
    run_state["adaptive_sampler_reward_penalty_weight"] = float(
        model.get("adaptive_sampler_reward_penalty_weight", 1.0)
    )
    if run_state.get("adaptive_sampler_previous_pattern") is not None:
        run_state["adaptive_sampler_previous_pattern"] = [
            str(granularity)
            for granularity in run_state["adaptive_sampler_previous_pattern"]
        ]
    return normalized_state


def _pattern_change_penalty(
    previous_pattern: Sequence[str] | None,
    current_pattern: Sequence[str],
) -> float:
    if not previous_pattern:
        return 0.0

    previous = [str(granularity) for granularity in previous_pattern]
    current = [str(granularity) for granularity in current_pattern]
    if not current:
        return 0.0

    difference_count = sum(
        1 for previous_granularity, current_granularity in zip(previous, current)
        if previous_granularity != current_granularity
    )
    difference_count += abs(len(previous) - len(current))
    return difference_count / max(len(current), len(previous), 1)


def _update_adaptive_sampler_runtime_state(
    config: Mapping[str, Any],
    run_state: dict[str, Any],
    adaptive_sampler_state: AdaptiveSamplerState,
    *,
    phase: str,
    latest_loss: float,
    selected_layer_granularities: Sequence[str],
    step: int,
    epoch: int,
) -> None:
    previous_loss = run_state.get("adaptive_sampler_previous_loss")
    previous_pattern = run_state.get("adaptive_sampler_previous_pattern")
    reward_penalty_weight = float(
        run_state.get(
            "adaptive_sampler_reward_penalty_weight",
            config.get("model", {}).get("adaptive_sampler_reward_penalty_weight", 1.0),
        )
    )
    correction_penalty = _pattern_change_penalty(
        previous_pattern,
        selected_layer_granularities,
    )
    reward_record = build_adaptive_reward_record(
        previous_loss=float(previous_loss) if previous_loss is not None else None,
        current_loss=latest_loss,
        correction_penalty=correction_penalty,
        reward_penalty_weight=reward_penalty_weight,
        phase=phase,
        step=step,
        epoch=epoch,
    )
    if previous_loss is not None:
        adaptive_sampler_state = update_adaptive_sampler_state(
            adaptive_sampler_state,
            reward_record,
            sampled_pattern=list(selected_layer_granularities),
        )

    run_state["adaptive_reward_summary"] = dict(reward_record)
    run_state["adaptive_correction_penalty_summary"] = {
        "correction_penalty": correction_penalty,
        "reward_penalty_weight": reward_penalty_weight,
        "normalized_correction_penalty": reward_record[
            "normalized_correction_penalty"
        ],
    }
    run_state["adaptive_sampler_previous_loss"] = latest_loss
    run_state["adaptive_sampler_previous_pattern"] = [
        str(granularity) for granularity in selected_layer_granularities
    ]
    run_state["adaptive_sampler_state"] = summarize_adaptive_sampler_state(
        adaptive_sampler_state
    )


def load_run_continuation_state(
    config: dict[str, Any],
    model,
    optimizer,
    scheduler,
    distributed_context=None,
) -> dict[str, Any]:
    checkpoint_path = Path(config["run"]["output_dir"]) / "checkpoints" / "latest.pt"
    return load_checkpoint_state(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        config=config,
        fallback_tokens_per_step=int(config["training"]["expected_tokens_per_step"]),
        distributed_context=distributed_context,
        output_dir=config["run"]["output_dir"],
        run_id=config["run"]["run_id"],
    )


def load_checkpoint_state(
    checkpoint_path: str | Path,
    model,
    optimizer,
    scheduler,
    config: Mapping[str, Any] | None = None,
    fallback_tokens_per_step: int | None = None,
    distributed_context=None,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        state = {
            "status": "fresh",
            "latest_checkpoint_path": None,
            "latest_checkpoint_step": 0,
            "last_completed_step": 0,
            "resume_count": 0,
            "tokens_seen": 0,
            "content_tokens_seen": 0,
            "step": 0,
            "epoch": 0,
            "batch_index": 0,
            "warmup_completed": False,
            "warmup_completion_step": None,
            "warmup_transition_reason": None,
        }
        if output_dir is not None:
            state["output_dir"] = str(output_dir)
        if run_id is not None:
            state["run_id"] = str(run_id)
        if config is not None:
            _populate_adaptive_sampler_state_metadata(state, config)
        return state

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state_dict = checkpoint.get("model_state_dict")
    if model_state_dict is None:
        raise ConfigError(f"Checkpoint missing model_state_dict: {checkpoint_path}")

    optimizer_state_dict = checkpoint.get("optimizer_state_dict")
    scheduler_state_dict = checkpoint.get("scheduler_state_dict")
    load_model_and_optimizer_state(
        model,
        optimizer,
        model_state_dict,
        optimizer_state_dict,
        distributed_context=distributed_context,
    )
    if scheduler is not None and scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)

    last_completed_step = int(
        checkpoint.get("step", checkpoint.get("last_completed_step", 0))
    )
    tokens_seen = int(
        checkpoint.get(
            "tokens_seen",
            fallback_tokens_per_step * last_completed_step
            if fallback_tokens_per_step is not None
            else 0,
        )
    )
    content_tokens_seen = int(checkpoint.get("content_tokens_seen", 0))
    resume_count = int(checkpoint.get("resume_count", 0)) + 1
    state = {
        "status": "resumed",
        "latest_checkpoint_path": str(checkpoint_path),
        "latest_checkpoint_step": last_completed_step,
        "last_completed_step": last_completed_step,
        "resume_count": resume_count,
        "tokens_seen": tokens_seen,
        "content_tokens_seen": content_tokens_seen,
        "step": last_completed_step,
        "epoch": int(checkpoint.get("epoch", 0)),
        "batch_index": int(checkpoint.get("batch_index", 0)),
        "warmup_completed": bool(checkpoint.get("warmup_completed", False)),
        "warmup_completion_step": checkpoint.get("warmup_completion_step"),
        "warmup_transition_reason": checkpoint.get("warmup_transition_reason"),
        "resolved_run_mode": checkpoint.get("resolved_run_mode"),
        "resolved_sampling_mode": checkpoint.get("resolved_sampling_mode"),
        "granularity_pattern_provenance": checkpoint.get(
            "granularity_pattern_provenance"
        ),
        "adaptive_sampler_state": checkpoint.get("adaptive_sampler_state"),
        "adaptive_sampler_previous_loss": checkpoint.get(
            "adaptive_sampler_previous_loss"
        ),
        "adaptive_sampler_previous_pattern": checkpoint.get(
            "adaptive_sampler_previous_pattern"
        ),
        "adaptive_reward_summary": checkpoint.get("adaptive_reward_summary"),
        "adaptive_correction_penalty_summary": checkpoint.get(
            "adaptive_correction_penalty_summary"
        ),
        "adaptive_sampler_strategy": checkpoint.get("adaptive_sampler_strategy"),
        "adaptive_sampler_exploration_scale": checkpoint.get(
            "adaptive_sampler_exploration_scale"
        ),
        "adaptive_sampler_decay_rate": checkpoint.get("adaptive_sampler_decay_rate"),
        "adaptive_sampler_reward_penalty_weight": checkpoint.get(
            "adaptive_sampler_reward_penalty_weight"
        ),
    }
    if output_dir is not None:
        state["output_dir"] = str(output_dir)
    if run_id is not None:
        state["run_id"] = str(run_id)
    if config is not None:
        _validate_loaded_adaptive_sampler_state(state, config, checkpoint_path)
    return state
