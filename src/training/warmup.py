"""Config-driven training flow for MatFormer reproduction runs."""

from __future__ import annotations

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*args, **kwargs):
        return None

load_dotenv()

import copy
from collections import Counter
from typing import Any, Mapping

import torch

from src.training.checkpointing import (
    build_initial_continuation_state,
    continuation_latest_checkpoint_policy,
    maybe_write_latest_checkpoint,
)
from src.training.monitoring import NoopHeartbeatWriter
from src.training.optimizer_state import PerGranularityOptimizerCollection
from src.utils.config import (
    ConfigError,
)


def update_pre_nested_warmup_state(
    config: dict[str, Any],
    state: Mapping[str, Any],
) -> None:
    training = config.setdefault("training", {})
    warmup = training.setdefault("pre_nested_warmup", {})
    if not isinstance(warmup, dict):
        raise ConfigError("training.pre_nested_warmup must be a mapping when provided")

    keys = [
        "enabled",
        "active",
        "duration",
        "unit",
        "policy",
        "completed",
        "completion_step",
        "transition_reason",
    ]
    if state.get("policy") == "balanced_global":
        keys.extend(
            [
                "action_interval_steps",
                "schedule_seed",
                "schedule_hash",
                "schedule",
                "passes",
                "current_window_index",
                "current_window_offset",
                "completed_steps",
                "per_granularity_counts",
                "controller_start_step",
                "schedule_initialized",
            ]
        )
    for key in keys:
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

    schedule = warmup.get("schedule")
    schedule = list(schedule) if isinstance(schedule, list) else None
    action_interval = warmup.get("action_interval_steps")
    action_interval = int(action_interval) if action_interval is not None else None
    completed_steps = int(warmup.get("completed_steps", 0))
    current_window_index = int(warmup.get("current_window_index", 0))
    current_window_offset = int(warmup.get("current_window_offset", 0))
    labels = [str(label) for label in config.get("model", {}).get("granularities", [])]

    return {
        "enabled": bool(warmup.get("enabled", False)),
        "active": bool(warmup.get("active", False)),
        "duration": int(warmup.get("duration", 0)),
        "unit": str(warmup.get("unit", "epochs")),
        "policy": str(warmup.get("policy", "full_only")),
        "action_interval_steps": action_interval,
        "schedule_seed": warmup.get("schedule_seed"),
        "schedule_hash": warmup.get("schedule_hash"),
        "schedule": schedule,
        "passes": warmup.get("passes"),
        "current_window_index": current_window_index,
        "current_window_offset": current_window_offset,
        "completed_steps": completed_steps,
        "per_granularity_counts": dict(
            warmup.get(
                "per_granularity_counts",
                {label: 0 for label in labels},
            )
        ),
        "controller_start_step": warmup.get("controller_start_step"),
        "schedule_initialized": bool(warmup.get("schedule_initialized", False)),
        "completed": completed,
        "completion_step": completion_step,
        "transition_reason": transition_reason,
    }


def validate_pre_nested_warmup_resume_state(
    config: Mapping[str, Any],
    state: Mapping[str, Any] | None,
    *,
    last_completed_step: int,
) -> dict[str, Any] | None:
    """Validate exact balanced-warmup checkpoint compatibility."""

    warmup = config.get("training", {}).get("pre_nested_warmup", {})
    if not isinstance(warmup, Mapping) or warmup.get("policy", "full_only") != "balanced_global":
        return copy.deepcopy(dict(state)) if isinstance(state, Mapping) else None
    if not bool(warmup.get("enabled", False)):
        return copy.deepcopy(dict(state)) if isinstance(state, Mapping) else None
    if state is None:
        if int(last_completed_step) > 0 and int(last_completed_step) < int(warmup["duration"]):
            raise ConfigError(
                "Balanced pre-nested warmup checkpoint is missing exact schedule state"
            )
        return None
    if not isinstance(state, Mapping):
        raise ConfigError("Balanced pre-nested warmup checkpoint state is malformed")

    expected_identity = {
        "policy": "balanced_global",
        "duration": int(warmup["duration"]),
        "unit": "steps",
        "action_interval_steps": int(warmup["action_interval_steps"]),
        "schedule_seed": int(warmup["schedule_seed"]),
        "schedule_hash": str(warmup["schedule_hash"]),
        "schedule": list(warmup["schedule"]),
        "controller_start_step": int(warmup["controller_start_step"]),
    }
    for key, expected in expected_identity.items():
        if state.get(key) != expected:
            raise ConfigError(
                f"Balanced pre-nested warmup checkpoint {key} does not match config"
            )

    completed_steps = int(state.get("completed_steps", -1))
    expected_completed_steps = min(
        int(last_completed_step),
        int(expected_identity["duration"]),
    )
    if completed_steps != expected_completed_steps:
        raise ConfigError(
            "Balanced pre-nested warmup checkpoint completed_steps does not match "
            "the checkpoint training step"
        )
    interval = int(expected_identity["action_interval_steps"])
    expected_window_index = completed_steps // interval
    expected_offset = completed_steps % interval
    if int(state.get("current_window_index", -1)) != expected_window_index:
        raise ConfigError("Balanced pre-nested warmup checkpoint window index is invalid")
    if int(state.get("current_window_offset", -1)) != expected_offset:
        raise ConfigError("Balanced pre-nested warmup checkpoint window offset is invalid")
    completed_schedule = expected_identity["schedule"][:expected_window_index]
    expected_counts = {label: 0 for label in config["model"]["granularities"]}
    expected_counts.update(Counter(completed_schedule))
    if dict(state.get("per_granularity_counts", {})) != expected_counts:
        raise ConfigError("Balanced pre-nested warmup checkpoint action counts are invalid")
    if completed_steps > 0 and state.get("schedule_initialized") is not True:
        raise ConfigError(
            "Balanced pre-nested warmup checkpoint did not initialize its schedule"
        )
    if bool(state.get("completed", False)):
        if completed_steps != int(expected_identity["duration"]):
            raise ConfigError(
                "Balanced pre-nested warmup checkpoint completed too early"
            )
        if state.get("completion_step") != int(expected_identity["duration"]):
            raise ConfigError(
                "Balanced pre-nested warmup checkpoint completion step is invalid"
            )
    elif completed_steps < int(expected_identity["duration"]) and state.get(
        "completion_step"
    ) is not None:
        raise ConfigError(
            "Incomplete balanced pre-nested warmup checkpoint has a completion step"
        )
    return copy.deepcopy(dict(state))


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


def _validate_per_granularity_warmup_ownership(
    config: Mapping[str, Any],
    optimizer,
) -> None:
    """Reject any warmup plan that cannot name one width owner per step."""

    if not isinstance(optimizer, PerGranularityOptimizerCollection):
        return
    warmup = config.get("training", {}).get("pre_nested_warmup", {})
    if not isinstance(warmup, Mapping) or not bool(warmup.get("enabled", False)):
        return
    policy = str(warmup.get("policy", "full_only"))
    if policy == "full_only":
        planned_owners: list[Any] = [config["model"]["granularities"][-1]]
    elif policy == "balanced_global":
        schedule = warmup.get("schedule")
        if not isinstance(schedule, list) or not schedule:
            raise ConfigError(
                "Per-granularity balanced warmup requires a non-empty owner schedule"
            )
        planned_owners = schedule
    else:
        raise ConfigError(
            "Per-granularity warmup requires full_only or balanced_global ownership"
        )

    for owner in planned_owners:
        if isinstance(owner, (list, tuple, set, Mapping)):
            raise ConfigError(
                "Per-granularity warmup steps must apply exactly one global width"
            )
        optimizer.optimizer_for(str(owner))


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
    metrics_journal=None,
    warmup_event_callback=None,
    successful_step_callback=None,
) -> list[dict[str, Any]]:
    run_state = run_state if run_state is not None else build_initial_continuation_state(config)
    warmup = config["training"].get("pre_nested_warmup", {})
    if not isinstance(warmup, Mapping):
        warmup = {}
    _validate_per_granularity_warmup_ownership(config, optimizer)

    saved_warmup_state = run_state.get("pre_nested_warmup_state")
    if isinstance(saved_warmup_state, Mapping):
        warmup_state = copy.deepcopy(dict(saved_warmup_state))
    else:
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
        policy = str(warmup.get("policy", "full_only"))
        if policy == "balanced_global":
            schedule = list(warmup["schedule"])
            interval = int(warmup["action_interval_steps"])
            counts = {str(label): 0 for label in config["model"]["granularities"]}
            counts.update(Counter(schedule))
            warmup_state.update(
                completed_steps=current_step,
                current_window_index=len(schedule),
                current_window_offset=0,
                per_granularity_counts=counts,
            )
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
        run_state["pre_nested_warmup_state"] = copy.deepcopy(warmup_state)
        update_pre_nested_warmup_state(config, warmup_state)
        if policy == "balanced_global" and warmup_event_callback is not None:
            last_window_index = len(schedule) - 1
            warmup_event_callback(
                {
                    "schema_version": 1,
                    "event_type": "warmup_completed",
                    "phase": "warmup",
                    "schedule_hash": warmup_state["schedule_hash"],
                    "schedule_seed": warmup_state["schedule_seed"],
                    "action_interval_steps": interval,
                    "requested_warmup_steps": warmup_target_steps,
                    "completed_warmup_steps": current_step,
                    "per_granularity_counts": counts,
                    "controller_start_step": warmup_state["controller_start_step"],
                    "action": _warmup_action(config, schedule[last_window_index]),
                    "warmup_window_index": last_window_index,
                    "window_index": last_window_index,
                    "boundary_step": current_step,
                    "boundary_step_start": last_window_index * interval,
                    "boundary_step_end": current_step,
                    "completed_optimizer_steps": interval,
                    "posterior_updated": False,
                }
            )
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

    policy = str(warmup.get("policy", "full_only"))
    schedule = list(warmup.get("schedule") or [])
    interval = warmup.get("action_interval_steps")
    if policy == "balanced_global":
        validate_pre_nested_warmup_resume_state(
            config,
            warmup_state,
            last_completed_step=current_step,
        )
        if not bool(warmup_state.get("schedule_initialized", False)):
            warmup_state["schedule_initialized"] = True
            run_state["pre_nested_warmup_state"] = copy.deepcopy(warmup_state)
            update_pre_nested_warmup_state(config, warmup_state)
            if warmup_event_callback is not None:
                warmup_event_callback(
                    {
                        "schema_version": 1,
                        "event_type": "warmup_schedule_initialized",
                        "phase": "warmup",
                        "schedule_hash": warmup_state["schedule_hash"],
                        "schedule_seed": warmup_state["schedule_seed"],
                        "schedule": list(schedule),
                        "passes": warmup_state["passes"],
                        "action_interval_steps": int(interval),
                        "requested_warmup_steps": warmup_target_steps,
                        "action": _warmup_action(config, schedule[0]),
                        "warmup_window_index": 0,
                        "window_index": 0,
                        "boundary_step": current_step,
                        "boundary_step_start": 0,
                        "boundary_step_end": int(interval),
                        "completed_optimizer_steps": current_step,
                        "posterior_updated": False,
                    }
                )

    warmup_config = copy.deepcopy(config)
    warmup_config["training"]["max_steps"] = min(
        int(config["training"]["max_steps"]),
        warmup_target_steps,
    )

    def forced_action(step: int) -> str:
        if policy == "full_only":
            return str(config["model"]["granularities"][-1])
        window_index = (int(step) - 1) // int(interval)
        if window_index < 0 or window_index >= len(schedule):
            raise ConfigError("Balanced pre-nested warmup schedule was exhausted")
        return str(schedule[window_index])

    def record_successful_warmup_step(*, step: int, tokens_seen: int) -> None:
        if policy == "balanced_global":
            completed_steps = int(step)
            completed_windows = completed_steps // int(interval)
            window_offset = completed_steps % int(interval)
            counts = {str(label): 0 for label in config["model"]["granularities"]}
            counts.update(Counter(schedule[:completed_windows]))
            warmup_state.update(
                completed_steps=completed_steps,
                current_window_index=completed_windows,
                current_window_offset=window_offset,
                per_granularity_counts=counts,
            )
            run_state["pre_nested_warmup_state"] = copy.deepcopy(warmup_state)
            update_pre_nested_warmup_state(config, warmup_state)
            if window_offset == 0 and warmup_event_callback is not None:
                completed_window_index = completed_windows - 1
                warmup_event_callback(
                    {
                        "schema_version": 1,
                        "event_type": "warmup_window_completed",
                        "phase": "warmup",
                        "schedule_hash": warmup_state["schedule_hash"],
                        "schedule_seed": warmup_state["schedule_seed"],
                        "action_interval_steps": int(interval),
                        "requested_warmup_steps": warmup_target_steps,
                        "action": _warmup_action(
                            config,
                            schedule[completed_window_index],
                        ),
                        "warmup_window_index": completed_window_index,
                        "window_index": completed_window_index,
                        "boundary_step": completed_steps,
                        "boundary_step_start": completed_window_index * int(interval),
                        "boundary_step_end": completed_steps,
                        "completed_optimizer_steps": int(interval),
                        "posterior_updated": False,
                    }
                )
                run_state["pre_nested_warmup_state"] = copy.deepcopy(warmup_state)
        if successful_step_callback is not None:
            successful_step_callback(step=step, tokens_seen=tokens_seen)

    # Import lazily to avoid a module-level cycle with src.training.steps.
    from src.training.steps import train_for_steps

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
        metrics_journal=metrics_journal,
        forced_global_action=forced_action,
        successful_step_callback=record_successful_warmup_step,
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
            "completed_steps": current_step,
            "completed": warmup_completed,
            "completion_step": completion_step,
            "transition_reason": transition_reason,
        }
    )
    if policy == "balanced_global":
        completed_windows = current_step // int(interval)
        warmup_state["current_window_index"] = completed_windows
        warmup_state["current_window_offset"] = current_step % int(interval)
        counts = {str(label): 0 for label in config["model"]["granularities"]}
        counts.update(Counter(schedule[:completed_windows]))
        warmup_state["per_granularity_counts"] = counts
    run_state.update(
        {
            "warmup_completed": warmup_completed,
            "warmup_completion_step": completion_step,
            "warmup_transition_reason": transition_reason,
        }
    )
    run_state["pre_nested_warmup_state"] = copy.deepcopy(warmup_state)
    update_pre_nested_warmup_state(config, warmup_state)

    if policy == "balanced_global" and warmup_event_callback is not None:
        if warmup_completed:
            last_window_index = len(schedule) - 1
            event_type = "warmup_completed"
            event_window_index = last_window_index
            event_action = schedule[last_window_index]
            event_start = last_window_index * int(interval)
            event_completed_steps = int(interval)
        else:
            event_type = "warmup_terminal_incomplete"
            event_window_index = min(current_step // int(interval), len(schedule) - 1)
            event_action = schedule[event_window_index]
            event_start = event_window_index * int(interval)
            event_completed_steps = current_step - event_start
        warmup_event_callback(
            {
                "schema_version": 1,
                "event_type": event_type,
                "phase": "warmup",
                "schedule_hash": warmup_state["schedule_hash"],
                "schedule_seed": warmup_state["schedule_seed"],
                "action_interval_steps": int(interval),
                "requested_warmup_steps": warmup_target_steps,
                "completed_warmup_steps": current_step,
                "per_granularity_counts": dict(
                    warmup_state["per_granularity_counts"]
                ),
                "controller_start_step": warmup_state["controller_start_step"],
                "action": _warmup_action(config, event_action),
                "warmup_window_index": event_window_index,
                "window_index": event_window_index,
                "boundary_step": current_step,
                "boundary_step_start": event_start,
                "boundary_step_end": current_step,
                "completed_optimizer_steps": event_completed_steps,
                "posterior_updated": False,
            }
        )

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


def _warmup_action(config: Mapping[str, Any], granularity: str) -> dict[str, Any]:
    label = str(granularity)
    return {
        "scope": "global",
        "global_granularity": label,
        "block_granularities": [label] * int(config["model"]["num_layers"]),
    }
