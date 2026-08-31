"""Config-driven training flow for MatFormer reproduction runs."""

from __future__ import annotations

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*args, **kwargs):
        return None

load_dotenv()

import copy
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn.utils import clip_grad_norm_
from transformers import get_scheduler

import src.training.checkpointing as training_checkpointing
import src.training.data as training_data
from src.training.optimizer_state import (
    GlobalSchedulerClock,
    PerGranularityOptimizerCollection,
    build_per_granularity_optimizer_runtime,
)
from src.evaluation.validation import (
    configure_model_granularity,
    evaluate_validation_per_granularity,
    move_batch_to_device,
    perplexity_from_loss,
    validation_results_to_metric_rows,
)
from src.models.adaptive_sampler import (
    build_adaptive_sampler_artifact_fields,
    select_adaptive_sampler_layer_granularities,
)
from src.models.correction import summarize_correction_context_from_config
from src.models.ffn import (
    get_ffn_prefix_metadata,
)
from src.models.granularity import (
    resolved_granularity_artifact_fields,
    summarize_granularity_pattern_from_config,
)
from src.training.checkpointing import (
    _prepare_adaptive_sampler_runtime_state,
    _update_adaptive_sampler_runtime_state,
    continuation_latest_checkpoint_policy,
    maybe_write_best_eval_checkpoint,
)
from src.training.distributed import (
    autocast_context,
    broadcast_object,
    sum_float,
    sum_int,
)
from src.training.monitoring import NoopHeartbeatWriter
from src.training.schedules import scheduler_metric_fields
from src.training.portfolio_catchup import (
    PortfolioCatchupError,
    portfolio_metric_fields,
    update_portfolio_catchup_state,
    uses_portfolio_catchup,
    validate_portfolio_catchup_state,
)
from src.utils.config import (
    ConfigError,
    resolve_optimizer_kwargs,
    resolve_sampling_mode_from_config_sections,
)
from src.utils.heartbeats import (
    build_heartbeat_cadence,
    heartbeat_stage,
    heartbeat_training_fields,
    maybe_emit_training_heartbeat,
)
from src.utils.metrics import (
    build_compact_controller_metric_fields,
    json_artifact_value,
    summarize_runtime_granularity_pattern_from_config,
    write_json_artifact,
)
from src.utils.reproducibility import (
    capture_rng_state,
    dedicated_random,
    restore_rng_state,
    seed_for,
)


def build_optimizer_and_scheduler(model, training: Mapping[str, Any]):
    """Build the training optimizer and scheduler from resolved config fields."""
    if training.get("optimizer_state_scope", "shared") == "per_granularity":
        return build_per_granularity_optimizer_runtime(model, training)

    optimizer_name = str(training.get("optimizer_name", "adamw"))
    optimizer_kwargs = resolve_optimizer_kwargs(
        optimizer_name,
        training.get("optimizer_kwargs", {}),
    )
    scheduler_name = str(training.get("scheduler_name", "cosine"))
    scheduler_kwargs = dict(training.get("scheduler_kwargs", {}))
    resolved_warmup_steps = int(
        training.get(
            "resolved_warmup_steps",
            training.get("scheduler", {}).get("resolved_warmup_steps", 0),
        )
    )
    learning_rate = training.get("resolved_learning_rate", training.get("learning_rate"))
    if learning_rate is None:
        raise ConfigError(
            "training must include learning_rate or resolved_learning_rate"
        )

    if optimizer_name == "adamw":
        if "betas" in optimizer_kwargs and isinstance(optimizer_kwargs["betas"], list):
            optimizer_kwargs["betas"] = tuple(optimizer_kwargs["betas"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            **optimizer_kwargs,
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            **optimizer_kwargs,
        )
    else:
        raise ConfigError(f"Unsupported optimizer name: {optimizer_name}")

    scheduler = get_scheduler(
        scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=resolved_warmup_steps,
        num_training_steps=(
            None
            if scheduler_name == "warmup_stable_decay"
            and "num_stable_steps" in scheduler_kwargs
            else int(training["max_steps"])
        ),
        scheduler_specific_kwargs=scheduler_kwargs,
    )
    return optimizer, scheduler


def _is_concat_lmc_module(module: torch.nn.Module) -> bool:
    return bool(
        getattr(module, "gradient_membership_counts", None)
        and any(
            hasattr(module, attr)
            for attr in (
                "gate_weight_blocks",
                "up_weight_blocks",
                "down_weight_blocks",
                "gate_bias_blocks",
                "up_bias_blocks",
            )
        )
    )


def _capture_concat_lmc_snapshots(
    model: torch.nn.Module,
    total_losses: int,
) -> list[tuple[torch.nn.Parameter, torch.Tensor, float]]:
    snapshots: list[tuple[torch.nn.Parameter, torch.Tensor, float]] = []
    if total_losses <= 0:
        return snapshots

    for module in model.modules():
        if not _is_concat_lmc_module(module):
            continue

        counts = list(getattr(module, "gradient_membership_counts", []))
        scales = [
            (float(total_losses) / float(count)) if int(count) > 0 else 1.0
            for count in counts
        ]
        block_groups = [
            getattr(module, "gate_weight_blocks", None),
            getattr(module, "up_weight_blocks", None),
            getattr(module, "down_weight_blocks", None),
            getattr(module, "gate_bias_blocks", None),
            getattr(module, "up_bias_blocks", None),
        ]

        for blocks in block_groups:
            if blocks is None:
                continue
            for block_index, param in enumerate(blocks):
                if block_index >= len(scales) or not isinstance(param, torch.nn.Parameter):
                    continue
                if not param.requires_grad:
                    continue
                scale = scales[block_index]
                if scale == 1.0:
                    continue
                snapshots.append((param, param.detach().clone(), scale))

    return snapshots


def _apply_concat_lmc_corrections(
    snapshots: list[tuple[torch.nn.Parameter, torch.Tensor, float]],
) -> None:
    if not snapshots:
        return

    with torch.no_grad():
        for param, pre_step_value, scale in snapshots:
            base_delta = pre_step_value - param.data
            if scale == 1.0:
                continue
            param.data.copy_(pre_step_value - (base_delta * scale))


def _maybe_apply_concat_lmc_optimizer_step(
    config: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    total_losses: int,
) -> None:
    if config.get("model", {}).get("correction_mode") != "lmc":
        optimizer.step()
        return

    snapshots = _capture_concat_lmc_snapshots(model, total_losses)
    optimizer.step()
    _apply_concat_lmc_corrections(snapshots)


def optimizer_window_loss_scale(
    *,
    local_valid_targets: int,
    total_window_valid_targets: int,
    distributed_context=None,
    granularity_count: int = 1,
) -> float:
    """Return the exact FSDP scale for one local loss contribution."""

    local_count = int(local_valid_targets)
    total_count = int(total_window_valid_targets)
    granularity_count = int(granularity_count)
    if local_count <= 0 or total_count <= 0 or local_count > total_count:
        raise ValueError("Optimizer-window weighting requires valid target counts")
    if granularity_count <= 0:
        raise ValueError("granularity_count must be positive")
    world_size = int(getattr(distributed_context, "world_size", 1))
    return world_size * local_count / total_count / granularity_count


def group_optimizer_windows(iterable, accumulation_steps: int):
    """Yield consecutive, nonempty windows, including a final partial window."""

    accumulation_steps = int(accumulation_steps)
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    iterator = iter(iterable)
    while True:
        window = []
        for _ in range(accumulation_steps):
            try:
                window.append(next(iterator))
            except StopIteration:
                break
        if not window:
            return
        yield window


def validation_token_thresholds_crossed(
    previous_tokens: int,
    committed_tokens: int,
    *,
    interval_tokens: int,
    next_threshold: int | None = None,
) -> tuple[list[int], int | None]:
    """Resolve every cadence threshold crossed by one committed optimizer update."""

    interval = int(interval_tokens)
    if interval <= 0:
        return [], None
    previous = int(previous_tokens)
    committed = int(committed_tokens)
    threshold = int(next_threshold or interval)
    if threshold <= previous:
        threshold = ((previous // interval) + 1) * interval
    crossed = []
    while threshold <= committed:
        crossed.append(threshold)
        threshold += interval
    return crossed, threshold


def _select_optimizer_window_action(
    config: dict[str, Any],
    granularities: list[str],
    device: torch.device,
    *,
    optimizer_step: int,
    tokens_seen: int,
    supports_layer_granularities: bool,
    forced_global_action=None,
    probabilistic_controller=None,
    panelgrad_controller=None,
    panelgrad_refresh_callback=None,
    distributed_context=None,
    adaptive_sampler_state=None,
    stage_name: str,
    run_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_sampling_mode = str(
        config["model"].get("granularity_sampling_mode", "global")
    )
    run_sampling_mode = str(config["run"].get("sampling_mode", "nested-random"))
    if forced_global_action is not None:
        selected = forced_global_action(optimizer_step) if callable(
            forced_global_action
        ) else forced_global_action
        selected = str(selected)
        if selected not in granularities:
            raise ConfigError(
                f"forced global action must be a resolved granularity: {selected!r}"
            )
        return {"kind": "global", "granularities": [selected]}
    if run_sampling_mode == "nested-all":
        return {"kind": "nested_all", "granularities": list(granularities)}
    if model_sampling_mode == "adaptive_global" and panelgrad_controller is not None:
        # PanelGrad policy decision point: refresh if due, then draw one global
        # action. The ordinary forward/backward path below remains unchanged.
        if panelgrad_controller.phase == "refresh_pending":
            if panelgrad_refresh_callback is None:
                raise ConfigError("PanelGrad refresh callback is required")
            panelgrad_refresh_callback(
                step=optimizer_step - 1,
                tokens_seen=tokens_seen,
            )
        if bool(getattr(distributed_context, "enabled", False)):
            payload = None
            if bool(getattr(distributed_context, "is_rank_zero", False)):
                selected_action = panelgrad_controller.sample_action()
                payload = {
                    "action": selected_action,
                    "state": panelgrad_controller.state_dict(),
                }
            payload = broadcast_object(payload, context=distributed_context, src=0)
            if not bool(getattr(distributed_context, "is_rank_zero", False)):
                panelgrad_controller.restore_transaction_snapshot(payload["state"])
            selected_action = payload["action"]
        else:
            selected_action = panelgrad_controller.sample_action()
        return {
            "kind": "global",
            "granularities": [selected_action["global_granularity"]],
            "panelgrad_probability": selected_action["probability"],
        }
    if model_sampling_mode == "adaptive_global" and probabilistic_controller is not None:
        selected = probabilistic_global_layer_granularities(
            config, probabilistic_controller
        )[0]
        return {"kind": "global", "granularities": [selected]}
    if (
        model_sampling_mode == "adaptive_per_block"
        and probabilistic_controller is not None
    ):
        return {
            "kind": "per_block",
            "granularities": probabilistic_per_block_layer_granularities(
                config, probabilistic_controller
            ),
        }
    if model_sampling_mode == "adaptive_per_block" and supports_layer_granularities:
        if adaptive_sampler_state is None:
            raise ConfigError("adaptive_per_block runs require adaptive sampler state")
        return {
            "kind": "per_block",
            "granularities": select_adaptive_sampler_layer_granularities(
                adaptive_sampler_state,
                block_count=int(config["model"]["num_layers"]),
                step=optimizer_step,
                phase=stage_name,
                granularities=granularities,
                adaptive_seed=seed_for(config, "adaptive_sampling"),
            ),
        }
    if model_sampling_mode == "per_block" and supports_layer_granularities:
        return {
            "kind": "per_block",
            "granularities": select_training_layer_granularities(
                config, granularities, device
            ),
        }
    if model_sampling_mode == "global" and run_sampling_mode == "nested-random":
        if run_state is None:
            raise ConfigError("Global sampling requires continuation state")
        state = run_state.get("global_sampling_state")
        if state is None:
            state = training_checkpointing.build_initial_global_sampling_state(config)
            run_state["global_sampling_state"] = state
        state = training_checkpointing.validate_global_sampling_state(
            state,
            config=config,
        )
        run_state["global_sampling_state"] = state
        interval = int(state["interval_steps"])
        if state.get("schedule") == "balanced_cycle":
            return {
                "kind": "global",
                "granularities": [str(state["held_granularity"])],
                "global_sampling_schedule": "balanced_cycle",
                "global_sampling_schedule_version": int(
                    state["schedule_version"]
                ),
                "global_sampling_interval_steps": interval,
                "global_sampling_window_index": int(state["window_index"]),
                "global_sampling_window_progress": int(
                    state["successful_updates_in_window"]
                ),
                "global_sampling_cycle_index": int(state["cycle_index"]),
                "global_sampling_cycle_position": int(state["cycle_position"]),
            }
        if int(state["successful_updates_in_window"]) == interval:
            state["window_index"] = int(state["window_index"]) + 1
            state["successful_updates_in_window"] = 0
            state["held_granularity"] = None
        if state["held_granularity"] is None:
            state["held_granularity"] = select_training_granularities(
                config, granularities, device
            )[0]
        return {
            "kind": "global",
            "granularities": [str(state["held_granularity"])],
            "global_sampling_interval_steps": interval,
            "global_sampling_window_index": int(state["window_index"]),
            # This is committed progress at selection time. It is updated after
            # the optimizer commits so metrics expose the post-commit value.
            "global_sampling_window_progress": int(
                state["successful_updates_in_window"]
            ),
        }
    action = {
        "kind": "global",
        "granularities": select_training_granularities(
            config, granularities, device
        ),
    }
    if model_sampling_mode == "fixed_global":
        selected = action["granularities"][0]
        action["sampled_probability"] = float(
            config["model"]["global_sampling_distribution"][selected]
        )
    return action


def _commit_global_sampling_window_action(
    config: Mapping[str, Any],
    run_state: dict[str, Any],
    action: dict[str, Any],
) -> None:
    """Advance a held global action only after its optimizer update commits."""

    if "global_sampling_window_index" not in action:
        return
    state = run_state.get("global_sampling_state")
    if not isinstance(state, dict):
        raise ConfigError("Global sampling state is missing at commit")
    selected = str(action["granularities"][0])
    if selected != state["held_granularity"]:
        raise ConfigError("Committed global action does not match held state")
    if int(action["global_sampling_window_index"]) != int(state["window_index"]):
        raise ConfigError("Committed global action has the wrong window index")
    if int(action["global_sampling_window_progress"]) != int(
        state["successful_updates_in_window"]
    ):
        raise ConfigError("Committed global action has stale window progress")
    if state.get("schedule") == "balanced_cycle" and (
        action.get("global_sampling_schedule") != "balanced_cycle"
        or int(action.get("global_sampling_schedule_version", -1))
        != int(state["schedule_version"])
        or int(action.get("global_sampling_cycle_index", -1))
        != int(state["cycle_index"])
        or int(action.get("global_sampling_cycle_position", -1))
        != int(state["cycle_position"])
    ):
        raise ConfigError(
            "Committed balanced global action has stale cycle identity"
        )

    state["successful_updates_in_window"] = int(
        state["successful_updates_in_window"]
    ) + 1
    state["total_successful_updates"] = int(state["total_successful_updates"]) + 1
    state["exposure_counts"][selected] = int(
        state["exposure_counts"][selected]
    ) + 1
    action["global_sampling_window_progress"] = int(
        state["successful_updates_in_window"]
    )
    if state.get("schedule") == "balanced_cycle" and int(
        state["successful_updates_in_window"]
    ) == int(state["interval_steps"]):
        state["successful_updates_in_window"] = 0
        state["window_index"] = int(state["window_index"]) + 1
        if int(state["cycle_position"]) + 1 < len(state["granularities"]):
            state["cycle_position"] = int(state["cycle_position"]) + 1
        else:
            previous_last = str(state["cycle_permutation"][-1])
            state["cycle_index"] = int(state["cycle_index"]) + 1
            state["cycle_position"] = 0
            state["cycle_permutation"] = (
                training_checkpointing._balanced_cycle_permutation(
                    config,
                    cycle_index=int(state["cycle_index"]),
                    previous_last=previous_last,
                )
            )
        state["held_granularity"] = state["cycle_permutation"][
            int(state["cycle_position"])
        ]
    run_state["global_sampling_state"] = state


def _training_action_heartbeat_fields(action: Mapping[str, Any]) -> dict[str, Any]:
    """Build one compact action description shared by all sampling modes."""

    kind = str(action.get("kind") or "unknown")
    granularities = [str(label) for label in action.get("granularities", [])]
    fields: dict[str, Any] = {"selection_kind": kind}
    if kind == "global" and len(granularities) == 1:
        fields["selected_granularity"] = granularities[0]
    elif kind == "nested_all":
        fields["selected_granularity"] = "all"
        fields["selected_granularities"] = granularities
    elif granularities:
        counts = {
            label: granularities.count(label)
            for label in dict.fromkeys(granularities)
        }
        fields["selected_granularity"] = "per_block"
        fields["selected_granularity_counts"] = counts
    sampled_probability = action.get(
        "sampled_probability", action.get("panelgrad_probability")
    )
    if sampled_probability is not None:
        fields["controller_sampled_probability"] = float(
            sampled_probability
        )
    for field in (
        "global_sampling_schedule_version",
        "global_sampling_interval_steps",
        "global_sampling_window_index",
        "global_sampling_window_progress",
        "global_sampling_cycle_index",
        "global_sampling_cycle_position",
    ):
        if action.get(field) is not None:
            fields[field] = int(action[field])
    if action.get("global_sampling_schedule") is not None:
        fields["global_sampling_schedule"] = str(
            action["global_sampling_schedule"]
        )
    return fields


def _optimizer_action_id(action: Mapping[str, Any], pending_step: int) -> str:
    selected = ",".join(str(value) for value in action.get("granularities", []))
    schedule = str(action.get("global_sampling_schedule") or "global")
    window = action.get("global_sampling_window_index", pending_step - 1)
    cycle = action.get("global_sampling_cycle_index", "-")
    position = action.get("global_sampling_cycle_position", "-")
    return f"{schedule}:{window}:{cycle}:{position}:{selected}"


def _optimizer_batch_provenance(
    *,
    epoch: int,
    window: list[tuple[int, Any]],
    sampler_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    provenance = {
        "epoch": int(epoch),
        "batch_indices": [int(batch_index) for batch_index, _ in window],
    }
    if isinstance(sampler_state, Mapping):
        for field in (
            "permutation_hash",
            "epoch",
            "global_batch_cursor",
            "sample_cursor",
        ):
            if field in sampler_state:
                provenance[field] = copy.deepcopy(sampler_state[field])
    return provenance


def _optimizer_exposure_counts(run_state: Mapping[str, Any]) -> dict[str, int]:
    sampling_state = run_state.get("global_sampling_state")
    if not isinstance(sampling_state, Mapping):
        return {}
    exposures = sampling_state.get("exposure_counts")
    if not isinstance(exposures, Mapping):
        return {}
    return {str(label): int(value) for label, value in exposures.items()}


def _optimizer_successful_update_counts(
    run_state: Mapping[str, Any],
) -> dict[str, int]:
    counts = run_state.get("optimizer_update_counts")
    if isinstance(counts, Mapping):
        return {str(label): int(value) for label, value in counts.items()}
    return _optimizer_exposure_counts(run_state)


def _backward_context(model, *, synchronize: bool):
    if synchronize or not hasattr(model, "no_sync"):
        return nullcontext()
    return model.no_sync()


def _forward_backward_microbatch(
    config: dict[str, Any],
    model,
    batch: dict[str, torch.Tensor],
    action: Mapping[str, Any],
    *,
    device: torch.device,
    local_valid_targets: int,
    total_window_valid_targets: int,
    distributed_context=None,
    is_last_microstep: bool,
) -> tuple[list[tuple[str, float, dict[str, Any], dict[str, Any]]], int]:
    """Run one microbatch under a preselected optimizer-window action."""

    selected = list(action["granularities"])
    metric_data = []
    if action["kind"] == "nested_all":
        total_losses = len(selected)
        for granularity_index, granularity in enumerate(selected):
            synchronize = is_last_microstep and granularity_index == total_losses - 1
            with _backward_context(model, synchronize=synchronize):
                configure_model_granularity(model, granularity)
                pattern, correction = _runtime_granularity_artifacts(config, model)
                with autocast_context(config, device):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch["labels"],
                    )
                loss_value = float(outputs.loss.detach().float().cpu().item())
                metric_data.append((granularity, loss_value, pattern, correction))
                scale = optimizer_window_loss_scale(
                    local_valid_targets=local_valid_targets,
                    total_window_valid_targets=total_window_valid_targets,
                    distributed_context=distributed_context,
                    granularity_count=total_losses,
                )
                (outputs.loss * scale).backward()
        return metric_data, total_losses

    with _backward_context(model, synchronize=is_last_microstep):
        if action["kind"] == "per_block":
            configure_model_layer_granularities(model, selected)
            label = ",".join(selected)
        else:
            configure_model_granularity(model, selected[0])
            label = selected[0]
        pattern, correction = _runtime_granularity_artifacts(config, model)
        with autocast_context(config, device):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
            )
        loss_value = float(outputs.loss.detach().float().cpu().item())
        metric_data.append((label, loss_value, pattern, correction))
        scale = optimizer_window_loss_scale(
            local_valid_targets=local_valid_targets,
            total_window_valid_targets=total_window_valid_targets,
            distributed_context=distributed_context,
        )
        (outputs.loss * scale).backward()
    return metric_data, 1


def train_for_steps(
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
    probabilistic_controller=None,
    probabilistic_boundary_callback=None,
    probabilistic_completion_callback=None,
    panelgrad_controller=None,
    panelgrad_refresh_callback=None,
    panelgrad_completion_callback=None,
    forced_global_action=None,
    successful_step_callback=None,
) -> list[dict[str, Any]]:
    training = config["training"]
    run = config["run"]
    granularities = list(config["model"]["granularities"])
    model_sampling_mode = str(config["model"].get("granularity_sampling_mode", "global"))
    target_model = model.module if hasattr(model, "module") else model
    supports_layer_granularities = hasattr(target_model, "configure_layer_granularities")
    token_budget = training["token_budget"]
    max_steps = training["max_steps"]
    validation_config = config.get("evaluation", {}).get("validation", {})
    eval_interval = int(validation_config.get("interval_steps", 0))
    eval_interval_tokens = int(validation_config.get("interval_tokens", 0))
    accumulation_steps = int(training.get("gradient_accumulation_steps", 1))

    metrics_rows = []
    start_time = time.time()
    run_state = run_state if run_state is not None else training_checkpointing.build_initial_continuation_state(config)
    step = int(run_state.get("last_completed_step", 0))
    epoch = int(run_state.get("epoch", 0))
    resume_batch_index = int(run_state.get("batch_index", 0))
    tokens_seen = int(
        run_state.get("tokens_seen", budget_tokens_seen_for_step(config, step))
    )
    content_tokens_seen = int(run_state.get("content_tokens_seen", 0))
    heartbeat_writer = heartbeat_writer or NoopHeartbeatWriter()
    heartbeat_cadence = build_heartbeat_cadence(config)
    # The stage-start event is the cadence baseline. This keeps ordinary fast
    # steps on committed boundaries (10, 20, ...) instead of consuming the
    # cadence on the first in-flight microbatch (1, 11, ...).
    heartbeat_cadence.mark_emitted(step=step, now=start_time)
    latest_committed_loss: float | None = None
    latest_committed_loss_step: int | None = None
    checkpoint_state = checkpoint_state if checkpoint_state is not None else {}
    latest_checkpoint_path = Path(
        run_state.get("latest_checkpoint_path")
        or Path(config["run"]["output_dir"]) / "checkpoints" / "latest.pt"
    )
    resume_epoch = epoch
    resume_batch_index = max(0, resume_batch_index)
    packed_batch_sampler = getattr(train_dataloader, "batch_sampler", None)
    if hasattr(packed_batch_sampler, "permutation_hash"):
        # The immutable sampler cursor already identifies the exact next global
        # batch; epoch/batch skipping would advance it a second time.
        resume_epoch = 0
        epoch = 0
        resume_batch_index = 0
    elif len(train_dataloader) > 0:
        while resume_batch_index >= len(train_dataloader):
            resume_batch_index -= len(train_dataloader)
            resume_epoch += 1
    else:
        resume_batch_index = 0
    run_state["epoch"] = resume_epoch
    run_state["batch_index"] = resume_batch_index
    run_state["content_tokens_seen"] = content_tokens_seen
    run_state.setdefault("microstep", int(run_state.get("microstep", step)))
    if eval_interval_tokens > 0:
        run_state.setdefault(
            "next_validation_tokens",
            ((tokens_seen // eval_interval_tokens) + 1) * eval_interval_tokens,
        )
    else:
        run_state.setdefault("next_validation_tokens", None)
    run_state.setdefault("latest_checkpoint_step", int(run_state.get("last_completed_step", 0)))
    run_state.setdefault("status", "fresh")
    if isinstance(optimizer, PerGranularityOptimizerCollection):
        if not isinstance(scheduler, GlobalSchedulerClock):
            raise ConfigError(
                "Per-granularity optimizer state requires a global scheduler clock"
            )
        expected_counts = run_state.get("optimizer_update_counts")
        if not isinstance(expected_counts, Mapping):
            expected_counts = {
                label: 0 for label in optimizer.ordered_granularities
            }
        if dict(expected_counts) != optimizer.successful_update_counts:
            raise ConfigError(
                "Restored optimizer update counts do not match the collection"
            )
        if (
            optimizer.total_successful_updates != step
            or scheduler.position != step
        ):
            raise ConfigError(
                "Restored optimizer, scheduler, and committed-step positions do not reconcile"
            )
        run_state["optimizer_update_counts"] = copy.deepcopy(
            optimizer.successful_update_counts
        )
        run_state["optimizer_total_successful_updates"] = (
            optimizer.total_successful_updates
        )
        run_state["optimizer_last_active_granularity"] = (
            optimizer.last_active_granularity
        )
        run_state["global_scheduler_position"] = scheduler.position
        run_state.setdefault("optimizer_active_owner_granularity", None)
    if continuation_latest_checkpoint_policy(config)["enabled"] and not run_state.get("latest_checkpoint_path"):
        run_state["latest_checkpoint_path"] = str(latest_checkpoint_path)

    adaptive_sampler_state = None
    if probabilistic_controller is None:
        if forced_global_action is None:
            if (
                config.get("model", {}).get("granularity_sampling_mode")
                == "adaptive_per_block"
                and config.get("model", {}).get("adaptive_sampler_strategy") != "ucb"
            ):
                raise ConfigError(
                    "Legacy heuristic Thompson is not selectable; Thompson runs "
                    "require the probabilistic controller"
                )
            adaptive_sampler_state = _prepare_adaptive_sampler_runtime_state(
                config,
                run_state,
            )
    else:
        run_state.pop("adaptive_sampler_state", None)

    model.train()
    with heartbeat_stage(heartbeat_writer, stage_name):
        while step < max_steps and tokens_seen < token_budget:
            set_dataloader_epoch(train_dataloader, epoch)
            made_progress = False
            current_epoch = epoch
            epoch += 1
            indexed_batches = iter(enumerate(train_dataloader))
            while step < max_steps and tokens_seen < token_budget:
                window_rng_snapshot = capture_rng_state()
                window_state_snapshot = copy.deepcopy(run_state)
                controller_snapshot = (
                    probabilistic_controller.transaction_snapshot()
                    if probabilistic_controller is not None
                    else None
                )
                panelgrad_snapshot = (
                    panelgrad_controller.transaction_snapshot()
                    if panelgrad_controller is not None
                    else None
                )
                window_sampler_snapshot = training_data.packed_sampler_state(
                    train_dataloader
                )
                in_flight_heartbeat_emitted = False
                window: list[tuple[int, dict[str, torch.Tensor]]] = []
                while len(window) < accumulation_steps:
                    try:
                        batch_index_in_epoch, batch = next(indexed_batches)
                    except StopIteration:
                        break
                    if (
                        current_epoch == resume_epoch
                        and batch_index_in_epoch < resume_batch_index
                    ):
                        continue
                    window.append((batch_index_in_epoch, batch))
                if not window:
                    break
                made_progress = True
                optimizer_committed = False
                action = None
                optimizer_owner = None
                optimizer_action_id = None
                optimizer_batch_provenance = None
                failure_stage = "window_preparation"
                try:
                    prepared_window = [
                        (batch_index, move_batch_to_device(batch, device))
                        for batch_index, batch in window
                    ]
                    local_target_counts = [
                        count_valid_prediction_targets(batch)
                        for _, batch in prepared_window
                    ]
                    if any(count <= 0 for count in local_target_counts):
                        raise ValueError(
                            "Training microbatch contains zero valid causal prediction targets"
                        )
                    total_window_valid_targets = sum_int(
                        sum(local_target_counts),
                        device=device,
                        context=distributed_context,
                    )
                    pending_step = step + 1
                    failure_stage = "action_selection"
                    action = _select_optimizer_window_action(
                        config,
                        granularities,
                        device,
                        optimizer_step=pending_step,
                        tokens_seen=tokens_seen,
                        supports_layer_granularities=supports_layer_granularities,
                        forced_global_action=forced_global_action,
                        probabilistic_controller=probabilistic_controller,
                        panelgrad_controller=panelgrad_controller,
                        panelgrad_refresh_callback=panelgrad_refresh_callback,
                        distributed_context=distributed_context,
                        adaptive_sampler_state=adaptive_sampler_state,
                        stage_name=stage_name,
                        run_state=run_state,
                    )
                    optimizer_action_id = _optimizer_action_id(action, pending_step)
                    optimizer_batch_provenance = _optimizer_batch_provenance(
                        epoch=current_epoch,
                        window=window,
                        sampler_state=window_sampler_snapshot,
                    )
                    step_optimizer = optimizer
                    if isinstance(optimizer, PerGranularityOptimizerCollection):
                        optimizer_owner = optimizer.owner_from_action(action)
                        step_optimizer = optimizer.optimizer_for(optimizer_owner)
                        run_state["optimizer_active_owner_granularity"] = (
                            optimizer_owner
                        )
                    optimizer.zero_grad(set_to_none=True)
                    local_loss_numerators: dict[str, float] = {}
                    runtime_artifacts: dict[
                        str, tuple[dict[str, Any], dict[str, Any]]
                    ] = {}
                    total_losses = 1
                    local_window_content_tokens = 0
                    failure_stage = "forward_backward"
                    for microstep_index, (
                        _,
                        batch,
                    ) in enumerate(prepared_window, start=1):
                        local_count = local_target_counts[microstep_index - 1]
                        micro_metrics, total_losses = _forward_backward_microbatch(
                            config,
                            model,
                            batch,
                            action,
                            device=device,
                            local_valid_targets=local_count,
                            total_window_valid_targets=total_window_valid_targets,
                            distributed_context=distributed_context,
                            is_last_microstep=microstep_index == len(prepared_window),
                        )
                        local_window_content_tokens += count_content_tokens(batch)
                        for label, loss_value, pattern, correction in micro_metrics:
                            local_loss_numerators[label] = (
                                local_loss_numerators.get(label, 0.0)
                                + loss_value * local_count
                            )
                            runtime_artifacts[label] = (pattern, correction)
                        now = time.time()
                        if heartbeat_cadence.should_emit(step=pending_step, now=now):
                            heartbeat_writer.heartbeat(
                                stage_name,
                                **heartbeat_training_fields(
                                    config,
                                    step=pending_step,
                                    tokens_seen=tokens_seen,
                                    content_tokens_seen=content_tokens_seen,
                                    latest_loss=latest_committed_loss,
                                    peak_gpu_memory_bytes=current_peak_memory_bytes(
                                        device
                                    ),
                                ),
                                progress_state="optimizer_window_in_progress",
                                latest_loss_step=latest_committed_loss_step,
                                **_training_action_heartbeat_fields(action),
                                pending_optimizer_step=pending_step,
                                pending_microstep=microstep_index,
                                optimizer_window_microsteps=len(prepared_window),
                                committed_microsteps=int(run_state["microstep"]),
                            )
                            heartbeat_cadence.mark_emitted(
                                step=pending_step, now=now
                            )
                            in_flight_heartbeat_emitted = True

                    committed_tokens = sum_int(
                        local_window_content_tokens,
                        device=device,
                        context=distributed_context,
                    )
                    if (
                        config.get("dataset", {}).get("mode") == "packed_mmap"
                        and tokens_seen + committed_tokens > token_budget
                    ):
                        raise AssertionError(
                            "Packed-mmap training would commit tokens beyond "
                            "training.token_budget"
                        )
                    gradient_clip_norm = training.get("gradient_clip_norm")
                    failure_stage = "gradient_clipping"
                    if gradient_clip_norm is not None:
                        clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
                    # LambdaLR position ``pending_step - 1`` is the rate applied by
                    # this update. Capture it before optimizer.step/scheduler.step
                    # so metrics never report the rate prepared for the next update.
                    if isinstance(optimizer, PerGranularityOptimizerCollection):
                        optimizer.validate_synchronized_learning_rates()
                    committed_learning_rates = [
                        float(group["lr"]) for group in step_optimizer.param_groups
                    ]
                    if not committed_learning_rates or any(
                        not math.isfinite(value) for value in committed_learning_rates
                    ):
                        raise RuntimeError("Optimizer learning rate is missing or non-finite")
                    if any(
                        value != committed_learning_rates[0]
                        for value in committed_learning_rates[1:]
                    ):
                        raise RuntimeError(
                            "Per-row learning_rate requires one shared optimizer rate"
                        )
                    committed_learning_rate = committed_learning_rates[0]
                    failure_stage = "optimizer_step"
                    _maybe_apply_concat_lmc_optimizer_step(
                        config,
                        model,
                        step_optimizer,
                        total_losses=total_losses,
                    )
                    # A successful optimizer return is irreversible. Scheduler
                    # and accounting failures after this point are fatal and do
                    # not restore the pre-window transactional snapshots.
                    optimizer_committed = True
                    failure_stage = "post_commit_accounting"
                    scheduler.step()
                    if isinstance(optimizer, PerGranularityOptimizerCollection):
                        if not isinstance(scheduler, GlobalSchedulerClock):
                            raise RuntimeError(
                                "Per-granularity optimizer state requires a global scheduler clock"
                            )
                        scheduler.synchronize(optimizer)
                        optimizer.record_successful_update(optimizer_owner)
                        run_state["optimizer_update_counts"] = copy.deepcopy(
                            optimizer.successful_update_counts
                        )
                        run_state["optimizer_total_successful_updates"] = (
                            optimizer.total_successful_updates
                        )
                        run_state["optimizer_last_active_granularity"] = (
                            optimizer.last_active_granularity
                        )
                        run_state["global_scheduler_position"] = scheduler.position
                    step = pending_step

                    previous_tokens_seen = tokens_seen
                    content_tokens_seen += committed_tokens
                    if config.get("dataset", {}).get("mode") == "packed_mmap":
                        tokens_seen += committed_tokens
                    else:
                        # Historical raw-tokenized runs retain their nominal packed
                        # budget semantics; production packed-mmap commits exact IDs.
                        tokens_seen = budget_tokens_seen_for_step(config, step)
                    crossed_thresholds, next_threshold = (
                        validation_token_thresholds_crossed(
                            previous_tokens_seen,
                            tokens_seen,
                            interval_tokens=eval_interval_tokens,
                            next_threshold=run_state.get(
                                "next_validation_tokens"
                            ),
                        )
                    )
                    if eval_interval_tokens > 0:
                        run_state["next_validation_tokens"] = next_threshold
                    committed_microsteps = int(run_state["microstep"]) + len(window)
                    last_batch_index = window[-1][0]
                    run_state.update(
                        {
                            "status": "resumed"
                            if int(run_state.get("resume_count", 0)) > 0
                            else "fresh",
                            "last_completed_step": step,
                            "step": step,
                            "microstep": committed_microsteps,
                            "epoch": current_epoch,
                            "batch_index": last_batch_index + 1,
                            "tokens_seen": tokens_seen,
                            "content_tokens_seen": content_tokens_seen,
                            "optimizer_window_microsteps": len(window),
                        }
                    )
                    sampler_state = training_data.packed_sampler_state(
                        train_dataloader
                    )
                    if sampler_state is not None:
                        run_state["sampler_state"] = sampler_state
                    if probabilistic_controller is not None:
                        probabilistic_controller.record_successful_optimizer_step()
                    _commit_global_sampling_window_action(
                        config,
                        run_state,
                        action,
                    )
                    run_state["optimizer_active_owner_granularity"] = None
                    if panelgrad_controller is not None:
                        panelgrad_controller.commit_pending_action(
                            completed_step=step
                        )
                        run_state["panelgrad_state"] = (
                            panelgrad_controller.state_dict()
                        )

                    global_losses = {
                        label: sum_float(
                            numerator,
                            device=device,
                            context=distributed_context,
                        )
                        / total_window_valid_targets
                        for label, numerator in local_loss_numerators.items()
                    }
                    latest_loss = sum(global_losses.values()) / len(global_losses)
                    latest_committed_loss = latest_loss
                    latest_committed_loss_step = step
                    if successful_step_callback is not None:
                        successful_step_callback(step=step, tokens_seen=tokens_seen)
                    if probabilistic_boundary_callback is not None:
                        probabilistic_boundary_callback(
                            step=step, tokens_seen=tokens_seen
                        )

                    if probabilistic_controller is not None:
                        run_state.pop("adaptive_sampler_previous_loss", None)
                        run_state.pop("adaptive_sampler_previous_pattern", None)
                        run_state.pop("adaptive_reward_summary", None)
                        run_state.pop("adaptive_correction_penalty_summary", None)
                    elif (
                        model_sampling_mode == "adaptive_per_block"
                        and adaptive_sampler_state is not None
                    ):
                        _update_adaptive_sampler_runtime_state(
                            config,
                            run_state,
                            adaptive_sampler_state,
                            phase=stage_name,
                            latest_loss=latest_loss,
                            selected_layer_granularities=list(
                                action["granularities"]
                            ),
                            step=step,
                            epoch=current_epoch,
                        )
                    elif model_sampling_mode != "adaptive_per_block":
                        run_state.pop("adaptive_sampler_previous_loss", None)
                        run_state.pop("adaptive_sampler_previous_pattern", None)
                        run_state.pop("adaptive_reward_summary", None)
                        run_state.pop("adaptive_correction_penalty_summary", None)

                    elapsed = time.time() - start_time
                    peak_memory_bytes = current_peak_memory_bytes(device)
                    tokens_per_second = tokens_seen / elapsed if elapsed > 0 else None
                    adaptive_artifacts = _runtime_sampler_artifact_fields(
                        config, run_state, probabilistic_controller
                    )
                    for field in (
                        "global_sampling_schedule",
                        "global_sampling_schedule_version",
                        "global_sampling_interval_steps",
                        "global_sampling_window_index",
                        "global_sampling_window_progress",
                        "global_sampling_cycle_index",
                        "global_sampling_cycle_position",
                    ):
                        if action.get(field) is not None:
                            adaptive_artifacts[field] = action[field]
                    adaptive_artifacts.update(
                        {
                            "microstep": committed_microsteps,
                            "optimizer_window_microsteps": len(window),
                            "committed_tokens_this_step": committed_tokens,
                            "optimizer_state_scope": str(
                                training.get("optimizer_state_scope") or "shared"
                            ),
                            "selected_optimizer_granularity": str(
                                optimizer_owner
                                or action["granularities"][0]
                            ),
                            "optimizer_step_attempted": True,
                            "optimizer_step_committed": True,
                            "optimizer_failure_stage": None,
                            "optimizer_action_id": optimizer_action_id,
                            "optimizer_batch_provenance": (
                                optimizer_batch_provenance
                            ),
                            "optimizer_successful_update_counts": (
                                _optimizer_successful_update_counts(run_state)
                            ),
                            "optimizer_exposure_counts": (
                                _optimizer_exposure_counts(run_state)
                            ),
                            "global_scheduler_position": step,
                            **scheduler_metric_fields(
                                training,
                                scheduler_position=step - 1,
                                learning_rate=committed_learning_rate,
                            ),
                        }
                    )
                    if action.get("sampled_probability") is not None:
                        adaptive_artifacts["controller_sampled_probability"] = float(
                            action["sampled_probability"]
                        )
                    step_metric_rows = []
                    for label, loss_value in global_losses.items():
                        pattern, correction = runtime_artifacts[label]
                        step_metric_rows.append(
                            build_training_metric_row(
                                config,
                                step=step,
                                granularity=label,
                                loss=loss_value,
                                tokens_seen=tokens_seen,
                                content_tokens_seen=content_tokens_seen,
                                wall_clock_seconds=elapsed,
                                peak_memory_bytes=peak_memory_bytes,
                                granularity_pattern_summary=pattern,
                                correction_context=correction,
                                adaptive_artifacts=adaptive_artifacts,
                            )
                        )
                    _record_metric_rows(
                        metrics_rows,
                        step_metric_rows,
                        metrics_journal=metrics_journal,
                    )
                    if monitoring_session is not None:
                        monitoring_session.log_rows(step_metric_rows)
                    maybe_emit_training_heartbeat(
                        heartbeat_writer,
                        heartbeat_cadence,
                        config,
                        step=step,
                        tokens_seen=tokens_seen,
                        content_tokens_seen=content_tokens_seen,
                        latest_loss=latest_loss,
                        tokens_per_second=tokens_per_second,
                        peak_gpu_memory_bytes=peak_memory_bytes,
                        stage_name=stage_name,
                        force=in_flight_heartbeat_emitted,
                        extra_fields={
                            "progress_state": "optimizer_step_committed",
                            "latest_loss_step": step,
                            **_training_action_heartbeat_fields(action),
                        },
                    )
                    if (
                        metrics_journal is not None
                        and training_checkpointing.should_save_latest_checkpoint(
                            config, step, "step"
                        )
                    ):
                        metrics_journal.flush()
                    training_checkpointing.maybe_write_latest_checkpoint(
                        config,
                        model,
                        optimizer,
                        scheduler,
                        heartbeat_writer,
                        run_state,
                        reason="step",
                        step=step,
                        distributed_context=distributed_context,
                    )

                    step_validation = eval_interval > 0 and step % eval_interval == 0
                    validation_triggers = (
                        crossed_thresholds
                        if crossed_thresholds
                        else ([None] if step_validation else [])
                    )
                    if not validation_triggers or not bool(
                        validation_config.get("enabled", False)
                    ):
                        continue
                    run_state["validation_trigger_tokens"] = validation_triggers[-1]
                    with heartbeat_stage(
                        heartbeat_writer,
                        "validation",
                        **heartbeat_training_fields(
                            config,
                            step=step,
                            tokens_seen=tokens_seen,
                            content_tokens_seen=content_tokens_seen,
                        ),
                    ):
                        validation_results = evaluate_validation_per_granularity(
                            model,
                            eval_dataloader,
                            granularities=granularities,
                            device=device,
                            distributed=(
                                distributed_context is not None
                                and distributed_context.enabled
                            ),
                            config=config,
                        )
                    validation_results = _process_portfolio_catchup_validation(
                        config,
                        validation_results,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        tokens_seen=tokens_seen,
                        heartbeat_writer=heartbeat_writer,
                        run_state=run_state,
                        distributed_context=distributed_context,
                    )
                    validation_runtime_pattern_summary, validation_correction_context = _runtime_granularity_artifacts(
                        config,
                        model,
                    )
                    validation_metric_rows = validation_results_to_metric_rows(
                        validation_results,
                        config,
                        step=step,
                        wall_clock_seconds=elapsed,
                        tokens_per_second=tokens_per_second,
                        peak_memory_bytes=peak_memory_bytes,
                        tokens_seen=tokens_seen,
                        content_tokens_seen=content_tokens_seen,
                        granularity_pattern_summary=validation_runtime_pattern_summary,
                        correction_context=validation_correction_context,
                        adaptive_artifacts=_runtime_sampler_artifact_fields(
                            config,
                            run_state,
                            probabilistic_controller,
                        ),
                    )
                    _record_metric_rows(
                        metrics_rows,
                        validation_metric_rows,
                        metrics_journal=metrics_journal,
                        force=True,
                    )
                    if monitoring_session is not None:
                        monitoring_session.log_rows(validation_metric_rows)
                    maybe_write_best_eval_checkpoint(
                        config,
                        model,
                        validation_results,
                        step,
                        heartbeat_writer,
                        checkpoint_state,
                        run_state,
                        distributed_context=distributed_context,
                    )
                    if metrics_journal is not None and run_state.get(
                        "checkpoint_selection_step"
                    ) is not None:
                        metrics_journal.record_checkpoint_selection(
                            {
                                "path": run_state.get("best_checkpoint_path"),
                                "metric": run_state.get("checkpoint_metric"),
                                "metric_value": run_state.get(
                                    "checkpoint_metric_value"
                                ),
                                "step": run_state.get(
                                    "checkpoint_selection_step"
                                ),
                            }
                        )
                    training_checkpointing.maybe_write_latest_checkpoint(
                        config,
                        model,
                        optimizer,
                        scheduler,
                        heartbeat_writer,
                        run_state,
                        reason="validation",
                        step=step,
                        distributed_context=distributed_context,
                    )
                except Exception:
                    if action is not None and optimizer_action_id is not None:
                        failed_label = str(
                            optimizer_owner or action.get("granularities", ["unknown"])[0]
                        )
                        failure_fields = {
                            "optimizer_state_scope": str(
                                training.get("optimizer_state_scope") or "shared"
                            ),
                            "selected_optimizer_granularity": failed_label,
                            "optimizer_step_attempted": True,
                            "optimizer_step_committed": bool(optimizer_committed),
                            "optimizer_failure_stage": failure_stage,
                            "optimizer_action_id": optimizer_action_id,
                            "optimizer_batch_provenance": optimizer_batch_provenance,
                            "optimizer_successful_update_counts": (
                                _optimizer_successful_update_counts(run_state)
                            ),
                            "optimizer_exposure_counts": (
                                _optimizer_exposure_counts(run_state)
                            ),
                            "global_scheduler_position": int(
                                run_state.get("last_completed_step", step)
                            ),
                        }
                        failed_row = build_training_metric_row(
                            config,
                            step=step + 1,
                            granularity=failed_label,
                            loss=float("nan"),
                            tokens_seen=tokens_seen,
                            content_tokens_seen=content_tokens_seen,
                            wall_clock_seconds=time.time() - start_time,
                            peak_memory_bytes=current_peak_memory_bytes(device),
                            adaptive_artifacts=failure_fields,
                        )
                        _record_metric_rows(
                            metrics_rows,
                            [failed_row],
                            metrics_journal=metrics_journal,
                            force=True,
                        )
                    if not optimizer_committed:
                        restore_rng_state(window_rng_snapshot)
                        if controller_snapshot is not None:
                            probabilistic_controller.restore_transaction_snapshot(
                                controller_snapshot
                            )
                        if panelgrad_snapshot is not None:
                            panelgrad_controller.restore_transaction_snapshot(
                                panelgrad_snapshot
                            )
                        run_state.clear()
                        run_state.update(window_state_snapshot)
                        if window_sampler_snapshot is not None:
                            training_data.restore_packed_sampler_state(
                                train_dataloader, window_sampler_snapshot
                            )
                            run_state["sampler_state"] = copy.deepcopy(
                                window_sampler_snapshot
                            )
                    optimizer.zero_grad(set_to_none=True)
                    raise
            if not made_progress:
                break

    if stage_name == "training" and probabilistic_completion_callback is not None:
        probabilistic_completion_callback(step=step, tokens_seen=tokens_seen)
    if stage_name == "training" and panelgrad_completion_callback is not None:
        panelgrad_completion_callback(step=step, tokens_seen=tokens_seen)
    if stage_name == "training":
        append_final_validation_if_needed(
            metrics_rows,
            config,
            model,
            eval_dataloader,
            granularities=granularities,
            device=device,
            step=step,
            tokens_seen=tokens_seen,
            content_tokens_seen=content_tokens_seen,
            start_time=start_time,
            heartbeat_writer=heartbeat_writer,
            distributed_context=distributed_context,
            checkpoint_state=checkpoint_state,
            run_state=run_state,
            monitoring_session=monitoring_session,
            metrics_journal=metrics_journal,
            probabilistic_controller=probabilistic_controller,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    training_checkpointing.maybe_write_latest_checkpoint(
        config,
        model,
        optimizer,
        scheduler,
        heartbeat_writer,
        run_state,
        reason="completion",
        step=step,
        distributed_context=distributed_context,
    )
    if metrics_journal is not None:
        metrics_journal.flush()

    return metrics_rows


def _runtime_sampler_artifact_fields(
    config: Mapping[str, Any],
    run_state: Mapping[str, Any],
    probabilistic_controller=None,
) -> dict[str, Any]:
    fields = build_adaptive_sampler_artifact_fields(config, run_state)
    fields.update(
        training_data.optimizer_iteration_artifact_fields(
            config,
            tokens_seen=int(run_state.get("tokens_seen", 0)),
        )
    )
    if probabilistic_controller is not None:
        fields.update(
            build_compact_controller_metric_fields(
                probabilistic_controller.state_dict(),
                run_state.get("latest_controller_event"),
            )
        )
    panelgrad_state = run_state.get("panelgrad_state")
    if isinstance(panelgrad_state, Mapping):
        fields.update(build_compact_controller_metric_fields(panelgrad_state))
    portfolio_state = run_state.get("portfolio_catchup_state")
    if isinstance(portfolio_state, Mapping):
        fields.update(portfolio_metric_fields(portfolio_state))
    global_sampling_state = run_state.get("global_sampling_state")
    if isinstance(global_sampling_state, Mapping):
        fields.update(
            {
                "global_sampling_schedule": str(
                    global_sampling_state.get(
                        "schedule", "random_with_replacement"
                    )
                ),
                "global_sampling_schedule_version": (
                    int(global_sampling_state["schedule_version"])
                    if global_sampling_state.get("schedule_version") is not None
                    else None
                ),
                "global_sampling_interval_steps": int(
                    global_sampling_state["interval_steps"]
                ),
                "global_sampling_window_index": int(
                    global_sampling_state["window_index"]
                ),
                "global_sampling_window_progress": int(
                    global_sampling_state["successful_updates_in_window"]
                ),
                "global_sampling_total_successful_updates": int(
                    global_sampling_state["total_successful_updates"]
                ),
                "global_sampling_exposure_counts": json_artifact_value(
                    global_sampling_state["exposure_counts"]
                ),
            }
        )
        if global_sampling_state.get("cycle_index") is not None:
            fields["global_sampling_cycle_index"] = int(
                global_sampling_state["cycle_index"]
            )
            fields["global_sampling_cycle_position"] = int(
                global_sampling_state["cycle_position"]
            )
    return fields


def select_training_granularities(
    config: dict[str, Any],
    granularities: list[str],
    device: torch.device,
) -> list[str]:
    granularities = _resolved_granularities(config, granularities)
    if str(config["run"].get("sampling_mode", "nested-random")) == "nested-all":
        return list(granularities)

    sampling_mode = config["training"].get("granularity_sampling", "all")
    if sampling_mode == "all":
        return list(granularities)
    if sampling_mode == "random":
        selected_index = select_random_granularity_index(
            config=config,
            granularity_count=len(granularities),
            device=device,
        )
        return [granularities[selected_index]]
    raise ValueError(f"Unknown granularity sampling mode: {sampling_mode}")


def select_training_layer_granularities(
    config: dict[str, Any],
    granularities: list[str],
    device: torch.device,
) -> list[str]:
    granularities = _resolved_granularities(config, granularities)
    layer_count = int(config["model"]["num_layers"])
    if layer_count <= 0:
        raise ValueError("model.num_layers must be positive")

    return [
        granularities[
            select_random_granularity_index(
                config=config,
                granularity_count=len(granularities),
                device=device,
            )
        ]
        for _ in range(layer_count)
    ]


def probabilistic_global_layer_granularities(
    config: Mapping[str, Any],
    probabilistic_controller,
) -> list[str]:
    """Repeat the active Bayesian global action for every transformer block."""

    state = probabilistic_controller.state_dict()
    window = state.get("window", {})
    if window.get("phase") != "active_window":
        raise ConfigError(
            "Bayesian global training requires an active controller window"
        )
    action = window.get("current_action")
    if not isinstance(action, Mapping):
        raise ConfigError("Bayesian global controller action is missing")
    selected = action.get("global_granularity")
    granularities = _resolved_granularities(
        config,
        list(config.get("model", {}).get("granularities", [])),
    )
    if selected not in granularities:
        raise ConfigError(
            "Bayesian global controller selected an unknown granularity: "
            f"{selected!r}"
        )
    block_count = int(config["model"]["num_layers"])
    if block_count <= 0:
        raise ConfigError("model.num_layers must be positive")
    return [str(selected)] * block_count


def probabilistic_per_block_layer_granularities(
    config: Mapping[str, Any],
    probabilistic_controller,
) -> list[str]:
    """Return the active Bayesian profile unchanged for the current window."""

    state = probabilistic_controller.state_dict()
    if state.get("scope") != "per_block":
        raise ConfigError(
            "Bayesian per-block training requires a per_block controller"
        )
    window = state.get("window", {})
    if window.get("phase") != "active_window":
        raise ConfigError(
            "Bayesian per-block training requires an active controller window"
        )
    action = window.get("current_action")
    if not isinstance(action, Mapping) or action.get("scope") != "per_block":
        raise ConfigError("Bayesian per-block controller action is missing")
    profile = action.get("block_granularities")
    block_count = int(config["model"]["num_layers"])
    if not isinstance(profile, list) or len(profile) != block_count:
        raise ConfigError(
            "Bayesian per-block action must contain one granularity per block"
        )
    granularities = _resolved_granularities(
        config,
        list(config.get("model", {}).get("granularities", [])),
    )
    if any(label not in granularities for label in profile):
        raise ConfigError(
            "Bayesian per-block controller selected an unknown granularity: "
            f"{profile!r}"
        )
    return [str(label) for label in profile]


def _resolved_granularities(
    config: Mapping[str, Any],
    fallback: list[str],
) -> list[str]:
    """Return the ordered granularity list resolved by configuration."""

    model = config.get("model", {})
    configured = model.get("granularities") if isinstance(model, Mapping) else None
    resolved = list(configured or fallback)
    if not resolved:
        raise ConfigError("model.granularities must be a non-empty resolved list")
    return resolved


def select_random_granularity_index(
    config: Mapping[str, Any],
    granularity_count: int,
    device: torch.device,
) -> int:
    if granularity_count <= 0:
        raise ValueError("granularity_count must be positive")

    model = config.get("model", {})
    fixed_distribution = None
    if (
        isinstance(model, Mapping)
        and model.get("granularity_sampling_mode") == "fixed_global"
    ):
        raw_distribution = model.get("global_sampling_distribution")
        granularities = model.get("granularities")
        if not isinstance(raw_distribution, Mapping) or not isinstance(
            granularities, list
        ):
            raise ConfigError(
                "fixed_global sampling requires a resolved "
                "model.global_sampling_distribution"
            )
        if len(granularities) != granularity_count:
            raise ConfigError(
                "fixed_global distribution does not match the active "
                "granularity count"
            )
        fixed_distribution = [
            float(raw_distribution[label]) for label in granularities
        ]

    def sample_index() -> int:
        generator = dedicated_random(config, "granularity_selection")
        if fixed_distribution is not None:
            return int(
                generator.choices(
                    range(granularity_count),
                    weights=fixed_distribution,
                    k=1,
                )[0]
            )
        return int(generator.randrange(granularity_count))

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        selected_index = torch.empty((), dtype=torch.long, device=device)
        if torch.distributed.get_rank() == 0:
            selected_index.fill_(sample_index())
        torch.distributed.broadcast(selected_index, src=0)
        return int(selected_index.item())

    return sample_index()


def configure_model_layer_granularities(
    model,
    layer_granularities: list[str] | tuple[str, ...],
) -> None:
    target = model.module if hasattr(model, "module") else model
    configure_layer_granularities = getattr(target, "configure_layer_granularities", None)
    if configure_layer_granularities is not None:
        configure_layer_granularities(layer_granularities)
        return

    configure_subnetwork = getattr(target, "configure_subnetwork", None)
    if configure_subnetwork is not None:
        layer_granularities = tuple(layer_granularities)
        if len(layer_granularities) == 1:
            configure_subnetwork(layer_granularities[0])
            return

    raise AttributeError(
        "configure_model_layer_granularities requires a model with "
        "configure_layer_granularities or a single-granularity configure_subnetwork"
    )


def set_dataloader_epoch(dataloader, epoch: int) -> None:
    sampler = getattr(dataloader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def summarize_training_outcome(
    config: dict[str, Any],
    metrics_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    training_rows = [row for row in metrics_rows if row["split"] == "train"]
    if not training_rows:
        return {
            "steps_completed": 0,
            "tokens_seen": 0,
            "content_tokens_seen": 0,
            "stop_reason": "not_started",
        }

    steps_completed = max(int(row["step"]) for row in training_rows)
    tokens_seen = max(int(row["tokens_seen"]) for row in training_rows)
    content_tokens_seen = max(
        int(row.get("content_tokens_seen", row["tokens_seen"]))
        for row in training_rows
    )
    return {
        "steps_completed": steps_completed,
        "tokens_seen": tokens_seen,
        "content_tokens_seen": content_tokens_seen,
        "stop_reason": stop_reason_for_training(
            config,
            tokens_seen=tokens_seen,
            steps_completed=steps_completed,
        ),
    }


def stop_reason_for_training(
    config: dict[str, Any],
    tokens_seen: int,
    steps_completed: int,
) -> str:
    training = config["training"]
    if steps_completed == 0:
        return "not_started"
    if tokens_seen >= training["token_budget"]:
        return "token_budget_reached"
    return "max_steps_reached_before_token_budget"


def append_final_validation_if_needed(
    metrics_rows: list[dict[str, Any]],
    config: dict[str, Any],
    model,
    eval_dataloader,
    granularities: list[str],
    device: torch.device,
    step: int,
    tokens_seen: int,
    content_tokens_seen: int,
    start_time: float,
    heartbeat_writer=None,
    distributed_context=None,
    checkpoint_state: dict[str, Any] | None = None,
    run_state: dict[str, Any] | None = None,
    monitoring_session=None,
    metrics_journal=None,
    probabilistic_controller=None,
    optimizer=None,
    scheduler=None,
) -> None:
    validation_config = config.get("evaluation", {}).get("validation", {})
    if not validation_config.get("run_at_completion", False):
        return
    has_final_validation = any(
        row["split"] == "validation" and row["step"] == step
        for row in metrics_rows
    )
    if metrics_journal is not None:
        has_final_validation = (
            has_final_validation
            or metrics_journal.has_validation_at_step(step)
        )
    if distributed_context is not None and distributed_context.enabled:
        # Only rank zero reloads the durable metrics journal. Reduce the local
        # evidence before branching so every FSDP rank either skips or enters
        # completion validation together, including after resume.
        has_final_validation = (
            sum_int(
                int(has_final_validation),
                device=device,
                context=distributed_context,
            )
            > 0
        )
    if has_final_validation:
        return

    elapsed = time.time() - start_time
    heartbeat_writer = heartbeat_writer or NoopHeartbeatWriter()
    runtime_pattern_summary, correction_context = _runtime_granularity_artifacts(
        config,
        model,
    )
    with heartbeat_stage(
        heartbeat_writer,
        "validation",
        **heartbeat_training_fields(
            config,
            step=step,
            tokens_seen=tokens_seen,
            content_tokens_seen=content_tokens_seen,
        ),
    ):
        validation_results = evaluate_validation_per_granularity(
            model,
            eval_dataloader,
            granularities=granularities,
            device=device,
            distributed=(
                distributed_context is not None and distributed_context.enabled
            ),
            config=config,
        )
    validation_results = _process_portfolio_catchup_validation(
        config,
        validation_results,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        tokens_seen=tokens_seen,
        heartbeat_writer=heartbeat_writer,
        run_state=(
            run_state
            if run_state is not None
            else training_checkpointing.build_initial_continuation_state(config)
        ),
        distributed_context=distributed_context,
    )
    validation_metric_rows = validation_results_to_metric_rows(
        validation_results,
        config,
        step=step,
        wall_clock_seconds=elapsed,
        tokens_per_second=tokens_seen / elapsed if elapsed > 0 else None,
        peak_memory_bytes=current_peak_memory_bytes(device),
        tokens_seen=tokens_seen,
        content_tokens_seen=content_tokens_seen,
        granularity_pattern_summary=runtime_pattern_summary,
        correction_context=correction_context,
        adaptive_artifacts=_runtime_sampler_artifact_fields(
            config,
            run_state if run_state is not None else {},
            probabilistic_controller,
        ),
    )
    _record_metric_rows(
        metrics_rows,
        validation_metric_rows,
        metrics_journal=metrics_journal,
        force=True,
    )
    if monitoring_session is not None:
        monitoring_session.log_rows(validation_metric_rows)
    maybe_write_best_eval_checkpoint(
        config,
        model,
        validation_results,
        step,
        heartbeat_writer,
        checkpoint_state if checkpoint_state is not None else {},
        run_state
        if run_state is not None
        else training_checkpointing.build_initial_continuation_state(config),
        distributed_context=distributed_context,
    )


def _process_portfolio_catchup_validation(
    config: dict[str, Any],
    validation_results: list[dict[str, Any]],
    *,
    model,
    optimizer,
    scheduler,
    step: int,
    tokens_seen: int,
    heartbeat_writer,
    run_state: dict[str, Any],
    distributed_context=None,
) -> list[dict[str, Any]]:
    if not uses_portfolio_catchup(config):
        return validation_results
    try:
        state = validate_portfolio_catchup_state(
            run_state.get("portfolio_catchup_state"),
            config=config,
        )
        if state is None:
            raise PortfolioCatchupError("Portfolio catch-up state is unavailable")
        state, decorated, newly_confirmed = update_portfolio_catchup_state(
            state,
            validation_results,
            step=step,
            tokens_seen=tokens_seen,
        )
        run_state["portfolio_catchup_state"] = state
        if newly_confirmed:
            if not config["controlled_experiment"]["portfolio_catchup"].get(
                "save_confirmation_checkpoint", True
            ):
                raise PortfolioCatchupError(
                    "Portfolio catch-up confirmation checkpoint is disabled"
                )
            training_checkpointing.write_portfolio_confirmation_checkpoint(
                config,
                model,
                optimizer,
                scheduler,
                heartbeat_writer,
                run_state,
                step=step,
                joint_max_loss_gap=float(state["last_joint_max_loss_gap"]),
                distributed_context=distributed_context,
            )
        return decorated
    except PortfolioCatchupError as error:
        raise ConfigError(str(error)) from error


def _record_metric_rows(
    metrics_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    *,
    metrics_journal=None,
    force: bool = False,
) -> None:
    if metrics_journal is None:
        metrics_rows.extend(new_rows)
        return
    metrics_journal.append(new_rows, force=force)


def write_extraction_metadata_if_nested(
    config: dict[str, Any],
    model,
    output_dir: Path,
    distributed_context=None,
) -> Path | None:
    if config["run"]["model_family"] != "nested":
        return None

    metadata = build_extraction_metadata(config, model)
    return write_json_artifact(
        output_dir / "extraction_metadata.json",
        metadata,
        distributed_context=distributed_context,
    )


def build_extraction_metadata(config: dict[str, Any], model) -> dict[str, Any]:
    run = config["run"]
    model_config = config["model"]
    configured_granularities = model_config["granularities"]
    prefix_metadata = prefix_metadata_by_granularity(model, model_config)

    return {
        "run_id": run["run_id"],
        "phase_id": run["phase_id"],
        "model_family": run["model_family"],
        "model_size_label": _model_shape_label(run),
        "model_shape_label": _model_shape_label(run),
        "granularities": [
            build_granularity_extraction_metadata(
                granularity,
                configured_granularities,
                prefix_metadata[granularity],
            )
            for granularity in configured_granularities
        ],
    }


def prefix_metadata_by_granularity(
    model,
    model_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    target = model.module if hasattr(model, "module") else model
    metadata = getattr(target, "ffn_prefix_metadata", None)
    if metadata is None:
        metadata = model_config.get("ffn_prefix_metadata")
    if metadata is None:
        metadata = get_ffn_prefix_metadata(
            model_config["intermediate_size"],
            granularity_prefixes=model_config.get("granularity_prefixes"),
            granularities=model_config.get("granularities"),
        )

    return {entry["name"]: dict(entry) for entry in metadata}


def build_granularity_extraction_metadata(
    granularity: str,
    configured_granularities: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    granularity_index = configured_granularities.index(granularity)
    return {
        "granularity": granularity,
        "display_name": metadata["display_name"],
        "ffn_ratio": metadata["ffn_ratio"],
        "full_intermediate_fraction": metadata["full_intermediate_fraction"],
        "prefix_width": metadata["prefix_width"],
        "strict_prefix_of": configured_granularities[granularity_index + 1 :],
    }


def build_training_metric_row(
    config: dict[str, Any],
    step: int,
    granularity: str,
    loss: float,
    tokens_seen: int,
    content_tokens_seen: int,
    wall_clock_seconds: float,
    peak_memory_bytes: int,
    granularity_pattern_summary: dict[str, Any] | None = None,
    correction_context: dict[str, Any] | None = None,
    adaptive_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run = config["run"]
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        model = {}
    training = config.get("training", {})
    if not isinstance(training, Mapping):
        training = {}
    tokens_per_second = tokens_seen / wall_clock_seconds if wall_clock_seconds > 0 else None
    row = {
        "run_id": run["run_id"],
        "step": step,
        "microstep": None,
        "split": "train",
        "model_family": run["model_family"],
        "model_size_label": _model_shape_label(run),
        "model_shape_label": _model_shape_label(run),
        "sampling_mode": resolve_sampling_mode_from_config_sections(
            run,
            training,
        ),
        "resolved_run_mode": run.get(
            "resolved_run_mode",
            resolve_sampling_mode_from_config_sections(
                run,
                training,
            ),
        ),
        "resolved_sampling_mode": model.get(
            "resolved_sampling_mode",
            model.get("granularity_sampling_mode", "global"),
        ),
        "granularity_sampling_mode": model.get("granularity_sampling_mode"),
        "global_sampling_distribution": json_artifact_value(
            model.get("global_sampling_distribution")
        ),
        "global_sampling_schedule": model.get(
            "global_sampling_schedule", "random_with_replacement"
        ),
        "global_sampling_schedule_version": model.get(
            "global_sampling_schedule_version"
        ),
        "global_sampling_interval_steps": model.get(
            "global_sampling_interval_steps", 1
        ),
        "granularity": granularity,
        **resolved_granularity_artifact_fields(model),
        "granularity_pattern_summary": json_artifact_value(
            granularity_pattern_summary
            if granularity_pattern_summary is not None
            else model.get("granularity_pattern_summary")
            or _default_granularity_pattern_summary(config)
        ),
        "correction_context": json_artifact_value(
            correction_context
            if correction_context is not None
            else model.get("correction_context")
            or _default_correction_context(config)
        ),
        "loss": loss,
        "perplexity": perplexity_from_loss(loss),
        "tokens_seen": tokens_seen,
        "content_tokens_seen": content_tokens_seen,
        "optimizer_window_microsteps": None,
        "committed_tokens_this_step": None,
        "optimizer_state_scope": str(
            training.get("optimizer_state_scope") or "shared"
        ),
        "selected_optimizer_granularity": granularity,
        "optimizer_step_attempted": True,
        "optimizer_step_committed": True,
        "optimizer_failure_stage": None,
        "optimizer_action_id": None,
        "optimizer_batch_provenance": None,
        "optimizer_successful_update_counts": None,
        "optimizer_exposure_counts": None,
        "global_scheduler_position": max(int(step), 0),
        "wall_clock_seconds": wall_clock_seconds,
        "tokens_per_second": tokens_per_second,
        "peak_memory_bytes": peak_memory_bytes,
        **scheduler_metric_fields(
            training,
            scheduler_position=max(int(step) - 1, 0),
        ),
    }
    if adaptive_artifacts:
        row.update(adaptive_artifacts)
    return row


def select_training_granularity(granularities: list[str], step: int) -> str:
    return granularities[(step - 1) % len(granularities)]


def budget_tokens_seen_for_step(config: dict[str, Any], step: int) -> int:
    planned_tokens = step * int(config["training"]["expected_tokens_per_step"])
    return min(planned_tokens, int(config["training"]["token_budget"]))


def global_content_tokens_for_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    distributed_context=None,
) -> int:
    return sum_int(
        count_content_tokens(batch),
        device=device,
        context=distributed_context,
    )


def count_content_tokens(batch: dict[str, torch.Tensor]) -> int:
    if "attention_mask" in batch and batch["attention_mask"] is not None:
        return int(batch["attention_mask"].sum().item())
    return int((batch["labels"] != -100).sum().item())


def count_valid_prediction_targets(batch: dict[str, torch.Tensor]) -> int:
    labels = batch["labels"]
    if labels.ndim < 2 or labels.shape[1] <= 1:
        return 0
    return int((labels[:, 1:] != -100).sum().item())


def weighted_loss_for_distributed_batch(
    local_mean_loss: torch.Tensor,
    *,
    local_valid_targets: int,
    global_valid_targets: int,
    distributed_context=None,
) -> torch.Tensor:
    """Scale a local mean so FSDP's averaged gradient is globally token weighted."""

    local_count = int(local_valid_targets)
    global_count = int(global_valid_targets)
    if local_count <= 0 or global_count <= 0 or local_count > global_count:
        raise ValueError("Distributed loss weighting requires valid target counts")
    world_size = int(getattr(distributed_context, "world_size", 1))
    if world_size <= 1:
        return local_mean_loss
    return local_mean_loss * (world_size * local_count / global_count)


def count_batch_tokens(batch: dict[str, torch.Tensor]) -> int:
    return count_content_tokens(batch)


def current_peak_memory_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return 0


def _model_shape_label(run: dict[str, Any]) -> Any:
    return run.get("model_shape_label", run.get("model_size_label"))


def _default_granularity_pattern_summary(config: dict[str, Any]) -> dict[str, Any]:
    return summarize_granularity_pattern_from_config(config)


def _default_correction_context(config: dict[str, Any]) -> dict[str, Any]:
    return summarize_correction_context_from_config(config)


def _runtime_granularity_artifacts(
    config: dict[str, Any],
    model,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_model = getattr(model, "module", model)
    runtime_pattern = getattr(target_model, "current_granularity_pattern", None)
    if str(config["run"].get("sampling_mode", "nested-random")) == "nested-all":
        runtime_pattern = None
    runtime_pattern_summary = summarize_runtime_granularity_pattern_from_config(
        config,
        runtime_pattern=runtime_pattern,
    )
    correction_context = summarize_correction_context_from_config(
        config,
        granularity_pattern=runtime_pattern,
    )
    return runtime_pattern_summary, correction_context
