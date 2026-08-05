"""Config-driven training flow for MatFormer reproduction runs."""

from __future__ import annotations

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*args, **kwargs):
        return None

load_dotenv()

import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn.utils import clip_grad_norm_
from transformers import get_scheduler

import src.training.checkpointing as training_checkpointing
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
    sum_int,
)
from src.training.monitoring import NoopHeartbeatWriter
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
from src.utils.reproducibility import dedicated_random, seed_for


def build_optimizer_and_scheduler(model, training: Mapping[str, Any]):
    """Build the training optimizer and scheduler from resolved config fields."""
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
        num_training_steps=int(training["max_steps"]),
        **scheduler_kwargs,
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
) -> list[dict[str, Any]]:
    training = config["training"]
    run = config["run"]
    granularities = list(config["model"]["granularities"])
    model_sampling_mode = str(config["model"].get("granularity_sampling_mode", "global"))
    run_sampling_mode = str(run.get("sampling_mode", "nested-random"))
    target_model = model.module if hasattr(model, "module") else model
    supports_layer_granularities = hasattr(target_model, "configure_layer_granularities")
    token_budget = training["token_budget"]
    max_steps = training["max_steps"]
    validation_config = config.get("evaluation", {}).get("validation", {})
    eval_interval = int(validation_config.get("interval_steps", 0))

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
    checkpoint_state = checkpoint_state if checkpoint_state is not None else {}
    latest_checkpoint_path = Path(
        run_state.get("latest_checkpoint_path")
        or Path(config["run"]["output_dir"]) / "checkpoints" / "latest.pt"
    )
    resume_epoch = epoch
    resume_batch_index = max(0, resume_batch_index)
    if len(train_dataloader) > 0:
        while resume_batch_index >= len(train_dataloader):
            resume_batch_index -= len(train_dataloader)
            resume_epoch += 1
    else:
        resume_batch_index = 0
    run_state["epoch"] = resume_epoch
    run_state["batch_index"] = resume_batch_index
    run_state["content_tokens_seen"] = content_tokens_seen
    run_state.setdefault("latest_checkpoint_step", int(run_state.get("last_completed_step", 0)))
    run_state.setdefault("status", "fresh")
    if continuation_latest_checkpoint_policy(config)["enabled"] and not run_state.get("latest_checkpoint_path"):
        run_state["latest_checkpoint_path"] = str(latest_checkpoint_path)

    adaptive_sampler_state = _prepare_adaptive_sampler_runtime_state(
        config,
        run_state,
    )

    model.train()
    with heartbeat_stage(heartbeat_writer, stage_name):
        while step < max_steps and tokens_seen < token_budget:
            set_dataloader_epoch(train_dataloader, epoch)
            made_progress = False
            current_epoch = epoch
            epoch += 1
            for batch_index_in_epoch, batch in enumerate(train_dataloader):
                if current_epoch == resume_epoch and batch_index_in_epoch < resume_batch_index:
                    continue
                if step >= max_steps or tokens_seen >= token_budget:
                    break

                made_progress = True
                step += 1
                batch = move_batch_to_device(batch, device)
                if count_valid_prediction_targets(batch) <= 0:
                    raise ValueError(
                        "Training batch contains zero valid causal prediction targets"
                    )
                content_tokens_seen += global_content_tokens_for_batch(
                    batch,
                    device=device,
                    distributed_context=distributed_context,
                )
                tokens_seen = budget_tokens_seen_for_step(config, step)

                optimizer.zero_grad(set_to_none=True)

                step_metric_rows_data: list[
                    tuple[str, float, dict[str, Any], dict[str, Any]]
                ] = []
                if run_sampling_mode == "nested-all":
                    selected_granularities = select_training_granularities(
                        config,
                        granularities,
                        device,
                    )
                    detached_losses: list[float] = []
                    total_losses = len(selected_granularities)
                    for granularity in selected_granularities:
                        configure_model_granularity(model, granularity)
                        step_runtime_pattern_summary, step_correction_context = _runtime_granularity_artifacts(
                            config,
                            model,
                        )
                        with autocast_context(config, device):
                            outputs = model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch.get("attention_mask"),
                                labels=batch["labels"],
                            )
                        detached_loss = float(outputs.loss.detach().float().cpu().item())
                        detached_losses.append(detached_loss)
                        step_metric_rows_data.append(
                            (
                                granularity,
                                detached_loss,
                                step_runtime_pattern_summary,
                                step_correction_context,
                            )
                        )
                        (outputs.loss / total_losses).backward()
                    combined_loss_value = sum(detached_losses) / total_losses
                elif model_sampling_mode == "adaptive_global" and probabilistic_controller is not None:
                    selected_layer_granularities = (
                        probabilistic_global_layer_granularities(
                            config,
                            probabilistic_controller,
                        )
                    )
                    configure_model_granularity(
                        model,
                        selected_layer_granularities[0],
                    )
                    step_runtime_pattern_summary, step_correction_context = _runtime_granularity_artifacts(
                        config,
                        model,
                    )
                    with autocast_context(config, device):
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch.get("attention_mask"),
                            labels=batch["labels"],
                        )
                    combined_loss = outputs.loss
                    combined_loss_value = float(
                        combined_loss.detach().float().cpu().item()
                    )
                    step_metric_rows_data.append(
                        (
                            selected_layer_granularities[0],
                            combined_loss_value,
                            step_runtime_pattern_summary,
                            step_correction_context,
                        )
                    )
                    total_losses = 1
                elif model_sampling_mode == "adaptive_per_block" and supports_layer_granularities:
                    if adaptive_sampler_state is None:
                        raise ConfigError(
                            "adaptive_per_block runs require adaptive sampler state"
                        )
                    selected_layer_granularities = select_adaptive_sampler_layer_granularities(
                        adaptive_sampler_state,
                        block_count=int(config["model"]["num_layers"]),
                        step=step,
                        phase=stage_name,
                        granularities=granularities,
                        adaptive_seed=seed_for(config, "adaptive_sampling"),
                    )
                    configure_model_layer_granularities(
                        model,
                        selected_layer_granularities,
                    )
                    step_runtime_pattern_summary, step_correction_context = _runtime_granularity_artifacts(
                        config,
                        model,
                    )
                    with autocast_context(config, device):
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch.get("attention_mask"),
                            labels=batch["labels"],
                        )
                    combined_loss = outputs.loss
                    combined_loss_value = float(
                        combined_loss.detach().float().cpu().item()
                    )
                    step_metric_rows_data.append(
                        (
                            ",".join(selected_layer_granularities),
                            combined_loss_value,
                            step_runtime_pattern_summary,
                            step_correction_context,
                        )
                    )
                    total_losses = 1
                elif model_sampling_mode == "per_block" and supports_layer_granularities:
                    selected_layer_granularities = select_training_layer_granularities(
                        config,
                        granularities,
                        device,
                    )
                    configure_model_layer_granularities(
                        model,
                        selected_layer_granularities,
                    )
                    step_runtime_pattern_summary, step_correction_context = _runtime_granularity_artifacts(
                        config,
                        model,
                    )
                    with autocast_context(config, device):
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch.get("attention_mask"),
                            labels=batch["labels"],
                        )
                    combined_loss = outputs.loss
                    combined_loss_value = float(
                        combined_loss.detach().float().cpu().item()
                    )
                    step_metric_rows_data.append(
                        (
                            ",".join(selected_layer_granularities),
                            combined_loss_value,
                            step_runtime_pattern_summary,
                            step_correction_context,
                        )
                    )
                    total_losses = 1
                else:
                    selected_granularities = select_training_granularities(
                        config,
                        granularities,
                        device,
                    )
                    forward_losses: list[torch.Tensor] = []
                    for granularity in selected_granularities:
                        configure_model_granularity(model, granularity)
                        step_runtime_pattern_summary, step_correction_context = _runtime_granularity_artifacts(
                            config,
                            model,
                        )
                        with autocast_context(config, device):
                            outputs = model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch.get("attention_mask"),
                                labels=batch["labels"],
                            )
                        forward_losses.append(outputs.loss)
                        detached_loss = float(
                            outputs.loss.detach().float().cpu().item()
                        )
                        step_metric_rows_data.append(
                            (
                                granularity,
                                detached_loss,
                                step_runtime_pattern_summary,
                                step_correction_context,
                            )
                        )
                    combined_loss = (
                        forward_losses[0]
                        if len(forward_losses) == 1
                        else torch.stack(forward_losses).mean()
                    )
                    total_losses = len(forward_losses)
                    combined_loss_value = float(
                        combined_loss.detach().float().cpu().item()
                    )

                if run_sampling_mode != "nested-all":
                    combined_loss.backward()

                gradient_clip_norm = training.get("gradient_clip_norm")
                if gradient_clip_norm is not None:
                    clip_grad_norm_(model.parameters(), float(gradient_clip_norm))

                _maybe_apply_concat_lmc_optimizer_step(
                    config,
                    model,
                    optimizer,
                    total_losses=total_losses,
                )
                if probabilistic_controller is not None:
                    probabilistic_controller.record_successful_optimizer_step()
                scheduler.step()
                run_state.update(
                    {
                        "last_completed_step": step,
                        "step": step,
                        "epoch": current_epoch,
                        "batch_index": batch_index_in_epoch + 1,
                        "tokens_seen": tokens_seen,
                        "content_tokens_seen": content_tokens_seen,
                    }
                )
                if probabilistic_boundary_callback is not None:
                    probabilistic_boundary_callback(
                        step=step,
                        tokens_seen=tokens_seen,
                    )

                elapsed = time.time() - start_time
                peak_memory_bytes = current_peak_memory_bytes(device)
                latest_loss = combined_loss_value
                if model_sampling_mode == "adaptive_per_block" and adaptive_sampler_state is not None:
                    _update_adaptive_sampler_runtime_state(
                        config,
                        run_state,
                        adaptive_sampler_state,
                        phase=stage_name,
                        latest_loss=latest_loss,
                        selected_layer_granularities=selected_layer_granularities,
                        step=step,
                        epoch=current_epoch,
                    )
                elif model_sampling_mode != "adaptive_per_block":
                    run_state.pop("adaptive_sampler_previous_loss", None)
                    run_state.pop("adaptive_sampler_previous_pattern", None)
                    run_state.pop("adaptive_reward_summary", None)
                    run_state.pop("adaptive_correction_penalty_summary", None)
                tokens_per_second = tokens_seen / elapsed if elapsed > 0 else None
                adaptive_artifacts = _runtime_sampler_artifact_fields(
                    config,
                    run_state,
                    probabilistic_controller,
                )
                step_metric_rows = []
                run_state.update(
                    {
                        "status": "resumed" if int(run_state.get("resume_count", 0)) > 0 else "fresh",
                        "last_completed_step": step,
                        "step": step,
                        "epoch": current_epoch,
                        "batch_index": batch_index_in_epoch + 1,
                        "tokens_seen": tokens_seen,
                        "content_tokens_seen": content_tokens_seen,
                    }
                )
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
                )

                for (
                    granularity,
                    loss_value,
                    step_runtime_pattern_summary,
                    step_correction_context,
                ) in step_metric_rows_data:
                    step_metric_rows.append(
                        build_training_metric_row(
                            config,
                            step=step,
                            granularity=granularity,
                            loss=loss_value,
                            tokens_seen=tokens_seen,
                            content_tokens_seen=content_tokens_seen,
                            wall_clock_seconds=elapsed,
                            peak_memory_bytes=peak_memory_bytes,
                            granularity_pattern_summary=step_runtime_pattern_summary,
                            correction_context=step_correction_context,
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
                if (
                    metrics_journal is not None
                    and training_checkpointing.should_save_latest_checkpoint(
                        config,
                        step,
                        "step",
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

                if (
                    bool(validation_config.get("enabled", False))
                    and eval_interval > 0
                    and step % eval_interval == 0
                ):
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

                if step >= max_steps or tokens_seen >= token_budget:
                    break
            if not made_progress:
                break

    if stage_name == "training" and probabilistic_completion_callback is not None:
        probabilistic_completion_callback(step=step, tokens_seen=tokens_seen)
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
    if probabilistic_controller is not None:
        fields.update(
            build_compact_controller_metric_fields(
                probabilistic_controller.state_dict(),
                run_state.get("latest_controller_event"),
            )
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

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        selected_index = torch.empty((), dtype=torch.long, device=device)
        if torch.distributed.get_rank() == 0:
            selected_index.fill_(
                dedicated_random(config, "granularity_selection").randrange(
                    granularity_count
                )
            )
        torch.distributed.broadcast(selected_index, src=0)
        return int(selected_index.item())

    return dedicated_random(config, "granularity_selection").randrange(
        granularity_count
    )


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
        "wall_clock_seconds": wall_clock_seconds,
        "tokens_per_second": tokens_per_second,
        "peak_memory_bytes": peak_memory_bytes,
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
