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
import src.training.checkpointing as training_checkpointing
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
def load_model_and_optimizer_state(
    model,
    optimizer,
    model_state_dict: Mapping[str, Any],
    optimizer_state_dict: Mapping[str, Any] | None,
    distributed_context=None,
) -> None:
    if (
        distributed_context is not None
        and distributed_context.enabled
        and distributed_context.strategy == "fsdp"
    ):
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_state_dict,
        )

        set_state_dict(
            model,
            optimizer if optimizer is not None else [],
            model_state_dict=model_state_dict,
            optim_state_dict=optimizer_state_dict or {},
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        return

    model.load_state_dict(dict(model_state_dict))
    if optimizer is not None and optimizer_state_dict is not None:
        optimizer.load_state_dict(dict(optimizer_state_dict))


def best_validation_metric_value(
    validation_results: list[dict[str, Any]],
) -> tuple[str | None, float | None]:
    loss_values = [
        float(result["loss"])
        for result in validation_results
        if result.get("loss") is not None
    ]
    if loss_values:
        return "validation_loss", min(loss_values)

    perplexity_values = [
        float(result["perplexity"])
        for result in validation_results
        if result.get("perplexity") is not None
    ]
    if perplexity_values:
        return "validation_perplexity", min(perplexity_values)

    return None, None


def build_dataloaders(
    config: dict[str, Any],
    tokenized_dataset,
    device: torch.device,
    distributed_context=None,
):
    training = config["training"]
    batch_size = training["batch_size_per_process"]
    eval_batches = training.get("eval_batches", 1)
    eval_example_count = max(1, eval_batches * batch_size)

    train_dataset, eval_dataset = split_train_eval_dataset(
        tokenized_dataset,
        eval_example_count,
    )
    if len(train_dataset) == 0:
        train_dataset = eval_dataset

    pin_memory = device.type == "cuda"
    train_sampler = build_distributed_sampler(
        train_dataset,
        distributed_context,
        shuffle=True,
        seed=config["run"].get("seed"),
    )
    eval_sampler = build_distributed_sampler(
        eval_dataset,
        distributed_context,
        shuffle=False,
        seed=config["run"].get("seed"),
    )
    train_dataloader = build_language_model_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=training.get("dataloader_num_workers", 0),
        pin_memory=pin_memory,
    )
    eval_dataloader = build_language_model_dataloader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=eval_sampler,
        num_workers=training.get("dataloader_num_workers", 0),
        pin_memory=pin_memory,
    )
    return train_dataloader, eval_dataloader


def build_distributed_sampler(
    dataset,
    distributed_context,
    shuffle: bool,
    seed: int | None,
):
    if distributed_context is None or not distributed_context.enabled:
        return None

    return DistributedSampler(
        dataset,
        num_replicas=distributed_context.world_size,
        rank=distributed_context.rank,
        shuffle=shuffle,
        seed=0 if seed is None else int(seed),
    )


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
) -> list[dict[str, Any]]:
    training = config["training"]
    run = config["run"]
    granularities = config["model"]["granularities"]
    model_sampling_mode = str(config["model"].get("granularity_sampling_mode", "global"))
    run_sampling_mode = str(run.get("sampling_mode", "nested-random"))
    target_model = model.module if hasattr(model, "module") else model
    supports_layer_granularities = hasattr(target_model, "configure_layer_granularities")
    token_budget = training["token_budget"]
    max_steps = training["max_steps"]
    eval_interval = training.get("eval_interval", 0)

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
                content_tokens_seen += global_content_tokens_for_batch(
                    batch,
                    device=device,
                    distributed_context=distributed_context,
                )
                tokens_seen = budget_tokens_seen_for_step(config, step)

                optimizer.zero_grad(set_to_none=True)

                step_metric_rows_data: list[
                    tuple[str, torch.Tensor, dict[str, Any], dict[str, Any]]
                ] = []
                if run_sampling_mode == "nested-all":
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
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch.get("attention_mask"),
                            labels=batch["labels"],
                        )
                        forward_losses.append(outputs.loss)
                        step_metric_rows_data.append(
                            (
                                granularity,
                                outputs.loss,
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
                    )
                    configure_model_layer_granularities(
                        model,
                        selected_layer_granularities,
                    )
                    step_runtime_pattern_summary, step_correction_context = _runtime_granularity_artifacts(
                        config,
                        model,
                    )
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch["labels"],
                    )
                    combined_loss = outputs.loss
                    for granularity in selected_layer_granularities:
                        step_metric_rows_data.append(
                            (
                                granularity,
                                combined_loss,
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
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch["labels"],
                    )
                    combined_loss = outputs.loss
                    for granularity in selected_layer_granularities:
                        step_metric_rows_data.append(
                            (
                                granularity,
                                combined_loss,
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
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch.get("attention_mask"),
                            labels=batch["labels"],
                        )
                        forward_losses.append(outputs.loss)
                        step_metric_rows_data.append(
                            (
                                granularity,
                                outputs.loss,
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
                scheduler.step()

                elapsed = time.time() - start_time
                peak_memory_bytes = current_peak_memory_bytes(device)
                latest_loss = float(combined_loss.detach().cpu().item())
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
                adaptive_artifacts = build_adaptive_sampler_artifact_fields(
                    config,
                    run_state,
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
                    loss,
                    step_runtime_pattern_summary,
                    step_correction_context,
                ) in step_metric_rows_data:
                    step_metric_rows.append(
                        build_training_metric_row(
                            config,
                            step=step,
                            granularity=granularity,
                            loss=float(loss.detach().cpu().item()),
                            tokens_seen=tokens_seen,
                            content_tokens_seen=content_tokens_seen,
                            wall_clock_seconds=elapsed,
                            peak_memory_bytes=peak_memory_bytes,
                            granularity_pattern_summary=step_runtime_pattern_summary,
                            correction_context=step_correction_context,
                            adaptive_artifacts=adaptive_artifacts,
                        )
                    )
                metrics_rows.extend(step_metric_rows)
                if monitoring_session is not None:
                    monitoring_session.log_rows(step_metric_rows)

                if eval_interval > 0 and step % eval_interval == 0:
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
                        )
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
                        adaptive_artifacts=build_adaptive_sampler_artifact_fields(
                            config,
                            run_state,
                        ),
                    )
                    metrics_rows.extend(validation_metric_rows)
                    if monitoring_session is not None:
                        monitoring_session.log_rows(validation_metric_rows)

                if step >= max_steps or tokens_seen >= token_budget:
                    break
            if not made_progress:
                break

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

    return metrics_rows


def select_training_granularities(
    config: dict[str, Any],
    granularities: list[str],
    device: torch.device,
) -> list[str]:
    if str(config["run"].get("sampling_mode", "nested-random")) == "nested-all":
        return list(granularities)

    sampling_mode = config["training"].get("granularity_sampling", "all")
    if sampling_mode == "all":
        return list(granularities)
    if sampling_mode == "random":
        selected_index = select_random_granularity_index(
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
    layer_count = int(config["model"]["num_layers"])
    if layer_count <= 0:
        raise ValueError("model.num_layers must be positive")

    return [
        granularities[
            select_random_granularity_index(
                granularity_count=len(granularities),
                device=device,
            )
        ]
        for _ in range(layer_count)
    ]


def select_random_granularity_index(
    granularity_count: int,
    device: torch.device,
) -> int:
    if granularity_count <= 0:
        raise ValueError("granularity_count must be positive")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        selected_index = torch.empty((), dtype=torch.long, device=device)
        if torch.distributed.get_rank() == 0:
            selected_index.fill_(random.randrange(granularity_count))
        torch.distributed.broadcast(selected_index, src=0)
        return int(selected_index.item())

    return random.randrange(granularity_count)


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
) -> None:
    if not config.get("evaluation", {}).get("validation", False):
        return
    has_final_validation = any(
        row["split"] == "validation" and row["step"] == step
        for row in metrics_rows
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
        )
    maybe_write_best_eval_checkpoint(
        config,
        model,
        validation_results,
        step,
        heartbeat_writer,
        checkpoint_state if checkpoint_state is not None else {},
        run_state if run_state is not None else training_checkpointing.build_initial_continuation_state(config),
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
        adaptive_artifacts=build_adaptive_sampler_artifact_fields(
            config,
            run_state if run_state is not None else {},
        ),
    )
    metrics_rows.extend(validation_metric_rows)
    if monitoring_session is not None:
        monitoring_session.log_rows(validation_metric_rows)


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


def count_batch_tokens(batch: dict[str, torch.Tensor]) -> int:
    return count_content_tokens(batch)


def current_peak_memory_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return 0


def set_random_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
