"""Validation loss and perplexity helpers."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

import torch
import torch.distributed as dist

from src.training.distributed import autocast_context
from src.utils.config import resolve_sampling_mode_from_config_sections
from src.models.granularity import resolved_granularity_artifact_fields
from src.utils.metrics import json_artifact_value


def perplexity_from_loss(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for name, value in batch.items()
    }


def configure_model_granularity(model, granularity: str | None) -> None:
    if granularity is None:
        return

    target = model.module if hasattr(model, "module") else model
    if hasattr(target, "current_layer_granularities"):
        target.current_layer_granularities = None
    if hasattr(target, "current_granularity_pattern"):
        target.current_granularity_pattern = None
    if hasattr(target, "current_sampling_mode"):
        target.current_sampling_mode = "global"
    configure_subnetwork = getattr(target, "configure_subnetwork", None)
    if configure_subnetwork is not None:
        configure_subnetwork(granularity)


def evaluate_validation_loss(
    model,
    dataloader,
    device: torch.device | str,
    granularity: str | None = None,
    distributed: bool = False,
    config: dict[str, Any] | None = None,
) -> dict[str, float | int | str | None]:
    was_training = model.training
    model.eval()

    total_nll = 0.0
    batch_count = 0
    example_count = 0
    target_count = 0
    skipped_batch_count = 0

    with torch.no_grad():
        configure_model_granularity(model, granularity)
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            batch_count += 1
            example_count += int(batch["input_ids"].shape[0])
            valid_targets = count_valid_prediction_targets(batch)
            if valid_targets == 0:
                skipped_batch_count += 1
                continue
            with autocast_context(config or {}, device):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
            total_nll += float(outputs.loss.detach().double().item()) * valid_targets
            target_count += valid_targets

    total_nll, batch_count, example_count, target_count, skipped_batch_count = (
        _reduce_validation_stats(
            total_nll,
            batch_count,
            example_count,
            target_count,
            skipped_batch_count,
            device,
            distributed,
        )
    )

    if was_training:
        model.train()

    if target_count == 0:
        raise ValueError(
            "Validation holdout contains zero valid causal prediction targets"
        )
    loss = total_nll / target_count
    return {
        "granularity": granularity,
        "loss": loss,
        "perplexity": perplexity_from_loss(loss),
        "tokens_seen": target_count,
        "evaluation_examples": example_count,
        "evaluation_batches": batch_count,
        "evaluation_target_tokens": target_count,
        "evaluation_skipped_batches": skipped_batch_count,
        "validation_loss_aggregation": "target_token_weighted_causal_shift_float64",
    }


def evaluate_validation_per_granularity(
    model,
    dataloader,
    granularities: list[str],
    device: torch.device | str,
    distributed: bool = False,
    config: dict[str, Any] | None = None,
) -> list[dict[str, float | int | str | None]]:
    runtime_state = _capture_runtime_granularity_state(model)
    try:
        return [
            evaluate_validation_loss(
                model,
                dataloader,
                device=device,
                granularity=granularity,
                distributed=distributed,
                config=config,
            )
            for granularity in granularities
        ]
    finally:
        _restore_runtime_granularity_state(model, runtime_state)


def evaluate_fixed_panel_objective(
    model,
    dataloader,
    granularities: list[str],
    device: torch.device | str,
    distributed: bool = False,
    config: dict[str, Any] | None = None,
    controller_manifest_hash: str | None = None,
    boundary_step: int | None = None,
) -> dict[str, Any]:
    """Evaluate the uniform all-granularity objective on one fixed panel."""

    ordered_granularities = [str(granularity) for granularity in granularities]
    if not ordered_granularities:
        raise ValueError("Controller objective requires at least one granularity")
    if len(set(ordered_granularities)) != len(ordered_granularities):
        raise ValueError("Controller objective granularities must be unique")

    component_results = evaluate_validation_per_granularity(
        model,
        dataloader,
        ordered_granularities,
        device,
        distributed=distributed,
        config=config,
    )
    component_losses = []
    for expected_granularity, result in zip(
        ordered_granularities,
        component_results,
        strict=True,
    ):
        if result.get("granularity") != expected_granularity:
            raise ValueError(
                "Controller objective granularity order changed during evaluation"
            )
        component_loss = float(result["loss"])
        if not math.isfinite(component_loss):
            raise ValueError(
                "Controller objective produced a non-finite component loss for "
                f"granularity {expected_granularity!r}"
            )
        component_losses.append(component_loss)

    uniform_objective = math.fsum(component_losses) / len(component_losses)
    if not math.isfinite(uniform_objective):
        raise ValueError("Controller objective is non-finite")

    evaluation_example_counts = {
        int(result["evaluation_examples"]) for result in component_results
    }
    evaluation_target_counts = {
        int(result["evaluation_target_tokens"]) for result in component_results
    }
    if len(evaluation_example_counts) != 1 or len(evaluation_target_counts) != 1:
        raise ValueError(
            "Controller objective component evaluations used inconsistent panel counts"
        )

    return {
        "boundary_step": boundary_step,
        "split": "controller",
        "ordered_granularities": ordered_granularities,
        "ordered_component_losses": component_losses,
        "objective": uniform_objective,
        "uniform_objective": uniform_objective,
        "evaluation_example_count": next(iter(evaluation_example_counts)),
        "evaluation_target_tokens": next(iter(evaluation_target_counts)),
        "aggregation_method": "target_token_weighted_causal_shift_float64",
        "objective_weighting": "uniform",
        "controller_manifest_hash": controller_manifest_hash,
        "evaluation_status": "complete",
        "component_results": component_results,
    }


# Keep both descriptive call-site names available while controller lifecycle code lands.
evaluate_controller_objective = evaluate_fixed_panel_objective
evaluate_controller_panel_objective = evaluate_fixed_panel_objective


def _capture_runtime_granularity_state(model) -> dict[str, Any]:
    target = model.module if hasattr(model, "module") else model
    layer_granularities = getattr(target, "current_layer_granularities", None)
    return {
        "current_granularity": getattr(target, "current_granularity", None),
        "current_layer_granularities": (
            None if layer_granularities is None else list(layer_granularities)
        ),
        "current_granularity_pattern": getattr(
            target,
            "current_granularity_pattern",
            None,
        ),
        "current_sampling_mode": getattr(target, "current_sampling_mode", None),
    }


def _restore_runtime_granularity_state(model, state: Mapping[str, Any]) -> None:
    target = model.module if hasattr(model, "module") else model
    pattern = state.get("current_granularity_pattern")
    layer_granularities = state.get("current_layer_granularities")
    pattern_type = getattr(pattern, "pattern_type", None)

    if layer_granularities and pattern_type == "per_block":
        configure_layers = getattr(target, "configure_layer_granularities", None)
        if configure_layers is not None:
            configure_layers(layer_granularities)
    elif state.get("current_granularity") is not None:
        configure_subnetwork = getattr(target, "configure_subnetwork", None)
        if configure_subnetwork is not None:
            configure_subnetwork(state["current_granularity"])

    for field_name, value in state.items():
        if hasattr(target, field_name):
            setattr(target, field_name, value)


def validation_results_to_metric_rows(
    results: list[dict[str, Any]],
    config: dict[str, Any],
    step: int,
    split: str = "validation",
    wall_clock_seconds: float | None = None,
    tokens_per_second: float | None = None,
    peak_memory_bytes: int | None = None,
    tokens_seen: int | None = None,
    content_tokens_seen: int | None = None,
    granularity_pattern_summary: dict[str, Any] | None = None,
    correction_context: dict[str, Any] | None = None,
    adaptive_artifacts: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    run = config["run"]
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        model = {}
    training = config.get("training", {})
    if not isinstance(training, Mapping):
        training = {}
    rows = []
    for result in results:
        row = {
            "run_id": run["run_id"],
            "step": step,
            "split": split,
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
            "granularity": result["granularity"],
            **resolved_granularity_artifact_fields(model),
            "granularity_pattern_summary": json_artifact_value(
                _evaluated_granularity_pattern_summary(
                    config,
                    str(result["granularity"]),
                    base_summary=granularity_pattern_summary,
                )
            ),
            "correction_context": json_artifact_value(
                correction_context
                if correction_context is not None
                else _default_correction_context(config)
            ),
            "loss": result["loss"],
            "perplexity": result["perplexity"],
            "tokens_seen": (
                result["tokens_seen"] if tokens_seen is None else tokens_seen
            ),
            "content_tokens_seen": (
                result["tokens_seen"]
                if content_tokens_seen is None
                else content_tokens_seen
            ),
            "evaluation_examples": result.get("evaluation_examples"),
            "evaluation_batches": result.get("evaluation_batches"),
            "evaluation_target_tokens": result.get("evaluation_target_tokens"),
            "evaluation_skipped_batches": result.get("evaluation_skipped_batches"),
            "validation_manifest_hash": config.get("validation_manifest_hash"),
            "validation_loss_aggregation": result.get(
                "validation_loss_aggregation",
                config.get("validation_loss_aggregation"),
            ),
            "comparison_control_signature": config.get("comparison_control_signature"),
            "wall_clock_seconds": wall_clock_seconds,
            "tokens_per_second": tokens_per_second,
            "peak_memory_bytes": peak_memory_bytes,
        }
        if adaptive_artifacts:
            row.update(adaptive_artifacts)
        rows.append(row)
    return rows


def _evaluated_granularity_pattern_summary(
    config: Mapping[str, Any],
    granularity: str,
    *,
    base_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Describe the subnetwork evaluated for this specific validation row."""

    summary = dict(
        base_summary
        if isinstance(base_summary, Mapping)
        else _default_granularity_pattern_summary(dict(config))
    )
    run = config.get("run", {})
    run_id = str(run.get("run_id") or "") if isinstance(run, Mapping) else ""
    summary.update(
        {
            "pattern_type": "single",
            "selected_granularities": [granularity],
            "repeatable_source": [
                run_id,
                f"validation.granularity={granularity}",
            ],
        }
    )
    return summary


def aggregate_scaling_summary(
    scaling_rows: Iterable[Mapping[str, Any]],
    task_result_rows: Iterable[Mapping[str, Any]],
    accuracy_metric_names: Iterable[str] = (
        "accuracy",
        "acc",
        "acc_norm",
        "acc,none",
        "acc_norm,none",
        "exact_match,none",
    ),
) -> list[dict[str, Any]]:
    downstream_accuracy = average_downstream_accuracy_by_run_granularity(
        task_result_rows,
        accuracy_metric_names=accuracy_metric_names,
    )
    aggregated_rows = []

    for row in scaling_rows:
        aggregated_row = dict(row)
        run_id = str(row.get("run_id") or "")
        granularity = str(row.get("granularity") or "")
        accuracy = downstream_accuracy.get((run_id, granularity))
        if accuracy is None:
            accuracy = downstream_accuracy.get((run_id, ""))
        if accuracy is not None:
            aggregated_row["average_downstream_accuracy"] = accuracy
        aggregated_rows.append(aggregated_row)

    return aggregated_rows


def average_downstream_accuracy_by_run_granularity(
    task_result_rows: Iterable[Mapping[str, Any]],
    accuracy_metric_names: Iterable[str] = (
        "accuracy",
        "acc",
        "acc_norm",
        "acc,none",
        "acc_norm,none",
        "exact_match,none",
    ),
) -> dict[tuple[str, str], float]:
    allowed_metric_names = {str(metric_name) for metric_name in accuracy_metric_names}
    values_by_run_granularity: dict[tuple[str, str], list[float]] = defaultdict(list)

    for row in task_result_rows:
        metric_name = str(row.get("metric_name") or "")
        if metric_name not in allowed_metric_names:
            continue
        run_id = str(row.get("run_id") or "")
        if not run_id:
            continue
        metric_value = _float_or_none(row.get("metric_value"))
        if metric_value is None:
            continue
        granularity = str(row.get("granularity") or "")
        values_by_run_granularity[(run_id, granularity)].append(metric_value)

    return {
        key: sum(values) / len(values)
        for key, values in values_by_run_granularity.items()
        if values
    }


def _count_tokens(batch: dict[str, torch.Tensor]) -> int:
    if "attention_mask" in batch and batch["attention_mask"] is not None:
        return int(batch["attention_mask"].sum().item())

    labels = batch["labels"]
    return int((labels != -100).sum().item())


def count_valid_prediction_targets(batch: dict[str, torch.Tensor]) -> int:
    labels = batch["labels"]
    if labels.ndim < 2 or labels.shape[1] <= 1:
        return 0
    return int((labels[:, 1:] != -100).sum().item())


def _reduce_validation_stats(
    total_nll: float,
    batch_count: int,
    example_count: int,
    target_count: int,
    skipped_batch_count: int,
    device: torch.device | str,
    distributed: bool,
) -> tuple[float, int, int, int, int]:
    if batch_count == 0 and not distributed:
        raise ValueError("Validation dataloader produced zero batches")

    if not distributed:
        return total_nll, batch_count, example_count, target_count, skipped_batch_count

    stats = torch.tensor(
        [
            total_nll,
            float(batch_count),
            float(example_count),
            float(target_count),
            float(skipped_batch_count),
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (
        float(stats[0].item()),
        int(stats[1].item()),
        int(stats[2].item()),
        int(stats[3].item()),
        int(stats[4].item()),
    )


def _model_shape_label(run: dict[str, Any]) -> Any:
    return run.get("model_shape_label", run.get("model_size_label"))


def _default_granularity_pattern_summary(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model", {})
    run = config.get("run", {})
    if not isinstance(model, Mapping):
        model = {}
    if not isinstance(run, Mapping):
        run = {}
    training = config.get("training", {})
    if not isinstance(training, Mapping):
        training = {}
    resolved_run_mode = resolve_sampling_mode_from_config_sections(run, training)
    sampling_mode = str(model.get("granularity_sampling_mode", "global"))
    if resolved_run_mode == "nested-all":
        pattern_type = "all_granularities"
    elif sampling_mode == "per_block":
        pattern_type = "per_block"
    else:
        pattern_type = "single"

    selected_granularities = list(model.get("granularities", []))
    if resolved_run_mode == "standalone" and run.get("granularity") is not None:
        selected_granularities = [str(run["granularity"])]

    repeatable_source = [
        str(run.get("run_id") or ""),
        f"run.sampling_mode={resolved_run_mode}",
        f"model.granularity_sampling_mode={sampling_mode}",
    ]
    if resolved_run_mode == "standalone" and run.get("granularity") is not None:
        repeatable_source.append(f"run.granularity={run['granularity']}")

    return {
        "pattern_type": pattern_type,
        "selected_granularities": selected_granularities,
        "layer_count": model.get("num_layers"),
        "repeatable_source": repeatable_source,
    }


def _default_correction_context(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        model = {}
    sampling_mode = str(model.get("granularity_sampling_mode", "global"))
    local_correction_active = sampling_mode == "per_block" and model.get(
        "correction_mode"
    ) in {"gmc", "lmc"}
    return {
        "correction_mode": model.get("correction_mode"),
        "sampling_mode": sampling_mode,
        "local_correction_active": local_correction_active,
        "derived_membership_pattern": (
            list(model.get("granularities", [])) if local_correction_active else []
        ),
    }


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
