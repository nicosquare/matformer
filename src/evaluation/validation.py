"""Validation loss and perplexity helpers."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

import torch
import torch.distributed as dist

from src.utils.config import resolve_sampling_mode_from_config_sections
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
) -> dict[str, float | int | str | None]:
    was_training = model.training
    model.eval()

    loss_sum = 0.0
    batch_count = 0
    token_count = 0

    with torch.no_grad():
        configure_model_granularity(model, granularity)
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
            )
            loss_sum += outputs.loss.detach().float().item()
            batch_count += 1
            token_count += _count_tokens(batch)

    loss_sum, batch_count, token_count = _reduce_validation_stats(
        loss_sum,
        batch_count,
        token_count,
        device,
        distributed,
    )

    if was_training:
        model.train()

    loss = loss_sum / batch_count
    return {
        "granularity": granularity,
        "loss": loss,
        "perplexity": perplexity_from_loss(loss),
        "tokens_seen": token_count,
    }


def evaluate_validation_per_granularity(
    model,
    dataloader,
    granularities: list[str],
    device: torch.device | str,
    distributed: bool = False,
) -> list[dict[str, float | int | str | None]]:
    return [
        evaluate_validation_loss(
            model,
            dataloader,
            device=device,
            granularity=granularity,
            distributed=distributed,
        )
        for granularity in granularities
    ]


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
            "granularity_sampling_mode": model.get(
                "granularity_sampling_mode"
            ),
            "granularity": result["granularity"],
            "granularity_pattern_summary": json_artifact_value(
                granularity_pattern_summary
                if granularity_pattern_summary is not None
                else _default_granularity_pattern_summary(config)
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
            "wall_clock_seconds": wall_clock_seconds,
            "tokens_per_second": tokens_per_second,
            "peak_memory_bytes": peak_memory_bytes,
        }
        if adaptive_artifacts:
            row.update(adaptive_artifacts)
        rows.append(row)
    return rows


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


def _reduce_validation_stats(
    loss_sum: float,
    batch_count: int,
    token_count: int,
    device: torch.device | str,
    distributed: bool,
) -> tuple[float, int, int]:
    if batch_count == 0:
        raise ValueError("Validation dataloader produced zero batches")

    if not distributed:
        return loss_sum, batch_count, token_count

    stats = torch.tensor(
        [loss_sum, float(batch_count), float(token_count)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float(stats[0].item()), int(stats[1].item()), int(stats[2].item())


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
    local_correction_active = (
        sampling_mode == "per_block"
        and model.get("correction_mode") in {"gmc", "lmc"}
    )
    return {
        "correction_mode": model.get("correction_mode"),
        "sampling_mode": sampling_mode,
        "local_correction_active": local_correction_active,
        "derived_membership_pattern": (
            list(model.get("granularities", []))
            if local_correction_active
            else []
        ),
    }


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
