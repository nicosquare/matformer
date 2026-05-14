"""Artifact writers for MatFormer reproduction experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from utils.config import write_resolved_config


METRICS_COLUMNS = [
    "run_id",
    "step",
    "split",
    "model_family",
    "model_size_label",
    "granularity",
    "loss",
    "perplexity",
    "tokens_seen",
    "wall_clock_seconds",
    "tokens_per_second",
    "peak_memory_bytes",
]

TASK_RESULTS_COLUMNS = [
    "run_id",
    "suite_id",
    "task",
    "model_family",
    "model_size_label",
    "granularity",
    "metric_name",
    "metric_value",
]

SCALING_RESULTS_COLUMNS = [
    "comparison_id",
    "run_id",
    "model_family",
    "model_size_label",
    "completion_label",
    "granularity",
    "total_parameters",
    "embedding_parameters",
    "lm_head_parameters",
    "non_embedding_parameters",
    "loss",
    "perplexity",
    "average_downstream_accuracy",
]

CONSISTENCY_RESULTS_COLUMNS = [
    "comparison_id",
    "small_run_id",
    "large_run_id",
    "small_granularity",
    "large_granularity",
    "metric_name",
    "metric_value",
    "sample_count",
]

RUN_SUMMARY_FIELDS = [
    "run_id",
    "phase_id",
    "model_family",
    "model_size_label",
    "completion_label",
    "dataset_name",
    "dataset_split",
    "token_budget",
    "expected_tokens_per_step",
    "derived_max_steps",
    "effective_world_size",
    "tokens_seen",
    "stop_reason",
    "seed",
    "status",
    "output_root",
    "output_dir",
    "paper_aligned",
    "notes",
]

BASELINE_MATCH_FIELDS = [
    "match_id",
    "nested_run_id",
    "standalone_run_id",
    "granularity",
    "non_embedding_parameters_nested",
    "non_embedding_parameters_standalone",
    "match_notes",
]


class ArtifactError(ValueError):
    """Raised when an artifact would miss required analysis fields."""


def write_config_artifact(
    config: Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> Path:
    return write_resolved_config(config, output_dir=output_dir)


def build_run_summary(
    config: Mapping[str, Any],
    tokens_seen: int | None = None,
    status: str = "completed",
    notes: Iterable[str] | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run = config["run"]
    model = config["model"]
    training = config["training"]
    dataset = config["dataset"]

    if tokens_seen is None:
        tokens_seen = training.get("tokens_seen", training["token_budget"])
    stop_reason = "failed" if status == "failed" else "not_started"

    summary = {
        "run_id": run["run_id"],
        "phase_id": run["phase_id"],
        "model_family": run["model_family"],
        "model_size_label": run["model_size_label"],
        "completion_label": run["completion_label"],
        "dataset_name": dataset["dataset_name"],
        "dataset_split": dataset["dataset_split"],
        "token_budget": training["token_budget"],
        "expected_tokens_per_step": training["expected_tokens_per_step"],
        "derived_max_steps": training["derived_max_steps"],
        "effective_world_size": training["effective_world_size"],
        "tokens_seen": tokens_seen,
        "stop_reason": stop_reason,
        "seed": run.get("seed"),
        "status": status,
        "output_root": run["output_root"],
        "output_dir": run["output_dir"],
        "paper_aligned": bool(model["paper_aligned"]),
        "notes": list(notes or []),
    }

    if extra_fields:
        summary.update(extra_fields)

    _require_fields(summary, RUN_SUMMARY_FIELDS, "run_summary.json")
    return summary


def write_run_summary(
    output_dir: str | Path,
    summary: Mapping[str, Any],
    filename: str = "run_summary.json",
) -> Path:
    _require_fields(summary, RUN_SUMMARY_FIELDS, filename)
    return write_json_artifact(Path(output_dir) / filename, summary)


def write_failed_run_summary(
    config: Mapping[str, Any],
    error_message: str,
    output_dir: str | Path | None = None,
    tokens_seen: int = 0,
    notes: Iterable[str] | None = None,
) -> Path:
    failure_notes = [error_message]
    failure_notes.extend(notes or [])
    summary = build_run_summary(
        config,
        tokens_seen=tokens_seen,
        status="failed",
        notes=failure_notes,
    )
    run_output_dir = output_dir or config["run"]["output_dir"]
    return write_run_summary(run_output_dir, summary)


def write_metrics_csv(
    output_dir: str | Path,
    rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    append: bool = False,
) -> Path:
    return write_csv_artifact(
        Path(output_dir) / "metrics.csv",
        rows,
        METRICS_COLUMNS,
        append=append,
    )


def write_task_results_csv(
    output_dir: str | Path,
    rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    append: bool = False,
) -> Path:
    return write_csv_artifact(
        Path(output_dir) / "task_results.csv",
        rows,
        TASK_RESULTS_COLUMNS,
        append=append,
    )


def write_scaling_results_csv(
    output_dir: str | Path,
    rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    append: bool = False,
) -> Path:
    return write_csv_artifact(
        Path(output_dir) / "scaling_results.csv",
        rows,
        SCALING_RESULTS_COLUMNS,
        append=append,
    )


def write_consistency_results_csv(
    output_dir: str | Path,
    rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    append: bool = False,
) -> Path:
    return write_csv_artifact(
        Path(output_dir) / "consistency_results.csv",
        rows,
        CONSISTENCY_RESULTS_COLUMNS,
        append=append,
    )


def build_parameter_counts_by_granularity(
    model: Any,
    granularities: Iterable[str],
    trainable_only: bool = False,
) -> dict[str, dict[str, int]]:
    from utils.model_size import model_parameter_counts

    return {
        granularity: model_parameter_counts(
            model,
            trainable_only=trainable_only,
            granularity=granularity,
        )
        for granularity in granularities
    }


def build_scaling_result_rows(
    config: Mapping[str, Any],
    metrics_rows: Iterable[Mapping[str, Any]],
    parameter_counts_by_granularity: Mapping[str, Mapping[str, int]],
    comparison_id_prefix: str | None = None,
) -> list[dict[str, Any]]:
    run = config["run"]
    model = config["model"]
    metrics_rows = list(metrics_rows)
    latest_rows = latest_metric_rows_by_granularity(metrics_rows, split="validation")
    if not latest_rows:
        latest_rows = latest_metric_rows_by_granularity(metrics_rows, split="train")

    rows = []
    for granularity in model["granularities"]:
        metric_row = latest_rows.get(granularity)
        if metric_row is None:
            raise ArtifactError(
                "scaling_results.csv missing metric row for "
                f"granularity={granularity}"
            )

        parameter_counts = parameter_counts_by_granularity.get(granularity)
        if parameter_counts is None:
            raise ArtifactError(
                "scaling_results.csv missing parameter counts for "
                f"granularity={granularity}"
            )
        _require_fields(
            parameter_counts,
            [
                "total_parameters",
                "embedding_parameters",
                "lm_head_parameters",
                "non_embedding_parameters",
            ],
            "scaling_results.csv",
        )

        comparison_id = f"{comparison_id_prefix or run['run_id']}__{granularity}"
        rows.append(
            {
                "comparison_id": comparison_id,
                "run_id": run["run_id"],
                "model_family": run["model_family"],
                "model_size_label": run["model_size_label"],
                "completion_label": run["completion_label"],
                "granularity": granularity,
                "total_parameters": parameter_counts["total_parameters"],
                "embedding_parameters": parameter_counts["embedding_parameters"],
                "lm_head_parameters": parameter_counts["lm_head_parameters"],
                "non_embedding_parameters": parameter_counts[
                    "non_embedding_parameters"
                ],
                "loss": metric_row["loss"],
                "perplexity": metric_row["perplexity"],
                "average_downstream_accuracy": None,
            }
        )

    return rows


def build_baseline_match_row(
    nested_config: Mapping[str, Any],
    standalone_config: Mapping[str, Any],
    granularity: str,
    nested_counts: Mapping[str, int] | None = None,
    standalone_counts: Mapping[str, int] | None = None,
    match_notes: Iterable[str] | None = None,
) -> dict[str, Any]:
    nested_run = nested_config["run"]
    standalone_run = standalone_config["run"]
    row = {
        "match_id": baseline_match_id(
            nested_run["run_id"],
            standalone_run["run_id"],
            granularity,
        ),
        "nested_run_id": nested_run["run_id"],
        "standalone_run_id": standalone_run["run_id"],
        "granularity": granularity,
        "non_embedding_parameters_nested": _non_embedding_count(nested_counts),
        "non_embedding_parameters_standalone": _non_embedding_count(
            standalone_counts
        ),
        "match_notes": list(match_notes or []),
    }
    _require_fields(row, BASELINE_MATCH_FIELDS, "baseline match row")
    return row


def baseline_match_id(
    nested_run_id: str,
    standalone_run_id: str,
    granularity: str,
) -> str:
    return f"{nested_run_id}__{standalone_run_id}__{granularity}"


def latest_metric_rows_by_granularity(
    metrics_rows: Iterable[Mapping[str, Any]],
    split: str,
) -> dict[str, Mapping[str, Any]]:
    latest_rows: dict[str, tuple[int, int, Mapping[str, Any]]] = {}
    for row_index, row in enumerate(metrics_rows):
        if row.get("split") != split:
            continue
        granularity = row.get("granularity")
        if granularity is None:
            continue

        row_key = (_int_value(row.get("step")), row_index)
        current = latest_rows.get(str(granularity))
        if current is None or row_key > current[:2]:
            latest_rows[str(granularity)] = (*row_key, row)

    return {
        granularity: row_with_key[2]
        for granularity, row_with_key in latest_rows.items()
    }


def write_json_artifact(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return output_path


def write_csv_artifact(
    path: str | Path,
    rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    columns: list[str],
    append: bool = False,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_rows = _normalize_rows(rows)
    for row in normalized_rows:
        _require_fields(row, columns, str(output_path))

    file_exists = output_path.exists()
    should_write_header = not append or not file_exists or output_path.stat().st_size == 0
    mode = "a" if append else "w"

    with output_path.open(mode, encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns, extrasaction="ignore")
        if should_write_header:
            writer.writeheader()
        writer.writerows(normalized_rows)

    return output_path


def _normalize_rows(
    rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if isinstance(rows, Mapping):
        return [rows]
    return list(rows)


def _non_embedding_count(counts: Mapping[str, int] | None):
    if counts is None:
        return None
    return counts.get("non_embedding_parameters")


def _require_fields(
    row: Mapping[str, Any],
    required_fields: list[str],
    artifact_name: str,
) -> None:
    missing_fields = [field_name for field_name in required_fields if field_name not in row]
    if missing_fields:
        raise ArtifactError(f"{artifact_name} missing fields: {missing_fields}")


def _int_value(value: Any) -> int:
    if value in (None, ""):
        return -1
    return int(value)
