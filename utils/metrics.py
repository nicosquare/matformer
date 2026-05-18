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
    "model_shape_label",
    "sampling_mode",
    "granularity",
    "loss",
    "perplexity",
    "tokens_seen",
    "content_tokens_seen",
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
    "model_shape_label",
    "sampling_mode",
    "granularity",
    "metric_name",
    "metric_value",
]

SCALING_RESULTS_COLUMNS = [
    "comparison_id",
    "run_id",
    "model_family",
    "model_size_label",
    "model_shape_label",
    "sampling_mode",
    "completion_label",
    "granularity",
    "d_model",
    "num_layers",
    "num_attention_heads",
    "context_length",
    "vocab_size_assumption",
    "token_budget",
    "effective_world_size",
    "total_parameters",
    "embedding_parameters",
    "lm_head_parameters",
    "non_embedding_parameters",
    "ffn_parameters",
    "attention_parameters",
    "other_non_embedding_parameters",
    "lm_head_counting",
    "checkpoint_path",
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
    "model_shape_label",
    "sampling_mode",
    "completion_label",
    "dataset_name",
    "dataset_split",
    "token_budget",
    "expected_tokens_per_step",
    "derived_max_steps",
    "effective_world_size",
    "tokens_seen",
    "content_tokens_seen",
    "stop_reason",
    "seed",
    "status",
    "output_root",
    "output_dir",
    "d_model",
    "num_layers",
    "num_attention_heads",
    "context_length",
    "vocab_size_assumption",
    "parameter_counts",
    "parameter_counts_by_granularity",
    "checkpoint_status",
    "best_checkpoint_path",
    "final_checkpoint_path",
    "checkpoint_metric",
    "checkpoint_metric_value",
    "checkpoint_selection_step",
    "checkpoint_unavailable_reason",
    "notes",
]

PARAMETER_COUNT_FIELDS = [
    "total_parameters",
    "embedding_parameters",
    "lm_head_parameters",
    "non_embedding_parameters",
    "ffn_parameters",
    "attention_parameters",
    "other_non_embedding_parameters",
    "lm_head_counting",
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
    distributed_context: Any | None = None,
) -> Path | None:
    if not _should_write_shared_artifact(distributed_context):
        return None
    return write_resolved_config(config, output_dir=output_dir)


def build_run_summary(
    config: Mapping[str, Any],
    tokens_seen: int | None = None,
    content_tokens_seen: int | None = None,
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
    if content_tokens_seen is None:
        content_tokens_seen = training.get("content_tokens_seen", tokens_seen)
    stop_reason = "failed" if status == "failed" else "not_started"

    summary = {
        "run_id": run["run_id"],
        "phase_id": run["phase_id"],
        "model_family": run["model_family"],
        "model_size_label": _model_shape_label(run),
        "model_shape_label": _model_shape_label(run),
        "sampling_mode": _sampling_mode(run, training),
        "completion_label": run["completion_label"],
        "dataset_name": dataset["dataset_name"],
        "dataset_split": dataset["dataset_split"],
        "token_budget": training["token_budget"],
        "expected_tokens_per_step": training["expected_tokens_per_step"],
        "derived_max_steps": training["derived_max_steps"],
        "effective_world_size": training["effective_world_size"],
        "tokens_seen": tokens_seen,
        "content_tokens_seen": content_tokens_seen,
        "stop_reason": stop_reason,
        "seed": run.get("seed"),
        "status": status,
        "output_root": run["output_root"],
        "output_dir": run["output_dir"],
        "d_model": model.get("d_model", model.get("hidden_size")),
        "num_layers": model.get("num_layers"),
        "num_attention_heads": model.get("num_attention_heads"),
        "context_length": model.get("context_length"),
        "vocab_size_assumption": model.get(
            "vocab_size_assumption",
            model.get("vocab_size"),
        ),
        "parameter_counts": config.get("parameter_counts"),
        "parameter_counts_by_granularity": config.get(
            "parameter_counts_by_granularity"
        ),
        **build_checkpoint_summary_fields(
            config,
            metrics_rows=[],
            validation_enabled=config.get("evaluation", {}).get("validation", False),
            save_checkpoints=config.get("outputs", {}).get(
                "save_checkpoints",
                False,
            ),
        ),
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
    distributed_context: Any | None = None,
) -> Path | None:
    if not _should_write_shared_artifact(distributed_context):
        return None
    _require_fields(summary, RUN_SUMMARY_FIELDS, filename)
    return write_json_artifact(Path(output_dir) / filename, summary)


def write_failed_run_summary(
    config: Mapping[str, Any],
    error_message: str,
    output_dir: str | Path | None = None,
    tokens_seen: int = 0,
    content_tokens_seen: int = 0,
    notes: Iterable[str] | None = None,
    distributed_context: Any | None = None,
) -> Path | None:
    if not _should_write_shared_artifact(distributed_context):
        return None
    failure_notes = [error_message]
    failure_notes.extend(notes or [])
    summary = build_run_summary(
        config,
        tokens_seen=tokens_seen,
        content_tokens_seen=content_tokens_seen,
        status="failed",
        notes=failure_notes,
    )
    run_output_dir = output_dir or config["run"]["output_dir"]
    return write_run_summary(run_output_dir, summary)


def build_checkpoint_summary_fields(
    config: Mapping[str, Any],
    metrics_rows: Iterable[Mapping[str, Any]],
    validation_enabled: bool | None = None,
    save_checkpoints: bool | None = None,
) -> dict[str, Any]:
    output_dir = Path(config["run"]["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"

    if validation_enabled is None:
        validation_enabled = bool(config.get("evaluation", {}).get("validation", False))
    if save_checkpoints is None:
        save_checkpoints = bool(config.get("outputs", {}).get("save_checkpoints", False))

    fields = {
        "checkpoint_status": "none",
        "best_checkpoint_path": None,
        "final_checkpoint_path": None,
        "checkpoint_metric": None,
        "checkpoint_metric_value": None,
        "checkpoint_selection_step": None,
        "checkpoint_unavailable_reason": None,
    }

    if not save_checkpoints:
        fields["checkpoint_unavailable_reason"] = "checkpoint writes disabled"
        return fields

    if not validation_enabled:
        fields["checkpoint_status"] = "final"
        fields["final_checkpoint_path"] = str(checkpoint_dir / "final.pt")
        return fields

    best_row, metric_name, metric_value = _best_validation_metric_row(metrics_rows)
    if best_row is None:
        fields["checkpoint_status"] = "unavailable"
        fields["checkpoint_unavailable_reason"] = (
            "validation enabled but no validation loss or perplexity rows were available"
        )
        return fields

    selection_step = _int_value(best_row.get("step"))
    fields.update(
        {
            "checkpoint_status": "best_eval",
            "best_checkpoint_path": str(
                checkpoint_dir / f"best_eval_step_{selection_step}.pt"
            ),
            "checkpoint_metric": metric_name,
            "checkpoint_metric_value": metric_value,
            "checkpoint_selection_step": selection_step,
        }
    )
    if best_row.get("granularity") is not None:
        fields["checkpoint_selection_granularity"] = best_row["granularity"]
    return fields


def write_metrics_csv(
    output_dir: str | Path,
    rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    append: bool = False,
    distributed_context: Any | None = None,
) -> Path | None:
    return write_csv_artifact(
        Path(output_dir) / "metrics.csv",
        rows,
        METRICS_COLUMNS,
        append=append,
        distributed_context=distributed_context,
    )


def write_task_results_csv(
    output_dir: str | Path,
    rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    append: bool = False,
    distributed_context: Any | None = None,
) -> Path | None:
    return write_csv_artifact(
        Path(output_dir) / "task_results.csv",
        rows,
        TASK_RESULTS_COLUMNS,
        append=append,
        distributed_context=distributed_context,
    )


def write_scaling_results_csv(
    output_dir: str | Path,
    rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    append: bool = False,
    distributed_context: Any | None = None,
) -> Path | None:
    return write_csv_artifact(
        Path(output_dir) / "scaling_results.csv",
        rows,
        SCALING_RESULTS_COLUMNS,
        append=append,
        distributed_context=distributed_context,
    )


def write_consistency_results_csv(
    output_dir: str | Path,
    rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    append: bool = False,
    distributed_context: Any | None = None,
) -> Path | None:
    return write_csv_artifact(
        Path(output_dir) / "consistency_results.csv",
        build_consistency_result_rows(rows),
        CONSISTENCY_RESULTS_COLUMNS,
        append=append,
        distributed_context=distributed_context,
    )


def build_consistency_result_rows(
    rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(rows, Mapping):
        rows = [rows]

    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {
                "comparison_id": row.get("comparison_id"),
                "small_run_id": row.get("small_run_id"),
                "large_run_id": row.get("large_run_id"),
                "small_granularity": row.get("small_granularity"),
                "large_granularity": row.get("large_granularity"),
                "metric_name": _normalize_consistency_metric_name(row),
                "metric_value": row.get("metric_value"),
                "sample_count": row.get("sample_count"),
            }
        )
    return normalized_rows


def _normalize_consistency_metric_name(row: Mapping[str, Any]) -> str:
    metric_name = str(row.get("metric_name") or "")
    top_k = row.get("top_k")

    if metric_name == "top_k_overlap" and top_k not in (None, ""):
        return f"top_k_overlap@{int(top_k)}"
    if metric_name == "kl_divergence" and row.get("deferred"):
        return "kl_divergence_deferred"
    return metric_name


def build_parameter_counts_by_granularity(
    model: Any,
    granularities: Iterable[str],
    trainable_only: bool = False,
) -> dict[str, dict[str, Any]]:
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
    parameter_counts_by_granularity: Mapping[str, Mapping[str, Any]],
    comparison_id_prefix: str | None = None,
) -> list[dict[str, Any]]:
    run = config["run"]
    model = config["model"]
    training = config.get("training", {})
    metrics_rows = list(metrics_rows)
    validation_rows = latest_metric_rows_by_granularity(
        metrics_rows,
        split="validation",
    )
    latest_rows = validation_rows
    if not latest_rows:
        latest_rows = latest_metric_rows_by_granularity(metrics_rows, split="train")

    allow_partial_training_rows = (
        not validation_rows
        and training.get("granularity_sampling") == "random"
    )
    granularities = list(model["granularities"])
    if allow_partial_training_rows:
        granularities = [
            granularity for granularity in granularities if granularity in latest_rows
        ]

    rows = []
    for granularity in granularities:
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
            PARAMETER_COUNT_FIELDS[:4],
            "scaling_results.csv",
        )

        comparison_id = f"{comparison_id_prefix or run['run_id']}__{granularity}"
        rows.append(
            {
                "comparison_id": comparison_id,
                "run_id": run["run_id"],
                "model_family": run["model_family"],
                "model_size_label": _model_shape_label(run),
                "model_shape_label": _model_shape_label(run),
                "sampling_mode": _sampling_mode(run, training),
                "completion_label": run["completion_label"],
                "granularity": granularity,
                "d_model": model.get("d_model", model.get("hidden_size")),
                "num_layers": model.get("num_layers"),
                "num_attention_heads": model.get("num_attention_heads"),
                "context_length": model.get("context_length"),
                "vocab_size_assumption": model.get(
                    "vocab_size_assumption",
                    model.get("vocab_size"),
                ),
                "token_budget": training.get("token_budget"),
                "effective_world_size": training.get("effective_world_size"),
                "total_parameters": parameter_counts["total_parameters"],
                "embedding_parameters": parameter_counts["embedding_parameters"],
                "lm_head_parameters": parameter_counts["lm_head_parameters"],
                "non_embedding_parameters": parameter_counts[
                    "non_embedding_parameters"
                ],
                "ffn_parameters": parameter_counts.get("ffn_parameters"),
                "attention_parameters": parameter_counts.get(
                    "attention_parameters"
                ),
                "other_non_embedding_parameters": parameter_counts.get(
                    "other_non_embedding_parameters"
                ),
                "lm_head_counting": parameter_counts.get("lm_head_counting"),
                "checkpoint_path": None,
                "loss": metric_row["loss"],
                "perplexity": metric_row["perplexity"],
                "average_downstream_accuracy": None,
            }
        )

    return rows


def build_pilot_comparison_rows(
    comparison_id: str,
    run_summaries: Iterable[Mapping[str, Any]],
    omitted_rows: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for summary in run_summaries:
        granularities = _summary_granularities(summary)
        for granularity in granularities:
            parameter_counts = _summary_parameter_counts(summary, granularity)
            rows.append(
                _with_artifact_defaults(
                    {
                        "comparison_id": comparison_id,
                        "run_id": summary["run_id"],
                        "run_status": summary.get("status", "completed"),
                        "omit_reason": None,
                        "model_family": summary.get("model_family"),
                        "model_size_label": _model_shape_label(summary),
                        "model_shape_label": _model_shape_label(summary),
                        "sampling_mode": summary.get("sampling_mode"),
                        "completion_label": summary.get(
                            "completion_label",
                            "reduced-token-pilot",
                        ),
                        "granularity": granularity,
                        "d_model": summary.get("d_model"),
                        "num_layers": summary.get("num_layers"),
                        "num_attention_heads": summary.get(
                            "num_attention_heads"
                        ),
                        "context_length": summary.get("context_length"),
                        "vocab_size_assumption": summary.get(
                            "vocab_size_assumption"
                        ),
                        "token_budget": summary.get("token_budget"),
                        "effective_world_size": summary.get(
                            "effective_world_size"
                        ),
                        "total_parameters": parameter_counts.get(
                            "total_parameters"
                        ),
                        "embedding_parameters": parameter_counts.get(
                            "embedding_parameters"
                        ),
                        "lm_head_parameters": parameter_counts.get(
                            "lm_head_parameters"
                        ),
                        "non_embedding_parameters": parameter_counts.get(
                            "non_embedding_parameters"
                        ),
                        "ffn_parameters": parameter_counts.get(
                            "ffn_parameters"
                        ),
                        "attention_parameters": parameter_counts.get(
                            "attention_parameters"
                        ),
                        "other_non_embedding_parameters": parameter_counts.get(
                            "other_non_embedding_parameters"
                        ),
                        "lm_head_counting": parameter_counts.get(
                            "lm_head_counting"
                        ),
                        "checkpoint_status": _summary_checkpoint_status(
                            summary
                        ),
                        "checkpoint_path": _summary_checkpoint_path(summary),
                        "checkpoint_metric": summary.get("checkpoint_metric"),
                    }
                )
            )

    for omitted_row in omitted_rows or []:
        omit_reason = omitted_row.get("omit_reason")
        rows.append(
            _with_artifact_defaults(
                {
                    "comparison_id": comparison_id,
                    "run_id": omitted_row.get("run_id"),
                    "run_status": "omitted",
                    "omit_reason": omit_reason,
                    "model_family": omitted_row.get("model_family", "standalone"),
                    "model_size_label": _model_shape_label(
                        {
                            "model_shape_label": omitted_row.get(
                                "model_shape_label",
                                "dmodel256",
                            )
                        }
                    ),
                    "model_shape_label": omitted_row.get(
                        "model_shape_label",
                        "dmodel256",
                    ),
                    "sampling_mode": omitted_row.get(
                        "sampling_mode",
                        "standalone",
                    ),
                    "completion_label": omitted_row.get(
                        "completion_label",
                        "reduced-token-pilot",
                    ),
                    "granularity": omitted_row.get("granularity"),
                    "d_model": omitted_row.get("d_model"),
                    "num_layers": omitted_row.get("num_layers"),
                    "num_attention_heads": omitted_row.get(
                        "num_attention_heads"
                    ),
                    "context_length": omitted_row.get("context_length"),
                    "vocab_size_assumption": omitted_row.get(
                        "vocab_size_assumption"
                    ),
                    "token_budget": omitted_row.get("token_budget"),
                    "effective_world_size": omitted_row.get(
                        "effective_world_size"
                    ),
                    "checkpoint_status": "unavailable",
                    "checkpoint_path": None,
                    "checkpoint_metric": None,
                }
            )
        )

    return rows


def build_speculative_task_rows(
    config: Mapping[str, Any],
    pair_results: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    run = config["run"]
    suite_id = "speculative-alignment"
    requested_metrics = [str(metric) for metric in config.get("metrics", [])]
    rows = []

    for result in pair_results:
        granularity_label = (
            f"{result.get('draft_granularity')}->{result.get('verifier_granularity')}"
        )
        row_base = {
            "run_id": run["run_id"],
            "suite_id": suite_id,
            "task": result["pair_id"],
            "model_family": result["pair_type"],
            "model_size_label": result.get("model_shape_label"),
            "model_shape_label": result.get("model_shape_label"),
            "sampling_mode": result.get("sampling_mode"),
            "granularity": granularity_label,
        }
        for metric_name in requested_metrics:
            rows.append(
                row_base
                | {
                    "metric_name": metric_name,
                    "metric_value": result.get(metric_name),
                }
            )

    return rows


def build_baseline_match_row(
    nested_config: Mapping[str, Any],
    standalone_config: Mapping[str, Any],
    granularity: str,
    nested_counts: Mapping[str, Any] | None = None,
    standalone_counts: Mapping[str, Any] | None = None,
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


def write_json_artifact(
    path: str | Path,
    payload: Mapping[str, Any],
    distributed_context: Any | None = None,
) -> Path | None:
    if not _should_write_shared_artifact(distributed_context):
        return None
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
    distributed_context: Any | None = None,
) -> Path | None:
    if not _should_write_shared_artifact(distributed_context):
        return None
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
) -> list[dict[str, Any]]:
    if isinstance(rows, Mapping):
        return [_with_artifact_defaults(rows)]
    return [_with_artifact_defaults(row) for row in rows]


def _with_artifact_defaults(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized_row = dict(row)
    model_shape_label = normalized_row.get(
        "model_shape_label",
        normalized_row.get("model_size_label"),
    )

    defaults = {
        "model_size_label": model_shape_label,
        "model_shape_label": model_shape_label,
        "sampling_mode": None,
        "d_model": None,
        "num_layers": None,
        "num_attention_heads": None,
        "context_length": None,
        "vocab_size_assumption": None,
        "token_budget": None,
        "effective_world_size": None,
        "content_tokens_seen": normalized_row.get("tokens_seen"),
        "ffn_parameters": None,
        "attention_parameters": None,
        "other_non_embedding_parameters": None,
        "lm_head_counting": None,
        "checkpoint_path": None,
        "checkpoint_status": None,
        "checkpoint_metric": None,
        "run_status": normalized_row.get("status"),
        "omit_reason": None,
    }

    for key, value in defaults.items():
        normalized_row.setdefault(key, value)

    return normalized_row


def _non_embedding_count(counts: Mapping[str, Any] | None):
    if counts is None:
        return None
    return counts.get("non_embedding_parameters")


def _model_shape_label(run: Mapping[str, Any]) -> Any:
    return run.get("model_shape_label", run.get("model_size_label"))


def _sampling_mode(run: Mapping[str, Any], training: Mapping[str, Any]) -> Any:
    if run.get("sampling_mode") is not None:
        return run["sampling_mode"]
    if run.get("model_family") == "standalone":
        return "standalone"

    granularity_sampling = training.get("granularity_sampling")
    if granularity_sampling == "random":
        return "nested-random"
    if granularity_sampling == "all":
        return "nested-all"
    return granularity_sampling


def _summary_granularities(summary: Mapping[str, Any]) -> list[str]:
    if summary.get("granularities"):
        return [str(granularity) for granularity in summary["granularities"]]
    if summary.get("granularity"):
        return [str(summary["granularity"])]
    parameter_counts = summary.get("parameter_counts_by_granularity") or {}
    return [str(granularity) for granularity in parameter_counts]


def _summary_parameter_counts(
    summary: Mapping[str, Any],
    granularity: str,
) -> Mapping[str, Any]:
    counts_by_granularity = summary.get("parameter_counts_by_granularity") or {}
    if granularity in counts_by_granularity:
        return counts_by_granularity[granularity]
    if summary.get("granularity") == granularity:
        return summary.get("parameter_counts") or {}
    return {}


def _summary_checkpoint_status(summary: Mapping[str, Any]) -> str:
    if summary.get("checkpoint_status"):
        return str(summary["checkpoint_status"])
    if summary.get("best_checkpoint_path"):
        return "best_eval"
    if summary.get("final_checkpoint_path"):
        return "final"
    if summary.get("checkpoint_path"):
        return "available"
    return "unavailable"


def _summary_checkpoint_path(summary: Mapping[str, Any]) -> Any:
    return (
        summary.get("checkpoint_path")
        or summary.get("best_checkpoint_path")
        or summary.get("final_checkpoint_path")
    )


def _require_fields(
    row: Mapping[str, Any],
    required_fields: list[str],
    artifact_name: str,
) -> None:
    missing_fields = [field_name for field_name in required_fields if field_name not in row]
    if missing_fields:
        raise ArtifactError(f"{artifact_name} missing fields: {missing_fields}")


def _best_validation_metric_row(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, str | None, float | None]:
    validation_rows = [row for row in rows if row.get("split") == "validation"]
    if not validation_rows:
        return None, None, None

    for field_name, metric_name in [
        ("loss", "validation_loss"),
        ("perplexity", "validation_perplexity"),
    ]:
        candidates = []
        for row in validation_rows:
            metric_value = _float_value(row.get(field_name))
            if metric_value is not None:
                candidates.append((metric_value, _int_value(row.get("step")), row))
        if candidates:
            metric_value, _, row = min(candidates, key=lambda candidate: candidate[:2])
            return row, metric_name, metric_value

    return None, None, None


def _int_value(value: Any) -> int:
    if value in (None, ""):
        return -1
    return int(value)


def _float_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _should_write_shared_artifact(distributed_context: Any | None) -> bool:
    if distributed_context is None:
        return True

    from training.distributed import should_write_shared_artifact

    return should_write_shared_artifact(distributed_context)
