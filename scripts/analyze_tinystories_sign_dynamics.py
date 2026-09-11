#!/usr/bin/env python3
"""Analyze completed elastic sign-dynamics runs and generate focused figures."""

# ruff: noqa: E402  # Add the repository root before importing src.

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import multiprocessing
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.reproducibility import stable_hash


class SignDynamicsAnalysisError(ValueError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SignDynamicsAnalysisError(f"cannot read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise SignDynamicsAnalysisError(f"JSON artifact must be a mapping: {path}")
    return value


def _progress(message: str) -> None:
    print(f"[sign-dynamics analysis] {message}", file=sys.stderr, flush=True)


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    records = []
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for line_number, line in enumerate(source, start=1):
                digest.update(line)
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise SignDynamicsAnalysisError(f"invalid JSONL artifact: {path}") from error
    if [int(record.get("step", -1)) for record in records] != list(
        range(1, len(records) + 1)
    ):
        raise SignDynamicsAnalysisError(f"journal steps are not contiguous: {path}")
    return records, digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(run_dir: Path, configured: Any, fallback: str) -> Path:
    candidate = Path(str(configured or fallback)).expanduser()
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    return candidate.resolve()


def _measurement_contract(
    config: Mapping[str, Any], support: Mapping[str, Any]
) -> dict[str, Any]:
    model = config.get("model", {})
    training = config.get("training", {})
    dataset = config.get("dataset", {})
    diagnostic = config.get("evaluation", {}).get("sign_dynamics", {})
    return {
        "model": {
            field: model.get(field)
            for field in (
                "base_model_name",
                "variant",
                "correction_mode",
                "membership_correction",
                "d_model",
                "hidden_size",
                "num_layers",
                "num_attention_heads",
                "intermediate_size",
                "context_length",
                "vocab_size",
                "granularities",
                "granularity_prefixes",
                "initializer_range",
            )
        },
        "data": {
            field: dataset.get(field)
            for field in (
                "mode",
                "prepared_corpus_dir",
                "dataset_name",
                "dataset_config_name",
                "dataset_split",
                "dataset_phase",
                "data_seed",
                "preprocessing_notes",
            )
            if field != "data_seed"
        }
        | {
            field: config.get(field)
            for field in (
                "corpus_hash",
                "tokenizer_manifest_hash",
                "data_roles_manifest_hash",
                "optimizer_training_manifest_hash",
                "validation_manifest_hash",
            )
        },
        "optimizer": {
            "name": training.get("optimizer_name"),
            "kwargs": training.get("optimizer_kwargs"),
            "state_scope": training.get("optimizer_state_scope"),
            "resolved_learning_rate": training.get("resolved_learning_rate"),
            "gradient_clip_norm": training.get("gradient_clip_norm"),
            "resolved_warmup_steps": training.get("resolved_warmup_steps"),
            "scheduler": training.get("scheduler_contract")
            or {
                "name": training.get("scheduler_name"),
                "kwargs": training.get("scheduler_kwargs"),
            },
        },
        "precision": training.get(
            "resolved_mixed_precision", training.get("mixed_precision")
        ),
        "support_hash": support.get("support_hash"),
        "diagnostic_contract_hash": diagnostic.get("diagnostic_contract_hash"),
    }


def discover_runs(campaign_root: Path) -> list[dict[str, Any]]:
    runs = []
    for summary_path in sorted(campaign_root.rglob("run_summary.json")):
        run_dir = summary_path.parent.resolve()
        summary = _read_json(summary_path)
        if summary.get("status") != "completed" or not summary.get(
            "sign_dynamics_enabled", False
        ):
            continue
        if not summary.get("sign_dynamics_step_coverage_complete", False):
            raise SignDynamicsAnalysisError(
                f"completed run has incomplete sign-dynamics coverage: {run_dir}"
            )
        config = _read_json(run_dir / "config.json")
        diagnostic = config.get("evaluation", {}).get("sign_dynamics", {})
        support_path = _artifact_path(
            run_dir,
            summary.get("sign_dynamics_support_path"),
            str(diagnostic.get("support_manifest_path", "sign_dynamics_support.json")),
        )
        journal_path = _artifact_path(
            run_dir,
            summary.get("sign_dynamics_journal_path"),
            str(diagnostic.get("journal_path", "sign_dynamics.jsonl")),
        )
        support = _read_json(support_path)
        support_hash = support.get("support_hash")
        unhashed_support = {
            key: value for key, value in support.items() if key != "support_hash"
        }
        if support_hash != stable_hash(unhashed_support):
            raise SignDynamicsAnalysisError(f"support hash is invalid: {support_path}")
        _progress(f"reading and hashing {journal_path}")
        records, journal_hash = _read_jsonl(journal_path)
        if len(records) != int(summary.get("steps_completed", -1)):
            raise SignDynamicsAnalysisError(f"journal coverage mismatch: {run_dir}")
        if journal_hash != summary.get("sign_dynamics_journal_hash"):
            raise SignDynamicsAnalysisError(f"journal hash mismatch: {run_dir}")
        if any(
            record.get("support_hash") != support_hash
            or record.get("diagnostic_contract_hash")
            != diagnostic.get("diagnostic_contract_hash")
            for record in records
        ):
            raise SignDynamicsAnalysisError(f"journal contract mismatch: {run_dir}")
        runs.append(
            {
                "run_dir": run_dir,
                "summary": summary,
                "config": config,
                "support": support,
                "records": records,
                "arm": str(summary.get("sign_dynamics_arm_id")),
                "seed": int(summary.get("seed")),
                "budget_multiplier": summary.get("sign_dynamics_budget_multiplier"),
                "contract": _measurement_contract(config, support),
            }
        )
    if not runs:
        raise SignDynamicsAnalysisError(
            f"no completed sign-dynamics runs found below {campaign_root}"
        )
    contract_hashes = {stable_hash(run["contract"]) for run in runs}
    if len(contract_hashes) != 1:
        raise SignDynamicsAnalysisError(
            "runs mix model, data, optimizer, precision, support, or measurement contracts"
        )
    identities = [(run["budget_multiplier"], run["arm"], run["seed"]) for run in runs]
    if len(set(identities)) != len(identities):
        raise SignDynamicsAnalysisError(
            "duplicate budget/arm/seed runs were discovered"
        )
    return runs


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_step_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        for record in run["records"]:
            common = {
                "budget_multiplier": run["budget_multiplier"],
                "arm_id": run["arm"],
                "seed": run["seed"],
                "step": int(record["step"]),
                "action": record["selected_granularity"],
                "sampled_probability": record.get("sampled_probability"),
                "h_window_interval_steps": record.get("h_window_interval_steps"),
                "h_window_index": record.get("h_window_index"),
                "h_window_position": record.get("h_window_position"),
            }
            for band, statistics_by_band in record["bands"].items():
                base = {
                    **common,
                    "band": band,
                    **{
                        field: statistics_by_band[field]
                        for field in (
                            "coordinate_count",
                            "forward_active",
                            "gradient_present_coordinate_count",
                            "nonzero_gradient_count",
                            "gradient_rms",
                            "changed_coordinate_count",
                            "changed_coordinate_rate",
                            "update_rms",
                            "relative_update_rms",
                            "raw_sign_flip_count",
                            "raw_sign_flip_rate",
                        )
                    },
                }
                for threshold, robust in statistics_by_band["hysteresis"].items():
                    rows.append(
                        {
                            **base,
                            "threshold": float(threshold),
                            "robust_transition_count": robust["transition_count"],
                            "robust_transition_rate": robust["transition_rate"],
                            "dead_band_occupancy": robust["dead_band_occupancy"],
                            "dead_band_entries": robust["dead_entries"],
                            "dead_band_exits": robust["dead_exits"],
                            "established_count": robust["established_count"],
                            "last_robust_transition_step": robust[
                                "last_robust_transition_step"
                            ],
                            "last_robust_transition_granularity": robust[
                                "last_robust_transition_granularity"
                            ],
                        }
                    )
    return rows


def _finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


def _read_validation_metrics(
    run: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, float]]:
    metrics_path = _artifact_path(
        run["run_dir"], run["summary"].get("metrics_path"), "metrics.csv"
    )
    validation_rows: list[dict[str, Any]] = []
    learning_rates: dict[int, float] = {}
    try:
        with metrics_path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            required = {"step", "split", "granularity", "loss"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise SignDynamicsAnalysisError(
                    f"metrics file lacks validation fields: {metrics_path}"
                )
            for row in reader:
                step = int(row["step"])
                if row["split"] == "train":
                    learning_rate = _finite_float(row.get("learning_rate"))
                    if learning_rate is not None:
                        learning_rates[step] = learning_rate
                    continue
                if row["split"] != "validation":
                    continue
                loss = _finite_float(row.get("loss"))
                if loss is None:
                    raise SignDynamicsAnalysisError(
                        f"validation loss is missing or non-finite: {metrics_path}:{step}"
                    )
                validation_rows.append(
                    {
                        "step": step,
                        "granularity": str(row["granularity"]),
                        "loss": loss,
                        "perplexity": _finite_float(row.get("perplexity")),
                        "tokens_seen": _finite_float(row.get("tokens_seen")),
                    }
                )
    except (OSError, csv.Error, TypeError, ValueError) as error:
        if isinstance(error, SignDynamicsAnalysisError):
            raise
        raise SignDynamicsAnalysisError(
            f"cannot read training/validation metrics: {metrics_path}"
        ) from error
    return validation_rows, learning_rates


def _root_mean_square_from_band_statistics(
    squared_sum: float, coordinate_observations: int
) -> float | None:
    if coordinate_observations <= 0:
        return None
    return math.sqrt(max(squared_sum, 0.0) / coordinate_observations)


def build_validation_dynamics_rows(
    runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join each ordinary-validation interval to exact cumulative-support dynamics."""

    output: list[dict[str, Any]] = []
    for run in runs:
        validation, learning_rates = _read_validation_metrics(run)
        if not validation:
            continue
        records = run["records"]
        max_step = len(records)
        ordered_granularities = [
            str(value) for value in run["support"]["ordered_granularities"]
        ]
        bands = [
            str(value)
            for value in run["support"]["primary_partition"]["bands"]
        ]
        if len(bands) != len(ordered_granularities):
            raise SignDynamicsAnalysisError(
                f"support granularity/band count mismatch: {run['run_dir']}"
            )
        unknown = sorted(
            {row["granularity"] for row in validation}
            - set(ordered_granularities)
        )
        if unknown:
            raise SignDynamicsAnalysisError(
                f"validation contains unsupported granularities {unknown}: "
                f"{run['run_dir']}"
            )
        thresholds = [
            float(value)
            for value in run["config"]["evaluation"]["sign_dynamics"][
                "hysteresis_thresholds"
            ]
        ]
        validation_by_granularity: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in validation:
            validation_by_granularity[row["granularity"]].append(row)

        for granularity_index, granularity in enumerate(ordered_granularities):
            checkpoints = sorted(
                validation_by_granularity.get(granularity, []),
                key=lambda row: int(row["step"]),
            )
            if len({int(row["step"]) for row in checkpoints}) != len(checkpoints):
                raise SignDynamicsAnalysisError(
                    f"duplicate validation step for {granularity}: {run['run_dir']}"
                )
            active_bands = bands[: granularity_index + 1]
            previous_step = 0
            previous_loss: float | None = None
            for checkpoint in checkpoints:
                step = int(checkpoint["step"])
                if step <= previous_step or step > max_step:
                    raise SignDynamicsAnalysisError(
                        f"validation step is outside the diagnostic journal: "
                        f"{run['run_dir']}:{step}"
                    )
                interval_records = records[previous_step:step]
                interval_steps = step - previous_step
                action_support_steps = 0
                coordinate_observations = 0
                raw_flip_count = 0
                changed_count = 0
                gradient_squared_sum = 0.0
                update_squared_sum = 0.0
                relative_update_squared_sum = 0.0
                robust_counts = {threshold: 0 for threshold in thresholds}
                dead_coordinate_observations = {
                    threshold: 0.0 for threshold in thresholds
                }
                for record in interval_records:
                    action_index = int(record["selected_granularity_index"])
                    action_support_steps += int(action_index >= granularity_index)
                    for band in active_bands:
                        statistics_by_band = record["bands"][band]
                        count = int(statistics_by_band["coordinate_count"])
                        coordinate_observations += count
                        raw_flip_count += int(
                            statistics_by_band["raw_sign_flip_count"]
                        )
                        changed_count += int(
                            statistics_by_band["changed_coordinate_count"]
                        )
                        gradient_squared_sum += count * float(
                            statistics_by_band["gradient_rms"]
                        ) ** 2
                        update_squared_sum += count * float(
                            statistics_by_band["update_rms"]
                        ) ** 2
                        relative_update_squared_sum += count * float(
                            statistics_by_band["relative_update_rms"]
                        ) ** 2
                        for threshold in thresholds:
                            hysteresis = statistics_by_band["hysteresis"][
                                format(threshold, ".15g")
                            ]
                            robust_counts[threshold] += int(
                                hysteresis["transition_count"]
                            )
                            dead_coordinate_observations[threshold] += count * float(
                                hysteresis["dead_band_occupancy"]
                            )

                raw_rate = raw_flip_count / max(coordinate_observations, 1)
                relative_update_rms = _root_mean_square_from_band_statistics(
                    relative_update_squared_sum, coordinate_observations
                )
                loss_improvement = (
                    previous_loss - float(checkpoint["loss"])
                    if previous_loss is not None
                    else None
                )
                loss_improvement_rate = (
                    loss_improvement * 1000.0 / interval_steps
                    if loss_improvement is not None
                    else None
                )
                learning_rate_values = [
                    learning_rates[record_step]
                    for record_step in range(previous_step + 1, step + 1)
                    if record_step in learning_rates
                ]
                progress = step / max(max_step, 1)
                phase = (
                    "early_0_10"
                    if progress <= 0.10
                    else "middle_10_50"
                    if progress <= 0.50
                    else "late_50_100"
                )
                for threshold in thresholds:
                    robust_rate = robust_counts[threshold] / max(
                        coordinate_observations, 1
                    )
                    output.append(
                        {
                            "budget_multiplier": run["budget_multiplier"],
                            "arm_id": run["arm"],
                            "seed": run["seed"],
                            "granularity": granularity,
                            "active_bands": json.dumps(active_bands),
                            "threshold": threshold,
                            "validation_step": step,
                            "previous_validation_step": previous_step,
                            "interval_steps": interval_steps,
                            "trajectory_fraction": progress,
                            "training_phase": phase,
                            "validation_loss": float(checkpoint["loss"]),
                            "validation_perplexity": checkpoint["perplexity"],
                            "validation_tokens_seen": checkpoint["tokens_seen"],
                            "previous_validation_loss": previous_loss,
                            "validation_loss_improvement": loss_improvement,
                            "validation_loss_improvement_per_1000_steps": (
                                loss_improvement_rate
                            ),
                            "raw_sign_flip_rate": raw_rate,
                            "robust_transition_rate": robust_rate,
                            "gradient_rms": _root_mean_square_from_band_statistics(
                                gradient_squared_sum, coordinate_observations
                            ),
                            "update_rms": _root_mean_square_from_band_statistics(
                                update_squared_sum, coordinate_observations
                            ),
                            "relative_update_rms": relative_update_rms,
                            "changed_coordinate_rate": changed_count
                            / max(coordinate_observations, 1),
                            "dead_band_occupancy": dead_coordinate_observations[
                                threshold
                            ]
                            / max(coordinate_observations, 1),
                            "raw_flips_per_relative_update": (
                                raw_rate / relative_update_rms
                                if relative_update_rms is not None
                                and relative_update_rms > 0.0
                                else None
                            ),
                            "robust_transitions_per_relative_update": (
                                robust_rate / relative_update_rms
                                if relative_update_rms is not None
                                and relative_update_rms > 0.0
                                else None
                            ),
                            "mean_learning_rate": _mean(learning_rate_values),
                            "parameter_support_exposure_rate": (
                                action_support_steps / interval_steps
                            ),
                            "coordinate_observations": coordinate_observations,
                        }
                    )
                previous_step = step
                previous_loss = float(checkpoint["loss"])
    return output


def _mean(values: Iterable[Any]) -> float | None:
    resolved = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return statistics.fmean(resolved) if resolved else None


_TIME_BIN_METRIC_FIELDS = (
    "raw_sign_flip_rate",
    "robust_transition_rate",
    "gradient_rms",
    "relative_update_rms",
    "changed_coordinate_rate",
    "dead_band_occupancy",
    "last_robust_transition_step",
)


def _standard_error(values: Sequence[float]) -> float | None:
    return (
        statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    )


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0.0:
        return None
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered, strict=True)
    ) / denominator


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        for index in ordered[start:stop]:
            ranks[index] = average_rank
        start = stop
    return ranks


def _partial_pearson(
    left: Sequence[float], right: Sequence[float], controls: Sequence[Sequence[float]]
) -> float | None:
    if len(left) < 4 or len(left) != len(right) or len(left) != len(controls):
        return None
    import numpy

    x = numpy.asarray(left, dtype=float)
    y = numpy.asarray(right, dtype=float)
    z = numpy.asarray(controls, dtype=float)
    if z.ndim != 2:
        return None
    means = z.mean(axis=0)
    scales = z.std(axis=0)
    useful = scales > 0.0
    z = (z[:, useful] - means[useful]) / scales[useful]
    design = numpy.column_stack((numpy.ones(len(x)), z))
    x_residual = x - design @ numpy.linalg.lstsq(design, x, rcond=None)[0]
    y_residual = y - design @ numpy.linalg.lstsq(design, y, rcond=None)[0]
    return _pearson(x_residual.tolist(), y_residual.tolist())


_PERFORMANCE_PHASES = {
    "all": lambda progress: True,
    "early_0_10": lambda progress: progress <= 0.10,
    "post_10_100": lambda progress: progress > 0.10,
    "middle_10_50": lambda progress: 0.10 < progress <= 0.50,
    "late_50_100": lambda progress: progress > 0.50,
}


def build_performance_correlation_rows(
    validation_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compute run-level longitudinal correlations without treating widths as IID."""

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in validation_rows:
        key = (
            row["budget_multiplier"],
            row["arm_id"],
            row["seed"],
            row["granularity"],
            float(row["threshold"]),
        )
        groups[key].append(row)

    output: list[dict[str, Any]] = []
    for key, raw_rows in groups.items():
        ordered = sorted(raw_rows, key=lambda row: int(row["validation_step"]))
        threshold = float(key[4])
        metric_fields = (
            ("raw_sign_flip_rate", "raw_sign_flip_rate"),
            ("raw_flips_per_relative_update", "raw_flips_per_relative_update"),
        ) if threshold == 0.0 else (
            ("robust_transition_rate", "robust_transition_rate"),
            (
                "robust_transitions_per_relative_update",
                "robust_transitions_per_relative_update",
            ),
        )
        for metric_name, metric_field in metric_fields:
            for relationship in ("concurrent", "next_interval"):
                candidates = []
                for index, row in enumerate(ordered):
                    target = (
                        row.get("validation_loss_improvement_per_1000_steps")
                        if relationship == "concurrent"
                        else ordered[index + 1].get(
                            "validation_loss_improvement_per_1000_steps"
                        )
                        if index + 1 < len(ordered)
                        else None
                    )
                    metric = row.get(metric_field)
                    controls = (
                        row.get("trajectory_fraction"),
                        float(row.get("trajectory_fraction", 0.0)) ** 2,
                        row.get("mean_learning_rate"),
                        row.get("relative_update_rms"),
                        row.get("parameter_support_exposure_rate"),
                    )
                    values = (metric, target, *controls)
                    if any(_finite_float(value) is None for value in values):
                        continue
                    candidates.append(
                        (
                            float(metric),
                            float(target),
                            [float(value) for value in controls],
                            float(row["trajectory_fraction"]),
                        )
                    )
                for phase, includes in _PERFORMANCE_PHASES.items():
                    selected = [row for row in candidates if includes(row[3])]
                    left = [row[0] for row in selected]
                    right = [row[1] for row in selected]
                    controls = [row[2] for row in selected]
                    output.append(
                        {
                            "budget_multiplier": key[0],
                            "arm_id": key[1],
                            "seed": key[2],
                            "granularity": key[3],
                            "threshold": threshold,
                            "metric": metric_name,
                            "relationship": relationship,
                            "phase": phase,
                            "observation_count": len(selected),
                            "pearson": _pearson(left, right),
                            "spearman": _pearson(_ranks(left), _ranks(right))
                            if len(left) >= 3
                            else None,
                            "partial_pearson": _partial_pearson(
                                left, right, controls
                            ),
                            "partial_controls": (
                                "trajectory_fraction,trajectory_fraction_squared,"
                                "mean_learning_rate,relative_update_rms,"
                                "parameter_support_exposure_rate"
                            ),
                        }
                    )
    return output


def _summarize_seed_group(
    key: tuple[Any, ...], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    block_rows: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        block = row.get("h_window_index")
        if block is None:
            interval = max(int(row.get("h_window_interval_steps") or 1), 1)
            block = (int(row["step"]) - 1) // interval
        block_rows[int(block)].append(row)

    block_standard_errors = {}
    for field in _TIME_BIN_METRIC_FIELDS:
        block_means = []
        for values in block_rows.values():
            block_mean = _mean(row.get(field) for row in values)
            if block_mean is not None:
                block_means.append(block_mean)
        block_standard_errors[field] = _standard_error(block_means)

    return {
        "budget_multiplier": key[0],
        "arm_id": key[1],
        "seed": key[2],
        "time_bin": key[3],
        "action": key[4],
        "band": key[5],
        "threshold": key[6],
        "step_count": len(rows),
        **{
            field: _mean(row.get(field) for row in rows)
            for field in _TIME_BIN_METRIC_FIELDS
        },
        "_block_standard_errors": block_standard_errors,
    }


def _compact_seed_group_task(
    key: tuple[Any, ...], rows: Sequence[Mapping[str, Any]]
) -> tuple[tuple[Any, ...], list[tuple[int, tuple[Any, ...]]]]:
    compact_rows = []
    for row in rows:
        block = row.get("h_window_index")
        if block is None:
            interval = max(int(row.get("h_window_interval_steps") or 1), 1)
            block = (int(row["step"]) - 1) // interval
        compact_rows.append(
            (
                int(block),
                tuple(row.get(field) for field in _TIME_BIN_METRIC_FIELDS),
            )
        )
    return key, compact_rows


def _summarize_compact_seed_group(
    task: tuple[tuple[Any, ...], list[tuple[int, tuple[Any, ...]]]],
) -> dict[str, Any]:
    key, compact_rows = task
    metric_values: list[list[float]] = [[] for _ in _TIME_BIN_METRIC_FIELDS]
    block_metric_values: dict[int, list[list[float]]] = {}
    for block, raw_values in compact_rows:
        values_by_metric = block_metric_values.setdefault(
            block, [[] for _ in _TIME_BIN_METRIC_FIELDS]
        )
        for index, raw_value in enumerate(raw_values):
            if raw_value is None or not math.isfinite(float(raw_value)):
                continue
            value = float(raw_value)
            metric_values[index].append(value)
            values_by_metric[index].append(value)

    means = {}
    block_standard_errors = {}
    for index, field in enumerate(_TIME_BIN_METRIC_FIELDS):
        means[field] = _mean(metric_values[index])
        block_means = [
            statistics.fmean(values_by_metric[index])
            for values_by_metric in block_metric_values.values()
            if values_by_metric[index]
        ]
        block_standard_errors[field] = _standard_error(block_means)

    return {
        "budget_multiplier": key[0],
        "arm_id": key[1],
        "seed": key[2],
        "time_bin": key[3],
        "action": key[4],
        "band": key[5],
        "threshold": key[6],
        "step_count": len(compact_rows),
        **means,
        "_block_standard_errors": block_standard_errors,
    }


def _summarize_seed_groups(
    seed_groups: Mapping[tuple[Any, ...], Sequence[Mapping[str, Any]]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    keys = list(seed_groups)
    if workers == 1 or len(keys) < 2:
        return [_summarize_seed_group(key, seed_groups[key]) for key in keys]

    worker_count = min(workers, len(keys))
    chunksize = max(1, len(keys) // (worker_count * 4))
    tasks = (_compact_seed_group_task(key, seed_groups[key]) for key in keys)
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=worker_count) as pool:
        return list(
            pool.imap(
                _summarize_compact_seed_group,
                tasks,
                chunksize=chunksize,
            )
        )


def build_time_bin_rows(
    step_rows: Sequence[Mapping[str, Any]], *, time_bin_count: int, workers: int = 1
) -> list[dict[str, Any]]:
    if workers <= 0:
        raise SignDynamicsAnalysisError("analysis workers must be positive")
    max_step = defaultdict(int)
    for row in step_rows:
        key = (row["budget_multiplier"], row["arm_id"], row["seed"])
        max_step[key] = max(max_step[key], int(row["step"]))
    seed_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in step_rows:
        run_key = (row["budget_multiplier"], row["arm_id"], row["seed"])
        time_bin = min(
            time_bin_count - 1,
            int((int(row["step"]) - 1) * time_bin_count / max_step[run_key]),
        )
        group_key = (
            row["budget_multiplier"],
            row["arm_id"],
            row["seed"],
            time_bin,
            row["action"],
            row["band"],
            row["threshold"],
        )
        seed_groups[group_key].append(row)
    seed_rows = _summarize_seed_groups(seed_groups, workers=workers)

    panel_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        key = tuple(
            row[field]
            for field in (
                "budget_multiplier",
                "arm_id",
                "time_bin",
                "action",
                "band",
                "threshold",
            )
        )
        panel_groups[key].append(row)
    output = []
    for key, rows in panel_groups.items():
        seeds = sorted({int(row["seed"]) for row in rows})
        result = {
            "budget_multiplier": key[0],
            "arm_id": key[1],
            "time_bin": key[2],
            "action": key[3],
            "band": key[4],
            "threshold": key[5],
            "seed_count": len(seeds),
            "seeds": json.dumps(seeds),
            "uncertainty_method": (
                "seed_standard_error"
                if len(seeds) > 1
                else "h_window_block_standard_error"
            ),
        }
        for field in _TIME_BIN_METRIC_FIELDS:
            values = [float(row[field]) for row in rows if row[field] is not None]
            result[field] = _mean(values)
            if len(seeds) > 1:
                uncertainty = _standard_error(values)
            else:
                uncertainty = rows[0]["_block_standard_errors"][field]
            result[f"{field}_uncertainty"] = uncertainty
        output.append(result)
    return output


def _slice_tensor(
    tensor: torch.Tensor, slice_record: Mapping[str, Any]
) -> torch.Tensor:
    import torch

    flat = tensor.reshape(-1)
    pieces = [
        flat[int(item["start"]) : int(item["stop"])]
        for item in slice_record["flat_ranges"]
    ]
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=tensor.dtype)


def _support_slices_by_band(
    support: Mapping[str, Any],
) -> dict[str, list[tuple[str, Mapping[str, Any]]]]:
    output: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for parameter in support["parameters"]:
        for slice_record in parameter["slices"]:
            output[str(slice_record["primary_band"])].append(
                (str(parameter["parameter_name"]), slice_record)
            )
    return output


def terminal_alignment_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    import torch

    rows = []
    for run in runs:
        identities = run["summary"].get("sign_dynamics_snapshot_identities", {})
        if not isinstance(identities, Mapping) or not identities:
            raise SignDynamicsAnalysisError(
                f"snapshot identities are missing: {run['run_dir']}"
            )
        snapshots = {}
        for step_text, identity in identities.items():
            path = Path(str(identity["path"]))
            if (
                not path.is_file()
                or int(identity.get("step", -1)) != int(step_text)
                or int(identity.get("bytes", -1)) != path.stat().st_size
                or identity.get("sha256") != _sha256(path)
            ):
                raise SignDynamicsAnalysisError(
                    f"snapshot is missing or changed: {path}"
                )
            snapshots[int(step_text)] = torch.load(
                path, map_location="cpu", weights_only=False
            )
        terminal_step = max(snapshots)
        terminal = snapshots[terminal_step]["parameters"]
        slices_by_band = _support_slices_by_band(run["support"])
        thresholds = run["config"]["evaluation"]["sign_dynamics"][
            "hysteresis_thresholds"
        ]
        for step, snapshot in sorted(snapshots.items()):
            for band, slices in slices_by_band.items():
                for threshold in thresholds:
                    aligned = covered = total = 0
                    for name, slice_record in slices:
                        current = _slice_tensor(
                            snapshot["parameters"][name], slice_record
                        ).float()
                        final = _slice_tensor(terminal[name], slice_record).float()
                        current_scale = (
                            torch.sqrt(torch.mean(current.double().square()))
                            if current.numel()
                            else torch.tensor(0.0)
                        )
                        final_scale = (
                            torch.sqrt(torch.mean(final.double().square()))
                            if final.numel()
                            else torch.tensor(0.0)
                        )
                        confident = (
                            current.abs() > float(threshold) * (current_scale + 1e-30)
                        ) & (final.abs() > float(threshold) * (final_scale + 1e-30))
                        aligned += int(
                            torch.count_nonzero(
                                (torch.sign(current) == torch.sign(final)) & confident
                            ).item()
                        )
                        covered += int(torch.count_nonzero(confident).item())
                        total += int(current.numel())
                    rows.append(
                        {
                            "budget_multiplier": run["budget_multiplier"],
                            "arm_id": run["arm"],
                            "seed": run["seed"],
                            "step": step,
                            "terminal_step": terminal_step,
                            "trajectory_fraction": step / max(terminal_step, 1),
                            "band": band,
                            "threshold": float(threshold),
                            "alignment": aligned / covered if covered else None,
                            "confident_coverage": covered / total if total else None,
                            "confident_coordinate_count": covered,
                        }
                    )
    return rows


def terminal_acquisition_rows(
    runs: Sequence[Mapping[str, Any]], *, time_bin_count: int
) -> list[dict[str, Any]]:
    import torch

    rows = []
    for run in runs:
        checkpoint_path = run["summary"].get("terminal_checkpoint_path")
        checkpoint = (
            Path(str(checkpoint_path))
            if checkpoint_path
            else run["run_dir"] / "checkpoints/latest.pt"
        )
        if not checkpoint.is_file():
            raise SignDynamicsAnalysisError(
                f"terminal resumable checkpoint is missing: {checkpoint}"
            )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("sign_dynamics_state")
        if not isinstance(state, Mapping):
            raise SignDynamicsAnalysisError(
                f"checkpoint has no sign-dynamics state: {checkpoint}"
            )
        terminal_step = int(state["measured_steps"])
        slices_by_band = _support_slices_by_band(run["support"])
        for threshold, last_by_name in state["last_transition_steps"].items():
            for band, slices in slices_by_band.items():
                groups: dict[tuple[int, int], int] = defaultdict(int)
                total = 0
                for name, slice_record in slices:
                    last_steps = _slice_tensor(last_by_name[name], slice_record).long()
                    last_actions = _slice_tensor(
                        state["last_transition_actions"][threshold][name], slice_record
                    ).long()
                    established_steps = _slice_tensor(
                        state["establishment_steps"][threshold][name], slice_record
                    ).long()
                    established_actions = _slice_tensor(
                        state["establishment_actions"][threshold][name], slice_record
                    ).long()
                    acquisition_steps = torch.where(
                        last_steps >= 0, last_steps, established_steps
                    )
                    acquisition_actions = torch.where(
                        last_steps >= 0, last_actions, established_actions
                    )
                    for acquisition_step, action_index in zip(
                        acquisition_steps.tolist(),
                        acquisition_actions.tolist(),
                        strict=True,
                    ):
                        if int(acquisition_step) < 0:
                            continue
                        time_bin = min(
                            time_bin_count - 1,
                            int(
                                max(int(acquisition_step) - 1, 0)
                                * time_bin_count
                                / max(terminal_step, 1)
                            ),
                        )
                        groups[(int(action_index), time_bin)] += 1
                        total += 1
                ordered = run["support"]["ordered_granularities"]
                for (action_index, time_bin), count in groups.items():
                    rows.append(
                        {
                            "budget_multiplier": run["budget_multiplier"],
                            "arm_id": run["arm"],
                            "seed": run["seed"],
                            "threshold": float(threshold),
                            "band": band,
                            "acquisition_action": (
                                ordered[action_index]
                                if 0 <= action_index < len(ordered)
                                else "initial"
                            ),
                            "acquisition_time_bin": time_bin,
                            "coordinate_count": count,
                            "acquisition_rate": count / max(total, 1),
                            "confident_terminal_coordinate_count": total,
                        }
                    )
    return rows


def _weighted_mean(pairs: Iterable[tuple[Any, Any]]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for raw_value, raw_weight in pairs:
        value = _finite_float(raw_value)
        weight = _finite_float(raw_weight)
        if value is None or weight is None or weight <= 0.0:
            continue
        numerator += value * weight
        denominator += weight
    return numerator / denominator if denominator else None


def _weighted_quantile(
    pairs: Iterable[tuple[Any, Any]], quantile: float
) -> float | None:
    values = sorted(
        (float(value), float(weight))
        for value, weight in pairs
        if _finite_float(value) is not None
        and _finite_float(weight) is not None
        and float(weight) > 0.0
    )
    total = sum(weight for _, weight in values)
    if total == 0.0:
        return None
    target = quantile * total
    cumulative = 0.0
    for value, weight in values:
        cumulative += weight
        if cumulative >= target:
            return value
    return values[-1][0]


def build_run_performance_dynamics_rows(
    validation_rows: Sequence[Mapping[str, Any]],
    alignment: Sequence[Mapping[str, Any]],
    acquisition: Sequence[Mapping[str, Any]],
    *,
    time_bin_count: int,
) -> list[dict[str, Any]]:
    """Create run/width endpoints for cross-run association, never winner ranking."""

    zero_rows = [row for row in validation_rows if float(row["threshold"]) == 0.0]
    positive_thresholds = sorted(
        {float(row["threshold"]) for row in validation_rows if row["threshold"] > 0.0}
    )
    robust_focus = 0.001 if 0.001 in positive_thresholds else (
        positive_thresholds[0] if positive_thresholds else 0.0
    )
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in zero_rows:
        groups[
            (
                row["budget_multiplier"],
                row["arm_id"],
                row["seed"],
                row["granularity"],
            )
        ].append(row)
    robust_lookup: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in validation_rows:
        if float(row["threshold"]) == robust_focus:
            robust_lookup[
                (
                    row["budget_multiplier"],
                    row["arm_id"],
                    row["seed"],
                    row["granularity"],
                )
            ].append(row)

    output = []
    for key, raw_rows in groups.items():
        ordered = sorted(raw_rows, key=lambda row: int(row["validation_step"]))
        terminal = ordered[-1]
        active_bands = set(json.loads(str(terminal["active_bands"])))
        early_raw = [
            row for row in ordered if float(row["trajectory_fraction"]) <= 0.10
        ]
        early_robust = [
            row
            for row in robust_lookup.get(key, [])
            if float(row["trajectory_fraction"]) <= 0.10
        ]
        matching_alignment = [
            row
            for row in alignment
            if (
                row["budget_multiplier"],
                row["arm_id"],
                row["seed"],
            )
            == key[:3]
            and row["band"] in active_bands
            and float(row["threshold"]) == robust_focus
        ]
        half_steps = {}
        for band in active_bands:
            band_rows = [row for row in matching_alignment if row["band"] == band]
            if band_rows:
                half_steps[band] = min(
                    band_rows,
                    key=lambda row: abs(float(row["trajectory_fraction"]) - 0.5),
                )["step"]
        half_alignment = [
            row
            for row in matching_alignment
            if row["band"] in half_steps and row["step"] == half_steps[row["band"]]
        ]
        matching_acquisition = [
            row
            for row in acquisition
            if (
                row["budget_multiplier"],
                row["arm_id"],
                row["seed"],
            )
            == key[:3]
            and row["band"] in active_bands
            and float(row["threshold"]) == robust_focus
        ]
        acquisition_pairs = [
            (
                (float(row["acquisition_time_bin"]) + 0.5) / time_bin_count,
                row["coordinate_count"],
            )
            for row in matching_acquisition
        ]
        output.append(
            {
                "budget_multiplier": key[0],
                "arm_id": key[1],
                "seed": key[2],
                "granularity": key[3],
                "active_bands": terminal["active_bands"],
                "robust_focus_threshold": robust_focus,
                "validation_checkpoint_count": len(ordered),
                "terminal_validation_step": terminal["validation_step"],
                "terminal_validation_loss": terminal["validation_loss"],
                "terminal_validation_perplexity": terminal[
                    "validation_perplexity"
                ],
                "mean_validation_loss": _mean(
                    row["validation_loss"] for row in ordered
                ),
                "early_mean_raw_sign_flip_rate": _weighted_mean(
                    (row["raw_sign_flip_rate"], row["interval_steps"])
                    for row in early_raw
                ),
                "early_mean_robust_transition_rate": _weighted_mean(
                    (row["robust_transition_rate"], row["interval_steps"])
                    for row in early_robust
                ),
                "early_mean_raw_flips_per_relative_update": _weighted_mean(
                    (row["raw_flips_per_relative_update"], row["interval_steps"])
                    for row in early_raw
                ),
                "early_validation_loss_improvement": (
                    early_raw[0]["validation_loss"]
                    - early_raw[-1]["validation_loss"]
                    if len(early_raw) > 1
                    else None
                ),
                "terminal_alignment_at_half": _weighted_mean(
                    (row["alignment"], row["confident_coordinate_count"])
                    for row in half_alignment
                ),
                "mean_terminal_sign_acquisition_fraction": _weighted_mean(
                    acquisition_pairs
                ),
                "median_terminal_sign_acquisition_fraction": _weighted_quantile(
                    acquisition_pairs, 0.5
                ),
                "p90_terminal_sign_acquisition_fraction": _weighted_quantile(
                    acquisition_pairs, 0.9
                ),
            }
        )
    return output


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def _matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    row_field: str,
    column_field: str,
    value_field: str,
) -> tuple[list[str], list[str], list[list[float]]]:
    row_labels = sorted({str(row[row_field]) for row in rows})
    column_labels = sorted({str(row[column_field]) for row in rows})
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_field)
        if value is not None and math.isfinite(float(value)):
            grouped[(str(row[row_field]), str(row[column_field]))].append(float(value))
    values = [
        [
            _mean(grouped[(row_label, column_label)]) or 0.0
            for column_label in column_labels
        ]
        for row_label in row_labels
    ]
    return row_labels, column_labels, values


def _heatmap(ax, rows, *, row_field, column_field, value_field, title):
    if not rows:
        ax.text(0.5, 0.5, "No observations", ha="center", va="center")
        ax.set_axis_off()
        ax.set_title(title)
        return None
    labels_y, labels_x, values = _matrix(
        rows, row_field=row_field, column_field=column_field, value_field=value_field
    )
    image = ax.imshow(values, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(labels_x)), labels_x, rotation=35, ha="right")
    ax.set_yticks(range(len(labels_y)), labels_y)
    ax.set_title(title)
    return image


def _granularity_axes(pyplot, granularities: Sequence[str], *, title: str):
    column_count = min(2, max(len(granularities), 1))
    row_count = max(1, math.ceil(max(len(granularities), 1) / column_count))
    figure, raw_axes = pyplot.subplots(
        row_count,
        column_count,
        figsize=(6.2 * column_count, 4.4 * row_count),
        constrained_layout=True,
        squeeze=False,
    )
    axes = list(raw_axes.flat)
    figure.suptitle(title)
    for axis in axes[len(granularities) :]:
        axis.set_axis_off()
    return figure, axes


def _ordered_validation_granularities(
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    support_sizes = {}
    for row in rows:
        try:
            support_sizes[str(row["granularity"])] = len(
                json.loads(str(row["active_bands"]))
            )
        except (KeyError, TypeError, json.JSONDecodeError):
            support_sizes.setdefault(str(row["granularity"]), len(support_sizes))
    return sorted(support_sizes, key=lambda value: (support_sizes[value], value))


def _correlation_value(
    rows: Sequence[Mapping[str, Any]],
    *,
    granularity: str,
    metric: str,
    relationship: str,
    phase: str,
    field: str = "pearson",
) -> float | None:
    return _mean(
        row[field]
        for row in rows
        if row["granularity"] == granularity
        and row["metric"] == metric
        and row["relationship"] == relationship
        and row["phase"] == phase
        and row.get(field) is not None
    )


def _performance_scatter_figure(
    pyplot,
    rows: Sequence[Mapping[str, Any]],
    correlations: Sequence[Mapping[str, Any]],
    *,
    x_field: str,
    metric: str,
    relationship: str,
    title: str,
    x_label: str,
):
    granularities = _ordered_validation_granularities(rows)
    if not granularities:
        granularities = ["no-validation"]
    figure, axes = _granularity_axes(pyplot, granularities, title=title)
    scatter = None
    for axis, granularity in zip(axes, granularities, strict=False):
        selected = sorted(
            (row for row in rows if row["granularity"] == granularity),
            key=lambda row: (
                row["budget_multiplier"],
                row["arm_id"],
                row["seed"],
                row["validation_step"],
            ),
        )
        x_values = []
        y_values = []
        progress = []
        for index, row in enumerate(selected):
            if relationship == "concurrent":
                target = row.get("validation_loss_improvement_per_1000_steps")
            else:
                target = None
                if index + 1 < len(selected):
                    next_row = selected[index + 1]
                    same_run = (
                        row["budget_multiplier"], row["arm_id"], row["seed"]
                    ) == (
                        next_row["budget_multiplier"],
                        next_row["arm_id"],
                        next_row["seed"],
                    )
                    if same_run:
                        target = next_row.get(
                            "validation_loss_improvement_per_1000_steps"
                        )
            x_value = _finite_float(row.get(x_field))
            y_value = _finite_float(target)
            if x_value is None or y_value is None:
                continue
            x_values.append(x_value)
            y_values.append(y_value)
            progress.append(float(row["trajectory_fraction"]))
        if x_values:
            scatter = axis.scatter(
                x_values,
                y_values,
                c=progress,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                s=10,
                alpha=0.45,
                linewidths=0,
                rasterized=True,
            )
            if min(x_values) >= 0.0 and max(x_values) > 0.0:
                positive = sorted(value for value in x_values if value > 0.0)
                linthresh = positive[max(0, len(positive) // 20 - 1)] if positive else 1e-12
                axis.set_xscale("symlog", linthresh=max(linthresh, 1e-12))
            axis.set_yscale("symlog", linthresh=1e-4)
        else:
            axis.text(0.5, 0.5, "No validation pairs", ha="center", va="center")
        all_correlation = _correlation_value(
            correlations,
            granularity=granularity,
            metric=metric,
            relationship=relationship,
            phase="all",
        )
        post_correlation = _correlation_value(
            correlations,
            granularity=granularity,
            metric=metric,
            relationship=relationship,
            phase="post_10_100",
        )
        axis.set_title(
            f"{granularity}  r(all)={all_correlation:.3f}, r(post-10%)={post_correlation:.3f}"
            if all_correlation is not None and post_correlation is not None
            else granularity
        )
        axis.set_xlabel(x_label)
        axis.set_ylabel("validation-loss decrease per 1,000 steps")
        axis.axhline(0.0, color="0.5", linewidth=0.7)
    if scatter is not None:
        figure.colorbar(scatter, ax=axes[: len(granularities)], label="training progress")
    return figure


def _correlation_heatmap(
    axis,
    rows: Sequence[Mapping[str, Any]],
    *,
    granularities: Sequence[str],
    phases: Sequence[str],
    relationship: str,
    field: str,
    title: str,
):
    import numpy

    values = numpy.full((len(granularities), len(phases)), numpy.nan)
    for row_index, granularity in enumerate(granularities):
        for column_index, phase in enumerate(phases):
            value = _correlation_value(
                rows,
                granularity=granularity,
                metric="raw_sign_flip_rate",
                relationship=relationship,
                phase=phase,
                field=field,
            )
            if value is not None:
                values[row_index, column_index] = value
    image = axis.imshow(
        values, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-1, vmax=1
    )
    phase_labels = {
        "early_0_10": "0–10%",
        "middle_10_50": "10–50%",
        "late_50_100": "50–100%",
        "post_10_100": ">10%",
    }
    axis.set_xticks(
        range(len(phases)), [phase_labels.get(phase, phase) for phase in phases]
    )
    axis.set_yticks(range(len(granularities)), granularities)
    axis.set_title(title)
    for row_index in range(len(granularities)):
        for column_index in range(len(phases)):
            value = values[row_index, column_index]
            if numpy.isfinite(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(value) > 0.55 else "black",
                    fontsize=8,
                )
    return image


def generate_performance_figures(
    validation_rows: Sequence[Mapping[str, Any]],
    correlations: Sequence[Mapping[str, Any]],
    run_performance: Sequence[Mapping[str, Any]],
    figures_dir: Path,
) -> list[Path]:
    pyplot = _pyplot()
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    zero_rows = [row for row in validation_rows if float(row["threshold"]) == 0.0]
    positive_thresholds = sorted(
        {float(row["threshold"]) for row in validation_rows if row["threshold"] > 0.0}
    )
    robust_focus = 0.001 if 0.001 in positive_thresholds else (
        positive_thresholds[0] if positive_thresholds else 0.0
    )
    robust_rows = [
        row for row in validation_rows if float(row["threshold"]) == robust_focus
    ]

    granularities = _ordered_validation_granularities(zero_rows)
    fig, axes = pyplot.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    colors = {
        granularity: pyplot.get_cmap("tab10")(index % 10)
        for index, granularity in enumerate(granularities)
    }
    seen_labels: set[str] = set()
    run_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in zero_rows:
        run_groups[
            (
                row["budget_multiplier"],
                row["arm_id"],
                row["seed"],
                row["granularity"],
            )
        ].append(row)
    granularity_order = {value: index for index, value in enumerate(granularities)}
    for key, rows in sorted(
        run_groups.items(),
        key=lambda item: (granularity_order[str(item[0][3])], item[0][:3]),
    ):
        selected = sorted(rows, key=lambda row: int(row["validation_step"]))
        granularity = str(key[3])
        label = granularity if granularity not in seen_labels else None
        seen_labels.add(granularity)
        axes[0].plot(
            [row["trajectory_fraction"] for row in selected],
            [row["validation_loss"] for row in selected],
            color=colors[granularity],
            alpha=0.7,
            linewidth=1.0,
            label=label,
        )
        axes[1].plot(
            [row["trajectory_fraction"] for row in selected],
            [row["raw_sign_flip_rate"] for row in selected],
            color=colors[granularity],
            alpha=0.7,
            linewidth=1.0,
            label=label,
        )
    for axis in axes:
        if not zero_rows:
            axis.text(0.5, 0.5, "No ordinary-validation records", ha="center", va="center")
        axis.set_xlabel("training progress")
    axes[0].set_ylabel("ordinary-validation loss")
    axes[0].set_title("Performance trajectory")
    axes[1].set_ylabel("raw flip rate over cumulative width support")
    axes[1].set_yscale("symlog", linthresh=1e-7)
    axes[1].set_title("Sign-flip trajectory")
    if zero_rows:
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
    path = figures_dir / "performance_and_flip_trajectories.png"
    fig.savefig(path, dpi=180)
    pyplot.close(fig)
    outputs.append(path)

    scatter_specs = (
        (
            zero_rows,
            "raw_sign_flip_rate",
            "raw_sign_flip_rate",
            "concurrent",
            "Raw sign flipping versus concurrent validation improvement",
            "raw sign-flip rate",
            "performance_raw_flip_concurrent.png",
        ),
        (
            robust_rows,
            "robust_transition_rate",
            "robust_transition_rate",
            "concurrent",
            f"Robust sign transitions (tau={robust_focus:g}) versus concurrent improvement",
            "robust transition rate",
            "performance_robust_flip_concurrent.png",
        ),
        (
            zero_rows,
            "raw_flips_per_relative_update",
            "raw_flips_per_relative_update",
            "concurrent",
            "Update-normalized sign flipping versus concurrent improvement",
            "raw flip rate / relative-update RMS",
            "performance_update_normalized_concurrent.png",
        ),
        (
            robust_rows,
            "robust_transitions_per_relative_update",
            "robust_transitions_per_relative_update",
            "next_interval",
            f"Update-normalized robust transitions versus next-interval improvement (tau={robust_focus:g})",
            "robust transition rate / relative-update RMS",
            "performance_robust_flip_lagged.png",
        ),
    )
    for rows, x_field, metric, relationship, title, x_label, filename in scatter_specs:
        fig = _performance_scatter_figure(
            pyplot,
            rows,
            correlations,
            x_field=x_field,
            metric=metric,
            relationship=relationship,
            title=title,
            x_label=x_label,
        )
        path = figures_dir / filename
        fig.savefig(path, dpi=180)
        pyplot.close(fig)
        outputs.append(path)

    fig, axes = pyplot.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    phases = ("early_0_10", "middle_10_50", "late_50_100", "post_10_100")
    heatmap_specs = (
        ("concurrent", "pearson", "Concurrent Pearson r"),
        ("concurrent", "partial_pearson", "Concurrent partial r"),
        ("next_interval", "pearson", "Next-interval Pearson r"),
        ("next_interval", "partial_pearson", "Next-interval partial r"),
    )
    last_image = None
    for axis, (relationship, field, title) in zip(
        axes.flat, heatmap_specs, strict=True
    ):
        if granularities:
            last_image = _correlation_heatmap(
                axis,
                correlations,
                granularities=granularities,
                phases=phases,
                relationship=relationship,
                field=field,
                title=title,
            )
        else:
            axis.text(0.5, 0.5, "No validation correlations", ha="center", va="center")
            axis.set_axis_off()
    if last_image is not None:
        fig.colorbar(last_image, ax=list(axes.flat), label="correlation")
    fig.suptitle(
        "Raw flip–performance correlations by phase\n"
        "Partial r controls progress, learning rate, update RMS, and support exposure"
    )
    path = figures_dir / "performance_phase_correlations.png"
    fig.savefig(path, dpi=180)
    pyplot.close(fig)
    outputs.append(path)

    terminal_granularities = _ordered_validation_granularities(run_performance)
    terminal_specs = (
        (
            "early_mean_raw_flips_per_relative_update",
            "early update-normalized flip rate (first 10%)",
        ),
        (
            "mean_terminal_sign_acquisition_fraction",
            "mean terminal-sign acquisition fraction",
        ),
    )
    column_count = max(len(terminal_granularities), 1)
    fig, axes = pyplot.subplots(
        2,
        column_count,
        figsize=(4.3 * column_count, 8.0),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, (x_field, x_label) in enumerate(terminal_specs):
        for column_index in range(column_count):
            axis = axes[row_index][column_index]
            if column_index >= len(terminal_granularities):
                axis.text(
                    0.5,
                    0.5,
                    "No ordinary-validation endpoints",
                    ha="center",
                    va="center",
                )
                axis.set_axis_off()
                continue
            granularity = terminal_granularities[column_index]
            selected = [
                row
                for row in run_performance
                if str(row["granularity"]) == granularity
                and _finite_float(row.get(x_field)) is not None
                and _finite_float(row.get("terminal_validation_loss")) is not None
            ]
            for index, row in enumerate(selected):
                x_value = float(row[x_field])
                y_value = float(row["terminal_validation_loss"])
                axis.scatter(
                    x_value,
                    y_value,
                    color=pyplot.get_cmap("tab10")(index % 10),
                    s=32,
                    alpha=0.8,
                )
                axis.annotate(
                    f"x{row['budget_multiplier']}:{row['arm_id']}:s{row['seed']}",
                    (x_value, y_value),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                    alpha=0.75,
                )
            if len(selected) < 2:
                axis.text(
                    0.5,
                    0.08,
                    f"n={len(selected)} run: no correlation estimate",
                    transform=axis.transAxes,
                    ha="center",
                    fontsize=8,
                )
            axis.set_xlabel(x_label)
            if column_index == 0:
                axis.set_ylabel("terminal ordinary-validation loss")
            axis.set_title(granularity)
    fig.suptitle(
        "Terminal performance versus sign dynamics within width\n"
        "Descriptive endpoints; infer within a fixed budget using independent runs"
    )
    path = figures_dir / "terminal_performance_dynamics.png"
    fig.savefig(path, dpi=180)
    pyplot.close(fig)
    outputs.append(path)
    return outputs


def generate_diagnostic_figures(
    step_rows: Sequence[Mapping[str, Any]],
    time_rows: Sequence[Mapping[str, Any]],
    alignment: Sequence[Mapping[str, Any]],
    acquisition: Sequence[Mapping[str, Any]],
    figures_dir: Path,
) -> list[Path]:
    pyplot = _pyplot()
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    zero_rows = [row for row in step_rows if float(row["threshold"]) == 0.0]
    fig, axes = pyplot.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, field, title in zip(
        axes,
        ("raw_sign_flip_rate", "robust_transition_rate"),
        ("Raw sign flips", "Hysteretic transitions (tau=0)"),
        strict=True,
    ):
        image = _heatmap(
            ax,
            zero_rows,
            row_field="action",
            column_field="band",
            value_field=field,
            title=title,
        )
        if image is not None:
            fig.colorbar(image, ax=ax)
    path = figures_dir / "raw_robust_flip_heatmaps.png"
    fig.savefig(path, dpi=180)
    pyplot.close(fig)
    outputs.append(path)

    normalized = []
    band_means = {
        band: _mean(
            row["raw_sign_flip_rate"] for row in zero_rows if row["band"] == band
        )
        or 0.0
        for band in {row["band"] for row in zero_rows}
    }
    for row in zero_rows:
        baseline = band_means[row["band"]]
        normalized.append(
            {
                **row,
                "normalized_effect": float(row["raw_sign_flip_rate"]) / baseline
                if baseline
                else 0.0,
            }
        )
    fig, ax = pyplot.subplots(figsize=(7, 5), constrained_layout=True)
    image = _heatmap(
        ax,
        normalized,
        row_field="action",
        column_field="band",
        value_field="normalized_effect",
        title="Action effect / band mean",
    )
    if image is not None:
        fig.colorbar(image, ax=ax)
    path = figures_dir / "normalized_action_effect_heatmaps.png"
    fig.savefig(path, dpi=180)
    pyplot.close(fig)
    outputs.append(path)

    off_support = [row for row in zero_rows if not bool(row["forward_active"])]
    fig, ax = pyplot.subplots(figsize=(7, 5), constrained_layout=True)
    for arm in sorted({row["arm_id"] for row in off_support}):
        selected = [row for row in off_support if row["arm_id"] == arm]
        ax.scatter(
            [row["gradient_rms"] for row in selected],
            [row["relative_update_rms"] for row in selected],
            s=9,
            alpha=0.35,
            label=arm,
        )
    ax.set_xlabel("off-support gradient RMS")
    ax.set_ylabel("off-support relative update RMS")
    ax.set_title("Gradient versus optimizer-mediated update")
    if off_support:
        ax.legend(fontsize=7)
    path = figures_dir / "gradient_update_off_support.png"
    fig.savefig(path, dpi=180)
    pyplot.close(fig)
    outputs.append(path)

    fig, ax = pyplot.subplots(figsize=(9, 5), constrained_layout=True)
    trajectory_rows = [row for row in time_rows if float(row["threshold"]) == 0.0]
    for band in sorted({row["band"] for row in trajectory_rows}):
        selected = [row for row in trajectory_rows if row["band"] == band]
        bins = sorted({int(row["time_bin"]) for row in selected})
        ax.plot(
            bins,
            [
                _mean(
                    row["raw_sign_flip_rate"]
                    for row in selected
                    if int(row["time_bin"]) == value
                )
                for value in bins
            ],
            label=band,
        )
    ax.set_xlabel("training time bin")
    ax.set_ylabel("raw flip rate")
    ax.set_title("Time-resolved per-band sign dynamics")
    ax.legend()
    path = figures_dir / "time_resolved_band_trajectories.png"
    fig.savefig(path, dpi=180)
    pyplot.close(fig)
    outputs.append(path)

    fig, axes = pyplot.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    configured_thresholds = sorted({float(row["threshold"]) for row in step_rows})
    robust_focus = (
        0.001
        if 0.001 in configured_thresholds
        else next(
            (threshold for threshold in configured_thresholds if threshold > 0.0),
            0.0,
        )
    )
    for band in sorted({row["band"] for row in alignment}):
        selected = [
            row
            for row in alignment
            if row["band"] == band and float(row["threshold"]) == robust_focus
        ]
        fractions = sorted({float(row["trajectory_fraction"]) for row in selected})
        axes[0].plot(
            fractions,
            [
                _mean(
                    row["alignment"]
                    for row in selected
                    if float(row["trajectory_fraction"]) == value
                )
                for value in fractions
            ],
            label=band,
        )
    axes[0].set_title("Terminal-sign alignment")
    axes[0].set_xlabel("trajectory fraction")
    axes[0].legend()
    for band in sorted({row["band"] for row in zero_rows}):
        selected = [row for row in zero_rows if row["band"] == band]
        axes[1].plot(
            [row["step"] for row in selected],
            [row["last_robust_transition_step"] or 0 for row in selected],
            ".",
            alpha=0.15,
            label=band,
        )
    axes[1].set_title("Last robust-transition curve")
    axes[1].set_xlabel("measurement step")
    axes[1].legend()
    path = figures_dir / "terminal_alignment_last_transition.png"
    fig.savefig(path, dpi=180)
    pyplot.close(fig)
    outputs.append(path)

    acquisition_focus = [
        row for row in acquisition if float(row["threshold"]) == robust_focus
    ]
    fig, ax = pyplot.subplots(figsize=(7, 5), constrained_layout=True)
    image = _heatmap(
        ax,
        acquisition_focus,
        row_field="acquisition_action",
        column_field="band",
        value_field="acquisition_rate",
        title="Terminal-sign acquisition action",
    )
    if image is not None:
        fig.colorbar(image, ax=ax)
    path = figures_dir / "terminal_sign_acquisition_heatmaps.png"
    fig.savefig(path, dpi=180)
    pyplot.close(fig)
    outputs.append(path)

    h_rows = [row for row in zero_rows if row.get("h_window_position") is not None]
    fig, ax = pyplot.subplots(figsize=(9, 5), constrained_layout=True)
    for band in sorted({row["band"] for row in h_rows}):
        selected = [row for row in h_rows if row["band"] == band]
        positions = sorted({int(row["h_window_position"]) for row in selected})
        ax.plot(
            positions,
            [
                _mean(
                    row["raw_sign_flip_rate"]
                    for row in selected
                    if int(row["h_window_position"]) == value
                )
                for value in positions
            ],
            label=band,
        )
    ax.set_xlabel("H-window position")
    ax.set_ylabel("raw flip rate")
    ax.set_title("Within-window comparisons")
    if h_rows:
        ax.legend()
    path = figures_dir / "h_window_position_comparisons.png"
    fig.savefig(path, dpi=180)
    pyplot.close(fig)
    outputs.append(path)
    return outputs


def analyze(
    *,
    campaign_root: Path,
    analysis_dir: Path,
    figures_dir: Path,
    time_bins: int,
    workers: int = 1,
) -> dict[str, Any]:
    if workers <= 0:
        raise SignDynamicsAnalysisError("analysis workers must be positive")
    _progress("discovering runs and validating journals")
    runs = discover_runs(campaign_root)
    _progress(
        f"loaded {len(runs)} run(s) and "
        f"{sum(len(run['records']) for run in runs):,} committed step record(s)"
    )
    _progress("joining ordinary validation to preceding sign-dynamics intervals")
    validation_dynamics = build_validation_dynamics_rows(runs)
    _progress(
        f"built {len(validation_dynamics):,} validation/dynamics rows "
        "over cumulative width supports"
    )
    performance_correlations = build_performance_correlation_rows(
        validation_dynamics
    )
    _progress("expanding per-band and per-threshold step rows")
    step_rows = flatten_step_rows(runs)
    for run in runs:
        run.pop("records", None)
    gc.collect()
    _progress(
        f"aggregating {len(step_rows):,} step rows into {time_bins} time bins "
        f"with {workers} worker(s)"
    )
    time_rows = build_time_bin_rows(
        step_rows, time_bin_count=time_bins, workers=workers
    )
    _progress("computing terminal-sign alignment from milestone snapshots")
    alignment = terminal_alignment_rows(runs)
    _progress("computing terminal-sign acquisition from checkpoint state")
    acquisition = terminal_acquisition_rows(runs, time_bin_count=time_bins)
    run_performance = build_run_performance_dynamics_rows(
        validation_dynamics,
        alignment,
        acquisition,
        time_bin_count=time_bins,
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "validation_dynamics": analysis_dir / "validation_sign_dynamics.csv",
        "performance_correlations": analysis_dir
        / "performance_sign_correlations.csv",
        "run_performance_dynamics": analysis_dir
        / "run_performance_dynamics.csv",
        "per_step": analysis_dir / "sign_dynamics_per_step.csv",
        "time_bins": analysis_dir / "sign_dynamics_time_bins.csv",
        "terminal_alignment": analysis_dir / "terminal_alignment.csv",
        "terminal_sign_acquisition": analysis_dir / "terminal_sign_acquisition.csv",
    }
    for name, path, rows in zip(
        tables,
        tables.values(),
        (
            validation_dynamics,
            performance_correlations,
            run_performance,
            step_rows,
            time_rows,
            alignment,
            acquisition,
        ),
        strict=True,
    ):
        _progress(f"writing {name} table ({len(rows):,} rows) to {path}")
        _write_csv(path, rows)

    seeds_by_panel: dict[tuple[Any, str], set[int]] = defaultdict(set)
    for run in runs:
        seeds_by_panel[(run["budget_multiplier"], run["arm"])].add(run["seed"])
    seed_sets = list(seeds_by_panel.values())
    provisional = (
        len({tuple(sorted(values)) for values in seed_sets}) != 1
        or min(len(values) for values in seed_sets) < 2
    )
    _progress(f"generating performance-linked figures in {figures_dir}")
    primary_figures = generate_performance_figures(
        validation_dynamics,
        performance_correlations,
        run_performance,
        figures_dir,
    )
    _progress(f"generating secondary diagnostic figures in {figures_dir}")
    diagnostic_figures = generate_diagnostic_figures(
        step_rows, time_rows, alignment, acquisition, figures_dir
    )
    figures = primary_figures + diagnostic_figures
    summary = {
        "schema_version": 2,
        "campaign_root": str(campaign_root),
        "analysis_focus": "performance_sign_dynamics_correlation",
        "status": "provisional" if provisional else "complete_multi_seed",
        "uncertainty": (
            "seed_level_when_multiple; H-window block exploratory when single"
        ),
        "run_count": len(runs),
        "workers": workers,
        "budgets": sorted({run["budget_multiplier"] for run in runs}),
        "arms": sorted({run["arm"] for run in runs}),
        "seed_panels": {
            f"x{budget}/{arm}": sorted(seeds)
            for (budget, arm), seeds in sorted(seeds_by_panel.items())
        },
        "measurement_contract_hash": stable_hash(runs[0]["contract"]),
        "performance_analysis_status": (
            "available" if validation_dynamics else "no_validation_records"
        ),
        "validation_dynamics_row_count": len(validation_dynamics),
        "tables": {name: str(path) for name, path in tables.items()},
        "primary_figures": [str(path) for path in primary_figures],
        "diagnostic_figures": [str(path) for path in diagnostic_figures],
        "figures": [str(path) for path in figures],
    }
    summary["analysis_hash"] = stable_hash(summary)
    summary_path = analysis_dir / "sign_dynamics_analysis.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _progress(f"analysis complete: {summary_path}")
    return summary


def _default_worker_count() -> int:
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return max(1, min(8, available))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--time-bins", type=int, default=20)
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_worker_count(),
        help="CPU workers for time-bin aggregation (default: up to 8 available CPUs)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.time_bins <= 0:
        raise SystemExit("--time-bins must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    import torch

    torch.set_num_threads(args.workers)
    campaign_root = args.campaign_root.expanduser().resolve()
    analysis_dir = (
        args.analysis_dir.expanduser().resolve()
        if args.analysis_dir
        else campaign_root / "analysis"
    )
    figures_dir = (
        args.figures_dir.expanduser().resolve()
        if args.figures_dir
        else campaign_root / "figures"
    )
    try:
        summary = analyze(
            campaign_root=campaign_root,
            analysis_dir=analysis_dir,
            figures_dir=figures_dir,
            time_bins=args.time_bins,
            workers=args.workers,
        )
    except SignDynamicsAnalysisError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
