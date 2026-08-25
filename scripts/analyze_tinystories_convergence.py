#!/usr/bin/env python3
"""Select a converged TinyStories dense recipe from completed run artifacts."""

# ruff: noqa: E402  # Add the repository root before importing src.

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.reproducibility import stable_hash


DEFAULT_MINIMUM_STEP = 512
DEFAULT_PATIENCE_EVALUATIONS = 5
DEFAULT_RELATIVE_IMPROVEMENT = 0.005
DEFAULT_TIE_TOLERANCE = 0.001


class ConvergenceAnalysisError(ValueError):
    """Raised when run artifacts cannot be analyzed safely."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", action="append", default=[])
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-step", type=int, default=DEFAULT_MINIMUM_STEP)
    parser.add_argument(
        "--patience-evaluations",
        type=int,
        default=DEFAULT_PATIENCE_EVALUATIONS,
    )
    parser.add_argument(
        "--relative-improvement",
        type=float,
        default=DEFAULT_RELATIVE_IMPROVEMENT,
    )
    parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=DEFAULT_TIE_TOLERANCE,
    )
    return parser.parse_args(argv)


def discover_run_dirs(
    runs_roots: Iterable[str | Path],
    run_dirs: Iterable[str | Path],
) -> list[Path]:
    discovered = {Path(path).expanduser().resolve() for path in run_dirs}
    for raw_root in runs_roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise ConvergenceAnalysisError(f"Runs root does not exist: {root}")
        discovered.update(path.parent for path in root.rglob("run_summary.json"))
    if not discovered:
        raise ConvergenceAnalysisError("No run directories were supplied or discovered")
    return sorted(discovered)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConvergenceAnalysisError(f"Cannot read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ConvergenceAnalysisError(f"JSON artifact must be a mapping: {path}")
    return value


def _read_metrics(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            return list(csv.DictReader(source))
    except (OSError, csv.Error, UnicodeError) as error:
        raise ConvergenceAnalysisError(f"Cannot read metrics artifact: {path}") from error


def _finite_metric_losses(rows: Iterable[Mapping[str, Any]]) -> bool:
    for row in rows:
        raw_loss = row.get("loss")
        if raw_loss in (None, ""):
            continue
        try:
            loss = float(raw_loss)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(loss):
            return False
    return True


def convergence_evidence(
    validation_rows: Iterable[Mapping[str, Any]],
    *,
    minimum_step: int = DEFAULT_MINIMUM_STEP,
    patience_evaluations: int = DEFAULT_PATIENCE_EVALUATIONS,
    relative_improvement: float = DEFAULT_RELATIVE_IMPROVEMENT,
) -> dict[str, Any]:
    if minimum_step < 0:
        raise ConvergenceAnalysisError("minimum_step must be nonnegative")
    if patience_evaluations <= 0:
        raise ConvergenceAnalysisError("patience_evaluations must be positive")
    if not 0.0 < relative_improvement < 1.0:
        raise ConvergenceAnalysisError("relative_improvement must be between 0 and 1")

    observations = sorted(
        (
            (int(row["step"]), float(row["loss"]))
            for row in validation_rows
            if row.get("step") not in (None, "")
            and row.get("loss") not in (None, "")
            and int(row["step"]) >= minimum_step
        ),
        key=lambda item: item[0],
    )
    if not observations:
        return {
            "converged": False,
            "eligible_evaluation_count": 0,
            "evaluations_since_significant_improvement": 0,
            "last_significant_improvement_step": None,
        }

    significant_best = observations[0][1]
    last_improvement_step = observations[0][0]
    since_improvement = 0
    for step, loss in observations[1:]:
        if loss <= significant_best * (1.0 - relative_improvement):
            significant_best = loss
            last_improvement_step = step
            since_improvement = 0
        else:
            since_improvement += 1
    return {
        "converged": since_improvement >= patience_evaluations,
        "eligible_evaluation_count": len(observations),
        "evaluations_since_significant_improvement": since_improvement,
        "last_significant_improvement_step": last_improvement_step,
        "significant_best_loss": significant_best,
    }


def analyze_run(
    run_dir: str | Path,
    *,
    minimum_step: int = DEFAULT_MINIMUM_STEP,
    patience_evaluations: int = DEFAULT_PATIENCE_EVALUATIONS,
    relative_improvement: float = DEFAULT_RELATIVE_IMPROVEMENT,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    summary_path = root / "run_summary.json"
    metrics_path = root / "metrics.csv"
    rejection_reasons: list[str] = []
    if not summary_path.is_file():
        return {
            "run_dir": str(root),
            "run_id": root.name,
            "stable": False,
            "converged": False,
            "rejection_reasons": ["missing run_summary.json"],
        }
    summary = _read_json(summary_path)
    if not metrics_path.is_file():
        return {
            "run_dir": str(root),
            "run_id": summary.get("run_id", root.name),
            "stable": False,
            "converged": False,
            "rejection_reasons": ["missing metrics.csv"],
        }
    metrics = _read_metrics(metrics_path)
    if summary.get("status") != "completed":
        rejection_reasons.append("run status is not completed")
    try:
        tokens_seen = int(summary.get("tokens_seen", -1))
        token_budget = int(summary.get("token_budget", -1))
    except (TypeError, ValueError):
        tokens_seen = -1
        token_budget = -1
    if token_budget <= 0 or tokens_seen < token_budget:
        rejection_reasons.append("run did not reach its requested token budget")
    if not _finite_metric_losses(metrics):
        rejection_reasons.append("metrics contain missing-format or non-finite losses")

    validation_rows = sorted(
        (
            row
            for row in metrics
            if row.get("split") == "validation"
            and row.get("granularity") == "g1000"
            and row.get("loss") not in (None, "")
        ),
        key=lambda row: int(row.get("step") or 0),
    )
    if not validation_rows:
        rejection_reasons.append("no dense g1000 validation observations")
    evidence = convergence_evidence(
        validation_rows,
        minimum_step=minimum_step,
        patience_evaluations=patience_evaluations,
        relative_improvement=relative_improvement,
    )
    losses = [float(row["loss"]) for row in validation_rows]
    best_validation_loss = min(losses) if losses else None
    best_validation_step = (
        min(validation_rows, key=lambda row: float(row["loss"])).get("step")
        if validation_rows
        else None
    )
    checkpoint_path_value = summary.get("best_checkpoint_path")
    checkpoint_exists = False
    if checkpoint_path_value not in (None, ""):
        configured_checkpoint = Path(str(checkpoint_path_value)).expanduser()
        checkpoint_candidates = [configured_checkpoint]
        if not configured_checkpoint.is_absolute():
            checkpoint_candidates.extend(
                (
                    root / configured_checkpoint,
                    root / "checkpoints" / configured_checkpoint.name,
                )
            )
        checkpoint_exists = any(
            candidate.is_file() and candidate.stat().st_size > 0
            for candidate in checkpoint_candidates
        )
    if summary.get("checkpoint_status") != "best_eval" or not checkpoint_exists:
        rejection_reasons.append("ordinary-validation best checkpoint is unavailable")
    if summary.get("unresolved_artifact_failures"):
        rejection_reasons.append("run reports unresolved artifact failures")
    learning_rate = summary.get(
        "base_learning_rate", summary.get("resolved_learning_rate")
    )
    try:
        learning_rate = float(learning_rate)
    except (TypeError, ValueError):
        learning_rate = None
    if learning_rate is None or not math.isfinite(learning_rate) or learning_rate <= 0:
        rejection_reasons.append("run learning rate is invalid")
    scheduler = summary.get("scheduler_name")
    if not isinstance(scheduler, str) or not scheduler.strip():
        rejection_reasons.append("run scheduler is invalid")

    wall_clock_values: list[float] = []
    invalid_wall_clock = False
    for row in metrics:
        raw_wall_clock = row.get("wall_clock_seconds")
        if raw_wall_clock in (None, ""):
            continue
        try:
            wall_clock = float(raw_wall_clock)
        except (TypeError, ValueError):
            invalid_wall_clock = True
            continue
        if not math.isfinite(wall_clock) or wall_clock < 0:
            invalid_wall_clock = True
            continue
        wall_clock_values.append(wall_clock)
    if invalid_wall_clock:
        rejection_reasons.append("metrics contain invalid wall-clock values")
    stable = not rejection_reasons
    result = {
        "run_dir": str(root),
        "run_id": summary.get("run_id", root.name),
        "stable": stable,
        "rejection_reasons": rejection_reasons,
        "best_validation_loss": best_validation_loss,
        "best_validation_step": (
            int(best_validation_step) if best_validation_step is not None else None
        ),
        "final_validation_loss": (
            float(validation_rows[-1]["loss"]) if validation_rows else None
        ),
        "wall_clock_seconds": max(wall_clock_values) if wall_clock_values else None,
        "learning_rate": learning_rate,
        "scheduler": scheduler,
        "token_budget": token_budget,
        "derived_max_steps": summary.get("derived_max_steps"),
        **evidence,
    }
    result["converged"] = bool(stable and evidence["converged"])
    return result


def select_recipe(
    rows: Iterable[Mapping[str, Any]],
    *,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not 0.0 <= tie_tolerance < 1.0:
        raise ConvergenceAnalysisError("tie_tolerance must be in [0, 1)")
    normalized = [dict(row) for row in rows]
    stable = sorted(
        (
            row
            for row in normalized
            if row.get("stable") and row.get("best_validation_loss") is not None
        ),
        key=lambda row: (float(row["best_validation_loss"]), str(row.get("run_id"))),
    )
    if stable and stable[0].get("converged"):
        best_loss = float(stable[0]["best_validation_loss"])
        tied = [
            row
            for row in stable
            if row.get("converged")
            and float(row["best_validation_loss"])
            <= best_loss * (1.0 + tie_tolerance)
        ]
        winner = min(
            tied,
            key=lambda row: (
                float("inf")
                if row.get("wall_clock_seconds") is None
                else float(row["wall_clock_seconds"]),
                float(row["best_validation_loss"]),
                str(row.get("run_id")),
            ),
        )
        return dict(winner), []

    return None, [dict(row) for row in stable[:2]]


def build_report(
    rows: list[dict[str, Any]],
    *,
    minimum_step: int,
    patience_evaluations: int,
    relative_improvement: float,
    tie_tolerance: float,
) -> dict[str, Any]:
    winner, fallback = select_recipe(rows, tie_tolerance=tie_tolerance)
    if winner is not None:
        status = "recipe_selected"
    elif fallback:
        status = "fallback_required"
    else:
        status = "no_stable_recipe"
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "selection_contract": {
            "minimum_step": minimum_step,
            "patience_evaluations": patience_evaluations,
            "relative_improvement": relative_improvement,
            "tie_tolerance": tie_tolerance,
            "selection_data": "ordinary_validation_only",
            "eligibility_rule": "global_best_stable_must_converge_v1",
        },
        "runs": sorted(rows, key=lambda row: str(row.get("run_id"))),
        "winner": winner,
        "fallback_candidates": fallback,
    }
    if winner is not None:
        report["frozen_recipe_overrides"] = [
            f"training.learning_rate={winner['learning_rate']}",
            f"training.scheduler.name={winner['scheduler']}",
            f"training.token_budget={winner['token_budget']}",
        ]
    report["report_hash"] = stable_hash(report)
    return report


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "run_id",
        "stable",
        "converged",
        "learning_rate",
        "scheduler",
        "token_budget",
        "derived_max_steps",
        "best_validation_loss",
        "best_validation_step",
        "final_validation_loss",
        "evaluations_since_significant_improvement",
        "wall_clock_seconds",
        "rejection_reasons",
        "run_dir",
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: str(item.get("run_id"))):
            output = {column: row.get(column) for column in columns}
            output["rejection_reasons"] = "; ".join(
                row.get("rejection_reasons", [])
            )
            writer.writerow(output)
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_dirs = discover_run_dirs(args.runs_root, args.run_dir)
    rows = [
        analyze_run(
            run_dir,
            minimum_step=args.minimum_step,
            patience_evaluations=args.patience_evaluations,
            relative_improvement=args.relative_improvement,
        )
        for run_dir in run_dirs
    ]
    report = build_report(
        rows,
        minimum_step=args.minimum_step,
        patience_evaluations=args.patience_evaluations,
        relative_improvement=args.relative_improvement,
        tie_tolerance=args.tie_tolerance,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_dir / "selection_report.json", report)
    _write_comparison_csv(output_dir / "run_comparison.csv", rows)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
