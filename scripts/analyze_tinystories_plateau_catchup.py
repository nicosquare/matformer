#!/usr/bin/env python3
"""Freeze TinyStories-Instruct plateaus, select elastic LR, and measure catch-up."""

# ruff: noqa: E402  # Add the repository root before importing src.

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.reproducibility import stable_hash
from src.training.schedules import (
    WSD_SCHEDULER_NAME,
    WSD_SCHEDULE_POLICY,
    WSD_SCHEDULE_POLICY_VERSION,
    wsd_learning_rate_factor,
)


PERPLEXITY_TOLERANCE = 0.005
LOSS_TOLERANCE = math.log1p(PERPLEXITY_TOLERANCE)
REQUIRED_SEEDS = (42, 43, 44)
ELASTIC_LRS = (0.004, 0.006, 0.008, 0.010)
ELASTIC_WIDTHS = ("g250", "g500", "g750", "g1000")
PLATEAU_WINDOW_FRACTION = 0.25
CATCHUP_STREAK = 5
MATCHED_HORIZON_EPOCHS = 3
STANDALONE_WSD_LRS = (0.002, 0.004, 0.006, 0.008)
WSD_DECAY_RATIO = 0.10
WSD_WARMUP_STEPS = 64


class PlateauCatchupError(ValueError):
    """Raised when experiment artifacts violate the frozen analysis contract."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlateauCatchupError(f"Cannot read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise PlateauCatchupError(f"JSON artifact must be a mapping: {path}")
    return value


def _read_metrics(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            return list(csv.DictReader(source))
    except (OSError, csv.Error, UnicodeError) as error:
        raise PlateauCatchupError(f"Cannot read metrics artifact: {path}") from error


def _discover_run_dirs(
    run_dirs: Iterable[str | Path],
    runs_roots: Iterable[str | Path] = (),
    *,
    seeds: Iterable[int] = (),
) -> list[Path]:
    discovered = {Path(path).expanduser().resolve() for path in run_dirs}
    for raw_root in runs_roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise PlateauCatchupError(f"Runs root does not exist: {root}")
        discovered.update(path.parent for path in root.rglob("run_summary.json"))
    if not discovered:
        raise PlateauCatchupError("No run directories were supplied or discovered")

    requested_seeds = {_integer(seed, "requested seed") for seed in seeds}
    if not requested_seeds:
        return sorted(discovered)

    matched: list[Path] = []
    for run_dir in sorted(discovered):
        summary_path = run_dir / "run_summary.json"
        config_path = run_dir / "config.json"
        if not summary_path.is_file():
            # Preserve the existing missing-artifact error for explicit bad paths.
            matched.append(run_dir)
            continue
        summary = _read_json(summary_path)
        config = _read_json(config_path) if config_path.is_file() else {}
        seed = _integer(
            summary.get("seed", config.get("run", {}).get("seed")),
            f"run seed in {run_dir}",
        )
        if seed in requested_seeds:
            matched.append(run_dir)
    if not matched:
        requested = ", ".join(str(seed) for seed in sorted(requested_seeds))
        raise PlateauCatchupError(
            f"No completed run directories matched requested seed(s): {requested}"
        )
    return matched


def _load_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    summary_path = root / "run_summary.json"
    config_path = root / "config.json"
    metrics_path = root / "metrics.csv"
    missing = [
        path.name
        for path in (summary_path, config_path, metrics_path)
        if not path.is_file()
    ]
    if missing:
        raise PlateauCatchupError(
            f"Run {root} is missing required artifacts: {', '.join(missing)}"
        )
    return {
        "run_dir": root,
        "summary": _read_json(summary_path),
        "config": _read_json(config_path),
        "metrics": _read_metrics(metrics_path),
    }


def _number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PlateauCatchupError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise PlateauCatchupError(f"{field} must be finite")
    return result


def _integer(value: Any, field: str) -> int:
    number = _number(value, field)
    result = int(number)
    if number != result:
        raise PlateauCatchupError(f"{field} must be an integer")
    return result


def _ordinary_validation_rows(
    run: Mapping[str, Any], granularity: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in run["metrics"]:
        if raw.get("split") != "validation" or raw.get("granularity") != granularity:
            continue
        if raw.get("loss") in (None, "") or raw.get("tokens_seen") in (None, ""):
            continue
        rows.append(
            {
                "step": _integer(raw.get("step"), "metrics.step"),
                "tokens_seen": _integer(raw.get("tokens_seen"), "metrics.tokens_seen"),
                "loss": _number(raw.get("loss"), "metrics.loss"),
            }
        )
    rows.sort(key=lambda row: (row["tokens_seen"], row["step"]))
    return rows


def _iteration_contract(run: Mapping[str, Any]) -> dict[str, Any]:
    config = run["config"]
    iteration = config.get("dataset", {}).get("optimizer_iteration")
    if not isinstance(iteration, Mapping):
        iteration = run["summary"].get("optimizer_iteration")
    if not isinstance(iteration, Mapping):
        raise PlateauCatchupError("Run lacks a resolved optimizer-iteration contract")
    if iteration.get("mode") != "repeat_epochs":
        raise PlateauCatchupError("Run does not use repeat_epochs")
    if iteration.get("epoch_order") != "deterministic_per_epoch":
        raise PlateauCatchupError("Run does not use deterministic_per_epoch")
    return dict(iteration)


def _seed(run: Mapping[str, Any]) -> int:
    return _integer(
        run["summary"].get("seed", run["config"].get("run", {}).get("seed")),
        "run seed",
    )


def _learning_rate(run: Mapping[str, Any]) -> float:
    summary = run["summary"]
    config = run["config"]
    return _number(
        summary.get(
            "resolved_learning_rate",
            config.get("training", {}).get(
                "resolved_learning_rate",
                config.get("training", {}).get("learning_rate"),
            ),
        ),
        "resolved learning rate",
    )


def _run_provenance(run: Mapping[str, Any]) -> dict[str, Any]:
    summary = run["summary"]
    config = run["config"]
    dataset = config.get("dataset", {})
    model = config.get("model", {})
    iteration = _iteration_contract(run)
    return {
        "dataset_name": dataset.get("dataset_name", summary.get("dataset_name")),
        "dataset_config_name": dataset.get("dataset_config_name"),
        "dataset_split": dataset.get("dataset_split", summary.get("dataset_split")),
        "dataset_phase": dataset.get("dataset_phase"),
        "corpus_hash": dataset.get("corpus_hash", summary.get("corpus_hash")),
        "optimizer_training_manifest_hash": iteration.get(
            "optimizer_training_manifest_hash",
            summary.get("optimizer_training_manifest_hash"),
        ),
        "tokenizer_manifest_hash": model.get(
            "tokenizer_manifest_hash", summary.get("tokenizer_manifest_hash")
        ),
        "d_model": model.get("d_model", model.get("hidden_size")),
        "num_layers": model.get("num_layers"),
        "context_length": model.get("context_length"),
        "aligned_epoch_samples": iteration.get("aligned_epoch_samples"),
        "aligned_epoch_tokens": iteration.get("aligned_epoch_tokens"),
        "excluded_tail_samples": iteration.get("excluded_tail_samples"),
        "permutation_version": iteration.get("permutation_version"),
        "permutation_hash": iteration.get("permutation_hash"),
        "fixed_epoch_set_hash": iteration.get("fixed_epoch_set_hash"),
        "ordering_policy_version": iteration.get("ordering_policy_version"),
    }


def _completed_rejections(run: Mapping[str, Any], expected_budget: int) -> list[str]:
    summary = run["summary"]
    rejections: list[str] = []
    if summary.get("status") != "completed":
        rejections.append("run status is not completed")
    tokens_seen = _integer(summary.get("tokens_seen", -1), "summary.tokens_seen")
    token_budget = _integer(summary.get("token_budget", -1), "summary.token_budget")
    if token_budget != expected_budget:
        rejections.append(f"token budget is {token_budget}, expected {expected_budget}")
    if tokens_seen < token_budget:
        rejections.append("run did not reach its token budget")
    if summary.get("unresolved_artifact_failures"):
        rejections.append("run reports unresolved artifact failures")
    return rejections


def _resolve_checkpoint(run_dir: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    raw = Path(str(value)).expanduser()
    candidates = (
        [raw]
        if raw.is_absolute()
        else [run_dir / raw, run_dir / "checkpoints" / raw.name]
    )
    return next(
        (candidate.resolve() for candidate in candidates if candidate.is_file()),
        None,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_plateau_run(run_dir: str | Path) -> dict[str, Any]:
    run = _load_run(run_dir)
    root = run["run_dir"]
    summary = run["summary"]
    config = run["config"]
    iteration = _iteration_contract(run)
    epoch_tokens = _integer(
        iteration.get("aligned_epoch_tokens"), "aligned_epoch_tokens"
    )
    analysis_horizon_tokens = MATCHED_HORIZON_EPOCHS * epoch_tokens
    rejections = _completed_rejections(run, analysis_horizon_tokens)
    if config.get("run", {}).get("model_family") != "standalone":
        rejections.append("run is not standalone")
    if config.get("model", {}).get("granularities") != ["g1000"]:
        rejections.append("run is not the standalone g1000 endpoint")
    if _integer(config.get("model", {}).get("d_model"), "model.d_model") != 64:
        rejections.append("run does not use d_model=64")
    if _integer(config.get("model", {}).get("num_layers"), "model.num_layers") != 4:
        rejections.append("run does not use num_layers=4")
    if not math.isclose(_learning_rate(run), 0.008, rel_tol=0.0, abs_tol=1e-12):
        rejections.append("run does not use learning rate 0.008")

    rows = _ordinary_validation_rows(run, "g1000")
    if not rows:
        rejections.append("no ordinary-validation g1000 rows")
    quarter_tokens = epoch_tokens // 4
    if quarter_tokens * 4 != epoch_tokens:
        rejections.append("aligned epoch cannot be divided into quarter windows")
    windows: list[dict[str, Any]] = []
    plateau_onset_tokens = None
    plateau_confirmation_tokens = None
    post_confirmation_best_improvement = None
    overall_best_loss = min((row["loss"] for row in rows), default=None)
    if rows and quarter_tokens * 4 == epoch_tokens:
        for start in range(epoch_tokens, analysis_horizon_tokens, quarter_tokens):
            end = start + quarter_tokens
            prior_losses = [row["loss"] for row in rows if row["tokens_seen"] <= start]
            through_losses = [row["loss"] for row in rows if row["tokens_seen"] <= end]
            in_window = [row for row in rows if start < row["tokens_seen"] <= end]
            if not prior_losses or not through_losses or not in_window:
                windows.append(
                    {
                        "start_tokens": start,
                        "end_tokens": end,
                        "validation_count": len(in_window),
                        "improvement": None,
                        "below_tolerance": False,
                    }
                )
                continue
            prior_best = min(prior_losses)
            cumulative_best = min(through_losses)
            improvement = max(prior_best - cumulative_best, 0.0)
            windows.append(
                {
                    "start_tokens": start,
                    "end_tokens": end,
                    "validation_count": len(in_window),
                    "prior_best_loss": prior_best,
                    "cumulative_best_loss": cumulative_best,
                    "improvement": improvement,
                    "below_tolerance": improvement < LOSS_TOLERANCE,
                }
            )
        for first, second in zip(windows, windows[1:]):
            if not first["below_tolerance"] or not second["below_tolerance"]:
                continue
            confirmation_best = second.get("cumulative_best_loss")
            if confirmation_best is None or overall_best_loss is None:
                continue
            later_improvement = max(confirmation_best - overall_best_loss, 0.0)
            if later_improvement >= LOSS_TOLERANCE:
                continue
            plateau_onset_tokens = first["start_tokens"]
            plateau_confirmation_tokens = second["end_tokens"]
            post_confirmation_best_improvement = later_improvement
            break
    if plateau_onset_tokens is None:
        rejections.append("no two consecutive qualifying quarter-epoch windows")

    best_row = min(rows, key=lambda row: row["loss"]) if rows else None
    frozen_best_loss = best_row["loss"] if best_row else None
    trailing_rows = rows[-5:]
    trailing_mean = (
        statistics.fmean(row["loss"] for row in trailing_rows)
        if len(trailing_rows) == 5
        else None
    )
    trailing_stable = bool(
        frozen_best_loss is not None
        and trailing_mean is not None
        and trailing_mean - frozen_best_loss <= LOSS_TOLERANCE
    )
    if len(trailing_rows) < 5:
        rejections.append("fewer than five trailing validation rows")
    elif not trailing_stable:
        rejections.append("trailing-five mean is outside the frozen-best tolerance")

    checkpoint = _resolve_checkpoint(root, summary.get("best_checkpoint_path"))
    if summary.get("checkpoint_status") != "best_eval":
        rejections.append("checkpoint status is not best_eval")
    if checkpoint is None:
        rejections.append("ordinary-validation best checkpoint is unavailable")
    checkpoint_step = (
        _integer(
            summary.get("checkpoint_selection_step")
            or (best_row["step"] if best_row else -1),
            "checkpoint selection step",
        )
        if best_row is not None
        else None
    )
    checkpoint_row = next((row for row in rows if row["step"] == checkpoint_step), None)
    if checkpoint_row is None:
        rejections.append("best checkpoint step has no ordinary-validation row")
    elif frozen_best_loss is not None and not math.isclose(
        checkpoint_row["loss"], frozen_best_loss, rel_tol=0.0, abs_tol=1e-12
    ):
        rejections.append("best checkpoint does not match the frozen best loss")
    checkpoint_tokens = (
        checkpoint_row["tokens_seen"]
        if checkpoint_row is not None
        else min(
            checkpoint_step
            * _integer(
                config.get("training", {}).get("expected_tokens_per_step"),
                "expected_tokens_per_step",
            ),
            analysis_horizon_tokens,
        )
        if checkpoint_step is not None
        else None
    )
    return {
        "run_id": summary.get("run_id", root.name),
        "run_dir": str(root),
        "seed": _seed(run),
        "contract_satisfied": not rejections,
        "rejection_reasons": rejections,
        "loss_tolerance": LOSS_TOLERANCE,
        "aligned_epoch_tokens": epoch_tokens,
        "analysis_horizon_epochs": MATCHED_HORIZON_EPOCHS,
        "analysis_horizon_tokens": analysis_horizon_tokens,
        "quarter_epoch_tokens": quarter_tokens,
        "windows": windows,
        "plateau_onset_tokens": plateau_onset_tokens,
        "plateau_confirmation_tokens": plateau_confirmation_tokens,
        "post_confirmation_best_improvement": (
            post_confirmation_best_improvement
        ),
        "frozen_best_loss": frozen_best_loss,
        "trailing_five_validation_mean": trailing_mean,
        "trailing_stable": trailing_stable,
        "checkpoint_step": checkpoint_step,
        "checkpoint_tokens": checkpoint_tokens,
        "checkpoint_path": str(checkpoint) if checkpoint is not None else None,
        "checkpoint_sha256": _sha256(checkpoint) if checkpoint is not None else None,
        "provenance": _run_provenance(run),
        "validation_rows": rows,
    }


def _manifest_hash(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("manifest_hash", None)
    return stable_hash(body)


def _validate_manifest(payload: Mapping[str, Any], name: str) -> None:
    if payload.get("manifest_hash") != _manifest_hash(payload):
        raise PlateauCatchupError(f"{name} manifest hash mismatch")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create a selection manifest once and reject later provenance drift."""

    if path.exists():
        existing = _read_json(path)
        if existing != dict(payload):
            raise PlateauCatchupError(
                f"Immutable selection manifest already exists with different "
                f"provenance: {path}"
            )
        return
    _write_json(path, payload)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def freeze_plateau_targets(
    run_dirs: Iterable[str | Path], output_dir: str | Path
) -> dict[str, Any]:
    analyses = [analyze_plateau_run(path) for path in run_dirs]
    by_seed: dict[int, dict[str, Any]] = {}
    for analysis in analyses:
        seed = int(analysis["seed"])
        if seed in by_seed:
            raise PlateauCatchupError(f"Duplicate standalone run for seed {seed}")
        by_seed[seed] = analysis
    missing = sorted(set(REQUIRED_SEEDS) - set(by_seed))
    extra = sorted(set(by_seed) - set(REQUIRED_SEEDS))
    provenance_hashes = {
        stable_hash(analysis["provenance"])
        for analysis in analyses
        if analysis["contract_satisfied"]
    }
    all_satisfied = (
        not missing
        and not extra
        and all(by_seed[seed]["contract_satisfied"] for seed in REQUIRED_SEEDS)
        and len(provenance_hashes) == 1
    )
    report = {
        "schema_version": 1,
        "analysis": "tinystories_instruct_standalone_plateau",
        "status": "targets_frozen" if all_satisfied else "plateau_not_robust",
        "perplexity_tolerance": PERPLEXITY_TOLERANCE,
        "loss_tolerance": LOSS_TOLERANCE,
        "window_fraction_of_epoch": PLATEAU_WINDOW_FRACTION,
        "matched_horizon_epochs": MATCHED_HORIZON_EPOCHS,
        "required_seeds": list(REQUIRED_SEEDS),
        "missing_seeds": missing,
        "unexpected_seeds": extra,
        "cross_seed_provenance_agreement": len(provenance_hashes) == 1,
        "runs": analyses,
    }
    report["report_hash"] = stable_hash(report)
    output = Path(output_dir).expanduser().resolve()
    _write_json(output / "plateau_report.json", report)
    _write_csv(
        output / "plateau_runs.csv",
        [
            {key: value for key, value in analysis.items() if key != "validation_rows"}
            for analysis in analyses
        ],
    )
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(8, 5))
    for analysis in analyses:
        rows = analysis["validation_rows"]
        axis.plot(
            [row["tokens_seen"] for row in rows],
            [row["loss"] for row in rows],
            label=f"seed {analysis['seed']}",
        )
        if analysis["plateau_onset_tokens"] is not None:
            axis.axvline(analysis["plateau_onset_tokens"], alpha=0.2)
    axis.set(xlabel="optimizer tokens", ylabel="ordinary-validation g1000 loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "plateau.png", dpi=160)
    pyplot.close(figure)

    frozen_path = output / "frozen_standalone_targets.json"
    if not all_satisfied:
        frozen_path.unlink(missing_ok=True)
        return report
    shared_provenance = by_seed[REQUIRED_SEEDS[0]]["provenance"]
    frozen = {
        "schema_version": 1,
        "analysis": "tinystories_instruct_frozen_standalone_targets",
        "perplexity_tolerance": PERPLEXITY_TOLERANCE,
        "loss_tolerance": LOSS_TOLERANCE,
        "required_seeds": list(REQUIRED_SEEDS),
        "shared_provenance": shared_provenance,
        "shared_provenance_hash": stable_hash(shared_provenance),
        "targets": {
            str(seed): {
                key: by_seed[seed][key]
                for key in (
                    "run_id",
                    "run_dir",
                    "seed",
                    "plateau_onset_tokens",
                    "plateau_confirmation_tokens",
                    "frozen_best_loss",
                    "checkpoint_step",
                    "checkpoint_tokens",
                    "checkpoint_path",
                    "checkpoint_sha256",
                    "provenance",
                )
            }
            for seed in REQUIRED_SEEDS
        },
        "plateau_report_hash": report["report_hash"],
    }
    frozen["manifest_hash"] = stable_hash(frozen)
    _write_json(frozen_path, frozen)
    return report


def _validate_elastic_run(
    run: Mapping[str, Any], frozen: Mapping[str, Any], *, expected_epochs: int
) -> tuple[list[str], dict[str, float]]:
    config = run["config"]
    iteration = _iteration_contract(run)
    epoch_tokens = _integer(iteration["aligned_epoch_tokens"], "aligned_epoch_tokens")
    rejections = _completed_rejections(run, expected_epochs * epoch_tokens)
    if _seed(run) not in REQUIRED_SEEDS:
        rejections.append("seed is outside the frozen contract")
    if config.get("run", {}).get("model_family") != "nested":
        rejections.append("run is not a nested elastic run")
    if tuple(config.get("model", {}).get("granularities", [])) != ELASTIC_WIDTHS:
        rejections.append("run does not use the locked four-width grid")
    if config.get("model", {}).get("granularity_sampling_mode") != "global":
        rejections.append("run does not use global width sampling")
    if (
        _integer(
            config.get("model", {}).get("global_sampling_interval_steps", -1),
            "global_sampling_interval_steps",
        )
        != 1
    ):
        rejections.append("run does not use H=1")
    provenance = _run_provenance(run)
    if provenance != frozen.get("shared_provenance"):
        rejections.append("run provenance does not match frozen standalone targets")
    diagnostics: dict[str, float] = {}
    for width in ELASTIC_WIDTHS:
        rows = _ordinary_validation_rows(run, width)
        if not rows:
            rejections.append(f"no ordinary-validation {width} rows")
        else:
            diagnostics[width] = min(row["loss"] for row in rows)
    return rejections, diagnostics


def select_elastic_lr(
    run_dirs: Iterable[str | Path],
    frozen_targets_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    frozen_path = Path(frozen_targets_path).expanduser().resolve()
    frozen = _read_json(frozen_path)
    _validate_manifest(frozen, "frozen standalone targets")
    runs = [_load_run(path) for path in run_dirs]
    candidates: list[dict[str, Any]] = []
    by_lr: dict[float, dict[str, Any]] = {}
    for run in runs:
        lr = _learning_rate(run)
        rejections, diagnostics = _validate_elastic_run(run, frozen, expected_epochs=1)
        if _seed(run) != 42:
            rejections.append("LR screen must use seed 42")
        candidate = {
            "run_id": run["summary"].get("run_id", run["run_dir"].name),
            "run_dir": str(run["run_dir"]),
            "seed": _seed(run),
            "learning_rate": lr,
            "stable": not rejections,
            "rejection_reasons": rejections,
            "best_losses_by_width": diagnostics,
            "g1000_best_loss": diagnostics.get("g1000"),
            "provenance": _run_provenance(run),
        }
        candidates.append(candidate)
        if lr in by_lr:
            raise PlateauCatchupError(f"Duplicate LR-screen run for learning rate {lr}")
        by_lr[lr] = candidate
    expected_lrs = set(ELASTIC_LRS)
    if set(by_lr) != expected_lrs:
        raise PlateauCatchupError(
            f"LR screen must contain exactly {sorted(expected_lrs)}; found {sorted(by_lr)}"
        )
    unstable = [candidate for candidate in candidates if not candidate["stable"]]
    if unstable:
        raise PlateauCatchupError(
            "LR screen contains invalid runs: "
            + "; ".join(
                f"{candidate['run_id']}: {candidate['rejection_reasons']}"
                for candidate in unstable
            )
        )
    winner = min(
        candidates, key=lambda row: (row["g1000_best_loss"], row["learning_rate"])
    )
    selection = {
        "schema_version": 1,
        "analysis": "tinystories_instruct_elastic_h1_lr_selection",
        "status": "lr_selected",
        "selection_endpoint": "ordinary_validation_g1000_best_loss",
        "candidate_learning_rates": list(ELASTIC_LRS),
        "selected_learning_rate": winner["learning_rate"],
        "selected_run_id": winner["run_id"],
        "selected_run_dir": winner["run_dir"],
        "selected_g1000_best_loss": winner["g1000_best_loss"],
        "diagnostic_best_losses_by_width": winner["best_losses_by_width"],
        "frozen_standalone_targets_path": str(frozen_path),
        "frozen_standalone_targets_hash": frozen["manifest_hash"],
        "shared_provenance": frozen["shared_provenance"],
        "candidates": candidates,
    }
    selection["manifest_hash"] = stable_hash(selection)
    output = Path(output_dir).expanduser().resolve()
    _write_json(output / "elastic_lr_selection.json", selection)
    _write_csv(output / "elastic_lr_candidates.csv", candidates)
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(7, 4))
    ordered = sorted(candidates, key=lambda row: row["learning_rate"])
    axis.plot(
        [row["learning_rate"] for row in ordered],
        [row["g1000_best_loss"] for row in ordered],
        marker="o",
    )
    axis.set(xlabel="learning rate", ylabel="best ordinary-validation g1000 loss")
    figure.tight_layout()
    figure.savefig(output / "elastic_lr_selection.png", dpi=160)
    pyplot.close(figure)
    return selection


def _resolved_wsd_contract(run: Mapping[str, Any]) -> dict[str, Any]:
    training = run["config"].get("training", {})
    scheduler = training.get("scheduler", {})
    contract = training.get("scheduler_contract")
    if not isinstance(contract, Mapping) and isinstance(scheduler, Mapping):
        contract = scheduler.get("contract")
    if not isinstance(contract, Mapping):
        raise PlateauCatchupError("Run lacks a resolved WSD scheduler contract")
    return dict(contract)


def _validate_ratio_wsd_contract(
    run: Mapping[str, Any],
    *,
    expected_epochs: int,
) -> tuple[list[str], dict[str, Any]]:
    """Validate one standalone seed-42 ratio-WSD calibration run."""

    config = run["config"]
    training = config.get("training", {})
    iteration = _iteration_contract(run)
    epoch_tokens = _integer(iteration["aligned_epoch_tokens"], "aligned_epoch_tokens")
    expected_budget = expected_epochs * epoch_tokens
    rejections = _completed_rejections(run, expected_budget)
    if _seed(run) != 42:
        rejections.append("WSD calibration requires seed 42")
    if config.get("run", {}).get("model_family") != "standalone":
        rejections.append("run is not standalone")
    if config.get("model", {}).get("granularities") != ["g1000"]:
        rejections.append("run is not the standalone g1000 endpoint")
    if _integer(config.get("model", {}).get("d_model"), "model.d_model") != 64:
        rejections.append("run does not use d_model=64")
    if _integer(config.get("model", {}).get("num_layers"), "model.num_layers") != 4:
        rejections.append("run does not use num_layers=4")

    expected_tokens_per_step = _integer(
        training.get("expected_tokens_per_step"), "expected_tokens_per_step"
    )
    expected_max_steps = math.ceil(expected_budget / expected_tokens_per_step)
    configured_max_steps = _integer(training.get("max_steps"), "training.max_steps")
    if configured_max_steps != expected_max_steps:
        rejections.append(
            f"resolved max_steps is {configured_max_steps}, expected {expected_max_steps}"
        )
    try:
        contract = _resolved_wsd_contract(run)
    except PlateauCatchupError as error:
        rejections.append(str(error))
        return rejections, {}

    expected_decay_steps = math.ceil(expected_max_steps * WSD_DECAY_RATIO)
    expected_stable_steps = (
        expected_max_steps - WSD_WARMUP_STEPS - expected_decay_steps
    )
    expected_cooldown_start = WSD_WARMUP_STEPS + expected_stable_steps
    expected_contract = {
        "name": WSD_SCHEDULER_NAME,
        "policy": WSD_SCHEDULE_POLICY,
        "policy_version": WSD_SCHEDULE_POLICY_VERSION,
        "max_steps": expected_max_steps,
        "warmup_steps": WSD_WARMUP_STEPS,
        "stable_steps": expected_stable_steps,
        "decay_steps": expected_decay_steps,
        "decay_ratio": WSD_DECAY_RATIO,
        "cooldown_start_step": expected_cooldown_start,
        "cooldown_start_tokens": min(
            expected_cooldown_start * expected_tokens_per_step,
            expected_budget,
        ),
        "schedule_end_tokens": expected_budget,
        "warmup_type": "linear",
        "decay_type": "cosine",
        "min_lr_ratio": 0.0,
        "min_learning_rate": 0.0,
        "num_cycles": 0.5,
    }
    for key, expected in expected_contract.items():
        actual = contract.get(key)
        if isinstance(expected, float):
            try:
                matches = math.isclose(
                    float(actual), expected, rel_tol=0.0, abs_tol=1e-12
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            rejections.append(
                f"WSD contract {key} is {actual!r}, expected {expected!r}"
            )
    return rejections, contract


def _checkpoint_for_metric_row(
    run: Mapping[str, Any], row: Mapping[str, Any]
) -> tuple[Path, str]:
    summary = run["summary"]
    checkpoint = _resolve_checkpoint(
        Path(run["run_dir"]), summary.get("best_checkpoint_path")
    )
    if summary.get("checkpoint_status") != "best_eval" or checkpoint is None:
        raise PlateauCatchupError(
            "Selected ordinary-validation loss has no available best checkpoint"
        )
    checkpoint_step = _integer(
        summary.get("checkpoint_selection_step"), "checkpoint selection step"
    )
    if checkpoint_step != _integer(row.get("step"), "selected metric step"):
        raise PlateauCatchupError(
            "Selected ordinary-validation loss is not the available best checkpoint"
        )
    return checkpoint, _sha256(checkpoint)


def select_standalone_wsd_lr(
    run_dirs: Iterable[str | Path], output_dir: str | Path
) -> dict[str, Any]:
    """Freeze the seed-42 standalone LR selected only from cooldown validation."""

    runs = [_load_run(path) for path in run_dirs]
    candidates: list[dict[str, Any]] = []
    by_lr: dict[float, dict[str, Any]] = {}
    for run in runs:
        learning_rate = _learning_rate(run)
        rejections, contract = _validate_ratio_wsd_contract(
            run, expected_epochs=1
        )
        cooldown_start = contract.get("cooldown_start_step")
        rows = _ordinary_validation_rows(run, "g1000")
        cooldown_rows = (
            [row for row in rows if row["step"] >= int(cooldown_start)]
            if cooldown_start is not None
            else []
        )
        if not cooldown_rows:
            rejections.append("no ordinary-validation g1000 rows after cooldown begins")
            cooldown_best = None
        else:
            cooldown_best = min(
                cooldown_rows, key=lambda row: (row["loss"], row["step"])
            )
        candidate = {
            "run_id": run["summary"].get("run_id", run["run_dir"].name),
            "run_dir": str(run["run_dir"]),
            "seed": _seed(run),
            "learning_rate": learning_rate,
            "contract_satisfied": not rejections,
            "rejection_reasons": rejections,
            "cooldown_start_step": cooldown_start,
            "cooldown_validation_count": len(cooldown_rows),
            "cooldown_best_step": (
                cooldown_best["step"] if cooldown_best is not None else None
            ),
            "cooldown_best_tokens": (
                cooldown_best["tokens_seen"] if cooldown_best is not None else None
            ),
            "cooldown_best_loss": (
                cooldown_best["loss"] if cooldown_best is not None else None
            ),
            "wsd_contract": contract,
            "provenance": _run_provenance(run),
        }
        candidates.append(candidate)
        matching_lr = next(
            (
                expected
                for expected in STANDALONE_WSD_LRS
                if math.isclose(
                    learning_rate, expected, rel_tol=0.0, abs_tol=1e-12
                )
            ),
            learning_rate,
        )
        if matching_lr in by_lr:
            raise PlateauCatchupError(
                f"Duplicate WSD LR-screen run for learning rate {matching_lr}"
            )
        by_lr[matching_lr] = candidate

    if set(by_lr) != set(STANDALONE_WSD_LRS):
        raise PlateauCatchupError(
            "WSD LR screen must contain exactly "
            f"{list(STANDALONE_WSD_LRS)}; found {sorted(by_lr)}"
        )
    invalid = [row for row in candidates if not row["contract_satisfied"]]
    if invalid:
        raise PlateauCatchupError(
            "WSD LR screen contains invalid runs: "
            + "; ".join(
                f"{row['run_id']}: {row['rejection_reasons']}" for row in invalid
            )
        )
    provenance_hashes = {stable_hash(row["provenance"]) for row in candidates}
    if len(provenance_hashes) != 1:
        raise PlateauCatchupError(
            "WSD LR-screen data/model provenance does not match across candidates"
        )

    winner = min(
        candidates,
        key=lambda row: (row["cooldown_best_loss"], row["learning_rate"]),
    )
    winner_run = next(
        run for run in runs if str(run["run_dir"]) == winner["run_dir"]
    )
    winner_row = {
        "step": winner["cooldown_best_step"],
        "loss": winner["cooldown_best_loss"],
    }
    checkpoint, checkpoint_sha256 = _checkpoint_for_metric_row(
        winner_run, winner_row
    )
    selection = {
        "schema_version": 1,
        "analysis": "tinystories_instruct_standalone_wsd_lr_selection",
        "status": "lr_selected",
        "selection_endpoint": "cooldown_ordinary_validation_g1000_best_loss",
        "required_seed": 42,
        "candidate_learning_rates": list(STANDALONE_WSD_LRS),
        "selected_learning_rate": winner["learning_rate"],
        "selected_run_id": winner["run_id"],
        "selected_run_dir": winner["run_dir"],
        "selected_cooldown_best_step": winner["cooldown_best_step"],
        "selected_cooldown_best_tokens": winner["cooldown_best_tokens"],
        "selected_cooldown_best_loss": winner["cooldown_best_loss"],
        "selected_checkpoint_path": str(checkpoint),
        "selected_checkpoint_sha256": checkpoint_sha256,
        "shared_provenance": winner["provenance"],
        "shared_provenance_hash": next(iter(provenance_hashes)),
        "wsd_contract": winner["wsd_contract"],
        "candidates": candidates,
    }
    selection["manifest_hash"] = stable_hash(selection)
    output = Path(output_dir).expanduser().resolve()
    _write_immutable_json(output / "standalone_wsd_lr_selection.json", selection)
    _write_csv(output / "standalone_wsd_lr_candidates.csv", candidates)
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(7, 4))
    ordered = sorted(candidates, key=lambda row: row["learning_rate"])
    axis.plot(
        [row["learning_rate"] for row in ordered],
        [row["cooldown_best_loss"] for row in ordered],
        marker="o",
    )
    axis.scatter(
        [winner["learning_rate"]],
        [winner["cooldown_best_loss"]],
        color="tab:red",
        zorder=3,
        label="selected",
    )
    axis.set(
        xlabel="stable learning rate",
        ylabel="best cooldown ordinary-validation g1000 loss",
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "standalone_wsd_lr_selection.png", dpi=160)
    pyplot.close(figure)
    return selection


def _phase_for_scheduler_position(
    position: int, contract: Mapping[str, Any]
) -> tuple[str, int, float]:
    warmup_steps = int(contract["warmup_steps"])
    stable_steps = int(contract["stable_steps"])
    decay_steps = int(contract["decay_steps"])
    cooldown_start = int(contract["cooldown_start_step"])
    if position < warmup_steps:
        return "warmup", position, position / max(warmup_steps, 1)
    if position < cooldown_start:
        phase_step = position - warmup_steps
        return "stable", phase_step, phase_step / max(stable_steps, 1)
    phase_step = min(max(position - cooldown_start, 0), decay_steps)
    return "cooldown", phase_step, min(phase_step / max(decay_steps, 1), 1.0)


def analyze_wsd_calibration(
    run_dir: str | Path,
    selection_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Analyze one fresh three-epoch winner without freezing any target."""

    output = Path(output_dir).expanduser().resolve()
    frozen_target_path = output / "frozen_standalone_targets.json"
    if frozen_target_path.exists():
        raise PlateauCatchupError(
            "wsd-calibration refuses an output directory containing "
            "frozen_standalone_targets.json"
        )
    selection_file = Path(selection_path).expanduser().resolve()
    selection = _read_json(selection_file)
    _validate_manifest(selection, "standalone WSD LR selection")
    if selection.get("analysis") != "tinystories_instruct_standalone_wsd_lr_selection":
        raise PlateauCatchupError("Selection is not a standalone WSD LR manifest")

    run = _load_run(run_dir)
    rejections, contract = _validate_ratio_wsd_contract(
        run, expected_epochs=MATCHED_HORIZON_EPOCHS
    )
    selected_lr = _number(selection.get("selected_learning_rate"), "selected LR")
    if not math.isclose(
        _learning_rate(run), selected_lr, rel_tol=0.0, abs_tol=1e-12
    ):
        rejections.append("calibration learning rate does not match the selection")
    if _run_provenance(run) != selection.get("shared_provenance"):
        rejections.append("calibration data/model provenance does not match selection")
    controlled_experiment = run["config"].get("controlled_experiment", {})
    configured_selection_hash = (
        controlled_experiment.get("selection_report_hash")
        if isinstance(controlled_experiment, Mapping)
        else None
    )
    if configured_selection_hash != selection.get("manifest_hash"):
        rejections.append(
            "calibration config does not link the immutable WSD selection hash"
        )
    if run["summary"].get("run_id") == selection.get("selected_run_id"):
        rejections.append("calibration run must be distinct from the LR-screen winner")
    if rejections:
        raise PlateauCatchupError(f"Invalid WSD calibration run: {rejections}")

    peak_lr = _learning_rate(run)
    rows = _ordinary_validation_rows(run, "g1000")
    if not rows:
        raise PlateauCatchupError("WSD calibration has no validation rows")
    validation_rows: list[dict[str, Any]] = []
    by_phase: dict[str, list[dict[str, Any]]] = {
        "warmup": [],
        "stable": [],
        "cooldown": [],
    }
    for row in rows:
        position = int(row["step"])
        phase, phase_step, phase_progress = _phase_for_scheduler_position(
            position, contract
        )
        factor = wsd_learning_rate_factor(
            position,
            warmup_steps=int(contract["warmup_steps"]),
            stable_steps=int(contract["stable_steps"]),
            decay_steps=int(contract["decay_steps"]),
            warmup_type=str(contract["warmup_type"]),
            decay_type=str(contract["decay_type"]),
            min_lr_ratio=float(contract["min_lr_ratio"]),
            num_cycles=float(contract["num_cycles"]),
        )
        decorated = {
            **row,
            "scheduler_position": position,
            "scheduler_phase": phase,
            "scheduler_phase_step": phase_step,
            "scheduler_phase_progress": phase_progress,
            "learning_rate": peak_lr * factor,
        }
        validation_rows.append(decorated)
        by_phase[phase].append(decorated)

    stable_rows = by_phase["stable"]
    cooldown_rows = by_phase["cooldown"]
    if not stable_rows or not cooldown_rows:
        raise PlateauCatchupError(
            "WSD calibration requires both stable and cooldown validation rows"
        )
    stable_best = min(stable_rows, key=lambda row: (row["loss"], row["step"]))
    cooldown_best = min(
        cooldown_rows, key=lambda row: (row["loss"], row["step"])
    )
    cooldown_final = max(cooldown_rows, key=lambda row: row["step"])

    iteration = _iteration_contract(run)
    epoch_tokens = _integer(iteration["aligned_epoch_tokens"], "aligned_epoch_tokens")
    quarter_tokens = epoch_tokens // 4
    if quarter_tokens * 4 != epoch_tokens:
        raise PlateauCatchupError("aligned epoch cannot be divided into quarters")
    horizon_tokens = MATCHED_HORIZON_EPOCHS * epoch_tokens
    stable_windows: list[dict[str, Any]] = []
    for start in range(0, horizon_tokens, quarter_tokens):
        end = start + quarter_tokens
        window_rows = [
            row for row in stable_rows if start < row["tokens_seen"] <= end
        ]
        prior_rows = [row for row in stable_rows if row["tokens_seen"] <= start]
        if not window_rows or not prior_rows:
            improvement = None
            prior_best = None
            through_best = None
        else:
            prior_best = min(row["loss"] for row in prior_rows)
            through_best = min(
                row["loss"]
                for row in stable_rows
                if row["tokens_seen"] <= end
            )
            improvement = max(prior_best - through_best, 0.0)
        stable_windows.append(
            {
                "start_tokens": start,
                "end_tokens": end,
                "validation_count": len(window_rows),
                "prior_best_loss": prior_best,
                "cumulative_best_loss": through_best,
                "improvement": improvement,
                "below_diagnostic_threshold": (
                    improvement is not None and improvement < LOSS_TOLERANCE
                ),
            }
        )

    checkpoint = _resolve_checkpoint(
        Path(run["run_dir"]), run["summary"].get("best_checkpoint_path")
    )
    if checkpoint is None:
        raise PlateauCatchupError("WSD calibration best checkpoint is unavailable")

    max_steps = int(contract["max_steps"])
    trajectory_positions = {
        round(index * max_steps / 1000) for index in range(1001)
    }
    trajectory_positions.update(
        {
            0,
            int(contract["warmup_steps"]),
            int(contract["cooldown_start_step"]),
            max_steps,
        }
    )
    lr_trajectory = []
    for position in sorted(trajectory_positions):
        phase, phase_step, phase_progress = _phase_for_scheduler_position(
            position, contract
        )
        factor = wsd_learning_rate_factor(
            position,
            warmup_steps=int(contract["warmup_steps"]),
            stable_steps=int(contract["stable_steps"]),
            decay_steps=int(contract["decay_steps"]),
            warmup_type=str(contract["warmup_type"]),
            decay_type=str(contract["decay_type"]),
            min_lr_ratio=float(contract["min_lr_ratio"]),
            num_cycles=float(contract["num_cycles"]),
        )
        lr_trajectory.append(
            {
                "scheduler_position": position,
                "tokens": min(
                    position
                    * _integer(
                        run["config"]["training"]["expected_tokens_per_step"],
                        "expected_tokens_per_step",
                    ),
                    horizon_tokens,
                ),
                "phase": phase,
                "phase_step": phase_step,
                "phase_progress": phase_progress,
                "learning_rate": peak_lr * factor,
            }
        )

    report = {
        "schema_version": 1,
        "analysis": "tinystories_instruct_wsd_calibration",
        "status": "calibration_reported",
        "seed": 42,
        "run_id": run["summary"].get("run_id", Path(run_dir).name),
        "run_dir": str(run["run_dir"]),
        "selected_learning_rate": selected_lr,
        "selection_manifest_path": str(selection_file),
        "selection_manifest_hash": selection["manifest_hash"],
        "wsd_contract": contract,
        "phase_boundaries": {
            "warmup_end_step": int(contract["warmup_steps"]),
            "cooldown_start_step": int(contract["cooldown_start_step"]),
            "cooldown_start_tokens": int(contract["cooldown_start_tokens"]),
            "schedule_end_step": int(contract["max_steps"]),
            "schedule_end_tokens": horizon_tokens,
        },
        "validation_counts_by_phase": {
            phase: len(phase_rows) for phase, phase_rows in by_phase.items()
        },
        "quarter_epoch_stable_improvements": stable_windows,
        "diagnostic_perplexity_threshold": PERPLEXITY_TOLERANCE,
        "diagnostic_loss_threshold": LOSS_TOLERANCE,
        "stable_best": stable_best,
        "cooldown_best": cooldown_best,
        "cooldown_final": cooldown_final,
        "cooldown_gain": stable_best["loss"] - cooldown_best["loss"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "lr_trajectory": lr_trajectory,
        "validation_rows_by_phase": by_phase,
        "targets_frozen": False,
        "elastic_claim": False,
    }
    report["report_hash"] = stable_hash(report)
    _write_json(output / "wsd_calibration_report.json", report)
    _write_csv(output / "wsd_calibration_validation.csv", validation_rows)

    pyplot = _pyplot()
    figure, loss_axis = pyplot.subplots(figsize=(9, 5))
    lr_axis = loss_axis.twinx()
    for phase, color in (
        ("warmup", "tab:orange"),
        ("stable", "tab:blue"),
        ("cooldown", "tab:green"),
    ):
        phase_rows = by_phase[phase]
        if phase_rows:
            loss_axis.plot(
                [row["tokens_seen"] for row in phase_rows],
                [row["loss"] for row in phase_rows],
                marker="o",
                markersize=2.5,
                linewidth=1.2,
                color=color,
                label=f"{phase} validation",
            )
    lr_axis.plot(
        [row["tokens"] for row in lr_trajectory],
        [row["learning_rate"] for row in lr_trajectory],
        color="black",
        alpha=0.55,
        linewidth=1.1,
        label="learning rate",
    )
    loss_axis.axvline(
        int(contract["cooldown_start_tokens"]),
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label="cooldown begins",
    )
    loss_axis.set(xlabel="optimizer tokens", ylabel="ordinary-validation g1000 loss")
    lr_axis.set_ylabel("learning rate")
    handles, labels = loss_axis.get_legend_handles_labels()
    lr_handles, lr_labels = lr_axis.get_legend_handles_labels()
    loss_axis.legend(handles + lr_handles, labels + lr_labels, frameon=False)
    figure.tight_layout()
    figure.savefig(output / "wsd_calibration.png", dpi=160)
    pyplot.close(figure)
    if frozen_target_path.exists():
        raise AssertionError("wsd-calibration must never freeze standalone targets")
    return report


def measure_catchup(
    run_dirs: Iterable[str | Path],
    frozen_targets_path: str | Path,
    elastic_selection_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    frozen_path = Path(frozen_targets_path).expanduser().resolve()
    selection_path = Path(elastic_selection_path).expanduser().resolve()
    frozen = _read_json(frozen_path)
    selection = _read_json(selection_path)
    _validate_manifest(frozen, "frozen standalone targets")
    _validate_manifest(selection, "elastic LR selection")
    if selection.get("frozen_standalone_targets_hash") != frozen["manifest_hash"]:
        raise PlateauCatchupError(
            "Elastic LR selection links a different target manifest"
        )
    selected_lr = _number(selection.get("selected_learning_rate"), "selected LR")
    runs = [_load_run(path) for path in run_dirs]
    by_seed: dict[int, Mapping[str, Any]] = {}
    for run in runs:
        seed = _seed(run)
        if seed in by_seed:
            raise PlateauCatchupError(f"Duplicate catch-up run for seed {seed}")
        by_seed[seed] = run
    if set(by_seed) != set(REQUIRED_SEEDS):
        raise PlateauCatchupError(
            f"Catch-up runs must contain seeds {list(REQUIRED_SEEDS)}"
        )

    results: list[dict[str, Any]] = []
    plot_rows: dict[int, list[dict[str, Any]]] = {}
    for seed in REQUIRED_SEEDS:
        run = by_seed[seed]
        rejections, best_by_width = _validate_elastic_run(
            run, frozen, expected_epochs=MATCHED_HORIZON_EPOCHS
        )
        if not math.isclose(
            _learning_rate(run), selected_lr, rel_tol=0.0, abs_tol=1e-12
        ):
            rejections.append(
                "run learning rate does not match the frozen LR selection"
            )
        if rejections:
            raise PlateauCatchupError(
                f"Invalid catch-up run for seed {seed}: {rejections}"
            )
        target = frozen["targets"][str(seed)]
        frozen_loss = _number(target["frozen_best_loss"], "frozen best loss")
        rows = _ordinary_validation_rows(run, "g1000")
        plot_rows[seed] = rows
        qualifying = [row["loss"] - frozen_loss <= LOSS_TOLERANCE for row in rows]
        catchup_index = next(
            (
                index
                for index in range(0, len(rows) - CATCHUP_STREAK + 1)
                if all(qualifying[index : index + CATCHUP_STREAK])
            ),
            None,
        )
        catchup_row = rows[catchup_index] if catchup_index is not None else None
        checkpoint_tokens = _integer(
            target["checkpoint_tokens"], "standalone checkpoint tokens"
        )
        confirmation_tokens = _integer(
            target["plateau_confirmation_tokens"], "plateau confirmation tokens"
        )
        catchup_tokens = catchup_row["tokens_seen"] if catchup_row else None
        signed_delta = catchup_tokens - checkpoint_tokens if catchup_row else None
        additional = max(signed_delta, 0) if signed_delta is not None else None
        diagnostic_at_catchup: dict[str, float | None] = {}
        if catchup_row is not None:
            for width in ELASTIC_WIDTHS[:-1]:
                width_rows = _ordinary_validation_rows(run, width)
                matched = next(
                    (row for row in width_rows if row["step"] == catchup_row["step"]),
                    None,
                )
                diagnostic_at_catchup[width] = matched["loss"] if matched else None
        results.append(
            {
                "seed": seed,
                "run_id": run["summary"].get("run_id", run["run_dir"].name),
                "run_dir": str(run["run_dir"]),
                "frozen_standalone_loss": frozen_loss,
                "qualifying_streak_length": CATCHUP_STREAK,
                "caught_up": catchup_row is not None,
                "censored": catchup_row is None,
                "catchup_step": catchup_row["step"] if catchup_row else None,
                "catchup_tokens": catchup_tokens,
                "signed_token_delta": signed_delta,
                "additional_token_budget": additional,
                "catchup_to_standalone_checkpoint_token_ratio": (
                    catchup_tokens / checkpoint_tokens if catchup_row else None
                ),
                "catchup_to_plateau_confirmation_token_ratio": (
                    catchup_tokens / confirmation_tokens if catchup_row else None
                ),
                "standalone_checkpoint_tokens": checkpoint_tokens,
                "plateau_confirmation_tokens": confirmation_tokens,
                "diagnostic_losses_at_catchup": diagnostic_at_catchup,
                "diagnostic_best_losses_by_width": best_by_width,
            }
        )

    all_caught_up = all(result["caught_up"] for result in results)
    additional_values = [
        int(result["additional_token_budget"])
        for result in results
        if result["additional_token_budget"] is not None
    ]
    aggregates = None
    if all_caught_up:
        aggregates = {
            "mean_additional_token_budget": statistics.fmean(additional_values),
            "median_additional_token_budget": statistics.median(additional_values),
            "minimum_additional_token_budget": min(additional_values),
            "maximum_additional_token_budget": max(additional_values),
        }
    report = {
        "schema_version": 1,
        "analysis": "tinystories_instruct_elastic_h1_catchup",
        "status": "cross_seed_catchup" if all_caught_up else "censored",
        "general_cross_seed_catchup_claim": all_caught_up,
        "perplexity_tolerance": PERPLEXITY_TOLERANCE,
        "loss_tolerance": LOSS_TOLERANCE,
        "required_consecutive_evaluations": CATCHUP_STREAK,
        "maximum_horizon_epochs": MATCHED_HORIZON_EPOCHS,
        "selected_learning_rate": selected_lr,
        "frozen_standalone_targets_hash": frozen["manifest_hash"],
        "elastic_lr_selection_hash": selection["manifest_hash"],
        "seeds": results,
        "additional_budget_summary": aggregates,
    }
    report["report_hash"] = stable_hash(report)
    output = Path(output_dir).expanduser().resolve()
    _write_json(output / "catchup_report.json", report)
    _write_csv(output / "catchup_by_seed.csv", results)
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(8, 5))
    for result in results:
        seed = int(result["seed"])
        target = float(result["frozen_standalone_loss"])
        rows = plot_rows[seed]
        axis.plot(
            [row["tokens_seen"] for row in rows],
            [row["loss"] - target for row in rows],
            label=f"seed {seed}",
        )
    axis.axhline(LOSS_TOLERANCE, color="black", linestyle="--", label="0.5% PPL")
    axis.set(xlabel="optimizer tokens", ylabel="elastic loss - frozen standalone loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "catchup.png", dpi=160)
    pyplot.close(figure)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plateau = subparsers.add_parser(
        "plateau", help="confirm and freeze standalone targets"
    )
    plateau.add_argument("--run-dir", action="append", default=[])
    plateau.add_argument("--runs-root", action="append", default=[])
    plateau.add_argument(
        "--seed",
        action="append",
        type=int,
        default=[],
        help=(
            "analyze only this seed beneath --runs-root; may be repeated and is "
            "useful before all confirmation runs exist"
        ),
    )
    plateau.add_argument("--output-dir", required=True)

    select = subparsers.add_parser(
        "select-elastic-lr", help="freeze the seed-42 H=1 LR"
    )
    select.add_argument("--run-dir", action="append", default=[])
    select.add_argument("--runs-root", action="append", default=[])
    select.add_argument("--frozen-targets", required=True)
    select.add_argument("--output-dir", required=True)

    select_wsd = subparsers.add_parser(
        "select-standalone-wsd-lr",
        help="freeze the seed-42 standalone ratio-WSD LR",
    )
    select_wsd.add_argument("--run-dir", action="append", default=[])
    select_wsd.add_argument("--runs-root", action="append", default=[])
    select_wsd.add_argument("--output-dir", required=True)

    calibration = subparsers.add_parser(
        "wsd-calibration",
        help="analyze one fresh three-epoch standalone WSD winner",
    )
    calibration.add_argument("--run-dir", required=True)
    calibration.add_argument("--wsd-selection", required=True)
    calibration.add_argument("--output-dir", required=True)

    catchup = subparsers.add_parser("catchup", help="measure matched-seed H=1 catch-up")
    catchup.add_argument("--run-dir", action="append", default=[])
    catchup.add_argument("--runs-root", action="append", default=[])
    catchup.add_argument("--frozen-targets", required=True)
    catchup.add_argument("--elastic-selection", required=True)
    catchup.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "wsd-calibration":
        result = analyze_wsd_calibration(
            args.run_dir,
            args.wsd_selection,
            args.output_dir,
        )
        print(json.dumps({"status": result["status"]}, sort_keys=True))
        return

    run_dirs = _discover_run_dirs(
        args.run_dir,
        args.runs_root,
        seeds=getattr(args, "seed", ()),
    )
    if args.command == "plateau":
        result = freeze_plateau_targets(run_dirs, args.output_dir)
    elif args.command == "select-elastic-lr":
        result = select_elastic_lr(run_dirs, args.frozen_targets, args.output_dir)
    elif args.command == "select-standalone-wsd-lr":
        result = select_standalone_wsd_lr(run_dirs, args.output_dir)
    elif args.command == "catchup":
        result = measure_catchup(
            run_dirs,
            args.frozen_targets,
            args.elastic_selection,
            args.output_dir,
        )
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
