#!/usr/bin/env python3
"""Freeze, verify, and holdout-check the fixed-recipe portfolio study."""

# ruff: noqa: E402  # Make the repository importable for direct script execution.

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

from src.training.portfolio_catchup import manifest_hash
from src.utils.reproducibility import stable_hash


REFERENCE_BUDGET_TOKENS = 713_785_344
ELASTIC_BUDGET_CAP_TOKENS = 2_141_356_032
AGGREGATE_REFERENCE_BUDGET_TOKENS = 2_855_141_376
REQUIRED_SEEDS = (42, 43, 44)
GRANULARITIES = ("g250", "g500", "g750", "g1000")
FIXED_LEARNING_RATE = 0.008
PERPLEXITY_TOLERANCE = 0.005
LOSS_TOLERANCE = math.log1p(PERPLEXITY_TOLERANCE)
REQUIRED_STREAK = 5
COMPARISON_GROUP_ID = "tinystories_instruct_portfolio_catchup_v1"


class PortfolioAnalysisError(ValueError):
    """Raised when saved artifacts violate the controlled experiment."""


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PortfolioAnalysisError(
            f"Cannot read JSON artifact: {resolved}"
        ) from error
    if not isinstance(value, dict):
        raise PortfolioAnalysisError(f"JSON artifact must be a mapping: {resolved}")
    return value


def _read_metrics(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            return list(csv.DictReader(source))
    except (OSError, csv.Error, UnicodeError) as error:
        raise PortfolioAnalysisError(f"Cannot read metrics artifact: {path}") from error


def _load_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    required = [root / "config.json", root / "run_summary.json", root / "metrics.csv"]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise PortfolioAnalysisError(
            f"Run {root} is missing required artifacts: {', '.join(missing)}"
        )
    return {
        "run_dir": root,
        "config": _read_json(required[0]),
        "summary": _read_json(required[1]),
        "metrics": _read_metrics(required[2]),
    }


def discover_run_dirs(
    run_dirs: Iterable[str | Path], runs_roots: Iterable[str | Path] = ()
) -> list[Path]:
    discovered = {Path(path).expanduser().resolve() for path in run_dirs}
    for raw_root in runs_roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise PortfolioAnalysisError(f"Runs root does not exist: {root}")
        discovered.update(path.parent for path in root.rglob("run_summary.json"))
    if not discovered:
        raise PortfolioAnalysisError("No run directories were supplied or discovered")
    return sorted(discovered)


def _number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PortfolioAnalysisError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise PortfolioAnalysisError(f"{field} must be finite")
    return result


def _integer(value: Any, field: str) -> int:
    number = _number(value, field)
    result = int(number)
    if number != result:
        raise PortfolioAnalysisError(f"{field} must be an integer")
    return result


def _seed(run: Mapping[str, Any]) -> int:
    return _integer(
        run["summary"].get("seed", run["config"].get("run", {}).get("seed")),
        "run seed",
    )


def _learning_rate(run: Mapping[str, Any]) -> float:
    training = run["config"].get("training", {})
    return _number(
        run["summary"].get(
            "resolved_learning_rate",
            training.get("resolved_learning_rate", training.get("learning_rate")),
        ),
        "resolved learning rate",
    )


def _controlled_contract(
    run: Mapping[str, Any],
) -> tuple[str | None, Mapping[str, Any]]:
    controlled = run["config"].get("controlled_experiment", {})
    if not isinstance(controlled, Mapping):
        return None, {}
    contract = controlled.get("portfolio_catchup", {})
    return (
        controlled.get("comparison_role"),
        contract if isinstance(contract, Mapping) else {},
    )


def _ordinary_rows(run: Mapping[str, Any], width: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_steps: set[int] = set()
    for raw in run["metrics"]:
        if raw.get("split") != "validation" or raw.get("granularity") != width:
            continue
        if raw.get("loss") in (None, ""):
            continue
        step = _integer(raw.get("step"), "metrics.step")
        if step in seen_steps:
            raise PortfolioAnalysisError(
                f"Duplicate ordinary validation for {width} at step {step}"
            )
        seen_steps.add(step)
        rows.append(
            {
                "step": step,
                "tokens_seen": _integer(raw.get("tokens_seen"), "metrics.tokens_seen"),
                "loss": _number(raw.get("loss"), "metrics.loss"),
                "perplexity": _number(
                    raw.get("perplexity", math.exp(_number(raw.get("loss"), "loss"))),
                    "metrics.perplexity",
                ),
            }
        )
    return sorted(rows, key=lambda row: (row["step"], row["tokens_seen"]))


def _joint_rows(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_width = {width: _ordinary_rows(run, width) for width in GRANULARITIES}
    if any(not rows for rows in by_width.values()):
        raise PortfolioAnalysisError(
            "Elastic run lacks validation for one or more widths"
        )
    step_sets = {
        width: {row["step"] for row in rows} for width, rows in by_width.items()
    }
    if len({tuple(sorted(steps)) for steps in step_sets.values()}) != 1:
        raise PortfolioAnalysisError(
            "Elastic ordinary validations are not simultaneous across all widths"
        )
    indexes = {
        width: {row["step"]: row for row in rows} for width, rows in by_width.items()
    }
    joint: list[dict[str, Any]] = []
    for step in sorted(next(iter(step_sets.values()))):
        rows = {width: indexes[width][step] for width in GRANULARITIES}
        token_values = {row["tokens_seen"] for row in rows.values()}
        if len(token_values) != 1:
            raise PortfolioAnalysisError(
                f"Elastic validation token boundary differs across widths at step {step}"
            )
        joint.append({"step": step, "tokens_seen": token_values.pop(), "widths": rows})
    return joint


def _resolve_checkpoint(run_dir: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    candidates = (
        [path]
        if path.is_absolute()
        else [run_dir / path, run_dir / "checkpoints" / path.name]
    )
    return next(
        (candidate.resolve() for candidate in candidates if candidate.is_file()), None
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run_provenance(run: Mapping[str, Any]) -> dict[str, Any]:
    config = run["config"]
    model = config.get("model", {})
    training = config.get("training", {})
    dataset = config.get("dataset", {})
    iteration = dataset.get("optimizer_iteration", {})
    distributed = training.get("distributed", {})
    return {
        "dataset_name": dataset.get("dataset_name"),
        "dataset_config_name": dataset.get("dataset_config_name"),
        "dataset_split": dataset.get("dataset_split"),
        "dataset_phase": dataset.get("dataset_phase"),
        "corpus_hash": dataset.get("corpus_hash", config.get("corpus_hash")),
        "optimizer_training_manifest_hash": iteration.get(
            "optimizer_training_manifest_hash",
            config.get("optimizer_training_manifest_hash"),
        ),
        "validation_manifest_hash": config.get("validation_manifest_hash"),
        "final_holdout_manifest_hash": config.get("final_holdout_manifest_hash"),
        "tokenizer_manifest_hash": model.get("tokenizer_manifest_hash"),
        "model_variant": model.get("variant"),
        "correction_mode": model.get("correction_mode"),
        "d_model": model.get("d_model", model.get("hidden_size")),
        "num_layers": model.get("num_layers"),
        "num_attention_heads": model.get("num_attention_heads"),
        "context_length": model.get("context_length"),
        "vocab_size": model.get("vocab_size"),
        "batch_size_per_process": training.get("batch_size_per_process"),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
        "expected_tokens_per_step": training.get("expected_tokens_per_step"),
        "optimizer": training.get("optimizer"),
        "warmup_steps": training.get("resolved_warmup_steps"),
        "mixed_precision": training.get("resolved_mixed_precision"),
        "expected_world_size": distributed.get(
            "expected_world_size", training.get("effective_world_size")
        ),
        "aligned_epoch_samples": iteration.get("aligned_epoch_samples"),
        "aligned_epoch_tokens": iteration.get("aligned_epoch_tokens"),
        "fixed_epoch_set_hash": iteration.get("fixed_epoch_set_hash"),
        "permutation_version": iteration.get("permutation_version"),
        "permutation_hash": iteration.get("permutation_hash"),
        "ordering_policy_version": iteration.get("ordering_policy_version"),
    }


def _base_rejections(run: Mapping[str, Any], *, role: str, budget: int) -> list[str]:
    config = run["config"]
    summary = run["summary"]
    actual_role, contract = _controlled_contract(run)
    rejections: list[str] = []
    if summary.get("status") != "completed":
        rejections.append("run status is not completed")
    if _integer(summary.get("token_budget", -1), "summary.token_budget") != budget:
        rejections.append(f"run budget is not exactly {budget}")
    if _integer(summary.get("tokens_seen", -1), "summary.tokens_seen") != budget:
        rejections.append("completed run did not spend its complete token budget")
    if summary.get("unresolved_artifact_failures"):
        rejections.append("run reports unresolved artifact failures")
    if actual_role != role:
        rejections.append(f"comparison role is not {role}")
    if (
        config.get("controlled_experiment", {}).get("comparison_group_id")
        != COMPARISON_GROUP_ID
    ):
        rejections.append("comparison group ID mismatch")
    schema_version = contract.get("schema_version")
    legacy_reference = (
        role == "standalone_reference"
        and actual_role == "standalone_reference"
        and schema_version == 1
        and contract.get("lr_selection_manifest_path") in (None, "")
        and contract.get("lr_selection_manifest_hash") in (None, "")
    )
    if schema_version != 2 and not legacy_reference:
        rejections.append("portfolio contract schema mismatch")
    if contract.get("reference_budget_tokens") != REFERENCE_BUDGET_TOKENS:
        rejections.append("reference budget contract mismatch")
    if contract.get("elastic_budget_cap_tokens") != ELASTIC_BUDGET_CAP_TOKENS:
        rejections.append("elastic budget contract mismatch")
    if contract.get("aggregate_reference_count") != 4:
        rejections.append("aggregate reference count mismatch")
    if list(contract.get("granularities", [])) != list(GRANULARITIES):
        rejections.append("portfolio width order mismatch")
    if contract.get("perplexity_tolerance") != PERPLEXITY_TOLERANCE:
        rejections.append("perplexity tolerance mismatch")
    if contract.get("required_consecutive_evaluations") != REQUIRED_STREAK:
        rejections.append("streak length mismatch")
    if config.get("training", {}).get("scheduler_name") != "cosine":
        rejections.append("scheduler is not cosine")
    return rejections


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != dict(payload):
            raise PortfolioAnalysisError(
                f"Immutable manifest exists with different provenance: {path}"
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


def _load_hashed_manifest(path: str | Path, name: str) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("manifest_hash") != manifest_hash(payload):
        raise PortfolioAnalysisError(f"{name} manifest hash mismatch")
    return payload


def freeze_references(
    run_dirs: Iterable[str | Path], output_dir: str | Path
) -> dict[str, Any]:
    runs = [_load_run(path) for path in run_dirs]
    matrix: dict[tuple[int, str], dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    for run in runs:
        seed = _seed(run)
        config = run["config"]
        widths = list(config.get("model", {}).get("granularities", []))
        width = widths[0] if len(widths) == 1 else None
        rejections = _base_rejections(
            run, role="standalone_reference", budget=REFERENCE_BUDGET_TOKENS
        )
        if (
            config.get("run", {}).get("model_family") != "standalone"
            or width not in GRANULARITIES
        ):
            rejections.append("standalone reference must activate one portfolio width")
        if not math.isclose(_learning_rate(run), 0.008, rel_tol=0.0, abs_tol=1e-12):
            rejections.append("standalone reference learning rate is not 0.008")
        rows = _ordinary_rows(run, str(width)) if width in GRANULARITIES else []
        if not rows:
            rejections.append("standalone reference has no ordinary validation")
        best = min(rows, key=lambda row: (row["loss"], row["step"])) if rows else None
        if best is not None and best["tokens_seen"] > REFERENCE_BUDGET_TOKENS:
            rejections.append("best checkpoint was selected beyond the B-token horizon")
        checkpoint = _resolve_checkpoint(
            run["run_dir"], run["summary"].get("best_checkpoint_path")
        )
        if run["summary"].get("checkpoint_status") != "best_eval" or checkpoint is None:
            rejections.append("ordinary-validation best checkpoint is unavailable")
        selection_step = run["summary"].get("checkpoint_selection_step")
        if (
            best is not None
            and _integer(selection_step, "checkpoint selection step") != best["step"]
        ):
            rejections.append("best checkpoint does not match the best validation row")
        key = (seed, str(width))
        if key in matrix:
            raise PortfolioAnalysisError(
                f"Duplicate reference for seed={seed}, width={width}"
            )
        matrix[key] = run
        diagnostics.append(
            {
                "seed": seed,
                "granularity": width,
                "run_id": run["summary"].get("run_id", run["run_dir"].name),
                "run_dir": str(run["run_dir"]),
                "contract_satisfied": not rejections,
                "rejection_reasons": rejections,
                "best_step": best["step"] if best else None,
                "best_tokens": best["tokens_seen"] if best else None,
                "target_loss": best["loss"] if best else None,
                "target_perplexity": best["perplexity"] if best else None,
                "checkpoint_path": str(checkpoint) if checkpoint else None,
                "checkpoint_sha256": _sha256(checkpoint) if checkpoint else None,
                "validation_count": len(rows),
                "provenance": _run_provenance(run),
            }
        )

    expected = {(seed, width) for seed in REQUIRED_SEEDS for width in GRANULARITIES}
    if set(matrix) != expected:
        missing = sorted(expected - set(matrix))
        extra = sorted(set(matrix) - expected)
        raise PortfolioAnalysisError(
            f"Reference matrix must be exactly 12 runs; missing={missing}, extra={extra}"
        )
    failed = [row for row in diagnostics if not row["contract_satisfied"]]
    if failed:
        raise PortfolioAnalysisError(f"Invalid standalone references: {failed}")
    provenance_hashes = {stable_hash(row["provenance"]) for row in diagnostics}
    if len(provenance_hashes) != 1:
        raise PortfolioAnalysisError(
            "Standalone references have mismatched data/model provenance"
        )

    targets: dict[str, dict[str, Any]] = {}
    for seed in REQUIRED_SEEDS:
        targets[str(seed)] = {}
        for width in GRANULARITIES:
            row = next(
                item
                for item in diagnostics
                if item["seed"] == seed and item["granularity"] == width
            )
            targets[str(seed)][width] = {
                key: row[key]
                for key in (
                    "run_id",
                    "run_dir",
                    "best_step",
                    "best_tokens",
                    "target_loss",
                    "target_perplexity",
                    "checkpoint_path",
                    "checkpoint_sha256",
                )
            }
    manifest = {
        "schema_version": 1,
        "analysis": "tinystories_instruct_standalone_portfolio_targets",
        "status": "references_frozen",
        "comparison_group_id": COMPARISON_GROUP_ID,
        "reference_budget_tokens": REFERENCE_BUDGET_TOKENS,
        "aggregate_reference_count": 4,
        "aggregate_reference_budget_tokens": AGGREGATE_REFERENCE_BUDGET_TOKENS,
        "granularities": list(GRANULARITIES),
        "seeds": list(REQUIRED_SEEDS),
        "perplexity_tolerance": PERPLEXITY_TOLERANCE,
        "loss_tolerance": LOSS_TOLERANCE,
        "target_definition": "best_ordinary_validation_checkpoint_within_B",
        "shared_provenance": diagnostics[0]["provenance"],
        "targets": targets,
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    output = Path(output_dir).expanduser().resolve()
    _write_immutable_json(output / "standalone_portfolio_targets.json", manifest)
    _write_csv(output / "standalone_portfolio_targets.csv", diagnostics)
    diagnostics_payload = {
        "schema_version": 1,
        "status": "complete",
        "target_manifest_hash": manifest["manifest_hash"],
        "runs": diagnostics,
    }
    diagnostics_payload["diagnostics_hash"] = stable_hash(diagnostics_payload)
    _write_json(output / "standalone_portfolio_diagnostics.json", diagnostics_payload)

    pyplot = _pyplot()
    figure, axes = pyplot.subplots(2, 2, figsize=(10, 7), sharex=True)
    for axis, width in zip(axes.flat, GRANULARITIES, strict=True):
        for seed in REQUIRED_SEEDS:
            run = matrix[(seed, width)]
            rows = _ordinary_rows(run, width)
            axis.plot(
                [row["tokens_seen"] for row in rows],
                [row["loss"] for row in rows],
                label=f"seed {seed}",
            )
            target = targets[str(seed)][width]
            axis.scatter([target["best_tokens"]], [target["target_loss"]], s=18)
        axis.set_title(f"{width} standalone, B={REFERENCE_BUDGET_TOKENS:,}")
        axis.set_ylabel("validation loss")
    for axis in axes[-1]:
        axis.set_xlabel("optimizer tokens")
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "standalone_portfolio_diagnostics.png", dpi=160)
    pyplot.close(figure)
    return manifest


def _validate_elastic_run(
    run: Mapping[str, Any], *, role: str, budget: int
) -> list[str]:
    config = run["config"]
    model = config.get("model", {})
    rejections = _base_rejections(run, role=role, budget=budget)
    if (
        config.get("run", {}).get("model_family") != "nested"
        or config.get("run", {}).get("sampling_mode") != "nested-random"
    ):
        rejections.append("elastic run is not nested-random")
    if list(model.get("granularities", [])) != list(GRANULARITIES):
        rejections.append("elastic run does not expose the four-width portfolio")
    if model.get("granularity_sampling_mode") != "global":
        rejections.append("elastic run is not uniform-global")
    if model.get("global_sampling_schedule") != "random_with_replacement":
        rejections.append("elastic run does not sample uniformly with replacement")
    if model.get("global_sampling_interval_steps") != 1:
        rejections.append("elastic run does not use H=1")
    return rejections


def _offline_catchup(
    run: Mapping[str, Any], target_by_width: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    streak = 0
    onset_step = None
    onset_tokens = None
    confirmation = None
    observations: list[dict[str, Any]] = []
    for point in _joint_rows(run):
        widths: dict[str, Any] = {}
        for width in GRANULARITIES:
            target_loss = _number(target_by_width[width]["target_loss"], "target loss")
            loss = point["widths"][width]["loss"]
            gap = loss - target_loss
            widths[width] = {
                "loss": loss,
                "target_loss": target_loss,
                "loss_gap": gap,
                "perplexity_deficit": math.expm1(gap),
                "qualifies": gap <= LOSS_TOLERANCE,
            }
        joint = all(row["qualifies"] for row in widths.values())
        if confirmation is None:
            if joint:
                if streak == 0:
                    onset_step = point["step"]
                    onset_tokens = point["tokens_seen"]
                streak += 1
            else:
                streak = 0
                onset_step = None
                onset_tokens = None
            if streak == REQUIRED_STREAK:
                confirmation = {
                    "onset_step": onset_step,
                    "onset_tokens": onset_tokens,
                    "confirmation_step": point["step"],
                    "confirmation_tokens": point["tokens_seen"],
                }
        observations.append(
            {
                "step": point["step"],
                "tokens_seen": point["tokens_seen"],
                "widths": widths,
                "joint_max_loss_gap": max(row["loss_gap"] for row in widths.values()),
                "joint_qualifies": joint,
                "streak_length": min(streak, REQUIRED_STREAK),
            }
        )
    return observations, confirmation or {
        "onset_step": None,
        "onset_tokens": None,
        "confirmation_step": None,
        "confirmation_tokens": None,
    }


def _validate_confirmation_checkpoint(
    checkpoint_path: Path,
    *,
    target_hash: str,
    confirmation_step: int,
) -> None:
    try:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise PortfolioAnalysisError(
            f"Cannot load portfolio confirmation checkpoint: {checkpoint_path}"
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise PortfolioAnalysisError("Portfolio confirmation checkpoint is malformed")
    state = checkpoint.get("portfolio_catchup_state")
    if checkpoint.get(
        "checkpoint_status"
    ) != "portfolio_catchup_confirmation" or not isinstance(state, Mapping):
        raise PortfolioAnalysisError(
            "Checkpoint is not a portfolio confirmation artifact"
        )
    if int(state.get("confirmation_step", -1)) != confirmation_step:
        raise PortfolioAnalysisError("Confirmation checkpoint step mismatch")
    if state.get("target_manifest_hash") != target_hash or not math.isclose(
        _number(state.get("learning_rate"), "checkpoint learning rate"),
        FIXED_LEARNING_RATE,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise PortfolioAnalysisError(
            "Confirmation checkpoint manifest provenance mismatch"
        )
    if not isinstance(checkpoint.get("model_state_dict"), Mapping):
        raise PortfolioAnalysisError("Confirmation checkpoint lacks exact model state")


def _validate_terminal_checkpoint(
    checkpoint_path: Path,
    *,
    run: Mapping[str, Any],
    target_hash: str,
) -> None:
    try:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise PortfolioAnalysisError(
            f"Cannot load terminal 3B checkpoint: {checkpoint_path}"
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise PortfolioAnalysisError("Terminal 3B checkpoint is malformed")
    summary = run["summary"]
    expected_step = _integer(summary.get("steps_completed"), "completed step")
    if (
        checkpoint.get("checkpoint_status") != "latest"
        or _integer(checkpoint.get("step"), "terminal checkpoint step")
        != expected_step
        or _integer(checkpoint.get("tokens_seen"), "terminal checkpoint tokens")
        != ELASTIC_BUDGET_CAP_TOKENS
    ):
        raise PortfolioAnalysisError(
            "Terminal diagnostic checkpoint is not the completed 3B boundary"
        )
    run_id = summary.get("run_id", run["run_dir"].name)
    if checkpoint.get("run_id") != run_id:
        raise PortfolioAnalysisError("Terminal checkpoint run ID mismatch")
    state = checkpoint.get("portfolio_catchup_state")
    if not isinstance(state, Mapping):
        raise PortfolioAnalysisError(
            "Terminal checkpoint lacks portfolio catch-up provenance"
        )
    if state.get("target_manifest_hash") != target_hash or not math.isclose(
        _number(state.get("learning_rate"), "terminal checkpoint learning rate"),
        FIXED_LEARNING_RATE,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise PortfolioAnalysisError(
            "Terminal checkpoint target or recipe provenance mismatch"
        )
    summary_state = summary.get("portfolio_catchup_state")
    if not isinstance(summary_state, Mapping) or any(
        state.get(field) != summary_state.get(field)
        for field in (
            "confirmed",
            "confirmation_step",
            "confirmation_tokens",
        )
    ):
        raise PortfolioAnalysisError(
            "Terminal checkpoint catch-up state disagrees with the completed run"
        )
    if not isinstance(checkpoint.get("model_state_dict"), Mapping):
        raise PortfolioAnalysisError("Terminal checkpoint lacks exact model state")


def _terminal_checkpoint_selection(
    run: Mapping[str, Any], *, target_hash: str
) -> tuple[Path, str, int]:
    summary = run["summary"]
    checkpoint = _resolve_checkpoint(
        run["run_dir"], summary.get("latest_checkpoint_path")
    )
    if checkpoint is None:
        raise PortfolioAnalysisError(
            "Completed elastic run lacks its terminal latest checkpoint"
        )
    _validate_terminal_checkpoint(checkpoint, run=run, target_hash=target_hash)
    return (
        checkpoint,
        _sha256(checkpoint),
        _integer(summary.get("steps_completed"), "completed step"),
    )


def portfolio_catchup(
    run_dirs: Iterable[str | Path],
    target_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    targets = _load_hashed_manifest(target_manifest_path, "standalone targets")
    runs = [_load_run(path) for path in run_dirs]
    by_seed: dict[int, Mapping[str, Any]] = {}
    for run in runs:
        seed = _seed(run)
        if seed in by_seed:
            raise PortfolioAnalysisError(f"Duplicate elastic candidate seed {seed}")
        by_seed[seed] = run
    if set(by_seed) != set(REQUIRED_SEEDS):
        raise PortfolioAnalysisError(
            f"Elastic candidates must contain seeds {list(REQUIRED_SEEDS)}"
        )

    reference_run_ids = {
        target["run_id"]
        for seed_targets in targets["targets"].values()
        for target in seed_targets.values()
    }
    seed_reports: list[dict[str, Any]] = []
    observations_by_seed: dict[int, list[dict[str, Any]]] = {}
    holdout_entries: list[dict[str, Any]] = []
    for seed in REQUIRED_SEEDS:
        run = by_seed[seed]
        rejections = _validate_elastic_run(
            run, role="elastic_candidate", budget=ELASTIC_BUDGET_CAP_TOKENS
        )
        config = run["config"]
        contract = config["controlled_experiment"]["portfolio_catchup"]
        if contract.get("target_manifest_hash") != targets["manifest_hash"]:
            rejections.append("candidate target-manifest hash mismatch")
        if not math.isclose(
            _learning_rate(run), FIXED_LEARNING_RATE, rel_tol=0.0, abs_tol=1e-12
        ):
            rejections.append("candidate learning rate is not fixed at 0.008")
        if _run_provenance(run) != targets.get("shared_provenance"):
            rejections.append("candidate data/model provenance differs from references")
        run_id = run["summary"].get("run_id", run["run_dir"].name)
        if run_id in reference_run_ids:
            rejections.append("candidate run is not fresh")
        if rejections:
            raise PortfolioAnalysisError(
                f"Invalid elastic candidate seed {seed}: {rejections}"
            )

        observations, offline = _offline_catchup(run, targets["targets"][str(seed)])
        observations_by_seed[seed] = observations
        state = run["summary"].get("portfolio_catchup_state")
        if not isinstance(state, Mapping):
            raise PortfolioAnalysisError(
                "Candidate run summary lacks online catch-up state"
            )
        if state.get("target_manifest_hash") != targets[
            "manifest_hash"
        ] or not math.isclose(
            _number(state.get("learning_rate"), "catch-up state learning rate"),
            FIXED_LEARNING_RATE,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise PortfolioAnalysisError(
                "Candidate online catch-up state violates fixed-recipe provenance"
            )
        state_confirmation = state.get("confirmation_step")
        if (
            state_confirmation != offline["confirmation_step"]
            or state.get("confirmation_tokens") != offline["confirmation_tokens"]
        ):
            raise PortfolioAnalysisError(
                "Online and independently recomputed confirmation disagree"
            )
        if (
            state.get("streak_onset_step") != offline["onset_step"]
            or state.get("streak_onset_tokens") != offline["onset_tokens"]
        ):
            raise PortfolioAnalysisError(
                "Online and independently recomputed onset disagree"
            )

        caught_up = offline["confirmation_step"] is not None
        checkpoint = None
        checkpoint_sha = None
        if caught_up:
            if state.get("confirmation_checkpoint_saved") is not True:
                raise PortfolioAnalysisError(
                    "Confirmed run did not persist its checkpoint"
                )
            checkpoint = _resolve_checkpoint(
                run["run_dir"], state.get("confirmation_checkpoint_path")
            )
            if checkpoint is None:
                raise PortfolioAnalysisError(
                    "Confirmation checkpoint artifact is unavailable"
                )
            checkpoint_sha = _sha256(checkpoint)
            if checkpoint_sha != state.get("confirmation_checkpoint_sha256"):
                raise PortfolioAnalysisError("Confirmation checkpoint SHA256 mismatch")
            _validate_confirmation_checkpoint(
                checkpoint,
                target_hash=targets["manifest_hash"],
                confirmation_step=int(offline["confirmation_step"]),
            )
        confirmation_tokens = offline["confirmation_tokens"]
        seed_report = {
            "seed": seed,
            "run_id": run_id,
            "run_dir": str(run["run_dir"]),
            "caught_up": caught_up,
            "censored": not caught_up,
            **offline,
            "confirmation_checkpoint_path": str(checkpoint) if checkpoint else None,
            "confirmation_checkpoint_sha256": checkpoint_sha,
            "t_star_over_B": confirmation_tokens / REFERENCE_BUDGET_TOKENS
            if caught_up
            else None,
            "t_star_over_4B": confirmation_tokens / AGGREGATE_REFERENCE_BUDGET_TOKENS
            if caught_up
            else None,
            "required_savings_fraction": 1.0
            - confirmation_tokens / AGGREGATE_REFERENCE_BUDGET_TOKENS
            if caught_up
            else None,
            "realized_full_run_tokens": ELASTIC_BUDGET_CAP_TOKENS,
            "realized_full_run_spend_over_4B": ELASTIC_BUDGET_CAP_TOKENS
            / AGGREGATE_REFERENCE_BUDGET_TOKENS,
            "final_per_width_deficits": observations[-1]["widths"],
            "validation_observations": observations,
        }
        seed_reports.append(seed_report)
    for seed in REQUIRED_SEEDS:
        for width in GRANULARITIES:
            target = targets["targets"][str(seed)][width]
            holdout_entries.append(
                {
                    "comparison_role": "standalone_reference",
                    "seed": seed,
                    "granularities": [width],
                    "run_id": target["run_id"],
                    "run_dir": target["run_dir"],
                    "checkpoint_selection": "ordinary_validation_best",
                    "checkpoint_path": target["checkpoint_path"],
                    "checkpoint_sha256": target["checkpoint_sha256"],
                    "checkpoint_step": target["best_step"],
                    "checkpoint_tokens": target["best_tokens"],
                    "result_path": str(
                        Path(target["run_dir"]) / "final_holdout_results.json"
                    ),
                }
            )

    all_caught_up = all(row["caught_up"] for row in seed_reports)
    selection_mode = (
        "portfolio_confirmation" if all_caught_up else "terminal_3B_censored"
    )
    for seed_report in seed_reports:
        seed = int(seed_report["seed"])
        run = by_seed[seed]
        if all_caught_up:
            checkpoint = Path(seed_report["confirmation_checkpoint_path"])
            checkpoint_sha = seed_report["confirmation_checkpoint_sha256"]
            checkpoint_step = int(seed_report["confirmation_step"])
            checkpoint_tokens = int(seed_report["confirmation_tokens"])
            checkpoint_selection = "portfolio_confirmation"
        else:
            checkpoint, checkpoint_sha, checkpoint_step = (
                _terminal_checkpoint_selection(
                    run,
                    target_hash=targets["manifest_hash"],
                )
            )
            checkpoint_tokens = ELASTIC_BUDGET_CAP_TOKENS
            checkpoint_selection = "terminal_3B"
        seed_report["holdout_checkpoint_selection"] = checkpoint_selection
        seed_report["holdout_checkpoint_path"] = str(checkpoint)
        seed_report["holdout_checkpoint_sha256"] = checkpoint_sha
        holdout_entries.append(
            {
                "comparison_role": "elastic_candidate",
                "seed": seed,
                "granularities": list(GRANULARITIES),
                "run_id": seed_report["run_id"],
                "run_dir": str(run["run_dir"]),
                "checkpoint_selection": checkpoint_selection,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_step": checkpoint_step,
                "checkpoint_tokens": checkpoint_tokens,
                "result_path": str(run["run_dir"] / "final_holdout_results.json"),
            }
        )
    confirmed_tokens = [
        row["confirmation_tokens"] for row in seed_reports if row["caught_up"]
    ]
    budget_summary = None
    if all_caught_up:
        required_tokens = max(confirmed_tokens)
        budget_summary = {
            "cross_seed_required_tokens": required_tokens,
            "cross_seed_t_star_over_B": required_tokens / REFERENCE_BUDGET_TOKENS,
            "cross_seed_t_star_over_4B": required_tokens
            / AGGREGATE_REFERENCE_BUDGET_TOKENS,
            "cross_seed_required_savings_fraction": 1.0
            - required_tokens / AGGREGATE_REFERENCE_BUDGET_TOKENS,
            "mean_confirmation_tokens": statistics.fmean(confirmed_tokens),
            "median_confirmation_tokens": statistics.median(confirmed_tokens),
        }
    report = {
        "schema_version": 2,
        "analysis": "tinystories_instruct_four_granularity_portfolio_catchup",
        "status": "portfolio_catchup_confirmed" if all_caught_up else "censored",
        "general_portfolio_catchup_claim": all_caught_up,
        "comparison_group_id": COMPARISON_GROUP_ID,
        "reference_budget_tokens": REFERENCE_BUDGET_TOKENS,
        "aggregate_reference_budget_tokens": AGGREGATE_REFERENCE_BUDGET_TOKENS,
        "elastic_budget_cap_tokens": ELASTIC_BUDGET_CAP_TOKENS,
        "realized_full_run_spend_over_4B": ELASTIC_BUDGET_CAP_TOKENS
        / AGGREGATE_REFERENCE_BUDGET_TOKENS,
        "perplexity_tolerance": PERPLEXITY_TOLERANCE,
        "loss_tolerance": LOSS_TOLERANCE,
        "required_consecutive_evaluations": REQUIRED_STREAK,
        "target_manifest_path": str(Path(target_manifest_path).expanduser().resolve()),
        "target_manifest_hash": targets["manifest_hash"],
        "learning_rate": FIXED_LEARNING_RATE,
        "optimizer_recipe_policy": "same_fixed_recipe_across_roles",
        "seeds": seed_reports,
        "budget_summary": budget_summary,
        "final_holdout_selection_status": (
            "ready_confirmatory"
            if all_caught_up
            else "ready_diagnostic_terminal_3B"
        ),
        "final_holdout_selection_mode": selection_mode,
        "final_holdout_claim_eligible": all_caught_up,
    }
    report["report_hash"] = stable_hash(report)
    output = Path(output_dir).expanduser().resolve()
    _write_json(output / "portfolio_catchup_report.json", report)
    flat_rows = []
    for seed_report in seed_reports:
        for observation in seed_report["validation_observations"]:
            for width, values in observation["widths"].items():
                flat_rows.append(
                    {
                        "seed": seed_report["seed"],
                        "run_id": seed_report["run_id"],
                        "step": observation["step"],
                        "tokens_seen": observation["tokens_seen"],
                        "granularity": width,
                        **values,
                        "joint_max_loss_gap": observation["joint_max_loss_gap"],
                        "joint_qualifies": observation["joint_qualifies"],
                        "streak_length": observation["streak_length"],
                        "confirmation_step": seed_report["confirmation_step"],
                        "confirmation_tokens": seed_report["confirmation_tokens"],
                    }
                )
    _write_csv(output / "portfolio_catchup.csv", flat_rows)

    holdout_manifest = {
        "schema_version": 2,
        "analysis": "portfolio_final_holdout_selection",
        "status": (
            "ready_confirmatory"
            if all_caught_up
            else "ready_diagnostic_terminal_3B"
        ),
        "selection_mode": selection_mode,
        "claim_eligible": all_caught_up,
        "interpretation": (
            "confirmation_checkpoint_generalization"
            if all_caught_up
            else "terminal_3B_diagnostic_only_no_catchup_claim"
        ),
        "comparison_group_id": COMPARISON_GROUP_ID,
        "target_manifest_hash": targets["manifest_hash"],
        "shared_corpus_hash": targets["shared_provenance"]["corpus_hash"],
        "final_holdout_manifest_hash": targets["shared_provenance"][
            "final_holdout_manifest_hash"
        ],
        "required_checkpoint_count": 15,
        "entries": sorted(
            holdout_entries,
            key=lambda row: (
                row["seed"],
                row["comparison_role"],
                row["granularities"],
            ),
        ),
    }
    holdout_manifest["manifest_hash"] = manifest_hash(holdout_manifest)
    _write_immutable_json(
        output / "final_holdout_selection_manifest.json", holdout_manifest
    )

    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(9, 5))
    for seed, observations in observations_by_seed.items():
        axis.plot(
            [row["tokens_seen"] for row in observations],
            [row["joint_max_loss_gap"] for row in observations],
            label=f"seed {seed}",
        )
        confirmation = next(row for row in seed_reports if row["seed"] == seed)[
            "confirmation_tokens"
        ]
        if confirmation is not None:
            axis.axvline(confirmation, alpha=0.35, linestyle=":")
    axis.axhline(
        LOSS_TOLERANCE, color="black", linestyle="--", label="0.5% PPL threshold"
    )
    axis.set(xlabel="optimizer tokens", ylabel="worst-width loss gap")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "portfolio_joint_deficit.png", dpi=160)
    pyplot.close(figure)

    figure, axes = pyplot.subplots(2, 2, figsize=(10, 7), sharex=True)
    for axis, width in zip(axes.flat, GRANULARITIES, strict=True):
        for seed, observations in observations_by_seed.items():
            axis.plot(
                [row["tokens_seen"] for row in observations],
                [row["widths"][width]["loss_gap"] for row in observations],
                label=f"seed {seed}",
            )
        axis.axhline(LOSS_TOLERANCE, color="black", linestyle="--")
        axis.set_title(width)
        axis.set_ylabel("loss gap")
    for axis in axes[-1]:
        axis.set_xlabel("optimizer tokens")
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "portfolio_per_granularity_deficits.png", dpi=160)
    pyplot.close(figure)
    return report


def final_holdout(
    selection_manifest_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    selection = _load_hashed_manifest(selection_manifest_path, "holdout selection")
    selection_mode = selection.get("selection_mode", "portfolio_confirmation")
    if selection_mode not in {"portfolio_confirmation", "terminal_3B_censored"}:
        raise PortfolioAnalysisError("Final-holdout selection mode is invalid")
    claim_eligible = selection.get(
        "claim_eligible", selection_mode == "portfolio_confirmation"
    )
    if not isinstance(claim_eligible, bool) or claim_eligible != (
        selection_mode == "portfolio_confirmation"
    ):
        raise PortfolioAnalysisError(
            "Final-holdout claim eligibility contradicts its selection mode"
        )
    entries = selection.get("entries")
    if not isinstance(entries, list) or len(entries) != 15:
        raise PortfolioAnalysisError(
            "Final holdout requires all 15 selected checkpoints"
        )
    results: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise PortfolioAnalysisError("Final-holdout selection entry is malformed")
        role = str(entry.get("comparison_role"))
        checkpoint_selection = entry.get("checkpoint_selection")
        expected_selection = (
            "ordinary_validation_best"
            if role == "standalone_reference"
            else (
                "portfolio_confirmation"
                if selection_mode == "portfolio_confirmation"
                else "terminal_3B"
            )
        )
        if checkpoint_selection != expected_selection:
            raise PortfolioAnalysisError(
                "Final-holdout checkpoint selection contradicts its mode"
            )
        checkpoint = Path(str(entry["checkpoint_path"])).expanduser().resolve()
        if not checkpoint.is_file() or _sha256(checkpoint) != entry.get(
            "checkpoint_sha256"
        ):
            raise PortfolioAnalysisError(
                "Final-holdout selected checkpoint is missing or stale"
            )
        result = _read_json(entry["result_path"])
        body = dict(result)
        saved_hash = body.pop("result_hash", None)
        if saved_hash != stable_hash(body):
            raise PortfolioAnalysisError("Final-holdout result hash mismatch")
        requested = checkpoint
        actual = Path(str(result.get("checkpoint_path", ""))).expanduser().resolve()
        if actual != requested:
            raise PortfolioAnalysisError(
                "Final-holdout result belongs to a different checkpoint"
            )
        if result.get("run_id") != entry.get("run_id"):
            raise PortfolioAnalysisError("Final-holdout result run ID mismatch")
        if result.get("checkpoint_sha256") != entry.get("checkpoint_sha256"):
            raise PortfolioAnalysisError(
                "Final-holdout result checkpoint checksum mismatch"
            )
        if result.get("final_holdout_manifest_hash") != selection.get(
            "final_holdout_manifest_hash"
        ):
            raise PortfolioAnalysisError(
                "Final-holdout result manifest provenance mismatch"
            )
        key = (str(entry["comparison_role"]), int(entry["seed"]))
        if key in results and key[0] == "elastic_candidate":
            raise PortfolioAnalysisError("Duplicate elastic final-holdout result")
        if key[0] == "standalone_reference":
            width = str(entry["granularities"][0])
            key = (f"standalone_reference:{width}", int(entry["seed"]))
        results[key] = result

    comparisons: list[dict[str, Any]] = []
    for seed in REQUIRED_SEEDS:
        elastic = results.get(("elastic_candidate", seed))
        if elastic is None:
            raise PortfolioAnalysisError(
                f"Missing elastic final-holdout result for seed {seed}"
            )
        elastic_by_width = {
            row["granularity"]: row
            for row in elastic.get("ordered_per_granularity_losses", [])
        }
        if set(elastic_by_width) != set(GRANULARITIES):
            raise PortfolioAnalysisError("Elastic final holdout lacks the four widths")
        for width in GRANULARITIES:
            standalone = results.get((f"standalone_reference:{width}", seed))
            if standalone is None:
                raise PortfolioAnalysisError(
                    f"Missing standalone final-holdout result for seed {seed}, {width}"
                )
            components = standalone.get("ordered_per_granularity_losses", [])
            if len(components) != 1 or components[0].get("granularity") != width:
                raise PortfolioAnalysisError("Standalone final holdout width mismatch")
            standalone_ppl = _number(
                components[0]["perplexity"], "standalone perplexity"
            )
            elastic_ppl = _number(
                elastic_by_width[width]["perplexity"], "elastic perplexity"
            )
            deficit = elastic_ppl / standalone_ppl - 1.0
            comparisons.append(
                {
                    "seed": seed,
                    "granularity": width,
                    "standalone_perplexity": standalone_ppl,
                    "elastic_perplexity": elastic_ppl,
                    "perplexity_deficit": deficit,
                    "passes": deficit <= PERPLEXITY_TOLERANCE,
                    "standalone_checkpoint_path": standalone["checkpoint_path"],
                    "elastic_checkpoint_path": elastic["checkpoint_path"],
                }
            )
    all_pass = all(row["passes"] for row in comparisons)
    general_claim = all_pass and claim_eligible
    diagnostic_equivalence = (
        all_pass if selection_mode == "terminal_3B_censored" else None
    )
    if selection_mode == "portfolio_confirmation":
        status = "portfolio_equivalent" if all_pass else "portfolio_not_equivalent"
    else:
        status = (
            "diagnostic_terminal_3B_equivalent"
            if all_pass
            else "diagnostic_terminal_3B_not_equivalent"
        )
    report = {
        "schema_version": 2,
        "analysis": "tinystories_instruct_portfolio_final_holdout",
        "status": status,
        "selection_mode": selection_mode,
        "claim_eligible": claim_eligible,
        "all_pairs_within_tolerance": all_pass,
        "diagnostic_terminal_3B_equivalence": diagnostic_equivalence,
        "general_portfolio_equivalence_claim": general_claim,
        "perplexity_tolerance": PERPLEXITY_TOLERANCE,
        "required_checkpoint_count": 15,
        "selection_manifest_path": str(
            Path(selection_manifest_path).expanduser().resolve()
        ),
        "selection_manifest_hash": selection["manifest_hash"],
        "comparisons": comparisons,
    }
    report["report_hash"] = stable_hash(report)
    output = Path(output_dir).expanduser().resolve()
    _write_json(output / "portfolio_final_holdout_report.json", report)
    _write_csv(output / "portfolio_final_holdout.csv", comparisons)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("freeze-references", "freeze the exact 12-run standalone matrix"),
        ("portfolio-catchup", "verify the three fresh 3B elastic candidates"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--run-dir", action="append", default=[])
        command.add_argument("--runs-root", action="append", default=[])
        command.add_argument("--output-dir", required=True)
        if name == "portfolio-catchup":
            command.add_argument("--target-manifest", required=True)
    holdout = commands.add_parser(
        "final-holdout", help="verify all 15 sealed holdout results"
    )
    holdout.add_argument("--selection-manifest", required=True)
    holdout.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "final-holdout":
        result = final_holdout(args.selection_manifest, args.output_dir)
    else:
        run_dirs = discover_run_dirs(args.run_dir, args.runs_root)
        if args.command == "freeze-references":
            result = freeze_references(run_dirs, args.output_dir)
        else:
            result = portfolio_catchup(
                run_dirs,
                args.target_manifest,
                args.output_dir,
            )
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
