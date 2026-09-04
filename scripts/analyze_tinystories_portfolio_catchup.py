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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.portfolio_catchup import (
    candidate_policy_contract,
    candidate_policy_contract_hash,
    manifest_hash,
)
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
GRANULARITY_PROFILES = {
    "quartile": {
        "comparison_group_id": COMPARISON_GROUP_ID,
        "granularities": GRANULARITIES,
    },
    "matformer": {
        "comparison_group_id": (
            "tinystories_instruct_portfolio_catchup_matformer_granularities_v1"
        ),
        "granularities": ("g125", "g250", "g500", "g1000"),
    },
}
CANDIDATE_ARMS = {
    "uniform_h1_3b": {
        "budget_tokens": ELASTIC_BUDGET_CAP_TOKENS,
        "schema_version": 2,
        "sampling_mode": "nested-random",
        "model_variant": "slicing",
        "post_hoc_diagnostic": False,
    },
    "uniform_h1_4b": {
        "budget_tokens": AGGREGATE_REFERENCE_BUDGET_TOKENS,
        "schema_version": 3,
        "sampling_mode": "nested-random",
        "model_variant": "slicing",
        "post_hoc_diagnostic": True,
    },
    "nested_all_b": {
        "budget_tokens": REFERENCE_BUDGET_TOKENS,
        "schema_version": 3,
        "sampling_mode": "nested-all",
        "model_variant": "slicing",
        "post_hoc_diagnostic": True,
    },
    "nested_all_4b": {
        "budget_tokens": AGGREGATE_REFERENCE_BUDGET_TOKENS,
        "schema_version": 3,
        "sampling_mode": "nested-all",
        "model_variant": "slicing",
        "post_hoc_diagnostic": True,
    },
    "concat_uniform_h1_4b": {
        "budget_tokens": AGGREGATE_REFERENCE_BUDGET_TOKENS,
        "schema_version": 3,
        "sampling_mode": "nested-random",
        "model_variant": "concat",
        "allowed_reference_provenance_differences": {"model_variant"},
        "post_hoc_diagnostic": True,
    },
}


def _bundle_contract(run: Mapping[str, Any]) -> bool:
    _, contract = _controlled_contract(run)
    return int(contract.get("schema_version", 0)) == 4


def _candidate_arm_spec(run: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize legacy and schema-4 candidate identities for analysis."""

    arm_id = _candidate_arm_id(run)
    _, contract = _controlled_contract(run)
    if _bundle_contract(run):
        if not isinstance(arm_id, str) or not arm_id:
            return None
        return {
            "id": arm_id,
            "schema_version": 4,
            "budget_tokens": _integer(
                contract.get("elastic_budget_cap_tokens"),
                "elastic budget contract",
            ),
            "sampling_mode": run["config"].get("run", {}).get("sampling_mode"),
            "model_variant": run["config"].get("model", {}).get("variant"),
            "post_hoc_diagnostic": True,
            "policy_contract": candidate_policy_contract(run["config"]),
            "policy_contract_hash": candidate_policy_contract_hash(run["config"]),
            "allowed_reference_provenance_differences": set(),
        }
    if arm_id not in CANDIDATE_ARMS:
        return None
    return {"id": arm_id, **CANDIDATE_ARMS[str(arm_id)]}


class PortfolioAnalysisError(ValueError):
    """Raised when saved artifacts violate the controlled experiment."""


def _profile_name_for_group(comparison_group_id: Any) -> str:
    matches = [
        name
        for name, profile in GRANULARITY_PROFILES.items()
        if profile["comparison_group_id"] == comparison_group_id
    ]
    if len(matches) != 1:
        raise PortfolioAnalysisError(
            f"Unsupported portfolio comparison group: {comparison_group_id!r}"
        )
    return matches[0]


@contextmanager
def _active_granularity_profile(profile_name: str):
    if profile_name not in GRANULARITY_PROFILES:
        raise PortfolioAnalysisError(
            f"Unknown granularity profile {profile_name!r}; expected one of "
            f"{sorted(GRANULARITY_PROFILES)}"
        )
    global GRANULARITIES, COMPARISON_GROUP_ID
    previous_granularities = GRANULARITIES
    previous_group_id = COMPARISON_GROUP_ID
    profile = GRANULARITY_PROFILES[profile_name]
    GRANULARITIES = tuple(profile["granularities"])
    COMPARISON_GROUP_ID = str(profile["comparison_group_id"])
    try:
        yield
    finally:
        GRANULARITIES = previous_granularities
        COMPARISON_GROUP_ID = previous_group_id


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


def _candidate_arm_id(run: Mapping[str, Any]) -> str | None:
    controlled = run["config"].get("controlled_experiment", {})
    if not isinstance(controlled, Mapping):
        return None
    arm_id = controlled.get("comparison_arm_id")
    if arm_id in (None, ""):
        _, contract = _controlled_contract(run)
        if contract.get("schema_version") == 2:
            return "uniform_h1_3b"
        return None
    return str(arm_id)


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


def _base_rejections(
    run: Mapping[str, Any],
    *,
    role: str,
    budget: int,
    candidate_arm_id: str | None = None,
) -> list[str]:
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
    bundle_run = _bundle_contract(run)
    expected_schema = 4 if bundle_run else (
        CANDIDATE_ARMS[candidate_arm_id]["schema_version"]
        if role == "elastic_candidate" and candidate_arm_id in CANDIDATE_ARMS
        else 2
    )
    if schema_version != expected_schema and not legacy_reference:
        rejections.append("portfolio contract schema mismatch")
    reference_budget = _integer(
        contract.get("reference_budget_tokens"), "reference budget contract"
    )
    if not bundle_run and reference_budget != REFERENCE_BUDGET_TOKENS:
        rejections.append("reference budget contract mismatch")
    expected_elastic_budget = (
        _integer(contract.get("elastic_budget_cap_tokens"), "elastic budget contract")
        if bundle_run
        else (
            CANDIDATE_ARMS[candidate_arm_id]["budget_tokens"]
            if role == "elastic_candidate" and candidate_arm_id in CANDIDATE_ARMS
            else ELASTIC_BUDGET_CAP_TOKENS
        )
    )
    if contract.get("elastic_budget_cap_tokens") != expected_elastic_budget:
        rejections.append("elastic budget contract mismatch")
    if bundle_run:
        if contract.get("aggregate_reference_count") != 4:
            rejections.append("aggregate reference count mismatch")
        if contract.get("claim_tier") != "diagnostic":
            rejections.append("schema-4 candidate is not diagnostic")
        if contract.get("budget_unit_tokens") != REFERENCE_BUDGET_TOKENS:
            rejections.append("portfolio B unit mismatch")
        expected_reference = (
            _integer(contract.get("reference_budget_multiplier"), "reference multiplier")
            * REFERENCE_BUDGET_TOKENS
        )
        expected_candidate = (
            _integer(contract.get("candidate_budget_multiplier"), "candidate multiplier")
            * REFERENCE_BUDGET_TOKENS
        )
        if reference_budget != expected_reference or expected_elastic_budget != expected_candidate:
            rejections.append("portfolio multiplier budget contract mismatch")
    if role == "elastic_candidate" and _candidate_arm_id(run) != candidate_arm_id:
        rejections.append("comparison arm ID mismatch")
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


def _freeze_references_active(
    run_dirs: Iterable[str | Path], output_dir: str | Path
) -> dict[str, Any]:
    runs = [_load_run(path) for path in run_dirs]
    reference_budgets = {
        _integer(
            _controlled_contract(run)[1].get("reference_budget_tokens"),
            "reference budget contract",
        )
        for run in runs
    }
    if len(reference_budgets) != 1:
        raise PortfolioAnalysisError("Standalone references use multiple budget lanes")
    reference_budget = reference_budgets.pop()
    reference_multipliers = {
        _controlled_contract(run)[1].get("reference_budget_multiplier", 1)
        for run in runs
    }
    if len(reference_multipliers) != 1:
        raise PortfolioAnalysisError("Standalone references use multiple R multipliers")
    reference_multiplier = reference_multipliers.pop()
    matrix: dict[tuple[int, str], dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    for run in runs:
        seed = _seed(run)
        config = run["config"]
        widths = list(config.get("model", {}).get("granularities", []))
        width = widths[0] if len(widths) == 1 else None
        rejections = _base_rejections(
            run, role="standalone_reference", budget=reference_budget
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
        if best is not None and best["tokens_seen"] > reference_budget:
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

    matrix_seeds = {seed for seed, _ in matrix}
    unexpected_seeds = sorted(matrix_seeds - set(REQUIRED_SEEDS))
    if unexpected_seeds:
        raise PortfolioAnalysisError(
            "Standalone references contain out-of-contract seeds: "
            f"{unexpected_seeds}; expected a subset of {list(REQUIRED_SEEDS)}"
        )
    observed_seeds = tuple(seed for seed in REQUIRED_SEEDS if seed in matrix_seeds)
    if not observed_seeds:
        raise PortfolioAnalysisError("At least one standalone reference seed is required")
    incomplete_panels = {}
    expected_widths = set(GRANULARITIES)
    for seed in observed_seeds:
        actual_widths = {width for matrix_seed, width in matrix if matrix_seed == seed}
        if actual_widths != expected_widths:
            incomplete_panels[seed] = {
                "missing": sorted(expected_widths - actual_widths),
                "extra": sorted(actual_widths - expected_widths),
            }
    if incomplete_panels:
        raise PortfolioAnalysisError(
            "Every observed reference seed requires a complete four-width panel; "
            f"invalid_panels={incomplete_panels}"
        )
    missing_seeds = tuple(seed for seed in REQUIRED_SEEDS if seed not in observed_seeds)
    seed_coverage_complete = not missing_seeds
    failed = [row for row in diagnostics if not row["contract_satisfied"]]
    if failed:
        raise PortfolioAnalysisError(f"Invalid standalone references: {failed}")
    provenance_hashes = {stable_hash(row["provenance"]) for row in diagnostics}
    if len(provenance_hashes) != 1:
        raise PortfolioAnalysisError(
            "Standalone references have mismatched data/model provenance"
        )

    targets: dict[str, dict[str, Any]] = {}
    for seed in observed_seeds:
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
        "schema_version": 2 if reference_budget != REFERENCE_BUDGET_TOKENS else 1,
        "analysis": "tinystories_instruct_standalone_portfolio_targets",
        "status": (
            "references_frozen"
            if seed_coverage_complete
            else "references_frozen_provisional"
        ),
        "provisional_analysis": not seed_coverage_complete,
        "expected_seeds": list(REQUIRED_SEEDS),
        "observed_seeds": list(observed_seeds),
        "missing_seeds": list(missing_seeds),
        "expected_seed_count": len(REQUIRED_SEEDS),
        "observed_seed_count": len(observed_seeds),
        "seed_coverage_complete": seed_coverage_complete,
        "comparison_group_id": COMPARISON_GROUP_ID,
        "granularity_profile": _profile_name_for_group(COMPARISON_GROUP_ID),
        "reference_budget_tokens": reference_budget,
        "reference_budget_multiplier": reference_multiplier,
        "aggregate_reference_count": 4,
        "aggregate_reference_budget_tokens": reference_budget * 4,
        "granularities": list(GRANULARITIES),
        "seeds": list(observed_seeds),
        "perplexity_tolerance": PERPLEXITY_TOLERANCE,
        "loss_tolerance": LOSS_TOLERANCE,
        "target_definition": "best_ordinary_validation_checkpoint_within_reference_budget",
        "shared_provenance": diagnostics[0]["provenance"],
        "targets": targets,
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    output = Path(output_dir).expanduser().resolve()
    _write_immutable_json(output / "standalone_portfolio_targets.json", manifest)
    _write_csv(output / "standalone_portfolio_targets.csv", diagnostics)
    diagnostics_payload = {
        "schema_version": 1,
        "status": "complete" if seed_coverage_complete else "provisional",
        "expected_seeds": list(REQUIRED_SEEDS),
        "observed_seeds": list(observed_seeds),
        "missing_seeds": list(missing_seeds),
        "seed_coverage_complete": seed_coverage_complete,
        "target_manifest_hash": manifest["manifest_hash"],
        "runs": diagnostics,
    }
    diagnostics_payload["diagnostics_hash"] = stable_hash(diagnostics_payload)
    _write_json(output / "standalone_portfolio_diagnostics.json", diagnostics_payload)

    pyplot = _pyplot()
    figure, axes = pyplot.subplots(2, 2, figsize=(10, 7), sharex=True)
    for axis, width in zip(axes.flat, GRANULARITIES, strict=True):
        for seed in observed_seeds:
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


def freeze_references(
    run_dirs: Iterable[str | Path],
    output_dir: str | Path,
    *,
    granularity_profile: str = "quartile",
) -> dict[str, Any]:
    with _active_granularity_profile(granularity_profile):
        return _freeze_references_active(run_dirs, output_dir)


def _validate_elastic_run(
    run: Mapping[str, Any], *, role: str, budget: int, candidate_arm_id: str
) -> list[str]:
    config = run["config"]
    model = config.get("model", {})
    arm = _candidate_arm_spec(run)
    if arm is None:
        return ["elastic candidate has an unsupported comparison arm"]
    rejections = _base_rejections(
        run,
        role=role,
        budget=budget,
        candidate_arm_id=candidate_arm_id,
    )
    if (
        config.get("run", {}).get("model_family") != "nested"
        or config.get("run", {}).get("sampling_mode") != arm["sampling_mode"]
    ):
        rejections.append("elastic topology does not match the comparison arm")
    if list(model.get("granularities", [])) != list(GRANULARITIES):
        rejections.append("elastic run does not expose the four-width portfolio")
    if model.get("variant") != arm["model_variant"]:
        rejections.append("elastic model variant does not match the comparison arm")
    if arm["schema_version"] >= 4:
        if model.get("variant") != "slicing":
            rejections.append("schema-4 candidate is not slicing")
        if arm["sampling_mode"] == "nested-random" and model.get(
            "granularity_sampling_mode"
        ) not in {"global", "fixed_global", "adaptive_global"}:
            rejections.append("schema-4 candidate does not use a single global policy")
    elif arm["sampling_mode"] == "nested-random":
        if model.get("granularity_sampling_mode") != "global":
            rejections.append("elastic run is not uniform-global")
        if model.get("global_sampling_schedule") != "random_with_replacement":
            rejections.append("elastic run does not sample uniformly with replacement")
        if model.get("global_sampling_interval_steps") != 1:
            rejections.append("elastic run does not use H=1")
    return rejections


def _candidate_provenance_rejections(
    run: Mapping[str, Any],
    reference_provenance: Mapping[str, Any],
    *,
    candidate_arm_id: str,
) -> list[str]:
    """Compare provenance while honoring an arm's explicit diagnostic delta."""

    arm = _candidate_arm_spec(run)
    if arm is None:
        return ["candidate arm is unsupported"]
    candidate_provenance = _run_provenance(run)
    allowed = set(arm.get("allowed_reference_provenance_differences", set()))
    mismatches = {
        key
        for key in set(candidate_provenance) | set(reference_provenance)
        if candidate_provenance.get(key) != reference_provenance.get(key)
    }
    unexpected = mismatches - allowed
    rejections = []
    if unexpected:
        rejections.append(
            "candidate data/model provenance differs from references: "
            + ", ".join(sorted(unexpected))
        )
    if "model_variant" in allowed:
        if reference_provenance.get("model_variant") != "slicing":
            rejections.append(
                "concat diagnostic requires slicing standalone references"
            )
        if candidate_provenance.get("model_variant") != arm["model_variant"]:
            rejections.append(
                "concat diagnostic candidate does not use the concat model variant"
            )
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
    candidate_arm_id: str,
    budget_tokens: int,
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
    if candidate_arm_id in CANDIDATE_ARMS and CANDIDATE_ARMS[candidate_arm_id][
        "schema_version"
    ] >= 3 and (
        state.get("comparison_arm_id") != candidate_arm_id
        or state.get("elastic_budget_cap_tokens") != budget_tokens
    ):
        raise PortfolioAnalysisError(
            "Confirmation checkpoint comparison-arm provenance mismatch"
        )
    if not isinstance(checkpoint.get("model_state_dict"), Mapping):
        raise PortfolioAnalysisError("Confirmation checkpoint lacks exact model state")


def _validate_terminal_checkpoint(
    checkpoint_path: Path,
    *,
    run: Mapping[str, Any],
    target_hash: str,
    budget_tokens: int,
) -> None:
    try:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise PortfolioAnalysisError(
            f"Cannot load terminal candidate checkpoint: {checkpoint_path}"
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise PortfolioAnalysisError("Terminal candidate checkpoint is malformed")
    summary = run["summary"]
    expected_step = _integer(summary.get("steps_completed"), "completed step")
    if (
        checkpoint.get("checkpoint_status") != "latest"
        or _integer(checkpoint.get("step"), "terminal checkpoint step")
        != expected_step
        or _integer(checkpoint.get("tokens_seen"), "terminal checkpoint tokens")
        != budget_tokens
    ):
        raise PortfolioAnalysisError(
            "Terminal diagnostic checkpoint is not the completed arm boundary"
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
    candidate_arm_id = _candidate_arm_id(run)
    if candidate_arm_id in CANDIDATE_ARMS and CANDIDATE_ARMS[candidate_arm_id][
        "schema_version"
    ] >= 3 and (
        state.get("comparison_arm_id") != candidate_arm_id
        or state.get("elastic_budget_cap_tokens") != budget_tokens
    ):
        raise PortfolioAnalysisError(
            "Terminal checkpoint comparison-arm provenance mismatch"
        )
    if _bundle_contract(run) and (
        state.get("comparison_arm_id") != candidate_arm_id
        or state.get("elastic_budget_cap_tokens") != budget_tokens
        or state.get("candidate_policy_contract_hash")
        != candidate_policy_contract_hash(run["config"])
    ):
        raise PortfolioAnalysisError(
            "Terminal checkpoint schema-4 policy provenance mismatch"
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
    run: Mapping[str, Any], *, target_hash: str, budget_tokens: int
) -> tuple[Path, str, int]:
    summary = run["summary"]
    checkpoint = _resolve_checkpoint(
        run["run_dir"], summary.get("latest_checkpoint_path")
    )
    if checkpoint is None:
        raise PortfolioAnalysisError(
            "Completed elastic run lacks its terminal latest checkpoint"
        )
    _validate_terminal_checkpoint(
        checkpoint,
        run=run,
        target_hash=target_hash,
        budget_tokens=budget_tokens,
    )
    return (
        checkpoint,
        _sha256(checkpoint),
        _integer(summary.get("steps_completed"), "completed step"),
    )


def _portfolio_catchup_active(
    run_dirs: Iterable[str | Path],
    target_manifest_path: str | Path,
    output_dir: str | Path,
    candidate_arm: str | None = None,
) -> dict[str, Any]:
    targets = _load_hashed_manifest(target_manifest_path, "standalone targets")
    reference_budget_tokens = _integer(
        targets.get("reference_budget_tokens"), "target reference budget"
    )
    aggregate_reference_budget_tokens = _integer(
        targets.get("aggregate_reference_budget_tokens"),
        "target aggregate reference budget",
    )
    runs = [_load_run(path) for path in run_dirs]
    observed_arms = {_candidate_arm_id(run) for run in runs}
    if None in observed_arms or any(_candidate_arm_spec(run) is None for run in runs):
        raise PortfolioAnalysisError(
            f"Elastic candidates have an unsupported comparison arm: {observed_arms}"
        )
    if candidate_arm is None:
        if len(observed_arms) != 1:
            raise PortfolioAnalysisError(
                "Elastic candidates must all belong to one comparison arm"
            )
        candidate_arm_id = str(next(iter(observed_arms)))
    else:
        candidate_arm_id = str(candidate_arm)
        if observed_arms != {candidate_arm_id}:
            raise PortfolioAnalysisError(
                "Requested candidate arm differs from the saved run contracts"
            )
    arm = _candidate_arm_spec(runs[0])
    if arm is None:
        raise PortfolioAnalysisError("Elastic candidates have no arm contract")
    policy_hashes = {
        spec["policy_contract_hash"]
        for run in runs
        for spec in [_candidate_arm_spec(run)]
        if spec is not None and spec["schema_version"] >= 4
    }
    if len(policy_hashes) > 1:
        raise PortfolioAnalysisError("Elastic candidates have mismatched policy contracts")
    candidate_budget_tokens = int(arm["budget_tokens"])
    post_hoc_diagnostic = bool(arm["post_hoc_diagnostic"])
    by_seed: dict[int, Mapping[str, Any]] = {}
    for run in runs:
        seed = _seed(run)
        if seed in by_seed:
            raise PortfolioAnalysisError(f"Duplicate elastic candidate seed {seed}")
        by_seed[seed] = run
    unexpected_seeds = sorted(set(by_seed) - set(REQUIRED_SEEDS))
    if unexpected_seeds:
        raise PortfolioAnalysisError(
            "Elastic candidates contain out-of-contract seeds: "
            f"{unexpected_seeds}; expected a subset of {list(REQUIRED_SEEDS)}"
        )
    if not by_seed:
        raise PortfolioAnalysisError("At least one elastic candidate is required")
    observed_seeds = tuple(seed for seed in REQUIRED_SEEDS if seed in by_seed)
    missing_seeds = tuple(seed for seed in REQUIRED_SEEDS if seed not in by_seed)
    seed_coverage_complete = not missing_seeds

    reference_run_ids = {
        target["run_id"]
        for seed_targets in targets["targets"].values()
        for target in seed_targets.values()
    }
    seed_reports: list[dict[str, Any]] = []
    observations_by_seed: dict[int, list[dict[str, Any]]] = {}
    holdout_entries: list[dict[str, Any]] = []
    for seed in observed_seeds:
        run = by_seed[seed]
        rejections = _validate_elastic_run(
            run,
            role="elastic_candidate",
            budget=candidate_budget_tokens,
            candidate_arm_id=candidate_arm_id,
        )
        config = run["config"]
        contract = config["controlled_experiment"]["portfolio_catchup"]
        if contract.get("target_manifest_hash") != targets["manifest_hash"]:
            rejections.append("candidate target-manifest hash mismatch")
        if not math.isclose(
            _learning_rate(run), FIXED_LEARNING_RATE, rel_tol=0.0, abs_tol=1e-12
        ):
            rejections.append("candidate learning rate is not fixed at 0.008")
        rejections.extend(
            _candidate_provenance_rejections(
                run,
                targets.get("shared_provenance", {}),
                candidate_arm_id=candidate_arm_id,
            )
        )
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
        if int(arm["schema_version"]) >= 3 and (
            state.get("comparison_arm_id") != candidate_arm_id
            or state.get("elastic_budget_cap_tokens") != candidate_budget_tokens
        ):
            raise PortfolioAnalysisError(
                "Candidate online catch-up state has the wrong comparison arm"
            )
        if int(arm["schema_version"]) >= 4 and (
            state.get("candidate_policy_contract_hash")
            != candidate_policy_contract_hash(config)
        ):
            raise PortfolioAnalysisError(
                "Candidate online catch-up state has the wrong policy contract"
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
                candidate_arm_id=candidate_arm_id,
                budget_tokens=candidate_budget_tokens,
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
            "t_star_over_B": confirmation_tokens / reference_budget_tokens
            if caught_up
            else None,
            "t_star_over_4B": confirmation_tokens / aggregate_reference_budget_tokens
            if caught_up
            else None,
            "required_savings_fraction": 1.0
            - confirmation_tokens / aggregate_reference_budget_tokens
            if caught_up
            else None,
            "realized_full_run_tokens": candidate_budget_tokens,
            "realized_full_run_spend_over_4B": candidate_budget_tokens
            / aggregate_reference_budget_tokens,
            "final_per_width_deficits": observations[-1]["widths"],
            "validation_observations": observations,
        }
        seed_reports.append(seed_report)
    for seed in observed_seeds:
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

    all_observed_caught_up = all(row["caught_up"] for row in seed_reports)
    arm_catchup_confirmed = seed_coverage_complete and all_observed_caught_up
    claim_eligible = arm_catchup_confirmed and not post_hoc_diagnostic
    if all_observed_caught_up:
        selection_mode = (
            "portfolio_confirmation_diagnostic"
            if post_hoc_diagnostic
            else "portfolio_confirmation"
        )
    else:
        selection_mode = (
            "terminal_candidate_budget_censored"
            if post_hoc_diagnostic
            else "terminal_3B_censored"
        )
    for seed_report in seed_reports:
        seed = int(seed_report["seed"])
        run = by_seed[seed]
        if all_observed_caught_up:
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
                    budget_tokens=candidate_budget_tokens,
                )
            )
            checkpoint_tokens = candidate_budget_tokens
            checkpoint_selection = (
                "terminal_candidate_budget"
                if post_hoc_diagnostic
                else "terminal_3B"
            )
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
    observed_seed_budget_summary = None
    if all_observed_caught_up:
        required_tokens = max(confirmed_tokens)
        observed_seed_budget_summary = {
            "scope": (
                "complete_required_seed_matrix"
                if seed_coverage_complete
                else "observed_seed_subset"
            ),
            "observed_seed_required_tokens": required_tokens,
            "observed_seed_t_star_over_B": required_tokens
            / reference_budget_tokens,
            "observed_seed_t_star_over_4B": required_tokens
            / aggregate_reference_budget_tokens,
            "observed_seed_required_savings_fraction": 1.0
            - required_tokens / aggregate_reference_budget_tokens,
            "mean_confirmation_tokens": statistics.fmean(confirmed_tokens),
            "median_confirmation_tokens": statistics.median(confirmed_tokens),
        }
    budget_summary = None
    if arm_catchup_confirmed:
        required_tokens = max(confirmed_tokens)
        budget_summary = {
            "cross_seed_required_tokens": required_tokens,
            "cross_seed_t_star_over_B": required_tokens / reference_budget_tokens,
            "cross_seed_t_star_over_4B": required_tokens
            / aggregate_reference_budget_tokens,
            "cross_seed_required_savings_fraction": 1.0
            - required_tokens / aggregate_reference_budget_tokens,
            "mean_confirmation_tokens": statistics.fmean(confirmed_tokens),
            "median_confirmation_tokens": statistics.median(confirmed_tokens),
        }
    if seed_coverage_complete:
        status = "portfolio_catchup_confirmed" if arm_catchup_confirmed else "censored"
    else:
        status = (
            "provisional_seed_subset_confirmed"
            if all_observed_caught_up
            else "provisional_seed_subset_censored"
        )
    if claim_eligible:
        holdout_status = "ready_confirmatory"
    elif not seed_coverage_complete:
        holdout_status = "ready_provisional"
    elif post_hoc_diagnostic:
        holdout_status = "ready_diagnostic"
    else:
        holdout_status = "ready_diagnostic_terminal_3B"
    report = {
        "schema_version": 3 if post_hoc_diagnostic else 2,
        "analysis": "tinystories_instruct_four_granularity_portfolio_catchup",
        "status": status,
        "general_portfolio_catchup_claim": claim_eligible,
        "arm_catchup_confirmed": arm_catchup_confirmed,
        "observed_seed_catchup_confirmed": all_observed_caught_up,
        "provisional_analysis": not seed_coverage_complete,
        "expected_seeds": list(REQUIRED_SEEDS),
        "observed_seeds": list(observed_seeds),
        "missing_seeds": list(missing_seeds),
        "expected_seed_count": len(REQUIRED_SEEDS),
        "observed_seed_count": len(observed_seeds),
        "seed_coverage_complete": seed_coverage_complete,
        "post_hoc_diagnostic": post_hoc_diagnostic,
        "comparison_arm_id": candidate_arm_id,
        "candidate_policy_contract": arm.get("policy_contract"),
        "candidate_policy_contract_hash": arm.get("policy_contract_hash"),
        "candidate_budget_multiplier": arm.get("policy_contract", {}).get(
            "candidate_budget_multiplier"
        ),
        "comparison_group_id": COMPARISON_GROUP_ID,
        "granularity_profile": _profile_name_for_group(COMPARISON_GROUP_ID),
        "granularities": list(GRANULARITIES),
        "reference_budget_tokens": reference_budget_tokens,
        "reference_budget_multiplier": targets.get("reference_budget_multiplier", 1),
        "aggregate_reference_budget_tokens": aggregate_reference_budget_tokens,
        "elastic_budget_cap_tokens": candidate_budget_tokens,
        "realized_full_run_spend_over_4B": candidate_budget_tokens
        / aggregate_reference_budget_tokens,
        "subnetwork_gradient_evaluations_per_optimizer_step": (
            len(GRANULARITIES)
            if arm["sampling_mode"] == "nested-all"
            else 1
        ),
        "realized_subnetwork_target_tokens": (
            candidate_budget_tokens * len(GRANULARITIES)
            if arm["sampling_mode"] == "nested-all"
            else candidate_budget_tokens
        ),
        "perplexity_tolerance": PERPLEXITY_TOLERANCE,
        "loss_tolerance": LOSS_TOLERANCE,
        "required_consecutive_evaluations": REQUIRED_STREAK,
        "target_manifest_path": str(Path(target_manifest_path).expanduser().resolve()),
        "target_manifest_hash": targets["manifest_hash"],
        "learning_rate": FIXED_LEARNING_RATE,
        "reference_model_variant": targets["shared_provenance"]["model_variant"],
        "candidate_model_variant": arm["model_variant"],
        "allowed_reference_provenance_differences": sorted(
            arm.get("allowed_reference_provenance_differences", set())
        ),
        "optimizer_recipe_policy": "same_fixed_recipe_across_roles",
        "seeds": seed_reports,
        "budget_summary": budget_summary,
        "observed_seed_budget_summary": observed_seed_budget_summary,
        "final_holdout_selection_status": holdout_status,
        "final_holdout_selection_mode": selection_mode,
        "final_holdout_claim_eligible": claim_eligible,
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
        "schema_version": 3 if post_hoc_diagnostic else 2,
        "analysis": "portfolio_final_holdout_selection",
        "status": (
            holdout_status
        ),
        "selection_mode": selection_mode,
        "claim_eligible": claim_eligible,
        "provisional_analysis": not seed_coverage_complete,
        "expected_seeds": list(REQUIRED_SEEDS),
        "observed_seeds": list(observed_seeds),
        "missing_seeds": list(missing_seeds),
        "expected_seed_count": len(REQUIRED_SEEDS),
        "observed_seed_count": len(observed_seeds),
        "seed_coverage_complete": seed_coverage_complete,
        "interpretation": (
            "confirmation_checkpoint_generalization"
            if claim_eligible
            else (
                "provisional_seed_subset_only_no_general_claim"
                if not seed_coverage_complete
                else (
                    "post_hoc_diagnostic_only_no_general_claim"
                    if post_hoc_diagnostic
                    else "terminal_3B_diagnostic_only_no_catchup_claim"
                )
            )
        ),
        "comparison_arm_id": candidate_arm_id,
        "candidate_policy_contract": arm.get("policy_contract"),
        "candidate_policy_contract_hash": arm.get("policy_contract_hash"),
        "candidate_budget_multiplier": contract.get("candidate_budget_multiplier"),
        "candidate_budget_tokens": candidate_budget_tokens,
        "comparison_group_id": COMPARISON_GROUP_ID,
        "granularity_profile": _profile_name_for_group(COMPARISON_GROUP_ID),
        "granularities": list(GRANULARITIES),
        "target_manifest_hash": targets["manifest_hash"],
        "shared_corpus_hash": targets["shared_provenance"]["corpus_hash"],
        "final_holdout_manifest_hash": targets["shared_provenance"][
            "final_holdout_manifest_hash"
        ],
        "required_checkpoint_count": len(observed_seeds)
        * (len(GRANULARITIES) + 1),
        "expected_full_checkpoint_count": len(REQUIRED_SEEDS)
        * (len(GRANULARITIES) + 1),
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
    _write_json(output / "final_holdout_selection_manifest.json", holdout_manifest)

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


def portfolio_catchup(
    run_dirs: Iterable[str | Path],
    target_manifest_path: str | Path,
    output_dir: str | Path,
    candidate_arm: str | None = None,
) -> dict[str, Any]:
    targets = _load_hashed_manifest(target_manifest_path, "standalone targets")
    profile_name = _profile_name_for_group(targets.get("comparison_group_id"))
    profile = GRANULARITY_PROFILES[profile_name]
    if targets.get("granularities") != list(profile["granularities"]):
        raise PortfolioAnalysisError(
            "Standalone target manifest granularity profile is inconsistent"
        )
    with _active_granularity_profile(profile_name):
        return _portfolio_catchup_active(
            run_dirs,
            target_manifest_path,
            output_dir,
            candidate_arm=candidate_arm,
        )


def portfolio_catchup_bundle(
    run_dirs: Iterable[str | Path],
    target_manifest_path: str | Path,
    output_dir: str | Path,
    candidate_arm: str | None = None,
) -> dict[str, Any]:
    """Analyze every completed candidate arm discovered in one bundle lane.

    A candidate arm is an immutable property of its resolved config, not an
    analysis-time input.  ``candidate_arm`` remains available only as a narrow
    compatibility filter for callers that deliberately select one saved arm.
    """

    loaded_runs = [_load_run(path) for path in run_dirs]
    completed_candidates = [
        run
        for run in loaded_runs
        if _controlled_contract(run)[0] == "elastic_candidate"
        and run["summary"].get("status") == "completed"
    ]
    if candidate_arm is not None:
        completed_candidates = [
            run
            for run in completed_candidates
            if _candidate_arm_id(run) == str(candidate_arm)
        ]
    if not completed_candidates:
        qualifier = (
            f" for comparison arm {candidate_arm!r}" if candidate_arm is not None else ""
        )
        raise PortfolioAnalysisError(
            "No completed elastic candidate runs were discovered" + qualifier
        )

    invalid_runs = [
        str(run["run_dir"])
        for run in completed_candidates
        if _candidate_arm_id(run) is None or _candidate_arm_spec(run) is None
    ]
    if invalid_runs:
        raise PortfolioAnalysisError(
            "Completed elastic candidates lack a supported saved comparison-arm "
            "contract; they cannot be assigned to a portfolio variant after the "
            f"fact: {invalid_runs}"
        )

    by_arm: dict[str, list[Mapping[str, Any]]] = {}
    for run in completed_candidates:
        arm_id = _candidate_arm_id(run)
        assert arm_id is not None  # Checked above; keeps the grouping typed.
        by_arm.setdefault(arm_id, []).append(run)

    output = Path(output_dir).expanduser().resolve()
    arm_reports: list[dict[str, Any]] = []
    for arm_id, runs_for_arm in sorted(by_arm.items()):
        arm_output = output / arm_id
        report = portfolio_catchup(
            [run["run_dir"] for run in runs_for_arm],
            target_manifest_path,
            arm_output,
            candidate_arm=arm_id,
        )
        arm_reports.append(
            {
                "comparison_arm_id": arm_id,
                "status": report["status"],
                "observed_seeds": report["observed_seeds"],
                "analysis_dir": str(arm_output),
                "report_path": str(arm_output / "portfolio_catchup_report.json"),
                "final_holdout_selection_manifest": str(
                    arm_output / "final_holdout_selection_manifest.json"
                ),
            }
        )
    bundle_report = {
        "schema_version": 1,
        "analysis": "tinystories_instruct_portfolio_bundle_candidates",
        "status": "completed_candidate_arms_analyzed",
        "target_manifest_path": str(Path(target_manifest_path).expanduser().resolve()),
        "completed_candidate_run_count": len(completed_candidates),
        "discovered_run_count": len(loaded_runs),
        "candidate_arms": arm_reports,
    }
    bundle_report["report_hash"] = stable_hash(bundle_report)
    _write_json(output / "portfolio_candidate_bundle_report.json", bundle_report)
    return bundle_report


def _final_holdout_active(
    selection_manifest_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    selection = _load_hashed_manifest(selection_manifest_path, "holdout selection")
    expected_seeds = tuple(selection.get("expected_seeds", REQUIRED_SEEDS))
    observed_seeds = tuple(selection.get("observed_seeds", expected_seeds))
    if expected_seeds != REQUIRED_SEEDS:
        raise PortfolioAnalysisError(
            f"Final holdout expected seeds must be {list(REQUIRED_SEEDS)}"
        )
    if (
        not observed_seeds
        or len(set(observed_seeds)) != len(observed_seeds)
        or any(seed not in REQUIRED_SEEDS for seed in observed_seeds)
        or observed_seeds
        != tuple(seed for seed in REQUIRED_SEEDS if seed in set(observed_seeds))
    ):
        raise PortfolioAnalysisError(
            "Final holdout observed seeds must be a non-empty ordered subset of "
            f"{list(REQUIRED_SEEDS)}"
        )
    missing_seeds = tuple(seed for seed in REQUIRED_SEEDS if seed not in observed_seeds)
    seed_coverage_complete = not missing_seeds
    if (
        selection.get("missing_seeds", list(missing_seeds)) != list(missing_seeds)
        or selection.get("seed_coverage_complete", seed_coverage_complete)
        is not seed_coverage_complete
        or selection.get("observed_seed_count", len(observed_seeds))
        != len(observed_seeds)
    ):
        raise PortfolioAnalysisError(
            "Final-holdout selection seed-coverage metadata is inconsistent"
        )
    selection_mode = selection.get("selection_mode", "portfolio_confirmation")
    confirmation_modes = {
        "portfolio_confirmation",
        "portfolio_confirmation_diagnostic",
    }
    terminal_modes = {
        "terminal_3B_censored",
        "terminal_candidate_budget_censored",
    }
    if selection_mode not in confirmation_modes | terminal_modes:
        raise PortfolioAnalysisError("Final-holdout selection mode is invalid")
    claim_eligible = selection.get(
        "claim_eligible", selection_mode == "portfolio_confirmation"
    )
    if not isinstance(claim_eligible, bool) or (
        claim_eligible and selection_mode != "portfolio_confirmation"
    ):
        raise PortfolioAnalysisError(
            "Final-holdout claim eligibility contradicts its selection mode"
        )
    if claim_eligible and not seed_coverage_complete:
        raise PortfolioAnalysisError(
            "A partial-seed final holdout cannot be eligible for a general claim"
        )
    entries = selection.get("entries")
    required_checkpoint_count = len(observed_seeds) * (len(GRANULARITIES) + 1)
    if (
        not isinstance(entries, list)
        or len(entries) != required_checkpoint_count
        or selection.get("required_checkpoint_count", required_checkpoint_count)
        != required_checkpoint_count
    ):
        raise PortfolioAnalysisError(
            "Final holdout requires exactly five selected checkpoints per "
            f"observed seed ({required_checkpoint_count} total)"
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
                if selection_mode in confirmation_modes
                else (
                    "terminal_candidate_budget"
                    if selection_mode == "terminal_candidate_budget_censored"
                    else "terminal_3B"
                )
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
    for seed in observed_seeds:
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
    diagnostic_equivalence = all_pass if not claim_eligible else None
    if not seed_coverage_complete:
        status = (
            "provisional_seed_subset_equivalent"
            if all_pass
            else "provisional_seed_subset_not_equivalent"
        )
    elif claim_eligible:
        status = "portfolio_equivalent" if all_pass else "portfolio_not_equivalent"
    elif selection_mode == "terminal_3B_censored":
        status = (
            "diagnostic_terminal_3B_equivalent"
            if all_pass
            else "diagnostic_terminal_3B_not_equivalent"
        )
    else:
        status = (
            "diagnostic_arm_equivalent"
            if all_pass
            else "diagnostic_arm_not_equivalent"
        )
    report = {
        "schema_version": 2,
        "analysis": "tinystories_instruct_portfolio_final_holdout",
        "status": status,
        "selection_mode": selection_mode,
        "claim_eligible": claim_eligible,
        "provisional_analysis": not seed_coverage_complete,
        "expected_seeds": list(REQUIRED_SEEDS),
        "observed_seeds": list(observed_seeds),
        "missing_seeds": list(missing_seeds),
        "expected_seed_count": len(REQUIRED_SEEDS),
        "observed_seed_count": len(observed_seeds),
        "seed_coverage_complete": seed_coverage_complete,
        "comparison_arm_id": selection.get("comparison_arm_id", "uniform_h1_3b"),
        "comparison_group_id": COMPARISON_GROUP_ID,
        "granularity_profile": _profile_name_for_group(COMPARISON_GROUP_ID),
        "granularities": list(GRANULARITIES),
        "all_pairs_within_tolerance": all_pass,
        "provisional_seed_subset_equivalence": (
            all_pass if not seed_coverage_complete else None
        ),
        "diagnostic_terminal_3B_equivalence": (
            diagnostic_equivalence
            if selection_mode == "terminal_3B_censored"
            else None
        ),
        "diagnostic_arm_equivalence": diagnostic_equivalence,
        "general_portfolio_equivalence_claim": general_claim,
        "perplexity_tolerance": PERPLEXITY_TOLERANCE,
        "required_checkpoint_count": required_checkpoint_count,
        "expected_full_checkpoint_count": len(REQUIRED_SEEDS)
        * (len(GRANULARITIES) + 1),
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


def final_holdout(
    selection_manifest_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    selection = _load_hashed_manifest(selection_manifest_path, "holdout selection")
    profile_name = _profile_name_for_group(selection.get("comparison_group_id"))
    profile = GRANULARITY_PROFILES[profile_name]
    saved_granularities = selection.get(
        "granularities", list(profile["granularities"])
    )
    if saved_granularities != list(profile["granularities"]):
        raise PortfolioAnalysisError(
            "Final-holdout selection granularity profile is inconsistent"
        )
    with _active_granularity_profile(profile_name):
        return _final_holdout_active(selection_manifest_path, output_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        (
            "freeze-references",
            "freeze complete four-width panels for one or more reference seeds",
        ),
        (
            "portfolio-catchup",
            "verify one or more fresh elastic candidates (three for a general claim)",
        ),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--run-dir", action="append", default=[])
        command.add_argument("--runs-root", action="append", default=[])
        command.add_argument("--output-dir", required=True)
        if name == "freeze-references":
            command.add_argument(
                "--granularity-profile",
                choices=sorted(GRANULARITY_PROFILES),
                default="quartile",
                help="standalone matrix geometry to freeze",
            )
        else:
            command.add_argument("--target-manifest", required=True)
            command.add_argument(
                "--candidate-arm",
                help=(
                    "expected elastic arm; otherwise inferred from the saved "
                    "run contracts"
                ),
            )
    holdout = commands.add_parser(
        "final-holdout",
        help="verify all manifest-selected holdout results",
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
            result = freeze_references(
                run_dirs,
                args.output_dir,
                granularity_profile=args.granularity_profile,
            )
        else:
            result = portfolio_catchup_bundle(
                run_dirs,
                args.target_manifest,
                args.output_dir,
                candidate_arm=args.candidate_arm,
            )
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
