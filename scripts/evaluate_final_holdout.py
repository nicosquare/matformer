"""Evaluate a completed run's untouched final holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed final holdout after training completes.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Completed run directory containing config and summary artifacts.",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=None,
        help=(
            "Portfolio final-holdout selection manifest. Evaluates four paired "
            "standalone checkpoints and one explicit elastic confirmation or "
            "terminal checkpoint per observed seed."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Explicit checkpoint belonging to the run. Required when ordinary "
            "validation did not select a best checkpoint."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device such as cpu or cuda; defaults to CUDA when available.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Return successfully without reevaluating when the run already has a "
            "valid, hashed final_holdout_results.json artifact."
        ),
    )
    args = parser.parse_args(argv)
    if (args.run_dir is None) == (args.selection_manifest is None):
        parser.error("provide exactly one of --run-dir or --selection-manifest")
    if args.selection_manifest is not None and args.checkpoint is not None:
        parser.error("--checkpoint cannot be combined with --selection-manifest")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from src.evaluation.final_holdout import (
        FinalHoldoutError,
        evaluate_final_holdout,
        resolve_final_holdout_checkpoint,
        resolve_existing_final_holdout_result,
    )

    try:
        if args.selection_manifest is not None:
            result_paths = []
            for entry in _portfolio_selection_entries(args.selection_manifest):
                run_dir = Path(entry["run_dir"])
                requested_checkpoint = Path(entry["checkpoint_path"])
                checkpoint_sha256 = _sha256(requested_checkpoint)
                if checkpoint_sha256 != entry["checkpoint_sha256"]:
                    raise FinalHoldoutError(
                        "Portfolio selection checkpoint is missing or stale: "
                        f"{requested_checkpoint}"
                    )
                explicit = entry["checkpoint_selection"] in {
                    "portfolio_confirmation",
                    "terminal_3B",
                    "terminal_candidate_budget",
                }
                if not explicit:
                    ordinary_checkpoint = resolve_final_holdout_checkpoint(run_dir)
                    if ordinary_checkpoint.resolve() != requested_checkpoint.resolve():
                        raise FinalHoldoutError(
                            "Portfolio standalone selection is not the run's "
                            "ordinary-validation best checkpoint"
                        )
                if args.skip_existing:
                    existing = resolve_existing_final_holdout_result(
                        run_dir,
                        checkpoint_path=requested_checkpoint,
                    )
                    if existing is not None:
                        result_paths.append(existing["result_path"])
                        continue
                result = evaluate_final_holdout(
                    run_dir,
                    checkpoint_path=requested_checkpoint if explicit else None,
                    device=args.device,
                )
                if Path(result["checkpoint_path"]).resolve() != requested_checkpoint.resolve():
                    raise FinalHoldoutError(
                        "Final holdout evaluated a different portfolio checkpoint"
                    )
                result_paths.append(result["result_path"])
            for result_path in result_paths:
                print(result_path)
            return 0
        if args.skip_existing:
            existing = resolve_existing_final_holdout_result(
                args.run_dir,
                checkpoint_path=args.checkpoint,
            )
            if existing is not None:
                print(existing["result_path"])
                return 0
        result = evaluate_final_holdout(
            args.run_dir,
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
    except (FinalHoldoutError, RuntimeError) as error:
        print(f"Final holdout evaluation failed: {error}", file=sys.stderr)
        return 2

    print(result["result_path"])
    return 0


def _portfolio_selection_entries(path: Path) -> list[dict]:
    from src.utils.reproducibility import stable_hash

    try:
        manifest = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read portfolio selection manifest: {path}") from error
    if not isinstance(manifest, dict):
        raise RuntimeError("Portfolio selection manifest must be a mapping")
    body = dict(manifest)
    saved_hash = body.pop("manifest_hash", None)
    if saved_hash != stable_hash(body):
        raise RuntimeError("Portfolio selection manifest hash mismatch")
    required_seeds = (42, 43, 44)
    granularity_profiles = {
        "tinystories_instruct_portfolio_catchup_v1": (
            "g250",
            "g500",
            "g750",
            "g1000",
        ),
        "tinystories_instruct_portfolio_catchup_matformer_granularities_v1": (
            "g125",
            "g250",
            "g500",
            "g1000",
        ),
    }
    comparison_group_id = manifest.get(
        "comparison_group_id", "tinystories_instruct_portfolio_catchup_v1"
    )
    expected_granularities = granularity_profiles.get(str(comparison_group_id))
    if expected_granularities is None:
        raise RuntimeError("Portfolio selection granularity profile is unsupported")
    granularities = tuple(
        manifest.get("granularities", expected_granularities)
    )
    if granularities != expected_granularities:
        raise RuntimeError("Portfolio selection granularity profile is inconsistent")
    expected_seeds = tuple(manifest.get("expected_seeds", required_seeds))
    observed_seeds = tuple(manifest.get("observed_seeds", expected_seeds))
    if expected_seeds != required_seeds:
        raise RuntimeError("Portfolio selection expected-seed contract is invalid")
    if (
        not observed_seeds
        or len(set(observed_seeds)) != len(observed_seeds)
        or any(seed not in required_seeds for seed in observed_seeds)
        or observed_seeds
        != tuple(seed for seed in required_seeds if seed in set(observed_seeds))
    ):
        raise RuntimeError(
            "Portfolio selection observed seeds must be a non-empty ordered subset"
        )
    missing_seeds = tuple(seed for seed in required_seeds if seed not in observed_seeds)
    seed_coverage_complete = not missing_seeds
    if (
        manifest.get("missing_seeds", list(missing_seeds)) != list(missing_seeds)
        or manifest.get("seed_coverage_complete", seed_coverage_complete)
        is not seed_coverage_complete
    ):
        raise RuntimeError("Portfolio selection seed-coverage metadata is inconsistent")
    required_checkpoint_count = len(observed_seeds) * (len(granularities) + 1)
    entries = manifest.get("entries")
    if (
        not isinstance(entries, list)
        or len(entries) != required_checkpoint_count
        or manifest.get("required_checkpoint_count", required_checkpoint_count)
        != required_checkpoint_count
    ):
        raise RuntimeError(
            "Portfolio selection manifest must list exactly five checkpoints per "
            f"observed seed ({required_checkpoint_count} total)"
        )
    normalized = []
    selection_mode = manifest.get("selection_mode", "portfolio_confirmation")
    confirmation_modes = {
        "portfolio_confirmation",
        "portfolio_confirmation_diagnostic",
    }
    terminal_modes = {
        "terminal_3B_censored",
        "terminal_candidate_budget_censored",
    }
    if selection_mode not in confirmation_modes | terminal_modes:
        raise RuntimeError("Portfolio selection manifest mode is invalid")
    claim_eligible = manifest.get(
        "claim_eligible", selection_mode == "portfolio_confirmation"
    )
    if not isinstance(claim_eligible, bool) or (
        claim_eligible and selection_mode != "portfolio_confirmation"
    ):
        raise RuntimeError(
            "Portfolio selection claim eligibility contradicts its mode"
        )
    if claim_eligible and not seed_coverage_complete:
        raise RuntimeError(
            "A partial-seed portfolio selection cannot support a general claim"
        )
    role_counts = {"standalone_reference": 0, "elastic_candidate": 0}
    reference_keys: set[tuple[int, str]] = set()
    elastic_seeds: set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("Portfolio selection entry must be a mapping")
        required = {
            "run_dir",
            "checkpoint_path",
            "checkpoint_sha256",
            "checkpoint_selection",
        }
        if not required.issubset(entry):
            raise RuntimeError("Portfolio selection entry is incomplete")
        role = entry.get("comparison_role")
        if role not in role_counts:
            raise RuntimeError("Portfolio selection entry role is invalid")
        try:
            seed = int(entry["seed"])
            widths = tuple(str(width) for width in entry["granularities"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Portfolio selection seed or widths are invalid") from error
        if seed not in observed_seeds:
            raise RuntimeError("Portfolio selection entry has an undeclared seed")
        role_counts[role] += 1
        if role == "standalone_reference" and len(widths) == 1:
            key = (seed, widths[0])
            if key in reference_keys:
                raise RuntimeError("Duplicate standalone portfolio selection")
            reference_keys.add(key)
        elif role == "elastic_candidate" and widths == granularities:
            if seed in elastic_seeds:
                raise RuntimeError("Duplicate elastic portfolio selection")
            elastic_seeds.add(seed)
        else:
            raise RuntimeError("Portfolio selection entry role or widths are invalid")
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
        if entry["checkpoint_selection"] != expected_selection:
            raise RuntimeError(
                "Portfolio checkpoint selection contradicts the manifest mode"
            )
        if (
            expected_selection == "terminal_3B"
            and entry.get("checkpoint_tokens") != 2_141_356_032
        ):
            raise RuntimeError(
                "Portfolio terminal checkpoint is not at the declared 3B cap"
            )
        if (
            expected_selection == "terminal_candidate_budget"
            and entry.get("checkpoint_tokens")
            != manifest.get("candidate_budget_tokens")
        ):
            raise RuntimeError(
                "Portfolio terminal checkpoint is not at the declared arm cap"
            )
        normalized.append(entry)
    expected_reference_keys = {
        (seed, width) for seed in observed_seeds for width in granularities
    }
    if (
        role_counts
        != {
            "standalone_reference": len(observed_seeds) * len(granularities),
            "elastic_candidate": len(observed_seeds),
        }
        or reference_keys != expected_reference_keys
        or elastic_seeds != set(observed_seeds)
    ):
        raise RuntimeError(
            "Portfolio selection manifest must contain four standalone and one "
            "elastic entry per observed seed"
        )
    return normalized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
