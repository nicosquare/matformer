#!/usr/bin/env python3
"""Freeze and report the six-run TinyStories optimizer-state comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.reporting_impl import (
    build_optimizer_state_comparison_report,
    validate_optimizer_state_pairs,
)
from src.evaluation.reporting_io import (
    OptimizerStateReportingError,
    load_optimizer_state_run,
)
from src.utils.metrics import write_json_artifact
from src.utils.reproducibility import stable_hash


MANIFEST_NAME = "optimizer_state_manifest.json"
REPORT_JSON_NAME = "optimizer_state_comparison.json"
REPORT_CSV_NAME = "optimizer_state_comparison.csv"


class OptimizerStateAnalysisError(ValueError):
    """Raised when a paired optimizer-state artifact cannot be frozen/reported."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OptimizerStateAnalysisError(f"Cannot read JSON artifact: {resolved}") from error
    if not isinstance(value, dict):
        raise OptimizerStateAnalysisError(f"JSON artifact must be a mapping: {resolved}")
    return value


def freeze_manifest(
    *,
    phase: str,
    run_dirs: Iterable[str | Path],
    output_dir: str | Path,
) -> Path:
    explicit_dirs = [Path(path).expanduser().resolve() for path in run_dirs]
    if len(explicit_dirs) != 6:
        raise OptimizerStateAnalysisError("Freeze requires exactly six explicit run directories")
    try:
        runs = validate_optimizer_state_pairs(
            [load_optimizer_state_run(path) for path in explicit_dirs], phase=phase
        )
    except (OptimizerStateReportingError, ValueError, KeyError, TypeError) as error:
        raise OptimizerStateAnalysisError(str(error)) from error

    entries = []
    for run in runs:
        config_path = run.run_dir / "config.json"
        summary_path = run.run_dir / "run_summary.json"
        metrics_path = run.run_dir / "metrics.csv"
        entries.append(
            {
                "run_id": run.run_id,
                "seed": run.seed,
                "state_scope": run.state_scope,
                "run_dir": str(run.run_dir),
                "paired_control_signature": run.paired_control_signature,
                "ordered_widths": list(run.ordered_widths),
                "config_path": str(config_path),
                "config_sha256": _sha256(config_path),
                "summary_path": str(summary_path),
                "summary_sha256": _sha256(summary_path),
                "metrics_path": str(metrics_path),
                "metrics_sha256": _sha256(metrics_path),
                "terminal_checkpoint_path": str(run.checkpoint_path),
                "terminal_checkpoint_sha256": run.checkpoint_sha256,
                "terminal_checkpoint_bytes": run.checkpoint_path.stat().st_size,
            }
        )
    holdout_opened_values = {
        bool(
            run.config.get("controlled_experiment", {}).get(
                "holdout_opened_during_pilot", False
            )
        )
        for run in runs
    }
    if len(holdout_opened_values) != 1:
        raise OptimizerStateAnalysisError(
            "Runs disagree about whether the pilot holdout was opened"
        )
    manifest = {
        "schema_version": 1,
        "phase": phase,
        "endpoint_definition": "terminal fixed-budget checkpoint",
        "required_seeds": [42, 43, 44],
        "required_scopes": ["shared", "per_granularity"],
        "holdout_opened_during_pilot": holdout_opened_values.pop(),
        "runs": entries,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    destination = Path(output_dir).expanduser().resolve() / MANIFEST_NAME
    if destination.exists():
        existing = _read_json(destination)
        if existing != manifest:
            raise OptimizerStateAnalysisError(
                f"Immutable manifest already exists with different content: {destination}"
            )
        return destination
    written = write_json_artifact(destination, manifest)
    if written is None:
        raise OptimizerStateAnalysisError("Manifest was not written")
    return written


def _validate_manifest_sources(manifest: Mapping[str, Any]) -> list[Any]:
    stored_hash = manifest.get("manifest_hash")
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if stored_hash != stable_hash(unhashed):
        raise OptimizerStateAnalysisError("Frozen manifest hash is invalid")
    entries = manifest.get("runs")
    if not isinstance(entries, list) or len(entries) != 6:
        raise OptimizerStateAnalysisError("Frozen manifest must contain exactly six runs")
    runs = []
    for entry in entries:
        for path_field, hash_field in (
            ("config_path", "config_sha256"),
            ("summary_path", "summary_sha256"),
            ("metrics_path", "metrics_sha256"),
            ("terminal_checkpoint_path", "terminal_checkpoint_sha256"),
        ):
            path = Path(str(entry[path_field])).expanduser().resolve()
            if not path.is_file() or _sha256(path) != entry[hash_field]:
                raise OptimizerStateAnalysisError(
                    f"Frozen artifact changed or is missing: {path}"
                )
        run = load_optimizer_state_run(entry["run_dir"])
        if (
            run.run_id != entry["run_id"]
            or run.seed != int(entry["seed"])
            or run.state_scope != entry["state_scope"]
        ):
            raise OptimizerStateAnalysisError(
                f"Frozen run identity changed: {entry['run_dir']}"
            )
        runs.append(run)
    try:
        return list(
            validate_optimizer_state_pairs(runs, phase=str(manifest["phase"]))
        )
    except ValueError as error:
        raise OptimizerStateAnalysisError(str(error)) from error


def _report_csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    deltas = {int(row["seed"]): row for row in report["paired_deltas"]}
    rows = []
    for outcome in report["outcomes"]:
        seed = int(outcome["seed"])
        for width in outcome["ordered_widths"]:
            endpoint = outcome["per_width_outcomes"][width]
            rows.append(
                {
                    "phase": report["phase"],
                    "evidence_label": report["evidence_label"],
                    "matched_compute_claim": report["matched_compute_claim"],
                    "seed": seed,
                    "state_scope": outcome["state_scope"],
                    "width": width,
                    "loss": endpoint["loss"],
                    "perplexity": endpoint["perplexity"],
                    "uniform_mean_loss": outcome["uniform_mean_loss"],
                    "worst_width_loss": outcome["worst_width_loss"],
                    "paired_uniform_mean_loss_delta": deltas[seed][
                        "uniform_mean_loss_delta"
                    ],
                    "paired_width_loss_delta": deltas[seed][
                        "per_width_loss_deltas"
                    ][width],
                    **outcome["resources"],
                }
            )
    return rows


def write_report(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    manifest = _read_json(manifest_path)
    runs = _validate_manifest_sources(manifest)
    try:
        report = build_optimizer_state_comparison_report(
            runs,
            phase=str(manifest["phase"]),
            holdout_opened_during_pilot=bool(
                manifest.get("holdout_opened_during_pilot", False)
            ),
        )
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise OptimizerStateAnalysisError(str(error)) from error
    report["manifest_path"] = str(Path(manifest_path).expanduser().resolve())
    report["manifest_hash"] = manifest["manifest_hash"]
    root = Path(output_dir).expanduser().resolve()
    json_path = root / REPORT_JSON_NAME
    written = write_json_artifact(json_path, report)
    if written is None:
        raise OptimizerStateAnalysisError("JSON report was not written")
    csv_path = root / REPORT_CSV_NAME
    rows = _report_csv_rows(report)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    return written, csv_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--phase", required=True, choices=("pilot", "confirmation"))
    freeze.add_argument("--run-dir", action="append", required=True)
    freeze.add_argument("--output-dir", required=True)
    report = subparsers.add_parser("report")
    report.add_argument("--manifest", required=True)
    report.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "freeze":
        print(freeze_manifest(phase=args.phase, run_dirs=args.run_dir, output_dir=args.output_dir))
        return
    for path in write_report(manifest_path=args.manifest, output_dir=args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
