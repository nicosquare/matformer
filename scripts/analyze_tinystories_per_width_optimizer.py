#!/usr/bin/env python3
"""Freeze and report the six-run TinyStories optimizer-state comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
VALIDATION_FIGURE_NAME = "optimizer_state_validation_loss_over_tokens.png"
ENDPOINT_FIGURE_NAME = "optimizer_state_endpoint_by_width.png"
RESOURCE_FIGURE_NAME = "optimizer_state_resource_costs.png"


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


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def _mean_min_max(values: Sequence[float]) -> tuple[float, float, float]:
    if not values:
        raise OptimizerStateAnalysisError("Cannot plot an empty optimizer-state series")
    return sum(values) / len(values), min(values), max(values)


def _save_figure(figure: Any, destination: Path, *, dpi: int = 160) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    figure.savefig(temporary, dpi=dpi, bbox_inches="tight")
    temporary.replace(destination)
    return destination


def _plot_validation_trajectories(
    runs: Sequence[Any], output_dir: Path, *, phase: str
) -> Path:
    pyplot = _pyplot()
    widths = tuple(runs[0].ordered_widths)
    figure, axes = pyplot.subplots(2, 2, figsize=(11, 7), sharex=True)
    colors = {"shared": "tab:blue", "per_granularity": "tab:orange"}
    labels = {"shared": "Shared", "per_granularity": "Per-granularity"}

    for axis, width in zip(axes.flat, widths):
        for scope in ("shared", "per_granularity"):
            scoped_runs = [run for run in runs if run.state_scope == scope]
            values_by_tokens: dict[int, list[float]] = {}
            for run in scoped_runs:
                for row in run.metrics:
                    if (
                        row.get("split") != "validation"
                        or row.get("granularity") != width
                        or row.get("loss") in (None, "")
                    ):
                        continue
                    tokens = int(row.get("tokens_seen") or 0)
                    values_by_tokens.setdefault(tokens, []).append(float(row["loss"]))
            tokens = sorted(values_by_tokens)
            summaries = [_mean_min_max(values_by_tokens[token]) for token in tokens]
            means = [item[0] for item in summaries]
            lows = [item[1] for item in summaries]
            highs = [item[2] for item in summaries]
            axis.plot(
                tokens,
                means,
                color=colors[scope],
                linewidth=1.8,
                label=f"{labels[scope]} · n={len(scoped_runs)} seeds",
            )
            axis.fill_between(tokens, lows, highs, color=colors[scope], alpha=0.15)
        axis.set_title(width)
        axis.set_ylabel("Validation loss")
        axis.grid(alpha=0.25)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    for axis in axes[-1]:
        axis.set_xlabel("Optimizer-training tokens")
    axes[0, 0].legend(frameon=False)
    figure.suptitle(
        f"Per-width optimizer state: ordinary-validation trajectories ({phase})"
    )
    figure.tight_layout()
    destination = output_dir / VALIDATION_FIGURE_NAME
    _save_figure(figure, destination)
    pyplot.close(figure)
    return destination


def _plot_endpoint_by_width(report: Mapping[str, Any], output_dir: Path) -> Path:
    pyplot = _pyplot()
    outcomes = list(report["outcomes"])
    widths = tuple(outcomes[0]["ordered_widths"])
    colors = {"shared": "tab:blue", "per_granularity": "tab:orange"}
    labels = {"shared": "Shared", "per_granularity": "Per-granularity"}
    figure, axes = pyplot.subplots(2, 2, figsize=(11, 7), sharex=True)

    for axis, metric, title in (
        (axes[0, 0], "loss", "Endpoint loss"),
        (axes[0, 1], "perplexity", "Endpoint perplexity"),
    ):
        for scope in ("shared", "per_granularity"):
            scoped = [row for row in outcomes if row["state_scope"] == scope]
            summaries = [
                _mean_min_max(
                    [float(row["per_width_outcomes"][width][metric]) for row in scoped]
                )
                for width in widths
            ]
            means = [item[0] for item in summaries]
            lows = [item[1] for item in summaries]
            highs = [item[2] for item in summaries]
            positions = list(range(len(widths)))
            axis.plot(
                positions,
                means,
                marker="o",
                linewidth=2,
                color=colors[scope],
                label=f"{labels[scope]} · n={len(scoped)} seeds",
            )
            axis.fill_between(positions, lows, highs, color=colors[scope], alpha=0.15)
        axis.set_title(title)
        axis.set_ylabel("Loss" if metric == "loss" else "Perplexity")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)

    paired = sorted(report["paired_deltas"], key=lambda row: int(row["seed"]))
    positions = list(range(len(widths)))
    loss_deltas_by_width = {
        width: [float(row["per_width_loss_deltas"][width]) for row in paired]
        for width in widths
    }
    for row in paired:
        seed = int(row["seed"])
        deltas = [float(row["per_width_loss_deltas"][width]) for width in widths]
        axes[1, 0].plot(
            positions,
            deltas,
            marker="o",
            linewidth=1,
            alpha=0.35,
            label=f"Seed {seed}",
        )
        axes[1, 1].plot(
            positions,
            [100.0 * math.expm1(value) for value in deltas],
            marker="o",
            linewidth=1,
            alpha=0.35,
            label=f"Seed {seed}",
        )
    mean_loss_deltas = [
        _mean_min_max(loss_deltas_by_width[width])[0] for width in widths
    ]
    axes[1, 0].plot(
        positions,
        mean_loss_deltas,
        color="black",
        marker="D",
        linewidth=2,
        label="Three-seed mean",
    )
    mean_perplexity_percent = [
        _mean_min_max(
            [100.0 * math.expm1(value) for value in loss_deltas_by_width[width]]
        )[0]
        for width in widths
    ]
    axes[1, 1].plot(
        positions,
        mean_perplexity_percent,
        color="black",
        marker="D",
        linewidth=2,
        label="Three-seed mean",
    )
    axes[1, 0].set_title("Paired loss delta: per-granularity − shared")
    axes[1, 0].set_ylabel("Loss difference")
    axes[1, 1].set_title("Paired perplexity change")
    axes[1, 1].set_ylabel("Perplexity change (%)")
    for axis in axes[1]:
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.grid(alpha=0.25)
        axis.set_xlabel("Model width")
    axes[1, 0].legend(frameon=False, fontsize=9)
    for axis in axes.flat:
        axis.set_xticks(positions, widths)

    source = (
        "final holdout"
        if bool(report.get("holdout_results_complete"))
        else "trailing-five ordinary validation"
    )
    figure.suptitle(
        f"Per-width optimizer-state endpoints: {report['phase']} {source}"
    )
    figure.tight_layout()
    destination = output_dir / ENDPOINT_FIGURE_NAME
    _save_figure(figure, destination)
    pyplot.close(figure)
    return destination


def _plot_resource_costs(report: Mapping[str, Any], output_dir: Path) -> Path:
    pyplot = _pyplot()
    outcomes = list(report["outcomes"])
    seeds = sorted({int(row["seed"]) for row in outcomes})
    colors = {"shared": "tab:blue", "per_granularity": "tab:orange"}
    labels = {"shared": "Shared", "per_granularity": "Per-granularity"}
    figure, axes = pyplot.subplots(1, 3, figsize=(12, 4))
    specs = (
        ("wall_time_seconds", 60.0, "Training wall time (minutes)"),
        ("peak_accelerator_memory_bytes", 1024.0**2, "Peak accelerator memory (MiB)"),
        ("resumable_checkpoint_bytes", 1024.0**2, "Terminal checkpoint (MiB)"),
    )
    width = 0.36
    positions = list(range(len(seeds)))
    for axis, (field, divisor, title) in zip(axes, specs):
        for index, scope in enumerate(("shared", "per_granularity")):
            scoped_values = {
                int(row["seed"]): float(row["resources"][field]) / divisor
                for row in outcomes
                if row["state_scope"] == scope
            }
            offsets = [position + (index - 0.5) * width for position in positions]
            axis.bar(
                offsets,
                [scoped_values[seed] for seed in seeds],
                width=width,
                color=colors[scope],
                label=labels[scope],
            )
        axis.set_title(title)
        axis.set_xticks(positions, [str(seed) for seed in seeds])
        axis.set_xlabel("Seed")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle("Per-width optimizer-state operational costs")
    figure.tight_layout()
    destination = output_dir / RESOURCE_FIGURE_NAME
    _save_figure(figure, destination)
    pyplot.close(figure)
    return destination


def _write_report_figures(
    runs: Sequence[Any], report: Mapping[str, Any], output_dir: Path
) -> tuple[Path, Path, Path]:
    return (
        _plot_validation_trajectories(runs, output_dir, phase=str(report["phase"])),
        _plot_endpoint_by_width(report, output_dir),
        _plot_resource_costs(report, output_dir),
    )


def write_report(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, ...]:
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
    figure_paths = _write_report_figures(runs, report, root)
    return written, csv_path, *figure_paths


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
