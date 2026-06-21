"""Helpers for documenting nested-vs-standalone baseline matches."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from src.utils.config import resolve_all_run_configs, resolve_run_config
from src.utils.metrics import (
    baseline_match_id as metrics_baseline_match_id,
)
from src.utils.metrics import (
    build_baseline_match_row,
    write_json_artifact,
)

BASELINE_MATCH_FIELDS = [
    ("dataset.dataset_name", "dataset"),
    ("dataset.dataset_split", "dataset split"),
    ("dataset.dataset_phase", "dataset phase"),
    ("dataset.preprocessing_notes", "preprocessing"),
    ("training.token_budget", "token budget"),
    ("model.context_length", "context length"),
    ("model.vocab_size_assumption", "vocabulary assumption"),
]

DEBUG_MATRIX_CONFIG = Path("configs/debug_matrix.yaml")
DEFAULT_DEBUG_NESTED_RUN_ID = "debug-nested-001"
DEFAULT_DEBUG_BASELINE_GRANULARITY = "s"
DEFAULT_DEBUG_BASELINE_GRANULARITIES = ("s", "m", "l", "xl")

RunCallable = Callable[[dict[str, Any]], dict[str, Any]]


def run_debug_nested_with_one_baseline(
    config_path: str | Path = DEBUG_MATRIX_CONFIG,
    nested_run_id: str = DEFAULT_DEBUG_NESTED_RUN_ID,
    baseline_granularity: str = DEFAULT_DEBUG_BASELINE_GRANULARITY,
    overrides: Iterable[str] | None = None,
    output_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    runner: RunCallable | None = None,
) -> dict[str, Any]:
    result = run_debug_nested_with_baselines(
        config_path=config_path,
        nested_run_id=nested_run_id,
        baseline_granularities=[baseline_granularity],
        overrides=overrides,
        output_root=output_root,
        output_dir=output_dir,
        runner=runner,
    )

    return {
        **result,
        "standalone_config": result["standalone_configs"][0],
        "standalone_result": result["standalone_results"][0],
    }


def run_debug_nested_with_baselines(
    config_path: str | Path = DEBUG_MATRIX_CONFIG,
    nested_run_id: str = DEFAULT_DEBUG_NESTED_RUN_ID,
    baseline_granularities: Iterable[str] | None = None,
    overrides: Iterable[str] | None = None,
    output_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    runner: RunCallable | None = None,
) -> dict[str, Any]:
    baseline_granularities = normalize_baseline_granularities(
        baseline_granularities,
    )
    nested_config, standalone_configs = resolve_debug_nested_baseline_matrix_configs(
        config_path=config_path,
        nested_run_id=nested_run_id,
        baseline_granularities=baseline_granularities,
        overrides=overrides,
        output_root=output_root,
        output_dir=output_dir,
    )

    if runner is None:
        runner = run_training_config

    nested_result = runner(nested_config)
    standalone_results = []
    baseline_match_records = []
    for standalone_config in standalone_configs:
        granularity = standalone_config["run"]["granularity"]
        standalone_result = runner(standalone_config)
        standalone_results.append(standalone_result)
        baseline_match_records.append(
            build_baseline_match_record(
                nested_config,
                standalone_config,
                granularity,
                nested_counts=_result_counts_for_granularity(
                    nested_result,
                    granularity,
                ),
                standalone_counts=_result_counts_for_granularity(
                    standalone_result,
                    granularity,
                ),
            )
        )

    nested_summary_path = write_baseline_matches_to_summary(
        nested_result["summary_path"],
        baseline_match_records,
    )

    return {
        "nested_config": nested_config,
        "standalone_configs": standalone_configs,
        "nested_result": nested_result,
        "standalone_results": standalone_results,
        "baseline_match_records": baseline_match_records,
        "nested_summary_path": nested_summary_path,
    }


def resolve_debug_nested_baseline_configs(
    config_path: str | Path = DEBUG_MATRIX_CONFIG,
    nested_run_id: str = DEFAULT_DEBUG_NESTED_RUN_ID,
    baseline_granularity: str = DEFAULT_DEBUG_BASELINE_GRANULARITY,
    overrides: Iterable[str] | None = None,
    output_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    nested_config, standalone_configs = resolve_debug_nested_baseline_matrix_configs(
        config_path=config_path,
        nested_run_id=nested_run_id,
        baseline_granularities=[baseline_granularity],
        overrides=overrides,
        output_root=output_root,
        output_dir=output_dir,
    )
    return nested_config, standalone_configs[0]


def resolve_debug_nested_baseline_matrix_configs(
    config_path: str | Path = DEBUG_MATRIX_CONFIG,
    nested_run_id: str = DEFAULT_DEBUG_NESTED_RUN_ID,
    baseline_granularities: Iterable[str] | None = None,
    overrides: Iterable[str] | None = None,
    output_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline_granularities = normalize_baseline_granularities(
        baseline_granularities,
    )
    resolved_overrides = output_overrides(overrides, output_root)
    configs = resolve_all_run_configs(config_path, overrides=resolved_overrides)
    if output_dir is None:
        nested_config = find_run_config(configs, nested_run_id, model_family="nested")
    else:
        nested_config = resolve_run_config(
            config_path,
            run_id=nested_run_id,
            overrides=resolved_overrides,
            output_dir=output_dir,
        )
    standalone_configs = [
        find_standalone_baseline_config(configs, granularity)
        for granularity in baseline_granularities
    ]
    return nested_config, standalone_configs


def normalize_baseline_granularities(
    baseline_granularities: Iterable[str] | None,
) -> list[str]:
    if baseline_granularities is None:
        return list(DEFAULT_DEBUG_BASELINE_GRANULARITIES)

    normalized = []
    for raw_granularity in baseline_granularities:
        for granularity in str(raw_granularity).replace(",", " ").split():
            if granularity:
                normalized.append(granularity)

    return normalized or list(DEFAULT_DEBUG_BASELINE_GRANULARITIES)


def output_overrides(
    overrides: Iterable[str] | None,
    output_root: str | Path | None,
) -> list[str]:
    resolved_overrides = list(overrides or [])
    if output_root is not None:
        resolved_overrides.append(f"run.output_root={output_root}")
    return resolved_overrides


def find_run_config(
    configs: list[dict[str, Any]],
    run_id: str,
    model_family: str | None = None,
) -> dict[str, Any]:
    for config in configs:
        run = config["run"]
        if run["run_id"] != run_id:
            continue
        if model_family is not None and run.get("model_family") != model_family:
            continue
        return config

    family_note = f" with model_family={model_family}" if model_family else ""
    raise ValueError(f"Could not find run_id={run_id}{family_note}")


def find_standalone_baseline_config(
    configs: list[dict[str, Any]],
    granularity: str,
) -> dict[str, Any]:
    for config in configs:
        run = config["run"]
        if run.get("model_family") != "standalone":
            continue
        if run.get("granularity") == granularity:
            return config

    raise ValueError(f"Could not find standalone baseline for {granularity}")


def run_training_config(config: dict[str, Any]) -> dict[str, Any]:
    from src.training.run import run_training

    return run_training(config)


def write_baseline_matches_to_summary(
    summary_path: str | Path,
    baseline_match_records: list[dict[str, Any]],
) -> Path:
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = add_baseline_notes_to_summary(summary, baseline_match_records)
    return write_json_artifact(summary_path, summary)


def build_baseline_match_record(
    nested_config: dict[str, Any],
    standalone_config: dict[str, Any],
    granularity: str,
    nested_counts: dict[str, int] | None = None,
    standalone_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    mismatch_notes = compare_baseline_configs(
        nested_config,
        standalone_config,
        granularity,
    )
    return build_baseline_match_row(
        nested_config,
        standalone_config,
        granularity,
        nested_counts=nested_counts,
        standalone_counts=standalone_counts,
        match_notes=mismatch_notes,
    )


def compare_baseline_configs(
    nested_config: dict[str, Any],
    standalone_config: dict[str, Any],
    granularity: str,
) -> list[str]:
    notes = []

    if nested_config["run"].get("model_family") != "nested":
        notes.append("nested config is not labeled model_family=nested")
    if standalone_config["run"].get("model_family") != "standalone":
        notes.append("standalone config is not labeled model_family=standalone")

    standalone_granularity = standalone_config["run"].get("granularity")
    if standalone_granularity != granularity:
        notes.append(
            f"standalone granularity {standalone_granularity} does not match {granularity}"
        )

    nested_granularities = nested_config["model"].get("granularities", [])
    if granularity not in nested_granularities:
        notes.append(f"nested run does not expose granularity {granularity}")

    for dotted_path, label in BASELINE_MATCH_FIELDS:
        nested_value = _get_dotted(nested_config, dotted_path)
        standalone_value = _get_dotted(standalone_config, dotted_path)
        if nested_value != standalone_value:
            notes.append(
                f"{label} mismatch: nested={nested_value}, standalone={standalone_value}"
            )

    return notes


def add_baseline_notes_to_summary(
    summary: dict[str, Any],
    baseline_match_records: list[dict[str, Any]],
) -> dict[str, Any]:
    mismatch_notes = []
    for record in baseline_match_records:
        for note in record.get("match_notes", []):
            mismatch_notes.append(f"{record['match_id']}: {note}")

    summary = dict(summary)
    summary.setdefault("baseline_matches", baseline_match_records)
    summary.setdefault("baseline_mismatch_notes", mismatch_notes)
    if mismatch_notes:
        notes = list(summary.get("notes", []))
        notes.extend(mismatch_notes)
        summary["notes"] = notes
    return summary


def baseline_match_id(
    nested_run_id: str,
    standalone_run_id: str,
    granularity: str,
) -> str:
    return metrics_baseline_match_id(nested_run_id, standalone_run_id, granularity)


def _get_dotted(config: dict[str, Any], dotted_path: str):
    value = config
    for part in dotted_path.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _result_counts_for_granularity(
    result: dict[str, Any],
    granularity: str,
) -> dict[str, int] | None:
    counts_by_granularity = result.get("parameter_counts_by_granularity", {})
    return counts_by_granularity.get(granularity)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEBUG_MATRIX_CONFIG),
        help="Debug matrix config path.",
    )
    parser.add_argument(
        "--nested-run-id",
        default=DEFAULT_DEBUG_NESTED_RUN_ID,
        help="Nested run id to pair with the standalone baseline.",
    )
    parser.add_argument(
        "--granularity",
        action="append",
        dest="granularities",
        help=(
            "Standalone baseline granularity to run. Repeat or use comma-separated "
            "values. Defaults to s,m,l,xl."
        ),
    )
    parser.add_argument(
        "--output-root",
        help="Root directory for all debug matrix run artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        help="Explicit output directory for the selected nested run.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted config override, for example training.max_steps=1.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_debug_nested_with_baselines(
        config_path=args.config,
        nested_run_id=args.nested_run_id,
        baseline_granularities=args.granularities,
        overrides=args.override,
        output_root=args.output_root or os.environ.get("OUTPUT_ROOT"),
        output_dir=args.output_dir,
    )
    print(result["nested_summary_path"])


if __name__ == "__main__":
    main()
