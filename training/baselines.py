"""Helpers for documenting nested-vs-standalone baseline matches."""

from __future__ import annotations

from typing import Any


BASELINE_MATCH_FIELDS = [
    ("dataset.dataset_name", "dataset"),
    ("dataset.dataset_split", "dataset split"),
    ("dataset.dataset_phase", "dataset phase"),
    ("dataset.preprocessing_notes", "preprocessing"),
    ("training.token_budget", "token budget"),
    ("model.context_length", "context length"),
    ("model.vocab_size_assumption", "vocabulary assumption"),
]


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
    nested_run = nested_config["run"]
    standalone_run = standalone_config["run"]

    return {
        "match_id": baseline_match_id(
            nested_run["run_id"],
            standalone_run["run_id"],
            granularity,
        ),
        "nested_run_id": nested_run["run_id"],
        "standalone_run_id": standalone_run["run_id"],
        "granularity": granularity,
        "non_embedding_parameters_nested": _non_embedding_count(nested_counts),
        "non_embedding_parameters_standalone": _non_embedding_count(standalone_counts),
        "match_notes": mismatch_notes,
    }


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
    return f"{nested_run_id}__{standalone_run_id}__{granularity}"


def _get_dotted(config: dict[str, Any], dotted_path: str):
    value = config
    for part in dotted_path.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _non_embedding_count(counts: dict[str, int] | None):
    if counts is None:
        return None
    return counts.get("non_embedding_parameters")
