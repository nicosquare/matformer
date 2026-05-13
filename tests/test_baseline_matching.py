import copy

from training.baselines import (
    add_baseline_notes_to_summary,
    build_baseline_match_record,
)
from utils.config import resolve_run_config


def _debug_nested_config():
    return resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
    )


def _debug_standalone_config(granularity):
    return resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id=f"debug-standalone-{granularity}-001",
    )


def _counts(value):
    return {"non_embedding_parameters": value}


def test_baseline_match_records_cover_debug_granularity_matrix():
    nested = _debug_nested_config()
    expected = {
        "s": "debug-standalone-s-001",
        "m": "debug-standalone-m-001",
        "l": "debug-standalone-l-001",
        "xl": "debug-standalone-xl-001",
    }

    records = []
    for index, (granularity, standalone_run_id) in enumerate(expected.items(), start=1):
        standalone = _debug_standalone_config(granularity)
        record = build_baseline_match_record(
            nested,
            standalone,
            granularity,
            nested_counts=_counts(index * 100),
            standalone_counts=_counts(index * 100 + 1),
        )
        records.append(record)

        assert record["match_id"] == (
            f"debug-nested-001__{standalone_run_id}__{granularity}"
        )
        assert record["nested_run_id"] == "debug-nested-001"
        assert record["standalone_run_id"] == standalone_run_id
        assert record["granularity"] == granularity
        assert record["non_embedding_parameters_nested"] == index * 100
        assert record["non_embedding_parameters_standalone"] == index * 100 + 1
        assert record["match_notes"] == []

    summary = {
        "run_id": "debug-nested-001",
        "notes": [],
    }
    updated = add_baseline_notes_to_summary(summary, records)

    assert [record["granularity"] for record in updated["baseline_matches"]] == [
        "s",
        "m",
        "l",
        "xl",
    ]
    assert updated["baseline_mismatch_notes"] == []


def test_baseline_match_record_exposes_mismatch_notes():
    nested = _debug_nested_config()
    standalone = _debug_standalone_config("s")
    mismatched = copy.deepcopy(standalone)
    mismatched["dataset"]["dataset_name"] = "other/debug-dataset"
    mismatched["training"]["token_budget"] = 4096

    record = build_baseline_match_record(nested, mismatched, "s")

    assert any("dataset mismatch" in note for note in record["match_notes"])
    assert any("token budget mismatch" in note for note in record["match_notes"])


def test_baseline_summary_collects_prefixed_mismatch_notes():
    nested = _debug_nested_config()
    standalone = _debug_standalone_config("m")
    mismatched = copy.deepcopy(standalone)
    mismatched["dataset"]["dataset_phase"] = "medium"
    record = build_baseline_match_record(nested, mismatched, "m")

    updated = add_baseline_notes_to_summary(
        {
            "run_id": "debug-nested-001",
            "notes": ["existing note"],
        },
        [record],
    )

    assert updated["baseline_matches"] == [record]
    assert updated["baseline_mismatch_notes"] == [
        "debug-nested-001__debug-standalone-m-001__m: "
        "dataset phase mismatch: nested=debug, standalone=medium"
    ]
    assert updated["notes"] == [
        "existing note",
        "debug-nested-001__debug-standalone-m-001__m: "
        "dataset phase mismatch: nested=debug, standalone=medium",
    ]
