import textwrap

import pytest
import torch
import yaml

from src.evaluation.consistency import (
    DEFAULT_TOP_K_VALUES,
    KL_DIVERGENCE_DEFERRED_REASON,
    build_consistency_rows,
    deferred_kl_divergence_note,
    summarize_consistency_suite,
    token_level_agreement,
    top_k_overlap,
)


VALID_GRANULARITIES = {"s", "m", "l", "xl"}
VALID_CONSISTENCY_METRICS = {
    "token_level_agreement",
    "distribution_divergence",
    "top_k_overlap",
}
VALID_DEFERRED_METRICS = {"kl_divergence"}


def load_consistency_config(path="configs/consistency.yaml"):
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def validate_consistency_metrics(metrics):
    assert isinstance(metrics, list) and metrics, "consistency.metrics must be non-empty"
    unknown_metrics = [metric for metric in metrics if metric not in VALID_CONSISTENCY_METRICS]
    assert not unknown_metrics, f"unknown consistency metrics: {unknown_metrics}"


def validate_consistency_pairs(pairs):
    assert isinstance(pairs, list) and pairs, "consistency.pairs must be non-empty"

    for pair in pairs:
        assert set(pair) == {"small_granularity", "large_granularity"}
        small = pair["small_granularity"]
        large = pair["large_granularity"]
        assert small in VALID_GRANULARITIES
        assert large in VALID_GRANULARITIES
        assert small != large


def validate_mix_and_match_patterns(patterns):
    assert isinstance(patterns, list) and patterns, "mix_and_match.patterns must be non-empty"

    seen_names = set()
    for pattern in patterns:
        assert "name" in pattern
        assert "layer_granularities" in pattern
        assert pattern["name"] not in seen_names
        seen_names.add(pattern["name"])

        layer_granularities = pattern["layer_granularities"]
        assert isinstance(layer_granularities, list) and layer_granularities
        assert all(granularity in VALID_GRANULARITIES for granularity in layer_granularities)
        assert len(set(layer_granularities)) > 1, (
            "mix-and-match patterns must be heterogeneous rather than fixed-width"
        )


def validate_top_k_values(top_k_values):
    assert isinstance(top_k_values, list) and top_k_values, (
        "consistency.top_k_values must be non-empty"
    )
    assert all(isinstance(value, int) and value > 0 for value in top_k_values)


def validate_deferred_metrics(metrics):
    assert isinstance(metrics, list), "consistency.deferred_metrics must be a list"
    unknown_metrics = [metric for metric in metrics if metric not in VALID_DEFERRED_METRICS]
    assert not unknown_metrics, f"unknown deferred metrics: {unknown_metrics}"


def test_consistency_config_includes_token_level_agreement_metric():
    config = load_consistency_config()

    assert config["run"]["phase_id"] == "consistency"
    validate_consistency_metrics(config["consistency"]["metrics"])
    assert "token_level_agreement" in config["consistency"]["metrics"]
    assert "top_k_overlap" in config["consistency"]["metrics"]
    validate_top_k_values(config["consistency"]["top_k_values"])
    validate_deferred_metrics(config["consistency"]["deferred_metrics"])
    assert "kl_divergence" in config["consistency"]["deferred_metrics"]
    assert config["consistency"]["sample_count"] > 0
    validate_consistency_pairs(config["consistency"]["pairs"])


def test_consistency_metric_validation_rejects_unknown_metric():
    with pytest.raises(AssertionError, match="unknown consistency metrics"):
        validate_consistency_metrics(["token_level_agreement", "bad_metric"])


def test_mix_and_match_patterns_use_canonical_heterogeneous_granularities():
    config = load_consistency_config()

    validate_mix_and_match_patterns(config["mix_and_match"]["patterns"])


def test_mix_and_match_pattern_validation_rejects_invalid_granularity():
    invalid_config = yaml.safe_load(
        textwrap.dedent(
            """
            mix_and_match:
              patterns:
                - name: invalid
                  layer_granularities: [xl, tiny, xl]
            """
        )
    )

    with pytest.raises(AssertionError):
        validate_mix_and_match_patterns(invalid_config["mix_and_match"]["patterns"])


def test_token_level_agreement_uses_attention_mask():
    small_logits = torch.tensor(
        [
            [
                [10.0, 1.0, 0.0],
                [0.0, 9.0, 1.0],
                [0.0, 8.0, 2.0],
            ]
        ]
    )
    large_logits = torch.tensor(
        [
            [
                [9.0, 2.0, 0.0],
                [8.0, 1.0, 0.0],
                [1.0, 7.0, 0.0],
            ]
        ]
    )
    attention_mask = torch.tensor([[1, 1, 0]])

    result = token_level_agreement(
        small_logits,
        large_logits,
        attention_mask=attention_mask,
    )

    assert result["metric_name"] == "token_level_agreement"
    assert result["sample_count"] == 2
    assert result["metric_value"] == pytest.approx(0.5)


def test_top_k_overlap_reports_top_k_field_and_overlap_value():
    small_logits = torch.tensor(
        [[[9.0, 8.0, 7.0, 1.0], [6.0, 5.0, 4.0, 3.0]]]
    )
    large_logits = torch.tensor(
        [[[8.0, 9.0, 1.0, 7.0], [6.0, 4.0, 5.0, 3.0]]]
    )

    result = top_k_overlap(small_logits, large_logits, k=2)

    assert result["metric_name"] == "top_k_overlap"
    assert result["top_k"] == 2
    assert result["sample_count"] == 2
    assert result["metric_value"] == pytest.approx(0.75)


def test_deferred_kl_divergence_note_is_explicit():
    note = deferred_kl_divergence_note(sample_count=8)

    assert note["metric_name"] == "kl_divergence"
    assert note["metric_value"] is None
    assert note["sample_count"] == 8
    assert note["deferred"] is True
    assert note["deferred_reason"] == KL_DIVERGENCE_DEFERRED_REASON


def test_build_consistency_rows_includes_token_agreement_top_k_and_kl_note():
    small_logits = torch.tensor(
        [[[9.0, 1.0, 0.0], [0.0, 8.0, 2.0]]]
    )
    large_logits = torch.tensor(
        [[[8.0, 2.0, 0.0], [1.0, 7.0, 0.0]]]
    )

    rows = build_consistency_rows(
        comparison_id="nested-s-xl",
        small_run_id="debug-nested-001",
        large_run_id="debug-nested-001",
        small_granularity="s",
        large_granularity="xl",
        small_logits=small_logits,
        large_logits=large_logits,
        top_k_values=DEFAULT_TOP_K_VALUES,
        include_deferred_kl_note=True,
    )
    summary = summarize_consistency_suite(rows)

    assert [row["metric_name"] for row in rows] == [
        "token_level_agreement",
        "top_k_overlap",
        "kl_divergence",
    ]
    assert all(row["comparison_id"] == "nested-s-xl" for row in rows)
    assert summary["token_level_agreement"]["sample_count"] == 2
    assert summary["top_k_overlap"]["top_k"] == 3
    assert summary["kl_divergence"]["deferred"] is True
