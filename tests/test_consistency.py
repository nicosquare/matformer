import textwrap

import pytest
import yaml


VALID_GRANULARITIES = {"s", "m", "l", "xl"}
VALID_CONSISTENCY_METRICS = {
    "token_level_agreement",
    "distribution_divergence",
    "top_k_overlap",
}


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


def test_consistency_config_includes_token_level_agreement_metric():
    config = load_consistency_config()

    assert config["run"]["phase_id"] == "consistency"
    validate_consistency_metrics(config["consistency"]["metrics"])
    assert "token_level_agreement" in config["consistency"]["metrics"]
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
