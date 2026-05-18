import textwrap

import pytest
import yaml


VALID_GRANULARITIES = {"s", "m", "l", "xl"}
VALID_SPECULATIVE_METRICS = {
    "acceptance_rate",
    "rollback_frequency",
    "throughput",
    "latency",
}


def load_speculative_config(path="configs/speculative.yaml"):
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def validate_speculative_metrics(metrics):
    assert isinstance(metrics, list) and metrics, "metrics must be non-empty"
    unknown_metrics = [
        metric for metric in metrics if metric not in VALID_SPECULATIVE_METRICS
    ]
    assert not unknown_metrics, f"unknown speculative metrics: {unknown_metrics}"


def validate_prompt_set(prompt_set):
    assert isinstance(prompt_set, dict), "prompt_set must be a mapping"
    assert set(prompt_set) == {"name", "path", "sample_count"}
    assert isinstance(prompt_set["name"], str) and prompt_set["name"].strip()
    assert isinstance(prompt_set["sample_count"], int) and prompt_set["sample_count"] > 0
    if prompt_set["path"] is not None:
        assert isinstance(prompt_set["path"], str) and prompt_set["path"].strip()


def validate_nested_pair(pair):
    assert isinstance(pair, dict), "nested pair must be a mapping"
    assert set(pair) == {
        "draft_run_id",
        "draft_granularity",
        "verifier_run_id",
        "verifier_granularity",
    }
    assert isinstance(pair["draft_run_id"], str) and pair["draft_run_id"].strip()
    assert isinstance(pair["verifier_run_id"], str) and pair["verifier_run_id"].strip()
    assert pair["draft_granularity"] in VALID_GRANULARITIES
    assert pair["verifier_granularity"] in VALID_GRANULARITIES
    assert pair["draft_granularity"] != pair["verifier_granularity"]
    assert granularity_rank(pair["draft_granularity"]) < granularity_rank(
        pair["verifier_granularity"]
    ), "nested speculative draft must be smaller than verifier"


def validate_standalone_pair(pair):
    assert isinstance(pair, dict), "standalone pair must be a mapping"
    assert set(pair) == {"draft_run_id", "verifier_run_id"}
    assert isinstance(pair["draft_run_id"], str) and pair["draft_run_id"].strip()
    assert isinstance(pair["verifier_run_id"], str) and pair["verifier_run_id"].strip()
    assert pair["draft_run_id"] != pair["verifier_run_id"], (
        "standalone speculative draft and verifier runs must differ"
    )


def granularity_rank(value):
    order = {"s": 0, "m": 1, "l": 2, "xl": 3}
    return order[value]


def test_speculative_config_declares_required_metrics_and_outputs():
    config = load_speculative_config()

    assert config["run"]["phase_id"] == "speculative"
    validate_speculative_metrics(config["metrics"])
    assert config["metrics"] == [
        "acceptance_rate",
        "rollback_frequency",
        "throughput",
        "latency",
    ]
    assert config["outputs"]["task_results_csv"] == "task_results.csv"
    assert config["outputs"]["run_summary_json"] == "run_summary.json"


def test_speculative_metric_validation_rejects_unknown_metric():
    with pytest.raises(AssertionError, match="unknown speculative metrics"):
        validate_speculative_metrics(["acceptance_rate", "bad_metric"])


def test_speculative_config_declares_shared_prompt_set_and_pairs():
    config = load_speculative_config()

    validate_prompt_set(config["prompt_set"])
    assert set(config["pairs"]) == {"nested", "standalone"}
    validate_nested_pair(config["pairs"]["nested"])
    validate_standalone_pair(config["pairs"]["standalone"])


def test_nested_pair_validation_rejects_non_increasing_granularity():
    invalid_config = yaml.safe_load(
        textwrap.dedent(
            """
            pairs:
              nested:
                draft_run_id: debug-nested-001
                draft_granularity: xl
                verifier_run_id: debug-nested-001
                verifier_granularity: s
            """
        )
    )

    with pytest.raises(
        AssertionError,
        match="nested speculative draft must be smaller than verifier",
    ):
        validate_nested_pair(invalid_config["pairs"]["nested"])


def test_prompt_set_validation_rejects_missing_sample_count():
    invalid_config = yaml.safe_load(
        textwrap.dedent(
            """
            prompt_set:
              name: debug-prompts
              path: null
              sample_count: 0
            """
        )
    )

    with pytest.raises(AssertionError):
        validate_prompt_set(invalid_config["prompt_set"])


def test_standalone_pair_validation_rejects_reused_run_id():
    invalid_config = yaml.safe_load(
        textwrap.dedent(
            """
            pairs:
              standalone:
                draft_run_id: debug-standalone-s-001
                verifier_run_id: debug-standalone-s-001
            """
        )
    )

    with pytest.raises(
        AssertionError,
        match="standalone speculative draft and verifier runs must differ",
    ):
        validate_standalone_pair(invalid_config["pairs"]["standalone"])
