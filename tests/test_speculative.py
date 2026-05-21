import textwrap

import pytest
import torch
import yaml

from evaluation.speculative import (
    evaluate_speculative_pair,
    load_prompt_texts,
    load_speculative_model_pairs,
    measure_acceptance_and_rollback,
    measure_speculative_metrics,
    measure_throughput_and_latency,
    resolve_speculative_config,
    run_speculative_evaluation,
)
from utils.metrics import build_speculative_task_rows


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
    assert set(prompt_set) == {
        "name",
        "path",
        "prompts",
        "text_field",
        "sample_count",
        "max_prompt_length",
    }
    assert isinstance(prompt_set["name"], str) and prompt_set["name"].strip()
    assert isinstance(prompt_set["text_field"], str) and prompt_set["text_field"].strip()
    assert isinstance(prompt_set["sample_count"], int) and prompt_set["sample_count"] > 0
    assert (
        isinstance(prompt_set["max_prompt_length"], int)
        and prompt_set["max_prompt_length"] > 0
    )
    if prompt_set["path"] is not None:
        assert isinstance(prompt_set["path"], str) and prompt_set["path"].strip()
    prompts = prompt_set["prompts"]
    if prompts is not None:
        assert isinstance(prompts, list) and prompts
        assert all(isinstance(prompt, str) and prompt.strip() for prompt in prompts)
    assert prompt_set["path"] is not None or prompt_set["prompts"] is not None, (
        "prompt_set must provide either a path or inline prompts"
    )


def validate_nested_pair(pair):
    assert isinstance(pair, dict), "nested pair must be a mapping"
    assert set(pair) == {
        "pair_id",
        "draft_run_id",
        "draft_granularity",
        "verifier_run_id",
        "verifier_granularity",
    }
    assert isinstance(pair["pair_id"], str) and pair["pair_id"].strip()
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
    assert set(pair) == {
        "pair_id",
        "draft_run_id",
        "draft_granularity",
        "verifier_run_id",
        "verifier_granularity",
    }
    assert isinstance(pair["pair_id"], str) and pair["pair_id"].strip()
    assert isinstance(pair["draft_run_id"], str) and pair["draft_run_id"].strip()
    assert isinstance(pair["verifier_run_id"], str) and pair["verifier_run_id"].strip()
    assert pair["draft_granularity"] in VALID_GRANULARITIES
    assert pair["verifier_granularity"] in VALID_GRANULARITIES
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
    assert config["decoding"]["max_draft_tokens"] > 0
    assert config["decoding"]["max_new_tokens"] > 0
    assert config["decoding"]["batch_size"] > 0
    assert config["decoding"]["temperature"] == 0.0
    assert config["decoding"]["do_sample"] is False


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
                pair_id: nested-invalid
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
                pair_id: standalone-invalid
                draft_run_id: debug-standalone-s-001
                draft_granularity: s
                verifier_run_id: debug-standalone-s-001
                verifier_granularity: xl
            """
        )
    )

    with pytest.raises(
        AssertionError,
        match="standalone speculative draft and verifier runs must differ",
    ):
        validate_standalone_pair(invalid_config["pairs"]["standalone"])


def test_load_speculative_model_pairs_uses_cached_run_artifacts():
    config = load_speculative_config()
    loaded_run_ids = []

    def fake_run_loader(run_id, output_root):
        loaded_run_ids.append((run_id, output_root))
        return {
            "run_id": run_id,
            "checkpoint_path": f"/tmp/{run_id}.pt",
            "config": {
                "run": {
                    "run_id": run_id,
                    "model_family": "nested" if "nested" in run_id else "standalone",
                    "granularity": "s" if run_id.endswith("s-001") else "xl",
                }
            },
            "summary": {"run_id": run_id},
        }

    def fake_model_loader(artifact, granularity=None):
        return {
            "loaded_run_id": artifact["run_id"],
            "loaded_granularity": granularity,
            "checkpoint_path": artifact["checkpoint_path"],
        }

    pairs = load_speculative_model_pairs(
        config,
        run_loader=fake_run_loader,
        model_loader=fake_model_loader,
    )

    assert loaded_run_ids == [
        ("debug-nested-001", "outputs"),
        ("debug-standalone-s-001", "outputs"),
        ("debug-standalone-xl-001", "outputs"),
    ]
    assert pairs["nested"]["pair_id"] == "nested-s-xl"
    assert pairs["nested"]["draft"]["model"]["loaded_granularity"] == "s"
    assert pairs["nested"]["verifier"]["model"]["loaded_granularity"] == "xl"
    assert pairs["standalone"]["pair_id"] == "standalone-s-xl"
    assert pairs["standalone"]["draft"]["model"]["loaded_run_id"] == "debug-standalone-s-001"
    assert (
        pairs["standalone"]["verifier"]["model"]["loaded_run_id"]
        == "debug-standalone-xl-001"
    )
    assert pairs["standalone"]["draft"]["model"]["loaded_granularity"] == "s"
    assert pairs["standalone"]["verifier"]["model"]["loaded_granularity"] == "xl"


def test_measure_acceptance_and_rollback_uses_prefix_acceptance():
    draft_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    verifier_tokens = torch.tensor([[1, 2, 0], [4, 0, 0]])

    result = measure_acceptance_and_rollback(draft_tokens, verifier_tokens)

    assert result["accepted_tokens"] == 3
    assert result["proposed_tokens"] == 6
    assert result["rollback_count"] == 2
    assert result["sample_count"] == 2
    assert result["acceptance_rate"] == pytest.approx(0.5)
    assert result["rollback_frequency"] == pytest.approx(1.0)


def test_measure_acceptance_and_rollback_full_match_has_zero_rollbacks():
    tokens = torch.tensor([[7, 8], [9, 10]])

    result = measure_acceptance_and_rollback(tokens, tokens.clone())

    assert result["accepted_tokens"] == 4
    assert result["proposed_tokens"] == 4
    assert result["rollback_count"] == 0
    assert result["acceptance_rate"] == pytest.approx(1.0)
    assert result["rollback_frequency"] == pytest.approx(0.0)


def test_measure_acceptance_and_rollback_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="must match exactly"):
        measure_acceptance_and_rollback(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 2]]),
        )


def test_measure_throughput_and_latency_reports_tokens_per_second_and_sample_latency():
    result = measure_throughput_and_latency(
        token_count=12,
        elapsed_seconds=3.0,
        sample_count=4,
    )

    assert result["generated_tokens"] == 12
    assert result["elapsed_seconds"] == pytest.approx(3.0)
    assert result["throughput"] == pytest.approx(4.0)
    assert result["latency"] == pytest.approx(0.75)


def test_measure_speculative_metrics_combines_alignment_and_timing():
    draft_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    verifier_tokens = torch.tensor([[1, 2, 0], [4, 0, 0]])

    result = measure_speculative_metrics(
        draft_tokens,
        verifier_tokens,
        elapsed_seconds=1.5,
    )

    assert result["accepted_tokens"] == 3
    assert result["rollback_count"] == 2
    assert result["throughput"] == pytest.approx(2.0)
    assert result["latency"] == pytest.approx(0.75)


def test_measure_throughput_and_latency_rejects_non_positive_elapsed_time():
    with pytest.raises(ValueError, match="elapsed_seconds must be > 0"):
        measure_throughput_and_latency(
            token_count=1,
            elapsed_seconds=0.0,
            sample_count=1,
        )


def test_resolve_speculative_config_uses_output_root_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "external-output"))

    config = resolve_speculative_config("configs/speculative.yaml")

    assert config["run"]["output_root"] == str(tmp_path / "external-output")
    assert config["run"]["output_dir"] == str(
        tmp_path / "external-output" / "speculative-001"
    )
    assert config["decoding"]["max_draft_tokens"] == 4


def test_load_prompt_texts_prefers_inline_prompts():
    config = load_speculative_config()

    prompts = load_prompt_texts(config)

    assert prompts == config["prompt_set"]["prompts"]


def test_evaluate_speculative_pair_uses_generator_metrics():
    pair = {
        "pair_id": "nested-s-xl",
        "pair_type": "nested",
        "draft": {
            "run_id": "debug-nested-001",
            "granularity": "s",
            "artifact": {
                "run_id": "debug-nested-001",
                "config": {"run": {"sampling_mode": "nested-random"}},
                "summary": {
                    "sampling_mode": "nested-random",
                    "model_shape_label": "dmodel256",
                },
            },
            "model": object(),
        },
        "verifier": {
            "run_id": "debug-nested-001",
            "granularity": "xl",
            "artifact": {
                "run_id": "debug-nested-001",
                "config": {"run": {"sampling_mode": "nested-random"}},
                "summary": {
                    "sampling_mode": "nested-random",
                    "model_shape_label": "dmodel256",
                },
            },
            "model": object(),
        },
    }

    def fake_tokenizer_loader(_artifact):
        return object()

    token_batches = [
        (torch.tensor([[1, 2, 3], [4, 5, 6]]), 1.0),
        (torch.tensor([[1, 2, 0], [4, 0, 0]]), 0.5),
    ]

    def fake_generator(_model, _tokenizer, _prompts, _decoding):
        return token_batches.pop(0)

    result = evaluate_speculative_pair(
        pair,
        prompts=["a", "b"],
        decoding={"max_draft_tokens": 3, "batch_size": 2},
        tokenizer_loader=fake_tokenizer_loader,
        generator=fake_generator,
    )

    assert result["pair_id"] == "nested-s-xl"
    assert result["pair_type"] == "nested"
    assert result["accepted_tokens"] == 3
    assert result["rollback_count"] == 2
    assert result["throughput"] == pytest.approx(2.0)
    assert result["latency"] == pytest.approx(0.75)
    assert result["sampling_mode"] == "nested-random"
    assert result["model_shape_label"] == "dmodel256"


def test_build_speculative_task_rows_serializes_pair_metrics():
    config = load_speculative_config()
    rows = build_speculative_task_rows(
        config,
        [
            {
                "pair_id": "nested-s-xl",
                "pair_type": "nested",
                "draft_granularity": "s",
                "verifier_granularity": "xl",
                "sampling_mode": "nested-random",
                "model_shape_label": "dmodel256",
                "acceptance_rate": 0.5,
                "rollback_frequency": 1.0,
                "throughput": 2.0,
                "latency": 0.75,
            }
        ],
    )

    assert [row["metric_name"] for row in rows] == config["metrics"]
    assert {row["task"] for row in rows} == {"nested-s-xl"}
    assert {row["model_family"] for row in rows} == {"nested"}
    assert {row["granularity"] for row in rows} == {"s->xl"}
    assert rows[0]["run_id"] == "speculative-001"


def test_run_speculative_evaluation_writes_task_results_csv(tmp_path):
    config = resolve_speculative_config(
        "configs/speculative.yaml",
        output_dir=tmp_path / "speculative-001",
    )

    def fake_pair_loader(_config):
        return {
            "nested": {
                "pair_id": "nested-s-xl",
                "pair_type": "nested",
            },
            "standalone": {
                "pair_id": "standalone-s-xl",
                "pair_type": "standalone",
            },
        }

    def fake_pair_evaluator(pair, prompts, decoding):
        assert prompts
        assert decoding["max_draft_tokens"] == 4
        return {
            "pair_id": pair["pair_id"],
            "pair_type": pair["pair_type"],
            "draft_granularity": "s",
            "verifier_granularity": "xl",
            "sampling_mode": (
                "nested-random" if pair["pair_type"] == "nested" else "standalone"
            ),
            "model_shape_label": "dmodel256",
            "acceptance_rate": 0.5,
            "rollback_frequency": 1.0,
            "throughput": 2.0,
            "latency": 0.75,
        }

    result = run_speculative_evaluation(
        config,
        pair_loader=fake_pair_loader,
        pair_evaluator=fake_pair_evaluator,
    )

    task_results_path = result["task_results_path"]
    assert task_results_path.exists()
    rows = task_results_path.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0] == (
        "run_id,suite_id,task,model_family,model_size_label,model_shape_label,"
        "sampling_mode,granularity,metric_name,metric_value"
    )
    assert len(rows) == 1 + 2 * len(config["metrics"])
