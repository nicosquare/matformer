import copy
import json
import math
import textwrap

import pytest

from utils.config import (
    ConfigError,
    resolve_all_run_configs,
    resolve_run_config,
    validate_run_config,
    write_resolved_config,
)


def _write_single_run_config(tmp_path):
    config_path = tmp_path / "single_run.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            run:
              run_id: single-output-root-001
              phase_id: debug_matrix
              model_family: nested
              model_size_label: debug
              completion_label: debug
              seed: 42

            model:
              base_model_name: debug-llama
              paper_aligned: false
              num_layers: 2
              num_attention_heads: 4
              hidden_size: 128
              intermediate_size: 512
              context_length: 64
              vocab_size_assumption: 32000
              granularities: [s, m, l, xl]

            training:
              token_budget: 8192
              max_steps: 1
              batch_size_per_process: 1
              learning_rate: 0.0003
              warmup_steps: 0
              eval_interval: 0

            dataset:
              dataset_name: roneneldan/TinyStories
              dataset_split: train
              dataset_phase: debug
              sample_limit: 2
              preprocessing_notes: debug_output_root_resolution

            outputs:
              save_config: true
              save_metrics_csv: true
              save_run_summary_json: true
              save_checkpoints: false
              make_plots: false

            evaluation:
              validation: true
              downstream_suite: []
              consistency: false
              speculative: false
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return config_path


def test_resolve_debug_matrix_expands_nested_and_standalone_runs():
    resolved_runs = resolve_all_run_configs("configs/debug_matrix.yaml")

    run_ids = [config["run"]["run_id"] for config in resolved_runs]
    assert run_ids == [
        "debug-nested-001",
        "debug-standalone-s-001",
        "debug-standalone-m-001",
        "debug-standalone-l-001",
        "debug-standalone-xl-001",
    ]

    nested = resolved_runs[0]
    assert nested["run"]["output_dir"] == "outputs/debug-nested-001"
    assert nested["model"]["granularities"] == ["s", "m", "l", "xl"]
    assert nested["training"]["granularity_sampling"] == "all"

    standalone_s = resolved_runs[1]
    assert standalone_s["run"]["model_family"] == "standalone"
    assert standalone_s["run"]["granularity"] == "s"
    assert standalone_s["model"]["granularities"] == ["s"]

    for resolved in resolved_runs:
        validate_run_config(resolved)


def test_cli_overrides_are_parsed_and_applied():
    resolved = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        overrides=[
            "training.max_steps=7",
            "run.seed=123",
            "outputs.save_checkpoints=false",
        ],
    )

    assert resolved["training"]["max_steps"] == 7
    assert resolved["run"]["seed"] == 123
    assert resolved["outputs"]["save_checkpoints"] is False


def test_granularity_sampling_mode_validation():
    random_sampling = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        overrides=["training.granularity_sampling=random"],
    )
    assert random_sampling["training"]["granularity_sampling"] == "random"

    with pytest.raises(ConfigError, match="granularity_sampling"):
        resolve_run_config(
            "configs/debug_matrix.yaml",
            run_id="debug-nested-001",
            overrides=["training.granularity_sampling=cyclic"],
        )


def test_write_resolved_config(tmp_path):
    output_dir = tmp_path / "dmodel256-pilot-comparison-001"
    resolved = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        output_dir=output_dir,
    )

    config_path = write_resolved_config(resolved)

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["run"]["run_id"] == "dmodel256-pilot-comparison-001"
    assert saved["run"]["model_shape_label"] == "dmodel256"
    assert saved["run"]["table_reference_label"] == "matlm_78m"
    assert saved["run"]["sampling_mode"] == "nested-random"
    assert saved["run"]["completion_label"] == "reduced-token-pilot"
    assert config_path == output_dir / "config.json"


def test_dmodel256_completion_label_validation():
    resolved = resolve_run_config("configs/dmodel256_pilot_comparison.yaml")
    validate_run_config(resolved)

    paper_budget = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        overrides=[
            "training.token_budget=10000000000",
            "run.completion_label=matlm-10b-budget-reference",
        ],
    )
    mislabeled = copy.deepcopy(paper_budget)
    mislabeled["run"]["completion_label"] = "reduced-token-pilot"

    with pytest.raises(ConfigError, match="matlm-10b-budget-reference"):
        validate_run_config(mislabeled)

    validate_run_config(paper_budget)


def test_standalone_requires_one_granularity():
    resolved = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-standalone-s-001",
    )

    invalid = copy.deepcopy(resolved)
    invalid["model"]["granularities"] = ["s", "m"]

    with pytest.raises(ConfigError, match="exactly one"):
        validate_run_config(invalid)


def test_debug_matrix_resolves_all_standalone_granularities():
    expected = {
        "debug-standalone-s-001": ("s", 64),
        "debug-standalone-m-001": ("m", 128),
        "debug-standalone-l-001": ("l", 256),
        "debug-standalone-xl-001": ("xl", 512),
    }

    resolved_runs = resolve_all_run_configs("configs/debug_matrix.yaml")
    by_run_id = {config["run"]["run_id"]: config for config in resolved_runs}

    assert set(expected).issubset(by_run_id)
    for run_id, (granularity, intermediate_size) in expected.items():
        resolved = by_run_id[run_id]
        assert resolved["run"]["model_family"] == "standalone"
        assert resolved["run"]["granularity"] == granularity
        assert resolved["model"]["granularities"] == [granularity]
        assert resolved["model"]["intermediate_size"] == intermediate_size
        assert resolved["model"]["matformer_source_intermediate_size"] == 512
        assert resolved["run"]["output_dir"] == f"outputs/{run_id}"
        validate_run_config(resolved)


def test_debug_standalone_granularity_must_match_model_granularities():
    resolved = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-standalone-m-001",
    )

    invalid = copy.deepcopy(resolved)
    invalid["model"]["granularities"] = ["s"]

    with pytest.raises(ConfigError, match="exactly one matching granularity"):
        validate_run_config(invalid)


def test_matrix_output_root_override_derives_each_run_directory(tmp_path):
    output_root = tmp_path / "matrix-output"

    resolved_runs = resolve_all_run_configs(
        "configs/debug_matrix.yaml",
        overrides=[f"run.output_root={output_root}"],
    )

    assert output_root.is_dir()
    for resolved in resolved_runs:
        run = resolved["run"]
        assert run["output_root"] == str(output_root)
        assert run["output_dir"] == str(output_root / run["run_id"])


def test_single_run_defaults_to_outputs_root(tmp_path):
    config_path = _write_single_run_config(tmp_path)

    resolved = resolve_run_config(config_path)

    assert resolved["run"]["output_root"] == "outputs"
    assert resolved["run"]["output_dir"] == "outputs/single-output-root-001"


def test_single_run_output_root_override_derives_output_dir(tmp_path):
    config_path = _write_single_run_config(tmp_path)
    output_root = tmp_path / "single-output"

    resolved = resolve_run_config(
        config_path,
        overrides=[f"run.output_root={output_root}"],
    )

    assert output_root.is_dir()
    assert resolved["run"]["output_root"] == str(output_root)
    assert resolved["run"]["output_dir"] == str(
        output_root / "single-output-root-001"
    )


def test_explicit_output_dir_override_wins_over_output_root(tmp_path):
    output_root = tmp_path / "matrix-output"
    explicit_output_dir = tmp_path / "explicit-output" / "debug-nested-001"

    resolved = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        overrides=[f"run.output_root={output_root}"],
        output_dir=explicit_output_dir,
    )

    assert resolved["run"]["output_root"] == str(output_root)
    assert resolved["run"]["output_dir"] == str(explicit_output_dir)


def test_unwritable_output_root_fails_before_training(tmp_path):
    output_root = tmp_path / "blocked-output"
    output_root.mkdir()
    output_root.chmod(0o555)

    try:
        with pytest.raises(ConfigError, match="writ|permission|output root"):
            resolve_run_config(
                "configs/debug_matrix.yaml",
                run_id="debug-nested-001",
                overrides=[f"run.output_root={output_root}"],
            )
    finally:
        output_root.chmod(0o755)


def test_dmodel256_pilot_config_preserves_clarified_terms_and_shape_fields():
    resolved = resolve_run_config("configs/dmodel256_pilot_comparison.yaml")

    run = resolved["run"]
    model = resolved["model"]

    assert run["model_shape_label"] == "dmodel256"
    assert run["table_reference_label"] == "matlm_78m"
    assert run["sampling_mode"] == "nested-random"
    assert run["completion_label"] == "reduced-token-pilot"
    assert "model_size_label" not in run

    assert model["paper_alignment_claim"] is False
    assert "paper_aligned" not in model
    assert model["d_model"] == 256
    assert model["num_layers"] == 16
    assert model["num_attention_heads"] == 16
    assert model["context_length"] == 1024
    assert model["vocab_size_assumption"] == 256000
    assert model["granularity_prefixes"] == {
        "s": 0.125,
        "m": 0.25,
        "l": 0.5,
        "xl": 1.0,
    }
    assert any("SwiGLU" in note for note in model["mismatch_notes"])
    assert any("LM-head" in note for note in model["mismatch_notes"])

    assert resolved["training"]["token_budget"] < 10_000_000_000
    assert resolved["training"]["granularity_sampling"] == "random"
    validate_run_config(resolved)


def test_dmodel256_reduced_budget_rejects_table_budget_reference_label():
    resolved = resolve_run_config("configs/dmodel256_pilot_comparison.yaml")

    mislabeled = copy.deepcopy(resolved)
    mislabeled["run"]["completion_label"] = "matlm-10b-budget-reference"

    with pytest.raises(ConfigError, match="reduced-token-pilot"):
        validate_run_config(mislabeled)


def test_dmodel256_sampling_mode_must_match_granularity_sampling():
    random_sampling = resolve_run_config("configs/dmodel256_pilot_comparison.yaml")
    assert random_sampling["run"]["sampling_mode"] == "nested-random"
    assert random_sampling["training"]["granularity_sampling"] == "random"

    nested_all = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        overrides=[
            "run.run_id=dmodel256-pilot-nested-all-001",
            "run.sampling_mode=nested-all",
            "training.granularity_sampling=all",
        ],
    )
    assert nested_all["run"]["sampling_mode"] == "nested-all"
    assert nested_all["training"]["granularity_sampling"] == "all"

    with pytest.raises(ConfigError, match="sampling_mode"):
        resolve_run_config(
            "configs/dmodel256_pilot_comparison.yaml",
            overrides=[
                "run.sampling_mode=nested-all",
                "training.granularity_sampling=random",
            ],
        )


def test_dmodel256_pilot_derives_training_length_with_default_world_size(tmp_path, monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    output_root = tmp_path / "pilot-output"

    resolved = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        overrides=[f"run.output_root={output_root}"],
    )

    training = resolved["training"]
    expected_tokens_per_step = (
        training["batch_size_per_process"]
        * resolved["model"]["context_length"]
    )
    expected_steps = math.ceil(training["token_budget"] / expected_tokens_per_step)

    for field_name in [
        "effective_world_size",
        "expected_tokens_per_step",
        "derived_max_steps",
    ]:
        assert field_name in training
    assert training["effective_world_size"] == 1
    assert training["expected_tokens_per_step"] == expected_tokens_per_step
    assert training["derived_max_steps"] == expected_steps
    assert training["max_steps"] == expected_steps


def test_dmodel256_pilot_derives_training_length_from_distributed_world_size(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("WORLD_SIZE", "4")
    output_root = tmp_path / "pilot-output"

    resolved = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        overrides=[f"run.output_root={output_root}"],
    )

    training = resolved["training"]
    expected_tokens_per_step = (
        training["batch_size_per_process"]
        * resolved["model"]["context_length"]
        * 4
    )
    expected_steps = math.ceil(training["token_budget"] / expected_tokens_per_step)

    for field_name in [
        "effective_world_size",
        "expected_tokens_per_step",
        "derived_max_steps",
    ]:
        assert field_name in training
    assert training["effective_world_size"] == 4
    assert training["expected_tokens_per_step"] == expected_tokens_per_step
    assert training["derived_max_steps"] == expected_steps
    assert training["max_steps"] == expected_steps
