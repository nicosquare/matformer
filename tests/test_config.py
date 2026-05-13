import copy
import json

import pytest

from utils.config import (
    ConfigError,
    resolve_all_run_configs,
    resolve_run_config,
    validate_run_config,
    write_resolved_config,
)


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


def test_write_resolved_config(tmp_path):
    output_dir = tmp_path / "78m-reduced-pilot-001"
    resolved = resolve_run_config(
        "configs/78m_reduced_pilot.yaml",
        output_dir=output_dir,
    )

    config_path = write_resolved_config(resolved)

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["run"]["run_id"] == "78m-reduced-pilot-001"
    assert saved["run"]["completion_label"] == "reduced-token-pilot"
    assert config_path == output_dir / "config.json"


def test_78m_completion_label_validation():
    resolved = resolve_run_config("configs/78m_reduced_pilot.yaml")
    validate_run_config(resolved)

    mislabeled = copy.deepcopy(resolved)
    mislabeled["training"]["token_budget"] = 10_000_000_000

    with pytest.raises(ConfigError, match="paper-budget-complete"):
        validate_run_config(mislabeled)

    mislabeled["run"]["completion_label"] = "paper-budget-complete"
    validate_run_config(mislabeled)


def test_standalone_requires_one_granularity():
    resolved = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-standalone-s-001",
    )

    invalid = copy.deepcopy(resolved)
    invalid["model"]["granularities"] = ["s", "m"]

    with pytest.raises(ConfigError, match="exactly one"):
        validate_run_config(invalid)
