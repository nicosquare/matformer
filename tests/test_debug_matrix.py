from utils.config import resolve_all_run_configs, resolve_run_config, validate_run_config


def test_debug_nested_run_resolves_phase3_p1_contract(tmp_path):
    output_dir = tmp_path / "debug-nested-001"

    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
    )

    assert config["run"]["phase_id"] == "debug_matrix"
    assert config["run"]["run_id"] == "debug-nested-001"
    assert config["run"]["model_family"] == "nested"
    assert config["run"]["model_size_label"] == "debug"
    assert config["run"]["completion_label"] == "debug"
    assert config["run"]["output_dir"] == str(output_dir)

    assert config["model"]["paper_aligned"] is False
    assert config["model"]["granularities"] == ["s", "m", "l", "xl"]
    assert config["dataset"]["dataset_phase"] == "debug"
    assert config["evaluation"]["validation"] is True

    validate_run_config(config)


def test_debug_matrix_exposes_nested_run_and_one_phase3_baseline():
    resolved_runs = resolve_all_run_configs("configs/debug_matrix.yaml")
    by_run_id = {config["run"]["run_id"]: config for config in resolved_runs}

    nested = by_run_id["debug-nested-001"]
    standalone_s = by_run_id["debug-standalone-s-001"]

    assert nested["run"]["model_family"] == "nested"
    assert nested["model"]["granularities"] == ["s", "m", "l", "xl"]
    assert standalone_s["run"]["model_family"] == "standalone"
    assert standalone_s["run"]["granularity"] == "s"
    assert standalone_s["model"]["granularities"] == ["s"]
