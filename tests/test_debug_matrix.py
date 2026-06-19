import os
import subprocess
from pathlib import Path

from src.utils.config import resolve_all_run_configs, resolve_run_config, validate_run_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _capture_debug_matrix_invocation(tmp_path, extra_args, env_updates=None):
    recorder = tmp_path / "python-recorder.sh"
    argv_path = tmp_path / "argv.txt"
    recorder.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$ARGV_FILE\"\n",
        encoding="utf-8",
    )
    recorder.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(recorder),
            "ARGV_FILE": str(argv_path),
        }
    )
    if env_updates:
        env.update(env_updates)

    subprocess.run(
        ["bash", "scripts/run_debug_matrix.sh", *extra_args],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return argv_path.read_text(encoding="utf-8").splitlines()


def _has_arg_pair(args, flag, value):
    return any(
        args[index] == flag and args[index + 1] == value
        for index in range(len(args) - 1)
    )


def _has_arg(args, flag):
    return flag in args


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
    assert nested["run"]["sampling_mode"] == "nested-all"
    assert nested["model"]["granularities"] == ["s", "m", "l", "xl"]
    assert standalone_s["run"]["model_family"] == "standalone"
    assert standalone_s["run"]["sampling_mode"] == "standalone"
    assert standalone_s["run"]["granularity"] == "s"
    assert standalone_s["model"]["granularities"] == ["s"]


def test_debug_matrix_runner_propagates_output_root_env(tmp_path):
    output_root = tmp_path / "external-output"

    args = _capture_debug_matrix_invocation(
        tmp_path,
        ["--override", "training.max_steps=1"],
        env_updates={"OUTPUT_ROOT": str(output_root)},
    )

    assert args[:2] == ["-m", "src.training.baselines"]
    assert _has_arg_pair(args, "--config", "configs/debug_matrix.yaml")
    assert _has_arg_pair(args, "--nested-run-id", "debug-nested-001")
    assert not _has_arg(args, "--granularity")
    assert _has_arg_pair(args, "--output-root", str(output_root))
    assert _has_arg_pair(args, "--override", "training.max_steps=1")


def test_debug_matrix_runner_can_select_baseline_granularities(tmp_path):
    args = _capture_debug_matrix_invocation(
        tmp_path,
        ["--override", "training.max_steps=1"],
        env_updates={"BASELINE_GRANULARITIES": "m,xl"},
    )

    assert _has_arg_pair(args, "--granularity", "m")
    assert _has_arg_pair(args, "--granularity", "xl")
    assert _has_arg_pair(args, "--override", "training.max_steps=1")


def test_debug_matrix_runner_forwards_output_arguments(tmp_path):
    output_root = tmp_path / "cli-output"
    explicit_output_dir = tmp_path / "explicit-output" / "debug-nested-001"

    args = _capture_debug_matrix_invocation(
        tmp_path,
        [
            "--output-root",
            str(output_root),
            "--output-dir",
            str(explicit_output_dir),
            "--override",
            "training.max_steps=1",
        ],
    )

    assert args[:2] == ["-m", "src.training.baselines"]
    assert _has_arg_pair(args, "--output-root", str(output_root))
    assert _has_arg_pair(args, "--output-dir", str(explicit_output_dir))
    assert _has_arg_pair(args, "--override", "training.max_steps=1")


def test_slurm_debug_matrix_wrapper_forwards_to_runner(tmp_path):
    recorder = tmp_path / "python-recorder.sh"
    argv_path = tmp_path / "argv.txt"
    output_root = tmp_path / "slurm-output"

    recorder.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$ARGV_FILE\"\n",
        encoding="utf-8",
    )
    recorder.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "ALLOW_LOCAL_SLURM_WRAPPER": "1",
            "PYTHON_BIN": str(recorder),
            "ARGV_FILE": str(argv_path),
        }
    )

    subprocess.run(
        [
            "bash",
            "scripts/slurm_debug_matrix.sh",
            "--output-root",
            str(output_root),
            "--baseline-granularity",
            "m",
            "--nested-run-id",
            "debug-nested-001",
            "--override",
            "training.max_steps=1",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    args = argv_path.read_text(encoding="utf-8").splitlines()
    assert args[:2] == ["-m", "src.training.baselines"]
    assert _has_arg_pair(args, "--config", "configs/debug_matrix.yaml")
    assert _has_arg_pair(args, "--nested-run-id", "debug-nested-001")
    assert _has_arg_pair(args, "--granularity", "m")
    assert _has_arg_pair(args, "--output-root", str(output_root))
    assert _has_arg_pair(args, "--override", "training.max_steps=1")


def test_slurm_debug_matrix_wrapper_rejects_direct_execution(tmp_path):
    recorder = tmp_path / "python-recorder.sh"
    argv_path = tmp_path / "argv.txt"

    recorder.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$ARGV_FILE\"\n",
        encoding="utf-8",
    )
    recorder.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(recorder),
            "ARGV_FILE": str(argv_path),
            "SLURM_JOB_ID": "123",
        }
    )
    env.pop("ALLOW_LOCAL_SLURM_WRAPPER", None)
    env.pop("SLURM_SUBMIT_DIR", None)
    env.pop("SLURM_JOB_NAME", None)

    result = subprocess.run(
        [
            "bash",
            "scripts/slurm_debug_matrix.sh",
            "--output-root",
            str(tmp_path / "slurm-output"),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "intended for sbatch" in result.stderr
    assert not argv_path.exists()
