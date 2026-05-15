import os
import re
import subprocess
from pathlib import Path

from utils.config import resolve_run_config, validate_run_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _capture_78m_pilot_invocation(tmp_path, extra_args, env_updates=None):
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
        ["bash", "scripts/run_78m_pilot.sh", *extra_args],
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


def _read_slurm_78m_script():
    return (REPO_ROOT / "scripts" / "slurm_78m_pilot.sh").read_text(
        encoding="utf-8"
    )


def _sbatch_option_value(script_text, option):
    for line in script_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#SBATCH"):
            continue

        payload = stripped.removeprefix("#SBATCH").strip()
        if payload.startswith(f"{option}="):
            return payload.split("=", 1)[1]

        parts = payload.split()
        if parts and parts[0] == option and len(parts) > 1:
            return parts[1]

    return None


def _resource_count(value):
    if value is None:
        return None
    match = re.search(r"(\d+)$", value.strip())
    return int(match.group(1)) if match else None


def test_78m_reduced_pilot_resolves_paper_aligned_config(tmp_path):
    output_root = tmp_path / "pilot-output"
    config = resolve_run_config(
        "configs/78m_reduced_pilot.yaml",
        overrides=[f"run.output_root={output_root}"],
    )

    assert config["run"]["run_id"] == "78m-reduced-pilot-001"
    assert config["run"]["phase_id"] == "78m_pilot"
    assert config["run"]["model_family"] == "nested"
    assert config["run"]["model_size_label"] == "78m"
    assert config["run"]["completion_label"] == "reduced-token-pilot"
    assert config["run"]["output_dir"] == str(output_root / "78m-reduced-pilot-001")

    assert config["model"]["paper_aligned"] is True
    assert config["model"]["num_layers"] == 16
    assert config["model"]["num_attention_heads"] == 16
    assert config["model"]["context_length"] == 1024
    assert config["model"]["vocab_size_assumption"] == 256000
    assert config["model"]["tokenizer_name"] == "hf-internal-testing/llama-tokenizer"
    assert config["model"]["tokenizer_name"] != config["model"]["base_model_name"]
    assert config["dataset"]["dataset_name"] == "HuggingFaceFW/fineweb"
    assert config["dataset"]["dataset_config_name"] == "sample-10BT"
    assert config["training"]["token_budget"] < config["training"]["paper_token_budget"]
    assert config["training"]["paper_token_budget"] == 10_000_000_000

    validate_run_config(config)


def test_78m_pilot_runner_propagates_output_root_env(tmp_path):
    output_root = tmp_path / "pilot-output"

    args = _capture_78m_pilot_invocation(
        tmp_path,
        ["--override", "training.max_steps_cap=1"],
        env_updates={"OUTPUT_ROOT": str(output_root)},
    )

    assert args[0] == "train.py"
    assert _has_arg_pair(args, "--config", "configs/78m_reduced_pilot.yaml")
    assert _has_arg_pair(args, "--run-id", "78m-reduced-pilot-001")
    assert _has_arg_pair(args, "--output-root", str(output_root))
    assert _has_arg_pair(args, "--override", "training.max_steps_cap=1")


def test_78m_pilot_runner_forwards_explicit_arguments(tmp_path):
    output_root = tmp_path / "pilot-output"
    output_dir = tmp_path / "explicit-output" / "78m-reduced-pilot-001"

    args = _capture_78m_pilot_invocation(
        tmp_path,
        [
            "--config",
            "configs/78m_reduced_pilot.yaml",
            "--run-id",
            "78m-reduced-pilot-001",
            "--output-root",
            str(output_root),
            "--output-dir",
            str(output_dir),
            "--override",
            "training.max_steps_cap=1",
        ],
        env_updates={"OUTPUT_ROOT": str(tmp_path / "ignored-env-output")},
    )

    assert _has_arg_pair(args, "--config", "configs/78m_reduced_pilot.yaml")
    assert _has_arg_pair(args, "--run-id", "78m-reduced-pilot-001")
    assert _has_arg_pair(args, "--output-root", str(output_root))
    assert args.count("--output-root") == 1
    assert _has_arg_pair(args, "--output-dir", str(output_dir))
    assert _has_arg_pair(args, "--override", "training.max_steps_cap=1")


def test_slurm_78m_pilot_requests_single_node_multi_gpu_resources():
    script_text = _read_slurm_78m_script()

    node_count = _sbatch_option_value(script_text, "--nodes") or _sbatch_option_value(
        script_text,
        "-N",
    )
    gpu_count = _resource_count(
        _sbatch_option_value(script_text, "--gpus-per-node")
        or _sbatch_option_value(script_text, "--gpus")
    )

    assert node_count == "1"
    assert gpu_count is not None and gpu_count > 1


def test_slurm_78m_pilot_launches_one_training_process_per_gpu():
    script_text = _read_slurm_78m_script()

    assert "torch.distributed.run" in script_text or "torchrun" in script_text
    assert "--nproc_per_node" in script_text or "--nproc-per-node" in script_text
    assert any(
        variable_name in script_text
        for variable_name in [
            "SLURM_GPUS_ON_NODE",
            "SLURM_GPUS_PER_NODE",
            "GPUS_PER_NODE",
        ]
    )


def test_slurm_78m_pilot_wrapper_forwards_to_runner(tmp_path):
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
            "scripts/slurm_78m_pilot.sh",
            "--output-root",
            str(output_root),
            "--run-id",
            "78m-reduced-pilot-001",
            "--override",
            "training.max_steps_cap=1",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    args = argv_path.read_text(encoding="utf-8").splitlines()
    assert args[0] == "train.py"
    assert _has_arg_pair(args, "--config", "configs/78m_reduced_pilot.yaml")
    assert _has_arg_pair(args, "--run-id", "78m-reduced-pilot-001")
    assert _has_arg_pair(args, "--output-root", str(output_root))
    assert _has_arg_pair(args, "--override", "training.max_steps_cap=1")


def test_slurm_78m_pilot_wrapper_rejects_direct_execution(tmp_path):
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
            "scripts/slurm_78m_pilot.sh",
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
