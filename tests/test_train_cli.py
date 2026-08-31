import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_preflight(*args: str) -> dict:
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    result = subprocess.run(
        [sys.executable, "train.py", *args, "--preflight"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(result.stdout)


def _run_preflight_failure(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    return subprocess.run(
        [sys.executable, "train.py", *args, "--preflight"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_normal_cpu_cli_resume_matches_uninterrupted_ownership_trace(
    tmp_path, monkeypatch
):
    import csv
    import train as train_module
    import src.training.data as training_data
    import src.training.modeling as training_modeling

    class CliResumeModel(torch.nn.Module):
        def __init__(self, fail_forward=None):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
            self.current_granularity = None
            self.fail_forward = fail_forward
            self.training_forwards = 0

        def configure_subnetwork(self, granularity):
            self.current_granularity = granularity

        def forward(self, input_ids, attention_mask=None, labels=None):
            if self.training:
                self.training_forwards += 1
                if self.training_forwards == self.fail_forward:
                    raise RuntimeError("simulated CLI resume interruption")
            scale = 1.0 if self.current_granularity == "narrow" else 2.0
            return SimpleNamespace(
                loss=self.weight.pow(2) * scale + input_ids.float().mean() * 0.0
            )

    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3] for _ in range(8)],
            "attention_mask": [[1, 1, 1] for _ in range(8)],
        }
    )
    build_count = {"value": 0}

    def build_model(_config):
        build_count["value"] += 1
        return CliResumeModel(fail_forward=3 if build_count["value"] == 2 else None)

    monkeypatch.setattr(training_modeling, "build_model", build_model)
    monkeypatch.setattr(training_modeling, "load_tokenizer", lambda _config: object())
    monkeypatch.setattr(
        training_data,
        "load_and_tokenize_dataset",
        lambda *_args, **_kwargs: dataset,
    )

    run_id = "per-granularity-optimizer-smoke-001"
    common = [
        "--config",
        "tests/fixtures/per_granularity_optimizer_smoke.yaml",
        "--override",
        "training.max_steps=4",
        "--override",
        "training.token_budget=128",
        "--override",
        "training.batch_size_per_process=1",
        "--override",
        "training.scheduler.name=constant",
        "--override",
        "training.eval_interval=0",
        "--override",
        "evaluation.validation=false",
        "--override",
        "evaluation.validation.interval_steps=0",
        "--override",
        "evaluation.validation.run_at_completion=false",
        "--override",
        "run.continuation.latest_checkpoint_save_interval_steps=1",
    ]
    uninterrupted_dir = tmp_path / "uninterrupted" / run_id
    resumed_dir = tmp_path / "resumed" / run_id

    train_module.main([*common, "--output-dir", str(uninterrupted_dir)])
    with pytest.raises(RuntimeError, match="CLI resume interruption"):
        train_module.main([*common, "--output-dir", str(resumed_dir)])
    train_module.main([*common, "--output-dir", str(resumed_dir)])

    def committed_trace(output_dir):
        with (output_dir / "metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as source:
            rows = [
                row
                for row in csv.DictReader(source)
                if row["split"] == "train"
                and row["optimizer_step_committed"] == "True"
            ]
        return [
            (
                row["step"],
                row["optimizer_action_id"],
                row["selected_optimizer_granularity"],
                row["learning_rate"],
                row["global_scheduler_position"],
                row["optimizer_successful_update_counts"],
                row["optimizer_exposure_counts"],
            )
            for row in rows
        ]

    assert committed_trace(resumed_dir) == committed_trace(uninterrupted_dir)
    assert len(committed_trace(resumed_dir)) == 4
    resumed_summary = json.loads(
        (resumed_dir / "run_summary.json").read_text(encoding="utf-8")
    )
    assert resumed_summary["continuation_state"]["resume_count"] == 1
    assert resumed_summary["optimizer_accounting_reconciled"] is True
    assert resumed_summary["optimizer_successful_update_counts"] == {
        "narrow": 2,
        "full": 2,
    }


def test_preflight_reports_compact_pre_nested_warmup_contract(tmp_path):
    payload = _run_preflight(
        "--config",
        "tests/fixtures/probabilistic_adaptive_global_smoke.yaml",
        "--output-dir",
        str(tmp_path / "probabilistic-adaptive-global-smoke-001"),
    )

    assert payload["pre_nested_warmup"] == {
        "active": False,
        "duration": 0,
        "enabled": False,
        "policy": "full_only",
        "unit": "steps",
    }


def test_preflight_reports_balanced_warmup_without_full_schedule(tmp_path):
    payload = _run_preflight(
        "--config",
        "tests/fixtures/probabilistic_adaptive_global_smoke.yaml",
        "--output-dir",
        str(tmp_path / "probabilistic-adaptive-global-smoke-001"),
        "--override",
        "training.pre_nested_warmup.enabled=true",
        "--override",
        "training.pre_nested_warmup.duration=12",
        "--override",
        "training.pre_nested_warmup.unit=steps",
        "--override",
        "training.pre_nested_warmup.policy=balanced_global",
    )

    assert payload["pre_nested_warmup"] == {
        "action_interval_steps": 2,
        "active": True,
        "controller_start_step": 12,
        "duration": 12,
        "enabled": True,
        "passes": 2,
        "policy": "balanced_global",
        "unit": "steps",
    }
    assert "schedule" not in payload["pre_nested_warmup"]


def test_preflight_reports_complete_optimizer_state_identity(tmp_path):
    payload = _run_preflight(
        "--config",
        "tests/fixtures/per_granularity_optimizer_smoke.yaml",
        "--output-dir",
        str(tmp_path / "per-granularity-optimizer-smoke-001"),
    )

    assert payload["optimizer_state_scope"] == "per_granularity"
    assert payload["optimizer_scheduler_clock"] == "global_step"
    assert payload["optimizer_state_contract"]["ordered_granularities"] == [
        "narrow",
        "full",
    ]
    assert payload["optimizer_state_contract"]["optimizer_name"] == "adamw"
    assert payload["optimizer_state_contract"]["scheduler_contract"] == payload[
        "scheduler"
    ]
    assert payload["sampling_policy"] == {
        "mode": "global",
        "schedule": "balanced_cycle",
        "schedule_version": 1,
        "interval_steps": 1,
    }
    assert payload["data_roles"] == {
        "data_roles_manifest_hash": None,
        "optimizer_training_manifest_hash": None,
        "controller_manifest_hash": None,
        "ordinary_validation_manifest_hash": None,
        "final_holdout_manifest_hash": None,
    }
    assert payload["run_budget"] == {
        "token_budget": 256,
        "global_steps": 4,
        "expected_tokens_per_step": 64,
    }
    assert len(payload["full_run_signature"]) == 64
    assert len(payload["paired_control_signature"]) == 64


def test_preflight_honors_shared_scope_and_global_clock_overrides(tmp_path):
    payload = _run_preflight(
        "--config",
        "tests/fixtures/per_granularity_optimizer_smoke.yaml",
        "--output-dir",
        str(tmp_path / "per-granularity-optimizer-smoke-001"),
        "--override",
        "training.optimizer.state_scope=shared",
        "--override",
        "training.optimizer.scheduler_clock=global_step",
    )

    assert payload["optimizer_state_scope"] == "shared"
    assert payload["optimizer_scheduler_clock"] == "global_step"
    assert payload["optimizer_state_contract"]["state_scope"] == "shared"


@pytest.mark.parametrize(
    "fixture_path, run_id, overrides, expected_mode",
    [
        (
            "tests/fixtures/per_granularity_optimizer_smoke.yaml",
            "per-granularity-optimizer-smoke-001",
            [],
            "global",
        ),
        (
            "tests/fixtures/per_granularity_optimizer_smoke.yaml",
            "per-granularity-optimizer-smoke-001",
            ["model.global_sampling_schedule=random_with_replacement"],
            "global",
        ),
        (
            "tests/fixtures/experiment_config_resolution.yaml",
            "experiment-config-resolution-001",
            [
                "run.sampling_mode=nested-random",
                "model.granularity_sampling_mode=fixed_global",
                "model.global_sampling_distribution={s: 0.1, m: 0.2, l: 0.3, xl: 0.4}",
                "training.optimizer.state_scope=per_granularity",
            ],
            "fixed_global",
        ),
        (
            "tests/fixtures/probabilistic_adaptive_global_smoke.yaml",
            "probabilistic-adaptive-global-smoke-001",
            ["training.optimizer.state_scope=per_granularity"],
            "adaptive_global",
        ),
    ],
)
def test_preflight_reports_eligible_single_owner_policy_identity(
    tmp_path,
    fixture_path,
    run_id,
    overrides,
    expected_mode,
):
    args = [
        "--config",
        fixture_path,
        "--output-dir",
        str(tmp_path / run_id),
    ]
    for override in overrides:
        args.extend(["--override", override])
    payload = _run_preflight(*args)

    assert payload["resolved_sampling_mode"] == expected_mode
    assert payload["optimizer_state_eligibility"]["eligible"] is True
    assert payload["optimizer_state_eligibility"]["required_action_kind"] == "global"
    assert payload["optimizer_state_eligibility"]["required_action_cardinality"] == 1


def test_invalid_optimizer_ownership_preflight_fails_before_output_mutation(tmp_path):
    output_dir = tmp_path / "experiment-config-resolution-001"
    result = _run_preflight_failure(
        "--config",
        "tests/fixtures/experiment_config_resolution.yaml",
        "--output-dir",
        str(output_dir),
        "--override",
        "training.optimizer.state_scope=per_granularity",
    )

    assert result.returncode != 0
    assert result.stderr.startswith("Preflight configuration error:")
    assert "per_granularity requires run.sampling_mode=nested-random" in result.stderr
    assert not output_dir.exists()
