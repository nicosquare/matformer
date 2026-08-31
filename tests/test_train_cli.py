import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_preflight(*args: str) -> dict:
    result = subprocess.run(
        [sys.executable, "train.py", *args, "--preflight"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


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
