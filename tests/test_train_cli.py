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
