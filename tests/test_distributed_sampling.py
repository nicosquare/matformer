from __future__ import annotations

import pytest

from src.utils.config import ConfigError, resolve_run_config


def test_per_granularity_scope_rejects_multi_process_world_size(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")

    with pytest.raises(ConfigError, match="per_granularity requires one process"):
        resolve_run_config(
            "tests/fixtures/per_granularity_optimizer_smoke.yaml",
            overrides={"training.distributed.expected_world_size": 2},
        )


def test_shared_scope_preserves_existing_multi_process_resolution(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")

    resolved = resolve_run_config(
        "tests/fixtures/per_granularity_optimizer_smoke.yaml",
        overrides={
            "training.optimizer.state_scope": "shared",
            "training.distributed.expected_world_size": 2,
        },
    )

    assert resolved["training"]["optimizer_state_scope"] == "shared"
    assert resolved["training"]["effective_world_size"] == 2
    assert resolved["training"]["distributed"]["expected_world_size"] == 2
