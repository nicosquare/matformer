import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.evaluation.reporting import (
    generate_figures,
    generate_gradient_interference_figures,
    plot_gradient_interference_cosine_heatmaps,
    plot_gradient_interference_cosine_trajectories,
)
from src.evaluation.reporting_io import (
    GradientInterferenceReportingError,
    iter_gradient_interference_histories,
)
from src.training.gradient_interference import snapshot_id


GRANULARITIES = tuple(f"g{fraction}" for fraction in range(125, 1001, 125))
STEPS = (0, 1, 2, 3, 4, 5)
TOKENS = (0, 4_000_000, 8_000_000, 12_000_000, 16_000_000, 20_000_000)
CONTRACT_HASH = "d" * 64
PROBE_HASH = "p" * 64
SUPPORT_HASH = "s" * 64


def _diagnostic_config(
    run_dir: Path,
    *,
    run_id: str,
    h: int,
    seed: int,
    schedule: str = "random_with_replacement",
) -> dict:
    milestones = [
        {"step": step, "reasons": [f"trajectory_fraction:{index / 5:g}"]}
        for index, step in enumerate(STEPS)
    ]
    return {
        "run": {
            "run_id": run_id,
            "seed": seed,
            "sampling_mode": "nested-random",
            "output_dir": str(run_dir),
        },
        "model": {
            "variant": "slicing",
            "correction_mode": "none",
            "membership_correction": False,
            "granularity_sampling_mode": "global",
            "global_sampling_schedule": schedule,
            "global_sampling_schedule_version": (
                1 if schedule == "balanced_cycle" else None
            ),
            "global_sampling_interval_steps": h,
            "granularity_pattern_provenance": {
                "policy": "uniform_global_window",
                "global_sampling_interval_steps": h,
            },
            "granularities": list(GRANULARITIES),
            "granularity_prefix_widths": {
                label: index * 128
                for index, label in enumerate(GRANULARITIES, start=1)
            },
            "d_model": 256,
            "num_layers": 16,
        },
        "training": {
            "token_budget": TOKENS[-1],
            "max_steps": STEPS[-1],
            "optimizer": {"name": "adamw", "kwargs": {"weight_decay": 0.1}},
            "scheduler": {"name": "cosine"},
            "expected_tokens_per_step": 4_000_000,
        },
        "evaluation": {
            "gradient_interference": {
                "enabled": True,
                "artifact_path": "gradient_interference.jsonl",
                "resolved_steps": list(STEPS),
                "resolved_milestones": milestones,
                "diagnostic_contract_hash": CONTRACT_HASH,
                "fixed_probe_manifest_hash": PROBE_HASH,
                "controlled_support_hash": SUPPORT_HASH,
                "gradient_semantics": "raw_pre_correction_pre_clipping",
                "loss_aggregation": "target_token_weighted_fixed_probe",
                "shared_support": "smaller_nested_controlled_ffn_support",
                "layerwise": True,
            }
        },
        "data_roles_manifest_hash": "roles",
        "optimizer_training_manifest_hash": "training-role",
        "validation_manifest_hash": "validation-role",
        "controller_manifest_hash": PROBE_HASH,
        "final_holdout_manifest_hash": "holdout-role",
    }


def _snapshot(config: dict, milestone_index: int) -> dict:
    step = STEPS[milestone_index]
    pairs = []
    pair_index = 0
    for left_index, left in enumerate(GRANULARITIES):
        for right in GRANULARITIES[left_index + 1 :]:
            zero_norm = pair_index == 27 and milestone_index == 2
            cosine = None if zero_norm else -0.8 + 0.05 * pair_index + 0.02 * milestone_index
            pairs.append(
                {
                    "left_granularity": left,
                    "right_granularity": right,
                    "distance": 1.0,
                    "shared_parameter_count": 128,
                    "dot_product": 1.0,
                    "left_shared_squared_norm": 1.0,
                    "right_shared_squared_norm": 1.0,
                    "left_shared_norm": 0.0 if zero_norm else 1.0,
                    "right_shared_norm": 1.0,
                    "cosine": cosine,
                    "has_zero_norm": zero_norm,
                    "zero_norm": {"left": zero_norm, "right": False},
                    "layer_contributions": [],
                }
            )
            pair_index += 1
    diagnostic = config["evaluation"]["gradient_interference"]
    return {
        "schema_version": 1,
        "event_type": "gradient_interference_snapshot",
        "snapshot_id": snapshot_id(config, step),
        "run_id": config["run"]["run_id"],
        "step": step,
        "tokens_seen": TOKENS[milestone_index],
        "milestone_reasons": diagnostic["resolved_milestones"][milestone_index][
            "reasons"
        ],
        "fixed_probe_manifest_hash": PROBE_HASH,
        "controlled_support_hash": SUPPORT_HASH,
        "diagnostic_contract_hash": CONTRACT_HASH,
        "semantics": {
            "gradient": diagnostic["gradient_semantics"],
            "loss_aggregation": diagnostic["loss_aggregation"],
            "shared_support": diagnostic["shared_support"],
            "layerwise": True,
        },
        "granularities": [
            {
                "granularity": label,
                "controlled_parameter_count": 128,
                "aggregate_loss": 1.0,
                "gradient_squared_norm": 1.0,
                "gradient_norm": 1.0,
            }
            for label in GRANULARITIES
        ],
        "pairs": pairs,
        "cost": {
            "packed_sequences": 128,
            "batches": 16,
            "targets": 512,
            "packed_sequence_evaluations": 1024,
            "target_evaluations": 4096,
            "backward_evaluations": 128,
            "duration_seconds": 1.0,
        },
    }


def _write_run(
    root: Path,
    *,
    run_id: str,
    h: int = 25,
    seed: int = 1,
    status: str = "completed",
    snapshot_count: int = 6,
    schedule: str = "random_with_replacement",
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir()
    config = _diagnostic_config(
        run_dir, run_id=run_id, h=h, seed=seed, schedule=schedule
    )
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    records = [_snapshot(config, index) for index in range(snapshot_count)]
    journal_payload = "".join(json.dumps(record) + "\n" for record in records)
    journal_path = run_dir / "gradient_interference.jsonl"
    journal_path.write_text(journal_payload, encoding="utf-8")
    summary = {
        "run_id": run_id,
        "status": status,
        "tokens_seen": TOKENS[-1] if status == "completed" else TOKENS[snapshot_count - 1],
        "gradient_interference_path": str(journal_path),
        "gradient_interference_journal_hash": hashlib.sha256(
            journal_payload.encode("utf-8")
        ).hexdigest(),
        "gradient_interference_snapshot_count": snapshot_count,
        "gradient_interference_measured_steps": list(STEPS[:snapshot_count]),
        "gradient_interference_expected_steps": list(STEPS),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


def _rewrite_journal_and_hash(run_dir: Path, payload: str) -> None:
    journal_path = run_dir / "gradient_interference.jsonl"
    journal_path.write_text(payload, encoding="utf-8")
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["gradient_interference_journal_hash"] = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()
    summary_path.write_text(json.dumps(summary), encoding="utf-8")


def test_disabled_stray_and_incomplete_diagnostics_are_silent(tmp_path):
    disabled = tmp_path / "disabled"
    disabled.mkdir()
    (disabled / "config.json").write_text(
        json.dumps({"evaluation": {"gradient_interference": {"enabled": False}}}),
        encoding="utf-8",
    )
    (disabled / "gradient_interference.jsonl").write_text("stray\n", encoding="utf-8")
    _write_run(
        tmp_path,
        run_id="incomplete",
        status="failed",
        snapshot_count=2,
    )

    assert list(iter_gradient_interference_histories(tmp_path)) == []
    assert generate_gradient_interference_figures(tmp_path, tmp_path / "figures", dpi=20) == []


def test_completed_diagnostic_validation_reports_run_and_artifact(tmp_path):
    run_dir = _write_run(tmp_path, run_id="bad-hash")
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["gradient_interference_journal_hash"] = "0" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(
        GradientInterferenceReportingError,
        match=r"bad-hash.*gradient_interference.jsonl.*hash mismatch",
    ):
        list(iter_gradient_interference_histories(tmp_path))


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("invalid-json", "invalid JSON"),
        ("duplicate-snapshot", "duplicate or unexpected snapshot"),
        ("missing-milestone", "missing resolved milestones"),
        ("identity", "diagnostic_contract_hash mismatch"),
        ("pair-count", "pair cardinality mismatch"),
        ("non-finite", "non-finite"),
    ],
)
def test_completed_diagnostic_rejects_invalid_journal_content(
    tmp_path, failure, message
):
    run_dir = _write_run(tmp_path, run_id=failure)
    journal_path = run_dir / "gradient_interference.jsonl"
    records = [json.loads(line) for line in journal_path.read_text().splitlines()]
    if failure == "invalid-json":
        payload = "{invalid\n" + "".join(
            json.dumps(record) + "\n" for record in records[1:]
        )
    else:
        if failure == "duplicate-snapshot":
            records[1] = dict(records[0])
        elif failure == "missing-milestone":
            records.pop()
        elif failure == "identity":
            records[0]["diagnostic_contract_hash"] = "x" * 64
        elif failure == "pair-count":
            records[0]["pairs"].pop()
        elif failure == "non-finite":
            records[0]["pairs"][0]["cosine"] = float("nan")
        payload = "".join(json.dumps(record) + "\n" for record in records)
    _rewrite_journal_and_hash(run_dir, payload)

    with pytest.raises(GradientInterferenceReportingError, match=message):
        list(iter_gradient_interference_histories(tmp_path))


def test_completed_diagnostic_requires_journal_and_summary(tmp_path):
    missing_journal = _write_run(tmp_path, run_id="missing-journal")
    (missing_journal / "gradient_interference.jsonl").unlink()
    with pytest.raises(GradientInterferenceReportingError, match="journal is missing"):
        list(iter_gradient_interference_histories(missing_journal))

    missing_summary = _write_run(tmp_path, run_id="missing-summary")
    (missing_summary / "run_summary.json").unlink()
    with pytest.raises(GradientInterferenceReportingError, match="no run_summary.json"):
        list(iter_gradient_interference_histories(missing_summary))


def test_duplicate_contract_h_and_seed_is_rejected(tmp_path):
    _write_run(tmp_path, run_id="duplicate-a", h=5, seed=7)
    _write_run(tmp_path, run_id="duplicate-b", h=5, seed=7)

    with pytest.raises(GradientInterferenceReportingError, match=r"H=5, seed=7"):
        list(iter_gradient_interference_histories(tmp_path))


def test_h_sweep_generates_one_trajectory_and_one_heatmap_per_run(tmp_path):
    for h in (1, 5, 25, 50):
        _write_run(tmp_path, run_id=f"diagnostic-h{h}", h=h, seed=h)

    paths = generate_gradient_interference_figures(
        tmp_path, tmp_path / "figures", dpi=20
    )
    names = {path.name for path in paths}
    assert len(paths) == 5
    assert len(
        [name for name in names if name.startswith("gradient_interference_cosine_trajectories__")]
    ) == 1
    assert len(
        [name for name in names if name.startswith("gradient_interference_cosine_heatmaps__")]
    ) == 4
    assert all(path.is_file() for path in paths)


def test_canonical_generator_returns_diagnostics_and_only_cleans_owned_pngs(tmp_path):
    run_dir = _write_run(tmp_path, run_id="canonical", h=25, seed=3)
    journal_path = run_dir / "gradient_interference.jsonl"
    journal_before = journal_path.read_bytes()
    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    stale_path = output_dir / "gradient_interference_obsolete.png"
    stale_path.write_bytes(b"stale")

    paths = generate_figures(
        tmp_path,
        output_dir,
        refresh_counts=False,
        dpi=20,
        variants=["slicing"],
        corrections=["none"],
    )
    names = {path.name for path in paths}
    assert any(name.startswith("gradient_interference_cosine_trajectories__") for name in names)
    assert "gradient_interference_cosine_heatmaps__canonical.png" in names
    assert not stale_path.exists()
    assert journal_path.read_bytes() == journal_before


def test_balanced_gradient_histories_group_across_h_but_not_with_iid(tmp_path):
    _write_run(
        tmp_path,
        run_id="balanced-h1",
        h=1,
        schedule="balanced_cycle",
    )
    _write_run(
        tmp_path,
        run_id="balanced-h50",
        h=50,
        schedule="balanced_cycle",
    )
    _write_run(tmp_path, run_id="iid-h50", h=50)

    histories = list(iter_gradient_interference_histories(tmp_path))
    by_id = {history.run_id: history for history in histories}
    assert (
        by_id["balanced-h1"].comparison_contract
        == by_id["balanced-h50"].comparison_contract
    )
    assert (
        by_id["balanced-h1"].comparison_contract
        != by_id["iid-h50"].comparison_contract
    )


def test_trajectory_seed_means_and_heatmap_masks_are_deterministic(
    tmp_path, monkeypatch
):
    _write_run(tmp_path, run_id="seed-one", h=5, seed=1)
    _write_run(tmp_path, run_id="seed-two", h=5, seed=2)
    histories = list(iter_gradient_interference_histories(tmp_path))
    saved_figures = []
    monkeypatch.setattr(
        Figure,
        "savefig",
        lambda figure, _output_path, **_kwargs: saved_figures.append(figure),
    )

    plot_gradient_interference_cosine_trajectories(
        histories, tmp_path / "trajectories.png", dpi=20
    )
    trajectory_figure = saved_figures.pop()
    assert len(trajectory_figure.axes) == 28
    first_axis = trajectory_figure.axes[0]
    assert first_axis.get_title() == "g125 vs g250"
    assert first_axis.get_ylim() == (-1.0, 1.0)
    assert [line.get_alpha() for line in first_axis.lines].count(0.22) == 2
    assert any(line.get_label() == "H=5" and line.get_linewidth() == 2.0 for line in first_axis.lines)
    assert trajectory_figure.legends[0].get_title().get_text() == (
        "H = consecutive optimizer steps per sampled width"
    )
    assert trajectory_figure.legends[0]._loc == 9  # upper center, below the panels
    assert [text.get_text() for text in trajectory_figure.legends[0].get_texts()] == [
        "H=5"
    ]

    plot_gradient_interference_cosine_heatmaps(
        histories[0], tmp_path / "heatmaps.png", dpi=20
    )
    heatmap_figure = saved_figures.pop()
    assert len(heatmap_figure.axes) == 7  # six milestones plus the shared colorbar
    matrix = np.asarray(heatmap_figure.axes[2].images[0].get_array().filled(np.nan))
    assert np.isnan(np.diag(matrix)).all()
    assert np.allclose(matrix, matrix.T, equal_nan=True)
    assert np.isnan(matrix[-2, -1]) and np.isnan(matrix[-1, -2])
    assert "step 2" in heatmap_figure.axes[2].get_title()
