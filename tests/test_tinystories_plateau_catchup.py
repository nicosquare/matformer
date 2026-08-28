from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from scripts.analyze_tinystories_plateau_catchup import (
    ELASTIC_LRS,
    LOSS_TOLERANCE,
    PlateauCatchupError,
    _discover_run_dirs,
    analyze_plateau_run,
    analyze_wsd_calibration,
    freeze_plateau_targets,
    measure_catchup,
    select_elastic_lr,
    select_standalone_wsd_lr,
)
from src.evaluation.reporting import generate_figures
from src.utils.reproducibility import stable_hash


EPOCH_TOKENS = 400


def _provenance():
    return {
        "dataset_name": "roneneldan/TinyStoriesInstruct",
        "dataset_config_name": "default",
        "dataset_split": "train+validation",
        "dataset_phase": "tinystories_instruct_controlled",
        "corpus_hash": "corpus-hash",
        "optimizer_training_manifest_hash": "optimizer-hash",
        "tokenizer_manifest_hash": "tokenizer-hash",
        "d_model": 64,
        "num_layers": 4,
        "context_length": 4,
        "aligned_epoch_samples": 100,
        "aligned_epoch_tokens": EPOCH_TOKENS,
        "excluded_tail_samples": 3,
        "permutation_version": "numpy_pcg64_uint64_le_v1",
        "permutation_hash": "permutation-hash",
        "fixed_epoch_set_hash": "fixed-set-hash",
        "ordering_policy_version": "numpy_pcg64_epoch_positions_v1",
    }


def _write_run(
    root: Path,
    run_id: str,
    *,
    seed: int,
    learning_rate: float,
    model_family: str,
    epochs: int,
    losses_by_width: dict[str, list[tuple[int, float]]],
    status: str = "completed",
    provenance_updates: dict | None = None,
    checkpoint: bool = True,
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    provenance = {**_provenance(), **(provenance_updates or {})}
    widths = (
        ["g1000"]
        if model_family == "standalone"
        else [
            "g250",
            "g500",
            "g750",
            "g1000",
        ]
    )
    iteration = {
        "mode": "repeat_epochs",
        "epoch_order": "deterministic_per_epoch",
        "ordering_policy_version": provenance["ordering_policy_version"],
        "planned_samples": epochs * 100,
        "aligned_epoch_samples": provenance["aligned_epoch_samples"],
        "aligned_epoch_tokens": provenance["aligned_epoch_tokens"],
        "excluded_tail_samples": provenance["excluded_tail_samples"],
        "excluded_tail_tokens": provenance["excluded_tail_samples"] * 4,
        "complete_epochs": epochs,
        "partial_final_epoch_samples": 0,
        "partial_final_epoch_tokens": 0,
        "planned_data_reuse_factor": float(epochs),
        "permutation_version": provenance["permutation_version"],
        "permutation_hash": provenance["permutation_hash"],
        "fixed_epoch_set_hash": provenance["fixed_epoch_set_hash"],
        "corpus_hash": provenance["corpus_hash"],
        "optimizer_training_manifest_hash": provenance[
            "optimizer_training_manifest_hash"
        ],
    }
    config = {
        "run": {
            "run_id": run_id,
            "seed": seed,
            "model_family": model_family,
            "sampling_mode": "standalone"
            if model_family == "standalone"
            else "nested-random",
        },
        "model": {
            "d_model": provenance["d_model"],
            "num_layers": provenance["num_layers"],
            "context_length": provenance["context_length"],
            "granularities": widths,
            "granularity_sampling_mode": "global",
            "global_sampling_interval_steps": 1,
            "tokenizer_manifest_hash": provenance["tokenizer_manifest_hash"],
        },
        "training": {
            "token_budget": epochs * provenance["aligned_epoch_tokens"],
            "expected_tokens_per_step": 25,
            "learning_rate": learning_rate,
            "resolved_learning_rate": learning_rate,
        },
        "dataset": {
            "mode": "packed_mmap",
            "dataset_name": provenance["dataset_name"],
            "dataset_config_name": provenance["dataset_config_name"],
            "dataset_split": provenance["dataset_split"],
            "dataset_phase": provenance["dataset_phase"],
            "corpus_hash": provenance["corpus_hash"],
            "optimizer_iteration": iteration,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    checkpoint_path = run_dir / "checkpoints" / "best_eval_step_16.pt"
    if checkpoint:
        checkpoint_path.parent.mkdir()
        checkpoint_path.write_bytes(f"checkpoint-{seed}".encode())
    g1000_rows = losses_by_width["g1000"]
    best_tokens, _ = min(g1000_rows, key=lambda item: item[1])
    best_step = best_tokens // 25
    summary = {
        "run_id": run_id,
        "seed": seed,
        "status": status,
        "tokens_seen": epochs * provenance["aligned_epoch_tokens"],
        "token_budget": epochs * provenance["aligned_epoch_tokens"],
        "resolved_learning_rate": learning_rate,
        "best_checkpoint_path": str(checkpoint_path),
        "checkpoint_status": "best_eval",
        "checkpoint_selection_step": best_step,
        "unresolved_artifact_failures": [],
        "optimizer_iteration": iteration,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "run_id",
                "run_seed",
                "model_family",
                "sampling_mode",
                "model_variant",
                "correction_mode",
                "split",
                "step",
                "tokens_seen",
                "granularity",
                "loss",
            ],
        )
        writer.writeheader()
        for width, observations in losses_by_width.items():
            for tokens, loss in observations:
                writer.writerow(
                    {
                        "run_id": run_id,
                        "run_seed": seed,
                        "model_family": model_family,
                        "sampling_mode": config["run"]["sampling_mode"],
                        "model_variant": "slicing",
                        "correction_mode": "none",
                        "split": "validation",
                        "step": tokens // 25,
                        "tokens_seen": tokens,
                        "granularity": width,
                        "loss": loss,
                    }
                )
        writer.writerow(
            {
                "run_id": run_id,
                "run_seed": seed,
                "model_family": model_family,
                "sampling_mode": config["run"]["sampling_mode"],
                "model_variant": "slicing",
                "correction_mode": "none",
                "split": "final_holdout",
                "step": (
                    epochs * provenance["aligned_epoch_tokens"] // 25
                ),
                "tokens_seen": epochs * provenance["aligned_epoch_tokens"],
                "granularity": "g1000",
                "loss": 0.1,
            }
        )
    return run_dir


def _plateau_losses(trailing: float = 1.001):
    tokens = list(range(300, 1201, 50))
    return [(token, 1.0 if token == 400 else trailing) for token in tokens]


def _freeze(tmp_path: Path):
    runs = [
        _write_run(
            tmp_path / "plateau-runs",
            f"standalone-s{seed}",
            seed=seed,
            learning_rate=0.008,
            model_family="standalone",
            epochs=3,
            losses_by_width={"g1000": _plateau_losses()},
        )
        for seed in (42, 43, 44)
    ]
    output = tmp_path / "plateau-analysis"
    report = freeze_plateau_targets(runs, output)
    assert report["status"] == "targets_frozen"
    return output / "frozen_standalone_targets.json"


def _elastic_losses(g1000: float, *, epochs: int):
    tokens = list(range(100, epochs * EPOCH_TOKENS + 1, 50))
    return {
        "g250": [(token, g1000 + 0.3) for token in tokens],
        "g500": [(token, g1000 + 0.2) for token in tokens],
        "g750": [(token, g1000 + 0.1) for token in tokens],
        "g1000": [(token, g1000) for token in tokens],
    }


def test_run_discovery_finds_output_group_and_filters_seed(tmp_path):
    runs_root = tmp_path / "standalone"
    output_group = runs_root / "matformer_llama_1m_2b_tokens"
    seed_42 = _write_run(
        output_group,
        "standalone-s42",
        seed=42,
        learning_rate=0.008,
        model_family="standalone",
        epochs=3,
        losses_by_width={"g1000": _plateau_losses()},
    )
    _write_run(
        output_group,
        "standalone-s43",
        seed=43,
        learning_rate=0.008,
        model_family="standalone",
        epochs=3,
        losses_by_width={"g1000": _plateau_losses()},
    )

    assert _discover_run_dirs([], [runs_root], seeds=[42]) == [seed_42.resolve()]
    with pytest.raises(PlateauCatchupError, match="requested seed.*44"):
        _discover_run_dirs([], [runs_root], seeds=[44])


def test_plateau_freezes_only_three_stable_seeds_and_records_checkpoint_hash(tmp_path):
    frozen_path = _freeze(tmp_path)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert set(frozen["targets"]) == {"42", "43", "44"}
    assert frozen["manifest_hash"] == stable_hash(
        {key: value for key, value in frozen.items() if key != "manifest_hash"}
    )
    assert frozen["targets"]["42"]["plateau_onset_tokens"] == EPOCH_TOKENS
    assert frozen["targets"]["42"]["plateau_confirmation_tokens"] == 600
    assert frozen["targets"]["42"]["checkpoint_sha256"]
    assert (frozen_path.parent / "plateau_runs.csv").is_file()
    assert (frozen_path.parent / "plateau.png").is_file()


def test_plateau_rejects_trailing_instability_and_missing_checkpoint(tmp_path):
    unstable = _write_run(
        tmp_path,
        "unstable",
        seed=42,
        learning_rate=0.008,
        model_family="standalone",
        epochs=3,
        losses_by_width={"g1000": _plateau_losses(trailing=1.02)},
        checkpoint=False,
    )
    result = analyze_plateau_run(unstable)
    assert result["contract_satisfied"] is False
    assert any("trailing-five" in reason for reason in result["rejection_reasons"])
    assert any("checkpoint" in reason for reason in result["rejection_reasons"])


def test_cross_seed_plateau_disagreement_does_not_emit_frozen_manifest(tmp_path):
    runs = []
    for seed in (42, 43, 44):
        losses = _plateau_losses()
        if seed == 44:
            losses = [(token, 2.0 - token / 10_000) for token in range(300, 801, 50)]
        runs.append(
            _write_run(
                tmp_path / "runs",
                f"s{seed}",
                seed=seed,
                learning_rate=0.008,
                model_family="standalone",
                epochs=3,
                losses_by_width={"g1000": losses},
            )
        )
    output = tmp_path / "analysis"
    report = freeze_plateau_targets(runs, output)
    assert report["status"] == "plateau_not_robust"
    assert not (output / "frozen_standalone_targets.json").exists()


def test_lr_selection_uses_g1000_endpoint_and_links_frozen_targets(tmp_path):
    frozen_path = _freeze(tmp_path)
    runs = []
    for lr in ELASTIC_LRS:
        losses = _elastic_losses(1.2 - lr * 10, epochs=1)
        if lr == 0.004:
            losses["g250"] = [(token, 0.2) for token, _ in losses["g250"]]
        runs.append(
            _write_run(
                tmp_path / "lr-runs",
                f"elastic-lr{lr}",
                seed=42,
                learning_rate=lr,
                model_family="nested",
                epochs=1,
                losses_by_width=losses,
            )
        )
    output = tmp_path / "lr-analysis"
    selection = select_elastic_lr(runs, frozen_path, output)
    assert selection["selected_learning_rate"] == 0.010
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert selection["frozen_standalone_targets_hash"] == frozen["manifest_hash"]
    assert selection["diagnostic_best_losses_by_width"]["g250"] != 0.2
    assert (output / "elastic_lr_candidates.csv").is_file()


def _selection(tmp_path: Path, frozen_path: Path, selected_lr: float = 0.008):
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    selection = {
        "schema_version": 1,
        "selected_learning_rate": selected_lr,
        "frozen_standalone_targets_hash": frozen["manifest_hash"],
    }
    selection["manifest_hash"] = stable_hash(selection)
    path = tmp_path / "elastic_lr_selection.json"
    path.write_text(json.dumps(selection), encoding="utf-8")
    return path


def test_catchup_rejects_transient_crossing_and_reports_censored_seed(tmp_path):
    frozen_path = _freeze(tmp_path)
    selection_path = _selection(tmp_path, frozen_path)
    runs = []
    for seed in (42, 43, 44):
        losses = _elastic_losses(1.03, epochs=3)
        g1000 = losses["g1000"]
        if seed == 42:
            # One transient crossing, then the first sustained five-point streak.
            g1000[8] = (g1000[8][0], 1.0 + LOSS_TOLERANCE / 2)
            for index in range(12, 17):
                g1000[index] = (g1000[index][0], 1.0 + LOSS_TOLERANCE / 2)
        elif seed == 43:
            for index in range(10, 15):
                g1000[index] = (g1000[index][0], 1.0)
        runs.append(
            _write_run(
                tmp_path / "catchup-runs",
                f"elastic-s{seed}",
                seed=seed,
                learning_rate=0.008,
                model_family="nested",
                epochs=3,
                losses_by_width=losses,
            )
        )
    report = measure_catchup(
        runs, frozen_path, selection_path, tmp_path / "catchup-analysis"
    )
    assert report["status"] == "censored"
    assert report["general_cross_seed_catchup_claim"] is False
    by_seed = {row["seed"]: row for row in report["seeds"]}
    assert by_seed[42]["catchup_tokens"] == 700
    assert by_seed[44]["censored"] is True
    assert report["additional_budget_summary"] is None


def test_catchup_reports_cross_seed_additional_budget_statistics(tmp_path):
    frozen_path = _freeze(tmp_path)
    selection_path = _selection(tmp_path, frozen_path)
    runs = []
    for offset, seed in enumerate((42, 43, 44)):
        losses = _elastic_losses(1.03, epochs=3)
        for index in range(8 + offset, 13 + offset):
            token, _ = losses["g1000"][index]
            losses["g1000"][index] = (token, 1.0)
        runs.append(
            _write_run(
                tmp_path / "all-catchup-runs",
                f"elastic-s{seed}",
                seed=seed,
                learning_rate=0.008,
                model_family="nested",
                epochs=3,
                losses_by_width=losses,
            )
        )
    report = measure_catchup(
        runs, frozen_path, selection_path, tmp_path / "all-catchup-analysis"
    )
    assert report["status"] == "cross_seed_catchup"
    assert report["general_cross_seed_catchup_claim"] is True
    assert report["additional_budget_summary"]["minimum_additional_token_budget"] >= 0


def test_lr_selection_rejects_provenance_mismatch(tmp_path):
    frozen_path = _freeze(tmp_path)
    runs = []
    for lr in ELASTIC_LRS:
        runs.append(
            _write_run(
                tmp_path / "bad-lr-runs",
                f"elastic-lr{lr}",
                seed=42,
                learning_rate=lr,
                model_family="nested",
                epochs=1,
                losses_by_width=_elastic_losses(1.1, epochs=1),
                provenance_updates={
                    "corpus_hash": "wrong" if lr == 0.006 else "corpus-hash"
                },
            )
        )
    with pytest.raises(PlateauCatchupError, match="provenance"):
        select_elastic_lr(runs, frozen_path, tmp_path / "bad-analysis")


def test_plateau_can_be_detected_after_two_epochs(tmp_path):
    losses = []
    for tokens in range(300, 1201, 50):
        loss = max(1.0, 1.8 - tokens / 1000)
        losses.append((tokens, loss))
    run = _write_run(
        tmp_path,
        "late-plateau",
        seed=42,
        learning_rate=0.008,
        model_family="standalone",
        epochs=3,
        losses_by_width={"g1000": losses},
    )
    result = analyze_plateau_run(run)
    assert result["contract_satisfied"] is True
    assert result["plateau_onset_tokens"] == 2 * EPOCH_TOKENS
    assert result["plateau_confirmation_tokens"] == 1000


def test_plateau_onset_moves_after_a_late_significant_improvement(tmp_path):
    losses = []
    for tokens in range(300, 1201, 50):
        loss = 1.0 if tokens < 900 else 0.9
        losses.append((tokens, loss))
    run = _write_run(
        tmp_path,
        "late-recovery",
        seed=42,
        learning_rate=0.008,
        model_family="standalone",
        epochs=3,
        losses_by_width={"g1000": losses},
    )
    result = analyze_plateau_run(run)
    assert result["contract_satisfied"] is True
    assert result["plateau_onset_tokens"] == 900
    assert result["plateau_confirmation_tokens"] == 1100
    assert result["post_confirmation_best_improvement"] == pytest.approx(0.0)


def test_matched_budget_runs_feed_make_figures_comparison(tmp_path):
    matched_root = tmp_path / "matched-runs"
    _write_run(
        matched_root / "standalone",
        "standalone-s42",
        seed=42,
        learning_rate=0.008,
        model_family="standalone",
        epochs=3,
        losses_by_width={"g1000": _plateau_losses()},
    )
    _write_run(
        matched_root / "elastic",
        "elastic-s42",
        seed=42,
        learning_rate=0.008,
        model_family="nested",
        epochs=3,
        losses_by_width=_elastic_losses(1.01, epochs=3),
    )
    output = tmp_path / "figures"
    paths = generate_figures(
        matched_root,
        output,
        refresh_counts=False,
        dpi=40,
    )
    comparison = output / "validation_loss_over_tokens_granularity_comparison.png"
    assert comparison in paths
    assert comparison.is_file()


def _write_wsd_run(
    root: Path,
    run_id: str,
    *,
    learning_rate: float,
    epochs: int,
    losses: list[tuple[int, float]],
    checkpoint: bool = True,
    provenance_updates: dict | None = None,
) -> Path:
    epoch_tokens = 2_500
    updates = {
        "aligned_epoch_samples": 625,
        "aligned_epoch_tokens": epoch_tokens,
        **(provenance_updates or {}),
    }
    run_dir = _write_run(
        root,
        run_id,
        seed=42,
        learning_rate=learning_rate,
        model_family="standalone",
        epochs=epochs,
        losses_by_width={"g1000": losses},
        checkpoint=checkpoint,
        provenance_updates=updates,
    )
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    max_steps = epochs * epoch_tokens // 25
    decay_steps = math.ceil(max_steps * 0.10)
    stable_steps = max_steps - 64 - decay_steps
    cooldown_start = 64 + stable_steps
    contract = {
        "name": "warmup_stable_decay",
        "policy": "ratio_decay_over_total_steps",
        "policy_version": 1,
        "max_steps": max_steps,
        "warmup_steps": 64,
        "stable_steps": stable_steps,
        "decay_steps": decay_steps,
        "decay_ratio": 0.10,
        "cooldown_start_step": cooldown_start,
        "cooldown_start_tokens": cooldown_start * 25,
        "schedule_end_tokens": epochs * epoch_tokens,
        "warmup_type": "linear",
        "decay_type": "cosine",
        "min_lr_ratio": 0.0,
        "min_learning_rate": 0.0,
        "num_cycles": 0.5,
    }
    config["training"].update(
        {
            "max_steps": max_steps,
            "resolved_warmup_steps": 64,
            "scheduler_name": "warmup_stable_decay",
            "scheduler": {
                "name": "warmup_stable_decay",
                "kwargs": {
                    "decay_ratio": 0.10,
                    "warmup_type": "linear",
                    "decay_type": "cosine",
                    "min_lr_ratio": 0.0,
                    "num_cycles": 0.5,
                },
                "resolved_warmup_steps": 64,
                "contract": contract,
            },
            "scheduler_kwargs": {
                "num_decay_steps": decay_steps,
                "num_stable_steps": stable_steps,
                "warmup_type": "linear",
                "decay_type": "cosine",
                "min_lr_ratio": 0.0,
                "num_cycles": 0.5,
            },
            "scheduler_contract": contract,
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["continuation_state"] = {"resume_count": 0}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


def _screen_losses(cooldown_loss: float):
    return [
        (1_600, cooldown_loss + 0.2),
        (2_250, cooldown_loss),
        (2_500, cooldown_loss + 0.01),
    ]


def test_standalone_wsd_lr_selection_uses_only_cooldown_and_freezes_provenance(
    tmp_path,
):
    runs = []
    for lr, cooldown_loss in zip(
        (0.002, 0.004, 0.006, 0.008),
        (1.0, 0.9, 0.8, 0.85),
        strict=True,
    ):
        runs.append(
            _write_wsd_run(
                tmp_path / "screen",
                f"wsd-lr-{lr}",
                learning_rate=lr,
                epochs=1,
                losses=_screen_losses(cooldown_loss),
            )
        )
    output = tmp_path / "selection"
    selection = select_standalone_wsd_lr(runs, output)

    assert selection["selected_learning_rate"] == 0.006
    assert selection["selected_cooldown_best_step"] == 90
    assert selection["selected_checkpoint_sha256"]
    assert (output / "standalone_wsd_lr_selection.json").is_file()
    assert (output / "standalone_wsd_lr_candidates.csv").is_file()
    assert (output / "standalone_wsd_lr_selection.png").is_file()
    assert not (output / "frozen_standalone_targets.json").exists()
    assert select_standalone_wsd_lr(runs, output) == selection


def test_standalone_wsd_lr_selection_rejects_incomplete_duplicate_and_provenance(
    tmp_path,
):
    runs = [
        _write_wsd_run(
            tmp_path / "screen",
            f"wsd-lr-{lr}",
            learning_rate=lr,
            epochs=1,
            losses=_screen_losses(1.0 - lr),
        )
        for lr in (0.002, 0.004, 0.006)
    ]
    with pytest.raises(PlateauCatchupError, match="exactly"):
        select_standalone_wsd_lr(runs, tmp_path / "incomplete")

    duplicate = _write_wsd_run(
        tmp_path / "duplicate",
        "wsd-lr-duplicate",
        learning_rate=0.006,
        epochs=1,
        losses=_screen_losses(0.7),
    )
    with pytest.raises(PlateauCatchupError, match="Duplicate"):
        select_standalone_wsd_lr(runs + [duplicate], tmp_path / "duplicate-output")

    mismatched = _write_wsd_run(
        tmp_path / "mismatch",
        "wsd-lr-0.008",
        learning_rate=0.008,
        epochs=1,
        losses=_screen_losses(0.7),
        provenance_updates={"corpus_hash": "different-corpus"},
    )
    with pytest.raises(PlateauCatchupError, match="provenance"):
        select_standalone_wsd_lr(runs + [mismatched], tmp_path / "mismatch-output")


def test_standalone_wsd_lr_selection_requires_winner_checkpoint(tmp_path):
    runs = []
    for lr in (0.002, 0.004, 0.006, 0.008):
        runs.append(
            _write_wsd_run(
                tmp_path / "screen",
                f"wsd-lr-{lr}",
                learning_rate=lr,
                epochs=1,
                losses=_screen_losses(0.7 if lr == 0.004 else 0.9),
                checkpoint=lr != 0.004,
            )
        )
    with pytest.raises(PlateauCatchupError, match="checkpoint"):
        select_standalone_wsd_lr(runs, tmp_path / "selection")


def test_wsd_calibration_separates_phases_and_never_freezes_targets(tmp_path):
    screen_runs = []
    for lr in (0.002, 0.004, 0.006, 0.008):
        screen_runs.append(
            _write_wsd_run(
                tmp_path / "screen",
                f"wsd-lr-{lr}",
                learning_rate=lr,
                epochs=1,
                losses=_screen_losses(0.7 if lr == 0.006 else 0.9),
            )
        )
    selection_dir = tmp_path / "selection"
    selection = select_standalone_wsd_lr(screen_runs, selection_dir)
    calibration = _write_wsd_run(
        tmp_path / "full",
        "wsd-calibration-full",
        learning_rate=selection["selected_learning_rate"],
        epochs=3,
        losses=[
            (1_600, 1.2),
            (3_200, 1.0),
            (5_000, 0.9),
            (6_750, 0.85),
            (7_000, 0.80),
            (7_500, 0.82),
        ],
    )
    calibration_config_path = calibration / "config.json"
    calibration_config = json.loads(
        calibration_config_path.read_text(encoding="utf-8")
    )
    calibration_config["controlled_experiment"] = {
        "selection_report_hash": selection["manifest_hash"]
    }
    calibration_config_path.write_text(
        json.dumps(calibration_config), encoding="utf-8"
    )
    output = tmp_path / "calibration-analysis"
    report = analyze_wsd_calibration(
        calibration,
        selection_dir / "standalone_wsd_lr_selection.json",
        output,
    )

    assert report["status"] == "calibration_reported"
    assert report["validation_counts_by_phase"]["stable"] > 0
    assert report["validation_counts_by_phase"]["cooldown"] == 3
    assert report["stable_best"]["loss"] == pytest.approx(0.9)
    assert report["cooldown_best"]["loss"] == pytest.approx(0.8)
    assert report["cooldown_gain"] == pytest.approx(0.1)
    assert report["checkpoint_sha256"]
    assert report["targets_frozen"] is False
    assert (output / "wsd_calibration_report.json").is_file()
    assert (output / "wsd_calibration_validation.csv").is_file()
    assert (output / "wsd_calibration.png").is_file()
    assert not (output / "frozen_standalone_targets.json").exists()
