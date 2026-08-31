import csv
import json
import math
from pathlib import Path

import pytest

from src.evaluation.reporting import (
    display_sampling_label_for_curve,
    generate_figures,
    scaling_curve_sampling_label,
)
from src.evaluation.reporting_io import (
    controller_method_family_from_saved_config,
    controller_method_version_from_saved_config,
    controller_scope_from_saved_config,
)
from src.utils.metrics import write_metrics_csv, write_scaling_results_csv
from src.utils.monitoring import group_loss_rows_by_series


def test_optimizer_scope_public_reporting_models_default_historical_scope_to_shared():
    from src.evaluation.reporting import OptimizerStateRunOutcome, ResourceCosts

    outcome = OptimizerStateRunOutcome.from_artifacts(
        config={"model": {"granularities": ["g250", "g1000"]}, "training": {}},
        summary={"run_id": "historical", "seed": 42},
        per_width_outcomes={"g250": {"loss": 2.0}, "g1000": {"loss": 3.0}},
        resources=ResourceCosts(1.0, 1024, 2048),
    )

    assert outcome.state_scope == "shared"
    assert outcome.ordered_widths == ("g250", "g1000")
    assert outcome.uniform_mean_loss == pytest.approx(2.5)
    assert outcome.worst_width_loss == pytest.approx(3.0)
    assert outcome.evidence_label == "diagnostic"


def test_optimizer_scope_reporting_requires_explicit_six_run_manifest(tmp_path):
    from scripts.analyze_tinystories_per_width_optimizer import (
        OptimizerStateAnalysisError,
        freeze_manifest,
    )

    with pytest.raises(OptimizerStateAnalysisError, match="exactly six"):
        freeze_manifest(
            phase="pilot",
            run_dirs=[tmp_path / "one"],
            output_dir=tmp_path / "analysis",
        )


def _write_optimizer_scope_reporting_runs(root: Path) -> list[Path]:
    from src.utils.metrics import METRICS_COLUMNS
    from src.utils.reproducibility import stable_hash

    run_dirs = []
    widths = ("g250", "g500", "g750", "g1000")
    for seed in (42, 43, 44):
        for scope in ("shared", "per_granularity"):
            run_id = f"optimizer-state-{scope}-s{seed}"
            run_dir = root / run_id
            checkpoint = run_dir / "checkpoints" / "latest.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(f"checkpoint-{seed}-{scope}".encode())
            signature = stable_hash({"seed": seed, "controls": "frozen"})
            config = {
                "controlled_experiment": {
                    "holdout_opened_during_pilot": False,
                },
                "run": {"run_id": run_id, "seed": seed},
                "model": {"granularities": list(widths)},
                "training": {
                    "optimizer_state_scope": scope,
                    "optimizer_state_contract": {
                        "ordered_granularities": list(widths)
                    },
                },
                "comparison_control_signature": signature,
            }
            summary = {
                "run_id": run_id,
                "seed": seed,
                "status": "completed",
                "optimizer_state_scope": scope,
                "ordered_optimizer_granularities": list(widths),
                "comparison_control_signature": signature,
                "committed_optimizer_steps": 4,
                "attempted_optimizer_steps": 4,
                "optimizer_successful_update_counts": {
                    width: 1 for width in widths
                },
                "optimizer_exposure_counts": {width: 1 for width in widths},
                "optimizer_accounting_reconciled": True,
                "training_wall_time_seconds": 10.0 + (scope != "shared"),
                "peak_accelerator_memory_bytes": 1024
                + 256 * (scope != "shared"),
                "terminal_checkpoint_path": str(checkpoint),
                "terminal_checkpoint_bytes": checkpoint.stat().st_size,
                "terminal_checkpoint_purpose": "resumable_training",
            }
            (run_dir / "config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            (run_dir / "run_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            metric_rows = []
            for step, width in enumerate(widths, start=1):
                metric_rows.append(
                    {
                        "run_id": run_id,
                        "step": step,
                        "split": "train",
                        "granularity": width,
                        "optimizer_state_scope": scope,
                        "selected_optimizer_granularity": width,
                        "optimizer_step_attempted": True,
                        "optimizer_step_committed": True,
                        "optimizer_action_id": f"balanced:{step}:{width}",
                        "optimizer_batch_provenance": {"batch": step},
                        "learning_rate": 0.008,
                        "global_scheduler_position": step,
                    }
                )
                loss = 2.0 + step / 10 + seed / 1000
                if scope == "per_granularity":
                    loss -= 0.01
                metric_rows.append(
                    {
                        "run_id": run_id,
                        "step": 4,
                        "split": "validation",
                        "granularity": width,
                        "loss": loss,
                        "perplexity": math.exp(loss),
                    }
                )
            for row in metric_rows:
                for field in METRICS_COLUMNS:
                    row.setdefault(field, None)
            write_metrics_csv(run_dir, metric_rows)
            run_dirs.append(run_dir)
    return run_dirs


def test_optimizer_scope_freeze_and_report_emit_auditable_endpoint_resource_tables(
    tmp_path,
):
    from scripts.analyze_tinystories_per_width_optimizer import (
        OptimizerStateAnalysisError,
        freeze_manifest,
        write_report,
    )

    run_dirs = _write_optimizer_scope_reporting_runs(tmp_path / "runs")
    from dataclasses import replace
    from src.evaluation.reporting_impl import validate_optimizer_state_pairs
    from src.evaluation.reporting_io import load_optimizer_state_run

    loaded = [load_optimizer_state_run(path) for path in run_dirs]
    mismatched = [*loaded[:-1], replace(loaded[-1], paired_control_signature="bad")]
    with pytest.raises(ValueError, match="paired-control signatures differ"):
        validate_optimizer_state_pairs(mismatched)
    analysis = tmp_path / "analysis"
    manifest_path = freeze_manifest(
        phase="pilot", run_dirs=run_dirs, output_dir=analysis
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["runs"]) == 6
    assert {entry["seed"] for entry in manifest["runs"]} == {42, 43, 44}
    assert {entry["state_scope"] for entry in manifest["runs"]} == {
        "shared",
        "per_granularity",
    }
    assert freeze_manifest(
        phase="pilot", run_dirs=reversed(run_dirs), output_dir=analysis
    ) == manifest_path
    with pytest.raises(OptimizerStateAnalysisError, match="Immutable manifest"):
        freeze_manifest(
            phase="confirmation", run_dirs=run_dirs, output_dir=analysis
        )

    report_path, csv_path = write_report(
        manifest_path=manifest_path, output_dir=analysis
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["evidence_label"] == "diagnostic"
    assert report["matched_compute_claim"] is False
    assert report["holdout_results_complete"] is False
    assert len(report["outcomes"]) == 6
    assert len(report["paired_deltas"]) == 3
    assert report["seed_aggregate"]["seed_count"] == 3
    assert set(report["resource_summary_by_scope"]) == {
        "shared",
        "per_granularity",
    }
    with csv_path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 24
    assert {
        "loss",
        "perplexity",
        "uniform_mean_loss",
        "worst_width_loss",
        "wall_time_seconds",
        "peak_accelerator_memory_bytes",
        "resumable_checkpoint_bytes",
    }.issubset(rows[0])


def test_optimizer_scope_confirmation_report_requires_complete_explicit_holdout(
    tmp_path,
):
    from scripts.analyze_tinystories_per_width_optimizer import (
        freeze_manifest,
        write_report,
    )

    run_dirs = _write_optimizer_scope_reporting_runs(tmp_path / "runs")
    manifest = freeze_manifest(
        phase="confirmation",
        run_dirs=run_dirs,
        output_dir=tmp_path / "analysis",
    )
    for run_dir in run_dirs:
        summary = json.loads((run_dir / "run_summary.json").read_text())
        widths = summary["ordered_optimizer_granularities"]
        checkpoint = Path(summary["terminal_checkpoint_path"])
        import hashlib

        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        result = {
            "checkpoint_sha256": checkpoint_hash,
            "ordered_granularities": widths,
            "ordered_per_granularity_losses": [
                {
                    "granularity": width,
                    "loss": 2.0 + index / 10,
                    "perplexity": math.exp(2.0 + index / 10),
                }
                for index, width in enumerate(widths)
            ],
        }
        (run_dir / "final_holdout_results.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
    report_path, _ = write_report(
        manifest_path=manifest, output_dir=tmp_path / "analysis"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["holdout_results_complete"] is True
    assert report["evidence_label"] == "confirmatory"


def _write_final_holdout_reporting_run(
    root: Path,
    *,
    run_id: str,
    granularities: tuple[str, ...],
    losses: tuple[float, ...],
    model_family: str = "nested",
    sampling_mode: str = "nested-random",
    final_holdout_manifest_hash: str = "final-holdout-hash",
    evaluation_examples: int = 512,
) -> Path:
    from src.evaluation.final_holdout import FINAL_HOLDOUT_AGGREGATION
    from src.utils.reproducibility import stable_hash

    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    role_hashes = {
        "data_roles_manifest_hash": "data-roles-hash",
        "optimizer_training_manifest_hash": "optimizer-training-hash",
        "controller_manifest_hash": "controller-hash",
        "validation_manifest_hash": "validation-hash",
        "final_holdout_manifest_hash": final_holdout_manifest_hash,
    }
    config = {
        "run": {
            "run_id": run_id,
            "model_family": model_family,
            "sampling_mode": sampling_mode,
            "seed": 42,
        },
        "model": {
            "variant": "slicing",
            "correction_mode": "none",
            "membership_correction": False,
            "granularity_sampling_mode": "global",
            "resolved_sampling_mode": "global",
            "global_sampling_interval_steps": 1,
            "global_sampling_schedule": "random_with_replacement",
            "granularities": list(granularities),
        },
        "training": {
            "token_budget": 1_000,
            "expected_tokens_per_step": 10,
        },
        "evaluation": {
            "final_holdout": {
                "enabled": True,
                "evaluate_during_training": False,
                "examples": 512,
                "fixed_manifest": True,
                "manifest_hash": final_holdout_manifest_hash,
            }
        },
        **role_hashes,
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "run_summary.json").write_text(
        json.dumps({"run_id": run_id, "status": "completed"}),
        encoding="utf-8",
    )
    (run_dir / "final_holdout_manifest.json").write_text(
        json.dumps(
            {
                "role": "final_holdout",
                "manifest_hash": final_holdout_manifest_hash,
                "example_count": 512,
                "source_document_count": 512,
                "sequence_count": evaluation_examples,
            }
        ),
        encoding="utf-8",
    )

    scaling_rows = []
    for index, granularity in enumerate(granularities, start=1):
        scaling_rows.append(
            {
                "comparison_id": f"{run_id}__{granularity}",
                "run_id": run_id,
                "model_family": model_family,
                "sampling_mode": sampling_mode,
                "model_variant": "slicing",
                "completion_label": "completed",
                "granularity": granularity,
                "total_parameters": index * 1_100,
                "embedding_parameters": 100,
                "lm_head_parameters": 100,
                "non_embedding_parameters": index * 1_000,
                "loss": 99.0,
                "perplexity": math.exp(99.0),
                "average_downstream_accuracy": None,
            }
        )
    write_scaling_results_csv(run_dir, scaling_rows)

    components = [
        {
            "granularity": granularity,
            "loss": loss,
            "perplexity": math.exp(loss),
            "evaluation_examples": evaluation_examples,
            "evaluation_target_tokens": 2_048,
        }
        for granularity, loss in zip(granularities, losses, strict=True)
    ]
    result_path = run_dir / "final_holdout_results.json"
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "checkpoint_path": str(run_dir / "checkpoints" / "best.pt"),
        "checkpoint_selection_provenance": {"source": "ordinary_validation"},
        "final_holdout_manifest_hash": final_holdout_manifest_hash,
        "run_summary_hash": "summary-hash",
        "controller_summary_hash": "controller-summary-hash",
        "controller_metrics_hash": "controller-metrics-hash",
        "ordered_granularities": list(granularities),
        "ordered_per_granularity_losses": components,
        "uniform_average_loss": math.fsum(losses) / len(losses),
        "aggregation_method": FINAL_HOLDOUT_AGGREGATION,
        "evaluation_example_count": 512,
        "result_path": str(result_path),
    }
    result["result_hash"] = stable_hash(result)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return run_dir


def _write_controller_timeline_run(
    root: Path,
    *,
    run_id: str,
    scope: str,
    events: list[dict],
    block_count: int = 2,
    ordered_granularities: tuple[str, ...] = ("micro", "medium", "full"),
    expected_tokens_per_step: int = 10,
    token_budget: int = 100,
    model_variant: str = "matformer_llama",
    correction_mode: str = "none",
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    config = {
        "run": {"run_id": run_id, "sampling_mode": "nested-random", "seed": 42},
        "model": {
            "variant": model_variant,
            "correction_mode": correction_mode,
            "adaptive_sampler_strategy": "thompson",
            "granularity_sampling_mode": (
                "adaptive_global" if scope == "global" else "adaptive_per_block"
            ),
            "granularities": list(ordered_granularities),
            "num_layers": block_count,
            "adaptive_controller": {
                "method_family": "bayesian_gaussian_linear_thompson",
                "method_version": 1,
                "scope": scope,
                "feature_schema": {"block_count": block_count},
                "decision_interval_steps": 2,
            },
        },
        "training": {
            "expected_tokens_per_step": expected_tokens_per_step,
            "token_budget": token_budget,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "controller_metrics.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    if scope == "global":
        rows = []
        for event in events:
            if "boundary_step_start" not in event or "boundary_step_end" not in event:
                continue
            action = event.get("action", {}).get("global_granularity")
            if action not in ordered_granularities:
                continue
            for step in range(
                int(event["boundary_step_start"]) + 1,
                int(event["boundary_step_end"]) + 1,
            ):
                rows.append(
                    {
                        "run_id": run_id,
                        "step": step,
                        "split": "train",
                        "tokens_seen": step * expected_tokens_per_step,
                        "granularity": action,
                        "controller_action": action,
                        "controller_window_index": event["window_index"],
                        "controller_sampled_probability": "",
                    }
                )
        if rows:
            with (run_dir / "metrics.csv").open(
                "w", encoding="utf-8", newline=""
            ) as metrics_file:
                writer = csv.DictWriter(metrics_file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return run_dir


def _completed_controller_window(
    *,
    run_id: str,
    scope: str,
    window_index: int,
    start_step: int,
    end_step: int,
    action: dict,
) -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "event_type": "completed_window",
        "method_family": "bayesian_gaussian_linear_thompson",
        "method_version": 1,
        "strategy": "thompson",
        "scope": scope,
        "window_index": window_index,
        "boundary_step": end_step,
        "boundary_step_start": start_step,
        "boundary_step_end": end_step,
        "completed_optimizer_steps": end_step - start_step,
        "action": action,
    }


def _write_panelgrad_history_run(
    root: Path,
    run_id: str = "panelgrad-run",
    importance_metric: str | None = None,
) -> Path:
    resolved_metric = importance_metric or "gradient_rms"
    method_family = f"panelgrad_{resolved_metric}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    config = {
        "run": {"run_id": run_id, "sampling_mode": "nested-random", "seed": 42},
        "model": {
            "variant": "slicing",
            "correction_mode": "none",
            "membership_correction": False,
            "adaptive_sampler_strategy": "panelgrad",
            "granularity_sampling_mode": "adaptive_global",
            "granularities": ["micro", "medium", "full"],
            "panelgrad": {
                "method_family": method_family,
                "method_version": 1,
                "scope": "global",
                "ordered_granularities": ["micro", "medium", "full"],
                "refresh_interval_steps": 2,
                "eta": 1e-12,
                "temperature": 1.0,
                "epsilon_schedule": {
                    "type": "linear",
                    "start": 0.5,
                    "end": 0.1,
                    "duration_steps": 4,
                },
                **(
                    {"importance_metric": importance_metric}
                    if importance_metric is not None
                    else {}
                ),
            },
        },
        "training": {"expected_tokens_per_step": 10, "token_budget": 40},
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    metric_fields = [
        "run_id",
        "step",
        "split",
        "tokens_seen",
        "granularity",
        "controller_action",
        "controller_sampled_probability",
        "loss",
        "perplexity",
    ]
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "run_id": run_id,
                    "step": 1,
                    "split": "train",
                    "tokens_seen": 10,
                    "granularity": "full",
                    "controller_action": "full",
                    "controller_sampled_probability": "",
                    "loss": 3.0,
                    "perplexity": 20.0,
                },
                {
                    "run_id": run_id,
                    "step": 2,
                    "split": "train",
                    "tokens_seen": 20,
                    "granularity": "micro",
                    "controller_action": "micro",
                    "controller_sampled_probability": 0.5,
                    "loss": 2.9,
                    "perplexity": 18.2,
                },
                {
                    "run_id": run_id,
                    "step": 3,
                    "split": "train",
                    "tokens_seen": 30,
                    "granularity": "full",
                    "controller_action": "full",
                    "controller_sampled_probability": 0.2,
                    "loss": 2.8,
                    "perplexity": 16.4,
                },
            ]
        )

    def refresh(
        window_index,
        boundary_step,
        scores,
        p,
        q,
        duration,
        active_epsilon,
        epsilon_schedule_step,
    ):
        return {
            "event_type": "panelgrad_refresh_completed",
            "method_family": method_family,
            "method_version": 1,
            "window_index": window_index,
            "boundary_step": boundary_step,
            "p": p,
            "q": q,
            "entropy": 1.0,
            "min_probability": min(p),
            "max_probability": max(p),
            "active_epsilon": active_epsilon,
            "epsilon_schedule_step": epsilon_schedule_step,
            "duration_seconds": duration,
            "backward_evaluation_count": 6,
            "controller_target_count": 30,
            "controller_target_evaluation_count": 90,
            **(
                {
                    "importance_metric": resolved_metric,
                    "importance_scores": scores,
                }
                if importance_metric is not None
                else {}
            ),
            "measurements": [
                {
                    "granularity": label,
                    "gradient_rms_score": score,
                    "gradient_norm": score,
                }
                for label, score in zip(("micro", "medium", "full"), scores)
            ],
        }

    events = [
        refresh(
            0,
            1,
            [0.3, 0.2, 0.1],
            [0.5, 0.3, 0.2],
            [0.52, 0.30, 0.18],
            2.0,
            0.5,
            0,
        ),
        refresh(
            1,
            3,
            [0.2, 0.25, 0.15],
            [0.3, 0.4, 0.3],
            [0.29, 0.42, 0.29],
            1.5,
            0.3,
            2,
        ),
    ]
    (run_dir / "controller_metrics.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    return run_dir


def _write_global_policy_run(
    root: Path,
    *,
    run_id: str,
    policy: str,
    actions: list[str],
    decision_interval_steps: int = 2,
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    model = {
        "variant": "slicing",
        "correction_mode": "none",
        "membership_correction": False,
        "granularities": ["micro", "medium", "full"],
    }
    if policy == "uniform":
        model["granularity_sampling_mode"] = "global"
    elif policy == "uniform_window":
        model.update(
            {
                "granularity_sampling_mode": "global",
                "global_sampling_interval_steps": decision_interval_steps,
            }
        )
    elif policy == "balanced_window":
        model.update(
            {
                "granularity_sampling_mode": "global",
                "global_sampling_schedule": "balanced_cycle",
                "global_sampling_schedule_version": 1,
                "global_sampling_interval_steps": decision_interval_steps,
            }
        )
    elif policy == "fixed":
        model.update(
            {
                "granularity_sampling_mode": "fixed_global",
                "global_sampling_distribution": {
                    "micro": 0.1,
                    "medium": 0.3,
                    "full": 0.6,
                },
            }
        )
    elif policy == "thompson":
        model.update(
            {
                "granularity_sampling_mode": "adaptive_global",
                "adaptive_sampler_strategy": "thompson",
                "adaptive_controller": {
                    "method_family": "bayesian_gaussian_linear_thompson",
                    "method_version": 1,
                    "scope": "global",
                    "decision_interval_steps": decision_interval_steps,
                },
            }
        )
    elif policy == "panelgrad":
        model.update(
            {
                "granularity_sampling_mode": "adaptive_global",
                "adaptive_sampler_strategy": "panelgrad",
                "panelgrad": {
                    "method_family": "panelgrad_gradient_rms",
                    "method_version": 1,
                    "scope": "global",
                },
            }
        )
    else:
        raise ValueError(policy)
    config = {
        "run": {"run_id": run_id, "sampling_mode": "nested-random", "seed": 42},
        "model": model,
        "training": {
            "token_budget": len(actions) * 10,
            "expected_tokens_per_step": 10,
            "gradient_accumulation_steps": 1,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    fields = [
        "run_id", "step", "split", "tokens_seen", "granularity",
        "controller_action", "controller_window_index",
        "controller_sampled_probability", "global_sampling_window_index",
        "global_sampling_window_progress",
    ]
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for step, action in enumerate(actions, start=1):
            controller_action = action
            if policy == "thompson" and step % decision_interval_steps == 0:
                controller_action = actions[min(step, len(actions) - 1)]
            writer.writerow(
                {
                    "run_id": run_id,
                    "step": step,
                    "split": "train",
                    "tokens_seen": step * 10,
                    "granularity": action,
                    "controller_action": controller_action,
                    "controller_window_index": (step - 1) // decision_interval_steps,
                    "controller_sampled_probability": (
                        0.5
                        if policy == "panelgrad"
                        else (
                            model["global_sampling_distribution"][action]
                            if policy == "fixed"
                            else ""
                        )
                    ),
                    "global_sampling_window_index": (
                        (step - 1) // decision_interval_steps
                        if policy in {"uniform_window", "balanced_window"}
                        else ""
                    ),
                    "global_sampling_window_progress": (
                        ((step - 1) % decision_interval_steps) + 1
                        if policy in {"uniform_window", "balanced_window"}
                        else ""
                    ),
                }
            )
    return run_dir


def test_make_figures_cli_forwards_validation_loss_log_y(monkeypatch):
    import scripts.make_figures as make_figures

    captured = {}

    def fake_generate_figures(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return [Path("outputs/figures/example.png")]

    monkeypatch.setattr(
        "src.evaluation.reporting.generate_figures", fake_generate_figures
    )

    make_figures.main(
        [
            "--validation-loss-log-y",
            "--include-incomplete-validation-traces",
            "--skip-final-holdout",
            "--variant",
            "slicing",
            "--variant",
            "concat",
            "--correction",
            "none",
            "--correction",
            "gmc",
            "--individual-size-panels",
            "--sampling-zoom-steps",
            "75",
        ]
    )

    assert captured["kwargs"]["validation_loss_log_y"] is True
    assert captured["kwargs"]["include_incomplete_validation_traces"] is True
    assert captured["kwargs"]["include_final_holdout"] is False
    assert captured["kwargs"]["variants"] == ["slicing", "concat"]
    assert captured["kwargs"]["corrections"] == ["none", "gmc"]
    assert captured["kwargs"]["include_individual_size_panels"] is True
    assert captured["kwargs"]["sampling_zoom_steps"] == 75


def test_make_figures_cli_sampling_zoom_is_opt_in_and_positive():
    import scripts.make_figures as make_figures

    assert make_figures.parse_args([]).sampling_zoom_steps is None
    with pytest.raises(SystemExit):
        make_figures.parse_args(["--sampling-zoom-steps", "0"])
    with pytest.raises(SystemExit):
        make_figures.parse_args(["--sampling-zoom-steps", "-1"])


def test_figure_generation_can_explicitly_skip_final_holdout(tmp_path, monkeypatch):
    from src.evaluation import reporting_io

    def fail_if_called(_input_root):
        raise AssertionError("final-holdout reporting must be skipped")

    monkeypatch.setattr(
        reporting_io,
        "read_final_holdout_scaling_rows",
        fail_if_called,
    )

    paths = generate_figures(
        tmp_path,
        tmp_path / "figures",
        refresh_counts=False,
        include_final_holdout=False,
    )
    assert all(path.is_file() for path in paths)


def test_configured_learning_rate_schedule_is_reconstructed_and_generated(tmp_path):
    from src.evaluation.reporting import _scheduler_curve_points
    from src.evaluation.reporting_io import iter_learning_rate_schedules

    run_dir = tmp_path / "cosine-run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "run": {"run_id": "cosine-run"},
                "model": {"variant": "slicing", "correction_mode": "none"},
                "training": {
                    "max_steps": 10,
                    "token_budget": 100,
                    "expected_tokens_per_step": 10,
                    "resolved_learning_rate": 0.001,
                    "resolved_warmup_steps": 2,
                    "scheduler_name": "cosine",
                    "scheduler_kwargs": {},
                },
            }
        ),
        encoding="utf-8",
    )

    [schedule] = list(iter_learning_rate_schedules(tmp_path))
    steps, learning_rates = _scheduler_curve_points(schedule)
    by_step = dict(zip(steps, learning_rates))
    assert by_step[0] == pytest.approx(0.0)
    assert by_step[2] == pytest.approx(0.001)
    assert by_step[10] == pytest.approx(0.0)

    paths = generate_figures(
        tmp_path,
        tmp_path / "figures",
        refresh_counts=False,
        dpi=20,
    )
    assert "learning_rate_schedule.png" in {path.name for path in paths}


def test_ratio_wsd_schedule_is_reconstructed_with_phase_boundaries(tmp_path):
    from src.evaluation.reporting import _scheduler_curve_points
    from src.evaluation.reporting_io import iter_learning_rate_schedules

    run_dir = tmp_path / "wsd-run"
    run_dir.mkdir()
    contract = {
        "name": "warmup_stable_decay",
        "policy": "ratio_decay_over_total_steps",
        "policy_version": 1,
        "max_steps": 10,
        "warmup_steps": 2,
        "stable_steps": 6,
        "decay_steps": 2,
        "decay_ratio": 0.2,
        "cooldown_start_step": 8,
        "cooldown_start_tokens": 80,
        "warmup_type": "linear",
        "decay_type": "cosine",
        "min_lr_ratio": 0.0,
        "min_learning_rate": 0.0,
        "num_cycles": 0.5,
    }
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "run": {"run_id": "wsd-run"},
                "model": {
                    "variant": "slicing",
                    "correction_mode": "none",
                    "membership_correction": False,
                },
                "training": {
                    "max_steps": 10,
                    "token_budget": 100,
                    "expected_tokens_per_step": 10,
                    "resolved_learning_rate": 1.0,
                    "resolved_warmup_steps": 2,
                    "scheduler_name": "warmup_stable_decay",
                    "scheduler_kwargs": {
                        "num_decay_steps": 2,
                        "num_stable_steps": 6,
                        "warmup_type": "linear",
                        "decay_type": "cosine",
                        "min_lr_ratio": 0.0,
                        "num_cycles": 0.5,
                    },
                    "scheduler_contract": contract,
                },
            }
        ),
        encoding="utf-8",
    )

    [schedule] = list(iter_learning_rate_schedules(tmp_path))
    assert schedule.scheduler_contract == contract
    steps, learning_rates = _scheduler_curve_points(schedule)
    by_step = dict(zip(steps, learning_rates))
    assert by_step[0] == pytest.approx(0.0)
    assert by_step[2] == pytest.approx(1.0)
    assert by_step[8] == pytest.approx(1.0)
    assert by_step[9] == pytest.approx(0.5)
    assert by_step[10] == pytest.approx(0.0)

    paths = generate_figures(tmp_path, tmp_path / "figures", refresh_counts=False)
    assert tmp_path / "figures" / "learning_rate_schedule.png" in paths


def test_global_history_carries_scheduler_warmup_boundaries(tmp_path):
    from src.evaluation.reporting_io import iter_global_sampling_histories

    run_dir = _write_global_policy_run(
        tmp_path,
        run_id="warmup-global",
        policy="uniform",
        actions=["micro", "medium", "full", "micro"],
    )
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["training"].update(
        {
            "max_steps": 4,
            "resolved_learning_rate": 0.001,
            "resolved_warmup_steps": 2,
            "scheduler_name": "cosine",
            "scheduler_kwargs": {},
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")

    [history] = list(iter_global_sampling_histories(tmp_path))
    assert history.scheduler_warmup_steps == 2
    assert history.scheduler_warmup_tokens == 20


def test_parameter_count_refresh_preserves_stored_counts_for_legacy_config(
    tmp_path, monkeypatch
):
    from src.evaluation import reporting_io

    run_dir = tmp_path / "legacy-run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    row = {
        "_source_csv": str(run_dir / "scaling_results.csv"),
        "granularity": "micro",
        "total_parameters": "123",
        "non_embedding_parameters": "100",
    }
    monkeypatch.setattr(
        reporting_io,
        "recompute_parameter_counts",
        lambda _path: (_ for _ in ()).throw(KeyError("vocab_size")),
    )

    with pytest.warns(RuntimeWarning, match="using scaling_results.csv values"):
        refreshed = reporting_io.refresh_scaling_parameter_counts(tmp_path, [row])

    assert refreshed[0]["total_parameters"] == "123"
    assert refreshed[0]["non_embedding_parameters"] == "100"


def test_final_holdout_rows_replace_validation_metrics_and_generate_size_figures(
    tmp_path,
):
    from src.evaluation.reporting_io import read_final_holdout_scaling_rows

    nested_run = _write_final_holdout_reporting_run(
        tmp_path,
        run_id="holdout-nested",
        granularities=("small", "full"),
        losses=(2.0, 1.0),
    )
    _write_final_holdout_reporting_run(
        tmp_path,
        run_id="holdout-standalone-small",
        granularities=("small",),
        losses=(1.8,),
        model_family="standalone",
        sampling_mode="standalone",
    )
    _write_final_holdout_reporting_run(
        tmp_path,
        run_id="holdout-standalone-full",
        granularities=("full",),
        losses=(0.9,),
        model_family="standalone",
        sampling_mode="standalone",
    )

    rows = read_final_holdout_scaling_rows(tmp_path)
    nested_small = next(
        row
        for row in rows
        if row["run_id"] == "holdout-nested" and row["granularity"] == "small"
    )
    assert nested_small["loss"] == pytest.approx(2.0)
    assert nested_small["perplexity"] == pytest.approx(math.exp(2.0))
    assert nested_small["metric_source"] == "final_holdout_results.json"
    assert nested_small["evaluation_split"] == "final_holdout"
    assert nested_small["_source_csv"] == str(nested_run / "scaling_results.csv")

    figure_names = {
        path.name
        for path in generate_figures(
            tmp_path,
            tmp_path / "figures",
            refresh_counts=False,
            dpi=20,
        )
    }
    assert "final_holdout_ppl_vs_size.png" in figure_names
    assert "final_holdout_loss_vs_size.png" not in figure_names


def test_final_holdout_rows_accept_packed_sequences_distinct_from_source_examples(
    tmp_path,
):
    from src.evaluation.reporting_io import read_final_holdout_scaling_rows

    _write_final_holdout_reporting_run(
        tmp_path,
        run_id="holdout-packed",
        granularities=("small", "full"),
        losses=(2.0, 1.0),
        evaluation_examples=898,
    )

    rows = read_final_holdout_scaling_rows(tmp_path)

    assert {row["evaluation_examples"] for row in rows} == {898}


def test_final_holdout_reporting_refuses_partial_or_mixed_comparison_sets(tmp_path):
    from src.evaluation.reporting_io import (
        FinalHoldoutReportingError,
        read_final_holdout_scaling_rows,
    )

    complete_run = _write_final_holdout_reporting_run(
        tmp_path,
        run_id="holdout-complete",
        granularities=("small", "full"),
        losses=(2.0, 1.0),
    )
    missing_run = _write_final_holdout_reporting_run(
        tmp_path,
        run_id="holdout-missing",
        granularities=("small", "full"),
        losses=(2.1, 1.1),
    )
    (missing_run / "final_holdout_results.json").unlink()

    with pytest.raises(FinalHoldoutReportingError, match="Missing"):
        read_final_holdout_scaling_rows(tmp_path)

    _write_final_holdout_reporting_run(
        tmp_path,
        run_id="holdout-missing",
        granularities=("small", "full"),
        losses=(2.1, 1.1),
        final_holdout_manifest_hash="different-final-holdout-hash",
    )
    assert (complete_run / "final_holdout_results.json").is_file()
    with pytest.raises(FinalHoldoutReportingError, match="mixes corpus"):
        read_final_holdout_scaling_rows(tmp_path)


def test_reporting_path_groups_loss_rows_and_writes_medium_trend_report(tmp_path):
    run_dir = tmp_path / "debug-nested-001"

    metric_rows = [
        {
            "run_id": "debug-nested-001",
            "step": 1,
            "split": "train",
            "model_family": "nested",
            "model_size_label": "debug",
            "model_shape_label": "debug-shape",
            "sampling_mode": "nested-random",
            "model_variant": "slicing",
            "granularity": "s",
            "metric_name": "train_loss",
            "loss": 2.3,
            "perplexity": 9.0,
            "tokens_seen": 128,
            "content_tokens_seen": 128,
            "wall_clock_seconds": 1.0,
            "tokens_per_second": 128.0,
            "peak_memory_bytes": 2048,
        },
        {
            "run_id": "debug-nested-001",
            "step": 2,
            "split": "validation",
            "model_family": "nested",
            "model_size_label": "debug",
            "model_shape_label": "debug-shape",
            "sampling_mode": "nested-random",
            "model_variant": "slicing",
            "granularity": "s",
            "metric_name": "validation_loss",
            "loss": 2.1,
            "perplexity": 8.2,
            "tokens_seen": 128,
            "content_tokens_seen": 128,
            "wall_clock_seconds": 1.0,
            "tokens_per_second": 128.0,
            "peak_memory_bytes": 2048,
        },
        {
            "run_id": "debug-nested-001",
            "step": 3,
            "split": "validation",
            "model_family": "nested",
            "model_size_label": "debug",
            "model_shape_label": "debug-shape",
            "sampling_mode": "nested-random",
            "model_variant": "slicing",
            "granularity": "xl",
            "metric_name": "loss",
            "loss": 1.8,
            "perplexity": 6.4,
            "tokens_seen": 256,
            "content_tokens_seen": 256,
            "wall_clock_seconds": 2.0,
            "tokens_per_second": 128.0,
            "peak_memory_bytes": 2048,
        },
    ]
    grouped_rows = group_loss_rows_by_series(metric_rows)

    assert set(grouped_rows) == {
        "train/loss/s",
        "validation/loss/s",
        "validation/loss/xl",
    }
    assert grouped_rows["train/loss/s"][0]["metric_name"] == "train_loss"
    assert grouped_rows["validation/loss/xl"][0]["granularity"] == "xl"

    scaling_rows = [
        {
            "comparison_id": "debug-s",
            "run_id": "debug-nested-001",
            "model_family": "nested",
            "model_size_label": "debug",
            "model_family_slug": "matformer_llama",
            "model_size_slug": "9m",
            "token_budget_slug": "1m_tokens",
            "output_group": "matformer_llama_9m_1m_tokens",
            "completion_label": "debug",
            "sampling_mode": "nested-random",
            "model_variant": "slicing",
            "resolved_sampling_mode": "global",
            "granularity": "s",
            "total_parameters": 1000,
            "embedding_parameters": 100,
            "lm_head_parameters": 100,
            "non_embedding_parameters": 800,
            "loss": 2.0,
            "perplexity": 7.4,
            "average_downstream_accuracy": 0.58,
        },
        {
            "comparison_id": "debug-xl",
            "run_id": "debug-nested-001",
            "model_family": "nested",
            "model_size_label": "debug",
            "model_family_slug": "matformer_llama",
            "model_size_slug": "9m",
            "token_budget_slug": "1m_tokens",
            "output_group": "matformer_llama_9m_1m_tokens",
            "completion_label": "debug",
            "sampling_mode": "nested-random",
            "model_variant": "slicing",
            "resolved_sampling_mode": "global",
            "granularity": "xl",
            "total_parameters": 2000,
            "embedding_parameters": 100,
            "lm_head_parameters": 100,
            "non_embedding_parameters": 1800,
            "loss": 1.5,
            "perplexity": 4.5,
            "average_downstream_accuracy": 0.58,
        },
    ]

    write_metrics_csv(run_dir, metric_rows)
    write_scaling_results_csv(run_dir, scaling_rows)
    (run_dir / "run_summary.json").write_text(
        json.dumps({"status": "completed", "tokens_seen": 256, "token_budget": 256}),
        encoding="utf-8",
    )

    figure_paths = generate_figures(
        tmp_path, tmp_path / "figures", refresh_counts=False
    )
    figure_names = {path.name for path in figure_paths}

    assert "loss_vs_size.png" not in figure_names
    assert "medium_trend_report.md" in figure_names
    assert any(name.startswith("validation_loss_over_tokens_") for name in figure_names)

    report = (tmp_path / "figures" / "medium_trend_report.md").read_text(
        encoding="utf-8"
    )
    assert "- nested-random: 2 rows; granularities=s, xl" in report
    assert "average_downstream_accuracy: 0.58" in report

    assert not any(name.startswith("ppl_vs_size__") for name in figure_names)
    assert not any(name.startswith("accuracy_vs_size__") for name in figure_names)

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "run": {
                    "run_id": "debug-nested-001",
                    "sampling_mode": "nested-random",
                    "seed": 42,
                },
                "model": {
                    "variant": "matformer_llama",
                    "granularity_sampling_mode": "global",
                    "granularities": ["s", "xl"],
                },
                "training": {
                    "token_budget": 256,
                    "expected_tokens_per_step": 128,
                    "gradient_accumulation_steps": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    individual_paths = generate_figures(
        tmp_path,
        tmp_path / "individual-figures",
        refresh_counts=False,
        include_individual_size_panels=True,
        dpi=20,
    )
    individual_names = {path.name for path in individual_paths}
    assert any(
        name.startswith("ppl_vs_size") and "__" in name
        for name in individual_names
    ), individual_names
    assert any(
        name.startswith("accuracy_vs_size") and "__" in name
        for name in individual_names
    )
    assert not any(name.startswith("loss_vs_size") for name in individual_names)


def test_default_generation_removes_superseded_artifacts(tmp_path):
    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    stale_names = {
        "loss_vs_size.png",
        "loss_vs_size__nested_random_slicing_global.png",
        "ppl_vs_size__nested_random_slicing_global.png",
        "accuracy_vs_size__nested_random_slicing_global.png",
        "final_holdout_loss_vs_size.png",
        "final_holdout_ppl_vs_size__nested_random_slicing_global.png",
        "selected_granularity_over_tokens_old-run.png",
        "selected_granularity_share_over_tokens_old-run.png",
        "selected_granularity_zoom_old-run.png",
    }
    for name in stale_names:
        (output_dir / name).write_bytes(b"stale")

    generate_figures(tmp_path, output_dir, refresh_counts=False, dpi=20)

    assert not any((output_dir / name).exists() for name in stale_names)


def test_multi_panel_size_figures_skip_empty_panels_and_share_y_limits(
    tmp_path,
    monkeypatch,
):
    from matplotlib.figure import Figure

    from src.evaluation.reporting_impl import plot_metric_vs_size

    saved_figures = []

    def capture_figure(figure, output_path, **_kwargs):
        saved_figures.append((Path(output_path), figure))

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    monkeypatch.setattr("src.evaluation.reporting_impl.plt.close", lambda _figure: None)

    rows = [
        {
            "sampling_mode": "nested-random",
            "model_variant": "slicing",
            "resolved_sampling_mode": "global",
            "non_embedding_parameters": 100,
            "loss": 1.0,
        },
        {
            "sampling_mode": "nested-random",
            "model_variant": "slicing",
            "resolved_sampling_mode": "global",
            "non_embedding_parameters": 200,
            "loss": 2.0,
        },
        {
            "sampling_mode": "nested-random",
            "model_variant": "concat",
            "resolved_sampling_mode": "per_block",
            "non_embedding_parameters": 100,
            "loss": 10.0,
        },
        {
            "sampling_mode": "nested-random",
            "model_variant": "concat",
            "resolved_sampling_mode": "per_block",
            "non_embedding_parameters": 200,
            "loss": 20.0,
        },
    ]
    panel_specs = [
        ("nested-random", "slicing", "global"),
        ("nested-random", "concat", "per_block"),
        ("nested-all", "slicing", None),
    ]

    output_paths = plot_metric_vs_size(
        rows,
        metric_name="loss",
        ylabel="Loss",
        output_path=tmp_path / "loss_vs_size.png",
        panel_specs=panel_specs,
        include_individual_panels=True,
    )

    assert [path.name for path in output_paths] == [
        "loss_vs_size.png",
        "loss_vs_size__nested_random_slicing_global.png",
        "loss_vs_size__nested_random_concat_per_block.png",
    ]
    assert len(saved_figures) == 3

    combined_figure = saved_figures[0][1]
    displayed_axes = [axis for axis in combined_figure.axes if axis.get_visible()]
    assert len(displayed_axes) == 2
    assert displayed_axes[0].get_ylim() == pytest.approx(displayed_axes[1].get_ylim())
    assert displayed_axes[0].get_ylim() == pytest.approx((-0.52, 21.52))
    assert all(
        text.get_text() != "No numeric points found"
        for axis in displayed_axes
        for text in axis.texts
    )

    for _, panel_figure in saved_figures[1:]:
        assert panel_figure.axes[0].get_ylim() == pytest.approx(
            displayed_axes[0].get_ylim()
        )


def test_multi_panel_size_figure_is_not_written_without_numeric_panels(
    tmp_path,
    monkeypatch,
):
    from matplotlib.figure import Figure

    from src.evaluation.reporting_impl import plot_metric_vs_size

    saved_paths = []
    monkeypatch.setattr(
        Figure,
        "savefig",
        lambda _figure, output_path, **_kwargs: saved_paths.append(Path(output_path)),
    )

    output_paths = plot_metric_vs_size(
        [
            {
                "sampling_mode": "nested-random",
                "model_variant": "slicing",
                "resolved_sampling_mode": "global",
                "non_embedding_parameters": 100,
                "loss": "",
            }
        ],
        metric_name="loss",
        ylabel="Loss",
        output_path=tmp_path / "loss_vs_size.png",
        panel_specs=[("nested-random", "slicing", "global")],
    )

    assert output_paths == []
    assert saved_paths == []


def test_perplexity_size_figures_add_loss_axes_without_duplicate_curves(
    tmp_path,
    monkeypatch,
):
    from matplotlib.figure import Figure

    from src.evaluation.reporting_impl import plot_metric_vs_size

    saved_figures = []
    monkeypatch.setattr(
        Figure,
        "savefig",
        lambda figure, output_path, **_kwargs: saved_figures.append(
            (Path(output_path), figure)
        ),
    )
    monkeypatch.setattr("src.evaluation.reporting_impl.plt.close", lambda _figure: None)
    rows = [
        {
            "sampling_mode": "nested-random",
            "model_variant": "slicing",
            "resolved_sampling_mode": "global",
            "non_embedding_parameters": 100,
            "perplexity": 2.0,
        },
        {
            "sampling_mode": "nested-random",
            "model_variant": "slicing",
            "resolved_sampling_mode": "global",
            "non_embedding_parameters": 200,
            "perplexity": 0.0,
        },
        {
            "sampling_mode": "nested-random",
            "model_variant": "concat",
            "resolved_sampling_mode": "global",
            "non_embedding_parameters": 100,
            "perplexity": 4.0,
        },
        {
            "sampling_mode": "nested-random",
            "model_variant": "slicing",
            "resolved_sampling_mode": "per_block",
            "non_embedding_parameters": 100,
            "perplexity": 8.0,
        },
    ]
    output_paths = plot_metric_vs_size(
        rows,
        metric_name="perplexity",
        ylabel="Perplexity",
        output_path=tmp_path / "ppl_vs_size.png",
        panel_specs=[
            ("nested-random", "slicing", "global"),
            ("nested-random", "concat", "global"),
            ("nested-random", "slicing", "per_block"),
        ],
        include_individual_panels=True,
        dpi=20,
    )

    assert len(output_paths) == 4
    combined = saved_figures[0][1]
    visible_axes = [axis for axis in combined.axes if axis.get_visible()]
    assert [len(axis.child_axes) for axis in visible_axes] == [0, 1, 1]
    assert all(
        value > 0.0
        for axis in visible_axes
        for line in axis.lines
        for value in line.get_ydata()
    )
    assert sum(len(axis.lines) for axis in visible_axes) == 3
    for axis in (visible_axes[1], visible_axes[2]):
        [loss_axis] = axis.child_axes
        assert loss_axis.get_ylabel() == "Loss (nats/token)"
        forward, inverse = loss_axis._functions
        assert forward([2.0])[0] == pytest.approx(0.69314718056)
        assert inverse([0.69314718056])[0] == pytest.approx(2.0)
    for _, panel_figure in saved_figures[1:]:
        assert len(panel_figure.axes[0].child_axes) == 1


@pytest.mark.parametrize(
    ("metric_name", "ylabel", "output_name"),
    [
        ("loss", "Loss", "loss_vs_size.png"),
        ("perplexity", "Perplexity", "ppl_vs_size.png"),
    ],
)
def test_panelgrad_is_included_in_size_plot_panels(
    tmp_path,
    metric_name,
    ylabel,
    output_name,
):
    from src.evaluation.reporting_impl import plot_metric_vs_size
    from src.evaluation.reporting_styles import SIZE_PLOT_PANELS_WITH_SAMPLING

    row = {
        "sampling_mode": "nested-random",
        "model_variant": "slicing",
        "resolved_sampling_mode": "adaptive_global",
        "adaptive_sampler_strategy": "panelgrad",
        "controller_method_family": "panelgrad_gradient_rms",
        "controller_method_version": 1,
        "controller_scope": "global",
        "non_embedding_parameters": 100,
        metric_name: 2.0,
    }

    output_paths = plot_metric_vs_size(
        [row],
        metric_name=metric_name,
        ylabel=ylabel,
        output_path=tmp_path / output_name,
        panel_specs=SIZE_PLOT_PANELS_WITH_SAMPLING,
        include_individual_panels=True,
        dpi=20,
    )

    assert [path.name for path in output_paths] == [
        output_name,
        output_name.replace(
            ".png", "__nested_random_slicing_panelgrad_global.png"
        ),
    ]


def test_uniform_global_windows_are_included_in_ppl_vs_size_panels(
    tmp_path,
    monkeypatch,
):
    from matplotlib.figure import Figure

    from src.evaluation.reporting_impl import plot_metric_vs_size
    from src.evaluation.reporting_styles import SIZE_PLOT_PANELS_WITH_SAMPLING

    rows = [
        {
            "run_id": f"uniform-window-{interval}",
            "sampling_mode": "nested-random",
            "model_variant": "slicing",
            "resolved_sampling_mode": "global",
            "global_sampling_interval_steps": interval,
            "correction_mode": "none",
            "non_embedding_parameters": 100 + interval,
            "perplexity": 2.0 + interval / 100.0,
        }
        for interval in (25, 50)
    ]
    saved_figures = []
    monkeypatch.setattr(
        Figure,
        "savefig",
        lambda figure, output_path, **_kwargs: saved_figures.append(
            (Path(output_path), figure)
        ),
    )

    output_paths = plot_metric_vs_size(
        rows,
        metric_name="perplexity",
        ylabel="Perplexity",
        output_path=tmp_path / "ppl_vs_size.png",
        panel_specs=SIZE_PLOT_PANELS_WITH_SAMPLING,
        include_individual_panels=True,
        dpi=20,
    )

    assert [path.name for path in output_paths] == [
        "ppl_vs_size.png",
        "ppl_vs_size__nested_random_slicing_uniform_global_window.png",
    ]
    main_figure = saved_figures[0][1]
    [axis] = main_figure.axes
    assert axis.get_title() == (
        "Nested-random · Slicing · Uniform global windows"
    )
    assert {line.get_label() for line in axis.lines} == {
        "Uniform global (H=25)",
        "Uniform global (H=50)",
    }


def test_uniform_global_window_panel_repeats_h1_as_dotted_reference():
    import matplotlib.pyplot as plt

    from src.evaluation.reporting_impl import plot_metric_vs_size_panel

    rows = [
        {
            "run_id": f"uniform-window-{interval}",
            "sampling_mode": "nested-random",
            "model_variant": "slicing",
            "resolved_sampling_mode": "global",
            "global_sampling_interval_steps": interval,
            "correction_mode": "none",
            "non_embedding_parameters": 100 + interval,
            "perplexity": 2.0 + interval / 100.0,
        }
        for interval in (1, 5, 25, 50)
    ]
    figure, axis = plt.subplots()

    plot_metric_vs_size_panel(
        axis,
        rows,
        metric_name="perplexity",
        ylabel="Perplexity",
        sampling_mode="nested-random",
        variant_label="slicing",
        sampling_label="uniform_global_window",
    )

    lines = {line.get_label(): line for line in axis.lines}
    assert list(lines) == [
        "Global sampling (H=1 reference)",
        "Uniform global (H=5)",
        "Uniform global (H=25)",
        "Uniform global (H=50)",
    ]
    assert lines["Global sampling (H=1 reference)"].get_linestyle() == ":"
    assert all(
        line.get_linestyle() != ":"
        for label, line in lines.items()
        if label != "Global sampling (H=1 reference)"
    )
    plt.close(figure)


def test_balanced_global_windows_use_iid_reference_and_solid_balanced_h1():
    import matplotlib.pyplot as plt

    from src.evaluation.reporting_impl import (
        plot_metric_vs_size_panel,
        scaling_curve_sampling_label,
    )
    from src.evaluation.reporting_styles import PPL_VS_SIZE_FIGURE_SPECS

    balanced_rows = [
        {
            "run_id": f"balanced-window-{interval}",
            "sampling_mode": "nested-random",
            "model_variant": "slicing",
            "resolved_sampling_mode": "global",
            "global_sampling_schedule": "balanced_cycle",
            "global_sampling_schedule_version": 1,
            "global_sampling_interval_steps": interval,
            "correction_mode": "none",
            "non_embedding_parameters": 100 + interval,
            "perplexity": 2.0 + interval / 100.0,
        }
        for interval in (50, 1, 25, 5)
    ]
    iid_h50_row = {
        **balanced_rows[0],
        "run_id": "iid-window-50",
        "global_sampling_schedule": "random_with_replacement",
    }
    iid_h1_row = {
        **balanced_rows[1],
        "run_id": "iid-h1-reference",
        "global_sampling_schedule": "random_with_replacement",
    }
    assert [scaling_curve_sampling_label(row) for row in balanced_rows] == [
        "balanced_global_h50",
        "balanced_global_h1",
        "balanced_global_h25",
        "balanced_global_h5",
    ]
    assert scaling_curve_sampling_label(iid_h50_row) == "uniform_global_h50"
    assert scaling_curve_sampling_label(iid_h1_row) == "global"
    assert any(
        spec["output_name"] == "ppl_vs_size_balanced_global_window.png"
        for spec in PPL_VS_SIZE_FIGURE_SPECS
    )

    figure, axis = plt.subplots()
    plot_metric_vs_size_panel(
        axis,
        [iid_h50_row, iid_h1_row, *balanced_rows],
        metric_name="perplexity",
        ylabel="Perplexity",
        sampling_mode="nested-random",
        variant_label="slicing",
        sampling_label="balanced_global_window",
    )

    lines = {line.get_label(): line for line in axis.lines}
    assert list(lines) == [
        "Random global (IID H=1 reference)",
        "Balanced global (H=1)",
        "Balanced global (H=5)",
        "Balanced global (H=25)",
        "Balanced global (H=50)",
    ]
    assert lines["Random global (IID H=1 reference)"].get_linestyle() == ":"
    assert lines["Balanced global (H=1)"].get_linestyle() != ":"
    assert "Uniform global (H=50)" not in lines
    plt.close(figure)


def test_plot_filters_select_only_uncorrected_slicing_rows():
    from src.evaluation.reporting_impl import filter_plot_rows

    rows = [
        {
            "run_id": "slicing-none",
            "model_variant": "matformer_llama",
            "correction_mode": "none",
        },
        {
            "run_id": "slicing-gmc",
            "model_variant": "matformer_llama",
            "correction_mode": "gmc",
        },
        {
            "run_id": "slicing-lmc",
            "model_variant": "matformer_llama",
            "correction_mode": "lmc",
        },
        {
            "run_id": "concat-none",
            "model_variant": "cat_llama",
            "correction_mode": "none",
        },
        {
            "run_id": "standalone",
            "sampling_mode": "standalone",
            "model_family": "standalone",
        },
    ]

    filtered = filter_plot_rows(
        rows,
        variants=["slicing"],
        corrections=["none"],
    )

    assert [row["run_id"] for row in filtered] == ["slicing-none"]


def _bayesian_size_row(
    *,
    run_id: str,
    seed: int,
    parameters: int,
    loss: float,
    process_noise: float,
    reset_enabled: bool = False,
    reset_policy: str = "full_prior",
    reset_interval: int | None = None,
) -> dict:
    return {
        "run_id": run_id,
        "run_seed": seed,
        "sampling_mode": "nested-random",
        "model_variant": "concat",
        "resolved_sampling_mode": "adaptive_global",
        "adaptive_sampler_strategy": "thompson",
        "controller_method_family": "bayesian_gaussian_linear_thompson",
        "controller_method_version": 1,
        "controller_scope": "global",
        "controller_decision_interval_steps": 50,
        "controller_observation_noise_variance": 1e-7,
        "controller_process_noise_covariance": process_noise,
        "controller_prior_mean": [0.0] * 5,
        "controller_prior_covariance": [1e-4] + [1e-6] * 4,
        "controller_reset_enabled": reset_enabled,
        "controller_reset_policy": reset_policy,
        "controller_reset_interval_steps": reset_interval,
        "controller_acquisition_policy": "balanced_global",
        "controller_acquisition_passes": 1,
        "pre_nested_warmup_enabled": True,
        "pre_nested_warmup_policy": "balanced_global",
        "pre_nested_warmup_duration": 500,
        "pre_nested_warmup_action_interval_steps": 50,
        "training_token_budget": 100_000_000,
        "training_learning_rate": 0.001,
        "training_scheduler_name": "cosine",
        "non_embedding_parameters": parameters,
        "loss": loss,
    }


def test_size_plot_unifies_global_bayesian_ts_aliases_without_changing_identity():
    from src.evaluation.reporting_impl import (
        panel_sampling_matches,
        scaling_curve_sampling_label,
    )

    rows = [
        _bayesian_size_row(
            run_id="no-reset",
            seed=42,
            parameters=100,
            loss=1.0,
            process_noise=0.0,
        ),
        _bayesian_size_row(
            run_id="full-prior",
            seed=42,
            parameters=100,
            loss=1.1,
            process_noise=0.0,
            reset_enabled=True,
            reset_interval=2000,
        ),
        _bayesian_size_row(
            run_id="acquisition-only",
            seed=42,
            parameters=100,
            loss=1.2,
            process_noise=0.0,
            reset_enabled=True,
            reset_policy="acquisition_only",
            reset_interval=2000,
        ),
    ]

    identities = [scaling_curve_sampling_label(row) for row in rows]

    assert identities == [
        "probabilistic_global_thompson",
        "probabilistic_global_thompson_reset",
        "probabilistic_global_thompson_acquisition_only",
    ]
    assert all(
        panel_sampling_matches(identity, "probabilistic_global_thompson")
        for identity in identities
    )


def test_size_plot_keeps_distinct_contracts_and_aggregates_seeds_with_min_max_band():
    import matplotlib.pyplot as plt

    from src.evaluation.reporting_impl import plot_metric_vs_size_panel

    rows = []
    for seed, offset in ((42, 0.0), (43, 0.2)):
        rows.extend(
            [
                _bayesian_size_row(
                    run_id=f"q0-s{seed}",
                    seed=seed,
                    parameters=100,
                    loss=1.0 + offset,
                    process_noise=0.0,
                ),
                _bayesian_size_row(
                    run_id=f"q0-s{seed}",
                    seed=seed,
                    parameters=200,
                    loss=2.0 + offset,
                    process_noise=0.0,
                ),
            ]
        )
    rows.extend(
        [
            _bayesian_size_row(
                run_id="q1e-10-s42",
                seed=42,
                parameters=100,
                loss=0.9,
                process_noise=1e-10,
            ),
            _bayesian_size_row(
                run_id="q1e-10-s42",
                seed=42,
                parameters=200,
                loss=1.9,
                process_noise=1e-10,
            ),
            _bayesian_size_row(
                run_id="reset-s42",
                seed=42,
                parameters=100,
                loss=1.2,
                process_noise=0.0,
                reset_enabled=True,
                reset_interval=2000,
            ),
            _bayesian_size_row(
                run_id="reset-s42",
                seed=42,
                parameters=200,
                loss=2.2,
                process_noise=0.0,
                reset_enabled=True,
                reset_interval=2000,
            ),
            _bayesian_size_row(
                run_id="acquisition-s42",
                seed=42,
                parameters=100,
                loss=1.3,
                process_noise=0.0,
                reset_enabled=True,
                reset_policy="acquisition_only",
                reset_interval=2000,
            ),
            _bayesian_size_row(
                run_id="acquisition-s42",
                seed=42,
                parameters=200,
                loss=2.3,
                process_noise=0.0,
                reset_enabled=True,
                reset_policy="acquisition_only",
                reset_interval=2000,
            ),
        ]
    )

    figure, axis = plt.subplots()
    plot_metric_vs_size_panel(
        axis,
        rows,
        metric_name="loss",
        ylabel="Loss",
        sampling_mode="nested-random",
        variant_label="concat",
        sampling_label="probabilistic_global_thompson",
    )

    assert axis.get_title() == "Nested-random · Concat · Bayesian global TS"
    lines_by_label = {line.get_label(): line for line in axis.lines}
    assert set(lines_by_label) == {
        "No reset · Q=0 · n=2 seeds",
        "No reset · Q=1e−10",
        "Full-prior · K=2k",
        "Acquisition-only · K=2k",
    }
    assert list(lines_by_label["No reset · Q=0 · n=2 seeds"].get_xdata()) == [
        100.0,
        200.0,
    ]
    assert list(lines_by_label["No reset · Q=0 · n=2 seeds"].get_ydata()) == (
        pytest.approx([1.1, 2.1])
    )
    assert len(axis.collections) == 1
    plt.close(figure)


def test_size_plot_missing_contract_metadata_falls_back_to_run_identity():
    from src.evaluation.reporting_impl import group_size_plot_rows

    rows = [
        {
            "run_id": run_id,
            "sampling_mode": "nested-random",
            "model_variant": "concat",
            "resolved_sampling_mode": "adaptive_global",
            "adaptive_sampler_strategy": "thompson",
            "controller_method_family": "bayesian_gaussian_linear_thompson",
            "controller_method_version": 1,
            "controller_scope": "global",
            "non_embedding_parameters": 100,
            "loss": loss,
        }
        for run_id, loss in (("historical-a", 1.0), ("historical-b", 2.0))
    ]

    grouped = group_size_plot_rows(rows)

    assert len(grouped) == 2
    assert {group[0]["run_id"] for group in grouped.values()} == {
        "historical-a",
        "historical-b",
    }


def test_size_plot_shared_limits_include_seed_band_without_spurious_zero():
    import matplotlib.pyplot as plt

    from src.evaluation.reporting_impl import axis_numeric_y_values

    figure, axis = plt.subplots()
    axis.fill_between([1, 2], [10, 20], [12, 22])

    values = axis_numeric_y_values(axis)

    assert min(values) == pytest.approx(10.0)
    assert max(values) == pytest.approx(22.0)
    plt.close(figure)


def test_size_plot_reserves_a_unique_standalone_reference_color():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    from src.evaluation.reporting_impl import plot_metric_vs_size_panel
    from src.evaluation.reporting_styles import (
        STANDALONE_REFERENCE_COLOR,
        STANDALONE_REFERENCE_EDGE_COLOR,
    )

    rows = [
        _bayesian_size_row(
            run_id=f"trained-{index}",
            seed=42,
            parameters=100 + index,
            loss=1.0 + index / 10,
            process_noise=10 ** (-index),
        )
        for index in range(10)
    ]
    rows.append(
        {
            "run_id": "standalone",
            "sampling_mode": "standalone",
            "model_family": "standalone",
            "non_embedding_parameters": 105,
            "loss": 1.25,
        }
    )

    figure, axis = plt.subplots()
    plot_metric_vs_size_panel(
        axis,
        rows,
        metric_name="loss",
        ylabel="Loss",
        sampling_mode="nested-random",
        variant_label="concat",
        sampling_label="probabilistic_global_thompson",
    )

    standalone = axis.collections[-1]
    standalone_color = tuple(standalone.get_facecolors()[0])
    assert standalone_color == pytest.approx(to_rgba(STANDALONE_REFERENCE_COLOR))
    assert tuple(standalone.get_edgecolors()[0]) == pytest.approx(
        to_rgba(STANDALONE_REFERENCE_EDGE_COLOR)
    )
    assert standalone.get_zorder() > max(line.get_zorder() for line in axis.lines)
    assert all(
        to_rgba(line.get_color()) != to_rgba(STANDALONE_REFERENCE_COLOR)
        for line in axis.lines
    )
    plt.close(figure)


def test_split_size_plot_cannot_override_reserved_standalone_color():
    from src.evaluation.reporting_impl import (
        comparison_series_style,
        resolve_plot_style,
    )
    from src.evaluation.reporting_styles import STANDALONE_REFERENCE_COLOR

    style = resolve_plot_style("nested_split_no_corrections")
    style["series_colors"]["standalone"] = "tab:brown"

    assert comparison_series_style("standalone", style)["color"] == (
        STANDALONE_REFERENCE_COLOR
    )


def test_scaling_metadata_enrichment_records_seed_independent_ts_contract(tmp_path):
    from src.evaluation.reporting_io import enrich_scaling_metadata_from_run_config

    run_dir = tmp_path / "ts-run"
    run_dir.mkdir()
    config = {
        "comparison_control_inputs": {
            "root_seed": 43,
            "dataset_name": "example/data",
            "dataset_config_name": "sample",
            "dataset_split": "train",
            "context_length": 1024,
            "batch_size_per_process": 4,
            "precision": "bf16",
            "tokenizer_name": "example/tokenizer",
        },
        "run": {"run_id": "ts-run", "seed": 43},
        "model": {
            "adaptive_controller": {
                "method_family": "bayesian_gaussian_linear_thompson",
                "method_version": 1,
                "scope": "global",
                "decision_interval_steps": 50,
                "observation_noise_variance": 1e-7,
                "process_noise_covariance_input": 1e-10,
                "prior_mean_input": [0.0] * 5,
                "prior_covariance_input": [1e-4] + [1e-6] * 4,
                "reset": {
                    "enabled": True,
                    "policy": "acquisition_only",
                    "interval_steps": 2000,
                    "acquisition_policy": "balanced_global",
                    "acquisition_passes": 1,
                },
            }
        },
        "training": {
            "token_budget": 100_000_000,
            "resolved_learning_rate": 0.001,
            "max_steps": 24_415,
            "optimizer": {"name": "adamw", "kwargs": {"weight_decay": 0.1}},
            "scheduler": {"name": "cosine", "kwargs": {"warmup_steps": 2000}},
            "pre_nested_warmup": {
                "enabled": True,
                "policy": "balanced_global",
                "duration": 500,
                "action_interval_steps": 50,
            },
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    [row] = enrich_scaling_metadata_from_run_config(
        tmp_path,
        [{"run_id": "ts-run", "_source_csv": str(run_dir / "scaling_results.csv")}],
    )

    assert row["run_seed"] == 43
    assert row["controller_process_noise_covariance"] == 1e-10
    assert row["controller_reset_policy"] == "acquisition_only"
    assert row["controller_reset_interval_steps"] == 2000
    assert row["pre_nested_warmup_duration"] == 500
    assert row["training_token_budget"] == 100_000_000
    assert row["training_scheduler_name"] == "cosine"
    assert row["training_dataset_name"] == "example/data"


@pytest.mark.parametrize(
    "row, expected_identity, expected_display_label",
    [
        (
            {
                "sampling_mode": "nested-random",
                "resolved_sampling_mode": "adaptive_global",
                "adaptive_sampler_strategy": "thompson",
                "controller_method_family": "bayesian_gaussian_linear_thompson",
                "controller_method_version": 1,
                "controller_scope": "global",
            },
            "probabilistic_global_thompson",
            "probabilistic global thompson",
        ),
        (
            {
                "sampling_mode": "nested-random",
                "resolved_sampling_mode": "adaptive_global",
                "adaptive_sampler_strategy": "thompson",
                "controller_method_family": "bayesian_gaussian_linear_thompson",
                "controller_method_version": 1,
                "controller_scope": "global",
                "controller_reset_enabled": True,
            },
            "probabilistic_global_thompson_reset",
            "probabilistic global thompson reset",
        ),
        (
            {
                "sampling_mode": "nested-random",
                "resolved_sampling_mode": "adaptive_global",
                "adaptive_sampler_strategy": "thompson",
                "controller_method_family": "bayesian_gaussian_linear_thompson",
                "controller_method_version": 1,
                "controller_scope": "global",
                "controller_reset_enabled": True,
                "controller_reset_policy": "acquisition_only",
            },
            "probabilistic_global_thompson_acquisition_only",
            "probabilistic global thompson acquisition-only",
        ),
        (
            {
                "sampling_mode": "nested-random",
                "resolved_sampling_mode": "adaptive_per_block",
                "adaptive_sampler_strategy": "thompson",
                "controller_method_family": "bayesian_gaussian_linear_thompson",
                "controller_method_version": 1,
                "controller_scope": "per_block",
            },
            "probabilistic_per_block_thompson",
            "probabilistic per-block thompson",
        ),
        (
            {
                "sampling_mode": "nested-random",
                "resolved_sampling_mode": "adaptive_per_block",
                "adaptive_sampler_strategy": "thompson",
            },
            "adaptive_per_block_thompson",
            "legacy heuristic thompson",
        ),
    ],
)
def test_reporting_uses_explicit_provenance_to_distinguish_bayesian_and_legacy_thompson(
    row,
    expected_identity,
    expected_display_label,
):
    from src.evaluation.reporting import display_sampling_label_for_curve
    from src.evaluation.reporting_impl import scaling_curve_sampling_label

    identity = scaling_curve_sampling_label(row)

    assert identity == expected_identity
    assert display_sampling_label_for_curve(identity) == expected_display_label


def test_reporting_preserves_ucb_identity_display_and_style():
    from src.evaluation.reporting import display_sampling_label_for_curve
    from src.evaluation.reporting_impl import (
        scaling_curve_sampling_label,
        scaling_curve_style,
    )
    from src.evaluation.reporting_styles import (
        SCALING_SAMPLING_MARKERS,
        SCALING_SAMPLING_TONES,
    )

    row = {
        "sampling_mode": "nested-random",
        "model_variant": "slicing",
        "resolved_sampling_mode": "adaptive_per_block",
        "adaptive_sampler_strategy": "ucb",
    }

    assert scaling_curve_sampling_label(row) == "adaptive_per_block_ucb"
    assert display_sampling_label_for_curve("adaptive_per_block_ucb") == (
        "adaptive per-block ucb"
    )
    assert SCALING_SAMPLING_MARKERS["adaptive_per_block_ucb"] == "X"
    assert SCALING_SAMPLING_TONES["adaptive_per_block_ucb"] == pytest.approx(0.55)
    assert scaling_curve_style([row])["marker"] == "X"


def test_reporting_defines_distinct_styles_for_each_bayesian_scope():
    from src.evaluation.reporting_styles import (
        SCALING_SAMPLING_MARKERS,
        SCALING_SAMPLING_TONES,
        SIZE_PLOT_PANELS_WITH_SAMPLING,
    )

    bayesian_identities = {
        "probabilistic_global_thompson",
        "probabilistic_global_thompson_acquisition_only",
        "probabilistic_per_block_thompson",
    }

    assert bayesian_identities <= set(SCALING_SAMPLING_MARKERS)
    assert bayesian_identities <= set(SCALING_SAMPLING_TONES)
    assert (
        SCALING_SAMPLING_MARKERS["probabilistic_global_thompson"]
        != (SCALING_SAMPLING_MARKERS["probabilistic_per_block_thompson"])
    )
    panel_identities = {panel[2] for panel in SIZE_PLOT_PANELS_WITH_SAMPLING}
    assert "uniform_global_window" in panel_identities
    assert "probabilistic_global_thompson" in panel_identities
    assert "probabilistic_global_thompson_acquisition_only" not in panel_identities
    assert "probabilistic_global_thompson_reset" not in panel_identities
    assert "probabilistic_per_block_thompson" in panel_identities
    assert "panelgrad_global" in panel_identities
    assert SCALING_SAMPLING_MARKERS["panelgrad_global"] == "d"
    assert SCALING_SAMPLING_TONES["panelgrad_global"] == pytest.approx(0.42)
    assert SCALING_SAMPLING_MARKERS["uniform_global_window"] == "s"
    assert SCALING_SAMPLING_TONES["uniform_global_window"] == pytest.approx(0.08)
    assert "adaptive_per_block_thompson" not in panel_identities
    assert "adaptive_per_block_thompson" not in SCALING_SAMPLING_MARKERS
    assert "adaptive_per_block_thompson" not in SCALING_SAMPLING_TONES


def test_reporting_excludes_legacy_heuristic_thompson_from_plot_rows():
    from src.evaluation.reporting_impl import filter_plot_rows

    legacy_row = {
        "run_id": "legacy",
        "resolved_sampling_mode": "adaptive_per_block",
        "adaptive_sampler_strategy": "thompson",
    }
    bayesian_per_block_row = {
        "run_id": "bayesian-per-block",
        "resolved_sampling_mode": "adaptive_per_block",
        "adaptive_sampler_strategy": "thompson",
        "controller_method_family": "bayesian_gaussian_linear_thompson",
        "controller_method_version": 1,
        "controller_scope": "per_block",
    }
    bayesian_global_row = {
        "run_id": "bayesian-global",
        "resolved_sampling_mode": "adaptive_global",
        "adaptive_sampler_strategy": "thompson",
        "controller_method_family": "bayesian_gaussian_linear_thompson",
        "controller_method_version": 1,
        "controller_scope": "global",
    }
    ucb_row = {
        "run_id": "ucb",
        "resolved_sampling_mode": "adaptive_per_block",
        "adaptive_sampler_strategy": "ucb",
    }

    filtered = filter_plot_rows(
        [legacy_row, bayesian_per_block_row, bayesian_global_row, ucb_row]
    )

    assert [row["run_id"] for row in filtered] == [
        "bayesian-per-block",
        "bayesian-global",
        "ucb",
    ]


def test_controller_timeline_extracts_global_windows_and_token_boundaries(tmp_path):
    from src.evaluation.reporting_io import iter_controller_granularity_timelines

    run_id = "global-controller"
    events = [
        {
            "schema_version": 1,
            "run_id": run_id,
            "event_type": "initial_boundary",
            "method_family": "bayesian_gaussian_linear_thompson",
            "method_version": 1,
            "strategy": "thompson",
            "scope": "global",
            "boundary_step": 2,
            "window_index": 0,
            "selected_action": {"global_granularity": "micro"},
        },
        _completed_controller_window(
            run_id=run_id,
            scope="global",
            window_index=0,
            start_step=2,
            end_step=5,
            action={"global_granularity": "medium"},
        ),
        {
            "schema_version": 1,
            "run_id": run_id,
            "event_type": "terminal_incomplete",
            "method_family": "bayesian_gaussian_linear_thompson",
            "method_version": 1,
            "strategy": "thompson",
            "scope": "global",
            "boundary_step": 5,
            "window_index": 1,
            "completed_optimizer_steps": 1,
            "decision_interval_steps": 3,
            "action": {"global_granularity": "full"},
            "observation_emitted": False,
        },
    ]
    _write_controller_timeline_run(
        tmp_path,
        run_id=run_id,
        scope="global",
        events=events,
        expected_tokens_per_step=10,
        token_budget=55,
    )

    [timeline] = list(iter_controller_granularity_timelines(tmp_path))

    assert timeline.run_id == run_id
    assert timeline.scope == "global"
    assert timeline.block_count == 1
    assert timeline.row_labels == ("all blocks",)
    assert timeline.ordered_granularities == ("micro", "medium", "full")
    assert [
        (
            window.window_index,
            window.start_step,
            window.end_step,
            window.start_tokens,
            window.end_tokens,
            window.block_granularities,
            window.terminal_incomplete,
        )
        for window in timeline.windows
    ] == [
        (0, 2, 5, 20, 50, ("medium",), False),
        (1, 5, 6, 50, 55, ("full",), True),
    ]


def test_controller_timeline_extracts_exact_sixteen_block_profile(tmp_path):
    from src.evaluation.reporting_io import iter_controller_granularity_timelines

    run_id = "per-block-controller"
    profile = tuple(
        "micro" if block_index % 2 == 0 else "full" for block_index in range(16)
    )
    events = [
        _completed_controller_window(
            run_id=run_id,
            scope="per_block",
            window_index=0,
            start_step=0,
            end_step=2,
            action={"block_granularities": list(profile)},
        )
    ]
    _write_controller_timeline_run(
        tmp_path,
        run_id=run_id,
        scope="per_block",
        events=events,
        block_count=16,
    )

    [timeline] = list(iter_controller_granularity_timelines(tmp_path))

    assert timeline.block_count == 16
    assert timeline.row_labels == tuple(str(index) for index in range(1, 17))
    assert timeline.windows[0].block_granularities == profile


def test_controller_selection_share_plots_one_exact_fraction_line_per_granularity(
    tmp_path,
    monkeypatch,
):
    from matplotlib.figure import Figure

    from src.evaluation.reporting import (
        plot_selected_granularity_share_over_tokens,
    )
    from src.evaluation.reporting_io import iter_controller_granularity_timelines

    run_id = "per-block-share"
    _write_controller_timeline_run(
        tmp_path,
        run_id=run_id,
        scope="per_block",
        block_count=3,
        events=[
            _completed_controller_window(
                run_id=run_id,
                scope="per_block",
                window_index=0,
                start_step=0,
                end_step=2,
                action={"block_granularities": ["micro", "micro", "full"]},
            ),
            _completed_controller_window(
                run_id=run_id,
                scope="per_block",
                window_index=1,
                start_step=2,
                end_step=4,
                action={"block_granularities": ["medium", "full", "full"]},
            ),
        ],
    )
    [timeline] = list(iter_controller_granularity_timelines(tmp_path))
    saved_figures = []
    monkeypatch.setattr(
        Figure,
        "savefig",
        lambda figure, output_path, **_kwargs: saved_figures.append(
            (Path(output_path), figure)
        ),
    )

    output_path = tmp_path / "share.png"
    result = plot_selected_granularity_share_over_tokens(
        timeline,
        output_path,
        dpi=40,
    )

    assert result == output_path
    [(saved_path, figure)] = saved_figures
    assert saved_path == output_path
    assert [axis.get_title() for axis in figure.axes] == [
        "micro",
        "medium",
        "full",
    ]
    assert figure.axes[-1].get_xlabel() == "Total training tokens"
    assert all(
        axis.get_ylabel() == "Selected\nblock fraction"
        for axis in figure.axes
    )
    expected_endpoints = {
        "micro": (2 / 3, 0.0),
        "medium": (0.0, 1 / 3),
        "full": (1 / 3, 2 / 3),
    }
    for axis in figure.axes:
        y_values = axis.lines[0].get_ydata()
        start, end = expected_endpoints[axis.get_title()]
        assert y_values[0] == pytest.approx(start)
        assert y_values[-1] == pytest.approx(end)
        assert axis.get_ylim() == pytest.approx((-0.05, 1.05))


def test_global_controller_selection_share_is_binary(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    from src.evaluation.reporting import (
        plot_selected_granularity_share_over_tokens,
    )
    from src.evaluation.reporting_io import iter_controller_granularity_timelines

    run_id = "global-share"
    _write_controller_timeline_run(
        tmp_path,
        run_id=run_id,
        scope="global",
        events=[
            _completed_controller_window(
                run_id=run_id,
                scope="global",
                window_index=0,
                start_step=0,
                end_step=2,
                action={"global_granularity": "micro"},
            ),
            _completed_controller_window(
                run_id=run_id,
                scope="global",
                window_index=1,
                start_step=2,
                end_step=4,
                action={"global_granularity": "full"},
            ),
        ],
    )
    [timeline] = list(iter_controller_granularity_timelines(tmp_path))
    saved_figures = []
    monkeypatch.setattr(
        Figure,
        "savefig",
        lambda figure, _output_path, **_kwargs: saved_figures.append(figure),
    )

    plot_selected_granularity_share_over_tokens(
        timeline,
        tmp_path / "global-share.png",
        dpi=40,
    )

    [figure] = saved_figures
    assert all(
        axis.get_ylabel() == "Selected\n(0 or 1)"
        for axis in figure.axes
    )
    for axis in figure.axes:
        assert set(axis.lines[0].get_ydata()).issubset({0.0, 1.0})
        assert axis.lines[0].get_linestyle() == "None"
    assert [axis.get_title() for axis in figure.axes] == [
        "micro — 1/2 steps (50.0%)",
        "medium — 0/2 steps (0.0%)",
        "full — 1/2 steps (50.0%)",
    ]


def test_global_controller_timeline_uses_borderless_categorical_mesh(
    tmp_path,
    monkeypatch,
):
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure

    from src.evaluation.reporting import plot_selected_granularity_over_tokens
    from src.evaluation.reporting_io import iter_controller_granularity_timelines

    run_id = "global-mesh"
    _write_controller_timeline_run(
        tmp_path,
        run_id=run_id,
        scope="global",
        events=[
            _completed_controller_window(
                run_id=run_id,
                scope="global",
                window_index=index,
                start_step=index,
                end_step=index + 1,
                action={
                    "global_granularity": ("micro", "medium", "full")[index % 3]
                },
            )
            for index in range(300)
        ],
        expected_tokens_per_step=10,
        token_budget=3_000,
    )
    [timeline] = list(iter_controller_granularity_timelines(tmp_path))
    saved_figures = []
    monkeypatch.setattr(
        Figure,
        "savefig",
        lambda figure, _output_path, **_kwargs: saved_figures.append(figure),
    )

    plot_selected_granularity_over_tokens(
        timeline,
        tmp_path / "global-mesh.png",
        dpi=40,
    )

    [figure] = saved_figures
    timeline_axis = figure.axes[0]
    meshes = [
        collection
        for collection in timeline_axis.collections
        if isinstance(collection, QuadMesh)
    ]
    assert len(meshes) == 1
    assert len(timeline_axis.patches) == 0


def test_dense_per_block_selection_history_is_stacked_into_local_frequencies(
    tmp_path,
    monkeypatch,
):
    from matplotlib.figure import Figure

    from src.evaluation.reporting import (
        plot_granularity_selection_frequency_over_tokens,
    )
    from src.evaluation.reporting_io import iter_controller_granularity_timelines

    events = [
        _completed_controller_window(
            run_id="dense-per-block-share",
            scope="per_block",
            window_index=index,
            start_step=index,
            end_step=index + 1,
            action={"block_granularities": ["micro", "full"]},
        )
        for index in range(200)
    ]
    _write_controller_timeline_run(
        tmp_path,
        run_id="dense-per-block-share",
        scope="per_block",
        block_count=2,
        events=events,
        token_budget=2_000,
    )
    [timeline] = list(iter_controller_granularity_timelines(tmp_path))
    saved_figures = []
    monkeypatch.setattr(
        Figure,
        "savefig",
        lambda figure, _output_path, **_kwargs: saved_figures.append(figure),
    )

    plot_granularity_selection_frequency_over_tokens(
        timeline,
        tmp_path / "dense-share.png",
        dpi=40,
        max_bins=10,
    )

    [figure] = saved_figures
    [axis] = figure.axes
    assert axis.get_ylim() == pytest.approx((0.0, 100.0))
    assert len(axis.containers) == 3
    for bin_index in range(10):
        assert sum(
            container.patches[bin_index].get_height()
            for container in axis.containers
        ) == pytest.approx(100.0)
    assert [
        container.patches[0].get_height() for container in axis.containers
    ] == pytest.approx([50.0, 0.0, 50.0])
    assert "10 equal-token bins" in figure._suptitle.get_text()


def test_controller_timeline_streams_490_current_schema_events_to_exact_budget(
    tmp_path,
):
    from src.evaluation.reporting_io import iter_controller_granularity_timelines

    run_id = "global-controller-490-events"
    events = []
    for window_index in range(490):
        event = _completed_controller_window(
            run_id=run_id,
            scope="global",
            window_index=window_index,
            start_step=window_index * 2,
            end_step=(window_index + 1) * 2,
            action={
                "global_granularity": ("micro", "medium", "full")[window_index % 3]
            },
        )
        # Representative fields that must not be retained in compact windows.
        event["posterior_mean"] = [0.1, 0.2, 0.3]
        event["posterior_covariance"] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        events.append(event)
    _write_controller_timeline_run(
        tmp_path,
        run_id=run_id,
        scope="global",
        events=events,
        expected_tokens_per_step=10,
        token_budget=9_800,
    )

    [timeline] = list(iter_controller_granularity_timelines(tmp_path))

    assert len(timeline.windows) == 490
    assert timeline.windows[-1].end_tokens == timeline.token_budget == 9_800
    assert not hasattr(timeline.windows[-1], "posterior_covariance")


def test_controller_timeline_uses_confirmed_windows_from_failed_or_running_jobs(
    tmp_path,
):
    from src.evaluation.reporting_io import iter_controller_granularity_timelines

    failed_run_id = "failed-after-one-window"
    completed = _completed_controller_window(
        run_id=failed_run_id,
        scope="global",
        window_index=0,
        start_step=0,
        end_step=2,
        action={"global_granularity": "micro"},
    )
    _write_controller_timeline_run(
        tmp_path,
        run_id=failed_run_id,
        scope="global",
        events=[
            completed,
            {
                "schema_version": 1,
                "run_id": failed_run_id,
                "event_type": "controller_failure",
                "boundary_step": 3,
                "window_index": 1,
                "action": {"global_granularity": "full"},
                "posterior_updated": False,
                "new_action_selected": False,
            },
        ],
    )
    _write_controller_timeline_run(
        tmp_path,
        run_id="running-with-uncommitted-action",
        scope="global",
        events=[
            {
                "schema_version": 1,
                "run_id": "running-with-uncommitted-action",
                "event_type": "initial_boundary",
                "boundary_step": 0,
                "window_index": 0,
                "selected_action": {"global_granularity": "medium"},
            }
        ],
    )

    timelines = list(iter_controller_granularity_timelines(tmp_path))

    assert [timeline.run_id for timeline in timelines] == [failed_run_id]
    assert len(timelines[0].windows) == 1


def test_controller_timeline_warns_and_skips_malformed_runs_without_losing_valid_run(
    tmp_path,
):
    from src.evaluation.reporting_io import iter_controller_granularity_timelines

    malformed_run_id = "malformed-profile"
    _write_controller_timeline_run(
        tmp_path,
        run_id=malformed_run_id,
        scope="per_block",
        block_count=2,
        events=[
            _completed_controller_window(
                run_id=malformed_run_id,
                scope="per_block",
                window_index=0,
                start_step=0,
                end_step=2,
                action={"block_granularities": ["micro", "unknown"]},
            )
        ],
    )
    missing_config_dir = tmp_path / "missing-config"
    missing_config_dir.mkdir()
    (missing_config_dir / "controller_metrics.jsonl").write_text(
        json.dumps({"event_type": "completed_window"}) + "\n",
        encoding="utf-8",
    )
    valid_run_id = "valid-controller"
    _write_controller_timeline_run(
        tmp_path,
        run_id=valid_run_id,
        scope="global",
        events=[
            _completed_controller_window(
                run_id=valid_run_id,
                scope="global",
                window_index=0,
                start_step=0,
                end_step=2,
                action={"global_granularity": "medium"},
            )
        ],
    )

    with pytest.warns(RuntimeWarning) as warning_records:
        timelines = list(iter_controller_granularity_timelines(tmp_path))

    assert [timeline.run_id for timeline in timelines] == [valid_run_id]
    messages = [str(record.message) for record in warning_records]
    assert any(
        "malformed-profile" in message and "unknown" in message for message in messages
    )
    assert any(
        "missing-config" in message and "config.json" in message for message in messages
    )


def test_generate_figures_returns_grouped_global_and_stacked_per_block_pngs(
    tmp_path,
):
    global_run_id = "global-timeline"
    _write_controller_timeline_run(
        tmp_path,
        run_id=global_run_id,
        scope="global",
        events=[
            _completed_controller_window(
                run_id=global_run_id,
                scope="global",
                window_index=0,
                start_step=0,
                end_step=2,
                action={"global_granularity": "micro"},
            )
        ],
    )
    per_block_run_id = "per-block-timeline"
    _write_controller_timeline_run(
        tmp_path,
        run_id=per_block_run_id,
        scope="per_block",
        block_count=3,
        events=[
            _completed_controller_window(
                run_id=per_block_run_id,
                scope="per_block",
                window_index=0,
                start_step=0,
                end_step=2,
                action={"block_granularities": ["full", "medium", "micro"]},
            )
        ],
    )

    paths = generate_figures(
        tmp_path,
        tmp_path / "figures",
        refresh_counts=False,
        dpi=40,
    )
    names = {path.name for path in paths}
    assert not any(name.startswith("selected_granularity_over_tokens_") for name in names)
    assert not any(name.startswith("selected_granularity_share_over_tokens_") for name in names)
    assert not any(name.startswith("selected_granularity_zoom_") for name in names)
    assert len(
        [name for name in names if name.startswith("global_sampling_exposure_comparison__")]
    ) == 1
    assert len(
        [name for name in names if name.startswith("global_sampling_policy_summary__")]
    ) == 1

    frequency_paths = {
        path.name: path
        for path in paths
        if path.name.startswith("granularity_selection_frequency_over_tokens_")
    }
    assert set(frequency_paths) == {
        "granularity_selection_frequency_over_tokens_per-block-timeline.png",
    }
    for path in frequency_paths.values():
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    from src.evaluation.reporting_impl import (
        generate_figures as generate_compatibility_figures,
    )

    compatibility_paths = generate_compatibility_figures(
        tmp_path,
        tmp_path / "compatibility-figures",
        refresh_counts=False,
        dpi=40,
    )
    assert {path.name for path in compatibility_paths} == names


def test_generate_figures_filters_controller_plots_by_variant_and_correction(
    tmp_path,
):
    for run_id, model_variant, correction_mode in (
        ("slicing-none", "matformer_llama", "none"),
        ("slicing-gmc", "matformer_llama", "gmc"),
        ("concat-none", "cat_llama", "none"),
    ):
        _write_controller_timeline_run(
            tmp_path,
            run_id=run_id,
            scope="global",
            model_variant=model_variant,
            correction_mode=correction_mode,
            events=[
                _completed_controller_window(
                    run_id=run_id,
                    scope="global",
                    window_index=0,
                    start_step=0,
                    end_step=2,
                    action={"global_granularity": "micro"},
                )
            ],
        )

    paths = generate_figures(
        tmp_path,
        tmp_path / "filtered-figures",
        refresh_counts=False,
        dpi=40,
        variants=["slicing"],
        corrections=["none"],
    )
    names = {path.name for path in paths}
    assert not any(name.startswith("selected_granularity_") for name in names)
    assert len(
        [name for name in names if name.startswith("global_sampling_exposure_comparison__")]
    ) == 1


def _validation_trace_row(
    *,
    run_id,
    contract,
    granularity,
    tokens,
    loss,
    seed=42,
    status="completed",
):
    return {
        "run_id": run_id,
        "run_seed": seed,
        "split": "validation",
        "sampling_mode": "nested-random",
        "resolved_run_mode": "nested-random",
        "resolved_sampling_mode": "global",
        "model_family": "nested",
        "model_variant": "concat",
        "correction_mode": "gmc",
        "granularity": granularity,
        "tokens_seen": tokens,
        "loss": loss,
        "_validation_contract": contract,
        "_run_status": status,
        "_run_progress_tokens": tokens,
        "_run_token_budget": 49,
        "_ordered_granularities": ["micro", "medium", "full"],
    }


def test_validation_gmc_fixture_keeps_three_contracts_at_49_checkpoints():
    import matplotlib.pyplot as plt

    from src.evaluation.reporting_impl import (
        plot_loss_over_tokens_for_experiment,
    )

    rows = []
    for contract_index in range(3):
        rows.extend(
            _validation_trace_row(
                run_id=f"gmc-{contract_index}",
                contract=f"contract-{contract_index}",
                granularity="micro",
                tokens=checkpoint,
                loss=3.0 - contract_index * 0.1 - checkpoint * 0.001,
            )
            for checkpoint in range(1, 50)
        )

    output = Path("/tmp/nicolas.avila/validation-gmc-contracts.png")
    plot_loss_over_tokens_for_experiment(rows, "GMC", output, dpi=20)
    figure = plt.figure()
    plt.close(figure)

    # Inspect aggregation directly: no contract is allowed to become a
    # concatenated 147-point curve.
    from src.evaluation.reporting_impl import (
        aggregate_validation_curve,
        group_completed_validation_contracts,
    )

    grouped = group_completed_validation_contracts(rows)
    assert len(grouped) == 3
    assert [
        len(aggregate_validation_curve(group)["xs"]) for group in grouped.values()
    ] == [
        49,
        49,
        49,
    ]


def test_validation_seed_aggregation_uses_mean_and_min_max_band():
    from src.evaluation.reporting_impl import aggregate_validation_curve

    rows = [
        _validation_trace_row(
            run_id=f"seed-{seed}",
            contract="same-contract",
            granularity="micro",
            tokens=tokens,
            loss=loss,
            seed=seed,
        )
        for seed, values in ((42, (2.0, 1.0)), (43, (4.0, 3.0)))
        for tokens, loss in zip((10, 20), values)
    ]

    aggregate = aggregate_validation_curve(rows)

    assert aggregate["xs"] == [10.0, 20.0]
    assert aggregate["means"] == pytest.approx([3.0, 2.0])
    assert aggregate["minimums"] == pytest.approx([2.0, 1.0])
    assert aggregate["maximums"] == pytest.approx([4.0, 3.0])
    assert aggregate["band_mask"] == [True, True]


def test_incomplete_validation_rows_are_opt_in_and_dashed(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    from src.evaluation.reporting_impl import (
        filter_validation_rows_by_completion,
        plot_loss_over_tokens_for_experiment,
    )

    complete = _validation_trace_row(
        run_id="complete",
        contract="contract",
        granularity="micro",
        tokens=49,
        loss=1.0,
    )
    incomplete = _validation_trace_row(
        run_id="partial-run",
        contract="contract",
        granularity="micro",
        tokens=20,
        loss=2.0,
        status="running",
    )
    assert filter_validation_rows_by_completion([complete, incomplete]) == [complete]
    assert filter_validation_rows_by_completion(
        [complete, incomplete], include_incomplete=True
    ) == [complete, incomplete]

    captured = []
    monkeypatch.setattr(
        Figure,
        "savefig",
        lambda figure, *_args, **_kwargs: captured.append(figure),
    )
    monkeypatch.setattr("src.evaluation.reporting_impl.plt.close", lambda _figure: None)
    plot_loss_over_tokens_for_experiment(
        [complete, incomplete],
        "trace",
        tmp_path / "trace.png",
    )
    lines = captured[0].axes[0].lines
    partial = next(line for line in lines if "partial-run" in line.get_label())
    assert partial.get_linestyle() == "--"
    assert "20/49 tokens" in partial.get_label()


def test_generate_figures_excludes_missing_summary_until_opted_in(tmp_path):
    run_dir = tmp_path / "missing-summary"
    write_metrics_csv(
        run_dir,
        {
            "run_id": "missing-summary",
            "step": 1,
            "split": "validation",
            "model_family": "nested",
            "model_size_label": "fixture",
            "sampling_mode": "nested-random",
            "resolved_sampling_mode": "global",
            "granularity": "micro",
            "loss": 1.25,
            "perplexity": 3.5,
            "tokens_seen": 10,
            "wall_clock_seconds": 1.0,
            "tokens_per_second": 10.0,
            "peak_memory_bytes": 0,
        },
    )

    default_paths = generate_figures(
        tmp_path, tmp_path / "default", refresh_counts=False, dpi=20
    )
    opted_in_paths = generate_figures(
        tmp_path,
        tmp_path / "opted-in",
        refresh_counts=False,
        dpi=20,
        include_incomplete_validation_traces=True,
    )

    assert not any(
        path.name.startswith("validation_loss_over_tokens_") for path in default_paths
    )
    assert any(
        path.name.startswith("validation_loss_over_tokens_") for path in opted_in_paths
    )


def test_historical_validation_contract_fallback_isolates_run_identity():
    from src.evaluation.reporting_io import seed_independent_validation_contract

    config = {"run": {"model_family": "nested"}, "model": {}, "training": {}}
    contract_a, fallback_a = seed_independent_validation_contract(
        config, run_id="historical-a"
    )
    contract_b, fallback_b = seed_independent_validation_contract(
        config, run_id="historical-b"
    )

    assert fallback_a is fallback_b is True
    assert contract_a != contract_b


def test_validation_contract_ignores_seed_but_separates_training_contract():
    from copy import deepcopy

    from src.evaluation.reporting_io import seed_independent_validation_contract

    base = {
        "run": {
            "run_id": "seed-42",
            "seed": 42,
            "model_family": "nested",
            "sampling_mode": "nested-random",
        },
        "model": {"variant": "cat_llama", "granularities": ["micro", "full"]},
        "training": {
            "token_budget": 100_000_000,
            "learning_rate": 0.001,
            "optimizer_name": "adamw",
        },
        "dataset": {"dataset_name": "fixture", "dataset_split": "train"},
    }
    other_seed = deepcopy(base)
    other_seed["run"].update(run_id="seed-43", seed=43)
    other_learning_rate = deepcopy(base)
    other_learning_rate["training"]["learning_rate"] = 0.002

    contract_42, fallback_42 = seed_independent_validation_contract(
        base, run_id="seed-42"
    )
    contract_43, fallback_43 = seed_independent_validation_contract(
        other_seed, run_id="seed-43"
    )
    different_contract, _ = seed_independent_validation_contract(
        other_learning_rate, run_id="different-lr"
    )

    assert fallback_42 is fallback_43 is False
    assert contract_42 == contract_43
    assert contract_42 != different_contract


def test_validation_panels_follow_configured_order_share_limits_and_label_axis(
    tmp_path, monkeypatch
):
    from matplotlib.figure import Figure

    from src.evaluation.reporting_impl import plot_loss_over_tokens_for_experiment

    rows = [
        _validation_trace_row(
            run_id="ordered",
            contract="contract",
            granularity=granularity,
            tokens=10,
            loss=loss,
        )
        for granularity, loss in (("full", 10.0), ("micro", 1.0), ("medium", 5.0))
    ]
    captured = []
    monkeypatch.setattr(
        Figure,
        "savefig",
        lambda figure, *_args, **_kwargs: captured.append(figure),
    )
    monkeypatch.setattr("src.evaluation.reporting_impl.plt.close", lambda _figure: None)

    plot_loss_over_tokens_for_experiment(rows, "ordered", tmp_path / "ordered.png")

    axes = captured[0].axes
    assert [axis.get_title() for axis in axes] == ["micro", "medium", "full"]
    assert len({axis.get_ylim() for axis in axes}) == 1
    assert axes[-1].get_xlabel() == "Total training tokens"
    assert [list(axis.lines[0].get_ydata()) for axis in axes] == [[1.0], [5.0], [10.0]]


def test_direct_selected_exposure_reconstruction_includes_warmup_and_deduplicates_resume():
    from src.evaluation.reporting_impl import reconstruct_direct_selected_exposure

    rows = [
        {"split": "train", "step": 1, "tokens_seen": 10, "granularity": "micro"},
        {"split": "train", "step": 2, "tokens_seen": 25, "granularity": "full"},
        {"split": "train", "step": 2, "tokens_seen": 25, "granularity": "full"},
        {"split": "train", "step": 3, "tokens_seen": 40, "granularity": "micro"},
    ]

    history = reconstruct_direct_selected_exposure(rows, ["micro", "full"])

    assert history[-1] == (40.0, {"micro": 25.0, "full": 15.0})
    with pytest.raises(ValueError, match="non-monotonic"):
        reconstruct_direct_selected_exposure(
            [
                rows[0],
                rows[-1],
                {"split": "train", "step": 2, "tokens_seen": 20, "granularity": "full"},
            ],
            ["micro", "full"],
        )


def test_selected_exposure_interpolation_uses_only_shared_boundaries():
    from src.evaluation.reporting_impl import (
        SelectedExposureObservation,
        interpolate_loss_over_shared_exposure,
    )

    aggregate = interpolate_loss_over_shared_exposure(
        {
            "42": [
                SelectedExposureObservation(1, 0, 10),
                SelectedExposureObservation(2, 10, 0),
            ],
            "43": [
                SelectedExposureObservation(1, 5, 10),
                SelectedExposureObservation(2, 15, 0),
            ],
        }
    )

    assert len(aggregate["xs"]) == 100
    assert aggregate["xs"][0] == 5
    assert aggregate["xs"][-1] == 10
    assert aggregate["shared_min"] == 5
    assert aggregate["shared_max"] == 10


def test_marginal_utility_uses_latest_five_distinct_exposures_without_clipping():
    from src.evaluation.reporting_impl import (
        SelectedExposureObservation,
        marginal_utility_observations,
    )

    observations = [
        SelectedExposureObservation(index, index * 1_000_000, loss)
        for index, loss in enumerate((100.0, 9.0, 7.0, 5.0, 3.0, 1.0))
    ]
    scores = marginal_utility_observations(observations)

    assert scores[-1] == pytest.approx((5.0, 2.0))
    degrading = [
        SelectedExposureObservation(index, index * 1_000_000, float(index))
        for index in range(5)
    ]
    assert marginal_utility_observations(degrading)[-1][1] == pytest.approx(-1.0)


def test_marginal_utility_figure_explains_score_and_shared_parameter_caveat(
    tmp_path,
    monkeypatch,
):
    from matplotlib.figure import Figure

    from src.evaluation.reporting_impl import (
        SelectedExposureObservation,
        SelectedExposureRun,
        _plot_marginal_utility,
    )

    observations = tuple(
        SelectedExposureObservation(
            total_tokens=index * 1_000_000,
            selected_exposure_tokens=index * 1_000_000,
            loss=10.0 - index,
        )
        for index in range(1, 7)
    )
    contract_row = {
        "sampling_mode": "nested-random",
        "model_family": "nested",
        "model_variant": "cat_llama",
        "resolved_sampling_mode": "global",
        "correction_mode": "none",
        "_scheduler_warmup_tokens": 3_000_000,
    }
    run = SelectedExposureRun(
        run_id="uniform-run",
        run_seed="42",
        method="nested-random / concat / global",
        contract="uniform-contract",
        contract_row=contract_row,
        ordered_granularities=("micro",),
        observations={"micro": observations},
    )
    saved_figures = []
    monkeypatch.setattr(
        Figure,
        "savefig",
        lambda figure, _output_path, **_kwargs: saved_figures.append(figure),
    )

    _plot_marginal_utility([run], tmp_path / "utility.png", dpi=30)

    [figure] = saved_figures
    assert figure.axes[0].get_ylabel() == ""
    assert figure._supylabel.get_text() == (
        "Estimated validation-loss reduction\n"
        "per 1M directly selected tokens"
    )
    assert any(
        line.get_label() == "LR warmup end"
        and tuple(line.get_xdata()) == (3_000_000, 3_000_000)
        for line in figure.axes[0].get_lines()
    )
    assert "Five-point OLS over direct exposure" in figure._suptitle.get_text()
    assert any(
        "shared parameters can improve a width" in text.get_text()
        for text in figure.texts
    )


def test_completed_global_run_generates_saturation_figures_and_ranking(tmp_path):
    from src.evaluation.reporting_impl import generate_saturation_diagnostics

    run_dir = tmp_path / "global-run"
    run_dir.mkdir()
    config = {
        "run": {
            "run_id": "global-run",
            "seed": 42,
            "model_family": "nested",
            "sampling_mode": "nested-random",
            "resolved_run_mode": "nested-random",
        },
        "model": {
            "variant": "cat_llama",
            "granularities": ["micro", "full"],
            "granularity_sampling_mode": "global",
            "resolved_sampling_mode": "global",
        },
        "training": {
            "token_budget": 6_000_000,
            "learning_rate": 0.001,
            "optimizer_name": "adamw",
            "max_steps": 6,
        },
        "dataset": {"dataset_name": "fixture", "dataset_split": "train"},
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "tokens_seen": 6_000_000,
                "token_budget": 6_000_000,
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for step in range(1, 7):
        total_tokens = step * 1_000_000
        action = "micro" if step <= 3 else "full"
        rows.append(
            {
                "run_id": "global-run",
                "step": step,
                "split": "train",
                "model_family": "nested",
                "sampling_mode": "nested-random",
                "resolved_sampling_mode": "global",
                "granularity": action,
                "tokens_seen": total_tokens,
                "loss": 3.0,
                "perplexity": 20.0,
                "wall_clock_seconds": float(step),
                "tokens_per_second": 1_000_000.0,
                "peak_memory_bytes": 0,
            }
        )
        for granularity, offset in (("micro", 0.0), ("full", 0.2)):
            rows.append(
                {
                    "run_id": "global-run",
                    "step": step,
                    "split": "validation",
                    "model_family": "nested",
                    "sampling_mode": "nested-random",
                    "resolved_sampling_mode": "global",
                    "granularity": granularity,
                    "tokens_seen": total_tokens,
                    "loss": 10.0 - step + offset,
                    "perplexity": 20.0,
                    "wall_clock_seconds": float(step),
                    "tokens_per_second": 1_000_000.0,
                    "peak_memory_bytes": 0,
                }
            )
    write_metrics_csv(run_dir, rows)

    paths = generate_saturation_diagnostics(tmp_path, tmp_path / "figures", dpi=30)
    names = {path.name for path in paths}

    assert (
        "validation_loss_over_selected_exposure_nested_random_concat_global.png"
        in names
    )
    assert (
        "validation_marginal_utility_over_tokens_nested_random_concat_global.png"
        in names
    )
    assert "validation_marginal_utility_ranking.md" in names
    ranking = (
        tmp_path / "figures" / "validation_marginal_utility_ranking.md"
    ).read_text(encoding="utf-8")
    assert "evidence about saturation" in ranking
    assert "MatFormer parameter sharing" in ranking
    assert "| micro |" in ranking

    filtered_paths = generate_saturation_diagnostics(
        tmp_path,
        tmp_path / "filtered-figures",
        dpi=30,
        variants=["slicing"],
        corrections=["none"],
    )
    assert filtered_paths == []


def test_panelgrad_has_explicit_reporting_identity_and_saved_provenance():
    config = {
        "model": {
            "adaptive_sampler_strategy": "panelgrad",
            "panelgrad": {
                "method_family": "panelgrad_gradient_rms",
                "method_version": 1,
                "scope": "global",
            },
        }
    }
    row = {
        "sampling_mode": "nested-random",
        "resolved_sampling_mode": "adaptive_global",
        "adaptive_sampler_strategy": "panelgrad",
        "controller_method_family": "panelgrad_gradient_rms",
        "controller_method_version": 1,
        "controller_scope": "global",
    }

    assert scaling_curve_sampling_label(row) == "panelgrad_global"
    assert display_sampling_label_for_curve("panelgrad_global") == "PanelGrad global"
    assert controller_method_family_from_saved_config(config) == (
        "panelgrad_gradient_rms"
    )
    assert controller_method_version_from_saved_config(config) == 1
    assert controller_scope_from_saved_config(config) == "global"


def test_panelgrad_size_curves_are_split_and_labeled_by_policy_configuration():
    from src.evaluation.reporting_impl import (
        compact_size_curve_labels,
        group_size_plot_rows,
    )
    from src.evaluation.reporting_io import (
        controller_contract_provenance_from_saved_config,
    )

    rows = []
    for temperature in (1.0, 2.0, 4.0):
        config = {
            "model": {
                "adaptive_sampler_strategy": "panelgrad",
                "panelgrad": {
                    "method_family": "panelgrad_gradient_rms",
                    "method_version": 1,
                    "scope": "global",
                    "refresh_interval_steps": 50,
                    "eta": 1e-12,
                    "temperature": temperature,
                    "epsilon": 0.1,
                },
            }
        }
        row = {
            "run_id": f"panelgrad-t{temperature:g}",
            "sampling_mode": "nested-random",
            "model_variant": "slicing",
            "resolved_sampling_mode": "adaptive_global",
            "adaptive_sampler_strategy": "panelgrad",
            "controller_method_family": "panelgrad_gradient_rms",
            "controller_method_version": 1,
            "controller_scope": "global",
            "granularity": "micro",
            "non_embedding_parameters": 100,
            "perplexity": 2.0,
        }
        row.update(controller_contract_provenance_from_saved_config(config))
        rows.append(row)

    grouped = group_size_plot_rows(rows)
    labels = compact_size_curve_labels(
        grouped,
        panel_sampling_label="panelgrad_global",
    )

    assert len(grouped) == 3
    assert set(labels.values()) == {"T=1", "T=2", "T=4"}


def test_panelgrad_schedule_identity_prevents_fixed_and_linear_curve_merging():
    from src.evaluation.reporting_impl import group_size_plot_rows
    from src.evaluation.reporting_io import (
        controller_contract_provenance_from_saved_config,
    )

    policies = [
        {"epsilon": 0.5},
        {
            "epsilon_schedule": {
                "type": "linear",
                "start": 0.5,
                "end": 0.1,
                "duration_steps": 24415,
            }
        },
    ]
    rows = []
    for index, epsilon_policy in enumerate(policies):
        panelgrad = {
            "method_family": "panelgrad_gradient_rms",
            "method_version": 1,
            "scope": "global",
            "refresh_interval_steps": 50,
            "eta": 1e-12,
            "temperature": 1.0,
            **epsilon_policy,
        }
        config = {
            "model": {
                "adaptive_sampler_strategy": "panelgrad",
                "panelgrad": panelgrad,
            }
        }
        row = {
            "run_id": f"panelgrad-schedule-{index}",
            "sampling_mode": "nested-random",
            "model_variant": "slicing",
            "resolved_sampling_mode": "adaptive_global",
            "adaptive_sampler_strategy": "panelgrad",
            "controller_method_family": "panelgrad_gradient_rms",
            "controller_method_version": 1,
            "controller_scope": "global",
            "granularity": "micro",
            "non_embedding_parameters": 100,
            "perplexity": 2.0,
        }
        row.update(controller_contract_provenance_from_saved_config(config))
        rows.append(row)

    grouped = group_size_plot_rows(rows)

    assert len(grouped) == 2
    identities = {
        (
            row[0]["panelgrad_epsilon_schedule_type"],
            row[0]["panelgrad_epsilon_schedule_start"],
            row[0]["panelgrad_epsilon_schedule_end"],
            row[0]["panelgrad_epsilon_schedule_duration_steps"],
        )
        for row in grouped.values()
    }
    assert identities == {
        ("fixed", 0.5, 0.5, None),
        ("linear", 0.5, 0.1, 24415),
    }


def test_panelgrad_history_extracts_committed_actions_and_refresh_diagnostics(
    tmp_path,
):
    from src.evaluation.reporting_io import (
        iter_controller_granularity_timelines,
        iter_panelgrad_histories,
        panelgrad_history_as_timeline,
    )

    _write_panelgrad_history_run(tmp_path)

    assert list(iter_controller_granularity_timelines(tmp_path)) == []
    [history] = list(iter_panelgrad_histories(tmp_path))

    assert history.run_id == "panelgrad-run"
    assert history.importance_metric == "gradient_rms"
    assert history.ordered_granularities == ("micro", "medium", "full")
    assert [
        (
            action.step,
            action.start_tokens,
            action.end_tokens,
            action.granularity,
            action.sampled_probability,
        )
        for action in history.actions
    ] == [
        (2, 10, 20, "micro", 0.5),
        (3, 20, 30, "full", 0.2),
    ]
    assert [refresh.boundary_tokens for refresh in history.refreshes] == [10, 30]
    assert history.refreshes[0].gradient_rms_scores == (0.3, 0.2, 0.1)
    assert history.refreshes[0].p == (0.5, 0.3, 0.2)
    assert history.refreshes[0].q == (0.52, 0.30, 0.18)
    assert history.refreshes[0].controller_target_tokens == 90
    assert [refresh.active_epsilon for refresh in history.refreshes] == pytest.approx(
        [0.5, 0.3]
    )
    assert [refresh.epsilon_schedule_step for refresh in history.refreshes] == [0, 2]

    timeline = panelgrad_history_as_timeline(history)
    assert timeline.scope == "global"
    assert [window.block_granularities for window in timeline.windows] == [
        ("micro",),
        ("full",),
    ]
    assert [(window.start_tokens, window.end_tokens) for window in timeline.windows] == [
        (10, 20),
        (20, 30),
    ]


def test_panelgrad_rms_and_l2_have_distinct_reporting_and_plot_identities(tmp_path):
    from src.evaluation.reporting import generate_figures
    from src.evaluation.reporting_io import (
        iter_global_sampling_histories,
        iter_panelgrad_histories,
    )

    _write_panelgrad_history_run(
        tmp_path, run_id="same-policy-rms", importance_metric="gradient_rms"
    )
    _write_panelgrad_history_run(
        tmp_path, run_id="same-policy-l2", importance_metric="gradient_l2"
    )

    histories = list(iter_panelgrad_histories(tmp_path))
    assert {history.importance_metric for history in histories} == {
        "gradient_rms",
        "gradient_l2",
    }
    global_histories = {
        history.policy_identity: history
        for history in iter_global_sampling_histories(tmp_path)
    }
    assert set(global_histories) == {
        "panelgrad_global_gradient_rms",
        "panelgrad_global_gradient_l2",
    }
    assert "metric=Gradient RMS" in global_histories[
        "panelgrad_global_gradient_rms"
    ].policy_label
    assert "metric=Gradient L2 norm" in global_histories[
        "panelgrad_global_gradient_l2"
    ].policy_label

    names = {
        path.name
        for path in generate_figures(tmp_path, tmp_path / "figures", dpi=30)
    }
    assert "panelgrad_refresh_diagnostics_same-policy-rms_gradient_rms.png" in names
    assert "panelgrad_refresh_diagnostics_same-policy-l2_gradient_l2.png" in names


def test_panelgrad_refresh_diagnostics_plot_epsilon_over_tokens(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    from src.evaluation.reporting import plot_panelgrad_refresh_diagnostics
    from src.evaluation.reporting_io import iter_panelgrad_histories

    _write_panelgrad_history_run(tmp_path)
    [history] = list(iter_panelgrad_histories(tmp_path))
    saved_figures = []
    monkeypatch.setattr(
        Figure,
        "savefig",
        lambda figure, _output_path, **_kwargs: saved_figures.append(figure),
    )

    plot_panelgrad_refresh_diagnostics(
        history,
        tmp_path / "panelgrad-refresh-diagnostics.png",
        dpi=40,
    )

    [figure] = saved_figures
    epsilon_axis = figure.axes[2]
    assert figure.axes[0].get_yscale() == "log"
    assert figure.axes[0].get_ylabel() == (
        "Gradient RMS (L2 / sqrt(N))\n(log scale)"
    )
    assert figure.axes[0].get_title() == "Controlled-FFN gradient importance"
    assert figure.axes[1].get_title() == (
        "Score distribution q (dashed) and epsilon-mixed sampling "
        "distribution p (solid)"
    )
    assert [line.get_label() for line in figure.axes[1].lines[:2]] == [
        "sampling p(micro)",
        "score q(micro)",
    ]
    assert epsilon_axis.get_title() == "Refresh-boundary exploration schedule"
    assert list(epsilon_axis.lines[0].get_xdata()) == [10, 30]
    assert list(epsilon_axis.lines[0].get_ydata()) == pytest.approx([0.5, 0.3])
    assert figure.axes[4].get_title().startswith(
        "PanelGrad measurement overhead"
    )
    assert figure.axes[4].get_ylabel() == "Refresh duration (s)"
    assert figure.axes[5].get_ylabel() == "Cumulative refresh duration (s)"


def test_generate_figures_writes_panelgrad_action_exposure_and_refresh_plots(tmp_path):
    _write_panelgrad_history_run(tmp_path)

    paths = generate_figures(tmp_path, tmp_path / "figures", dpi=30)
    names = {path.name for path in paths}

    assert "selected_granularity_over_tokens_panelgrad-run.png" not in names
    assert "selected_granularity_share_over_tokens_panelgrad-run.png" not in names
    assert "selected_granularity_zoom_panelgrad-run.png" not in names
    assert "granularity_selection_frequency_over_tokens_panelgrad-run.png" not in names
    assert "panelgrad_cumulative_exposure_panelgrad-run.png" not in names
    assert "panelgrad_refresh_diagnostics_panelgrad-run_gradient_rms.png" in names
    assert len(
        [name for name in names if name.startswith("global_sampling_exposure_comparison__")]
    ) == 1
    assert len(
        [name for name in names if name.startswith("global_sampling_policy_summary__")]
    ) == 1
    for name in names:
        assert (tmp_path / "figures" / name).is_file()


def test_global_sampling_histories_use_committed_metrics_actions_for_all_policies(
    tmp_path,
):
    from src.evaluation.reporting_io import iter_global_sampling_histories

    actions = ["micro", "micro", "full", "full"]
    for policy in ("uniform", "fixed", "thompson", "panelgrad"):
        _write_global_policy_run(
            tmp_path,
            run_id=f"{policy}-run",
            policy=policy,
            actions=actions,
        )

    histories = {
        history.policy_identity: history
        for history in iter_global_sampling_histories(tmp_path)
    }
    assert {
        identity for identity in histories if not identity.startswith("fixed_global_")
    } == {
        "uniform_global",
        "thompson_global",
        "panelgrad_global_gradient_rms",
    }
    [fixed_identity] = [
        identity for identity in histories if identity.startswith("fixed_global_")
    ]
    assert histories[fixed_identity].policy_label == (
        "Fixed global (micro=10.0%, medium=30.0%, full=60.0%)"
    )
    assert histories[fixed_identity].target_probabilities == pytest.approx(
        (0.1, 0.3, 0.6)
    )
    assert [
        action.sampled_probability for action in histories[fixed_identity].actions
    ] == pytest.approx([0.1, 0.1, 0.6, 0.6])
    assert all(
        [action.granularity for action in history.actions] == actions
        for history in histories.values()
    )
    assert histories["uniform_global"].decision_count == 4
    assert histories[fixed_identity].decision_count == 4
    assert histories["panelgrad_global_gradient_rms"].decision_count == 4
    assert histories["thompson_global"].decision_count == 2
    assert len({history.comparison_key for history in histories.values()}) == 1


def test_uniform_global_window_reporting_identity_and_decision_count(tmp_path):
    from src.evaluation.reporting_io import iter_global_sampling_histories

    actions = ["micro"] * 12_207
    _write_global_policy_run(
        tmp_path,
        run_id="uniform-window25-run",
        policy="uniform_window",
        actions=actions,
        decision_interval_steps=25,
    )

    [history] = list(iter_global_sampling_histories(tmp_path))

    assert history.policy_identity == "uniform_global_h25"
    assert history.policy_label == "Uniform global (H=25)"
    assert history.decision_count == 489
    assert history.actions[24].decision_index == 0
    assert history.actions[25].decision_index == 1
    assert history.actions[-1].decision_index == 488


def test_balanced_global_window_reporting_identity_and_schedule_provenance(tmp_path):
    from src.evaluation.reporting import generate_global_sampling_policy_figures
    from src.evaluation.reporting_io import iter_global_sampling_histories

    actions = ["micro"] * 2 + ["medium"] * 2 + ["full"] * 2
    _write_global_policy_run(
        tmp_path,
        run_id="balanced-window2-run",
        policy="balanced_window",
        actions=actions,
        decision_interval_steps=2,
    )

    [history] = list(iter_global_sampling_histories(tmp_path))

    assert history.policy_identity == "balanced_global_h2"
    assert history.policy_label == "Balanced global (H=2)"
    assert history.global_sampling_schedule == "balanced_cycle"
    assert history.global_sampling_schedule_version == 1
    assert history.decision_count == 3
    paths = generate_global_sampling_policy_figures(
        tmp_path, tmp_path / "figures", dpi=20, sampling_bin_steps=2
    )
    names = {path.name for path in paths}
    assert any(
        name.startswith(
            "global_sampling_exposure_comparison__slicing__none__balanced_cycle_v1__"
        )
        for name in names
    )
    [summary_path] = [path for path in paths if path.suffix == ".csv"]
    with summary_path.open(encoding="utf-8", newline="") as csv_file:
        [row] = list(csv.DictReader(csv_file))
    assert row["policy"] == "Balanced global (H=2)"
    assert row["global_sampling_schedule"] == "balanced_cycle"
    assert row["global_sampling_schedule_version"] == "1"


def test_global_sampling_bins_preserve_step_resolution_and_partial_final_bin(tmp_path):
    from src.evaluation.reporting import global_sampling_bin_series
    from src.evaluation.reporting_io import iter_global_sampling_histories

    actions = ["micro"] * 25 + ["full"] * 25 + ["medium"] * 15
    _write_global_policy_run(
        tmp_path,
        run_id="panelgrad-bins",
        policy="panelgrad",
        actions=actions,
    )
    [history] = list(iter_global_sampling_histories(tmp_path))

    _x_values, shares, bin_sizes = global_sampling_bin_series(history, bin_steps=50)

    assert bin_sizes == [50, 15]
    assert shares["micro"] == pytest.approx([0.5, 0.0])
    assert shares["medium"] == pytest.approx([0.0, 1.0])
    assert shares["full"] == pytest.approx([0.5, 0.0])
    assert [sum(values) for values in zip(*shares.values())] == pytest.approx([1.0, 1.0])


def test_grouped_sampling_exposure_uses_stacked_actual_token_spans(
    tmp_path,
    monkeypatch,
):
    from matplotlib.figure import Figure

    from src.evaluation.reporting import (
        plot_global_sampling_cumulative_comparison,
        plot_global_sampling_exposure_comparison,
    )
    from src.evaluation.reporting_io import GlobalSamplingAction, GlobalSamplingHistory

    history = GlobalSamplingHistory(
        run_id="in-progress-seed-7",
        policy_identity="uniform_global",
        policy_label="Uniform global",
        ordered_granularities=("micro", "medium", "full"),
        token_budget=100,
        actions=(
            GlobalSamplingAction(1, 0, 5, "micro", 0, None),
            GlobalSamplingAction(2, 5, 20, "full", 1, None),
            GlobalSamplingAction(3, 20, 35, "medium", 2, None),
        ),
        decision_count=3,
        comparison_key="comparison",
        model_variant="slicing",
        correction_mode="none",
        membership_correction=False,
        seed=7,
        config_path=tmp_path / "config.json",
        metrics_path=tmp_path / "metrics.csv",
    )
    saved_figures = []
    monkeypatch.setattr(
        Figure,
        "savefig",
        lambda figure, _path, **_kwargs: saved_figures.append(figure),
    )

    plot_global_sampling_exposure_comparison(
        [history], tmp_path / "exposure.png", bin_steps=2, dpi=20
    )
    [exposure_figure] = saved_figures
    [axis] = exposure_figure.axes
    assert axis.get_xlim() == pytest.approx((0.0, 35.0))
    assert axis.get_ylim() == pytest.approx((0.0, 100.0))
    assert "seed 7" in axis.get_title(loc="left")
    assert "in-progress-seed-7" not in axis.get_title(loc="left")
    assert exposure_figure._suptitle.get_text() == (
        "Empirical global selection distribution\n2-optimizer-step bins"
    )
    assert [patch.get_width() for patch in axis.containers[0].patches] == pytest.approx(
        [20.0, 15.0]
    )
    for bin_index in range(2):
        assert sum(
            container.patches[bin_index].get_height()
            for container in axis.containers
        ) == pytest.approx(100.0)

    saved_figures.clear()
    plot_global_sampling_cumulative_comparison(
        [history], tmp_path / "cumulative.png", bin_steps=2, dpi=20
    )
    [cumulative_figure] = saved_figures
    assert cumulative_figure.axes[-1].get_xlim() == pytest.approx((0.0, 35.0))
    assert all(
        axis.get_ylabel() == "Cumulative\nselected-step share"
        for axis in cumulative_figure.axes
    )
    assert not any(axis.get_legend() for axis in cumulative_figure.axes)
    assert len(cumulative_figure.legends) == 1


def test_generate_figures_writes_unified_global_policy_comparisons(tmp_path):
    action_sets = {
        "uniform": ["micro", "medium", "full", "micro"],
        "fixed": ["full", "full", "medium", "full"],
        "thompson": ["micro", "micro", "full", "full"],
        "panelgrad": ["full", "medium", "micro", "full"],
    }
    for policy, actions in action_sets.items():
        _write_global_policy_run(
            tmp_path,
            run_id=f"{policy}-comparison",
            policy=policy,
            actions=actions,
        )

    paths = generate_figures(
        tmp_path,
        tmp_path / "figures",
        dpi=30,
        sampling_bin_steps=2,
        sampling_zoom_steps=2,
    )
    names = {path.name for path in paths}

    assert not any(name.startswith("selected_granularity_over_tokens_") for name in names)
    assert not any(name.startswith("selected_granularity_share_over_tokens_") for name in names)
    assert len([name for name in names if name.startswith("selected_granularity_zoom_")]) == 4
    assert len([name for name in names if name.startswith("global_sampling_exposure_comparison__")]) == 1
    assert len([name for name in names if name.startswith("global_sampling_cumulative_comparison__")]) == 1
    [summary_name] = [
        name for name in names if name.startswith("global_sampling_policy_summary__")
    ]
    with (tmp_path / "figures" / summary_name).open(
        "r", encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))
    assert {row["policy"] for row in rows} == {
        "Uniform global",
        "Fixed global (micro=10.0%, medium=30.0%, full=60.0%)",
        "Thompson global",
        "PanelGrad (metric=Gradient RMS)",
    }
    assert {row["completed_steps"] for row in rows} == {"4"}
    assert {
        row["policy"]: row["policy_decisions"] for row in rows
    } == {
        "PanelGrad (metric=Gradient RMS)": "4",
        "Fixed global (micro=10.0%, medium=30.0%, full=60.0%)": "4",
        "Thompson global": "2",
        "Uniform global": "4",
    }
    fixed_row = next(row for row in rows if row["policy"].startswith("Fixed global"))
    assert [
        fixed_row[f"{label}_target_probability"]
        for label in ("micro", "medium", "full")
    ] == ["0.1", "0.3", "0.6"]


def test_inverse_membership_fixed_global_uses_compact_policy_label():
    from src.evaluation.reporting_io import _fixed_global_policy_label

    labels = ("g125", "g250", "g375", "g500", "g625", "g750", "g875", "g1000")
    weights = [1.0 / membership for membership in range(8, 0, -1)]
    normalizer = sum(weights)
    distribution = {
        label: weight / normalizer for label, weight in zip(labels, weights)
    }

    assert (
        _fixed_global_policy_label(distribution, labels)
        == "Fixed global (inverse-membership)"
    )


def test_validation_comparison_labels_are_concise_and_deduplicate_standalone():
    from src.evaluation.reporting_impl import (
        concise_validation_comparison_contract_labels,
    )

    standalone = {
        "sampling_mode": "standalone",
        "resolved_run_mode": "standalone",
        "model_family": "standalone",
        "model_variant": "slicing",
    }
    slicing_uniform = {
        "sampling_mode": "nested-random",
        "model_family": "nested",
        "model_variant": "matformer_llama",
        "resolved_sampling_mode": "global",
    }
    labels = concise_validation_comparison_contract_labels(
        {
            "standalone-micro": [{**standalone, "granularity": "micro"}],
            "standalone-full": [{**standalone, "granularity": "full"}],
            "slicing-uniform": [slicing_uniform],
        }
    )

    assert labels == {
        "standalone-micro": "Standalone",
        "standalone-full": "Standalone",
        "slicing-uniform": "Slicing · Uniform",
    }


def test_non_panelgrad_reporting_labels_remain_unchanged():
    assert scaling_curve_sampling_label(
        {
            "sampling_mode": "nested-random",
            "resolved_sampling_mode": "adaptive_per_block",
            "adaptive_sampler_strategy": "ucb",
        }
    ) == "adaptive_per_block_ucb"
    assert display_sampling_label_for_curve("global") is None
    assert scaling_curve_sampling_label(
        {
            "sampling_mode": "nested-random",
            "resolved_sampling_mode": "fixed_global",
        }
    ) == "fixed_global"
    assert (
        display_sampling_label_for_curve("fixed_global")
        == "fixed non-uniform global"
    )


def test_uniform_global_window_scaling_identity_is_distinct():
    row = {
        "sampling_mode": "nested-random",
        "resolved_sampling_mode": "global",
        "global_sampling_interval_steps": 25,
    }

    assert scaling_curve_sampling_label(row) == "uniform_global_h25"
    assert (
        display_sampling_label_for_curve("uniform_global_h25")
        == "Uniform global (H=25)"
    )


def test_uniform_global_window_interval_is_enriched_from_config(tmp_path):
    from src.evaluation.reporting_io import enrich_scaling_metadata_from_run_config

    run_dir = tmp_path / "uniform-window"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "run": {"sampling_mode": "nested-random"},
                "model": {
                    "granularity_sampling_mode": "global",
                    "global_sampling_interval_steps": 25,
                },
            }
        ),
        encoding="utf-8",
    )

    [row] = enrich_scaling_metadata_from_run_config(
        tmp_path,
        [
            {
                "run_id": "uniform-window",
                "sampling_mode": "nested-random",
                "_source_csv": str(run_dir / "scaling_results.csv"),
            }
        ],
    )

    assert row["global_sampling_interval_steps"] == 25
    assert scaling_curve_sampling_label(row) == "uniform_global_h25"


def test_size_reporting_does_not_merge_distinct_fixed_global_distributions():
    from src.evaluation.reporting_impl import (
        compact_size_curve_labels,
        group_size_plot_rows,
    )

    base = {
        "sampling_mode": "nested-random",
        "resolved_sampling_mode": "fixed_global",
        "model_variant": "slicing",
        "granularity": "s",
        "non_embedding_parameters": "100",
        "perplexity": "10",
    }
    rows = [
        {
            **base,
            "run_id": "fixed-a",
            "global_sampling_distribution": {
                "s": 0.1,
                "m": 0.2,
                "l": 0.3,
                "xl": 0.4,
            },
        },
        {
            **base,
            "run_id": "fixed-b",
            "global_sampling_distribution": {
                "s": 0.4,
                "m": 0.3,
                "l": 0.2,
                "xl": 0.1,
            },
        },
    ]

    grouped = group_size_plot_rows(rows)
    labels = compact_size_curve_labels(grouped)

    assert len(grouped) == 2
    assert len(set(labels.values())) == 2
    assert all("Sampling distribution=" in label for label in labels.values())


def test_size_reporting_treats_blank_and_absent_contract_fields_as_missing():
    from src.evaluation.reporting_impl import compact_size_curve_labels

    base = {
        "sampling_mode": "standalone",
        "resolved_run_mode": "standalone",
        "model_variant": "slicing",
        "granularity": "s",
        "non_embedding_parameters": "100",
        "perplexity": "10",
    }
    grouped = {
        "contract-with-absent-field": [{**base, "run_id": "standalone-a"}],
        "contract-with-blank-field": [
            {
                **base,
                "run_id": "standalone-b",
                "global_sampling_distribution": "",
            }
        ],
    }

    labels = compact_size_curve_labels(grouped)

    assert len(labels) == 2
    assert all("Sampling distribution=" not in label for label in labels.values())
