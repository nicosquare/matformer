import csv
import json
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

from utils.config import resolve_run_config
from utils.metrics import (
    ArtifactError,
    SCALING_RESULTS_COLUMNS,
    build_run_summary,
    build_consistency_result_rows,
    build_scaling_result_rows,
    write_config_artifact,
    write_consistency_results_csv,
    write_failed_run_summary,
    write_metrics_csv,
    write_run_summary,
    write_scaling_results_csv,
    write_task_results_csv,
)
from training.run import run_training


class TinyExtractionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.current_granularity = None
        self.ffn_prefix_metadata = [
            {
                "name": "s",
                "display_name": "S",
                "ffn_ratio": 0.5,
                "full_intermediate_fraction": 0.125,
                "prefix_width": 8,
            },
            {
                "name": "m",
                "display_name": "M",
                "ffn_ratio": 1.0,
                "full_intermediate_fraction": 0.25,
                "prefix_width": 16,
            },
            {
                "name": "l",
                "display_name": "L",
                "ffn_ratio": 2.0,
                "full_intermediate_fraction": 0.5,
                "prefix_width": 32,
            },
            {
                "name": "xl",
                "display_name": "XL",
                "ffn_ratio": 4.0,
                "full_intermediate_fraction": 1.0,
                "prefix_width": 64,
            },
        ]

    def configure_subnetwork(self, granularity):
        self.current_granularity = granularity

    def forward(self, input_ids, attention_mask=None, labels=None):
        loss = self.weight.pow(2) + input_ids.float().mean() * 0.0
        return SimpleNamespace(loss=loss)


def test_write_config_metrics_and_run_summary(tmp_path):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
    )

    config_path = write_config_artifact(config)
    metrics_path = write_metrics_csv(
        output_dir,
        [
            {
                "run_id": "debug-nested-001",
                "step": 0,
                "split": "validation",
                "model_family": "nested",
                "model_size_label": "debug",
                "granularity": "s",
                "loss": 2.1,
                "perplexity": 8.17,
                "tokens_seen": 128,
                "wall_clock_seconds": 1.5,
                "tokens_per_second": 85.3,
                "peak_memory_bytes": 2048,
            },
            {
                "run_id": "debug-nested-001",
                "step": 0,
                "split": "validation",
                "model_family": "nested",
                "model_size_label": "debug",
                "granularity": "xl",
                "loss": 1.7,
                "perplexity": 5.47,
                "tokens_seen": 128,
                "wall_clock_seconds": 1.5,
                "tokens_per_second": 85.3,
                "peak_memory_bytes": 2048,
            },
        ],
    )
    summary = build_run_summary(config, tokens_seen=128, notes=["smoke test"])
    summary_path = write_run_summary(output_dir, summary)

    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_config["run"]["run_id"] == "debug-nested-001"
    assert saved_config["model"]["variant"] == "matformer_llama"
    assert saved_config["training"]["base_learning_rate"] == 0.0003
    assert saved_config["training"]["learning_rate_scale_rule"] == "none"
    assert saved_config["training"]["learning_rate_scale_factor"] == 1.0
    assert saved_config["training"]["resolved_learning_rate"] == 0.0003
    assert saved_config["training"]["warmup_ratio"] == 0.0
    assert saved_config["training"]["warmup_steps"] == 0
    assert saved_config["training"]["resolved_warmup_steps"] == 0
    assert saved_config["training"]["gradient_clip_norm"] == 1.0
    assert saved_config["training"]["scheduler_name"] == "cosine"
    assert saved_config["training"]["scheduler"]["kwargs"]["warmup_steps"] == 0
    assert saved_config["training"]["scheduler"]["resolved_warmup_steps"] == 0
    assert saved_config["training"]["scheduler_kwargs"] == {}
    assert saved_config["training"]["optimizer_name"] == "adamw"
    assert saved_config["training"]["optimizer_kwargs"] == {
        "betas": [0.9, 0.95],
        "eps": 1e-08,
        "weight_decay": 0.1,
    }

    with metrics_path.open("r", encoding="utf-8", newline="") as metrics_file:
        metric_rows = list(csv.DictReader(metrics_file))
    assert [row["granularity"] for row in metric_rows] == ["s", "xl"]
    assert metric_rows[0]["peak_memory_bytes"] == "2048"

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["status"] == "completed"
    assert saved_summary["tokens_seen"] == 128
    assert saved_summary["model_variant"] == "matformer_llama"
    assert saved_summary["base_learning_rate"] == 0.0003
    assert saved_summary["learning_rate_scale_rule"] == "none"
    assert saved_summary["learning_rate_scale_factor"] == 1.0
    assert saved_summary["resolved_learning_rate"] == 0.0003
    assert saved_summary["warmup_ratio"] == 0.0
    assert saved_summary["warmup_steps"] == 0
    assert saved_summary["resolved_warmup_steps"] == 0
    assert saved_summary["gradient_clip_norm"] == 1.0
    assert saved_summary["scheduler_name"] == "cosine"
    assert saved_summary["scheduler_warmup_steps"] == 0
    assert saved_summary["scheduler_resolved_warmup_steps"] == 0
    assert saved_summary["scheduler_kwargs"] == {}
    assert saved_summary["optimizer_name"] == "adamw"
    assert saved_summary["optimizer_kwargs"] == {
        "betas": [0.9, 0.95],
        "eps": 1e-08,
        "weight_decay": 0.1,
    }
    assert saved_summary["notes"] == ["smoke test"]


def test_run_summary_includes_default_long_run_metadata(tmp_path):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
    )

    summary = build_run_summary(config, tokens_seen=0)
    summary_path = write_run_summary(output_dir, summary)

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["monitoring_enabled"] is False
    assert saved_summary["monitoring_backend"] == "wandb"
    assert saved_summary["monitoring_series_metadata"] == []
    assert saved_summary["latest_checkpoint_path"] is None
    assert saved_summary["continuation_state"] == {
        "run_id": "debug-nested-001",
        "output_dir": str(output_dir),
        "latest_checkpoint_path": None,
        "last_completed_step": 0,
        "tokens_seen": 0,
        "status": "fresh",
        "resume_count": 0,
    }
    assert saved_summary["warmup_policy"] == {
        "enabled": False,
        "duration": 0,
        "unit": "epochs",
        "completed": False,
        "completion_step": None,
        "transition_reason": None,
    }
    assert saved_summary["warmup_completion_step"] is None
    assert saved_summary["warmup_completed"] is False


@pytest.mark.parametrize(
    "continuation_overrides, expected_state",
    [
        (
            [],
            {
                "status": "fresh",
                "latest_checkpoint_path": None,
                "last_completed_step": 0,
                "resume_count": 0,
            },
        ),
        (
            [
                "run.continuation.enabled=true",
                "run.continuation.status=resumed",
                "run.continuation.latest_checkpoint_path=/tmp/debug-nested-001/checkpoints/latest.pt",
                "run.continuation.last_completed_step=8",
                "run.continuation.resume_count=1",
            ],
            {
                "status": "resumed",
                "latest_checkpoint_path": "/tmp/debug-nested-001/checkpoints/latest.pt",
                "last_completed_step": 8,
                "resume_count": 1,
            },
        ),
        (
            [
                "run.continuation.enabled=true",
                "run.continuation.status=completed",
                "run.continuation.latest_checkpoint_path=/tmp/debug-nested-001/checkpoints/final.pt",
                "run.continuation.last_completed_step=16",
                "run.continuation.resume_count=2",
            ],
            {
                "status": "completed",
                "latest_checkpoint_path": "/tmp/debug-nested-001/checkpoints/final.pt",
                "last_completed_step": 16,
                "resume_count": 2,
            },
        ),
    ],
)
def test_run_summary_records_continuation_state_transitions(
    tmp_path,
    continuation_overrides,
    expected_state,
):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=continuation_overrides,
    )

    summary = build_run_summary(config, tokens_seen=128)
    summary_path = write_run_summary(output_dir, summary)

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["continuation_state"]["run_id"] == "debug-nested-001"
    assert saved_summary["continuation_state"]["output_dir"] == str(output_dir)
    assert saved_summary["continuation_state"]["status"] == expected_state["status"]
    assert (
        saved_summary["continuation_state"]["latest_checkpoint_path"]
        == expected_state["latest_checkpoint_path"]
    )
    assert (
        saved_summary["continuation_state"]["last_completed_step"]
        == expected_state["last_completed_step"]
    )
    assert (
        saved_summary["continuation_state"]["resume_count"]
        == expected_state["resume_count"]
    )
    assert saved_summary["latest_checkpoint_path"] == expected_state[
        "latest_checkpoint_path"
    ]


def test_write_failed_run_summary_records_failure_note(tmp_path):
    output_dir = tmp_path / "debug-standalone-s-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-standalone-s-001",
        output_dir=output_dir,
    )

    summary_path = write_failed_run_summary(
        config,
        error_message="CUDA out of memory during debug smoke",
        tokens_seen=64,
    )

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["status"] == "failed"
    assert saved_summary["tokens_seen"] == 64
    assert saved_summary["model_variant"] == "matformer_llama"
    assert saved_summary["notes"] == ["CUDA out of memory during debug smoke"]


def test_baseline_and_cat_run_summaries_share_schema_and_differ_by_variant(tmp_path):
    baseline_output_dir = tmp_path / "baseline" / "debug-nested-001"
    cat_output_dir = tmp_path / "cat" / "debug-nested-001"

    baseline_config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=baseline_output_dir,
    )
    cat_config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=cat_output_dir,
        overrides=["model.variant=cat_llama"],
    )

    baseline_summary = build_run_summary(
        baseline_config,
        tokens_seen=128,
        notes=["baseline comparison smoke"],
    )
    cat_summary = build_run_summary(
        cat_config,
        tokens_seen=128,
        notes=["cat comparison smoke"],
    )

    assert set(baseline_summary) == set(cat_summary)
    assert baseline_summary["model_variant"] == "matformer_llama"
    assert cat_summary["model_variant"] == "cat_llama"
    assert baseline_summary["model_family"] == cat_summary["model_family"] == "nested"


def test_run_summary_includes_budget_derived_fields(tmp_path):
    output_dir = tmp_path / "dmodel256-pilot-comparison-001"
    config = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        output_dir=output_dir,
    )

    summary = build_run_summary(config, tokens_seen=128, notes=["budget smoke"])

    for field_name in [
        "expected_tokens_per_step",
        "derived_max_steps",
        "effective_world_size",
        "stop_reason",
        "model_family_slug",
        "model_size_slug",
        "token_budget_slug",
        "output_group",
    ]:
        assert field_name in summary
    assert summary["expected_tokens_per_step"] == config["training"][
        "expected_tokens_per_step"
    ]
    assert summary["derived_max_steps"] == config["training"]["derived_max_steps"]
    assert summary["effective_world_size"] == config["training"][
        "effective_world_size"
    ]
    assert summary["stop_reason"] == "not_started"


def test_run_summary_records_resolved_schedule_and_optimizer_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    output_dir = tmp_path / "dmodel256-pilot-comparison-001"
    config = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        output_dir=output_dir,
        overrides=[
            "training.warmup_ratio=0.9",
            "training.warmup_steps=7",
            "training.optimizer.name=sgd",
            "training.optimizer.kwargs.momentum=0.8",
            "training.optimizer.kwargs.nesterov=true",
        ],
    )

    summary = build_run_summary(config, tokens_seen=128, notes=["schedule smoke"])
    summary_path = write_run_summary(output_dir, summary)

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["base_learning_rate"] == 0.0003
    assert saved_summary["learning_rate_scale_rule"] == "linear"
    assert saved_summary["learning_rate_scale_factor"] == 4.0
    assert saved_summary["resolved_learning_rate"] == 0.0012
    assert saved_summary["warmup_ratio"] == 0.9
    assert saved_summary["warmup_steps"] == 7
    assert saved_summary["resolved_warmup_steps"] == 7
    assert saved_summary["scheduler_warmup_steps"] == 7
    assert saved_summary["scheduler_resolved_warmup_steps"] == 7
    assert saved_summary["gradient_clip_norm"] == 1.0
    assert saved_summary["optimizer_name"] == "sgd"
    assert saved_summary["optimizer_kwargs"] == {
        "momentum": 0.8,
        "dampening": 0.0,
        "nesterov": True,
        "weight_decay": 0.0,
    }


def test_run_summary_schema_requires_budget_derived_fields(tmp_path):
    output_dir = tmp_path / "dmodel256-pilot-comparison-001"
    config = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        output_dir=output_dir,
    )
    summary = build_run_summary(
        config,
        tokens_seen=128,
        extra_fields={
            "expected_tokens_per_step": 8192,
            "derived_max_steps": 12208,
            "effective_world_size": 1,
            "stop_reason": "token_budget_reached",
        },
    )
    summary.pop("stop_reason")

    with pytest.raises(ArtifactError, match="stop_reason"):
        write_run_summary(output_dir, summary)


def _checkpoint_summary_builder():
    import utils.metrics as metrics

    builder = getattr(metrics, "build_checkpoint_summary_fields", None)
    assert (
        builder is not None
    ), "utils.metrics.build_checkpoint_summary_fields is required"
    return builder


def test_best_eval_checkpoint_summary_selects_lowest_validation_loss(tmp_path):
    output_dir = tmp_path / "dmodel256-pilot-comparison-001"
    config = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        output_dir=output_dir,
    )
    metrics_rows = [
        {
            "run_id": "dmodel256-pilot-comparison-001",
            "step": 100,
            "split": "validation",
            "granularity": "xl",
            "loss": 2.0,
            "perplexity": 7.39,
        },
        {
            "run_id": "dmodel256-pilot-comparison-001",
            "step": 200,
            "split": "validation",
            "granularity": "xl",
            "loss": 1.5,
            "perplexity": 4.48,
        },
        {
            "run_id": "dmodel256-pilot-comparison-001",
            "step": 300,
            "split": "validation",
            "granularity": "xl",
            "loss": 1.8,
            "perplexity": 6.05,
        },
    ]

    fields = _checkpoint_summary_builder()(
        config,
        metrics_rows,
        validation_enabled=True,
        save_checkpoints=True,
    )

    assert fields["checkpoint_status"] == "best_eval"
    assert fields["checkpoint_metric"] == "validation_loss"
    assert fields["checkpoint_metric_value"] == 1.5
    assert fields["checkpoint_selection_step"] == 200
    assert fields["best_checkpoint_path"] == str(
        output_dir / "checkpoints" / "best_eval_step_200.pt"
    )
    assert fields["final_checkpoint_path"] is None

    summary = build_run_summary(config, tokens_seen=1024, extra_fields=fields)
    for field_name in [
        "checkpoint_status",
        "best_checkpoint_path",
        "final_checkpoint_path",
        "checkpoint_metric",
    ]:
        assert field_name in summary


def test_final_checkpoint_summary_when_validation_is_disabled(tmp_path):
    output_dir = tmp_path / "dmodel256-pilot-comparison-001"
    config = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        output_dir=output_dir,
        overrides=["evaluation.validation=false"],
    )

    fields = _checkpoint_summary_builder()(
        config,
        metrics_rows=[],
        validation_enabled=False,
        save_checkpoints=True,
    )

    assert fields["checkpoint_status"] == "final"
    assert fields["best_checkpoint_path"] is None
    assert fields["final_checkpoint_path"] == str(
        output_dir / "checkpoints" / "final.pt"
    )
    assert fields["checkpoint_metric"] is None
    assert fields["checkpoint_unavailable_reason"] is None


def test_no_checkpoint_summary_when_checkpoint_writes_are_disabled(tmp_path):
    output_dir = tmp_path / "dmodel256-pilot-comparison-001"
    config = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        output_dir=output_dir,
        overrides=[
            "evaluation.validation=false",
            "outputs.save_checkpoints=false",
        ],
    )

    fields = _checkpoint_summary_builder()(
        config,
        metrics_rows=[],
        validation_enabled=False,
        save_checkpoints=False,
    )

    assert fields["checkpoint_status"] == "none"
    assert fields["best_checkpoint_path"] is None
    assert fields["final_checkpoint_path"] is None
    assert fields["checkpoint_metric"] is None
    assert "disabled" in fields["checkpoint_unavailable_reason"]


def test_rank_zero_only_shared_artifact_helper_writes_on_rank_zero(tmp_path):
    from training.distributed import DistributedContext, rank_zero_only

    context = DistributedContext(
        enabled=True,
        rank=0,
        local_rank=0,
        world_size=2,
        strategy="fsdp",
        device="cpu",
    )
    artifact_path = tmp_path / "rank-zero-artifact.json"
    calls = []

    def write_artifact():
        calls.append("write")
        artifact_path.write_text('{"status": "written"}\n', encoding="utf-8")
        return artifact_path

    result = rank_zero_only(context, write_artifact)

    assert result == artifact_path
    assert calls == ["write"]
    assert artifact_path.exists()


def test_rank_zero_only_shared_artifact_helper_skips_nonzero_rank(tmp_path):
    from training.distributed import DistributedContext, rank_zero_only

    context = DistributedContext(
        enabled=True,
        rank=1,
        local_rank=1,
        world_size=2,
        strategy="fsdp",
        device="cpu",
    )
    artifact_path = tmp_path / "nonzero-rank-artifact.json"
    calls = []

    def write_artifact():
        calls.append("write")
        artifact_path.write_text('{"status": "written"}\n', encoding="utf-8")
        return artifact_path

    result = rank_zero_only(context, write_artifact)

    assert result is None
    assert calls == []
    assert not artifact_path.exists()


def test_write_all_csv_artifact_types(tmp_path):
    output_dir = tmp_path / "debug-nested-001"

    task_path = write_task_results_csv(
        output_dir,
        {
            "run_id": "debug-nested-001",
            "suite_id": "debug-downstream",
            "task": "hellaswag",
            "model_family": "nested",
            "model_size_label": "debug",
            "granularity": "s",
            "metric_name": "accuracy",
            "metric_value": 0.25,
        },
    )
    scaling_path = write_scaling_results_csv(
        output_dir,
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
            "granularity": "s",
            "total_parameters": 1000,
            "embedding_parameters": 100,
            "lm_head_parameters": 100,
            "non_embedding_parameters": 800,
            "loss": 2.1,
            "perplexity": 8.17,
            "average_downstream_accuracy": 0.25,
        },
    )
    consistency_path = write_consistency_results_csv(
        output_dir,
        {
            "comparison_id": "debug-s-xl",
            "small_run_id": "debug-nested-001",
            "large_run_id": "debug-nested-001",
            "small_granularity": "s",
            "large_granularity": "xl",
            "metric_name": "argmax_agreement",
            "metric_value": 0.72,
            "sample_count": 16,
        },
    )

    for artifact_path in [task_path, scaling_path, consistency_path]:
        with artifact_path.open("r", encoding="utf-8", newline="") as artifact_file:
            rows = list(csv.DictReader(artifact_file))
        assert len(rows) == 1


def test_build_consistency_result_rows_normalizes_top_k_and_deferred_metrics():
    rows = build_consistency_result_rows(
        [
            {
                "comparison_id": "debug-s-xl",
                "small_run_id": "debug-nested-001",
                "large_run_id": "debug-nested-001",
                "small_granularity": "s",
                "large_granularity": "xl",
                "metric_name": "top_k_overlap",
                "metric_value": 0.75,
                "sample_count": 16,
                "top_k": 5,
            },
            {
                "comparison_id": "debug-s-xl",
                "small_run_id": "debug-nested-001",
                "large_run_id": "debug-nested-001",
                "small_granularity": "s",
                "large_granularity": "xl",
                "metric_name": "kl_divergence",
                "metric_value": None,
                "sample_count": 16,
                "deferred": True,
                "deferred_reason": "later phase",
            },
        ]
    )

    assert rows == [
        {
            "comparison_id": "debug-s-xl",
            "small_run_id": "debug-nested-001",
            "large_run_id": "debug-nested-001",
            "small_granularity": "s",
            "large_granularity": "xl",
            "metric_name": "top_k_overlap@5",
            "metric_value": 0.75,
            "sample_count": 16,
        },
        {
            "comparison_id": "debug-s-xl",
            "small_run_id": "debug-nested-001",
            "large_run_id": "debug-nested-001",
            "small_granularity": "s",
            "large_granularity": "xl",
            "metric_name": "kl_divergence_deferred",
            "metric_value": None,
            "sample_count": 16,
        },
    ]


def test_write_consistency_results_csv_preserves_normalized_metric_names(tmp_path):
    output_dir = tmp_path / "consistency-001"

    artifact_path = write_consistency_results_csv(
        output_dir,
        [
            {
                "comparison_id": "debug-s-xl",
                "small_run_id": "debug-nested-001",
                "large_run_id": "debug-nested-001",
                "small_granularity": "s",
                "large_granularity": "xl",
                "metric_name": "top_k_overlap",
                "metric_value": 0.75,
                "sample_count": 16,
                "top_k": 5,
            },
            {
                "comparison_id": "debug-s-xl",
                "small_run_id": "debug-nested-001",
                "large_run_id": "debug-nested-001",
                "small_granularity": "s",
                "large_granularity": "xl",
                "metric_name": "kl_divergence",
                "metric_value": None,
                "sample_count": 16,
                "deferred": True,
            },
        ],
    )

    with artifact_path.open("r", encoding="utf-8", newline="") as artifact_file:
        rows = list(csv.DictReader(artifact_file))

    assert [row["metric_name"] for row in rows] == [
        "top_k_overlap@5",
        "kl_divergence_deferred",
    ]
    assert rows[1]["metric_value"] == ""


def test_build_scaling_rows_uses_latest_validation_metrics():
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
    )
    metrics_rows = [
        {
            "run_id": "debug-nested-001",
            "step": 1,
            "split": "validation",
            "model_family": "nested",
            "model_size_label": "debug",
            "granularity": "s",
            "loss": 2.5,
            "perplexity": 12.18,
            "tokens_seen": 32,
            "wall_clock_seconds": 1.0,
            "tokens_per_second": 32.0,
            "peak_memory_bytes": 0,
        },
        {
            "run_id": "debug-nested-001",
            "step": 2,
            "split": "validation",
            "model_family": "nested",
            "model_size_label": "debug",
            "granularity": "s",
            "loss": 2.0,
            "perplexity": 7.39,
            "tokens_seen": 64,
            "wall_clock_seconds": 2.0,
            "tokens_per_second": 32.0,
            "peak_memory_bytes": 0,
        },
    ]
    for granularity in ["m", "l", "xl"]:
        row = dict(metrics_rows[-1])
        row["granularity"] = granularity
        metrics_rows.append(row)

    parameter_counts = {
        granularity: {
            "total_parameters": index * 1000,
            "embedding_parameters": 100,
            "lm_head_parameters": 100,
            "non_embedding_parameters": index * 1000 - 200,
        }
        for index, granularity in enumerate(["s", "m", "l", "xl"], start=1)
    }

    rows = build_scaling_result_rows(config, metrics_rows, parameter_counts)

    assert [row["granularity"] for row in rows] == ["s", "m", "l", "xl"]
    assert rows[0]["comparison_id"] == "debug-nested-001__s"
    assert rows[0]["loss"] == 2.0
    assert rows[0]["non_embedding_parameters"] == 800


def test_scaling_result_schema_exposes_phase5_reporting_fields(tmp_path):
    output_dir = tmp_path / "dmodel256-pilot-comparison-001"
    config = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        output_dir=output_dir,
    )
    metrics_rows = [
        {
            "run_id": "dmodel256-pilot-comparison-001",
            "step": 10,
            "split": "validation",
            "model_family": "nested",
            "model_size_label": "dmodel256",
            "sampling_mode": "nested-random",
            "model_shape_label": "dmodel256",
            "granularity": granularity,
            "loss": 2.0 + index * 0.1,
            "perplexity": 7.0 + index,
            "tokens_seen": 81920,
            "content_tokens_seen": 80000,
            "wall_clock_seconds": 20.0,
            "tokens_per_second": 4096.0,
            "peak_memory_bytes": 1024,
        }
        for index, granularity in enumerate(["s", "m", "l", "xl"])
    ]
    parameter_counts = {
        granularity: {
            "total_parameters": 1000 + index,
            "embedding_parameters": 100,
            "lm_head_parameters": 100,
            "non_embedding_parameters": 800 + index,
            "ffn_parameters": 400 + index,
            "attention_parameters": 200,
            "other_non_embedding_parameters": 200 + index,
            "lm_head_counting": "separately_counted",
        }
        for index, granularity in enumerate(["s", "m", "l", "xl"])
    }

    rows = build_scaling_result_rows(config, metrics_rows, parameter_counts)
    scaling_path = write_scaling_results_csv(output_dir, rows)

    with scaling_path.open("r", encoding="utf-8", newline="") as scaling_file:
        reader = csv.DictReader(scaling_file)
        assert reader.fieldnames == SCALING_RESULTS_COLUMNS
        saved_rows = list(reader)

    assert len(saved_rows) == 4
    row = saved_rows[0]
    for field_name in [
        "comparison_id",
        "run_id",
        "model_family",
        "model_size_label",
        "sampling_mode",
        "model_shape_label",
        "model_family_slug",
        "model_size_slug",
        "token_budget_slug",
        "output_group",
        "completion_label",
        "granularity",
        "d_model",
        "num_layers",
        "num_attention_heads",
        "context_length",
        "vocab_size_assumption",
        "token_budget",
        "effective_world_size",
        "total_parameters",
        "embedding_parameters",
        "lm_head_parameters",
        "non_embedding_parameters",
        "ffn_parameters",
        "attention_parameters",
        "other_non_embedding_parameters",
        "lm_head_counting",
        "checkpoint_path",
        "loss",
        "perplexity",
        "average_downstream_accuracy",
    ]:
        assert field_name in row

    assert row["sampling_mode"] == "nested-random"
    assert row["model_shape_label"] == "dmodel256"
    assert row["token_budget"] == "100000000"
    assert row["effective_world_size"] == "1"
    assert row["lm_head_counting"] == "separately_counted"


def test_append_metrics_keeps_one_header(tmp_path):
    output_dir = tmp_path / "debug-nested-001"
    first_row = {
        "run_id": "debug-nested-001",
        "step": 0,
        "split": "validation",
        "model_family": "nested",
        "model_size_label": "debug",
        "granularity": "s",
        "loss": 2.1,
        "perplexity": 8.17,
        "tokens_seen": 128,
        "wall_clock_seconds": 1.5,
        "tokens_per_second": 85.3,
        "peak_memory_bytes": 2048,
    }
    second_row = dict(first_row)
    second_row["step"] = 1
    second_row["tokens_seen"] = 256

    metrics_path = write_metrics_csv(output_dir, first_row)
    write_metrics_csv(output_dir, second_row, append=True)

    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("run_id,step,split")
    assert sum(1 for line in lines if line.startswith("run_id,step,split")) == 1
    assert len(lines) == 3


def test_metric_writer_rejects_missing_required_fields(tmp_path):
    with pytest.raises(ArtifactError, match="peak_memory_bytes"):
        write_metrics_csv(
            tmp_path / "debug-nested-001",
            {
                "run_id": "debug-nested-001",
                "step": 0,
                "split": "validation",
                "model_family": "nested",
                "model_size_label": "debug",
                "granularity": "s",
                "loss": 2.1,
                "perplexity": 8.17,
                "tokens_seen": 128,
                "wall_clock_seconds": 1.5,
                "tokens_per_second": 85.3,
            },
        )


def test_nested_run_writes_extraction_metadata_artifact(tmp_path):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "training.max_steps=1",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.scheduler.kwargs.warmup_steps=0",
        ],
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5]],
            "attention_mask": [[1, 1, 0], [1, 1, 1]],
        }
    )

    run_training(
        config,
        model=TinyExtractionModel(),
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    metadata_path = output_dir / "extraction_metadata.json"
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["run_id"] == "debug-nested-001"
    assert metadata["model_family"] == "nested"

    granularities = metadata["granularities"]
    assert [entry["granularity"] for entry in granularities] == ["s", "m", "l", "xl"]
    assert [entry["display_name"] for entry in granularities] == ["S", "M", "L", "XL"]
    assert [entry["prefix_width"] for entry in granularities] == [8, 16, 32, 64]
    assert granularities[0]["strict_prefix_of"] == ["m", "l", "xl"]
    assert granularities[-1]["strict_prefix_of"] == []
