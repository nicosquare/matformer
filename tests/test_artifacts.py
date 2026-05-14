import csv
import json
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

from utils.config import resolve_run_config
from utils.metrics import (
    ArtifactError,
    build_run_summary,
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

    with metrics_path.open("r", encoding="utf-8", newline="") as metrics_file:
        metric_rows = list(csv.DictReader(metrics_file))
    assert [row["granularity"] for row in metric_rows] == ["s", "xl"]
    assert metric_rows[0]["peak_memory_bytes"] == "2048"

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["status"] == "completed"
    assert saved_summary["tokens_seen"] == 128
    assert saved_summary["notes"] == ["smoke test"]


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
    assert saved_summary["notes"] == ["CUDA out of memory during debug smoke"]


def test_run_summary_includes_budget_derived_fields(tmp_path):
    output_dir = tmp_path / "78m-reduced-pilot-001"
    config = resolve_run_config(
        "configs/78m_reduced_pilot.yaml",
        output_dir=output_dir,
    )

    summary = build_run_summary(config, tokens_seen=128, notes=["budget smoke"])

    for field_name in [
        "expected_tokens_per_step",
        "derived_max_steps",
        "effective_world_size",
        "stop_reason",
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


def test_run_summary_schema_requires_budget_derived_fields(tmp_path):
    output_dir = tmp_path / "78m-reduced-pilot-001"
    config = resolve_run_config(
        "configs/78m_reduced_pilot.yaml",
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
            "training.warmup_steps=0",
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
