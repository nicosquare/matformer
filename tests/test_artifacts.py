import csv
import json

import pytest

from utils.config import resolve_run_config
from utils.metrics import (
    ArtifactError,
    build_run_summary,
    write_config_artifact,
    write_consistency_results_csv,
    write_failed_run_summary,
    write_metrics_csv,
    write_run_summary,
    write_scaling_results_csv,
    write_task_results_csv,
)


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
