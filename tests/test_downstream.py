import csv

import pytest

from utils.metrics import (
    ArtifactError,
    TASK_RESULTS_COLUMNS,
    write_task_results_csv,
)


MINIMAL_DOWNSTREAM_TASKS = [
    "hellaswag",
    "piqa",
    "arc_challenge",
    "boolq",
    "winogrande",
    "openbookqa",
]


def test_downstream_task_results_schema_records_minimal_suite(tmp_path):
    output_dir = tmp_path / "dmodel256-nested-random-001"
    rows = [
        {
            "run_id": "dmodel256-nested-random-001",
            "suite_id": "minimal-downstream",
            "task": task_name,
            "model_family": "nested",
            "sampling_mode": "nested-random",
            "model_shape_label": "dmodel256",
            "table_reference_label": "matlm_78m",
            "granularity": "s",
            "metric_name": "accuracy",
            "metric_value": 0.25 + index * 0.01,
        }
        for index, task_name in enumerate(MINIMAL_DOWNSTREAM_TASKS)
    ]

    task_results_path = write_task_results_csv(output_dir, rows)

    with task_results_path.open("r", encoding="utf-8", newline="") as results_file:
        reader = csv.DictReader(results_file)
        assert reader.fieldnames == TASK_RESULTS_COLUMNS
        saved_rows = list(reader)

    assert [row["task"] for row in saved_rows] == MINIMAL_DOWNSTREAM_TASKS
    for row in saved_rows:
        assert row["suite_id"] == "minimal-downstream"
        assert row["model_family"] == "nested"
        assert row["sampling_mode"] == "nested-random"
        assert row["model_shape_label"] == "dmodel256"
        assert row["table_reference_label"] == "matlm_78m"
        assert row["granularity"] == "s"
        assert row["metric_name"] == "accuracy"
        assert row["metric_value"] != ""


def test_downstream_task_results_schema_requires_metric_fields(tmp_path):
    output_dir = tmp_path / "dmodel256-standalone-s-001"

    with pytest.raises(ArtifactError, match="metric_value"):
        write_task_results_csv(
            output_dir,
            {
                "run_id": "dmodel256-standalone-s-001",
                "suite_id": "minimal-downstream",
                "task": "hellaswag",
                "model_family": "standalone",
                "sampling_mode": "standalone",
                "model_shape_label": "dmodel256",
                "table_reference_label": "matlm_78m",
                "granularity": "s",
                "metric_name": "accuracy",
            },
        )

