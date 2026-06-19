import csv

import pytest

from src.evaluation.downstream import (
    MINIMAL_DOWNSTREAM_TASKS,
    build_lm_eval_command,
    downstream_suite_from_config,
    lm_eval_results_to_task_rows,
    select_lm_eval_metric,
)
from src.evaluation.validation import aggregate_scaling_summary
from src.utils.config import resolve_run_config
from src.utils.metrics import (
    ArtifactError,
    TASK_RESULTS_COLUMNS,
    write_task_results_csv,
)


def test_downstream_task_results_schema_records_minimal_suite(tmp_path):
    output_dir = tmp_path / "dmodel256-nested-random-001"
    rows = [
        {
            "run_id": "dmodel256-nested-random-001",
            "suite_id": "minimal-downstream",
            "task": task_name,
            "model_family": "nested",
            "model_size_label": "dmodel256",
            "sampling_mode": "nested-random",
            "model_shape_label": "dmodel256",
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
        assert row["model_size_label"] == "dmodel256"
        assert row["model_shape_label"] == "dmodel256"
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
                "model_size_label": "dmodel256",
                "sampling_mode": "standalone",
                "model_shape_label": "dmodel256",
                "granularity": "s",
                "metric_name": "accuracy",
            },
        )


def test_dmodel256_config_declares_minimal_lm_eval_suite():
    config = resolve_run_config("configs/dmodel256_pilot_comparison.yaml")

    suite = downstream_suite_from_config(config)

    assert suite["suite_id"] == "minimal-downstream"
    assert suite["suite_type"] == "downstream"
    assert suite["tool"] == "lm-eval"
    assert suite["tasks"] == MINIMAL_DOWNSTREAM_TASKS
    assert suite["metric_name"] == "accuracy"
    assert suite["model"] == "hf"
    assert suite["batch_size"] == "auto"


def test_lm_eval_command_uses_preferred_harness_cli(tmp_path):
    config = resolve_run_config("configs/dmodel256_pilot_comparison.yaml")
    checkpoint_path = tmp_path / "checkpoints" / "best_eval.pt"
    output_path = tmp_path / "lm_eval_results.json"

    command = build_lm_eval_command(
        config,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
    )

    assert command[0] == "lm_eval"
    assert _arg_value(command, "--model") == "hf"
    assert _arg_value(command, "--tasks") == ",".join(MINIMAL_DOWNSTREAM_TASKS)
    assert _arg_value(command, "--batch_size") == "auto"
    assert _arg_value(command, "--output_path") == str(output_path)
    model_args = _arg_value(command, "--model_args")
    assert f"pretrained={checkpoint_path}" in model_args
    assert "tokenizer=hf-internal-testing/llama-tokenizer" in model_args


def test_lm_eval_results_convert_to_task_rows():
    config = resolve_run_config("configs/dmodel256_pilot_comparison.yaml")
    lm_eval_results = {
        "results": {
            task_name: {"acc_norm,none": 0.2 + index * 0.01}
            for index, task_name in enumerate(MINIMAL_DOWNSTREAM_TASKS)
        }
    }

    rows = lm_eval_results_to_task_rows(
        lm_eval_results,
        config,
        granularity="xl",
    )

    assert [row["task"] for row in rows] == MINIMAL_DOWNSTREAM_TASKS
    assert {row["metric_name"] for row in rows} == {"accuracy"}
    assert {row["model_family"] for row in rows} == {"nested"}
    assert {row["sampling_mode"] for row in rows} == {"nested-random"}
    assert {row["model_shape_label"] for row in rows} == {"dmodel256"}
    assert {row["granularity"] for row in rows} == {"xl"}
    assert rows[0]["metric_value"] == 0.2


def test_lm_eval_metric_selection_uses_first_available_candidate():
    value = select_lm_eval_metric(
        {"acc,none": 0.3, "acc_norm,none": 0.4},
        ["acc_norm,none", "acc,none"],
    )

    assert value == 0.4


def test_scaling_summary_aggregation_adds_average_downstream_accuracy():
    scaling_rows = [
        {
            "run_id": "dmodel256-nested-random-001",
            "granularity": "s",
            "average_downstream_accuracy": None,
        },
        {
            "run_id": "dmodel256-nested-random-001",
            "granularity": "xl",
            "average_downstream_accuracy": None,
        },
    ]
    task_rows = [
        {
            "run_id": "dmodel256-nested-random-001",
            "granularity": "s",
            "metric_name": "accuracy",
            "metric_value": 0.2,
        },
        {
            "run_id": "dmodel256-nested-random-001",
            "granularity": "s",
            "metric_name": "accuracy",
            "metric_value": 0.4,
        },
        {
            "run_id": "dmodel256-nested-random-001",
            "granularity": "xl",
            "metric_name": "accuracy",
            "metric_value": 0.8,
        },
    ]

    rows = aggregate_scaling_summary(scaling_rows, task_rows)

    assert rows[0]["average_downstream_accuracy"] == pytest.approx(0.3)
    assert rows[1]["average_downstream_accuracy"] == pytest.approx(0.8)


def _arg_value(command, flag):
    return command[command.index(flag) + 1]
