import csv
import json
from types import SimpleNamespace

import torch
from datasets import Dataset

from scripts.make_figures import generate_figures
from train import parse_args
from training.baselines import (
    add_baseline_notes_to_summary,
    build_baseline_match_record,
    compare_baseline_configs,
    run_debug_nested_with_one_baseline,
)
from training.run import run_training
from utils.config import resolve_run_config
from utils.metrics import (
    build_run_summary,
    write_consistency_results_csv,
    write_run_summary,
    write_scaling_results_csv,
)


class TinyTrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.configured_granularities = []

    def configure_subnetwork(self, granularity):
        self.configured_granularities.append(granularity)
        self.current_granularity = granularity

    def forward(self, input_ids, attention_mask=None, labels=None):
        loss = (self.weight - 0.25).pow(2) + input_ids.float().mean() * 0.0
        return SimpleNamespace(loss=loss)


def test_configured_training_writes_metrics_config_and_summary(tmp_path):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "training.max_steps=2",
            "training.eval_interval=1",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.warmup_steps=0",
        ],
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5], [6, 0, 0]],
            "attention_mask": [[1, 1, 0], [1, 1, 1], [1, 0, 0]],
        }
    )
    model = TinyTrainModel()

    result = run_training(
        config,
        model=model,
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    assert (output_dir / "config.json").exists()
    assert result["metrics_path"] == output_dir / "metrics.csv"
    assert result["summary_path"] == output_dir / "run_summary.json"

    with (output_dir / "metrics.csv").open("r", encoding="utf-8", newline="") as metrics_file:
        metric_rows = list(csv.DictReader(metrics_file))
    assert {row["split"] for row in metric_rows} == {"train", "validation"}
    train_rows = [row for row in metric_rows if row["split"] == "train"]
    assert [row["granularity"] for row in train_rows] == [
        "s",
        "m",
        "l",
        "xl",
        "s",
        "m",
        "l",
        "xl",
    ]

    summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["steps_completed"] == 2
    assert summary["tokens_seen"] > 0


def test_baseline_match_records_mismatches_in_summary(tmp_path):
    nested = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=tmp_path / "debug-nested-001",
    )
    standalone = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-standalone-s-001",
        output_dir=tmp_path / "debug-standalone-s-001",
    )

    assert compare_baseline_configs(nested, standalone, "s") == []

    standalone["training"]["token_budget"] = 123
    record = build_baseline_match_record(
        nested,
        standalone,
        "s",
        nested_counts={"non_embedding_parameters": 100},
        standalone_counts={"non_embedding_parameters": 95},
    )
    summary = build_run_summary(nested, tokens_seen=10)
    summary = add_baseline_notes_to_summary(summary, [record])

    assert record["non_embedding_parameters_nested"] == 100
    assert record["non_embedding_parameters_standalone"] == 95
    assert "token budget mismatch" in record["match_notes"][0]
    assert summary["baseline_mismatch_notes"]
    assert summary["baseline_matches"][0]["match_id"] == record["match_id"]


def test_debug_nested_with_one_baseline_path_updates_summary(tmp_path):
    output_root = tmp_path / "outputs"
    called_run_ids = []

    def fake_runner(config):
        called_run_ids.append(config["run"]["run_id"])
        output_dir = config["run"]["output_dir"]
        summary = build_run_summary(config, tokens_seen=1)
        summary_path = write_run_summary(output_dir, summary)
        return {"summary_path": summary_path}

    result = run_debug_nested_with_one_baseline(
        overrides=[f"run.output_root={output_root}"],
        runner=fake_runner,
    )

    assert called_run_ids == ["debug-nested-001", "debug-standalone-s-001"]
    assert result["standalone_config"]["run"]["granularity"] == "s"

    summary = json.loads(
        result["nested_summary_path"].read_text(encoding="utf-8")
    )
    assert summary["baseline_matches"][0]["nested_run_id"] == "debug-nested-001"
    assert (
        summary["baseline_matches"][0]["standalone_run_id"]
        == "debug-standalone-s-001"
    )
    assert summary["baseline_mismatch_notes"] == []


def test_make_figures_reads_csv_artifacts(tmp_path):
    run_dir = tmp_path / "debug-nested-001"
    write_scaling_results_csv(
        run_dir,
        [
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
                "loss": 2.0,
                "perplexity": 7.4,
                "average_downstream_accuracy": 0.2,
            },
            {
                "comparison_id": "debug-xl",
                "run_id": "debug-nested-001",
                "model_family": "nested",
                "model_size_label": "debug",
                "completion_label": "debug",
                "granularity": "xl",
                "total_parameters": 2000,
                "embedding_parameters": 100,
                "lm_head_parameters": 100,
                "non_embedding_parameters": 1800,
                "loss": 1.5,
                "perplexity": 4.5,
                "average_downstream_accuracy": 0.3,
            },
        ],
    )
    write_consistency_results_csv(
        run_dir,
        {
            "comparison_id": "debug-s-xl",
            "small_run_id": "debug-nested-001",
            "large_run_id": "debug-nested-001",
            "small_granularity": "s",
            "large_granularity": "xl",
            "metric_name": "argmax_agreement",
            "metric_value": 0.7,
            "sample_count": 16,
        },
    )

    figure_paths = generate_figures(tmp_path, tmp_path / "figures")

    figure_names = {path.name for path in figure_paths}
    assert {"loss_vs_size.png", "ppl_vs_size.png", "consistency_vs_size.png"} <= figure_names
    for path in figure_paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_train_cli_accepts_config_run_id_and_overrides():
    args = parse_args(
        [
            "--config",
            "configs/debug_matrix.yaml",
            "--run-id",
            "debug-nested-001",
            "--override",
            "training.max_steps=1",
        ]
    )

    assert args.config == "configs/debug_matrix.yaml"
    assert args.run_id == "debug-nested-001"
    assert args.override == ["training.max_steps=1"]
