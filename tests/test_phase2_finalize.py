import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import Dataset

from scripts.make_figures import (
    enrich_scaling_metadata_from_run_config,
    generate_figures,
    group_scaling_rows,
    loss_moving_average_window_size,
    refresh_scaling_parameter_counts,
    scaling_curve_style,
    with_default_model_variant,
)
from train import parse_args
from training.baselines import (
    add_baseline_notes_to_summary,
    build_baseline_match_record,
    compare_baseline_configs,
    run_debug_nested_with_baselines,
    run_debug_nested_with_one_baseline,
)
from training.run import run_training
from utils.config import resolve_run_config
from utils.metrics import (
    build_run_summary,
    write_consistency_results_csv,
    write_metrics_csv,
    write_run_summary,
    write_scaling_results_csv,
    write_task_results_csv,
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
            "training.scheduler.kwargs.warmup_steps=0",
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
    assert result["scaling_path"] == output_dir / "scaling_results.csv"
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

    with (output_dir / "scaling_results.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as scaling_file:
        scaling_rows = list(csv.DictReader(scaling_file))
    assert [row["granularity"] for row in scaling_rows] == ["s", "m", "l", "xl"]
    assert scaling_rows[0]["comparison_id"] == "debug-nested-001__s"
    assert scaling_rows[0]["non_embedding_parameters"] == "1"

    summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["steps_completed"] == 2
    assert summary["tokens_seen"] > 0
    assert summary["scaling_results_path"] == str(output_dir / "scaling_results.csv")


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
        is_nested = config["run"]["model_family"] == "nested"
        non_embedding_parameters = 100 if is_nested else 95
        return {
            "summary_path": summary_path,
            "parameter_counts_by_granularity": {
                "s": {"non_embedding_parameters": non_embedding_parameters}
            },
        }

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
    assert summary["baseline_matches"][0]["non_embedding_parameters_nested"] == 100
    assert summary["baseline_matches"][0]["non_embedding_parameters_standalone"] == 95
    assert summary["baseline_mismatch_notes"] == []


def test_debug_nested_with_all_baselines_updates_summary(tmp_path):
    output_root = tmp_path / "outputs"
    called_run_ids = []

    def fake_runner(config):
        called_run_ids.append(config["run"]["run_id"])
        output_dir = config["run"]["output_dir"]
        summary = build_run_summary(config, tokens_seen=1)
        summary_path = write_run_summary(output_dir, summary)
        granularity = config["model"]["granularities"][0]
        if config["run"]["model_family"] == "nested":
            parameter_counts = {
                granularity: {"non_embedding_parameters": index * 100}
                for index, granularity in enumerate(["s", "m", "l", "xl"], start=1)
            }
        else:
            parameter_counts = {
                granularity: {"non_embedding_parameters": 95}
            }
        return {
            "summary_path": summary_path,
            "parameter_counts_by_granularity": parameter_counts,
        }

    result = run_debug_nested_with_baselines(
        overrides=[f"run.output_root={output_root}"],
        runner=fake_runner,
    )

    assert called_run_ids == [
        "debug-nested-001",
        "debug-standalone-s-001",
        "debug-standalone-m-001",
        "debug-standalone-l-001",
        "debug-standalone-xl-001",
    ]
    assert [
        config["run"]["granularity"]
        for config in result["standalone_configs"]
    ] == ["s", "m", "l", "xl"]

    summary = json.loads(
        result["nested_summary_path"].read_text(encoding="utf-8")
    )
    assert [
        record["standalone_run_id"]
        for record in summary["baseline_matches"]
    ] == [
        "debug-standalone-s-001",
        "debug-standalone-m-001",
        "debug-standalone-l-001",
        "debug-standalone-xl-001",
    ]
    assert [
        record["non_embedding_parameters_nested"]
        for record in summary["baseline_matches"]
    ] == [100, 200, 300, 400]
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
                "loss": 2.0,
                "perplexity": 7.4,
                "average_downstream_accuracy": 0.2,
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
    assert {
        "loss_vs_size.png",
        "ppl_vs_size.png",
        "accuracy_vs_size.png",
        "consistency_vs_size.png",
        "medium_trend_report.md",
    } <= figure_names
    for path in figure_paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_make_figures_writes_one_loss_figure_per_experiment_group(tmp_path):
    first_run_dir = tmp_path / "debug-nested-001"
    second_run_dir = tmp_path / "debug-standalone-s-001"
    third_run_dir = tmp_path / "debug-standalone-m-001"

    write_metrics_csv(
        first_run_dir,
        [
            {
                "run_id": "debug-nested-001",
                "step": 1,
                "split": "validation",
                "model_family": "nested",
                "model_size_label": "debug",
                "model_shape_label": "debug-shape",
                "sampling_mode": "nested-random",
                "granularity": "s",
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
                "step": 2,
                "split": "validation",
                "model_family": "nested",
                "model_size_label": "debug",
                "model_shape_label": "debug-shape",
                "sampling_mode": "nested-random",
                "granularity": "s",
                "loss": 1.9,
                "perplexity": 6.7,
                "tokens_seen": 256,
                "content_tokens_seen": 256,
                "wall_clock_seconds": 2.0,
                "tokens_per_second": 128.0,
                "peak_memory_bytes": 2048,
            },
            {
                "run_id": "debug-nested-001",
                "step": 1,
                "split": "validation",
                "model_family": "nested",
                "model_size_label": "debug",
                "model_shape_label": "debug-shape",
                "sampling_mode": "nested-random",
                "granularity": "m",
                "loss": 2.3,
                "perplexity": 10.0,
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
                "granularity": "m",
                "loss": 2.0,
                "perplexity": 6.7,
                "tokens_seen": 256,
                "content_tokens_seen": 256,
                "wall_clock_seconds": 2.0,
                "tokens_per_second": 128.0,
                "peak_memory_bytes": 2048,
            },
        ],
    )
    write_metrics_csv(
        second_run_dir,
        [
            {
                "run_id": "debug-standalone-s-001",
                "step": 1,
                "split": "validation",
                "model_family": "standalone",
                "model_size_label": "debug",
                "model_shape_label": "debug-shape",
                "sampling_mode": "standalone",
                "granularity": "s",
                "loss": 2.4,
                "perplexity": 11.0,
                "tokens_seen": 128,
                "content_tokens_seen": 128,
                "wall_clock_seconds": 1.0,
                "tokens_per_second": 128.0,
                "peak_memory_bytes": 2048,
            },
            {
                "run_id": "debug-standalone-s-001",
                "step": 2,
                "split": "validation",
                "model_family": "standalone",
                "model_size_label": "debug",
                "model_shape_label": "debug-shape",
                "sampling_mode": "standalone",
                "granularity": "s",
                "loss": 2.0,
                "perplexity": 7.4,
                "tokens_seen": 256,
                "content_tokens_seen": 256,
                "wall_clock_seconds": 2.0,
                "tokens_per_second": 128.0,
                "peak_memory_bytes": 2048,
            },
        ],
    )
    write_metrics_csv(
        third_run_dir,
        [
            {
                "run_id": "debug-standalone-m-001",
                "step": 1,
                "split": "validation",
                "model_family": "standalone",
                "model_size_label": "debug",
                "model_shape_label": "debug-shape",
                "sampling_mode": "standalone",
                "granularity": "m",
                "loss": 2.6,
                "perplexity": 13.5,
                "tokens_seen": 128,
                "content_tokens_seen": 128,
                "wall_clock_seconds": 1.0,
                "tokens_per_second": 128.0,
                "peak_memory_bytes": 2048,
            },
            {
                "run_id": "debug-standalone-m-001",
                "step": 2,
                "split": "validation",
                "model_family": "standalone",
                "model_size_label": "debug",
                "model_shape_label": "debug-shape",
                "sampling_mode": "standalone",
                "granularity": "m",
                "loss": 2.2,
                "perplexity": 9.0,
                "tokens_seen": 256,
                "content_tokens_seen": 256,
                "wall_clock_seconds": 2.0,
                "tokens_per_second": 128.0,
                "peak_memory_bytes": 2048,
            },
        ],
    )

    figure_paths = generate_figures(tmp_path, tmp_path / "figures")

    figure_names = {path.name for path in figure_paths}
    assert "loss_over_steps_debug_nested_001.png" in figure_names
    assert "loss_over_steps_debug_standalone_001.png" in figure_names
    assert "loss_over_steps.png" not in figure_names
    assert "loss_over_steps_grid.png" not in figure_names
    assert "ppl_over_steps.png" in figure_names

    nested_path = tmp_path / "figures" / "loss_over_steps_debug_nested_001.png"
    standalone_path = tmp_path / "figures" / "loss_over_steps_debug_standalone_001.png"
    assert nested_path.exists()
    assert nested_path.stat().st_size > 0
    assert standalone_path.exists()
    assert standalone_path.stat().st_size > 0


def test_make_figures_plots_grouped_consistency_metrics_and_skips_deferred_rows(tmp_path):
    run_dir = tmp_path / "consistency-001"
    write_consistency_results_csv(
        run_dir,
        [
            {
                "comparison_id": "nested-s-xl",
                "small_run_id": "debug-nested-001",
                "large_run_id": "debug-nested-001",
                "small_granularity": "s",
                "large_granularity": "xl",
                "metric_name": "token_level_agreement",
                "metric_value": 0.7,
                "sample_count": 16,
            },
            {
                "comparison_id": "nested-s-xl",
                "small_run_id": "debug-nested-001",
                "large_run_id": "debug-nested-001",
                "small_granularity": "s",
                "large_granularity": "xl",
                "metric_name": "top_k_overlap",
                "metric_value": 0.8,
                "sample_count": 16,
                "top_k": 5,
            },
            {
                "comparison_id": "nested-s-xl",
                "small_run_id": "debug-nested-001",
                "large_run_id": "debug-nested-001",
                "small_granularity": "s",
                "large_granularity": "xl",
                "metric_name": "kl_divergence",
                "metric_value": None,
                "sample_count": 16,
                "deferred": True,
            },
            {
                "comparison_id": "nested-m-l",
                "small_run_id": "debug-nested-001",
                "large_run_id": "debug-nested-001",
                "small_granularity": "m",
                "large_granularity": "l",
                "metric_name": "token_level_agreement",
                "metric_value": 0.6,
                "sample_count": 16,
            },
        ],
    )

    figure_paths = generate_figures(tmp_path, tmp_path / "figures")

    consistency_path = tmp_path / "figures" / "consistency_vs_size.png"
    assert consistency_path in figure_paths
    assert consistency_path.exists()
    assert consistency_path.stat().st_size > 0


def test_make_figures_aggregates_task_results_for_accuracy_plot(tmp_path):
    run_dir = tmp_path / "dmodel256-nested-random-001"
    write_scaling_results_csv(
        run_dir,
        [
            {
                "comparison_id": "dmodel256-s",
                "run_id": "dmodel256-nested-random-001",
                "model_family": "nested",
                "model_size_label": "dmodel256",
                "model_family_slug": "matformer_llama",
                "model_size_slug": "148m",
                "token_budget_slug": "100m_tokens",
                "output_group": "matformer_llama_148m_100m_tokens",
                "sampling_mode": "nested-random",
                "model_shape_label": "dmodel256",
                "completion_label": "run",
                "granularity": "s",
                "total_parameters": 1000,
                "embedding_parameters": 100,
                "lm_head_parameters": 100,
                "non_embedding_parameters": 800,
                "loss": 2.0,
                "perplexity": 7.4,
                "average_downstream_accuracy": None,
            },
            {
                "comparison_id": "dmodel256-xl",
                "run_id": "dmodel256-nested-random-001",
                "model_family": "nested",
                "model_size_label": "dmodel256",
                "model_family_slug": "matformer_llama",
                "model_size_slug": "148m",
                "token_budget_slug": "100m_tokens",
                "output_group": "matformer_llama_148m_100m_tokens",
                "sampling_mode": "nested-random",
                "model_shape_label": "dmodel256",
                "completion_label": "run",
                "granularity": "xl",
                "total_parameters": 2000,
                "embedding_parameters": 100,
                "lm_head_parameters": 100,
                "non_embedding_parameters": 1800,
                "loss": 1.5,
                "perplexity": 4.5,
                "average_downstream_accuracy": None,
            },
        ],
    )
    write_task_results_csv(
        run_dir,
        [
            {
                "run_id": "dmodel256-nested-random-001",
                "suite_id": "minimal-downstream",
                "task": "hellaswag",
                "model_family": "nested",
                "model_size_label": "dmodel256",
                "sampling_mode": "nested-random",
                "model_shape_label": "dmodel256",
                "granularity": "s",
                "metric_name": "accuracy",
                "metric_value": 0.2,
            },
            {
                "run_id": "dmodel256-nested-random-001",
                "suite_id": "minimal-downstream",
                "task": "piqa",
                "model_family": "nested",
                "model_size_label": "dmodel256",
                "sampling_mode": "nested-random",
                "model_shape_label": "dmodel256",
                "granularity": "s",
                "metric_name": "accuracy",
                "metric_value": 0.4,
            },
            {
                "run_id": "dmodel256-nested-random-001",
                "suite_id": "minimal-downstream",
                "task": "hellaswag",
                "model_family": "nested",
                "model_size_label": "dmodel256",
                "sampling_mode": "nested-random",
                "model_shape_label": "dmodel256",
                "granularity": "xl",
                "metric_name": "accuracy",
                "metric_value": 0.8,
            },
        ],
    )

    figure_paths = generate_figures(
        tmp_path,
        tmp_path / "figures",
        refresh_counts=False,
    )

    figure_names = {path.name for path in figure_paths}
    assert "accuracy_vs_size.png" in figure_names
    assert "medium_trend_report.md" in figure_names
    report = (tmp_path / "figures" / "medium_trend_report.md").read_text(
        encoding="utf-8"
    )
    assert "average_downstream_accuracy: 0.8" in report
    assert "nested-random / xl / dmodel256-nested-random-001" in report


def test_make_figures_refreshes_parameter_counts_from_run_config(tmp_path):
    run_dir = tmp_path / "debug-standalone-s-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-standalone-s-001",
        output_dir=run_dir,
    )
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str),
        encoding="utf-8",
    )
    stale_row = {
        "_source_csv": str(run_dir / "scaling_results.csv"),
        "run_id": "debug-standalone-s-001",
        "granularity": "s",
        "total_parameters": "999",
        "embedding_parameters": "0",
        "lm_head_parameters": "0",
        "non_embedding_parameters": "999",
        "ffn_parameters": "",
        "attention_parameters": "",
        "other_non_embedding_parameters": "",
        "lm_head_counting": "",
    }

    refreshed_rows = refresh_scaling_parameter_counts(tmp_path, [stale_row])

    from training.run import build_model
    from utils.model_size import model_parameter_counts

    model = build_model(config)
    expected_counts = model_parameter_counts(model, granularity="s")
    del model

    refreshed_row = refreshed_rows[0]
    assert refreshed_row["total_parameters"] == expected_counts["total_parameters"]
    assert (
        refreshed_row["non_embedding_parameters"]
        == expected_counts["non_embedding_parameters"]
    )
    assert refreshed_row["total_parameters"] != stale_row["total_parameters"]


def test_make_figures_enriches_model_variant_from_run_config(tmp_path):
    output_dir = tmp_path / "outputs"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir / "debug-nested-001",
        overrides=[
            "model.variant=cat_llama",
            "model.gradient_membership_correction=false",
        ],
    )
    write_run_summary(
        config["run"]["output_dir"],
        build_run_summary(config, tokens_seen=128),
    )
    config_path = Path(config["run"]["output_dir"]) / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = [
        {
            "run_id": "debug-nested-001",
            "model_family": "nested",
            "sampling_mode": "nested-random",
            "granularity": "s",
            "_source_csv": str(Path(config["run"]["output_dir"]) / "scaling_results.csv"),
        }
    ]

    enriched_rows = enrich_scaling_metadata_from_run_config(output_dir, rows)

    assert enriched_rows[0]["model_variant"] == "cat_llama"
    assert enriched_rows[0]["gradient_membership_correction"] is False


def test_make_figures_defaults_missing_model_variant_for_legacy_configs(tmp_path):
    output_dir = tmp_path / "outputs"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir / "debug-nested-001",
    )
    legacy_config = with_default_model_variant(config)
    legacy_config["model"].pop("variant")

    config_path = Path(config["run"]["output_dir"]) / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(legacy_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    stale_row = {
        "run_id": "debug-nested-001",
        "model_family": "nested",
        "sampling_mode": "nested-random",
        "granularity": "s",
        "total_parameters": "999",
        "embedding_parameters": "0",
        "lm_head_parameters": "0",
        "non_embedding_parameters": "999",
        "ffn_parameters": "",
        "attention_parameters": "",
        "other_non_embedding_parameters": "",
        "lm_head_counting": "",
        "_source_csv": str(Path(config["run"]["output_dir"]) / "scaling_results.csv"),
    }

    enriched_rows = enrich_scaling_metadata_from_run_config(output_dir, [stale_row])
    refreshed_rows = refresh_scaling_parameter_counts(output_dir, [stale_row])

    assert enriched_rows[0]["model_variant"] == "matformer_llama"
    assert refreshed_rows[0]["total_parameters"] != stale_row["total_parameters"]


def test_make_figures_groups_scaling_curves_by_sampling_mode():
    rows = [
        {"model_family": "nested", "sampling_mode": "nested-random", "granularity": "s"},
        {"model_family": "nested", "sampling_mode": "nested-random", "granularity": "xl"},
        {"model_family": "nested", "sampling_mode": "nested-all", "granularity": "s"},
        {"model_family": "nested", "sampling_mode": "nested-all", "granularity": "xl"},
        {"model_family": "standalone", "sampling_mode": "standalone", "granularity": "s"},
        {"model_family": "standalone", "sampling_mode": "standalone", "granularity": "xl"},
    ]

    grouped = group_scaling_rows(rows)

    assert set(grouped) == {"nested-random", "nested-all", "standalone"}
    assert [row["granularity"] for row in grouped["nested-random"]] == ["s", "xl"]
    assert [row["granularity"] for row in grouped["standalone"]] == ["s", "xl"]


def test_make_figures_groups_scaling_curves_by_sampling_mode_and_variant():
    rows = [
        {
            "model_family": "nested",
            "sampling_mode": "nested-random",
            "model_variant": "matformer_llama",
            "granularity": "s",
        },
        {
            "model_family": "nested",
            "sampling_mode": "nested-random",
            "model_variant": "matformer_llama",
            "granularity": "xl",
        },
        {
            "model_family": "nested",
            "sampling_mode": "nested-random",
            "model_variant": "cat_llama",
            "granularity": "s",
        },
        {
            "model_family": "nested",
            "sampling_mode": "nested-random",
            "model_variant": "cat_llama",
            "granularity": "xl",
        },
    ]

    grouped = group_scaling_rows(rows)

    assert set(grouped) == {
        "nested-random / matformer_llama",
        "nested-random / cat_llama",
    }
    assert [row["granularity"] for row in grouped["nested-random / cat_llama"]] == [
        "s",
        "xl",
    ]


def test_make_figures_groups_scaling_curves_by_sampling_mode_variant_and_gmc():
    rows = [
        {
            "model_family": "nested",
            "sampling_mode": "nested-random",
            "model_variant": "matformer_llama",
            "gradient_membership_correction": True,
            "granularity": "s",
        },
        {
            "model_family": "nested",
            "sampling_mode": "nested-random",
            "model_variant": "matformer_llama",
            "gradient_membership_correction": True,
            "granularity": "xl",
        },
        {
            "model_family": "nested",
            "sampling_mode": "nested-random",
            "model_variant": "cat_llama",
            "gradient_membership_correction": False,
            "granularity": "s",
        },
        {
            "model_family": "nested",
            "sampling_mode": "nested-random",
            "model_variant": "cat_llama",
            "gradient_membership_correction": False,
            "granularity": "xl",
        },
    ]

    grouped = group_scaling_rows(rows)

    assert set(grouped) == {
        "nested-random / matformer_llama / gmc=on",
        "nested-random / cat_llama / gmc=off",
    }
    assert scaling_curve_style(grouped["nested-random / matformer_llama / gmc=on"]) == {
        "marker": "o",
        "linestyle": "-",
        "linewidth": 1.4,
    }
    assert scaling_curve_style(grouped["nested-random / cat_llama / gmc=off"]) == {
        "marker": "s",
        "linestyle": "--",
        "linewidth": 1.2,
    }


def test_loss_moving_average_window_size_scales_with_point_count():
    assert loss_moving_average_window_size(1) == 1
    assert loss_moving_average_window_size(5) == 3
    assert loss_moving_average_window_size(20) == 3
    assert loss_moving_average_window_size(50) == 5
    assert loss_moving_average_window_size(100) == 11


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
