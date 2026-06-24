from pathlib import Path

from src.evaluation.reporting import generate_figures
from src.utils.metrics import write_metrics_csv, write_scaling_results_csv
from src.utils.monitoring import group_loss_rows_by_series


def test_make_figures_cli_forwards_validation_loss_log_y(monkeypatch):
    import scripts.make_figures as make_figures

    captured = {}

    def fake_generate_figures(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return [Path("outputs/figures/example.png")]

    monkeypatch.setattr("src.evaluation.reporting.generate_figures", fake_generate_figures)

    make_figures.main(["--validation-loss-log-y"])

    assert captured["kwargs"]["validation_loss_log_y"] is True


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

    figure_paths = generate_figures(tmp_path, tmp_path / "figures", refresh_counts=False)
    figure_names = {path.name for path in figure_paths}

    assert "loss_vs_size.png" in figure_names
    assert "medium_trend_report.md" in figure_names
    assert any(name.startswith("validation_loss_over_tokens_") for name in figure_names)

    report = (tmp_path / "figures" / "medium_trend_report.md").read_text(
        encoding="utf-8"
    )
    assert "- nested-random: 2 rows; granularities=s, xl" in report
    assert "average_downstream_accuracy: 0.58" in report
