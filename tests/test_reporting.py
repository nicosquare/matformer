import json
from pathlib import Path

import pytest

from src.evaluation.reporting import generate_figures
from src.utils.metrics import write_metrics_csv, write_scaling_results_csv
from src.utils.monitoring import group_loss_rows_by_series


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
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    config = {
        "run": {"run_id": run_id},
        "model": {
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

    assert "loss_vs_size.png" not in figure_names
    assert "medium_trend_report.md" in figure_names
    assert any(name.startswith("validation_loss_over_tokens_") for name in figure_names)

    report = (tmp_path / "figures" / "medium_trend_report.md").read_text(
        encoding="utf-8"
    )
    assert "- nested-random: 2 rows; granularities=s, xl" in report
    assert "average_downstream_accuracy: 0.58" in report


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
        "probabilistic_per_block_thompson",
    }

    assert bayesian_identities <= set(SCALING_SAMPLING_MARKERS)
    assert bayesian_identities <= set(SCALING_SAMPLING_TONES)
    assert SCALING_SAMPLING_MARKERS["probabilistic_global_thompson"] != (
        SCALING_SAMPLING_MARKERS["probabilistic_per_block_thompson"]
    )
    panel_identities = {panel[2] for panel in SIZE_PLOT_PANELS_WITH_SAMPLING}
    assert bayesian_identities <= panel_identities
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
        "micro" if block_index % 2 == 0 else "full"
        for block_index in range(16)
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
                "global_granularity": ("micro", "medium", "full")[
                    window_index % 3
                ]
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
    assert any("malformed-profile" in message and "unknown" in message for message in messages)
    assert any("missing-config" in message and "config.json" in message for message in messages)


def test_generate_figures_returns_global_and_per_block_controller_timeline_pngs(
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
                action={
                    "block_granularities": ["full", "medium", "micro"]
                },
            )
        ],
    )

    paths = generate_figures(
        tmp_path,
        tmp_path / "figures",
        refresh_counts=False,
        dpi=40,
    )
    timeline_paths = {
        path.name: path
        for path in paths
        if path.name.startswith("selected_granularity_over_tokens_")
    }

    assert set(timeline_paths) == {
        "selected_granularity_over_tokens_global-timeline.png",
        "selected_granularity_over_tokens_per-block-timeline.png",
    }
    for path in timeline_paths.values():
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
    assert {
        path.name
        for path in compatibility_paths
        if path.name.startswith("selected_granularity_over_tokens_")
    } == set(timeline_paths)
