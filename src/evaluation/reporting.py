"""Shared plotting helpers for figure generation and reporting."""

from __future__ import annotations

import argparse
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from . import reporting_styles
from .reporting_styles import PLOT_STYLE_BASE, PLOT_STYLE_PRESETS

__all__ = [
    "axis_numeric_y_values",
    "blend_color_toward_white",
    "combine_shades",
    "create_figure_with_side_legend",
    "display_sampling_label_for_curve",
    "flatten_axes",
    "finalize_side_legend_figure",
    "granularity_sort_key",
    "generate_figures",
    "metric_row_limits_for_panel_specs",
    "panel_spec_label",
    "panel_sampling_matches",
    "padded_limits",
    "place_legend_on_right",
    "resolve_plot_style",
    "resolve_series_alias",
    "safe_filename_fragment",
    "to_float",
    "to_float_or_none",
    "main",
]


def resolve_plot_style(style_name: str) -> dict[str, Any]:
    merged = dict(PLOT_STYLE_BASE)
    preset = PLOT_STYLE_PRESETS.get(style_name, {})
    for key, value in preset.items():
        if isinstance(value, dict):
            nested = dict(merged.get(key, {}))
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def resolve_series_alias(series_key: str, style_config: dict[str, Any]) -> str:
    return str(
        style_config.get("series_aliases", {}).get(
            series_key,
            style_config.get("curve_aliases", {}).get(series_key, series_key),
        )
    )


def safe_filename_fragment(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return normalized or "unknown"


def flatten_axes(axes) -> list[Any]:
    if hasattr(axes, "flat"):
        return list(axes.flat)
    return [axes]


def metric_row_limits_for_panel_specs(
    axes_list: list[Any],
    panel_specs: list[tuple[str, str, str | None]],
    column_count: int,
) -> list[tuple[float, float] | None]:
    row_limits: list[tuple[float, float] | None] = []
    row_count = math.ceil(len(panel_specs) / column_count)
    for row_index in range(row_count):
        start = row_index * column_count
        end = min(start + column_count, len(axes_list))
        row_axes = axes_list[start:end]
        row_values: list[float] = []
        for axis in row_axes:
            row_values.extend(axis_numeric_y_values(axis))
        if not row_values:
            row_limits.append(None)
            continue
        row_min = min(row_values)
        row_max = max(row_values)
        row_limits.append(padded_limits(row_min, row_max))
    return row_limits


def axis_numeric_y_values(axis) -> list[float]:
    values: list[float] = []
    for line in axis.get_lines():
        for y_value in line.get_ydata():
            y = to_float_or_none(y_value)
            if y is not None and math.isfinite(y):
                values.append(y)
    for collection in axis.collections:
        if not hasattr(collection, "get_offsets"):
            continue
        offsets = collection.get_offsets()
        for _, y_value in offsets:
            y = to_float_or_none(y_value)
            if y is not None and math.isfinite(y):
                values.append(y)
    return values


def padded_limits(min_value: float, max_value: float) -> tuple[float, float]:
    if min_value == max_value:
        if min_value == 0.0:
            return (-1.0, 1.0)
        padding = abs(min_value) * 0.05
        return (min_value - padding, max_value + padding)
    padding = (max_value - min_value) * 0.08
    return (min_value - padding, max_value + padding)


def panel_spec_label(
    sampling_mode: str,
    variant_label: str,
    sampling_label: str | None,
) -> str:
    parts = [sampling_mode, variant_label]
    if sampling_label is not None:
        parts.append(sampling_label)
    return " / ".join(parts)


def panel_sampling_matches(
    actual_sampling_label: str | None,
    expected_sampling_label: str | None,
) -> bool:
    if expected_sampling_label is None:
        return True
    if expected_sampling_label == "global":
        return actual_sampling_label in (None, "global")
    return actual_sampling_label == expected_sampling_label


def combine_shades(*shades: float) -> float:
    combined = 0.0
    for shade in shades:
        shade = min(max(shade, 0.0), 1.0)
        combined = 1.0 - (1.0 - combined) * (1.0 - shade)
    return combined


def blend_color_toward_white(color: str, shade: float) -> tuple[float, float, float]:
    rgb = to_rgb(color)
    shade = min(max(shade, 0.0), 1.0)
    return tuple(component + (1.0 - component) * shade for component in rgb)


def create_figure_with_side_legend(
    plot_width: float,
    plot_height: float,
    legend_width: float,
):
    figure = plt.figure(figsize=(plot_width + legend_width, plot_height))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=[plot_width, legend_width],
        wspace=0.08,
    )
    axis = figure.add_subplot(grid[0])
    legend_axis = figure.add_subplot(grid[1])
    legend_axis.set_axis_off()
    legend_axis.set_in_layout(False)
    return figure, axis, legend_axis


def place_legend_on_right(legend_axis, axis) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if not handles:
        return

    legend_axis.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
    )


def finalize_side_legend_figure(figure, *, trace_description: str) -> None:
    # GridSpec + a hidden legend axis triggers tight_layout warnings in Matplotlib.
    # Use explicit margins instead; bbox_inches='tight' handles the final crop.
    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.88 if trace_description else 0.92,
        bottom=0.14 if trace_description else 0.11,
    )


def granularity_sort_key(value: str) -> tuple[int, str]:
    order = {"s": 0, "m": 1, "l": 2, "xl": 3}
    return (order.get(value, len(order)), value)


def display_sampling_label_for_curve(sampling_label: str | None) -> str | None:
    if sampling_label is None:
        return None
    if sampling_label == "global":
        return None
    if sampling_label == "per_block":
        return "per_block sampling"
    if sampling_label == "adaptive_per_block_thompson":
        return "adaptive per-block thompson"
    if sampling_label == "adaptive_per_block_ucb":
        return "adaptive per-block ucb"
    return sampling_label


def to_float(value: Any) -> float:
    return float(value)


def to_float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def generate_figures(
    input_root: str | Path,
    output_dir: str | Path,
    refresh_counts: bool = True,
    dpi: int = 300,
) -> list[Path]:
    from . import reporting_io
    from .reporting_impl import (
        plot_consistency_results,
        plot_metric_over_steps,
        plot_metric_vs_size,
        plot_metric_vs_size_split_comparison,
        plot_validation_loss_over_tokens_by_experiment,
        plot_validation_loss_over_tokens_by_granularity_comparison,
        resolve_figure_row_filter,
        write_medium_trend_report,
    )
    from .validation import aggregate_scaling_summary

    input_root = Path(input_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: list[Path] = []
    scaling_rows = reporting_io.read_csv_artifacts(input_root, "scaling_results.csv")
    scaling_rows = reporting_io.enrich_scaling_metadata_from_run_config(
        input_root,
        scaling_rows,
    )
    if refresh_counts:
        scaling_rows = reporting_io.refresh_scaling_parameter_counts(
            input_root,
            scaling_rows,
        )

    task_result_rows = reporting_io.read_csv_artifacts(input_root, "task_results.csv")
    consistency_rows = reporting_io.read_csv_artifacts(
        input_root,
        "consistency_results.csv",
    )

    if scaling_rows and task_result_rows:
        scaling_rows = aggregate_scaling_summary(scaling_rows, task_result_rows)

    if scaling_rows:
        figure_paths.extend(
            plot_metric_vs_size(
                scaling_rows,
                metric_name="loss",
                ylabel="Loss",
                output_path=output_dir / "loss_vs_size.png",
                panel_specs=reporting_styles.SIZE_PLOT_PANELS_WITH_SAMPLING,
                dpi=dpi,
            )
        )
        for figure_spec in reporting_styles.PPL_VS_SIZE_FIGURE_SPECS:
            figure_paths.extend(
                plot_metric_vs_size(
                    scaling_rows,
                    metric_name="perplexity",
                    ylabel="Perplexity",
                    output_path=output_dir / figure_spec["output_name"],
                    panel_specs=figure_spec["panel_specs"],
                    row_filter=resolve_figure_row_filter(
                        figure_spec["row_filter_name"]
                    ),
                    figure_title=figure_spec["figure_title"],
                    style=figure_spec["style"],
                    figure_alias=figure_spec["figure_alias"],
                    dpi=dpi,
                )
            )
        figure_paths.append(
            plot_metric_vs_size_split_comparison(
                scaling_rows,
                metric_name="perplexity",
                ylabel="Perplexity",
                output_path=output_dir
                / reporting_styles.PPL_VS_SIZE_SPLIT_FIGURE_SPEC["output_name"],
                figure_title=reporting_styles.PPL_VS_SIZE_SPLIT_FIGURE_SPEC["figure_title"],
                style=reporting_styles.PPL_VS_SIZE_SPLIT_FIGURE_SPEC["style"],
                left_panel_spec=reporting_styles.PPL_VS_SIZE_SPLIT_FIGURE_SPEC["left"],
                right_panel_spec=reporting_styles.PPL_VS_SIZE_SPLIT_FIGURE_SPEC["right"],
                dpi=dpi,
            )
        )
        if any(row.get("average_downstream_accuracy") for row in scaling_rows):
            figure_paths.extend(
                plot_metric_vs_size(
                    scaling_rows,
                    metric_name="average_downstream_accuracy",
                    ylabel="Average downstream accuracy",
                    output_path=output_dir / "accuracy_vs_size.png",
                    dpi=dpi,
                )
            )
        figure_paths.append(
            write_medium_trend_report(
                scaling_rows,
                output_dir / "medium_trend_report.md",
            )
        )

    if scaling_rows:
        metrics_rows = reporting_io.read_csv_artifacts_filtered(
            input_root,
            "metrics.csv",
            row_filter=reporting_io.validation_split_filter,
        )
        metrics_rows = reporting_io.enrich_metrics_metadata_from_run_config(
            input_root,
            metrics_rows,
        )
        figure_paths.extend(
            plot_validation_loss_over_tokens_by_experiment(
                metrics_rows,
                output_dir,
                dpi=dpi,
            )
        )
        figure_paths.extend(
            plot_validation_loss_over_tokens_by_granularity_comparison(
                metrics_rows,
                output_dir,
                dpi=dpi,
            )
        )
    else:
        metrics_rows = reporting_io.read_csv_artifacts(input_root, "metrics.csv")
        metrics_rows = reporting_io.enrich_metrics_metadata_from_run_config(
            input_root,
            metrics_rows,
        )
        validation_metrics_rows = [
            row for row in metrics_rows if reporting_io.validation_split_filter(row)
        ]
        if validation_metrics_rows:
            figure_paths.extend(
                plot_validation_loss_over_tokens_by_experiment(
                    validation_metrics_rows,
                    output_dir,
                    dpi=dpi,
                )
            )
            figure_paths.extend(
                plot_validation_loss_over_tokens_by_granularity_comparison(
                    validation_metrics_rows,
                    output_dir,
                    dpi=dpi,
                )
            )
        figure_paths.append(
            plot_metric_over_steps(
                metrics_rows,
                metric_name="perplexity",
                ylabel="Perplexity",
                output_path=output_dir / "ppl_over_steps.png",
                dpi=dpi,
            )
        )

    if consistency_rows:
        figure_paths.append(
            plot_consistency_results(
                consistency_rows,
                output_dir / "consistency_vs_size.png",
                dpi=dpi,
            )
        )

    return figure_paths


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs", help="Root containing run CSV artifacts.")
    parser.add_argument("--output", default="outputs/figures", help="Figure output directory.")
    parser.add_argument(
        "--no-refresh-counts",
        action="store_true",
        help=(
            "Use parameter counts already stored in scaling_results.csv instead "
            "of recomputing counts from each run's config.json."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI to use when saving figures.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    figure_paths = generate_figures(
        args.input,
        args.output,
        refresh_counts=not args.no_refresh_counts,
        dpi=args.dpi,
    )
    for path in figure_paths:
        print(path)
