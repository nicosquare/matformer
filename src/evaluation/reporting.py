"""Shared plotting helpers for figure generation and reporting."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle

from . import reporting_styles
from .reporting_styles import PLOT_STYLE_BASE, PLOT_STYLE_PRESETS

__all__ = [
    "axis_numeric_y_values",
    "blend_color_toward_white",
    "combine_shades",
    "controller_selection_share_filename",
    "controller_timeline_filename",
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
    "plot_selected_granularity_over_tokens",
    "plot_selected_granularity_share_over_tokens",
    "resolve_plot_style",
    "resolve_series_alias",
    "safe_filename_fragment",
    "scaling_curve_sampling_label",
    "to_float",
    "to_float_or_none",
    "main",
]


BAYESIAN_CONTROLLER_METHOD_FAMILY = "bayesian_gaussian_linear_thompson"


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
    row_count = math.ceil(len(panel_specs) / column_count)
    values: list[float] = []
    for axis in axes_list[: len(panel_specs)]:
        values.extend(axis_numeric_y_values(axis))
    shared_limits = padded_limits(min(values), max(values)) if values else None
    return [shared_limits] * row_count


def axis_numeric_y_values(axis) -> list[float]:
    values: list[float] = []
    for line in axis.get_lines():
        for y_value in line.get_ydata():
            y = to_float_or_none(y_value)
            if y is not None and math.isfinite(y):
                values.append(y)
    for collection in axis.collections:
        if isinstance(collection, PolyCollection):
            for path in collection.get_paths():
                for _, y_value in path.vertices:
                    y = to_float_or_none(y_value)
                    if y is not None and math.isfinite(y):
                        values.append(y)
            continue
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
    if expected_sampling_label == "probabilistic_global_thompson":
        return actual_sampling_label in {
            "probabilistic_global_thompson",
            "probabilistic_global_thompson_reset",
            "probabilistic_global_thompson_acquisition_only",
        }
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
    order = {
        "micro": 0,
        "s": 0,
        "small": 1,
        "m": 1,
        "medium": 2,
        "l": 2,
        "large": 3,
        "xl": 3,
        "full": 4,
    }
    return (order.get(value, len(order)), value)


def display_sampling_label_for_curve(sampling_label: str | None) -> str | None:
    if sampling_label is None:
        return None
    if sampling_label == "global":
        return None
    if sampling_label == "per_block":
        return "per_block sampling"
    if sampling_label == "adaptive_per_block_thompson":
        return "legacy heuristic thompson"
    if sampling_label == "adaptive_per_block_ucb":
        return "adaptive per-block ucb"
    if sampling_label == "probabilistic_global_thompson":
        return "probabilistic global thompson"
    if sampling_label == "probabilistic_global_thompson_reset":
        return "probabilistic global thompson reset"
    if sampling_label == "probabilistic_global_thompson_acquisition_only":
        return "probabilistic global thompson acquisition-only"
    if sampling_label == "probabilistic_per_block_thompson":
        return "probabilistic per-block thompson"
    return sampling_label


def scaling_curve_sampling_label(row: dict[str, Any]) -> str | None:
    """Classify sampling from explicit Bayesian provenance when it is present."""

    sampling_mode = row.get("sampling_mode")
    if sampling_mode not in {"nested-random", "nested-all"}:
        sampling_mode = row.get("resolved_run_mode")
        if sampling_mode not in {"nested-random", "nested-all"}:
            return None

    resolved_sampling_mode = row.get("resolved_sampling_mode")
    if resolved_sampling_mode in (None, ""):
        resolved_sampling_mode = row.get("granularity_sampling_mode")
    if resolved_sampling_mode in (None, ""):
        return None
    normalized_mode = str(resolved_sampling_mode).strip().lower()

    strategy = row.get("adaptive_sampler_strategy")
    strategy = None if strategy in (None, "") else str(strategy).strip().lower()
    method_family = row.get("controller_method_family")
    method_version = row.get("controller_method_version")
    scope = row.get("controller_scope")
    scope = None if scope in (None, "") else str(scope).strip().lower()
    has_bayesian_provenance = (
        str(method_family or "").strip().lower() == BAYESIAN_CONTROLLER_METHOD_FAMILY
        and method_version not in (None, "")
        and strategy == "thompson"
        and scope in {"global", "per_block"}
    )
    if has_bayesian_provenance:
        reset_enabled = str(row.get("controller_reset_enabled", "")).strip().lower()
        if scope == "global" and reset_enabled in {"1", "true", "yes"}:
            reset_policy = (
                str(row.get("controller_reset_policy", "full_prior")).strip().lower()
            )
            if reset_policy == "acquisition_only":
                return "probabilistic_global_thompson_acquisition_only"
            return "probabilistic_global_thompson_reset"
        return (
            "probabilistic_global_thompson"
            if scope == "global"
            else "probabilistic_per_block_thompson"
        )

    if normalized_mode in {"global", "per_block"}:
        return normalized_mode
    if normalized_mode == "adaptive_per_block":
        if strategy == "thompson":
            return "adaptive_per_block_thompson"
        if strategy == "ucb":
            return "adaptive_per_block_ucb"
        return normalized_mode
    return None


def to_float(value: Any) -> float:
    return float(value)


def to_float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def plot_selected_granularity_over_tokens(
    timeline,
    output_path: Path,
    dpi: int = 300,
) -> Path:
    """Render one non-interpolated categorical controller-selection timeline."""

    granularities = timeline.ordered_granularities
    color_positions = (
        [0.5]
        if len(granularities) == 1
        else [index / (len(granularities) - 1) for index in range(len(granularities))]
    )
    color_map = ListedColormap(
        [plt.get_cmap("viridis")(position) for position in color_positions]
    )
    color_norm = BoundaryNorm(
        [index - 0.5 for index in range(len(granularities) + 1)],
        color_map.N,
    )
    granularity_indices = {label: index for index, label in enumerate(granularities)}

    figure_height = max(2.4, min(9.0, 1.6 + 0.32 * timeline.block_count))
    figure, axis = plt.subplots(figsize=(10, figure_height))
    for window in timeline.windows:
        width = window.end_tokens - window.start_tokens
        for row_index, label in enumerate(window.block_granularities, start=1):
            axis.add_patch(
                Rectangle(
                    (window.start_tokens, row_index - 0.5),
                    width,
                    1.0,
                    facecolor=color_map(color_norm(granularity_indices[label])),
                    edgecolor="white",
                    linewidth=0.25,
                )
            )

    axis.set_xlim(0, timeline.token_budget)
    axis.set_ylim(timeline.block_count + 0.5, 0.5)
    axis.set_yticks(range(1, timeline.block_count + 1), timeline.row_labels)
    axis.set_xlabel("Budget tokens seen")
    axis.set_ylabel("Transformer block" if timeline.scope == "per_block" else "Scope")
    axis.set_title(f"Selected granularity over tokens — {timeline.run_id}")
    axis.grid(axis="x", alpha=0.2)
    axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    scalar_mappable = matplotlib.cm.ScalarMappable(norm=color_norm, cmap=color_map)
    colorbar = figure.colorbar(
        scalar_mappable,
        ax=axis,
        ticks=range(len(granularities)),
        pad=0.02,
    )
    colorbar.ax.set_yticklabels(granularities)
    colorbar.set_label("Granularity")
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return output_path


def plot_selected_granularity_share_over_tokens(
    timeline,
    output_path: Path,
    dpi: int = 300,
) -> Path:
    """Render one exact step panel per selected granularity.

    Global actions resolve to binary 0/1 traces. Per-block actions use the
    fraction of transformer blocks assigned to the granularity in each window.
    Gaps between confirmed controller windows remain visually disconnected.
    """

    granularities = timeline.ordered_granularities
    color_positions = (
        [0.5]
        if len(granularities) == 1
        else [index / (len(granularities) - 1) for index in range(len(granularities))]
    )
    colors = [plt.get_cmap("viridis")(position) for position in color_positions]
    figure_height = max(3.0, 2.0 * len(granularities))
    figure, axes = plt.subplots(
        len(granularities),
        1,
        figsize=(12, figure_height),
        sharex=True,
    )
    axes = [axes] if len(granularities) == 1 else list(axes)

    for axis, granularity, color in zip(axes, granularities, colors):
        segment_xs: list[float] = []
        segment_ys: list[float] = []
        previous_end = None
        previous_share = None
        for window in timeline.windows:
            share = window.block_granularities.count(granularity) / len(
                window.block_granularities
            )
            if previous_end is None or window.start_tokens != previous_end:
                if segment_xs:
                    axis.plot(segment_xs, segment_ys, color=color, linewidth=1.8)
                segment_xs = [window.start_tokens, window.end_tokens]
                segment_ys = [share, share]
            else:
                segment_xs.extend(
                    [window.start_tokens, window.start_tokens, window.end_tokens]
                )
                segment_ys.extend([previous_share, share, share])
            previous_end = window.end_tokens
            previous_share = share
        if segment_xs:
            axis.plot(segment_xs, segment_ys, color=color, linewidth=1.8)

        axis.set_ylim(-0.05, 1.05)
        axis.set_yticks([0.0, 0.5, 1.0])
        axis.set_ylabel("Selected\nblock fraction")
        axis.set_title(granularity, fontsize=11, pad=4)
        axis.grid(True, axis="both", alpha=0.25)
        axis.set_axisbelow(True)

    axes[-1].set_xlim(0, timeline.token_budget)
    axes[-1].set_xlabel("Total training tokens")
    axes[-1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    figure.suptitle(
        f"Selected granularity share over training — {timeline.run_id}",
        fontsize=15,
        y=0.995,
    )
    figure.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.94,
        bottom=0.08,
        hspace=0.42,
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def controller_timeline_filename(run_id: str) -> str:
    safe_run_id = re.sub(r"[^A-Za-z0-9._-]+", "_", run_id).strip("._-")
    return f"selected_granularity_over_tokens_{safe_run_id or 'unknown'}.png"


def controller_selection_share_filename(run_id: str) -> str:
    safe_run_id = re.sub(r"[^A-Za-z0-9._-]+", "_", run_id).strip("._-")
    return f"selected_granularity_share_over_tokens_{safe_run_id or 'unknown'}.png"


def generate_figures(
    input_root: str | Path,
    output_dir: str | Path,
    refresh_counts: bool = True,
    dpi: int = 300,
    validation_loss_log_y: bool = False,
    include_incomplete_validation_traces: bool = False,
    variants: list[str] | tuple[str, ...] | None = None,
    corrections: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    from . import reporting_io
    from .reporting_impl import (
        filter_plot_rows,
        generate_saturation_diagnostics,
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
    scaling_rows = filter_plot_rows(
        scaling_rows,
        variants=variants,
        corrections=corrections,
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
    consistency_rows = reporting_io.enrich_metrics_metadata_from_run_config(
        input_root,
        consistency_rows,
    )
    consistency_rows = filter_plot_rows(
        consistency_rows,
        variants=variants,
        corrections=corrections,
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
        split_comparison_path = plot_metric_vs_size_split_comparison(
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
        if split_comparison_path is not None:
            figure_paths.append(split_comparison_path)
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
        metrics_rows = filter_plot_rows(
            metrics_rows,
            variants=variants,
            corrections=corrections,
        )
        figure_paths.extend(
            plot_validation_loss_over_tokens_by_experiment(
                metrics_rows,
                output_dir,
                dpi=dpi,
                validation_loss_log_y=validation_loss_log_y,
                include_incomplete_validation_traces=include_incomplete_validation_traces,
            )
        )
        figure_paths.extend(
            plot_validation_loss_over_tokens_by_granularity_comparison(
                metrics_rows,
                output_dir,
                dpi=dpi,
                validation_loss_log_y=validation_loss_log_y,
                include_incomplete_validation_traces=include_incomplete_validation_traces,
            )
        )
    else:
        metrics_rows = reporting_io.read_csv_artifacts(input_root, "metrics.csv")
        metrics_rows = reporting_io.enrich_metrics_metadata_from_run_config(
            input_root,
            metrics_rows,
        )
        metrics_rows = filter_plot_rows(
            metrics_rows,
            variants=variants,
            corrections=corrections,
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
                    validation_loss_log_y=validation_loss_log_y,
                    include_incomplete_validation_traces=include_incomplete_validation_traces,
                )
            )
            figure_paths.extend(
                plot_validation_loss_over_tokens_by_granularity_comparison(
                    validation_metrics_rows,
                    output_dir,
                    dpi=dpi,
                    validation_loss_log_y=validation_loss_log_y,
                    include_incomplete_validation_traces=include_incomplete_validation_traces,
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

    figure_paths.extend(
        generate_saturation_diagnostics(
            input_root,
            output_dir,
            dpi=dpi,
            variants=variants,
            corrections=corrections,
        )
    )

    for timeline in reporting_io.iter_controller_granularity_timelines(input_root):
        timeline_rows = filter_plot_rows(
            [
                {
                    "model_variant": timeline.model_variant,
                    "correction_mode": timeline.correction_mode,
                    "membership_correction": timeline.membership_correction,
                }
            ],
            variants=variants,
            corrections=corrections,
        )
        if not timeline_rows:
            continue
        figure_paths.append(
            plot_selected_granularity_over_tokens(
                timeline,
                output_dir / controller_timeline_filename(timeline.run_id),
                dpi=dpi,
            )
        )
        figure_paths.append(
            plot_selected_granularity_share_over_tokens(
                timeline,
                output_dir / controller_selection_share_filename(timeline.run_id),
                dpi=dpi,
            )
        )

    return figure_paths


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="outputs", help="Root containing run CSV artifacts."
    )
    parser.add_argument(
        "--output", default="outputs/figures", help="Figure output directory."
    )
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
    parser.add_argument(
        "--validation-loss-log-y",
        action="store_true",
        help="Render validation loss figures with a logarithmic y axis.",
    )
    parser.add_argument(
        "--include-incomplete-validation-traces",
        action="store_true",
        help="Include incomplete runs as separate dashed validation traces.",
    )
    parser.add_argument(
        "--variant",
        dest="variants",
        action="append",
        choices=("slicing", "concat"),
        help="Only include this model variant; repeat to include multiple variants.",
    )
    parser.add_argument(
        "--correction",
        dest="corrections",
        action="append",
        choices=("none", "gmc", "lmc"),
        help="Only include this correction mode; use 'none' for uncorrected runs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    figure_paths = generate_figures(
        args.input,
        args.output,
        refresh_counts=not args.no_refresh_counts,
        dpi=args.dpi,
        validation_loss_log_y=args.validation_loss_log_y,
        include_incomplete_validation_traces=(
            args.include_incomplete_validation_traces
        ),
        variants=args.variants,
        corrections=args.corrections,
    )
    for path in figure_paths:
        print(path)
