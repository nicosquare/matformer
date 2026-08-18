"""Shared plotting helpers for figure generation and reporting."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb
from matplotlib.collections import PolyCollection

from . import reporting_styles
from .reporting_styles import PLOT_STYLE_BASE, PLOT_STYLE_PRESETS

__all__ = [
    "axis_numeric_y_values",
    "blend_color_toward_white",
    "combine_shades",
    "controller_selection_share_filename",
    "controller_selection_frequency_filename",
    "controller_timeline_filename",
    "panelgrad_exposure_share_filename",
    "panelgrad_refresh_diagnostics_filename",
    "create_figure_with_side_legend",
    "display_sampling_label_for_curve",
    "flatten_axes",
    "finalize_side_legend_figure",
    "granularity_sort_key",
    "generate_figures",
    "generate_global_sampling_policy_figures",
    "global_sampling_bin_series",
    "metric_row_limits_for_panel_specs",
    "panel_spec_label",
    "panel_sampling_matches",
    "padded_limits",
    "place_legend_on_right",
    "plot_selected_granularity_over_tokens",
    "plot_selected_granularity_share_over_tokens",
    "plot_granularity_selection_frequency_over_tokens",
    "plot_global_sampling_cumulative_comparison",
    "plot_global_sampling_exposure",
    "plot_global_sampling_exposure_comparison",
    "plot_global_sampling_zoom",
    "plot_panelgrad_cumulative_exposure_share",
    "plot_panelgrad_refresh_diagnostics",
    "resolve_plot_style",
    "resolve_series_alias",
    "safe_filename_fragment",
    "scaling_curve_sampling_label",
    "to_float",
    "to_float_or_none",
    "main",
]


BAYESIAN_CONTROLLER_METHOD_FAMILY = "bayesian_gaussian_linear_thompson"
PANELGRAD_METHOD_FAMILIES = {
    "panelgrad_gradient_rms",
    "panelgrad_gradient_l2",
}


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
    if sampling_label == "panelgrad_global":
        return "PanelGrad global"
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
    if (
        str(method_family or "").strip().lower() in PANELGRAD_METHOD_FAMILIES
        and method_version not in (None, "")
        and strategy == "panelgrad"
        and scope == "global"
    ):
        return "panelgrad_global"
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
    contiguous_segments: list[list[Any]] = []
    for window in timeline.windows:
        if (
            not contiguous_segments
            or window.start_tokens != contiguous_segments[-1][-1].end_tokens
        ):
            contiguous_segments.append([window])
        else:
            contiguous_segments[-1].append(window)
    for segment in contiguous_segments:
        x_edges = [segment[0].start_tokens] + [
            window.end_tokens for window in segment
        ]
        for row_index in range(timeline.block_count):
            values = [
                granularity_indices[window.block_granularities[row_index]]
                for window in segment
            ]
            axis.pcolormesh(
                x_edges,
                [row_index + 0.5, row_index + 1.5],
                [values],
                cmap=color_map,
                norm=color_norm,
                shading="flat",
                edgecolors="none",
                linewidth=0,
                antialiased=False,
                rasterized=True,
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
    """Render the exact selected share in every controller decision window.

    Global actions are binary 0/1 traces. Per-block actions are the fraction
    of transformer blocks assigned to each granularity in that window.
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
        if timeline.scope == "global":
            x_values = [
                (window.start_tokens + window.end_tokens) / 2
                for window in timeline.windows
            ]
            y_values = [
                1.0 if window.block_granularities[0] == granularity else 0.0
                for window in timeline.windows
            ]
            selected_count = int(sum(y_values))
            total_count = len(y_values)
            selected_fraction = selected_count / total_count if total_count else 0.0
            axis.plot(
                x_values,
                y_values,
                color=color,
                linestyle="None",
                marker=".",
                markersize=1.5,
                alpha=0.7,
                rasterized=True,
            )
            axis.set_title(
                f"{granularity} — {selected_count:,}/{total_count:,} steps "
                f"({selected_fraction:.1%})",
                fontsize=11,
                pad=4,
            )
            axis.set_ylim(-0.05, 1.05)
            axis.set_yticks([0.0, 0.5, 1.0])
            axis.set_ylabel("Selected\n(0 or 1)")
            axis.grid(True, axis="both", alpha=0.25)
            axis.set_axisbelow(True)
            continue

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
    interpretation = (
        "Exact per-step indicator (0/1); dense switches may alias visually"
        if timeline.scope == "global"
        else "Exact selected block fraction per controller window"
    )
    figure.suptitle(
        f"Selected granularity share over training — {timeline.run_id}\n"
        f"{interpretation}",
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


def plot_granularity_selection_frequency_over_tokens(
    timeline,
    output_path: Path,
    dpi: int = 300,
    max_bins: int = 100,
) -> Path:
    """Render local, token-weighted granularity selection frequencies.

    Short controller histories retain one point per decision window. Dense
    histories are aggregated into equal-token bins so rapidly alternating
    global actions do not render as several apparently solid 0/1 traces.
    """

    granularities = timeline.ordered_granularities
    color_positions = (
        [0.5]
        if len(granularities) == 1
        else [index / (len(granularities) - 1) for index in range(len(granularities))]
    )
    colors = [plt.get_cmap("viridis")(position) for position in color_positions]
    x_values, shares, binned = _selection_share_series(
        timeline,
        max_bins=max_bins,
    )
    figure_height = max(3.0, 2.0 * len(granularities))
    figure, axes = plt.subplots(
        len(granularities),
        1,
        figsize=(12, figure_height),
        sharex=True,
    )
    axes = [axes] if len(granularities) == 1 else list(axes)
    for axis, granularity, color in zip(axes, granularities, colors):
        axis.plot(
            x_values,
            shares[granularity],
            color=color,
            linewidth=1.8,
        )
        axis.set_ylim(-0.02, 1.02)
        axis.set_yticks([0.0, 0.5, 1.0])
        axis.set_ylabel(
            "Selection rate\nwithin token bin"
            if timeline.scope == "global"
            else "Mean selected block\nfraction within token bin"
        )
        axis.set_title(granularity, fontsize=11, pad=4)
        axis.grid(True, axis="both", alpha=0.25)
        axis.set_axisbelow(True)

    axes[-1].set_xlim(0, timeline.token_budget)
    axes[-1].set_xlabel("Total training tokens")
    axes[-1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    aggregation = (
        f"{len(x_values)} equal-token bins"
        if binned
        else "one point per controller window"
    )
    figure.suptitle(
        f"Granularity allocation over training — {timeline.run_id}\n{aggregation}",
        fontsize=15,
        y=0.995,
    )
    figure.subplots_adjust(
        left=0.14,
        right=0.98,
        top=0.92,
        bottom=0.08,
        hspace=0.42,
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _selection_share_series(timeline, *, max_bins: int = 100):
    """Return token positions and allocation shares that sum to one."""

    if max_bins <= 0:
        raise ValueError("max_bins must be positive")
    granularities = tuple(timeline.ordered_granularities)
    windows = tuple(timeline.windows)
    if not windows:
        return [], {label: [] for label in granularities}, False

    if len(windows) <= max_bins:
        x_values = [
            (window.start_tokens + window.end_tokens) / 2 for window in windows
        ]
        shares = {
            label: [
                window.block_granularities.count(label)
                / len(window.block_granularities)
                for window in windows
            ]
            for label in granularities
        }
        return x_values, shares, False

    observed_start = min(window.start_tokens for window in windows)
    observed_end = max(window.end_tokens for window in windows)
    bin_width = (observed_end - observed_start) / max_bins
    if bin_width <= 0:
        raise ValueError("controller timeline must cover a positive token interval")
    bin_edges = [observed_start + index * bin_width for index in range(max_bins + 1)]
    weighted = {label: [0.0] * max_bins for label in granularities}
    coverage = [0.0] * max_bins

    for window in windows:
        start_index = max(
            0,
            min(max_bins - 1, int((window.start_tokens - observed_start) / bin_width)),
        )
        end_index = max(
            start_index,
            min(
                max_bins - 1,
                int(
                    (
                        math.nextafter(float(window.end_tokens), -math.inf)
                        - observed_start
                    )
                    / bin_width
                ),
            ),
        )
        window_shares = {
            label: window.block_granularities.count(label)
            / len(window.block_granularities)
            for label in granularities
        }
        for bin_index in range(start_index, end_index + 1):
            overlap = max(
                0.0,
                min(window.end_tokens, bin_edges[bin_index + 1])
                - max(window.start_tokens, bin_edges[bin_index]),
            )
            if overlap <= 0:
                continue
            coverage[bin_index] += overlap
            for label in granularities:
                weighted[label][bin_index] += overlap * window_shares[label]

    populated = [index for index, value in enumerate(coverage) if value > 0]
    x_values = [
        (bin_edges[index] + bin_edges[index + 1]) / 2 for index in populated
    ]
    shares = {
        label: [weighted[label][index] / coverage[index] for index in populated]
        for label in granularities
    }
    return x_values, shares, True


def plot_panelgrad_cumulative_exposure_share(
    history,
    output_path: Path,
    dpi: int = 300,
) -> Path:
    """Render cumulative committed-action shares for one PanelGrad run."""

    if not history.actions:
        raise ValueError("PanelGrad exposure plot requires at least one action")
    granularities = history.ordered_granularities
    color_positions = (
        [0.5]
        if len(granularities) == 1
        else [index / (len(granularities) - 1) for index in range(len(granularities))]
    )
    colors = [plt.get_cmap("viridis")(position) for position in color_positions]
    counts = {label: 0 for label in granularities}
    tokens: list[int] = []
    shares = {label: [] for label in granularities}
    for draw_count, action in enumerate(history.actions, start=1):
        counts[action.granularity] += 1
        tokens.append(action.end_tokens)
        for label in granularities:
            shares[label].append(counts[label] / draw_count)

    figure, axis = plt.subplots(figsize=(12, 5.5))
    for label, color in zip(granularities, colors):
        axis.plot(
            tokens,
            shares[label],
            color=color,
            label=label,
            linewidth=1.8,
            drawstyle="steps-post",
        )
    axis.set_xlim(0, history.token_budget)
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("Total training tokens")
    axis.set_ylabel("Cumulative action share")
    axis.set_title(f"PanelGrad cumulative exposure — {history.run_id}")
    axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axis.grid(True, alpha=0.25)
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_panelgrad_refresh_diagnostics(
    history,
    output_path: Path,
    dpi: int = 300,
) -> Path:
    """Render PanelGrad score, distribution, entropy, and refresh-cost traces."""

    if not history.refreshes:
        raise ValueError("PanelGrad diagnostics require at least one refresh")
    granularities = history.ordered_granularities
    color_positions = (
        [0.5]
        if len(granularities) == 1
        else [index / (len(granularities) - 1) for index in range(len(granularities))]
    )
    colors = [plt.get_cmap("viridis")(position) for position in color_positions]
    tokens = [refresh.boundary_tokens for refresh in history.refreshes]

    figure, axes = plt.subplots(5, 1, figsize=(13, 17), sharex=True)
    for granularity_index, (label, color) in enumerate(zip(granularities, colors)):
        axes[0].plot(
            tokens,
            [
                refresh.importance_scores[granularity_index]
                for refresh in history.refreshes
            ],
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3,
            label=label,
        )
        axes[1].plot(
            tokens,
            [refresh.p[granularity_index] for refresh in history.refreshes],
            color=color,
            linewidth=1.8,
            label=f"p({label})",
        )
        axes[1].plot(
            tokens,
            [refresh.q[granularity_index] for refresh in history.refreshes],
            color=color,
            linewidth=1.1,
            linestyle="--",
            alpha=0.75,
            label=f"q({label})",
        )

    score_label = (
        "Gradient RMS"
        if history.importance_metric == "gradient_rms"
        else "Gradient L2 norm"
    )
    axes[0].set_ylabel(score_label)
    axes[0].set_title(f"Whole-active-subnetwork {score_label.lower()}")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("Probability")
    axes[1].set_title("Categorical distributions (solid p, dashed q)")
    axes[1].legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        ncol=1,
    )

    axes[2].plot(
        tokens,
        [refresh.active_epsilon for refresh in history.refreshes],
        color="tab:cyan",
        linewidth=1.8,
        marker="o",
        markersize=3,
        drawstyle="steps-post",
        label="active epsilon",
    )
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_ylabel("Epsilon")
    axes[2].set_title("Refresh-boundary exploration schedule")
    axes[2].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    axes[3].plot(
        tokens,
        [refresh.entropy for refresh in history.refreshes],
        color="tab:purple",
        linewidth=1.8,
        label="entropy",
    )
    axes[3].plot(
        tokens,
        [refresh.min_probability for refresh in history.refreshes],
        color="tab:blue",
        linestyle="--",
        label="min p",
    )
    axes[3].plot(
        tokens,
        [refresh.max_probability for refresh in history.refreshes],
        color="tab:orange",
        linestyle="--",
        label="max p",
    )
    axes[3].set_ylabel("Value")
    axes[3].set_title("Distribution entropy and probability extrema")
    axes[3].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    cumulative_duration = []
    duration_total = 0.0
    for refresh in history.refreshes:
        duration_total += refresh.duration_seconds
        cumulative_duration.append(duration_total)
    duration_line = axes[4].plot(
        tokens,
        [refresh.duration_seconds for refresh in history.refreshes],
        color="tab:green",
        linewidth=1.6,
        marker="o",
        markersize=3,
        label="refresh duration",
    )
    cumulative_axis = axes[4].twinx()
    cumulative_line = cumulative_axis.plot(
        tokens,
        cumulative_duration,
        color="tab:red",
        linewidth=1.8,
        label="cumulative duration",
    )
    axes[4].set_ylabel("Refresh seconds", color="tab:green")
    cumulative_axis.set_ylabel("Cumulative seconds", color="tab:red")
    axes[4].tick_params(axis="y", labelcolor="tab:green")
    cumulative_axis.tick_params(axis="y", labelcolor="tab:red")
    axes[4].set_title(
        "Measurement cost — "
        f"{sum(refresh.backward_evaluations for refresh in history.refreshes):,} "
        "backward evaluations, "
        f"{sum(refresh.controller_target_tokens for refresh in history.refreshes):,} "
        "controller target tokens"
    )
    cost_lines = duration_line + cumulative_line
    axes[4].legend(
        cost_lines,
        [line.get_label() for line in cost_lines],
        loc="center left",
        bbox_to_anchor=(1.11, 0.5),
        frameon=False,
    )
    axes[4].set_xlim(0, history.token_budget)
    axes[4].set_xlabel("Total training tokens")
    axes[4].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.suptitle(f"PanelGrad refresh diagnostics — {history.run_id}", fontsize=15)
    figure.subplots_adjust(
        left=0.09,
        right=0.76,
        top=0.94,
        bottom=0.06,
        hspace=0.34,
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def global_sampling_bin_series(history, *, bin_steps: int = 50):
    """Return exact committed-action fractions in fixed optimizer-step bins."""

    if bin_steps <= 0:
        raise ValueError("bin_steps must be positive")
    granularities = tuple(history.ordered_granularities)
    x_values: list[float] = []
    shares = {label: [] for label in granularities}
    bin_sizes: list[int] = []
    actions = tuple(history.actions)
    for start in range(0, len(actions), bin_steps):
        chunk = actions[start : start + bin_steps]
        if not chunk:
            continue
        counts = {label: 0 for label in granularities}
        for action in chunk:
            counts[action.granularity] += 1
        x_values.append((chunk[0].start_tokens + chunk[-1].end_tokens) / 2)
        bin_sizes.append(len(chunk))
        for label in granularities:
            shares[label].append(counts[label] / len(chunk))
    return x_values, shares, bin_sizes


def _sampling_colors(granularities):
    positions = (
        [0.5]
        if len(granularities) == 1
        else [index / (len(granularities) - 1) for index in range(len(granularities))]
    )
    return [plt.get_cmap("viridis")(position) for position in positions]


def plot_global_sampling_exposure(
    history,
    output_path: Path,
    *,
    bin_steps: int = 50,
    dpi: int = 300,
) -> Path:
    """Plot per-granularity exposure in exact optimizer-step bins."""

    granularities = history.ordered_granularities
    x_values, shares, bin_sizes = global_sampling_bin_series(
        history, bin_steps=bin_steps
    )
    colors = _sampling_colors(granularities)
    figure, axes = plt.subplots(
        len(granularities),
        1,
        figsize=(12, max(3.0, 2.0 * len(granularities))),
        sharex=True,
    )
    axes = [axes] if len(granularities) == 1 else list(axes)
    for axis, label, color in zip(axes, granularities, colors):
        axis.plot(x_values, shares[label], color=color, linewidth=1.5)
        axis.set_ylim(-0.02, 1.02)
        axis.set_yticks([0.0, 0.5, 1.0])
        axis.set_ylabel("Step fraction")
        axis.set_title(label, fontsize=11, pad=4)
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlim(0, history.token_budget)
    axes[-1].set_xlabel("Total training tokens")
    axes[-1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    partial = bin_sizes[-1] if bin_sizes and bin_sizes[-1] != bin_steps else None
    suffix = f"; final bin has {partial} steps" if partial is not None else ""
    figure.suptitle(
        f"Global granularity exposure — {history.run_id}\n"
        f"Committed actions in {bin_steps}-optimizer-step bins{suffix}",
        fontsize=15,
        y=0.995,
    )
    figure.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.08, hspace=0.42)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_global_sampling_zoom(
    history,
    output_path: Path,
    *,
    zoom_steps: int = 250,
    dpi: int = 300,
) -> Path:
    """Show exact early and late committed actions without dense-line aliasing."""

    if zoom_steps <= 0:
        raise ValueError("zoom_steps must be positive")
    actions = tuple(history.actions)
    if not actions:
        raise ValueError("sampling zoom requires at least one action")
    granularity_index = {
        label: index for index, label in enumerate(history.ordered_granularities)
    }
    if len(actions) <= 2 * zoom_steps:
        panels = [("All committed steps", actions)]
    else:
        panels = [
            (f"First {zoom_steps} steps", actions[:zoom_steps]),
            (f"Last {zoom_steps} steps", actions[-zoom_steps:]),
        ]
    figure, axes = plt.subplots(len(panels), 1, figsize=(12, 3.2 * len(panels)))
    axes = [axes] if len(panels) == 1 else list(axes)
    colors = _sampling_colors(history.ordered_granularities)
    for axis, (title, panel_actions) in zip(axes, panels):
        for label, color in zip(history.ordered_granularities, colors):
            selected = [action for action in panel_actions if action.granularity == label]
            axis.scatter(
                [action.step for action in selected],
                [granularity_index[label]] * len(selected),
                color=color,
                marker="s",
                s=12,
                label=label,
                rasterized=True,
            )
        axis.set_yticks(
            range(len(history.ordered_granularities)), history.ordered_granularities
        )
        axis.set_xlim(panel_actions[0].step - 0.5, panel_actions[-1].step + 0.5)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    axes[-1].set_xlabel("Completed optimizer step")
    figure.suptitle(
        f"Exact global sampling decisions — {history.run_id}", fontsize=15, y=0.995
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _sampling_history_labels(histories):
    counts: dict[str, int] = {}
    for history in histories:
        counts[history.policy_label] = counts.get(history.policy_label, 0) + 1
    labels = []
    for history in histories:
        label = history.policy_label
        if counts[history.policy_label] > 1:
            label += f" (seed {history.seed})" if history.seed is not None else f" ({history.run_id})"
        labels.append(label)
    return labels


def plot_global_sampling_exposure_comparison(
    histories,
    output_path: Path,
    *,
    bin_steps: int = 50,
    dpi: int = 300,
) -> Path:
    """Compare local exposure trajectories on one common step resolution."""

    histories = tuple(histories)
    granularities = histories[0].ordered_granularities
    labels = _sampling_history_labels(histories)
    figure, axes = plt.subplots(
        len(granularities), 1, figsize=(13, max(4.0, 2.2 * len(granularities))), sharex=True
    )
    axes = [axes] if len(granularities) == 1 else list(axes)
    for history, policy_label in zip(histories, labels):
        x_values, shares, _ = global_sampling_bin_series(history, bin_steps=bin_steps)
        for axis, granularity in zip(axes, granularities):
            axis.plot(x_values, shares[granularity], linewidth=1.35, label=policy_label)
    for axis, granularity in zip(axes, granularities):
        axis.set_ylim(-0.02, 1.02)
        axis.set_ylabel("Step fraction")
        axis.set_title(granularity, fontsize=11)
        axis.grid(True, alpha=0.25)
    axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    axes[-1].set_xlim(0, max(history.token_budget for history in histories))
    axes[-1].set_xlabel("Total training tokens")
    axes[-1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    figure.suptitle(
        f"Global sampling-policy exposure comparison\n{bin_steps}-optimizer-step bins",
        fontsize=15,
        y=0.995,
    )
    figure.subplots_adjust(left=0.11, right=0.79, top=0.94, bottom=0.07, hspace=0.38)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_global_sampling_cumulative_comparison(
    histories,
    output_path: Path,
    *,
    bin_steps: int = 50,
    dpi: int = 300,
) -> Path:
    """Compare cumulative exposure without conflating it with local behavior."""

    histories = tuple(histories)
    granularities = histories[0].ordered_granularities
    labels = _sampling_history_labels(histories)
    figure, axes = plt.subplots(
        len(granularities), 1, figsize=(13, max(4.0, 2.2 * len(granularities))), sharex=True
    )
    axes = [axes] if len(granularities) == 1 else list(axes)
    for history, policy_label in zip(histories, labels):
        counts = {label: 0 for label in granularities}
        xs: list[int] = []
        values = {label: [] for label in granularities}
        for index, action in enumerate(history.actions, start=1):
            counts[action.granularity] += 1
            if index % bin_steps and index != len(history.actions):
                continue
            xs.append(action.end_tokens)
            for label in granularities:
                values[label].append(counts[label] / index)
        for axis, granularity in zip(axes, granularities):
            axis.plot(xs, values[granularity], linewidth=1.35, label=policy_label)
    for axis, granularity in zip(axes, granularities):
        axis.set_ylim(-0.02, 1.02)
        axis.set_ylabel("Cumulative fraction")
        axis.set_title(granularity, fontsize=11)
        axis.grid(True, alpha=0.25)
    axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    axes[-1].set_xlim(0, max(history.token_budget for history in histories))
    axes[-1].set_xlabel("Total training tokens")
    axes[-1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    figure.suptitle("Cumulative global granularity exposure", fontsize=15, y=0.995)
    figure.subplots_adjust(left=0.11, right=0.79, top=0.95, bottom=0.07, hspace=0.38)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _write_global_sampling_summary(histories, output_path: Path) -> Path:
    granularities = histories[0].ordered_granularities
    fieldnames = [
        "run_id", "policy", "seed", "completed_steps", "policy_decisions",
        "decisions_per_step", "action_transitions",
    ]
    for label in granularities:
        fieldnames.extend([f"{label}_steps", f"{label}_fraction"])
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for history in histories:
            counts = {label: 0 for label in granularities}
            transitions = 0
            previous = None
            for action in history.actions:
                counts[action.granularity] += 1
                transitions += int(previous is not None and action.granularity != previous)
                previous = action.granularity
            row = {
                "run_id": history.run_id,
                "policy": history.policy_label,
                "seed": "" if history.seed is None else history.seed,
                "completed_steps": len(history.actions),
                "policy_decisions": history.decision_count,
                "decisions_per_step": history.decision_count / len(history.actions),
                "action_transitions": transitions,
            }
            for label in granularities:
                row[f"{label}_steps"] = counts[label]
                row[f"{label}_fraction"] = counts[label] / len(history.actions)
            writer.writerow(row)
    return output_path


def controller_timeline_filename(run_id: str) -> str:
    safe_run_id = re.sub(r"[^A-Za-z0-9._-]+", "_", run_id).strip("._-")
    return f"selected_granularity_over_tokens_{safe_run_id or 'unknown'}.png"


def controller_selection_share_filename(run_id: str) -> str:
    safe_run_id = re.sub(r"[^A-Za-z0-9._-]+", "_", run_id).strip("._-")
    return f"selected_granularity_share_over_tokens_{safe_run_id or 'unknown'}.png"


def controller_selection_frequency_filename(run_id: str) -> str:
    safe_run_id = re.sub(r"[^A-Za-z0-9._-]+", "_", run_id).strip("._-")
    return f"granularity_selection_frequency_over_tokens_{safe_run_id or 'unknown'}.png"


def global_sampling_zoom_filename(run_id: str) -> str:
    safe_run_id = re.sub(r"[^A-Za-z0-9._-]+", "_", run_id).strip("._-")
    return f"selected_granularity_zoom_{safe_run_id or 'unknown'}.png"


def panelgrad_exposure_share_filename(run_id: str) -> str:
    safe_run_id = re.sub(r"[^A-Za-z0-9._-]+", "_", run_id).strip("._-")
    return f"panelgrad_cumulative_exposure_{safe_run_id or 'unknown'}.png"


def panelgrad_refresh_diagnostics_filename(
    run_id: str,
    importance_metric: str | None = None,
) -> str:
    safe_run_id = re.sub(r"[^A-Za-z0-9._-]+", "_", run_id).strip("._-")
    metric_suffix = (
        ""
        if importance_metric in (None, "")
        else f"_{safe_filename_fragment(str(importance_metric))}"
    )
    return (
        f"panelgrad_refresh_diagnostics_{safe_run_id or 'unknown'}"
        f"{metric_suffix}.png"
    )


def generate_global_sampling_policy_figures(
    input_root: str | Path,
    output_dir: str | Path,
    *,
    dpi: int = 300,
    variants: list[str] | tuple[str, ...] | None = None,
    corrections: list[str] | tuple[str, ...] | None = None,
    sampling_bin_steps: int = 50,
    sampling_zoom_steps: int = 250,
) -> list[Path]:
    """Generate action-grounded views shared by all global policies."""

    from . import reporting_io
    from .reporting_impl import filter_plot_rows

    if sampling_bin_steps <= 0:
        raise ValueError("sampling_bin_steps must be positive")
    if sampling_zoom_steps <= 0:
        raise ValueError("sampling_zoom_steps must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    histories = []
    for history in reporting_io.iter_global_sampling_histories(input_root):
        keep = filter_plot_rows(
            [{
                "model_variant": history.model_variant,
                "correction_mode": history.correction_mode,
                "membership_correction": history.membership_correction,
            }],
            variants=variants,
            corrections=corrections,
        )
        if keep:
            histories.append(history)

    paths: list[Path] = []
    for history in histories:
        timeline = reporting_io.global_sampling_history_as_timeline(history)
        paths.append(
            plot_selected_granularity_over_tokens(
                timeline,
                output_dir / controller_timeline_filename(history.run_id),
                dpi=dpi,
            )
        )
        paths.append(
            plot_global_sampling_exposure(
                history,
                output_dir / controller_selection_share_filename(history.run_id),
                bin_steps=sampling_bin_steps,
                dpi=dpi,
            )
        )
        paths.append(
            plot_global_sampling_zoom(
                history,
                output_dir / global_sampling_zoom_filename(history.run_id),
                zoom_steps=sampling_zoom_steps,
                dpi=dpi,
            )
        )
        # These names came from incompatible token-bin and PanelGrad-only views.
        # Removing them prevents stale files from being mistaken for current output.
        for stale_name in (
            controller_selection_frequency_filename(history.run_id),
            panelgrad_exposure_share_filename(history.run_id),
        ):
            (output_dir / stale_name).unlink(missing_ok=True)

    groups: dict[str, list[Any]] = {}
    for history in histories:
        groups.setdefault(history.comparison_key, []).append(history)
    for comparison_key, group in sorted(groups.items()):
        if len({history.policy_identity for history in group}) < 2:
            continue
        granularities = {history.ordered_granularities for history in group}
        if len(granularities) != 1:
            continue
        group.sort(key=lambda item: (item.policy_identity, item.seed or -1, item.run_id))
        first = group[0]
        descriptor = "__".join(
            [
                safe_filename_fragment(first.model_variant or "unknown"),
                safe_filename_fragment(first.correction_mode or "unknown"),
                comparison_key[:8],
            ]
        )
        paths.append(
            plot_global_sampling_exposure_comparison(
                group,
                output_dir / f"global_sampling_exposure_comparison__{descriptor}.png",
                bin_steps=sampling_bin_steps,
                dpi=dpi,
            )
        )
        paths.append(
            plot_global_sampling_cumulative_comparison(
                group,
                output_dir / f"global_sampling_cumulative_comparison__{descriptor}.png",
                bin_steps=sampling_bin_steps,
                dpi=dpi,
            )
        )
        paths.append(
            _write_global_sampling_summary(
                group,
                output_dir / f"global_sampling_policy_summary__{descriptor}.csv",
            )
        )
    return paths


def generate_figures(
    input_root: str | Path,
    output_dir: str | Path,
    refresh_counts: bool = True,
    dpi: int = 300,
    validation_loss_log_y: bool = False,
    include_incomplete_validation_traces: bool = False,
    variants: list[str] | tuple[str, ...] | None = None,
    corrections: list[str] | tuple[str, ...] | None = None,
    sampling_bin_steps: int = 50,
    sampling_zoom_steps: int = 250,
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

    figure_paths.extend(
        generate_global_sampling_policy_figures(
            input_root,
            output_dir,
            dpi=dpi,
            variants=variants,
            corrections=corrections,
            sampling_bin_steps=sampling_bin_steps,
            sampling_zoom_steps=sampling_zoom_steps,
        )
    )

    for timeline in reporting_io.iter_controller_granularity_timelines(input_root):
        if timeline.scope == "global":
            continue
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
        figure_paths.append(
            plot_granularity_selection_frequency_over_tokens(
                timeline,
                output_dir / controller_selection_frequency_filename(timeline.run_id),
                dpi=dpi,
            )
        )

    for history in reporting_io.iter_panelgrad_histories(input_root):
        history_rows = filter_plot_rows(
            [
                {
                    "model_variant": history.model_variant,
                    "correction_mode": history.correction_mode,
                    "membership_correction": history.membership_correction,
                }
            ],
            variants=variants,
            corrections=corrections,
        )
        if not history_rows:
            continue
        if history.refreshes:
            figure_paths.append(
                plot_panelgrad_refresh_diagnostics(
                    history,
                    output_dir
                    / panelgrad_refresh_diagnostics_filename(
                        history.run_id, history.importance_metric
                    ),
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
    parser.add_argument(
        "--sampling-bin-steps",
        type=int,
        default=50,
        help="Optimizer steps per local sampling-policy exposure bin (default: 50).",
    )
    parser.add_argument(
        "--sampling-zoom-steps",
        type=int,
        default=250,
        help="Exact actions shown at each end of sampling zoom plots (default: 250).",
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
        sampling_bin_steps=args.sampling_bin_steps,
        sampling_zoom_steps=args.sampling_zoom_steps,
    )
    for path in figure_paths:
        print(path)
