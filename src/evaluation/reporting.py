"""Shared plotting helpers for figure generation and reporting."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

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

