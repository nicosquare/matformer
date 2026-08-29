"""Shared plotting helpers for figure generation and reporting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb
from matplotlib.collections import PolyCollection
from src.utils.reproducibility import stable_hash

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
    "gradient_interference_cosine_heatmaps_filename",
    "gradient_interference_cosine_trajectories_filename",
    "generate_figures",
    "generate_gradient_interference_figures",
    "generate_global_sampling_policy_figures",
    "global_sampling_bin_series",
    "metric_row_limits_for_panel_specs",
    "mark_scheduler_warmup",
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
    "plot_gradient_interference_cosine_heatmaps",
    "plot_gradient_interference_cosine_trajectories",
    "plot_learning_rate_schedules",
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
    if expected_sampling_label == "uniform_global_window":
        return bool(
            actual_sampling_label
            and re.fullmatch(
                r"uniform_global_h[1-9][0-9]*", actual_sampling_label
            )
        )
    if expected_sampling_label == "balanced_global_window":
        return bool(
            actual_sampling_label
            and re.fullmatch(
                r"balanced_global_h[1-9][0-9]*", actual_sampling_label
            )
        )
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
    uniform_window_match = re.fullmatch(r"uniform_global_h([1-9][0-9]*)", sampling_label)
    if uniform_window_match is not None:
        return f"Uniform global (H={uniform_window_match.group(1)})"
    balanced_window_match = re.fullmatch(
        r"balanced_global_h([1-9][0-9]*)", sampling_label
    )
    if balanced_window_match is not None:
        return f"Balanced global (H={balanced_window_match.group(1)})"
    if sampling_label == "fixed_global":
        return "fixed non-uniform global"
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

    if normalized_mode == "global":
        schedule = str(
            row.get("global_sampling_schedule") or "random_with_replacement"
        ).strip().lower()
        interval_value = row.get("global_sampling_interval_steps", 1)
        try:
            interval_steps = int(interval_value)
        except (TypeError, ValueError):
            interval_steps = 1
        if schedule == "balanced_cycle":
            return f"balanced_global_h{interval_steps}"
        if interval_steps > 1:
            return f"uniform_global_h{interval_steps}"

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

    if normalized_mode in {"global", "fixed_global", "per_block"}:
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


def mark_scheduler_warmup(
    axis: Any,
    positions: list[float] | tuple[float, ...] | set[float],
    *,
    label: bool = True,
) -> None:
    """Mark distinct positive scheduler-warmup boundaries on a time axis."""

    finite_positions = sorted(
        {
            float(position)
            for position in positions
            if position is not None
            and math.isfinite(float(position))
            and float(position) > 0.0
        }
    )
    if not finite_positions:
        return
    x_min, x_max = axis.get_xlim()
    visible = [position for position in finite_positions if x_min <= position <= x_max]
    for index, position in enumerate(visible):
        marker_label = None
        if label and index == 0:
            marker_label = (
                "LR warmup end"
                if len(visible) == 1
                else "LR warmup ends"
            )
        axis.axvline(
            position,
            color="#4d4d4d",
            linestyle="--",
            linewidth=1.05,
            alpha=0.8,
            zorder=6,
            label=marker_label,
        )


def _scheduler_curve_points(schedule: Any, *, max_points: int = 2001):
    """Reconstruct configured HF scheduler values without replaying training."""

    import torch
    from transformers import get_scheduler

    parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    optimizer = torch.optim.SGD([parameter], lr=schedule.peak_learning_rate)
    scheduler = get_scheduler(
        schedule.scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=schedule.warmup_steps,
        num_training_steps=(
            None
            if schedule.scheduler_name == "warmup_stable_decay"
            and "num_stable_steps" in schedule.scheduler_kwargs
            else schedule.max_steps
        ),
        scheduler_specific_kwargs=dict(schedule.scheduler_kwargs),
    )
    lr_lambdas = getattr(scheduler, "lr_lambdas", None)
    if not lr_lambdas:
        raise ValueError(
            f"scheduler {schedule.scheduler_name!r} cannot be reconstructed "
            "as a deterministic step-only trajectory"
        )

    if schedule.max_steps + 1 <= max_points:
        steps = list(range(schedule.max_steps + 1))
    else:
        steps = [
            round(index * schedule.max_steps / (max_points - 1))
            for index in range(max_points)
        ]
    steps.extend(
        step
        for step in (
            0,
            schedule.warmup_steps - 1,
            schedule.warmup_steps,
            schedule.warmup_steps + 1,
            schedule.max_steps,
        )
        if 0 <= step <= schedule.max_steps
    )
    scheduler_contract = getattr(schedule, "scheduler_contract", None)
    if isinstance(scheduler_contract, Mapping):
        cooldown_start = scheduler_contract.get("cooldown_start_step")
        if isinstance(cooldown_start, int):
            steps.extend(
                step
                for step in (
                    cooldown_start - 1,
                    cooldown_start,
                    cooldown_start + 1,
                )
                if 0 <= step <= schedule.max_steps
            )
    steps = sorted(set(steps))
    factor = lr_lambdas[0]
    learning_rates = [schedule.peak_learning_rate * float(factor(step)) for step in steps]
    return steps, learning_rates


def plot_learning_rate_schedules(
    schedules: list[Any] | tuple[Any, ...],
    output_path: str | Path,
    *,
    dpi: int = 300,
) -> Path:
    """Plot unique configured LR schedules over steps and nominal tokens."""

    grouped: dict[str, list[Any]] = {}
    for schedule in schedules:
        contract = json.dumps(
            {
                "name": schedule.scheduler_name,
                "kwargs": schedule.scheduler_kwargs,
                "peak": schedule.peak_learning_rate,
                "warmup": schedule.warmup_steps,
                "max_steps": schedule.max_steps,
                "tokens_per_step": schedule.expected_tokens_per_step,
                "token_budget": schedule.token_budget,
                "contract": getattr(schedule, "scheduler_contract", None),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        grouped.setdefault(contract, []).append(schedule)

    curves = []
    for replications in grouped.values():
        schedule = replications[0]
        try:
            steps, learning_rates = _scheduler_curve_points(schedule)
        except (TypeError, ValueError, KeyError) as error:
            warnings.warn(
                f"Skipping configured LR curve for {schedule.run_id}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        run_suffix = (
            schedule.run_id
            if len(replications) == 1
            else f"{len(replications)} runs"
        )
        label = (
            f"{schedule.scheduler_name} · warmup "
            f"{schedule.warmup_steps:,}/{schedule.max_steps:,} · "
            f"peak {schedule.peak_learning_rate:.3g} · {run_suffix}"
        )
        token_positions = [
            min(step * schedule.expected_tokens_per_step, schedule.token_budget)
            for step in steps
        ]
        curves.append((schedule, steps, token_positions, learning_rates, label))
    if not curves:
        raise ValueError("no deterministic configured LR schedules were found")

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for index, (schedule, steps, tokens, learning_rates, label) in enumerate(curves):
        color = colors(index % 10)
        axes[0].plot(steps, learning_rates, color=color, linewidth=1.8, label=label)
        axes[1].plot(tokens, learning_rates, color=color, linewidth=1.8, label=label)

    mark_scheduler_warmup(
        axes[0],
        {schedule.warmup_steps for schedule, *_ in curves},
        label=False,
    )
    cooldown_steps = {
        int(schedule.scheduler_contract["cooldown_start_step"])
        for schedule, *_ in curves
        if isinstance(getattr(schedule, "scheduler_contract", None), Mapping)
        and schedule.scheduler_contract.get("cooldown_start_step") is not None
    }
    cooldown_tokens = {
        float(schedule.scheduler_contract["cooldown_start_tokens"])
        for schedule, *_ in curves
        if isinstance(getattr(schedule, "scheduler_contract", None), Mapping)
        and schedule.scheduler_contract.get("cooldown_start_tokens") is not None
    }
    for axis, positions in zip(axes, (cooldown_steps, cooldown_tokens), strict=True):
        for index, position in enumerate(sorted(positions)):
            axis.axvline(
                position,
                color="#b35806",
                linestyle=":",
                linewidth=1.25,
                alpha=0.9,
                zorder=6,
                label="WSD cooldown begins" if index == 0 else None,
            )
    mark_scheduler_warmup(
        axes[1],
        {
            min(
                schedule.warmup_steps * schedule.expected_tokens_per_step,
                schedule.token_budget,
            )
            for schedule, *_ in curves
        },
        label=False,
    )
    axes[0].set_xlabel("Completed optimizer step")
    axes[1].set_xlabel("Nominal training tokens")
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    for axis in axes:
        axis.set_ylabel("Learning rate")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axis.grid(True, alpha=0.25)
    axes[0].legend(loc="best", frameon=False, fontsize=8)
    figure.suptitle(
        "Configured learning-rate schedules\n"
        "Reconstructed from the resolved config.json contract",
        fontsize=15,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


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
    mark_scheduler_warmup(
        axis,
        {getattr(timeline, "scheduler_warmup_tokens", 0)},
        label=False,
    )

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
    for axis in axes:
        mark_scheduler_warmup(
            axis,
            {getattr(timeline, "scheduler_warmup_tokens", 0)},
            label=False,
        )
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
    """Render token-weighted per-block allocation as a 100%-stacked timeline."""

    granularities = timeline.ordered_granularities
    colors = _sampling_colors(granularities)
    spans, shares, binned = _selection_share_bins(timeline, max_bins=max_bins)
    if not spans:
        raise ValueError("granularity allocation plot requires at least one window")
    centers = [(start + end) / 2 for start, end in spans]
    widths = [end - start for start, end in spans]
    figure, axis = plt.subplots(figsize=(12, 4.8))
    bottoms = [0.0] * len(spans)
    for granularity, color in zip(granularities, colors):
        heights = [share * 100.0 for share in shares[granularity]]
        axis.bar(
            centers,
            heights,
            width=widths,
            bottom=bottoms,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            label=granularity,
            align="center",
        )
        bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]

    axis.set_xlim(min(start for start, _ in spans), max(end for _, end in spans))
    axis.set_ylim(0.0, 100.0)
    axis.set_yticks([0.0, 50.0, 100.0])
    axis.set_xlabel("Total training tokens")
    axis.set_ylabel("Selected block fraction (%)")
    axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axis.grid(True, axis="both", alpha=0.25)
    axis.set_axisbelow(True)
    mark_scheduler_warmup(
        axis,
        {getattr(timeline, "scheduler_warmup_tokens", 0)},
    )
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    aggregation = (
        f"{len(spans)} equal-token bins"
        if binned
        else "one bar per controller window"
    )
    figure.suptitle(
        f"Granularity allocation over training — {timeline.run_id}\n{aggregation}",
        fontsize=15,
        y=0.995,
    )
    figure.subplots_adjust(left=0.10, right=0.82, top=0.86, bottom=0.14)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _selection_share_series(timeline, *, max_bins: int = 100):
    """Return token positions and allocation shares that sum to one."""

    spans, shares, binned = _selection_share_bins(timeline, max_bins=max_bins)
    return ([(start + end) / 2 for start, end in spans], shares, binned)


def _selection_share_bins(timeline, *, max_bins: int = 100):
    """Return token spans and token-weighted block shares that sum to one."""

    if max_bins <= 0:
        raise ValueError("max_bins must be positive")
    granularities = tuple(timeline.ordered_granularities)
    windows = tuple(timeline.windows)
    if not windows:
        return [], {label: [] for label in granularities}, False

    if len(windows) <= max_bins:
        spans = [(window.start_tokens, window.end_tokens) for window in windows]
        shares = {
            label: [
                window.block_granularities.count(label)
                / len(window.block_granularities)
                for window in windows
            ]
            for label in granularities
        }
        return spans, shares, False

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
    spans = [(bin_edges[index], bin_edges[index + 1]) for index in populated]
    shares = {
        label: [weighted[label][index] / coverage[index] for index in populated]
        for label in granularities
    }
    return spans, shares, True


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
    mark_scheduler_warmup(
        axis,
        {getattr(history, "scheduler_warmup_tokens", 0)},
    )
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
            label=f"sampling p({label})",
        )
        axes[1].plot(
            tokens,
            [refresh.q[granularity_index] for refresh in history.refreshes],
            color=color,
            linewidth=1.1,
            linestyle="--",
            alpha=0.75,
            label=f"score q({label})",
        )

    score_label = (
        "Gradient RMS (L2 / sqrt(N))"
        if history.importance_metric == "gradient_rms"
        else "Gradient L2 norm"
    )
    axes[0].set_yscale("log", nonpositive="clip")
    axes[0].set_ylabel(f"{score_label}\n(log scale)")
    axes[0].set_title("Controlled-FFN gradient importance")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("Probability")
    axes[1].set_title(
        "Score distribution q (dashed) and epsilon-mixed sampling distribution p (solid)"
    )
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
        label="cumulative refresh duration",
    )
    axes[4].set_ylabel("Refresh duration (s)", color="tab:green")
    cumulative_axis.set_ylabel("Cumulative refresh duration (s)", color="tab:red")
    axes[4].tick_params(axis="y", labelcolor="tab:green")
    cumulative_axis.tick_params(axis="y", labelcolor="tab:red")
    axes[4].set_title(
        "PanelGrad measurement overhead — "
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
        mark_scheduler_warmup(
            axis,
            {getattr(history, "scheduler_warmup_tokens", 0)},
            label=False,
        )
    figure.suptitle(
        f"PanelGrad refresh diagnostics — {history.run_id}\n"
        "Dashed line = LR warmup end",
        fontsize=15,
    )
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

    bins = _global_sampling_bins(history, bin_steps=bin_steps)
    granularities = tuple(history.ordered_granularities)
    x_values = [(item[0] + item[1]) / 2 for item in bins]
    shares = {
        label: [item[3][label] for item in bins]
        for label in granularities
    }
    bin_sizes = [item[2] for item in bins]
    return x_values, shares, bin_sizes


def _global_sampling_bins(history, *, bin_steps: int = 50):
    """Return token spans, sizes, and fractions for fixed step-count bins."""

    if bin_steps <= 0:
        raise ValueError("bin_steps must be positive")
    granularities = tuple(history.ordered_granularities)
    bins = []
    actions = tuple(history.actions)
    for start in range(0, len(actions), bin_steps):
        chunk = actions[start : start + bin_steps]
        if not chunk:
            continue
        counts = {label: 0 for label in granularities}
        for action in chunk:
            counts[action.granularity] += 1
        bins.append(
            (
                chunk[0].start_tokens,
                chunk[-1].end_tokens,
                len(chunk),
                {label: counts[label] / len(chunk) for label in granularities},
            )
        )
    return bins


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
    for index, axis in enumerate(axes):
        mark_scheduler_warmup(
            axis,
            {getattr(history, "scheduler_warmup_tokens", 0)},
            label=index == 0,
        )
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
        mark_scheduler_warmup(
            axis,
            {getattr(history, "scheduler_warmup_steps", 0)},
            label=False,
        )
    axes[-1].set_xlabel("Completed optimizer step")
    figure.suptitle(
        f"Exact global sampling decisions — {history.run_id}\n"
        "Dashed line = LR warmup end when visible",
        fontsize=15,
        y=0.995,
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
    """Compare local exposure as one 100%-stacked token-width timeline per run."""

    histories = tuple(histories)
    if not histories:
        raise ValueError("sampling exposure comparison requires at least one history")
    granularities = histories[0].ordered_granularities
    if any(history.ordered_granularities != granularities for history in histories):
        raise ValueError("sampling histories must use the same granularity order")
    colors = _sampling_colors(granularities)
    figure, axes = plt.subplots(
        len(histories),
        1,
        figsize=(13, 1.3 + 2.6 * len(histories)),
        sharex=True,
    )
    axes = [axes] if len(histories) == 1 else list(axes)
    observed_start = min(history.actions[0].start_tokens for history in histories)
    observed_end = max(history.actions[-1].end_tokens for history in histories)
    for axis, history in zip(axes, histories):
        bins = _global_sampling_bins(history, bin_steps=bin_steps)
        centers = [(start + end) / 2 for start, end, _, _ in bins]
        widths = [end - start for start, end, _, _ in bins]
        bottoms = [0.0] * len(bins)
        for granularity, color in zip(granularities, colors):
            heights = [fractions[granularity] * 100.0 for _, _, _, fractions in bins]
            axis.bar(
                centers,
                heights,
                width=widths,
                bottom=bottoms,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                label=granularity,
                align="center",
            )
            bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]
        axis.set_ylim(0.0, 100.0)
        axis.set_yticks([0.0, 50.0, 100.0])
        axis.set_ylabel("Exposure (%)")
        seed_label = "unknown" if history.seed is None else str(history.seed)
        axis.set_title(
            f"{history.policy_label} · seed {seed_label}",
            fontsize=11,
            loc="left",
        )
        axis.grid(True, alpha=0.25)
        axis.set_axisbelow(True)
        mark_scheduler_warmup(
            axis,
            {getattr(history, "scheduler_warmup_tokens", 0)},
            label=axis is axes[0],
        )
    axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    axes[-1].set_xlim(observed_start, observed_end)
    axes[-1].set_xlabel("Total training tokens")
    axes[-1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    figure.suptitle(
        "Empirical global selection distribution\n"
        f"{bin_steps}-optimizer-step bins",
        fontsize=15,
        y=0.995,
    )
    figure.subplots_adjust(left=0.11, right=0.82, top=0.84, bottom=0.09, hspace=0.48)
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
        axis.set_ylabel("Cumulative\nselected-step share")
        axis.set_title(granularity, fontsize=11)
        axis.grid(True, alpha=0.25)
        mark_scheduler_warmup(
            axis,
            {
                getattr(history, "scheduler_warmup_tokens", 0)
                for history in histories
            },
            label=axis is axes[0],
        )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=min(4, len(legend_labels)),
            frameon=False,
        )
    axes[-1].set_xlim(
        min(history.actions[0].start_tokens for history in histories),
        max(history.actions[-1].end_tokens for history in histories),
    )
    axes[-1].set_xlabel("Total training tokens")
    axes[-1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    figure.suptitle("Cumulative global granularity exposure", fontsize=15, y=0.995)
    figure.subplots_adjust(left=0.11, right=0.98, top=0.91, bottom=0.07, hspace=0.38)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _write_global_sampling_summary(histories, output_path: Path) -> Path:
    granularities = histories[0].ordered_granularities
    fieldnames = [
        "run_id", "policy", "global_sampling_schedule",
        "global_sampling_schedule_version", "seed", "completed_steps",
        "policy_decisions", "decisions_per_step", "action_transitions",
    ]
    for label in granularities:
        fieldnames.extend(
            [
                f"{label}_steps",
                f"{label}_fraction",
                f"{label}_target_probability",
            ]
        )
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
                "global_sampling_schedule": history.global_sampling_schedule or "",
                "global_sampling_schedule_version": (
                    ""
                    if history.global_sampling_schedule_version is None
                    else history.global_sampling_schedule_version
                ),
                "seed": "" if history.seed is None else history.seed,
                "completed_steps": len(history.actions),
                "policy_decisions": history.decision_count,
                "decisions_per_step": history.decision_count / len(history.actions),
                "action_transitions": transitions,
            }
            for label in granularities:
                row[f"{label}_steps"] = counts[label]
                row[f"{label}_fraction"] = counts[label] / len(history.actions)
                row[f"{label}_target_probability"] = (
                    ""
                    if history.target_probabilities is None
                    else history.target_probabilities[granularities.index(label)]
                )
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
    sampling_zoom_steps: int | None = None,
) -> list[Path]:
    """Generate action-grounded views shared by all global policies."""

    from . import reporting_io
    from .reporting_impl import filter_plot_rows

    if sampling_bin_steps <= 0:
        raise ValueError("sampling_bin_steps must be positive")
    if sampling_zoom_steps is not None and sampling_zoom_steps <= 0:
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
        if sampling_zoom_steps is not None:
            paths.append(
                plot_global_sampling_zoom(
                    history,
                    output_dir / global_sampling_zoom_filename(history.run_id),
                    zoom_steps=sampling_zoom_steps,
                    dpi=dpi,
                )
            )
        # Per-run global heatmaps and share plots were replaced by grouped views.
        for stale_name in (
            controller_timeline_filename(history.run_id),
            controller_selection_share_filename(history.run_id),
            controller_selection_frequency_filename(history.run_id),
            panelgrad_exposure_share_filename(history.run_id),
        ):
            (output_dir / stale_name).unlink(missing_ok=True)

    groups: dict[str, list[Any]] = {}
    for history in histories:
        groups.setdefault(history.comparison_key, []).append(history)
    for comparison_key, group in sorted(groups.items()):
        granularities = {history.ordered_granularities for history in group}
        if len(granularities) != 1:
            continue
        group.sort(key=lambda item: (item.policy_identity, item.seed or -1, item.run_id))
        first = group[0]
        schedule_contracts = {
            (
                history.global_sampling_schedule,
                history.global_sampling_schedule_version,
            )
            for history in group
        }
        if len(schedule_contracts) == 1:
            schedule, schedule_version = next(iter(schedule_contracts))
            schedule_descriptor = safe_filename_fragment(schedule or "global_policies")
            if schedule_version is not None:
                schedule_descriptor += f"_v{schedule_version}"
        else:
            schedule_descriptor = "mixed_global_policies"
        descriptor = "__".join(
            [
                safe_filename_fragment(first.model_variant or "unknown"),
                safe_filename_fragment(first.correction_mode or "unknown"),
                schedule_descriptor,
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


def gradient_interference_cosine_trajectories_filename(contract: str) -> str:
    return f"gradient_interference_cosine_trajectories__{contract[:12]}.png"


def gradient_interference_cosine_heatmaps_filename(run_id: str) -> str:
    return (
        "gradient_interference_cosine_heatmaps__"
        f"{safe_filename_fragment(run_id)}.png"
    )


def _gradient_interval_color(interval_steps: int) -> Any:
    fixed = {
        1: "#1f77b4",
        5: "#ff7f0e",
        25: "#2ca02c",
        50: "#d62728",
    }
    if interval_steps in fixed:
        return fixed[interval_steps]
    color_index = int(
        hashlib.sha256(str(interval_steps).encode("utf-8")).hexdigest()[:8], 16
    ) % 20
    return plt.get_cmap("tab20")(color_index)


def plot_gradient_interference_cosine_trajectories(
    histories: list[Any] | tuple[Any, ...],
    output_path: str | Path,
    *,
    dpi: int = 300,
) -> Path:
    """Compare every unordered granularity-pair cosine across H policies."""

    if not histories:
        raise ValueError("gradient-interference trajectory histories are empty")
    histories = sorted(
        histories,
        key=lambda history: (
            history.sampling_interval_steps,
            history.seed,
            history.run_id,
        ),
    )
    first = histories[0]
    expected_pairs = first.unordered_pairs
    if any(
        history.comparison_contract != first.comparison_contract
        or history.unordered_pairs != expected_pairs
        for history in histories
    ):
        raise ValueError("gradient-interference trajectory histories are incompatible")
    reference_milestones = tuple(
        (snapshot.step, snapshot.tokens_seen, snapshot.milestone_reasons)
        for snapshot in first.snapshots
    )
    if any(
        tuple(
            (snapshot.step, snapshot.tokens_seen, snapshot.milestone_reasons)
            for snapshot in history.snapshots
        )
        != reference_milestones
        for history in histories[1:]
    ):
        raise ValueError(
            "gradient-interference histories are not milestone/token aligned"
        )

    column_count = 4
    row_count = max(1, math.ceil(len(expected_pairs) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(16, 2.45 * row_count),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes_list = list(axes.flat)
    by_interval: dict[int, list[Any]] = {}
    for history in histories:
        by_interval.setdefault(history.sampling_interval_steps, []).append(history)

    for pair_index, pair in enumerate(expected_pairs):
        axis = axes_list[pair_index]
        axis.axhline(0.0, color="#777777", linewidth=0.7, linestyle=":", zorder=0)
        for interval_steps, replications in sorted(by_interval.items()):
            color = _gradient_interval_color(interval_steps)
            reference_x = tuple(
                snapshot.tokens_seen for snapshot in replications[0].snapshots
            )
            for replication in replications[1:]:
                if tuple(snapshot.tokens_seen for snapshot in replication.snapshots) != reference_x:
                    raise ValueError(
                        "gradient-interference seed replications are not token-aligned "
                        f"for H={interval_steps}"
                    )
            seed_values: list[list[float]] = []
            for replication in replications:
                values = [
                    (
                        math.nan
                        if snapshot.pairs[pair_index].cosine is None
                        else float(snapshot.pairs[pair_index].cosine)
                    )
                    for snapshot in replication.snapshots
                ]
                seed_values.append(values)
                if len(replications) > 1:
                    axis.plot(
                        reference_x,
                        values,
                        color=color,
                        linewidth=0.85,
                        alpha=0.22,
                        marker="o",
                        markersize=2.6,
                    )
            mean_values = []
            for milestone_index in range(len(reference_x)):
                finite_values = [
                    values[milestone_index]
                    for values in seed_values
                    if math.isfinite(values[milestone_index])
                ]
                mean_values.append(
                    sum(finite_values) / len(finite_values)
                    if finite_values
                    else math.nan
                )
            axis.plot(
                reference_x,
                mean_values,
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=3.8,
                label=f"H={interval_steps}",
            )
        mark_scheduler_warmup(
            axis,
            {
                getattr(history, "scheduler_warmup_tokens", 0)
                for history in histories
            },
            label=pair_index == 0,
        )
        axis.set_title(f"{pair[0]} vs {pair[1]}", fontsize=9)
        axis.set_ylim(-1.0, 1.0)
        axis.grid(axis="y", alpha=0.18, linewidth=0.5)
        if pair_index % column_count == 0:
            axis.set_ylabel("Cosine")
        if pair_index // column_count == row_count - 1:
            axis.set_xlabel("Tokens seen")

    for axis in axes_list[len(expected_pairs) :]:
        axis.set_visible(False)
    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            title="H = consecutive optimizer steps per sampled width",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=len(handles),
            frameon=True,
            handlelength=2.8,
            columnspacing=1.6,
            borderaxespad=0.0,
        )
    figure.suptitle(
        "Pairwise controlled-gradient cosine trajectories",
        fontsize=15,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _format_gradient_milestone_title(snapshot: Any) -> str:
    if snapshot.tokens_seen >= 1_000_000:
        token_label = f"{snapshot.tokens_seen / 1_000_000:g}M tokens"
    else:
        token_label = f"{snapshot.tokens_seen:,} tokens"
    reasons = ", ".join(snapshot.milestone_reasons)
    return f"step {snapshot.step} · {token_label}\n{reasons}"


def plot_gradient_interference_cosine_heatmaps(
    history: Any,
    output_path: str | Path,
    *,
    dpi: int = 300,
) -> Path:
    """Render symmetric milestone cosine matrices for one diagnostic run."""

    column_count = 3
    row_count = max(1, math.ceil(len(history.snapshots) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(13.5, 4.35 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    axes_list = list(axes.flat)
    granularity_count = len(history.ordered_granularities)
    colormap = plt.get_cmap("coolwarm").copy()
    colormap.set_bad("#eeeeee")
    image = None
    for snapshot_index, snapshot in enumerate(history.snapshots):
        axis = axes_list[snapshot_index]
        matrix = np.full((granularity_count, granularity_count), np.nan, dtype=float)
        for pair in snapshot.pairs:
            left_index = history.ordered_granularities.index(pair.left_granularity)
            right_index = history.ordered_granularities.index(pair.right_granularity)
            if pair.cosine is not None and not pair.has_zero_norm:
                matrix[left_index, right_index] = pair.cosine
                matrix[right_index, left_index] = pair.cosine
        image = axis.imshow(
            np.ma.masked_invalid(matrix),
            cmap=colormap,
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axis.set_title(_format_gradient_milestone_title(snapshot), fontsize=9)
        axis.set_xticks(range(granularity_count))
        axis.set_xticklabels(
            history.ordered_granularities,
            rotation=45,
            ha="right",
            fontsize=8,
        )
        axis.set_yticks(range(granularity_count))
        axis.set_yticklabels(history.ordered_granularities, fontsize=8)
    for axis in axes_list[len(history.snapshots) :]:
        axis.set_visible(False)
    if image is not None:
        colorbar = figure.colorbar(
            image,
            ax=[axis for axis in axes_list[: len(history.snapshots)]],
            shrink=0.82,
            pad=0.02,
        )
        colorbar.set_label("Cosine")
    figure.suptitle(
        f"Controlled-gradient cosine milestones · {history.run_id}",
        fontsize=14,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def generate_gradient_interference_figures(
    input_root: str | Path,
    output_dir: str | Path,
    *,
    dpi: int = 300,
    variants: list[str] | tuple[str, ...] | None = None,
    corrections: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    """Discover eligible journals and render grouped and per-run diagnostics."""

    from . import reporting_io
    from .reporting_impl import filter_plot_rows

    output_dir = Path(output_dir)
    histories = []
    for history in reporting_io.iter_gradient_interference_histories(input_root):
        keep = filter_plot_rows(
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
        if keep:
            histories.append(history)

    paths: list[Path] = []
    groups: dict[str, list[Any]] = {}
    for history in histories:
        groups.setdefault(history.comparison_contract, []).append(history)
        paths.append(
            plot_gradient_interference_cosine_heatmaps(
                history,
                output_dir
                / gradient_interference_cosine_heatmaps_filename(history.run_id),
                dpi=dpi,
            )
        )
    for contract, group in sorted(groups.items()):
        paths.append(
            plot_gradient_interference_cosine_trajectories(
                group,
                output_dir
                / gradient_interference_cosine_trajectories_filename(contract),
                dpi=dpi,
            )
        )
    return paths


def _generate_portfolio_comparison_figures(
    input_root: Path,
    comparison_manifest: Path,
    output_dir: Path,
    *,
    dpi: int,
    include_final_holdout: bool,
    refresh_counts: bool,
    validation_loss_log_y: bool,
) -> list[Path]:
    """Render the complete figure set for one sealed portfolio comparison."""

    del input_root  # Membership comes exclusively from the immutable report.
    manifest_path = comparison_manifest.expanduser().resolve()
    try:
        report = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read portfolio comparison manifest: {manifest_path}") from error
    if not isinstance(report, dict) or report.get("analysis") != (
        "tinystories_instruct_four_granularity_portfolio_catchup"
    ):
        raise ValueError("Comparison manifest is not a portfolio catch-up report")
    body = dict(report)
    saved_hash = body.pop("report_hash", None)
    if saved_hash != stable_hash(body):
        raise ValueError("Portfolio catch-up report hash mismatch")

    target_path = Path(str(report.get("target_manifest_path", ""))).expanduser().resolve()
    try:
        targets = json.loads(target_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read portfolio target manifest: {target_path}") from error
    if not isinstance(targets, dict):
        raise ValueError("Portfolio target manifest must be a mapping")
    target_body = dict(targets)
    target_hash = target_body.pop("manifest_hash", None)
    if target_hash != stable_hash(target_body) or target_hash != report.get(
        "target_manifest_hash"
    ):
        raise ValueError("Portfolio target manifest hash mismatch")
    selection = _load_portfolio_holdout_selection(
        manifest_path.parent,
        report=report,
        targets=targets,
    )

    declared: list[dict[str, Any]] = []
    for seed_text, seed_targets in targets.get("targets", {}).items():
        if not isinstance(seed_targets, Mapping):
            raise ValueError("Portfolio targets are malformed")
        for width, target in seed_targets.items():
            declared.append(
                {
                    "role": "standalone_reference",
                    "seed": int(seed_text),
                    "width": str(width),
                    "run_dir": Path(str(target["run_dir"])).expanduser().resolve(),
                    "budget": int(report["reference_budget_tokens"]),
                }
            )
    for seed_record in report.get("seeds", []):
        for width in targets.get("granularities", []):
            declared.append(
                {
                    "role": "elastic_candidate",
                    "seed": int(seed_record["seed"]),
                    "width": str(width),
                    "run_dir": Path(str(seed_record["run_dir"])).expanduser().resolve(),
                    "budget": int(report["elastic_budget_cap_tokens"]),
                }
            )
    expected_count = len(targets.get("seeds", [])) * len(
        targets.get("granularities", [])
    ) * 2
    if len(declared) != expected_count or expected_count != 24:
        raise ValueError("Portfolio report does not declare the complete 12+3 run set")

    config_cache: dict[Path, dict[str, Any]] = {}
    metrics_cache: dict[Path, list[dict[str, str]]] = {}
    rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    provenance_values: set[str] = set()
    for entry in declared:
        run_dir = entry["run_dir"]
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.csv"
        if not config_path.is_file() or not metrics_path.is_file():
            raise ValueError(f"Declared portfolio run is incomplete: {run_dir}")
        if config_path not in config_cache:
            with config_path.open("r", encoding="utf-8") as source:
                config = json.load(source)
            if not isinstance(config, dict):
                raise ValueError(f"Saved portfolio config is malformed: {config_path}")
            config_cache[config_path] = config
        config = config_cache[config_path]
        training = config.get("training", {})
        controlled = config.get("controlled_experiment", {})
        role = controlled.get("comparison_role") if isinstance(controlled, Mapping) else None
        if role != entry["role"] or int(training.get("token_budget", -1)) != entry["budget"]:
            raise ValueError("Declared portfolio role or budget differs from its saved config")
        scheduler = str(training.get("scheduler_name"))
        if scheduler != "cosine":
            raise ValueError("Portfolio comparison requires cosine schedules")
        provenance_values.add(_portfolio_reporting_provenance(config))
        if metrics_path not in metrics_cache:
            with metrics_path.open("r", encoding="utf-8", newline="") as source:
                metrics_cache[metrics_path] = list(csv.DictReader(source))
        metrics = metrics_cache[metrics_path]
        validation = []
        for raw in metrics:
            if raw.get("split") != "validation" or raw.get("granularity") != entry["width"]:
                continue
            try:
                step = int(float(raw["step"]))
                tokens = int(float(raw["tokens_seen"]))
                loss = float(raw["loss"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid portfolio validation row: {metrics_path}") from error
            if math.isfinite(loss):
                validation.append(
                    {
                        "step": step,
                        "tokens_seen": tokens,
                        "loss": loss,
                        "seed": entry["seed"],
                    }
                )
        if not validation or max(row["tokens_seen"] for row in validation) > entry["budget"]:
            raise ValueError("Portfolio validation trace exceeds or lacks its declared budget")
        rows_by_key.setdefault((entry["role"], entry["width"]), []).extend(validation)
    if len(provenance_values) != 1:
        raise ValueError(
            "Portfolio manifest mode requires identical corpus and holdout provenance"
        )

    granularities = [str(value) for value in targets["granularities"]]
    parameter_counts = _portfolio_non_embedding_parameter_counts(
        declared,
        granularities=granularities,
        config_cache=config_cache,
        refresh_counts=refresh_counts,
    )
    selected_validation_rows = _portfolio_selected_validation_rows(
        report,
        targets=targets,
        selection=selection,
    )
    selection_label = _portfolio_elastic_selection_label(selection)
    for filename in (
        "ppl_vs_size.png",
        "portfolio_validation_loss_over_tokens.png",
        "portfolio_per_granularity_deficits.png",
        "portfolio_worst_width_deficit.png",
        "learning_rate_schedule.png",
        "final_holdout_ppl_vs_size.png",
        "portfolio_final_holdout_perplexity.png",
        "portfolio_final_holdout_deficit_vs_size.png",
    ):
        stale_path = output_dir / filename
        if stale_path.is_file():
            stale_path.unlink()

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, width in zip(axes.flat, granularities, strict=True):
        for role, color in (
            ("standalone_reference", "tab:gray"),
            ("elastic_candidate", "tab:blue"),
        ):
            group = rows_by_key[(role, width)]
            curve = _portfolio_mean_curve(group)
            budget = (
                int(report["reference_budget_tokens"])
                if role == "standalone_reference"
                else int(report["elastic_budget_cap_tokens"])
            )
            active = width if role == "standalone_reference" else "all four"
            arm_fragment = (
                f"; arm={report['comparison_arm_id']}"
                if role == "elastic_candidate"
                and report.get("post_hoc_diagnostic") is True
                else ""
            )
            axis.plot(
                [row["tokens_seen"] for row in curve],
                [row["mean"] for row in curve],
                color=color,
                linewidth=1.5,
                label=(
                    f"{role}{arm_fragment}; active={active}; budget={budget:,}; "
                    f"scheduler=cosine; n={curve[0]['seed_count']} seeds"
                ),
            )
            axis.fill_between(
                [row["tokens_seen"] for row in curve],
                [row["minimum"] for row in curve],
                [row["maximum"] for row in curve],
                color=color,
                alpha=0.12,
            )
        width_targets = [
            float(targets["targets"][str(seed)][width]["target_loss"])
            for seed in targets["seeds"]
        ]
        lower = min(width_targets)
        upper = max(width_targets)
        tolerance = float(report["loss_tolerance"])
        axis.axhspan(lower, upper, color="tab:green", alpha=0.12, label="standalone target range")
        axis.axhspan(upper, upper + tolerance, color="tab:green", alpha=0.06, label="0.5% PPL tolerance band")
        for seed_record in report["seeds"]:
            if seed_record.get("confirmation_tokens") is not None:
                axis.axvline(float(seed_record["confirmation_tokens"]), color="tab:red", linestyle=":", alpha=0.28)
        axis.set_title(width)
        axis.set_ylabel("ordinary-validation loss")
        if validation_loss_log_y:
            axis.set_yscale("log")
    for axis in axes[-1]:
        axis.set_xlabel("optimizer tokens")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, 0.11, 1, 1))
    curve_path = output_dir / "portfolio_validation_loss_over_tokens.png"
    figure.savefig(curve_path, dpi=dpi)
    plt.close(figure)

    validation_size_path = _plot_portfolio_ppl_vs_size(
        selected_validation_rows,
        parameter_counts=parameter_counts,
        granularities=granularities,
        output_paths=[output_dir / "ppl_vs_size.png"],
        title="Ordinary-validation perplexity at manifest-selected checkpoints",
        ylabel="Perplexity",
        elastic_label=selection_label,
        perplexity_tolerance=float(report["perplexity_tolerance"]),
        dpi=dpi,
    )[0]

    per_width_deficit_path = _plot_portfolio_per_granularity_deficits(
        report,
        granularities=granularities,
        output_dir=output_dir,
        dpi=dpi,
    )

    figure, axis = plt.subplots(figsize=(9, 5))
    for seed_record in report["seeds"]:
        observations = seed_record.get("validation_observations", [])
        axis.plot(
            [row["tokens_seen"] for row in observations],
            [100.0 * math.expm1(float(row["joint_max_loss_gap"])) for row in observations],
            label=(
                "elastic_candidate"
                + (
                    f" arm={report['comparison_arm_id']}"
                    if report.get("post_hoc_diagnostic") is True
                    else ""
                )
                + f" seed {seed_record['seed']}; "
                f"budget={report['elastic_budget_cap_tokens']:,}; scheduler=cosine"
            ),
        )
        if seed_record.get("onset_tokens") is not None:
            onset = next(
                row
                for row in observations
                if row["tokens_seen"] == seed_record["onset_tokens"]
            )
            axis.scatter(
                [seed_record["onset_tokens"]],
                [100.0 * math.expm1(float(onset["joint_max_loss_gap"]))],
                s=36,
                facecolors="none",
                edgecolors="tab:orange",
                label="joint-streak onset" if int(seed_record["seed"]) == 42 else None,
                zorder=4,
            )
        if seed_record.get("confirmation_tokens") is not None:
            confirmation = next(
                row
                for row in observations
                if row["tokens_seen"] == seed_record["confirmation_tokens"]
            )
            axis.scatter(
                [seed_record["confirmation_tokens"]],
                [100.0 * math.expm1(float(confirmation["joint_max_loss_gap"]))],
                marker="*",
                s=70,
                color="tab:red",
                label="five-validation confirmation"
                if int(seed_record["seed"]) == 42
                else None,
                zorder=5,
            )
    axis.axhline(
        100.0 * float(report["perplexity_tolerance"]),
        color="black",
        linestyle="--",
        label="joint 0.5% PPL threshold",
    )
    axis.set_yscale(
        "symlog",
        linthresh=100.0 * float(report["perplexity_tolerance"]),
    )
    axis.set(
        xlabel="optimizer tokens",
        ylabel="worst-width PPL deficit (%) · symlog",
    )
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    deficit_path = output_dir / "portfolio_worst_width_deficit.png"
    figure.savefig(deficit_path, dpi=dpi)
    plt.close(figure)
    learning_rate_path = _plot_portfolio_learning_rate_schedules(
        declared,
        config_cache=config_cache,
        output_dir=output_dir,
        dpi=dpi,
    )
    paths = [
        validation_size_path,
        curve_path,
        per_width_deficit_path,
        deficit_path,
        learning_rate_path,
    ]

    if include_final_holdout:
        holdout_paths = _plot_portfolio_final_holdout_if_complete(
            selection,
            output_dir,
            parameter_counts=parameter_counts,
            granularities=granularities,
            perplexity_tolerance=float(report["perplexity_tolerance"]),
            dpi=dpi,
        )
        paths.extend(holdout_paths)
    return paths


def _portfolio_reporting_provenance(config: Mapping[str, Any]) -> str:
    model = config.get("model", {})
    dataset = config.get("dataset", {})
    return stable_hash(
        {
            "dataset_name": dataset.get("dataset_name"),
            "dataset_config_name": dataset.get("dataset_config_name"),
            "dataset_split": dataset.get("dataset_split"),
            "dataset_phase": dataset.get("dataset_phase"),
            "corpus_hash": dataset.get("corpus_hash", config.get("corpus_hash")),
            "optimizer_training_manifest_hash": config.get("optimizer_training_manifest_hash"),
            "validation_manifest_hash": config.get("validation_manifest_hash"),
            "final_holdout_manifest_hash": config.get("final_holdout_manifest_hash"),
            "tokenizer_manifest_hash": model.get("tokenizer_manifest_hash"),
            "variant": model.get("variant"),
            "correction_mode": model.get("correction_mode"),
            "d_model": model.get("d_model", model.get("hidden_size")),
            "num_layers": model.get("num_layers"),
            "context_length": model.get("context_length"),
            "vocab_size": model.get("vocab_size"),
        }
    )


def _portfolio_mean_curve(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_tokens: dict[int, list[float]] = {}
    seeds_by_tokens: dict[int, set[int]] = {}
    for row in rows:
        tokens = int(row["tokens_seen"])
        by_tokens.setdefault(tokens, []).append(float(row["loss"]))
        seeds_by_tokens.setdefault(tokens, set()).add(int(row["seed"]))
    seed_counts = {len(seeds) for seeds in seeds_by_tokens.values()}
    if seed_counts != {3}:
        raise ValueError("Portfolio replications do not share validation boundaries")
    return [
        {
            "tokens_seen": tokens,
            "mean": statistics.fmean(values),
            "minimum": min(values),
            "maximum": max(values),
            "seed_count": len(seeds_by_tokens[tokens]),
        }
        for tokens, values in sorted(by_tokens.items())
    ]


def _load_portfolio_holdout_selection(
    analysis_dir: Path,
    *,
    report: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> dict[str, Any]:
    selection_path = analysis_dir / "final_holdout_selection_manifest.json"
    try:
        with selection_path.open("r", encoding="utf-8") as source:
            selection = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot read portfolio holdout selection: {selection_path}"
        ) from error
    if not isinstance(selection, dict) or selection.get("analysis") != (
        "portfolio_final_holdout_selection"
    ):
        raise ValueError("Portfolio holdout selection manifest is malformed")
    selection_body = dict(selection)
    selection_hash = selection_body.pop("manifest_hash", None)
    if selection_hash != stable_hash(selection_body):
        raise ValueError("Portfolio holdout selection manifest hash mismatch")
    if (
        selection.get("target_manifest_hash") != targets.get("manifest_hash")
        or selection.get("comparison_group_id") != report.get("comparison_group_id")
    ):
        raise ValueError("Portfolio holdout selection provenance mismatch")
    selection_mode = selection.get("selection_mode", "portfolio_confirmation")
    confirmation_modes = {
        "portfolio_confirmation",
        "portfolio_confirmation_diagnostic",
    }
    terminal_modes = {
        "terminal_3B_censored",
        "terminal_candidate_budget_censored",
    }
    if selection_mode not in confirmation_modes | terminal_modes:
        raise ValueError("Portfolio holdout selection mode is invalid")
    if selection_mode != report.get(
        "final_holdout_selection_mode", selection_mode
    ):
        raise ValueError("Portfolio report and holdout selection mode differ")
    claim_eligible = selection.get(
        "claim_eligible", selection_mode == "portfolio_confirmation"
    )
    if not isinstance(claim_eligible, bool) or (
        claim_eligible and selection_mode != "portfolio_confirmation"
    ):
        raise ValueError("Portfolio holdout claim eligibility is inconsistent")

    entries = selection.get("entries")
    if not isinstance(entries, list) or len(entries) != 15:
        raise ValueError("Portfolio holdout selection must declare exactly 15 checkpoints")
    expected_seeds = {int(seed) for seed in targets.get("seeds", [])}
    granularities = [str(width) for width in targets.get("granularities", [])]
    reference_keys: set[tuple[int, str]] = set()
    elastic_seeds: set[int] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Portfolio holdout selection entry is malformed")
        role = str(entry.get("comparison_role"))
        seed = int(entry.get("seed", -1))
        widths = [str(width) for width in entry.get("granularities", [])]
        if seed not in expected_seeds:
            raise ValueError("Portfolio holdout selection contains an unexpected seed")
        expected_checkpoint_selection = (
            "ordinary_validation_best"
            if role == "standalone_reference"
            else (
                "portfolio_confirmation"
                if selection_mode in confirmation_modes
                else (
                    "terminal_candidate_budget"
                    if selection_mode == "terminal_candidate_budget_censored"
                    else "terminal_3B"
                )
            )
        )
        if entry.get("checkpoint_selection") != expected_checkpoint_selection:
            raise ValueError(
                "Portfolio holdout checkpoint selection contradicts its mode"
            )
        if role == "standalone_reference" and len(widths) == 1:
            reference_keys.add((seed, widths[0]))
        elif role == "elastic_candidate" and widths == granularities:
            if seed in elastic_seeds:
                raise ValueError("Duplicate elastic portfolio holdout selection")
            elastic_seeds.add(seed)
        else:
            raise ValueError("Portfolio holdout selection role or widths are invalid")
    if reference_keys != {
        (seed, width) for seed in expected_seeds for width in granularities
    } or elastic_seeds != expected_seeds:
        raise ValueError("Portfolio holdout selection does not contain the exact matrix")
    return selection


def _portfolio_non_embedding_parameter_counts(
    declared: list[Mapping[str, Any]],
    *,
    granularities: list[str],
    config_cache: Mapping[Path, Mapping[str, Any]],
    refresh_counts: bool,
) -> dict[str, int]:
    """Resolve one x coordinate per width without mixing role replications."""

    from . import reporting_io

    representatives: dict[tuple[str, str], Mapping[str, Any]] = {}
    model_contracts: dict[tuple[str, str], set[str]] = {}
    for entry in declared:
        key = (str(entry["role"]), str(entry["width"]))
        representatives.setdefault(key, entry)
        config_path = Path(entry["run_dir"]) / "config.json"
        config = config_cache[config_path]
        model_contracts.setdefault(key, set()).add(stable_hash(config.get("model", {})))
    if any(len(values) != 1 for values in model_contracts.values()):
        raise ValueError("Portfolio parameter-count model contract differs across seeds")

    refreshed_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    saved_cache: dict[Path, list[dict[str, str]]] = {}
    values: dict[tuple[str, str], int] = {}
    for key, entry in representatives.items():
        run_dir = Path(entry["run_dir"])
        config_path = run_dir / "config.json"
        width = key[1]
        if refresh_counts:
            if config_path not in refreshed_cache:
                refreshed_cache[config_path] = reporting_io.recompute_parameter_counts(
                    config_path
                )
            raw_count = refreshed_cache[config_path].get(width, {}).get(
                "non_embedding_parameters"
            )
        else:
            scaling_path = run_dir / "scaling_results.csv"
            if scaling_path not in saved_cache:
                try:
                    with scaling_path.open(
                        "r", encoding="utf-8", newline=""
                    ) as source:
                        saved_cache[scaling_path] = list(csv.DictReader(source))
                except (OSError, csv.Error, UnicodeError) as error:
                    raise ValueError(
                        "--no-refresh-counts requires scaling_results.csv for every "
                        f"manifest-selected run: {scaling_path}"
                    ) from error
            matches = [
                row
                for row in saved_cache[scaling_path]
                if str(row.get("granularity")) == width
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Manifest-scoped scaling_results.csv must contain one row per width"
                )
            raw_count = matches[0].get("non_embedding_parameters")
        try:
            numeric_count = float(raw_count)
            count = int(numeric_count)
        except (TypeError, ValueError) as error:
            raise ValueError("Portfolio non-embedding parameter count is invalid") from error
        if not math.isfinite(numeric_count) or numeric_count != count or count <= 0:
            raise ValueError("Portfolio non-embedding parameter count must be positive")
        values[key] = count

    by_width: dict[str, int] = {}
    for width in granularities:
        standalone = values[("standalone_reference", width)]
        elastic = values[("elastic_candidate", width)]
        if standalone != elastic:
            raise ValueError(
                "Corresponding standalone and elastic widths have different "
                f"non-embedding parameter counts for {width}: "
                f"{standalone} != {elastic}"
            )
        by_width[width] = standalone
    if len(set(by_width.values())) != len(granularities):
        raise ValueError("Portfolio widths do not resolve to distinct model sizes")
    return by_width


def _portfolio_selected_validation_rows(
    report: Mapping[str, Any],
    *,
    targets: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    granularities = [str(width) for width in targets["granularities"]]
    rows: list[dict[str, Any]] = []
    for seed in targets["seeds"]:
        for width in granularities:
            target = targets["targets"][str(seed)][width]
            loss = float(target["target_loss"])
            perplexity = float(target.get("target_perplexity", math.exp(loss)))
            if not math.isfinite(loss) or not math.isfinite(perplexity) or perplexity <= 0:
                raise ValueError("Portfolio standalone target metric is invalid")
            rows.append(
                {
                    "role": "standalone_reference",
                    "seed": int(seed),
                    "width": width,
                    "loss": loss,
                    "perplexity": perplexity,
                    "checkpoint_selection": "ordinary_validation_best",
                }
            )

    elastic_entries = {
        int(entry["seed"]): entry
        for entry in selection["entries"]
        if entry["comparison_role"] == "elastic_candidate"
    }
    seed_reports = {int(item["seed"]): item for item in report["seeds"]}
    for seed in targets["seeds"]:
        entry = elastic_entries[int(seed)]
        seed_report = seed_reports[int(seed)]
        matches = [
            observation
            for observation in seed_report.get("validation_observations", [])
            if int(observation["step"]) == int(entry["checkpoint_step"])
            and int(observation["tokens_seen"]) == int(entry["checkpoint_tokens"])
        ]
        if len(matches) != 1:
            raise ValueError(
                "Elastic selected checkpoint lacks an exact simultaneous ordinary "
                f"validation for seed {seed}"
            )
        selected = matches[0]
        width_values = selected.get("widths")
        if not isinstance(width_values, Mapping) or set(width_values) != set(
            granularities
        ):
            raise ValueError("Elastic selected validation does not contain all widths")
        for width in granularities:
            loss = float(width_values[width]["loss"])
            perplexity = math.exp(loss)
            if not math.isfinite(loss) or not math.isfinite(perplexity):
                raise ValueError("Elastic selected validation metric is invalid")
            rows.append(
                {
                    "role": "elastic_candidate",
                    "seed": int(seed),
                    "width": width,
                    "loss": loss,
                    "perplexity": perplexity,
                    "checkpoint_selection": entry["checkpoint_selection"],
                }
            )
    return rows


def _portfolio_elastic_selection_label(selection: Mapping[str, Any]) -> str:
    mode = selection.get("selection_mode")
    if mode == "terminal_3B_censored":
        return (
            "elastic_candidate; checkpoint=terminal 3B diagnostic; "
            "run budget=3B; scheduler=cosine; claim-ineligible; n=3 seeds"
        )
    if mode == "terminal_candidate_budget_censored":
        budget = int(selection["candidate_budget_tokens"])
        return (
            f"elastic_candidate; arm={selection['comparison_arm_id']}; "
            "checkpoint=terminal arm-budget diagnostic; "
            f"run budget={budget:,} tokens; scheduler=cosine; "
            "claim-ineligible; n=3 seeds"
        )
    if mode == "portfolio_confirmation_diagnostic":
        budget = int(selection["candidate_budget_tokens"])
        return (
            f"elastic_candidate; arm={selection['comparison_arm_id']}; "
            "checkpoint=five-validation confirmation "
            f"diagnostic; run budget={budget:,} tokens; scheduler=cosine; "
            "claim-ineligible; n=3 seeds"
        )
    return (
        "elastic_candidate; checkpoint=five-validation confirmation; "
        "run budget=3B; scheduler=cosine; n=3 seeds"
    )


def _plot_portfolio_ppl_vs_size(
    rows: list[Mapping[str, Any]],
    *,
    parameter_counts: Mapping[str, int],
    granularities: list[str],
    output_paths: list[Path],
    title: str,
    ylabel: str,
    elastic_label: str,
    perplexity_tolerance: float,
    dpi: int,
) -> list[Path]:
    from .reporting_impl import add_loss_secondary_axis

    style_config = resolve_plot_style("default")
    x_values = [int(parameter_counts[width]) for width in granularities]
    colors = {
        "standalone_reference": reporting_styles.STANDALONE_REFERENCE_COLOR,
        "elastic_candidate": plt.get_cmap("tab10")(0),
    }
    figure, axis = plt.subplots(figsize=(14, 5.2))
    means_by_role: dict[str, list[float]] = {}
    ranges_by_role: dict[str, list[list[float]]] = {}
    for role in ("standalone_reference", "elastic_candidate"):
        role_rows = [row for row in rows if row["role"] == role]
        by_seed_width = {
            (int(row["seed"]), str(row["width"])): float(row["perplexity"])
            for row in role_rows
        }
        seeds = sorted({int(row["seed"]) for row in role_rows})
        if len(seeds) != 3 or set(by_seed_width) != {
            (seed, width) for seed in seeds for width in granularities
        }:
            raise ValueError("Portfolio size plot requires the exact three-seed matrix")
        values_by_width = [
            [by_seed_width[(seed, width)] for seed in seeds]
            for width in granularities
        ]
        means = [statistics.fmean(values) for values in values_by_width]
        means_by_role[role] = means
        ranges_by_role[role] = values_by_width
        label = (
            "standalone_reference; checkpoint=best ordinary validation <=B; "
            "budget=B; scheduler=cosine; n=3 seeds"
            if role == "standalone_reference"
            else elastic_label
        )
        if role == "standalone_reference":
            axis.scatter(
                x_values,
                means,
                marker="^",
                s=58,
                color=colors[role],
                edgecolors=reporting_styles.STANDALONE_REFERENCE_EDGE_COLOR,
                linewidths=0.8,
                label=label,
                zorder=5,
            )
        else:
            axis.plot(
                x_values,
                means,
                color=colors[role],
                marker="o",
                linewidth=1.8,
                label=label,
                zorder=3,
            )
        axis.fill_between(
            x_values,
            [min(values) for values in values_by_width],
            [max(values) for values in values_by_width],
            color=colors[role],
            alpha=0.12 if role == "standalone_reference" else 0.16,
            linewidth=0,
        )
    standalone_means = means_by_role["standalone_reference"]
    axis.fill_between(
        x_values,
        standalone_means,
        [value * (1.0 + perplexity_tolerance) for value in standalone_means],
        color="tab:green",
        alpha=0.13,
        label=f"paired standalone +{100.0 * perplexity_tolerance:.1f}% PPL tolerance",
    )
    plotted_values = [
        value
        for values_by_width in ranges_by_role.values()
        for values in values_by_width
        for value in values
    ]
    plotted_values.extend(
        value * (1.0 + perplexity_tolerance) for value in standalone_means
    )
    y_limits = padded_limits(min(plotted_values), max(plotted_values))
    if y_limits[0] <= 0.0:
        y_limits = (min(plotted_values) * 0.92, y_limits[1])
    axis.set_ylim(*y_limits)
    axis.set(
        xlabel="Non-embedding parameters",
        ylabel=ylabel,
    )
    axis.set_title(
        "Nested-random · Slicing · Global sampling",
        fontsize=style_config["panel_title_fontsize"],
        pad=6,
    )
    axis.tick_params(labelsize=style_config["tick_label_fontsize"])
    axis.xaxis.label.set_size(style_config["axis_label_fontsize"])
    axis.yaxis.label.set_size(style_config["axis_label_fontsize"])
    axis.grid(True, alpha=0.3)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, fontsize=style_config["legend_fontsize"])
    add_loss_secondary_axis(axis)
    figure.suptitle(title, fontsize=style_config["figure_title_fontsize"])
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(figure)
    return output_paths


def _plot_portfolio_per_granularity_deficits(
    report: Mapping[str, Any],
    *,
    granularities: list[str],
    output_dir: Path,
    dpi: int,
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for axis, width in zip(axes.flat, granularities, strict=True):
        for seed_record in report["seeds"]:
            observations = seed_record.get("validation_observations", [])
            axis.plot(
                [row["tokens_seen"] for row in observations],
                [
                    100.0 * float(row["widths"][width]["perplexity_deficit"])
                    for row in observations
                ],
                label=f"seed {seed_record['seed']}",
            )
            if seed_record.get("confirmation_tokens") is not None:
                confirmation = next(
                    row
                    for row in observations
                    if row["tokens_seen"] == seed_record["confirmation_tokens"]
                )
                axis.scatter(
                    [seed_record["confirmation_tokens"]],
                    [
                        100.0
                        * float(
                            confirmation["widths"][width]["perplexity_deficit"]
                        )
                    ],
                    marker="*",
                    s=54,
                    color="tab:red",
                    zorder=4,
                )
        axis.axhline(
            100.0 * float(report["perplexity_tolerance"]),
            color="black",
            linestyle="--",
            label="0.5% threshold" if width == granularities[0] else None,
        )
        axis.set_yscale(
            "symlog",
            linthresh=100.0 * float(report["perplexity_tolerance"]),
        )
        axis.set_title(width)
        axis.set_ylabel("PPL deficit (%) · symlog")
    for axis in axes[-1]:
        axis.set_xlabel("optimizer tokens")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    path = output_dir / "portfolio_per_granularity_deficits.png"
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def _plot_portfolio_learning_rate_schedules(
    declared: list[Mapping[str, Any]],
    *,
    config_cache: Mapping[Path, Mapping[str, Any]],
    output_dir: Path,
    dpi: int,
) -> Path:
    from . import reporting_io

    role_configs: dict[str, list[tuple[Path, Mapping[str, Any]]]] = {}
    seen: set[tuple[str, Path]] = set()
    for entry in declared:
        role = str(entry["role"])
        config_path = Path(entry["run_dir"]) / "config.json"
        if (role, config_path) in seen:
            continue
        seen.add((role, config_path))
        role_configs.setdefault(role, []).append((config_path, config_cache[config_path]))

    schedules = []
    for role in ("standalone_reference", "elastic_candidate"):
        candidates = role_configs.get(role, [])
        if not candidates:
            raise ValueError(f"Portfolio comparison is missing scheduler role {role}")
        resolved = [
            reporting_io._learning_rate_schedule_from_saved_config(
                config,
                config_path=config_path,
            )
            for config_path, config in candidates
        ]
        schedule_contracts = {
            stable_hash(
                {
                    "scheduler_name": item.scheduler_name,
                    "scheduler_kwargs": item.scheduler_kwargs,
                    "peak_learning_rate": item.peak_learning_rate,
                    "warmup_steps": item.warmup_steps,
                    "max_steps": item.max_steps,
                    "expected_tokens_per_step": item.expected_tokens_per_step,
                    "token_budget": item.token_budget,
                    "scheduler_contract": item.scheduler_contract,
                }
            )
            for item in resolved
        }
        if len(schedule_contracts) != 1:
            raise ValueError(f"Portfolio {role} scheduler differs across runs")
        role_label = (
            "standalone_reference; active=one width; budget=B; n=3 seeds"
            if role == "standalone_reference"
            else "elastic_candidate; active=all four; budget=3B; n=3 seeds"
        )
        schedules.append(replace(resolved[0], run_id=role_label))
    return plot_learning_rate_schedules(
        schedules,
        output_dir / "learning_rate_schedule.png",
        dpi=dpi,
    )


def _plot_portfolio_final_holdout_if_complete(
    selection: Mapping[str, Any],
    output_dir: Path,
    *,
    parameter_counts: Mapping[str, int],
    granularities: list[str],
    perplexity_tolerance: float,
    dpi: int,
) -> list[Path]:
    entries = selection["entries"]
    result_paths = [Path(str(entry.get("result_path", ""))) for entry in entries]
    available_count = sum(path.is_file() for path in result_paths)
    if available_count == 0:
        return []
    if available_count != len(entries):
        warnings.warn(
            "Portfolio final-holdout figures require all 15 selected results; "
            f"found {available_count}/15",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    rows: list[dict[str, Any]] = []
    for entry in entries:
        result_path = Path(str(entry["result_path"])).expanduser().resolve()
        try:
            with result_path.open("r", encoding="utf-8") as source:
                result = json.load(source)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Cannot read manifest-scoped final-holdout result: {result_path}"
            ) from error
        if not isinstance(result, dict):
            raise ValueError("Manifest-scoped final-holdout result is malformed")
        result_body = dict(result)
        result_hash = result_body.pop("result_hash", None)
        if result_hash != stable_hash(result_body):
            raise ValueError("Manifest-scoped final-holdout result hash mismatch")
        if Path(str(result.get("checkpoint_path", ""))).resolve() != Path(
            str(entry["checkpoint_path"])
        ).resolve():
            raise ValueError("Manifest-scoped final-holdout result uses a stale checkpoint")
        if (
            result.get("checkpoint_sha256") != entry.get("checkpoint_sha256")
            or result.get("run_id") != entry.get("run_id")
            or result.get("final_holdout_manifest_hash")
            != selection.get("final_holdout_manifest_hash")
        ):
            raise ValueError("Manifest-scoped final-holdout result provenance mismatch")
        components = result.get("ordered_per_granularity_losses")
        if not isinstance(components, list):
            raise ValueError("Manifest-scoped final-holdout result lacks components")
        expected_widths = [str(value) for value in entry["granularities"]]
        component_widths = [str(component.get("granularity")) for component in components]
        if component_widths != expected_widths:
            raise ValueError("Manifest-scoped final-holdout result width order differs")
        for component in components:
            perplexity = float(component["perplexity"])
            if not math.isfinite(perplexity) or perplexity <= 0.0:
                raise ValueError("Manifest-scoped final-holdout perplexity is invalid")
            rows.append(
                {
                    "role": str(entry["comparison_role"]),
                    "seed": int(entry["seed"]),
                    "width": str(component["granularity"]),
                    "perplexity": perplexity,
                    "checkpoint_selection": str(entry["checkpoint_selection"]),
                }
            )

    elastic_label = _portfolio_elastic_selection_label(selection)
    size_paths = _plot_portfolio_ppl_vs_size(
        rows,
        parameter_counts=parameter_counts,
        granularities=granularities,
        output_paths=[
            output_dir / "final_holdout_ppl_vs_size.png",
            output_dir / "portfolio_final_holdout_perplexity.png",
        ],
        title="Final-holdout perplexity at manifest-selected checkpoints",
        ylabel="Final holdout perplexity",
        elastic_label=elastic_label,
        perplexity_tolerance=perplexity_tolerance,
        dpi=dpi,
    )

    by_key = {
        (str(row["role"]), int(row["seed"]), str(row["width"])): float(
            row["perplexity"]
        )
        for row in rows
    }
    expected_keys = {
        (role, int(seed), width)
        for role in ("standalone_reference", "elastic_candidate")
        for seed in (42, 43, 44)
        for width in granularities
    }
    if set(by_key) != expected_keys:
        raise ValueError("Portfolio final holdout does not contain the exact matrix")
    x_values = [parameter_counts[width] for width in granularities]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    deficits_by_width: list[list[float]] = []
    for seed in (42, 43, 44):
        deficits = [
            100.0
            * (
                by_key[("elastic_candidate", seed, width)]
                / by_key[("standalone_reference", seed, width)]
                - 1.0
            )
            for width in granularities
        ]
        deficits_by_width.append(deficits)
        axis.plot(
            x_values,
            deficits,
            marker="o",
            linewidth=1.0,
            alpha=0.5,
            label=f"seed {seed}",
        )
    mean_deficits = [
        statistics.fmean(values)
        for values in zip(*deficits_by_width, strict=True)
    ]
    axis.plot(
        x_values,
        mean_deficits,
        color="black",
        marker="o",
        linewidth=2.0,
        label="three-seed mean",
    )
    axis.axhline(
        100.0 * perplexity_tolerance,
        color="tab:red",
        linestyle="--",
        label=f"{100.0 * perplexity_tolerance:.1f}% tolerance",
    )
    axis.set_xticks(
        x_values,
        [f"{width}\n{count:,}" for width, count in zip(granularities, x_values)],
    )
    axis.set(
        xlabel="non-embedding parameters (granularity and exact count)",
        ylabel="elastic minus same-seed standalone PPL (%)",
        title=f"Final-holdout paired deficit · {elastic_label}",
    )
    axis.grid(True, alpha=0.22)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    deficit_path = output_dir / "portfolio_final_holdout_deficit_vs_size.png"
    figure.savefig(deficit_path, dpi=dpi)
    plt.close(figure)
    return [*size_paths, deficit_path]


def generate_figures(
    input_root: str | Path,
    output_dir: str | Path,
    refresh_counts: bool = True,
    dpi: int = 300,
    validation_loss_log_y: bool = False,
    include_incomplete_validation_traces: bool = False,
    include_final_holdout: bool = True,
    variants: list[str] | tuple[str, ...] | None = None,
    corrections: list[str] | tuple[str, ...] | None = None,
    sampling_bin_steps: int = 50,
    sampling_zoom_steps: int | None = None,
    include_individual_size_panels: bool = False,
    comparison_manifest: str | Path | None = None,
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
    if comparison_manifest is not None:
        return _generate_portfolio_comparison_figures(
            input_root,
            Path(comparison_manifest),
            output_dir,
            dpi=dpi,
            include_final_holdout=include_final_holdout,
            refresh_counts=refresh_counts,
            validation_loss_log_y=validation_loss_log_y,
        )
    _remove_stale_generator_artifacts(
        output_dir,
        sampling_zoom_steps=sampling_zoom_steps,
    )

    figure_paths: list[Path] = []
    learning_rate_schedules = []
    for schedule in reporting_io.iter_learning_rate_schedules(input_root):
        keep = filter_plot_rows(
            [
                {
                    "model_variant": schedule.model_variant,
                    "correction_mode": schedule.correction_mode,
                    "membership_correction": schedule.membership_correction,
                }
            ],
            variants=variants,
            corrections=corrections,
        )
        if keep:
            learning_rate_schedules.append(schedule)
    if learning_rate_schedules:
        try:
            figure_paths.append(
                plot_learning_rate_schedules(
                    learning_rate_schedules,
                    output_dir / "learning_rate_schedule.png",
                    dpi=dpi,
                )
            )
        except ValueError as error:
            warnings.warn(
                f"Could not generate configured LR schedule plot: {error}",
                RuntimeWarning,
                stacklevel=2,
            )

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

    final_holdout_rows = (
        reporting_io.read_final_holdout_scaling_rows(input_root)
        if include_final_holdout
        else []
    )
    final_holdout_rows = filter_plot_rows(
        final_holdout_rows,
        variants=variants,
        corrections=corrections,
    )
    if refresh_counts and final_holdout_rows:
        final_holdout_rows = reporting_io.refresh_scaling_parameter_counts(
            input_root,
            final_holdout_rows,
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
                    include_individual_panels=include_individual_size_panels,
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
                    include_individual_panels=include_individual_size_panels,
                    dpi=dpi,
                )
            )
        figure_paths.append(
            write_medium_trend_report(
                scaling_rows,
                output_dir / "medium_trend_report.md",
            )
        )

    if final_holdout_rows:
        figure_paths.extend(
            plot_metric_vs_size(
                final_holdout_rows,
                metric_name="perplexity",
                ylabel="Final holdout perplexity",
                output_path=output_dir / "final_holdout_ppl_vs_size.png",
                panel_specs=reporting_styles.SIZE_PLOT_PANELS_WITH_SAMPLING,
                figure_title=("Final holdout perplexity vs Non-embedding parameters"),
                style="default",
                figure_alias="all",
                include_individual_panels=include_individual_size_panels,
                dpi=dpi,
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

    figure_paths.extend(
        generate_gradient_interference_figures(
            input_root,
            output_dir,
            dpi=dpi,
            variants=variants,
            corrections=corrections,
        )
    )

    return figure_paths


def _remove_stale_generator_artifacts(
    output_dir: Path,
    *,
    sampling_zoom_steps: int | None,
) -> None:
    """Remove superseded figure-generator outputs before rendering."""

    stale_patterns = [
        "gradient_interference_*.png",
        "global_sampling_exposure_comparison__*.png",
        "global_sampling_cumulative_comparison__*.png",
        "global_sampling_policy_summary__*.csv",
        "loss_vs_size.png",
        "loss_vs_size__*.png",
        "final_holdout_loss_vs_size*.png",
        "final_holdout_ppl_vs_size*.png",
        "learning_rate_schedule.png",
        "selected_granularity_over_tokens_*.png",
        "selected_granularity_share_over_tokens_*.png",
    ]
    size_output_names = [
        spec["output_name"] for spec in reporting_styles.PPL_VS_SIZE_FIGURE_SPECS
    ] + ["accuracy_vs_size.png"]
    stale_patterns.extend(
        f"{Path(output_name).stem}__*.png" for output_name in size_output_names
    )
    if sampling_zoom_steps is None:
        stale_patterns.append("selected_granularity_zoom_*.png")

    for pattern in stale_patterns:
        for path in output_dir.glob(pattern):
            path.unlink(missing_ok=True)


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
        type=_positive_cli_int,
        default=None,
        help="Emit exact-action zooms using this many steps at each end.",
    )
    parser.add_argument(
        "--individual-size-panels",
        action="store_true",
        help="Also emit one companion PNG per PPL and accuracy size panel.",
    )
    return parser.parse_args(argv)


def _positive_cli_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


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
        include_individual_size_panels=args.individual_size_panels,
    )
    for path in figure_paths:
        print(path)
