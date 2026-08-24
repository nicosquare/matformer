"""Generate plots from structured CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any
from collections.abc import Callable
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.collections import PolyCollection

from src.evaluation import reporting_styles


PARAMETER_COUNT_FIELDS = [
    "total_parameters",
    "embedding_parameters",
    "lm_head_parameters",
    "non_embedding_parameters",
    "ffn_parameters",
    "attention_parameters",
    "other_non_embedding_parameters",
    "lm_head_counting",
]

LOSS_MOVING_AVERAGE_FRACTION = 0.1
STANDALONE_REFERENCE_COLOR = reporting_styles.STANDALONE_REFERENCE_COLOR
STANDALONE_REFERENCE_EDGE_COLOR = reporting_styles.STANDALONE_REFERENCE_EDGE_COLOR
SIZE_PLOT_PANELS_DEFAULT = [
    ("nested-random", "slicing", None),
    ("nested-random", "concat", None),
    ("nested-all", "slicing", None),
    ("nested-all", "concat", None),
]
SIZE_PLOT_PANELS_WITH_SAMPLING = [
    ("nested-random", "slicing", "global"),
    ("nested-random", "concat", "global"),
    ("nested-random", "slicing", "uniform_global_window"),
    ("nested-random", "concat", "uniform_global_window"),
    ("nested-random", "slicing", "balanced_global_window"),
    ("nested-random", "concat", "balanced_global_window"),
    ("nested-random", "slicing", "fixed_global"),
    ("nested-random", "concat", "fixed_global"),
    ("nested-random", "slicing", "per_block"),
    ("nested-random", "concat", "per_block"),
    ("nested-random", "slicing", "probabilistic_global_thompson"),
    ("nested-random", "concat", "probabilistic_global_thompson"),
    ("nested-random", "slicing", "panelgrad_global"),
    ("nested-random", "concat", "panelgrad_global"),
    ("nested-random", "slicing", "probabilistic_per_block_thompson"),
    ("nested-random", "concat", "probabilistic_per_block_thompson"),
    ("nested-random", "slicing", "adaptive_per_block_ucb"),
    ("nested-random", "concat", "adaptive_per_block_ucb"),
    ("nested-all", "slicing", None),
    ("nested-all", "concat", None),
]
SCALING_GROUP_COLORS = {
    "nested-random / slicing / global": "tab:blue",
    "nested-random / slicing / balanced_global_window": "tab:purple",
    "nested-random / slicing / fixed_global": "tab:green",
    "nested-random / slicing / per_block": "tab:cyan",
    "nested-random / slicing / probabilistic_global_thompson": "tab:blue",
    "nested-random / slicing / probabilistic_global_thompson_reset": "tab:purple",
    "nested-random / slicing / probabilistic_global_thompson_acquisition_only": "tab:green",
    "nested-random / slicing / panelgrad_global": "tab:purple",
    "nested-random / slicing / probabilistic_per_block_thompson": "tab:cyan",
    "nested-random / slicing / adaptive_per_block_ucb": "tab:olive",
    "nested-random / concat / global": "tab:orange",
    "nested-random / concat / balanced_global_window": "tab:brown",
    "nested-random / concat / fixed_global": "tab:olive",
    "nested-random / concat / per_block": "tab:red",
    "nested-random / concat / probabilistic_global_thompson": "tab:orange",
    "nested-random / concat / probabilistic_global_thompson_reset": "tab:brown",
    "nested-random / concat / probabilistic_global_thompson_acquisition_only": "tab:olive",
    "nested-random / concat / panelgrad_global": "tab:brown",
    "nested-random / concat / probabilistic_per_block_thompson": "tab:red",
    "nested-random / concat / adaptive_per_block_ucb": "tab:pink",
    "nested-all / slicing": "tab:purple",
    "nested-all / concat": "tab:green",
    "standalone": STANDALONE_REFERENCE_COLOR,
}
SCALING_CORRECTION_STYLES = {
    "none": {"linestyle": "-", "marker": "o", "shade": 0.0},
    "gmc": {"linestyle": "--", "marker": "s", "shade": 0.2},
    "lmc": {"linestyle": "-.", "marker": "^", "shade": 0.35},
}
SCALING_SAMPLING_TONES = {
    "global": 0.0,
    "balanced_global_window": 0.16,
    "fixed_global": 0.12,
    "per_block": 0.28,
    "probabilistic_global_thompson": 0.16,
    "probabilistic_global_thompson_reset": 0.24,
    "probabilistic_global_thompson_acquisition_only": 0.20,
    "panelgrad_global": 0.42,
    "probabilistic_per_block_thompson": 0.34,
    "adaptive_per_block_ucb": 0.55,
}
SCALING_SAMPLING_MARKERS = {
    "global": "o",
    "balanced_global_window": "D",
    "fixed_global": "H",
    "per_block": "D",
    "probabilistic_global_thompson": "*",
    "probabilistic_global_thompson_reset": "P",
    "probabilistic_global_thompson_acquisition_only": "h",
    "panelgrad_global": "d",
    "probabilistic_per_block_thompson": "v",
    "adaptive_per_block_ucb": "X",
}

BAYESIAN_CONTROLLER_METHOD_FAMILY = "bayesian_gaussian_linear_thompson"
PANELGRAD_METHOD_FAMILIES = {
    "panelgrad_gradient_rms",
    "panelgrad_gradient_l2",
}

PLOT_STYLE_BASE = {
    "figure_title_fontsize": 17,
    "panel_title_fontsize": 12,
    "subfigure_title_fontsize": 13,
    "axis_label_fontsize": 11,
    "tick_label_fontsize": 10,
    "legend_fontsize": 11,
    "standalone_label": "standalone reference",
    "series_colors": SCALING_GROUP_COLORS,
    "series_aliases": {},
    "comparison_linestyle": None,
    "comparison_markers_by_variant": {},
    "curve_aliases": {},
}
PLOT_STYLE_PRESETS = {
    "default": {},
    # These presets keep the existing rendering behavior but expose the knobs
    # in one place so the figure script can be tuned without hunting through
    # the plotting code.
    "nested_all_no_corrections": {
        "figure_title_fontsize": 15,
        "curve_aliases": {
            "nested-all / slicing": "nested-all / slicing",
            "nested-all / concat": "nested-all / concat",
        },
        "series_colors": {
            "nested-all / slicing": "tab:blue",
            "nested-all / concat": "tab:orange",
            "standalone": STANDALONE_REFERENCE_COLOR,
        },
    },
    "nested_random_no_corrections": {
        "figure_title_fontsize": 15,
        "curve_aliases": {
            "nested-random / slicing / global": "nested-random / slicing / global",
            "nested-random / concat / global": "nested-random / concat / global",
            "nested-random / slicing / per_block": "nested-random / slicing / per_block",
            "nested-random / concat / per_block": "nested-random / concat / per_block",
            "nested-random / slicing / panelgrad_global": "nested-random / slicing / PanelGrad global",
            "nested-random / concat / panelgrad_global": "nested-random / concat / PanelGrad global",
            "nested-random / slicing / adaptive_per_block_ucb": "nested-random / slicing / adaptive_per_block_ucb",
            "nested-random / concat / adaptive_per_block_ucb": "nested-random / concat / adaptive_per_block_ucb",
        },
        "series_colors": {
            "nested-random / slicing / global": "tab:blue",
            "nested-random / concat / global": "tab:orange",
            "nested-random / slicing / per_block": "tab:cyan",
            "nested-random / concat / per_block": "tab:red",
            "nested-random / slicing / panelgrad_global": "tab:purple",
            "nested-random / concat / panelgrad_global": "tab:brown",
            "nested-random / slicing / adaptive_per_block_ucb": "tab:olive",
            "nested-random / concat / adaptive_per_block_ucb": "tab:pink",
            "standalone": STANDALONE_REFERENCE_COLOR,
        },
    },
    "nested_split_no_corrections": {
        "figure_title_fontsize": 17,
        "subfigure_title_fontsize": 13,
        "legend_fontsize": 12,
        "comparison_linestyle": "-",
        "comparison_markers_by_variant": {
            "slicing": "s",
            "concat": "o",
        },
        "series_aliases": {
            "standalone": "Individual",
            "nested-random / slicing / none / global": "Slicing",
            "nested-random / concat / none / global": "Concat",
            "nested-random / slicing / none / panelgrad_global": "Slicing / PanelGrad",
            "nested-random / concat / none / panelgrad_global": "Concat / PanelGrad",
            "nested-random / concat / lmc": "Concat/LMC",
            "nested-random / concat / gmc": "Concat/GMC",
            "nested-all / slicing / none / global": "Slicing",
            "nested-all / concat / none / global": "Concat",
            "nested-all / concat / lmc": "Concat/LMC",
            "nested-all / concat / gmc": "Concat/GMC",
        },
        "series_colors": {
            "standalone": STANDALONE_REFERENCE_COLOR,
            "nested-random / slicing / none / global": "tab:red",
            "nested-random / concat / none / global": "tab:blue",
            "nested-random / slicing / none / panelgrad_global": "tab:purple",
            "nested-random / concat / none / panelgrad_global": "tab:brown",
            "nested-random / concat / lmc": "tab:purple",
            "nested-random / concat / gmc": "tab:green",
            "nested-all / slicing / none / global": "tab:red",
            "nested-all / concat / none / global": "tab:blue",
            "nested-all / concat / lmc": "tab:purple",
            "nested-all / concat / gmc": "tab:green",
        },
    },
}
PPL_VS_SIZE_FIGURE_SPECS = [
    {
        "output_name": "ppl_vs_size.png",
        "figure_title": "Perplexity vs Non-embedding parameters",
        "figure_alias": "all",
        "panel_specs": SIZE_PLOT_PANELS_WITH_SAMPLING,
        "style": "default",
        "row_filter_name": None,
    },
    {
        "output_name": "ppl_vs_size_balanced_global_window.png",
        "figure_title": "Balanced global windows: perplexity vs non-embedding parameters",
        "figure_alias": "balanced_global_window",
        "panel_specs": [
            ("nested-random", "slicing", "balanced_global_window"),
            ("nested-random", "concat", "balanced_global_window"),
        ],
        "style": "default",
        "row_filter_name": None,
    },
    {
        "output_name": "ppl_vs_size_nested_all_no_corrections.png",
        "figure_title": "Perplexity vs Non-embedding parameters: nested-all, no corrections",
        "figure_alias": "nested_all",
        "panel_specs": [
            ("nested-all", "slicing", None),
            ("nested-all", "concat", None),
        ],
        "style": "nested_all_no_corrections",
        "row_filter_name": "no_corrections",
    },
    {
        "output_name": "ppl_vs_size_nested_random_no_corrections.png",
        "figure_title": "Perplexity vs Non-embedding parameters: nested-random, no corrections",
        "figure_alias": "nested_random",
        "panel_specs": [
            ("nested-random", "slicing", None),
            ("nested-random", "concat", None),
        ],
        "style": "nested_random_no_corrections",
        "row_filter_name": "no_corrections",
    },
]
PPL_VS_SIZE_SPLIT_FIGURE_SPEC = {
    "output_name": "ppl_vs_size_nested_random_vs_nested_all_no_corrections.png",
    "figure_title": "Perplexity vs Non-embedding parameters: nested-random and nested-all, no corrections",
    "style": "nested_split_no_corrections",
    "left": {
        "subfigure_title": "One width per batch",
        "series_keys": [
            "standalone",
            "nested-random / slicing / none / global",
            "nested-random / concat / none / global",
            "nested-random / slicing / none / panelgrad_global",
            "nested-random / concat / none / panelgrad_global",
            "nested-random / concat / lmc",
            "nested-random / concat / gmc",
        ],
    },
    "right": {
        "subfigure_title": "All widths per batch",
        "series_keys": [
            "standalone",
            "nested-all / slicing / none / global",
            "nested-all / concat / none / global",
            "nested-all / concat / lmc",
            "nested-all / concat / gmc",
        ],
    },
}


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
    parser.add_argument("--sampling-bin-steps", type=int, default=50)
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


def _legacy_generate_figures(
    input_root: str | Path,
    output_dir: str | Path,
    refresh_counts: bool = True,
    dpi: int = 300,
    validation_loss_log_y: bool = False,
    include_incomplete_validation_traces: bool = False,
    variants: list[str] | tuple[str, ...] | None = None,
    corrections: list[str] | tuple[str, ...] | None = None,
    sampling_bin_steps: int = 50,
    sampling_zoom_steps: int | None = None,
    include_individual_size_panels: bool = False,
) -> list[Path]:
    input_root = Path(input_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = []
    scaling_rows = read_csv_artifacts(input_root, "scaling_results.csv")
    scaling_rows = enrich_scaling_metadata_from_run_config(input_root, scaling_rows)
    scaling_rows = filter_plot_rows(
        scaling_rows,
        variants=variants,
        corrections=corrections,
    )
    if refresh_counts:
        scaling_rows = refresh_scaling_parameter_counts(input_root, scaling_rows)
    task_result_rows = read_csv_artifacts(input_root, "task_results.csv")
    consistency_rows = read_csv_artifacts(input_root, "consistency_results.csv")
    consistency_rows = enrich_metrics_metadata_from_run_config(
        input_root,
        consistency_rows,
    )
    consistency_rows = filter_plot_rows(
        consistency_rows,
        variants=variants,
        corrections=corrections,
    )

    if scaling_rows and task_result_rows:
        from src.evaluation.validation import aggregate_scaling_summary

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
        metrics_rows = read_csv_artifacts_filtered(
            input_root,
            "metrics.csv",
            row_filter=validation_split_filter,
        )
        metrics_rows = enrich_metrics_metadata_from_run_config(input_root, metrics_rows)
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
        metrics_rows = read_csv_artifacts(input_root, "metrics.csv")
        metrics_rows = enrich_metrics_metadata_from_run_config(input_root, metrics_rows)
        metrics_rows = filter_plot_rows(
            metrics_rows,
            variants=variants,
            corrections=corrections,
        )
        validation_metrics_rows = [
            row for row in metrics_rows if validation_split_filter(row)
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

    from src.evaluation.reporting import (
        controller_selection_frequency_filename,
        controller_selection_share_filename,
        controller_timeline_filename,
        generate_global_sampling_policy_figures,
        plot_granularity_selection_frequency_over_tokens,
        plot_selected_granularity_over_tokens,
        plot_selected_granularity_share_over_tokens,
    )
    from src.evaluation.reporting_io import iter_controller_granularity_timelines

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

    for timeline in iter_controller_granularity_timelines(input_root):
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

    return figure_paths


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
    sampling_zoom_steps: int | None = None,
    include_individual_size_panels: bool = False,
) -> list[Path]:
    """Compatibility entrypoint forwarding to the canonical generator."""

    from src.evaluation.reporting import generate_figures as canonical_generate_figures

    return canonical_generate_figures(
        input_root,
        output_dir,
        refresh_counts=refresh_counts,
        dpi=dpi,
        validation_loss_log_y=validation_loss_log_y,
        include_incomplete_validation_traces=include_incomplete_validation_traces,
        variants=variants,
        corrections=corrections,
        sampling_bin_steps=sampling_bin_steps,
        sampling_zoom_steps=sampling_zoom_steps,
        include_individual_size_panels=include_individual_size_panels,
    )


def read_csv_artifacts(input_root: Path, filename: str) -> list[dict[str, str]]:
    return read_csv_artifacts_filtered(input_root, filename, row_filter=None)


def read_csv_artifacts_filtered(
    input_root: Path,
    filename: str,
    row_filter: Any | None,
) -> list[dict[str, str]]:
    rows = []
    for path in sorted(input_root.rglob(filename)):
        with path.open("r", encoding="utf-8", newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                if row_filter is not None and not row_filter(row):
                    continue
                row["_source_csv"] = str(path)
                rows.append(row)
    return rows


def validation_split_filter(row: dict[str, str]) -> bool:
    return str(row.get("split") or "") == "validation"


def refresh_scaling_parameter_counts(
    input_root: Path,
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    count_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    refreshed_rows = []

    for row in rows:
        refreshed_row = dict(row)
        config_path = config_path_for_scaling_row(input_root, row)
        granularity = str(row.get("granularity") or "")
        if config_path is not None and granularity:
            if config_path not in count_cache:
                count_cache[config_path] = recompute_parameter_counts(config_path)
            counts = count_cache[config_path].get(granularity)
            if counts is not None:
                for field_name in reporting_styles.PARAMETER_COUNT_FIELDS:
                    refreshed_row[field_name] = counts.get(field_name)
        refreshed_rows.append(refreshed_row)

    return refreshed_rows


def enrich_scaling_metadata_from_run_config(
    input_root: Path,
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    config_cache: dict[Path, dict[str, Any]] = {}
    enriched_rows = []

    for row in rows:
        enriched_row = dict(row)
        config_path = config_path_for_scaling_row(input_root, row)
        if config_path is not None:
            if config_path not in config_cache:
                with config_path.open("r", encoding="utf-8") as config_file:
                    config_cache[config_path] = json.load(config_file)
            model_variant = model_variant_from_saved_config(config_cache[config_path])
            if model_variant not in (None, ""):
                enriched_row["model_variant"] = str(model_variant)
            resolved_sampling_mode = resolved_sampling_mode_from_saved_config(
                config_cache[config_path]
            )
            if resolved_sampling_mode is not None:
                enriched_row["resolved_sampling_mode"] = resolved_sampling_mode
            granularity_sampling_mode = granularity_sampling_mode_from_saved_config(
                config_cache[config_path]
            )
            if granularity_sampling_mode is not None:
                enriched_row["granularity_sampling_mode"] = granularity_sampling_mode
            global_sampling_interval_steps = (
                global_sampling_interval_steps_from_saved_config(
                    config_cache[config_path]
                )
            )
            if global_sampling_interval_steps is not None:
                enriched_row["global_sampling_interval_steps"] = (
                    global_sampling_interval_steps
                )
            global_sampling_schedule = global_sampling_schedule_from_saved_config(
                config_cache[config_path]
            )
            if global_sampling_schedule is not None:
                enriched_row["global_sampling_schedule"] = global_sampling_schedule
            global_sampling_schedule_version = (
                global_sampling_schedule_version_from_saved_config(
                    config_cache[config_path]
                )
            )
            if global_sampling_schedule_version is not None:
                enriched_row["global_sampling_schedule_version"] = (
                    global_sampling_schedule_version
                )
            membership_correction = membership_correction_from_saved_config(
                config_cache[config_path]
            )
            if membership_correction is not None:
                enriched_row["membership_correction"] = membership_correction
            correction_mode = correction_mode_from_saved_config(
                config_cache[config_path]
            )
            if correction_mode is not None:
                enriched_row["correction_mode"] = correction_mode
            adaptive_sampler_strategy = adaptive_sampler_strategy_from_saved_config(
                config_cache[config_path]
            )
            if adaptive_sampler_strategy is not None:
                enriched_row["adaptive_sampler_strategy"] = adaptive_sampler_strategy
            _enrich_controller_provenance(
                enriched_row,
                config_cache[config_path],
            )
        enriched_rows.append(enriched_row)

    return enriched_rows


def enrich_metrics_metadata_from_run_config(
    input_root: Path,
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    config_cache: dict[Path, dict[str, Any]] = {}
    enriched_rows = []

    for row in rows:
        enriched_row = dict(row)
        config_path = config_path_for_scaling_row(input_root, row)
        if config_path is not None:
            if config_path not in config_cache:
                with config_path.open("r", encoding="utf-8") as config_file:
                    config_cache[config_path] = json.load(config_file)
            model_variant = model_variant_from_saved_config(config_cache[config_path])
            if model_variant not in (None, ""):
                enriched_row["model_variant"] = str(model_variant)
            resolved_sampling_mode = resolved_sampling_mode_from_saved_config(
                config_cache[config_path]
            )
            if resolved_sampling_mode is not None:
                enriched_row["resolved_sampling_mode"] = resolved_sampling_mode
            granularity_sampling_mode = granularity_sampling_mode_from_saved_config(
                config_cache[config_path]
            )
            if granularity_sampling_mode is not None:
                enriched_row["granularity_sampling_mode"] = granularity_sampling_mode
            global_sampling_interval_steps = (
                global_sampling_interval_steps_from_saved_config(
                    config_cache[config_path]
                )
            )
            if global_sampling_interval_steps is not None:
                enriched_row["global_sampling_interval_steps"] = (
                    global_sampling_interval_steps
                )
            global_sampling_schedule = global_sampling_schedule_from_saved_config(
                config_cache[config_path]
            )
            if global_sampling_schedule is not None:
                enriched_row["global_sampling_schedule"] = global_sampling_schedule
            global_sampling_schedule_version = (
                global_sampling_schedule_version_from_saved_config(
                    config_cache[config_path]
                )
            )
            if global_sampling_schedule_version is not None:
                enriched_row["global_sampling_schedule_version"] = (
                    global_sampling_schedule_version
                )
            membership_correction = membership_correction_from_saved_config(
                config_cache[config_path]
            )
            if membership_correction is not None:
                enriched_row["membership_correction"] = membership_correction
            correction_mode = correction_mode_from_saved_config(
                config_cache[config_path]
            )
            if correction_mode is not None:
                enriched_row["correction_mode"] = correction_mode
            adaptive_sampler_strategy = adaptive_sampler_strategy_from_saved_config(
                config_cache[config_path]
            )
            if adaptive_sampler_strategy is not None:
                enriched_row["adaptive_sampler_strategy"] = adaptive_sampler_strategy
            _enrich_controller_provenance(
                enriched_row,
                config_cache[config_path],
            )
        enriched_rows.append(enriched_row)

    return enriched_rows


def config_path_for_scaling_row(
    input_root: Path,
    row: dict[str, str],
) -> Path | None:
    source_csv = row.get("_source_csv")
    if source_csv:
        candidate = Path(source_csv).parent / "config.json"
        if candidate.exists():
            return candidate

    run_id = row.get("run_id")
    if run_id:
        candidates = sorted(input_root.rglob(f"{run_id}/config.json"))
        if candidates:
            return candidates[0]

    return None


def recompute_parameter_counts(config_path: Path) -> dict[str, dict[str, Any]]:
    from src.training.modeling import build_model
    from src.utils.metrics import build_parameter_counts_by_granularity

    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    config = with_default_model_variant(config)

    model = build_model(config)
    try:
        return build_parameter_counts_by_granularity(
            model,
            config["model"]["granularities"],
        )
    finally:
        del model


def model_variant_from_saved_config(config: dict[str, Any]) -> str | None:
    model = config.get("model")
    if not isinstance(model, dict):
        return None
    variant = model.get("variant")
    if variant in (None, ""):
        return "matformer_llama"
    return str(variant)


def membership_correction_from_saved_config(config: dict[str, Any]) -> bool | None:
    model = config.get("model")
    if not isinstance(model, dict):
        return None
    value = model.get("membership_correction")
    if value in (None, ""):
        value = model.get("gradient_membership_correction")
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def correction_mode_from_saved_config(config: dict[str, Any]) -> str | None:
    model = config.get("model")
    if not isinstance(model, dict):
        return None
    value = model.get("correction_mode")
    if value in (None, ""):
        value = model.get("requested_correction_mode")
    if value in (None, ""):
        return None
    return str(value).strip().lower()


def adaptive_sampler_strategy_from_saved_config(config: dict[str, Any]) -> str | None:
    model = config.get("model")
    if not isinstance(model, dict):
        return None
    value = model.get("adaptive_sampler_strategy")
    if value in (None, ""):
        return None
    return str(value).strip().lower()


def _adaptive_controller_from_saved_config(
    config: dict[str, Any],
) -> dict[str, Any] | None:
    model = config.get("model")
    if not isinstance(model, dict):
        return None
    controller = (
        model.get("panelgrad")
        if model.get("adaptive_sampler_strategy") == "panelgrad"
        else model.get("adaptive_controller")
    )
    return controller if isinstance(controller, dict) else None


def controller_method_family_from_saved_config(
    config: dict[str, Any],
) -> str | None:
    controller = _adaptive_controller_from_saved_config(config)
    if controller is None or controller.get("method_family") in (None, ""):
        return None
    return str(controller["method_family"]).strip().lower()


def controller_method_version_from_saved_config(
    config: dict[str, Any],
) -> Any | None:
    controller = _adaptive_controller_from_saved_config(config)
    if controller is None or controller.get("method_version") in (None, ""):
        return None
    return controller["method_version"]


def controller_scope_from_saved_config(config: dict[str, Any]) -> str | None:
    controller = _adaptive_controller_from_saved_config(config)
    if controller is None or controller.get("scope") in (None, ""):
        return None
    return str(controller["scope"]).strip().lower()


def controller_reset_enabled_from_saved_config(
    config: dict[str, Any],
) -> bool | None:
    controller = _adaptive_controller_from_saved_config(config)
    if controller is None:
        return None
    reset = controller.get("reset")
    if not isinstance(reset, dict):
        return None
    return bool(reset.get("enabled", False))


def controller_reset_policy_from_saved_config(
    config: dict[str, Any],
) -> str | None:
    controller = _adaptive_controller_from_saved_config(config)
    if controller is None:
        return None
    reset = controller.get("reset")
    if not isinstance(reset, dict) or reset.get("policy") in (None, ""):
        return None
    return str(reset["policy"]).strip().lower()


def _enrich_controller_provenance(
    row: dict[str, Any],
    config: dict[str, Any],
) -> None:
    provenance = {
        "controller_method_family": controller_method_family_from_saved_config(config),
        "controller_method_version": controller_method_version_from_saved_config(
            config
        ),
        "controller_scope": controller_scope_from_saved_config(config),
        "controller_reset_enabled": controller_reset_enabled_from_saved_config(config),
        "controller_reset_policy": controller_reset_policy_from_saved_config(config),
    }
    for field_name, value in provenance.items():
        if value not in (None, ""):
            row[field_name] = value


def resolved_sampling_mode_from_saved_config(config: dict[str, Any]) -> str | None:
    model = config.get("model")
    if not isinstance(model, dict):
        return None
    value = model.get("resolved_sampling_mode")
    if value in (None, ""):
        value = model.get("granularity_sampling_mode")
    if value in (None, ""):
        return None
    return str(value).strip().lower()


def granularity_sampling_mode_from_saved_config(config: dict[str, Any]) -> str | None:
    model = config.get("model")
    if not isinstance(model, dict):
        return None
    value = model.get("granularity_sampling_mode")
    if value in (None, ""):
        return None
    return str(value).strip().lower()


def global_sampling_interval_steps_from_saved_config(
    config: dict[str, Any],
) -> int | None:
    model = config.get("model")
    if not isinstance(model, dict):
        return None
    value = model.get("global_sampling_interval_steps")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def global_sampling_schedule_from_saved_config(
    config: dict[str, Any],
) -> str | None:
    model = config.get("model")
    if not isinstance(model, dict):
        return None
    value = model.get("global_sampling_schedule", "random_with_replacement")
    if value not in {"random_with_replacement", "balanced_cycle"}:
        return None
    return str(value)


def global_sampling_schedule_version_from_saved_config(
    config: dict[str, Any],
) -> int | None:
    model = config.get("model")
    if not isinstance(model, dict):
        return None
    value = model.get("global_sampling_schedule_version")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return int(value)


def with_default_model_variant(config: dict[str, Any]) -> dict[str, Any]:
    normalized_config = json.loads(json.dumps(config))
    model = normalized_config.setdefault("model", {})
    if isinstance(model, dict) and model.get("variant") in (None, ""):
        model["variant"] = "matformer_llama"
    return normalized_config


def plot_metric_vs_size(
    rows: list[dict[str, str]],
    metric_name: str,
    ylabel: str,
    output_path: Path,
    panel_specs: list[tuple[str, str, str | None]] | None = None,
    row_filter: Callable[[dict[str, str]], bool] | None = None,
    figure_title: str | None = None,
    style: str = "default",
    figure_alias: str | None = None,
    include_individual_panels: bool = False,
    dpi: int = 300,
) -> list[Path]:
    panel_specs = panel_specs or reporting_styles.SIZE_PLOT_PANELS_DEFAULT
    style_config = resolve_plot_style(style)
    plot_rows = rows if row_filter is None else [row for row in rows if row_filter(row)]
    available_panel_specs = [
        panel_spec
        for panel_spec in panel_specs
        if metric_panel_has_numeric_points(
            plot_rows,
            metric_name=metric_name,
            panel_spec=panel_spec,
        )
    ]
    if not available_panel_specs:
        return []

    column_count = 2 if len(available_panel_specs) > 1 else 1
    row_count = math.ceil(len(available_panel_specs) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(14, 5.2 * row_count),
        sharex=True,
        sharey=False,
    )
    axes_list = flatten_axes(axes)

    for axis, (sampling_mode, variant_label, sampling_label) in zip(
        axes_list,
        available_panel_specs,
    ):
        plot_metric_vs_size_panel(
            axis,
            plot_rows,
            metric_name=metric_name,
            ylabel=ylabel,
            sampling_mode=sampling_mode,
            variant_label=variant_label,
            sampling_label=sampling_label,
            style_config=style_config,
        )

    displayed_axes = axes_list[: len(available_panel_specs)]
    for axis in axes_list[len(available_panel_specs) :]:
        axis.set_visible(False)

    shared_y_limits = shared_metric_limits(displayed_axes, metric_name=metric_name)
    if shared_y_limits is not None:
        for axis in displayed_axes:
            axis.set_ylim(*shared_y_limits)
    if metric_name == "perplexity":
        for row_start in range(0, len(displayed_axes), column_count):
            add_loss_secondary_axis(
                displayed_axes[min(row_start + column_count, len(displayed_axes)) - 1]
            )

    figure.suptitle(
        figure_title or f"{ylabel} vs Non-embedding parameters",
        fontsize=style_config["figure_title_fontsize"],
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(figure)

    output_paths = [output_path]
    if not include_individual_panels:
        return output_paths

    panel_stem = output_path.stem
    if figure_alias:
        panel_stem = f"{panel_stem}__{safe_filename_fragment(figure_alias)}"
    for panel_spec in available_panel_specs:
        panel_path = output_path.with_name(
            f"{panel_stem}__{safe_filename_fragment(panel_spec_label(*panel_spec))}.png"
        )
        output_paths.append(
            plot_metric_vs_size_panel_figure(
                plot_rows,
                metric_name=metric_name,
                ylabel=ylabel,
                panel_spec=panel_spec,
                output_path=panel_path,
                style_config=style_config,
                y_limits=shared_y_limits,
                dpi=dpi,
            )
        )

    return output_paths


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
    shared_limits = shared_metric_limits(axes_list[: len(panel_specs)])
    return [shared_limits] * row_count


def shared_metric_limits(
    axes_list: list[Any],
    *,
    metric_name: str | None = None,
) -> tuple[float, float] | None:
    values: list[float] = []
    for axis in axes_list:
        values.extend(axis_numeric_y_values(axis))
    if not values:
        return None
    limits = padded_limits(min(values), max(values))
    if metric_name == "perplexity" and limits[0] <= 0.0:
        limits = (min(values) * 0.92, limits[1])
    return limits


def add_loss_secondary_axis(axis):
    """Expose loss as an exact transform of a positive perplexity axis."""

    y_min, y_max = axis.get_ylim()
    if y_min <= 0.0 or y_max <= 0.0:
        raise ValueError("perplexity axis limits must be positive")
    loss_axis = axis.secondary_yaxis(
        "right",
        functions=(np.log, np.exp),
    )
    loss_axis.set_ylabel("Loss (nats/token)")
    return loss_axis


def numeric_metric_point(
    row: dict[str, str],
    metric_name: str,
) -> tuple[float, float] | None:
    x_value = to_float_or_none(row.get("non_embedding_parameters"))
    y_value = to_float_or_none(row.get(metric_name))
    if (
        x_value is None
        or y_value is None
        or not math.isfinite(x_value)
        or not math.isfinite(y_value)
        or (metric_name == "perplexity" and y_value <= 0.0)
    ):
        return None
    return (x_value, y_value)


def metric_panel_has_numeric_points(
    rows: list[dict[str, str]],
    metric_name: str,
    panel_spec: tuple[str, str, str | None],
) -> bool:
    sampling_mode, variant_label, sampling_label = panel_spec
    return any(
        scaling_curve_family_label(row) == sampling_mode
        and scaling_curve_variant_label(row) == variant_label
        and panel_sampling_matches(
            scaling_curve_sampling_label(row),
            sampling_label,
        )
        and numeric_metric_point(row, metric_name) is not None
        for row in rows
    )


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


def plot_metric_vs_size_panel_figure(
    rows: list[dict[str, str]],
    metric_name: str,
    ylabel: str,
    panel_spec: tuple[str, str, str | None],
    output_path: Path,
    style_config: dict[str, Any],
    y_limits: tuple[float, float] | None = None,
    dpi: int = 300,
) -> Path:
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    plot_metric_vs_size_panel(
        axis,
        rows,
        metric_name=metric_name,
        ylabel=ylabel,
        sampling_mode=panel_spec[0],
        variant_label=panel_spec[1],
        sampling_label=panel_spec[2],
        style_config=style_config,
    )
    if y_limits is not None:
        axis.set_ylim(*y_limits)
    if metric_name == "perplexity":
        add_loss_secondary_axis(axis)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(figure)
    return output_path


def plot_metric_vs_size_split_comparison(
    rows: list[dict[str, str]],
    metric_name: str,
    ylabel: str,
    output_path: Path,
    figure_title: str,
    style: str,
    left_panel_spec: dict[str, Any],
    right_panel_spec: dict[str, Any],
    dpi: int = 300,
) -> Path | None:
    style_config = resolve_plot_style(style)
    available_panel_specs = [
        panel_spec
        for panel_spec in (left_panel_spec, right_panel_spec)
        if comparison_panel_has_numeric_points(
            rows,
            metric_name=metric_name,
            panel_spec=panel_spec,
        )
    ]
    if not available_panel_specs:
        return None

    figure, axes = plt.subplots(
        1,
        len(available_panel_specs),
        figsize=(7.5 * len(available_panel_specs), 8.0),
        squeeze=False,
        sharey=False,
    )
    axes_list = list(axes.flat)
    shared_values: list[float] = []
    for axis, panel_spec in zip(axes_list, available_panel_specs):
        shared_values.extend(
            plot_metric_vs_size_split_panel(
                axis,
                rows,
                metric_name=metric_name,
                ylabel=ylabel,
                panel_spec=panel_spec,
                style_config=style_config,
            )
        )
        axis.set_title(
            str(panel_spec["subfigure_title"]),
            fontsize=style_config["subfigure_title_fontsize"],
            pad=10,
        )

    shared_limits = padded_limits(min(shared_values), max(shared_values))
    if metric_name == "perplexity" and shared_limits[0] <= 0.0:
        shared_limits = (min(shared_values) * 0.92, shared_limits[1])
    for axis in axes_list:
        axis.set_ylim(*shared_limits)
    if metric_name == "perplexity":
        add_loss_secondary_axis(axes_list[-1])

    figure.suptitle(
        figure_title,
        fontsize=style_config["figure_title_fontsize"],
        y=0.985,
    )
    figure.subplots_adjust(top=0.83, bottom=0.12, left=0.06, right=0.98, wspace=0.07)
    figure.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(figure)
    return output_path


def comparison_panel_has_numeric_points(
    rows: list[dict[str, str]],
    metric_name: str,
    panel_spec: dict[str, Any],
) -> bool:
    series_keys = set(panel_spec["series_keys"]) - {"standalone"}
    return any(
        comparison_series_key(row) in series_keys
        and numeric_metric_point(row, metric_name) is not None
        for row in rows
    )


def plot_metric_vs_size_split_panel(
    axis,
    rows: list[dict[str, str]],
    metric_name: str,
    ylabel: str,
    panel_spec: dict[str, Any],
    style_config: dict[str, Any],
) -> list[float]:
    series_keys = list(panel_spec["series_keys"])
    panel_rows = [row for row in rows if comparison_series_key(row) in series_keys]

    axis.set_xlabel(
        "Non-embedding parameters", fontsize=style_config["axis_label_fontsize"]
    )
    axis.set_ylabel(ylabel, fontsize=style_config["axis_label_fontsize"])
    axis.tick_params(labelsize=style_config["tick_label_fontsize"])
    axis.grid(True, alpha=0.3)
    axis.set_axisbelow(True)

    if not panel_rows:
        axis.text(
            0.5,
            0.5,
            "No numeric points found",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        return []

    grouped = group_rows_by_series_key(panel_rows, series_keys)
    series_values: list[float] = []

    for series_key in series_keys:
        series_rows = grouped.get(series_key)
        if not series_rows:
            continue

        points = [
            point
            for row in series_rows
            if (point := numeric_metric_point(row, metric_name)) is not None
        ]
        if not points:
            continue

        points.sort(key=lambda point: point[0])
        xs, ys = zip(*points)
        series_values.extend(ys)

        if series_key == "standalone":
            axis.scatter(
                xs,
                ys,
                marker="^",
                s=58,
                color=STANDALONE_REFERENCE_COLOR,
                edgecolors=STANDALONE_REFERENCE_EDGE_COLOR,
                linewidths=0.8,
                label=resolve_series_alias(series_key, style_config),
                zorder=5,
            )
            continue

        axis.plot(
            xs,
            ys,
            label=resolve_series_alias(series_key, style_config),
            **comparison_series_style(series_key, style_config),
        )

    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(frameon=False, fontsize=style_config["legend_fontsize"])

    return series_values


def panel_spec_label(
    sampling_mode: str,
    variant_label: str,
    sampling_label: str | None,
) -> str:
    parts = [sampling_mode, variant_label]
    if sampling_label is not None:
        parts.append(sampling_label)
    return " / ".join(parts)


def _humanize_panel_part(value: str) -> str:
    return value.replace("_", "-").capitalize()


def size_plot_panel_title(
    sampling_mode: str,
    variant_label: str,
    sampling_label: str | None,
) -> str:
    parts = [_humanize_panel_part(sampling_mode), _humanize_panel_part(variant_label)]
    sampling_titles = {
        "global": "Global sampling",
        "uniform_global_window": "Uniform global windows",
        "balanced_global_window": "Balanced global windows",
        "per_block": "Per-block sampling",
        "probabilistic_global_thompson": "Bayesian global TS",
        "probabilistic_per_block_thompson": "Bayesian per-block TS",
        "adaptive_per_block_ucb": "Per-block UCB",
        "panelgrad_global": "PanelGrad global",
    }
    if sampling_label is not None:
        parts.append(
            sampling_titles.get(sampling_label, _humanize_panel_part(sampling_label))
        )
    return " · ".join(parts)


def _format_scientific(value: Any) -> str:
    numeric = to_float_or_none(value)
    if numeric is None:
        canonical = json.dumps(
            _canonical_contract_value(value),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return f"structured-{hashlib.sha256(canonical.encode()).hexdigest()[:6]}"
    if numeric == 0.0:
        return "0"
    if abs(numeric) >= 1000 and float(numeric).is_integer():
        for divisor, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1000, "k")):
            if abs(numeric) >= divisor:
                scaled = numeric / divisor
                return f"{scaled:g}{suffix}"
    rendered = f"{numeric:g}"
    if "e" in rendered:
        mantissa, exponent = rendered.split("e", maxsplit=1)
        exponent_value = int(exponent)
        sign = "−" if exponent_value < 0 else "+"
        return f"{mantissa}e{sign}{abs(exponent_value)}"
    return rendered


def _format_contract_legend_value(value: Any) -> str:
    if isinstance(value, str) and to_float_or_none(value) is None:
        compact = value.replace("_", "-")
        return compact if len(compact) <= 24 else f"{compact[:12]}…{compact[-6:]}"
    if isinstance(value, bool):
        return "on" if value else "off"
    return _format_scientific(value)


def _reset_mode_key(row: dict[str, Any]) -> str:
    if not _truthy(row.get("controller_reset_enabled")):
        return "no_reset"
    policy = str(row.get("controller_reset_policy") or "full_prior").strip().lower()
    return "acquisition_only" if policy == "acquisition_only" else "full_prior"


GLOBAL_TS_IDENTITIES = {
    "probabilistic_global_thompson",
    "probabilistic_global_thompson_reset",
    "probabilistic_global_thompson_acquisition_only",
}


def _size_plot_sampling_family(row: dict[str, Any]) -> str:
    identity = scaling_curve_sampling_label(row) or "global"
    if re.fullmatch(r"uniform_global_h[1-9][0-9]*", identity):
        return "uniform_global_window"
    if re.fullmatch(r"balanced_global_h[1-9][0-9]*", identity):
        return "balanced_global_window"
    return (
        "probabilistic_global_thompson"
        if identity in GLOBAL_TS_IDENTITIES
        else identity
    )


def _size_plot_sampling_display(row: dict[str, Any]) -> str:
    identity = scaling_curve_sampling_label(row) or "global"
    if re.fullmatch(
        r"(?:uniform|balanced)_global_h[1-9][0-9]*", identity
    ):
        return display_sampling_label_for_curve(identity) or identity
    labels = {
        "global": "Global sampling",
        "fixed_global": "Fixed non-uniform global",
        "per_block": "Per-block sampling",
        "probabilistic_global_thompson": "Bayesian global TS",
        "probabilistic_per_block_thompson": "Bayesian per-block TS",
        "adaptive_per_block_ucb": "Per-block UCB",
    }
    family = _size_plot_sampling_family(row)
    return labels.get(family, _humanize_panel_part(family))


def _size_plot_group_sort_key(rows: list[dict[str, Any]]) -> tuple[Any, ...]:
    row = rows[0]
    reset_rank = {"no_reset": 0, "full_prior": 1, "acquisition_only": 2}
    q_value = to_float_or_none(row.get("controller_process_noise_covariance"))
    panelgrad_temperature = to_float_or_none(row.get("panelgrad_temperature"))
    sampling_identity = scaling_curve_sampling_label(row) or "global"
    uniform_window_match = re.fullmatch(
        r"(?:uniform|balanced)_global_h([1-9][0-9]*)", sampling_identity
    )
    uniform_interval = (
        int(uniform_window_match.group(1))
        if uniform_window_match is not None
        else (1 if sampling_identity == "global" else math.inf)
    )
    return (
        reset_rank.get(_reset_mode_key(row), 9),
        q_value if q_value is not None else math.inf,
        panelgrad_temperature
        if panelgrad_temperature is not None
        else math.inf,
        uniform_interval,
        size_plot_experiment_contract(row),
    )


def _differing_fields(groups: list[list[dict[str, Any]]]) -> set[str]:
    differing: set[str] = set()
    for field_name in SIZE_PLOT_CONTRACT_FIELDS:
        values = {
            json.dumps(
                _canonical_contract_value(group[0].get(field_name)),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            for group in groups
        }
        if len(values) > 1:
            differing.add(field_name)
    corrections = {scaling_curve_correction_label(group[0]) for group in groups}
    if len(corrections) > 1:
        differing.add("correction")
    return differing


def _field_contract_value(row: dict[str, Any], field_name: str) -> str:
    if field_name == "correction":
        return str(scaling_curve_correction_label(row) or "none")
    value = row.get(field_name)
    if _value_is_missing(value):
        value = None
    return json.dumps(
        _canonical_contract_value(value),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _minimal_peer_differentiators(
    items: list[tuple[str, list[dict[str, Any]]]],
) -> dict[str, list[str]]:
    """Select a compact field set only for contracts still in collision."""

    selected = {contract: [] for contract, _ in items}
    candidate_fields = (
        "global_sampling_distribution",
        "panelgrad_importance_metric",
        "panelgrad_temperature",
        "panelgrad_epsilon_schedule_type",
        "panelgrad_epsilon_schedule_start",
        "panelgrad_epsilon_schedule_end",
        "panelgrad_epsilon_schedule_duration_steps",
        "panelgrad_epsilon",
        "panelgrad_refresh_interval_steps",
        "panelgrad_eta",
        "controller_process_noise_covariance",
        "correction",
        "controller_decision_interval_steps",
        "controller_observation_noise_variance",
        "controller_acquisition_passes",
        "controller_acquisition_policy",
        "controller_context_model",
        "controller_compute_weight",
        "pre_nested_warmup_duration",
        "pre_nested_warmup_policy",
        "pre_nested_warmup_action_interval_steps",
        "training_token_budget",
        "training_max_steps",
        "training_learning_rate",
        "training_optimizer_name",
        "training_optimizer_kwargs",
        "training_scheduler_name",
        "training_scheduler_kwargs",
        "training_context_length",
        "training_batch_size_per_process",
        "training_precision",
        "training_dataset_name",
        "training_dataset_config_name",
        "training_dataset_split",
        "training_tokenizer_name",
        "controller_prior_covariance",
        "controller_prior_mean",
        "controller_feature_schema_hash",
    )
    initial_partitions: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {}
    for contract, rows in items:
        row = rows[0]
        interval_key = (
            _field_contract_value(row, "controller_reset_interval_steps")
            if _reset_mode_key(row) != "no_reset"
            else ""
        )
        initial_partitions.setdefault(interval_key, []).append((contract, rows))

    pending = [
        partition for partition in initial_partitions.values() if len(partition) > 1
    ]
    while pending:
        partition = pending.pop(0)
        already_selected = {
            field_name for contract, _ in partition for field_name in selected[contract]
        }
        best_field = None
        best_partition_count = 1
        for field_name in candidate_fields:
            if field_name in already_selected:
                continue
            partition_count = len(
                {_field_contract_value(rows[0], field_name) for _, rows in partition}
            )
            if partition_count > best_partition_count:
                best_field = field_name
                best_partition_count = partition_count
        if best_field is None:
            continue
        split: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {}
        for contract, rows in partition:
            selected[contract].append(best_field)
            split.setdefault(_field_contract_value(rows[0], best_field), []).append(
                (contract, rows)
            )
        pending.extend(group for group in split.values() if len(group) > 1)
    return selected


def compact_size_curve_labels(
    grouped_rows: dict[str, list[dict[str, Any]]],
    panel_sampling_label: str | None = None,
) -> dict[str, str]:
    """Describe only contract fields that distinguish curves within a panel."""

    ordered_items = sorted(
        grouped_rows.items(), key=lambda item: _size_plot_group_sort_key(item[1])
    )
    all_groups = [rows for _, rows in ordered_items]
    global_ts_only = bool(all_groups) and all(
        scaling_curve_sampling_label(group[0]) in GLOBAL_TS_IDENTITIES
        for group in all_groups
    )
    show_sampling_method = panel_sampling_label in (
        None,
        "uniform_global_window",
        "balanced_global_window",
    )
    groups_by_comparison_family: dict[tuple[str, str], list[list[dict[str, Any]]]] = {}
    for group in all_groups:
        comparison_family = (
            _size_plot_sampling_family(group[0]),
            _reset_mode_key(group[0]),
        )
        groups_by_comparison_family.setdefault(comparison_family, []).append(group)
    globally_differing = _differing_fields(all_groups)
    differentiators: dict[str, list[str]] = {}
    for comparison_family in groups_by_comparison_family:
        reset_items = [
            (contract, rows)
            for contract, rows in ordered_items
            if (
                _size_plot_sampling_family(rows[0]),
                _reset_mode_key(rows[0]),
            )
            == comparison_family
        ]
        differentiators.update(_minimal_peer_differentiators(reset_items))
    labels: dict[str, str] = {}

    for contract, group in ordered_items:
        row = group[0]
        parts: list[str] = []
        reset_mode = _reset_mode_key(row)
        is_global_ts = scaling_curve_sampling_label(row) in GLOBAL_TS_IDENTITIES
        if show_sampling_method:
            parts.append(_size_plot_sampling_display(row))
        if is_global_ts and (len(all_groups) > 1 or show_sampling_method):
            parts.append(
                {
                    "no_reset": "No reset",
                    "full_prior": "Full-prior",
                    "acquisition_only": "Acquisition-only",
                }.get(reset_mode, _humanize_panel_part(reset_mode))
            )
            if reset_mode != "no_reset":
                interval = row.get("controller_reset_interval_steps")
                if not _value_is_missing(interval):
                    parts.append(f"K={_format_scientific(interval)}")

        field_labels = {
            "global_sampling_distribution": "Sampling distribution",
            "panelgrad_importance_metric": "importance metric",
            "panelgrad_temperature": "T",
            "panelgrad_epsilon": "ε",
            "panelgrad_epsilon_schedule_type": "ε schedule",
            "panelgrad_epsilon_schedule_start": "ε start",
            "panelgrad_epsilon_schedule_end": "ε end",
            "panelgrad_epsilon_schedule_duration_steps": "ε steps",
            "panelgrad_refresh_interval_steps": "H",
            "panelgrad_eta": "η",
            "controller_process_noise_covariance": "Q",
            "controller_decision_interval_steps": "h",
            "controller_observation_noise_variance": "R",
            "controller_prior_covariance": "Prior cov",
            "controller_prior_mean": "Prior mean",
            "controller_acquisition_passes": "Acquisition passes",
            "controller_acquisition_policy": "Acquisition",
            "controller_context_model": "Context",
            "controller_compute_weight": "Compute weight",
            "pre_nested_warmup_duration": "Warmup",
            "pre_nested_warmup_policy": "Warmup policy",
            "pre_nested_warmup_action_interval_steps": "Warmup h",
            "training_token_budget": "Budget",
            "training_max_steps": "Steps",
            "training_learning_rate": "LR",
            "training_optimizer_name": "Optimizer",
            "training_optimizer_kwargs": "Optimizer args",
            "training_scheduler_name": "Scheduler",
            "training_scheduler_kwargs": "Scheduler args",
            "training_context_length": "Context length",
            "training_batch_size_per_process": "Batch/process",
            "training_precision": "Precision",
            "training_dataset_name": "Dataset",
            "training_dataset_config_name": "Dataset config",
            "training_dataset_split": "Dataset split",
            "training_tokenizer_name": "Tokenizer",
            "controller_feature_schema_hash": "Feature schema",
        }
        for field_name in differentiators.get(contract, []):
            if field_name == "correction":
                correction = scaling_curve_correction_label(row) or "none"
                parts.append(
                    correction.upper() if correction != "none" else "No correction"
                )
                continue
            parts.append(
                f"{field_labels[field_name]}="
                f"{_format_contract_legend_value(row.get(field_name))}"
            )

        if (
            "correction" in globally_differing
            and "correction" not in differentiators.get(contract, [])
            and not global_ts_only
        ):
            correction = scaling_curve_correction_label(row) or "none"
            parts.append(
                correction.upper() if correction != "none" else "No correction"
            )
        if not parts:
            parts.append("Trained model")

        seed_count = len(
            {
                str(candidate.get("run_seed"))
                for candidate in group
                if not _value_is_missing(candidate.get("run_seed"))
            }
        )
        if seed_count > 1:
            parts.append(f"n={seed_count} seeds")
        labels[contract] = " · ".join(parts)

    # If two contracts differ only in a field that is too verbose for the
    # compact legend, retain compactness while still making them distinguishable.
    labels_to_contracts: dict[str, list[str]] = {}
    for contract, label in labels.items():
        labels_to_contracts.setdefault(label, []).append(contract)
    for duplicate_contracts in labels_to_contracts.values():
        if len(duplicate_contracts) < 2:
            continue
        for contract in duplicate_contracts:
            suffix = hashlib.sha256(contract.encode()).hexdigest()[:6]
            labels[contract] = f"{labels[contract]} · contract {suffix}"
    return labels


def _size_plot_curve_style(
    rows: list[dict[str, Any]],
    curve_index: int,
    style_config: dict[str, Any],
) -> dict[str, Any]:
    base_style = scaling_curve_style(rows, style_config=style_config)
    palette = plt.get_cmap("tab10")
    markers = ("o", "s", "D", "^", "v", "P", "X", "*", "h", "<")
    base_style["color"] = palette(curve_index % 10)
    base_style["marker"] = markers[curve_index % len(markers)]
    return base_style


def plot_metric_vs_size_panel(
    axis,
    rows: list[dict[str, str]],
    metric_name: str,
    ylabel: str,
    sampling_mode: str,
    variant_label: str,
    sampling_label: str | None = None,
    style_config: dict[str, Any] | None = None,
) -> None:
    style_config = style_config or resolve_plot_style("default")
    panel_rows = [
        row
        for row in rows
        if scaling_curve_family_label(row) == sampling_mode
        and scaling_curve_variant_label(row) == variant_label
        and (
            panel_sampling_matches(
                scaling_curve_sampling_label(row),
                sampling_label,
            )
            or (
                sampling_label == "uniform_global_window"
                and scaling_curve_sampling_label(row) == "global"
            )
            or (
                sampling_label == "balanced_global_window"
                and scaling_curve_sampling_label(row) == "global"
            )
        )
    ]
    panel_title = size_plot_panel_title(
        sampling_mode,
        variant_label,
        sampling_label,
    )
    axis.set_title(panel_title, fontsize=style_config["panel_title_fontsize"], pad=6)
    axis.set_xlabel(
        "Non-embedding parameters", fontsize=style_config["axis_label_fontsize"]
    )
    axis.set_ylabel(ylabel, fontsize=style_config["axis_label_fontsize"])
    axis.tick_params(labelsize=style_config["tick_label_fontsize"])
    axis.grid(True, alpha=0.3)

    if not panel_rows:
        axis.text(
            0.5,
            0.5,
            "No numeric points found",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        return

    grouped = group_size_plot_rows(panel_rows)
    legend_labels = compact_size_curve_labels(
        grouped,
        panel_sampling_label=sampling_label,
    )
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            0
            if sampling_label == "balanced_global_window"
            and scaling_curve_sampling_label(item[1][0]) == "global"
            else 1,
            _size_plot_group_sort_key(item[1]),
        ),
    )
    for curve_index, (contract, group_rows_for_label) in enumerate(ordered_groups):
        style = _size_plot_curve_style(
            group_rows_for_label,
            curve_index,
            style_config,
        )
        is_iid_h1_reference = (
            sampling_label in {"uniform_global_window", "balanced_global_window"}
            and scaling_curve_sampling_label(group_rows_for_label[0]) == "global"
        )
        if is_iid_h1_reference:
            style.update(
                {
                    "color": "#555555",
                    "linestyle": ":",
                    "linewidth": 1.8,
                    "marker": "o",
                }
            )
        aggregate = aggregate_size_curve(group_rows_for_label, metric_name)
        if not aggregate["xs"]:
            continue
        axis.plot(
            aggregate["xs"],
            aggregate["means"],
            label=(
                "Global sampling (H=1 reference)"
                if is_iid_h1_reference
                and sampling_label == "uniform_global_window"
                else "Random global (IID H=1 reference)"
                if is_iid_h1_reference
                else legend_labels[contract]
            ),
            **style,
        )
        if any(aggregate["band_mask"]):
            lower = [
                value if include else math.nan
                for value, include in zip(aggregate["minimums"], aggregate["band_mask"])
            ]
            upper = [
                value if include else math.nan
                for value, include in zip(aggregate["maximums"], aggregate["band_mask"])
            ]
            axis.fill_between(
                aggregate["xs"],
                lower,
                upper,
                color=style["color"],
                alpha=0.16,
                linewidth=0,
                zorder=1,
            )

    standalone_rows = [
        row for row in rows if scaling_curve_family_label(row) == "standalone"
    ]
    standalone_aggregate = aggregate_size_curve(standalone_rows, metric_name)
    if standalone_aggregate["xs"]:
        axis.scatter(
            standalone_aggregate["xs"],
            standalone_aggregate["means"],
            marker="^",
            s=58,
            color=STANDALONE_REFERENCE_COLOR,
            edgecolors=STANDALONE_REFERENCE_EDGE_COLOR,
            linewidths=0.8,
            label=style_config["standalone_label"],
            zorder=5,
        )
        if any(standalone_aggregate["band_mask"]):
            lower = [
                value if include else math.nan
                for value, include in zip(
                    standalone_aggregate["minimums"],
                    standalone_aggregate["band_mask"],
                )
            ]
            upper = [
                value if include else math.nan
                for value, include in zip(
                    standalone_aggregate["maximums"],
                    standalone_aggregate["band_mask"],
                )
            ]
            axis.fill_between(
                standalone_aggregate["xs"],
                lower,
                upper,
                color=STANDALONE_REFERENCE_COLOR,
                alpha=0.12,
                linewidth=0,
                zorder=1,
            )

    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(frameon=False, fontsize=style_config["legend_fontsize"])


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


def comparison_series_key(row: dict[str, str]) -> str | None:
    family_label = scaling_curve_family_label(row)
    if family_label == "standalone":
        return "standalone"
    if family_label not in {"nested-random", "nested-all"}:
        return None

    variant_label = scaling_curve_variant_label(row) or "slicing"
    correction_label = scaling_curve_correction_label(row) or "none"
    if correction_label != "none":
        sampling_label = scaling_curve_sampling_label(row) or "global"
        if sampling_label != "global":
            return None
        return f"{family_label} / {variant_label} / {correction_label}"

    sampling_label = scaling_curve_sampling_label(row) or "global"
    return f"{family_label} / {variant_label} / {correction_label} / {sampling_label}"


def group_rows_by_series_key(
    rows: list[dict[str, str]],
    series_keys: list[str],
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {key: [] for key in series_keys}
    for row in rows:
        series_key = comparison_series_key(row)
        if series_key is None or series_key not in grouped:
            continue
        grouped[series_key].append(row)
    return grouped


def comparison_series_style(
    series_key: str,
    style_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    style_config = style_config or resolve_plot_style("default")
    if series_key == "standalone":
        return {
            "linewidth": 1.6,
            "linestyle": "None",
            "color": STANDALONE_REFERENCE_COLOR,
        }

    parts = series_key.split(" / ")
    variant_label = parts[1] if len(parts) > 1 else "slicing"
    correction_label = parts[2] if len(parts) > 2 else "none"
    sampling_label = "global"
    if len(parts) > 3:
        sampling_label = parts[3]
    correction_style = reporting_styles.SCALING_CORRECTION_STYLES.get(
        correction_label,
        reporting_styles.SCALING_CORRECTION_STYLES["none"],
    )
    base_color = style_config["series_colors"].get(series_key, "tab:gray")
    linestyle = style_config.get("comparison_linestyle")
    if not linestyle:
        linestyle = correction_style["linestyle"]
    marker = style_config.get("comparison_markers_by_variant", {}).get(
        variant_label,
        reporting_styles.SCALING_SAMPLING_MARKERS.get(
            sampling_label,
            correction_style["marker"],
        ),
    )
    return {
        "linewidth": 1.4,
        "linestyle": linestyle,
        "marker": marker,
        "markersize": 5,
        "color": blend_color_toward_white(base_color, correction_style["shade"]),
    }


def plot_metric_over_steps(
    rows: list[dict[str, str]],
    metric_name: str,
    ylabel: str,
    output_path: Path,
    dpi: int = 300,
) -> Path:
    figure, axis, legend_axis = create_figure_with_side_legend(
        plot_width=7,
        plot_height=4,
        legend_width=2.4,
    )
    grouped = group_rows(rows, ["split", "granularity"])

    for label, group_rows_for_label in grouped.items():
        points = [
            (to_float(row["step"]), to_float(row[metric_name]))
            for row in group_rows_for_label
            if row.get("step") not in (None, "")
            and row.get(metric_name) not in (None, "")
        ]
        if not points:
            continue
        points.sort(key=lambda point: point[0])
        xs, ys = zip(*points)
        axis.plot(xs, ys, marker="o", label=label)

    axis.set_xlabel("Step")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    from .reporting import mark_scheduler_warmup

    mark_scheduler_warmup(
        axis,
        {
            value
            for row in rows
            if (value := to_float_or_none(row.get("_scheduler_warmup_steps")))
            is not None
        },
    )
    place_legend_on_right(legend_axis, axis)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return output_path


def validation_run_is_completed(row: dict[str, Any]) -> bool:
    """Use run_summary-derived status when enrichment made it available."""

    status = row.get("_run_status")
    if status in (None, ""):
        # Low-level plotting helpers also accept already-vetted synthetic rows.
        return True
    return str(status).strip().lower() == "completed"


def filter_validation_rows_by_completion(
    rows: list[dict[str, Any]],
    *,
    include_incomplete: bool = False,
) -> list[dict[str, Any]]:
    if include_incomplete:
        return list(rows)
    return [row for row in rows if validation_run_is_completed(row)]


def validation_experiment_contract(row: dict[str, Any]) -> str:
    """Return the seed-independent contract, isolating unproven history."""

    enriched_contract = row.get("_validation_contract")
    if enriched_contract not in (None, ""):
        return str(enriched_contract)
    payload = {
        "family": scaling_curve_family_label(row),
        "variant": scaling_curve_variant_label(row),
        "sampling": scaling_curve_sampling_label(row),
        "correction": scaling_curve_correction_label(row),
        "model_shape": row.get("model_shape_label") or row.get("model_size_label"),
        "historical_run_fallback": experiment_label(row),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def validation_granularity_order(rows: list[dict[str, Any]]) -> list[str]:
    present = {
        str(row["granularity"])
        for row in rows
        if row.get("granularity") not in (None, "")
    }
    ordered: list[str] = []
    for row in rows:
        configured = row.get("_ordered_granularities")
        if isinstance(configured, str):
            try:
                configured = json.loads(configured)
            except json.JSONDecodeError:
                configured = None
        if isinstance(configured, list):
            for label in configured:
                label = str(label)
                if label in present and label not in ordered:
                    ordered.append(label)
    ordered.extend(sorted(present - set(ordered), key=granularity_sort_key))
    return ordered


def _validation_replicate_id(row: dict[str, Any]) -> str:
    seed = row.get("run_seed")
    if seed not in (None, ""):
        return f"seed:{seed}"
    return f"run:{experiment_label(row)}"


def aggregate_validation_curve(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate matching completed seeds at exact total-token checkpoints."""

    by_x_and_replicate: dict[float, dict[str, list[float]]] = {}
    replicate_ids: set[str] = set()
    known_seeds: set[str] = set()
    for row in rows:
        x_value = to_float_or_none(row.get("tokens_seen"))
        y_value = to_float_or_none(row.get("loss"))
        if x_value is None or y_value is None:
            continue
        replicate_id = _validation_replicate_id(row)
        replicate_ids.add(replicate_id)
        if replicate_id.startswith("seed:"):
            known_seeds.add(replicate_id.removeprefix("seed:"))
        by_x_and_replicate.setdefault(x_value, {}).setdefault(replicate_id, []).append(
            y_value
        )

    xs: list[float] = []
    means: list[float] = []
    minimums: list[float] = []
    maximums: list[float] = []
    band_mask: list[bool] = []
    for x_value in sorted(by_x_and_replicate):
        replicate_values = []
        for values in by_x_and_replicate[x_value].values():
            if len(set(values)) > 1:
                warnings.warn(
                    "Conflicting duplicate validation rows at total-token "
                    f"checkpoint {x_value}; using the latest persisted value",
                    RuntimeWarning,
                    stacklevel=2,
                )
            replicate_values.append(values[-1])
        xs.append(x_value)
        means.append(sum(replicate_values) / len(replicate_values))
        minimums.append(min(replicate_values))
        maximums.append(max(replicate_values))
        band_mask.append(len(replicate_values) > 1)
    return {
        "xs": xs,
        "means": means,
        "minimums": minimums,
        "maximums": maximums,
        "band_mask": band_mask,
        "seed_count": len(known_seeds),
        "replicate_count": len(replicate_ids),
    }


def group_completed_validation_contracts(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if validation_run_is_completed(row):
            grouped.setdefault(validation_experiment_contract(row), []).append(row)
    return grouped


def group_incomplete_validation_runs(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not validation_run_is_completed(row):
            grouped.setdefault(experiment_label(row), []).append(row)
    return grouped


def _validation_contract_base_label(rows: list[dict[str, Any]]) -> str:
    row = rows[0]
    parts: list[str] = []
    correction = scaling_curve_correction_label(row)
    parts.append(correction.upper() if correction else "No correction")
    shape = row.get("model_shape_label") or row.get("model_size_label")
    if shape not in (None, ""):
        parts.append(str(shape))
    if row.get("_validation_contract_fallback"):
        parts.append(experiment_label(row))
    return " · ".join(parts)


def compact_validation_contract_labels(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    include_method: bool = False,
) -> dict[str, str]:
    labels: dict[str, str] = {}
    size_labels = compact_size_curve_labels(grouped)
    base_counts: dict[str, int] = {}
    bases = {
        contract: _validation_contract_base_label(group_rows)
        for contract, group_rows in grouped.items()
    }
    for base in bases.values():
        base_counts[base] = base_counts.get(base, 0) + 1
    for contract, group_rows in grouped.items():
        row = group_rows[0]
        parts: list[str] = []
        if include_method:
            method_key = validation_comparison_method_key(row)
            if method_key is not None:
                parts.append(validation_comparison_display_label(method_key))
        base = bases[contract]
        parts.append(base)
        size_label = size_labels.get(contract)
        if size_label not in (None, "", "Trained model"):
            parts.append(size_label)
        if base_counts[base] > 1:
            peer_labels = {
                size_labels.get(peer_contract)
                for peer_contract, peer_base in bases.items()
                if peer_base == base
            }
            if len(peer_labels) <= 1:
                parts.append(
                    f"contract {hashlib.sha256(contract.encode()).hexdigest()[:8]}"
                )
        seed_count = len(
            {
                str(row.get("run_seed"))
                for row in group_rows
                if row.get("run_seed") not in (None, "")
            }
        )
        if seed_count > 1 and not any("n=" in part for part in parts):
            parts.append(f"n={seed_count} seeds")
        labels[contract] = " · ".join(part for part in parts if part)
    return labels


def concise_validation_comparison_contract_labels(
    grouped: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    """Label comparison curves without repeating panel-level information."""

    labels: dict[str, str] = {}
    size_labels = compact_size_curve_labels(grouped)
    method_keys = {
        contract: validation_comparison_method_key(rows[0])
        for contract, rows in grouped.items()
    }
    method_counts: dict[str, int] = {}
    for method_key in method_keys.values():
        if method_key is not None:
            method_counts[method_key] = method_counts.get(method_key, 0) + 1

    for contract, method_key in method_keys.items():
        if method_key is None:
            labels[contract] = "Unknown method"
            continue
        base = validation_comparison_display_label(method_key)
        # Standalone width is already encoded by the subplot title, so every
        # standalone reference intentionally shares one legend entry.
        if method_key == "standalone" or method_counts[method_key] == 1:
            labels[contract] = base
            continue
        size_label = size_labels.get(contract)
        if size_label not in (None, "", "Trained model"):
            labels[contract] = f"{base} · {size_label}"
        else:
            labels[contract] = (
                f"{base} · config "
                f"{hashlib.sha256(contract.encode()).hexdigest()[:8]}"
            )
    return labels


def incomplete_validation_label(run_id: str, rows: list[dict[str, Any]]) -> str:
    progress = max(
        (
            to_float_or_none(row.get("_run_progress_tokens"))
            or to_float_or_none(row.get("tokens_seen"))
            or 0.0
        )
        for row in rows
    )
    budgets = [to_float_or_none(row.get("_run_token_budget")) for row in rows]
    budget = next((value for value in budgets if value is not None), None)
    progress_text = (
        f"{progress:.0f}/{budget:.0f} tokens"
        if budget not in (None, 0.0)
        else f"{progress:.0f} tokens"
    )
    status = str(rows[0].get("_run_status") or "incomplete")
    return f"{run_id} ({status}: {progress_text})"


def _apply_common_validation_y_limits(axes: list[Any]) -> None:
    values: list[float] = []
    for axis in axes:
        if axis.get_visible():
            values.extend(axis_numeric_y_values(axis))
    if not values:
        return
    limits = padded_limits(min(values), max(values))
    for axis in axes:
        if axis.get_visible():
            axis.set_ylim(limits)


@dataclass(frozen=True)
class SelectedExposureObservation:
    total_tokens: float
    selected_exposure_tokens: float
    loss: float


@dataclass(frozen=True)
class SelectedExposureRun:
    run_id: str
    run_seed: str
    method: str
    contract: str
    contract_row: dict[str, Any]
    ordered_granularities: tuple[str, ...]
    observations: dict[str, tuple[SelectedExposureObservation, ...]]


class _DirectExposureAccumulator:
    def __init__(self, ordered_granularities: list[str] | tuple[str, ...]):
        granularities = tuple(str(label) for label in ordered_granularities)
        self.exposure = {label: 0.0 for label in granularities}
        self.history: list[tuple[float, dict[str, float]]] = []
        self.seen_steps: dict[int, tuple[float, str]] = {}
        self.previous_step = 0
        self.previous_tokens = 0.0

    def consume(self, row: dict[str, Any]) -> None:
        if str(row.get("split") or "") != "train":
            return
        try:
            step = int(row.get("step"))
        except (TypeError, ValueError) as error:
            raise ValueError("training row has a non-integer step") from error
        tokens = to_float_or_none(row.get("tokens_seen"))
        action = str(row.get("granularity") or "")
        if tokens is None or not math.isfinite(tokens):
            raise ValueError(f"training step {step} has invalid planned tokens")
        signature = (tokens, action)
        if step in self.seen_steps:
            if self.seen_steps[step] != signature:
                raise ValueError(
                    f"resumed training step {step} conflicts with its earlier row"
                )
            return
        if step <= self.previous_step:
            raise ValueError(f"training steps are non-monotonic at unseen step {step}")
        if tokens <= self.previous_tokens:
            raise ValueError(
                f"planned training tokens are non-monotonic at step {step}"
            )
        if action not in self.exposure:
            raise ValueError(
                f"training step {step} selected unknown global granularity {action!r}"
            )
        self.exposure[action] += tokens - self.previous_tokens
        self.history.append((tokens, dict(self.exposure)))
        self.seen_steps[step] = signature
        self.previous_step = step
        self.previous_tokens = tokens


def reconstruct_direct_selected_exposure(
    training_rows: Any,
    ordered_granularities: list[str] | tuple[str, ...],
) -> list[tuple[float, dict[str, float]]]:
    """Reconstruct cumulative selected-action exposure from planned tokens."""

    accumulator = _DirectExposureAccumulator(ordered_granularities)
    for row in training_rows:
        accumulator.consume(row)
    return accumulator.history


def _exposure_at_total_tokens(
    history: list[tuple[float, dict[str, float]]],
    granularity: str,
    total_tokens: float,
) -> float:
    selected_exposure = 0.0
    for checkpoint_tokens, exposures in history:
        if checkpoint_tokens > total_tokens:
            break
        selected_exposure = exposures[granularity]
    return selected_exposure


def _completed_global_run_config(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    config_path = run_dir / "config.json"
    summary_path = run_dir / "run_summary.json"
    if not config_path.is_file() or not summary_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(config, dict) or not isinstance(summary, dict):
        return None
    if summary.get("status") != "completed":
        return None
    run = config.get("run", {})
    model = config.get("model", {})
    if not isinstance(run, dict) or not isinstance(model, dict):
        return None
    run_mode = (
        str(run.get("resolved_run_mode") or run.get("sampling_mode") or "")
        .strip()
        .lower()
    )
    sampling_mode = (
        str(
            model.get("resolved_sampling_mode")
            or model.get("granularity_sampling_mode")
            or ""
        )
        .strip()
        .lower()
    )
    controller = model.get("adaptive_controller")
    controller_scope = (
        str(controller.get("scope") or "").strip().lower()
        if isinstance(controller, dict)
        else ""
    )
    is_global = sampling_mode == "global" or (
        sampling_mode == "adaptive_global" and controller_scope == "global"
    )
    if run_mode != "nested-random" or not is_global:
        return None
    return config, summary


def iter_selected_exposure_runs(input_root: str | Path):
    """Stream completed global-run metrics into compact exposure observations."""

    from src.evaluation import reporting_io

    input_root = Path(input_root)
    for metrics_path in sorted(input_root.rglob("metrics.csv")):
        run_dir = metrics_path.parent
        loaded = _completed_global_run_config(run_dir)
        if loaded is None:
            continue
        config, _summary = loaded
        model = config["model"]
        ordered_granularities = tuple(
            str(label) for label in model.get("granularities", [])
        )
        if not ordered_granularities:
            warnings.warn(
                f"Skipping saturation diagnostics for {run_dir}: no granularities",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        exposure_accumulator = _DirectExposureAccumulator(ordered_granularities)
        validation_rows: list[dict[str, str]] = []
        try:
            for row in reporting_io.iter_csv_artifact_rows(metrics_path):
                split = str(row.get("split") or "")
                if split == "train":
                    exposure_accumulator.consume(row)
                elif reporting_io.validation_split_filter(row):
                    validation_rows.append(row)
        except ValueError as error:
            warnings.warn(
                f"Skipping saturation diagnostics for {run_dir}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        exposure_history = exposure_accumulator.history
        if not exposure_history or not validation_rows:
            continue

        enriched_validation = reporting_io.enrich_metrics_metadata_from_run_config(
            input_root,
            validation_rows,
        )
        if not enriched_validation:
            continue
        contract_row = enriched_validation[0]
        contract = validation_experiment_contract(contract_row)
        method = loss_figure_label(contract_row)
        run_id = experiment_label(contract_row)
        run_seed = str(contract_row.get("run_seed") or run_id)
        observations: dict[str, list[SelectedExposureObservation]] = {
            label: [] for label in ordered_granularities
        }
        seen_validation: dict[tuple[str, float], float] = {}
        invalid_validation = False
        for row in enriched_validation:
            granularity = str(row.get("granularity") or "")
            total_tokens = to_float_or_none(row.get("tokens_seen"))
            loss = to_float_or_none(row.get("loss"))
            if (
                granularity not in observations
                or total_tokens is None
                or loss is None
                or not math.isfinite(total_tokens)
                or not math.isfinite(loss)
            ):
                continue
            key = (granularity, total_tokens)
            if key in seen_validation:
                if seen_validation[key] != loss:
                    warnings.warn(
                        f"Skipping saturation diagnostics for {run_dir}: "
                        f"conflicting resumed validation checkpoint {key}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    invalid_validation = True
                    break
                continue
            seen_validation[key] = loss
            observations[granularity].append(
                SelectedExposureObservation(
                    total_tokens=total_tokens,
                    selected_exposure_tokens=_exposure_at_total_tokens(
                        exposure_history,
                        granularity,
                        total_tokens,
                    ),
                    loss=loss,
                )
            )
        if invalid_validation:
            continue
        normalized_observations = {
            granularity: tuple(
                sorted(values, key=lambda observation: observation.total_tokens)
            )
            for granularity, values in observations.items()
            if values
        }
        if not normalized_observations:
            continue
        yield SelectedExposureRun(
            run_id=run_id,
            run_seed=run_seed,
            method=method,
            contract=contract,
            contract_row=contract_row,
            ordered_granularities=ordered_granularities,
            observations=normalized_observations,
        )


def _latest_loss_by_distinct_exposure(
    observations: tuple[SelectedExposureObservation, ...]
    | list[SelectedExposureObservation],
) -> list[SelectedExposureObservation]:
    distinct: list[SelectedExposureObservation] = []
    for observation in observations:
        if (
            distinct
            and observation.selected_exposure_tokens
            == distinct[-1].selected_exposure_tokens
        ):
            distinct[-1] = observation
        else:
            distinct.append(observation)
    return distinct


def interpolate_loss_over_shared_exposure(
    seed_observations: dict[str, Any],
    *,
    grid_points: int = 100,
) -> dict[str, Any]:
    """Interpolate seeds only on their common direct-exposure support."""

    curves: dict[str, list[SelectedExposureObservation]] = {}
    for seed, observations in seed_observations.items():
        distinct = _latest_loss_by_distinct_exposure(observations)
        if distinct:
            curves[seed] = distinct
    if not curves:
        return {"xs": [], "means": [], "minimums": [], "maximums": []}
    shared_min = max(curve[0].selected_exposure_tokens for curve in curves.values())
    shared_max = min(curve[-1].selected_exposure_tokens for curve in curves.values())
    if shared_min > shared_max:
        return {"xs": [], "means": [], "minimums": [], "maximums": []}
    if shared_min == shared_max or grid_points <= 1:
        grid = [shared_min]
    else:
        grid = [
            shared_min + (shared_max - shared_min) * index / (grid_points - 1)
            for index in range(grid_points)
        ]

    def interpolate(curve: list[SelectedExposureObservation], x_value: float) -> float:
        if (
            x_value < curve[0].selected_exposure_tokens
            or x_value > curve[-1].selected_exposure_tokens
        ):
            raise ValueError("interpolation attempted outside seed exposure support")
        for left, right in zip(curve, curve[1:]):
            if (
                left.selected_exposure_tokens
                <= x_value
                <= right.selected_exposure_tokens
            ):
                width = right.selected_exposure_tokens - left.selected_exposure_tokens
                if width == 0:
                    return right.loss
                fraction = (x_value - left.selected_exposure_tokens) / width
                return left.loss + fraction * (right.loss - left.loss)
        return curve[-1].loss

    per_seed = {
        seed: [interpolate(curve, x_value) for x_value in grid]
        for seed, curve in curves.items()
    }
    values_by_x = [
        [values[index] for values in per_seed.values()] for index in range(len(grid))
    ]
    return {
        "xs": grid,
        "means": [sum(values) / len(values) for values in values_by_x],
        "minimums": [min(values) for values in values_by_x],
        "maximums": [max(values) for values in values_by_x],
        "seed_count": len(curves),
        "shared_min": shared_min,
        "shared_max": shared_max,
    }


def marginal_utility_observations(
    observations: tuple[SelectedExposureObservation, ...]
    | list[SelectedExposureObservation],
    *,
    window_size: int = 5,
    minimum_observations: int = 3,
) -> list[tuple[float, float]]:
    """Return negative five-point OLS loss slopes per million selected tokens."""

    distinct: list[SelectedExposureObservation] = []
    scores: list[tuple[float, float]] = []
    for observation in observations:
        if (
            distinct
            and observation.selected_exposure_tokens
            == distinct[-1].selected_exposure_tokens
        ):
            distinct[-1] = observation
        else:
            distinct.append(observation)
        window = distinct[-window_size:]
        if len(window) < minimum_observations:
            continue
        xs = [item.selected_exposure_tokens / 1_000_000.0 for item in window]
        ys = [item.loss for item in window]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        denominator = sum((x_value - mean_x) ** 2 for x_value in xs)
        if denominator <= 0:
            continue
        slope = (
            sum(
                (x_value - mean_x) * (y_value - mean_y)
                for x_value, y_value in zip(xs, ys)
            )
            / denominator
        )
        scores.append((observation.total_tokens, -slope))
    return scores


def _aggregate_aligned_score_curves(
    seed_curves: dict[str, list[tuple[float, float]]],
) -> dict[str, Any]:
    if not seed_curves:
        return {
            "xs": [],
            "means": [],
            "minimums": [],
            "maximums": [],
            "band_mask": [],
        }
    curve_maps = {seed: dict(curve) for seed, curve in seed_curves.items()}
    common_checkpoints = set.intersection(
        *(set(curve) for curve in curve_maps.values())
    )
    xs = sorted(common_checkpoints)
    by_x = {
        x_value: [curve[x_value] for curve in curve_maps.values()] for x_value in xs
    }
    return {
        "xs": xs,
        "means": [sum(by_x[x]) / len(by_x[x]) for x in xs],
        "minimums": [min(by_x[x]) for x in xs],
        "maximums": [max(by_x[x]) for x in xs],
        "band_mask": [len(by_x[x]) > 1 for x in xs],
    }


def _diagnostic_contract_labels(runs: list[SelectedExposureRun]) -> dict[str, str]:
    grouped = {
        contract: [run.contract_row for run in runs if run.contract == contract]
        for contract in {run.contract for run in runs}
    }
    return compact_validation_contract_labels(grouped)


def _plot_selected_exposure_loss(
    runs: list[SelectedExposureRun],
    output_path: Path,
    *,
    dpi: int,
) -> Path:
    granularities = validation_granularity_order(
        [
            {
                "granularity": granularity,
                "_ordered_granularities": list(run.ordered_granularities),
            }
            for run in runs
            for granularity in run.observations
        ]
    )
    figure, axes = plt.subplots(
        len(granularities), 1, figsize=(14, max(3.0, 2.5 * len(granularities)))
    )
    axes = [axes] if len(granularities) == 1 else list(axes)
    labels = _diagnostic_contract_labels(runs)
    colors = list(plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"]))
    contracts = sorted(labels, key=lambda contract: labels[contract])
    for axis, granularity in zip(axes, granularities):
        for index, contract in enumerate(contracts):
            seed_observations = {
                run.run_seed: run.observations[granularity]
                for run in runs
                if run.contract == contract and granularity in run.observations
            }
            aggregate = interpolate_loss_over_shared_exposure(seed_observations)
            if not aggregate["xs"]:
                continue
            xs_millions = [value / 1_000_000.0 for value in aggregate["xs"]]
            color = colors[index % len(colors)]
            axis.plot(
                xs_millions, aggregate["means"], color=color, label=labels[contract]
            )
            if aggregate.get("seed_count", 0) > 1:
                axis.fill_between(
                    xs_millions,
                    aggregate["minimums"],
                    aggregate["maximums"],
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                )
        axis.set_title(granularity, fontsize=11)
        axis.set_ylabel("Validation loss")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Directly selected training exposure (million tokens)")
    _apply_common_validation_y_limits(axes)
    handles_by_label: dict[str, Any] = {}
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            handles_by_label.setdefault(label, handle)
    if handles_by_label:
        figure.legend(
            list(handles_by_label.values()),
            list(handles_by_label),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(4, len(handles_by_label)),
            frameon=False,
        )
    figure.suptitle(
        f"Validation loss over directly selected exposure — {runs[0].method}"
    )
    figure.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.17, hspace=0.35)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_marginal_utility(
    runs: list[SelectedExposureRun],
    output_path: Path,
    *,
    dpi: int,
) -> tuple[Path, list[dict[str, Any]]]:
    granularities = validation_granularity_order(
        [
            {
                "granularity": granularity,
                "_ordered_granularities": list(run.ordered_granularities),
            }
            for run in runs
            for granularity in run.observations
        ]
    )
    figure, axes = plt.subplots(
        len(granularities),
        1,
        figsize=(14, max(3.0, 2.5 * len(granularities))),
        sharex=True,
    )
    axes = [axes] if len(granularities) == 1 else list(axes)
    labels = _diagnostic_contract_labels(runs)
    contracts = sorted(labels, key=lambda contract: labels[contract])
    colors = list(plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"]))
    ranking_rows: list[dict[str, Any]] = []
    warmup_token_positions = {
        value
        for run in runs
        if (
            value := to_float_or_none(
                run.contract_row.get("_scheduler_warmup_tokens")
            )
        )
        is not None
    }
    for axis, granularity in zip(axes, granularities):
        for index, contract in enumerate(contracts):
            seed_curves = {
                run.run_seed: marginal_utility_observations(
                    run.observations[granularity]
                )
                for run in runs
                if run.contract == contract and granularity in run.observations
            }
            seed_curves = {seed: curve for seed, curve in seed_curves.items() if curve}
            aggregate = _aggregate_aligned_score_curves(seed_curves)
            if not aggregate["xs"]:
                continue
            color = colors[index % len(colors)]
            axis.plot(
                aggregate["xs"], aggregate["means"], color=color, label=labels[contract]
            )
            if any(aggregate["band_mask"]):
                axis.fill_between(
                    aggregate["xs"],
                    aggregate["minimums"],
                    aggregate["maximums"],
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                )
            common_checkpoints = set.intersection(
                *(set(x for x, _ in curve) for curve in seed_curves.values())
            )
            if common_checkpoints:
                final_checkpoint = max(common_checkpoints)
                final_scores = [
                    dict(curve)[final_checkpoint] for curve in seed_curves.values()
                ]
                ranking_rows.append(
                    {
                        "method": runs[0].method,
                        "contract": labels[contract],
                        "granularity": granularity,
                        "score": sum(final_scores) / len(final_scores),
                        "minimum": min(final_scores),
                        "maximum": max(final_scores),
                        "seed_count": len(final_scores),
                        "total_tokens": final_checkpoint,
                    }
                )
        axis.axhline(0.0, color="0.35", linewidth=0.8, linestyle=":")
        axis.set_title(granularity, fontsize=11)
        axis.grid(True, alpha=0.3)
    from .reporting import mark_scheduler_warmup

    for axis in axes:
        mark_scheduler_warmup(
            axis,
            warmup_token_positions,
            label=axis is axes[0],
        )
    axes[-1].set_xlabel("Total training tokens")
    figure.supylabel(
        "Estimated validation-loss reduction\n"
        "per 1M directly selected tokens",
        x=0.015,
    )
    _apply_common_validation_y_limits(axes)
    handles_by_label: dict[str, Any] = {}
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            handles_by_label.setdefault(label, handle)
    if handles_by_label:
        figure.legend(
            list(handles_by_label.values()),
            list(handles_by_label),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.052),
            ncol=min(4, len(handles_by_label)),
            frameon=False,
        )
    figure.suptitle(
        "Validation marginal utility over total training tokens — "
        f"{runs[0].method}\n"
        "Five-point OLS over direct exposure: positive improves, zero saturates, "
        "negative degrades"
    )
    figure.text(
        0.5,
        0.012,
        "Diagnostic only: shared parameters can improve a width without direct selection.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.11, right=0.98, top=0.84, bottom=0.20, hspace=0.45)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path, ranking_rows


def write_marginal_utility_ranking(
    ranking_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    lines = [
        "# Validation Marginal-Utility Ranking",
        "",
        "Scores are evidence about saturation, not a binary saturation decision or a controller action. Higher is better; negative values show degradation.",
        "",
        "Direct exposure measures the selected global action. MatFormer parameter sharing can improve a granularity even while another action is selected.",
        "",
        "| Rank | Method | Contract | Granularity | Final score | Seed range | Seeds | Total training tokens |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    ordered = sorted(
        ranking_rows,
        key=lambda row: (
            str(row["method"]),
            str(row["contract"]),
            -float(row["score"]),
            granularity_sort_key(str(row["granularity"])),
        ),
    )
    rank_by_group: dict[tuple[str, str], int] = {}
    for row in ordered:
        group = (str(row["method"]), str(row["contract"]))
        rank_by_group[group] = rank_by_group.get(group, 0) + 1
        lines.append(
            "| {rank} | {method} | {contract} | {granularity} | {score:.6g} | "
            "[{minimum:.6g}, {maximum:.6g}] | {seed_count} | {total_tokens:.0f} |".format(
                rank=rank_by_group[group],
                **row,
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def generate_saturation_diagnostics(
    input_root: str | Path,
    output_dir: str | Path,
    *,
    dpi: int = 300,
    variants: list[str] | tuple[str, ...] | None = None,
    corrections: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    runs = [
        run
        for run in iter_selected_exposure_runs(input_root)
        if row_matches_plot_filters(
            run.contract_row,
            variants=variants,
            corrections=corrections,
        )
    ]
    if not runs:
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    ranking_rows: list[dict[str, Any]] = []
    methods = sorted({run.method for run in runs})
    for method in methods:
        method_runs = [run for run in runs if run.method == method]
        method_slug = safe_filename_fragment(method)
        paths.append(
            _plot_selected_exposure_loss(
                method_runs,
                output_dir
                / f"validation_loss_over_selected_exposure_{method_slug}.png",
                dpi=dpi,
            )
        )
        marginal_path, method_rankings = _plot_marginal_utility(
            method_runs,
            output_dir / f"validation_marginal_utility_over_tokens_{method_slug}.png",
            dpi=dpi,
        )
        paths.append(marginal_path)
        ranking_rows.extend(method_rankings)
    paths.append(
        write_marginal_utility_ranking(
            ranking_rows,
            output_dir / "validation_marginal_utility_ranking.md",
        )
    )
    return paths


def plot_validation_loss_over_tokens_by_experiment(
    rows: list[dict[str, str]],
    output_dir: Path,
    dpi: int = 300,
    validation_loss_log_y: bool = False,
    include_incomplete_validation_traces: bool = False,
) -> list[Path]:
    output_paths = []
    rows = filter_validation_rows_by_completion(
        rows,
        include_incomplete=include_incomplete_validation_traces,
    )
    grouped = group_loss_rows_by_figure(
        [row for row in rows if str(row.get("split") or "") == "validation"]
    )
    for figure_label in sorted(grouped):
        figure_rows = grouped[figure_label]
        output_paths.append(
            plot_loss_over_tokens_for_experiment(
                figure_rows,
                figure_label,
                output_dir
                / f"validation_loss_over_tokens_{safe_filename_fragment(figure_label)}.png",
                dpi=dpi,
                validation_loss_log_y=validation_loss_log_y,
            )
        )
    return output_paths


def plot_validation_loss_over_tokens_by_granularity_comparison(
    rows: list[dict[str, str]],
    output_dir: Path,
    dpi: int = 300,
    validation_loss_log_y: bool = False,
    include_incomplete_validation_traces: bool = False,
) -> list[Path]:
    rows = filter_validation_rows_by_completion(
        rows,
        include_incomplete=include_incomplete_validation_traces,
    )
    comparison_rows = [
        row
        for row in rows
        if str(row.get("split") or "") == "validation"
        and validation_variant_key(row) == "none"
        and validation_comparison_method_key(row) is not None
    ]
    if not comparison_rows:
        return []

    return [
        plot_validation_loss_over_tokens_by_granularity_comparison_figure(
            comparison_rows,
            output_dir / "validation_loss_over_tokens_granularity_comparison.png",
            dpi=dpi,
            validation_loss_log_y=validation_loss_log_y,
        )
    ]


def plot_validation_loss_over_tokens_by_granularity_comparison_figure(
    rows: list[dict[str, str]],
    output_path: Path,
    dpi: int = 300,
    validation_loss_log_y: bool = False,
) -> Path:
    granularity_rows = [row for row in rows if row.get("granularity") not in (None, "")]
    granularity_labels = validation_granularity_order(granularity_rows)

    if not granularity_labels:
        figure, axis = plt.subplots(figsize=(12, 8))
        axis.text(
            0.5,
            0.5,
            "No granularity metadata found",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        figure.suptitle("Validation loss comparison by granularity", fontsize=16)
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return output_path

    figure_height = max(3.0, 2.5 * len(granularity_labels))
    figure, axes = plt.subplots(
        len(granularity_labels),
        1,
        figsize=(14, figure_height),
        sharex=True,
    )

    if len(granularity_labels) == 1:
        axes = [axes]

    completed_contracts = group_completed_validation_contracts(granularity_rows)
    contract_labels = concise_validation_comparison_contract_labels(
        completed_contracts
    )
    prop_cycle = plt.rcParams.get("axes.prop_cycle")
    colors = list(prop_cycle.by_key().get("color", [])) if prop_cycle else []
    if not colors:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    contract_order = sorted(completed_contracts, key=lambda key: contract_labels[key])
    contract_styles: dict[str, dict[str, Any]] = {}
    for index, contract in enumerate(contract_order):
        method_key = validation_comparison_method_key(completed_contracts[contract][0])
        base_style = validation_comparison_styles([method_key]).get(
            method_key,
            {
                "marker": "o",
                "linestyle": "-",
                "linewidth": 1.4,
                "markersize": 3.5,
            },
        )
        contract_styles[contract] = {
            **base_style,
            "color": (
                STANDALONE_REFERENCE_COLOR
                if method_key == "standalone"
                else colors[index % len(colors)]
            ),
        }
    incomplete_runs = group_incomplete_validation_runs(granularity_rows)
    warmup_token_positions = {
        value
        for row in granularity_rows
        if (value := to_float_or_none(row.get("_scheduler_warmup_tokens")))
        is not None
    }

    for axis, granularity in zip(axes, granularity_labels):
        plotted = False
        for contract in contract_order:
            contract_rows = [
                row
                for row in completed_contracts[contract]
                if str(row.get("granularity") or "") == granularity
            ]
            aggregate = aggregate_validation_curve(contract_rows)
            if not aggregate["xs"]:
                continue
            plotted = True
            style = contract_styles[contract]
            axis.plot(
                aggregate["xs"],
                aggregate["means"],
                label=contract_labels[contract],
                **style,
            )
            if any(aggregate["band_mask"]):
                axis.fill_between(
                    aggregate["xs"],
                    aggregate["minimums"],
                    aggregate["maximums"],
                    color=style["color"],
                    alpha=0.14,
                    linewidth=0,
                )

        for index, (run_id, run_rows) in enumerate(sorted(incomplete_runs.items())):
            curve_rows = [
                row
                for row in run_rows
                if str(row.get("granularity") or "") == granularity
            ]
            points = sorted(
                (
                    (to_float(row["tokens_seen"]), to_float(row["loss"]))
                    for row in curve_rows
                    if row.get("tokens_seen") not in (None, "")
                    and row.get("loss") not in (None, "")
                ),
                key=lambda point: point[0],
            )
            if not points:
                continue
            plotted = True
            method_key = validation_comparison_method_key(curve_rows[0])
            method_label = (
                validation_comparison_display_label(method_key)
                if method_key is not None
                else "unknown method"
            )
            xs, ys = zip(*points)
            axis.plot(
                xs,
                ys,
                color=colors[(len(contract_order) + index) % len(colors)],
                marker="o",
                markersize=3.0,
                linestyle="--",
                linewidth=1.1,
                label=f"{method_label} · {incomplete_validation_label(run_id, run_rows)}",
            )

        if not plotted:
            axis.text(
                0.5,
                0.5,
                "No numeric validation points found",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            continue

        axis.set_title(granularity, fontsize=11, pad=6)
        if validation_loss_log_y:
            axis.set_yscale("log", nonpositive="clip")
        axis.set_ylabel("Loss")
        axis.grid(True, which="major", alpha=0.30, linewidth=0.6)
        axis.minorticks_on()
        axis.grid(True, which="minor", alpha=0.15, linewidth=0.3)
        axis.set_axisbelow(True)
        from .reporting import mark_scheduler_warmup

        mark_scheduler_warmup(axis, warmup_token_positions)

    axes[-1].set_xlabel("Total training tokens")
    _apply_common_validation_y_limits(list(axes))
    figure.suptitle(
        "Validation loss: standalone vs uncorrected nested-random methods",
        fontsize=16,
        y=0.98,
    )

    handles_by_label: dict[str, Any] = {}
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            handles_by_label.setdefault(label, handle)
    if handles_by_label:
        figure.legend(
            handles=list(handles_by_label.values()),
            labels=list(handles_by_label),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            frameon=False,
        )

    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.92,
        bottom=0.11,
        hspace=0.35,
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def group_validation_rows_by_method(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        method_key = validation_comparison_method_key(row)
        if method_key is None:
            continue
        grouped.setdefault(method_key, []).append(row)
    return grouped


def validation_comparison_method_key(row: dict[str, str]) -> str | None:
    family_label = scaling_curve_family_label(row)
    if family_label == "standalone":
        return "standalone"
    if family_label != "nested-random":
        return None

    variant_label = scaling_curve_variant_label(row)
    if variant_label not in {"slicing", "concat"}:
        return None

    sampling_label = scaling_curve_sampling_label(row)
    if sampling_label not in {
        "global",
        "per_block",
        "adaptive_per_block_ucb",
        "probabilistic_global_thompson",
        "probabilistic_per_block_thompson",
    }:
        return None

    return f"nested-random / {variant_label} / {sampling_label}"


def validation_comparison_method_order(rows: list[dict[str, str]]) -> list[str]:
    preferred = [
        "standalone",
        "nested-random / slicing / global",
        "nested-random / concat / global",
        "nested-random / slicing / per_block",
        "nested-random / concat / per_block",
        "nested-random / slicing / probabilistic_global_thompson",
        "nested-random / concat / probabilistic_global_thompson",
        "nested-random / slicing / probabilistic_per_block_thompson",
        "nested-random / concat / probabilistic_per_block_thompson",
        "nested-random / slicing / adaptive_per_block_ucb",
        "nested-random / concat / adaptive_per_block_ucb",
    ]
    present = {validation_comparison_method_key(row) for row in rows}
    return [label for label in preferred if label in present]


def validation_comparison_display_label(method_key: str) -> str:
    if method_key == "standalone":
        return "Standalone"
    _, variant_label, sampling_label = method_key.split(" / ")
    sampling_display = {
        "global": "Uniform",
        "per_block": "Random per block",
        "adaptive_per_block_ucb": "UCB per block",
        "probabilistic_global_thompson": "Thompson global",
        "probabilistic_per_block_thompson": "Thompson per block",
    }.get(
        sampling_label,
        display_sampling_label_for_curve(sampling_label) or sampling_label,
    )
    return f"{variant_label.title()} · {sampling_display}"


def validation_comparison_styles(method_keys: list[str]) -> dict[str, dict[str, Any]]:
    variant_colors = {
        "standalone": STANDALONE_REFERENCE_COLOR,
        "slicing": "tab:blue",
        "concat": "tab:orange",
    }
    sampling_linestyles = {
        "global": "-",
        "per_block": "--",
        "probabilistic_global_thompson": "-",
        "probabilistic_per_block_thompson": "--",
        "adaptive_per_block_ucb": ":",
    }
    sampling_markers = {
        "global": "o",
        "per_block": "s",
        "probabilistic_global_thompson": "*",
        "probabilistic_per_block_thompson": "v",
        "adaptive_per_block_ucb": "D",
    }
    styles: dict[str, dict[str, Any]] = {}
    for method_key in method_keys:
        if method_key == "standalone":
            styles[method_key] = {
                "color": variant_colors["standalone"],
                "marker": "o",
                "linestyle": "-",
                "linewidth": 1.5,
                "markersize": 3.5,
            }
            continue

        _, variant_label, sampling_label = method_key.split(" / ")
        styles[method_key] = {
            "color": variant_colors.get(variant_label, "tab:gray"),
            "marker": sampling_markers.get(sampling_label, "o"),
            "linestyle": sampling_linestyles.get(sampling_label, "-"),
            "linewidth": 1.4,
            "markersize": 3.5,
        }
    return styles


def plot_loss_over_tokens_for_experiment(
    rows: list[dict[str, str]],
    figure_label: str,
    output_path: Path,
    dpi: int = 300,
    validation_loss_log_y: bool = False,
) -> Path:
    granularity_rows = [row for row in rows if row.get("granularity") not in (None, "")]
    granularity_labels = validation_granularity_order(granularity_rows)

    if not granularity_labels:
        figure, axis = plt.subplots(figsize=(12, 8))

        axis.text(
            0.5,
            0.5,
            "No granularity metadata found",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()

        figure.suptitle(figure_label, fontsize=16)

        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return output_path

    figure_height = max(
        3.0,
        2.5 * len(granularity_labels),
    )

    figure, axes = plt.subplots(
        len(granularity_labels),
        1,
        figsize=(14, figure_height),
        sharex=True,
    )

    if len(granularity_labels) == 1:
        axes = [axes]

    completed_contracts = group_completed_validation_contracts(granularity_rows)
    contract_labels = compact_validation_contract_labels(completed_contracts)
    contract_order = sorted(completed_contracts, key=lambda key: contract_labels[key])
    prop_cycle = plt.rcParams.get("axes.prop_cycle")
    colors = list(prop_cycle.by_key().get("color", [])) if prop_cycle else []
    if not colors:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    contract_styles = {
        contract: {
            "color": colors[index % len(colors)],
            "marker": markers[index % len(markers)],
            "linestyle": "-",
            "linewidth": 1.4,
            "markersize": 3.5,
        }
        for index, contract in enumerate(contract_order)
    }
    incomplete_runs = group_incomplete_validation_runs(granularity_rows)
    warmup_token_positions = {
        value
        for row in granularity_rows
        if (value := to_float_or_none(row.get("_scheduler_warmup_tokens")))
        is not None
    }

    for axis, granularity in zip(axes, granularity_labels):
        plotted = False
        for contract in contract_order:
            contract_rows = [
                row
                for row in completed_contracts[contract]
                if str(row.get("granularity") or "") == granularity
            ]
            aggregate = aggregate_validation_curve(contract_rows)
            if not aggregate["xs"]:
                continue
            plotted = True
            style = contract_styles[contract]
            axis.plot(
                aggregate["xs"],
                aggregate["means"],
                label=contract_labels[contract],
                **style,
            )
            if any(aggregate["band_mask"]):
                axis.fill_between(
                    aggregate["xs"],
                    aggregate["minimums"],
                    aggregate["maximums"],
                    color=style["color"],
                    alpha=0.14,
                    linewidth=0,
                )

        for index, (run_id, run_rows) in enumerate(sorted(incomplete_runs.items())):
            curve_rows = [
                row
                for row in run_rows
                if str(row.get("granularity") or "") == granularity
            ]
            points = sorted(
                (
                    (to_float(row["tokens_seen"]), to_float(row["loss"]))
                    for row in curve_rows
                    if row.get("tokens_seen") not in (None, "")
                    and row.get("loss") not in (None, "")
                ),
                key=lambda point: point[0],
            )
            if not points:
                continue
            plotted = True
            xs, ys = zip(*points)
            axis.plot(
                xs,
                ys,
                color=colors[(len(contract_order) + index) % len(colors)],
                marker="o",
                markersize=3.0,
                linestyle="--",
                linewidth=1.1,
                label=incomplete_validation_label(run_id, run_rows),
            )

        if not plotted:
            axis.text(
                0.5,
                0.5,
                "No numeric validation points found",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            continue

        axis.set_title(
            granularity,
            fontsize=11,
            pad=6,
        )

        if validation_loss_log_y:
            axis.set_yscale("log", nonpositive="clip")
        axis.set_ylabel("Loss")

        axis.grid(
            True,
            which="major",
            alpha=0.30,
            linewidth=0.6,
        )

        axis.minorticks_on()

        axis.grid(
            True,
            which="minor",
            alpha=0.15,
            linewidth=0.3,
        )

        axis.set_axisbelow(True)
        from .reporting import mark_scheduler_warmup

        mark_scheduler_warmup(axis, warmup_token_positions)

    axes[-1].set_xlabel("Total training tokens")
    _apply_common_validation_y_limits(list(axes))

    figure.suptitle(
        figure_label,
        fontsize=16,
        y=0.98,
    )

    handles_by_label: dict[str, Any] = {}
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            handles_by_label.setdefault(label, handle)
    if handles_by_label:
        figure.legend(
            handles=list(handles_by_label.values()),
            labels=list(handles_by_label),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(len(handles_by_label), 5),
            frameon=False,
        )

    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.92,
        bottom=0.10,
        hspace=0.35,
    )

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def group_validation_rows_by_variant(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(validation_variant_key(row), []).append(row)
    return grouped


def validation_variant_display_labels(rows: list[dict[str, str]]) -> dict[str, str]:
    display_labels: dict[str, str] = {}
    for variant_key in validation_variant_order(rows):
        display_labels[variant_key] = validation_variant_display_label(variant_key)
    return display_labels


def validation_variant_order(rows: list[dict[str, str]]) -> list[str]:
    preferred = ["none", "gmc", "lmc"]
    present = {validation_variant_key(row) for row in rows}
    return [label for label in preferred if label in present]


def validation_variant_display_label(variant_key: str) -> str:
    return variant_key


def validation_variant_styles(variant_keys: list[str]) -> dict[str, dict[str, Any]]:
    prop_cycle = plt.rcParams.get("axes.prop_cycle")
    colors = list(prop_cycle.by_key().get("color", [])) if prop_cycle else []
    if not colors:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    styles: dict[str, dict[str, Any]] = {}
    for index, variant_key in enumerate(variant_keys):
        styles[variant_key] = {
            "color": colors[index % len(colors)],
            "marker": markers[index % len(markers)],
            "linestyle": "-",
            "linewidth": 1.4,
            "markersize": 3.5,
        }
    return styles


def validation_variant_key(row: dict[str, str]) -> str:
    correction_label = scaling_curve_correction_label(row)
    return correction_label or "none"


def plot_consistency_results(
    rows: list[dict[str, str]],
    output_path: Path,
    dpi: int = 300,
) -> Path:
    figure, axis, legend_axis = create_figure_with_side_legend(
        plot_width=10,
        plot_height=5,
        legend_width=4.8,
    )
    numeric_rows = [
        row for row in rows if to_float_or_none(row.get("metric_value")) is not None
    ]

    if not numeric_rows:
        axis.text(
            0.5,
            0.5,
            "No numeric consistency metrics found",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        finalize_side_legend_figure(figure, trace_description="")
        figure.savefig(output_path, bbox_inches="tight", dpi=dpi)
        plt.close(figure)
        return output_path

    pair_labels = sorted(
        {consistency_pair_label(row) for row in numeric_rows},
        key=consistency_pair_sort_key,
    )
    metric_names = sorted(
        {str(row["metric_name"]) for row in numeric_rows},
        key=consistency_metric_sort_key,
    )
    pair_to_metric_values = {
        pair_label: {
            str(row["metric_name"]): to_float(row["metric_value"])
            for row in numeric_rows
            if consistency_pair_label(row) == pair_label
        }
        for pair_label in pair_labels
    }

    group_width = 0.8
    bar_width = group_width / max(len(metric_names), 1)
    offsets = [
        (index - (len(metric_names) - 1) / 2.0) * bar_width
        for index in range(len(metric_names))
    ]
    x_positions = list(range(len(pair_labels)))

    for offset, metric_name in zip(offsets, metric_names):
        values = [
            pair_to_metric_values[pair_label].get(metric_name, float("nan"))
            for pair_label in pair_labels
        ]
        axis.bar(
            [position + offset for position in x_positions],
            values,
            width=bar_width,
            label=metric_name,
        )

    axis.set_xticks(x_positions, pair_labels, rotation=0, ha="center")
    axis.set_xlabel("Granularity pair")
    axis.set_ylabel("Metric value")
    axis.grid(True, axis="y", alpha=0.3)
    place_legend_on_right(legend_axis, axis)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return output_path


def finalize_side_legend_figure(figure, *, trace_description: str) -> None:
    # GridSpec + a hidden legend axis triggers tight_layout warnings in Matplotlib.
    # Use explicit margins instead; bbox_inches='tight' handles the final crop.
    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.88 if trace_description else 0.92,
        bottom=0.14 if trace_description else 0.11,
    )


def write_medium_trend_report(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(build_medium_trend_report_lines(rows)) + "\n",
        encoding="utf-8",
    )
    return output_path


def build_medium_trend_report_lines(rows: list[dict[str, Any]]) -> list[str]:
    source_csvs = sorted(
        {
            str(row["_source_csv"])
            for row in rows
            if row.get("_source_csv") not in (None, "")
        }
    )
    curve_groups = group_scaling_rows(rows)
    lines = [
        "# Medium Trend Report",
        "",
        "Generated from structured scaling and downstream result CSV artifacts.",
        "",
        "## Inputs",
        f"- Scaling rows: {len(rows)}",
    ]

    if source_csvs:
        lines.append(f"- Source CSV files: {format_list(source_csvs)}")

    run_ids = sorted({str(row["run_id"]) for row in rows if row.get("run_id")})
    granularities = sorted(
        {str(row["granularity"]) for row in rows if row.get("granularity")},
        key=granularity_sort_key,
    )
    sampling_modes = sorted(
        {
            str(row["sampling_mode"])
            for row in rows
            if row.get("sampling_mode") not in (None, "")
        }
    )
    lines.extend(
        [
            f"- Runs: {format_list(run_ids)}",
            f"- Granularities: {format_list(granularities)}",
            f"- Sampling modes: {format_list(sampling_modes)}",
            "",
            "## Curve Groups",
        ]
    )

    for label, group_rows_for_label in curve_groups.items():
        group_granularities = sorted(
            {
                str(row["granularity"])
                for row in group_rows_for_label
                if row.get("granularity")
            },
            key=granularity_sort_key,
        )
        lines.append(
            f"- {label}: {len(group_rows_for_label)} rows; "
            f"granularities={format_list(group_granularities)}"
        )

    lines.extend(["", "## Best Observed Points"])
    metric_summaries = [
        summarize_metric(rows, "loss", lower_is_better=True),
        summarize_metric(rows, "perplexity", lower_is_better=True),
        summarize_metric(rows, "average_downstream_accuracy", lower_is_better=False),
    ]
    for summary in metric_summaries:
        if summary is None:
            continue
        lines.append(f"- {summary}")

    return lines


def summarize_metric(
    rows: list[dict[str, Any]],
    metric_name: str,
    lower_is_better: bool,
) -> str | None:
    points = []
    for row in rows:
        metric_value = to_float_or_none(row.get(metric_name))
        if metric_value is None:
            continue
        points.append((metric_value, row))

    if not points:
        return None

    metric_value, row = (
        min(points, key=lambda point: point[0])
        if lower_is_better
        else max(points, key=lambda point: point[0])
    )
    parameters = to_float_or_none(row.get("non_embedding_parameters"))
    parameter_text = (
        "unknown non-embedding parameters"
        if parameters is None
        else f"{parameters:.0f} non-embedding parameters"
    )
    return (
        f"{metric_name}: {metric_value:.6g} at {describe_scaling_row(row)} "
        f"({parameter_text})"
    )


def describe_scaling_row(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("sampling_mode") or row.get("model_family") or "unknown"),
    ]
    if row.get("model_variant") not in (None, ""):
        parts.append(str(row["model_variant"]))
    parts.extend(
        [
            str(row.get("granularity") or "unknown-granularity"),
            str(row.get("run_id") or "unknown-run"),
        ]
    )
    return " / ".join(parts)


def format_list(values: list[str], limit: int = 8) -> str:
    if not values:
        return "none"
    if len(values) <= limit:
        return ", ".join(values)
    shown = ", ".join(values[:limit])
    return f"{shown}, ... ({len(values)} total)"


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


def group_rows(
    rows: list[dict[str, str]], keys: list[str]
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        label = " / ".join(row.get(key, "") for key in keys)
        grouped.setdefault(label, []).append(row)
    return grouped


def group_scaling_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(scaling_curve_label(row), []).append(row)
    return grouped


SIZE_PLOT_CONTRACT_FIELDS = (
    "global_sampling_distribution",
    "global_sampling_schedule_version",
    "controller_method_family",
    "controller_method_version",
    "controller_scope",
    "controller_decision_interval_steps",
    "controller_observation_noise_variance",
    "controller_process_noise_covariance",
    "controller_prior_mean",
    "controller_prior_covariance",
    "controller_context_model",
    "controller_compute_weight",
    "controller_feature_schema_hash",
    "controller_reset_enabled",
    "controller_reset_policy",
    "panelgrad_importance_metric",
    "panelgrad_refresh_interval_steps",
    "panelgrad_eta",
    "panelgrad_temperature",
    "panelgrad_epsilon",
    "panelgrad_epsilon_schedule_type",
    "panelgrad_epsilon_schedule_start",
    "panelgrad_epsilon_schedule_end",
    "panelgrad_epsilon_schedule_duration_steps",
    "controller_reset_interval_steps",
    "controller_acquisition_policy",
    "controller_acquisition_passes",
    "pre_nested_warmup_enabled",
    "pre_nested_warmup_policy",
    "pre_nested_warmup_duration",
    "pre_nested_warmup_action_interval_steps",
    "training_token_budget",
    "training_learning_rate",
    "training_max_steps",
    "training_optimizer_name",
    "training_optimizer_kwargs",
    "training_scheduler_name",
    "training_scheduler_kwargs",
    "training_context_length",
    "training_batch_size_per_process",
    "training_precision",
    "training_dataset_name",
    "training_dataset_config_name",
    "training_dataset_split",
    "training_tokenizer_name",
)

SIZE_PLOT_REQUIRED_BAYESIAN_FIELDS = (
    "controller_decision_interval_steps",
    "controller_observation_noise_variance",
    "controller_process_noise_covariance",
    "controller_prior_mean",
    "controller_prior_covariance",
    "controller_reset_enabled",
    "pre_nested_warmup_enabled",
    "training_token_budget",
    "training_learning_rate",
    "training_scheduler_name",
)


def _canonical_contract_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonical_contract_value(nested_value)
            for key, nested_value in sorted(
                value.items(), key=lambda item: str(item[0])
            )
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_contract_value(item) for item in value]
    if isinstance(value, float) and value == 0.0:
        return 0.0
    return value


def _value_is_missing(value: Any) -> bool:
    return value is None or value == ""


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def size_plot_experiment_contract(row: dict[str, Any]) -> str:
    """Build a stable seed-independent contract for one size-plot curve."""

    payload = {
        "family": scaling_curve_family_label(row),
        "variant": scaling_curve_variant_label(row),
        "sampling": scaling_curve_sampling_label(row),
        "correction": scaling_curve_correction_label(row),
    }
    for field_name in SIZE_PLOT_CONTRACT_FIELDS:
        value = row.get(field_name)
        if not _value_is_missing(value):
            payload[field_name] = _canonical_contract_value(value)

    is_bayesian = (
        str(row.get("controller_method_family") or "").strip().lower()
        == BAYESIAN_CONTROLLER_METHOD_FAMILY
    )
    required_fields = list(SIZE_PLOT_REQUIRED_BAYESIAN_FIELDS) if is_bayesian else []
    if is_bayesian and _truthy(row.get("pre_nested_warmup_enabled")):
        required_fields.extend(
            (
                "pre_nested_warmup_policy",
                "pre_nested_warmup_duration",
                "pre_nested_warmup_action_interval_steps",
            )
        )
    if is_bayesian and _truthy(row.get("controller_reset_enabled")):
        required_fields.extend(
            (
                "controller_reset_policy",
                "controller_reset_interval_steps",
                "controller_acquisition_policy",
                "controller_acquisition_passes",
            )
        )
    if any(_value_is_missing(row.get(field_name)) for field_name in required_fields):
        # Older artifacts did not record enough of the experimental contract to
        # prove equivalence. Keeping the run identity prevents accidental merge.
        payload["historical_run_fallback"] = experiment_label(row)

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def group_size_plot_rows(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(size_plot_experiment_contract(row), []).append(row)
    return grouped


def aggregate_size_curve(
    rows: list[dict[str, Any]], metric_name: str
) -> dict[str, Any]:
    """Average equal-contract seeds by x and retain per-x min/max envelopes."""

    by_x_and_replicate: dict[float, dict[str, list[float]]] = {}
    known_seeds: set[str] = set()
    replicate_ids: set[str] = set()
    for row in rows:
        point = numeric_metric_point(row, metric_name)
        if point is None:
            continue
        x_value, y_value = point
        seed = row.get("run_seed")
        if not _value_is_missing(seed):
            replicate_id = f"seed:{seed}"
            known_seeds.add(str(seed))
        else:
            replicate_id = f"run:{experiment_label(row)}"
        replicate_ids.add(replicate_id)
        by_x_and_replicate.setdefault(x_value, {}).setdefault(replicate_id, []).append(
            y_value
        )

    xs: list[float] = []
    means: list[float] = []
    minimums: list[float] = []
    maximums: list[float] = []
    band_mask: list[bool] = []
    for x_value in sorted(by_x_and_replicate):
        replicate_values = [
            sum(values) / len(values) for values in by_x_and_replicate[x_value].values()
        ]
        xs.append(x_value)
        means.append(sum(replicate_values) / len(replicate_values))
        minimums.append(min(replicate_values))
        maximums.append(max(replicate_values))
        band_mask.append(len(replicate_values) > 1)

    return {
        "xs": xs,
        "means": means,
        "minimums": minimums,
        "maximums": maximums,
        "band_mask": band_mask,
        "seed_count": len(known_seeds),
        "replicate_count": len(replicate_ids),
    }


def group_loss_rows_by_figure(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(loss_figure_label(row), []).append(row)
    return grouped


def experiment_label(row: dict[str, str]) -> str:
    run_id = row.get("run_id")
    if run_id not in (None, ""):
        return str(run_id)
    source_csv = row.get("_source_csv")
    if source_csv not in (None, ""):
        return Path(str(source_csv)).parent.name
    return "unknown-run"


def loss_figure_label(row: dict[str, str]) -> str:
    family_label = scaling_curve_family_label(row)
    if family_label == "unknown":
        resolved_run_mode = str(row.get("resolved_run_mode") or "")
        if resolved_run_mode in {"nested-random", "nested-all", "standalone"}:
            family_label = resolved_run_mode
    if family_label == "standalone":
        return "standalone"

    variant_label = scaling_curve_variant_label(row) or "slicing"
    if family_label == "nested-random":
        sampling_label = scaling_curve_sampling_label(row) or str(
            row.get("resolved_sampling_mode")
            or row.get("granularity_sampling_mode")
            or "global"
        )
        return f"{family_label} / {variant_label} / {sampling_label}"

    return f"{family_label} / {variant_label}"


def standalone_figure_label(row: dict[str, str]) -> str:
    run_id = row.get("run_id")
    granularity = row.get("granularity")
    if run_id not in (None, ""):
        normalized = normalize_standalone_run_id(str(run_id), str(granularity or ""))
        if normalized:
            return normalized

    sampling_mode = row.get("sampling_mode") or row.get("model_family") or "standalone"
    model_size_label = row.get("model_size_label") or row.get("model_shape_label")
    if model_size_label not in (None, ""):
        return f"{sampling_mode}-{model_size_label}"
    return str(sampling_mode)


def normalize_standalone_run_id(run_id: str, granularity: str) -> str:
    parts = run_id.split("-")
    if granularity in parts:
        removed = False
        normalized_parts = []
        for part in parts:
            if not removed and part == granularity:
                removed = True
                continue
            normalized_parts.append(part)
        if normalized_parts:
            return "-".join(normalized_parts)
    return run_id


def loss_trace_kind(rows: list[dict[str, str]]) -> str:
    resolved_run_mode = _first_row_value(rows, "resolved_run_mode")
    resolved_sampling_mode = _first_row_value(rows, "resolved_sampling_mode")
    if resolved_run_mode == "nested-random" and resolved_sampling_mode in {
        "global",
        "fixed_global",
        "per_block",
        "adaptive_per_block",
    }:
        return "run"
    return "granularity"


def group_loss_trace_rows(
    rows: list[dict[str, str]],
    trace_kind: str,
) -> dict[str, list[dict[str, str]]]:
    run_ids = {
        str(row["run_id"]) for row in rows if row.get("run_id") not in (None, "")
    }
    include_run_id = len(run_ids) > 1
    grouped: dict[str, list[dict[str, str]]] = {}

    for row in rows:
        if trace_kind == "run":
            label = str(row.get("run_id") or "unknown-run")
        else:
            label_parts = []
            if include_run_id:
                label_parts.append(str(row.get("run_id") or "unknown-run"))
            label_parts.append(str(row.get("granularity") or "unknown-granularity"))
            label = " / ".join(label_parts)
        grouped.setdefault(label, []).append(row)

    return grouped


def loss_trace_series_sort_key(label: str, trace_kind: str) -> tuple[int, str]:
    if trace_kind == "run":
        return (0, label)
    _, _, granularity = label.rpartition(" / ")
    granularity_rank = granularity_sort_key(granularity)[0] if granularity else 99
    return (granularity_rank, label)


def safe_filename_fragment(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return normalized or "unknown"


def scaling_curve_label(row: dict[str, str]) -> str:
    family_label = scaling_curve_family_label(row)
    if family_label == "standalone":
        return "standalone"

    variant_label = scaling_curve_variant_label(row)
    parts = [family_label]
    if variant_label is not None:
        parts.append(variant_label)

    sampling_label = scaling_curve_sampling_label(row)
    if sampling_label is not None:
        parts.append(sampling_label)

    correction_label = scaling_curve_correction_label(row)
    if correction_label is not None:
        parts.append(correction_label)
    return " / ".join(parts)


def scaling_curve_display_label(
    rows: list[dict[str, str]],
    alias_map: dict[str, str] | None = None,
) -> str:
    row = rows[0]
    family_label = scaling_curve_family_label(row)
    if family_label == "standalone":
        label = "standalone"
        return alias_map.get(label, label) if alias_map else label

    parts = [family_label]
    variant_label = scaling_curve_variant_label(row)
    if variant_label is not None:
        parts.append(variant_label)

    sampling_label = scaling_curve_sampling_label(row)
    display_sampling_label = display_sampling_label_for_curve(sampling_label)
    if display_sampling_label is not None:
        parts.append(display_sampling_label)

    correction_label = scaling_curve_correction_label(row)
    if correction_label is not None:
        parts.append(correction_label)

    label = " / ".join(parts)
    return alias_map.get(label, label) if alias_map else label


def scaling_curve_color_group_label(row: dict[str, str]) -> str:
    family_label = scaling_curve_family_label(row)
    if family_label == "standalone":
        return "standalone"

    variant_label = scaling_curve_variant_label(row) or "slicing"
    if family_label == "nested-random":
        sampling_label = scaling_curve_sampling_label(row) or "global"
        if re.fullmatch(r"uniform_global_h[1-9][0-9]*", sampling_label):
            sampling_label = "uniform_global_window"
        if re.fullmatch(r"balanced_global_h[1-9][0-9]*", sampling_label):
            sampling_label = "balanced_global_window"
        return f"{family_label} / {variant_label} / {sampling_label}"

    return f"{family_label} / {variant_label}"


def scaling_curve_group_label(row: dict[str, str]) -> str:
    family_label = scaling_curve_family_label(row)
    if family_label == "standalone":
        return "standalone"

    variant_label = scaling_curve_variant_label(row) or "slice"
    return f"{family_label} / {variant_label}"


def scaling_curve_family_label(row: dict[str, str]) -> str:
    sampling_mode = row.get("sampling_mode")
    if sampling_mode == "standalone":
        return "standalone"
    if sampling_mode in {"nested-all", "nested-random"}:
        return str(sampling_mode)
    model_family = row.get("model_family")
    if model_family == "standalone":
        return "standalone"
    if model_family in {"nested", "standalone"}:
        return str(model_family)
    return str(sampling_mode or model_family or "unknown")


def scaling_curve_variant_label(row: dict[str, str]) -> str | None:
    variant = row.get("model_variant")
    if variant in (None, ""):
        return None
    normalized = str(variant).strip().lower()
    if normalized in {"cat_llama", "cat"}:
        return "concat"
    if normalized in {"matformer_llama", "slice"}:
        return "slicing"
    return normalized


def scaling_curve_sampling_label(row: dict[str, str]) -> str | None:
    sampling_mode = row.get("sampling_mode")
    if sampling_mode not in {"nested-random", "nested-all"}:
        resolved_run_mode = row.get("resolved_run_mode")
        if resolved_run_mode in {"nested-random", "nested-all"}:
            sampling_mode = str(resolved_run_mode)
        else:
            return None

    resolved_sampling_mode = row.get("resolved_sampling_mode")
    if resolved_sampling_mode not in (None, ""):
        normalized = str(resolved_sampling_mode).strip().lower()
        if normalized == "global":
            schedule = str(
                row.get("global_sampling_schedule")
                or "random_with_replacement"
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
        probabilistic_label = _probabilistic_sampling_label(row)
        if probabilistic_label is not None:
            return probabilistic_label
        if normalized in {"global", "fixed_global", "per_block"}:
            return normalized
        if normalized == "adaptive_per_block":
            strategy = adaptive_sampler_strategy_for_row(row)
            if strategy in {"thompson", "ucb"}:
                return f"adaptive_per_block_{strategy}"
            return normalized

    granularity_sampling_mode = row.get("granularity_sampling_mode")
    if granularity_sampling_mode not in (None, ""):
        normalized = str(granularity_sampling_mode).strip().lower()
        if normalized == "global":
            schedule = str(
                row.get("global_sampling_schedule")
                or "random_with_replacement"
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
        probabilistic_label = _probabilistic_sampling_label(row)
        if probabilistic_label is not None:
            return probabilistic_label
        if normalized in {"global", "fixed_global", "per_block"}:
            return normalized
        if normalized == "adaptive_per_block":
            strategy = adaptive_sampler_strategy_for_row(row)
            if strategy in {"thompson", "ucb"}:
                return f"adaptive_per_block_{strategy}"
            return normalized

    return None


def _probabilistic_sampling_label(row: dict[str, str]) -> str | None:
    method_family = str(row.get("controller_method_family") or "").strip().lower()
    method_version = row.get("controller_method_version")
    strategy = adaptive_sampler_strategy_for_row(row)
    scope = str(row.get("controller_scope") or "").strip().lower()
    if (
        method_family in PANELGRAD_METHOD_FAMILIES
        and method_version not in (None, "")
        and strategy == "panelgrad"
        and scope == "global"
    ):
        return "panelgrad_global"
    if (
        method_family != BAYESIAN_CONTROLLER_METHOD_FAMILY
        or method_version in (None, "")
        or strategy != "thompson"
        or scope not in {"global", "per_block"}
    ):
        return None
    if scope == "global":
        reset_enabled = str(row.get("controller_reset_enabled", "")).strip().lower()
        if reset_enabled in {"1", "true", "yes"}:
            reset_policy = (
                str(row.get("controller_reset_policy", "full_prior")).strip().lower()
            )
            if reset_policy == "acquisition_only":
                return "probabilistic_global_thompson_acquisition_only"
            return "probabilistic_global_thompson_reset"
        return "probabilistic_global_thompson"
    return "probabilistic_per_block_thompson"


def is_legacy_heuristic_thompson_row(row: dict[str, Any]) -> bool:
    resolved_sampling_mode = row.get("resolved_sampling_mode")
    if resolved_sampling_mode in (None, ""):
        resolved_sampling_mode = row.get("granularity_sampling_mode")
    normalized_mode = str(resolved_sampling_mode or "").strip().lower()
    if normalized_mode != "adaptive_per_block":
        return False

    if adaptive_sampler_strategy_for_row(row) != "thompson":
        return False

    return _probabilistic_sampling_label(row) is None


def row_matches_plot_filters(
    row: dict[str, Any],
    *,
    variants: list[str] | tuple[str, ...] | None = None,
    corrections: list[str] | tuple[str, ...] | None = None,
) -> bool:
    """Match one artifact row against explicit experiment-facing filters."""

    if variants:
        allowed_variants = {str(value).strip().lower() for value in variants}
        if scaling_curve_variant_label(row) not in allowed_variants:
            return False
    if corrections:
        allowed_corrections = {str(value).strip().lower() for value in corrections}
        correction = scaling_curve_correction_label(row) or "none"
        if correction not in allowed_corrections:
            return False
    return True


def filter_plot_rows(
    rows: list[dict[str, Any]],
    *,
    variants: list[str] | tuple[str, ...] | None = None,
    corrections: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Exclude unsafe history and apply optional variant/correction filters."""

    return [
        row
        for row in rows
        if not is_legacy_heuristic_thompson_row(row)
        and row_matches_plot_filters(
            row,
            variants=variants,
            corrections=corrections,
        )
    ]


def adaptive_sampler_strategy_for_row(row: dict[str, str]) -> str | None:
    value = row.get("adaptive_sampler_strategy")
    if value in (None, ""):
        return None
    return str(value).strip().lower()


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


def scaling_curve_correction_label(row: dict[str, str]) -> str | None:
    correction_mode = row.get("correction_mode")
    if correction_mode not in (None, ""):
        normalized = str(correction_mode).strip().lower()
        if normalized in {"gmc", "lmc"}:
            return normalized
        return None

    if (
        row.get("model_family") == "standalone"
        or row.get("sampling_mode") == "standalone"
    ):
        return None

    raw_value = row.get("membership_correction")
    if raw_value in (None, ""):
        raw_value = row.get("gradient_membership_correction")
    if raw_value in (None, ""):
        return None
    if isinstance(raw_value, bool):
        enabled = raw_value
    else:
        normalized = str(raw_value).strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            enabled = True
        elif normalized in {"false", "0", "no", "off"}:
            enabled = False
        else:
            enabled = bool(raw_value)
    return "gmc" if enabled else None


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
                r"uniform_global_h[1-9][0-9]*",
                actual_sampling_label,
            )
        )
    if expected_sampling_label == "balanced_global_window":
        return bool(
            actual_sampling_label
            and re.fullmatch(
                r"balanced_global_h[1-9][0-9]*",
                actual_sampling_label,
            )
        )
    if expected_sampling_label == "probabilistic_global_thompson":
        return actual_sampling_label in {
            "probabilistic_global_thompson",
            "probabilistic_global_thompson_reset",
            "probabilistic_global_thompson_acquisition_only",
        }
    return actual_sampling_label == expected_sampling_label


def scaling_curve_style(
    rows: list[dict[str, str]],
    style_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    style_config = style_config or resolve_plot_style("default")
    group_key = None
    color_group_key = None
    correction_label = None
    sampling_label = None
    for row in rows:
        group_label = scaling_curve_group_label(row)
        correction_label = scaling_curve_correction_label(row)
        sampling_label = scaling_curve_sampling_label(row)
        if group_label:
            group_key = group_label
            color_group_key = scaling_curve_color_group_label(row)
            break

    correction_style = reporting_styles.SCALING_CORRECTION_STYLES.get(
        correction_label or "none",
        reporting_styles.SCALING_CORRECTION_STYLES["none"],
    )
    base_color = style_config["series_colors"].get(
        color_group_key or "",
        reporting_styles.SCALING_GROUP_COLORS.get(color_group_key or "", "tab:gray"),
    )
    tone_sampling_label = sampling_label or "global"
    marker_sampling_label = sampling_label or ""
    if re.fullmatch(r"uniform_global_h[1-9][0-9]*", tone_sampling_label):
        tone_sampling_label = "uniform_global_window"
        marker_sampling_label = "uniform_global_window"
    if re.fullmatch(r"balanced_global_h[1-9][0-9]*", tone_sampling_label):
        tone_sampling_label = "balanced_global_window"
        marker_sampling_label = "balanced_global_window"
    sampling_tone = reporting_styles.SCALING_SAMPLING_TONES.get(
        tone_sampling_label, 0.0
    )
    style = {
        "linewidth": 1.4,
        "linestyle": correction_style["linestyle"],
        "color": blend_color_toward_white(
            base_color,
            combine_shades(sampling_tone, correction_style["shade"]),
        ),
        "markersize": 5,
    }
    style["marker"] = reporting_styles.SCALING_SAMPLING_MARKERS.get(
        marker_sampling_label,
        correction_style["marker"],
    )
    if group_key == "standalone":
        style["linewidth"] = 1.6
    return style


def no_corrections_row_filter(row: dict[str, str]) -> bool:
    family_label = scaling_curve_family_label(row)
    if family_label == "standalone":
        return True
    return scaling_curve_correction_label(row) is None


def resolve_figure_row_filter(
    row_filter_name: str | None,
) -> Callable[[dict[str, str]], bool] | None:
    if row_filter_name is None:
        return None
    if row_filter_name == "no_corrections":
        return no_corrections_row_filter
    raise ValueError(f"Unknown figure row filter: {row_filter_name}")


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


def consistency_pair_label(row: dict[str, Any]) -> str:
    return f"{row['small_granularity']} -> {row['large_granularity']}"


def consistency_pair_sort_key(
    value: str,
) -> tuple[tuple[int, str], tuple[int, str], str]:
    left, _, right = value.partition(" -> ")
    return (
        granularity_sort_key(left),
        granularity_sort_key(right),
        value,
    )


def consistency_metric_sort_key(value: str) -> tuple[int, int, str]:
    if value == "token_level_agreement":
        return (0, 0, value)
    if value.startswith("top_k_overlap@"):
        try:
            return (1, int(value.split("@", 1)[1]), value)
        except ValueError:
            return (1, 0, value)
    if value == "kl_divergence_deferred":
        return (2, 0, value)
    return (3, 0, value)


def to_float(value: Any) -> float:
    return float(value)


def moving_average(values: list[float], window_size: int) -> list[float]:
    if window_size <= 1 or len(values) <= 1:
        return values

    smoothed = []
    left_radius = (window_size - 1) // 2
    right_radius = window_size // 2

    for index in range(len(values)):
        start = max(0, index - left_radius)
        end = min(len(values), index + right_radius + 1)
        window = values[start:end]
        smoothed.append(sum(window) / len(window))

    return smoothed


def loss_moving_average_window_size(point_count: int) -> int:
    if point_count <= 1:
        return point_count

    window_size = max(
        3, math.ceil(point_count * reporting_styles.LOSS_MOVING_AVERAGE_FRACTION)
    )
    if window_size % 2 == 0:
        window_size += 1
    if window_size > point_count:
        window_size = point_count if point_count % 2 == 1 else point_count - 1
    return max(1, window_size)


def loss_trace_description(
    rows: list[dict[str, str]],
    *,
    validation: bool = False,
) -> str:
    if not rows:
        return ""

    resolved_run_mode = _first_row_value(rows, "resolved_run_mode")
    resolved_sampling_mode = _first_row_value(rows, "resolved_sampling_mode")
    sampling_mode = _first_row_value(rows, "sampling_mode")

    if validation:
        return (
            "Validation evaluates each granularity independently, so each "
            "curve is a per-granularity validation loss trace."
        )

    if resolved_run_mode == "nested-all":
        return (
            "nested-all evaluates every configured granularity on each step, "
            "so these are per-granularity training loss traces."
        )
    if resolved_run_mode == "standalone":
        return (
            "standalone keeps one fixed granularity for the whole run, so "
            "each curve is a fixed-granularity training loss trace."
        )
    if resolved_sampling_mode == "per_block":
        return (
            "nested-random + per_block logs one shared step loss across the "
            "selected granularities for each step."
        )
    if resolved_sampling_mode == "adaptive_per_block":
        return (
            "nested-random + adaptive_per_block logs one shared step loss "
            "across the selected granularities for each step."
        )
    if resolved_sampling_mode in {"global", "fixed_global"} or sampling_mode == "global":
        return (
            f"nested-random + {resolved_sampling_mode or 'global'} samples one "
            "granularity per step, so each "
            "curve is a sampled training loss trace."
        )
    return ""


def loss_trace_panel_suffix(
    rows: list[dict[str, str]],
    *,
    validation: bool = False,
) -> str:
    if not rows:
        return ""

    resolved_run_mode = _first_row_value(rows, "resolved_run_mode")
    resolved_sampling_mode = _first_row_value(rows, "resolved_sampling_mode")
    sampling_mode = _first_row_value(rows, "sampling_mode")

    if validation:
        return "validation loss"
    if resolved_run_mode == "nested-all":
        return "training loss"
    if resolved_run_mode == "standalone":
        return "fixed training loss"
    if resolved_sampling_mode == "per_block":
        return "shared step loss"
    if resolved_sampling_mode == "adaptive_per_block":
        return "adaptive shared step loss"
    if resolved_sampling_mode in {"global", "fixed_global"} or sampling_mode == "global":
        return "sampled training loss"
    return ""


def _first_row_value(rows: list[dict[str, str]], key: str) -> str:
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def to_float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
