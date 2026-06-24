"""Generate plots from structured CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any
from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

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
SIZE_PLOT_PANELS_DEFAULT = [
    ("nested-random", "slicing", None),
    ("nested-random", "concat", None),
    ("nested-all", "slicing", None),
    ("nested-all", "concat", None),
]
SIZE_PLOT_PANELS_WITH_SAMPLING = [
    ("nested-random", "slicing", "global"),
    ("nested-random", "concat", "global"),
    ("nested-random", "slicing", "per_block"),
    ("nested-random", "concat", "per_block"),
    ("nested-random", "slicing", "adaptive_per_block_thompson"),
    ("nested-random", "concat", "adaptive_per_block_thompson"),
    ("nested-random", "slicing", "adaptive_per_block_ucb"),
    ("nested-random", "concat", "adaptive_per_block_ucb"),
    ("nested-all", "slicing", None),
    ("nested-all", "concat", None),
]
SCALING_GROUP_COLORS = {
    "nested-random / slicing / global": "tab:blue",
    "nested-random / slicing / per_block": "tab:cyan",
    "nested-random / slicing / adaptive_per_block_thompson": "tab:green",
    "nested-random / slicing / adaptive_per_block_ucb": "tab:olive",
    "nested-random / concat / global": "tab:orange",
    "nested-random / concat / per_block": "tab:red",
    "nested-random / concat / adaptive_per_block_thompson": "tab:purple",
    "nested-random / concat / adaptive_per_block_ucb": "tab:pink",
    "nested-all / slicing": "tab:purple",
    "nested-all / concat": "tab:green",
    "standalone": "tab:brown",
}
SCALING_CORRECTION_STYLES = {
    "none": {"linestyle": "-", "marker": "o", "shade": 0.0},
    "gmc": {"linestyle": "--", "marker": "s", "shade": 0.2},
    "lmc": {"linestyle": "-.", "marker": "^", "shade": 0.35},
}
SCALING_SAMPLING_TONES = {
    "global": 0.0,
    "per_block": 0.28,
    "adaptive_per_block_thompson": 0.4,
    "adaptive_per_block_ucb": 0.55,
}
SCALING_SAMPLING_MARKERS = {
    "global": "o",
    "per_block": "D",
    "adaptive_per_block_thompson": "P",
    "adaptive_per_block_ucb": "X",
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
            "standalone": "tab:brown",
        },
    },
    "nested_random_no_corrections": {
        "figure_title_fontsize": 15,
        "curve_aliases": {
            "nested-random / slicing / global": "nested-random / slicing / global",
            "nested-random / concat / global": "nested-random / concat / global",
            "nested-random / slicing / per_block": "nested-random / slicing / per_block",
            "nested-random / concat / per_block": "nested-random / concat / per_block",
            "nested-random / slicing / adaptive_per_block_thompson": "nested-random / slicing / adaptive_per_block_thompson",
            "nested-random / concat / adaptive_per_block_thompson": "nested-random / concat / adaptive_per_block_thompson",
            "nested-random / slicing / adaptive_per_block_ucb": "nested-random / slicing / adaptive_per_block_ucb",
            "nested-random / concat / adaptive_per_block_ucb": "nested-random / concat / adaptive_per_block_ucb",
        },
        "series_colors": {
            "nested-random / slicing / global": "tab:blue",
            "nested-random / concat / global": "tab:orange",
            "nested-random / slicing / per_block": "tab:cyan",
            "nested-random / concat / per_block": "tab:red",
            "nested-random / slicing / adaptive_per_block_thompson": "tab:green",
            "nested-random / concat / adaptive_per_block_thompson": "tab:purple",
            "nested-random / slicing / adaptive_per_block_ucb": "tab:olive",
            "nested-random / concat / adaptive_per_block_ucb": "tab:pink",
            "standalone": "tab:brown",
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
            "nested-random / concat / lmc": "Concat/LMC",
            "nested-random / concat / gmc": "Concat/GMC",
            "nested-all / slicing / none / global": "Slicing",
            "nested-all / concat / none / global": "Concat",
            "nested-all / concat / lmc": "Concat/LMC",
            "nested-all / concat / gmc": "Concat/GMC",
        },
        "series_colors": {
            "standalone": "tab:brown",
            "nested-random / slicing / none / global": "tab:red",
            "nested-random / concat / none / global": "tab:blue",
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
    )
    for path in figure_paths:
        print(path)


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
    parser.add_argument(
        "--validation-loss-log-y",
        action="store_true",
        help="Render validation loss figures with a logarithmic y axis.",
    )
    return parser.parse_args(argv)


def generate_figures(
    input_root: str | Path,
    output_dir: str | Path,
    refresh_counts: bool = True,
    dpi: int = 300,
    validation_loss_log_y: bool = False,
) -> list[Path]:
    input_root = Path(input_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = []
    scaling_rows = read_csv_artifacts(input_root, "scaling_results.csv")
    scaling_rows = enrich_scaling_metadata_from_run_config(input_root, scaling_rows)
    if refresh_counts:
        scaling_rows = refresh_scaling_parameter_counts(input_root, scaling_rows)
    task_result_rows = read_csv_artifacts(input_root, "task_results.csv")
    consistency_rows = read_csv_artifacts(input_root, "consistency_results.csv")

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
                    row_filter=resolve_figure_row_filter(figure_spec["row_filter_name"]),
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
        metrics_rows = read_csv_artifacts_filtered(
            input_root,
            "metrics.csv",
            row_filter=validation_split_filter,
        )
        metrics_rows = enrich_metrics_metadata_from_run_config(input_root, metrics_rows)
        figure_paths.extend(
            plot_validation_loss_over_tokens_by_experiment(
                metrics_rows,
                output_dir,
                dpi=dpi,
                validation_loss_log_y=validation_loss_log_y,
            )
        )
        figure_paths.extend(
            plot_validation_loss_over_tokens_by_granularity_comparison(
                metrics_rows,
                output_dir,
                dpi=dpi,
                validation_loss_log_y=validation_loss_log_y,
            )
        )
    else:
        metrics_rows = read_csv_artifacts(input_root, "metrics.csv")
        metrics_rows = enrich_metrics_metadata_from_run_config(input_root, metrics_rows)
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
                )
            )
            figure_paths.extend(
                plot_validation_loss_over_tokens_by_granularity_comparison(
                    validation_metrics_rows,
                    output_dir,
                    dpi=dpi,
                    validation_loss_log_y=validation_loss_log_y,
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
            membership_correction = (
                membership_correction_from_saved_config(
                    config_cache[config_path]
                )
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
    dpi: int = 300,
) -> list[Path]:
    panel_specs = panel_specs or reporting_styles.SIZE_PLOT_PANELS_DEFAULT
    style_config = resolve_plot_style(style)
    plot_rows = rows if row_filter is None else [row for row in rows if row_filter(row)]
    column_count = 2 if len(panel_specs) > 1 else 1
    row_count = math.ceil(len(panel_specs) / column_count)
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
        panel_specs,
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

    row_limits = metric_row_limits_for_panel_specs(
        axes_list,
        panel_specs,
        column_count,
    )
    for row_index, row_limit in enumerate(row_limits):
        if row_limit is None:
            continue
        start = row_index * column_count
        end = min(start + column_count, len(axes_list))
        for axis in axes_list[start:end]:
            axis.set_ylim(*row_limit)

    figure.suptitle(
        figure_title or f"{ylabel} vs Non-embedding parameters",
        fontsize=style_config["figure_title_fontsize"],
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(figure)

    output_paths = [output_path]
    panel_stem = output_path.stem
    if figure_alias:
        panel_stem = f"{panel_stem}__{safe_filename_fragment(figure_alias)}"
    for panel_spec in panel_specs:
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


def plot_metric_vs_size_panel_figure(
    rows: list[dict[str, str]],
    metric_name: str,
    ylabel: str,
    panel_spec: tuple[str, str, str | None],
    output_path: Path,
    style_config: dict[str, Any],
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
) -> Path:
    style_config = resolve_plot_style(style)
    figure = plt.figure(figsize=(15.0, 8.0))
    left_subfigure, right_subfigure = figure.subfigures(1, 2, wspace=0.06)
    left_axis = left_subfigure.subplots()
    right_axis = right_subfigure.subplots()

    left_values = plot_metric_vs_size_split_panel(
        left_axis,
        rows,
        metric_name=metric_name,
        ylabel=ylabel,
        panel_spec=left_panel_spec,
        style_config=style_config,
    )
    right_values = plot_metric_vs_size_split_panel(
        right_axis,
        rows,
        metric_name=metric_name,
        ylabel=ylabel,
        panel_spec=right_panel_spec,
        style_config=style_config,
    )

    shared_values = left_values + right_values
    if shared_values:
        shared_limits = padded_limits(min(shared_values), max(shared_values))
        left_axis.set_ylim(*shared_limits)
        right_axis.set_ylim(*shared_limits)

    left_subfigure.suptitle(
        str(left_panel_spec["subfigure_title"]),
        fontsize=style_config["subfigure_title_fontsize"],
        y=0.88,
    )
    right_subfigure.suptitle(
        str(right_panel_spec["subfigure_title"]),
        fontsize=style_config["subfigure_title_fontsize"],
        y=0.88,
    )
    figure.suptitle(
        figure_title,
        fontsize=style_config["figure_title_fontsize"],
        y=0.985,
    )
    figure.subplots_adjust(top=0.83, bottom=0.12, left=0.06, right=0.98, wspace=0.07)
    figure.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(figure)
    return output_path


def plot_metric_vs_size_split_panel(
    axis,
    rows: list[dict[str, str]],
    metric_name: str,
    ylabel: str,
    panel_spec: dict[str, Any],
    style_config: dict[str, Any],
) -> list[float]:
    series_keys = list(panel_spec["series_keys"])
    panel_rows = [
        row
        for row in rows
        if comparison_series_key(row) in series_keys
    ]

    axis.set_xlabel("Non-embedding parameters", fontsize=style_config["axis_label_fontsize"])
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
            (to_float(row["non_embedding_parameters"]), to_float(row[metric_name]))
            for row in series_rows
            if row.get("non_embedding_parameters") not in (None, "")
            and row.get(metric_name) not in (None, "")
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
                s=42,
                color=style_config["series_colors"].get(
                    series_key,
                    reporting_styles.SCALING_GROUP_COLORS["standalone"],
                ),
                label=resolve_series_alias(series_key, style_config),
                zorder=3,
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
        and panel_sampling_matches(
            scaling_curve_sampling_label(row),
            sampling_label,
        )
        ]
    panel_title = f"{sampling_mode} / {variant_label}"
    if sampling_label is not None:
        panel_title = f"{panel_title} / {sampling_label}"
    axis.set_title(panel_title, fontsize=style_config["panel_title_fontsize"], pad=6)
    axis.set_xlabel("Non-embedding parameters", fontsize=style_config["axis_label_fontsize"])
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

    grouped = group_scaling_rows(panel_rows)
    for group_rows_for_label in grouped.values():
        style = scaling_curve_style(
            group_rows_for_label,
            style_config=style_config,
        )
        legend_label = scaling_curve_display_label(
            group_rows_for_label,
            alias_map=style_config["curve_aliases"],
        )
        points = [
            (to_float(row["non_embedding_parameters"]), to_float(row[metric_name]))
            for row in group_rows_for_label
            if row.get("non_embedding_parameters") not in (None, "")
            and row.get(metric_name) not in (None, "")
        ]
        if not points:
            continue
        points.sort(key=lambda point: point[0])
        xs, ys = zip(*points)
        axis.plot(xs, ys, label=legend_label, **style)

    standalone_points = [
        (to_float(row["non_embedding_parameters"]), to_float(row[metric_name]))
        for row in rows
        if scaling_curve_family_label(row) == "standalone"
        and row.get("non_embedding_parameters") not in (None, "")
        and row.get(metric_name) not in (None, "")
    ]
    if standalone_points:
        standalone_points.sort(key=lambda point: point[0])
        xs, ys = zip(*standalone_points)
        axis.scatter(
            xs,
            ys,
            marker="^",
            s=42,
            color=style_config["series_colors"].get(
                "standalone",
                reporting_styles.SCALING_GROUP_COLORS["standalone"],
            ),
            label=style_config["standalone_label"],
            zorder=3,
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
            "color": style_config["series_colors"].get(
                series_key,
                    reporting_styles.SCALING_GROUP_COLORS["standalone"],
            ),
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
            if row.get("step") not in (None, "") and row.get(metric_name) not in (None, "")
        ]
        if not points:
            continue
        points.sort(key=lambda point: point[0])
        xs, ys = zip(*points)
        axis.plot(xs, ys, marker="o", label=label)

    axis.set_xlabel("Step")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    place_legend_on_right(legend_axis, axis)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return output_path


def plot_validation_loss_over_tokens_by_experiment(
    rows: list[dict[str, str]],
    output_dir: Path,
    dpi: int = 300,
    validation_loss_log_y: bool = False,
) -> list[Path]:
    output_paths = []
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
) -> list[Path]:
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
    granularity_rows = [
        row for row in rows
        if row.get("granularity") not in (None, "")
    ]

    granularity_labels = sorted(
        {str(row["granularity"]) for row in granularity_rows},
        key=granularity_sort_key,
    )

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

    method_keys = validation_comparison_method_order(rows)
    method_styles = validation_comparison_styles(method_keys)
    method_labels = {
        method_key: validation_comparison_display_label(method_key)
        for method_key in method_keys
    }
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=method_styles[method_key]["color"],
            marker=method_styles[method_key]["marker"],
            linestyle=method_styles[method_key]["linestyle"],
            linewidth=method_styles[method_key]["linewidth"],
            markersize=method_styles[method_key]["markersize"],
            label=method_labels[method_key],
        )
        for method_key in method_keys
    ]

    for axis, granularity in zip(axes, granularity_labels):
        sub_rows = [
            row
            for row in granularity_rows
            if str(row.get("granularity") or "") == granularity
        ]
        method_groups = group_validation_rows_by_method(sub_rows)

        if not method_groups:
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

        for method_key in method_keys:
            method_rows = method_groups.get(method_key)
            if not method_rows:
                continue

            points = [
                (
                    to_float(row["tokens_seen"]),
                    to_float(row["loss"]),
                )
                for row in method_rows
                if row.get("tokens_seen") not in (None, "")
                and row.get("loss") not in (None, "")
            ]

            if not points:
                continue

            points.sort(key=lambda point: point[0])
            xs, ys = zip(*points)
            axis.plot(
                xs,
                ys,
                label=method_labels[method_key],
                **method_styles[method_key],
            )

        axis.set_title(granularity, fontsize=11, pad=6)
        if validation_loss_log_y:
            axis.set_yscale("log", nonpositive="clip")
        axis.set_ylabel("Loss")
        axis.grid(True, which="major", alpha=0.30, linewidth=0.6)
        axis.minorticks_on()
        axis.grid(True, which="minor", alpha=0.15, linewidth=0.3)
        axis.set_axisbelow(True)

    axes[-1].set_xlabel("Tokens seen")
    figure.suptitle(
        "Validation loss: standalone vs uncorrected nested-random methods",
        fontsize=16,
        y=0.98,
    )

    if legend_handles:
        figure.legend(
            handles=legend_handles,
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
        "adaptive_per_block_thompson",
        "adaptive_per_block_ucb",
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
        "nested-random / slicing / adaptive_per_block_thompson",
        "nested-random / concat / adaptive_per_block_thompson",
        "nested-random / slicing / adaptive_per_block_ucb",
        "nested-random / concat / adaptive_per_block_ucb",
    ]
    present = {validation_comparison_method_key(row) for row in rows}
    return [label for label in preferred if label in present]


def validation_comparison_display_label(method_key: str) -> str:
    if method_key == "standalone":
        return "standalone"
    _, variant_label, sampling_label = method_key.split(" / ")
    return f"{variant_label} / {display_sampling_label_for_curve(sampling_label) or sampling_label}"


def validation_comparison_styles(method_keys: list[str]) -> dict[str, dict[str, Any]]:
    variant_colors = {
        "standalone": "tab:brown",
        "slicing": "tab:blue",
        "concat": "tab:orange",
    }
    sampling_linestyles = {
        "global": "-",
        "per_block": "--",
        "adaptive_per_block_thompson": "-.",
        "adaptive_per_block_ucb": ":",
    }
    sampling_markers = {
        "global": "o",
        "per_block": "s",
        "adaptive_per_block_thompson": "^",
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
    granularity_rows = [
        row for row in rows
        if row.get("granularity") not in (None, "")
    ]

    granularity_labels = sorted(
        {str(row["granularity"]) for row in granularity_rows},
        key=granularity_sort_key,
    )

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

    variant_display_labels = validation_variant_display_labels(rows)
    variant_keys = validation_variant_order(rows)
    variant_styles = validation_variant_styles(variant_keys)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=variant_styles[variant_key]["color"],
            marker=variant_styles[variant_key]["marker"],
            linestyle=variant_styles[variant_key]["linestyle"],
            linewidth=variant_styles[variant_key]["linewidth"],
            markersize=variant_styles[variant_key]["markersize"],
            label=variant_display_labels[variant_key],
        )
        for variant_key in variant_keys
    ]

    for axis, granularity in zip(axes, granularity_labels):
        sub_rows = [
            row
            for row in granularity_rows
            if str(row.get("granularity") or "") == granularity
        ]

        variant_groups = group_validation_rows_by_variant(sub_rows)

        if not variant_groups:
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

        for variant_key in variant_keys:
            variant_rows = variant_groups.get(variant_key)

            if not variant_rows:
                continue

            points = [
                (
                    to_float(row["tokens_seen"]),
                    to_float(row["loss"]),
                )
                for row in variant_rows
                if row.get("tokens_seen") not in (None, "")
                and row.get("loss") not in (None, "")
            ]

            if not points:
                continue

            points.sort(key=lambda point: point[0])

            xs, ys = zip(*points)

            axis.plot(
                xs,
                ys,
                label=variant_display_labels[variant_key],
                **variant_styles[variant_key],
            )

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

    axes[-1].set_xlabel("Tokens seen")

    figure.suptitle(
        figure_label,
        fontsize=16,
        y=0.98,
    )

    if legend_handles:
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(len(legend_handles), 5),
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
        {
            consistency_pair_label(row)
            for row in numeric_rows
        },
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
    order = {"s": 0, "m": 1, "l": 2, "xl": 3}
    return (order.get(value, len(order)), value)


def group_rows(rows: list[dict[str, str]], keys: list[str]) -> dict[str, list[dict[str, str]]]:
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


def group_loss_rows_by_figure(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
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
        "per_block",
        "adaptive_per_block",
    }:
        return "run"
    return "granularity"


def group_loss_trace_rows(
    rows: list[dict[str, str]],
    trace_kind: str,
) -> dict[str, list[dict[str, str]]]:
    run_ids = {str(row["run_id"]) for row in rows if row.get("run_id") not in (None, "")}
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
        if normalized in {"global", "per_block"}:
            return normalized
        if normalized == "adaptive_per_block":
            strategy = adaptive_sampler_strategy_for_row(row)
            if strategy in {"thompson", "ucb"}:
                return f"adaptive_per_block_{strategy}"
            return normalized

    granularity_sampling_mode = row.get("granularity_sampling_mode")
    if granularity_sampling_mode not in (None, ""):
        normalized = str(granularity_sampling_mode).strip().lower()
        if normalized in {"global", "per_block"}:
            return normalized
        if normalized == "adaptive_per_block":
            strategy = adaptive_sampler_strategy_for_row(row)
            if strategy in {"thompson", "ucb"}:
                return f"adaptive_per_block_{strategy}"
            return normalized

    return None


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
    if sampling_label == "per_block":
        return "per_block sampling"
    if sampling_label == "adaptive_per_block_thompson":
        return "adaptive per-block thompson"
    if sampling_label == "adaptive_per_block_ucb":
        return "adaptive per-block ucb"
    return sampling_label


def scaling_curve_correction_label(row: dict[str, str]) -> str | None:
    correction_mode = row.get("correction_mode")
    if correction_mode not in (None, ""):
        normalized = str(correction_mode).strip().lower()
        if normalized in {"gmc", "lmc"}:
            return normalized
        return None

    if row.get("model_family") == "standalone" or row.get("sampling_mode") == "standalone":
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
    sampling_tone = reporting_styles.SCALING_SAMPLING_TONES.get(sampling_label or "global", 0.0)
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
        sampling_label or "",
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


def consistency_pair_sort_key(value: str) -> tuple[tuple[int, str], tuple[int, str], str]:
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

    window_size = max(3, math.ceil(point_count * reporting_styles.LOSS_MOVING_AVERAGE_FRACTION))
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
    if resolved_sampling_mode == "global" or sampling_mode == "global":
        return (
            "nested-random + global samples one granularity per step, so each "
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
    if resolved_sampling_mode == "global" or sampling_mode == "global":
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
