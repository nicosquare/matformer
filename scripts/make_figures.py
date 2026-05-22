"""Generate plots from structured CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    figure_paths = generate_figures(
        args.input,
        args.output,
        refresh_counts=not args.no_refresh_counts,
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
    return parser.parse_args(argv)


def generate_figures(
    input_root: str | Path,
    output_dir: str | Path,
    refresh_counts: bool = True,
) -> list[Path]:
    input_root = Path(input_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = []
    scaling_rows = read_csv_artifacts(input_root, "scaling_results.csv")
    scaling_rows = enrich_scaling_metadata_from_run_config(input_root, scaling_rows)
    if refresh_counts:
        scaling_rows = refresh_scaling_parameter_counts(input_root, scaling_rows)
    metrics_rows = read_csv_artifacts(input_root, "metrics.csv")
    task_result_rows = read_csv_artifacts(input_root, "task_results.csv")
    consistency_rows = read_csv_artifacts(input_root, "consistency_results.csv")

    if scaling_rows and task_result_rows:
        from evaluation.validation import aggregate_scaling_summary

        scaling_rows = aggregate_scaling_summary(scaling_rows, task_result_rows)

    if scaling_rows:
        figure_paths.append(
            plot_metric_vs_size(
                scaling_rows,
                metric_name="loss",
                ylabel="Loss",
                output_path=output_dir / "loss_vs_size.png",
            )
        )
        figure_paths.append(
            plot_metric_vs_size(
                scaling_rows,
                metric_name="perplexity",
                ylabel="Perplexity",
                output_path=output_dir / "ppl_vs_size.png",
            )
        )
        if any(row.get("average_downstream_accuracy") for row in scaling_rows):
            figure_paths.append(
                plot_metric_vs_size(
                    scaling_rows,
                    metric_name="average_downstream_accuracy",
                    ylabel="Average downstream accuracy",
                    output_path=output_dir / "accuracy_vs_size.png",
                )
            )
        figure_paths.append(
            write_medium_trend_report(
                scaling_rows,
                output_dir / "medium_trend_report.md",
            )
        )

    if metrics_rows and not scaling_rows:
        figure_paths.append(
            plot_metric_over_steps(
                metrics_rows,
                metric_name="perplexity",
                ylabel="Perplexity",
                output_path=output_dir / "ppl_over_steps.png",
            )
        )

    if metrics_rows:
        figure_paths.extend(plot_loss_over_steps_by_experiment(metrics_rows, output_dir))

    if consistency_rows:
        figure_paths.append(
            plot_consistency_results(
                consistency_rows,
                output_dir / "consistency_vs_size.png",
            )
        )

    return figure_paths


def read_csv_artifacts(input_root: Path, filename: str) -> list[dict[str, str]]:
    rows = []
    for path in sorted(input_root.rglob(filename)):
        with path.open("r", encoding="utf-8", newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                row["_source_csv"] = str(path)
                rows.append(row)
    return rows


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
                for field_name in PARAMETER_COUNT_FIELDS:
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
    from training.run import build_model
    from utils.metrics import build_parameter_counts_by_granularity

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
) -> Path:
    figure, axis = plt.subplots(figsize=(10, 5))
    grouped = group_scaling_rows(rows)

    for label, group_rows_for_label in grouped.items():
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
        axis.plot(xs, ys, marker="o", label=label)

    axis.set_xlabel("Non-embedding parameters")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    place_legend_outside(axis)
    figure.tight_layout(rect=[0, 0, 0.78, 1])
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_metric_over_steps(
    rows: list[dict[str, str]],
    metric_name: str,
    ylabel: str,
    output_path: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(7, 4))
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
    place_legend_outside(axis)
    figure.tight_layout(rect=[0, 0, 0.78, 1])
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_loss_over_steps_by_experiment(rows: list[dict[str, str]], output_dir: Path) -> list[Path]:
    output_paths = []
    grouped = group_loss_rows_by_figure(rows)
    for figure_label in sorted(grouped):
        figure_rows = grouped[figure_label]
        output_paths.append(
            plot_loss_over_steps_for_experiment(
                figure_rows,
                figure_label,
                output_dir / f"loss_over_steps_{safe_filename_fragment(figure_label)}.png",
            )
        )
    return output_paths


def plot_loss_over_steps_for_experiment(
    rows: list[dict[str, str]],
    figure_label: str,
    output_path: Path,
) -> Path:
    granularity_labels = sorted(
        {str(row["granularity"]) for row in rows if row.get("granularity")},
        key=granularity_sort_key,
    )
    if not granularity_labels:
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.text(
            0.5,
            0.5,
            "No granularity metadata found",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        figure.suptitle(figure_label)
        figure.tight_layout(rect=[0, 0, 1, 0.96])
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)
        return output_path

    column_count = 1
    row_count = len(granularity_labels)

    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(8, 3.6 * row_count),
        squeeze=False,
    )

    for axis, granularity in zip(axes.flat, granularity_labels):
        granularity_rows = [row for row in rows if row.get("granularity") == granularity]
        series = group_loss_subplot_rows(granularity_rows)
        has_points = False

        for label, group_rows_for_label in series.items():
            points = [
                (to_float(row["step"]), to_float(row["loss"]))
                for row in group_rows_for_label
                if row.get("step") not in (None, "") and row.get("loss") not in (None, "")
            ]
            if not points:
                continue
            has_points = True
            points.sort(key=lambda point: point[0])
            xs, ys = zip(*points)
            smoothed_ys = moving_average(
                list(ys),
                window_size=loss_moving_average_window_size(len(ys)),
            )
            axis.plot(
                xs,
                smoothed_ys,
                marker="o",
                markersize=3,
                linewidth=1.0,
                label=label,
            )

        axis.set_title(f"Granularity {granularity}")
        axis.set_xlabel("Step")
        axis.set_ylabel("Loss")
        axis.minorticks_on()
        axis.grid(True, which="major", alpha=0.28, linewidth=0.6)
        axis.grid(True, which="minor", alpha=0.14, linewidth=0.35)
        if has_points:
            axis.legend(frameon=False, fontsize="small")
        else:
            axis.text(
                0.5,
                0.5,
                "No numeric loss points found",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

    for axis in axes.flat[len(granularity_labels):]:
        axis.set_visible(False)

    figure.suptitle(figure_label)
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_consistency_results(rows: list[dict[str, str]], output_path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(10, 5))
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
        figure.tight_layout()
        figure.savefig(output_path, bbox_inches="tight")
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
    place_legend_outside(axis)
    figure.tight_layout(rect=[0, 0, 0.78, 1])
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


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
    if row.get("model_family") == "standalone":
        return standalone_figure_label(row)
    return experiment_label(row)


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


def group_loss_subplot_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    run_ids = {str(row["run_id"]) for row in rows if row.get("run_id") not in (None, "")}
    include_run_id = len(run_ids) > 1
    grouped: dict[str, list[dict[str, str]]] = {}

    for row in rows:
        label_parts = [str(row.get("split") or "unknown-split")]
        if include_run_id:
            label_parts.append(str(row.get("run_id") or "unknown-run"))
        label = " / ".join(label_parts)
        grouped.setdefault(label, []).append(row)

    return grouped


def safe_filename_fragment(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return normalized or "unknown"


def scaling_curve_label(row: dict[str, str]) -> str:
    sampling_mode = row.get("sampling_mode")
    model_variant = row.get("model_variant")
    if sampling_mode and model_variant:
        return f"{sampling_mode} / {model_variant}"
    if sampling_mode:
        return sampling_mode
    model_family = row.get("model_family")
    if model_family and model_variant:
        return f"{model_family} / {model_variant}"
    return model_family or "unknown"


def place_legend_outside(axis) -> None:
    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=False,
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

    window_size = max(3, math.ceil(point_count * LOSS_MOVING_AVERAGE_FRACTION))
    if window_size % 2 == 0:
        window_size += 1
    if window_size > point_count:
        window_size = point_count if point_count % 2 == 1 else point_count - 1
    return max(1, window_size)


def to_float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
