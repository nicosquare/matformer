"""Generate plots from structured CSV artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    figure_paths = generate_figures(args.input, args.output)
    for path in figure_paths:
        print(path)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs", help="Root containing run CSV artifacts.")
    parser.add_argument("--output", default="outputs/figures", help="Figure output directory.")
    return parser.parse_args(argv)


def generate_figures(input_root: str | Path, output_dir: str | Path) -> list[Path]:
    input_root = Path(input_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = []
    scaling_rows = read_csv_artifacts(input_root, "scaling_results.csv")
    metrics_rows = read_csv_artifacts(input_root, "metrics.csv")
    consistency_rows = read_csv_artifacts(input_root, "consistency_results.csv")

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

    if metrics_rows and not scaling_rows:
        figure_paths.append(
            plot_metric_over_steps(
                metrics_rows,
                metric_name="loss",
                ylabel="Loss",
                output_path=output_dir / "loss_over_steps.png",
            )
        )
        figure_paths.append(
            plot_metric_over_steps(
                metrics_rows,
                metric_name="perplexity",
                ylabel="Perplexity",
                output_path=output_dir / "ppl_over_steps.png",
            )
        )

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


def plot_metric_vs_size(
    rows: list[dict[str, str]],
    metric_name: str,
    ylabel: str,
    output_path: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(7, 4))
    grouped = group_rows(rows, ["model_family", "granularity"])

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
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
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
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def plot_consistency_results(rows: list[dict[str, str]], output_path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(7, 4))
    labels = [
        f"{row['small_granularity']}->{row['large_granularity']}\n{row['metric_name']}"
        for row in rows
    ]
    values = [to_float(row["metric_value"]) for row in rows]
    axis.bar(range(len(values)), values)
    axis.set_xticks(range(len(values)), labels, rotation=30, ha="right")
    axis.set_ylabel("Metric value")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def group_rows(rows: list[dict[str, str]], keys: list[str]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        label = " / ".join(row.get(key, "") for key in keys)
        grouped.setdefault(label, []).append(row)
    return grouped


def to_float(value: Any) -> float:
    return float(value)


if __name__ == "__main__":
    main()
