#!/usr/bin/env python3
"""Compare the completed coverage-balanced S1 run with original S1 and standalones."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_tinystories_s1_peak_lr import read_curves, required, save, sha256


BASE = Path("/nfs-stor/ivo.navarrete/results/elasticnn")
GRIDS = {
    "linear": ("optimizer-ownership-coverage-balanced-v1", "optimizer-ownership-v1",
               "tinystories-coverage-balanced-v1-S1-s42", ("g250", "g500", "g750", "g1000")),
    "geometric": ("optimizer-ownership-coverage-balanced-geometric-v1",
                  "optimizer-ownership-matformer-widths-v1",
                  "tinystories-coverage-balanced-geometric-v1-S1-s42",
                  ("g125", "g250", "g500", "g1000")),
}
UPDATES = 348528
COLORS = {"original": "#666666", "balanced": "#0072B2", "standalone": "#009E73"}


def terminal_reference(reference: Path, arm: str, expected_updates: int) -> dict:
    path = reference / arm / "terminal_validation_results.json"
    data = json.loads(path.read_text())
    required(data["actual_updates"] == expected_updates and data["evaluation_role"] == "ordinary_validation",
             f"{arm}: incomplete or wrong validation role")
    return {row["granularity"]: row for row in data["endpoints"]}


def balanced_terminal(root: Path, run: Path, widths: tuple[str, ...]) -> dict:
    summary = json.loads((run / "run_summary.json").read_text())
    config = json.loads((run / "config.json").read_text())
    audit = json.loads((root / "campaign" / "coverage_audit.json").read_text())
    required(summary["status"] == "completed" and summary["committed_optimizer_steps"] == UPDATES,
             "Balanced run is incomplete")
    required(summary["validation_loss_aggregation"] == "target_token_weighted_causal_shift_float64",
             "Unexpected validation aggregation")
    required(config["training"]["resolved_learning_rate"] == .008
             and config["training"]["resolved_warmup_steps"] == 64
             and config["training"]["optimizer_state_scope"] == "shared",
             "Unexpected optimizer or learning-rate controls")
    required(config["model"]["global_sampling_schedule"] == "balanced_cycle"
             and config["dataset"]["optimizer_iteration"]["epoch_order"] == "coverage_balanced_batches",
             "Unexpected coverage controls")
    required(audit["status"] == "passed" and all(
        audit["selected_updates_per_width"][w] == 87132
        and audit["unique_batch_groups_per_width"][w] == 87132 for w in widths),
        "Coverage audit is incomplete")
    required(sha256(Path(summary["terminal_checkpoint_path"])) == summary["terminal_checkpoint_sha256"],
             "Terminal checkpoint checksum mismatch")
    with (run / "scaling_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required(len(rows) == 4 and {r["granularity"] for r in rows} == set(widths),
             "Expected four balanced endpoints")
    return {r["granularity"]: r for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=tuple(GRIDS), default="linear")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root_name, reference_name, run_name, widths = GRIDS[args.grid]
    root = BASE / root_name
    reference = BASE / reference_name / "runs"
    run = root / "runs" / run_name
    output = args.output_dir or root / "reports" / "comparison"
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is occupied: {output}")

    original = terminal_reference(reference, "S1", UPDATES)
    standalone = {w: terminal_reference(reference, f"ST-{w}", 87132)[w] for w in widths}
    balanced = balanced_terminal(root, run, widths)
    rows = []
    for w in widths:
        endpoints = (original[w], balanced[w], standalone[w])
        required(len({int(e["non_embedding_parameters"]) for e in endpoints}) == 1,
                 f"{w}: parameter counts differ")
        required(len({e["validation_manifest_hash"] for e in endpoints}) == 1,
                 f"{w}: validation sets differ")
        for e in endpoints:
            required(int(e["evaluation_target_tokens"]) > 0 and np.isclose(
                np.exp(float(e["loss"] if "loss" in e else e["final_validation_loss"])),
                float(e["perplexity"] if "perplexity" in e else e["final_validation_perplexity"]),
                rtol=1e-10), f"{w}: invalid endpoint")
        o, b, s = [float(e["loss"] if "loss" in e else e["final_validation_loss"]) for e in endpoints]
        rows.append(dict(width=w, non_embedding_parameters=int(endpoints[0]["non_embedding_parameters"]),
                         original_s1_loss=o, balanced_s1_loss=b, standalone_loss=s,
                         balanced_minus_original=b-o, original_minus_standalone=o-s,
                         balanced_minus_standalone=b-s))

    original_curves, _ = read_curves(reference / "S1" / "metrics.csv", widths)
    balanced_curves, _ = read_curves(run / "metrics.csv", widths)
    for w, e in ((w, original[w]) for w in widths):
        required(np.isclose(float(original_curves[w]["loss"].iloc[-1]), float(e["loss"]), atol=1e-12, rtol=0),
                 f"{w}: original terminal curve mismatch")
    for w in widths:
        required(np.isclose(float(balanced_curves[w]["loss"].iloc[-1]),
                            float(balanced[w]["final_validation_loss"]), atol=1e-12, rtol=0),
                 f"{w}: balanced terminal curve mismatch")

    output.mkdir(parents=True)
    with (output / "endpoints.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "endpoints.json").write_text(json.dumps(rows, indent=2) + "\n")

    x = np.array([r["non_embedding_parameters"] for r in rows])
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for field, label, marker in (("original_s1_loss", "Original S1", "o"),
                                 ("balanced_s1_loss", "Coverage-balanced S1", "s"),
                                 ("standalone_loss", "Standalone (one epoch)", "^")):
        color = COLORS["original" if field.startswith("original") else
                       "balanced" if field.startswith("balanced") else "standalone"]
        ax.plot(x, [r[field] for r in rows], marker=marker, linewidth=2, color=color, label=label)
    ax.set(title=f"TinyStories-Instruct · {args.grid} grid · seed 42 · terminal ordinary validation",
           xlabel="Active non-embedding parameters", ylabel="Validation loss")
    ax.ticklabel_format(axis="x", style="plain")
    ax.set_xticks(x, widths)
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    save(fig, output, "terminal_validation_loss")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    for ax, field, title, color in ((axes[0], "balanced_minus_original", "Balanced − original S1", COLORS["balanced"]),
                                    (axes[1], "balanced_minus_standalone", "Balanced − standalone", COLORS["standalone"])):
        ax.axhline(0, color="#666666", linewidth=1)
        ax.bar(widths, [r[field] for r in rows], color=color, alpha=.8)
        ax.set(title=title, xlabel="Width")
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Terminal validation loss difference (negative is better)")
    fig.suptitle(f"TinyStories-Instruct · {args.grid} coverage-balanced S1 · seed 42")
    fig.tight_layout()
    save(fig, output, "terminal_loss_differences")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, w in zip(axes.flat, widths):
        for name, curves in (("original", original_curves), ("balanced", balanced_curves)):
            frame = curves[w]
            ax.plot(frame["step"] / 1000, frame["loss"], color=COLORS[name],
                    linewidth=.9, label="Original S1" if name == "original" else "Coverage-balanced S1")
        ax.axhline(float(standalone[w]["loss"]), color=COLORS["standalone"],
                   linestyle=":", linewidth=1.5, label="Standalone terminal")
        ax.set(title=w, xlabel="Optimizer updates (thousands)", ylabel="Validation loss",
               xlim=(0, UPDATES / 1000))
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"TinyStories-Instruct · {args.grid} ordinary-validation trajectories · seed 42")
    fig.tight_layout()
    save(fig, output, "validation_loss_vs_updates")

    sources = [root / "campaign" / "coverage_audit.json", run / "run_summary.json",
               run / "config.json", run / "scaling_results.csv", run / "metrics.csv",
               reference / "S1" / "terminal_validation_results.json", reference / "S1" / "metrics.csv"]
    sources += [reference / f"ST-{w}" / "terminal_validation_results.json" for w in widths]
    (output / "plot_sources.json").write_text(json.dumps({str(p): sha256(p) for p in sources}, indent=2) + "\n")
    print(f"Saved 3 figures in PNG/PDF and endpoint tables to {output}")
    for r in rows:
        print(f"{r['width']}: balanced {r['balanced_s1_loss']:.6f}, original {r['original_s1_loss']:.6f}, "
              f"standalone {r['standalone_loss']:.6f}, balanced−original {r['balanced_minus_original']:+.6f}")


if __name__ == "__main__":
    main()
