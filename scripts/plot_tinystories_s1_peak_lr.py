#!/usr/bin/env python3
"""Plot completed S1 peak-LR runs against the original 64-step S1 and standalones."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("/nfs-stor/ivo.navarrete/results/elasticnn")
NEW = BASE / "optimizer-ownership-s1-peak-lr-v1"
REFERENCES = {
    "linear": BASE / "optimizer-ownership-v1",
    "geometric": BASE / "optimizer-ownership-matformer-widths-v1",
}
WIDTHS = {
    "linear": ("g250", "g500", "g750", "g1000"),
    "geometric": ("g125", "g250", "g500", "g1000"),
}
COLORS = {0.004: "#0072B2", 0.008: "#444444", 0.012: "#D55E00"}
UPDATES = 348528


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def reference_rows(root: Path, arm: str) -> dict[str, dict]:
    terminal = json.loads((root / "runs" / arm / "terminal_validation_results.json").read_text())
    required(terminal["actual_updates"] == UPDATES if arm == "S1" else terminal["actual_updates"] == 87132,
             f"{root}/{arm}: incomplete reference")
    required(terminal["evaluation_role"] == "ordinary_validation", f"{root}/{arm}: wrong evaluation role")
    return {row["granularity"]: row for row in terminal["endpoints"]}


def candidate_rows(root: Path, arm: str, widths: tuple[str, ...], lr: float) -> dict[str, dict]:
    run = root / "runs" / arm
    summary = json.loads((run / "run_summary.json").read_text())
    config = json.loads((run / "config.json").read_text())
    training = config["training"]
    required(summary["status"] == "completed" and summary["committed_optimizer_steps"] == UPDATES,
             f"{arm}: incomplete terminal")
    required(summary["validation_loss_aggregation"] == "target_token_weighted_causal_shift_float64",
             f"{arm}: unexpected validation aggregation")
    required(training["resolved_learning_rate"] == lr and training["resolved_warmup_steps"] == 64,
             f"{arm}: unexpected LR schedule")
    required(training["resolved_mixed_precision"] == "bf16", f"{arm}: unexpected precision")
    required(sha256(Path(summary["terminal_checkpoint_path"])) == summary["terminal_checkpoint_sha256"],
             f"{arm}: terminal checkpoint changed")
    with (run / "scaling_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required(len(rows) == 4 and {row["granularity"] for row in rows} == set(widths),
             f"{arm}: expected four distinct widths")
    result = {}
    for row in rows:
        required(int(row["evaluation_target_tokens"]) > 0, f"{arm}: missing validation targets")
        loss = float(row["final_validation_loss"])
        perplexity = float(row["final_validation_perplexity"])
        required(np.isfinite(loss) and np.isclose(np.exp(loss), perplexity, rtol=1e-10),
                 f"{arm}: invalid loss/perplexity")
        result[row["granularity"]] = {
            "granularity": row["granularity"],
            "non_embedding_parameters": int(row["non_embedding_parameters"]),
            "loss": loss,
            "perplexity": perplexity,
            "evaluation_target_tokens": int(row["evaluation_target_tokens"]),
        }
    return result


def read_curves(path: Path, widths: tuple[str, ...]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    frames = {width: [] for width in widths}
    early = []
    for chunk in pd.read_csv(path, usecols=["step", "split", "granularity", "loss", "learning_rate"],
                             chunksize=50000, low_memory=False):
        validation = chunk.loc[chunk["split"] == "validation"]
        for width in widths:
            selected = validation.loc[validation["granularity"] == width, ["step", "loss"]]
            if len(selected):
                frames[width].append(selected)
        first = chunk.loc[(chunk["split"] == "train") & (chunk["step"] <= 1024),
                          ["step", "loss", "learning_rate"]]
        if len(first):
            early.append(first)
    curves = {}
    for width, parts in frames.items():
        required(bool(parts), f"{path}: missing {width} validation curve")
        frame = pd.concat(parts, ignore_index=True).sort_values("step")
        required(frame["step"].is_unique and int(frame["step"].iloc[-1]) == UPDATES,
                 f"{path}: invalid {width} validation steps")
        required(np.isfinite(frame["loss"]).all(), f"{path}: nonfinite {width} loss")
        curves[width] = frame
    early_frame = pd.concat(early, ignore_index=True).sort_values("step")
    required(early_frame["step"].is_unique and int(early_frame["step"].iloc[0]) == 1
             and int(early_frame["step"].iloc[-1]) == 1024,
             f"{path}: incomplete early curve")
    return curves, early_frame


def save(fig, output: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=tuple(WIDTHS), default="linear")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    grid = args.grid
    widths = WIDTHS[grid]
    output = args.output_dir or NEW / "reports" / grid
    reference = REFERENCES[grid]
    baseline = reference_rows(reference, "S1")
    standalones = {width: reference_rows(reference, f"ST-{width}")[width] for width in widths}
    candidates = {lr: candidate_rows(NEW, f"S1-{grid}-lr{int(lr * 1000):04d}", widths, lr)
                  for lr in (0.004, 0.012)}
    series = {0.004: candidates[0.004], 0.008: baseline, 0.012: candidates[0.012]}
    for width in widths:
        counts = {int(series[lr][width]["non_embedding_parameters"]) for lr in series}
        counts.add(int(standalones[width]["non_embedding_parameters"]))
        required(len(counts) == 1, f"{width}: active parameter counts differ")
        manifests = {series[lr][width].get("validation_manifest_hash") for lr in series if lr == 0.008}
        required(len(manifests) == 1, f"{width}: reference validation manifest missing")

    curves = {}
    early = {}
    for lr, arm, root in ((0.004, f"S1-{grid}-lr0004", NEW),
                          (0.008, "S1", reference),
                          (0.012, f"S1-{grid}-lr0012", NEW)):
        curves[lr], early[lr] = read_curves(root / "runs" / arm / "metrics.csv", widths)
        for width in widths:
            terminal = float(curves[lr][width]["loss"].iloc[-1])
            required(np.isclose(terminal, float(series[lr][width]["loss"]), atol=1e-12, rtol=0),
                     f"{arm}/{width}: terminal curve differs from endpoint")

    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for width in widths:
        params = int(baseline[width]["non_embedding_parameters"])
        for lr in (0.004, 0.008, 0.012):
            e = series[lr][width]
            rows.append({"grid": grid, "role": "S1", "width": width,
                         "peak_lr": lr, "warmup_steps": 64, "assigned_updates": UPDATES,
                         "non_embedding_parameters": params, "loss": float(e["loss"]),
                         "perplexity": float(e["perplexity"]),
                         "delta_loss_vs_lr0008": float(e["loss"]) - float(baseline[width]["loss"]),
                         "gap_to_standalone": float(e["loss"]) - float(standalones[width]["loss"])})
        e = standalones[width]
        rows.append({"grid": grid, "role": "standalone", "width": width,
                     "peak_lr": 0.008, "warmup_steps": 64, "assigned_updates": 87132,
                     "non_embedding_parameters": params, "loss": float(e["loss"]),
                     "perplexity": float(e["perplexity"]),
                     "delta_loss_vs_lr0008": "", "gap_to_standalone": 0.0})
    with (output / "endpoints.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "endpoints.json").write_text(json.dumps(rows, indent=2) + "\n")

    x = np.array([int(baseline[w]["non_embedding_parameters"]) for w in widths])
    for metric, ylabel in (("loss", "Ordinary-validation loss"),
                          ("perplexity", "Ordinary-validation perplexity")):
        fig, ax = plt.subplots(figsize=(9, 5.4))
        for lr in (0.004, 0.008, 0.012):
            ax.plot(x, [series[lr][w][metric] for w in widths], marker="o", linewidth=2,
                    color=COLORS[lr], label=f"S1 · peak LR {lr:g} · 64 warmup")
        ax.scatter(x, [standalones[w][metric] for w in widths], marker="^", s=85,
                   color="#009E73", label="Standalone · peak LR 0.008 · 64 warmup · 1 epoch", zorder=4)
        for at, width in zip(x, widths):
            ax.annotate(width, (at, min(series[0.004][width][metric],
                                        standalones[width][metric])), xytext=(0, -15),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.set(title=f"TinyStories-Instruct · {grid.title()} grid · seed 42 · terminal ordinary validation",
               xlabel="Active non-embedding parameters", ylabel=ylabel)
        ax.ticklabel_format(axis="x", style="plain")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        save(fig, output, f"{grid}_{metric}_vs_parameters")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, width in zip(axes.flat, widths):
        for lr in (0.004, 0.008, 0.012):
            frame = curves[lr][width]
            ax.plot(frame["step"] / 1000, frame["loss"], color=COLORS[lr],
                    linewidth=0.8, alpha=0.8, label=f"S1 LR {lr:g}")
        ax.axhline(standalones[width]["loss"], color="#009E73", linestyle=":",
                   linewidth=1.3, label="Standalone terminal")
        ax.set(title=f"{width} · FFN {int(round(float(width[1:]) / 1000 * 256))}",
               xlabel="Optimizer updates (thousands)", ylabel="Ordinary-validation loss",
               xlim=(0, UPDATES / 1000))
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"{grid.title()} · raw validation curves · S1 4 epochs, standalones 1 epoch · seed 42")
    fig.tight_layout()
    save(fig, output, f"{grid}_validation_loss_vs_updates")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for lr in (0.004, 0.008, 0.012):
        frame = early[lr]
        axes[0].plot(frame["step"], frame["learning_rate"], color=COLORS[lr], label=f"Peak LR {lr:g}")
        axes[1].plot(frame["step"], frame["loss"], color=COLORS[lr], alpha=0.7, linewidth=0.9,
                     label=f"Peak LR {lr:g}")
    axes[0].axvline(65, color="#999999", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Recorded applied LR")
    axes[1].set(xlabel="Optimizer update", ylabel="Recorded training loss", xlim=(1, 1024))
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"{grid.title()} · first 1,024 updates · same 64-step warmup · seed 42")
    fig.tight_layout()
    save(fig, output, f"{grid}_early_lr_and_train_loss")

    sources = [reference / "runs" / "S1" / "terminal_validation_results.json"]
    sources += [reference / "runs" / f"ST-{width}" / "terminal_validation_results.json" for width in widths]
    for arm in (f"S1-{grid}-lr0004", f"S1-{grid}-lr0012"):
        sources += [NEW / "runs" / arm / name for name in ("run_summary.json", "scaling_results.csv", "config.json")]
    for arm, root in ((f"S1-{grid}-lr0004", NEW), ("S1", reference),
                      (f"S1-{grid}-lr0012", NEW)):
        sources.append(root / "runs" / arm / "metrics.csv")
    (output / "plot_sources.json").write_text(json.dumps({str(p): sha256(p) for p in sources}, indent=2) + "\n")
    print(f"Saved 4 figures in PNG/PDF, endpoints and source hashes to {output}")


if __name__ == "__main__":
    main()
