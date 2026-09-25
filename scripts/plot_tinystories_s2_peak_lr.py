#!/usr/bin/env python3
"""Plot completed S1/S2 peak-LR sweeps with 64-step warmup."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_tinystories_s1_peak_lr import BASE, COLORS, REFERENCES, UPDATES, WIDTHS, read_curves, required, save, sha256
from plot_tinystories_s2_warmup import saved_endpoints


S1_NEW = BASE / "optimizer-ownership-s1-peak-lr-v1"
S2_NEW = BASE / "optimizer-ownership-s2-warmup-peak-lr-v1"
LRS = (0.004, 0.008, 0.012)
OWNERS = ("S1", "S2")


def location(grid: str, owner: str, lr: float) -> tuple[Path, str]:
    if lr == 0.008:
        return REFERENCES[grid], owner
    return (S1_NEW if owner == "S1" else S2_NEW), f"{owner}-{grid}-lr{int(lr * 1000):04d}"


def candidate_endpoints(root: Path, arm: str, widths: tuple[str, ...], lr: float) -> dict[str, dict]:
    run = root / "runs" / arm
    summary = json.loads((run / "run_summary.json").read_text())
    config = json.loads((run / "config.json").read_text())
    training = config["training"]
    required(summary["status"] == "completed" and summary["committed_optimizer_steps"] == UPDATES,
             f"{arm}: incomplete terminal")
    required(summary["validation_loss_aggregation"] == "target_token_weighted_causal_shift_float64",
             f"{arm}: unexpected validation aggregation")
    required(training["resolved_learning_rate"] == lr and training["resolved_warmup_steps"] == 64
             and training["resolved_mixed_precision"] == "bf16"
             and training["optimizer_state_scope"] == ("shared" if arm.startswith("S1-") else "per_granularity"),
             f"{arm}: unexpected scientific controls")
    required(sha256(Path(summary["terminal_checkpoint_path"])) == summary["terminal_checkpoint_sha256"],
             f"{arm}: terminal checkpoint changed")
    with (run / "scaling_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required(len(rows) == len(widths) and {row["granularity"] for row in rows} == set(widths),
             f"{arm}: missing or duplicate width endpoints")
    endpoints = {}
    for row in rows:
        loss = float(row["final_validation_loss"])
        perplexity = float(row["final_validation_perplexity"])
        required(np.isfinite(loss) and np.isclose(np.exp(loss), perplexity, rtol=1e-10)
                 and int(row["evaluation_target_tokens"]) > 0, f"{arm}: invalid endpoint")
        endpoints[row["granularity"]] = dict(granularity=row["granularity"], loss=loss,
            perplexity=perplexity, non_embedding_parameters=int(row["non_embedding_parameters"]),
            validation_manifest_hash=row["validation_manifest_hash"],
            evaluation_target_tokens=int(row["evaluation_target_tokens"]))
    return endpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=S2_NEW / "reports" / "peak_lr")
    args = parser.parse_args()
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output already occupied: {output}")

    endpoints_table = []
    deltas_table = []
    source_hashes = {}
    for grid in ("linear", "geometric"):
        widths = WIDTHS[grid]
        endpoints = {}
        curves = {}
        early = {}
        standalones = {w: saved_endpoints(REFERENCES[grid], f"ST-{w}", 87132)[w] for w in widths}
        for owner in OWNERS:
            for lr in LRS:
                root, arm = location(grid, owner, lr)
                run = root / "runs" / arm
                key = owner, lr
                endpoints[key] = (saved_endpoints(root, arm, UPDATES) if lr == 0.008
                                  else candidate_endpoints(root, arm, widths, lr))
                curves[key], early[key] = read_curves(run / "metrics.csv", widths)
                for width in widths:
                    required(np.isclose(float(curves[key][width]["loss"].iloc[-1]),
                                        float(endpoints[key][width]["loss"]), atol=1e-12, rtol=0),
                             f"{arm}/{width}: terminal curve differs from endpoint")
                for name in ("metrics.csv", "config.json"):
                    p = run / name
                    source_hashes[str(p)] = sha256(p)
                for name in (("terminal_validation_results.json",) if lr == 0.008
                             else ("run_summary.json", "scaling_results.csv")):
                    p = run / name
                    source_hashes[str(p)] = sha256(p)

        for width in widths:
            p = REFERENCES[grid] / "runs" / f"ST-{width}" / "terminal_validation_results.json"
            source_hashes[str(p)] = sha256(p)
            counts = {int(standalones[width]["non_embedding_parameters"])}
            manifests = {standalones[width]["validation_manifest_hash"]}
            for series in endpoints.values():
                counts.add(int(series[width]["non_embedding_parameters"]))
                manifests.add(series[width]["validation_manifest_hash"])
            required(len(counts) == 1 and len(manifests) == 1,
                     f"{grid}/{width}: parameter count or validation manifest differs")
            params = counts.pop()
            for owner in OWNERS:
                for lr in LRS:
                    e = endpoints[owner, lr][width]
                    baseline = endpoints[owner, 0.008][width]
                    endpoints_table.append(dict(grid=grid, owner=owner, width=width, peak_lr=lr,
                        warmup_steps=64, assigned_updates=UPDATES, non_embedding_parameters=params,
                        loss=float(e["loss"]), perplexity=float(e["perplexity"]),
                        delta_loss_vs_lr0008=float(e["loss"])-float(baseline["loss"]),
                        gap_to_standalone=float(e["loss"])-float(standalones[width]["loss"])))
            e = standalones[width]
            endpoints_table.append(dict(grid=grid, owner="standalone", width=width, peak_lr=.008,
                warmup_steps=64, assigned_updates=87132, non_embedding_parameters=params,
                loss=float(e["loss"]), perplexity=float(e["perplexity"]),
                delta_loss_vs_lr0008="", gap_to_standalone=0.0))
            for lr in LRS:
                s1 = float(endpoints["S1", lr][width]["loss"])
                s2 = float(endpoints["S2", lr][width]["loss"])
                deltas_table.append(dict(grid=grid, width=width, non_embedding_parameters=params,
                    peak_lr=lr, s1_loss=s1, s2_loss=s2, s2_minus_s1=s2-s1))

        x = np.array([int(standalones[w]["non_embedding_parameters"]) for w in widths])
        for metric, ylabel in (("loss", "Ordinary-validation loss"),
                               ("perplexity", "Ordinary-validation perplexity")):
            fig, ax = plt.subplots(figsize=(10, 5.8))
            for owner in OWNERS:
                for lr in LRS:
                    ax.plot(x, [endpoints[owner, lr][w][metric] for w in widths],
                            color=COLORS[lr], linestyle="-" if owner == "S1" else "--",
                            marker="o" if owner == "S1" else "s", linewidth=1.8,
                            label=f"{owner} · peak LR {lr:g}")
            ax.scatter(x, [standalones[w][metric] for w in widths], color="#009E73",
                       marker="^", s=80, zorder=4, label="Standalone · 1 epoch")
            ax.set(title=f"{grid.title()} · TinyStories-Instruct · 64 warmup · seed 42",
                   xlabel="Active non-embedding parameters", ylabel=ylabel)
            ax.ticklabel_format(axis="x", style="plain")
            ax.grid(alpha=.25)
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            output.mkdir(parents=True, exist_ok=True)
            save(fig, output, f"{grid}_{metric}_vs_parameters")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
        for ax, owner in zip(axes, OWNERS):
            ax.axhline(0, color="#666666", linewidth=1)
            for lr in (0.004, 0.012):
                ax.plot(x, [float(endpoints[owner, lr][w]["loss"])
                            - float(endpoints[owner, .008][w]["loss"]) for w in widths],
                        color=COLORS[lr], marker="o", linewidth=2, label=f"LR {lr:g} − 0.008")
            ax.set(title=owner, xlabel="Active non-embedding parameters")
            ax.ticklabel_format(axis="x", style="plain")
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Terminal validation loss difference")
        fig.suptitle(f"{grid.title()} · peak LR sensitivity · 64 warmup · seed 42")
        fig.tight_layout()
        save(fig, output, f"{grid}_peak_lr_delta_loss")

        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.axhline(0, color="#666666", linewidth=1)
        for lr in LRS:
            ax.plot(x, [float(endpoints["S2", lr][w]["loss"])
                        - float(endpoints["S1", lr][w]["loss"]) for w in widths],
                    color=COLORS[lr], marker="o", linewidth=2, label=f"Peak LR {lr:g}")
        ax.set(title=f"{grid.title()} · S2 − S1 terminal validation loss · 64 warmup · seed 42",
               xlabel="Active non-embedding parameters", ylabel="Loss(S2) − Loss(S1)")
        ax.ticklabel_format(axis="x", style="plain")
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        save(fig, output, f"{grid}_s2_minus_s1_loss")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for ax, width in zip(axes.flat, widths):
            for owner in OWNERS:
                for lr in LRS:
                    frame = curves[owner, lr][width]
                    ax.plot(frame["step"] / 1000, frame["loss"], color=COLORS[lr],
                            linestyle="-" if owner == "S1" else "--", linewidth=.7,
                            alpha=.8, label=f"{owner} · LR {lr:g}")
            ax.axhline(standalones[width]["loss"], color="#009E73", linestyle=":",
                       linewidth=1.2, label="Standalone terminal")
            ax.set(title=width, xlabel="Optimizer updates (thousands)",
                   ylabel="Ordinary-validation loss", xlim=(0, UPDATES / 1000))
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"{grid.title()} · raw validation curves · S1/S2 4 epochs · seed 42")
        fig.tight_layout()
        save(fig, output, f"{grid}_validation_loss_vs_updates")

        fig, axes = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True)
        for lr in LRS:
            frame = early["S2", lr]
            axes[0].plot(frame["step"], frame["learning_rate"], color=COLORS[lr],
                         label=f"Peak LR {lr:g}")
        for owner in OWNERS:
            for lr in LRS:
                frame = early[owner, lr]
                axes[1].plot(frame["step"], frame["loss"], color=COLORS[lr],
                             linestyle="-" if owner == "S1" else "--", linewidth=.75,
                             alpha=.7, label=f"{owner} · LR {lr:g}")
        axes[0].axvline(65, color="#999999", linestyle=":", linewidth=1)
        axes[0].set_ylabel("Recorded applied LR")
        axes[1].set(xlabel="Optimizer update", ylabel="Recorded training loss", xlim=(1, 1024))
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, ncol=2)
        fig.suptitle(f"{grid.title()} · first 1,024 updates · 64 warmup · seed 42")
        fig.tight_layout()
        save(fig, output, f"{grid}_early_lr_and_train_loss")
        print(f"Completed {grid} peak LR plots", flush=True)

    for name, rows in (("endpoints", endpoints_table), ("s2_minus_s1", deltas_table)):
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output / "plot_sources.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    print(f"Saved 12 figures in PNG/PDF plus tables and source hashes to {output}")


if __name__ == "__main__":
    main()
