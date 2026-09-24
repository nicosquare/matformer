#!/usr/bin/env python3
"""Plot S1/S2 64- versus 256-update warmup comparisons from completed runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_tinystories_s1_peak_lr import BASE, REFERENCES, UPDATES, WIDTHS, read_curves, required, save, sha256


S1_WARMUP = BASE / "optimizer-ownership-s1-warmup-v1"
S2_WARMUP = BASE / "optimizer-ownership-s2-warmup-peak-lr-v1"
STYLES = {
    ("S1", 64): ("#0072B2", "-"),
    ("S1", 256): ("#0072B2", "--"),
    ("S2", 64): ("#D55E00", "-"),
    ("S2", 256): ("#D55E00", "--"),
}


def run_location(grid: str, owner: str, warmup: int) -> tuple[Path, str]:
    if warmup == 64:
        return REFERENCES[grid], owner
    if owner == "S1":
        return S1_WARMUP, f"S1-{grid}-w256"
    return S2_WARMUP, f"S2-{grid}-w256"


def saved_endpoints(root: Path, arm: str, expected_updates: int) -> dict[str, dict]:
    path = root / "runs" / arm / "terminal_validation_results.json"
    sidecar = json.loads(path.read_text())
    required(sidecar["actual_updates"] == expected_updates, f"{arm}: incomplete terminal")
    required(sidecar["evaluation_role"] == "ordinary_validation", f"{arm}: wrong evaluation role")
    checkpoint = Path(sidecar["checkpoint_path"])
    required(sha256(checkpoint) == sidecar["checkpoint_sha256"], f"{arm}: terminal checkpoint changed")
    rows = sidecar["endpoints"]
    required(len(rows) == 4 if expected_updates == UPDATES else len(rows) == 1,
             f"{arm}: unexpected endpoint count")
    return {row["granularity"]: row for row in rows}


def new_s2_endpoints(grid: str, widths: tuple[str, ...]) -> dict[str, dict]:
    arm = f"S2-{grid}-w256"
    run = S2_WARMUP / "runs" / arm
    summary = json.loads((run / "run_summary.json").read_text())
    config = json.loads((run / "config.json").read_text())
    training = config["training"]
    required(summary["status"] == "completed" and summary["committed_optimizer_steps"] == UPDATES,
             f"{arm}: incomplete terminal")
    required(summary["validation_loss_aggregation"] == "target_token_weighted_causal_shift_float64",
             f"{arm}: unexpected validation aggregation")
    required(training["resolved_learning_rate"] == 0.008
             and training["resolved_warmup_steps"] == 256
             and training["optimizer_state_scope"] == "per_granularity"
             and training["resolved_mixed_precision"] == "bf16",
             f"{arm}: unexpected scientific controls")
    required(sha256(Path(summary["terminal_checkpoint_path"])) == summary["terminal_checkpoint_sha256"],
             f"{arm}: terminal checkpoint changed")
    with (run / "scaling_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required(len(rows) == 4 and {row["granularity"] for row in rows} == set(widths),
             f"{arm}: expected four unique width endpoints")
    endpoints = {}
    for row in rows:
        loss, perplexity = float(row["final_validation_loss"]), float(row["final_validation_perplexity"])
        required(np.isfinite(loss) and np.isclose(np.exp(loss), perplexity, rtol=1e-10),
                 f"{arm}: invalid loss/perplexity")
        required(int(row["evaluation_target_tokens"]) > 0, f"{arm}: missing validation targets")
        endpoints[row["granularity"]] = dict(granularity=row["granularity"], loss=loss,
            perplexity=perplexity, non_embedding_parameters=int(row["non_embedding_parameters"]),
            evaluation_target_tokens=int(row["evaluation_target_tokens"]),
            validation_manifest_hash=row["validation_manifest_hash"])
    return endpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=S2_WARMUP / "reports" / "warmup")
    args = parser.parse_args()
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output already occupied: {output}")

    all_rows = []
    all_deltas = []
    plotted_sources = {}
    for grid in ("linear", "geometric"):
        widths = WIDTHS[grid]
        data = {}
        curves = {}
        early = {}
        standalones = {w: saved_endpoints(REFERENCES[grid], f"ST-{w}", 87132)[w] for w in widths}
        for owner, warmup in (("S1", 64), ("S1", 256), ("S2", 64), ("S2", 256)):
            root, arm = run_location(grid, owner, warmup)
            key = owner, warmup
            data[key] = new_s2_endpoints(grid, widths) if key == ("S2", 256) else saved_endpoints(root, arm, UPDATES)
            run = root / "runs" / arm
            curves[key], early[key] = read_curves(run / "metrics.csv", widths)
            for width in widths:
                required(int(curves[key][width]["step"].iloc[-1]) == UPDATES, f"{arm}/{width}: missing terminal curve")
                required(np.isclose(float(curves[key][width]["loss"].iloc[-1]),
                                    float(data[key][width]["loss"]), atol=1e-12, rtol=0),
                         f"{arm}/{width}: curve differs from terminal")
            for name in ("metrics.csv", "config.json"):
                p = run / name
                plotted_sources[str(p)] = sha256(p)
            terminal = run / "terminal_validation_results.json"
            if terminal.exists():
                plotted_sources[str(terminal)] = sha256(terminal)
            else:
                for name in ("run_summary.json", "scaling_results.csv"):
                    p = run / name
                    plotted_sources[str(p)] = sha256(p)
        for width in widths:
            st = standalones[width]
            plotted_sources[str(REFERENCES[grid] / "runs" / f"ST-{width}" / "terminal_validation_results.json")] = sha256(
                REFERENCES[grid] / "runs" / f"ST-{width}" / "terminal_validation_results.json")
            counts = {int(st["non_embedding_parameters"])}
            manifests = {st["validation_manifest_hash"]}
            for key in data:
                e = data[key][width]
                counts.add(int(e["non_embedding_parameters"]))
                manifests.add(e["validation_manifest_hash"])
            required(len(counts) == 1 and len(manifests) == 1,
                     f"{grid}/{width}: parameter count or validation manifest differs")
            params = counts.pop()
            for (owner, warmup), points in data.items():
                e = points[width]
                all_rows.append(dict(grid=grid, owner=owner, warmup_steps=warmup, peak_lr=.008,
                    width=width, non_embedding_parameters=params, assigned_updates=UPDATES,
                    loss=float(e["loss"]), perplexity=float(e["perplexity"]),
                    evaluation_target_tokens=int(e["evaluation_target_tokens"])))
            all_rows.append(dict(grid=grid, owner="standalone", warmup_steps=64, peak_lr=.008,
                width=width, non_embedding_parameters=params, assigned_updates=87132,
                loss=float(st["loss"]), perplexity=float(st["perplexity"]),
                evaluation_target_tokens=int(st["evaluation_target_tokens"])))
            s1_delta = float(data["S1", 256][width]["loss"]) - float(data["S1", 64][width]["loss"])
            s2_delta = float(data["S2", 256][width]["loss"]) - float(data["S2", 64][width]["loss"])
            all_deltas.append(dict(grid=grid, width=width, non_embedding_parameters=params,
                s1_warmup_delta=s1_delta, s2_warmup_delta=s2_delta, s2_minus_s1_delta=s2_delta-s1_delta))

        x = np.array([int(standalones[w]["non_embedding_parameters"]) for w in widths])
        for metric, ylabel in (("loss", "Ordinary-validation loss"),
                              ("perplexity", "Ordinary-validation perplexity")):
            fig, ax = plt.subplots(figsize=(9.5, 5.7))
            for key, points in data.items():
                color, linestyle = STYLES[key]
                ax.plot(x, [points[w][metric] for w in widths], marker="o", color=color,
                        linestyle=linestyle, linewidth=2, label=f"{key[0]} · {key[1]} warmup")
            ax.scatter(x, [standalones[w][metric] for w in widths], color="#009E73",
                       marker="^", s=80, zorder=4, label="Standalone · 1 epoch")
            ax.set(title=f"{grid.title()} · TinyStories-Instruct · seed 42 · peak LR 0.008",
                   xlabel="Active non-embedding parameters", ylabel=ylabel)
            ax.ticklabel_format(axis="x", style="plain")
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            output.mkdir(parents=True, exist_ok=True)
            save(fig, output, f"{grid}_{metric}_vs_parameters")

        selected = [row for row in all_deltas if row["grid"] == grid]
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.axhline(0, color="#666666", linewidth=1)
        for owner, color, field in (("S1", "#0072B2", "s1_warmup_delta"),
                                    ("S2", "#D55E00", "s2_warmup_delta")):
            ax.plot(x, [row[field] for row in selected], marker="o", color=color,
                    linewidth=2, label=owner)
        ax.set(title=f"{grid.title()} · terminal loss change from 64 to 256 warmup · peak LR 0.008",
               xlabel="Active non-embedding parameters", ylabel="Loss(256) − Loss(64)")
        ax.ticklabel_format(axis="x", style="plain")
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        save(fig, output, f"{grid}_warmup_delta_loss")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for ax, width in zip(axes.flat, widths):
            for key, points in curves.items():
                color, linestyle = STYLES[key]
                frame = points[width]
                ax.plot(frame["step"] / 1000, frame["loss"], color=color,
                        linestyle=linestyle, linewidth=.85, alpha=.85,
                        label=f"{key[0]} · {key[1]} warmup")
            ax.axhline(standalones[width]["loss"], color="#009E73", linestyle=":",
                       linewidth=1.3, label="Standalone terminal")
            ax.set(title=width, xlabel="Optimizer updates (thousands)",
                   ylabel="Ordinary-validation loss", xlim=(0, UPDATES / 1000))
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"{grid.title()} · raw validation curves · S1/S2 4 epochs, standalones 1 epoch · seed 42")
        fig.tight_layout()
        save(fig, output, f"{grid}_validation_loss_vs_updates")

        fig, axes = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True)
        for warmup, color in ((64, "#444444"), (256, "#AA3377")):
            frame = early["S2", warmup]
            axes[0].plot(frame["step"], frame["learning_rate"], color=color,
                         label=f"{warmup} warmup (same LR for S1/S2)")
        for key, frame in early.items():
            color, linestyle = STYLES[key]
            axes[1].plot(frame["step"], frame["loss"], color=color, linestyle=linestyle,
                         linewidth=.8, alpha=.8, label=f"{key[0]} · {key[1]} warmup")
        for boundary in (65, 257):
            axes[0].axvline(boundary, color="#999999", linestyle=":", linewidth=1)
        axes[0].set_ylabel("Recorded applied LR")
        axes[1].set(xlabel="Optimizer update", ylabel="Recorded training loss", xlim=(1, 1024))
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle(f"{grid.title()} · first 1,024 updates · peak LR 0.008 · seed 42")
        fig.tight_layout()
        save(fig, output, f"{grid}_early_lr_and_train_loss")
        print(f"Completed {grid} warmup plots", flush=True)

    with (output / "endpoints.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (output / "warmup_deltas.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_deltas[0]))
        writer.writeheader()
        writer.writerows(all_deltas)
    (output / "endpoints.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    (output / "warmup_deltas.json").write_text(json.dumps(all_deltas, indent=2) + "\n")
    (output / "plot_sources.json").write_text(json.dumps(plotted_sources, indent=2) + "\n")
    print(f"Saved 10 figures in PNG/PDF plus tables and source hashes to {output}")


if __name__ == "__main__":
    main()
