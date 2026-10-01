#!/usr/bin/env python3
"""Compare completed epoch-wise inverse-membership runs with uniform baselines."""

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
from plot_tinystories_s2_warmup import saved_endpoints


ROOT = BASE / "optimizer-ownership-epoch-linear-im-v1"
OWNERS = ("S1", "S2")
COLORS = {"S1": "#0072B2", "S2": "#D55E00"}
STANDALONE_COLOR = "#009E73"


def new_endpoints(grid: str, owner: str, widths: tuple[str, ...]) -> tuple[dict[str, dict], Path]:
    run = ROOT / "runs" / f"tinystories-epoch-linear-im-v1-{owner}-{grid}-s42"
    summary = json.loads((run / "run_summary.json").read_text())
    config = json.loads((run / "config.json").read_text())
    training, model = config["training"], config["model"]
    required(summary["status"] == "completed" and summary["committed_optimizer_steps"] == UPDATES,
             f"{run.name}: incomplete terminal")
    required(summary["validation_loss_aggregation"] == "target_token_weighted_causal_shift_float64",
             f"{run.name}: unexpected validation aggregation")
    required(training["resolved_learning_rate"] == .008 and training["resolved_warmup_steps"] == 64
             and training["resolved_mixed_precision"] == "bf16"
             and training["optimizer_state_scope"] == ("shared" if owner == "S1" else "per_granularity"),
             f"{run.name}: unexpected training controls")
    required(model["global_sampling_schedule"] == "epoch_categorical"
             and tuple(model["granularities"]) == widths
             and len(model["global_sampling_epoch_distributions"]) == 4,
             f"{run.name}: unexpected width schedule")
    required(sha256(Path(summary["terminal_checkpoint_path"])) == summary["terminal_checkpoint_sha256"],
             f"{run.name}: terminal checkpoint checksum mismatch")
    with (run / "scaling_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required(len(rows) == 4 and {row["granularity"] for row in rows} == set(widths),
             f"{run.name}: expected four distinct endpoints")
    endpoints = {}
    for row in rows:
        loss, perplexity = float(row["final_validation_loss"]), float(row["final_validation_perplexity"])
        required(np.isfinite(loss) and np.isclose(np.exp(loss), perplexity, rtol=1e-10)
                 and int(row["evaluation_target_tokens"]) > 0,
                 f"{run.name}/{row['granularity']}: invalid endpoint")
        endpoints[row["granularity"]] = dict(loss=loss, perplexity=perplexity,
            non_embedding_parameters=int(row["non_embedding_parameters"]),
            validation_manifest_hash=row["validation_manifest_hash"])
    return endpoints, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "comparison")
    args = parser.parse_args()
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is occupied: {output}")

    rows = []
    sources = {}
    for grid, widths in WIDTHS.items():
        reference = REFERENCES[grid]
        endpoints = {}
        curves = {}
        standalones = {w: saved_endpoints(reference, f"ST-{w}", 87132)[w] for w in widths}
        for width in widths:
            path = reference / "runs" / f"ST-{width}" / "terminal_validation_results.json"
            sources[str(path)] = sha256(path)

        for owner in OWNERS:
            endpoints[owner, "uniform"] = saved_endpoints(reference, owner, UPDATES)
            endpoints[owner, "epoch_linear_im"], run = new_endpoints(grid, owner, widths)
            for policy, location, arm in (("uniform", reference / "runs" / owner, owner),
                                          ("epoch_linear_im", run, run.name)):
                curves[owner, policy], _ = read_curves(location / "metrics.csv", widths)
                for width in widths:
                    required(np.isclose(float(curves[owner, policy][width]["loss"].iloc[-1]),
                                        float(endpoints[owner, policy][width]["loss"]), atol=1e-12, rtol=0),
                             f"{arm}/{width}: terminal curve does not match endpoint")
                for name in (("terminal_validation_results.json", "metrics.csv") if policy == "uniform"
                             else ("run_summary.json", "config.json", "scaling_results.csv", "metrics.csv")):
                    path = location / name
                    sources[str(path)] = sha256(path)

        for width in widths:
            all_endpoints = [standalones[width]] + [endpoints[o, p][width] for o in OWNERS
                                                    for p in ("uniform", "epoch_linear_im")]
            required(len({int(e["non_embedding_parameters"]) for e in all_endpoints}) == 1,
                     f"{grid}/{width}: active parameter counts differ")
            required(len({e["validation_manifest_hash"] for e in all_endpoints}) == 1,
                     f"{grid}/{width}: validation manifests differ")
            params = int(all_endpoints[0]["non_embedding_parameters"])
            standalone_loss = float(standalones[width]["loss"])
            for owner in OWNERS:
                baseline_loss = float(endpoints[owner, "uniform"][width]["loss"])
                for policy in ("uniform", "epoch_linear_im"):
                    e = endpoints[owner, policy][width]
                    rows.append(dict(grid=grid, width=width, owner=owner, policy=policy,
                        non_embedding_parameters=params, assigned_updates=UPDATES,
                        loss=float(e["loss"]), perplexity=float(e["perplexity"]),
                        delta_vs_uniform=float(e["loss"]) - baseline_loss,
                        gap_to_standalone=float(e["loss"]) - standalone_loss))
            e = standalones[width]
            rows.append(dict(grid=grid, width=width, owner="standalone", policy="standalone",
                non_embedding_parameters=params, assigned_updates=87132,
                loss=float(e["loss"]), perplexity=float(e["perplexity"]),
                delta_vs_uniform="", gap_to_standalone=0.0))

        x = np.array([int(standalones[w]["non_embedding_parameters"]) for w in widths])
        for metric, ylabel in (("loss", "Ordinary-validation loss"),
                               ("perplexity", "Ordinary-validation perplexity")):
            fig, ax = plt.subplots(figsize=(9.5, 5.5))
            for owner in OWNERS:
                for policy, linestyle, marker, name in (("uniform", ":", "o", "uniform"),
                                                        ("epoch_linear_im", "-", "s", "epoch shift")):
                    ax.plot(x, [float(endpoints[owner, policy][w][metric]) for w in widths],
                            color=COLORS[owner], linestyle=linestyle, marker=marker,
                            linewidth=1.8, label=f"{owner} · {name}")
            ax.scatter(x, [float(standalones[w][metric]) for w in widths],
                       color=STANDALONE_COLOR, marker="^", s=95, zorder=5,
                       label="Standalone · 1 epoch")
            ax.set(title=f"{grid.title()} · epoch-wise uniform → inverse membership · seed 42",
                   xlabel="Active non-embedding parameters", ylabel=ylabel)
            ax.set_xticks(x, widths)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            output.mkdir(parents=True, exist_ok=True)
            save(fig, output, f"{grid}_{metric}_vs_parameters")

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
        for ax, owner in zip(axes, OWNERS):
            ax.axhline(0, color="#666666", linewidth=1)
            delta = [float(endpoints[owner, "epoch_linear_im"][w]["loss"])
                     - float(endpoints[owner, "uniform"][w]["loss"]) for w in widths]
            ax.plot(x, delta, color=COLORS[owner], marker="s", linewidth=2)
            ax.set(title=owner, xlabel="Active non-embedding parameters")
            ax.set_xticks(x, widths)
            ax.grid(alpha=.25)
        axes[0].set_ylabel("Epoch shift − uniform validation loss (negative is better)")
        fig.suptitle(f"{grid.title()} · effect of epoch-wise width probabilities · seed 42")
        fig.tight_layout()
        save(fig, output, f"{grid}_delta_vs_uniform")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for ax, width in zip(axes.flat, widths):
            for owner in OWNERS:
                for policy, linestyle, name in (("uniform", ":", "uniform"),
                                                ("epoch_linear_im", "-", "epoch shift")):
                    frame = curves[owner, policy][width]
                    ax.plot(frame["step"] / 1000, frame["loss"], color=COLORS[owner],
                            linestyle=linestyle, linewidth=.9, label=f"{owner} · {name}")
            # A standalone has one terminal evaluation, so it is one point at its own update count.
            ax.scatter([87.132], [float(standalones[width]["loss"])],
                       color=STANDALONE_COLOR, marker="^", s=75, zorder=5,
                       label="Standalone terminal")
            ax.set(title=width, xlabel="Optimizer updates (thousands)",
                   ylabel="Ordinary-validation loss", xlim=(0, UPDATES / 1000))
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"{grid.title()} · ordinary-validation trajectories · seed 42")
        fig.tight_layout()
        save(fig, output, f"{grid}_validation_loss_vs_updates")

    for name, data in (("endpoints", rows),):
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        (output / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    audit_path = ROOT / "campaign" / "schedule_audit.json"
    sources[str(audit_path)] = sha256(audit_path)
    (output / "plot_sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    print(f"Saved 8 PNG/PDF figures and endpoint tables to {output}")
    for row in rows:
        if row["width"] == "g1000" and row["policy"] == "epoch_linear_im":
            print(f"{row['grid']} {row['owner']} g1000: {row['loss']:.6f}; "
                  f"vs uniform {row['delta_vs_uniform']:+.6f}; "
                  f"vs standalone {row['gap_to_standalone']:+.6f}")


if __name__ == "__main__":
    main()
