#!/usr/bin/env python3
"""Preflight or submit the configurable elastic sign-dynamics matrix."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/controlled_exps/tinystories_instruct_sign_dynamics.yaml"
BUDGET_UNIT = 713_785_344
DEFAULT_LONG_H = 548
DEFAULT_SLURM_TIME = "24:00:00"
DEFAULT_ARMS = (
    "uniform_h1",
    "uniform_hlong",
    "balanced_h1",
    "balanced_hlong",
    "fixed_inverse_membership",
    "thompson",
)


class CampaignError(ValueError):
    pass


def _positive_int(value: str, name: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise CampaignError(f"{name} must be a positive integer") from error
    if result <= 0:
        raise CampaignError(f"{name} must be a positive integer")
    return result


def _nonnegative_int(value: str, name: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise CampaignError(f"{name} must be a nonnegative integer") from error
    if result < 0:
        raise CampaignError(f"{name} must be a nonnegative integer")
    return result


def _seeds(value: str) -> list[int]:
    seeds = [
        _nonnegative_int(item.strip(), "SIGN_DYNAMICS_SEEDS")
        for item in value.split(",")
        if item.strip()
    ]
    if not seeds or len(set(seeds)) != len(seeds):
        raise CampaignError("SIGN_DYNAMICS_SEEDS must contain unique comma-separated integers")
    return seeds


def _selected_arms(value: str | None) -> list[str]:
    arms = list(DEFAULT_ARMS) if not value else [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(arms) - set(DEFAULT_ARMS))
    if unknown or not arms or len(set(arms)) != len(arms):
        raise CampaignError(f"invalid SIGN_DYNAMICS_ARMS selection: {unknown or arms}")
    return arms


def _arm_contract(alias: str, long_h: int) -> tuple[str, str, list[str]]:
    if alias == "uniform_h1":
        return "uniform_h1", "uniform", [
            "model.granularity_sampling_mode=global",
            "model.global_sampling_schedule=random_with_replacement",
            "model.global_sampling_interval_steps=1",
        ]
    if alias == "uniform_hlong":
        return f"uniform_h{long_h}", "uniform", [
            "model.granularity_sampling_mode=global",
            "model.global_sampling_schedule=random_with_replacement",
            f"model.global_sampling_interval_steps={long_h}",
        ]
    if alias == "balanced_h1":
        return "balanced_h1", "balanced", [
            "model.granularity_sampling_mode=global",
            "model.global_sampling_schedule=balanced_cycle",
            "model.global_sampling_interval_steps=1",
        ]
    if alias == "balanced_hlong":
        return f"balanced_h{long_h}", "balanced", [
            "model.granularity_sampling_mode=global",
            "model.global_sampling_schedule=balanced_cycle",
            f"model.global_sampling_interval_steps={long_h}",
        ]
    if alias == "fixed_inverse_membership":
        return alias, "fixed_inverse_membership", [
            "model.granularity_sampling_mode=fixed_global",
            'model.global_sampling_distribution={"g250":0.12,"g500":0.16,"g750":0.24,"g1000":0.48}',
        ]
    if alias == "thompson":
        return alias, "thompson", [
            "model.granularity_sampling_mode=adaptive_global",
            "model.adaptive_sampler_strategy=thompson",
            "model.adaptive_controller.decision_interval_steps=25",
            "model.adaptive_controller.prior_mean=0.0",
            "model.adaptive_controller.prior_covariance=1.0",
            "model.adaptive_controller.observation_noise_variance=0.01",
            "model.adaptive_controller.process_noise_covariance=0.0001",
        ]
    raise CampaignError(f"unknown arm alias: {alias}")


def build_train_command(
    *,
    python_bin: str,
    experiment_root: Path,
    budget_multiplier: int,
    long_h: int,
    seed: int,
    arm_alias: str,
    tokenizer_dir: str | None,
    corpus_dir: str | None,
    preflight: bool,
) -> tuple[str, Path, list[str]]:
    arm_id, policy, arm_overrides = _arm_contract(arm_alias, long_h)
    run_dir = (
        experiment_root
        / "tinystories-instruct-sign-dynamics-v1"
        / f"x{budget_multiplier}"
        / "runs"
        / arm_id
        / f"s{seed}"
    )
    overrides = [
        f"run.run_id=s{seed}",
        f"run.seed={seed}",
        f"dataset.data_seed={seed}",
        f"training.token_budget={BUDGET_UNIT * budget_multiplier}",
        f"controlled_experiment.budget_multiplier={budget_multiplier}",
        f"controlled_experiment.arm_id={arm_id}",
        f"controlled_experiment.policy={policy}",
        f"evaluation.sign_dynamics.arm_id={arm_id}",
        *arm_overrides,
    ]
    if tokenizer_dir:
        overrides.append(f"model.tokenizer_dir={tokenizer_dir}")
    if corpus_dir:
        overrides.append(f"dataset.prepared_corpus_dir={corpus_dir}")
    command = [
        python_bin,
        "train.py",
        "--config",
        str(CONFIG.relative_to(REPO_ROOT)),
        "--run-id",
        f"s{seed}",
        "--output-dir",
        str(run_dir),
    ]
    if preflight:
        command.append("--preflight")
    for override in overrides:
        command.extend(("--override", override))
    return arm_id, run_dir, command


def _run(command: Sequence[str], *, dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "submit", "resume"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--slurm-script", default="scripts/slurm_tinystories_controlled.sh")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        multiplier = _positive_int(os.environ.get("SIGN_DYNAMICS_B_MULTIPLIER", "4"), "SIGN_DYNAMICS_B_MULTIPLIER")
        long_h = _positive_int(os.environ.get("SIGN_DYNAMICS_LONG_H", str(DEFAULT_LONG_H)), "SIGN_DYNAMICS_LONG_H")
        seeds = _seeds(os.environ.get("SIGN_DYNAMICS_SEEDS", "42"))
        arms = _selected_arms(os.environ.get("SIGN_DYNAMICS_ARMS"))
        slurm_time = os.environ.get(
            "SIGN_DYNAMICS_SLURM_TIME", DEFAULT_SLURM_TIME
        ).strip()
        if not slurm_time:
            raise CampaignError("SIGN_DYNAMICS_SLURM_TIME must be non-empty")
    except CampaignError as error:
        raise SystemExit(str(error)) from error
    experiment_root = Path(os.environ.get("MATFORMER_EXPERIMENT_ROOT", "outputs")).expanduser().resolve()
    tokenizer_dir = os.environ.get("SIGN_DYNAMICS_TOKENIZER_DIR")
    corpus_dir = os.environ.get("SIGN_DYNAMICS_CORPUS_DIR")
    exclusions = os.environ.get("SIGN_DYNAMICS_NODE_EXCLUSIONS", "").strip()

    for arm in arms:
        for seed in seeds:
            arm_id, run_dir, train_command = build_train_command(
                python_bin=args.python_bin,
                experiment_root=experiment_root,
                budget_multiplier=multiplier,
                long_h=long_h,
                seed=seed,
                arm_alias=arm,
                tokenizer_dir=tokenizer_dir,
                corpus_dir=corpus_dir,
                preflight=args.command == "preflight",
            )
            if args.command == "preflight":
                _run(train_command, dry_run=args.dry_run)
                continue
            # The shared TinyStories wrapper carries a 30-minute directive for
            # short controlled jobs. Override it for this long campaign.
            slurm = ["sbatch", f"--time={slurm_time}"]
            if exclusions:
                slurm.extend(("--exclude", exclusions))
            slurm.extend(
                (
                    f"--job-name=sign-{arm_id}-s{seed}-x{multiplier}",
                    args.slurm_script,
                    "--python-bin",
                    args.python_bin,
                    *train_command[2:],
                )
            )
            _run(slurm, dry_run=args.dry_run)
            print(f"matrix_run={run_dir} mode={args.command}", flush=True)


if __name__ == "__main__":
    main()
