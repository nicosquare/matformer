#!/usr/bin/env python3
"""Queue the d_model=256 pilot matrix through Slurm with skip/resume logic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import ConfigError, parse_override, resolve_run_config
from utils.model_size import derive_token_budget_slug


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "dmodel256_pilot_comparison.yaml"
DEFAULT_SLURM_SCRIPT = REPO_ROOT / "scripts" / "slurm_dmodel256_pilot.sh"
DEFAULT_OUTPUT_ROOT = "outputs"
DEFAULT_TOKEN_BUDGET = 100_000_000
DEFAULT_OPTIMIZER_PRESET = "adam"
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_LEARNING_RATE_SCALE_RULE = "none"

RESERVED_OVERRIDE_KEYS = {
    "run.model_family",
    "run.sampling_mode",
    "run.granularity",
    "run.output_root",
    "run.run_id",
    "model.variant",
    "model.correction_mode",
    "model.membership_correction",
    "model.granularity_sampling_mode",
}

BATCH_OVERRIDE_KEYS = {
    "training.token_budget",
    "training.optimizer.name",
    "training.optimizer.preset",
    "training.learning_rate",
    "training.learning_rate_scale_rule",
}


@dataclass(frozen=True)
class ExperimentSpec:
    label: str
    run_overrides: tuple[str, ...]
    model_overrides: tuple[str, ...]


@dataclass(frozen=True)
class BatchSettings:
    token_budget: int
    optimizer_name: str | None
    optimizer_preset: str | None
    learning_rate: float
    learning_rate_scale_rule: str
    passthrough_overrides: tuple[str, ...]
    passthrough_override_hash: str | None


@dataclass(frozen=True)
class QueuedRun:
    spec: ExperimentSpec
    run_id: str
    output_dir: Path
    run_summary_path: Path
    resolved_config: dict[str, Any]
    sbatch_command: tuple[str, ...]
    completed: bool


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Queue the d_model=256 pilot matrix through Slurm.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Pilot config path.",
    )
    parser.add_argument(
        "--slurm-script",
        default=str(DEFAULT_SLURM_SCRIPT),
        help="Slurm submission script.",
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT),
        help="Root directory for run artifacts.",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=DEFAULT_TOKEN_BUDGET,
        help="Token budget shared by the whole batch.",
    )
    optimizer_group = parser.add_mutually_exclusive_group()
    optimizer_group.add_argument(
        "--optimizer-preset",
        default=DEFAULT_OPTIMIZER_PRESET,
        help="Optimizer preset to use for the batch.",
    )
    optimizer_group.add_argument(
        "--optimizer-name",
        default=None,
        help="Direct optimizer name to use for the batch.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Base learning rate shared by the whole batch.",
    )
    parser.add_argument(
        "--learning-rate-scale-rule",
        default=DEFAULT_LEARNING_RATE_SCALE_RULE,
        choices=("none", "linear", "sqrt"),
        help="Learning-rate scaling rule.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Additional training/model overrides forwarded to every run. "
            "Repeat this flag to pass multiple overrides."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sbatch commands without submitting them.",
    )
    return parser.parse_args(argv)


def build_experiment_specs() -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []

    for variant, variant_slug in (
        ("cat_llama", "cat"),
        ("matformer_llama", "slice"),
    ):
        specs.append(
            ExperimentSpec(
                label=f"nested-random-{variant_slug}-none-global",
                run_overrides=("run.model_family=nested",),
                model_overrides=(
                    f"model.variant={variant}",
                    "model.correction_mode=none",
                    "model.membership_correction=false",
                    "model.granularity_sampling_mode=global",
                ),
            )
        )
        specs.append(
            ExperimentSpec(
                label=f"nested-random-{variant_slug}-gmc-global",
                run_overrides=("run.model_family=nested",),
                model_overrides=(
                    f"model.variant={variant}",
                    "model.correction_mode=gmc",
                    "model.membership_correction=true",
                    "model.granularity_sampling_mode=global",
                ),
            )
        )
        specs.append(
            ExperimentSpec(
                label=f"nested-random-{variant_slug}-none-per-layer",
                run_overrides=("run.model_family=nested",),
                model_overrides=(
                    f"model.variant={variant}",
                    "model.correction_mode=none",
                    "model.membership_correction=false",
                    "model.granularity_sampling_mode=per_layer",
                ),
            )
        )
        specs.append(
            ExperimentSpec(
                label=f"nested-random-{variant_slug}-gmc-per-layer",
                run_overrides=("run.model_family=nested",),
                model_overrides=(
                    f"model.variant={variant}",
                    "model.correction_mode=gmc",
                    "model.membership_correction=true",
                    "model.granularity_sampling_mode=per_layer",
                ),
            )
        )
        if variant == "cat_llama":
            specs.append(
                ExperimentSpec(
                    label="nested-random-cat-lmc-per-layer",
                    run_overrides=("run.model_family=nested",),
                    model_overrides=(
                        "model.variant=cat_llama",
                        "model.correction_mode=lmc",
                        "model.membership_correction=true",
                        "model.granularity_sampling_mode=per_layer",
                    ),
                )
            )

    for variant, variant_slug in (
        ("cat_llama", "cat"),
        ("matformer_llama", "slice"),
    ):
        specs.append(
            ExperimentSpec(
                label=f"nested-all-{variant_slug}-none-global",
                run_overrides=(
                    "run.model_family=nested",
                    "run.sampling_mode=nested-all",
                ),
                model_overrides=(
                    f"model.variant={variant}",
                    "model.correction_mode=none",
                    "model.membership_correction=false",
                    "model.granularity_sampling_mode=global",
                ),
            )
        )
        specs.append(
            ExperimentSpec(
                label=f"nested-all-{variant_slug}-gmc-global",
                run_overrides=(
                    "run.model_family=nested",
                    "run.sampling_mode=nested-all",
                ),
                model_overrides=(
                    f"model.variant={variant}",
                    "model.correction_mode=gmc",
                    "model.membership_correction=true",
                    "model.granularity_sampling_mode=global",
                ),
            )
        )

    for granularity in ("s", "m", "l", "xl"):
        specs.append(
            ExperimentSpec(
                label=f"standalone-{granularity}",
                run_overrides=(
                    "run.model_family=standalone",
                    "run.sampling_mode=standalone",
                    f"run.granularity={granularity}",
                ),
                model_overrides=(
                    "model.correction_mode=none",
                    "model.membership_correction=false",
                ),
            )
        )

    return specs


def _parse_override_map(raw_overrides: Iterable[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw_override in raw_overrides:
        key, value = parse_override(raw_override)
        parsed[key] = value
    return parsed


def _format_override_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return str(value)


def _override(key: str, value: Any) -> str:
    return f"{key}={_format_override_value(value)}"


def _token_budget_slug(token_budget: int) -> str:
    budget_slug = derive_token_budget_slug(token_budget).removesuffix("_tokens")
    return f"tb{budget_slug}"


def _learning_rate_slug(learning_rate: float) -> str:
    return f"lr{format(learning_rate, '.6g').replace('.', 'p')}"


def _optimizer_label(
    optimizer_preset: str | None,
    optimizer_name: str | None,
) -> str:
    if optimizer_name:
        return optimizer_name.strip()
    if not optimizer_preset:
        return "adamw"
    preset = optimizer_preset.strip()
    if preset == "adam":
        return "adamw"
    return preset


def _extra_override_hash(extra_overrides: Iterable[str]) -> str | None:
    values = sorted(extra_overrides)
    if not values:
        return None
    digest = hashlib.sha1("\n".join(values).encode("utf-8")).hexdigest()
    return digest[:8]


def _normalize_batch_settings(args: argparse.Namespace) -> BatchSettings:
    override_map = _parse_override_map(args.override)

    token_budget = int(override_map.pop("training.token_budget", args.token_budget))
    learning_rate = float(
        override_map.pop("training.learning_rate", args.learning_rate)
    )
    learning_rate_scale_rule = str(
        override_map.pop(
            "training.learning_rate_scale_rule",
            args.learning_rate_scale_rule,
        )
    ).strip()
    if not learning_rate_scale_rule:
        learning_rate_scale_rule = DEFAULT_LEARNING_RATE_SCALE_RULE

    optimizer_name_override = override_map.pop("training.optimizer.name", None)
    optimizer_preset_override = override_map.pop("training.optimizer.preset", None)
    if optimizer_name_override is not None and optimizer_preset_override is not None:
        raise ConfigError(
            "training.optimizer.name and training.optimizer.preset cannot be set at the same time"
        )

    optimizer_name: str | None = None
    optimizer_preset: str | None = None
    if optimizer_name_override is not None:
        optimizer_name = str(optimizer_name_override).strip()
        if not optimizer_name:
            raise ConfigError("training.optimizer.name must be non-empty when set")
    else:
        if optimizer_preset_override is not None:
            optimizer_preset = str(optimizer_preset_override).strip()
            if not optimizer_preset:
                raise ConfigError(
                    "training.optimizer.preset must be non-empty when set"
                )
        elif args.optimizer_name is not None:
            optimizer_name = args.optimizer_name.strip()
            if not optimizer_name:
                raise ConfigError(
                    "training.optimizer.name must be non-empty when set"
                )
        else:
            optimizer_preset = args.optimizer_preset

    passthrough_overrides: list[str] = []
    for key in override_map:
        if key in RESERVED_OVERRIDE_KEYS:
            raise ConfigError(
                f"{key} is managed by the queue launcher and cannot be overridden directly"
            )
    for raw_override in args.override:
        key, _ = parse_override(raw_override)
        if key in BATCH_OVERRIDE_KEYS:
            continue
        if key in RESERVED_OVERRIDE_KEYS:
            raise ConfigError(
                f"{key} is managed by the queue launcher and cannot be overridden directly"
            )
        passthrough_overrides.append(raw_override)

    passthrough_hash = _extra_override_hash(passthrough_overrides)
    return BatchSettings(
        token_budget=token_budget,
        optimizer_name=optimizer_name,
        optimizer_preset=optimizer_preset,
        learning_rate=learning_rate,
        learning_rate_scale_rule=learning_rate_scale_rule,
        passthrough_overrides=tuple(passthrough_overrides),
        passthrough_override_hash=passthrough_hash,
    )


def _build_batch_overrides(settings: BatchSettings) -> list[str]:
    overrides = [
        _override("training.token_budget", settings.token_budget),
        _override("training.learning_rate", settings.learning_rate),
        _override(
            "training.learning_rate_scale_rule",
            settings.learning_rate_scale_rule,
        ),
    ]
    if settings.optimizer_name is not None:
        overrides.append(_override("training.optimizer.preset", None))
        overrides.append(_override("training.optimizer.name", settings.optimizer_name))
    else:
        overrides.append(
            _override("training.optimizer.preset", settings.optimizer_preset)
        )
    return overrides


def _build_batch_slug(settings: BatchSettings) -> str:
    optimizer_label = _optimizer_label(
        settings.optimizer_preset,
        settings.optimizer_name,
    )
    slug_parts = [
        _token_budget_slug(settings.token_budget),
        f"opt{optimizer_label}",
        _learning_rate_slug(settings.learning_rate),
        f"scale-{settings.learning_rate_scale_rule}",
    ]
    if settings.passthrough_override_hash is not None:
        slug_parts.append(f"x{settings.passthrough_override_hash}")
    return "-".join(slug_parts)


def _build_run_id(spec: ExperimentSpec, batch_slug: str) -> str:
    return f"dmodel256-{spec.label}-{batch_slug}-001"


def _build_submission_command(
    slurm_script: Path,
    config_path: Path,
    output_root: Path,
    run_id: str,
    spec: ExperimentSpec,
    settings: BatchSettings,
) -> list[str]:
    command = [
        "sbatch",
        str(slurm_script),
        "--repo-root",
        str(REPO_ROOT),
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
    ]
    for override in _build_batch_overrides(settings):
        command.extend(["--override", override])
    for override in settings.passthrough_overrides:
        command.extend(["--override", override])
    for override in spec.run_overrides:
        command.extend(["--override", override])
    for override in spec.model_overrides:
        command.extend(["--override", override])

    return command


def _build_resolve_overrides(
    output_root: Path,
    run_id: str,
    spec: ExperimentSpec,
    settings: BatchSettings,
) -> list[str]:
    overrides = [
        _override("run.output_root", str(output_root)),
        _override("run.run_id", run_id),
    ]
    overrides.extend(_build_batch_overrides(settings))
    overrides.extend(settings.passthrough_overrides)
    overrides.extend(spec.run_overrides)
    overrides.extend(spec.model_overrides)
    return overrides


def _run_summary_status(run_summary_path: Path) -> str | None:
    if not run_summary_path.is_file():
        return None
    try:
        with run_summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    status = summary.get("status")
    return status if isinstance(status, str) else None


def build_queued_runs(
    config_path: Path,
    output_root: Path,
    slurm_script: Path,
    settings: BatchSettings,
) -> list[QueuedRun]:
    batch_slug = _build_batch_slug(settings)
    queued_runs: list[QueuedRun] = []

    for spec in build_experiment_specs():
        run_id = _build_run_id(spec, batch_slug)
        resolve_overrides = _build_resolve_overrides(
            output_root,
            run_id,
            spec,
            settings,
        )
        resolved = resolve_run_config(
            config_path,
            overrides=resolve_overrides,
        )
        output_dir = Path(resolved["run"]["output_dir"])
        run_summary_path = output_dir / "run_summary.json"
        completed = _run_summary_status(run_summary_path) == "completed"
        queued_runs.append(
            QueuedRun(
                spec=spec,
                run_id=run_id,
                output_dir=output_dir,
                run_summary_path=run_summary_path,
                resolved_config=resolved,
                sbatch_command=tuple(
                    _build_submission_command(
                        slurm_script=slurm_script,
                        config_path=config_path,
                        output_root=output_root,
                        run_id=run_id,
                        spec=spec,
                        settings=settings,
                    )
                ),
                completed=completed,
            )
        )

    return queued_runs


def _print_queue_summary(
    queued_runs: Iterable[QueuedRun],
    submitted: int,
    skipped: int,
) -> None:
    print(f"Submitted: {submitted}")
    print(f"Skipped: {skipped}")
    for run in queued_runs:
        state = "completed" if run.completed else "pending"
        print(f"{state}: {run.run_id} -> {run.output_dir}")


def _submit_run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def main(argv: Iterable[str] | None = None) -> int:
    os.chdir(REPO_ROOT)
    args = parse_args(argv)

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    slurm_script = Path(args.slurm_script).expanduser()
    if not slurm_script.is_absolute():
        slurm_script = (REPO_ROOT / slurm_script).resolve()
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()

    settings = _normalize_batch_settings(args)
    queued_runs = build_queued_runs(
        config_path=config_path,
        output_root=output_root,
        slurm_script=slurm_script,
        settings=settings,
    )

    submitted = 0
    skipped = 0
    for run in queued_runs:
        if run.completed:
            skipped += 1
            print(f"Skipping completed run: {run.run_id}")
            continue

        print(f"Queueing run: {run.run_id}")
        print(f"  output_dir: {run.output_dir}")
        print(f"  sbatch: {' '.join(run.sbatch_command)}")
        if args.dry_run:
            submitted += 1
            continue

        result = _submit_run(list(run.sbatch_command))
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        submitted += 1

    _print_queue_summary(queued_runs, submitted, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
