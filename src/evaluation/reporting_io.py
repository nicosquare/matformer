"""CSV artifact loading and metadata enrichment helpers for reporting."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .reporting_styles import PARAMETER_COUNT_FIELDS

__all__ = [
    "adaptive_sampler_strategy_from_saved_config",
    "controller_method_family_from_saved_config",
    "controller_method_version_from_saved_config",
    "controller_reset_enabled_from_saved_config",
    "controller_reset_policy_from_saved_config",
    "controller_scope_from_saved_config",
    "ControllerGranularityTimeline",
    "ControllerSelectionWindow",
    "GlobalSamplingAction",
    "GlobalSamplingHistory",
    "GradientInterferenceHistory",
    "GradientInterferencePair",
    "GradientInterferenceReportingError",
    "GradientInterferenceSnapshot",
    "FinalHoldoutReportingError",
    "LearningRateSchedule",
    "PanelGradAction",
    "PanelGradHistory",
    "PanelGradRefresh",
    "OptimizerStateRunArtifacts",
    "OptimizerStateReportingError",
    "config_path_for_scaling_row",
    "correction_mode_from_saved_config",
    "enrich_metrics_metadata_from_run_config",
    "enrich_scaling_metadata_from_run_config",
    "granularity_sampling_mode_from_saved_config",
    "global_sampling_distribution_from_saved_config",
    "global_sampling_interval_steps_from_saved_config",
    "global_sampling_schedule_from_saved_config",
    "global_sampling_schedule_version_from_saved_config",
    "iter_controller_granularity_timelines",
    "iter_global_sampling_histories",
    "iter_gradient_interference_histories",
    "iter_learning_rate_schedules",
    "iter_panelgrad_histories",
    "iter_csv_artifact_rows",
    "membership_correction_from_saved_config",
    "model_variant_from_saved_config",
    "panelgrad_history_as_timeline",
    "global_sampling_history_as_timeline",
    "read_csv_artifacts",
    "read_csv_artifacts_filtered",
    "read_final_holdout_scaling_rows",
    "recompute_parameter_counts",
    "refresh_scaling_parameter_counts",
    "resolved_sampling_mode_from_saved_config",
    "seed_independent_validation_contract",
    "validation_split_filter",
    "with_default_model_variant",
    "load_optimizer_state_run",
]


BAYESIAN_CONTROLLER_METHOD_FAMILY = "bayesian_gaussian_linear_thompson"
PANELGRAD_METHOD_FAMILIES = {
    "panelgrad_gradient_rms": "gradient_rms",
    "panelgrad_gradient_l2": "gradient_l2",
}


class FinalHoldoutReportingError(ValueError):
    """Raised when final-holdout rows cannot form one valid comparison set."""


class OptimizerStateReportingError(ValueError):
    """Raised when one explicit paired-run directory is not auditable."""


@dataclass(frozen=True)
class OptimizerStateRunArtifacts:
    run_dir: Path
    config: Mapping[str, Any]
    summary: Mapping[str, Any]
    metrics: tuple[Mapping[str, str], ...]
    run_id: str
    seed: int
    state_scope: str
    ordered_widths: tuple[str, ...]
    paired_control_signature: str
    checkpoint_path: Path
    checkpoint_sha256: str


def load_optimizer_state_run(run_dir: str | Path) -> OptimizerStateRunArtifacts:
    """Load and validate one explicit optimizer-state comparison directory."""

    root = Path(run_dir).expanduser().resolve()
    required = {
        "config": root / "config.json",
        "summary": root / "run_summary.json",
        "metrics": root / "metrics.csv",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise OptimizerStateReportingError(
            f"Run {root} is missing required artifacts: {', '.join(missing)}"
        )
    config = _read_json_mapping(required["config"], artifact_name="config.json")
    summary = _read_json_mapping(
        required["summary"], artifact_name="run_summary.json"
    )
    try:
        with required["metrics"].open("r", encoding="utf-8", newline="") as source:
            metrics = tuple(dict(row) for row in csv.DictReader(source))
    except (OSError, csv.Error, UnicodeError) as error:
        raise OptimizerStateReportingError(
            f"Cannot read metrics.csv for {root}"
        ) from error
    if summary.get("status") != "completed":
        raise OptimizerStateReportingError(f"Run {root} is not completed")
    training = config.get("training", {})
    model = config.get("model", {})
    scope = str(
        summary.get("optimizer_state_scope")
        or training.get("optimizer_state_scope")
        or "shared"
    )
    if scope not in {"shared", "per_granularity"}:
        raise OptimizerStateReportingError(
            f"Run {root} has unknown optimizer state scope {scope!r}"
        )
    ordered = tuple(
        str(label)
        for label in (
            summary.get("ordered_optimizer_granularities")
            or training.get("optimizer_state_contract", {}).get(
                "ordered_granularities", []
            )
            or model.get("granularities", [])
        )
    )
    if len(ordered) < 2 or len(set(ordered)) != len(ordered):
        raise OptimizerStateReportingError(
            f"Run {root} has invalid ordered optimizer widths"
        )
    signature = summary.get("comparison_control_signature") or config.get(
        "comparison_control_signature"
    )
    if not signature:
        raise OptimizerStateReportingError(
            f"Run {root} lacks comparison_control_signature"
        )
    checkpoint_value = (
        summary.get("terminal_checkpoint_path")
        or summary.get("latest_checkpoint_path")
        or summary.get("final_checkpoint_path")
    )
    if checkpoint_value in (None, ""):
        raise OptimizerStateReportingError(f"Run {root} lacks a terminal checkpoint")
    checkpoint = Path(str(checkpoint_value)).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    if not checkpoint.is_file():
        alternate = root / "checkpoints" / checkpoint.name
        checkpoint = alternate if alternate.is_file() else checkpoint
    if not checkpoint.is_file():
        raise OptimizerStateReportingError(
            f"Run {root} terminal checkpoint does not exist: {checkpoint}"
        )
    purpose = summary.get("terminal_checkpoint_purpose", "resumable_training")
    if purpose != "resumable_training":
        raise OptimizerStateReportingError(
            f"Run {root} terminal checkpoint is not resumable_training"
        )
    committed = int(summary.get("committed_optimizer_steps", 0))
    updates = summary.get("optimizer_successful_update_counts", {})
    exposures = summary.get("optimizer_exposure_counts", {})
    if (
        not summary.get("optimizer_accounting_reconciled", False)
        or sum(int(value) for value in updates.values()) != committed
        or {str(key): int(value) for key, value in updates.items()}
        != {str(key): int(value) for key, value in exposures.items()}
    ):
        raise OptimizerStateReportingError(
            f"Run {root} optimizer ownership counts do not reconcile"
        )
    committed_rows = [
        row
        for row in metrics
        if row.get("split") == "train"
        and str(row.get("optimizer_step_committed", "")).strip().lower()
        in {"1", "true", "yes"}
    ]
    if len(committed_rows) != committed:
        raise OptimizerStateReportingError(
            f"Run {root} committed metric rows do not match the summary"
        )
    action_ids = [row.get("optimizer_action_id") for row in committed_rows]
    if any(value in (None, "") for value in action_ids) or len(set(action_ids)) != committed:
        raise OptimizerStateReportingError(
            f"Run {root} lacks one unique action ID per committed step"
        )
    owner_counts = {width: 0 for width in ordered}
    for row in committed_rows:
        owner = str(row.get("selected_optimizer_granularity") or "")
        if owner not in owner_counts:
            raise OptimizerStateReportingError(
                f"Run {root} metric row has unknown optimizer owner {owner!r}"
            )
        owner_counts[owner] += 1
    if owner_counts != {str(key): int(value) for key, value in exposures.items()}:
        raise OptimizerStateReportingError(
            f"Run {root} metric ownership does not match exposures"
        )
    recorded_bytes = summary.get("terminal_checkpoint_bytes")
    if recorded_bytes is not None and int(recorded_bytes) != checkpoint.stat().st_size:
        raise OptimizerStateReportingError(
            f"Run {root} terminal checkpoint byte count changed"
        )
    checkpoint_sha256 = _sha256_file(checkpoint)
    recorded_hash = summary.get("terminal_checkpoint_sha256")
    if recorded_hash not in (None, checkpoint_sha256):
        raise OptimizerStateReportingError(
            f"Run {root} terminal checkpoint hash changed"
        )
    return OptimizerStateRunArtifacts(
        run_dir=root,
        config=config,
        summary=summary,
        metrics=metrics,
        run_id=str(summary.get("run_id") or config.get("run", {}).get("run_id")),
        seed=int(summary.get("seed", config.get("run", {}).get("seed"))),
        state_scope=scope,
        ordered_widths=ordered,
        paired_control_signature=str(signature),
        checkpoint_path=checkpoint.resolve(),
        checkpoint_sha256=checkpoint_sha256,
    )


@dataclass(frozen=True)
class ControllerSelectionWindow:
    """Compact selection record extracted from one controller journal event."""

    window_index: int
    start_step: int
    end_step: int
    start_tokens: int
    end_tokens: int
    block_granularities: tuple[str, ...]
    terminal_incomplete: bool


@dataclass(frozen=True)
class ControllerGranularityTimeline:
    """Plot-ready controller history without posterior vectors or covariances."""

    run_id: str
    scope: str
    ordered_granularities: tuple[str, ...]
    block_count: int
    row_labels: tuple[str, ...]
    token_budget: int
    windows: tuple[ControllerSelectionWindow, ...]
    model_variant: str | None
    correction_mode: str | None
    membership_correction: bool | None
    config_path: Path
    journal_path: Path
    scheduler_warmup_steps: int = 0
    scheduler_warmup_tokens: int = 0


@dataclass(frozen=True)
class GlobalSamplingAction:
    """One complete global granularity trained for one committed step."""

    step: int
    start_tokens: int
    end_tokens: int
    granularity: str
    decision_index: int
    sampled_probability: float | None


@dataclass(frozen=True)
class GlobalSamplingHistory:
    """Canonical per-step history shared by all global sampling policies."""

    run_id: str
    policy_identity: str
    policy_label: str
    ordered_granularities: tuple[str, ...]
    token_budget: int
    actions: tuple[GlobalSamplingAction, ...]
    decision_count: int
    comparison_key: str
    model_variant: str | None
    correction_mode: str | None
    membership_correction: bool | None
    seed: int | None
    config_path: Path
    metrics_path: Path
    target_probabilities: tuple[float, ...] | None = None
    global_sampling_schedule: str | None = None
    global_sampling_schedule_version: int | None = None
    scheduler_warmup_steps: int = 0
    scheduler_warmup_tokens: int = 0


@dataclass(frozen=True)
class LearningRateSchedule:
    """Configured optimizer learning-rate trajectory for one run."""

    run_id: str
    scheduler_name: str
    scheduler_kwargs: Mapping[str, Any]
    peak_learning_rate: float
    warmup_steps: int
    max_steps: int
    expected_tokens_per_step: float
    token_budget: int
    model_variant: str | None
    correction_mode: str | None
    membership_correction: bool | None
    config_path: Path
    scheduler_contract: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PanelGradAction:
    """One committed PanelGrad categorical action."""

    step: int
    start_tokens: int
    end_tokens: int
    granularity: str
    sampled_probability: float


@dataclass(frozen=True)
class PanelGradRefresh:
    """Compact diagnostics from one complete full-panel refresh."""

    refresh_index: int
    boundary_step: int
    boundary_tokens: int
    importance_metric: str
    importance_scores: tuple[float, ...]
    gradient_rms_scores: tuple[float, ...]
    q: tuple[float, ...]
    p: tuple[float, ...]
    entropy: float
    min_probability: float
    max_probability: float
    active_epsilon: float
    epsilon_schedule_step: int
    duration_seconds: float
    backward_evaluations: int
    controller_target_tokens: int

@dataclass(frozen=True)
class PanelGradHistory:
    """Plot-ready PanelGrad actions and refresh diagnostics."""

    run_id: str
    importance_metric: str
    ordered_granularities: tuple[str, ...]
    token_budget: int
    actions: tuple[PanelGradAction, ...]
    refreshes: tuple[PanelGradRefresh, ...]
    model_variant: str | None
    correction_mode: str | None
    membership_correction: bool | None
    config_path: Path
    metrics_path: Path
    journal_path: Path
    scheduler_warmup_steps: int = 0
    scheduler_warmup_tokens: int = 0


class GradientInterferenceReportingError(ValueError):
    """A completed diagnostic run has unusable reporting artifacts."""


@dataclass(frozen=True)
class GradientInterferencePair:
    """One compact unordered-granularity cosine measurement."""

    left_granularity: str
    right_granularity: str
    cosine: float | None
    has_zero_norm: bool


@dataclass(frozen=True)
class GradientInterferenceSnapshot:
    """Plot-ready values retained from one diagnostic milestone."""

    step: int
    tokens_seen: int
    milestone_reasons: tuple[str, ...]
    pairs: tuple[GradientInterferencePair, ...]


@dataclass(frozen=True)
class GradientInterferenceHistory:
    """A complete diagnostic run with large journal fields discarded."""

    run_id: str
    ordered_granularities: tuple[str, ...]
    unordered_pairs: tuple[tuple[str, str], ...]
    token_budget: int
    sampling_interval_steps: int
    seed: int
    comparison_contract: str
    diagnostic_contract_hash: str
    fixed_probe_manifest_hash: str
    controlled_support_hash: str
    snapshots: tuple[GradientInterferenceSnapshot, ...]
    model_variant: str | None
    correction_mode: str | None
    membership_correction: bool | None
    config_path: Path
    summary_path: Path
    journal_path: Path
    scheduler_warmup_steps: int = 0
    scheduler_warmup_tokens: int = 0


def _learning_rate_schedule_from_saved_config(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> LearningRateSchedule:
    run = _required_mapping(config.get("run"), "config.run")
    training = _required_mapping(config.get("training"), "config.training")
    run_id = str(run.get("run_id") or config_path.parent.name).strip()
    if not run_id:
        raise ValueError("config.run.run_id must be nonempty")

    max_steps = _positive_int(training.get("max_steps"), "config.training.max_steps")
    peak_learning_rate = _finite_float(
        training.get("resolved_learning_rate", training.get("learning_rate")),
        "config.training.resolved_learning_rate",
    )
    if peak_learning_rate <= 0.0:
        raise ValueError("config.training.resolved_learning_rate must be positive")

    scheduler = training.get("scheduler")
    scheduler = scheduler if isinstance(scheduler, Mapping) else {}
    scheduler_name = str(
        training.get("scheduler_name") or scheduler.get("name") or "cosine"
    ).strip()
    if not scheduler_name:
        raise ValueError("configured scheduler name must be nonempty")
    scheduler_kwargs = training.get("scheduler_kwargs")
    if not isinstance(scheduler_kwargs, Mapping):
        scheduler_kwargs = scheduler.get("kwargs", {})
    if not isinstance(scheduler_kwargs, Mapping):
        raise ValueError("configured scheduler kwargs must be a mapping")
    scheduler_kwargs = {
        str(key): value
        for key, value in scheduler_kwargs.items()
        if str(key) != "warmup_steps"
    }
    scheduler_contract = training.get("scheduler_contract")
    if not isinstance(scheduler_contract, Mapping):
        scheduler_contract = scheduler.get("contract")
    if scheduler_contract is not None and not isinstance(
        scheduler_contract, Mapping
    ):
        raise ValueError("configured scheduler contract must be a mapping")

    warmup_raw = training.get("resolved_warmup_steps")
    if warmup_raw is None:
        warmup_raw = scheduler.get("resolved_warmup_steps")
    if warmup_raw is None:
        warmup_raw = training.get("warmup_steps")
    if warmup_raw is None:
        warmup_ratio = _finite_float(
            training.get("warmup_ratio", 0.0), "config.training.warmup_ratio"
        )
        if warmup_ratio < 0.0:
            raise ValueError("config.training.warmup_ratio must be nonnegative")
        warmup_raw = math.ceil(max_steps * warmup_ratio)
    warmup_steps = _nonnegative_int(
        warmup_raw, "config.training.resolved_warmup_steps"
    )
    if warmup_steps > max_steps:
        raise ValueError("scheduler warmup cannot exceed config.training.max_steps")

    expected_tokens_raw = training.get("expected_tokens_per_step")
    token_budget = _positive_int(
        training.get("token_budget"), "config.training.token_budget"
    )
    if expected_tokens_raw is None:
        expected_tokens_per_step = token_budget / max_steps
    else:
        expected_tokens_per_step = _finite_float(
            expected_tokens_raw, "config.training.expected_tokens_per_step"
        )
        if expected_tokens_per_step <= 0.0:
            raise ValueError("config.training.expected_tokens_per_step must be positive")

    return LearningRateSchedule(
        run_id=run_id,
        scheduler_name=scheduler_name,
        scheduler_kwargs=dict(scheduler_kwargs),
        peak_learning_rate=peak_learning_rate,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        expected_tokens_per_step=expected_tokens_per_step,
        token_budget=token_budget,
        model_variant=model_variant_from_saved_config(dict(config)),
        correction_mode=correction_mode_from_saved_config(dict(config)),
        membership_correction=membership_correction_from_saved_config(dict(config)),
        config_path=config_path,
        scheduler_contract=(
            dict(scheduler_contract)
            if isinstance(scheduler_contract, Mapping)
            else None
        ),
    )


def iter_learning_rate_schedules(
    input_root: str | Path,
) -> Iterator[LearningRateSchedule]:
    """Discover configured LR schedules without requiring per-step LR logging."""

    for config_path in sorted(Path(input_root).rglob("config.json")):
        try:
            with config_path.open("r", encoding="utf-8") as config_file:
                config = json.load(config_file)
            training = config.get("training") if isinstance(config, Mapping) else None
            if not isinstance(training, Mapping):
                continue
            if (
                training.get("max_steps") is None
                or training.get("token_budget") is None
                or training.get(
                    "resolved_learning_rate", training.get("learning_rate")
                )
                is None
            ):
                # Historical reporting fixtures and old runs may not contain the
                # resolved scheduler contract needed for faithful reconstruction.
                continue
            yield _learning_rate_schedule_from_saved_config(
                config,
                config_path=config_path,
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            warnings.warn(
                f"Skipping LR schedule {config_path}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )


def _scheduler_warmup_boundaries(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> tuple[int, int]:
    try:
        schedule = _learning_rate_schedule_from_saved_config(
            config,
            config_path=config_path,
        )
    except (ValueError, TypeError, KeyError):
        return 0, 0
    return (
        schedule.warmup_steps,
        min(
            int(schedule.warmup_steps * schedule.expected_tokens_per_step),
            schedule.token_budget,
        ),
    )


def iter_gradient_interference_histories(
    input_root: str | Path,
) -> Iterator[GradientInterferenceHistory]:
    """Yield complete, enabled diagnostic runs as compact streaming histories.

    Discovery begins from saved configurations so a stray journal can never
    activate reporting. Incomplete runs are intentionally silent. Once a run
    claims completion, every artifact and identity is treated as required.
    """

    input_root = Path(input_root)
    seen_replications: dict[tuple[str, int, int], Path] = {}
    for config_path in sorted(input_root.rglob("config.json")):
        try:
            with config_path.open("r", encoding="utf-8") as config_file:
                config = json.load(config_file)
        except (OSError, json.JSONDecodeError):
            # A malformed config cannot establish that the opt-in diagnostic
            # was enabled. Other reporting readers retain their own behavior.
            continue
        if not isinstance(config, Mapping):
            continue
        diagnostic = config.get("evaluation", {}).get("gradient_interference", {})
        if not isinstance(diagnostic, Mapping) or not diagnostic.get("enabled", False):
            continue

        run = config.get("run")
        run_id = (
            str(run.get("run_id") or config_path.parent.name).strip()
            if isinstance(run, Mapping)
            else config_path.parent.name
        )
        artifact_name = str(diagnostic.get("artifact_path") or "gradient_interference.jsonl")
        journal_path = config_path.parent / artifact_name
        summary_path = config_path.parent / "run_summary.json"
        expected_count = len(diagnostic.get("resolved_steps", ()))

        if not summary_path.is_file():
            # A partial journal without a summary is an in-progress run. A full
            # journal proves that training reached every milestone, so losing
            # its completion summary is an attributable reporting failure.
            if _nonempty_line_count(journal_path) < expected_count:
                continue
            raise _gradient_reporting_error(
                run_id, summary_path, "completed diagnostic journal has no run_summary.json"
            )
        try:
            with summary_path.open("r", encoding="utf-8") as summary_file:
                summary = json.load(summary_file)
        except (OSError, json.JSONDecodeError) as error:
            raise _gradient_reporting_error(
                run_id, summary_path, f"invalid run summary: {error}"
            ) from error
        if not isinstance(summary, Mapping):
            raise _gradient_reporting_error(run_id, summary_path, "run summary is not a JSON object")
        if str(summary.get("status") or "").strip().lower() != "completed":
            continue

        try:
            history = _read_gradient_interference_history(
                config=config,
                summary=summary,
                config_path=config_path,
                summary_path=summary_path,
                journal_path=journal_path,
            )
        except GradientInterferenceReportingError:
            raise
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise _gradient_reporting_error(run_id, config_path, str(error)) from error
        replication_key = (
            history.comparison_contract,
            history.sampling_interval_steps,
            history.seed,
        )
        previous_path = seen_replications.get(replication_key)
        if previous_path is not None:
            raise GradientInterferenceReportingError(
                "Duplicate gradient-interference diagnostic runs for comparison "
                f"contract {history.comparison_contract}, H={history.sampling_interval_steps}, "
                f"seed={history.seed}: {previous_path} and {config_path}"
            )
        seen_replications[replication_key] = config_path
        yield history


def _nonempty_line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open("r", encoding="utf-8") as artifact:
            return sum(1 for line in artifact if line.strip())
    except OSError:
        return 0


def _gradient_reporting_error(
    run_id: str, artifact_path: Path, message: str
) -> GradientInterferenceReportingError:
    return GradientInterferenceReportingError(
        f"Gradient-interference reporting failed for run {run_id!r}, "
        f"artifact {artifact_path}: {message}"
    )


def _read_gradient_interference_history(
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    config_path: Path,
    summary_path: Path,
    journal_path: Path,
) -> GradientInterferenceHistory:
    from src.training.gradient_interference import (
        GradientInterferenceError,
        snapshot_id,
        validate_snapshot_record,
    )

    run = _required_mapping(config.get("run"), "config.run")
    model = _required_mapping(config.get("model"), "config.model")
    training = _required_mapping(config.get("training"), "config.training")
    evaluation = _required_mapping(config.get("evaluation"), "config.evaluation")
    diagnostic = _required_mapping(
        evaluation.get("gradient_interference"),
        "config.evaluation.gradient_interference",
    )
    run_id = str(run.get("run_id") or "").strip()
    if not run_id:
        raise _gradient_reporting_error(run_id, config_path, "config.run.run_id is empty")

    raw_granularities = model.get("granularities")
    if not isinstance(raw_granularities, list) or len(raw_granularities) < 2:
        raise _gradient_reporting_error(
            run_id, config_path, "config.model.granularities must contain at least two labels"
        )
    ordered_granularities = tuple(str(label) for label in raw_granularities)
    if any(not label for label in ordered_granularities) or len(
        set(ordered_granularities)
    ) != len(ordered_granularities):
        raise _gradient_reporting_error(
            run_id, config_path, "config.model.granularities contains empty or duplicate labels"
        )
    unordered_pairs = tuple(
        (ordered_granularities[left], ordered_granularities[right])
        for left in range(len(ordered_granularities))
        for right in range(left + 1, len(ordered_granularities))
    )
    token_budget = _positive_int(training.get("token_budget"), "config.training.token_budget")
    interval_steps = _positive_int(
        model.get("global_sampling_interval_steps", 1),
        "config.model.global_sampling_interval_steps",
    )
    seed = _signed_int(run.get("seed"), "config.run.seed")

    expected_milestones = diagnostic.get("resolved_milestones")
    if not isinstance(expected_milestones, list) or not expected_milestones:
        raise _gradient_reporting_error(
            run_id, config_path, "resolved diagnostic milestones are missing"
        )
    expected_steps = [
        item.get("step") if isinstance(item, Mapping) else None
        for item in expected_milestones
    ]
    expected_reasons = [
        item.get("reasons") if isinstance(item, Mapping) else None
        for item in expected_milestones
    ]
    if (
        any(isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in expected_steps)
        or expected_steps != sorted(set(expected_steps))
        or any(
            not isinstance(reasons, list)
            or not reasons
            or any(not isinstance(reason, str) or not reason for reason in reasons)
            for reasons in expected_reasons
        )
        or diagnostic.get("resolved_steps") != expected_steps
    ):
        raise _gradient_reporting_error(
            run_id, config_path, "resolved diagnostic milestones are invalid"
        )

    summary_expected = summary.get("gradient_interference_expected_steps")
    summary_measured = summary.get("gradient_interference_measured_steps")
    summary_count = summary.get("gradient_interference_snapshot_count")
    if (
        summary.get("run_id") != run_id
        or summary_expected != expected_steps
        or summary_measured != expected_steps
        or isinstance(summary_count, bool)
        or not isinstance(summary_count, int)
        or summary_count != len(expected_steps)
        or _nonnegative_int(summary.get("tokens_seen"), "run_summary.tokens_seen")
        < token_budget
    ):
        raise _gradient_reporting_error(
            run_id,
            summary_path,
            "completed run summary has mismatched identity or incomplete diagnostic coverage",
        )
    recorded_path = summary.get("gradient_interference_path")
    if not isinstance(recorded_path, str) or Path(recorded_path).name != journal_path.name:
        raise _gradient_reporting_error(
            run_id, summary_path, "gradient-interference journal path does not match config"
        )
    expected_hash = summary.get("gradient_interference_journal_hash")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise _gradient_reporting_error(
            run_id, summary_path, "gradient-interference journal hash is invalid"
        )
    if not journal_path.is_file():
        raise _gradient_reporting_error(run_id, journal_path, "journal is missing")
    actual_hash = _sha256_file(journal_path)
    if actual_hash != expected_hash:
        raise _gradient_reporting_error(run_id, journal_path, "journal hash mismatch")

    fixed_probe_hash = _required_identity_hash(
        diagnostic, "fixed_probe_manifest_hash", run_id, config_path
    )
    controlled_support_hash = _required_identity_hash(
        diagnostic, "controlled_support_hash", run_id, config_path
    )
    diagnostic_contract_hash = _required_identity_hash(
        diagnostic, "diagnostic_contract_hash", run_id, config_path
    )
    snapshots: list[GradientInterferenceSnapshot] = []
    snapshot_ids: set[str] = set()
    previous_tokens = -1
    with journal_path.open("r", encoding="utf-8") as journal_file:
        for line_number, line in enumerate(journal_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise _gradient_reporting_error(
                    run_id, journal_path, f"invalid JSON on line {line_number}: {error.msg}"
                ) from error
            if not isinstance(record, Mapping):
                raise _gradient_reporting_error(
                    run_id, journal_path, f"line {line_number} is not a JSON object"
                )
            try:
                validate_snapshot_record(
                    record, expected_granularity_count=len(ordered_granularities)
                )
            except (GradientInterferenceError, TypeError, ValueError) as error:
                raise _gradient_reporting_error(
                    run_id, journal_path, f"schema failure on line {line_number}: {error}"
                ) from error
            if len(snapshots) >= len(expected_steps):
                raise _gradient_reporting_error(
                    run_id, journal_path, "journal contains unexpected extra snapshots"
                )
            expected_step = expected_steps[len(snapshots)]
            expected_reason_list = expected_reasons[len(snapshots)]
            step = _record_nonnegative_int(record.get("step"), "step", run_id, journal_path, line_number)
            tokens_seen = _record_nonnegative_int(
                record.get("tokens_seen"), "tokens_seen", run_id, journal_path, line_number
            )
            identifier = str(record.get("snapshot_id") or "")
            if identifier in snapshot_ids or step != expected_step:
                raise _gradient_reporting_error(
                    run_id, journal_path, f"duplicate or unexpected snapshot on line {line_number}"
                )
            if identifier != snapshot_id(config, step):
                raise _gradient_reporting_error(
                    run_id, journal_path, f"snapshot identity mismatch on line {line_number}"
                )
            if tokens_seen < previous_tokens:
                raise _gradient_reporting_error(
                    run_id, journal_path, f"tokens_seen decreases on line {line_number}"
                )
            if record.get("milestone_reasons") != expected_reason_list:
                raise _gradient_reporting_error(
                    run_id, journal_path, f"milestone reasons mismatch on line {line_number}"
                )
            _validate_snapshot_identities(
                record,
                run_id=run_id,
                fixed_probe_hash=fixed_probe_hash,
                controlled_support_hash=controlled_support_hash,
                diagnostic_contract_hash=diagnostic_contract_hash,
                diagnostic=diagnostic,
                artifact_path=journal_path,
                line_number=line_number,
            )
            measurements = record["granularities"]
            if [item.get("granularity") for item in measurements if isinstance(item, Mapping)] != list(ordered_granularities):
                raise _gradient_reporting_error(
                    run_id, journal_path, f"granularity ordering mismatch on line {line_number}"
                )
            pairs = _compact_gradient_pairs(
                record["pairs"],
                expected_pairs=unordered_pairs,
                run_id=run_id,
                journal_path=journal_path,
                line_number=line_number,
            )
            snapshot_ids.add(identifier)
            previous_tokens = tokens_seen
            snapshots.append(
                GradientInterferenceSnapshot(
                    step=step,
                    tokens_seen=tokens_seen,
                    milestone_reasons=tuple(expected_reason_list),
                    pairs=pairs,
                )
            )
    if len(snapshots) != len(expected_steps):
        raise _gradient_reporting_error(
            run_id, journal_path, "journal is missing resolved milestones"
        )

    comparison_contract = _gradient_comparison_contract(
        config,
        ordered_granularities=ordered_granularities,
        diagnostic_contract_hash=diagnostic_contract_hash,
        fixed_probe_hash=fixed_probe_hash,
        controlled_support_hash=controlled_support_hash,
    )
    scheduler_warmup_steps, scheduler_warmup_tokens = (
        _scheduler_warmup_boundaries(config, config_path=config_path)
    )
    return GradientInterferenceHistory(
        run_id=run_id,
        ordered_granularities=ordered_granularities,
        unordered_pairs=unordered_pairs,
        token_budget=token_budget,
        sampling_interval_steps=interval_steps,
        seed=seed,
        comparison_contract=comparison_contract,
        diagnostic_contract_hash=diagnostic_contract_hash,
        fixed_probe_manifest_hash=fixed_probe_hash,
        controlled_support_hash=controlled_support_hash,
        snapshots=tuple(snapshots),
        model_variant=model_variant_from_saved_config(dict(config)),
        correction_mode=correction_mode_from_saved_config(dict(config)),
        membership_correction=membership_correction_from_saved_config(dict(config)),
        config_path=config_path,
        summary_path=summary_path,
        journal_path=journal_path,
        scheduler_warmup_steps=scheduler_warmup_steps,
        scheduler_warmup_tokens=scheduler_warmup_tokens,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_identity_hash(
    mapping: Mapping[str, Any], field: str, run_id: str, artifact_path: Path
) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value or value == "pending":
        raise _gradient_reporting_error(
            run_id, artifact_path, f"{field} is unresolved"
        )
    return value


def _record_nonnegative_int(
    value: Any,
    field: str,
    run_id: str,
    journal_path: Path,
    line_number: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _gradient_reporting_error(
            run_id, journal_path, f"{field} is invalid on line {line_number}"
        )
    return value


def _validate_snapshot_identities(
    record: Mapping[str, Any],
    *,
    run_id: str,
    fixed_probe_hash: str,
    controlled_support_hash: str,
    diagnostic_contract_hash: str,
    diagnostic: Mapping[str, Any],
    artifact_path: Path,
    line_number: int,
) -> None:
    expected = {
        "run_id": run_id,
        "fixed_probe_manifest_hash": fixed_probe_hash,
        "controlled_support_hash": controlled_support_hash,
        "diagnostic_contract_hash": diagnostic_contract_hash,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise _gradient_reporting_error(
                run_id, artifact_path, f"{field} mismatch on line {line_number}"
            )
    semantics = record.get("semantics")
    expected_semantics = {
        "gradient": diagnostic.get("gradient_semantics"),
        "loss_aggregation": diagnostic.get("loss_aggregation"),
        "shared_support": diagnostic.get("shared_support"),
        "layerwise": bool(diagnostic.get("layerwise")),
    }
    if semantics != expected_semantics:
        raise _gradient_reporting_error(
            run_id, artifact_path, f"diagnostic semantics mismatch on line {line_number}"
        )


def _compact_gradient_pairs(
    raw_pairs: Any,
    *,
    expected_pairs: tuple[tuple[str, str], ...],
    run_id: str,
    journal_path: Path,
    line_number: int,
) -> tuple[GradientInterferencePair, ...]:
    if not isinstance(raw_pairs, list) or len(raw_pairs) != len(expected_pairs):
        raise _gradient_reporting_error(
            run_id, journal_path, f"pair cardinality mismatch on line {line_number}"
        )
    compact: list[GradientInterferencePair] = []
    for pair_index, (raw_pair, expected_pair) in enumerate(
        zip(raw_pairs, expected_pairs, strict=True)
    ):
        if not isinstance(raw_pair, Mapping) or (
            raw_pair.get("left_granularity"), raw_pair.get("right_granularity")
        ) != expected_pair:
            raise _gradient_reporting_error(
                run_id,
                journal_path,
                f"pair ordering mismatch at index {pair_index} on line {line_number}",
            )
        has_zero_norm = raw_pair.get("has_zero_norm")
        cosine = raw_pair.get("cosine")
        if not isinstance(has_zero_norm, bool):
            raise _gradient_reporting_error(
                run_id, journal_path, f"zero-norm flag is invalid on line {line_number}"
            )
        if cosine is None:
            if not has_zero_norm:
                raise _gradient_reporting_error(
                    run_id, journal_path, f"cosine is null without a zero norm on line {line_number}"
                )
            normalized_cosine = None
        elif (
            isinstance(cosine, bool)
            or not isinstance(cosine, (int, float))
            or not math.isfinite(float(cosine))
            or not -1.0 <= float(cosine) <= 1.0
            or has_zero_norm
        ):
            raise _gradient_reporting_error(
                run_id, journal_path, f"cosine/zero-norm state is invalid on line {line_number}"
            )
        else:
            normalized_cosine = float(cosine)
        compact.append(
            GradientInterferencePair(
                left_granularity=expected_pair[0],
                right_granularity=expected_pair[1],
                cosine=normalized_cosine,
                has_zero_norm=has_zero_norm,
            )
        )
    return tuple(compact)


def _gradient_comparison_contract(
    config: Mapping[str, Any],
    *,
    ordered_granularities: tuple[str, ...],
    diagnostic_contract_hash: str,
    fixed_probe_hash: str,
    controlled_support_hash: str,
) -> str:
    model = dict(_required_mapping(config.get("model"), "config.model"))
    training = dict(_required_mapping(config.get("training"), "config.training"))
    # H is the intervention being compared and therefore cannot enter the
    # grouping contract. Seed and run/output identities live under run and are
    # deliberately absent as well.
    model.pop("global_sampling_interval_steps", None)
    model.pop("global_sampling_state", None)
    granularity_pattern_provenance = model.get("granularity_pattern_provenance")
    if isinstance(granularity_pattern_provenance, Mapping):
        granularity_pattern_provenance = dict(granularity_pattern_provenance)
        granularity_pattern_provenance.pop(
            "global_sampling_interval_steps", None
        )
        model["granularity_pattern_provenance"] = granularity_pattern_provenance
    width_ordering = _gradient_width_ordering(model, ordered_granularities)
    payload = {
        "width_ordering": width_ordering,
        "token_budget": training.get("token_budget"),
        "model_contract": model,
        "training_contract": training,
        "role_hashes": {
            field: config.get(field)
            for field in (
                "data_roles_manifest_hash",
                "optimizer_training_manifest_hash",
                "validation_manifest_hash",
                "controller_manifest_hash",
                "final_holdout_manifest_hash",
            )
        },
        "diagnostic_contract_hash": diagnostic_contract_hash,
        "fixed_probe_manifest_hash": fixed_probe_hash,
        "controlled_support_hash": controlled_support_hash,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _gradient_width_ordering(
    model: Mapping[str, Any], ordered_granularities: tuple[str, ...]
) -> list[list[Any]]:
    widths = model.get("granularity_prefix_widths")
    if not isinstance(widths, Mapping):
        metadata = model.get("ffn_prefix_metadata")
        if isinstance(metadata, list):
            widths = {
                str(item.get("name")): item.get("prefix_width")
                for item in metadata
                if isinstance(item, Mapping)
            }
    if not isinstance(widths, Mapping):
        widths = model.get("granularity_prefixes")
    if not isinstance(widths, Mapping):
        widths = {}
    return [[label, widths.get(label)] for label in ordered_granularities]


def read_csv_artifacts(input_root: Path, filename: str) -> list[dict[str, str]]:
    return read_csv_artifacts_filtered(input_root, filename, row_filter=None)


def read_final_holdout_scaling_rows(
    input_root: str | Path,
) -> list[dict[str, Any]]:
    """Build plot rows from complete, contract-compatible final-holdout results.

    The per-run scaling CSV supplies only model metadata and parameter counts;
    loss and perplexity always come from the separately sealed holdout artifact.
    A completed run that opted into the post-training holdout must have a valid
    result before any rows are returned, preventing partial comparison figures.
    """

    from .final_holdout import (
        FINAL_HOLDOUT_AGGREGATION,
        FinalHoldoutError,
        resolve_existing_final_holdout_result,
    )

    root = Path(input_root)
    expected_runs: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for config_path in sorted(root.rglob("config.json")):
        run_dir = config_path.parent
        config = _read_json_mapping(config_path, artifact_name="saved run config")
        if not _post_training_final_holdout_enabled(config):
            continue

        summary_path = run_dir / "run_summary.json"
        if not summary_path.is_file():
            continue
        summary = _read_json_mapping(summary_path, artifact_name="run summary")
        if summary.get("status") != "completed":
            if (run_dir / "final_holdout_results.json").is_file():
                raise FinalHoldoutReportingError(
                    f"Final-holdout result exists for an incomplete run: {run_dir}"
                )
            continue
        expected_runs.append((run_dir, config, summary))

    if not expected_runs:
        return []

    missing_results = [
        run_dir
        for run_dir, _, _ in expected_runs
        if not (run_dir / "final_holdout_results.json").is_file()
    ]
    if missing_results:
        formatted = "\n".join(f"- {path}" for path in missing_results)
        raise FinalHoldoutReportingError(
            "Final-holdout reporting requires results for every completed, "
            "holdout-enabled run. Missing:\n"
            f"{formatted}"
        )

    contracts: dict[str, list[Path]] = {}
    contract_payloads: dict[str, dict[str, Any]] = {}
    for run_dir, config, _ in expected_runs:
        payload = _final_holdout_comparison_contract(config, run_dir=run_dir)
        key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        contracts.setdefault(key, []).append(run_dir)
        contract_payloads[key] = payload
    if len(contracts) != 1:
        summaries = []
        for key, run_dirs in contracts.items():
            summaries.append(
                f"- runs={','.join(path.name for path in run_dirs)} "
                f"contract={json.dumps(contract_payloads[key], sort_keys=True)}"
            )
        raise FinalHoldoutReportingError(
            "Final-holdout comparison mixes corpus, holdout, budget, seed, or "
            "granularity contracts:\n" + "\n".join(summaries)
        )

    comparison_contract = next(iter(contract_payloads.values()))
    nested_orders: dict[tuple[str, ...], list[Path]] = {}
    for run_dir, config, _ in expected_runs:
        run = config["run"]
        if (
            run.get("model_family") == "standalone"
            or run.get("sampling_mode") == "standalone"
        ):
            continue
        order = tuple(str(value) for value in config["model"]["granularities"])
        nested_orders.setdefault(order, []).append(run_dir)
    if len(nested_orders) > 1:
        summaries = [
            f"- granularities={list(order)} "
            f"runs={','.join(path.name for path in run_dirs)}"
            for order, run_dirs in nested_orders.items()
        ]
        raise FinalHoldoutReportingError(
            "Final-holdout comparison mixes nested granularity grids:\n"
            + "\n".join(summaries)
        )
    nested_universe = set(next(iter(nested_orders))) if nested_orders else set()
    if nested_universe:
        for run_dir, config, _ in expected_runs:
            run = config["run"]
            if (
                run.get("model_family") != "standalone"
                and run.get("sampling_mode") != "standalone"
            ):
                continue
            standalone_granularities = {
                str(value) for value in config["model"]["granularities"]
            }
            if not standalone_granularities.issubset(nested_universe):
                raise FinalHoldoutReportingError(
                    "Standalone granularity is outside the nested comparison grid: "
                    f"{run_dir}"
                )

    rows: list[dict[str, Any]] = []
    for run_dir, config, summary in expected_runs:
        run_id = _saved_run_id(config, run_dir=run_dir)
        if summary.get("run_id") != run_id:
            raise FinalHoldoutReportingError(
                f"Run ID mismatch between config and summary: {run_dir}"
            )
        try:
            result = resolve_existing_final_holdout_result(run_dir)
        except FinalHoldoutError as error:
            raise FinalHoldoutReportingError(
                f"Invalid final-holdout result for {run_dir}: {error}"
            ) from error
        if result is None:
            raise FinalHoldoutReportingError(
                f"Missing final-holdout result for completed run: {run_dir}"
            )

        expected_evaluation_examples = _final_holdout_evaluation_example_count(
            run_dir,
            expected_manifest_hash=comparison_contract[
                "final_holdout_manifest_hash"
            ],
            expected_source_examples=comparison_contract["final_holdout_examples"],
        )
        components = _validate_final_holdout_reporting_result(
            result,
            run_dir=run_dir,
            expected_run_id=run_id,
            expected_contract=comparison_contract,
            expected_evaluation_examples=expected_evaluation_examples,
            expected_granularities=[
                str(value) for value in config["model"]["granularities"]
            ],
            expected_aggregation=FINAL_HOLDOUT_AGGREGATION,
        )
        scaling_path = run_dir / "scaling_results.csv"
        if not scaling_path.is_file():
            raise FinalHoldoutReportingError(
                f"Missing scaling metadata for final-holdout run: {scaling_path}"
            )
        scaling_by_granularity: dict[str, dict[str, Any]] = {}
        for scaling_row in iter_csv_artifact_rows(scaling_path):
            granularity = str(scaling_row.get("granularity") or "")
            if not granularity or granularity in scaling_by_granularity:
                raise FinalHoldoutReportingError(
                    "Scaling metadata must contain one row per granularity: "
                    f"{scaling_path}"
                )
            if scaling_row.get("run_id") != run_id:
                raise FinalHoldoutReportingError(
                    f"Scaling metadata run ID mismatch: {scaling_path}"
                )
            scaling_by_granularity[granularity] = scaling_row

        ordered_granularities = [row["granularity"] for row in components]
        if set(scaling_by_granularity) != set(ordered_granularities):
            raise FinalHoldoutReportingError(
                "Scaling metadata granularities do not match the final holdout "
                f"for {run_dir}"
            )
        result_path = run_dir / "final_holdout_results.json"
        for component in components:
            granularity = component["granularity"]
            row = dict(scaling_by_granularity[granularity])
            row.update(
                {
                    "comparison_id": f"{run_id}__{granularity}__final_holdout",
                    "loss": component["loss"],
                    "perplexity": component["perplexity"],
                    "evaluation_examples": component["evaluation_examples"],
                    "evaluation_target_tokens": component["evaluation_target_tokens"],
                    "evaluation_split": "final_holdout",
                    "metric_source": "final_holdout_results.json",
                    "final_holdout_manifest_hash": result[
                        "final_holdout_manifest_hash"
                    ],
                    "_source_json": str(result_path),
                }
            )
            rows.append(row)

    return enrich_scaling_metadata_from_run_config(root, rows)


def _read_json_mapping(path: Path, *, artifact_name: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as artifact_file:
            value = json.load(artifact_file)
    except (OSError, json.JSONDecodeError) as error:
        raise FinalHoldoutReportingError(
            f"Could not read {artifact_name}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise FinalHoldoutReportingError(
            f"{artifact_name.capitalize()} must be a mapping: {path}"
        )
    return value


def _post_training_final_holdout_enabled(config: Mapping[str, Any]) -> bool:
    evaluation = config.get("evaluation")
    final_holdout = (
        evaluation.get("final_holdout") if isinstance(evaluation, Mapping) else None
    )
    return bool(
        isinstance(final_holdout, Mapping)
        and final_holdout.get("enabled") is True
        and final_holdout.get("evaluate_during_training") is False
    )


def _saved_run_id(config: Mapping[str, Any], *, run_dir: Path) -> str:
    run = config.get("run")
    run_id = run.get("run_id") if isinstance(run, Mapping) else None
    if run_id in (None, ""):
        raise FinalHoldoutReportingError(f"Saved config has no run ID: {run_dir}")
    return str(run_id)


def _final_holdout_evaluation_example_count(
    run_dir: Path,
    *,
    expected_manifest_hash: str,
    expected_source_examples: int,
) -> int:
    """Resolve evaluated rows without confusing packed sequences with sources."""

    manifest_path = run_dir / "final_holdout_manifest.json"
    if not manifest_path.is_file():
        raise FinalHoldoutReportingError(
            f"Missing final-holdout manifest: {manifest_path}"
        )
    manifest = _read_json_mapping(
        manifest_path,
        artifact_name="final-holdout manifest",
    )
    try:
        source_examples = _csv_positive_int(
            manifest.get("example_count"),
            "final holdout manifest example_count",
        )
        evaluation_examples = _csv_positive_int(
            manifest.get("sequence_count", source_examples),
            "final holdout manifest sequence_count",
        )
    except ValueError as error:
        raise FinalHoldoutReportingError(f"{error}: {run_dir}") from error
    if (
        manifest.get("role") != "final_holdout"
        or manifest.get("manifest_hash") != expected_manifest_hash
        or source_examples != expected_source_examples
    ):
        raise FinalHoldoutReportingError(
            f"Final-holdout manifest contract mismatch: {run_dir}"
        )
    source_document_count = manifest.get("source_document_count")
    if source_document_count is not None:
        try:
            source_document_count = _csv_positive_int(
                source_document_count,
                "final holdout manifest source_document_count",
            )
        except ValueError as error:
            raise FinalHoldoutReportingError(f"{error}: {run_dir}") from error
        if source_document_count != source_examples:
            raise FinalHoldoutReportingError(
                f"Final-holdout manifest source count mismatch: {run_dir}"
            )
    return evaluation_examples


def _final_holdout_comparison_contract(
    config: Mapping[str, Any],
    *,
    run_dir: Path,
) -> dict[str, Any]:
    run = config.get("run")
    model = config.get("model")
    training = config.get("training")
    evaluation = config.get("evaluation")
    if not all(
        isinstance(value, Mapping) for value in (run, model, training, evaluation)
    ):
        raise FinalHoldoutReportingError(
            f"Saved config lacks run/model/training/evaluation mappings: {run_dir}"
        )
    final_holdout = evaluation.get("final_holdout")
    granularities = model.get("granularities")
    if (
        not isinstance(final_holdout, Mapping)
        or final_holdout.get("enabled") is not True
        or final_holdout.get("evaluate_during_training") is not False
        or final_holdout.get("fixed_manifest") is not True
    ):
        raise FinalHoldoutReportingError(
            f"Saved final-holdout contract is incompatible: {run_dir}"
        )
    if (
        not isinstance(granularities, list)
        or not granularities
        or len({str(value) for value in granularities}) != len(granularities)
    ):
        raise FinalHoldoutReportingError(
            f"Saved granularity order is invalid: {run_dir}"
        )

    ordinary_validation_hash = config.get("ordinary_validation_manifest_hash")
    if ordinary_validation_hash in (None, ""):
        ordinary_validation_hash = config.get("validation_manifest_hash")
    payload = {
        "data_roles_manifest_hash": config.get("data_roles_manifest_hash"),
        "optimizer_training_manifest_hash": config.get(
            "optimizer_training_manifest_hash"
        ),
        "ordinary_validation_manifest_hash": ordinary_validation_hash,
        "controller_manifest_hash": config.get("controller_manifest_hash"),
        "final_holdout_manifest_hash": config.get("final_holdout_manifest_hash"),
        "final_holdout_examples": final_holdout.get("examples"),
        "token_budget": training.get("token_budget"),
        "expected_tokens_per_step": training.get("expected_tokens_per_step"),
        "seed": run.get("seed"),
    }
    missing = [key for key, value in payload.items() if value in (None, "", [])]
    if missing:
        raise FinalHoldoutReportingError(
            f"Saved comparison contract is incomplete for {run_dir}: "
            f"{', '.join(missing)}"
        )
    if final_holdout.get("manifest_hash") not in (
        None,
        payload["final_holdout_manifest_hash"],
    ):
        raise FinalHoldoutReportingError(
            f"Final-holdout manifest hash disagrees with saved config: {run_dir}"
        )
    return payload


def _validate_final_holdout_reporting_result(
    result: Mapping[str, Any],
    *,
    run_dir: Path,
    expected_run_id: str,
    expected_contract: Mapping[str, Any],
    expected_evaluation_examples: int,
    expected_granularities: list[str],
    expected_aggregation: str,
) -> list[dict[str, Any]]:
    ordered_granularities = result.get("ordered_granularities")
    components = result.get("ordered_per_granularity_losses")
    if (
        result.get("run_id") != expected_run_id
        or ordered_granularities != expected_granularities
        or not isinstance(components, list)
        or len(components) != len(expected_granularities)
    ):
        raise FinalHoldoutReportingError(
            f"Final-holdout run ID or granularity order mismatch: {run_dir}"
        )
    if (
        result.get("final_holdout_manifest_hash")
        != expected_contract["final_holdout_manifest_hash"]
        or result.get("aggregation_method") != expected_aggregation
        or result.get("evaluation_example_count")
        != expected_contract["final_holdout_examples"]
    ):
        raise FinalHoldoutReportingError(
            f"Final-holdout evaluation contract mismatch: {run_dir}"
        )
    selection = result.get("checkpoint_selection_provenance")
    if not isinstance(selection, Mapping) or selection.get("source") != (
        "ordinary_validation"
    ):
        raise FinalHoldoutReportingError(
            "Final reported comparisons require an ordinary-validation-selected "
            f"checkpoint: {run_dir}"
        )

    validated: list[dict[str, Any]] = []
    for expected_granularity, component in zip(
        expected_granularities, components, strict=True
    ):
        if not isinstance(component, Mapping) or component.get("granularity") != (
            expected_granularity
        ):
            raise FinalHoldoutReportingError(
                f"Final-holdout component order mismatch: {run_dir}"
            )
        try:
            loss = _finite_float(component.get("loss"), "final holdout loss")
            perplexity = _finite_float(
                component.get("perplexity"), "final holdout perplexity"
            )
            evaluation_examples = _csv_positive_int(
                component.get("evaluation_examples"),
                "final holdout evaluation_examples",
            )
            evaluation_target_tokens = _csv_positive_int(
                component.get("evaluation_target_tokens"),
                "final holdout evaluation_target_tokens",
            )
        except ValueError as error:
            raise FinalHoldoutReportingError(f"{error}: {run_dir}") from error
        try:
            expected_perplexity = math.exp(loss)
        except OverflowError as error:
            raise FinalHoldoutReportingError(
                f"Final-holdout loss cannot produce finite perplexity: {run_dir}"
            ) from error
        if not math.isclose(
            perplexity,
            expected_perplexity,
            rel_tol=1e-6,
            abs_tol=1e-9,
        ):
            raise FinalHoldoutReportingError(
                f"Final-holdout loss/perplexity mismatch: {run_dir}"
            )
        if evaluation_examples != expected_evaluation_examples:
            raise FinalHoldoutReportingError(
                f"Final-holdout component example count mismatch: {run_dir}"
            )
        validated.append(
            {
                "granularity": expected_granularity,
                "loss": loss,
                "perplexity": perplexity,
                "evaluation_examples": evaluation_examples,
                "evaluation_target_tokens": evaluation_target_tokens,
            }
        )

    uniform_average_loss = _finite_float(
        result.get("uniform_average_loss"), "uniform_average_loss"
    )
    expected_average = math.fsum(row["loss"] for row in validated) / len(validated)
    if not math.isclose(
        uniform_average_loss,
        expected_average,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise FinalHoldoutReportingError(
            f"Final-holdout uniform average mismatch: {run_dir}"
        )
    return validated


def iter_csv_artifact_rows(path: str | Path) -> Iterator[dict[str, str]]:
    """Yield one CSV row at a time without retaining the artifact in memory."""

    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            row["_source_csv"] = str(path)
            yield row


def iter_global_sampling_histories(
    input_root: str | Path,
) -> Iterator[GlobalSamplingHistory]:
    """Stream canonical per-step histories for comparable global policies."""

    input_root = Path(input_root)
    for metrics_path in sorted(input_root.rglob("metrics.csv")):
        config_path = metrics_path.parent / "config.json"
        if not config_path.is_file():
            continue
        try:
            with config_path.open("r", encoding="utf-8") as config_file:
                config = json.load(config_file)
            history = _read_global_sampling_history(
                config=config,
                config_path=config_path,
                metrics_path=metrics_path,
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            warnings.warn(
                f"Skipping global sampling history {metrics_path}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if history is not None and history.actions:
            yield history


def _read_global_sampling_history(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    metrics_path: Path,
) -> GlobalSamplingHistory | None:
    model = _required_mapping(config.get("model"), "config.model")
    run = _required_mapping(config.get("run"), "config.run")
    training = _required_mapping(config.get("training"), "config.training")
    if str(run.get("sampling_mode") or "").strip().lower() != "nested-random":
        return None

    granularities = model.get("granularities")
    if not isinstance(granularities, list) or not granularities:
        raise ValueError("config.model.granularities must be a nonempty list")
    ordered_granularities = tuple(str(label) for label in granularities)
    if len(set(ordered_granularities)) != len(ordered_granularities):
        raise ValueError("config.model.granularities contains duplicates")

    sampling_mode = str(model.get("granularity_sampling_mode") or "global").lower()
    strategy = str(model.get("adaptive_sampler_strategy") or "").lower()
    fixed_distribution: dict[str, float] | None = None
    uniform_interval = 1
    global_sampling_schedule: str | None = None
    global_sampling_schedule_version: int | None = None
    if sampling_mode == "global":
        uniform_interval = _positive_int(
            model.get("global_sampling_interval_steps", 1),
            "config.model.global_sampling_interval_steps",
        )
        global_sampling_schedule = global_sampling_schedule_from_saved_config(
            dict(config)
        )
        if global_sampling_schedule is None:
            raise ValueError("config.model.global_sampling_schedule is invalid")
        global_sampling_schedule_version = (
            global_sampling_schedule_version_from_saved_config(dict(config))
        )
        if global_sampling_schedule == "balanced_cycle":
            if global_sampling_schedule_version is None:
                raise ValueError(
                    "balanced global history requires a positive schedule version"
                )
            policy_identity = f"balanced_global_h{uniform_interval}"
            policy_label = f"Balanced global (H={uniform_interval})"
        elif uniform_interval == 1:
            policy_identity, policy_label = "uniform_global", "Uniform global"
        else:
            policy_identity = f"uniform_global_h{uniform_interval}"
            policy_label = f"Uniform global (H={uniform_interval})"
    elif sampling_mode == "fixed_global":
        fixed_distribution = _fixed_global_distribution(
            model, ordered_granularities
        )
        distribution_json = json.dumps(
            fixed_distribution, sort_keys=True, separators=(",", ":")
        )
        distribution_id = hashlib.sha256(distribution_json.encode()).hexdigest()[:8]
        policy_identity = f"fixed_global_{distribution_id}"
        policy_label = _fixed_global_policy_label(
            fixed_distribution,
            ordered_granularities,
        )
    elif sampling_mode == "adaptive_global" and strategy == "panelgrad":
        panelgrad = model.get("panelgrad")
        if not isinstance(panelgrad, Mapping) or str(
            panelgrad.get("method_family") or ""
        ).lower() not in PANELGRAD_METHOD_FAMILIES:
            return None
        importance_metric = _panelgrad_importance_metric(panelgrad)
        policy_identity = f"panelgrad_global_{importance_metric}"
        policy_label = _panelgrad_policy_label(panelgrad)
    elif sampling_mode == "adaptive_global" and strategy == "thompson":
        controller = model.get("adaptive_controller")
        if (
            not isinstance(controller, Mapping)
            or str(controller.get("method_family") or "").lower()
            != BAYESIAN_CONTROLLER_METHOD_FAMILY
            or str(controller.get("scope") or "").lower() != "global"
        ):
            return None
        policy_identity, policy_label = "thompson_global", "Thompson global"
    else:
        return None

    run_id = str(run.get("run_id") or config_path.parent.name).strip()
    if not run_id:
        raise ValueError("config.run.run_id must be nonempty")
    token_budget = _positive_int(
        training.get("token_budget"), "config.training.token_budget"
    )
    thompson_interval = None
    if policy_identity == "thompson_global":
        thompson_interval = _positive_int(
            model["adaptive_controller"].get("decision_interval_steps"),
            "config.model.adaptive_controller.decision_interval_steps",
        )

    actions: list[GlobalSamplingAction] = []
    previous_step = 0
    previous_tokens = 0
    decision_indices: set[int] = set()
    for row in iter_csv_artifact_rows(metrics_path):
        if str(row.get("split") or "") != "train":
            continue
        step = _csv_positive_int(row.get("step"), "metrics.csv step")
        if step != previous_step + 1:
            raise ValueError(
                "global training rows must contain exactly one contiguous row per step"
            )
        end_tokens = _csv_nonnegative_int(
            row.get("tokens_seen"), f"metrics.csv step {step} tokens_seen"
        )
        if end_tokens <= previous_tokens:
            raise ValueError("global training tokens must increase at every step")
        trained = str(row.get("granularity") or "").strip()
        recorded_action = str(row.get("controller_action") or "").strip()
        if (
            policy_identity.startswith("panelgrad_global_")
            and recorded_action
            and recorded_action != trained
        ):
            raise ValueError(
                f"metrics.csv step {step} controller action disagrees with granularity"
            )
        # `granularity` is the action that actually ran. Thompson updates its
        # compact controller state at a boundary before writing that step's
        # metrics, so `controller_action` can already name the next window.
        action = trained
        if action not in ordered_granularities:
            raise ValueError(
                f"metrics.csv step {step} has unknown global action {action!r}"
            )

        if policy_identity == "thompson_global":
            decision_index = (step - 1) // int(thompson_interval)
        elif sampling_mode == "global":
            decision_index = (step - 1) // int(uniform_interval)
        else:
            decision_index = step - 1
        decision_indices.add(decision_index)
        if sampling_mode == "global":
            recorded_window = row.get("global_sampling_window_index")
            if recorded_window not in (None, "") and _csv_nonnegative_int(
                recorded_window,
                f"metrics.csv step {step} global sampling window index",
            ) != decision_index:
                raise ValueError(
                    f"metrics.csv step {step} has inconsistent global sampling "
                    "window index"
                )
            recorded_progress = row.get("global_sampling_window_progress")
            expected_progress = ((step - 1) % int(uniform_interval)) + 1
            if recorded_progress not in (None, "") and _csv_positive_int(
                recorded_progress,
                f"metrics.csv step {step} global sampling window progress",
            ) != expected_progress:
                raise ValueError(
                    f"metrics.csv step {step} has inconsistent global sampling "
                    "window progress"
                )
        probability_raw = row.get("controller_sampled_probability")
        sampled_probability = None
        if probability_raw not in (None, ""):
            sampled_probability = _finite_float(
                probability_raw,
                f"metrics.csv step {step} sampled probability",
            )
            if not 0.0 <= sampled_probability <= 1.0:
                raise ValueError("sampled probability must be in [0, 1]")
        if fixed_distribution is not None:
            expected_probability = fixed_distribution[action]
            if sampled_probability is not None and not math.isclose(
                sampled_probability,
                expected_probability,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"metrics.csv step {step} sampled probability disagrees "
                    "with config.model.global_sampling_distribution"
                )
            sampled_probability = expected_probability
        actions.append(
            GlobalSamplingAction(
                step=step,
                start_tokens=previous_tokens,
                end_tokens=min(end_tokens, token_budget),
                granularity=action,
                decision_index=decision_index,
                sampled_probability=sampled_probability,
            )
        )
        previous_step = step
        previous_tokens = min(end_tokens, token_budget)

    comparison_payload = {
        "variant": model_variant_from_saved_config(dict(config)),
        "correction_mode": correction_mode_from_saved_config(dict(config)),
        "membership_correction": membership_correction_from_saved_config(dict(config)),
        "granularities": list(ordered_granularities),
        "context_length": model.get("context_length"),
        "token_budget": token_budget,
        "expected_tokens_per_step": training.get("expected_tokens_per_step"),
        "optimizer": training.get("optimizer"),
        "resolved_learning_rate": training.get("resolved_learning_rate"),
        "scheduler": training.get("scheduler"),
        "batch_size_per_process": training.get("batch_size_per_process"),
        "gradient_accumulation_steps": int(
            training.get("gradient_accumulation_steps") or 1
        ),
    }
    comparison_key = hashlib.sha256(
        json.dumps(comparison_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    seed_raw = run.get("seed")
    seed = None if seed_raw in (None, "") else int(seed_raw)
    scheduler_warmup_steps, scheduler_warmup_tokens = (
        _scheduler_warmup_boundaries(config, config_path=config_path)
    )
    return GlobalSamplingHistory(
        run_id=run_id,
        policy_identity=policy_identity,
        policy_label=policy_label,
        ordered_granularities=ordered_granularities,
        token_budget=token_budget,
        actions=tuple(actions),
        decision_count=len(decision_indices),
        comparison_key=comparison_key,
        model_variant=model_variant_from_saved_config(dict(config)),
        correction_mode=correction_mode_from_saved_config(dict(config)),
        membership_correction=membership_correction_from_saved_config(dict(config)),
        seed=seed,
        config_path=config_path,
        metrics_path=metrics_path,
        target_probabilities=(
            tuple(fixed_distribution[label] for label in ordered_granularities)
            if fixed_distribution is not None
            else None
        ),
        global_sampling_schedule=global_sampling_schedule,
        global_sampling_schedule_version=global_sampling_schedule_version,
        scheduler_warmup_steps=scheduler_warmup_steps,
        scheduler_warmup_tokens=scheduler_warmup_tokens,
    )


def _fixed_global_distribution(
    model: Mapping[str, Any],
    ordered_granularities: tuple[str, ...],
) -> dict[str, float]:
    raw = model.get("global_sampling_distribution")
    if not isinstance(raw, Mapping):
        raise ValueError(
            "fixed_global config requires model.global_sampling_distribution"
        )
    if {str(label) for label in raw} != set(ordered_granularities):
        raise ValueError(
            "fixed_global distribution keys must match config.model.granularities"
        )
    distribution = {
        label: _finite_float(
            raw[label], f"config.model.global_sampling_distribution.{label}"
        )
        for label in ordered_granularities
    }
    if any(probability < 0.0 for probability in distribution.values()):
        raise ValueError("fixed_global probabilities must be nonnegative")
    if not math.isclose(
        math.fsum(distribution.values()), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("fixed_global probabilities must sum to 1")
    return distribution


def _fixed_global_policy_label(
    distribution: Mapping[str, float],
    ordered_granularities: tuple[str, ...],
) -> str:
    """Build a compact legend label while retaining exact values in artifacts."""

    arm_count = len(ordered_granularities)
    inverse_weights = [1.0 / (arm_count - index) for index in range(arm_count)]
    normalizer = math.fsum(inverse_weights)
    inverse_membership = {
        label: weight / normalizer
        for label, weight in zip(ordered_granularities, inverse_weights)
    }
    if all(
        math.isclose(
            float(distribution[label]),
            inverse_membership[label],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for label in ordered_granularities
    ):
        return "Fixed global (inverse-membership)"

    rounded = ", ".join(
        f"{label}={100.0 * float(distribution[label]):.1f}%"
        for label in ordered_granularities
    )
    return f"Fixed global ({rounded})"


def global_sampling_history_as_timeline(
    history: GlobalSamplingHistory,
) -> ControllerGranularityTimeline:
    """Adapt canonical committed actions to the categorical timeline renderer."""

    return ControllerGranularityTimeline(
        run_id=history.run_id,
        scope="global",
        ordered_granularities=history.ordered_granularities,
        block_count=1,
        row_labels=("all blocks",),
        token_budget=history.token_budget,
        windows=tuple(
            ControllerSelectionWindow(
                window_index=action.step - 1,
                start_step=action.step,
                end_step=action.step,
                start_tokens=action.start_tokens,
                end_tokens=action.end_tokens,
                block_granularities=(action.granularity,),
                terminal_incomplete=False,
            )
            for action in history.actions
        ),
        model_variant=history.model_variant,
        correction_mode=history.correction_mode,
        membership_correction=history.membership_correction,
        config_path=history.config_path,
        journal_path=history.metrics_path,
        scheduler_warmup_steps=history.scheduler_warmup_steps,
        scheduler_warmup_tokens=history.scheduler_warmup_tokens,
    )


def read_csv_artifacts_filtered(
    input_root: Path,
    filename: str,
    row_filter: Any | None,
) -> list[dict[str, str]]:
    rows = []
    for path in sorted(input_root.rglob(filename)):
        for row in iter_csv_artifact_rows(path):
            if row_filter is not None and not row_filter(row):
                continue
            rows.append(row)
    return rows


def iter_controller_granularity_timelines(
    input_root: str | Path,
) -> Iterator[ControllerGranularityTimeline]:
    """Stream Bayesian journals into compact, plot-ready selection timelines.

    Each JSONL object is discarded immediately after its action and boundary
    fields are normalized. This keeps the large posterior covariance records
    out of the retained reporting representation.
    """

    input_root = Path(input_root)
    for journal_path in sorted(input_root.rglob("controller_metrics.jsonl")):
        config_path = journal_path.parent / "config.json"
        if not config_path.is_file():
            warnings.warn(
                f"Skipping controller timeline {journal_path}: config.json is missing",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        try:
            with config_path.open("r", encoding="utf-8") as config_file:
                config = json.load(config_file)
            timeline = _read_controller_granularity_timeline(
                config=config,
                config_path=config_path,
                journal_path=journal_path,
            )
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ) as error:
            warnings.warn(
                f"Skipping controller timeline {journal_path}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if timeline is not None and timeline.windows:
            yield timeline


def iter_panelgrad_histories(
    input_root: str | Path,
) -> Iterator[PanelGradHistory]:
    """Stream PanelGrad action rows and refresh events into compact histories."""

    input_root = Path(input_root)
    for journal_path in sorted(input_root.rglob("controller_metrics.jsonl")):
        config_path = journal_path.parent / "config.json"
        metrics_path = journal_path.parent / "metrics.csv"
        if not config_path.is_file():
            warnings.warn(
                f"Skipping PanelGrad history {journal_path}: config.json is missing",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        try:
            with config_path.open("r", encoding="utf-8") as config_file:
                config = json.load(config_file)
            history = _read_panelgrad_history(
                config=config,
                config_path=config_path,
                metrics_path=metrics_path,
                journal_path=journal_path,
            )
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ) as error:
            warnings.warn(
                f"Skipping PanelGrad history {journal_path}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if history is not None and (history.actions or history.refreshes):
            yield history


def panelgrad_history_as_timeline(
    history: PanelGradHistory,
) -> ControllerGranularityTimeline:
    """Adapt committed PanelGrad actions to the shared global timeline plots."""

    windows = tuple(
        ControllerSelectionWindow(
            window_index=action.step,
            start_step=action.step - 1,
            end_step=action.step,
            start_tokens=action.start_tokens,
            end_tokens=action.end_tokens,
            block_granularities=(action.granularity,),
            terminal_incomplete=False,
        )
        for action in history.actions
    )
    return ControllerGranularityTimeline(
        run_id=history.run_id,
        scope="global",
        ordered_granularities=history.ordered_granularities,
        block_count=1,
        row_labels=("all blocks",),
        token_budget=history.token_budget,
        windows=windows,
        model_variant=history.model_variant,
        correction_mode=history.correction_mode,
        membership_correction=history.membership_correction,
        config_path=history.config_path,
        journal_path=history.journal_path,
        scheduler_warmup_steps=history.scheduler_warmup_steps,
        scheduler_warmup_tokens=history.scheduler_warmup_tokens,
    )


def _read_panelgrad_history(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    metrics_path: Path,
    journal_path: Path,
) -> PanelGradHistory | None:
    model = _required_mapping(config.get("model"), "config.model")
    policy = model.get("panelgrad")
    if not isinstance(policy, Mapping):
        return None
    method_family = str(policy.get("method_family") or "").strip().lower()
    strategy = str(model.get("adaptive_sampler_strategy") or "").strip().lower()
    scope = str(policy.get("scope") or "").strip().lower()
    if not (
        method_family in PANELGRAD_METHOD_FAMILIES
        and policy.get("method_version") not in (None, "")
        and strategy == "panelgrad"
        and scope == "global"
    ):
        return None
    importance_metric = _panelgrad_importance_metric(policy)

    run = _required_mapping(config.get("run"), "config.run")
    run_id = str(run.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("config.run.run_id must be nonempty")
    training = _required_mapping(config.get("training"), "config.training")
    expected_tokens_per_step = _positive_int(
        training.get("expected_tokens_per_step"),
        "config.training.expected_tokens_per_step",
    )
    token_budget = _positive_int(
        training.get("token_budget"),
        "config.training.token_budget",
    )
    granularities = policy.get("ordered_granularities")
    if granularities in (None, ""):
        granularities = model.get("granularities")
    if not isinstance(granularities, list) or not granularities:
        raise ValueError("config.model.panelgrad.ordered_granularities must be nonempty")
    ordered_granularities = tuple(str(label) for label in granularities)
    if any(not label for label in ordered_granularities):
        raise ValueError("PanelGrad granularities contain an empty label")
    if len(set(ordered_granularities)) != len(ordered_granularities):
        raise ValueError("PanelGrad granularities contain duplicates")

    if not metrics_path.is_file():
        raise ValueError("metrics.csv is missing")
    actions: list[PanelGradAction] = []
    previous_training_tokens = 0
    previous_step = 0
    for row in iter_csv_artifact_rows(metrics_path):
        if str(row.get("split") or "") != "train":
            continue
        step = _csv_positive_int(row.get("step"), "metrics.csv step")
        end_tokens = _csv_nonnegative_int(
            row.get("tokens_seen"),
            f"metrics.csv step {step} tokens_seen",
        )
        if step <= previous_step or end_tokens < previous_training_tokens:
            raise ValueError("PanelGrad training rows are not strictly ordered")
        sampled_probability_raw = row.get("controller_sampled_probability")
        if sampled_probability_raw not in (None, ""):
            granularity = str(
                row.get("controller_action") or row.get("granularity") or ""
            ).strip()
            if granularity not in ordered_granularities:
                raise ValueError(
                    f"metrics.csv step {step} has unknown PanelGrad action {granularity!r}"
                )
            sampled_probability = _finite_float(
                sampled_probability_raw,
                f"metrics.csv step {step} sampled probability",
            )
            if sampled_probability < 0.0 or sampled_probability > 1.0:
                raise ValueError("PanelGrad sampled probability must be in [0, 1]")
            actions.append(
                PanelGradAction(
                    step=step,
                    start_tokens=previous_training_tokens,
                    end_tokens=min(end_tokens, token_budget),
                    granularity=granularity,
                    sampled_probability=sampled_probability,
                )
            )
        previous_step = step
        previous_training_tokens = min(end_tokens, token_budget)

    refreshes: list[PanelGradRefresh] = []
    epsilon_schedule_identity = _panelgrad_epsilon_schedule_identity(policy)
    with journal_path.open("r", encoding="utf-8") as journal_file:
        for line_number, line in enumerate(journal_file, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, Mapping):
                raise ValueError(f"journal line {line_number} is not a JSON object")
            if event.get("event_type") != "panelgrad_refresh_completed":
                continue
            refreshes.append(
                _normalize_panelgrad_refresh(
                    event,
                    line_number=line_number,
                    ordered_granularities=ordered_granularities,
                    expected_tokens_per_step=expected_tokens_per_step,
                    token_budget=token_budget,
                    fallback_epsilon=float(epsilon_schedule_identity["start"]),
                    importance_metric=importance_metric,
                )
            )
    refreshes.sort(key=lambda refresh: (refresh.boundary_step, refresh.refresh_index))
    scheduler_warmup_steps, scheduler_warmup_tokens = (
        _scheduler_warmup_boundaries(config, config_path=config_path)
    )

    return PanelGradHistory(
        run_id=run_id,
        importance_metric=importance_metric,
        ordered_granularities=ordered_granularities,
        token_budget=token_budget,
        actions=tuple(actions),
        refreshes=tuple(refreshes),
        model_variant=model_variant_from_saved_config(dict(config)),
        correction_mode=correction_mode_from_saved_config(dict(config)),
        membership_correction=membership_correction_from_saved_config(dict(config)),
        config_path=config_path,
        metrics_path=metrics_path,
        journal_path=journal_path,
        scheduler_warmup_steps=scheduler_warmup_steps,
        scheduler_warmup_tokens=scheduler_warmup_tokens,
    )


def _normalize_panelgrad_refresh(
    event: Mapping[str, Any],
    *,
    line_number: int,
    ordered_granularities: tuple[str, ...],
    expected_tokens_per_step: int,
    token_budget: int,
    fallback_epsilon: float,
    importance_metric: str,
) -> PanelGradRefresh:
    prefix = f"journal line {line_number}"
    refresh_index = _nonnegative_int(event.get("window_index"), f"{prefix} window_index")
    boundary_step = _nonnegative_int(event.get("boundary_step"), f"{prefix} boundary_step")
    p = _probability_vector(event.get("p"), f"{prefix} p", len(ordered_granularities))
    q = _probability_vector(event.get("q"), f"{prefix} q", len(ordered_granularities))
    measurements = event.get("measurements")
    if not isinstance(measurements, list):
        raise ValueError(f"{prefix} measurements must be a list")
    score_by_granularity = {}
    rms_by_granularity = {}
    event_importance_metric = event.get("importance_metric", importance_metric)
    if event_importance_metric != importance_metric:
        raise ValueError(f"{prefix} importance metric does not match config")
    measurement_field = {
        "gradient_rms": "gradient_rms_score",
        "gradient_l2": "gradient_norm",
    }[importance_metric]
    for measurement in measurements:
        if not isinstance(measurement, Mapping):
            raise ValueError(f"{prefix} measurement must be a mapping")
        label = str(measurement.get("granularity") or "")
        if label in score_by_granularity:
            raise ValueError(f"{prefix} repeats measurement for {label!r}")
        score_by_granularity[label] = _finite_float(
            measurement.get(measurement_field),
            f"{prefix} {label} {measurement_field}",
        )
        rms_by_granularity[label] = _finite_float(
            measurement.get("gradient_rms_score"),
            f"{prefix} {label} gradient_rms_score",
        )
    if set(score_by_granularity) != set(ordered_granularities):
        raise ValueError(f"{prefix} measurements do not cover every granularity")
    importance_scores = tuple(
        score_by_granularity[label] for label in ordered_granularities
    )
    recorded_importance_scores = event.get("importance_scores")
    if recorded_importance_scores is not None:
        normalized_scores = tuple(
            _finite_float(value, f"{prefix} importance_scores")
            for value in recorded_importance_scores
        )
        if normalized_scores != importance_scores:
            raise ValueError(f"{prefix} importance_scores disagree with measurements")
    duration_seconds = _finite_float(
        event.get("duration_seconds"), f"{prefix} duration_seconds"
    )
    if duration_seconds < 0.0:
        raise ValueError(f"{prefix} duration_seconds must be nonnegative")
    backward_evaluations = _nonnegative_int(
        event.get("backward_evaluation_count"),
        f"{prefix} backward_evaluation_count",
    )
    controller_target_count = _nonnegative_int(
        event.get("controller_target_count"),
        f"{prefix} controller_target_count",
    )
    controller_target_evaluations = _nonnegative_int(
        event.get(
            "controller_target_evaluation_count",
            controller_target_count * len(ordered_granularities),
        ),
        f"{prefix} controller_target_evaluation_count",
    )
    active_epsilon = _finite_float(
        event.get("active_epsilon", fallback_epsilon),
        f"{prefix} active_epsilon",
    )
    if not 0.0 <= active_epsilon <= 1.0:
        raise ValueError(f"{prefix} active_epsilon must be in [0, 1]")
    return PanelGradRefresh(
        refresh_index=refresh_index,
        boundary_step=boundary_step,
        boundary_tokens=min(boundary_step * expected_tokens_per_step, token_budget),
        importance_metric=importance_metric,
        importance_scores=importance_scores,
        gradient_rms_scores=tuple(
            rms_by_granularity[label] for label in ordered_granularities
        ),
        q=q,
        p=p,
        entropy=_finite_float(event.get("entropy"), f"{prefix} entropy"),
        min_probability=_finite_float(
            event.get("min_probability"), f"{prefix} min_probability"
        ),
        max_probability=_finite_float(
            event.get("max_probability"), f"{prefix} max_probability"
        ),
        active_epsilon=active_epsilon,
        epsilon_schedule_step=_nonnegative_int(
            event.get("epsilon_schedule_step", boundary_step),
            f"{prefix} epsilon_schedule_step",
        ),
        duration_seconds=duration_seconds,
        backward_evaluations=backward_evaluations,
        controller_target_tokens=controller_target_evaluations,
    )


def _probability_vector(value: Any, field_name: str, expected_size: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != expected_size:
        raise ValueError(f"{field_name} must contain {expected_size} values")
    values = tuple(
        _finite_float(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )
    if any(item < 0.0 or item > 1.0 for item in values):
        raise ValueError(f"{field_name} values must be in [0, 1]")
    if not math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(f"{field_name} must sum to one")
    return values


def _read_controller_granularity_timeline(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    journal_path: Path,
) -> ControllerGranularityTimeline | None:
    model = _required_mapping(config.get("model"), "config.model")
    controller = model.get("adaptive_controller")
    if not isinstance(controller, Mapping):
        return None
    method_family = str(controller.get("method_family") or "").strip().lower()
    strategy = str(model.get("adaptive_sampler_strategy") or "").strip().lower()
    scope = str(controller.get("scope") or "").strip().lower()
    if not (
        method_family == BAYESIAN_CONTROLLER_METHOD_FAMILY
        and controller.get("method_version") not in (None, "")
        and strategy == "thompson"
        and scope in {"global", "per_block"}
    ):
        return None

    run = _required_mapping(config.get("run"), "config.run")
    run_id = str(run.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("config.run.run_id must be nonempty")
    training = _required_mapping(config.get("training"), "config.training")
    expected_tokens_per_step = _positive_int(
        training.get("expected_tokens_per_step"),
        "config.training.expected_tokens_per_step",
    )
    token_budget = _positive_int(
        training.get("token_budget"),
        "config.training.token_budget",
    )
    granularities = controller.get("ordered_granularities")
    if granularities in (None, ""):
        granularities = model.get("granularities")
    if not isinstance(granularities, list) or not granularities:
        raise ValueError("config.model.granularities must be a nonempty list")
    ordered_granularities = tuple(str(label) for label in granularities)
    if any(not label for label in ordered_granularities):
        raise ValueError("config.model.granularities contains an empty label")
    if len(set(ordered_granularities)) != len(ordered_granularities):
        raise ValueError("config.model.granularities contains duplicate labels")

    if scope == "global":
        block_count = 1
        row_labels = ("all blocks",)
    else:
        feature_schema = controller.get("feature_schema")
        configured_block_count = None
        if isinstance(feature_schema, Mapping):
            configured_block_count = feature_schema.get("block_count")
        if configured_block_count in (None, ""):
            configured_block_count = controller.get("block_count")
        if configured_block_count in (None, ""):
            configured_block_count = model.get("num_layers")
        block_count = _positive_int(
            configured_block_count,
            "config.model.adaptive_controller.feature_schema.block_count",
        )
        row_labels = tuple(str(index) for index in range(1, block_count + 1))

    windows: list[ControllerSelectionWindow] = []
    with journal_path.open("r", encoding="utf-8") as journal_file:
        for line_number, line in enumerate(journal_file, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON on journal line {line_number}: {error.msg}"
                ) from error
            if not isinstance(event, Mapping):
                raise ValueError(f"journal line {line_number} is not a JSON object")
            event_type = event.get("event_type")
            if event_type not in {"completed_window", "terminal_incomplete"}:
                continue
            windows.append(
                _normalize_controller_window(
                    event,
                    line_number=line_number,
                    run_id=run_id,
                    scope=scope,
                    block_count=block_count,
                    ordered_granularities=ordered_granularities,
                    expected_tokens_per_step=expected_tokens_per_step,
                    token_budget=token_budget,
                )
            )

    windows.sort(key=lambda window: (window.start_step, window.window_index))
    for previous, current in zip(windows, windows[1:]):
        if current.start_step < previous.end_step:
            raise ValueError(
                "controller windows overlap: "
                f"window {previous.window_index} ends at step {previous.end_step}, "
                f"window {current.window_index} starts at step {current.start_step}"
            )
    scheduler_warmup_steps, scheduler_warmup_tokens = (
        _scheduler_warmup_boundaries(config, config_path=config_path)
    )
    return ControllerGranularityTimeline(
        run_id=run_id,
        scope=scope,
        ordered_granularities=ordered_granularities,
        block_count=block_count,
        row_labels=row_labels,
        token_budget=token_budget,
        windows=tuple(windows),
        model_variant=model_variant_from_saved_config(dict(config)),
        correction_mode=correction_mode_from_saved_config(dict(config)),
        membership_correction=membership_correction_from_saved_config(dict(config)),
        config_path=config_path,
        journal_path=journal_path,
        scheduler_warmup_steps=scheduler_warmup_steps,
        scheduler_warmup_tokens=scheduler_warmup_tokens,
    )


def _normalize_controller_window(
    event: Mapping[str, Any],
    *,
    line_number: int,
    run_id: str,
    scope: str,
    block_count: int,
    ordered_granularities: tuple[str, ...],
    expected_tokens_per_step: int,
    token_budget: int,
) -> ControllerSelectionWindow:
    event_run_id = event.get("run_id")
    if event_run_id not in (None, run_id):
        raise ValueError(
            f"journal line {line_number} run_id {event_run_id!r} does not match {run_id!r}"
        )
    event_scope = event.get("scope")
    if event_scope not in (None, scope):
        raise ValueError(
            f"journal line {line_number} scope {event_scope!r} does not match {scope!r}"
        )
    window_index = _nonnegative_int(
        event.get("window_index"),
        f"journal line {line_number} window_index",
    )
    event_type = event["event_type"]
    if event_type == "completed_window":
        start_step = _nonnegative_int(
            event.get("boundary_step_start"),
            f"journal line {line_number} boundary_step_start",
        )
        end_step = _nonnegative_int(
            event.get("boundary_step_end"),
            f"journal line {line_number} boundary_step_end",
        )
        completed_steps = _positive_int(
            event.get("completed_optimizer_steps"),
            f"journal line {line_number} completed_optimizer_steps",
        )
        if end_step - start_step != completed_steps:
            raise ValueError(
                f"journal line {line_number} completed step count does not match boundaries"
            )
        terminal_incomplete = False
    else:
        if event.get("observation_emitted") is not False:
            raise ValueError(
                f"journal line {line_number} terminal window must set observation_emitted=false"
            )
        start_step = _nonnegative_int(
            event.get("boundary_step"),
            f"journal line {line_number} boundary_step",
        )
        completed_steps = _positive_int(
            event.get("completed_optimizer_steps"),
            f"journal line {line_number} completed_optimizer_steps",
        )
        interval = _positive_int(
            event.get("decision_interval_steps"),
            f"journal line {line_number} decision_interval_steps",
        )
        if completed_steps >= interval:
            raise ValueError(
                f"journal line {line_number} terminal window is not partial"
            )
        end_step = start_step + completed_steps
        terminal_incomplete = True

    action = _required_mapping(
        event.get("action"),
        f"journal line {line_number} action",
    )
    if scope == "global":
        label = str(action.get("global_granularity") or "")
        block_granularities = (label,)
    else:
        profile = action.get("block_granularities")
        if not isinstance(profile, list) or len(profile) != block_count:
            raise ValueError(
                f"journal line {line_number} per-block profile must contain "
                f"exactly {block_count} labels"
            )
        block_granularities = tuple(str(label) for label in profile)
    unknown = sorted(set(block_granularities) - set(ordered_granularities))
    if unknown:
        raise ValueError(
            f"journal line {line_number} uses unknown granularities: {unknown}"
        )

    start_tokens = min(start_step * expected_tokens_per_step, token_budget)
    end_tokens = min(end_step * expected_tokens_per_step, token_budget)
    if end_tokens <= start_tokens:
        raise ValueError(
            f"journal line {line_number} has an empty token interval after budget clipping"
        )
    return ControllerSelectionWindow(
        window_index=window_index,
        start_step=start_step,
        end_step=end_step,
        start_tokens=start_tokens,
        end_tokens=end_tokens,
        block_granularities=block_granularities,
        terminal_incomplete=terminal_incomplete,
    )


def _required_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    integer = _nonnegative_int(value, field_name)
    if integer <= 0:
        raise ValueError(f"{field_name} must be positive")
    return integer


def _signed_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return value


def _csv_positive_int(value: Any, field_name: str) -> int:
    integer = _csv_nonnegative_int(value, field_name)
    if integer <= 0:
        raise ValueError(f"{field_name} must be positive")
    return integer


def _csv_nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a nonnegative integer")
    try:
        integer = int(str(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a nonnegative integer") from error
    if integer < 0 or str(value).strip() not in {str(integer), f"{integer}.0"}:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return integer


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def validation_split_filter(row: dict[str, str]) -> bool:
    return str(row.get("split") or "") == "validation"


def refresh_scaling_parameter_counts(
    input_root: Path,
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    count_cache: dict[Path, dict[str, dict[str, Any]] | None] = {}
    refreshed_rows = []

    for row in rows:
        refreshed_row = dict(row)
        config_path = config_path_for_scaling_row(input_root, row)
        granularity = str(row.get("granularity") or "")
        if config_path is not None and granularity:
            if config_path not in count_cache:
                try:
                    count_cache[config_path] = recompute_parameter_counts(config_path)
                except Exception as error:
                    # Count refresh is a reporting enhancement. Historical configs
                    # may predate required model-construction fields, so preserve
                    # their stored scaling counts instead of blocking all figures.
                    warnings.warn(
                        f"Could not refresh parameter counts from {config_path}; "
                        f"using scaling_results.csv values: {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    count_cache[config_path] = None
            refreshed_counts = count_cache[config_path]
            counts = (
                refreshed_counts.get(granularity)
                if refreshed_counts is not None
                else None
            )
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
            global_sampling_distribution = (
                global_sampling_distribution_from_saved_config(
                    config_cache[config_path]
                )
            )
            if global_sampling_distribution is not None:
                enriched_row["global_sampling_distribution"] = (
                    global_sampling_distribution
                )
            global_sampling_interval_steps = (
                global_sampling_interval_steps_from_saved_config(
                    config_cache[config_path]
                )
            )
            if global_sampling_interval_steps is not None:
                enriched_row["global_sampling_interval_steps"] = (
                    global_sampling_interval_steps
                )
            global_sampling_schedule = global_sampling_schedule_from_saved_config(
                config_cache[config_path]
            )
            if global_sampling_schedule is not None:
                enriched_row["global_sampling_schedule"] = global_sampling_schedule
            global_sampling_schedule_version = (
                global_sampling_schedule_version_from_saved_config(
                    config_cache[config_path]
                )
            )
            if global_sampling_schedule_version is not None:
                enriched_row["global_sampling_schedule_version"] = (
                    global_sampling_schedule_version
                )
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
            _enrich_controller_provenance(
                enriched_row,
                config_cache[config_path],
            )
            _enrich_scheduler_provenance(
                enriched_row,
                config_cache[config_path],
                config_path=config_path,
            )
        enriched_rows.append(enriched_row)

    return enriched_rows


def enrich_metrics_metadata_from_run_config(
    input_root: Path,
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    config_cache: dict[Path, dict[str, Any]] = {}
    completion_cache: dict[Path, dict[str, Any] | None] = {}
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
            global_sampling_distribution = (
                global_sampling_distribution_from_saved_config(
                    config_cache[config_path]
                )
            )
            if global_sampling_distribution is not None:
                enriched_row["global_sampling_distribution"] = (
                    global_sampling_distribution
                )
            global_sampling_interval_steps = (
                global_sampling_interval_steps_from_saved_config(
                    config_cache[config_path]
                )
            )
            if global_sampling_interval_steps is not None:
                enriched_row["global_sampling_interval_steps"] = (
                    global_sampling_interval_steps
                )
            global_sampling_schedule = global_sampling_schedule_from_saved_config(
                config_cache[config_path]
            )
            if global_sampling_schedule is not None:
                enriched_row["global_sampling_schedule"] = global_sampling_schedule
            global_sampling_schedule_version = (
                global_sampling_schedule_version_from_saved_config(
                    config_cache[config_path]
                )
            )
            if global_sampling_schedule_version is not None:
                enriched_row["global_sampling_schedule_version"] = (
                    global_sampling_schedule_version
                )
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
            _enrich_controller_provenance(
                enriched_row,
                config_cache[config_path],
            )
            _enrich_scheduler_provenance(
                enriched_row,
                config_cache[config_path],
                config_path=config_path,
            )
            contract, historical_fallback = seed_independent_validation_contract(
                config_cache[config_path],
                run_id=str(enriched_row.get("run_id") or config_path.parent.name),
            )
            enriched_row["_validation_contract"] = contract
            enriched_row["_validation_contract_fallback"] = historical_fallback
            ordered_granularities = _mapping_value(
                config_cache[config_path], "model", "granularities"
            )
            if isinstance(ordered_granularities, list):
                enriched_row["_ordered_granularities"] = list(ordered_granularities)
        _enrich_run_completion(enriched_row, completion_cache)
        enriched_rows.append(enriched_row)

    return enriched_rows


_VALIDATION_CONTRACT_IGNORED_KEYS = {
    "artifact_retry_count",
    "completion_label",
    "continuation",
    "derived_seeds",
    "extraction_metadata_path",
    "metrics_path",
    "output_dir",
    "output_group",
    "output_root",
    "parameter_counts",
    "parameter_counts_by_granularity",
    "phase_id",
    "reproducibility",
    "resolved_seeds",
    "run_id",
    "scaling_results_path",
    "schedule",
    "schedule_hash",
    "schedule_seed",
    "seed",
    "seed_streams",
    "status",
    "tokens_seen",
    "content_tokens_seen",
}


def _seed_independent_contract_value(value: Any) -> Any:
    if isinstance(value, dict):
        normalized = {}
        for raw_key, nested_value in sorted(
            value.items(), key=lambda item: str(item[0])
        ):
            key = str(raw_key)
            normalized_key = key.strip().lower()
            if normalized_key in _VALIDATION_CONTRACT_IGNORED_KEYS:
                continue
            if "seed" in normalized_key or normalized_key.endswith("_hash"):
                continue
            if normalized_key.endswith("_path") or normalized_key.endswith("_paths"):
                continue
            normalized[key] = _seed_independent_contract_value(nested_value)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_seed_independent_contract_value(item) for item in value]
    if isinstance(value, float) and value == 0.0:
        return 0.0
    return value


def seed_independent_validation_contract(
    config: Mapping[str, Any],
    *,
    run_id: str,
) -> tuple[str, bool]:
    """Return a conservative complete contract and historical-fallback flag.

    A saved config is considered provably comparable only when it contains the
    core model, optimizer-budget, and dataset identities. Older artifacts that
    lack any of those sections are isolated by run identity.
    """

    model = config.get("model")
    training = config.get("training")
    dataset = config.get("dataset")
    run = config.get("run")
    required_values = (
        _mapping_value(model, "variant"),
        _mapping_value(model, "granularities"),
        _mapping_value(training, "token_budget"),
        _first_config_value(
            dict(config),
            ("training", "resolved_learning_rate"),
            ("training", "learning_rate"),
            ("training", "base_learning_rate"),
        ),
        _first_config_value(
            dict(config),
            ("training", "optimizer", "name"),
            ("training", "optimizer_name"),
        ),
        _mapping_value(dataset, "dataset_name"),
        _mapping_value(dataset, "dataset_split"),
        _mapping_value(run, "model_family"),
    )
    historical_fallback = any(value in (None, "", []) for value in required_values)
    payload: dict[str, Any] = {"config": _seed_independent_contract_value(dict(config))}
    if historical_fallback:
        payload["historical_run_fallback"] = run_id
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str),
        historical_fallback,
    )


def _enrich_run_completion(
    row: dict[str, Any],
    cache: dict[Path, dict[str, Any] | None],
) -> None:
    source_csv = row.get("_source_csv")
    summary_path = (
        Path(str(source_csv)).parent / "run_summary.json" if source_csv else None
    )
    summary: dict[str, Any] | None = None
    if summary_path is not None:
        if summary_path not in cache:
            loaded_summary = None
            if summary_path.is_file():
                try:
                    with summary_path.open("r", encoding="utf-8") as summary_file:
                        loaded = json.load(summary_file)
                    if isinstance(loaded, dict):
                        loaded_summary = loaded
                except (OSError, json.JSONDecodeError):
                    loaded_summary = None
            cache[summary_path] = loaded_summary
        summary = cache[summary_path]

    row["_run_summary_present"] = summary is not None
    row["_run_status"] = (
        str(summary.get("status") or "missing") if summary else "missing"
    )
    progress_tokens = None
    token_budget = None
    if summary is not None:
        progress_tokens = summary.get("tokens_seen")
        token_budget = summary.get("token_budget")
    if progress_tokens in (None, ""):
        progress_tokens = row.get("tokens_seen")
    row["_run_progress_tokens"] = progress_tokens
    row["_run_token_budget"] = token_budget


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


def _adaptive_controller_from_saved_config(
    config: dict[str, Any],
) -> dict[str, Any] | None:
    model = config.get("model")
    if not isinstance(model, dict):
        return None
    controller = (
        model.get("panelgrad")
        if model.get("adaptive_sampler_strategy") == "panelgrad"
        else model.get("adaptive_controller")
    )
    return controller if isinstance(controller, dict) else None


def controller_method_family_from_saved_config(
    config: dict[str, Any],
) -> str | None:
    controller = _adaptive_controller_from_saved_config(config)
    if controller is None or controller.get("method_family") in (None, ""):
        return None
    return str(controller["method_family"]).strip().lower()


def controller_method_version_from_saved_config(
    config: dict[str, Any],
) -> Any | None:
    controller = _adaptive_controller_from_saved_config(config)
    if controller is None or controller.get("method_version") in (None, ""):
        return None
    return controller["method_version"]


def controller_scope_from_saved_config(config: dict[str, Any]) -> str | None:
    controller = _adaptive_controller_from_saved_config(config)
    if controller is None or controller.get("scope") in (None, ""):
        return None
    return str(controller["scope"]).strip().lower()


def controller_reset_enabled_from_saved_config(
    config: dict[str, Any],
) -> bool | None:
    controller = _adaptive_controller_from_saved_config(config)
    if controller is None:
        return None
    reset = controller.get("reset")
    if not isinstance(reset, dict):
        return None
    return bool(reset.get("enabled", False))


def controller_reset_policy_from_saved_config(
    config: dict[str, Any],
) -> str | None:
    controller = _adaptive_controller_from_saved_config(config)
    if controller is None:
        return None
    reset = controller.get("reset")
    if not isinstance(reset, dict) or reset.get("policy") in (None, ""):
        return None
    return str(reset["policy"]).strip().lower()


def _mapping_value(mapping: Any, *keys: str) -> Any | None:
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return None if current in (None, "") else current


def _first_config_value(config: dict[str, Any], *paths: tuple[str, ...]) -> Any | None:
    for path in paths:
        value = _mapping_value(config, *path)
        if value is not None:
            return value
    return None


def _controller_reset_value(config: dict[str, Any], key: str) -> Any | None:
    controller = _adaptive_controller_from_saved_config(config)
    if controller is None:
        return None
    reset = controller.get("reset")
    if not isinstance(reset, dict):
        return None
    value = reset.get(key)
    return None if value in (None, "") else value


def controller_contract_provenance_from_saved_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return seed-independent fields needed to compare size-plot experiments."""

    controller = _adaptive_controller_from_saved_config(config) or {}
    warmup = _first_config_value(config, ("training", "pre_nested_warmup"))
    warmup = warmup if isinstance(warmup, dict) else {}
    comparison = config.get("comparison_control_inputs")
    comparison = comparison if isinstance(comparison, dict) else {}
    optimizer = _first_config_value(config, ("training", "optimizer"))
    optimizer = optimizer if isinstance(optimizer, dict) else {}
    scheduler = _first_config_value(config, ("training", "scheduler"))
    scheduler = scheduler if isinstance(scheduler, dict) else {}
    panelgrad_schedule = _panelgrad_epsilon_schedule_identity(controller)

    return {
        "run_seed": _first_config_value(
            config,
            ("run", "seed"),
            ("comparison_control_inputs", "root_seed"),
            ("data", "seed"),
        ),
        "controller_decision_interval_steps": controller.get("decision_interval_steps"),
        "controller_observation_noise_variance": controller.get(
            "observation_noise_variance"
        ),
        "controller_process_noise_covariance": controller.get(
            "process_noise_covariance_input",
            controller.get("process_noise_covariance"),
        ),
        "controller_prior_mean": controller.get(
            "prior_mean_input", controller.get("prior_mean")
        ),
        "controller_prior_covariance": controller.get(
            "prior_covariance_input", controller.get("prior_covariance")
        ),
        "controller_context_model": controller.get("context_model"),
        "controller_compute_weight": controller.get("compute_weight"),
        "controller_feature_schema_hash": _mapping_value(
            controller, "feature_schema", "schema_hash"
        ),
        "panelgrad_refresh_interval_steps": controller.get(
            "refresh_interval_steps"
        ),
        "panelgrad_importance_metric": (
            _panelgrad_importance_metric(controller)
            if (
                str(controller.get("method_family") or "").strip().lower()
                in PANELGRAD_METHOD_FAMILIES
                or "importance_metric" in controller
            )
            else None
        ),
        "panelgrad_eta": controller.get("eta"),
        "panelgrad_temperature": controller.get("temperature"),
        "panelgrad_epsilon": controller.get("epsilon"),
        "panelgrad_epsilon_schedule_type": panelgrad_schedule.get("type"),
        "panelgrad_epsilon_schedule_start": panelgrad_schedule.get("start"),
        "panelgrad_epsilon_schedule_end": panelgrad_schedule.get("end"),
        "panelgrad_epsilon_schedule_duration_steps": (
            panelgrad_schedule.get("duration_steps")
        ),
        "controller_reset_interval_steps": _controller_reset_value(
            config, "interval_steps"
        ),
        "controller_acquisition_policy": _controller_reset_value(
            config, "acquisition_policy"
        ),
        "controller_acquisition_passes": _controller_reset_value(
            config, "acquisition_passes"
        ),
        "pre_nested_warmup_enabled": warmup.get("enabled"),
        "pre_nested_warmup_policy": warmup.get("policy"),
        "pre_nested_warmup_duration": warmup.get("duration"),
        "pre_nested_warmup_action_interval_steps": warmup.get("action_interval_steps"),
        "training_token_budget": _first_config_value(
            config,
            ("training", "token_budget"),
            ("comparison_control_inputs", "token_budget"),
        ),
        "training_learning_rate": _first_config_value(
            config,
            ("training", "resolved_learning_rate"),
            ("training", "learning_rate"),
            ("comparison_control_inputs", "learning_rate"),
        ),
        "training_max_steps": _first_config_value(config, ("training", "max_steps")),
        "training_optimizer_name": optimizer.get("name")
        or _first_config_value(config, ("training", "optimizer_name")),
        "training_optimizer_kwargs": optimizer.get("kwargs")
        or _first_config_value(config, ("training", "optimizer_kwargs")),
        "training_scheduler_name": scheduler.get("name")
        or _first_config_value(config, ("training", "scheduler_name")),
        "training_scheduler_kwargs": scheduler.get("kwargs")
        or _first_config_value(config, ("training", "scheduler_kwargs")),
        "training_context_length": comparison.get("context_length"),
        "training_batch_size_per_process": comparison.get("batch_size_per_process"),
        "training_precision": comparison.get("precision"),
        "training_dataset_name": comparison.get("dataset_name"),
        "training_dataset_config_name": comparison.get("dataset_config_name"),
        "training_dataset_split": comparison.get("dataset_split"),
        "training_tokenizer_name": comparison.get("tokenizer_name"),
    }


def _panelgrad_policy_label(policy: Mapping[str, Any]) -> str:
    """Render the scientific knobs that distinguish PanelGrad policies."""

    schedule = _panelgrad_epsilon_schedule_identity(policy)
    importance_metric = _panelgrad_importance_metric(policy)
    labels = [
        (
            "metric",
            {
                "gradient_rms": "Gradient RMS",
                "gradient_l2": "Gradient L2 norm",
            }[importance_metric],
        ),
        ("T", policy.get("temperature")),
    ]
    if schedule["type"] == "linear":
        labels.extend(
            (
                ("ε type", schedule["type"]),
                ("ε start", schedule["start"]),
                ("ε end", schedule["end"]),
                ("ε steps", schedule["duration_steps"]),
            )
        )
    elif "epsilon" in policy or "epsilon_schedule" in policy:
        labels.append(("ε", schedule["start"]))
    labels.extend(
        (
            ("H", policy.get("refresh_interval_steps")),
            ("η", policy.get("eta")),
        )
    )
    parts = []
    for name, value in labels:
        if value in (None, ""):
            continue
        if isinstance(value, float):
            rendered = f"{value:g}"
        else:
            rendered = str(value)
        parts.append(f"{name}={rendered}")
    return "PanelGrad" if not parts else f"PanelGrad ({', '.join(parts)})"


def _panelgrad_epsilon_schedule_identity(
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    is_panelgrad = (
        str(policy.get("method_family") or "").strip().lower()
        in PANELGRAD_METHOD_FAMILIES
        or "epsilon" in policy
        or "epsilon_schedule" in policy
    )
    if not is_panelgrad:
        return {"type": None, "start": None, "end": None, "duration_steps": None}
    schedule = policy.get("epsilon_schedule")
    if isinstance(schedule, Mapping) and schedule.get("type") == "linear":
        return {
            "type": "linear",
            "start": schedule.get("start"),
            "end": schedule.get("end"),
            "duration_steps": schedule.get("duration_steps"),
        }
    epsilon = policy.get("epsilon", 0.1)
    return {
        "type": "fixed",
        "start": epsilon,
        "end": epsilon,
        "duration_steps": None,
    }


def _panelgrad_importance_metric(policy: Mapping[str, Any]) -> str:
    metric = policy.get("importance_metric")
    family = str(policy.get("method_family") or "").strip().lower()
    if metric in (None, ""):
        metric = PANELGRAD_METHOD_FAMILIES.get(family, "gradient_rms")
    metric = str(metric)
    if metric not in {"gradient_rms", "gradient_l2"}:
        raise ValueError(f"Unknown PanelGrad importance metric: {metric!r}")
    expected = PANELGRAD_METHOD_FAMILIES.get(family)
    if expected is not None and metric != expected:
        raise ValueError("PanelGrad importance metric and method family disagree")
    return metric


def _enrich_controller_provenance(
    row: dict[str, Any],
    config: dict[str, Any],
) -> None:
    provenance = {
        "controller_method_family": controller_method_family_from_saved_config(config),
        "controller_method_version": controller_method_version_from_saved_config(
            config
        ),
        "controller_scope": controller_scope_from_saved_config(config),
        "controller_reset_enabled": controller_reset_enabled_from_saved_config(config),
        "controller_reset_policy": controller_reset_policy_from_saved_config(config),
    }
    for field_name, value in provenance.items():
        if value not in (None, ""):
            row[field_name] = value


def _enrich_scheduler_provenance(
    row: dict[str, Any],
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> None:
    """Attach plotting-only scheduler boundaries reconstructed from config."""

    try:
        schedule = _learning_rate_schedule_from_saved_config(
            config,
            config_path=config_path,
        )
    except (ValueError, TypeError, KeyError):
        return
    row["_scheduler_name"] = schedule.scheduler_name
    row["_scheduler_peak_learning_rate"] = schedule.peak_learning_rate
    row["_scheduler_warmup_steps"] = schedule.warmup_steps
    row["_scheduler_warmup_tokens"] = min(
        schedule.warmup_steps * schedule.expected_tokens_per_step,
        schedule.token_budget,
    )
    row["_scheduler_max_steps"] = schedule.max_steps
    for field_name, value in controller_contract_provenance_from_saved_config(
        config
    ).items():
        if value not in (None, ""):
            row[field_name] = value


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


def global_sampling_distribution_from_saved_config(
    config: dict[str, Any],
) -> dict[str, float] | None:
    model = config.get("model")
    if not isinstance(model, dict):
        return None
    value = model.get("global_sampling_distribution")
    if not isinstance(value, Mapping):
        return None
    return {str(label): float(probability) for label, probability in value.items()}


def global_sampling_interval_steps_from_saved_config(
    config: dict[str, Any],
) -> int | None:
    model = config.get("model")
    if not isinstance(model, dict):
        return None
    value = model.get("global_sampling_interval_steps")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def global_sampling_schedule_from_saved_config(
    config: dict[str, Any],
) -> str | None:
    model = config.get("model")
    if not isinstance(model, dict):
        return None
    value = model.get("global_sampling_schedule", "random_with_replacement")
    if value not in {"random_with_replacement", "balanced_cycle"}:
        return None
    return str(value)


def global_sampling_schedule_version_from_saved_config(
    config: dict[str, Any],
) -> int | None:
    model = config.get("model")
    if not isinstance(model, dict):
        return None
    value = model.get("global_sampling_schedule_version")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return int(value)


def with_default_model_variant(config: dict[str, Any]) -> dict[str, Any]:
    normalized_config = json.loads(json.dumps(config))
    model = normalized_config.setdefault("model", {})
    if isinstance(model, dict) and model.get("variant") in (None, ""):
        model["variant"] = "matformer_llama"
    return normalized_config
