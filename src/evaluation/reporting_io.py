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
    "PanelGradAction",
    "PanelGradHistory",
    "PanelGradRefresh",
    "config_path_for_scaling_row",
    "correction_mode_from_saved_config",
    "enrich_metrics_metadata_from_run_config",
    "enrich_scaling_metadata_from_run_config",
    "granularity_sampling_mode_from_saved_config",
    "iter_controller_granularity_timelines",
    "iter_global_sampling_histories",
    "iter_panelgrad_histories",
    "iter_csv_artifact_rows",
    "membership_correction_from_saved_config",
    "model_variant_from_saved_config",
    "panelgrad_history_as_timeline",
    "global_sampling_history_as_timeline",
    "read_csv_artifacts",
    "read_csv_artifacts_filtered",
    "recompute_parameter_counts",
    "refresh_scaling_parameter_counts",
    "resolved_sampling_mode_from_saved_config",
    "seed_independent_validation_contract",
    "validation_split_filter",
    "with_default_model_variant",
]


BAYESIAN_CONTROLLER_METHOD_FAMILY = "bayesian_gaussian_linear_thompson"
PANELGRAD_METHOD_FAMILY = "panelgrad_gradient_rms"


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


def read_csv_artifacts(input_root: Path, filename: str) -> list[dict[str, str]]:
    return read_csv_artifacts_filtered(input_root, filename, row_filter=None)


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

    sampling_mode = str(model.get("granularity_sampling_mode") or "global").lower()
    strategy = str(model.get("adaptive_sampler_strategy") or "").lower()
    if sampling_mode == "global":
        policy_identity, policy_label = "uniform_global", "Uniform global"
    elif sampling_mode == "adaptive_global" and strategy == "panelgrad":
        panelgrad = model.get("panelgrad")
        if not isinstance(panelgrad, Mapping) or str(
            panelgrad.get("method_family") or ""
        ).lower() != PANELGRAD_METHOD_FAMILY:
            return None
        policy_identity = "panelgrad_global"
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
    granularities = model.get("granularities")
    if not isinstance(granularities, list) or not granularities:
        raise ValueError("config.model.granularities must be a nonempty list")
    ordered_granularities = tuple(str(label) for label in granularities)
    if len(set(ordered_granularities)) != len(ordered_granularities):
        raise ValueError("config.model.granularities contains duplicates")
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
            policy_identity == "panelgrad_global"
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
        else:
            decision_index = step - 1
        decision_indices.add(decision_index)
        probability_raw = row.get("controller_sampled_probability")
        sampled_probability = None
        if probability_raw not in (None, ""):
            sampled_probability = _finite_float(
                probability_raw,
                f"metrics.csv step {step} sampled probability",
            )
            if not 0.0 <= sampled_probability <= 1.0:
                raise ValueError("sampled probability must be in [0, 1]")
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
    )


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
        method_family == PANELGRAD_METHOD_FAMILY
        and policy.get("method_version") not in (None, "")
        and strategy == "panelgrad"
        and scope == "global"
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
                )
            )
    refreshes.sort(key=lambda refresh: (refresh.boundary_step, refresh.refresh_index))

    return PanelGradHistory(
        run_id=run_id,
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
    )


def _normalize_panelgrad_refresh(
    event: Mapping[str, Any],
    *,
    line_number: int,
    ordered_granularities: tuple[str, ...],
    expected_tokens_per_step: int,
    token_budget: int,
    fallback_epsilon: float,
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
    for measurement in measurements:
        if not isinstance(measurement, Mapping):
            raise ValueError(f"{prefix} measurement must be a mapping")
        label = str(measurement.get("granularity") or "")
        if label in score_by_granularity:
            raise ValueError(f"{prefix} repeats measurement for {label!r}")
        score_by_granularity[label] = _finite_float(
            measurement.get("gradient_rms_score"),
            f"{prefix} {label} gradient_rms_score",
        )
    if set(score_by_granularity) != set(ordered_granularities):
        raise ValueError(f"{prefix} measurements do not cover every granularity")
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
        gradient_rms_scores=tuple(
            score_by_granularity[label] for label in ordered_granularities
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
        controller_target_tokens=controller_target_count * backward_evaluations,
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
    labels = [("T", policy.get("temperature"))]
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
        == PANELGRAD_METHOD_FAMILY
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


def with_default_model_variant(config: dict[str, Any]) -> dict[str, Any]:
    normalized_config = json.loads(json.dumps(config))
    model = normalized_config.setdefault("model", {})
    if isinstance(model, dict) and model.get("variant") in (None, ""):
        model["variant"] = "matformer_llama"
    return normalized_config
