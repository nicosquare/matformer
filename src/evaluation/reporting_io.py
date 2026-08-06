"""CSV artifact loading and metadata enrichment helpers for reporting."""

from __future__ import annotations

import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .reporting_styles import PARAMETER_COUNT_FIELDS

__all__ = [
    "adaptive_sampler_strategy_from_saved_config",
    "controller_method_family_from_saved_config",
    "controller_method_version_from_saved_config",
    "controller_scope_from_saved_config",
    "ControllerGranularityTimeline",
    "ControllerSelectionWindow",
    "config_path_for_scaling_row",
    "correction_mode_from_saved_config",
    "enrich_metrics_metadata_from_run_config",
    "enrich_scaling_metadata_from_run_config",
    "granularity_sampling_mode_from_saved_config",
    "iter_controller_granularity_timelines",
    "membership_correction_from_saved_config",
    "model_variant_from_saved_config",
    "read_csv_artifacts",
    "read_csv_artifacts_filtered",
    "recompute_parameter_counts",
    "refresh_scaling_parameter_counts",
    "resolved_sampling_mode_from_saved_config",
    "validation_split_filter",
    "with_default_model_variant",
]


BAYESIAN_CONTROLLER_METHOD_FAMILY = "bayesian_gaussian_linear_thompson"


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
    config_path: Path
    journal_path: Path


def read_csv_artifacts(input_root: Path, filename: str) -> list[dict[str, str]]:
    return read_csv_artifacts_filtered(input_root, filename, row_filter=None)


def read_csv_artifacts_filtered(
    input_root: Path,
    filename: str,
    row_filter: Any | None,
) -> list[dict[str, str]]:
    rows = []
    for path in sorted(input_root.rglob(filename)):
        with path.open("r", encoding="utf-8", newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                if row_filter is not None and not row_filter(row):
                    continue
                row["_source_csv"] = str(path)
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
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            warnings.warn(
                f"Skipping controller timeline {journal_path}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if timeline is not None and timeline.windows:
            yield timeline


def _read_controller_granularity_timeline(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    journal_path: Path,
) -> ControllerGranularityTimeline | None:
    model = _required_mapping(config.get("model"), "config.model")
    controller = _required_mapping(
        model.get("adaptive_controller"),
        "config.model.adaptive_controller",
    )
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


def validation_split_filter(row: dict[str, str]) -> bool:
    return str(row.get("split") or "") == "validation"


def refresh_scaling_parameter_counts(
    input_root: Path,
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    count_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    refreshed_rows = []

    for row in rows:
        refreshed_row = dict(row)
        config_path = config_path_for_scaling_row(input_root, row)
        granularity = str(row.get("granularity") or "")
        if config_path is not None and granularity:
            if config_path not in count_cache:
                count_cache[config_path] = recompute_parameter_counts(config_path)
            counts = count_cache[config_path].get(granularity)
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
    controller = model.get("adaptive_controller")
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
    }
    for field_name, value in provenance.items():
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
