"""Configuration helpers for MatFormer reproduction runs."""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from utils.model_size import (
    MODEL_FAMILY_SLUG,
    derive_model_size_slug,
    derive_token_budget_slug,
)
from utils.monitoring import DEFAULT_MONITORING_BACKEND, VALID_MONITORING_BACKENDS


VALID_GRANULARITIES = {"s", "m", "l", "xl"}
VALID_MODEL_TOPOLOGIES = {"nested", "standalone"}
VALID_MODEL_VARIANTS = {"slicing", "concat"}
VALID_CORRECTION_MODES = {"none", "gmc", "lmc"}
VALID_MODEL_GRANULARITY_SAMPLING_MODES = {"global", "per_layer"}
VALID_LEARNING_RATE_SCALE_RULES = {"none", "linear", "sqrt"}
VALID_OPTIMIZER_NAMES = {"adamw", "sgd"}
VALID_COMPLETION_LABELS = {"debug", "run"}
VALID_GRANULARITY_SAMPLING = {"all", "random"}
VALID_PRE_NESTED_WARMUP_UNITS = {"epochs", "steps"}
DEFAULT_MODEL_VARIANT = "slicing"
VALID_SAMPLING_MODES = {"nested-random", "nested-all", "standalone"}
CANONICAL_GRANULARITY_ORDER = ("s", "m", "l", "xl")
CANONICAL_GRANULARITY_PREFIX_FRACTIONS = {
    "s": (1, 8),
    "m": (1, 4),
    "l": (1, 2),
    "xl": (1, 1),
}
DEFAULT_FFN_MULTIPLIER = 4
CONFIG_ROOT = Path(__file__).resolve().parent.parent
PRESET_REGISTRY_ROOT = CONFIG_ROOT / "configs" / "presets"
OPTIMIZER_DEFAULT_KWARGS = {
    "adamw": {
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.1,
    },
    "sgd": {
        "momentum": 0.0,
        "dampening": 0.0,
        "nesterov": False,
        "weight_decay": 0.0,
    },
}
OPTIMIZER_ALLOWED_KWARGS = {
    "adamw": {"betas", "eps", "weight_decay"},
    "sgd": {"momentum", "dampening", "nesterov", "weight_decay"},
}
SCHEDULER_RESERVED_KWARGS = {"num_warmup_steps", "num_training_steps", "optimizer"}


class ConfigError(ValueError):
    """Raised when a config would silently mislabel an experiment."""


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ConfigError(f"Config must be a YAML mapping: {config_path}")

    return config


def parse_override(raw_override: str) -> tuple[str, Any]:
    if "=" not in raw_override:
        raise ConfigError(f"Override must use dotted.path=value: {raw_override}")

    key, raw_value = raw_override.split("=", 1)
    key = key.strip()
    if not key or any(part.strip() == "" for part in key.split(".")):
        raise ConfigError(f"Override has an invalid dotted path: {raw_override}")

    return key, yaml.safe_load(raw_value)


def resolve_optimizer_kwargs(
    optimizer_name: str,
    raw_kwargs: Any | None,
) -> dict[str, Any]:
    normalized_name = _normalize_optimizer_name(optimizer_name)
    return _resolve_optimizer_kwargs(normalized_name, raw_kwargs)


def apply_overrides(
    config: Mapping[str, Any],
    overrides: Mapping[str, Any] | Iterable[str] | None = None,
) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(config))
    if not overrides:
        return resolved

    if isinstance(overrides, Mapping):
        override_items = overrides.items()
    else:
        override_items = (parse_override(override) for override in overrides)

    for key, value in override_items:
        _set_dotted_value(resolved, key, value)

    return resolved


def _snapshot_overrides(
    overrides: Mapping[str, Any] | Iterable[str] | None,
) -> Mapping[str, Any] | list[str] | None:
    if overrides is None or isinstance(overrides, Mapping):
        return overrides
    return list(overrides)


def _override_keys(overrides: Mapping[str, Any] | Iterable[str] | None) -> set[str]:
    if not overrides:
        return set()

    if isinstance(overrides, Mapping):
        keys: set[str] = set()
        for key, value in overrides.items():
            key_text = str(key)
            keys.add(key_text)
            if isinstance(value, Mapping):
                keys.update(
                    f"{key_text}.{nested_key}"
                    for nested_key in _mapping_dotted_keys(value)
                )
        return keys

    return {raw_override.split("=", 1)[0].strip() for raw_override in overrides}


def _mapping_dotted_keys(mapping: Mapping[str, Any]) -> set[str]:
    keys: set[str] = set()
    for key, value in mapping.items():
        key_text = str(key)
        keys.add(key_text)
        if isinstance(value, Mapping):
            keys.update(
                f"{key_text}.{nested_key}"
                for nested_key in _mapping_dotted_keys(value)
            )
    return keys


def resolve_run_config(
    config_path: str | Path,
    run_id: str | None = None,
    overrides: Mapping[str, Any] | Iterable[str] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    overrides = _snapshot_overrides(overrides)
    explicit_override_keys = _override_keys(overrides)
    config = apply_overrides(load_yaml_config(config_path), overrides)
    requested_granularity_sampling_alias = _configured_granularity_sampling_alias(
        config,
        explicit_override_keys,
    )
    requested_run_sampling_mode = _configured_run_sampling_mode(config)
    family_size_slug = _configured_family_size_slug(config)
    if "matrix" in config:
        if family_size_slug is None:
            family_size_slug = _resolve_family_size_slug(config)

    if "matrix" in config:
        run_entry = _select_matrix_run(config, run_id)
        resolved = _compose_matrix_run(config, run_entry)
        if family_size_slug is not None:
            resolved["run"]["family_size_slug"] = family_size_slug
    else:
        resolved = _compose_single_run(config)
        if run_id is not None and resolved["run"].get("run_id") != run_id:
            raise ConfigError(
                f"Requested run_id={run_id}, but config defines "
                f"run_id={resolved['run'].get('run_id')}"
            )

    if output_dir is not None:
        resolved["run"]["output_dir"] = str(output_dir)

    _resolve_model_variant_defaults(resolved)
    _resolve_model_correction_defaults(resolved)
    _resolve_model_dimension_and_granularity_metadata(resolved)
    if family_size_slug is None:
        family_size_slug = _resolve_family_size_slug(resolved)
    resolved["run"]["family_size_slug"] = family_size_slug
    _resolve_naming_defaults(resolved)
    _resolve_output_paths(resolved)
    _resolve_sampling_mode_defaults(
        resolved,
        requested_granularity_sampling_alias=requested_granularity_sampling_alias,
        requested_run_sampling_mode=requested_run_sampling_mode,
        explicit_override_keys=explicit_override_keys,
    )
    _resolve_training_length(resolved, explicit_override_keys=explicit_override_keys)
    _resolve_parameter_reporting_defaults(resolved)
    _resolve_long_run_defaults(resolved)
    validate_run_config(resolved)
    return resolved


def resolve_all_run_configs(
    config_path: str | Path,
    overrides: Mapping[str, Any] | Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    overrides = _snapshot_overrides(overrides)
    explicit_override_keys = _override_keys(overrides)
    config = apply_overrides(load_yaml_config(config_path), overrides)
    requested_granularity_sampling_alias = _configured_granularity_sampling_alias(
        config,
        explicit_override_keys,
    )
    requested_run_sampling_mode = _configured_run_sampling_mode(config)
    shared_family_size_slug = _configured_family_size_slug(config)
    if "matrix" in config:
        if shared_family_size_slug is None:
            shared_family_size_slug = _resolve_family_size_slug(config)

    if "matrix" not in config:
        resolved = _compose_single_run(config)
        _resolve_model_variant_defaults(resolved)
        _resolve_model_correction_defaults(resolved)
        _resolve_model_dimension_and_granularity_metadata(resolved)
        resolved["run"]["family_size_slug"] = _resolve_family_size_slug(resolved)
        _resolve_naming_defaults(resolved)
        _resolve_output_paths(resolved)
        _resolve_sampling_mode_defaults(
            resolved,
            requested_granularity_sampling_alias=requested_granularity_sampling_alias,
            requested_run_sampling_mode=requested_run_sampling_mode,
            explicit_override_keys=explicit_override_keys,
        )
        _resolve_training_length(resolved, explicit_override_keys=explicit_override_keys)
        _resolve_parameter_reporting_defaults(resolved)
        _resolve_long_run_defaults(resolved)
        validate_run_config(resolved)
        return [resolved]

    runs = []
    matrix = config["matrix"]
    if isinstance(matrix.get("nested"), dict):
        runs.append(matrix["nested"])
    runs.extend(matrix.get("standalone", []))

    resolved_runs = []
    for run_entry in runs:
        resolved = _compose_matrix_run(config, run_entry)
        if shared_family_size_slug is not None:
            resolved["run"]["family_size_slug"] = shared_family_size_slug
        _resolve_model_variant_defaults(resolved)
        _resolve_model_correction_defaults(resolved)
        _resolve_model_dimension_and_granularity_metadata(resolved)
        _resolve_naming_defaults(resolved)
        _resolve_output_paths(resolved)
        _resolve_sampling_mode_defaults(
            resolved,
            requested_granularity_sampling_alias=requested_granularity_sampling_alias,
            requested_run_sampling_mode=requested_run_sampling_mode,
        )
        _resolve_training_length(resolved, explicit_override_keys=explicit_override_keys)
        _resolve_parameter_reporting_defaults(resolved)
        _resolve_long_run_defaults(resolved)
        validate_run_config(resolved)
        resolved_runs.append(resolved)

    return resolved_runs


def _configured_family_size_slug(config: Mapping[str, Any]) -> str | None:
    run = config.get("run", {})
    if not isinstance(run, Mapping):
        return None

    family_size_slug = run.get("family_size_slug")
    if not isinstance(family_size_slug, str):
        return None

    family_size_slug = family_size_slug.strip()
    if not family_size_slug:
        return None

    return family_size_slug


def _configured_granularity_sampling_alias(
    config: Mapping[str, Any],
    explicit_override_keys: set[str] | None = None,
) -> str | None:
    training = config.get("training", {})
    if not isinstance(training, Mapping):
        return None

    granularity_sampling = training.get("granularity_sampling")
    if not isinstance(granularity_sampling, str):
        return None

    granularity_sampling = granularity_sampling.strip()
    if not granularity_sampling:
        return None

    if (
        granularity_sampling == "all"
        and explicit_override_keys is not None
        and "training.granularity_sampling" not in explicit_override_keys
    ):
        return None

    return granularity_sampling


def _configured_run_sampling_mode(config: Mapping[str, Any]) -> str | None:
    run = config.get("run", {})
    if not isinstance(run, Mapping):
        return None

    sampling_mode = run.get("sampling_mode")
    if not isinstance(sampling_mode, str):
        return None

    sampling_mode = sampling_mode.strip()
    if not sampling_mode:
        return None

    return sampling_mode


def resolve_sampling_mode_from_config_sections(
    run: Mapping[str, Any],
    training: Mapping[str, Any],
) -> Any:
    if not isinstance(run, Mapping):
        run = {}
    if not isinstance(training, Mapping):
        training = {}

    if run.get("sampling_mode") is not None:
        return run["sampling_mode"]
    if run.get("model_family") == "standalone":
        return "standalone"
    granularity_sampling = training.get("granularity_sampling")
    if granularity_sampling == "random":
        return "nested-random"
    if granularity_sampling == "all":
        return "nested-all"
    return granularity_sampling


def _normalize_model_granularity_sampling_mode(raw_mode: Any) -> str:
    if not isinstance(raw_mode, str):
        raise ConfigError("model.granularity_sampling_mode must be a string")

    granularity_sampling_mode = raw_mode.strip()
    if not granularity_sampling_mode:
        raise ConfigError(
            "model.granularity_sampling_mode must be a non-empty string"
        )
    if granularity_sampling_mode == "per_block":
        granularity_sampling_mode = "per_layer"
    if granularity_sampling_mode not in VALID_MODEL_GRANULARITY_SAMPLING_MODES:
        raise ConfigError(
            "model.granularity_sampling_mode must be one of "
            f"{sorted(VALID_MODEL_GRANULARITY_SAMPLING_MODES)}"
        )

    return granularity_sampling_mode


def write_resolved_config(
    config: Mapping[str, Any],
    output_dir: str | Path | None = None,
    filename: str = "config.json",
) -> Path:
    run = config.get("run", {})
    resolved_output_dir = output_dir or run.get("output_dir")
    if resolved_output_dir is None:
        raise ConfigError("Cannot write config without run.output_dir")

    output_path = Path(resolved_output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2, sort_keys=True)
        config_file.write("\n")

    return output_path


def attach_parameter_counts_to_config(
    config: dict[str, Any],
    parameter_counts_by_granularity: Mapping[str, Mapping[str, Any]],
) -> None:
    counts_by_granularity = {
        str(granularity): copy.deepcopy(dict(counts))
        for granularity, counts in parameter_counts_by_granularity.items()
    }
    config["parameter_counts_by_granularity"] = counts_by_granularity

    selected_counts = _select_representative_parameter_counts(
        config,
        counts_by_granularity,
    )
    if selected_counts is not None:
        config["parameter_counts"] = copy.deepcopy(dict(selected_counts))

    _resolve_parameter_reporting_defaults(config)


def validate_run_config(config: Mapping[str, Any]) -> None:
    run = _require_mapping(config, "run")
    model = _require_mapping(config, "model")
    training = _require_mapping(config, "training")
    dataset = _require_mapping(config, "dataset")
    continuation = run.get("continuation")
    if not isinstance(continuation, Mapping):
        raise ConfigError("Missing mapping section: run.continuation")
    monitoring = _require_mapping(config, "monitoring")
    warmup = training.get("pre_nested_warmup")
    if not isinstance(warmup, Mapping):
        raise ConfigError("Missing mapping section: training.pre_nested_warmup")
    preset_selections = training.get("preset_selections")
    if not isinstance(preset_selections, Mapping):
        raise ConfigError("Missing mapping section: training.preset_selections")
    preset_registry_paths = training.get("preset_registry_paths")
    if not isinstance(preset_registry_paths, Mapping):
        raise ConfigError(
            "Missing mapping section: training.preset_registry_paths"
        )
    _require_mapping(config, "outputs")
    _require_mapping(config, "evaluation")

    _require_fields(
        run,
        "run",
        [
            "run_id",
            "phase_id",
            "model_family",
            "completion_label",
            "model_family_slug",
            "model_size_slug",
            "family_size_slug",
            "token_budget_slug",
            "output_group",
            "active_size_label",
            "family_resolution_rule",
            "output_root",
            "output_dir",
        ],
    )
    _require_one_of_fields(
        run,
        "run",
        ["model_shape_label", "model_size_label"],
    )
    _require_fields(
        model,
        "model",
        [
            "base_model_name",
            "variant",
            "correction_mode",
            "membership_correction",
            "granularity_sampling_mode",
            "num_layers",
            "num_attention_heads",
            "intermediate_size",
            "context_length",
            "vocab_size_assumption",
            "granularities",
        ],
    )
    _require_one_of_fields(
        model,
        "model",
        ["d_model", "hidden_size"],
    )
    _require_fields(
        training,
        "training",
        [
            "token_budget",
            "effective_world_size",
            "expected_tokens_per_step",
            "derived_max_steps",
            "max_steps",
            "base_learning_rate",
            "learning_rate_scale_rule",
            "learning_rate_scale_factor",
            "resolved_learning_rate",
            "warmup_ratio",
            "warmup_steps",
            "resolved_warmup_steps",
            "gradient_clip_norm",
            "scheduler",
            "scheduler_name",
            "scheduler_kwargs",
            "optimizer",
            "optimizer_name",
            "optimizer_kwargs",
            "preset_selections",
            "preset_registry_paths",
        ],
    )
    _require_fields(continuation, "run.continuation", ["enabled"])
    _require_fields(
        monitoring,
        "monitoring",
        [
            "enabled",
            "backend",
            "project",
            "entity",
            "group",
            "job_type",
            "name",
            "tags",
            "notes",
            "mode",
            "log_loss_by_granularity",
            "log_validation_loss",
            "log_stage_events",
        ],
    )
    _require_fields(
        warmup,
        "training.pre_nested_warmup",
        ["enabled", "duration", "unit"],
    )
    scheduler = training.get("scheduler")
    if not isinstance(scheduler, Mapping):
        raise ConfigError("Missing mapping section: training.scheduler")
    _require_fields(
        scheduler,
        "training.scheduler",
        [
            "name",
            "kwargs",
            "resolved_warmup_steps",
        ],
    )
    _require_fields(
        dataset,
        "dataset",
        ["dataset_name", "dataset_split", "dataset_phase", "preprocessing_notes"],
    )

    run_id = str(run["run_id"])
    output_dir = Path(str(run["output_dir"]))
    if output_dir.name != run_id:
        raise ConfigError(
            f"run.output_dir must end with run.run_id: {output_dir} vs {run_id}"
        )

    model_topology = run["model_family"]
    if model_topology not in VALID_MODEL_TOPOLOGIES:
        raise ConfigError(f"Unknown training topology: {model_topology}")

    completion_label = run["completion_label"]
    if completion_label not in VALID_COMPLETION_LABELS:
        raise ConfigError(f"Unknown completion label: {completion_label}")

    if run["model_family_slug"] != MODEL_FAMILY_SLUG:
        raise ConfigError(
            f"run.model_family_slug must be {MODEL_FAMILY_SLUG}: "
            f"{run['model_family_slug']}"
        )
    if not isinstance(run.get("active_size_label"), str) or not run[
        "active_size_label"
    ].strip():
        raise ConfigError("run.active_size_label must be a non-empty string")
    if not isinstance(run.get("family_size_slug"), str) or not run[
        "family_size_slug"
    ].strip():
        raise ConfigError("run.family_size_slug must be a non-empty string")
    if not isinstance(run.get("family_resolution_rule"), str) or not run[
        "family_resolution_rule"
    ].strip():
        raise ConfigError("run.family_resolution_rule must be a non-empty string")

    expected_output_group = (
        f"{run['model_family_slug']}_{run['family_size_slug']}"
        f"_{run['token_budget_slug']}"
    )
    if run["output_group"] != expected_output_group:
        raise ConfigError(
            "run.output_group must match "
            "<model_family_slug>_<family_size_slug>_<token_budget_slug>"
        )

    granularities = model["granularities"]
    if not isinstance(granularities, list) or not granularities:
        raise ConfigError("model.granularities must be a non-empty list")

    unknown_granularities = [
        granularity
        for granularity in granularities
        if granularity not in VALID_GRANULARITIES
    ]
    if unknown_granularities:
        raise ConfigError(f"Unknown granularities: {unknown_granularities}")

    if model["variant"] not in VALID_MODEL_VARIANTS:
        raise ConfigError(
            "model.variant must be one of "
            f"{sorted(VALID_MODEL_VARIANTS)}"
        )
    correction_mode = model.get("correction_mode")
    if correction_mode not in VALID_CORRECTION_MODES:
        raise ConfigError(
            "model.correction_mode must be one of "
            f"{sorted(VALID_CORRECTION_MODES)}"
        )
    if not isinstance(model.get("membership_correction"), bool):
        raise ConfigError("model.membership_correction must be a boolean")
    if model.get("correction_mode") == "lmc" and not _is_concat_model_path(config):
        raise ConfigError(
            "model.correction_mode=lmc is only valid for concat runs"
        )
    granularity_sampling_mode = model.get("granularity_sampling_mode")
    if granularity_sampling_mode not in VALID_MODEL_GRANULARITY_SAMPLING_MODES:
        raise ConfigError(
            "model.granularity_sampling_mode must be one of "
            f"{sorted(VALID_MODEL_GRANULARITY_SAMPLING_MODES)}"
        )
    requested_mode = model.get("requested_correction_mode")
    if requested_mode not in (None, ""):
        if not isinstance(requested_mode, str):
            raise ConfigError(
                "model.requested_correction_mode must be a string or null"
            )
        expected_membership_correction = requested_mode.strip() != "none"
        if model["membership_correction"] != expected_membership_correction:
            raise ConfigError(
                "model.correction_mode and model.membership_correction must not disagree"
            )

    if "d_model" in model and "hidden_size" in model:
        if _positive_int(model["d_model"], "model.d_model") != _positive_int(
            model["hidden_size"],
            "model.hidden_size",
        ):
            raise ConfigError("model.d_model must match model.hidden_size when both are set")
    if "d_model" in model:
        _positive_int(model["d_model"], "model.d_model")

    if not isinstance(continuation.get("enabled"), bool):
        raise ConfigError("run.continuation.enabled must be a boolean")

    if not isinstance(monitoring.get("enabled"), bool):
        raise ConfigError("monitoring.enabled must be a boolean")
    if monitoring.get("project") is not None and not isinstance(
        monitoring.get("project"),
        str,
    ):
        raise ConfigError("monitoring.project must be a string or null")
    if monitoring.get("entity") is not None and not isinstance(
        monitoring.get("entity"),
        str,
    ):
        raise ConfigError("monitoring.entity must be a string or null")
    if monitoring.get("group") is not None and not isinstance(
        monitoring.get("group"),
        str,
    ):
        raise ConfigError("monitoring.group must be a string or null")
    if monitoring.get("job_type") is not None and not isinstance(
        monitoring.get("job_type"),
        str,
    ):
        raise ConfigError("monitoring.job_type must be a string or null")
    if monitoring.get("name") is not None and not isinstance(
        monitoring.get("name"),
        str,
    ):
        raise ConfigError("monitoring.name must be a string or null")
    if monitoring.get("mode") is not None and not isinstance(
        monitoring.get("mode"),
        str,
    ):
        raise ConfigError("monitoring.mode must be a string or null")
    if not isinstance(monitoring.get("tags"), list):
        raise ConfigError("monitoring.tags must be a list")
    if any(not isinstance(tag, str) for tag in monitoring.get("tags", [])):
        raise ConfigError("monitoring.tags must contain only strings")
    if monitoring.get("notes") is not None and not isinstance(
        monitoring.get("notes"),
        str,
    ):
        raise ConfigError("monitoring.notes must be a string or null")
    if not isinstance(monitoring.get("log_loss_by_granularity"), bool):
        raise ConfigError("monitoring.log_loss_by_granularity must be a boolean")
    if not isinstance(monitoring.get("log_validation_loss"), bool):
        raise ConfigError("monitoring.log_validation_loss must be a boolean")
    if not isinstance(monitoring.get("log_stage_events"), bool):
        raise ConfigError("monitoring.log_stage_events must be a boolean")
    if monitoring.get("backend") not in VALID_MONITORING_BACKENDS:
        raise ConfigError(
            "monitoring.backend must be one of "
            f"{sorted(VALID_MONITORING_BACKENDS)}"
        )

    warmup_enabled = warmup.get("enabled")
    if not isinstance(warmup_enabled, bool):
        raise ConfigError("training.pre_nested_warmup.enabled must be a boolean")
    warmup_duration = _nonnegative_int(
        warmup["duration"],
        "training.pre_nested_warmup.duration",
    )
    warmup_unit = warmup.get("unit")
    if not isinstance(warmup_unit, str):
        raise ConfigError("training.pre_nested_warmup.unit must be a string")
    warmup_unit = warmup_unit.strip()
    if warmup_unit not in VALID_PRE_NESTED_WARMUP_UNITS:
        raise ConfigError(
            "training.pre_nested_warmup.unit must be one of "
            f"{sorted(VALID_PRE_NESTED_WARMUP_UNITS)}"
        )
    if warmup_enabled and warmup_duration <= 0:
        raise ConfigError(
            "training.pre_nested_warmup.duration must be positive when enabled"
        )

    if model_topology == "standalone":
        granularity = run.get("granularity")
        if granularity not in VALID_GRANULARITIES:
            raise ConfigError("standalone runs require run.granularity")
        if granularities != [granularity]:
            raise ConfigError(
                "standalone runs must resolve to exactly one matching granularity"
            )

    granularity_sampling = training.get("granularity_sampling", "all")
    if granularity_sampling not in VALID_GRANULARITY_SAMPLING:
        raise ConfigError(
            "training.granularity_sampling must be one of "
            f"{sorted(VALID_GRANULARITY_SAMPLING)}"
        )
    _validate_sampling_mode(run, granularity_sampling)

    _validate_granularity_prefix_layout(model)

    _validate_dmodel256_pilot_fields(run, model, training)

    _validate_derived_training_length(training, model)


def _compose_single_run(config: Mapping[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(config))
    resolved.pop("matrix", None)
    _apply_run_granularities(resolved)
    return resolved


def _compose_matrix_run(
    config: Mapping[str, Any],
    run_entry: Mapping[str, Any],
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for section_name in [
        "model",
        "training",
        "dataset",
        "outputs",
        "evaluation",
        "monitoring",
    ]:
        if section_name in config:
            resolved[section_name] = copy.deepcopy(config[section_name])

    run = copy.deepcopy(config.get("run", {}))
    run.update(copy.deepcopy(dict(run_entry)))
    resolved["run"] = run
    _apply_run_granularities(resolved)
    return resolved


def _resolve_model_variant_defaults(config: dict[str, Any]) -> None:
    model = config.setdefault("model", {})
    model["variant"] = _normalize_model_variant(
        model.get("variant", DEFAULT_MODEL_VARIANT)
    )
    model["membership_correction"] = _normalize_bool(
        model.get("membership_correction", model.get("gradient_membership_correction", True)),
        "model.membership_correction",
    )
    model.pop("gradient_membership_correction", None)


def _resolve_model_correction_defaults(config: dict[str, Any]) -> None:
    model = config.setdefault("model", {})
    requested_mode = model.get("correction_mode")
    if requested_mode in (None, ""):
        requested_mode = None
    elif not isinstance(requested_mode, str):
        raise ConfigError("model.correction_mode must be a string")
    else:
        requested_mode = requested_mode.strip()
        if not requested_mode:
            requested_mode = None

    membership_correction = model.get("membership_correction")
    if not isinstance(membership_correction, bool):
        raise ConfigError("model.membership_correction must be a boolean")

    if requested_mode is None:
        resolved_mode = "gmc" if membership_correction else "none"
    else:
        resolved_mode = _normalize_correction_mode(requested_mode)
        expected_membership_correction = resolved_mode != "none"
        if membership_correction != expected_membership_correction:
            raise ConfigError(
                "model.correction_mode and model.membership_correction must not disagree"
            )

    if resolved_mode == "lmc" and not _is_concat_model_path(config):
        raise ConfigError(
            "model.correction_mode=lmc is only valid for concat runs"
        )

    model["requested_correction_mode"] = requested_mode
    model["correction_mode"] = resolved_mode
    model["membership_correction"] = membership_correction


def _normalize_correction_mode(raw_mode: Any) -> str:
    if not isinstance(raw_mode, str):
        raise ConfigError("model.correction_mode must be a string")

    correction_mode = raw_mode.strip()
    if not correction_mode:
        raise ConfigError("model.correction_mode must be a non-empty string")
    if correction_mode not in VALID_CORRECTION_MODES:
        raise ConfigError(
            "model.correction_mode must be one of "
            f"{sorted(VALID_CORRECTION_MODES)}"
        )
    return correction_mode


def _is_concat_model_path(config: Mapping[str, Any]) -> bool:
    run = config.get("run", {})
    model = config.get("model", {})
    if not isinstance(run, Mapping) or not isinstance(model, Mapping):
        return False

    return run.get("model_family") == "nested" and model.get("variant") == "concat"


def _resolve_model_dimension_and_granularity_metadata(config: dict[str, Any]) -> None:
    model = config.setdefault("model", {})
    run = config.get("run", {})

    hidden_size = model.get("hidden_size")
    d_model = model.get("d_model")
    if d_model is None and hidden_size is not None:
        model["d_model"] = hidden_size
    elif d_model is not None and hidden_size is not None:
        if _positive_int(d_model, "model.d_model") != _positive_int(
            hidden_size,
            "model.hidden_size",
        ):
            raise ConfigError("model.d_model must match model.hidden_size when both are set")

    if run.get("model_family") == "standalone":
        _resolve_source_intermediate_size_from_d_model(model)
    else:
        _resolve_intermediate_size_from_d_model(model)

    granularities = model.get("granularities")
    if not isinstance(granularities, list) or not granularities:
        return

    prefixes = model.get("granularity_prefixes")
    if prefixes is None:
        prefixes = {
            granularity: (
                CANONICAL_GRANULARITY_PREFIX_FRACTIONS[granularity][0]
                / CANONICAL_GRANULARITY_PREFIX_FRACTIONS[granularity][1]
            )
            for granularity in granularities
        }
    elif not isinstance(prefixes, Mapping):
        raise ConfigError("model.granularity_prefixes must be a mapping")

    resolved_prefixes = _resolve_granularity_prefix_map(
        prefixes,
        granularities,
        model["intermediate_size"],
    )
    model["granularity_prefixes"] = resolved_prefixes
    model["ffn_prefix_metadata"] = _build_ffn_prefix_metadata(
        model["intermediate_size"],
        resolved_prefixes,
        granularities,
    )
    if model.get("variant") == "concat":
        model["ffn_concat_block_metadata"] = _build_concat_block_metadata(
            model["intermediate_size"],
            resolved_prefixes,
            granularities,
        )


def _resolve_intermediate_size_from_d_model(model: dict[str, Any]) -> None:
    d_model = model.get("d_model")
    if d_model is None:
        return

    resolved_d_model = _positive_int(d_model, "model.d_model")
    expected_intermediate_size = resolved_d_model * DEFAULT_FFN_MULTIPLIER

    if "intermediate_size" in model:
        resolved_intermediate_size = _positive_int(
            model["intermediate_size"],
            "model.intermediate_size",
        )
        if resolved_intermediate_size != expected_intermediate_size:
            raise ConfigError(
                "model.intermediate_size must equal "
                f"model.d_model * {DEFAULT_FFN_MULTIPLIER}"
            )

    model["intermediate_size"] = expected_intermediate_size


def _resolve_source_intermediate_size_from_d_model(model: dict[str, Any]) -> None:
    d_model = model.get("d_model")
    if d_model is None:
        return

    resolved_d_model = _positive_int(d_model, "model.d_model")
    expected_intermediate_size = resolved_d_model * DEFAULT_FFN_MULTIPLIER
    source_intermediate_size = model.get("matformer_source_intermediate_size")
    if source_intermediate_size is not None:
        resolved_source_intermediate_size = _positive_int(
            source_intermediate_size,
            "model.matformer_source_intermediate_size",
        )
        if resolved_source_intermediate_size != expected_intermediate_size:
            raise ConfigError(
                "model.matformer_source_intermediate_size must equal "
                f"model.d_model * {DEFAULT_FFN_MULTIPLIER}"
            )
        return

    model["matformer_source_intermediate_size"] = expected_intermediate_size


def _normalize_model_variant(raw_variant: Any) -> str:
    if not isinstance(raw_variant, str):
        raise ConfigError("model.variant must be a string")

    variant = raw_variant.strip()
    if not variant:
        raise ConfigError("model.variant must be a non-empty string")
    alias_map = {
        "matformer_llama": "slicing",
        "cat_llama": "concat",
    }
    variant = alias_map.get(variant, variant)
    if variant not in VALID_MODEL_VARIANTS:
        raise ConfigError(
            f"Unsupported model.variant={variant!r}; expected one of "
            f"{sorted(VALID_MODEL_VARIANTS)}"
        )

    return variant


def _select_matrix_run(
    config: Mapping[str, Any],
    run_id: str | None,
) -> Mapping[str, Any]:
    matrix = _require_mapping(config, "matrix")
    runs = []
    if isinstance(matrix.get("nested"), dict):
        runs.append(matrix["nested"])
    runs.extend(matrix.get("standalone", []))

    if run_id is None:
        if not runs:
            raise ConfigError("matrix config does not define any runs")
        return runs[0]

    for run_entry in runs:
        if run_entry.get("run_id") == run_id:
            return run_entry

    available_run_ids = [run_entry.get("run_id") for run_entry in runs]
    raise ConfigError(f"Unknown run_id={run_id}; available={available_run_ids}")


def _apply_run_granularities(config: dict[str, Any]) -> None:
    run = config.get("run", {})
    model = config.setdefault("model", {})

    if run.get("model_family") == "standalone" and "granularity" in run:
        _apply_standalone_fixed_width(model, run["granularity"])
        model["granularities"] = [run["granularity"]]
    elif "granularities" in run:
        model["granularities"] = list(run["granularities"])


def _model_shape_label(run: Mapping[str, Any]) -> str | None:
    label = run.get("model_shape_label", run.get("model_size_label"))
    if label is None:
        return None
    return str(label)


def _resolve_naming_defaults(config: dict[str, Any]) -> None:
    run = config.setdefault("run", {})
    model = config.setdefault("model", {})
    training = config.setdefault("training", {})

    phase_id = str(run.get("phase_id") or "")
    model_shape_label = _model_shape_label(run)
    if phase_id.startswith("debug") or model_shape_label == "debug":
        run["completion_label"] = "debug"
    else:
        run["completion_label"] = "run"

    run["model_family_slug"] = MODEL_FAMILY_SLUG
    run["model_size_slug"] = derive_model_size_slug(model)
    family_size_slug = run.get("family_size_slug")
    if not isinstance(family_size_slug, str) or not family_size_slug.strip():
        family_size_slug = run["model_size_slug"]
    else:
        family_size_slug = family_size_slug.strip()
    run["family_size_slug"] = family_size_slug
    run["token_budget_slug"] = derive_token_budget_slug(
        _positive_int(training.get("token_budget"), "training.token_budget")
    )
    run["output_group"] = (
        f"{run['model_family_slug']}_{family_size_slug}"
        f"_{run['token_budget_slug']}"
    )
    run["active_size_label"] = _resolve_active_size_label(run)
    run["family_resolution_rule"] = (
        "output_group is keyed from the largest configured family size"
    )


def _resolve_family_size_slug(config: Mapping[str, Any]) -> str:
    family_config = copy.deepcopy(dict(config))
    _resolve_model_dimension_and_granularity_metadata(family_config)
    model = family_config.get("model", {})
    if not isinstance(model, Mapping):
        raise ConfigError("model must be a mapping when resolving family size")

    family_size_source = model.get("matformer_source_intermediate_size")
    if family_size_source is not None:
        source_model = dict(model)
        source_model["intermediate_size"] = family_size_source
        source_model.pop("matformer_source_intermediate_size", None)
        family_size_slug = derive_model_size_slug(source_model)
    else:
        family_size_slug = derive_model_size_slug(model)
    if not isinstance(family_size_slug, str) or not family_size_slug.strip():
        raise ConfigError("Unable to derive family size slug from resolved model")
    return family_size_slug


def _resolve_active_size_label(run: Mapping[str, Any]) -> str:
    for field_name in ("granularity", "model_size_label", "model_shape_label"):
        value = run.get(field_name)
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
    return str(run.get("model_family", "unknown"))


def _resolve_sampling_mode_defaults(
    config: dict[str, Any],
    requested_granularity_sampling_alias: str | None = None,
    requested_run_sampling_mode: str | None = None,
    explicit_override_keys: set[str] | None = None,
) -> None:
    run = config.setdefault("run", {})
    training = config.setdefault("training", {})
    model = config.setdefault("model", {})
    if not isinstance(run, dict) or not isinstance(training, dict) or not isinstance(model, dict):
        return

    model_family = run.get("model_family")
    if model_family not in VALID_MODEL_TOPOLOGIES:
        return

    explicit_model_mode = model.get("granularity_sampling_mode")
    if explicit_model_mode in (None, ""):
        explicit_model_mode = None
    else:
        explicit_model_mode = _normalize_model_granularity_sampling_mode(
            explicit_model_mode
        )

    legacy_alias_mode = None
    if requested_granularity_sampling_alias is not None:
        legacy_alias_mode = _granularity_sampling_mode_from_legacy_alias(
            requested_granularity_sampling_alias
        )

    run_sampling_mode = None
    if requested_run_sampling_mode is not None:
        run_sampling_mode = _normalize_run_sampling_mode(requested_run_sampling_mode)
    else:
        configured_run_sampling_mode = run.get("sampling_mode")
        if configured_run_sampling_mode is not None:
            run_sampling_mode = _normalize_run_sampling_mode(
                configured_run_sampling_mode
            )

    candidate_modes: list[str] = []
    if explicit_model_mode is not None:
        candidate_modes.append(explicit_model_mode)
    if legacy_alias_mode is not None:
        candidate_modes.append(legacy_alias_mode)

    if candidate_modes:
        canonical_mode = candidate_modes[0]
        for candidate_mode in candidate_modes[1:]:
            if candidate_mode != canonical_mode:
                raise ConfigError(
                    "model.granularity_sampling_mode, training.granularity_sampling, "
                    "and run.sampling_mode conflicts"
                )
    else:
        canonical_mode = "global"

    if requested_run_sampling_mode is not None:
        derived_run_sampling_mode = run_sampling_mode
    elif legacy_alias_mode == "global":
        derived_run_sampling_mode = "nested-all"
    elif explicit_model_mode == "per_layer" or legacy_alias_mode == "per_layer":
        derived_run_sampling_mode = "nested-random"
    elif run_sampling_mode is not None:
        derived_run_sampling_mode = run_sampling_mode
    elif model_family == "standalone":
        derived_run_sampling_mode = "standalone"
    else:
        derived_run_sampling_mode = "nested-random"

    if (
        requested_run_sampling_mode is not None
        and derived_run_sampling_mode in {"nested-all", "standalone"}
        and legacy_alias_mode == "per_layer"
        and explicit_override_keys is not None
        and "run.sampling_mode" in explicit_override_keys
    ):
        raise ConfigError(
            "model.granularity_sampling_mode, training.granularity_sampling, "
            "and run.sampling_mode conflicts"
        )
    if derived_run_sampling_mode in {"nested-all", "standalone"} and canonical_mode != "global":
        raise ConfigError(
            "model.granularity_sampling_mode=per_layer requires nested runs"
        )
    if model_family == "standalone" and canonical_mode != "global":
        raise ConfigError(
            "model.granularity_sampling_mode=per_layer requires nested runs"
        )

    training_sampling = _granularity_sampling_alias_from_mode(
        _granularity_sampling_mode_from_run_sampling_mode(derived_run_sampling_mode)
    )

    run["sampling_mode"] = derived_run_sampling_mode

    training["granularity_sampling"] = training_sampling
    model["granularity_sampling_mode"] = canonical_mode
    run["resolved_run_mode"] = derived_run_sampling_mode
    model["resolved_sampling_mode"] = canonical_mode
    model["requested_granularity_sampling_alias"] = (
        requested_granularity_sampling_alias
        if requested_granularity_sampling_alias is not None
        else None
    )
    model["granularity_pattern_provenance"] = _build_granularity_pattern_provenance(
        model,
        run,
        requested_granularity_sampling_alias=requested_granularity_sampling_alias,
    )


def _normalize_run_sampling_mode(raw_mode: Any) -> str:
    if not isinstance(raw_mode, str):
        raise ConfigError("run.sampling_mode must be a string")

    sampling_mode = raw_mode.strip()
    if not sampling_mode:
        raise ConfigError("run.sampling_mode must be a non-empty string")
    return sampling_mode


def _granularity_sampling_mode_from_legacy_alias(alias: str) -> str:
    if alias not in VALID_GRANULARITY_SAMPLING:
        raise ConfigError(
            "training.granularity_sampling must be one of "
            f"{sorted(VALID_GRANULARITY_SAMPLING)}"
        )
    return {
        "all": "global",
        "random": "per_layer",
    }[alias]


def _granularity_sampling_alias_from_mode(mode: str) -> str:
    if mode not in VALID_MODEL_GRANULARITY_SAMPLING_MODES:
        raise ConfigError(
            "model.granularity_sampling_mode must be one of "
            f"{sorted(VALID_MODEL_GRANULARITY_SAMPLING_MODES)}"
        )
    return {
        "global": "all",
        "per_layer": "random",
    }[mode]


def _granularity_sampling_mode_from_run_sampling_mode(run_sampling_mode: str) -> str:
    if run_sampling_mode not in VALID_SAMPLING_MODES:
        raise ConfigError(
            f"run.sampling_mode must be one of {sorted(VALID_SAMPLING_MODES)}"
        )
    return {
        "nested-random": "per_layer",
        "nested-all": "global",
        "standalone": "global",
    }[run_sampling_mode]


def _build_granularity_pattern_provenance(
    model: Mapping[str, Any],
    run: Mapping[str, Any],
    requested_granularity_sampling_alias: str | None = None,
) -> dict[str, Any]:
    granularity_sampling_mode = model.get("granularity_sampling_mode")
    run_sampling_mode = run.get("sampling_mode")
    provenance = {
        "pattern_type": (
            "all_granularities"
            if run_sampling_mode == "nested-all"
            else (
                "single"
                if granularity_sampling_mode == "global"
                else "per_layer"
            )
        ),
        "scope": "model",
        "source": "model.granularity_sampling_mode",
        "requested_alias": requested_granularity_sampling_alias,
        "layer_count": model.get("num_layers"),
        "available_granularities": list(model.get("granularities", []))
        if isinstance(model.get("granularities"), list)
        else [],
    }
    if requested_granularity_sampling_alias is not None or run.get("granularity") is not None:
        provenance["active_granularity"] = run.get("granularity")
    return provenance


def _validate_sampling_mode(
    run: Mapping[str, Any],
    granularity_sampling: str,
) -> None:
    sampling_mode = run.get("sampling_mode")
    if sampling_mode is None:
        return

    if sampling_mode not in VALID_SAMPLING_MODES:
        raise ConfigError(
            f"run.sampling_mode must be one of {sorted(VALID_SAMPLING_MODES)}"
        )

    model_family = run["model_family"]
    if sampling_mode == "standalone":
        if model_family != "standalone":
            raise ConfigError("run.sampling_mode=standalone requires standalone")
    elif model_family != "nested":
        raise ConfigError(f"run.sampling_mode={sampling_mode} requires nested")

    expected_sampling = {
        "nested-random": "random",
        "nested-all": "all",
        "standalone": "all",
    }[sampling_mode]
    if granularity_sampling != expected_sampling:
        raise ConfigError(
            f"run.sampling_mode={sampling_mode} requires "
            f"training.granularity_sampling={expected_sampling}"
        )


def _validate_dmodel256_pilot_fields(
    run: Mapping[str, Any],
    model: Mapping[str, Any],
    training: Mapping[str, Any],
) -> None:
    model_shape_label = _model_shape_label(run)
    is_legacy_78m = str(run.get("model_size_label")) == "78m"
    is_dmodel256_pilot = model_shape_label == "dmodel256" or is_legacy_78m
    if not is_dmodel256_pilot:
        return

    if model_shape_label == "dmodel256":
        _require_fields(
            model,
            "model",
            [
                "d_model",
                "num_layers",
                "num_attention_heads",
                "context_length",
                "vocab_size_assumption",
                "granularity_prefixes",
            ],
        )
        if _positive_int(model["d_model"], "model.d_model") != 256:
            raise ConfigError("model_shape_label=dmodel256 requires model.d_model=256")


def _validate_granularity_prefix_layout(model: Mapping[str, Any]) -> None:
    granularities = model.get("granularities")
    if not isinstance(granularities, list) or not granularities:
        return

    prefixes = model.get("granularity_prefixes")
    if prefixes is None:
        raise ConfigError("model.granularity_prefixes must be a mapping")

    _positive_int(model.get("intermediate_size"), "model.intermediate_size")
    _resolve_granularity_prefix_map(
        prefixes,
        granularities,
        model["intermediate_size"],
    )


def _resolve_granularity_prefix_map(
    prefixes: Any,
    granularities: list[str],
    intermediate_size: Any,
) -> dict[str, float]:
    if not isinstance(prefixes, Mapping):
        raise ConfigError("model.granularity_prefixes must be a mapping")

    missing = [granularity for granularity in granularities if granularity not in prefixes]
    extra = sorted(str(granularity) for granularity in prefixes if granularity not in granularities)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing keys: {missing}")
        if extra:
            details.append(f"extra keys: {extra}")
        raise ConfigError(
            "model.granularity_prefixes must match model.granularities; "
            + ", ".join(details)
        )

    resolved: dict[str, float] = {}
    previous_width = 0
    resolved_intermediate_size = _positive_int(
        intermediate_size,
        "model.intermediate_size",
    )
    for granularity in granularities:
        try:
            fraction = float(prefixes[granularity])
        except (TypeError, ValueError) as error:
            raise ConfigError(
                f"model.granularity_prefixes.{granularity} must be numeric"
            ) from error
        if fraction <= 0:
            raise ConfigError(
                f"model.granularity_prefixes.{granularity} must be positive"
            )

        prefix_width = int(resolved_intermediate_size * fraction)
        if prefix_width <= 0:
            raise ConfigError(
                f"model.granularity_prefixes.{granularity} resolved to an empty width"
            )
        if prefix_width > resolved_intermediate_size:
            raise ConfigError(
                f"model.granularity_prefixes.{granularity} exceeds "
                f"model.intermediate_size={resolved_intermediate_size}"
            )
        if prefix_width <= previous_width:
            raise ConfigError(
                "model.granularity_prefixes must resolve to strictly nested widths "
                "in model.granularities order"
            )
        resolved[granularity] = fraction
        previous_width = prefix_width

    if previous_width != resolved_intermediate_size:
        last_granularity = granularities[-1]
        raise ConfigError(
            f"model.granularity_prefixes.{last_granularity} must resolve to "
            f"model.intermediate_size={resolved_intermediate_size}"
        )

    return resolved


def _build_ffn_prefix_metadata(
    intermediate_size: Any,
    granularity_prefixes: Mapping[str, Any],
    granularities: list[str],
) -> list[dict[str, Any]]:
    resolved_intermediate_size = _positive_int(
        intermediate_size,
        "model.intermediate_size",
    )
    smallest_fraction = float(granularity_prefixes[granularities[0]])
    metadata = []
    previous_prefix_width = 0
    for granularity in granularities:
        fraction = float(granularity_prefixes[granularity])
        prefix_width = int(resolved_intermediate_size * fraction)
        if prefix_width <= previous_prefix_width:
            raise ConfigError(
                "model.granularity_prefixes must resolve to strictly increasing "
                "FFN prefix widths"
            )
        metadata.append(
            {
                "name": granularity,
                "display_name": granularity.upper(),
                "ffn_ratio": fraction / smallest_fraction,
                "full_intermediate_fraction": fraction,
                "prefix_width": prefix_width,
            }
        )
        previous_prefix_width = prefix_width
    return metadata


def _build_concat_block_metadata(
    intermediate_size: Any,
    granularity_prefixes: Mapping[str, Any],
    granularities: list[str],
) -> list[dict[str, Any]]:
    prefix_metadata = _build_ffn_prefix_metadata(
        intermediate_size,
        granularity_prefixes,
        granularities,
    )
    resolved_intermediate_size = _positive_int(
        intermediate_size,
        "model.intermediate_size",
    )
    base_block_width = prefix_metadata[0]["prefix_width"]
    if base_block_width <= 0:
        raise ConfigError("model.granularity_prefixes produced an empty base block")
    if resolved_intermediate_size % base_block_width != 0:
        raise ConfigError(
            "model.intermediate_size must be divisible by the smallest FFN prefix "
            "width to build CatLlama blocks"
        )

    block_metadata = []
    previous_prefix_width = 0
    for block_index, prefix_entry in enumerate(prefix_metadata):
        prefix_width = prefix_entry["prefix_width"]
        if prefix_width % base_block_width != 0:
            raise ConfigError(
                "model.granularity_prefixes must align with CatLlama block widths"
            )
        block_width = prefix_width - previous_prefix_width
        if block_width <= 0:
            raise ConfigError(
                "model.granularity_prefixes must resolve to strictly increasing "
                "CatLlama block widths"
            )
        block_metadata.append(
            {
                "name": f"block_{block_index + 1}",
                "display_name": f"B{block_index + 1}",
                "ffn_ratio": block_width / base_block_width,
                "full_intermediate_fraction": prefix_width / resolved_intermediate_size,
                "prefix_width": prefix_width,
                "block_width": block_width,
                "cumulative_prefix_width": prefix_width,
            }
        )
        previous_prefix_width = prefix_width
    return block_metadata


def _apply_standalone_fixed_width(model: dict[str, Any], granularity: str) -> None:
    source_intermediate_size = model.get("matformer_source_intermediate_size")
    if source_intermediate_size is None:
        source_intermediate_size = model.get("intermediate_size")
    if source_intermediate_size is None:
        source_d_model = model.get("d_model", model.get("hidden_size"))
        if source_d_model is None:
            return
        source_intermediate_size = (
            _positive_int(source_d_model, "model.d_model")
            * DEFAULT_FFN_MULTIPLIER
        )
    else:
        source_intermediate_size = _positive_int(
            source_intermediate_size,
            "model.intermediate_size",
        )

    source_prefixes = model.get("granularity_prefixes")
    if source_prefixes is None:
        source_prefixes = {
            granularity_name: numerator / denominator
            for granularity_name, (numerator, denominator) in CANONICAL_GRANULARITY_PREFIX_FRACTIONS.items()
        }
    elif not isinstance(source_prefixes, Mapping):
        raise ConfigError("model.granularity_prefixes must be a mapping")

    if granularity not in source_prefixes:
        raise ConfigError(f"Unknown granularity for standalone run: {granularity}")

    source_fraction = float(source_prefixes[granularity])
    intermediate_size = int(source_intermediate_size * source_fraction)
    if intermediate_size <= 0:
        raise ConfigError(
            f"Granularity {granularity} produced empty standalone FFN width for "
            f"intermediate_size={source_intermediate_size}"
        )

    model["matformer_source_intermediate_size"] = source_intermediate_size
    model["intermediate_size"] = intermediate_size
    model["matformer_source_granularity_prefixes"] = copy.deepcopy(
        dict(source_prefixes)
    )
    model["granularity_prefixes"] = {granularity: 1.0}


def _resolve_output_paths(config: dict[str, Any]) -> None:
    run = config.setdefault("run", {})
    if "run_id" not in run:
        return

    explicit_output_dir = "output_dir" in run
    output_dir = Path(str(run["output_dir"])) if explicit_output_dir else None

    if "output_root" in run:
        output_root = Path(str(run["output_root"]))
    elif output_dir is not None:
        output_root = output_dir.parent
    else:
        output_root = Path("outputs")

    run["output_root"] = str(output_root)
    if output_dir is None:
        output_dir = output_root / str(run["output_group"]) / str(run["run_id"])
    run["output_dir"] = str(output_dir)
    run["explicit_output_dir"] = explicit_output_dir

    _ensure_writable_directory(output_root, "output root")
    if explicit_output_dir:
        _ensure_writable_directory(output_dir.parent, "output directory parent")


def _resolve_training_length(
    config: dict[str, Any],
    explicit_override_keys: set[str] | None = None,
) -> None:
    resolve_training_length_for_world_size(
        config,
        explicit_override_keys=explicit_override_keys,
    )


def _resolve_parameter_reporting_defaults(config: dict[str, Any]) -> None:
    reporting = config.setdefault("parameter_reporting", {})
    if not isinstance(reporting, dict):
        raise ConfigError("parameter_reporting must be a mapping when provided")

    reporting.setdefault("lm_head_counting", "separately_counted")


def _resolve_long_run_defaults(config: dict[str, Any]) -> None:
    _resolve_continuation_defaults(config)
    _resolve_monitoring_defaults(config)
    _resolve_pre_nested_warmup_defaults(config)


def _select_representative_parameter_counts(
    config: Mapping[str, Any],
    counts_by_granularity: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    run = config.get("run", {})
    model = config.get("model", {})

    preferred_granularity = run.get("granularity")
    if preferred_granularity in counts_by_granularity:
        return counts_by_granularity[str(preferred_granularity)]

    if "xl" in counts_by_granularity:
        return counts_by_granularity["xl"]

    for granularity in model.get("granularities", []):
        if granularity in counts_by_granularity:
            return counts_by_granularity[str(granularity)]

    return next(iter(counts_by_granularity.values()), None)


def resolve_training_length_for_world_size(
    config: dict[str, Any],
    effective_world_size: int | None = None,
    world_size_source: str | None = None,
    explicit_override_keys: set[str] | None = None,
) -> None:
    training = config.get("training")
    model = config.get("model")
    if not isinstance(training, dict) or not isinstance(model, Mapping):
        return
    if "token_budget" not in training:
        return

    token_budget = _positive_int(training["token_budget"], "training.token_budget")
    batch_size_per_process = _positive_int(
        training.get("batch_size_per_process"),
        "training.batch_size_per_process",
    )
    context_length = _positive_int(model.get("context_length"), "model.context_length")
    if effective_world_size is None:
        effective_world_size = _resolve_effective_world_size()
        if world_size_source is None:
            has_world_size = os.environ.get("WORLD_SIZE") not in (None, "")
            world_size_source = (
                "WORLD_SIZE"
                if has_world_size
                else "single_process"
            )
    else:
        effective_world_size = _positive_int(
            effective_world_size,
            "training.effective_world_size",
        )
        if world_size_source is None:
            world_size_source = "distributed_context"

    expected_tokens_per_step = (
        batch_size_per_process * context_length * effective_world_size
    )
    derived_max_steps = math.ceil(token_budget / expected_tokens_per_step)

    has_existing_derived_fields = "derived_max_steps" in training
    max_steps_cap = training.get("max_steps_cap")
    if max_steps_cap is None and not has_existing_derived_fields:
        max_steps_cap = training.get("max_steps")
    if max_steps_cap is not None:
        max_steps_cap = _positive_int(max_steps_cap, "training.max_steps_cap")

    training["token_budget"] = token_budget
    training["batch_size_per_process"] = batch_size_per_process
    training["effective_world_size"] = effective_world_size
    training["effective_world_size_source"] = world_size_source
    training["expected_tokens_per_step"] = expected_tokens_per_step
    training["derived_max_steps"] = derived_max_steps
    training["max_steps_cap"] = max_steps_cap
    training["granularity_sampling"] = training.get("granularity_sampling", "all")
    training["max_steps"] = (
        min(derived_max_steps, max_steps_cap)
        if max_steps_cap is not None
        else derived_max_steps
    )

    _resolve_training_schedule_defaults(
        training,
        effective_world_size,
        explicit_override_keys=explicit_override_keys,
    )


def _resolve_effective_world_size() -> int:
    raw_world_size = os.environ.get("WORLD_SIZE")
    if raw_world_size in (None, ""):
        return 1
    return _positive_int(raw_world_size, "WORLD_SIZE")


def _resolve_training_schedule_defaults(
    training: dict[str, Any],
    effective_world_size: int,
    explicit_override_keys: set[str] | None = None,
) -> None:
    base_learning_rate = _positive_float(
        training.get("learning_rate"),
        "training.learning_rate",
    )
    training["base_learning_rate"] = base_learning_rate

    scale_rule = _normalize_learning_rate_scale_rule(
        training.get("learning_rate_scale_rule"),
        effective_world_size,
    )
    training["learning_rate_scale_rule"] = scale_rule
    learning_rate_scale_factor = _compute_learning_rate_scale_factor(
        scale_rule,
        effective_world_size,
    )
    training["learning_rate_scale_factor"] = learning_rate_scale_factor
    training["resolved_learning_rate"] = base_learning_rate * learning_rate_scale_factor

    warmup_ratio = training.get("warmup_ratio")
    if warmup_ratio is None:
        warmup_ratio = 0.0
    warmup_ratio = _nonnegative_float(
        warmup_ratio,
        "training.warmup_ratio",
    )
    training["warmup_ratio"] = warmup_ratio

    optimizer = training.get("optimizer")
    if optimizer is None:
        optimizer = {}
    if not isinstance(optimizer, dict):
        raise ConfigError("training.optimizer must be a mapping when provided")

    optimizer, optimizer_preset_name, optimizer_preset_registry_path = (
        _resolve_training_optimizer_preset(
            optimizer,
            explicit_override_keys=explicit_override_keys,
        )
    )
    optimizer_name = _normalize_optimizer_name(optimizer.get("name", "adamw"))
    optimizer_kwargs = _resolve_optimizer_kwargs(
        optimizer_name,
        _resolve_component_kwargs(
            component=optimizer,
            component_path="training.optimizer",
            explicit_override_keys=explicit_override_keys,
        ),
    )
    training["optimizer"] = {
        "name": optimizer_name,
        "kwargs": copy.deepcopy(optimizer_kwargs),
    }
    training["optimizer_name"] = optimizer_name
    training["optimizer_kwargs"] = optimizer_kwargs
    training["preset_selections"] = (
        {"optimizer": optimizer_preset_name} if optimizer_preset_name else {}
    )
    training["preset_registry_paths"] = (
        {"optimizer": optimizer_preset_registry_path}
        if optimizer_preset_registry_path
        else {}
    )

    scheduler = training.get("scheduler")
    if scheduler is None:
        scheduler = {}
    if not isinstance(scheduler, dict):
        raise ConfigError("training.scheduler must be a mapping when provided")

    scheduler_raw_kwargs = scheduler.get("kwargs", {})
    if scheduler_raw_kwargs is None:
        scheduler_raw_kwargs = {}
    if not isinstance(scheduler_raw_kwargs, dict):
        raise ConfigError("training.scheduler.kwargs must be a mapping when provided")

    source_warmup_steps = training.get("warmup_steps")
    if source_warmup_steps is None and "warmup_steps" in scheduler_raw_kwargs:
        source_warmup_steps = scheduler_raw_kwargs["warmup_steps"]
    if source_warmup_steps is not None:
        source_warmup_steps = _nonnegative_int(
            source_warmup_steps,
            "training.warmup_steps",
        )
    training["warmup_steps"] = source_warmup_steps

    if source_warmup_steps is not None:
        resolved_warmup_steps = source_warmup_steps
    else:
        resolved_warmup_steps = math.ceil(int(training["max_steps"]) * warmup_ratio)
    training["resolved_warmup_steps"] = resolved_warmup_steps
    training["gradient_clip_norm"] = _positive_float(
        training.get("gradient_clip_norm", 1.0),
        "training.gradient_clip_norm",
    )

    scheduler_name = _normalize_scheduler_name(scheduler.get("name", "cosine"))
    scheduler_input_kwargs = copy.deepcopy(scheduler_raw_kwargs)
    scheduler_input_kwargs["warmup_steps"] = resolved_warmup_steps
    scheduler_kwargs = _resolve_scheduler_kwargs(
        scheduler_name,
        {
            key: value
            for key, value in scheduler_input_kwargs.items()
            if key != "warmup_steps"
        },
    )
    training["scheduler"] = {
        "name": scheduler_name,
        "kwargs": copy.deepcopy(scheduler_input_kwargs),
        "resolved_warmup_steps": int(resolved_warmup_steps),
    }
    training["scheduler_name"] = scheduler_name
    training["scheduler_kwargs"] = scheduler_kwargs


def _resolve_continuation_defaults(config: dict[str, Any]) -> None:
    run = config.setdefault("run", {})
    continuation = run.get("continuation")
    if continuation is None:
        continuation = {}
    if not isinstance(continuation, dict):
        raise ConfigError("run.continuation must be a mapping when provided")

    continuation["enabled"] = _normalize_bool(
        continuation.get("enabled", False),
        "run.continuation.enabled",
    )
    run["continuation"] = continuation


def _resolve_monitoring_defaults(config: dict[str, Any]) -> None:
    monitoring = config.get("monitoring")
    if monitoring is None:
        monitoring = {}
    if not isinstance(monitoring, dict):
        raise ConfigError("monitoring must be a mapping when provided")

    monitoring["enabled"] = _normalize_bool(
        monitoring.get("enabled", False),
        "monitoring.enabled",
    )
    backend = monitoring.get("backend", DEFAULT_MONITORING_BACKEND)
    if not isinstance(backend, str):
        raise ConfigError("monitoring.backend must be a string")
    backend = backend.strip()
    if backend not in VALID_MONITORING_BACKENDS:
        raise ConfigError(
            "monitoring.backend must be one of "
            f"{sorted(VALID_MONITORING_BACKENDS)}"
        )
    monitoring["backend"] = backend
    run = config.get("run", {})
    monitoring["project"] = _normalize_optional_string(
        monitoring.get("project", run.get("phase_id") or run.get("output_group"))
    )
    monitoring["entity"] = _normalize_optional_string(monitoring.get("entity"))
    monitoring["group"] = _normalize_optional_string(
        monitoring.get("group", run.get("output_group"))
    )
    monitoring["job_type"] = _normalize_optional_string(
        monitoring.get("job_type", "train")
    )
    monitoring["name"] = _normalize_optional_string(
        monitoring.get("name", run.get("run_id"))
    )
    monitoring["tags"] = _normalize_string_list(monitoring.get("tags", []))
    monitoring["notes"] = _normalize_optional_string(monitoring.get("notes"))
    monitoring["mode"] = _normalize_optional_string(monitoring.get("mode"))
    monitoring["log_loss_by_granularity"] = _normalize_bool(
        monitoring.get("log_loss_by_granularity", True),
        "monitoring.log_loss_by_granularity",
    )
    monitoring["log_validation_loss"] = _normalize_bool(
        monitoring.get("log_validation_loss", True),
        "monitoring.log_validation_loss",
    )
    monitoring["log_stage_events"] = _normalize_bool(
        monitoring.get("log_stage_events", True),
        "monitoring.log_stage_events",
    )
    config["monitoring"] = monitoring


def _resolve_pre_nested_warmup_defaults(config: dict[str, Any]) -> None:
    training = config.setdefault("training", {})
    warmup = training.get("pre_nested_warmup")
    if warmup is None:
        warmup = {}
    if not isinstance(warmup, dict):
        raise ConfigError(
            "training.pre_nested_warmup must be a mapping when provided"
        )

    warmup["enabled"] = _normalize_bool(
        warmup.get("enabled", False),
        "training.pre_nested_warmup.enabled",
    )
    warmup["duration"] = _nonnegative_int(
        warmup.get("duration", 0),
        "training.pre_nested_warmup.duration",
    )
    warmup_unit = warmup.get("unit", "epochs")
    if not isinstance(warmup_unit, str):
        raise ConfigError("training.pre_nested_warmup.unit must be a string")
    warmup_unit = warmup_unit.strip()
    if warmup_unit not in VALID_PRE_NESTED_WARMUP_UNITS:
        raise ConfigError(
            "training.pre_nested_warmup.unit must be one of "
            f"{sorted(VALID_PRE_NESTED_WARMUP_UNITS)}"
        )
    warmup["unit"] = warmup_unit
    run = config.get("run", {})
    warmup["active"] = bool(warmup["enabled"]) and run.get("model_family") == "nested"
    warmup["completed"] = bool(warmup.get("completed", False))
    warmup["completion_step"] = warmup.get("completion_step")
    warmup["transition_reason"] = warmup.get("transition_reason")
    training["pre_nested_warmup"] = warmup


def _normalize_learning_rate_scale_rule(
    raw_scale_rule: Any,
    effective_world_size: int,
) -> str:
    if raw_scale_rule is None or raw_scale_rule == "":
        return "linear" if effective_world_size > 1 else "none"
    if not isinstance(raw_scale_rule, str):
        raise ConfigError("training.learning_rate_scale_rule must be a string")

    scale_rule = raw_scale_rule.strip()
    if not scale_rule:
        return "linear" if effective_world_size > 1 else "none"
    if scale_rule not in VALID_LEARNING_RATE_SCALE_RULES:
        raise ConfigError(
            "training.learning_rate_scale_rule must be one of "
            f"{sorted(VALID_LEARNING_RATE_SCALE_RULES)}"
        )
    return scale_rule


def _compute_learning_rate_scale_factor(
    scale_rule: str,
    effective_world_size: int,
) -> float:
    if scale_rule == "none":
        return 1.0
    if scale_rule == "linear":
        return float(effective_world_size)
    if scale_rule == "sqrt":
        return math.sqrt(effective_world_size)
    raise ConfigError(
        "training.learning_rate_scale_rule must be one of "
        f"{sorted(VALID_LEARNING_RATE_SCALE_RULES)}"
    )


def _normalize_optimizer_name(raw_name: Any) -> str:
    if not isinstance(raw_name, str):
        raise ConfigError("training.optimizer.name must be a string")

    optimizer_name = raw_name.strip()
    if not optimizer_name:
        raise ConfigError("training.optimizer.name must be a non-empty string")
    if optimizer_name not in VALID_OPTIMIZER_NAMES:
        raise ConfigError(
            "training.optimizer.name must be one of "
            f"{sorted(VALID_OPTIMIZER_NAMES)}"
        )
    return optimizer_name


def _normalize_scheduler_name(raw_name: Any) -> str:
    if not isinstance(raw_name, str):
        raise ConfigError("training.scheduler.name must be a string")

    scheduler_name = raw_name.strip()
    if not scheduler_name:
        raise ConfigError("training.scheduler.name must be a non-empty string")
    return scheduler_name


def _resolve_optimizer_kwargs(
    optimizer_name: str,
    raw_kwargs: Any,
) -> dict[str, Any]:
    if raw_kwargs is None:
        raw_kwargs = {}
    if not isinstance(raw_kwargs, dict):
        raise ConfigError("training.optimizer.kwargs must be a mapping when provided")

    allowed_kwargs = OPTIMIZER_ALLOWED_KWARGS[optimizer_name]
    resolved_kwargs = copy.deepcopy(OPTIMIZER_DEFAULT_KWARGS[optimizer_name])
    for key, value in raw_kwargs.items():
        if key not in allowed_kwargs:
            raise ConfigError(
                f"training.optimizer.kwargs.{key} is not supported for {optimizer_name}"
            )
        resolved_kwargs[key] = _normalize_optimizer_kwarg(optimizer_name, key, value)

    return resolved_kwargs


def _resolve_scheduler_kwargs(
    scheduler_name: str,
    raw_kwargs: Any,
) -> dict[str, Any]:
    if raw_kwargs is None:
        raw_kwargs = {}
    if not isinstance(raw_kwargs, dict):
        raise ConfigError("training.scheduler.kwargs must be a mapping when provided")

    forbidden_keys = sorted(
        key for key in raw_kwargs if str(key) in SCHEDULER_RESERVED_KWARGS
    )
    if forbidden_keys:
        raise ConfigError(
            "training.scheduler.kwargs must not define reserved keys: "
            f"{forbidden_keys}"
        )

    return copy.deepcopy(raw_kwargs)


def _resolve_component_kwargs(
    component: Mapping[str, Any],
    component_path: str,
    explicit_override_keys: set[str] | None,
) -> dict[str, Any]:
    raw_kwargs = component.get("kwargs", {})
    if raw_kwargs is None:
        raw_kwargs = {}
    if not isinstance(raw_kwargs, dict):
        raise ConfigError(f"{component_path}.kwargs must be a mapping when provided")

    if explicit_override_keys is None:
        return copy.deepcopy(raw_kwargs)

    if f"{component_path}.kwargs" in explicit_override_keys:
        return copy.deepcopy(raw_kwargs)

    if f"{component_path}.name" not in explicit_override_keys:
        return copy.deepcopy(raw_kwargs)

    explicit_kwargs: dict[str, Any] = {}
    prefix = f"{component_path}.kwargs."
    for override_key in explicit_override_keys:
        if not override_key.startswith(prefix):
            continue
        kwarg_name = override_key[len(prefix) :]
        if not kwarg_name or "." in kwarg_name:
            continue
        if kwarg_name in raw_kwargs:
            explicit_kwargs[kwarg_name] = copy.deepcopy(raw_kwargs[kwarg_name])

    return explicit_kwargs


def _resolve_training_optimizer_preset(
    optimizer: dict[str, Any],
    explicit_override_keys: set[str] | None = None,
) -> tuple[dict[str, Any], str | None, str | None]:
    preset_name = optimizer.get("preset")
    if preset_name in (None, ""):
        return optimizer, None, None

    if not isinstance(preset_name, str):
        raise ConfigError("training.optimizer.preset must be a string")

    preset_name = preset_name.strip()
    if not preset_name:
        raise ConfigError("training.optimizer.preset must be a non-empty string")

    preset_path = PRESET_REGISTRY_ROOT / "optimizer" / f"{preset_name}.yaml"
    preset = _load_preset_registry_entry(preset_path, preset_name)

    if (
        explicit_override_keys is not None
        and "training.optimizer.name" in explicit_override_keys
    ):
        raise ConfigError(
            "training.optimizer.name cannot be overridden when "
            "training.optimizer.preset is set"
        )

    preset_optimizer_name = preset.get("name")
    configured_optimizer_name = optimizer.get("name")
    if (
        isinstance(configured_optimizer_name, str)
        and configured_optimizer_name.strip()
        and configured_optimizer_name != preset_optimizer_name
    ):
        raise ConfigError(
            "training.optimizer.preset conflicts with the effective optimizer name; "
            f"preset {preset_name!r} resolves to optimizer.name={preset_optimizer_name!r}"
        )

    merged_optimizer = _deep_merge_dicts(preset, optimizer)
    return merged_optimizer, preset_name, str(preset_path)


def _load_preset_registry_entry(
    preset_path: Path,
    preset_name: str,
) -> dict[str, Any]:
    if not preset_path.is_file():
        raise ConfigError(
            f"Unknown training.optimizer.preset={preset_name!r}; "
            f"missing registry file: {preset_path}"
        )

    preset = load_yaml_config(preset_path)
    if preset_path.stem != preset_name:
        raise ConfigError(
            f"Preset registry file name must match the preset name: {preset_path}"
        )
    _validate_preset_registry_entry(preset, preset_path)
    return preset


def _validate_preset_registry_entry(
    preset: Mapping[str, Any],
    preset_path: Path,
) -> None:
    if not isinstance(preset, Mapping):
        raise ConfigError(f"Preset registry entry must be a mapping: {preset_path}")

    preset_name = preset.get("name")
    if not isinstance(preset_name, str) or not preset_name.strip():
        raise ConfigError(f"Preset registry entry {preset_path} must define name")

    kwargs = preset.get("kwargs", {})
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, Mapping):
        raise ConfigError(f"Preset registry entry {preset_path} must define kwargs")


def _deep_merge_dicts(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _normalize_optimizer_kwarg(
    optimizer_name: str,
    key: str,
    value: Any,
) -> Any:
    if optimizer_name == "adamw" and key == "betas":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ConfigError("training.optimizer.kwargs.betas must be a pair of floats")
        return [
            _nonnegative_float(value[0], "training.optimizer.kwargs.betas[0]"),
            _nonnegative_float(value[1], "training.optimizer.kwargs.betas[1]"),
        ]

    if key in {"eps", "weight_decay", "momentum", "dampening"}:
        return _nonnegative_float(value, f"training.optimizer.kwargs.{key}")

    if key == "nesterov":
        if isinstance(value, bool):
            return value
        raise ConfigError("training.optimizer.kwargs.nesterov must be a boolean")

    raise ConfigError(f"Unsupported optimizer kwarg: {key}")


def _positive_float(value: Any, field_name: str) -> float:
    number = _coerce_float(value, field_name)
    if number <= 0:
        raise ConfigError(f"{field_name} must be a positive number")
    return number


def _nonnegative_float(value: Any, field_name: str) -> float:
    number = _coerce_float(value, field_name)
    if number < 0:
        raise ConfigError(f"{field_name} must be a non-negative number")
    return number


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{field_name} must be a non-negative integer") from error
    if parsed < 0:
        raise ConfigError(f"{field_name} must be a non-negative integer")
    return parsed


def _coerce_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{field_name} must be a number") from error


def _validate_derived_training_length(
    training: Mapping[str, Any],
    model: Mapping[str, Any],
) -> None:
    token_budget = _positive_int(training["token_budget"], "training.token_budget")
    batch_size_per_process = _positive_int(
        training["batch_size_per_process"],
        "training.batch_size_per_process",
    )
    context_length = _positive_int(model["context_length"], "model.context_length")
    effective_world_size = _positive_int(
        training["effective_world_size"],
        "training.effective_world_size",
    )
    expected_tokens_per_step = (
        batch_size_per_process * context_length * effective_world_size
    )
    if training["expected_tokens_per_step"] != expected_tokens_per_step:
        raise ConfigError(
            "training.expected_tokens_per_step must equal "
            "batch_size_per_process * context_length * effective_world_size"
        )

    derived_max_steps = math.ceil(token_budget / expected_tokens_per_step)
    if training["derived_max_steps"] != derived_max_steps:
        raise ConfigError(
            "training.derived_max_steps must equal "
            "ceil(token_budget / expected_tokens_per_step)"
        )

    max_steps = _positive_int(training["max_steps"], "training.max_steps")
    max_steps_cap = training.get("max_steps_cap")
    if max_steps_cap is not None:
        max_steps_cap = _positive_int(max_steps_cap, "training.max_steps_cap")
        expected_max_steps = min(derived_max_steps, max_steps_cap)
    else:
        expected_max_steps = derived_max_steps
    if max_steps != expected_max_steps:
        raise ConfigError("training.max_steps must match the resolved budget step count")


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{field_name} must be a positive integer") from error
    if parsed <= 0:
        raise ConfigError(f"{field_name} must be a positive integer")
    return parsed


def _normalize_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ConfigError(f"{field_name} must be a boolean")


def _normalize_optional_string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ConfigError("Expected a string or null")
    normalized = value.strip()
    return normalized or None


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError("Expected a list of strings")
    normalized = []
    for item in value:
        if not isinstance(item, str):
            raise ConfigError("Expected a list of strings")
        item = item.strip()
        if item:
            normalized.append(item)
    return normalized


def _ensure_writable_directory(path: Path, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ConfigError(f"Cannot create {label} {path}: {error}") from error

    if not path.is_dir():
        raise ConfigError(f"Resolved {label} is not a directory: {path}")

    mode_allows_write = bool(path.stat().st_mode & 0o222)
    if not mode_allows_write or not os.access(path, os.W_OK):
        raise ConfigError(f"Resolved {label} is not writable: {path}")


def _set_dotted_value(config: dict[str, Any], key: str, value: Any) -> None:
    path = key.split(".")
    current = config
    for part in path[:-1]:
        if part not in current:
            current[part] = {}
        if not isinstance(current[part], dict):
            raise ConfigError(f"Cannot set override {key}; {part} is not a mapping")
        current = current[part]

    current[path[-1]] = value


def _require_mapping(config: Mapping[str, Any], section_name: str) -> Mapping[str, Any]:
    section = config.get(section_name)
    if not isinstance(section, Mapping):
        raise ConfigError(f"Missing mapping section: {section_name}")
    return section


def _require_fields(
    section: Mapping[str, Any],
    section_name: str,
    field_names: list[str],
) -> None:
    missing_fields = [field_name for field_name in field_names if field_name not in section]
    if missing_fields:
        raise ConfigError(f"Missing {section_name} fields: {missing_fields}")


def _require_one_of_fields(
    section: Mapping[str, Any],
    section_name: str,
    field_names: list[str],
) -> None:
    if any(field_name in section for field_name in field_names):
        return
    raise ConfigError(f"Missing {section_name} field; expected one of {field_names}")
