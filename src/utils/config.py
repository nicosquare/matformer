"""Configuration helpers for MatFormer reproduction runs."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from src.utils.artifact_io import DEFAULT_ARTIFACT_IO
from src.utils.artifact_io import retry_artifact_io
from src.utils.model_size import (
    MODEL_FAMILY_SLUG,
    derive_model_size_slug,
    derive_token_budget_slug,
)
from src.utils.monitoring import DEFAULT_MONITORING_BACKEND, VALID_MONITORING_BACKENDS
from src.utils.reproducibility import (
    DATA_SPLIT_VERSION,
    SEED_STREAM_VERSION,
    build_balanced_warmup_schedule,
    seed_for,
    stable_hash,
)
from src.training.schedules import (
    WSD_SCHEDULER_NAME,
    WSD_SCHEDULE_POLICY,
    WSD_SCHEDULE_POLICY_VERSION,
)


VALID_GRANULARITIES = {"s", "m", "l", "xl"}
VALID_MODEL_TOPOLOGIES = {"nested", "standalone"}
VALID_MODEL_VARIANTS = {"slicing", "concat"}
VALID_CORRECTION_MODES = {"none", "gmc", "lmc"}
VALID_MODEL_GRANULARITY_SAMPLING_MODES = {
    "global",
    "fixed_global",
    "per_block",
    "adaptive_global",
    "adaptive_per_block",
}
VALID_GLOBAL_SAMPLING_SCHEDULES = {
    "random_with_replacement",
    "balanced_cycle",
}
BALANCED_GLOBAL_SAMPLING_SCHEDULE_VERSION = 1
VALID_ADAPTIVE_SAMPLER_STRATEGIES = {"panelgrad", "thompson", "ucb"}
PROBABILISTIC_ADAPTIVE_SAMPLING_MODES = {
    "adaptive_global",
    "adaptive_per_block",
}
BAYESIAN_CONTROLLER_METHOD_FAMILY = "bayesian_gaussian_linear_thompson"
BAYESIAN_CONTROLLER_METHOD_VERSION = 1
BAYESIAN_COVARIANCE_TOLERANCE = 1e-10
PANELGRAD_DEFAULT_IMPORTANCE_METRIC = "gradient_rms"
PANELGRAD_METHOD_FAMILIES = {
    "gradient_rms": "panelgrad_gradient_rms",
    "gradient_l2": "panelgrad_gradient_l2",
}
PANELGRAD_METHOD_FAMILY = PANELGRAD_METHOD_FAMILIES[
    PANELGRAD_DEFAULT_IMPORTANCE_METRIC
]
PANELGRAD_SCORE_DEFINITIONS = {
    "gradient_rms": "raw_aggregate_controller_gradient_rms",
    "gradient_l2": "raw_aggregate_controller_gradient_l2",
}
PANELGRAD_METHOD_VERSION = 1
PANELGRAD_RELATIVE_TOLERANCE = 1e-6
PANELGRAD_ABSOLUTE_TOLERANCE = 1e-8
VALID_LEARNING_RATE_SCALE_RULES = {"none", "linear", "sqrt"}
VALID_OPTIMIZER_NAMES = {"adamw", "sgd"}
VALID_COMPLETION_LABELS = {"debug", "run"}
VALID_GRANULARITY_SAMPLING = {"all", "random"}
VALID_PRE_NESTED_WARMUP_UNITS = {"epochs", "steps"}
VALID_PRE_NESTED_WARMUP_POLICIES = {"full_only", "balanced_global"}
VALID_CONTROLLER_RESET_POLICIES = {"full_prior", "acquisition_only"}
VALID_CONTROLLER_RESET_ACQUISITION_POLICIES = {"balanced_global"}
VALID_MIXED_PRECISION_MODES = {"none", "bf16", "fp16"}
DEFAULT_MODEL_VARIANT = "slicing"
VALID_SAMPLING_MODES = {"nested-random", "nested-all", "standalone"}
CANONICAL_GRANULARITY_ORDER = ("s", "m", "l", "xl")
CANONICAL_GRANULARITY_PREFIX_FRACTIONS = {
    "s": (1, 8),
    "m": (1, 4),
    "l": (1, 2),
    "xl": (1, 1),
}
PRODUCTION_GRANULARITY_ORDER = (
    "g125",
    "g250",
    "g375",
    "g500",
    "g625",
    "g750",
    "g875",
    "g1000",
)
PRODUCTION_GRANULARITY_PREFIXES = {
    label: (index + 1) / 8.0
    for index, label in enumerate(PRODUCTION_GRANULARITY_ORDER)
}
TINYSTORIES_CONTROLLED_DATASET_PHASES = {
    "tinystories_controlled",
    "tinystories_instruct_controlled",
}
PORTFOLIO_COMPARISON_GROUP_ID = "tinystories_instruct_portfolio_catchup_v1"
PORTFOLIO_REFERENCE_BUDGET_TOKENS = 713_785_344
PORTFOLIO_ELASTIC_BUDGET_CAP_TOKENS = 2_141_356_032
PORTFOLIO_AGGREGATE_REFERENCE_BUDGET_TOKENS = 2_855_141_376
PORTFOLIO_AGGREGATE_REFERENCE_COUNT = 4
PORTFOLIO_GRANULARITIES = ("g250", "g500", "g750", "g1000")
PORTFOLIO_COMPARISON_ROLES = {
    "standalone_reference",
    "elastic_candidate",
}
PORTFOLIO_EXTENSION_ARMS = {
    "uniform_h1_4b": {
        "budget_tokens": PORTFOLIO_AGGREGATE_REFERENCE_BUDGET_TOKENS,
        "sampling_mode": "nested-random",
    },
    "nested_all_b": {
        "budget_tokens": PORTFOLIO_REFERENCE_BUDGET_TOKENS,
        "sampling_mode": "nested-all",
    },
}
DEFAULT_FFN_MULTIPLIER = 4
CONFIG_ROOT = Path(__file__).resolve().parent.parent.parent
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
WSD_INPUT_KWARGS = {
    "decay_ratio",
    "warmup_type",
    "decay_type",
    "min_lr_ratio",
    "num_cycles",
}


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
    _validate_packed_mmap_sampling_overrides(resolved)
    _resolve_model_tokenizer_defaults(resolved)
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
    _resolve_fixed_global_sampling_distribution(resolved)
    _resolve_global_sampling_interval_steps(resolved)
    _resolve_global_sampling_schedule(resolved)
    _resolve_adaptive_sampler_defaults(resolved)
    _resolve_distributed_contract_defaults(resolved)
    _resolve_training_length(resolved, explicit_override_keys=explicit_override_keys)
    _resolve_portfolio_controlled_experiment(resolved)
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
        _validate_packed_mmap_sampling_overrides(resolved)
        _resolve_model_tokenizer_defaults(resolved)
        resolved["run"]["family_size_slug"] = _resolve_family_size_slug(resolved)
        _resolve_naming_defaults(resolved)
        _resolve_output_paths(resolved)
        _resolve_sampling_mode_defaults(
            resolved,
            requested_granularity_sampling_alias=requested_granularity_sampling_alias,
            requested_run_sampling_mode=requested_run_sampling_mode,
            explicit_override_keys=explicit_override_keys,
        )
        _resolve_fixed_global_sampling_distribution(resolved)
        _resolve_global_sampling_interval_steps(resolved)
        _resolve_global_sampling_schedule(resolved)
        _resolve_adaptive_sampler_defaults(resolved)
        _resolve_distributed_contract_defaults(resolved)
        _resolve_training_length(resolved, explicit_override_keys=explicit_override_keys)
        _resolve_portfolio_controlled_experiment(resolved)
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
        _validate_packed_mmap_sampling_overrides(resolved)
        _resolve_model_tokenizer_defaults(resolved)
        _resolve_naming_defaults(resolved)
        _resolve_output_paths(resolved)
        _resolve_sampling_mode_defaults(
            resolved,
            requested_granularity_sampling_alias=requested_granularity_sampling_alias,
            requested_run_sampling_mode=requested_run_sampling_mode,
        )
        _resolve_fixed_global_sampling_distribution(resolved)
        _resolve_global_sampling_interval_steps(resolved)
        _resolve_global_sampling_schedule(resolved)
        _resolve_adaptive_sampler_defaults(resolved)
        _resolve_distributed_contract_defaults(resolved)
        _resolve_training_length(resolved, explicit_override_keys=explicit_override_keys)
        _resolve_portfolio_controlled_experiment(resolved)
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
    if granularity_sampling_mode not in VALID_MODEL_GRANULARITY_SAMPLING_MODES:
        raise ConfigError(
            "model.granularity_sampling_mode must be one of "
            f"{sorted(VALID_MODEL_GRANULARITY_SAMPLING_MODES)}"
        )
    return granularity_sampling_mode


def _normalize_adaptive_sampler_strategy(raw_strategy: Any) -> str:
    if not isinstance(raw_strategy, str):
        raise ConfigError("model.adaptive_sampler_strategy must be a string")

    strategy = raw_strategy.strip()
    if not strategy:
        raise ConfigError(
            "model.adaptive_sampler_strategy must be a non-empty string"
        )
    if strategy not in VALID_ADAPTIVE_SAMPLER_STRATEGIES:
        raise ConfigError(
            "model.adaptive_sampler_strategy must be one of "
            f"{sorted(VALID_ADAPTIVE_SAMPLER_STRATEGIES)}"
        )
    return strategy


def _nonnegative_finite_float(value: Any, field_name: str) -> float:
    number = _nonnegative_float(value, field_name)
    if not math.isfinite(number):
        raise ConfigError(f"{field_name} must be finite")
    return number


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
    def write_attempt(_attempt: int) -> Path:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as config_file:
                temporary_path = Path(config_file.name)
                json.dump(config, config_file, indent=2, sort_keys=True)
                config_file.write("\n")
                config_file.flush()
                os.fsync(config_file.fileno())
            os.replace(temporary_path, output_path)
            return output_path
        except Exception:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    return retry_artifact_io(
        write_attempt,
        target_path=output_path,
        operation_name="resolved_config_replace",
        settings=config,
    )


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
    outputs = _require_mapping(config, "outputs")
    artifact_io = outputs.get("artifact_io")
    if not isinstance(artifact_io, Mapping):
        raise ConfigError("Missing mapping section: outputs.artifact_io")
    evaluation = _require_mapping(config, "evaluation")
    reproducibility = run.get("reproducibility")
    if not isinstance(reproducibility, Mapping):
        raise ConfigError("Missing mapping section: run.reproducibility")
    _validate_portfolio_controlled_experiment(config)

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
            "seed",
            "reproducibility",
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
            "granularity_mode",
            "correction_mode",
            "membership_correction",
            "granularity_sampling_mode",
            "num_layers",
            "num_attention_heads",
            "intermediate_size",
            "context_length",
            "vocab_size",
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
            "expected_tokens_per_microstep",
            "expected_tokens_per_step",
            "gradient_accumulation_steps",
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
    _require_fields(
        continuation,
        "run.continuation",
        ["enabled", "retain_previous_latest"],
    )
    _require_fields(
        outputs,
        "outputs",
        ["metrics_flush_interval_steps", "best_eval_retention_count", "artifact_io"],
    )
    _require_fields(
        artifact_io,
        "outputs.artifact_io",
        list(DEFAULT_ARTIFACT_IO),
    )
    _require_fields(
        evaluation,
        "evaluation",
        ["validation", "test", "gradient_interference"],
    )
    validation = evaluation.get("validation")
    if not isinstance(validation, Mapping):
        raise ConfigError("evaluation.validation must be a mapping")
    holdout = validation.get("holdout")
    if not isinstance(holdout, Mapping):
        raise ConfigError("evaluation.validation.holdout must be a mapping")
    test_evaluation = evaluation.get("test")
    if not isinstance(test_evaluation, Mapping):
        raise ConfigError("evaluation.test must be a mapping")
    _require_fields(
        validation,
        "evaluation.validation",
        [
            "enabled",
            "interval_steps",
            "interval_tokens",
            "run_at_completion",
            "holdout",
            "trailing_summary_evaluations",
        ],
    )
    _require_fields(
        holdout,
        "evaluation.validation.holdout",
        ["source", "examples"],
    )
    _require_fields(test_evaluation, "evaluation.test", ["enabled"])
    _validate_gradient_interference_configuration(config)
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

    seed = run.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError("run.seed must be an explicit nonnegative integer")
    if reproducibility.get("mode") != "strict":
        raise ConfigError("run.reproducibility.mode must be strict")
    _positive_int(
        reproducibility.get("seed_stream_version"),
        "run.reproducibility.seed_stream_version",
    )
    _positive_int(
        reproducibility.get("data_split_version"),
        "run.reproducibility.data_split_version",
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

    if any(
        not isinstance(granularity, str) or not granularity.strip()
        for granularity in granularities
    ):
        raise ConfigError(
            "model.granularities must contain only non-empty string labels"
        )
    if len(set(granularities)) != len(granularities):
        raise ConfigError("model.granularities must contain unique labels")

    granularity_mode = model.get("granularity_mode")
    if granularity_mode not in {"canonical", "explicit"}:
        raise ConfigError(
            "model.granularity_mode must be one of ['canonical', 'explicit']"
        )

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
    _resolve_fixed_global_sampling_distribution(copy.deepcopy(dict(config)))
    interval = model.get("global_sampling_interval_steps", 1)
    if (
        isinstance(interval, bool)
        or not isinstance(interval, int)
        or interval <= 0
    ):
        raise ConfigError(
            "model.global_sampling_interval_steps must be a positive integer"
        )
    if interval != 1 and (
        granularity_sampling_mode != "global"
        or run.get("sampling_mode") != "nested-random"
    ):
        raise ConfigError(
            "model.global_sampling_interval_steps is valid only for "
            "nested-random runs with model.granularity_sampling_mode=global"
        )
    global_schedule = model.get(
        "global_sampling_schedule", "random_with_replacement"
    )
    if global_schedule not in VALID_GLOBAL_SAMPLING_SCHEDULES:
        raise ConfigError(
            "model.global_sampling_schedule must be one of "
            f"{sorted(VALID_GLOBAL_SAMPLING_SCHEDULES)}"
        )
    if global_schedule == "balanced_cycle":
        if (
            granularity_sampling_mode != "global"
            or run.get("sampling_mode") != "nested-random"
        ):
            raise ConfigError(
                "model.global_sampling_schedule=balanced_cycle requires "
                "nested-random + global sampling"
            )
        if model.get("global_sampling_schedule_version") != (
            BALANCED_GLOBAL_SAMPLING_SCHEDULE_VERSION
        ):
            raise ConfigError(
                "model.global_sampling_schedule_version is invalid"
            )
        if len(granularities) < 2:
            raise ConfigError(
                "model.global_sampling_schedule=balanced_cycle requires at "
                "least two unique granularities"
            )
        if bool(warmup.get("enabled", False)):
            raise ConfigError(
                "model.global_sampling_schedule=balanced_cycle requires "
                "training.pre_nested_warmup.enabled=false"
            )
        max_steps = training.get("max_steps")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise ConfigError(
                "balanced-cycle sampling requires resolved training.max_steps"
            )
        cycle_steps = len(granularities) * interval
        if max_steps <= 0 or max_steps % cycle_steps != 0:
            raise ConfigError(
                "training.max_steps must be divisible by the number of "
                "granularities times model.global_sampling_interval_steps "
                "for balanced-cycle sampling"
            )
    requested_mode = model.get("requested_correction_mode")
    if requested_mode not in (None, ""):
        if not isinstance(requested_mode, str):
            raise ConfigError(
                "model.requested_correction_mode must be a string or null"
            )
    if model["membership_correction"] != (correction_mode != "none"):
        raise ConfigError(
            "model.membership_correction must be derived from model.correction_mode"
        )

    if granularity_sampling_mode in PROBABILISTIC_ADAPTIVE_SAMPLING_MODES:
        if run.get("sampling_mode") != "nested-random":
            raise ConfigError(
                f"model.granularity_sampling_mode={granularity_sampling_mode} requires "
                "nested-random runs"
            )
        _require_fields(model, "model", ["adaptive_sampler_strategy"])
        strategy = _normalize_adaptive_sampler_strategy(
            model["adaptive_sampler_strategy"]
        )
        if strategy == "thompson":
            _validate_resolved_bayesian_adaptive_configuration(config)
        elif strategy == "panelgrad":
            _validate_resolved_panelgrad_configuration(config)
        else:
            if granularity_sampling_mode != "adaptive_per_block":
                raise ConfigError(
                    "model.granularity_sampling_mode=adaptive_global does not "
                    "support model.adaptive_sampler_strategy=ucb"
                )
            _require_fields(
                model,
                "model",
                [
                    "adaptive_sampler_exploration_scale",
                    "adaptive_sampler_decay_rate",
                    "adaptive_sampler_reward_penalty_weight",
                ],
            )
            _nonnegative_finite_float(
                model["adaptive_sampler_exploration_scale"],
                "model.adaptive_sampler_exploration_scale",
            )
            _nonnegative_finite_float(
                model["adaptive_sampler_decay_rate"],
                "model.adaptive_sampler_decay_rate",
            )
            _nonnegative_finite_float(
                model["adaptive_sampler_reward_penalty_weight"],
                "model.adaptive_sampler_reward_penalty_weight",
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
    if not isinstance(continuation.get("retain_previous_latest"), bool):
        raise ConfigError(
            "run.continuation.retain_previous_latest must be a boolean"
        )

    _positive_int(
        outputs.get("metrics_flush_interval_steps"),
        "outputs.metrics_flush_interval_steps",
    )
    _positive_int(
        outputs.get("best_eval_retention_count"),
        "outputs.best_eval_retention_count",
    )
    _positive_int(
        artifact_io.get("max_attempts"),
        "outputs.artifact_io.max_attempts",
    )
    initial_backoff = _nonnegative_float(
        artifact_io.get("initial_backoff_seconds"),
        "outputs.artifact_io.initial_backoff_seconds",
    )
    max_backoff = _nonnegative_float(
        artifact_io.get("max_backoff_seconds"),
        "outputs.artifact_io.max_backoff_seconds",
    )
    if max_backoff < initial_backoff:
        raise ConfigError(
            "outputs.artifact_io.max_backoff_seconds must be greater than or "
            "equal to initial_backoff_seconds"
        )
    jitter_fraction = _nonnegative_float(
        artifact_io.get("jitter_fraction"),
        "outputs.artifact_io.jitter_fraction",
    )
    if jitter_fraction > 1.0:
        raise ConfigError("outputs.artifact_io.jitter_fraction must be <= 1")
    if artifact_io.get("checkpoint_staging") not in {"auto", "local", "direct"}:
        raise ConfigError(
            "outputs.artifact_io.checkpoint_staging must be one of "
            "['auto', 'direct', 'local']"
        )
    if artifact_io.get("periodic_checkpoint_failure_policy") not in {
        "continue_if_previous",
        "strict",
    }:
        raise ConfigError(
            "outputs.artifact_io.periodic_checkpoint_failure_policy must be one "
            "of ['continue_if_previous', 'strict']"
        )
    _positive_int(
        artifact_io.get("metrics_pending_row_limit"),
        "outputs.artifact_io.metrics_pending_row_limit",
    )
    if not isinstance(validation.get("enabled"), bool):
        raise ConfigError("evaluation.validation.enabled must be a boolean")
    _nonnegative_int(
        validation.get("interval_steps"),
        "evaluation.validation.interval_steps",
    )
    _nonnegative_int(
        validation.get("interval_tokens"),
        "evaluation.validation.interval_tokens",
    )
    if int(validation.get("interval_steps", 0)) > 0 and int(
        validation.get("interval_tokens", 0)
    ) > 0:
        raise ConfigError(
            "evaluation.validation.interval_tokens and a positive interval_steps "
            "are mutually exclusive"
        )
    if not isinstance(validation.get("run_at_completion"), bool):
        raise ConfigError(
            "evaluation.validation.run_at_completion must be a boolean"
        )
    if holdout.get("source") != "configured_dataset_split":
        raise ConfigError(
            "evaluation.validation.holdout.source must be configured_dataset_split"
        )
    _positive_int(
        holdout.get("examples"),
        "evaluation.validation.holdout.examples",
    )
    _positive_int(
        validation.get("trailing_summary_evaluations"),
        "evaluation.validation.trailing_summary_evaluations",
    )
    if test_evaluation.get("enabled") is not False:
        raise ConfigError("evaluation.test.enabled must be false")
    if completion_label == "run" and not validation["run_at_completion"]:
        raise ConfigError(
            "evaluation.validation.run_at_completion is required when "
            "run.completion_label=run"
        )

    requested_precision = training.get(
        "requested_mixed_precision",
        training.get("mixed_precision", "none"),
    )
    if requested_precision not in VALID_MIXED_PRECISION_MODES:
        raise ConfigError(
            "training.mixed_precision must be one of "
            f"{sorted(VALID_MIXED_PRECISION_MODES)}"
        )
    if not isinstance(training.get("requested_activation_checkpointing"), bool):
        raise ConfigError(
            "training.requested_activation_checkpointing must be a boolean"
        )

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
    warmup_policy = warmup.get("policy")
    if warmup_policy not in VALID_PRE_NESTED_WARMUP_POLICIES:
        raise ConfigError(
            "training.pre_nested_warmup.policy must be one of "
            f"{sorted(VALID_PRE_NESTED_WARMUP_POLICIES)}"
        )
    if warmup_policy == "balanced_global":
        sampling_mode = model.get("granularity_sampling_mode")
        strategy = model.get("adaptive_sampler_strategy")
        valid_thompson_warmup = (
            strategy == "thompson"
            and sampling_mode in ("adaptive_global", "adaptive_per_block")
        )
        valid_panelgrad_warmup = (
            strategy == "panelgrad" and sampling_mode == "adaptive_global"
        )
        if not (valid_thompson_warmup or valid_panelgrad_warmup):
            raise ConfigError(
                "training.pre_nested_warmup.policy=balanced_global requires a "
                "probabilistic Thompson adaptive run or adaptive_global + panelgrad"
            )
        if warmup_unit != "steps":
            raise ConfigError(
                "training.pre_nested_warmup.policy=balanced_global requires unit=steps"
            )
        interval = warmup.get("action_interval_steps")
        if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
            raise ConfigError(
                "training.pre_nested_warmup.action_interval_steps must be a "
                "positive integer"
            )
        if warmup_enabled:
            granularities = list(model.get("granularities", []))
            denominator = interval * len(granularities)
            if denominator <= 0 or warmup_duration % denominator != 0:
                raise ConfigError(
                    "training.pre_nested_warmup.duration must be divisible by "
                    "action_interval_steps * number of granularities"
                )
            passes = warmup_duration // denominator
            if passes < 2:
                raise ConfigError(
                    "training.pre_nested_warmup balanced_global requires at least "
                    "two complete passes over all granularities"
                )
            schedule = warmup.get("schedule")
            if not isinstance(schedule, list) or len(schedule) != (
                warmup_duration // interval
            ):
                raise ConfigError(
                    "training.pre_nested_warmup balanced schedule is incomplete"
                )

    if model_topology == "standalone":
        granularity = run.get("granularity")
        if not isinstance(granularity, str) or not granularity.strip():
            raise ConfigError("standalone runs require run.granularity")
        if granularity not in granularities:
            raise ConfigError(
                f"run.granularity={granularity!r} is not valid for standalone run; "
                f"available labels={granularities}"
            )
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
    _validate_distributed_and_prepared_corpus_contract(config)
    _validate_portfolio_aligned_epoch_contract(config)


def _validate_distributed_and_prepared_corpus_contract(
    config: Mapping[str, Any],
) -> None:
    training = config["training"]
    model = config["model"]
    dataset = config["dataset"]
    distributed = training.get("distributed", {})
    if not isinstance(distributed, Mapping):
        raise ConfigError("training.distributed must be a mapping")
    expected_world_size = _positive_int(
        distributed.get("expected_world_size", training["effective_world_size"]),
        "training.distributed.expected_world_size",
    )
    if expected_world_size not in {1, 2, 3, 4}:
        raise ConfigError(
            "training.distributed.expected_world_size must be between 1 and 4"
        )
    if int(training["effective_world_size"]) != expected_world_size:
        raise ConfigError(
            "training.effective_world_size must match "
            "training.distributed.expected_world_size"
        )
    if (
        expected_world_size > 1
        and model.get("granularity_sampling_mode") == "adaptive_per_block"
        and model.get("adaptive_sampler_strategy") == "ucb"
    ):
        raise ConfigError(
            "adaptive_per_block + ucb is unsupported when world size exceeds one"
        )
    diagnostic_enabled = bool(
        config.get("evaluation", {})
        .get("gradient_interference", {})
        .get("enabled", False)
    )
    if expected_world_size > 1 and (
        (
            model.get("granularity_sampling_mode") == "adaptive_global"
            and model.get("adaptive_sampler_strategy") == "panelgrad"
        )
        or diagnostic_enabled
    ):
        fsdp = distributed.get("fsdp", {})
        if not isinstance(fsdp, Mapping):
            raise ConfigError(
                "distributed controlled-gradient probing requires "
                "training.distributed.fsdp mapping"
            )
        if fsdp.get("use_orig_params", True) is not True:
            raise ConfigError(
                "distributed controlled-gradient probing requires "
                "fsdp.use_orig_params=true"
            )
        if bool(fsdp.get("cpu_offload", False)):
            raise ConfigError(
                "distributed controlled-gradient probing does not support "
                "FSDP CPU offload"
            )

    mode = dataset.get("mode", "raw_tokenized")
    optimizer_iteration_supplied = "optimizer_iteration" in dataset
    if mode != "packed_mmap":
        if optimizer_iteration_supplied:
            raise ConfigError(
                "dataset.optimizer_iteration is valid only when "
                "dataset.mode=packed_mmap"
            )
        return
    if dataset.get("sample_limit") is not None:
        raise ConfigError(
            "dataset.sample_limit is forbidden when dataset.mode=packed_mmap"
        )
    if bool(dataset.get("tokenization_keep_in_memory", False)):
        raise ConfigError(
            "in-memory tokenization is forbidden when dataset.mode=packed_mmap"
        )
    if int(dataset.get("data_seed", -1)) != 42:
        raise ConfigError("dataset.mode=packed_mmap requires dataset.data_seed=42")
    prepared_dir = dataset.get("prepared_corpus_dir")
    if not isinstance(prepared_dir, str) or not prepared_dir.strip():
        raise ConfigError(
            "dataset.mode=packed_mmap requires dataset.prepared_corpus_dir"
        )
    if config["run"]["model_family"] == "nested":
        granularities = tuple(model.get("granularities", ()))
        dataset_phase = dataset.get("dataset_phase")
        if dataset_phase in TINYSTORIES_CONTROLLED_DATASET_PHASES:
            if not granularities:
                raise ConfigError(
                    "TinyStories packed-mmap nested runs require granularities"
                )
        elif dataset_phase == "prepared_100m":
            production_subset = tuple(
                label
                for label in PRODUCTION_GRANULARITY_ORDER
                if label in granularities
            )
            if not granularities or granularities != production_subset:
                raise ConfigError(
                    "100M packed-mmap nested runs require an ordered subset of "
                    f"production granularities {list(PRODUCTION_GRANULARITY_ORDER)}"
                )
        elif granularities != PRODUCTION_GRANULARITY_ORDER:
            raise ConfigError(
                "10B packed-mmap nested runs require ordered granularities "
                f"{list(PRODUCTION_GRANULARITY_ORDER)}"
            )
        if dataset_phase not in TINYSTORIES_CONTROLLED_DATASET_PHASES:
            prefixes = model.get("granularity_prefixes", {})
            if any(
                not math.isclose(
                    float(prefixes.get(label, -1.0)),
                    PRODUCTION_GRANULARITY_PREFIXES[label],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for label in granularities
            ):
                raise ConfigError(
                    "packed-mmap runs require canonical production granularity "
                    "prefixes"
                )
    try:
        from src.training.packed_corpus import (
            REPEATED_EPOCH_ORDER_VERSION,
            load_corpus_manifest,
        )

        manifest = load_corpus_manifest(prepared_dir, verify_shards=False)
    except Exception as error:
        raise ConfigError(f"Prepared corpus validation failed: {error}") from error
    if int(manifest["context_length"]) != int(model["context_length"]):
        raise ConfigError("Prepared corpus context length does not match the model")
    if int(manifest["data_seed"]) != int(dataset["data_seed"]):
        raise ConfigError("Prepared corpus data seed does not match the run")
    token_budget = int(training["token_budget"])
    context_length = int(model["context_length"])
    if token_budget <= 0 or token_budget % context_length:
        raise ConfigError(
            "training.token_budget must be positive and divisible by "
            "model.context_length for a prepared corpus"
        )
    available_tokens = int(
        manifest["roles"]["optimizer_training"]["token_count"]
    )
    optimizer_iteration = dataset.get("optimizer_iteration", {})
    if not isinstance(optimizer_iteration, Mapping):
        raise ConfigError("dataset.optimizer_iteration must be a mapping")
    iteration_mode = optimizer_iteration.get("mode", "single_pass")
    epoch_order = optimizer_iteration.get("epoch_order", "stored_permutation")
    if iteration_mode not in {"single_pass", "repeat_epochs"}:
        raise ConfigError(
            "dataset.optimizer_iteration.mode must be single_pass or repeat_epochs"
        )
    if epoch_order not in {"stored_permutation", "deterministic_per_epoch"}:
        raise ConfigError(
            "dataset.optimizer_iteration.epoch_order must be stored_permutation "
            "or deterministic_per_epoch"
        )
    expected_order = (
        "stored_permutation"
        if iteration_mode == "single_pass"
        else "deterministic_per_epoch"
    )
    if epoch_order != expected_order:
        raise ConfigError(
            "dataset.optimizer_iteration mode/order must be "
            "single_pass+stored_permutation or "
            "repeat_epochs+deterministic_per_epoch"
        )
    if iteration_mode == "single_pass" and token_budget > available_tokens:
        raise ConfigError(
            "training.token_budget exceeds the prepared corpus optimizer tokens "
            "in single_pass mode"
        )

    available_samples = int(
        manifest["roles"]["optimizer_training"].get(
            "sequence_count", available_tokens // context_length
        )
    )
    samples_per_optimizer_step = (
        int(training["expected_tokens_per_step"]) // context_length
    )
    aligned_epoch_samples = (
        available_samples // samples_per_optimizer_step
    ) * samples_per_optimizer_step
    if aligned_epoch_samples <= 0:
        raise ConfigError(
            "Prepared corpus cannot fill one optimizer-step-aligned epoch"
        )
    planned_samples = token_budget // context_length
    complete_epochs, partial_final_epoch_samples = divmod(
        planned_samples, aligned_epoch_samples
    )
    resolved_optimizer_iteration = {
        "mode": iteration_mode,
        "epoch_order": epoch_order,
        "ordering_policy_version": (
            REPEATED_EPOCH_ORDER_VERSION
            if iteration_mode == "repeat_epochs"
            else manifest["training_order"]["permutation_version"]
        ),
        "planned_samples": planned_samples,
        "aligned_epoch_samples": aligned_epoch_samples,
        "aligned_epoch_tokens": aligned_epoch_samples * context_length,
        "excluded_tail_samples": available_samples - aligned_epoch_samples,
        "excluded_tail_tokens": (
            available_samples - aligned_epoch_samples
        ) * context_length,
        "complete_epochs": complete_epochs,
        "partial_final_epoch_samples": partial_final_epoch_samples,
        "partial_final_epoch_tokens": partial_final_epoch_samples * context_length,
        "planned_data_reuse_factor": planned_samples / aligned_epoch_samples,
        "permutation_version": manifest["training_order"]["permutation_version"],
        "permutation_hash": manifest["training_order"]["sha256"],
        "fixed_epoch_set_hash": stable_hash(
            {
                "permutation_hash": manifest["training_order"]["sha256"],
                "permutation_version": manifest["training_order"][
                    "permutation_version"
                ],
                "epoch_sample_count": aligned_epoch_samples,
            }
        ),
        "corpus_hash": manifest["corpus_hash"],
        "optimizer_training_manifest_hash": manifest["roles"][
            "optimizer_training"
        ].get(
            "manifest_hash",
            manifest.get("role_manifest_hashes", {}).get("optimizer_training"),
        ),
    }
    source = manifest["source"]
    expected_source = {
        "dataset_name": dataset.get("dataset_name"),
        "dataset_config_name": dataset.get("dataset_config_name"),
        "split": dataset.get("dataset_split"),
    }
    mismatched_source_fields = [
        field
        for field, expected in expected_source.items()
        if source.get(field) != expected
    ]
    if mismatched_source_fields:
        raise ConfigError(
            "Prepared corpus source identity does not match dataset config: "
            + ", ".join(mismatched_source_fields)
        )
    tokenizer = manifest["tokenizer"]
    if tokenizer.get("name") != model.get("tokenizer_name") or tokenizer.get(
        "revision"
    ) != model.get("tokenizer_revision"):
        raise ConfigError("Prepared corpus tokenizer identity does not match the model")
    if tokenizer.get("manifest_hash") != model.get("tokenizer_manifest_hash"):
        raise ConfigError(
            "Prepared corpus tokenizer manifest hash does not match the model"
        )
    if tokenizer.get("sentencepiece_model_sha256") != model.get(
        "tokenizer_model_sha256"
    ):
        raise ConfigError(
            "Prepared corpus tokenizer model checksum does not match the model"
        )
    if int(tokenizer.get("vocab_size", -1)) != int(model["vocab_size"]):
        raise ConfigError("Prepared corpus tokenizer vocabulary does not match the model")
    if isinstance(dataset, dict):
        dataset["optimizer_iteration"] = resolved_optimizer_iteration
        dataset["corpus_hash"] = manifest["corpus_hash"]
        dataset["role_manifest_hashes"] = dict(manifest["role_manifest_hashes"])
        dataset["available_optimizer_tokens"] = available_tokens
        dataset["selected_optimizer_samples"] = token_budget // context_length
        dataset["training_order_sha256"] = manifest["training_order"]["sha256"]
        dataset["training_order_version"] = manifest["training_order"][
            "permutation_version"
        ]


def _validate_portfolio_aligned_epoch_contract(config: Mapping[str, Any]) -> None:
    controlled = config.get("controlled_experiment", {})
    contract = (
        controlled.get("portfolio_catchup")
        if isinstance(controlled, Mapping)
        else None
    )
    if not isinstance(contract, Mapping):
        return
    iteration = config.get("dataset", {}).get("optimizer_iteration", {})
    if not isinstance(iteration, Mapping):
        raise ConfigError("Portfolio runs require a resolved optimizer epoch contract")
    if iteration.get("aligned_epoch_tokens") != PORTFOLIO_REFERENCE_BUDGET_TOKENS:
        raise ConfigError(
            "Portfolio B must equal one optimizer-step-aligned corpus epoch"
        )
    role = controlled.get("comparison_role")
    expected_budget = (
        contract.get("elastic_budget_cap_tokens")
        if role == "elastic_candidate"
        else PORTFOLIO_REFERENCE_BUDGET_TOKENS
    )
    if (
        isinstance(expected_budget, bool)
        or not isinstance(expected_budget, int)
        or expected_budget <= 0
        or expected_budget % PORTFOLIO_REFERENCE_BUDGET_TOKENS != 0
    ):
        raise ConfigError("Portfolio budget must be a positive whole number of B")
    expected_epochs = expected_budget // PORTFOLIO_REFERENCE_BUDGET_TOKENS
    if (
        iteration.get("complete_epochs") != expected_epochs
        or iteration.get("partial_final_epoch_tokens") != 0
    ):
        raise ConfigError(
            f"Portfolio {role} must resolve to exactly {expected_epochs} complete epoch(s)"
        )
    training = config.get("training", {})
    expected_steps = 87_132 * expected_epochs
    if (
        training.get("expected_tokens_per_step") != 8_192
        or training.get("derived_max_steps") != expected_steps
        or training.get("max_steps") != expected_steps
    ):
        raise ConfigError(
            "Portfolio budgets require 8,192 tokens per step and 87,132 steps per B"
        )


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
        membership_correction = resolved_mode != "none"

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
    granularity_mode = model.get("granularity_mode", "canonical")
    if not isinstance(granularity_mode, str):
        raise ConfigError("model.granularity_mode must be a string")
    granularity_mode = granularity_mode.strip().lower()
    if granularity_mode not in {"canonical", "explicit"}:
        raise ConfigError(
            "model.granularity_mode must be one of ['canonical', 'explicit']"
        )
    model["granularity_mode"] = granularity_mode

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
        if granularity_mode == "explicit":
            raise ConfigError(
                "model.granularities is required when "
                "model.granularity_mode=explicit; provide an ordered, non-empty "
                "label list"
            )
        return
    if any(
        not isinstance(granularity, str) or not granularity.strip()
        for granularity in granularities
    ):
        raise ConfigError(
            "model.granularities must contain only non-empty string labels"
        )
    if len(set(granularities)) != len(granularities):
        raise ConfigError("model.granularities must contain unique labels")

    prefixes = model.get("granularity_prefixes")
    if prefixes is None:
        if granularity_mode == "explicit":
            raise ConfigError(
                "model.granularity_prefixes is required when "
                "model.granularity_mode=explicit; provide one prefix fraction "
                "for each model.granularities label"
            )
        try:
            prefixes = {
                granularity: (
                    CANONICAL_GRANULARITY_PREFIX_FRACTIONS[granularity][0]
                    / CANONICAL_GRANULARITY_PREFIX_FRACTIONS[granularity][1]
                )
                for granularity in granularities
            }
        except KeyError as error:
            raise ConfigError(
                "model.granularity_prefixes is required for non-canonical labels; "
                f"provide entries for model.granularities={granularities}"
            ) from error
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


def _resolve_model_tokenizer_defaults(config: dict[str, Any]) -> None:
    """Resolve an immutable local tokenizer while retaining historical Hub fields."""

    model = config.setdefault("model", {})
    tokenizer_dir = model.get("tokenizer_dir")
    if tokenizer_dir in (None, ""):
        return
    if not isinstance(tokenizer_dir, str):
        raise ConfigError("model.tokenizer_dir must be a string")
    try:
        from src.training.fineweb_tokenizer import load_tokenizer_manifest

        manifest = load_tokenizer_manifest(tokenizer_dir, verify_files=True)
    except Exception as error:
        raise ConfigError(f"Local tokenizer validation failed: {error}") from error
    configured_vocab_size = _positive_int(
        model.get("vocab_size"), "model.vocab_size"
    )
    if int(manifest["vocab_size"]) != configured_vocab_size:
        raise ConfigError(
            "Local tokenizer vocabulary must equal model.vocab_size"
        )
    model["tokenizer_dir"] = str(Path(tokenizer_dir).expanduser().resolve())
    model["tokenizer_name"] = manifest["tokenizer_name"]
    model["tokenizer_revision"] = manifest["manifest_hash"]
    model["tokenizer_manifest_hash"] = manifest["manifest_hash"]
    model["tokenizer_model_sha256"] = manifest["sentencepiece_model_sha256"]


def _validate_packed_mmap_sampling_overrides(config: Mapping[str, Any]) -> None:
    """Reject per-run sampling before resolving external corpus dependencies."""

    dataset = config.get("dataset", {})
    if (
        isinstance(dataset, Mapping)
        and dataset.get("mode") == "packed_mmap"
        and dataset.get("sample_limit") is not None
    ):
        raise ConfigError(
            "dataset.sample_limit is forbidden when dataset.mode=packed_mmap"
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
        configured_granularities = model.get("granularities")
        granularity_mode = str(model.get("granularity_mode", "canonical")).strip().lower()
        if granularity_mode == "explicit" and (
            not isinstance(configured_granularities, list)
            or not configured_granularities
        ):
            raise ConfigError(
                "model.granularities is required when "
                "model.granularity_mode=explicit; provide an ordered, non-empty "
                "label list before selecting run.granularity"
            )
        if (
            isinstance(configured_granularities, list)
            and configured_granularities
            and run["granularity"] not in configured_granularities
        ):
            raise ConfigError(
                f"run.granularity={run['granularity']!r} is not valid for "
                f"standalone run; available labels={configured_granularities}"
            )
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

    if explicit_model_mode is not None:
        canonical_mode = explicit_model_mode
    elif legacy_alias_mode is not None:
        canonical_mode = legacy_alias_mode
    else:
        canonical_mode = "global"

    if requested_run_sampling_mode is not None:
        derived_run_sampling_mode = run_sampling_mode
    elif legacy_alias_mode == "global":
        derived_run_sampling_mode = "nested-all"
    elif canonical_mode in {
        "fixed_global",
        "per_block",
        "adaptive_global",
        "adaptive_per_block",
    } or legacy_alias_mode == "per_block":
        derived_run_sampling_mode = "nested-random"
    elif run_sampling_mode is not None:
        derived_run_sampling_mode = run_sampling_mode
    elif model_family == "standalone":
        derived_run_sampling_mode = "standalone"
    else:
        derived_run_sampling_mode = "nested-random"

    def _raise_granularity_sampling_conflict(requirement: str) -> None:
        raise ConfigError(
            f"model.granularity_sampling_mode={canonical_mode} conflicts with {requirement}"
        )

    if (
        canonical_mode in PROBABILISTIC_ADAPTIVE_SAMPLING_MODES
        and derived_run_sampling_mode != "nested-random"
    ):
        _raise_granularity_sampling_conflict("nested-random runs")
    if (
        canonical_mode == "fixed_global"
        and derived_run_sampling_mode != "nested-random"
    ):
        _raise_granularity_sampling_conflict("nested-random runs")
    if (
        canonical_mode in PROBABILISTIC_ADAPTIVE_SAMPLING_MODES
        and requested_granularity_sampling_alias is not None
        and requested_granularity_sampling_alias != "random"
    ):
        _raise_granularity_sampling_conflict("nested-random runs")
    if canonical_mode == "per_block" and derived_run_sampling_mode != "nested-random":
        _raise_granularity_sampling_conflict("nested runs")
    if derived_run_sampling_mode in {"nested-all", "standalone"} and canonical_mode != "global":
        _raise_granularity_sampling_conflict(
            "nested-random runs"
            if canonical_mode in PROBABILISTIC_ADAPTIVE_SAMPLING_MODES
            else "nested runs"
        )
    if model_family == "standalone" and canonical_mode != "global":
        _raise_granularity_sampling_conflict(
            "nested-random runs"
            if canonical_mode in PROBABILISTIC_ADAPTIVE_SAMPLING_MODES
            else "nested runs"
        )
    if requested_run_sampling_mode is not None and requested_granularity_sampling_alias is not None:
        expected_alias = _granularity_sampling_alias_from_mode(
            _granularity_sampling_mode_from_run_sampling_mode(derived_run_sampling_mode)
        )
        if requested_granularity_sampling_alias != expected_alias:
            raise ConfigError(
                "run.sampling_mode conflicts with training.granularity_sampling; "
                f"run.sampling_mode={derived_run_sampling_mode} requires "
                f"training.granularity_sampling={expected_alias}"
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


def _resolve_fixed_global_sampling_distribution(config: dict[str, Any]) -> None:
    """Validate and canonicalize the opt-in fixed global categorical policy."""

    model = config.get("model")
    run = config.get("run")
    if not isinstance(model, dict) or not isinstance(run, Mapping):
        return

    sampling_mode = model.get("granularity_sampling_mode")
    raw_distribution = model.get("global_sampling_distribution")
    if sampling_mode != "fixed_global":
        if raw_distribution is not None:
            raise ConfigError(
                "model.global_sampling_distribution requires "
                "model.granularity_sampling_mode=fixed_global"
            )
        return

    if not isinstance(raw_distribution, Mapping) or not raw_distribution:
        raise ConfigError(
            "model.global_sampling_distribution must be a non-empty mapping "
            "when model.granularity_sampling_mode=fixed_global"
        )

    granularities = [str(label) for label in model.get("granularities", [])]
    if any(not isinstance(label, str) for label in raw_distribution):
        raise ConfigError(
            "model.global_sampling_distribution keys must be granularity labels"
        )
    configured_labels = set(raw_distribution)
    expected_labels = set(granularities)
    if configured_labels != expected_labels:
        missing = sorted(expected_labels - configured_labels)
        extra = sorted(configured_labels - expected_labels)
        raise ConfigError(
            "model.global_sampling_distribution keys must exactly match "
            f"model.granularities; missing={missing}, extra={extra}"
        )

    distribution: dict[str, float] = {}
    for label in granularities:
        probability = raw_distribution[label]
        if isinstance(probability, bool):
            raise ConfigError(
                f"model.global_sampling_distribution.{label} must be a finite "
                "nonnegative number"
            )
        distribution[label] = _nonnegative_finite_float(
            probability,
            f"model.global_sampling_distribution.{label}",
        )

    total = math.fsum(distribution.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ConfigError(
            "model.global_sampling_distribution probabilities must sum to 1; "
            f"got {total:.17g}"
        )
    if len(distribution) < 2 or len(set(distribution.values())) == 1:
        raise ConfigError(
            "model.global_sampling_distribution must be non-uniform; use "
            "model.granularity_sampling_mode=global for uniform sampling"
        )

    model["global_sampling_distribution"] = distribution
    model["granularity_pattern_provenance"] = _build_granularity_pattern_provenance(
        model,
        run,
        requested_granularity_sampling_alias=model.get(
            "requested_granularity_sampling_alias"
        ),
    )


def _resolve_global_sampling_interval_steps(config: dict[str, Any]) -> None:
    """Resolve the uniform-global action hold interval without widening its scope."""

    model = config.get("model")
    run = config.get("run")
    if not isinstance(model, dict) or not isinstance(run, Mapping):
        return

    explicitly_configured = "global_sampling_interval_steps" in model
    value = model.get("global_sampling_interval_steps", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(
            "model.global_sampling_interval_steps must be a positive integer"
        )

    eligible = (
        model.get("granularity_sampling_mode") == "global"
        and run.get("sampling_mode") == "nested-random"
    )
    if explicitly_configured and not eligible:
        raise ConfigError(
            "model.global_sampling_interval_steps is valid only for "
            "nested-random runs with model.granularity_sampling_mode=global"
        )

    model["global_sampling_interval_steps"] = int(value)
    provenance = model.get("granularity_pattern_provenance")
    if isinstance(provenance, dict):
        if eligible:
            provenance["global_sampling_interval_steps"] = int(value)
        else:
            provenance.pop("global_sampling_interval_steps", None)


def _resolve_global_sampling_schedule(config: dict[str, Any]) -> None:
    """Resolve the opt-in global schedule without changing IID defaults."""

    model = config.get("model")
    run = config.get("run")
    if not isinstance(model, dict) or not isinstance(run, Mapping):
        return

    explicitly_configured = "global_sampling_schedule" in model
    raw_schedule = model.get(
        "global_sampling_schedule", "random_with_replacement"
    )
    if not isinstance(raw_schedule, str):
        raise ConfigError(
            "model.global_sampling_schedule must be one of "
            f"{sorted(VALID_GLOBAL_SAMPLING_SCHEDULES)}"
        )
    schedule = raw_schedule.strip().lower()
    if schedule not in VALID_GLOBAL_SAMPLING_SCHEDULES:
        raise ConfigError(
            "model.global_sampling_schedule must be one of "
            f"{sorted(VALID_GLOBAL_SAMPLING_SCHEDULES)}"
        )

    eligible = (
        model.get("granularity_sampling_mode") == "global"
        and run.get("sampling_mode") == "nested-random"
    )
    if explicitly_configured and not eligible:
        raise ConfigError(
            "model.global_sampling_schedule is valid only for nested-random "
            "runs with model.granularity_sampling_mode=global"
        )

    model["global_sampling_schedule"] = schedule
    model["global_sampling_schedule_version"] = (
        BALANCED_GLOBAL_SAMPLING_SCHEDULE_VERSION
        if schedule == "balanced_cycle"
        else None
    )
    provenance = model.get("granularity_pattern_provenance")
    if isinstance(provenance, dict) and eligible:
        if schedule == "balanced_cycle":
            provenance["global_sampling_schedule"] = schedule
            provenance["global_sampling_schedule_version"] = (
                BALANCED_GLOBAL_SAMPLING_SCHEDULE_VERSION
            )
        else:
            provenance.pop("global_sampling_schedule", None)
            provenance.pop("global_sampling_schedule_version", None)


def _resolve_adaptive_sampler_defaults(config: dict[str, Any]) -> None:
    model = config.setdefault("model", {})
    if not isinstance(model, dict):
        return

    sampling_mode = model.get("granularity_sampling_mode")
    adaptive_fields_requested = any(
        field in model
        for field in (
            "adaptive_sampler_strategy",
            "adaptive_sampler_exploration_scale",
            "adaptive_sampler_decay_rate",
            "adaptive_sampler_reward_penalty_weight",
            "adaptive_controller",
            "panelgrad",
        )
    )
    if (
        sampling_mode not in PROBABILISTIC_ADAPTIVE_SAMPLING_MODES
        and not adaptive_fields_requested
    ):
        return

    default_strategy = "panelgrad" if "panelgrad" in model else "thompson"
    strategy = model.get("adaptive_sampler_strategy", default_strategy)
    strategy = _normalize_adaptive_sampler_strategy(strategy)
    model["adaptive_sampler_strategy"] = strategy

    if strategy == "panelgrad":
        if sampling_mode != "adaptive_global":
            raise ConfigError(
                "model.adaptive_sampler_strategy=panelgrad requires "
                "model.granularity_sampling_mode=adaptive_global"
            )
        _resolve_panelgrad_configuration(config)
        return

    if "panelgrad" in model:
        raise ConfigError(
            "model.panelgrad is valid only when "
            "model.adaptive_sampler_strategy=panelgrad"
        )

    if sampling_mode == "adaptive_global" and strategy == "ucb":
        raise ConfigError(
            "model.granularity_sampling_mode=adaptive_global does not support "
            "model.adaptive_sampler_strategy=ucb"
        )

    if sampling_mode in PROBABILISTIC_ADAPTIVE_SAMPLING_MODES and strategy == "thompson":
        _resolve_bayesian_adaptive_configuration(config)
        return

    _resolve_legacy_adaptive_sampler_defaults(model, strategy)


def _resolve_panelgrad_configuration(config: dict[str, Any]) -> None:
    model = config["model"]
    evaluation = config.setdefault("evaluation", {})
    raw_panelgrad = model.get("panelgrad", {})
    if not isinstance(raw_panelgrad, Mapping):
        raise ConfigError("model.panelgrad must be a mapping")
    panelgrad = copy.deepcopy(dict(raw_panelgrad))

    allowed_fields = {
        "refresh_interval_steps",
        "eta",
        "temperature",
        "epsilon",
        "epsilon_schedule",
        "importance_metric",
        "method_family",
        "method_version",
        "scope",
        "score",
        "support",
        "probability_mapping",
        "action_distribution",
        "relative_tolerance",
        "absolute_tolerance",
        "inverse_probability_weighting",
        "compute_correction",
        "ordered_granularities",
        "controlled_support_counts",
        "controlled_support_hash",
        "controller_panel_contract",
        "final_holdout_contract",
        "sampling_seed_stream",
        "sampling_seed",
    }
    unknown_fields = sorted(set(panelgrad) - allowed_fields)
    if unknown_fields:
        raise ConfigError(f"Unknown model.panelgrad fields: {unknown_fields}")

    incompatible_model_fields = sorted(
        field_name
        for field_name in (
            "adaptive_controller",
            "adaptive_sampler_exploration_scale",
            "adaptive_sampler_decay_rate",
            "adaptive_sampler_reward_penalty_weight",
        )
        if field_name in model
    )
    if incompatible_model_fields:
        raise ConfigError(
            "PanelGrad cannot mix Bayesian or UCB configuration fields: "
            f"{incompatible_model_fields}"
        )

    granularities = model.get("granularities")
    if not isinstance(granularities, list) or not granularities:
        raise ConfigError(
            "PanelGrad requires model.granularities to be a non-empty list"
        )
    if len(set(granularities)) != len(granularities):
        raise ConfigError("PanelGrad requires unique model.granularities")

    panelgrad["refresh_interval_steps"] = _strict_positive_int(
        panelgrad.get("refresh_interval_steps", 50),
        "model.panelgrad.refresh_interval_steps",
    )
    eta = _finite_float(panelgrad.get("eta", 1e-12), "model.panelgrad.eta")
    if eta <= 0.0:
        raise ConfigError("model.panelgrad.eta must be positive")
    panelgrad["eta"] = eta
    temperature = _finite_float(
        panelgrad.get("temperature", 1.0),
        "model.panelgrad.temperature",
    )
    if temperature <= 0.0:
        raise ConfigError("model.panelgrad.temperature must be positive")
    panelgrad["temperature"] = temperature
    importance_metric = panelgrad.get(
        "importance_metric", PANELGRAD_DEFAULT_IMPORTANCE_METRIC
    )
    if (
        not isinstance(importance_metric, str)
        or importance_metric not in PANELGRAD_METHOD_FAMILIES
    ):
        raise ConfigError(
            "model.panelgrad.importance_metric must be one of "
            f"{sorted(PANELGRAD_METHOD_FAMILIES)}"
        )
    panelgrad["importance_metric"] = importance_metric
    has_epsilon = "epsilon" in panelgrad
    has_epsilon_schedule = "epsilon_schedule" in panelgrad
    if has_epsilon and has_epsilon_schedule:
        raise ConfigError(
            "model.panelgrad accepts either epsilon or epsilon_schedule, not both"
        )
    if has_epsilon_schedule:
        raw_schedule = panelgrad["epsilon_schedule"]
        if not isinstance(raw_schedule, Mapping):
            raise ConfigError("model.panelgrad.epsilon_schedule must be a mapping")
        schedule = copy.deepcopy(dict(raw_schedule))
        unknown_schedule_fields = sorted(
            set(schedule) - {"type", "start", "end", "duration_steps"}
        )
        if unknown_schedule_fields:
            raise ConfigError(
                "Unknown model.panelgrad.epsilon_schedule fields: "
                f"{unknown_schedule_fields}"
            )
        schedule_type = schedule.get("type")
        if schedule_type != "linear":
            raise ConfigError(
                "model.panelgrad.epsilon_schedule.type must be 'linear'"
            )
        for endpoint in ("start", "end"):
            value = _finite_float(
                schedule.get(endpoint),
                f"model.panelgrad.epsilon_schedule.{endpoint}",
            )
            if value < 0.0 or value > 1.0:
                raise ConfigError(
                    f"model.panelgrad.epsilon_schedule.{endpoint} must be "
                    "between zero and one"
                )
            schedule[endpoint] = value
        schedule["duration_steps"] = _strict_positive_int(
            schedule.get("duration_steps"),
            "model.panelgrad.epsilon_schedule.duration_steps",
        )
        panelgrad["epsilon_schedule"] = schedule
    else:
        epsilon = _finite_float(
            panelgrad.get("epsilon", 0.1),
            "model.panelgrad.epsilon",
        )
        if epsilon < 0.0 or epsilon > 1.0:
            raise ConfigError("model.panelgrad.epsilon must be between zero and one")
        panelgrad["epsilon"] = epsilon

    fixed_fields = {
        "method_family": PANELGRAD_METHOD_FAMILIES[importance_metric],
        "method_version": PANELGRAD_METHOD_VERSION,
        "scope": "global",
        "score": PANELGRAD_SCORE_DEFINITIONS[importance_metric],
        "support": "granularity_controlled_ffn",
        "probability_mapping": "powered_score_uniform_mixture",
        "action_distribution": "categorical",
        "relative_tolerance": PANELGRAD_RELATIVE_TOLERANCE,
        "absolute_tolerance": PANELGRAD_ABSOLUTE_TOLERANCE,
        "inverse_probability_weighting": False,
        "compute_correction": False,
    }
    for field_name, expected_value in fixed_fields.items():
        if field_name in panelgrad and panelgrad[field_name] != expected_value:
            raise ConfigError(
                f"model.panelgrad.{field_name} must be {expected_value!r}"
            )
        panelgrad[field_name] = expected_value

    ordered_granularities = list(granularities)
    if "ordered_granularities" in panelgrad and panelgrad[
        "ordered_granularities"
    ] != ordered_granularities:
        raise ConfigError(
            "model.panelgrad.ordered_granularities must match model.granularities"
        )
    panelgrad["ordered_granularities"] = ordered_granularities
    support_counts = panelgrad.get("controlled_support_counts", "pending")
    if support_counts != "pending":
        if not isinstance(support_counts, Mapping) or list(support_counts) != (
            ordered_granularities
        ):
            raise ConfigError(
                "model.panelgrad.controlled_support_counts must follow "
                "model.granularities"
            )
        if any(
            isinstance(support_counts[label], bool)
            or not isinstance(support_counts[label], int)
            or support_counts[label] <= 0
            for label in ordered_granularities
        ):
            raise ConfigError(
                "model.panelgrad.controlled_support_counts must be positive integers"
            )
        support_counts = dict(support_counts)
    support_hash = panelgrad.get("controlled_support_hash", "pending")
    if support_hash != "pending" and (
        not isinstance(support_hash, str) or not support_hash.strip()
    ):
        raise ConfigError(
            "model.panelgrad.controlled_support_hash must be pending or non-empty"
        )
    panelgrad["controlled_support_counts"] = support_counts
    panelgrad["controlled_support_hash"] = support_hash
    panelgrad["sampling_seed_stream"] = "panelgrad_sampling"
    panelgrad["sampling_seed"] = seed_for(config, "panelgrad_sampling")

    controller_role = _resolve_fixed_controller_data_role(
        evaluation,
        "adaptive_controller",
        {
            "enabled": True,
            "source": "configured_dataset_split",
            "examples": 128,
            "objective_weights": "uniform",
            "fixed_manifest": True,
        },
        method_name="PanelGrad",
    )
    final_holdout_role = _resolve_fixed_controller_data_role(
        evaluation,
        "final_holdout",
        {
            "enabled": True,
            "source": "configured_dataset_split",
            "examples": 512,
            "fixed_manifest": True,
            "evaluate_during_training": False,
        },
        method_name="PanelGrad",
    )
    controller_role.setdefault("manifest_hash", "pending")
    final_holdout_role.setdefault("manifest_hash", "pending")
    panelgrad["controller_panel_contract"] = copy.deepcopy(controller_role)
    panelgrad["final_holdout_contract"] = copy.deepcopy(final_holdout_role)

    model["panelgrad"] = panelgrad
    evaluation["adaptive_controller"] = controller_role
    evaluation["final_holdout"] = final_holdout_role


def _resolve_legacy_adaptive_sampler_defaults(
    model: dict[str, Any],
    strategy: str,
) -> None:
    exploration_scale = model.get("adaptive_sampler_exploration_scale", 1.0)
    decay_rate = model.get("adaptive_sampler_decay_rate", 0.0)
    reward_penalty_weight = model.get(
        "adaptive_sampler_reward_penalty_weight",
        1.0,
    )

    model["adaptive_sampler_strategy"] = strategy
    model["adaptive_sampler_exploration_scale"] = _nonnegative_finite_float(
        exploration_scale,
        "model.adaptive_sampler_exploration_scale",
    )
    model["adaptive_sampler_decay_rate"] = _nonnegative_finite_float(
        decay_rate,
        "model.adaptive_sampler_decay_rate",
    )
    model["adaptive_sampler_reward_penalty_weight"] = _nonnegative_finite_float(
        reward_penalty_weight,
        "model.adaptive_sampler_reward_penalty_weight",
    )


def _resolve_bayesian_adaptive_configuration(config: dict[str, Any]) -> None:
    model = config["model"]
    sampling_mode = model["granularity_sampling_mode"]

    legacy_fields = [
        field_name
        for field_name in (
            "adaptive_sampler_exploration_scale",
            "adaptive_sampler_decay_rate",
            "adaptive_sampler_reward_penalty_weight",
        )
        if field_name in model
    ]
    if legacy_fields:
        raise ConfigError(
            "Thompson migration to the Bayesian controller cannot mix legacy "
            f"adaptive sampler fields: {legacy_fields}"
        )

    raw_controller = model.get("adaptive_controller")
    if not isinstance(raw_controller, Mapping):
        raise ConfigError(
            "Thompson migration requires an explicit Bayesian mapping at "
            "model.adaptive_controller"
        )
    controller = copy.deepcopy(dict(raw_controller))
    controller = _resolve_bayesian_controller_preset(config, controller)
    evaluation = config.setdefault("evaluation", {})
    required_controller_fields = (
        "prior_mean",
        "prior_covariance",
        "observation_noise_variance",
        "process_noise_covariance",
    )
    missing_controller_fields = [
        field_name
        for field_name in required_controller_fields
        if field_name not in controller
    ]
    if missing_controller_fields:
        raise ConfigError(
            "Thompson migration requires explicit Bayesian controller fields: "
            f"{missing_controller_fields}"
        )

    granularities = model.get("granularities")
    if not isinstance(granularities, list) or not granularities:
        raise ConfigError(
            "Bayesian Thompson requires model.granularities to be a non-empty list"
        )
    block_count = _strict_positive_int(
        model.get("num_layers"),
        "model.num_layers",
    )
    scope = "global" if sampling_mode == "adaptive_global" else "per_block"
    feature_model = "arms" if scope == "global" else "additive"
    coefficient_dimension = (
        len(granularities)
        if scope == "global"
        else 1 + block_count * (len(granularities) - 1)
    )

    controller["decision_interval_steps"] = _strict_positive_int(
        controller.get("decision_interval_steps", 50),
        "model.adaptive_controller.decision_interval_steps",
    )
    resolved_prior_mean = _resolve_bayesian_mean(
        controller["prior_mean"],
        coefficient_dimension,
        "model.adaptive_controller.prior_mean",
    )
    resolved_prior_covariance = _resolve_bayesian_covariance(
        controller["prior_covariance"],
        coefficient_dimension,
        "model.adaptive_controller.prior_covariance",
    )
    observation_noise_variance = _finite_float(
        controller["observation_noise_variance"],
        "model.adaptive_controller.observation_noise_variance",
    )
    if observation_noise_variance <= 0.0:
        raise ConfigError(
            "model.adaptive_controller.observation_noise_variance must be positive"
        )
    resolved_process_noise_covariance = _resolve_bayesian_covariance(
        controller["process_noise_covariance"],
        coefficient_dimension,
        "model.adaptive_controller.process_noise_covariance",
    )

    fixed_fields = {
        "strategy": "thompson",
        "scope": scope,
        "feature_model": feature_model,
        "context_model": "intercept_only",
        "transition_model": "identity",
        "compute_weight": 0.0,
        "switch_weight": 0.0,
        "method_family": BAYESIAN_CONTROLLER_METHOD_FAMILY,
        "method_version": BAYESIAN_CONTROLLER_METHOD_VERSION,
    }
    for field_name, expected_value in fixed_fields.items():
        if field_name in controller and controller[field_name] != expected_value:
            if field_name in {"compute_weight", "switch_weight"}:
                raise ConfigError(
                    f"model.adaptive_controller.{field_name} must be zero"
                )
            raise ConfigError(
                f"model.adaptive_controller.{field_name} must be "
                f"{expected_value!r}"
            )
        controller[field_name] = expected_value

    controller["coefficient_dimension"] = coefficient_dimension
    controller["block_count"] = block_count
    controller["ordered_granularities"] = list(granularities)
    controller["prior_mean_input"] = copy.deepcopy(controller["prior_mean"])
    controller["prior_covariance_input"] = copy.deepcopy(
        controller["prior_covariance"]
    )
    controller["process_noise_covariance_input"] = copy.deepcopy(
        controller["process_noise_covariance"]
    )
    controller["resolved_prior_mean"] = resolved_prior_mean
    controller["resolved_prior_covariance"] = resolved_prior_covariance
    controller["observation_noise_variance"] = observation_noise_variance
    controller["resolved_process_noise_covariance"] = (
        resolved_process_noise_covariance
    )
    controller["reset"] = _resolve_controller_reset_configuration(
        config,
        controller,
        scope=scope,
        ordered_granularities=list(granularities),
    )

    controller_role = _resolve_fixed_controller_data_role(
        evaluation,
        "adaptive_controller",
        {
            "enabled": True,
            "source": "configured_dataset_split",
            "examples": 128,
            "objective_weights": "uniform",
            "fixed_manifest": True,
        },
    )
    final_holdout_role = _resolve_fixed_controller_data_role(
        evaluation,
        "final_holdout",
        {
            "enabled": True,
            "source": "configured_dataset_split",
            "examples": 512,
            "fixed_manifest": True,
            "evaluate_during_training": False,
        },
    )
    controller_role.setdefault("manifest_hash", "pending")
    final_holdout_role.setdefault("manifest_hash", "pending")
    controller["controller_panel_contract"] = copy.deepcopy(controller_role)
    controller["final_holdout_contract"] = copy.deepcopy(final_holdout_role)

    model["adaptive_controller"] = controller
    evaluation["adaptive_controller"] = controller_role
    evaluation["final_holdout"] = final_holdout_role


def _resolve_controller_reset_configuration(
    config: dict[str, Any],
    controller: Mapping[str, Any],
    *,
    scope: str,
    ordered_granularities: list[str],
) -> dict[str, Any]:
    raw_reset = controller.get("reset", {})
    if raw_reset is None:
        raw_reset = {}
    if not isinstance(raw_reset, Mapping):
        raise ConfigError("model.adaptive_controller.reset must be a mapping")
    reset = copy.deepcopy(dict(raw_reset))
    reset["enabled"] = _normalize_bool(
        reset.get("enabled", False),
        "model.adaptive_controller.reset.enabled",
    )

    policy = reset.get("policy", "full_prior")
    if not isinstance(policy, str) or policy.strip() not in VALID_CONTROLLER_RESET_POLICIES:
        raise ConfigError(
            "model.adaptive_controller.reset.policy must be 'full_prior' or "
            "'acquisition_only'"
        )
    reset["policy"] = policy.strip()
    acquisition_policy = reset.get("acquisition_policy", "balanced_global")
    if (
        not isinstance(acquisition_policy, str)
        or acquisition_policy.strip()
        not in VALID_CONTROLLER_RESET_ACQUISITION_POLICIES
    ):
        raise ConfigError(
            "model.adaptive_controller.reset.acquisition_policy must be "
            "'balanced_global'"
        )
    reset["acquisition_policy"] = acquisition_policy.strip()
    reset["acquisition_passes"] = _strict_positive_int(
        reset.get("acquisition_passes", 1),
        "model.adaptive_controller.reset.acquisition_passes",
    )
    reset["schedule_seed_stream_name"] = "controller_reset_schedule"
    reset["schedule_seed"] = seed_for(config, "controller_reset_schedule")

    if not reset["enabled"]:
        interval = reset.get("interval_steps")
        if interval is not None:
            interval = _strict_positive_int(
                interval,
                "model.adaptive_controller.reset.interval_steps",
            )
        return {
            "enabled": False,
            "interval_steps": interval,
            "policy": reset["policy"],
            "acquisition_policy": reset["acquisition_policy"],
            "acquisition_passes": reset["acquisition_passes"],
            "schedule_seed_stream_name": reset["schedule_seed_stream_name"],
            "schedule_seed": reset["schedule_seed"],
        }

    if scope != "global":
        raise ConfigError(
            "model.adaptive_controller.reset is supported only for adaptive_global"
        )
    resolved_process_noise = np.asarray(
        controller["resolved_process_noise_covariance"],
        dtype=np.float64,
    )
    if not np.all(resolved_process_noise == 0.0):
        raise ConfigError(
            "model.adaptive_controller.process_noise_covariance must be exactly "
            "zero when reset is enabled"
        )
    if "interval_steps" not in reset:
        raise ConfigError(
            "model.adaptive_controller.reset.interval_steps is required when reset "
            "is enabled"
        )
    interval_steps = _strict_positive_int(
        reset["interval_steps"],
        "model.adaptive_controller.reset.interval_steps",
    )
    decision_interval = int(controller["decision_interval_steps"])
    if interval_steps % decision_interval != 0:
        raise ConfigError(
            "model.adaptive_controller.reset.interval_steps must be divisible by "
            "model.adaptive_controller.decision_interval_steps"
        )
    episode_windows = interval_steps // decision_interval
    acquisition_windows = len(ordered_granularities) * reset["acquisition_passes"]
    if episode_windows < 2 * acquisition_windows:
        raise ConfigError(
            "model.adaptive_controller.reset.interval_steps must provide enough "
            "windows for acquisition plus at least an equal number of Thompson "
            "windows"
        )
    return {
        "enabled": True,
        "interval_steps": interval_steps,
        "policy": reset["policy"],
        "acquisition_policy": reset["acquisition_policy"],
        "acquisition_passes": reset["acquisition_passes"],
        "schedule_seed_stream_name": reset["schedule_seed_stream_name"],
        "schedule_seed": reset["schedule_seed"],
        "episode_window_count": episode_windows,
        "acquisition_window_count": acquisition_windows,
        "minimum_thompson_window_count": acquisition_windows,
    }


def _resolve_bayesian_controller_preset(
    config: dict[str, Any],
    controller: dict[str, Any],
) -> dict[str, Any]:
    preset_name = controller.get("preset")
    if preset_name in (None, ""):
        return controller

    if not isinstance(preset_name, str):
        raise ConfigError("model.adaptive_controller.preset must be a string")

    preset_name = preset_name.strip()
    if not preset_name:
        raise ConfigError(
            "model.adaptive_controller.preset must be a non-empty string"
        )

    preset_path = (
        PRESET_REGISTRY_ROOT / "adaptive_controller" / f"{preset_name}.yaml"
    )
    preset = _load_preset_registry_entry(
        preset_path,
        preset_name,
        preset_field="model.adaptive_controller.preset",
    )
    preset_evaluation = preset.get("evaluation", {})
    if not isinstance(preset_evaluation, Mapping):
        raise ConfigError(
            f"Preset registry entry {preset_path} must define evaluation as a mapping"
        )

    evaluation = config.setdefault("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ConfigError("evaluation must be a mapping")
    config["evaluation"] = _deep_merge_dicts(preset_evaluation, evaluation)

    configured_controller = {
        key: value for key, value in controller.items() if key != "preset"
    }
    resolved_controller = _deep_merge_dicts(
        preset.get("kwargs", {}),
        configured_controller,
    )
    resolved_controller["preset"] = preset_name
    resolved_controller["preset_registry_path"] = str(preset_path)
    return resolved_controller


def _resolve_fixed_controller_data_role(
    evaluation: dict[str, Any],
    section_name: str,
    expected_fields: Mapping[str, Any],
    *,
    method_name: str = "Bayesian Thompson",
) -> dict[str, Any]:
    section = evaluation.get(section_name)
    if not isinstance(section, Mapping):
        raise ConfigError(
            f"{method_name} requires an explicit controller data-role mapping "
            f"at evaluation.{section_name}"
        )
    resolved = copy.deepcopy(dict(section))
    missing_fields = [
        field_name for field_name in expected_fields if field_name not in resolved
    ]
    if missing_fields:
        raise ConfigError(
            f"{method_name} requires explicit controller data-role fields at "
            f"evaluation.{section_name}: {missing_fields}"
        )
    for field_name, expected_value in expected_fields.items():
        field_path = f"evaluation.{section_name}.{field_name}"
        actual_value = resolved[field_name]
        if isinstance(expected_value, bool):
            actual_value = _normalize_bool(actual_value, field_path)
        elif isinstance(expected_value, int):
            actual_value = _strict_positive_int(actual_value, field_path)
        elif isinstance(expected_value, str):
            if not isinstance(actual_value, str):
                raise ConfigError(f"{field_path} must be {expected_value!r}")
            actual_value = actual_value.strip()
        if actual_value != expected_value:
            raise ConfigError(f"{field_path} must be {expected_value!r}")
        resolved[field_name] = actual_value
    return resolved


def _resolve_bayesian_mean(
    value: Any,
    dimension: int,
    field_name: str,
) -> list[float]:
    if isinstance(value, list):
        if len(value) != dimension:
            raise ConfigError(
                f"{field_name} dimension must be {dimension}; found {len(value)}"
            )
        return [
            _finite_float(component, f"{field_name}[{index}]")
            for index, component in enumerate(value)
        ]
    scalar = _finite_float(value, field_name)
    return [scalar] * dimension


def _resolve_bayesian_covariance(
    value: Any,
    dimension: int,
    field_name: str,
) -> list[list[float]]:
    if isinstance(value, Mapping):
        if set(value) != {"intercept", "effects"}:
            raise ConfigError(
                f"{field_name} structured diagonal requires exactly "
                "intercept and effects"
            )
        intercept = _finite_float(value["intercept"], f"{field_name}.intercept")
        effects = _finite_float(value["effects"], f"{field_name}.effects")
        if intercept < 0.0 or effects < 0.0:
            raise ConfigError(f"{field_name} must be positive semidefinite")
        diagonal = [intercept, *([effects] * (dimension - 1))]
        return [
            [diagonal[row] if row == column else 0.0 for column in range(dimension)]
            for row in range(dimension)
        ]
    if not isinstance(value, list):
        scalar = _finite_float(value, field_name)
        if scalar < 0.0:
            raise ConfigError(f"{field_name} must be positive semidefinite")
        return [
            [scalar if row == column else 0.0 for column in range(dimension)]
            for row in range(dimension)
        ]

    if len(value) != dimension:
        raise ConfigError(
            f"{field_name} dimension must be {dimension}; found {len(value)}"
        )
    is_dense = any(isinstance(component, list) for component in value)
    if not is_dense:
        diagonal = [
            _finite_float(component, f"{field_name}[{index}]")
            for index, component in enumerate(value)
        ]
        if any(component < 0.0 for component in diagonal):
            raise ConfigError(f"{field_name} must be positive semidefinite")
        return [
            [diagonal[row] if row == column else 0.0 for column in range(dimension)]
            for row in range(dimension)
        ]

    if any(not isinstance(row, list) or len(row) != dimension for row in value):
        raise ConfigError(f"{field_name} dimension must be {dimension}x{dimension}")
    dense = np.asarray(
        [
            [
                _finite_float(component, f"{field_name}[{row_index}][{column_index}]")
                for column_index, component in enumerate(row)
            ]
            for row_index, row in enumerate(value)
        ],
        dtype=np.float64,
    )
    scale = max(1.0, float(np.max(np.abs(dense))))
    tolerance = BAYESIAN_COVARIANCE_TOLERANCE * scale
    if not np.allclose(dense, dense.T, rtol=0.0, atol=tolerance):
        raise ConfigError(f"{field_name} must be symmetric")
    dense = (dense + dense.T) * 0.5
    try:
        minimum_eigenvalue = float(np.linalg.eigvalsh(dense)[0])
    except np.linalg.LinAlgError as error:
        raise ConfigError(
            f"{field_name} positive semidefinite validation failed"
        ) from error
    if minimum_eigenvalue < -tolerance:
        raise ConfigError(f"{field_name} must be positive semidefinite")
    return dense.tolist()


def _finite_float(value: Any, field_name: str) -> float:
    number = _coerce_float(value, field_name)
    if not math.isfinite(number):
        raise ConfigError(f"{field_name} must be finite")
    return number


def _strict_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{field_name} must be a positive integer")
    return value


def _validate_resolved_bayesian_adaptive_configuration(
    config: Mapping[str, Any],
) -> None:
    expected = copy.deepcopy(dict(config))
    _resolve_bayesian_adaptive_configuration(expected)

    model = config.get("model", {})
    evaluation = config.get("evaluation", {})
    expected_model = expected["model"]
    expected_evaluation = expected["evaluation"]
    if model.get("adaptive_controller") != expected_model.get(
        "adaptive_controller"
    ):
        raise ConfigError(
            "model.adaptive_controller does not match the resolved Bayesian contract"
        )
    for section_name in ("adaptive_controller", "final_holdout"):
        if evaluation.get(section_name) != expected_evaluation.get(section_name):
            raise ConfigError(
                f"evaluation.{section_name} does not match the resolved Bayesian contract"
            )


def _validate_resolved_panelgrad_configuration(
    config: Mapping[str, Any],
) -> None:
    expected = copy.deepcopy(dict(config))
    _resolve_panelgrad_configuration(expected)

    model = config.get("model", {})
    evaluation = config.get("evaluation", {})
    expected_model = expected["model"]
    expected_evaluation = expected["evaluation"]
    if model.get("panelgrad") != expected_model.get("panelgrad"):
        raise ConfigError(
            "model.panelgrad does not match the resolved PanelGrad contract"
        )
    for section_name in ("adaptive_controller", "final_holdout"):
        if evaluation.get(section_name) != expected_evaluation.get(section_name):
            raise ConfigError(
                f"evaluation.{section_name} does not match the resolved "
                "PanelGrad contract"
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
        "random": "per_block",
    }[alias]


def _granularity_sampling_alias_from_mode(mode: str) -> str:
    if mode not in VALID_MODEL_GRANULARITY_SAMPLING_MODES:
        raise ConfigError(
            "model.granularity_sampling_mode must be one of "
            f"{sorted(VALID_MODEL_GRANULARITY_SAMPLING_MODES)}"
        )
    return {
        "global": "all",
        "fixed_global": "random",
        "per_block": "random",
        "adaptive_global": "random",
        "adaptive_per_block": "random",
    }[mode]


def _granularity_sampling_mode_from_run_sampling_mode(run_sampling_mode: str) -> str:
    if run_sampling_mode not in VALID_SAMPLING_MODES:
        raise ConfigError(
            f"run.sampling_mode must be one of {sorted(VALID_SAMPLING_MODES)}"
        )
    return {
        "nested-random": "per_block",
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
                if granularity_sampling_mode
                in {"global", "fixed_global", "adaptive_global"}
                else "per_block"
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
    if granularity_sampling_mode == "fixed_global":
        provenance["sampling_distribution"] = dict(
            model.get("global_sampling_distribution", {})
        )
    if (
        granularity_sampling_mode == "global"
        and run_sampling_mode == "nested-random"
    ):
        interval = model.get("global_sampling_interval_steps", 1)
        if isinstance(interval, int) and not isinstance(interval, bool):
            provenance["global_sampling_interval_steps"] = int(interval)
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
                "vocab_size",
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
            f"model.granularity_prefixes.{last_granularity} must resolve to full "
            f"model.intermediate_size={resolved_intermediate_size}; "
            f"got prefix_width={previous_width} from "
            f"fraction={resolved[last_granularity]}"
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
        raise ConfigError(
            f"run.granularity={granularity!r} is not valid for standalone run; "
            f"available labels={list(source_prefixes)}"
        )

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


def _resolve_distributed_contract_defaults(config: dict[str, Any]) -> None:
    training = config.setdefault("training", {})
    distributed = training.setdefault("distributed", {})
    if not isinstance(distributed, dict):
        raise ConfigError("training.distributed must be a mapping")
    env_world_size = _resolve_effective_world_size()
    expected = distributed.get("expected_world_size", env_world_size)
    expected = _positive_int(
        expected,
        "training.distributed.expected_world_size",
    )
    if expected not in {1, 2, 3, 4}:
        raise ConfigError(
            "training.distributed.expected_world_size must be between 1 and 4"
        )
    raw_world_size = os.environ.get("WORLD_SIZE")
    if raw_world_size not in (None, "") and env_world_size != expected:
        raise ConfigError(
            "training.distributed.expected_world_size does not match WORLD_SIZE: "
            f"expected={expected}, runtime={env_world_size}"
        )
    distributed["expected_world_size"] = expected


def _resolve_portfolio_controlled_experiment(config: dict[str, Any]) -> None:
    controlled = config.get("controlled_experiment")
    if not isinstance(controlled, dict):
        return
    raw_contract = controlled.get("portfolio_catchup")
    if raw_contract is None:
        return
    if not isinstance(raw_contract, dict):
        raise ConfigError(
            "controlled_experiment.portfolio_catchup must be a mapping"
        )
    role = controlled.get("comparison_role")
    if not isinstance(role, str) or role not in PORTFOLIO_COMPARISON_ROLES:
        raise ConfigError(
            "controlled_experiment.comparison_role must be one of "
            f"{sorted(PORTFOLIO_COMPARISON_ROLES)}"
        )
    schema_version = raw_contract.get("schema_version", 2)
    legacy_reference = role == "standalone_reference" and schema_version == 1
    extension_candidate = role == "elastic_candidate" and schema_version == 3
    if schema_version != 2 and not legacy_reference and not extension_candidate:
        raise ConfigError(
            "Portfolio catch-up schema must be 2, legacy reference schema 1, "
            "or extension candidate schema 3"
        )
    comparison_arm_id = controlled.get("comparison_arm_id")
    if extension_candidate:
        if comparison_arm_id not in PORTFOLIO_EXTENSION_ARMS:
            raise ConfigError(
                "controlled_experiment.comparison_arm_id must be one of "
                f"{sorted(PORTFOLIO_EXTENSION_ARMS)} for schema-3 candidates"
            )
    elif comparison_arm_id not in (None, ""):
        raise ConfigError(
            "controlled_experiment.comparison_arm_id is reserved for "
            "schema-3 extension candidates"
        )
    removed_lr_selection_fields = {
        "lr_selection_manifest_path",
        "lr_selection_manifest_hash",
    }.intersection(raw_contract)
    if removed_lr_selection_fields and (
        not legacy_reference
        or any(raw_contract[field] is not None for field in removed_lr_selection_fields)
    ):
        raise ConfigError(
            "Portfolio catch-up uses the same fixed LR 0.008 for both roles; "
            "LR-selection manifest fields are not supported except as null "
            "legacy metadata on schema-1 standalone references"
        )
    group_id = controlled.get(
        "comparison_group_id", PORTFOLIO_COMPARISON_GROUP_ID
    )
    if group_id != PORTFOLIO_COMPARISON_GROUP_ID:
        raise ConfigError(
            "controlled_experiment.comparison_group_id must be "
            f"{PORTFOLIO_COMPARISON_GROUP_ID}"
        )
    controlled["comparison_group_id"] = group_id

    defaults = {
        "enabled": role == "elastic_candidate",
        "schema_version": 2,
        "reference_budget_tokens": PORTFOLIO_REFERENCE_BUDGET_TOKENS,
        "elastic_budget_cap_tokens": PORTFOLIO_ELASTIC_BUDGET_CAP_TOKENS,
        "aggregate_reference_count": PORTFOLIO_AGGREGATE_REFERENCE_COUNT,
        "granularities": list(PORTFOLIO_GRANULARITIES),
        "perplexity_tolerance": 0.005,
        "required_consecutive_evaluations": 5,
        "target_manifest_path": None,
        "target_manifest_hash": None,
        "save_confirmation_checkpoint": True,
        "stop_on_confirmation": False,
    }
    contract = {**defaults, **raw_contract}
    contract["target_manifest_path"] = _normalize_optional_string(
        contract.get("target_manifest_path")
    )
    contract["target_manifest_hash"] = _normalize_optional_string(
        contract.get("target_manifest_hash")
    )
    controlled["portfolio_catchup"] = contract
    config["controlled_experiment"] = controlled

    if role == "elastic_candidate":
        _validate_portfolio_manifest_link(
            contract["target_manifest_path"],
            contract["target_manifest_hash"],
            field_prefix="controlled_experiment.portfolio_catchup.target_manifest",
        )


def _validate_portfolio_manifest_link(
    path_value: Any,
    hash_value: Any,
    *,
    field_prefix: str,
) -> dict[str, Any]:
    if isinstance(path_value, os.PathLike):
        path_value = os.fspath(path_value)
    if not isinstance(path_value, str) or not path_value.strip():
        raise ConfigError(f"{field_prefix}_path is required")
    if not isinstance(hash_value, str) or len(hash_value) != 64:
        raise ConfigError(f"{field_prefix}_hash must be a SHA256-style hash")
    path = Path(path_value).expanduser().resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"Cannot read immutable portfolio manifest: {path}") from error
    if not isinstance(value, dict):
        raise ConfigError(f"Immutable portfolio manifest must be a mapping: {path}")
    body = copy.deepcopy(value)
    embedded_hash = body.pop("manifest_hash", None)
    actual_hash = stable_hash(body)
    if embedded_hash != actual_hash or hash_value != actual_hash:
        raise ConfigError(f"{field_prefix} hash mismatch")
    return value


def _validate_portfolio_controlled_experiment(config: Mapping[str, Any]) -> None:
    controlled = config.get("controlled_experiment")
    if not isinstance(controlled, Mapping):
        return
    contract = controlled.get("portfolio_catchup")
    if contract is None:
        return
    if not isinstance(contract, Mapping):
        raise ConfigError(
            "controlled_experiment.portfolio_catchup must be a mapping"
        )
    role = controlled.get("comparison_role")
    if role not in PORTFOLIO_COMPARISON_ROLES:
        raise ConfigError(
            "controlled_experiment.comparison_role must be one of "
            f"{sorted(PORTFOLIO_COMPARISON_ROLES)}"
        )
    schema_version = contract.get("schema_version")
    legacy_reference = role == "standalone_reference" and schema_version == 1
    extension_candidate = role == "elastic_candidate" and schema_version == 3
    if schema_version != 2 and not legacy_reference and not extension_candidate:
        raise ConfigError(
            "Portfolio catch-up schema must be 2, legacy reference schema 1, "
            "or extension candidate schema 3"
        )
    comparison_arm_id = controlled.get("comparison_arm_id")
    if extension_candidate:
        if comparison_arm_id not in PORTFOLIO_EXTENSION_ARMS:
            raise ConfigError(
                "controlled_experiment.comparison_arm_id must identify a "
                "supported schema-3 extension arm"
            )
    elif comparison_arm_id not in (None, ""):
        raise ConfigError(
            "controlled_experiment.comparison_arm_id is reserved for "
            "schema-3 extension candidates"
        )
    removed_lr_selection_fields = {
        "lr_selection_manifest_path",
        "lr_selection_manifest_hash",
    }.intersection(contract)
    if removed_lr_selection_fields and (
        not legacy_reference
        or any(contract[field] is not None for field in removed_lr_selection_fields)
    ):
        raise ConfigError(
            "Portfolio catch-up uses the same fixed LR 0.008 for both roles; "
            "LR-selection manifest fields are not supported except as null "
            "legacy metadata on schema-1 standalone references"
        )
    if controlled.get("comparison_group_id") != PORTFOLIO_COMPARISON_GROUP_ID:
        raise ConfigError("Portfolio comparison group ID is invalid")

    expected_elastic_budget = (
        PORTFOLIO_EXTENSION_ARMS[str(comparison_arm_id)]["budget_tokens"]
        if extension_candidate
        else PORTFOLIO_ELASTIC_BUDGET_CAP_TOKENS
    )
    exact_fields = {
        "reference_budget_tokens": PORTFOLIO_REFERENCE_BUDGET_TOKENS,
        "elastic_budget_cap_tokens": expected_elastic_budget,
        "aggregate_reference_count": PORTFOLIO_AGGREGATE_REFERENCE_COUNT,
        "granularities": list(PORTFOLIO_GRANULARITIES),
        "perplexity_tolerance": 0.005,
        "required_consecutive_evaluations": 5,
        "save_confirmation_checkpoint": True,
        "stop_on_confirmation": False,
    }
    mismatches = {
        field: (contract.get(field), expected)
        for field, expected in exact_fields.items()
        if contract.get(field) != expected
    }
    if mismatches:
        raise ConfigError(f"Portfolio catch-up fixed contract mismatch: {mismatches}")

    training = config.get("training", {})
    model = config.get("model", {})
    run = config.get("run", {})
    expected_budget = (
        expected_elastic_budget
        if role == "elastic_candidate"
        else PORTFOLIO_REFERENCE_BUDGET_TOKENS
    )
    if training.get("token_budget") != expected_budget:
        raise ConfigError(
            f"Portfolio {role} token budget must be exactly {expected_budget}"
        )
    if training.get("scheduler_name") != "cosine":
        raise ConfigError("Portfolio runs require a cosine scheduler")

    resolved_lr = float(training.get("resolved_learning_rate", -1.0))
    granularities = list(model.get("granularities", []))
    if not math.isclose(resolved_lr, 0.008, rel_tol=0.0, abs_tol=1e-12):
        raise ConfigError("Portfolio references and candidates require fixed LR 0.008")

    if role == "standalone_reference":
        if contract.get("enabled") is not False:
            raise ConfigError("Standalone references must disable online catch-up")
        if run.get("model_family") != "standalone" or len(granularities) != 1:
            raise ConfigError(
                "Standalone references require exactly one active granularity"
            )
        if granularities[0] not in PORTFOLIO_GRANULARITIES:
            raise ConfigError("Standalone reference granularity is outside the portfolio")
    else:
        expected_sampling_mode = (
            PORTFOLIO_EXTENSION_ARMS[str(comparison_arm_id)]["sampling_mode"]
            if extension_candidate
            else "nested-random"
        )
        if run.get("model_family") != "nested" or run.get(
            "sampling_mode"
        ) != expected_sampling_mode:
            raise ConfigError(
                "Elastic portfolio topology does not match its comparison arm"
            )
        if granularities != list(PORTFOLIO_GRANULARITIES):
            raise ConfigError(
                "Elastic portfolio runs require all four ordered granularities"
            )
        if expected_sampling_mode == "nested-random":
            if model.get("granularity_sampling_mode") != "global":
                raise ConfigError(
                    "Uniform-H1 portfolio arms require uniform global sampling"
                )
            if model.get("global_sampling_schedule") != "random_with_replacement":
                raise ConfigError(
                    "Uniform-H1 portfolio arms require random-with-replacement sampling"
                )
            if model.get("global_sampling_interval_steps") != 1:
                raise ConfigError("Uniform-H1 portfolio arms require H=1")

    if role == "elastic_candidate":
        if contract.get("enabled") is not True:
            raise ConfigError("Elastic candidates must enable online catch-up")
        if int(run.get("seed", -1)) not in {42, 43, 44}:
            raise ConfigError("Elastic candidate seed must be 42, 43, or 44")
        for name in (
            "target_manifest_path",
            "target_manifest_hash",
        ):
            if contract.get(name) in (None, ""):
                raise ConfigError(
                    f"controlled_experiment.portfolio_catchup.{name} is required"
                )


def _resolve_parameter_reporting_defaults(config: dict[str, Any]) -> None:
    reporting = config.setdefault("parameter_reporting", {})
    if not isinstance(reporting, dict):
        raise ConfigError("parameter_reporting must be a mapping when provided")

    reporting.setdefault("lm_head_counting", "separately_counted")


def _resolve_long_run_defaults(config: dict[str, Any]) -> None:
    _resolve_reproducibility_defaults(config)
    _resolve_continuation_defaults(config)
    _resolve_monitoring_defaults(config)
    _resolve_pre_nested_warmup_defaults(config)
    _resolve_reliability_defaults(config)


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
    gradient_accumulation_steps = _positive_int(
        training.get("gradient_accumulation_steps", 1),
        "training.gradient_accumulation_steps",
    )
    if effective_world_size is None:
        distributed = training.get("distributed", {})
        configured_world_size = (
            distributed.get("expected_world_size")
            if isinstance(distributed, Mapping)
            else None
        )
        effective_world_size = (
            int(configured_world_size)
            if configured_world_size is not None
            else _resolve_effective_world_size()
        )
        if world_size_source is None:
            world_size_source = (
                "training.distributed.expected_world_size"
                if configured_world_size is not None
                else "WORLD_SIZE"
                if os.environ.get("WORLD_SIZE") not in (None, "")
                else "single_process"
            )
    else:
        effective_world_size = _positive_int(
            effective_world_size,
            "training.effective_world_size",
        )
        if world_size_source is None:
            world_size_source = "distributed_context"

    expected_tokens_per_microstep = (
        batch_size_per_process * context_length * effective_world_size
    )
    expected_tokens_per_step = (
        expected_tokens_per_microstep * gradient_accumulation_steps
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
    training["gradient_accumulation_steps"] = gradient_accumulation_steps
    training["expected_tokens_per_microstep"] = expected_tokens_per_microstep
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
    raw_scheduler_specific_kwargs = {
        key: value
        for key, value in scheduler_input_kwargs.items()
        if key != "warmup_steps"
    }
    scheduler_contract = None
    if scheduler_name == WSD_SCHEDULER_NAME:
        scheduler_kwargs, scheduler_contract = _resolve_wsd_scheduler_contract(
            raw_scheduler_specific_kwargs,
            max_steps=int(training["max_steps"]),
            warmup_steps=int(resolved_warmup_steps),
            expected_tokens_per_step=int(training["expected_tokens_per_step"]),
            token_budget=int(training["token_budget"]),
            peak_learning_rate=float(training["resolved_learning_rate"]),
        )
    else:
        scheduler_kwargs = _resolve_scheduler_kwargs(
            scheduler_name,
            raw_scheduler_specific_kwargs,
        )
    training["scheduler"] = {
        "name": scheduler_name,
        "kwargs": copy.deepcopy(scheduler_input_kwargs),
        "resolved_warmup_steps": int(resolved_warmup_steps),
    }
    if scheduler_contract is not None:
        training["scheduler"]["contract"] = copy.deepcopy(scheduler_contract)
    training["scheduler_name"] = scheduler_name
    training["scheduler_kwargs"] = scheduler_kwargs
    training["scheduler_specific_kwargs"] = copy.deepcopy(scheduler_kwargs)
    training["scheduler_contract"] = copy.deepcopy(scheduler_contract)


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
    continuation["retain_previous_latest"] = _normalize_bool(
        continuation.get("retain_previous_latest", True),
        "run.continuation.retain_previous_latest",
    )
    run["continuation"] = continuation


def _resolve_reliability_defaults(config: dict[str, Any]) -> None:
    run = config.setdefault("run", {})
    training = config.setdefault("training", {})
    outputs = config.setdefault("outputs", {})
    evaluation = config.setdefault("evaluation", {})
    if not isinstance(outputs, dict):
        raise ConfigError("outputs must be a mapping when provided")
    if not isinstance(evaluation, dict):
        raise ConfigError("evaluation must be a mapping when provided")

    requested_precision = training.get(
        "requested_mixed_precision",
        training.get("mixed_precision", "none"),
    )
    if not isinstance(requested_precision, str):
        raise ConfigError("training.mixed_precision must be a string")
    requested_precision = requested_precision.strip().lower()
    if requested_precision not in VALID_MIXED_PRECISION_MODES:
        raise ConfigError(
            "training.mixed_precision must be one of "
            f"{sorted(VALID_MIXED_PRECISION_MODES)}"
        )
    requested_checkpointing = _normalize_bool(
        training.get(
            "requested_activation_checkpointing",
            training.get("activation_checkpointing", False),
        ),
        "training.activation_checkpointing",
    )
    training["mixed_precision"] = requested_precision
    training["requested_mixed_precision"] = requested_precision
    training["resolved_mixed_precision"] = requested_precision
    training["activation_checkpointing"] = requested_checkpointing
    training["requested_activation_checkpointing"] = requested_checkpointing
    training["resolved_activation_checkpointing"] = requested_checkpointing

    outputs["metrics_flush_interval_steps"] = _positive_int(
        outputs.get("metrics_flush_interval_steps", 100),
        "outputs.metrics_flush_interval_steps",
    )
    outputs["best_eval_retention_count"] = _positive_int(
        outputs.get("best_eval_retention_count", 1),
        "outputs.best_eval_retention_count",
    )
    configured_artifact_io = outputs.get("artifact_io", {})
    if not isinstance(configured_artifact_io, dict):
        raise ConfigError("outputs.artifact_io must be a mapping when provided")
    artifact_io = DEFAULT_ARTIFACT_IO | configured_artifact_io
    artifact_io["max_attempts"] = _positive_int(
        artifact_io["max_attempts"],
        "outputs.artifact_io.max_attempts",
    )
    artifact_io["initial_backoff_seconds"] = _nonnegative_float(
        artifact_io["initial_backoff_seconds"],
        "outputs.artifact_io.initial_backoff_seconds",
    )
    artifact_io["max_backoff_seconds"] = _nonnegative_float(
        artifact_io["max_backoff_seconds"],
        "outputs.artifact_io.max_backoff_seconds",
    )
    artifact_io["jitter_fraction"] = _nonnegative_float(
        artifact_io["jitter_fraction"],
        "outputs.artifact_io.jitter_fraction",
    )
    artifact_io["checkpoint_staging"] = str(
        artifact_io["checkpoint_staging"]
    ).strip().lower()
    artifact_io["periodic_checkpoint_failure_policy"] = str(
        artifact_io["periodic_checkpoint_failure_policy"]
    ).strip().lower()
    artifact_io["metrics_pending_row_limit"] = _positive_int(
        artifact_io["metrics_pending_row_limit"],
        "outputs.artifact_io.metrics_pending_row_limit",
    )
    outputs["artifact_io"] = artifact_io

    _resolve_evaluation_defaults(config)
    validation = evaluation["validation"]
    if run.get("completion_label") == "run" and not validation["run_at_completion"]:
        raise ConfigError(
            "evaluation.validation.run_at_completion is required when "
            "run.completion_label=run"
        )
    validation["run_at_completion_reason"] = (
        "required_for_completed_run"
        if run.get("completion_label") == "run"
        else (
            "enabled_for_debug_run"
            if validation["run_at_completion"]
            else "explicitly_disabled_for_debug_run"
        )
    )


def _resolve_reproducibility_defaults(config: dict[str, Any]) -> None:
    run = config.setdefault("run", {})
    if "seed" not in run:
        raise ConfigError("run.seed must be an explicit nonnegative integer")
    seed = run["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError("run.seed must be an explicit nonnegative integer")

    reproducibility = run.get("reproducibility", {})
    if not isinstance(reproducibility, dict):
        raise ConfigError("run.reproducibility must be a mapping when provided")
    mode = reproducibility.get("mode", "strict")
    if mode != "strict":
        raise ConfigError("run.reproducibility.mode must be strict")
    reproducibility["mode"] = "strict"
    reproducibility["seed_stream_version"] = _positive_int(
        reproducibility.get("seed_stream_version", SEED_STREAM_VERSION),
        "run.reproducibility.seed_stream_version",
    )
    reproducibility["data_split_version"] = _positive_int(
        reproducibility.get("data_split_version", DATA_SPLIT_VERSION),
        "run.reproducibility.data_split_version",
    )
    run["reproducibility"] = reproducibility


def _resolve_evaluation_defaults(config: dict[str, Any]) -> None:
    training = config.setdefault("training", {})
    evaluation = config.setdefault("evaluation", {})
    raw_validation = evaluation.get("validation", False)
    legacy_enabled = raw_validation if isinstance(raw_validation, bool) else None
    if isinstance(raw_validation, Mapping):
        validation = copy.deepcopy(dict(raw_validation))
    elif isinstance(raw_validation, bool):
        validation = {}
    else:
        raise ConfigError("evaluation.validation must be a boolean or mapping")

    if legacy_enabled is not None:
        validation["enabled"] = legacy_enabled
    else:
        validation["enabled"] = _normalize_bool(
            validation.get("enabled", False),
            "evaluation.validation.enabled",
        )

    legacy_interval = training.get("eval_interval")
    canonical_interval = validation.get("interval_steps")
    if legacy_interval is not None and canonical_interval is not None:
        if int(legacy_interval) != int(canonical_interval):
            raise ConfigError(
                "Conflicting training.eval_interval and "
                "evaluation.validation.interval_steps"
            )
    validation["interval_steps"] = _nonnegative_int(
        canonical_interval if canonical_interval is not None else legacy_interval or 0,
        "evaluation.validation.interval_steps",
    )
    validation["interval_tokens"] = _nonnegative_int(
        validation.get("interval_tokens", 0),
        "evaluation.validation.interval_tokens",
    )
    if validation["interval_steps"] > 0 and validation["interval_tokens"] > 0:
        raise ConfigError(
            "evaluation.validation.interval_tokens and a positive interval_steps "
            "are mutually exclusive"
        )

    legacy_completion = evaluation.get("final_validation")
    canonical_completion = validation.get("run_at_completion")
    if legacy_completion is not None and canonical_completion is not None:
        if bool(legacy_completion) != bool(canonical_completion):
            raise ConfigError(
                "Conflicting evaluation.final_validation and "
                "evaluation.validation.run_at_completion"
            )
    validation["run_at_completion"] = _normalize_bool(
        canonical_completion
        if canonical_completion is not None
        else (legacy_completion if legacy_completion is not None else True),
        "evaluation.validation.run_at_completion",
    )

    holdout = validation.get("holdout", {})
    if not isinstance(holdout, dict):
        raise ConfigError("evaluation.validation.holdout must be a mapping")
    holdout["source"] = str(
        holdout.get("source", "configured_dataset_split")
    )
    legacy_batches = training.get("eval_batches")
    legacy_examples = None
    if legacy_batches is not None:
        legacy_examples = _positive_int(
            legacy_batches, "training.eval_batches"
        ) * _positive_int(
            training.get("batch_size_per_process"),
            "training.batch_size_per_process",
        )
    canonical_examples = holdout.get("examples")
    if legacy_examples is not None and canonical_examples is not None:
        if legacy_examples != int(canonical_examples):
            raise ConfigError(
                "Conflicting training.eval_batches and "
                "evaluation.validation.holdout.examples"
            )
    holdout["examples"] = _positive_int(
        canonical_examples
        if canonical_examples is not None
        else (legacy_examples if legacy_examples is not None else 512),
        "evaluation.validation.holdout.examples",
    )
    validation["holdout"] = holdout
    validation["trailing_summary_evaluations"] = _positive_int(
        validation.get("trailing_summary_evaluations", 5),
        "evaluation.validation.trailing_summary_evaluations",
    )

    test_evaluation = evaluation.get("test", {"enabled": False})
    if not isinstance(test_evaluation, dict):
        raise ConfigError("evaluation.test must be a mapping")
    test_evaluation["enabled"] = _normalize_bool(
        test_evaluation.get("enabled", False), "evaluation.test.enabled"
    )
    if test_evaluation["enabled"]:
        raise ConfigError("evaluation.test.enabled must be false")

    _resolve_gradient_interference_defaults(config)

    training.pop("eval_interval", None)
    training.pop("eval_batches", None)
    evaluation.pop("final_validation", None)
    evaluation["validation"] = validation
    evaluation["test"] = test_evaluation


def _resolve_gradient_interference_defaults(config: dict[str, Any]) -> None:
    """Resolve the opt-in H-window raw-gradient compatibility diagnostic."""

    evaluation = config.setdefault("evaluation", {})
    raw = evaluation.get("gradient_interference", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError("evaluation.gradient_interference must be a mapping")
    diagnostic = copy.deepcopy(dict(raw))
    user_fields = {
        "enabled",
        "trajectory_fractions",
        "include_warmup_completion",
        "layerwise",
    }
    resolved_fields = {
        "schema_version",
        "event_type",
        "artifact_path",
        "resolved_steps",
        "resolved_milestones",
        "milestone_reasons",
        "fixed_probe_contract",
        "fixed_probe_manifest_hash",
        "controlled_support_hash",
        "diagnostic_contract_hash",
        "gradient_semantics",
        "loss_aggregation",
        "shared_support",
    }
    unknown = sorted(set(diagnostic) - user_fields - resolved_fields)
    if unknown:
        raise ConfigError(
            "Unknown evaluation.gradient_interference fields: " f"{unknown}"
        )

    diagnostic["enabled"] = _normalize_bool(
        diagnostic.get("enabled", False),
        "evaluation.gradient_interference.enabled",
    )
    raw_fractions = diagnostic.get(
        "trajectory_fractions", [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    if not isinstance(raw_fractions, list) or not raw_fractions:
        raise ConfigError(
            "evaluation.gradient_interference.trajectory_fractions must be a "
            "non-empty list"
        )
    fractions: list[float] = []
    for index, raw_fraction in enumerate(raw_fractions):
        if isinstance(raw_fraction, bool) or not isinstance(raw_fraction, (int, float)):
            raise ConfigError(
                "evaluation.gradient_interference.trajectory_fractions "
                f"entry {index} must be numeric"
            )
        fraction = float(raw_fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ConfigError(
                "evaluation.gradient_interference.trajectory_fractions must "
                "contain only finite values between zero and one"
            )
        fractions.append(fraction)
    diagnostic["trajectory_fractions"] = fractions
    diagnostic["include_warmup_completion"] = _normalize_bool(
        diagnostic.get("include_warmup_completion", True),
        "evaluation.gradient_interference.include_warmup_completion",
    )
    diagnostic["layerwise"] = _normalize_bool(
        diagnostic.get("layerwise", True),
        "evaluation.gradient_interference.layerwise",
    )

    max_steps = _nonnegative_int(
        config.get("training", {}).get("max_steps"), "training.max_steps"
    )
    reasons_by_step: dict[int, list[str]] = {}
    for fraction in fractions:
        step = int(math.ceil(fraction * max_steps))
        reason = f"trajectory_fraction:{format(fraction, '.17g')}"
        reasons_by_step.setdefault(step, [])
        if reason not in reasons_by_step[step]:
            reasons_by_step[step].append(reason)
    if diagnostic["include_warmup_completion"]:
        warmup_step = _nonnegative_int(
            config.get("training", {}).get("resolved_warmup_steps"),
            "training.resolved_warmup_steps",
        )
        reasons_by_step.setdefault(warmup_step, [])
        if "warmup_completion" not in reasons_by_step[warmup_step]:
            reasons_by_step[warmup_step].append("warmup_completion")

    resolved_steps = sorted(reasons_by_step)
    resolved_milestones = [
        {"step": step, "reasons": list(reasons_by_step[step])}
        for step in resolved_steps
    ]
    diagnostic.update(
        {
            "schema_version": 1,
            "event_type": "gradient_interference_snapshot",
            "artifact_path": "gradient_interference.jsonl",
            "resolved_steps": resolved_steps,
            "resolved_milestones": resolved_milestones,
            "milestone_reasons": {
                str(step): list(reasons_by_step[step]) for step in resolved_steps
            },
            "gradient_semantics": "raw_pre_correction_pre_clipping",
            "loss_aggregation": "target_token_weighted_fixed_probe",
            "shared_support": "smaller_nested_controlled_ffn_support",
            "fixed_probe_manifest_hash": diagnostic.get(
                "fixed_probe_manifest_hash", "pending"
            ),
            "controlled_support_hash": diagnostic.get(
                "controlled_support_hash", "pending"
            ),
        }
    )

    if diagnostic["enabled"]:
        run = config.get("run", {})
        model = config.get("model", {})
        dataset = config.get("dataset", {})
        granularities = model.get("granularities", [])
        if run.get("model_family") != "nested":
            raise ConfigError(
                "gradient interference requires a nested model run"
            )
        if run.get("sampling_mode") != "nested-random":
            raise ConfigError(
                "gradient interference requires run.sampling_mode=nested-random"
            )
        if model.get("granularity_sampling_mode") != "global":
            raise ConfigError(
                "gradient interference requires uniform global sampling"
            )
        if not isinstance(granularities, list) or len(granularities) < 2:
            raise ConfigError(
                "gradient interference requires at least two granularities"
            )
        if not bool(dataset.get("fixed_four_role_partition", False)):
            raise ConfigError(
                "gradient interference requires dataset.fixed_four_role_partition=true"
            )
        fixed_probe = _resolve_fixed_controller_data_role(
            evaluation,
            "adaptive_controller",
            {
                "enabled": True,
                "source": "configured_dataset_split",
                "examples": 128,
                "objective_weights": "uniform",
                "fixed_manifest": True,
            },
            method_name="gradient interference",
        )
        _resolve_fixed_controller_data_role(
            evaluation,
            "final_holdout",
            {
                "enabled": True,
                "source": "configured_dataset_split",
                "examples": 512,
                "fixed_manifest": True,
                "evaluate_during_training": False,
            },
            method_name="gradient interference",
        )
        fixed_probe.setdefault("manifest_hash", "pending")
        diagnostic["fixed_probe_contract"] = fixed_probe
    else:
        diagnostic["fixed_probe_contract"] = diagnostic.get(
            "fixed_probe_contract", None
        )

    contract_payload = {
        key: diagnostic[key]
        for key in (
            "schema_version",
            "event_type",
            "trajectory_fractions",
            "include_warmup_completion",
            "layerwise",
            "resolved_milestones",
            "gradient_semantics",
            "loss_aggregation",
            "shared_support",
        )
    }
    diagnostic["diagnostic_contract_hash"] = hashlib.sha256(
        json.dumps(
            contract_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    evaluation["gradient_interference"] = diagnostic


def _validate_gradient_interference_configuration(
    config: Mapping[str, Any],
) -> None:
    evaluation = config.get("evaluation", {})
    diagnostic = evaluation.get("gradient_interference")
    if not isinstance(diagnostic, Mapping):
        raise ConfigError("evaluation.gradient_interference must be a mapping")
    required = {
        "enabled",
        "trajectory_fractions",
        "include_warmup_completion",
        "layerwise",
        "schema_version",
        "event_type",
        "artifact_path",
        "resolved_steps",
        "resolved_milestones",
        "milestone_reasons",
        "diagnostic_contract_hash",
        "gradient_semantics",
        "loss_aggregation",
        "shared_support",
    }
    missing = sorted(required - set(diagnostic))
    if missing:
        raise ConfigError(
            "evaluation.gradient_interference is missing resolved fields: "
            f"{missing}"
        )
    if diagnostic.get("schema_version") != 1:
        raise ConfigError("gradient interference schema_version must be 1")
    if diagnostic.get("event_type") != "gradient_interference_snapshot":
        raise ConfigError("gradient interference event_type is invalid")
    if diagnostic.get("artifact_path") != "gradient_interference.jsonl":
        raise ConfigError("gradient interference artifact_path is invalid")
    steps = diagnostic.get("resolved_steps")
    milestones = diagnostic.get("resolved_milestones")
    if (
        not isinstance(steps, list)
        or any(isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in steps)
        or steps != sorted(set(steps))
        or not isinstance(milestones, list)
        or [item.get("step") for item in milestones if isinstance(item, Mapping)]
        != steps
        or len(milestones) != len(steps)
        or any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("reasons"), list)
            or not item["reasons"]
            or any(not isinstance(reason, str) or not reason for reason in item["reasons"])
            for item in milestones
        )
    ):
        raise ConfigError("gradient interference milestone schedule is invalid")
    contract_hash = diagnostic.get("diagnostic_contract_hash")
    if not isinstance(contract_hash, str) or len(contract_hash) != 64:
        raise ConfigError("gradient interference diagnostic_contract_hash is invalid")
    for field in ("fixed_probe_manifest_hash", "controlled_support_hash"):
        value = diagnostic.get(field, "pending")
        if value != "pending" and (not isinstance(value, str) or not value):
            raise ConfigError(f"gradient interference {field} is invalid")


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
    policy = warmup.get("policy", "full_only")
    if not isinstance(policy, str):
        raise ConfigError("training.pre_nested_warmup.policy must be a string")
    policy = policy.strip()
    if policy not in VALID_PRE_NESTED_WARMUP_POLICIES:
        raise ConfigError(
            "training.pre_nested_warmup.policy must be one of "
            f"{sorted(VALID_PRE_NESTED_WARMUP_POLICIES)}"
        )
    warmup["policy"] = policy

    if policy == "balanced_global" and warmup["enabled"]:
        model = config.get("model", {})
        strategy = model.get("adaptive_sampler_strategy")
        interval_source = (
            model.get("panelgrad", {})
            if strategy == "panelgrad"
            else model.get("adaptive_controller", {})
        )
        default_interval = (
            interval_source.get(
                "refresh_interval_steps"
                if strategy == "panelgrad"
                else "decision_interval_steps"
            )
            if isinstance(interval_source, Mapping)
            else None
        )
        raw_interval = warmup.get("action_interval_steps", default_interval)
        if raw_interval is None:
            raise ConfigError(
                "training.pre_nested_warmup.action_interval_steps is required for "
                "balanced_global warmup"
            )
        warmup["action_interval_steps"] = _positive_int(
            raw_interval,
            "training.pre_nested_warmup.action_interval_steps",
        )
        granularities = list(config.get("model", {}).get("granularities", []))
        interval = warmup["action_interval_steps"]
        denominator = int(interval) * len(granularities)
        if denominator <= 0 or int(warmup["duration"]) % denominator != 0:
            raise ConfigError(
                "training.pre_nested_warmup.duration must be divisible by "
                "action_interval_steps * number of granularities"
            )
        passes = int(warmup["duration"]) // denominator
        if passes < 2:
            raise ConfigError(
                "training.pre_nested_warmup balanced_global requires at least two "
                "complete passes over all granularities"
            )
        schedule_seed = seed_for(config, "pre_nested_warmup_schedule")
        schedule, schedule_hash = build_balanced_warmup_schedule(
            granularities,
            passes=passes,
            seed=schedule_seed,
            action_interval_steps=int(interval),
            duration_steps=int(warmup["duration"]),
        )
        warmup["schedule_seed"] = schedule_seed
        warmup["schedule_hash"] = schedule_hash
        warmup["schedule"] = schedule
        warmup["passes"] = passes
        warmup["controller_start_step"] = int(warmup["duration"])
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


def _resolve_wsd_scheduler_contract(
    raw_kwargs: Any,
    *,
    max_steps: int,
    warmup_steps: int,
    expected_tokens_per_step: int,
    token_budget: int,
    peak_learning_rate: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve the horizon-independent ratio WSD policy into HF arguments."""

    if not isinstance(raw_kwargs, dict):
        raise ConfigError("training.scheduler.kwargs must be a mapping when provided")
    unsupported = sorted(str(key) for key in raw_kwargs if key not in WSD_INPUT_KWARGS)
    if unsupported:
        raise ConfigError(
            "training.scheduler.kwargs contains unsupported WSD fields: "
            f"{unsupported}"
        )
    if "decay_ratio" not in raw_kwargs:
        raise ConfigError(
            "training.scheduler.kwargs.decay_ratio is required for "
            "warmup_stable_decay"
        )
    raw_ratio = raw_kwargs["decay_ratio"]
    if isinstance(raw_ratio, bool) or not isinstance(raw_ratio, (int, float)):
        raise ConfigError("training.scheduler.kwargs.decay_ratio must be numeric")
    decay_ratio = float(raw_ratio)
    if not math.isfinite(decay_ratio) or not 0.0 < decay_ratio < 1.0:
        raise ConfigError(
            "training.scheduler.kwargs.decay_ratio must be finite and strictly "
            "between 0 and 1"
        )

    warmup_type = raw_kwargs.get("warmup_type", "linear")
    decay_type = raw_kwargs.get("decay_type", "cosine")
    if warmup_type != "linear":
        raise ConfigError(
            "training.scheduler.kwargs.warmup_type must be 'linear' for the "
            "ratio WSD policy"
        )
    if decay_type != "cosine":
        raise ConfigError(
            "training.scheduler.kwargs.decay_type must be 'cosine' for the "
            "ratio WSD policy"
        )

    min_lr_ratio = _nonnegative_finite_float(
        raw_kwargs.get("min_lr_ratio", 0.0),
        "training.scheduler.kwargs.min_lr_ratio",
    )
    if min_lr_ratio != 0.0:
        raise ConfigError(
            "training.scheduler.kwargs.min_lr_ratio must be 0.0 for the ratio "
            "WSD policy"
        )
    num_cycles = _nonnegative_finite_float(
        raw_kwargs.get("num_cycles", 0.5),
        "training.scheduler.kwargs.num_cycles",
    )
    if num_cycles != 0.5:
        raise ConfigError(
            "training.scheduler.kwargs.num_cycles must be 0.5 for one cosine "
            "cooldown"
        )

    decay_steps = math.ceil(max_steps * decay_ratio)
    stable_steps = max_steps - warmup_steps - decay_steps
    if stable_steps < 1:
        raise ConfigError(
            "warmup_stable_decay requires at least one stable step after resolving "
            f"max_steps={max_steps}, warmup_steps={warmup_steps}, "
            f"decay_steps={decay_steps}"
        )
    cooldown_start_step = warmup_steps + stable_steps
    cooldown_start_tokens = min(
        cooldown_start_step * expected_tokens_per_step,
        token_budget,
    )
    contract = {
        "name": WSD_SCHEDULER_NAME,
        "policy": WSD_SCHEDULE_POLICY,
        "policy_version": WSD_SCHEDULE_POLICY_VERSION,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "stable_steps": stable_steps,
        "decay_steps": decay_steps,
        "decay_ratio": decay_ratio,
        "cooldown_start_step": cooldown_start_step,
        "cooldown_start_tokens": cooldown_start_tokens,
        "schedule_end_tokens": token_budget,
        "warmup_type": warmup_type,
        "decay_type": decay_type,
        "min_lr_ratio": min_lr_ratio,
        "min_learning_rate": peak_learning_rate * min_lr_ratio,
        "num_cycles": num_cycles,
    }
    scheduler_specific_kwargs = {
        "num_decay_steps": decay_steps,
        "num_stable_steps": stable_steps,
        "warmup_type": warmup_type,
        "decay_type": decay_type,
        "min_lr_ratio": min_lr_ratio,
        "num_cycles": num_cycles,
    }
    return scheduler_specific_kwargs, contract


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
    *,
    preset_field: str = "training.optimizer.preset",
) -> dict[str, Any]:
    if not preset_path.is_file():
        raise ConfigError(
            f"Unknown {preset_field}={preset_name!r}; "
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
    gradient_accumulation_steps = _positive_int(
        training.get("gradient_accumulation_steps", 1),
        "training.gradient_accumulation_steps",
    )
    expected_tokens_per_microstep = (
        batch_size_per_process * context_length * effective_world_size
    )
    if training.get("expected_tokens_per_microstep") != expected_tokens_per_microstep:
        raise ConfigError(
            "training.expected_tokens_per_microstep must equal "
            "batch_size_per_process * context_length * effective_world_size"
        )
    expected_tokens_per_step = (
        expected_tokens_per_microstep * gradient_accumulation_steps
    )
    if training["expected_tokens_per_step"] != expected_tokens_per_step:
        raise ConfigError(
            "training.expected_tokens_per_step must equal "
            "batch_size_per_process * context_length * effective_world_size"
            " * gradient_accumulation_steps"
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
        elif isinstance(current[part], bool):
            # Legacy configuration allows boolean feature switches such as
            # ``evaluation.validation: true``. Preserve that switch when a
            # more specific command-line override promotes it to a mapping.
            current[part] = {"enabled": current[part]}
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
