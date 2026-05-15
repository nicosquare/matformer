"""Configuration helpers for MatFormer reproduction runs."""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


VALID_GRANULARITIES = {"s", "m", "l", "xl"}
VALID_MODEL_FAMILIES = {"nested", "standalone"}
VALID_COMPLETION_LABELS = {
    "debug",
    "reduced-token-pilot",
    "matlm-10b-budget-reference",
    "paper-budget-complete",
}
VALID_GRANULARITY_SAMPLING = {"all", "random"}
VALID_SAMPLING_MODES = {"nested-random", "nested-all", "standalone"}
GRANULARITY_INTERMEDIATE_FRACTIONS = {
    "s": (1, 8),
    "m": (1, 4),
    "l": (1, 2),
    "xl": (1, 1),
}

PAPER_78M_TOKEN_BUDGET = 10_000_000_000
PAPER_ALIGNED_ARCHITECTURE = {
    "num_layers": 16,
    "num_attention_heads": 16,
    "context_length": 1024,
    "vocab_size_assumption": 256000,
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


def resolve_run_config(
    config_path: str | Path,
    run_id: str | None = None,
    overrides: Mapping[str, Any] | Iterable[str] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = apply_overrides(load_yaml_config(config_path), overrides)

    if "matrix" in config:
        run_entry = _select_matrix_run(config, run_id)
        resolved = _compose_matrix_run(config, run_entry)
    else:
        resolved = _compose_single_run(config)
        if run_id is not None and resolved["run"].get("run_id") != run_id:
            raise ConfigError(
                f"Requested run_id={run_id}, but config defines "
                f"run_id={resolved['run'].get('run_id')}"
            )

    if output_dir is not None:
        resolved["run"]["output_dir"] = str(output_dir)

    _resolve_output_paths(resolved)
    _resolve_training_length(resolved)
    _resolve_parameter_reporting_defaults(resolved)
    validate_run_config(resolved)
    return resolved


def resolve_all_run_configs(
    config_path: str | Path,
    overrides: Mapping[str, Any] | Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    config = apply_overrides(load_yaml_config(config_path), overrides)

    if "matrix" not in config:
        resolved = _compose_single_run(config)
        _resolve_output_paths(resolved)
        _resolve_training_length(resolved)
        _resolve_parameter_reporting_defaults(resolved)
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
        _resolve_output_paths(resolved)
        _resolve_training_length(resolved)
        _resolve_parameter_reporting_defaults(resolved)
        validate_run_config(resolved)
        resolved_runs.append(resolved)

    return resolved_runs


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
        parameter_counts = copy.deepcopy(dict(selected_counts))
        parameter_counts["mismatch_notes"] = _parameter_mismatch_notes(
            config,
            parameter_counts,
        )
        config["parameter_counts"] = parameter_counts

    _resolve_parameter_reporting_defaults(config)


def validate_run_config(config: Mapping[str, Any]) -> None:
    run = _require_mapping(config, "run")
    model = _require_mapping(config, "model")
    training = _require_mapping(config, "training")
    dataset = _require_mapping(config, "dataset")
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
    _require_one_of_fields(
        model,
        "model",
        ["paper_alignment_claim", "paper_aligned"],
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

    model_family = run["model_family"]
    if model_family not in VALID_MODEL_FAMILIES:
        raise ConfigError(f"Unknown model family: {model_family}")

    completion_label = run["completion_label"]
    if completion_label not in VALID_COMPLETION_LABELS:
        raise ConfigError(f"Unknown completion label: {completion_label}")

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

    if model_family == "standalone":
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

    paper_alignment_claim = model.get(
        "paper_alignment_claim",
        model.get("paper_aligned", False),
    )
    if paper_alignment_claim:
        for field_name, expected_value in PAPER_ALIGNED_ARCHITECTURE.items():
            if model.get(field_name) != expected_value:
                raise ConfigError(
                    f"paper-alignment claims require model.{field_name}="
                    f"{expected_value}, got {model.get(field_name)}"
                )

    _validate_dmodel256_pilot_fields(run, model, training, completion_label)

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
    for section_name in ["model", "training", "dataset", "outputs", "evaluation"]:
        if section_name in config:
            resolved[section_name] = copy.deepcopy(config[section_name])

    run = copy.deepcopy(config.get("run", {}))
    run.update(copy.deepcopy(dict(run_entry)))
    resolved["run"] = run
    _apply_run_granularities(resolved)
    return resolved


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


def _table_reference_label(run: Mapping[str, Any]) -> str | None:
    label = run.get("table_reference_label")
    if label is None:
        return None
    return str(label)


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
        return

    if model_family != "nested":
        raise ConfigError(f"run.sampling_mode={sampling_mode} requires nested")

    expected_sampling = {
        "nested-random": "random",
        "nested-all": "all",
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
    completion_label: str,
) -> None:
    model_shape_label = _model_shape_label(run)
    table_reference_label = _table_reference_label(run)
    is_legacy_78m = str(run.get("model_size_label")) == "78m"
    is_dmodel256_pilot = (
        model_shape_label == "dmodel256"
        or table_reference_label == "matlm_78m"
        or is_legacy_78m
    )
    if not is_dmodel256_pilot:
        return

    token_budget = training["token_budget"]
    if token_budget < PAPER_78M_TOKEN_BUDGET:
        expected_label = "reduced-token-pilot"
    elif is_legacy_78m:
        expected_label = "paper-budget-complete"
    else:
        expected_label = "matlm-10b-budget-reference"

    if completion_label != expected_label:
        raise ConfigError(
            f"{model_shape_label or '78m'} token_budget={token_budget} "
            f"requires completion_label={expected_label}"
        )

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
        _validate_granularity_prefixes(model["granularity_prefixes"])

    paper_alignment_claim = model.get(
        "paper_alignment_claim",
        model.get("paper_aligned", False),
    )
    if table_reference_label and not paper_alignment_claim:
        mismatch_notes = model.get("mismatch_notes")
        if not isinstance(mismatch_notes, list) or not mismatch_notes:
            raise ConfigError(
                "table_reference_label requires non-empty model.mismatch_notes "
                "unless paper_alignment_claim=true"
            )


def _validate_granularity_prefixes(prefixes: Any) -> None:
    if not isinstance(prefixes, Mapping):
        raise ConfigError("model.granularity_prefixes must be a mapping")

    missing = sorted(VALID_GRANULARITIES - set(prefixes))
    if missing:
        raise ConfigError(f"model.granularity_prefixes missing keys: {missing}")

    for granularity in VALID_GRANULARITIES:
        try:
            value = float(prefixes[granularity])
        except (TypeError, ValueError) as error:
            raise ConfigError(
                f"model.granularity_prefixes.{granularity} must be numeric"
            ) from error
        if value <= 0:
            raise ConfigError(
                f"model.granularity_prefixes.{granularity} must be positive"
            )


def _apply_standalone_fixed_width(model: dict[str, Any], granularity: str) -> None:
    if "intermediate_size" not in model:
        return

    source_intermediate_size = int(
        model.get("matformer_source_intermediate_size", model["intermediate_size"])
    )
    if granularity not in GRANULARITY_INTERMEDIATE_FRACTIONS:
        raise ConfigError(f"Unknown granularity for standalone run: {granularity}")
    numerator, denominator = GRANULARITY_INTERMEDIATE_FRACTIONS[granularity]
    intermediate_size = source_intermediate_size * numerator // denominator
    if intermediate_size <= 0:
        raise ConfigError(
            f"Granularity {granularity} produced empty standalone FFN width for "
            f"intermediate_size={source_intermediate_size}"
        )

    model["matformer_source_intermediate_size"] = source_intermediate_size
    model["intermediate_size"] = intermediate_size


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
        output_dir = output_root / str(run["run_id"])
    run["output_dir"] = str(output_dir)
    run["explicit_output_dir"] = explicit_output_dir

    _ensure_writable_directory(output_root, "output root")
    if explicit_output_dir:
        _ensure_writable_directory(output_dir.parent, "output directory parent")


def _resolve_training_length(config: dict[str, Any]) -> None:
    resolve_training_length_for_world_size(config)


def _resolve_parameter_reporting_defaults(config: dict[str, Any]) -> None:
    reporting = config.setdefault("parameter_reporting", {})
    if not isinstance(reporting, dict):
        raise ConfigError("parameter_reporting must be a mapping when provided")

    reporting.setdefault("lm_head_counting", "separately_counted")
    reporting.setdefault("paper_total_parameters", None)
    reporting.setdefault("paper_non_embedding_parameters", None)
    reporting.setdefault("paper_ffn_parameters", None)
    reporting.setdefault("paper_attention_parameters", None)
    reporting.setdefault("mismatch_notes", _parameter_mismatch_notes(config))


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


def _parameter_mismatch_notes(
    config: Mapping[str, Any],
    parameter_counts: Mapping[str, Any] | None = None,
) -> list[str]:
    notes: list[str] = []
    model = config.get("model", {})
    reporting = config.get("parameter_reporting", {})

    notes.extend(str(note) for note in model.get("mismatch_notes", []) if note)
    if isinstance(reporting, Mapping):
        notes.extend(str(note) for note in reporting.get("mismatch_notes", []) if note)
    if parameter_counts is not None:
        notes.extend(
            str(note)
            for note in parameter_counts.get("mismatch_notes", [])
            if note
        )

    deduplicated_notes: list[str] = []
    for note in notes:
        if note not in deduplicated_notes:
            deduplicated_notes.append(note)
    return deduplicated_notes


def resolve_training_length_for_world_size(
    config: dict[str, Any],
    effective_world_size: int | None = None,
    world_size_source: str | None = None,
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


def _resolve_effective_world_size() -> int:
    raw_world_size = os.environ.get("WORLD_SIZE")
    if raw_world_size in (None, ""):
        return 1
    return _positive_int(raw_world_size, "WORLD_SIZE")


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
