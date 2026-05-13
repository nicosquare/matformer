"""Configuration helpers for MatFormer reproduction runs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


VALID_GRANULARITIES = {"s", "m", "l", "xl"}
VALID_MODEL_FAMILIES = {"nested", "standalone"}
VALID_COMPLETION_LABELS = {"debug", "reduced-token-pilot", "paper-budget-complete"}

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

    _ensure_output_dir(resolved)
    validate_run_config(resolved)
    return resolved


def resolve_all_run_configs(
    config_path: str | Path,
    overrides: Mapping[str, Any] | Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    config = apply_overrides(load_yaml_config(config_path), overrides)

    if "matrix" not in config:
        resolved = _compose_single_run(config)
        _ensure_output_dir(resolved)
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
        _ensure_output_dir(resolved)
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
            "model_size_label",
            "completion_label",
            "output_dir",
        ],
    )
    _require_fields(
        model,
        "model",
        [
            "base_model_name",
            "paper_aligned",
            "num_layers",
            "num_attention_heads",
            "hidden_size",
            "intermediate_size",
            "context_length",
            "vocab_size_assumption",
            "granularities",
        ],
    )
    _require_fields(training, "training", ["token_budget"])
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

    if model.get("paper_aligned"):
        for field_name, expected_value in PAPER_ALIGNED_ARCHITECTURE.items():
            if model.get(field_name) != expected_value:
                raise ConfigError(
                    f"paper-aligned runs require model.{field_name}="
                    f"{expected_value}, got {model.get(field_name)}"
                )

    if str(run["model_size_label"]) == "78m":
        token_budget = training["token_budget"]
        if token_budget < PAPER_78M_TOKEN_BUDGET:
            expected_label = "reduced-token-pilot"
        else:
            expected_label = "paper-budget-complete"

        if completion_label != expected_label:
            raise ConfigError(
                f"78m token_budget={token_budget} requires "
                f"completion_label={expected_label}"
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
        model["granularities"] = [run["granularity"]]
    elif "granularities" in run:
        model["granularities"] = list(run["granularities"])


def _ensure_output_dir(config: dict[str, Any]) -> None:
    run = config.setdefault("run", {})
    if "output_dir" not in run and "run_id" in run:
        output_root = run.get("output_root", "outputs")
        run["output_dir"] = str(Path(str(output_root)) / str(run["run_id"]))


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
