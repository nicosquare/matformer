"""Milestone raw-gradient compatibility measurement and durable journal state."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.training.gradient_probe import (
    ControlledGradientProbeError,
    measure_controlled_gradients,
)
from src.utils.artifact_io import append_jsonl_artifact


SCHEMA_VERSION = 1
EVENT_TYPE = "gradient_interference_snapshot"
EMPTY_JOURNAL_HASH = hashlib.sha256(b"").hexdigest()


class GradientInterferenceError(ValueError):
    """Raised when diagnostic measurement or journal provenance is invalid."""


def uses_gradient_interference(config: Mapping[str, Any]) -> bool:
    diagnostic = config.get("evaluation", {}).get("gradient_interference", {})
    return bool(isinstance(diagnostic, Mapping) and diagnostic.get("enabled", False))


def snapshot_id(config: Mapping[str, Any], step: int) -> str:
    diagnostic = config["evaluation"]["gradient_interference"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(config["run"]["run_id"]),
        "step": int(step),
        "diagnostic_contract_hash": diagnostic["diagnostic_contract_hash"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_gradient_interference_state(
    config: Mapping[str, Any],
    *,
    fixed_probe_manifest_hash: str,
    controlled_support_hash: str,
) -> dict[str, Any]:
    diagnostic = config["evaluation"]["gradient_interference"]
    expected = copy.deepcopy(list(diagnostic["resolved_milestones"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "expected_milestones": expected,
        "completed_snapshot_ids": [],
        "completed_steps": [],
        "identity": {
            "run_id": str(config["run"]["run_id"]),
            "fixed_probe_manifest_hash": str(fixed_probe_manifest_hash),
            "controlled_support_hash": str(controlled_support_hash),
            "diagnostic_contract_hash": str(diagnostic["diagnostic_contract_hash"]),
        },
        "journal": {
            "path": str(
                Path(config["run"]["output_dir"]) / diagnostic["artifact_path"]
            ),
            "event_count": 0,
            "last_committed_offset": 0,
            "last_committed_hash": EMPTY_JOURNAL_HASH,
        },
        "cost": {
            "packed_sequences": 0,
            "batches": 0,
            "targets": 0,
            "backward_evaluations": 0,
            "duration_seconds": 0.0,
        },
    }


def validate_gradient_interference_state(
    state: Mapping[str, Any] | None,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if not uses_gradient_interference(config):
        if state is not None:
            raise GradientInterferenceError(
                "non-diagnostic checkpoint contains gradient_interference_state"
            )
        return {}
    if not isinstance(state, Mapping):
        raise GradientInterferenceError(
            "checkpoint is missing gradient_interference_state"
        )
    result = copy.deepcopy(dict(state))
    diagnostic = config["evaluation"]["gradient_interference"]
    expected_identity = {
        "run_id": str(config["run"]["run_id"]),
        "fixed_probe_manifest_hash": str(diagnostic.get("fixed_probe_manifest_hash")),
        "controlled_support_hash": str(diagnostic.get("controlled_support_hash")),
        "diagnostic_contract_hash": str(diagnostic["diagnostic_contract_hash"]),
    }
    if result.get("schema_version") != SCHEMA_VERSION:
        raise GradientInterferenceError(
            "gradient_interference_state schema version mismatch"
        )
    if result.get("identity") != expected_identity:
        raise GradientInterferenceError(
            "gradient_interference_state config/probe/support identity mismatch"
        )
    if result.get("expected_milestones") != diagnostic["resolved_milestones"]:
        raise GradientInterferenceError(
            "gradient_interference_state milestone schedule mismatch"
        )
    ids = result.get("completed_snapshot_ids")
    steps = result.get("completed_steps")
    expected_steps = set(diagnostic["resolved_steps"])
    if (
        not isinstance(ids, list)
        or not isinstance(steps, list)
        or len(ids) != len(steps)
        or len(set(ids)) != len(ids)
        or len(set(steps)) != len(steps)
        or steps != sorted(steps)
        or any(
            not isinstance(identifier, str) or len(identifier) != 64
            for identifier in ids
        )
        or any(
            isinstance(step, bool)
            or not isinstance(step, int)
            or step not in expected_steps
            for step in steps
        )
        or any(
            identifier != snapshot_id(config, step)
            for identifier, step in zip(ids, steps, strict=True)
        )
    ):
        raise GradientInterferenceError(
            "gradient_interference_state completed snapshot provenance is invalid"
        )
    journal = result.get("journal")
    expected_path = str(Path(config["run"]["output_dir"]) / diagnostic["artifact_path"])
    if (
        not isinstance(journal, Mapping)
        or journal.get("path") != expected_path
        or isinstance(journal.get("event_count"), bool)
        or not isinstance(journal.get("event_count"), int)
        or journal["event_count"] != len(ids)
        or isinstance(journal.get("last_committed_offset"), bool)
        or not isinstance(journal.get("last_committed_offset"), int)
        or journal["last_committed_offset"] < 0
        or not isinstance(journal.get("last_committed_hash"), str)
        or len(journal["last_committed_hash"]) != 64
    ):
        raise GradientInterferenceError(
            "gradient_interference_state journal provenance is invalid"
        )
    cost = result.get("cost")
    if not isinstance(cost, Mapping):
        raise GradientInterferenceError(
            "gradient_interference_state cost provenance is invalid"
        )
    for field in ("packed_sequences", "batches", "targets", "backward_evaluations"):
        value = cost.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GradientInterferenceError(
                "gradient_interference_state cost provenance is invalid"
            )
    duration = cost.get("duration_seconds")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) < 0.0
    ):
        raise GradientInterferenceError(
            "gradient_interference_state cost provenance is invalid"
        )
    return result


def milestone_reasons(config: Mapping[str, Any], step: int) -> list[str] | None:
    reasons = config["evaluation"]["gradient_interference"]["milestone_reasons"].get(
        str(int(step))
    )
    return None if reasons is None else list(reasons)


def milestone_due(
    config: Mapping[str, Any], state: Mapping[str, Any], step: int
) -> bool:
    reasons = milestone_reasons(config, step)
    return reasons is not None and snapshot_id(config, step) not in set(
        state["completed_snapshot_ids"]
    )


def _relative_shared_tensor(
    larger: Mapping[str, Any], smaller: Mapping[str, Any]
) -> torch.Tensor:
    tensor = larger["tensor"]
    smaller_tensor = smaller["tensor"]
    larger_selection = larger["selection"]
    smaller_selection = smaller["selection"]
    if len(larger_selection) != len(smaller_selection):
        raise GradientInterferenceError("nested support rank mismatch")
    slices = []
    for dimension, (large_slice, small_slice) in enumerate(
        zip(larger_selection, smaller_selection, strict=True)
    ):
        large_step = large_slice.get("step") or 1
        small_step = small_slice.get("step") or 1
        if large_step != 1 or small_step != 1:
            raise GradientInterferenceError(
                "strided controlled support is not supported"
            )
        large_start = large_slice.get("start") or 0
        small_start = small_slice.get("start") or 0
        relative_start = int(small_start) - int(large_start)
        length = int(smaller_tensor.shape[dimension])
        if relative_start < 0 or relative_start + length > int(tensor.shape[dimension]):
            raise GradientInterferenceError(
                "granularity supports are not nested on shared coordinates"
            )
        slices.append(slice(relative_start, relative_start + length))
    selected = tensor[tuple(slices)]
    if selected.shape != smaller_tensor.shape:
        raise GradientInterferenceError("nested shared-support shape mismatch")
    return selected


def _pair_statistics(
    left: str,
    right: str,
    snapshots: Mapping[str, Any],
    *,
    include_layerwise: bool,
) -> dict[str, Any]:
    left_layers = snapshots[left]
    right_layers = snapshots[right]
    if list(left_layers) != list(right_layers):
        raise GradientInterferenceError("controlled-gradient layer order mismatch")
    layer_contributions = []
    total_count = 0
    total_dot = 0.0
    total_left_sq = 0.0
    total_right_sq = 0.0
    for layer_key in left_layers:
        left_layer = left_layers[layer_key]
        right_layer = right_layers[layer_key]
        layer_count = 0
        layer_dot = 0.0
        layer_left_sq = 0.0
        layer_right_sq = 0.0
        for parameter_name, left_entry in left_layer["entries"].items():
            right_entry = right_layer["entries"].get(parameter_name)
            if right_entry is None:
                raise GradientInterferenceError(
                    "larger granularity is missing smaller shared support"
                )
            left_tensor = left_entry["tensor"].reshape(-1).double()
            right_tensor = (
                _relative_shared_tensor(right_entry, left_entry).reshape(-1).double()
            )
            layer_count += int(left_tensor.numel())
            layer_dot += float(torch.dot(left_tensor, right_tensor).item())
            layer_left_sq += float(torch.dot(left_tensor, left_tensor).item())
            layer_right_sq += float(torch.dot(right_tensor, right_tensor).item())
        total_count += layer_count
        total_dot += layer_dot
        total_left_sq += layer_left_sq
        total_right_sq += layer_right_sq
        if include_layerwise:
            layer_contributions.append(
                {
                    "layer": layer_key,
                    "module_name": left_layer["module_name"],
                    "shared_parameter_count": layer_count,
                    "dot_product": layer_dot,
                    "left_shared_squared_norm": layer_left_sq,
                    "right_shared_squared_norm": layer_right_sq,
                }
            )
    distance_squared = total_left_sq + total_right_sq - 2.0 * total_dot
    tolerance = 1e-10 * max(1.0, total_left_sq + total_right_sq)
    if distance_squared < -tolerance:
        raise GradientInterferenceError("shared-gradient distance is negative")
    distance_squared = max(0.0, distance_squared)
    left_norm = math.sqrt(total_left_sq)
    right_norm = math.sqrt(total_right_sq)
    left_zero = left_norm == 0.0
    right_zero = right_norm == 0.0
    cosine = None
    if not left_zero and not right_zero:
        cosine = total_dot / (left_norm * right_norm)
        cosine = max(-1.0, min(1.0, cosine))
    return {
        "left_granularity": left,
        "right_granularity": right,
        "distance": math.sqrt(distance_squared),
        "shared_parameter_count": total_count,
        "dot_product": total_dot,
        "left_shared_squared_norm": total_left_sq,
        "right_shared_squared_norm": total_right_sq,
        "left_shared_norm": left_norm,
        "right_shared_norm": right_norm,
        "cosine": cosine,
        "has_zero_norm": left_zero or right_zero,
        "zero_norm": {"left": left_zero, "right": right_zero},
        "layer_contributions": layer_contributions,
    }


def measure_gradient_interference(
    model,
    dataloader,
    granularities: Sequence[str],
    *,
    device: torch.device | str,
    config: Mapping[str, Any],
    support_identity: Mapping[str, Any],
    step: int,
    tokens_seen: int,
) -> dict[str, Any]:
    """Measure one complete milestone snapshot without retaining gradients."""

    ordered = [str(label) for label in granularities]
    try:
        measurement = measure_controlled_gradients(
            model,
            dataloader,
            ordered,
            device=device,
            config=config,
            support_identity=support_identity,
            retain_gradients=True,
        )
    except ControlledGradientProbeError as error:
        raise GradientInterferenceError(str(error)) from error
    snapshots = measurement.pop("_gradient_snapshots")
    include_layerwise = bool(config["evaluation"]["gradient_interference"]["layerwise"])
    pairs = [
        _pair_statistics(
            ordered[left_index],
            ordered[right_index],
            snapshots,
            include_layerwise=include_layerwise,
        )
        for left_index in range(len(ordered))
        for right_index in range(left_index + 1, len(ordered))
    ]
    del snapshots

    diagnostic = config["evaluation"]["gradient_interference"]
    record = {
        "schema_version": SCHEMA_VERSION,
        "event_type": EVENT_TYPE,
        "snapshot_id": snapshot_id(config, step),
        "run_id": str(config["run"]["run_id"]),
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "milestone_reasons": milestone_reasons(config, step),
        "fixed_probe_manifest_hash": diagnostic["fixed_probe_manifest_hash"],
        "controlled_support_hash": support_identity["controlled_support_hash"],
        "diagnostic_contract_hash": diagnostic["diagnostic_contract_hash"],
        "semantics": {
            "gradient": diagnostic["gradient_semantics"],
            "loss_aggregation": diagnostic["loss_aggregation"],
            "shared_support": diagnostic["shared_support"],
            "layerwise": include_layerwise,
        },
        "granularities": measurement["measurements"],
        "pairs": pairs,
        "cost": {
            "packed_sequences": int(measurement["controller_packed_sequence_count"]),
            "batches": int(measurement["controller_batch_count"]),
            "targets": int(measurement["controller_target_count"]),
            "packed_sequence_evaluations": int(
                measurement["controller_packed_sequence_evaluation_count"]
            ),
            "target_evaluations": int(
                measurement["controller_target_evaluation_count"]
            ),
            "backward_evaluations": int(measurement["backward_evaluation_count"]),
            "duration_seconds": float(measurement["duration_seconds"]),
        },
    }
    if measurement.get("controller_source_example_count") is not None:
        record["cost"]["source_examples"] = int(
            measurement["controller_source_example_count"]
        )
    validate_snapshot_record(record, expected_granularity_count=len(ordered))
    return record


def _validate_finite_json(value: Any, path: str = "record") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GradientInterferenceError(f"{path} contains a non-finite value")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_json(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{path}[{index}]")
        return
    raise GradientInterferenceError(f"{path} contains a non-JSON value")


def validate_snapshot_record(
    record: Mapping[str, Any], *, expected_granularity_count: int | None = None
) -> None:
    required = {
        "schema_version",
        "event_type",
        "snapshot_id",
        "run_id",
        "step",
        "tokens_seen",
        "milestone_reasons",
        "fixed_probe_manifest_hash",
        "controlled_support_hash",
        "diagnostic_contract_hash",
        "semantics",
        "granularities",
        "pairs",
        "cost",
    }
    if required - set(record):
        raise GradientInterferenceError("gradient-interference record is incomplete")
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("event_type") != EVENT_TYPE
    ):
        raise GradientInterferenceError(
            "gradient-interference record identity is invalid"
        )
    if (
        not isinstance(record.get("snapshot_id"), str)
        or len(record["snapshot_id"]) != 64
    ):
        raise GradientInterferenceError("gradient-interference snapshot ID is invalid")
    granularities = record.get("granularities")
    pairs = record.get("pairs")
    if not isinstance(granularities, list) or not isinstance(pairs, list):
        raise GradientInterferenceError("gradient-interference vectors are invalid")
    count = len(granularities)
    if expected_granularity_count is not None and count != expected_granularity_count:
        raise GradientInterferenceError(
            "gradient-interference granularity count mismatch"
        )
    if len(pairs) != count * (count - 1) // 2:
        raise GradientInterferenceError(
            "gradient-interference pair cardinality mismatch"
        )
    _validate_finite_json(record)
    try:
        json.dumps(record, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise GradientInterferenceError(
            "gradient-interference record is not finite JSON"
        ) from error


def snapshot_records_equivalent(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    """Compare rank-local snapshots while excluding wall-clock duration."""

    def compare(first: Any, second: Any, path: tuple[str, ...]) -> bool:
        if path[-2:] == ("cost", "duration_seconds"):
            return True
        if isinstance(first, Mapping) and isinstance(second, Mapping):
            return set(first) == set(second) and all(
                compare(first[key], second[key], (*path, str(key))) for key in first
            )
        if isinstance(first, list) and isinstance(second, list):
            return len(first) == len(second) and all(
                compare(a, b, (*path, str(index)))
                for index, (a, b) in enumerate(zip(first, second, strict=True))
            )
        if (
            isinstance(first, (int, float))
            and not isinstance(first, bool)
            and isinstance(second, (int, float))
            and not isinstance(second, bool)
        ):
            return math.isclose(float(first), float(second), rel_tol=1e-6, abs_tol=1e-8)
        return first == second

    return compare(left, right, ())


def append_snapshot_record(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    artifact_io: Mapping[str, Any] | None = None,
    heartbeat_writer=None,
    artifact_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_snapshot_record(record)
    commit = append_jsonl_artifact(
        path,
        record,
        settings=artifact_io,
        heartbeat_writer=heartbeat_writer,
        state=artifact_state,
    )
    journal_path = Path(path)
    commit["journal_hash"] = hashlib.sha256(journal_path.read_bytes()).hexdigest()
    return commit


def record_snapshot_commit(
    state: dict[str, Any],
    record: Mapping[str, Any],
    commit: Mapping[str, Any],
) -> None:
    identifier = str(record["snapshot_id"])
    step = int(record["step"])
    if (
        identifier in state["completed_snapshot_ids"]
        or step in state["completed_steps"]
    ):
        raise GradientInterferenceError("duplicate gradient-interference snapshot")
    state["completed_snapshot_ids"].append(identifier)
    state["completed_steps"].append(step)
    state["journal"].update(
        {
            "event_count": int(state["journal"]["event_count"]) + 1,
            "last_committed_offset": int(commit["last_committed_offset"]),
            "last_committed_hash": str(commit["journal_hash"]),
        }
    )
    for field in ("packed_sequences", "batches", "targets", "backward_evaluations"):
        state["cost"][field] = int(state["cost"][field]) + int(record["cost"][field])
    state["cost"]["duration_seconds"] = float(
        state["cost"]["duration_seconds"]
    ) + float(record["cost"]["duration_seconds"])


def _read_prefix_records(payload: bytes) -> list[dict[str, Any]]:
    if payload and not payload.endswith(b"\n"):
        raise GradientInterferenceError(
            "committed gradient-interference journal offset is not a record boundary"
        )
    records = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise GradientInterferenceError(
                f"invalid gradient-interference JSON on line {line_number}"
            ) from error
        validate_snapshot_record(record)
        records.append(record)
    return records


def reconcile_snapshot_journal(
    state: dict[str, Any],
    *,
    config: Mapping[str, Any],
    restored_step: int,
) -> list[dict[str, Any]]:
    """Validate the checkpointed prefix and discard any newer journal tail."""

    validated = validate_gradient_interference_state(state, config=config)
    state.clear()
    state.update(validated)
    path = Path(state["journal"]["path"])
    offset = int(state["journal"]["last_committed_offset"])
    if not path.exists():
        if offset != 0:
            raise GradientInterferenceError(
                "committed gradient-interference journal is missing"
            )
        payload = b""
    else:
        payload = path.read_bytes()
        if len(payload) < offset:
            raise GradientInterferenceError(
                "gradient-interference journal is shorter than its checkpointed offset"
            )
        committed = payload[:offset]
        actual_hash = hashlib.sha256(committed).hexdigest()
        if actual_hash != state["journal"]["last_committed_hash"]:
            raise GradientInterferenceError(
                "gradient-interference journal hash mismatch"
            )
        if len(payload) > offset:
            with path.open("r+b") as journal_file:
                journal_file.truncate(offset)
                journal_file.flush()
                os.fsync(journal_file.fileno())
        payload = committed
    records = _read_prefix_records(payload)
    identifiers = [record["snapshot_id"] for record in records]
    steps = [int(record["step"]) for record in records]
    if len(set(identifiers)) != len(identifiers) or len(set(steps)) != len(steps):
        raise GradientInterferenceError(
            "duplicate snapshot in committed gradient-interference journal"
        )
    if (
        identifiers != state["completed_snapshot_ids"]
        or steps != state["completed_steps"]
    ):
        raise GradientInterferenceError(
            "gradient-interference journal does not match checkpoint provenance"
        )
    if len(records) != int(state["journal"]["event_count"]):
        raise GradientInterferenceError(
            "gradient-interference journal event count mismatch"
        )
    missing_historical = [
        step
        for step in config["evaluation"]["gradient_interference"]["resolved_steps"]
        if int(step) < int(restored_step) and int(step) not in set(steps)
    ]
    if missing_historical:
        raise GradientInterferenceError(
            "gradient-interference journal is missing historical milestones: "
            f"{missing_historical}"
        )
    return records


def summary_fields(
    config: Mapping[str, Any], state: Mapping[str, Any] | None
) -> dict[str, Any]:
    if not uses_gradient_interference(config) or not isinstance(state, Mapping):
        return {}
    expected_steps = list(
        config["evaluation"]["gradient_interference"]["resolved_steps"]
    )
    measured_steps = list(state["completed_steps"])
    return {
        "gradient_interference_path": state["journal"]["path"],
        "gradient_interference_journal_hash": state["journal"]["last_committed_hash"],
        "gradient_interference_snapshot_count": int(state["journal"]["event_count"]),
        "gradient_interference_measured_steps": measured_steps,
        "gradient_interference_expected_steps": expected_steps,
        "gradient_interference_measurement_cost": copy.deepcopy(state["cost"]),
    }


__all__ = [
    "GradientInterferenceError",
    "append_snapshot_record",
    "build_gradient_interference_state",
    "measure_gradient_interference",
    "milestone_due",
    "reconcile_snapshot_journal",
    "record_snapshot_commit",
    "snapshot_id",
    "snapshot_records_equivalent",
    "summary_fields",
    "uses_gradient_interference",
    "validate_gradient_interference_state",
    "validate_snapshot_record",
]
