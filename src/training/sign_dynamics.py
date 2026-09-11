"""Exact all-parameter sign-dynamics measurement for elastic training.

The runtime deliberately keeps only the previous committed parameter values,
per-coordinate hysteresis/transition state, compact per-step aggregates, and
full milestone snapshots.  It never writes per-coordinate values for ordinary
steps.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.training.gradient_probe import controlled_mlps
from src.utils.reproducibility import stable_hash


SIGN_DYNAMICS_SCHEMA_VERSION = 1
EMPTY_JOURNAL_HASH = hashlib.sha256(b"").hexdigest()


class SignDynamicsError(RuntimeError):
    """Raised when exact sign-dynamics measurement cannot be guaranteed."""


def uses_sign_dynamics(config: Mapping[str, Any]) -> bool:
    diagnostic = config.get("evaluation", {}).get("sign_dynamics", {})
    return isinstance(diagnostic, Mapping) and bool(diagnostic.get("enabled", False))


def _threshold_key(value: float) -> str:
    return format(float(value), ".17g")


def _sha256_hasher(path: Path, *, limit: int | None = None) -> Any:
    digest = hashlib.sha256()
    remaining = limit
    with path.open("rb") as source:
        while remaining is None or remaining > 0:
            size = 8 * 1024 * 1024 if remaining is None else min(remaining, 8 * 1024 * 1024)
            chunk = source.read(size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return digest


def _sha256(path: Path, *, limit: int | None = None) -> str:
    return _sha256_hasher(path, limit=limit).hexdigest()


def _flat_ranges(indices: torch.Tensor) -> list[dict[str, int]]:
    values = indices.detach().to(device="cpu", dtype=torch.int64).tolist()
    if not values:
        return []
    ranges: list[dict[str, int]] = []
    start = previous = int(values[0])
    for value in values[1:]:
        value = int(value)
        if value != previous + 1:
            ranges.append({"start": start, "stop": previous + 1})
            start = value
        previous = value
    ranges.append({"start": start, "stop": previous + 1})
    return ranges


def _layer_index(parameter_name: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", parameter_name)
    return int(match.group(1)) if match else None


def _shared_family(name: str) -> str:
    lowered = name.lower()
    if "embed" in lowered:
        return "embedding"
    if "lm_head" in lowered:
        return "language_model_head"
    if any(token in lowered for token in ("self_attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj")):
        return "attention"
    if any(token in lowered for token in ("norm", "layernorm")):
        return "normalization"
    if ".mlp." in lowered or ".ffn." in lowered:
        return "shared_ffn"
    return "other_shared"


@dataclass(frozen=True)
class SupportSegment:
    segment_id: str
    parameter_name: str
    parameter: torch.nn.Parameter
    flat_indices: torch.Tensor
    stratum: str
    primary_band: str
    layer: int | None
    family: str

    @property
    def coordinate_count(self) -> int:
        return int(self.flat_indices.numel())

    def select(self, tensor: torch.Tensor) -> torch.Tensor:
        indices = self.flat_indices.to(device=tensor.device)
        return torch.index_select(tensor.reshape(-1), 0, indices)


def resolve_sign_dynamics_support(
    model: torch.nn.Module,
    granularities: Sequence[str],
) -> tuple[dict[str, Any], list[SupportSegment]]:
    """Resolve the literal B partition and its non-overlapping audit strata."""

    target = model.module if hasattr(model, "module") else model
    ordered = [str(label) for label in granularities]
    if len(ordered) < 2 or len(set(ordered)) != len(ordered):
        raise SignDynamicsError("sign dynamics requires at least two unique granularities")

    named_parameters = [
        (str(name), parameter)
        for name, parameter in target.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        raise SignDynamicsError("sign dynamics found no trainable parameters")
    name_by_identity = {id(parameter): name for name, parameter in named_parameters}
    band_masks = {
        name: torch.full((parameter.numel(),), -1, dtype=torch.int16)
        for name, parameter in named_parameters
    }
    previous_controlled_masks = {
        name: torch.zeros(parameter.numel(), dtype=torch.bool)
        for name, parameter in named_parameters
    }
    controlled_family: dict[tuple[str, int], str] = {}

    mlps = controlled_mlps(target)
    if not mlps:
        raise SignDynamicsError("sign dynamics found no MatFormer FFN layers")
    variants = {
        "concat" if module.__class__.__name__ == "CatLlamaMLP" else "slicing"
        for _, module in mlps
    }
    if len(variants) != 1:
        raise SignDynamicsError("sign dynamics requires a single FFN layout variant")

    for band_index, granularity in enumerate(ordered, start=1):
        current_controlled_masks = {
            name: torch.zeros_like(mask)
            for name, mask in previous_controlled_masks.items()
        }
        for _module_name, mlp in mlps:
            try:
                entries = mlp.controlled_ffn_support(granularity)
            except ValueError as error:
                raise SignDynamicsError(str(error)) from error
            for entry in entries:
                parameter_name = name_by_identity.get(id(entry.parameter))
                if parameter_name is None:
                    if entry.parameter.requires_grad:
                        raise SignDynamicsError(
                            "controlled FFN support contains an unnamed trainable parameter"
                        )
                    continue
                current_controlled_masks[parameter_name].reshape(
                    entry.parameter.shape
                )[entry.selection] = True
                controlled_family[(parameter_name, band_index)] = str(
                    entry.parameter_family
                )
        for parameter_name in band_masks:
            previous = previous_controlled_masks[parameter_name]
            current = current_controlled_masks[parameter_name]
            if bool(torch.any(previous & ~current)):
                raise SignDynamicsError(
                    "configured granularities do not form ordered nested FFN support"
                )
            band_masks[parameter_name][current & ~previous] = band_index
        previous_controlled_masks = current_controlled_masks

    segments: list[SupportSegment] = []
    manifest_parameters: list[dict[str, Any]] = []
    stratum_counts = {"shared": 0, **{f"ffn_B{j}": 0 for j in range(1, len(ordered) + 1)}}
    band_counts = {f"B{j}": 0 for j in range(1, len(ordered) + 1)}
    total_count = 0
    for parameter_name, parameter in named_parameters:
        mask = band_masks[parameter_name]
        mask[mask.eq(-1)] = 0
        parameter_slices = []
        for band_index in sorted(int(value) for value in torch.unique(mask).tolist()):
            indices = torch.nonzero(mask.eq(band_index), as_tuple=False).reshape(-1)
            if indices.numel() == 0:
                continue
            stratum = "shared" if band_index == 0 else f"ffn_B{band_index}"
            primary_band = "B1" if band_index <= 1 else f"B{band_index}"
            family = (
                _shared_family(parameter_name)
                if band_index == 0
                else controlled_family.get((parameter_name, band_index), "controlled_ffn")
            )
            segment_id = f"{parameter_name}:{stratum}"
            segment = SupportSegment(
                segment_id=segment_id,
                parameter_name=parameter_name,
                parameter=parameter,
                flat_indices=indices,
                stratum=stratum,
                primary_band=primary_band,
                layer=_layer_index(parameter_name),
                family=family,
            )
            segments.append(segment)
            stratum_counts[stratum] += segment.coordinate_count
            band_counts[primary_band] += segment.coordinate_count
            total_count += segment.coordinate_count
            parameter_slices.append(
                {
                    "segment_id": segment_id,
                    "primary_band": primary_band,
                    "stratum": stratum,
                    "layer": segment.layer,
                    "family": family,
                    "flat_ranges": _flat_ranges(indices),
                    "coordinate_count": segment.coordinate_count,
                }
            )
        if sum(item["coordinate_count"] for item in parameter_slices) != parameter.numel():
            raise SignDynamicsError(
                f"support does not exactly cover trainable parameter {parameter_name!r}"
            )
        manifest_parameters.append(
            {
                "parameter_name": parameter_name,
                "shape": list(parameter.shape),
                "coordinate_count": int(parameter.numel()),
                "slices": parameter_slices,
            }
        )

    expected_total = sum(parameter.numel() for _, parameter in named_parameters)
    if total_count != expected_total or sum(band_counts.values()) != expected_total:
        raise SignDynamicsError(
            "sign-dynamics support is not an exact disjoint cover of the trainable model"
        )
    if any(stratum_counts[f"ffn_B{j}"] <= 0 for j in range(1, len(ordered) + 1)):
        raise SignDynamicsError("every configured granularity must add controlled FFN support")

    manifest: dict[str, Any] = {
        "schema_version": SIGN_DYNAMICS_SCHEMA_VERSION,
        "measurement_scope": "all_trainable_parameters",
        "variant": next(iter(variants)),
        "ordered_granularities": ordered,
        "primary_partition": {
            "definition": "B1=A1; Bj=Aj\\A(j-1)",
            "bands": [f"B{j}" for j in range(1, len(ordered) + 1)],
            "counts": band_counts,
        },
        "audit_strata": {
            "definition": "B1=shared+ffn_B1; Bj=ffn_Bj for j>1",
            "strata": ["shared", *[f"ffn_B{j}" for j in range(1, len(ordered) + 1)]],
            "counts": stratum_counts,
        },
        "trainable_parameter_count": int(expected_total),
        "named_trainable_parameter_count": len(named_parameters),
        "layer_count": len(mlps),
        "parameters": manifest_parameters,
        "coverage": {
            "disjoint": True,
            "exhaustive": True,
            "covered_coordinate_count": int(total_count),
        },
    }
    manifest["support_hash"] = stable_hash(manifest)
    return manifest, segments


def strict_sign_flip_mask(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    return ((previous > 0) & (current < 0)) | ((previous < 0) & (current > 0))


def update_hysteresis_state(
    previous_state: torch.Tensor,
    normalized_current: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return new state, robust-transition mask, and establishment mask."""

    new_state = previous_state.clone()
    positive = normalized_current >= float(threshold)
    negative = normalized_current <= -float(threshold)
    if float(threshold) == 0.0:
        positive &= normalized_current > 0
        negative &= normalized_current < 0
    new_state[positive] = 1
    new_state[negative] = -1
    transitions = ((previous_state == 1) & (new_state == -1)) | (
        (previous_state == -1) & (new_state == 1)
    )
    establishments = previous_state.eq(0) & new_state.ne(0)
    return new_state, transitions, establishments


class _Aggregate:
    def __init__(self) -> None:
        self.count = 0
        self.old_sq = 0.0
        self.update_sq = 0.0
        self.gradient_sq = 0.0
        self.gradient_present_count = 0
        self.nonzero_gradient_count = 0
        self.changed_count = 0
        self.raw_flip_count = 0
        self.thresholds: dict[str, dict[str, int]] = {}
        self.last_transition: dict[str, tuple[int, int]] = {}

    def add_base(
        self,
        *,
        count: int,
        old_sq: float,
        update_sq: float,
        gradient_sq: float,
        gradient_present_count: int,
        nonzero_gradient_count: int,
        changed_count: int,
        raw_flip_count: int,
    ) -> None:
        self.count += int(count)
        self.old_sq += float(old_sq)
        self.update_sq += float(update_sq)
        self.gradient_sq += float(gradient_sq)
        self.gradient_present_count += int(gradient_present_count)
        self.nonzero_gradient_count += int(nonzero_gradient_count)
        self.changed_count += int(changed_count)
        self.raw_flip_count += int(raw_flip_count)

    def add_threshold(
        self,
        key: str,
        *,
        transition_count: int,
        dead_count: int,
        dead_entries: int,
        dead_exits: int,
        established_count: int,
        last_step: int,
        last_action: int,
    ) -> None:
        item = self.thresholds.setdefault(
            key,
            {
                "transition_count": 0,
                "dead_count": 0,
                "dead_entries": 0,
                "dead_exits": 0,
                "established_count": 0,
            },
        )
        item["transition_count"] += int(transition_count)
        item["dead_count"] += int(dead_count)
        item["dead_entries"] += int(dead_entries)
        item["dead_exits"] += int(dead_exits)
        item["established_count"] += int(established_count)
        if last_step >= self.last_transition.get(key, (-1, -1))[0]:
            self.last_transition[key] = (int(last_step), int(last_action))

    def record(self, ordered: Sequence[str], *, forward_active: bool) -> dict[str, Any]:
        count = max(self.count, 1)
        update_rms = math.sqrt(self.update_sq / count)
        parameter_rms = math.sqrt(self.old_sq / count)
        threshold_records = {}
        for key, item in self.thresholds.items():
            last_step, last_action = self.last_transition.get(key, (-1, -1))
            threshold_records[key] = {
                **item,
                "transition_rate": item["transition_count"] / count,
                "dead_band_occupancy": item["dead_count"] / count,
                "last_robust_transition_step": None if last_step < 0 else last_step,
                "last_robust_transition_granularity": (
                    ordered[last_action]
                    if 0 <= last_action < len(ordered)
                    else None
                ),
            }
        return {
            "coordinate_count": self.count,
            "forward_active": bool(forward_active),
            "gradient_present_coordinate_count": self.gradient_present_count,
            "nonzero_gradient_count": self.nonzero_gradient_count,
            "gradient_rms": math.sqrt(self.gradient_sq / count),
            "changed_coordinate_count": self.changed_count,
            "changed_coordinate_rate": self.changed_count / count,
            "update_rms": update_rms,
            "parameter_rms_before_update": parameter_rms,
            "relative_update_rms": update_rms / (parameter_rms + 1e-30),
            "raw_sign_flip_count": self.raw_flip_count,
            "raw_sign_flip_rate": self.raw_flip_count / count,
            "hysteresis": threshold_records,
        }


class SignDynamicsRuntime:
    """Observe every trainable coordinate at every committed optimizer step."""

    def __init__(
        self,
        config: Mapping[str, Any],
        model: torch.nn.Module,
        support_manifest: Mapping[str, Any],
        segments: Sequence[SupportSegment],
    ) -> None:
        self.config = config
        self.model = model
        self.diagnostic = copy.deepcopy(config["evaluation"]["sign_dynamics"])
        self.ordered = [str(value) for value in config["model"]["granularities"]]
        self.output_dir = Path(str(config["run"]["output_dir"]))
        self.support_path = self.output_dir / "sign_dynamics_support.json"
        self.journal_path = self.output_dir / "sign_dynamics.jsonl"
        self.snapshot_dir = self.output_dir / "sign_dynamics_snapshots"
        self.support_manifest = copy.deepcopy(dict(support_manifest))
        self.segments = list(segments)
        self.parameters = {
            str(name): parameter
            for name, parameter in (model.module if hasattr(model, "module") else model).named_parameters()
            if parameter.requires_grad
        }
        self.previous_values: dict[str, torch.Tensor] = {}
        self.hysteresis_states: dict[str, dict[str, torch.Tensor]] = {}
        self.last_transition_steps: dict[str, dict[str, torch.Tensor]] = {}
        self.last_transition_actions: dict[str, dict[str, torch.Tensor]] = {}
        self.establishment_steps: dict[str, dict[str, torch.Tensor]] = {}
        self.establishment_actions: dict[str, dict[str, torch.Tensor]] = {}
        self.measured_steps = 0
        self.journal_offset = 0
        self.journal_hash = EMPTY_JOURNAL_HASH
        self._journal_hasher = hashlib.sha256()
        self._reconciled_journal_hasher: Any | None = None
        self.snapshot_identities: dict[str, dict[str, Any]] = {}
        self.runtime_seconds = 0.0
        self._pending: dict[str, Any] | None = None

    @classmethod
    def build(cls, config: Mapping[str, Any], model: torch.nn.Module) -> "SignDynamicsRuntime":
        manifest, segments = resolve_sign_dynamics_support(
            model, config["model"]["granularities"]
        )
        manifest["diagnostic_contract_hash"] = config["evaluation"][
            "sign_dynamics"
        ]["diagnostic_contract_hash"]
        manifest["support_hash"] = stable_hash(
            {key: value for key, value in manifest.items() if key != "support_hash"}
        )
        return cls(config, model, manifest, segments)

    def install_support_contract(self, *, resuming: bool) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        expected = copy.deepcopy(self.support_manifest)
        if self.support_path.exists():
            try:
                existing = json.loads(self.support_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise SignDynamicsError("cannot read sign-dynamics support manifest") from error
            if existing != expected:
                raise SignDynamicsError("sign-dynamics support manifest identity mismatch")
        elif resuming:
            raise SignDynamicsError("resume is missing sign_dynamics_support.json")
        else:
            self._atomic_json(self.support_path, expected)

    def initialize_fresh(self) -> None:
        if self.journal_path.exists() and self.journal_path.stat().st_size:
            raise SignDynamicsError("fresh sign-dynamics run found a non-empty journal")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.previous_values = {
            name: parameter.detach().to(dtype=torch.float32).clone()
            for name, parameter in self.parameters.items()
        }
        self._initialize_coordinate_state()
        self.measured_steps = 0
        self.journal_offset = 0
        self.journal_hash = EMPTY_JOURNAL_HASH
        self._journal_hasher = hashlib.sha256()
        self._reconciled_journal_hasher = None
        if 0 in set(self.diagnostic["resolved_snapshot_steps"]):
            self._write_snapshot(0)

    def _initialize_coordinate_state(self) -> None:
        threshold_keys = [_threshold_key(value) for value in self.diagnostic["hysteresis_thresholds"]]
        self.hysteresis_states = {}
        self.last_transition_steps = {}
        self.last_transition_actions = {}
        self.establishment_steps = {}
        self.establishment_actions = {}
        for key in threshold_keys:
            threshold = float(key)
            states = {
                name: torch.zeros_like(value, dtype=torch.int8)
                for name, value in self.previous_values.items()
            }
            establishment_steps = {
                name: torch.full_like(value, -1, dtype=torch.int64)
                for name, value in self.previous_values.items()
            }
            establishment_actions = {
                name: torch.full_like(value, -1, dtype=torch.int16)
                for name, value in self.previous_values.items()
            }
            for segment in self.segments:
                values = segment.select(self.previous_values[segment.parameter_name])
                scale = math.sqrt(float(torch.sum(values.double().square()).item()) / max(segment.coordinate_count, 1))
                normalized = values / (scale + 1e-30)
                selected_state, _, established = update_hysteresis_state(
                    segment.select(states[segment.parameter_name]), normalized, threshold
                )
                indices = segment.flat_indices.to(states[segment.parameter_name].device)
                states[segment.parameter_name].reshape(-1).index_copy_(0, indices, selected_state)
                selected_steps = segment.select(establishment_steps[segment.parameter_name])
                selected_steps[established] = 0
                establishment_steps[segment.parameter_name].reshape(-1).index_copy_(0, indices, selected_steps)
            self.hysteresis_states[key] = states
            self.last_transition_steps[key] = {
                name: torch.full_like(value, -1, dtype=torch.int64)
                for name, value in self.previous_values.items()
            }
            self.last_transition_actions[key] = {
                name: torch.full_like(value, -1, dtype=torch.int16)
                for name, value in self.previous_values.items()
            }
            self.establishment_steps[key] = establishment_steps
            self.establishment_actions[key] = establishment_actions

    def capture_pre_update(self, *, step: int, action: Mapping[str, Any]) -> None:
        if self._pending is not None:
            raise SignDynamicsError("a sign-dynamics measurement is already pending")
        started = time.perf_counter()
        selected = self._selected_action(action)
        gradients: dict[str, dict[str, Any]] = {}
        for segment in self.segments:
            gradient = segment.parameter.grad
            present = gradient is not None
            selected_gradient = (
                torch.zeros(segment.coordinate_count, device=segment.parameter.device, dtype=torch.float32)
                if gradient is None
                else segment.select(gradient.detach()).to(dtype=torch.float32)
            )
            if not bool(torch.all(torch.isfinite(selected_gradient))):
                raise SignDynamicsError(f"non-finite gradient in {segment.segment_id}")
            gradients[segment.segment_id] = {
                "squared_sum": float(torch.sum(selected_gradient.double().square()).item()),
                "present_count": segment.coordinate_count if present else 0,
                "nonzero_count": int(torch.count_nonzero(selected_gradient).item()),
            }
        self._pending = {
            "step": int(step),
            "selected": selected,
            "action": copy.deepcopy(dict(action)),
            "gradients": gradients,
        }
        self.runtime_seconds += time.perf_counter() - started

    def abort_pending(self) -> None:
        self._pending = None

    def commit_step(self, *, step: int, action: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        if self._pending is None or int(self._pending["step"]) != int(step):
            raise SignDynamicsError("sign-dynamics commit has no matching pre-update state")
        selected = self._selected_action(action)
        if selected != self._pending["selected"]:
            raise SignDynamicsError("sign-dynamics action changed across optimizer commit")
        if int(step) != self.measured_steps + 1:
            raise SignDynamicsError("sign-dynamics steps must be contiguous and committed once")
        action_index = self.ordered.index(selected)
        strata = {name: _Aggregate() for name in self.support_manifest["audit_strata"]["strata"]}
        bands = {name: _Aggregate() for name in self.support_manifest["primary_partition"]["bands"]}

        for segment in self.segments:
            old_full = self.previous_values[segment.parameter_name]
            current_full = segment.parameter.detach().to(dtype=torch.float32)
            old = segment.select(old_full)
            current = segment.select(current_full)
            if not bool(torch.all(torch.isfinite(current))):
                raise SignDynamicsError(f"non-finite parameter in {segment.segment_id}")
            update = current - old
            changed = update.ne(0)
            raw_flips = strict_sign_flip_mask(old, current)
            gradient = self._pending["gradients"][segment.segment_id]
            targets = (strata[segment.stratum], bands[segment.primary_band])
            for target in targets:
                target.add_base(
                    count=segment.coordinate_count,
                    old_sq=float(torch.sum(old.double().square()).item()),
                    update_sq=float(torch.sum(update.double().square()).item()),
                    gradient_sq=gradient["squared_sum"],
                    gradient_present_count=gradient["present_count"],
                    nonzero_gradient_count=gradient["nonzero_count"],
                    changed_count=int(torch.count_nonzero(changed).item()),
                    raw_flip_count=int(torch.count_nonzero(raw_flips).item()),
                )

            old_scale = math.sqrt(float(torch.sum(old.double().square()).item()) / max(segment.coordinate_count, 1))
            current_scale = math.sqrt(float(torch.sum(current.double().square()).item()) / max(segment.coordinate_count, 1))
            for threshold_value in self.diagnostic["hysteresis_thresholds"]:
                key = _threshold_key(threshold_value)
                threshold = float(threshold_value)
                state_full = self.hysteresis_states[key][segment.parameter_name]
                previous_state = segment.select(state_full)
                normalized = current / (current_scale + 1e-30)
                new_state, transitions, establishments = update_hysteresis_state(
                    previous_state, normalized, threshold
                )
                old_dead = old.abs() <= threshold * (old_scale + 1e-30)
                new_dead = current.abs() <= threshold * (current_scale + 1e-30)
                indices = segment.flat_indices.to(state_full.device)
                state_full.reshape(-1).index_copy_(0, indices, new_state)

                last_steps_full = self.last_transition_steps[key][segment.parameter_name]
                last_actions_full = self.last_transition_actions[key][segment.parameter_name]
                selected_steps = segment.select(last_steps_full)
                selected_actions = segment.select(last_actions_full)
                selected_steps[transitions] = int(step)
                selected_actions[transitions] = int(action_index)
                last_steps_full.reshape(-1).index_copy_(0, indices, selected_steps)
                last_actions_full.reshape(-1).index_copy_(0, indices, selected_actions)

                establishment_steps_full = self.establishment_steps[key][segment.parameter_name]
                establishment_actions_full = self.establishment_actions[key][segment.parameter_name]
                selected_establishment_steps = segment.select(establishment_steps_full)
                selected_establishment_actions = segment.select(establishment_actions_full)
                selected_establishment_steps[establishments] = int(step)
                selected_establishment_actions[establishments] = int(action_index)
                establishment_steps_full.reshape(-1).index_copy_(
                    0, indices, selected_establishment_steps
                )
                establishment_actions_full.reshape(-1).index_copy_(
                    0, indices, selected_establishment_actions
                )

                if selected_steps.numel():
                    max_index = int(torch.argmax(selected_steps).item())
                    last_step = int(selected_steps[max_index].item())
                    last_action = int(selected_actions[max_index].item())
                else:
                    last_step = last_action = -1
                threshold_values = {
                    "transition_count": int(torch.count_nonzero(transitions).item()),
                    "dead_count": int(torch.count_nonzero(new_dead).item()),
                    "dead_entries": int(torch.count_nonzero(~old_dead & new_dead).item()),
                    "dead_exits": int(torch.count_nonzero(old_dead & ~new_dead).item()),
                    "established_count": int(torch.count_nonzero(establishments).item()),
                    "last_step": last_step,
                    "last_action": last_action,
                }
                for target in targets:
                    target.add_threshold(key, **threshold_values)

        for name, parameter in self.parameters.items():
            self.previous_values[name] = parameter.detach().to(dtype=torch.float32).clone()

        action_fields = self._action_fields(action, selected)
        record = {
            "schema_version": SIGN_DYNAMICS_SCHEMA_VERSION,
            "event_type": "sign_dynamics_step",
            "campaign_id": self.diagnostic.get("campaign_id"),
            "arm_id": self.diagnostic.get("arm_id"),
            "budget_unit_tokens": self.config.get("controlled_experiment", {}).get(
                "budget_unit_tokens"
            ),
            "budget_multiplier": self.config.get("controlled_experiment", {}).get(
                "budget_multiplier"
            ),
            "seed": self.config.get("run", {}).get("seed"),
            "policy": self.config.get("controlled_experiment", {}).get("policy"),
            "optimizer_state_scope": self.config.get("training", {}).get(
                "optimizer_state_scope", "shared"
            ),
            "precision": self.config.get("training", {}).get(
                "resolved_mixed_precision",
                self.config.get("training", {}).get("mixed_precision"),
            ),
            "diagnostic_contract_hash": self.diagnostic["diagnostic_contract_hash"],
            "support_hash": self.support_manifest["support_hash"],
            "step": int(step),
            **action_fields,
            "bands": {
                name: aggregate.record(
                    self.ordered,
                    forward_active=int(name[1:]) <= action_index + 1,
                )
                for name, aggregate in bands.items()
            },
            "audit_strata": {
                name: aggregate.record(
                    self.ordered,
                    forward_active=(name == "shared" or int(name[5:]) <= action_index + 1),
                )
                for name, aggregate in strata.items()
            },
        }
        payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        self._append_journal(payload)
        self.measured_steps = int(step)
        self._pending = None
        if int(step) in set(self.diagnostic["resolved_snapshot_steps"]):
            self._write_snapshot(int(step))
        self.runtime_seconds += time.perf_counter() - started
        return record

    def _selected_action(self, action: Mapping[str, Any]) -> str:
        values = action.get("granularities")
        if action.get("kind") != "global" or not isinstance(values, list) or len(values) != 1:
            raise SignDynamicsError("sign dynamics requires exactly one global action")
        selected = str(values[0])
        if selected not in self.ordered:
            raise SignDynamicsError(f"unknown sign-dynamics action: {selected!r}")
        return selected

    def _action_fields(self, action: Mapping[str, Any], selected: str) -> dict[str, Any]:
        interval = int(
            action.get(
                "global_sampling_interval_steps",
                action.get(
                    "controller_window_interval_steps",
                    self.config.get("model", {}).get("adaptive_controller", {}).get(
                        "decision_interval_steps", 1
                    ),
                ),
            )
        )
        progress = action.get(
            "global_sampling_window_progress",
            action.get("controller_window_progress"),
        )
        position = int(progress) + 1 if progress is not None else None
        probability = action.get("sampled_probability", action.get("panelgrad_probability"))
        probability_semantics = "recorded_policy_probability"
        if probability is None:
            model = self.config.get("model", {})
            if model.get("granularity_sampling_mode") == "global":
                if model.get("global_sampling_schedule") == "balanced_cycle":
                    probability = 1.0
                    probability_semantics = "deterministic_balanced_action"
                else:
                    probability = 1.0 / len(self.ordered)
                    probability_semantics = "uniform_draw_probability"
            else:
                probability_semantics = "unavailable_for_thompson_draw"
        return {
            "selected_granularity": selected,
            "selected_granularity_index": self.ordered.index(selected),
            "sampled_probability": None if probability is None else float(probability),
            "sampled_probability_semantics": probability_semantics,
            "h_window_interval_steps": interval,
            "h_window_index": action.get(
                "global_sampling_window_index", action.get("controller_window_index")
            ),
            "h_window_position": position,
            "global_sampling_schedule": action.get(
                "global_sampling_schedule",
                self.config.get("model", {}).get("global_sampling_schedule"),
            ),
            "global_sampling_cycle_index": action.get("global_sampling_cycle_index"),
            "global_sampling_cycle_position": action.get("global_sampling_cycle_position"),
        }

    def _append_journal(self, payload: bytes) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.journal_path.open("ab") as journal:
                journal.write(payload)
                journal.flush()
                os.fsync(journal.fileno())
                self.journal_offset = int(journal.tell())
        except OSError as error:
            raise SignDynamicsError("failed to commit sign-dynamics journal record") from error
        self._journal_hasher.update(payload)
        self.journal_hash = self._journal_hasher.hexdigest()

    def _write_snapshot(self, step: int) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.snapshot_dir / f"step-{int(step):08d}.pt"
        payload = {
            "schema_version": SIGN_DYNAMICS_SCHEMA_VERSION,
            "step": int(step),
            "diagnostic_contract_hash": self.diagnostic["diagnostic_contract_hash"],
            "support_hash": self.support_manifest["support_hash"],
            "parameters": {
                name: parameter.detach().to(device="cpu", dtype=torch.float32).clone()
                for name, parameter in self.parameters.items()
            },
        }
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b", dir=self.snapshot_dir, prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as destination:
                temporary = Path(destination.name)
                torch.save(payload, destination)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self.snapshot_identities[str(step)] = {
            "step": int(step),
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "reasons": copy.deepcopy(
                self.diagnostic.get("snapshot_milestone_reasons", {}).get(str(step), [])
            ),
        }

    def state_dict(self, *, copy_tensors: bool = True) -> dict[str, Any]:
        def stored_tensor(tensor: torch.Tensor, *, dtype=None) -> torch.Tensor:
            if not copy_tensors:
                return tensor
            return tensor.detach().to(
                device="cpu", dtype=dtype if dtype is not None else tensor.dtype
            ).clone()

        def cpu_nested(value: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, Any]:
            return {
                key: {
                    name: stored_tensor(tensor)
                    for name, tensor in tensors.items()
                }
                for key, tensors in value.items()
            }

        return {
            "schema_version": SIGN_DYNAMICS_SCHEMA_VERSION,
            "diagnostic_contract_hash": self.diagnostic["diagnostic_contract_hash"],
            "support_hash": self.support_manifest["support_hash"],
            "measured_steps": int(self.measured_steps),
            "previous_values": {
                name: stored_tensor(tensor, dtype=torch.float32)
                for name, tensor in self.previous_values.items()
            },
            "hysteresis_states": cpu_nested(self.hysteresis_states),
            "last_transition_steps": cpu_nested(self.last_transition_steps),
            "last_transition_actions": cpu_nested(self.last_transition_actions),
            "establishment_steps": cpu_nested(self.establishment_steps),
            "establishment_actions": cpu_nested(self.establishment_actions),
            "runtime_seconds": float(self.runtime_seconds),
            "snapshot_identities": copy.deepcopy(self.snapshot_identities),
            "journal_offset": int(self.journal_offset),
            "journal_hash": str(self.journal_hash),
        }

    def validate_checkpoint_state(
        self, state: Mapping[str, Any] | None, *, expected_step: int
    ) -> dict[str, Any]:
        if not isinstance(state, Mapping):
            raise SignDynamicsError("resumable checkpoint is missing sign_dynamics_state")
        staged = copy.deepcopy(dict(state))
        if staged.get("schema_version") != SIGN_DYNAMICS_SCHEMA_VERSION:
            raise SignDynamicsError("sign-dynamics checkpoint schema mismatch")
        if staged.get("diagnostic_contract_hash") != self.diagnostic["diagnostic_contract_hash"]:
            raise SignDynamicsError("sign-dynamics diagnostic contract mismatch")
        if staged.get("support_hash") != self.support_manifest["support_hash"]:
            raise SignDynamicsError("sign-dynamics support identity mismatch")
        if int(staged.get("measured_steps", -1)) != int(expected_step):
            raise SignDynamicsError("sign-dynamics checkpoint has missing historical steps")
        expected_names = set(self.parameters)
        previous = staged.get("previous_values")
        if not isinstance(previous, Mapping) or set(previous) != expected_names:
            raise SignDynamicsError("sign-dynamics previous-value state is incomplete")
        for name, parameter in self.parameters.items():
            value = previous[name]
            if not torch.is_tensor(value) or value.dtype != torch.float32 or tuple(value.shape) != tuple(parameter.shape):
                raise SignDynamicsError(f"invalid sign-dynamics previous value for {name}")
            if not bool(torch.all(torch.isfinite(value))):
                raise SignDynamicsError(f"non-finite sign-dynamics previous value for {name}")
        threshold_keys = {_threshold_key(value) for value in self.diagnostic["hysteresis_thresholds"]}
        for field, dtype in (
            ("hysteresis_states", torch.int8),
            ("last_transition_steps", torch.int64),
            ("last_transition_actions", torch.int16),
            ("establishment_steps", torch.int64),
            ("establishment_actions", torch.int16),
        ):
            nested = staged.get(field)
            if not isinstance(nested, Mapping) or set(nested) != threshold_keys:
                raise SignDynamicsError(f"sign-dynamics {field} thresholds are incomplete")
            for key in threshold_keys:
                tensors = nested[key]
                if not isinstance(tensors, Mapping) or set(tensors) != expected_names:
                    raise SignDynamicsError(f"sign-dynamics {field} parameters are incomplete")
                for name, parameter in self.parameters.items():
                    tensor = tensors[name]
                    if not torch.is_tensor(tensor) or tensor.dtype != dtype or tuple(tensor.shape) != tuple(parameter.shape):
                        raise SignDynamicsError(f"invalid sign-dynamics {field} for {name}")
        self._reconcile_journal(staged, expected_step=int(expected_step))
        self._validate_snapshots(staged, expected_step=int(expected_step))
        return staged

    def restore_validated_state(self, state: Mapping[str, Any]) -> None:
        device_by_name = {name: parameter.device for name, parameter in self.parameters.items()}

        def restore_nested(value: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, Any]:
            return {
                key: {
                    name: tensor.to(device=device_by_name[name]).clone()
                    for name, tensor in tensors.items()
                }
                for key, tensors in value.items()
            }

        self.previous_values = {
            name: tensor.to(device=device_by_name[name], dtype=torch.float32).clone()
            for name, tensor in state["previous_values"].items()
        }
        for name, parameter in self.parameters.items():
            if not torch.equal(
                self.previous_values[name], parameter.detach().to(dtype=torch.float32)
            ):
                raise SignDynamicsError(
                    "restored model does not match sign-dynamics previous committed values"
                )
        self.hysteresis_states = restore_nested(state["hysteresis_states"])
        self.last_transition_steps = restore_nested(state["last_transition_steps"])
        self.last_transition_actions = restore_nested(state["last_transition_actions"])
        self.establishment_steps = restore_nested(state["establishment_steps"])
        self.establishment_actions = restore_nested(state["establishment_actions"])
        self.measured_steps = int(state["measured_steps"])
        self.runtime_seconds = float(state.get("runtime_seconds", 0.0))
        self.snapshot_identities = copy.deepcopy(dict(state["snapshot_identities"]))
        self.journal_offset = int(state["journal_offset"])
        self.journal_hash = str(state["journal_hash"])
        if self._reconciled_journal_hasher is None:
            raise SignDynamicsError("sign-dynamics journal was not reconciled before restore")
        self._journal_hasher = self._reconciled_journal_hasher.copy()
        self._reconciled_journal_hasher = None

    def _reconcile_journal(self, state: Mapping[str, Any], *, expected_step: int) -> None:
        offset = int(state.get("journal_offset", -1))
        expected_hash = state.get("journal_hash")
        if offset < 0 or not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise SignDynamicsError("invalid committed sign-dynamics journal identity")
        if not self.journal_path.exists():
            if offset != 0:
                raise SignDynamicsError("sign-dynamics journal is missing historical steps")
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            self.journal_path.touch()
        size = self.journal_path.stat().st_size
        prefix_hasher = (
            _sha256_hasher(self.journal_path, limit=offset)
            if size >= offset
            else None
        )
        if (
            size < offset
            or prefix_hasher is None
            or prefix_hasher.hexdigest() != expected_hash
        ):
            raise SignDynamicsError("sign-dynamics journal prefix does not match checkpoint")
        if size > offset:
            with self.journal_path.open("r+b") as journal:
                journal.truncate(offset)
                journal.flush()
                os.fsync(journal.fileno())
        records = []
        with self.journal_path.open("r", encoding="utf-8") as journal:
            for line_number, line in enumerate(journal, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SignDynamicsError("invalid committed sign-dynamics journal") from error
                if int(record.get("step", -1)) != line_number:
                    raise SignDynamicsError("sign-dynamics journal has missing historical steps")
                if record.get("support_hash") != self.support_manifest["support_hash"]:
                    raise SignDynamicsError("sign-dynamics journal support mismatch")
                records.append(record)
        if len(records) != expected_step:
            raise SignDynamicsError("sign-dynamics journal has missing historical steps")
        self._reconciled_journal_hasher = prefix_hasher

    def _validate_snapshots(self, state: Mapping[str, Any], *, expected_step: int) -> None:
        identities = state.get("snapshot_identities")
        if not isinstance(identities, Mapping):
            raise SignDynamicsError("sign-dynamics snapshot identities are missing")
        required_steps = {
            int(step)
            for step in self.diagnostic["resolved_snapshot_steps"]
            if int(step) <= expected_step
        }
        if set(int(key) for key in identities) != required_steps:
            raise SignDynamicsError("sign-dynamics milestone snapshot coverage is incomplete")
        for step in sorted(required_steps):
            identity = identities[str(step)]
            path = Path(str(identity.get("path", "")))
            expected_path = self.snapshot_dir / f"step-{step:08d}.pt"
            if (
                path.resolve() != expected_path.resolve()
                or not path.is_file()
                or int(identity.get("step", -1)) != step
                or int(identity.get("bytes", -1)) != path.stat().st_size
                or _sha256(path) != identity.get("sha256")
            ):
                raise SignDynamicsError(f"sign-dynamics milestone snapshot is missing or changed: step {step}")

            try:
                snapshot = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as error:
                raise SignDynamicsError(
                    f"sign-dynamics milestone snapshot cannot be loaded: step {step}"
                ) from error
            values = snapshot.get("parameters") if isinstance(snapshot, Mapping) else None
            if (
                snapshot.get("schema_version") != SIGN_DYNAMICS_SCHEMA_VERSION
                or int(snapshot.get("step", -1)) != step
                or snapshot.get("diagnostic_contract_hash")
                != self.diagnostic["diagnostic_contract_hash"]
                or snapshot.get("support_hash")
                != self.support_manifest["support_hash"]
                or not isinstance(values, Mapping)
                or set(values) != set(self.parameters)
            ):
                raise SignDynamicsError(
                    f"sign-dynamics milestone snapshot contract is invalid: step {step}"
                )
            for name, parameter in self.parameters.items():
                value = values[name]
                if (
                    not torch.is_tensor(value)
                    or value.dtype != torch.float32
                    or tuple(value.shape) != tuple(parameter.shape)
                    or not bool(torch.all(torch.isfinite(value)))
                ):
                    raise SignDynamicsError(
                        f"sign-dynamics milestone snapshot parameter is invalid: step {step}, {name}"
                    )

    def validate_completed(self, *, expected_step: int) -> None:
        if self._pending is not None or self.measured_steps != int(expected_step):
            raise SignDynamicsError("sign-dynamics measured-step coverage is incomplete")
        state = self.state_dict(copy_tensors=False)
        self._reconcile_journal(state, expected_step=int(expected_step))
        self._validate_snapshots(state, expected_step=int(expected_step))

    def summary_fields(self, *, expected_steps: int | None = None) -> dict[str, Any]:
        paths = [self.support_path, self.journal_path]
        paths.extend(Path(item["path"]) for item in self.snapshot_identities.values())
        storage_bytes = sum(path.stat().st_size for path in paths if path.is_file())
        expected = int(
            self.measured_steps if expected_steps is None else expected_steps
        )
        return {
            "sign_dynamics_enabled": True,
            "sign_dynamics_campaign_id": self.diagnostic.get("campaign_id"),
            "sign_dynamics_arm_id": self.diagnostic.get("arm_id"),
            "sign_dynamics_budget_unit_tokens": self.config.get(
                "controlled_experiment", {}
            ).get("budget_unit_tokens"),
            "sign_dynamics_budget_multiplier": self.config.get(
                "controlled_experiment", {}
            ).get("budget_multiplier"),
            "sign_dynamics_policy": self.config.get("controlled_experiment", {}).get(
                "policy"
            ),
            "sign_dynamics_contract_hash": self.diagnostic["diagnostic_contract_hash"],
            "sign_dynamics_support_path": str(self.support_path),
            "sign_dynamics_support_hash": self.support_manifest["support_hash"],
            "sign_dynamics_journal_path": str(self.journal_path),
            "sign_dynamics_journal_hash": self.journal_hash,
            "sign_dynamics_snapshot_dir": str(self.snapshot_dir),
            "sign_dynamics_snapshot_identities": copy.deepcopy(self.snapshot_identities),
            "sign_dynamics_trainable_coordinate_count": self.support_manifest["trainable_parameter_count"],
            "sign_dynamics_measured_steps": self.measured_steps,
            "sign_dynamics_expected_steps": expected,
            "sign_dynamics_step_coverage_complete": self.measured_steps == expected,
            "sign_dynamics_runtime_seconds": self.runtime_seconds,
            "sign_dynamics_storage_bytes": storage_bytes,
        }

    @staticmethod
    def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as destination:
                temporary = Path(destination.name)
                json.dump(value, destination, indent=2, sort_keys=True)
                destination.write("\n")
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def checkpoint_state_to_cpu(state: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Materialize a live diagnostic state only when a checkpoint is written."""

    if state is None:
        return None

    def convert(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().to(device="cpu").clone()
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, tuple):
            return tuple(convert(item) for item in value)
        return copy.deepcopy(value)

    return convert(state)


def disabled_summary_fields() -> dict[str, Any]:
    return {
        "sign_dynamics_enabled": False,
        "sign_dynamics_campaign_id": None,
        "sign_dynamics_arm_id": None,
        "sign_dynamics_budget_unit_tokens": None,
        "sign_dynamics_budget_multiplier": None,
        "sign_dynamics_policy": None,
        "sign_dynamics_contract_hash": None,
        "sign_dynamics_support_path": None,
        "sign_dynamics_support_hash": None,
        "sign_dynamics_journal_path": None,
        "sign_dynamics_journal_hash": None,
        "sign_dynamics_snapshot_dir": None,
        "sign_dynamics_snapshot_identities": {},
        "sign_dynamics_trainable_coordinate_count": 0,
        "sign_dynamics_measured_steps": 0,
        "sign_dynamics_expected_steps": 0,
        "sign_dynamics_step_coverage_complete": False,
        "sign_dynamics_runtime_seconds": 0.0,
        "sign_dynamics_storage_bytes": 0,
    }
