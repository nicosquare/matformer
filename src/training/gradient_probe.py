"""Neutral fixed-panel raw-gradient measurement over controlled FFN support."""

from __future__ import annotations

import copy
import math
import time
from contextlib import ExitStack
from typing import Any, Mapping, Sequence

import torch

from src.evaluation.validation import (
    configure_model_granularity,
    count_valid_prediction_targets,
    move_batch_to_device,
)
from src.models.ffn import CatLlamaMLP, ControlledFFNSupportEntry, ModifiedLlamaMLP
from src.training.distributed import autocast_context
from src.utils.reproducibility import capture_rng_state, restore_rng_state, stable_hash


CONTROLLED_SUPPORT_SCHEMA_VERSION = 1


class ControlledGradientProbeError(ValueError):
    """Raised when a fixed-panel controlled-gradient measurement is invalid."""


def _support_entry_key(
    entry: ControlledFFNSupportEntry,
) -> tuple[int, tuple[tuple[int | None, ...], ...]]:
    return (
        id(entry.parameter),
        tuple((item.start, item.stop, item.step) for item in entry.selection),
    )


def controlled_mlps(model) -> list[tuple[str, ModifiedLlamaMLP | CatLlamaMLP]]:
    target = model.module if hasattr(model, "module") else model
    return [
        (name, module)
        for name, module in target.named_modules()
        if isinstance(module, (ModifiedLlamaMLP, CatLlamaMLP))
    ]


def resolve_controlled_ffn_support(
    model,
    granularities: Sequence[str],
) -> dict[str, Any]:
    """Resolve stable controlled-FFN descriptors and counts before sharding."""

    ordered = [str(label) for label in granularities]
    if not ordered or len(set(ordered)) != len(ordered):
        raise ControlledGradientProbeError(
            "controlled support granularities must be non-empty and unique"
        )
    mlps = controlled_mlps(model)
    if not mlps:
        raise ControlledGradientProbeError(
            "controlled gradient probe found no MatFormer FFN layers"
        )
    variants = {
        "concat" if isinstance(module, CatLlamaMLP) else "slicing" for _, module in mlps
    }
    if len(variants) != 1:
        raise ControlledGradientProbeError(
            "controlled gradient probe requires one FFN layout variant"
        )

    supports: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for granularity in ordered:
        seen: set[tuple[int, tuple[tuple[int | None, ...], ...]]] = set()
        descriptors: list[dict[str, Any]] = []
        count = 0
        for module_name, mlp in mlps:
            try:
                entries = mlp.controlled_ffn_support(granularity)
            except ValueError as error:
                raise ControlledGradientProbeError(str(error)) from error
            for entry in entries:
                key = _support_entry_key(entry)
                if key in seen:
                    continue
                seen.add(key)
                descriptor = entry.descriptor()
                descriptor["module_name"] = module_name
                descriptors.append(descriptor)
                count += entry.scalar_count
        if count <= 0:
            raise ControlledGradientProbeError(
                f"Granularity {granularity!r} has zero controlled trainable FFN scalars"
            )
        counts[granularity] = count
        supports[granularity] = descriptors

    identity = {
        "support_schema_version": CONTROLLED_SUPPORT_SCHEMA_VERSION,
        "variant": next(iter(variants)),
        "layer_count": len(mlps),
        "ordered_granularities": ordered,
        "controlled_support_counts": counts,
        "supports": supports,
        "excluded_parameter_families": [
            "shared_down_bias",
            "embeddings",
            "attention",
            "normalization",
            "language_model_head",
            "other_granularity_independent_parameters",
        ],
    }
    identity["controlled_support_hash"] = stable_hash(identity)
    return identity


def validate_support_identity(
    support_identity: Mapping[str, Any], ordered_granularities: Sequence[str]
) -> dict[str, Any]:
    support = copy.deepcopy(dict(support_identity))
    ordered = [str(label) for label in ordered_granularities]
    if support.get("ordered_granularities") != ordered:
        raise ControlledGradientProbeError(
            "controlled support granularity order mismatch"
        )
    saved_hash = support.get("controlled_support_hash")
    payload = {
        key: value for key, value in support.items() if key != "controlled_support_hash"
    }
    if not isinstance(saved_hash, str) or stable_hash(payload) != saved_hash:
        raise ControlledGradientProbeError("controlled support hash mismatch")
    return support


def _capture_runtime_granularity_state(model) -> dict[str, Any]:
    target = model.module if hasattr(model, "module") else model
    layer_granularities = getattr(target, "current_layer_granularities", None)
    return {
        "current_granularity": getattr(target, "current_granularity", None),
        "current_layer_granularities": (
            None if layer_granularities is None else list(layer_granularities)
        ),
        "current_granularity_pattern": getattr(
            target, "current_granularity_pattern", None
        ),
        "current_sampling_mode": getattr(target, "current_sampling_mode", None),
    }


def _restore_runtime_granularity_state(model, state: Mapping[str, Any]) -> None:
    target = model.module if hasattr(model, "module") else model
    layer_granularities = state.get("current_layer_granularities")
    if layer_granularities:
        if len(set(layer_granularities)) == 1:
            configure = getattr(target, "configure_subnetwork", None)
            if configure is not None:
                configure(layer_granularities[0])
        else:
            configure_layers = getattr(target, "configure_layer_granularities", None)
            if configure_layers is not None:
                configure_layers(layer_granularities)
    elif state.get("current_granularity") is not None:
        configure = getattr(target, "configure_subnetwork", None)
        if configure is not None:
            configure(state["current_granularity"])
    for field_name, value in state.items():
        if hasattr(target, field_name):
            setattr(target, field_name, value)


def materialize_fixed_probe(dataloader) -> tuple[list[dict[str, Any]], int, int]:
    batches = [batch for batch in dataloader]
    target_count = sum(count_valid_prediction_targets(batch) for batch in batches)
    packed_sequence_count = sum(int(batch["input_ids"].shape[0]) for batch in batches)
    if not batches or target_count <= 0:
        raise ControlledGradientProbeError(
            "fixed gradient probe must contain valid causal targets"
        )
    return batches, target_count, packed_sequence_count


def _source_example_count(dataloader) -> int | None:
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        return None
    try:
        count = int(len(dataset))
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def _extract_layer_gradients(
    model,
    granularity: str,
    *,
    device: torch.device | str,
    retain_gradients: bool,
) -> tuple[torch.Tensor, int, dict[str, dict[str, Any]] | None]:
    """Extract exact support, summoning one FSDP decoder layer at a time."""

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:  # pragma: no cover
        FSDP = None

    squared_norm = torch.zeros((), dtype=torch.float64, device=device)
    measured_count = 0
    seen: set[tuple[int, tuple[tuple[int | None, ...], ...]]] = set()
    retained: dict[str, dict[str, Any]] | None = {} if retain_gradients else None
    layer_index = 0

    def accumulate(mlps):
        nonlocal squared_norm, measured_count, layer_index
        for module_name, mlp in mlps:
            layer_key = f"layer_{layer_index:04d}"
            layer_index += 1
            layer_payload: dict[str, Any] = {}
            for entry in mlp.controlled_ffn_support(granularity):
                key = _support_entry_key(entry)
                if key in seen:
                    continue
                seen.add(key)
                measured_count += entry.scalar_count
                if entry.parameter.grad is None:
                    gradient = torch.zeros_like(entry.selected_tensor())
                else:
                    gradient = entry.selected_tensor(entry.parameter.grad)
                if not bool(torch.all(torch.isfinite(gradient))):
                    raise ControlledGradientProbeError(
                        f"controlled gradient is non-finite for {granularity!r}"
                    )
                squared_norm += torch.sum(gradient.double().square())
                if retained is not None:
                    layer_payload[entry.parameter_name] = {
                        "parameter_family": entry.parameter_family,
                        "selection": [
                            {"start": item.start, "stop": item.stop, "step": item.step}
                            for item in entry.selection
                        ],
                        "tensor": gradient.detach()
                        .to(device="cpu", dtype=torch.float32)
                        .clone(),
                    }
            if retained is not None:
                retained[layer_key] = {
                    "module_name": module_name,
                    "entries": layer_payload,
                }

    if FSDP is not None and isinstance(model, FSDP):
        wrappers = [
            module
            for module in model.modules()
            if isinstance(module, FSDP) and module is not model
        ]
        if not wrappers:
            raise ControlledGradientProbeError(
                "distributed controlled-gradient probing requires per-layer FSDP wrapping"
            )
        for wrapper in wrappers:
            with FSDP.summon_full_params(
                wrapper, recurse=False, writeback=False, with_grads=True
            ):
                accumulate(controlled_mlps(wrapper))
    else:
        accumulate(controlled_mlps(model))
    return squared_norm, measured_count, retained


def measure_controlled_gradients(
    model,
    dataloader,
    granularities: Sequence[str],
    *,
    device: torch.device | str,
    config: Mapping[str, Any] | None = None,
    support_identity: Mapping[str, Any] | None = None,
    retain_gradients: bool = False,
) -> dict[str, Any]:
    """Measure complete target-weighted fixed-probe gradients per granularity."""

    ordered = [str(label) for label in granularities]
    support = validate_support_identity(
        resolve_controlled_ffn_support(model, ordered)
        if support_identity is None
        else support_identity,
        ordered,
    )
    mlps = controlled_mlps(model)
    if not mlps:
        raise ControlledGradientProbeError(
            "controlled gradient probe found no MatFormer FFN layers"
        )

    was_training = bool(model.training)
    runtime_state = _capture_runtime_granularity_state(model)
    rng_state = capture_rng_state()
    started_at = time.perf_counter()
    measurements: list[dict[str, Any]] = []
    retained_by_granularity: dict[str, Any] = {}
    batches: list[dict[str, Any]] = []
    target_count = 0
    packed_sequence_count = 0
    try:
        batches, target_count, packed_sequence_count = materialize_fixed_probe(
            dataloader
        )
        model.zero_grad(set_to_none=True)
        model.eval()
        with ExitStack() as stack:
            for _, mlp in mlps:
                stack.enter_context(mlp.suspend_gradient_membership_correction())
            for granularity in ordered:
                configure_model_granularity(model, granularity)
                model.zero_grad(set_to_none=True)
                loss_numerator = 0.0
                for raw_batch in batches:
                    batch = move_batch_to_device(raw_batch, device)
                    valid_targets = count_valid_prediction_targets(batch)
                    if valid_targets <= 0:
                        continue
                    with autocast_context(dict(config or {}), torch.device(device)):
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch.get("attention_mask"),
                            labels=batch["labels"],
                        )
                    loss = outputs.loss
                    if not bool(torch.isfinite(loss.detach())):
                        raise ControlledGradientProbeError(
                            f"fixed-probe loss is non-finite for {granularity!r}"
                        )
                    loss_numerator += float(loss.detach().double()) * valid_targets
                    (loss * (valid_targets / target_count)).backward()

                squared_norm, measured_count, retained = _extract_layer_gradients(
                    model,
                    granularity,
                    device=device,
                    retain_gradients=retain_gradients,
                )
                expected_count = int(support["controlled_support_counts"][granularity])
                if measured_count != expected_count:
                    raise ControlledGradientProbeError(
                        f"controlled support count mismatch for {granularity!r}: "
                        f"expected {expected_count}, measured {measured_count}"
                    )
                squared_value = float(squared_norm.item())
                if not math.isfinite(squared_value) or squared_value < 0.0:
                    raise ControlledGradientProbeError(
                        f"gradient squared norm is invalid for {granularity!r}"
                    )
                gradient_norm = math.sqrt(squared_value)
                measurements.append(
                    {
                        "granularity": granularity,
                        "controlled_parameter_count": expected_count,
                        "aggregate_loss": loss_numerator / target_count,
                        "gradient_squared_norm": squared_value,
                        "gradient_norm": gradient_norm,
                        "gradient_rms_score": gradient_norm / math.sqrt(expected_count),
                    }
                )
                if retain_gradients:
                    retained_by_granularity[granularity] = retained
                model.zero_grad(set_to_none=True)
    finally:
        model.zero_grad(set_to_none=True)
        _restore_runtime_granularity_state(model, runtime_state)
        model.train(was_training)
        restore_rng_state(rng_state)

    granularity_count = len(ordered)
    cost: dict[str, Any] = {
        "ordered_granularities": ordered,
        "measurements": measurements,
        "controller_example_count": packed_sequence_count,
        "controller_packed_sequence_count": packed_sequence_count,
        "controller_batch_count": len(batches),
        "controller_granularity_count": granularity_count,
        "controller_target_count": target_count,
        "controller_packed_sequence_evaluation_count": (
            packed_sequence_count * granularity_count
        ),
        "controller_target_evaluation_count": target_count * granularity_count,
        "backward_evaluation_count": len(batches) * granularity_count,
        "duration_seconds": time.perf_counter() - started_at,
        "controlled_support_hash": support["controlled_support_hash"],
    }
    source_count = _source_example_count(dataloader)
    if source_count is not None:
        cost["controller_source_example_count"] = source_count
    if retain_gradients:
        cost["_gradient_snapshots"] = retained_by_granularity
    return cost


__all__ = [
    "CONTROLLED_SUPPORT_SCHEMA_VERSION",
    "ControlledGradientProbeError",
    "measure_controlled_gradients",
    "resolve_controlled_ffn_support",
    "validate_support_identity",
]
