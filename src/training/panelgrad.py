"""PanelGrad full-panel gradient measurement and categorical policy state."""

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


PANELGRAD_STATE_SCHEMA_VERSION = 2
CONTROLLED_SUPPORT_SCHEMA_VERSION = 1
PANELGRAD_METHOD_FAMILY = "panelgrad_gradient_rms"
PANELGRAD_METHOD_VERSION = 1


class PanelGradError(ValueError):
    """Raised when PanelGrad cannot form or advance a valid policy state."""


def uses_panelgrad(config: Mapping[str, Any]) -> bool:
    model = config.get("model", {})
    return (
        model.get("granularity_sampling_mode") == "adaptive_global"
        and model.get("adaptive_sampler_strategy") == "panelgrad"
    )


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise PanelGradError(f"{field_name} must be a finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PanelGradError(f"{field_name} must be a finite scalar") from error
    if not math.isfinite(result):
        raise PanelGradError(f"{field_name} must be finite")
    return result


def resolve_epsilon_schedule(
    *,
    epsilon: Any | None,
    epsilon_schedule: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return one explicit fixed or linear schedule for controller state."""

    if epsilon is not None and epsilon_schedule is not None:
        raise PanelGradError("provide either epsilon or epsilon_schedule, not both")
    if epsilon_schedule is None:
        value = _finite_float(0.1 if epsilon is None else epsilon, "epsilon")
        if not 0.0 <= value <= 1.0:
            raise PanelGradError("epsilon must be between zero and one")
        return {
            "type": "fixed",
            "start": value,
            "end": value,
            "duration_steps": None,
        }
    if not isinstance(epsilon_schedule, Mapping):
        raise PanelGradError("epsilon schedule must be a mapping")
    schedule = copy.deepcopy(dict(epsilon_schedule))
    if set(schedule) != {"type", "start", "end", "duration_steps"}:
        raise PanelGradError(
            "epsilon schedule requires exactly type, start, end, and duration_steps"
        )
    if schedule["type"] != "linear":
        raise PanelGradError("epsilon schedule type must be 'linear'")
    for endpoint in ("start", "end"):
        value = _finite_float(schedule[endpoint], f"epsilon schedule {endpoint}")
        if not 0.0 <= value <= 1.0:
            raise PanelGradError(
                f"epsilon schedule {endpoint} must be between zero and one"
            )
        schedule[endpoint] = value
    duration_steps = schedule["duration_steps"]
    if (
        isinstance(duration_steps, bool)
        or not isinstance(duration_steps, int)
        or duration_steps <= 0
    ):
        raise PanelGradError("epsilon schedule duration_steps must be a positive integer")
    schedule["duration_steps"] = int(duration_steps)
    return schedule


def epsilon_at_schedule_step(
    epsilon_schedule: Mapping[str, Any],
    schedule_step: Any,
) -> float:
    """Evaluate a resolved epsilon schedule at a committed PanelGrad step."""

    if (
        isinstance(schedule_step, bool)
        or not isinstance(schedule_step, int)
        or schedule_step < 0
    ):
        raise PanelGradError("epsilon schedule step must be a nonnegative integer")
    schedule = resolve_epsilon_schedule(
        epsilon=(
            epsilon_schedule.get("start")
            if epsilon_schedule.get("type") == "fixed"
            else None
        ),
        epsilon_schedule=(
            None if epsilon_schedule.get("type") == "fixed" else epsilon_schedule
        ),
    )
    if schedule["type"] == "fixed":
        return float(schedule["start"])
    fraction = min(schedule_step / int(schedule["duration_steps"]), 1.0)
    return float(
        schedule["start"] + (schedule["end"] - schedule["start"]) * fraction
    )


def gradient_rms_from_squared_norm(
    gradient_squared_norm: Any,
    controlled_parameter_count: Any,
) -> tuple[float, float]:
    """Return ``(||d||_2, ||d||_2/sqrt(N))`` using float64 arithmetic."""

    squared_norm = _finite_float(gradient_squared_norm, "gradient squared norm")
    if squared_norm < 0.0:
        raise PanelGradError("gradient squared norm must be nonnegative")
    if (
        isinstance(controlled_parameter_count, bool)
        or not isinstance(controlled_parameter_count, int)
        or controlled_parameter_count <= 0
    ):
        raise PanelGradError("controlled parameter count must be a positive integer")
    gradient_norm = math.sqrt(squared_norm)
    score = gradient_norm / math.sqrt(controlled_parameter_count)
    if not math.isfinite(score):
        raise PanelGradError("gradient RMS score must be finite")
    return gradient_norm, score


def build_probability_snapshot(
    scores: Sequence[Any],
    *,
    eta: Any,
    temperature: Any,
    epsilon: Any,
) -> dict[str, Any]:
    """Map contemporaneous nonnegative scores to validated ``q`` and ``p``."""

    if not scores:
        raise PanelGradError("PanelGrad requires at least one score")
    score_values = torch.tensor(
        [_finite_float(value, "PanelGrad score") for value in scores],
        dtype=torch.float64,
    )
    if bool(torch.any(score_values < 0.0)):
        raise PanelGradError("PanelGrad scores must be nonnegative")
    eta_value = _finite_float(eta, "eta")
    temperature_value = _finite_float(temperature, "temperature")
    epsilon_value = _finite_float(epsilon, "epsilon")
    if eta_value <= 0.0:
        raise PanelGradError("eta must be positive")
    if temperature_value <= 0.0:
        raise PanelGradError("temperature must be positive")
    if not 0.0 <= epsilon_value <= 1.0:
        raise PanelGradError("epsilon must be between zero and one")

    log_weights = torch.log(score_values + eta_value) / temperature_value
    q = torch.softmax(log_weights, dim=0)
    arm_count = len(scores)
    p = (1.0 - epsilon_value) * q + epsilon_value / arm_count
    if not bool(torch.all(torch.isfinite(q))) or not bool(
        torch.all(torch.isfinite(p))
    ):
        raise PanelGradError("PanelGrad probabilities must be finite")
    if bool(torch.any(q < 0.0)) or bool(torch.any(p < 0.0)):
        raise PanelGradError("PanelGrad probabilities must be nonnegative")
    if not math.isclose(float(q.sum()), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise PanelGradError("PanelGrad q does not sum to one")
    if not math.isclose(float(p.sum()), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise PanelGradError("PanelGrad p does not sum to one")
    entropy = -float(torch.sum(p * torch.log(p)).item())
    return {
        "scores": score_values.tolist(),
        "q": q.tolist(),
        "p": p.tolist(),
        "entropy": entropy,
        "min_probability": float(torch.min(p).item()),
        "max_probability": float(torch.max(p).item()),
    }


def _support_entry_key(
    entry: ControlledFFNSupportEntry,
) -> tuple[int, tuple[tuple[int | None, ...], ...]]:
    return (
        id(entry.parameter),
        tuple((item.start, item.stop, item.step) for item in entry.selection),
    )


def _controlled_mlps(model) -> list[tuple[str, ModifiedLlamaMLP | CatLlamaMLP]]:
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
        raise PanelGradError("controlled support granularities must be non-empty and unique")
    mlps = _controlled_mlps(model)
    if not mlps:
        raise PanelGradError("PanelGrad model has no controlled MatFormer FFN layers")

    variants = {
        "concat" if isinstance(module, CatLlamaMLP) else "slicing"
        for _, module in mlps
    }
    if len(variants) != 1:
        raise PanelGradError("PanelGrad requires one consistent FFN layout variant")
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
                raise PanelGradError(str(error)) from error
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
            raise PanelGradError(
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


def _materialize_controller_panel(dataloader) -> tuple[list[dict[str, Any]], int, int]:
    batches = [batch for batch in dataloader]
    target_count = sum(count_valid_prediction_targets(batch) for batch in batches)
    example_count = sum(int(batch["input_ids"].shape[0]) for batch in batches)
    if not batches or target_count <= 0:
        raise PanelGradError("controller panel must contain valid causal targets")
    return batches, target_count, example_count


def _controlled_gradient_squared_norm(
    model,
    granularity: str,
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, int]:
    """Extract exact support gradients, summoning one FSDP layer at a time."""

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:  # pragma: no cover - unavailable in minimal torch builds
        FSDP = None

    squared_norm = torch.zeros((), dtype=torch.float64, device=device)
    measured_count = 0
    seen: set[tuple[int, tuple[tuple[int | None, ...], ...]]] = set()

    def accumulate(mlps):
        nonlocal squared_norm, measured_count
        for _, mlp in mlps:
            for entry in mlp.controlled_ffn_support(granularity):
                key = _support_entry_key(entry)
                if key in seen:
                    continue
                seen.add(key)
                measured_count += entry.scalar_count
                if entry.parameter.grad is not None:
                    gradient = entry.selected_tensor(entry.parameter.grad)
                    squared_norm += torch.sum(gradient.double().square())

    if FSDP is not None and isinstance(model, FSDP):
        layer_wrappers = [
            module
            for module in model.modules()
            if isinstance(module, FSDP) and module is not model
        ]
        if not layer_wrappers:
            raise PanelGradError(
                "distributed PanelGrad requires per-decoder-layer FSDP wrapping"
            )
        for wrapper in layer_wrappers:
            with FSDP.summon_full_params(
                wrapper,
                recurse=False,
                writeback=False,
                with_grads=True,
            ):
                accumulate(_controlled_mlps(wrapper))
    else:
        accumulate(_controlled_mlps(model))
    return squared_norm, measured_count


def measure_panelgrad_gradients(
    model,
    dataloader,
    granularities: Sequence[str],
    *,
    device: torch.device | str,
    config: Mapping[str, Any] | None = None,
    support_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure one aggregate raw controller gradient for each global granularity."""

    ordered = [str(label) for label in granularities]
    support = (
        resolve_controlled_ffn_support(model, ordered)
        if support_identity is None
        else copy.deepcopy(dict(support_identity))
    )
    if support.get("ordered_granularities") != ordered:
        raise PanelGradError("controlled support granularity order mismatch")
    saved_support_hash = support.get("controlled_support_hash")
    support_payload = {
        key: value
        for key, value in support.items()
        if key != "controlled_support_hash"
    }
    if (
        not isinstance(saved_support_hash, str)
        or stable_hash(support_payload) != saved_support_hash
    ):
        raise PanelGradError("controlled support hash mismatch")
    mlps = _controlled_mlps(model)
    if not mlps:
        raise PanelGradError("PanelGrad model has no controlled MatFormer FFN layers")

    was_training = bool(model.training)
    runtime_state = _capture_runtime_granularity_state(model)
    rng_state = capture_rng_state()
    started_at = time.perf_counter()
    measurements: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    target_count = 0
    example_count = 0
    try:
        batches, target_count, example_count = _materialize_controller_panel(dataloader)
        model.zero_grad(set_to_none=True)
        model.eval()
        with ExitStack() as stack:
            for _, mlp in mlps:
                stack.enter_context(mlp.suspend_gradient_membership_correction())
            for granularity in ordered:
                configure_model_granularity(model, granularity)
                model.zero_grad(set_to_none=True)
                loss_numerator = 0.0
                for batch in batches:
                    batch = move_batch_to_device(batch, device)
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
                        raise PanelGradError(
                            f"controller loss is non-finite for {granularity!r}"
                        )
                    loss_numerator += float(loss.detach().double()) * valid_targets
                    (loss * (valid_targets / target_count)).backward()

                squared_norm, measured_count = _controlled_gradient_squared_norm(
                    model,
                    granularity,
                    device=device,
                )
                expected_count = int(
                    support["controlled_support_counts"][granularity]
                )
                if measured_count != expected_count:
                    raise PanelGradError(
                        f"controlled support count mismatch for {granularity!r}: "
                        f"expected {expected_count}, measured {measured_count}"
                    )
                gradient_norm, score = gradient_rms_from_squared_norm(
                    float(squared_norm.item()), expected_count
                )
                measurements.append(
                    {
                        "granularity": granularity,
                        "controlled_parameter_count": expected_count,
                        "aggregate_loss": loss_numerator / target_count,
                        "gradient_squared_norm": float(squared_norm.item()),
                        "gradient_norm": gradient_norm,
                        "gradient_rms_score": score,
                    }
                )
                model.zero_grad(set_to_none=True)
    finally:
        model.zero_grad(set_to_none=True)
        _restore_runtime_granularity_state(model, runtime_state)
        model.train(was_training)
        restore_rng_state(rng_state)

    return {
        "ordered_granularities": ordered,
        "measurements": measurements,
        "controller_example_count": example_count,
        "controller_target_count": target_count,
        "backward_evaluation_count": len(batches) * len(ordered),
        "duration_seconds": time.perf_counter() - started_at,
        "controlled_support_hash": support["controlled_support_hash"],
    }


class PanelGradController:
    """Explicit refresh/distribution/draw state for PanelGrad."""

    def __init__(
        self,
        *,
        ordered_granularities: Sequence[str],
        refresh_interval_steps: int,
        eta: float,
        temperature: float,
        epsilon: float | None,
        sampling_seed: int,
        support_identity: Mapping[str, Any],
        manifest_hashes: Mapping[str, str | None] | None = None,
        epsilon_schedule: Mapping[str, Any] | None = None,
    ) -> None:
        ordered = [str(label) for label in ordered_granularities]
        if not ordered or len(set(ordered)) != len(ordered):
            raise PanelGradError("ordered granularities must be non-empty and unique")
        if isinstance(refresh_interval_steps, bool) or refresh_interval_steps <= 0:
            raise PanelGradError("refresh interval steps must be a positive integer")
        resolved_epsilon_schedule = resolve_epsilon_schedule(
            epsilon=epsilon,
            epsilon_schedule=epsilon_schedule,
        )
        build_probability_snapshot(
            [0.0] * len(ordered),
            eta=eta,
            temperature=temperature,
            epsilon=epsilon_at_schedule_step(resolved_epsilon_schedule, 0),
        )
        support = copy.deepcopy(dict(support_identity))
        if support.get("ordered_granularities") != ordered:
            raise PanelGradError("controlled support granularity order mismatch")
        counts = support.get("controlled_support_counts")
        if not isinstance(counts, Mapping) or any(
            isinstance(counts.get(label), bool)
            or not isinstance(counts.get(label), int)
            or int(counts[label]) <= 0
            for label in ordered
        ):
            raise PanelGradError("every granularity requires positive controlled support")
        self._generator = torch.Generator(device="cpu").manual_seed(int(sampling_seed))
        self._state: dict[str, Any] = {
            "schema_version": PANELGRAD_STATE_SCHEMA_VERSION,
            "method_family": PANELGRAD_METHOD_FAMILY,
            "method_version": PANELGRAD_METHOD_VERSION,
            "scope": "global",
            "ordered_granularities": ordered,
            "policy": {
                "refresh_interval_steps": int(refresh_interval_steps),
                "eta": float(eta),
                "temperature": float(temperature),
                "epsilon_schedule": resolved_epsilon_schedule,
                "relative_tolerance": 1e-6,
                "absolute_tolerance": 1e-8,
            },
            "support": support,
            "manifest_hashes": dict(manifest_hashes or {}),
            "refresh": {
                "phase": "refresh_pending",
                "refresh_index": -1,
                "last_boundary_step": None,
                "next_boundary_step": 0,
                "completed_steps_since_refresh": 0,
                "measurements": None,
                "scores": None,
                "q": None,
                "p": None,
                "entropy": None,
                "min_probability": None,
                "max_probability": None,
                "active_epsilon": None,
                "epsilon_schedule_step": None,
                "cost": None,
            },
            "sampling": {
                "seed_stream_name": "panelgrad_sampling",
                "resolved_seed": int(sampling_seed),
                "generator_state": self._generator.get_state(),
                "sample_count": 0,
                "exposure_counts": {label: 0 for label in ordered},
                "pending_action": None,
                "pending_generator_state_before_draw": None,
                "last_committed_action": None,
                "last_committed_probability": None,
            },
            "warmup": {
                "event_count": 0,
                "last_event": None,
            },
            "terminal": {
                "status": "continuing",
                "completed_step": None,
                "unused_refresh_performed": False,
                "unused_draw_performed": False,
            },
            "journal": {
                "path": "controller_metrics.jsonl",
                "event_count": 0,
                "last_committed_offset": None,
                "last_committed_hash": None,
            },
            "resume": {
                "resume_count": 0,
                "source_checkpoint": None,
                "compatibility_status": "fresh",
            },
            "failure": None,
        }
        if resolved_epsilon_schedule["type"] == "fixed":
            self._state["policy"]["epsilon"] = float(
                resolved_epsilon_schedule["start"]
            )

    @property
    def phase(self) -> str:
        return str(self._state["refresh"]["phase"])

    def state_dict(self) -> dict[str, Any]:
        self._state["sampling"]["generator_state"] = self._generator.get_state()
        return copy.deepcopy(self._state)

    def transaction_snapshot(self) -> dict[str, Any]:
        return self.state_dict()

    def restore_transaction_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        restored = copy.deepcopy(dict(snapshot))
        generator_state = restored["sampling"]["generator_state"]
        self._generator.set_state(generator_state.cpu())
        restored["sampling"]["generator_state"] = self._generator.get_state()
        self._state = restored

    def epsilon_for_next_refresh(self) -> dict[str, Any]:
        """Resolve refresh policy from committed PanelGrad optimizer steps."""

        schedule_step = int(self._state["sampling"]["sample_count"])
        return {
            "active_epsilon": epsilon_at_schedule_step(
                self._state["policy"]["epsilon_schedule"],
                schedule_step,
            ),
            "epsilon_schedule_step": schedule_step,
        }

    def install_refresh(
        self,
        measurement_result: Mapping[str, Any],
        *,
        boundary_step: int,
    ) -> dict[str, Any]:
        if self.phase != "refresh_pending":
            raise PanelGradError("PanelGrad refresh is not pending")
        ordered = self._state["ordered_granularities"]
        measurements = copy.deepcopy(list(measurement_result.get("measurements", [])))
        if [item.get("granularity") for item in measurements] != ordered:
            raise PanelGradError("refresh measurement granularity order mismatch")
        scores = [
            _finite_float(item.get("gradient_rms_score"), "gradient RMS score")
            for item in measurements
        ]
        refresh_policy = self.epsilon_for_next_refresh()
        probability = build_probability_snapshot(
            scores,
            eta=self._state["policy"]["eta"],
            temperature=self._state["policy"]["temperature"],
            epsilon=refresh_policy["active_epsilon"],
        )
        refresh = copy.deepcopy(self._state["refresh"])
        refresh.update(
            {
                "phase": "active_interval",
                "refresh_index": int(refresh["refresh_index"]) + 1,
                "last_boundary_step": int(boundary_step),
                "next_boundary_step": int(boundary_step)
                + int(self._state["policy"]["refresh_interval_steps"]),
                "completed_steps_since_refresh": 0,
                "measurements": measurements,
                "scores": probability["scores"],
                "q": probability["q"],
                "p": probability["p"],
                "entropy": probability["entropy"],
                "min_probability": probability["min_probability"],
                "max_probability": probability["max_probability"],
                **refresh_policy,
                "cost": {
                    key: copy.deepcopy(value)
                    for key, value in measurement_result.items()
                    if key
                    in {
                        "controller_example_count",
                        "controller_target_count",
                        "backward_evaluation_count",
                        "duration_seconds",
                    }
                },
            }
        )
        self._state["refresh"] = refresh
        return copy.deepcopy(refresh)

    def sample_action(self) -> dict[str, Any]:
        if self.phase != "active_interval":
            raise PanelGradError("PanelGrad requires a complete refresh before sampling")
        sampling = self._state["sampling"]
        if sampling["pending_action"] is not None:
            raise PanelGradError("PanelGrad already has an uncommitted action")
        before_draw = self._generator.get_state()
        p = torch.tensor(self._state["refresh"]["p"], dtype=torch.float64)
        index = int(torch.multinomial(p, 1, generator=self._generator).item())
        label = self._state["ordered_granularities"][index]
        action = {
            "scope": "global",
            "global_granularity": label,
            "block_granularities": None,
            "probability": float(p[index].item()),
            "refresh_index": int(self._state["refresh"]["refresh_index"]),
        }
        sampling["pending_generator_state_before_draw"] = before_draw
        sampling["pending_action"] = copy.deepcopy(action)
        sampling["generator_state"] = self._generator.get_state()
        return action

    def rollback_pending_action(self) -> None:
        sampling = self._state["sampling"]
        before_draw = sampling.get("pending_generator_state_before_draw")
        if before_draw is not None:
            self._generator.set_state(before_draw.cpu())
        sampling["pending_action"] = None
        sampling["pending_generator_state_before_draw"] = None
        sampling["generator_state"] = self._generator.get_state()

    def commit_pending_action(self, *, completed_step: int) -> dict[str, Any]:
        sampling = self._state["sampling"]
        action = sampling.get("pending_action")
        if action is None:
            raise PanelGradError("PanelGrad has no pending action to commit")
        label = str(action["global_granularity"])
        sampling["sample_count"] = int(sampling["sample_count"]) + 1
        sampling["exposure_counts"][label] = (
            int(sampling["exposure_counts"][label]) + 1
        )
        sampling["last_committed_action"] = label
        sampling["last_committed_probability"] = float(action["probability"])
        sampling["pending_action"] = None
        sampling["pending_generator_state_before_draw"] = None
        sampling["generator_state"] = self._generator.get_state()

        refresh = self._state["refresh"]
        refresh["completed_steps_since_refresh"] = (
            int(refresh["completed_steps_since_refresh"]) + 1
        )
        if refresh["completed_steps_since_refresh"] == int(
            self._state["policy"]["refresh_interval_steps"]
        ):
            refresh["phase"] = "refresh_pending"
            refresh["next_boundary_step"] = int(completed_step)
        elif refresh["completed_steps_since_refresh"] > int(
            self._state["policy"]["refresh_interval_steps"]
        ):
            raise PanelGradError("PanelGrad interval progress exceeded H")
        return copy.deepcopy(action)

    def record_warmup_event(self, event: Mapping[str, Any]) -> None:
        """Audit warmup without changing PanelGrad refresh or exposure state."""

        warmup = self._state["warmup"]
        warmup["event_count"] = int(warmup["event_count"]) + 1
        warmup["last_event"] = copy.deepcopy(dict(event))

    def record_journal_commit(self, commit: Mapping[str, Any] | None) -> None:
        if not isinstance(commit, Mapping):
            return
        journal = self._state["journal"]
        journal["event_count"] = int(journal["event_count"]) + 1
        if commit.get("path") is not None:
            journal["path"] = str(commit["path"])
        journal["last_committed_offset"] = commit.get("last_committed_offset")
        journal["last_committed_hash"] = commit.get(
            "event_hash", commit.get("last_committed_hash")
        )

    def finish_training(self, *, completed_step: int) -> dict[str, Any]:
        """Enter a terminal phase without performing another refresh or draw."""

        sampling = self._state["sampling"]
        if sampling["pending_action"] is not None:
            raise PanelGradError("cannot finish with an uncommitted PanelGrad action")
        refresh = self._state["refresh"]
        terminal_status = (
            "terminal_complete"
            if refresh["phase"] == "refresh_pending"
            else "terminal_partial"
        )
        refresh["phase"] = terminal_status
        terminal = self._state["terminal"]
        terminal.update(
            {
                "status": terminal_status,
                "completed_step": int(completed_step),
                "unused_refresh_performed": False,
                "unused_draw_performed": False,
            }
        )
        return copy.deepcopy(terminal)


def build_panelgrad_controller(
    config: Mapping[str, Any],
    support_identity: Mapping[str, Any],
) -> PanelGradController:
    model = config["model"]
    panelgrad = model["panelgrad"]
    manifest_fields = (
        "data_roles_manifest_hash",
        "optimizer_training_manifest_hash",
        "controller_manifest_hash",
        "validation_manifest_hash",
        "final_holdout_manifest_hash",
    )
    return PanelGradController(
        ordered_granularities=panelgrad["ordered_granularities"],
        refresh_interval_steps=panelgrad["refresh_interval_steps"],
        eta=panelgrad["eta"],
        temperature=panelgrad["temperature"],
        epsilon=panelgrad.get("epsilon"),
        epsilon_schedule=panelgrad.get("epsilon_schedule"),
        sampling_seed=panelgrad["sampling_seed"],
        support_identity=support_identity,
        manifest_hashes={field: config.get(field) for field in manifest_fields},
    )


def validate_panelgrad_state(
    state: Mapping[str, Any] | None,
    *,
    config: Mapping[str, Any],
    support_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the versioned PanelGrad state before save or resume."""

    if not isinstance(state, Mapping):
        raise PanelGradError("checkpoint is missing panelgrad_state")
    result = copy.deepcopy(dict(state))
    panelgrad = config["model"]["panelgrad"]
    if result.get("schema_version") == 1:
        if "epsilon_schedule" in panelgrad:
            raise PanelGradError(
                "panelgrad_state version 1 cannot resume an epsilon-scheduled run"
            )
        legacy_policy = result.get("policy")
        legacy_refresh = result.get("refresh")
        legacy_sampling = result.get("sampling")
        if not all(
            isinstance(value, Mapping)
            for value in (legacy_policy, legacy_refresh, legacy_sampling)
        ):
            raise PanelGradError("panelgrad_state version 1 is malformed")
        legacy_epsilon = legacy_policy.get("epsilon")
        legacy_policy["epsilon_schedule"] = resolve_epsilon_schedule(
            epsilon=legacy_epsilon,
            epsilon_schedule=None,
        )
        if legacy_refresh.get("p") is None:
            legacy_refresh["active_epsilon"] = None
            legacy_refresh["epsilon_schedule_step"] = None
        else:
            sample_count = int(legacy_sampling.get("sample_count", -1))
            interval_progress = int(
                legacy_refresh.get("completed_steps_since_refresh", -1)
            )
            schedule_step = sample_count - interval_progress
            if schedule_step < 0:
                raise PanelGradError(
                    "panelgrad_state version 1 interval progress is invalid"
                )
            legacy_refresh["active_epsilon"] = float(legacy_epsilon)
            legacy_refresh["epsilon_schedule_step"] = schedule_step
        result["schema_version"] = PANELGRAD_STATE_SCHEMA_VERSION
        resume = result.setdefault("resume", {})
        if isinstance(resume, dict):
            resume["state_migrated_from_schema_version"] = 1

    resolved_epsilon_schedule = resolve_epsilon_schedule(
        epsilon=panelgrad.get("epsilon"),
        epsilon_schedule=panelgrad.get("epsilon_schedule"),
    )
    expected = {
        "schema_version": PANELGRAD_STATE_SCHEMA_VERSION,
        "method_family": PANELGRAD_METHOD_FAMILY,
        "method_version": PANELGRAD_METHOD_VERSION,
        "scope": "global",
        "ordered_granularities": list(panelgrad["ordered_granularities"]),
    }
    mismatches = {
        field: (result.get(field), value)
        for field, value in expected.items()
        if result.get(field) != value
    }
    if mismatches:
        raise PanelGradError(f"panelgrad_state identity mismatch: {mismatches}")
    expected_policy = {
        "refresh_interval_steps": int(panelgrad["refresh_interval_steps"]),
        "eta": float(panelgrad["eta"]),
        "temperature": float(panelgrad["temperature"]),
        "epsilon_schedule": resolved_epsilon_schedule,
        "relative_tolerance": float(panelgrad["relative_tolerance"]),
        "absolute_tolerance": float(panelgrad["absolute_tolerance"]),
    }
    if resolved_epsilon_schedule["type"] == "fixed":
        expected_policy["epsilon"] = float(resolved_epsilon_schedule["start"])
    if result.get("policy") != expected_policy:
        raise PanelGradError("panelgrad_state policy mismatch")
    expected_support = support_identity or {
        "controlled_support_counts": panelgrad["controlled_support_counts"],
        "controlled_support_hash": panelgrad["controlled_support_hash"],
    }
    saved_support = result.get("support", {})
    for field in ("controlled_support_counts", "controlled_support_hash"):
        if saved_support.get(field) != expected_support.get(field):
            raise PanelGradError(f"panelgrad_state {field} mismatch")
    refresh = result.get("refresh")
    sampling = result.get("sampling")
    if not isinstance(refresh, Mapping) or refresh.get("phase") not in {
        "refresh_pending",
        "active_interval",
        "terminal_partial",
        "terminal_complete",
        "failed",
    }:
        raise PanelGradError("panelgrad_state refresh phase is invalid")
    if not isinstance(sampling, Mapping) or not torch.is_tensor(
        sampling.get("generator_state")
    ):
        raise PanelGradError("panelgrad_state sampling generator is invalid")
    ordered = expected["ordered_granularities"]
    exposures = sampling.get("exposure_counts")
    if not isinstance(exposures, Mapping) or list(exposures) != ordered or any(
        isinstance(exposures[label], bool)
        or not isinstance(exposures[label], int)
        or exposures[label] < 0
        for label in ordered
    ):
        raise PanelGradError("panelgrad_state exposure counts are invalid")
    if int(sampling.get("sample_count", -1)) != sum(exposures.values()):
        raise PanelGradError("panelgrad_state sample and exposure counts disagree")
    if sampling.get("pending_action") is not None:
        raise PanelGradError("checkpoint cannot contain an uncommitted PanelGrad action")
    if refresh.get("p") is not None:
        active_epsilon = _finite_float(
            refresh.get("active_epsilon"),
            "panelgrad_state active epsilon",
        )
        schedule_step = refresh.get("epsilon_schedule_step")
        if (
            isinstance(schedule_step, bool)
            or not isinstance(schedule_step, int)
            or schedule_step < 0
        ):
            raise PanelGradError("panelgrad_state epsilon schedule step is invalid")
        interval_progress = refresh.get("completed_steps_since_refresh")
        if (
            isinstance(interval_progress, bool)
            or not isinstance(interval_progress, int)
            or interval_progress < 0
            or interval_progress > expected_policy["refresh_interval_steps"]
        ):
            raise PanelGradError("panelgrad_state interval progress is invalid")
        expected_schedule_step = int(sampling["sample_count"]) - interval_progress
        if schedule_step != expected_schedule_step:
            raise PanelGradError("panelgrad_state epsilon schedule progress is invalid")
        scheduled_epsilon = epsilon_at_schedule_step(
            resolved_epsilon_schedule,
            schedule_step,
        )
        if not math.isclose(active_epsilon, scheduled_epsilon, rel_tol=0.0, abs_tol=1e-15):
            raise PanelGradError("panelgrad_state active epsilon is invalid")
        expected_probability = build_probability_snapshot(
            refresh["scores"],
            eta=expected_policy["eta"],
            temperature=expected_policy["temperature"],
            epsilon=active_epsilon,
        )
        p = torch.tensor(refresh["p"], dtype=torch.float64)
        if len(p) != len(ordered) or not math.isclose(
            float(p.sum()), 1.0, rel_tol=1e-6, abs_tol=1e-8
        ):
            raise PanelGradError("panelgrad_state p is invalid")
        if not torch.allclose(
            p,
            torch.tensor(expected_probability["p"], dtype=torch.float64),
            rtol=expected_policy["relative_tolerance"],
            atol=expected_policy["absolute_tolerance"],
        ):
            raise PanelGradError("panelgrad_state p does not match saved scores")
    elif (
        refresh.get("active_epsilon") is not None
        or refresh.get("epsilon_schedule_step") is not None
    ):
        raise PanelGradError(
            "panelgrad_state without a snapshot cannot have active epsilon state"
        )
    expected_manifests = {
        field: config.get(field)
        for field in (
            "data_roles_manifest_hash",
            "optimizer_training_manifest_hash",
            "controller_manifest_hash",
            "validation_manifest_hash",
            "final_holdout_manifest_hash",
        )
    }
    if result.get("manifest_hashes") != expected_manifests:
        raise PanelGradError("panelgrad_state data-role manifest mismatch")
    journal = result.get("journal")
    if not isinstance(journal, Mapping) or int(journal.get("event_count", -1)) < 0:
        raise PanelGradError("panelgrad_state journal provenance is invalid")
    return result


def restore_panelgrad_controller(
    config: Mapping[str, Any],
    support_identity: Mapping[str, Any],
    state: Mapping[str, Any] | None,
) -> PanelGradController:
    validated = validate_panelgrad_state(
        state,
        config=config,
        support_identity=support_identity,
    )
    controller = build_panelgrad_controller(config, support_identity)
    controller.restore_transaction_snapshot(validated)
    controller._state["resume"]["resume_count"] = (
        int(controller._state["resume"]["resume_count"]) + 1
    )
    controller._state["resume"]["compatibility_status"] = "resumed"
    return controller


__all__ = [
    "CONTROLLED_SUPPORT_SCHEMA_VERSION",
    "PANELGRAD_METHOD_FAMILY",
    "PANELGRAD_METHOD_VERSION",
    "PANELGRAD_STATE_SCHEMA_VERSION",
    "PanelGradController",
    "PanelGradError",
    "build_panelgrad_controller",
    "build_probability_snapshot",
    "epsilon_at_schedule_step",
    "gradient_rms_from_squared_norm",
    "measure_panelgrad_gradients",
    "resolve_controlled_ffn_support",
    "restore_panelgrad_controller",
    "resolve_epsilon_schedule",
    "uses_panelgrad",
    "validate_panelgrad_state",
]
