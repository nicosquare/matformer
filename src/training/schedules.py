"""Resolved learning-rate schedule contracts and metric provenance."""

from __future__ import annotations

import math
from typing import Any, Mapping


WSD_SCHEDULER_NAME = "warmup_stable_decay"
WSD_SCHEDULE_POLICY = "ratio_decay_over_total_steps"
WSD_SCHEDULE_POLICY_VERSION = 1


def uses_wsd_schedule(training: Mapping[str, Any]) -> bool:
    """Return whether a resolved training section selects ratio-based WSD."""

    return str(training.get("scheduler_name", "")) == WSD_SCHEDULER_NAME


def scheduler_contract_from_training(
    training: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the immutable resolved scheduler contract when one is available."""

    contract = training.get("scheduler_contract")
    if isinstance(contract, Mapping):
        return dict(contract)
    scheduler = training.get("scheduler")
    if isinstance(scheduler, Mapping):
        nested = scheduler.get("contract")
        if isinstance(nested, Mapping):
            return dict(nested)
    return None


def wsd_learning_rate_factor(
    scheduler_position: int,
    *,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    warmup_type: str,
    decay_type: str,
    min_lr_ratio: float,
    num_cycles: float,
) -> float:
    """Reproduce Transformers 5.8's WSD lambda at one scheduler position."""

    position = int(scheduler_position)
    if position < warmup_steps:
        progress = float(position) / float(max(1, warmup_steps))
        if warmup_type == "linear":
            factor = progress
        elif warmup_type == "cosine":
            factor = 0.5 * (1.0 - math.cos(math.pi * progress))
        elif warmup_type == "1-sqrt":
            factor = 1.0 - math.sqrt(1.0 - progress)
        else:
            raise ValueError(f"Unsupported WSD warmup type: {warmup_type}")
        return max(0.0, factor * (1.0 - min_lr_ratio) + min_lr_ratio)

    cooldown_start = warmup_steps + stable_steps
    if position < cooldown_start:
        return 1.0

    schedule_end = cooldown_start + decay_steps
    if position < schedule_end:
        progress = float(position - cooldown_start) / float(max(1, decay_steps))
        if decay_type == "linear":
            factor = 1.0 - progress
        elif decay_type == "cosine":
            factor = 0.5 * (
                1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)
            )
        elif decay_type == "1-sqrt":
            factor = 1.0 - math.sqrt(progress)
        else:
            raise ValueError(f"Unsupported WSD decay type: {decay_type}")
        return max(0.0, factor * (1.0 - min_lr_ratio) + min_lr_ratio)
    return float(min_lr_ratio)


def scheduler_metric_fields(
    training: Mapping[str, Any],
    *,
    scheduler_position: int,
    learning_rate: float | None = None,
) -> dict[str, Any]:
    """Build per-row schedule phase fields for an update or validation boundary.

    ``scheduler_position`` follows ``LambdaLR``: position zero is the learning
    rate used by the first optimizer update, while a validation after committed
    step ``n`` observes scheduler position ``n``.
    """

    position = max(0, int(scheduler_position))
    warmup_steps = int(training.get("resolved_warmup_steps", 0))
    peak_lr = float(
        training.get("resolved_learning_rate", training.get("learning_rate", 0.0))
    )
    contract = scheduler_contract_from_training(training)
    fields: dict[str, Any] = {
        "scheduler_position": position,
        "scheduler_phase": "warmup" if position < warmup_steps else "schedule",
        "scheduler_phase_step": (
            position if position < warmup_steps else position - warmup_steps
        ),
        "scheduler_phase_progress": (
            min(position / max(warmup_steps, 1), 1.0)
            if position < warmup_steps
            else None
        ),
        "scheduler_policy_version": None,
        "scheduler_warmup_steps": warmup_steps,
        "scheduler_stable_steps": None,
        "scheduler_decay_steps": None,
        "scheduler_decay_ratio": None,
        "scheduler_cooldown_start_step": None,
        "scheduler_cooldown_start_tokens": None,
        "scheduler_warmup_type": None,
        "scheduler_decay_type": None,
        "scheduler_min_lr_ratio": None,
        "scheduler_min_learning_rate": None,
        "scheduler_num_cycles": None,
        "scheduler_schedule_end_step": None,
        "scheduler_schedule_end_tokens": None,
    }

    if contract is not None and contract.get("name") == WSD_SCHEDULER_NAME:
        stable_steps = int(contract["stable_steps"])
        decay_steps = int(contract["decay_steps"])
        cooldown_start = int(contract["cooldown_start_step"])
        max_steps = int(contract["max_steps"])
        if position < warmup_steps:
            phase = "warmup"
            phase_step = position
            phase_progress = position / max(warmup_steps, 1)
        elif position < cooldown_start:
            phase = "stable"
            phase_step = position - warmup_steps
            phase_progress = phase_step / max(stable_steps, 1)
        else:
            phase = "cooldown"
            phase_step = min(max(position - cooldown_start, 0), decay_steps)
            phase_progress = min(phase_step / max(decay_steps, 1), 1.0)
        fields.update(
            {
                "scheduler_phase": phase,
                "scheduler_phase_step": phase_step,
                "scheduler_phase_progress": phase_progress,
                "scheduler_policy_version": contract["policy_version"],
                "scheduler_warmup_steps": warmup_steps,
                "scheduler_stable_steps": stable_steps,
                "scheduler_decay_steps": decay_steps,
                "scheduler_decay_ratio": float(contract["decay_ratio"]),
                "scheduler_cooldown_start_step": cooldown_start,
                "scheduler_cooldown_start_tokens": int(
                    contract["cooldown_start_tokens"]
                ),
                "scheduler_warmup_type": contract["warmup_type"],
                "scheduler_decay_type": contract["decay_type"],
                "scheduler_min_lr_ratio": float(contract["min_lr_ratio"]),
                "scheduler_min_learning_rate": float(
                    contract["min_learning_rate"]
                ),
                "scheduler_num_cycles": float(contract["num_cycles"]),
                "scheduler_schedule_end_step": max_steps,
                "scheduler_schedule_end_tokens": int(
                    contract.get(
                        "schedule_end_tokens", training.get("token_budget", 0)
                    )
                ),
            }
        )
        if learning_rate is None:
            factor = wsd_learning_rate_factor(
                min(position, max_steps),
                warmup_steps=warmup_steps,
                stable_steps=stable_steps,
                decay_steps=decay_steps,
                warmup_type=str(contract["warmup_type"]),
                decay_type=str(contract["decay_type"]),
                min_lr_ratio=float(contract["min_lr_ratio"]),
                num_cycles=float(contract["num_cycles"]),
            )
            learning_rate = peak_lr * factor

    fields["learning_rate"] = (
        float(learning_rate) if learning_rate is not None else None
    )
    return fields


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def complexity_log_exponents(
    counts: Mapping[str, int], *, gamma_min: float = 0.5, gamma_max: float = 2.0,
) -> dict[str, float]:
    """Derive decay exponents from active trainable scalars, never width fractions."""
    if not counts or any(isinstance(c, bool) or not isinstance(c, int) or c <= 0
                         for c in counts.values()):
        raise ValueError("complexity counts must be positive integers")
    low = _positive_finite(gamma_min, "gamma_min")
    high = _positive_finite(gamma_max, "gamma_max")
    if low >= high:
        raise ValueError("gamma_min must be less than gamma_max")
    smallest, largest = min(counts.values()), max(counts.values())
    if smallest == largest:
        raise ValueError("CaLR requires distinct complexity extrema")
    denominator = math.log(largest / smallest)
    return {width: high if count == smallest else low if count == largest else
            high - (high - low) * math.log(count / smallest) / denominator
            for width, count in counts.items()}


def warmup_polynomial_learning_rate(
    position: int, *, peak: float, warmup_steps: int, horizon: int,
    exponent: float = 1.0,
) -> float:
    """Analytic pre-update LR; the nominal clock always uses exponent one."""
    if any(isinstance(v, bool) or not isinstance(v, int)
           for v in (position, warmup_steps, horizon)):
        raise ValueError("position, warmup_steps and horizon must be integers")
    if not 0 < warmup_steps < horizon or not 0 <= position <= horizon:
        raise ValueError("require 0 < warmup_steps < horizon and 0 <= position <= horizon")
    peak = _positive_finite(peak, "peak")
    exponent = _positive_finite(exponent, "exponent")
    if position == horizon:
        return 0.0
    if position < warmup_steps:
        return peak * position / warmup_steps
    return peak * ((horizon - position) / (horizon - warmup_steps)) ** exponent


def polynomial_schedule_evidence(contract: Mapping[str, Any], width: str,
                                 position: int) -> dict[str, Any]:
    """Return analytic evidence, explicitly distinct from measured applied LRs."""
    from src.utils.config import parse_linear_calr_schedule_contract
    resolved = parse_linear_calr_schedule_contract(contract)
    if width not in resolved['ordered_granularities']:
        raise ValueError(f"Unknown schedule width: {width}")
    arguments = dict(peak=resolved['peak'], warmup_steps=resolved['warmup_steps'],
                     horizon=resolved['horizon'])
    return {'evidence_kind': 'analytic', 'pre_update_position': position,
            'width': width, 'complexity': resolved['complexity_counts'][width],
            'exponent': resolved['exponents'][width],
            'analytic_effective_learning_rate': warmup_polynomial_learning_rate(
                position, exponent=resolved['exponents'][width], **arguments),
            'analytic_nominal_learning_rate': warmup_polynomial_learning_rate(
                position, exponent=1.0, **arguments)}
