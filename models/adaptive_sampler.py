"""Adaptive per-block sampling scaffolding for MatFormer experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VALID_ADAPTIVE_SAMPLER_STRATEGIES = ("thompson", "ucb")


@dataclass(slots=True)
class AdaptiveSamplerBlockStat:
    """Running statistics for one block/granularity pair."""

    mean_reward: float = 0.0
    count: int = 0
    last_seen_step: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AdaptiveSamplerState:
    """Serializable state for adaptive per-block sampling."""

    strategy_name: str = "thompson"
    phase: str = "fresh"
    step: int = 0
    epoch: int = 0
    exploration_scale: float = 1.0
    decay_rate: float = 0.0
    stats: dict[int, dict[str, AdaptiveSamplerBlockStat]] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_adaptive_sampler_strategy(strategy_name: str) -> None:
    if strategy_name not in VALID_ADAPTIVE_SAMPLER_STRATEGIES:
        raise ValueError(
            "strategy_name must be one of "
            f"{list(VALID_ADAPTIVE_SAMPLER_STRATEGIES)}"
        )


def build_adaptive_sampler_state(
    strategy_name: str = "thompson",
    *,
    phase: str = "fresh",
    step: int = 0,
    epoch: int = 0,
    exploration_scale: float = 1.0,
    decay_rate: float = 0.0,
) -> AdaptiveSamplerState:
    validate_adaptive_sampler_strategy(strategy_name)
    return AdaptiveSamplerState(
        strategy_name=strategy_name,
        phase=phase,
        step=step,
        epoch=epoch,
        exploration_scale=exploration_scale,
        decay_rate=decay_rate,
    )


def summarize_adaptive_sampler_state(
    state: AdaptiveSamplerState,
) -> dict[str, Any]:
    return state.to_dict()


__all__ = [
    "VALID_ADAPTIVE_SAMPLER_STRATEGIES",
    "AdaptiveSamplerBlockStat",
    "AdaptiveSamplerState",
    "validate_adaptive_sampler_strategy",
    "build_adaptive_sampler_state",
    "summarize_adaptive_sampler_state",
]
