"""Adaptive per-block sampling helpers for MatFormer experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
import random
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from models.granularity import MATFORMER_GRANULARITY_ORDER, validate_granularity


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
    stats: dict[int, dict[str, AdaptiveSamplerBlockStat]] | None = None,
) -> AdaptiveSamplerState:
    validate_adaptive_sampler_strategy(strategy_name)
    return AdaptiveSamplerState(
        strategy_name=strategy_name,
        phase=phase,
        step=step,
        epoch=epoch,
        exploration_scale=exploration_scale,
        decay_rate=decay_rate,
        stats=stats or {},
    )


def coerce_adaptive_sampler_state(
    state: AdaptiveSamplerState | Mapping[str, Any] | None,
) -> AdaptiveSamplerState | None:
    if state is None:
        return None
    if isinstance(state, AdaptiveSamplerState):
        return state
    if not isinstance(state, Mapping):
        raise TypeError("Adaptive sampler state must be a mapping or dataclass")

    stats: dict[int, dict[str, AdaptiveSamplerBlockStat]] = {}
    raw_stats = state.get("stats", {})
    if isinstance(raw_stats, Mapping):
        for raw_block_index, raw_block_stats in raw_stats.items():
            block_index = int(raw_block_index)
            stats[block_index] = {}
            if not isinstance(raw_block_stats, Mapping):
                continue
            for granularity, raw_stat in raw_block_stats.items():
                stats[block_index][str(granularity)] = _coerce_block_stat(raw_stat)

    strategy_name = str(state.get("strategy_name", "thompson"))
    validate_adaptive_sampler_strategy(strategy_name)
    return AdaptiveSamplerState(
        strategy_name=strategy_name,
        phase=str(state.get("phase", "fresh")),
        step=int(state.get("step", 0)),
        epoch=int(state.get("epoch", 0)),
        exploration_scale=float(state.get("exploration_scale", 1.0)),
        decay_rate=float(state.get("decay_rate", 0.0)),
        stats=stats,
    )


def normalize_adaptive_sampler_state(
    state: AdaptiveSamplerState | Mapping[str, Any],
    *,
    block_count: int,
    granularities: Sequence[str] | None = None,
) -> AdaptiveSamplerState:
    """Ensure the state has compatible stats for the configured model."""

    if block_count <= 0:
        raise ValueError("block_count must be positive")

    state_obj = coerce_adaptive_sampler_state(state)
    if state_obj is None:
        raise ValueError("state cannot be None")

    validate_adaptive_sampler_state(
        state_obj,
        expected_block_count=block_count,
        granularities=granularities,
        allow_missing_stats=True,
    )

    ordered_granularities = tuple(granularities or MATFORMER_GRANULARITY_ORDER)
    for block_index in range(block_count):
        block_stats = state_obj.stats.setdefault(block_index, {})
        for granularity in ordered_granularities:
            validate_granularity(granularity)
            block_stats.setdefault(granularity, AdaptiveSamplerBlockStat())
    return state_obj


def validate_adaptive_sampler_state(
    state: AdaptiveSamplerState | Mapping[str, Any],
    *,
    expected_block_count: int | None = None,
    granularities: Sequence[str] | None = None,
    allow_missing_stats: bool = False,
) -> None:
    state_obj = coerce_adaptive_sampler_state(state)
    if state_obj is None:
        raise ValueError("state cannot be None")

    validate_adaptive_sampler_strategy(state_obj.strategy_name)
    if state_obj.step < 0:
        raise ValueError("step must be non-negative")
    if state_obj.epoch < 0:
        raise ValueError("epoch must be non-negative")
    if state_obj.exploration_scale < 0:
        raise ValueError("exploration_scale must be non-negative")
    if state_obj.decay_rate < 0:
        raise ValueError("decay_rate must be non-negative")

    ordered_granularities = tuple(granularities or MATFORMER_GRANULARITY_ORDER)
    if granularities is not None:
        for granularity in ordered_granularities:
            validate_granularity(granularity)

    if expected_block_count is not None and expected_block_count < 0:
        raise ValueError("expected_block_count must be non-negative")

    if expected_block_count is not None and not allow_missing_stats:
        if len(state_obj.stats) != expected_block_count:
            raise ValueError(
                "adaptive sampler state block count does not match the model"
            )

    for block_index, block_stats in state_obj.stats.items():
        if expected_block_count is not None and (
            block_index < 0 or block_index >= expected_block_count
        ):
            raise ValueError(
                "adaptive sampler state contains an incompatible block index"
            )
        if not isinstance(block_stats, Mapping):
            raise ValueError("adaptive sampler block stats must be mappings")
        missing_granularities = [
            granularity
            for granularity in ordered_granularities
            if granularity not in block_stats
        ]
        if missing_granularities and not allow_missing_stats:
            raise ValueError(
                "adaptive sampler state is missing granularity stats: "
                f"{missing_granularities}"
            )
        for granularity, block_stat in block_stats.items():
            validate_granularity(str(granularity))
            if not isinstance(block_stat, AdaptiveSamplerBlockStat):
                _coerce_block_stat(block_stat)


def score_adaptive_sampler_actions(
    state: AdaptiveSamplerState | Mapping[str, Any],
    block_index: int,
    step: int,
    phase: str,
    granularities: Sequence[str] | None = None,
) -> dict[str, float]:
    """Return one score per granularity for a single transformer block."""

    state_obj = coerce_adaptive_sampler_state(state)
    if state_obj is None:
        raise ValueError("state cannot be None")
    validate_adaptive_sampler_state(
        state_obj,
        granularities=granularities,
        allow_missing_stats=True,
    )

    ordered_granularities = tuple(granularities or MATFORMER_GRANULARITY_ORDER)
    block_stats = _ensure_block_stats(
        state_obj,
        block_index,
        ordered_granularities,
    )
    scores: dict[str, float] = {}
    for granularity in ordered_granularities:
        stat = block_stats[granularity]
        mean_factor = _mean_factor(state_obj, stat, step)
        age_factor = _age_factor(stat, step)
        if state_obj.strategy_name == "ucb":
            exploration_bonus = _ucb_bonus(
                exploration_scale=state_obj.exploration_scale,
                count=stat.count,
                step=step,
            )
        else:
            exploration_bonus = _thompson_bonus(
                state=state_obj,
                block_index=block_index,
                granularity=granularity,
                step=step,
                phase=phase,
                count=stat.count,
                exploration_scale=state_obj.exploration_scale,
            )
        scores[granularity] = stat.mean_reward * mean_factor + (
            exploration_bonus * age_factor
        )
    return scores


def select_adaptive_sampler_layer_granularities(
    state: AdaptiveSamplerState | Mapping[str, Any],
    *,
    block_count: int,
    step: int,
    phase: str,
    granularities: Sequence[str] | None = None,
) -> list[str]:
    """Select one granularity per transformer block."""

    state_obj = coerce_adaptive_sampler_state(state)
    if state_obj is None:
        raise ValueError("state cannot be None")
    normalized_state = normalize_adaptive_sampler_state(
        state_obj,
        block_count=block_count,
        granularities=granularities,
    )
    normalized_state.phase = phase
    normalized_state.step = step

    ordered_granularities = tuple(granularities or MATFORMER_GRANULARITY_ORDER)
    selected: list[str] = []
    for block_index in range(block_count):
        scores = score_adaptive_sampler_actions(
            normalized_state,
            block_index=block_index,
            step=step,
            phase=phase,
            granularities=ordered_granularities,
        )
        selected.append(
            max(
                ordered_granularities,
                key=lambda granularity: (
                    scores[granularity],
                    -ordered_granularities.index(granularity),
                ),
            )
        )
    return selected


def build_adaptive_reward_record(
    previous_loss: float | None,
    current_loss: float,
    correction_penalty: float,
    reward_penalty_weight: float,
    *,
    phase: str,
    step: int,
    epoch: int,
) -> dict[str, Any]:
    """Build the scalar reward used to update the adaptive sampler."""

    previous_loss_value = float(previous_loss) if previous_loss is not None else None
    current_loss_value = float(current_loss)
    correction_penalty_value = float(correction_penalty)
    reward_penalty_weight_value = float(reward_penalty_weight)
    normalized_correction_penalty = (
        correction_penalty_value * reward_penalty_weight_value
    )
    loss_improvement = (
        0.0
        if previous_loss_value is None
        else previous_loss_value - current_loss_value
    )
    reward = loss_improvement - normalized_correction_penalty
    return {
        "previous_loss": previous_loss_value,
        "current_loss": current_loss_value,
        "loss_improvement": loss_improvement,
        "correction_penalty": correction_penalty_value,
        "reward_penalty_weight": reward_penalty_weight_value,
        "normalized_correction_penalty": normalized_correction_penalty,
        "reward": reward,
        "phase": phase,
        "step": step,
        "epoch": epoch,
    }


def update_adaptive_sampler_state(
    state: AdaptiveSamplerState | Mapping[str, Any],
    reward_record: Mapping[str, Any],
    sampled_pattern: Mapping[int, str] | Sequence[str],
) -> AdaptiveSamplerState:
    """Update the running state after one sampled training step."""

    state_obj = coerce_adaptive_sampler_state(state)
    if state_obj is None:
        raise ValueError("state cannot be None")
    validate_adaptive_sampler_state(
        state_obj,
        allow_missing_stats=True,
    )

    step = int(reward_record.get("step", state_obj.step))
    epoch = int(reward_record.get("epoch", state_obj.epoch))
    phase = str(reward_record.get("phase", state_obj.phase))
    reward = float(reward_record.get("reward", 0.0))

    sampled_items = _normalize_sampled_pattern(sampled_pattern)
    if not sampled_items:
        raise ValueError("sampled_pattern must not be empty")

    for block_index, granularity in sampled_items:
        validate_granularity(granularity)
        block_stats = _ensure_block_stats(
            state_obj,
            block_index,
            tuple(sampled_granularity_keys(state_obj, block_index, granularity)),
        )
        stat = block_stats[granularity]
        stat.count += 1
        stat.mean_reward = (
            (1.0 - state_obj.decay_rate) * stat.mean_reward
            + state_obj.decay_rate * reward
        )
        stat.last_seen_step = step

    state_obj.phase = phase
    state_obj.step = step
    state_obj.epoch = epoch
    return state_obj


def sampled_granularity_keys(
    state: AdaptiveSamplerState,
    block_index: int,
    fallback_granularity: str | None = None,
) -> tuple[str, ...]:
    block_stats = state.stats.get(block_index)
    if isinstance(block_stats, Mapping) and block_stats:
        return tuple(str(granularity) for granularity in block_stats.keys())
    if fallback_granularity is not None:
        return (fallback_granularity,)
    return MATFORMER_GRANULARITY_ORDER


def summarize_adaptive_sampler_state(
    state: AdaptiveSamplerState | Mapping[str, Any],
) -> dict[str, Any]:
    coerced_state = coerce_adaptive_sampler_state(state)
    if coerced_state is None:
        raise ValueError("state cannot be None")
    return coerced_state.to_dict()


def build_adaptive_sampler_artifact_fields(
    config: Mapping[str, Any],
    run_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize adaptive sampler provenance for config and run artifacts."""

    model = config.get("model", {})
    run = config.get("run", {})
    if not isinstance(model, Mapping):
        model = {}
    if not isinstance(run, Mapping):
        run = {}
    if run_state is None or not isinstance(run_state, Mapping):
        run_state = {}

    output_dir_value = run.get("output_dir")
    output_dir = Path(str(output_dir_value)) if output_dir_value else None

    reward_summary = _first_present_mapping_value(
        run_state,
        model,
        key="adaptive_reward_summary",
    )
    correction_penalty_summary = _first_present_mapping_value(
        run_state,
        model,
        key="adaptive_correction_penalty_summary",
    )
    sampler_state = _first_present_mapping_value(
        run_state,
        model,
        key="adaptive_sampler_state",
    )

    return {
        "correction_mode": model.get("correction_mode"),
        "membership_correction": model.get("membership_correction"),
        "sampler_strategy": model.get("adaptive_sampler_strategy"),
        "adaptive_sampler_strategy": model.get("adaptive_sampler_strategy"),
        "adaptive_sampler_exploration_scale": model.get(
            "adaptive_sampler_exploration_scale"
        ),
        "adaptive_sampler_decay_rate": model.get("adaptive_sampler_decay_rate"),
        "adaptive_sampler_reward_penalty_weight": model.get(
            "adaptive_sampler_reward_penalty_weight"
        ),
        "sampler_state": sampler_state,
        "adaptive_sampler_state": sampler_state,
        "adaptive_sampler_previous_loss": _first_present_mapping_value(
            run_state,
            model,
            key="adaptive_sampler_previous_loss",
        ),
        "adaptive_sampler_previous_pattern": _first_present_mapping_value(
            run_state,
            model,
            key="adaptive_sampler_previous_pattern",
        ),
        "adaptive_reward_summary": reward_summary,
        "adaptive_correction_penalty_summary": correction_penalty_summary,
        "reward": (
            reward_summary.get("reward")
            if isinstance(reward_summary, Mapping)
            else None
        ),
        "correction_penalty": (
            correction_penalty_summary.get("correction_penalty")
            if isinstance(correction_penalty_summary, Mapping)
            else None
        ),
        "output_root": run.get("output_root"),
        "output_dir": run.get("output_dir"),
        "metrics_path": str(output_dir / "metrics.csv") if output_dir else None,
        "scaling_results_path": (
            str(output_dir / "scaling_results.csv") if output_dir else None
        ),
        "extraction_metadata_path": (
            str(output_dir / "extraction_metadata.json")
            if output_dir and run.get("model_family") == "nested"
            else None
        ),
    }


def _coerce_block_stat(raw_stat: Any) -> AdaptiveSamplerBlockStat:
    if isinstance(raw_stat, AdaptiveSamplerBlockStat):
        return raw_stat
    if isinstance(raw_stat, Mapping):
        return AdaptiveSamplerBlockStat(
            mean_reward=float(raw_stat.get("mean_reward", 0.0)),
            count=int(raw_stat.get("count", 0)),
            last_seen_step=(
                None
                if raw_stat.get("last_seen_step") is None
                else int(raw_stat.get("last_seen_step"))
            ),
        )
    raise TypeError("Adaptive sampler block stats must be mappings")


def _ensure_block_stats(
    state: AdaptiveSamplerState,
    block_index: int,
    granularities: Sequence[str],
) -> dict[str, AdaptiveSamplerBlockStat]:
    if block_index < 0:
        raise ValueError("block_index must be non-negative")

    validate_adaptive_sampler_state(
        state,
        granularities=granularities,
        allow_missing_stats=True,
    )
    block_stats = state.stats.setdefault(block_index, {})
    for granularity in granularities:
        validate_granularity(granularity)
        raw_stat = block_stats.get(granularity)
        if raw_stat is None:
            block_stats[granularity] = AdaptiveSamplerBlockStat()
        elif not isinstance(raw_stat, AdaptiveSamplerBlockStat):
            block_stats[granularity] = _coerce_block_stat(raw_stat)
    return block_stats


def _normalize_sampled_pattern(
    sampled_pattern: Mapping[int, str] | Sequence[str],
) -> list[tuple[int, str]]:
    if isinstance(sampled_pattern, Mapping):
        return [
            (int(block_index), str(granularity))
            for block_index, granularity in sampled_pattern.items()
        ]
    return [(block_index, str(granularity)) for block_index, granularity in enumerate(sampled_pattern)]


def _first_present_mapping_value(
    *mappings: Mapping[str, Any],
    key: str,
) -> Any:
    for mapping in mappings:
        if isinstance(mapping, Mapping) and key in mapping:
            return mapping[key]
    return None


def _mean_factor(
    state: AdaptiveSamplerState,
    stat: AdaptiveSamplerBlockStat,
    step: int,
) -> float:
    if stat.last_seen_step is None:
        return 0.0
    age = max(int(step) - int(stat.last_seen_step), 0)
    if state.decay_rate <= 0:
        return 1.0
    return math.exp(-state.decay_rate * age)


def _age_factor(
    stat: AdaptiveSamplerBlockStat,
    step: int,
) -> float:
    if stat.last_seen_step is None:
        return 1.0
    age = max(int(step) - int(stat.last_seen_step), 0)
    return 1.0 / (age + 1.0)


def _ucb_bonus(*, exploration_scale: float, count: int, step: int) -> float:
    return exploration_scale * math.sqrt(math.log(max(step, 0) + 2.0)) / (count + 1.0)


def _thompson_bonus(
    *,
    state: AdaptiveSamplerState,
    block_index: int,
    granularity: str,
    step: int,
    phase: str,
    count: int,
    exploration_scale: float,
) -> float:
    seed_material = "|".join(
        [
            state.strategy_name,
            str(block_index),
            granularity,
            str(step),
            phase,
            str(state.epoch),
            str(count),
        ]
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    rng = random.Random(seed)
    return exploration_scale * rng.gauss(0.0, 1.0) / math.sqrt(count + 1.0)


__all__ = [
    "VALID_ADAPTIVE_SAMPLER_STRATEGIES",
    "AdaptiveSamplerBlockStat",
    "AdaptiveSamplerState",
    "validate_adaptive_sampler_strategy",
    "build_adaptive_sampler_state",
    "coerce_adaptive_sampler_state",
    "normalize_adaptive_sampler_state",
    "validate_adaptive_sampler_state",
    "score_adaptive_sampler_actions",
    "select_adaptive_sampler_layer_granularities",
    "build_adaptive_reward_record",
    "update_adaptive_sampler_state",
    "summarize_adaptive_sampler_state",
    "build_adaptive_sampler_artifact_fields",
]
