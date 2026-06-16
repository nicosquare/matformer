from __future__ import annotations

from collections.abc import Mapping

import pytest

from models.adaptive_sampler import (
    AdaptiveSamplerBlockStat,
    AdaptiveSamplerState,
    build_adaptive_sampler_state,
    summarize_adaptive_sampler_state,
)


def test_adaptive_sampler_state_defaults_and_summary_round_trip():
    state = build_adaptive_sampler_state()

    assert state == AdaptiveSamplerState(
        strategy_name="thompson",
        phase="fresh",
        step=0,
        epoch=0,
        exploration_scale=1.0,
        decay_rate=0.0,
        stats={},
    )
    assert summarize_adaptive_sampler_state(state) == {
        "strategy_name": "thompson",
        "phase": "fresh",
        "step": 0,
        "epoch": 0,
        "exploration_scale": 1.0,
        "decay_rate": 0.0,
        "stats": {},
    }


def _build_sample_state(strategy_name: str, exploration_scale: float, decay_rate: float):
    state = build_adaptive_sampler_state(
        strategy_name=strategy_name,
        phase="mid_train",
        step=12,
        epoch=3,
        exploration_scale=exploration_scale,
        decay_rate=decay_rate,
    )
    state.stats = {
        0: {
            "s": AdaptiveSamplerBlockStat(
                mean_reward=0.8,
                count=8,
                last_seen_step=11,
            ),
            "m": AdaptiveSamplerBlockStat(
                mean_reward=0.3,
                count=3,
                last_seen_step=10,
            ),
            "l": AdaptiveSamplerBlockStat(
                mean_reward=0.15,
                count=1,
                last_seen_step=7,
            ),
            "xl": AdaptiveSamplerBlockStat(
                mean_reward=0.05,
                count=0,
                last_seen_step=None,
            ),
        }
    }
    return state


@pytest.mark.xfail(
    reason="Adaptive sampler scoring helpers are implemented in T012",
    strict=False,
)
def test_thompson_scoring_prefers_the_historical_mean_when_exploration_is_zero():
    import models.adaptive_sampler as adaptive_sampler

    score_fn = getattr(adaptive_sampler, "score_adaptive_sampler_actions")
    state = _build_sample_state(
        strategy_name="thompson",
        exploration_scale=0.0,
        decay_rate=0.25,
    )

    scores = score_fn(state=state, block_index=0, step=13, phase="mid_train")

    assert isinstance(scores, Mapping)
    assert scores["s"] > scores["m"] > scores["l"] > scores["xl"]


@pytest.mark.xfail(
    reason="Adaptive sampler scoring and decay helpers are implemented in T012",
    strict=False,
)
def test_ucb_scoring_and_reward_updates_follow_the_bandit_plan():
    import models.adaptive_sampler as adaptive_sampler

    score_fn = getattr(adaptive_sampler, "score_adaptive_sampler_actions")
    reward_fn = getattr(adaptive_sampler, "build_adaptive_reward_record")
    update_fn = getattr(adaptive_sampler, "update_adaptive_sampler_state")

    state = _build_sample_state(
        strategy_name="ucb",
        exploration_scale=1.5,
        decay_rate=0.25,
    )

    scores = score_fn(state=state, block_index=0, step=13, phase="mid_train")

    assert scores["xl"] > scores["s"]
    assert scores["m"] > scores["l"]

    reward_record = reward_fn(
        previous_loss=10.0,
        current_loss=9.2,
        correction_penalty=0.2,
        reward_penalty_weight=0.5,
        phase="mid_train",
        step=13,
        epoch=3,
    )
    assert reward_record["loss_improvement"] == pytest.approx(0.8)
    assert reward_record["reward"] == pytest.approx(0.7)

    updated_state = update_fn(
        state=state,
        reward_record=reward_record,
        sampled_pattern={0: "m"},
    )
    if updated_state is None:
        updated_state = state

    assert updated_state.stats[0]["m"].count == 4
    assert updated_state.stats[0]["m"].last_seen_step == 13
    assert updated_state.stats[0]["m"].mean_reward == pytest.approx(
        (1 - 0.25) * 0.3 + 0.25 * 0.7
    )
