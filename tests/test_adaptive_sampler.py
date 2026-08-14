from __future__ import annotations

import pytest

from src.training.checkpointing import build_initial_continuation_state
from src.models.adaptive_sampler import (
    VALID_ADAPTIVE_SAMPLER_STRATEGIES,
    AdaptiveSamplerBlockStat,
    AdaptiveSamplerState,
    build_adaptive_sampler_state,
    coerce_adaptive_sampler_state,
    summarize_adaptive_sampler_state,
)


def test_ucb_adaptive_sampler_state_defaults_and_summary_round_trip():
    state = build_adaptive_sampler_state()

    assert state == AdaptiveSamplerState(
        strategy_name="ucb",
        phase="fresh",
        step=0,
        epoch=0,
        exploration_scale=1.0,
        decay_rate=0.0,
        stats={},
    )
    assert summarize_adaptive_sampler_state(state) == {
        "strategy_name": "ucb",
        "phase": "fresh",
        "step": 0,
        "epoch": 0,
        "exploration_scale": 1.0,
        "decay_rate": 0.0,
        "stats": {},
    }


def test_legacy_heuristic_thompson_is_not_selectable():
    assert VALID_ADAPTIVE_SAMPLER_STRATEGIES == ("ucb",)

    with pytest.raises(ValueError, match="strategy_name.*ucb"):
        build_adaptive_sampler_state(strategy_name="thompson")

    with pytest.raises(ValueError, match="strategy_name.*ucb"):
        coerce_adaptive_sampler_state(
            {
                "strategy_name": "thompson",
                "phase": "mid_train",
                "step": 12,
                "epoch": 3,
                "stats": {},
            }
        )


@pytest.mark.parametrize(
    "sampling_mode",
    ["global", "per_block", "nested-all", "standalone"],
)
def test_non_ucb_sampling_modes_do_not_create_adaptive_or_panelgrad_state(
    tmp_path,
    sampling_mode,
):
    state = build_initial_continuation_state(
        {
            "run": {
                "output_dir": str(tmp_path / sampling_mode),
                "resolved_run_mode": sampling_mode,
            },
            "model": {"granularity_sampling_mode": sampling_mode},
            "training": {},
            "evaluation": {},
        }
    )

    assert state["adaptive_sampler_state"] is None
    assert state["probabilistic_controller_state"] is None
    assert state["panelgrad_state"] is None


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


def test_ucb_scoring_and_reward_updates_follow_the_bandit_plan():
    import src.models.adaptive_sampler as adaptive_sampler

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

    restored_state = coerce_adaptive_sampler_state(
        summarize_adaptive_sampler_state(updated_state)
    )
    assert restored_state == updated_state
    assert restored_state.strategy_name == "ucb"
