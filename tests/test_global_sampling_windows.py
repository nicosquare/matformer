import copy

import pytest
import torch

from src.training.checkpointing import (
    build_initial_continuation_state,
    load_checkpoint_state,
    save_model_checkpoint,
)
from src.training.steps import (
    _commit_global_sampling_window_action,
    _select_optimizer_window_action,
    select_training_granularities,
)
from src.utils.config import ConfigError, resolve_run_config
from src.utils.reproducibility import (
    capture_rng_state,
    restore_rng_state,
    seed_training_randomness,
)


def _config(tmp_path, interval):
    return resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        output_dir=tmp_path / "dmodel256-pilot-comparison-001",
        overrides={
            "model.granularity_sampling_mode": "global",
            "model.global_sampling_interval_steps": interval,
        },
    )


def _draw_and_commit(config, state, step):
    action = _select_optimizer_window_action(
        config,
        list(config["model"]["granularities"]),
        torch.device("cpu"),
        optimizer_step=step,
        tokens_seen=step - 1,
        supports_layer_granularities=False,
        distributed_context=None,
        adaptive_sampler_state=None,
        stage_name="training",
        run_state=state,
    )
    _commit_global_sampling_window_action(config, state, action)
    return action


def test_h1_matches_existing_per_update_uniform_sequence(tmp_path):
    config = _config(tmp_path, 1)
    granularities = list(config["model"]["granularities"])

    seed_training_randomness(config)
    expected = [
        select_training_granularities(
            config, granularities, torch.device("cpu")
        )[0]
        for _ in range(20)
    ]

    seed_training_randomness(config)
    state = build_initial_continuation_state(config)
    actual = [
        _draw_and_commit(config, state, step)["granularities"][0]
        for step in range(1, 21)
    ]

    assert actual == expected
    assert state["global_sampling_state"]["total_successful_updates"] == 20


def test_h25_holds_steps_1_through_25_and_starts_window_1_at_step_26(tmp_path):
    config = _config(tmp_path, 25)
    seed_training_randomness(config)
    state = build_initial_continuation_state(config)

    actions = [_draw_and_commit(config, state, step) for step in range(1, 27)]

    assert len({action["granularities"][0] for action in actions[:25]}) == 1
    assert [action["global_sampling_window_index"] for action in actions[:25]] == [
        0
    ] * 25
    assert [
        action["global_sampling_window_progress"] for action in actions[:25]
    ] == list(range(1, 26))
    assert actions[25]["global_sampling_window_index"] == 1
    assert actions[25]["global_sampling_window_progress"] == 1
    assert sum(state["global_sampling_state"]["exposure_counts"].values()) == 26


def test_failed_attempt_restores_window_progress_and_rng(tmp_path):
    config = _config(tmp_path, 2)
    seed_training_randomness(config)
    state = build_initial_continuation_state(config)
    _draw_and_commit(config, state, 1)
    _draw_and_commit(config, state, 2)
    state_snapshot = copy.deepcopy(state)
    rng_snapshot = capture_rng_state()

    failed_action = _select_optimizer_window_action(
        config,
        list(config["model"]["granularities"]),
        torch.device("cpu"),
        optimizer_step=3,
        tokens_seen=2,
        supports_layer_granularities=False,
        distributed_context=None,
        adaptive_sampler_state=None,
        stage_name="training",
        run_state=state,
    )
    assert state["global_sampling_state"]["window_index"] == 1
    assert state["global_sampling_state"]["total_successful_updates"] == 2

    # The training transaction restores both snapshots after the attempt fails.
    state.clear()
    state.update(copy.deepcopy(state_snapshot))
    restore_rng_state(rng_snapshot)
    retried_action = _select_optimizer_window_action(
        config,
        list(config["model"]["granularities"]),
        torch.device("cpu"),
        optimizer_step=3,
        tokens_seen=2,
        supports_layer_granularities=False,
        distributed_context=None,
        adaptive_sampler_state=None,
        stage_name="training",
        run_state=state,
    )

    assert retried_action == failed_action
    _commit_global_sampling_window_action(config, state, retried_action)
    assert state["global_sampling_state"]["total_successful_updates"] == 3


@pytest.mark.parametrize("checkpoint_step", [12, 25])
def test_resume_inside_and_at_h25_boundary_preserves_full_state(
    tmp_path,
    checkpoint_step,
):
    config = _config(tmp_path, 25)
    seed_training_randomness(config)
    uninterrupted = build_initial_continuation_state(config)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def update_model(action):
        optimizer.zero_grad(set_to_none=True)
        selected_index = config["model"]["granularities"].index(
            action["granularities"][0]
        )
        loss = model(torch.tensor([[1.0, -0.5]])).sum() * (selected_index + 1)
        loss.backward()
        optimizer.step()

    for step in range(1, checkpoint_step + 1):
        update_model(_draw_and_commit(config, uninterrupted, step))
    uninterrupted.update(
        step=checkpoint_step,
        last_completed_step=checkpoint_step,
        tokens_seen=checkpoint_step * 8,
        content_tokens_seen=checkpoint_step * 8,
    )
    checkpoint_path = tmp_path / f"uniform-window-step-{checkpoint_step}.pt"
    save_model_checkpoint(
        config,
        model,
        optimizer,
        scheduler=None,
        output_path=checkpoint_path,
        checkpoint_fields={
            "checkpoint_status": "latest",
            "checkpoint_metric": None,
            "checkpoint_metric_value": None,
            "checkpoint_selection_step": None,
        },
        run_state=uninterrupted,
    )

    suffix = []
    for step in range(checkpoint_step + 1, 32):
        action = _draw_and_commit(config, uninterrupted, step)
        update_model(action)
        suffix.append(action)
    final_rng = capture_rng_state()
    final_parameters = [parameter.detach().clone() for parameter in model.parameters()]

    resumed_model = torch.nn.Linear(2, 1)
    resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.01)
    resumed = load_checkpoint_state(
        checkpoint_path,
        resumed_model,
        resumed_optimizer,
        scheduler=None,
        config=config,
    )

    def update_resumed_model(action):
        resumed_optimizer.zero_grad(set_to_none=True)
        selected_index = config["model"]["granularities"].index(
            action["granularities"][0]
        )
        loss = resumed_model(torch.tensor([[1.0, -0.5]])).sum() * (
            selected_index + 1
        )
        loss.backward()
        resumed_optimizer.step()

    resumed_suffix = []
    for step in range(checkpoint_step + 1, 32):
        action = _draw_and_commit(config, resumed, step)
        update_resumed_model(action)
        resumed_suffix.append(action)

    assert resumed_suffix == suffix
    assert resumed["global_sampling_state"] == uninterrupted["global_sampling_state"]
    resumed_rng = capture_rng_state()
    assert resumed_rng["dedicated"] == final_rng["dedicated"]
    assert torch.equal(resumed_rng["torch_cpu"], final_rng["torch_cpu"])
    for resumed_parameter, final_parameter in zip(
        resumed_model.parameters(), final_parameters, strict=True
    ):
        assert torch.equal(resumed_parameter, final_parameter)


def test_partial_final_window_and_single_granularity(tmp_path):
    config = _config(tmp_path, 4)
    config["model"]["granularities"] = ["xl"]
    seed_training_randomness(config)
    state = build_initial_continuation_state(config)

    actions = [_draw_and_commit(config, state, step) for step in range(1, 4)]

    assert [action["granularities"] for action in actions] == [["xl"]] * 3
    assert state["global_sampling_state"] == {
        "schema_version": 1,
        "interval_steps": 4,
        "held_granularity": "xl",
        "window_index": 0,
        "successful_updates_in_window": 3,
        "total_successful_updates": 3,
        "exposure_counts": {"xl": 3},
    }


def test_checkpoint_persists_and_validates_uniform_window_state(tmp_path):
    config = _config(tmp_path, 2)
    seed_training_randomness(config)
    state = build_initial_continuation_state(config)
    for step in range(1, 4):
        _draw_and_commit(config, state, step)
    state.update(
        step=3,
        last_completed_step=3,
        tokens_seen=24,
        content_tokens_seen=24,
    )
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    checkpoint_path = tmp_path / "uniform-window.pt"

    save_model_checkpoint(
        config,
        model,
        optimizer,
        scheduler=None,
        output_path=checkpoint_path,
        checkpoint_fields={
            "checkpoint_status": "latest",
            "checkpoint_metric": None,
            "checkpoint_metric_value": None,
            "checkpoint_selection_step": None,
        },
        run_state=state,
    )

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.01)
    restored = load_checkpoint_state(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        scheduler=None,
        config=config,
    )
    assert restored["global_sampling_state"] == state["global_sampling_state"]

    changed = copy.deepcopy(config)
    changed["model"]["global_sampling_interval_steps"] = 3
    changed_model = torch.nn.Linear(2, 1)
    changed_optimizer = torch.optim.SGD(changed_model.parameters(), lr=0.01)
    with pytest.raises(ConfigError, match="global sampling interval"):
        load_checkpoint_state(
            checkpoint_path,
            changed_model,
            changed_optimizer,
            scheduler=None,
            config=changed,
        )
