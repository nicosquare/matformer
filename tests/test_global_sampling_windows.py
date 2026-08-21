import copy

import pytest
import torch

from src.training.checkpointing import (
    build_initial_continuation_state,
    load_checkpoint_state,
    save_model_checkpoint,
    validate_balanced_global_sampling_completion,
    validate_global_sampling_state,
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


def _balanced_config(tmp_path, interval, *, width_count=4, seed=42):
    labels = {
        2: ["m", "xl"],
        4: ["s", "m", "l", "xl"],
        8: [f"g{fraction}" for fraction in range(125, 1001, 125)],
    }[width_count]
    prefixes = {
        label: (index + 1) / width_count for index, label in enumerate(labels)
    }
    run_id = f"balanced-{width_count}-h{interval}-s{seed}"
    return resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        output_dir=tmp_path / run_id,
        overrides={
            "run.run_id": run_id,
            "run.seed": seed,
            "model.granularities": labels,
            "model.granularity_prefixes": prefixes,
            "model.global_sampling_schedule": "balanced_cycle",
            "model.global_sampling_interval_steps": interval,
            "training.token_budget": 2_400 * 4_096,
            "training.max_steps": 2_400,
            "training.pre_nested_warmup.enabled": False,
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


def test_balanced_cycles_are_complete_h_sized_and_do_not_merge(tmp_path):
    config = _balanced_config(tmp_path, 5)
    state = build_initial_continuation_state(config)
    actions = [_draw_and_commit(config, state, step) for step in range(1, 61)]
    labels = [action["granularities"][0] for action in actions]
    windows = [labels[index : index + 5] for index in range(0, len(labels), 5)]

    assert all(len(set(window)) == 1 for window in windows)
    cycle_windows = [windows[index : index + 4] for index in range(0, 12, 4)]
    assert all(
        sorted(window[0] for window in cycle) == sorted(config["model"]["granularities"])
        for cycle in cycle_windows
    )
    assert all(
        cycle_windows[index][-1][0] != cycle_windows[index + 1][0][0]
        for index in range(len(cycle_windows) - 1)
    )


def test_two_width_balanced_cycles_also_keep_boundaries_distinct(tmp_path):
    config = _balanced_config(tmp_path, 1, width_count=2)
    state = build_initial_continuation_state(config)
    labels = [
        _draw_and_commit(config, state, step)["granularities"][0]
        for step in range(1, 9)
    ]

    assert labels[:2] == labels[2:4] == labels[4:6] == labels[6:8]
    assert labels[1] != labels[2]


def test_balanced_cycle_order_is_repeatable_seed_sensitive_and_h_independent(tmp_path):
    def window_order(config, cycle_count=5):
        state = build_initial_continuation_state(config)
        h = config["model"]["global_sampling_interval_steps"]
        actions = [
            _draw_and_commit(config, state, step)["granularities"][0]
            for step in range(1, cycle_count * 4 * h + 1)
        ]
        return actions[::h]

    h1 = _balanced_config(tmp_path, 1, seed=7)
    h50 = _balanced_config(tmp_path, 50, seed=7)
    other_seed = _balanced_config(tmp_path, 1, seed=8)

    assert window_order(h1) == window_order(copy.deepcopy(h1))
    assert window_order(h1) == window_order(h50)
    assert window_order(h1) != window_order(other_seed)


@pytest.mark.parametrize("width_count, expected", [(8, 300), (4, 600)])
@pytest.mark.parametrize("interval", [1, 5, 25, 50])
def test_balanced_2400_step_exposure_is_exact(
    tmp_path, width_count, expected, interval
):
    config = _balanced_config(tmp_path, interval, width_count=width_count)
    state = build_initial_continuation_state(config)
    for step in range(1, 2_401):
        _draw_and_commit(config, state, step)

    sampling_state = state["global_sampling_state"]
    assert set(sampling_state["exposure_counts"].values()) == {expected}
    assert sampling_state["total_successful_updates"] == 2_400
    assert sampling_state["successful_updates_in_window"] == 0
    assert sampling_state["cycle_position"] == 0


def test_balanced_failed_attempt_does_not_advance_and_retry_is_identical(tmp_path):
    config = _balanced_config(tmp_path, 2)
    state = build_initial_continuation_state(config)
    before = copy.deepcopy(state)

    failed_action = _select_optimizer_window_action(
        config,
        list(config["model"]["granularities"]),
        torch.device("cpu"),
        optimizer_step=1,
        tokens_seen=0,
        supports_layer_granularities=False,
        distributed_context=None,
        adaptive_sampler_state=None,
        stage_name="training",
        run_state=state,
    )
    assert state == before
    retried_action = _select_optimizer_window_action(
        config,
        list(config["model"]["granularities"]),
        torch.device("cpu"),
        optimizer_step=1,
        tokens_seen=0,
        supports_layer_granularities=False,
        distributed_context=None,
        adaptive_sampler_state=None,
        stage_name="training",
        run_state=state,
    )

    assert retried_action == failed_action
    _commit_global_sampling_window_action(config, state, retried_action)
    assert state["global_sampling_state"]["total_successful_updates"] == 1
    with pytest.raises(ConfigError, match="cannot complete"):
        validate_balanced_global_sampling_completion(
            state["global_sampling_state"], config=config
        )


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


@pytest.mark.parametrize("checkpoint_step", [3, 8])
def test_balanced_checkpoint_resume_matches_uninterrupted(
    tmp_path, checkpoint_step
):
    config = _balanced_config(tmp_path, 2)
    uninterrupted = build_initial_continuation_state(config)
    for step in range(1, checkpoint_step + 1):
        _draw_and_commit(config, uninterrupted, step)
    uninterrupted.update(
        step=checkpoint_step,
        last_completed_step=checkpoint_step,
        tokens_seen=checkpoint_step * 8,
        content_tokens_seen=checkpoint_step * 8,
    )
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    checkpoint_path = tmp_path / f"balanced-step-{checkpoint_step}.pt"
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

    expected = [
        _draw_and_commit(config, uninterrupted, step)
        for step in range(checkpoint_step + 1, 25)
    ]
    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.01)
    restored = load_checkpoint_state(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        scheduler=None,
        config=config,
    )
    actual = [
        _draw_and_commit(config, restored, step)
        for step in range(checkpoint_step + 1, 25)
    ]

    assert actual == expected
    assert restored["global_sampling_state"] == uninterrupted[
        "global_sampling_state"
    ]


def test_balanced_checkpoint_rejects_schedule_h_labels_and_counters(tmp_path):
    config = _balanced_config(tmp_path, 2)
    state = build_initial_continuation_state(config)
    _draw_and_commit(config, state, 1)
    for mutation, message in (
        (lambda value: value.update(schedule="random_with_replacement"), "identity"),
        (lambda value: value.update(interval_steps=3), "interval"),
        (lambda value: value.update(granularities=["bad"]), "labels"),
        (lambda value: value.update(total_successful_updates=2), "counters"),
    ):
        malformed = copy.deepcopy(state["global_sampling_state"])
        mutation(malformed)
        with pytest.raises(ConfigError, match=message):
            validate_global_sampling_state(malformed, config=config)
