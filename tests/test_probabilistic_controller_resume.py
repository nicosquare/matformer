from __future__ import annotations

import copy
from collections.abc import Mapping

import pytest
import torch

from src.training.probabilistic_controller import (
    ProbabilisticControllerError,
    build_probabilistic_controller,
    restore_probabilistic_controller,
)
from src.training.checkpointing import build_initial_continuation_state
from src.training.warmup import validate_pre_nested_warmup_resume_state
from src.utils.config import ConfigError, resolve_run_config
from src.utils.reproducibility import seed_for


MANIFEST_HASHES = {
    "data_roles_manifest_hash": "parent-manifest-hash",
    "optimizer_training_manifest_hash": "training-manifest-hash",
    "controller_manifest_hash": "controller-manifest-hash",
    "ordinary_validation_manifest_hash": "validation-manifest-hash",
    "final_holdout_manifest_hash": "final-holdout-manifest-hash",
}


def _resolved_global_config(tmp_path):
    return resolve_run_config(
        "tests/fixtures/probabilistic_adaptive_global_smoke.yaml",
        output_dir=tmp_path / "probabilistic-adaptive-global-smoke-001",
        overrides={
            "training.eval_batches": 4,
            "model.adaptive_controller.decision_interval_steps": 2,
        },
    )


def _resolved_per_block_config(tmp_path):
    return resolve_run_config(
        "tests/fixtures/probabilistic_adaptive_per_block_smoke.yaml",
        output_dir=tmp_path / "probabilistic-adaptive-per-block-smoke-001",
        overrides={
            "training.eval_batches": 4,
            "model.adaptive_controller.decision_interval_steps": 2,
        },
    )


def _resolved_reset_config(tmp_path, *, interval_steps=12):
    return resolve_run_config(
        "tests/fixtures/probabilistic_adaptive_global_smoke.yaml",
        output_dir=tmp_path / "probabilistic-adaptive-global-smoke-001",
        overrides={
            "training.eval_batches": 4,
            "model.adaptive_controller.decision_interval_steps": 2,
            "model.adaptive_controller.process_noise_covariance": 0.0,
            "model.adaptive_controller.reset.enabled": True,
            "model.adaptive_controller.reset.interval_steps": interval_steps,
        },
    )


def _build_controller(config):
    return build_probabilistic_controller(
        controller_config=config["model"]["adaptive_controller"],
        sampling_seed=seed_for(config, "posterior_sampling"),
        manifest_hashes=MANIFEST_HASHES,
    )


def _initialize_controller(controller):
    return controller.initialize_boundary(
        boundary_step=0,
        controller_objective=10.0,
        ordered_component_losses=[9.0, 10.0, 11.0],
        evaluation_target_tokens=384,
    )


def _assert_nested_exact(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        assert torch.is_tensor(left) and torch.is_tensor(right)
        assert torch.equal(left, right)
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        assert isinstance(left, Mapping) and isinstance(right, Mapping)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_exact(left[key], right[key])
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_exact(left_item, right_item)
        return
    assert left == right


def _assert_saved_resume_identity(saved_state, restored_state):
    assert restored_state["manifest_hashes"] == saved_state["manifest_hashes"]
    _assert_nested_exact(restored_state["feature_schema"], saved_state["feature_schema"])
    _assert_nested_exact(restored_state["window"], saved_state["window"])
    assert (
        restored_state["sampling"]["sample_count"]
        == saved_state["sampling"]["sample_count"]
    )
    assert torch.equal(
        restored_state["sampling"]["generator_state"],
        saved_state["sampling"]["generator_state"],
    )


def _assert_completed_events_close(fresh_event, resumed_event):
    assert fresh_event["event_type"] == resumed_event["event_type"] == "completed_window"
    for key in (
        "window_index",
        "boundary_step_start",
        "boundary_step_end",
        "completed_optimizer_steps",
        "action",
        "ordered_component_losses",
        "evaluation_target_tokens",
    ):
        _assert_nested_exact(fresh_event[key], resumed_event[key])
    for key in (
        "pre_window_objective",
        "post_window_objective",
        "reward",
        "predicted_reward",
        "prediction_error",
    ):
        assert resumed_event[key] == pytest.approx(
            fresh_event[key],
            rel=1e-6,
            abs=1e-8,
        )
    for key in (
        "predictive_mean",
        "predictive_covariance",
        "gain_vector",
        "posterior_mean",
        "posterior_covariance",
    ):
        torch.testing.assert_close(
            resumed_event[key],
            fresh_event[key],
            rtol=1e-6,
            atol=1e-8,
        )


def _complete_window(controller, *, boundary_step, objective=9.0):
    while controller.state_dict()["window"]["phase"] == "active_window":
        controller.record_successful_optimizer_step()
    return controller.complete_boundary(
        boundary_step=boundary_step,
        controller_objective=objective,
        ordered_component_losses=[objective - 1.0, objective, objective + 1.0],
        evaluation_target_tokens=384,
        training_will_continue=True,
    )


def test_initial_boundary_records_objective_before_reward_and_starts_first_window(
    tmp_path,
):
    config = _resolved_global_config(tmp_path)
    controller = _build_controller(config)
    prior_state = copy.deepcopy(controller.state_dict())

    event = _initialize_controller(controller)
    state = controller.state_dict()

    assert event["event_type"] == "initial_boundary"
    assert event["boundary_step"] == 0
    assert event["controller_objective"] == pytest.approx(10.0)
    assert event["ordered_component_losses"] == [9.0, 10.0, 11.0]
    assert event["evaluation_target_tokens"] == 384
    assert "reward" not in event
    assert "prediction_error" not in event
    assert state["window"]["phase"] == "active_window"
    assert state["window"]["window_index"] == 0
    assert state["window"]["boundary_step"] == 0
    assert state["window"]["completed_optimizer_steps"] == 0
    assert state["window"]["pre_window_objective"] == pytest.approx(10.0)
    _assert_nested_exact(
        state["window"]["current_action"],
        event["selected_action"],
    )
    assert state["sampling"]["sample_count"] == 1
    _assert_nested_exact(
        state["belief"]["posterior_mean"],
        prior_state["belief"]["posterior_mean"],
    )
    _assert_nested_exact(
        state["belief"]["posterior_covariance"],
        prior_state["belief"]["posterior_covariance"],
    )


def test_balanced_warmup_resume_state_validates_mid_window_schedule_identity(tmp_path):
    config = resolve_run_config(
        "tests/fixtures/probabilistic_adaptive_global_smoke.yaml",
        output_dir=tmp_path / "probabilistic-adaptive-global-smoke-001",
        overrides={
            "training.pre_nested_warmup.enabled": True,
            "training.pre_nested_warmup.duration": 12,
            "training.pre_nested_warmup.unit": "steps",
            "training.pre_nested_warmup.policy": "balanced_global",
            "training.pre_nested_warmup.action_interval_steps": 2,
        },
    )
    state = build_initial_continuation_state(config)["pre_nested_warmup_state"]
    first_action = state["schedule"][0]
    state.update(
        schedule_initialized=True,
        completed_steps=3,
        current_window_index=1,
        current_window_offset=1,
        per_granularity_counts={
            label: int(label == first_action)
            for label in config["model"]["granularities"]
        },
    )

    restored = validate_pre_nested_warmup_resume_state(
        config,
        state,
        last_completed_step=3,
    )
    assert restored == state

    incompatible = copy.deepcopy(state)
    incompatible["schedule_hash"] = "different-schedule"
    with pytest.raises(ConfigError, match="schedule_hash does not match"):
        validate_pre_nested_warmup_resume_state(
            config,
            incompatible,
            last_completed_step=3,
        )


@pytest.mark.parametrize(
    "checkpoint_progress, expected_phase",
    [
        pytest.param(1, "active_window", id="inside-window"),
        pytest.param(2, "boundary_evaluation_pending", id="exact-boundary"),
    ],
)
def test_fresh_and_resumed_controller_match_from_inside_window_and_exact_boundary(
    tmp_path,
    checkpoint_progress,
    expected_phase,
):
    config = _resolved_global_config(tmp_path)
    uninterrupted = _build_controller(config)
    _initialize_controller(uninterrupted)
    for _ in range(checkpoint_progress):
        uninterrupted.record_successful_optimizer_step()

    saved_state = copy.deepcopy(uninterrupted.state_dict())
    assert saved_state["window"]["phase"] == expected_phase
    resumed = restore_probabilistic_controller(
        saved_state,
        controller_config=config["model"]["adaptive_controller"],
        sampling_seed=seed_for(config, "posterior_sampling"),
        expected_manifest_hashes=MANIFEST_HASHES,
        source_checkpoint=tmp_path / "checkpoints" / "latest.pt",
        logger=lambda _message: None,
    )
    _assert_saved_resume_identity(saved_state, resumed.state_dict())

    if checkpoint_progress < 2:
        uninterrupted.record_successful_optimizer_step()
        resumed.record_successful_optimizer_step()

    uninterrupted_event = uninterrupted.complete_boundary(
        boundary_step=2,
        controller_objective=8.0,
        ordered_component_losses=[7.0, 8.0, 9.0],
        evaluation_target_tokens=384,
        training_will_continue=True,
    )
    resumed_event = resumed.complete_boundary(
        boundary_step=2,
        controller_objective=8.0,
        ordered_component_losses=[7.0, 8.0, 9.0],
        evaluation_target_tokens=384,
        training_will_continue=True,
    )

    _assert_completed_events_close(uninterrupted_event, resumed_event)
    uninterrupted_state = uninterrupted.state_dict()
    resumed_state = resumed.state_dict()
    assert uninterrupted_state["manifest_hashes"] == resumed_state[
        "manifest_hashes"
    ]
    _assert_nested_exact(
        uninterrupted_state["window"],
        resumed_state["window"],
    )
    assert uninterrupted_state["sampling"]["sample_count"] == 2
    assert resumed_state["sampling"]["sample_count"] == 2
    assert torch.equal(
        uninterrupted_state["sampling"]["generator_state"],
        resumed_state["sampling"]["generator_state"],
    )
    torch.testing.assert_close(
        resumed_state["belief"]["posterior_mean"],
        uninterrupted_state["belief"]["posterior_mean"],
        rtol=1e-6,
        atol=1e-8,
    )
    torch.testing.assert_close(
        resumed_state["belief"]["posterior_covariance"],
        uninterrupted_state["belief"]["posterior_covariance"],
        rtol=1e-6,
        atol=1e-8,
    )
    assert resumed_state["resume"]["resume_count"] == 1
    assert resumed_state["resume"]["source_checkpoint"] == str(
        tmp_path / "checkpoints" / "latest.pt"
    )


@pytest.mark.parametrize(
    "completed_windows, expected_source",
    [
        pytest.param(0, "forced_acquisition", id="inside-acquisition-window"),
        pytest.param(3, "thompson", id="inside-thompson-window"),
        pytest.param(6, "forced_acquisition", id="exact-reset-boundary"),
    ],
)
def test_reset_resume_matches_inside_acquisition_thompson_and_reset_boundaries(
    tmp_path,
    completed_windows,
    expected_source,
):
    config = _resolved_reset_config(tmp_path)
    uninterrupted = _build_controller(config)
    _initialize_controller(uninterrupted)
    for window_index in range(completed_windows):
        _complete_window(
            uninterrupted,
            boundary_step=2 + 2 * window_index,
            objective=9.0 - 0.1 * window_index,
        )

    if completed_windows != 6:
        uninterrupted.record_successful_optimizer_step()
    saved_state = copy.deepcopy(uninterrupted.state_dict())
    assert saved_state["window"]["selection_source"] == expected_source
    resumed = restore_probabilistic_controller(
        saved_state,
        controller_config=config["model"]["adaptive_controller"],
        sampling_seed=seed_for(config, "posterior_sampling"),
        expected_manifest_hashes=MANIFEST_HASHES,
        source_checkpoint=tmp_path / "checkpoints" / "latest.pt",
        logger=lambda _message: None,
    )
    _assert_saved_resume_identity(saved_state, resumed.state_dict())
    _assert_nested_exact(saved_state["reset"], resumed.state_dict()["reset"])

    if completed_windows != 6:
        uninterrupted.record_successful_optimizer_step()
        resumed.record_successful_optimizer_step()
        fresh_event = uninterrupted.complete_boundary(
            boundary_step=2 + 2 * completed_windows,
            controller_objective=8.0,
            ordered_component_losses=[7.0, 8.0, 9.0],
            evaluation_target_tokens=384,
            training_will_continue=True,
        )
        resumed_event = resumed.complete_boundary(
            boundary_step=2 + 2 * completed_windows,
            controller_objective=8.0,
            ordered_component_losses=[7.0, 8.0, 9.0],
            evaluation_target_tokens=384,
            training_will_continue=True,
        )
        _assert_completed_events_close(fresh_event, resumed_event)
    else:
        for window_index in range(3):
            fresh_event = _complete_window(
                uninterrupted,
                boundary_step=14 + 2 * window_index,
                objective=8.0 - 0.1 * window_index,
            )
            resumed_event = _complete_window(
                resumed,
                boundary_step=14 + 2 * window_index,
                objective=8.0 - 0.1 * window_index,
            )
            _assert_completed_events_close(fresh_event, resumed_event)

    _assert_nested_exact(
        uninterrupted.state_dict()["window"],
        resumed.state_dict()["window"],
    )
    _assert_nested_exact(
        uninterrupted.state_dict()["reset"],
        resumed.state_dict()["reset"],
    )
    assert torch.equal(
        uninterrupted.state_dict()["sampling"]["generator_state"],
        resumed.state_dict()["sampling"]["generator_state"],
    )


def test_old_controller_schema_is_allowed_only_when_reset_is_disabled(tmp_path):
    disabled_config = _resolved_global_config(tmp_path)
    disabled = _build_controller(disabled_config)
    _initialize_controller(disabled)
    legacy_state = copy.deepcopy(disabled.state_dict())
    legacy_state["schema_version"] = 1
    legacy_state.pop("reset")
    legacy_state["window"].pop("selection_source")

    restored = restore_probabilistic_controller(
        legacy_state,
        controller_config=disabled_config["model"]["adaptive_controller"],
        sampling_seed=seed_for(disabled_config, "posterior_sampling"),
        expected_manifest_hashes=MANIFEST_HASHES,
        source_checkpoint=tmp_path / "checkpoints" / "old.pt",
        logger=lambda _message: None,
    )
    assert restored.state_dict()["schema_version"] == 2
    assert restored.state_dict()["reset"]["enabled"] is False

    reset_config = _resolved_reset_config(tmp_path)
    with pytest.raises(
        ProbabilisticControllerError,
        match="reset-enabled continuation requires complete reset state",
    ):
        restore_probabilistic_controller(
            legacy_state,
            controller_config=reset_config["model"]["adaptive_controller"],
            sampling_seed=seed_for(reset_config, "posterior_sampling"),
            expected_manifest_hashes=MANIFEST_HASHES,
            source_checkpoint=tmp_path / "checkpoints" / "old.pt",
            logger=lambda _message: None,
        )


def test_reset_resume_rejects_incompatible_interval_without_mutation(tmp_path):
    config = _resolved_reset_config(tmp_path, interval_steps=12)
    controller = _build_controller(config)
    _initialize_controller(controller)
    saved_state = copy.deepcopy(controller.state_dict())
    incompatible = _resolved_reset_config(tmp_path, interval_steps=18)

    with pytest.raises(ProbabilisticControllerError, match="reset contract mismatch"):
        restore_probabilistic_controller(
            saved_state,
            controller_config=incompatible["model"]["adaptive_controller"],
            sampling_seed=seed_for(incompatible, "posterior_sampling"),
            expected_manifest_hashes=MANIFEST_HASHES,
            source_checkpoint=tmp_path / "checkpoints" / "latest.pt",
            logger=lambda _message: None,
        )


def test_additive_per_block_resume_preserves_fixed_complete_profile_and_provenance(
    tmp_path,
):
    config = _resolved_per_block_config(tmp_path)
    uninterrupted = _build_controller(config)
    initial_event = _initialize_controller(uninterrupted)
    initial_state = uninterrupted.state_dict()
    initial_profile = initial_event["selected_action"]["block_granularities"]

    assert len(initial_profile) == config["model"]["num_layers"]
    assert initial_state["scope"] == "per_block"
    assert initial_state["feature_schema"]["encoding"] == (
        "intercept_plus_per_block_sum_to_zero_contrasts"
    )
    assert initial_state["feature_schema"]["dimension"] == (
        1
        + config["model"]["num_layers"]
        * (len(config["model"]["granularities"]) - 1)
    )

    uninterrupted.record_successful_optimizer_step()
    assert uninterrupted.current_action["block_granularities"] == initial_profile
    saved_state = copy.deepcopy(uninterrupted.state_dict())
    checkpoint_path = tmp_path / "checkpoints" / "latest.pt"
    resumed = restore_probabilistic_controller(
        saved_state,
        controller_config=config["model"]["adaptive_controller"],
        sampling_seed=seed_for(config, "posterior_sampling"),
        expected_manifest_hashes=MANIFEST_HASHES,
        source_checkpoint=checkpoint_path,
        logger=lambda _message: None,
    )

    _assert_saved_resume_identity(saved_state, resumed.state_dict())
    assert resumed.current_action["block_granularities"] == initial_profile
    uninterrupted.record_successful_optimizer_step()
    resumed.record_successful_optimizer_step()
    fresh_event = uninterrupted.complete_boundary(
        boundary_step=2,
        controller_objective=8.0,
        ordered_component_losses=[7.0, 8.0, 9.0],
        evaluation_target_tokens=384,
        training_will_continue=True,
    )
    resumed_event = resumed.complete_boundary(
        boundary_step=2,
        controller_objective=8.0,
        ordered_component_losses=[7.0, 8.0, 9.0],
        evaluation_target_tokens=384,
        training_will_continue=True,
    )

    _assert_completed_events_close(fresh_event, resumed_event)
    assert fresh_event["reward"] == pytest.approx((10.0 - 8.0) / 2)
    assert fresh_event["action"]["block_granularities"] == initial_profile
    assert fresh_event["next_action"]["block_granularities"] == resumed_event[
        "next_action"
    ]["block_granularities"]
    final_state = resumed.state_dict()
    assert final_state["resume"]["source_checkpoint"] == str(checkpoint_path)
    assert "enumerated_profile" not in repr(final_state)
    assert "profile_table" not in repr(final_state)


def test_terminal_incomplete_window_emits_no_observation_update_or_unused_sample(
    tmp_path,
):
    config = _resolved_global_config(tmp_path)
    controller = _build_controller(config)
    _initialize_controller(controller)
    controller.record_successful_optimizer_step()
    state_before_termination = copy.deepcopy(controller.state_dict())

    event = controller.finish_training()
    terminal_state = controller.state_dict()

    assert event["event_type"] == "terminal_incomplete"
    assert event["observation_emitted"] is False
    assert event["completed_optimizer_steps"] == 1
    assert event["decision_interval_steps"] == 2
    assert "reward" not in event
    assert terminal_state["window"]["phase"] == "terminal_incomplete"
    assert terminal_state["window"]["completed_optimizer_steps"] == 1
    _assert_nested_exact(
        terminal_state["belief"],
        state_before_termination["belief"],
    )
    assert (
        terminal_state["sampling"]["sample_count"]
        == state_before_termination["sampling"]["sample_count"]
        == 1
    )
    assert torch.equal(
        terminal_state["sampling"]["generator_state"],
        state_before_termination["sampling"]["generator_state"],
    )


def test_resume_rejects_missing_bayesian_controller_state(tmp_path):
    config = _resolved_global_config(tmp_path)

    with pytest.raises(
        ProbabilisticControllerError,
        match="(?i)missing.*Bayesian controller state",
    ):
        restore_probabilistic_controller(
            None,
            controller_config=config["model"]["adaptive_controller"],
            sampling_seed=seed_for(config, "posterior_sampling"),
            expected_manifest_hashes=MANIFEST_HASHES,
            source_checkpoint=tmp_path / "checkpoints" / "legacy.pt",
            logger=lambda _message: None,
        )


@pytest.mark.parametrize(
    "mutation, expected_message",
    [
        pytest.param(
            lambda state: state.update(method_version=state["method_version"] + 1),
            "method version",
            id="method-version",
        ),
        pytest.param(
            lambda state: state.update(scope="per_block"),
            "scope",
            id="scope",
        ),
        pytest.param(
            lambda state: state["feature_schema"].update(schema_hash="wrong"),
            "feature schema",
            id="feature-schema",
        ),
        pytest.param(
            lambda state: state["belief"].update(posterior_mean=[0.0]),
            "posterior.*dimension",
            id="posterior-dimension",
        ),
        pytest.param(
            lambda state: state["sampling"].pop("generator_state"),
            "sampling.*generator state",
            id="sampling-state",
        ),
    ],
)
def test_resume_rejects_incompatible_bayesian_state(
    tmp_path,
    mutation,
    expected_message,
):
    config = _resolved_global_config(tmp_path)
    controller = _build_controller(config)
    _initialize_controller(controller)
    incompatible_state = copy.deepcopy(controller.state_dict())
    mutation(incompatible_state)

    with pytest.raises(ProbabilisticControllerError, match=expected_message):
        restore_probabilistic_controller(
            incompatible_state,
            controller_config=config["model"]["adaptive_controller"],
            sampling_seed=seed_for(config, "posterior_sampling"),
            expected_manifest_hashes=MANIFEST_HASHES,
            source_checkpoint=tmp_path / "checkpoints" / "latest.pt",
            logger=lambda _message: None,
        )


def test_resume_rejects_any_data_role_manifest_mismatch(tmp_path):
    config = _resolved_global_config(tmp_path)
    controller = _build_controller(config)
    _initialize_controller(controller)
    state = copy.deepcopy(controller.state_dict())
    state["manifest_hashes"]["controller_manifest_hash"] = "different-controller"

    with pytest.raises(
        ProbabilisticControllerError,
        match="controller_manifest_hash.*mismatch",
    ):
        restore_probabilistic_controller(
            state,
            controller_config=config["model"]["adaptive_controller"],
            sampling_seed=seed_for(config, "posterior_sampling"),
            expected_manifest_hashes=MANIFEST_HASHES,
            source_checkpoint=tmp_path / "checkpoints" / "latest.pt",
            logger=lambda _message: None,
        )


def test_resume_emits_one_concise_log_with_source_phase_window_progress_and_action(
    tmp_path,
):
    config = _resolved_global_config(tmp_path)
    controller = _build_controller(config)
    _initialize_controller(controller)
    controller.record_successful_optimizer_step()
    saved_state = copy.deepcopy(controller.state_dict())
    messages = []
    checkpoint_path = tmp_path / "checkpoints" / "latest.pt"

    restored = restore_probabilistic_controller(
        saved_state,
        controller_config=config["model"]["adaptive_controller"],
        sampling_seed=seed_for(config, "posterior_sampling"),
        expected_manifest_hashes=MANIFEST_HASHES,
        source_checkpoint=checkpoint_path,
        logger=messages.append,
    )

    assert restored.state_dict()["resume"]["resume_count"] == 1
    assert len(messages) == 1
    message = messages[0]
    assert "probabilistic-controller-resume" in message
    assert f"source_checkpoint={checkpoint_path}" in message
    assert "restored_phase=active_window" in message
    assert "window_index=0" in message
    assert "progress=1/2" in message
    assert "current_action=" in message
    assert "posterior_mean" not in message
    assert "posterior_covariance" not in message
