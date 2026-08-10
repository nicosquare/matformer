from __future__ import annotations

import math
from itertools import product

import pytest
import torch

import src.training.probabilistic_controller as probabilistic_controller
from src.training.probabilistic_controller import (
    ProbabilisticControllerError,
    build_global_feature_schema,
    build_probabilistic_controller,
    condition_gaussian_belief,
    encode_global_action,
    predict_gaussian_belief,
    sample_gaussian_coefficients,
    select_global_action,
    validate_gaussian_belief,
)
from src.utils.config import resolve_run_config
from src.utils.reproducibility import seed_for


FLOAT64 = torch.float64


def _tensor(values):
    return torch.tensor(values, dtype=FLOAT64)


def test_global_feature_schema_uses_ordered_orthonormal_sum_to_zero_contrasts():
    schema = build_global_feature_schema(["micro", "medium", "full"])

    expected_basis = _tensor(
        [
            [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0)],
            [-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0)],
            [0.0, -2.0 / math.sqrt(6.0)],
        ]
    )
    basis = _tensor(schema["contrast_basis"])

    assert schema["scope"] == "global"
    assert schema["encoding"] == "intercept_plus_sum_to_zero_contrasts"
    assert schema["ordered_granularities"] == ["micro", "medium", "full"]
    assert schema["dimension"] == 3
    assert schema["coefficient_names"][0] == "intercept"
    assert len(schema["coefficient_names"]) == schema["dimension"]
    assert len(schema["schema_hash"]) == 64
    torch.testing.assert_close(basis, expected_basis, rtol=0.0, atol=1e-15)
    torch.testing.assert_close(
        basis.sum(dim=0),
        torch.zeros(2, dtype=FLOAT64),
        rtol=0.0,
        atol=1e-15,
    )
    torch.testing.assert_close(
        basis.T @ basis,
        torch.eye(2, dtype=FLOAT64),
        rtol=0.0,
        atol=1e-15,
    )

    encoded = torch.stack(
        [encode_global_action(schema, label) for label in schema["ordered_granularities"]]
    )
    torch.testing.assert_close(encoded[:, 0], torch.ones(3, dtype=FLOAT64))
    torch.testing.assert_close(encoded[:, 1:], expected_basis, rtol=0.0, atol=1e-15)


def test_global_feature_schema_is_stable_and_rejects_invalid_labels():
    first = build_global_feature_schema(["tiny", "wide", "maximal"])
    second = build_global_feature_schema(["tiny", "wide", "maximal"])

    assert first == second
    with pytest.raises(ProbabilisticControllerError, match="(?i)unique"):
        build_global_feature_schema(["tiny", "tiny"])
    with pytest.raises(ProbabilisticControllerError, match="(?i)nonempty"):
        build_global_feature_schema([])
    with pytest.raises(ProbabilisticControllerError, match="unknown.*missing"):
        encode_global_action(first, "missing")


@pytest.mark.parametrize(
    "process_noise, expected_covariance",
    [
        (
            [[0.0, 0.0], [0.0, 0.0]],
            [[2.0, 0.25], [0.25, 1.0]],
        ),
        (
            [[0.5, 0.1], [0.1, 0.25]],
            [[2.5, 0.35], [0.35, 1.25]],
        ),
    ],
)
def test_identity_prediction_preserves_mean_and_adds_process_noise(
    process_noise,
    expected_covariance,
):
    posterior_mean = _tensor([1.25, -0.5])
    posterior_covariance = _tensor([[2.0, 0.25], [0.25, 1.0]])

    prediction = predict_gaussian_belief(
        posterior_mean=posterior_mean,
        posterior_covariance=posterior_covariance,
        process_noise_covariance=_tensor(process_noise),
    )

    assert prediction["transition_model"] == "identity"
    assert prediction["predictive_mean"].dtype == FLOAT64
    assert prediction["predictive_covariance"].dtype == FLOAT64
    assert prediction["predictive_mean"].device.type == "cpu"
    torch.testing.assert_close(prediction["predictive_mean"], posterior_mean)
    torch.testing.assert_close(
        prediction["predictive_covariance"],
        _tensor(expected_covariance),
    )


def test_gaussian_conditioning_matches_closed_form_mean_covariance_and_gain():
    predictive_mean = _tensor([0.5, -0.25])
    predictive_covariance = _tensor([[2.0, 0.5], [0.5, 1.5]])
    feature = _tensor([1.0, 2.0])
    reward = 3.0
    observation_noise_variance = 0.5

    result = condition_gaussian_belief(
        predictive_mean=predictive_mean,
        predictive_covariance=predictive_covariance,
        feature_vector=feature,
        reward=reward,
        observation_noise_variance=observation_noise_variance,
    )

    covariance_times_feature = predictive_covariance @ feature
    denominator = float(
        feature @ covariance_times_feature + observation_noise_variance
    )
    expected_gain = covariance_times_feature / denominator
    expected_prediction = float(feature @ predictive_mean)
    expected_error = reward - expected_prediction
    expected_mean = predictive_mean + expected_gain * expected_error
    expected_covariance = predictive_covariance - torch.outer(
        expected_gain,
        covariance_times_feature,
    )

    assert result["predicted_reward"] == pytest.approx(expected_prediction)
    assert result["prediction_error"] == pytest.approx(expected_error)
    torch.testing.assert_close(result["gain_vector"], expected_gain)
    torch.testing.assert_close(result["posterior_mean"], expected_mean)
    torch.testing.assert_close(result["posterior_covariance"], expected_covariance)
    torch.testing.assert_close(
        result["posterior_covariance"],
        result["posterior_covariance"].T,
        rtol=0.0,
        atol=1e-15,
    )
    assert torch.trace(result["posterior_covariance"]) < torch.trace(
        predictive_covariance
    )


@pytest.mark.parametrize(
    "mean, covariance, expected_message",
    [
        ([0.0, math.nan], [[1.0, 0.0], [0.0, 1.0]], "finite"),
        ([0.0, 0.0], [[1.0, math.inf], [math.inf, 1.0]], "finite"),
        ([0.0, 0.0], [[1.0, 0.25], [0.0, 1.0]], "symmetric"),
        ([0.0, 0.0], [[1.0, 0.0], [0.0, -0.01]], "positive semidefinite"),
        ([0.0, 0.0], [[1.0]], "dimension"),
    ],
)
def test_gaussian_belief_validation_rejects_invalid_state(
    mean,
    covariance,
    expected_message,
):
    with pytest.raises(ProbabilisticControllerError, match=expected_message):
        validate_gaussian_belief(
            mean=_tensor(mean),
            covariance=_tensor(covariance),
            state_name="controlled belief",
        )


def test_gaussian_belief_validation_accepts_degenerate_psd_covariance():
    mean, covariance = validate_gaussian_belief(
        mean=[1.0, -2.0],
        covariance=[[1.0, 0.0], [0.0, 0.0]],
        state_name="degenerate belief",
    )

    assert mean.dtype == covariance.dtype == FLOAT64
    assert mean.device.type == covariance.device.type == "cpu"
    torch.testing.assert_close(mean, _tensor([1.0, -2.0]))
    torch.testing.assert_close(covariance, _tensor([[1.0, 0.0], [0.0, 0.0]]))


def test_gaussian_conditioning_rejects_nonfinite_observation_without_state_commit():
    predictive_mean = _tensor([0.0, 0.0])
    predictive_covariance = torch.eye(2, dtype=FLOAT64)

    with pytest.raises(ProbabilisticControllerError, match="reward.*finite"):
        condition_gaussian_belief(
            predictive_mean=predictive_mean,
            predictive_covariance=predictive_covariance,
            feature_vector=_tensor([1.0, 0.0]),
            reward=math.nan,
            observation_noise_variance=0.1,
        )

    torch.testing.assert_close(predictive_mean, _tensor([0.0, 0.0]))
    torch.testing.assert_close(
        predictive_covariance,
        torch.eye(2, dtype=FLOAT64),
    )


def test_posterior_sampling_uses_a_dedicated_generator_and_symmetric_factor():
    mean = _tensor([0.5, -1.0])
    covariance = _tensor([[4.0, 0.0], [0.0, 0.25]])
    expected_generator = torch.Generator(device="cpu").manual_seed(731)
    expected_standard_normal = torch.randn(
        2,
        dtype=FLOAT64,
        generator=expected_generator,
    )
    expected_sample = mean + _tensor([2.0, 0.5]) * expected_standard_normal

    torch.manual_seed(991)
    expected_next_global_draw = torch.randn(4)
    torch.manual_seed(991)
    sample = sample_gaussian_coefficients(
        mean=mean,
        covariance=covariance,
        generator=torch.Generator(device="cpu").manual_seed(731),
    )
    actual_next_global_draw = torch.randn(4)

    assert sample.dtype == FLOAT64
    assert sample.device.type == "cpu"
    torch.testing.assert_close(sample, expected_sample, rtol=0.0, atol=1e-15)
    assert torch.equal(actual_next_global_draw, expected_next_global_draw)


def test_posterior_sampling_supports_zero_covariance_without_jitter():
    mean = _tensor([1.0, -3.0, 2.0])

    sample = sample_gaussian_coefficients(
        mean=mean,
        covariance=torch.zeros((3, 3), dtype=FLOAT64),
        generator=torch.Generator(device="cpu").manual_seed(17),
    )

    torch.testing.assert_close(sample, mean, rtol=0.0, atol=0.0)


def test_global_selection_uses_arbitrary_labels_and_resolved_order_for_ties():
    labels = ["needle", "balanced", "entire"]
    schema = build_global_feature_schema(labels)
    zero_sample = torch.zeros(schema["dimension"], dtype=FLOAT64)

    tied = select_global_action(schema, zero_sample)

    assert tied["global_granularity"] == "needle"
    assert tied["tie_resolution"] == "resolved_granularity_order"
    assert tied["sampled_predicted_reward"] == pytest.approx(0.0)
    torch.testing.assert_close(
        tied["feature_vector"],
        encode_global_action(schema, "needle"),
    )

    target = "entire"
    target_feature = encode_global_action(schema, target)
    selected = select_global_action(schema, target_feature)
    scores = {
        label: float(encode_global_action(schema, label) @ target_feature)
        for label in labels
    }
    assert max(scores, key=scores.get) == target
    assert selected["global_granularity"] == target


def test_one_arm_schema_is_intercept_only_and_always_selects_the_only_label():
    schema = build_global_feature_schema(["only-choice"])

    assert schema["dimension"] == 1
    assert schema["contrast_basis"] == [[]]
    assert schema["coefficient_names"] == ["intercept"]
    torch.testing.assert_close(
        encode_global_action(schema, "only-choice"),
        _tensor([1.0]),
    )
    selected = select_global_action(schema, _tensor([-7.5]))
    assert selected["global_granularity"] == "only-choice"
    assert selected["sampled_predicted_reward"] == pytest.approx(-7.5)


def test_additive_feature_schema_is_identifiable_and_uses_stable_block_coefficients():
    labels = ["needle", "balanced", "entire"]
    schema = probabilistic_controller.build_additive_feature_schema(
        labels,
        block_count=2,
    )

    assert schema["scope"] == "per_block"
    assert schema["encoding"] == (
        "intercept_plus_per_block_sum_to_zero_contrasts"
    )
    assert schema["ordered_granularities"] == labels
    assert schema["block_count"] == 2
    assert schema["dimension"] == 1 + 2 * (len(labels) - 1)
    assert schema["coefficient_names"] == [
        "intercept",
        "block_0_contrast_0",
        "block_0_contrast_1",
        "block_1_contrast_0",
        "block_1_contrast_1",
    ]
    assert len(schema["schema_hash"]) == 64
    assert schema == probabilistic_controller.build_additive_feature_schema(
        labels,
        block_count=2,
    )

    complete_profiles = list(product(labels, repeat=2))
    design = torch.stack(
        [
            probabilistic_controller.encode_additive_action(schema, profile)
            for profile in complete_profiles
        ]
    )
    assert design.shape == (len(labels) ** 2, schema["dimension"])
    assert int(torch.linalg.matrix_rank(design)) == schema["dimension"]
    torch.testing.assert_close(
        design[:, 0],
        torch.ones(len(complete_profiles), dtype=FLOAT64),
    )


def test_additive_selection_supports_divergent_preferences_arbitrary_labels_and_ties():
    labels = ["needle", "entire"]
    schema = probabilistic_controller.build_additive_feature_schema(
        labels,
        block_count=2,
    )

    tied = probabilistic_controller.select_additive_action(
        schema,
        torch.zeros(schema["dimension"], dtype=FLOAT64),
    )
    assert tied["scope"] == "per_block"
    assert tied["block_granularities"] == ["needle", "needle"]
    assert tied["tie_resolution"] == "resolved_granularity_order_per_block"
    assert tied["sampled_predicted_reward"] == pytest.approx(0.0)

    target_profile = ["needle", "entire"]
    sampled_coefficients = probabilistic_controller.encode_additive_action(
        schema,
        target_profile,
    )
    selected = probabilistic_controller.select_additive_action(
        schema,
        sampled_coefficients,
    )

    assert selected["block_granularities"] == target_profile
    torch.testing.assert_close(
        selected["feature_vector"],
        probabilistic_controller.encode_additive_action(schema, target_profile),
    )
    assert "enumerated_profiles" not in selected
    assert "profile_scores" not in selected


def test_additive_posterior_learns_divergent_block_preferences_from_profile_rewards():
    labels = ["needle", "entire"]
    schema = probabilistic_controller.build_additive_feature_schema(
        labels,
        block_count=2,
    )
    true_coefficients = _tensor([0.25, 1.5, -2.0])
    posterior_mean = torch.zeros(schema["dimension"], dtype=FLOAT64)
    posterior_covariance = torch.eye(schema["dimension"], dtype=FLOAT64) * 100.0
    observation_count = 0

    for profile in product(labels, repeat=2):
        feature = probabilistic_controller.encode_additive_action(schema, profile)
        scalar_reward = float(feature @ true_coefficients)
        update = condition_gaussian_belief(
            predictive_mean=posterior_mean,
            predictive_covariance=posterior_covariance,
            feature_vector=feature,
            reward=scalar_reward,
            observation_noise_variance=1e-6,
        )
        posterior_mean = update["posterior_mean"]
        posterior_covariance = update["posterior_covariance"]
        observation_count += 1

    assert observation_count == len(labels) ** 2
    torch.testing.assert_close(
        posterior_mean,
        true_coefficients,
        rtol=1e-5,
        atol=1e-7,
    )
    selected = probabilistic_controller.select_additive_action(
        schema,
        posterior_mean,
    )
    assert selected["block_granularities"] == ["needle", "entire"]


@pytest.mark.parametrize(
    "labels, block_count, expected_dimension, expected_profile",
    [
        (["small", "large"], 1, 2, ["small"]),
        (["only-choice"], 3, 1, ["only-choice"] * 3),
    ],
)
def test_additive_schema_handles_one_block_and_one_granularity(
    labels,
    block_count,
    expected_dimension,
    expected_profile,
):
    schema = probabilistic_controller.build_additive_feature_schema(
        labels,
        block_count=block_count,
    )

    assert schema["dimension"] == expected_dimension
    encoded = probabilistic_controller.encode_additive_action(
        schema,
        expected_profile,
    )
    assert encoded.shape == (expected_dimension,)
    selected = probabilistic_controller.select_additive_action(
        schema,
        torch.zeros(expected_dimension, dtype=FLOAT64),
    )
    assert selected["block_granularities"] == expected_profile


def test_additive_complete_profile_conditions_once_on_one_scalar_window_reward(
    tmp_path,
):
    config = resolve_run_config(
        "tests/fixtures/probabilistic_adaptive_per_block_smoke.yaml",
        output_dir=tmp_path / "probabilistic-adaptive-per-block-smoke-001",
        overrides={"model.adaptive_controller.decision_interval_steps": 2},
    )
    controller = build_probabilistic_controller(
        controller_config=config["model"]["adaptive_controller"],
        sampling_seed=seed_for(config, "posterior_sampling"),
        manifest_hashes={
            "data_roles_manifest_hash": "parent-manifest-hash",
            "optimizer_training_manifest_hash": "training-manifest-hash",
            "controller_manifest_hash": "controller-manifest-hash",
            "ordinary_validation_manifest_hash": "validation-manifest-hash",
            "final_holdout_manifest_hash": "final-holdout-manifest-hash",
        },
    )
    controller.initialize_boundary(
        boundary_step=0,
        controller_objective=10.0,
        ordered_component_losses=[9.0, 10.0, 11.0],
        evaluation_target_tokens=384,
    )
    controller.record_successful_optimizer_step()
    controller.record_successful_optimizer_step()
    state_before_update = controller.state_dict()
    action = state_before_update["window"]["current_action"]
    expected_reward = (10.0 - 8.0) / 2
    expected_update = condition_gaussian_belief(
        predictive_mean=state_before_update["belief"]["predictive_mean"],
        predictive_covariance=state_before_update["belief"]["predictive_covariance"],
        feature_vector=action["feature_vector"],
        reward=expected_reward,
        observation_noise_variance=state_before_update["probabilistic_inputs"][
            "observation_noise_variance"
        ],
    )

    event = controller.complete_boundary(
        boundary_step=2,
        controller_objective=8.0,
        ordered_component_losses=[7.0, 8.0, 9.0],
        evaluation_target_tokens=384,
        training_will_continue=False,
    )
    state_after_update = controller.state_dict()

    assert event["reward"] == pytest.approx(expected_reward)
    assert event["action"]["block_granularities"] == action[
        "block_granularities"
    ]
    assert "block_rewards" not in event
    assert state_after_update["belief"]["round_index"] == 1
    torch.testing.assert_close(
        state_after_update["belief"]["posterior_mean"],
        expected_update["posterior_mean"],
    )
    torch.testing.assert_close(
        state_after_update["belief"]["posterior_covariance"],
        expected_update["posterior_covariance"],
    )


def _build_reset_controller(tmp_path):
    config = resolve_run_config(
        "tests/fixtures/probabilistic_adaptive_global_smoke.yaml",
        output_dir=tmp_path / "probabilistic-adaptive-global-smoke-001",
        overrides={
            "model.adaptive_controller.process_noise_covariance": 0.0,
            "model.adaptive_controller.reset.enabled": True,
            "model.adaptive_controller.reset.interval_steps": 12,
        },
    )
    controller = build_probabilistic_controller(
        controller_config=config["model"]["adaptive_controller"],
        sampling_seed=seed_for(config, "posterior_sampling"),
        manifest_hashes={
            "data_roles_manifest_hash": "parent",
            "optimizer_training_manifest_hash": "training",
            "controller_manifest_hash": "controller",
            "ordinary_validation_manifest_hash": "validation",
            "final_holdout_manifest_hash": "final",
        },
    )
    return config, controller


def _finish_reset_window(controller, *, boundary_step, objective, continuing=True):
    controller.record_successful_optimizer_step()
    controller.record_successful_optimizer_step()
    return controller.complete_boundary(
        boundary_step=boundary_step,
        controller_objective=objective,
        ordered_component_losses=[objective - 1.0, objective, objective + 1.0],
        evaluation_target_tokens=384,
        training_will_continue=continuing,
    )


def test_reset_episode_forces_balanced_acquisition_without_advancing_thompson_rng(
    tmp_path,
):
    _config, controller = _build_reset_controller(tmp_path)
    initial = controller.initialize_boundary(
        boundary_step=500,
        controller_objective=10.0,
        ordered_component_losses=[9.0, 10.0, 11.0],
        evaluation_target_tokens=384,
    )
    initial_state = controller.state_dict()

    assert initial["selected_action"]["selection_source"] == "forced_acquisition"
    assert initial_state["sampling"]["sample_count"] == 0
    assert initial_state["reset"]["controller_start_step"] == 500
    assert initial_state["reset"]["episode_start_step"] == 500
    assert initial_state["reset"]["episode_end_step"] == 512
    assert [event["event_type"] for event in initial["_journal_events"]] == [
        "episode_initialized",
        "initial_boundary",
    ]

    forced_actions = []
    for window_index in range(3):
        event = _finish_reset_window(
            controller,
            boundary_step=502 + 2 * window_index,
            objective=9.0 - 0.25 * window_index,
        )
        forced_actions.append(event["action"]["global_granularity"])
        assert event["selection_source"] == "forced_acquisition"
        assert controller.state_dict()["belief"]["round_index"] == window_index + 1

    state = controller.state_dict()
    assert set(forced_actions) == set(state["ordered_granularities"])
    assert state["reset"]["acquisition_counts"] == {
        label: 1 for label in state["ordered_granularities"]
    }
    assert state["sampling"]["sample_count"] == 1
    assert state["window"]["selection_source"] == "thompson"


def test_reset_boundary_conditions_then_archives_restores_prior_and_forces_next_episode(
    tmp_path,
):
    _config, controller = _build_reset_controller(tmp_path)
    controller.initialize_boundary(
        boundary_step=500,
        controller_objective=10.0,
        ordered_component_losses=[9.0, 10.0, 11.0],
        evaluation_target_tokens=384,
    )
    event = None
    for window_index in range(6):
        event = _finish_reset_window(
            controller,
            boundary_step=502 + 2 * window_index,
            objective=9.0 - 0.1 * window_index,
        )

    assert event is not None
    state = controller.state_dict()
    event_types = [item["event_type"] for item in event["_journal_events"]]
    assert event_types == [
        "completed_window",
        "episode_completed",
        "posterior_reset",
        "episode_initialized",
    ]
    archive = state["reset"]["completed_episodes"][0]
    assert archive["forced_acquisition_window_count"] == 3
    assert archive["thompson_window_count"] == 3
    assert state["reset"]["reset_count"] == 1
    assert state["reset"]["reset_steps"] == [512]
    assert state["reset"]["episode_index"] == 1
    assert state["reset"]["episode_start_step"] == 512
    assert state["reset"]["episode_offset_steps"] == 0
    assert state["window"]["selection_source"] == "forced_acquisition"
    assert state["sampling"]["sample_count"] == 3
    torch.testing.assert_close(
        state["belief"]["posterior_mean"],
        torch.tensor(
            state["probabilistic_inputs"]["resolved_prior_mean"],
            dtype=FLOAT64,
        ),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        state["belief"]["posterior_covariance"],
        torch.tensor(
            state["probabilistic_inputs"]["resolved_prior_covariance"],
            dtype=FLOAT64,
        ),
        rtol=0.0,
        atol=0.0,
    )


def test_exact_terminal_episode_boundary_archives_without_unused_reset(tmp_path):
    _config, controller = _build_reset_controller(tmp_path)
    controller.initialize_boundary(
        boundary_step=0,
        controller_objective=10.0,
        ordered_component_losses=[9.0, 10.0, 11.0],
        evaluation_target_tokens=384,
    )
    final_event = None
    for window_index in range(6):
        final_event = _finish_reset_window(
            controller,
            boundary_step=2 + 2 * window_index,
            objective=9.0 - 0.1 * window_index,
            continuing=window_index < 5,
        )

    state = controller.state_dict()
    assert final_event is not None
    assert [item["event_type"] for item in final_event["_journal_events"]] == [
        "completed_window",
        "episode_completed",
    ]
    assert state["reset"]["completed_episode_count"] == 1
    assert state["reset"]["reset_count"] == 0
    assert state["reset"]["reset_steps"] == []
    assert state["window"]["phase"] == "ready_for_action"
