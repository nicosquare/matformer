from __future__ import annotations

import math

import pytest
import torch

from src.training.probabilistic_controller import (
    ProbabilisticControllerError,
    build_global_feature_schema,
    condition_gaussian_belief,
    encode_global_action,
    predict_gaussian_belief,
    sample_gaussian_coefficients,
    select_global_action,
    validate_gaussian_belief,
)


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
