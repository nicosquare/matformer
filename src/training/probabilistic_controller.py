"""Small, explicit Gaussian Thompson controller for adaptive granularity."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from src.utils.config import (
    BAYESIAN_CONTROLLER_METHOD_FAMILY,
    BAYESIAN_CONTROLLER_METHOD_VERSION,
    BAYESIAN_COVARIANCE_TOLERANCE,
)
from src.utils.reproducibility import build_controller_reset_schedule, stable_hash


FLOAT64 = torch.float64
CONTROLLER_SCHEMA_VERSION = 2
LEGACY_CONTROLLER_SCHEMA_VERSION = 1
FEATURE_SCHEMA_VERSION = 1
SAMPLING_FACTORIZATION_CONTRACT = "symmetric_eigh_float64_v1"


class ProbabilisticControllerError(ValueError):
    """Raised when controller state would make an experiment invalid."""


def _float64_cpu_tensor(value: Any, *, field_name: str) -> torch.Tensor:
    try:
        return torch.as_tensor(value, dtype=FLOAT64, device="cpu").clone()
    except (TypeError, ValueError, RuntimeError) as error:
        raise ProbabilisticControllerError(
            f"{field_name} must be convertible to a CPU float64 tensor"
        ) from error


def _finite_scalar(value: Any, *, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ProbabilisticControllerError(
            f"{field_name} must be a finite scalar"
        ) from error
    if not math.isfinite(result):
        raise ProbabilisticControllerError(f"{field_name} must be finite")
    return result


def validate_gaussian_belief(
    mean: Any,
    covariance: Any,
    *,
    state_name: str = "Gaussian belief",
    tolerance: float = BAYESIAN_COVARIANCE_TOLERANCE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a validated CPU float64 mean/covariance pair."""

    mean_tensor = _float64_cpu_tensor(mean, field_name=f"{state_name} mean")
    covariance_tensor = _float64_cpu_tensor(
        covariance,
        field_name=f"{state_name} covariance",
    )
    if mean_tensor.ndim != 1:
        raise ProbabilisticControllerError(
            f"{state_name} mean dimension must be one-dimensional"
        )
    dimension = int(mean_tensor.numel())
    if dimension <= 0:
        raise ProbabilisticControllerError(
            f"{state_name} mean dimension must be positive"
        )
    if covariance_tensor.ndim != 2 or tuple(covariance_tensor.shape) != (
        dimension,
        dimension,
    ):
        raise ProbabilisticControllerError(
            f"{state_name} covariance dimension must be {dimension}x{dimension}"
        )
    if not bool(torch.isfinite(mean_tensor).all()):
        raise ProbabilisticControllerError(f"{state_name} mean must be finite")
    if not bool(torch.isfinite(covariance_tensor).all()):
        raise ProbabilisticControllerError(
            f"{state_name} covariance must be finite"
        )
    if not torch.allclose(
        covariance_tensor,
        covariance_tensor.T,
        rtol=0.0,
        atol=float(tolerance),
    ):
        raise ProbabilisticControllerError(
            f"{state_name} covariance must be symmetric"
        )

    covariance_tensor = (covariance_tensor + covariance_tensor.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_tensor)
    smallest_eigenvalue = float(eigenvalues.min())
    if smallest_eigenvalue < -float(tolerance):
        raise ProbabilisticControllerError(
            f"{state_name} covariance must be positive semidefinite"
        )
    if smallest_eigenvalue < 0.0:
        covariance_tensor = (
            eigenvectors * eigenvalues.clamp_min(0.0).unsqueeze(0)
        ) @ eigenvectors.T
        covariance_tensor = (covariance_tensor + covariance_tensor.T) * 0.5
    return mean_tensor, covariance_tensor


def _validate_covariance_for_dimension(
    covariance: Any,
    *,
    dimension: int,
    state_name: str,
) -> torch.Tensor:
    zero_mean = torch.zeros(dimension, dtype=FLOAT64)
    _, covariance_tensor = validate_gaussian_belief(
        zero_mean,
        covariance,
        state_name=state_name,
    )
    return covariance_tensor


def build_global_feature_schema(
    ordered_granularities: Sequence[str],
    *,
    block_count: int = 1,
) -> dict[str, Any]:
    """Build intercept plus ordered orthonormal Helmert contrasts."""

    labels = [str(label) for label in ordered_granularities]
    if not labels or any(not label for label in labels):
        raise ProbabilisticControllerError(
            "ordered granularities must contain nonempty labels"
        )
    if len(set(labels)) != len(labels):
        raise ProbabilisticControllerError(
            "ordered granularity labels must be unique"
        )
    if isinstance(block_count, bool) or int(block_count) <= 0:
        raise ProbabilisticControllerError("block_count must be positive")

    label_count = len(labels)
    contrast_basis = torch.zeros(
        (label_count, max(0, label_count - 1)),
        dtype=FLOAT64,
    )
    for column in range(label_count - 1):
        positive_count = column + 1
        normalizer = math.sqrt(positive_count * (positive_count + 1))
        contrast_basis[:positive_count, column] = 1.0 / normalizer
        contrast_basis[positive_count, column] = -positive_count / normalizer

    schema_without_hash = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "scope": "global",
        "encoding": "intercept_plus_sum_to_zero_contrasts",
        "ordered_granularities": labels,
        "block_count": int(block_count),
        "contrast_basis": contrast_basis.tolist(),
        "coefficient_names": ["intercept"]
        + [f"global_contrast_{index}" for index in range(label_count - 1)],
        "dimension": label_count,
        "tie_order": labels,
    }
    return {
        **schema_without_hash,
        "schema_hash": stable_hash(schema_without_hash),
    }


def build_additive_feature_schema(
    ordered_granularities: Sequence[str],
    *,
    block_count: int,
) -> dict[str, Any]:
    """Build one ordered sum-to-zero contrast group per transformer block."""

    global_schema = build_global_feature_schema(
        ordered_granularities,
        block_count=block_count,
    )
    labels = list(global_schema["ordered_granularities"])
    contrast_count = max(0, len(labels) - 1)
    coefficient_names = ["intercept"]
    for block_index in range(int(block_count)):
        coefficient_names.extend(
            f"block_{block_index}_contrast_{contrast_index}"
            for contrast_index in range(contrast_count)
        )

    schema_without_hash = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "scope": "per_block",
        "encoding": "intercept_plus_per_block_sum_to_zero_contrasts",
        "ordered_granularities": labels,
        "block_count": int(block_count),
        "contrast_basis": copy.deepcopy(global_schema["contrast_basis"]),
        "coefficient_names": coefficient_names,
        "dimension": 1 + int(block_count) * contrast_count,
        "tie_order": {
            "block_order": list(range(int(block_count))),
            "granularity_order": labels,
        },
    }
    return {
        **schema_without_hash,
        "schema_hash": stable_hash(schema_without_hash),
    }


def encode_global_action(
    feature_schema: Mapping[str, Any],
    granularity: str,
) -> torch.Tensor:
    """Encode one global label in the schema's coefficient order."""

    labels = list(feature_schema.get("ordered_granularities", []))
    if granularity not in labels:
        raise ProbabilisticControllerError(
            f"unknown global granularity {granularity!r}; expected one of {labels}"
        )
    basis = _float64_cpu_tensor(
        feature_schema.get("contrast_basis"),
        field_name="global contrast basis",
    )
    expected_shape = (len(labels), max(0, len(labels) - 1))
    if tuple(basis.shape) != expected_shape:
        raise ProbabilisticControllerError(
            f"global contrast basis dimension must be {expected_shape}"
        )
    row = basis[labels.index(granularity)]
    return torch.cat((torch.ones(1, dtype=FLOAT64), row))


def encode_additive_action(
    feature_schema: Mapping[str, Any],
    block_granularities: Sequence[str],
) -> torch.Tensor:
    """Encode one complete profile without assigning separate block rewards."""

    labels = list(feature_schema.get("ordered_granularities", []))
    block_count = int(feature_schema.get("block_count", 0))
    profile = list(block_granularities)
    if len(profile) != block_count:
        raise ProbabilisticControllerError(
            "per-block action must contain one granularity per block"
        )
    unknown_labels = [label for label in profile if label not in labels]
    if unknown_labels:
        raise ProbabilisticControllerError(
            f"unknown per-block granularities {unknown_labels}; expected {labels}"
        )
    basis = _float64_cpu_tensor(
        feature_schema.get("contrast_basis"),
        field_name="additive contrast basis",
    )
    expected_shape = (len(labels), max(0, len(labels) - 1))
    if tuple(basis.shape) != expected_shape:
        raise ProbabilisticControllerError(
            f"additive contrast basis dimension must be {expected_shape}"
        )
    rows = [basis[labels.index(label)] for label in profile]
    return torch.cat((torch.ones(1, dtype=FLOAT64), *rows))


def predict_gaussian_belief(
    *,
    posterior_mean: Any,
    posterior_covariance: Any,
    process_noise_covariance: Any,
) -> dict[str, Any]:
    """Apply the fixed identity transition and additive process covariance."""

    mean, covariance = validate_gaussian_belief(
        posterior_mean,
        posterior_covariance,
        state_name="posterior belief",
    )
    process_covariance = _validate_covariance_for_dimension(
        process_noise_covariance,
        dimension=int(mean.numel()),
        state_name="process noise",
    )
    predictive_covariance = (covariance + process_covariance)
    predictive_covariance = (
        predictive_covariance + predictive_covariance.T
    ) * 0.5
    predictive_mean, predictive_covariance = validate_gaussian_belief(
        mean,
        predictive_covariance,
        state_name="predictive belief",
    )
    return {
        "transition_model": "identity",
        "predictive_mean": predictive_mean,
        "predictive_covariance": predictive_covariance,
    }


def condition_gaussian_belief(
    *,
    predictive_mean: Any,
    predictive_covariance: Any,
    feature_vector: Any,
    reward: Any,
    observation_noise_variance: Any,
) -> dict[str, Any]:
    """Condition a linear Gaussian belief on one scalar window reward."""

    mean, covariance = validate_gaussian_belief(
        predictive_mean,
        predictive_covariance,
        state_name="predictive belief",
    )
    feature = _float64_cpu_tensor(feature_vector, field_name="feature vector")
    if feature.ndim != 1 or feature.numel() != mean.numel():
        raise ProbabilisticControllerError(
            "feature vector dimension must match predictive belief"
        )
    if not bool(torch.isfinite(feature).all()):
        raise ProbabilisticControllerError("feature vector must be finite")
    reward_value = _finite_scalar(reward, field_name="reward")
    noise_variance = _finite_scalar(
        observation_noise_variance,
        field_name="observation noise variance",
    )
    if noise_variance <= 0.0:
        raise ProbabilisticControllerError(
            "observation noise variance must be positive"
        )

    covariance_times_feature = covariance @ feature
    denominator = float(feature @ covariance_times_feature) + noise_variance
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ProbabilisticControllerError(
            "Gaussian conditioning denominator must be finite and positive"
        )
    gain = covariance_times_feature / denominator
    predicted_reward = float(feature @ mean)
    prediction_error = reward_value - predicted_reward
    posterior_mean = mean + gain * prediction_error
    posterior_covariance = covariance - torch.outer(
        gain,
        covariance_times_feature,
    )
    posterior_covariance = (posterior_covariance + posterior_covariance.T) * 0.5
    posterior_mean, posterior_covariance = validate_gaussian_belief(
        posterior_mean,
        posterior_covariance,
        state_name="posterior belief",
    )
    return {
        "predicted_reward": predicted_reward,
        "prediction_error": prediction_error,
        "gain_vector": gain,
        "posterior_mean": posterior_mean,
        "posterior_covariance": posterior_covariance,
    }


def sample_gaussian_coefficients(
    *,
    mean: Any,
    covariance: Any,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw with a dedicated generator through the symmetric PSD square root."""

    if not isinstance(generator, torch.Generator):
        raise ProbabilisticControllerError(
            "posterior sampling requires a dedicated torch.Generator"
        )
    mean_tensor, covariance_tensor = validate_gaussian_belief(
        mean,
        covariance,
        state_name="sampling belief",
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_tensor)
    eigenvalues = eigenvalues.clamp_min(0.0)
    symmetric_factor = (
        eigenvectors * torch.sqrt(eigenvalues).unsqueeze(0)
    ) @ eigenvectors.T
    standard_normal = torch.randn(
        mean_tensor.numel(),
        dtype=FLOAT64,
        device="cpu",
        generator=generator,
    )
    sample = mean_tensor + symmetric_factor @ standard_normal
    if not bool(torch.isfinite(sample).all()):
        raise ProbabilisticControllerError(
            "sampled Gaussian coefficients must be finite"
        )
    return sample


def select_global_action(
    feature_schema: Mapping[str, Any],
    sampled_coefficients: Any,
) -> dict[str, Any]:
    """Maximize sampled global reward, breaking exact ties by label order."""

    coefficients = _float64_cpu_tensor(
        sampled_coefficients,
        field_name="sampled coefficients",
    )
    dimension = int(feature_schema.get("dimension", -1))
    if coefficients.ndim != 1 or coefficients.numel() != dimension:
        raise ProbabilisticControllerError(
            "sampled coefficient dimension must match the feature schema"
        )
    if not bool(torch.isfinite(coefficients).all()):
        raise ProbabilisticControllerError(
            "sampled coefficients must be finite"
        )

    labels = list(feature_schema["ordered_granularities"])
    features = [encode_global_action(feature_schema, label) for label in labels]
    scores = [float(feature @ coefficients) for feature in features]
    selected_index = max(range(len(labels)), key=scores.__getitem__)
    selected_label = labels[selected_index]
    block_count = int(feature_schema.get("block_count", 1))
    return {
        "scope": "global",
        "global_granularity": selected_label,
        "block_granularities": [selected_label] * block_count,
        "feature_vector": features[selected_index],
        "sampled_predicted_reward": scores[selected_index],
        "tie_resolution": "resolved_granularity_order",
    }


def select_additive_action(
    feature_schema: Mapping[str, Any],
    sampled_coefficients: Any,
) -> dict[str, Any]:
    """Maximize an additive sample independently in O(B|G|) work."""

    coefficients = _float64_cpu_tensor(
        sampled_coefficients,
        field_name="sampled coefficients",
    )
    dimension = int(feature_schema.get("dimension", -1))
    if coefficients.ndim != 1 or coefficients.numel() != dimension:
        raise ProbabilisticControllerError(
            "sampled coefficient dimension must match the feature schema"
        )
    if not bool(torch.isfinite(coefficients).all()):
        raise ProbabilisticControllerError(
            "sampled coefficients must be finite"
        )

    labels = list(feature_schema.get("ordered_granularities", []))
    block_count = int(feature_schema.get("block_count", 0))
    basis = _float64_cpu_tensor(
        feature_schema.get("contrast_basis"),
        field_name="additive contrast basis",
    )
    contrast_count = max(0, len(labels) - 1)
    expected_shape = (len(labels), contrast_count)
    if tuple(basis.shape) != expected_shape:
        raise ProbabilisticControllerError(
            f"additive contrast basis dimension must be {expected_shape}"
        )

    selected_profile = []
    for block_index in range(block_count):
        start = 1 + block_index * contrast_count
        block_coefficients = coefficients[start : start + contrast_count]
        block_scores = [float(row @ block_coefficients) for row in basis]
        selected_index = max(
            range(len(labels)),
            key=block_scores.__getitem__,
        )
        selected_profile.append(labels[selected_index])

    feature = encode_additive_action(feature_schema, selected_profile)
    return {
        "scope": "per_block",
        "global_granularity": None,
        "block_granularities": selected_profile,
        "feature_vector": feature,
        "sampled_predicted_reward": float(feature @ coefficients),
        "tie_resolution": "resolved_granularity_order_per_block",
    }


class ProbabilisticController:
    """Own Bayesian controller belief, RNG, and decision-window state."""

    def __init__(
        self,
        *,
        controller_config: Mapping[str, Any],
        sampling_seed: int,
        manifest_hashes: Mapping[str, str],
    ) -> None:
        config = copy.deepcopy(dict(controller_config))
        expected_fixed_fields = {
            "method_family": BAYESIAN_CONTROLLER_METHOD_FAMILY,
            "method_version": BAYESIAN_CONTROLLER_METHOD_VERSION,
            "strategy": "thompson",
            "context_model": "intercept_only",
            "transition_model": "identity",
            "compute_weight": 0.0,
            "switch_weight": 0.0,
        }
        for field_name, expected_value in expected_fixed_fields.items():
            if config.get(field_name) != expected_value:
                raise ProbabilisticControllerError(
                    f"controller {field_name} must be {expected_value!r}"
                )
        scope = config.get("scope")
        if scope not in {"global", "per_block"}:
            raise ProbabilisticControllerError(
                "controller scope must be 'global' or 'per_block'"
            )
        expected_feature_model = "arms" if scope == "global" else "additive"
        if config.get("feature_model") != expected_feature_model:
            raise ProbabilisticControllerError(
                f"controller feature_model must be {expected_feature_model!r}"
            )
        ordered_granularities = list(config.get("ordered_granularities", []))
        block_count = int(config.get("block_count", 0))
        if scope == "global":
            feature_schema = build_global_feature_schema(
                ordered_granularities,
                block_count=block_count,
            )
        else:
            feature_schema = build_additive_feature_schema(
                ordered_granularities,
                block_count=block_count,
            )
        dimension = feature_schema["dimension"]
        prior_mean, prior_covariance = validate_gaussian_belief(
            config.get("resolved_prior_mean"),
            config.get("resolved_prior_covariance"),
            state_name="resolved prior",
        )
        if prior_mean.numel() != dimension:
            raise ProbabilisticControllerError(
                "resolved prior dimension must match the feature schema"
            )
        process_covariance = _validate_covariance_for_dimension(
            config.get("resolved_process_noise_covariance"),
            dimension=dimension,
            state_name="resolved process noise",
        )
        observation_noise_variance = _finite_scalar(
            config.get("observation_noise_variance"),
            field_name="observation noise variance",
        )
        if observation_noise_variance <= 0.0:
            raise ProbabilisticControllerError(
                "observation noise variance must be positive"
            )
        interval = config.get("decision_interval_steps")
        if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
            raise ProbabilisticControllerError(
                "decision interval steps must be a positive integer"
            )
        reset_contract = copy.deepcopy(dict(config.get("reset", {})))
        reset_enabled = bool(reset_contract.get("enabled", False))
        if reset_enabled and scope != "global":
            raise ProbabilisticControllerError(
                "episodic reset is supported only for the global controller"
            )

        self._generator = torch.Generator(device="cpu").manual_seed(int(sampling_seed))
        self._state: dict[str, Any] = {
            "schema_version": CONTROLLER_SCHEMA_VERSION,
            "method_family": BAYESIAN_CONTROLLER_METHOD_FAMILY,
            "method_version": BAYESIAN_CONTROLLER_METHOD_VERSION,
            "strategy": "thompson",
            "scope": scope,
            "ordered_granularities": ordered_granularities,
            "block_count": block_count,
            "feature_schema": feature_schema,
            "probabilistic_inputs": {
                "resolved_prior_mean": prior_mean.tolist(),
                "resolved_prior_covariance": prior_covariance.tolist(),
                "observation_noise_variance": observation_noise_variance,
                "resolved_process_noise_covariance": process_covariance.tolist(),
                "transition_model": "identity",
                "context_model": "intercept_only",
                "compute_weight": 0.0,
                "switch_weight": 0.0,
            },
            "manifest_hashes": dict(manifest_hashes),
            "belief": {
                "round_index": 0,
                "posterior_mean": prior_mean,
                "posterior_covariance": prior_covariance,
                "predictive_mean": None,
                "predictive_covariance": None,
                "last_prediction_step": None,
                "last_update_step": None,
            },
            "sampling": {
                "seed_stream_name": "posterior_sampling",
                "resolved_seed": int(sampling_seed),
                "generator_state": self._generator.get_state(),
                "sample_count": 0,
                "factorization_contract": SAMPLING_FACTORIZATION_CONTRACT,
            },
            "reset": {
                "contract": reset_contract,
                "enabled": reset_enabled,
                "controller_start_step": None,
                "episode_index": None,
                "episode_start_step": None,
                "episode_end_step": None,
                "episode_offset_steps": 0,
                "reset_count": 0,
                "reset_steps": [],
                "acquisition_completed_windows": 0,
                "acquisition_total_windows": (
                    int(reset_contract.get("acquisition_window_count", 0))
                    if reset_enabled
                    else 0
                ),
                "acquisition_counts": {
                    label: 0 for label in ordered_granularities
                },
                "selection_source": None,
                "schedule_seed": None,
                "schedule": [],
                "schedule_hash": None,
                "completed_episode_count": 0,
                "completed_episodes": [],
            },
            "window": {
                "phase": "initial_objective_pending",
                "window_index": 0,
                "decision_interval_steps": interval,
                "boundary_step": 0,
                "current_action": None,
                "selection_source": None,
                "completed_optimizer_steps": 0,
                "pre_window_objective": None,
                "ordered_pre_window_component_losses": None,
                "boundary_evaluation_status": "not_started",
                "terminal_status": "continuing",
            },
            "journal": {
                "path": "controller_metrics.jsonl",
                "event_count": 0,
                "last_committed_offset": None,
                "last_committed_hash": None,
            },
            "resume": {
                "resume_count": 0,
                "source_checkpoint": None,
                "compatibility_status": "fresh",
            },
            "failure": None,
        }

    @property
    def current_action(self) -> dict[str, Any] | None:
        action = self._state["window"]["current_action"]
        return copy.deepcopy(action)

    def state_dict(self) -> dict[str, Any]:
        self._state["sampling"]["generator_state"] = self._generator.get_state()
        return copy.deepcopy(self._state)

    def transaction_snapshot(self) -> dict[str, Any]:
        """Capture belief, window, journal, and controller-local RNG atomically."""

        return self.state_dict()

    def restore_transaction_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        restored = copy.deepcopy(dict(snapshot))
        generator_state = restored["sampling"]["generator_state"]
        self._generator.set_state(generator_state.cpu())
        restored["sampling"]["generator_state"] = self._generator.get_state()
        self._state = restored

    def record_journal_commit(self, commit: Mapping[str, Any]) -> None:
        """Attach durable journal provenance after an event append succeeds."""

        journal = self._state["journal"]
        if commit.get("path") is not None:
            journal["path"] = str(commit["path"])
        journal["last_committed_offset"] = int(commit["last_committed_offset"])
        journal["last_committed_hash"] = str(commit["event_hash"])

    def record_warmup_journal_commit(self, commit: Mapping[str, Any]) -> None:
        """Count an audit-only warmup event without changing Bayesian state."""

        self._state["journal"]["event_count"] += 1
        self.record_journal_commit(commit)

    def _validate_objective(
        self,
        *,
        controller_objective: Any,
        ordered_component_losses: Sequence[Any],
        evaluation_target_tokens: Any,
    ) -> tuple[float, list[float], int]:
        objective = _finite_scalar(
            controller_objective,
            field_name="controller objective",
        )
        losses = [
            _finite_scalar(value, field_name="controller component loss")
            for value in ordered_component_losses
        ]
        if len(losses) != len(self._state["ordered_granularities"]):
            raise ProbabilisticControllerError(
                "controller component loss count must match ordered granularities"
            )
        if (
            isinstance(evaluation_target_tokens, bool)
            or not isinstance(evaluation_target_tokens, int)
            or evaluation_target_tokens <= 0
        ):
            raise ProbabilisticControllerError(
                "evaluation target tokens must be a positive integer"
            )
        return objective, losses, evaluation_target_tokens

    def _predict_and_select(self, *, boundary_step: int) -> dict[str, Any]:
        prediction = self._predict(boundary_step=boundary_step)
        sample = sample_gaussian_coefficients(
            mean=prediction["predictive_mean"],
            covariance=prediction["predictive_covariance"],
            generator=self._generator,
        )
        if self._state["scope"] == "global":
            action = select_global_action(self._state["feature_schema"], sample)
        else:
            action = select_additive_action(self._state["feature_schema"], sample)
        action["selection_source"] = "thompson"
        action["selection_round"] = int(self._state["belief"]["round_index"])
        sampling = self._state["sampling"]
        sampling["sample_count"] = int(sampling["sample_count"]) + 1
        sampling["generator_state"] = self._generator.get_state()
        return action

    def _predict(self, *, boundary_step: int) -> dict[str, Any]:
        belief = self._state["belief"]
        inputs = self._state["probabilistic_inputs"]
        prediction = predict_gaussian_belief(
            posterior_mean=belief["posterior_mean"],
            posterior_covariance=belief["posterior_covariance"],
            process_noise_covariance=inputs["resolved_process_noise_covariance"],
        )
        belief["predictive_mean"] = prediction["predictive_mean"]
        belief["predictive_covariance"] = prediction["predictive_covariance"]
        belief["last_prediction_step"] = int(boundary_step)
        return prediction

    def _select_forced_acquisition(
        self,
        *,
        boundary_step: int,
        schedule_index: int,
    ) -> dict[str, Any]:
        reset = self._state["reset"]
        schedule = list(reset["schedule"])
        if schedule_index < 0 or schedule_index >= len(schedule):
            raise ProbabilisticControllerError(
                "forced acquisition schedule index is out of range"
            )
        prediction = self._predict(boundary_step=boundary_step)
        label = schedule[schedule_index]
        feature = encode_global_action(self._state["feature_schema"], label)
        action = {
            "scope": "global",
            "global_granularity": label,
            "block_granularities": [label] * int(self._state["block_count"]),
            "feature_vector": feature,
            "sampled_predicted_reward": float(
                feature @ prediction["predictive_mean"]
            ),
            "tie_resolution": "forced_acquisition_schedule",
            "selection_source": "forced_acquisition",
            "selection_round": int(self._state["belief"]["round_index"]),
            "acquisition_schedule_index": int(schedule_index),
        }
        return action

    def _initialize_reset_episode(
        self,
        *,
        episode_index: int,
        boundary_step: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        reset = self._state["reset"]
        contract = reset["contract"]
        schedule, episode_seed, schedule_hash = build_controller_reset_schedule(
            self._state["ordered_granularities"],
            acquisition_passes=int(contract["acquisition_passes"]),
            root_seed=int(contract["schedule_seed"]),
            episode_index=int(episode_index),
        )
        interval_steps = int(contract["interval_steps"])
        reset.update(
            controller_start_step=(
                int(boundary_step)
                if reset["controller_start_step"] is None
                else int(reset["controller_start_step"])
            ),
            episode_index=int(episode_index),
            episode_start_step=int(boundary_step),
            episode_end_step=int(boundary_step) + interval_steps,
            episode_offset_steps=0,
            acquisition_completed_windows=0,
            acquisition_total_windows=len(schedule),
            acquisition_counts={
                label: 0 for label in self._state["ordered_granularities"]
            },
            selection_source="forced_acquisition",
            schedule_seed=int(episode_seed),
            schedule=schedule,
            schedule_hash=schedule_hash,
        )
        action = self._select_forced_acquisition(
            boundary_step=boundary_step,
            schedule_index=0,
        )
        event = {
            "schema_version": CONTROLLER_SCHEMA_VERSION,
            "event_type": "episode_initialized",
            "boundary_step": int(boundary_step),
            "window_index": int(self._state["window"]["window_index"]),
            "episode_index": int(episode_index),
            "episode_start_step": int(boundary_step),
            "episode_end_step": int(boundary_step) + interval_steps,
            "episode_offset_steps": 0,
            "schedule_seed": int(episode_seed),
            "schedule": list(schedule),
            "schedule_hash": schedule_hash,
            "acquisition_total_windows": len(schedule),
            "acquisition_completed_windows": 0,
            "acquisition_counts": copy.deepcopy(reset["acquisition_counts"]),
            "selected_action": copy.deepcopy(action),
            "selection_source": "forced_acquisition",
        }
        return action, event

    def initialize_boundary(
        self,
        *,
        boundary_step: int,
        controller_objective: Any,
        ordered_component_losses: Sequence[Any],
        evaluation_target_tokens: int,
    ) -> dict[str, Any]:
        snapshot = self.transaction_snapshot()
        try:
            return self._initialize_boundary(
                boundary_step=boundary_step,
                controller_objective=controller_objective,
                ordered_component_losses=ordered_component_losses,
                evaluation_target_tokens=evaluation_target_tokens,
            )
        except Exception:
            self.restore_transaction_snapshot(snapshot)
            raise

    def _initialize_boundary(
        self,
        *,
        boundary_step: int,
        controller_objective: Any,
        ordered_component_losses: Sequence[Any],
        evaluation_target_tokens: int,
    ) -> dict[str, Any]:
        window = self._state["window"]
        if window["phase"] != "initial_objective_pending":
            raise ProbabilisticControllerError(
                "initial boundary requires initial_objective_pending phase"
            )
        objective, losses, target_tokens = self._validate_objective(
            controller_objective=controller_objective,
            ordered_component_losses=ordered_component_losses,
            evaluation_target_tokens=evaluation_target_tokens,
        )
        journal_events: list[dict[str, Any]] = []
        if self._state["reset"]["enabled"]:
            action, episode_event = self._initialize_reset_episode(
                episode_index=0,
                boundary_step=boundary_step,
            )
            journal_events.append(episode_event)
        else:
            action = self._predict_and_select(boundary_step=boundary_step)
        window.update(
            phase="active_window",
            window_index=0,
            boundary_step=int(boundary_step),
            current_action=action,
            selection_source=action["selection_source"],
            completed_optimizer_steps=0,
            pre_window_objective=objective,
            ordered_pre_window_component_losses=losses,
            boundary_evaluation_status="not_started",
            terminal_status="continuing",
        )
        initial_event = {
            "schema_version": CONTROLLER_SCHEMA_VERSION,
            "event_type": "initial_boundary",
            "boundary_step": int(boundary_step),
            "window_index": 0,
            "controller_objective": objective,
            "ordered_component_losses": losses,
            "evaluation_target_tokens": target_tokens,
            "predictive_mean": copy.deepcopy(self._state["belief"]["predictive_mean"]),
            "predictive_covariance": copy.deepcopy(
                self._state["belief"]["predictive_covariance"]
            ),
            "selected_action": copy.deepcopy(action),
            "selection_source": action["selection_source"],
        }
        journal_events.append(copy.deepcopy(initial_event))
        initial_event["_journal_events"] = journal_events
        self._state["journal"]["event_count"] += len(journal_events)
        return initial_event

    def record_successful_optimizer_step(self) -> None:
        window = self._state["window"]
        if window["phase"] != "active_window":
            raise ProbabilisticControllerError(
                "successful optimizer steps require an active controller window"
            )
        completed = int(window["completed_optimizer_steps"]) + 1
        interval = int(window["decision_interval_steps"])
        if completed > interval:
            raise ProbabilisticControllerError(
                "controller window cannot exceed its decision interval"
            )
        window["completed_optimizer_steps"] = completed
        reset = self._state["reset"]
        if reset["enabled"]:
            reset["episode_offset_steps"] = int(reset["episode_offset_steps"]) + 1
            if reset["episode_offset_steps"] > int(
                reset["contract"]["interval_steps"]
            ):
                raise ProbabilisticControllerError(
                    "controller episode cannot exceed its reset interval"
                )
        if completed == interval:
            window["phase"] = "boundary_evaluation_pending"
            window["boundary_evaluation_status"] = "pending"

    def complete_boundary(
        self,
        *,
        boundary_step: int,
        controller_objective: Any,
        ordered_component_losses: Sequence[Any],
        evaluation_target_tokens: int,
        training_will_continue: bool,
    ) -> dict[str, Any]:
        snapshot = self.transaction_snapshot()
        try:
            return self._complete_boundary(
                boundary_step=boundary_step,
                controller_objective=controller_objective,
                ordered_component_losses=ordered_component_losses,
                evaluation_target_tokens=evaluation_target_tokens,
                training_will_continue=training_will_continue,
            )
        except Exception:
            self.restore_transaction_snapshot(snapshot)
            raise

    def _complete_boundary(
        self,
        *,
        boundary_step: int,
        controller_objective: Any,
        ordered_component_losses: Sequence[Any],
        evaluation_target_tokens: int,
        training_will_continue: bool,
    ) -> dict[str, Any]:
        window = self._state["window"]
        if window["phase"] != "boundary_evaluation_pending":
            raise ProbabilisticControllerError(
                "completed boundary requires boundary_evaluation_pending phase"
            )
        objective, losses, target_tokens = self._validate_objective(
            controller_objective=controller_objective,
            ordered_component_losses=ordered_component_losses,
            evaluation_target_tokens=evaluation_target_tokens,
        )
        interval = int(window["decision_interval_steps"])
        reward = (
            float(window["pre_window_objective"]) - objective
        ) / interval
        reward = _finite_scalar(reward, field_name="reward")
        belief = self._state["belief"]
        action = copy.deepcopy(window["current_action"])
        update = condition_gaussian_belief(
            predictive_mean=belief["predictive_mean"],
            predictive_covariance=belief["predictive_covariance"],
            feature_vector=action["feature_vector"],
            reward=reward,
            observation_noise_variance=self._state["probabilistic_inputs"][
                "observation_noise_variance"
            ],
        )
        event = {
            "schema_version": CONTROLLER_SCHEMA_VERSION,
            "event_type": "completed_window",
            "window_index": int(window["window_index"]),
            "boundary_step_start": int(window["boundary_step"]),
            "boundary_step_end": int(boundary_step),
            "completed_optimizer_steps": int(window["completed_optimizer_steps"]),
            "action": action,
            "selection_source": action.get("selection_source", "thompson"),
            "pre_window_objective": float(window["pre_window_objective"]),
            "post_window_objective": objective,
            "ordered_component_losses": losses,
            "evaluation_target_tokens": target_tokens,
            "reward": reward,
            "predicted_reward": update["predicted_reward"],
            "prediction_error": update["prediction_error"],
            "predictive_mean": copy.deepcopy(belief["predictive_mean"]),
            "predictive_covariance": copy.deepcopy(belief["predictive_covariance"]),
            "gain_vector": update["gain_vector"],
            "posterior_mean": update["posterior_mean"],
            "posterior_covariance": update["posterior_covariance"],
        }
        belief.update(
            round_index=int(belief["round_index"]) + 1,
            posterior_mean=update["posterior_mean"],
            posterior_covariance=update["posterior_covariance"],
            last_update_step=int(boundary_step),
        )
        next_window_index = int(window["window_index"]) + 1
        window.update(
            phase="ready_for_action",
            window_index=next_window_index,
            boundary_step=int(boundary_step),
            current_action=None,
            selection_source=None,
            completed_optimizer_steps=0,
            pre_window_objective=objective,
            ordered_pre_window_component_losses=losses,
            boundary_evaluation_status="complete",
            terminal_status="complete_boundary",
        )
        journal_events: list[dict[str, Any]] = [copy.deepcopy(event)]
        reset = self._state["reset"]
        reset_boundary = False
        if reset["enabled"]:
            episode_index = int(reset["episode_index"])
            selection_source = str(action.get("selection_source"))
            if selection_source == "forced_acquisition":
                label = str(action["global_granularity"])
                reset["acquisition_completed_windows"] = int(
                    reset["acquisition_completed_windows"]
                ) + 1
                reset["acquisition_counts"][label] = int(
                    reset["acquisition_counts"].get(label, 0)
                ) + 1
                acquisition_event = {
                    "schema_version": CONTROLLER_SCHEMA_VERSION,
                    "event_type": (
                        "acquisition_completed"
                        if int(reset["acquisition_completed_windows"])
                        == int(reset["acquisition_total_windows"])
                        else "acquisition_progress"
                    ),
                    "boundary_step": int(boundary_step),
                    "window_index": int(event["window_index"]),
                    "episode_index": episode_index,
                    "episode_offset_steps": int(reset["episode_offset_steps"]),
                    "schedule_seed": int(reset["schedule_seed"]),
                    "schedule_hash": reset["schedule_hash"],
                    "acquisition_completed_windows": int(
                        reset["acquisition_completed_windows"]
                    ),
                    "acquisition_total_windows": int(
                        reset["acquisition_total_windows"]
                    ),
                    "acquisition_counts": copy.deepcopy(
                        reset["acquisition_counts"]
                    ),
                    "action": copy.deepcopy(action),
                    "selection_source": "forced_acquisition",
                    "posterior_updated": True,
                }
                journal_events.append(acquisition_event)

            event.update(
                episode_index=episode_index,
                episode_start_step=int(reset["episode_start_step"]),
                episode_end_step=int(reset["episode_end_step"]),
                episode_offset_steps=int(reset["episode_offset_steps"]),
                acquisition_completed_windows=int(
                    reset["acquisition_completed_windows"]
                ),
                acquisition_total_windows=int(reset["acquisition_total_windows"]),
                acquisition_counts=copy.deepcopy(reset["acquisition_counts"]),
            )
            journal_events[0] = copy.deepcopy(event)
            reset_boundary = int(reset["episode_offset_steps"]) == int(
                reset["contract"]["interval_steps"]
            )
            if reset_boundary:
                completed_windows = int(reset["episode_offset_steps"]) // interval
                episode_archive = {
                    "episode_index": episode_index,
                    "episode_start_step": int(reset["episode_start_step"]),
                    "episode_end_step": int(boundary_step),
                    "episode_offset_steps": int(reset["episode_offset_steps"]),
                    "completed_window_count": completed_windows,
                    "forced_acquisition_window_count": int(
                        reset["acquisition_completed_windows"]
                    ),
                    "thompson_window_count": completed_windows
                    - int(reset["acquisition_completed_windows"]),
                    "acquisition_counts": copy.deepcopy(
                        reset["acquisition_counts"]
                    ),
                    "schedule_seed": int(reset["schedule_seed"]),
                    "schedule": list(reset["schedule"]),
                    "schedule_hash": reset["schedule_hash"],
                    "pre_reset_posterior_mean": copy.deepcopy(
                        belief["posterior_mean"]
                    ),
                    "pre_reset_posterior_covariance": copy.deepcopy(
                        belief["posterior_covariance"]
                    ),
                }
                reset["completed_episodes"].append(episode_archive)
                reset["completed_episode_count"] = len(
                    reset["completed_episodes"]
                )
                journal_events.append(
                    {
                        "schema_version": CONTROLLER_SCHEMA_VERSION,
                        "event_type": "episode_completed",
                        "boundary_step": int(boundary_step),
                        "window_index": int(event["window_index"]),
                        **copy.deepcopy(episode_archive),
                        "training_will_continue": bool(training_will_continue),
                    }
                )

                if training_will_continue:
                    pre_reset_mean = copy.deepcopy(belief["posterior_mean"])
                    pre_reset_covariance = copy.deepcopy(
                        belief["posterior_covariance"]
                    )
                    prior_mean = _float64_cpu_tensor(
                        self._state["probabilistic_inputs"]["resolved_prior_mean"],
                        field_name="configured reset prior mean",
                    )
                    prior_covariance = _float64_cpu_tensor(
                        self._state["probabilistic_inputs"]
                        ["resolved_prior_covariance"],
                        field_name="configured reset prior covariance",
                    )
                    belief.update(
                        posterior_mean=prior_mean,
                        posterior_covariance=prior_covariance,
                        predictive_mean=None,
                        predictive_covariance=None,
                    )
                    reset["reset_count"] = int(reset["reset_count"]) + 1
                    reset["reset_steps"].append(int(boundary_step))
                    journal_events.append(
                        {
                            "schema_version": CONTROLLER_SCHEMA_VERSION,
                            "event_type": "posterior_reset",
                            "boundary_step": int(boundary_step),
                            "window_index": int(next_window_index),
                            "episode_index": episode_index,
                            "reset_count": int(reset["reset_count"]),
                            "pre_reset_posterior_mean": pre_reset_mean,
                            "pre_reset_posterior_covariance": pre_reset_covariance,
                            "restored_prior_mean": copy.deepcopy(prior_mean),
                            "restored_prior_covariance": copy.deepcopy(
                                prior_covariance
                            ),
                            "policy": "full_prior",
                        }
                    )
                    next_action, episode_event = self._initialize_reset_episode(
                        episode_index=episode_index + 1,
                        boundary_step=boundary_step,
                    )
                    journal_events.append(episode_event)
                    window.update(
                        phase="active_window",
                        current_action=next_action,
                        selection_source=next_action["selection_source"],
                        boundary_evaluation_status="not_started",
                        terminal_status="continuing",
                    )
                    event["next_action"] = copy.deepcopy(next_action)

        if training_will_continue and not reset_boundary:
            if reset["enabled"] and int(
                reset["acquisition_completed_windows"]
            ) < int(reset["acquisition_total_windows"]):
                next_action = self._select_forced_acquisition(
                    boundary_step=boundary_step,
                    schedule_index=int(reset["acquisition_completed_windows"]),
                )
            else:
                next_action = self._predict_and_select(boundary_step=boundary_step)
            window.update(
                phase="active_window",
                current_action=next_action,
                selection_source=next_action["selection_source"],
                boundary_evaluation_status="not_started",
                terminal_status="continuing",
            )
            event["next_action"] = copy.deepcopy(next_action)
            if reset["enabled"]:
                reset["selection_source"] = next_action["selection_source"]
        journal_events[0] = copy.deepcopy(event)
        event["_journal_events"] = journal_events
        self._state["journal"]["event_count"] += len(journal_events)
        return event

    def fail(
        self,
        *,
        boundary_step: int,
        failing_stage: str,
        error_category: str,
        error_message: str,
        offending_field: str | None = None,
    ) -> dict[str, Any]:
        """Enter the failed phase without changing belief or sampling state."""

        window = self._state["window"]
        last_valid_phase = str(window["phase"])
        belief = self._state["belief"]
        belief_hash = stable_hash(
            {
                "round_index": belief["round_index"],
                "posterior_mean": belief["posterior_mean"].tolist(),
                "posterior_covariance": belief["posterior_covariance"].tolist(),
                "predictive_mean": (
                    None
                    if belief["predictive_mean"] is None
                    else belief["predictive_mean"].tolist()
                ),
                "predictive_covariance": (
                    None
                    if belief["predictive_covariance"] is None
                    else belief["predictive_covariance"].tolist()
                ),
            }
        )
        failure = {
            "stage": str(failing_stage),
            "error_category": str(error_category),
            "error_message": str(error_message)[:500],
            "offending_field": offending_field,
            "last_valid_phase": last_valid_phase,
            "belief_hash": belief_hash,
            "journal_position": int(self._state["journal"]["event_count"]),
            "posterior_updated": False,
            "new_action_selected": False,
        }
        window.update(
            phase="failed",
            boundary_evaluation_status="failed",
            terminal_status="failed",
        )
        self._state["failure"] = failure
        self._state["journal"]["event_count"] += 1
        return {
            "schema_version": CONTROLLER_SCHEMA_VERSION,
            "event_type": "controller_failure",
            "boundary_step": int(boundary_step),
            "window_index": int(window["window_index"]),
            "action": copy.deepcopy(window.get("current_action")),
            "failing_stage": failure["stage"],
            "error_category": failure["error_category"],
            "error_message": failure["error_message"],
            "offending_field": offending_field,
            "last_valid_phase": last_valid_phase,
            "belief_hash": belief_hash,
            "journal_position": failure["journal_position"],
            "posterior_updated": False,
            "new_action_selected": False,
            "sample_count": int(self._state["sampling"]["sample_count"]),
        }

    def finish_training(self) -> dict[str, Any] | None:
        window = self._state["window"]
        if window["phase"] == "active_window":
            window.update(
                phase="terminal_incomplete",
                terminal_status="incomplete",
            )
            self._state["journal"]["event_count"] += 1
            event = {
                "schema_version": CONTROLLER_SCHEMA_VERSION,
                "event_type": "terminal_incomplete",
                "window_index": int(window["window_index"]),
                "boundary_step": int(window["boundary_step"]),
                "completed_optimizer_steps": int(
                    window["completed_optimizer_steps"]
                ),
                "decision_interval_steps": int(
                    window["decision_interval_steps"]
                ),
                "action": copy.deepcopy(window["current_action"]),
                "pre_window_objective": float(window["pre_window_objective"]),
                "observation_emitted": False,
                "selection_source": window.get("selection_source"),
            }
            reset = self._state["reset"]
            if reset["enabled"]:
                event.update(
                    episode_index=reset.get("episode_index"),
                    episode_start_step=reset.get("episode_start_step"),
                    episode_end_step=reset.get("episode_end_step"),
                    episode_offset_steps=reset.get("episode_offset_steps"),
                    acquisition_completed_windows=reset.get(
                        "acquisition_completed_windows"
                    ),
                    acquisition_total_windows=reset.get(
                        "acquisition_total_windows"
                    ),
                    acquisition_counts=copy.deepcopy(
                        reset.get("acquisition_counts", {})
                    ),
                    episode_complete=False,
                )
            return event
        if window["phase"] == "boundary_evaluation_pending":
            raise ProbabilisticControllerError(
                "training cannot finish with a pending completed boundary"
            )
        return None


def build_probabilistic_controller(
    *,
    controller_config: Mapping[str, Any],
    sampling_seed: int,
    manifest_hashes: Mapping[str, str],
) -> ProbabilisticController:
    return ProbabilisticController(
        controller_config=controller_config,
        sampling_seed=sampling_seed,
        manifest_hashes=manifest_hashes,
    )


def restore_probabilistic_controller(
    saved_state: Mapping[str, Any] | None,
    *,
    controller_config: Mapping[str, Any],
    sampling_seed: int,
    expected_manifest_hashes: Mapping[str, str],
    source_checkpoint: str | Path,
    logger: Callable[[str], None] = print,
) -> ProbabilisticController:
    """Validate and restore a complete versioned Bayesian controller state."""

    if not isinstance(saved_state, Mapping):
        raise ProbabilisticControllerError(
            "missing Bayesian controller state in resume checkpoint"
        )
    controller = build_probabilistic_controller(
        controller_config=controller_config,
        sampling_seed=sampling_seed,
        manifest_hashes=expected_manifest_hashes,
    )
    expected = controller.state_dict()
    restored = copy.deepcopy(dict(saved_state))
    if restored.get("schema_version") == LEGACY_CONTROLLER_SCHEMA_VERSION:
        if expected["reset"]["enabled"]:
            raise ProbabilisticControllerError(
                "reset-enabled continuation requires complete reset state"
            )
        restored["schema_version"] = CONTROLLER_SCHEMA_VERSION
        restored["reset"] = copy.deepcopy(expected["reset"])
        legacy_window = restored.get("window")
        if isinstance(legacy_window, dict):
            legacy_action = legacy_window.get("current_action")
            legacy_window["selection_source"] = (
                legacy_action.get("selection_source", "thompson")
                if isinstance(legacy_action, Mapping)
                else None
            )
    if restored.get("schema_version") != CONTROLLER_SCHEMA_VERSION:
        raise ProbabilisticControllerError("controller schema version mismatch")
    if restored.get("method_family") != expected["method_family"]:
        raise ProbabilisticControllerError("controller method family mismatch")
    if restored.get("method_version") != expected["method_version"]:
        raise ProbabilisticControllerError("controller method version mismatch")
    if restored.get("scope") != expected["scope"]:
        raise ProbabilisticControllerError("controller scope mismatch")
    saved_schema = restored.get("feature_schema")
    if (
        not isinstance(saved_schema, Mapping)
        or dict(saved_schema) != expected["feature_schema"]
    ):
        raise ProbabilisticControllerError("controller feature schema mismatch")
    if restored.get("probabilistic_inputs") != expected["probabilistic_inputs"]:
        raise ProbabilisticControllerError(
            "controller probabilistic inputs mismatch"
        )
    reset = restored.get("reset")
    if not isinstance(reset, Mapping):
        raise ProbabilisticControllerError("controller reset state is missing")
    if reset.get("contract") != expected["reset"]["contract"]:
        raise ProbabilisticControllerError("controller reset contract mismatch")
    if bool(reset.get("enabled")) != bool(expected["reset"]["enabled"]):
        raise ProbabilisticControllerError("controller reset enabled state mismatch")
    if reset.get("enabled"):
        required_reset_fields = {
            "controller_start_step",
            "episode_index",
            "episode_start_step",
            "episode_end_step",
            "episode_offset_steps",
            "reset_count",
            "reset_steps",
            "acquisition_completed_windows",
            "acquisition_total_windows",
            "acquisition_counts",
            "selection_source",
            "schedule_seed",
            "schedule",
            "schedule_hash",
            "completed_episode_count",
            "completed_episodes",
        }
        missing_reset_fields = required_reset_fields - set(reset)
        if missing_reset_fields:
            raise ProbabilisticControllerError(
                "controller reset state is incomplete: "
                f"{sorted(missing_reset_fields)}"
            )
        if reset.get("episode_index") is not None:
            expected_schedule, expected_seed, expected_hash = (
                build_controller_reset_schedule(
                    expected["ordered_granularities"],
                    acquisition_passes=int(
                        expected["reset"]["contract"]["acquisition_passes"]
                    ),
                    root_seed=int(
                        expected["reset"]["contract"]["schedule_seed"]
                    ),
                    episode_index=int(reset["episode_index"]),
                )
            )
            if (
                list(reset.get("schedule", [])) != expected_schedule
                or reset.get("schedule_seed") != expected_seed
                or reset.get("schedule_hash") != expected_hash
            ):
                raise ProbabilisticControllerError(
                    "controller reset acquisition schedule mismatch"
                )
    saved_manifests = restored.get("manifest_hashes")
    if not isinstance(saved_manifests, Mapping):
        raise ProbabilisticControllerError("controller manifest hashes are missing")
    for name, expected_hash in expected_manifest_hashes.items():
        if saved_manifests.get(name) != expected_hash:
            raise ProbabilisticControllerError(
                f"{name} mismatch: saved={saved_manifests.get(name)!r}, "
                f"expected={expected_hash!r}"
            )

    belief = restored.get("belief")
    if not isinstance(belief, Mapping):
        raise ProbabilisticControllerError("controller belief state is missing")
    try:
        posterior_mean, posterior_covariance = validate_gaussian_belief(
            belief.get("posterior_mean"),
            belief.get("posterior_covariance"),
            state_name="posterior belief",
        )
    except ProbabilisticControllerError as error:
        raise ProbabilisticControllerError(
            f"posterior belief dimension or numerical state is invalid: {error}"
        ) from error
    if posterior_mean.numel() != expected["feature_schema"]["dimension"]:
        raise ProbabilisticControllerError(
            "posterior belief dimension does not match feature schema"
        )
    belief = dict(belief)
    belief["posterior_mean"] = posterior_mean
    belief["posterior_covariance"] = posterior_covariance
    if belief.get("predictive_mean") is not None:
        predictive_mean, predictive_covariance = validate_gaussian_belief(
            belief["predictive_mean"],
            belief.get("predictive_covariance"),
            state_name="predictive belief",
        )
        if predictive_mean.numel() != posterior_mean.numel():
            raise ProbabilisticControllerError(
                "predictive belief dimension does not match posterior"
            )
        belief["predictive_mean"] = predictive_mean
        belief["predictive_covariance"] = predictive_covariance
    restored["belief"] = belief

    sampling = restored.get("sampling")
    if not isinstance(sampling, Mapping) or "generator_state" not in sampling:
        raise ProbabilisticControllerError(
            "controller sampling generator state is missing"
        )
    generator_state = sampling["generator_state"]
    if not torch.is_tensor(generator_state):
        raise ProbabilisticControllerError(
            "controller sampling generator state must be a tensor"
        )
    try:
        controller._generator.set_state(generator_state.cpu())
    except RuntimeError as error:
        raise ProbabilisticControllerError(
            "controller sampling generator state is invalid"
        ) from error
    restored["sampling"] = dict(sampling)
    restored["sampling"]["generator_state"] = controller._generator.get_state()

    window = restored.get("window")
    if not isinstance(window, Mapping) or window.get("phase") not in {
        "initial_objective_pending",
        "ready_for_action",
        "active_window",
        "boundary_evaluation_pending",
        "terminal_incomplete",
        "failed",
    }:
        raise ProbabilisticControllerError("controller window phase is invalid")
    if "selection_source" not in window:
        raise ProbabilisticControllerError(
            "controller window selection source is missing"
        )
    controller._state = restored
    resume = dict(controller._state.get("resume", {}))
    resume.update(
        resume_count=int(resume.get("resume_count", 0)) + 1,
        source_checkpoint=str(source_checkpoint),
        compatibility_status="compatible",
    )
    controller._state["resume"] = resume
    progress = int(window.get("completed_optimizer_steps", 0))
    interval = int(window.get("decision_interval_steps", 0))
    action = window.get("current_action")
    action_summary = None
    if isinstance(action, Mapping):
        if action.get("scope") == "per_block":
            action_summary = ",".join(action.get("block_granularities", []))
        else:
            action_summary = action.get("global_granularity")
    logger(
        "[probabilistic-controller-resume] "
        f"source_checkpoint={source_checkpoint} "
        f"restored_phase={window.get('phase')} "
        f"window_index={window.get('window_index')} "
        f"progress={progress}/{interval} "
        f"current_action={action_summary}"
    )
    return controller
