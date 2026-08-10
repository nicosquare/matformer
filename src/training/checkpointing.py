"""Config-driven training flow for MatFormer reproduction runs."""

from __future__ import annotations

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*args, **kwargs):
        return None

load_dotenv()

import copy
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.models.adaptive_sampler import (
    AdaptiveSamplerState,
    build_adaptive_reward_record,
    build_adaptive_sampler_state,
    coerce_adaptive_sampler_state,
    normalize_adaptive_sampler_state,
    summarize_adaptive_sampler_state,
    update_adaptive_sampler_state,
)
from src.models.granularity import resolved_granularity_artifact_fields
from src.training.distributed import (
    broadcast_object,
    should_write_shared_artifact,
)
from src.training.probabilistic_controller import (
    CONTROLLER_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    LEGACY_CONTROLLER_SCHEMA_VERSION,
    ProbabilisticControllerError,
    SAMPLING_FACTORIZATION_CONTRACT,
    build_additive_feature_schema,
    encode_additive_action,
    validate_gaussian_belief,
)
from src.utils.config import (
    BAYESIAN_CONTROLLER_METHOD_FAMILY,
    BAYESIAN_CONTROLLER_METHOD_VERSION,
    ConfigError,
    resolve_sampling_mode_from_config_sections,
)
from src.utils.heartbeats import heartbeat_stage
from src.utils.artifact_io import (
    artifact_errno,
    emit_artifact_event,
    remove_resolved_failure,
    resolved_artifact_io,
    retry_artifact_io,
)
from src.utils.metrics import (
    best_validation_metric_value,
    build_checkpoint_summary_fields,
)
from src.utils.reproducibility import (
    build_controller_reset_schedule,
    capture_rng_state,
    deterministic_runtime_settings,
    restore_rng_state,
)


PROBABILISTIC_CONTROLLER_PHASES = {
    "initial_objective_pending",
    "ready_for_action",
    "active_window",
    "boundary_evaluation_pending",
    "terminal_incomplete",
    "failed",
}


def uses_probabilistic_controller(config: Mapping[str, Any]) -> bool:
    model = config.get("model", {})
    return bool(
        isinstance(model, Mapping)
        and model.get("granularity_sampling_mode")
        in {"adaptive_global", "adaptive_per_block"}
        and model.get("adaptive_sampler_strategy") == "thompson"
    )


def validate_probabilistic_controller_checkpoint_state(
    controller_state: Mapping[str, Any] | None,
    *,
    config: Mapping[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the complete Bayesian state before saving or restoring it."""

    location = f" at {checkpoint_path}" if checkpoint_path is not None else ""
    if not isinstance(controller_state, Mapping):
        raise ConfigError(
            f"Checkpoint is missing Bayesian controller state{location}"
        )
    state = copy.deepcopy(dict(controller_state))
    if state.get("schema_version") == LEGACY_CONTROLLER_SCHEMA_VERSION:
        configured_reset = (
            config.get("model", {}).get("adaptive_controller", {}).get("reset", {})
            if isinstance(config, Mapping)
            else None
        )
        if not isinstance(configured_reset, Mapping) or bool(
            configured_reset.get("enabled", False)
        ):
            raise ConfigError(
                "reset-enabled continuation requires complete reset state"
                f"{location}"
            )
        granularities = list(state.get("ordered_granularities", []))
        state["schema_version"] = CONTROLLER_SCHEMA_VERSION
        state["reset"] = {
            "contract": copy.deepcopy(dict(configured_reset)),
            "enabled": False,
            "controller_start_step": None,
            "episode_index": None,
            "episode_start_step": None,
            "episode_end_step": None,
            "episode_offset_steps": 0,
            "reset_count": 0,
            "reset_steps": [],
            "acquisition_completed_windows": 0,
            "acquisition_total_windows": 0,
            "acquisition_counts": {label: 0 for label in granularities},
            "selection_source": None,
            "schedule_seed": None,
            "schedule": [],
            "schedule_hash": None,
            "completed_episode_count": 0,
            "completed_episodes": [],
        }
        legacy_window = state.get("window")
        if isinstance(legacy_window, dict):
            legacy_action = legacy_window.get("current_action")
            legacy_window["selection_source"] = (
                legacy_action.get("selection_source", "thompson")
                if isinstance(legacy_action, Mapping)
                else None
            )
    required_top_level = {
        "schema_version",
        "method_family",
        "method_version",
        "strategy",
        "scope",
        "ordered_granularities",
        "block_count",
        "feature_schema",
        "probabilistic_inputs",
        "manifest_hashes",
        "belief",
        "sampling",
        "reset",
        "window",
        "journal",
        "resume",
        "failure",
    }
    missing = required_top_level - set(state)
    if missing:
        raise ConfigError(
            "Bayesian controller checkpoint state is incomplete"
            f"{location}: {sorted(missing)}"
        )
    if state["schema_version"] != CONTROLLER_SCHEMA_VERSION:
        raise ConfigError(f"Bayesian controller schema version mismatch{location}")
    if state["method_family"] != BAYESIAN_CONTROLLER_METHOD_FAMILY:
        raise ConfigError(f"Bayesian controller method family mismatch{location}")
    if state["method_version"] != BAYESIAN_CONTROLLER_METHOD_VERSION:
        raise ConfigError(f"Bayesian controller method version mismatch{location}")
    if state["strategy"] != "thompson":
        raise ConfigError(f"Bayesian controller strategy mismatch{location}")
    if state["scope"] not in {"global", "per_block"}:
        raise ConfigError(f"Bayesian controller scope is invalid{location}")

    granularities = state["ordered_granularities"]
    if (
        not isinstance(granularities, list)
        or not granularities
        or any(not isinstance(label, str) or not label for label in granularities)
        or len(set(granularities)) != len(granularities)
    ):
        raise ConfigError(
            f"Bayesian controller ordered granularities are invalid{location}"
        )
    block_count = state["block_count"]
    if isinstance(block_count, bool) or not isinstance(block_count, int) or block_count <= 0:
        raise ConfigError(f"Bayesian controller block count is invalid{location}")

    feature_schema = state["feature_schema"]
    if not isinstance(feature_schema, Mapping):
        raise ConfigError(f"Bayesian controller feature schema is missing{location}")
    required_feature_fields = {
        "schema_version",
        "scope",
        "encoding",
        "dimension",
        "coefficient_names",
        "schema_hash",
    }
    missing_features = required_feature_fields - set(feature_schema)
    if missing_features:
        raise ConfigError(
            f"Bayesian controller feature schema is incomplete{location}: "
            f"{sorted(missing_features)}"
        )
    dimension = feature_schema["dimension"]
    coefficient_names = feature_schema.get("coefficient_names")
    expected_dimension = (
        len(granularities)
        if state["scope"] == "global"
        else 1 + block_count * (len(granularities) - 1)
    )
    if (
        feature_schema["schema_version"] != FEATURE_SCHEMA_VERSION
        or feature_schema["scope"] != state["scope"]
        or isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
        or dimension != expected_dimension
        or not isinstance(coefficient_names, list)
        or len(coefficient_names) != dimension
        or any(not isinstance(name, str) or not name for name in coefficient_names)
        or not isinstance(feature_schema["schema_hash"], str)
        or not feature_schema["schema_hash"]
    ):
        raise ConfigError(f"Bayesian controller feature schema is invalid{location}")
    if state["scope"] == "per_block":
        expected_feature_schema = build_additive_feature_schema(
            granularities,
            block_count=block_count,
        )
        if dict(feature_schema) != expected_feature_schema:
            raise ConfigError(
                "Bayesian additive feature schema coefficient identities or "
                f"hash are incompatible{location}"
            )

    probabilistic_inputs = state["probabilistic_inputs"]
    required_inputs = {
        "resolved_prior_mean",
        "resolved_prior_covariance",
        "observation_noise_variance",
        "resolved_process_noise_covariance",
        "transition_model",
        "context_model",
        "compute_weight",
        "switch_weight",
    }
    if not isinstance(probabilistic_inputs, Mapping) or (
        required_inputs - set(probabilistic_inputs)
    ):
        raise ConfigError(
            f"Bayesian controller probabilistic inputs are incomplete{location}"
        )
    try:
        prior_mean, _ = validate_gaussian_belief(
            probabilistic_inputs["resolved_prior_mean"],
            probabilistic_inputs["resolved_prior_covariance"],
            state_name="checkpoint prior",
        )
        _, process_covariance = validate_gaussian_belief(
            [0.0] * dimension,
            probabilistic_inputs["resolved_process_noise_covariance"],
            state_name="checkpoint process noise",
        )
    except ProbabilisticControllerError as error:
        raise ConfigError(f"Invalid Bayesian checkpoint inputs{location}: {error}") from error
    if prior_mean.numel() != dimension or tuple(process_covariance.shape) != (
        dimension,
        dimension,
    ):
        raise ConfigError(
            f"Bayesian controller probabilistic input dimension mismatch{location}"
        )
    observation_noise = probabilistic_inputs["observation_noise_variance"]
    compute_weight = probabilistic_inputs["compute_weight"]
    switch_weight = probabilistic_inputs["switch_weight"]
    numerical_values_are_valid = all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        for value in (observation_noise, compute_weight, switch_weight)
    )
    if (
        not numerical_values_are_valid
        or float(observation_noise) <= 0.0
        or probabilistic_inputs["transition_model"] != "identity"
        or probabilistic_inputs["context_model"] != "intercept_only"
        or float(compute_weight) != 0.0
        or float(switch_weight) != 0.0
    ):
        raise ConfigError(f"Bayesian controller probabilistic inputs are invalid{location}")

    manifests = state["manifest_hashes"]
    required_manifests = {
        "data_roles_manifest_hash",
        "optimizer_training_manifest_hash",
        "controller_manifest_hash",
        "ordinary_validation_manifest_hash",
        "final_holdout_manifest_hash",
    }
    if not isinstance(manifests, Mapping) or any(
        not isinstance(manifests.get(name), str) or not manifests.get(name)
        for name in required_manifests
    ):
        raise ConfigError(f"Bayesian controller manifest hashes are incomplete{location}")

    belief = state["belief"]
    required_belief_fields = {
        "round_index",
        "posterior_mean",
        "posterior_covariance",
        "predictive_mean",
        "predictive_covariance",
        "last_prediction_step",
        "last_update_step",
    }
    if (
        not isinstance(belief, Mapping)
        or required_belief_fields - set(belief)
        or isinstance(belief.get("round_index"), bool)
        or not isinstance(belief.get("round_index"), int)
        or belief["round_index"] < 0
        or any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
            for value in (
                belief.get("last_prediction_step"),
                belief.get("last_update_step"),
            )
        )
    ):
        raise ConfigError(f"Bayesian controller belief is missing{location}")
    try:
        posterior_mean, posterior_covariance = validate_gaussian_belief(
            belief.get("posterior_mean"),
            belief.get("posterior_covariance"),
            state_name="checkpoint posterior",
        )
        has_predictive_mean = belief.get("predictive_mean") is not None
        has_predictive_covariance = belief.get("predictive_covariance") is not None
        if has_predictive_mean != has_predictive_covariance:
            raise ProbabilisticControllerError(
                "checkpoint predictive mean and covariance must appear together"
            )
        if has_predictive_mean:
            predictive_mean, predictive_covariance = validate_gaussian_belief(
                belief.get("predictive_mean"),
                belief.get("predictive_covariance"),
                state_name="checkpoint predictive",
            )
            if predictive_mean.numel() != dimension:
                raise ProbabilisticControllerError(
                    "checkpoint predictive dimension must match feature schema"
                )
            belief = dict(belief)
            belief["predictive_mean"] = predictive_mean
            belief["predictive_covariance"] = predictive_covariance
    except ProbabilisticControllerError as error:
        raise ConfigError(f"Invalid Bayesian checkpoint belief{location}: {error}") from error
    if posterior_mean.numel() != dimension:
        raise ConfigError(f"Bayesian posterior dimension mismatch{location}")
    belief = dict(belief)
    belief["posterior_mean"] = posterior_mean
    belief["posterior_covariance"] = posterior_covariance
    state["belief"] = belief

    sampling = state["sampling"]
    required_sampling_fields = {
        "seed_stream_name",
        "resolved_seed",
        "generator_state",
        "sample_count",
        "factorization_contract",
    }
    if (
        not isinstance(sampling, Mapping)
        or required_sampling_fields - set(sampling)
        or sampling.get("seed_stream_name") != "posterior_sampling"
        or isinstance(sampling.get("resolved_seed"), bool)
        or not isinstance(sampling.get("resolved_seed"), int)
        or sampling.get("factorization_contract") != SAMPLING_FACTORIZATION_CONTRACT
        or not isinstance(sampling.get("generator_state"), torch.Tensor)
        or isinstance(sampling.get("sample_count"), bool)
        or not isinstance(sampling.get("sample_count"), int)
        or sampling["sample_count"] < 0
    ):
        raise ConfigError(f"Bayesian controller sampling state is invalid{location}")
    try:
        torch.Generator(device="cpu").set_state(sampling["generator_state"].cpu())
    except RuntimeError as error:
        raise ConfigError(
            f"Bayesian controller generator state is invalid{location}"
        ) from error

    reset = state["reset"]
    required_reset_fields = {
        "contract",
        "enabled",
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
    if (
        not isinstance(reset, Mapping)
        or required_reset_fields - set(reset)
        or not isinstance(reset.get("enabled"), bool)
    ):
        raise ConfigError(f"Bayesian controller reset state is incomplete{location}")
    if reset["enabled"]:
        if state["scope"] != "global":
            raise ConfigError(
                f"Bayesian controller reset state requires global scope{location}"
            )
        if (
            not isinstance(reset.get("reset_steps"), list)
            or not isinstance(reset.get("acquisition_counts"), Mapping)
            or not isinstance(reset.get("completed_episodes"), list)
        ):
            raise ConfigError(
                f"Bayesian controller reset collections are invalid{location}"
            )
        contract = reset.get("contract")
        if not isinstance(contract, Mapping) or not bool(contract.get("enabled")):
            raise ConfigError(
                f"Bayesian controller reset contract is invalid{location}"
            )
        if reset.get("episode_index") is None:
            if any(
                reset.get(name) is not None
                for name in (
                    "controller_start_step",
                    "episode_start_step",
                    "episode_end_step",
                    "schedule_seed",
                    "schedule_hash",
                )
            ) or list(reset.get("schedule", [])):
                raise ConfigError(
                    f"Bayesian controller uninitialized reset state is invalid{location}"
                )
        else:
            integer_fields = (
                "controller_start_step",
                "episode_index",
                "episode_start_step",
                "episode_end_step",
                "episode_offset_steps",
                "reset_count",
                "acquisition_completed_windows",
                "acquisition_total_windows",
                "completed_episode_count",
            )
            if any(
                isinstance(reset.get(name), bool)
                or not isinstance(reset.get(name), int)
                or int(reset[name]) < 0
                for name in integer_fields
            ):
                raise ConfigError(
                    f"Bayesian controller reset episode progress is invalid{location}"
                )
        if int(reset["episode_offset_steps"]) > int(contract["interval_steps"]):
            raise ConfigError(
                f"Bayesian controller reset episode offset is invalid{location}"
            )
        if reset.get("episode_index") is not None:
            expected_schedule, expected_seed, expected_hash = build_controller_reset_schedule(
                granularities,
                acquisition_passes=int(contract["acquisition_passes"]),
                root_seed=int(contract["schedule_seed"]),
                episode_index=int(reset["episode_index"]),
            )
            if (
                list(reset.get("schedule", [])) != expected_schedule
                or reset.get("schedule_seed") != expected_seed
                or reset.get("schedule_hash") != expected_hash
            ):
                raise ConfigError(
                    f"Bayesian controller reset schedule is incompatible{location}"
                )

    window = state["window"]
    required_window_fields = {
        "phase",
        "window_index",
        "decision_interval_steps",
        "boundary_step",
        "current_action",
        "selection_source",
        "completed_optimizer_steps",
        "pre_window_objective",
        "ordered_pre_window_component_losses",
        "boundary_evaluation_status",
        "terminal_status",
    }
    if (
        not isinstance(window, Mapping)
        or required_window_fields - set(window)
        or window.get("phase") not in PROBABILISTIC_CONTROLLER_PHASES
        or isinstance(window.get("window_index"), bool)
        or not isinstance(window.get("window_index"), int)
        or window["window_index"] < 0
        or isinstance(window.get("boundary_step"), bool)
        or not isinstance(window.get("boundary_step"), int)
        or window["boundary_step"] < 0
    ):
        raise ConfigError(f"Bayesian controller window phase is invalid{location}")
    interval = window.get("decision_interval_steps")
    progress = window.get("completed_optimizer_steps")
    if (
        isinstance(interval, bool)
        or not isinstance(interval, int)
        or interval <= 0
        or isinstance(progress, bool)
        or not isinstance(progress, int)
        or progress < 0
        or progress > interval
    ):
        raise ConfigError(f"Bayesian controller window progress is invalid{location}")
    phase = window["phase"]
    action = window.get("current_action")
    if phase in {"initial_objective_pending", "ready_for_action"} and action is not None:
        raise ConfigError(f"Bayesian controller phase cannot have an action{location}")
    if phase in {"active_window", "boundary_evaluation_pending", "terminal_incomplete"} and not isinstance(action, Mapping):
        raise ConfigError(f"Bayesian controller phase requires an action{location}")
    if phase == "boundary_evaluation_pending" and progress != interval:
        raise ConfigError(f"Bayesian boundary checkpoint requires full progress{location}")
    if phase in {"active_window", "terminal_incomplete"} and progress >= interval:
        raise ConfigError(f"Bayesian active/incomplete progress must be below interval{location}")
    if isinstance(action, Mapping) and window.get("selection_source") not in {
        "forced_acquisition",
        "thompson",
    }:
        raise ConfigError(
            f"Bayesian controller selection source is invalid{location}"
        )
    if reset["enabled"] and reset.get("episode_index") is not None:
        reset_interval = int(reset["contract"]["interval_steps"])
        expected_episode_start = int(reset["controller_start_step"]) + int(
            reset["episode_index"]
        ) * reset_interval
        if (
            int(reset["episode_start_step"]) != expected_episode_start
            or int(reset["episode_end_step"])
            != expected_episode_start + reset_interval
            or int(reset["reset_count"]) != len(reset["reset_steps"])
            or int(reset["completed_episode_count"])
            != len(reset["completed_episodes"])
            or int(reset["acquisition_completed_windows"])
            > int(reset["acquisition_total_windows"])
            or sum(int(value) for value in reset["acquisition_counts"].values())
            != int(reset["acquisition_completed_windows"])
        ):
            raise ConfigError(
                f"Bayesian controller reset episode provenance is invalid{location}"
            )
        expected_episode_offset = (
            int(window["boundary_step"])
            - int(reset["episode_start_step"])
            + int(window["completed_optimizer_steps"])
        )
        if expected_episode_offset != int(reset["episode_offset_steps"]):
            raise ConfigError(
                f"Bayesian controller reset episode offset does not match window{location}"
            )
        if isinstance(action, Mapping) and (
            window.get("selection_source") != reset.get("selection_source")
            or action.get("selection_source") != window.get("selection_source")
        ):
            raise ConfigError(
                f"Bayesian controller reset selection source mismatch{location}"
            )
    if phase == "failed" and not isinstance(state["failure"], Mapping):
        raise ConfigError(f"Bayesian failed checkpoint requires failure provenance{location}")
    if phase != "failed" and state["failure"] is not None:
        raise ConfigError(
            f"Bayesian non-failed checkpoint cannot contain failure provenance{location}"
        )
    failed_before_initial_objective = bool(
        phase == "failed"
        and isinstance(state.get("failure"), Mapping)
        and state["failure"].get("last_valid_phase")
        == "initial_objective_pending"
    )
    if phase == "initial_objective_pending" or failed_before_initial_objective:
        if (
            window["pre_window_objective"] is not None
            or window["ordered_pre_window_component_losses"] is not None
        ):
            raise ConfigError(
                f"Bayesian initial checkpoint cannot contain an objective{location}"
            )
    else:
        objective = window["pre_window_objective"]
        component_losses = window["ordered_pre_window_component_losses"]
        if (
            isinstance(objective, bool)
            or not isinstance(objective, (int, float))
            or not math.isfinite(float(objective))
            or not isinstance(component_losses, list)
            or len(component_losses) != len(granularities)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in component_losses
            )
        ):
            raise ConfigError(
                f"Bayesian controller objective provenance is invalid{location}"
            )

    if isinstance(action, Mapping):
        feature_vector = action.get("feature_vector")
        if (
            action.get("scope") != state["scope"]
            or not isinstance(feature_vector, (list, tuple, torch.Tensor))
            or len(feature_vector) != dimension
            or not bool(torch.isfinite(torch.as_tensor(feature_vector)).all())
        ):
            raise ConfigError(f"Bayesian controller action is invalid{location}")
        if state["scope"] == "global":
            selected_label = action.get("global_granularity")
            if (
                selected_label not in granularities
                or action.get("block_granularities")
                != [selected_label] * block_count
            ):
                raise ConfigError(
                    f"Bayesian global controller action is invalid{location}"
                )
        else:
            profile = action.get("block_granularities")
            if (
                action.get("global_granularity") is not None
                or not isinstance(profile, list)
                or len(profile) != block_count
                or any(label not in granularities for label in profile)
            ):
                raise ConfigError(
                    f"Bayesian per-block controller action is invalid{location}"
                )
            expected_feature_vector = encode_additive_action(
                feature_schema,
                profile,
            )
            if not torch.equal(
                torch.as_tensor(feature_vector, dtype=torch.float64, device="cpu"),
                expected_feature_vector,
            ):
                raise ConfigError(
                    "Bayesian per-block action feature vector does not match its "
                    f"complete profile{location}"
                )

    journal = state["journal"]
    if (
        not isinstance(journal, Mapping)
        or not isinstance(journal.get("path"), str)
        or not journal["path"]
        or isinstance(journal.get("event_count"), bool)
        or not isinstance(journal.get("event_count"), int)
        or journal["event_count"] < 0
        or (
            journal.get("last_committed_offset") is not None
            and (
                isinstance(journal["last_committed_offset"], bool)
                or not isinstance(journal["last_committed_offset"], int)
                or journal["last_committed_offset"] < 0
            )
        )
        or (
            journal.get("last_committed_hash") is not None
            and not isinstance(journal["last_committed_hash"], str)
        )
    ):
        raise ConfigError(f"Bayesian controller journal provenance is invalid{location}")
    resume = state["resume"]
    if (
        not isinstance(resume, Mapping)
        or isinstance(resume.get("resume_count"), bool)
        or not isinstance(resume.get("resume_count"), int)
        or resume["resume_count"] < 0
        or (
            resume.get("source_checkpoint") is not None
            and not isinstance(resume["source_checkpoint"], str)
        )
        or not isinstance(resume.get("compatibility_status"), str)
        or not resume["compatibility_status"]
    ):
        raise ConfigError(f"Bayesian controller resume provenance is invalid{location}")

    if config is not None:
        model = config.get("model", {})
        controller_config = model.get("adaptive_controller", {})
        expected_scope = controller_config.get("scope")
        expected_granularities = list(model.get("granularities", []))
        expected_manifests = {
            "data_roles_manifest_hash": config.get("data_roles_manifest_hash"),
            "optimizer_training_manifest_hash": config.get("optimizer_training_manifest_hash"),
            "controller_manifest_hash": config.get("controller_manifest_hash"),
            "ordinary_validation_manifest_hash": config.get("validation_manifest_hash"),
            "final_holdout_manifest_hash": config.get("final_holdout_manifest_hash"),
        }
        if state["scope"] != expected_scope:
            raise ConfigError(f"Bayesian controller scope does not match config{location}")
        if granularities != expected_granularities:
            raise ConfigError(
                f"Bayesian controller granularity order does not match config{location}"
            )
        if block_count != int(model.get("num_layers", 0)):
            raise ConfigError(
                f"Bayesian controller block count does not match config{location}"
            )
        for name, expected_hash in expected_manifests.items():
            if expected_hash is not None and manifests[name] != expected_hash:
                raise ConfigError(f"Bayesian controller {name} mismatch{location}")
        expected_inputs = {
            "resolved_prior_mean": controller_config.get("resolved_prior_mean"),
            "resolved_prior_covariance": controller_config.get("resolved_prior_covariance"),
            "observation_noise_variance": controller_config.get("observation_noise_variance"),
            "resolved_process_noise_covariance": controller_config.get("resolved_process_noise_covariance"),
            "transition_model": controller_config.get("transition_model"),
            "context_model": controller_config.get("context_model"),
            "compute_weight": controller_config.get("compute_weight"),
            "switch_weight": controller_config.get("switch_weight"),
        }
        if dict(probabilistic_inputs) != expected_inputs:
            raise ConfigError(
                f"Bayesian controller probabilistic inputs do not match config{location}"
            )
        expected_reset = controller_config.get("reset")
        if not isinstance(expected_reset, Mapping) or dict(
            reset.get("contract", {})
        ) != dict(expected_reset):
            raise ConfigError(
                f"Bayesian controller reset contract does not match config{location}"
            )
    return state

def continuation_latest_checkpoint_policy(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    run = config.get("run", {})
    continuation = run.get("continuation", {})
    if not isinstance(continuation, Mapping):
        continuation = {}

    enabled = bool(continuation.get("enabled", False))
    interval_steps = continuation.get("latest_checkpoint_save_interval_steps", 1)
    if interval_steps is None:
        interval_steps = 0
    interval_steps = int(interval_steps)
    if interval_steps < 0:
        raise ConfigError(
            "run.continuation.latest_checkpoint_save_interval_steps must be non-negative"
        )

    return {
        "enabled": enabled,
        "retain_previous_latest": bool(
            continuation.get("retain_previous_latest", True)
        ),
        "save_interval_steps": interval_steps,
        "save_on_validation": bool(
            continuation.get("latest_checkpoint_save_on_validation", False)
        ),
        "save_on_completion": bool(
            continuation.get("latest_checkpoint_save_on_completion", True)
        ),
    }


def should_save_latest_checkpoint(
    config: Mapping[str, Any],
    step: int,
    reason: str,
) -> bool:
    policy = continuation_latest_checkpoint_policy(config)
    if not policy["enabled"]:
        return False

    if reason == "validation":
        return policy["save_on_validation"]
    if reason == "completion":
        return policy["save_on_completion"]
    if reason == "failure":
        return True

    interval_steps = policy["save_interval_steps"]
    return interval_steps > 0 and step > 0 and step % interval_steps == 0


def maybe_write_latest_checkpoint(
    config: dict[str, Any],
    model,
    optimizer,
    scheduler,
    heartbeat_writer,
    run_state: dict[str, Any],
    reason: str,
    step: int,
    distributed_context=None,
    force: bool = False,
) -> None:
    pending_retry = bool(run_state.get("pending_latest_checkpoint", False))
    if not force and not pending_retry and not should_save_latest_checkpoint(config, step, reason):
        return
    if not force and int(run_state.get("latest_checkpoint_step", 0)) == int(step):
        return

    latest_checkpoint_path = Path(
        run_state.get("latest_checkpoint_path")
        or Path(config["run"]["output_dir"]) / "checkpoints" / "latest.pt"
    )
    checkpoint_fields = {
        "checkpoint_status": "latest",
        "checkpoint_metric": None,
        "checkpoint_metric_value": None,
        "checkpoint_selection_step": None,
        "checkpoint_unavailable_reason": None,
    }

    save_error: Exception | None = None
    try:
        with heartbeat_stage(
            heartbeat_writer,
            "checkpointing",
            checkpoint_status="latest",
            checkpoint_reason=reason,
            pending_retry=pending_retry,
        ):
            save_model_checkpoint(
                config,
                model,
                optimizer,
                scheduler,
                latest_checkpoint_path,
                checkpoint_fields,
                run_state,
                distributed_context=distributed_context,
                heartbeat_writer=heartbeat_writer,
            )
    except Exception as error:
        save_error = error

    if should_write_shared_artifact(distributed_context):
        result = {
            "error": str(save_error) if save_error is not None else None,
            "errno": artifact_errno(save_error) if save_error is not None else None,
            "artifact_state": _artifact_state_fields(run_state),
        }
    else:
        result = None
    result = broadcast_object(result, distributed_context)
    if result:
        run_state.update(result.get("artifact_state", {}))
    if result and result["error"] is not None:
        if _can_defer_periodic_checkpoint(
            config,
            reason=reason,
            latest_checkpoint_path=latest_checkpoint_path,
        ):
            run_state["pending_latest_checkpoint"] = True
            run_state["skipped_periodic_checkpoints"] = int(
                run_state.get("skipped_periodic_checkpoints", 0)
            ) + 1
            emit_artifact_event(
                heartbeat_writer,
                "stage_failed",
                "checkpointing",
                checkpoint_status="latest",
                checkpoint_reason=reason,
                checkpoint_pending=True,
                skipped_periodic_checkpoints=run_state[
                    "skipped_periodic_checkpoints"
                ],
                last_durable_checkpoint_step=run_state.get(
                    "last_durable_checkpoint_step"
                ),
                errno=result["errno"],
                error=result["error"],
                recoverable=True,
            )
            return
        raise OSError(
            result["errno"],
            result["error"],
            str(latest_checkpoint_path),
        ) from save_error

    run_state["latest_checkpoint_path"] = str(latest_checkpoint_path)
    run_state["latest_checkpoint_step"] = step
    run_state["last_durable_checkpoint_step"] = step
    run_state["pending_latest_checkpoint"] = False


def _can_defer_periodic_checkpoint(
    config: Mapping[str, Any],
    *,
    reason: str,
    latest_checkpoint_path: Path,
) -> bool:
    artifact_io = resolved_artifact_io(config)
    if artifact_io["periodic_checkpoint_failure_policy"] != "continue_if_previous":
        return False
    if reason not in {"step", "validation"}:
        return False
    return latest_checkpoint_path.exists() or latest_checkpoint_path.with_name(
        "latest.prev.pt"
    ).exists()


def _artifact_state_fields(run_state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(run_state.get(key))
        for key in (
            "artifact_retry_count",
            "artifact_last_errno",
            "last_durable_checkpoint_step",
            "deferred_metric_rows",
            "skipped_periodic_checkpoints",
            "checkpoint_staging_mode",
            "pending_latest_checkpoint",
            "pending_best_checkpoint",
            "unresolved_artifact_failures",
        )
    }


def write_checkpoint_if_needed(
    config: dict[str, Any],
    model,
    optimizer,
    scheduler,
    metrics_rows: list[dict[str, Any]],
    heartbeat_writer,
    run_state: dict[str, Any],
    distributed_context=None,
) -> dict[str, Any]:
    if should_write_shared_artifact(distributed_context):
        checkpoint_fields = build_checkpoint_summary_fields(config, metrics_rows)
        checkpoint_path = checkpoint_fields.get("best_checkpoint_path")
        if checkpoint_path is None:
            checkpoint_path = checkpoint_fields.get("final_checkpoint_path")

        should_save = False
        if checkpoint_path is not None:
            output_path = Path(str(checkpoint_path))
            should_save = not output_path.exists()

        payload = {
            "checkpoint_fields": checkpoint_fields,
            "checkpoint_path": checkpoint_path,
            "should_save": should_save,
        }
    else:
        payload = None

    payload = broadcast_object(payload, distributed_context)
    if payload is None:
        return build_checkpoint_summary_fields(config, metrics_rows)

    checkpoint_fields = payload["checkpoint_fields"]
    checkpoint_path = payload["checkpoint_path"]
    if checkpoint_path is None or not payload["should_save"]:
        return checkpoint_fields

    output_path = Path(str(checkpoint_path))

    save_error: Exception | None = None
    try:
        with heartbeat_stage(
            heartbeat_writer,
            "checkpointing",
            checkpoint_status=checkpoint_fields["checkpoint_status"],
        ):
            save_model_checkpoint(
                config,
                model,
                optimizer,
                scheduler,
                output_path,
                checkpoint_fields,
                run_state,
                distributed_context=distributed_context,
                heartbeat_writer=heartbeat_writer,
            )
    except Exception as error:
        save_error = error
    if should_write_shared_artifact(distributed_context):
        save_result = {
            "error": str(save_error) if save_error is not None else None,
            "errno": artifact_errno(save_error) if save_error is not None else None,
            "artifact_state": _artifact_state_fields(run_state),
        }
    else:
        save_result = None
    save_result = broadcast_object(save_result, distributed_context)
    if save_result:
        run_state.update(save_result.get("artifact_state", {}))
    if save_result and save_result["error"] is not None:
        raise OSError(
            save_result["errno"],
            save_result["error"],
            str(output_path),
        ) from save_error

    return checkpoint_fields


def maybe_write_best_eval_checkpoint(
    config: dict[str, Any],
    model,
    validation_results: list[dict[str, Any]],
    step: int,
    heartbeat_writer,
    checkpoint_state: dict[str, Any],
    run_state: dict[str, Any],
    distributed_context=None,
) -> None:
    if not config.get("outputs", {}).get("save_checkpoints", False):
        return
    evaluation = config.get("evaluation", {})
    if not (
        evaluation.get("validation", {}).get("enabled", False)
        or evaluation.get("validation", {}).get("run_at_completion", False)
    ):
        return

    if should_write_shared_artifact(distributed_context):
        payload = build_best_eval_checkpoint_payload(
            config,
            validation_results,
            step,
            checkpoint_state,
        )
    else:
        payload = None

    payload = broadcast_object(payload, distributed_context)
    if payload is None or not payload["should_save"]:
        return

    checkpoint_fields = payload["checkpoint_fields"]
    checkpoint_path = Path(str(checkpoint_fields["best_checkpoint_path"]))

    save_error: Exception | None = None
    try:
        with heartbeat_stage(
            heartbeat_writer,
            "checkpointing",
            checkpoint_status=checkpoint_fields["checkpoint_status"],
        ):
            save_model_checkpoint(
                config,
                model,
                None,
                None,
                checkpoint_path,
                checkpoint_fields,
                run_state,
                distributed_context=distributed_context,
                heartbeat_writer=heartbeat_writer,
            )
    except Exception as error:
        save_error = error

    if should_write_shared_artifact(distributed_context):
        save_result = {
            "error": str(save_error) if save_error is not None else None,
            "errno": artifact_errno(save_error) if save_error is not None else None,
            "artifact_state": _artifact_state_fields(run_state),
        }
    else:
        save_result = None
    save_result = broadcast_object(save_result, distributed_context)
    if save_result:
        run_state.update(save_result.get("artifact_state", {}))
    if save_result and save_result["error"] is not None:
        run_state["pending_best_checkpoint"] = {
            "path": str(checkpoint_path),
            "step": step,
            "metric": checkpoint_fields.get("checkpoint_metric"),
            "metric_value": checkpoint_fields.get("checkpoint_metric_value"),
        }
        emit_artifact_event(
            heartbeat_writer,
            "stage_failed",
            "checkpointing",
            checkpoint_status="best_eval",
            checkpoint_pending=True,
            errno=save_result["errno"],
            error=save_result["error"],
            recoverable=True,
        )
        return

    checkpoint_state.update(checkpoint_fields)
    run_state.update(checkpoint_fields)
    run_state["pending_best_checkpoint"] = None
    _prune_best_eval_checkpoints(
        checkpoint_path.parent,
        retain_count=int(
            config.get("outputs", {}).get("best_eval_retention_count", 1)
        ),
    )
def build_best_eval_checkpoint_payload(
    config: dict[str, Any],
    validation_results: list[dict[str, Any]],
    step: int,
    checkpoint_state: dict[str, Any],
) -> dict[str, Any]:
    metric_name, metric_value = best_validation_metric_value(validation_results)
    if metric_name is None or metric_value is None:
        return {"should_save": False, "checkpoint_fields": None}

    previous_metric_value = checkpoint_state.get("checkpoint_metric_value")
    if previous_metric_value is not None and metric_value >= previous_metric_value:
        return {"should_save": False, "checkpoint_fields": None}

    metric_field = "loss" if metric_name == "validation_loss" else "perplexity"
    best_result = min(
        (
            result
            for result in validation_results
            if result.get(metric_field) is not None
        ),
        key=lambda result: float(result[metric_field]),
    )
    metric_row = {
        "split": "validation",
        "step": step,
        "granularity": best_result.get("granularity"),
        metric_field: metric_value,
    }
    checkpoint_fields = build_checkpoint_summary_fields(
        config,
        [metric_row],
        validation_enabled=True,
        save_checkpoints=True,
    )
    return {"should_save": True, "checkpoint_fields": checkpoint_fields}


def save_model_checkpoint(
    config: dict[str, Any],
    model,
    optimizer,
    scheduler,
    output_path: Path,
    checkpoint_fields: dict[str, Any],
    run_state: dict[str, Any],
    distributed_context=None,
    heartbeat_writer=None,
) -> None:
    if not should_write_shared_artifact(distributed_context):
        return

    model_state_dict, optimizer_state_dict = checkpoint_state_dicts(
        model,
        optimizer,
        distributed_context,
    )

    probabilistic_controller_state = run_state.get(
        "probabilistic_controller_state"
    )
    if uses_probabilistic_controller(config):
        probabilistic_controller_state = (
            validate_probabilistic_controller_checkpoint_state(
                probabilistic_controller_state,
                config=config,
                checkpoint_path=output_path,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
            "run_id": config["run"]["run_id"],
            "reproducibility": {
                "root_seed": config["run"]["seed"],
                "seed_stream_version": config["run"]["reproducibility"][
                    "seed_stream_version"
                ],
                "data_split_version": config["run"]["reproducibility"][
                    "data_split_version"
                ],
                "validation_manifest_hash": config.get(
                    "validation_manifest_hash"
                ),
                "comparison_control_signature": config.get(
                    "comparison_control_signature"
                ),
                "batch_size_per_process": config["training"][
                    "batch_size_per_process"
                ],
                "world_size": int(
                    getattr(distributed_context, "world_size", 1)
                ),
                "rank_topology": list(
                    range(int(getattr(distributed_context, "world_size", 1)))
                ),
                "rng_state": capture_rng_state(),
                "deterministic_runtime_settings": deterministic_runtime_settings(),
                "adaptive_sampler_seed_provenance": "adaptive_sampling",
            },
            "checkpoint_status": checkpoint_fields["checkpoint_status"],
            "checkpoint_metric": checkpoint_fields["checkpoint_metric"],
            "checkpoint_metric_value": checkpoint_fields[
                "checkpoint_metric_value"
            ],
            "checkpoint_selection_step": checkpoint_fields[
                "checkpoint_selection_step"
            ],
            **resolved_granularity_artifact_fields(config.get("model", {})),
            "step": run_state.get("step", run_state.get("last_completed_step", 0)),
            "epoch": run_state.get("epoch", 0),
            "batch_index": run_state.get("batch_index", 0),
            "tokens_seen": run_state.get("tokens_seen", 0),
            "content_tokens_seen": run_state.get("content_tokens_seen", 0),
            "resume_count": run_state.get("resume_count", 0),
            "warmup_completed": run_state.get("warmup_completed", False),
            "warmup_completion_step": run_state.get("warmup_completion_step"),
            "warmup_transition_reason": run_state.get("warmup_transition_reason"),
            "pre_nested_warmup_state": copy.deepcopy(
                run_state.get("pre_nested_warmup_state")
            ),
            "resolved_run_mode": run_state.get("resolved_run_mode"),
            "resolved_sampling_mode": run_state.get("resolved_sampling_mode"),
            "granularity_pattern_provenance": run_state.get(
                "granularity_pattern_provenance"
            ),
            "adaptive_sampler_state": run_state.get("adaptive_sampler_state"),
            "probabilistic_controller_state": probabilistic_controller_state,
            "adaptive_sampler_previous_loss": run_state.get(
                "adaptive_sampler_previous_loss"
            ),
            "adaptive_sampler_previous_pattern": run_state.get(
                "adaptive_sampler_previous_pattern"
            ),
            "adaptive_reward_summary": run_state.get("adaptive_reward_summary"),
            "adaptive_correction_penalty_summary": run_state.get(
                "adaptive_correction_penalty_summary"
            ),
            "adaptive_sampler_strategy": run_state.get(
                "adaptive_sampler_strategy"
            ),
            "adaptive_sampler_exploration_scale": run_state.get(
                "adaptive_sampler_exploration_scale"
            ),
            "adaptive_sampler_decay_rate": run_state.get(
                "adaptive_sampler_decay_rate"
            ),
            "adaptive_sampler_reward_penalty_weight": run_state.get(
                "adaptive_sampler_reward_penalty_weight"
            ),
            "latest_checkpoint_path": run_state.get("latest_checkpoint_path"),
            "artifact_retry_count": run_state.get("artifact_retry_count", 0),
            "artifact_last_errno": run_state.get("artifact_last_errno"),
            "last_durable_checkpoint_step": run_state.get(
                "step", run_state.get("last_completed_step", 0)
            )
            if output_path.name == "latest.pt"
            else run_state.get(
                "last_durable_checkpoint_step", 0
            ),
            "deferred_metric_rows": run_state.get("deferred_metric_rows", 0),
            "skipped_periodic_checkpoints": run_state.get(
                "skipped_periodic_checkpoints", 0
            ),
            "checkpoint_staging_mode": run_state.get(
                "checkpoint_staging_mode", "direct"
            ),
            "unresolved_artifact_failures": run_state.get(
                "unresolved_artifact_failures", []
            ),
            "best_checkpoint_path": checkpoint_fields.get(
                "best_checkpoint_path"
            )
            or run_state.get("best_checkpoint_path"),
            "best_checkpoint_metric": (
                checkpoint_fields.get("checkpoint_metric")
                if checkpoint_fields.get("checkpoint_status") == "best_eval"
                else run_state.get("checkpoint_metric")
            ),
            "best_checkpoint_metric_value": (
                checkpoint_fields.get("checkpoint_metric_value")
                if checkpoint_fields.get("checkpoint_status") == "best_eval"
                else run_state.get("checkpoint_metric_value")
            ),
            "best_checkpoint_selection_step": (
                checkpoint_fields.get("checkpoint_selection_step")
                if checkpoint_fields.get("checkpoint_status") == "best_eval"
                else run_state.get("checkpoint_selection_step")
            ),
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        }

    artifact_io = resolved_artifact_io(config)
    staging_path = _stage_checkpoint_payload(
        payload,
        output_path=output_path,
        artifact_io=artifact_io,
        heartbeat_writer=heartbeat_writer,
        run_state=run_state,
    )
    installed = False
    rotated = False

    def install_attempt(_attempt: int) -> Path:
        nonlocal installed, rotated
        if installed:
            _fsync_directory(output_path.parent)
            return output_path

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                if staging_path is None:
                    torch.save(payload, temporary_file)
                else:
                    with staging_path.open("rb") as staged_file:
                        shutil.copyfileobj(staged_file, temporary_file)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            if (
                not rotated
                and output_path.name == "latest.pt"
                and output_path.exists()
                and config.get("run", {})
                .get("continuation", {})
                .get("retain_previous_latest", True)
                and not str(
                    run_state.get("continuation_source_checkpoint_path") or ""
                ).endswith("latest.prev.pt")
            ):
                os.replace(
                    output_path,
                    output_path.with_name("latest.prev.pt"),
                )
                rotated = True
            os.replace(temporary_path, output_path)
            temporary_path = None
            installed = True
            _fsync_directory(output_path.parent)
            return output_path
        finally:
            if temporary_path is not None:
                _unlink_best_effort(temporary_path)

    try:
        retry_artifact_io(
            install_attempt,
            target_path=output_path,
            operation_name="checkpoint_install",
            settings=artifact_io,
            heartbeat_writer=heartbeat_writer,
            state=run_state,
        )
    finally:
        if staging_path is not None:
            _unlink_best_effort(staging_path)

    if output_path.name == "latest.pt":
        run_state["continuation_source_checkpoint_path"] = str(output_path)
        run_state["last_durable_checkpoint_step"] = int(
            run_state.get("step", run_state.get("last_completed_step", 0))
        )
    run_state["pending_latest_checkpoint"] = False
    remove_resolved_failure(
        run_state,
        operation_name="checkpoint_install",
        target_path=output_path,
    )


def _stage_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    output_path: Path,
    artifact_io: Mapping[str, Any],
    heartbeat_writer,
    run_state: dict[str, Any],
) -> Path | None:
    requested_mode = str(artifact_io.get("checkpoint_staging", "auto"))
    local_root = os.environ.get("SLURM_TMPDIR")
    if requested_mode == "direct" or not local_root:
        run_state["checkpoint_staging_mode"] = "direct"
        if isinstance(payload, dict):
            payload["checkpoint_staging_mode"] = "direct"
        return None

    staging_dir = Path(local_root)
    estimated_size = _estimate_payload_bytes(payload)
    try:
        usable = staging_dir.is_dir() and os.access(staging_dir, os.W_OK)
        enough_space = shutil.disk_usage(staging_dir).free >= max(
            estimated_size * 2,
            64 * 1024 * 1024,
        )
    except OSError:
        usable = False
        enough_space = False
    if not usable or not enough_space:
        run_state["checkpoint_staging_mode"] = "direct"
        if isinstance(payload, dict):
            payload["checkpoint_staging_mode"] = "direct"
        return None

    run_state["checkpoint_staging_mode"] = "slurm_tmpdir"
    if isinstance(payload, dict):
        payload["checkpoint_staging_mode"] = "slurm_tmpdir"

    def stage_attempt(_attempt: int) -> Path:
        staged_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=staging_dir,
                prefix=f"{output_path.name}.",
                suffix=".staged",
                delete=False,
            ) as staged_file:
                staged_path = Path(staged_file.name)
                torch.save(payload, staged_file)
                staged_file.flush()
                os.fsync(staged_file.fileno())
            return staged_path
        except Exception:
            if staged_path is not None:
                _unlink_best_effort(staged_path)
            raise

    try:
        staged_path = retry_artifact_io(
            stage_attempt,
            target_path=output_path,
            operation_name="checkpoint_stage",
            settings=artifact_io,
            heartbeat_writer=heartbeat_writer,
            state=run_state,
        )
    except OSError as error:
        emit_artifact_event(
            heartbeat_writer,
            "stage_failed",
            "checkpoint_stage",
            artifact_path=str(output_path),
            errno=error.errno,
            fallback="direct",
        )
        remove_resolved_failure(
            run_state,
            operation_name="checkpoint_stage",
            target_path=output_path,
        )
        run_state["checkpoint_staging_mode"] = "direct"
        if isinstance(payload, dict):
            payload["checkpoint_staging_mode"] = "direct"
        return None

    emit_artifact_event(
        heartbeat_writer,
        "stage_complete",
        "checkpoint_stage",
        artifact_path=str(output_path),
        staging_mode="slurm_tmpdir",
        estimated_bytes=estimated_size,
    )
    return staged_path


def _estimate_payload_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel()) * int(value.element_size())
    if isinstance(value, Mapping):
        return sum(_estimate_payload_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_estimate_payload_bytes(item) for item in value)
    return 256


def _unlink_best_effort(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return


def _fsync_directory(path: Path) -> None:
    try:
        directory_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _prune_best_eval_checkpoints(
    checkpoint_dir: Path,
    *,
    retain_count: int,
) -> None:
    checkpoints = sorted(
        checkpoint_dir.glob("best_eval_step_*.pt"),
        key=lambda path: (_best_checkpoint_step(path), path.name),
        reverse=True,
    )
    for checkpoint_path in checkpoints[max(1, int(retain_count)) :]:
        checkpoint_path.unlink(missing_ok=True)


def _best_checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def load_model_and_optimizer_state(
    model,
    optimizer,
    model_state_dict: Mapping[str, Any],
    optimizer_state_dict: Mapping[str, Any] | None,
    distributed_context=None,
) -> None:
    if (
        distributed_context is not None
        and distributed_context.enabled
        and distributed_context.strategy == "fsdp"
    ):
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_state_dict,
        )

        set_state_dict(
            model,
            optimizer if optimizer is not None else [],
            model_state_dict=model_state_dict,
            optim_state_dict=optimizer_state_dict or {},
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        return

    model.load_state_dict(dict(model_state_dict))
    if optimizer is not None and optimizer_state_dict is not None:
        optimizer.load_state_dict(dict(optimizer_state_dict))


def checkpoint_state_dicts(
    model,
    optimizer=None,
    distributed_context=None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if (
        distributed_context is None
        or not distributed_context.enabled
        or distributed_context.strategy != "fsdp"
    ):
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict() if optimizer is not None else None
        return model_state_dict, optimizer_state_dict

    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    model_state_dict, optimizer_state_dict = get_state_dict(
        model,
        optimizer if optimizer is not None else [],
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    return model_state_dict, optimizer_state_dict


def checkpoint_state_dict(model, distributed_context=None) -> dict[str, Any]:
    model_state_dict, _ = checkpoint_state_dicts(model, distributed_context=distributed_context)
    return model_state_dict


def build_initial_continuation_state(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["run"]["output_dir"])
    run = config.get("run", {})
    model = config.get("model", {})
    if not isinstance(run, Mapping):
        run = {}
    if not isinstance(model, Mapping):
        model = {}
    return {
        "status": "fresh",
        "latest_checkpoint_path": None,
        "latest_checkpoint_step": 0,
        "continuation_source_checkpoint_path": None,
        "best_checkpoint_path": None,
        "checkpoint_metric": None,
        "checkpoint_metric_value": None,
        "checkpoint_selection_step": None,
        "artifact_retry_count": 0,
        "artifact_last_errno": None,
        "last_durable_checkpoint_step": 0,
        "deferred_metric_rows": 0,
        "skipped_periodic_checkpoints": 0,
        "checkpoint_staging_mode": "direct",
        "pending_latest_checkpoint": False,
        "pending_best_checkpoint": None,
        "unresolved_artifact_failures": [],
        "last_completed_step": 0,
        "resume_count": 0,
        "tokens_seen": 0,
        "content_tokens_seen": 0,
        "step": 0,
        "epoch": 0,
        "batch_index": 0,
        "warmup_completed": False,
        "warmup_completion_step": None,
        "warmup_transition_reason": None,
        "pre_nested_warmup_state": _initial_pre_nested_warmup_state(config),
        "output_dir": str(output_dir),
        "resolved_run_mode": str(
            run.get(
                "resolved_run_mode",
                resolve_sampling_mode_from_config_sections(
                    run,
                    config.get("training", {}),
                ),
            )
        ),
        "resolved_sampling_mode": str(
            model.get(
                "resolved_sampling_mode",
                model.get("granularity_sampling_mode", "global"),
            )
        ),
        "granularity_pattern_provenance": copy.deepcopy(
            model.get("granularity_pattern_provenance")
        ),
        "adaptive_sampler_state": _build_initial_adaptive_sampler_state(config),
        "probabilistic_controller_state": None,
        "adaptive_sampler_previous_loss": None,
        "adaptive_sampler_previous_pattern": None,
        "adaptive_reward_summary": None,
        "adaptive_correction_penalty_summary": None,
        "adaptive_sampler_strategy": model.get(
            "adaptive_sampler_strategy",
            None,
        ),
        "adaptive_sampler_exploration_scale": model.get(
            "adaptive_sampler_exploration_scale",
            None,
        ),
        "adaptive_sampler_decay_rate": model.get(
            "adaptive_sampler_decay_rate",
            None,
        ),
        "adaptive_sampler_reward_penalty_weight": model.get(
            "adaptive_sampler_reward_penalty_weight",
            None,
        ),
    }


def update_run_continuation_state(
    config: dict[str, Any],
    state: Mapping[str, Any],
) -> None:
    continuation = config["run"].setdefault("continuation", {})
    if not isinstance(continuation, dict):
        raise ConfigError("run.continuation must be a mapping when provided")
    for key in [
        "status",
        "latest_checkpoint_path",
        "latest_checkpoint_step",
        "continuation_source_checkpoint_path",
        "best_checkpoint_path",
        "checkpoint_metric",
        "checkpoint_metric_value",
        "checkpoint_selection_step",
        "artifact_retry_count",
        "artifact_last_errno",
        "last_durable_checkpoint_step",
        "deferred_metric_rows",
        "skipped_periodic_checkpoints",
        "checkpoint_staging_mode",
        "pending_latest_checkpoint",
        "pending_best_checkpoint",
        "unresolved_artifact_failures",
        "last_completed_step",
        "resume_count",
        "tokens_seen",
        "content_tokens_seen",
        "step",
        "epoch",
        "batch_index",
    ]:
        if key in state and state[key] is not None:
            continuation[key] = state[key]


def _build_initial_adaptive_sampler_state(
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        model = {}
    if (
        model.get("granularity_sampling_mode") != "adaptive_per_block"
        or model.get("adaptive_sampler_strategy") != "ucb"
    ):
        return None

    state = build_adaptive_sampler_state(
        strategy_name="ucb",
        exploration_scale=float(model.get("adaptive_sampler_exploration_scale", 1.0)),
        decay_rate=float(model.get("adaptive_sampler_decay_rate", 0.0)),
    )
    normalized_state = normalize_adaptive_sampler_state(
        state,
        block_count=int(model["num_layers"]),
        granularities=model.get("granularities"),
    )
    return summarize_adaptive_sampler_state(normalized_state)


def _populate_adaptive_sampler_state_metadata(
    state: dict[str, Any],
    config: Mapping[str, Any],
) -> None:
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        model = {}

    if (
        model.get("granularity_sampling_mode") == "adaptive_per_block"
        and model.get("adaptive_sampler_strategy") == "ucb"
    ):
        state["adaptive_sampler_state"] = _build_initial_adaptive_sampler_state(
            config
        )
    else:
        state.setdefault("adaptive_sampler_state", None)
    state["adaptive_sampler_previous_loss"] = None
    state["adaptive_sampler_previous_pattern"] = None
    state["adaptive_reward_summary"] = None
    state["adaptive_correction_penalty_summary"] = None
    state["adaptive_sampler_strategy"] = model.get(
        "adaptive_sampler_strategy"
    )
    state["adaptive_sampler_exploration_scale"] = model.get(
        "adaptive_sampler_exploration_scale"
    )
    state["adaptive_sampler_decay_rate"] = model.get(
        "adaptive_sampler_decay_rate"
    )
    state["adaptive_sampler_reward_penalty_weight"] = model.get(
        "adaptive_sampler_reward_penalty_weight"
    )


def _validate_loaded_adaptive_sampler_state(
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    checkpoint_path: Path,
) -> None:
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        model = {}
    if (
        model.get("granularity_sampling_mode") != "adaptive_per_block"
        or model.get("adaptive_sampler_strategy") != "ucb"
    ):
        return

    expected_strategy = "ucb"
    expected_exploration_scale = float(
        model.get("adaptive_sampler_exploration_scale", 1.0)
    )
    expected_decay_rate = float(model.get("adaptive_sampler_decay_rate", 0.0))
    expected_reward_penalty_weight = float(
        model.get("adaptive_sampler_reward_penalty_weight", 1.0)
    )

    if state.get("adaptive_sampler_strategy") not in (None, expected_strategy):
        raise ConfigError(
            "Checkpoint adaptive sampler strategy does not match the current config "
            f"for {checkpoint_path}"
        )
    if state.get("adaptive_sampler_exploration_scale") not in (
        None,
        expected_exploration_scale,
    ):
        raise ConfigError(
            "Checkpoint adaptive sampler exploration scale does not match the current config "
            f"for {checkpoint_path}"
        )
    if state.get("adaptive_sampler_decay_rate") not in (None, expected_decay_rate):
        raise ConfigError(
            "Checkpoint adaptive sampler decay rate does not match the current config "
            f"for {checkpoint_path}"
        )
    if state.get("adaptive_sampler_reward_penalty_weight") not in (
        None,
        expected_reward_penalty_weight,
    ):
        raise ConfigError(
            "Checkpoint adaptive sampler reward penalty weight does not match the current config "
            f"for {checkpoint_path}"
        )

    adaptive_state = coerce_adaptive_sampler_state(state.get("adaptive_sampler_state"))
    if adaptive_state is None:
        raise ConfigError(
            "Checkpoint is missing adaptive sampler state required for resume "
            f"at {checkpoint_path}"
        )

    normalize_adaptive_sampler_state(
        adaptive_state,
        block_count=int(model["num_layers"]),
        granularities=model.get("granularities"),
    )


def _prepare_adaptive_sampler_runtime_state(
    config: Mapping[str, Any],
    run_state: dict[str, Any],
) -> AdaptiveSamplerState | None:
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        model = {}
    if (
        model.get("granularity_sampling_mode") != "adaptive_per_block"
        or model.get("adaptive_sampler_strategy") != "ucb"
    ):
        run_state.pop("adaptive_sampler_state", None)
        return None

    adaptive_state = coerce_adaptive_sampler_state(
        run_state.get("adaptive_sampler_state")
    )
    if adaptive_state is None:
        adaptive_state = build_adaptive_sampler_state(
            strategy_name="ucb",
            exploration_scale=float(model.get("adaptive_sampler_exploration_scale", 1.0)),
            decay_rate=float(model.get("adaptive_sampler_decay_rate", 0.0)),
        )

    normalized_state = normalize_adaptive_sampler_state(
        adaptive_state,
        block_count=int(model["num_layers"]),
        granularities=model.get("granularities"),
    )
    run_state["adaptive_sampler_state"] = summarize_adaptive_sampler_state(
        normalized_state
    )
    run_state["adaptive_sampler_strategy"] = normalized_state.strategy_name
    run_state["adaptive_sampler_exploration_scale"] = (
        normalized_state.exploration_scale
    )
    run_state["adaptive_sampler_decay_rate"] = normalized_state.decay_rate
    run_state["adaptive_sampler_reward_penalty_weight"] = float(
        model.get("adaptive_sampler_reward_penalty_weight", 1.0)
    )
    if run_state.get("adaptive_sampler_previous_pattern") is not None:
        run_state["adaptive_sampler_previous_pattern"] = [
            str(granularity)
            for granularity in run_state["adaptive_sampler_previous_pattern"]
        ]
    return normalized_state


def _pattern_change_penalty(
    previous_pattern: Sequence[str] | None,
    current_pattern: Sequence[str],
) -> float:
    if not previous_pattern:
        return 0.0

    previous = [str(granularity) for granularity in previous_pattern]
    current = [str(granularity) for granularity in current_pattern]
    if not current:
        return 0.0

    difference_count = sum(
        1 for previous_granularity, current_granularity in zip(previous, current)
        if previous_granularity != current_granularity
    )
    difference_count += abs(len(previous) - len(current))
    return difference_count / max(len(current), len(previous), 1)


def _update_adaptive_sampler_runtime_state(
    config: Mapping[str, Any],
    run_state: dict[str, Any],
    adaptive_sampler_state: AdaptiveSamplerState,
    *,
    phase: str,
    latest_loss: float,
    selected_layer_granularities: Sequence[str],
    step: int,
    epoch: int,
) -> None:
    previous_loss = run_state.get("adaptive_sampler_previous_loss")
    previous_pattern = run_state.get("adaptive_sampler_previous_pattern")
    reward_penalty_weight = float(
        run_state.get(
            "adaptive_sampler_reward_penalty_weight",
            config.get("model", {}).get("adaptive_sampler_reward_penalty_weight", 1.0),
        )
    )
    correction_penalty = _pattern_change_penalty(
        previous_pattern,
        selected_layer_granularities,
    )
    reward_record = build_adaptive_reward_record(
        previous_loss=float(previous_loss) if previous_loss is not None else None,
        current_loss=latest_loss,
        correction_penalty=correction_penalty,
        reward_penalty_weight=reward_penalty_weight,
        phase=phase,
        step=step,
        epoch=epoch,
    )
    if previous_loss is not None:
        adaptive_sampler_state = update_adaptive_sampler_state(
            adaptive_sampler_state,
            reward_record,
            sampled_pattern=list(selected_layer_granularities),
        )

    run_state["adaptive_reward_summary"] = dict(reward_record)
    run_state["adaptive_correction_penalty_summary"] = {
        "correction_penalty": correction_penalty,
        "reward_penalty_weight": reward_penalty_weight,
        "normalized_correction_penalty": reward_record[
            "normalized_correction_penalty"
        ],
    }
    run_state["adaptive_sampler_previous_loss"] = latest_loss
    run_state["adaptive_sampler_previous_pattern"] = [
        str(granularity) for granularity in selected_layer_granularities
    ]
    run_state["adaptive_sampler_state"] = summarize_adaptive_sampler_state(
        adaptive_sampler_state
    )


def load_run_continuation_state(
    config: dict[str, Any],
    model,
    optimizer,
    scheduler,
    distributed_context=None,
) -> dict[str, Any]:
    checkpoint_path = Path(config["run"]["output_dir"]) / "checkpoints" / "latest.pt"
    previous_path = checkpoint_path.with_name("latest.prev.pt")
    load_kwargs = {
        "config": config,
        "fallback_tokens_per_step": int(
            config["training"]["expected_tokens_per_step"]
        ),
        "distributed_context": distributed_context,
        "output_dir": config["run"]["output_dir"],
        "run_id": config["run"]["run_id"],
    }

    primary_error: Exception | None = None
    if checkpoint_path.exists():
        try:
            state = load_checkpoint_state(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                **load_kwargs,
            )
            state["continuation_source_checkpoint_path"] = str(checkpoint_path)
            return state
        except Exception as error:
            primary_error = error

    if previous_path.exists():
        try:
            state = load_checkpoint_state(
                previous_path,
                model,
                optimizer,
                scheduler,
                **load_kwargs,
            )
        except Exception as fallback_error:
            if primary_error is not None:
                raise ConfigError(
                    "Unable to load continuation checkpoints "
                    f"{checkpoint_path} and {previous_path}: "
                    f"primary={primary_error}; fallback={fallback_error}"
                ) from fallback_error
            raise
        state["continuation_source_checkpoint_path"] = str(previous_path)
        state["latest_checkpoint_path"] = str(checkpoint_path)
        return state

    if primary_error is not None:
        raise primary_error
    return load_checkpoint_state(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        **load_kwargs,
    )


def load_checkpoint_state(
    checkpoint_path: str | Path,
    model,
    optimizer,
    scheduler,
    config: Mapping[str, Any] | None = None,
    fallback_tokens_per_step: int | None = None,
    distributed_context=None,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        state = {
            "status": "fresh",
            "latest_checkpoint_path": None,
            "latest_checkpoint_step": 0,
            "continuation_source_checkpoint_path": None,
            "best_checkpoint_path": None,
            "checkpoint_metric": None,
            "checkpoint_metric_value": None,
            "checkpoint_selection_step": None,
            "artifact_retry_count": 0,
            "artifact_last_errno": None,
            "last_durable_checkpoint_step": 0,
            "deferred_metric_rows": 0,
            "skipped_periodic_checkpoints": 0,
            "checkpoint_staging_mode": "direct",
            "pending_latest_checkpoint": False,
            "pending_best_checkpoint": None,
            "unresolved_artifact_failures": [],
            "last_completed_step": 0,
            "resume_count": 0,
            "tokens_seen": 0,
            "content_tokens_seen": 0,
            "step": 0,
            "epoch": 0,
            "batch_index": 0,
            "warmup_completed": False,
            "warmup_completion_step": None,
            "warmup_transition_reason": None,
            "pre_nested_warmup_state": (
                _initial_pre_nested_warmup_state(config)
                if config is not None
                else None
            ),
            "probabilistic_controller_state": None,
        }
        if output_dir is not None:
            state["output_dir"] = str(output_dir)
        if run_id is not None:
            state["run_id"] = str(run_id)
        if config is not None:
            _populate_adaptive_sampler_state_metadata(state, config)
        return state

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    reproducibility_payload = _validate_reproducibility_payload(
        checkpoint,
        config=config,
        checkpoint_path=checkpoint_path,
        distributed_context=distributed_context,
    )
    probabilistic_controller_state = checkpoint.get(
        "probabilistic_controller_state"
    )
    if config is not None and uses_probabilistic_controller(config):
        probabilistic_controller_state = (
            validate_probabilistic_controller_checkpoint_state(
                probabilistic_controller_state,
                config=config,
                checkpoint_path=checkpoint_path,
            )
        )
    model_state_dict = checkpoint.get("model_state_dict")
    if model_state_dict is None:
        raise ConfigError(f"Checkpoint missing model_state_dict: {checkpoint_path}")

    optimizer_state_dict = checkpoint.get("optimizer_state_dict")
    scheduler_state_dict = checkpoint.get("scheduler_state_dict")
    load_model_and_optimizer_state(
        model,
        optimizer,
        model_state_dict,
        optimizer_state_dict,
        distributed_context=distributed_context,
    )
    if scheduler is not None and scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)
    if reproducibility_payload is not None:
        restore_rng_state(reproducibility_payload["rng_state"])

    last_completed_step = int(
        checkpoint.get("step", checkpoint.get("last_completed_step", 0))
    )
    tokens_seen = int(
        checkpoint.get(
            "tokens_seen",
            fallback_tokens_per_step * last_completed_step
            if fallback_tokens_per_step is not None
            else 0,
        )
    )
    content_tokens_seen = int(checkpoint.get("content_tokens_seen", 0))
    resume_count = int(checkpoint.get("resume_count", 0)) + 1
    state = {
        "status": "resumed",
        "latest_checkpoint_path": str(checkpoint_path),
        "latest_checkpoint_step": last_completed_step,
        "continuation_source_checkpoint_path": str(checkpoint_path),
        "last_completed_step": last_completed_step,
        "resume_count": resume_count,
        "tokens_seen": tokens_seen,
        "content_tokens_seen": content_tokens_seen,
        "step": last_completed_step,
        "epoch": int(checkpoint.get("epoch", 0)),
        "batch_index": int(checkpoint.get("batch_index", 0)),
        "warmup_completed": bool(checkpoint.get("warmup_completed", False)),
        "warmup_completion_step": checkpoint.get("warmup_completion_step"),
        "warmup_transition_reason": checkpoint.get("warmup_transition_reason"),
        "pre_nested_warmup_state": copy.deepcopy(
            checkpoint.get("pre_nested_warmup_state")
        ),
        "resolved_run_mode": checkpoint.get("resolved_run_mode"),
        "resolved_sampling_mode": checkpoint.get("resolved_sampling_mode"),
        "granularity_pattern_provenance": checkpoint.get(
            "granularity_pattern_provenance"
        ),
        "adaptive_sampler_state": checkpoint.get("adaptive_sampler_state"),
        "probabilistic_controller_state": probabilistic_controller_state,
        "adaptive_sampler_previous_loss": checkpoint.get(
            "adaptive_sampler_previous_loss"
        ),
        "adaptive_sampler_previous_pattern": checkpoint.get(
            "adaptive_sampler_previous_pattern"
        ),
        "adaptive_reward_summary": checkpoint.get("adaptive_reward_summary"),
        "adaptive_correction_penalty_summary": checkpoint.get(
            "adaptive_correction_penalty_summary"
        ),
        "adaptive_sampler_strategy": checkpoint.get("adaptive_sampler_strategy"),
        "adaptive_sampler_exploration_scale": checkpoint.get(
            "adaptive_sampler_exploration_scale"
        ),
        "adaptive_sampler_decay_rate": checkpoint.get("adaptive_sampler_decay_rate"),
        "adaptive_sampler_reward_penalty_weight": checkpoint.get(
            "adaptive_sampler_reward_penalty_weight"
        ),
        "best_checkpoint_path": checkpoint.get("best_checkpoint_path"),
        "checkpoint_metric": checkpoint.get("best_checkpoint_metric"),
        "checkpoint_metric_value": checkpoint.get(
            "best_checkpoint_metric_value"
        ),
        "checkpoint_selection_step": checkpoint.get(
            "best_checkpoint_selection_step"
        ),
        "artifact_retry_count": int(checkpoint.get("artifact_retry_count", 0)),
        "artifact_last_errno": checkpoint.get("artifact_last_errno"),
        "last_durable_checkpoint_step": int(
            checkpoint.get("last_durable_checkpoint_step", last_completed_step)
        ),
        "deferred_metric_rows": int(checkpoint.get("deferred_metric_rows", 0)),
        "skipped_periodic_checkpoints": int(
            checkpoint.get("skipped_periodic_checkpoints", 0)
        ),
        "checkpoint_staging_mode": checkpoint.get(
            "checkpoint_staging_mode", "direct"
        ),
        "pending_latest_checkpoint": False,
        "pending_best_checkpoint": None,
        "unresolved_artifact_failures": list(
            checkpoint.get("unresolved_artifact_failures", [])
        ),
    }
    if output_dir is not None:
        state["output_dir"] = str(output_dir)
    if run_id is not None:
        state["run_id"] = str(run_id)
    if config is not None:
        from src.training.warmup import validate_pre_nested_warmup_resume_state

        state["pre_nested_warmup_state"] = validate_pre_nested_warmup_resume_state(
            config,
            state.get("pre_nested_warmup_state"),
            last_completed_step=last_completed_step,
        )
        _validate_loaded_adaptive_sampler_state(state, config, checkpoint_path)
    return state


def _initial_pre_nested_warmup_state(
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    warmup = config.get("training", {}).get("pre_nested_warmup", {})
    if not isinstance(warmup, Mapping):
        return None
    labels = [str(label) for label in config.get("model", {}).get("granularities", [])]
    return {
        "enabled": bool(warmup.get("enabled", False)),
        "active": bool(warmup.get("active", False)),
        "duration": int(warmup.get("duration", 0)),
        "unit": str(warmup.get("unit", "epochs")),
        "policy": str(warmup.get("policy", "full_only")),
        "action_interval_steps": warmup.get("action_interval_steps"),
        "schedule_seed": warmup.get("schedule_seed"),
        "schedule_hash": warmup.get("schedule_hash"),
        "schedule": copy.deepcopy(warmup.get("schedule")),
        "passes": warmup.get("passes"),
        "current_window_index": 0,
        "current_window_offset": 0,
        "completed_steps": 0,
        "per_granularity_counts": {label: 0 for label in labels},
        "controller_start_step": warmup.get("controller_start_step"),
        "schedule_initialized": False,
        "completed": bool(warmup.get("completed", False)),
        "completion_step": warmup.get("completion_step"),
        "transition_reason": warmup.get("transition_reason"),
    }


def _validate_reproducibility_payload(
    checkpoint: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None,
    checkpoint_path: Path,
    distributed_context=None,
) -> Mapping[str, Any] | None:
    payload = checkpoint.get("reproducibility")
    if not isinstance(payload, Mapping):
        if config is not None and config.get("validation_manifest_hash") is None:
            return None
        raise ConfigError(
            "Checkpoint lacks the reproducibility payload required for corrected "
            f"runs: {checkpoint_path}"
        )
    required = {
        "root_seed",
        "seed_stream_version",
        "data_split_version",
        "validation_manifest_hash",
        "comparison_control_signature",
        "batch_size_per_process",
        "world_size",
        "rank_topology",
        "rng_state",
        "deterministic_runtime_settings",
        "adaptive_sampler_seed_provenance",
    }
    missing = required - set(payload)
    if missing:
        raise ConfigError(
            f"Checkpoint reproducibility payload is incomplete: {sorted(missing)}"
        )
    if config is None:
        return payload

    expected = {
        "root_seed": config["run"]["seed"],
        "seed_stream_version": config["run"]["reproducibility"][
            "seed_stream_version"
        ],
        "data_split_version": config["run"]["reproducibility"][
            "data_split_version"
        ],
        "validation_manifest_hash": config.get("validation_manifest_hash"),
        "comparison_control_signature": config.get(
            "comparison_control_signature"
        ),
        "batch_size_per_process": config["training"]["batch_size_per_process"],
        "world_size": int(getattr(distributed_context, "world_size", 1)),
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ConfigError(
            "Checkpoint reproducibility controls do not match the current run: "
            f"{mismatches}"
        )
    expected_topology = list(range(expected["world_size"]))
    if list(payload["rank_topology"]) != expected_topology:
        raise ConfigError("Checkpoint rank topology does not match the current run")
    settings = payload["deterministic_runtime_settings"]
    if config.get("validation_manifest_hash") is not None and (
        not isinstance(settings, Mapping)
        or not settings.get("deterministic_algorithms", False)
    ):
        raise ConfigError("Checkpoint did not record strict deterministic settings")
    return payload
