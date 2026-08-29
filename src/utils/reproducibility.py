"""Named seed streams and strict deterministic runtime helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any, Mapping

import numpy as np


SEED_STREAM_VERSION = 1
DATA_SPLIT_VERSION = 1
STRICT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
PROBABILISTIC_SEED_STREAMS = (
    "controller_panel",
    "final_holdout",
    "posterior_sampling",
    "pre_nested_warmup_schedule",
    "controller_reset_schedule",
)
PANELGRAD_SEED_STREAMS = (
    "controller_panel",
    "final_holdout",
    "pre_nested_warmup_schedule",
    "panelgrad_sampling",
)
SEED_STREAMS = (
    "model_initialization",
    "python_training",
    "numpy_training",
    "torch_training",
    "dataset_selection",
    "validation_holdout",
    "training_sampler",
    "dataloader_workers",
    "granularity_selection",
    "adaptive_sampling",
    "artifact_retry_jitter",
    "panelgrad_sampling",
    *PROBABILISTIC_SEED_STREAMS,
)

_DEDICATED_RANDOM_GENERATORS: dict[str, random.Random] = {}


def derive_seed(root_seed: int, stream_name: str, version: int = 1) -> int:
    """Derive a stable nonnegative 63-bit seed from the documented contract."""

    if isinstance(root_seed, bool) or not isinstance(root_seed, int) or root_seed < 0:
        raise ValueError("run.seed must be an explicit nonnegative integer")
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise ValueError("run.reproducibility.seed_stream_version must be positive")
    if stream_name not in SEED_STREAMS:
        raise ValueError(f"Unknown reproducibility seed stream: {stream_name}")
    material = f"{version} | {root_seed} | {stream_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def seed_for(config: Mapping[str, Any], stream_name: str) -> int:
    run = config.get("run", {})
    reproducibility = run.get("reproducibility", {})
    return derive_seed(
        run.get("seed"),
        stream_name,
        int(reproducibility.get("seed_stream_version", SEED_STREAM_VERSION)),
    )


def configure_strict_determinism(config: Mapping[str, Any]) -> dict[str, Any]:
    """Configure deterministic Torch behavior before any CUDA initialization."""

    import torch

    run = config.get("run", {})
    reproducibility = run.get("reproducibility", {})
    mode = str(reproducibility.get("mode", "strict"))
    if mode != "strict":
        raise RuntimeError(f"Unsupported reproducibility mode: {mode}")
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "Strict determinism must be configured before CUDA initialization"
        )

    configured_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured_workspace not in (None, STRICT_CUBLAS_WORKSPACE_CONFIG):
        raise RuntimeError(
            "Strict determinism requires CUBLAS_WORKSPACE_CONFIG="
            f"{STRICT_CUBLAS_WORKSPACE_CONFIG}, got {configured_workspace!r}"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = STRICT_CUBLAS_WORKSPACE_CONFIG

    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception as error:  # keep deterministic-operation failures visible
        raise RuntimeError(f"Unable to enable strict Torch determinism: {error}") from error

    settings = deterministic_runtime_settings()
    if not settings["deterministic_algorithms"]:
        raise RuntimeError("Torch did not enable deterministic algorithms")
    return settings


def deterministic_runtime_settings() -> dict[str, Any]:
    import torch

    return {
        "mode": "strict",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }


def seed_model_initialization(config: Mapping[str, Any]) -> None:
    seed = seed_for(config, "model_initialization")
    _seed_all_global_rngs(seed, seed, seed)


def seed_training_randomness(config: Mapping[str, Any]) -> None:
    _seed_all_global_rngs(
        seed_for(config, "python_training"),
        seed_for(config, "numpy_training"),
        seed_for(config, "torch_training"),
    )
    _DEDICATED_RANDOM_GENERATORS.clear()
    for stream_name in (
        "granularity_selection",
        "adaptive_sampling",
        "artifact_retry_jitter",
    ):
        _DEDICATED_RANDOM_GENERATORS[stream_name] = random.Random(
            seed_for(config, stream_name)
        )


def _seed_all_global_rngs(python_seed: int, numpy_seed: int, torch_seed: int) -> None:
    import torch

    random.seed(python_seed)
    np.random.seed(numpy_seed % (2**32))
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)


def dedicated_random(config: Mapping[str, Any], stream_name: str) -> random.Random:
    generator = _DEDICATED_RANDOM_GENERATORS.get(stream_name)
    if generator is None:
        generator = random.Random(seed_for(config, stream_name))
        _DEDICATED_RANDOM_GENERATORS[stream_name] = generator
    return generator


def capture_rng_state() -> dict[str, Any]:
    import torch

    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "dedicated": {
            name: generator.getstate()
            for name, generator in _DEDICATED_RANDOM_GENERATORS.items()
        },
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    import torch

    required = {"python", "numpy", "torch_cpu", "torch_cuda", "dedicated"}
    missing = required - set(state)
    if missing:
        raise RuntimeError(f"Checkpoint RNG state is incomplete: {sorted(missing)}")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise RuntimeError("Checkpoint NumPy RNG state is malformed")
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state["torch_cuda"]
    if torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("Checkpoint CUDA RNG topology does not match this runtime")
        torch.cuda.set_rng_state_all(cuda_states)
    elif cuda_states:
        raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
    _DEDICATED_RANDOM_GENERATORS.clear()
    for name, generator_state in state["dedicated"].items():
        generator = random.Random()
        generator.setstate(generator_state)
        _DEDICATED_RANDOM_GENERATORS[str(name)] = generator


def stable_hash(value: Any) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_balanced_warmup_schedule(
    granularities: list[str] | tuple[str, ...],
    *,
    passes: int,
    seed: int,
    action_interval_steps: int,
    duration_steps: int,
) -> tuple[list[str], str]:
    """Build independently shuffled, complete granularity passes.

    A local generator deliberately isolates schedule construction from all
    training, dataloader, and posterior-sampling random state.
    """

    labels = [str(label) for label in granularities]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("balanced warmup granularities must be nonempty and unique")
    if isinstance(passes, bool) or not isinstance(passes, int) or passes <= 0:
        raise ValueError("balanced warmup passes must be a positive integer")
    if (
        isinstance(action_interval_steps, bool)
        or not isinstance(action_interval_steps, int)
        or action_interval_steps <= 0
    ):
        raise ValueError("balanced warmup action interval must be positive")
    if (
        isinstance(duration_steps, bool)
        or not isinstance(duration_steps, int)
        or duration_steps <= 0
    ):
        raise ValueError("balanced warmup duration must be positive")
    if duration_steps != passes * action_interval_steps * len(labels):
        raise ValueError("balanced warmup duration does not match complete passes")

    generator = random.Random(int(seed))
    schedule: list[str] = []
    for _ in range(passes):
        permutation = list(labels)
        generator.shuffle(permutation)
        schedule.extend(permutation)

    schedule_identity = {
        "version": 1,
        "policy": "balanced_global",
        "seed_stream_name": "pre_nested_warmup_schedule",
        "resolved_seed": int(seed),
        "ordered_granularities": labels,
        "passes": int(passes),
        "action_interval_steps": int(action_interval_steps),
        "duration_steps": int(duration_steps),
        "schedule": schedule,
    }
    return schedule, stable_hash(schedule_identity)


def build_controller_reset_schedule(
    granularities: list[str] | tuple[str, ...],
    *,
    acquisition_passes: int,
    root_seed: int,
    episode_index: int,
) -> tuple[list[str], int, str]:
    """Build deterministic, episode-indexed balanced acquisition passes."""

    labels = [str(label) for label in granularities]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("controller reset granularities must be nonempty and unique")
    if (
        isinstance(acquisition_passes, bool)
        or not isinstance(acquisition_passes, int)
        or acquisition_passes <= 0
    ):
        raise ValueError("controller reset acquisition passes must be positive")
    if isinstance(root_seed, bool) or not isinstance(root_seed, int) or root_seed < 0:
        raise ValueError("controller reset root seed must be nonnegative")
    if (
        isinstance(episode_index, bool)
        or not isinstance(episode_index, int)
        or episode_index < 0
    ):
        raise ValueError("controller reset episode index must be nonnegative")

    seed_material = (
        f"controller-reset-schedule-v1 | {root_seed} | {episode_index}"
    ).encode("utf-8")
    episode_seed = int.from_bytes(
        hashlib.sha256(seed_material).digest()[:8], "big"
    ) & ((1 << 63) - 1)
    generator = random.Random(episode_seed)
    schedule: list[str] = []
    for _ in range(acquisition_passes):
        permutation = list(labels)
        generator.shuffle(permutation)
        schedule.extend(permutation)

    identity = {
        "version": 1,
        "policy": "balanced_global",
        "seed_stream_name": "controller_reset_schedule",
        "root_seed": int(root_seed),
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "ordered_granularities": labels,
        "acquisition_passes": int(acquisition_passes),
        "schedule": schedule,
    }
    return schedule, episode_seed, stable_hash(identity)


def probabilistic_seed_provenance(
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Resolve the independent data-role and posterior seed streams."""

    seed_stream_version = int(
        config["run"]["reproducibility"]["seed_stream_version"]
    )
    return {
        stream_name: {
            "stream_name": stream_name,
            "seed_stream_version": seed_stream_version,
            "resolved_seed": seed_for(config, stream_name),
        }
        for stream_name in PROBABILISTIC_SEED_STREAMS
    }


def panelgrad_seed_provenance(
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Resolve only the data-role, warmup, and categorical PanelGrad streams."""

    seed_stream_version = int(
        config["run"]["reproducibility"]["seed_stream_version"]
    )
    return {
        stream_name: {
            "stream_name": stream_name,
            "seed_stream_version": seed_stream_version,
            "resolved_seed": seed_for(config, stream_name),
        }
        for stream_name in PANELGRAD_SEED_STREAMS
    }


def _uses_probabilistic_adaptive_controller(config: Mapping[str, Any]) -> bool:
    model = config.get("model", {})
    return (
        model.get("granularity_sampling_mode")
        in {"adaptive_global", "adaptive_per_block"}
        and model.get("adaptive_sampler_strategy") == "thompson"
    )


def _uses_panelgrad(config: Mapping[str, Any]) -> bool:
    model = config.get("model", {})
    return (
        model.get("granularity_sampling_mode") == "adaptive_global"
        and model.get("adaptive_sampler_strategy") == "panelgrad"
    )


def build_comparison_control_signature(config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    run = config["run"]
    training = config["training"]
    model = config["model"]
    dataset = config["dataset"]
    validation = config["evaluation"]["validation"]
    inputs = {
        "root_seed": run["seed"],
        "seed_stream_version": run["reproducibility"]["seed_stream_version"],
        "validation_manifest_hash": config.get("validation_manifest_hash"),
        "optimizer": training.get("optimizer"),
        "learning_rate": training.get("resolved_learning_rate"),
        "scheduler": training.get("scheduler"),
        "warmup_steps": training.get("resolved_warmup_steps"),
        "token_budget": training.get("token_budget"),
        "batch_size_per_process": training.get("batch_size_per_process"),
        "gradient_accumulation_steps": training.get(
            "gradient_accumulation_steps"
        ),
        "effective_world_size": training.get("effective_world_size"),
        "distributed_topology": training.get("distributed"),
        "expected_tokens_per_microstep": training.get(
            "expected_tokens_per_microstep"
        ),
        "expected_tokens_per_step": training.get("expected_tokens_per_step"),
        "precision": training.get("resolved_mixed_precision"),
        "context_length": model.get("context_length"),
        "model_family": run.get("model_family"),
        "sampling_mode": run.get("sampling_mode"),
        "active_granularity": run.get("granularity"),
        "ordered_granularities": model.get("granularities"),
        "granularity_prefixes": model.get("granularity_prefixes"),
        "granularity_sampling_mode": model.get("granularity_sampling_mode"),
        "global_sampling_schedule": model.get("global_sampling_schedule"),
        "global_sampling_interval_steps": model.get(
            "global_sampling_interval_steps"
        ),
        "validation_interval_steps": validation.get("interval_steps"),
        "validation_interval_tokens": validation.get("interval_tokens"),
        "validation_run_at_completion": validation.get("run_at_completion"),
        "dataset_name": dataset.get("dataset_name"),
        "dataset_config_name": dataset.get("dataset_config_name"),
        "dataset_split": dataset.get("dataset_split"),
        "tokenizer_name": model.get("tokenizer_name"),
        "tokenizer_revision": model.get("tokenizer_revision"),
        "tokenizer_manifest_hash": model.get("tokenizer_manifest_hash"),
        "prepared_corpus_hash": dataset.get("corpus_hash"),
        "optimizer_training_role_hash": config.get(
            "optimizer_training_manifest_hash"
        ),
    }
    optimizer_iteration = dataset.get("optimizer_iteration")
    if (
        isinstance(optimizer_iteration, Mapping)
        and optimizer_iteration.get("mode") == "repeat_epochs"
    ):
        inputs["optimizer_iteration"] = dict(optimizer_iteration)
    controlled_experiment = config.get("controlled_experiment", {})
    if isinstance(controlled_experiment, Mapping) and controlled_experiment.get(
        "selection_report_hash"
    ) not in (None, ""):
        inputs["selection_report_hash"] = controlled_experiment[
            "selection_report_hash"
        ]
    portfolio_catchup = (
        controlled_experiment.get("portfolio_catchup")
        if isinstance(controlled_experiment, Mapping)
        else None
    )
    if isinstance(portfolio_catchup, Mapping):
        portfolio_contract = {
            "comparison_group_id": controlled_experiment.get(
                "comparison_group_id"
            ),
            "comparison_role": controlled_experiment.get("comparison_role"),
            "schema_version": portfolio_catchup.get("schema_version"),
            "reference_budget_tokens": portfolio_catchup.get(
                "reference_budget_tokens"
            ),
            "elastic_budget_cap_tokens": portfolio_catchup.get(
                "elastic_budget_cap_tokens"
            ),
            "aggregate_reference_count": portfolio_catchup.get(
                "aggregate_reference_count"
            ),
            "granularities": portfolio_catchup.get("granularities"),
            "perplexity_tolerance": portfolio_catchup.get(
                "perplexity_tolerance"
            ),
            "required_consecutive_evaluations": portfolio_catchup.get(
                "required_consecutive_evaluations"
            ),
            "target_manifest_hash": portfolio_catchup.get(
                "target_manifest_hash"
            ),
            "save_confirmation_checkpoint": portfolio_catchup.get(
                "save_confirmation_checkpoint"
            ),
            "stop_on_confirmation": portfolio_catchup.get(
                "stop_on_confirmation"
            ),
        }
        if portfolio_catchup.get("schema_version") == 3:
            portfolio_contract["comparison_arm_id"] = controlled_experiment.get(
                "comparison_arm_id"
            )
        # Preserve the exact signature shape used by already-running schema-1
        # standalone references. New schema-2 contracts omit this retired field.
        if "lr_selection_manifest_hash" in portfolio_catchup:
            portfolio_contract["lr_selection_manifest_hash"] = portfolio_catchup.get(
                "lr_selection_manifest_hash"
            )
        inputs["portfolio_catchup_contract"] = portfolio_contract
    if _uses_probabilistic_adaptive_controller(config):
        inputs["probabilistic_seed_streams"] = probabilistic_seed_provenance(config)
        inputs["probabilistic_data_role_manifests"] = {
            "data_roles": config.get("data_roles_manifest_hash"),
            "optimizer_training": config.get(
                "optimizer_training_manifest_hash"
            ),
            "controller": config.get("controller_manifest_hash"),
            "ordinary_validation": config.get("validation_manifest_hash"),
            "final_holdout": config.get("final_holdout_manifest_hash"),
        }
    if _uses_panelgrad(config):
        inputs["panelgrad_seed_streams"] = panelgrad_seed_provenance(config)
        inputs["panelgrad_data_role_manifests"] = {
            "data_roles": config.get("data_roles_manifest_hash"),
            "optimizer_training": config.get(
                "optimizer_training_manifest_hash"
            ),
            "controller": config.get("controller_manifest_hash"),
            "ordinary_validation": config.get("validation_manifest_hash"),
            "final_holdout": config.get("final_holdout_manifest_hash"),
        }
    return stable_hash(inputs), inputs
