"""Config-driven training orchestration for MatFormer reproduction runs."""

# ruff: noqa: E402  # Load .env before modules that cache environment settings.

from __future__ import annotations

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency

    def load_dotenv(*args, **kwargs):
        return None


load_dotenv()

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

import src.training.checkpointing as training_checkpointing
import src.training.data as training_data
import src.training.distributed as training_distributed
import src.training.modeling as training_modeling
import src.training.monitoring as training_monitoring
import src.training.steps as training_steps
import src.training.warmup as training_warmup
from src.evaluation.validation import evaluate_controller_objective
from src.models.adaptive_sampler import (
    build_adaptive_sampler_artifact_fields,
)
from src.models.correction import summarize_correction_context_from_config
from src.models.ffn import build_concat_layout_diagnostic
from src.models.wiring import (
    record_runtime_sampling_provenance,
)
from src.utils.config import (
    ConfigError,
    attach_parameter_counts_to_config,
    resolve_run_config,
    validate_run_config,
)
from src.utils.metrics import (
    MetricsJournal,
    append_controller_event,
    build_checkpoint_summary_fields,
    build_controller_summary,
    build_monitoring_summary_fields,
    build_run_summary,
    build_scaling_result_rows,
    controller_action_frequency_counts,
    controller_uncertainty_summary,
    format_controller_lifecycle_log,
    read_controller_events,
    summarize_runtime_granularity_pattern_from_config,
    write_controller_summary,
    write_config_artifact,
    write_metrics_csv,
    write_run_summary,
    write_scaling_results_csv,
    write_json_artifact,
)
from src.utils.reproducibility import (
    build_comparison_control_signature,
    configure_strict_determinism,
    seed_for,
    seed_model_initialization,
    seed_training_randomness,
    stable_hash,
)
from src.training.probabilistic_controller import (
    build_probabilistic_controller,
    restore_probabilistic_controller,
)


def run_from_config_path(
    config_path: str | Path,
    run_id: str | None = None,
    overrides: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    ensure_single_process_runtime()
    config = resolve_run_config(
        config_path,
        run_id=run_id,
        overrides=overrides,
        output_dir=output_dir,
    )
    return run_training(config)


def ensure_single_process_runtime() -> None:
    raw_world_size = os.environ.get("WORLD_SIZE", "1")
    try:
        world_size = int(raw_world_size)
    except (TypeError, ValueError):
        world_size = 1

    if world_size > 1:
        raise ConfigError(
            "single-process only: distributed or multi-process execution is not supported"
        )


def uses_probabilistic_controller(config: Mapping[str, Any]) -> bool:
    model = config.get("model", {})
    return (
        model.get("granularity_sampling_mode")
        in {"adaptive_global", "adaptive_per_block"}
        and model.get("adaptive_sampler_strategy") == "thompson"
    )


def prepare_probabilistic_data_roles(
    config: dict[str, Any],
    tokenized_dataset,
    device: torch.device,
    distributed_context=None,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Create fixed Bayesian data roles and their runtime dataloaders."""

    validation = config["evaluation"]["validation"]
    partition = training_data.partition_probabilistic_data_roles(
        tokenized_dataset,
        ordinary_validation_example_count=int(validation["holdout"]["examples"]),
        ordinary_validation_seed=seed_for(config, "validation_holdout"),
        controller_seed=seed_for(config, "controller_panel"),
        final_holdout_seed=seed_for(config, "final_holdout"),
        source_provenance=_probabilistic_source_provenance(
            config,
            tokenized_dataset,
        ),
    )
    training_data.validate_data_role_disjointness(partition["role_manifests"])
    _attach_probabilistic_role_provenance(config, partition)

    role_datasets = partition["datasets"]
    training = config["training"]
    batch_size = int(training["batch_size_per_process"])
    rank = int(getattr(distributed_context, "rank", 0))
    world_size = int(getattr(distributed_context, "world_size", 1))
    pin_memory = device.type == "cuda"
    if distributed_context is not None and distributed_context.enabled:
        train_sampler = training_data.DeterministicDistributedTrainingSampler(
            role_datasets["optimizer_training"],
            seed_for(config, "training_sampler"),
            rank,
            world_size,
        )
        validation_sampler = training_data.DistributedValidationSampler(
            role_datasets["ordinary_validation"],
            rank,
            world_size,
        )
        controller_sampler = training_data.DistributedValidationSampler(
            role_datasets["controller"],
            rank,
            world_size,
        )
    else:
        train_sampler = training_data.EpochRandomSampler(
            role_datasets["optimizer_training"],
            seed_for(config, "training_sampler"),
        )
        validation_sampler = None
        controller_sampler = None

    num_workers = int(training.get("dataloader_num_workers", 0))
    worker_seed = seed_for(config, "dataloader_workers")
    train_dataloader = training_data.build_language_model_dataloader(
        role_datasets["optimizer_training"],
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=training_data.SeededWorkerInitializer(
            worker_seed,
            train_sampler,
            rank,
            world_size,
        ),
    )
    validation_dataloader = training_data.build_language_model_dataloader(
        role_datasets["ordinary_validation"],
        batch_size=batch_size,
        sampler=validation_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=training_data.SeededWorkerInitializer(
            worker_seed,
            validation_sampler,
            rank,
            world_size,
        ),
    )
    controller_dataloader = training_data.build_language_model_dataloader(
        role_datasets["controller"],
        batch_size=batch_size,
        sampler=controller_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=training_data.SeededWorkerInitializer(
            worker_seed,
            controller_sampler,
            rank,
            world_size,
        ),
    )
    return (
        train_dataloader,
        validation_dataloader,
        controller_dataloader,
        partition,
    )


def _probabilistic_source_provenance(
    config: Mapping[str, Any],
    tokenized_dataset,
) -> dict[str, Any]:
    dataset = config["dataset"]
    model = config["model"]
    tokenization_identity = dataset.get("tokenization_identity")
    if tokenization_identity is None:
        tokenization_identity = {
            "preprocessing_version": 1,
            "tokenizer_name": model.get("tokenizer_name"),
            "context_length": model["context_length"],
            "truncation": True,
            "padding": "max_length",
            "attention_mask": True,
        }
        tokenization_identity["identity_hash"] = stable_hash(tokenization_identity)
        dataset["tokenization_identity"] = tokenization_identity
    return {
        "dataset_name": dataset["dataset_name"],
        "dataset_config_name": dataset.get("dataset_config_name"),
        "source_split": dataset["dataset_split"],
        "source_dataset_fingerprint": dataset.get(
            "source_dataset_fingerprint",
            getattr(tokenized_dataset, "_fingerprint", None),
        ),
        "tokenization_identity": tokenization_identity,
    }


def _attach_probabilistic_role_provenance(
    config: dict[str, Any],
    partition: Mapping[str, Any],
) -> None:
    manifests = partition["role_manifests"]
    parent_manifest = partition["parent_manifest"]
    manifest_hashes = {
        role: manifests[role]["manifest_hash"]
        for role in training_data.PROBABILISTIC_DATA_ROLE_NAMES
    }
    config["data_roles_manifest_hash"] = parent_manifest["parent_manifest_hash"]
    config["optimizer_training_manifest_hash"] = manifest_hashes["optimizer_training"]
    config["controller_manifest_hash"] = manifest_hashes["controller"]
    config["validation_manifest_hash"] = manifest_hashes["ordinary_validation"]
    config["final_holdout_manifest_hash"] = manifest_hashes["final_holdout"]
    config["data_role_manifests"] = {
        "data_roles": {
            "path": "data_roles_manifest.json",
            "manifest_hash": config["data_roles_manifest_hash"],
        },
        "optimizer_training": {
            "path": "training_manifest.json",
            "manifest_hash": manifest_hashes["optimizer_training"],
        },
        "controller": {
            "path": "controller_manifest.json",
            "manifest_hash": manifest_hashes["controller"],
        },
        "ordinary_validation": {
            "path": "validation_manifest.json",
            "manifest_hash": manifest_hashes["ordinary_validation"],
        },
        "final_holdout": {
            "path": "final_holdout_manifest.json",
            "manifest_hash": manifest_hashes["final_holdout"],
        },
    }
    config["validation_loss_aggregation"] = "target_token_weighted_causal_shift_float64"

    evaluation = config["evaluation"]
    evaluation["validation"]["manifest_hash"] = manifest_hashes["ordinary_validation"]
    evaluation["adaptive_controller"]["manifest_hash"] = manifest_hashes["controller"]
    evaluation["final_holdout"]["manifest_hash"] = manifest_hashes["final_holdout"]
    controller = config["model"]["adaptive_controller"]
    controller["controller_panel_contract"]["manifest_hash"] = manifest_hashes[
        "controller"
    ]
    controller["final_holdout_contract"]["manifest_hash"] = manifest_hashes[
        "final_holdout"
    ]
    controller["data_roles_manifest_hash"] = config["data_roles_manifest_hash"]
    controller["optimizer_training_manifest_hash"] = manifest_hashes[
        "optimizer_training"
    ]
    controller["controller_manifest_hash"] = manifest_hashes["controller"]
    controller["ordinary_validation_manifest_hash"] = manifest_hashes[
        "ordinary_validation"
    ]
    controller["final_holdout_manifest_hash"] = manifest_hashes["final_holdout"]

    signature, inputs = build_comparison_control_signature(config)
    config["comparison_control_signature"] = signature
    config["comparison_control_inputs"] = inputs


def write_probabilistic_data_role_artifacts(
    config: Mapping[str, Any],
    partition: Mapping[str, Any],
    distributed_context=None,
) -> None:
    """Validate resume manifests, then atomically persist the resolved split."""

    output_dir = Path(config["run"]["output_dir"])
    manifests = partition["role_manifests"]
    artifacts = {
        "data_roles_manifest.json": (
            partition["parent_manifest"],
            "parent_manifest_hash",
        ),
        "training_manifest.json": (
            manifests["optimizer_training"],
            "manifest_hash",
        ),
        "controller_manifest.json": (
            manifests["controller"],
            "manifest_hash",
        ),
        "validation_manifest.json": (
            manifests["ordinary_validation"],
            "manifest_hash",
        ),
        "final_holdout_manifest.json": (
            manifests["final_holdout"],
            "manifest_hash",
        ),
    }
    _validate_probabilistic_resume_manifests(config, output_dir, artifacts)
    for filename, (payload, _) in artifacts.items():
        write_json_artifact(
            output_dir / filename,
            payload,
            distributed_context=distributed_context,
            artifact_io=config,
        )
    write_config_artifact(config, distributed_context=distributed_context)


def _validate_probabilistic_resume_manifests(
    config: Mapping[str, Any],
    output_dir: Path,
    artifacts: Mapping[str, tuple[Mapping[str, Any], str]],
) -> None:
    continuation = config["run"].get("continuation", {})
    if not continuation.get("enabled", False):
        return
    checkpoint_dir = output_dir / "checkpoints"
    has_resume_checkpoint = any(
        (checkpoint_dir / filename).exists()
        for filename in ("latest.pt", "latest.prev.pt")
    )
    for filename, (expected_payload, hash_field) in artifacts.items():
        artifact_path = output_dir / filename
        if not artifact_path.exists():
            if has_resume_checkpoint:
                raise ConfigError(
                    "Bayesian resume requires the saved data-role manifest: "
                    f"{artifact_path}"
                )
            continue
        try:
            saved_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigError(
                f"Unable to validate Bayesian resume manifest {artifact_path}: {error}"
            ) from error
        saved_hash = saved_payload.get(hash_field)
        expected_hash = expected_payload[hash_field]
        if saved_hash != expected_hash:
            raise ConfigError(
                "Bayesian resume data-role manifest hash mismatch for "
                f"{filename}: saved={saved_hash!r}, expected={expected_hash!r}"
            )


def _probabilistic_manifest_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    return {
        "data_roles_manifest_hash": str(config["data_roles_manifest_hash"]),
        "optimizer_training_manifest_hash": str(
            config["optimizer_training_manifest_hash"]
        ),
        "controller_manifest_hash": str(config["controller_manifest_hash"]),
        "ordinary_validation_manifest_hash": str(config["validation_manifest_hash"]),
        "final_holdout_manifest_hash": str(config["final_holdout_manifest_hash"]),
    }


def _build_or_restore_probabilistic_controller(
    config: Mapping[str, Any],
    run_state: Mapping[str, Any],
):
    controller_config = config["model"]["adaptive_controller"]
    sampling_seed = seed_for(config, "posterior_sampling")
    manifest_hashes = _probabilistic_manifest_hashes(config)
    saved_state = run_state.get("probabilistic_controller_state")
    if isinstance(saved_state, Mapping):
        source_checkpoint = run_state.get(
            "continuation_source_checkpoint_path",
            run_state.get("latest_checkpoint_path", "unknown-checkpoint"),
        )
        return restore_probabilistic_controller(
            saved_state,
            controller_config=controller_config,
            sampling_seed=sampling_seed,
            expected_manifest_hashes=manifest_hashes,
            source_checkpoint=source_checkpoint,
        )
    return build_probabilistic_controller(
        controller_config=controller_config,
        sampling_seed=sampling_seed,
        manifest_hashes=manifest_hashes,
    )


def _controller_event_common_fields(
    config: Mapping[str, Any],
    controller_state: Mapping[str, Any],
) -> dict[str, Any]:
    resume = controller_state["resume"]
    return {
        "run_id": config["run"]["run_id"],
        "method_family": controller_state["method_family"],
        "method_version": controller_state["method_version"],
        "strategy": controller_state["strategy"],
        "scope": controller_state["scope"],
        "ordered_granularities": list(controller_state["ordered_granularities"]),
        "feature_schema": copy.deepcopy(controller_state["feature_schema"]),
        "feature_schema_hash": controller_state["feature_schema"]["schema_hash"],
        "probabilistic_inputs": copy.deepcopy(controller_state["probabilistic_inputs"]),
        "controller_manifest_hash": controller_state["manifest_hashes"][
            "controller_manifest_hash"
        ],
        "data_roles_manifest_hash": controller_state["manifest_hashes"][
            "data_roles_manifest_hash"
        ],
        "decision_interval_steps": controller_state["window"][
            "decision_interval_steps"
        ],
        "resume_count": resume.get("resume_count", 0),
        "resume_source_checkpoint": resume.get("source_checkpoint"),
    }


def _attach_probabilistic_controller_artifact_provenance(
    config: dict[str, Any],
    controller_state: Mapping[str, Any],
) -> None:
    """Persist the resolved feature identity and stable controller artifact links."""

    controller_config = config["model"]["adaptive_controller"]
    controller_config["feature_schema"] = copy.deepcopy(
        controller_state["feature_schema"]
    )
    controller_config["controller_metrics_path"] = "controller_metrics.jsonl"
    controller_config["controller_summary_path"] = "controller_summary.json"
    config["controller_metrics_path"] = "controller_metrics.jsonl"
    config["controller_summary_path"] = "controller_summary.json"


def _enrich_controller_event(
    config: Mapping[str, Any],
    controller,
    event: Mapping[str, Any],
    controller_events: list[dict[str, Any]],
) -> dict[str, Any]:
    state = controller.state_dict()
    enriched = {
        **_controller_event_common_fields(config, state),
        **dict(event),
    }
    if "boundary_step" not in enriched and "boundary_step_end" in enriched:
        enriched["boundary_step"] = enriched["boundary_step_end"]
    enriched["sample_count"] = state["sampling"]["sample_count"]
    enriched["action_frequencies"] = controller_action_frequency_counts(
        [*controller_events, enriched]
    )
    enriched["uncertainty_summary"] = controller_uncertainty_summary(
        state["belief"]["posterior_covariance"]
    )
    return enriched


def _commit_controller_event(
    config: Mapping[str, Any],
    controller,
    event: Mapping[str, Any],
    controller_events: list[dict[str, Any]],
    *,
    distributed_context,
    heartbeat_writer,
    run_state: dict[str, Any],
) -> dict[str, Any]:
    enriched = _enrich_controller_event(
        config,
        controller,
        event,
        controller_events,
    )
    journal_path = Path(config["run"]["output_dir"]) / "controller_metrics.jsonl"
    commit = append_controller_event(
        journal_path,
        enriched,
        distributed_context=distributed_context,
        artifact_io=config,
        heartbeat_writer=heartbeat_writer,
        artifact_state=run_state,
    )
    if commit is not None:
        controller.record_journal_commit(commit)
        controller_events.append(enriched)
        print(format_controller_lifecycle_log(enriched), flush=True)
    run_state["latest_controller_event"] = enriched
    run_state["probabilistic_controller_state"] = controller.state_dict()
    return enriched


def _evaluate_probabilistic_boundary(
    config: Mapping[str, Any],
    model,
    controller_dataloader,
    device: torch.device,
    distributed_context,
    *,
    boundary_step: int,
) -> dict[str, Any]:
    return evaluate_controller_objective(
        model,
        controller_dataloader,
        granularities=list(config["model"]["granularities"]),
        device=device,
        distributed=bool(
            distributed_context is not None and distributed_context.enabled
        ),
        config=config,
        controller_manifest_hash=config["controller_manifest_hash"],
        boundary_step=int(boundary_step),
    )


def _controller_error_fields(error: BaseException) -> tuple[str, str | None]:
    message = str(error).lower()
    if "component" in message and ("non-finite" in message or "finite" in message):
        return "non_finite_component_loss", "ordered_component_losses"
    if "objective" in message and ("non-finite" in message or "finite" in message):
        return "non_finite_objective", "uniform_objective"
    if "reward" in message and "finite" in message:
        return "non_finite_reward", "reward"
    if "covariance" in message:
        return "invalid_covariance", "posterior_covariance"
    if isinstance(error, (FloatingPointError, ArithmeticError)):
        return "numerical_error", None
    return type(error).__name__.lower(), None


def _record_controller_failure(
    config: Mapping[str, Any],
    controller,
    controller_events: list[dict[str, Any]],
    error: BaseException,
    *,
    boundary_step: int,
    failing_stage: str,
    distributed_context,
    heartbeat_writer,
    run_state: dict[str, Any],
) -> None:
    error_category, offending_field = _controller_error_fields(error)
    event = controller.fail(
        boundary_step=boundary_step,
        failing_stage=failing_stage,
        error_category=error_category,
        error_message=str(error),
        offending_field=offending_field,
    )
    run_state["probabilistic_controller_state"] = controller.state_dict()
    _commit_controller_event(
        config,
        controller,
        event,
        controller_events,
        distributed_context=distributed_context,
        heartbeat_writer=heartbeat_writer,
        run_state=run_state,
    )


def run_training(
    config: dict[str, Any],
    model=None,
    tokenizer=None,
    tokenized_dataset=None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    ensure_single_process_runtime()
    validate_run_config(config)
    deterministic_settings = configure_strict_determinism(config)
    seed_model_initialization(config)
    run = config["run"]
    training = config["training"]
    output_dir = Path(run["output_dir"])
    run_state = training_checkpointing.build_initial_continuation_state(config)
    checkpoint_state: dict[str, Any] = {}
    metrics_journal = None
    continuation_load_succeeded = not bool(run["continuation"]["enabled"])
    optimizer = None
    scheduler = None
    distributed_context = training_distributed.prepare_distributed_context(
        config,
        device=device,
    )
    training_modeling.sync_config_with_distributed_context(
        config,
        distributed_context,
    )
    monitoring_session = training_monitoring.create_monitoring_session(
        config,
        distributed_context,
    )
    heartbeat_writer = training_monitoring.build_heartbeat_writer(
        config,
        distributed_context,
    )
    parameter_counts_by_granularity = {}
    _controller_dataloader = None
    probabilistic_controller = None
    controller_events: list[dict[str, Any]] = []
    controller_summary = None
    controller_summary_path = None

    with training_monitoring.heartbeat_stage(heartbeat_writer, "artifact_writing"):
        write_config_artifact(config, distributed_context=distributed_context)

    device = torch.device(distributed_context.device)

    try:
        with training_monitoring.heartbeat_stage(
            heartbeat_writer, "model_initialization"
        ):
            if model is None:
                model = training_modeling.build_model(config)
            seed_training_randomness(config)
            record_runtime_sampling_provenance(model, config)
            if (
                distributed_context.is_rank_zero
                and config["model"]["variant"] == "concat"
            ):
                diagnostic = build_concat_layout_diagnostic(
                    config["model"]["intermediate_size"],
                    config["model"]["granularities"],
                    granularity_prefixes=config["model"].get("granularity_prefixes"),
                )
                print(f"[concat-diagnostic] {diagnostic}", flush=True)
            parameter_counts_by_granularity = (
                training_modeling.build_artifact_parameter_counts(
                    config,
                    model,
                    distributed_context,
                )
            )
            if parameter_counts_by_granularity:
                attach_parameter_counts_to_config(
                    config,
                    parameter_counts_by_granularity,
                )
            model = model.to(device)

        if parameter_counts_by_granularity:
            with training_monitoring.heartbeat_stage(
                heartbeat_writer,
                "artifact_writing",
            ):
                write_config_artifact(config, distributed_context=distributed_context)

        with training_monitoring.heartbeat_stage(heartbeat_writer, "fsdp_wrapping"):
            model = training_distributed.wrap_model_for_distributed(
                model,
                distributed_context,
            )

        if tokenized_dataset is None:
            if tokenizer is None:
                with training_monitoring.heartbeat_stage(
                    heartbeat_writer,
                    "tokenizer_loading",
                ):
                    tokenizer = training_modeling.load_tokenizer(config)
            with training_monitoring.heartbeat_stage(
                heartbeat_writer,
                "dataset_loading_preprocessing",
            ):
                tokenized_dataset = training_data.load_and_tokenize_dataset(
                    config,
                    tokenizer,
                    num_proc=training.get("preprocess_num_proc", 1),
                )

        with training_monitoring.heartbeat_stage(
            heartbeat_writer,
            "dataloader_creation",
        ):
            if uses_probabilistic_controller(config):
                (
                    train_dataloader,
                    eval_dataloader,
                    _controller_dataloader,
                    role_partition,
                ) = prepare_probabilistic_data_roles(
                    config,
                    tokenized_dataset,
                    device,
                    distributed_context=distributed_context,
                )
                write_probabilistic_data_role_artifacts(
                    config,
                    role_partition,
                    distributed_context=distributed_context,
                )
                if training_distributed.should_write_shared_artifact(
                    distributed_context
                ):
                    print(
                        "[data-roles] manifest_hash="
                        f"{config['data_roles_manifest_hash']} "
                        "controller_manifest_hash="
                        f"{config['controller_manifest_hash']} "
                        "final_holdout_manifest_hash="
                        f"{config['final_holdout_manifest_hash']}",
                        flush=True,
                    )
            else:
                train_dataloader, eval_dataloader = training_data.build_dataloaders(
                    config,
                    tokenized_dataset,
                    device,
                    distributed_context=distributed_context,
                )
                validation_manifest = config.pop("_validation_manifest")
                if training_distributed.should_write_shared_artifact(
                    distributed_context
                ):
                    write_json_artifact(
                        output_dir / "validation_manifest.json",
                        validation_manifest,
                        distributed_context=distributed_context,
                        artifact_io=config,
                    )
                    print(
                        "[validation] manifest_hash="
                        f"{config['validation_manifest_hash']}",
                        flush=True,
                    )
        optimizer, scheduler = training_steps.build_optimizer_and_scheduler(
            model,
            training,
        )
        if run["continuation"]["enabled"]:
            run_state = training_checkpointing.load_run_continuation_state(
                config,
                model,
                optimizer,
                scheduler,
                distributed_context=distributed_context,
            )
            continuation_load_succeeded = True
        if uses_probabilistic_controller(config):
            probabilistic_controller = _build_or_restore_probabilistic_controller(
                config,
                run_state,
            )
            controller_state = probabilistic_controller.state_dict()
            run_state["probabilistic_controller_state"] = controller_state
            _attach_probabilistic_controller_artifact_provenance(
                config,
                controller_state,
            )
            write_config_artifact(
                config,
                distributed_context=distributed_context,
            )
            controller_events = read_controller_events(
                output_dir / "controller_metrics.jsonl"
            )
        training_monitoring.emit_run_start_continuation_state(
            heartbeat_writer,
            run_state,
        )
        checkpoint_state.update(run_state)
        training_checkpointing.update_run_continuation_state(config, run_state)
        metrics_journal = MetricsJournal(
            output_dir,
            flush_interval_steps=int(config["outputs"]["metrics_flush_interval_steps"]),
            checkpoint_step=int(run_state.get("last_completed_step", 0)),
            artifact_io_config=config,
            heartbeat_writer=heartbeat_writer,
            artifact_state=run_state,
        )
        metrics_rows = []
        if training_warmup.should_run_pre_nested_warmup(config, run_state):
            metrics_rows.extend(
                training_warmup.run_pre_nested_warmup_phase(
                    config,
                    model,
                    train_dataloader,
                    eval_dataloader,
                    optimizer,
                    scheduler,
                    device,
                    heartbeat_writer=heartbeat_writer,
                    distributed_context=distributed_context,
                    checkpoint_state=checkpoint_state,
                    run_state=run_state,
                    monitoring_session=monitoring_session,
                    metrics_journal=metrics_journal,
                )
            )
        else:
            training_warmup.update_pre_nested_warmup_state(
                config,
                training_warmup.build_pre_nested_warmup_state(
                    config,
                    completed=bool(run_state.get("warmup_completed", False)),
                    completion_step=(
                        int(run_state["warmup_completion_step"])
                        if run_state.get("warmup_completion_step") is not None
                        else None
                    ),
                    transition_reason=run_state.get("warmup_transition_reason"),
                ),
            )

        warmup_config = config["training"].get("pre_nested_warmup", {})
        warmup_active = bool(
            isinstance(warmup_config, Mapping) and warmup_config.get("active", False)
        )
        warmup_budget_exhausted = (
            warmup_active
            and not bool(run_state.get("warmup_completed", False))
            and (
                int(run_state.get("last_completed_step", 0))
                >= int(config["training"]["max_steps"])
                or int(run_state.get("tokens_seen", 0))
                >= int(config["training"]["token_budget"])
            )
        )

        def process_probabilistic_boundary(*, step: int, tokens_seen: int) -> None:
            if probabilistic_controller is None:
                return
            state = probabilistic_controller.state_dict()
            if state["window"]["phase"] != "boundary_evaluation_pending":
                run_state["probabilistic_controller_state"] = state
                return
            snapshot = probabilistic_controller.transaction_snapshot()
            failing_stage = "controller_evaluation"
            try:
                objective = _evaluate_probabilistic_boundary(
                    config,
                    model,
                    _controller_dataloader,
                    device,
                    distributed_context,
                    boundary_step=step,
                )
                training_will_continue = int(step) < int(training["max_steps"]) and int(
                    tokens_seen
                ) < int(training["token_budget"])
                failing_stage = "posterior_update_and_action_selection"
                event = probabilistic_controller.complete_boundary(
                    boundary_step=step,
                    controller_objective=objective["uniform_objective"],
                    ordered_component_losses=objective["ordered_component_losses"],
                    evaluation_target_tokens=int(objective["evaluation_target_tokens"]),
                    training_will_continue=training_will_continue,
                )
                failing_stage = "controller_journal_commit"
                _commit_controller_event(
                    config,
                    probabilistic_controller,
                    event,
                    controller_events,
                    distributed_context=distributed_context,
                    heartbeat_writer=heartbeat_writer,
                    run_state=run_state,
                )
            except Exception as error:
                probabilistic_controller.restore_transaction_snapshot(snapshot)
                try:
                    _record_controller_failure(
                        config,
                        probabilistic_controller,
                        controller_events,
                        error,
                        boundary_step=step,
                        failing_stage=failing_stage,
                        distributed_context=distributed_context,
                        heartbeat_writer=heartbeat_writer,
                        run_state=run_state,
                    )
                except Exception as failure_record_error:
                    print(
                        "Failed to persist controller failure record: "
                        f"{failure_record_error}",
                        flush=True,
                    )
                raise

        def finish_probabilistic_training(*, step: int, tokens_seen: int) -> None:
            del tokens_seen
            if probabilistic_controller is None:
                return
            event = probabilistic_controller.finish_training()
            if event is None:
                run_state["probabilistic_controller_state"] = (
                    probabilistic_controller.state_dict()
                )
                return
            _commit_controller_event(
                config,
                probabilistic_controller,
                event,
                controller_events,
                distributed_context=distributed_context,
                heartbeat_writer=heartbeat_writer,
                run_state=run_state,
            )

        if probabilistic_controller is not None and not warmup_budget_exhausted:
            controller_state = probabilistic_controller.state_dict()
            boundary_step = int(run_state.get("last_completed_step", 0))
            if controller_state["window"]["phase"] == "initial_objective_pending":
                snapshot = probabilistic_controller.transaction_snapshot()
                failing_stage = "initial_controller_evaluation"
                try:
                    objective = _evaluate_probabilistic_boundary(
                        config,
                        model,
                        _controller_dataloader,
                        device,
                        distributed_context,
                        boundary_step=boundary_step,
                    )
                    failing_stage = "initial_action_selection"
                    event = probabilistic_controller.initialize_boundary(
                        boundary_step=boundary_step,
                        controller_objective=objective["uniform_objective"],
                        ordered_component_losses=objective["ordered_component_losses"],
                        evaluation_target_tokens=int(
                            objective["evaluation_target_tokens"]
                        ),
                    )
                    failing_stage = "controller_journal_commit"
                    _commit_controller_event(
                        config,
                        probabilistic_controller,
                        event,
                        controller_events,
                        distributed_context=distributed_context,
                        heartbeat_writer=heartbeat_writer,
                        run_state=run_state,
                    )
                except Exception as error:
                    probabilistic_controller.restore_transaction_snapshot(snapshot)
                    try:
                        _record_controller_failure(
                            config,
                            probabilistic_controller,
                            controller_events,
                            error,
                            boundary_step=boundary_step,
                            failing_stage=failing_stage,
                            distributed_context=distributed_context,
                            heartbeat_writer=heartbeat_writer,
                            run_state=run_state,
                        )
                    except Exception as failure_record_error:
                        print(
                            "Failed to persist controller failure record: "
                            f"{failure_record_error}",
                            flush=True,
                        )
                    raise
            elif controller_state["window"]["phase"] == "boundary_evaluation_pending":
                process_probabilistic_boundary(
                    step=boundary_step,
                    tokens_seen=int(run_state.get("tokens_seen", 0)),
                )

        if not warmup_budget_exhausted:
            metrics_rows.extend(
                training_steps.train_for_steps(
                    config,
                    model,
                    train_dataloader,
                    eval_dataloader,
                    optimizer,
                    scheduler,
                    device,
                    heartbeat_writer=heartbeat_writer,
                    distributed_context=distributed_context,
                    checkpoint_state=checkpoint_state,
                    run_state=run_state,
                    monitoring_session=monitoring_session,
                    stage_name="training",
                    metrics_journal=metrics_journal,
                    probabilistic_controller=probabilistic_controller,
                    probabilistic_boundary_callback=process_probabilistic_boundary,
                    probabilistic_completion_callback=finish_probabilistic_training,
                )
            )
        metrics_rows = metrics_journal.read_all()
        extraction_metadata_path = None
        metrics_path = None
        scaling_path = None
        scaling_rows = []
        checkpoint_summary_fields = build_checkpoint_summary_fields(
            config,
            metrics_rows,
        )

        completed_run_state = dict(run_state)
        completed_run_state["status"] = "completed"
        if run["continuation"]["enabled"]:
            completed_run_state["latest_checkpoint_path"] = completed_run_state.get(
                "latest_checkpoint_path"
            ) or str(output_dir / "checkpoints" / "latest.pt")
        checkpoint_summary_fields = training_checkpointing.write_checkpoint_if_needed(
            config,
            model,
            optimizer,
            scheduler,
            metrics_rows,
            heartbeat_writer,
            completed_run_state,
            distributed_context=distributed_context,
        )

        if run["continuation"]["enabled"]:
            run_state.update(completed_run_state)
            training_checkpointing.update_run_continuation_state(config, run_state)

        if training_distributed.should_write_shared_artifact(distributed_context):
            with training_monitoring.heartbeat_stage(
                heartbeat_writer,
                "artifact_writing",
            ):
                extraction_metadata_path = (
                    training_steps.write_extraction_metadata_if_nested(
                        config,
                        model,
                        output_dir,
                        distributed_context=distributed_context,
                    )
                )
                metrics_path = write_metrics_csv(
                    output_dir,
                    metrics_rows,
                    distributed_context=distributed_context,
                    artifact_io=config,
                    heartbeat_writer=heartbeat_writer,
                    artifact_state=run_state,
                )
                write_config_artifact(config, distributed_context=distributed_context)
                scaling_rows = build_scaling_result_rows(
                    config,
                    metrics_rows,
                    parameter_counts_by_granularity,
                )
                if scaling_rows:
                    scaling_path = write_scaling_results_csv(
                        output_dir,
                        scaling_rows,
                        distributed_context=distributed_context,
                    )
                if probabilistic_controller is not None:
                    controller_summary = build_controller_summary(
                        controller_state=probabilistic_controller.state_dict(),
                        controller_events=controller_events,
                        controller_metrics_path=(
                            output_dir / "controller_metrics.jsonl"
                        ),
                    )
                    controller_summary_path = write_controller_summary(
                        output_dir,
                        controller_summary,
                        distributed_context=distributed_context,
                        artifact_io=config,
                        heartbeat_writer=heartbeat_writer,
                        artifact_state=run_state,
                    )

        training_outcome = training_steps.summarize_training_outcome(
            config, metrics_rows
        )
        tokens_seen = training_outcome["tokens_seen"]
        target_model = getattr(model, "module", model)
        runtime_pattern = getattr(target_model, "current_granularity_pattern", None)
        if str(config["run"].get("sampling_mode", "nested-random")) == "nested-all":
            runtime_pattern = None
        runtime_pattern_summary = summarize_runtime_granularity_pattern_from_config(
            config,
            runtime_pattern=runtime_pattern,
        )
        correction_context = summarize_correction_context_from_config(
            config,
            granularity_pattern=runtime_pattern,
        )
        resolved_run_mode = str(config["run"].get("sampling_mode", "nested-random"))
        config["run"]["resolved_run_mode"] = resolved_run_mode
        config["model"]["resolved_sampling_mode"] = config["model"].get(
            "granularity_sampling_mode",
            "global",
        )
        config["model"]["granularity_pattern_summary"] = runtime_pattern_summary
        config["model"]["correction_context"] = correction_context
        extra_summary_fields = {
            "steps_completed": training_outcome["steps_completed"],
            "stop_reason": training_outcome["stop_reason"],
            "content_tokens_seen": training_outcome["content_tokens_seen"],
            **training_modeling.build_granularity_artifact_fields(config),
            "model_variant": config["model"]["variant"],
            "granularities": config["model"]["granularities"],
            "granularity_sampling": training.get("granularity_sampling", "all"),
            "resolved_run_mode": resolved_run_mode,
            "resolved_sampling_mode": config["model"].get(
                "granularity_sampling_mode",
                "global",
            ),
            "granularity_pattern_summary": runtime_pattern_summary,
            "correction_context": correction_context,
            "parameter_counts_by_granularity": parameter_counts_by_granularity,
            **build_monitoring_summary_fields(config, metrics_rows),
            **checkpoint_summary_fields,
            **training_modeling.distributed_summary_fields(distributed_context),
            **build_adaptive_sampler_artifact_fields(config, run_state),
            "requested_mixed_precision": training.get("requested_mixed_precision"),
            "resolved_mixed_precision": training.get("resolved_mixed_precision"),
            "requested_activation_checkpointing": training.get(
                "requested_activation_checkpointing"
            ),
            "resolved_activation_checkpointing": training.get(
                "resolved_activation_checkpointing"
            ),
            "final_validation": config.get("evaluation", {})
            .get("validation", {})
            .get("run_at_completion"),
            "final_validation_reason": config.get("evaluation", {})
            .get("validation", {})
            .get("run_at_completion_reason"),
            "artifact_retry_count": int(run_state.get("artifact_retry_count", 0)),
            "validation_manifest_hash": config.get("validation_manifest_hash"),
            "validation_loss_aggregation": config.get("validation_loss_aggregation"),
            "comparison_control_signature": config.get("comparison_control_signature"),
            "data_roles_manifest_hash": config.get("data_roles_manifest_hash"),
            "optimizer_training_manifest_hash": config.get(
                "optimizer_training_manifest_hash"
            ),
            "controller_manifest_hash": config.get("controller_manifest_hash"),
            "final_holdout_manifest_hash": config.get("final_holdout_manifest_hash"),
            "data_role_manifests": config.get("data_role_manifests"),
            "controller_summary": controller_summary,
            "controller_metrics_path": (
                str(output_dir / "controller_metrics.jsonl")
                if probabilistic_controller is not None
                else None
            ),
            "controller_summary_path": (
                str(controller_summary_path)
                if controller_summary_path is not None
                else None
            ),
            "controller_summary_hash": (
                stable_hash(controller_summary)
                if controller_summary is not None
                else None
            ),
            "controller_metrics_hash": (
                controller_summary.get("controller_metrics_hash")
                if controller_summary is not None
                else None
            ),
            "deterministic_runtime_settings": deterministic_settings,
            "artifact_last_errno": run_state.get("artifact_last_errno"),
            "last_durable_checkpoint_step": int(
                run_state.get("last_durable_checkpoint_step", 0)
            ),
            "deferred_metric_rows": int(run_state.get("deferred_metric_rows", 0)),
            "skipped_periodic_checkpoints": int(
                run_state.get("skipped_periodic_checkpoints", 0)
            ),
            "checkpoint_staging_mode": run_state.get(
                "checkpoint_staging_mode", "direct"
            ),
            "unresolved_artifact_failures": list(
                run_state.get("unresolved_artifact_failures", [])
            ),
        }
        if controller_summary is not None:
            extra_summary_fields.update(
                {
                    "controller_method_family": controller_summary.get("method_family"),
                    "controller_method_version": controller_summary.get(
                        "method_version"
                    ),
                    "controller_strategy": controller_summary.get("strategy"),
                    "controller_scope": controller_summary.get("scope"),
                    "controller_feature_schema": controller_summary.get(
                        "feature_schema"
                    ),
                    "controller_probabilistic_inputs": controller_summary.get(
                        "probabilistic_inputs"
                    ),
                }
            )
        if not scaling_rows:
            extra_summary_fields["scaling_results_unavailable_reason"] = (
                "no validation rows were produced; scaling comparisons require "
                "uniform validation metrics"
            )
        if metrics_path is not None:
            extra_summary_fields["metrics_path"] = str(metrics_path)
        if scaling_path is not None:
            extra_summary_fields["scaling_results_path"] = str(scaling_path)
        if extraction_metadata_path is not None:
            extra_summary_fields["extraction_metadata_path"] = str(
                extraction_metadata_path
            )

        summary = build_run_summary(
            config,
            tokens_seen=tokens_seen,
            notes=["completed config-driven training loop"],
            extra_fields=extra_summary_fields,
        )
        with training_monitoring.heartbeat_stage(heartbeat_writer, "artifact_writing"):
            summary_path = write_run_summary(
                output_dir,
                summary,
                distributed_context=distributed_context,
                artifact_io=config,
                heartbeat_writer=heartbeat_writer,
                artifact_state=run_state,
            )

        return {
            "config": config,
            "metrics_path": metrics_path,
            "scaling_path": scaling_path,
            "summary_path": summary_path,
            "metrics_rows": metrics_rows,
            "scaling_rows": scaling_rows,
            "parameter_counts_by_granularity": parameter_counts_by_granularity,
            "controller_summary_path": controller_summary_path,
        }
    except Exception as error:
        try:
            if metrics_journal is not None:
                metrics_journal.flush()
            if (
                run["continuation"]["enabled"]
                and continuation_load_succeeded
                and model is not None
                and optimizer is not None
                and scheduler is not None
            ):
                training_checkpointing.maybe_write_latest_checkpoint(
                    config,
                    model,
                    optimizer,
                    scheduler,
                    heartbeat_writer,
                    run_state,
                    reason="failure",
                    step=int(
                        run_state.get("step", run_state.get("last_completed_step", 0))
                    ),
                    distributed_context=distributed_context,
                    force=True,
                )
            run_state["status"] = "failed"
            if run["continuation"]["enabled"]:
                training_checkpointing.update_run_continuation_state(config, run_state)
            with training_monitoring.heartbeat_stage(
                heartbeat_writer,
                "artifact_writing",
            ):
                if (
                    probabilistic_controller is not None
                    and training_distributed.should_write_shared_artifact(
                        distributed_context
                    )
                ):
                    controller_summary = build_controller_summary(
                        controller_state=probabilistic_controller.state_dict(),
                        controller_events=controller_events,
                        controller_metrics_path=(
                            output_dir / "controller_metrics.jsonl"
                        ),
                    )
                    controller_summary_path = write_controller_summary(
                        output_dir,
                        controller_summary,
                        distributed_context=distributed_context,
                        artifact_io=config,
                        heartbeat_writer=heartbeat_writer,
                        artifact_state=run_state,
                    )
                failure_extra_fields = {
                    "data_roles_manifest_hash": config.get(
                        "data_roles_manifest_hash"
                    ),
                    "optimizer_training_manifest_hash": config.get(
                        "optimizer_training_manifest_hash"
                    ),
                    "controller_manifest_hash": config.get(
                        "controller_manifest_hash"
                    ),
                    "final_holdout_manifest_hash": config.get(
                        "final_holdout_manifest_hash"
                    ),
                    "data_role_manifests": config.get("data_role_manifests"),
                    "controller_summary": controller_summary,
                    "controller_metrics_path": (
                        str(output_dir / "controller_metrics.jsonl")
                        if probabilistic_controller is not None
                        else None
                    ),
                    "controller_summary_path": (
                        str(controller_summary_path)
                        if controller_summary_path is not None
                        else None
                    ),
                    "controller_summary_hash": (
                        stable_hash(controller_summary)
                        if controller_summary is not None
                        else None
                    ),
                    "controller_metrics_hash": (
                        controller_summary.get("controller_metrics_hash")
                        if controller_summary is not None
                        else None
                    ),
                }
                failure_summary = build_run_summary(
                    config,
                    tokens_seen=int(run_state.get("tokens_seen", 0)),
                    content_tokens_seen=int(run_state.get("content_tokens_seen", 0)),
                    status="failed",
                    notes=[str(error)],
                    extra_fields=failure_extra_fields,
                )
                write_run_summary(
                    output_dir,
                    failure_summary,
                    distributed_context=distributed_context,
                    artifact_io=config,
                    heartbeat_writer=heartbeat_writer,
                    artifact_state=run_state,
                )
        except Exception as summary_error:
            print(
                "Failed to write failure summary: "
                f"{summary_error}. Original error: {error}",
                flush=True,
            )
        raise
    finally:
        monitoring_session.close()
        training_distributed.destroy_distributed_process_group(distributed_context)
