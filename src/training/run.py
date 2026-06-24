"""Config-driven training orchestration for MatFormer reproduction runs."""

from __future__ import annotations

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*args, **kwargs):
        return None

load_dotenv()

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
from src.models.adaptive_sampler import (
    build_adaptive_sampler_artifact_fields,
)
from src.models.correction import summarize_correction_context_from_config
from src.models.ffn import build_concat_layout_diagnostic
from src.models.wiring import (
    record_runtime_sampling_provenance,
)
from src.training.steps import set_random_seed
from src.utils.config import (
    ConfigError,
    attach_parameter_counts_to_config,
    resolve_run_config,
    validate_run_config,
)
from src.utils.metrics import (
    build_checkpoint_summary_fields,
    build_monitoring_summary_fields,
    build_run_summary,
    build_scaling_result_rows,
    summarize_runtime_granularity_pattern_from_config,
    write_config_artifact,
    write_failed_run_summary,
    write_metrics_csv,
    write_run_summary,
    write_scaling_results_csv,
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


def run_training(
    config: dict[str, Any],
    model=None,
    tokenizer=None,
    tokenized_dataset=None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    ensure_single_process_runtime()
    validate_run_config(config)
    run = config["run"]
    training = config["training"]
    output_dir = Path(run["output_dir"])
    run_state = training_checkpointing.build_initial_continuation_state(config)
    checkpoint_state: dict[str, Any] = {}
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

    with training_monitoring.heartbeat_stage(heartbeat_writer, "artifact_writing"):
        write_config_artifact(config, distributed_context=distributed_context)
    set_random_seed(run.get("seed"))

    device = torch.device(distributed_context.device)

    try:
        with training_monitoring.heartbeat_stage(heartbeat_writer, "model_initialization"):
            if model is None:
                model = training_modeling.build_model(config)
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
            parameter_counts_by_granularity = training_modeling.build_artifact_parameter_counts(
                config,
                model,
                distributed_context,
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
            train_dataloader, eval_dataloader = training_data.build_dataloaders(
                config,
                tokenized_dataset,
                device,
                distributed_context=distributed_context,
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
        training_monitoring.emit_run_start_continuation_state(
            heartbeat_writer,
            run_state,
        )
        checkpoint_state.update(run_state)
        training_checkpointing.update_run_continuation_state(config, run_state)
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
                )
            )
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
                extraction_metadata_path = training_steps.write_extraction_metadata_if_nested(
                    config,
                    model,
                    output_dir,
                    distributed_context=distributed_context,
                )
                metrics_path = write_metrics_csv(
                    output_dir,
                    metrics_rows,
                    distributed_context=distributed_context,
                )
                write_config_artifact(config, distributed_context=distributed_context)
                scaling_rows = build_scaling_result_rows(
                    config,
                    metrics_rows,
                    parameter_counts_by_granularity,
                )
                scaling_path = write_scaling_results_csv(
                    output_dir,
                    scaling_rows,
                    distributed_context=distributed_context,
                )

        training_outcome = training_steps.summarize_training_outcome(config, metrics_rows)
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
        }
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
            )

        return {
            "config": config,
            "metrics_path": metrics_path,
            "scaling_path": scaling_path,
            "summary_path": summary_path,
            "metrics_rows": metrics_rows,
            "scaling_rows": scaling_rows,
            "parameter_counts_by_granularity": parameter_counts_by_granularity,
        }
    except Exception as error:
        try:
            if (
                run["continuation"]["enabled"]
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
                    step=int(run_state.get("step", run_state.get("last_completed_step", 0))),
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
                write_failed_run_summary(
                    config,
                    str(error),
                    output_dir=output_dir,
                    tokens_seen=int(run_state.get("tokens_seen", 0)),
                    content_tokens_seen=int(
                        run_state.get("content_tokens_seen", 0)
                    ),
                    distributed_context=distributed_context,
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


    return result
