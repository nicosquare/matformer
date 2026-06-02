"""Config-driven training flow for MatFormer reproduction runs."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import copy
from contextlib import contextmanager
import random
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, get_scheduler

from evaluation.validation import (
    configure_model_granularity,
    evaluate_validation_per_granularity,
    move_batch_to_device,
    perplexity_from_loss,
    validation_results_to_metric_rows,
)
from modified_llama import (
    CatLlamaMLP,
    ModifiedLlamaForCausalLM,
    get_concat_layout_diagnostic,
    get_ffn_prefix_metadata,
)
from training.data import (
    build_language_model_dataloader,
    load_and_tokenize_dataset,
    split_train_eval_dataset,
)
from training.distributed import (
    broadcast_object,
    destroy_distributed_process_group,
    prepare_distributed_context,
    should_write_shared_artifact,
    sum_int,
    wrap_model_for_distributed,
)
from utils.config import (
    ConfigError,
    attach_parameter_counts_to_config,
    resolve_run_config,
    resolve_optimizer_kwargs,
    resolve_training_length_for_world_size,
)
from utils.heartbeats import HeartbeatCadence, HeartbeatWriter
from utils.metrics import (
    build_checkpoint_summary_fields,
    build_monitoring_summary_fields,
    build_parameter_counts_by_granularity,
    build_run_summary,
    build_scaling_result_rows,
    write_config_artifact,
    write_failed_run_summary,
    write_json_artifact,
    write_metrics_csv,
    write_run_summary,
    write_scaling_results_csv,
)
from utils.monitoring import group_loss_rows_by_series


def run_from_config_path(
    config_path: str | Path,
    run_id: str | None = None,
    overrides: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = resolve_run_config(
        config_path,
        run_id=run_id,
        overrides=overrides,
        output_dir=output_dir,
    )
    return run_training(config)


def run_training(
    config: dict[str, Any],
    model=None,
    tokenizer=None,
    tokenized_dataset=None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    run = config["run"]
    training = config["training"]
    output_dir = Path(run["output_dir"])
    run_state = build_initial_continuation_state(config)
    checkpoint_state: dict[str, Any] = {}
    optimizer = None
    scheduler = None
    distributed_context = prepare_distributed_context(config, device=device)
    sync_config_with_distributed_context(config, distributed_context)
    monitoring_session = create_monitoring_session(config, distributed_context)
    heartbeat_writer = build_heartbeat_writer(config, distributed_context)
    parameter_counts_by_granularity = {}

    with heartbeat_stage(heartbeat_writer, "artifact_writing"):
        write_config_artifact(config, distributed_context=distributed_context)
    set_random_seed(run.get("seed"))

    device = torch.device(distributed_context.device)

    try:
        with heartbeat_stage(heartbeat_writer, "model_initialization"):
            if model is None:
                model = build_model(config)
            if (
                distributed_context.is_rank_zero
                and config["model"]["variant"] == "cat_llama"
            ):
                diagnostic = get_concat_layout_diagnostic(
                    config["model"]["intermediate_size"],
                    config["model"]["granularities"],
                    granularity_prefixes=config["model"].get("granularity_prefixes"),
                )
                print(f"[cat-llama-diagnostic] {diagnostic}", flush=True)
            parameter_counts_by_granularity = build_artifact_parameter_counts(
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
            with heartbeat_stage(heartbeat_writer, "artifact_writing"):
                write_config_artifact(config, distributed_context=distributed_context)

        with heartbeat_stage(heartbeat_writer, "fsdp_wrapping"):
            model = wrap_model_for_distributed(model, distributed_context)

        if tokenized_dataset is None:
            if tokenizer is None:
                with heartbeat_stage(heartbeat_writer, "tokenizer_loading"):
                    tokenizer = load_tokenizer(config)
            with heartbeat_stage(heartbeat_writer, "dataset_loading_preprocessing"):
                tokenized_dataset = load_and_tokenize_dataset(
                    config,
                    tokenizer,
                    num_proc=training.get("preprocess_num_proc", 1),
                )

        with heartbeat_stage(heartbeat_writer, "dataloader_creation"):
            train_dataloader, eval_dataloader = build_dataloaders(
                config,
                tokenized_dataset,
                device,
                distributed_context=distributed_context,
            )
        optimizer, scheduler = build_optimizer_and_scheduler(model, training)
        if run["continuation"]["enabled"]:
            run_state = load_run_continuation_state(
                config,
                model,
                optimizer,
                scheduler,
                distributed_context=distributed_context,
            )
        emit_run_start_continuation_state(heartbeat_writer, run_state)
        checkpoint_state.update(run_state)
        update_run_continuation_state(config, run_state)
        metrics_rows = []
        if should_run_pre_nested_warmup(config, run_state):
            metrics_rows.extend(
                run_pre_nested_warmup_phase(
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
            update_pre_nested_warmup_state(
                config,
                build_pre_nested_warmup_state(
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
                train_for_steps(
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
        checkpoint_summary_fields = write_checkpoint_if_needed(
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
            update_run_continuation_state(config, run_state)

        if should_write_shared_artifact(distributed_context):
            with heartbeat_stage(heartbeat_writer, "artifact_writing"):
                extraction_metadata_path = write_extraction_metadata_if_nested(
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

        training_outcome = summarize_training_outcome(config, metrics_rows)
        tokens_seen = training_outcome["tokens_seen"]
        extra_summary_fields = {
            "steps_completed": training_outcome["steps_completed"],
            "stop_reason": training_outcome["stop_reason"],
            "content_tokens_seen": training_outcome["content_tokens_seen"],
            "model_variant": config["model"]["variant"],
            "granularities": config["model"]["granularities"],
            "granularity_sampling": training.get("granularity_sampling", "all"),
            "parameter_counts_by_granularity": parameter_counts_by_granularity,
            **build_monitoring_summary_fields(config, metrics_rows),
            **checkpoint_summary_fields,
            **distributed_summary_fields(distributed_context),
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
        with heartbeat_stage(heartbeat_writer, "artifact_writing"):
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
                maybe_write_latest_checkpoint(
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
                update_run_continuation_state(config, run_state)
            with heartbeat_stage(heartbeat_writer, "artifact_writing"):
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
        destroy_distributed_process_group(distributed_context)


def build_artifact_parameter_counts(
    config: dict[str, Any],
    model,
    distributed_context=None,
) -> dict[str, dict[str, Any]]:
    if not should_write_shared_artifact(distributed_context):
        return {}
    return build_parameter_counts_by_granularity(
        model,
        config["model"]["granularities"],
    )


def build_model(config: dict[str, Any]):
    llama_config = build_llama_config(config)
    mlp_kwargs = {
        "trained_granularities": tuple(config["model"]["granularities"]),
        "gradient_membership_correction_enabled": config["model"].get(
            "gradient_membership_correction",
            config["model"]["variant"] == "cat_llama",
        ),
    }
    if config["run"]["model_family"] == "standalone":
        return LlamaForCausalLM(llama_config)

    if config["model"]["variant"] == "cat_llama":
        return ModifiedLlamaForCausalLM(
            llama_config,
            mlp_cls=CatLlamaMLP,
            mlp_kwargs=mlp_kwargs,
        )

    return ModifiedLlamaForCausalLM(llama_config, mlp_kwargs=mlp_kwargs)


def build_llama_config(config: dict[str, Any]) -> LlamaConfig:
    model = config["model"]
    llama_config = LlamaConfig(
        vocab_size=model["vocab_size_assumption"],
        hidden_size=model.get("d_model", model.get("hidden_size")),
        intermediate_size=model["intermediate_size"],
        num_hidden_layers=model["num_layers"],
        num_attention_heads=model["num_attention_heads"],
        num_key_value_heads=model["num_attention_heads"],
        max_position_embeddings=model["context_length"],
        tie_word_embeddings=False,
        use_cache=False,
    )
    if "d_model" in model or "hidden_size" in model:
        llama_config.d_model = model.get("d_model", model.get("hidden_size"))
    if "granularities" in model:
        llama_config.granularities = list(model["granularities"])
    if "granularity_prefixes" in model:
        llama_config.granularity_prefixes = copy.deepcopy(
            model["granularity_prefixes"]
        )
    if "matformer_source_granularity_prefixes" in model:
        llama_config.matformer_source_granularity_prefixes = copy.deepcopy(
            model["matformer_source_granularity_prefixes"]
        )
    if "ffn_prefix_metadata" in model:
        llama_config.ffn_prefix_metadata = copy.deepcopy(model["ffn_prefix_metadata"])
    if "ffn_concat_block_metadata" in model:
        llama_config.ffn_concat_block_metadata = copy.deepcopy(
            model["ffn_concat_block_metadata"]
        )
    if "matformer_source_intermediate_size" in model:
        llama_config.matformer_source_intermediate_size = model[
            "matformer_source_intermediate_size"
        ]
    return llama_config


def load_tokenizer(config: dict[str, Any]):
    model = config["model"]
    dataset = config["dataset"]
    tokenizer_name = dataset.get("tokenizer_name") or model.get("tokenizer_name")
    tokenizer_name = tokenizer_name or model["base_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def sync_config_with_distributed_context(config: dict[str, Any], context) -> None:
    fsdp_config = copy.deepcopy(getattr(context, "fsdp_config", {}) or {})
    resolve_training_length_for_world_size(
        config,
        effective_world_size=context.world_size,
        world_size_source="distributed_context" if context.enabled else "single_process",
    )
    distributed_config = config["training"].setdefault("distributed", {})
    distributed_config.update(
        {
            "enabled": bool(context.enabled),
            "strategy": context.strategy,
            "rank": context.rank,
            "local_rank": context.local_rank,
            "world_size": context.world_size,
            "fsdp": fsdp_config,
        }
    )


def distributed_summary_fields(context) -> dict[str, Any]:
    fsdp_config = copy.deepcopy(getattr(context, "fsdp_config", {}) or {})
    return {
        "distributed_strategy": context.strategy,
        "distributed_rank": context.rank,
        "distributed_local_rank": context.local_rank,
        "distributed_world_size": context.world_size,
        "distributed_fsdp_config": fsdp_config,
    }


class NoopHeartbeatWriter:
    path = None

    def stage_start(self, stage: str, **fields: Any):
        return None

    def stage_complete(self, stage: str, **fields: Any):
        return None

    def heartbeat(self, stage: str, **fields: Any):
        return None


class NoopMonitoringSession:
    def __init__(self, distributed_context=None):
        self.distributed_context = distributed_context
        self.enabled = False

    def log_rows(self, rows) -> None:
        return None

    def close(self) -> None:
        return None


class WandbMonitoringSession(NoopMonitoringSession):
    def __init__(self, config: dict[str, Any], distributed_context=None):
        super().__init__(distributed_context=distributed_context)
        self._config = config
        self._wandb = None
        self._defined_series: set[str] = set()
        self._step_metric_defined = False
        self._logged_rows: list[dict[str, Any]] = []
        self.run = None

        if not config.get("monitoring", {}).get("enabled", False):
            return
        if not should_write_shared_artifact(distributed_context):
            return
        if config.get("monitoring", {}).get("backend") != "wandb":
            return

        try:
            import wandb
        except Exception:
            return

        init_kwargs: dict[str, Any] = {
            "id": str(config["run"]["run_id"]),
            "resume": "allow",
            "reinit": True,
            "dir": str(Path(config["run"]["output_dir"])),
        }
        monitoring = config.get("monitoring", {})
        project = monitoring.get("project") or config["run"].get("phase_id")
        if project is None:
            project = config["run"].get("output_group")
        entity = monitoring.get("entity")
        group = monitoring.get("group") or config["run"].get("output_group")
        job_type = monitoring.get("job_type")
        name = monitoring.get("name") or config["run"]["run_id"]
        tags = monitoring.get("tags") or []
        notes = monitoring.get("notes")
        mode = monitoring.get("mode")

        if project:
            init_kwargs["project"] = str(project)
        if entity:
            init_kwargs["entity"] = str(entity)
        if group:
            init_kwargs["group"] = str(group)
        if job_type:
            init_kwargs["job_type"] = str(job_type)
        if name:
            init_kwargs["name"] = str(name)
        if tags:
            init_kwargs["tags"] = list(tags)
        if notes:
            init_kwargs["notes"] = str(notes)
        if mode:
            init_kwargs["mode"] = str(mode)

        try:
            self.run = wandb.init(**init_kwargs)
        except Exception:
            return

        self._wandb = wandb
        self.enabled = True
        try:
            self._configure_run_metadata(config)
            self._define_expected_series(config)
        except Exception:
            self.enabled = False

    def _configure_run_metadata(self, config: dict[str, Any]) -> None:
        if self.run is None:
            return

        run = config["run"]
        training = config["training"]
        monitoring = config.get("monitoring", {})
        metadata = {
            "run_id": run["run_id"],
            "model_family": run["model_family"],
            "model_variant": config["model"]["variant"],
            "model_shape_label": run.get("model_shape_label"),
            "output_group": run.get("output_group"),
            "monitoring_project": monitoring.get("project"),
            "monitoring_entity": monitoring.get("entity"),
            "monitoring_group": monitoring.get("group"),
            "monitoring_job_type": monitoring.get("job_type"),
            "monitoring_name": monitoring.get("name"),
            "monitoring_tags": list(monitoring.get("tags", [])),
            "monitoring_notes": monitoring.get("notes"),
            "monitoring_mode": monitoring.get("mode"),
            "granularities": list(config["model"]["granularities"]),
            "granularity_sampling": training.get("granularity_sampling", "all"),
            "continuation_enabled": bool(run.get("continuation", {}).get("enabled", False)),
            "continuation_status": run.get("continuation", {}).get("status", "fresh"),
            "warmup_enabled": bool(
                training.get("pre_nested_warmup", {}).get("enabled", False)
            ),
            "warmup_duration": training.get("pre_nested_warmup", {}).get("duration", 0),
            "warmup_unit": training.get("pre_nested_warmup", {}).get("unit", "epochs"),
            "monitoring_enabled": bool(monitoring.get("enabled", False)),
            "monitoring_backend": monitoring.get("backend", "wandb"),
            "log_loss_by_granularity": monitoring.get("log_loss_by_granularity", True),
            "log_validation_loss": monitoring.get("log_validation_loss", True),
            "log_stage_events": monitoring.get("log_stage_events", True),
        }
        self.run.config.update(metadata, allow_val_change=True)
        self.run.summary.update(
            {
                "monitoring_enabled": metadata["monitoring_enabled"],
                "monitoring_backend": metadata["monitoring_backend"],
            }
        )

    def _define_expected_series(self, config: dict[str, Any]) -> None:
        if self._wandb is None:
            return

        monitoring = config.get("monitoring", {})
        if not monitoring.get("enabled", False):
            return

        metric_split_flags = {
            "train": bool(monitoring.get("log_loss_by_granularity", True)),
            "validation": bool(monitoring.get("log_validation_loss", True)),
        }
        for split, enabled in metric_split_flags.items():
            if not enabled:
                continue
            for granularity in config["model"]["granularities"]:
                series_name = f"{split}/loss/{granularity}"
                if series_name in self._defined_series:
                    continue
                if not self._step_metric_defined:
                    self._wandb.define_metric("step")
                    self._step_metric_defined = True
                self._wandb.define_metric(series_name, step_metric="step")
                self._defined_series.add(series_name)

    def log_rows(self, rows) -> None:
        if not self.enabled or self._wandb is None:
            return

        try:
            grouped_rows = group_loss_rows_by_series(rows)
            for series_name, series_rows in grouped_rows.items():
                if series_name not in self._defined_series:
                    if not self._step_metric_defined:
                        self._wandb.define_metric("step")
                        self._step_metric_defined = True
                    self._wandb.define_metric(series_name, step_metric="step")
                    self._defined_series.add(series_name)
                for row in series_rows:
                    step = int(row["step"])
                    value = row.get("loss")
                    if value is None:
                        continue
                    self._wandb.log({series_name: value}, step=step)
                    self._logged_rows.append(dict(row))
        except Exception:
            self.enabled = False

    def close(self) -> None:
        if self._wandb is None or self.run is None:
            return

        try:
            if self.enabled:
                self.run.summary.update(
                    {
                        "monitoring_series_metadata": build_monitoring_summary_fields(
                            self._config,
                            self._logged_rows,
                        )["monitoring_series_metadata"],
                    }
                )
            self._wandb.finish()
        except Exception:
            self.enabled = False


def create_monitoring_session(
    config: dict[str, Any],
    distributed_context=None,
):
    return WandbMonitoringSession(config, distributed_context=distributed_context)


def build_heartbeat_writer(config: dict[str, Any], distributed_context):
    training = config["training"]
    heartbeat_enabled = training.get("heartbeat_enabled", True)
    if not heartbeat_enabled or not should_write_shared_artifact(distributed_context):
        return NoopHeartbeatWriter()

    run = config["run"]
    return HeartbeatWriter(
        output_dir=run["output_dir"],
        run_id=run["run_id"],
        rank=distributed_context.rank,
        world_size=distributed_context.world_size,
    )


@contextmanager
def heartbeat_stage(heartbeat_writer, stage: str, **fields: Any):
    heartbeat_writer.stage_start(stage, **fields)
    try:
        yield
    finally:
        heartbeat_writer.stage_complete(stage, **fields)


def heartbeat_training_fields(
    config: dict[str, Any],
    step: int | None = None,
    tokens_seen: int | None = None,
    content_tokens_seen: int | None = None,
    latest_loss: float | None = None,
    tokens_per_second: float | None = None,
    peak_gpu_memory_bytes: int | None = None,
    eta_seconds: float | None = None,
) -> dict[str, Any]:
    training = config["training"]
    return {
        "step": step,
        "derived_max_steps": training.get("derived_max_steps"),
        "tokens_seen": tokens_seen,
        "content_tokens_seen": content_tokens_seen,
        "token_budget": training.get("token_budget"),
        "latest_loss": latest_loss,
        "tokens_per_second": tokens_per_second,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "eta_seconds": eta_seconds,
    }


def build_heartbeat_cadence(config: dict[str, Any]) -> HeartbeatCadence:
    training = config["training"]
    return HeartbeatCadence(
        step_interval=training.get("heartbeat_step_interval", 10),
        time_interval_seconds=training.get("heartbeat_time_interval_seconds", 60.0),
    )


def maybe_emit_training_heartbeat(
    heartbeat_writer,
    heartbeat_cadence: HeartbeatCadence,
    config: dict[str, Any],
    step: int,
    tokens_seen: int,
    content_tokens_seen: int,
    latest_loss: float,
    tokens_per_second: float | None,
    peak_gpu_memory_bytes: int,
    stage_name: str = "training",
) -> None:
    now = time.time()
    if not heartbeat_cadence.should_emit(step=step, now=now):
        return

    heartbeat_writer.heartbeat(
        stage_name,
        **heartbeat_training_fields(
            config,
            step=step,
            tokens_seen=tokens_seen,
            content_tokens_seen=content_tokens_seen,
            latest_loss=latest_loss,
            tokens_per_second=tokens_per_second,
            peak_gpu_memory_bytes=peak_gpu_memory_bytes,
            eta_seconds=estimate_eta_seconds(
                config,
                tokens_seen=tokens_seen,
                tokens_per_second=tokens_per_second,
            ),
        ),
    )
    heartbeat_cadence.mark_emitted(step=step, now=now)


def emit_run_start_continuation_state(
    heartbeat_writer,
    run_state: Mapping[str, Any],
) -> None:
    status = str(run_state.get("status", "fresh"))
    latest_checkpoint_path = run_state.get("latest_checkpoint_path")
    last_completed_step = int(run_state.get("last_completed_step", 0))
    resume_count = int(run_state.get("resume_count", 0))
    if status == "resumed":
        message = (
            f"Resuming run from {latest_checkpoint_path} "
            f"at step {last_completed_step} (resume_count={resume_count})"
        )
    else:
        message = "Starting fresh run"

    heartbeat_writer.emit(
        "run_state",
        "continuation",
        message=message,
        continuation_status=status,
        latest_checkpoint_path=latest_checkpoint_path,
        last_completed_step=last_completed_step,
        resume_count=resume_count,
    )


def estimate_eta_seconds(
    config: dict[str, Any],
    tokens_seen: int,
    tokens_per_second: float | None,
) -> float | None:
    if tokens_per_second is None or tokens_per_second <= 0:
        return None
    remaining_tokens = max(config["training"]["token_budget"] - tokens_seen, 0)
    return remaining_tokens / tokens_per_second


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
    if not force and not should_save_latest_checkpoint(config, step, reason):
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

    with heartbeat_stage(
        heartbeat_writer,
        "checkpointing",
        checkpoint_status="latest",
        checkpoint_reason=reason,
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
        )

    run_state["latest_checkpoint_path"] = str(latest_checkpoint_path)
    run_state["latest_checkpoint_step"] = step


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
        )

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
    if not config.get("evaluation", {}).get("validation", False):
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
        )

    checkpoint_state.update(checkpoint_fields)


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
) -> None:
    if not should_write_shared_artifact(distributed_context):
        return

    model_state_dict, optimizer_state_dict = checkpoint_state_dicts(
        model,
        optimizer,
        distributed_context,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "run_id": config["run"]["run_id"],
            "checkpoint_status": checkpoint_fields["checkpoint_status"],
            "checkpoint_metric": checkpoint_fields["checkpoint_metric"],
            "checkpoint_metric_value": checkpoint_fields[
                "checkpoint_metric_value"
            ],
            "checkpoint_selection_step": checkpoint_fields[
                "checkpoint_selection_step"
            ],
            "step": run_state.get("step", run_state.get("last_completed_step", 0)),
            "epoch": run_state.get("epoch", 0),
            "batch_index": run_state.get("batch_index", 0),
            "tokens_seen": run_state.get("tokens_seen", 0),
            "content_tokens_seen": run_state.get("content_tokens_seen", 0),
            "resume_count": run_state.get("resume_count", 0),
            "warmup_completed": run_state.get("warmup_completed", False),
            "warmup_completion_step": run_state.get("warmup_completion_step"),
            "warmup_transition_reason": run_state.get("warmup_transition_reason"),
            "latest_checkpoint_path": run_state.get("latest_checkpoint_path"),
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        },
        output_path,
    )


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
    return {
        "status": "fresh",
        "latest_checkpoint_path": None,
        "latest_checkpoint_step": 0,
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
        "output_dir": str(output_dir),
    }


def load_run_continuation_state(
    config: dict[str, Any],
    model,
    optimizer,
    scheduler,
    distributed_context=None,
) -> dict[str, Any]:
    checkpoint_path = Path(config["run"]["output_dir"]) / "checkpoints" / "latest.pt"
    return load_checkpoint_state(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        fallback_tokens_per_step=int(config["training"]["expected_tokens_per_step"]),
        distributed_context=distributed_context,
        output_dir=config["run"]["output_dir"],
        run_id=config["run"]["run_id"],
    )


def load_checkpoint_state(
    checkpoint_path: str | Path,
    model,
    optimizer,
    scheduler,
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
        }
        if output_dir is not None:
            state["output_dir"] = str(output_dir)
        if run_id is not None:
            state["run_id"] = str(run_id)
        return state

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
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
    }
    if output_dir is not None:
        state["output_dir"] = str(output_dir)
    if run_id is not None:
        state["run_id"] = str(run_id)
    return state


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


def update_pre_nested_warmup_state(
    config: dict[str, Any],
    state: Mapping[str, Any],
) -> None:
    training = config.setdefault("training", {})
    warmup = training.setdefault("pre_nested_warmup", {})
    if not isinstance(warmup, dict):
        raise ConfigError("training.pre_nested_warmup must be a mapping when provided")

    for key in [
        "enabled",
        "active",
        "duration",
        "unit",
        "completed",
        "completion_step",
        "transition_reason",
    ]:
        if key in state:
            warmup[key] = state[key]


def build_pre_nested_warmup_state(
    config: Mapping[str, Any],
    *,
    completed: bool,
    completion_step: int | None,
    transition_reason: str | None,
) -> dict[str, Any]:
    training = config.get("training", {})
    warmup = training.get("pre_nested_warmup", {})
    if not isinstance(warmup, Mapping):
        warmup = {}

    return {
        "enabled": bool(warmup.get("enabled", False)),
        "active": bool(warmup.get("active", False)),
        "duration": int(warmup.get("duration", 0)),
        "unit": str(warmup.get("unit", "epochs")),
        "completed": completed,
        "completion_step": completion_step,
        "transition_reason": transition_reason,
    }


def resolve_pre_nested_warmup_target_steps(
    config: Mapping[str, Any],
    train_dataloader,
) -> int:
    warmup = config.get("training", {}).get("pre_nested_warmup", {})
    if not isinstance(warmup, Mapping):
        warmup = {}

    duration = int(warmup.get("duration", 0))
    unit = str(warmup.get("unit", "epochs"))
    if unit == "steps":
        return duration
    if unit == "epochs":
        return duration * max(1, len(train_dataloader))
    raise ConfigError(f"Unsupported pre_nested_warmup unit: {unit}")


def should_run_pre_nested_warmup(
    config: Mapping[str, Any],
    run_state: Mapping[str, Any],
) -> bool:
    warmup = config.get("training", {}).get("pre_nested_warmup", {})
    if not isinstance(warmup, Mapping):
        warmup = {}
    if not bool(warmup.get("enabled", False)):
        return False
    if config.get("run", {}).get("model_family") != "nested":
        return False
    return not bool(run_state.get("warmup_completed", False))


def run_pre_nested_warmup_phase(
    config: dict[str, Any],
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    scheduler,
    device: torch.device,
    heartbeat_writer=None,
    distributed_context=None,
    checkpoint_state: dict[str, Any] | None = None,
    run_state: dict[str, Any] | None = None,
    monitoring_session=None,
    stage_name: str = "training",
) -> list[dict[str, Any]]:
    run_state = run_state if run_state is not None else build_initial_continuation_state(config)
    warmup = config["training"].get("pre_nested_warmup", {})
    if not isinstance(warmup, Mapping):
        warmup = {}

    warmup_state = build_pre_nested_warmup_state(
        config,
        completed=bool(run_state.get("warmup_completed", False)),
        completion_step=(
            int(run_state["warmup_completion_step"])
            if run_state.get("warmup_completion_step") is not None
            else None
        ),
        transition_reason=run_state.get("warmup_transition_reason"),
    )

    if not should_run_pre_nested_warmup(config, run_state):
        update_pre_nested_warmup_state(config, warmup_state)
        return []

    warmup_target_steps = resolve_pre_nested_warmup_target_steps(config, train_dataloader)
    current_step = int(run_state.get("last_completed_step", 0))
    if current_step >= warmup_target_steps:
        warmup_state.update(
            {
                "completed": True,
                "completion_step": current_step,
                "transition_reason": "warmup_duration_reached",
            }
        )
        run_state.update(
            {
                "warmup_completed": True,
                "warmup_completion_step": current_step,
                "warmup_transition_reason": "warmup_duration_reached",
            }
        )
        update_pre_nested_warmup_state(config, warmup_state)
        if checkpoint_state is not None:
            checkpoint_state.update(
                {
                    "warmup_completed": True,
                    "warmup_completion_step": current_step,
                    "warmup_transition_reason": "warmup_duration_reached",
                }
            )
        if continuation_latest_checkpoint_policy(config)["enabled"]:
            maybe_write_latest_checkpoint(
                config,
                model,
                optimizer,
                scheduler,
                heartbeat_writer or NoopHeartbeatWriter(),
                run_state,
                reason="warmup_completion",
                step=current_step,
                distributed_context=distributed_context,
                force=True,
            )
        return []

    warmup_config = copy.deepcopy(config)
    warmup_config["training"]["max_steps"] = min(
        int(config["training"]["max_steps"]),
        warmup_target_steps,
    )
    warmup_config["model"]["granularities"] = [
        config["model"]["granularities"][-1]
    ]
    warmup_config["training"]["granularity_sampling"] = "all"

    warmup_metrics_rows = train_for_steps(
        warmup_config,
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
        stage_name="warmup",
    )

    current_step = int(run_state.get("last_completed_step", 0))
    warmup_completed = current_step >= warmup_target_steps
    transition_reason = (
        "warmup_duration_reached"
        if warmup_completed
        else "budget_exhausted_before_nested_phase"
    )
    completion_step = current_step if warmup_completed else None
    warmup_state.update(
        {
            "completed": warmup_completed,
            "completion_step": completion_step,
            "transition_reason": transition_reason,
        }
    )
    run_state.update(
        {
            "warmup_completed": warmup_completed,
            "warmup_completion_step": completion_step,
            "warmup_transition_reason": transition_reason,
        }
    )
    update_pre_nested_warmup_state(config, warmup_state)

    if checkpoint_state is not None:
        checkpoint_state.update(
            {
                "warmup_completed": warmup_completed,
                "warmup_completion_step": completion_step,
                "warmup_transition_reason": transition_reason,
            }
        )

    if continuation_latest_checkpoint_policy(config)["enabled"]:
        maybe_write_latest_checkpoint(
            config,
            model,
            optimizer,
            scheduler,
            heartbeat_writer or NoopHeartbeatWriter(),
            run_state,
            reason="warmup_completion",
            step=current_step,
            distributed_context=distributed_context,
            force=True,
        )

    return warmup_metrics_rows


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


def best_validation_metric_value(
    validation_results: list[dict[str, Any]],
) -> tuple[str | None, float | None]:
    loss_values = [
        float(result["loss"])
        for result in validation_results
        if result.get("loss") is not None
    ]
    if loss_values:
        return "validation_loss", min(loss_values)

    perplexity_values = [
        float(result["perplexity"])
        for result in validation_results
        if result.get("perplexity") is not None
    ]
    if perplexity_values:
        return "validation_perplexity", min(perplexity_values)

    return None, None


def build_dataloaders(
    config: dict[str, Any],
    tokenized_dataset,
    device: torch.device,
    distributed_context=None,
):
    training = config["training"]
    batch_size = training["batch_size_per_process"]
    eval_batches = training.get("eval_batches", 1)
    eval_example_count = max(1, eval_batches * batch_size)

    train_dataset, eval_dataset = split_train_eval_dataset(
        tokenized_dataset,
        eval_example_count,
    )
    if len(train_dataset) == 0:
        train_dataset = eval_dataset

    pin_memory = device.type == "cuda"
    train_sampler = build_distributed_sampler(
        train_dataset,
        distributed_context,
        shuffle=True,
        seed=config["run"].get("seed"),
    )
    eval_sampler = build_distributed_sampler(
        eval_dataset,
        distributed_context,
        shuffle=False,
        seed=config["run"].get("seed"),
    )
    train_dataloader = build_language_model_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=training.get("dataloader_num_workers", 0),
        pin_memory=pin_memory,
    )
    eval_dataloader = build_language_model_dataloader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=eval_sampler,
        num_workers=training.get("dataloader_num_workers", 0),
        pin_memory=pin_memory,
    )
    return train_dataloader, eval_dataloader


def build_distributed_sampler(
    dataset,
    distributed_context,
    shuffle: bool,
    seed: int | None,
):
    if distributed_context is None or not distributed_context.enabled:
        return None

    return DistributedSampler(
        dataset,
        num_replicas=distributed_context.world_size,
        rank=distributed_context.rank,
        shuffle=shuffle,
        seed=0 if seed is None else int(seed),
    )


def build_optimizer_and_scheduler(model, training: Mapping[str, Any]):
    """Build the training optimizer and scheduler from resolved config fields."""
    optimizer_name = str(training.get("optimizer_name", "adamw"))
    optimizer_kwargs = resolve_optimizer_kwargs(
        optimizer_name,
        training.get("optimizer_kwargs", {}),
    )
    scheduler_name = str(training.get("scheduler_name", "cosine"))
    scheduler_kwargs = dict(training.get("scheduler_kwargs", {}))
    resolved_warmup_steps = int(
        training.get(
            "resolved_warmup_steps",
            training.get("scheduler", {}).get("resolved_warmup_steps", 0),
        )
    )
    learning_rate = training.get("resolved_learning_rate", training.get("learning_rate"))
    if learning_rate is None:
        raise ConfigError(
            "training must include learning_rate or resolved_learning_rate"
        )

    if optimizer_name == "adamw":
        if "betas" in optimizer_kwargs and isinstance(optimizer_kwargs["betas"], list):
            optimizer_kwargs["betas"] = tuple(optimizer_kwargs["betas"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            **optimizer_kwargs,
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            **optimizer_kwargs,
        )
    else:
        raise ConfigError(f"Unsupported optimizer name: {optimizer_name}")

    scheduler = get_scheduler(
        scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=resolved_warmup_steps,
        num_training_steps=int(training["max_steps"]),
        **scheduler_kwargs,
    )
    return optimizer, scheduler


def train_for_steps(
    config: dict[str, Any],
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    scheduler,
    device: torch.device,
    heartbeat_writer=None,
    distributed_context=None,
    checkpoint_state: dict[str, Any] | None = None,
    run_state: dict[str, Any] | None = None,
    monitoring_session=None,
    stage_name: str = "training",
) -> list[dict[str, Any]]:
    training = config["training"]
    granularities = config["model"]["granularities"]
    token_budget = training["token_budget"]
    max_steps = training["max_steps"]
    eval_interval = training.get("eval_interval", 0)

    metrics_rows = []
    start_time = time.time()
    run_state = run_state if run_state is not None else build_initial_continuation_state(config)
    step = int(run_state.get("last_completed_step", 0))
    epoch = int(run_state.get("epoch", 0))
    resume_batch_index = int(run_state.get("batch_index", 0))
    tokens_seen = int(
        run_state.get("tokens_seen", budget_tokens_seen_for_step(config, step))
    )
    content_tokens_seen = int(run_state.get("content_tokens_seen", 0))
    heartbeat_writer = heartbeat_writer or NoopHeartbeatWriter()
    heartbeat_cadence = build_heartbeat_cadence(config)
    checkpoint_state = checkpoint_state if checkpoint_state is not None else {}
    latest_checkpoint_path = Path(
        run_state.get("latest_checkpoint_path")
        or Path(config["run"]["output_dir"]) / "checkpoints" / "latest.pt"
    )
    resume_epoch = epoch
    resume_batch_index = max(0, resume_batch_index)
    if len(train_dataloader) > 0:
        while resume_batch_index >= len(train_dataloader):
            resume_batch_index -= len(train_dataloader)
            resume_epoch += 1
    else:
        resume_batch_index = 0
    run_state["epoch"] = resume_epoch
    run_state["batch_index"] = resume_batch_index
    run_state["content_tokens_seen"] = content_tokens_seen
    run_state.setdefault("latest_checkpoint_step", int(run_state.get("last_completed_step", 0)))
    run_state.setdefault("status", "fresh")
    if continuation_latest_checkpoint_policy(config)["enabled"] and not run_state.get("latest_checkpoint_path"):
        run_state["latest_checkpoint_path"] = str(latest_checkpoint_path)

    model.train()
    with heartbeat_stage(heartbeat_writer, stage_name):
        while step < max_steps and tokens_seen < token_budget:
            set_dataloader_epoch(train_dataloader, epoch)
            made_progress = False
            current_epoch = epoch
            epoch += 1
            for batch_index_in_epoch, batch in enumerate(train_dataloader):
                if current_epoch == resume_epoch and batch_index_in_epoch < resume_batch_index:
                    continue
                if step >= max_steps or tokens_seen >= token_budget:
                    break

                made_progress = True
                step += 1
                batch = move_batch_to_device(batch, device)
                content_tokens_seen += global_content_tokens_for_batch(
                    batch,
                    device=device,
                    distributed_context=distributed_context,
                )
                tokens_seen = budget_tokens_seen_for_step(config, step)

                optimizer.zero_grad(set_to_none=True)

                selected_granularities = select_training_granularities(
                    config,
                    granularities,
                    device,
                )
                granularity_losses = []
                for granularity in selected_granularities:
                    configure_model_granularity(model, granularity)
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch["labels"],
                    )
                    granularity_losses.append((granularity, outputs.loss))

                combined_loss = torch.stack(
                    [loss for _, loss in granularity_losses]
                ).mean()
                combined_loss.backward()

                gradient_clip_norm = training.get("gradient_clip_norm")
                if gradient_clip_norm is not None:
                    clip_grad_norm_(model.parameters(), float(gradient_clip_norm))

                optimizer.step()
                scheduler.step()

                elapsed = time.time() - start_time
                peak_memory_bytes = current_peak_memory_bytes(device)
                latest_loss = float(combined_loss.detach().cpu().item())
                tokens_per_second = tokens_seen / elapsed if elapsed > 0 else None
                step_metric_rows = []
                run_state.update(
                    {
                        "status": "resumed" if int(run_state.get("resume_count", 0)) > 0 else "fresh",
                        "last_completed_step": step,
                        "step": step,
                        "epoch": current_epoch,
                        "batch_index": batch_index_in_epoch + 1,
                        "tokens_seen": tokens_seen,
                        "content_tokens_seen": content_tokens_seen,
                    }
                )
                maybe_write_latest_checkpoint(
                    config,
                    model,
                    optimizer,
                    scheduler,
                    heartbeat_writer,
                    run_state,
                    reason="step",
                    step=step,
                    distributed_context=distributed_context,
                )
                maybe_emit_training_heartbeat(
                    heartbeat_writer,
                    heartbeat_cadence,
                    config,
                    step=step,
                    tokens_seen=tokens_seen,
                    content_tokens_seen=content_tokens_seen,
                    latest_loss=latest_loss,
                    tokens_per_second=tokens_per_second,
                    peak_gpu_memory_bytes=peak_memory_bytes,
                    stage_name=stage_name,
                )

                for granularity, loss in granularity_losses:
                    step_metric_rows.append(
                        build_training_metric_row(
                            config,
                            step=step,
                            granularity=granularity,
                            loss=float(loss.detach().cpu().item()),
                            tokens_seen=tokens_seen,
                            content_tokens_seen=content_tokens_seen,
                            wall_clock_seconds=elapsed,
                            peak_memory_bytes=peak_memory_bytes,
                        )
                    )
                metrics_rows.extend(step_metric_rows)
                if monitoring_session is not None:
                    monitoring_session.log_rows(step_metric_rows)

                if eval_interval > 0 and step % eval_interval == 0:
                    with heartbeat_stage(
                        heartbeat_writer,
                        "validation",
                        **heartbeat_training_fields(
                            config,
                            step=step,
                            tokens_seen=tokens_seen,
                            content_tokens_seen=content_tokens_seen,
                        ),
                    ):
                        validation_results = evaluate_validation_per_granularity(
                            model,
                            eval_dataloader,
                            granularities=granularities,
                            device=device,
                            distributed=(
                                distributed_context is not None
                                and distributed_context.enabled
                            ),
                        )
                    maybe_write_best_eval_checkpoint(
                        config,
                        model,
                        validation_results,
                        step,
                        heartbeat_writer,
                        checkpoint_state,
                        run_state,
                        distributed_context=distributed_context,
                    )
                    maybe_write_latest_checkpoint(
                        config,
                        model,
                        optimizer,
                        scheduler,
                        heartbeat_writer,
                        run_state,
                        reason="validation",
                        step=step,
                        distributed_context=distributed_context,
                    )
                    validation_metric_rows = validation_results_to_metric_rows(
                        validation_results,
                        config,
                        step=step,
                        wall_clock_seconds=elapsed,
                        tokens_per_second=tokens_per_second,
                        peak_memory_bytes=peak_memory_bytes,
                        tokens_seen=tokens_seen,
                        content_tokens_seen=content_tokens_seen,
                    )
                    metrics_rows.extend(validation_metric_rows)
                    if monitoring_session is not None:
                        monitoring_session.log_rows(validation_metric_rows)

                if step >= max_steps or tokens_seen >= token_budget:
                    break
            if not made_progress:
                break

    append_final_validation_if_needed(
        metrics_rows,
        config,
        model,
        eval_dataloader,
        granularities=granularities,
        device=device,
        step=step,
        tokens_seen=tokens_seen,
        content_tokens_seen=content_tokens_seen,
        start_time=start_time,
        heartbeat_writer=heartbeat_writer,
        distributed_context=distributed_context,
        checkpoint_state=checkpoint_state,
        run_state=run_state,
        monitoring_session=monitoring_session,
    )
    maybe_write_latest_checkpoint(
        config,
        model,
        optimizer,
        scheduler,
        heartbeat_writer,
        run_state,
        reason="completion",
        step=step,
        distributed_context=distributed_context,
    )

    return metrics_rows


def select_training_granularities(
    config: dict[str, Any],
    granularities: list[str],
    device: torch.device,
) -> list[str]:
    sampling_mode = config["training"].get("granularity_sampling", "all")
    if sampling_mode == "all":
        return list(granularities)
    if sampling_mode == "random":
        selected_index = select_random_granularity_index(
            granularity_count=len(granularities),
            device=device,
        )
        return [granularities[selected_index]]
    raise ValueError(f"Unknown granularity sampling mode: {sampling_mode}")


def select_random_granularity_index(
    granularity_count: int,
    device: torch.device,
) -> int:
    if granularity_count <= 0:
        raise ValueError("granularity_count must be positive")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        selected_index = torch.empty((), dtype=torch.long, device=device)
        if torch.distributed.get_rank() == 0:
            selected_index.fill_(random.randrange(granularity_count))
        torch.distributed.broadcast(selected_index, src=0)
        return int(selected_index.item())

    return random.randrange(granularity_count)


def set_dataloader_epoch(dataloader, epoch: int) -> None:
    sampler = getattr(dataloader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def summarize_training_outcome(
    config: dict[str, Any],
    metrics_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    training_rows = [row for row in metrics_rows if row["split"] == "train"]
    if not training_rows:
        return {
            "steps_completed": 0,
            "tokens_seen": 0,
            "content_tokens_seen": 0,
            "stop_reason": "not_started",
        }

    steps_completed = max(int(row["step"]) for row in training_rows)
    tokens_seen = max(int(row["tokens_seen"]) for row in training_rows)
    content_tokens_seen = max(
        int(row.get("content_tokens_seen", row["tokens_seen"]))
        for row in training_rows
    )
    return {
        "steps_completed": steps_completed,
        "tokens_seen": tokens_seen,
        "content_tokens_seen": content_tokens_seen,
        "stop_reason": stop_reason_for_training(
            config,
            tokens_seen=tokens_seen,
            steps_completed=steps_completed,
        ),
    }


def stop_reason_for_training(
    config: dict[str, Any],
    tokens_seen: int,
    steps_completed: int,
) -> str:
    training = config["training"]
    if steps_completed == 0:
        return "not_started"
    if tokens_seen >= training["token_budget"]:
        return "token_budget_reached"
    return "max_steps_reached_before_token_budget"


def append_final_validation_if_needed(
    metrics_rows: list[dict[str, Any]],
    config: dict[str, Any],
    model,
    eval_dataloader,
    granularities: list[str],
    device: torch.device,
    step: int,
    tokens_seen: int,
    content_tokens_seen: int,
    start_time: float,
    heartbeat_writer=None,
    distributed_context=None,
    checkpoint_state: dict[str, Any] | None = None,
    run_state: dict[str, Any] | None = None,
    monitoring_session=None,
) -> None:
    if not config.get("evaluation", {}).get("validation", False):
        return
    has_final_validation = any(
        row["split"] == "validation" and row["step"] == step
        for row in metrics_rows
    )
    if has_final_validation:
        return

    elapsed = time.time() - start_time
    heartbeat_writer = heartbeat_writer or NoopHeartbeatWriter()
    with heartbeat_stage(
        heartbeat_writer,
        "validation",
        **heartbeat_training_fields(
            config,
            step=step,
            tokens_seen=tokens_seen,
            content_tokens_seen=content_tokens_seen,
        ),
    ):
        validation_results = evaluate_validation_per_granularity(
            model,
            eval_dataloader,
            granularities=granularities,
            device=device,
            distributed=(
                distributed_context is not None and distributed_context.enabled
            ),
        )
    maybe_write_best_eval_checkpoint(
        config,
        model,
        validation_results,
        step,
        heartbeat_writer,
        checkpoint_state if checkpoint_state is not None else {},
        run_state if run_state is not None else build_initial_continuation_state(config),
        distributed_context=distributed_context,
    )
    validation_metric_rows = validation_results_to_metric_rows(
        validation_results,
        config,
        step=step,
        wall_clock_seconds=elapsed,
        tokens_per_second=tokens_seen / elapsed if elapsed > 0 else None,
        peak_memory_bytes=current_peak_memory_bytes(device),
        tokens_seen=tokens_seen,
        content_tokens_seen=content_tokens_seen,
    )
    metrics_rows.extend(validation_metric_rows)
    if monitoring_session is not None:
        monitoring_session.log_rows(validation_metric_rows)


def write_extraction_metadata_if_nested(
    config: dict[str, Any],
    model,
    output_dir: Path,
    distributed_context=None,
) -> Path | None:
    if config["run"]["model_family"] != "nested":
        return None

    metadata = build_extraction_metadata(config, model)
    return write_json_artifact(
        output_dir / "extraction_metadata.json",
        metadata,
        distributed_context=distributed_context,
    )


def build_extraction_metadata(config: dict[str, Any], model) -> dict[str, Any]:
    run = config["run"]
    model_config = config["model"]
    configured_granularities = model_config["granularities"]
    prefix_metadata = prefix_metadata_by_granularity(model, model_config)

    return {
        "run_id": run["run_id"],
        "phase_id": run["phase_id"],
        "model_family": run["model_family"],
        "model_size_label": _model_shape_label(run),
        "model_shape_label": _model_shape_label(run),
        "granularities": [
            build_granularity_extraction_metadata(
                granularity,
                configured_granularities,
                prefix_metadata[granularity],
            )
            for granularity in configured_granularities
        ],
    }


def prefix_metadata_by_granularity(
    model,
    model_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    target = model.module if hasattr(model, "module") else model
    metadata = getattr(target, "ffn_prefix_metadata", None)
    if metadata is None:
        metadata = model_config.get("ffn_prefix_metadata")
    if metadata is None:
        metadata = get_ffn_prefix_metadata(
            model_config["intermediate_size"],
            granularity_prefixes=model_config.get("granularity_prefixes"),
            granularities=model_config.get("granularities"),
        )

    return {entry["name"]: dict(entry) for entry in metadata}


def build_granularity_extraction_metadata(
    granularity: str,
    configured_granularities: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    granularity_index = configured_granularities.index(granularity)
    return {
        "granularity": granularity,
        "display_name": metadata["display_name"],
        "ffn_ratio": metadata["ffn_ratio"],
        "full_intermediate_fraction": metadata["full_intermediate_fraction"],
        "prefix_width": metadata["prefix_width"],
        "strict_prefix_of": configured_granularities[granularity_index + 1 :],
    }


def build_training_metric_row(
    config: dict[str, Any],
    step: int,
    granularity: str,
    loss: float,
    tokens_seen: int,
    content_tokens_seen: int,
    wall_clock_seconds: float,
    peak_memory_bytes: int,
) -> dict[str, Any]:
    run = config["run"]
    tokens_per_second = tokens_seen / wall_clock_seconds if wall_clock_seconds > 0 else None
    return {
        "run_id": run["run_id"],
        "step": step,
        "split": "train",
        "model_family": run["model_family"],
        "model_size_label": _model_shape_label(run),
        "model_shape_label": _model_shape_label(run),
        "sampling_mode": _sampling_mode(run, config["training"]),
        "granularity": granularity,
        "loss": loss,
        "perplexity": perplexity_from_loss(loss),
        "tokens_seen": tokens_seen,
        "content_tokens_seen": content_tokens_seen,
        "wall_clock_seconds": wall_clock_seconds,
        "tokens_per_second": tokens_per_second,
        "peak_memory_bytes": peak_memory_bytes,
    }


def select_training_granularity(granularities: list[str], step: int) -> str:
    return granularities[(step - 1) % len(granularities)]


def budget_tokens_seen_for_step(config: dict[str, Any], step: int) -> int:
    planned_tokens = step * int(config["training"]["expected_tokens_per_step"])
    return min(planned_tokens, int(config["training"]["token_budget"]))


def global_content_tokens_for_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    distributed_context=None,
) -> int:
    return sum_int(
        count_content_tokens(batch),
        device=device,
        context=distributed_context,
    )


def count_content_tokens(batch: dict[str, torch.Tensor]) -> int:
    if "attention_mask" in batch and batch["attention_mask"] is not None:
        return int(batch["attention_mask"].sum().item())
    return int((batch["labels"] != -100).sum().item())


def count_batch_tokens(batch: dict[str, torch.Tensor]) -> int:
    return count_content_tokens(batch)


def current_peak_memory_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return 0


def set_random_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_shape_label(run: dict[str, Any]) -> Any:
    return run.get("model_shape_label", run.get("model_size_label"))


def _sampling_mode(run: dict[str, Any], training: dict[str, Any]) -> Any:
    if run.get("sampling_mode") is not None:
        return run["sampling_mode"]
    if run.get("model_family") == "standalone":
        return "standalone"
    granularity_sampling = training.get("granularity_sampling")
    if granularity_sampling == "random":
        return "nested-random"
    if granularity_sampling == "all":
        return "nested-all"
    return granularity_sampling
