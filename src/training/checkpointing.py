"""Config-driven training flow for MatFormer reproduction runs."""

from __future__ import annotations

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*args, **kwargs):
        return None

load_dotenv()

import copy
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
from src.utils.config import (
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
        evaluation.get("validation", False)
        or evaluation.get("final_validation", False)
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
            "run_id": config["run"]["run_id"],
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
            "resolved_run_mode": run_state.get("resolved_run_mode"),
            "resolved_sampling_mode": run_state.get("resolved_sampling_mode"),
            "granularity_pattern_provenance": run_state.get(
                "granularity_pattern_provenance"
            ),
            "adaptive_sampler_state": run_state.get("adaptive_sampler_state"),
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
    if model.get("granularity_sampling_mode") != "adaptive_per_block":
        return None

    state = build_adaptive_sampler_state(
        strategy_name=str(model.get("adaptive_sampler_strategy", "thompson")),
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

    if model.get("granularity_sampling_mode") == "adaptive_per_block":
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
    if model.get("granularity_sampling_mode") != "adaptive_per_block":
        return

    expected_strategy = str(model.get("adaptive_sampler_strategy", "thompson"))
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
    if model.get("granularity_sampling_mode") != "adaptive_per_block":
        run_state.pop("adaptive_sampler_state", None)
        return None

    adaptive_state = coerce_adaptive_sampler_state(
        run_state.get("adaptive_sampler_state")
    )
    if adaptive_state is None:
        adaptive_state = build_adaptive_sampler_state(
            strategy_name=str(model.get("adaptive_sampler_strategy", "thompson")),
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
        }
        if output_dir is not None:
            state["output_dir"] = str(output_dir)
        if run_id is not None:
            state["run_id"] = str(run_id)
        if config is not None:
            _populate_adaptive_sampler_state_metadata(state, config)
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
        "resolved_run_mode": checkpoint.get("resolved_run_mode"),
        "resolved_sampling_mode": checkpoint.get("resolved_sampling_mode"),
        "granularity_pattern_provenance": checkpoint.get(
            "granularity_pattern_provenance"
        ),
        "adaptive_sampler_state": checkpoint.get("adaptive_sampler_state"),
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
        _validate_loaded_adaptive_sampler_state(state, config, checkpoint_path)
    return state
