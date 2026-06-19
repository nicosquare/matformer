"""Monitoring-session helpers for training runs."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.training.distributed import should_write_shared_artifact
from src.utils.heartbeats import HeartbeatWriter
from src.utils.metrics import build_monitoring_summary_fields
from src.utils.monitoring import group_loss_rows_by_series

__all__ = [
    "NoopHeartbeatWriter",
    "NoopMonitoringSession",
    "WandbMonitoringSession",
    "build_heartbeat_writer",
    "create_monitoring_session",
    "emit_run_start_continuation_state",
    "heartbeat_stage",
]


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


@contextmanager
def heartbeat_stage(heartbeat_writer, stage: str, **fields: Any):
    heartbeat_writer.stage_start(stage, **fields)
    try:
        yield
    finally:
        heartbeat_writer.stage_complete(stage, **fields)


def create_monitoring_session(
    config: dict[str, Any],
    distributed_context=None,
):
    return WandbMonitoringSession(config, distributed_context=distributed_context)


def emit_run_start_continuation_state(
    heartbeat_writer,
    run_state,
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
