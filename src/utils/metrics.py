"""Artifact writers for MatFormer reproduction experiments."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.utils.config import (
    resolve_sampling_mode_from_config_sections,
    write_resolved_config,
)
from src.utils.artifact_io import (
    append_jsonl_artifact,
    append_jsonl_artifacts,
    emit_artifact_event,
    remove_resolved_failure,
    resolved_artifact_io,
    retry_artifact_io,
)
from src.models.adaptive_sampler import build_adaptive_sampler_artifact_fields
from src.models.correction import summarize_correction_context_from_config
from src.models.granularity import (
    resolved_granularity_artifact_fields,
    summarize_granularity_pattern_from_config,
)
from src.utils.monitoring import (
    DEFAULT_MONITORING_BACKEND,
    build_monitoring_series_metadata,
)


METRICS_COLUMNS = [
    "run_id",
    "step",
    "split",
    "microstep",
    "model_family",
    "model_size_label",
    "model_shape_label",
    "sampling_mode",
    "resolved_run_mode",
    "resolved_sampling_mode",
    "granularity_sampling_mode",
    "correction_mode",
    "membership_correction",
    "granularity",
    "granularity_pattern_summary",
    "granularity_mode",
    "granularities",
    "granularity_prefixes",
    "granularity_prefix_widths",
    "correction_context",
    "sampler_strategy",
    "adaptive_sampler_strategy",
    "adaptive_sampler_exploration_scale",
    "adaptive_sampler_decay_rate",
    "adaptive_sampler_reward_penalty_weight",
    "sampler_state",
    "adaptive_sampler_state",
    "adaptive_sampler_previous_loss",
    "adaptive_sampler_previous_pattern",
    "adaptive_reward_summary",
    "adaptive_correction_penalty_summary",
    "controller_method_family",
    "controller_method_version",
    "controller_strategy",
    "controller_scope",
    "controller_action",
    "controller_sampled_probability",
    "controller_exposure_counts",
    "controller_phase",
    "controller_entropy",
    "controller_min_probability",
    "controller_max_probability",
    "controller_window_index",
    "controller_window_progress",
    "controller_boundary_step",
    "controller_latest_objective",
    "controller_latest_reward",
    "controller_latest_prediction_error",
    "controller_manifest_hash",
    "final_holdout_manifest_hash",
    "controller_metrics_path",
    "controller_summary_path",
    "controller_reset_enabled",
    "controller_reset_policy",
    "controller_episode_index",
    "controller_episode_offset_steps",
    "controller_selection_source",
    "reward",
    "correction_penalty",
    "loss",
    "perplexity",
    "tokens_seen",
    "content_tokens_seen",
    "optimizer_window_microsteps",
    "committed_tokens_this_step",
    "evaluation_examples",
    "evaluation_batches",
    "evaluation_target_tokens",
    "evaluation_skipped_batches",
    "validation_manifest_hash",
    "validation_loss_aggregation",
    "comparison_control_signature",
    "wall_clock_seconds",
    "tokens_per_second",
    "peak_memory_bytes",
    "output_root",
    "output_dir",
    "metrics_path",
    "scaling_results_path",
    "extraction_metadata_path",
]

TASK_RESULTS_COLUMNS = [
    "run_id",
    "suite_id",
    "task",
    "model_family",
    "model_size_label",
    "model_shape_label",
    "sampling_mode",
    "granularity",
    "metric_name",
    "metric_value",
]

SCALING_RESULTS_COLUMNS = [
    "comparison_id",
    "run_id",
    "model_family",
    "model_size_label",
    "model_shape_label",
    "sampling_mode",
    "model_family_slug",
    "model_size_slug",
    "token_budget_slug",
    "output_group",
    "completion_label",
    "granularity",
    "d_model",
    "num_layers",
    "num_attention_heads",
    "context_length",
    "vocab_size",
    "token_budget",
    "effective_world_size",
    "total_parameters",
    "embedding_parameters",
    "lm_head_parameters",
    "non_embedding_parameters",
    "ffn_parameters",
    "attention_parameters",
    "other_non_embedding_parameters",
    "lm_head_counting",
    "checkpoint_path",
    "loss",
    "perplexity",
    "final_validation_loss",
    "final_validation_perplexity",
    "best_validation_loss",
    "best_validation_perplexity",
    "best_validation_step",
    "best_validation_checkpoint",
    "trailing_validation_mean",
    "trailing_validation_sample_stddev",
    "trailing_validation_min",
    "trailing_validation_max",
    "trailing_validation_count",
    "final_minus_best_loss",
    "evaluation_target_tokens",
    "effective_width",
    "validation_manifest_hash",
    "comparison_control_signature",
    "average_downstream_accuracy",
]

CONSISTENCY_RESULTS_COLUMNS = [
    "comparison_id",
    "small_run_id",
    "large_run_id",
    "small_granularity",
    "large_granularity",
    "metric_name",
    "metric_value",
    "sample_count",
]

RUN_SUMMARY_FIELDS = [
    "run_id",
    "phase_id",
    "model_family",
    "model_variant",
    "correction_mode",
    "membership_correction",
    "continuation_state",
    "monitoring_enabled",
    "monitoring_backend",
    "monitoring_series_metadata",
    "warmup_policy",
    "warmup_completion_step",
    "warmup_completed",
    "latest_checkpoint_path",
    "model_size_label",
    "model_shape_label",
    "active_size_label",
    "family_size_slug",
    "family_resolution_rule",
    "sampling_mode",
    "resolved_run_mode",
    "resolved_sampling_mode",
    "requested_granularity_sampling_alias",
    "granularity_sampling_mode",
    "granularity_pattern_provenance",
    "granularity_pattern_summary",
    "granularity_mode",
    "granularities",
    "granularity_prefixes",
    "granularity_prefix_widths",
    "correction_context",
    "adaptive_sampler_strategy",
    "adaptive_sampler_exploration_scale",
    "adaptive_sampler_decay_rate",
    "adaptive_sampler_reward_penalty_weight",
    "adaptive_sampler_state",
    "adaptive_sampler_previous_loss",
    "adaptive_sampler_previous_pattern",
    "adaptive_reward_summary",
    "adaptive_correction_penalty_summary",
    "model_family_slug",
    "model_size_slug",
    "token_budget_slug",
    "output_group",
    "completion_label",
    "dataset_name",
    "dataset_split",
    "token_budget",
    "base_learning_rate",
    "learning_rate_scale_rule",
    "learning_rate_scale_factor",
    "resolved_learning_rate",
    "warmup_ratio",
    "warmup_steps",
    "resolved_warmup_steps",
    "gradient_accumulation_steps",
    "gradient_clip_norm",
    "scheduler_name",
    "scheduler_warmup_steps",
    "scheduler_resolved_warmup_steps",
    "scheduler_kwargs",
    "optimizer_name",
    "optimizer_kwargs",
    "requested_mixed_precision",
    "resolved_mixed_precision",
    "requested_activation_checkpointing",
    "resolved_activation_checkpointing",
    "final_validation",
    "final_validation_reason",
    "expected_tokens_per_step",
    "expected_tokens_per_microstep",
    "derived_max_steps",
    "effective_world_size",
    "validation_interval_tokens",
    "tokenizer_manifest_hash",
    "tokens_seen",
    "content_tokens_seen",
    "stop_reason",
    "seed",
    "validation_manifest_hash",
    "validation_loss_aggregation",
    "comparison_control_signature",
    "status",
    "output_root",
    "output_dir",
    "d_model",
    "num_layers",
    "num_attention_heads",
    "context_length",
    "vocab_size",
    "parameter_counts",
    "parameter_counts_by_granularity",
    "preset_selections",
    "preset_registry_paths",
    "checkpoint_status",
    "best_checkpoint_path",
    "final_checkpoint_path",
    "checkpoint_metric",
    "checkpoint_metric_value",
    "checkpoint_selection_step",
    "checkpoint_unavailable_reason",
    "artifact_retry_count",
    "artifact_last_errno",
    "last_durable_checkpoint_step",
    "deferred_metric_rows",
    "skipped_periodic_checkpoints",
    "checkpoint_staging_mode",
    "unresolved_artifact_failures",
    "metrics_path",
    "scaling_results_path",
    "scaling_results_unavailable_reason",
    "extraction_metadata_path",
    "notes",
]

PARAMETER_COUNT_FIELDS = [
    "total_parameters",
    "embedding_parameters",
    "lm_head_parameters",
    "non_embedding_parameters",
    "ffn_parameters",
    "attention_parameters",
    "other_non_embedding_parameters",
    "lm_head_counting",
]

BASELINE_MATCH_FIELDS = [
    "match_id",
    "nested_run_id",
    "standalone_run_id",
    "granularity",
    "non_embedding_parameters_nested",
    "non_embedding_parameters_standalone",
    "match_notes",
]


class ArtifactError(ValueError):
    """Raised when an artifact would miss required analysis fields."""


class StreamingMetricsAccumulator:
    """Checkpointable bounded summary of an arbitrarily large metrics stream."""

    def __init__(self, state: Mapping[str, Any] | None = None, *, trailing_count: int = 5):
        state = dict(state or {})
        self.trailing_count = max(1, int(state.get("trailing_count", trailing_count)))
        self.last_training_step = int(state.get("last_training_step", 0))
        self.tokens_seen = int(state.get("tokens_seen", 0))
        self.content_tokens_seen = int(state.get("content_tokens_seen", 0))
        self.training_row_count = int(state.get("training_row_count", 0))
        self.validation_row_count = int(state.get("validation_row_count", 0))
        self.best_validation = copy_json_mapping(state.get("best_validation"))
        self.best_validation_by_granularity = {
            str(key): dict(value)
            for key, value in dict(
                state.get("best_validation_by_granularity", {})
            ).items()
        }
        self.trailing_validation_by_granularity = {
            str(key): [dict(row) for row in rows][-self.trailing_count :]
            for key, rows in dict(
                state.get("trailing_validation_by_granularity", {})
            ).items()
        }
        self.selection_counts = {
            str(key): int(value)
            for key, value in dict(state.get("selection_counts", {})).items()
        }
        self.checkpoint_selection = copy_json_mapping(
            state.get("checkpoint_selection")
        )

    def update(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for raw_row in rows:
            row = dict(raw_row)
            split = str(row.get("split") or "")
            if split == "train":
                self.training_row_count += 1
                step = _int_value(row.get("step"))
                if step >= self.last_training_step:
                    self.last_training_step = step
                    self.tokens_seen = max(
                        self.tokens_seen, _int_value(row.get("tokens_seen"))
                    )
                    self.content_tokens_seen = max(
                        self.content_tokens_seen,
                        _int_value(
                            row.get("content_tokens_seen", row.get("tokens_seen"))
                        ),
                    )
                action = row.get("controller_action") or row.get("granularity")
                if action not in (None, ""):
                    key = str(action)
                    self.selection_counts[key] = self.selection_counts.get(key, 0) + 1
                continue
            if split != "validation" or row.get("loss") in (None, ""):
                continue
            self.validation_row_count += 1
            loss = float(row["loss"])
            if not math.isfinite(loss):
                continue
            compact = dict(row)
            if self.best_validation is None or loss < float(
                self.best_validation["loss"]
            ):
                self.best_validation = compact
            granularity = str(row.get("granularity") or "")
            current = self.best_validation_by_granularity.get(granularity)
            if current is None or loss < float(current["loss"]):
                self.best_validation_by_granularity[granularity] = compact
            trailing = self.trailing_validation_by_granularity.setdefault(
                granularity, []
            )
            trailing.append(compact)
            if len(trailing) > self.trailing_count:
                del trailing[: len(trailing) - self.trailing_count]

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "trailing_count": self.trailing_count,
            "last_training_step": self.last_training_step,
            "tokens_seen": self.tokens_seen,
            "content_tokens_seen": self.content_tokens_seen,
            "training_row_count": self.training_row_count,
            "validation_row_count": self.validation_row_count,
            "best_validation": self.best_validation,
            "best_validation_by_granularity": self.best_validation_by_granularity,
            "trailing_validation_by_granularity": (
                self.trailing_validation_by_granularity
            ),
            "selection_counts": self.selection_counts,
            "checkpoint_selection": self.checkpoint_selection,
        }

    def validation_summary_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for trailing in self.trailing_validation_by_granularity.values():
            rows.extend(dict(row) for row in trailing)
        for row in self.best_validation_by_granularity.values():
            rows.append(dict(row))
        unique: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
        for row in rows:
            key = (row.get("step"), row.get("granularity"), row.get("split"))
            unique[key] = row
        return sorted(unique.values(), key=lambda row: _int_value(row.get("step")))


def copy_json_mapping(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


class MetricsJournal:
    """Bounded metrics buffer with durable incremental flushes and resume repair."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        flush_interval_steps: int = 100,
        checkpoint_step: int = 0,
        artifact_io_config: Mapping[str, Any] | None = None,
        heartbeat_writer=None,
        artifact_state: dict[str, Any] | None = None,
        write_enabled: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / "metrics.csv"
        self.flush_interval_steps = max(1, int(flush_interval_steps))
        self._buffer: list[dict[str, Any]] = []
        self._last_flush_step = int(checkpoint_step)
        self._validation_steps: set[int] = set()
        self.artifact_io = resolved_artifact_io(artifact_io_config)
        self.heartbeat_writer = heartbeat_writer
        self.artifact_state = artifact_state
        self.write_enabled = bool(write_enabled)
        self.pending_row_limit = int(self.artifact_io["metrics_pending_row_limit"])
        self.spool_path = self._build_spool_path()
        saved_accumulator = (
            artifact_state.get("metrics_accumulator_state")
            if isinstance(artifact_state, Mapping)
            else None
        )
        self.accumulator = StreamingMetricsAccumulator(saved_accumulator)
        self._retained_row_limit = 100_000
        self._retained_rows: list[dict[str, Any]] = []
        self._retention_overflow = False
        if self.write_enabled:
            self._repair_existing_metrics(
                checkpoint_step=int(checkpoint_step),
                rebuild_accumulator=saved_accumulator is None,
            )
        if self.write_enabled and not self.path.exists():
            write_metrics_csv(
                self.output_dir,
                [],
                artifact_io=self.artifact_io,
                heartbeat_writer=self.heartbeat_writer,
                artifact_state=self.artifact_state,
            )
        self._checkpoint_accumulator()

    def append(
        self,
        rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
        *,
        force: bool = False,
    ) -> None:
        normalized = _normalize_rows(rows)
        if not normalized:
            if force:
                self.flush()
            return
        self.accumulator.update(normalized)
        self._retain_rows(normalized)
        self._checkpoint_accumulator()
        if not self.write_enabled:
            return
        self._buffer.extend(normalized)
        if len(self._buffer) >= self.pending_row_limit:
            self.flush()
        steps = [_int_value(row.get("step")) for row in normalized]
        latest_step = max(steps, default=self._last_flush_step)
        self._validation_steps.update(
            _int_value(row.get("step"))
            for row in normalized
            if row.get("split") == "validation"
        )
        if force or latest_step - self._last_flush_step >= self.flush_interval_steps:
            self.flush()
            self._last_flush_step = latest_step

    def has_validation_at_step(self, step: int) -> bool:
        return int(step) in self._validation_steps

    def flush(self, *, strict: bool = False) -> None:
        if not self.write_enabled:
            return
        if not self._buffer:
            return
        try:
            write_metrics_csv(
                self.output_dir,
                self._buffer,
                append=True,
                artifact_io=self.artifact_io,
                heartbeat_writer=self.heartbeat_writer,
                artifact_state=self.artifact_state,
            )
        except OSError as remote_error:
            if strict:
                raise
            try:
                self._write_spool()
            except OSError as spool_error:
                if len(self._buffer) >= self.pending_row_limit:
                    raise RuntimeError(
                        "Metrics could not be persisted remotely or spooled "
                        f"locally and the pending-row limit ({self.pending_row_limit}) "
                        "was reached"
                    ) from spool_error
                return
            if self.artifact_state is not None:
                self.artifact_state["deferred_metric_rows"] = len(self._buffer)
            emit_artifact_event(
                self.heartbeat_writer,
                "stage_failed",
                "metrics_persistence",
                artifact_path=str(self.path),
                spool_path=str(self.spool_path),
                deferred_metric_rows=len(self._buffer),
                errno=remote_error.errno,
                recoverable=True,
            )
            return

        self._buffer.clear()
        if self.spool_path.exists():
            self.spool_path.unlink(missing_ok=True)
        if self.artifact_state is not None:
            self.artifact_state["deferred_metric_rows"] = 0
        remove_resolved_failure(
            self.artifact_state,
            operation_name="metrics_csv_append",
            target_path=self.path,
        )

    def read_all(self) -> list[dict[str, Any]]:
        self.flush(strict=True)
        return read_metrics_csv(self.path)

    def summary_rows(self) -> list[dict[str, Any]]:
        """Return exact small-run rows or a bounded validation summary."""

        self.flush(strict=True)
        if not self._retention_overflow:
            return [dict(row) for row in self._retained_rows]
        return self.accumulator.validation_summary_rows()

    def iter_rows(self, *, split: str | None = None):
        self.flush(strict=True)
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8", newline="") as metrics_file:
            for row in csv.DictReader(metrics_file):
                if split is None or row.get("split") == split:
                    yield dict(row)

    def training_outcome(self) -> dict[str, int]:
        return {
            "steps_completed": self.accumulator.last_training_step,
            "tokens_seen": self.accumulator.tokens_seen,
            "content_tokens_seen": self.accumulator.content_tokens_seen,
        }

    def record_checkpoint_selection(self, selection: Mapping[str, Any]) -> None:
        self.accumulator.checkpoint_selection = dict(selection)
        self._checkpoint_accumulator()

    def artifact_summary_fields(self) -> dict[str, Any]:
        return {
            "deferred_metric_rows": len(self._buffer),
            "metrics_spool_path": (
                str(self.spool_path) if self.spool_path.exists() else None
            ),
        }

    def _build_spool_path(self) -> Path:
        local_root = os.environ.get("SLURM_TMPDIR")
        if not local_root:
            local_root = tempfile.gettempdir()
        identity = hashlib.sha256(str(self.output_dir).encode("utf-8")).hexdigest()[:12]
        run_id = self.output_dir.name or "run"
        return (
            Path(local_root)
            / "matformer-artifact-spool"
            / f"{run_id}-{identity}.metrics.pending.csv"
        )

    def _checkpoint_accumulator(self) -> None:
        if self.artifact_state is not None:
            self.artifact_state["metrics_accumulator_state"] = (
                self.accumulator.state_dict()
            )

    def _retain_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        if self._retention_overflow:
            return
        for row in rows:
            if len(self._retained_rows) >= self._retained_row_limit:
                self._retained_rows.clear()
                self._retention_overflow = True
                return
            self._retained_rows.append(dict(row))

    def _repair_existing_metrics(
        self,
        *,
        checkpoint_step: int,
        rebuild_accumulator: bool,
    ) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=self.output_dir,
                prefix=".metrics.repair-",
                suffix=".csv",
                delete=False,
            ) as target:
                temporary_path = Path(target.name)
                writer = csv.DictWriter(target, fieldnames=METRICS_COLUMNS)
                writer.writeheader()
                with self.path.open(
                    "r", encoding="utf-8", newline=""
                ) as source:
                    reader = csv.DictReader(source)
                    if reader.fieldnames != METRICS_COLUMNS:
                        return
                    for row in reader:
                        if None in row or any(value is None for value in row.values()):
                            break
                        try:
                            row_step = int(row.get("step") or 0)
                        except (TypeError, ValueError):
                            break
                        if row_step > checkpoint_step:
                            continue
                        writer.writerow(row)
                        if row.get("split") == "validation":
                            self._validation_steps.add(row_step)
                        if rebuild_accumulator:
                            self.accumulator.update([row])
                        self._retain_rows([row])
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _write_spool(self) -> None:
        self.spool_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv_artifact(
            self.spool_path,
            self._buffer,
            METRICS_COLUMNS,
            append=False,
            artifact_io={**self.artifact_io, "max_attempts": 1},
        )


def recover_metrics_rows(
    path: str | Path,
    *,
    checkpoint_step: int,
) -> list[dict[str, Any]]:
    """Keep the valid CSV prefix at or before the loaded checkpoint step."""

    metrics_path = Path(path)
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        return []

    recovered: list[dict[str, Any]] = []
    try:
        with metrics_path.open("r", encoding="utf-8", newline="") as metrics_file:
            reader = csv.DictReader(metrics_file)
            if reader.fieldnames != METRICS_COLUMNS:
                return []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    break
                try:
                    row_step = int(row.get("step") or 0)
                except (TypeError, ValueError):
                    break
                if row_step > int(checkpoint_step):
                    continue
                recovered.append(dict(row))
    except (OSError, csv.Error, UnicodeError):
        return recovered
    return recovered


def read_metrics_csv(path: str | Path) -> list[dict[str, Any]]:
    metrics_path = Path(path)
    if not metrics_path.exists():
        return []
    with metrics_path.open("r", encoding="utf-8", newline="") as metrics_file:
        return [dict(row) for row in csv.DictReader(metrics_file)]


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


def write_config_artifact(
    config: Mapping[str, Any],
    output_dir: str | Path | None = None,
    distributed_context: Any | None = None,
) -> Path | None:
    if not _should_write_shared_artifact(distributed_context):
        return None
    return write_resolved_config(config, output_dir=output_dir)


def summarize_runtime_granularity_pattern_from_config(
    config: Mapping[str, Any],
    runtime_pattern: Any | None = None,
) -> dict[str, Any]:
    """Build a runtime granularity summary that keeps nested-all explicit."""

    run = config.get("run", {})
    training = config.get("training", {})
    if not isinstance(run, Mapping):
        run = {}
    if not isinstance(training, Mapping):
        training = {}

    if resolve_sampling_mode_from_config_sections(run, training) == "nested-all":
        return summarize_granularity_pattern_from_config(config)
    return summarize_granularity_pattern_from_config(
        config,
        runtime_pattern=runtime_pattern,
    )


def build_run_summary(
    config: Mapping[str, Any],
    tokens_seen: int | None = None,
    content_tokens_seen: int | None = None,
    status: str = "completed",
    notes: Iterable[str] | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run = config["run"]
    model = config["model"]
    training = config["training"]
    dataset = config["dataset"]

    if tokens_seen is None:
        tokens_seen = training.get("tokens_seen", training["token_budget"])
    if content_tokens_seen is None:
        content_tokens_seen = training.get("content_tokens_seen", tokens_seen)
    stop_reason = "failed" if status == "failed" else "not_started"
    continuation_state = _build_continuation_state(config, tokens_seen, status)
    warmup_policy = _build_warmup_policy(config)

    summary = {
        "run_id": run["run_id"],
        "phase_id": run["phase_id"],
        "model_family": run["model_family"],
        "model_variant": model["variant"],
        "correction_mode": model.get("correction_mode"),
        "membership_correction": model.get("membership_correction"),
        "continuation_state": continuation_state,
        "monitoring_enabled": bool(config.get("monitoring", {}).get("enabled", False)),
        "monitoring_backend": config.get("monitoring", {}).get(
            "backend",
            DEFAULT_MONITORING_BACKEND,
        ),
        "monitoring_series_metadata": [],
        "warmup_policy": warmup_policy,
        "warmup_completion_step": warmup_policy.get("completion_step"),
        "warmup_completed": warmup_policy.get("completed", False),
        "latest_checkpoint_path": continuation_state.get("latest_checkpoint_path"),
        "model_size_label": _model_shape_label(run),
        "model_shape_label": _model_shape_label(run),
        "active_size_label": run.get("active_size_label"),
        "family_size_slug": run.get("family_size_slug"),
        "family_resolution_rule": run.get("family_resolution_rule"),
        "sampling_mode": resolve_sampling_mode_from_config_sections(run, training),
        "resolved_run_mode": run.get(
            "resolved_run_mode",
            resolve_sampling_mode_from_config_sections(run, training),
        ),
        "resolved_sampling_mode": model.get(
            "resolved_sampling_mode",
            model.get("granularity_sampling_mode", "global"),
        ),
        "requested_granularity_sampling_alias": model.get(
            "requested_granularity_sampling_alias"
        ),
        "granularity_sampling_mode": model.get("granularity_sampling_mode"),
        "granularity_pattern_provenance": _granularity_pattern_provenance(config),
        "granularity_pattern_summary": _granularity_pattern_summary(config),
        **resolved_granularity_artifact_fields(model),
        "correction_context": _correction_context_summary(config),
        **build_adaptive_sampler_artifact_fields(config),
        "completion_label": run["completion_label"],
        "model_family_slug": run.get("model_family_slug"),
        "model_size_slug": run.get("model_size_slug"),
        "token_budget_slug": run.get("token_budget_slug"),
        "output_group": run.get("output_group"),
        "dataset_name": dataset["dataset_name"],
        "dataset_split": dataset["dataset_split"],
        "token_budget": training["token_budget"],
        "base_learning_rate": training["base_learning_rate"],
        "learning_rate_scale_rule": training["learning_rate_scale_rule"],
        "learning_rate_scale_factor": training["learning_rate_scale_factor"],
        "resolved_learning_rate": training["resolved_learning_rate"],
        "warmup_ratio": training["warmup_ratio"],
        "warmup_steps": training.get("warmup_steps"),
        "resolved_warmup_steps": training.get(
            "resolved_warmup_steps",
            training["scheduler"]["resolved_warmup_steps"],
        ),
        "gradient_accumulation_steps": training.get(
            "gradient_accumulation_steps", 1
        ),
        "gradient_clip_norm": training.get("gradient_clip_norm"),
        "scheduler_name": training["scheduler_name"],
        "scheduler_warmup_steps": training["scheduler"]["kwargs"]["warmup_steps"],
        "scheduler_resolved_warmup_steps": training["scheduler"]["resolved_warmup_steps"],
        "scheduler_kwargs": training["scheduler_kwargs"],
        "optimizer_name": training["optimizer_name"],
        "optimizer_kwargs": training["optimizer_kwargs"],
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
        "expected_tokens_per_step": training["expected_tokens_per_step"],
        "expected_tokens_per_microstep": training.get(
            "expected_tokens_per_microstep"
        ),
        "derived_max_steps": training["derived_max_steps"],
        "effective_world_size": training["effective_world_size"],
        "validation_interval_tokens": config.get("evaluation", {})
        .get("validation", {})
        .get("interval_tokens", 0),
        "tokenizer_manifest_hash": model.get("tokenizer_manifest_hash"),
        "tokens_seen": tokens_seen,
        "content_tokens_seen": content_tokens_seen,
        "stop_reason": stop_reason,
        "seed": run.get("seed"),
        "validation_manifest_hash": config.get("validation_manifest_hash"),
        "validation_loss_aggregation": config.get(
            "validation_loss_aggregation"
        ),
        "comparison_control_signature": config.get(
            "comparison_control_signature"
        ),
        "status": status,
        "output_root": run["output_root"],
        "output_dir": run["output_dir"],
        "d_model": model.get("d_model", model.get("hidden_size")),
        "num_layers": model.get("num_layers"),
        "num_attention_heads": model.get("num_attention_heads"),
        "context_length": model.get("context_length"),
        "vocab_size": model.get("vocab_size"),
        "parameter_counts": config.get("parameter_counts"),
        "parameter_counts_by_granularity": config.get(
            "parameter_counts_by_granularity"
        ),
        "preset_selections": training.get("preset_selections", {}),
        "preset_registry_paths": training.get("preset_registry_paths", {}),
        **build_checkpoint_summary_fields(
            config,
            metrics_rows=[],
            validation_enabled=bool(
                config.get("evaluation", {})
                .get("validation", {})
                .get("enabled", False)
                or config.get("evaluation", {})
                .get("validation", {})
                .get("run_at_completion", False)
            ),
            save_checkpoints=config.get("outputs", {}).get(
                "save_checkpoints",
                False,
            ),
        ),
        "notes": list(notes or []),
        "metrics_path": None,
        "scaling_results_path": None,
        "extraction_metadata_path": None,
        "scaling_results_unavailable_reason": None,
        "artifact_retry_count": 0,
        "artifact_last_errno": None,
        "last_durable_checkpoint_step": 0,
        "deferred_metric_rows": 0,
        "skipped_periodic_checkpoints": 0,
        "checkpoint_staging_mode": config.get("outputs", {})
        .get("artifact_io", {})
        .get("checkpoint_staging", "auto"),
        "unresolved_artifact_failures": [],
    }

    if extra_fields:
        summary.update(extra_fields)

    _require_fields(summary, RUN_SUMMARY_FIELDS, "run_summary.json")
    return summary


def build_monitoring_summary_fields(
    config: Mapping[str, Any],
    metrics_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    monitoring = config.get("monitoring", {})
    if not isinstance(monitoring, Mapping) or not monitoring.get("enabled", False):
        return {
            "monitoring_backend": monitoring.get(
                "backend",
                DEFAULT_MONITORING_BACKEND,
            )
            if isinstance(monitoring, Mapping)
            else DEFAULT_MONITORING_BACKEND,
            "monitoring_series_metadata": [],
        }

    return {
        "monitoring_backend": monitoring.get(
            "backend",
            DEFAULT_MONITORING_BACKEND,
        ),
        "monitoring_series_metadata": build_monitoring_series_metadata(
            config,
            metrics_rows,
        ),
    }


def _build_continuation_state(
    config: Mapping[str, Any],
    tokens_seen: int,
    status: str,
) -> dict[str, Any]:
    run = config["run"]
    continuation = run.get("continuation", {})
    if not isinstance(continuation, Mapping):
        continuation = {}

    continuation_enabled = bool(continuation.get("enabled", False))
    continuation_status = continuation.get("status")
    if continuation_status in (None, ""):
        continuation_status = "fresh"
        if continuation_enabled:
            if status == "failed":
                continuation_status = "failed"
            elif status == "completed":
                continuation_status = "completed"

    return {
        "run_id": run["run_id"],
        "output_dir": run["output_dir"],
        "latest_checkpoint_path": continuation.get("latest_checkpoint_path"),
        "last_completed_step": continuation.get("last_completed_step", 0),
        "tokens_seen": tokens_seen,
        "status": continuation_status,
        "resume_count": continuation.get("resume_count", 0),
    }


def _build_warmup_policy(config: Mapping[str, Any]) -> dict[str, Any]:
    training = config.get("training", {})
    warmup = training.get("pre_nested_warmup", {})
    if not isinstance(warmup, Mapping):
        warmup = {}

    policy = warmup.get("policy", "full_only")
    resolved = {
        "enabled": bool(warmup.get("enabled", False)),
        "duration": warmup.get("duration", 0),
        "unit": warmup.get("unit", "epochs"),
        "policy": policy,
        "completed": warmup.get("completed", False),
        "completion_step": warmup.get("completion_step"),
        "transition_reason": warmup.get("transition_reason"),
    }
    if policy == "balanced_global":
        resolved.update(
            action_interval_steps=warmup.get("action_interval_steps"),
            schedule_seed=warmup.get("schedule_seed"),
            schedule_hash=warmup.get("schedule_hash"),
            schedule=warmup.get("schedule"),
            passes=warmup.get("passes"),
            requested_steps=warmup.get("duration", 0),
            completed_steps=warmup.get("completed_steps", 0),
            current_window_index=warmup.get("current_window_index", 0),
            current_window_offset=warmup.get("current_window_offset", 0),
            per_granularity_counts=warmup.get("per_granularity_counts", {}),
            controller_start_step=warmup.get("controller_start_step"),
            posterior_updated_during_warmup=False,
        )
    return resolved


def write_run_summary(
    output_dir: str | Path,
    summary: Mapping[str, Any],
    filename: str = "run_summary.json",
    distributed_context: Any | None = None,
    artifact_io: Mapping[str, Any] | None = None,
    heartbeat_writer=None,
    artifact_state: dict[str, Any] | None = None,
) -> Path | None:
    if not _should_write_shared_artifact(distributed_context):
        return None
    _require_fields(summary, RUN_SUMMARY_FIELDS, filename)
    return write_json_artifact(
        Path(output_dir) / filename,
        summary,
        artifact_io=artifact_io,
        heartbeat_writer=heartbeat_writer,
        artifact_state=artifact_state,
    )


def write_failed_run_summary(
    config: Mapping[str, Any],
    error_message: str,
    output_dir: str | Path | None = None,
    tokens_seen: int = 0,
    content_tokens_seen: int = 0,
    notes: Iterable[str] | None = None,
    distributed_context: Any | None = None,
) -> Path | None:
    if not _should_write_shared_artifact(distributed_context):
        return None
    failure_notes = [error_message]
    failure_notes.extend(notes or [])
    summary = build_run_summary(
        config,
        tokens_seen=tokens_seen,
        content_tokens_seen=content_tokens_seen,
        status="failed",
        notes=failure_notes,
    )
    run_output_dir = output_dir or config["run"]["output_dir"]
    return write_run_summary(run_output_dir, summary, artifact_io=config)


def build_checkpoint_summary_fields(
    config: Mapping[str, Any],
    metrics_rows: Iterable[Mapping[str, Any]],
    validation_enabled: bool | None = None,
    save_checkpoints: bool | None = None,
) -> dict[str, Any]:
    output_dir = Path(config["run"]["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"

    if validation_enabled is None:
        evaluation = config.get("evaluation", {})
        validation_enabled = bool(
            evaluation.get("validation", {}).get("enabled", False)
            or evaluation.get("validation", {}).get(
                "run_at_completion", False
            )
        )
    if save_checkpoints is None:
        save_checkpoints = bool(config.get("outputs", {}).get("save_checkpoints", False))

    fields = {
        "checkpoint_status": "none",
        "best_checkpoint_path": None,
        "final_checkpoint_path": None,
        "checkpoint_metric": None,
        "checkpoint_metric_value": None,
        "checkpoint_selection_step": None,
        "checkpoint_unavailable_reason": None,
        **resolved_granularity_artifact_fields(config.get("model", {})),
    }

    if not save_checkpoints:
        fields["checkpoint_unavailable_reason"] = "checkpoint writes disabled"
        return fields

    if not validation_enabled:
        fields["checkpoint_status"] = "final"
        fields["final_checkpoint_path"] = str(checkpoint_dir / "final.pt")
        return fields

    best_row, metric_name, metric_value = _best_validation_metric_row(metrics_rows)
    if best_row is None:
        fields["checkpoint_status"] = "unavailable"
        fields["checkpoint_unavailable_reason"] = (
            "validation enabled but no validation loss or perplexity rows were available"
        )
        return fields

    selection_step = _int_value(best_row.get("step"))
    fields.update(
        {
            "checkpoint_status": "best_eval",
            "best_checkpoint_path": str(
                checkpoint_dir / f"best_eval_step_{selection_step}.pt"
            ),
            "checkpoint_metric": metric_name,
            "checkpoint_metric_value": metric_value,
            "checkpoint_selection_step": selection_step,
        }
    )
    if best_row.get("granularity") is not None:
        fields["checkpoint_selection_granularity"] = best_row["granularity"]
    return fields


def write_metrics_csv(
    output_dir: str | Path,
    rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    append: bool = False,
    distributed_context: Any | None = None,
    artifact_io: Mapping[str, Any] | None = None,
    heartbeat_writer=None,
    artifact_state: dict[str, Any] | None = None,
) -> Path | None:
    return write_csv_artifact(
        Path(output_dir) / "metrics.csv",
        rows,
        METRICS_COLUMNS,
        append=append,
        distributed_context=distributed_context,
        artifact_io=artifact_io,
        heartbeat_writer=heartbeat_writer,
        artifact_state=artifact_state,
    )


def write_task_results_csv(
    output_dir: str | Path,
    rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    append: bool = False,
    distributed_context: Any | None = None,
) -> Path | None:
    return write_csv_artifact(
        Path(output_dir) / "task_results.csv",
        rows,
        TASK_RESULTS_COLUMNS,
        append=append,
        distributed_context=distributed_context,
    )


def write_scaling_results_csv(
    output_dir: str | Path,
    rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    append: bool = False,
    distributed_context: Any | None = None,
) -> Path | None:
    return write_csv_artifact(
        Path(output_dir) / "scaling_results.csv",
        rows,
        SCALING_RESULTS_COLUMNS,
        append=append,
        distributed_context=distributed_context,
    )


def write_consistency_results_csv(
    output_dir: str | Path,
    rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    append: bool = False,
    distributed_context: Any | None = None,
) -> Path | None:
    return write_csv_artifact(
        Path(output_dir) / "consistency_results.csv",
        build_consistency_result_rows(rows),
        CONSISTENCY_RESULTS_COLUMNS,
        append=append,
        distributed_context=distributed_context,
    )


def build_consistency_result_rows(
    rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(rows, Mapping):
        rows = [rows]

    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {
                "comparison_id": row.get("comparison_id"),
                "small_run_id": row.get("small_run_id"),
                "large_run_id": row.get("large_run_id"),
                "small_granularity": row.get("small_granularity"),
                "large_granularity": row.get("large_granularity"),
                "metric_name": _normalize_consistency_metric_name(row),
                "metric_value": row.get("metric_value"),
                "sample_count": row.get("sample_count"),
            }
        )
    return normalized_rows


def _normalize_consistency_metric_name(row: Mapping[str, Any]) -> str:
    metric_name = str(row.get("metric_name") or "")
    top_k = row.get("top_k")

    if metric_name == "top_k_overlap" and top_k not in (None, ""):
        return f"top_k_overlap@{int(top_k)}"
    if metric_name == "kl_divergence" and row.get("deferred"):
        return "kl_divergence_deferred"
    return metric_name


def build_parameter_counts_by_granularity(
    model: Any,
    granularities: Iterable[str],
    trainable_only: bool = False,
) -> dict[str, dict[str, Any]]:
    from src.utils.model_size import model_parameter_counts

    return {
        granularity: model_parameter_counts(
            model,
            trainable_only=trainable_only,
            granularity=granularity,
        )
        for granularity in granularities
    }


def build_scaling_result_rows(
    config: Mapping[str, Any],
    metrics_rows: Iterable[Mapping[str, Any]],
    parameter_counts_by_granularity: Mapping[str, Mapping[str, Any]],
    comparison_id_prefix: str | None = None,
) -> list[dict[str, Any]]:
    run = config["run"]
    model = config["model"]
    training = config.get("training", {})
    metrics_rows = list(metrics_rows)
    validation_rows = latest_metric_rows_by_granularity(
        metrics_rows,
        split="validation",
    )
    latest_rows = validation_rows
    if not latest_rows:
        return []

    configured_granularities = list(model["granularities"])
    preferred_order = ["micro", "small", "medium", "large", "full"]
    granularities = [
        granularity
        for granularity in preferred_order
        if granularity in configured_granularities
    ] + [
        granularity
        for granularity in configured_granularities
        if granularity not in preferred_order
    ]
    trailing_count = int(
        config.get("evaluation", {})
        .get("validation", {})
        .get("trailing_summary_evaluations", 5)
    )
    prefix_widths = {
        str(entry["name"]): int(entry["prefix_width"])
        for entry in model.get("ffn_prefix_metadata", [])
    }

    rows = []
    for granularity in granularities:
        metric_row = latest_rows.get(granularity)
        if metric_row is None:
            raise ArtifactError(
                "scaling_results.csv missing metric row for "
                f"granularity={granularity}"
            )

        parameter_counts = parameter_counts_by_granularity.get(granularity)
        if parameter_counts is None:
            raise ArtifactError(
                "scaling_results.csv missing parameter counts for "
                f"granularity={granularity}"
            )
        _require_fields(
            parameter_counts,
            PARAMETER_COUNT_FIELDS[:4],
            "scaling_results.csv",
        )

        comparison_id = f"{comparison_id_prefix or run['run_id']}__{granularity}"
        granularity_rows = sorted(
            (
                row
                for row in metrics_rows
                if row.get("split") == "validation"
                and row.get("granularity") == granularity
                and row.get("loss") is not None
            ),
            key=lambda row: int(row.get("step", 0)),
        )
        best_row = min(granularity_rows, key=lambda row: float(row["loss"]))
        trailing_rows = granularity_rows[-trailing_count:]
        trailing_losses = [float(row["loss"]) for row in trailing_rows]
        final_loss = float(metric_row["loss"])
        best_loss = float(best_row["loss"])
        best_step = int(best_row["step"])
        rows.append(
            {
                "comparison_id": comparison_id,
                "run_id": run["run_id"],
                "model_family": run["model_family"],
                "model_size_label": _model_shape_label(run),
                "model_shape_label": _model_shape_label(run),
                "sampling_mode": resolve_sampling_mode_from_config_sections(
                    run,
                    training,
                ),
                "model_family_slug": run.get("model_family_slug"),
                "model_size_slug": run.get("model_size_slug"),
                "token_budget_slug": run.get("token_budget_slug"),
                "output_group": run.get("output_group"),
                "completion_label": run["completion_label"],
                "granularity": granularity,
                "d_model": model.get("d_model", model.get("hidden_size")),
                "num_layers": model.get("num_layers"),
                "num_attention_heads": model.get("num_attention_heads"),
                "context_length": model.get("context_length"),
                "vocab_size": model.get("vocab_size"),
                "token_budget": training.get("token_budget"),
                "effective_world_size": training.get("effective_world_size"),
                "total_parameters": parameter_counts["total_parameters"],
                "embedding_parameters": parameter_counts["embedding_parameters"],
                "lm_head_parameters": parameter_counts["lm_head_parameters"],
                "non_embedding_parameters": parameter_counts[
                    "non_embedding_parameters"
                ],
                "ffn_parameters": parameter_counts.get("ffn_parameters"),
                "attention_parameters": parameter_counts.get(
                    "attention_parameters"
                ),
                "other_non_embedding_parameters": parameter_counts.get(
                    "other_non_embedding_parameters"
                ),
                "lm_head_counting": parameter_counts.get("lm_head_counting"),
                "checkpoint_path": str(
                    Path(run["output_dir"])
                    / "checkpoints"
                    / f"best_eval_step_{best_step}.pt"
                ),
                "loss": final_loss,
                "perplexity": metric_row["perplexity"],
                "final_validation_loss": final_loss,
                "final_validation_perplexity": metric_row["perplexity"],
                "best_validation_loss": best_loss,
                "best_validation_perplexity": best_row.get("perplexity"),
                "best_validation_step": best_step,
                "best_validation_checkpoint": str(
                    Path(run["output_dir"])
                    / "checkpoints"
                    / f"best_eval_step_{best_step}.pt"
                ),
                "trailing_validation_mean": statistics.fmean(trailing_losses),
                "trailing_validation_sample_stddev": (
                    statistics.stdev(trailing_losses)
                    if len(trailing_losses) > 1
                    else 0.0
                ),
                "trailing_validation_min": min(trailing_losses),
                "trailing_validation_max": max(trailing_losses),
                "trailing_validation_count": len(trailing_losses),
                "final_minus_best_loss": final_loss - best_loss,
                "evaluation_target_tokens": metric_row.get(
                    "evaluation_target_tokens"
                ),
                "effective_width": prefix_widths.get(granularity),
                "validation_manifest_hash": config.get(
                    "validation_manifest_hash"
                ),
                "comparison_control_signature": config.get(
                    "comparison_control_signature"
                ),
                "average_downstream_accuracy": None,
            }
        )

    return rows


def build_pilot_comparison_rows(
    comparison_id: str,
    run_summaries: Iterable[Mapping[str, Any]],
    omitted_rows: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    run_summaries = list(run_summaries)
    _validate_comparison_provenance(run_summaries)
    rows: list[dict[str, Any]] = []

    for summary in run_summaries:
        granularities = _summary_granularities(summary)
        for granularity in granularities:
            parameter_counts = _summary_parameter_counts(summary, granularity)
            rows.append(
                _with_artifact_defaults(
                    {
                        "comparison_id": comparison_id,
                        "run_id": summary["run_id"],
                        "run_status": summary.get("status", "completed"),
                        "omit_reason": None,
                        "model_family": summary.get("model_family"),
                        "model_variant": summary.get("model_variant"),
                        "model_size_label": _model_shape_label(summary),
                        "model_shape_label": _model_shape_label(summary),
                        "sampling_mode": summary.get("sampling_mode"),
                        "model_family_slug": summary.get("model_family_slug"),
                        "model_size_slug": summary.get("model_size_slug"),
                        "token_budget_slug": summary.get("token_budget_slug"),
                        "output_group": summary.get("output_group"),
                        "completion_label": summary.get(
                            "completion_label",
                            "run",
                        ),
                        "granularity": granularity,
                        "d_model": summary.get("d_model"),
                        "num_layers": summary.get("num_layers"),
                        "num_attention_heads": summary.get(
                            "num_attention_heads"
                        ),
                        "context_length": summary.get("context_length"),
                        "vocab_size": summary.get("vocab_size"),
                        "token_budget": summary.get("token_budget"),
                        "effective_world_size": summary.get(
                            "effective_world_size"
                        ),
                        "total_parameters": parameter_counts.get(
                            "total_parameters"
                        ),
                        "embedding_parameters": parameter_counts.get(
                            "embedding_parameters"
                        ),
                        "lm_head_parameters": parameter_counts.get(
                            "lm_head_parameters"
                        ),
                        "non_embedding_parameters": parameter_counts.get(
                            "non_embedding_parameters"
                        ),
                        "ffn_parameters": parameter_counts.get(
                            "ffn_parameters"
                        ),
                        "attention_parameters": parameter_counts.get(
                            "attention_parameters"
                        ),
                        "other_non_embedding_parameters": parameter_counts.get(
                            "other_non_embedding_parameters"
                        ),
                        "lm_head_counting": parameter_counts.get(
                            "lm_head_counting"
                        ),
                        "checkpoint_status": _summary_checkpoint_status(
                            summary
                        ),
                        "checkpoint_path": _summary_checkpoint_path(summary),
                        "checkpoint_metric": summary.get("checkpoint_metric"),
                    }
                )
            )

    for omitted_row in omitted_rows or []:
        omit_reason = omitted_row.get("omit_reason")
        rows.append(
            _with_artifact_defaults(
                {
                    "comparison_id": comparison_id,
                    "run_id": omitted_row.get("run_id"),
                    "run_status": "omitted",
                    "omit_reason": omit_reason,
                    "model_family": omitted_row.get("model_family", "standalone"),
                    "model_variant": omitted_row.get("model_variant"),
                    "model_size_label": _model_shape_label(
                        {
                            "model_shape_label": omitted_row.get(
                                "model_shape_label",
                                "dmodel256",
                            )
                        }
                    ),
                    "model_shape_label": omitted_row.get(
                        "model_shape_label",
                        "dmodel256",
                    ),
                    "sampling_mode": omitted_row.get(
                        "sampling_mode",
                        "standalone",
                    ),
                    "completion_label": omitted_row.get("completion_label", "run"),
                    "model_family_slug": omitted_row.get("model_family_slug"),
                    "model_size_slug": omitted_row.get("model_size_slug"),
                    "token_budget_slug": omitted_row.get("token_budget_slug"),
                    "output_group": omitted_row.get("output_group"),
                    "granularity": omitted_row.get("granularity"),
                    "d_model": omitted_row.get("d_model"),
                    "num_layers": omitted_row.get("num_layers"),
                    "num_attention_heads": omitted_row.get(
                        "num_attention_heads"
                    ),
                    "context_length": omitted_row.get("context_length"),
                    "vocab_size": omitted_row.get("vocab_size"),
                    "token_budget": omitted_row.get("token_budget"),
                    "effective_world_size": omitted_row.get(
                        "effective_world_size"
                    ),
                    "checkpoint_status": "unavailable",
                    "checkpoint_path": None,
                    "checkpoint_metric": None,
                }
            )
        )

    return rows


def _validate_comparison_provenance(
    run_summaries: Iterable[Mapping[str, Any]],
) -> None:
    summaries = [
        summary
        for summary in run_summaries
        if summary.get("status", "completed") == "completed"
    ]
    manifests = {
        str(summary["validation_manifest_hash"])
        for summary in summaries
        if summary.get("validation_manifest_hash")
    }
    signatures = {
        str(summary["comparison_control_signature"])
        for summary in summaries
        if summary.get("comparison_control_signature")
    }
    corrected = bool(manifests or signatures)
    if corrected and any(
        not summary.get("validation_manifest_hash")
        or not summary.get("comparison_control_signature")
        for summary in summaries
    ):
        raise ArtifactError(
            "Corrected comparison rows require validation-manifest and control signatures"
        )
    if len(manifests) > 1 or len(signatures) > 1:
        raise ArtifactError(
            "Results are not comparable: validation manifests or control signatures differ"
        )


def build_speculative_task_rows(
    config: Mapping[str, Any],
    pair_results: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    run = config["run"]
    suite_id = "speculative-alignment"
    requested_metrics = [str(metric) for metric in config.get("metrics", [])]
    rows = []

    for result in pair_results:
        granularity_label = (
            f"{result.get('draft_granularity')}->{result.get('verifier_granularity')}"
        )
        row_base = {
            "run_id": run["run_id"],
            "suite_id": suite_id,
            "task": result["pair_id"],
            "model_family": result["pair_type"],
            "model_size_label": result.get("model_shape_label"),
            "model_shape_label": result.get("model_shape_label"),
            "sampling_mode": result.get("sampling_mode"),
            "model_family_slug": run.get("model_family_slug"),
            "model_size_slug": run.get("model_size_slug"),
            "token_budget_slug": run.get("token_budget_slug"),
            "output_group": run.get("output_group"),
            "granularity": granularity_label,
        }
        for metric_name in requested_metrics:
            rows.append(
                row_base
                | {
                    "metric_name": metric_name,
                    "metric_value": result.get(metric_name),
                }
            )

    return rows


def build_baseline_match_row(
    nested_config: Mapping[str, Any],
    standalone_config: Mapping[str, Any],
    granularity: str,
    nested_counts: Mapping[str, Any] | None = None,
    standalone_counts: Mapping[str, Any] | None = None,
    match_notes: Iterable[str] | None = None,
) -> dict[str, Any]:
    nested_run = nested_config["run"]
    standalone_run = standalone_config["run"]
    row = {
        "match_id": baseline_match_id(
            nested_run["run_id"],
            standalone_run["run_id"],
            granularity,
        ),
        "nested_run_id": nested_run["run_id"],
        "standalone_run_id": standalone_run["run_id"],
        "granularity": granularity,
        "non_embedding_parameters_nested": _non_embedding_count(nested_counts),
        "non_embedding_parameters_standalone": _non_embedding_count(
            standalone_counts
        ),
        "match_notes": list(match_notes or []),
    }
    _require_fields(row, BASELINE_MATCH_FIELDS, "baseline match row")
    return row


def baseline_match_id(
    nested_run_id: str,
    standalone_run_id: str,
    granularity: str,
) -> str:
    return f"{nested_run_id}__{standalone_run_id}__{granularity}"


def latest_metric_rows_by_granularity(
    metrics_rows: Iterable[Mapping[str, Any]],
    split: str,
) -> dict[str, Mapping[str, Any]]:
    latest_rows: dict[str, tuple[int, int, Mapping[str, Any]]] = {}
    for row_index, row in enumerate(metrics_rows):
        if row.get("split") != split:
            continue
        granularity = row.get("granularity")
        if granularity is None:
            continue

        row_key = (_int_value(row.get("step")), row_index)
        current = latest_rows.get(str(granularity))
        if current is None or row_key > current[:2]:
            latest_rows[str(granularity)] = (*row_key, row)

    return {
        granularity: row_with_key[2]
        for granularity, row_with_key in latest_rows.items()
    }


CONTROLLER_EVENT_TYPES = {
    "initial_boundary",
    "completed_window",
    "terminal_incomplete",
    "controller_failure",
    "warmup_schedule_initialized",
    "warmup_window_completed",
    "warmup_completed",
    "warmup_terminal_incomplete",
    "episode_initialized",
    "episode_completed",
    "posterior_reset",
    "posterior_preserved",
    "acquisition_progress",
    "acquisition_completed",
    "panelgrad_refresh_completed",
    "panelgrad_refresh_failed",
    "panelgrad_terminal_partial",
    "panelgrad_terminal_complete",
}


def append_controller_event(
    path: str | Path,
    event: Mapping[str, Any],
    *,
    distributed_context: Any | None = None,
    artifact_io: Mapping[str, Any] | None = None,
    heartbeat_writer=None,
    artifact_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Validate and append one committed controller lifecycle event."""

    if not _should_write_shared_artifact(distributed_context):
        return None
    normalized = _controller_json_value(event)
    _validate_controller_event(normalized)
    return append_jsonl_artifact(
        path,
        normalized,
        settings=artifact_io,
        heartbeat_writer=heartbeat_writer,
        state=artifact_state,
    )


def append_controller_events(
    path: str | Path,
    events: Iterable[Mapping[str, Any]],
    *,
    distributed_context: Any | None = None,
    artifact_io: Mapping[str, Any] | None = None,
    heartbeat_writer=None,
    artifact_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Validate and transactionally append one same-boundary event batch."""

    if not _should_write_shared_artifact(distributed_context):
        return None
    normalized = [_controller_json_value(event) for event in events]
    if not normalized:
        raise ArtifactError("Controller event batch must not be empty")
    for event in normalized:
        _validate_controller_event(event)
    return append_jsonl_artifacts(
        path,
        normalized,
        settings=artifact_io,
        heartbeat_writer=heartbeat_writer,
        state=artifact_state,
    )


def read_controller_events(path: str | Path) -> list[dict[str, Any]]:
    journal_path = Path(path)
    if not journal_path.exists():
        return []
    events = []
    with journal_path.open("r", encoding="utf-8") as journal_file:
        for line_number, line in enumerate(journal_file, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ArtifactError(
                    f"Invalid controller journal JSON on line {line_number}"
                ) from error
            _validate_controller_event(event)
            events.append(event)
    return events


def build_controller_summary(
    *,
    controller_state: Mapping[str, Any],
    controller_events: Iterable[Mapping[str, Any]],
    controller_metrics_path: str | Path,
) -> dict[str, Any]:
    """Aggregate the append-only journal without discarding final belief state."""

    state = _controller_json_value(controller_state)
    events = [_controller_json_value(event) for event in controller_events]
    for event in events:
        _validate_controller_event(event)
    if state.get("method_family") == "panelgrad_gradient_rms":
        refresh = state.get("refresh", {})
        sampling = state.get("sampling", {})
        completed = [
            event
            for event in events
            if event.get("event_type") == "panelgrad_refresh_completed"
        ]
        total_duration = sum(
            float(event.get("duration_seconds", 0.0)) for event in completed
        )
        epsilon_history = [
            {
                "refresh_index": event.get("window_index"),
                "active_epsilon": event.get("active_epsilon"),
                "epsilon_schedule_step": event.get("epsilon_schedule_step"),
            }
            for event in completed
        ]
        journal_path = Path(controller_metrics_path)
        return {
            "schema_version": state.get("schema_version"),
            "method_family": state.get("method_family"),
            "method_version": state.get("method_version"),
            "strategy": "panelgrad",
            "scope": state.get("scope"),
            "ordered_granularities": state.get("ordered_granularities"),
            "policy": state.get("policy"),
            "epsilon_schedule": state.get("policy", {}).get(
                "epsilon_schedule"
            ),
            "support": state.get("support"),
            "manifest_hashes": state.get("manifest_hashes"),
            "refresh_count": len(completed),
            "final_scores": refresh.get("scores"),
            "final_q": refresh.get("q"),
            "final_p": refresh.get("p"),
            "final_entropy": refresh.get("entropy"),
            "active_epsilon": refresh.get("active_epsilon"),
            "epsilon_schedule_step": refresh.get("epsilon_schedule_step"),
            "epsilon_history": epsilon_history,
            "exposure_counts": sampling.get("exposure_counts"),
            "sample_count": sampling.get("sample_count"),
            "terminal": state.get("terminal"),
            "warmup": state.get("warmup"),
            "cumulative_measurement_duration_seconds": total_duration,
            "cumulative_backward_evaluations": sum(
                int(event.get("backward_evaluation_count", 0))
                for event in completed
            ),
            "controller_metrics_path": str(journal_path),
            "controller_metrics_hash": (
                hashlib.sha256(journal_path.read_bytes()).hexdigest()
                if journal_path.exists()
                else None
            ),
        }
    journal_path = Path(controller_metrics_path)
    completed = [
        event for event in events if event["event_type"] == "completed_window"
    ]
    evaluations = [
        event
        for event in events
        if event["event_type"] in {"initial_boundary", "completed_window"}
    ]
    failures = [
        event for event in events if event["event_type"] == "controller_failure"
    ]
    warmup_events = [
        event for event in events if str(event.get("event_type", "")).startswith("warmup_")
    ]
    warmup_initialized = next(
        (
            event
            for event in warmup_events
            if event["event_type"] == "warmup_schedule_initialized"
        ),
        None,
    )
    warmup_terminal = next(
        (
            event
            for event in reversed(warmup_events)
            if event["event_type"]
            in {"warmup_completed", "warmup_terminal_incomplete"}
        ),
        None,
    )
    initial_boundary = next(
        (event for event in events if event["event_type"] == "initial_boundary"),
        None,
    )
    belief = state.get("belief", {})
    covariance = belief.get("posterior_covariance")
    window = state.get("window", {})
    terminal_status = window.get("terminal_status", "continuing")
    if terminal_status == "complete_boundary":
        terminal_status = "complete"
    action_frequencies = controller_action_frequency_counts(events)
    forced_action_frequencies = controller_action_frequency_counts(
        events,
        selection_source="forced_acquisition",
    )
    thompson_action_frequencies = controller_action_frequency_counts(
        events,
        selection_source="thompson",
    )
    effect_uncertainty = controller_effect_uncertainty_summary(
        state.get("feature_schema"),
        covariance,
    )

    return {
        "schema_version": int(state.get("schema_version", 1)),
        "method_family": state.get("method_family"),
        "method_version": state.get("method_version"),
        "strategy": state.get("strategy"),
        "scope": state.get("scope"),
        "ordered_granularities": list(state.get("ordered_granularities", [])),
        "feature_schema": state.get("feature_schema"),
        "probabilistic_inputs": state.get("probabilistic_inputs"),
        "manifest_hashes": state.get("manifest_hashes"),
        "decision_interval_steps": window.get("decision_interval_steps"),
        "warmup_policy": (
            "balanced_global" if warmup_initialized is not None else "full_only"
        ),
        "requested_warmup_steps": (
            warmup_initialized.get("requested_warmup_steps", 0)
            if warmup_initialized is not None
            else 0
        ),
        "completed_warmup_steps": (
            warmup_terminal.get("completed_warmup_steps", 0)
            if warmup_terminal is not None
            else 0
        ),
        "warmup_schedule_seed": (
            warmup_initialized.get("schedule_seed")
            if warmup_initialized is not None
            else None
        ),
        "warmup_schedule_hash": (
            warmup_initialized.get("schedule_hash")
            if warmup_initialized is not None
            else None
        ),
        "warmup_schedule": (
            warmup_initialized.get("schedule")
            if warmup_initialized is not None
            else None
        ),
        "warmup_action_interval_steps": (
            warmup_initialized.get("action_interval_steps")
            if warmup_initialized is not None
            else None
        ),
        "warmup_action_counts": (
            warmup_terminal.get("per_granularity_counts", {})
            if warmup_terminal is not None
            else {}
        ),
        "controller_start_step": (
            initial_boundary.get("boundary_step")
            if initial_boundary is not None
            else (
                warmup_terminal.get("controller_start_step")
                if warmup_terminal is not None
                else None
            )
        ),
        "baseline_step": (
            initial_boundary.get("boundary_step")
            if initial_boundary is not None
            else None
        ),
        "first_adaptive_action": (
            initial_boundary.get("selected_action")
            if initial_boundary is not None
            else None
        ),
        "prior_untouched": (
            initial_boundary.get("prior_untouched")
            if initial_boundary is not None
            else None
        ),
        "posterior_updated_during_warmup": any(
            event.get("posterior_updated") is not False for event in warmup_events
        )
        if warmup_events
        else False,
        "completed_observation_count": len(completed),
        "controller_evaluation_count": len(evaluations),
        "action_frequencies": action_frequencies,
        "forced_acquisition_action_frequencies": forced_action_frequencies,
        "thompson_action_frequencies": thompson_action_frequencies,
        "action_entropy": _action_frequency_entropy(action_frequencies),
        "forced_acquisition_action_entropy": _action_frequency_entropy(
            forced_action_frequencies
        ),
        "thompson_action_entropy": _action_frequency_entropy(
            thompson_action_frequencies
        ),
        "per_block_granularity_frequencies": (
            action_frequencies if state.get("scope") == "per_block" else None
        ),
        "final_posterior_mean": belief.get("posterior_mean"),
        "final_posterior_covariance": covariance,
        "uncertainty_summary": controller_uncertainty_summary(covariance),
        "effect_uncertainty": effect_uncertainty,
        "boundary_summaries": {
            "objective": _scalar_summary(
                [
                    event.get("post_window_objective", event.get("controller_objective"))
                    for event in evaluations
                ]
            ),
            "reward": _scalar_summary([event.get("reward") for event in completed]),
            "prediction_error": _scalar_summary(
                [event.get("prediction_error") for event in completed]
            ),
        },
        "terminal_window": {
            "status": terminal_status,
            "window_index": window.get("window_index"),
            "completed_optimizer_steps": window.get("completed_optimizer_steps"),
            "decision_interval_steps": window.get("decision_interval_steps"),
        },
        "resume_provenance": state.get("resume"),
        "failure_summary": failures[-1] if failures else state.get("failure"),
        "reset": state.get("reset"),
        "reset_enabled": bool(state.get("reset", {}).get("enabled", False)),
        "reset_count": state.get("reset", {}).get("reset_count", 0),
        "reset_steps": state.get("reset", {}).get("reset_steps", []),
        "completed_episodes": state.get("reset", {}).get(
            "completed_episodes", []
        ),
        "controller_metrics_path": str(journal_path),
        "controller_metrics_hash": (
            hashlib.sha256(journal_path.read_bytes()).hexdigest()
            if journal_path.exists()
            else hashlib.sha256(b"").hexdigest()
        ),
    }


def write_controller_summary(
    output_dir: str | Path,
    summary: Mapping[str, Any],
    *,
    distributed_context: Any | None = None,
    artifact_io: Mapping[str, Any] | None = None,
    heartbeat_writer=None,
    artifact_state: dict[str, Any] | None = None,
) -> Path | None:
    return write_json_artifact(
        Path(output_dir) / "controller_summary.json",
        _controller_json_value(summary),
        distributed_context=distributed_context,
        artifact_io=artifact_io,
        heartbeat_writer=heartbeat_writer,
        artifact_state=artifact_state,
    )


def controller_action_frequency_counts(
    events: Iterable[Mapping[str, Any]],
    *,
    selection_source: str | None = None,
) -> dict[str, Any]:
    event_list = list(events)
    scope = next(
        (
            event.get("scope")
            for event in event_list
            if event.get("scope") in {"global", "per_block"}
        ),
        "global",
    )
    if scope == "per_block":
        completed_actions = [
            event.get("action")
            for event in event_list
            if event.get("event_type") == "completed_window"
            and (
                selection_source is None
                or event.get("selection_source", "thompson") == selection_source
                or (
                    isinstance(event.get("action"), Mapping)
                    and event["action"].get(
                        "selection_source",
                        event.get("selection_source", "thompson"),
                    )
                    == selection_source
                )
            )
            and isinstance(event.get("action"), Mapping)
        ]
        counted_actions = completed_actions
        if not counted_actions and selection_source is None:
            counted_actions = [
                event.get("selected_action")
                for event in event_list
                if isinstance(event.get("selected_action"), Mapping)
            ]
        profiles = [
            list(action.get("block_granularities", []))
            for action in counted_actions
            if isinstance(action.get("block_granularities"), list)
        ]
        labels = []
        for event in event_list:
            for label in event.get("ordered_granularities", []):
                if str(label) not in labels:
                    labels.append(str(label))
        block_count = max((len(profile) for profile in profiles), default=0)
        counts = {
            f"block_{block_index}": {label: 0 for label in labels}
            for block_index in range(block_count)
        }
        for profile in profiles:
            for block_index, label in enumerate(profile):
                block_key = f"block_{block_index}"
                label_key = str(label)
                counts.setdefault(block_key, {}).setdefault(label_key, 0)
                counts[block_key][label_key] += 1
        return counts

    counts: dict[str, int] = {}
    for event in event_list:
        for label in event.get("ordered_granularities", []):
            counts.setdefault(str(label), 0)
    for event in event_list:
        if event.get("event_type") != "completed_window":
            continue
        action = event.get("action")
        if not isinstance(action, Mapping):
            continue
        if selection_source is not None and (
            event.get(
                "selection_source",
                action.get("selection_source", "thompson"),
            )
            != selection_source
        ):
            continue
        label = action.get("global_granularity")
        if label is not None:
            counts[str(label)] = counts.get(str(label), 0) + 1
    if not counts and selection_source is None:
        for event in event_list:
            action = event.get("selected_action")
            if isinstance(action, Mapping) and action.get("global_granularity") is not None:
                label = str(action["global_granularity"])
                counts[label] = counts.get(label, 0) + 1
    return counts


def _action_frequency_entropy(frequencies: Mapping[str, Any]) -> float | None:
    counts = [
        int(value)
        for value in frequencies.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    total = sum(counts)
    if total <= 0:
        return None
    probabilities = [count / total for count in counts if count > 0]
    return -sum(probability * math.log(probability) for probability in probabilities)


def controller_uncertainty_summary(covariance: Any) -> dict[str, Any]:
    matrix = _controller_json_value(covariance)
    if not isinstance(matrix, list) or not matrix:
        return {}
    diagonal = []
    for index, row in enumerate(matrix):
        if not isinstance(row, list) or index >= len(row):
            return {}
        variance = float(row[index])
        diagonal.append(max(variance, 0.0) ** 0.5)
    return {
        "posterior_stddev": diagonal,
        "mean_posterior_stddev": statistics.fmean(diagonal),
        "min_posterior_stddev": min(diagonal),
        "max_posterior_stddev": max(diagonal),
        "posterior_covariance_trace": sum(value * value for value in diagonal),
    }


def controller_effect_uncertainty_summary(
    feature_schema: Any,
    covariance: Any,
) -> dict[str, Any]:
    """Expose coefficient and additive block/label posterior uncertainty."""

    schema = _controller_json_value(feature_schema)
    matrix = _controller_json_value(covariance)
    if not isinstance(schema, Mapping) or not isinstance(matrix, list) or not matrix:
        return {}
    dimension = int(schema.get("dimension", 0))
    coefficient_names = schema.get("coefficient_names")
    if (
        dimension <= 0
        or len(matrix) != dimension
        or not isinstance(coefficient_names, list)
        or len(coefficient_names) != dimension
        or any(not isinstance(row, list) or len(row) != dimension for row in matrix)
    ):
        return {}

    coefficient_stddev = {
        str(name): max(float(matrix[index][index]), 0.0) ** 0.5
        for index, name in enumerate(coefficient_names)
    }
    summary: dict[str, Any] = {
        "coefficient_stddev": coefficient_stddev,
    }
    if schema.get("scope") != "per_block":
        return summary

    labels = list(schema.get("ordered_granularities", []))
    basis = schema.get("contrast_basis")
    block_count = int(schema.get("block_count", 0))
    contrast_count = max(0, len(labels) - 1)
    if (
        block_count <= 0
        or not isinstance(basis, list)
        or len(basis) != len(labels)
        or any(not isinstance(row, list) or len(row) != contrast_count for row in basis)
    ):
        return summary

    per_block_effects: dict[str, dict[str, float]] = {}
    for block_index in range(block_count):
        block_effects = {}
        start = 1 + block_index * contrast_count
        for label_index, label in enumerate(labels):
            effect = [0.0] * dimension
            effect[start : start + contrast_count] = [
                float(value) for value in basis[label_index]
            ]
            variance = sum(
                effect[row_index]
                * float(matrix[row_index][column_index])
                * effect[column_index]
                for row_index in range(dimension)
                for column_index in range(dimension)
            )
            block_effects[str(label)] = max(variance, 0.0) ** 0.5
        per_block_effects[f"block_{block_index}"] = block_effects
    summary["per_block_granularity_effect_stddev"] = per_block_effects
    return summary


def build_compact_controller_metric_fields(
    controller_state: Mapping[str, Any] | None,
    latest_event: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(controller_state, Mapping):
        return {}
    if controller_state.get("method_family") == "panelgrad_gradient_rms":
        refresh = controller_state.get("refresh", {})
        sampling = controller_state.get("sampling", {})
        return {
            "controller_method_family": controller_state.get("method_family"),
            "controller_method_version": controller_state.get("method_version"),
            "controller_strategy": "panelgrad",
            "controller_scope": "global",
            "controller_action": sampling.get("last_committed_action"),
            "controller_sampled_probability": sampling.get(
                "last_committed_probability"
            ),
            "controller_exposure_counts": json_artifact_value(
                sampling.get("exposure_counts", {})
            ),
            "controller_window_index": refresh.get("refresh_index"),
            "controller_window_progress": refresh.get(
                "completed_steps_since_refresh"
            ),
            "controller_boundary_step": refresh.get("last_boundary_step"),
            "controller_phase": refresh.get("phase"),
            "controller_entropy": refresh.get("entropy"),
            "controller_min_probability": refresh.get("min_probability"),
            "controller_max_probability": refresh.get("max_probability"),
            "controller_manifest_hash": controller_state.get(
                "manifest_hashes", {}
            ).get("controller_manifest_hash"),
            "final_holdout_manifest_hash": controller_state.get(
                "manifest_hashes", {}
            ).get("final_holdout_manifest_hash"),
            "controller_metrics_path": "controller_metrics.jsonl",
            "controller_summary_path": "controller_summary.json",
        }
    window = controller_state.get("window", {})
    manifests = controller_state.get("manifest_hashes", {})
    action = window.get("current_action")
    if not isinstance(action, Mapping) and isinstance(latest_event, Mapping):
        action = latest_event.get("action") or latest_event.get("selected_action")
    event = latest_event if isinstance(latest_event, Mapping) else {}
    objective = event.get("post_window_objective", event.get("controller_objective"))
    return {
        "controller_method_family": controller_state.get("method_family"),
        "controller_method_version": controller_state.get("method_version"),
        "controller_strategy": controller_state.get("strategy"),
        "controller_scope": controller_state.get("scope"),
        "controller_action": _compact_action_summary(action),
        "controller_window_index": window.get("window_index"),
        "controller_window_progress": window.get("completed_optimizer_steps"),
        "controller_boundary_step": window.get("boundary_step"),
        "controller_latest_objective": objective,
        "controller_latest_reward": event.get("reward"),
        "controller_latest_prediction_error": event.get("prediction_error"),
        "controller_manifest_hash": manifests.get("controller_manifest_hash"),
        "final_holdout_manifest_hash": manifests.get("final_holdout_manifest_hash"),
        "controller_metrics_path": controller_state.get("journal", {}).get("path"),
        "controller_summary_path": "controller_summary.json",
        "controller_reset_enabled": bool(
            controller_state.get("reset", {}).get("enabled", False)
        ),
        "controller_reset_policy": controller_state.get("reset", {})
        .get("contract", {})
        .get("policy"),
        "controller_episode_index": controller_state.get("reset", {}).get(
            "episode_index"
        ),
        "controller_episode_offset_steps": controller_state.get("reset", {}).get(
            "episode_offset_steps"
        ),
        "controller_selection_source": window.get("selection_source"),
    }


def format_controller_lifecycle_log(event: Mapping[str, Any]) -> str:
    event_type = str(event.get("event_type"))
    action = event.get("selected_action") or event.get("action")
    fields = [
        "[probabilistic-controller]",
        f"event={event_type}",
        f"method={event.get('method_family')}",
        f"scope={event.get('scope')}",
        f"boundary_step={event.get('boundary_step', event.get('boundary_step_end'))}",
        f"window_index={event.get('window_index')}",
        f"action={_compact_action_summary(action)}",
    ]
    if event_type == "initial_boundary":
        fields.append(f"objective={event.get('controller_objective')}")
    elif event_type == "completed_window":
        fields.extend(
            [
                f"pre_objective={event.get('pre_window_objective')}",
                f"post_objective={event.get('post_window_objective')}",
                f"reward={event.get('reward')}",
                f"prediction_error={event.get('prediction_error')}",
            ]
        )
    elif event_type == "terminal_incomplete":
        fields.extend(
            [
                "progress="
                f"{event.get('completed_optimizer_steps')}/"
                f"{event.get('decision_interval_steps')}",
                f"pre_objective={event.get('pre_window_objective')}",
                f"observation_emitted={event.get('observation_emitted')}",
            ]
        )
    elif event_type == "controller_failure":
        fields.extend(
            [
                f"failing_stage={event.get('failing_stage')}",
                f"error_category={event.get('error_category')}",
                f"posterior_updated={event.get('posterior_updated')}",
                f"new_action_selected={event.get('new_action_selected')}",
            ]
        )
    elif event_type.startswith("warmup_"):
        fields.extend(
            [
                f"phase={event.get('phase')}",
                f"schedule_hash={event.get('schedule_hash')}",
                "progress="
                f"{event.get('completed_optimizer_steps')}/"
                f"{event.get('action_interval_steps')}",
                f"posterior_updated={event.get('posterior_updated')}",
            ]
        )
    elif event_type in {
        "episode_initialized",
        "episode_completed",
        "posterior_reset",
        "posterior_preserved",
        "acquisition_progress",
        "acquisition_completed",
    }:
        fields.extend(
            [
                f"episode_index={event.get('episode_index')}",
                f"episode_offset_steps={event.get('episode_offset_steps')}",
                f"selection_source={event.get('selection_source')}",
                f"schedule_hash={event.get('schedule_hash')}",
            ]
        )
    fields.append(f"uncertainty={_compact_uncertainty(event.get('uncertainty_summary'))}")
    return " ".join(fields)


def _validate_controller_event(event: Mapping[str, Any]) -> None:
    event_type = event.get("event_type")
    if event_type not in CONTROLLER_EVENT_TYPES:
        raise ArtifactError(f"Unknown controller event type: {event_type!r}")
    required = {
        "schema_version",
        "event_type",
        "boundary_step",
        "window_index",
    }
    missing = sorted(field for field in required if field not in event)
    if missing:
        raise ArtifactError(
            f"Controller {event_type} event missing required fields: {missing}"
        )
    if str(event_type).startswith("panelgrad_"):
        if event.get("method_family") != "panelgrad_gradient_rms":
            raise ArtifactError("PanelGrad event method family is invalid")
        if event_type == "panelgrad_refresh_completed":
            for field in ("measurements", "q", "p", "entropy"):
                if field not in event:
                    raise ArtifactError(
                        f"PanelGrad refresh event missing {field}"
                    )
        if event_type == "panelgrad_refresh_failed" and event.get(
            "new_distribution_installed"
        ) is not False:
            raise ArtifactError(
                "PanelGrad failure must not install a new distribution"
            )
        return
    if event_type == "initial_boundary" and "reward" in event:
        raise ArtifactError("initial boundary event must not contain reward")
    if event_type == "controller_failure":
        if event.get("posterior_updated") is not False:
            raise ArtifactError("failure event posterior_updated must be false")
        if event.get("new_action_selected") is not False:
            raise ArtifactError("failure event new_action_selected must be false")
        if "posterior_mean" in event or "posterior_covariance" in event:
            raise ArtifactError("failure event must not contain posterior state")
    if event_type.startswith("warmup_"):
        if event.get("phase") != "warmup":
            raise ArtifactError("warmup event phase must be warmup")
        if event.get("posterior_updated") is not False:
            raise ArtifactError("warmup event posterior_updated must be false")
        if not event.get("schedule_hash"):
            raise ArtifactError("warmup event schedule_hash is required")
    if event_type in {"episode_initialized", "acquisition_progress", "acquisition_completed"}:
        if not event.get("schedule_hash"):
            raise ArtifactError(f"{event_type} event schedule_hash is required")
    if event_type == "posterior_reset" and event.get("policy") != "full_prior":
        raise ArtifactError("posterior reset event policy must be full_prior")
    if event_type == "posterior_preserved":
        if event.get("policy") != "acquisition_only":
            raise ArtifactError(
                "posterior preserved event policy must be acquisition_only"
            )
        if event.get("posterior_updated") is not False:
            raise ArtifactError(
                "posterior preserved event posterior_updated must be false"
            )
        if "posterior_mean" not in event or "posterior_covariance" not in event:
            raise ArtifactError(
                "posterior preserved event must contain posterior state"
            )


def _controller_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _controller_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_controller_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _controller_json_value(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def _compact_action_summary(action: Any) -> Any:
    if not isinstance(action, Mapping):
        return None
    if action.get("global_granularity") is not None:
        return str(action["global_granularity"])
    block_actions = action.get("block_granularities")
    return ",".join(str(label) for label in block_actions) if block_actions else None


def _compact_uncertainty(summary: Any) -> Any:
    if not isinstance(summary, Mapping):
        return None
    return summary.get("mean_posterior_stddev")


def _scalar_summary(values: Iterable[Any]) -> dict[str, Any]:
    finite_values = [float(value) for value in values if value is not None]
    if not finite_values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(finite_values),
        "mean": statistics.fmean(finite_values),
        "min": min(finite_values),
        "max": max(finite_values),
    }


def write_json_artifact(
    path: str | Path,
    payload: Mapping[str, Any],
    distributed_context: Any | None = None,
    artifact_io: Mapping[str, Any] | None = None,
    heartbeat_writer=None,
    artifact_state: dict[str, Any] | None = None,
) -> Path | None:
    if not _should_write_shared_artifact(distributed_context):
        return None
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    def write_attempt(_attempt: int) -> Path:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as output_file:
                temporary_path = Path(output_file.name)
                json.dump(payload, output_file, indent=2, sort_keys=True)
                output_file.write("\n")
                output_file.flush()
                os.fsync(output_file.fileno())
            os.replace(temporary_path, output_path)
            _fsync_directory(output_path.parent)
            return output_path
        except Exception:
            if temporary_path is not None:
                _unlink_best_effort(temporary_path)
            raise

    return retry_artifact_io(
        write_attempt,
        target_path=output_path,
        operation_name="json_replace",
        settings=artifact_io,
        heartbeat_writer=heartbeat_writer,
        state=artifact_state,
    )


def write_csv_artifact(
    path: str | Path,
    rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    columns: list[str],
    append: bool = False,
    distributed_context: Any | None = None,
    artifact_io: Mapping[str, Any] | None = None,
    heartbeat_writer=None,
    artifact_state: dict[str, Any] | None = None,
) -> Path | None:
    if not _should_write_shared_artifact(distributed_context):
        return None
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_rows = _normalize_rows(rows)
    for row in normalized_rows:
        _require_fields(row, columns, str(output_path))

    if append:
        original_offset = output_path.stat().st_size if output_path.exists() else 0
        payload_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            payload_buffer,
            fieldnames=columns,
            extrasaction="ignore",
        )
        if original_offset == 0:
            writer.writeheader()
        writer.writerows(normalized_rows)
        encoded_rows = payload_buffer.getvalue().encode("utf-8")

        def append_attempt(_attempt: int) -> Path:
            mode = "r+b" if output_path.exists() else "w+b"
            try:
                with output_path.open(mode) as output_file:
                    output_file.truncate(original_offset)
                    output_file.seek(original_offset)
                    output_file.write(encoded_rows)
                    output_file.flush()
                    os.fsync(output_file.fileno())
            except Exception:
                _truncate_best_effort(output_path, original_offset)
                raise
            return output_path

        operation = append_attempt
        operation_name = "metrics_csv_append" if output_path.name == "metrics.csv" else "csv_append"
    else:
        def replace_attempt(_attempt: int) -> Path:
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="",
                    dir=output_path.parent,
                    prefix=f".{output_path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as output_file:
                    temporary_path = Path(output_file.name)
                    writer = csv.DictWriter(
                        output_file,
                        fieldnames=columns,
                        extrasaction="ignore",
                    )
                    writer.writeheader()
                    writer.writerows(normalized_rows)
                    output_file.flush()
                    os.fsync(output_file.fileno())
                os.replace(temporary_path, output_path)
                _fsync_directory(output_path.parent)
                return output_path
            except Exception:
                if temporary_path is not None:
                    _unlink_best_effort(temporary_path)
                raise

        operation = replace_attempt
        operation_name = "csv_replace"

    return retry_artifact_io(
        operation,
        target_path=output_path,
        operation_name=operation_name,
        settings=artifact_io,
        heartbeat_writer=heartbeat_writer,
        state=artifact_state,
    )


def _truncate_best_effort(path: Path, offset: int) -> None:
    try:
        with path.open("r+b") as output_file:
            output_file.truncate(offset)
            output_file.flush()
            os.fsync(output_file.fileno())
    except OSError:
        return


def _unlink_best_effort(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _normalize_rows(
    rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(rows, Mapping):
        return [_with_artifact_defaults(rows)]
    return [_with_artifact_defaults(row) for row in rows]


def _with_artifact_defaults(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized_row = dict(row)
    model_shape_label = normalized_row.get(
        "model_shape_label",
        normalized_row.get("model_size_label"),
    )

    defaults = {
        "microstep": normalized_row.get("step"),
        "model_size_label": model_shape_label,
        "model_shape_label": model_shape_label,
        "sampling_mode": None,
        "resolved_run_mode": None,
        "resolved_sampling_mode": None,
        "granularity_sampling_mode": None,
        "correction_mode": None,
        "membership_correction": None,
        "granularity_pattern_summary": None,
        "granularity_mode": None,
        "granularities": None,
        "granularity_prefixes": None,
        "granularity_prefix_widths": None,
        "correction_context": None,
        "sampler_strategy": None,
        "adaptive_sampler_strategy": None,
        "adaptive_sampler_exploration_scale": None,
        "adaptive_sampler_decay_rate": None,
        "adaptive_sampler_reward_penalty_weight": None,
        "sampler_state": None,
        "adaptive_sampler_state": None,
        "adaptive_sampler_previous_loss": None,
        "adaptive_sampler_previous_pattern": None,
        "adaptive_reward_summary": None,
        "adaptive_correction_penalty_summary": None,
        "controller_method_family": None,
        "controller_method_version": None,
        "controller_strategy": None,
        "controller_scope": None,
        "controller_action": None,
        "controller_sampled_probability": None,
        "controller_exposure_counts": None,
        "controller_phase": None,
        "controller_entropy": None,
        "controller_min_probability": None,
        "controller_max_probability": None,
        "controller_window_index": None,
        "controller_window_progress": None,
        "controller_boundary_step": None,
        "controller_latest_objective": None,
        "controller_latest_reward": None,
        "controller_latest_prediction_error": None,
        "controller_manifest_hash": None,
        "final_holdout_manifest_hash": None,
        "controller_metrics_path": None,
        "controller_summary_path": None,
        "controller_reset_enabled": None,
        "controller_reset_policy": None,
        "controller_episode_index": None,
        "controller_episode_offset_steps": None,
        "controller_selection_source": None,
        "reward": None,
        "correction_penalty": None,
        "model_family_slug": None,
        "model_variant": None,
        "model_size_slug": None,
        "token_budget_slug": None,
        "output_group": None,
        "d_model": None,
        "num_layers": None,
        "num_attention_heads": None,
        "context_length": None,
        "vocab_size": None,
        "token_budget": None,
        "effective_world_size": None,
        "content_tokens_seen": normalized_row.get("tokens_seen"),
        "optimizer_window_microsteps": None,
        "committed_tokens_this_step": None,
        "evaluation_examples": None,
        "evaluation_batches": None,
        "evaluation_target_tokens": None,
        "evaluation_skipped_batches": None,
        "validation_manifest_hash": None,
        "validation_loss_aggregation": None,
        "comparison_control_signature": None,
        "ffn_parameters": None,
        "attention_parameters": None,
        "other_non_embedding_parameters": None,
        "lm_head_counting": None,
        "checkpoint_path": None,
        "checkpoint_status": None,
        "checkpoint_metric": None,
        "final_validation_loss": normalized_row.get("loss"),
        "final_validation_perplexity": normalized_row.get("perplexity"),
        "best_validation_loss": None,
        "best_validation_perplexity": None,
        "best_validation_step": None,
        "best_validation_checkpoint": None,
        "trailing_validation_mean": None,
        "trailing_validation_sample_stddev": None,
        "trailing_validation_min": None,
        "trailing_validation_max": None,
        "trailing_validation_count": None,
        "final_minus_best_loss": None,
        "effective_width": None,
        "run_status": normalized_row.get("status"),
        "omit_reason": None,
        "output_root": None,
        "output_dir": None,
        "metrics_path": None,
        "scaling_results_path": None,
        "extraction_metadata_path": None,
    }

    for key, value in defaults.items():
        normalized_row.setdefault(key, value)

    for key in (
        "granularity_pattern_summary",
        "correction_context",
        "sampler_state",
        "adaptive_sampler_state",
        "adaptive_sampler_previous_pattern",
        "adaptive_reward_summary",
        "adaptive_correction_penalty_summary",
        "granularities",
        "granularity_prefixes",
        "granularity_prefix_widths",
    ):
        if normalized_row.get(key) is not None:
            normalized_row[key] = json_artifact_value(normalized_row[key])

    return normalized_row


def _non_embedding_count(counts: Mapping[str, Any] | None):
    if counts is None:
        return None
    return counts.get("non_embedding_parameters")


def _model_shape_label(run: Mapping[str, Any]) -> Any:
    return run.get("model_shape_label", run.get("model_size_label"))


def json_artifact_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _granularity_pattern_provenance(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    model = config.get("model", {})
    run = config.get("run", {})
    training = config.get("training", {})
    if not isinstance(model, Mapping):
        model = {}
    if not isinstance(run, Mapping):
        run = {}
    if not isinstance(training, Mapping):
        training = {}

    provenance = model.get("granularity_pattern_provenance")
    if isinstance(provenance, Mapping):
        return dict(provenance)

    granularity_sampling_mode = model.get("granularity_sampling_mode")
    if granularity_sampling_mode is None:
        granularity_sampling_mode = "global"
    resolved_run_mode = resolve_sampling_mode_from_config_sections(run, training)

    provenance = {
        "pattern_type": (
            "all_granularities"
            if resolved_run_mode == "nested-all"
            else (
                "per_block"
                if granularity_sampling_mode == "per_block"
                else "single"
            )
        ),
        "scope": "model",
        "source": "model.granularity_sampling_mode",
        "requested_alias": model.get("requested_granularity_sampling_alias"),
        "layer_count": model.get("num_layers"),
        "available_granularities": list(model.get("granularities", []))
        if isinstance(model.get("granularities"), list)
        else [],
    }
    if model.get("requested_granularity_sampling_alias") is not None or run.get(
        "granularity"
    ) is not None:
        provenance["active_granularity"] = run.get("granularity")
    return provenance


def _granularity_pattern_summary(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        model = {}
    stored_summary = model.get("granularity_pattern_summary")
    if isinstance(stored_summary, Mapping):
        return dict(stored_summary)

    return summarize_granularity_pattern_from_config(config)


def _correction_context_summary(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    model = config.get("model", {})
    if isinstance(model, Mapping):
        stored_context = model.get("correction_context")
        if isinstance(stored_context, Mapping):
            return dict(stored_context)

    return summarize_correction_context_from_config(config)


def _summary_granularities(summary: Mapping[str, Any]) -> list[str]:
    if summary.get("granularities"):
        return [str(granularity) for granularity in summary["granularities"]]
    if summary.get("granularity"):
        return [str(summary["granularity"])]
    parameter_counts = summary.get("parameter_counts_by_granularity") or {}
    return [str(granularity) for granularity in parameter_counts]


def _summary_parameter_counts(
    summary: Mapping[str, Any],
    granularity: str,
) -> Mapping[str, Any]:
    counts_by_granularity = summary.get("parameter_counts_by_granularity") or {}
    if granularity in counts_by_granularity:
        return counts_by_granularity[granularity]
    if summary.get("granularity") == granularity:
        return summary.get("parameter_counts") or {}
    return {}


def _summary_checkpoint_status(summary: Mapping[str, Any]) -> str:
    if summary.get("checkpoint_status"):
        return str(summary["checkpoint_status"])
    if summary.get("best_checkpoint_path"):
        return "best_eval"
    if summary.get("final_checkpoint_path"):
        return "final"
    if summary.get("checkpoint_path"):
        return "available"
    return "unavailable"


def _summary_checkpoint_path(summary: Mapping[str, Any]) -> Any:
    return (
        summary.get("checkpoint_path")
        or summary.get("best_checkpoint_path")
        or summary.get("final_checkpoint_path")
    )


def _require_fields(
    row: Mapping[str, Any],
    required_fields: list[str],
    artifact_name: str,
) -> None:
    missing_fields = [field_name for field_name in required_fields if field_name not in row]
    if missing_fields:
        raise ArtifactError(f"{artifact_name} missing fields: {missing_fields}")


def _best_validation_metric_row(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, str | None, float | None]:
    validation_rows = [row for row in rows if row.get("split") == "validation"]
    if not validation_rows:
        return None, None, None

    for field_name, metric_name in [
        ("loss", "validation_loss"),
        ("perplexity", "validation_perplexity"),
    ]:
        candidates = []
        for row in validation_rows:
            metric_value = _float_value(row.get(field_name))
            if metric_value is not None:
                candidates.append((metric_value, _int_value(row.get("step")), row))
        if candidates:
            metric_value, _, row = min(candidates, key=lambda candidate: candidate[:2])
            return row, metric_name, metric_value

    return None, None, None


def _int_value(value: Any) -> int:
    if value in (None, ""):
        return -1
    return int(value)


def _float_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _should_write_shared_artifact(distributed_context: Any | None) -> bool:
    if distributed_context is None:
        return True

    from src.training.distributed import should_write_shared_artifact

    return should_write_shared_artifact(distributed_context)
