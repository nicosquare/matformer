import csv
import errno
from pathlib import Path

import pytest
import torch

import src.training.checkpointing as checkpointing
import src.utils.artifact_io as artifact_io
import src.utils.metrics as metrics
from src.training.monitoring import NoopHeartbeatWriter
from src.utils.config import resolve_run_config
from src.utils.heartbeats import heartbeat_stage


def _metric_row(step, split="train", granularity="s"):
    return {
        "run_id": "debug-nested-001",
        "step": step,
        "split": split,
        "model_family": "nested",
        "granularity": granularity,
        "loss": 1.0 / max(step, 1),
        "perplexity": 2.0,
        "tokens_seen": step * 64,
        "wall_clock_seconds": float(step),
        "tokens_per_second": 64.0,
        "peak_memory_bytes": 0,
    }


def _checkpoint_fields():
    return {
        "checkpoint_status": "latest",
        "checkpoint_metric": None,
        "checkpoint_metric_value": None,
        "checkpoint_selection_step": None,
        "checkpoint_unavailable_reason": None,
    }


def _optimizer_and_scheduler(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return optimizer, scheduler


def _fast_artifact_io(config, *, max_attempts=3):
    config["outputs"]["artifact_io"].update(
        {
            "max_attempts": max_attempts,
            "initial_backoff_seconds": 0.0,
            "max_backoff_seconds": 0.0,
            "jitter_fraction": 0.0,
            "checkpoint_staging": "direct",
        }
    )
    return config


def test_artifact_io_defaults_are_resolved(tmp_path):
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=tmp_path / "debug-nested-001",
    )

    assert config["outputs"]["artifact_io"] == artifact_io.DEFAULT_ARTIFACT_IO


def test_retry_helper_finds_transient_errno_in_chained_torch_error(tmp_path):
    attempts = []
    sleeps = []
    state = {}
    events = []

    class RecordingWriter:
        def emit(self, event_type, stage, **fields):
            events.append((event_type, stage, fields))

    def operation(attempt):
        attempts.append(attempt)
        if attempt < 3:
            try:
                raise OSError(errno.EFAULT, "simulated Lustre fault")
            except OSError as cause:
                raise RuntimeError("unexpected pos 123 vs 456") from cause
        return "recovered"

    result = artifact_io.retry_artifact_io(
        operation,
        target_path=tmp_path / "checkpoint.pt",
        operation_name="checkpoint_serialize",
        settings={
            "max_attempts": 3,
            "initial_backoff_seconds": 0.01,
            "max_backoff_seconds": 1.0,
            "jitter_fraction": 0.0,
        },
        state=state,
        heartbeat_writer=RecordingWriter(),
        sleep_fn=sleeps.append,
    )

    assert result == "recovered"
    assert attempts == [1, 2, 3]
    assert sleeps == [0.01, 0.02]
    assert state["artifact_retry_count"] == 2
    assert state["artifact_last_errno"] == errno.EFAULT
    assert [event[0] for event in events] == [
        "artifact_retry",
        "artifact_retry",
        "stage_complete",
    ]


@pytest.mark.parametrize(
    "error_number",
    [errno.ENOSPC, getattr(errno, "EDQUOT", 122), errno.EACCES, errno.EROFS],
)
def test_permanent_artifact_errors_are_not_retried(tmp_path, error_number):
    attempts = []

    def operation(attempt):
        attempts.append(attempt)
        raise OSError(error_number, "permanent failure")

    with pytest.raises(OSError) as raised:
        artifact_io.retry_artifact_io(
            operation,
            target_path=tmp_path / "artifact.json",
            operation_name="json_replace",
            settings={"max_attempts": 5},
            sleep_fn=lambda _delay: None,
        )

    assert attempts == [1]
    assert raised.value.errno == error_number
    assert raised.value.filename == str(tmp_path / "artifact.json")


def test_csv_append_fsync_failure_rolls_back_without_duplicates(tmp_path, monkeypatch):
    path = tmp_path / "metrics.csv"
    metrics.write_csv_artifact(path, [_metric_row(1)], metrics.METRICS_COLUMNS)
    real_fsync = metrics.os.fsync
    calls = 0

    def fail_first_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(errno.EFAULT, "transient append fsync")
        return real_fsync(descriptor)

    monkeypatch.setattr(metrics.os, "fsync", fail_first_fsync)
    metrics.write_csv_artifact(
        path,
        [_metric_row(2)],
        metrics.METRICS_COLUMNS,
        append=True,
        artifact_io={
            "max_attempts": 2,
            "initial_backoff_seconds": 0,
            "max_backoff_seconds": 0,
            "jitter_fraction": 0,
        },
    )

    assert [int(row["step"]) for row in metrics.read_metrics_csv(path)] == [1, 2]


def test_atomic_json_transient_replace_failure_recovers(tmp_path, monkeypatch):
    path = tmp_path / "summary.json"
    real_replace = metrics.os.replace
    attempts = 0

    def transient_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(errno.EFAULT, "transient rename fault")
        return real_replace(source, destination)

    monkeypatch.setattr(metrics.os, "replace", transient_replace)
    metrics.write_json_artifact(
        path,
        {"status": "completed"},
        artifact_io={
            "max_attempts": 2,
            "initial_backoff_seconds": 0,
            "max_backoff_seconds": 0,
            "jitter_fraction": 0,
        },
    )

    assert attempts == 2
    assert path.read_text(encoding="utf-8").strip() == '{\n  "status": "completed"\n}'


def test_metrics_journal_spools_then_drains_rows_in_order(tmp_path, monkeypatch):
    output_dir = tmp_path / "debug-nested-001"
    state = {}
    journal = metrics.MetricsJournal(
        output_dir,
        flush_interval_steps=1,
        artifact_io_config={
            "max_attempts": 1,
            "metrics_pending_row_limit": 10,
        },
        artifact_state=state,
    )
    real_write = metrics.write_metrics_csv
    remote_available = False

    def conditional_write(*args, **kwargs):
        if kwargs.get("append") and not remote_available:
            raise OSError(errno.EIO, "remote unavailable", str(journal.path))
        return real_write(*args, **kwargs)

    monkeypatch.setattr(metrics, "write_metrics_csv", conditional_write)
    journal.append(_metric_row(1))
    journal.append(_metric_row(2))

    assert journal.spool_path.exists()
    assert state["deferred_metric_rows"] == 2
    assert [int(row["step"]) for row in journal._buffer] == [1, 2]

    remote_available = True
    journal.flush()

    assert not journal.spool_path.exists()
    assert state["deferred_metric_rows"] == 0
    assert [int(row["step"]) for row in metrics.read_metrics_csv(journal.path)] == [1, 2]


def test_checkpoint_serialization_retries_transient_efault(tmp_path, monkeypatch):
    config = _fast_artifact_io(
        resolve_run_config(
            "configs/debug_matrix.yaml",
            run_id="debug-nested-001",
            output_dir=tmp_path / "debug-nested-001",
        )
    )
    model = torch.nn.Linear(1, 1)
    optimizer, scheduler = _optimizer_and_scheduler(model)
    run_state = checkpointing.build_initial_continuation_state(config)
    run_state.update({"step": 1, "last_completed_step": 1})
    real_save = checkpointing.torch.save
    attempts = 0

    def transient_save(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(errno.EFAULT, "transient serialization fault")
        return real_save(*args, **kwargs)

    monkeypatch.setattr(checkpointing.torch, "save", transient_save)
    path = Path(config["run"]["output_dir"]) / "checkpoints" / "latest.pt"
    checkpointing.save_model_checkpoint(
        config,
        model,
        optimizer,
        scheduler,
        path,
        _checkpoint_fields(),
        run_state,
    )

    assert attempts == 2
    assert torch.load(path, map_location="cpu")["step"] == 1
    assert run_state["artifact_retry_count"] == 1


def test_staged_checkpoint_transfer_retries_with_fresh_target_temp(
    tmp_path,
    monkeypatch,
):
    local_dir = tmp_path / "slurm-local"
    local_dir.mkdir()
    monkeypatch.setenv("SLURM_TMPDIR", str(local_dir))
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=tmp_path / "debug-nested-001",
    )
    config["outputs"]["artifact_io"].update(
        {
            "max_attempts": 2,
            "initial_backoff_seconds": 0,
            "max_backoff_seconds": 0,
            "jitter_fraction": 0,
            "checkpoint_staging": "auto",
        }
    )
    model = torch.nn.Linear(1, 1)
    optimizer, scheduler = _optimizer_and_scheduler(model)
    run_state = checkpointing.build_initial_continuation_state(config)
    run_state.update({"step": 1, "last_completed_step": 1})
    real_copy = checkpointing.shutil.copyfileobj
    target_temp_names = []

    def transient_copy(source, destination, *args, **kwargs):
        target_temp_names.append(destination.name)
        if len(target_temp_names) == 1:
            raise OSError(errno.EFAULT, "transient transfer fault")
        return real_copy(source, destination, *args, **kwargs)

    monkeypatch.setattr(checkpointing.shutil, "copyfileobj", transient_copy)
    path = Path(config["run"]["output_dir"]) / "checkpoints" / "latest.pt"
    checkpointing.save_model_checkpoint(
        config,
        model,
        optimizer,
        scheduler,
        path,
        _checkpoint_fields(),
        run_state,
    )

    assert len(target_temp_names) == 2
    assert target_temp_names[0] != target_temp_names[1]
    assert run_state["checkpoint_staging_mode"] == "slurm_tmpdir"
    assert torch.load(path, map_location="cpu")["step"] == 1


def test_checkpoint_install_rename_retries_transient_efault(tmp_path, monkeypatch):
    config = _fast_artifact_io(
        resolve_run_config(
            "configs/debug_matrix.yaml",
            run_id="debug-nested-001",
            output_dir=tmp_path / "debug-nested-001",
        ),
        max_attempts=2,
    )
    model = torch.nn.Linear(1, 1)
    optimizer, scheduler = _optimizer_and_scheduler(model)
    run_state = checkpointing.build_initial_continuation_state(config)
    run_state.update({"step": 1, "last_completed_step": 1})
    path = Path(config["run"]["output_dir"]) / "checkpoints" / "latest.pt"
    real_replace = checkpointing.os.replace
    attempts = 0

    def transient_install(source, destination):
        nonlocal attempts
        if Path(destination) == path:
            attempts += 1
            if attempts == 1:
                raise OSError(errno.EFAULT, "transient install fault")
        return real_replace(source, destination)

    monkeypatch.setattr(checkpointing.os, "replace", transient_install)
    checkpointing.save_model_checkpoint(
        config,
        model,
        optimizer,
        scheduler,
        path,
        _checkpoint_fields(),
        run_state,
    )

    assert attempts == 2
    assert torch.load(path, map_location="cpu")["step"] == 1


def test_heartbeat_stage_emits_failure_without_completion():
    events = []

    class RecordingWriter:
        def stage_start(self, stage, **fields):
            events.append(("start", stage))

        def stage_complete(self, stage, **fields):
            events.append(("complete", stage))

        def stage_failed(self, stage, **fields):
            events.append(("failed", stage))

    with pytest.raises(RuntimeError):
        with heartbeat_stage(RecordingWriter(), "checkpointing"):
            raise RuntimeError("boom")

    assert events == [("start", "checkpointing"), ("failed", "checkpointing")]


def test_metrics_journal_repairs_malformed_and_ahead_of_checkpoint_tail(tmp_path):
    output_dir = tmp_path / "debug-nested-001"
    metrics.write_metrics_csv(
        output_dir,
        [_metric_row(1), _metric_row(2), _metric_row(3)],
    )
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("a", encoding="utf-8", newline="") as output_file:
        output_file.write("malformed,tail\n")

    journal = metrics.MetricsJournal(
        output_dir,
        flush_interval_steps=100,
        checkpoint_step=2,
    )

    assert [int(row["step"]) for row in journal.read_all()] == [1, 2]
    with metrics_path.open("r", encoding="utf-8", newline="") as metrics_file:
        assert [int(row["step"]) for row in csv.DictReader(metrics_file)] == [1, 2]


def test_metrics_journal_completion_summary_stays_bounded_and_checkpointable(tmp_path):
    output_dir = tmp_path / "bounded-metrics"
    run_state = {}
    journal = metrics.MetricsJournal(
        output_dir,
        flush_interval_steps=10,
        checkpoint_step=0,
        artifact_state=run_state,
    )
    journal._retained_row_limit = 20
    for step in range(1, 101):
        journal.append([_metric_row(step)], force=step % 10 == 0)
    journal.append([_metric_row(100, split="validation")], force=True)

    summary_rows = journal.summary_rows()
    assert len(summary_rows) <= 6
    assert journal.training_outcome()["steps_completed"] == 100
    assert run_state["metrics_accumulator_state"]["training_row_count"] == 100
    assert run_state["metrics_accumulator_state"]["validation_row_count"] == 1
    assert sum(1 for _ in journal.iter_rows()) == 101


def test_scaling_rows_never_treat_per_block_pattern_as_uniform_granularity(tmp_path):
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=tmp_path / "debug-nested-001",
        overrides=["evaluation.final_validation=false"],
    )
    pattern_row = _metric_row(
        1,
        split="train",
        granularity="s,xl,s,m",
    )
    pattern_row["granularity_sampling_mode"] = "per_block"

    assert metrics.build_scaling_result_rows(config, [pattern_row], {}) == []


def test_metrics_journal_flushes_on_configured_step_interval(tmp_path):
    output_dir = tmp_path / "debug-nested-001"
    journal = metrics.MetricsJournal(
        output_dir,
        flush_interval_steps=2,
        checkpoint_step=0,
    )

    journal.append(_metric_row(1))
    assert metrics.read_metrics_csv(output_dir / "metrics.csv") == []
    journal.append(_metric_row(2))

    assert [
        int(row["step"])
        for row in metrics.read_metrics_csv(output_dir / "metrics.csv")
    ] == [1, 2]


def test_atomic_json_write_reports_original_path_and_errno(tmp_path, monkeypatch):
    output_path = tmp_path / "summary.json"

    def fail_replace(source, destination):
        raise OSError(errno.ENOSPC, "simulated full device")

    monkeypatch.setattr(metrics.os, "replace", fail_replace)

    with pytest.raises(OSError) as raised:
        metrics.write_json_artifact(output_path, {"status": "completed"})

    assert raised.value.errno == errno.ENOSPC
    assert raised.value.filename == str(output_path)
    assert str(output_path) in str(raised.value)


def test_latest_checkpoint_rotation_and_corruption_fallback(tmp_path):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "run.continuation.enabled=true",
            "outputs.save_checkpoints=true",
        ],
    )
    model = torch.nn.Linear(1, 1)
    optimizer, scheduler = _optimizer_and_scheduler(model)
    latest_path = output_dir / "checkpoints" / "latest.pt"
    run_state = checkpointing.build_initial_continuation_state(config)

    run_state.update({"step": 1, "last_completed_step": 1})
    checkpointing.save_model_checkpoint(
        config,
        model,
        optimizer,
        scheduler,
        latest_path,
        _checkpoint_fields(),
        run_state,
    )
    first_weight = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(3.0)
    run_state.update({"step": 2, "last_completed_step": 2})
    checkpointing.save_model_checkpoint(
        config,
        model,
        optimizer,
        scheduler,
        latest_path,
        _checkpoint_fields(),
        run_state,
    )
    previous_path = latest_path.with_name("latest.prev.pt")
    assert previous_path.exists()
    latest_path.write_bytes(b"corrupt checkpoint")

    restored_model = torch.nn.Linear(1, 1)
    restored_optimizer, restored_scheduler = _optimizer_and_scheduler(restored_model)
    restored_state = checkpointing.load_run_continuation_state(
        config,
        restored_model,
        restored_optimizer,
        restored_scheduler,
    )

    assert restored_state["last_completed_step"] == 1
    assert restored_state["continuation_source_checkpoint_path"] == str(
        previous_path
    )
    assert restored_state["latest_checkpoint_path"] == str(latest_path)
    torch.testing.assert_close(restored_model.weight, first_weight)


def test_checkpoint_install_failure_keeps_previous_checkpoint_loadable(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=["run.continuation.enabled=true"],
    )
    model = torch.nn.Linear(1, 1)
    optimizer, scheduler = _optimizer_and_scheduler(model)
    latest_path = output_dir / "checkpoints" / "latest.pt"
    run_state = checkpointing.build_initial_continuation_state(config)
    run_state.update({"step": 1, "last_completed_step": 1})
    checkpointing.save_model_checkpoint(
        config,
        model,
        optimizer,
        scheduler,
        latest_path,
        _checkpoint_fields(),
        run_state,
    )

    real_replace = checkpointing.os.replace

    def fail_new_latest(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == latest_path and source_path.suffix == ".tmp":
            raise OSError(errno.ENOSPC, "simulated full device")
        return real_replace(source, destination)

    monkeypatch.setattr(checkpointing.os, "replace", fail_new_latest)
    run_state.update({"step": 2, "last_completed_step": 2})

    with pytest.raises(OSError) as raised:
        checkpointing.save_model_checkpoint(
            config,
            model,
            optimizer,
            scheduler,
            latest_path,
            _checkpoint_fields(),
            run_state,
        )

    previous_path = latest_path.with_name("latest.prev.pt")
    assert raised.value.errno == errno.ENOSPC
    assert raised.value.filename == str(latest_path)
    assert torch.load(previous_path, map_location="cpu")["step"] == 1


def test_best_checkpoint_is_installed_before_retention_prunes_predecessor(tmp_path):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "outputs.save_checkpoints=true",
            "outputs.best_eval_retention_count=1",
        ],
    )
    model = torch.nn.Linear(1, 1)
    checkpoint_state = {}
    run_state = checkpointing.build_initial_continuation_state(config)
    heartbeat = NoopHeartbeatWriter()

    run_state.update({"step": 1, "last_completed_step": 1})
    checkpointing.maybe_write_best_eval_checkpoint(
        config,
        model,
        [{"granularity": "s", "loss": 2.0, "perplexity": 7.4}],
        1,
        heartbeat,
        checkpoint_state,
        run_state,
    )
    run_state.update({"step": 2, "last_completed_step": 2})
    checkpointing.maybe_write_best_eval_checkpoint(
        config,
        model,
        [{"granularity": "s", "loss": 1.0, "perplexity": 2.7}],
        2,
        heartbeat,
        checkpoint_state,
        run_state,
    )

    checkpoint_paths = list((output_dir / "checkpoints").glob("best_eval_step_*.pt"))
    assert [path.name for path in checkpoint_paths] == ["best_eval_step_2.pt"]
    assert run_state["best_checkpoint_path"].endswith("best_eval_step_2.pt")
    assert run_state["checkpoint_metric_value"] == 1.0


def test_best_checkpoint_metadata_survives_latest_checkpoint_resume(tmp_path):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "run.continuation.enabled=true",
            "outputs.save_checkpoints=true",
        ],
    )
    model = torch.nn.Linear(1, 1)
    optimizer, scheduler = _optimizer_and_scheduler(model)
    checkpoint_state = {}
    run_state = checkpointing.build_initial_continuation_state(config)
    run_state.update({"step": 4, "last_completed_step": 4})
    checkpointing.maybe_write_best_eval_checkpoint(
        config,
        model,
        [{"granularity": "m", "loss": 0.75, "perplexity": 2.1}],
        4,
        NoopHeartbeatWriter(),
        checkpoint_state,
        run_state,
    )
    latest_path = output_dir / "checkpoints" / "latest.pt"
    checkpointing.save_model_checkpoint(
        config,
        model,
        optimizer,
        scheduler,
        latest_path,
        _checkpoint_fields(),
        run_state,
    )

    restored_model = torch.nn.Linear(1, 1)
    restored_optimizer, restored_scheduler = _optimizer_and_scheduler(restored_model)
    restored = checkpointing.load_run_continuation_state(
        config,
        restored_model,
        restored_optimizer,
        restored_scheduler,
    )

    assert restored["best_checkpoint_path"].endswith("best_eval_step_4.pt")
    assert restored["checkpoint_metric"] == "validation_loss"
    assert restored["checkpoint_metric_value"] == 0.75
    assert restored["checkpoint_selection_step"] == 4


def test_exhausted_periodic_checkpoint_defers_only_with_durable_fallback(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "debug-nested-001"
    config = _fast_artifact_io(
        resolve_run_config(
            "configs/debug_matrix.yaml",
            run_id="debug-nested-001",
            output_dir=output_dir,
            overrides=[
                "run.continuation.enabled=true",
                "run.continuation.latest_checkpoint_save_interval_steps=1",
            ],
        ),
        max_attempts=1,
    )
    model = torch.nn.Linear(1, 1)
    optimizer, scheduler = _optimizer_and_scheduler(model)
    run_state = checkpointing.build_initial_continuation_state(config)
    run_state.update({"step": 1, "last_completed_step": 1})
    checkpointing.maybe_write_latest_checkpoint(
        config,
        model,
        optimizer,
        scheduler,
        NoopHeartbeatWriter(),
        run_state,
        reason="step",
        step=1,
    )

    latest_path = output_dir / "checkpoints" / "latest.pt"
    real_replace = checkpointing.os.replace

    def fail_install(source, destination):
        if Path(destination) == latest_path and Path(source).suffix == ".tmp":
            raise OSError(errno.EIO, "persistent Lustre outage")
        return real_replace(source, destination)

    monkeypatch.setattr(checkpointing.os, "replace", fail_install)
    run_state.update({"step": 2, "last_completed_step": 2})
    checkpointing.maybe_write_latest_checkpoint(
        config,
        model,
        optimizer,
        scheduler,
        NoopHeartbeatWriter(),
        run_state,
        reason="step",
        step=2,
    )

    previous_path = latest_path.with_name("latest.prev.pt")
    assert previous_path.exists()
    assert torch.load(previous_path, map_location="cpu")["step"] == 1
    assert run_state["latest_checkpoint_step"] == 1
    assert run_state["last_durable_checkpoint_step"] == 1
    assert run_state["pending_latest_checkpoint"] is True
    assert run_state["skipped_periodic_checkpoints"] == 1


def test_first_and_completion_checkpoint_failures_are_strict(tmp_path, monkeypatch):
    output_dir = tmp_path / "debug-nested-001"
    config = _fast_artifact_io(
        resolve_run_config(
            "configs/debug_matrix.yaml",
            run_id="debug-nested-001",
            output_dir=output_dir,
            overrides=[
                "run.continuation.enabled=true",
                "run.continuation.latest_checkpoint_save_interval_steps=1",
            ],
        ),
        max_attempts=1,
    )
    model = torch.nn.Linear(1, 1)
    optimizer, scheduler = _optimizer_and_scheduler(model)
    run_state = checkpointing.build_initial_continuation_state(config)
    run_state.update({"step": 1, "last_completed_step": 1})

    def fail_save(*args, **kwargs):
        raise OSError(errno.EIO, "persistent serialization failure")

    monkeypatch.setattr(checkpointing.torch, "save", fail_save)
    with pytest.raises(OSError):
        checkpointing.maybe_write_latest_checkpoint(
            config,
            model,
            optimizer,
            scheduler,
            NoopHeartbeatWriter(),
            run_state,
            reason="step",
            step=1,
        )
    with pytest.raises(OSError):
        checkpointing.maybe_write_latest_checkpoint(
            config,
            model,
            optimizer,
            scheduler,
            NoopHeartbeatWriter(),
            run_state,
            reason="completion",
            step=1,
            force=True,
        )


def test_failed_best_checkpoint_keeps_metadata_and_predecessor(tmp_path, monkeypatch):
    output_dir = tmp_path / "debug-nested-001"
    config = _fast_artifact_io(
        resolve_run_config(
            "configs/debug_matrix.yaml",
            run_id="debug-nested-001",
            output_dir=output_dir,
            overrides=[
                "outputs.save_checkpoints=true",
                "outputs.best_eval_retention_count=1",
            ],
        ),
        max_attempts=1,
    )
    model = torch.nn.Linear(1, 1)
    checkpoint_state = {}
    run_state = checkpointing.build_initial_continuation_state(config)
    run_state.update({"step": 1, "last_completed_step": 1})
    checkpointing.maybe_write_best_eval_checkpoint(
        config,
        model,
        [{"granularity": "s", "loss": 2.0, "perplexity": 7.4}],
        1,
        NoopHeartbeatWriter(),
        checkpoint_state,
        run_state,
    )
    predecessor = output_dir / "checkpoints" / "best_eval_step_1.pt"
    original_metadata = dict(checkpoint_state)
    real_replace = checkpointing.os.replace

    def fail_new_best(source, destination):
        if Path(destination).name == "best_eval_step_2.pt":
            raise OSError(errno.EIO, "best checkpoint install failed")
        return real_replace(source, destination)

    monkeypatch.setattr(checkpointing.os, "replace", fail_new_best)
    run_state.update({"step": 2, "last_completed_step": 2})
    checkpointing.maybe_write_best_eval_checkpoint(
        config,
        model,
        [{"granularity": "s", "loss": 1.0, "perplexity": 2.7}],
        2,
        NoopHeartbeatWriter(),
        checkpoint_state,
        run_state,
    )

    assert predecessor.exists()
    assert checkpoint_state == original_metadata
    assert not (output_dir / "checkpoints" / "best_eval_step_2.pt").exists()
    assert run_state["pending_best_checkpoint"]["step"] == 2
