"""Small heartbeat helpers for long-running experiment jobs."""

from __future__ import annotations

from contextlib import contextmanager
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


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


def build_heartbeat_cadence(config: dict[str, Any]) -> "HeartbeatCadence":
    training = config["training"]
    return HeartbeatCadence(
        step_interval=training.get("heartbeat_step_interval", 10),
        time_interval_seconds=training.get("heartbeat_time_interval_seconds", 60.0),
    )


@contextmanager
def heartbeat_stage(heartbeat_writer, stage: str, **fields: Any):
    heartbeat_writer.stage_start(stage, **fields)
    try:
        yield
    finally:
        heartbeat_writer.stage_complete(stage, **fields)


def estimate_eta_seconds(
    config: dict[str, Any],
    tokens_seen: int,
    tokens_per_second: float | None,
) -> float | None:
    if tokens_per_second is None or tokens_per_second <= 0:
        return None
    remaining_tokens = max(config["training"]["token_budget"] - tokens_seen, 0)
    return remaining_tokens / tokens_per_second


def maybe_emit_training_heartbeat(
    heartbeat_writer,
    heartbeat_cadence: "HeartbeatCadence",
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


class HeartbeatCadence:
    """Tracks step- or time-based heartbeat emission."""

    def __init__(
        self,
        step_interval: int | None = 10,
        time_interval_seconds: float | None = 60.0,
    ):
        self.step_interval = _positive_int_or_none(step_interval, "step_interval")
        self.time_interval_seconds = _positive_float_or_none(
            time_interval_seconds,
            "time_interval_seconds",
        )
        self.last_step: int | None = None
        self.last_time: float | None = None

    def should_emit(self, step: int | None, now: float | None = None) -> bool:
        if now is None:
            now = time.time()

        if self.last_time is None:
            return True

        if (
            step is not None
            and self.step_interval is not None
            and self.last_step is not None
            and step - self.last_step >= self.step_interval
        ):
            return True

        if (
            self.time_interval_seconds is not None
            and now - self.last_time >= self.time_interval_seconds
        ):
            return True

        return False

    def mark_emitted(self, step: int | None, now: float | None = None) -> None:
        if now is None:
            now = time.time()
        if step is not None:
            self.last_step = step
        self.last_time = now


class HeartbeatWriter:
    """Writes heartbeat events to stdout and run-local JSONL."""

    def __init__(
        self,
        output_dir: str | Path,
        run_id: str,
        rank: int = 0,
        world_size: int = 1,
        stdout: TextIO | None = None,
        time_fn=time.time,
        filename: str = "heartbeats.jsonl",
    ):
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.stdout = stdout if stdout is not None else sys.stdout
        self.time_fn = time_fn
        self.path = self.output_dir / filename
        self.start_time = float(self.time_fn())

    def emit(
        self,
        event_type: str,
        stage: str,
        step: int | None = None,
        derived_max_steps: int | None = None,
        tokens_seen: int | None = None,
        content_tokens_seen: int | None = None,
        token_budget: int | None = None,
        latest_loss: float | None = None,
        tokens_per_second: float | None = None,
        peak_gpu_memory_bytes: int | None = None,
        eta_seconds: float | None = None,
        extra_fields: dict[str, Any] | None = None,
        **fields: Any,
    ) -> Path:
        now = float(self.time_fn())
        event = {
            "event_type": event_type,
            "run_id": self.run_id,
            "stage": stage,
            "rank": self.rank,
            "world_size": self.world_size,
            "timestamp": _utc_timestamp(now),
            "elapsed_seconds": now - self.start_time,
            "step": step,
            "derived_max_steps": derived_max_steps,
            "tokens_seen": tokens_seen,
            "content_tokens_seen": content_tokens_seen,
            "token_budget": token_budget,
            "latest_loss": latest_loss,
            "tokens_per_second": tokens_per_second,
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
            "eta_seconds": eta_seconds,
        }
        if extra_fields:
            event.update(extra_fields)
        if fields:
            event.update(fields)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as heartbeat_file:
            heartbeat_file.write(json.dumps(event, sort_keys=True))
            heartbeat_file.write("\n")

        print(self.format_stdout_line(event), file=self.stdout, flush=True)
        return self.path

    def stage_start(self, stage: str, **fields: Any) -> Path:
        return self.emit("stage_start", stage, **fields)

    def stage_complete(self, stage: str, **fields: Any) -> Path:
        return self.emit("stage_complete", stage, **fields)

    def heartbeat(self, stage: str, **fields: Any) -> Path:
        return self.emit("heartbeat", stage, **fields)

    def format_stdout_line(self, event: dict[str, Any]) -> str:
        parts = [
            event["event_type"],
            f"stage={event['stage']}",
            f"rank={event['rank']}/{event['world_size']}",
            f"elapsed={event['elapsed_seconds']:.1f}s",
        ]
        if event.get("message"):
            parts.append(f"message={event['message']}")
        if event.get("step") is not None:
            if event.get("derived_max_steps") is not None:
                parts.append(f"step={event['step']}/{event['derived_max_steps']}")
            else:
                parts.append(f"step={event['step']}")
        if event.get("tokens_seen") is not None:
            if event.get("token_budget") is not None:
                parts.append(f"tokens={event['tokens_seen']}/{event['token_budget']}")
            else:
                parts.append(f"tokens={event['tokens_seen']}")
        if event.get("content_tokens_seen") is not None:
            parts.append(f"content_tokens={event['content_tokens_seen']}")
        if event.get("latest_loss") is not None:
            parts.append(f"loss={event['latest_loss']}")
        if event.get("tokens_per_second") is not None:
            parts.append(f"tok/s={event['tokens_per_second']}")
        if event.get("eta_seconds") is not None:
            parts.append(f"eta={event['eta_seconds']}s")
        if event.get("continuation_status") is not None:
            parts.append(f"continuation_status={event['continuation_status']}")
        if event.get("latest_checkpoint_path") is not None:
            parts.append(f"latest_checkpoint_path={event['latest_checkpoint_path']}")
        if event.get("last_completed_step") is not None:
            parts.append(f"last_completed_step={event['last_completed_step']}")
        if event.get("resume_count") is not None:
            parts.append(f"resume_count={event['resume_count']}")
        return " ".join(parts)


def _utc_timestamp(seconds: float) -> str:
    return (
        datetime.fromtimestamp(seconds, timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _positive_int_or_none(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive or None")
    return parsed


def _positive_float_or_none(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive or None")
    return parsed
