"""Bounded retry and reporting helpers for experiment artifact I/O."""

from __future__ import annotations

import errno
import random
import time
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

from src.utils.reproducibility import dedicated_random


T = TypeVar("T")

DEFAULT_ARTIFACT_IO = {
    "max_attempts": 5,
    "initial_backoff_seconds": 1.0,
    "max_backoff_seconds": 30.0,
    "jitter_fraction": 0.2,
    "checkpoint_staging": "auto",
    "periodic_checkpoint_failure_policy": "continue_if_previous",
    "metrics_pending_row_limit": 10_000,
}

TRANSIENT_ERRNOS = {
    errno.EFAULT,
    errno.EIO,
    errno.ESTALE,
    errno.ETIMEDOUT,
    errno.EAGAIN,
    errno.EINTR,
    errno.ECONNABORTED,
    errno.ECONNRESET,
    errno.ENETRESET,
    errno.ENETDOWN,
    errno.ENETUNREACH,
    errno.EHOSTDOWN,
    errno.EHOSTUNREACH,
    errno.EPIPE,
}

PERMANENT_ERRNOS = {
    errno.ENOSPC,
    getattr(errno, "EDQUOT", 122),
    errno.EACCES,
    errno.EPERM,
    errno.EROFS,
}


def resolved_artifact_io(config_or_settings: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return retry settings from a full config or an artifact_io mapping."""

    settings: Mapping[str, Any] = {}
    if isinstance(config_or_settings, Mapping):
        outputs = config_or_settings.get("outputs")
        if isinstance(outputs, Mapping):
            configured = outputs.get("artifact_io")
            if isinstance(configured, Mapping):
                settings = configured
        elif "max_attempts" in config_or_settings:
            settings = config_or_settings
    return DEFAULT_ARTIFACT_IO | dict(settings)


def iter_exception_chain(error: BaseException):
    """Yield an exception and every explicit/implicit cause without cycles."""

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def artifact_errno(error: BaseException) -> int | None:
    """Find the first errno carried by an OSError in the exception chain."""

    for chained in iter_exception_chain(error):
        if isinstance(chained, OSError) and chained.errno is not None:
            return int(chained.errno)
    return None


def is_transient_artifact_error(error: BaseException) -> bool:
    return artifact_errno(error) in TRANSIENT_ERRNOS


def retry_artifact_io(
    operation: Callable[[int], T],
    *,
    target_path: str | Path,
    operation_name: str,
    settings: Mapping[str, Any] | None = None,
    heartbeat_writer=None,
    state: dict[str, Any] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    random_fn: Callable[[], float] | None = None,
) -> T:
    """Retry transient I/O and report the original target after exhaustion.

    ``operation`` receives the one-based attempt number and is responsible for
    creating a fresh temporary file for that attempt.
    """

    resolved = resolved_artifact_io(settings)
    max_attempts = max(1, int(resolved["max_attempts"]))
    initial = max(0.0, float(resolved["initial_backoff_seconds"]))
    maximum = max(0.0, float(resolved["max_backoff_seconds"]))
    jitter = max(0.0, float(resolved["jitter_fraction"]))
    target = Path(target_path)
    if random_fn is None:
        if isinstance(settings, Mapping) and isinstance(settings.get("run"), Mapping):
            random_fn = dedicated_random(settings, "artifact_retry_jitter").random
        else:
            random_fn = random.Random(0).random

    for attempt in range(1, max_attempts + 1):
        try:
            result = operation(attempt)
        except Exception as error:
            error_number = artifact_errno(error)
            retryable = is_transient_artifact_error(error)
            if state is not None:
                state["artifact_last_errno"] = error_number
            if not retryable or attempt >= max_attempts:
                if state is not None:
                    failures = state.setdefault("unresolved_artifact_failures", [])
                    failures[:] = [
                        failure
                        for failure in failures
                        if not (
                            failure.get("operation") == operation_name
                            and failure.get("path") == str(target)
                        )
                    ]
                    failures.append(
                        {
                            "operation": operation_name,
                            "path": str(target),
                            "errno": error_number,
                            "attempts": attempt,
                            "error": str(error),
                        }
                    )
                raise artifact_path_error(target, error) from error

            delay = min(maximum, initial * (2 ** (attempt - 1)))
            delay *= 1.0 + jitter * ((2.0 * random_fn()) - 1.0)
            delay = max(0.0, delay)
            if state is not None:
                state["artifact_retry_count"] = int(
                    state.get("artifact_retry_count", 0)
                ) + 1
            _emit_best_effort(
                heartbeat_writer,
                "artifact_retry",
                operation_name,
                artifact_operation=operation_name,
                artifact_path=str(target),
                attempt=attempt,
                next_attempt=attempt + 1,
                errno=error_number,
                backoff_seconds=delay,
                error=str(error),
            )
            sleep_fn(delay)
            continue

        if attempt > 1:
            _emit_best_effort(
                heartbeat_writer,
                "stage_complete",
                operation_name,
                artifact_operation=operation_name,
                artifact_path=str(target),
                recovered=True,
                attempts=attempt,
            )
        return result

    raise AssertionError("unreachable artifact retry state")


def artifact_path_error(path: str | Path, error: BaseException) -> OSError:
    """Wrap any exhausted error while retaining its chained errno and target."""

    error_number = artifact_errno(error)
    return OSError(
        error_number,
        f"Failed artifact operation for {path}: {error}",
        str(path),
    )


def remove_resolved_failure(
    state: dict[str, Any] | None,
    *,
    operation_name: str,
    target_path: str | Path,
) -> None:
    if state is None:
        return
    failures = state.get("unresolved_artifact_failures")
    if not isinstance(failures, list):
        return
    target = str(target_path)
    state["unresolved_artifact_failures"] = [
        failure
        for failure in failures
        if not (
            failure.get("operation") == operation_name
            and failure.get("path") == target
        )
    ]


def emit_artifact_event(heartbeat_writer, event_type: str, stage: str, **fields: Any):
    _emit_best_effort(heartbeat_writer, event_type, stage, **fields)


def _emit_best_effort(heartbeat_writer, event_type: str, stage: str, **fields: Any):
    if heartbeat_writer is None:
        return
    try:
        emit = getattr(heartbeat_writer, "emit", None)
        if emit is not None:
            emit(event_type, stage, **fields)
        elif event_type == "stage_failed":
            failed = getattr(heartbeat_writer, "stage_failed", None)
            if failed is not None:
                failed(stage, **fields)
    except Exception:
        # Reporting an artifact failure must never replace that failure.
        return
