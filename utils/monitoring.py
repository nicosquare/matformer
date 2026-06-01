"""Helpers for run-scoped W&B loss-series labeling and grouping."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


DEFAULT_MONITORING_BACKEND = "wandb"
DEFAULT_LOSS_METRIC_NAME = "loss"
VALID_MONITORING_BACKENDS = {DEFAULT_MONITORING_BACKEND}


def build_loss_series_name(
    split: str,
    granularity: str | None = None,
    metric_name: str = DEFAULT_LOSS_METRIC_NAME,
    stage: str | None = None,
) -> str:
    """Return a readable W&B series name for one run-local loss trace."""

    parts = [
        _normalize_label(split, default="unknown-split"),
        _normalize_metric_name(metric_name),
    ]
    if granularity not in (None, ""):
        parts.append(_normalize_label(granularity, default="unknown-granularity"))
    if stage not in (None, ""):
        parts.append(_normalize_label(stage, default="unknown-stage"))
    return "/".join(parts)


def build_loss_series_metadata(
    *,
    run_id: str,
    topology: str,
    split: str,
    granularity: str | None = None,
    metric_name: str = DEFAULT_LOSS_METRIC_NAME,
    stage: str | None = None,
    backend: str = DEFAULT_MONITORING_BACKEND,
) -> dict[str, Any]:
    """Build the metadata bundle that should accompany one monitored series."""

    normalized_backend = _normalize_backend(backend)
    normalized_metric_name = _normalize_metric_name(metric_name)
    return {
        "run_id": str(run_id),
        "topology": _normalize_label(topology, default="unknown-topology"),
        "split": _normalize_label(split, default="unknown-split"),
        "granularity": (
            None
            if granularity in (None, "")
            else _normalize_label(granularity, default="unknown-granularity")
        ),
        "metric_name": normalized_metric_name,
        "stage": None if stage in (None, "") else _normalize_label(stage, default="unknown-stage"),
        "backend": normalized_backend,
        "series_name": build_loss_series_name(
            split=split,
            granularity=granularity,
            metric_name=normalized_metric_name,
            stage=stage,
        ),
    }


def group_loss_rows_by_series(
    rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group metric rows by the W&B series label they should emit to."""

    if isinstance(rows, Mapping):
        rows = [rows]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        normalized_row = dict(row)
        series_name = build_loss_series_name(
            split=str(normalized_row.get("split") or "unknown-split"),
            granularity=normalized_row.get("granularity"),
            metric_name=str(normalized_row.get("metric_name") or DEFAULT_LOSS_METRIC_NAME),
            stage=normalized_row.get("stage"),
        )
        grouped.setdefault(series_name, []).append(normalized_row)
    return grouped


def _normalize_backend(raw_backend: Any) -> str:
    backend = _normalize_label(raw_backend, default=DEFAULT_MONITORING_BACKEND)
    if backend not in VALID_MONITORING_BACKENDS:
        raise ValueError(
            f"Unsupported monitoring backend: {backend!r}; expected one of "
            f"{sorted(VALID_MONITORING_BACKENDS)}"
        )
    return backend


def _normalize_metric_name(raw_metric_name: Any) -> str:
    metric_name = _normalize_label(raw_metric_name, default=DEFAULT_LOSS_METRIC_NAME)
    if metric_name in {"train_loss", "validation_loss"}:
        return DEFAULT_LOSS_METRIC_NAME
    return metric_name


def _normalize_label(raw_value: Any, default: str) -> str:
    if raw_value in (None, ""):
        return default
    label = str(raw_value).strip()
    return label or default
