"""Model assembly and layer wiring for granularity-aware MatFormer models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from models.granularity import (
    GranularityPattern,
    build_granularity_pattern,
    expand_layer_granularity_pattern,
)


def build_global_granularity_pattern(
    config: Mapping[str, Any],
    granularities: Sequence[str] | None = None,
) -> GranularityPattern:
    """Build the explicit global sampling pattern for a run.

    The current global path applies all configured granularities one after the
    other, so the selected set is the configured granularity list in order.
    """

    model = config.get("model", {})
    run = config.get("run", {})
    if not isinstance(model, Mapping):
        model = {}
    if not isinstance(run, Mapping):
        run = {}

    sampling_mode = model.get("granularity_sampling_mode")
    if sampling_mode not in (None, "global"):
        raise ValueError(
            "build_global_granularity_pattern requires model.granularity_sampling_mode=global"
        )

    selected_granularities = tuple(
        granularities or model.get("granularities", [])
    )
    if not selected_granularities:
        raise ValueError("granularities must be a non-empty sequence")

    layer_count = int(model.get("num_layers") or len(selected_granularities))
    return build_granularity_pattern(
        pattern_type="single",
        selected_granularities=selected_granularities,
        layer_count=layer_count,
        repeatable_source=(
            str(run.get("run_id") or ""),
            "model.granularity_sampling_mode=global",
        ),
    )


def build_per_layer_granularity_pattern(
    config: Mapping[str, Any],
    layer_granularities: Sequence[str],
) -> GranularityPattern:
    """Build the explicit per-layer sampling pattern for a run.

    The input sequence is treated as the seed pattern that repeats across the
    configured transformer layers.
    """

    model = config.get("model", {})
    run = config.get("run", {})
    if not isinstance(model, Mapping):
        model = {}
    if not isinstance(run, Mapping):
        run = {}

    sampling_mode = model.get("granularity_sampling_mode")
    if sampling_mode not in (None, "per_layer"):
        raise ValueError(
            "build_per_layer_granularity_pattern requires "
            "model.granularity_sampling_mode=per_layer"
        )

    seed_granularities = tuple(layer_granularities)
    if not seed_granularities:
        raise ValueError("layer_granularities must be a non-empty sequence")

    layer_count = int(model.get("num_layers") or len(seed_granularities))
    selected_granularities = tuple(
        expand_layer_granularity_pattern(seed_granularities, layer_count)
    )
    return build_granularity_pattern(
        pattern_type="per_layer",
        selected_granularities=selected_granularities,
        layer_count=layer_count,
        repeatable_source=(
            str(run.get("run_id") or ""),
            "model.granularity_sampling_mode=per_layer",
            *seed_granularities,
        ),
    )


def apply_granularity_pattern_to_model(
    model,
    selected_granularities: Sequence[str] | str,
    sampling_mode: str,
) -> GranularityPattern:
    """Apply a resolved granularity pattern to a MatFormer model instance."""

    target = model.module if hasattr(model, "module") else model
    layers = getattr(target, "matformer_layers", None)
    if layers is None:
        raise AttributeError(
            "apply_granularity_pattern_to_model requires a MatFormer model with "
            "matformer_layers"
        )

    granularities = _normalize_selected_granularities(selected_granularities)
    run_id = str(getattr(target, "current_run_id", "") or "")
    config = {
        "model": {
            "granularity_sampling_mode": sampling_mode,
            "granularities": list(granularities),
            "num_layers": len(layers),
        },
        "run": {"run_id": run_id},
    }

    if sampling_mode == "global":
        pattern = build_global_granularity_pattern(
            config,
            granularities=granularities,
        )
    elif sampling_mode == "per_layer":
        pattern = build_per_layer_granularity_pattern(
            config,
            layer_granularities=granularities,
        )
    else:
        raise ValueError(
            "sampling_mode must be one of {'global', 'per_layer'}"
        )

    target.current_sampling_mode = sampling_mode
    target.current_granularity_pattern = pattern
    if sampling_mode == "global":
        configured_granularities = [granularities[0]] * len(layers)
    else:
        configured_granularities = list(pattern.selected_granularities)
    target.current_layer_granularities = configured_granularities
    for layer, granularity in zip(layers, configured_granularities):
        layer.configure_subnetwork(granularity)
    return pattern


def _normalize_selected_granularities(
    selected_granularities: Sequence[str] | str,
) -> tuple[str, ...]:
    if isinstance(selected_granularities, str):
        return (selected_granularities,)
    granularities = tuple(selected_granularities)
    if not granularities:
        raise ValueError("selected_granularities must be a non-empty sequence")
    return granularities
