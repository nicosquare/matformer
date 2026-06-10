"""Model assembly and layer wiring for granularity-aware MatFormer models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from models.granularity import (
    GranularityPattern,
    build_granularity_pattern,
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
