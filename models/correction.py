"""Correction-context helpers for sampling-mode dependent behavior."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping, Sequence
from typing import Any

from models.granularity import GranularityPattern


VALID_CORRECTION_MODES = {"none", "gmc", "lmc"}
VALID_SAMPLING_MODES = {"global", "per_layer"}


@dataclass(frozen=True, slots=True)
class CorrectionContext:
    """Resolved correction behavior for a sampling decision."""

    correction_mode: str
    sampling_mode: str
    local_correction_active: bool
    derived_membership_pattern: tuple[Any, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def should_activate_local_correction(
    correction_mode: str,
    sampling_mode: str,
) -> bool:
    validate_correction_mode(correction_mode)
    validate_sampling_mode(sampling_mode)
    return sampling_mode == "per_layer" and correction_mode in {"gmc", "lmc"}


def validate_correction_mode(correction_mode: str) -> None:
    if correction_mode not in VALID_CORRECTION_MODES:
        raise ValueError(
            "correction_mode must be one of "
            f"{sorted(VALID_CORRECTION_MODES)}"
        )


def validate_sampling_mode(sampling_mode: str) -> None:
    if sampling_mode not in VALID_SAMPLING_MODES:
        raise ValueError(
            "sampling_mode must be one of "
            f"{sorted(VALID_SAMPLING_MODES)}"
        )


def build_correction_context(
    correction_mode: str,
    sampling_mode: str,
    derived_membership_pattern: Sequence[Any] | None = None,
) -> CorrectionContext:
    validate_correction_mode(correction_mode)
    validate_sampling_mode(sampling_mode)
    local_correction_active = should_activate_local_correction(
        correction_mode,
        sampling_mode,
    )
    if not local_correction_active:
        derived_membership_pattern = ()
    return CorrectionContext(
        correction_mode=correction_mode,
        sampling_mode=sampling_mode,
        local_correction_active=local_correction_active,
        derived_membership_pattern=tuple(derived_membership_pattern or ()),
    )


def derive_local_membership_pattern(
    granularity_pattern: GranularityPattern | Sequence[str] | None,
) -> tuple[Any, ...]:
    """Extract the layer-wise membership pattern used for local correction."""

    if granularity_pattern is None:
        return ()
    if isinstance(granularity_pattern, GranularityPattern):
        return tuple(granularity_pattern.selected_granularities)
    return tuple(granularity_pattern)


def build_local_correction_context_from_pattern(
    correction_mode: str,
    granularity_pattern: GranularityPattern | Sequence[str] | None = None,
) -> CorrectionContext:
    """Build a per-layer correction context from a sampled layer pattern."""

    return build_correction_context(
        correction_mode,
        "per_layer",
        derived_membership_pattern=derive_local_membership_pattern(
            granularity_pattern
        ),
    )


def build_correction_context_from_pattern(
    correction_mode: str,
    sampling_mode: str,
    granularity_pattern: GranularityPattern | Sequence[str] | None = None,
) -> CorrectionContext:
    derived_membership_pattern: Sequence[Any] | None
    if sampling_mode == "per_layer":
        derived_membership_pattern = derive_local_membership_pattern(
            granularity_pattern
        )
    elif granularity_pattern is None:
        derived_membership_pattern = None
    elif isinstance(granularity_pattern, GranularityPattern):
        derived_membership_pattern = granularity_pattern.selected_granularities
    else:
        derived_membership_pattern = tuple(granularity_pattern)
    return build_correction_context(
        correction_mode,
        sampling_mode,
        derived_membership_pattern=derived_membership_pattern,
    )


def summarize_correction_context(context: CorrectionContext) -> dict[str, Any]:
    return context.to_dict()


def correction_context_from_config(
    config: Mapping[str, Any],
    granularity_pattern: GranularityPattern | Sequence[str] | None = None,
) -> CorrectionContext:
    model = config.get("model", {})
    run = config.get("run", {})
    if not isinstance(model, Mapping):
        model = {}
    if not isinstance(run, Mapping):
        run = {}

    correction_mode = str(model.get("correction_mode", "none"))
    sampling_mode = str(model.get("granularity_sampling_mode", "global"))
    if sampling_mode not in VALID_SAMPLING_MODES and run.get("sampling_mode"):
        run_sampling_mode = str(run["sampling_mode"])
        if run_sampling_mode == "nested-random":
            sampling_mode = "per_layer"
        else:
            sampling_mode = "global"
    if sampling_mode == "per_layer":
        return build_local_correction_context_from_pattern(
            correction_mode,
            granularity_pattern=granularity_pattern,
        )
    return build_correction_context_from_pattern(
        correction_mode,
        sampling_mode,
        granularity_pattern=granularity_pattern,
    )
