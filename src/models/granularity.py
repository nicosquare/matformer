"""Granularity metadata helpers for the MatFormer model family."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from collections.abc import Mapping, Sequence
from typing import Any

from src.utils.config import CANONICAL_GRANULARITY_PREFIX_FRACTIONS


MATFORMER_GRANULARITY_ORDER = ("s", "m", "l", "xl")
MATFORMER_GRANULARITY_DISPLAY_NAMES = {
    "s": "S",
    "m": "M",
    "l": "L",
    "xl": "XL",
}


@dataclass(frozen=True, slots=True)
class GranularityMetadata:
    """Canonical metadata for a single MatFormer granularity."""

    name: str
    display_name: str
    ffn_ratio: float
    full_intermediate_fraction: float
    prefix_width: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GranularityPattern:
    """Compact description of a sampled granularity pattern."""

    pattern_type: str
    selected_granularities: tuple[str, ...]
    layer_count: int
    repeatable_source: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_granularity(
    granularity: str,
    allowed_granularities: Sequence[str] | None = None,
) -> None:
    allowed = tuple(allowed_granularities or MATFORMER_GRANULARITY_ORDER)
    if not isinstance(granularity, str) or not granularity.strip():
        raise ValueError("MatFormer granularity labels must be non-empty strings")
    if granularity not in allowed:
        raise ValueError(f"Unknown MatFormer granularity: {granularity}")


def validate_granularity_sequence(
    granularities: Sequence[str],
) -> tuple[str, ...]:
    """Validate and preserve an ordered granularity label sequence."""

    if isinstance(granularities, (str, bytes)) or not granularities:
        raise ValueError("granularities must be a non-empty sequence")

    ordered = tuple(granularities)
    for granularity in ordered:
        validate_granularity(granularity, allowed_granularities=ordered)
    if len(set(ordered)) != len(ordered):
        raise ValueError("granularities must contain unique labels")
    return ordered


def canonical_prefix_fraction(granularity: str) -> float:
    numerator, denominator = CANONICAL_GRANULARITY_PREFIX_FRACTIONS[granularity]
    return numerator / denominator


def get_granularity_metadata(
    granularity: str,
    granularity_prefixes: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    allowed = tuple(granularity_prefixes) if granularity_prefixes is not None else None
    validate_granularity(granularity, allowed_granularities=allowed)
    fraction = _granularity_prefix_fraction(
        granularity,
        granularity_prefixes=granularity_prefixes,
    )
    if granularity_prefixes is None:
        reference_fraction = canonical_prefix_fraction("m")
    elif "m" in granularity_prefixes:
        reference_fraction = float(granularity_prefixes["m"])
    else:
        reference_fraction = fraction
    metadata = GranularityMetadata(
        name=granularity,
        display_name=MATFORMER_GRANULARITY_DISPLAY_NAMES.get(
            granularity,
            granularity.upper(),
        ),
        ffn_ratio=round(fraction / reference_fraction, 12),
        full_intermediate_fraction=fraction,
    )
    return {
        "display_name": metadata.display_name,
        "ffn_ratio": metadata.ffn_ratio,
        "full_intermediate_fraction": metadata.full_intermediate_fraction,
    }


def granularity_prefix_width(
    intermediate_size: int,
    granularity: str,
    granularity_prefixes: Mapping[str, object] | None = None,
) -> int:
    allowed = tuple(granularity_prefixes) if granularity_prefixes is not None else None
    validate_granularity(granularity, allowed_granularities=allowed)
    fraction = _granularity_prefix_fraction(
        granularity,
        granularity_prefixes=granularity_prefixes,
    )
    prefix_width = int(intermediate_size * fraction)
    if prefix_width <= 0:
        raise ValueError(
            f"Granularity {granularity} produced empty FFN prefix for "
            f"intermediate_size={intermediate_size}"
        )
    return prefix_width


def get_ffn_prefix_metadata(
    intermediate_size: int,
    granularity_prefixes: Mapping[str, object] | None = None,
    granularities: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    granularities = validate_granularity_sequence(
        granularities or MATFORMER_GRANULARITY_ORDER
    )
    resolved_prefixes = _resolve_granularity_prefixes(
        granularities,
        granularity_prefixes=granularity_prefixes,
    )
    ratio_reference_fraction = (
        canonical_prefix_fraction("m")
        if granularities == MATFORMER_GRANULARITY_ORDER
        else resolved_prefixes[granularities[0]]
    )
    metadata = []
    for granularity in granularities:
        granularity_metadata = get_granularity_metadata(
            granularity,
            granularity_prefixes=resolved_prefixes,
        )
        fraction = resolved_prefixes[granularity]
        entry = GranularityMetadata(
            name=granularity,
            display_name=granularity_metadata["display_name"],
            ffn_ratio=round(fraction / ratio_reference_fraction, 12),
            full_intermediate_fraction=fraction,
            prefix_width=granularity_prefix_width(
                intermediate_size,
                granularity,
                granularity_prefixes=resolved_prefixes,
            ),
        )
        metadata.append(entry.to_dict())
    return metadata


def granularity_block_count(
    granularity: str,
    granularities: Sequence[str] | None = None,
) -> int:
    if granularities is not None:
        ordered = validate_granularity_sequence(granularities)
        validate_granularity(granularity, allowed_granularities=ordered)
        return ordered.index(granularity) + 1

    metadata = get_granularity_metadata(granularity)
    smallest_granularity = MATFORMER_GRANULARITY_ORDER[0]
    smallest_ratio = get_granularity_metadata(smallest_granularity)["ffn_ratio"]
    ratio = metadata["ffn_ratio"]
    block_count = ratio / smallest_ratio
    if int(block_count) != block_count:
        raise ValueError(
            f"Granularity {granularity} has incompatible ffn_ratio={ratio} "
            f"for smallest_ratio={smallest_ratio}"
        )
    return int(block_count)


def get_concat_block_metadata(
    intermediate_size: int,
    granularity_prefixes: Mapping[str, object] | None = None,
    granularities: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    granularities = validate_granularity_sequence(
        granularities or MATFORMER_GRANULARITY_ORDER
    )
    prefix_metadata = get_ffn_prefix_metadata(
        intermediate_size,
        granularity_prefixes=granularity_prefixes,
        granularities=granularities,
    )
    base_block_width = prefix_metadata[0]["prefix_width"]
    block_metadata = []
    previous_prefix_width = 0

    for block_index, prefix_entry in enumerate(prefix_metadata):
        prefix_width = prefix_entry["prefix_width"]
        if prefix_width <= previous_prefix_width:
            raise ValueError(
                "Granularity order must expand to strictly larger prefix blocks"
            )
        block_width = prefix_width - previous_prefix_width
        block_metadata.append(
            {
                "name": f"block_{block_index + 1}",
                "display_name": f"B{block_index + 1}",
                "ffn_ratio": block_width / base_block_width,
                "full_intermediate_fraction": prefix_width / intermediate_size,
                "prefix_width": prefix_width,
                "block_width": block_width,
                "cumulative_prefix_width": prefix_width,
            }
        )
        previous_prefix_width = prefix_width

    return block_metadata


def granularity_concat_block_count(
    granularity: str,
    granularities: Sequence[str] | None = None,
) -> int:
    ordered = validate_granularity_sequence(
        granularities or MATFORMER_GRANULARITY_ORDER
    )
    validate_granularity(granularity, allowed_granularities=ordered)
    return ordered.index(granularity) + 1


def get_block_membership_counts(
    granularities: Sequence[str],
    total_blocks: int | None = None,
) -> list[int]:
    if not granularities:
        raise ValueError("granularities must be a non-empty sequence")

    ordered = validate_granularity_sequence(granularities)
    if set(ordered).issubset(MATFORMER_GRANULARITY_ORDER):
        # Preserve the canonical block geometry when callers provide a
        # canonical subset, while explicit custom labels use their resolved
        # sequence order as the only geometry available to this helper.
        block_counts = [granularity_block_count(granularity) for granularity in ordered]
    else:
        block_counts = [index + 1 for index, _ in enumerate(ordered)]
    max_blocks = max(block_counts)
    if total_blocks is None:
        total_blocks = max_blocks
    elif total_blocks < max_blocks:
        raise ValueError(
            "total_blocks cannot be smaller than the widest configured granularity"
        )

    return [
        sum(1 for block_count in block_counts if block_index < block_count)
        for block_index in range(total_blocks)
    ]


def get_gradient_membership_correction_scales(
    granularities: Sequence[str],
    total_blocks: int | None = None,
) -> list[float]:
    membership_counts = get_block_membership_counts(
        granularities,
        total_blocks=total_blocks,
    )
    total_losses = len(granularities)
    scales = []
    for membership_count in membership_counts:
        if membership_count == 0:
            scales.append(1.0)
            continue
        scales.append(total_losses / membership_count)
    return scales


def get_concat_block_membership_counts(
    intermediate_size: int,
    granularities: Sequence[str],
    granularity_prefixes: Mapping[str, object] | None = None,
) -> list[int]:
    if not granularities:
        raise ValueError("granularities must be a non-empty sequence")

    return get_concat_block_membership_counts_from_metadata(
        get_concat_block_metadata(
            intermediate_size,
            granularity_prefixes=granularity_prefixes,
            granularities=granularities,
        ),
        granularities,
    )


def get_concat_block_membership_counts_from_metadata(
    block_metadata: Sequence[Mapping[str, Any]],
    granularities: Sequence[str],
) -> list[int]:
    if not block_metadata:
        raise ValueError("block_metadata must be a non-empty sequence")
    if not granularities:
        raise ValueError("granularities must be a non-empty sequence")

    use_canonical_positions = set(granularities).issubset(
        MATFORMER_GRANULARITY_ORDER
    )
    prefix_widths = []
    for granularity in granularities:
        try:
            block_index = (
                MATFORMER_GRANULARITY_ORDER.index(granularity)
                if use_canonical_positions
                else granularities.index(granularity)
            )
        except ValueError as error:
            raise ValueError(
                f"Unknown configured granularity: {granularity}"
            ) from error
        if block_index >= len(block_metadata):
            raise ValueError(
                "Configured granularities exceed the available concat blocks"
            )
        prefix_widths.append(block_metadata[block_index]["cumulative_prefix_width"])
    return [
        sum(
            1
            for prefix_width in prefix_widths
            if prefix_width >= block["cumulative_prefix_width"]
        )
        for block in block_metadata
    ]


def get_concat_gradient_membership_correction_scales_from_metadata(
    block_metadata: Sequence[Mapping[str, Any]],
    granularities: Sequence[str],
) -> list[float]:
    membership_counts = get_concat_block_membership_counts_from_metadata(
        block_metadata,
        granularities,
    )
    total_losses = len(granularities)
    scales = []
    for membership_count in membership_counts:
        if membership_count == 0:
            scales.append(1.0)
            continue
        scales.append(total_losses / membership_count)
    return scales


def get_concat_gradient_membership_correction_scales(
    intermediate_size: int,
    granularities: Sequence[str],
    granularity_prefixes: Mapping[str, object] | None = None,
) -> list[float]:
    return get_concat_gradient_membership_correction_scales_from_metadata(
        get_concat_block_metadata(
            intermediate_size,
            granularity_prefixes=granularity_prefixes,
            granularities=granularities,
        ),
        granularities,
    )


def get_concat_layout_diagnostic(
    intermediate_size: int,
    granularities: Sequence[str],
    granularity_prefixes: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    block_metadata = get_concat_block_metadata(
        intermediate_size,
        granularity_prefixes=granularity_prefixes,
        granularities=granularities,
    )
    return {
        "intermediate_size": intermediate_size,
        "granularities": list(granularities),
        "block_widths": [block["block_width"] for block in block_metadata],
        "prefix_widths": [block["prefix_width"] for block in block_metadata],
        "gradient_membership_counts": get_concat_block_membership_counts_from_metadata(
            block_metadata,
            granularities,
        ),
        "gradient_membership_correction_scales": get_concat_gradient_membership_correction_scales_from_metadata(
            block_metadata,
            granularities,
        ),
    }


def get_prefix_membership_segment_metadata(
    intermediate_size: int,
    granularities: Sequence[str],
    granularity_prefixes: Mapping[str, object] | None = None,
) -> list[dict[str, Any]]:
    if not granularities:
        raise ValueError("granularities must be a non-empty sequence")

    prefix_widths = [
        granularity_prefix_width(
            intermediate_size,
            granularity,
            granularity_prefixes=granularity_prefixes,
        )
        for granularity in granularities
    ]
    total_losses = len(granularities)
    boundaries = sorted(set(prefix_widths))
    metadata = []
    start = 0

    for end in boundaries:
        if end <= start:
            continue
        membership_count = sum(1 for prefix_width in prefix_widths if prefix_width >= end)
        metadata.append(
            {
                "start": start,
                "end": end,
                "width": end - start,
                "membership_count": membership_count,
                "scale": total_losses / membership_count,
            }
        )
        start = end

    if start < intermediate_size:
        metadata.append(
            {
                "start": start,
                "end": intermediate_size,
                "width": intermediate_size - start,
                "membership_count": 0,
                "scale": 1.0,
            }
        )

    return metadata


def build_granularity_pattern(
    pattern_type: str,
    selected_granularities: Sequence[str],
    layer_count: int,
    repeatable_source: Sequence[str] | None = None,
    available_granularities: Sequence[str] | None = None,
) -> GranularityPattern:
    selected_granularities = tuple(selected_granularities)
    if not selected_granularities:
        raise ValueError("selected_granularities must be a non-empty sequence")
    allowed = validate_granularity_sequence(
        available_granularities or MATFORMER_GRANULARITY_ORDER
    )
    for granularity in selected_granularities:
        validate_granularity(granularity, allowed_granularities=allowed)
    if layer_count < 0:
        raise ValueError("layer_count must be non-negative")
    return GranularityPattern(
        pattern_type=pattern_type,
        selected_granularities=tuple(selected_granularities),
        layer_count=layer_count,
        repeatable_source=tuple(repeatable_source or ()),
    )


def summarize_granularity_pattern(pattern: GranularityPattern) -> dict[str, Any]:
    return pattern.to_dict()


def resolved_granularity_artifact_fields(
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Return stable granularity metadata for experiment artifacts."""

    granularities = list(model.get("granularities", []))
    prefixes = model.get("granularity_prefixes", {})
    if not isinstance(prefixes, Mapping):
        prefixes = {}
    metadata = model.get("ffn_prefix_metadata", [])
    widths = {}
    if isinstance(metadata, Sequence):
        widths = {
            str(entry["name"]): int(entry["prefix_width"])
            for entry in metadata
            if isinstance(entry, Mapping)
            and "name" in entry
            and entry.get("prefix_width") is not None
        }
    return {
        "granularity_mode": model.get("granularity_mode", "canonical"),
        "granularities": granularities,
        "granularity_prefixes": dict(prefixes),
        "granularity_prefix_widths": widths,
    }


def summarize_granularity_pattern_from_config(
    config: Mapping[str, Any],
    runtime_pattern: GranularityPattern | None = None,
) -> dict[str, Any]:
    """Build a stable granularity-pattern summary from resolved config state."""

    model = config.get("model", {})
    run = config.get("run", {})
    training = config.get("training", {})
    if not isinstance(model, Mapping):
        model = {}
    if not isinstance(run, Mapping):
        run = {}
    if not isinstance(training, Mapping):
        training = {}

    if runtime_pattern is not None:
        summary = runtime_pattern.to_dict()
        repeatable_source = summary.get("repeatable_source")
        if isinstance(repeatable_source, (list, tuple)) and repeatable_source:
            summary["repeatable_source"] = [
                str(run.get("run_id") or repeatable_source[0]),
                *repeatable_source[1:],
            ]
        return summary

    resolved_run_mode = str(run.get("sampling_mode") or "nested-random")
    sampling_mode = str(model.get("granularity_sampling_mode", "global"))

    if resolved_run_mode == "nested-all":
        pattern_type = "all_granularities"
    elif sampling_mode in {"per_block", "adaptive_per_block"}:
        pattern_type = "per_block"
    else:
        pattern_type = "single"

    selected_granularities = list(model.get("granularities", []))
    if resolved_run_mode == "standalone" and run.get("granularity") is not None:
        selected_granularities = [str(run["granularity"])]

    repeatable_source = [
        str(run.get("run_id") or ""),
        f"run.sampling_mode={resolved_run_mode}",
        f"model.granularity_sampling_mode={sampling_mode}",
    ]
    if sampling_mode == "fixed_global":
        repeatable_source.append(
            "model.global_sampling_distribution="
            + json.dumps(
                model.get("global_sampling_distribution", {}),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    if sampling_mode == "global" and resolved_run_mode == "nested-random":
        if model.get("global_sampling_schedule") == "balanced_cycle":
            repeatable_source.append(
                "model.global_sampling_schedule=balanced_cycle"
            )
            repeatable_source.append(
                "model.global_sampling_schedule_version="
                f"{int(model['global_sampling_schedule_version'])}"
            )
        repeatable_source.append(
            "model.global_sampling_interval_steps="
            f"{int(model.get('global_sampling_interval_steps', 1))}"
        )
    if resolved_run_mode == "standalone" and run.get("granularity") is not None:
        repeatable_source.append(f"run.granularity={run['granularity']}")

    return {
        "pattern_type": pattern_type,
        "selected_granularities": selected_granularities,
        "layer_count": model.get("num_layers"),
        "repeatable_source": repeatable_source,
    }


def _canonical_prefix_fraction(granularity: str) -> float:
    return canonical_prefix_fraction(granularity)


def _granularity_prefix_fraction(
    granularity: str,
    granularity_prefixes: Mapping[str, object] | None = None,
) -> float:
    if granularity_prefixes is None:
        return canonical_prefix_fraction(granularity)
    if granularity not in granularity_prefixes:
        raise ValueError(f"Missing MatFormer granularity prefix: {granularity}")
    return float(granularity_prefixes[granularity])


def _resolve_granularity_prefixes(
    granularities: Sequence[str],
    granularity_prefixes: Mapping[str, object] | None = None,
) -> dict[str, float]:
    granularities = validate_granularity_sequence(granularities)
    if granularity_prefixes is None:
        try:
            prefixes = {
                granularity: canonical_prefix_fraction(granularity)
                for granularity in granularities
            }
        except KeyError as error:
            raise ValueError(
                "Custom granularity labels require granularity_prefixes"
            ) from error
    else:
        prefixes = granularity_prefixes
    resolved = {}
    previous_fraction = 0.0
    for granularity in granularities:
        fraction = _granularity_prefix_fraction(
            granularity,
            granularity_prefixes=prefixes,
        )
        if fraction <= 0:
            raise ValueError(
                f"Granularity {granularity} must resolve to a positive prefix fraction"
            )
        if fraction <= previous_fraction:
            raise ValueError(
                "MatFormer granularity prefixes must be strictly increasing "
                "in the configured order"
            )
        resolved[granularity] = fraction
        previous_fraction = fraction
    return resolved
