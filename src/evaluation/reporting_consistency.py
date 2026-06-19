"""Consistency figure helpers for reporting."""

from __future__ import annotations

from .reporting_impl import (
    consistency_metric_sort_key,
    consistency_pair_label,
    consistency_pair_sort_key,
    finalize_side_legend_figure,
    plot_consistency_results,
)

__all__ = [
    "consistency_metric_sort_key",
    "consistency_pair_label",
    "consistency_pair_sort_key",
    "finalize_side_legend_figure",
    "plot_consistency_results",
]
