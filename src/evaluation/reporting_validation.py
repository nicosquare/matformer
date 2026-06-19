"""Validation figure helpers for reporting."""

from __future__ import annotations

from .reporting_impl import (
    group_validation_rows_by_method,
    group_validation_rows_by_variant,
    loss_trace_description,
    loss_trace_panel_suffix,
    loss_trace_series_sort_key,
    loss_trace_kind,
    moving_average,
    plot_metric_over_steps,
    plot_loss_over_tokens_for_experiment,
    plot_validation_loss_over_tokens_by_experiment,
    plot_validation_loss_over_tokens_by_granularity_comparison,
    plot_validation_loss_over_tokens_by_granularity_comparison_figure,
    validation_comparison_display_label,
    validation_comparison_method_key,
    validation_comparison_method_order,
    validation_comparison_styles,
    validation_variant_display_label,
    validation_variant_display_labels,
    validation_variant_key,
    validation_variant_order,
    validation_variant_styles,
)
from .validation import aggregate_scaling_summary

__all__ = [
    "aggregate_scaling_summary",
    "group_validation_rows_by_method",
    "group_validation_rows_by_variant",
    "loss_trace_description",
    "loss_trace_panel_suffix",
    "loss_trace_series_sort_key",
    "loss_trace_kind",
    "moving_average",
    "plot_metric_over_steps",
    "plot_loss_over_tokens_for_experiment",
    "plot_validation_loss_over_tokens_by_experiment",
    "plot_validation_loss_over_tokens_by_granularity_comparison",
    "plot_validation_loss_over_tokens_by_granularity_comparison_figure",
    "validation_comparison_display_label",
    "validation_comparison_method_key",
    "validation_comparison_method_order",
    "validation_comparison_styles",
    "validation_variant_display_label",
    "validation_variant_display_labels",
    "validation_variant_key",
    "validation_variant_order",
    "validation_variant_styles",
]
