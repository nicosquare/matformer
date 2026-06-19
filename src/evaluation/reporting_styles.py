"""Shared style constants for figure generation and reporting."""

from __future__ import annotations

PARAMETER_COUNT_FIELDS = [
    "total_parameters",
    "embedding_parameters",
    "lm_head_parameters",
    "non_embedding_parameters",
    "ffn_parameters",
    "attention_parameters",
    "other_non_embedding_parameters",
    "lm_head_counting",
]

LOSS_MOVING_AVERAGE_FRACTION = 0.1

SIZE_PLOT_PANELS_DEFAULT = [
    ("nested-random", "slicing", None),
    ("nested-random", "concat", None),
    ("nested-all", "slicing", None),
    ("nested-all", "concat", None),
]

SIZE_PLOT_PANELS_WITH_SAMPLING = [
    ("nested-random", "slicing", "global"),
    ("nested-random", "concat", "global"),
    ("nested-random", "slicing", "per_block"),
    ("nested-random", "concat", "per_block"),
    ("nested-random", "slicing", "adaptive_per_block_thompson"),
    ("nested-random", "concat", "adaptive_per_block_thompson"),
    ("nested-random", "slicing", "adaptive_per_block_ucb"),
    ("nested-random", "concat", "adaptive_per_block_ucb"),
    ("nested-all", "slicing", None),
    ("nested-all", "concat", None),
]

SCALING_GROUP_COLORS = {
    "nested-random / slicing / global": "tab:blue",
    "nested-random / slicing / per_block": "tab:cyan",
    "nested-random / slicing / adaptive_per_block_thompson": "tab:green",
    "nested-random / slicing / adaptive_per_block_ucb": "tab:olive",
    "nested-random / concat / global": "tab:orange",
    "nested-random / concat / per_block": "tab:red",
    "nested-random / concat / adaptive_per_block_thompson": "tab:purple",
    "nested-random / concat / adaptive_per_block_ucb": "tab:pink",
    "nested-all / slicing": "tab:purple",
    "nested-all / concat": "tab:green",
    "standalone": "tab:brown",
}

SCALING_CORRECTION_STYLES = {
    "none": {"linestyle": "-", "marker": "o", "shade": 0.0},
    "gmc": {"linestyle": "--", "marker": "s", "shade": 0.2},
    "lmc": {"linestyle": "-.", "marker": "^", "shade": 0.35},
}

SCALING_SAMPLING_TONES = {
    "global": 0.0,
    "per_block": 0.28,
    "adaptive_per_block_thompson": 0.4,
    "adaptive_per_block_ucb": 0.55,
}

SCALING_SAMPLING_MARKERS = {
    "global": "o",
    "per_block": "D",
    "adaptive_per_block_thompson": "P",
    "adaptive_per_block_ucb": "X",
}

PLOT_STYLE_BASE = {
    "figure_title_fontsize": 17,
    "panel_title_fontsize": 12,
    "subfigure_title_fontsize": 13,
    "axis_label_fontsize": 11,
    "tick_label_fontsize": 10,
    "legend_fontsize": 11,
    "standalone_label": "standalone reference",
    "series_colors": SCALING_GROUP_COLORS,
    "series_aliases": {},
    "comparison_linestyle": None,
    "comparison_markers_by_variant": {},
    "curve_aliases": {},
}

PLOT_STYLE_PRESETS = {
    "default": {},
    # These presets keep the existing rendering behavior but expose the knobs
    # in one place so the figure script can be tuned without hunting through
    # the plotting code.
    "nested_all_no_corrections": {
        "figure_title_fontsize": 15,
        "curve_aliases": {
            "nested-all / slicing": "nested-all / slicing",
            "nested-all / concat": "nested-all / concat",
        },
        "series_colors": {
            "nested-all / slicing": "tab:blue",
            "nested-all / concat": "tab:orange",
            "standalone": "tab:brown",
        },
    },
    "nested_random_no_corrections": {
        "figure_title_fontsize": 15,
        "curve_aliases": {
            "nested-random / slicing / global": "nested-random / slicing / global",
            "nested-random / concat / global": "nested-random / concat / global",
            "nested-random / slicing / per_block": "nested-random / slicing / per_block",
            "nested-random / concat / per_block": "nested-random / concat / per_block",
            "nested-random / slicing / adaptive_per_block_thompson": "nested-random / slicing / adaptive_per_block_thompson",
            "nested-random / concat / adaptive_per_block_thompson": "nested-random / concat / adaptive_per_block_thompson",
            "nested-random / slicing / adaptive_per_block_ucb": "nested-random / slicing / adaptive_per_block_ucb",
            "nested-random / concat / adaptive_per_block_ucb": "nested-random / concat / adaptive_per_block_ucb",
        },
        "series_colors": {
            "nested-random / slicing / global": "tab:blue",
            "nested-random / concat / global": "tab:orange",
            "nested-random / slicing / per_block": "tab:cyan",
            "nested-random / concat / per_block": "tab:red",
            "nested-random / slicing / adaptive_per_block_thompson": "tab:green",
            "nested-random / concat / adaptive_per_block_thompson": "tab:purple",
            "nested-random / slicing / adaptive_per_block_ucb": "tab:olive",
            "nested-random / concat / adaptive_per_block_ucb": "tab:pink",
            "standalone": "tab:brown",
        },
    },
    "nested_split_no_corrections": {
        "figure_title_fontsize": 17,
        "subfigure_title_fontsize": 13,
        "legend_fontsize": 12,
        "comparison_linestyle": "-",
        "comparison_markers_by_variant": {
            "slicing": "s",
            "concat": "o",
        },
        "series_aliases": {
            "standalone": "Individual",
            "nested-random / slicing / none / global": "Slicing",
            "nested-random / concat / none / global": "Concat",
            "nested-random / concat / lmc": "Concat/LMC",
            "nested-random / concat / gmc": "Concat/GMC",
            "nested-all / slicing / none / global": "Slicing",
            "nested-all / concat / none / global": "Concat",
            "nested-all / concat / lmc": "Concat/LMC",
            "nested-all / concat / gmc": "Concat/GMC",
        },
        "series_colors": {
            "standalone": "tab:brown",
            "nested-random / slicing / none / global": "tab:red",
            "nested-random / concat / none / global": "tab:blue",
            "nested-random / concat / lmc": "tab:purple",
            "nested-random / concat / gmc": "tab:green",
            "nested-all / slicing / none / global": "tab:red",
            "nested-all / concat / none / global": "tab:blue",
            "nested-all / concat / lmc": "tab:purple",
            "nested-all / concat / gmc": "tab:green",
        },
    },
}

PPL_VS_SIZE_FIGURE_SPECS = [
    {
        "output_name": "ppl_vs_size.png",
        "figure_title": "Perplexity vs Non-embedding parameters",
        "figure_alias": "all",
        "panel_specs": SIZE_PLOT_PANELS_WITH_SAMPLING,
        "style": "default",
        "row_filter_name": None,
    },
    {
        "output_name": "ppl_vs_size_nested_all_no_corrections.png",
        "figure_title": "Perplexity vs Non-embedding parameters: nested-all, no corrections",
        "figure_alias": "nested_all",
        "panel_specs": [
            ("nested-all", "slicing", None),
            ("nested-all", "concat", None),
        ],
        "style": "nested_all_no_corrections",
        "row_filter_name": "no_corrections",
    },
    {
        "output_name": "ppl_vs_size_nested_random_no_corrections.png",
        "figure_title": "Perplexity vs Non-embedding parameters: nested-random, no corrections",
        "figure_alias": "nested_random",
        "panel_specs": [
            ("nested-random", "slicing", None),
            ("nested-random", "concat", None),
        ],
        "style": "nested_random_no_corrections",
        "row_filter_name": "no_corrections",
    },
]

PPL_VS_SIZE_SPLIT_FIGURE_SPEC = {
    "output_name": "ppl_vs_size_nested_random_vs_nested_all_no_corrections.png",
    "figure_title": "Perplexity vs Non-embedding parameters: nested-random and nested-all, no corrections",
    "style": "nested_split_no_corrections",
    "left": {
        "subfigure_title": "One width per batch",
        "series_keys": [
            "standalone",
            "nested-random / slicing / none / global",
            "nested-random / concat / none / global",
            "nested-random / concat / lmc",
            "nested-random / concat / gmc",
        ],
    },
    "right": {
        "subfigure_title": "All widths per batch",
        "series_keys": [
            "standalone",
            "nested-all / slicing / none / global",
            "nested-all / concat / none / global",
            "nested-all / concat / lmc",
            "nested-all / concat / gmc",
        ],
    },
}

