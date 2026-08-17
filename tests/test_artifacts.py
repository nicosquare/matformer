import csv
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

from src.evaluation.reporting_impl import (
    blend_color_toward_white,
    comparison_series_key,
    comparison_series_style,
    scaling_curve_color_group_label,
    scaling_curve_display_label,
    scaling_curve_label,
    scaling_curve_style,
    resolve_plot_style,
    resolve_series_alias,
)
from src.evaluation.reporting import generate_figures
from src.evaluation.reporting_styles import STANDALONE_REFERENCE_COLOR
from src.models.correction import (
    correction_context_from_config,
    summarize_correction_context,
)
from src.models.granularity import summarize_granularity_pattern
from src.models.wiring import (
    build_global_granularity_pattern,
    build_per_block_granularity_pattern,
)
from src.utils.config import ConfigError, resolve_all_run_configs, resolve_run_config
from src.utils.metrics import (
    ArtifactError,
    METRICS_COLUMNS,
    SCALING_RESULTS_COLUMNS,
    build_run_summary,
    build_consistency_result_rows,
    build_scaling_result_rows,
    append_controller_events,
    build_compact_controller_metric_fields,
    build_controller_summary,
    write_config_artifact,
    write_consistency_results_csv,
    write_failed_run_summary,
    write_metrics_csv,
    write_run_summary,
    write_scaling_results_csv,
    write_task_results_csv,
)
from src.training.run import run_training
from src.training.panelgrad import PanelGradController
from src.training.steps import build_training_metric_row
from src.utils.reproducibility import configure_strict_determinism, derive_seed


class TinyExtractionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.current_granularity = None
        self.ffn_prefix_metadata = [
            {
                "name": "s",
                "display_name": "S",
                "ffn_ratio": 0.5,
                "full_intermediate_fraction": 0.125,
                "prefix_width": 8,
            },
            {
                "name": "m",
                "display_name": "M",
                "ffn_ratio": 1.0,
                "full_intermediate_fraction": 0.25,
                "prefix_width": 16,
            },
            {
                "name": "l",
                "display_name": "L",
                "ffn_ratio": 2.0,
                "full_intermediate_fraction": 0.5,
                "prefix_width": 32,
            },
            {
                "name": "xl",
                "display_name": "XL",
                "ffn_ratio": 4.0,
                "full_intermediate_fraction": 1.0,
                "prefix_width": 64,
            },
        ]

    def configure_subnetwork(self, granularity):
        self.current_granularity = granularity

    def forward(self, input_ids, attention_mask=None, labels=None):
        loss = self.weight.pow(2) + input_ids.float().mean() * 0.0
        return SimpleNamespace(loss=loss)


def test_write_config_metrics_and_run_summary(tmp_path):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
    )
    config_path = write_config_artifact(config)
    metrics_path = write_metrics_csv(
        output_dir,
        [
            {
                "run_id": "debug-nested-001",
                "step": 0,
                "split": "validation",
                "model_family": "nested",
                "model_size_label": "debug",
                "granularity": "s",
                "loss": 2.1,
                "perplexity": 8.17,
                "tokens_seen": 128,
                "wall_clock_seconds": 1.5,
                "tokens_per_second": 85.3,
                "peak_memory_bytes": 2048,
            },
            {
                "run_id": "debug-nested-001",
                "step": 0,
                "split": "validation",
                "model_family": "nested",
                "model_size_label": "debug",
                "granularity": "xl",
                "loss": 1.7,
                "perplexity": 5.47,
                "tokens_seen": 128,
                "wall_clock_seconds": 1.5,
                "tokens_per_second": 85.3,
                "peak_memory_bytes": 2048,
            },
        ],
    )
    summary = build_run_summary(config, tokens_seen=128, notes=["smoke test"])
    summary_path = write_run_summary(output_dir, summary)

    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_config["run"]["run_id"] == "debug-nested-001"
    assert saved_config["model"]["variant"] == "slicing"
    assert saved_config["training"]["base_learning_rate"] == 0.0003
    assert saved_config["training"]["learning_rate_scale_rule"] == "none"
    assert saved_config["training"]["learning_rate_scale_factor"] == 1.0
    assert saved_config["training"]["resolved_learning_rate"] == 0.0003
    assert saved_config["training"]["warmup_ratio"] == 0.0
    assert saved_config["training"]["warmup_steps"] == 0
    assert saved_config["training"]["resolved_warmup_steps"] == 0
    assert saved_config["training"]["gradient_clip_norm"] == 1.0
    assert saved_config["training"]["scheduler_name"] == "cosine"
    assert saved_config["training"]["scheduler"]["kwargs"]["warmup_steps"] == 0
    assert saved_config["training"]["scheduler"]["resolved_warmup_steps"] == 0
    assert saved_config["training"]["scheduler_kwargs"] == {}
    assert saved_config["training"]["preset_selections"] == {"optimizer": "adam"}
    assert set(saved_config["training"]["preset_registry_paths"]) == {"optimizer"}
    assert saved_config["training"]["preset_registry_paths"]["optimizer"].endswith(
        "configs/presets/optimizer/adam.yaml"
    )
    assert saved_config["training"]["optimizer_name"] == "adamw"
    assert saved_config["training"]["optimizer_kwargs"] == {
        "betas": [0.9, 0.95],
        "eps": 1e-08,
        "weight_decay": 0.1,
    }

    with metrics_path.open("r", encoding="utf-8", newline="") as metrics_file:
        metric_rows = list(csv.DictReader(metrics_file))
    assert [row["granularity"] for row in metric_rows] == ["s", "xl"]
    assert metric_rows[0]["peak_memory_bytes"] == "2048"

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["status"] == "completed"
    assert saved_summary["tokens_seen"] == 128
    assert saved_summary["model_variant"] == "slicing"
    assert saved_summary["base_learning_rate"] == 0.0003
    assert saved_summary["learning_rate_scale_rule"] == "none"
    assert saved_summary["learning_rate_scale_factor"] == 1.0
    assert saved_summary["resolved_learning_rate"] == 0.0003
    assert saved_summary["warmup_ratio"] == 0.0
    assert saved_summary["warmup_steps"] == 0
    assert saved_summary["resolved_warmup_steps"] == 0
    assert saved_summary["gradient_clip_norm"] == 1.0
    assert saved_summary["scheduler_name"] == "cosine"
    assert saved_summary["scheduler_warmup_steps"] == 0
    assert saved_summary["scheduler_resolved_warmup_steps"] == 0
    assert saved_summary["scheduler_kwargs"] == {}
    assert saved_summary["preset_selections"] == {"optimizer": "adam"}
    assert set(saved_summary["preset_registry_paths"]) == {"optimizer"}
    assert saved_summary["preset_registry_paths"]["optimizer"].endswith(
        "configs/presets/optimizer/adam.yaml"
    )
    assert saved_summary["optimizer_name"] == "adamw"
    assert saved_summary["optimizer_kwargs"] == {
        "betas": [0.9, 0.95],
        "eps": 1e-08,
        "weight_decay": 0.1,
    }
    assert saved_summary["family_size_slug"] == saved_summary["model_size_slug"]
    assert saved_summary["notes"] == ["smoke test"]


def test_run_summary_includes_default_long_run_metadata(tmp_path):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
    )

    summary = build_run_summary(config, tokens_seen=0)
    summary_path = write_run_summary(output_dir, summary)

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["monitoring_enabled"] is False
    assert saved_summary["monitoring_backend"] == "wandb"
    assert saved_summary["monitoring_series_metadata"] == []
    assert saved_summary["latest_checkpoint_path"] is None
    assert saved_summary["continuation_state"] == {
        "run_id": "debug-nested-001",
        "output_dir": str(output_dir),
        "latest_checkpoint_path": None,
        "last_completed_step": 0,
        "tokens_seen": 0,
        "status": "fresh",
        "resume_count": 0,
    }
    assert saved_summary["warmup_policy"] == {
        "enabled": False,
        "duration": 0,
        "unit": "epochs",
        "policy": "full_only",
        "completed": False,
        "completion_step": None,
        "transition_reason": None,
    }
    assert saved_summary["warmup_completion_step"] is None
    assert saved_summary["warmup_completed"] is False


def test_run_summary_default_pattern_summary_distinguishes_nested_all(tmp_path):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=["run.sampling_mode=nested-all"],
    )

    summary = build_run_summary(config, tokens_seen=0)

    assert summary["sampling_mode"] == "nested-all"
    assert summary["granularity_pattern_summary"] == {
        "pattern_type": "all_granularities",
        "selected_granularities": config["model"]["granularities"],
        "layer_count": config["model"]["num_layers"],
        "repeatable_source": [
            config["run"]["run_id"],
            "run.sampling_mode=nested-all",
        "model.granularity_sampling_mode=global",
        ],
    }


def test_artifacts_record_nested_all_sampling_mode_and_pattern_provenance(tmp_path):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "run.sampling_mode=nested-all",
            "model.granularity_sampling_mode=global",
        ],
    )

    config_path = write_config_artifact(config)
    summary = build_run_summary(config, tokens_seen=128, notes=["artifact smoke"])
    summary_path = write_run_summary(output_dir, summary)

    metric_rows = [
        build_training_metric_row(
            config,
            step=1,
            granularity=granularity,
            loss=float(index + 1),
            tokens_seen=8,
            content_tokens_seen=8,
            wall_clock_seconds=2.0,
            peak_memory_bytes=512,
        )
        for index, granularity in enumerate(config["model"]["granularities"])
    ]
    metrics_path = write_metrics_csv(output_dir, metric_rows)

    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_config["run"]["sampling_mode"] == "nested-all"
    assert saved_config["model"]["granularity_sampling_mode"] == "global"
    assert saved_config["model"]["granularity_pattern_provenance"] == {
        "pattern_type": "all_granularities",
        "scope": "model",
        "source": "model.granularity_sampling_mode",
        "requested_alias": None,
        "layer_count": config["model"]["num_layers"],
        "available_granularities": ["s", "m", "l", "xl"],
    }

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["sampling_mode"] == "nested-all"
    assert saved_summary["resolved_run_mode"] == "nested-all"
    assert saved_summary["resolved_sampling_mode"] == "global"
    assert saved_summary["granularity_pattern_summary"] == {
        "pattern_type": "all_granularities",
        "selected_granularities": config["model"]["granularities"],
        "layer_count": config["model"]["num_layers"],
        "repeatable_source": [
            config["run"]["run_id"],
            "run.sampling_mode=nested-all",
            "model.granularity_sampling_mode=global",
        ],
    }
    assert saved_summary["granularity_pattern_provenance"] == saved_config["model"][
        "granularity_pattern_provenance"
    ]
    assert saved_summary["correction_context"]["local_correction_active"] is False

    with metrics_path.open("r", encoding="utf-8", newline="") as metrics_file:
        rows = list(csv.DictReader(metrics_file))
    assert [row["granularity"] for row in rows] == config["model"]["granularities"]
    assert {row["sampling_mode"] for row in rows} == {"nested-all"}
    assert {row["granularity_sampling_mode"] for row in rows} == {"global"}
    assert all(
        json.loads(row["granularity_pattern_summary"])["pattern_type"]
        == "all_granularities"
        for row in rows
    )
    assert all(
        json.loads(row["correction_context"])["local_correction_active"] is False
        for row in rows
    )


@pytest.mark.parametrize(
    "alias, expected_mode, expected_sampling_mode, expected_pattern_type, pattern_builder, layer_granularities",
    [
        (
            "all",
            "global",
            "nested-all",
            "all_granularities",
            build_global_granularity_pattern,
            None,
        ),
        (
            "random",
            "per_block",
            "nested-random",
            "per_block",
            build_per_block_granularity_pattern,
            ["s", "m"],
        ),
    ],
)
def test_artifacts_record_sampling_mode_and_pattern_provenance(
    tmp_path,
    alias,
    expected_mode,
    expected_sampling_mode,
    expected_pattern_type,
    pattern_builder,
    layer_granularities,
):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[f"training.granularity_sampling={alias}"],
    )

    if layer_granularities is None:
        runtime_pattern = pattern_builder(config)
    else:
        runtime_pattern = pattern_builder(config, layer_granularities)

    runtime_pattern_summary = json.loads(
        json.dumps(summarize_granularity_pattern(runtime_pattern))
    )
    correction_context = json.loads(
        json.dumps(
            summarize_correction_context(
                correction_context_from_config(
                    config,
                    granularity_pattern=runtime_pattern,
                )
            )
        )
    )

    config_path = write_config_artifact(config)
    summary = build_run_summary(
        config,
        tokens_seen=128,
        notes=["artifact provenance smoke"],
        extra_fields={
            "granularity_pattern_summary": runtime_pattern_summary,
            "correction_context": correction_context,
        },
    )
    summary_path = write_run_summary(output_dir, summary)

    metric_row = build_training_metric_row(
        config,
        step=1,
        granularity=config["model"]["granularities"][0],
        loss=1.25,
        tokens_seen=8,
        content_tokens_seen=8,
        wall_clock_seconds=2.0,
        peak_memory_bytes=512,
        granularity_pattern_summary=runtime_pattern_summary,
        correction_context=correction_context,
    )
    metrics_path = write_metrics_csv(output_dir, [metric_row])

    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_config["model"]["granularity_sampling_mode"] == expected_mode
    assert saved_config["model"]["requested_granularity_sampling_alias"] == alias
    assert saved_config["model"]["granularity_pattern_provenance"] == {
        "pattern_type": expected_pattern_type,
        "scope": "model",
        "source": "model.granularity_sampling_mode",
        "requested_alias": alias,
        "layer_count": config["model"]["num_layers"],
        "available_granularities": ["s", "m", "l", "xl"],
        "active_granularity": None,
    }

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["sampling_mode"] == expected_sampling_mode
    assert saved_summary["resolved_sampling_mode"] == expected_mode
    assert saved_summary["requested_granularity_sampling_alias"] == alias
    assert saved_summary["granularity_sampling_mode"] == expected_mode
    assert saved_summary["granularity_pattern_summary"] == runtime_pattern_summary
    assert saved_summary["correction_context"] == correction_context
    assert saved_summary["granularity_pattern_provenance"] == saved_config[
        "model"
    ]["granularity_pattern_provenance"]

    with metrics_path.open("r", encoding="utf-8", newline="") as metrics_file:
        metric_rows = list(csv.DictReader(metrics_file))
    assert len(metric_rows) == 1
    assert metric_rows[0]["sampling_mode"] == expected_sampling_mode
    assert metric_rows[0]["granularity_sampling_mode"] == expected_mode
    assert json.loads(metric_rows[0]["granularity_pattern_summary"]) == (
        runtime_pattern_summary
    )
    assert json.loads(metric_rows[0]["correction_context"]) == correction_context
    assert metric_rows[0]["granularity"] == config["model"]["granularities"][0]


def test_artifacts_reconstruct_standalone_mode_from_saved_files(tmp_path):
    output_dir = tmp_path / "debug-standalone-m-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-standalone-m-001",
        output_dir=output_dir,
    )

    config_path = write_config_artifact(config)
    summary = build_run_summary(
        config,
        tokens_seen=128,
        notes=["artifact reconstruction smoke"],
    )
    summary_path = write_run_summary(output_dir, summary)

    metric_row = build_training_metric_row(
        config,
        step=1,
        granularity="m",
        loss=1.25,
        tokens_seen=8,
        content_tokens_seen=8,
        wall_clock_seconds=2.0,
        peak_memory_bytes=512,
    )
    metrics_path = write_metrics_csv(output_dir, [metric_row])

    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert saved_config["run"]["sampling_mode"] == "standalone"
    assert saved_config["run"]["resolved_run_mode"] == "standalone"
    assert saved_config["run"]["granularity"] == "m"
    assert saved_config["model"]["granularities"] == ["m"]
    assert saved_config["model"]["granularity_sampling_mode"] == "global"
    assert saved_config["model"]["resolved_sampling_mode"] == "global"
    assert saved_config["model"]["granularity_pattern_provenance"] == {
        "pattern_type": "single",
        "scope": "model",
        "source": "model.granularity_sampling_mode",
        "requested_alias": None,
        "layer_count": config["model"]["num_layers"],
        "available_granularities": ["m"],
        "active_granularity": "m",
    }

    assert saved_summary["sampling_mode"] == "standalone"
    assert saved_summary["resolved_run_mode"] == "standalone"
    assert saved_summary["resolved_sampling_mode"] == "global"
    assert saved_summary["granularity_sampling_mode"] == "global"
    assert saved_summary["granularity_pattern_summary"] == {
        "pattern_type": "single",
        "selected_granularities": ["m"],
        "layer_count": config["model"]["num_layers"],
        "repeatable_source": [
            config["run"]["run_id"],
            "run.sampling_mode=standalone",
            "model.granularity_sampling_mode=global",
            "run.granularity=m",
        ],
    }
    assert saved_summary["granularity_pattern_provenance"] == saved_config["model"][
        "granularity_pattern_provenance"
    ]
    assert saved_summary["correction_context"]["local_correction_active"] is False

    with metrics_path.open("r", encoding="utf-8", newline="") as metrics_file:
        metric_rows = list(csv.DictReader(metrics_file))
    assert len(metric_rows) == 1
    assert metric_rows[0]["sampling_mode"] == "standalone"
    assert metric_rows[0]["resolved_run_mode"] == "standalone"
    assert metric_rows[0]["resolved_sampling_mode"] == "global"
    assert metric_rows[0]["granularity_sampling_mode"] == "global"
    assert metric_rows[0]["granularity"] == "m"
    assert json.loads(metric_rows[0]["granularity_pattern_summary"]) == (
        saved_summary["granularity_pattern_summary"]
    )


@pytest.mark.parametrize(
    "sampling_mode, pattern_builder, expected_pattern_type, expected_local_correction_active, expected_adaptive_sampler",
    [
        (
            "global",
            build_global_granularity_pattern,
            "single",
            False,
            None,
        ),
        (
            "per_block",
            build_per_block_granularity_pattern,
            "per_block",
            True,
            None,
        ),
        (
            "adaptive_per_block",
            build_per_block_granularity_pattern,
            "per_block",
            True,
            {
                "adaptive_sampler_strategy": "ucb",
                "adaptive_sampler_exploration_scale": 1.0,
                "adaptive_sampler_decay_rate": 0.0,
                "adaptive_sampler_reward_penalty_weight": 1.0,
            },
        ),
    ],
)
def test_artifacts_record_explicit_nested_random_global_per_block_and_adaptive_paths(
    tmp_path,
    sampling_mode,
    pattern_builder,
    expected_pattern_type,
    expected_local_correction_active,
    expected_adaptive_sampler,
):
    output_dir = tmp_path / "dmodel256-pilot-comparison-001"
    overrides = [f"model.granularity_sampling_mode={sampling_mode}"]
    if sampling_mode == "adaptive_per_block":
        overrides.append("model.adaptive_sampler_strategy=ucb")
    config = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        output_dir=output_dir,
        overrides=overrides,
    )

    if sampling_mode == "global":
        runtime_pattern = pattern_builder(config, granularities=("m",))
    else:
        runtime_pattern = pattern_builder(
            config,
            layer_granularities=config["model"]["granularities"]
            * (config["model"]["num_layers"] // len(config["model"]["granularities"])),
        )
    runtime_pattern_summary = json.loads(
        json.dumps(summarize_granularity_pattern(runtime_pattern))
    )
    correction_context = json.loads(
        json.dumps(
            summarize_correction_context(
                correction_context_from_config(
                    config,
                    granularity_pattern=runtime_pattern,
                )
            )
        )
    )

    config_path = write_config_artifact(config)
    summary = build_run_summary(
        config,
        tokens_seen=128,
        notes=["artifact provenance smoke"],
        extra_fields={
            "granularity_pattern_summary": runtime_pattern_summary,
            "correction_context": correction_context,
        },
    )
    summary_path = write_run_summary(output_dir, summary)

    metric_row = build_training_metric_row(
        config,
        step=1,
        granularity=runtime_pattern.selected_granularities[0],
        loss=1.25,
        tokens_seen=8,
        content_tokens_seen=8,
        wall_clock_seconds=2.0,
        peak_memory_bytes=512,
        granularity_pattern_summary=runtime_pattern_summary,
        correction_context=correction_context,
    )
    metrics_path = write_metrics_csv(output_dir, [metric_row])

    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_config["run"]["sampling_mode"] == "nested-random"
    assert saved_config["run"]["output_root"] == config["run"]["output_root"]
    assert saved_config["run"]["output_dir"] == str(output_dir)
    assert saved_config["model"]["correction_mode"] == "gmc"
    assert saved_config["model"]["membership_correction"] is True
    assert saved_config["model"]["granularity_sampling_mode"] == sampling_mode
    assert saved_config["model"]["resolved_sampling_mode"] == sampling_mode
    assert saved_config["model"]["granularity_pattern_provenance"] == {
        "pattern_type": expected_pattern_type,
        "scope": "model",
        "source": "model.granularity_sampling_mode",
        "requested_alias": None,
        "layer_count": config["model"]["num_layers"],
        "available_granularities": ["s", "m", "l", "xl"],
    }
    if expected_adaptive_sampler is None:
        assert "adaptive_sampler_strategy" not in saved_config["model"]
        assert "adaptive_sampler_exploration_scale" not in saved_config["model"]
        assert "adaptive_sampler_decay_rate" not in saved_config["model"]
        assert "adaptive_sampler_reward_penalty_weight" not in saved_config["model"]
    else:
        for field_name, expected_value in expected_adaptive_sampler.items():
            assert saved_config["model"][field_name] == expected_value

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["sampling_mode"] == "nested-random"
    assert saved_summary["resolved_run_mode"] == "nested-random"
    assert saved_summary["resolved_sampling_mode"] == sampling_mode
    assert saved_summary["granularity_sampling_mode"] == sampling_mode
    assert saved_summary["correction_mode"] == "gmc"
    assert saved_summary["membership_correction"] is True
    assert saved_summary["output_root"] == config["run"]["output_root"]
    assert saved_summary["output_dir"] == str(output_dir)
    assert saved_summary["granularity_pattern_summary"] == runtime_pattern_summary
    assert saved_summary["correction_context"] == correction_context
    assert saved_summary["granularity_pattern_provenance"] == saved_config[
        "model"
    ]["granularity_pattern_provenance"]
    assert saved_summary["granularity_pattern_summary"]["repeatable_source"][1] == (
        f"model.granularity_sampling_mode={sampling_mode}"
    )
    assert correction_context["local_correction_active"] is (
        expected_local_correction_active
    )

    with metrics_path.open("r", encoding="utf-8", newline="") as metrics_file:
        metric_rows = list(csv.DictReader(metrics_file))
    assert len(metric_rows) == 1
    assert metric_rows[0]["sampling_mode"] == "nested-random"
    assert metric_rows[0]["granularity_sampling_mode"] == sampling_mode
    assert json.loads(metric_rows[0]["granularity_pattern_summary"]) == (
        runtime_pattern_summary
    )
    assert json.loads(metric_rows[0]["correction_context"]) == correction_context


@pytest.mark.parametrize(
    "continuation_overrides, expected_state",
    [
        (
            [],
            {
                "status": "fresh",
                "latest_checkpoint_path": None,
                "last_completed_step": 0,
                "resume_count": 0,
            },
        ),
        (
            [
                "run.continuation.enabled=true",
                "run.continuation.status=resumed",
                "run.continuation.latest_checkpoint_path=/tmp/debug-nested-001/checkpoints/latest.pt",
                "run.continuation.last_completed_step=8",
                "run.continuation.resume_count=1",
            ],
            {
                "status": "resumed",
                "latest_checkpoint_path": "/tmp/debug-nested-001/checkpoints/latest.pt",
                "last_completed_step": 8,
                "resume_count": 1,
            },
        ),
        (
            [
                "run.continuation.enabled=true",
                "run.continuation.status=completed",
                "run.continuation.latest_checkpoint_path=/tmp/debug-nested-001/checkpoints/final.pt",
                "run.continuation.last_completed_step=16",
                "run.continuation.resume_count=2",
            ],
            {
                "status": "completed",
                "latest_checkpoint_path": "/tmp/debug-nested-001/checkpoints/final.pt",
                "last_completed_step": 16,
                "resume_count": 2,
            },
        ),
    ],
)
def test_run_summary_records_continuation_state_transitions(
    tmp_path,
    continuation_overrides,
    expected_state,
):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=continuation_overrides,
    )

    summary = build_run_summary(config, tokens_seen=128)
    summary_path = write_run_summary(output_dir, summary)

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["continuation_state"]["run_id"] == "debug-nested-001"
    assert saved_summary["continuation_state"]["output_dir"] == str(output_dir)
    assert saved_summary["continuation_state"]["status"] == expected_state["status"]
    assert (
        saved_summary["continuation_state"]["latest_checkpoint_path"]
        == expected_state["latest_checkpoint_path"]
    )
    assert (
        saved_summary["continuation_state"]["last_completed_step"]
        == expected_state["last_completed_step"]
    )
    assert (
        saved_summary["continuation_state"]["resume_count"]
        == expected_state["resume_count"]
    )
    assert saved_summary["latest_checkpoint_path"] == expected_state[
        "latest_checkpoint_path"
    ]


def test_warmup_run_summary_records_completion_and_transition_fields(tmp_path):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "training.max_steps=2",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.scheduler.kwargs.warmup_steps=0",
            "training.pre_nested_warmup.enabled=true",
            "training.pre_nested_warmup.duration=1",
            "training.pre_nested_warmup.unit=steps",
            "evaluation.validation=false",
        ],
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5]],
            "attention_mask": [[1, 1, 0], [1, 1, 1]],
        }
    )

    run_training(
        config,
        model=TinyExtractionModel(),
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    saved_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    saved_summary = json.loads(
        (output_dir / "run_summary.json").read_text(encoding="utf-8")
    )

    assert saved_config["training"]["pre_nested_warmup"] == {
        "enabled": True,
        "duration": 1,
        "unit": "steps",
        "policy": "full_only",
        "active": True,
        "completed": True,
        "completion_step": 1,
        "transition_reason": "warmup_duration_reached",
    }
    assert saved_summary["warmup_policy"] == {
        "enabled": True,
        "duration": 1,
        "unit": "steps",
        "policy": "full_only",
        "completed": True,
        "completion_step": 1,
        "transition_reason": "warmup_duration_reached",
    }
    assert saved_summary["warmup_completion_step"] == 1
    assert saved_summary["warmup_completed"] is True


def test_write_failed_run_summary_records_failure_note(tmp_path):
    output_dir = tmp_path / "debug-standalone-s-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-standalone-s-001",
        output_dir=output_dir,
    )

    summary_path = write_failed_run_summary(
        config,
        error_message="CUDA out of memory during debug smoke",
        tokens_seen=64,
    )

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["status"] == "failed"
    assert saved_summary["tokens_seen"] == 64
    assert saved_summary["model_variant"] == "slicing"
    assert saved_summary["notes"] == ["CUDA out of memory during debug smoke"]


def test_baseline_and_cat_run_summaries_share_schema_and_differ_by_variant(tmp_path):
    baseline_output_dir = tmp_path / "baseline" / "debug-nested-001"
    cat_output_dir = tmp_path / "cat" / "debug-nested-001"

    baseline_config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=baseline_output_dir,
    )
    cat_config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=cat_output_dir,
        overrides=["model.variant=concat"],
    )

    baseline_summary = build_run_summary(
        baseline_config,
        tokens_seen=128,
        notes=["baseline comparison smoke"],
    )
    cat_summary = build_run_summary(
        cat_config,
        tokens_seen=128,
        notes=["cat comparison smoke"],
    )

    assert set(baseline_summary) == set(cat_summary)
    assert baseline_summary["model_variant"] == "slicing"
    assert cat_summary["model_variant"] == "concat"
    assert baseline_summary["model_family"] == cat_summary["model_family"] == "nested"


def test_shared_family_folder_artifacts_can_be_read_directly_by_figures(tmp_path):
    output_root = tmp_path / "outputs"
    resolved_runs = {
        config["run"]["run_id"]: config
        for config in resolve_all_run_configs(
            "configs/debug_matrix.yaml",
            overrides=[f"run.output_root={output_root}"],
        )
    }

    standalone_runs = [
        resolved_runs["debug-standalone-s-001"],
        resolved_runs["debug-standalone-m-001"],
        resolved_runs["debug-standalone-l-001"],
    ]
    shared_output_groups = {
        config["run"]["output_group"] for config in standalone_runs
    }
    assert len(shared_output_groups) == 1

    for config in standalone_runs:
        write_config_artifact(config)
        granularity = config["model"]["granularities"][0]
        scaling_rows = build_scaling_result_rows(
            config,
            [
                {
                    "step": 1,
                    "split": "validation",
                    "granularity": granularity,
                    "loss": 1.0,
                    "perplexity": 2.0,
                }
            ],
            {
                granularity: {
                    "total_parameters": 1,
                    "embedding_parameters": 0,
                    "lm_head_parameters": 0,
                    "non_embedding_parameters": 1,
                }
            },
        )
        write_scaling_results_csv(
            output_root / config["run"]["output_group"],
            scaling_rows,
        )

    shared_group = next(iter(shared_output_groups))
    figure_paths = generate_figures(
        output_root / shared_group,
        tmp_path / "figures",
        refresh_counts=False,
    )
    figure_names = {path.name for path in figure_paths}

    assert "medium_trend_report.md" in figure_names
    assert {
        "loss_vs_size.png",
        "ppl_vs_size.png",
        "ppl_vs_size_nested_all_no_corrections.png",
        "ppl_vs_size_nested_random_no_corrections.png",
        "ppl_vs_size_nested_random_vs_nested_all_no_corrections.png",
    }.isdisjoint(figure_names)


def test_scaling_curve_label_prefers_correction_mode_when_available():
    labeled_row = {
        "sampling_mode": "nested-random",
        "model_family": "nested",
        "model_variant": "concat",
        "resolved_sampling_mode": "per_block",
        "correction_mode": "lmc",
    }
    legacy_row = {
        "sampling_mode": "nested-random",
        "model_family": "nested",
        "model_variant": "concat",
        "granularity_sampling_mode": "global",
        "membership_correction": True,
    }
    standalone_row = {
        "sampling_mode": "standalone",
        "model_family": "standalone",
        "model_variant": "slicing",
        "membership_correction": False,
    }

    assert scaling_curve_label(labeled_row) == (
        "nested-random / concat / per_block / lmc"
    )
    assert scaling_curve_label(legacy_row) == (
        "nested-random / concat / global / gmc"
    )
    assert scaling_curve_label(standalone_row) == "standalone"


def test_scaling_curve_display_label_makes_per_block_sampling_explicit():
    per_block_row = {
        "sampling_mode": "nested-random",
        "model_family": "nested",
        "model_variant": "concat",
        "resolved_sampling_mode": "per_block",
        "correction_mode": "lmc",
    }
    global_row = {
        "sampling_mode": "nested-random",
        "model_family": "nested",
        "model_variant": "concat",
        "granularity_sampling_mode": "global",
        "membership_correction": True,
    }

    assert scaling_curve_display_label([per_block_row]) == (
        "nested-random / concat / per_block sampling / lmc"
    )
    assert scaling_curve_display_label([global_row]) == "nested-random / concat / gmc"


def test_comparison_series_alias_and_color_presets_are_configurable():
    style = resolve_plot_style("nested_split_no_corrections")
    series_key = "nested-random / concat / gmc"

    assert style["figure_title_fontsize"] == 17
    assert style["subfigure_title_fontsize"] == 13
    assert style["legend_fontsize"] == 12
    assert style["comparison_linestyle"] == "-"
    assert style["comparison_markers_by_variant"] == {"slicing": "s", "concat": "o"}
    assert comparison_series_key(
        {
            "sampling_mode": "nested-random",
            "model_family": "nested",
            "model_variant": "concat",
            "correction_mode": "gmc",
            "resolved_sampling_mode": "global",
        }
    ) == series_key
    assert (
        comparison_series_key(
            {
                "sampling_mode": "nested-random",
                "model_family": "nested",
                "model_variant": "concat",
                "correction_mode": "gmc",
                "resolved_sampling_mode": "per_block",
            }
        )
        is None
    )
    assert comparison_series_key(
        {
            "sampling_mode": "nested-all",
            "model_family": "nested",
            "model_variant": "concat",
            "correction_mode": "lmc",
            "resolved_sampling_mode": "global",
        }
    ) == "nested-all / concat / lmc"
    assert resolve_series_alias(series_key, style) == "Concat/GMC"
    assert resolve_series_alias("nested-all / concat / lmc", style) == "Concat/LMC"
    style["series_aliases"][series_key] = "random concat with correction"
    style["series_colors"][series_key] = "tab:cyan"
    assert resolve_series_alias(series_key, style) == "random concat with correction"
    gmc_style = comparison_series_style(series_key, style)
    slicing_style = comparison_series_style(
        "nested-random / slicing / none / global",
        style,
    )
    assert gmc_style["color"] == blend_color_toward_white("tab:cyan", 0.2)
    assert gmc_style["linestyle"] == "-"
    assert gmc_style["marker"] == "o"
    assert slicing_style["linestyle"] == "-"
    assert slicing_style["marker"] == "s"
    lmc_style = comparison_series_style(
        "nested-all / concat / lmc",
        style,
    )
    assert lmc_style["linestyle"] == "-"
    assert lmc_style["marker"] == "o"


def test_scaling_curve_style_groups_family_colors_markers_and_shades():
    nested_all_concat_none_style = scaling_curve_style(
        [
            {
                "sampling_mode": "nested-all",
                "model_family": "nested",
                "model_variant": "concat",
                "correction_mode": "none",
            }
        ]
    )
    nested_all_concat_gmc_style = scaling_curve_style(
        [
            {
                "sampling_mode": "nested-all",
                "model_family": "nested",
                "model_variant": "concat",
                "membership_correction": True,
            }
        ]
    )
    nested_all_concat_lmc_style = scaling_curve_style(
        [
            {
                "sampling_mode": "nested-all",
                "model_family": "nested",
                "model_variant": "concat",
                "correction_mode": "lmc",
            }
        ]
    )
    nested_random_concat_none_style = scaling_curve_style(
        [
            {
                "sampling_mode": "nested-random",
                "model_family": "nested",
                "model_variant": "concat",
                "correction_mode": "none",
            }
        ]
    )
    nested_random_concat_gmc_style = scaling_curve_style(
        [
            {
                "sampling_mode": "nested-random",
                "model_family": "nested",
                "model_variant": "concat",
                "membership_correction": True,
            }
        ]
    )
    nested_all_slice_none_style = scaling_curve_style(
        [
            {
                "sampling_mode": "nested-all",
                "model_family": "nested",
                "model_variant": "slicing",
                "correction_mode": "none",
            }
        ]
    )
    nested_all_slice_gmc_style = scaling_curve_style(
        [
            {
                "sampling_mode": "nested-all",
                "model_family": "nested",
                "model_variant": "slicing",
                "membership_correction": True,
            }
        ]
    )
    nested_random_slice_none_style = scaling_curve_style(
        [
            {
                "sampling_mode": "nested-random",
                "model_family": "nested",
                "model_variant": "slicing",
                "correction_mode": "none",
            }
        ]
    )
    nested_random_slice_gmc_style = scaling_curve_style(
        [
            {
                "sampling_mode": "nested-random",
                "model_family": "nested",
                "model_variant": "slicing",
                "membership_correction": True,
            }
        ]
    )
    nested_random_concat_global_style = scaling_curve_style(
        [
            {
                "sampling_mode": "nested-random",
                "model_family": "nested",
                "model_variant": "concat",
                "resolved_sampling_mode": "global",
                "correction_mode": "none",
            }
        ]
    )
    nested_random_concat_per_block_style = scaling_curve_style(
        [
            {
                "sampling_mode": "nested-random",
                "model_family": "nested",
                "model_variant": "concat",
                "resolved_sampling_mode": "per_block",
                "correction_mode": "none",
            }
        ]
    )
    nested_random_slice_global_style = scaling_curve_style(
        [
            {
                "sampling_mode": "nested-random",
                "model_family": "nested",
                "model_variant": "slicing",
                "resolved_sampling_mode": "global",
                "correction_mode": "none",
            }
        ]
    )
    nested_random_slice_per_block_style = scaling_curve_style(
        [
            {
                "sampling_mode": "nested-random",
                "model_family": "nested",
                "model_variant": "slicing",
                "resolved_sampling_mode": "per_block",
                "correction_mode": "none",
            }
        ]
    )
    standalone_style = scaling_curve_style(
        [
            {
                "sampling_mode": "standalone",
                "model_family": "standalone",
                "model_variant": "slicing",
            }
        ]
    )

    assert scaling_curve_color_group_label(
        {
            "sampling_mode": "nested-random",
            "model_family": "nested",
            "model_variant": "slicing",
            "resolved_sampling_mode": "global",
        }
    ) == "nested-random / slicing / global"
    assert scaling_curve_color_group_label(
        {
            "sampling_mode": "nested-random",
            "model_family": "nested",
            "model_variant": "slicing",
            "resolved_sampling_mode": "per_block",
        }
    ) == "nested-random / slicing / per_block"
    assert scaling_curve_color_group_label(
        {
            "sampling_mode": "nested-random",
            "model_family": "nested",
            "model_variant": "concat",
            "resolved_sampling_mode": "global",
        }
    ) == "nested-random / concat / global"
    assert scaling_curve_color_group_label(
        {
            "sampling_mode": "nested-random",
            "model_family": "nested",
            "model_variant": "concat",
            "resolved_sampling_mode": "per_block",
        }
    ) == "nested-random / concat / per_block"
    assert scaling_curve_color_group_label(
        {
            "sampling_mode": "nested-all",
            "model_family": "nested",
            "model_variant": "slicing",
        }
    ) == "nested-all / slicing"
    assert scaling_curve_color_group_label(
        {
            "sampling_mode": "nested-all",
            "model_family": "nested",
            "model_variant": "concat",
        }
    ) == "nested-all / concat"
    assert scaling_curve_color_group_label(
        {
            "sampling_mode": "standalone",
            "model_family": "standalone",
            "model_variant": "slicing",
        }
    ) == "standalone"

    assert nested_random_slice_global_style["color"] == blend_color_toward_white(
        "tab:blue",
        0.0,
    )
    assert nested_random_slice_per_block_style["color"] == blend_color_toward_white(
        "tab:cyan",
        0.28,
    )
    assert nested_random_concat_global_style["color"] == blend_color_toward_white(
        "tab:orange",
        0.0,
    )
    assert nested_random_concat_per_block_style["color"] == blend_color_toward_white(
        "tab:red",
        0.28,
    )
    assert nested_all_slice_none_style["color"] == blend_color_toward_white(
        "tab:purple",
        0.0,
    )
    assert nested_all_concat_none_style["color"] == blend_color_toward_white(
        "tab:green",
        0.0,
    )
    assert standalone_style["color"] == blend_color_toward_white(
        STANDALONE_REFERENCE_COLOR,
        0.0,
    )

    assert standalone_style["color"] not in {
        nested_all_concat_none_style["color"],
        nested_all_slice_none_style["color"],
        nested_random_concat_global_style["color"],
        nested_random_concat_per_block_style["color"],
        nested_random_slice_global_style["color"],
        nested_random_slice_per_block_style["color"],
    }

    assert nested_all_concat_none_style["marker"] == "o"
    assert nested_all_concat_gmc_style["marker"] == "s"
    assert nested_all_concat_lmc_style["marker"] == "^"
    assert nested_random_concat_none_style["marker"] == "o"
    assert nested_random_concat_gmc_style["marker"] == "s"
    assert nested_all_slice_none_style["marker"] == "o"
    assert nested_all_slice_gmc_style["marker"] == "s"
    assert nested_random_slice_none_style["marker"] == "o"
    assert nested_random_slice_gmc_style["marker"] == "s"
    assert nested_random_concat_global_style["marker"] == "o"
    assert nested_random_concat_per_block_style["marker"] == "D"
    assert nested_random_slice_global_style["marker"] == "o"
    assert nested_random_slice_per_block_style["marker"] == "D"
    assert nested_random_concat_global_style["marker"] != nested_random_concat_per_block_style["marker"]

    assert nested_all_concat_none_style["linestyle"] == "-"
    assert nested_all_concat_gmc_style["linestyle"] == "--"
    assert nested_all_concat_lmc_style["linestyle"] == "-."
    assert nested_random_concat_none_style["linestyle"] == "-"
    assert nested_random_concat_gmc_style["linestyle"] == "--"
    assert nested_all_slice_none_style["linestyle"] == "-"
    assert nested_all_slice_gmc_style["linestyle"] == "--"
    assert nested_random_slice_none_style["linestyle"] == "-"
    assert nested_random_slice_gmc_style["linestyle"] == "--"
    assert standalone_style["linestyle"] == "-"


def test_run_summary_includes_budget_derived_fields(tmp_path):
    output_dir = tmp_path / "dmodel256-pilot-comparison-001"
    config = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        output_dir=output_dir,
    )

    summary = build_run_summary(config, tokens_seen=128, notes=["budget smoke"])

    for field_name in [
        "expected_tokens_per_step",
        "derived_max_steps",
        "effective_world_size",
        "stop_reason",
        "model_family_slug",
        "model_size_slug",
        "family_size_slug",
        "token_budget_slug",
        "output_group",
    ]:
        assert field_name in summary
    assert summary["expected_tokens_per_step"] == config["training"][
        "expected_tokens_per_step"
    ]
    assert summary["derived_max_steps"] == config["training"]["derived_max_steps"]
    assert summary["effective_world_size"] == config["training"][
        "effective_world_size"
    ]
    assert summary["stop_reason"] == "not_started"


def test_run_summary_records_resolved_schedule_and_optimizer_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    output_dir = tmp_path / "dmodel256-pilot-comparison-001"
    config = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        output_dir=output_dir,
        overrides=[
            "training.warmup_ratio=0.9",
            "training.warmup_steps=7",
            "training.optimizer.preset=null",
            "training.optimizer.name=sgd",
            "training.optimizer.kwargs.momentum=0.8",
            "training.optimizer.kwargs.nesterov=true",
        ],
    )

    summary = build_run_summary(config, tokens_seen=128, notes=["schedule smoke"])
    summary_path = write_run_summary(output_dir, summary)

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["base_learning_rate"] == 0.001
    assert saved_summary["learning_rate_scale_rule"] == "none"
    assert saved_summary["learning_rate_scale_factor"] == 1.0
    assert saved_summary["resolved_learning_rate"] == 0.001
    assert saved_summary["warmup_ratio"] == 0.9
    assert saved_summary["warmup_steps"] == 7
    assert saved_summary["resolved_warmup_steps"] == 7
    assert saved_summary["scheduler_warmup_steps"] == 7
    assert saved_summary["scheduler_resolved_warmup_steps"] == 7
    assert saved_summary["gradient_clip_norm"] == 1.0
    assert saved_summary["optimizer_name"] == "sgd"
    assert saved_summary["optimizer_kwargs"] == {
        "momentum": 0.8,
        "dampening": 0.0,
        "nesterov": True,
        "weight_decay": 0.0,
    }


def test_run_summary_schema_requires_budget_derived_fields(tmp_path):
    output_dir = tmp_path / "dmodel256-pilot-comparison-001"
    config = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        output_dir=output_dir,
    )
    summary = build_run_summary(
        config,
        tokens_seen=128,
        extra_fields={
            "expected_tokens_per_step": 8192,
            "derived_max_steps": 12208,
            "effective_world_size": 1,
            "stop_reason": "token_budget_reached",
        },
    )
    summary.pop("stop_reason")

    with pytest.raises(ArtifactError, match="stop_reason"):
        write_run_summary(output_dir, summary)


def _checkpoint_summary_builder():
    import src.utils.metrics as metrics

    builder = getattr(metrics, "build_checkpoint_summary_fields", None)
    assert (
        builder is not None
    ), "utils.metrics.build_checkpoint_summary_fields is required"
    return builder


def test_best_eval_checkpoint_summary_selects_lowest_validation_loss(tmp_path):
    output_dir = tmp_path / "dmodel256-pilot-comparison-001"
    config = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        output_dir=output_dir,
    )
    metrics_rows = [
        {
            "run_id": "dmodel256-pilot-comparison-001",
            "step": 100,
            "split": "validation",
            "granularity": "xl",
            "loss": 2.0,
            "perplexity": 7.39,
        },
        {
            "run_id": "dmodel256-pilot-comparison-001",
            "step": 200,
            "split": "validation",
            "granularity": "xl",
            "loss": 1.5,
            "perplexity": 4.48,
        },
        {
            "run_id": "dmodel256-pilot-comparison-001",
            "step": 300,
            "split": "validation",
            "granularity": "xl",
            "loss": 1.8,
            "perplexity": 6.05,
        },
    ]

    fields = _checkpoint_summary_builder()(
        config,
        metrics_rows,
        validation_enabled=True,
        save_checkpoints=True,
    )

    assert fields["checkpoint_status"] == "best_eval"
    assert fields["checkpoint_metric"] == "validation_loss"
    assert fields["checkpoint_metric_value"] == 1.5
    assert fields["checkpoint_selection_step"] == 200
    assert fields["best_checkpoint_path"] == str(
        output_dir / "checkpoints" / "best_eval_step_200.pt"
    )
    assert fields["final_checkpoint_path"] is None

    summary = build_run_summary(config, tokens_seen=1024, extra_fields=fields)
    for field_name in [
        "checkpoint_status",
        "best_checkpoint_path",
        "final_checkpoint_path",
        "checkpoint_metric",
    ]:
        assert field_name in summary


def test_final_checkpoint_summary_when_validation_is_disabled(tmp_path):
    output_dir = tmp_path / "dmodel256-pilot-comparison-001"
    config = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        output_dir=output_dir,
        overrides=["evaluation.validation=false"],
    )

    fields = _checkpoint_summary_builder()(
        config,
        metrics_rows=[],
        validation_enabled=False,
        save_checkpoints=True,
    )

    assert fields["checkpoint_status"] == "final"
    assert fields["best_checkpoint_path"] is None
    assert fields["final_checkpoint_path"] == str(
        output_dir / "checkpoints" / "final.pt"
    )
    assert fields["checkpoint_metric"] is None
    assert fields["checkpoint_unavailable_reason"] is None


def test_no_checkpoint_summary_when_checkpoint_writes_are_disabled(tmp_path):
    output_dir = tmp_path / "dmodel256-pilot-comparison-001"
    config = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        output_dir=output_dir,
        overrides=[
            "evaluation.validation=false",
            "outputs.save_checkpoints=false",
        ],
    )

    fields = _checkpoint_summary_builder()(
        config,
        metrics_rows=[],
        validation_enabled=False,
        save_checkpoints=False,
    )

    assert fields["checkpoint_status"] == "none"
    assert fields["best_checkpoint_path"] is None
    assert fields["final_checkpoint_path"] is None
    assert fields["checkpoint_metric"] is None
    assert "disabled" in fields["checkpoint_unavailable_reason"]


def test_rank_zero_only_shared_artifact_helper_writes_on_rank_zero(tmp_path):
    from src.training.distributed import DistributedContext, rank_zero_only

    context = DistributedContext(
        enabled=True,
        rank=0,
        local_rank=0,
        world_size=2,
        strategy="fsdp",
        device="cpu",
    )
    artifact_path = tmp_path / "rank-zero-artifact.json"
    calls = []

    def write_artifact():
        calls.append("write")
        artifact_path.write_text('{"status": "written"}\n', encoding="utf-8")
        return artifact_path

    result = rank_zero_only(context, write_artifact)

    assert result == artifact_path
    assert calls == ["write"]
    assert artifact_path.exists()


def test_rank_zero_only_shared_artifact_helper_skips_nonzero_rank(tmp_path):
    from src.training.distributed import DistributedContext, rank_zero_only

    context = DistributedContext(
        enabled=True,
        rank=1,
        local_rank=1,
        world_size=2,
        strategy="fsdp",
        device="cpu",
    )
    artifact_path = tmp_path / "nonzero-rank-artifact.json"
    calls = []

    def write_artifact():
        calls.append("write")
        artifact_path.write_text('{"status": "written"}\n', encoding="utf-8")
        return artifact_path

    result = rank_zero_only(context, write_artifact)

    assert result is None
    assert calls == []
    assert not artifact_path.exists()


def test_write_all_csv_artifact_types(tmp_path):
    output_dir = tmp_path / "debug-nested-001"

    task_path = write_task_results_csv(
        output_dir,
        {
            "run_id": "debug-nested-001",
            "suite_id": "debug-downstream",
            "task": "hellaswag",
            "model_family": "nested",
            "model_size_label": "debug",
            "granularity": "s",
            "metric_name": "accuracy",
            "metric_value": 0.25,
        },
    )
    scaling_path = write_scaling_results_csv(
        output_dir,
        {
            "comparison_id": "debug-s",
            "run_id": "debug-nested-001",
            "model_family": "nested",
            "model_size_label": "debug",
            "model_family_slug": "matformer_llama",
            "model_size_slug": "9m",
            "token_budget_slug": "1m_tokens",
            "output_group": "matformer_llama_9m_1m_tokens",
            "completion_label": "debug",
            "granularity": "s",
            "total_parameters": 1000,
            "embedding_parameters": 100,
            "lm_head_parameters": 100,
            "non_embedding_parameters": 800,
            "loss": 2.1,
            "perplexity": 8.17,
            "average_downstream_accuracy": 0.25,
        },
    )
    consistency_path = write_consistency_results_csv(
        output_dir,
        {
            "comparison_id": "debug-s-xl",
            "small_run_id": "debug-nested-001",
            "large_run_id": "debug-nested-001",
            "small_granularity": "s",
            "large_granularity": "xl",
            "metric_name": "argmax_agreement",
            "metric_value": 0.72,
            "sample_count": 16,
        },
    )

    for artifact_path in [task_path, scaling_path, consistency_path]:
        with artifact_path.open("r", encoding="utf-8", newline="") as artifact_file:
            rows = list(csv.DictReader(artifact_file))
        assert len(rows) == 1


def test_build_consistency_result_rows_normalizes_top_k_and_deferred_metrics():
    rows = build_consistency_result_rows(
        [
            {
                "comparison_id": "debug-s-xl",
                "small_run_id": "debug-nested-001",
                "large_run_id": "debug-nested-001",
                "small_granularity": "s",
                "large_granularity": "xl",
                "metric_name": "top_k_overlap",
                "metric_value": 0.75,
                "sample_count": 16,
                "top_k": 5,
            },
            {
                "comparison_id": "debug-s-xl",
                "small_run_id": "debug-nested-001",
                "large_run_id": "debug-nested-001",
                "small_granularity": "s",
                "large_granularity": "xl",
                "metric_name": "kl_divergence",
                "metric_value": None,
                "sample_count": 16,
                "deferred": True,
                "deferred_reason": "later phase",
            },
        ]
    )

    assert rows == [
        {
            "comparison_id": "debug-s-xl",
            "small_run_id": "debug-nested-001",
            "large_run_id": "debug-nested-001",
            "small_granularity": "s",
            "large_granularity": "xl",
            "metric_name": "top_k_overlap@5",
            "metric_value": 0.75,
            "sample_count": 16,
        },
        {
            "comparison_id": "debug-s-xl",
            "small_run_id": "debug-nested-001",
            "large_run_id": "debug-nested-001",
            "small_granularity": "s",
            "large_granularity": "xl",
            "metric_name": "kl_divergence_deferred",
            "metric_value": None,
            "sample_count": 16,
        },
    ]


def test_write_consistency_results_csv_preserves_normalized_metric_names(tmp_path):
    output_dir = tmp_path / "consistency-001"

    artifact_path = write_consistency_results_csv(
        output_dir,
        [
            {
                "comparison_id": "debug-s-xl",
                "small_run_id": "debug-nested-001",
                "large_run_id": "debug-nested-001",
                "small_granularity": "s",
                "large_granularity": "xl",
                "metric_name": "top_k_overlap",
                "metric_value": 0.75,
                "sample_count": 16,
                "top_k": 5,
            },
            {
                "comparison_id": "debug-s-xl",
                "small_run_id": "debug-nested-001",
                "large_run_id": "debug-nested-001",
                "small_granularity": "s",
                "large_granularity": "xl",
                "metric_name": "kl_divergence",
                "metric_value": None,
                "sample_count": 16,
                "deferred": True,
            },
        ],
    )

    with artifact_path.open("r", encoding="utf-8", newline="") as artifact_file:
        rows = list(csv.DictReader(artifact_file))

    assert [row["metric_name"] for row in rows] == [
        "top_k_overlap@5",
        "kl_divergence_deferred",
    ]
    assert rows[1]["metric_value"] == ""


def test_build_scaling_rows_uses_latest_validation_metrics():
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
    )
    metrics_rows = [
        {
            "run_id": "debug-nested-001",
            "step": 1,
            "split": "validation",
            "model_family": "nested",
            "model_size_label": "debug",
            "granularity": "s",
            "loss": 2.5,
            "perplexity": 12.18,
            "tokens_seen": 32,
            "wall_clock_seconds": 1.0,
            "tokens_per_second": 32.0,
            "peak_memory_bytes": 0,
        },
        {
            "run_id": "debug-nested-001",
            "step": 2,
            "split": "validation",
            "model_family": "nested",
            "model_size_label": "debug",
            "granularity": "s",
            "loss": 2.0,
            "perplexity": 7.39,
            "tokens_seen": 64,
            "wall_clock_seconds": 2.0,
            "tokens_per_second": 32.0,
            "peak_memory_bytes": 0,
        },
    ]
    for granularity in ["m", "l", "xl"]:
        row = dict(metrics_rows[-1])
        row["granularity"] = granularity
        metrics_rows.append(row)

    parameter_counts = {
        granularity: {
            "total_parameters": index * 1000,
            "embedding_parameters": 100,
            "lm_head_parameters": 100,
            "non_embedding_parameters": index * 1000 - 200,
        }
        for index, granularity in enumerate(["s", "m", "l", "xl"], start=1)
    }

    rows = build_scaling_result_rows(config, metrics_rows, parameter_counts)

    assert [row["granularity"] for row in rows] == ["s", "m", "l", "xl"]
    assert rows[0]["comparison_id"] == "debug-nested-001__s"
    assert rows[0]["loss"] == 2.0
    assert rows[0]["non_embedding_parameters"] == 800


def test_scaling_result_schema_exposes_phase5_reporting_fields(tmp_path):
    output_dir = tmp_path / "dmodel256-pilot-comparison-001"
    config = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        output_dir=output_dir,
    )
    metrics_rows = [
        {
            "run_id": "dmodel256-pilot-comparison-001",
            "step": 10,
            "split": "validation",
            "model_family": "nested",
            "model_size_label": "dmodel256",
            "sampling_mode": "nested-random",
            "model_shape_label": "dmodel256",
            "granularity": granularity,
            "loss": 2.0 + index * 0.1,
            "perplexity": 7.0 + index,
            "tokens_seen": 81920,
            "content_tokens_seen": 80000,
            "wall_clock_seconds": 20.0,
            "tokens_per_second": 4096.0,
            "peak_memory_bytes": 1024,
        }
        for index, granularity in enumerate(["s", "m", "l", "xl"])
    ]
    parameter_counts = {
        granularity: {
            "total_parameters": 1000 + index,
            "embedding_parameters": 100,
            "lm_head_parameters": 100,
            "non_embedding_parameters": 800 + index,
            "ffn_parameters": 400 + index,
            "attention_parameters": 200,
            "other_non_embedding_parameters": 200 + index,
            "lm_head_counting": "separately_counted",
        }
        for index, granularity in enumerate(["s", "m", "l", "xl"])
    }

    rows = build_scaling_result_rows(config, metrics_rows, parameter_counts)
    scaling_path = write_scaling_results_csv(output_dir, rows)

    with scaling_path.open("r", encoding="utf-8", newline="") as scaling_file:
        reader = csv.DictReader(scaling_file)
        assert reader.fieldnames == SCALING_RESULTS_COLUMNS
        saved_rows = list(reader)

    assert len(saved_rows) == 4
    row = saved_rows[0]
    for field_name in [
        "comparison_id",
        "run_id",
        "model_family",
        "model_size_label",
        "sampling_mode",
        "model_shape_label",
        "model_family_slug",
        "model_size_slug",
        "token_budget_slug",
        "output_group",
        "completion_label",
        "granularity",
        "d_model",
        "num_layers",
        "num_attention_heads",
        "context_length",
        "vocab_size",
        "token_budget",
        "effective_world_size",
        "total_parameters",
        "embedding_parameters",
        "lm_head_parameters",
        "non_embedding_parameters",
        "ffn_parameters",
        "attention_parameters",
        "other_non_embedding_parameters",
        "lm_head_counting",
        "checkpoint_path",
        "loss",
        "perplexity",
        "average_downstream_accuracy",
    ]:
        assert field_name in row

    assert row["sampling_mode"] == "nested-random"
    assert row["model_shape_label"] == "dmodel256"
    assert row["token_budget"] == "100000000"
    assert row["effective_world_size"] == "1"
    assert row["lm_head_counting"] == "separately_counted"


def test_append_metrics_keeps_one_header(tmp_path):
    output_dir = tmp_path / "debug-nested-001"
    first_row = {
        "run_id": "debug-nested-001",
        "step": 0,
        "split": "validation",
        "model_family": "nested",
        "model_size_label": "debug",
        "granularity": "s",
        "loss": 2.1,
        "perplexity": 8.17,
        "tokens_seen": 128,
        "wall_clock_seconds": 1.5,
        "tokens_per_second": 85.3,
        "peak_memory_bytes": 2048,
    }
    second_row = dict(first_row)
    second_row["step"] = 1
    second_row["tokens_seen"] = 256

    metrics_path = write_metrics_csv(output_dir, first_row)
    write_metrics_csv(output_dir, second_row, append=True)

    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("run_id,step,split")
    assert sum(1 for line in lines if line.startswith("run_id,step,split")) == 1
    assert len(lines) == 3


def test_metric_writer_rejects_missing_required_fields(tmp_path):
    with pytest.raises(ArtifactError, match="peak_memory_bytes"):
        write_metrics_csv(
            tmp_path / "debug-nested-001",
            {
                "run_id": "debug-nested-001",
                "step": 0,
                "split": "validation",
                "model_family": "nested",
                "model_size_label": "debug",
                "granularity": "s",
                "loss": 2.1,
                "perplexity": 8.17,
                "tokens_seen": 128,
                "wall_clock_seconds": 1.5,
                "tokens_per_second": 85.3,
            },
        )


def test_nested_run_writes_extraction_metadata_artifact(tmp_path):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "training.max_steps=1",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.scheduler.kwargs.warmup_steps=0",
        ],
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5]],
            "attention_mask": [[1, 1, 0], [1, 1, 1]],
        }
    )

    run_training(
        config,
        model=TinyExtractionModel(),
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    metadata_path = output_dir / "extraction_metadata.json"
    assert metadata_path.exists()

    summary_path = output_dir / "run_summary.json"
    metrics_path = output_dir / "metrics.csv"

    with metrics_path.open("r", encoding="utf-8", newline="") as metrics_file:
        metrics_reader = csv.DictReader(metrics_file)
        assert metrics_reader.fieldnames == METRICS_COLUMNS
        metric_rows = list(metrics_reader)

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["status"] == "completed"
    assert saved_summary["metrics_path"] == str(metrics_path)
    assert saved_summary["scaling_results_path"] == str(
        output_dir / "scaling_results.csv"
    )
    assert saved_summary["extraction_metadata_path"] == str(metadata_path)
    assert saved_summary["checkpoint_status"] == "none"
    assert saved_summary["checkpoint_unavailable_reason"] == "checkpoint writes disabled"
    assert saved_summary["granularity_pattern_summary"]["repeatable_source"][0] == (
        "debug-nested-001"
    )
    train_rows = [row for row in metric_rows if row["split"] == "train"]
    assert train_rows
    assert train_rows[0]["run_id"] == "debug-nested-001"
    assert train_rows[0]["split"] == "train"
    assert json.loads(train_rows[0]["granularity_pattern_summary"])[
        "repeatable_source"
    ][0] == "debug-nested-001"
    assert "local_correction_active" in json.loads(
        train_rows[0]["correction_context"]
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["run_id"] == "debug-nested-001"
    assert metadata["model_family"] == "nested"

    granularities = metadata["granularities"]
    assert [entry["granularity"] for entry in granularities] == ["s", "m", "l", "xl"]
    assert [entry["display_name"] for entry in granularities] == ["S", "M", "L", "XL"]
    assert [entry["prefix_width"] for entry in granularities] == [8, 16, 32, 64]
    assert granularities[0]["strict_prefix_of"] == ["m", "l", "xl"]
    assert granularities[-1]["strict_prefix_of"] == []


PROBABILISTIC_CHECKPOINT_PHASES = (
    "initial_objective_pending",
    "ready_for_action",
    "active_window",
    "boundary_evaluation_pending",
    "terminal_incomplete",
    "failed",
)


def _probabilistic_checkpoint_state(phase):
    generator = torch.Generator(device="cpu").manual_seed(731)
    generator_state = generator.get_state()
    action = {
        "scope": "global",
        "global_granularity": "micro",
        "block_granularities": ["micro", "micro"],
        "feature_vector": [1.0, 0.7071067811865476, 0.4082482904638631],
        "sampled_predicted_reward": 0.125,
        "tie_resolution": "resolved_granularity_order",
        "selection_round": 0,
        "selection_source": "thompson",
    }
    window = {
        "phase": phase,
        "window_index": 0,
        "decision_interval_steps": 2,
        "boundary_step": 0,
        "current_action": action,
        "selection_source": "thompson",
        "completed_optimizer_steps": 1,
        "pre_window_objective": 10.0,
        "ordered_pre_window_component_losses": [9.0, 10.0, 11.0],
        "boundary_evaluation_status": "not_started",
        "terminal_status": "continuing",
    }
    if phase == "initial_objective_pending":
        window.update(
            current_action=None,
            selection_source=None,
            completed_optimizer_steps=0,
            pre_window_objective=None,
            ordered_pre_window_component_losses=None,
        )
    elif phase == "ready_for_action":
        window.update(
            current_action=None,
            selection_source=None,
            completed_optimizer_steps=0,
        )
    elif phase == "boundary_evaluation_pending":
        window.update(
            completed_optimizer_steps=2,
            boundary_evaluation_status="pending",
        )
    elif phase == "terminal_incomplete":
        window.update(
            window_index=1,
            boundary_step=2,
            pre_window_objective=8.0,
            ordered_pre_window_component_losses=[7.0, 8.0, 9.0],
            terminal_status="incomplete",
        )
    elif phase == "failed":
        window.update(
            window_index=1,
            boundary_step=2,
            completed_optimizer_steps=2,
            boundary_evaluation_status="failed",
            terminal_status="failed",
        )

    return {
        "schema_version": 2,
        "method_family": "bayesian_gaussian_linear_thompson",
        "method_version": 1,
        "strategy": "thompson",
        "scope": "global",
        "ordered_granularities": ["micro", "medium", "full"],
        "block_count": 2,
        "feature_schema": {
            "schema_version": 1,
            "scope": "global",
            "encoding": "intercept_plus_sum_to_zero_contrasts",
            "dimension": 3,
            "coefficient_names": [
                "intercept",
                "global_contrast_0",
                "global_contrast_1",
            ],
            "schema_hash": "feature-schema-hash",
        },
        "probabilistic_inputs": {
            "resolved_prior_mean": [0.0, 0.0, 0.0],
            "resolved_prior_covariance": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "observation_noise_variance": 0.01,
            "resolved_process_noise_covariance": [
                [0.0001, 0.0, 0.0],
                [0.0, 0.0001, 0.0],
                [0.0, 0.0, 0.0001],
            ],
            "transition_model": "identity",
            "context_model": "intercept_only",
            "compute_weight": 0.0,
            "switch_weight": 0.0,
        },
        "manifest_hashes": {
            "data_roles_manifest_hash": "parent-manifest-hash",
            "optimizer_training_manifest_hash": "training-manifest-hash",
            "controller_manifest_hash": "controller-manifest-hash",
            "ordinary_validation_manifest_hash": "validation-manifest-hash",
            "final_holdout_manifest_hash": "final-holdout-manifest-hash",
        },
        "belief": {
            "round_index": 1,
            "posterior_mean": torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64),
            "posterior_covariance": torch.tensor(
                [
                    [0.8, 0.1, 0.0],
                    [0.1, 0.7, 0.05],
                    [0.0, 0.05, 0.9],
                ],
                dtype=torch.float64,
            ),
            "predictive_mean": torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64),
            "predictive_covariance": torch.tensor(
                [
                    [0.8001, 0.1, 0.0],
                    [0.1, 0.7001, 0.05],
                    [0.0, 0.05, 0.9001],
                ],
                dtype=torch.float64,
            ),
            "last_prediction_step": 0,
            "last_update_step": None,
        },
        "sampling": {
            "seed_stream_name": "posterior_sampling",
            "resolved_seed": 731,
            "generator_state": generator_state,
            "sample_count": 1,
            "factorization_contract": "symmetric_eigh_float64_v1",
        },
        "reset": {
            "contract": {
                "enabled": False,
                "interval_steps": None,
                "policy": "full_prior",
                "acquisition_policy": "balanced_global",
                "acquisition_passes": 1,
                "schedule_seed_stream_name": "controller_reset_schedule",
                "schedule_seed": derive_seed(42, "controller_reset_schedule"),
            },
            "enabled": False,
            "controller_start_step": None,
            "episode_index": None,
            "episode_start_step": None,
            "episode_end_step": None,
            "episode_offset_steps": 0,
            "reset_count": 0,
            "reset_steps": [],
            "acquisition_completed_windows": 0,
            "acquisition_total_windows": 0,
            "acquisition_counts": {"micro": 0, "medium": 0, "full": 0},
            "selection_source": None,
            "schedule_seed": None,
            "schedule": [],
            "schedule_hash": None,
            "completed_episode_count": 0,
            "completed_episodes": [],
        },
        "window": window,
        "journal": {
            "path": "controller_metrics.jsonl",
            "event_count": 1,
            "last_committed_offset": 1024,
            "last_committed_hash": "journal-event-hash",
        },
        "resume": {
            "resume_count": 0,
            "source_checkpoint": None,
            "compatibility_status": "fresh",
        },
        "failure": (
            {
                "stage": "controller_evaluation",
                "error_category": "non_finite_objective",
                "posterior_updated": False,
                "new_action_selected": False,
            }
            if phase == "failed"
            else None
        ),
    }


def _probabilistic_checkpoint_config(tmp_path):
    config = resolve_run_config(
        "tests/fixtures/probabilistic_adaptive_global_smoke.yaml",
        output_dir=tmp_path / "probabilistic-adaptive-global-smoke-001",
        overrides={"training.eval_batches": 4},
    )
    config.update(
        {
            "data_roles_manifest_hash": "parent-manifest-hash",
            "optimizer_training_manifest_hash": "training-manifest-hash",
            "controller_manifest_hash": "controller-manifest-hash",
            "validation_manifest_hash": "validation-manifest-hash",
            "final_holdout_manifest_hash": "final-holdout-manifest-hash",
        }
    )
    return config


@pytest.mark.parametrize("phase", PROBABILISTIC_CHECKPOINT_PHASES)
def test_probabilistic_controller_checkpoint_persists_complete_state_for_every_phase(
    tmp_path,
    phase,
):
    import src.training.checkpointing as training_checkpointing

    config = _probabilistic_checkpoint_config(tmp_path)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    run_state = training_checkpointing.build_initial_continuation_state(config)
    expected_controller_state = _probabilistic_checkpoint_state(phase)
    run_state.update(
        {
            "step": 1,
            "last_completed_step": 1,
            "probabilistic_controller_state": expected_controller_state,
        }
    )
    checkpoint_path = tmp_path / phase / "checkpoint.pt"

    training_checkpointing.save_model_checkpoint(
        config,
        model,
        optimizer,
        scheduler=None,
        output_path=checkpoint_path,
        checkpoint_fields={
            "checkpoint_status": "latest",
            "checkpoint_metric": None,
            "checkpoint_metric_value": None,
            "checkpoint_selection_step": None,
        },
        run_state=run_state,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    persisted = checkpoint["probabilistic_controller_state"]
    assert persisted["window"]["phase"] == phase
    assert persisted["method_family"] == "bayesian_gaussian_linear_thompson"
    assert persisted["method_version"] == 1
    assert persisted["strategy"] == "thompson"
    assert persisted["scope"] == "global"
    assert persisted["feature_schema"] == expected_controller_state[
        "feature_schema"
    ]
    assert persisted["probabilistic_inputs"] == expected_controller_state[
        "probabilistic_inputs"
    ]
    assert persisted["manifest_hashes"] == expected_controller_state[
        "manifest_hashes"
    ]
    assert persisted["window"] == expected_controller_state["window"]
    assert persisted["journal"] == expected_controller_state["journal"]
    assert persisted["resume"] == expected_controller_state["resume"]
    assert persisted["failure"] == expected_controller_state["failure"]
    for field_name in (
        "posterior_mean",
        "posterior_covariance",
        "predictive_mean",
        "predictive_covariance",
    ):
        assert torch.equal(
            persisted["belief"][field_name],
            expected_controller_state["belief"][field_name],
        )
    assert persisted["sampling"]["sample_count"] == 1
    assert persisted["sampling"]["resolved_seed"] == 731
    assert torch.equal(
        persisted["sampling"]["generator_state"],
        expected_controller_state["sampling"]["generator_state"],
    )


def _controller_journal_events():
    common = {
        "schema_version": 1,
        "run_id": "probabilistic-adaptive-global-smoke-001",
        "method_family": "bayesian_gaussian_linear_thompson",
        "method_version": 1,
        "strategy": "thompson",
        "scope": "global",
        "ordered_granularities": ["micro", "medium", "full"],
        "feature_schema_hash": "feature-schema-hash",
        "controller_manifest_hash": "controller-manifest-hash",
        "data_roles_manifest_hash": "parent-manifest-hash",
        "decision_interval_steps": 2,
        "resume_count": 0,
        "resume_source_checkpoint": None,
    }
    initial = {
        **common,
        "event_type": "initial_boundary",
        "boundary_step": 0,
        "window_index": 0,
        "ordered_component_losses": [9.0, 10.0, 11.0],
        "controller_objective": 10.0,
        "predictive_mean": [0.0, 0.0, 0.0],
        "predictive_covariance": [
            [1.0001, 0.0, 0.0],
            [0.0, 1.0001, 0.0],
            [0.0, 0.0, 1.0001],
        ],
        "selected_action": {"global_granularity": "micro"},
        "sample_count": 1,
    }
    completed = {
        **common,
        "event_type": "completed_window",
        "boundary_step": 2,
        "window_index": 0,
        "boundary_step_start": 0,
        "boundary_step_end": 2,
        "completed_optimizer_steps": 2,
        "action": {"global_granularity": "micro"},
        "pre_window_objective": 10.0,
        "post_window_objective": 8.0,
        "ordered_component_losses": [7.0, 8.0, 9.0],
        "reward": 1.0,
        "predicted_reward": 0.0,
        "prediction_error": 1.0,
        "predictive_mean": [0.0, 0.0, 0.0],
        "predictive_covariance": [
            [1.0001, 0.0, 0.0],
            [0.0, 1.0001, 0.0],
            [0.0, 0.0, 1.0001],
        ],
        "gain_vector": [0.5, 0.25, 0.0],
        "posterior_mean": [0.5, 0.25, 0.0],
        "posterior_covariance": [
            [0.5, -0.25, 0.0],
            [-0.25, 0.875, 0.0],
            [0.0, 0.0, 1.0001],
        ],
        "action_frequencies": {"micro": 1, "medium": 0, "full": 0},
        "uncertainty_summary": {"mean_posterior_stddev": 0.89},
    }
    incomplete = {
        **common,
        "event_type": "terminal_incomplete",
        "boundary_step": 3,
        "window_index": 1,
        "action": {"global_granularity": "medium"},
        "completed_optimizer_steps": 1,
        "pre_window_objective": 8.0,
        "observation_emitted": False,
        "sample_count": 2,
    }
    failure = {
        **common,
        "event_type": "controller_failure",
        "boundary_step": 4,
        "window_index": 1,
        "failing_stage": "controller_evaluation",
        "error_category": "non_finite_objective",
        "last_valid_phase": "boundary_evaluation_pending",
        "belief_hash": "last-valid-belief-hash",
        "journal_position": 3,
        "posterior_updated": False,
        "new_action_selected": False,
        "offending_field": "uniform_objective",
    }
    return [initial, completed, incomplete, failure]


def test_probabilistic_controller_journal_is_append_only_and_transactional(
    tmp_path,
):
    from src.utils.metrics import append_controller_event

    journal_path = tmp_path / "controller_metrics.jsonl"
    events = _controller_journal_events()
    for event in events:
        append_controller_event(journal_path, event)

    saved_events = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
    ]
    assert saved_events == events
    assert [event["event_type"] for event in saved_events] == [
        "initial_boundary",
        "completed_window",
        "terminal_incomplete",
        "controller_failure",
    ]
    assert "reward" not in saved_events[0]
    assert saved_events[1]["reward"] == pytest.approx(1.0)
    assert saved_events[2]["observation_emitted"] is False
    assert saved_events[3]["posterior_updated"] is False
    assert saved_events[3]["new_action_selected"] is False
    assert "posterior_mean" not in saved_events[3]

    invalid_failure = dict(events[-1], posterior_updated=True)
    with pytest.raises(ArtifactError, match="failure.*posterior_updated"):
        append_controller_event(journal_path, invalid_failure)
    assert len(journal_path.read_text(encoding="utf-8").splitlines()) == 4


def test_same_boundary_controller_events_append_as_one_transaction(tmp_path):
    from src.utils.metrics import append_controller_events

    journal_path = tmp_path / "controller_metrics.jsonl"
    completed = _controller_journal_events()[1]
    episode_completed = {
        **{key: value for key, value in completed.items() if key != "event_type"},
        "event_type": "episode_completed",
        "episode_index": 0,
        "episode_start_step": 0,
        "episode_end_step": 2,
        "episode_offset_steps": 2,
        "completed_window_count": 1,
        "forced_acquisition_window_count": 1,
        "thompson_window_count": 0,
        "schedule_seed": 17,
        "schedule": ["micro", "medium", "full"],
        "schedule_hash": "schedule-hash",
        "pre_reset_posterior_mean": [0.5, 0.25, 0.0],
        "pre_reset_posterior_covariance": [
            [0.5, -0.25, 0.0],
            [-0.25, 0.875, 0.0],
            [0.0, 0.0, 1.0001],
        ],
    }

    commit = append_controller_events(
        journal_path,
        [completed, episode_completed],
    )

    assert commit["event_count"] == 2
    assert len(commit["event_hashes"]) == 2
    saved = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event_type"] for event in saved] == [
        "completed_window",
        "episode_completed",
    ]


def test_acquisition_only_posterior_preservation_event_is_auditable(tmp_path):
    from src.utils.metrics import append_controller_event

    journal_path = tmp_path / "controller_metrics.jsonl"
    event = {
        "schema_version": 2,
        "event_type": "posterior_preserved",
        "boundary_step": 12,
        "window_index": 6,
        "episode_index": 0,
        "policy": "acquisition_only",
        "posterior_mean": [0.5, 0.25, 0.0],
        "posterior_covariance": [
            [0.5, -0.25, 0.0],
            [-0.25, 0.875, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "posterior_updated": False,
    }

    append_controller_event(journal_path, event)
    assert json.loads(journal_path.read_text(encoding="utf-8")) == event

    invalid = dict(event, policy="full_prior")
    with pytest.raises(ArtifactError, match="preserved.*acquisition_only"):
        append_controller_event(journal_path, invalid)


def test_checkpoint_schema_one_is_migrated_only_for_reset_disabled_continuation(
    tmp_path,
):
    from src.training.checkpointing import (
        validate_probabilistic_controller_checkpoint_state,
    )

    config = _probabilistic_checkpoint_config(tmp_path)
    legacy = _probabilistic_checkpoint_state("active_window")
    legacy["schema_version"] = 1
    legacy.pop("reset")
    legacy["window"].pop("selection_source")

    migrated = validate_probabilistic_controller_checkpoint_state(
        legacy,
        config=config,
        checkpoint_path=tmp_path / "old.pt",
    )
    assert migrated["schema_version"] == 2
    assert migrated["reset"]["enabled"] is False

    reset_config = resolve_run_config(
        "tests/fixtures/probabilistic_adaptive_global_smoke.yaml",
        output_dir=tmp_path / "probabilistic-adaptive-global-smoke-001",
        overrides={
            "model.adaptive_controller.process_noise_covariance": 0.0,
            "model.adaptive_controller.reset.enabled": True,
            "model.adaptive_controller.reset.interval_steps": 12,
        },
    )
    with pytest.raises(
        ConfigError,
        match="reset-enabled continuation requires complete reset state",
    ):
        validate_probabilistic_controller_checkpoint_state(
            legacy,
            config=reset_config,
            checkpoint_path=tmp_path / "old.pt",
        )


def test_controller_summary_separates_forced_and_thompson_frequencies(tmp_path):
    from src.utils.metrics import build_controller_summary

    template = _controller_journal_events()[1]
    events = []
    for index, (label, source) in enumerate(
        [
            ("micro", "forced_acquisition"),
            ("medium", "forced_acquisition"),
            ("micro", "thompson"),
        ]
    ):
        events.append(
            {
                **template,
                "window_index": index,
                "boundary_step": 2 + 2 * index,
                "boundary_step_start": 2 * index,
                "boundary_step_end": 2 + 2 * index,
                "action": {
                    "global_granularity": label,
                    "selection_source": source,
                },
                "selection_source": source,
            }
        )
    journal_path = tmp_path / "controller_metrics.jsonl"
    journal_path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )

    summary = build_controller_summary(
        controller_state=_probabilistic_checkpoint_state("terminal_incomplete"),
        controller_events=events,
        controller_metrics_path=journal_path,
    )

    assert summary["action_frequencies"] == {
        "micro": 2,
        "medium": 1,
        "full": 0,
    }
    assert summary["forced_acquisition_action_frequencies"] == {
        "micro": 1,
        "medium": 1,
        "full": 0,
    }
    assert summary["thompson_action_frequencies"] == {
        "micro": 1,
        "medium": 0,
        "full": 0,
    }
    assert summary["thompson_action_entropy"] == pytest.approx(0.0)


def test_probabilistic_controller_summary_preserves_auditable_state_and_hashes(
    tmp_path,
):
    from src.utils.metrics import (
        build_controller_summary,
        write_controller_summary,
    )

    journal_path = tmp_path / "controller_metrics.jsonl"
    events = _controller_journal_events()
    journal_path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    controller_state = _probabilistic_checkpoint_state("terminal_incomplete")

    summary = build_controller_summary(
        controller_state=controller_state,
        controller_events=events,
        controller_metrics_path=journal_path,
    )
    summary_path = write_controller_summary(tmp_path, summary)
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary_path == tmp_path / "controller_summary.json"
    assert saved_summary["method_family"] == "bayesian_gaussian_linear_thompson"
    assert saved_summary["method_version"] == 1
    assert saved_summary["scope"] == "global"
    assert saved_summary["completed_observation_count"] == 1
    assert saved_summary["controller_evaluation_count"] == 2
    assert saved_summary["action_frequencies"]["micro"] >= 1
    assert saved_summary["final_posterior_mean"] == [0.1, -0.2, 0.3]
    assert saved_summary["final_posterior_covariance"] == [
        [0.8, 0.1, 0.0],
        [0.1, 0.7, 0.05],
        [0.0, 0.05, 0.9],
    ]
    assert saved_summary["uncertainty_summary"]
    assert saved_summary["terminal_window"] == {
        "status": "incomplete",
        "window_index": 1,
        "completed_optimizer_steps": 1,
        "decision_interval_steps": 2,
    }
    assert saved_summary["resume_provenance"] == controller_state["resume"]
    assert saved_summary["failure_summary"]["error_category"] == (
        "non_finite_objective"
    )
    assert saved_summary["controller_metrics_path"] == str(journal_path)
    assert len(saved_summary["controller_metrics_hash"]) == 64


def test_balanced_warmup_events_do_not_contaminate_adaptive_statistics(tmp_path):
    from src.utils.metrics import append_controller_event, build_controller_summary

    adaptive_events = _controller_journal_events()
    common = {
        key: adaptive_events[0][key]
        for key in (
            "schema_version",
            "run_id",
            "method_family",
            "method_version",
            "strategy",
            "scope",
            "ordered_granularities",
            "feature_schema_hash",
            "controller_manifest_hash",
            "data_roles_manifest_hash",
            "decision_interval_steps",
            "resume_count",
            "resume_source_checkpoint",
        )
    }
    schedule = ["micro", "medium", "full", "full", "micro", "medium"]
    warmup_events = [
        {
            **common,
            "event_type": "warmup_schedule_initialized",
            "phase": "warmup",
            "schedule_hash": "warmup-schedule-hash",
            "schedule_seed": 123,
            "schedule": schedule,
            "action_interval_steps": 2,
            "requested_warmup_steps": 12,
            "action": {"global_granularity": "micro"},
            "boundary_step": 0,
            "window_index": 0,
            "warmup_window_index": 0,
            "boundary_step_start": 0,
            "boundary_step_end": 2,
            "completed_optimizer_steps": 0,
            "posterior_updated": False,
        },
        {
            **common,
            "event_type": "warmup_completed",
            "phase": "warmup",
            "schedule_hash": "warmup-schedule-hash",
            "schedule_seed": 123,
            "action_interval_steps": 2,
            "requested_warmup_steps": 12,
            "completed_warmup_steps": 12,
            "per_granularity_counts": {"micro": 2, "medium": 2, "full": 2},
            "controller_start_step": 12,
            "action": {"global_granularity": "medium"},
            "boundary_step": 12,
            "window_index": 5,
            "warmup_window_index": 5,
            "boundary_step_start": 10,
            "boundary_step_end": 12,
            "completed_optimizer_steps": 2,
            "posterior_updated": False,
        },
    ]
    events = [*warmup_events, *adaptive_events]
    journal_path = tmp_path / "controller_metrics.jsonl"
    for event in events:
        append_controller_event(journal_path, event)

    summary = build_controller_summary(
        controller_state=_probabilistic_checkpoint_state("terminal_incomplete"),
        controller_events=events,
        controller_metrics_path=journal_path,
    )

    assert summary["completed_observation_count"] == 1
    assert summary["controller_evaluation_count"] == 2
    assert summary["action_frequencies"] == {"micro": 1, "medium": 0, "full": 0}
    assert summary["requested_warmup_steps"] == 12
    assert summary["completed_warmup_steps"] == 12
    assert summary["warmup_action_counts"] == {
        "micro": 2,
        "medium": 2,
        "full": 2,
    }
    assert summary["posterior_updated_during_warmup"] is False


def test_probabilistic_artifacts_preserve_end_to_end_controller_provenance(
    tmp_path,
):
    from src.utils.metrics import (
        build_compact_controller_metric_fields,
        build_controller_summary,
    )

    config = _probabilistic_checkpoint_config(tmp_path)
    controller_state = _probabilistic_checkpoint_state("terminal_incomplete")
    controller_events = _controller_journal_events()[:3]
    journal_path = tmp_path / "controller_metrics.jsonl"
    journal_path.write_text(
        "".join(
            json.dumps(event, sort_keys=True) + "\n"
            for event in controller_events
        ),
        encoding="utf-8",
    )
    controller_summary = build_controller_summary(
        controller_state=controller_state,
        controller_events=controller_events,
        controller_metrics_path=journal_path,
    )
    compact_metric = build_compact_controller_metric_fields(
        controller_state,
        controller_events[1],
    )
    run_summary = build_run_summary(
        config,
        tokens_seen=256,
        content_tokens_seen=256,
        extra_fields={
            "data_roles_manifest_hash": controller_state["manifest_hashes"][
                "data_roles_manifest_hash"
            ],
            "optimizer_training_manifest_hash": controller_state[
                "manifest_hashes"
            ]["optimizer_training_manifest_hash"],
            "controller_manifest_hash": controller_state["manifest_hashes"][
                "controller_manifest_hash"
            ],
            "final_holdout_manifest_hash": controller_state["manifest_hashes"][
                "final_holdout_manifest_hash"
            ],
            "controller_summary": controller_summary,
            "controller_metrics_path": str(journal_path),
            "controller_summary_path": str(tmp_path / "controller_summary.json"),
        },
    )

    controller_config = config["model"]["adaptive_controller"]
    expected_identity = {
        "method_family": "bayesian_gaussian_linear_thompson",
        "method_version": 1,
        "strategy": "thompson",
        "scope": "global",
    }
    for field_name, expected_value in expected_identity.items():
        assert controller_config[field_name] == expected_value
        assert controller_state[field_name] == expected_value
        assert controller_events[0][field_name] == expected_value
        assert controller_summary[field_name] == expected_value

    assert controller_state["feature_schema"]["schema_hash"] == (
        controller_events[0]["feature_schema_hash"]
    )
    assert controller_state["feature_schema"] == controller_summary[
        "feature_schema"
    ]
    assert controller_state["probabilistic_inputs"] == controller_summary[
        "probabilistic_inputs"
    ]
    assert controller_state["manifest_hashes"] == controller_summary[
        "manifest_hashes"
    ]
    assert controller_state["belief"]["posterior_mean"].tolist() == (
        controller_summary["final_posterior_mean"]
    )
    assert controller_state["belief"]["posterior_covariance"].tolist() == (
        controller_summary["final_posterior_covariance"]
    )
    assert controller_state["sampling"]["seed_stream_name"] == (
        "posterior_sampling"
    )
    assert controller_state["sampling"]["sample_count"] == 1
    assert controller_state["window"]["phase"] == "terminal_incomplete"
    assert controller_summary["terminal_window"]["completed_optimizer_steps"] == 1
    assert controller_summary["resume_provenance"] == controller_state["resume"]

    assert compact_metric == {
        "controller_method_family": expected_identity["method_family"],
        "controller_method_version": expected_identity["method_version"],
        "controller_strategy": expected_identity["strategy"],
        "controller_scope": expected_identity["scope"],
        "controller_action": "micro",
        "controller_window_index": 1,
        "controller_window_progress": 1,
        "controller_boundary_step": 2,
        "controller_latest_objective": 8.0,
        "controller_latest_reward": 1.0,
        "controller_latest_prediction_error": 1.0,
        "controller_manifest_hash": "controller-manifest-hash",
        "final_holdout_manifest_hash": "final-holdout-manifest-hash",
            "controller_metrics_path": "controller_metrics.jsonl",
            "controller_summary_path": "controller_summary.json",
            "controller_reset_enabled": False,
            "controller_reset_policy": "full_prior",
            "controller_episode_index": None,
            "controller_episode_offset_steps": 0,
            "controller_selection_source": "thompson",
        }
    assert run_summary["controller_summary"] == controller_summary
    assert run_summary["data_roles_manifest_hash"] == "parent-manifest-hash"
    assert run_summary["controller_manifest_hash"] == "controller-manifest-hash"
    assert run_summary["final_holdout_manifest_hash"] == (
        "final-holdout-manifest-hash"
    )


def test_probabilistic_checkpoint_rejects_historical_heuristic_thompson_state():
    from src.training.checkpointing import (
        validate_probabilistic_controller_checkpoint_state,
    )
    from src.utils.config import ConfigError

    historical_state = {
        "strategy": "thompson",
        "mean_reward_by_granularity": {"micro": 0.25, "full": 0.5},
        "selection_count_by_granularity": {"micro": 3, "full": 4},
        "previous_loss": 1.75,
    }

    with pytest.raises(
        ConfigError,
        match="Bayesian controller checkpoint state is incomplete",
    ):
        validate_probabilistic_controller_checkpoint_state(
            historical_state,
            checkpoint_path="historical-thompson.pt",
        )


def test_nonadaptive_checkpoint_round_trip_remains_free_of_controller_state(tmp_path):
    import src.training.checkpointing as training_checkpointing

    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "run.continuation.enabled=true",
            "model.granularity_sampling_mode=global",
        ],
    )
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    run_state = training_checkpointing.build_initial_continuation_state(config)
    run_state.update(
        {
            "step": 3,
            "last_completed_step": 3,
            "tokens_seen": 24,
            "content_tokens_seen": 18,
        }
    )
    checkpoint_path = output_dir / "checkpoints" / "latest.pt"

    training_checkpointing.save_model_checkpoint(
        config,
        model,
        optimizer,
        scheduler=None,
        output_path=checkpoint_path,
        checkpoint_fields={
            "checkpoint_status": "latest",
            "checkpoint_metric": None,
            "checkpoint_metric_value": None,
            "checkpoint_selection_step": None,
        },
        run_state=run_state,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["adaptive_sampler_state"] is None
    assert checkpoint["probabilistic_controller_state"] is None
    assert checkpoint["panelgrad_state"] is None
    assert checkpoint["adaptive_sampler_strategy"] is None
    assert checkpoint["step"] == 3
    assert checkpoint["tokens_seen"] == 24

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.01)
    restored_state = training_checkpointing.load_checkpoint_state(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        scheduler=None,
        config=config,
    )

    assert restored_state["status"] == "resumed"
    assert restored_state["last_completed_step"] == 3
    assert restored_state["resume_count"] == 1
    assert restored_state["adaptive_sampler_state"] is None
    assert restored_state["probabilistic_controller_state"] is None
    assert restored_state["panelgrad_state"] is None
    for parameter, restored_parameter in zip(
        model.parameters(),
        restored_model.parameters(),
    ):
        assert torch.equal(parameter, restored_parameter)


def test_panelgrad_refresh_terminal_summary_and_compact_fields_are_auditable(tmp_path):
    labels = ["small", "full"]
    controller = PanelGradController(
        ordered_granularities=labels,
        refresh_interval_steps=2,
        eta=1e-12,
        temperature=1.0,
        epsilon=0.1,
        sampling_seed=17,
        support_identity={
            "ordered_granularities": labels,
            "controlled_support_counts": {"small": 10, "full": 20},
            "controlled_support_hash": "support-hash",
        },
    )
    measurement = {
        "measurements": [
            {
                "granularity": "small",
                "controlled_parameter_count": 10,
                "gradient_rms_score": 1.0,
            },
            {
                "granularity": "full",
                "controlled_parameter_count": 20,
                "gradient_rms_score": 2.0,
            },
        ],
        "duration_seconds": 0.25,
        "backward_evaluation_count": 4,
    }
    refresh = controller.install_refresh(measurement, boundary_step=0)
    action = controller.sample_action()
    controller.commit_pending_action(completed_step=1)
    controller.finish_training(completed_step=1)
    events = [
        {
            "schema_version": 1,
            "event_type": "panelgrad_refresh_completed",
            "method_family": "panelgrad_gradient_rms",
            "method_version": 1,
            "boundary_step": 0,
            "window_index": 0,
            "measurements": refresh["measurements"],
            "q": refresh["q"],
            "p": refresh["p"],
            "entropy": refresh["entropy"],
            "active_epsilon": refresh["active_epsilon"],
            "epsilon_schedule_step": refresh["epsilon_schedule_step"],
            "duration_seconds": 0.25,
            "backward_evaluation_count": 4,
        },
        {
            "schema_version": 1,
            "event_type": "panelgrad_terminal_partial",
            "method_family": "panelgrad_gradient_rms",
            "method_version": 1,
            "boundary_step": 1,
            "window_index": 0,
        },
    ]
    path = tmp_path / "controller_metrics.jsonl"
    append_controller_events(path, events)

    summary = build_controller_summary(
        controller_state=controller.state_dict(),
        controller_events=events,
        controller_metrics_path=path,
    )
    compact = build_compact_controller_metric_fields(controller.state_dict())

    assert summary["refresh_count"] == 1
    assert summary["final_p"] == refresh["p"]
    assert summary["active_epsilon"] == pytest.approx(0.1)
    assert summary["epsilon_schedule_step"] == 0
    assert summary["epsilon_schedule"]["type"] == "fixed"
    assert summary["epsilon_history"] == [
        {
            "refresh_index": 0,
            "active_epsilon": 0.1,
            "epsilon_schedule_step": 0,
        }
    ]
    assert summary["exposure_counts"][action["global_granularity"]] == 1
    assert summary["cumulative_measurement_duration_seconds"] == 0.25
    assert summary["cumulative_backward_evaluations"] == 4
    assert summary["controller_metrics_hash"]
    assert compact["controller_strategy"] == "panelgrad"
    assert compact["controller_sampled_probability"] == action["probability"]
    assert compact["controller_phase"] == "terminal_partial"
    assert "active_epsilon" not in compact


def test_panelgrad_checkpoint_round_trips_versioned_policy_and_rng_state(tmp_path):
    import src.training.checkpointing as training_checkpointing

    output_dir = tmp_path / "panelgrad-checkpoint" / "panelgrad-smoke-001"
    config = resolve_run_config(
        "tests/fixtures/panelgrad_smoke.yaml",
        output_dir=output_dir,
    )
    configure_strict_determinism(config)
    labels = list(config["model"]["granularities"])
    support = {
        "ordered_granularities": labels,
        "controlled_support_counts": {label: 10 for label in labels},
        "controlled_support_hash": "checkpoint-support-hash",
    }
    config["model"]["panelgrad"]["controlled_support_counts"] = copy.deepcopy(
        support["controlled_support_counts"]
    )
    config["model"]["panelgrad"]["controlled_support_hash"] = support[
        "controlled_support_hash"
    ]
    for field in (
        "data_roles_manifest_hash",
        "optimizer_training_manifest_hash",
        "controller_manifest_hash",
        "validation_manifest_hash",
        "final_holdout_manifest_hash",
    ):
        config[field] = f"{field}-value"
    controller = PanelGradController(
        ordered_granularities=labels,
        refresh_interval_steps=2,
        eta=1e-12,
        temperature=1.0,
        epsilon=0.1,
        sampling_seed=config["model"]["panelgrad"]["sampling_seed"],
        support_identity=support,
        manifest_hashes={
            field: config[field]
            for field in (
                "data_roles_manifest_hash",
                "optimizer_training_manifest_hash",
                "controller_manifest_hash",
                "validation_manifest_hash",
                "final_holdout_manifest_hash",
            )
        },
    )
    measurement = {
        "measurements": [
            {
                "granularity": label,
                "controlled_parameter_count": 10,
                "gradient_rms_score": float(index + 1),
            }
            for index, label in enumerate(labels)
        ]
    }
    controller.install_refresh(measurement, boundary_step=0)
    controller.sample_action()
    controller.commit_pending_action(completed_step=1)
    run_state = training_checkpointing.build_initial_continuation_state(config)
    run_state.update(step=1, last_completed_step=1)
    run_state["panelgrad_state"] = controller.state_dict()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    checkpoint_path = output_dir / "checkpoints" / "latest.pt"
    training_checkpointing.save_model_checkpoint(
        config,
        model,
        optimizer,
        scheduler=None,
        output_path=checkpoint_path,
        checkpoint_fields={
            "checkpoint_status": "latest",
            "checkpoint_metric": None,
            "checkpoint_metric_value": None,
            "checkpoint_selection_step": 1,
        },
        run_state=run_state,
    )
    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.01)

    restored = training_checkpointing.load_checkpoint_state(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        scheduler=None,
        config=config,
    )

    assert restored["panelgrad_state"]["refresh"]["p"] == (
        controller.state_dict()["refresh"]["p"]
    )
    assert restored["panelgrad_state"]["sampling"]["exposure_counts"] == (
        controller.state_dict()["sampling"]["exposure_counts"]
    )
    assert torch.equal(
        restored["panelgrad_state"]["sampling"]["generator_state"],
        controller.state_dict()["sampling"]["generator_state"],
    )
