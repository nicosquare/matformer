from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

import scripts.evaluate_final_holdout as evaluate_holdout_cli
import src.evaluation.final_holdout as final_holdout_module
import src.evaluation.reporting_impl as reporting_impl
from scripts.analyze_tinystories_portfolio_catchup import (
    AGGREGATE_REFERENCE_BUDGET_TOKENS,
    CANDIDATE_ARMS,
    ELASTIC_BUDGET_CAP_TOKENS,
    FIXED_LEARNING_RATE,
    GRANULARITIES,
    LOSS_TOLERANCE,
    REFERENCE_BUDGET_TOKENS,
    PortfolioAnalysisError,
    final_holdout,
    freeze_references,
    portfolio_catchup,
    portfolio_catchup_bundle,
)
from src.evaluation.final_holdout import (
    FinalHoldoutError,
    resolve_existing_final_holdout_result,
)
from src.evaluation.reporting import (
    _portfolio_elastic_selection_label,
    _portfolio_figure_identity,
    generate_figures,
)
from src.training import checkpointing
from src.training import steps as training_steps
from src.training.portfolio_catchup import (
    PortfolioCatchupError,
    build_portfolio_catchup_state,
    candidate_policy_contract_hash,
    update_portfolio_catchup_state,
    validate_portfolio_catchup_state,
)
from src.utils.config import (
    ConfigError,
    _resolve_global_sampling_interval_steps,
    _resolve_global_sampling_schedule,
    _resolve_portfolio_controlled_experiment,
    _validate_portfolio_aligned_epoch_contract,
    _validate_portfolio_controlled_experiment,
    _validate_portfolio_manifest_link,
    load_yaml_config,
)
from src.utils.reproducibility import stable_hash
from src.utils.reproducibility import build_comparison_control_signature
from src.utils.metrics import METRICS_COLUMNS, MetricsJournal


def _contract(
    role: str,
    *,
    target=None,
    granularities=GRANULARITIES,
    comparison_group="tinystories_instruct_portfolio_catchup_v1",
):
    return {
        "comparison_group_id": comparison_group,
        "comparison_role": role,
        "portfolio_catchup": {
            "enabled": role == "elastic_candidate",
            "schema_version": 2,
            "reference_budget_tokens": REFERENCE_BUDGET_TOKENS,
            "elastic_budget_cap_tokens": ELASTIC_BUDGET_CAP_TOKENS,
            "aggregate_reference_count": 4,
            "granularities": list(granularities),
            "perplexity_tolerance": 0.005,
            "required_consecutive_evaluations": 5,
            "target_manifest_path": str(target) if target else None,
            "target_manifest_hash": (
                json.loads(Path(target).read_text())["manifest_hash"]
                if target
                else None
            ),
            "save_confirmation_checkpoint": True,
            "stop_on_confirmation": False,
        },
    }


def _provenance_iteration():
    return {
        "mode": "repeat_epochs",
        "epoch_order": "deterministic_per_epoch",
        "aligned_epoch_samples": 100,
        "aligned_epoch_tokens": REFERENCE_BUDGET_TOKENS,
        "fixed_epoch_set_hash": "fixed-set",
        "permutation_version": "permutation-v1",
        "permutation_hash": "permutation-hash",
        "ordering_policy_version": "ordering-v1",
        "optimizer_training_manifest_hash": "optimizer-hash",
    }


def _config(
    role: str,
    *,
    seed: int,
    width: str | None = None,
    lr: float = 0.008,
    target=None,
    granularities=GRANULARITIES,
    comparison_group="tinystories_instruct_portfolio_catchup_v1",
):
    granularities = tuple(granularities)
    prefixes = {
        granularity: int(granularity.removeprefix("g")) / 1000.0
        for granularity in granularities
    }
    standalone = role == "standalone_reference"
    budget = (
        ELASTIC_BUDGET_CAP_TOKENS
        if role == "elastic_candidate"
        else REFERENCE_BUDGET_TOKENS
    )
    config = {
        "controlled_experiment": _contract(
            role,
            target=target,
            granularities=granularities,
            comparison_group=comparison_group,
        ),
        "run": {
            "run_id": f"{role}-{width or 'elastic'}-s{seed}-lr{lr}",
            "seed": seed,
            "model_family": "standalone" if standalone else "nested",
            "sampling_mode": "standalone" if standalone else "nested-random",
            "granularity": width if standalone else None,
            "reproducibility": {"seed_stream_version": 1},
        },
        "model": {
            "variant": "slicing",
            "correction_mode": "none",
            "granularity_mode": "explicit",
            "granularities": [width] if standalone else list(granularities),
            "granularity_sampling_mode": "global",
            "global_sampling_schedule": "random_with_replacement",
            "global_sampling_interval_steps": 1,
            "d_model": 64,
            "intermediate_size": (
                int(256 * prefixes[str(width)])
                if standalone
                else 256
            ),
            "matformer_source_intermediate_size": 256 if standalone else None,
            "granularity_prefixes": (
                {width: 1.0}
                if standalone
                else prefixes
            ),
            "ffn_prefix_metadata": (
                [
                    {
                        "name": width,
                        "display_name": width.upper(),
                        "ffn_ratio": 1.0,
                        "full_intermediate_fraction": 1.0,
                        "prefix_width": int(256 * prefixes[str(width)]),
                    }
                ]
                if standalone
                else [
                    {
                        "name": item,
                        "display_name": item.upper(),
                        "ffn_ratio": 4.0 * prefixes[item],
                        "full_intermediate_fraction": prefixes[item],
                        "prefix_width": int(256 * prefixes[item]),
                    }
                    for item in granularities
                ]
            ),
            "num_layers": 4,
            "num_attention_heads": 4,
            "context_length": 128,
            "vocab_size": 2048,
            "tokenizer_manifest_hash": "tokenizer-hash",
        },
        "training": {
            "token_budget": budget,
            "learning_rate": lr,
            "resolved_learning_rate": lr,
            "scheduler_name": "cosine",
            "batch_size_per_process": 64,
            "effective_world_size": 1,
            "gradient_accumulation_steps": 1,
            "expected_tokens_per_step": 8192,
            "derived_max_steps": 261396 if role == "elastic_candidate" else 87132,
            "max_steps": 261396 if role == "elastic_candidate" else 87132,
            "resolved_warmup_steps": 64,
            "resolved_mixed_precision": "bf16",
            "optimizer": {"name": "adamw", "kwargs": {"weight_decay": 0.1}},
            "scheduler": {"name": "cosine", "kwargs": {}},
            "distributed": {"expected_world_size": 1},
        },
        "dataset": {
            "dataset_name": "roneneldan/TinyStoriesInstruct",
            "dataset_config_name": "default",
            "dataset_split": "train+validation",
            "dataset_phase": "tinystories_instruct_controlled",
            "corpus_hash": "corpus-hash",
            "optimizer_iteration": _provenance_iteration(),
        },
        "validation_manifest_hash": "validation-hash",
        "final_holdout_manifest_hash": "holdout-hash",
        "optimizer_training_manifest_hash": "optimizer-hash",
        "evaluation": {
            "validation": {
                "interval_steps": 64,
                "interval_tokens": 0,
                "run_at_completion": True,
            }
        },
    }
    config["dataset"]["optimizer_iteration"].update(
        {
            "complete_epochs": 3 if role == "elastic_candidate" else 1,
            "partial_final_epoch_tokens": 0,
        }
    )
    return config


def _write_metrics(
    path: Path,
    losses: dict[str, list[float]],
    *,
    terminal_step: int | None = None,
    terminal_tokens: int | None = None,
) -> None:
    fieldnames = [
        "run_id",
        "split",
        "step",
        "tokens_seen",
        "granularity",
        "loss",
        "perplexity",
    ]
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for width, values in losses.items():
            for index, loss in enumerate(values, 1):
                is_terminal = index == len(values) and terminal_step is not None
                writer.writerow(
                    {
                        "run_id": path.parent.name,
                        "split": "validation",
                        "step": terminal_step if is_terminal else index,
                        "tokens_seen": terminal_tokens if is_terminal else index * 100,
                        "granularity": width,
                        "loss": loss,
                        "perplexity": math.exp(loss),
                    }
                )


def _write_run(
    root: Path,
    config: dict,
    losses: dict[str, list[float]],
    *,
    checkpoint_payload: dict | None = None,
    portfolio_state: dict | None = None,
):
    run_id = config["run"]["run_id"]
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    checkpoint_path = run_dir / "checkpoints" / "best_eval_step_2.pt"
    checkpoint_path.parent.mkdir()
    if checkpoint_payload is None:
        checkpoint_path.write_bytes(f"checkpoint-{run_id}".encode())
    else:
        torch.save(checkpoint_payload, checkpoint_path)
    width = next(iter(losses))
    best_index = min(range(len(losses[width])), key=lambda index: losses[width][index])
    summary = {
        "run_id": run_id,
        "seed": config["run"]["seed"],
        "status": "completed",
        "tokens_seen": config["training"]["token_budget"],
        "token_budget": config["training"]["token_budget"],
        "resolved_learning_rate": config["training"]["resolved_learning_rate"],
        "checkpoint_status": "best_eval",
        "best_checkpoint_path": str(checkpoint_path),
        "checkpoint_selection_step": best_index + 1,
        "unresolved_artifact_failures": [],
        "optimizer_iteration": config["dataset"]["optimizer_iteration"],
    }
    if portfolio_state is not None:
        summary["portfolio_catchup_state"] = portfolio_state
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _write_metrics(
        run_dir / "metrics.csv",
        losses,
        terminal_step=config["training"]["max_steps"],
        terminal_tokens=config["training"]["token_budget"],
    )
    prefix_widths = {
        str(item["name"]): int(item["prefix_width"])
        for item in config["model"]["ffn_prefix_metadata"]
    }
    d_model = int(config["model"]["d_model"])
    num_layers = int(config["model"]["num_layers"])
    with (run_dir / "scaling_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["granularity", "non_embedding_parameters"],
        )
        writer.writeheader()
        for granularity in config["model"]["granularities"]:
            prefix_width = prefix_widths[granularity]
            writer.writerow(
                {
                    "granularity": granularity,
                    "non_embedding_parameters": d_model
                    + num_layers
                    * (
                        4 * d_model * d_model
                        + 3 * d_model * prefix_width
                        + 2 * d_model
                    ),
                }
            )
    return run_dir, checkpoint_path


def _reference_runs(
    tmp_path: Path,
    *,
    granularities=GRANULARITIES,
    comparison_group="tinystories_instruct_portfolio_catchup_v1",
    seeds=(42, 43, 44),
):
    runs = []
    for seed in seeds:
        for width_index, width in enumerate(granularities):
            target_loss = 1.0 + width_index * 0.1 + (seed - 42) * 0.001
            config = _config(
                "standalone_reference",
                seed=seed,
                width=width,
                granularities=granularities,
                comparison_group=comparison_group,
            )
            values = [target_loss + 0.1, target_loss, target_loss + 0.02]
            run_dir, _ = _write_run(tmp_path / "references", config, {width: values})
            runs.append(run_dir)
    return runs


def _freeze(
    tmp_path: Path,
    *,
    granularities=GRANULARITIES,
    comparison_group="tinystories_instruct_portfolio_catchup_v1",
    granularity_profile="quartile",
    seeds=(42, 43, 44),
):
    output = tmp_path / "reference-analysis"
    manifest = freeze_references(
        _reference_runs(
            tmp_path,
            granularities=granularities,
            comparison_group=comparison_group,
            seeds=seeds,
        ),
        output,
        granularity_profile=granularity_profile,
    )
    return output / "standalone_portfolio_targets.json", manifest


def _candidate_runs(
    tmp_path: Path,
    target_path: Path,
    targets: dict,
    *,
    candidate_arm_id: str = "uniform_h1_3b",
    granularities=GRANULARITIES,
    comparison_group="tinystories_instruct_portfolio_catchup_v1",
    seeds=(42, 43, 44),
):
    runs = []
    for seed in seeds:
        config = _config(
            "elastic_candidate",
            seed=seed,
            lr=FIXED_LEARNING_RATE,
            target=target_path,
            granularities=granularities,
            comparison_group=comparison_group,
        )
        if candidate_arm_id != "uniform_h1_3b":
            arm = CANDIDATE_ARMS[candidate_arm_id]
            budget = int(arm["budget_tokens"])
            epochs = budget // REFERENCE_BUDGET_TOKENS
            config["controlled_experiment"]["comparison_arm_id"] = candidate_arm_id
            contract = config["controlled_experiment"]["portfolio_catchup"]
            contract["schema_version"] = 3
            contract["elastic_budget_cap_tokens"] = budget
            config["training"]["token_budget"] = budget
            config["training"]["derived_max_steps"] = 87132 * epochs
            config["training"]["max_steps"] = 87132 * epochs
            config["dataset"]["optimizer_iteration"]["complete_epochs"] = epochs
            config["run"]["sampling_mode"] = arm["sampling_mode"]
            config["model"]["variant"] = arm["model_variant"]
        losses = {}
        for width in granularities:
            target_loss = targets["targets"][str(seed)][width]["target_loss"]
            losses[width] = [target_loss + 0.02] + [target_loss] * 6
        run_id = config["run"]["run_id"]
        run_dir = tmp_path / "candidates" / run_id
        checkpoint_path = run_dir / "checkpoints" / "portfolio_catchup_step_6.pt"
        state = {
            "confirmed": True,
            "streak_onset_step": 2,
            "streak_onset_tokens": 200,
            "confirmation_step": 6,
            "confirmation_tokens": 600,
            "confirmation_checkpoint_saved": True,
            "confirmation_checkpoint_path": str(checkpoint_path),
            "target_manifest_hash": targets["manifest_hash"],
            "learning_rate": FIXED_LEARNING_RATE,
        }
        if candidate_arm_id != "uniform_h1_3b":
            state["comparison_arm_id"] = candidate_arm_id
            state["elastic_budget_cap_tokens"] = config["training"]["token_budget"]
        payload = {
            "checkpoint_status": "portfolio_catchup_confirmation",
            "portfolio_catchup_state": state,
            "model_state_dict": {"weight": torch.tensor([float(seed)])},
        }
        # _write_run uses a best-eval filename; replace it with the immutable one.
        created_dir, ordinary_checkpoint = _write_run(
            tmp_path / "candidates",
            config,
            losses,
            checkpoint_payload=payload,
            portfolio_state=state,
        )
        checkpoint_path.parent.mkdir(exist_ok=True)
        ordinary_checkpoint.replace(checkpoint_path)
        state["confirmation_checkpoint_sha256"] = hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest()
        terminal_path = created_dir / "checkpoints" / "latest.pt"
        torch.save(
            {
                "checkpoint_status": "latest",
                "step": config["training"]["max_steps"],
                "tokens_seen": config["training"]["token_budget"],
                "run_id": run_id,
                "portfolio_catchup_state": state,
                "model_state_dict": {"weight": torch.tensor([float(seed)])},
            },
            terminal_path,
        )
        summary = json.loads((created_dir / "run_summary.json").read_text())
        summary.update(
            {
                "portfolio_catchup_state": state,
                "steps_completed": config["training"]["max_steps"],
                "latest_checkpoint_path": str(terminal_path),
            }
        )
        (created_dir / "run_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        runs.append(created_dir)
    return runs


def test_portfolio_config_contract_exact_budgets_roles_and_h1():
    for width in GRANULARITIES:
        _validate_portfolio_controlled_experiment(
            _config("standalone_reference", seed=42, width=width)
        )
    target_stub = {"manifest_hash": "0" * 64}
    # Direct contract validation only requires the immutable link fields to exist.
    candidate = _config("elastic_candidate", seed=42)
    candidate["controlled_experiment"]["portfolio_catchup"].update(
        target_manifest_path="targets.json",
        target_manifest_hash=target_stub["manifest_hash"],
    )
    _validate_portfolio_controlled_experiment(candidate)
    _validate_portfolio_aligned_epoch_contract(candidate)
    bad = json.loads(json.dumps(candidate))
    bad["training"]["token_budget"] += 1
    with pytest.raises(ConfigError, match="exactly"):
        _validate_portfolio_controlled_experiment(bad)
    bad = json.loads(json.dumps(candidate))
    bad["model"]["global_sampling_interval_steps"] = 2
    with pytest.raises(ConfigError, match="H=1"):
        _validate_portfolio_controlled_experiment(bad)
    bad = json.loads(json.dumps(candidate))
    bad["dataset"]["optimizer_iteration"]["aligned_epoch_tokens"] -= 128
    with pytest.raises(ConfigError, match="aligned"):
        _validate_portfolio_aligned_epoch_contract(bad)
    bad = json.loads(json.dumps(candidate))
    bad["training"]["resolved_learning_rate"] = 0.006
    with pytest.raises(ConfigError, match="fixed LR 0.008"):
        _validate_portfolio_controlled_experiment(bad)
    bad = json.loads(json.dumps(candidate))
    bad["controlled_experiment"]["comparison_role"] = "elastic_lr_screen"
    with pytest.raises(ConfigError, match="comparison_role"):
        _validate_portfolio_controlled_experiment(bad)
    bad = json.loads(json.dumps(candidate))
    bad["controlled_experiment"]["portfolio_catchup"][
        "lr_selection_manifest_hash"
    ] = "0" * 64
    with pytest.raises(ConfigError, match="not supported"):
        _validate_portfolio_controlled_experiment(bad)
    assert ELASTIC_BUDGET_CAP_TOKENS / AGGREGATE_REFERENCE_BUDGET_TOKENS == 0.75


def test_schema3_extension_arms_reuse_targets_with_arm_specific_budgets(tmp_path):
    target_path, _ = _freeze(tmp_path)
    for arm_id in (
        "uniform_h1_4b",
        "nested_all_b",
        "nested_all_4b",
        "concat_uniform_h1_4b",
    ):
        arm = CANDIDATE_ARMS[arm_id]
        config = _config(
            "elastic_candidate",
            seed=42,
            target=target_path,
        )
        budget = int(arm["budget_tokens"])
        epochs = budget // REFERENCE_BUDGET_TOKENS
        config["controlled_experiment"]["comparison_arm_id"] = arm_id
        contract = config["controlled_experiment"]["portfolio_catchup"]
        contract["schema_version"] = 3
        contract["elastic_budget_cap_tokens"] = budget
        config["run"]["sampling_mode"] = arm["sampling_mode"]
        config["model"]["variant"] = arm["model_variant"]
        config["training"]["token_budget"] = budget
        config["training"]["derived_max_steps"] = 87132 * epochs
        config["training"]["max_steps"] = 87132 * epochs
        config["dataset"]["optimizer_iteration"]["complete_epochs"] = epochs

        _validate_portfolio_controlled_experiment(config)
        _validate_portfolio_aligned_epoch_contract(config)
        state = build_portfolio_catchup_state(config)
        assert state["schema_version"] == 3
        assert state["comparison_arm_id"] == arm_id
        assert state["elastic_budget_cap_tokens"] == budget

    mismatched = json.loads(json.dumps(config))
    mismatched["training"]["token_budget"] += REFERENCE_BUDGET_TOKENS
    with pytest.raises(ConfigError, match="exactly"):
        _validate_portfolio_controlled_experiment(mismatched)

    wrong_variant = json.loads(json.dumps(config))
    wrong_variant["model"]["variant"] = "slicing"
    with pytest.raises(ConfigError, match="model variant"):
        _validate_portfolio_controlled_experiment(wrong_variant)


def test_schema4_bundle_contract_accepts_independent_budget_and_fixed_policy(tmp_path):
    target_path, _ = _freeze(tmp_path)
    config = _config("elastic_candidate", seed=42, target=target_path)
    contract = config["controlled_experiment"]["portfolio_catchup"]
    config["controlled_experiment"]["comparison_arm_id"] = "fixed-large-x2"
    contract.update(
        {
            "schema_version": 4,
            "budget_unit_tokens": REFERENCE_BUDGET_TOKENS,
            "reference_budget_multiplier": 1,
            "candidate_budget_multiplier": 2,
            "claim_tier": "diagnostic",
        }
    )
    config["training"].update(
        {
            "token_budget": 2 * REFERENCE_BUDGET_TOKENS,
            "derived_max_steps": 2 * 87132,
            "max_steps": 2 * 87132,
        }
    )
    config["dataset"]["optimizer_iteration"]["complete_epochs"] = 2
    config["model"].update(
        {
            "granularity_sampling_mode": "fixed_global",
            "global_sampling_distribution": {
                "g250": 0.20,
                "g500": 0.20,
                "g750": 0.25,
                "g1000": 0.35,
            },
        }
    )

    _resolve_portfolio_controlled_experiment(config)
    _validate_portfolio_controlled_experiment(config)
    state = build_portfolio_catchup_state(config)
    assert state["schema_version"] == 4
    assert state["comparison_arm_id"] == "fixed-large-x2"
    assert state["candidate_policy_contract_hash"]
    assert (
        config["controlled_experiment"]["portfolio_catchup"][
            "elastic_budget_cap_tokens"
        ]
        == 2 * REFERENCE_BUDGET_TOKENS
    )

    bad = json.loads(json.dumps(config))
    bad["controlled_experiment"]["portfolio_catchup"]["claim_tier"] = "confirmatory"
    with pytest.raises(ConfigError, match="bundle contract"):
        _validate_portfolio_controlled_experiment(bad)


def test_schema4_thompson_policy_hash_ignores_runtime_and_seed_provenance(tmp_path):
    target_path, _ = _freeze(tmp_path)
    config = _config("elastic_candidate", seed=42, target=target_path)
    config["controlled_experiment"]["comparison_arm_id"] = "thompson-x3"
    contract = config["controlled_experiment"]["portfolio_catchup"]
    contract.update(
        {
            "schema_version": 4,
            "budget_unit_tokens": REFERENCE_BUDGET_TOKENS,
            "reference_budget_multiplier": 1,
            "candidate_budget_multiplier": 3,
            "claim_tier": "diagnostic",
        }
    )
    config["model"].update(
        {
            "granularity_sampling_mode": "adaptive_global",
            "adaptive_sampler_strategy": "thompson",
            "adaptive_controller": {
                "strategy": "thompson",
                "method_family": "bayesian_gaussian_linear_thompson",
                "method_version": 1,
                "scope": "global",
                "ordered_granularities": list(GRANULARITIES),
                "decision_interval_steps": 25,
                "prior_mean": 0.0,
                "prior_covariance": 1.0,
                "observation_noise_variance": 0.01,
                "process_noise_covariance": 0.0001,
                "reset": {
                    "enabled": False,
                    "schedule_seed_stream_name": "controller_reset_schedule",
                    "schedule_seed": 42,
                },
                "controller_panel_contract": {
                    "enabled": True,
                    "examples": 128,
                    "manifest_hash": "pending",
                },
                "final_holdout_contract": {
                    "enabled": True,
                    "examples": 512,
                    "manifest_hash": "pending",
                },
                "preset": "bayesian_thompson",
                "preset_registry_path": "/checkout/configs/preset.yaml",
            },
            "panelgrad": None,
        }
    )

    initial_hash = candidate_policy_contract_hash(config)
    state = build_portfolio_catchup_state(config)
    enriched = json.loads(json.dumps(config))
    controller = enriched["model"]["adaptive_controller"]
    controller.update(
        {
            "feature_schema": {"schema_version": 1, "schema_hash": "runtime"},
            "controller_metrics_path": "controller_metrics.jsonl",
            "controller_summary_path": "controller_summary.json",
            "data_roles_manifest_hash": "data-roles",
            "optimizer_training_manifest_hash": "optimizer-training",
            "controller_manifest_hash": "controller",
            "ordinary_validation_manifest_hash": "validation",
            "final_holdout_manifest_hash": "holdout",
            "preset_registry_path": "/another/checkout/configs/preset.yaml",
        }
    )
    controller["controller_panel_contract"]["manifest_hash"] = "controller"
    controller["final_holdout_contract"]["manifest_hash"] = "holdout"
    controller["reset"]["schedule_seed"] = 43

    assert candidate_policy_contract_hash(enriched) == initial_hash
    assert validate_portfolio_catchup_state(state, config=enriched) == state

    changed_policy = json.loads(json.dumps(enriched))
    changed_policy["model"]["adaptive_controller"]["decision_interval_steps"] = 50
    assert candidate_policy_contract_hash(changed_policy) != initial_hash
    with pytest.raises(PortfolioCatchupError, match="does not match"):
        validate_portfolio_catchup_state(state, config=changed_policy)


def test_schema4_panelgrad_policy_hash_ignores_runtime_support_and_seed():
    config = {
        "controlled_experiment": {
            "comparison_arm_id": "panelgrad-l2-x3",
            "portfolio_catchup": {
                "candidate_budget_multiplier": 3,
                "reference_budget_multiplier": 1,
                "elastic_budget_cap_tokens": ELASTIC_BUDGET_CAP_TOKENS,
            },
        },
        "run": {"sampling_mode": "nested-random"},
        "model": {
            "variant": "slicing",
            "correction_mode": "none",
            "granularity_sampling_mode": "adaptive_global",
            "adaptive_sampler_strategy": "panelgrad",
            "adaptive_controller": None,
            "panelgrad": {
                "importance_metric": "gradient_l2",
                "refresh_interval_steps": 25,
                "epsilon": 0.1,
                "ordered_granularities": list(GRANULARITIES),
                "controlled_support_counts": "pending",
                "controlled_support_hash": "pending",
                "sampling_seed_stream": "panelgrad_sampling",
                "sampling_seed": 42,
                "controller_panel_contract": {"manifest_hash": "pending"},
                "final_holdout_contract": {"manifest_hash": "pending"},
            },
        },
        "training": {"optimizer": {"state_scope": "shared"}},
    }
    initial_hash = candidate_policy_contract_hash(config)
    enriched = json.loads(json.dumps(config))
    panelgrad = enriched["model"]["panelgrad"]
    panelgrad["controlled_support_counts"] = {width: 64 for width in GRANULARITIES}
    panelgrad["controlled_support_hash"] = "resolved-support"
    panelgrad["sampling_seed"] = 43
    panelgrad["controller_panel_contract"]["manifest_hash"] = "controller"
    panelgrad["final_holdout_contract"]["manifest_hash"] = "holdout"

    assert candidate_policy_contract_hash(enriched) == initial_hash
    enriched["model"]["panelgrad"]["refresh_interval_steps"] = 50
    assert candidate_policy_contract_hash(enriched) != initial_hash


def test_schema4_bundle_arm_is_analyzed_without_a_registered_arm_name(tmp_path):
    target_path, targets = _freeze(tmp_path)
    runs = _candidate_runs(tmp_path, target_path, targets, seeds=(42,))
    for run_dir in runs:
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text())
        config["controlled_experiment"]["comparison_arm_id"] = "balanced-h5-x3"
        contract = config["controlled_experiment"]["portfolio_catchup"]
        contract.update(
            {
                "schema_version": 4,
                "budget_unit_tokens": REFERENCE_BUDGET_TOKENS,
                "reference_budget_multiplier": 1,
                "candidate_budget_multiplier": 3,
                "claim_tier": "diagnostic",
            }
        )
        config["model"].update(
            {
                "global_sampling_schedule": "balanced_cycle",
                "global_sampling_interval_steps": 5,
            }
        )
        policy_hash = candidate_policy_contract_hash(config)
        config_path.write_text(json.dumps(config), encoding="utf-8")

        summary_path = run_dir / "run_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["portfolio_catchup_state"].update(
            {
                "comparison_arm_id": "balanced-h5-x3",
                "elastic_budget_cap_tokens": ELASTIC_BUDGET_CAP_TOKENS,
                "candidate_policy_contract_hash": policy_hash,
            }
        )
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        for checkpoint_path in (
            Path(summary["latest_checkpoint_path"]),
            Path(summary["portfolio_catchup_state"]["confirmation_checkpoint_path"]),
        ):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint["portfolio_catchup_state"].update(
                summary["portfolio_catchup_state"]
            )
            torch.save(checkpoint, checkpoint_path)
        confirmation_path = Path(
            summary["portfolio_catchup_state"]["confirmation_checkpoint_path"]
        )
        summary["portfolio_catchup_state"]["confirmation_checkpoint_sha256"] = (
            hashlib.sha256(confirmation_path.read_bytes()).hexdigest()
        )
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

    report = portfolio_catchup(
        runs,
        target_path,
        tmp_path / "balanced-h5-analysis",
        candidate_arm="balanced-h5-x3",
    )
    assert report["comparison_arm_id"] == "balanced-h5-x3"
    assert report["candidate_policy_contract_hash"]
    assert report["general_portfolio_catchup_claim"] is False


def test_portfolio_bundle_discovers_completed_candidate_arms(tmp_path):
    target_path, targets = _freeze(tmp_path)
    uniform_runs = _candidate_runs(
        tmp_path / "uniform", target_path, targets, seeds=(42,)
    )
    fixed_budget_runs = _candidate_runs(
        tmp_path / "fixed-budget",
        target_path,
        targets,
        candidate_arm_id="uniform_h1_4b",
        seeds=(42,),
    )

    report = portfolio_catchup_bundle(
        [*uniform_runs, *fixed_budget_runs],
        target_path,
        tmp_path / "candidate-analysis",
    )

    assert report["status"] == "completed_candidate_arms_analyzed"
    assert [row["comparison_arm_id"] for row in report["candidate_arms"]] == [
        "uniform_h1_3b",
        "uniform_h1_4b",
    ]
    for row in report["candidate_arms"]:
        assert Path(row["report_path"]).is_file()
        assert Path(row["final_holdout_selection_manifest"]).is_file()


def test_extension_report_is_diagnostic_and_preserves_4b_cost_accounting(tmp_path):
    target_path, targets = _freeze(tmp_path)
    runs = _candidate_runs(
        tmp_path,
        target_path,
        targets,
        candidate_arm_id="uniform_h1_4b",
    )
    output = tmp_path / "uniform-h1-4b-analysis"
    report = portfolio_catchup(
        runs,
        target_path,
        output,
        candidate_arm="uniform_h1_4b",
    )
    assert report["comparison_arm_id"] == "uniform_h1_4b"
    assert report["arm_catchup_confirmed"] is True
    assert report["general_portfolio_catchup_claim"] is False
    assert report["post_hoc_diagnostic"] is True
    assert report["realized_full_run_spend_over_4B"] == 1.0
    assert report["final_holdout_selection_mode"] == (
        "portfolio_confirmation_diagnostic"
    )
    selection = json.loads(
        (output / "final_holdout_selection_manifest.json").read_text()
    )
    assert selection["claim_eligible"] is False
    assert selection["candidate_budget_tokens"] == (
        AGGREGATE_REFERENCE_BUDGET_TOKENS
    )
    normalized = evaluate_holdout_cli._portfolio_selection_entries(
        output / "final_holdout_selection_manifest.json"
    )
    assert len(normalized) == 15

    figure_paths = generate_figures(
        tmp_path,
        tmp_path / "uniform-h1-4b-figures",
        comparison_manifest=output / "portfolio_catchup_report.json",
        include_final_holdout=False,
        dpi=40,
    )
    assert "ppl_vs_size.png" in {path.name for path in figure_paths}
    assert "portfolio_worst_width_deficit.png" in {
        path.name for path in figure_paths
    }


def test_portfolio_figure_identity_comes_from_candidate_contract():
    report = {
        "reference_budget_tokens": REFERENCE_BUDGET_TOKENS,
        "elastic_budget_cap_tokens": AGGREGATE_REFERENCE_BUDGET_TOKENS,
        "candidate_model_variant": "slicing",
    }
    nested_all = _config("elastic_candidate", seed=42)
    nested_all["run"]["sampling_mode"] = "nested-all"
    nested_all["training"]["token_budget"] = AGGREGATE_REFERENCE_BUDGET_TOKENS
    identity = _portfolio_figure_identity(report, nested_all, seed_count=1)
    assert identity["elastic_short_label"] == "Elastic: Nested-all (4B)"
    assert identity["protocol_subtitle"] == (
        "Nested-all · Slicing · cosine over 4B · n=1 seed"
    )

    concat = _config("elastic_candidate", seed=42)
    concat["model"]["variant"] = "concat"
    report["candidate_model_variant"] = "concat"
    identity = _portfolio_figure_identity(report, concat, seed_count=3)
    assert identity["elastic_short_label"] == "Elastic: Uniform H=1 (4B)"
    assert "Concat" in identity["protocol_subtitle"]
    assert identity["protocol_subtitle"].endswith("n=3 seeds")


def test_nested_all_4b_reports_full_optimizer_and_subnetwork_exposure(tmp_path):
    target_path, targets = _freeze(tmp_path)
    runs = _candidate_runs(
        tmp_path,
        target_path,
        targets,
        candidate_arm_id="nested_all_4b",
    )
    report = portfolio_catchup(
        runs[:1],
        target_path,
        tmp_path / "nested-all-4b-analysis",
        candidate_arm="nested_all_4b",
    )

    assert report["comparison_arm_id"] == "nested_all_4b"
    assert report["post_hoc_diagnostic"] is True
    assert report["general_portfolio_catchup_claim"] is False
    assert report["elastic_budget_cap_tokens"] == (
        AGGREGATE_REFERENCE_BUDGET_TOKENS
    )
    assert report["realized_full_run_spend_over_4B"] == 1.0
    assert report["subnetwork_gradient_evaluations_per_optimizer_step"] == 4
    assert report["realized_subnetwork_target_tokens"] == (
        4 * AGGREGATE_REFERENCE_BUDGET_TOKENS
    )


def test_concat_extension_reuses_targets_with_only_variant_provenance_delta(tmp_path):
    target_path, targets = _freeze(tmp_path)
    runs = _candidate_runs(
        tmp_path,
        target_path,
        targets,
        candidate_arm_id="concat_uniform_h1_4b",
    )
    output = tmp_path / "concat-uniform-h1-4b-analysis"
    report = portfolio_catchup(
        runs,
        target_path,
        output,
        candidate_arm="concat_uniform_h1_4b",
    )
    assert report["comparison_arm_id"] == "concat_uniform_h1_4b"
    assert report["post_hoc_diagnostic"] is True
    assert report["general_portfolio_catchup_claim"] is False
    assert report["reference_model_variant"] == "slicing"
    assert report["candidate_model_variant"] == "concat"
    assert report["allowed_reference_provenance_differences"] == ["model_variant"]

    partial_output = tmp_path / "concat-seed-42-analysis"
    partial_report = portfolio_catchup(
        runs[:1],
        target_path,
        partial_output,
        candidate_arm="concat_uniform_h1_4b",
    )
    assert partial_report["status"] == "provisional_seed_subset_confirmed"
    assert partial_report["observed_seeds"] == [42]
    assert partial_report["missing_seeds"] == [43, 44]
    assert partial_report["post_hoc_diagnostic"] is True
    assert partial_report["general_portfolio_catchup_claim"] is False
    partial_selection = json.loads(
        (partial_output / "final_holdout_selection_manifest.json").read_text()
    )
    assert partial_selection["required_checkpoint_count"] == 5
    assert partial_selection["selection_mode"] == (
        "portfolio_confirmation_diagnostic"
    )

    figure_paths = generate_figures(
        tmp_path,
        tmp_path / "concat-uniform-h1-4b-figures",
        comparison_manifest=output / "portfolio_catchup_report.json",
        include_final_holdout=False,
        dpi=40,
    )
    assert {path.name for path in figure_paths} == {
        "learning_rate_schedule.png",
        "ppl_vs_size.png",
        "portfolio_per_granularity_deficits.png",
        "portfolio_validation_loss_over_tokens.png",
        "portfolio_worst_width_deficit.png",
    }

    drifted_config_path = runs[0] / "config.json"
    drifted = json.loads(drifted_config_path.read_text(encoding="utf-8"))
    drifted["model"]["d_model"] = 128
    drifted_config_path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(PortfolioAnalysisError, match="d_model"):
        portfolio_catchup(
            runs,
            target_path,
            tmp_path / "drifted-concat-analysis",
            candidate_arm="concat_uniform_h1_4b",
        )
    with pytest.raises(ValueError, match="d_model"):
        generate_figures(
            tmp_path,
            tmp_path / "drifted-concat-figures",
            comparison_manifest=output / "portfolio_catchup_report.json",
            include_final_holdout=False,
            dpi=40,
        )


def test_schema1_compatibility_is_narrowly_limited_to_legacy_references():
    legacy = _config("standalone_reference", seed=42, width="g250")
    legacy_contract = legacy["controlled_experiment"]["portfolio_catchup"]
    legacy_contract["schema_version"] = 1
    legacy_contract["lr_selection_manifest_path"] = None
    legacy_contract["lr_selection_manifest_hash"] = None
    _validate_portfolio_controlled_experiment(legacy)

    unresolved_legacy = {
        "controlled_experiment": json.loads(
            json.dumps(legacy["controlled_experiment"])
        )
    }
    _resolve_portfolio_controlled_experiment(unresolved_legacy)
    resolved_legacy_contract = unresolved_legacy["controlled_experiment"][
        "portfolio_catchup"
    ]
    assert resolved_legacy_contract["schema_version"] == 1
    assert resolved_legacy_contract["lr_selection_manifest_path"] is None
    assert resolved_legacy_contract["lr_selection_manifest_hash"] is None

    _, legacy_inputs = build_comparison_control_signature(legacy)
    assert (
        legacy_inputs["portfolio_catchup_contract"]["lr_selection_manifest_hash"]
        is None
    )

    current = _config("standalone_reference", seed=42, width="g250")
    _, current_inputs = build_comparison_control_signature(current)
    assert (
        "lr_selection_manifest_hash"
        not in current_inputs["portfolio_catchup_contract"]
    )

    bad_reference = json.loads(json.dumps(legacy))
    bad_reference["controlled_experiment"]["portfolio_catchup"][
        "lr_selection_manifest_hash"
    ] = "0" * 64
    with pytest.raises(ConfigError, match="not supported"):
        _validate_portfolio_controlled_experiment(bad_reference)

    legacy_candidate = _config("elastic_candidate", seed=42)
    candidate_contract = legacy_candidate["controlled_experiment"][
        "portfolio_catchup"
    ]
    candidate_contract["schema_version"] = 1
    candidate_contract["lr_selection_manifest_path"] = None
    candidate_contract["lr_selection_manifest_hash"] = None
    with pytest.raises(ConfigError, match="schema 1"):
        _validate_portfolio_controlled_experiment(legacy_candidate)


def test_schema1_metrics_header_is_migrated_on_resume(tmp_path):
    legacy_columns = list(METRICS_COLUMNS)
    insertion_index = legacy_columns.index("portfolio_target_manifest_hash") + 1
    legacy_columns.insert(insertion_index, "portfolio_lr_selection_hash")
    output_dir = tmp_path / "legacy-reference"
    output_dir.mkdir()
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=legacy_columns)
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "legacy-reference",
                "step": 64,
                "split": "validation",
                "portfolio_lr_selection_hash": "",
                "tokens_seen": 524288,
            }
        )

    journal = MetricsJournal(output_dir, checkpoint_step=64)
    assert journal.has_validation_at_step(64)
    with metrics_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        assert reader.fieldnames == METRICS_COLUMNS
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["step"] == "64"
    assert "portfolio_lr_selection_hash" not in rows[0]


def test_standalone_recipe_does_not_explicitly_configure_elastic_sampling_fields():
    config = load_yaml_config(
        "configs/controlled_exps/tinystories_instruct_portfolio_catchup.yaml"
    )
    assert config["run"]["sampling_mode"] == "standalone"
    assert "global_sampling_interval_steps" not in config["model"]
    assert "global_sampling_schedule" not in config["model"]
    _resolve_global_sampling_interval_steps(config)
    _resolve_global_sampling_schedule(config)


def test_manifest_links_and_online_streak_reset_and_confirmation(tmp_path):
    target_path, targets = _freeze(tmp_path)
    config = _config(
        "elastic_candidate",
        seed=42,
        target=target_path,
    )
    state = build_portfolio_catchup_state(config)
    assert state is not None
    target_losses = {
        width: state["targets"][width]["target_loss"] for width in GRANULARITIES
    }

    def results(failing_width=None):
        return [
            {
                "granularity": width,
                "loss": target_losses[width]
                + (LOSS_TOLERANCE * 2 if width == failing_width else 0.0),
                "perplexity": math.exp(target_losses[width]),
            }
            for width in GRANULARITIES
        ]

    for step in range(1, 4):
        state, _, confirmed = update_portfolio_catchup_state(
            state, results(), step=step, tokens_seen=step * 100
        )
        assert confirmed is False
    state, _, _ = update_portfolio_catchup_state(
        state, results("g500"), step=4, tokens_seen=400
    )
    assert state["streak_length"] == 0
    assert state["streak_onset_step"] is None
    for step in range(5, 10):
        state, _, confirmed = update_portfolio_catchup_state(
            state, results(), step=step, tokens_seen=step * 100
        )
    assert confirmed is True
    assert state["streak_onset_step"] == 5
    assert state["confirmation_step"] == 9
    validate_portfolio_catchup_state(state, config=config)

    drifted = json.loads(json.dumps(state))
    drifted["target_manifest_hash"] = "0" * 64
    with pytest.raises(PortfolioCatchupError, match="contract"):
        validate_portfolio_catchup_state(drifted, config=config)
    _validate_portfolio_manifest_link(
        target_path, targets["manifest_hash"], field_prefix="target_manifest"
    )
    with pytest.raises(ConfigError, match="hash mismatch"):
        _validate_portfolio_manifest_link(
            target_path, "0" * 64, field_prefix="target_manifest"
        )


def test_width_local_streaks_never_substitute_for_joint_qualification(tmp_path):
    target_path, targets = _freeze(tmp_path)
    config = _config(
        "elastic_candidate",
        seed=42,
        target=target_path,
    )
    state = build_portfolio_catchup_state(config)
    assert state is not None
    step = 0
    for qualifying_width in GRANULARITIES:
        for _ in range(5):
            step += 1
            results = [
                {
                    "granularity": width,
                    "loss": state["targets"][width]["target_loss"]
                    + (0.0 if width == qualifying_width else LOSS_TOLERANCE * 2),
                }
                for width in GRANULARITIES
            ]
            state, _, newly_confirmed = update_portfolio_catchup_state(
                state,
                results,
                step=step,
                tokens_seen=step * 100,
            )
            assert newly_confirmed is False
    assert state["confirmed"] is False
    assert state["streak_length"] == 0


def test_portfolio_resume_signature_rejects_scientific_contract_changes(tmp_path):
    target_path, targets = _freeze(tmp_path)
    config = _config(
        "elastic_candidate",
        seed=42,
        target=target_path,
    )
    baseline, _ = build_comparison_control_signature(config)
    mutations = [
        (
            "role",
            lambda value: value["controlled_experiment"].update(
                comparison_role="standalone_reference"
            ),
        ),
        (
            "budget",
            lambda value: value["training"].update(
                token_budget=REFERENCE_BUDGET_TOKENS
            ),
        ),
        ("lr", lambda value: value["training"].update(resolved_learning_rate=0.006)),
        (
            "schedule",
            lambda value: value["training"].update(
                scheduler={"name": "constant", "kwargs": {}}
            ),
        ),
        (
            "widths",
            lambda value: value["model"].update(granularities=list(GRANULARITIES[:-1])),
        ),
        (
            "initializer",
            lambda value: value["model"].update(initializer_range=0.03),
        ),
        ("corpus", lambda value: value["dataset"].update(corpus_hash="other-corpus")),
        ("topology", lambda value: value["training"].update(effective_world_size=2)),
    ]
    for _, mutate in mutations:
        changed = json.loads(json.dumps(config))
        mutate(changed)
        signature, _ = build_comparison_control_signature(changed)
        assert signature != baseline


def test_confirmation_checkpoint_save_once_and_artifact_failure(tmp_path, monkeypatch):
    target_path, targets = _freeze(tmp_path)
    config = _config(
        "elastic_candidate",
        seed=42,
        target=target_path,
    )
    config["run"]["output_dir"] = str(tmp_path / "checkpoint-run")
    state = build_portfolio_catchup_state(config)
    state.update(
        {
            "streak_length": 5,
            "streak_onset_step": 1,
            "streak_onset_tokens": 100,
            "confirmed": True,
            "confirmation_step": 5,
            "confirmation_tokens": 500,
        }
    )
    run_state = {"portfolio_catchup_state": state}

    def fake_save(*args, **kwargs):
        output_path = args[4]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"immutable-confirmation")

    monkeypatch.setattr(checkpointing, "save_model_checkpoint", fake_save)
    path = checkpointing.write_portfolio_confirmation_checkpoint(
        config,
        object(),
        object(),
        object(),
        None,
        run_state,
        step=5,
        joint_max_loss_gap=0.0,
    )
    assert path.is_file()
    assert run_state["portfolio_catchup_state"]["confirmation_checkpoint_saved"]
    with pytest.raises(ConfigError, match="already saved"):
        checkpointing.write_portfolio_confirmation_checkpoint(
            config,
            object(),
            object(),
            object(),
            None,
            run_state,
            step=5,
            joint_max_loss_gap=0.0,
        )

    config["run"]["output_dir"] = str(tmp_path / "failed-run")
    failed_state = build_portfolio_catchup_state(config)
    failed_state.update(
        {
            "streak_length": 5,
            "streak_onset_step": 1,
            "streak_onset_tokens": 100,
            "confirmed": True,
            "confirmation_step": 5,
            "confirmation_tokens": 500,
        }
    )
    failed_run_state = {"portfolio_catchup_state": failed_state}

    def fail_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(checkpointing, "save_model_checkpoint", fail_save)
    with pytest.raises(OSError, match="disk full"):
        checkpointing.write_portfolio_confirmation_checkpoint(
            config,
            object(),
            object(),
            object(),
            None,
            failed_run_state,
            step=5,
            joint_max_loss_gap=0.0,
        )
    assert not failed_run_state["portfolio_catchup_state"][
        "confirmation_checkpoint_saved"
    ]


def test_training_validation_confirms_once_continues_and_resumes_exactly(
    tmp_path, monkeypatch
):
    target_path, targets = _freeze(tmp_path)
    config = _config(
        "elastic_candidate",
        seed=42,
        target=target_path,
    )
    state = build_portfolio_catchup_state(config)
    run_state = {"portfolio_catchup_state": state}
    calls = []

    def fake_confirmation(*args, **kwargs):
        calls.append(kwargs["step"])
        current = args[5]["portfolio_catchup_state"]
        path = tmp_path / f"portfolio_catchup_step_{kwargs['step']}.pt"
        path.write_bytes(b"confirmation")
        current["confirmation_checkpoint_path"] = str(path)
        current["confirmation_checkpoint_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        current["confirmation_checkpoint_saved"] = True
        return path

    monkeypatch.setattr(
        checkpointing,
        "write_portfolio_confirmation_checkpoint",
        fake_confirmation,
    )
    target_losses = {
        width: state["targets"][width]["target_loss"] for width in GRANULARITIES
    }
    results = [
        {
            "granularity": width,
            "loss": target_losses[width],
            "perplexity": math.exp(target_losses[width]),
        }
        for width in GRANULARITIES
    ]
    snapshots = []
    for step in range(1, 7):
        decorated = training_steps._process_portfolio_catchup_validation(
            config,
            results,
            model=object(),
            optimizer=object(),
            scheduler=object(),
            step=step,
            tokens_seen=step * 100,
            heartbeat_writer=None,
            run_state=run_state,
        )
        assert all(row["portfolio_qualifies"] for row in decorated)
        if step in {1, 3, 5}:
            snapshots.append(
                json.loads(json.dumps(run_state["portfolio_catchup_state"]))
            )
    assert calls == [5]
    assert run_state["portfolio_catchup_state"]["evaluation_count"] == 6
    assert run_state["portfolio_catchup_state"]["confirmation_step"] == 5
    for snapshot in snapshots:
        validate_portfolio_catchup_state(snapshot, config=config)

    # Continuing from an inside-streak snapshot reproduces the same confirmation.
    resumed = snapshots[1]
    for step in (4, 5):
        resumed, _, _ = update_portfolio_catchup_state(
            resumed, results, step=step, tokens_seen=step * 100
        )
    assert resumed["confirmation_step"] == 5
    assert resumed["confirmation_tokens"] == 500


def test_reference_matrix_fixed_recipe_and_portfolio_report(tmp_path):
    target_path, targets = _freeze(tmp_path)
    assert len(targets["targets"]) == 3
    assert sum(len(value) for value in targets["targets"].values()) == 12
    runs = _candidate_runs(tmp_path, target_path, targets)
    output = tmp_path / "catchup-analysis"
    report = portfolio_catchup(runs, target_path, output)
    assert report["general_portfolio_catchup_claim"] is True
    assert report["learning_rate"] == FIXED_LEARNING_RATE
    assert report["optimizer_recipe_policy"] == "same_fixed_recipe_across_roles"
    assert not any(key.startswith("lr_screen") for key in report)
    assert "lr_selection_manifest_hash" not in report
    assert report["budget_summary"]["cross_seed_required_tokens"] == 600
    assert report["budget_summary"]["cross_seed_t_star_over_4B"] == pytest.approx(
        600 / AGGREGATE_REFERENCE_BUDGET_TOKENS
    )
    assert report["realized_full_run_spend_over_4B"] == 0.75
    assert (
        len(
            json.loads((output / "final_holdout_selection_manifest.json").read_text())[
                "entries"
            ]
        )
        == 15
    )
    assert (output / "portfolio_joint_deficit.png").is_file()
    assert (output / "portfolio_per_granularity_deficits.png").is_file()

    figure_dir = tmp_path / "figures"
    figure_dir.mkdir()
    stale_holdout_path = figure_dir / "final_holdout_ppl_vs_size.png"
    stale_holdout_path.write_bytes(b"stale")
    figure_paths = generate_figures(
        tmp_path,
        figure_dir,
        comparison_manifest=output / "portfolio_catchup_report.json",
        include_final_holdout=False,
        dpi=40,
    )
    assert {path.name for path in figure_paths} == {
        "learning_rate_schedule.png",
        "ppl_vs_size.png",
        "portfolio_per_granularity_deficits.png",
        "portfolio_validation_loss_over_tokens.png",
        "portfolio_worst_width_deficit.png",
    }
    assert not stale_holdout_path.exists()


def test_one_seed_candidate_emits_provisional_reports_holdout_and_figures(tmp_path):
    target_path, targets = _freeze(tmp_path)
    candidate_runs = _candidate_runs(tmp_path, target_path, targets)
    seed_42_run = candidate_runs[:1]
    output = tmp_path / "seed-42-catchup-analysis"

    report = portfolio_catchup(seed_42_run, target_path, output)

    assert report["status"] == "provisional_seed_subset_confirmed"
    assert report["expected_seeds"] == [42, 43, 44]
    assert report["observed_seeds"] == [42]
    assert report["missing_seeds"] == [43, 44]
    assert report["seed_coverage_complete"] is False
    assert report["provisional_analysis"] is True
    assert report["observed_seed_catchup_confirmed"] is True
    assert report["arm_catchup_confirmed"] is False
    assert report["general_portfolio_catchup_claim"] is False
    assert report["budget_summary"] is None
    assert report["observed_seed_budget_summary"]["scope"] == (
        "observed_seed_subset"
    )
    assert report["observed_seed_budget_summary"][
        "observed_seed_required_tokens"
    ] == 600
    assert report["final_holdout_selection_status"] == "ready_provisional"

    selection_path = output / "final_holdout_selection_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["observed_seeds"] == [42]
    assert selection["missing_seeds"] == [43, 44]
    assert selection["claim_eligible"] is False
    assert selection["required_checkpoint_count"] == 5
    assert selection["expected_full_checkpoint_count"] == 15
    assert len(selection["entries"]) == 5
    assert {int(entry["seed"]) for entry in selection["entries"]} == {42}
    assert len(evaluate_holdout_cli._portfolio_selection_entries(selection_path)) == 5

    validation_figure_paths = generate_figures(
        tmp_path,
        tmp_path / "seed-42-validation-figures",
        comparison_manifest=output / "portfolio_catchup_report.json",
        include_final_holdout=False,
        dpi=40,
    )
    assert "ppl_vs_size.png" in {path.name for path in validation_figure_paths}

    _write_holdout_results(selection_path)
    holdout_report = final_holdout(selection_path, tmp_path / "seed-42-holdout")
    assert holdout_report["status"] == "provisional_seed_subset_equivalent"
    assert holdout_report["observed_seeds"] == [42]
    assert holdout_report["missing_seeds"] == [43, 44]
    assert holdout_report["required_checkpoint_count"] == 5
    assert holdout_report["all_pairs_within_tolerance"] is True
    assert holdout_report["provisional_seed_subset_equivalence"] is True
    assert holdout_report["general_portfolio_equivalence_claim"] is False

    final_figure_paths = generate_figures(
        tmp_path,
        tmp_path / "seed-42-final-figures",
        comparison_manifest=output / "portfolio_catchup_report.json",
        dpi=40,
    )
    assert "final_holdout_ppl_vs_size.png" in {
        path.name for path in final_figure_paths
    }
    assert "portfolio_final_holdout_deficit_vs_size.png" in {
        path.name for path in final_figure_paths
    }

    complete_report = portfolio_catchup(candidate_runs, target_path, output)
    assert complete_report["status"] == "portfolio_catchup_confirmed"
    assert complete_report["observed_seeds"] == [42, 43, 44]
    assert complete_report["seed_coverage_complete"] is True
    replaced_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert replaced_selection["required_checkpoint_count"] == 15
    assert len(replaced_selection["entries"]) == 15


def test_freeze_references_accepts_null_metadata_schema1_runs(tmp_path):
    runs = _reference_runs(tmp_path)
    for run_dir in runs:
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        contract = config["controlled_experiment"]["portfolio_catchup"]
        contract["schema_version"] = 1
        contract["lr_selection_manifest_path"] = None
        contract["lr_selection_manifest_hash"] = None
        config_path.write_text(json.dumps(config), encoding="utf-8")

    targets = freeze_references(runs, tmp_path / "legacy-reference-analysis")
    assert sum(len(widths) for widths in targets["targets"].values()) == 12


def test_censored_candidate_emits_terminal_diagnostic_holdout_selection(
    tmp_path, monkeypatch
):
    target_path, targets = _freeze(tmp_path)
    runs = _candidate_runs(tmp_path, target_path, targets)
    censored_run = next(
        run
        for run in runs
        if json.loads((run / "config.json").read_text())["run"]["seed"] == 44
    )
    losses = {
        width: [
            targets["targets"]["44"][width]["target_loss"]
            + (0.02 if width == "g750" else 0.0)
        ]
        * 7
        for width in GRANULARITIES
    }
    _write_metrics(
        censored_run / "metrics.csv",
        losses,
        terminal_step=261396,
        terminal_tokens=ELASTIC_BUDGET_CAP_TOKENS,
    )
    summary_path = censored_run / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["portfolio_catchup_state"].update(
        {
            "confirmed": False,
            "streak_onset_step": None,
            "streak_onset_tokens": None,
            "confirmation_step": None,
            "confirmation_tokens": None,
            "confirmation_checkpoint_saved": False,
            "confirmation_checkpoint_path": None,
            "confirmation_checkpoint_sha256": None,
        }
    )
    terminal_path = Path(summary["latest_checkpoint_path"])
    terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
    terminal["portfolio_catchup_state"] = summary["portfolio_catchup_state"]
    torch.save(terminal, terminal_path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    output = tmp_path / "censored-analysis"
    report = portfolio_catchup(runs, target_path, output)
    assert report["status"] == "censored"
    assert report["general_portfolio_catchup_claim"] is False
    assert report["budget_summary"] is None
    assert report["final_holdout_selection_status"] == "ready_diagnostic_terminal_3B"
    selection_path = output / "final_holdout_selection_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["selection_mode"] == "terminal_3B_censored"
    assert selection["claim_eligible"] is False
    elastic_entries = [
        entry
        for entry in selection["entries"]
        if entry["comparison_role"] == "elastic_candidate"
    ]
    assert len(selection["entries"]) == 15
    assert len(elastic_entries) == 3
    assert {
        entry["checkpoint_selection"] for entry in elastic_entries
    } == {"terminal_3B"}
    assert all(entry["checkpoint_step"] == 261396 for entry in elastic_entries)
    assert all(
        entry["checkpoint_tokens"] == ELASTIC_BUDGET_CAP_TOKENS
        for entry in elastic_entries
    )

    partial_output = tmp_path / "censored-seed-44-analysis"
    partial_report = portfolio_catchup(
        [censored_run], target_path, partial_output
    )
    assert partial_report["status"] == "provisional_seed_subset_censored"
    assert partial_report["observed_seeds"] == [44]
    assert partial_report["missing_seeds"] == [42, 43]
    assert partial_report["observed_seed_catchup_confirmed"] is False
    assert partial_report["arm_catchup_confirmed"] is False
    assert partial_report["general_portfolio_catchup_claim"] is False
    partial_selection = json.loads(
        (partial_output / "final_holdout_selection_manifest.json").read_text()
    )
    assert partial_selection["required_checkpoint_count"] == 5
    assert partial_selection["selection_mode"] == "terminal_3B_censored"
    assert {
        entry["checkpoint_selection"]
        for entry in partial_selection["entries"]
        if entry["comparison_role"] == "elastic_candidate"
    } == {"terminal_3B"}

    _write_holdout_results(selection_path)
    holdout_report = final_holdout(selection_path, tmp_path / "diagnostic-holdout")
    assert holdout_report["selection_mode"] == "terminal_3B_censored"
    assert holdout_report["all_pairs_within_tolerance"] is True
    assert holdout_report["diagnostic_terminal_3B_equivalence"] is True
    assert holdout_report["general_portfolio_equivalence_claim"] is False
    assert holdout_report["status"] == "diagnostic_terminal_3B_equivalent"
    assert "terminal 3B diagnostic" in _portfolio_elastic_selection_label(selection)
    secondary_axis_calls = []
    original_add_loss_secondary_axis = reporting_impl.add_loss_secondary_axis

    def track_loss_secondary_axis(axis):
        secondary_axis_calls.append(axis)
        return original_add_loss_secondary_axis(axis)

    monkeypatch.setattr(
        reporting_impl,
        "add_loss_secondary_axis",
        track_loss_secondary_axis,
    )
    figure_paths = generate_figures(
        tmp_path,
        tmp_path / "censored-figures",
        comparison_manifest=output / "portfolio_catchup_report.json",
        dpi=40,
    )
    assert {path.name for path in figure_paths} == {
        "final_holdout_ppl_vs_size.png",
        "learning_rate_schedule.png",
        "ppl_vs_size.png",
        "portfolio_final_holdout_deficit_vs_size.png",
        "portfolio_per_granularity_deficits.png",
        "portfolio_validation_loss_over_tokens.png",
        "portfolio_worst_width_deficit.png",
    }
    assert len(secondary_axis_calls) == 2


def _write_holdout_results(selection_path: Path, *, failing=False):
    selection = json.loads(selection_path.read_text())
    selected_granularities = tuple(selection.get("granularities", GRANULARITIES))
    standalone_ppl: dict[tuple[int, str], float] = {}
    for entry in selection["entries"]:
        seed = int(entry["seed"])
        role = entry["comparison_role"]
        components = []
        for width in entry["granularities"]:
            base = (
                2.0
                + selected_granularities.index(width) * 0.1
                + (seed - 42) * 0.001
            )
            if role == "standalone_reference":
                standalone_ppl[(seed, width)] = base
                ppl = base
            else:
                ppl = base * (
                    1.006 if failing and seed == 44 and width == "g750" else 1.004
                )
            components.append(
                {
                    "granularity": width,
                    "loss": math.log(ppl),
                    "perplexity": ppl,
                    "evaluation_examples": 512,
                    "evaluation_target_tokens": 1024,
                }
            )
        result_path = Path(entry["result_path"])
        result = {
            "schema_version": 1,
            "run_id": entry["run_id"],
            "checkpoint_path": entry["checkpoint_path"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "final_holdout_manifest_hash": selection["final_holdout_manifest_hash"],
            "ordered_per_granularity_losses": components,
            "result_path": str(result_path),
        }
        result["result_hash"] = stable_hash(result)
        result_path.write_text(json.dumps(result), encoding="utf-8")


def test_matformer_granularity_profile_isolated_pipeline(tmp_path):
    granularities = ("g125", "g250", "g500", "g1000")
    comparison_group = (
        "tinystories_instruct_portfolio_catchup_matformer_granularities_v1"
    )
    target_path, targets = _freeze(
        tmp_path,
        granularities=granularities,
        comparison_group=comparison_group,
        granularity_profile="matformer",
        seeds=(42,),
    )
    assert targets["comparison_group_id"] == comparison_group
    assert targets["granularity_profile"] == "matformer"
    assert targets["granularities"] == list(granularities)
    assert targets["status"] == "references_frozen_provisional"
    assert targets["observed_seeds"] == [42]
    assert targets["missing_seeds"] == [43, 44]
    assert set(targets["targets"]["42"]) == set(granularities)

    for width in granularities:
        _validate_portfolio_controlled_experiment(
            _config(
                "standalone_reference",
                seed=42,
                width=width,
                granularities=granularities,
                comparison_group=comparison_group,
            )
        )
    candidate_config = _config(
        "elastic_candidate",
        seed=42,
        target=target_path,
        granularities=granularities,
        comparison_group=comparison_group,
    )
    _resolve_portfolio_controlled_experiment(candidate_config)
    _validate_portfolio_controlled_experiment(candidate_config)
    _validate_portfolio_aligned_epoch_contract(candidate_config)

    candidate_runs = _candidate_runs(
        tmp_path,
        target_path,
        targets,
        granularities=granularities,
        comparison_group=comparison_group,
        seeds=(42,),
    )
    catchup_output = tmp_path / "matformer-profile-catchup"
    report = portfolio_catchup(
        candidate_runs[:1],
        target_path,
        catchup_output,
    )
    assert report["comparison_group_id"] == comparison_group
    assert report["granularity_profile"] == "matformer"
    assert report["granularities"] == list(granularities)
    assert report["observed_seeds"] == [42]

    selection_path = catchup_output / "final_holdout_selection_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["comparison_group_id"] == comparison_group
    assert selection["granularities"] == list(granularities)
    normalized = evaluate_holdout_cli._portfolio_selection_entries(selection_path)
    assert len(normalized) == 5
    assert {
        tuple(entry["granularities"])
        for entry in normalized
        if entry["comparison_role"] == "elastic_candidate"
    } == {granularities}

    _write_holdout_results(selection_path)
    holdout_report = final_holdout(
        selection_path,
        tmp_path / "matformer-profile-holdout",
    )
    assert holdout_report["comparison_group_id"] == comparison_group
    assert holdout_report["granularity_profile"] == "matformer"
    assert {row["granularity"] for row in holdout_report["comparisons"]} == set(
        granularities
    )

    figure_paths = generate_figures(
        tmp_path,
        tmp_path / "matformer-profile-figures",
        comparison_manifest=catchup_output / "portfolio_catchup_report.json",
        dpi=40,
    )
    assert "ppl_vs_size.png" in {path.name for path in figure_paths}
    assert "portfolio_per_granularity_deficits.png" in {
        path.name for path in figure_paths
    }

    extension_runs = _candidate_runs(
        tmp_path / "matformer-profile-extension",
        target_path,
        targets,
        candidate_arm_id="nested_all_4b",
        granularities=granularities,
        comparison_group=comparison_group,
        seeds=(42,),
    )
    extension_report = portfolio_catchup(
        extension_runs[:1],
        target_path,
        tmp_path / "matformer-profile-nested-all-analysis",
        candidate_arm="nested_all_4b",
    )
    assert extension_report["comparison_arm_id"] == "nested_all_4b"
    assert extension_report["granularities"] == list(granularities)
    assert extension_report["granularity_profile"] == "matformer"


def test_final_holdout_all_15_exact_checkpoints_and_failure_blocks_claim(tmp_path):
    target_path, targets = _freeze(tmp_path)
    runs = _candidate_runs(tmp_path, target_path, targets)
    catchup_output = tmp_path / "catchup-analysis"
    portfolio_catchup(runs, target_path, catchup_output)
    holdout_selection = catchup_output / "final_holdout_selection_manifest.json"
    selection = json.loads(holdout_selection.read_text(encoding="utf-8"))
    assert selection["selection_mode"] == "portfolio_confirmation"
    assert selection["claim_eligible"] is True
    _write_holdout_results(holdout_selection)
    report = final_holdout(holdout_selection, tmp_path / "holdout-pass")
    assert report["general_portfolio_equivalence_claim"] is True
    assert report["diagnostic_terminal_3B_equivalence"] is None
    assert len(report["comparisons"]) == 12
    figure_paths = generate_figures(
        tmp_path,
        tmp_path / "confirmed-figures",
        comparison_manifest=catchup_output / "portfolio_catchup_report.json",
        refresh_counts=False,
        dpi=40,
    )
    assert "five-validation confirmation" in _portfolio_elastic_selection_label(
        selection
    )
    assert "final_holdout_ppl_vs_size.png" in {
        path.name for path in figure_paths
    }
    assert "portfolio_final_holdout_deficit_vs_size.png" in {
        path.name for path in figure_paths
    }

    _write_holdout_results(holdout_selection, failing=True)
    report = final_holdout(holdout_selection, tmp_path / "holdout-fail")
    assert report["general_portfolio_equivalence_claim"] is False

    _write_holdout_results(holdout_selection)
    selection_manifest = json.loads(holdout_selection.read_text())
    stale_result_path = Path(selection_manifest["entries"][0]["result_path"])
    stale_result = json.loads(stale_result_path.read_text())
    stale_result["checkpoint_sha256"] = "0" * 64
    stale_result.pop("result_hash")
    stale_result["result_hash"] = stable_hash(stale_result)
    stale_result_path.write_text(json.dumps(stale_result), encoding="utf-8")
    with pytest.raises(PortfolioAnalysisError, match="checksum mismatch"):
        final_holdout(holdout_selection, tmp_path / "holdout-wrong-result-checksum")

    _write_holdout_results(holdout_selection)
    Path(selection_manifest["entries"][0]["checkpoint_path"]).write_bytes(b"stale")
    with pytest.raises(PortfolioAnalysisError, match="missing or stale"):
        final_holdout(holdout_selection, tmp_path / "holdout-stale")


def test_manifest_evaluator_passes_terminal_checkpoint_explicitly(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "candidate"
    run_dir.mkdir()
    checkpoint_path = run_dir / "latest.pt"
    checkpoint_path.write_bytes(b"terminal-checkpoint")
    result_path = run_dir / "final_holdout_results.json"
    entry = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_selection": "terminal_3B",
    }
    monkeypatch.setattr(
        evaluate_holdout_cli,
        "_portfolio_selection_entries",
        lambda _path: [entry],
    )
    evaluated = {}

    def fake_evaluate(run_directory, *, checkpoint_path=None, device=None):
        evaluated["run_dir"] = Path(run_directory)
        evaluated["checkpoint_path"] = checkpoint_path
        evaluated["device"] = device
        return {
            "checkpoint_path": str(entry["checkpoint_path"]),
            "result_path": str(result_path),
        }

    monkeypatch.setattr(final_holdout_module, "evaluate_final_holdout", fake_evaluate)
    assert (
        evaluate_holdout_cli.main(
            ["--selection-manifest", "selection.json", "--device", "cpu"]
        )
        == 0
    )
    assert evaluated == {
        "run_dir": run_dir,
        "checkpoint_path": checkpoint_path,
        "device": "cpu",
    }


def test_skip_existing_verifies_requested_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint_path = run_dir / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint-a")
    (run_dir / "run_summary.json").write_text(
        json.dumps({"run_id": "run", "status": "completed"}), encoding="utf-8"
    )
    result_path = run_dir / "final_holdout_results.json"
    result = {
        "schema_version": 1,
        "run_id": "run",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "result_path": str(result_path),
    }
    result["result_hash"] = stable_hash(result)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert (
        resolve_existing_final_holdout_result(run_dir, checkpoint_path=checkpoint_path)
        is not None
    )
    checkpoint_path.write_bytes(b"checkpoint-b")
    with pytest.raises(FinalHoldoutError, match="stale"):
        resolve_existing_final_holdout_result(run_dir, checkpoint_path=checkpoint_path)


def test_reference_matrix_accepts_seed_subsets_but_rejects_incomplete_panels(
    tmp_path,
):
    runs = _reference_runs(tmp_path)
    targets = freeze_references(runs[:4], tmp_path / "one-seed-reference-analysis")
    assert targets["status"] == "references_frozen_provisional"
    assert targets["observed_seeds"] == [42]
    assert targets["missing_seeds"] == [43, 44]
    assert set(targets["targets"]) == {"42"}

    with pytest.raises(PortfolioAnalysisError, match="complete four-width panel"):
        freeze_references(runs[:3], tmp_path / "bad-reference-analysis")


def test_portfolio_docs_referenced_commands_exist():
    assert Path(
        "configs/controlled_exps/tinystories_instruct_portfolio_catchup.yaml"
    ).is_file()
    assert Path("scripts/analyze_tinystories_portfolio_catchup.py").is_file()
    doc_path = Path("docs/tinystories-portfolio-catchup.md")
    assert doc_path.is_file()
    doc = doc_path.read_text(encoding="utf-8")
    assert "Run this section first in one **zsh** session" in doc
    assert "exec bash" not in doc
    assert "mapfile" not in doc
    assert "declare -A" not in doc
    assert "```zsh" not in doc
    assert "```bash" in doc
    assert doc.count("## 1. Freeze the standalone reference lane") == 1
    assert doc.count("## 2. Run a reusable elastic variant") == 1
    assert doc.count("## 3. Resume timed-out training jobs") == 1
    assert doc.count("## 4. Analyze every completed candidate in the lane") == 1
    assert doc.count("## 5. Submit holdout jobs for the discovered selections") == 1
    assert doc.count("## 6. Report completed holdouts and render figures") == 1
    assert doc.count("export TARGET_MANIFEST=") == 1
    assert doc.count("mkdir -p") == 1
    assert "mkdir -p logs" in doc

    shell_blocks = re.findall(r"```bash\n(.*?)\n```", doc, flags=re.DOTALL)
    assert shell_blocks
    syntax_shell = shutil.which("zsh")
    if syntax_shell is None:
        return
    for index, block in enumerate(shell_blocks):
        result = subprocess.run(
            [syntax_shell, "-n"],
            input=block,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, f"shell block {index}: {result.stderr}"
