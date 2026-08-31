"""CLI wrapper for config-driven training runs."""

from __future__ import annotations

import argparse
import json

from src.utils.config import ConfigError, resolve_run_config
from src.utils.reproducibility import (
    build_full_run_signature,
    build_paired_control_signature,
)


def _pre_nested_warmup_preflight(training: dict) -> dict:
    """Return the compact warmup contract needed to validate a launch."""
    warmup = training.get("pre_nested_warmup", {})
    keys = (
        "enabled",
        "active",
        "duration",
        "unit",
        "policy",
        "action_interval_steps",
        "passes",
        "controller_start_step",
    )
    return {key: warmup[key] for key in keys if key in warmup}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a MatFormer training job.")
    parser.add_argument("--config", required=True, help="Path to the run config YAML")
    parser.add_argument("--run-id", help="Override the run identifier")
    parser.add_argument(
        "--output-root",
        help="Override the run output root before resolving the config",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Resolve and print reproducibility controls without starting training",
    )
    parser.add_argument(
        "--output-dir",
        help="Write artifacts to an explicit output directory",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config override in dotted.key=value form",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    overrides = list(args.override)
    if args.output_root:
        overrides.append(f"run.output_root={args.output_root}")
    if args.preflight:
        try:
            resolved = resolve_run_config(
                args.config,
                run_id=args.run_id,
                overrides=overrides,
                output_dir=args.output_dir,
                create_output_dirs=False,
            )
        except ConfigError as error:
            raise SystemExit(f"Preflight configuration error: {error}") from error
        paired_control_signature, control_inputs = build_paired_control_signature(
            resolved
        )
        full_run_signature, _ = build_full_run_signature(resolved)
        training = resolved["training"]
        model = resolved["model"]
        print(
            json.dumps(
                {
                    "seed": resolved["run"]["seed"],
                    "reproducibility": resolved["run"]["reproducibility"],
                    "optimizer": training["optimizer"],
                    "optimizer_state_scope": training["optimizer_state_scope"],
                    "requested_optimizer_state_scope": training.get(
                        "requested_optimizer_state_scope"
                    ),
                    "optimizer_scheduler_clock": training[
                        "optimizer_scheduler_clock"
                    ],
                    "requested_optimizer_scheduler_clock": training.get(
                        "requested_optimizer_scheduler_clock"
                    ),
                    "optimizer_state_contract": training[
                        "optimizer_state_contract"
                    ],
                    "optimizer_state_eligibility": training[
                        "optimizer_state_eligibility"
                    ],
                    "full_run_signature": full_run_signature,
                    "paired_control_signature": paired_control_signature,
                    "learning_rate": training["resolved_learning_rate"],
                    "token_budget": training["token_budget"],
                    "batch_size_per_process": training[
                        "batch_size_per_process"
                    ],
                    "gradient_accumulation_steps": training[
                        "gradient_accumulation_steps"
                    ],
                    "dataset_sample_limit": resolved["dataset"].get(
                        "sample_limit"
                    ),
                    "tokenization_keep_in_memory": resolved["dataset"].get(
                        "tokenization_keep_in_memory",
                        False,
                    ),
                    "dataset_mode": resolved["dataset"].get("mode", "raw_tokenized"),
                    "prepared_corpus_dir": resolved["dataset"].get(
                        "prepared_corpus_dir"
                    ),
                    "data_seed": resolved["dataset"].get("data_seed"),
                    "corpus_hash": resolved["dataset"].get("corpus_hash"),
                    "optimizer_iteration": resolved["dataset"].get(
                        "optimizer_iteration"
                    ),
                    "expected_world_size": resolved["training"]
                    .get("distributed", {})
                    .get("expected_world_size"),
                    "effective_world_size": resolved["training"][
                        "effective_world_size"
                    ],
                    "expected_tokens_per_microstep": resolved["training"][
                        "expected_tokens_per_microstep"
                    ],
                    "expected_tokens_per_step": resolved["training"][
                        "expected_tokens_per_step"
                    ],
                    "derived_max_steps": resolved["training"][
                        "derived_max_steps"
                    ],
                    "resolved_warmup_steps": resolved["training"][
                        "resolved_warmup_steps"
                    ],
                    "scheduler": resolved["training"]["scheduler"],
                    "scheduler_specific_kwargs": resolved["training"].get(
                        "scheduler_specific_kwargs", {}
                    ),
                    "scheduler_contract": resolved["training"].get(
                        "scheduler_contract"
                    ),
                    "pre_nested_warmup": _pre_nested_warmup_preflight(
                        resolved["training"]
                    ),
                    "tokenizer_manifest_hash": resolved["model"].get(
                        "tokenizer_manifest_hash"
                    ),
                    "ordered_granularities": resolved["model"]["granularities"],
                    "granularity_prefixes": resolved["model"].get(
                        "granularity_prefixes"
                    ),
                    "resolved_sampling_mode": resolved["model"].get(
                        "resolved_sampling_mode"
                    ),
                    "global_sampling_interval_steps": resolved["model"].get(
                        "global_sampling_interval_steps"
                    ),
                    "global_sampling_schedule": resolved["model"].get(
                        "global_sampling_schedule"
                    ),
                    "global_sampling_schedule_version": resolved["model"].get(
                        "global_sampling_schedule_version"
                    ),
                    "adaptive_sampler_strategy": resolved["model"].get(
                        "adaptive_sampler_strategy"
                    ),
                    "panelgrad": resolved["model"].get("panelgrad"),
                    "adaptive_controller_role": resolved["evaluation"].get(
                        "adaptive_controller"
                    ),
                    "final_holdout_role": resolved["evaluation"].get(
                        "final_holdout"
                    ),
                    "validation": resolved["evaluation"]["validation"],
                    "sampling_policy": {
                        "mode": model.get("resolved_sampling_mode"),
                        "schedule": model.get("global_sampling_schedule"),
                        "schedule_version": model.get(
                            "global_sampling_schedule_version"
                        ),
                        "interval_steps": model.get(
                            "global_sampling_interval_steps"
                        ),
                    },
                    "data_roles": {
                        "data_roles_manifest_hash": resolved.get(
                            "data_roles_manifest_hash"
                        ),
                        "optimizer_training_manifest_hash": resolved.get(
                            "optimizer_training_manifest_hash"
                        ),
                        "controller_manifest_hash": resolved.get(
                            "controller_manifest_hash"
                        ),
                        "ordinary_validation_manifest_hash": resolved.get(
                            "validation_manifest_hash"
                        ),
                        "final_holdout_manifest_hash": resolved.get(
                            "final_holdout_manifest_hash"
                        ),
                    },
                    "run_budget": {
                        "token_budget": training["token_budget"],
                        "global_steps": training["max_steps"],
                        "expected_tokens_per_step": training[
                            "expected_tokens_per_step"
                        ],
                    },
                    "comparison_control_inputs": control_inputs,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    # Keep configuration-only preflight independent of model, optimizer,
    # data-loader, tracker, and output-runtime imports.
    import src.training.run as training_run

    training_run.run_from_config_path(
        args.config,
        run_id=args.run_id,
        overrides=overrides,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
