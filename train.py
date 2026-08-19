"""CLI wrapper for config-driven training runs."""

from __future__ import annotations

import argparse
import json

import src.training.run as training_run
from src.utils.config import resolve_run_config
from src.utils.reproducibility import build_comparison_control_signature


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
        resolved = resolve_run_config(
            args.config,
            run_id=args.run_id,
            overrides=overrides,
            output_dir=args.output_dir,
        )
        _, control_inputs = build_comparison_control_signature(resolved)
        print(
            json.dumps(
                {
                    "seed": resolved["run"]["seed"],
                    "reproducibility": resolved["run"]["reproducibility"],
                    "optimizer": resolved["training"]["optimizer"],
                    "learning_rate": resolved["training"][
                        "resolved_learning_rate"
                    ],
                    "token_budget": resolved["training"]["token_budget"],
                    "batch_size_per_process": resolved["training"][
                        "batch_size_per_process"
                    ],
                    "gradient_accumulation_steps": resolved["training"][
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
                    "comparison_control_inputs": control_inputs,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    training_run.run_from_config_path(
        args.config,
        run_id=args.run_id,
        overrides=overrides,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
