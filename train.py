"""CLI wrapper for config-driven training runs."""

from __future__ import annotations

import argparse

import src.training.run as training_run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a MatFormer training job.")
    parser.add_argument("--config", required=True, help="Path to the run config YAML")
    parser.add_argument("--run-id", help="Override the run identifier")
    parser.add_argument(
        "--output-root",
        help="Override the run output root before resolving the config",
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
    training_run.run_from_config_path(
        args.config,
        run_id=args.run_id,
        overrides=overrides,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
