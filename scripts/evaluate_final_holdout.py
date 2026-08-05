"""Evaluate a completed run's untouched final holdout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed final holdout after training completes.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Completed run directory containing config and summary artifacts.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Explicit checkpoint belonging to the run. Required when ordinary "
            "validation did not select a best checkpoint."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device such as cpu or cuda; defaults to CUDA when available.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from src.evaluation.final_holdout import (
        FinalHoldoutError,
        evaluate_final_holdout,
    )

    try:
        result = evaluate_final_holdout(
            args.run_dir,
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
    except FinalHoldoutError as error:
        print(f"Final holdout evaluation failed: {error}", file=sys.stderr)
        return 2

    print(result["result_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
