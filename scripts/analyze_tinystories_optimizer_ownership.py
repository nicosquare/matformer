#!/usr/bin/env python3
"""Validate and materialize the fixed TinyStories optimizer-ownership campaign."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.optimizer_ownership import preflight_campaign


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    preflight = subcommands.add_parser(
        "preflight", help="Audit data, models and full traces; never train"
    )
    for option in (
        "campaign",
        "prepared-corpus-dir",
        "tokenizer-dir",
        "output-dir",
        "run-output-root",
    ):
        preflight.add_argument("--" + option, required=True)
    args = parser.parse_args(argv)
    try:
        report = preflight_campaign(
            campaign_path=args.campaign,
            prepared_corpus_dir=args.prepared_corpus_dir,
            tokenizer_dir=args.tokenizer_dir,
            output_dir=args.output_dir,
            run_output_root=args.run_output_root,
        )
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Campaign preflight failed: {error}\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
