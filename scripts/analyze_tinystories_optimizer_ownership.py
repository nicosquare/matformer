#!/usr/bin/env python3
"""Preflight, freeze and report the fixed TinyStories optimizer-ownership campaign."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.optimizer_ownership import preflight_campaign, freeze_campaign, report_campaign, report_correction_comparison, report_inverse_membership_comparison


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
    freeze = subcommands.add_parser('freeze', help='Freeze saved terminal checkpoints and ordinary-validation evidence')
    freeze.add_argument('--campaign-manifest', required=True)
    locations = freeze.add_mutually_exclusive_group(required=True)
    locations.add_argument('--run-root')
    locations.add_argument('--run-dir', action='append', dest='run_dirs')
    report_parser = subcommands.add_parser('report', help='Export saved terminal endpoints and figures')
    report_parser.add_argument('--manifest', required=True)
    for command_parser in (freeze, report_parser):
        command_parser.add_argument('--output-dir', required=True)
        command_parser.add_argument('--allow-partial', action='store_true')
    correction = subcommands.add_parser('report-corrections', help='Compare six corrected terminals with original references')
    for option in ('manifest', 'reference-manifest', 'output-dir'):
        correction.add_argument('--' + option, required=True)
    inverse = subcommands.add_parser('report-inverse-membership', help='Compare five fixed IM terminals with original uniform references')
    for option in ('manifest', 'reference-manifest', 'output-dir'):
        inverse.add_argument('--' + option, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'report-inverse-membership':
            report = report_inverse_membership_comparison(manifest=args.manifest, reference_manifest=args.reference_manifest, output_dir=args.output_dir)
        elif args.command == 'report-corrections':
            report = report_correction_comparison(manifest=args.manifest, reference_manifest=args.reference_manifest, output_dir=args.output_dir)
        elif args.command == 'freeze':
            report = freeze_campaign(campaign_manifest=args.campaign_manifest, run_root=args.run_root,
                run_dirs=args.run_dirs, output_dir=args.output_dir, allow_partial=args.allow_partial)
        elif args.command == 'report':
            report = report_campaign(manifest=args.manifest, output_dir=args.output_dir, allow_partial=args.allow_partial)
        else:
            report = preflight_campaign(
                campaign_path=args.campaign,
                prepared_corpus_dir=args.prepared_corpus_dir,
                tokenizer_dir=args.tokenizer_dir,
                output_dir=args.output_dir,
                run_output_root=args.run_output_root,
            )
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Campaign {args.command} failed: {error}\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
