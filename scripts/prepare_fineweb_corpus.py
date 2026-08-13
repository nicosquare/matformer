#!/usr/bin/env python3
"""Prepare the immutable FineWeb sample-10BT corpus used by all 10B runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from src.training.packed_corpus import (
    DEFAULT_TRAINING_TOKEN_BUDGET,
    prepare_packed_corpus,
)
from src.training.fineweb_tokenizer import load_tokenizer_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prepared-tokenizer-dir",
        required=True,
        help="Verified immutable tokenizer directory containing tokenizer_manifest.json",
    )
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument(
        "--training-token-budget",
        type=int,
        default=DEFAULT_TRAINING_TOKEN_BUDGET,
    )
    parser.add_argument(
        "--shard-token-capacity",
        type=int,
        default=256 * 1024 * 1024,
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=100_000,
        help="FineWeb streaming shuffle buffer; the resulting order is seed-pinned",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_manifest = load_tokenizer_manifest(
        args.prepared_tokenizer_dir, verify_files=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.prepared_tokenizer_dir,
        local_files_only=True,
    )
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        streaming=True,
    )
    fingerprint = getattr(dataset, "_fingerprint", None)
    if not fingerprint:
        fingerprint = (
            f"{args.dataset}:{args.dataset_config}:{args.split}:"
            f"streaming-shuffle-{args.data_seed}-{args.shuffle_buffer_size}"
        )
    dataset = dataset.shuffle(
        seed=args.data_seed,
        buffer_size=args.shuffle_buffer_size,
    )
    manifest = prepare_packed_corpus(
        dataset,
        tokenizer,
        Path(args.output_dir),
        tokenizer_name=tokenizer_manifest["tokenizer_name"],
        tokenizer_revision=tokenizer_manifest["manifest_hash"],
        source_dataset=args.dataset,
        source_config=args.dataset_config,
        source_split=args.split,
        source_fingerprint=str(fingerprint),
        text_column=args.text_column,
        data_seed=args.data_seed,
        context_length=args.context_length,
        training_token_budget=args.training_token_budget,
        shard_token_capacity=args.shard_token_capacity,
        tokenizer_manifest=tokenizer_manifest,
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).expanduser().resolve()),
                "corpus_hash": manifest["corpus_hash"],
                "training_token_count": manifest["roles"]["optimizer_training"][
                    "token_count"
                ],
                "role_manifest_hashes": manifest["role_manifest_hashes"],
                "tokenizer_manifest_hash": tokenizer_manifest["manifest_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
