#!/usr/bin/env python3
"""Train the immutable 256k FineWeb SentencePiece tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from src.training.fineweb_tokenizer import train_fineweb_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer-size", type=int, default=100_000)
    parser.add_argument("--document-count", type=int, default=5_000_000)
    parser.add_argument("--max-chunk-bytes", type=int, default=4_096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        streaming=True,
    )
    fingerprint = getattr(dataset, "_fingerprint", None) or (
        f"{args.dataset}:{args.dataset_config}:{args.split}:streaming"
    )
    dataset = dataset.shuffle(
        seed=args.data_seed,
        buffer_size=args.shuffle_buffer_size,
    )
    manifest = train_fineweb_tokenizer(
        dataset,
        Path(args.output_dir),
        source_dataset=args.dataset,
        source_config=args.dataset_config,
        source_split=args.split,
        source_fingerprint=str(fingerprint),
        text_column=args.text_column,
        data_seed=args.data_seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        document_count=args.document_count,
        max_chunk_bytes=args.max_chunk_bytes,
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).expanduser().resolve()),
                "tokenizer_revision": manifest["manifest_hash"],
                "vocab_size": manifest["vocab_size"],
                "training_document_count": manifest["training_document_count"],
                "training_chunk_count": manifest["training_chunk_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
