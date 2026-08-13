#!/usr/bin/env python3
"""Validate checksums, role separation, and exact size of a prepared corpus."""

from __future__ import annotations

import argparse
import json

from src.training.packed_corpus import (
    DEFAULT_TRAINING_TOKEN_BUDGET,
    audit_packed_corpus,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-corpus-dir", required=True)
    parser.add_argument("--prepared-tokenizer-dir")
    parser.add_argument("--required-vocab-size", type=int)
    parser.add_argument(
        "--required-training-tokens",
        type=int,
        default=DEFAULT_TRAINING_TOKEN_BUDGET,
    )
    args = parser.parse_args()
    print(
        json.dumps(
            audit_packed_corpus(
                args.prepared_corpus_dir,
                required_training_tokens=args.required_training_tokens,
                prepared_tokenizer_dir=args.prepared_tokenizer_dir,
                required_vocab_size=args.required_vocab_size,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
