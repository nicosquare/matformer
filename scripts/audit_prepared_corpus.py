#!/usr/bin/env python3
"""Validate checksums, role separation, and minimum size of a prepared corpus."""

from __future__ import annotations

import argparse
import json

from src.training.packed_corpus import audit_packed_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-corpus-dir", required=True)
    parser.add_argument("--prepared-tokenizer-dir")
    parser.add_argument("--required-vocab-size", type=int)
    parser.add_argument(
        "--minimum-training-tokens",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    print(
        json.dumps(
            audit_packed_corpus(
                args.prepared_corpus_dir,
                minimum_training_tokens=args.minimum_training_tokens,
                prepared_tokenizer_dir=args.prepared_tokenizer_dir,
                required_vocab_size=args.required_vocab_size,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
