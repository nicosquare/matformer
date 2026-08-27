#!/usr/bin/env python3
"""Prepare the controlled TinyStories tokenizer and packed-mmap corpus."""

# ruff: noqa: E402  # Add the repository root before importing src.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from datasets import load_dataset
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.packed_corpus import (
    load_existing_corpus_if_matching,
    preparation_work_dir,
    prepare_packed_corpus,
)
from src.training.tinystories import (
    TINYSTORIES_CONTEXT_LENGTH,
    TINYSTORIES_DATASET_CONFIG,
    TINYSTORIES_DATASET_NAME,
    TINYSTORIES_DATASET_REVISION,
    TINYSTORIES_DATASET_SPLIT,
    TINYSTORIES_ROLE_COUNTS,
    TINYSTORIES_TOKENIZER_STORY_COUNT,
    corpus_source_contract,
    dataset_split_fingerprint,
    load_existing_tinystories_tokenizer_if_matching,
    role_ordered_stories,
    train_tinystories_tokenizer,
)
from src.training.tinystories_preparation import (
    TinyStoriesPreparationSpec,
    parse_preparation_args,
    run_tinystories_preparation,
)


def parse_args(argv: Sequence[str] | None = None):
    return parse_preparation_args(
        argv,
        description=__doc__ or "",
        dataset_name=TINYSTORIES_DATASET_NAME,
        dataset_revision=TINYSTORIES_DATASET_REVISION,
        tokenizer_document_count=TINYSTORIES_TOKENIZER_STORY_COUNT,
    )


def _preparation_spec() -> TinyStoriesPreparationSpec:
    return TinyStoriesPreparationSpec(
        display_name="TinyStories",
        progress_prefix="tinystories-progress",
        dataset_name=TINYSTORIES_DATASET_NAME,
        dataset_revision=TINYSTORIES_DATASET_REVISION,
        dataset_config=TINYSTORIES_DATASET_CONFIG,
        dataset_split=TINYSTORIES_DATASET_SPLIT,
        context_length=TINYSTORIES_CONTEXT_LENGTH,
        tokenizer_document_count=TINYSTORIES_TOKENIZER_STORY_COUNT,
        tokenizer_document_noun="stories",
        role_counts=TINYSTORIES_ROLE_COUNTS,
        corpus_name=None,
        train_count_label="train_stories",
        validation_count_label="validation_stories",
        split_fingerprint=dataset_split_fingerprint,
        load_existing_tokenizer=lambda path, **kwargs: (
            load_existing_tinystories_tokenizer_if_matching(
                path,
                story_count=kwargs.pop("document_count"),
                **kwargs,
            )
        ),
        train_tokenizer=lambda dataset, path, **kwargs: (
            train_tinystories_tokenizer(
                dataset,
                path,
                story_count=kwargs.pop("document_count"),
                **kwargs,
            )
        ),
        corpus_source_contract=corpus_source_contract,
        role_ordered_documents=role_ordered_stories,
    )


def main(argv: Sequence[str] | None = None) -> None:
    run_tinystories_preparation(
        parse_args(argv),
        spec=_preparation_spec(),
        load_dataset_fn=load_dataset,
        auto_tokenizer_class=AutoTokenizer,
        load_existing_corpus_fn=load_existing_corpus_if_matching,
        prepare_packed_corpus_fn=prepare_packed_corpus,
        preparation_work_dir_fn=preparation_work_dir,
    )


if __name__ == "__main__":
    main()
