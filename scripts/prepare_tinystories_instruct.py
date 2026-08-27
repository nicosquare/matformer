#!/usr/bin/env python3
"""Prepare TinyStories-Instruct as a complete-record packed corpus."""

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
from src.training.tinystories_instruct import (
    TINYSTORIES_INSTRUCT_CONTEXT_LENGTH,
    TINYSTORIES_INSTRUCT_CORPUS_NAME,
    TINYSTORIES_INSTRUCT_DATASET_CONFIG,
    TINYSTORIES_INSTRUCT_DATASET_NAME,
    TINYSTORIES_INSTRUCT_DATASET_REVISION,
    TINYSTORIES_INSTRUCT_DATASET_SPLIT,
    TINYSTORIES_INSTRUCT_ROLE_COUNTS,
    TINYSTORIES_INSTRUCT_TOKENIZER_RECORD_COUNT,
    corpus_source_contract,
    dataset_split_fingerprint,
    load_existing_tinystories_instruct_tokenizer_if_matching,
    role_ordered_instruction_records,
    train_tinystories_instruct_tokenizer,
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
        dataset_name=TINYSTORIES_INSTRUCT_DATASET_NAME,
        dataset_revision=TINYSTORIES_INSTRUCT_DATASET_REVISION,
        tokenizer_document_count=TINYSTORIES_INSTRUCT_TOKENIZER_RECORD_COUNT,
    )


def _preparation_spec() -> TinyStoriesPreparationSpec:
    return TinyStoriesPreparationSpec(
        display_name="TinyStories-Instruct",
        progress_prefix="tinystories-instruct-progress",
        dataset_name=TINYSTORIES_INSTRUCT_DATASET_NAME,
        dataset_revision=TINYSTORIES_INSTRUCT_DATASET_REVISION,
        dataset_config=TINYSTORIES_INSTRUCT_DATASET_CONFIG,
        dataset_split=TINYSTORIES_INSTRUCT_DATASET_SPLIT,
        context_length=TINYSTORIES_INSTRUCT_CONTEXT_LENGTH,
        tokenizer_document_count=TINYSTORIES_INSTRUCT_TOKENIZER_RECORD_COUNT,
        tokenizer_document_noun="records",
        role_counts=TINYSTORIES_INSTRUCT_ROLE_COUNTS,
        corpus_name=TINYSTORIES_INSTRUCT_CORPUS_NAME,
        train_count_label="train_physical_rows",
        validation_count_label="validation_physical_rows",
        split_fingerprint=dataset_split_fingerprint,
        load_existing_tokenizer=lambda path, **kwargs: (
            load_existing_tinystories_instruct_tokenizer_if_matching(
                path,
                record_count=kwargs.pop("document_count"),
                **kwargs,
            )
        ),
        train_tokenizer=lambda dataset, path, **kwargs: (
            train_tinystories_instruct_tokenizer(
                dataset,
                path,
                record_count=kwargs.pop("document_count"),
                **kwargs,
            )
        ),
        corpus_source_contract=corpus_source_contract,
        role_ordered_documents=role_ordered_instruction_records,
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
