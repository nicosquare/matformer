"""Deterministic TinyStories dataset and tokenizer preparation helpers."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sized

from src.training.fineweb_tokenizer import (
    DEFAULT_MAX_CHUNK_BYTES,
    FineWebTokenizerError,
    load_tokenizer_manifest,
    train_sentencepiece_tokenizer,
)
from src.utils.reproducibility import stable_hash


TINYSTORIES_DATASET_NAME = "roneneldan/TinyStories"
TINYSTORIES_DATASET_CONFIG = "default"
TINYSTORIES_DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TINYSTORIES_DATASET_SPLIT = "train+validation"
TINYSTORIES_ROW_FILTER = "skip_blank_text_preserve_physical_row_index_v1"
TINYSTORIES_TOKENIZER_NAME = "tinystories_sentencepiece_bpe_2k"
TINYSTORIES_TOKENIZER_TRAINING_VERSION = "tinystories_sentencepiece_bpe_v1"
TINYSTORIES_VOCAB_SIZE = 2_048
TINYSTORIES_TOKENIZER_STORY_COUNT = 50_000
TINYSTORIES_CONTROLLER_STORY_COUNT = 128
TINYSTORIES_VALIDATION_STORY_COUNT = 128
TINYSTORIES_FINAL_HOLDOUT_STORY_COUNT = 512
TINYSTORIES_OPTIMIZER_TOKEN_COUNT = 33_554_432
TINYSTORIES_CONTEXT_LENGTH = 128
TINYSTORIES_ROLE_COUNTS = {
    "ordinary_validation": TINYSTORIES_VALIDATION_STORY_COUNT,
    "controller": TINYSTORIES_CONTROLLER_STORY_COUNT,
    "final_holdout": TINYSTORIES_FINAL_HOLDOUT_STORY_COUNT,
}


class TinyStoriesError(ValueError):
    """Raised when TinyStories source preparation violates its contract."""


def dataset_split_fingerprint(
    dataset: Sized,
    *,
    split: str,
    dataset_name: str = TINYSTORIES_DATASET_NAME,
    dataset_revision: str = TINYSTORIES_DATASET_REVISION,
) -> str:
    """Build a stable identity for one materialized Hugging Face split."""

    native_fingerprint = getattr(dataset, "_fingerprint", None)
    if not isinstance(native_fingerprint, str) or not native_fingerprint:
        raise TinyStoriesError(
            f"TinyStories {split} split does not expose a datasets fingerprint"
        )
    return stable_hash(
        {
            "dataset_name": str(dataset_name),
            "dataset_config_name": TINYSTORIES_DATASET_CONFIG,
            "dataset_revision": str(dataset_revision),
            "split": str(split),
            "datasets_fingerprint": native_fingerprint,
            "row_count": len(dataset),
            "text_column": "text",
            "row_filter": TINYSTORIES_ROW_FILTER,
        }
    )


def iter_dataset_stories(
    dataset: Iterable[Mapping[str, Any]],
    *,
    split: str,
) -> Iterator[dict[str, Any]]:
    """Yield complete stories in the immutable order exposed by ``datasets``."""

    story_count = 0
    for source_row_index, row in enumerate(dataset):
        if not isinstance(row, Mapping) or "text" not in row:
            raise TinyStoriesError(
                f"TinyStories {split} row {source_row_index} has no text column"
            )
        text = row["text"]
        if text is None or (isinstance(text, str) and not text.strip()):
            continue
        if not isinstance(text, str):
            raise TinyStoriesError(
                f"TinyStories {split} row {source_row_index} text is not a string"
            )
        identity = f"{split}:{source_row_index}"
        yield {
            "id": identity,
            "source_row_identity": identity,
            "source_split": str(split),
            "source_story_index": source_row_index,
            "nonempty_story_index": story_count,
            "text": text.strip(),
        }
        story_count += 1
    if story_count == 0:
        raise TinyStoriesError(f"TinyStories {split} split contains no stories")


def optimizer_eligible_stories(
    train_dataset: Iterable[Mapping[str, Any]],
) -> Iterator[dict[str, Any]]:
    stories = iter_dataset_stories(train_dataset, split="train")
    yield from itertools.islice(
        stories,
        TINYSTORIES_CONTROLLER_STORY_COUNT,
        None,
    )


def tokenizer_training_stories(
    train_dataset: Iterable[Mapping[str, Any]],
    *,
    story_count: int = TINYSTORIES_TOKENIZER_STORY_COUNT,
) -> Iterator[dict[str, Any]]:
    requested = int(story_count)
    if requested <= 0:
        raise TinyStoriesError("Tokenizer story count must be positive")
    selected = itertools.islice(optimizer_eligible_stories(train_dataset), requested)
    yielded = 0
    for story in selected:
        yielded += 1
        yield story
    if yielded != requested:
        raise TinyStoriesError(
            f"TinyStories train split contains fewer than {requested} "
            "optimizer-eligible tokenizer stories"
        )


def tokenizer_source_fingerprint(
    train_split_fingerprint: str,
    *,
    dataset_name: str = TINYSTORIES_DATASET_NAME,
    dataset_revision: str = TINYSTORIES_DATASET_REVISION,
) -> str:
    if not str(train_split_fingerprint):
        raise TinyStoriesError("TinyStories train split fingerprint is missing")
    return stable_hash(
        {
            "dataset": str(dataset_name),
            "config": TINYSTORIES_DATASET_CONFIG,
            "revision": str(dataset_revision),
            "split": "train",
            "split_fingerprint": str(train_split_fingerprint),
            "controller_story_count": TINYSTORIES_CONTROLLER_STORY_COUNT,
        }
    )


def load_existing_tinystories_tokenizer_if_matching(
    tokenizer_dir: str | Path,
    *,
    train_split_fingerprint: str,
    dataset_name: str = TINYSTORIES_DATASET_NAME,
    dataset_revision: str = TINYSTORIES_DATASET_REVISION,
    story_count: int = TINYSTORIES_TOKENIZER_STORY_COUNT,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
) -> dict[str, Any] | None:
    root = Path(tokenizer_dir).expanduser().resolve()
    if not root.exists():
        return None
    manifest = load_tokenizer_manifest(root, verify_files=False)
    expected = {
        "training_version": TINYSTORIES_TOKENIZER_TRAINING_VERSION,
        "tokenizer_name": TINYSTORIES_TOKENIZER_NAME,
        "dataset": {
            "name": str(dataset_name),
            "config_name": TINYSTORIES_DATASET_CONFIG,
            "split": "train",
            "fingerprint": tokenizer_source_fingerprint(
                train_split_fingerprint,
                dataset_name=dataset_name,
                dataset_revision=dataset_revision,
            ),
            "text_column": "text",
        },
        "reserved_document_count": 0,
        "training_document_count": int(story_count),
        "max_chunk_bytes": int(max_chunk_bytes),
        "vocab_size": TINYSTORIES_VOCAB_SIZE,
    }
    mismatches = {
        key: {"actual": manifest.get(key), "expected": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise TinyStoriesError(
            "Existing TinyStories tokenizer does not match the request: "
            f"{mismatches}"
        )
    return load_tokenizer_manifest(root, verify_files=True)


def train_tinystories_tokenizer(
    train_dataset: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    train_split_fingerprint: str,
    dataset_name: str = TINYSTORIES_DATASET_NAME,
    dataset_revision: str = TINYSTORIES_DATASET_REVISION,
    story_count: int = TINYSTORIES_TOKENIZER_STORY_COUNT,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
) -> dict[str, Any]:
    try:
        return train_sentencepiece_tokenizer(
            tokenizer_training_stories(train_dataset, story_count=story_count),
            output_dir,
            tokenizer_name=TINYSTORIES_TOKENIZER_NAME,
            training_version=TINYSTORIES_TOKENIZER_TRAINING_VERSION,
            source_dataset=str(dataset_name),
            source_config=TINYSTORIES_DATASET_CONFIG,
            source_split="train",
            source_fingerprint=tokenizer_source_fingerprint(
                train_split_fingerprint,
                dataset_name=dataset_name,
                dataset_revision=dataset_revision,
            ),
            text_column="text",
            data_seed=42,
            shuffle_buffer_size=0,
            document_count=int(story_count),
            reserved_document_count=0,
            max_chunk_bytes=int(max_chunk_bytes),
            vocab_size=TINYSTORIES_VOCAB_SIZE,
        )
    except FineWebTokenizerError as error:
        raise TinyStoriesError(str(error)) from error


def role_ordered_stories(
    train_dataset: Iterable[Mapping[str, Any]],
    validation_dataset: Iterable[Mapping[str, Any]],
) -> Iterator[Mapping[str, Any]]:
    """Compose the role order consumed by the shared packed-corpus builder."""

    validation_stories = iter_dataset_stories(
        validation_dataset,
        split="validation",
    )
    ordinary = list(
        itertools.islice(validation_stories, TINYSTORIES_VALIDATION_STORY_COUNT)
    )
    final_holdout = list(
        itertools.islice(validation_stories, TINYSTORIES_FINAL_HOLDOUT_STORY_COUNT)
    )
    if len(ordinary) != TINYSTORIES_VALIDATION_STORY_COUNT or len(
        final_holdout
    ) != TINYSTORIES_FINAL_HOLDOUT_STORY_COUNT:
        raise TinyStoriesError(
            "TinyStories validation split cannot fill ordinary validation and "
            "final holdout roles"
        )

    train_stories = iter_dataset_stories(train_dataset, split="train")
    controller = list(
        itertools.islice(train_stories, TINYSTORIES_CONTROLLER_STORY_COUNT)
    )
    if len(controller) != TINYSTORIES_CONTROLLER_STORY_COUNT:
        raise TinyStoriesError(
            "TinyStories train split cannot fill the controller role"
        )

    yield from ordinary
    yield from controller
    yield from final_holdout
    yield from train_stories


def corpus_source_contract(
    *,
    train_split_fingerprint: str,
    validation_split_fingerprint: str,
    train_row_count: int,
    validation_row_count: int,
    dataset_name: str = TINYSTORIES_DATASET_NAME,
    dataset_revision: str = TINYSTORIES_DATASET_REVISION,
) -> tuple[str, dict[str, dict[str, Any]]]:
    if not str(train_split_fingerprint) or not str(validation_split_fingerprint):
        raise TinyStoriesError("TinyStories split fingerprints must be non-empty")
    shared = {
        "dataset_name": str(dataset_name),
        "dataset_config_name": TINYSTORIES_DATASET_CONFIG,
        "dataset_revision": str(dataset_revision),
        "source_interface": "huggingface_datasets",
        "row_filter": TINYSTORIES_ROW_FILTER,
    }
    role_sources = {
        "optimizer_training": {
            **shared,
            "split": "train",
            "split_fingerprint": str(train_split_fingerprint),
            "split_row_count": int(train_row_count),
            "first_story_index": TINYSTORIES_CONTROLLER_STORY_COUNT,
        },
        "controller": {
            **shared,
            "split": "train",
            "split_fingerprint": str(train_split_fingerprint),
            "split_row_count": int(train_row_count),
            "first_story_index": 0,
            "story_count": TINYSTORIES_CONTROLLER_STORY_COUNT,
        },
        "ordinary_validation": {
            **shared,
            "split": "validation",
            "split_fingerprint": str(validation_split_fingerprint),
            "split_row_count": int(validation_row_count),
            "first_story_index": 0,
            "story_count": TINYSTORIES_VALIDATION_STORY_COUNT,
        },
        "final_holdout": {
            **shared,
            "split": "validation",
            "split_fingerprint": str(validation_split_fingerprint),
            "split_row_count": int(validation_row_count),
            "first_story_index": TINYSTORIES_VALIDATION_STORY_COUNT,
            "story_count": TINYSTORIES_FINAL_HOLDOUT_STORY_COUNT,
        },
    }
    return stable_hash(role_sources), role_sources


__all__ = [
    "TINYSTORIES_CONTEXT_LENGTH",
    "TINYSTORIES_CONTROLLER_STORY_COUNT",
    "TINYSTORIES_DATASET_CONFIG",
    "TINYSTORIES_DATASET_NAME",
    "TINYSTORIES_DATASET_REVISION",
    "TINYSTORIES_DATASET_SPLIT",
    "TINYSTORIES_FINAL_HOLDOUT_STORY_COUNT",
    "TINYSTORIES_OPTIMIZER_TOKEN_COUNT",
    "TINYSTORIES_ROW_FILTER",
    "TINYSTORIES_ROLE_COUNTS",
    "TINYSTORIES_TOKENIZER_NAME",
    "TINYSTORIES_TOKENIZER_STORY_COUNT",
    "TINYSTORIES_VALIDATION_STORY_COUNT",
    "TINYSTORIES_VOCAB_SIZE",
    "TinyStoriesError",
    "corpus_source_contract",
    "dataset_split_fingerprint",
    "iter_dataset_stories",
    "load_existing_tinystories_tokenizer_if_matching",
    "optimizer_eligible_stories",
    "role_ordered_stories",
    "tokenizer_training_stories",
    "train_tinystories_tokenizer",
]
