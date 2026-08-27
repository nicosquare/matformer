"""Deterministic TinyStories-Instruct record and tokenizer preparation."""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sized

from src.training.fineweb_tokenizer import (
    DEFAULT_MAX_CHUNK_BYTES,
    FineWebTokenizerError,
    load_tokenizer_manifest,
    train_sentencepiece_tokenizer,
)
from src.utils.reproducibility import stable_hash


TINYSTORIES_INSTRUCT_DATASET_NAME = "roneneldan/TinyStoriesInstruct"
TINYSTORIES_INSTRUCT_DATASET_CONFIG = "default"
TINYSTORIES_INSTRUCT_DATASET_REVISION = (
    "ee050ed1f8720795be342921335e821856a2b42e"
)
TINYSTORIES_INSTRUCT_DATASET_SPLIT = "train+validation"
TINYSTORIES_INSTRUCT_DELIMITER = "<|endoftext|>"
TINYSTORIES_INSTRUCT_ASSEMBLY_PARSER_VERSION = (
    "exact_delimited_physical_rows_v1"
)
TINYSTORIES_INSTRUCT_DELIMITER_POLICY = (
    "exact_physical_row_delimiter_removed_before_eos_v1"
)
TINYSTORIES_INSTRUCT_CONTENT_POLICY = (
    "complete_record_lf_join_preserve_fields_and_internal_newlines_v1"
)
# The pinned train blob ends mid-word after these eleven physical rows.  The
# exact range and content hash make dropping that one non-record deterministic;
# every other nonblank unterminated sequence remains an error.
TINYSTORIES_INSTRUCT_KNOWN_TRUNCATED_TAIL_FIRST_ROW = 21_755_670
TINYSTORIES_INSTRUCT_KNOWN_TRUNCATED_TAIL_LAST_ROW = 21_755_680
TINYSTORIES_INSTRUCT_KNOWN_TRUNCATED_TAIL_HASH = (
    "4303446e117b2a05af5ac1c4ee956980224b7a81eef396c44d57b185b704cf02"
)
TINYSTORIES_INSTRUCT_TOKENIZER_NAME = (
    "tinystories-instruct-sentencepiece-bpe-2k-v1"
)
TINYSTORIES_INSTRUCT_TOKENIZER_TRAINING_VERSION = (
    "tinystories_instruct_sentencepiece_bpe_v1"
)
TINYSTORIES_INSTRUCT_CORPUS_NAME = "tinystories-instruct-packed-full-v1"
TINYSTORIES_INSTRUCT_VOCAB_SIZE = 2_048
TINYSTORIES_INSTRUCT_TOKENIZER_RECORD_COUNT = 50_000
TINYSTORIES_INSTRUCT_CONTROLLER_RECORD_COUNT = 128
TINYSTORIES_INSTRUCT_VALIDATION_RECORD_COUNT = 128
TINYSTORIES_INSTRUCT_FINAL_HOLDOUT_RECORD_COUNT = 512
TINYSTORIES_INSTRUCT_CONTEXT_LENGTH = 128
TINYSTORIES_INSTRUCT_ROLE_COUNTS = {
    "ordinary_validation": TINYSTORIES_INSTRUCT_VALIDATION_RECORD_COUNT,
    "controller": TINYSTORIES_INSTRUCT_CONTROLLER_RECORD_COUNT,
    "final_holdout": TINYSTORIES_INSTRUCT_FINAL_HOLDOUT_RECORD_COUNT,
}


class TinyStoriesInstructError(ValueError):
    """Raised when TinyStories-Instruct preparation violates its contract."""


class TinyStoriesInstructWarning(UserWarning):
    """Warns about a recognized, deterministically excluded source anomaly."""


def _is_known_pinned_truncated_train_tail(
    *,
    split: str,
    first_physical_row_index: int | None,
    last_physical_row_index: int,
    buffered_rows: list[str],
) -> bool:
    return (
        split == "train"
        and first_physical_row_index
        == TINYSTORIES_INSTRUCT_KNOWN_TRUNCATED_TAIL_FIRST_ROW
        and last_physical_row_index
        == TINYSTORIES_INSTRUCT_KNOWN_TRUNCATED_TAIL_LAST_ROW
        and stable_hash(buffered_rows)
        == TINYSTORIES_INSTRUCT_KNOWN_TRUNCATED_TAIL_HASH
    )


def dataset_split_fingerprint(
    dataset: Sized,
    *,
    split: str,
    dataset_name: str = TINYSTORIES_INSTRUCT_DATASET_NAME,
    dataset_revision: str = TINYSTORIES_INSTRUCT_DATASET_REVISION,
    assembly_parser_version: str = TINYSTORIES_INSTRUCT_ASSEMBLY_PARSER_VERSION,
    delimiter_policy: str = TINYSTORIES_INSTRUCT_DELIMITER_POLICY,
    content_policy: str = TINYSTORIES_INSTRUCT_CONTENT_POLICY,
) -> str:
    """Identify a raw split together with the complete-record parser contract."""

    native_fingerprint = getattr(dataset, "_fingerprint", None)
    if not isinstance(native_fingerprint, str) or not native_fingerprint:
        raise TinyStoriesInstructError(
            f"TinyStories-Instruct {split} split does not expose a datasets "
            "fingerprint"
        )
    return stable_hash(
        {
            "dataset_name": str(dataset_name),
            "dataset_config_name": TINYSTORIES_INSTRUCT_DATASET_CONFIG,
            "dataset_revision": str(dataset_revision),
            "split": str(split),
            "datasets_fingerprint": native_fingerprint,
            "physical_row_count": len(dataset),
            "text_column": "text",
            "assembly_parser_version": str(assembly_parser_version),
            "delimiter": TINYSTORIES_INSTRUCT_DELIMITER,
            "delimiter_policy": str(delimiter_policy),
            "content_policy": str(content_policy),
        }
    )


def iter_dataset_instruction_records(
    dataset: Iterable[Mapping[str, Any]],
    *,
    split: str,
    delimiter: str = TINYSTORIES_INSTRUCT_DELIMITER,
) -> Iterator[dict[str, Any]]:
    """Assemble physical text rows into exact-delimited instruction records."""

    if not isinstance(delimiter, str) or not delimiter:
        raise TinyStoriesInstructError(
            "TinyStories-Instruct delimiter must be a non-empty string"
        )
    buffered_rows: list[str] = []
    first_physical_row_index: int | None = None
    record_count = 0
    last_physical_row_index = -1

    for physical_row_index, row in enumerate(dataset):
        last_physical_row_index = physical_row_index
        if not isinstance(row, Mapping) or "text" not in row:
            raise TinyStoriesInstructError(
                f"TinyStories-Instruct {split} row {physical_row_index} has no "
                "text column"
            )
        text = row["text"]
        if not isinstance(text, str):
            raise TinyStoriesInstructError(
                f"TinyStories-Instruct {split} row {physical_row_index} text "
                "is not a string"
            )
        if first_physical_row_index is None:
            first_physical_row_index = physical_row_index
        if text != delimiter:
            buffered_rows.append(text)
            continue

        assembled_text = "\n".join(buffered_rows)
        if not assembled_text.strip():
            raise TinyStoriesInstructError(
                f"TinyStories-Instruct {split} rows "
                f"{first_physical_row_index}-{physical_row_index} form an "
                "empty instruction record"
            )
        identity = (
            f"{split}:{first_physical_row_index}-{physical_row_index}"
        )
        yield {
            "id": identity,
            "source_row_identity": identity,
            "source_split": str(split),
            "source_first_physical_row_index": first_physical_row_index,
            "source_last_physical_row_index": physical_row_index,
            "source_physical_row_range_inclusive": [
                first_physical_row_index,
                physical_row_index,
            ],
            "assembled_record_index": record_count,
            "text": assembled_text,
        }
        record_count += 1
        buffered_rows = []
        first_physical_row_index = None

    if buffered_rows and any(row.strip() for row in buffered_rows):
        if _is_known_pinned_truncated_train_tail(
            split=split,
            first_physical_row_index=first_physical_row_index,
            last_physical_row_index=last_physical_row_index,
            buffered_rows=buffered_rows,
        ):
            warnings.warn(
                "TinyStories-Instruct is dropping known truncated terminal "
                f"train rows {first_physical_row_index}-"
                f"{last_physical_row_index} from pinned revision "
                f"{TINYSTORIES_INSTRUCT_DATASET_REVISION}",
                TinyStoriesInstructWarning,
                stacklevel=2,
            )
        else:
            raise TinyStoriesInstructError(
                f"TinyStories-Instruct {split} rows "
                f"{first_physical_row_index}-{last_physical_row_index} contain a "
                "nonblank unterminated instruction record"
            )
    if record_count == 0:
        raise TinyStoriesInstructError(
            f"TinyStories-Instruct {split} split contains no complete "
            "instruction records"
        )


def optimizer_eligible_instruction_records(
    train_dataset: Iterable[Mapping[str, Any]],
) -> Iterator[dict[str, Any]]:
    records = iter_dataset_instruction_records(train_dataset, split="train")
    yield from itertools.islice(
        records,
        TINYSTORIES_INSTRUCT_CONTROLLER_RECORD_COUNT,
        None,
    )


def tokenizer_training_instruction_records(
    train_dataset: Iterable[Mapping[str, Any]],
    *,
    record_count: int = TINYSTORIES_INSTRUCT_TOKENIZER_RECORD_COUNT,
) -> Iterator[dict[str, Any]]:
    requested = int(record_count)
    if requested <= 0:
        raise TinyStoriesInstructError("Tokenizer record count must be positive")
    selected = itertools.islice(
        optimizer_eligible_instruction_records(train_dataset), requested
    )
    yielded = 0
    for record in selected:
        yielded += 1
        yield record
    if yielded != requested:
        raise TinyStoriesInstructError(
            f"TinyStories-Instruct train split contains fewer than {requested} "
            "optimizer-eligible tokenizer records"
        )


def tokenizer_source_fingerprint(
    train_split_fingerprint: str,
    *,
    dataset_name: str = TINYSTORIES_INSTRUCT_DATASET_NAME,
    dataset_revision: str = TINYSTORIES_INSTRUCT_DATASET_REVISION,
    assembly_parser_version: str = TINYSTORIES_INSTRUCT_ASSEMBLY_PARSER_VERSION,
    delimiter_policy: str = TINYSTORIES_INSTRUCT_DELIMITER_POLICY,
    content_policy: str = TINYSTORIES_INSTRUCT_CONTENT_POLICY,
) -> str:
    if not str(train_split_fingerprint):
        raise TinyStoriesInstructError(
            "TinyStories-Instruct train split fingerprint is missing"
        )
    return stable_hash(
        {
            "dataset": str(dataset_name),
            "config": TINYSTORIES_INSTRUCT_DATASET_CONFIG,
            "revision": str(dataset_revision),
            "split": "train",
            "split_fingerprint": str(train_split_fingerprint),
            "controller_record_count": (
                TINYSTORIES_INSTRUCT_CONTROLLER_RECORD_COUNT
            ),
            "assembly_parser_version": str(assembly_parser_version),
            "delimiter": TINYSTORIES_INSTRUCT_DELIMITER,
            "delimiter_policy": str(delimiter_policy),
            "content_policy": str(content_policy),
        }
    )


def load_existing_tinystories_instruct_tokenizer_if_matching(
    tokenizer_dir: str | Path,
    *,
    train_split_fingerprint: str,
    dataset_name: str = TINYSTORIES_INSTRUCT_DATASET_NAME,
    dataset_revision: str = TINYSTORIES_INSTRUCT_DATASET_REVISION,
    record_count: int = TINYSTORIES_INSTRUCT_TOKENIZER_RECORD_COUNT,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
    assembly_parser_version: str = TINYSTORIES_INSTRUCT_ASSEMBLY_PARSER_VERSION,
    delimiter_policy: str = TINYSTORIES_INSTRUCT_DELIMITER_POLICY,
    content_policy: str = TINYSTORIES_INSTRUCT_CONTENT_POLICY,
) -> dict[str, Any] | None:
    root = Path(tokenizer_dir).expanduser().resolve()
    if not root.exists():
        return None
    manifest = load_tokenizer_manifest(root, verify_files=False)
    expected = {
        "training_version": TINYSTORIES_INSTRUCT_TOKENIZER_TRAINING_VERSION,
        "tokenizer_name": TINYSTORIES_INSTRUCT_TOKENIZER_NAME,
        "dataset": {
            "name": str(dataset_name),
            "config_name": TINYSTORIES_INSTRUCT_DATASET_CONFIG,
            "split": "train",
            "fingerprint": tokenizer_source_fingerprint(
                train_split_fingerprint,
                dataset_name=dataset_name,
                dataset_revision=dataset_revision,
                assembly_parser_version=assembly_parser_version,
                delimiter_policy=delimiter_policy,
                content_policy=content_policy,
            ),
            "text_column": "text",
        },
        "reserved_document_count": 0,
        "training_document_count": int(record_count),
        "max_chunk_bytes": int(max_chunk_bytes),
        "vocab_size": TINYSTORIES_INSTRUCT_VOCAB_SIZE,
    }
    mismatches = {
        key: {"actual": manifest.get(key), "expected": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise TinyStoriesInstructError(
            "Existing TinyStories-Instruct tokenizer does not match the request: "
            f"{mismatches}"
        )
    return load_tokenizer_manifest(root, verify_files=True)


def train_tinystories_instruct_tokenizer(
    train_dataset: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    train_split_fingerprint: str,
    dataset_name: str = TINYSTORIES_INSTRUCT_DATASET_NAME,
    dataset_revision: str = TINYSTORIES_INSTRUCT_DATASET_REVISION,
    record_count: int = TINYSTORIES_INSTRUCT_TOKENIZER_RECORD_COUNT,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
    assembly_parser_version: str = TINYSTORIES_INSTRUCT_ASSEMBLY_PARSER_VERSION,
    delimiter_policy: str = TINYSTORIES_INSTRUCT_DELIMITER_POLICY,
    content_policy: str = TINYSTORIES_INSTRUCT_CONTENT_POLICY,
) -> dict[str, Any]:
    try:
        return train_sentencepiece_tokenizer(
            tokenizer_training_instruction_records(
                train_dataset, record_count=record_count
            ),
            output_dir,
            tokenizer_name=TINYSTORIES_INSTRUCT_TOKENIZER_NAME,
            training_version=TINYSTORIES_INSTRUCT_TOKENIZER_TRAINING_VERSION,
            source_dataset=str(dataset_name),
            source_config=TINYSTORIES_INSTRUCT_DATASET_CONFIG,
            source_split="train",
            source_fingerprint=tokenizer_source_fingerprint(
                train_split_fingerprint,
                dataset_name=dataset_name,
                dataset_revision=dataset_revision,
                assembly_parser_version=assembly_parser_version,
                delimiter_policy=delimiter_policy,
                content_policy=content_policy,
            ),
            text_column="text",
            data_seed=42,
            shuffle_buffer_size=0,
            document_count=int(record_count),
            reserved_document_count=0,
            max_chunk_bytes=int(max_chunk_bytes),
            vocab_size=TINYSTORIES_INSTRUCT_VOCAB_SIZE,
        )
    except FineWebTokenizerError as error:
        raise TinyStoriesInstructError(str(error)) from error


def role_ordered_instruction_records(
    train_dataset: Iterable[Mapping[str, Any]],
    validation_dataset: Iterable[Mapping[str, Any]],
) -> Iterator[Mapping[str, Any]]:
    """Compose assembled records in the order expected by the corpus builder."""

    validation_records = iter_dataset_instruction_records(
        validation_dataset, split="validation"
    )
    ordinary = list(
        itertools.islice(
            validation_records, TINYSTORIES_INSTRUCT_VALIDATION_RECORD_COUNT
        )
    )
    final_holdout = list(
        itertools.islice(
            validation_records, TINYSTORIES_INSTRUCT_FINAL_HOLDOUT_RECORD_COUNT
        )
    )
    if len(ordinary) != TINYSTORIES_INSTRUCT_VALIDATION_RECORD_COUNT or len(
        final_holdout
    ) != TINYSTORIES_INSTRUCT_FINAL_HOLDOUT_RECORD_COUNT:
        raise TinyStoriesInstructError(
            "TinyStories-Instruct validation split cannot fill ordinary "
            "validation and final holdout roles"
        )

    train_records = iter_dataset_instruction_records(train_dataset, split="train")
    controller = list(
        itertools.islice(
            train_records, TINYSTORIES_INSTRUCT_CONTROLLER_RECORD_COUNT
        )
    )
    if len(controller) != TINYSTORIES_INSTRUCT_CONTROLLER_RECORD_COUNT:
        raise TinyStoriesInstructError(
            "TinyStories-Instruct train split cannot fill the controller role"
        )

    yield from ordinary
    yield from controller
    yield from final_holdout
    yield from train_records


def corpus_source_contract(
    *,
    train_split_fingerprint: str,
    validation_split_fingerprint: str,
    train_row_count: int,
    validation_row_count: int,
    dataset_name: str = TINYSTORIES_INSTRUCT_DATASET_NAME,
    dataset_revision: str = TINYSTORIES_INSTRUCT_DATASET_REVISION,
    assembly_parser_version: str = TINYSTORIES_INSTRUCT_ASSEMBLY_PARSER_VERSION,
    delimiter_policy: str = TINYSTORIES_INSTRUCT_DELIMITER_POLICY,
    content_policy: str = TINYSTORIES_INSTRUCT_CONTENT_POLICY,
) -> tuple[str, dict[str, dict[str, Any]]]:
    if not str(train_split_fingerprint) or not str(validation_split_fingerprint):
        raise TinyStoriesInstructError(
            "TinyStories-Instruct split fingerprints must be non-empty"
        )
    shared = {
        "dataset_name": str(dataset_name),
        "dataset_config_name": TINYSTORIES_INSTRUCT_DATASET_CONFIG,
        "dataset_revision": str(dataset_revision),
        "source_interface": "huggingface_datasets_physical_text_rows",
        "assembly_parser_version": str(assembly_parser_version),
        "delimiter": TINYSTORIES_INSTRUCT_DELIMITER,
        "delimiter_policy": str(delimiter_policy),
        "content_policy": str(content_policy),
    }
    role_sources = {
        "optimizer_training": {
            **shared,
            "split": "train",
            "split_fingerprint": str(train_split_fingerprint),
            "physical_split_row_count": int(train_row_count),
            "first_record_index": TINYSTORIES_INSTRUCT_CONTROLLER_RECORD_COUNT,
        },
        "controller": {
            **shared,
            "split": "train",
            "split_fingerprint": str(train_split_fingerprint),
            "physical_split_row_count": int(train_row_count),
            "first_record_index": 0,
            "record_count": TINYSTORIES_INSTRUCT_CONTROLLER_RECORD_COUNT,
        },
        "ordinary_validation": {
            **shared,
            "split": "validation",
            "split_fingerprint": str(validation_split_fingerprint),
            "physical_split_row_count": int(validation_row_count),
            "first_record_index": 0,
            "record_count": TINYSTORIES_INSTRUCT_VALIDATION_RECORD_COUNT,
        },
        "final_holdout": {
            **shared,
            "split": "validation",
            "split_fingerprint": str(validation_split_fingerprint),
            "physical_split_row_count": int(validation_row_count),
            "first_record_index": TINYSTORIES_INSTRUCT_VALIDATION_RECORD_COUNT,
            "record_count": TINYSTORIES_INSTRUCT_FINAL_HOLDOUT_RECORD_COUNT,
        },
    }
    return stable_hash(role_sources), role_sources


__all__ = [
    "TINYSTORIES_INSTRUCT_ASSEMBLY_PARSER_VERSION",
    "TINYSTORIES_INSTRUCT_CONTENT_POLICY",
    "TINYSTORIES_INSTRUCT_CONTEXT_LENGTH",
    "TINYSTORIES_INSTRUCT_CONTROLLER_RECORD_COUNT",
    "TINYSTORIES_INSTRUCT_CORPUS_NAME",
    "TINYSTORIES_INSTRUCT_DATASET_CONFIG",
    "TINYSTORIES_INSTRUCT_DATASET_NAME",
    "TINYSTORIES_INSTRUCT_DATASET_REVISION",
    "TINYSTORIES_INSTRUCT_DATASET_SPLIT",
    "TINYSTORIES_INSTRUCT_DELIMITER",
    "TINYSTORIES_INSTRUCT_DELIMITER_POLICY",
    "TINYSTORIES_INSTRUCT_FINAL_HOLDOUT_RECORD_COUNT",
    "TINYSTORIES_INSTRUCT_KNOWN_TRUNCATED_TAIL_FIRST_ROW",
    "TINYSTORIES_INSTRUCT_KNOWN_TRUNCATED_TAIL_HASH",
    "TINYSTORIES_INSTRUCT_KNOWN_TRUNCATED_TAIL_LAST_ROW",
    "TINYSTORIES_INSTRUCT_ROLE_COUNTS",
    "TINYSTORIES_INSTRUCT_TOKENIZER_NAME",
    "TINYSTORIES_INSTRUCT_TOKENIZER_RECORD_COUNT",
    "TINYSTORIES_INSTRUCT_VALIDATION_RECORD_COUNT",
    "TINYSTORIES_INSTRUCT_VOCAB_SIZE",
    "TinyStoriesInstructError",
    "TinyStoriesInstructWarning",
    "corpus_source_contract",
    "dataset_split_fingerprint",
    "iter_dataset_instruction_records",
    "load_existing_tinystories_instruct_tokenizer_if_matching",
    "optimizer_eligible_instruction_records",
    "role_ordered_instruction_records",
    "tokenizer_source_fingerprint",
    "tokenizer_training_instruction_records",
    "train_tinystories_instruct_tokenizer",
]
