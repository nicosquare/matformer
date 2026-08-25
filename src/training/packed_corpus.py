"""Immutable packed-token corpus preparation and memory-mapped access.

The production path assigns deterministic source-document roles before packing
each record. Reserved identities remain explicit, while optimizer-training
provenance uses a resumable rolling hash chain rather than one JSON identity per
packed sequence.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
from torch.utils.data import Dataset, Sampler

from src.utils.reproducibility import stable_hash


PACKED_CORPUS_SCHEMA_VERSION = 3
PACKING_VERSION = "contiguous_eos_uint32_v1"
PERMUTATION_VERSION = "numpy_pcg64_uint64_le_v1"
DOCUMENT_HASH_CHAIN_VERSION = "sha256_chain_v1"
PREPARATION_PROGRESS_SCHEMA_VERSION = 1
DEFAULT_DATA_SEED = 42
DEFAULT_CONTEXT_LENGTH = 1024
DEFAULT_SHUFFLE_BUFFER_SIZE = 100_000
DEFAULT_SHARD_TOKEN_CAPACITY = 256 * 1024 * 1024
RESERVED_ROLE_COUNTS = {
    "ordinary_validation": 512,
    "controller": 128,
    "final_holdout": 512,
}
ALL_CORPUS_ROLES = (
    "optimizer_training",
    "ordinary_validation",
    "controller",
    "final_holdout",
)


class PackedCorpusError(ValueError):
    """Raised when a prepared corpus violates the immutable data contract."""


def preparation_work_dir(output_dir: str | Path) -> Path:
    output = Path(output_dir).expanduser().resolve()
    return output.parent / f".{output.name}.preparing"


def preparation_lock_path(output_dir: str | Path) -> Path:
    output = Path(output_dir).expanduser().resolve()
    return output.parent / f".{output.name}.prepare.lock"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as target:
        json.dump(payload, target, indent=2, sort_keys=True)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _progress_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body["progress_hash"] = stable_hash(body)
    return body


def _load_progress(path: Path) -> dict[str, Any]:
    try:
        progress = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PackedCorpusError(
            f"Corrupt resumable preparation checkpoint: {path}"
        ) from error
    saved_hash = progress.pop("progress_hash", None)
    if saved_hash != stable_hash(progress):
        raise PackedCorpusError(
            f"Corrupt resumable preparation checkpoint hash: {path}"
        )
    return progress


def _initial_document_hash() -> str:
    return hashlib.sha256(DOCUMENT_HASH_CHAIN_VERSION.encode("ascii")).hexdigest()


def _extend_document_hash(previous: str, identity: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    try:
        digest.update(bytes.fromhex(previous))
    except ValueError as error:
        raise PackedCorpusError("Invalid rolling document-identity hash state") from error
    digest.update(stable_hash(identity).encode("ascii"))
    return digest.hexdigest()


def _validate_partial_artifacts(work_root: Path, progress: Mapping[str, Any]) -> None:
    """Validate every artifact referenced by resumable state without changing it."""

    accounted_documents = 0
    for role, role_manifest in progress.get("role_manifests", {}).items():
        shards = role_manifest.get("shards", [])
        if role_manifest.get("shard_set_hash") != stable_hash(shards):
            raise PackedCorpusError(f"Corrupt completed role metadata: {role}")
        token_count = 0
        for shard in shards:
            path = work_root / shard["path"]
            if not path.is_file() or path.stat().st_size != int(shard["byte_count"]):
                raise PackedCorpusError(f"Missing partial preparation shard: {path}")
            if sha256_file(path) != shard["sha256"]:
                raise PackedCorpusError(
                    f"Partial preparation shard checksum mismatch: {path}"
                )
            token_count += int(shard["token_count"])
        if token_count != int(role_manifest.get("token_count", -1)):
            raise PackedCorpusError(f"Corrupt completed role token count: {role}")
        accounted_documents += int(role_manifest.get("source_document_count", 0))

    for role, state in progress.get("role_states", {}).items():
        completed = state.get("completed_shards", [])
        completed_tokens = 0
        for shard in completed:
            path = work_root / shard["path"]
            if not path.is_file() or path.stat().st_size != int(shard["byte_count"]):
                raise PackedCorpusError(f"Missing partial preparation shard: {path}")
            if sha256_file(path) != shard["sha256"]:
                raise PackedCorpusError(
                    f"Partial preparation shard checksum mismatch: {path}"
                )
            completed_tokens += int(shard["token_count"])
        shard_offset = int(state.get("shard_offset", -1))
        if completed_tokens + shard_offset != int(state.get("token_count", -1)):
            raise PackedCorpusError(f"Corrupt partial writer token count: {role}")
        if shard_offset:
            configuration = progress["configuration"]
            partial_path = (
                work_root / role / f"shard-{int(state['shard_index']):05d}.bin"
            )
            expected_bytes = int(
                configuration["shard_token_capacity_resolved"]
            ) * np.dtype(np.uint32).itemsize
            if (
                not partial_path.is_file()
                or partial_path.stat().st_size != expected_bytes
            ):
                raise PackedCorpusError(
                    f"Corrupt partial preparation shard: {partial_path}"
                )
        pending = state.get("pending_tokens", [])
        if len(pending) >= int(progress["configuration"]["context_length"]):
            raise PackedCorpusError(f"Corrupt pending-token state: {role}")
        accounted_documents += int(state.get("source_document_count", 0))

    if accounted_documents != int(
        progress.get("committed_shuffled_document_count", -1)
    ):
        raise PackedCorpusError("Partial preparation document-count mismatch")

    ordering = progress.get("training_order")
    if ordering is not None:
        path = work_root / ordering["path"]
        if not path.is_file() or path.stat().st_size != int(ordering["byte_count"]):
            raise PackedCorpusError("Partial preparation order artifact size mismatch")
        if sha256_file(path) != ordering["sha256"]:
            raise PackedCorpusError(
                "Partial preparation order artifact checksum mismatch"
            )


def _resolved_shard_token_capacity(
    shard_token_capacity: int, context_length: int
) -> int:
    return max(
        int(context_length),
        (int(shard_token_capacity) // int(context_length)) * int(context_length),
    )


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_identity(
    tokenizer: Any,
    *,
    name: str,
    revision: str,
    tokenizer_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise PackedCorpusError("The pinned tokenizer must define eos_token_id")
    identity = {
        "name": str(name),
        "revision": str(revision),
        "eos_token_id": int(eos_token_id),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "tokenization": "no_padding_no_truncation",
    }
    manifest_payload = dict(tokenizer_manifest)
    manifest_hash = manifest_payload.pop("manifest_hash", None)
    if not isinstance(manifest_hash, str) or manifest_hash != stable_hash(
        manifest_payload
    ):
        raise PackedCorpusError("Tokenizer manifest hash mismatch")
    manifest_vocab_size = int(tokenizer_manifest.get("vocab_size", 0) or 0)
    if manifest_vocab_size != identity["vocab_size"]:
        raise PackedCorpusError(
            "Tokenizer vocabulary does not match its prepared manifest"
        )
    if str(name) != str(tokenizer_manifest.get("tokenizer_name")):
        raise PackedCorpusError("Tokenizer name does not match its manifest")
    if str(revision) != manifest_hash:
        raise PackedCorpusError("Tokenizer revision must be its manifest hash")
    special_ids = tokenizer_manifest.get("special_token_ids", {})
    if int(special_ids.get("eos", -1)) != identity["eos_token_id"]:
        raise PackedCorpusError("Tokenizer EOS ID does not match its manifest")
    model_checksum = tokenizer_manifest.get("sentencepiece_model_sha256")
    if not isinstance(model_checksum, str) or not model_checksum:
        raise PackedCorpusError("Tokenizer model checksum is missing")
    identity.update(
        {
            "manifest_hash": manifest_hash,
            "sentencepiece_model_sha256": model_checksum,
        }
    )
    identity["identity_hash"] = stable_hash(identity)
    return identity


def _document_identity(
    document: Mapping[str, Any],
    *,
    source_index: int,
    source_fingerprint: str,
) -> dict[str, Any]:
    source_id = document.get("id", document.get("source_row_identity", source_index))
    return {
        "source_dataset_fingerprint": source_fingerprint,
        "source_row_identity": source_id,
    }


def reserve_source_documents(
    documents: Iterable[Mapping[str, Any]],
    *,
    source_fingerprint: str,
    role_counts: Mapping[str, int] = RESERVED_ROLE_COUNTS,
) -> tuple[dict[str, list[tuple[dict[str, Any], Mapping[str, Any]]]], Iterator[Mapping[str, Any]]]:
    """Reserve exact source-document roles from an already seeded source order."""

    iterator = iter(documents)
    reserved: dict[str, list[tuple[dict[str, Any], Mapping[str, Any]]]] = {
        role: [] for role in role_counts
    }
    source_index = 0
    for role, count in role_counts.items():
        if int(count) < 0:
            raise PackedCorpusError(f"Reserved role count must be nonnegative: {role}")
        for _ in range(int(count)):
            try:
                document = next(iterator)
            except StopIteration as error:
                raise PackedCorpusError(
                    "FineWeb source ended before all reserved documents were selected"
                ) from error
            identity = _document_identity(
                document,
                source_index=source_index,
                source_fingerprint=source_fingerprint,
            )
            reserved[role].append((identity, document))
            source_index += 1
    return reserved, iterator


class _RoleShardWriter:
    """Write contiguous uint32 tokens without holding a shard in RAM."""

    def __init__(
        self,
        root: Path,
        role: str,
        *,
        context_length: int,
        shard_token_capacity: int,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = root / role
        self.root.mkdir(parents=True, exist_ok=state is not None)
        self.role = role
        self.context_length = int(context_length)
        self.shard_token_capacity = _resolved_shard_token_capacity(
            shard_token_capacity, self.context_length
        )
        self.token_count = 0
        self.source_document_count = 0
        self.discarded_trailing_tokens = 0
        self._discarded_limit_tokens = 0
        self._pending: list[int] = []
        self._shard_index = 0
        self._shard_path: Path | None = None
        self._shard_map: np.memmap | None = None
        self._shard_offset = 0
        self._completed_shards: list[dict[str, Any]] = []
        if state is not None:
            self._restore_state(state)

    def _restore_state(self, state: Mapping[str, Any]) -> None:
        if state.get("role") != self.role:
            raise PackedCorpusError(f"Writer checkpoint role mismatch: {self.role}")
        self.token_count = int(state.get("token_count", -1))
        self.source_document_count = int(state.get("source_document_count", -1))
        self._discarded_limit_tokens = int(
            state.get("discarded_limit_tokens", 0)
        )
        self._pending = [int(value) for value in state.get("pending_tokens", [])]
        self._shard_index = int(state.get("shard_index", -1))
        self._shard_offset = int(state.get("shard_offset", -1))
        self._completed_shards = [dict(item) for item in state.get("completed_shards", [])]
        if min(
            self.token_count,
            self.source_document_count,
            self._shard_index,
            self._shard_offset,
        ) < 0:
            raise PackedCorpusError(f"Invalid writer checkpoint for {self.role}")
        completed_tokens = sum(
            int(shard["token_count"]) for shard in self._completed_shards
        )
        if completed_tokens + self._shard_offset != self.token_count:
            raise PackedCorpusError(f"Writer token-count mismatch for {self.role}")
        for shard in self._completed_shards:
            path = self.root.parent / shard["path"]
            if not path.is_file() or path.stat().st_size != int(shard["byte_count"]):
                raise PackedCorpusError(f"Missing checkpointed shard: {path}")
            if sha256_file(path) != shard["sha256"]:
                raise PackedCorpusError(f"Checkpointed shard checksum mismatch: {path}")
        if self._shard_offset:
            self._shard_path = self.root / f"shard-{self._shard_index:05d}.bin"
            expected_bytes = self.shard_token_capacity * np.dtype(np.uint32).itemsize
            if (
                not self._shard_path.is_file()
                or self._shard_path.stat().st_size != expected_bytes
            ):
                raise PackedCorpusError(
                    f"Invalid partial checkpointed shard: {self._shard_path}"
                )
            self._shard_map = np.memmap(
                self._shard_path,
                mode="r+",
                dtype=np.uint32,
                shape=(self.shard_token_capacity,),
            )

    def add_document(
        self,
        token_ids: Sequence[int],
        *,
        eos_token_id: int,
        max_total_tokens: int | None = None,
    ) -> bool:
        completed_before = len(self._completed_shards)
        self.source_document_count += 1
        values = [int(value) for value in token_ids]
        values.append(int(eos_token_id))
        if any(value < 0 or value > np.iinfo(np.uint32).max for value in values):
            raise PackedCorpusError("Tokenizer emitted an ID outside uint32 range")
        if max_total_tokens is not None:
            remaining = (
                int(max_total_tokens) - self.token_count - len(self._pending)
            )
            if remaining < 0:
                raise PackedCorpusError(
                    f"Packed role {self.role} exceeded its token limit"
                )
            if len(values) > remaining:
                self._discarded_limit_tokens += len(values) - remaining
                values = values[:remaining]
        self._pending.extend(values)
        complete_count = (len(self._pending) // self.context_length) * self.context_length
        if complete_count:
            self._write_tokens(self._pending[:complete_count])
            del self._pending[:complete_count]
        return len(self._completed_shards) != completed_before

    def _open_shard(self) -> None:
        self._shard_path = self.root / f"shard-{self._shard_index:05d}.bin"
        self._shard_map = np.memmap(
            self._shard_path,
            mode="w+",
            dtype=np.uint32,
            shape=(self.shard_token_capacity,),
        )
        self._shard_offset = 0

    def _write_tokens(self, values: Sequence[int]) -> None:
        offset = 0
        while offset < len(values):
            if self._shard_map is None:
                self._open_shard()
            if self._shard_map is None:
                raise PackedCorpusError("Packed shard could not be opened")
            available = len(self._shard_map) - self._shard_offset
            take = min(available, len(values) - offset)
            self._shard_map[self._shard_offset : self._shard_offset + take] = values[
                offset : offset + take
            ]
            self._shard_offset += take
            self.token_count += take
            offset += take
            if self._shard_offset == len(self._shard_map):
                self._close_shard()

    def _close_shard(self) -> None:
        if self._shard_map is None or self._shard_path is None:
            return
        self._shard_map.flush()
        del self._shard_map
        self._shard_map = None
        token_count = self._shard_offset
        byte_count = self._shard_path.stat().st_size
        self._completed_shards.append(
            {
                "path": str(self._shard_path.relative_to(self.root.parent)),
                "token_count": token_count,
                "byte_count": byte_count,
                "sha256": sha256_file(self._shard_path),
            }
        )
        self._shard_path.chmod(0o444)
        self._shard_path = None
        self._shard_offset = 0
        self._shard_index += 1

    def flush(self) -> None:
        if self._shard_map is not None:
            self._shard_map.flush()

    def state_dict(self) -> dict[str, Any]:
        self.flush()
        return {
            "role": self.role,
            "token_count": self.token_count,
            "source_document_count": self.source_document_count,
            "discarded_limit_tokens": self._discarded_limit_tokens,
            "pending_tokens": list(self._pending),
            "shard_index": self._shard_index,
            "shard_offset": self._shard_offset,
            "completed_shards": [dict(item) for item in self._completed_shards],
        }

    def finish(self) -> dict[str, Any]:
        if self._shard_map is not None:
            # Reserved roles may end before the preallocated shard is full.
            path = self._shard_path
            used = self._shard_offset
            self._shard_map.flush()
            del self._shard_map
            self._shard_map = None
            if path is None:
                raise PackedCorpusError("Packed shard path disappeared")
            with path.open("r+b") as shard_file:
                shard_file.truncate(used * np.dtype(np.uint32).itemsize)
            self._completed_shards.append(
                {
                    "path": str(path.relative_to(self.root.parent)),
                    "token_count": used,
                    "byte_count": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
            path.chmod(0o444)
            self._shard_path = None
            self._shard_offset = 0
            self._shard_index += 1
        self.discarded_trailing_tokens = (
            len(self._pending) + self._discarded_limit_tokens
        )
        if self.token_count % self.context_length:
            raise PackedCorpusError(f"Packed role {self.role} is not sequence aligned")
        return {
            "role": self.role,
            "token_count": self.token_count,
            "sequence_count": self.token_count // self.context_length,
            "source_document_count": self.source_document_count,
            "discarded_trailing_tokens": self.discarded_trailing_tokens,
            "shards": self._completed_shards,
            "shard_set_hash": stable_hash(self._completed_shards),
        }


def _tokenize_document(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, padding=False, truncation=False)
    token_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded
    if token_ids is None:
        raise PackedCorpusError("Tokenizer output is missing input_ids")
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise PackedCorpusError("Corpus tokenizer must return one document at a time")
        token_ids = token_ids[0]
    return [int(value) for value in token_ids]


def _ordered_tokenize_documents(
    records: Iterable[tuple[Any, Mapping[str, Any]]],
    tokenizer: Any,
    *,
    text_column: str,
    workers: int,
) -> Iterator[tuple[Any, list[int]]]:
    """Tokenize with bounded concurrency while yielding source order exactly."""

    worker_count = int(workers)

    def tokenize_record(document: Mapping[str, Any]) -> list[int]:
        if text_column not in document:
            raise PackedCorpusError(
                f"Source document is missing text column {text_column!r}"
            )
        return _tokenize_document(tokenizer, str(document[text_column]))

    if worker_count == 1:
        for metadata, document in records:
            yield metadata, tokenize_record(document)
        return

    # Keep memory and streamed-source lookahead bounded. Futures are always
    # consumed from the left, so completion timing cannot reorder the corpus.
    max_pending = worker_count * 4
    pending: deque[tuple[Any, Future[list[int]]]] = deque()
    executor = ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="fineweb-tokenize",
    )
    try:
        for metadata, document in records:
            pending.append((metadata, executor.submit(tokenize_record, document)))
            if len(pending) >= max_pending:
                first_metadata, future = pending.popleft()
                yield first_metadata, future.result()
        while pending:
            metadata, future = pending.popleft()
            yield metadata, future.result()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def iter_streaming_documents_with_ordered_prefetch(
    dataset: Any,
    *,
    workers: int,
) -> Iterator[Mapping[str, Any]]:
    """Read shuffled HF source shards concurrently without changing their order.

    Hugging Face's streaming shuffle first permutes data sources, then applies a
    deterministic in-memory buffer shuffle. We parallelize only the independent
    source-shard reads, consume those futures in their original order, and apply
    the same buffer algorithm in the caller. This makes fresh preparation and
    legacy resume replay use the requested CPUs while preserving corpus identity.

    Unsupported iterable layouts fail explicitly and can be run with one reader
    rather than silently claiming parallelism. The optimization deliberately
    depends on no change to saved preparation state.
    """

    worker_count = int(workers)
    if worker_count <= 0:
        raise PackedCorpusError("Source-read workers must be positive")
    if worker_count == 1:
        yield from dataset
        return

    prepare = getattr(dataset, "_prepare_ex_iterable_for_iteration", None)
    if not callable(prepare):
        raise PackedCorpusError(
            "Parallel source reading requires a sharded Hugging Face "
            "IterableDataset; use source_read_workers=1 for this source"
        )
    prepared = prepare()
    buffered = prepared
    while buffered is not None and not (
        isinstance(getattr(buffered, "buffer_size", None), int)
        and getattr(buffered, "generator", None) is not None
        and callable(getattr(buffered, "_iter_random_indices", None))
    ):
        buffered = getattr(buffered, "ex_iterable", None)
    child = getattr(buffered, "ex_iterable", None)
    source_count = getattr(child, "num_shards", 0) if child is not None else 0
    shard_sources = getattr(child, "shard_data_sources", None)
    if (
        buffered is None
        or child is None
        or not isinstance(source_count, int)
        or source_count <= 1
        or not callable(shard_sources)
    ):
        raise PackedCorpusError(
            "Parallel source reading does not support this Hugging Face iterable "
            "layout; use source_read_workers=1"
        )

    def read_source_shard(index: int, *, as_arrow: bool) -> list[Any]:
        shard = child.shard_data_sources(
            num_shards=source_count,
            index=index,
            contiguous=True,
        )
        if as_arrow:
            arrow_iterator = getattr(shard, "iter_arrow", None)
            if not callable(arrow_iterator):
                raise PackedCorpusError(
                    "A parallel source shard unexpectedly lacks Arrow iteration"
                )
            return list(arrow_iterator())
        return list(shard)

    def ordered_source_records(*, as_arrow: bool) -> Iterator[Any]:
        max_pending = min(source_count, worker_count * 2)
        pending: deque[Future[list[Any]]] = deque()
        next_source = 0
        executor = ThreadPoolExecutor(
            max_workers=min(worker_count, source_count),
            thread_name_prefix="fineweb-read",
        )
        try:
            while next_source < max_pending:
                pending.append(
                    executor.submit(
                        read_source_shard,
                        next_source,
                        as_arrow=as_arrow,
                    )
                )
                next_source += 1
            while pending:
                records = pending.popleft().result()
                if next_source < source_count:
                    pending.append(
                        executor.submit(
                            read_source_shard,
                            next_source,
                            as_arrow=as_arrow,
                        )
                    )
                    next_source += 1
                yield from records
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    class OrderedParallelChild:
        def __iter__(self):
            yield from ordered_source_records(as_arrow=False)

        @property
        def iter_arrow(self):
            return (
                self._iter_arrow
                if callable(getattr(child, "iter_arrow", None))
                else None
            )

        def _iter_arrow(self):
            yield from ordered_source_records(as_arrow=True)

        def __getattr__(self, name: str):
            return getattr(child, name)

    buffered.ex_iterable = OrderedParallelChild()
    try:
        for record in prepared:
            yield record[1] if isinstance(record, tuple) and len(record) == 2 else record
    finally:
        buffered.ex_iterable = child


def prepare_packed_corpus(
    documents: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    output_dir: str | Path,
    *,
    tokenizer_name: str,
    tokenizer_revision: str,
    source_dataset: str = "HuggingFaceFW/fineweb",
    source_config: str = "sample-100BT",
    source_split: str = "train",
    source_fingerprint: str,
    text_column: str = "text",
    data_seed: int = DEFAULT_DATA_SEED,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    shard_token_capacity: int = DEFAULT_SHARD_TOKEN_CAPACITY,
    shuffle_buffer_size: int = DEFAULT_SHUFFLE_BUFFER_SIZE,
    tokenization_workers: int = 1,
    source_read_workers: int = 1,
    tokenizer_manifest: Mapping[str, Any],
    reserved_role_counts: Mapping[str, int] = RESERVED_ROLE_COUNTS,
    optimizer_token_limit: int | None = None,
    minimum_optimizer_document_count: int | None = None,
    role_source_provenance: Mapping[str, Mapping[str, Any]] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    progress_interval_seconds: float = 60.0,
) -> dict[str, Any]:
    """Consume a seeded source and atomically publish a v3 packed corpus."""

    output_path = Path(output_dir).expanduser().resolve()
    work_root = preparation_work_dir(output_path)
    progress_path = work_root / "preparation_progress.json"
    if int(data_seed) != DEFAULT_DATA_SEED:
        raise PackedCorpusError("Unique-token production corpora require data_seed=42")
    if int(context_length) <= 1:
        raise PackedCorpusError("context_length must exceed one token")
    if isinstance(tokenization_workers, bool) or int(tokenization_workers) <= 0:
        raise PackedCorpusError("tokenization_workers must be positive")
    if isinstance(source_read_workers, bool) or int(source_read_workers) <= 0:
        raise PackedCorpusError("source_read_workers must be positive")
    if float(progress_interval_seconds) <= 0:
        raise PackedCorpusError("progress_interval_seconds must be positive")
    resolved_role_counts = {
        str(role): int(count) for role, count in reserved_role_counts.items()
    }
    if tuple(resolved_role_counts) != tuple(RESERVED_ROLE_COUNTS):
        raise PackedCorpusError(
            "reserved_role_counts must preserve ordinary_validation, controller, "
            "final_holdout order"
        )
    if any(count <= 0 for count in resolved_role_counts.values()):
        raise PackedCorpusError("Reserved role counts must be positive")
    if optimizer_token_limit is not None:
        optimizer_token_limit = int(optimizer_token_limit)
        if optimizer_token_limit <= 0 or optimizer_token_limit % int(context_length):
            raise PackedCorpusError(
                "optimizer_token_limit must be positive and context-length aligned"
            )
    if minimum_optimizer_document_count is not None:
        minimum_optimizer_document_count = int(minimum_optimizer_document_count)
        if minimum_optimizer_document_count <= 0:
            raise PackedCorpusError(
                "minimum_optimizer_document_count must be positive"
            )
    normalized_role_sources = (
        {
            str(role): dict(provenance)
            for role, provenance in role_source_provenance.items()
        }
        if role_source_provenance is not None
        else None
    )
    if normalized_role_sources is not None and set(normalized_role_sources) != set(
        ALL_CORPUS_ROLES
    ):
        raise PackedCorpusError(
            "role_source_provenance must describe every packed corpus role"
        )

    token_identity = tokenizer_identity(
        tokenizer,
        name=tokenizer_name,
        revision=tokenizer_revision,
        tokenizer_manifest=tokenizer_manifest,
    )
    resolved_capacity = _resolved_shard_token_capacity(
        shard_token_capacity, context_length
    )
    preparation_configuration = {
        "source": {
            "dataset_name": str(source_dataset),
            "dataset_config_name": str(source_config),
            "split": str(source_split),
            "fingerprint": str(source_fingerprint),
        },
        "tokenizer_identity_hash": token_identity["identity_hash"],
        "context_length": int(context_length),
        "data_seed": int(data_seed),
        "text_column": str(text_column),
        "shuffle_buffer_size": int(shuffle_buffer_size),
        "packing_version": PACKING_VERSION,
        "shard_token_capacity_requested": int(shard_token_capacity),
        "shard_token_capacity_resolved": resolved_capacity,
    }
    if resolved_role_counts != RESERVED_ROLE_COUNTS:
        preparation_configuration["reserved_role_counts"] = dict(
            resolved_role_counts
        )
    if optimizer_token_limit is not None:
        preparation_configuration["optimizer_token_limit"] = int(
            optimizer_token_limit
        )
    if minimum_optimizer_document_count is not None:
        preparation_configuration["minimum_optimizer_document_count"] = int(
            minimum_optimizer_document_count
        )
    if normalized_role_sources is not None:
        preparation_configuration["role_source_provenance"] = copy.deepcopy(
            normalized_role_sources
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = preparation_lock_path(output_path)
    with lock_path.open("a+b") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PackedCorpusError(
                f"Corpus preparation is already running for {output_path}"
            ) from error
        if output_path.exists():
            raise PackedCorpusError(
                f"Prepared corpus directory already exists: {output_path}"
            )

        if work_root.exists():
            if not progress_path.is_file():
                raise PackedCorpusError(
                    f"Incomplete preparation has no valid checkpoint: {work_root}"
                )
            progress = _load_progress(progress_path)
            if progress.get("schema_version") != PREPARATION_PROGRESS_SCHEMA_VERSION:
                raise PackedCorpusError("Resumable preparation schema mismatch")
            if progress.get("configuration") != preparation_configuration:
                raise PackedCorpusError(
                    "Existing partial preparation is incompatible with the requested "
                    "configuration"
                )
            _validate_partial_artifacts(work_root, progress)
        else:
            work_root.mkdir()
            progress = {
                "schema_version": PREPARATION_PROGRESS_SCHEMA_VERSION,
                "configuration": preparation_configuration,
                "phase": "tokenization",
                "committed_shuffled_document_count": 0,
                "rolling_document_identity_hash_version": DOCUMENT_HASH_CHAIN_VERSION,
                "rolling_document_identity_hash": _initial_document_hash(),
                "role_document_identity_hashes": {
                    role: _initial_document_hash() for role in ALL_CORPUS_ROLES
                },
                "role_states": {},
                "role_manifests": {},
                "reserved_identities": {role: [] for role in resolved_role_counts},
                "source_exhausted": False,
            }
            _atomic_write_json(progress_path, _progress_envelope(progress))

        writers: dict[str, _RoleShardWriter] = {}
        last_progress_at = time.monotonic()

        def emit_progress(
            event: str,
            role: str | None = None,
            **event_fields: Any,
        ) -> None:
            if progress_callback is None:
                return
            training_writer = writers.get("optimizer_training")
            training_manifest = progress["role_manifests"].get(
                "optimizer_training"
            )
            if training_writer is not None:
                token_count = training_writer.token_count
                shard_count = len(training_writer._completed_shards)
                current_shard_index = training_writer._shard_index
                current_shard_offset = training_writer._shard_offset
            elif training_manifest is not None:
                token_count = int(training_manifest["token_count"])
                shard_count = len(training_manifest["shards"])
                current_shard_index = shard_count
                current_shard_offset = 0
            else:
                token_count = 0
                shard_count = 0
                current_shard_index = 0
                current_shard_offset = 0
            try:
                payload = {
                    "event": event,
                    "phase": progress["phase"],
                    "role": role,
                    "committed_document_count": int(
                        progress["committed_shuffled_document_count"]
                    ),
                    "optimizer_token_count": int(token_count),
                    "completed_optimizer_shard_count": int(shard_count),
                    "current_optimizer_shard_index": int(current_shard_index),
                    "current_optimizer_shard_offset": int(current_shard_offset),
                    "source_read_workers": int(source_read_workers),
                    "tokenization_workers": int(tokenization_workers),
                }
                payload.update(event_fields)
                progress_callback(payload)
            except Exception:
                # Progress reporting must never invalidate expensive corpus work.
                return

        def get_writer(role: str) -> _RoleShardWriter:
            if role not in writers:
                saved = progress["role_states"].get(role)
                writers[role] = _RoleShardWriter(
                    work_root,
                    role,
                    context_length=context_length,
                    shard_token_capacity=shard_token_capacity,
                    state=saved,
                )
            return writers[role]

        def save_progress() -> None:
            for role, writer in writers.items():
                if role not in progress["role_manifests"]:
                    progress["role_states"][role] = writer.state_dict()
            _atomic_write_json(progress_path, _progress_envelope(progress))

        def finish_role(role: str) -> None:
            if role in progress["role_manifests"]:
                return
            writer = get_writer(role)
            role_manifest = writer.finish()
            identities = progress["reserved_identities"].get(role)
            if identities is not None:
                role_manifest.update(
                    ordered_source_document_identities=identities,
                    source_document_identity_hash=stable_hash(identities),
                    example_count=len(identities),
                    ordered_example_identities=identities,
                    example_identity_hash=stable_hash(identities),
                )
            else:
                role_manifest.update(
                    source_document_identity_hash_version=DOCUMENT_HASH_CHAIN_VERSION,
                    source_document_identity_stream_hash=progress[
                        "role_document_identity_hashes"
                    ][role],
                    example_count=role_manifest["sequence_count"],
                )
            progress["role_manifests"][role] = role_manifest
            progress["role_states"].pop(role, None)
            save_progress()
            emit_progress("role_completed", role)

        reserved_boundaries: list[tuple[int, str]] = []
        boundary = 0
        for role, count in resolved_role_counts.items():
            boundary += int(count)
            reserved_boundaries.append((boundary, role))
        reserved_total = boundary
        reserved_identity_hashes = {
            stable_hash(identity)
            for role in resolved_role_counts
            for identity in progress["reserved_identities"][role]
        }

        def role_for_index(index: int) -> str:
            for end, role in reserved_boundaries:
                if index < end:
                    return role
            return "optimizer_training"

        try:
            if progress["phase"] == "tokenization":
                source_iterator = iter(documents)
                committed = int(progress["committed_shuffled_document_count"])
                replay_started_at = time.monotonic()
                for skipped in range(committed):
                    try:
                        next(source_iterator)
                    except StopIteration as error:
                        raise PackedCorpusError(
                            "Source ended before the resumable checkpoint offset"
                        ) from error
                    now = time.monotonic()
                    if now - last_progress_at >= float(progress_interval_seconds):
                        replayed = skipped + 1
                        replay_elapsed = max(now - replay_started_at, 1e-9)
                        emit_progress(
                            "resume_replay",
                            "optimizer_training",
                            replayed_document_count=replayed,
                            replay_target_document_count=committed,
                            replay_elapsed_seconds=replay_elapsed,
                            replay_documents_per_second=replayed / replay_elapsed,
                        )
                        last_progress_at = now
                if committed:
                    replay_elapsed = max(
                        time.monotonic() - replay_started_at,
                        1e-9,
                    )
                    emit_progress(
                        "resume_replay_completed",
                        "optimizer_training",
                        replayed_document_count=committed,
                        replay_target_document_count=committed,
                        replay_elapsed_seconds=replay_elapsed,
                        replay_documents_per_second=committed / replay_elapsed,
                    )

                def pending_records():
                    for source_index, document in enumerate(
                        source_iterator, start=committed
                    ):
                        identity = _document_identity(
                            document,
                            source_index=source_index,
                            source_fingerprint=source_fingerprint,
                        )
                        yield (source_index, identity, document)

                optimizer_limit_already_reached = bool(
                    optimizer_token_limit is not None
                    and get_writer("optimizer_training").token_count
                    == optimizer_token_limit
                )
                for metadata, token_ids in _ordered_tokenize_documents(
                    (
                        ((source_index, identity), document)
                        for source_index, identity, document in (
                            () if optimizer_limit_already_reached else pending_records()
                        )
                    ),
                    tokenizer,
                    text_column=text_column,
                    workers=int(tokenization_workers),
                ):
                    source_index, identity = metadata
                    role = role_for_index(source_index)
                    identity_hash = stable_hash(identity)
                    if (
                        role == "optimizer_training"
                        and identity_hash in reserved_identity_hashes
                    ):
                        raise PackedCorpusError(
                            "A reserved source document reappeared in training"
                        )
                    writer = get_writer(role)
                    completed_shard = writer.add_document(
                        token_ids,
                        eos_token_id=int(token_identity["eos_token_id"]),
                        max_total_tokens=(
                            optimizer_token_limit
                            if role == "optimizer_training"
                            else None
                        ),
                    )
                    if role in resolved_role_counts:
                        progress["reserved_identities"][role].append(identity)
                        reserved_identity_hashes.add(identity_hash)
                    progress["rolling_document_identity_hash"] = _extend_document_hash(
                        progress["rolling_document_identity_hash"], identity
                    )
                    progress["role_document_identity_hashes"][role] = (
                        _extend_document_hash(
                            progress["role_document_identity_hashes"][role], identity
                        )
                    )
                    progress["committed_shuffled_document_count"] = source_index + 1
                    if source_index + 1 <= reserved_total:
                        for end, reserved_role in reserved_boundaries:
                            if source_index + 1 == end:
                                finish_role(reserved_role)
                                break
                    elif completed_shard:
                        save_progress()
                        emit_progress("shard_completed", role)
                    now = time.monotonic()
                    if now - last_progress_at >= float(progress_interval_seconds):
                        emit_progress("progress", role)
                        last_progress_at = now
                    if (
                        role == "optimizer_training"
                        and optimizer_token_limit is not None
                        and writer.token_count == optimizer_token_limit
                    ):
                        if (
                            minimum_optimizer_document_count is not None
                            and writer.source_document_count
                            < minimum_optimizer_document_count
                        ):
                            raise PackedCorpusError(
                                "optimizer_token_limit was reached before all "
                                "tokenizer-training stories entered the optimizer "
                                "corpus"
                            )
                        break

                if int(progress["committed_shuffled_document_count"]) < reserved_total:
                    raise PackedCorpusError(
                        "Source ended before all reserved documents were selected"
                    )
                if optimizer_token_limit is not None:
                    actual_optimizer_tokens = get_writer(
                        "optimizer_training"
                    ).token_count
                    if actual_optimizer_tokens != optimizer_token_limit:
                        raise PackedCorpusError(
                            "Source ended before optimizer_token_limit was reached: "
                            f"expected {optimizer_token_limit}, found "
                            f"{actual_optimizer_tokens}"
                        )
                actual_optimizer_documents = get_writer(
                    "optimizer_training"
                ).source_document_count
                if (
                    minimum_optimizer_document_count is not None
                    and actual_optimizer_documents
                    < minimum_optimizer_document_count
                ):
                    raise PackedCorpusError(
                        "Optimizer corpus contains fewer documents than required: "
                        f"expected at least {minimum_optimizer_document_count}, "
                        f"found {actual_optimizer_documents}"
                    )
                finish_role("optimizer_training")
                progress["source_exhausted"] = True
                progress["phase"] = "permutation"
                save_progress()
                emit_progress("source_exhausted", "optimizer_training")

            if progress["phase"] == "permutation":
                sequence_count = int(
                    progress["role_manifests"]["optimizer_training"][
                        "sequence_count"
                    ]
                )
                ordering_path = work_root / "optimizer_training_order.u64"
                temporary_ordering_path = work_root / ".optimizer_training_order.u64.tmp"
                permutation = np.random.Generator(
                    np.random.PCG64(int(data_seed))
                ).permutation(sequence_count)
                if sequence_count:
                    ordering = np.memmap(
                        temporary_ordering_path,
                        mode="w+",
                        dtype="<u8",
                        shape=(sequence_count,),
                    )
                    ordering[:] = permutation
                    ordering.flush()
                    del ordering
                else:
                    temporary_ordering_path.touch()
                os.replace(temporary_ordering_path, ordering_path)
                ordering_path.chmod(0o444)
                progress["training_order"] = {
                    "path": ordering_path.name,
                    "dtype": "uint64_le",
                    "count": sequence_count,
                    "byte_count": ordering_path.stat().st_size,
                    "sha256": sha256_file(ordering_path),
                    "seed": int(data_seed),
                    "permutation_version": PERMUTATION_VERSION,
                }
                progress["phase"] = "publish"
                save_progress()
                emit_progress("permutation_completed", "optimizer_training")

            role_manifests = {
                role: dict(progress["role_manifests"][role])
                for role in ALL_CORPUS_ROLES
            }
            role_hashes: dict[str, str] = {}
            for role in ALL_CORPUS_ROLES:
                role_manifests[role]["manifest_hash"] = stable_hash(
                    role_manifests[role]
                )
                role_hashes[role] = role_manifests[role]["manifest_hash"]

            reserved_identity_sets = {
                role: {
                    stable_hash(identity)
                    for identity in progress["reserved_identities"][role]
                }
                for role in resolved_role_counts
            }
            pairwise_intersections: dict[str, int] = {}
            reserved_roles = tuple(resolved_role_counts)
            for left_index, left in enumerate(reserved_roles):
                for right in reserved_roles[left_index + 1 :]:
                    count = len(
                        reserved_identity_sets[left] & reserved_identity_sets[right]
                    )
                    pairwise_intersections[f"{left}__{right}"] = count
                    if count:
                        raise PackedCorpusError(
                            f"Reserved document overlap: {left}, {right}"
                        )

            training_role = role_manifests["optimizer_training"]
            manifest = {
                "schema_version": PACKED_CORPUS_SCHEMA_VERSION,
                "packing_version": PACKING_VERSION,
                "data_seed": int(data_seed),
                "context_length": int(context_length),
                "dtype": "uint32",
                "source": {
                    "dataset_name": source_dataset,
                    "dataset_config_name": source_config,
                    "split": source_split,
                    "fingerprint": source_fingerprint,
                    "source_exhausted": optimizer_token_limit is None,
                    "termination": (
                        "optimizer_token_limit"
                        if optimizer_token_limit is not None
                        else "source_exhausted"
                    ),
                    "shuffled_document_count": int(
                        progress["committed_shuffled_document_count"]
                    ),
                    "rolling_document_identity_hash_version": DOCUMENT_HASH_CHAIN_VERSION,
                    "rolling_document_identity_hash": progress[
                        "rolling_document_identity_hash"
                    ],
                },
                "tokenizer": token_identity,
                "available_optimizer_token_count": int(training_role["token_count"]),
                "available_optimizer_sequence_count": int(
                    training_role["sequence_count"]
                ),
                "training_order": dict(progress["training_order"]),
                "preparation": {
                    "identity_hash": stable_hash(preparation_configuration),
                    "text_column": str(text_column),
                    "shuffle": {
                        "seed": int(data_seed),
                        "buffer_size": int(shuffle_buffer_size),
                    },
                    "shard_token_capacity": {
                        "requested": int(shard_token_capacity),
                        "resolved": resolved_capacity,
                    },
                },
                "role_selection_order": list(resolved_role_counts),
                "reserved_role_counts": dict(resolved_role_counts),
                "reserved_pairwise_intersection_counts": pairwise_intersections,
                "role_manifest_hashes": role_hashes,
                "roles": role_manifests,
            }
            if optimizer_token_limit is not None:
                manifest["optimizer_token_limit"] = int(optimizer_token_limit)
            if minimum_optimizer_document_count is not None:
                manifest["minimum_optimizer_document_count"] = int(
                    minimum_optimizer_document_count
                )
            if normalized_role_sources is not None:
                manifest["role_source_provenance"] = copy.deepcopy(
                    normalized_role_sources
                )
            manifest["corpus_hash"] = stable_hash(manifest)
            manifest_path = work_root / "corpus_manifest.json"
            _atomic_write_json(manifest_path, manifest)
            manifest_path.chmod(0o444)
            progress_path.unlink()
            os.replace(work_root, output_path)
            _fsync_directory(output_path.parent)
            return manifest
        except Exception:
            save_progress()
            raise


def load_corpus_manifest(
    prepared_corpus_dir: str | Path,
    *,
    verify_shards: bool = False,
) -> dict[str, Any]:
    root = Path(prepared_corpus_dir).expanduser().resolve()
    manifest_path = root / "corpus_manifest.json"
    if not manifest_path.is_file():
        raise PackedCorpusError(f"Missing prepared corpus manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PackedCorpusError(f"Invalid prepared corpus manifest: {manifest_path}") from error
    saved_hash = manifest.get("corpus_hash")
    payload = dict(manifest)
    payload.pop("corpus_hash", None)
    if saved_hash != stable_hash(payload):
        raise PackedCorpusError("Prepared corpus manifest hash mismatch")
    schema_version = manifest.get("schema_version")
    if schema_version != PACKED_CORPUS_SCHEMA_VERSION:
        if schema_version == 2:
            raise PackedCorpusError(
                "Prepared corpus schema v2 is no longer supported; rebuild the "
                "full source as a schema-v3 corpus"
            )
        raise PackedCorpusError(
            f"Prepared corpus schema version mismatch: expected v3, found "
            f"{schema_version!r}; rebuild the corpus"
        )
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, Mapping) or any(
        field not in tokenizer
        for field in (
            "manifest_hash",
            "sentencepiece_model_sha256",
            "vocab_size",
        )
    ):
        raise PackedCorpusError("Prepared corpus tokenizer provenance is incomplete")
    if manifest.get("packing_version") != PACKING_VERSION:
        raise PackedCorpusError("Prepared corpus packing version mismatch")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise PackedCorpusError("Prepared corpus source provenance is missing")
    termination = source.get("termination", "source_exhausted")
    if termination == "source_exhausted":
        if source.get("source_exhausted") is not True:
            raise PackedCorpusError(
                "Prepared corpus does not record source exhaustion"
            )
    elif termination == "optimizer_token_limit":
        if source.get("source_exhausted") is not False:
            raise PackedCorpusError(
                "Token-limited corpus must record an unexhausted physical source"
            )
        declared_limit = manifest.get("optimizer_token_limit")
        if (
            isinstance(declared_limit, bool)
            or not isinstance(declared_limit, int)
            or declared_limit <= 0
        ):
            raise PackedCorpusError(
                "Token-limited corpus optimizer token limit is missing"
            )
    else:
        raise PackedCorpusError("Prepared corpus source termination is invalid")
    if manifest.get("dtype") != "uint32":
        raise PackedCorpusError("Prepared corpus dtype must be uint32")
    roles = manifest.get("roles")
    if not isinstance(roles, Mapping) or any(role not in roles for role in ALL_CORPUS_ROLES):
        raise PackedCorpusError("Prepared corpus roles are incomplete")
    for role in ALL_CORPUS_ROLES:
        role_manifest = roles[role]
        saved_role_hash = role_manifest.get("manifest_hash")
        role_payload = dict(role_manifest)
        role_payload.pop("manifest_hash", None)
        if saved_role_hash != stable_hash(role_payload):
            raise PackedCorpusError(f"Prepared corpus role hash mismatch: {role}")
        if manifest["role_manifest_hashes"].get(role) != saved_role_hash:
            raise PackedCorpusError(f"Prepared corpus parent role hash mismatch: {role}")
        total_tokens = 0
        for shard in role_manifest.get("shards", []):
            shard_path = root / shard["path"]
            if not shard_path.is_file():
                raise PackedCorpusError(f"Prepared corpus shard is missing: {shard_path}")
            if shard_path.stat().st_size != int(shard["byte_count"]):
                raise PackedCorpusError(f"Prepared corpus shard size mismatch: {shard_path}")
            if verify_shards and sha256_file(shard_path) != shard["sha256"]:
                raise PackedCorpusError(f"Prepared corpus shard checksum mismatch: {shard_path}")
            total_tokens += int(shard["token_count"])
        if total_tokens != int(role_manifest["token_count"]):
            raise PackedCorpusError(f"Prepared corpus role token count mismatch: {role}")
    training = roles["optimizer_training"]
    if int(manifest.get("available_optimizer_token_count", -1)) != int(
        training["token_count"]
    ) or int(manifest.get("available_optimizer_sequence_count", -1)) != int(
        training["sequence_count"]
    ):
        raise PackedCorpusError("Prepared corpus available optimizer counts mismatch")
    optimizer_token_limit = manifest.get("optimizer_token_limit")
    if optimizer_token_limit is not None and int(optimizer_token_limit) != int(
        training["token_count"]
    ):
        raise PackedCorpusError(
            "Prepared corpus optimizer token limit does not match its training role"
        )
    minimum_optimizer_documents = manifest.get("minimum_optimizer_document_count")
    if minimum_optimizer_documents is not None:
        if (
            isinstance(minimum_optimizer_documents, bool)
            or not isinstance(minimum_optimizer_documents, int)
            or minimum_optimizer_documents <= 0
        ):
            raise PackedCorpusError(
                "Prepared corpus minimum optimizer document count is invalid"
            )
        if minimum_optimizer_documents > int(training["source_document_count"]):
            raise PackedCorpusError(
                "Prepared corpus does not contain its required optimizer documents"
            )
    ordering = manifest.get("training_order")
    if not isinstance(ordering, Mapping):
        raise PackedCorpusError("Prepared corpus training-order metadata is missing")
    if ordering.get("permutation_version") != PERMUTATION_VERSION:
        raise PackedCorpusError("Prepared corpus permutation version mismatch")
    if ordering.get("dtype") != "uint64_le":
        raise PackedCorpusError("Prepared corpus permutation dtype must be uint64_le")
    if int(ordering.get("seed", -1)) != int(manifest["data_seed"]):
        raise PackedCorpusError("Prepared corpus permutation seed mismatch")
    if int(ordering.get("count", -1)) != int(training["sequence_count"]):
        raise PackedCorpusError("Prepared corpus permutation count mismatch")
    if int(ordering.get("byte_count", -1)) != int(ordering["count"]) * 8:
        raise PackedCorpusError("Prepared corpus permutation byte count mismatch")
    order_path = root / str(ordering.get("path", ""))
    if not order_path.is_file():
        raise PackedCorpusError(f"Prepared corpus permutation is missing: {order_path}")
    if order_path.stat().st_size != int(ordering.get("byte_count", -1)):
        raise PackedCorpusError("Prepared corpus permutation byte size mismatch")
    if verify_shards and sha256_file(order_path) != ordering.get("sha256"):
        raise PackedCorpusError("Prepared corpus permutation checksum mismatch")
    return manifest


def load_existing_corpus_if_matching(
    prepared_corpus_dir: str | Path,
    *,
    tokenizer_manifest: Mapping[str, Any],
    tokenizer_name: str,
    tokenizer_revision: str,
    source_dataset: str = "HuggingFaceFW/fineweb",
    source_config: str = "sample-100BT",
    source_split: str = "train",
    text_column: str = "text",
    data_seed: int = DEFAULT_DATA_SEED,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    shard_token_capacity: int = DEFAULT_SHARD_TOKEN_CAPACITY,
    shuffle_buffer_size: int = DEFAULT_SHUFFLE_BUFFER_SIZE,
    reserved_role_counts: Mapping[str, int] = RESERVED_ROLE_COUNTS,
    optimizer_token_limit: int | None = None,
    minimum_optimizer_document_count: int | None = None,
    role_source_provenance: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Load a fully verified exact-match corpus, or return ``None`` if absent."""

    root = Path(prepared_corpus_dir).expanduser().resolve()
    if not root.exists():
        return None
    # A changed request should fail from metadata without scanning the full-source
    # artifact. Exact-match reuse still requires a complete checksum pass.
    manifest = load_corpus_manifest(root, verify_shards=False)
    tokenizer_payload = dict(tokenizer_manifest)
    manifest_hash = tokenizer_payload.pop("manifest_hash", None)
    if not isinstance(manifest_hash, str) or manifest_hash != stable_hash(
        tokenizer_payload
    ):
        raise PackedCorpusError("Tokenizer manifest hash mismatch")
    special_ids = tokenizer_manifest.get("special_token_ids", {})
    expected = {
        "data_seed": int(data_seed),
        "context_length": int(context_length),
        "source.dataset_name": str(source_dataset),
        "source.dataset_config_name": str(source_config),
        "source.split": str(source_split),
        "tokenizer.name": str(tokenizer_name),
        "tokenizer.revision": str(tokenizer_revision),
        "tokenizer.manifest_hash": manifest_hash,
        "tokenizer.sentencepiece_model_sha256": tokenizer_manifest.get(
            "sentencepiece_model_sha256"
        ),
        "tokenizer.vocab_size": int(tokenizer_manifest.get("vocab_size", 0) or 0),
        "tokenizer.eos_token_id": int(special_ids.get("eos", -1)),
        "preparation.text_column": str(text_column),
        "preparation.shuffle.seed": int(data_seed),
        "preparation.shuffle.buffer_size": int(shuffle_buffer_size),
        "preparation.shard_token_capacity.requested": int(shard_token_capacity),
        "preparation.shard_token_capacity.resolved": _resolved_shard_token_capacity(
            shard_token_capacity, context_length
        ),
        "reserved_role_counts": {
            str(role): int(count) for role, count in reserved_role_counts.items()
        },
    }
    if optimizer_token_limit is not None:
        expected["optimizer_token_limit"] = int(optimizer_token_limit)
        expected["source.termination"] = "optimizer_token_limit"
        expected["source.source_exhausted"] = False
    else:
        expected["optimizer_token_limit"] = None
        expected["source.termination"] = "source_exhausted"
        expected["source.source_exhausted"] = True
    if minimum_optimizer_document_count is not None:
        expected["minimum_optimizer_document_count"] = int(
            minimum_optimizer_document_count
        )
    if role_source_provenance is not None:
        expected["role_source_provenance"] = {
            str(role): dict(provenance)
            for role, provenance in role_source_provenance.items()
        }

    def value_at(path: str) -> Any:
        value: Any = manifest
        for component in path.split("."):
            if not isinstance(value, Mapping) or component not in value:
                return None
            value = value[component]
        return value

    mismatches = {
        field: {"actual": value_at(field), "expected": expected_value}
        for field, expected_value in expected.items()
        if value_at(field) != expected_value
    }
    if mismatches:
        raise PackedCorpusError(
            "Existing prepared corpus does not match the requested preparation: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return load_corpus_manifest(root, verify_shards=True)


def audit_packed_corpus(
    prepared_corpus_dir: str | Path,
    *,
    minimum_training_tokens: int = 0,
    prepared_tokenizer_dir: str | Path | None = None,
    required_vocab_size: int | None = None,
) -> dict[str, Any]:
    manifest = load_corpus_manifest(prepared_corpus_dir, verify_shards=True)
    training = manifest["roles"]["optimizer_training"]
    if int(minimum_training_tokens) < 0:
        raise PackedCorpusError("minimum_training_tokens must be nonnegative")
    if int(training["token_count"]) < int(minimum_training_tokens):
        raise PackedCorpusError(
            f"Training token audit requires at least {minimum_training_tokens}, "
            f"found {training['token_count']}"
        )
    ordering = manifest["training_order"]
    ordering_count = int(ordering["count"])
    order = (
        np.memmap(
            Path(prepared_corpus_dir).expanduser().resolve() / ordering["path"],
            mode="r",
            dtype="<u8",
        )
        if ordering_count
        else np.asarray([], dtype="<u8")
    )
    seen = np.zeros(ordering_count, dtype=np.bool_)
    chunk_size = 4 * 1024 * 1024
    for start in range(0, ordering_count, chunk_size):
        values = np.asarray(order[start : start + chunk_size])
        if values.size and int(values.max()) >= ordering_count:
            raise PackedCorpusError("Prepared corpus permutation index is out of range")
        if len(np.unique(values)) != len(values) or np.any(seen[values]):
            raise PackedCorpusError("Prepared corpus permutation contains duplicates")
        seen[values] = True
    if ordering_count and not bool(np.all(seen)):
        raise PackedCorpusError("Prepared corpus permutation is incomplete")
    intersections = manifest.get("reserved_pairwise_intersection_counts", {})
    if any(int(value) != 0 for value in intersections.values()):
        raise PackedCorpusError("Prepared corpus reserved roles overlap")
    tokenizer = manifest["tokenizer"]
    if required_vocab_size is not None and int(tokenizer.get("vocab_size", -1)) != int(
        required_vocab_size
    ):
        raise PackedCorpusError(
            "Prepared corpus tokenizer vocabulary does not match the required size"
        )
    if prepared_tokenizer_dir is not None:
        from src.training.fineweb_tokenizer import load_tokenizer_manifest

        tokenizer_manifest = load_tokenizer_manifest(
            prepared_tokenizer_dir, verify_files=True
        )
        expected = {
            "manifest_hash": tokenizer_manifest["manifest_hash"],
            "sentencepiece_model_sha256": tokenizer_manifest[
                "sentencepiece_model_sha256"
            ],
            "vocab_size": tokenizer_manifest["vocab_size"],
        }
        mismatches = {
            field: (tokenizer.get(field), value)
            for field, value in expected.items()
            if tokenizer.get(field) != value
        }
        if mismatches:
            raise PackedCorpusError(
                f"Prepared corpus tokenizer provenance mismatch: {mismatches}"
            )
    return {
        "status": "passed",
        "corpus_hash": manifest["corpus_hash"],
        "training_token_count": int(training["token_count"]),
        "training_sequence_count": int(training["sequence_count"]),
        "role_manifest_hashes": dict(manifest["role_manifest_hashes"]),
        "verified_shard_count": sum(
            len(role["shards"]) for role in manifest["roles"].values()
        ),
        "source_document_count": int(
            manifest["source"]["shuffled_document_count"]
        ),
        "training_order_sha256": manifest["training_order"]["sha256"],
        "verified_training_order_count": ordering_count,
        "schema_version": manifest["schema_version"],
        "tokenizer_manifest_hash": tokenizer.get("manifest_hash"),
        "tokenizer_model_sha256": tokenizer.get("sentencepiece_model_sha256"),
        "tokenizer_vocab_size": tokenizer.get("vocab_size"),
    }


class PackedMMapDataset(Dataset):
    """Lazy uint32 memory-map view over one prepared role."""

    def __init__(self, prepared_corpus_dir: str | Path, role: str) -> None:
        if role not in ALL_CORPUS_ROLES:
            raise PackedCorpusError(f"Unknown prepared corpus role: {role}")
        self.root = Path(prepared_corpus_dir).expanduser().resolve()
        self.manifest = load_corpus_manifest(self.root, verify_shards=False)
        self.role = role
        self.role_manifest = self.manifest["roles"][role]
        self.context_length = int(self.manifest["context_length"])
        self._sequence_offsets: list[tuple[int, int, int]] = []
        sequence_cursor = 0
        for shard_index, shard in enumerate(self.role_manifest["shards"]):
            count = int(shard["token_count"]) // self.context_length
            self._sequence_offsets.append(
                (sequence_cursor, sequence_cursor + count, shard_index)
            )
            sequence_cursor += count
        self._maps: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return int(self.role_manifest["sequence_count"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        for start, end, shard_index in self._sequence_offsets:
            if start <= index < end:
                mmap = self._maps.get(shard_index)
                shard = self.role_manifest["shards"][shard_index]
                if mmap is None:
                    mmap = np.memmap(
                        self.root / shard["path"],
                        mode="r",
                        dtype=np.uint32,
                    )
                    self._maps[shard_index] = mmap
                local = (index - start) * self.context_length
                input_ids = np.asarray(
                    mmap[local : local + self.context_length], dtype=np.int64
                ).copy()
                return {
                    "input_ids": input_ids,
                    "attention_mask": np.ones(self.context_length, dtype=np.int64),
                    "packed_sequence_index": index,
                }
        raise IndexError(index)


def deterministic_permutation(length: int, *, data_seed: int) -> np.ndarray:
    if int(length) < 0:
        raise PackedCorpusError("Permutation length must be nonnegative")
    return np.random.Generator(np.random.PCG64(int(data_seed))).permutation(int(length))


@dataclass
class NoPaddingDistributedBatchSampler(Sampler[list[int]]):
    """Stored corpus permutation prefix, partitioned without rank padding."""

    dataset_size: int
    batch_size_per_rank: int
    rank: int
    world_size: int
    data_seed: int = DEFAULT_DATA_SEED
    cursor: int = 0
    selected_sample_count: int | None = None
    permutation_path: str | Path | None = None
    permutation_hash_expected: str | None = None
    permutation_version: str = PERMUTATION_VERSION

    def __post_init__(self) -> None:
        self.dataset_size = int(self.dataset_size)
        self.batch_size_per_rank = int(self.batch_size_per_rank)
        self.rank = int(self.rank)
        self.world_size = int(self.world_size)
        self.data_seed = int(self.data_seed)
        self.cursor = int(self.cursor)
        self.selected_sample_count = (
            self.dataset_size
            if self.selected_sample_count is None
            else int(self.selected_sample_count)
        )
        if self.dataset_size <= 0:
            raise PackedCorpusError("No-padding sampler requires a nonempty dataset")
        if self.batch_size_per_rank <= 0:
            raise PackedCorpusError("batch_size_per_rank must be positive")
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise PackedCorpusError("Invalid distributed rank topology")
        if not 0 < self.selected_sample_count <= self.dataset_size:
            raise PackedCorpusError(
                "selected_sample_count must be within the available corpus"
            )
        if not 0 <= self.cursor <= self.selected_sample_count:
            raise PackedCorpusError("Sampler cursor is outside the permutation")
        if self.permutation_version != PERMUTATION_VERSION:
            raise PackedCorpusError("Unsupported stored permutation version")
        if self.permutation_path is None:
            # Retained for small standalone sampler tests; packed-mmap production
            # always supplies the corpus-owned memory map.
            self._permutation = deterministic_permutation(
                self.dataset_size, data_seed=self.data_seed
            ).astype("<u8", copy=False)
            self._permutation_hash = hashlib.sha256(
                self._permutation.tobytes()
            ).hexdigest()
        else:
            path = Path(self.permutation_path).expanduser().resolve()
            if not path.is_file() or path.stat().st_size != self.dataset_size * 8:
                raise PackedCorpusError("Stored permutation size does not match dataset")
            self._permutation = np.memmap(path, mode="r", dtype="<u8")
            actual_hash = sha256_file(path)
            if (
                self.permutation_hash_expected is not None
                and actual_hash != str(self.permutation_hash_expected)
            ):
                raise PackedCorpusError("Stored permutation checksum mismatch")
            self._permutation_hash = actual_hash
        self.last_yielded_cursor = self.cursor

    @property
    def global_batch_size(self) -> int:
        return self.batch_size_per_rank * self.world_size

    @property
    def permutation_hash(self) -> str:
        return self._permutation_hash

    def __iter__(self):
        cursor = self.cursor
        while cursor < self.selected_sample_count:
            end = min(self.selected_sample_count, cursor + self.global_batch_size)
            global_indices = self._permutation[cursor:end]
            base, remainder = divmod(len(global_indices), self.world_size)
            local_start = self.rank * base + min(self.rank, remainder)
            local_count = base + (1 if self.rank < remainder else 0)
            local_indices = global_indices[local_start : local_start + local_count]
            if local_count == 0:
                raise PackedCorpusError(
                    "Final global batch is smaller than world_size; it cannot "
                    "participate in a collective optimizer step without padding"
                )
            cursor = end
            self.last_yielded_cursor = cursor
            yield [int(value) for value in local_indices]

    def __len__(self) -> int:
        remaining = self.selected_sample_count - self.cursor
        return (remaining + self.global_batch_size - 1) // self.global_batch_size

    def set_cursor(self, cursor: int) -> None:
        cursor = int(cursor)
        if not 0 <= cursor <= self.selected_sample_count:
            raise PackedCorpusError("Sampler cursor is outside the permutation")
        if cursor != self.selected_sample_count and cursor % self.global_batch_size:
            raise PackedCorpusError("Sampler cursor must identify a global-batch boundary")
        self.cursor = cursor
        self.last_yielded_cursor = cursor

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "permutation_version": self.permutation_version,
            "data_seed": self.data_seed,
            "dataset_size": self.dataset_size,
            "selected_sample_count": self.selected_sample_count,
            "batch_size_per_rank": self.batch_size_per_rank,
            "world_size": self.world_size,
            "cursor": self.last_yielded_cursor,
            "permutation_hash": self.permutation_hash,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = self.state_dict()
        for field in (
            "schema_version",
            "permutation_version",
            "data_seed",
            "dataset_size",
            "selected_sample_count",
            "batch_size_per_rank",
            "world_size",
            "permutation_hash",
        ):
            if state.get(field) != expected[field]:
                raise PackedCorpusError(f"Sampler resume mismatch for {field}")
        self.set_cursor(int(state.get("cursor", -1)))


def partition_permutation_without_padding(
    length: int,
    *,
    world_size: int,
    data_seed: int = DEFAULT_DATA_SEED,
) -> list[list[int]]:
    """Test/audit helper returning disjoint rank partitions of one permutation."""

    permutation = deterministic_permutation(length, data_seed=data_seed).tolist()
    return [permutation[rank::world_size] for rank in range(int(world_size))]


__all__ = [
    "ALL_CORPUS_ROLES",
    "DEFAULT_CONTEXT_LENGTH",
    "DEFAULT_DATA_SEED",
    "DOCUMENT_HASH_CHAIN_VERSION",
    "NoPaddingDistributedBatchSampler",
    "PACKED_CORPUS_SCHEMA_VERSION",
    "PACKING_VERSION",
    "PERMUTATION_VERSION",
    "PackedCorpusError",
    "PackedMMapDataset",
    "RESERVED_ROLE_COUNTS",
    "audit_packed_corpus",
    "deterministic_permutation",
    "load_corpus_manifest",
    "partition_permutation_without_padding",
    "preparation_lock_path",
    "preparation_work_dir",
    "prepare_packed_corpus",
    "reserve_source_documents",
    "sha256_file",
]
