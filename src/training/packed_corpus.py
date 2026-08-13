"""Immutable packed-token corpus preparation and memory-mapped access.

The production path deliberately keeps source-document selection separate from
packing.  Reserved documents are selected before any token is written, while
training provenance is represented by hashes instead of a JSON identity per
packed sequence.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
from torch.utils.data import Dataset, Sampler

from src.utils.reproducibility import stable_hash


PACKED_CORPUS_SCHEMA_VERSION = 2
PACKING_VERSION = "contiguous_eos_uint32_v1"
PERMUTATION_VERSION = "numpy_pcg64_permutation_v1"
DEFAULT_DATA_SEED = 42
DEFAULT_CONTEXT_LENGTH = 1024
DEFAULT_TRAINING_TOKEN_BUDGET = 10_000_000_000
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
        exact_token_limit: int | None,
    ) -> None:
        self.root = root / role
        self.root.mkdir(parents=True, exist_ok=False)
        self.role = role
        self.context_length = int(context_length)
        self.shard_token_capacity = max(
            self.context_length,
            (int(shard_token_capacity) // self.context_length) * self.context_length,
        )
        self.exact_token_limit = (
            None if exact_token_limit is None else int(exact_token_limit)
        )
        self.token_count = 0
        self.source_document_count = 0
        self.discarded_trailing_tokens = 0
        self._pending: list[int] = []
        self._shard_index = 0
        self._shard_path: Path | None = None
        self._shard_map: np.memmap | None = None
        self._shard_offset = 0
        self._completed_shards: list[dict[str, Any]] = []

    @property
    def full(self) -> bool:
        return self.exact_token_limit is not None and self.token_count >= self.exact_token_limit

    def add_document(self, token_ids: Sequence[int], *, eos_token_id: int) -> None:
        if self.full:
            return
        self.source_document_count += 1
        values = [int(value) for value in token_ids]
        values.append(int(eos_token_id))
        if any(value < 0 or value > np.iinfo(np.uint32).max for value in values):
            raise PackedCorpusError("Tokenizer emitted an ID outside uint32 range")
        self._pending.extend(values)
        complete_count = (len(self._pending) // self.context_length) * self.context_length
        if self.exact_token_limit is not None:
            complete_count = min(
                complete_count,
                self.exact_token_limit - self.token_count,
            )
        if complete_count:
            self._write_tokens(self._pending[:complete_count])
            del self._pending[:complete_count]

    def _open_shard(self) -> None:
        remaining = (
            self.shard_token_capacity
            if self.exact_token_limit is None
            else min(
                self.shard_token_capacity,
                self.exact_token_limit - self.token_count,
            )
        )
        if remaining <= 0:
            return
        self._shard_path = self.root / f"shard-{self._shard_index:05d}.bin"
        self._shard_map = np.memmap(
            self._shard_path,
            mode="w+",
            dtype=np.uint32,
            shape=(remaining,),
        )
        self._shard_offset = 0

    def _write_tokens(self, values: Sequence[int]) -> None:
        offset = 0
        while offset < len(values):
            if self._shard_map is None:
                self._open_shard()
            if self._shard_map is None:
                raise PackedCorpusError("Packed corpus received more than its token limit")
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
        self.discarded_trailing_tokens = len(self._pending)
        if self.exact_token_limit is not None and self.token_count != self.exact_token_limit:
            raise PackedCorpusError(
                f"Insufficient source data for {self.role}: expected exactly "
                f"{self.exact_token_limit} packed tokens, found {self.token_count}"
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


def prepare_packed_corpus(
    documents: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    output_dir: str | Path,
    *,
    tokenizer_name: str,
    tokenizer_revision: str,
    source_dataset: str = "HuggingFaceFW/fineweb",
    source_config: str = "sample-10BT",
    source_split: str = "train",
    source_fingerprint: str,
    text_column: str = "text",
    data_seed: int = DEFAULT_DATA_SEED,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    training_token_budget: int = DEFAULT_TRAINING_TOKEN_BUDGET,
    shard_token_capacity: int = 256 * 1024 * 1024,
    tokenizer_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Consume one seeded FineWeb stream and atomically install a packed corpus."""

    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists():
        raise PackedCorpusError(f"Prepared corpus directory already exists: {output_path}")
    if int(data_seed) != DEFAULT_DATA_SEED:
        raise PackedCorpusError("Unique-token production corpora require data_seed=42")
    if int(context_length) <= 1:
        raise PackedCorpusError("context_length must exceed one token")
    if int(training_token_budget) <= 0 or int(training_token_budget) % int(context_length):
        raise PackedCorpusError(
            "training_token_budget must be positive and divisible by context_length"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.preparing-", dir=output_path.parent)
    )
    try:
        reserved, training_documents = reserve_source_documents(
            documents,
            source_fingerprint=source_fingerprint,
        )
        token_identity = tokenizer_identity(
            tokenizer,
            name=tokenizer_name,
            revision=tokenizer_revision,
            tokenizer_manifest=tokenizer_manifest,
        )
        eos_token_id = int(token_identity["eos_token_id"])
        role_manifests: dict[str, dict[str, Any]] = {}
        reserved_identity_sets: dict[str, set[str]] = {}

        for role, selected in reserved.items():
            writer = _RoleShardWriter(
                temporary_root,
                role,
                context_length=context_length,
                shard_token_capacity=shard_token_capacity,
                exact_token_limit=None,
            )
            identities: list[dict[str, Any]] = []
            for identity, document in selected:
                if text_column not in document:
                    raise PackedCorpusError(
                        f"Source document is missing text column {text_column!r}"
                    )
                identities.append(identity)
                writer.add_document(
                    _tokenize_document(tokenizer, str(document[text_column])),
                    eos_token_id=eos_token_id,
                )
            role_manifest = writer.finish()
            role_manifest.update(
                ordered_source_document_identities=identities,
                source_document_identity_hash=stable_hash(identities),
                example_count=len(identities),
                ordered_example_identities=identities,
                example_identity_hash=stable_hash(identities),
            )
            reserved_identity_sets[role] = {
                stable_hash(identity) for identity in identities
            }
            role_manifests[role] = role_manifest

        training_writer = _RoleShardWriter(
            temporary_root,
            "optimizer_training",
            context_length=context_length,
            shard_token_capacity=shard_token_capacity,
            exact_token_limit=training_token_budget,
        )
        training_identity_digest = hashlib.sha256()
        training_document_count = 0
        for source_offset, document in enumerate(training_documents, start=sum(RESERVED_ROLE_COUNTS.values())):
            if text_column not in document:
                raise PackedCorpusError(
                    f"Source document is missing text column {text_column!r}"
                )
            identity = _document_identity(
                document,
                source_index=source_offset,
                source_fingerprint=source_fingerprint,
            )
            identity_hash = stable_hash(identity)
            if any(identity_hash in identities for identities in reserved_identity_sets.values()):
                raise PackedCorpusError("A reserved source document reappeared in training")
            training_identity_digest.update(identity_hash.encode("ascii"))
            training_document_count += 1
            training_writer.add_document(
                _tokenize_document(tokenizer, str(document[text_column])),
                eos_token_id=eos_token_id,
            )
            if training_writer.full:
                break
        training_manifest = training_writer.finish()
        training_manifest.update(
            source_document_count=training_document_count,
            source_document_identity_stream_hash=training_identity_digest.hexdigest(),
            example_count=training_manifest["sequence_count"],
        )
        role_manifests["optimizer_training"] = training_manifest

        role_hashes: dict[str, str] = {}
        for role in ALL_CORPUS_ROLES:
            payload = role_manifests[role]
            payload["manifest_hash"] = stable_hash(payload)
            role_hashes[role] = payload["manifest_hash"]

        pairwise_intersections: dict[str, int] = {}
        reserved_roles = tuple(RESERVED_ROLE_COUNTS)
        for left_index, left in enumerate(reserved_roles):
            for right in reserved_roles[left_index + 1 :]:
                count = len(reserved_identity_sets[left] & reserved_identity_sets[right])
                pairwise_intersections[f"{left}__{right}"] = count
                if count:
                    raise PackedCorpusError(f"Reserved document overlap: {left}, {right}")

        manifest = {
            "schema_version": PACKED_CORPUS_SCHEMA_VERSION,
            "packing_version": PACKING_VERSION,
            "permutation_version": PERMUTATION_VERSION,
            "data_seed": int(data_seed),
            "context_length": int(context_length),
            "dtype": "uint32",
            "source": {
                "dataset_name": source_dataset,
                "dataset_config_name": source_config,
                "split": source_split,
                "fingerprint": source_fingerprint,
            },
            "tokenizer": token_identity,
            "training_token_budget": int(training_token_budget),
            "role_selection_order": list(RESERVED_ROLE_COUNTS),
            "reserved_role_counts": dict(RESERVED_ROLE_COUNTS),
            "reserved_pairwise_intersection_counts": pairwise_intersections,
            "role_manifest_hashes": role_hashes,
            "roles": role_manifests,
        }
        manifest["corpus_hash"] = stable_hash(manifest)
        manifest_path = temporary_root / "corpus_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_path.chmod(0o444)
        os.replace(temporary_root, output_path)
        return manifest
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
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
    if manifest.get("schema_version") != PACKED_CORPUS_SCHEMA_VERSION:
        raise PackedCorpusError("Prepared corpus schema version mismatch")
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
    if manifest.get("permutation_version") != PERMUTATION_VERSION:
        raise PackedCorpusError("Prepared corpus permutation version mismatch")
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
    return manifest


def audit_packed_corpus(
    prepared_corpus_dir: str | Path,
    *,
    required_training_tokens: int = DEFAULT_TRAINING_TOKEN_BUDGET,
    prepared_tokenizer_dir: str | Path | None = None,
    required_vocab_size: int | None = None,
) -> dict[str, Any]:
    manifest = load_corpus_manifest(prepared_corpus_dir, verify_shards=True)
    training = manifest["roles"]["optimizer_training"]
    if int(training["token_count"]) != int(required_training_tokens):
        raise PackedCorpusError(
            f"Training token audit expected {required_training_tokens}, "
            f"found {training['token_count']}"
        )
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
    """One deterministic global permutation, partitioned once without padding."""

    dataset_size: int
    batch_size_per_rank: int
    rank: int
    world_size: int
    data_seed: int = DEFAULT_DATA_SEED
    cursor: int = 0

    def __post_init__(self) -> None:
        self.dataset_size = int(self.dataset_size)
        self.batch_size_per_rank = int(self.batch_size_per_rank)
        self.rank = int(self.rank)
        self.world_size = int(self.world_size)
        self.data_seed = int(self.data_seed)
        self.cursor = int(self.cursor)
        if self.dataset_size <= 0:
            raise PackedCorpusError("No-padding sampler requires a nonempty dataset")
        if self.batch_size_per_rank <= 0:
            raise PackedCorpusError("batch_size_per_rank must be positive")
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise PackedCorpusError("Invalid distributed rank topology")
        if not 0 <= self.cursor <= self.dataset_size:
            raise PackedCorpusError("Sampler cursor is outside the permutation")
        self._permutation = deterministic_permutation(
            self.dataset_size, data_seed=self.data_seed
        )
        self.last_yielded_cursor = self.cursor

    @property
    def global_batch_size(self) -> int:
        return self.batch_size_per_rank * self.world_size

    @property
    def permutation_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(PERMUTATION_VERSION.encode("ascii"))
        digest.update(str(self.data_seed).encode("ascii"))
        digest.update(self._permutation.astype("<u8", copy=False).tobytes())
        return digest.hexdigest()

    def __iter__(self):
        cursor = self.cursor
        while cursor < self.dataset_size:
            end = min(self.dataset_size, cursor + self.global_batch_size)
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
        remaining = self.dataset_size - self.cursor
        return (remaining + self.global_batch_size - 1) // self.global_batch_size

    def set_cursor(self, cursor: int) -> None:
        cursor = int(cursor)
        if not 0 <= cursor <= self.dataset_size:
            raise PackedCorpusError("Sampler cursor is outside the permutation")
        if cursor != self.dataset_size and cursor % self.global_batch_size:
            raise PackedCorpusError("Sampler cursor must identify a global-batch boundary")
        self.cursor = cursor
        self.last_yielded_cursor = cursor

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "permutation_version": PERMUTATION_VERSION,
            "data_seed": self.data_seed,
            "dataset_size": self.dataset_size,
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
    "DEFAULT_TRAINING_TOKEN_BUDGET",
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
    "prepare_packed_corpus",
    "reserve_source_documents",
    "sha256_file",
]
