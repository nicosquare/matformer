"""Immutable FineWeb SentencePiece tokenizer preparation and verification."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from src.training.packed_corpus import RESERVED_ROLE_COUNTS, sha256_file
from src.utils.reproducibility import stable_hash


TOKENIZER_MANIFEST_SCHEMA_VERSION = 1
TOKENIZER_TRAINING_VERSION = "fineweb_sentencepiece_bpe_v1"
DEFAULT_TOKENIZER_DOCUMENT_COUNT = 5_000_000
DEFAULT_MAX_CHUNK_BYTES = 4_096
DEFAULT_VOCAB_SIZE = 256_000
SPECIAL_TOKEN_IDS = {"unk": 0, "bos": 1, "eos": 2, "pad": 3}
SPECIAL_TOKEN_PIECES = {
    "unk": "<unk>",
    "bos": "<s>",
    "eos": "</s>",
    "pad": "<pad>",
}


class FineWebTokenizerError(ValueError):
    """Raised when tokenizer artifacts or their provenance are invalid."""


def sentencepiece_trainer_options(
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> dict[str, Any]:
    """Return the immutable SentencePiece training contract recorded in manifests."""

    return {
        "model_type": "bpe",
        "vocab_size": int(vocab_size),
        "unk_id": SPECIAL_TOKEN_IDS["unk"],
        "bos_id": SPECIAL_TOKEN_IDS["bos"],
        "eos_id": SPECIAL_TOKEN_IDS["eos"],
        "pad_id": SPECIAL_TOKEN_IDS["pad"],
        "unk_piece": SPECIAL_TOKEN_PIECES["unk"],
        "bos_piece": SPECIAL_TOKEN_PIECES["bos"],
        "eos_piece": SPECIAL_TOKEN_PIECES["eos"],
        "pad_piece": SPECIAL_TOKEN_PIECES["pad"],
        "byte_fallback": True,
        "character_coverage": 0.9995,
        "normalization_rule_name": "nmt_nfkc",
        "hard_vocab_limit": True,
        "shuffle_input_sentence": False,
        "input_sentence_size": 0,
        "num_threads": 1,
    }


def utf8_safe_chunks(text: str, max_bytes: int = DEFAULT_MAX_CHUNK_BYTES) -> Iterator[str]:
    """Yield nonempty character-aligned chunks whose UTF-8 encoding fits the limit."""

    if isinstance(max_bytes, bool) or int(max_bytes) <= 0:
        raise FineWebTokenizerError("max_bytes must be a positive integer")
    limit = int(max_bytes)
    characters: list[str] = []
    byte_count = 0
    for character in str(text):
        encoded = character.encode("utf-8")
        if len(encoded) > limit:
            raise FineWebTokenizerError(
                "max_bytes is too small to contain one UTF-8 code point"
            )
        if characters and byte_count + len(encoded) > limit:
            yield "".join(characters)
            characters = []
            byte_count = 0
        characters.append(character)
        byte_count += len(encoded)
    if characters:
        yield "".join(characters)


def select_tokenizer_training_documents(
    documents: Iterable[Mapping[str, Any]],
    *,
    document_count: int = DEFAULT_TOKENIZER_DOCUMENT_COUNT,
    reserved_document_count: int = sum(RESERVED_ROLE_COUNTS.values()),
) -> Iterator[tuple[int, Mapping[str, Any]]]:
    """Skip all reserved roles and yield exactly the following training-role rows."""

    requested = int(document_count)
    reserved = int(reserved_document_count)
    if requested <= 0 or reserved < 0:
        raise FineWebTokenizerError("document counts must be positive/nonnegative")
    iterator = iter(documents)
    for _ in range(reserved):
        try:
            next(iterator)
        except StopIteration as error:
            raise FineWebTokenizerError(
                "FineWeb source ended while skipping reserved documents"
            ) from error
    for training_offset in range(requested):
        try:
            document = next(iterator)
        except StopIteration as error:
            raise FineWebTokenizerError(
                f"FineWeb source ended before {requested} tokenizer documents"
            ) from error
        yield reserved + training_offset, document


def tokenizer_source_identity(
    document: Mapping[str, Any],
    *,
    source_index: int,
    source_fingerprint: str,
) -> dict[str, Any]:
    return {
        "source_dataset_fingerprint": str(source_fingerprint),
        "source_row_identity": document.get(
            "id", document.get("source_row_identity", int(source_index))
        ),
    }


def _manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    return payload


def load_tokenizer_manifest(
    tokenizer_dir: str | Path,
    *,
    verify_files: bool = True,
) -> dict[str, Any]:
    root = Path(tokenizer_dir).expanduser().resolve()
    manifest_path = root / "tokenizer_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FineWebTokenizerError(
            f"Invalid tokenizer manifest: {manifest_path}"
        ) from error
    if manifest.get("schema_version") != TOKENIZER_MANIFEST_SCHEMA_VERSION:
        raise FineWebTokenizerError("Tokenizer manifest schema version mismatch")
    if manifest.get("manifest_hash") != stable_hash(_manifest_payload(manifest)):
        raise FineWebTokenizerError("Tokenizer manifest hash mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise FineWebTokenizerError("Tokenizer manifest file checksums are missing")
    model_file = manifest.get("sentencepiece_model_file")
    if not isinstance(model_file, str) or model_file not in files:
        raise FineWebTokenizerError("Tokenizer manifest model file is missing")
    for relative_path, expected_sha256 in files.items():
        path = root / str(relative_path)
        if not path.is_file():
            raise FineWebTokenizerError(f"Tokenizer file is missing: {path}")
        if verify_files and sha256_file(path) != expected_sha256:
            raise FineWebTokenizerError(f"Tokenizer file checksum mismatch: {path}")
    if manifest.get("sentencepiece_model_sha256") != files[model_file]:
        raise FineWebTokenizerError("Tokenizer model checksum provenance mismatch")
    if int(manifest.get("vocab_size", -1)) <= 0:
        raise FineWebTokenizerError("Tokenizer vocabulary size is invalid")
    if manifest.get("special_token_ids") != SPECIAL_TOKEN_IDS:
        raise FineWebTokenizerError("Tokenizer special IDs do not match the contract")
    return manifest


def load_existing_tokenizer_if_matching(
    tokenizer_dir: str | Path,
    *,
    source_dataset: str = "HuggingFaceFW/fineweb",
    source_config: str = "sample-10BT",
    source_split: str = "train",
    text_column: str = "text",
    data_seed: int = 42,
    shuffle_buffer_size: int = 100_000,
    document_count: int = DEFAULT_TOKENIZER_DOCUMENT_COUNT,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> dict[str, Any] | None:
    """Load a completed exact-match artifact, or return ``None`` when absent.

    An existing directory is never treated as resumable scratch space. It must
    be a fully checksum-valid tokenizer with the same preparation inputs.
    """

    root = Path(tokenizer_dir).expanduser().resolve()
    if not root.exists():
        return None
    # Reject a changed request from manifest metadata before spending time on
    # checksums. An accepted reuse is always followed by full verification.
    manifest = load_tokenizer_manifest(root, verify_files=False)
    expected = {
        "training_version": TOKENIZER_TRAINING_VERSION,
        "tokenizer_name": "fineweb_sentencepiece_bpe_256k",
        "sentencepiece_version": "0.2.1",
        "sentencepiece_options": sentencepiece_trainer_options(vocab_size),
        "dataset.name": str(source_dataset),
        "dataset.config_name": str(source_config),
        "dataset.split": str(source_split),
        "dataset.text_column": str(text_column),
        "shuffle.seed": int(data_seed),
        "shuffle.buffer_size": int(shuffle_buffer_size),
        "reserved_document_count": sum(RESERVED_ROLE_COUNTS.values()),
        "training_document_count": int(document_count),
        "max_chunk_bytes": int(max_chunk_bytes),
        "vocab_size": int(vocab_size),
        "special_token_ids": dict(SPECIAL_TOKEN_IDS),
        "special_token_pieces": dict(SPECIAL_TOKEN_PIECES),
        "subword_sampling": False,
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
        raise FineWebTokenizerError(
            "Existing tokenizer does not match the requested preparation: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if int(manifest.get("training_chunk_count", 0)) <= 0:
        raise FineWebTokenizerError(
            "Existing tokenizer training_chunk_count must be positive"
        )
    return load_tokenizer_manifest(root, verify_files=True)


def _write_training_input(
    documents: Iterable[Mapping[str, Any]],
    output_path: Path,
    *,
    source_fingerprint: str,
    text_column: str,
    document_count: int,
    max_chunk_bytes: int,
) -> dict[str, Any]:
    source_digest = hashlib.sha256()
    raw_input_digest = hashlib.sha256()
    chunk_count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as destination:
        for source_index, document in select_tokenizer_training_documents(
            documents,
            document_count=document_count,
        ):
            if text_column not in document:
                raise FineWebTokenizerError(
                    f"Source document is missing text column {text_column!r}"
                )
            identity = tokenizer_source_identity(
                document,
                source_index=source_index,
                source_fingerprint=source_fingerprint,
            )
            source_digest.update(stable_hash(identity).encode("ascii"))
            for chunk in utf8_safe_chunks(
                str(document[text_column]), max_bytes=max_chunk_bytes
            ):
                encoded = chunk.encode("utf-8")
                raw_input_digest.update(len(encoded).to_bytes(8, "big"))
                raw_input_digest.update(encoded)
                destination.write(chunk.replace("\n", " ").replace("\r", " "))
                destination.write("\n")
                chunk_count += 1
    if chunk_count <= 0:
        raise FineWebTokenizerError("Tokenizer training selection produced no text chunks")
    return {
        "document_count": int(document_count),
        "chunk_count": chunk_count,
        "source_identity_stream_hash": source_digest.hexdigest(),
        "raw_input_hash": raw_input_digest.hexdigest(),
    }


def _normalized_input_hash(training_input: Path, model_file: Path) -> str:
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=str(model_file))
    digest = hashlib.sha256()
    with training_input.open("r", encoding="utf-8") as source:
        for line in source:
            normalized = processor.normalize(line.rstrip("\n"))
            encoded = normalized.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _auto_tokenizer_from_sentencepiece(model_file: Path, processor):
    """Convert SentencePiece BPE to tokenizer.json, including byte fallback.

    Transformers 5's generic SentencePiece BPE converter currently calls its
    merge extractor with the legacy argument shape.  Keeping the tiny corrected
    adapter here makes the produced directory version-independent and lets
    ``AutoTokenizer`` load it without custom code.
    """

    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import LlamaConverter, SpmConverter
    from transformers.tokenization_utils_base import generate_merges

    class SentencePieceAdapter:
        vocab_file = str(model_file)
        legacy = False
        add_prefix_space = True

        def convert_ids_to_tokens(self, token_id):
            return processor.id_to_piece(int(token_id))

    class FineWebLlamaConverter(LlamaConverter):
        def tokenizer(self, proto):
            vocab_scores = self.vocab(proto)
            vocabulary = {
                piece: token_id
                for token_id, (piece, _score) in enumerate(vocab_scores)
            }
            return Tokenizer(
                BPE(
                    vocabulary,
                    generate_merges(vocabulary),
                    unk_token=proto.trainer_spec.unk_piece,
                    fuse_unk=True,
                    byte_fallback=True,
                    dropout=None,
                )
            )

        def normalizer(self, proto):
            return SpmConverter.normalizer(self, proto)

    return PreTrainedTokenizerFast(
        tokenizer_object=FineWebLlamaConverter(SentencePieceAdapter()).converted(),
        unk_token=SPECIAL_TOKEN_PIECES["unk"],
        bos_token=SPECIAL_TOKEN_PIECES["bos"],
        eos_token=SPECIAL_TOKEN_PIECES["eos"],
        pad_token=SPECIAL_TOKEN_PIECES["pad"],
    )


def train_fineweb_tokenizer(
    documents: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    source_dataset: str = "HuggingFaceFW/fineweb",
    source_config: str = "sample-10BT",
    source_split: str = "train",
    source_fingerprint: str,
    text_column: str = "text",
    data_seed: int = 42,
    shuffle_buffer_size: int = 100_000,
    document_count: int = DEFAULT_TOKENIZER_DOCUMENT_COUNT,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> dict[str, Any]:
    """Train and atomically install an AutoTokenizer-compatible tokenizer."""

    if int(data_seed) != 42:
        raise FineWebTokenizerError("FineWeb tokenizer training requires data_seed=42")
    if int(vocab_size) != DEFAULT_VOCAB_SIZE:
        raise FineWebTokenizerError(
            f"FineWeb tokenizer vocabulary must be exactly {DEFAULT_VOCAB_SIZE}"
        )
    try:
        import sentencepiece as spm
    except ImportError as error:
        raise FineWebTokenizerError(
            "Tokenizer training requires sentencepiece==0.2.1 and transformers"
        ) from error
    if getattr(spm, "__version__", None) != "0.2.1":
        raise FineWebTokenizerError("Tokenizer training requires sentencepiece==0.2.1")

    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists():
        raise FineWebTokenizerError(f"Tokenizer directory already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.training-", dir=output_path.parent)
    )
    try:
        input_path = staging / "sentencepiece-input.txt"
        counts = _write_training_input(
            documents,
            input_path,
            source_fingerprint=source_fingerprint,
            text_column=text_column,
            document_count=int(document_count),
            max_chunk_bytes=int(max_chunk_bytes),
        )
        model_prefix = staging / "tokenizer"
        trainer_options = sentencepiece_trainer_options(vocab_size)
        spm.SentencePieceTrainer.train(
            input=str(input_path),
            model_prefix=str(model_prefix),
            **trainer_options,
        )
        model_file = staging / "tokenizer.model"
        processor = spm.SentencePieceProcessor(model_file=str(model_file))
        if int(processor.vocab_size()) != int(vocab_size):
            raise FineWebTokenizerError("SentencePiece did not produce the exact vocabulary")
        for name, expected_id in SPECIAL_TOKEN_IDS.items():
            actual = int(getattr(processor, f"{name}_id")())
            if actual != expected_id:
                raise FineWebTokenizerError(
                    f"SentencePiece special ID mismatch for {name}: {actual}"
                )
        normalized_hash = _normalized_input_hash(input_path, model_file)
        tokenizer = _auto_tokenizer_from_sentencepiece(model_file, processor)
        tokenizer.save_pretrained(staging)
        input_path.unlink()
        files = {
            path.name: sha256_file(path)
            for path in sorted(staging.iterdir())
            if path.is_file() and path.name != "tokenizer_manifest.json"
        }
        manifest = {
            "schema_version": TOKENIZER_MANIFEST_SCHEMA_VERSION,
            "training_version": TOKENIZER_TRAINING_VERSION,
            "tokenizer_name": "fineweb_sentencepiece_bpe_256k",
            "sentencepiece_version": spm.__version__,
            "sentencepiece_options": trainer_options,
            "dataset": {
                "name": source_dataset,
                "config_name": source_config,
                "split": source_split,
                "fingerprint": source_fingerprint,
                "text_column": text_column,
            },
            "shuffle": {
                "seed": int(data_seed),
                "buffer_size": int(shuffle_buffer_size),
            },
            "reserved_document_count": sum(RESERVED_ROLE_COUNTS.values()),
            "training_document_count": counts["document_count"],
            "training_chunk_count": counts["chunk_count"],
            "max_chunk_bytes": int(max_chunk_bytes),
            "source_identity_stream_hash": counts["source_identity_stream_hash"],
            "raw_input_hash": counts["raw_input_hash"],
            "normalized_input_hash": normalized_hash,
            "vocab_size": int(processor.vocab_size()),
            "special_token_ids": dict(SPECIAL_TOKEN_IDS),
            "special_token_pieces": dict(SPECIAL_TOKEN_PIECES),
            "subword_sampling": False,
            "sentencepiece_model_file": "tokenizer.model",
            "sentencepiece_model_sha256": files["tokenizer.model"],
            "files": files,
        }
        manifest["manifest_hash"] = stable_hash(manifest)
        (staging / "tokenizer_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for path in staging.iterdir():
            if path.is_file():
                path.chmod(0o444)
        os.replace(staging, output_path)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "DEFAULT_MAX_CHUNK_BYTES",
    "DEFAULT_TOKENIZER_DOCUMENT_COUNT",
    "DEFAULT_VOCAB_SIZE",
    "FineWebTokenizerError",
    "SPECIAL_TOKEN_IDS",
    "SPECIAL_TOKEN_PIECES",
    "TOKENIZER_MANIFEST_SCHEMA_VERSION",
    "TOKENIZER_TRAINING_VERSION",
    "load_existing_tokenizer_if_matching",
    "load_tokenizer_manifest",
    "select_tokenizer_training_documents",
    "sentencepiece_trainer_options",
    "train_fineweb_tokenizer",
    "utf8_safe_chunks",
]
