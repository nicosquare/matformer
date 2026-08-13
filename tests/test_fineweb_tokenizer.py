from __future__ import annotations

import json

import pytest

from src.training.fineweb_tokenizer import (
    FineWebTokenizerError,
    SPECIAL_TOKEN_IDS,
    load_tokenizer_manifest,
    select_tokenizer_training_documents,
    utf8_safe_chunks,
)
from src.training.packed_corpus import sha256_file
from src.utils.reproducibility import stable_hash


def test_tokenizer_selection_excludes_all_reserved_roles_and_takes_exact_count():
    documents = ({"id": f"doc-{index}", "text": str(index)} for index in range(12))
    selected = list(
        select_tokenizer_training_documents(
            documents,
            reserved_document_count=4,
            document_count=5,
        )
    )
    assert [index for index, _ in selected] == [4, 5, 6, 7, 8]
    assert [document["id"] for _, document in selected] == [
        "doc-4", "doc-5", "doc-6", "doc-7", "doc-8"
    ]


def test_utf8_chunks_are_nonempty_lossless_and_never_split_code_points():
    text = "ab🙂café漢字"
    chunks = list(utf8_safe_chunks(text, max_bytes=5))
    assert "".join(chunks) == text
    assert all(chunk and len(chunk.encode("utf-8")) <= 5 for chunk in chunks)
    assert chunks == list(utf8_safe_chunks(text, max_bytes=5))


def test_tokenizer_manifest_hash_and_file_checksums_are_authoritative(tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    model = tokenizer_dir / "tokenizer.model"
    config = tokenizer_dir / "tokenizer_config.json"
    model.write_bytes(b"model")
    config.write_text("{}\n", encoding="utf-8")
    files = {path.name: sha256_file(path) for path in (model, config)}
    manifest = {
        "schema_version": 1,
        "vocab_size": 256_000,
        "special_token_ids": SPECIAL_TOKEN_IDS,
        "sentencepiece_model_file": model.name,
        "sentencepiece_model_sha256": files[model.name],
        "files": files,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    (tokenizer_dir / "tokenizer_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert load_tokenizer_manifest(tokenizer_dir)["manifest_hash"] == manifest[
        "manifest_hash"
    ]

    model.chmod(0o644)
    model.write_bytes(b"changed")
    with pytest.raises(FineWebTokenizerError, match="checksum mismatch"):
        load_tokenizer_manifest(tokenizer_dir)


def test_local_sentencepiece_directory_loads_through_auto_tokenizer(tmp_path):
    import sentencepiece as spm
    from transformers import AutoTokenizer

    from src.training.fineweb_tokenizer import _auto_tokenizer_from_sentencepiece

    input_path = tmp_path / "input.txt"
    input_path.write_text(
        "hello fineweb tokenizer\nbytes and unicode 🙂 café\n" * 100,
        encoding="utf-8",
    )
    prefix = tmp_path / "sp"
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=320,
        byte_fallback=True,
        hard_vocab_limit=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        normalization_rule_name="nmt_nfkc",
        character_coverage=0.9995,
        shuffle_input_sentence=False,
    )
    tokenizer_dir = tmp_path / "tokenizer"
    processor = spm.SentencePieceProcessor(
        model_file=str(prefix.with_suffix(".model"))
    )

    tokenizer = _auto_tokenizer_from_sentencepiece(
        prefix.with_suffix(".model"), processor
    )
    tokenizer.save_pretrained(tokenizer_dir)
    loaded = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    assert loaded.vocab_size == 320
    assert [loaded.unk_token_id, loaded.bos_token_id, loaded.eos_token_id, loaded.pad_token_id] == [
        0, 1, 2, 3
    ]
