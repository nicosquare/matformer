from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.training.packed_corpus import (
    PackedCorpusError,
    audit_packed_corpus,
    load_existing_corpus_if_matching,
    prepare_packed_corpus,
    sha256_file,
)
from src.training.tinystories_instruct import (
    TINYSTORIES_INSTRUCT_ASSEMBLY_PARSER_VERSION,
    TINYSTORIES_INSTRUCT_CONTENT_POLICY,
    TINYSTORIES_INSTRUCT_DATASET_NAME,
    TINYSTORIES_INSTRUCT_DATASET_REVISION,
    TINYSTORIES_INSTRUCT_DELIMITER_POLICY,
    TINYSTORIES_INSTRUCT_ROLE_COUNTS,
    TINYSTORIES_INSTRUCT_TOKENIZER_NAME,
    TinyStoriesInstructError,
    TinyStoriesInstructWarning,
    corpus_source_contract,
    dataset_split_fingerprint,
    iter_dataset_instruction_records,
    load_existing_tinystories_instruct_tokenizer_if_matching,
    role_ordered_instruction_records,
    tokenizer_source_fingerprint,
    tokenizer_training_instruction_records,
)
from src.utils.reproducibility import stable_hash


class PhysicalRowDataset(list):
    column_names = ["text"]

    def __init__(self, rows, *, fingerprint: str = "native-fingerprint"):
        super().__init__({"text": value} for value in rows)
        self._fingerprint = fingerprint


class InstructTokenizer:
    eos_token_id = 2
    vocab_size = 2_048

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": [10 + (len(str(text)) % 20)]}


def _tokenizer_artifact(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "tokenizer"
    root.mkdir()
    model = root / "tokenizer.model"
    config = root / "tokenizer_config.json"
    model.write_bytes(b"tinystories-instruct-tokenizer")
    config.write_text("{}\n", encoding="utf-8")
    files = {path.name: sha256_file(path) for path in (model, config)}
    manifest = {
        "schema_version": 1,
        "training_version": "tinystories_instruct_sentencepiece_bpe_v1",
        "tokenizer_name": TINYSTORIES_INSTRUCT_TOKENIZER_NAME,
        "vocab_size": 2_048,
        "special_token_ids": {"unk": 0, "bos": 1, "eos": 2, "pad": 3},
        "sentencepiece_model_file": model.name,
        "sentencepiece_model_sha256": files[model.name],
        "files": files,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    (root / "tokenizer_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return root, manifest


def _physical_rows(prefix: str, count: int) -> PhysicalRowDataset:
    rows = []
    for index in range(count):
        rows.extend(
            [
                f"Features: {prefix}-{index}",
                f"Words: word-{index}",
                f"Summary: summary-{index}",
                f"Story: story-{index}",
                "<|endoftext|>",
            ]
        )
    return PhysicalRowDataset(rows, fingerprint=f"{prefix}-native")


def test_instruction_record_assembly_preserves_content_and_physical_identity():
    dataset = PhysicalRowDataset(
        [
            "Features: Dialogue",
            "",
            "Words: oak, gloomy",
            "Summary: A cold day.",
            "Story: First paragraph.",
            "Second paragraph.",
            "<|endoftext|>",
            "Features: Twist",
            "Story: The end.",
            "<|endoftext|>",
            "",
        ]
    )

    records = list(iter_dataset_instruction_records(dataset, split="train"))

    assert [record["id"] for record in records] == ["train:0-6", "train:7-9"]
    assert records[0]["source_first_physical_row_index"] == 0
    assert records[0]["source_last_physical_row_index"] == 6
    assert records[0]["text"] == (
        "Features: Dialogue\n\nWords: oak, gloomy\nSummary: A cold day.\n"
        "Story: First paragraph.\nSecond paragraph."
    )
    assert "<|endoftext|>" not in records[0]["text"]
    assert records[1]["assembled_record_index"] == 1


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"other": "missing"}], "has no text column"),
        ([{"text": 123}], "text is not a string"),
        ([{"text": "<|endoftext|>"}], "empty instruction record"),
        (
            [{"text": "Features: Dialogue"}],
            "nonblank unterminated instruction record",
        ),
        (
            [{"text": " <|endoftext|> "}],
            "nonblank unterminated instruction record",
        ),
    ],
)
def test_instruction_record_assembly_rejects_malformed_rows(rows, message):
    with pytest.raises(TinyStoriesInstructError, match=message):
        list(iter_dataset_instruction_records(rows, split="train"))


def test_instruction_record_assembly_allows_only_blank_unterminated_eof():
    dataset = PhysicalRowDataset(
        ["Story: complete", "<|endoftext|>", "", "   "]
    )
    records = list(iter_dataset_instruction_records(dataset, split="validation"))
    assert [record["id"] for record in records] == ["validation:0-1"]


def test_only_known_pinned_truncated_train_tail_is_dropped(
    monkeypatch,
):
    import src.training.tinystories_instruct as instruct

    truncated_tail = [
        "Words: read, tower, glad",
        "Features: Dialogue, Foreshadowing",
        "Story: ",
        "",
        "Sam liked to read books.",
        "Sam was glad that he had b",
    ]
    rows = ["Story: complete", "<|endoftext|>", *truncated_tail]
    monkeypatch.setattr(
        instruct, "TINYSTORIES_INSTRUCT_KNOWN_TRUNCATED_TAIL_FIRST_ROW", 2
    )
    monkeypatch.setattr(
        instruct,
        "TINYSTORIES_INSTRUCT_KNOWN_TRUNCATED_TAIL_LAST_ROW",
        len(rows) - 1,
    )
    monkeypatch.setattr(
        instruct,
        "TINYSTORIES_INSTRUCT_KNOWN_TRUNCATED_TAIL_HASH",
        stable_hash(truncated_tail),
    )

    with pytest.warns(TinyStoriesInstructWarning, match="dropping known"):
        records = list(
            iter_dataset_instruction_records(
                PhysicalRowDataset(rows), split="train"
            )
        )
    assert [record["text"] for record in records] == ["Story: complete"]

    with pytest.raises(
        TinyStoriesInstructError, match="nonblank unterminated"
    ):
        list(
            iter_dataset_instruction_records(
                PhysicalRowDataset([*rows[:-1], "different truncation"]),
                split="train",
            )
        )


def test_role_assignment_counts_assembled_records_not_physical_rows():
    train = _physical_rows("train", 132)
    validation = _physical_rows("validation", 640)

    tokenizer_records = list(
        tokenizer_training_instruction_records(train, record_count=3)
    )
    assert [record["assembled_record_index"] for record in tokenizer_records] == [
        128,
        129,
        130,
    ]
    assert [record["id"] for record in tokenizer_records] == [
        "train:640-644",
        "train:645-649",
        "train:650-654",
    ]

    ordered = iter(role_ordered_instruction_records(train, validation))
    ordinary_ids = [next(ordered)["id"] for _ in range(128)]
    controller_ids = [next(ordered)["id"] for _ in range(128)]
    final_ids = [next(ordered)["id"] for _ in range(512)]
    optimizer_first = next(ordered)["id"]

    assert ordinary_ids[0] == "validation:0-4"
    assert ordinary_ids[-1] == "validation:635-639"
    assert controller_ids[0] == "train:0-4"
    assert controller_ids[-1] == "train:635-639"
    assert final_ids[0] == "validation:640-644"
    assert final_ids[-1] == "validation:3195-3199"
    assert optimizer_first == "train:640-644"
    assert len(set(ordinary_ids + controller_ids + final_ids)) == 768


def test_split_and_tokenizer_fingerprints_cover_complete_assembly_contract():
    dataset = PhysicalRowDataset(
        ["Story: complete", "<|endoftext|>"], fingerprint="underlying"
    )
    base = dataset_split_fingerprint(dataset, split="train")

    assert base != dataset_split_fingerprint(
        dataset,
        split="train",
        dataset_revision="different-revision",
    )
    assert base != dataset_split_fingerprint(
        dataset,
        split="train",
        assembly_parser_version="different-parser",
    )
    assert base != dataset_split_fingerprint(
        dataset,
        split="train",
        delimiter_policy="different-delimiter-policy",
    )
    assert base != dataset_split_fingerprint(
        dataset,
        split="train",
        content_policy="story-only",
    )

    tokenizer_base = tokenizer_source_fingerprint(base)
    assert tokenizer_base != tokenizer_source_fingerprint(
        base,
        dataset_revision="different-revision",
    )
    assert tokenizer_base != tokenizer_source_fingerprint(
        base,
        assembly_parser_version="different-parser",
    )
    assert tokenizer_base != tokenizer_source_fingerprint(
        base,
        content_policy="story-only",
    )


def test_corpus_source_contract_exposes_revision_split_and_parser_identity():
    source_fingerprint, roles = corpus_source_contract(
        train_split_fingerprint="train-fingerprint",
        validation_split_fingerprint="validation-fingerprint",
        train_row_count=1_000,
        validation_row_count=800,
    )

    assert source_fingerprint == stable_hash(roles)
    assert set(roles) == {
        "optimizer_training",
        "ordinary_validation",
        "controller",
        "final_holdout",
    }
    for provenance in roles.values():
        assert provenance["dataset_name"] == TINYSTORIES_INSTRUCT_DATASET_NAME
        assert provenance["dataset_revision"] == (
            TINYSTORIES_INSTRUCT_DATASET_REVISION
        )
        assert provenance["assembly_parser_version"] == (
            TINYSTORIES_INSTRUCT_ASSEMBLY_PARSER_VERSION
        )
        assert provenance["delimiter_policy"] == (
            TINYSTORIES_INSTRUCT_DELIMITER_POLICY
        )
        assert provenance["content_policy"] == TINYSTORIES_INSTRUCT_CONTENT_POLICY


def test_existing_tokenizer_rejects_changed_identity_contract(
    tmp_path, monkeypatch
):
    import src.training.tinystories_instruct as instruct

    tokenizer_dir, manifest = _tokenizer_artifact(tmp_path)
    manifest.update(
        {
            "dataset": {
                "name": TINYSTORIES_INSTRUCT_DATASET_NAME,
                "config_name": "default",
                "split": "train",
                "fingerprint": tokenizer_source_fingerprint("train-fingerprint"),
                "text_column": "text",
            },
            "reserved_document_count": 0,
            "training_document_count": 50_000,
            "max_chunk_bytes": 4_096,
        }
    )
    manifest["manifest_hash"] = stable_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    monkeypatch.setattr(instruct, "load_tokenizer_manifest", lambda *a, **k: manifest)

    assert load_existing_tinystories_instruct_tokenizer_if_matching(
        tokenizer_dir,
        train_split_fingerprint="train-fingerprint",
    ) == manifest
    with pytest.raises(
        TinyStoriesInstructError, match="does not match the request"
    ):
        load_existing_tinystories_instruct_tokenizer_if_matching(
            tokenizer_dir,
            train_split_fingerprint="train-fingerprint",
            content_policy="story-only",
        )


def test_instruct_corpus_resume_matches_uninterrupted_hashes(
    tmp_path, monkeypatch
):
    import src.training.packed_corpus as packed_corpus

    _, tokenizer_manifest = _tokenizer_artifact(tmp_path)
    train = _physical_rows("train", 132)
    validation = _physical_rows("validation", 640)
    source_fingerprint, role_sources = corpus_source_contract(
        train_split_fingerprint=dataset_split_fingerprint(train, split="train"),
        validation_split_fingerprint=dataset_split_fingerprint(
            validation, split="validation"
        ),
        train_row_count=len(train),
        validation_row_count=len(validation),
    )

    def prepare(name: str, progress_events: list[dict]):
        return prepare_packed_corpus(
            role_ordered_instruction_records(train, validation),
            InstructTokenizer(),
            tmp_path / name,
            corpus_name="tinystories-instruct-packed-full-v1",
            tokenizer_name=tokenizer_manifest["tokenizer_name"],
            tokenizer_revision=tokenizer_manifest["manifest_hash"],
            tokenizer_manifest=tokenizer_manifest,
            source_dataset=TINYSTORIES_INSTRUCT_DATASET_NAME,
            source_config="default",
            source_split="train+validation",
            source_fingerprint=source_fingerprint,
            context_length=4,
            shard_token_capacity=8,
            shuffle_buffer_size=0,
            reserved_role_counts=TINYSTORIES_INSTRUCT_ROLE_COUNTS,
            minimum_optimizer_document_count=3,
            role_source_provenance=role_sources,
            progress_callback=progress_events.append,
            progress_interval_seconds=60,
        )

    real_finish = packed_corpus._RoleShardWriter.finish
    interrupted = False

    def interrupt_optimizer_finish(writer):
        nonlocal interrupted
        result = real_finish(writer)
        if writer.role == "optimizer_training" and not interrupted:
            interrupted = True
            raise RuntimeError("interrupt instruct preparation")
        return result

    monkeypatch.setattr(
        packed_corpus._RoleShardWriter, "finish", interrupt_optimizer_finish
    )
    with pytest.raises(RuntimeError, match="interrupt instruct preparation"):
        prepare("resumed", [])
    monkeypatch.setattr(packed_corpus._RoleShardWriter, "finish", real_finish)

    resumed_events: list[dict] = []
    resumed = prepare("resumed", resumed_events)
    fresh = prepare("fresh", [])

    assert resumed == fresh
    assert resumed["corpus_hash"] == fresh["corpus_hash"]
    assert resumed["source"]["source_exhausted"] is True
    assert resumed["corpus_name"] == "tinystories-instruct-packed-full-v1"
    assert any(
        event["event"] == "resume_replay_completed" for event in resumed_events
    )

    audit = audit_packed_corpus(
        tmp_path / "fresh",
        required_vocab_size=2_048,
    )
    assert audit["corpus_name"] == "tinystories-instruct-packed-full-v1"
    assert audit["source_exhausted"] is True
    assert audit["source_termination"] == "source_exhausted"
    assert audit["reserved_role_counts"] == TINYSTORIES_INSTRUCT_ROLE_COUNTS
    assert audit["role_source_document_counts"] == {
        "ordinary_validation": 128,
        "controller": 128,
        "final_holdout": 512,
        "optimizer_training": 4,
    }
    assert all(
        count == 0
        for count in audit["reserved_pairwise_intersection_counts"].values()
    )

    exact_match_kwargs = {
        "corpus_name": "tinystories-instruct-packed-full-v1",
        "tokenizer_manifest": tokenizer_manifest,
        "tokenizer_name": tokenizer_manifest["tokenizer_name"],
        "tokenizer_revision": tokenizer_manifest["manifest_hash"],
        "source_dataset": TINYSTORIES_INSTRUCT_DATASET_NAME,
        "source_config": "default",
        "source_split": "train+validation",
        "context_length": 4,
        "shard_token_capacity": 8,
        "shuffle_buffer_size": 0,
        "reserved_role_counts": TINYSTORIES_INSTRUCT_ROLE_COUNTS,
        "minimum_optimizer_document_count": 3,
        "role_source_provenance": role_sources,
    }
    assert load_existing_corpus_if_matching(
        tmp_path / "fresh", **exact_match_kwargs
    ) == fresh

    with pytest.raises(PackedCorpusError, match="does not match"):
        load_existing_corpus_if_matching(
            tmp_path / "fresh",
            **{**exact_match_kwargs, "corpus_name": "different-corpus"},
        )

    changed_parser_sources = copy.deepcopy(role_sources)
    for provenance in changed_parser_sources.values():
        provenance["assembly_parser_version"] = "different-parser"
    with pytest.raises(PackedCorpusError, match="does not match"):
        load_existing_corpus_if_matching(
            tmp_path / "fresh",
            **{
                **exact_match_kwargs,
                "role_source_provenance": changed_parser_sources,
            },
        )

    changed_content_sources = copy.deepcopy(role_sources)
    for provenance in changed_content_sources.values():
        provenance["content_policy"] = "story-only"
    with pytest.raises(PackedCorpusError, match="does not match"):
        load_existing_corpus_if_matching(
            tmp_path / "fresh",
            **{
                **exact_match_kwargs,
                "role_source_provenance": changed_content_sources,
            },
        )

    with pytest.raises(PackedCorpusError, match="does not match"):
        load_existing_corpus_if_matching(
            tmp_path / "fresh",
            **{**exact_match_kwargs, "optimizer_token_limit": 8},
        )


def test_prepare_instruct_command_uses_pinned_dataset_and_reuses_artifacts(
    tmp_path, monkeypatch, capsys
):
    from scripts import prepare_tinystories_instruct as command

    train = _physical_rows("train", 1)
    validation = _physical_rows("validation", 1)
    load_calls = []
    corpus_load_calls = []

    monkeypatch.setattr(
        command,
        "load_dataset",
        lambda name, **kwargs: (
            load_calls.append((name, kwargs))
            or {"train": train, "validation": validation}
        ),
    )
    tokenizer_manifest = {
        "tokenizer_name": TINYSTORIES_INSTRUCT_TOKENIZER_NAME,
        "manifest_hash": "tokenizer-hash",
    }
    corpus_manifest = {
        "corpus_hash": "corpus-hash",
        "available_optimizer_token_count": 100_000_000,
        "available_optimizer_sequence_count": 781_250,
        "reserved_role_counts": TINYSTORIES_INSTRUCT_ROLE_COUNTS,
    }
    monkeypatch.setattr(
        command,
        "load_existing_tinystories_instruct_tokenizer_if_matching",
        lambda *args, **kwargs: tokenizer_manifest,
    )
    monkeypatch.setattr(
        command,
        "load_existing_corpus_if_matching",
        lambda *args, **kwargs: (
            corpus_load_calls.append(kwargs) or corpus_manifest
        ),
    )

    command.main(
        [
            "--tokenizer-dir",
            str(tmp_path / "tokenizer"),
            "--corpus-dir",
            str(tmp_path / "corpus"),
        ]
    )

    assert load_calls == [
        (
            TINYSTORIES_INSTRUCT_DATASET_NAME,
            {
                "revision": TINYSTORIES_INSTRUCT_DATASET_REVISION,
                "cache_dir": None,
            },
        )
    ]
    assert corpus_load_calls[0]["corpus_name"] == (
        "tinystories-instruct-packed-full-v1"
    )
    assert corpus_load_calls[0]["optimizer_token_limit"] is None
    captured = capsys.readouterr()
    assert "event=tokenizer_reused" in captured.err
    assert "event=already_prepared" in captured.err
    summary = json.loads(captured.out)
    assert summary["tokenizer_status"] == "already_prepared"
    assert summary["corpus_status"] == "already_prepared"
    assert summary["dataset_revision"] == TINYSTORIES_INSTRUCT_DATASET_REVISION


def test_audit_script_imports_project_from_outside_repository(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/audit_prepared_corpus.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--prepared-corpus-dir" in completed.stdout
