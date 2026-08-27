from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.analyze_tinystories_convergence import (
    analyze_run,
    build_report,
    convergence_evidence,
    select_recipe,
)
from src.training.packed_corpus import (
    PackedCorpusError,
    load_existing_corpus_if_matching,
    prepare_packed_corpus,
    sha256_file,
)
from src.training.fineweb_tokenizer import (
    load_tokenizer_manifest,
    train_sentencepiece_tokenizer,
)
from src.training.tinystories import (
    TinyStoriesError,
    dataset_split_fingerprint,
    iter_dataset_stories,
    role_ordered_stories,
    tokenizer_training_stories,
)
from src.utils.config import resolve_run_config
from src.utils.reproducibility import stable_hash


class ControlledTokenizer:
    eos_token_id = 2
    vocab_size = 2_048

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": [10 + (len(str(text)) % 20)]}


class ControlledDataset(list):
    column_names = ["text"]

    def __init__(self, prefix: str, count: int, *, fingerprint: str):
        super().__init__(
            {"text": f"{prefix} complete story {index}."}
            for index in range(count)
        )
        self._fingerprint = fingerprint


def _tokenizer_artifact(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "tokenizer"
    root.mkdir()
    model = root / "tokenizer.model"
    config = root / "tokenizer_config.json"
    model.write_bytes(b"controlled-tokenizer")
    config.write_text("{}\n", encoding="utf-8")
    files = {path.name: sha256_file(path) for path in (model, config)}
    manifest = {
        "schema_version": 1,
        "training_version": "tinystories_sentencepiece_bpe_v1",
        "tokenizer_name": "tinystories_sentencepiece_bpe_2k",
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


def test_dataset_story_rows_and_role_assignment_are_disjoint():
    train = ControlledDataset("train", 140, fingerprint="train-fingerprint")
    train[130]["text"] = "   "
    validation = ControlledDataset(
        "validation", 650, fingerprint="validation-fingerprint"
    )

    parsed = list(iter_dataset_stories(validation, split="validation"))
    assert len(parsed) == 650
    assert parsed[0]["text"] == "validation complete story 0."
    assert parsed[-1]["id"] == "validation:649"

    tokenizer_rows = list(tokenizer_training_stories(train, story_count=3))
    assert [row["source_story_index"] for row in tokenizer_rows] == [128, 129, 131]

    ordered = iter(role_ordered_stories(train, validation))
    ordinary_ids = [next(ordered)["id"] for _ in range(128)]
    controller_ids = [next(ordered)["id"] for _ in range(128)]
    final_ids = [next(ordered)["id"] for _ in range(512)]
    optimizer_first = next(ordered)["id"]
    assert ordinary_ids[0] == "validation:0"
    assert controller_ids[0] == "train:0"
    assert final_ids[0] == "validation:128"
    assert optimizer_first == "train:128"
    assert len(set(ordinary_ids + controller_ids + final_ids)) == 768


def test_dataset_story_rows_skip_blanks_and_require_native_fingerprint():
    with pytest.raises(TinyStoriesError, match="has no text column"):
        list(iter_dataset_stories([{"other": "story"}], split="train"))
    stories = list(
        iter_dataset_stories(
            [
                {"text": "first"},
                {"text": "  \n"},
                {"text": None},
                {"text": "fourth"},
            ],
            split="train",
        )
    )
    assert [story["id"] for story in stories] == ["train:0", "train:3"]
    assert [story["nonempty_story_index"] for story in stories] == [0, 1]
    with pytest.raises(TinyStoriesError, match="datasets fingerprint"):
        dataset_split_fingerprint([{"text": "story"}], split="train")

    dataset = ControlledDataset("train", 2, fingerprint="native-fingerprint")
    assert dataset_split_fingerprint(dataset, split="train") == stable_hash(
        {
            "dataset_name": "roneneldan/TinyStories",
            "dataset_config_name": "default",
            "dataset_revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
            "split": "train",
            "datasets_fingerprint": "native-fingerprint",
            "row_count": 2,
            "text_column": "text",
            "row_filter": "skip_blank_text_preserve_physical_row_index_v1",
        }
    )


def test_prepare_command_loads_pinned_huggingface_dataset(
    tmp_path, monkeypatch, capsys
):
    from scripts import prepare_tinystories as command

    train = ControlledDataset("train", 1, fingerprint="train-native")
    validation = ControlledDataset(
        "validation", 1, fingerprint="validation-native"
    )
    load_calls = []
    corpus_load_calls = []

    def fake_load_dataset(name, **kwargs):
        load_calls.append((name, kwargs))
        return {"train": train, "validation": validation}

    tokenizer_manifest = {
        "tokenizer_name": "tinystories_sentencepiece_bpe_2k",
        "manifest_hash": "tokenizer-hash",
    }
    corpus_manifest = {
        "corpus_hash": "corpus-hash",
        "available_optimizer_token_count": 33_554_432,
        "available_optimizer_sequence_count": 262_144,
        "reserved_role_counts": {
            "ordinary_validation": 128,
            "controller": 128,
            "final_holdout": 512,
        },
    }
    monkeypatch.setattr(command, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        command,
        "load_existing_tinystories_tokenizer_if_matching",
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
            "roneneldan/TinyStories",
            {
                "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
                "cache_dir": None,
            },
        )
    ]
    summary = json.loads(capsys.readouterr().out)
    assert summary["dataset"] == "roneneldan/TinyStories"
    assert summary["dataset_revision"] == load_calls[0][1]["revision"]
    assert summary["optimizer_token_request"] == "all"
    assert corpus_load_calls[0]["optimizer_token_limit"] is None


def test_prepare_command_accepts_explicit_or_full_optimizer_token_count():
    from scripts import prepare_tinystories as command

    shared = ["--tokenizer-dir", "tokenizer", "--corpus-dir", "corpus"]
    assert command.parse_args(shared).optimizer_token_count is None
    assert command.parse_args(
        [*shared, "--optimizer-token-count", "all"]
    ).optimizer_token_count is None
    assert command.parse_args(
        [*shared, "--optimizer-token-count", "67108864"]
    ).optimizer_token_count == 67_108_864


def test_tinystories_slurm_finalizes_multiple_runs_sequentially(tmp_path):
    recorder = tmp_path / "python-recorder.sh"
    argv_path = tmp_path / "argv.txt"
    first_run = tmp_path / "completed-run-a"
    second_run = tmp_path / "completed-run-b"
    recorder.write_text(
        "#!/usr/bin/env bash\n"
        "printf '__CALL__\\n' >> \"$ARGV_FILE\"\n"
        'printf \'%s\\n\' "$@" >> "$ARGV_FILE"\n',
        encoding="utf-8",
    )
    recorder.chmod(0o755)
    env = os.environ.copy()
    env["ARGV_FILE"] = str(argv_path)

    subprocess.run(
        [
            "bash",
            "scripts/slurm_tinystories_controlled.sh",
            "--python-bin",
            str(recorder),
            "--final-holdout-only",
            str(first_run),
            "--final-holdout-only",
            str(second_run),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    calls = argv_path.read_text(encoding="utf-8").split("__CALL__\n")[1:]
    assert len(calls) == 2
    for call, run_dir in zip(calls, (first_run, second_run), strict=True):
        assert call.splitlines() == [
            "scripts/evaluate_final_holdout.py",
            "--run-dir",
            str(run_dir),
            "--device",
            "cuda",
            "--skip-existing",
        ]


@pytest.mark.parametrize(
    ("profile", "dataset_name", "tokenizer_name", "corpus_name", "phase"),
    [
        (
            "stories",
            "roneneldan/TinyStories",
            "tinystories-sentencepiece-bpe-2k-v1",
            "tinystories-packed-full-v1",
            "tinystories_controlled",
        ),
        (
            "instruct",
            "roneneldan/TinyStoriesInstruct",
            "tinystories-instruct-sentencepiece-bpe-2k-v1",
            "tinystories-instruct-packed-full-v1",
            "tinystories_instruct_controlled",
        ),
    ],
)
def test_tinystories_profile_selector_binds_data_contract(
    tmp_path,
    profile,
    dataset_name,
    tokenizer_name,
    corpus_name,
    phase,
):
    env = os.environ.copy()
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    env["TINYSTORIES_PROFILE"] = profile
    env["MATFORMER_TOKENIZER_ROOT"] = str(tmp_path / "tokenizers")
    env["MATFORMER_CORPUS_ROOT"] = str(tmp_path / "corpora")
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                "source scripts/select_tinystories_profile.sh >/dev/null; "
                "printf '%s\\n' \"$DATASET_NAME\" \"$TOKENIZER\" "
                "\"$CORPUS\" \"$DATASET_PHASE\""
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        dataset_name,
        str(tmp_path / "tokenizers" / tokenizer_name),
        str(tmp_path / "corpora" / corpus_name),
        phase,
    ]


def test_tinystories_profile_selector_requires_explicit_profile(tmp_path):
    env = os.environ.copy()
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    env.pop("TINYSTORIES_PROFILE", None)
    result = subprocess.run(
        ["bash", "-c", "source scripts/select_tinystories_profile.sh"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "must be stories or instruct" in result.stderr


def test_prepare_command_reports_resume_replay_and_live_throughput(
    tmp_path, monkeypatch, capsys
):
    from scripts import prepare_tinystories as command

    train = ControlledDataset("train", 1, fingerprint="train-native")
    validation = ControlledDataset(
        "validation", 1, fingerprint="validation-native"
    )
    monkeypatch.setattr(
        command,
        "load_dataset",
        lambda *args, **kwargs: {"train": train, "validation": validation},
    )
    monkeypatch.setattr(
        command,
        "load_existing_tinystories_tokenizer_if_matching",
        lambda *args, **kwargs: {
            "tokenizer_name": "tinystories_sentencepiece_bpe_2k",
            "manifest_hash": "tokenizer-hash",
        },
    )
    monkeypatch.setattr(
        command,
        "load_existing_corpus_if_matching",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        command.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    corpus_dir = tmp_path / "corpus"
    work_dir = command.preparation_work_dir(corpus_dir)
    work_dir.mkdir()
    (work_dir / "preparation_progress.json").write_text(
        json.dumps(
            {
                "committed_shuffled_document_count": 10,
                "role_states": {
                    "optimizer_training": {"token_count": 1_024}
                },
                "role_manifests": {},
            }
        ),
        encoding="utf-8",
    )

    def fake_prepare(*args, progress_callback, **kwargs):
        progress_callback(
            {
                "event": "resume_replay_completed",
                "replayed_document_count": 10,
                "replay_target_document_count": 10,
                "replay_elapsed_seconds": 2.0,
                "replay_documents_per_second": 5.0,
                "source_read_workers": 1,
            }
        )
        progress_callback(
            {
                "event": "progress",
                "phase": "tokenization",
                "role": "optimizer_training",
                "committed_document_count": 14,
                "optimizer_token_count": 2_048,
                "completed_optimizer_shard_count": 1,
                "current_optimizer_shard_index": 1,
                "current_optimizer_shard_offset": 256,
                "source_read_workers": 1,
                "tokenization_workers": 4,
            }
        )
        return {
            "source": {"shuffled_document_count": 14},
            "corpus_hash": "corpus-hash",
            "available_optimizer_token_count": 2_048,
            "available_optimizer_sequence_count": 16,
            "reserved_role_counts": {
                "ordinary_validation": 128,
                "controller": 128,
                "final_holdout": 512,
            },
        }

    monkeypatch.setattr(command, "prepare_packed_corpus", fake_prepare)
    command.main(
        [
            "--tokenizer-dir",
            str(tmp_path / "tokenizer"),
            "--corpus-dir",
            str(corpus_dir),
            "--tokenization-workers",
            "4",
        ]
    )

    captured = capsys.readouterr()
    assert "event=resuming documents=10 optimizer_tokens=1,024" in captured.err
    assert "target_optimizer_tokens=all_available" in captured.err
    assert "event=resume_replay_completed phase=resume_replay" in captured.err
    assert "replay_percent=100.00" in captured.err
    assert "event=progress phase=tokenization" in captured.err
    assert "completed_shards=1 current_shard=00001" in captured.err
    assert "documents_per_second=" in captured.err
    assert "tokens_per_second=" in captured.err
    assert "event=completed status=resumed_and_prepared" in captured.err
    assert json.loads(captured.out)["corpus_status"] == "resumed_and_prepared"
    assert json.loads(captured.out)["optimizer_token_request"] == "all"


def test_generic_sentencepiece_trainer_supports_small_dataset_tokenizer(tmp_path):
    documents = (
        {
            "id": f"train:{index}",
            "text": f"story number {index} has character {chr(97 + index % 26)}",
        }
        for index in range(300)
    )
    manifest = train_sentencepiece_tokenizer(
        documents,
        tmp_path / "small-tokenizer",
        tokenizer_name="test_sentencepiece_bpe_320",
        training_version="test_sentencepiece_bpe_v1",
        source_dataset="test/TinyStories",
        source_config="test",
        source_split="train",
        source_fingerprint="test-fingerprint",
        data_seed=42,
        shuffle_buffer_size=0,
        document_count=300,
        reserved_document_count=0,
        vocab_size=320,
    )
    assert manifest["vocab_size"] == 320
    assert manifest["reserved_document_count"] == 0
    assert manifest["training_document_count"] == 300
    assert load_tokenizer_manifest(tmp_path / "small-tokenizer") == manifest


def test_shared_packed_builder_supports_custom_roles_and_exact_token_limit(tmp_path):
    tokenizer_dir, tokenizer_manifest = _tokenizer_artifact(tmp_path)
    del tokenizer_dir
    role_counts = {
        "ordinary_validation": 2,
        "controller": 1,
        "final_holdout": 2,
    }
    role_sources = {
        role: {"split": "validation" if "validation" in role or role == "final_holdout" else "train"}
        for role in (
            "optimizer_training",
            "ordinary_validation",
            "controller",
            "final_holdout",
        )
    }
    manifest = prepare_packed_corpus(
        (
            {"id": f"doc-{index}", "text": f"story {index}"}
            for index in range(20)
        ),
        ControlledTokenizer(),
        tmp_path / "corpus",
        tokenizer_name=tokenizer_manifest["tokenizer_name"],
        tokenizer_revision=tokenizer_manifest["manifest_hash"],
        tokenizer_manifest=tokenizer_manifest,
        source_dataset="roneneldan/TinyStories",
        source_config="official-raw",
        source_split="train+validation",
        source_fingerprint="controlled-source",
        context_length=4,
        shard_token_capacity=8,
        shuffle_buffer_size=0,
        reserved_role_counts=role_counts,
        optimizer_token_limit=12,
        minimum_optimizer_document_count=6,
        role_source_provenance=role_sources,
    )
    assert manifest["available_optimizer_token_count"] == 12
    assert manifest["available_optimizer_sequence_count"] == 3
    assert manifest["optimizer_token_limit"] == 12
    assert manifest["minimum_optimizer_document_count"] == 6
    assert manifest["reserved_role_counts"] == role_counts
    assert manifest["source"]["termination"] == "optimizer_token_limit"
    assert manifest["source"]["source_exhausted"] is False
    assert manifest["roles"]["optimizer_training"]["source_document_count"] == 6

    reused = load_existing_corpus_if_matching(
        tmp_path / "corpus",
        tokenizer_manifest=tokenizer_manifest,
        tokenizer_name=tokenizer_manifest["tokenizer_name"],
        tokenizer_revision=tokenizer_manifest["manifest_hash"],
        source_dataset="roneneldan/TinyStories",
        source_config="official-raw",
        source_split="train+validation",
        context_length=4,
        shard_token_capacity=8,
        shuffle_buffer_size=0,
        reserved_role_counts=role_counts,
        optimizer_token_limit=12,
        minimum_optimizer_document_count=6,
        role_source_provenance=role_sources,
    )
    assert reused == manifest

    with pytest.raises(
        PackedCorpusError,
        match="Existing prepared corpus does not match the requested preparation",
    ):
        load_existing_corpus_if_matching(
            tmp_path / "corpus",
            tokenizer_manifest=tokenizer_manifest,
            tokenizer_name=tokenizer_manifest["tokenizer_name"],
            tokenizer_revision=tokenizer_manifest["manifest_hash"],
            source_dataset="roneneldan/TinyStories",
            source_config="official-raw",
            source_split="train+validation",
            context_length=4,
            shard_token_capacity=8,
            shuffle_buffer_size=0,
            reserved_role_counts=role_counts,
            optimizer_token_limit=None,
            minimum_optimizer_document_count=6,
            role_source_provenance=role_sources,
        )


def test_exact_token_limit_resume_does_not_consume_an_extra_story(
    tmp_path, monkeypatch
):
    import src.training.packed_corpus as packed_corpus

    _, tokenizer_manifest = _tokenizer_artifact(tmp_path)
    role_counts = {
        "ordinary_validation": 1,
        "controller": 1,
        "final_holdout": 1,
    }

    def prepare(output_name: str):
        return prepare_packed_corpus(
            ({"id": f"doc-{index}", "text": str(index)} for index in range(20)),
            ControlledTokenizer(),
            tmp_path / output_name,
            tokenizer_name=tokenizer_manifest["tokenizer_name"],
            tokenizer_revision=tokenizer_manifest["manifest_hash"],
            tokenizer_manifest=tokenizer_manifest,
            source_fingerprint="resume-source",
            context_length=4,
            shard_token_capacity=8,
            reserved_role_counts=role_counts,
            optimizer_token_limit=12,
            minimum_optimizer_document_count=6,
        )

    real_finish = packed_corpus._RoleShardWriter.finish
    interrupted = False

    def interrupt_optimizer_finish(writer):
        nonlocal interrupted
        manifest = real_finish(writer)
        if writer.role == "optimizer_training" and not interrupted:
            interrupted = True
            raise RuntimeError("interrupt after optimizer token limit")
        return manifest

    monkeypatch.setattr(
        packed_corpus._RoleShardWriter,
        "finish",
        interrupt_optimizer_finish,
    )
    with pytest.raises(RuntimeError, match="optimizer token limit"):
        prepare("resumed")
    monkeypatch.setattr(packed_corpus._RoleShardWriter, "finish", real_finish)

    resumed = prepare("resumed")
    fresh = prepare("fresh")
    assert resumed == fresh
    assert resumed["source"]["shuffled_document_count"] == 9


def test_token_limit_rejects_tokenizer_stories_outside_optimizer_corpus(tmp_path):
    _, tokenizer_manifest = _tokenizer_artifact(tmp_path)
    with pytest.raises(
        PackedCorpusError,
        match="tokenizer-training stories entered the optimizer corpus",
    ):
        prepare_packed_corpus(
            ({"id": f"doc-{index}", "text": str(index)} for index in range(20)),
            ControlledTokenizer(),
            tmp_path / "too-small",
            tokenizer_name=tokenizer_manifest["tokenizer_name"],
            tokenizer_revision=tokenizer_manifest["manifest_hash"],
            tokenizer_manifest=tokenizer_manifest,
            source_fingerprint="too-small-source",
            context_length=4,
            shard_token_capacity=8,
            reserved_role_counts={
                "ordinary_validation": 1,
                "controller": 1,
                "final_holdout": 1,
            },
            optimizer_token_limit=12,
            minimum_optimizer_document_count=7,
        )


def _prepare_config_fixture(tmp_path: Path) -> tuple[Path, Path]:
    tokenizer_dir, tokenizer_manifest = _tokenizer_artifact(tmp_path)
    role_counts = {
        "ordinary_validation": 128,
        "controller": 128,
        "final_holdout": 512,
    }
    prepare_packed_corpus(
        (
            {"id": f"story-{index}", "text": str(index)}
            for index in range(2_000)
        ),
        ControlledTokenizer(),
        tmp_path / "corpus",
        tokenizer_name=tokenizer_manifest["tokenizer_name"],
        tokenizer_revision=tokenizer_manifest["manifest_hash"],
        tokenizer_manifest=tokenizer_manifest,
        source_dataset="roneneldan/TinyStories",
        source_config="default",
        source_split="train+validation",
        source_fingerprint="config-source",
        context_length=128,
        shard_token_capacity=512,
        shuffle_buffer_size=0,
        reserved_role_counts=role_counts,
        optimizer_token_limit=2_048,
    )
    return tokenizer_dir, tmp_path / "corpus"


@pytest.mark.parametrize("learning_rate", [3e-4, 1e-3, 3e-3])
@pytest.mark.parametrize("scheduler", ["cosine", "constant_with_warmup"])
def test_controlled_config_resolves_dense_grid(
    tmp_path, learning_rate, scheduler
):
    tokenizer_dir, corpus_dir = _prepare_config_fixture(tmp_path)
    config = resolve_run_config(
        "configs/controlled_exps/tinystories_controlled_convergence.yaml",
        overrides=[
            f"model.tokenizer_dir={tokenizer_dir}",
            f"dataset.prepared_corpus_dir={corpus_dir}",
            f"run.output_root={tmp_path / 'outputs'}",
            "training.token_budget=2048",
            "training.batch_size_per_process=1",
            f"training.learning_rate={learning_rate}",
            f"training.scheduler.name={scheduler}",
        ],
    )
    assert config["run"]["model_family"] == "standalone"
    assert config["model"]["granularities"] == ["g1000"]
    assert config["model"]["intermediate_size"] == 512
    assert config["training"]["scheduler_name"] == scheduler
    assert config["training"]["resolved_learning_rate"] == learning_rate
    assert config["dataset"]["mode"] == "packed_mmap"


def test_same_config_resolves_later_nested_window_override(tmp_path):
    tokenizer_dir, corpus_dir = _prepare_config_fixture(tmp_path)
    config = resolve_run_config(
        "configs/controlled_exps/tinystories_controlled_convergence.yaml",
        overrides=[
            f"model.tokenizer_dir={tokenizer_dir}",
            f"dataset.prepared_corpus_dir={corpus_dir}",
            f"run.output_root={tmp_path / 'outputs'}",
            "training.token_budget=2048",
            "training.batch_size_per_process=1",
            "run.model_family=nested",
            "run.sampling_mode=nested-random",
            "run.granularity=null",
            "model.granularity_sampling_mode=global",
            "model.global_sampling_schedule=balanced_cycle",
            "model.global_sampling_interval_steps=2",
        ],
    )
    assert config["run"]["model_family"] == "nested"
    assert config["model"]["granularities"] == [
        "g125",
        "g250",
        "g375",
        "g500",
        "g625",
        "g750",
        "g875",
        "g1000",
    ]
    assert config["model"]["global_sampling_interval_steps"] == 2
    assert config["model"]["global_sampling_schedule"] == "balanced_cycle"


def test_controlled_config_freezes_selected_recipe(tmp_path):
    tokenizer_dir, corpus_dir = _prepare_config_fixture(tmp_path)
    config = resolve_run_config(
        "configs/controlled_exps/tinystories_controlled_convergence.yaml",
        overrides=[
            f"model.tokenizer_dir={tokenizer_dir}",
            f"dataset.prepared_corpus_dir={corpus_dir}",
            f"run.output_root={tmp_path / 'outputs'}",
            "training.token_budget=1024",
            "training.batch_size_per_process=1",
        ],
    )
    assert config["controlled_experiment"] == {
        "recipe_status": "frozen",
        "recipe_source_run_id": (
            "tinystories-dense-lr3e-3-schedcosine-4096-s42"
        ),
        "selection_report_hash": (
            "ecf84f1131b57255e945c10b51599bd1e84dfc872d48e65bc7a4818fe92c1c69"
        ),
    }
    assert config["run"]["phase_id"] == "tinystories_frozen_elastic"
    assert config["monitoring"]["project"] == "tinystories_frozen_elastic"
    assert config["training"]["learning_rate"] == pytest.approx(3e-3)
    assert config["training"]["scheduler_name"] == "cosine"


@pytest.mark.parametrize(
    "dataset_phase",
    ["tinystories_controlled", "tinystories_instruct_controlled"],
)
def test_controlled_config_resolves_four_granularity_scope(
    tmp_path, dataset_phase
):
    tokenizer_dir, corpus_dir = _prepare_config_fixture(tmp_path)
    config = resolve_run_config(
        "configs/controlled_exps/tinystories_controlled_convergence.yaml",
        overrides=[
            f"model.tokenizer_dir={tokenizer_dir}",
            f"dataset.prepared_corpus_dir={corpus_dir}",
            f"run.output_root={tmp_path / 'outputs'}",
            "training.token_budget=1024",
            "training.batch_size_per_process=1",
            "run.model_family=nested",
            "run.sampling_mode=nested-random",
            "run.granularity=null",
            f"dataset.dataset_phase={dataset_phase}",
            "model.granularities=[g250,g500,g750,g1000]",
            (
                "model.granularity_prefixes={g250: 0.25, g500: 0.5, "
                "g750: 0.75, g1000: 1.0}"
            ),
        ],
    )
    assert config["model"]["granularities"] == [
        "g250",
        "g500",
        "g750",
        "g1000",
    ]
    assert config["model"]["granularity_prefixes"] == {
        "g250": 0.25,
        "g500": 0.5,
        "g750": 0.75,
        "g1000": 1.0,
    }
    assert config["dataset"]["dataset_phase"] == dataset_phase
    assert config["model"]["intermediate_size"] == 512


def test_controlled_config_runs_one_cpu_step_through_train_entrypoint(tmp_path):
    import train

    tokenizer_dir, corpus_dir = _prepare_config_fixture(tmp_path)
    run_id = "tinystories-controlled-cpu-smoke"
    output_dir = tmp_path / "outputs" / run_id
    train.main(
        [
            "--config",
            "configs/controlled_exps/tinystories_controlled_convergence.yaml",
            "--output-dir",
            str(output_dir),
            "--override",
            f"run.run_id={run_id}",
            "--override",
            f"model.tokenizer_dir={tokenizer_dir}",
            "--override",
            f"dataset.prepared_corpus_dir={corpus_dir}",
            "--override",
            "training.token_budget=128",
            "--override",
            "training.batch_size_per_process=1",
            "--override",
            "training.warmup_steps=0",
            "--override",
            "training.mixed_precision=none",
            "--override",
            "evaluation.validation.interval_steps=1",
        ]
    )
    summary = json.loads(
        (output_dir / "run_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "completed"
    assert summary["tokens_seen"] == 128
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "checkpoints" / "latest.pt").is_file()


def _write_run(
    root: Path,
    run_id: str,
    losses: list[float],
    *,
    learning_rate: float,
    scheduler: str,
    wall_clock_seconds: float,
    status: str = "completed",
    tokens_seen: int = 16_777_216,
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "checkpoints" / "best_eval_step_512.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    summary = {
        "run_id": run_id,
        "status": status,
        "tokens_seen": tokens_seen,
        "token_budget": 16_777_216,
        "derived_max_steps": 2_048,
        "base_learning_rate": learning_rate,
        "scheduler_name": scheduler,
        "checkpoint_status": "best_eval",
        "best_checkpoint_path": str(checkpoint),
        "unresolved_artifact_failures": [],
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["split", "step", "granularity", "loss", "wall_clock_seconds"],
        )
        writer.writeheader()
        for index, loss in enumerate(losses):
            writer.writerow(
                {
                    "split": "validation",
                    "step": 512 + index * 64,
                    "granularity": "g1000",
                    "loss": loss,
                    "wall_clock_seconds": wall_clock_seconds,
                }
            )
    return run_dir


def test_convergence_analysis_selects_loss_then_runtime_tiebreak(tmp_path):
    plateau = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    assert convergence_evidence(plateau_rows := [
        {"step": 512 + index * 64, "loss": loss}
        for index, loss in enumerate(plateau)
    ])["converged"] is True
    assert len(plateau_rows) == 6

    first = _write_run(
        tmp_path,
        "lr1e-3-cosine",
        plateau,
        learning_rate=1e-3,
        scheduler="cosine",
        wall_clock_seconds=100.0,
    )
    second = _write_run(
        tmp_path,
        "lr3e-4-constant",
        [2.001] * 6,
        learning_rate=3e-4,
        scheduler="constant_with_warmup",
        wall_clock_seconds=80.0,
    )
    rows = [analyze_run(first), analyze_run(second)]
    winner, fallback = select_recipe(rows)
    assert winner["run_id"] == "lr3e-4-constant"
    assert fallback == []
    report = build_report(
        rows,
        minimum_step=512,
        patience_evaluations=5,
        relative_improvement=0.005,
        tie_tolerance=0.001,
    )
    assert report["status"] == "recipe_selected"
    assert report["selection_contract"]["eligibility_rule"] == (
        "global_best_stable_must_converge_v1"
    )
    assert report["report_hash"] == stable_hash(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    assert "training.scheduler.name=constant_with_warmup" in report[
        "frozen_recipe_overrides"
    ]


def test_worse_converged_run_cannot_displace_global_best_stable_run(tmp_path):
    best = _write_run(
        tmp_path,
        "best-still-improving",
        [2.0, 1.95, 1.9, 1.85, 1.8, 1.75],
        learning_rate=3e-3,
        scheduler="cosine",
        wall_clock_seconds=100.0,
    )
    second = _write_run(
        tmp_path,
        "second-still-improving",
        [2.2, 2.14, 2.08, 2.02, 1.96, 1.9],
        learning_rate=3e-3,
        scheduler="constant_with_warmup",
        wall_clock_seconds=90.0,
    )
    worse_plateau = _write_run(
        tmp_path,
        "worse-plateau",
        [2.5] * 6,
        learning_rate=3e-4,
        scheduler="cosine",
        wall_clock_seconds=80.0,
    )
    rows = [analyze_run(path) for path in (best, second, worse_plateau)]
    assert next(row for row in rows if row["run_id"] == "worse-plateau")[
        "converged"
    ]

    winner, fallback = select_recipe(rows)
    assert winner is None
    assert [row["run_id"] for row in fallback] == [
        "best-still-improving",
        "second-still-improving",
    ]
    report = build_report(
        rows,
        minimum_step=512,
        patience_evaluations=5,
        relative_improvement=0.005,
        tie_tolerance=0.001,
    )
    assert report["status"] == "fallback_required"
    assert report["winner"] is None
    assert "frozen_recipe_overrides" not in report


def test_analysis_rejects_incomplete_run_and_requests_two_fallbacks(tmp_path):
    improving = [3.0, 2.9, 2.8, 2.7, 2.6, 2.5]
    stable_a = _write_run(
        tmp_path,
        "stable-a",
        improving,
        learning_rate=1e-3,
        scheduler="cosine",
        wall_clock_seconds=90.0,
    )
    stable_b = _write_run(
        tmp_path,
        "stable-b",
        [value + 0.1 for value in improving],
        learning_rate=3e-4,
        scheduler="cosine",
        wall_clock_seconds=85.0,
    )
    incomplete = _write_run(
        tmp_path,
        "incomplete",
        [2.0] * 6,
        learning_rate=3e-3,
        scheduler="cosine",
        wall_clock_seconds=50.0,
        status="failed",
        tokens_seen=100,
    )
    rows = [analyze_run(path) for path in (stable_a, stable_b, incomplete)]
    winner, fallback = select_recipe(rows)
    assert winner is None
    assert [row["run_id"] for row in fallback] == ["stable-a", "stable-b"]
    rejected = next(row for row in rows if row["run_id"] == "incomplete")
    assert rejected["stable"] is False
    assert rejected["converged"] is False
