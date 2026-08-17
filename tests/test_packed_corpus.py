from __future__ import annotations

import fcntl
import itertools
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.training.distributed import DistributedContext
from src.training.data import build_packed_mmap_dataloaders, packed_sampler_state
from src.training.packed_corpus import (
    NoPaddingDistributedBatchSampler,
    PERMUTATION_VERSION,
    PackedCorpusError,
    PackedMMapDataset,
    _ordered_tokenize_documents,
    audit_packed_corpus,
    iter_streaming_documents_with_ordered_prefetch,
    load_existing_corpus_if_matching,
    load_corpus_manifest,
    partition_permutation_without_padding,
    preparation_lock_path,
    preparation_work_dir,
    prepare_packed_corpus,
)
from src.training.steps import weighted_loss_for_distributed_batch
from src.training.steps import build_optimizer_and_scheduler, train_for_steps
from src.utils.config import resolve_run_config
from src.utils.reproducibility import stable_hash


class TinyTokenizer:
    eos_token_id = 2
    vocab_size = 128

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": [int(value) for value in text.split()]}


class FailingTokenizer(TinyTokenizer):
    def __init__(self, fail_on_call: int):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def __call__(self, text, **kwargs):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("injected tokenization interruption")
        return super().__call__(text, **kwargs)


class ConcurrencyProbeTokenizer(TinyTokenizer):
    def __init__(self):
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def __call__(self, text, **kwargs):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.01)
            return super().__call__(text, **kwargs)
        finally:
            with self._lock:
                self.active -= 1


class TinyPackedTrainingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def configure_subnetwork(self, granularity):
        del granularity

    def forward(self, input_ids, attention_mask=None, labels=None):
        del attention_mask, labels
        return SimpleNamespace(
            loss=self.weight.square() + input_ids.float().mean() * 0.0
        )


def documents(count: int = 1176):
    for index in range(count):
        yield {"id": f"doc-{index}", "text": f"{index % 40} {(index + 1) % 40}"}


def tiny_tokenizer_manifest():
    manifest = {
        "schema_version": 1,
        "tokenizer_name": "tiny",
        "vocab_size": TinyTokenizer.vocab_size,
        "special_token_ids": {"unk": 0, "bos": 1, "eos": 2, "pad": 3},
        "sentencepiece_model_sha256": "tiny-model-checksum",
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    return manifest


def prepare_small(
    tmp_path,
    *,
    output_name="corpus",
    tokenization_workers=1,
    source_read_workers=1,
    tokenizer=None,
    context_length=4,
    document_count=1176,
    progress_callback=None,
    progress_interval_seconds=60.0,
):
    tokenizer_manifest = tiny_tokenizer_manifest()
    return prepare_packed_corpus(
        documents(document_count),
        tokenizer or TinyTokenizer(),
        tmp_path / output_name,
        tokenizer_name="tiny",
        tokenizer_revision=tokenizer_manifest["manifest_hash"],
        tokenizer_manifest=tokenizer_manifest,
        source_fingerprint="fineweb-test-fingerprint",
        context_length=context_length,
        shard_token_capacity=16,
        tokenization_workers=tokenization_workers,
        source_read_workers=source_read_workers,
        progress_callback=progress_callback,
        progress_interval_seconds=progress_interval_seconds,
    )


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_finite_source_publishes_actual_size_roles_hashes_and_order(tmp_path):
    created = prepare_small(tmp_path)
    loaded = load_corpus_manifest(tmp_path / "corpus", verify_shards=True)
    assert loaded == created
    assert created["schema_version"] == 3
    assert created["source"]["source_exhausted"] is True
    assert created["source"]["shuffled_document_count"] == 1176
    assert created["roles"]["optimizer_training"]["token_count"] == 72
    assert created["roles"]["optimizer_training"]["sequence_count"] == 18
    assert created["available_optimizer_token_count"] == 72
    assert created["available_optimizer_sequence_count"] == 18
    assert "training_token_budget" not in created
    assert created["reserved_role_counts"] == {
        "ordinary_validation": 512,
        "controller": 128,
        "final_holdout": 512,
    }
    identity_sets = {
        role: {
            item["source_row_identity"]
            for item in created["roles"][role][
                "ordered_source_document_identities"
            ]
        }
        for role in ("ordinary_validation", "controller", "final_holdout")
    }
    assert identity_sets["ordinary_validation"].isdisjoint(
        identity_sets["controller"]
    )
    assert identity_sets["ordinary_validation"].isdisjoint(
        identity_sets["final_holdout"]
    )
    assert identity_sets["controller"].isdisjoint(identity_sets["final_holdout"])
    validation = PackedMMapDataset(tmp_path / "corpus", "ordinary_validation")
    flattened = list(
        itertools.chain.from_iterable(
            validation[index]["input_ids"] for index in range(2)
        )
    )
    assert flattened[:6] == [0, 1, 2, 1, 2, 2]
    audit = audit_packed_corpus(tmp_path / "corpus", minimum_training_tokens=64)
    assert audit["status"] == "passed"
    with pytest.raises(PackedCorpusError, match="requires at least 76"):
        audit_packed_corpus(tmp_path / "corpus", minimum_training_tokens=76)

    order_metadata = created["training_order"]
    assert order_metadata["permutation_version"] == PERMUTATION_VERSION
    order = np.memmap(
        tmp_path / "corpus" / order_metadata["path"], mode="r", dtype="<u8"
    )
    assert len(order) == 18
    assert sorted(int(value) for value in order) == list(range(18))


def test_preparation_reports_progress_without_changing_artifact(tmp_path):
    events = []
    reported = prepare_small(
        tmp_path,
        output_name="reported",
        progress_callback=lambda event: events.append(dict(event)),
        progress_interval_seconds=10_000,
    )
    quiet = prepare_small(tmp_path, output_name="quiet")
    assert reported == quiet
    assert tree_bytes(tmp_path / "reported") == tree_bytes(tmp_path / "quiet")
    assert any(event["event"] == "shard_completed" for event in events)
    assert any(event["event"] == "source_exhausted" for event in events)
    assert events[-1]["event"] == "permutation_completed"
    assert events[-1]["optimizer_token_count"] == 72


def test_matching_v3_corpus_is_reused_and_mismatch_is_rejected(tmp_path):
    created = prepare_small(tmp_path)
    corpus_dir = tmp_path / "corpus"
    tokenizer_manifest = tiny_tokenizer_manifest()

    loaded = load_existing_corpus_if_matching(
        corpus_dir,
        tokenizer_manifest=tokenizer_manifest,
        tokenizer_name="tiny",
        tokenizer_revision=tokenizer_manifest["manifest_hash"],
        context_length=4,
        shard_token_capacity=16,
    )
    assert loaded == created
    assert (
        load_existing_corpus_if_matching(
            tmp_path / "missing",
            tokenizer_manifest=tokenizer_manifest,
            tokenizer_name="tiny",
            tokenizer_revision=tokenizer_manifest["manifest_hash"],
            context_length=4,
            shard_token_capacity=16,
        )
        is None
    )
    with pytest.raises(PackedCorpusError, match="does not match.*context_length"):
        load_existing_corpus_if_matching(
            corpus_dir,
            tokenizer_manifest=tokenizer_manifest,
            tokenizer_name="tiny",
            tokenizer_revision=tokenizer_manifest["manifest_hash"],
            context_length=8,
            shard_token_capacity=16,
        )


def test_parallel_and_resumed_preparations_are_byte_identical(tmp_path):
    serial = prepare_small(tmp_path, output_name="serial", tokenization_workers=1)
    parallel = prepare_small(tmp_path, output_name="parallel", tokenization_workers=8)
    assert parallel == serial

    for output_name, fail_on_call in (
        ("resumed-at-shard", 1159),
        ("resumed-partial-shard", 1161),
    ):
        with pytest.raises(RuntimeError, match="injected"):
            prepare_small(
                tmp_path,
                output_name=output_name,
                tokenizer=FailingTokenizer(fail_on_call),
                tokenization_workers=4,
            )
        progress_path = preparation_work_dir(tmp_path / output_name) / (
            "preparation_progress.json"
        )
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        assert progress["committed_shuffled_document_count"] == fail_on_call - 1
        resume_events = []
        resumed = prepare_small(
            tmp_path,
            output_name=output_name,
            tokenization_workers=3,
            source_read_workers=3,
            progress_callback=lambda event: resume_events.append(dict(event)),
            progress_interval_seconds=1e-9,
        )
        assert resumed == serial
        assert tree_bytes(tmp_path / output_name) == tree_bytes(tmp_path / "serial")
        replay_events = [
            event
            for event in resume_events
            if event["event"].startswith("resume_replay")
        ]
        assert replay_events
        assert replay_events[-1]["event"] == "resume_replay_completed"
        assert replay_events[-1]["replayed_document_count"] == fail_on_call - 1
        assert replay_events[-1]["replay_target_document_count"] == fail_on_call - 1
        assert replay_events[-1]["source_read_workers"] == 3


def test_interrupted_permutation_restarts_without_retokenizing(
    tmp_path, monkeypatch
):
    import src.training.packed_corpus as packed_corpus

    real_sha256_file = packed_corpus.sha256_file
    interrupted = False

    def interrupt_ordering(path, **kwargs):
        nonlocal interrupted
        if Path(path).name == "optimizer_training_order.u64" and not interrupted:
            interrupted = True
            raise RuntimeError("injected permutation interruption")
        return real_sha256_file(path, **kwargs)

    monkeypatch.setattr(packed_corpus, "sha256_file", interrupt_ordering)
    with pytest.raises(RuntimeError, match="permutation interruption"):
        prepare_small(tmp_path, output_name="resumed")
    progress = json.loads(
        (
            preparation_work_dir(tmp_path / "resumed")
            / "preparation_progress.json"
        ).read_text(encoding="utf-8")
    )
    assert progress["phase"] == "permutation"

    class NoTokenizationAllowed(TinyTokenizer):
        def __call__(self, text, **kwargs):
            raise AssertionError("permutation resume must not tokenize")

    resumed = prepare_small(
        tmp_path, output_name="resumed", tokenizer=NoTokenizationAllowed()
    )
    monkeypatch.setattr(packed_corpus, "sha256_file", real_sha256_file)
    fresh = prepare_small(tmp_path, output_name="fresh")
    assert resumed == fresh
    assert tree_bytes(tmp_path / "resumed") == tree_bytes(tmp_path / "fresh")


def test_incompatible_corrupt_and_concurrent_partial_state_is_retained(tmp_path):
    output = tmp_path / "corpus"
    with pytest.raises(RuntimeError):
        prepare_small(tmp_path, tokenizer=FailingTokenizer(1160))
    progress_path = preparation_work_dir(output) / "preparation_progress.json"
    original = progress_path.read_bytes()
    with pytest.raises(PackedCorpusError, match="incompatible"):
        prepare_small(tmp_path, context_length=8)
    assert progress_path.read_bytes() == original

    progress_path.write_text("not-json", encoding="utf-8")
    corrupt = progress_path.read_bytes()
    with pytest.raises(PackedCorpusError, match="Corrupt"):
        prepare_small(tmp_path)
    assert progress_path.read_bytes() == corrupt

    separate_output = tmp_path / "locked"
    lock_path = preparation_lock_path(separate_output)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(PackedCorpusError, match="already running"):
            prepare_small(tmp_path, output_name="locked")
    assert not separate_output.exists()


def test_corrupt_partial_shard_is_rejected_without_cleanup(tmp_path):
    output = tmp_path / "corrupt-shard"
    with pytest.raises(RuntimeError):
        prepare_small(
            tmp_path,
            output_name="corrupt-shard",
            tokenizer=FailingTokenizer(1161),
        )
    work_dir = preparation_work_dir(output)
    progress_path = work_dir / "preparation_progress.json"
    checkpoint = progress_path.read_bytes()
    shard = next(work_dir.glob("optimizer_training/*.bin"))
    shard.chmod(0o644)
    corrupted = shard.read_bytes()[:-1] + b"x"
    shard.write_bytes(corrupted)
    with pytest.raises(PackedCorpusError, match="checksum mismatch"):
        prepare_small(tmp_path, output_name="corrupt-shard")
    assert progress_path.read_bytes() == checkpoint
    assert shard.read_bytes() == corrupted
    assert work_dir.exists()


def test_ordered_tokenization_workers_run_concurrently_without_reordering():
    tokenizer = ConcurrencyProbeTokenizer()
    records = [(index, {"text": str(index)}) for index in range(12)]
    tokenized = list(
        _ordered_tokenize_documents(
            records, tokenizer, text_column="text", workers=4
        )
    )
    assert tokenized == [(index, [index]) for index in range(12)]
    assert tokenizer.max_active > 1


def test_ordered_source_prefetch_matches_huggingface_shuffle_exactly():
    from datasets import Dataset

    base = Dataset.from_dict(
        {
            "id": list(range(1000)),
            "text": [str(index) for index in range(1000)],
        }
    )
    serial_dataset = base.to_iterable_dataset(num_shards=17).shuffle(
        seed=42, buffer_size=31
    )
    parallel_dataset = base.to_iterable_dataset(num_shards=17).shuffle(
        seed=42, buffer_size=31
    )

    serial = [document["id"] for document in serial_dataset]
    parallel = [
        document["id"]
        for document in iter_streaming_documents_with_ordered_prefetch(
            parallel_dataset,
            workers=8,
        )
    ]

    assert parallel == serial


def test_ordered_source_prefetch_reads_multiple_shards_concurrently():
    class Probe:
        def __init__(self):
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0

    probe = Probe()

    class Shard:
        def __init__(self, index):
            self.index = index

        def __iter__(self):
            with probe.lock:
                probe.active += 1
                probe.max_active = max(probe.max_active, probe.active)
            try:
                time.sleep(0.02)
                yield self.index, {"id": self.index}
            finally:
                with probe.lock:
                    probe.active -= 1

    class Child:
        num_shards = 12

        def shard_data_sources(self, num_shards, index, contiguous=True):
            assert num_shards == self.num_shards
            assert contiguous is True
            return Shard(index)

    class Prepared:
        ex_iterable = Child()
        buffer_size = 1
        generator = np.random.default_rng(42)

        def __iter__(self):
            yield from self.ex_iterable

        @staticmethod
        def _iter_random_indices(rng, buffer_size):
            while True:
                yield int(rng.integers(0, buffer_size))

    class StreamingDataset:
        def _prepare_ex_iterable_for_iteration(self):
            return Prepared()

        def __iter__(self):
            raise AssertionError("parallel source prefetch unexpectedly fell back")

    documents_read = list(
        iter_streaming_documents_with_ordered_prefetch(
            StreamingDataset(),
            workers=4,
        )
    )

    assert [document["id"] for document in documents_read] == list(range(12))
    assert probe.max_active > 1


def test_existing_corpus_reuse_verifies_shard_and_order_checksums(tmp_path):
    prepare_small(tmp_path)
    corpus_dir = tmp_path / "corpus"
    shard = next(corpus_dir.glob("optimizer_training/*.bin"))
    shard.chmod(0o644)
    shard.write_bytes(shard.read_bytes()[:-1] + b"x")
    tokenizer_manifest = tiny_tokenizer_manifest()
    with pytest.raises(PackedCorpusError, match="checksum mismatch"):
        load_existing_corpus_if_matching(
            corpus_dir,
            tokenizer_manifest=tokenizer_manifest,
            tokenizer_name="tiny",
            tokenizer_revision=tokenizer_manifest["manifest_hash"],
            context_length=4,
            shard_token_capacity=16,
        )


def test_existing_corpus_cli_exits_before_loading_source_or_tokenizer(
    tmp_path, monkeypatch, capsys
):
    from scripts import prepare_fineweb_corpus as command

    created = prepare_small(tmp_path)
    tokenizer_manifest = tiny_tokenizer_manifest()
    monkeypatch.setattr(
        command, "load_tokenizer_manifest", lambda *args, **kwargs: tokenizer_manifest
    )

    def fail(*args, **kwargs):
        raise AssertionError("FineWeb and AutoTokenizer must not load on reuse")

    monkeypatch.setattr(command, "load_dataset", fail)
    monkeypatch.setattr(command.AutoTokenizer, "from_pretrained", fail)
    command.main(
        [
            "--output-dir",
            str(tmp_path / "corpus"),
            "--prepared-tokenizer-dir",
            str(tmp_path / "tokenizer"),
            "--context-length",
            "4",
            "--shard-token-capacity",
            "16",
            "--tokenization-workers",
            "8",
        ]
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "already_prepared"
    assert summary["corpus_hash"] == created["corpus_hash"]
    assert summary["training_token_count"] == 72
    assert summary["tokenization_workers"] == 8
    assert summary["source_read_workers"] == 8


def test_schema_v2_is_rejected_with_rebuild_message(tmp_path):
    prepare_small(tmp_path)
    manifest_path = tmp_path / "corpus" / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    manifest.pop("corpus_hash")
    manifest["corpus_hash"] = stable_hash(manifest)
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PackedCorpusError, match="schema v2.*rebuild"):
        load_corpus_manifest(tmp_path / "corpus")


def test_stored_permutation_prefix_is_nested_and_partitioned_without_padding(
    tmp_path,
):
    manifest = prepare_small(tmp_path)
    order_path = tmp_path / "corpus" / manifest["training_order"]["path"]
    small = NoPaddingDistributedBatchSampler(
        18,
        2,
        0,
        1,
        selected_sample_count=8,
        permutation_path=order_path,
        permutation_hash_expected=manifest["training_order"]["sha256"],
    )
    large = NoPaddingDistributedBatchSampler(
        18,
        2,
        0,
        1,
        selected_sample_count=16,
        permutation_path=order_path,
        permutation_hash_expected=manifest["training_order"]["sha256"],
    )
    small_indices = list(itertools.chain.from_iterable(small))
    large_indices = list(itertools.chain.from_iterable(large))
    assert large_indices[:8] == small_indices

    samplers = [
        NoPaddingDistributedBatchSampler(
            18,
            2,
            rank,
            4,
            selected_sample_count=15,
            permutation_path=order_path,
            permutation_hash_expected=manifest["training_order"]["sha256"],
        )
        for rank in range(4)
    ]
    batches = [list(sampler) for sampler in samplers]
    assert [len(rank_batches[-1]) for rank_batches in batches] == [2, 2, 2, 1]
    consumed = [
        index
        for rank_batches in batches
        for batch in rank_batches
        for index in batch
    ]
    stored = np.memmap(order_path, mode="r", dtype="<u8")
    assert sorted(consumed) == sorted(int(value) for value in stored[:15])


def test_packed_dataloader_stops_at_run_budget_and_exposes_order_state(tmp_path):
    manifest = prepare_small(tmp_path)
    config = {
        "dataset": {
            "prepared_corpus_dir": str(tmp_path / "corpus"),
            "data_seed": 42,
            "dataset_name": "HuggingFaceFW/fineweb",
            "dataset_config_name": "sample-100BT",
            "dataset_split": "train",
        },
        "model": {
            "context_length": 4,
            "tokenizer_name": "tiny",
            "tokenizer_revision": tiny_tokenizer_manifest()["manifest_hash"],
            "tokenizer_manifest_hash": tiny_tokenizer_manifest()["manifest_hash"],
            "vocab_size": 128,
        },
        "training": {
            "token_budget": 32,
            "batch_size_per_process": 3,
            "dataloader_num_workers": 0,
        },
    }
    train_loader, _, _, loaded = build_packed_mmap_dataloaders(
        config, torch.device("cpu")
    )
    assert loaded == manifest
    batches = list(train_loader)
    assert [len(batch["packed_sequence_index"]) for batch in batches] == [3, 3, 2]
    assert sum(batch["input_ids"].numel() for batch in batches) == 32
    state = packed_sampler_state(train_loader)
    assert state["selected_sample_count"] == 8
    assert state["cursor"] == 8
    assert state["permutation_hash"] == manifest["training_order"]["sha256"]


def test_packed_training_processes_exact_budget_with_partial_accumulation(tmp_path):
    prepare_small(tmp_path)
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=tmp_path / "debug-nested-001",
        overrides=[
            "model.correction_mode=none",
            "evaluation.validation=false",
            "evaluation.final_validation=false",
            "training.eval_interval=0",
        ],
    )
    tokenizer_manifest = tiny_tokenizer_manifest()
    config["dataset"].update(
        {
            "mode": "packed_mmap",
            "prepared_corpus_dir": str(tmp_path / "corpus"),
            "data_seed": 42,
            "dataset_name": "HuggingFaceFW/fineweb",
            "dataset_config_name": "sample-100BT",
            "dataset_split": "train",
        }
    )
    config["model"].update(
        {
            "context_length": 4,
            "tokenizer_name": "tiny",
            "tokenizer_revision": tokenizer_manifest["manifest_hash"],
            "tokenizer_manifest_hash": tokenizer_manifest["manifest_hash"],
            "vocab_size": 128,
        }
    )
    config["training"].update(
        {
            "token_budget": 32,
            "max_steps": 1,
            "batch_size_per_process": 3,
            "gradient_accumulation_steps": 4,
            "dataloader_num_workers": 0,
        }
    )
    train_loader, validation_loader, _, _ = build_packed_mmap_dataloaders(
        config, torch.device("cpu")
    )
    model = TinyPackedTrainingModel()
    optimizer, scheduler = build_optimizer_and_scheduler(model, config["training"])
    run_state = {
        "last_completed_step": 0,
        "epoch": 0,
        "batch_index": 0,
        "tokens_seen": 0,
        "content_tokens_seen": 0,
        "microstep": 0,
        "status": "fresh",
    }
    rows = train_for_steps(
        config,
        model,
        train_loader,
        validation_loader,
        optimizer,
        scheduler,
        torch.device("cpu"),
        run_state=run_state,
    )
    assert run_state["tokens_seen"] == 32
    assert run_state["content_tokens_seen"] == 32
    assert run_state["optimizer_window_microsteps"] == 3
    assert run_state["sampler_state"]["cursor"] == 8
    assert {row["tokens_seen"] for row in rows if row["split"] == "train"} == {32}


def test_no_padding_partitions_are_deterministic_disjoint_and_complete():
    first = partition_permutation_without_padding(23, world_size=4, data_seed=42)
    second = partition_permutation_without_padding(23, world_size=4, data_seed=42)
    assert first == second
    assert sorted(itertools.chain.from_iterable(first)) == list(range(23))
    for left_index, left in enumerate(first):
        for right in first[left_index + 1 :]:
            assert set(left).isdisjoint(right)


def test_sampler_checkpoint_records_selected_prefix_and_exact_resume(tmp_path):
    manifest = prepare_small(tmp_path)
    order_path = tmp_path / "corpus" / manifest["training_order"]["path"]
    original = NoPaddingDistributedBatchSampler(
        18,
        4,
        0,
        1,
        selected_sample_count=17,
        permutation_path=order_path,
        permutation_hash_expected=manifest["training_order"]["sha256"],
    )
    iterator = iter(original)
    first_batch = next(iterator)
    state = original.state_dict()
    assert state["selected_sample_count"] == 17
    assert state["permutation_hash"] == manifest["training_order"]["sha256"]
    resumed = NoPaddingDistributedBatchSampler(
        18,
        4,
        0,
        1,
        selected_sample_count=17,
        permutation_path=order_path,
        permutation_hash_expected=manifest["training_order"]["sha256"],
    )
    resumed.load_state_dict(state)
    full = NoPaddingDistributedBatchSampler(
        18,
        4,
        0,
        1,
        selected_sample_count=17,
        permutation_path=order_path,
        permutation_hash_expected=manifest["training_order"]["sha256"],
    )
    assert first_batch + list(itertools.chain.from_iterable(resumed)) == list(
        itertools.chain.from_iterable(full)
    )


def test_uneven_rank_loss_scaling_matches_global_token_weighted_gradient():
    parameter = torch.tensor(2.0, requires_grad=True)
    counts = [9, 6]
    local_means = [parameter * 3.0, parameter * 5.0]
    context = DistributedContext(enabled=True, rank=0, world_size=2)
    scaled = [
        weighted_loss_for_distributed_batch(
            loss,
            local_valid_targets=count,
            global_valid_targets=sum(counts),
            distributed_context=context,
        )
        for loss, count in zip(local_means, counts)
    ]
    fsdp_averaged_loss = sum(scaled) / 2
    fsdp_averaged_loss.backward()
    assert parameter.grad.item() == pytest.approx((9 * 3 + 6 * 5) / 15)
