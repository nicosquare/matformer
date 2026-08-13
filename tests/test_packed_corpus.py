from __future__ import annotations

import itertools
import json

import pytest
import torch

from src.training.distributed import DistributedContext
from src.training.packed_corpus import (
    NoPaddingDistributedBatchSampler,
    PackedCorpusError,
    PackedMMapDataset,
    audit_packed_corpus,
    load_corpus_manifest,
    partition_permutation_without_padding,
    prepare_packed_corpus,
)
from src.training.steps import weighted_loss_for_distributed_batch
from src.utils.reproducibility import stable_hash


class TinyTokenizer:
    eos_token_id = 2
    vocab_size = 128

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": [int(value) for value in text.split()]}


def documents(count: int):
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


def prepare_small(tmp_path, *, training_tokens=32):
    tokenizer_manifest = tiny_tokenizer_manifest()
    return prepare_packed_corpus(
        documents(1200),
        TinyTokenizer(),
        tmp_path / "corpus",
        tokenizer_name="tiny",
        tokenizer_revision=tokenizer_manifest["manifest_hash"],
        tokenizer_manifest=tokenizer_manifest,
        source_fingerprint="fineweb-test-fingerprint",
        context_length=4,
        training_token_budget=training_tokens,
        shard_token_capacity=16,
    )


def test_packing_has_eos_boundaries_exact_budget_roles_and_hashes(tmp_path):
    created = prepare_small(tmp_path)
    loaded = load_corpus_manifest(tmp_path / "corpus", verify_shards=True)
    assert loaded == created
    assert created["roles"]["optimizer_training"]["token_count"] == 32
    assert created["roles"]["optimizer_training"]["sequence_count"] == 8
    assert created["reserved_role_counts"] == {
        "ordinary_validation": 512,
        "controller": 128,
        "final_holdout": 512,
    }
    identity_sets = {
        role: {
            item["source_row_identity"]
            for item in created["roles"][role]["ordered_source_document_identities"]
        }
        for role in ("ordinary_validation", "controller", "final_holdout")
    }
    assert identity_sets["ordinary_validation"].isdisjoint(identity_sets["controller"])
    assert identity_sets["ordinary_validation"].isdisjoint(identity_sets["final_holdout"])
    assert identity_sets["controller"].isdisjoint(identity_sets["final_holdout"])
    validation = PackedMMapDataset(tmp_path / "corpus", "ordinary_validation")
    flattened = list(itertools.chain.from_iterable(validation[index]["input_ids"] for index in range(2)))
    assert flattened[:6] == [0, 1, 2, 1, 2, 2]
    audit = audit_packed_corpus(tmp_path / "corpus", required_training_tokens=32)
    assert audit["status"] == "passed"
    assert audit["corpus_hash"] == created["corpus_hash"]


def test_preparation_fails_when_unique_training_budget_is_unavailable(tmp_path):
    tokenizer_manifest = tiny_tokenizer_manifest()
    with pytest.raises(PackedCorpusError, match="Insufficient source data"):
        prepare_packed_corpus(
            documents(1153),
            TinyTokenizer(),
            tmp_path / "corpus",
            tokenizer_name="tiny",
            tokenizer_revision=tokenizer_manifest["manifest_hash"],
            tokenizer_manifest=tokenizer_manifest,
            source_fingerprint="fineweb-test-fingerprint",
            context_length=4,
            training_token_budget=64,
            shard_token_capacity=16,
        )


def test_corpus_loader_rejects_pre_provenance_schema(tmp_path):
    prepare_small(tmp_path)
    manifest_path = tmp_path / "corpus" / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    manifest.pop("corpus_hash")
    manifest["corpus_hash"] = stable_hash(manifest)
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PackedCorpusError, match="schema version mismatch"):
        load_corpus_manifest(tmp_path / "corpus")


def test_no_padding_partitions_are_deterministic_disjoint_and_complete():
    first = partition_permutation_without_padding(23, world_size=4, data_seed=42)
    second = partition_permutation_without_padding(23, world_size=4, data_seed=42)
    assert first == second
    assert sorted(itertools.chain.from_iterable(first)) == list(range(23))
    assert sum(len(partition) for partition in first) == 23
    for left_index, left in enumerate(first):
        for right in first[left_index + 1 :]:
            assert set(left).isdisjoint(right)


def test_no_padding_batch_sampler_uneven_final_batch_and_exact_cursor_resume():
    samplers = [
        NoPaddingDistributedBatchSampler(25, 4, rank, 4, data_seed=42)
        for rank in range(4)
    ]
    batches = [list(sampler) for sampler in samplers]
    assert [len(rank_batches[-1]) for rank_batches in batches] == [3, 2, 2, 2]
    consumed = [index for rank_batches in batches for batch in rank_batches for index in batch]
    assert sorted(consumed) == list(range(25))

    original = NoPaddingDistributedBatchSampler(25, 4, 0, 4, data_seed=42)
    iterator = iter(original)
    first_batch = next(iterator)
    state = original.state_dict()
    resumed = NoPaddingDistributedBatchSampler(25, 4, 0, 4, data_seed=42)
    resumed.load_state_dict(state)
    assert first_batch + list(itertools.chain.from_iterable(resumed)) == list(
        itertools.chain.from_iterable(NoPaddingDistributedBatchSampler(25, 4, 0, 4, data_seed=42))
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
