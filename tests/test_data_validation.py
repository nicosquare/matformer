from types import SimpleNamespace
import json

import pytest
import torch
from datasets import Dataset

import src.training.data as training_data
import src.training.run as training_run
from src.evaluation.validation import (
    evaluate_validation_per_granularity,
    perplexity_from_loss,
    validation_results_to_metric_rows,
)
from src.training.data import (
    DataError,
    build_language_model_dataloader,
    collate_language_model_batch,
    load_text_dataset,
    prepare_text_dataset,
    split_train_eval_dataset,
    tokenize_text_dataset,
)
from src.utils.reproducibility import stable_hash
from src.utils.config import resolve_run_config


class TinyTokenizer:
    pad_token = None
    eos_token = "<eos>"

    def __call__(self, texts, truncation, padding, max_length):
        input_ids = []
        attention_masks = []
        for text in texts:
            token_ids = [(ord(char) % 50) + 1 for char in text]
            token_ids = token_ids[:max_length]
            attention_mask = [1] * len(token_ids)

            if padding == "max_length":
                pad_count = max_length - len(token_ids)
                token_ids = token_ids + [0] * pad_count
                attention_mask = attention_mask + [0] * pad_count

            input_ids.append(token_ids)
            attention_masks.append(attention_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
        }


class TinyValidationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.current_granularity = None
        self.loss_by_granularity = {"s": 1.0, "xl": 2.0}

    def configure_subnetwork(self, granularity):
        self.current_granularity = granularity

    def forward(self, input_ids, attention_mask=None, labels=None):
        loss = self.loss_by_granularity[self.current_granularity]
        return SimpleNamespace(loss=torch.tensor(loss, device=input_ids.device))


def test_prepare_tokenize_split_and_collate_text_dataset():
    dataset = Dataset.from_dict(
        {
            "text": ["a", "bb", "ccc"],
            "source": ["one", "two", "three"],
        }
    )
    tokenizer = TinyTokenizer()

    prepared = prepare_text_dataset(
        dataset,
        sample_limit=2,
        text_column="text",
        shuffle=False,
    )
    tokenized = tokenize_text_dataset(
        prepared,
        tokenizer,
        context_length=4,
        text_column="text",
    )
    train_dataset, eval_dataset = split_train_eval_dataset(tokenized, 1)
    batch = collate_language_model_batch([tokenized[0], tokenized[1]])

    assert tokenizer.pad_token == tokenizer.eos_token
    assert len(train_dataset) == 1
    assert len(eval_dataset) == 1
    assert tokenized.column_names == ["input_ids", "attention_mask"]
    assert batch["input_ids"].shape == (2, 4)
    assert batch["labels"].tolist() == [
        [48, -100, -100, -100],
        [49, 49, -100, -100],
    ]


def test_tokenize_text_dataset_can_avoid_arrow_cache(tmp_path):
    source_path = tmp_path / "source"
    Dataset.from_dict({"text": ["one", "two"]}).save_to_disk(source_path)
    disk_dataset = Dataset.load_from_disk(source_path)

    tokenized = tokenize_text_dataset(
        disk_dataset,
        TinyTokenizer(),
        context_length=4,
        keep_in_memory=True,
    )

    assert disk_dataset.cache_files
    assert tokenized.cache_files == []


def test_prepare_text_dataset_requires_text_column():
    dataset = Dataset.from_dict({"body": ["missing text"]})

    with pytest.raises(DataError, match="text column"):
        prepare_text_dataset(dataset, text_column="text")


def test_load_text_dataset_passes_dataset_config_name(monkeypatch):
    calls = {}

    def fake_load_dataset(path, name=None, split=None):
        calls["path"] = path
        calls["name"] = name
        calls["split"] = split
        return Dataset.from_dict({"text": ["a", "bb", "ccc"]})

    monkeypatch.setattr("src.training.data.load_dataset", fake_load_dataset)

    dataset = load_text_dataset(
        "HuggingFaceFW/fineweb",
        "train",
        dataset_config_name="sample-10BT",
        sample_limit=2,
        shuffle=False,
    )

    assert calls == {
        "path": "HuggingFaceFW/fineweb",
        "name": "sample-10BT",
        "split": "train",
    }
    assert len(dataset) == 2


def test_validation_loss_perplexity_and_metric_rows():
    examples = [
        {"input_ids": [1, 2, 0], "attention_mask": [1, 1, 0]},
        {"input_ids": [3, 4, 5], "attention_mask": [1, 1, 1]},
    ]
    dataloader = build_language_model_dataloader(examples, batch_size=1)
    model = TinyValidationModel()
    model.train()

    results = evaluate_validation_per_granularity(
        model,
        dataloader,
        granularities=["s", "xl"],
        device="cpu",
    )

    assert model.training is True
    assert results[0]["loss"] == 1.0
    assert results[0]["perplexity"] == perplexity_from_loss(1.0)
    assert results[0]["tokens_seen"] == 3
    assert results[0]["evaluation_target_tokens"] == 3
    assert results[1]["loss"] == 2.0

    rows = validation_results_to_metric_rows(
        results,
        config={
            "run": {
                "run_id": "debug-nested-001",
                "model_family": "nested",
                "model_size_label": "debug",
            }
        },
        step=10,
        peak_memory_bytes=2048,
    )
    assert rows[0]["run_id"] == "debug-nested-001"
    assert rows[0]["granularity"] == "s"
    assert rows[0]["tokens_seen"] == 3
    assert rows[0]["content_tokens_seen"] == 3
    assert rows[1]["granularity"] == "xl"
    assert rows[0]["peak_memory_bytes"] == 2048
    assert json.loads(rows[0]["granularity_pattern_summary"])[
        "selected_granularities"
    ] == ["s"]
    assert json.loads(rows[1]["granularity_pattern_summary"])[
        "selected_granularities"
    ] == ["xl"]


PROBABILISTIC_ROLE_NAMES = (
    "optimizer_training",
    "controller",
    "ordinary_validation",
    "final_holdout",
)


def _role_partition_dataset(size, *, unusable_indices=()):
    unusable_indices = set(unusable_indices)
    return Dataset.from_dict(
        {
            "source_row_identity": list(range(size)),
            "input_ids": [[index + 1, index + 2, 0] for index in range(size)],
            "attention_mask": [
                [1, 0, 0] if index in unusable_indices else [1, 1, 0]
                for index in range(size)
            ],
        }
    )


def _partition_probabilistic_roles(dataset):
    return training_data.partition_probabilistic_data_roles(
        dataset,
        ordinary_validation_example_count=8,
        ordinary_validation_seed=101,
        controller_seed=202,
        final_holdout_seed=303,
        source_provenance={
            "dataset_name": "controlled/role-partition",
            "dataset_config_name": None,
            "source_split": "train",
            "source_dataset_fingerprint": "controlled-fingerprint",
            "tokenization_identity": "controlled-tokenization-v1",
        },
    )


def _manifest_identity_sets(role_manifests):
    return {
        role: {
            stable_hash(identity)
            for identity in role_manifests[role]["ordered_example_identities"]
        }
        for role in PROBABILISTIC_ROLE_NAMES
    }


def test_probabilistic_four_role_partition_has_fixed_counts_and_six_empty_intersections():
    dataset = _role_partition_dataset(660, unusable_indices={0, 1, 2})

    partition = _partition_probabilistic_roles(dataset)

    role_datasets = partition["datasets"]
    assert len(role_datasets["ordinary_validation"]) == 8
    assert len(role_datasets["controller"]) == 128
    assert len(role_datasets["final_holdout"]) == 512
    assert len(role_datasets["optimizer_training"]) == 9

    identity_sets = _manifest_identity_sets(partition["role_manifests"])
    pairwise_intersections = {
        f"{left}__{right}": identity_sets[left] & identity_sets[right]
        for index, left in enumerate(PROBABILISTIC_ROLE_NAMES)
        for right in PROBABILISTIC_ROLE_NAMES[index + 1 :]
    }
    assert len(pairwise_intersections) == 6
    assert all(not overlap for overlap in pairwise_intersections.values())
    assert partition["parent_manifest"]["pairwise_intersection_counts"] == {
        pair: 0 for pair in pairwise_intersections
    }
    selected_source_rows = {
        identity["source_row_identity"]
        for manifest in partition["role_manifests"].values()
        for identity in manifest["ordered_example_identities"]
    }
    assert not ({0, 1, 2} & selected_source_rows)


def test_probabilistic_four_role_manifests_and_hashes_are_stable():
    dataset = _role_partition_dataset(660)

    first = _partition_probabilistic_roles(dataset)
    second = _partition_probabilistic_roles(dataset)

    assert first["role_manifests"] == second["role_manifests"]
    assert first["parent_manifest"] == second["parent_manifest"]
    for role in PROBABILISTIC_ROLE_NAMES:
        manifest = first["role_manifests"][role]
        assert manifest["role"] == role
        assert manifest["example_count"] == len(manifest["ordered_example_identities"])
        assert len(manifest["example_identity_hash"]) == 64
        assert len(manifest["manifest_hash"]) == 64
    assert len(first["parent_manifest"]["source_pool_hash"]) == 64
    assert len(first["parent_manifest"]["parent_manifest_hash"]) == 64


def test_probabilistic_four_role_partition_rejects_insufficient_usable_data():
    # 8 ordinary + 128 controller + 512 final + one training example are required.
    dataset = _role_partition_dataset(648)

    with pytest.raises(DataError, match="649 usable examples"):
        _partition_probabilistic_roles(dataset)


def test_panelgrad_activates_shared_roles_without_activating_thompson_controller():
    config = resolve_run_config("tests/fixtures/panelgrad_smoke.yaml")

    assert training_data.uses_controller_panel(config) is True
    assert training_data.uses_panelgrad_controller_panel(config) is True
    assert training_run.uses_controller_panel(config) is True
    assert training_run.uses_probabilistic_controller(config) is False


def test_panelgrad_raw_controller_loader_is_replicated_and_final_holdout_is_not_loaded():
    config = resolve_run_config("tests/fixtures/panelgrad_smoke.yaml")
    dataset = _role_partition_dataset(660)
    distributed = SimpleNamespace(enabled=True, rank=1, world_size=2)

    _, _, controller_loader, partition = training_run.prepare_controller_data_roles(
        config,
        dataset,
        torch.device("cpu"),
        distributed_context=distributed,
    )

    assert controller_loader.sampler.__class__.__name__ == "SequentialSampler"
    assert len(controller_loader.dataset) == 128
    assert set(partition["datasets"]) == {
        "optimizer_training",
        "controller",
        "ordinary_validation",
        "final_holdout",
    }
    assert controller_loader.dataset is partition["datasets"]["controller"]
    assert controller_loader.dataset is not partition["datasets"]["final_holdout"]
    assert partition["parent_manifest"]["final_holdout_manifest_hash"] == (
        partition["role_manifests"]["final_holdout"]["manifest_hash"]
    )
    assert all(
        count == 0
        for count in partition["parent_manifest"][
            "pairwise_intersection_counts"
        ].values()
    )


def test_probabilistic_role_overlap_is_rejected_before_consumption():
    partition = _partition_probabilistic_roles(_role_partition_dataset(660))
    manifests = partition["role_manifests"]
    duplicated_identity = manifests["ordinary_validation"][
        "ordered_example_identities"
    ][0]
    manifests["controller"]["ordered_example_identities"][0] = duplicated_identity

    with pytest.raises(DataError, match="overlap") as error:
        training_data.validate_data_role_disjointness(manifests)
    assert "controller" in str(error.value)
    assert "ordinary_validation" in str(error.value)


def test_probabilistic_partition_preserves_existing_ordinary_validation_selection():
    dataset = _role_partition_dataset(660)
    _, legacy_validation = split_train_eval_dataset(dataset, 8, seed=101)

    partition = _partition_probabilistic_roles(dataset)

    ordinary_source_rows = [
        identity["source_row_identity"]
        for identity in partition["role_manifests"]["ordinary_validation"][
            "ordered_example_identities"
        ]
    ]
    assert ordinary_source_rows == legacy_validation["source_row_identity"]
