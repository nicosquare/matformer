from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

from evaluation.validation import (
    evaluate_validation_per_granularity,
    perplexity_from_loss,
    validation_results_to_metric_rows,
)
from training.data import (
    DataError,
    build_language_model_dataloader,
    collate_language_model_batch,
    load_text_dataset,
    prepare_text_dataset,
    split_train_eval_dataset,
    tokenize_text_dataset,
)


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
    assert torch.equal(batch["labels"], batch["input_ids"])


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

    monkeypatch.setattr("training.data.load_dataset", fake_load_dataset)

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
    assert results[0]["tokens_seen"] == 5
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
    assert rows[0]["tokens_seen"] == 5
    assert rows[0]["content_tokens_seen"] == 5
    assert rows[1]["granularity"] == "xl"
    assert rows[0]["peak_memory_bytes"] == 2048
