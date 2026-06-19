"""Dataset loading and preprocessing helpers for training runs."""

from __future__ import annotations

import os
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


class DataError(ValueError):
    """Raised when a dataset cannot support the planned training flow."""


def _log_dataset_cache_context(dataset_name: str, dataset_split: str) -> None:
    hf_home = os.environ.get("HF_HOME")
    hf_datasets_cache = os.environ.get("HF_DATASETS_CACHE")
    transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
    print(
        "[dataset] "
        f"name={dataset_name} split={dataset_split} "
        f"HF_HOME={hf_home or 'unset'} "
        f"HF_DATASETS_CACHE={hf_datasets_cache or 'unset'} "
        f"TRANSFORMERS_CACHE={transformers_cache or 'unset'}",
        flush=True,
    )


def load_text_dataset(
    dataset_name: str,
    dataset_split: str,
    dataset_config_name: str | None = None,
    sample_limit: int | None = None,
    seed: int | None = None,
    text_column: str = "text",
    shuffle: bool = True,
):
    _log_dataset_cache_context(dataset_name, dataset_split)
    if dataset_config_name:
        dataset = load_dataset(dataset_name, dataset_config_name, split=dataset_split)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split)
    print(
        "[dataset] "
        f"loaded cache_files={getattr(dataset, 'cache_files', None)}",
        flush=True,
    )
    return prepare_text_dataset(
        dataset,
        sample_limit=sample_limit,
        seed=seed,
        text_column=text_column,
        shuffle=shuffle,
    )


def prepare_text_dataset(
    dataset,
    sample_limit: int | None = None,
    seed: int | None = None,
    text_column: str = "text",
    shuffle: bool = True,
):
    if text_column not in dataset.column_names:
        raise DataError(f"Dataset does not contain text column: {text_column}")

    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    if sample_limit is not None:
        dataset = dataset.select(range(min(sample_limit, len(dataset))))

    return dataset


def tokenize_text_dataset(
    dataset,
    tokenizer,
    context_length: int,
    text_column: str = "text",
    num_proc: int = 1,
    remove_source_columns: bool = True,
):
    if text_column not in dataset.column_names:
        raise DataError(f"Dataset does not contain text column: {text_column}")

    if getattr(tokenizer, "pad_token", None) is None and getattr(
        tokenizer,
        "eos_token",
        None,
    ) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            padding="max_length",
            max_length=context_length,
        )

    map_kwargs = {"batched": True}
    if num_proc and num_proc > 1:
        map_kwargs["num_proc"] = num_proc
    if remove_source_columns:
        map_kwargs["remove_columns"] = dataset.column_names

    tokenized_dataset = dataset.map(tokenize_batch, **map_kwargs)
    print(
        "[dataset] "
        f"tokenized cache_files={getattr(tokenized_dataset, 'cache_files', None)}",
        flush=True,
    )
    return tokenized_dataset


def load_and_tokenize_dataset(
    config: dict[str, Any],
    tokenizer,
    text_column: str = "text",
    num_proc: int = 1,
    shuffle: bool = True,
):
    run = config["run"]
    dataset_config = config["dataset"]
    model_config = config["model"]

    dataset = load_text_dataset(
        dataset_config["dataset_name"],
        dataset_config["dataset_split"],
        dataset_config_name=dataset_config.get("dataset_config_name"),
        sample_limit=dataset_config.get("sample_limit"),
        seed=run.get("seed"),
        text_column=text_column,
        shuffle=shuffle,
    )
    return tokenize_text_dataset(
        dataset,
        tokenizer,
        context_length=model_config["context_length"],
        text_column=text_column,
        num_proc=num_proc,
    )


def split_train_eval_dataset(dataset, eval_example_count: int):
    eval_size = min(eval_example_count, len(dataset))
    eval_dataset = dataset.select(range(eval_size))
    train_dataset = dataset.select(range(eval_size, len(dataset)))
    return train_dataset, eval_dataset


def build_dataloaders(
    config: dict[str, Any],
    tokenized_dataset,
    device: torch.device,
    distributed_context=None,
):
    training = config["training"]
    batch_size = training["batch_size_per_process"]
    eval_batches = training.get("eval_batches", 1)
    eval_example_count = max(1, eval_batches * batch_size)

    train_dataset, eval_dataset = split_train_eval_dataset(
        tokenized_dataset,
        eval_example_count,
    )
    if len(train_dataset) == 0:
        train_dataset = eval_dataset

    pin_memory = device.type == "cuda"
    train_sampler = build_distributed_sampler(
        train_dataset,
        distributed_context,
        shuffle=True,
        seed=config["run"].get("seed"),
    )
    eval_sampler = build_distributed_sampler(
        eval_dataset,
        distributed_context,
        shuffle=False,
        seed=config["run"].get("seed"),
    )
    train_dataloader = build_language_model_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=training.get("dataloader_num_workers", 0),
        pin_memory=pin_memory,
    )
    eval_dataloader = build_language_model_dataloader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=eval_sampler,
        num_workers=training.get("dataloader_num_workers", 0),
        pin_memory=pin_memory,
    )
    return train_dataloader, eval_dataloader


def build_distributed_sampler(
    dataset,
    distributed_context,
    shuffle: bool,
    seed: int | None,
):
    if distributed_context is None or not distributed_context.enabled:
        return None

    return DistributedSampler(
        dataset,
        num_replicas=distributed_context.world_size,
        rank=distributed_context.rank,
        shuffle=shuffle,
        seed=0 if seed is None else int(seed),
    )


def collate_language_model_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    input_ids = _stack_feature(batch, "input_ids")

    if "attention_mask" in batch[0]:
        attention_mask = _stack_feature(batch, "attention_mask")
    else:
        attention_mask = torch.ones_like(input_ids)

    labels = input_ids.clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_language_model_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = False,
    sampler=None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collate_language_model_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _stack_feature(batch: list[dict[str, Any]], name: str) -> torch.Tensor:
    values = [example[name] for example in batch]
    tensors = [
        value if isinstance(value, torch.Tensor) else torch.tensor(value)
        for value in values
    ]
    return torch.stack(tensors).long()
