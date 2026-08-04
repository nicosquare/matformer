"""Dataset loading and preprocessing helpers for training runs."""

from __future__ import annotations

import os
import hashlib
import random
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Sampler

from src.utils.reproducibility import (
    build_comparison_control_signature,
    seed_for,
    stable_hash,
)


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
    metadata_target: dict[str, Any] | None = None,
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
    if metadata_target is not None:
        metadata_target["source_dataset_fingerprint"] = getattr(
            dataset, "_fingerprint", None
        )
        metadata_target["source_dataset_size"] = len(dataset)
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
    keep_in_memory: bool = False,
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

    map_kwargs = {
        "batched": True,
        "keep_in_memory": bool(keep_in_memory),
    }
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
    dataset_config = config["dataset"]
    model_config = config["model"]

    dataset = load_text_dataset(
        dataset_config["dataset_name"],
        dataset_config["dataset_split"],
        dataset_config_name=dataset_config.get("dataset_config_name"),
        sample_limit=dataset_config.get("sample_limit"),
        seed=seed_for(config, "dataset_selection"),
        text_column=text_column,
        shuffle=shuffle,
        metadata_target=dataset_config,
    )
    return tokenize_text_dataset(
        dataset,
        tokenizer,
        context_length=model_config["context_length"],
        text_column=text_column,
        num_proc=num_proc,
        keep_in_memory=dataset_config.get("tokenization_keep_in_memory", False),
    )


def split_train_eval_dataset(
    dataset,
    eval_example_count: int,
    seed: int | None = None,
):
    if len(dataset) < eval_example_count + 1:
        raise DataError(
            "Validation holdout requires at least "
            f"{eval_example_count + 1} usable examples; found {len(dataset)}"
        )
    if seed is None:
        eval_indices = list(range(eval_example_count))
    else:
        eval_indices = sorted(random.Random(seed).sample(range(len(dataset)), eval_example_count))
    eval_index_set = set(eval_indices)
    train_indices = [index for index in range(len(dataset)) if index not in eval_index_set]
    eval_dataset = dataset.select(eval_indices)
    train_dataset = dataset.select(train_indices)
    return train_dataset, eval_dataset


class EpochRandomSampler(Sampler[int]):
    """Epoch-addressable sampler for deterministic single-process training."""

    def __init__(self, dataset, seed: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(_epoch_seed(self.seed, self.epoch, 0, 1))
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.dataset)


class DeterministicDistributedTrainingSampler(Sampler[int]):
    """Deterministic distributed permutation with equal rank lengths."""

    def __init__(self, dataset, seed: int, rank: int, world_size: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        self.num_samples = (len(dataset) + self.world_size - 1) // self.world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(_epoch_seed(self.seed, self.epoch, 0, self.world_size))
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        total_size = self.num_samples * self.world_size
        if len(indices) < total_size:
            indices.extend(indices[: total_size - len(indices)])
        return iter(indices[self.rank:total_size:self.world_size])

    def __len__(self) -> int:
        return self.num_samples


class DistributedValidationSampler(Sampler[int]):
    """Fixed-order validation partition without padding or duplication."""

    def __init__(self, dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.world_size - 1) // self.world_size)


class SeededWorkerInitializer:
    def __init__(self, seed: int, sampler, rank: int, world_size: int):
        self.seed = int(seed)
        self.sampler = sampler
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __call__(self, worker_id: int) -> None:
        epoch = int(getattr(self.sampler, "epoch", 0))
        worker_seed = _epoch_seed(
            self.seed,
            epoch,
            self.rank * 1_000_003 + int(worker_id),
            self.world_size,
        )
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32))
        torch.manual_seed(worker_seed)


def _epoch_seed(seed: int, epoch: int, rank: int, world_size: int) -> int:
    material = f"{seed}|{epoch}|{rank}|{world_size}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def build_dataloaders(
    config: dict[str, Any],
    tokenized_dataset,
    device: torch.device,
    distributed_context=None,
):
    training = config["training"]
    batch_size = training["batch_size_per_process"]
    validation = config["evaluation"]["validation"]
    eval_example_count = int(validation["holdout"]["examples"])
    validation_seed = seed_for(config, "validation_holdout")

    train_dataset, eval_dataset = split_train_eval_dataset(
        tokenized_dataset,
        eval_example_count,
        seed=validation_seed,
    )

    pin_memory = device.type == "cuda"
    rank = int(getattr(distributed_context, "rank", 0))
    world_size = int(getattr(distributed_context, "world_size", 1))
    sampler_seed = seed_for(config, "training_sampler")
    if distributed_context is not None and distributed_context.enabled:
        train_sampler = DeterministicDistributedTrainingSampler(
            train_dataset, sampler_seed, rank, world_size
        )
        eval_sampler = DistributedValidationSampler(eval_dataset, rank, world_size)
    else:
        train_sampler = EpochRandomSampler(train_dataset, sampler_seed)
        eval_sampler = None
    worker_initializer = SeededWorkerInitializer(
        seed_for(config, "dataloader_workers"), train_sampler, rank, world_size
    )
    train_dataloader = build_language_model_dataloader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=training.get("dataloader_num_workers", 0),
        pin_memory=pin_memory,
        worker_init_fn=worker_initializer,
    )
    eval_dataloader = build_language_model_dataloader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=eval_sampler,
        num_workers=training.get("dataloader_num_workers", 0),
        pin_memory=pin_memory,
        worker_init_fn=SeededWorkerInitializer(
            seed_for(config, "dataloader_workers"), eval_sampler, rank, world_size
        ),
    )
    _attach_validation_provenance(
        config,
        tokenized_dataset,
        eval_example_count=eval_example_count,
        validation_seed=validation_seed,
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

    if shuffle:
        return DeterministicDistributedTrainingSampler(
            dataset,
            0 if seed is None else int(seed),
            distributed_context.rank,
            distributed_context.world_size,
        )
    return DistributedValidationSampler(
        dataset,
        distributed_context.rank,
        distributed_context.world_size,
    )


def collate_language_model_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    input_ids = _stack_feature(batch, "input_ids")

    if "attention_mask" in batch[0]:
        attention_mask = _stack_feature(batch, "attention_mask")
    else:
        attention_mask = torch.ones_like(input_ids)

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
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
    worker_init_fn=None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collate_language_model_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )


def _attach_validation_provenance(
    config: dict[str, Any],
    dataset,
    *,
    eval_example_count: int,
    validation_seed: int,
) -> None:
    validation_indices = sorted(
        random.Random(validation_seed).sample(range(len(dataset)), eval_example_count)
    )
    validation_index_set = set(validation_indices)
    training_indices = [
        index for index in range(len(dataset)) if index not in validation_index_set
    ]
    dataset_config = config["dataset"]
    model = config["model"]
    run = config["run"]
    manifest = {
        "dataset_name": dataset_config["dataset_name"],
        "dataset_config_name": dataset_config.get("dataset_config_name"),
        "source_split": dataset_config["dataset_split"],
        "dataset_fingerprint": dataset_config.get(
            "source_dataset_fingerprint", getattr(dataset, "_fingerprint", None)
        ),
        "dataset_size_before_sample_limit": dataset_config.get(
            "source_dataset_size", len(dataset)
        ),
        "dataset_size_before_splitting": len(dataset),
        "root_seed": run["seed"],
        "validation_holdout_seed": validation_seed,
        "split_algorithm": "sha256_seeded_random_sample_sorted_v1",
        "data_split_version": run["reproducibility"]["data_split_version"],
        "validation_indices": validation_indices,
        "validation_indices_hash": stable_hash(validation_indices),
        "training_index_hash": stable_hash(training_indices),
        "validation_example_count": eval_example_count,
        "tokenizer": model.get("tokenizer_name"),
        "context_length": model["context_length"],
        "padding_policy": "max_length_attention_mask_labels_minus_100",
        "aggregation_method": "target_token_weighted_causal_shift_float64",
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    config["validation_manifest_hash"] = manifest["manifest_hash"]
    config["validation_loss_aggregation"] = manifest["aggregation_method"]
    config["_validation_manifest"] = manifest
    signature, inputs = build_comparison_control_signature(config)
    config["comparison_control_signature"] = signature
    config["comparison_control_inputs"] = inputs


def _stack_feature(batch: list[dict[str, Any]], name: str) -> torch.Tensor:
    values = [example[name] for example in batch]
    tensors = [
        value if isinstance(value, torch.Tensor) else torch.tensor(value)
        for value in values
    ]
    return torch.stack(tensors).long()
