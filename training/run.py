"""Config-driven training flow for MatFormer reproduction runs."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, get_scheduler

from evaluation.validation import (
    configure_model_granularity,
    evaluate_validation_per_granularity,
    move_batch_to_device,
    perplexity_from_loss,
    validation_results_to_metric_rows,
)
from modified_llama import ModifiedLlamaForCausalLM, get_ffn_prefix_metadata
from training.data import (
    build_language_model_dataloader,
    load_and_tokenize_dataset,
    split_train_eval_dataset,
)
from utils.config import resolve_run_config
from utils.metrics import (
    build_parameter_counts_by_granularity,
    build_run_summary,
    build_scaling_result_rows,
    write_config_artifact,
    write_failed_run_summary,
    write_json_artifact,
    write_metrics_csv,
    write_run_summary,
    write_scaling_results_csv,
)


def run_from_config_path(
    config_path: str | Path,
    run_id: str | None = None,
    overrides: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = resolve_run_config(
        config_path,
        run_id=run_id,
        overrides=overrides,
        output_dir=output_dir,
    )
    return run_training(config)


def run_training(
    config: dict[str, Any],
    model=None,
    tokenizer=None,
    tokenized_dataset=None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    run = config["run"]
    training = config["training"]
    output_dir = Path(run["output_dir"])

    write_config_artifact(config)
    set_random_seed(run.get("seed"))

    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device(device)

    try:
        if model is None:
            model = build_model(config)
        model = model.to(device)

        if tokenized_dataset is None:
            if tokenizer is None:
                tokenizer = load_tokenizer(config)
            tokenized_dataset = load_and_tokenize_dataset(
                config,
                tokenizer,
                num_proc=training.get("preprocess_num_proc", 1),
            )

        train_dataloader, eval_dataloader = build_dataloaders(config, tokenized_dataset, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training["learning_rate"],
        )
        scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=training.get("warmup_steps", 0),
            num_training_steps=training["max_steps"],
        )

        metrics_rows = train_for_steps(
            config,
            model,
            train_dataloader,
            eval_dataloader,
            optimizer,
            scheduler,
            device,
        )
        extraction_metadata_path = write_extraction_metadata_if_nested(
            config,
            model,
            output_dir,
        )
        metrics_path = write_metrics_csv(output_dir, metrics_rows)
        parameter_counts_by_granularity = build_parameter_counts_by_granularity(
            model,
            config["model"]["granularities"],
        )
        scaling_rows = build_scaling_result_rows(
            config,
            metrics_rows,
            parameter_counts_by_granularity,
        )
        scaling_path = write_scaling_results_csv(output_dir, scaling_rows)

        tokens_seen = max(
            row["tokens_seen"]
            for row in metrics_rows
            if row["split"] == "train"
        )
        extra_summary_fields = {
            "metrics_path": str(metrics_path),
            "scaling_results_path": str(scaling_path),
            "steps_completed": training["max_steps"],
            "granularities": config["model"]["granularities"],
            "parameter_counts_by_granularity": parameter_counts_by_granularity,
        }
        if extraction_metadata_path is not None:
            extra_summary_fields["extraction_metadata_path"] = str(
                extraction_metadata_path
            )

        summary = build_run_summary(
            config,
            tokens_seen=tokens_seen,
            notes=["completed config-driven training loop"],
            extra_fields=extra_summary_fields,
        )
        summary_path = write_run_summary(output_dir, summary)

        return {
            "config": config,
            "metrics_path": metrics_path,
            "scaling_path": scaling_path,
            "summary_path": summary_path,
            "metrics_rows": metrics_rows,
            "scaling_rows": scaling_rows,
            "parameter_counts_by_granularity": parameter_counts_by_granularity,
        }
    except Exception as error:
        write_failed_run_summary(config, str(error), output_dir=output_dir)
        raise


def build_model(config: dict[str, Any]):
    llama_config = build_llama_config(config)
    if config["run"]["model_family"] == "standalone":
        return LlamaForCausalLM(llama_config)
    return ModifiedLlamaForCausalLM(llama_config)


def build_llama_config(config: dict[str, Any]) -> LlamaConfig:
    model = config["model"]
    return LlamaConfig(
        vocab_size=model["vocab_size_assumption"],
        hidden_size=model["hidden_size"],
        intermediate_size=model["intermediate_size"],
        num_hidden_layers=model["num_layers"],
        num_attention_heads=model["num_attention_heads"],
        num_key_value_heads=model["num_attention_heads"],
        max_position_embeddings=model["context_length"],
        tie_word_embeddings=False,
        use_cache=False,
    )


def load_tokenizer(config: dict[str, Any]):
    model = config["model"]
    dataset = config["dataset"]
    tokenizer_name = dataset.get("tokenizer_name") or model.get("tokenizer_name")
    tokenizer_name = tokenizer_name or model["base_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_dataloaders(config: dict[str, Any], tokenized_dataset, device: torch.device):
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
    train_dataloader = build_language_model_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=training.get("dataloader_num_workers", 0),
        pin_memory=pin_memory,
    )
    eval_dataloader = build_language_model_dataloader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=training.get("dataloader_num_workers", 0),
        pin_memory=pin_memory,
    )
    return train_dataloader, eval_dataloader


def train_for_steps(
    config: dict[str, Any],
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    scheduler,
    device: torch.device,
) -> list[dict[str, Any]]:
    training = config["training"]
    granularities = config["model"]["granularities"]
    max_steps = training["max_steps"]
    eval_interval = training.get("eval_interval", 0)

    metrics_rows = []
    tokens_seen = 0
    start_time = time.time()
    step = 0

    model.train()
    while step < max_steps:
        for batch in train_dataloader:
            step += 1
            batch = move_batch_to_device(batch, device)
            batch_tokens = count_batch_tokens(batch)
            tokens_seen += batch_tokens

            optimizer.zero_grad(set_to_none=True)

            granularity_losses = []
            for granularity in granularities:
                configure_model_granularity(model, granularity)
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
                granularity_losses.append((granularity, outputs.loss))

            combined_loss = torch.stack(
                [loss for _, loss in granularity_losses]
            ).mean()
            combined_loss.backward()
            optimizer.step()
            scheduler.step()

            elapsed = time.time() - start_time
            for granularity, loss in granularity_losses:
                metrics_rows.append(
                    build_training_metric_row(
                        config,
                        step=step,
                        granularity=granularity,
                        loss=float(loss.detach().cpu().item()),
                        tokens_seen=tokens_seen,
                        wall_clock_seconds=elapsed,
                        peak_memory_bytes=current_peak_memory_bytes(device),
                    )
                )

            if eval_interval > 0 and step % eval_interval == 0:
                validation_results = evaluate_validation_per_granularity(
                    model,
                    eval_dataloader,
                    granularities=granularities,
                    device=device,
                )
                metrics_rows.extend(
                    validation_results_to_metric_rows(
                        validation_results,
                        config,
                        step=step,
                        wall_clock_seconds=elapsed,
                        tokens_per_second=tokens_seen / elapsed if elapsed > 0 else None,
                        peak_memory_bytes=current_peak_memory_bytes(device),
                    )
                )

            if step >= max_steps:
                break

    append_final_validation_if_needed(
        metrics_rows,
        config,
        model,
        eval_dataloader,
        granularities=granularities,
        device=device,
        step=step,
        tokens_seen=tokens_seen,
        start_time=start_time,
    )

    return metrics_rows


def append_final_validation_if_needed(
    metrics_rows: list[dict[str, Any]],
    config: dict[str, Any],
    model,
    eval_dataloader,
    granularities: list[str],
    device: torch.device,
    step: int,
    tokens_seen: int,
    start_time: float,
) -> None:
    if not config.get("evaluation", {}).get("validation", False):
        return
    has_final_validation = any(
        row["split"] == "validation" and row["step"] == step
        for row in metrics_rows
    )
    if has_final_validation:
        return

    elapsed = time.time() - start_time
    validation_results = evaluate_validation_per_granularity(
        model,
        eval_dataloader,
        granularities=granularities,
        device=device,
    )
    metrics_rows.extend(
        validation_results_to_metric_rows(
            validation_results,
            config,
            step=step,
            wall_clock_seconds=elapsed,
            tokens_per_second=tokens_seen / elapsed if elapsed > 0 else None,
            peak_memory_bytes=current_peak_memory_bytes(device),
        )
    )


def write_extraction_metadata_if_nested(
    config: dict[str, Any],
    model,
    output_dir: Path,
) -> Path | None:
    if config["run"]["model_family"] != "nested":
        return None

    metadata = build_extraction_metadata(config, model)
    return write_json_artifact(output_dir / "extraction_metadata.json", metadata)


def build_extraction_metadata(config: dict[str, Any], model) -> dict[str, Any]:
    run = config["run"]
    model_config = config["model"]
    configured_granularities = model_config["granularities"]
    prefix_metadata = prefix_metadata_by_granularity(model, model_config)

    return {
        "run_id": run["run_id"],
        "phase_id": run["phase_id"],
        "model_family": run["model_family"],
        "model_size_label": run["model_size_label"],
        "granularities": [
            build_granularity_extraction_metadata(
                granularity,
                configured_granularities,
                prefix_metadata[granularity],
            )
            for granularity in configured_granularities
        ],
    }


def prefix_metadata_by_granularity(
    model,
    model_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    target = model.module if hasattr(model, "module") else model
    metadata = getattr(target, "ffn_prefix_metadata", None)
    if metadata is None:
        metadata = get_ffn_prefix_metadata(model_config["intermediate_size"])

    return {entry["name"]: dict(entry) for entry in metadata}


def build_granularity_extraction_metadata(
    granularity: str,
    configured_granularities: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    granularity_index = configured_granularities.index(granularity)
    return {
        "granularity": granularity,
        "display_name": metadata["display_name"],
        "ffn_ratio": metadata["ffn_ratio"],
        "full_intermediate_fraction": metadata["full_intermediate_fraction"],
        "prefix_width": metadata["prefix_width"],
        "strict_prefix_of": configured_granularities[granularity_index + 1 :],
    }


def build_training_metric_row(
    config: dict[str, Any],
    step: int,
    granularity: str,
    loss: float,
    tokens_seen: int,
    wall_clock_seconds: float,
    peak_memory_bytes: int,
) -> dict[str, Any]:
    run = config["run"]
    tokens_per_second = tokens_seen / wall_clock_seconds if wall_clock_seconds > 0 else None
    return {
        "run_id": run["run_id"],
        "step": step,
        "split": "train",
        "model_family": run["model_family"],
        "model_size_label": run["model_size_label"],
        "granularity": granularity,
        "loss": loss,
        "perplexity": perplexity_from_loss(loss),
        "tokens_seen": tokens_seen,
        "wall_clock_seconds": wall_clock_seconds,
        "tokens_per_second": tokens_per_second,
        "peak_memory_bytes": peak_memory_bytes,
    }


def select_training_granularity(granularities: list[str], step: int) -> str:
    return granularities[(step - 1) % len(granularities)]


def count_batch_tokens(batch: dict[str, torch.Tensor]) -> int:
    if "attention_mask" in batch and batch["attention_mask"] is not None:
        return int(batch["attention_mask"].sum().item())
    return int((batch["labels"] != -100).sum().item())


def current_peak_memory_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return 0


def set_random_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
