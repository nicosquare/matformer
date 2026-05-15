"""Config-driven training flow for MatFormer reproduction runs."""

from __future__ import annotations

from contextlib import contextmanager
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data.distributed import DistributedSampler
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
from training.distributed import (
    barrier,
    prepare_distributed_context,
    should_write_shared_artifact,
    wrap_model_for_distributed,
)
from utils.config import resolve_run_config, resolve_training_length_for_world_size
from utils.heartbeats import HeartbeatCadence, HeartbeatWriter
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

    distributed_context = prepare_distributed_context(config, device=device)
    sync_config_with_distributed_context(config, distributed_context)
    heartbeat_writer = build_heartbeat_writer(config, distributed_context)

    with heartbeat_stage(heartbeat_writer, "artifact_writing"):
        write_config_artifact(config, distributed_context=distributed_context)
    set_random_seed(run.get("seed"))

    device = torch.device(distributed_context.device)

    try:
        with heartbeat_stage(heartbeat_writer, "model_initialization"):
            if model is None:
                model = build_model(config)
            model = model.to(device)

        with heartbeat_stage(heartbeat_writer, "fsdp_wrapping"):
            model = wrap_model_for_distributed(model, distributed_context)

        if tokenized_dataset is None:
            if tokenizer is None:
                with heartbeat_stage(heartbeat_writer, "tokenizer_loading"):
                    tokenizer = load_tokenizer(config)
            with heartbeat_stage(heartbeat_writer, "dataset_loading_preprocessing"):
                tokenized_dataset = load_and_tokenize_dataset(
                    config,
                    tokenizer,
                    num_proc=training.get("preprocess_num_proc", 1),
                )

        with heartbeat_stage(heartbeat_writer, "dataloader_creation"):
            train_dataloader, eval_dataloader = build_dataloaders(
                config,
                tokenized_dataset,
                device,
                distributed_context=distributed_context,
            )
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
            heartbeat_writer=heartbeat_writer,
        )
        extraction_metadata_path = None
        metrics_path = None
        scaling_path = None
        scaling_rows = []
        parameter_counts_by_granularity = {}

        if should_write_shared_artifact(distributed_context):
            emit_checkpointing_placeholder_if_needed(config, heartbeat_writer)
            with heartbeat_stage(heartbeat_writer, "artifact_writing"):
                extraction_metadata_path = write_extraction_metadata_if_nested(
                    config,
                    model,
                    output_dir,
                    distributed_context=distributed_context,
                )
                metrics_path = write_metrics_csv(
                    output_dir,
                    metrics_rows,
                    distributed_context=distributed_context,
                )
                parameter_counts_by_granularity = build_parameter_counts_by_granularity(
                    model,
                    config["model"]["granularities"],
                )
                scaling_rows = build_scaling_result_rows(
                    config,
                    metrics_rows,
                    parameter_counts_by_granularity,
                )
                scaling_path = write_scaling_results_csv(
                    output_dir,
                    scaling_rows,
                    distributed_context=distributed_context,
                )

        training_outcome = summarize_training_outcome(config, metrics_rows)
        tokens_seen = training_outcome["tokens_seen"]
        extra_summary_fields = {
            "steps_completed": training_outcome["steps_completed"],
            "stop_reason": training_outcome["stop_reason"],
            "granularities": config["model"]["granularities"],
            "granularity_sampling": training.get("granularity_sampling", "all"),
            "parameter_counts_by_granularity": parameter_counts_by_granularity,
            **distributed_summary_fields(distributed_context),
        }
        if metrics_path is not None:
            extra_summary_fields["metrics_path"] = str(metrics_path)
        if scaling_path is not None:
            extra_summary_fields["scaling_results_path"] = str(scaling_path)
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
        with heartbeat_stage(heartbeat_writer, "artifact_writing"):
            summary_path = write_run_summary(
                output_dir,
                summary,
                distributed_context=distributed_context,
            )
        barrier(distributed_context)

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
        with heartbeat_stage(heartbeat_writer, "artifact_writing"):
            write_failed_run_summary(
                config,
                str(error),
                output_dir=output_dir,
                distributed_context=distributed_context,
            )
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


def sync_config_with_distributed_context(config: dict[str, Any], context) -> None:
    resolve_training_length_for_world_size(
        config,
        effective_world_size=context.world_size,
        world_size_source="distributed_context" if context.enabled else "single_process",
    )
    distributed_config = config["training"].setdefault("distributed", {})
    distributed_config.update(
        {
            "enabled": bool(context.enabled),
            "strategy": context.strategy,
            "rank": context.rank,
            "local_rank": context.local_rank,
            "world_size": context.world_size,
        }
    )


def distributed_summary_fields(context) -> dict[str, Any]:
    return {
        "distributed_strategy": context.strategy,
        "distributed_rank": context.rank,
        "distributed_local_rank": context.local_rank,
        "distributed_world_size": context.world_size,
    }


class NoopHeartbeatWriter:
    path = None

    def stage_start(self, stage: str, **fields: Any):
        return None

    def stage_complete(self, stage: str, **fields: Any):
        return None

    def heartbeat(self, stage: str, **fields: Any):
        return None


def build_heartbeat_writer(config: dict[str, Any], distributed_context):
    training = config["training"]
    heartbeat_enabled = training.get("heartbeat_enabled", True)
    if not heartbeat_enabled or not should_write_shared_artifact(distributed_context):
        return NoopHeartbeatWriter()

    run = config["run"]
    return HeartbeatWriter(
        output_dir=run["output_dir"],
        run_id=run["run_id"],
        rank=distributed_context.rank,
        world_size=distributed_context.world_size,
    )


@contextmanager
def heartbeat_stage(heartbeat_writer, stage: str, **fields: Any):
    heartbeat_writer.stage_start(stage, **fields)
    try:
        yield
    finally:
        heartbeat_writer.stage_complete(stage, **fields)


def heartbeat_training_fields(
    config: dict[str, Any],
    step: int | None = None,
    tokens_seen: int | None = None,
    latest_loss: float | None = None,
    tokens_per_second: float | None = None,
    peak_gpu_memory_bytes: int | None = None,
    eta_seconds: float | None = None,
) -> dict[str, Any]:
    training = config["training"]
    return {
        "step": step,
        "derived_max_steps": training.get("derived_max_steps"),
        "tokens_seen": tokens_seen,
        "token_budget": training.get("token_budget"),
        "latest_loss": latest_loss,
        "tokens_per_second": tokens_per_second,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "eta_seconds": eta_seconds,
    }


def build_heartbeat_cadence(config: dict[str, Any]) -> HeartbeatCadence:
    training = config["training"]
    return HeartbeatCadence(
        step_interval=training.get("heartbeat_step_interval", 10),
        time_interval_seconds=training.get("heartbeat_time_interval_seconds", 60.0),
    )


def maybe_emit_training_heartbeat(
    heartbeat_writer,
    heartbeat_cadence: HeartbeatCadence,
    config: dict[str, Any],
    step: int,
    tokens_seen: int,
    latest_loss: float,
    tokens_per_second: float | None,
    peak_gpu_memory_bytes: int,
) -> None:
    now = time.time()
    if not heartbeat_cadence.should_emit(step=step, now=now):
        return

    heartbeat_writer.heartbeat(
        "training",
        **heartbeat_training_fields(
            config,
            step=step,
            tokens_seen=tokens_seen,
            latest_loss=latest_loss,
            tokens_per_second=tokens_per_second,
            peak_gpu_memory_bytes=peak_gpu_memory_bytes,
            eta_seconds=estimate_eta_seconds(
                config,
                tokens_seen=tokens_seen,
                tokens_per_second=tokens_per_second,
            ),
        ),
    )
    heartbeat_cadence.mark_emitted(step=step, now=now)


def estimate_eta_seconds(
    config: dict[str, Any],
    tokens_seen: int,
    tokens_per_second: float | None,
) -> float | None:
    if tokens_per_second is None or tokens_per_second <= 0:
        return None
    remaining_tokens = max(config["training"]["token_budget"] - tokens_seen, 0)
    return remaining_tokens / tokens_per_second


def emit_checkpointing_placeholder_if_needed(
    config: dict[str, Any],
    heartbeat_writer,
) -> None:
    if not config.get("outputs", {}).get("save_checkpoints", False):
        return
    with heartbeat_stage(
        heartbeat_writer,
        "checkpointing",
        extra_fields={"status": "not_implemented_for_config_driven_loop"},
    ):
        return


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


def train_for_steps(
    config: dict[str, Any],
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    scheduler,
    device: torch.device,
    heartbeat_writer=None,
) -> list[dict[str, Any]]:
    training = config["training"]
    granularities = config["model"]["granularities"]
    token_budget = training["token_budget"]
    max_steps = training["max_steps"]
    eval_interval = training.get("eval_interval", 0)

    metrics_rows = []
    tokens_seen = 0
    start_time = time.time()
    step = 0
    epoch = 0
    heartbeat_writer = heartbeat_writer or NoopHeartbeatWriter()
    heartbeat_cadence = build_heartbeat_cadence(config)

    model.train()
    with heartbeat_stage(heartbeat_writer, "training"):
        while step < max_steps and tokens_seen < token_budget:
            set_dataloader_epoch(train_dataloader, epoch)
            epoch += 1
            made_progress = False
            for batch in train_dataloader:
                if step >= max_steps or tokens_seen >= token_budget:
                    break

                made_progress = True
                step += 1
                batch = move_batch_to_device(batch, device)
                batch_tokens = count_batch_tokens(batch)
                tokens_seen += batch_tokens

                optimizer.zero_grad(set_to_none=True)

                selected_granularities = select_training_granularities(
                    config,
                    granularities,
                    device,
                )
                granularity_losses = []
                for granularity in selected_granularities:
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
                peak_memory_bytes = current_peak_memory_bytes(device)
                latest_loss = float(combined_loss.detach().cpu().item())
                tokens_per_second = tokens_seen / elapsed if elapsed > 0 else None
                maybe_emit_training_heartbeat(
                    heartbeat_writer,
                    heartbeat_cadence,
                    config,
                    step=step,
                    tokens_seen=tokens_seen,
                    latest_loss=latest_loss,
                    tokens_per_second=tokens_per_second,
                    peak_gpu_memory_bytes=peak_memory_bytes,
                )

                for granularity, loss in granularity_losses:
                    metrics_rows.append(
                        build_training_metric_row(
                            config,
                            step=step,
                            granularity=granularity,
                            loss=float(loss.detach().cpu().item()),
                            tokens_seen=tokens_seen,
                            wall_clock_seconds=elapsed,
                            peak_memory_bytes=peak_memory_bytes,
                        )
                    )

                if eval_interval > 0 and step % eval_interval == 0:
                    with heartbeat_stage(
                        heartbeat_writer,
                        "validation",
                        **heartbeat_training_fields(
                            config,
                            step=step,
                            tokens_seen=tokens_seen,
                        ),
                    ):
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
                            tokens_per_second=tokens_per_second,
                            peak_memory_bytes=peak_memory_bytes,
                        )
                    )

                if step >= max_steps or tokens_seen >= token_budget:
                    break
            if not made_progress:
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
        heartbeat_writer=heartbeat_writer,
    )

    return metrics_rows


def select_training_granularities(
    config: dict[str, Any],
    granularities: list[str],
    device: torch.device,
) -> list[str]:
    sampling_mode = config["training"].get("granularity_sampling", "all")
    if sampling_mode == "all":
        return list(granularities)
    if sampling_mode == "random":
        selected_index = select_random_granularity_index(
            granularity_count=len(granularities),
            device=device,
        )
        return [granularities[selected_index]]
    raise ValueError(f"Unknown granularity sampling mode: {sampling_mode}")


def select_random_granularity_index(
    granularity_count: int,
    device: torch.device,
) -> int:
    if granularity_count <= 0:
        raise ValueError("granularity_count must be positive")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        selected_index = torch.empty((), dtype=torch.long, device=device)
        if torch.distributed.get_rank() == 0:
            selected_index.fill_(random.randrange(granularity_count))
        torch.distributed.broadcast(selected_index, src=0)
        return int(selected_index.item())

    return random.randrange(granularity_count)


def set_dataloader_epoch(dataloader, epoch: int) -> None:
    sampler = getattr(dataloader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def summarize_training_outcome(
    config: dict[str, Any],
    metrics_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    training_rows = [row for row in metrics_rows if row["split"] == "train"]
    if not training_rows:
        return {
            "steps_completed": 0,
            "tokens_seen": 0,
            "stop_reason": "not_started",
        }

    steps_completed = max(int(row["step"]) for row in training_rows)
    tokens_seen = max(int(row["tokens_seen"]) for row in training_rows)
    return {
        "steps_completed": steps_completed,
        "tokens_seen": tokens_seen,
        "stop_reason": stop_reason_for_training(
            config,
            tokens_seen=tokens_seen,
            steps_completed=steps_completed,
        ),
    }


def stop_reason_for_training(
    config: dict[str, Any],
    tokens_seen: int,
    steps_completed: int,
) -> str:
    training = config["training"]
    if steps_completed == 0:
        return "not_started"
    if tokens_seen >= training["token_budget"]:
        return "token_budget_reached"
    return "max_steps_reached_before_token_budget"


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
    heartbeat_writer=None,
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
    heartbeat_writer = heartbeat_writer or NoopHeartbeatWriter()
    with heartbeat_stage(
        heartbeat_writer,
        "validation",
        **heartbeat_training_fields(
            config,
            step=step,
            tokens_seen=tokens_seen,
        ),
    ):
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
    distributed_context=None,
) -> Path | None:
    if config["run"]["model_family"] != "nested":
        return None

    metadata = build_extraction_metadata(config, model)
    return write_json_artifact(
        output_dir / "extraction_metadata.json",
        metadata,
        distributed_context=distributed_context,
    )


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
