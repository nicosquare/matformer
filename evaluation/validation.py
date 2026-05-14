"""Validation loss and perplexity helpers."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist


def perplexity_from_loss(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for name, value in batch.items()
    }


def configure_model_granularity(model, granularity: str | None) -> None:
    if granularity is None:
        return

    target = model.module if hasattr(model, "module") else model
    configure_subnetwork = getattr(target, "configure_subnetwork", None)
    if configure_subnetwork is not None:
        configure_subnetwork(granularity)


def evaluate_validation_loss(
    model,
    dataloader,
    device: torch.device | str,
    granularity: str | None = None,
    distributed: bool = False,
) -> dict[str, float | int | str | None]:
    was_training = model.training
    model.eval()

    loss_sum = 0.0
    batch_count = 0
    token_count = 0

    with torch.no_grad():
        configure_model_granularity(model, granularity)
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
            )
            loss_sum += outputs.loss.detach().float().item()
            batch_count += 1
            token_count += _count_tokens(batch)

    loss_sum, batch_count, token_count = _reduce_validation_stats(
        loss_sum,
        batch_count,
        token_count,
        device,
        distributed,
    )

    if was_training:
        model.train()

    loss = loss_sum / batch_count
    return {
        "granularity": granularity,
        "loss": loss,
        "perplexity": perplexity_from_loss(loss),
        "tokens_seen": token_count,
    }


def evaluate_validation_per_granularity(
    model,
    dataloader,
    granularities: list[str],
    device: torch.device | str,
    distributed: bool = False,
) -> list[dict[str, float | int | str | None]]:
    return [
        evaluate_validation_loss(
            model,
            dataloader,
            device=device,
            granularity=granularity,
            distributed=distributed,
        )
        for granularity in granularities
    ]


def validation_results_to_metric_rows(
    results: list[dict[str, Any]],
    config: dict[str, Any],
    step: int,
    split: str = "validation",
    wall_clock_seconds: float | None = None,
    tokens_per_second: float | None = None,
    peak_memory_bytes: int | None = None,
) -> list[dict[str, Any]]:
    run = config["run"]
    rows = []
    for result in results:
        rows.append(
            {
                "run_id": run["run_id"],
                "step": step,
                "split": split,
                "model_family": run["model_family"],
                "model_size_label": run["model_size_label"],
                "granularity": result["granularity"],
                "loss": result["loss"],
                "perplexity": result["perplexity"],
                "tokens_seen": result["tokens_seen"],
                "wall_clock_seconds": wall_clock_seconds,
                "tokens_per_second": tokens_per_second,
                "peak_memory_bytes": peak_memory_bytes,
            }
        )
    return rows


def _count_tokens(batch: dict[str, torch.Tensor]) -> int:
    if "attention_mask" in batch and batch["attention_mask"] is not None:
        return int(batch["attention_mask"].sum().item())

    labels = batch["labels"]
    return int((labels != -100).sum().item())


def _reduce_validation_stats(
    loss_sum: float,
    batch_count: int,
    token_count: int,
    device: torch.device | str,
    distributed: bool,
) -> tuple[float, int, int]:
    if batch_count == 0:
        raise ValueError("Validation dataloader produced zero batches")

    if not distributed:
        return loss_sum, batch_count, token_count

    stats = torch.tensor(
        [loss_sum, float(batch_count), float(token_count)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float(stats[0].item()), int(stats[1].item()), int(stats[2].item())
