from __future__ import annotations

import pytest
import torch

from src.training.distributed import DistributedContext
from src.training.steps import (
    group_optimizer_windows,
    optimizer_window_loss_scale,
    validation_token_thresholds_crossed,
)


def test_optimizer_windows_group_64_microsteps_and_keep_final_48():
    windows = list(group_optimizer_windows(range(176), 64))
    assert [len(window) for window in windows] == [64, 64, 48]
    assert [item for window in windows for item in window] == list(range(176))


def test_accumulated_uneven_microbatches_equal_one_large_target_weighted_batch():
    parameter = torch.tensor(2.0, requires_grad=True)
    counts = [3, 7, 2]
    coefficients = [1.0, 4.0, 9.0]
    for count, coefficient in zip(counts, coefficients):
        loss = parameter * coefficient
        scale = optimizer_window_loss_scale(
            local_valid_targets=count,
            total_window_valid_targets=sum(counts),
        )
        (loss * scale).backward()
    expected = sum(
        count * coefficient for count, coefficient in zip(counts, coefficients)
    ) / sum(counts)
    assert parameter.grad.item() == pytest.approx(expected)


def test_nested_all_window_scale_averages_granularities_and_fsdp_ranks():
    context = DistributedContext(enabled=True, rank=0, world_size=2)
    scale = optimizer_window_loss_scale(
        local_valid_targets=5,
        total_window_valid_targets=16,
        distributed_context=context,
        granularity_count=8,
    )
    assert scale == pytest.approx(2 * 5 / 16 / 8)


def test_token_validation_cadence_crosses_once_and_resumes_at_next_threshold():
    crossed, next_threshold = validation_token_thresholds_crossed(
        499_000_000,
        500_048_576,
        interval_tokens=500_000_000,
        next_threshold=500_000_000,
    )
    assert crossed == [500_000_000]
    assert next_threshold == 1_000_000_000
    resumed_crossed, resumed_next = validation_token_thresholds_crossed(
        500_048_576,
        501_097_152,
        interval_tokens=500_000_000,
        next_threshold=next_threshold,
    )
    assert resumed_crossed == []
    assert resumed_next == 1_000_000_000
