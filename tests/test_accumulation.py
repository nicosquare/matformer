from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from src.training.checkpointing import build_initial_continuation_state
from src.training.distributed import DistributedContext
from src.training.optimizer_state import PerGranularityOptimizerCollection
from src.training.steps import (
    build_optimizer_and_scheduler,
    group_optimizer_windows,
    optimizer_window_loss_scale,
    train_for_steps,
    validation_token_thresholds_crossed,
)
from src.utils.config import resolve_run_config


class _AccumulationOwnerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.current_granularity = None
        self.current_granularity_pattern = None
        self.forward_owners = []
        self.grad_absent_at_forward = []

    def configure_subnetwork(self, granularity):
        self.current_granularity = str(granularity)

    def forward(self, input_ids, attention_mask=None, labels=None):
        del input_ids, attention_mask, labels
        self.forward_owners.append(self.current_granularity)
        self.grad_absent_at_forward.append(self.weight.grad is None)
        scale = 1.0 if self.current_granularity == "narrow" else 3.0
        return SimpleNamespace(loss=(self.weight * scale).pow(2))


def _per_width_accumulation_config(tmp_path, *, steps=2):
    return resolve_run_config(
        "tests/fixtures/per_granularity_optimizer_smoke.yaml",
        output_dir=tmp_path / "per-granularity-optimizer-smoke-001",
        overrides=[
            "run.continuation.enabled=false",
            "outputs.save_checkpoints=false",
            f"training.max_steps={steps}",
            f"training.token_budget={steps * 64}",
            "training.gradient_accumulation_steps=2",
            "training.batch_size_per_process=1",
            "training.scheduler.name=constant",
            "training.warmup_steps=0",
            "training.eval_interval=0",
            "evaluation.validation=false",
            "evaluation.validation.interval_steps=0",
            "evaluation.validation.run_at_completion=false",
        ],
    )


def _tiny_batches(count):
    return [
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "labels": torch.tensor([[1, 2, 3]], dtype=torch.long),
        }
        for _ in range(count)
    ]


def _assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_state_equal(left_item, right_item)
    else:
        assert left == right


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


def test_one_width_owns_every_microbatch_and_exactly_one_optimizer_commit(tmp_path):
    config = _per_width_accumulation_config(tmp_path)
    model = _AccumulationOwnerModel()
    optimizer, scheduler = build_optimizer_and_scheduler(model, config["training"])
    assert isinstance(optimizer, PerGranularityOptimizerCollection)
    run_state = build_initial_continuation_state(config)

    train_for_steps(
        config,
        model,
        _tiny_batches(4),
        [],
        optimizer,
        scheduler,
        torch.device("cpu"),
        run_state=run_state,
    )

    assert model.forward_owners[0] == model.forward_owners[1]
    assert model.forward_owners[2] == model.forward_owners[3]
    assert model.forward_owners[0] != model.forward_owners[2]
    assert model.grad_absent_at_forward == [True, False, True, False]
    assert optimizer.successful_update_counts == {"narrow": 1, "full": 1}
    assert optimizer.total_successful_updates == 2
    assert run_state["optimizer_update_counts"] == {"narrow": 1, "full": 1}
    assert run_state["last_completed_step"] == 2


def test_precommit_failure_restores_window_and_advances_no_optimizer_state(tmp_path):
    config = _per_width_accumulation_config(tmp_path, steps=2)
    model = _AccumulationOwnerModel()
    optimizer, scheduler = build_optimizer_and_scheduler(model, config["training"])
    assert isinstance(optimizer, PerGranularityOptimizerCollection)
    run_state = build_initial_continuation_state(config)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    clock_before = copy.deepcopy(scheduler.state_dict())

    def fail_before_commit(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("injected optimizer failure")

    for entry in optimizer.entries:
        entry.optimizer.step = fail_before_commit

    with pytest.raises(RuntimeError, match="injected optimizer failure"):
        train_for_steps(
            config,
            model,
            _tiny_batches(2),
            [],
            optimizer,
            scheduler,
            torch.device("cpu"),
            run_state=run_state,
        )

    _assert_state_equal(optimizer.state_dict(), optimizer_before)
    _assert_state_equal(scheduler.state_dict(), clock_before)
    assert optimizer.successful_update_counts == {"narrow": 0, "full": 0}
    assert optimizer.total_successful_updates == 0
    assert run_state["last_completed_step"] == 0
    assert model.weight.grad is None
