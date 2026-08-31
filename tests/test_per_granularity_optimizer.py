from __future__ import annotations

import copy

import pytest
import torch

from src.training.optimizer_state import (
    GlobalSchedulerClock,
    PerGranularityOptimizerCollection,
    build_per_granularity_optimizer_runtime,
)


class _TwoWidthParameters(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        self.wide_only = torch.nn.Parameter(torch.tensor([3.0]))


def _training(optimizer_name: str, *, scheduler_name: str = "constant"):
    optimizer_kwargs = (
        {"betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1}
        if optimizer_name == "adamw"
        else {"momentum": 0.9, "weight_decay": 0.1}
    )
    return {
        "optimizer_name": optimizer_name,
        "optimizer_kwargs": optimizer_kwargs,
        "optimizer_state_scope": "per_granularity",
        "optimizer_state_contract": {
            "ordered_granularities": ["narrow", "full"]
        },
        "resolved_learning_rate": 0.01,
        "scheduler_name": scheduler_name,
        "scheduler_kwargs": {},
        "resolved_warmup_steps": 0,
        "max_steps": 8,
    }


def _commit(
    collection: PerGranularityOptimizerCollection,
    clock: GlobalSchedulerClock,
    owner: str,
) -> None:
    collection.optimizer_for(owner).step()
    clock.step()
    clock.synchronize(collection)
    collection.record_successful_update(owner)


def _assert_nested_state_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_state_equal(left_item, right_item)
    else:
        assert left == right


@pytest.mark.parametrize("optimizer_name", ["adamw", "sgd"])
def test_ordered_width_optimizers_share_parameters_but_keep_distinct_histories(
    optimizer_name,
):
    model = _TwoWidthParameters()
    collection, clock = build_per_granularity_optimizer_runtime(
        model, _training(optimizer_name)
    )

    assert collection.ordered_granularities == ("narrow", "full")
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    assert [
        tuple(
            id(parameter)
            for group in entry.optimizer.param_groups
            for parameter in group["params"]
        )
        for entry in collection.entries
    ] == [parameter_ids, parameter_ids]

    collection.zero_grad(set_to_none=True)
    model.shared.grad = torch.tensor([1.0, -1.0])
    _commit(collection, clock, "narrow")
    collection.zero_grad(set_to_none=True)
    model.shared.grad = torch.tensor([4.0, 2.0])
    model.wide_only.grad = torch.tensor([3.0])
    _commit(collection, clock, "full")

    narrow_state = collection.optimizer_for("narrow").state[model.shared]
    full_state = collection.optimizer_for("full").state[model.shared]
    state_key = "exp_avg" if optimizer_name == "adamw" else "momentum_buffer"
    assert not torch.equal(narrow_state[state_key], full_state[state_key])
    assert collection.successful_update_counts == {"narrow": 1, "full": 1}
    assert collection.total_successful_updates == 2
    assert collection.last_active_granularity == "full"


@pytest.mark.parametrize("optimizer_name", ["adamw", "sgd"])
def test_inactive_parameters_and_nonselected_optimizer_state_are_bitwise_frozen(
    optimizer_name,
):
    model = _TwoWidthParameters()
    collection, clock = build_per_granularity_optimizer_runtime(
        model, _training(optimizer_name)
    )
    full_optimizer = collection.optimizer_for("full")

    collection.zero_grad(set_to_none=True)
    model.shared.grad = torch.tensor([2.0, 2.0])
    model.wide_only.grad = torch.tensor([1.0])
    _commit(collection, clock, "full")
    wide_before = model.wide_only.detach().clone()
    full_state_before = copy.deepcopy(full_optimizer.state_dict())

    collection.zero_grad(set_to_none=True)
    model.shared.grad = torch.tensor([-1.0, 1.0])
    assert model.wide_only.grad is None
    _commit(collection, clock, "narrow")

    assert torch.equal(model.wide_only, wide_before)
    assert model.wide_only not in collection.optimizer_for("narrow").state
    _assert_nested_state_equal(full_optimizer.state_dict(), full_state_before)


@pytest.mark.parametrize("optimizer_name", ["adamw", "sgd"])
def test_present_zero_gradient_keeps_ordinary_optimizer_semantics(optimizer_name):
    model = _TwoWidthParameters()
    collection, clock = build_per_granularity_optimizer_runtime(
        model, _training(optimizer_name)
    )
    before = model.wide_only.detach().clone()

    collection.zero_grad(set_to_none=True)
    model.wide_only.grad = torch.zeros_like(model.wide_only)
    _commit(collection, clock, "narrow")

    assert not torch.equal(model.wide_only, before)
    assert model.wide_only in collection.optimizer_for("narrow").state
