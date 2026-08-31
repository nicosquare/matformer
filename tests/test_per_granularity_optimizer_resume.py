from __future__ import annotations

import copy

import pytest
import torch

import src.training.checkpointing as training_checkpointing
from src.training.optimizer_state import build_per_granularity_optimizer_runtime
from src.utils.config import ConfigError, resolve_run_config


class _ResumeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        self.wide_only = torch.nn.Parameter(torch.tensor([3.0]))


def _training(optimizer_name: str = "adamw") -> dict:
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
            "schema_version": 1,
            "state_scope": "per_granularity",
            "scheduler_clock": "global_step",
            "ordered_granularities": ["narrow", "full"],
            "optimizer_name": optimizer_name,
            "optimizer_kwargs": optimizer_kwargs,
        },
        "resolved_learning_rate": 0.01,
        "scheduler_name": "constant",
        "scheduler_kwargs": {},
        "resolved_warmup_steps": 0,
        "max_steps": 8,
    }


def _commit(model, collection, clock, owner: str, *, scale: float) -> None:
    collection.zero_grad(set_to_none=True)
    model.shared.grad = torch.tensor([scale, -scale])
    if owner == "full":
        model.wide_only.grad = torch.tensor([scale])
    collection.optimizer_for(owner).step()
    clock.step()
    clock.synchronize(collection)
    collection.record_successful_update(owner)


def _assert_nested_equal(left, right) -> None:
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


@pytest.mark.parametrize("optimizer_name", ["adamw", "sgd"])
def test_collection_and_global_clock_exact_resume_round_trip(optimizer_name):
    source_model = _ResumeModel()
    source, source_clock = build_per_granularity_optimizer_runtime(
        source_model, _training(optimizer_name)
    )
    _commit(source_model, source, source_clock, "narrow", scale=1.0)
    _commit(source_model, source, source_clock, "full", scale=3.0)
    _commit(source_model, source, source_clock, "narrow", scale=2.0)

    restored_model = _ResumeModel()
    restored, restored_clock = build_per_granularity_optimizer_runtime(
        restored_model, _training(optimizer_name)
    )
    restored.load_state_dict(source.state_dict())
    restored_clock.load_state_dict(source_clock.state_dict())
    restored_clock.synchronize(restored)

    _assert_nested_equal(restored.state_dict(), source.state_dict())
    _assert_nested_equal(restored_clock.state_dict(), source_clock.state_dict())
    assert restored.successful_update_counts == {"narrow": 2, "full": 1}
    assert restored.total_successful_updates == restored_clock.position == 3
    assert restored.last_active_granularity == "narrow"


def _mutate_collection_state(case: str, state: dict) -> None:
    if case == "version":
        state["schema_version"] = 99
    elif case == "reordered":
        state["ordered_entries"].reverse()
    elif case == "missing":
        state["ordered_entries"].pop()
    elif case == "extra":
        state["ordered_entries"].append(copy.deepcopy(state["ordered_entries"][0]))
        state["ordered_entries"][-1]["granularity"] = "extra"
    elif case == "negative_count":
        state["successful_update_counts"]["narrow"] = -1
    elif case == "total_mismatch":
        state["total_successful_updates"] += 1
    elif case == "unknown_owner":
        state["last_active_granularity"] = "unknown"
    elif case == "nonfinite_rate":
        state["current_learning_rates"][0] = float("nan")
    elif case in {"shape", "nonfinite_tensor"}:
        optimizer_state = state["ordered_entries"][0]["state_dict"]["state"]
        first_parameter_state = next(iter(optimizer_state.values()))
        tensor_key = next(
            key
            for key, value in first_parameter_state.items()
            if torch.is_tensor(value) and value.ndim > 0
        )
        if case == "shape":
            first_parameter_state[tensor_key] = torch.zeros(3)
        else:
            first_parameter_state[tensor_key].reshape(-1)[0] = float("nan")
    else:  # pragma: no cover - keeps parametrization edits attributable
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "version",
        "reordered",
        "missing",
        "extra",
        "negative_count",
        "total_mismatch",
        "unknown_owner",
        "nonfinite_rate",
        "shape",
        "nonfinite_tensor",
    ],
)
def test_collection_rejects_malformed_state_without_runtime_mutation(case):
    source_model = _ResumeModel()
    source, source_clock = build_per_granularity_optimizer_runtime(
        source_model, _training()
    )
    _commit(source_model, source, source_clock, "narrow", scale=1.0)
    candidate = copy.deepcopy(source.state_dict())
    _mutate_collection_state(case, candidate)

    restored_model = _ResumeModel()
    restored, _ = build_per_granularity_optimizer_runtime(
        restored_model, _training()
    )
    runtime_before = copy.deepcopy(restored.state_dict())
    parameters_before = copy.deepcopy(restored_model.state_dict())

    with pytest.raises(ConfigError):
        restored.load_state_dict(candidate)

    _assert_nested_equal(restored.state_dict(), runtime_before)
    _assert_nested_equal(restored_model.state_dict(), parameters_before)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda state: state.update(position=-1),
        lambda state: state["scheduler_state_dict"].update(last_epoch=float("nan")),
        lambda state: state["carrier_optimizer_state_dict"]["param_groups"][0].update(
            lr=float("inf")
        ),
    ],
)
def test_global_clock_rejects_invalid_state_without_mutation(mutation):
    model = _ResumeModel()
    collection, clock = build_per_granularity_optimizer_runtime(model, _training())
    _commit(model, collection, clock, "narrow", scale=1.0)
    candidate = copy.deepcopy(clock.state_dict())
    mutation(candidate)

    fresh_model = _ResumeModel()
    _, restored_clock = build_per_granularity_optimizer_runtime(
        fresh_model, _training()
    )
    before = copy.deepcopy(restored_clock.state_dict())
    with pytest.raises(ConfigError):
        restored_clock.load_state_dict(candidate)
    _assert_nested_equal(restored_clock.state_dict(), before)


@pytest.mark.parametrize(
    "case",
    [
        "scope",
        "family",
        "hyperparameters",
        "clock",
        "width_order",
        "scheduler",
        "topology",
        "checkpoint_version",
        "collection_version",
        "missing_collection",
        "extra_entry",
    ],
)
def test_checkpoint_contract_mismatch_rejects_before_any_runtime_mutation(
    tmp_path, case
):
    output_dir = tmp_path / "per-granularity-optimizer-smoke-001"
    config = resolve_run_config(
        "tests/fixtures/per_granularity_optimizer_smoke.yaml",
        output_dir=output_dir,
        overrides={
            "training.scheduler.name": "constant",
            "evaluation.validation": False,
        },
    )
    source_model = _ResumeModel()
    source_optimizer, source_scheduler = build_per_granularity_optimizer_runtime(
        source_model, config["training"]
    )
    source_path = output_dir / "checkpoints/latest.pt"
    training_checkpointing.save_model_checkpoint(
        config,
        source_model,
        source_optimizer,
        source_scheduler,
        source_path,
        {
            "checkpoint_status": "latest",
            "checkpoint_metric": None,
            "checkpoint_metric_value": None,
            "checkpoint_selection_step": None,
        },
        training_checkpointing.build_initial_continuation_state(config),
    )
    candidate = torch.load(source_path, map_location="cpu", weights_only=False)
    if case == "scope":
        candidate["optimizer_state_contract"]["state_scope"] = "shared"
    elif case == "family":
        candidate["optimizer_state_contract"]["optimizer_name"] = "sgd"
    elif case == "hyperparameters":
        candidate["optimizer_state_contract"]["optimizer_kwargs"][
            "weight_decay"
        ] = 0.25
    elif case == "clock":
        candidate["optimizer_state_contract"]["scheduler_clock"] = "width_step"
    elif case == "width_order":
        candidate["optimizer_state_contract"]["ordered_granularities"].reverse()
    elif case == "scheduler":
        candidate["optimizer_state_contract"]["scheduler_contract"][
            "name"
        ] = "cosine"
    elif case == "topology":
        candidate["optimizer_state_contract"]["topology_identity"]["variant"] = (
            "concat"
        )
    elif case == "checkpoint_version":
        candidate["checkpoint_schema_version"] = 99
    elif case == "collection_version":
        candidate["optimizer_state_collection"]["schema_version"] = 99
    elif case == "missing_collection":
        candidate["optimizer_state_collection"] = None
    elif case == "extra_entry":
        candidate["optimizer_state_collection"]["ordered_entries"].append(
            copy.deepcopy(
                candidate["optimizer_state_collection"]["ordered_entries"][0]
            )
        )
    candidate_path = source_path.with_name(f"invalid-{case}.pt")
    torch.save(candidate, candidate_path)

    restored_model = _ResumeModel()
    restored_optimizer, restored_scheduler = build_per_granularity_optimizer_runtime(
        restored_model, config["training"]
    )
    model_before = copy.deepcopy(restored_model.state_dict())
    optimizer_before = copy.deepcopy(restored_optimizer.state_dict())
    scheduler_before = copy.deepcopy(restored_scheduler.state_dict())

    with pytest.raises(ConfigError):
        training_checkpointing.load_checkpoint_state(
            candidate_path,
            restored_model,
            restored_optimizer,
            restored_scheduler,
            config=config,
        )

    _assert_nested_equal(restored_model.state_dict(), model_before)
    _assert_nested_equal(restored_optimizer.state_dict(), optimizer_before)
    _assert_nested_equal(restored_scheduler.state_dict(), scheduler_before)
