from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import src.training.steps as training_steps
import src.training.checkpointing as training_checkpointing
from src.training.schedules import scheduler_metric_fields, wsd_learning_rate_factor
from src.utils.config import resolve_run_config


def _resolved_wsd_training(max_steps=10, warmup_steps=2, decay_ratio=0.2):
    decay_steps = __import__("math").ceil(max_steps * decay_ratio)
    stable_steps = max_steps - warmup_steps - decay_steps
    contract = {
        "name": "warmup_stable_decay",
        "policy": "ratio_decay_over_total_steps",
        "policy_version": 1,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "stable_steps": stable_steps,
        "decay_steps": decay_steps,
        "decay_ratio": decay_ratio,
        "cooldown_start_step": warmup_steps + stable_steps,
        "cooldown_start_tokens": (warmup_steps + stable_steps) * 8,
        "warmup_type": "linear",
        "decay_type": "cosine",
        "min_lr_ratio": 0.0,
        "min_learning_rate": 0.0,
        "num_cycles": 0.5,
    }
    return {
        "optimizer_name": "sgd",
        "optimizer_kwargs": {
            "momentum": 0.0,
            "dampening": 0.0,
            "nesterov": False,
            "weight_decay": 0.0,
        },
        "scheduler_name": "warmup_stable_decay",
        "scheduler_kwargs": {
            "num_decay_steps": decay_steps,
            "num_stable_steps": stable_steps,
            "warmup_type": "linear",
            "decay_type": "cosine",
            "min_lr_ratio": 0.0,
            "num_cycles": 0.5,
        },
        "resolved_warmup_steps": warmup_steps,
        "resolved_learning_rate": 1.0,
        "max_steps": max_steps,
        "scheduler_contract": contract,
    }


def test_wsd_scheduler_ramps_holds_decays_and_reaches_zero():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    training = _resolved_wsd_training()
    optimizer, scheduler = training_steps.build_optimizer_and_scheduler(
        torch.nn.ParameterList([parameter]), training
    )
    used_rates = []
    for _ in range(training["max_steps"]):
        used_rates.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.step()
        scheduler.step()

    assert used_rates[:3] == pytest.approx([0.0, 0.5, 1.0])
    assert used_rates[2:8] == pytest.approx([1.0] * 6)
    assert used_rates[8:] == pytest.approx([1.0, 0.5])
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)


def test_wsd_scheduler_arguments_are_forwarded_via_scheduler_specific_kwargs(
    monkeypatch,
):
    captured = {}

    def fake_get_scheduler(name, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(training_steps, "get_scheduler", fake_get_scheduler)
    training = _resolved_wsd_training()
    training_steps.build_optimizer_and_scheduler(
        torch.nn.Linear(1, 1), training
    )

    assert captured["name"] == "warmup_stable_decay"
    assert captured["scheduler_specific_kwargs"] == training["scheduler_kwargs"]
    assert "num_decay_steps" not in {
        key for key in captured if key != "scheduler_specific_kwargs"
    }


@pytest.mark.parametrize("resume_position", [1, 7, 8, 9])
def test_wsd_scheduler_state_resume_matches_uninterrupted(resume_position):
    training = _resolved_wsd_training()

    def build():
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer, scheduler = training_steps.build_optimizer_and_scheduler(
            torch.nn.ParameterList([parameter]), training
        )
        return parameter, optimizer, scheduler

    full_parameter, full_optimizer, full_scheduler = build()
    full_rates = []
    for _ in range(training["max_steps"]):
        full_rates.append(float(full_optimizer.param_groups[0]["lr"]))
        full_parameter.grad = torch.ones_like(full_parameter)
        full_optimizer.step()
        full_scheduler.step()

    partial_parameter, partial_optimizer, partial_scheduler = build()
    for _ in range(resume_position):
        partial_parameter.grad = torch.ones_like(partial_parameter)
        partial_optimizer.step()
        partial_scheduler.step()
    saved_parameter = partial_parameter.detach().clone()
    optimizer_state = partial_optimizer.state_dict()
    scheduler_state = partial_scheduler.state_dict()

    resumed_parameter, resumed_optimizer, resumed_scheduler = build()
    resumed_parameter.data.copy_(saved_parameter)
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)
    resumed_rates = []
    for _ in range(resume_position, training["max_steps"]):
        resumed_rates.append(float(resumed_optimizer.param_groups[0]["lr"]))
        resumed_parameter.grad = torch.ones_like(resumed_parameter)
        resumed_optimizer.step()
        resumed_scheduler.step()

    assert resumed_rates == pytest.approx(full_rates[resume_position:])
    assert float(resumed_parameter.detach()) == pytest.approx(
        float(full_parameter.detach())
    )
    assert resumed_optimizer.state_dict() == full_optimizer.state_dict()
    assert resumed_scheduler.state_dict() == full_scheduler.state_dict()


def test_wsd_metric_fields_use_actual_committed_rate_and_boundaries():
    training = _resolved_wsd_training()
    fields = scheduler_metric_fields(
        training,
        scheduler_position=8,
        learning_rate=0.987,
    )
    assert fields["learning_rate"] == pytest.approx(0.987)
    assert fields["scheduler_phase"] == "cooldown"
    assert fields["scheduler_phase_step"] == 0
    assert fields["scheduler_phase_progress"] == 0.0
    assert fields["scheduler_cooldown_start_step"] == 8
    assert fields["scheduler_decay_steps"] == 2
    assert wsd_learning_rate_factor(
        10,
        warmup_steps=2,
        stable_steps=6,
        decay_steps=2,
        warmup_type="linear",
        decay_type="cosine",
        min_lr_ratio=0.0,
        num_cycles=0.5,
    ) == pytest.approx(0.0)


class _TinyScheduledModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.current_granularity = None

    def configure_subnetwork(self, granularity):
        self.current_granularity = granularity

    def forward(self, input_ids, attention_mask=None, labels=None):
        del attention_mask, labels
        loss = self.weight.square() + input_ids.float().mean() * 0.0
        return SimpleNamespace(loss=loss)


def test_training_rows_record_the_rate_used_before_scheduler_advance(tmp_path):
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=tmp_path / "debug-nested-001",
        overrides={
            "training.token_budget": 512,
            "training.max_steps": 4,
            "training.warmup_steps": 1,
            "training.scheduler": {
                "name": "warmup_stable_decay",
                "kwargs": {
                    "decay_ratio": 0.25,
                    "warmup_type": "linear",
                    "decay_type": "cosine",
                    "min_lr_ratio": 0.0,
                    "num_cycles": 0.5,
                },
            },
            "evaluation.validation.enabled": False,
            "evaluation.validation.run_at_completion": False,
            "evaluation.validation.interval_steps": 0,
            "training.eval_interval": 0,
            "outputs.save_checkpoints": False,
        },
    )
    model = _TinyScheduledModel()
    optimizer, scheduler = training_steps.build_optimizer_and_scheduler(
        model, config["training"]
    )
    batches = [
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "labels": torch.tensor([[1, 2, 3]], dtype=torch.long),
        }
        for _ in range(4)
    ]
    rows = training_steps.train_for_steps(
        config,
        model,
        batches,
        [],
        optimizer,
        scheduler,
        torch.device("cpu"),
        run_state=training_checkpointing.build_initial_continuation_state(config),
    )
    representative = {
        int(row["step"]): row
        for row in rows
        if row["split"] == "train"
    }
    peak_lr = config["training"]["resolved_learning_rate"]

    assert representative[1]["learning_rate"] == pytest.approx(0.0)
    assert representative[2]["learning_rate"] == pytest.approx(peak_lr)
    assert representative[3]["learning_rate"] == pytest.approx(peak_lr)
    assert representative[4]["learning_rate"] == pytest.approx(peak_lr)
    assert representative[1]["scheduler_phase"] == "warmup"
    assert representative[2]["scheduler_phase"] == "stable"
    assert representative[4]["scheduler_phase"] == "cooldown"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)
