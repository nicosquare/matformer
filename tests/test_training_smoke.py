import csv
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

from src.models.granularity import build_granularity_pattern
import src.training.distributed as training_distributed
import src.training.modeling as training_modeling
import src.training.monitoring as training_monitoring
import src.training.steps as training_steps
from src.training.run import run_training
from src.utils.config import ConfigError, resolve_run_config
from src.utils.monitoring import group_loss_rows_by_series


class TinyNestedTrainingModel(torch.nn.Module):
    def __init__(self, loss_scale_by_granularity=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.current_granularity = None
        self.current_layer_granularities = None
        self.current_granularity_pattern = None
        self.train_forward_granularities = []
        self.train_forward_layer_granularities = []
        self.loss_scale_by_granularity = loss_scale_by_granularity

    def configure_subnetwork(self, granularity):
        self.current_granularity = granularity
        self.current_layer_granularities = None
        self.current_granularity_pattern = None

    def configure_layer_granularities(self, granularities):
        self.current_granularity = None
        self.current_layer_granularities = list(granularities)
        self.current_granularity_pattern = build_granularity_pattern(
            pattern_type="per_block",
            selected_granularities=tuple(self.current_layer_granularities),
            layer_count=len(self.current_layer_granularities),
            repeatable_source=("tiny-nested-training-model", "per_block"),
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        if self.training:
            if self.current_layer_granularities is not None:
                self.train_forward_layer_granularities.append(
                    list(self.current_layer_granularities)
                )
            else:
                    self.train_forward_granularities.append(self.current_granularity)

        if self.loss_scale_by_granularity is None:
            loss = self.weight.pow(2)
        else:
            if self.current_granularity is None:
                raise ValueError(
                    "Granularity must be configured before a nested-all forward pass"
                )
            loss_scale = self.loss_scale_by_granularity[self.current_granularity]
            loss = self.weight * self.weight.new_tensor(loss_scale)

        loss = loss + input_ids.float().mean() * 0.0
        return SimpleNamespace(loss=loss)


class TinyNestedRuntimePatternModel(torch.nn.Module):
    def __init__(self, loss_scale_by_granularity=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.current_granularity = None
        self.current_layer_granularities = None
        self.current_granularity_pattern = None
        self.train_forward_granularities = []
        self.loss_scale_by_granularity = loss_scale_by_granularity or {
            "s": 1.0,
            "m": 2.0,
            "l": 4.0,
            "xl": 8.0,
        }

    def configure_subnetwork(self, granularity):
        self.current_granularity = granularity
        self.current_layer_granularities = [granularity]
        self.current_granularity_pattern = build_granularity_pattern(
            pattern_type="single",
            selected_granularities=(granularity,),
            layer_count=1,
            repeatable_source=("tiny-nested-runtime-pattern-model", granularity),
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        self.train_forward_granularities.append(self.current_granularity)
        loss_scale = self.loss_scale_by_granularity[self.current_granularity]
        loss = self.weight * self.weight.new_tensor(loss_scale)
        loss = loss + input_ids.float().mean() * 0.0
        return SimpleNamespace(loss=loss)


class FlatParameterRuntimeWrapper(torch.nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped
        self.flat_param = torch.nn.Parameter(torch.ones(999))

    def configure_subnetwork(self, granularity):
        self.wrapped.configure_subnetwork(granularity)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.wrapped(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        yield "flat_param", self.flat_param


class RecordingMonitoringSession:
    def __init__(self, distributed_context=None):
        self.distributed_context = distributed_context
        self.logged_rows = []
        self.closed = False

    def log_rows(self, rows):
        self.logged_rows.extend(dict(row) for row in rows)

    def close(self):
        self.closed = True


class _FakeWandbRun:
    def __init__(self):
        self.config = _FakeWandbConfig()
        self.summary = {}


class _FakeWandbConfig(dict):
    def update(self, *args, **kwargs):
        return super().update(*args)


class _FakeWandbModule:
    def __init__(self):
        self.init_kwargs = None
        self.finish_calls = 0

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return _FakeWandbRun()

    def define_metric(self, *args, **kwargs):
        return None

    def log(self, *args, **kwargs):
        return None

    def finish(self):
        self.finish_calls += 1


class ToyConcatLMCModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.current_granularity = None
        self.current_subset_blocks = None
        self.gradient_membership_counts = [4, 3, 2, 1]
        self.gate_weight_blocks = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(value)) for value in [1.0, 2.0, 3.0, 4.0]]
        )
        self.up_weight_blocks = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(value)) for value in [5.0, 6.0, 7.0, 8.0]]
        )
        self.down_weight_blocks = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(value)) for value in [9.0, 10.0, 11.0, 12.0]]
        )
        self.gate_bias_blocks = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(value)) for value in [13.0, 14.0, 15.0, 16.0]]
        )
        self.up_bias_blocks = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(value)) for value in [17.0, 18.0, 19.0, 20.0]]
        )
        self.down_bias = torch.nn.Parameter(torch.tensor(21.0))

    def configure_subnetwork(self, granularity):
        self.current_granularity = granularity
        self.current_subset_blocks = {"s": 1, "m": 2, "l": 3, "xl": 4}[granularity]

    def forward(self, input_ids, attention_mask=None, labels=None):
        if self.current_subset_blocks is None:
            raise ValueError("Subnetwork size not configured. Call `configure_subnetwork` first.")

        loss = self.down_bias * 1.0
        for blocks in [
            self.gate_weight_blocks,
            self.up_weight_blocks,
            self.down_weight_blocks,
            self.gate_bias_blocks,
            self.up_bias_blocks,
        ]:
            for param in list(blocks)[: self.current_subset_blocks]:
                loss = loss + param

        return SimpleNamespace(loss=loss + input_ids.float().mean() * 0.0)


def _snapshot_named_parameters(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def _snapshot_named_grads(model):
    return {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def _snapshot_optimizer_state(optimizer):
    state = optimizer.state_dict()["state"]
    normalized = {}
    for key, value in state.items():
        normalized[key] = {}
        for state_key, state_value in value.items():
            if torch.is_tensor(state_value):
                normalized[key][state_key] = state_value.detach().cpu().clone()
            else:
                normalized[key][state_key] = state_value
    return normalized


def _assert_optimizer_states_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert left[key].keys() == right[key].keys()
        for state_key in left[key]:
            left_value = left[key][state_key]
            right_value = right[key][state_key]
            if torch.is_tensor(left_value):
                torch.testing.assert_close(left_value, right_value)
            else:
                assert left_value == right_value


def _run_concat_lmc_case(tmp_path, monkeypatch, correction_mode):
    import src.training.run as training_run

    output_dir = tmp_path / f"concat-{correction_mode}" / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "model.variant=concat",
            f"model.correction_mode={correction_mode}",
            "training.max_steps=1",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.scheduler.name=constant",
            "training.scheduler.kwargs.warmup_steps=0",
            "training.gradient_clip_norm=1000",
            "evaluation.validation=false",
        ]
        + (["model.membership_correction=false"] if correction_mode == "none" else []),
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5]],
            "attention_mask": [[1, 1, 0], [1, 1, 1]],
        }
    )
    model = ToyConcatLMCModel()
    initial_parameters = _snapshot_named_parameters(model)
    captured = {}
    original_build_optimizer_and_scheduler = (
        training_steps.build_optimizer_and_scheduler
    )
    parameter_counts = {
        "total_parameters": 21,
        "embedding_parameters": 0,
        "lm_head_parameters": 0,
        "non_embedding_parameters": 21,
    }

    def capturing_build_optimizer_and_scheduler(model_arg, training):
        optimizer, scheduler = original_build_optimizer_and_scheduler(model_arg, training)
        captured["optimizer"] = optimizer
        return optimizer, scheduler

    monkeypatch.setattr(
        training_steps,
        "build_optimizer_and_scheduler",
        capturing_build_optimizer_and_scheduler,
    )
    monkeypatch.setattr(
        training_modeling,
        "build_artifact_parameter_counts",
        lambda *args, **kwargs: {
            granularity: dict(parameter_counts)
            for granularity in ["s", "m", "l", "xl"]
        },
    )
    # Keep the legacy concat-LMC smoke deterministic so it still exercises the
    # full four-granularity correction path after the sampling-mode refactor.
    monkeypatch.setattr(
        training_steps,
        "select_training_granularities",
        lambda config, granularities, device: list(granularities),
    )

    result = run_training(
        config,
        model=model,
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    return {
        "initial_parameters": initial_parameters,
        "final_parameters": _snapshot_named_parameters(model),
        "grads": _snapshot_named_grads(model),
        "optimizer_state": _snapshot_optimizer_state(captured["optimizer"]),
        "summary": json.loads(result["summary_path"].read_text(encoding="utf-8")),
        "model": model,
    }


def _run_slicing_case(tmp_path, monkeypatch, correction_mode):
    import src.training.run as training_run

    output_dir = tmp_path / f"slicing-{correction_mode}" / "debug-nested-001"
    overrides = [
        f"model.correction_mode={correction_mode}",
        "training.max_steps=1",
        "training.eval_interval=0",
        "training.batch_size_per_process=1",
        "training.learning_rate=0.01",
        "training.scheduler.kwargs.warmup_steps=0",
        "evaluation.validation=false",
    ]
    if correction_mode == "none":
        overrides.append("model.membership_correction=false")

    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=overrides,
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5]],
            "attention_mask": [[1, 1, 0], [1, 1, 1]],
        }
    )
    model = TinyNestedTrainingModel()
    initial_parameters = _snapshot_named_parameters(model)
    captured = {}
    original_build_optimizer_and_scheduler = (
        training_steps.build_optimizer_and_scheduler
    )

    def capturing_build_optimizer_and_scheduler(model_arg, training):
        optimizer, scheduler = original_build_optimizer_and_scheduler(model_arg, training)
        captured["optimizer"] = optimizer
        return optimizer, scheduler

    monkeypatch.setattr(
        training_steps,
        "build_optimizer_and_scheduler",
        capturing_build_optimizer_and_scheduler,
    )

    result = run_training(
        config,
        model=model,
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    return {
        "initial_parameters": initial_parameters,
        "final_parameters": _snapshot_named_parameters(model),
        "grads": _snapshot_named_grads(model),
        "optimizer_state": _snapshot_optimizer_state(captured["optimizer"]),
        "train_forward_granularities": list(model.train_forward_granularities),
        "summary": json.loads(result["summary_path"].read_text(encoding="utf-8")),
        "model": model,
    }


def test_tiny_nested_training_can_sample_one_granularity_for_the_nested_random_global_path(
    tmp_path,
    monkeypatch,
):
    import src.training.run as training_run

    output_dir = tmp_path / "dmodel256-pilot-comparison-001"
    config = resolve_run_config(
        "configs/dmodel256_pilot_comparison.yaml",
        output_dir=output_dir,
        overrides=[
            "model.granularity_sampling_mode=global",
            "run.continuation.enabled=false",
            "training.max_steps=1",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.scheduler.kwargs.warmup_steps=0",
            "outputs.save_checkpoints=false",
            "evaluation.validation=false",
        ],
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5]],
            "attention_mask": [[1, 1, 0], [1, 1, 1]],
        }
    )
    model = TinyNestedTrainingModel()
    captured_randrange_values = iter([2])
    monkeypatch.setattr(
        training_steps.random,
        "randrange",
        lambda count: next(captured_randrange_values),
    )

    result = run_training(
        config,
        model=model,
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    with result["metrics_path"].open("r", encoding="utf-8", newline="") as metrics_file:
        train_rows = [
            row
            for row in csv.DictReader(metrics_file)
            if row["split"] == "train" and row["step"] == "1"
        ]

    assert summary["sampling_mode"] == "nested-random"
    assert summary["resolved_sampling_mode"] == "global"
    assert model.train_forward_granularities == ["l"]
    assert model.train_forward_layer_granularities == []
    assert [row["granularity"] for row in train_rows] == ["l"]
    assert all(
        json.loads(row["granularity_pattern_summary"])["pattern_type"] == "single"
        for row in train_rows
    )


def test_refactored_training_flow_writes_representative_artifacts(tmp_path):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "training.max_steps=1",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.scheduler.kwargs.warmup_steps=0",
            "outputs.save_checkpoints=false",
            "evaluation.validation=false",
        ],
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5]],
            "attention_mask": [[1, 1, 0], [1, 1, 1]],
        }
    )

    result = run_training(
        config,
        model=TinyNestedTrainingModel(),
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    with result["metrics_path"].open("r", encoding="utf-8", newline="") as metrics_file:
        metrics_reader = csv.DictReader(metrics_file)
        metrics_rows = list(metrics_reader)

    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    metrics_row = metrics_rows[0]

    assert result["metrics_path"] == output_dir / "metrics.csv"
    assert result["scaling_path"] == output_dir / "scaling_results.csv"
    assert result["summary_path"] == output_dir / "run_summary.json"
    assert summary["status"] == "completed"
    assert summary["output_dir"] == str(output_dir)
    assert summary["metrics_path"] == str(result["metrics_path"])
    assert summary["scaling_results_path"] == str(result["scaling_path"])
    assert summary["extraction_metadata_path"] == str(
        output_dir / "extraction_metadata.json"
    )
    assert summary["checkpoint_status"] == "none"
    assert summary["checkpoint_unavailable_reason"] == "checkpoint writes disabled"
    assert summary["granularity_pattern_summary"]["repeatable_source"][0] == (
        "debug-nested-001"
    )
    assert summary["correction_context"]["local_correction_active"] is False
    assert metrics_row["run_id"] == "debug-nested-001"
    assert metrics_row["split"] == "train"
    assert json.loads(metrics_row["granularity_pattern_summary"])[
        "repeatable_source"
    ][0] == "debug-nested-001"
    assert "local_correction_active" in json.loads(metrics_row["correction_context"])
    assert len(metrics_rows) >= 1
    for artifact_name in [
        "config.json",
        "metrics.csv",
        "scaling_results.csv",
        "run_summary.json",
        "extraction_metadata.json",
    ]:
        assert (output_dir / artifact_name).exists()


def _run_monitoring_smoke_case(tmp_path, run_id: str):
    output_dir = tmp_path / run_id
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id=run_id,
        output_dir=output_dir,
        overrides=[
            "monitoring.enabled=true",
            "training.max_steps=1",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.scheduler.kwargs.warmup_steps=0",
            "evaluation.validation=false",
        ],
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5]],
            "attention_mask": [[1, 1, 0], [1, 1, 1]],
        }
    )

    result = run_training(
        config,
        model=TinyNestedTrainingModel(),
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    with result["metrics_path"].open("r", encoding="utf-8", newline="") as metrics_file:
        train_rows = [
            row
            for row in csv.DictReader(metrics_file)
            if row["split"] == "train" and row["step"] == "1"
        ]

    return summary, group_loss_rows_by_series(train_rows)


def _read_heartbeat_events(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as heartbeat_file:
        return [
            json.loads(line)
            for line in heartbeat_file
            if line.strip()
        ]


def test_train_py_config_flag_dispatches_to_configured_run(monkeypatch, tmp_path):
    import train as train_module
    import src.training.run as training_run

    captured = {}

    def fake_run_from_config_path(
        config_path,
        *,
        run_id=None,
        overrides=None,
        output_dir=None,
    ):
        captured["config_path"] = config_path
        captured["run_id"] = run_id
        captured["overrides"] = list(overrides or [])
        captured["output_dir"] = output_dir

    monkeypatch.setattr(
        training_run,
        "run_from_config_path",
        fake_run_from_config_path,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/debug_matrix.yaml",
            "--run-id",
            "debug-nested-001",
            "--output-root",
            str(tmp_path / "artifacts"),
            "--override",
            "training.max_steps=1",
        ],
    )

    train_module.main()

    assert captured == {
        "config_path": "configs/debug_matrix.yaml",
        "run_id": "debug-nested-001",
        "overrides": [
            "training.max_steps=1",
            f"run.output_root={tmp_path / 'artifacts'}",
        ],
        "output_dir": None,
    }


@pytest.mark.xfail(
    reason="Run resumption wiring is implemented in T009/T010, not yet here",
    strict=False,
)
def test_interrupted_and_relaunched_run_preserves_the_same_output_dir(
    tmp_path,
    monkeypatch,
):
    import src.training.run as training_run

    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "run.continuation.enabled=true",
            "training.max_steps=1",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.scheduler.kwargs.warmup_steps=0",
            "evaluation.validation=false",
        ],
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5]],
            "attention_mask": [[1, 1, 0], [1, 1, 1]],
        }
    )
    original_train_for_steps = training_steps.train_for_steps
    train_invocations = {"count": 0}

    def interrupting_train_for_steps(*args, **kwargs):
        train_invocations["count"] += 1
        if train_invocations["count"] == 1:
            raise RuntimeError("simulated scheduler preemption")
        return original_train_for_steps(*args, **kwargs)

    monkeypatch.setattr(
        training_steps,
        "train_for_steps",
        interrupting_train_for_steps,
    )

    with pytest.raises(RuntimeError, match="simulated scheduler preemption"):
        run_training(
            config,
            model=TinyNestedTrainingModel(),
            tokenized_dataset=tokenized_dataset,
            device="cpu",
        )

    result = run_training(
        config,
        model=TinyNestedTrainingModel(),
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    assert result["summary_path"] == output_dir / "run_summary.json"
    assert summary["run_id"] == "debug-nested-001"
    assert summary["output_dir"] == str(output_dir)
    assert summary["continuation_state"]["status"] == "resumed"
    assert summary["continuation_state"]["resume_count"] == 1
    assert summary["continuation_state"]["last_completed_step"] == 1
    assert summary["latest_checkpoint_path"] == str(
        output_dir / "checkpoints" / "latest.pt"
    )


def test_pre_nested_warmup_disabled_path_keeps_the_run_in_the_standard_flow(
    tmp_path,
):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "training.max_steps=1",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.scheduler.kwargs.warmup_steps=0",
            "training.pre_nested_warmup.enabled=false",
            "evaluation.validation=false",
        ],
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5]],
            "attention_mask": [[1, 1, 0], [1, 1, 1]],
        }
    )

    result = run_training(
        config,
        model=TinyNestedTrainingModel(),
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    heartbeat_events = _read_heartbeat_events(output_dir / "heartbeats.jsonl")

    assert summary["warmup_policy"] == {
        "enabled": False,
        "duration": 0,
        "unit": "epochs",
        "completed": False,
        "completion_step": None,
        "transition_reason": None,
    }
    assert summary["warmup_completion_step"] is None
    assert summary["warmup_completed"] is False
    assert all("warmup" not in str(event["stage"]) for event in heartbeat_events)


def test_pre_nested_warmup_transition_records_a_warmup_stage_before_training(
    tmp_path,
):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "training.max_steps=2",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.scheduler.kwargs.warmup_steps=0",
            "training.pre_nested_warmup.enabled=true",
            "training.pre_nested_warmup.duration=1",
            "training.pre_nested_warmup.unit=steps",
            "evaluation.validation=false",
        ],
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5]],
            "attention_mask": [[1, 1, 0], [1, 1, 1]],
        }
    )

    result = run_training(
        config,
        model=TinyNestedTrainingModel(),
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    heartbeat_events = _read_heartbeat_events(output_dir / "heartbeats.jsonl")
    stage_names = [str(event["stage"]) for event in heartbeat_events]
    warmup_stage_index = next(
        index for index, stage_name in enumerate(stage_names) if "warmup" in stage_name
    )
    training_stage_index = stage_names.index("training")

    assert summary["warmup_policy"]["enabled"] is True
    assert summary["warmup_policy"]["duration"] == 1
    assert summary["warmup_policy"]["unit"] == "steps"
    assert summary["warmup_policy"]["completed"] is True
    assert summary["warmup_completion_step"] == 1
    assert summary["warmup_completed"] is True
    assert warmup_stage_index < training_stage_index


def test_tiny_nested_training_accumulates_all_granularities_per_batch(
    tmp_path,
    monkeypatch,
):
    import src.training.run as training_run

    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "training.max_steps=1",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.warmup_steps=0",
        ],
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5]],
            "attention_mask": [[1, 1, 0], [1, 1, 1]],
        }
    )
    model = TinyNestedTrainingModel()
    clip_calls = []

    def fake_clip_grad_norm_(parameters, max_norm, *args, **kwargs):
        clip_calls.append(max_norm)
        return torch.tensor(0.0)

    monkeypatch.setattr(training_steps, "clip_grad_norm_", fake_clip_grad_norm_)

    result = run_training(
        config,
        model=model,
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    assert model.train_forward_granularities == ["s", "m", "l", "xl"]
    assert clip_calls == [1.0]
    assert result["metrics_path"] == output_dir / "metrics.csv"

    with result["metrics_path"].open("r", encoding="utf-8", newline="") as metrics_file:
        train_rows = [
            row
            for row in csv.DictReader(metrics_file)
            if row["split"] == "train" and row["step"] == "1"
        ]

    assert [row["granularity"] for row in train_rows] == ["s", "m", "l", "xl"]


def test_tiny_nested_training_averages_losses_across_all_granularities_for_nested_all(
    tmp_path,
):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "run.sampling_mode=nested-all",
            "model.granularity_sampling_mode=global",
            "run.continuation.enabled=false",
            "training.max_steps=1",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.gradient_clip_norm=1000",
            "training.scheduler.kwargs.warmup_steps=0",
            "outputs.save_checkpoints=false",
            "evaluation.validation=false",
        ],
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5]],
            "attention_mask": [[1, 1, 0], [1, 1, 1]],
        }
    )
    model = TinyNestedTrainingModel(
        loss_scale_by_granularity={
            "s": 1.0,
            "m": 2.0,
            "l": 4.0,
            "xl": 8.0,
        }
    )

    result = run_training(
        config,
        model=model,
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    with result["metrics_path"].open("r", encoding="utf-8", newline="") as metrics_file:
        train_rows = [
            row
            for row in csv.DictReader(metrics_file)
            if row["split"] == "train" and row["step"] == "1"
        ]

    assert summary["sampling_mode"] == "nested-all"
    assert summary["resolved_sampling_mode"] == "global"
    assert summary["granularity_pattern_summary"]["pattern_type"] == "all_granularities"
    assert summary["granularity_pattern_summary"]["selected_granularities"] == [
        "s",
        "m",
        "l",
        "xl",
    ]
    assert model.train_forward_granularities == ["s", "m", "l", "xl"]
    assert [row["granularity"] for row in train_rows] == ["s", "m", "l", "xl"]
    assert [float(row["loss"]) for row in train_rows] == [0.5, 1.0, 2.0, 4.0]
    assert float(model.weight.grad.detach().cpu().item()) == pytest.approx(3.75)


def test_tiny_nested_training_records_nested_all_summary_even_when_runtime_pattern_stays_single(
    tmp_path,
):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "run.sampling_mode=nested-all",
            "model.granularity_sampling_mode=global",
            "run.continuation.enabled=false",
            "training.max_steps=1",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.gradient_clip_norm=1000",
            "training.scheduler.kwargs.warmup_steps=0",
            "outputs.save_checkpoints=false",
            "evaluation.validation=false",
        ],
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5]],
            "attention_mask": [[1, 1, 0], [1, 1, 1]],
        }
    )
    model = TinyNestedRuntimePatternModel()

    result = run_training(
        config,
        model=model,
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    with result["metrics_path"].open("r", encoding="utf-8", newline="") as metrics_file:
        train_rows = [
            row
            for row in csv.DictReader(metrics_file)
            if row["split"] == "train" and row["step"] == "1"
        ]

    assert model.current_granularity_pattern.pattern_type == "single"
    assert summary["sampling_mode"] == "nested-all"
    assert summary["granularity_pattern_summary"]["pattern_type"] == "all_granularities"
    assert summary["granularity_pattern_summary"]["selected_granularities"] == [
        "s",
        "m",
        "l",
        "xl",
    ]
    assert [row["granularity"] for row in train_rows] == ["s", "m", "l", "xl"]
    assert all(
        json.loads(row["granularity_pattern_summary"])["pattern_type"]
        == "all_granularities"
        for row in train_rows
    )


def test_training_counts_parameters_before_runtime_wrapping(tmp_path, monkeypatch):
    import src.training.run as training_run

    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "training.max_steps=1",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.scheduler.kwargs.warmup_steps=0",
            "evaluation.validation=false",
        ],
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5]],
            "attention_mask": [[1, 1, 0], [1, 1, 1]],
        }
    )
    monkeypatch.setattr(
        training_distributed,
        "wrap_model_for_distributed",
        lambda model, context: FlatParameterRuntimeWrapper(model),
    )

    result = run_training(
        config,
        model=TinyNestedTrainingModel(),
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    counts = result["parameter_counts_by_granularity"]["s"]
    assert counts["total_parameters"] == 1
    assert counts["non_embedding_parameters"] == 1

    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    assert summary["parameter_counts_by_granularity"]["s"]["total_parameters"] == 1

    with result["scaling_path"].open("r", encoding="utf-8", newline="") as scaling_file:
        rows = list(csv.DictReader(scaling_file))
    assert {row["total_parameters"] for row in rows} == {"1"}


def test_tiny_nested_training_can_sample_one_granularity_per_block_per_batch(
    tmp_path,
    monkeypatch,
):
    import src.training.run as training_run

    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "run.sampling_mode=nested-random",
            "model.granularity_sampling_mode=per_block",
            "run.continuation.enabled=false",
            "training.max_steps=1",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.scheduler.kwargs.warmup_steps=0",
            "outputs.save_checkpoints=false",
            "evaluation.validation=false",
        ],
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5]],
            "attention_mask": [[1, 1, 0], [1, 1, 1]],
        }
    )
    model = TinyNestedTrainingModel()
    randrange_values = iter([0, 1])
    monkeypatch.setattr(
        training_steps.random,
        "randrange",
        lambda count: next(randrange_values),
    )

    result = run_training(
        config,
        model=model,
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    assert summary["sampling_mode"] == "nested-random"
    assert summary["resolved_sampling_mode"] == "per_block"
    assert model.train_forward_granularities == []
    assert model.train_forward_layer_granularities == [["s", "m"]]
    with result["metrics_path"].open("r", encoding="utf-8", newline="") as metrics_file:
        train_rows = [
            row
            for row in csv.DictReader(metrics_file)
            if row["split"] == "train" and row["step"] == "1"
        ]
    assert [row["granularity"] for row in train_rows] == ["s", "m"]
    assert all(
        json.loads(row["granularity_pattern_summary"])["pattern_type"] == "per_block"
        for row in train_rows
    )


def test_config_driven_nested_training_records_concat_variant_in_summary(tmp_path):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "model.variant=concat",
            "training.max_steps=1",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.scheduler.kwargs.warmup_steps=0",
        ],
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5]],
            "attention_mask": [[1, 1, 0], [1, 1, 1]],
        }
    )

    result = run_training(
        config,
        model=TinyNestedTrainingModel(),
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    assert config["model"]["variant"] == "concat"
    assert summary["model_variant"] == "concat"


def test_config_driven_nested_training_uses_resolved_sgd_optimizer(tmp_path, monkeypatch):
    import src.training.run as training_run

    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "training.max_steps=1",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.02",
            "training.scheduler.kwargs.warmup_steps=0",
            "training.optimizer.preset=null",
            "training.optimizer.name=sgd",
            "training.optimizer.kwargs.momentum=0.8",
            "training.optimizer.kwargs.nesterov=true",
            "training.scheduler.name=constant",
        ],
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5]],
            "attention_mask": [[1, 1, 0], [1, 1, 1]],
        }
    )
    captured = {}
    original_helper = training_steps.build_optimizer_and_scheduler

    def capturing_build_optimizer_and_scheduler(model, training):
        captured["optimizer_name"] = training["optimizer_name"]
        captured["resolved_learning_rate"] = training["resolved_learning_rate"]
        captured["scheduler_warmup_steps"] = training["scheduler"]["kwargs"]["warmup_steps"]
        captured["scheduler_resolved_warmup_steps"] = training["scheduler"]["resolved_warmup_steps"]
        captured["optimizer_kwargs"] = training["optimizer_kwargs"]
        captured["scheduler_name"] = training["scheduler_name"]
        captured["scheduler_kwargs"] = training["scheduler_kwargs"]
        optimizer, scheduler = original_helper(model, training)
        captured["optimizer_type"] = type(optimizer).__name__
        return optimizer, scheduler

    monkeypatch.setattr(
        training_steps,
        "build_optimizer_and_scheduler",
        capturing_build_optimizer_and_scheduler,
    )

    result = run_training(
        config,
        model=TinyNestedTrainingModel(),
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    assert captured["optimizer_name"] == "sgd"
    assert captured["optimizer_type"] == "SGD"
    assert captured["resolved_learning_rate"] == 0.02
    assert captured["scheduler_warmup_steps"] == 0
    assert captured["scheduler_resolved_warmup_steps"] == 0
    assert captured["optimizer_kwargs"] == {
        "momentum": 0.8,
        "dampening": 0.0,
        "nesterov": True,
        "weight_decay": 0.0,
    }
    assert captured["scheduler_name"] == "constant"
    assert captured["scheduler_kwargs"] == {}
    assert summary["optimizer_name"] == "sgd"
    assert summary["scheduler_name"] == "constant"
    assert summary["scheduler_warmup_steps"] == 0
    assert summary["scheduler_resolved_warmup_steps"] == 0


def test_concat_lmc_applies_block_specific_effective_learning_rates_without_changing_gradients_or_optimizer_state(
    tmp_path,
    monkeypatch,
):
    none_case = _run_concat_lmc_case(tmp_path, monkeypatch, "none")
    lmc_case = _run_concat_lmc_case(tmp_path, monkeypatch, "lmc")

    expected_scales = {
        "gate_weight_blocks": [1.0, 4.0 / 3.0, 2.0, 4.0],
        "up_weight_blocks": [1.0, 4.0 / 3.0, 2.0, 4.0],
        "down_weight_blocks": [1.0, 4.0 / 3.0, 2.0, 4.0],
        "gate_bias_blocks": [1.0, 4.0 / 3.0, 2.0, 4.0],
        "up_bias_blocks": [1.0, 4.0 / 3.0, 2.0, 4.0],
    }

    for name, initial_value in none_case["initial_parameters"].items():
        none_delta = initial_value - none_case["final_parameters"][name]
        lmc_delta = initial_value - lmc_case["final_parameters"][name]
        if name == "down_bias":
            torch.testing.assert_close(lmc_delta, none_delta)
            continue

        block_group, block_index = name.split(".")
        scale = expected_scales[block_group][int(block_index)]
        torch.testing.assert_close(lmc_delta, none_delta * scale)

    for name in none_case["grads"]:
        torch.testing.assert_close(none_case["grads"][name], lmc_case["grads"][name])

    _assert_optimizer_states_equal(
        none_case["optimizer_state"],
        lmc_case["optimizer_state"],
    )
    assert none_case["summary"]["correction_mode"] == "none"
    assert lmc_case["summary"]["correction_mode"] == "lmc"


def test_slicing_runs_ignore_correction_mode_for_none_and_gmc(
    tmp_path,
    monkeypatch,
):
    none_case = _run_slicing_case(tmp_path, monkeypatch, "none")
    gmc_case = _run_slicing_case(tmp_path, monkeypatch, "gmc")

    assert none_case["train_forward_granularities"] == ["s", "m", "l", "xl"]
    assert gmc_case["train_forward_granularities"] == ["s", "m", "l", "xl"]

    for name, initial_value in none_case["initial_parameters"].items():
        none_delta = initial_value - none_case["final_parameters"][name]
        gmc_delta = initial_value - gmc_case["final_parameters"][name]
        torch.testing.assert_close(gmc_delta, none_delta)

    for name in none_case["grads"]:
        torch.testing.assert_close(none_case["grads"][name], gmc_case["grads"][name])

    _assert_optimizer_states_equal(
        none_case["optimizer_state"],
        gmc_case["optimizer_state"],
    )
    assert none_case["summary"]["correction_mode"] == "none"
    assert gmc_case["summary"]["correction_mode"] == "gmc"


def test_external_output_root_keeps_required_artifacts_outside_repo_outputs(tmp_path):
    output_root = tmp_path / "external-output-root"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        overrides=[
            f"run.output_root={output_root}",
            "training.max_steps=1",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.scheduler.kwargs.warmup_steps=0",
        ],
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5]],
            "attention_mask": [[1, 1, 0], [1, 1, 1]],
        }
    )

    result = run_training(
        config,
        model=TinyNestedTrainingModel(),
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    run_dir = output_root / config["run"]["output_group"] / "debug-nested-001"
    required_artifacts = {
        run_dir / "config.json",
        run_dir / "metrics.csv",
        run_dir / "scaling_results.csv",
        run_dir / "run_summary.json",
        run_dir / "extraction_metadata.json",
    }

    assert config["run"]["output_dir"] == str(run_dir)
    assert result["metrics_path"] == run_dir / "metrics.csv"
    assert result["scaling_path"] == run_dir / "scaling_results.csv"
    assert result["summary_path"] == run_dir / "run_summary.json"
    for artifact_path in required_artifacts:
        assert artifact_path.exists()
        assert artifact_path.resolve().is_relative_to(output_root.resolve())
        assert not artifact_path.resolve().is_relative_to(
            (Path.cwd() / "outputs").resolve()
        )

    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["output_root"] == str(output_root)
    assert summary["output_dir"] == str(run_dir)


def test_budgeted_training_stops_at_token_budget_before_manual_step_cap(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "training.token_budget=64",
            "training.max_steps=10",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.scheduler.kwargs.warmup_steps=0",
            "evaluation.validation=false",
        ],
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            "attention_mask": [[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1]],
        }
    )

    result = run_training(
        config,
        model=TinyNestedTrainingModel(),
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    for field_name in [
        "stop_reason",
        "expected_tokens_per_step",
        "derived_max_steps",
        "effective_world_size",
        "content_tokens_seen",
    ]:
        assert field_name in summary
    assert summary["stop_reason"] == "token_budget_reached"
    assert summary["token_budget"] == 64
    assert summary["tokens_seen"] == 64
    assert summary["content_tokens_seen"] == 2
    assert summary["expected_tokens_per_step"] == 64
    assert summary["derived_max_steps"] == 1
    assert summary["effective_world_size"] == 1
    assert summary["steps_completed"] == 1

    with result["metrics_path"].open("r", encoding="utf-8", newline="") as metrics_file:
        train_rows = [
            row for row in csv.DictReader(metrics_file) if row["split"] == "train"
        ]
        train_steps = {row["step"] for row in train_rows}
    assert train_steps == {"1"}
    assert {row["tokens_seen"] for row in train_rows} == {"64"}
    assert {row["content_tokens_seen"] for row in train_rows} == {"2"}


def test_config_driven_training_rejects_multi_process_execution_before_setup(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "training.token_budget=4",
            "training.max_steps_cap=1",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.scheduler.kwargs.warmup_steps=0",
            "training.distributed.strategy=fsdp",
            "evaluation.validation=false",
        ],
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4], [5, 6, 7, 8]],
            "attention_mask": [[1, 1, 1, 1], [1, 1, 1, 1]],
        }
    )

    import src.training.run as training_run

    def fake_prepare_distributed_context(*args, **kwargs):
        raise AssertionError("prepare_distributed_context should not be called")

    def fake_wrap_model_for_distributed(*args, **kwargs):
        raise AssertionError("wrap_model_for_distributed should not be called")

    monkeypatch.setattr(
        training_distributed,
        "prepare_distributed_context",
        fake_prepare_distributed_context,
    )
    monkeypatch.setattr(
        training_distributed,
        "wrap_model_for_distributed",
        fake_wrap_model_for_distributed,
    )

    with pytest.raises(
        ConfigError,
        match="single-process only: distributed or multi-process execution is not supported",
    ):
        run_training(
            config,
            model=TinyNestedTrainingModel(),
            tokenized_dataset=tokenized_dataset,
            device="cpu",
        )


def test_run_training_rejects_invalid_adaptive_pairing_before_setup(
    tmp_path,
    monkeypatch,
):
    import src.training.run as training_run

    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "training.token_budget=4",
            "training.max_steps_cap=1",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.scheduler.kwargs.warmup_steps=0",
            "evaluation.validation=false",
        ],
    )
    config["run"]["sampling_mode"] = "nested-all"
    config["model"]["granularity_sampling_mode"] = "adaptive_per_block"

    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4], [5, 6, 7, 8]],
            "attention_mask": [[1, 1, 1, 1], [1, 1, 1, 1]],
        }
    )

    def fake_prepare_distributed_context(*args, **kwargs):
        raise AssertionError("prepare_distributed_context should not be called")

    def fake_wrap_model_for_distributed(*args, **kwargs):
        raise AssertionError("wrap_model_for_distributed should not be called")

    monkeypatch.setattr(
        training_distributed,
        "prepare_distributed_context",
        fake_prepare_distributed_context,
    )
    monkeypatch.setattr(
        training_distributed,
        "wrap_model_for_distributed",
        fake_wrap_model_for_distributed,
    )

    with pytest.raises(
        ConfigError,
        match="model.granularity_sampling_mode=adaptive_per_block requires nested-random runs",
    ):
        run_training(
            config,
            model=TinyNestedTrainingModel(),
            tokenized_dataset=tokenized_dataset,
            device="cpu",
        )


@pytest.mark.xfail(
    reason="Adaptive per-block runtime wiring lands in T013/T014",
    strict=False,
)
def test_adaptive_per_block_smoke_shifts_patterns_and_resumes_from_checkpoint(
    tmp_path,
    monkeypatch,
):
    import src.training.run as training_run

    output_dir = tmp_path / "debug-nested-001"
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5]],
            "attention_mask": [[1, 1, 0], [1, 1, 1]],
        }
    )

    fresh_pattern_calls = {"count": 0}

    def fake_select_training_layer_granularities(config, granularities, device):
        resume_count = int(config["run"].get("continuation", {}).get("resume_count", 0))
        if resume_count > 0:
            return ["m", "l"]

        fresh_pattern_calls["count"] += 1
        if fresh_pattern_calls["count"] == 1:
            return ["s", "m"]
        return ["m", "l"]

    monkeypatch.setattr(
        training_run,
        "select_training_layer_granularities",
        fake_select_training_layer_granularities,
    )
    monkeypatch.setattr(
        training_run,
        "select_training_granularities",
        lambda *args, **kwargs: pytest.fail(
            "adaptive_per_block should use per-block selection"
        ),
    )

    first_config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "run.sampling_mode=nested-random",
            "model.granularity_sampling_mode=adaptive_per_block",
            "run.continuation.enabled=true",
            "training.max_steps=2",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.scheduler.kwargs.warmup_steps=0",
            "training.token_budget=16",
            "outputs.save_checkpoints=true",
            "evaluation.validation=false",
        ],
    )

    first_model = TinyNestedTrainingModel()
    first_result = run_training(
        first_config,
        model=first_model,
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    first_summary = json.loads(first_result["summary_path"].read_text(encoding="utf-8"))
    assert first_summary["resolved_sampling_mode"] == "adaptive_per_block"
    assert first_summary["continuation_state"]["status"] == "fresh"
    assert first_model.train_forward_layer_granularities == [
        ["s", "m"],
        ["m", "l"],
    ]

    with first_result["metrics_path"].open(
        "r",
        encoding="utf-8",
        newline="",
    ) as metrics_file:
        first_train_rows = [
            row for row in csv.DictReader(metrics_file) if row["split"] == "train"
        ]
    assert [
        json.loads(row["granularity_pattern_summary"])["selected_granularities"]
        for row in first_train_rows
    ] == [["s", "m"], ["m", "l"]]

    resumed_config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "run.sampling_mode=nested-random",
            "model.granularity_sampling_mode=adaptive_per_block",
            "run.continuation.enabled=true",
            "training.max_steps=3",
            "training.eval_interval=0",
            "training.batch_size_per_process=1",
            "training.learning_rate=0.01",
            "training.scheduler.kwargs.warmup_steps=0",
            "training.token_budget=16",
            "outputs.save_checkpoints=true",
            "evaluation.validation=false",
        ],
    )

    resumed_model = TinyNestedTrainingModel()
    resumed_result = run_training(
        resumed_config,
        model=resumed_model,
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    resumed_summary = json.loads(
        resumed_result["summary_path"].read_text(encoding="utf-8")
    )
    assert resumed_summary["continuation_state"]["status"] == "resumed"
    assert resumed_summary["continuation_state"]["resume_count"] == 1
    assert resumed_model.train_forward_layer_granularities[0] == ["m", "l"]


def test_monitoring_smoke_groups_nested_and_standalone_runs_by_series(
    tmp_path,
    monkeypatch,
):
    import src.training.run as training_run

    created_sessions = []

    def fake_create_monitoring_session(config, distributed_context=None):
        session = RecordingMonitoringSession(distributed_context=distributed_context)
        created_sessions.append(session)
        return session

    monkeypatch.setattr(
        training_monitoring,
        "create_monitoring_session",
        fake_create_monitoring_session,
    )

    nested_summary, nested_series = _run_monitoring_smoke_case(
        tmp_path,
        "debug-nested-001",
    )
    standalone_summary, standalone_series = _run_monitoring_smoke_case(
        tmp_path,
        "debug-standalone-s-001",
    )

    assert nested_summary["monitoring_enabled"] is True
    assert standalone_summary["monitoring_enabled"] is True
    assert set(nested_series) == {
        "train/loss/s",
        "train/loss/m",
        "train/loss/l",
        "train/loss/xl",
    }
    assert set(standalone_series) == {"train/loss/s"}
    assert all(len(rows) == 1 for rows in nested_series.values())
    assert len(standalone_series["train/loss/s"]) == 1
    assert len(created_sessions) == 2
    assert created_sessions[0].closed is True
    assert created_sessions[1].closed is True
    assert set(group_loss_rows_by_series(created_sessions[0].logged_rows)) == {
        "train/loss/s",
        "train/loss/m",
        "train/loss/l",
        "train/loss/xl",
    }
    assert set(group_loss_rows_by_series(created_sessions[1].logged_rows)) == {
        "train/loss/s",
    }
    assert nested_summary["monitoring_backend"] == "wandb"
    assert standalone_summary["monitoring_backend"] == "wandb"
    assert [entry["series_name"] for entry in nested_summary["monitoring_series_metadata"]] == [
        "train/loss/s",
        "train/loss/m",
        "train/loss/l",
        "train/loss/xl",
    ]
    assert [entry["series_name"] for entry in standalone_summary["monitoring_series_metadata"]] == [
        "train/loss/s",
    ]


def test_wandb_session_uses_explicit_project_and_entity_settings(
    tmp_path,
    monkeypatch,
):
    import src.training.run as training_run

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.init_kwargs = None
    fake_wandb.finish_calls = 0

    def fake_init(**kwargs):
        fake_wandb.init_kwargs = kwargs
        return _FakeWandbRun()

    def fake_define_metric(*args, **kwargs):
        return None

    def fake_log(*args, **kwargs):
        return None

    def fake_finish():
        fake_wandb.finish_calls += 1

    fake_wandb.init = fake_init
    fake_wandb.define_metric = fake_define_metric
    fake_wandb.log = fake_log
    fake_wandb.finish = fake_finish
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=tmp_path / "debug-nested-001",
        overrides=[
            "monitoring.enabled=true",
            "monitoring.project=custom-project",
            "monitoring.entity=research-team",
            "monitoring.group=shared-group",
            "monitoring.job_type=evaluation",
            "monitoring.name=custom-run-name",
            "monitoring.tags=[alpha,beta]",
            "monitoring.notes=long run smoke",
            "monitoring.mode=offline",
        ],
    )

    session = training_monitoring.WandbMonitoringSession(
        config,
        distributed_context=SimpleNamespace(enabled=False),
    )

    assert fake_wandb.init_kwargs is not None
    assert fake_wandb.init_kwargs["project"] == "custom-project"
    assert fake_wandb.init_kwargs["entity"] == "research-team"
    assert fake_wandb.init_kwargs["group"] == "shared-group"
    assert fake_wandb.init_kwargs["job_type"] == "evaluation"
    assert fake_wandb.init_kwargs["name"] == "custom-run-name"
    assert fake_wandb.init_kwargs["tags"] == ["alpha", "beta"]
    assert fake_wandb.init_kwargs["notes"] == "long run smoke"
    assert fake_wandb.init_kwargs["mode"] == "offline"
    assert fake_wandb.init_kwargs["id"] == "debug-nested-001"
    session.close()
    assert fake_wandb.finish_calls == 1
