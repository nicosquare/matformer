import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import Dataset

from training.run import run_training
from utils.config import resolve_run_config


class TinyNestedTrainingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.current_granularity = None
        self.train_forward_granularities = []

    def configure_subnetwork(self, granularity):
        self.current_granularity = granularity

    def forward(self, input_ids, attention_mask=None, labels=None):
        if self.training:
            self.train_forward_granularities.append(self.current_granularity)

        loss = self.weight.pow(2) + input_ids.float().mean() * 0.0
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


def test_tiny_nested_training_accumulates_all_granularities_per_batch(tmp_path):
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
        ],
    )
    tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 0], [3, 4, 5]],
            "attention_mask": [[1, 1, 0], [1, 1, 1]],
        }
    )
    model = TinyNestedTrainingModel()

    result = run_training(
        config,
        model=model,
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    assert model.train_forward_granularities == ["s", "m", "l", "xl"]
    assert result["metrics_path"] == output_dir / "metrics.csv"

    with result["metrics_path"].open("r", encoding="utf-8", newline="") as metrics_file:
        train_rows = [
            row
            for row in csv.DictReader(metrics_file)
            if row["split"] == "train" and row["step"] == "1"
        ]

    assert [row["granularity"] for row in train_rows] == ["s", "m", "l", "xl"]


def test_training_counts_parameters_before_runtime_wrapping(tmp_path, monkeypatch):
    import training.run as training_run

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
        training_run,
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


def test_tiny_nested_training_can_sample_one_random_granularity_per_batch(
    tmp_path,
    monkeypatch,
):
    import training.run as training_run

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
            "training.granularity_sampling=random",
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
    monkeypatch.setattr(training_run.random, "randrange", lambda count: 2)

    result = run_training(
        config,
        model=model,
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    assert model.train_forward_granularities == ["l"]
    with result["metrics_path"].open("r", encoding="utf-8", newline="") as metrics_file:
        train_rows = [
            row
            for row in csv.DictReader(metrics_file)
            if row["split"] == "train" and row["step"] == "1"
        ]
    assert [row["granularity"] for row in train_rows] == ["l"]


def test_config_driven_nested_training_records_cat_llama_variant_in_summary(tmp_path):
    output_dir = tmp_path / "debug-nested-001"
    config = resolve_run_config(
        "configs/debug_matrix.yaml",
        run_id="debug-nested-001",
        output_dir=output_dir,
        overrides=[
            "model.variant=cat_llama",
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
    assert config["model"]["variant"] == "cat_llama"
    assert summary["model_variant"] == "cat_llama"


def test_config_driven_nested_training_uses_resolved_sgd_optimizer(tmp_path, monkeypatch):
    import training.run as training_run

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
    original_helper = training_run.build_optimizer_and_scheduler

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
        training_run,
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


def test_config_driven_training_uses_distributed_fsdp_path_when_enabled(
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

    import training.run as training_run

    prepare_calls = []
    wrap_calls = []

    def fake_prepare_distributed_context(*args, **kwargs):
        prepare_calls.append((args, kwargs))
        return SimpleNamespace(
            enabled=True,
            rank=0,
            local_rank=0,
            world_size=2,
            is_rank_zero=True,
            strategy="fsdp",
            device=torch.device("cpu"),
        )

    def fake_wrap_model_for_distributed(*args, **kwargs):
        wrap_calls.append((args, kwargs))
        model = args[0] if args else kwargs["model"]
        model.fsdp_wrapped = True
        return model

    monkeypatch.setattr(
        training_run,
        "prepare_distributed_context",
        fake_prepare_distributed_context,
    )
    monkeypatch.setattr(
        training_run,
        "wrap_model_for_distributed",
        fake_wrap_model_for_distributed,
    )

    result = run_training(
        config,
        model=TinyNestedTrainingModel(),
        tokenized_dataset=tokenized_dataset,
        device="cpu",
    )

    assert prepare_calls
    assert wrap_calls
    assert wrap_calls[0][0][0].fsdp_wrapped is True

    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    assert summary["effective_world_size"] == 2
    assert summary["distributed_strategy"] == "fsdp"
    assert summary["distributed_rank"] == 0
    assert summary["distributed_local_rank"] == 0
    assert summary["distributed_world_size"] == 2
    assert summary["distributed_fsdp_config"] == {}
