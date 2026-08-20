import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from transformers import LlamaConfig

from src.models.ffn import ModifiedLlamaMLP
from src.training.gradient_interference import (
    EMPTY_JOURNAL_HASH,
    GradientInterferenceError,
    _pair_statistics,
    append_snapshot_record,
    build_gradient_interference_state,
    reconcile_snapshot_journal,
    record_snapshot_commit,
    snapshot_id,
    validate_snapshot_record,
)
from src.training.gradient_probe import (
    ControlledGradientProbeError,
    measure_controlled_gradients,
    resolve_controlled_ffn_support,
)
from src.training.run import run_training
import src.training.gradient_interference as gradient_interference_module
from src.utils.config import ConfigError, resolve_run_config


def _enabled_config(tmp_path, **overrides):
    values = {
        "evaluation.gradient_interference.enabled": True,
        "evaluation.gradient_interference.trajectory_fractions": [0.0, 0.5, 1.0],
        "evaluation.gradient_interference.include_warmup_completion": True,
        "evaluation.gradient_interference.layerwise": True,
        "training.token_budget": 12288,
        "training.max_steps": 3,
        "training.batch_size_per_process": 64,
        "training.warmup_steps": 0,
        "run.continuation.enabled": False,
        "outputs.save_checkpoints": False,
        "evaluation.validation.enabled": False,
        "evaluation.validation.run_at_completion": False,
        "evaluation.validation.interval_steps": 0,
        "training.eval_interval": 0,
        "training.eval_batches": None,
        "model.granularities": ["small", "full"],
        "model.granularity_prefixes": {"small": 0.5, "full": 1.0},
    }
    values.update(overrides)
    return resolve_run_config(
        "configs/opt-in_exps/panelgrad_uniform_baseline.yaml",
        output_dir=tmp_path / "panelgrad-uniform-baseline-001",
        overrides=values,
    )


def test_gradient_interference_is_disabled_by_default_and_resolves_collisions(tmp_path):
    disabled = resolve_run_config(
        "configs/opt-in_exps/panelgrad_uniform_baseline.yaml",
        output_dir=tmp_path / "disabled" / "panelgrad-uniform-baseline-001",
    )
    assert disabled["evaluation"]["gradient_interference"]["enabled"] is False

    resolved = _enabled_config(tmp_path)
    diagnostic = resolved["evaluation"]["gradient_interference"]
    assert diagnostic["resolved_steps"] == [0, 2, 3]
    assert diagnostic["milestone_reasons"]["0"] == [
        "trajectory_fraction:0",
        "warmup_completion",
    ]
    assert diagnostic["milestone_reasons"]["2"] == ["trajectory_fraction:0.5"]
    assert len(diagnostic["diagnostic_contract_hash"]) == 64


def test_exact_production_milestone_resolution(tmp_path):
    resolved = _enabled_config(
        tmp_path,
        **{
            "training.token_budget": 100_000_000,
            "training.max_steps": 12_207,
            "training.warmup_steps": 1_000,
            "evaluation.gradient_interference.trajectory_fractions": [
                0.0,
                0.25,
                0.5,
                0.75,
                1.0,
            ],
        },
    )
    assert resolved["evaluation"]["gradient_interference"]["resolved_steps"] == [
        0,
        1000,
        3052,
        6104,
        9156,
        12207,
    ]


@pytest.mark.parametrize(
    "overrides, match",
    [
        (
            {"evaluation.gradient_interference.trajectory_fractions": [-0.1]},
            "between zero and one",
        ),
        (
            {"evaluation.adaptive_controller.enabled": False},
            "adaptive_controller.enabled must be True",
        ),
        (
            {
                "model.granularities": ["full"],
                "model.granularity_prefixes": {"full": 1.0},
            },
            "at least two granularities",
        ),
        (
            {
                "model.granularity_sampling_mode": "fixed_global",
                "model.global_sampling_distribution": {
                    "small": 0.25,
                    "full": 0.75,
                },
            },
            "uniform global sampling",
        ),
        (
            {
                "run.sampling_mode": "nested-all",
                "training.granularity_sampling": "all",
            },
            "nested-random",
        ),
        (
            {"dataset.fixed_four_role_partition": False},
            "fixed_four_role_partition",
        ),
    ],
)
def test_gradient_interference_rejects_incompatible_configuration(
    tmp_path, overrides, match
):
    with pytest.raises(ConfigError, match=match):
        _enabled_config(tmp_path, **overrides)


def _entry(values, stop):
    return {
        "parameter_family": "gate_weight",
        "selection": [
            {"start": 0, "stop": stop, "step": None},
        ],
        "tensor": torch.tensor(values, dtype=torch.float32),
    }


def test_pairwise_statistics_use_only_smaller_nested_support_and_sum_layers():
    snapshots = {
        "small": {
            "layer_0000": {
                "module_name": "layers.0.mlp",
                "entries": {"gate.weight": _entry([1.0, 2.0], 2)},
            },
            "layer_0001": {
                "module_name": "layers.1.mlp",
                "entries": {"gate.weight": _entry([3.0], 1)},
            },
        },
        "full": {
            "layer_0000": {
                "module_name": "layers.0.mlp",
                "entries": {"gate.weight": _entry([2.0, 0.0, 1000.0], 3)},
            },
            "layer_0001": {
                "module_name": "layers.1.mlp",
                "entries": {"gate.weight": _entry([-1.0, -900.0], 2)},
            },
        },
    }
    pair = _pair_statistics("small", "full", snapshots, include_layerwise=True)
    assert pair["shared_parameter_count"] == 3
    assert pair["dot_product"] == pytest.approx(-1.0)
    assert pair["left_shared_squared_norm"] == pytest.approx(14.0)
    assert pair["right_shared_squared_norm"] == pytest.approx(5.0)
    assert pair["distance"] == pytest.approx(21.0**0.5)
    assert pair["cosine"] == pytest.approx(-1.0 / (14.0 * 5.0) ** 0.5)
    assert pair["dot_product"] == pytest.approx(
        sum(layer["dot_product"] for layer in pair["layer_contributions"])
    )
    assert pair["left_shared_squared_norm"] == pytest.approx(
        sum(layer["left_shared_squared_norm"] for layer in pair["layer_contributions"])
    )


def test_zero_norm_cosine_is_null_with_indicator():
    snapshots = {
        label: {
            "layer_0000": {
                "module_name": "layers.0.mlp",
                "entries": {"gate.weight": _entry(values, 2)},
            }
        }
        for label, values in (("small", [0.0, 0.0]), ("full", [1.0, 2.0]))
    }
    pair = _pair_statistics("small", "full", snapshots, include_layerwise=True)
    assert pair["cosine"] is None
    assert pair["has_zero_norm"] is True
    assert pair["zero_norm"] == {"left": True, "right": False}


class _ToyLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = ModifiedLlamaMLP(config)

    def configure_subnetwork(self, granularity):
        self.mlp.configure_subnetwork(granularity)


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model_config = LlamaConfig(
            vocab_size=32,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            max_position_embeddings=8,
            tie_word_embeddings=False,
        )
        model_config.granularities = ["small", "full"]
        model_config.granularity_prefixes = {"small": 0.5, "full": 1.0}
        self.matformer_layers = torch.nn.ModuleList(
            [_ToyLayer(model_config), _ToyLayer(model_config)]
        )
        self.current_granularity = "full"
        self.current_layer_granularities = ["full", "full"]
        self.current_granularity_pattern = {"before": True}
        self.current_sampling_mode = "global"
        self.configure_subnetwork("full")

    def configure_subnetwork(self, granularity):
        self.current_granularity = granularity
        self.current_layer_granularities = [granularity, granularity]
        for layer in self.matformer_layers:
            layer.configure_subnetwork(granularity)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 4) / 7.0
        for layer in self.matformer_layers:
            hidden = layer.mlp(hidden)
        return SimpleNamespace(loss=hidden.square().mean())


def _batches(batch_size):
    ids = torch.tensor(
        [[1, 2, 3, 4], [2, 4, 1, 3], [3, 1, 4, 2], [4, 3, 2, 1]],
        dtype=torch.long,
    )
    return [
        {
            "input_ids": ids[start : start + batch_size],
            "labels": ids[start : start + batch_size].clone(),
            "attention_mask": torch.ones_like(ids[start : start + batch_size]),
        }
        for start in range(0, len(ids), batch_size)
    ]


def test_neutral_probe_is_microbatch_invariant_and_restores_runtime():
    torch.manual_seed(7)
    model = _ToyModel()
    support = resolve_controlled_ffn_support(model, ["small", "full"])
    before_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    before_mode = model.training
    before_granularity = model.current_granularity
    one = measure_controlled_gradients(
        model,
        _batches(4),
        ["small", "full"],
        device="cpu",
        support_identity=support,
        retain_gradients=True,
    )
    split = measure_controlled_gradients(
        model,
        _batches(1),
        ["small", "full"],
        device="cpu",
        support_identity=support,
        retain_gradients=True,
    )
    assert [item["gradient_squared_norm"] for item in one["measurements"]] == (
        pytest.approx(
            [item["gradient_squared_norm"] for item in split["measurements"]],
            rel=1e-6,
            abs=1e-8,
        )
    )
    assert model.training is before_mode
    assert model.current_granularity == before_granularity
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(
        torch.equal(before, after)
        for before, after in zip(before_parameters, model.parameters(), strict=True)
    )


def test_neutral_probe_restores_after_injected_failure():
    model = _ToyModel()
    support = resolve_controlled_ffn_support(model, ["small", "full"])
    before = model.current_granularity
    bad_batches = _batches(4)
    bad_batches[0]["input_ids"] = torch.full_like(
        bad_batches[0]["input_ids"], 1.0e20, dtype=torch.float32
    )
    with pytest.raises(ControlledGradientProbeError, match="non-finite"):
        measure_controlled_gradients(
            model,
            bad_batches,
            ["small", "full"],
            device="cpu",
            support_identity=support,
        )
    assert model.current_granularity == before
    assert all(parameter.grad is None for parameter in model.parameters())


def _runtime_identity(config):
    diagnostic = config["evaluation"]["gradient_interference"]
    diagnostic["fixed_probe_manifest_hash"] = "probe-hash"
    diagnostic["controlled_support_hash"] = "support-hash"
    return build_gradient_interference_state(
        config,
        fixed_probe_manifest_hash="probe-hash",
        controlled_support_hash="support-hash",
    )


def _record(config, step):
    record = {
        "schema_version": 1,
        "event_type": "gradient_interference_snapshot",
        "snapshot_id": snapshot_id(config, step),
        "run_id": config["run"]["run_id"],
        "step": step,
        "tokens_seen": step * 10,
        "milestone_reasons": config["evaluation"]["gradient_interference"][
            "milestone_reasons"
        ][str(step)],
        "fixed_probe_manifest_hash": "probe-hash",
        "controlled_support_hash": "support-hash",
        "diagnostic_contract_hash": config["evaluation"]["gradient_interference"][
            "diagnostic_contract_hash"
        ],
        "semantics": {
            "gradient": "raw_pre_correction_pre_clipping",
            "loss_aggregation": "target_token_weighted_fixed_probe",
            "shared_support": "smaller_nested_controlled_ffn_support",
            "layerwise": True,
        },
        "granularities": [
            {
                "granularity": label,
                "controlled_parameter_count": 1,
                "aggregate_loss": 1.0,
                "gradient_squared_norm": 1.0,
                "gradient_norm": 1.0,
            }
            for label in ("small", "full")
        ],
        "pairs": [
            {
                "left_granularity": "small",
                "right_granularity": "full",
                "distance": 0.0,
                "shared_parameter_count": 1,
                "dot_product": 1.0,
                "left_shared_squared_norm": 1.0,
                "right_shared_squared_norm": 1.0,
                "left_shared_norm": 1.0,
                "right_shared_norm": 1.0,
                "cosine": 1.0,
                "has_zero_norm": False,
                "zero_norm": {"left": False, "right": False},
                "layer_contributions": [],
            }
        ],
        "cost": {
            "packed_sequences": 4,
            "batches": 2,
            "targets": 12,
            "packed_sequence_evaluations": 8,
            "target_evaluations": 24,
            "backward_evaluations": 4,
            "duration_seconds": 0.1,
        },
    }
    validate_snapshot_record(record, expected_granularity_count=2)
    return record


def test_journal_reconciliation_truncates_uncheckpointed_tail(tmp_path):
    config = _enabled_config(tmp_path)
    state = _runtime_identity(config)
    path = Path(state["journal"]["path"])
    first = _record(config, 0)
    commit = append_snapshot_record(path, first)
    record_snapshot_commit(state, first, commit)
    committed_payload = path.read_bytes()
    path.write_bytes(
        committed_payload + json.dumps(_record(config, 2)).encode() + b"\n"
    )

    records = reconcile_snapshot_journal(state, config=config, restored_step=1)
    assert [record["step"] for record in records] == [0]
    assert path.read_bytes() == committed_payload
    assert (
        state["journal"]["last_committed_hash"]
        == hashlib.sha256(committed_payload).hexdigest()
    )


def test_journal_rejects_corruption_duplicate_and_missing_history(tmp_path):
    config = _enabled_config(tmp_path)
    state = _runtime_identity(config)
    path = Path(state["journal"]["path"])
    assert state["journal"]["last_committed_hash"] == EMPTY_JOURNAL_HASH
    with pytest.raises(GradientInterferenceError, match="missing historical"):
        reconcile_snapshot_journal(state, config=config, restored_step=2)

    first = _record(config, 0)
    commit = append_snapshot_record(path, first)
    record_snapshot_commit(state, first, commit)
    payload = bytearray(path.read_bytes())
    payload[0] = ord("[")
    path.write_bytes(payload)
    with pytest.raises(GradientInterferenceError, match="hash mismatch"):
        reconcile_snapshot_journal(state, config=config, restored_step=1)

    path.write_bytes(
        json.dumps(first, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    state["journal"]["last_committed_hash"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    duplicate = copy.deepcopy(state)
    duplicate["completed_snapshot_ids"].append(first["snapshot_id"])
    duplicate["completed_steps"].append(0)
    duplicate["journal"]["event_count"] = 2
    with pytest.raises(GradientInterferenceError, match="completed snapshot"):
        reconcile_snapshot_journal(duplicate, config=config, restored_step=1)


def test_training_writes_step_zero_and_every_successful_milestone(tmp_path):
    config = _enabled_config(tmp_path)
    model = _ToyModel()
    rows = 700
    tokenized = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4]] * rows,
            "attention_mask": [[1, 1, 1, 1]] * rows,
        }
    )
    result = run_training(
        config,
        model=model,
        tokenized_dataset=tokenized,
        device="cpu",
    )
    journal_path = Path(config["run"]["output_dir"]) / "gradient_interference.jsonl"
    records = [json.loads(line) for line in journal_path.read_text().splitlines()]
    assert [record["step"] for record in records] == [0, 2, 3]
    assert len({record["snapshot_id"] for record in records}) == 3
    assert all(len(record["pairs"]) == 1 for record in records)
    assert all(
        len(record["pairs"][0]["layer_contributions"]) == 2 for record in records
    )

    summary = json.loads(Path(result["summary_path"]).read_text())
    assert summary["gradient_interference_snapshot_count"] == 3
    assert summary["gradient_interference_measured_steps"] == [0, 2, 3]
    assert summary["gradient_interference_expected_steps"] == [0, 2, 3]
    assert summary["gradient_interference_path"] == str(journal_path)
    assert len(summary["gradient_interference_journal_hash"]) == 64
    assert (
        summary["gradient_interference_measurement_cost"]["backward_evaluations"] == 12
    )


def test_append_failure_after_commit_resumes_due_milestone_without_duplicate(
    tmp_path, monkeypatch
):
    config = _enabled_config(
        tmp_path,
        **{
            "run.continuation.enabled": True,
            "outputs.save_checkpoints": True,
        },
    )
    rows = 700
    tokenized = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4]] * rows,
            "attention_mask": [[1, 1, 1, 1]] * rows,
        }
    )
    original_append = gradient_interference_module.append_snapshot_record

    def fail_step_two(path, record, **kwargs):
        if int(record["step"]) == 2:
            raise OSError("injected diagnostic append failure")
        return original_append(path, record, **kwargs)

    monkeypatch.setattr(
        gradient_interference_module, "append_snapshot_record", fail_step_two
    )
    with pytest.raises(GradientInterferenceError, match="append failure"):
        run_training(
            config,
            model=_ToyModel(),
            tokenized_dataset=tokenized,
            device="cpu",
        )
    checkpoint_path = Path(config["run"]["output_dir"]) / "checkpoints" / "latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 2
    assert checkpoint["gradient_interference_state"]["completed_steps"] == [0]

    monkeypatch.setattr(
        gradient_interference_module,
        "append_snapshot_record",
        original_append,
    )
    resumed_config = _enabled_config(
        tmp_path,
        **{
            "run.continuation.enabled": True,
            "outputs.save_checkpoints": True,
        },
    )
    result = run_training(
        resumed_config,
        model=_ToyModel(),
        tokenized_dataset=tokenized,
        device="cpu",
    )
    journal_path = (
        Path(resumed_config["run"]["output_dir"]) / "gradient_interference.jsonl"
    )
    records = [json.loads(line) for line in journal_path.read_text().splitlines()]
    assert [record["step"] for record in records] == [0, 2, 3]
    assert len({record["snapshot_id"] for record in records}) == 3
    summary = json.loads(Path(result["summary_path"]).read_text())
    assert summary["gradient_interference_measured_steps"] == [0, 2, 3]
