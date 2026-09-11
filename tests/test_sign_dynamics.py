import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from transformers import LlamaConfig

from src.models.ffn import CatLlamaMLP, ModifiedLlamaMLP
from src.training.sign_dynamics import (
    SignDynamicsError,
    SignDynamicsRuntime,
    resolve_sign_dynamics_support,
    strict_sign_flip_mask,
    update_hysteresis_state,
)
from src.training.run import run_training
from src.utils.config import ConfigError, resolve_run_config
from scripts.analyze_tinystories_sign_dynamics import (
    analyze,
    build_performance_correlation_rows,
    build_time_bin_rows,
    build_validation_dynamics_rows,
)


class TinySignModel(torch.nn.Module):
    def __init__(self, *, variant="slicing", layers=2):
        super().__init__()
        config = LlamaConfig(
            hidden_size=4,
            intermediate_size=16,
            num_hidden_layers=layers,
            num_attention_heads=1,
            num_key_value_heads=1,
            vocab_size=16,
            hidden_act="silu",
            mlp_bias=False,
        )
        config.granularities = ["narrow", "full"]
        config.granularity_prefixes = {"narrow": 0.5, "full": 1.0}
        mlp_class = CatLlamaMLP if variant == "concat" else ModifiedLlamaMLP
        self.layers = torch.nn.ModuleList(
            [
                mlp_class(
                    config,
                    trained_granularities=config.granularities,
                    gradient_membership_correction_enabled=False,
                )
                for _ in range(layers)
            ]
        )
        self.shared = torch.nn.Parameter(torch.tensor([-1.0, 1.0, 1.0]))


class TinySignTrainingModel(TinySignModel):
    def __init__(self):
        super().__init__(variant="slicing", layers=1)
        self.embedding = torch.nn.Embedding(16, 4)
        self.current_granularity = None

    def configure_subnetwork(self, granularity):
        self.current_granularity = str(granularity)
        for layer in self.layers:
            layer.configure_subnetwork(str(granularity))

    def forward(self, input_ids, attention_mask=None, labels=None):
        del attention_mask, labels
        hidden = self.embedding(input_ids)
        hidden = self.layers[0](hidden)
        loss = hidden.float().square().mean() + self.shared.float().square().mean()
        return type("Output", (), {"loss": loss})()


def _runtime_config(tmp_path, *, max_steps=1, variant="slicing"):
    milestones = [0, max_steps]
    contract = {
        "schema_version": 1,
        "event_type": "sign_dynamics_step",
        "campaign_id": "toy",
        "arm_id": "uniform_h1",
        "measurement_scope": "all_trainable_parameters",
        "cadence_steps": 1,
        "retention": "sufficient_state",
        "hysteresis_thresholds": [0.0, 0.001, 0.01],
        "snapshot_trajectory_fractions": [0.0, 1.0],
        "include_warmup_completion": False,
        "resolved_snapshot_steps": milestones,
        "snapshot_milestone_reasons": {
            "0": ["trajectory_fraction:0"],
            str(max_steps): ["trajectory_fraction:1"],
        },
    }
    from src.utils.reproducibility import stable_hash

    contract["diagnostic_contract_hash"] = stable_hash(
        {
            key: value
            for key, value in contract.items()
            if key not in {"resolved_snapshot_steps", "snapshot_milestone_reasons"}
        }
    )
    return {
        "run": {"output_dir": str(tmp_path), "seed": 42},
        "model": {
            "granularities": ["narrow", "full"],
            "variant": variant,
            "granularity_sampling_mode": "global",
            "global_sampling_schedule": "random_with_replacement",
        },
        "training": {
            "optimizer_state_scope": "shared",
            "mixed_precision": "bf16",
        },
        "controlled_experiment": {
            "budget_unit_tokens": 8,
            "budget_multiplier": 1,
            "policy": "uniform",
        },
        "evaluation": {"sign_dynamics": contract},
    }


@pytest.mark.parametrize("variant", ["slicing", "concat"])
def test_support_is_disjoint_exhaustive_and_includes_every_layer_projection(variant):
    model = TinySignModel(variant=variant, layers=2)
    manifest, segments = resolve_sign_dynamics_support(model, ["narrow", "full"])
    total = sum(parameter.numel() for parameter in model.parameters())
    assert manifest["trainable_parameter_count"] == total
    assert manifest["coverage"] == {
        "disjoint": True,
        "exhaustive": True,
        "covered_coordinate_count": total,
    }
    assert manifest["audit_strata"]["counts"]["shared"] == 3
    assert manifest["audit_strata"]["counts"]["ffn_B1"] == 192
    assert manifest["audit_strata"]["counts"]["ffn_B2"] == 192
    controlled = [segment for segment in segments if segment.stratum != "shared"]
    assert {segment.layer for segment in controlled} == {0, 1}
    assert {segment.family for segment in controlled} == {
        "gate_weight",
        "up_weight",
        "down_weight",
    }


def test_exact_raw_and_hysteretic_transition_semantics():
    previous = torch.tensor([-1.0, 1.0, -1.0, 0.0])
    current = torch.tensor([1.0, 0.0, -0.5, -1.0])
    assert strict_sign_flip_mask(previous, current).tolist() == [
        True,
        False,
        False,
        False,
    ]

    state = torch.tensor([-1, 1, 0, 0], dtype=torch.int8)
    normalized = torch.tensor([0.02, 0.001, -0.02, 0.0])
    new_state, transitions, establishments = update_hysteresis_state(
        state, normalized, 0.01
    )
    assert new_state.tolist() == [1, 1, -1, 0]
    assert transitions.tolist() == [True, False, False, False]
    assert establishments.tolist() == [False, False, True, False]


def test_time_bin_block_uncertainty_is_linear_and_parallel_equivalent():
    class CountingRows(list):
        def __init__(self, values):
            super().__init__(values)
            self.iteration_count = 0

        def __iter__(self):
            self.iteration_count += 1
            return super().__iter__()

    base_rows = [
        {
            "budget_multiplier": 1,
            "arm_id": "uniform_h1",
            "seed": 42,
            "step": step,
            "action": "narrow",
            "band": "B1",
            "threshold": 0.0,
            "h_window_index": (step - 1) // 2,
            "h_window_interval_steps": 2,
            **{
                field: float(value)
                for field in (
                    "raw_sign_flip_rate",
                    "robust_transition_rate",
                    "gradient_rms",
                    "relative_update_rms",
                    "changed_coordinate_rate",
                    "dead_band_occupancy",
                    "last_robust_transition_step",
                )
            },
        }
        for step, value in enumerate((1.0, 3.0, 5.0, 7.0), start=1)
    ]
    rows = CountingRows(base_rows + [{**row, "action": "full"} for row in base_rows])

    serial = build_time_bin_rows(rows, time_bin_count=1, workers=1)
    assert rows.iteration_count == 2

    parallel = build_time_bin_rows(list(rows), time_bin_count=1, workers=2)
    assert parallel == serial
    assert serial[0]["raw_sign_flip_rate"] == pytest.approx(4.0)
    assert serial[0]["raw_sign_flip_rate_uncertainty"] == pytest.approx(2.0)


def test_validation_dynamics_uses_cumulative_width_support(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text(
        "step,split,granularity,loss,perplexity,tokens_seen,learning_rate\n"
        "1,train,narrow,4.0,54.6,8,0.01\n"
        "2,train,full,3.0,20.1,16,0.01\n"
        "2,validation,narrow,3.0,20.1,16,\n"
        "2,validation,full,4.0,54.6,16,\n"
        "3,train,narrow,2.5,12.2,24,0.005\n"
        "4,train,full,2.0,7.4,32,0.005\n"
        "4,validation,narrow,2.0,7.4,32,\n"
        "4,validation,full,2.0,7.4,32,\n",
        encoding="utf-8",
    )

    def band(count):
        return {
            "coordinate_count": count,
            "raw_sign_flip_count": 1,
            "changed_coordinate_count": count,
            "gradient_rms": 2.0,
            "update_rms": 0.5,
            "relative_update_rms": 0.25,
            "hysteresis": {
                "0": {"transition_count": 1, "dead_band_occupancy": 0.0},
                "0.001": {
                    "transition_count": 0,
                    "dead_band_occupancy": 0.25,
                },
            },
        }

    records = [
        {
            "step": step,
            "selected_granularity_index": selected,
            "bands": {"B1": band(2), "B2": band(1)},
        }
        for step, selected in enumerate((0, 1, 0, 1), start=1)
    ]
    run = {
        "run_dir": tmp_path,
        "summary": {"metrics_path": str(metrics_path)},
        "config": {
            "evaluation": {
                "sign_dynamics": {"hysteresis_thresholds": [0.0, 0.001]}
            }
        },
        "support": {
            "ordered_granularities": ["narrow", "full"],
            "primary_partition": {"bands": ["B1", "B2"]},
        },
        "records": records,
        "budget_multiplier": 1,
        "arm": "uniform_h1",
        "seed": 42,
    }
    rows = build_validation_dynamics_rows([run])
    assert len(rows) == 8
    second_narrow = next(
        row
        for row in rows
        if row["granularity"] == "narrow"
        and row["validation_step"] == 4
        and row["threshold"] == 0.0
    )
    second_full = next(
        row
        for row in rows
        if row["granularity"] == "full"
        and row["validation_step"] == 4
        and row["threshold"] == 0.0
    )
    assert second_narrow["active_bands"] == '["B1"]'
    assert second_narrow["raw_sign_flip_rate"] == pytest.approx(0.5)
    assert second_narrow["validation_loss_improvement_per_1000_steps"] == 500.0
    assert second_narrow["parameter_support_exposure_rate"] == 1.0
    assert second_full["active_bands"] == '["B1", "B2"]'
    assert second_full["raw_sign_flip_rate"] == pytest.approx(2.0 / 3.0)
    assert second_full["validation_loss_improvement_per_1000_steps"] == 1000.0
    assert second_full["parameter_support_exposure_rate"] == 0.5


def test_performance_correlations_report_concurrent_and_lagged_relationships():
    rows = []
    for index in range(1, 7):
        value = index / 100.0
        rows.append(
            {
                "budget_multiplier": 1,
                "arm_id": "uniform_h1",
                "seed": 42,
                "granularity": "narrow",
                "threshold": 0.0,
                "validation_step": index,
                "trajectory_fraction": index / 10.0,
                "validation_loss_improvement_per_1000_steps": value,
                "raw_sign_flip_rate": value,
                "raw_flips_per_relative_update": value / 0.5,
                "mean_learning_rate": 1.0 - index / 20.0,
                "relative_update_rms": 0.5,
                "parameter_support_exposure_rate": 1.0,
            }
        )
    correlations = build_performance_correlation_rows(rows)
    concurrent = next(
        row
        for row in correlations
        if row["metric"] == "raw_sign_flip_rate"
        and row["relationship"] == "concurrent"
        and row["phase"] == "all"
    )
    lagged = next(
        row
        for row in correlations
        if row["metric"] == "raw_sign_flip_rate"
        and row["relationship"] == "next_interval"
        and row["phase"] == "all"
    )
    assert concurrent["observation_count"] == 6
    assert concurrent["pearson"] == pytest.approx(1.0)
    assert concurrent["spearman"] == pytest.approx(1.0)
    assert lagged["observation_count"] == 5
    assert lagged["pearson"] == pytest.approx(1.0)


def test_runtime_exact_gradient_update_dead_band_and_transition_aggregates(tmp_path):
    model = TinySignModel(variant="slicing", layers=1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.0)
        model.shared.copy_(torch.tensor([-1.0, 1.0, 1.0]))
    runtime = SignDynamicsRuntime.build(_runtime_config(tmp_path), model)
    runtime.install_support_contract(resuming=False)
    runtime.initialize_fresh()
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    model.shared.grad.copy_(torch.tensor([3.0, 4.0, 0.0]))
    action = {"kind": "global", "granularities": ["narrow"]}
    runtime.capture_pre_update(step=1, action=action)
    with torch.no_grad():
        model.shared.copy_(torch.tensor([1.0, 0.0, 0.5]))
    record = runtime.commit_step(step=1, action=action)

    band = record["bands"]["B1"]
    assert band["coordinate_count"] == 99
    assert band["nonzero_gradient_count"] == 2
    assert band["gradient_rms"] == pytest.approx((25.0 / 99.0) ** 0.5)
    assert band["changed_coordinate_count"] == 3
    assert band["update_rms"] == pytest.approx((5.25 / 99.0) ** 0.5)
    assert band["relative_update_rms"] == pytest.approx((5.25 / 99.0) ** 0.5)
    assert band["raw_sign_flip_count"] == 1
    robust = band["hysteresis"]["0"]
    assert robust["transition_count"] == 1
    assert robust["dead_count"] == 1
    assert robust["dead_entries"] == 1
    assert robust["dead_exits"] == 0
    assert robust["dead_band_occupancy"] == pytest.approx(1.0 / 99.0)


def test_runtime_records_every_band_and_optimizer_mediated_off_support_update(tmp_path):
    model = TinySignModel(variant="slicing", layers=1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.0)
        model.shared.copy_(torch.tensor([-1.0, 1.0, 1.0]))
    runtime = SignDynamicsRuntime.build(
        _runtime_config(tmp_path, variant="slicing"), model
    )
    runtime.install_support_contract(resuming=False)
    runtime.initialize_fresh()
    assert (
        runtime.support_manifest["diagnostic_contract_hash"]
        == (runtime.diagnostic["diagnostic_contract_hash"])
    )
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
    action = {"kind": "global", "granularities": ["narrow"]}
    runtime.capture_pre_update(step=1, action=action)
    optimizer.step()
    with torch.no_grad():
        model.shared[0] = 1.0
    record = runtime.commit_step(step=1, action=action)

    assert record["bands"]["B2"]["forward_active"] is False
    assert record["bands"]["B2"]["nonzero_gradient_count"] == 0
    assert record["bands"]["B2"]["changed_coordinate_count"] > 0
    assert record["bands"]["B1"]["raw_sign_flip_count"] == 1
    assert record["bands"]["B1"]["gradient_rms"] == 0.0
    assert record["sampled_probability"] == 0.5
    assert record["sampled_probability_semantics"] == "uniform_draw_probability"
    assert len((tmp_path / "sign_dynamics.jsonl").read_text().splitlines()) == 1
    assert (
        runtime.journal_hash
        == hashlib.sha256((tmp_path / "sign_dynamics.jsonl").read_bytes()).hexdigest()
    )
    assert set(runtime.snapshot_identities) == {"0", "1"}


def test_concat_inactive_parameters_with_none_grad_do_not_move(tmp_path):
    model = TinySignModel(variant="concat", layers=1)
    runtime = SignDynamicsRuntime.build(
        _runtime_config(tmp_path, variant="concat"), model
    )
    runtime.install_support_contract(resuming=False)
    runtime.initialize_fresh()
    for segment in runtime.segments:
        if segment.stratum in {"shared", "ffn_B1"}:
            segment.parameter.grad = torch.zeros_like(segment.parameter)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
    action = {"kind": "global", "granularities": ["narrow"]}
    runtime.capture_pre_update(step=1, action=action)
    optimizer.step()
    record = runtime.commit_step(step=1, action=action)
    assert record["bands"]["B2"]["gradient_present_coordinate_count"] == 0
    assert record["bands"]["B2"]["changed_coordinate_count"] == 0


def test_resume_truncates_uncheckpointed_tail_and_rejects_missing_snapshot(tmp_path):
    model = TinySignModel(layers=1)
    runtime = SignDynamicsRuntime.build(_runtime_config(tmp_path), model)
    runtime.install_support_contract(resuming=False)
    runtime.initialize_fresh()
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    action = {"kind": "global", "granularities": ["narrow"]}
    runtime.capture_pre_update(step=1, action=action)
    with torch.no_grad():
        model.shared[0] *= -1
    runtime.commit_step(step=1, action=action)
    state = runtime.state_dict()
    journal = tmp_path / "sign_dynamics.jsonl"
    with journal.open("a", encoding="utf-8") as destination:
        destination.write('{"step":2,"uncheckpointed":true}\n')

    restored = SignDynamicsRuntime.build(_runtime_config(tmp_path), model)
    restored.install_support_contract(resuming=True)
    validated = restored.validate_checkpoint_state(state, expected_step=1)
    assert len(journal.read_text().splitlines()) == 1
    restored.restore_validated_state(validated)
    assert restored.state_dict()["journal_hash"] == state["journal_hash"]

    Path(state["snapshot_identities"]["1"]["path"]).unlink()
    with pytest.raises(SignDynamicsError, match="snapshot"):
        restored.validate_checkpoint_state(state, expected_step=1)


def test_config_resolves_exact_contract_and_rejects_non_global_scope(tmp_path):
    overrides = {
        "run.sampling_mode": "nested-random",
        "training.granularity_sampling": "random",
        "model.granularity_sampling_mode": "global",
        "evaluation.sign_dynamics.enabled": True,
        "evaluation.validation.enabled": False,
        "evaluation.validation.run_at_completion": False,
    }
    resolved = resolve_run_config(
        "tests/fixtures/explicit_granularity_smoke.yaml",
        output_dir=tmp_path / "explicit-granularity-smoke-001",
        overrides=overrides,
    )
    diagnostic = resolved["evaluation"]["sign_dynamics"]
    assert diagnostic["measurement_scope"] == "all_trainable_parameters"
    assert diagnostic["cadence_steps"] == 1
    assert diagnostic["retention"] == "sufficient_state"
    assert diagnostic["resolved_snapshot_steps"][0] == 0
    assert (
        diagnostic["resolved_snapshot_steps"][-1] == resolved["training"]["max_steps"]
    )

    invalid = copy.deepcopy(overrides)
    invalid["model.granularity_sampling_mode"] = "per_block"
    with pytest.raises(ConfigError, match="supports only global"):
        resolve_run_config(
            "tests/fixtures/explicit_granularity_smoke.yaml",
            output_dir=tmp_path / "invalid" / "explicit-granularity-smoke-001",
            overrides=invalid,
        )


def test_cpu_end_to_end_has_complete_step_snapshot_summary_and_figures(tmp_path):
    run_dir = tmp_path / "campaign" / "x1" / "runs" / "balanced_h1" / "s42"
    config = resolve_run_config(
        "tests/fixtures/per_granularity_optimizer_smoke.yaml",
        output_dir=run_dir,
        overrides={
            "run.run_id": "s42",
            "run.continuation.enabled": True,
            "training.optimizer.state_scope": "shared",
            "training.token_budget": 128,
            "training.max_steps": 2,
            "training.batch_size_per_process": 1,
            "training.gradient_accumulation_steps": 2,
            "training.warmup_steps": 0,
            "evaluation.validation.enabled": False,
            "evaluation.validation.run_at_completion": False,
            "evaluation.validation.interval_steps": 0,
            "evaluation.validation.holdout.examples": 1,
            "training.eval_interval": 0,
            "training.eval_batches": None,
            "evaluation.sign_dynamics.enabled": True,
            "evaluation.sign_dynamics.campaign_id": "toy_e2e",
            "evaluation.sign_dynamics.arm_id": "balanced_h1",
            "evaluation.sign_dynamics.snapshot_trajectory_fractions": [0.0, 1.0],
            "controlled_experiment.budget_unit_tokens": 128,
            "controlled_experiment.budget_multiplier": 1,
            "controlled_experiment.arm_id": "balanced_h1",
            "controlled_experiment.policy": "balanced",
        },
    )
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3], [3, 2, 1]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]],
        }
    )
    result = run_training(
        config,
        model=TinySignTrainingModel(),
        tokenized_dataset=dataset,
        device="cpu",
    )
    summary = json.loads(Path(result["summary_path"]).read_text())
    records = [
        json.loads(line)
        for line in (run_dir / "sign_dynamics.jsonl").read_text().splitlines()
    ]
    assert summary["sign_dynamics_step_coverage_complete"] is True
    assert summary["sign_dynamics_measured_steps"] == 2
    assert [record["step"] for record in records] == [1, 2]
    assert [record["sampled_probability"] for record in records] == [1.0, 1.0]
    assert [record["h_window_position"] for record in records] == [1, 1]
    assert set(summary["sign_dynamics_snapshot_identities"]) == {"0", "2"}

    analysis = analyze(
        campaign_root=tmp_path / "campaign",
        analysis_dir=tmp_path / "campaign" / "analysis",
        figures_dir=tmp_path / "campaign" / "figures",
        time_bins=2,
    )
    assert analysis["status"] == "provisional"
    assert analysis["performance_analysis_status"] == "no_validation_records"
    assert len(analysis["primary_figures"]) == 7
    assert len(analysis["diagnostic_figures"]) == 7
    assert len(analysis["figures"]) == 14
    assert all(Path(path).is_file() for path in analysis["figures"])


def test_interrupted_and_resumed_run_matches_uninterrupted_diagnostic_state(tmp_path):
    class FailBeforeSecondCommit(TinySignTrainingModel):
        def __init__(self):
            super().__init__()
            self.training_forwards = 0

        def forward(self, *args, **kwargs):
            if self.training:
                self.training_forwards += 1
                if self.training_forwards == 2:
                    raise RuntimeError("simulated sign-dynamics interruption")
            return super().forward(*args, **kwargs)

    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3] for _ in range(8)],
            "attention_mask": [[1, 1, 1] for _ in range(8)],
        }
    )

    def make_config(run_dir):
        return resolve_run_config(
            "tests/fixtures/per_granularity_optimizer_smoke.yaml",
            output_dir=run_dir,
            overrides={
                "run.run_id": "s42",
                "run.continuation.enabled": True,
                "run.continuation.latest_checkpoint_save_interval_steps": 1,
                "training.optimizer.state_scope": "shared",
                "training.scheduler.name": "constant",
                "training.token_budget": 128,
                "training.max_steps": 4,
                "training.batch_size_per_process": 1,
                "training.warmup_steps": 0,
                "evaluation.validation.enabled": False,
                "evaluation.validation.run_at_completion": False,
                "evaluation.validation.interval_steps": 0,
                "evaluation.validation.holdout.examples": 1,
                "training.eval_interval": 0,
                "training.eval_batches": None,
                "evaluation.sign_dynamics.enabled": True,
                "evaluation.sign_dynamics.campaign_id": "toy_resume",
                "evaluation.sign_dynamics.arm_id": "balanced_h1",
                "evaluation.sign_dynamics.snapshot_trajectory_fractions": [
                    0.0,
                    0.5,
                    1.0,
                ],
                "evaluation.sign_dynamics.include_warmup_completion": False,
                "controlled_experiment.budget_unit_tokens": 128,
                "controlled_experiment.budget_multiplier": 1,
                "controlled_experiment.arm_id": "balanced_h1",
                "controlled_experiment.policy": "balanced",
            },
        )

    torch.manual_seed(1234)
    initial = TinySignTrainingModel().state_dict()

    uninterrupted_dir = tmp_path / "uninterrupted" / "s42"
    uninterrupted_model = TinySignTrainingModel()
    uninterrupted_model.load_state_dict(initial)
    run_training(
        make_config(uninterrupted_dir),
        model=uninterrupted_model,
        tokenized_dataset=dataset,
        device="cpu",
    )

    resumed_dir = tmp_path / "resumed" / "s42"
    interrupted_model = FailBeforeSecondCommit()
    interrupted_model.load_state_dict(initial)
    with pytest.raises(RuntimeError, match="sign-dynamics interruption"):
        run_training(
            make_config(resumed_dir),
            model=interrupted_model,
            tokenized_dataset=dataset,
            device="cpu",
        )
    assert len((resumed_dir / "sign_dynamics.jsonl").read_text().splitlines()) == 1

    resumed_model = TinySignTrainingModel()
    resumed_model.load_state_dict(initial)
    run_training(
        make_config(resumed_dir),
        model=resumed_model,
        tokenized_dataset=dataset,
        device="cpu",
    )

    for name, value in uninterrupted_model.state_dict().items():
        assert torch.equal(value, resumed_model.state_dict()[name]), name
    assert (uninterrupted_dir / "sign_dynamics.jsonl").read_bytes() == (
        resumed_dir / "sign_dynamics.jsonl"
    ).read_bytes()

    uninterrupted_checkpoint = torch.load(
        uninterrupted_dir / "checkpoints/latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    resumed_checkpoint = torch.load(
        resumed_dir / "checkpoints/latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    left_state = uninterrupted_checkpoint["sign_dynamics_state"]
    right_state = resumed_checkpoint["sign_dynamics_state"]
    for field in (
        "previous_values",
        "hysteresis_states",
        "last_transition_steps",
        "last_transition_actions",
        "establishment_steps",
        "establishment_actions",
    ):
        left_tensors = left_state[field]
        right_tensors = right_state[field]
        if field != "previous_values":
            for threshold in left_tensors:
                for name in left_tensors[threshold]:
                    assert torch.equal(
                        left_tensors[threshold][name], right_tensors[threshold][name]
                    )
        else:
            for name in left_tensors:
                assert torch.equal(left_tensors[name], right_tensors[name])
    assert left_state["journal_hash"] == right_state["journal_hash"]

    for step in (0, 2, 4):
        left_path = (
            uninterrupted_dir / "sign_dynamics_snapshots" / f"step-{step:08d}.pt"
        )
        right_path = resumed_dir / "sign_dynamics_snapshots" / f"step-{step:08d}.pt"
        left_snapshot = torch.load(left_path, map_location="cpu", weights_only=False)
        right_snapshot = torch.load(right_path, map_location="cpu", weights_only=False)
        assert left_snapshot.keys() == right_snapshot.keys()
        for name, value in left_snapshot["parameters"].items():
            assert torch.equal(value, right_snapshot["parameters"][name])
        assert (
            left_state["snapshot_identities"][str(step)]["sha256"]
            == (right_state["snapshot_identities"][str(step)]["sha256"])
        )
