from types import SimpleNamespace
import copy
import random

import pytest
import torch
import numpy as np
from transformers import LlamaConfig

from src.models.ffn import ModifiedLlamaMLP
from src.training.panelgrad import (
    PanelGradController,
    PanelGradError,
    build_probability_snapshot,
    epsilon_at_schedule_step,
    gradient_rms_from_squared_norm,
    measure_panelgrad_gradients,
    resolve_controlled_ffn_support,
)


class _ToyPanelGradLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = ModifiedLlamaMLP(config)

    def configure_subnetwork(self, granularity):
        self.mlp.configure_subnetwork(granularity)


class _ToyPanelGradModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            max_position_embeddings=8,
            tie_word_embeddings=False,
        )
        config.granularities = ["small", "full"]
        config.granularity_prefixes = {"small": 0.5, "full": 1.0}
        self.matformer_layers = torch.nn.ModuleList([_ToyPanelGradLayer(config)])
        self.current_granularity = None
        self.current_layer_granularities = None
        self.current_granularity_pattern = None
        self.current_sampling_mode = "global"

    def configure_subnetwork(self, granularity):
        self.current_granularity = granularity
        self.current_layer_granularities = [granularity]
        for layer in self.matformer_layers:
            layer.configure_subnetwork(granularity)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = input_ids.float().unsqueeze(-1).repeat(1, 1, 4) / 7.0
        output = self.matformer_layers[0].mlp(x)
        return SimpleNamespace(loss=output.square().mean())


def _panel_batches(batch_size):
    input_ids = torch.tensor(
        [[1, 2, 3, 4], [2, 4, 1, 3], [3, 1, 4, 2], [4, 3, 2, 1]],
        dtype=torch.long,
    )
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    return [
        {
            "input_ids": input_ids[start : start + batch_size],
            "labels": labels[start : start + batch_size],
            "attention_mask": attention_mask[start : start + batch_size],
        }
        for start in range(0, len(input_ids), batch_size)
    ]


def _support_identity(labels=("small", "full")):
    model = _ToyPanelGradModel()
    return resolve_controlled_ffn_support(model, labels)


def _measurement(scores=(1.0, 2.0)):
    labels = ["small", "full"]
    return {
        "measurements": [
            {
                "granularity": label,
                "controlled_parameter_count": 1,
                "aggregate_loss": 0.0,
                "gradient_squared_norm": score * score,
                "gradient_norm": score,
                "gradient_rms_score": score,
            }
            for label, score in zip(labels, scores, strict=True)
        ],
        "controller_example_count": 4,
        "controller_target_count": 12,
        "backward_evaluation_count": 4,
        "duration_seconds": 0.1,
    }


def _controller(seed=13, interval=2, epsilon=0.1, epsilon_schedule=None):
    return PanelGradController(
        ordered_granularities=["small", "full"],
        refresh_interval_steps=interval,
        eta=1e-12,
        temperature=1.0,
        epsilon=None if epsilon_schedule is not None else epsilon,
        epsilon_schedule=epsilon_schedule,
        sampling_seed=seed,
        support_identity=_support_identity(),
    )


@pytest.mark.parametrize(
    ("schedule", "steps", "expected"),
    [
        (
            {"type": "linear", "start": 0.5, "end": 0.1, "duration_steps": 100},
            (0, 50, 100, 150),
            (0.5, 0.3, 0.1, 0.1),
        ),
        (
            {"type": "linear", "start": 0.1, "end": 0.5, "duration_steps": 100},
            (0, 50, 100, 150),
            (0.1, 0.3, 0.5, 0.5),
        ),
    ],
)
def test_linear_epsilon_schedule_interpolates_and_clamps(schedule, steps, expected):
    assert [epsilon_at_schedule_step(schedule, step) for step in steps] == (
        pytest.approx(expected)
    )


def test_gradient_rms_uses_stable_controlled_parameter_count():
    norm, score = gradient_rms_from_squared_norm(36.0, 9)
    assert norm == 6.0
    assert score == 2.0
    assert gradient_rms_from_squared_norm(0.0, 9) == (0.0, 0.0)


def test_probability_mapping_covers_zero_one_arm_temperature_and_epsilon():
    all_zero = build_probability_snapshot(
        [0.0, 0.0, 0.0], eta=1e-12, temperature=1.0, epsilon=0.0
    )
    assert all_zero["q"] == pytest.approx([1 / 3] * 3)
    assert all_zero["p"] == pytest.approx([1 / 3] * 3)

    one_arm = build_probability_snapshot(
        [7.0], eta=1e-12, temperature=0.2, epsilon=0.8
    )
    assert one_arm["q"] == [1.0]
    assert one_arm["p"] == [1.0]

    tempered = build_probability_snapshot(
        [1.0, 4.0], eta=1e-300, temperature=2.0, epsilon=0.0
    )
    assert tempered["q"] == pytest.approx([1 / 3, 2 / 3])
    uniform = build_probability_snapshot(
        [1.0, 99.0], eta=1e-12, temperature=1.0, epsilon=1.0
    )
    assert uniform["p"] == pytest.approx([0.5, 0.5])


def test_probability_mapping_is_stable_for_extreme_finite_scores():
    snapshot = build_probability_snapshot(
        [1e-300, 1e300], eta=1e-300, temperature=0.25, epsilon=0.1
    )
    assert sum(snapshot["q"]) == pytest.approx(1.0)
    assert sum(snapshot["p"]) == pytest.approx(1.0)
    assert snapshot["p"][0] >= 0.05


@pytest.mark.parametrize(
    "scores, kwargs, match",
    [
        ([], {"eta": 1e-12, "temperature": 1.0, "epsilon": 0.1}, "at least one"),
        ([-1.0], {"eta": 1e-12, "temperature": 1.0, "epsilon": 0.1}, "nonnegative"),
        ([float("nan")], {"eta": 1e-12, "temperature": 1.0, "epsilon": 0.1}, "finite"),
        ([1.0], {"eta": 0.0, "temperature": 1.0, "epsilon": 0.1}, "eta must be positive"),
        ([1.0], {"eta": 1e-12, "temperature": 0.0, "epsilon": 0.1}, "temperature must be positive"),
        ([1.0], {"eta": 1e-12, "temperature": 1.0, "epsilon": 1.1}, "between zero and one"),
    ],
)
def test_probability_mapping_rejects_invalid_numerics(scores, kwargs, match):
    with pytest.raises(PanelGradError, match=match):
        build_probability_snapshot(scores, **kwargs)


def test_panelgrad_controller_refresh_draw_commit_and_boundary_transitions():
    controller = _controller(interval=2)
    assert controller.phase == "refresh_pending"
    with pytest.raises(PanelGradError, match="complete refresh"):
        controller.sample_action()

    refresh = controller.install_refresh(_measurement(), boundary_step=0)
    frozen_p = list(refresh["p"])
    first = controller.sample_action()
    controller.commit_pending_action(completed_step=1)
    assert controller.phase == "active_interval"
    second = controller.sample_action()
    controller.commit_pending_action(completed_step=2)

    state = controller.state_dict()
    assert state["refresh"]["p"] == frozen_p
    assert state["refresh"]["phase"] == "refresh_pending"
    assert state["refresh"]["completed_steps_since_refresh"] == 2
    assert state["sampling"]["sample_count"] == 2
    assert sum(state["sampling"]["exposure_counts"].values()) == 2
    assert first["global_granularity"] in {"small", "full"}
    assert second["global_granularity"] in {"small", "full"}


def test_scheduled_epsilon_updates_only_at_exact_refresh_boundaries_and_freezes_p():
    controller = _controller(
        interval=2,
        epsilon_schedule={
            "type": "linear",
            "start": 0.5,
            "end": 0.1,
            "duration_steps": 4,
        },
    )

    initial = controller.install_refresh(_measurement(), boundary_step=100)
    assert initial["active_epsilon"] == pytest.approx(0.5)
    assert initial["epsilon_schedule_step"] == 0
    initial_p = list(initial["p"])
    for completed_step in (101, 102):
        controller.sample_action()
        controller.commit_pending_action(completed_step=completed_step)
        assert controller.state_dict()["refresh"]["p"] == initial_p

    midpoint = controller.install_refresh(_measurement(), boundary_step=102)
    assert midpoint["active_epsilon"] == pytest.approx(0.3)
    assert midpoint["epsilon_schedule_step"] == 2
    midpoint_p = list(midpoint["p"])
    for completed_step in (103, 104):
        controller.sample_action()
        controller.commit_pending_action(completed_step=completed_step)
        assert controller.state_dict()["refresh"]["p"] == midpoint_p

    endpoint = controller.install_refresh(_measurement(), boundary_step=104)
    assert endpoint["active_epsilon"] == pytest.approx(0.1)
    assert endpoint["epsilon_schedule_step"] == 4


def test_scheduled_epsilon_excludes_failed_steps_and_failed_refreshes():
    controller = _controller(
        interval=2,
        epsilon_schedule={
            "type": "linear",
            "start": 0.5,
            "end": 0.1,
            "duration_steps": 4,
        },
    )
    controller.install_refresh(_measurement(), boundary_step=80)
    controller.sample_action()
    controller.rollback_pending_action()
    state_after_rollback = controller.state_dict()
    assert state_after_rollback["sampling"]["sample_count"] == 0
    assert state_after_rollback["refresh"]["active_epsilon"] == pytest.approx(0.5)

    for completed_step in (81, 82):
        controller.sample_action()
        controller.commit_pending_action(completed_step=completed_step)
    before_failed_refresh = controller.state_dict()["refresh"]
    invalid = _measurement()
    invalid["measurements"][1]["gradient_rms_score"] = float("nan")
    with pytest.raises(PanelGradError, match="finite"):
        controller.install_refresh(invalid, boundary_step=82)
    assert controller.state_dict()["refresh"] == before_failed_refresh

    refreshed = controller.install_refresh(_measurement(), boundary_step=82)
    assert refreshed["active_epsilon"] == pytest.approx(0.3)
    assert refreshed["epsilon_schedule_step"] == 2


def test_categorical_draw_sequence_is_deterministic_and_rollback_retries_action():
    left = _controller(seed=917)
    right = _controller(seed=917)
    for controller in (left, right):
        controller.install_refresh(_measurement(scores=(1.0, 3.0)), boundary_step=0)

    left_actions = []
    right_actions = []
    for step in range(1, 5):
        left_actions.append(left.sample_action()["global_granularity"])
        left.commit_pending_action(completed_step=step)
        right_actions.append(right.sample_action()["global_granularity"])
        right.commit_pending_action(completed_step=step)
        if step == 2:
            left.install_refresh(_measurement(scores=(1.0, 3.0)), boundary_step=2)
            right.install_refresh(_measurement(scores=(1.0, 3.0)), boundary_step=2)
    assert left_actions == right_actions

    retry = _controller(seed=5)
    retry.install_refresh(_measurement(), boundary_step=0)
    first = retry.sample_action()
    retry.rollback_pending_action()
    assert retry.sample_action() == first


def test_refresh_installation_is_atomic_on_invalid_measurement():
    controller = _controller()
    before = controller.state_dict()
    invalid = _measurement()
    invalid["measurements"][1]["gradient_rms_score"] = float("nan")

    with pytest.raises(PanelGradError, match="finite"):
        controller.install_refresh(invalid, boundary_step=0)

    assert controller.state_dict()["refresh"] == before["refresh"]


def test_aggregate_gradient_rms_is_microbatch_invariant_and_exact():
    torch.manual_seed(41)
    full_batch_model = _ToyPanelGradModel()
    split_batch_model = _ToyPanelGradModel()
    split_batch_model.load_state_dict(full_batch_model.state_dict())

    full = measure_panelgrad_gradients(
        full_batch_model,
        _panel_batches(4),
        ["small", "full"],
        device="cpu",
    )
    split = measure_panelgrad_gradients(
        split_batch_model,
        _panel_batches(1),
        ["small", "full"],
        device="cpu",
    )

    assert [item["controlled_parameter_count"] for item in full["measurements"]] == [
        48,
        96,
    ]
    for full_item, split_item in zip(
        full["measurements"], split["measurements"], strict=True
    ):
        assert full_item["aggregate_loss"] == pytest.approx(
            split_item["aggregate_loss"], rel=1e-6, abs=1e-8
        )
        assert full_item["gradient_squared_norm"] == pytest.approx(
            split_item["gradient_squared_norm"], rel=1e-6, abs=1e-8
        )
        assert full_item["gradient_rms_score"] == pytest.approx(
            full_item["gradient_norm"]
            / full_item["controlled_parameter_count"] ** 0.5
        )


def test_zero_controlled_support_fails_before_measurement_or_action_selection():
    model = _ToyPanelGradModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with pytest.raises(PanelGradError, match="zero controlled trainable FFN scalars"):
        resolve_controlled_ffn_support(model, ["small", "full"])


def test_measurement_isolates_model_runtime_rng_gradients_and_optimization_state():
    torch.manual_seed(9)
    random.seed(10)
    np.random.seed(11)
    model = _ToyPanelGradModel()
    model.train()
    model.configure_subnetwork("small")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    parameter_snapshot = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
    scheduler_snapshot = copy.deepcopy(scheduler.state_dict())
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    training_cursor = {"epoch": 3, "batch_index": 7}

    measure_panelgrad_gradients(
        model,
        _panel_batches(2),
        ["small", "full"],
        device="cpu",
    )

    assert model.training is True
    assert model.current_granularity == "small"
    assert model.current_layer_granularities == ["small"]
    assert training_cursor == {"epoch": 3, "batch_index": 7}
    assert optimizer.state_dict() == optimizer_snapshot
    assert scheduler.state_dict() == scheduler_snapshot
    assert random.getstate() == python_state
    restored_numpy_state = np.random.get_state()
    assert restored_numpy_state[0] == numpy_state[0]
    assert np.array_equal(restored_numpy_state[1], numpy_state[1])
    assert restored_numpy_state[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, parameter_snapshot[name])
        assert parameter.grad is None
    assert all(
        not layer.mlp.gradient_membership_correction_suspended
        for layer in model.matformer_layers
    )


def test_measurement_failure_restores_runtime_rng_and_leaves_no_gradients():
    model = _ToyPanelGradModel()
    model.eval()
    model.configure_subnetwork("small")
    original_forward = model.forward

    def failing_forward(*args, **kwargs):
        if model.current_granularity == "full":
            raise RuntimeError("controller failure")
        return original_forward(*args, **kwargs)

    model.forward = failing_forward
    torch_state = torch.get_rng_state().clone()

    with pytest.raises(RuntimeError, match="controller failure"):
        measure_panelgrad_gradients(
            model,
            _panel_batches(2),
            ["small", "full"],
            device="cpu",
        )

    assert model.training is False
    assert model.current_granularity == "small"
    assert model.current_layer_granularities == ["small"]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(
        not layer.mlp.gradient_membership_correction_suspended
        for layer in model.matformer_layers
    )


def test_measurement_rejects_stale_support_identity_before_refresh():
    model = _ToyPanelGradModel()
    support = resolve_controlled_ffn_support(model, ["small", "full"])
    support["controlled_support_counts"]["small"] += 1

    with pytest.raises(PanelGradError, match="support hash mismatch"):
        measure_panelgrad_gradients(
            model,
            _panel_batches(2),
            ["small", "full"],
            device="cpu",
            support_identity=support,
        )

    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize(
    "completed_actions, expected_phase",
    [(1, "terminal_partial"), (2, "terminal_complete")],
)
def test_terminal_state_never_performs_an_unused_refresh_or_draw(
    completed_actions,
    expected_phase,
):
    controller = _controller(interval=2)
    controller.install_refresh(_measurement(), boundary_step=0)
    for step in range(1, completed_actions + 1):
        controller.sample_action()
        controller.commit_pending_action(completed_step=step)
    draws_before = controller.state_dict()["sampling"]["sample_count"]

    terminal = controller.finish_training(completed_step=completed_actions)
    state = controller.state_dict()

    assert state["refresh"]["phase"] == expected_phase
    assert state["sampling"]["sample_count"] == draws_before
    assert terminal["unused_refresh_performed"] is False
    assert terminal["unused_draw_performed"] is False
    with pytest.raises(PanelGradError, match="complete refresh"):
        controller.sample_action()
