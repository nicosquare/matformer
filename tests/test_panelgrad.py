from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig

from src.models.ffn import ModifiedLlamaMLP
from src.training.panelgrad import (
    PanelGradController,
    PanelGradError,
    build_probability_snapshot,
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


def _controller(seed=13, interval=2, epsilon=0.1):
    return PanelGradController(
        ordered_granularities=["small", "full"],
        refresh_interval_steps=interval,
        eta=1e-12,
        temperature=1.0,
        epsilon=epsilon,
        sampling_seed=seed,
        support_identity=_support_identity(),
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
