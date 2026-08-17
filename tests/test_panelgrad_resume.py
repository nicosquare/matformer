import copy

import pytest
import torch

from src.training.panelgrad import (
    PanelGradError,
    build_panelgrad_controller,
    restore_panelgrad_controller,
    validate_panelgrad_state,
)
from src.utils.config import resolve_run_config
from src.utils.reproducibility import stable_hash


def _resume_fixture(*, scheduled=False):
    config = resolve_run_config("tests/fixtures/panelgrad_smoke.yaml")
    if scheduled:
        config["model"]["panelgrad"].pop("epsilon")
        config["model"]["panelgrad"]["epsilon_schedule"] = {
            "type": "linear",
            "start": 0.5,
            "end": 0.1,
            "duration_steps": 4,
        }
    labels = list(config["model"]["granularities"])
    support = {
        "support_schema_version": 1,
        "variant": "slicing",
        "layer_count": 1,
        "ordered_granularities": labels,
        "controlled_support_counts": {
            label: (index + 1) * 10 for index, label in enumerate(labels)
        },
        "supports": {label: [] for label in labels},
        "excluded_parameter_families": ["shared_down_bias"],
    }
    support["controlled_support_hash"] = stable_hash(support)
    config["model"]["panelgrad"]["controlled_support_counts"] = copy.deepcopy(
        support["controlled_support_counts"]
    )
    config["model"]["panelgrad"]["controlled_support_hash"] = support[
        "controlled_support_hash"
    ]
    for field in (
        "data_roles_manifest_hash",
        "optimizer_training_manifest_hash",
        "controller_manifest_hash",
        "validation_manifest_hash",
        "final_holdout_manifest_hash",
    ):
        config[field] = f"{field}-value"
    controller = build_panelgrad_controller(config, support)
    measurement = {
        "measurements": [
            {
                "granularity": label,
                "controlled_parameter_count": support[
                    "controlled_support_counts"
                ][label],
                "aggregate_loss": 1.0,
                "gradient_squared_norm": float(index + 1),
                "gradient_norm": float(index + 1) ** 0.5,
                "gradient_rms_score": float(index + 1),
            }
            for index, label in enumerate(labels)
        ]
    }
    return config, support, controller, measurement


def _committed_actions(controller, start_step, count, measurement):
    actions = []
    for step in range(start_step, start_step + count):
        if controller.phase == "refresh_pending":
            controller.install_refresh(measurement, boundary_step=step - 1)
        action = controller.sample_action()
        actions.append(action["global_granularity"])
        controller.commit_pending_action(completed_step=step)
    return actions


@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("resume_after", [1, 2])
def test_panelgrad_resume_inside_interval_and_at_refresh_boundary_is_exact(
    resume_after,
    scheduled,
):
    config, support, uninterrupted, measurement = _resume_fixture(
        scheduled=scheduled
    )
    resumed_source = build_panelgrad_controller(config, support)
    uninterrupted.install_refresh(measurement, boundary_step=0)
    resumed_source.install_refresh(measurement, boundary_step=0)

    assert _committed_actions(uninterrupted, 1, resume_after, measurement) == (
        _committed_actions(resumed_source, 1, resume_after, measurement)
    )
    restored = restore_panelgrad_controller(
        config, support, resumed_source.state_dict()
    )
    expected = _committed_actions(
        uninterrupted, resume_after + 1, 5 - resume_after, measurement
    )
    actual = _committed_actions(
        restored, resume_after + 1, 5 - resume_after, measurement
    )

    assert actual == expected
    restored_sampling = restored.state_dict()["sampling"]
    uninterrupted_sampling = uninterrupted.state_dict()["sampling"]
    assert torch.equal(
        restored_sampling.pop("generator_state"),
        uninterrupted_sampling.pop("generator_state"),
    )
    assert restored_sampling == uninterrupted_sampling
    assert restored.state_dict()["refresh"] == uninterrupted.state_dict()[
        "refresh"
    ]


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda state: state.update(method_version=99), "identity mismatch"),
        (lambda state: state["policy"].update(epsilon=0.9), "policy mismatch"),
        (
            lambda state: state["support"].update(
                controlled_support_hash="wrong"
            ),
            "support_hash mismatch",
        ),
        (
            lambda state: state["manifest_hashes"].update(
                controller_manifest_hash="wrong"
            ),
            "manifest mismatch",
        ),
        (
            lambda state: state["sampling"].update(generator_state=None),
            "generator is invalid",
        ),
    ],
)
def test_panelgrad_resume_rejects_malformed_or_incompatible_state(mutation, match):
    config, support, controller, measurement = _resume_fixture()
    controller.install_refresh(measurement, boundary_step=0)
    state = controller.state_dict()
    mutation(state)

    with pytest.raises(PanelGradError, match=match):
        validate_panelgrad_state(state, config=config, support_identity=support)


def test_panelgrad_terminal_state_round_trips_without_an_extra_action():
    config, support, controller, measurement = _resume_fixture()
    controller.install_refresh(measurement, boundary_step=0)
    controller.sample_action()
    controller.commit_pending_action(completed_step=1)
    controller.finish_training(completed_step=1)

    restored = restore_panelgrad_controller(config, support, controller.state_dict())

    assert restored.phase == "terminal_partial"
    assert restored.state_dict()["sampling"]["sample_count"] == 1
    with pytest.raises(PanelGradError, match="complete refresh"):
        restored.sample_action()


def test_fixed_epsilon_version_one_state_migrates_exactly():
    config, support, source, measurement = _resume_fixture()
    reference = build_panelgrad_controller(config, support)
    for controller in (source, reference):
        controller.install_refresh(measurement, boundary_step=0)
        _committed_actions(controller, 1, 1, measurement)

    version_one = source.state_dict()
    version_one["schema_version"] = 1
    version_one["policy"].pop("epsilon_schedule")
    version_one["refresh"].pop("active_epsilon")
    version_one["refresh"].pop("epsilon_schedule_step")

    restored = restore_panelgrad_controller(config, support, version_one)

    assert restored.state_dict()["schema_version"] == 2
    assert restored.state_dict()["policy"]["epsilon_schedule"]["type"] == "fixed"
    assert restored.state_dict()["refresh"]["active_epsilon"] == pytest.approx(0.1)
    assert _committed_actions(restored, 2, 4, measurement) == _committed_actions(
        reference, 2, 4, measurement
    )


def test_version_one_state_cannot_initialize_a_scheduled_policy():
    fixed_config, support, controller, measurement = _resume_fixture()
    controller.install_refresh(measurement, boundary_step=0)
    version_one = controller.state_dict()
    version_one["schema_version"] = 1
    version_one["policy"].pop("epsilon_schedule")
    version_one["refresh"].pop("active_epsilon")
    version_one["refresh"].pop("epsilon_schedule_step")
    scheduled_config, _, _, _ = _resume_fixture(scheduled=True)

    with pytest.raises(PanelGradError, match="cannot resume.*scheduled"):
        validate_panelgrad_state(
            version_one,
            config=scheduled_config,
            support_identity=support,
        )


def test_scheduled_checkpoint_probability_is_validated_with_snapshot_epsilon():
    config, support, controller, measurement = _resume_fixture(scheduled=True)
    controller.install_refresh(measurement, boundary_step=0)
    state = controller.state_dict()
    state["refresh"]["active_epsilon"] = 0.4

    with pytest.raises(PanelGradError, match="active epsilon is invalid"):
        validate_panelgrad_state(state, config=config, support_identity=support)
