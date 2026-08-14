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


def _resume_fixture():
    config = resolve_run_config("tests/fixtures/panelgrad_smoke.yaml")
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


@pytest.mark.parametrize("resume_after", [1, 2])
def test_panelgrad_resume_inside_interval_and_at_refresh_boundary_is_exact(
    resume_after,
):
    config, support, uninterrupted, measurement = _resume_fixture()
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
