"""C4 selected-block semantics on the two production FFN layouts."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from src.training import steps
from src.training.modeling import build_model
from src.utils.config import ConfigError, resolve_run_config


ROOT = Path(__file__).resolve().parents[1]


def config_for(grid):
    return resolve_run_config(
        ROOT / f"configs/controlled_exps/tinystories_instruct_c4_selected_block_{grid}.yaml",
        create_output_dirs=False,
    )


@pytest.mark.parametrize("grid", ["linear", "geometric"])
def test_c4_selected_owner_joint_clip_and_resume(grid):
    config = config_for(grid)
    model = build_model(config)
    optimizer, clock = steps.build_optimizer_and_scheduler(model, config["training"])
    width = config["model"]["granularities"][-1]
    assert optimizer.active_owner_ids(width) == ("O-D", "O-common")

    original = {owner.owner_id: [p.detach().clone() for p in owner.parameters]
                for owner in optimizer.owners}
    tokens = torch.arange(1, 17).reshape(1, 16)
    model.configure_subnetwork(width)
    model(input_ids=tokens, labels=tokens).loss.backward()
    assert any(p.grad is not None for owner in optimizer.owners[:3] for p in owner.parameters)
    selected = optimizer.active_owner_ids(width)
    for owner in optimizer.owners:
        if owner.owner_id not in selected:
            for parameter in owner.parameters:
                parameter.grad = None
    observation = steps.clip_optimizer_gradients(
        model, config["training"], width, owners=optimizer.owners,
        selected_owner_ids=selected,
    )
    assert observation["mode"] == "global"
    assert observation["combined_post_norm"] <= 1.00001
    assert observation["groups"]["O-A"]["active"] is False
    assert observation["groups"]["O-D"]["coefficient"] == observation["groups"]["O-common"]["coefficient"]

    for owner_id in selected:
        optimizer.optimizer_for(owner_id).step()
    clock.step()
    clock.synchronize(optimizer)
    optimizer.record_successful_update(width, returned_owners=selected)
    optimizer.validate_accounting(step=1, width_counts={**dict.fromkeys(config["model"]["granularities"], 0), width: 1})
    for owner in optimizer.owners[:3]:
        assert all(torch.equal(p, before) for p, before in zip(owner.parameters, original[owner.owner_id]))
        assert not optimizer.optimizer_for(owner.owner_id).state
    assert optimizer.optimizer_for("O-D").state
    assert optimizer.successful_update_counts == {"O-A": 0, "O-B": 0, "O-C": 0, "O-D": 1, "O-common": 1}

    saved = optimizer.state_dict()
    clone_model = build_model(config)
    clone, _ = steps.build_optimizer_and_scheduler(clone_model, config["training"])
    clone.load_state_dict(saved)
    assert clone.successful_update_counts == optimizer.successful_update_counts
    damaged = copy.deepcopy(saved)
    damaged["successful_update_counts"]["O-A"] = 1
    with pytest.raises(ConfigError):
        clone.load_state_dict(damaged)

    from src.utils.metrics import build_optimizer_state_summary_fields
    counts = optimizer.width_selection_counts
    quarters = {f"O-{block}": sum(counts[w] for w in config["model"]["granularities"][index:])
                for index, block in enumerate("ABCD")}
    summary = build_optimizer_state_summary_fields(config, run_state={
        "last_completed_step": 1, "global_scheduler_position": 1,
        "optimizer_width_selection_counts": counts,
        "optimizer_update_counts": optimizer.successful_update_counts,
        "optimizer_quarter_activation_counts": quarters,
    })
    assert summary["optimizer_accounting_reconciled"] is True


def test_c4_config_rejects_c3_clipping_or_policy(tmp_path):
    import yaml
    raw = yaml.safe_load((ROOT / "configs/controlled_exps/tinystories_instruct_c4_selected_block_linear.yaml").read_text())
    raw["training"]["gradient_clipping"] = {"mode": "per_owner", "norm_type": 2,
        "owner_max_norms": dict.fromkeys(("O-A", "O-B", "O-C", "O-D", "O-common"), 1.0)}
    path = tmp_path / "wrong.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError):
        resolve_run_config(path, create_output_dirs=False)
