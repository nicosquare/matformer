#!/usr/bin/env python3
"""Prepare four S1/S2 runs with per-epoch uniform-to-inverse sampling."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import yaml

from src.utils.config import resolve_run_config
from src.utils.reproducibility import seed_for


BASE = Path("/nfs-stor/ivo.navarrete/results/elasticnn")
CAMPAIGN_ID = "tinystories-epoch-linear-im-v1"
GRIDS = {
    "linear": (BASE / "optimizer-ownership-v1", ("g250", "g500", "g750", "g1000")),
    "geometric": (BASE / "optimizer-ownership-matformer-widths-v1", ("g125", "g250", "g500", "g1000")),
}
INVERSE = (.12, .16, .24, .48)
EPOCHS = 4
EPOCH_STEPS = 87132


def differences(left: dict, right: dict, prefix: str = "") -> list[str]:
    changes = []
    for key in sorted(set(left) | set(right)):
        old, new = left.get(key, "<absent>"), right.get(key, "<absent>")
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(old, dict) and isinstance(new, dict):
            changes.extend(differences(old, new, path))
        elif old != new:
            changes.append(path)
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if root.exists():
        raise ValueError(f"Fresh campaign root required: {root}")
    configs = root / "campaign" / "configs"
    configs.mkdir(parents=True)

    reports = {}
    resolved_configs = {}
    for grid, (reference, widths) in GRIDS.items():
        schedule = [
            {width: .25 + (epoch / (EPOCHS - 1)) * (INVERSE[index] - .25)
             for index, width in enumerate(widths)}
            for epoch in range(EPOCHS)
        ]
        for owner in ("S1", "S2"):
            arm = f"{owner}-{grid}"
            source = reference / "campaign" / "configs" / f"{owner}.yaml"
            original = yaml.safe_load(source.read_text())
            config = yaml.safe_load(source.read_text())
            config.pop("optimizer_ownership_contract", None)
            config.pop("optimizer_ownership_contract_hash", None)
            config["run"].pop("campaign_schema_version", None)
            run_id = f"{CAMPAIGN_ID}-{arm}-s42"
            config["run"].update(
                campaign_id=CAMPAIGN_ID, arm_id=arm, run_id=run_id,
                output_dir=str(root / "runs" / run_id),
            )
            config["model"]["global_sampling_schedule"] = "epoch_categorical"
            config["model"]["global_sampling_interval_steps"] = 1
            config["model"]["global_sampling_epoch_distributions"] = schedule
            config["dataset"]["optimizer_iteration"] = dict(
                mode="repeat_epochs", epoch_order="deterministic_per_epoch"
            )
            assert tuple(config["model"]["granularities"]) == widths
            expected_changes = {
                "run.campaign_id", "run.arm_id", "run.run_id", "run.output_dir",
                "model.global_sampling_schedule", "model.global_sampling_epoch_distributions",
                "optimizer_ownership_contract", "optimizer_ownership_contract_hash",
            }
            if grid == "geometric":
                expected_changes.add("run.campaign_schema_version")
            changes = differences(original, config)
            assert set(changes) == expected_changes, (arm, changes)
            path = configs / f"{arm}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            resolved = resolve_run_config(path, create_output_dirs=False)
            iteration = resolved["dataset"]["optimizer_iteration"]
            assert iteration["complete_epochs"] == EPOCHS
            assert iteration["partial_final_epoch_samples"] == 0
            assert iteration["epoch_order"] == "deterministic_per_epoch"
            assert resolved["training"]["max_steps"] == EPOCHS * EPOCH_STEPS
            assert resolved["training"]["resolved_learning_rate"] == .008
            assert resolved["training"]["resolved_warmup_steps"] == 64
            assert resolved["training"]["optimizer_state_scope"] == (
                "shared" if owner == "S1" else "per_granularity"
            )
            assert resolved["model"]["global_sampling_epoch_distributions"] == schedule
            resolved_configs[arm] = resolved
            reports[arm] = dict(source_config=str(source), config_path=str(path),
                                changes_from_source=changes, widths=list(widths),
                                optimizer_state_scope=resolved["training"]["optimizer_state_scope"])

    traces = {}
    for grid in GRIDS:
        resolved = resolved_configs[f"S1-{grid}"]
        widths = tuple(resolved["model"]["granularities"])
        distributions = resolved["model"]["global_sampling_epoch_distributions"]
        generator = random.Random(seed_for(resolved, "granularity_selection"))
        action_indices = bytearray()
        counts = []
        for row in distributions:
            weights = [row[width] for width in widths]
            indices = [
                generator.randrange(4) if len(set(weights)) == 1
                else generator.choices(range(4), weights=weights, k=1)[0]
                for _ in range(EPOCH_STEPS)
            ]
            action_indices.extend(indices)
            counts.append({width: indices.count(index) for index, width in enumerate(widths)})
        digest = hashlib.sha256(action_indices).hexdigest()
        trace_path = root / "campaign" / f"actions_{grid}_indices_u8.bin"
        trace_path.write_bytes(action_indices)
        traces[grid] = dict(action_trace_path=str(trace_path), action_trace_sha256=digest,
                            distributions=distributions, realized_counts_by_epoch=counts,
                            realized_counts_total={width: sum(row[width] for row in counts) for width in widths})
    assert traces["linear"]["action_trace_sha256"] == traces["geometric"]["action_trace_sha256"]
    audit = dict(status="passed", campaign_id=CAMPAIGN_ID, seed=42,
                 complete_epochs=EPOCHS, optimizer_updates_per_epoch=EPOCH_STEPS,
                 global_updates=EPOCHS * EPOCH_STEPS,
                 independent_draws_with_replacement=True,
                 original_deterministic_batch_order=True,
                 first_epoch_matches_original_uniform_rng_method=True,
                 arms=reports, grids=traces)
    (root / "campaign" / "schedule_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"status": "passed", "arms": list(reports),
                      "grids": {grid: traces[grid]["realized_counts_total"] for grid in GRIDS}}, indent=2))


if __name__ == "__main__":
    main()
