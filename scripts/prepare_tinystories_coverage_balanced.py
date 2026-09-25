#!/usr/bin/env python3
"""Prepare one four-pass, complete-per-width-coverage S1 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from src.training.checkpointing import _balanced_cycle_permutation
from src.utils.config import resolve_run_config


BASE = Path("/nfs-stor/ivo.navarrete/results/elasticnn")
SOURCES = {
    "linear": BASE / "optimizer-ownership-v1/campaign/configs/S1.yaml",
    "geometric": BASE / "optimizer-ownership-matformer-widths-v1/campaign/configs/S1.yaml",
}
CAMPAIGNS = {
    "linear": "tinystories-coverage-balanced-v1",
    "geometric": "tinystories-coverage-balanced-geometric-v1",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--grid", choices=tuple(SOURCES), default="linear")
    args = parser.parse_args()
    source = SOURCES[args.grid]
    campaign_id = CAMPAIGNS[args.grid]
    run_id = f"{campaign_id}-S1-s42"
    root = args.root.resolve()
    if root.exists():
        raise ValueError(f"Fresh output root required: {root}")
    campaign = root / "campaign"
    campaign.mkdir(parents=True)
    config = yaml.safe_load(source.read_text())
    config.pop("optimizer_ownership_contract")
    config.pop("optimizer_ownership_contract_hash")
    config["run"].pop("campaign_schema_version", None)
    config["run"].update(campaign_id=campaign_id, run_id=run_id,
                         arm_id="S1-coverage", output_dir=str(root / "runs" / run_id))
    config["model"]["global_sampling_schedule"] = "balanced_cycle"
    config["model"]["global_sampling_interval_steps"] = 1
    widths = tuple(config["model"]["granularities"])
    assert len(widths) == 4 and widths == (
        ("g250", "g500", "g750", "g1000") if args.grid == "linear"
        else ("g125", "g250", "g500", "g1000")
    )
    config["dataset"]["optimizer_iteration"] = dict(mode="repeat_epochs", epoch_order="deterministic_per_epoch")
    temporary = campaign / "generation_config.yaml"
    temporary.write_text(yaml.safe_dump(config, sort_keys=False))
    resolved = resolve_run_config(temporary, create_output_dirs=False)
    count = int(resolved["dataset"]["optimizer_iteration"]["aligned_epoch_samples"]) // int(
        resolved["training"]["batch_size_per_process"]
    )
    assert count == 87132 and count % 4 == 0
    actions = np.empty((4, count), dtype=np.uint8)
    previous_last = None
    for cycle_index in range(count):
        permutation = _balanced_cycle_permutation(resolved, cycle_index=cycle_index,
                                                   previous_last=previous_last)
        for position, width in enumerate(permutation):
            step = cycle_index * 4 + position
            actions[step // count, step % count] = widths.index(width)
        previous_last = permutation[-1]
    order = np.empty((4, count), dtype="<u4")
    order[0] = np.arange(count, dtype="<u4")
    first_width = actions[0]
    for epoch in range(1, 4):
        generator = np.random.Generator(np.random.PCG64(
            np.random.SeedSequence([int(config["run"]["seed"]), epoch, 9])))
        for width_index in range(4):
            groups = np.flatnonzero((first_width + epoch) % 4 == width_index).astype("<u4")
            slots = np.flatnonzero(actions[epoch] == width_index)
            assert len(groups) == len(slots) == count // 4
            generator.shuffle(groups)
            order[epoch, slots] = groups
    assert np.array_equal(order[0], np.arange(count))
    assert all(np.array_equal(np.sort(row), np.arange(count)) for row in order)
    coverage = np.zeros((4, count), dtype=np.uint8)
    for epoch in range(4):
        coverage[actions[epoch], order[epoch]] += 1
    assert np.all(coverage == 1), "Every width must see every batch group exactly once"
    manifest = campaign / "batch_order_u32le.bin"
    order.tofile(manifest)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    config["dataset"]["optimizer_iteration"] = dict(
        mode="repeat_epochs", epoch_order="coverage_balanced_batches",
        batch_order_path=str(manifest), batch_order_sha256=digest,
    )
    final = campaign / "S1-coverage.yaml"
    final.write_text(yaml.safe_dump(config, sort_keys=False))
    resolved_final = resolve_run_config(final, create_output_dirs=False)
    assert resolved_final["dataset"]["optimizer_iteration"]["batch_order_sha256"] == digest
    report = dict(status="passed", source_config=str(source), grid=args.grid,
                  run_id=run_id, widths=list(widths),
                  width_schedule="balanced_cycle", width_probability=.25, epochs=4,
                  batches_per_epoch=count, global_updates=4*count,
                  selected_updates_per_width={w:int((actions==i).sum()) for i,w in enumerate(widths)},
                  unique_batch_groups_per_width={w:int((coverage[i]>0).sum()) for i,w in enumerate(widths)},
                  batch_order_path=str(manifest), batch_order_sha256=digest,
                  first_epoch_matches_stored_permutation=True)
    (campaign / "coverage_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    temporary.unlink()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
