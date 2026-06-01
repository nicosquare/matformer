# Implementation Plan: Long Run Support

**Branch**: `003-run-monitoring-warmup` | **Date**: 2026-06-01 | **Spec**: [/home/nicolas.avila/dev/references/matformer/specs/003-run-monitoring-warmup/spec.md](/home/nicolas.avila/dev/references/matformer/specs/003-run-monitoring-warmup/spec.md)
**Input**: Feature specification from `/specs/003-run-monitoring-warmup/spec.md`

## Summary

Add support for long-running experiments that can resume after a Slurm time
limit, publish loss-by-granularity metrics to Weights & Biases on a per-run
basis, and run a configurable warmup phase before nested block
slicing/splitting so nested training starts from better weights.

## Technical Context

**Language/Version**: Python 3.12  
**Primary Dependencies**: PyTorch, transformers, datasets, pandas, matplotlib,
PyYAML, pytest, `wandb` for optional monitoring  
**Storage**: filesystem artifacts under the configured output root, checkpoint
files, heartbeat logs, and W&B run history/metadata  
**Testing**: pytest plus focused smoke checks for config resolution, resume
behavior, monitoring series mapping, and warmup transitions  
**Target Platform**: Linux research workstation and single-node GPU cluster
with Slurm  
**Project Type**: research script/model change/training pipeline  
**Experiment Scope**: training continuation, experiment monitoring, and a
pre-nested warmup phase; no dataset or model-family change  
**Datasets/Data Assumptions**: existing debug and d_model=256 configs remain the
validation inputs; per-run granularity labels come from the training metrics
rows, while cross-run figure generation remains separate  
**Configuration Inputs**: YAML configs plus `--override` values; new
continuation, monitoring, and warmup fields are resolved alongside the existing
`run`, `training`, `dataset`, `outputs`, and `evaluation` sections  
**Experiment Outputs**: `config.json`, `run_summary.json`, `metrics.csv`,
`scaling_results.csv`, checkpoints, `heartbeats.jsonl`, and W&B scalar history  
**Reproducibility Notes**: resolved config saved on rank 0, seed and output
root recorded, latest checkpoint pointer recorded, warmup transition recorded,
and resumed runs reuse the same `run_id`/output directory  
**Performance Goals**: preserve current throughput when monitoring is disabled;
resume should avoid redoing completed work after preemption  
**Constraints**: keep the flow direct and visible, avoid a callback framework or
deep orchestration layer, and keep monitoring/warmup behavior config-driven  
**Scale/Scope**: debug matrix runs plus d_model=256 pilot runs, with the new
ability to span multiple Slurm allocations when token budgets exceed one job
window

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Research code first**: Pass. The feature stays inside the existing training
  entry points and artifact writers.
- **Simplicity and local reasoning**: Pass. Resume, monitoring, and warmup
  remain explicit in the shared run path instead of being hidden behind a
  framework.
- **Minimal abstraction and validation**: Pass. Only the validation needed to
  keep continuation, monitoring series, and warmup configuration coherent is
  added.
- **Transparent configuration and reproducibility**: Pass. The resolved config
  and run summary will record continuation, monitoring, and warmup state.
- **Useful outputs and logging**: Pass. Metrics still land in CSV/JSON, and the
  same series become available in W&B for live inspection.
- **Shallow organization**: Pass. Changes stay in the existing `train.py`,
  `training/`, `utils/`, `evaluation/`, and `scripts/` layout.

## Project Structure

### Documentation (this feature)

```text
specs/003-run-monitoring-warmup/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── contracts/
    ├── cli.md
    ├── experiment-config.md
    ├── monitoring.md
    └── run-artifacts.md
```

### Source Code (repository root)

```text
train.py
training/
evaluation/
utils/
configs/
scripts/
tests/
outputs/
```

**Structure Decision**: Keep the implementation in the current shallow
research layout. The shared run path in `train.py` and `training/run.py`
remains the execution entry point, `utils/config.py` and `utils/metrics.py`
remain the central configuration/artifact resolvers, and the W&B-specific
behavior should stay in a small utility module or the existing shared helpers
instead of a new package hierarchy.

## Implementation Strategy

### Workstream 1: Scheduler-Resilient Continuation

- Reuse the existing resolved output directory and checkpoint artifacts as the
  resume anchor for long runs.
- Record continuation state in the run summary so a relaunched run can detect
  whether it is fresh, resumed, or finished.
- Restore model, optimizer, scheduler, and step counters from the latest saved
  checkpoint before continuing the training loop.
- Keep the same logic available from both the config-driven and legacy direct
  entry points so the launch path does not change.

### Workstream 2: W&B Monitoring Aligned With Per-Run Loss Views

- Emit W&B scalars from the same per-step/per-granularity rows used to write
  `metrics.csv`.
- Publish nested runs with one series per active granularity and standalone
  runs with only the active standalone series.
- Keep the live dashboard scoped to a single run; the cross-run size/perplexity
  plots remain the responsibility of `scripts/make_figures.py`.
- Include run identity, topology, granularity, split, and stage metadata so the
  dashboard remains comparable with the saved CSV/JSON outputs.
- Keep monitoring optional and non-blocking; if W&B is disabled or unavailable,
  the training run should still complete and write the normal filesystem
  artifacts.

### Workstream 3: Pre-Nested Warmup

- Introduce an explicit warmup phase separate from scheduler warmup so the
  model can be warmed before nested block slicing/splitting starts.
- Support warmup duration in both epochs and optimization steps so very short
  debug runs and longer Slurm runs can both be configured naturally.
- Keep warmup nested-only so standalone runs bypass it and remain comparable to
  the existing standalone baseline path.
- Persist warmup completion and the transition into the nested phase so resumed
  runs continue from the warmed state rather than from the beginning.

### Validation Strategy

- Add config-resolution coverage for continuation, monitoring, and warmup
  fields.
- Add smoke tests for resumed long runs, W&B series labeling, and the warmup
  transition before nested training.
- Keep artifact assertions focused on the existing shared outputs plus the new
  continuation and warmup metadata.

## Complexity Tracking

No constitution violations require justification for this plan.
