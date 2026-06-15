# Implementation Plan: Adaptive Per-Block Sampling

**Branch**: `007-adaptive-per-block-sampling` | **Date**: 2026-06-15 | **Spec**: `specs/007-adaptive-per-block-sampling/spec.md`
**Input**: Feature specification from `specs/007-adaptive-per-block-sampling/spec.md`

## Summary

Add a config-controlled `adaptive_per_block` mode for nested-random MatFormer
runs while preserving the existing random `per_block` baseline unchanged.
The new mode uses a lightweight contextual bandit to pick one granularity per
transformer block, records the selected strategy and sampler state, and writes
the resolved mode and provenance into config, summary, and metric artifacts.

## Technical Context

**Language/Version**: Python 3.12  
**Primary Dependencies**: PyTorch, transformers, datasets, PyYAML, pytest,
`wandb` for optional monitoring  
**Storage**: filesystem artifacts under `outputs/`, checkpoints in the run
directory, metrics CSV/JSON, summary JSON, extraction metadata, and logs  
**Testing**: pytest plus focused config-resolution, artifact, sampler-state,
and training-smoke tests for `global`, `per_block`, and `adaptive_per_block`
paths  
**Target Platform**: Linux research workstation and single-node GPU cluster;
single-process execution only  
**Project Type**: research script/model change/training pipeline  
**Experiment Scope**: additive sampling-mode normalization, adaptive
per-block selection, reward bookkeeping, and provenance tracking  
**Datasets/Data Assumptions**: existing debug and `d_model=256` pilot configs;
no dataset redesign or distributed validation in this feature  
**Configuration Inputs**: YAML configs plus `--override`; top-level
`run.sampling_mode`, model-level `model.granularity_sampling_mode`, and
adaptive controls such as `model.adaptive_sampler_strategy`,
`model.adaptive_sampler_exploration_scale`, and
`model.adaptive_sampler_decay_rate`,
`model.adaptive_sampler_reward_penalty_weight`  
**Experiment Outputs**: `config.json`, `run_summary.json`, `metrics.csv`,
`scaling_results.csv`, checkpoints, extraction metadata, and console/log files
with adaptive strategy, reward, correction-penalty, and sampler-state fields  
**Reproducibility Notes**: save the resolved config per run, record the seed
when set, and persist the resolved mode, runtime granularity pattern,
correction context, strategy, and sampler state in run metadata  
**Performance Goals**: preserve current global and random per-block behavior;
adaptive mode adds local sampling and update bookkeeping only, with no
distributed sync  
**Constraints**: single-process only; invalid mode or correction pairings must
fail before training begins  
**Scale/Scope**: debug matrix and `d_model=256` pilot runs, with unit and
smoke coverage for each supported mode

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Research code first**: Pass. The plan keeps the work inside the existing
  config, model, training, and metrics modules instead of adding a framework
  layer.
- **Simplicity and local reasoning**: Pass. The adaptive sampler is scoped to a
  small state object and the existing per-block sampling flow.
- **Minimal abstraction and validation**: Pass. The feature uses explicit mode
  fields and direct validation instead of a generic registry.
- **Transparent configuration and reproducibility**: Pass. The resolved mode,
  strategy, pattern provenance, and sampler state are planned for config and
  summary artifacts.
- **Useful outputs and logging**: Pass. The run already writes structured JSON
  and CSV artifacts; this feature extends them with adaptive provenance.
- **Shallow organization**: Pass. The repository already uses the shallow
  `models/`, `training/`, `evaluation/`, `utils/`, `configs/`, `scripts/`, and
  `tests/` layout that fits the change.

## Project Structure

### Documentation (this feature)

```text
specs/007-adaptive-per-block-sampling/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── model-sampling.md
│   └── run-artifacts.md
└── tasks.md
```

### Source Code (repository root)

```text
train.py
models/
├── correction.py
├── ffn.py
├── granularity.py
└── wiring.py
training/
├── baselines.py
├── data.py
├── distributed.py
└── run.py
evaluation/
├── downstream.py
└── validation.py
utils/
├── config.py
├── heartbeats.py
├── metrics.py
├── model_size.py
└── monitoring.py
configs/
├── consistency.yaml
├── debug_matrix.yaml
├── dmodel256_pilot_comparison.yaml
└── speculative.yaml
scripts/
tests/
```

**Structure Decision**: Keep the current shallow experiment layout. Mode
resolution stays in `utils/config.py`, runtime provenance and summaries stay in
`utils/metrics.py` and `training/run.py`, model-level sampling and correction
behavior stay in `models/`, and validation stays in `tests/`.

## Phase 0 Research

Research is complete and the open questions from the feature spec are resolved:

1. `run.sampling_mode` remains the top-level run selector with values
   `nested-random`, `nested-all`, and `standalone`.
2. `model.granularity_sampling_mode` becomes the explicit model-level choice
   with values `global`, `per_block`, and `adaptive_per_block`.
3. `adaptive_per_block` uses a contextual bandit with per-block, per-granularity
   statistics plus global phase, step, epoch, exploration, and decay state.
4. The reward combines loss improvement and a normalized correction penalty.
5. Validation covers config resolution, correction behavior, artifact
   serialization, resume behavior, and smoke runs for the explicit nested-random
   modes.

See `research.md` for the detailed decisions and alternatives considered.

## Phase 1 Design

Design artifacts are complete:

- `data-model.md` defines the sampling mode, strategy, adaptive state, reward,
  pattern, and provenance entities.
- `contracts/model-sampling.md` documents the public config contract for the
  explicit mode surface.
- `contracts/run-artifacts.md` documents the artifact contract required to
  distinguish and resume adaptive runs.
- `quickstart.md` provides validation commands for the canonical run paths.

## Complexity Tracking

No constitution violations require justification for this plan.
