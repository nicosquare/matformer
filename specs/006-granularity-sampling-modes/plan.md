# Implementation Plan: Granularity Operation Modes

**Branch**: `006-granularity-sampling-modes` | **Date**: 2026-06-11 | **Spec**: `specs/006-granularity-sampling-modes/spec.md`
**Input**: Feature specification from `specs/006-granularity-sampling-modes/spec.md`

## Summary

Normalize MatFormer training into three explicit run modes: `nested-random`,
`nested-all`, and `standalone`. Preserve the current whole-model path as the
named `nested-random + global` behavior, add `nested-random + per_layer` as an
explicit submode, and persist the resolved mode, correction mode, and runtime
granularity-pattern provenance in saved artifacts.

## Technical Context

**Language/Version**: Python 3.12  
**Primary Dependencies**: PyTorch, transformers, datasets, PyYAML, pytest,
`wandb` for optional monitoring  
**Storage**: filesystem artifacts under `outputs/`, checkpoints in the run
directory, metrics CSV/JSON, summary JSON, extraction metadata, and logs  
**Testing**: pytest plus focused config-resolution, artifact, model-wiring, and
training-smoke tests for global, per-layer, nested-all, and standalone paths  
**Target Platform**: Linux research workstation and single-node GPU cluster;
single-process execution only  
**Project Type**: research script/model change/training pipeline  
**Experiment Scope**: explicit run-mode normalization, elastic sampling
semantics, correction derivation, and provenance tracking  
**Datasets/Data Assumptions**: existing debug and d_model=256 pilot configs; no
dataset redesign or distributed validation in this feature  
**Configuration Inputs**: YAML configs plus `--override`; top-level
`run.sampling_mode` and model-level `model.granularity_sampling_mode`  
**Experiment Outputs**: `config.json`, `run_summary.json`, `metrics.csv`,
`scaling_results.csv`, checkpoints, extraction metadata, and console/log files  
**Reproducibility Notes**: save the resolved config per run, record the seed
when set, and persist the resolved mode, runtime granularity pattern, and
correction context in run metadata  
**Performance Goals**: preserve current global-path behavior; per-layer mode
adds only local sampling and correction bookkeeping, with no distributed sync  
**Constraints**: single-process only; invalid mode or correction pairings must
fail before training begins  
**Scale/Scope**: debug matrix and d_model=256 pilot runs, with unit and smoke
coverage for each supported mode

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Research code first**: Pass. The plan keeps the model, training, evaluation,
  and utilities visible in the current codebase instead of introducing a
  framework layer.
- **Simplicity and local reasoning**: Pass. The feature is split across the
  existing shallow modules that already handle granularity metadata, correction
  logic, config resolution, and run artifact writing.
- **Minimal abstraction and validation**: Pass. The plan uses explicit run-mode
  and sampling-mode fields instead of a generic dispatch framework.
- **Transparent configuration and reproducibility**: Pass. The resolved run
  mode, model sampling mode, and pattern provenance are all saved in config
  and summary artifacts.
- **Useful outputs and logging**: Pass. The existing CSV, JSON, checkpoint, and
  extraction outputs remain the audit trail for a run.
- **Shallow organization**: Pass. The repository already uses a shallow
  `models/`, `training/`, `evaluation/`, and `utils/` layout, which matches the
  constitution guidance.

## Project Structure

### Documentation (this feature)

```text
specs/006-granularity-sampling-modes/
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
└── metrics.py
configs/
├── consistency.yaml
├── debug_matrix.yaml
├── dmodel256_pilot_comparison.yaml
└── speculative.yaml
scripts/
tests/
```

**Structure Decision**: Keep the current shallow experiment layout. The mode
normalization lives in `utils/config.py`, runtime provenance and summaries live
in `utils/metrics.py` and `training/run.py`, model-level sampling and
correction behavior live in `models/`, and validation stays in `tests/`.

## Phase 0 Research

Research is complete and the open questions from the feature spec are resolved:

1. The top-level run selector remains `run.sampling_mode` with values
   `nested-random`, `nested-all`, and `standalone`.
2. `model.granularity_sampling_mode` remains the model-level sampling submode
   and carries the explicit `global` or `per_layer` choice.
3. Validation should cover config resolution, correction activation, artifact
   serialization, and smoke runs for `nested-random + global`,
   `nested-random + per_layer`, `nested-all`, and `standalone`.

See `research.md` for the detailed decisions and rejected alternatives.

## Phase 1 Design

Design artifacts are complete:

- `data-model.md` defines the run-mode, sampling-submode, granularity-pattern,
  correction-context, and provenance entities.
- `contracts/model-sampling.md` documents the config and runtime contract for
  the explicit mode surface.
- `contracts/run-artifacts.md` documents the provenance fields required in
  saved config, summary, and metric outputs.
- `quickstart.md` provides the validation commands for the canonical run paths.

## Complexity Tracking

No constitution violations require justification for this plan.
