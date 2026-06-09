# Implementation Plan: Granularity Sampling Modes

**Branch**: `005-granularity-sampling-modes` | **Date**: 2026-06-09 | **Spec**: [/home/nicolas.avila/dev/references/matformer/specs/005-granularity-sampling-modes/spec.md](/home/nicolas.avila/dev/references/matformer/specs/005-granularity-sampling-modes/spec.md)
**Input**: Feature specification from `/specs/005-granularity-sampling-modes/spec.md`

## Summary

Refactor the MatFormer model code into a shallow `models/` package with clear
boundaries for granularity metadata, FFN implementations, correction logic, and
model assembly. Add an explicit model-level granularity sampling mode with
named `global` and `per_layer` behavior, keep the current whole-model path as
a first-class configuration, and derive local GMC/LMC only when per-layer
sampling is active. Treat the legacy `training.granularity_sampling` input as a
compatibility alias that resolves into the new canonical mode rather than as a
separate experiment surface.

## Technical Context

**Language/Version**: Python 3.12  
**Primary Dependencies**: PyTorch, transformers, datasets, PyYAML, pytest,
`wandb` for optional monitoring  
**Storage**: filesystem artifacts under `outputs/`, checkpoints under the run
directory, metrics CSV/JSON, summary JSON, extraction metadata, and logs  
**Testing**: pytest plus focused smoke tests for config resolution, legacy
alias handling, model sampling semantics, correction behavior, and debug/pilot
runs  
**Target Platform**: Linux research workstation and single-node GPU cluster;
single-process execution only for this feature  
**Project Type**: research script/model change/training pipeline  
**Experiment Scope**: MatFormer model refactor, explicit sampling-mode API,
FFN correction semantics, and run-metadata provenance  
**Datasets/Data Assumptions**: existing debug and d_model=256 pilot configs,
no dataset redesign, and no distributed validation in this feature  
**Configuration Inputs**: YAML configs plus `--override` values; add a
model-level `granularity_sampling_mode` with `global` and `per_layer` options,
and resolve legacy `training.granularity_sampling` inputs through a
compatibility alias into the canonical mode  
**Experiment Outputs**: `config.json`, `run_summary.json`, `metrics.csv`,
`scaling_results.csv`, checkpoints, extraction metadata, and console/log files  
**Reproducibility Notes**: resolved config saved per run, seed recorded when
set, requested legacy alias and resolved sampling mode saved in run metadata,
sampling mode and granularity-pattern summary saved in run metadata, and
correction mode recorded explicitly  
**Performance Goals**: keep the global path behaviorally equivalent to the
current implementation; per-layer mode should add only local sampling and
correction bookkeeping, with no distributed synchronization  
**Constraints**: keep the implementation explicit and shallow; distributed
training is out of scope; local GMC/LMC must only activate in per-layer mode  
**Scale/Scope**: debug matrix and d_model=256 pilot runs, one synthetic unit
test for each sampling mode, and smoke validation for the global and per-layer
paths

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Research code first**: Pass. The plan keeps the flow visible and only adds
  a shallow model package where the current monolith is hard to follow.
- **Simplicity and local reasoning**: Pass. Granularity metadata, FFN logic,
  correction behavior, and model assembly are separated into obvious modules.
- **Minimal abstraction and validation**: Pass. The feature adds a dedicated
  sampling-mode surface, but avoids a general framework or dispatch layer.
- **Transparent configuration and reproducibility**: Pass. Sampling mode,
  correction mode, and granularity-pattern provenance are saved with each run.
- **Useful outputs and logging**: Pass. Existing CSV/JSON/checkpoint outputs
  remain, and the new mode needs no special logging infrastructure.
- **Shallow organization**: Pass. The new code lives in a shallow `models/`
  package; the rest of the repository keeps its current layout.

## Project Structure

### Documentation (this feature)

```text
specs/005-granularity-sampling-modes/
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
models/
├── granularity.py
├── correction.py
├── ffn.py
└── wiring.py
modified_llama.py
train.py
training/
evaluation/
utils/
configs/
scripts/
tests/
```

**Structure Decision**: Introduce a shallow `models/` package for common
sampling metadata, FFN implementations, correction helpers, and model assembly;
keep `modified_llama.py` as a compatibility façade while call sites migrate;
leave `train.py`, `training/`, `evaluation/`, `utils/`, `configs/`, `scripts/`,
and `tests/` in place so the experiment flow remains traceable.

## Phase 0 Research

### Questions to Resolve

1. Where should the explicit sampling-mode API live so it can coexist with the
   legacy `training.granularity_sampling` alias without ambiguity?
2. How should the model code be split so granularity metadata, FFN behavior,
   correction logic, and model assembly stay easy to inspect and extend?
3. What should be recorded in run metadata so the selected sampling mode and
   resulting granularity pattern can be reconstructed later?
4. Which validation cases are sufficient to prove the global path is unchanged
   and the per-layer path activates local GMC/LMC only when requested?

### Research Decisions

See `/specs/005-granularity-sampling-modes/research.md` for the resolved
choices and rejected alternatives.

## Phase 1 Design

### Data Model

See `/specs/005-granularity-sampling-modes/data-model.md` for the entities,
relationships, and validation rules for sampling mode, granularity patterns,
correction context, and FFN module profiles.

### Contracts

See `/specs/005-granularity-sampling-modes/contracts/model-sampling.md` and
`/specs/005-granularity-sampling-modes/contracts/run-artifacts.md` for the
config and artifact contracts that downstream code and tests will rely on.

### Quickstart

See `/specs/005-granularity-sampling-modes/quickstart.md` for the validation
steps and smoke commands for the global and per-layer sampling modes.

## Complexity Tracking

No constitution violations require justification for this plan.
