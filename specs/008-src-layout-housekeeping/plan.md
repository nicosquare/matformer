# Implementation Plan: Source Layout Housekeeping

**Branch**: `008-src-layout-housekeeping` | **Date**: 2026-06-18 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/008-src-layout-housekeeping/spec.md`

## Summary

Move the importable production code into a `src/` layout while preserving the current package names, keep root-level entrypoints as thin wrappers, and split the oversized configuration, training, metrics, and figure-generation modules into smaller focused units without changing user-facing behavior.

## Technical Context

**Language/Version**: Python 3.12  
**Primary Dependencies**: PyTorch, transformers, datasets, PyYAML, pytest, matplotlib, optional `wandb`  
**Storage**: File-based experiment artifacts under `outputs/`, logs under `logs/`, checkpoints in run directories  
**Testing**: pytest with focused import/layout checks, training smoke tests, and figure-generation verification  
**Target Platform**: Linux research workstation and single-node GPU cluster  
**Project Type**: research script/model change/training pipeline  
**Experiment Scope**: repository housekeeping and module decomposition only; no experiment logic changes  
**Datasets/Data Assumptions**: existing debug and d_model=256 pilot configs; no dataset redesign  
**Configuration Inputs**: YAML configs, `--override`, and the existing training/figure-generation CLI flags  
**Experiment Outputs**: resolved config JSON, metrics CSV/JSON, run summaries, scaling results, figures, and checkpoints  
**Reproducibility Notes**: preserve config saving, resolved provenance fields, and existing seed behavior  
**Performance Goals**: no measurable regression in representative training or figure-generation smoke runs  
**Constraints**: single-process runtime behavior stays unchanged; root wrappers remain stable  
**Scale/Scope**: current repository packages plus the four named oversized modules

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Research code first**: Pass. The refactor preserves explicit experiment flow and avoids framework-style indirection.
- **Simplicity and local reasoning**: Pass. The split is organized around obvious responsibilities and short wrapper entrypoints.
- **Minimal abstraction and validation**: Pass. The plan favors direct module boundaries over new registries or factories.
- **Transparent configuration and reproducibility**: Pass. Configuration resolution and artifact provenance remain explicit and saved.
- **Useful outputs and logging**: Pass. Structured outputs remain the same and are still written to disk.
- **Shallow organization**: Pass. The move uses a shallow `src/` tree with the existing package names preserved.

## Project Structure

### Documentation (this feature)

```text
specs/008-src-layout-housekeeping/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── contracts/
    └── cli-entrypoints.md
```

### Source Code

```text
pyproject.toml
train.py
scripts/
├── make_figures.py
├── diagnose_fsdp_mlp_equivalence.py
└── queue_dmodel256_pilot.py
src/
├── evaluation/
│   ├── __init__.py
│   ├── consistency.py
│   ├── downstream.py
│   ├── speculative.py
│   ├── validation.py
│   ├── reporting.py
│   └── reporting_styles.py
├── models/
│   ├── __init__.py
│   ├── adaptive_sampler.py
│   ├── correction.py
│   ├── ffn.py
│   ├── granularity.py
│   └── wiring.py
├── training/
│   ├── __init__.py
│   ├── baselines.py
│   ├── checkpointing.py
│   ├── data.py
│   ├── distributed.py
│   ├── run.py
│   ├── steps.py
│   └── warmup.py
└── utils/
    ├── __init__.py
    ├── config.py
    ├── config_resolution.py
    ├── heartbeats.py
    ├── metrics.py
    ├── metrics_io.py
    ├── metrics_summary.py
    ├── model_size.py
    └── monitoring.py
configs/
tests/
```

**Structure Decision**: Use a `src/` layout that preserves the current package names (`models`, `training`, `evaluation`, `utils`) while moving reusable code out of the repository root. Keep `train.py` and `scripts/make_figures.py` as thin wrappers, place training orchestration in `src/training`, and place reusable reporting logic in `src/evaluation`.

## Phase 0 Research

Research decisions are resolved and reflected in the design artifacts:

1. Use a minimal `pyproject.toml` with a `src/` package layout so imports resolve from the packaged source tree instead of relying on ad hoc path mutation.
2. Preserve the existing package names under `src/` to avoid broad import churn in tests and internal modules.
3. Split `training/run.py` by responsibility into orchestration, checkpointing/continuation, step-loop helpers, and warmup helpers.
4. Split `utils/config.py` and `utils/metrics.py` by responsibility so config resolution and metrics/artifact writing stay readable.
5. Move reusable figure-generation helpers out of the standalone script and into importable reporting modules under `src/evaluation`.

See `research.md` for the decision record and alternatives considered.

## Phase 1 Design

Design artifacts are complete:

- `data-model.md` defines the source package, wrapper, and module-boundary entities used by the refactor.
- `contracts/cli-entrypoints.md` documents the stable command interfaces for the training and figure-generation entrypoints.
- `quickstart.md` provides validation commands for importability, wrapper execution, and representative smoke runs.

## Complexity Tracking

No constitution violations require justification for this plan.
