# Implementation Plan: Cat Llama Granularity Pipeline

**Branch**: `002-cat-llama` | **Date**: 2026-05-21 | **Spec**: [/home/nicolas.avila/dev/references/matformer/specs/002-cat-llama/spec.md](/home/nicolas.avila/dev/references/matformer/specs/002-cat-llama/spec.md)
**Input**: Feature specification from `/specs/002-cat-llama/spec.md`

## Summary

Add a config-selectable `cat_llama` model variant that uses the same
config-driven training path as `matformer_llama`, while preserving the existing
default behavior and artifact schema for direct comparison.

## Technical Context

**Language/Version**: Python 3.12  
**Primary Dependencies**: PyTorch, transformers, datasets, PyYAML, pandas, matplotlib, pytest, lm-eval  
**Storage**: filesystem artifacts under the configured output root plus model checkpoints  
**Testing**: pytest and focused config/artifact smoke checks  
**Target Platform**: Linux research workstation or single-node GPU cluster  
**Project Type**: research script/model change/training pipeline  
**Experiment Scope**: model variant toggle for nested MatFormer runs; no dataset or evaluation change  
**Datasets/Data Assumptions**: existing debug and d_model=256 experiment configs stay unchanged  
**Configuration Inputs**: YAML configs plus `--override` values; planned `model.variant` selects `matformer_llama` or `cat_llama`, while `run.model_family` continues to mean `nested` or `standalone` topology  
**Experiment Outputs**: resolved `config.json`, `run_summary.json`, `metrics.csv`, `scaling_results.csv`, checkpoints, and heartbeat logs where enabled  
**Reproducibility Notes**: saved resolved configs, explicit seeds, explicit output root, and rank-0-only shared artifacts  
**Performance Goals**: N/A; preserve baseline behavior rather than chase a new throughput target  
**Constraints**: keep the shared experiment path visible, avoid new framework layers, and keep the default variant unchanged  
**Scale/Scope**: debug matrix plus d_model=256 pilot comparison runs; optional distributed execution remains unchanged

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Research code first**: Pass. The change stays inside the existing training entry points and model construction path.
- **Simplicity and local reasoning**: Pass. A single model-variant selection point is easier to trace than a second pipeline.
- **Minimal abstraction and validation**: Pass. The plan does not introduce a registry or factory layer.
- **Transparent configuration and reproducibility**: Pass. The variant is config-driven and will be recorded in resolved artifacts.
- **Useful outputs and logging**: Pass. Existing CSV/JSON artifacts remain the comparison surface.
- **Shallow organization**: Pass. No new deep package structure is required.

## Project Structure

### Documentation (this feature)

```text
specs/002-cat-llama/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── contracts/
    ├── cli.md
    ├── experiment-config.md
    └── run-artifacts.md
```

### Source Code (repository root)

```text
train.py
modified_llama.py
training/
evaluation/
utils/
configs/
scripts/
tests/
outputs/
```

**Structure Decision**: Keep the feature in the current shallow research layout. The
shared `train.py`/`training/` path remains the entry point, `modified_llama.py`
continues to own the model variants, and the new documentation lives beside the
other Spec Kit artifacts under `specs/002-cat-llama/`.
