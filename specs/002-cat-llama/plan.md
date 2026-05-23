# Implementation Plan: Cat Llama Granularity Pipeline

**Branch**: `002-cat-llama` | **Date**: 2026-05-23 | **Spec**: [/home/nicolas.avila/dev/references/matformer/specs/002-cat-llama/spec.md](/home/nicolas.avila/dev/references/matformer/specs/002-cat-llama/spec.md)
**Input**: Feature specification from `/specs/002-cat-llama/spec.md`

## Summary

Add a config-selectable `cat_llama` model variant that uses the same
config-driven training path as `matformer_llama`, while preserving the existing
default behavior and artifact schema for direct comparison. Extend the plan with
Phase 6.1 so distributed training uses an explicit, config-driven learning-rate
and warmup policy plus selectable optimizer controls for debugging the
cat-llama vs matformer-llama path.

## Technical Context

**Language/Version**: Python 3.12  
**Primary Dependencies**: PyTorch, transformers, datasets, PyYAML, pandas, matplotlib, pytest, lm-eval  
**Storage**: filesystem artifacts under the configured output root plus model checkpoints  
**Testing**: pytest and focused config/artifact smoke checks  
**Target Platform**: Linux research workstation or single-node GPU cluster  
**Project Type**: research script/model change/training pipeline  
**Experiment Scope**: model variant toggle plus training-schedule and optimizer controls for nested MatFormer runs; no dataset or evaluation change
**Datasets/Data Assumptions**: existing debug and d_model=256 experiment configs stay unchanged
**Configuration Inputs**: YAML configs plus `--override` values; `model.variant` selects `matformer_llama` or `cat_llama`, while `run.model_family` continues to mean `nested` or `standalone` topology; `training.learning_rate_scale_rule`, `training.warmup_ratio`, `training.warmup_steps`, and `training.optimizer.{name,kwargs}` are resolved from config
**Experiment Outputs**: resolved `config.json`, `run_summary.json`, `metrics.csv`, `scaling_results.csv`, checkpoints, and heartbeat logs where enabled
**Reproducibility Notes**: saved resolved configs, explicit seeds, explicit output root, rank-0-only shared artifacts, and recorded schedule/optimizer resolution fields
**Performance Goals**: preserve baseline behavior rather than chase a new throughput target; keep distributed runs numerically traceable when `effective_world_size` changes
**Constraints**: keep the shared experiment path visible, avoid new framework layers, keep the default variant unchanged, and keep schedule/optimizer controls config-driven for debugging
**Scale/Scope**: debug matrix plus d_model=256 pilot comparison runs; optional distributed execution remains unchanged

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Research code first**: Pass. The change stays inside the existing training entry points, config resolution, and model construction path.
- **Simplicity and local reasoning**: Pass. The learning-rate, warmup, and optimizer policy are resolved directly from config rather than behind a registry or helper framework.
- **Minimal abstraction and validation**: Pass. The plan adds only the validation needed to keep optimizer and schedule resolution explicit.
- **Transparent configuration and reproducibility**: Pass. Variant, schedule, and optimizer choices are written into resolved artifacts.
- **Useful outputs and logging**: Pass. Existing CSV/JSON artifacts remain the comparison surface, with added schedule metadata.
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

**Structure Decision**: Keep the feature in the current shallow research layout.
The shared `train.py`/`training/` path remains the entry point,
`modified_llama.py` continues to own the model variants, and the new
documentation lives beside the other Spec Kit artifacts under
`specs/002-cat-llama/`.

## Phase 6.1: Training Schedule and Optimizer Controls

**Purpose**: Make budgeted runs more robust when `effective_world_size` changes and expose optimizer choice for the `cat_llama` vs `matformer_llama` debugging path without changing the shared model-variant pipeline.

**Design Decisions**

- Keep `training.token_budget` as the source of truth for total training length and keep `training.max_steps` derived from the resolved budget/world-size combination.
- Resolve warmup primarily as a ratio against `training.max_steps`, while still supporting explicit `training.warmup_steps` for exact control. If both are present, `warmup_steps` wins. Record the resolved warmup step count in both `config.json` and `run_summary.json`.
- Treat `training.learning_rate` as the author-written base learning rate and scale it from the resolved global batch size with an explicit `training.learning_rate_scale_rule`. The recommended default for distributed runs is `linear`; `none` remains available for strict baseline replication and `sqrt` remains available for conservative sweeps.
- Add `training.optimizer.name` plus `training.optimizer.kwargs` so runs can switch between `adamw` and `sgd` from config. Keep `adamw` as the default for current runs, but make `sgd` available for the FSDP and parameterization debugging path.
- Support AdamW kwargs `betas`, `eps`, and `weight_decay` at minimum. Support SGD kwargs `momentum`, `dampening`, `nesterov`, and `weight_decay` at minimum.
- Persist `base_learning_rate`, `learning_rate_scale_rule`, `learning_rate_scale_factor`, `resolved_learning_rate`, `warmup_ratio`, `warmup_steps`, `resolved_warmup_steps`, `optimizer_name`, and `optimizer_kwargs` in the resolved config and run summary artifacts so distributed runs remain comparable and auditable.
- Keep the `cat_llama` and `matformer_llama` variant work, plus the existing FSDP diagnostics, intact. This phase only changes how resolved training hyperparameters are interpreted and recorded.

## Complexity Tracking

No constitution violations require justification for this plan.
