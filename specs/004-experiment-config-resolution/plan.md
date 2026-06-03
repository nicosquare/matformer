# Implementation Plan: Experiment Config Resolution

**Branch**: `004-experiment-config-resolution` | **Date**: 2026-06-03 | **Spec**: [/home/nicolas.avila/dev/references/matformer/specs/004-experiment-config-resolution/spec.md](/home/nicolas.avila/dev/references/matformer/specs/004-experiment-config-resolution/spec.md)
**Input**: Feature specification from `/specs/004-experiment-config-resolution/spec.md`

## Summary

Add concat-only learning-rate membership correction as an explicit correction
mode, resolve standalone family runs into the shared largest-size experiment
folder, and introduce section-scoped config presets starting with optimizer
defaults. Keep the changes config-driven, local to the existing config resolver
and training loop, and record the resolved correction mode, folder rule, and
preset provenance in run artifacts.

## Technical Context

**Language/Version**: Python 3.12  
**Primary Dependencies**: PyTorch, transformers, datasets, pandas, matplotlib,
PyYAML, pytest, `wandb` for optional monitoring  
**Storage**: filesystem artifacts under the configured output root, checkpoint
files, heartbeat logs, and saved JSON/CSV summaries  
**Testing**: pytest plus focused smoke tests for config resolution, synthetic
concat update behavior, family-folder resolution, and figure generation  
**Target Platform**: Linux research workstation and single-node GPU cluster
with Slurm  
**Project Type**: research script/model change/training pipeline  
**Experiment Scope**: concat-model optimizer behavior, experiment artifact
naming, and config preset resolution; no dataset or evaluation redesign  
**Datasets/Data Assumptions**: existing debug and d_model=256 configs remain
the validation inputs, with one synthetic concat test for membership-driven
updates  
**Configuration Inputs**: YAML configs plus `--override` values; new
`model.correction_mode`, `presets.optimizer.<name>`, and family-folder
resolution fields are resolved alongside existing `run`, `model`, `training`,
`dataset`, `monitoring`, and `outputs` sections  
**Experiment Outputs**: `config.json`, `run_summary.json`, `metrics.csv`,
`scaling_results.csv`, checkpoints, `heartbeats.jsonl`, and figure inputs under
the resolved shared family folder  
**Reproducibility Notes**: resolved config saved on rank 0, seed and output
root recorded, resolved correction mode and preset selection recorded, and the
family-folder rule written to saved metadata  
**Performance Goals**: preserve current throughput when correction mode is
`none` or `gmc`; keep LMC overhead limited to concat runs that actually use it  
**Constraints**: keep the flow direct and visible, avoid a callback framework or
deep orchestration layer, and keep the slicing path unchanged unless later work
needs shared plumbing  
**Scale/Scope**: debug matrix runs, d_model=256 pilot runs, a synthetic concat
unit test, and one real concat experiment configuration for smoke validation

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Research code first**: Pass. The feature stays in the existing config
  resolver, training loop, and artifact writers.
- **Simplicity and local reasoning**: Pass. Correction-mode selection, folder
  resolution, and preset merging remain explicit and close to the code that
  uses them.
- **Minimal abstraction and validation**: Pass. Only the validation needed to
  keep correction mode, folder naming, and preset selection coherent is added.
- **Transparent configuration and reproducibility**: Pass. The resolved config
  and run summary will record correction mode, family-folder resolution, and
  preset provenance.
- **Useful outputs and logging**: Pass. Metrics, checkpoints, summaries, and
  figure inputs remain structured filesystem artifacts.
- **Shallow organization**: Pass. Changes stay in the current `train.py`,
  `training/`, `utils/`, `modified_llama.py`, `configs/`, `scripts/`, and
  `tests/` layout.

## Project Structure

### Documentation (this feature)

```text
specs/004-experiment-config-resolution/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── contracts/
    ├── experiment-config.md
    └── run-artifacts.md
```

### Source Code (repository root)

```text
train.py
training/
evaluation/
utils/
modified_llama.py
configs/
scripts/
tests/
outputs/
```

**Structure Decision**: Keep the implementation in the existing shallow
research layout. Use `utils/config.py` for resolved config, naming, and preset
rules; `training/run.py` for the concat optimizer-step seam and run-state
recording; `modified_llama.py` for existing concat metadata and membership
logic; `utils/metrics.py` for saved summary provenance; and focused tests under
`tests/` for config, artifact, and synthetic concat behavior.

## Implementation Strategy

### Workstream 1: Concat Correction Mode

- Add a resolved correction-mode field with the values `none`, `gmc`, and
  `lmc`, and keep the value visible in resolved config and run summary output.
- Reuse the existing concat membership-count metadata already used for GMC
  scale derivation.
- Keep GMC behavior unchanged and move LMC into the optimizer-step path so it
  adjusts the effective learning rate only for concat blocks.
- Fail fast when LMC is selected for a non-concat path or when legacy and new
  correction settings conflict.
- Add synthetic and real concat tests that verify gradients and optimizer
  state remain unchanged under LMC while block update magnitudes change.

### Workstream 2: Shared Family Folder Resolution

- Resolve the shared experiment folder from the largest configured family size
  in the family rather than the active standalone size.
- Preserve the existing family and token-budget components of the folder key
  so only the size component changes when a family comparison folder is
  resolved.
- Apply the resolved folder consistently to config snapshots, summaries,
  metrics, scaling outputs, checkpoints, and figure inputs.
- Record the folder-resolution rule in saved metadata so reruns remain
  deterministic and explainable.
- Add regression tests that prove standalone `s`, `m`, and `l` runs resolve to
  the same shared folder when they belong to the same family definition.

### Workstream 3: Section-Scoped Presets

- Introduce a config-driven preset registry for reusable sections, starting
  with optimizer presets under `presets.optimizer.<name>`.
- Resolve presets during config loading, then merge explicit config values and
  CLI overrides on top of the selected preset.
- Preserve deep-merge behavior for nested mappings so partial overrides change
  only the targeted nested fields.
- Record both the selected preset name and the final merged values in resolved
  config and run summary artifacts.
- Add validation for unknown preset names and conflicting preset combinations.

### Validation Strategy

- Add config-resolution coverage for correction mode, family-folder naming,
  and preset selection/override precedence.
- Add a synthetic concat test for LMC that checks effective learning-rate
  changes without gradient or optimizer-state mutation.
- Add a real concat smoke test that exercises the new correction mode on an
  actual concat experiment config.
- Add artifact assertions for the shared family folder and the resolved
  metadata recorded in `config.json` and `run_summary.json`.
- Add a slicing-path non-regression check so existing `none` and `gmc`
  behavior remains unchanged while the concat LMC path is introduced.
- Add a figure-generation smoke test that reads the new shared folder structure
  directly from `outputs/` without manual copying or renaming.

## Complexity Tracking

No constitution violations require justification for this plan.
