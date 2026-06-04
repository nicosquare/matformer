# Implementation Plan: Experiment Config Resolution

**Branch**: `004-experiment-config-resolution` | **Date**: 2026-06-03 | **Spec**: [/home/nicolas.avila/dev/references/matformer/specs/004-experiment-config-resolution/spec.md](/home/nicolas.avila/dev/references/matformer/specs/004-experiment-config-resolution/spec.md)
**Input**: Feature specification from `/specs/004-experiment-config-resolution/spec.md`

## Summary

Add concat-only learning-rate membership correction as an explicit correction
mode, resolve standalone family runs into the shared largest-size experiment
folder, and introduce section-scoped config presets stored as separate YAML
registry files under `configs/presets/`, starting with optimizer presets.
Keep the changes config-driven, local to the existing config resolver and
training loop, and record the resolved correction mode, folder rule, and
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
`model.correction_mode`, `model.membership_correction`,
`training.optimizer.preset`, and separate preset registry YAML files under
`configs/presets/` are resolved alongside existing `run`, `model`,
`training`, `dataset`, `monitoring`, and `outputs` sections  
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
configs/presets/
scripts/
tests/
outputs/
```

**Structure Decision**: Keep the implementation in the existing shallow
research layout. Use `utils/config.py` for resolved config, naming, and preset
registry loading rules; `training/run.py` for the concat optimizer-step seam
and run-state recording; `modified_llama.py` for existing concat metadata and
membership logic; `utils/metrics.py` for saved summary provenance; `configs/presets/`
for reusable optimizer preset YAML files; and focused tests under `tests/` for
config, artifact, and synthetic concat behavior.

## Complexity Tracking

No constitution violations require justification for this plan.
