# Implementation Plan: MatFormer Language Model Reproduction

**Branch**: `001-matformer-lm-reproduction` | **Date**: 2026-05-13 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-matformer-lm-reproduction/spec.md`

## Summary

Build a phased MatFormer language-model reproduction around the existing
`train.py` and `modified_llama.py` research code. The implementation will first
prove a debug-size nested S/M/L/XL matrix against matched standalone baselines,
then add a paper-aligned 78M path with explicit reduced-token pilot versus
78M/10B paper-budget completion labels. The plan keeps experiment flow visible,
adds only shallow helper modules, and makes every run write configs, metrics,
summaries, plots, and checkpoints when needed for comparison.

## Technical Context

**Language/Version**: Python 3.12.3 observed in default shell. Current default
`python3` environment does not have `torch`, `transformers`, or `datasets`
installed, so runnable implementation requires an experiment environment with
those dependencies.  
**Primary Dependencies**: PyTorch for training/FSDP, Hugging Face Transformers
for Llama model/config/tokenizer primitives, Hugging Face Datasets for public
text datasets, pandas or Python CSV/JSON for summaries, matplotlib for plots,
and EleutherAI LM Evaluation Harness for downstream tasks once downstream
evaluation begins.  
**Storage**: Local filesystem outputs under `outputs/<run_id>/`, plus optional
checkpoint directories for model extraction/resume/inspection.  
**Testing**: Focused smoke checks and lightweight pytest-style tests for
configuration parsing, FFN prefix slicing, non-embedding parameter counting,
artifact writing, and small debug runs. Quickstart commands provide manual
end-to-end validation.  
**Target Platform**: Linux workstation or server with CPU support for tiny
smoke tests and CUDA GPUs for meaningful training; FSDP remains available for
multi-GPU runs through `torch.distributed.run`.  
**Project Type**: Research training pipeline, model variant reproduction,
baseline comparison workflow, and evaluation/reporting toolchain.  
**Experiment Scope**: MatFormer nested FFN training, standalone FFN-width
baselines, debug-size S/M/L/XL matrix, 78M reduced-token pilot, optional
78M/10B paper-budget completion, scaling reports, consistency analysis,
mix-and-match granularities, and speculative decoding alignment.  
**Datasets/Data Assumptions**: TinyStories or Tiny Shakespeare for debug
validation; FineWeb/SlimPajama subsets for medium trend reproduction; FineWeb,
SlimPajama, and C4 candidates for larger phases. Exact paper data is
proprietary and out of scope.  
**Configuration Inputs**: Simple YAML files plus CLI overrides for phase,
model family, granularity, model size label, architecture scale, dataset,
token budget, seed, run id, output directory, checkpoint policy, and evaluation
suite.  
**Experiment Outputs**: `config.json`, `metrics.csv`, `task_results.csv`,
`scaling_results.csv`, `consistency_results.csv`, `run_summary.json`, plots,
and checkpoints when needed.  
**Reproducibility Notes**: Save resolved config for every run, log seeds when
set, record dataset identity/preprocessing assumptions, label reduced-token
pilots separately from paper-budget complete runs, and link every plot point
back to run artifacts.  
**Performance Goals**: Debug matrix completes quickly enough for iteration;
78M path records tokens/sec, wall-clock time, estimated compute when available,
and peak memory. The reproduction prioritizes trend fidelity over exact
numbers.  
**Constraints**: Current default Python environment lacks ML dependencies;
proprietary training data is unavailable; compute may not cover full 78M/10B or
larger budgets; paper-aligned runs must preserve 16 layers, 16 heads, context
1024, and 256k vocabulary assumption unless labeled non-paper-aligned.  
**Scale/Scope**: Required first implementation scope is debug-size nested plus
S/M/L/XL standalone matrix. Next scope is 78M paper-aligned architecture with
reduced-token pilot support and explicit 78M/10B completion label.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Research code first**: PASS. The plan extends the existing scripts with
  shallow `configs/`, `training/`, `evaluation/`, `utils/`, `scripts/`, and
  `outputs/` structure. It avoids service layers and framework-style
  orchestration.
- **Simplicity and local reasoning**: PASS. Core experiment flow remains in
  visible scripts; helper modules are scoped to config loading, metrics, plots,
  and model utilities with explicit tensor-shape checks.
- **Minimal abstraction and validation**: PASS. Validation focuses on silent
  research failures: FFN prefix shape mistakes, mismatched baselines, missing
  artifacts, and mislabeled token budgets.
- **Transparent configuration and reproducibility**: PASS. Each run writes a
  resolved config, seed, dataset assumptions, run summary, and completion
  labels.
- **Useful outputs and logging**: PASS. The plan requires CSV/JSON summaries,
  plot inputs, readable console logging, and no metrics stored only in terminal
  output.
- **Shallow organization**: PASS. New directories are one level deep and match
  the constitution's research-code layout.

**Post-design re-check**: PASS. The Phase 1 artifacts preserve the same shallow
structure and do not introduce unjustified abstraction.

## Project Structure

### Documentation (this feature)

```text
specs/001-matformer-lm-reproduction/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── cli.md
│   ├── experiment-config.md
│   └── run-artifacts.md
└── tasks.md
```

### Source Code (repository root)

```text
train.py                 # Existing training entry point; evolves into visible run flow
modified_llama.py        # Existing MatFormer FFN slicing implementation
configs/
├── debug_matrix.yaml
└── 78m_reduced_pilot.yaml
training/
├── data.py
├── run.py
└── baselines.py
evaluation/
├── validation.py
├── downstream.py
├── consistency.py
└── speculative.py
utils/
├── config.py
├── metrics.py
├── model_size.py
└── plotting.py
scripts/
├── run_debug_matrix.sh
├── run_78m_pilot.sh
└── make_figures.py
outputs/
└── <run_id>/
tests/
├── test_config.py
├── test_matformer_prefixes.py
├── test_model_size.py
└── test_artifacts.py
```

**Structure Decision**: Use a shallow research layout. Keep the main training
path readable from `train.py` or `training/run.py`, and only split repeated,
stable concerns into small files. Do not introduce registries, factories, or
deep package hierarchies.

## Complexity Tracking

No constitution violations identified. The only added structure is shallow and
directly supports experiment comparison, reproducibility, and output analysis.
