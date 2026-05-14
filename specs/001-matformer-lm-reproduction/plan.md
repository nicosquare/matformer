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
**Storage**: Configurable filesystem outputs under `<output_root>/<run_id>/`,
defaulting to `outputs/<run_id>/`, plus optional checkpoint directories for
model extraction/resume/inspection. Runners must allow the output root to live
outside the repository filesystem for machines with restricted local space or
inode quotas.
**Testing**: Focused smoke checks and lightweight pytest-style tests for
configuration parsing, FFN prefix slicing, non-embedding parameter counting,
artifact writing, and small debug runs. Quickstart commands provide manual
end-to-end validation.  
**Target Platform**: Linux workstation or server with CPU support for tiny
smoke tests and CUDA GPUs for meaningful training; FSDP remains available for
multi-GPU runs through `torch.distributed.run`. Phase 4.6 narrows the first
distributed pilot target to single-node Slurm jobs.
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
token budget, seed, run id, output root, optional explicit output directory,
checkpoint policy, and evaluation suite.
**Experiment Outputs**: `config.json`, `metrics.csv`, `task_results.csv`,
`scaling_results.csv`, `consistency_results.csv`, `run_summary.json`, plots,
heartbeats, and checkpoints when needed.
**Reproducibility Notes**: Save resolved config for every run, log seeds when
set, record dataset identity/preprocessing assumptions, record the resolved
output root and output directory, label reduced-token pilots separately from
paper-budget complete runs, and link every plot point back to run artifacts.
Budget-derived training length must be saved in resolved configs and summaries.
**Performance Goals**: Debug matrix completes quickly enough for iteration;
78M path records tokens/sec, wall-clock time, estimated compute when available,
and peak memory. The reproduction prioritizes trend fidelity over exact
numbers.  
**Constraints**: Current default Python environment lacks ML dependencies;
proprietary training data is unavailable; compute may not cover full 78M/10B or
larger budgets; paper-aligned runs must preserve 16 layers, 16 heads, context
1024, and 256k vocabulary assumption unless labeled non-paper-aligned.
Repository-local storage may have restricted space or inode capacity, so run
artifacts and optional caches must be redirectable before larger phases begin.
Distributed pilot work is limited to single-node Slurm execution before any
multi-node scaling path is planned.
**Scale/Scope**: Required first implementation scope is debug-size nested plus
S/M/L/XL standalone matrix. Next scope is 78M paper-aligned architecture with
reduced-token pilot support, explicit 78M/10B completion label, derived budget
length, and single-node distributed pilot observability.

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
  artifacts, mislabeled token budgets, duplicate distributed artifact writes,
  and missing long-running-job observability.
- **Transparent configuration and reproducibility**: PASS. Each run writes a
  resolved config, seed, dataset assumptions, run summary, completion labels,
  and budget-derived training length.
- **Useful outputs and logging**: PASS. The plan requires CSV/JSON summaries,
  plot inputs, readable console logging, heartbeat artifacts for long-running
  Slurm jobs, and no metrics stored only in terminal output.
- **Shallow organization**: PASS. New files use existing shallow directories
  such as `training/`, `utils/`, and `scripts/`; no framework layer is needed
  for distributed pilot orchestration.

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
<output_root>/
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

## Output Storage and Cache Policy

`run.output_root` is the preferred storage control. It defaults to `outputs/`
and each run resolves `run.output_dir` to `<output_root>/<run_id>`. A direct
`run.output_dir` value remains an explicit escape hatch for one-off runs, but
matrix runners should prefer output-root derivation so all runs in a matrix can
move together.

Researcher-facing commands must accept output-root control through config,
CLI arguments, and `OUTPUT_ROOT`. Runners should create a missing output root
when possible and fail before training if the resolved root is not writable.

External dependency caches are separate from run artifacts. Documentation and
runner guidance should tell researchers to move Hugging Face caches with
`HF_HOME`, `HF_DATASETS_CACHE`, and `TRANSFORMERS_CACHE` when repository or
home filesystems have limited space or inodes.

## Phase 4.5 Budget-Derived Training Length

Budgeted runs treat `training.token_budget` as the source of truth. Source YAML
contains hand-authored inputs such as token budget, per-process batch size, and
optional safety caps. Resolved `config.json` records derived fields:
`effective_world_size`, `expected_tokens_per_step`, `derived_max_steps`, and
the effective `max_steps` used by training.

`effective_world_size` must come from the active distributed `WORLD_SIZE` when
distributed training is launched, otherwise 1. It must not be inferred from the
number of visible or allocated GPUs. Run summaries copy the resolved planning
fields and add runtime outcomes including `tokens_seen` and `stop_reason`.

This phase blocks real 78M pilot execution because reduced-token pilot labels
are only trustworthy when the planned and observed token budgets are explicit.

## Phase 4.6 Distributed Pilot Execution and Runtime Observability

Phase 4.6 is a cross-cutting planning addendum discovered after the initial
Phase 4 work. It prepares the 78M reduced-token pilot for single-node
multi-GPU Slurm execution and durable progress reporting before downstream or
larger scaling work begins.

The first distributed target is deliberately narrow:

- Use a Slurm launcher for the 78M pilot that requests one node with multiple
  GPUs and launches one process per GPU.
- Use the config-driven training path with PyTorch distributed/FSDP rather than
  the older CLI-only training path.
- Resolve budget-derived training length from the effective distributed
  `WORLD_SIZE` exposed to the training process.
- Restrict shared artifact writes to rank 0 for resolved configs, metrics,
  summaries, checkpoints, and heartbeat JSONL.
- Keep local interactive runs readable; progress bars may be used locally, but
  Slurm defaults to clean heartbeat lines and durable JSONL events.

Runtime observability is required because the 78M pilot can spend meaningful
time in dataset loading, preprocessing, model initialization, FSDP wrapping,
training, validation, checkpointing, and artifact writing. The implementation
should emit stage start/completion events to stdout and a run-local JSONL
heartbeat artifact. Heartbeats are not a replacement for metrics; they are an
operational trace for diagnosing long-running scheduler jobs.

Follow-up task generation for Phase 4.6 should cover:

- Slurm wrapper updates for single-node multi-GPU 78M pilot submission.
- Distributed config-driven model wrapping and dataloader behavior.
- Rank-aware artifact writing and summary finalization.
- Heartbeat JSONL schema, writer, and stage instrumentation.
- Focused tests for rank-0-only writes, effective world-size propagation, and
  heartbeat artifacts.
- Quickstart documentation for queueing the distributed pilot and inspecting
  progress.

This phase should remain shallow and visible. Avoid introducing callback
systems, training frameworks, or logging stacks; small helpers are acceptable
only where they prevent duplicate rank-aware write logic or make heartbeats
consistent across stages.

## Complexity Tracking

No constitution violations identified. The only added structure is shallow and
directly supports experiment comparison, reproducibility, and output analysis.
