# Implementation Plan: MatFormer Language Model Reproduction

**Branch**: `001-matformer-lm-reproduction` | **Date**: 2026-05-13 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-matformer-lm-reproduction/spec.md`

## Summary

Build a phased MatFormer language-model reproduction around the existing
`train.py` and `modified_llama.py` research code. The implementation will first
prove a debug-size nested S/M/L/XL matrix against matched standalone baselines,
then correct the pilot path into a d_model=256 MatFormer-Llama/SwiGLU
comparison workflow inspired by the MatLM 78M table row. The pilot is not
parameter-count aligned with the paper table because the implementation uses a
Llama/SwiGLU-style gated FFN and a different language-model-head counting
convention. The plan keeps experiment flow visible, adds only shallow helper
modules, and makes every run write configs, metrics, summaries, plots,
disaggregated parameter counts, mismatch notes, and checkpoints when needed for
comparison.

## Technical Context

**Language/Version**: Python 3.12 in the `elasticnn` conda environment is the
working runtime for implementation and smoke validation. Direct interpreter
check during planning observed Python 3.12.13 with `torch`,
`transformers`, and `datasets` importable. The repository MUST keep
`requirements.txt` updated as the portable dependency manifest for rebuilding a
compatible environment when `elasticnn` is unavailable.
**Primary Dependencies**: PyTorch for training/FSDP, Hugging Face Transformers
for Llama model/config/tokenizer primitives, Hugging Face Datasets for public
text datasets, PyYAML for config loading, pandas or Python CSV/JSON for
summaries, matplotlib for plots, pytest for focused checks, and EleutherAI LM
Evaluation Harness for downstream tasks once downstream evaluation begins.
Dependencies belong in `requirements.txt` when added or removed.
**Storage**: Configurable filesystem outputs under `<output_root>/<run_id>/`,
defaulting to `outputs/<run_id>/`, plus optional checkpoint directories for
model extraction/resume/inspection. Pilot runs with validation enabled must
write a rank-0-safe best-eval checkpoint under the run output directory and
record the path in `run_summary.json`; runs without validation must record
final-checkpoint or no-checkpoint status. Runners must allow the output root to
live outside the repository filesystem for machines with restricted local space
or inode quotas.
**Testing**: Focused smoke checks and lightweight pytest-style tests for
configuration parsing, FFN prefix slicing, non-embedding parameter counting,
artifact writing, checkpoint summary fields, pilot comparison rows, and small
debug runs. Quickstart commands provide manual end-to-end validation.
**Target Platform**: Linux workstation or server with CPU support for tiny
smoke tests and CUDA GPUs for meaningful training; FSDP remains available for
multi-GPU runs through `torch.distributed.run`. Phase 4.6 narrows the first
distributed pilot target to single-node Slurm jobs.
**Project Type**: Research training pipeline, model variant reproduction,
baseline comparison workflow, and evaluation/reporting toolchain.  
**Experiment Scope**: MatFormer-Llama/SwiGLU nested FFN training, standalone
FFN-width baselines, debug-size S/M/L/XL matrix, d_model=256 pilot comparison
workflow inspired by the MatLM 78M table row, optional runs using the MatLM
table-row 10B-token budget reference, scaling reports, consistency analysis,
mix-and-match granularities, and speculative decoding alignment.  
**Datasets/Data Assumptions**: TinyStories or Tiny Shakespeare for debug
validation; FineWeb/SlimPajama subsets for medium trend reproduction; FineWeb,
SlimPajama, and C4 candidates for larger phases. Exact paper data is
proprietary and out of scope.  
**Configuration Inputs**: Simple YAML files plus CLI overrides for phase,
model family, sampling mode, granularity, explicit shape fields, MatLM table
reference label when applicable, dataset, token budget, seed, run id, output
root, optional explicit output directory, checkpoint policy, parameter-count
convention, and evaluation suite.
**Experiment Outputs**: `config.json`, `metrics.csv`, `task_results.csv`,
`scaling_results.csv`, `consistency_results.csv`, `run_summary.json`, plots,
heartbeats, parameter-count fields, mismatch notes, and checkpoints when
needed.
**Reproducibility Notes**: Save resolved config for every run, log seeds when
set, record dataset identity/preprocessing assumptions, record the resolved
output root and output directory, label d_model=256 reduced-token pilots
separately from MatLM table-row 10B-token budget reference runs, state whether
the LM head is tied, untied, excluded, or separately counted, and link every
plot point back to run artifacts. Budget-derived training length and actual
implementation parameter counts must be saved in resolved configs and
summaries.
**Performance Goals**: Debug matrix completes quickly enough for iteration;
d_model=256 pilot comparison records tokens/sec, wall-clock time, estimated
compute when available, and peak memory. The reproduction prioritizes trend
fidelity and honest comparison labels over exact numerical or parameter-count
replication.
**Constraints**: The operational environment is `elasticnn`; direct shell
`python3` may not match it and should not be treated as the experiment runtime.
The `requirements.txt` file is the dependency reproduction contract and must
stay in sync with code and docs. Proprietary training data is unavailable;
compute may not cover the MatLM table-row 10B-token budget or larger budgets;
the pilot must not claim exact paper architecture, parameter count, or
training-behavior alignment unless the implementation actually matches those
claims. Pilot artifacts must preserve explicit shape fields such as d_model,
layer count, attention-head count, context length, vocabulary assumption, token
budget, and granularity prefixes. Repository-local storage may have restricted
space or inode capacity, so run artifacts and optional caches must be
redirectable before larger phases begin. Distributed pilot work is limited to
single-node Slurm execution before any multi-node scaling path is planned.
**Scale/Scope**: Required first implementation scope is debug-size nested plus
S/M/L/XL standalone matrix. Next scope is the d_model=256 pilot comparison with
`nested-random`, `nested-all`, and standalone S/M/L/XL modes where compute
allows, derived budget length, disaggregated parameter counts, best-eval
checkpoint persistence, and single-node distributed pilot observability.

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
  artifacts, mislabeled token budgets, misleading parameter-count labels,
  duplicate distributed artifact writes, missing checkpoint paths, and missing
  long-running-job observability.
- **Transparent configuration and reproducibility**: PASS. Each run writes a
  resolved config, seed, dataset assumptions, run summary, completion labels,
  budget-derived training length, actual parameter-count components, and
  mismatch notes.
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
└── dmodel256_pilot_comparison.yaml
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
├── run_dmodel256_pilot.sh
├── slurm_dmodel256_pilot.sh
└── make_figures.py
<output_root>/
└── <run_id>/
tests/
├── test_config.py
├── test_matformer_prefixes.py
├── test_model_size.py
├── test_pilot_comparison.py
├── test_checkpoints.py
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

This phase blocks real d_model=256 pilot comparison execution because
reduced-token pilot labels are only trustworthy when the planned and observed
token budgets are explicit.
After Phase 4.7, documentation and summaries should refer to this as blocking
the d_model=256 pilot comparison rather than an exact MatLM-paper
reproduction.

## Phase 4.6 Distributed Pilot Execution and Runtime Observability

Phase 4.6 is a cross-cutting planning addendum discovered after the initial
Phase 4 work. It prepares the d_model=256 reduced-token pilot for single-node
multi-GPU Slurm execution and durable progress reporting before downstream or
larger scaling work begins.

The first distributed target is deliberately narrow:

- Use a Slurm launcher for the d_model=256 pilot that requests one node with
  multiple GPUs and launches one process per GPU.
- Use the config-driven training path with PyTorch distributed/FSDP rather than
  the older CLI-only training path.
- Resolve budget-derived training length from the effective distributed
  `WORLD_SIZE` exposed to the training process.
- Restrict shared artifact writes to rank 0 for resolved configs, metrics,
  summaries, checkpoints, and heartbeat JSONL.
- Keep local interactive runs readable; progress bars may be used locally, but
  Slurm defaults to clean heartbeat lines and durable JSONL events.

Runtime observability is required because the d_model=256 pilot can spend
meaningful time in dataset loading, preprocessing, model initialization, FSDP
wrapping, training, validation, checkpointing, and artifact writing. The
implementation should emit stage start/completion events to stdout and a
run-local JSONL heartbeat artifact. Heartbeats are not a replacement for
metrics; they are an operational trace for diagnosing long-running scheduler
jobs.

Follow-up task generation for Phase 4.6 should cover:

- Slurm wrapper updates for single-node multi-GPU d_model=256 pilot submission.
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

## Phase 4.7 Pilot Terminology, Parameter Reporting, Checkpointing, and Comparison Scope

Phase 4.7 is a cross-cutting correction layer before downstream or larger
evaluation phases. It does not restart planning. It aligns the existing pilot
with the clarified spec: the pilot is a d_model=256 MatFormer-Llama/SwiGLU
workflow inspired by the MatLM 78M table row, not an exact MatLM-paper
reproduction.

The plan should preserve explicit shape fields and table references without
turning them into claims:

- Keep fields for d_model, layer count, attention-head count, context length,
  vocabulary-size assumption, token budget, and granularity prefixes in
  configs, resolved configs, summaries, scaling rows, and comparison artifacts.
- Replace single-label reporting such as "78M" with a combination of
  `model_shape_label`, optional `table_reference_label`, and actual
  implementation parameter-count fields.
- Record mismatch notes for the Llama/SwiGLU gated FFN, LM-head counting
  convention, token-budget reductions, and any future architecture or training
  behavior deviation.

Parameter reporting should be implemented as a small, inspectable utility in
`utils/model_size.py` rather than a framework layer. It should report at least
`total_parameters`, `embedding_parameters`, `lm_head_parameters`,
`non_embedding_parameters`, and `ffn_parameters`; when feasible it should also
report `attention_parameters` and `other_non_embedding_parameters`. Every
artifact that exposes counts must also state whether the LM head is tied,
untied, excluded, or separately counted.

Checkpoint persistence should remain part of the visible training flow. Pilot
runs with validation enabled save a best-eval checkpoint under the run output
directory, selected by validation loss or perplexity and written only by rank 0
under distributed/FSDP execution. `run_summary.json` records
`checkpoint_status`, `best_checkpoint_path` when available, and the metric used
to select it. Runs without validation either save a final checkpoint or record
that no best-eval checkpoint was produced.

The pilot runner should default to comparison mode, not a single nested run.
The default comparison workflow includes:

- `nested-random`: the primary nested pilot mode matching the original
  `train.py` random granularity sampling behavior.
- `nested-all`: the existing all-at-once averaged-loss behavior, retained as an
  ablation.
- `standalone`: independently trained S, M, L, and XL pilot baselines where
  compute allows.

Smoke and debug invocations may run only one selected mode, but configs,
metrics, scaling rows, summaries, and documentation must label the selected
mode explicitly. Missing standalone baselines are acceptable only when
comparison artifacts mark them as omitted or pending.

Follow-up task generation for Phase 4.7 should cover:

- Config and CLI schema changes for `model_shape_label`,
  `table_reference_label`, `sampling_mode`, checkpoint policy, and explicit
  shape fields.
- Dependency-manifest maintenance so `requirements.txt` remains sufficient to
  rebuild a compatible environment when `elasticnn` is unavailable.
- Preferred runner and config names for the d_model=256 pilot comparison.
- Actual implementation parameter-count reporting and mismatch notes across
  resolved configs, run summaries, scaling rows, and comparison artifacts.
- Rank-0-safe best-eval checkpoint writing and `run_summary.json` references.
- Default pilot comparison orchestration for `nested-random`, `nested-all`, and
  standalone S/M/L/XL runs.
- Focused tests for count components, LM-head convention, checkpoint summaries,
  sampling-mode labels, and omitted-baseline rows.

## Re-evaluated Later Phases

Phase 5, Phase 6, and Phase 7 remain enhancements to the running pilot, but
they should consume Phase 4.7-corrected artifacts:

- **Phase 5 - Downstream trend evaluation**: run only after pilot comparison
  rows expose actual parameter counts, sampling mode, token budget, effective
  world size, checkpoint path when available, and mismatch notes.
- **Phase 6 - Consistency and mix-and-match evaluation**: compare nested and
  standalone alignment with the same `nested-random`, `nested-all`, and
  `standalone` labels so alignment results cannot be confused across training
  modes.
- **Phase 7 - Speculative decoding alignment**: use saved best-eval or final
  checkpoints from Phase 4.7 where available, and report whether missing
  checkpoints prevent a draft/verifier comparison.

## Complexity Tracking

No constitution violations identified. The only added structure is shallow and
directly supports experiment comparison, reproducibility, checkpoint reuse, and
output analysis. Phase 4.7 adds schema fields and small utilities rather than a
new orchestration framework.
