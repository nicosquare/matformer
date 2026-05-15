# Feature Specification: MatFormer Language Model Reproduction

**Feature Branch**: `001-matformer-lm-reproduction`  
**Created**: 2026-05-13  
**Status**: Draft  
**Input**: User description: "Look at notes/step_1.md as a detailed guidance
for this specification. You are free to propose freely about the phases and
order of execution and we can clarify any possible improvement in the detailed
preliminary plan."

## Clarifications

### Session 2026-05-13

- Q: What is the required first milestone boundary? -> A: P1 plus minimal
  baseline: small nested proof-of-concept plus at least one matched standalone
  baseline comparison.
- Q: What standalone baseline coverage is required before scaling work? -> A:
  One-size full granularity matrix: S, M, L, and XL standalone baselines for the
  first small model size.
- Q: What is the first model-size milestone? -> A: Two-stage: complete a
  debug-size S/M/L/XL nested and standalone matrix first, then run a
  d_model=256 MatFormer-Llama/SwiGLU pilot inspired by the MatLM 78M table row.
- Q: What is the architecture fidelity boundary? -> A: Debug runs may reduce
  architecture scale, but pilot and scaling reports preserve explicit shape
  fields and avoid exact paper-architecture or parameter-count claims unless
  the implementation actually matches them.
- Q: How should pilot training-token completion be labeled? -> A: Distinguish
  d_model=256 reduced-token pilot runs from runs that use the MatLM table-row
  10B-token budget reference.
- Q: How should restricted repository storage be handled? -> A: All experiment
  outputs must support a configurable output root outside the repository,
  defaulting to `outputs/`, with runner support through `OUTPUT_ROOT` or
  command arguments.

### Session 2026-05-14

- Q: How should budgeted training length be derived? -> A: `training.token_budget`
  is authoritative; derive planned steps from batch size, context length, and
  effective distributed `WORLD_SIZE`, defaulting to 1 and never inferred from
  available GPU count alone.
- Q: What multi-GPU scope should the d_model=256 pilot support before Phase 5?
  -> A: Single-node multi-GPU execution now; multi-node execution is explicitly
  out of scope for this phase.
- Q: What distributed strategy should the config-driven d_model=256 pilot use? -> A:
  Fully Sharded Data Parallel (FSDP).
- Q: What heartbeat logging outputs are required for long Slurm jobs? -> A:
  Both human-readable stdout lines and durable JSONL event artifacts.
- Q: What should trigger heartbeat emission? -> A: Both step interval and
  elapsed-time interval, whichever comes first, with time-based heartbeats for
  non-step pipeline stages.

### Session 2026-05-15

- Q: How should Phase 4.7 pilot terminology be framed? -> A: Frame it as a
  d_model=256 MatFormer-Llama/SwiGLU pilot inspired by the MatLM 78M table row,
  not as an exact MatLM-paper reproduction.
- Q: How must pilot parameter reporting be structured? -> A: Report
  disaggregated actual implementation counts, LM-head counting convention, and
  mismatch notes against paper table counts.
- Q: What checkpoint persistence is required for pilot runs? -> A: Save a
  rank-0-safe best-eval checkpoint when validation is enabled; otherwise record
  final-checkpoint or no-checkpoint status in `run_summary.json`.
- Q: What comparison workflow should the pilot default to? -> A: Default to
  nested-random, nested-all, and standalone S/M/L/XL baselines where compute
  allows, with explicit labels.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Validate Nested MatFormer Training (Priority: P1)

As a researcher, I want a small-scale MatFormer language-model reproduction
that trains all nested FFN granularities together, extracts each submodel, and
reports validation loss and perplexity so I can verify that the nested training
mechanism works before investing in larger runs.

**Why this priority**: The nested training and extraction behavior is the core
claim. No baseline, scaling, or downstream comparison is meaningful until this
path works end-to-end.

**Independent Test**: Run a small-scale reproduction using a tiny public text
dataset and confirm that every granularity produces a saved configuration,
scalar metrics, extracted model identity, validation loss, perplexity, and at
least one matched standalone baseline comparison.

**Acceptance Scenarios**:

1. **Given** a small public text dataset and the configured MatFormer
   granularities, **When** the researcher runs a proof-of-concept experiment,
   **Then** the run evaluates S, M, L, and XL granularities and writes structured
   metrics for each one.
2. **Given** a completed nested training run, **When** the researcher extracts a
   smaller granularity, **Then** the extracted submodel is a strict prefix of the
   larger FFN and can be evaluated independently.
3. **Given** a completed proof-of-concept run, **When** the researcher reviews
   outputs, **Then** configuration, metrics, and reproducibility metadata are
   available without reading terminal logs.
4. **Given** the first milestone is complete, **When** the researcher reviews
   baseline coverage, **Then** the debug-size experiment includes S, M, L, and
   XL nested and standalone comparisons.
5. **Given** the repository filesystem has restricted space or inode capacity,
   **When** the researcher sets an external output root, **Then** all run
   artifacts for the proof-of-concept and matched baseline are written under
   that configured root.

---

### User Story 2 - Compare Against Standalone Baselines (Priority: P2)

As a researcher, I want independently trained standalone models for each
MatFormer granularity so I can compare each extracted nested submodel against a
fixed-size baseline at approximately equal parameter count.

**Why this priority**: The central reproduction comparison is nested extracted
submodel versus independently trained model. Without standalone baselines, the
evaluation is incomplete.

**Independent Test**: For the debug-size matrix and d_model=256
MatFormer-Llama/SwiGLU pilot, confirm S, M, L, and XL each have a matching
standalone run with the same model family, tokenizer assumptions, dataset
phase, optimizer assumptions, and training-token budget except for FFN width,
and that comparison artifacts expose actual implementation parameter counts.

**Acceptance Scenarios**:

1. **Given** a MatFormer run for the four granularities, **When** standalone
   baselines are prepared, **Then** each granularity has a matched independently
   trained baseline.
2. **Given** the debug-size matrix has completed nested training, **When** the
   researcher moves toward the d_model=256 pilot, **Then** standalone S, M, L,
   and XL baselines exist for the debug size.
3. **Given** completed nested and standalone runs, **When** the researcher
   compares validation metrics, **Then** the comparison is grouped by
   granularity and non-embedding parameter count.
4. **Given** an unmatched baseline configuration, **When** the researcher
   reviews the comparison, **Then** the mismatch is visible in the recorded
   configuration or run summary.

---

### Cross-Cutting Phase 4.6 - Distributed Pilot Execution and Runtime Observability

As a researcher, I want the d_model=256 reduced-token pilot to run as a
single-node multi-GPU Slurm job with traceable runtime progress so I can
complete the pilot on available GPU resources and diagnose long-running jobs
without depending on terminal-only output.

**Why this phase**: The d_model=256 pilot blocks later scaling and downstream
work, and the available GPU memory or throughput may be insufficient unless
the config-driven path can use multiple GPUs. Slurm jobs also need observable
heartbeats during dataset loading, preprocessing, model initialization,
training, validation, and artifact writing.

**Independent Test**: Submit a short single-node multi-GPU Slurm smoke job for
the d_model=256 pilot with a capped step count and confirm the run launches one
process per GPU, records the effective world size, writes shared artifacts only
once, and emits both stdout heartbeat lines and JSONL heartbeat events.

**Acceptance Scenarios**:

1. **Given** a single-node Slurm allocation with multiple GPUs, **When** the
   researcher submits the d_model=256 pilot, **Then** the wrapper launches one
   process per GPU and the config-driven training path uses FSDP.
2. **Given** a distributed d_model=256 pilot run, **When** artifacts are written,
   **Then** shared artifacts such as resolved config, metrics, summaries, and
   heartbeat JSONL are written only by rank 0.
3. **Given** a long-running Slurm job, **When** the job is loading assets,
   preprocessing data, initializing or wrapping the model, training, validating,
   checkpointing, or writing artifacts, **Then** stage start/completion events
   and applicable heartbeat events are visible in scheduler logs and durable
   JSONL artifacts.
4. **Given** an interactive local run, **When** progress bars are enabled,
   **Then** tqdm-style output may be used for user-facing loops; Slurm jobs
   default to clean heartbeat lines instead of progress bars.

---

### Cross-Cutting Phase 4.7 - Pilot Terminology, Checkpointing, and Comparison Scope

As a researcher, I want the pilot to use precise terminology, reusable
checkpoints, disaggregated parameter reports, and matched comparison modes so I
can assess nested MatFormer training against standalone training without
overstating alignment with the MatLM paper table.

**Why this phase**: The implementation uses a Llama/SwiGLU-style gated FFN and
a language-model-head counting convention that differ from the paper table.
Calling the pilot simply "78M" or claiming paper alignment hides real
parameter-count and architecture differences. The pilot also needs reusable
checkpoints and a
default comparison workflow before later evaluation phases depend on its
outputs.

**Independent Test**: Run a capped pilot comparison workflow and confirm it
produces clearly labeled nested-random, nested-all, and standalone artifacts
where configured; writes actual disaggregated parameter counts and mismatch
notes; and records a usable checkpoint path in `run_summary.json` when
validation is enabled.

**Acceptance Scenarios**:

1. **Given** a pilot run configuration, **When** the researcher reads the
   resolved config, run summary, scaling rows, or comparison artifacts, **Then**
   the run is labeled as a d_model=256 MatFormer-Llama/SwiGLU pilot inspired by
   the MatLM 78M table row rather than as an exact MatLM-paper reproduction.
2. **Given** pilot artifacts, **When** parameter counts are reported, **Then**
   the artifacts expose actual implementation counts for total, embedding,
   language-model head, non-embedding, FFN, and feasible attention or other
   non-embedding components, plus the LM-head counting convention.
3. **Given** paper table count references are included, **When** the researcher
   compares them with implementation counts, **Then** mismatches against the
   paper's reported total, non-embedding, and FFN counts are explicit.
4. **Given** validation is enabled for a pilot training run, **When** training
   completes or improves validation loss/perplexity, **Then** a rank-0-safe
   best-eval checkpoint is saved under the run output directory and referenced
   from `run_summary.json`.
5. **Given** the pilot runner is invoked without an explicit smoke/debug mode,
   **When** compute allows the comparison workflow, **Then** it runs or
   schedules nested-random, nested-all, and standalone S/M/L/XL pilot baselines
   with clear labels for model family, granularity, and sampling mode.

---

### User Story 3 - Reproduce Scaling and Downstream Trends (Priority: P3)

As a researcher, I want phased medium- and large-scale reproduction runs that
report scaling behavior and downstream benchmark trends so I can evaluate
whether the reproduction supports the paper's qualitative claims even without
the original proprietary data.

**Why this priority**: Scaling and downstream behavior are the main external
evidence that the reproduction captures the paper's trends beyond a toy run.

**Independent Test**: Complete at least one medium-scale phase with nested and
standalone runs, then generate comparison tables and plots for loss,
perplexity, and average downstream accuracy versus non-embedding parameters.

**Acceptance Scenarios**:

1. **Given** medium-scale nested and standalone runs, **When** the researcher
   generates a scaling report, **Then** the report includes loss and perplexity
   curves for both model families.
2. **Given** downstream evaluation results, **When** the researcher summarizes
   benchmark performance, **Then** the report includes average accuracy and
   per-task results for the configured task suite.
3. **Given** incomplete large-scale resources, **When** only small and
   medium phases are available, **Then** the report clearly labels the achieved
   phase and does not imply exact numerical replication of the original paper.

---

### User Story 4 - Measure Consistency and Elastic Behavior (Priority: P4)

As a researcher, I want consistency and mix-and-match evaluations across nested
granularities so I can measure whether smaller and larger submodels preserve
similar predictions and support elastic inference tradeoffs.

**Why this priority**: Consistency across nested submodels is one of the
distinct claims separating MatFormer from separately trained models.

**Independent Test**: Evaluate at least one smaller/larger granularity pair and
produce token-level agreement or an equivalent alignment metric, plus a
mix-and-match report for heterogeneous layer granularities.

**Acceptance Scenarios**:

1. **Given** two extracted nested granularities, **When** they are evaluated on
   the same text sample, **Then** the output includes an alignment metric such as
   token-level agreement, distribution divergence, or top-k overlap.
2. **Given** standalone baselines of comparable sizes, **When** the researcher
   compares alignment, **Then** nested and standalone alignment results are
   reported separately.
3. **Given** heterogeneous layer granularities, **When** the researcher runs a
   mix-and-match evaluation, **Then** the result records the layer-granularity
   pattern and the corresponding quality or efficiency tradeoff.

---

### User Story 5 - Evaluate Speculative Decoding Alignment (Priority: P5)

As a researcher, I want speculative decoding comparisons between nested
draft/verifier pairs and standalone draft/verifier pairs so I can test whether
nested models are better aligned for draft-token acceptance.

**Why this priority**: Speculative decoding is a downstream use case where
distributional alignment between small and large models can become directly
measurable.

**Independent Test**: Run one nested draft/verifier pair and one standalone
draft/verifier pair on the same prompt set, then compare acceptance rate,
rollback frequency, throughput, and latency.

**Acceptance Scenarios**:

1. **Given** a smaller nested submodel and a larger nested submodel, **When**
   the smaller model is used as a draft model, **Then** the evaluation reports
   draft acceptance and rollback metrics.
2. **Given** standalone draft and verifier models, **When** the same evaluation
   is run, **Then** the comparison separates nested and standalone outcomes.
3. **Given** speculative decoding results, **When** the researcher reviews the
   report, **Then** the report states whether nested draft/verifier alignment is
   better, worse, or inconclusive relative to standalone alignment.

### Edge Cases

- Proprietary pretraining data from the paper is unavailable, so results must
  be framed as faithful trend reproduction rather than exact numerical
  replication.
- Hardware or runtime limits may prevent larger model sizes or token budgets
  from completing; reports must label completed phases and partial coverage.
- Debug-size runs may reduce architecture scale to keep iteration cheap, but
  pilot and scaling artifacts must preserve explicit shape fields and avoid
  exact paper-architecture claims unless those claims are true.
- d_model=256 runs with fewer than the MatLM table-row 10B-token budget must be
  labeled as reduced-token pilots, not table-budget complete runs.
- A single model-size label such as "78M" can hide mismatches introduced by the
  Llama/SwiGLU gated FFN and language-model-head counting convention; pilot
  artifacts must expose actual parameter components instead of relying on that
  label alone.
- Available GPU count can differ from the effective data-parallel world size;
  budget-derived step counts must use the active distributed `WORLD_SIZE` when
  distributed training is launched, otherwise 1.
- Multi-GPU pilot execution is limited to single-node Slurm jobs in Phase 4.6;
  multi-node execution is explicitly out of scope until a later phase.
- A distributed rank can fail before shared artifacts are complete; the run
  must leave enough heartbeat and failure context to identify the last completed
  stage and effective rank/world-size state.
- Tqdm-style progress output can make scheduler logs unreadable; Slurm runs
  must default to heartbeat lines while local interactive runs may opt into
  progress bars.
- A standalone baseline can be accidentally mismatched by dataset, token
  budget, tokenizer assumption, or FFN width; comparison artifacts must expose
  those run attributes.
- Very small datasets can produce unstable perplexity or downstream metrics;
  reports must identify dataset scale and avoid overclaiming.
- Embedding parameters can dominate small models; model-size plots must use
  non-embedding parameter counts when reproducing Figure 2-style results.
- Tied, untied, excluded, or separately counted language-model heads can change
  reported parameter totals; every count report must state the convention used.
- Validation can be disabled for smoke runs; such runs must record whether a
  final checkpoint was saved or that no best-eval checkpoint was produced.
- A run can complete training but fail to export metrics, summaries, or plots;
  such a run is incomplete for reproduction reporting.
- The repository filesystem may have restricted space or inode capacity; run
  artifacts must be redirected to a configured output root before training
  creates large files.
- A configured output root may be missing or unwritable; runners must create a
  missing root when possible and fail before training starts when it cannot be
  written.
- Downstream tasks may be unavailable or too expensive in early phases; the
  minimal evaluation suite must be sufficient to show representative trends.
- Speculative decoding may produce inconclusive alignment differences; the
  feature must report the outcome rather than forcing a positive result.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The reproduction MUST define MatFormer nested language models as
  decoder-only causal language models with shared depth, attention-head count,
  context length, tokenizer assumption, and vocabulary-size assumption across
  granularities.
- **FR-002**: The reproduction MUST support four nested FFN granularities: S,
  M, L, and XL, corresponding to expansion ratios 0.5, 1, 2, and 4.
- **FR-003**: Smaller nested granularities MUST be strict FFN prefixes of larger
  granularities.
- **FR-004**: The nested training objective MUST support explicit sampling
  modes: `nested-all`, which evaluates all configured granularities for each
  training batch and combines their language-model losses, and
  `nested-random`, which samples one configured granularity per batch or step
  using the original `train.py`-style behavior.
- **FR-005**: The reproduction MUST include independently trained standalone
  baselines for every evaluated MatFormer granularity.
- **FR-006**: Standalone baselines MUST match the corresponding nested
  granularity on architecture assumptions, tokenizer assumptions, dataset
  phase, and training-token budget except for the absence of nesting.
- **FR-007**: The reproduction MUST compare each extracted nested submodel with
  its matched standalone baseline at approximately equal non-embedding
  parameter count.
- **FR-008**: The reproduction MUST define phased datasets for small-scale
  validation, medium-scale trend reproduction, and large-scale reproduction.
- **FR-009**: Small-scale validation MUST prioritize fast debugging of nested
  FFN training, extraction logic, and perplexity reporting.
- **FR-010**: The first milestone MUST include at least one matched standalone
  baseline comparison in addition to nested proof-of-concept validation.
- **FR-011**: Before the d_model=256 pilot comparison is treated as complete,
  the reproduction MUST complete a debug-size matrix with S, M, L, and XL
  nested and matched standalone baselines.
- **FR-012**: Medium-scale reproduction MUST compare nested and standalone
  behavior on validation metrics and a representative downstream evaluation
  suite.
- **FR-013**: Large-scale reproduction MUST target scaling-curve behavior,
  Figure 2-style reporting, and comparison against standalone baselines when
  resources allow.
- **FR-014**: The first post-debug pilot MUST be framed as a d_model=256
  MatFormer-Llama/SwiGLU pilot inspired by the MatLM 78M table row, not as an
  exact MatLM-paper reproduction. Pilot configs, resolved configs, run
  summaries, scaling rows, and comparison artifacts MUST expose explicit shape
  and budget fields including `d_model`, layer count, attention-head count,
  context length, vocabulary-size assumption, token budget, and granularity
  prefixes. For budgeted training runs, `training.token_budget` MUST be the
  source of truth for planned training length. The runner MUST derive the
  planned step count from `training.token_budget`,
  `training.batch_size_per_process`, `model.context_length`, and the effective
  data-parallel world size. The effective world size MUST be the active
  distributed `WORLD_SIZE` when distributed training is launched, otherwise 1;
  it MUST NOT be inferred from available GPU count alone. Resolved configs and
  run summaries MUST record `expected_tokens_per_step`, `derived_max_steps`,
  `token_budget`, `tokens_seen`, `effective_world_size`, and `stop_reason`.
- **FR-015**: Debug-size runs MAY reduce architecture scale for speed while
  preserving matched nested-versus-standalone comparisons.
- **FR-016**: Pilot and scaling reports MUST avoid exact paper-architecture,
  parameter-count, or training-behavior alignment claims unless those claims
  are true for the implementation. Known deviations such as the Llama/SwiGLU
  gated FFN, language-model-head counting convention, token-budget reductions,
  or any changed shape field MUST be recorded as mismatch notes.
- **FR-017**: Figure 2-style reporting MUST plot loss, perplexity, downstream
  accuracy, and consistency against non-embedding parameters.
- **FR-018**: Parameter reporting MUST disaggregate actual implementation
  counts into `total_parameters`, `embedding_parameters`,
  `lm_head_parameters`, `non_embedding_parameters`, and `ffn_parameters`, plus
  `attention_parameters` and `other_non_embedding_parameters` when feasible.
  Reports MUST state whether the language-model head is tied, untied, excluded,
  or separately counted. `non_embedding_parameters` MUST exclude token
  embeddings and output embedding or language-model-head parameters.
- **FR-019**: Downstream evaluation MUST include a minimal representative suite
  covering completion, commonsense reasoning, general reasoning, question
  answering, and coreference-style tasks.
- **FR-020**: The reproduction SHOULD allow broader downstream evaluation
  across open-domain QA, cloze/completion, Winograd-style reasoning, reading
  comprehension, commonsense reasoning, SuperGLUE, and ANLI categories.
- **FR-021**: Consistency evaluation MUST compare predictions or distributions
  between smaller and larger nested granularities.
- **FR-022**: Consistency evaluation MUST report at least one metric such as
  token-level agreement, distribution divergence, top-k overlap, or speculative
  decoding acceptance rate.
- **FR-023**: Mix-and-match evaluation MUST record the layer-level granularity
  pattern used for each heterogeneous nested model.
- **FR-024**: Speculative decoding evaluation MUST compare nested draft/verifier
  pairs against standalone draft/verifier pairs.
- **FR-025**: Speculative decoding evaluation MUST report acceptance rate,
  rollback frequency, throughput, and latency.
- **FR-026**: Reports MUST explicitly state that exact numerical reproduction
  is not expected because the original pretraining data is proprietary.
- **FR-027**: Reports MUST distinguish completed phases from proposed or
  resource-dependent phases.
- **FR-028**: Reports MUST distinguish d_model=256 reduced-token pilot runs
  from runs that use the MatLM table-row 10B-token budget reference.
- **FR-029**: Experiment workflows MUST support a configurable output root for
  all generated run artifacts, defaulting to `outputs/` when the researcher does
  not provide one.
- **FR-030**: Matrix and single-run workflows MUST resolve run artifacts under
  `<output_root>/<run_id>` unless an explicit per-run output directory is
  intentionally provided.
- **FR-031**: Researcher-facing runner commands MUST allow the output root to be
  set through configuration, command arguments, or an `OUTPUT_ROOT` environment
  variable so artifacts can be written outside the repository filesystem.
- **FR-032**: The config-driven d_model=256 pilot MUST support single-node
  multi-GPU Slurm execution before Phase 5; multi-node execution is out of
  scope for this phase.
- **FR-033**: Distributed config-driven pilot execution MUST launch one process
  per GPU, initialize torch distributed state, set each process to its local
  CUDA device, and use FSDP for the local MatFormer/Llama model.
- **FR-034**: Distributed config-driven training MUST use distributed-aware data
  sampling and MUST ensure shared artifacts are written only by rank 0.
- **FR-035**: Resolved configs and run summaries for distributed runs MUST
  expose the active rank/world-size context needed to interpret
  token-budget-derived step counts and runtime outcomes.
- **FR-036**: Long-running pipeline stages MUST emit structured runtime events
  for stage start, stage completion, and heartbeat progress during tokenizer
  loading, dataset loading, dataset preprocessing, dataloader creation, model
  initialization, FSDP wrapping, training, validation, checkpointing, and
  artifact writing when those stages occur.
- **FR-037**: Heartbeat logging MUST write both human-readable stdout lines and
  durable JSONL event artifacts.
- **FR-038**: Heartbeat emission MUST be configurable by both elapsed-time
  interval and training-step interval, with emission occurring when either
  threshold is reached.
- **FR-039**: Slurm jobs MUST default to heartbeat lines rather than tqdm-style
  progress bars, while local interactive runs MAY enable tqdm-style progress
  output through configuration.
- **FR-040**: The pilot runner MUST make comparison scope explicit and MUST
  default to a comparison workflow rather than a single nested run. Smoke and
  debug invocations MAY run only one selected mode when requested.
- **FR-041**: The default pilot comparison workflow MUST include
  `nested-random`, `nested-all`, and independently trained standalone S, M, L,
  and XL pilot baselines where compute allows. Omitted baselines MUST be marked
  explicitly in comparison artifacts.
- **FR-042**: Configs, metrics, scaling rows, summaries, and documentation MUST
  use clear labels that distinguish `nested-random`, `nested-all`, and
  `standalone` runs.
- **FR-043**: Pilot comparison artifacts MUST expose actual parameter counts,
  model family, granularity, sampling mode, token budget, effective world size,
  checkpoint path when available, and known mismatch notes.
- **FR-044**: Pilot training runs MUST persist a usable best-eval checkpoint
  when validation is enabled. The checkpoint MUST be written only by rank 0
  under the run output directory during distributed/FSDP execution, and
  `run_summary.json` MUST reference its path. Best-eval MUST be defined by
  validation loss or perplexity when available. If validation is disabled, the
  run MUST save a final checkpoint or explicitly record that no best-eval
  checkpoint was produced.

### Research & Experiment Requirements *(include for experiment-facing changes)*

- **EX-001**: Each run MUST expose configuration values for model family,
  granularity, sampling mode, explicit shape fields, dataset phase, token
  budget, seed when set, run budget, and evaluation suite.
- **EX-002**: Each run MUST save the exact configuration used.
- **EX-003**: Each completed run MUST write scalar metrics to CSV or JSON, not
  only to terminal logs.
- **EX-004**: Each run MUST record dataset identity, dataset phase, and
  preprocessing assumptions needed to interpret results.
- **EX-005**: Reproduction outputs MUST include `metrics.csv`,
  `task_results.csv`, `scaling_results.csv`, and `consistency_results.csv` when
  the corresponding evaluations have run.
- **EX-006**: Plot artifacts MUST be reproducible from exported CSV files.
- **EX-007**: Checkpoints MUST be saved when they are needed to extract nested
  submodels, resume interrupted runs, inspect a reported comparison, or satisfy
  the pilot best-eval/final-checkpoint policy.
- **EX-008**: Training-efficiency reporting SHOULD include tokens per second,
  wall-clock time, and estimated compute cost when available.
- **EX-009**: Memory reporting SHOULD include peak device memory and enough
  detail to compare nested training overhead against standalone baselines.
- **EX-010**: Every comparison report MUST link each plotted point or aggregate
  metric back to the run configuration and metrics artifact that produced it.
- **EX-011**: Generated experiment artifacts, checkpoints, plots, and summaries
  MUST be written under the configured output root rather than requiring space
  or inodes on the repository filesystem.
- **EX-012**: Runtime heartbeat events MUST include stage, rank, world size,
  elapsed time, and, when meaningful, step, derived max steps, tokens seen,
  token budget, latest loss, tokens per second, peak GPU memory, and ETA.
- **EX-013**: Heartbeat JSONL artifacts MUST be written under the configured
  output root and be sufficient for postmortem analysis without requiring
  terminal logs.
- **EX-014**: Default heartbeat cadence SHOULD be 10 training steps or 60
  seconds for training, whichever comes first, and 60 seconds for non-step
  preprocessing-style stages.
- **EX-015**: Resolved configs, run summaries, scaling rows, and comparison
  artifacts MUST distinguish actual implementation parameter counts from any
  MatLM paper table reference counts.
- **EX-016**: Comparison artifacts MUST expose mismatch notes for architecture,
  token budget, parameter-count convention, dataset, tokenizer assumption,
  effective world size, and checkpoint availability when those fields differ or
  are unavailable.

### Key Entities *(include if feature involves data)*

- **Reproduction Phase**: A planned scale of work such as small validation,
  medium trend reproduction, or large scaling reproduction, with datasets,
  expected outputs, and completion criteria.
- **Model Family**: Either MatFormer-Llama/SwiGLU nested or standalone
  baseline, used to group training runs and comparisons.
- **Granularity**: One of S, M, L, or XL, identifying the FFN expansion ratio
  and the corresponding nested prefix.
- **Sampling Mode**: The training or comparison mode label, such as
  `nested-random`, `nested-all`, or `standalone`.
- **Parameter Count Report**: A structured set of actual implementation counts
  including total, embedding, language-model head, non-embedding, FFN, and
  feasible attention or other non-embedding parameters, plus any paper table
  reference counts and mismatch notes.
- **Training Run**: A single execution with a model family, granularity or
  granularity set, sampling mode, dataset phase, token budget, configuration,
  metrics, parameter count report, and optional checkpoint. Budgeted runs also
  record expected tokens per step, derived max steps, effective world size,
  tokens seen, and stop reason.
- **Baseline Match**: The pairing between an extracted nested submodel and an
  independently trained standalone model of comparable non-embedding parameter
  count.
- **Evaluation Suite**: A named set of validation, downstream, consistency,
  mix-and-match, or speculative decoding evaluations.
- **Metrics Artifact**: A structured output file containing scalar metrics,
  per-task results, scaling summaries, consistency summaries, or efficiency
  statistics.
- **Figure Artifact**: A plot or report generated from structured metrics,
  especially Figure 2-style scaling and comparison outputs.
- **Checkpoint Artifact**: A saved model state under the run output directory,
  either a best-eval checkpoint selected by validation loss or perplexity, a
  final checkpoint, or an explicit no-checkpoint status in the run summary.
- **Distributed Execution Context**: The active rank, local rank, world size,
  device assignment, and distributed strategy for a config-driven run.
- **Heartbeat Event**: A structured runtime event written to stdout and JSONL
  that records pipeline stage progress, timing, rank/world-size context, and
  available training or memory measurements.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A small-scale proof-of-concept run evaluates all four MatFormer
  granularities, produces saved configuration, validation loss, perplexity, and
  extraction metadata for each granularity, and includes at least one matched
  standalone baseline comparison.
- **SC-002**: For every reported nested granularity comparison, a matched
  standalone baseline exists or the report explicitly marks the baseline as
  missing.
- **SC-003**: Before the d_model=256 pilot comparison is treated as complete,
  the debug-size matrix includes S, M, L, and XL nested and standalone baseline
  comparisons.
- **SC-004**: Each completed training or evaluation run produces structured
  metrics and configuration artifacts, with zero required metrics stored only in
  terminal logs.
- **SC-005**: Figure 2-style reporting includes at least loss or perplexity
  versus non-embedding parameters for both nested submodels and standalone
  baselines once P2 is complete.
- **SC-006**: The d_model=256 MatFormer-Llama/SwiGLU pilot records explicit
  shape fields, the actual training-token budget used, and whether the run is a
  reduced-token pilot or uses the MatLM table-row 10B-token budget reference.
  The resolved config or run summary also records derived max steps, expected
  tokens per step, effective world size, actual tokens seen, and stop reason.
- **SC-007**: Pilot and scaling reports avoid exact MatLM alignment claims
  unless true, and record mismatch notes for known differences in architecture,
  parameter counting, token budget, or training behavior.
- **SC-008**: Medium-scale reporting includes per-task and average downstream
  results for at least six representative tasks when P3 is complete.
- **SC-009**: Consistency reporting includes at least one alignment metric for
  at least one smaller/larger nested granularity pair when P4 is complete.
- **SC-010**: Speculative decoding reporting includes acceptance rate, rollback
  frequency, throughput, and latency for both nested and standalone
  draft/verifier comparisons when P5 is complete.
- **SC-011**: Every final phase report states the completed reproduction phase,
  dataset coverage, model-size coverage, and whether the evidence supports,
  weakens, or is inconclusive for each central paper claim.
- **SC-012**: A run launched with a custom output root writes `config.json`,
  metrics, summaries, checkpoints when enabled, and generated plots under that
  root with no required run artifact written under the repository `outputs/`
  directory.
- **SC-013**: Before Phase 5 begins, a short single-node multi-GPU Slurm smoke
  run of the d_model=256 pilot completes with FSDP enabled, records the
  effective world size, and writes shared artifacts only once.
- **SC-014**: A long-running Slurm pilot emits heartbeat stdout lines and a
  heartbeat JSONL artifact that identify the active stage, rank/world size,
  elapsed time, and available training progress without requiring tqdm output.
- **SC-015**: Pilot resolved configs, run summaries, scaling rows, and
  comparison artifacts report `total_parameters`, `embedding_parameters`,
  `lm_head_parameters`, `non_embedding_parameters`, and `ffn_parameters`, plus
  `attention_parameters` and `other_non_embedding_parameters` when feasible,
  and state the language-model-head counting convention.
- **SC-016**: A pilot run with validation enabled writes a rank-0-safe best-eval
  checkpoint under the run output directory and records the checkpoint path in
  `run_summary.json`; a pilot run without validation records final-checkpoint
  or no-checkpoint status.
- **SC-017**: The default pilot comparison workflow produces or explicitly
  marks missing `nested-random`, `nested-all`, and standalone S, M, L, and XL
  baseline rows with model family, granularity, sampling mode, token budget,
  effective world size, checkpoint path when available, and mismatch notes.

## Assumptions

- The planned execution order is P1 debug-size nested validation with S, M, L,
  and XL matched standalone baseline comparisons, P2 d_model=256
  MatFormer-Llama/SwiGLU pilot comparison, Phase 4.6 distributed pilot
  execution and runtime observability, Phase 4.7 pilot terminology,
  checkpointing, and comparison scope, P3 scaling and downstream trends, P4
  consistency and mix-and-match evaluation, then P5 speculative decoding.
- TinyStories and Tiny Shakespeare are appropriate small-scale validation
  datasets for early debugging and extraction checks.
- FineWeb and SlimPajama subsets are appropriate medium-scale public datasets
  for trend reproduction.
- FineWeb, SlimPajama, and C4 are appropriate large-scale public dataset
  candidates when resources allow.
- The minimal downstream suite includes HellaSwag, PIQA, ARC-Challenge, BoolQ,
  WinoGrande, and OpenBookQA.
- Exact numerical replication of the original paper is out of scope because the
  original pretraining data is proprietary.
- MatLM table-row targets may be reduced or phased if available compute cannot
  cover the full set, and implementation artifacts must report actual
  parameter counts rather than relying on table-row labels alone.
- Debug-size runs may shrink architecture constants for speed; pilot and
  scaling runs preserve explicit shape fields and record any deviations from
  paper table assumptions as mismatch notes.
- The MatLM table-row 10B-token budget is a reference budget for the
  d_model=256 pilot; smaller token budgets are reduced-token pilots.
- A standard public language-model evaluation runner can be selected during
  planning to improve comparability, but the specification does not require a
  particular tool.
- The default output root is `outputs/`, but researchers may redirect outputs
  to a larger filesystem for runs that would exceed repository filesystem space
  or inode limits.
