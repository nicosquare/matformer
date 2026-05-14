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
  debug-size S/M/L/XL nested and standalone matrix first, then treat 78M as the
  first paper-aligned scaling point.
- Q: What is the architecture fidelity boundary? -> A: Debug runs may reduce
  architecture scale, but paper-aligned runs preserve 16 layers, 16 heads,
  context length 1024, and the 256k vocabulary assumption.
- Q: How should 78M training-token completion be labeled? -> A: Distinguish
  78M reduced-token pilot runs from 78M/10B paper-budget complete runs.
- Q: How should restricted repository storage be handled? -> A: All experiment
  outputs must support a configurable output root outside the repository,
  defaulting to `outputs/`, with runner support through `OUTPUT_ROOT` or
  command arguments.

### Session 2026-05-14

- Q: How should budgeted training length be derived? -> A: `training.token_budget`
  is authoritative; derive planned steps from batch size, context length, and
  effective distributed `WORLD_SIZE`, defaulting to 1 and never inferred from
  available GPU count alone.

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

**Independent Test**: For the debug-size matrix and first paper-aligned 78M
scaling point, confirm S, M, L, and XL each have a matching standalone run with
the same model family, tokenizer assumptions, dataset phase, optimizer
assumptions, and training-token budget except for FFN width.

**Acceptance Scenarios**:

1. **Given** a MatFormer run for the four granularities, **When** standalone
   baselines are prepared, **Then** each granularity has a matched independently
   trained baseline.
2. **Given** the debug-size matrix has completed nested training, **When** the
   researcher moves toward the first paper-aligned scaling point, **Then**
   standalone S, M, L, and XL baselines exist for the debug size.
3. **Given** completed nested and standalone runs, **When** the researcher
   compares validation metrics, **Then** the comparison is grouped by
   granularity and non-embedding parameter count.
4. **Given** an unmatched baseline configuration, **When** the researcher
   reviews the comparison, **Then** the mismatch is visible in the recorded
   configuration or run summary.

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
  paper-aligned runs must preserve the stated architecture assumptions.
- 78M runs with fewer than 10B training tokens must be labeled as reduced-token
  pilots, not paper-budget complete runs.
- Available GPU count can differ from the effective data-parallel world size;
  budget-derived step counts must use the active distributed `WORLD_SIZE` when
  distributed training is launched, otherwise 1.
- A standalone baseline can be accidentally mismatched by dataset, token
  budget, tokenizer assumption, or FFN width; comparison artifacts must expose
  those run attributes.
- Very small datasets can produce unstable perplexity or downstream metrics;
  reports must identify dataset scale and avoid overclaiming.
- Embedding parameters can dominate small models; model-size plots must use
  non-embedding parameter counts when reproducing Figure 2-style results.
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
- **FR-004**: The nested training objective MUST evaluate all configured
  granularities for each training batch and combine their language-model losses
  into a single training objective.
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
- **FR-011**: Before paper-aligned scaling work, the reproduction MUST complete
  a debug-size matrix with S, M, L, and XL nested and matched standalone
  baselines.
- **FR-012**: Medium-scale reproduction MUST compare nested and standalone
  behavior on validation metrics and a representative downstream evaluation
  suite.
- **FR-013**: Large-scale reproduction MUST target scaling-curve behavior,
  Figure 2-style reporting, and comparison against standalone baselines when
  resources allow.
- **FR-014**: The first paper-aligned scaling point MUST be the 78M model-size
  target with its training-token budget and completion label tracked explicitly.
  For budgeted training runs, `training.token_budget` MUST be the source of
  truth for planned training length. The runner MUST derive the planned step
  count from `training.token_budget`, `training.batch_size_per_process`,
  `model.context_length`, and the effective data-parallel world size. The
  effective world size MUST be the active distributed `WORLD_SIZE` when
  distributed training is launched, otherwise 1; it MUST NOT be inferred from
  available GPU count alone. Resolved configs and run summaries MUST record
  `expected_tokens_per_step`, `derived_max_steps`, `token_budget`,
  `tokens_seen`, `effective_world_size`, and `stop_reason`.
- **FR-015**: Debug-size runs MAY reduce architecture scale for speed while
  preserving matched nested-versus-standalone comparisons.
- **FR-016**: Paper-aligned runs MUST preserve 16 layers, 16 attention heads,
  context length 1024, and the 256k vocabulary-size assumption unless the report
  explicitly marks the run as non-paper-aligned.
- **FR-017**: Figure 2-style reporting MUST plot loss, perplexity, downstream
  accuracy, and consistency against non-embedding parameters.
- **FR-018**: Non-embedding parameter counts MUST exclude token embeddings and
  output embeddings or language-model-head parameters.
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
- **FR-028**: Reports MUST distinguish 78M reduced-token pilot runs from 78M/10B
  paper-budget complete runs.
- **FR-029**: Experiment workflows MUST support a configurable output root for
  all generated run artifacts, defaulting to `outputs/` when the researcher does
  not provide one.
- **FR-030**: Matrix and single-run workflows MUST resolve run artifacts under
  `<output_root>/<run_id>` unless an explicit per-run output directory is
  intentionally provided.
- **FR-031**: Researcher-facing runner commands MUST allow the output root to be
  set through configuration, command arguments, or an `OUTPUT_ROOT` environment
  variable so artifacts can be written outside the repository filesystem.

### Research & Experiment Requirements *(include for experiment-facing changes)*

- **EX-001**: Each run MUST expose configuration values for model size,
  granularity, dataset phase, token budget, seed when set, run budget, and
  evaluation suite.
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
  submodels, resume interrupted runs, or inspect a reported comparison.
- **EX-008**: Training-efficiency reporting SHOULD include tokens per second,
  wall-clock time, and estimated compute cost when available.
- **EX-009**: Memory reporting SHOULD include peak device memory and enough
  detail to compare nested training overhead against standalone baselines.
- **EX-010**: Every comparison report MUST link each plotted point or aggregate
  metric back to the run configuration and metrics artifact that produced it.
- **EX-011**: Generated experiment artifacts, checkpoints, plots, and summaries
  MUST be written under the configured output root rather than requiring space
  or inodes on the repository filesystem.

### Key Entities *(include if feature involves data)*

- **Reproduction Phase**: A planned scale of work such as small validation,
  medium trend reproduction, or large scaling reproduction, with datasets,
  expected outputs, and completion criteria.
- **Model Family**: Either MatFormer nested or standalone baseline, used to
  group training runs and comparisons.
- **Granularity**: One of S, M, L, or XL, identifying the FFN expansion ratio
  and the corresponding nested prefix.
- **Training Run**: A single execution with a model family, granularity or
  granularity set, dataset phase, token budget, configuration, metrics, and
  optional checkpoint. Budgeted runs also record expected tokens per step,
  derived max steps, effective world size, tokens seen, and stop reason.
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

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A small-scale proof-of-concept run evaluates all four MatFormer
  granularities, produces saved configuration, validation loss, perplexity, and
  extraction metadata for each granularity, and includes at least one matched
  standalone baseline comparison.
- **SC-002**: For every reported nested granularity comparison, a matched
  standalone baseline exists or the report explicitly marks the baseline as
  missing.
- **SC-003**: Before any paper-aligned scaling report is treated as complete,
  the debug-size matrix includes S, M, L, and XL nested and standalone baseline
  comparisons.
- **SC-004**: Each completed training or evaluation run produces structured
  metrics and configuration artifacts, with zero required metrics stored only in
  terminal logs.
- **SC-005**: Figure 2-style reporting includes at least loss or perplexity
  versus non-embedding parameters for both nested submodels and standalone
  baselines once P2 is complete.
- **SC-006**: The first paper-aligned scaling point records the 78M model-size
  target, the actual training-token budget used, and whether the run is a
  reduced-token pilot or a 78M/10B paper-budget complete run. The resolved
  config or run summary also records derived max steps, expected tokens per
  step, effective world size, actual tokens seen, and stop reason.
- **SC-007**: Any report labeled paper-aligned uses the stated architecture
  assumptions or explicitly marks deviations as non-paper-aligned.
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

## Assumptions

- The planned execution order is P1 debug-size nested validation with S, M, L,
  and XL matched standalone baseline comparisons, P2 first paper-aligned 78M
  scaling point, P3 scaling and downstream trends, P4 consistency and
  mix-and-match evaluation, then P5 speculative decoding.
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
- Model-size targets may be reduced or phased if available compute cannot cover
  the full 78M, 180M, 310M, 463M, and 850M set.
- Debug-size runs may shrink architecture constants for speed; paper-aligned
  runs preserve the paper's shared architecture assumptions.
- A 78M/10B label means the 78M model-size target trained with the 10B
  training-token budget; smaller token budgets are reduced-token pilots.
- A standard public language-model evaluation runner can be selected during
  planning to improve comparability, but the specification does not require a
  particular tool.
- The default output root is `outputs/`, but researchers may redirect outputs
  to a larger filesystem for runs that would exceed repository filesystem space
  or inode limits.
