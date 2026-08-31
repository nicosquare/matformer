# Feature Specification: Per-Width Optimizer State

**Feature Branch**: `012-per-width-optimizer-state`  
**Created**: 2026-08-31  
**Status**: Draft  
**Input**: User description: `docs/tinystories-per-width-optimizer-experiment.md`, using its designated full feature prompt in `notes/tinystories_per_width_optimizer_speckit_prompt.md`

## Clarifications

### Session 2026-08-31

- Q: Is distributed multi-process execution part of the initial per-granularity optimizer-state feature? → A: Single-process only; reject per-granularity state in distributed runs.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Isolate optimizer histories by width (Priority: P1)

As a researcher, I can train one shared nested model while giving each resolved global width an independent optimizer history, so I can test whether cross-width sharing of optimizer state helps or harms learning.

**Why this priority**: Independent optimizer histories over shared weights are the scientific intervention. Without correct state isolation, the comparison does not test the stated hypothesis.

**Independent Test**: Alternate two widths whose gradients differ on a shared parameter, then verify that both widths update the same model weights while retaining distinct optimizer moments and update ages.

**Acceptance Scenarios**:

1. **Given** two global widths that produce different gradients on a shared parameter, **When** each width completes an update, **Then** the model parameter remains shared and each width retains its own optimizer history for that parameter.
2. **Given** a narrow width that does not use wider-only parameters, **When** the narrow width completes an update, **Then** the wider-only parameters and all state associated with non-selected optimizers remain bitwise unchanged.
3. **Given** either supported optimizer family, **When** widths alternate over several successful updates, **Then** exactly the selected width's optimizer advances on each update and ordinary optimizer behavior is preserved.
4. **Given** accumulated gradients spanning multiple microbatches, **When** an update window begins, **Then** one global width owns the complete window and exactly its optimizer owns the resulting commit.
5. **Given** an attempted update that fails before commit, **When** training recovers or terminates, **Then** no optimizer state or successful-update count advances.

---

### User Story 2 - Run a controlled shared-versus-per-width comparison (Priority: P2)

As a researcher, I can choose shared or per-granularity optimizer state while holding the model, data, sampled widths, learning-rate schedule, budget, and evaluation protocol fixed, so differences can be attributed to optimizer-state scope.

**Why this priority**: A scientifically useful result requires paired arms that differ only in the intervention and expose the resource cost of that intervention.

**Independent Test**: Execute short paired runs from the same initial state and compare their resolved experiment identities, width-action traces, batches, learning rates, budgets, and recorded resource measurements.

**Acceptance Scenarios**:

1. **Given** matched shared and per-granularity runs, **When** the same global step is reached, **Then** both use the same selected width, training batch, learning rate, evaluation cadence, and shared model initialization.
2. **Given** a per-granularity run whose widths have unequal prior exposure, **When** the next global step begins, **Then** every width-specific optimizer receives the learning rate assigned to that global step rather than a width-local schedule position.
3. **Given** the dedicated four-width balanced-cycle recipe, **When** a complete four-step cycle finishes, **Then** each width has exactly one committed update.
4. **Given** a completed paired run, **When** researchers inspect its results, **Then** wall time, peak accelerator memory, checkpoint size, optimizer-state scope, and per-width update counts are available alongside performance outcomes.
5. **Given** a one-seed or pilot result, **When** a report is produced, **Then** it is identified as diagnostic and is not presented as evidence of general superiority.

---

### User Story 3 - Configure only valid ownership modes (Priority: P3)

As an experiment operator, I can select optimizer-state scope through normal experiment configuration and receive an early, specific rejection when the training mode cannot assign exactly one optimizer owner to each update.

**Why this priority**: Silent ambiguity about which optimizer owns an update would invalidate state isolation and make checkpoints unsafe to resume.

**Independent Test**: Preflight valid single-global-width modes and each prohibited topology, then verify that valid combinations resolve consistently and invalid combinations stop before training changes state.

**Acceptance Scenarios**:

1. **Given** deterministic, stochastic, or adaptive global sampling that selects exactly one of at least two widths per update, **When** per-granularity state is selected, **Then** preflight succeeds and identifies the ordered width set.
2. **Given** standalone, nested-all, per-block, adaptive-per-block, distributed multi-process, zero-width, multi-width, or single-unique-width ownership, **When** per-granularity state is selected, **Then** preflight rejects the run before training.
3. **Given** a warmup mode that applies multiple widths in one optimizer step, **When** per-granularity state is selected, **Then** the run is rejected because no single optimizer owner exists.
4. **Given** an existing configuration with no optimizer-state scope, **When** it is resolved, **Then** it uses shared state and retains historical behavior and artifact interpretation.
5. **Given** an unknown state scope or a non-global scheduler clock, **When** configuration is resolved, **Then** the run fails before training with the incompatible field identified.

---

### User Story 4 - Resume and audit exact optimizer ownership (Priority: P4)

As a researcher, I can resume a per-granularity run and audit which optimizer advanced at every committed step, so interrupted experiments remain comparable to uninterrupted experiments.

**Why this priority**: The additional state collection is useful only if it is complete, validated, and reconciled with the width-sampling history.

**Independent Test**: Compare uninterrupted execution with resumes inside and exactly at a balanced-cycle boundary, and exercise every required checkpoint mismatch.

**Acceptance Scenarios**:

1. **Given** a compatible resumable checkpoint captured inside a sampling interval, **When** the run resumes, **Then** subsequent width actions, learning rates, optimizer histories, metrics, and final model parameters match uninterrupted execution within existing reproducibility limits.
2. **Given** a compatible checkpoint at a balanced-cycle boundary, **When** the run resumes, **Then** the next action and optimizer update occur exactly once without skipping or duplicating the boundary transition.
3. **Given** missing, extra, reordered, malformed, or mismatched optimizer state, **When** resume is requested, **Then** resume fails rather than initializing, merging, or remapping state silently.
4. **Given** a checkpoint whose state scope, optimizer contract, ordered widths, model topology, or scheduler contract differs from the run, **When** resume is requested, **Then** resume fails before any training update.
5. **Given** a completed run, **When** its accounting is reconciled, **Then** per-width optimizer updates sum to committed global optimizer steps and exactly match persisted width exposures.
6. **Given** the normal training entrypoint and a small processor-only configuration, **When** a short per-granularity run completes, **Then** it produces a resumable checkpoint and auditable training artifacts without using a separate execution path.

### Edge Cases

- A resolved configuration contains duplicate width labels or the same effective granularity more than once; validation treats the effective ordered set as invalid rather than creating ambiguous optimizer ownership.
- A parameter is shared by multiple widths but first receives a gradient late in training; only the selected optimizer may create or advance its state on that step.
- A wider-only parameter is known to a narrow width's optimizer but has no gradient; it must not acquire state, decay, or advance through the narrow-width update.
- A gradient exists for an inactive parameter because of stale state from a prior step; the run must stop or clear the stale gradient before it can affect another optimizer.
- A failure occurs after backward computation but before the optimizer commit; optimizer states, global schedule position, width exposures, and successful-update counters remain unchanged.
- Gradient clipping or mixed-precision overflow causes an update to be skipped; the event is not counted as a committed optimizer update.
- A resumable checkpoint contains the right number of optimizer states but associates them with reordered width labels; resume is rejected.
- An historical artifact omits optimizer-state scope; it is interpreted as shared, but cannot be resumed as per-granularity.
- A best-evaluation checkpoint intentionally contains only model state; it may be used for evaluation but must not be advertised or accepted as resumable.
- The pilot final holdout is opened before a possible confirmation; any later result on the same holdout must be labeled descriptive rather than fresh confirmation.
- An incomplete balanced cycle ends at the fixed budget; the recorded exposure counts reflect committed steps without adding synthetic updates to complete the cycle.

## Requirements *(mandatory)*

### Functional Requirements

#### State scope and update ownership

- **FR-001**: Researchers MUST be able to select one of two explicit optimizer-state scopes: shared state for all widths or independent state per resolved global granularity.
- **FR-002**: Shared state MUST remain the default and MUST preserve the behavior of configurations and historical artifacts that omit state scope.
- **FR-003**: Per-granularity state MUST maintain one independent optimizer state machine for every unique member of the ordered resolved global-granularity set.
- **FR-004**: All optimizer state machines MUST operate on the same shared model parameters; the feature MUST NOT create separate model weights, parameter replicas, or permanently disjoint parameter ownership.
- **FR-005**: Each optimizer commit in per-granularity mode MUST have exactly one selected global granularity and exactly one owning optimizer.
- **FR-006**: Only the selected optimizer MUST perform the commit, and its internal state MUST advance only for parameters that receive gradients from the selected width.
- **FR-007**: Non-selected optimizers, including all their moments, counters, and parameter-specific state, MUST remain bitwise unchanged across a successful commit.
- **FR-008**: Parameters inactive for the selected width MUST receive no update, weight decay, state creation, or state advancement.
- **FR-009**: Gradients from a completed or failed attempt MUST NOT leak into a later update or another optimizer's state.
- **FR-010**: Per-granularity mode MUST support the repository's existing AdamW and SGD families while preserving configured hyperparameters, weight decay, gradient clipping, mixed-precision behavior, and ordinary update semantics.
- **FR-011**: An accumulated-gradient window MUST use one selected global width for all contributing microbatches and MUST commit through that width's optimizer only.
- **FR-012**: Any attempt that does not produce a successful optimizer commit MUST leave optimizer states and successful-update accounting unchanged.

#### Global schedule and eligible training modes

- **FR-013**: Both state scopes MUST use one global optimizer-step scheduler clock.
- **FR-014**: At every matched global step, the learning rate MUST be identical between shared and per-granularity arms when all other experiment inputs match.
- **FR-015**: Every width-specific optimizer MUST use the learning rate assigned to the current committed global step regardless of its own prior selection count.
- **FR-016**: Optimizer moments, bias-correction ages, and other optimizer-internal counters MUST remain width-specific in per-granularity mode; learning-rate warmup and decay progress MUST remain global.
- **FR-017**: Per-granularity state MUST be accepted only when each optimizer step resolves to exactly one global granularity from at least two unique resolved granularities.
- **FR-018**: Valid ownership MUST include deterministic, stochastic, and adaptive global-width policies that select one complete global width per update.
- **FR-019**: Per-granularity state MUST be rejected before training for standalone, nested-all, per-block, adaptive-per-block, distributed multi-process, empty-action, multi-width-action, and fewer-than-two-unique-width configurations.
- **FR-020**: Warmup behavior that applies more than one width in a single optimizer step MUST be rejected with per-granularity state.
- **FR-021**: Unknown state scopes, non-global scheduler clocks, and incompatible scope combinations MUST fail during preflight with an attributable error.
- **FR-022**: Selecting optimizer-state scope MUST NOT change width-sampling probabilities, model mathematics, correction behavior, data roles, or evaluation cadence.

#### Configuration, identity, and compatibility

- **FR-023**: Researchers MUST be able to select state scope and the fixed global scheduler meaning through normal experiment configuration and command-line overrides without source changes.
- **FR-024**: Resolved configuration, monitoring identity, training metrics, checkpoints, run summaries, and comparison inputs MUST distinguish shared from per-granularity state.
- **FR-025**: Historical artifacts without state-scope identity MUST be interpreted as shared state.
- **FR-026**: Existing runs that omit state scope MUST remain behaviorally unchanged, including their action sequence and ordinary artifact meaning.
- **FR-027**: The implementation MUST expose enough resolved identity during preflight to verify state scope, ordered widths, optimizer contract, scheduler contract, data roles, run budget, and sampling policy before submission.

#### Checkpoint and exact resume

- **FR-028**: Every checkpoint advertised as resumable in per-granularity mode MUST preserve the ordered granularity labels, one complete optimizer state per granularity, global scheduler state and clock position, per-granularity successful-update counts, explicit versioned state-scope identity, and any active-optimizer identity required for exact continuation.
- **FR-029**: Model-only evaluation checkpoints MAY remain model-only but MUST NOT be accepted or advertised as resumable training checkpoints.
- **FR-030**: Resume MUST reject a state-scope mismatch in either direction; shared state MUST NOT be silently expanded into per-granularity state and per-granularity state MUST NOT be merged into shared state.
- **FR-031**: Resume MUST reject mismatches in optimizer family or hyperparameters, global scheduler contract, ordered widths, model topology, state version, or any required optimizer state.
- **FR-032**: Resume MUST reject missing, extra, reordered, malformed, or non-finite optimizer states rather than reinitializing or remapping them silently.
- **FR-033**: Under the repository's documented deterministic conditions, a matching resume inside or at a width-sampling boundary MUST reproduce subsequent width actions, parameter updates, learning rates, optimizer states, metrics, and final parameters within existing reproducibility limits.

#### Metrics, accounting, and reporting

- **FR-034**: Every ordinary training record MUST identify the selected global width and resolved optimizer-state scope.
- **FR-035**: Per-granularity runs MUST maintain compact successful-update counts for every ordered width in structured run artifacts.
- **FR-036**: At every completed run, per-width optimizer-update counts MUST sum exactly to committed global optimizer steps and MUST match persisted global-width exposure counts.
- **FR-037**: Run artifacts MUST make it possible to verify that non-selected optimizers did not advance and that all optimizer instances followed the same global scheduler position.
- **FR-038**: Every comparison result MUST report training wall time, peak accelerator memory, and resumable-checkpoint size so the cost of additional optimizer state is visible.
- **FR-039**: Reports MUST NOT describe arms as matched-compute unless the recorded measurements support that claim.
- **FR-040**: Failures and skipped commits MUST be recorded distinctly enough to reconcile attempted steps, committed steps, scheduler position, optimizer updates, and width exposures.

### Research & Experiment Requirements

- **EX-001**: The feature MUST provide a dedicated opt-in TinyStories-Instruct paired-comparison recipe with four global widths, balanced-cycle sampling that selects one width per step, shared model weights, no correction, and strict reproducibility.
- **EX-002**: The recipe MUST fix the model at hidden size 64, four transformer layers, four attention heads, context length 128, vocabulary size 2048, and widths `g250`, `g500`, `g750`, and `g1000`.
- **EX-003**: The recipe MUST fix AdamW to learning rate 0.008, betas 0.9 and 0.95, epsilon 0.00000001, weight decay 0.1, cosine scheduling, and 64 global warmup steps.
- **EX-004**: The recipe MUST use batches of 64 sequences, one process, no gradient accumulation, fixed prepared TinyStories-Instruct data roles, and seeds 42, 43, and 44.
- **EX-005**: The pilot MUST use a fixed budget of 713,785,344 optimizer tokens, equivalent to 87,132 global steps and exactly 21,783 committed updates per width.
- **EX-006**: The optional confirmation MUST start fresh and use 2,141,356,032 optimizer tokens, equivalent to 261,396 global steps and exactly 65,349 committed updates per width.
- **EX-007**: For each seed and budget, the paired arms MUST have identical model initialization, optimizer-training batches, data-role manifests, width-action sequence, global learning rates, evaluation cadence, number of steps, and token budget; state scope MUST be the only intended difference.
- **EX-008**: The primary endpoint MUST be the paired seed-level difference in uniform mean final-holdout loss across all four widths at the terminal fixed-budget checkpoint.
- **EX-009**: Secondary endpoints MUST include per-width final-holdout loss and perplexity, worst-width final-holdout loss, trailing-five ordinary-validation mean per width, matched-token convergence trajectories, per-width optimizer updates, wall time, peak memory, and checkpoint size.
- **EX-010**: Pilot trajectory and stability decisions MUST use ordinary validation; the final holdout MUST remain sealed through the pilot whenever a fresh three-epoch confirmation may be performed.
- **EX-011**: Before the final holdout is opened, researchers MUST freeze the arms, seeds, budgets, terminal-checkpoint rule, primary endpoint, secondary endpoints, and analysis manifest.
- **EX-012**: Pilot or single-seed findings MUST be labeled diagnostic; a confirmatory claim MUST require all predefined seeds, the frozen confirmation protocol, and a holdout not inspected during the pilot.
- **EX-013**: Each pilot and confirmation set MUST save the resolved run identities, terminal checkpoint paths and hashes, seeds, scopes, budgets, endpoint definitions, and terminal-checkpoint selection rule.

### Key Entities

- **Optimizer-state scope**: The experiment choice between one optimizer history shared by all widths and one independent history for every resolved global granularity.
- **Resolved granularity set**: The ordered, unique global-width labels eligible for selection; order is part of configuration and checkpoint identity.
- **Width-specific optimizer state**: The moments, counters, parameter-specific state, and successful-update count belonging to one width while operating on shared model parameters.
- **Global scheduler state**: The single warmup and decay position advanced by committed global optimizer steps and applied consistently to every optimizer.
- **Optimizer commit**: One successful shared-parameter update owned by exactly one selected width; failed or skipped attempts are not commits.
- **Resumable training checkpoint**: A versioned snapshot containing the complete optimizer collection, global scheduler state, ordered-width identity, counters, and continuation state.
- **Paired experiment arm**: One shared-scope or per-granularity run matched to its counterpart by seed, initialization, data, actions, schedule, budget, and evaluation protocol.
- **Comparison manifest**: The frozen record of paired runs, checkpoints, hashes, budgets, endpoints, and holdout-use status used to audit the scientific comparison.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In 100% of focused per-granularity tests, exactly one optimizer advances per committed global step and every non-selected optimizer remains bitwise unchanged.
- **SC-002**: In 100% of narrow-width tests, wider-only parameters and their optimizer state remain unchanged when they receive no gradient.
- **SC-003**: Across both supported optimizer families, alternating-width tests retain distinct histories for shared parameters while all widths continue to update one shared parameter set.
- **SC-004**: In matched-arm tests, 100% of compared steps use identical width actions, training batches, global learning rates, and evaluation cadence.
- **SC-005**: In every completed run, the sum of per-width successful updates equals the committed global-step count exactly and every per-width count equals its persisted exposure count.
- **SC-006**: Under documented deterministic conditions, uninterrupted and resumed runs match all subsequent width actions exactly and match optimizer state, scheduler state, metrics, and final model parameters within the repository's established reproducibility tolerance.
- **SC-007**: All invalid topology, action, state-scope, scheduler, and checkpoint combinations covered by this specification fail before silently changing model or optimizer state.
- **SC-008**: Existing configurations that omit state scope resolve to shared state, and 100% of affected compatibility tests retain their prior expected behavior and artifact interpretation.
- **SC-009**: A short processor-only end-to-end run completes through the normal training entrypoint and produces a valid resumable checkpoint, selected-width records, state-scope identity, and reconciled update counts.
- **SC-010**: All six one-epoch paired pilot runs preflight with exactly four widths, 87,132 steps, 21,783 updates per width, their requested state scope, one global schedule clock, and no unintended paired-arm differences.
- **SC-011**: Every completed research run provides sufficient structured provenance for an independent reviewer to identify the selected scope, reproduce the resolved protocol, reconcile optimizer ownership, and distinguish diagnostic from confirmatory results without relying on terminal logs.
- **SC-012**: Every paired result table includes uniform mean loss, worst-width loss, all per-width outcomes, wall time, peak memory, and checkpoint size, so scientific benefit and operational cost can be assessed together.

## Assumptions

- The detailed prompt linked by the requested runbook is the authoritative feature description; the runbook remains the operational companion and will be updated during implementation if planning resolves different user-facing names.
- Existing global-width sampling already produces deterministic action sequences when run identity, seed, configuration, and runtime conditions match; this feature preserves that sampling behavior.
- Existing optimizer training, ordinary validation, controller data, and final holdout roles are fixed and disjoint for the controlled comparison.
- A committed global optimizer step is the authoritative unit for global scheduler progress, optimizer-update accounting, balanced-cycle exposure, and fixed-budget endpoints.
- AdamW and SGD use their existing repository-supported hyperparameter and mixed-precision contracts; different optimizer algorithms or hyperparameters by width are outside scope.
- The initial per-granularity optimizer-state feature is single-process only; distributed multi-process configurations are rejected before training rather than conditionally enabled during planning.
- Separate model weights, width-local schedules, nested-all ownership, per-block ownership, modified sampling probabilities, and changes to MatFormer mathematics or correction behavior are outside scope.
- Pilot and confirmation runs start from fresh, matched initial states; pilot checkpoints are never continued into confirmation because the global schedule horizon is part of the experimental contract.
- Existing documented reproducibility limits and tolerances apply to numerical comparisons; discrete width actions, ordered identities, and update counts are expected to match exactly under equivalent deterministic conditions.
