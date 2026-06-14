# Feature Specification: Granularity Operation Modes

**Feature Branch**: `006-granularity-sampling-modes`  
**Created**: 2026-06-11  
**Status**: Draft  
**Input**: User description: "Refactor the MatFormer training feature so the operation modes are coherent and explicitly specified. The feature must consolidate three top-level run modes: nested-random, nested-all, and standalone. nested-random is elastic training and supports two sampling submodes, global and per_block. nested-all evaluates all configured granularities on every iteration and optimizes the mean of the per-granularity losses. standalone trains the full token budget using a single fixed granularity for the entire run. The feature must also define correction rules clearly, keep a canonical resolved sampling mode in the model config, and record the resolved mode and runtime granularity pattern in run artifacts."

## Clarifications

### Session 2026-06-12

- Q: Should the canonical nested-random submode be named `per_block` or `per_layer`? → A: `per_block`

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train with explicit nested-random sampling (Priority: P1)

As a researcher, I can run elastic training in nested-random mode and choose whether sampling is global or per-block so that the current adaptive behavior remains available as a named, explicit mode.

**Why this priority**: This is the primary elastic-training path and the most important behavior to preserve while making the mode selection explicit.

**Independent Test**: Configure nested-random on a multi-block model, run one or more iterations, and verify that the selected submode controls whether one granularity is shared across the whole model or sampled separately per block.

**Acceptance Scenarios**:

1. **Given** nested-random with `global` selected, **When** an iteration starts, **Then** one granularity choice applies to every FFN in the model for that iteration.
2. **Given** nested-random with `per_block` selected, **When** an iteration starts, **Then** each transformer block receives its own granularity choice for that iteration.
3. **Given** nested-random with either sampling submode, **When** correction is computed, **Then** the correction context matches the selected pattern, with a shared correction scalar or pattern in `global` and a per-block correction derived from the sampled pattern in `per_block`, and the saved run metadata records the resolved mode, submode, and runtime pattern summary.

---

### User Story 2 - Evaluate all granularities together (Priority: P2)

As a researcher, I can run nested-all mode to compare all configured granularities on every iteration and optimize the mean of their losses.

**Why this priority**: This mode is required to support exhaustive nested evaluation without random submodel selection.

**Independent Test**: Configure nested-all with a set of granularities, run an iteration, and verify that every configured granularity is evaluated and that the reported training objective is the mean of those per-granularity losses.

**Acceptance Scenarios**:

1. **Given** nested-all with a configured set of granularities, **When** an iteration runs, **Then** every configured granularity is evaluated during that iteration.
2. **Given** nested-all evaluation of multiple granularities, **When** the training objective is computed, **Then** it equals the mean of the per-granularity losses for that iteration.
3. **Given** nested-all with correction enabled, **When** a particular granularity is evaluated, **Then** correction follows that uniform granularity across the model and does not introduce per-block random sampling.

---

### User Story 3 - Run a fixed standalone granularity and reconstruct it later (Priority: P3)

As a researcher or maintainer, I can run a fixed-granularity standalone experiment and later reconstruct which mode was used from saved artifacts without reading logs.

**Why this priority**: Standalone runs are needed for baseline comparisons, and traceable artifacts are required to make runs reproducible and auditable.

**Independent Test**: Configure standalone for each supported granularity, complete a run, and inspect the saved artifacts to confirm that the run stayed on one granularity and that the chosen mode can be recovered from metadata alone.

**Acceptance Scenarios**:

1. **Given** standalone mode with one of `s`, `m`, `l`, or `xl`, **When** the run executes, **Then** the chosen granularity remains fixed for the entire run.
2. **Given** a completed standalone run, **When** the saved artifacts are inspected, **Then** they identify the requested input, resolved mode, correction mode, and runtime granularity pattern without requiring console logs.
### Edge Cases

- A per-block run may coincidentally choose the same granularity for every block in one iteration; that is still per-block sampling because the choices were made independently.
- Nested-all must not perform per-block random sampling, even if the evaluated granularity is repeated across the model.
- Standalone must reject nested sampling submodes and must only accept the supported fixed granularities `s`, `m`, `l`, and `xl`.
- Invalid mode combinations must fail early with a clear error rather than falling back to an unintended behavior.
- The feature remains single-process only; distributed training is out of scope for this release.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST expose a canonical run mode with exactly three top-level values: `nested-random`, `nested-all`, and `standalone`.
- **FR-002**: The system MUST keep `nested-random` as the elastic-training path and MUST not treat it as a dense-model training mode.
- **FR-003**: The canonical configuration for `nested-random` MUST include an explicit sampling submode with exactly two values: `global` and `per_block`.
- **FR-004**: In `nested-random + global`, the system MUST choose one granularity once per dataloader iteration and apply that same granularity to every FFN in the model for that iteration.
- **FR-005**: In `nested-random + per_block`, the system MUST choose one granularity independently for each transformer block on every dataloader iteration.
- **FR-006**: In `nested-random + global`, correction MUST use one shared scalar or pattern for the whole iteration.
- **FR-007**: In `nested-random + per_block`, correction MUST be derived per block from the sampled per-block pattern.
- **FR-008**: The system MUST support `nested-all` by evaluating every configured granularity on every iteration and optimizing the mean of the per-granularity losses.
- **FR-009**: In `nested-all`, correction MUST follow the granularity selected for evaluation and MUST not introduce per-block random sampling.
- **FR-010**: The system MUST support `standalone` as a single fixed-granularity run for the entire training job.
- **FR-011**: `standalone` MUST accept only the supported fixed granularities `s`, `m`, `l`, and `xl`.
- **FR-012**: The system MUST preserve the correction modes `none`, `gmc`, and `lmc` where those modes are valid for the selected run mode, and MUST reject correction modes that are not valid for that run mode.
- **FR-013**: The system MUST use `slicing` and `concat` as the canonical public names for the two model variants.
- **FR-014**: The canonical model configuration MUST retain the resolved run mode and, when applicable, the resolved `nested-random` submode as separate explicit values.
- **FR-015**: Saved run artifacts MUST record the resolved canonical mode, the resolved `nested-random` submode when applicable, the correction mode, and the runtime granularity pattern summary.
- **FR-016**: The saved artifacts and run summary MUST make it possible to reconstruct which mode was used without reading console logs.
- **FR-017**: Invalid combinations MUST be rejected before training begins, including `nested-all` with per-block sampling, `standalone` with nested sampling submodes, unsupported standalone granularities, and any mode/correction pairing that is not valid.
- **FR-018**: The system MUST preserve the named `nested-random + global` configuration as the explicit whole-model path.
- **FR-019**: The resolved configuration and runtime pattern summary MUST make the difference between `nested-random + global` and `nested-random + per_block` observable.

### Key Entities

- **Run Mode**: The canonical top-level operating mode for a training run, with values `nested-random`, `nested-all`, and `standalone`.
- **Sampling Submode**: The nested-random-specific choice between `global` and `per_block`.
- **Granularity Pattern**: The effective granularity selection used during a run or iteration, represented as a shared choice, a per-block pattern, a per-evaluated-granularity summary, or a fixed standalone choice.
- **Correction Context**: The correction behavior paired with the selected mode and granularity pattern for a given iteration or evaluation.
- **Run Provenance Record**: The saved metadata needed to reconstruct the resolved mode, correction mode, and runtime pattern without logs.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In validated `nested-random + global` runs, 100% of checked iterations use exactly one granularity across all FFNs in the model.
- **SC-002**: In validated `nested-random + per_block` runs, 100% of saved pattern summaries contain one entry per transformer block and preserve a per-block pattern shape that is distinct from the single-value global summary format.
- **SC-003**: In validated `nested-all` runs, 100% of checked iterations evaluate every configured granularity and report an aggregate loss equal to the mean of the per-granularity losses within a tolerance of `1e-6`.
- **SC-004**: In validated `standalone` runs, all four supported granularities `s`, `m`, `l`, and `xl` complete successfully and remain fixed for the full run.
- **SC-005**: 100% of completed runs store the resolved mode, resolved submode when applicable, correction mode, and runtime granularity pattern summary in saved artifacts.
- **SC-006**: 100% of invalid mode combinations in the acceptance matrix fail before training begins and produce a clear rejection instead of silently falling back.
- **SC-007**: 100% of accepted configurations in the scope of this feature can be reconstructed from saved artifacts without consulting console logs.

## Assumptions

- The existing supported granularity set includes `s`, `m`, `l`, and `xl`, and `nested-all` evaluates the granularities configured for that run.
- Single-process execution remains the only supported runtime model for this feature release.
- Slicing and concat variants already exist in the codebase and are not redefined by this spec.
- The correction families `none`, `gmc`, and `lmc` remain the supported correction set where each one is valid today.
- Saved artifacts can be extended to carry the provenance needed to reconstruct a run without reading logs.
