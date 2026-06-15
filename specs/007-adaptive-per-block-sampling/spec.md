# Feature Specification: Adaptive Per-Block Sampling

**Feature Branch**: `007-adaptive-per-block-sampling`  
**Created**: 2026-06-15  
**Status**: Draft  
**Input**: User description: "Add a new additive granularity-sampling operation mode for MatFormer, aligned with notes/adaptive_per_block_proposal.md. Preserve the existing random nested-random + per_block path, and add nested-random + adaptive_per_block as a separate config-oriented mode that uses a stateful, nonstationary per-block sampler."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Keep the explicit nested-random modes stable (Priority: P1)

As a researcher, I can choose `nested-random + global`, `nested-random + per_block`, or `nested-random + adaptive_per_block` explicitly, and invalid mode combinations fail before training starts.

**Why this priority**: This preserves the existing baseline behavior while making the new mode surface safe and unambiguous.

**Independent Test**: Configure each supported nested-random combination and a representative set of invalid combinations, then verify that the valid ones resolve cleanly and the invalid ones are rejected before any training begins.

**Acceptance Scenarios**:

1. **Given** `nested-random + global`, **When** a run is resolved, **Then** it keeps the current whole-model random path and does not change the model wiring or correction family.
2. **Given** `nested-random + per_block`, **When** a run is resolved, **Then** it keeps the current random per-block baseline unchanged.
3. **Given** any unsupported mode pairing, **When** configuration is resolved, **Then** the run is rejected before training starts.

---

### User Story 2 - Learn adaptive per-block preferences over time (Priority: P2)

As a researcher, I can enable `nested-random + adaptive_per_block` to bias future per-block choices from recent training outcomes without replacing the existing model path.

**Why this priority**: This is the new experimental capability the feature adds.

**Independent Test**: Run the adaptive mode across multiple steps or phases and verify that the selected per-block pattern and recorded preferences can change over time, then resume from saved state and confirm the sampler continues from the prior state.

**Acceptance Scenarios**:

1. **Given** `nested-random + adaptive_per_block`, **When** a training step completes, **Then** one granularity is selected for each transformer block and the sampler updates its preferences from the observed reward signal.
2. **Given** the same adaptive run enters a later phase or a decayed-exploration period, **When** later steps are sampled, **Then** the sampled granularity preferences can shift rather than remaining fixed.
3. **Given** a resumed adaptive run, **When** training continues, **Then** it uses the saved sampler state instead of starting from an empty history.

---

### User Story 3 - Make runs interpretable and resumable from artifacts (Priority: P3)

As a maintainer, I can inspect saved artifacts and tell whether a run used `global`, random `per_block`, or `adaptive_per_block`, which strategy drove adaptive sampling, and what state is needed to resume training.

**Why this priority**: This keeps the feature auditable and makes comparisons between runs meaningful.

**Independent Test**: Inspect the saved artifacts from each supported mode and verify that the resolved mode, selected strategy, sampled pattern summary, and resumable state are visible without reading console logs.

**Acceptance Scenarios**:

1. **Given** a completed run in any supported mode, **When** its artifacts are inspected, **Then** the resolved mode, correction mode, and pattern summary are visible without consulting logs.
2. **Given** a completed `adaptive_per_block` run, **When** its artifacts are inspected, **Then** the selected sampler strategy, reward summary, correction-penalty summary, and resumable sampler state are present.
3. **Given** a saved random `per_block` run and a saved `adaptive_per_block` run, **When** their artifacts are compared, **Then** the difference between baseline random selection and stateful adaptive selection is observable from artifacts alone.

### Edge Cases

- A random `per_block` run may coincidentally choose the same granularity for every block in one step; it still counts as `per_block` if the choices were made independently.
- An `adaptive_per_block` run may temporarily converge on the same granularity across many blocks; it still counts as adaptive if the state and strategy are recorded.
- `nested-all` and `standalone` must continue to normalize to the global path, with `training.granularity_sampling` preserved as a compatibility alias for existing configs.
- The reward calculation must remain valid when correction is small or zero, as long as the correction penalty has been normalized into the same scale as the loss-improvement signal.
- A resumed adaptive run must either restore the saved sampler state or fail clearly if that state is missing or incompatible.
- `adaptive_per_block` is invalid outside `nested-random`.
- Unsupported strategy values must be rejected before training begins.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST expose `nested-random` with exactly three explicit submodes: `global`, `per_block`, and `adaptive_per_block`.
- **FR-002**: `nested-random + global` MUST preserve the current whole-model random path and MUST remain behaviorally unchanged apart from explicit configuration and provenance fields.
- **FR-003**: `nested-random + per_block` MUST preserve the current random per-block baseline and MUST remain behaviorally unchanged apart from explicit configuration and provenance fields.
- **FR-004**: `adaptive_per_block` MUST use a lightweight contextual bandit rather than full reinforcement learning.
- **FR-005**: `adaptive_per_block` MUST select from the discrete action set `s`, `m`, `l`, and `xl`.
- **FR-006**: The adaptive sampler MUST maintain per-block, per-granularity statistics plus global sampler state.
- **FR-007**: The global sampler state MUST include the current phase or equivalent training progress marker, the current step, the current epoch, and exploration or decay settings.
- **FR-008**: The adaptive reward MUST be derived from loss improvement minus an explicit correction penalty term.
- **FR-009**: The correction penalty term MUST be normalized or scaled so that it is comparable to the loss-improvement signal before the two are combined into reward.
- **FR-010**: `adaptive_per_block` MUST support time-varying behavior across epochs or phases through decay, phase awareness, or both.
- **FR-011**: The sampler strategy for `adaptive_per_block` MUST be explicit in configuration and provenance, and MUST support at least Thompson-style sampling and UCB-style sampling as named options.
- **FR-012**: The resolved configuration and saved artifacts MUST record the selected sampler strategy for every adaptive run.
- **FR-013**: Saved artifacts for `adaptive_per_block` MUST include the resolved mode, selected strategy, sampled per-block pattern summary, reward summary, correction-penalty summary, and the sampler state required to resume training.
- **FR-014**: Saved artifacts for `per_block` and `adaptive_per_block` MUST be distinguishable without consulting console logs.
- **FR-015**: The feature MUST reject invalid mode combinations before training begins, including any attempt to use `adaptive_per_block` outside `nested-random`.
- **FR-016**: The feature MUST keep the existing model wiring and correction machinery intact, changing only how per-block patterns are selected.
- **FR-017**: The feature MUST remain single-process only.

### Research & Experiment Requirements

- **EX-001**: Researchers MUST be able to choose `global`, random `per_block`, or `adaptive_per_block` without changing code.
- **EX-002**: Validation MUST include a smoke case for each of `nested-random + global`, `nested-random + per_block`, and `nested-random + adaptive_per_block`.
- **EX-003**: Validation MUST cover config resolution, correction behavior, persistence and provenance, and resume behavior for the adaptive sampler.
- **EX-004**: Saved outputs MUST be sufficient to reconstruct which mode and strategy were used without relying on terminal output.
- **EX-005**: The design SHOULD keep the adaptive sampler interpretable enough that future strategy changes can be compared across runs.

### Key Entities *(include if data involved)*

- **Sampling Mode**: The resolved nested-random choice that determines whether selection is global, random per-block, or adaptive per-block.
- **Sampler Strategy**: The named adaptive policy used when `adaptive_per_block` is active, such as Thompson-style or UCB-style sampling.
- **Adaptive Sampler State**: The persistent state that tracks per-block, per-granularity evidence plus global progress and exploration settings.
- **Reward Record**: The outcome signal used to update the adaptive sampler, including loss improvement and normalized correction penalty.
- **Granularity Pattern**: The selected granularity or per-block pattern used for a given run step.
- **Run Provenance Record**: The saved metadata needed to distinguish modes, interpret runs, and resume adaptive training.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In checked `nested-random + global` runs, 100% of validated iterations preserve the current whole-model selection behavior.
- **SC-002**: In checked `nested-random + per_block` runs, 100% of validated iterations preserve the current random per-block baseline behavior.
- **SC-003**: In a two-phase adaptive smoke run, at least one transformer block shows a different top-preferred granularity after the phase change or decay point, demonstrating nonstationary behavior.
- **SC-004**: 100% of completed `adaptive_per_block` runs record the resolved mode, selected strategy, sampled per-block pattern summary, reward summary, correction-penalty summary, and resumable sampler state in saved artifacts.
- **SC-005**: 100% of invalid mode combinations in the validation matrix fail before training begins.
- **SC-006**: 100% of validated runs in the feature scope can be distinguished from artifacts alone as `global`, random `per_block`, or `adaptive_per_block`.
- **SC-007**: 100% of successful adaptive resumes continue from prior sampler state rather than starting from an empty policy history.

## Assumptions

- The current random `per_block` baseline remains available and unchanged apart from explicit config and provenance fields.
- The supported granularity choices remain `s`, `m`, `l`, and `xl`.
- Single-process execution remains the only supported runtime model for this feature.
- Training exposes enough progress information to support phase-aware or decay-based adaptation.
- Saved artifacts can be extended to store sampler strategy, reward summaries, and resume state.
- Thompson-style sampling is the default adaptive strategy, and UCB-style sampling is also supported as an explicit named option.
