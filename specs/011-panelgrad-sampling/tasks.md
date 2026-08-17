# Tasks: PanelGrad Sampling

**Input**: Design documents from `/specs/011-panelgrad-sampling/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests/Verification**: The specification requires focused unit, smoke, isolation, resume, distributed, artifact, reporting, and compatibility coverage. Verification tasks are listed before their corresponding implementation tasks where practical.

**Organization**: Tasks are grouped by user story so the scientific behavior, isolation guarantees, and auditability can be implemented and validated as explicit increments.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel because it changes different files and does not depend on another incomplete task in the same group
- **[Story]**: User story served by the task (`US1`, `US2`, or `US3`)
- Every task names the exact file or files it changes

## Phase 1: Setup (Shared Experiment Structure)

**Purpose**: Add explicit, opt-in inputs for developing and exercising PanelGrad without changing default experiment queues.

- [X] T001 [P] Create a minimal deterministic PanelGrad test fixture with multiple global granularities and a short refresh interval in `tests/fixtures/panelgrad_smoke.yaml`
- [X] T002 [P] Add the explicit opt-in PanelGrad smoke experiment, without adding it to default pilot queues, in `configs/opt-in_exps/panelgrad_smoke.yaml`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Resolve PanelGrad identity, controller-data availability, independent randomness, and exact FFN support before implementing the policy lifecycle.

**CRITICAL**: Complete this phase before starting any user-story implementation.

- [X] T003 Add `panelgrad` strategy resolution, the `model.panelgrad` defaults and conflict validation, fixed method provenance, balanced-warmup eligibility, and preflight output in `src/utils/config.py`
- [X] T004 [P] Register the dedicated `panelgrad_sampling` seed stream and include its provenance in reproducibility and comparison signatures in `src/utils/reproducibility.py`
- [X] T005 [P] Separate the shared controller-panel predicate from the TS-only controller predicate so raw and packed PanelGrad runs reuse the existing four disjoint data roles in `src/training/data.py` and `src/training/run.py`
- [X] T006 [P] Add slicing and concat controlled-FFN support descriptors/counts plus a scoped membership-correction suspension that excludes shared down bias and non-FFN parameters in `src/models/ffn.py`
- [X] T007 [P] Cover slicing/concat support coordinates, unique scalar counts, shared-bias exclusion, zero-gradient stability, correction restoration, and rejection of frozen layouts with zero controlled trainable scalars in `tests/test_matformer_prefixes.py`

**Checkpoint**: PanelGrad configuration, controller role, RNG identity, and controlled parameter support are explicit and testable.

---

## Phase 3: User Story 1 - Sample Global Granularities from Current Importance (Priority: P1) MVP

**Goal**: Measure every complete global granularity at a common boundary, construct one valid categorical distribution, and independently sample one global action for each ordinary optimizer step.

**Independent Test**: Use controlled gradients and a fixed PanelGrad sampling seed to verify exact RMS scores and `q`/`p` values, then verify the expected action sequence with refreshes at steps `0,H,2H` and one ordinary global training path per sampled action.

### Verification for User Story 1

- [X] T008 [P] [US1] Add valid/default/override and invalid scope, policy-value, empty/duplicate resolved-granularity, and preflight cases, explicitly rejecting Thompson posterior/reset inputs, UCB controls, EMA, EXP3, inverse-probability weighting, cost-aware correction, and per-block PanelGrad in `tests/test_config.py`
- [X] T009 [P] [US1] Add unit cases for aggregate-gradient RMS, microbatch invariance, one/all-zero arms, temperature, epsilon endpoints, stable normalization, categorical determinism, state transitions, invalid numerics, and zero controlled support failing before refresh or action selection in `tests/test_panelgrad.py`
- [X] T010 [P] [US1] Add lifecycle smoke cases for initial refresh, frozen `p`, per-step resampling, exactly-`H` completed-step boundaries, and equality with the ordinary unweighted global loss/update path without inverse-probability or cost correction in `tests/test_training_smoke.py`

### Implementation for User Story 1

- [X] T011 [US1] Implement PanelGrad configuration/state validation, float64 score and `q`/`p` mathematics, dedicated CPU generator handling, categorical draws, and exposure accounting in `src/training/panelgrad.py`
- [X] T012 [US1] Implement one-granularity-at-a-time aggregate target-token-weighted controller-gradient measurement over the exact controlled FFN support in `src/training/panelgrad.py`
- [X] T013 [US1] Construct the fixed controller-panel loader and initialize a new PanelGrad controller only for selected PanelGrad runs in `src/training/data.py` and `src/training/run.py`
- [X] T014 [US1] Add the named pre-action `refresh-if-due -> sample` block and post-optimizer-commit accounting while reusing the existing global action and training path in `src/training/steps.py`
- [X] T015 [US1] Wire refresh execution, selected-action adaptation, controller callbacks, and clean terminal handoff into the main experiment flow in `src/training/run.py`

**Checkpoint**: PanelGrad is a functional MVP that refreshes a common full panel and samples one complete global granularity per optimizer step.

---

## Phase 4: User Story 2 - Preserve Data and Optimization Separation (Priority: P2)

**Goal**: Ensure controller measurement affects only the sampling decision, never becomes an optimizer update, never advances training state, and never consults the final holdout.

**Independent Test**: Snapshot model, optimizer, scheduler, gradients, data cursor, RNG, model mode, and active granularity around a refresh and prove that only PanelGrad state changes; separately verify post-warmup refresh and unchanged non-PanelGrad behavior.

### Verification for User Story 2

- [X] T016 [P] [US2] Add PanelGrad activation, four-role disjointness, controller manifest reuse, replicated controller iteration, and final-holdout non-use cases in `tests/test_data_validation.py`
- [X] T017 [P] [US2] Add success/failure isolation cases proving unchanged parameters, optimizer, scheduler, mixed-precision scaler, training metrics, training-data cursor, ordinary RNG, model/granularity mode, and correction hooks; empty gradients afterward; stale-distribution rejection; and refresh atomicity in `tests/test_panelgrad.py`
- [X] T018 [P] [US2] Add balanced-global warmup transition, no-remaining-budget, failed optimizer attempt, terminal partial/exact interval, and unselected-strategy compatibility cases in `tests/test_training_smoke.py`

### Implementation for User Story 2

- [X] T019 [US2] Make the refresh a try/finally transaction that restores model mode, global granularity, correction state, ordinary RNG, and empty gradients, never invokes or mutates optimizer, scheduler, mixed-precision, training-metric, or training-data-cursor state, and atomically installs only a complete measurement in `src/training/panelgrad.py`
- [X] T020 [US2] Snapshot and roll back the pending PanelGrad draw and generator state on pre-commit failure, and advance interval/exposure state only after optimizer and scheduler commit in `src/training/steps.py`
- [X] T021 [US2] Reuse balanced-global warmup without PanelGrad exposure, TS resets, or acquisition episodes, and force a fresh measurement before the first adaptive action when training continues in `src/utils/config.py`, `src/training/warmup.py`, and `src/training/run.py`
- [X] T022 [US2] Enforce controller-only measurement provenance and terminal partial/complete handling with no unused refresh or draw in `src/training/run.py` and `src/training/panelgrad.py`

**Checkpoint**: PanelGrad decisions are isolated from optimization and final evaluation, including warmup, failure, and terminal boundaries.

---

## Phase 5: User Story 3 - Reproduce and Audit PanelGrad Decisions (Priority: P3)

**Goal**: Make refresh boundaries, categorical draws, distributed agreement, exposure history, provenance, and measurement cost exactly resumable and inspectable.

**Independent Test**: Compare uninterrupted and resumed runs inside an interval and at `refresh_pending`, verify identical subsequent actions/exposures and tolerance-equal numeric state, then inspect one rank-zero artifact set that reconstructs every refresh and action.

### Verification for User Story 3

- [X] T023 [P] [US3] Add uninterrupted-versus-resumed cases inside an interval, exactly at a refresh boundary, after warmup, and at terminal state plus malformed/config/support/role/RNG/journal mismatch rejection in `tests/test_panelgrad_resume.py`
- [X] T024 [P] [US3] Add single-process-versus-FSDP equivalence cases for `N_g`, gradient norm, RMS, `q`, and `p`; verify counts are not world-size multiplied, replicated losses use `n_b/N` without a world-size multiplier, backward counts match, per-layer support is exact, actions synchronize from rank zero, and only rank zero writes artifacts in `tests/test_distributed.py`
- [X] T025 [P] [US3] Add checkpoint and artifact cases for every PanelGrad phase, transactional refresh/failure events, compact action rows, summary hashes, exposure totals, terminal state, and separate measurement cost in `tests/test_artifacts.py`
- [X] T026 [P] [US3] Add explicit PanelGrad method classification, provenance parsing, final vectors, exposure summaries, and non-PanelGrad reporting compatibility cases in `tests/test_reporting.py`

### Implementation for User Story 3

- [X] T027 [P] [US3] Add strict versioned `panelgrad_state` validation and explicit checkpoint save/load fields for policy, support, role hashes, refresh phase, probabilities, RNG, exposures, journal commit state, and failure provenance in `src/training/checkpointing.py`
- [X] T028 [P] [US3] Resolve `N_g` before FSDP wrapping without all-reducing it, use replicated `n_b/N` loss scaling without a world-size multiplier, summon one wrapped decoder layer at a time with gradients, accumulate exact support squares in float64, reproduce single-process RMS within tolerance, and reject unsupported settings in `src/training/panelgrad.py` and `src/training/distributed.py`
- [X] T029 [US3] Make rank zero authoritative for PanelGrad state and categorical draws, broadcast the committed state/action, and validate cross-rank hashes without changing existing samplers in `src/training/steps.py` and `src/training/run.py`
- [X] T030 [P] [US3] Add PanelGrad compact training columns, refresh/failure/terminal event validation, and controller-summary aggregation with measurement-cost totals in `src/utils/metrics.py`
- [X] T031 [US3] Integrate transactional rank-zero controller journal commits, rollback reconciliation, controller/run summaries, provenance hashes, and separate refresh cost into `src/training/run.py`
- [X] T032 [US3] Restore and reconcile PanelGrad checkpoint, named generator, boundary, exposure, warmup, controller manifest, and journal state before further action selection in `src/training/run.py` and `src/training/checkpointing.py`
- [X] T033 [P] [US3] Classify and parse PanelGrad explicitly, expose its distributions/exposures/cost without treating it as TS or UCB, and preserve existing report labels in `src/evaluation/reporting.py`, `src/evaluation/reporting_io.py`, and `src/evaluation/reporting_impl.py`

**Checkpoint**: PanelGrad runs are exactly resumable under equivalent conditions and fully auditable from checkpoints and structured artifacts.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Validate the opt-in experiment end to end and protect all existing sampling strategies.

- [X] T034 [P] Add explicit regressions proving non-PanelGrad checkpoints contain no PanelGrad state and Thompson, UCB, random global/per-block, nested-all, and standalone semantics remain unchanged in `tests/test_artifacts.py`, `tests/test_probabilistic_controller.py`, and `tests/test_adaptive_sampler.py`
- [X] T035 [P] Verify the PanelGrad preflight and short smoke commands, and record any corrected commands or expected artifact checks in `specs/011-panelgrad-sampling/quickstart.md`
- [X] T036 Run `tests/test_panelgrad.py`, `tests/test_panelgrad_resume.py`, `tests/test_distributed.py`, `tests/test_artifacts.py`, `tests/test_reporting.py`, and `tests/test_data_validation.py`; correct exposed implementation defects in `src/training/panelgrad.py`, `src/training/run.py`, `src/training/steps.py`, `src/training/checkpointing.py`, `src/training/distributed.py`, `src/training/data.py`, `src/utils/metrics.py`, `src/evaluation/reporting.py`, `src/evaluation/reporting_io.py`, and `src/evaluation/reporting_impl.py`, and modify test assertions only when they conflict with `specs/011-panelgrad-sampling/spec.md`
- [X] T037 Run the existing sampling/configuration compatibility suite and make only necessary compatibility corrections in `src/utils/config.py`, `src/training/steps.py`, `src/training/run.py`, and `src/training/checkpointing.py`
- [X] T038 Create a matched uniform-global baseline in `configs/opt-in_exps/panelgrad_uniform_baseline.yaml`, run it beside `configs/opt-in_exps/panelgrad_smoke.yaml` with identical model, data manifests, seed, optimizer, scheduler, validation, and step/token budget, apply an explicit checkpoint/final-holdout selection rule, write exposures, validation/final-holdout metrics, training time, and separate PanelGrad measurement cost to `outputs/panelgrad-comparison/comparison.json`, and document the commands in `specs/011-panelgrad-sampling/quickstart.md`

---

## Phase 7: Linear Epsilon Schedule Extension

**Purpose**: Add an opt-in refresh-boundary linear exploration schedule while preserving fixed-epsilon behavior.

- [X] T039 Add mutually exclusive fixed/linear epsilon configuration resolution and validation in `src/utils/config.py` and `tests/test_config.py`
- [X] T040 Evaluate epsilon from committed PanelGrad steps only at refresh boundaries, freeze it in each probability snapshot, and preserve rollback/warmup/terminal semantics in `src/training/panelgrad.py`, `src/training/run.py`, `tests/test_panelgrad.py`, and `tests/test_training_smoke.py`
- [X] T041 Increment PanelGrad state schema, validate `p` with the snapshot epsilon, and migrate version-1 fixed-epsilon state only for fixed-policy resumes in `src/training/panelgrad.py`, `tests/test_panelgrad_resume.py`, and `tests/test_artifacts.py`
- [X] T042 Record epsilon schedule progress in refresh artifacts/controller summaries and add schedule-aware plot identities plus epsilon-over-tokens diagnostics in `src/utils/metrics.py`, `src/evaluation/reporting.py`, `src/evaluation/reporting_io.py`, `src/evaluation/reporting_impl.py`, and `tests/test_reporting.py`
- [X] T043 Verify distributed agreement, categorical reproducibility, scheduled and fixed lifecycle behavior, and non-PanelGrad compatibility in `tests/test_distributed.py` and the focused compatibility suite

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; T001 and T002 can proceed independently.
- **Foundational (Phase 2)**: Depends on Setup and blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational and delivers the scientific MVP.
- **User Story 2 (Phase 4)**: Depends on the working US1 lifecycle because it verifies and hardens its isolation, rollback, warmup, and terminal semantics.
- **User Story 3 (Phase 5)**: Depends on US1; checkpoint/artifact failure and terminal coverage also depends on US2.
- **Polish (Phase 6)**: Depends on all selected user stories.

### User Story Dependency Graph

```text
Setup -> Foundational -> US1 (MVP) -> US2
                              |        |
                              +------> US3 -> Polish
                                       ^
                                       +---- US2 failure/terminal semantics
```

### Within Each User Story

- Write the listed verification cases before or alongside implementation so tensor-support, lifecycle, and reproducibility errors fail visibly.
- In US1, implement state/math before gradient measurement, then integrate the explicit training-loop decision point.
- In US2, establish refresh isolation and sample rollback before warmup and terminal orchestration.
- In US3, establish the checkpoint schema before resume reconciliation; metrics schemas before journal/run-summary integration; distributed measurement before cross-rank lifecycle validation.
- A story is complete only after its independent test passes without relying on the final holdout.

## Parallel Opportunities

- T001 and T002 can run in parallel.
- After T003 establishes the method surface, T004-T007 touch separate implementation or test concerns and can proceed in parallel.
- US1 verification tasks T008-T010 can run in parallel; implementation then proceeds T011 -> T012 -> T013 -> T014 -> T015.
- US2 verification tasks T016-T018 can run in parallel before T019-T022.
- US3 verification tasks T023-T026 can run in parallel. After their contracts are fixed, T027, T028, T030, and T033 can proceed in parallel; T029, T031, and T032 integrate those pieces.
- T034 and T035 can run in parallel after all stories; T036-T038 are final integration checks.

## Parallel Example: User Story 1

```text
Task T008: Add PanelGrad configuration contract tests in tests/test_config.py
Task T009: Add score, probability, gradient, categorical, and state tests in tests/test_panelgrad.py
Task T010: Add refresh and per-step sampling lifecycle tests in tests/test_training_smoke.py
```

## Parallel Example: User Story 2

```text
Task T016: Add controller/final data-role separation tests in tests/test_data_validation.py
Task T017: Add refresh success/failure isolation tests in tests/test_panelgrad.py
Task T018: Add warmup, rollback, terminal, and compatibility smoke tests in tests/test_training_smoke.py
```

## Parallel Example: User Story 3

```text
Task T023: Add exact resume and mismatch-rejection tests in tests/test_panelgrad_resume.py
Task T024: Add distributed FSDP and action-synchronization tests in tests/test_distributed.py
Task T025: Add checkpoint and structured artifact tests in tests/test_artifacts.py
Task T026: Add reporting identity and provenance tests in tests/test_reporting.py
```

## Implementation Strategy

### MVP First: User Story 1

1. Complete Setup and Foundational work.
2. Implement T008-T015 for the complete PanelGrad score-to-action lifecycle.
3. Stop and validate the controlled score, probability, boundary, and action-sequence tests.
4. Run the short opt-in smoke without claiming isolation/resume guarantees until later stories pass.

### Incremental Delivery

1. **US1**: Establish comparable full-panel scores and one categorical global action per step.
2. **US2**: Prove measurement isolation, data separation, rollback, warmup, and terminal correctness.
3. **US3**: Add exact resume, distributed agreement, artifacts, reporting, and measurement-cost accounting.
4. **Polish**: Run compatibility coverage and the first matched-step or matched-token comparison.

## Notes

- Keep the PanelGrad policy decision block visible in `src/training/steps.py`; do not introduce a generic policy registry.
- Keep `panelgrad_state` independent from TS posterior/UCB state and use `p`, not `q`, for every categorical draw.
- Count only unique trainable scalars selected by the resolved FFN granularity; never infer `N_g` from nonzero gradients or whole-model parameter counts.
- Measurement gradients are disposable and must remain separate from ordinary training gradients and optimizer state.
- Preserve unrelated worktree changes and existing strategy behavior while completing each task.
