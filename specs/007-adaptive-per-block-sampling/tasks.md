# Tasks: Adaptive Per-Block Sampling

**Input**: Design documents from `/specs/007-adaptive-per-block-sampling/`
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`

**Tests/Verification**: Include focused config-resolution, artifact, adaptive-sampler, and training-smoke checks for failure modes likely to waste research time.

**Organization**: Tasks are grouped by user story so each story can be implemented and tested independently.

## Phase 1: Setup (Shared Experiment Structure)

**Purpose**: Create the shared module and export surface for the new adaptive sampler.

- [X] T001 [P] Create the adaptive sampler module scaffold and export it from `models/adaptive_sampler.py` and `models/__init__.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core research-code pieces that all user stories depend on.

**CRITICAL**: No user story work should begin until this phase is complete.

- [ ] T002 Update `utils/config.py` to resolve and validate `adaptive_per_block`, `model.adaptive_sampler_strategy`, `model.adaptive_sampler_exploration_scale`, `model.adaptive_sampler_decay_rate`, and `model.adaptive_sampler_reward_penalty_weight`
- [ ] T003 [P] Update `models/correction.py` and `models/granularity.py` to keep correction context and runtime pattern summaries stable for explicit `global`, `per_block`, and `adaptive_per_block` resolution
- [ ] T004 [P] Update `models/wiring.py` and `training/run.py` to carry resolved sampling mode and runtime provenance through model setup and training-state initialization
- [ ] T005 [P] Add foundational mode-resolution coverage in `tests/test_config.py` for explicit `global`, random `per_block`, adaptive `per_block`, and legacy `nested-all` / `standalone` normalization plus `training.granularity_sampling` alias stability
- [ ] T022 [P] Update `train.py` and `training/run.py` to reject distributed or multi-process execution paths before any training setup begins

**Checkpoint**: Foundation ready - user story implementation can now begin.

---

## Phase 3: User Story 1 - Keep the explicit nested-random modes stable (Priority: P1)

**Goal**: Researchers can choose `nested-random + global`, `nested-random + per_block`, or `nested-random + adaptive_per_block`, and invalid pairings fail before training starts.

**Independent Test**: Configure each supported nested-random combination plus representative invalid pairings, then verify valid runs resolve cleanly and invalid ones are rejected before training begins.

### Verification for User Story 1

- [ ] T006 [P] [US1] Add regression tests for valid nested-random mode resolution, legacy alias normalization, and invalid adaptive pairings in `tests/test_config.py`

### Implementation for User Story 1

- [ ] T007 [P] [US1] Update `models/correction.py` so `adaptive_per_block` uses the same local-correction activation rules as the baseline per-block path
- [ ] T008 [P] [US1] Update `models/granularity.py` and `models/wiring.py` so global and random per-block provenance stay unchanged while adaptive resolves to its own per-block pattern summary
- [ ] T009 [US1] Update `training/run.py` and `train.py` to fail fast on unsupported `adaptive_per_block` pairings before any training step begins
- [ ] T021 [P] Add a single-process-only startup guard regression test in `tests/test_training_smoke.py` for distributed or multi-process execution paths

**Checkpoint**: User Story 1 should now be fully functional and independently testable.

---

## Phase 4: User Story 2 - Learn adaptive per-block preferences over time (Priority: P2)

**Goal**: `nested-random + adaptive_per_block` can bias future per-block choices from recent training outcomes and resume from saved state.

**Independent Test**: Run adaptive mode across multiple steps or phases, verify the selected per-block pattern changes over time, then resume from saved state and confirm the sampler continues from prior history.

### Verification for User Story 2

- [ ] T010 [P] [US2] Add unit tests for Thompson/UCB scoring, decay, and reward updates in `tests/test_adaptive_sampler.py`
- [ ] T011 [P] [US2] Add smoke coverage for adaptive pattern shifts and resume behavior in `tests/test_training_smoke.py`

### Implementation for User Story 2

- [ ] T012 [P] [US2] Implement `models/adaptive_sampler.py` with per-block statistics, Thompson/UCB scoring, decay, and reward-update helpers
- [ ] T013 [US2] Integrate adaptive pattern selection and state updates into `training/run.py` so each step chooses one granularity per transformer block and applies the sampled pattern to the model
- [ ] T014 [US2] Add adaptive sampler checkpoint and resume plumbing in `training/run.py` so saved runs restore sampler history before training continues

**Checkpoint**: User Story 2 should now be independently functional and resumable.

---

## Phase 5: User Story 3 - Make runs interpretable and resumable from artifacts (Priority: P3)

**Goal**: Saved artifacts show whether a run used `global`, random `per_block`, or `adaptive_per_block`, which strategy was used, and what state is needed to resume training.

**Independent Test**: Inspect saved artifacts from each supported mode and verify that the resolved mode, selected strategy, sampled pattern summary, reward summary, correction-penalty summary, and resumable state are visible without logs.

### Verification for User Story 3

- [ ] T015 [P] [US3] Update `tests/test_artifacts.py` to assert artifact distinguishability for `nested-random + global`, `nested-random + per_block`, and `nested-random + adaptive_per_block`, including correction mode, membership flags, sampler hyperparameters, correction context, and output locations

### Implementation for User Story 3

- [ ] T016 [P] [US3] Extend `utils/metrics.py` to write adaptive strategy, reward summary, correction-penalty summary, sampler-state fields, correction mode, membership flags, sampler hyperparameters, correction context, and output locations into `config.json`, `run_summary.json`, and `metrics.csv`
- [ ] T017 [US3] Thread adaptive provenance through `training/run.py` so the artifact writers receive the resolved mode, sampled pattern, reward, sampler-state values, correction mode, membership flags, sampler hyperparameters, correction context, and output locations for each step
- [ ] T018 [US3] Update `specs/007-adaptive-per-block-sampling/quickstart.md` and `README.md` with the final adaptive-mode validation and artifact inspection commands

**Checkpoint**: All user stories should now be independently functional.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final cleanup and validation that affect multiple user stories.

- [ ] T019 [P] Clean up duplicated provenance handling and keep the shallow code paths readable in `models/adaptive_sampler.py`, `training/run.py`, and `utils/metrics.py`
- [ ] T020 [P] Run the focused validation commands from `specs/007-adaptive-per-block-sampling/quickstart.md` and fix any regressions in `tests/test_config.py`, `tests/test_adaptive_sampler.py`, `tests/test_artifacts.py`, and `tests/test_training_smoke.py`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - blocks all user stories
- **User Stories (Phase 3+)**: Depend on Foundational completion
  - User stories can then proceed in priority order
  - US2 may integrate with US1, but should remain independently testable
  - US3 depends on the adaptive runtime and artifact fields established by US1 and US2
- **Polish (Final Phase)**: Depends on completion of the desired user stories

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational - no dependency on other stories
- **User Story 2 (P2)**: Can start after Foundational, but uses the explicit mode surface from US1
- **User Story 3 (P3)**: Can start after US1 and US2, because the artifact contract depends on runtime provenance and adaptive state

### Within Each User Story

- Verification tasks should be written before or alongside implementation when they help catch likely research failures
- Configuration and validation come before runtime wiring
- Runtime wiring comes before summary and artifact serialization
- Story complete before moving to the next priority

### Parallel Opportunities

- T001 can run in parallel with future docs-only edits, since it only creates the new module scaffold
- T003, T004, and T005 can run in parallel once T002 is complete, because they touch different files
- T006 can run in parallel with T007 after the foundational config rules are in place
- T010 and T011 can run in parallel for the adaptive story
- T012 can run in parallel with T010 and T011
- T015 and T016 can run in parallel for the artifact story
- T018 can run alongside T016 and T017 if docs and code changes are split cleanly

---

## Parallel Example: User Story 1

```bash
Task: "Add regression tests for valid nested-random mode resolution and invalid adaptive pairings in `tests/test_config.py`"
Task: "Update `models/correction.py` so `adaptive_per_block` uses the same local-correction activation rules as the baseline per-block path"
Task: "Update `models/granularity.py` and `models/wiring.py` so global and random per-block provenance stay unchanged while adaptive resolves to its own per-block pattern summary"
```

## Parallel Example: User Story 2

```bash
Task: "Add unit tests for Thompson/UCB scoring, decay, and reward updates in `tests/test_adaptive_sampler.py`"
Task: "Add smoke coverage for adaptive pattern shifts and resume behavior in `tests/test_training_smoke.py`"
Task: "Implement `models/adaptive_sampler.py` with per-block statistics, Thompson/UCB scoring, decay, and reward-update helpers"
```

## Parallel Example: User Story 3

```bash
Task: "Update `tests/test_artifacts.py` to assert artifact distinguishability for `nested-random + global`, `nested-random + per_block`, and `nested-random + adaptive_per_block`"
Task: "Extend `utils/metrics.py` to write adaptive strategy, reward summary, correction-penalty summary, and sampler-state fields into `config.json`, `run_summary.json`, and `metrics.csv`"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. Stop and validate User Story 1 independently

### Incremental Delivery

1. Complete Setup + Foundational
2. Add User Story 1 and verify explicit nested-random resolution
3. Add User Story 2 and verify adaptive sampling and resume behavior
4. Add User Story 3 and verify artifact provenance and distinguishability
5. Run the polish phase to clean up and validate the final state

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently
