# Tasks: Long Run Support

**Input**: Design documents from `/specs/003-run-monitoring-warmup/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/, quickstart.md

**Tests/Verification**: Include focused smoke checks for resumed runs, monitoring series grouping, and warmup transitions because those are the highest-risk failure modes for this feature.

**Organization**: Tasks are grouped by user story so each story can be implemented and tested independently.

## Format: `[ID] [P?] [Story] Description with file path`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to, using `US1`, `US2`, `US3`
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Experiment Structure)

**Purpose**: Make the long-run feature visible in source configs, add the W&B dependency, and create the shared monitoring helper used by later phases.

- [X] T001 [P] Add explicit continuation, monitoring, and pre-nested warmup defaults to `configs/debug_matrix.yaml` and `configs/dmodel256_pilot_comparison.yaml`
- [X] T002 [P] Create per-run granularity-series labeling helpers in `utils/monitoring.py` for W&B loss traces
- [X] T003 [P] Add `wandb` to `requirements.txt` so optional monitoring is installable in the default environment

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Add the shared config, artifact, and baseline test plumbing that every user story depends on.

**Checkpoint**: Foundation ready - user story implementation can now begin.

- [X] T004 Add continuation, monitoring, and warmup resolution plus validation to `utils/config.py`
- [X] T005 Extend resolved config and run-summary field handling for continuation and warmup metadata in `utils/metrics.py`
- [X] T006 [P] Add baseline smoke coverage for default-off continuation, monitoring, and warmup behavior in `tests/test_config.py` and `tests/test_artifacts.py`

---

## Phase 3: User Story 1 - Continue Long Runs Across Scheduler Limits (Priority: P1)

**Goal**: Let a relaunched run resume from the latest saved progress after a scheduler time limit without changing the launch path.

**Independent Test**: Start a run that is interrupted, relaunch the same resolved run, and confirm it continues from the latest checkpoint instead of repeating completed work.

### Verification for User Story 1

- [X] T007 [P] [US1] Add interrupted-and-relaunched resume smoke coverage in `tests/test_training_smoke.py`
- [X] T008 [P] [US1] Add continuation-state assertions for fresh, resumed, and completed runs in `tests/test_artifacts.py`

### Implementation for User Story 1

- [X] T009 [US1] Implement checkpoint discovery, model restore, optimizer restore, scheduler restore, and step counter restore for resumed runs in `training/run.py`
- [X] T010 [US1] Reuse the same continuation behavior in the legacy direct execution path in `train.py`

**Checkpoint**: User Story 1 should now resume long runs independently of the other stories.

---

## Phase 4: User Story 2 - Monitor Losses by Granularity in Weights & Biases (Priority: P2)

**Goal**: Publish the same granularity-level loss view to W&B that the single-run training trace already uses offline.

**Independent Test**: Run one nested experiment and one standalone experiment with monitoring enabled, then verify the dashboard shows one loss series per active granularity for nested runs and only the standalone series for standalone runs.

### Verification for User Story 2

- [X] T011 [P] [US2] Add nested-versus-standalone monitoring smoke coverage in `tests/test_training_smoke.py`
- [X] T012 [P] [US2] Add a focused series-label/unit test for the per-run grouping helper in `tests/test_monitoring.py`

### Implementation for User Story 2

- [ ] T013 [US2] Wire W&B logging into the shared training loop so per-step loss rows are emitted as granularity-aligned series in `training/run.py`
- [ ] T014 [US2] Persist monitoring enablement and series-label metadata in saved artifacts through `utils/metrics.py`

**Checkpoint**: User Story 2 should now expose the monitored loss-by-granularity view independently of warmup changes.

---

## Phase 5: User Story 3 - Warm Up Before Nested Splitting (Priority: P3)

**Goal**: Run a short configurable warmup before nested block slicing/splitting begins so nested training starts from better weights.

**Independent Test**: Run a nested experiment with warmup enabled, verify the warmup stage completes first, and confirm nested training starts from the warmed state rather than a fresh initialization.

### Verification for User Story 3

- [ ] T015 [P] [US3] Add warmup configuration validation coverage in `tests/test_config.py`
- [ ] T016 [P] [US3] Add warmup transition and disabled-path smoke coverage in `tests/test_training_smoke.py`

### Implementation for User Story 3

- [ ] T017 [US3] Implement `training.pre_nested_warmup` parsing, validation, and resolved warmup state in `utils/config.py`
- [ ] T018 [US3] Implement the pre-nested warmup stage and warmup-state persistence in `training/run.py`
- [ ] T019 [US3] Assert warmup completion, warmup unit, and nested-transition fields in `tests/test_artifacts.py`

**Checkpoint**: All user stories should now be independently functional.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Align docs and run the focused validation path after implementation.

- [ ] T020 [P] Update `specs/003-run-monitoring-warmup/quickstart.md` and `README.md` with the final continuation, monitoring, and warmup usage examples
- [ ] T021 Run the focused smoke checks from `specs/003-run-monitoring-warmup/quickstart.md` and fix any drift in `tests/test_config.py`, `tests/test_training_smoke.py`, `tests/test_artifacts.py`, and `tests/test_monitoring.py`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - blocks all user stories
- **User Stories (Phase 3+)**: Depend on Foundational phase completion
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Starts after Foundational; no dependency on later stories
- **User Story 2 (P2)**: Starts after Foundational; can be implemented independently of User Story 3
- **User Story 3 (P3)**: Starts after Foundational; can be implemented independently of User Story 2

### Within Each User Story

- Verification comes before or alongside implementation when it helps expose the likely failure mode
- Config/schema changes before run-loop wiring
- Shared helpers before consumers
- Artifact writing before assertions against saved files
- Story complete before moving to the next priority

### Parallel Opportunities

- `T001` and `T002` can run in parallel because they touch different files
- `T005` can run in parallel with the foundational implementation tasks once the schema shape is agreed
- `T006` and `T007` can run in parallel because they target different execution paths
- `T010` and `T011` can run in parallel because one covers run behavior and the other covers the grouping helper
- `T014` and `T015` can run in parallel because one validates config parsing and the other validates the warmup transition

## Parallel Example: User Story 1

```bash
Task: "Add interrupted-and-relaunched resume smoke coverage in tests/test_training_smoke.py"
Task: "Implement checkpoint discovery, model restore, optimizer restore, scheduler restore, and step counter restore for resumed runs in training/run.py"
```

## Parallel Example: User Story 2

```bash
Task: "Add nested-versus-standalone monitoring smoke coverage in tests/test_training_smoke.py"
Task: "Add a focused series-label/unit test for the shared grouping helper in tests/test_monitoring.py"
```

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. Stop and validate resumed long-run behavior independently

### Incremental Delivery

1. Setup + Foundational -> shared continuation, monitoring, and warmup plumbing
2. User Story 1 -> scheduler-resilient run continuation
3. User Story 2 -> live granularity monitoring in W&B
4. User Story 3 -> configurable pre-nested warmup
5. Polish -> docs alignment and focused validation

### Suggested MVP Scope

- User Story 1 only: continuation after scheduler limits, with the existing launch path preserved
