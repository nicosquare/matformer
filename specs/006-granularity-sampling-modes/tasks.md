# Tasks: Granularity Operation Modes

**Input**: Design documents from `/specs/006-granularity-sampling-modes/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/, quickstart.md

**Tests/Verification**: Focused tests and smoke checks for run-mode resolution, sampling semantics, correction activation, artifact provenance, and nested-all/standalone regressions.

**Organization**: Tasks are grouped by user story so each story can be implemented and verified independently.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Experiment Structure)

**Purpose**: Refresh shared experiment fixtures and validation entry points before shared plumbing changes begin

- [X] T001 [P] Update `configs/debug_matrix.yaml` so the debug matrix exercises explicit `nested-random`, `nested-all`, and `standalone` mode combinations for smoke validation
- [X] T002 [P] Update `configs/dmodel256_pilot_comparison.yaml` so the pilot comparison keeps the canonical `nested-random` path and artifact provenance fields explicit
- [X] T003 [P] Refresh `tests/fixtures/experiment_config_resolution.yaml` so config-resolution tests cover the resolved canonical mode fields

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared config, pattern, correction, and provenance plumbing that all user stories depend on

**CRITICAL**: No user story work should begin until this phase is complete

- [X] T004 Implement canonical `run.sampling_mode` and `model.granularity_sampling_mode` resolution in `utils/config.py`
- [X] T005 [P] Add invalid-combination validation for nested modes, standalone granularities, and correction modes in `utils/config.py`
- [X] T006 [P] Extend provenance serialization helpers for resolved mode and pattern summaries in `utils/metrics.py`
- [X] T007 Propagate the resolved provenance fields into config writing and run-summary generation in `training/run.py`
- [X] T008 [P] Keep granularity-pattern modeling explicit in `models/granularity.py`
- [X] T009 [P] Keep correction-context modeling explicit in `models/correction.py`

**Checkpoint**: Shared run-mode, pattern, correction, and provenance plumbing is ready

---

## Phase 3: User Story 1 - Train with explicit nested-random sampling (Priority: P1) MVP

**Goal**: Preserve the current elastic-training path as an explicit `nested-random` mode with `global` and `per_block` sampling submodes

**Independent Test**: A debug run with `run.sampling_mode=nested-random` and `model.granularity_sampling_mode=global` still chooses one granularity for the full forward pass; switching to `per_block` produces a block-wise pattern, and both runs save the resolved mode, submode, and pattern summary

### Verification for User Story 1

- [X] T010 [P] [US1] Add config-resolution coverage for `nested-random`, `global`, and `per_block` in `tests/test_config.py`
- [X] T011 [P] [US1] Add model-wiring, correction, smoke-run, and artifact verification coverage for the explicit global and per-block paths in `tests/test_matformer_prefixes.py`, `tests/test_training_smoke.py`, and `tests/test_artifacts.py`

### Implementation for User Story 1

- [X] T012 [P] [US1] Implement explicit `nested-random` global/per-block pattern application in `models/wiring.py`
- [X] T013 [US1] Implement per-block correction derivation and global-path parity handling in `models/correction.py`
- [X] T014 [US1] Record nested-random provenance fields and pattern summaries in `training/run.py` and `utils/metrics.py`

**Checkpoint**: User Story 1 should now be fully functional and independently verifiable

---

## Phase 4: User Story 2 - Evaluate all granularities together (Priority: P2)

**Goal**: Support `nested-all` so every configured granularity is evaluated on every iteration and the training objective is the mean of the per-granularity losses

**Independent Test**: Configure `nested-all` with a granularities list, run an iteration, and verify that every configured granularity is evaluated and that the aggregate loss equals the mean of the per-granularity losses

### Verification for User Story 2

- [X] T015 [P] [US2] Add nested-all evaluation and mean-loss coverage in `tests/test_training_smoke.py`
- [X] T016 [P] [US2] Add nested-all artifact and provenance coverage in `tests/test_artifacts.py`

### Implementation for User Story 2

- [X] T017 [P] [US2] Implement nested-all iteration over every configured granularity in `training/run.py`
- [X] T018 [US2] Update `evaluation/validation.py` and `models/correction.py` so nested-all uses the evaluated granularity and does not activate per-block random sampling
- [X] T019 [US2] Record nested-all runtime granularity summaries in `utils/metrics.py`

**Checkpoint**: User Story 2 should now work independently of any later cleanup

---

## Phase 5: User Story 3 - Run a fixed standalone granularity and reconstruct it later (Priority: P3)

**Goal**: Support `standalone` as a fixed-granularity run and make the resolved mode reconstructable from saved artifacts without reading logs

**Independent Test**: Configure standalone for each supported granularity, complete a run, and inspect the saved artifacts to confirm that the run stayed on one granularity and that the chosen mode can be recovered from metadata alone

### Verification for User Story 3

- [X] T020 [P] [US3] Add standalone config-validation coverage for fixed granularities and rejected nested submodes in `tests/test_config.py`
- [X] T021 [P] [US3] Add standalone smoke and artifact reconstruction coverage in `tests/test_model_size.py` and `tests/test_artifacts.py`

### Implementation for User Story 3

- [X] T022 [P] [US3] Implement standalone fixed-granularity resolution in `utils/config.py`
- [X] T023 [US3] Preserve fixed-granularity application for standalone runs in `models/wiring.py`
- [X] T024 [US3] Record standalone provenance fields in `training/run.py` and `utils/metrics.py`

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Validation, cleanup, and artifact checks that affect multiple user stories

- [X] T025 [P] Run the focused quickstart validation commands from `specs/006-granularity-sampling-modes/quickstart.md` and capture any follow-up fixes in `tests/test_training_smoke.py` or `tests/test_artifacts.py`
- [X] T026 [P] Remove redundant helper logic left behind in `utils/config.py`, `utils/metrics.py`, and `training/run.py`
- [X] T027 Confirm `config.json`, `run_summary.json`, and `metrics.csv` reconstruct `nested-random`, `nested-all`, and `standalone` runs without log inspection in `tests/test_artifacts.py`
- [X] T028 [P] Rename remaining `matformer_llama` and `cat_llama` references to the canonical `slicing` and `concat` names in `models/ffn.py`, `models/wiring.py`, `configs/debug_matrix.yaml`, `configs/dmodel256_pilot_comparison.yaml`, `tests/test_matformer_prefixes.py`, `tests/test_artifacts.py`, and `specs/006-granularity-sampling-modes/*`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: Depend on Foundational completion
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Starts after Foundational completion - no dependency on other stories
- **User Story 2 (P2)**: Starts after Foundational completion - should preserve User Story 1 behavior
- **User Story 3 (P3)**: Starts after Foundational completion - reuses the same resolved mode and provenance machinery

### Within Each User Story

- Verification comes first when it helps catch likely research regressions
- Shared config and data structures before model wiring changes
- Model logic before run-summary and artifact updates
- Preserve the current global path before introducing the per-block path
- Keep the canonical mode surface clear and direct

### Parallel Opportunities

- `T001`, `T002`, and `T003` can run in parallel
- `T005`, `T006`, `T008`, and `T009` can run in parallel after `T004`
- `T010` and `T011` can run in parallel
- `T012` can run alongside `T010` and `T011`
- `T015` and `T016` can run in parallel
- `T017` can run alongside `T015` and `T016`
- `T020` and `T021` can run in parallel
- `T022` can run alongside `T020` and `T021`

---

## Parallel Example: User Story 1

```bash
# Run the verification tasks together:
Task: "Add config-resolution coverage for nested-random, global, and per_block in tests/test_config.py"
Task: "Add model-wiring and correction regression coverage for the explicit global and per-block paths in tests/test_matformer_prefixes.py"

# Run the independent implementation tasks together:
Task: "Implement explicit nested-random global/per-block pattern application in models/wiring.py"
Task: "Implement per-block correction derivation and global-path parity handling in models/correction.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. Stop and validate the explicit nested-random sampling paths

### Incremental Delivery

1. Complete Setup + Foundational
2. Deliver User Story 1 and validate parity with the current global behavior
3. Deliver User Story 2 and validate nested-all mean-loss aggregation
4. Deliver User Story 3 and validate standalone provenance reconstruction
5. Finish with Polish and smoke validation

### Parallel Team Strategy

1. Team completes Setup + Foundational together
2. Once foundational work is complete:
   - Developer A: User Story 1 nested-random path
   - Developer B: User Story 2 nested-all path
   - Developer C: User Story 3 standalone provenance path
3. Rejoin for polish and smoke validation

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Keep verification focused on likely research failures
- Prefer preserving the current global behavior before expanding the sampling surface
- Avoid introducing any distributed-training behavior in this feature
