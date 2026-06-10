# Tasks: Granularity Sampling Modes

**Input**: Design documents from `/specs/005-granularity-sampling-modes/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/, quickstart.md

**Tests/Verification**: Include focused tests and smoke checks for model sampling semantics, correction activation, metadata provenance, and global/parity regressions.

**Organization**: Tasks are grouped by user story so each story can be implemented and verified independently.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Experiment Structure)

**Purpose**: Create the shallow model package and shared scaffolding needed for all story work

- [X] T001 Create the shallow `models/` package scaffold with `models/__init__.py`, `models/granularity.py`, `models/correction.py`, `models/ffn.py`, and `models/wiring.py`
- [X] T002 Update the feature plan reference in `AGENTS.md` to point at `specs/005-granularity-sampling-modes/plan.md`
- [X] T003 [P] Add a small compatibility export layer in `modified_llama.py` so current imports still resolve while the new package is introduced

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared sampling/config/data plumbing that all user stories depend on

**CRITICAL**: No user story work should begin until this phase is complete

- [X] T004 Add explicit model-level sampling-mode config resolution and legacy `training.granularity_sampling` alias handling in `utils/config.py`
- [X] T005 [P] Add validation rules for `global` and `per_layer` sampling modes plus legacy alias resolution and their interaction with `model.correction_mode` in `utils/config.py`
- [X] T006 [P] Add run-metadata fields for requested legacy alias, resolved sampling mode, and granularity-pattern provenance in `utils/config.py` and `utils/metrics.py`
- [X] T007 Introduce shared granularity-pattern data structures and helpers in `models/granularity.py`
- [X] T008 Introduce shared correction-context helpers for global vs per-layer activation in `models/correction.py`

**Checkpoint**: Shared model/config plumbing is ready and user story work can proceed

---

## Phase 3: User Story 1 - Preserve the current global path (Priority: P1) MVP

**Goal**: Keep the current whole-model granularity behavior available as an explicit `global` sampling mode with unchanged correction semantics

**Independent Test**: A debug run with `model.granularity_sampling_mode=global` still chooses one granularity for the full forward pass and produces the same global correction behavior as before

### Verification for User Story 1

- [X] T009 [P] [US1] Add a focused config-resolution test for legacy `training.granularity_sampling` aliases in `tests/test_config.py`
- [X] T010 [P] [US1] Add a focused regression test for the explicit global sampling path in `tests/test_matformer_prefixes.py`

### Implementation for User Story 1

- [X] T011 [P] [US1] Implement explicit global sampling selection in `models/wiring.py`
- [X] T012 [P] [US1] Move the existing global FFN configuration path into `models/ffn.py` so the current behavior is preserved under the new mode
- [X] T013 [US1] Update `modified_llama.py` to route global sampling through the new model-level API without changing the observed forward-pass behavior
- [X] T014 [US1] Record the resolved global sampling mode and pattern summary in run summaries through `training/run.py`

**Checkpoint**: User Story 1 should now be fully functional and independently verifiable

---

## Phase 4: User Story 2 - Sample granularity per layer (Priority: P2)

**Goal**: Enable per-layer sampling so each transformer block can receive its own granularity choice and local correction interpretation

**Independent Test**: A debug run with `model.granularity_sampling_mode=per_layer` produces a block-wise granularity pattern and activates local GMC/LMC only in that mode

### Verification for User Story 2

- [X] T015 [P] [US2] Add a focused per-layer sampling regression test in `tests/test_matformer_prefixes.py`
- [X] T016 [P] [US2] Add a focused correction-activation test for per-layer mode in `tests/test_config.py`

### Implementation for User Story 2

- [X] T017 [P] [US2] Implement per-layer sampling pattern selection in `models/wiring.py`
- [X] T018 [P] [US2] Implement local GMC/LMC derivation from a sampled per-layer pattern in `models/correction.py`
- [X] T019 [US2] Update the model assembly path in `models/wiring.py` so each transformer block consumes the per-layer granularity choice for the current forward pass
- [X] T020 [US2] Update `training/run.py` and `utils/metrics.py` so per-layer runs record the sampled pattern and correction context in saved artifacts

**Checkpoint**: User Story 2 should now work independently of any later refactor cleanup

---

## Phase 5: User Story 3 - Keep model responsibilities clear (Priority: P3)

**Goal**: Separate granularity metadata, FFN implementations, correction logic, and model wiring so future sampling strategies are easier to add

**Independent Test**: The refactored modules can be reviewed and tested separately, with metadata, FFN behavior, correction behavior, and wiring each changing in isolation

### Verification for User Story 3

- [X] T021 [P] [US3] Add focused unit coverage for the new granularity metadata helpers in `tests/test_matformer_prefixes.py`
- [X] T022 [P] [US3] Add focused artifact/provenance coverage for sampling mode and pattern summaries in `tests/test_artifacts.py`

### Implementation for User Story 3

- [X] T023 [P] [US3] Move the granularity metadata helpers out of `modified_llama.py` into `models/granularity.py`
- [X] T024 [P] [US3] Move the MatFormer FFN implementations out of `modified_llama.py` into `models/ffn.py`
- [X] T025 [P] [US3] Move the correction helpers and local/global correction boundaries into `models/correction.py`
- [X] T026 [US3] Move the model assembly and layer-wiring logic into `models/wiring.py`
- [X] T027 [US3] Keep `modified_llama.py` as a thin compatibility façade that delegates to the new `models/` package
- [X] T028 [US3] Update `train.py` and `training/run.py` call sites so the explicit sampling-mode API is used consistently end to end

**Checkpoint**: All user stories should now be independently functional and the model responsibilities should be clearer

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Finish validation, cleanup, and run-documentation updates that touch multiple stories

- [X] T029 [P] Run the focused test set from `specs/005-granularity-sampling-modes/quickstart.md` and capture any follow-up fixes in `tests/test_config.py`, `tests/test_matformer_prefixes.py`, or `tests/test_training_smoke.py`
- [X] T030 [P] Verify the global, per-layer, and legacy-alias smoke commands from `specs/005-granularity-sampling-modes/quickstart.md` and adjust `training/run.py` or `utils/metrics.py` if saved artifacts are incomplete
- [X] T031 Clean up any redundant helpers left behind in `modified_llama.py` after the module split
- [X] T032 Confirm `config.json`, `run_summary.json`, and `metrics.csv` include the selected sampling mode and granularity-pattern summary for both supported modes

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - blocks all user stories
- **User Stories (Phase 3+)**: Depend on Foundational completion
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Starts after Foundational completion - no dependency on other stories
- **User Story 2 (P2)**: Starts after Foundational completion - should preserve User Story 1 behavior
- **User Story 3 (P3)**: Starts after User Stories 1 and 2 are in place - refactors and clarifies the shared implementation

### Within Each User Story

- Verification comes first when it helps catch likely research regressions
- Shared config and data structures before model wiring changes
- Model logic before run-summary and artifact updates
- Preserve the current global path before introducing the per-layer path
- Keep the compatibility façade working until the new modules are fully wired

### Parallel Opportunities

- `T003` can run in parallel with the package scaffold work in `T001`
- `T005` and `T006` can run in parallel after `T004`
- `T009` and `T010` can run in parallel
- `T011` and `T012` can run in parallel
- `T015` and `T016` can run in parallel
- `T017` and `T018` can run in parallel
- `T021` and `T022` can run in parallel
- `T023`, `T024`, and `T025` can run in parallel because they touch separate modules

---

## Parallel Example: User Story 1

```bash
# Run the verification tasks together:
Task: "Add a focused regression test for the explicit global sampling path in tests/test_matformer_prefixes.py"
Task: "Add a focused config-resolution test for model.granularity_sampling_mode=global in tests/test_config.py"

# Run the independent implementation tasks together:
Task: "Implement explicit global sampling selection in models/wiring.py"
Task: "Move the existing global FFN configuration path into models/ffn.py so the current behavior is preserved under the new mode"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. Stop and validate the explicit global sampling path

### Incremental Delivery

1. Complete Setup + Foundational
2. Deliver User Story 1 and validate parity with the current global behavior
3. Deliver User Story 2 and validate per-layer sampling plus local correction
4. Deliver User Story 3 and validate the refactor boundaries and provenance
5. Finish with Polish and smoke validation

### Parallel Team Strategy

1. Team completes Setup + Foundational together
2. Once foundational work is complete:
   - Developer A: User Story 1 global path
   - Developer B: User Story 2 per-layer path
   - Developer C: User Story 3 refactor and provenance cleanup
3. Rejoin for polish and smoke validation

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Keep verification focused on likely research failures
- Prefer preserving the current global behavior before expanding the sampling surface
- Avoid introducing any distributed-training behavior in this feature
