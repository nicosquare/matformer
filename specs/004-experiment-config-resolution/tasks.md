# Tasks: Experiment Config Resolution

**Input**: Design documents from `/specs/004-experiment-config-resolution/`  
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`

**Organization**: Tasks are grouped by user story to keep each slice independently implementable and testable.

## Phase 1: Setup (Shared Experiment Structure)

**Purpose**: Add shared fixtures and example configs used by the feature tests.

- [X] T001 [P] Create a shared experiment-config fixture at `tests/fixtures/experiment_config_resolution.yaml` plus separate preset registry files under `configs/presets/optimizer/` with representative concat, standalone, Adam, and SGD values.
- [X] T002 [P] Update `configs/debug_matrix.yaml` to include explicit `model.correction_mode`, `model.membership_correction`, and `training.optimizer.preset: adam` fields used by the feature tests.
- [X] T003 [P] Update `configs/dmodel256_pilot_comparison.yaml` to include explicit `model.correction_mode`, `model.membership_correction`, and `training.optimizer.preset: adam` fields used by the feature tests.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core config and artifact plumbing required before any user story can be completed.

**CRITICAL**: No user story work should proceed until this phase is complete.

- [ ] T004 [P] Extend `utils/config.py` with resolved `model.correction_mode` handling, `model.membership_correction` validation, family-folder key resolution hooks, and preset registry parsing/loading/validation scaffolding.
- [ ] T005 [P] Extend `utils/metrics.py` so `build_run_summary()` and saved artifacts can carry correction mode, family-folder rule, active size label, preset provenance, and preset registry path fields.

**Checkpoint**: Foundation ready - user story implementation can now begin.

---

## Phase 3: User Story 1 - Select Correction Mode for Concat Runs (Priority: P1)

**Goal**: Support `none`, `gmc`, and `lmc` correction modes for concat runs, with `lmc` applying membership correction only at optimizer-step time.

**Independent Test**: Run a synthetic concat model, take one optimizer step under each correction mode, and confirm that `lmc` changes effective learning rates without changing gradients or optimizer state.

### Verification for User Story 1

- [ ] T006 [P] [US1] Add correction-mode validation coverage in `tests/test_config.py` for `none`, `gmc`, `lmc`, and `model.membership_correction` conflict cases.
- [ ] T007 [P] [US1] Add a synthetic concat LMC regression and slicing-path non-regression in `tests/test_training_smoke.py` that checks per-block effective learning rates, unchanged gradients/optimizer state, and unchanged `none`/`gmc` behavior on slicing runs.

### Implementation for User Story 1

- [ ] T008 [US1] Apply LMC during the concat optimizer step in `training/run.py` using membership counts from `modified_llama.py`.
- [ ] T009 [US1] Thread the resolved correction mode through saved config and run-summary artifacts in `utils/config.py` and `utils/metrics.py`.

**Checkpoint**: User Story 1 should now be independently functional and testable.

---

## Phase 4: User Story 2 - Resolve Runs Into Shared Family Artifact Folders (Priority: P2)

**Goal**: Resolve standalone `s`, `m`, and `l` runs into the shared family folder for the largest configured size so comparison plots and downstream analysis do not require manual copying or renaming.

**Independent Test**: Resolve the same family config for standalone `s`, `m`, and `l` runs, verify they share the same folder key, and confirm the figure workflow can read the shared folder directly.

### Verification for User Story 2

- [ ] T010 [P] [US2] Add shared-family folder resolution coverage in `tests/test_config.py` for standalone `s`, `m`, and `l` runs.
- [ ] T011 [P] [US2] Add artifact and figure-generation smoke coverage for the shared folder in `tests/test_artifacts.py`.

### Implementation for User Story 2

- [ ] T012 [US2] Update `utils/config.py` to resolve `output_group` from the largest configured family size while preserving the existing family/token-budget components and recording the resolution rule.
- [ ] T013 [US2] Record the active size and family-folder provenance in `utils/metrics.py` so `config.json` and `run_summary.json` explain how the folder was chosen.

**Checkpoint**: User Stories 1 and 2 should now both work independently.

---

## Phase 5: User Story 3 - Reuse Config Sections With Presets (Priority: P3)

**Goal**: Define reusable section-scoped presets, starting with optimizer presets, so common config blocks can be referenced by name and partially overridden.

**Independent Test**: Resolve a config that selects `training.optimizer.preset=adam` and overrides one nested field, then confirm the preset values remain in effect except for the explicit override.

### Verification for User Story 3

- [ ] T014 [P] [US3] Add optimizer preset resolution coverage and partial-override cases in `tests/test_config.py`, including loading from `configs/presets/optimizer/*.yaml`.
- [ ] T015 [P] [US3] Add invalid preset and conflicting-preset validation coverage in `tests/test_config.py`.

### Implementation for User Story 3

- [ ] T016 [P] [US3] Switch example optimizer sections to `training.optimizer.preset: adam` in `configs/debug_matrix.yaml` and `configs/dmodel256_pilot_comparison.yaml`, referencing the preset registry files created in Phase 1 under `configs/presets/optimizer/`.
- [ ] T017 [US3] Implement section-scoped preset resolution from registry files and deep-merge precedence in `utils/config.py`.
- [ ] T018 [US3] Persist selected preset name, preset registry path, and final merged optimizer values in resolved config and run-summary artifacts in `utils/metrics.py`.

**Checkpoint**: All user stories should now be independently functional.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final cleanup and validation across the feature.

- [ ] T019 [P] Align `specs/004-experiment-config-resolution/quickstart.md` and the contracts under `specs/004-experiment-config-resolution/contracts/` with the final config and artifact field names after implementation.
- [ ] T020 [P] Run the focused validation suite from `specs/004-experiment-config-resolution/quickstart.md` and fix any regressions in `tests/`, `configs/`, `training/run.py`, `utils/config.py`, and `utils/metrics.py`.

---

## Dependencies & Execution Order

### Phase Dependencies

- Setup (Phase 1): No dependencies - can start immediately.
- Foundational (Phase 2): Depends on Setup completion - blocks all user stories.
- User Stories (Phase 3+): Depend on the Foundational phase.
- Polish (Final Phase): Depends on completion of the desired user stories.

### User Story Dependencies

- User Story 1 (P1): Can start after Foundational - no dependencies on other stories.
- User Story 2 (P2): Can start after Foundational - independent of User Story 1.
- User Story 3 (P3): Can start after Foundational - independent of User Stories 1 and 2.

### Within Each User Story

- Run verification tasks before or alongside implementation when it helps catch likely research failures.
- Keep configuration work before training-loop wiring.
- Keep runtime behavior before artifact assertions.
- Finish one story before moving to the next priority if working sequentially.

### Parallel Opportunities

- Tasks marked `[P]` can run in parallel when they touch different files and do not depend on incomplete work.
- Setup tasks T001-T003 can run in parallel.
- Foundational tasks T004-T005 can run in parallel.
- User Story 1 verification tasks T006-T007 can run in parallel.
- User Story 2 verification tasks T010-T011 can run in parallel.
- User Story 3 verification tasks T014-T015 can run in parallel.
- User Story 3 implementation tasks T016-T018 can be split across config, training, and artifact ownership.

---

## Implementation Strategy

### MVP First

1. Complete Setup and Foundational phases.
2. Complete User Story 1.
3. Stop and validate the concat LMC path independently.

### Incremental Delivery

1. Setup + Foundational -> shared plumbing ready.
2. Add User Story 1 -> concat correction behavior works.
3. Add User Story 2 -> shared family folders work for comparison runs.
4. Add User Story 3 -> reusable presets work for optimizer config.
5. Finish with polish and quickstart validation.

### Parallel Team Strategy

1. One engineer can own User Story 1 training-loop behavior.
2. One engineer can own User Story 2 artifact-path validation.
3. One engineer can own User Story 3 preset resolution and sample config updates.
4. All engineers share the foundational config and artifact plumbing.
