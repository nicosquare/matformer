# Tasks: Source Layout Housekeeping

**Input**: Design documents from `/specs/008-src-layout-housekeeping/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests/Verification**: Include focused smoke checks for import resolution, wrapper execution, training flow, artifact generation, and module boundaries.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Experiment Structure)

**Purpose**: Create the source-package layout and packaging plumbing needed for all later work

- [X] T001 Create `pyproject.toml` at the repository root with editable-install support for the `src/` layout and the preserved package names
- [X] T002 Create the `src/` package root and move package initialization files into `src/models/__init__.py`, `src/training/__init__.py`, `src/evaluation/__init__.py`, and `src/utils/__init__.py`
- [X] T003 [P] Update import resolution in `train.py` and `scripts/make_figures.py` so wrappers can import from `src/` without path hacks
- [X] T004 [P] Add a quick import smoke check in `tests/test_src_layout.py` to verify `models`, `training`, `evaluation`, and `utils` resolve from `src/`

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared package moves and module splits that block all user stories

- [X] T005 Move shared configuration logic into `src/utils/config.py` and `src/utils/config_resolution.py`, keeping the current config semantics intact
- [X] T006 Move shared metrics serialization and summary helpers into `src/utils/metrics.py`, `src/utils/metrics_io.py`, and `src/utils/metrics_summary.py`
- [ ] T007 [P] Update `src/utils/__init__.py` to re-export the config and metrics helpers needed by existing imports
- [ ] T008 Split figure/report helper logic out of the standalone script into `src/evaluation/reporting.py` and `src/evaluation/reporting_styles.py`
- [ ] T009 Update the stable CLI contract in `specs/008-src-layout-housekeeping/contracts/cli-entrypoints.md` if any wrapper-visible behavior needs clarification during implementation

## Phase 3: User Story 1 - Find the real code in one place (Priority: P1)

**Goal**: Make the repository importable from `src/` while keeping the current package names and thin root wrappers

**Independent Test**: `python -m pip install -e .` succeeds, and a smoke import confirms `models`, `training`, `evaluation`, and `utils` resolve from `src/`

### Verification for User Story 1

- [ ] T010 [P] [US1] Add or update `tests/test_src_layout.py` to assert the importable production packages resolve from `src/`
- [ ] T011 [P] [US1] Add or update `tests/test_training_smoke.py` to confirm `python train.py --config ...` still reaches the config-driven flow after the layout move

### Implementation for User Story 1

- [ ] T012 [US1] Update `train.py` to remain a thin wrapper that imports the training entrypoint from `src/training/run.py`
- [ ] T013 [US1] Update `scripts/make_figures.py` to remain a thin wrapper that imports the figure entrypoint from `src/evaluation/reporting.py`
- [ ] T014 [P] [US1] Move package imports used by `tests/` and internal modules to the preserved `src/models`, `src/training`, `src/evaluation`, and `src/utils` package tree
- [ ] T015 [US1] Update `README.md` and any root-level usage docs if they refer to the old source layout or import paths

## Phase 4: User Story 2 - Understand and modify training flow in smaller pieces (Priority: P2)

**Goal**: Split the oversized training module into focused units without changing training behavior

**Independent Test**: Representative config-driven training smoke runs still resolve config, initialize the model, and write the expected run artifacts

### Verification for User Story 2

- [ ] T016 [P] [US2] Add or update `tests/test_training_smoke.py` to cover the refactored training flow and representative artifact outputs
- [ ] T017 [P] [US2] Add or update `tests/test_artifacts.py` to confirm training summaries and metrics still serialize with the expected fields after the split

### Implementation for User Story 2

- [ ] T018 [US2] Move training orchestration from `training/run.py` into `src/training/run.py` and keep the public entrypoint stable
- [ ] T019 [P] [US2] Extract checkpoint and continuation handling into `src/training/checkpointing.py`
- [ ] T020 [P] [US2] Extract step-loop helpers and training step utilities into `src/training/steps.py`
- [ ] T021 [P] [US2] Extract warmup-specific logic into `src/training/warmup.py`
- [ ] T022 [P] [US2] Keep dataloader and distributed helpers in `src/training/data.py` and `src/training/distributed.py`, updating imports to use the new source tree
- [ ] T023 [US2] Update `src/training/__init__.py` to expose the training helpers needed by existing callers

## Phase 5: User Story 3 - Understand and modify figure generation in smaller pieces (Priority: P3)

**Goal**: Split the oversized figure-generation script into reusable reporting modules while preserving output behavior

**Independent Test**: Representative figure-generation runs still read existing CSV artifacts and write the same classes of figures and reports

### Verification for User Story 3

- [ ] T024 [P] [US3] Add or update `tests/test_pilot_comparison.py` or `tests/test_artifacts.py` to verify the refactored figure-generation path still reads and summarizes existing artifacts
- [ ] T025 [P] [US3] Add or update `tests/test_monitoring.py` or a focused reporting test to confirm metrics grouping and report generation still behave as expected

### Implementation for User Story 3

- [ ] T026 [US3] Move figure-generation orchestration out of `scripts/make_figures.py` into `src/evaluation/reporting.py`
- [ ] T027 [P] [US3] Extract reusable plot styling and figure constants into `src/evaluation/reporting_styles.py`
- [ ] T028 [P] [US3] Extract CSV artifact loading and metadata enrichment helpers into `src/evaluation/reporting_io.py`
- [ ] T029 [P] [US3] Extract plot-specific logic for scaling, validation, and consistency figures into focused helpers under `src/evaluation/`
- [ ] T030 [US3] Update `scripts/make_figures.py` to import the new reporting modules and keep the CLI surface unchanged

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final cleanup and consistency checks across the refactor

- [ ] T031 [P] Audit `src/utils/config.py`, `src/utils/metrics.py`, `src/training/run.py`, and `scripts/make_figures.py` to verify every in-scope production file is at or below 500 lines; if any file exceeds the limit, add a short rationale in the relevant module docstring or adjacent design note and record the exception explicitly
- [ ] T032 [P] Update `AGENTS.md` or root docs if any remaining source-layout references still point at the old layout
- [ ] T033 Run the quickstart validation commands from `specs/008-src-layout-housekeeping/quickstart.md` and fix any import or wrapper regressions
- [ ] T034 Verify `tests/test_src_layout.py`, `tests/test_training_smoke.py`, and the reporting tests all pass against the moved source tree
- [ ] T035 [P] Add or update `tests/test_layout_boundary.py` to assert notebooks, logs, outputs, paper assets, and reference material remain outside the `src/` move and are not relocated

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Setup completion and blocks all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational completion
- **User Story 2 (Phase 4)**: Depends on Foundational completion
- **User Story 3 (Phase 5)**: Depends on Foundational completion
- **Polish (Phase 6)**: Depends on completion of the desired user stories

### User Story Dependencies

- **US1**: Enables stable imports and wrappers; should land first
- **US2**: Can proceed after the shared package move, but should preserve the US1 wrapper/import contract
- **US3**: Can proceed after the shared reporting helpers are available, and should preserve the CLI contract

### Parallel Opportunities

- Setup tasks T003 and T004 can run in parallel
- Foundational tasks T007 and T008 can run in parallel
- US1 verification tasks T010 and T011 can run in parallel
- US2 extraction tasks T019, T020, T021, and T022 can run in parallel once T018 is complete
- US3 extraction tasks T027, T028, and T029 can run in parallel once T026 is complete

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. Stop and validate imports and wrappers before moving on

### Incremental Delivery

1. Land the `src/` layout and wrapper stability first
2. Split training code into smaller modules next
3. Split figure-generation/reporting code last
4. Finish with cleanup and line-count checks

### Parallel Team Strategy

- One developer can work on source-layout/packaging plumbing while another updates the thin wrappers
- After the foundation is complete, training and reporting splits can proceed in parallel
- Final validation can be split between import checks, training smoke runs, and figure-generation smoke runs
