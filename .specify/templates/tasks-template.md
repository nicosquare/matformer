---

description: "Task list template for feature implementation"
---

# Tasks: [FEATURE NAME]

**Input**: Design documents from `/specs/[###-feature-name]/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests/Verification**: Include focused tests or smoke checks for failure modes
likely to waste research time, such as tensor-shape mistakes, broken training or
evaluation flow, missing outputs, and unreproducible configs. Broader tests are
optional unless explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Minimal flat research layout**: `train.py`, `modified_llama.py`, `configs/`,
  `outputs/`
- **Shallow research layout**: `models/`, `experiments/`, `training/`,
  `evaluation/`, `utils/`, `configs/`, `scripts/`, `outputs/`, `tests/`
- Paths shown below assume a research-code repository - adjust based on
  plan.md structure

<!-- 
  ============================================================================
  IMPORTANT: The tasks below are SAMPLE TASKS for illustration purposes only.
  
  The /speckit-tasks command MUST replace these with actual tasks based on:
  - User stories from spec.md (with their priorities P1, P2, P3...)
  - Feature requirements from plan.md
  - Entities from data-model.md
  - Endpoints from contracts/
  
  Tasks MUST be organized by user story so each story can be:
  - Implemented independently
  - Tested independently
  - Delivered as an MVP increment
  
  DO NOT keep these sample tasks in the generated tasks.md file.
  ============================================================================
-->

## Phase 1: Setup (Shared Experiment Structure)

**Purpose**: Create only the files and directories needed for the experiment

- [ ] T001 Create or update shallow project structure per implementation plan
- [ ] T002 Define explicit configuration inputs for [experiment/feature]
- [ ] T003 [P] Create output directory conventions for metrics, configs, plots, and checkpoints

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core research-code pieces that MUST be complete before ANY story can be implemented

**CRITICAL**: No user story work can begin until this phase is complete

Examples of foundational tasks (adjust based on your project):

- [ ] T004 Implement shared model/training/evaluation code needed by all stories
- [ ] T005 [P] Add focused tensor-shape or smoke checks for shared experiment flow
- [ ] T006 [P] Implement config saving and seed logging for each run
- [ ] T007 Implement structured metric writing to CSV or JSON
- [ ] T008 Configure readable console logging for key experiment events
- [ ] T009 Document dataset and preprocessing assumptions used by the feature

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - [Title] (Priority: P1) MVP

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Verification for User Story 1

> **NOTE**: Keep verification focused on the most useful research failure modes.

- [ ] T010 [P] [US1] Smoke check for [training/evaluation path] in tests/test_[name].py
- [ ] T011 [P] [US1] Tensor-shape or output-artifact check for [component] in tests/test_[name].py

### Implementation for User Story 1

- [ ] T012 [P] [US1] Implement [model/experiment component] in [path]
- [ ] T013 [P] [US1] Add explicit config values for [research concept] in [path]
- [ ] T014 [US1] Integrate [component] into visible experiment flow in [path]
- [ ] T015 [US1] Write scalar metrics and run summary artifacts to outputs/[run-id]/
- [ ] T016 [US1] Add minimal checks for silent failure risks
- [ ] T017 [US1] Add concise logging for key user story 1 experiment events

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - [Title] (Priority: P2)

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Verification for User Story 2

- [ ] T018 [P] [US2] Smoke check for [training/evaluation path] in tests/test_[name].py
- [ ] T019 [P] [US2] Output-artifact check for [metrics/config/plot/checkpoint] in tests/test_[name].py

### Implementation for User Story 2

- [ ] T020 [P] [US2] Implement [model/experiment component] in [path]
- [ ] T021 [US2] Integrate [component] into visible experiment flow in [path]
- [ ] T022 [US2] Save config, metrics, and summaries for [experiment/feature]
- [ ] T023 [US2] Integrate with User Story 1 components if needed, keeping flow traceable

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - [Title] (Priority: P3)

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Verification for User Story 3

- [ ] T024 [P] [US3] Smoke check for [training/evaluation path] in tests/test_[name].py
- [ ] T025 [P] [US3] Output-artifact check for [metrics/config/plot/checkpoint] in tests/test_[name].py

### Implementation for User Story 3

- [ ] T026 [P] [US3] Implement [model/experiment component] in [path]
- [ ] T027 [US3] Integrate [component] into visible experiment flow in [path]
- [ ] T028 [US3] Save config, metrics, and summaries for [experiment/feature]

**Checkpoint**: All user stories should now be independently functional

---

[Add more user story phases as needed, following the same pattern]

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] TXXX [P] Documentation updates in docs/
- [ ] TXXX Code cleanup for readability and local reasoning
- [ ] TXXX Remove unnecessary abstraction introduced during iteration
- [ ] TXXX [P] Additional focused tests or smoke checks if requested
- [ ] TXXX [P] Generate comparison plots or CSV summaries if relevant
- [ ] TXXX Confirm metrics are not stored only in terminal logs
- [ ] TXXX Run quickstart.md validation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 -> P2 -> P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable

### Within Each User Story

- Verification can be written before or alongside implementation when it helps
  catch likely research failures
- Configuration before experiment wiring
- Model/training/evaluation changes before output summaries
- Core implementation before comparison plots and cleanup
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All verification tasks for a user story marked [P] can run in parallel
- Independent model, config, output, and analysis files marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all verification tasks for User Story 1 together:
Task: "Smoke check for [training/evaluation path] in tests/test_[name].py"
Task: "Tensor-shape or output-artifact check for [component] in tests/test_[name].py"

# Launch independent implementation tasks for User Story 1 together:
Task: "Implement [model/experiment component] in [path]"
Task: "Add explicit config values for [research concept] in [path]"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Run or demo the experiment slice if ready

### Incremental Delivery

1. Complete Setup + Foundational -> Foundation ready
2. Add User Story 1 -> Test independently -> Run/Demo (MVP)
3. Add User Story 2 -> Test independently -> Run/Demo
4. Add User Story 3 -> Test independently -> Run/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Keep verification focused on likely research failures
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
