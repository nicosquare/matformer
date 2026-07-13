# Tasks: Configurable Granularity Modes

**Input**: Design documents from `/specs/009-configurable-granularity-modes/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Organization**: Tasks are grouped by user story so each story can be implemented and tested independently.

## Phase 1: Setup (Shared Experiment Structure)

**Purpose**: Create the shared fixtures needed to validate the new granularity contract.

- [ ] T001 [P] Create an explicit-mode smoke config fixture at `tests/fixtures/explicit_granularity_smoke.yaml`
- [ ] T002 [P] Create a malformed-layout fixture at `tests/fixtures/explicit_granularity_invalid.yaml` for non-increasing and concat-misaligned layouts

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Add the shared plumbing that all granularity modes use before any story-specific work.

**Checkpoint**: No user-story work should start until the resolved granularity plumbing is in place.

- [ ] T003 Implement sequence-based granularity metadata helpers in `src/models/granularity.py` so prefix widths can be derived from any ordered label list
- [ ] T004 [P] Update `src/training/modeling.py` and `src/training/monitoring.py` to use `model.granularities` and `model.granularity_prefixes` when emitting artifacts and metrics
- [ ] T005 [P] Update `src/training/checkpointing.py` and `src/training/run.py` to persist the resolved granularity mode, ordered labels, prefix fractions, and derived prefix widths in checkpoint and run-summary artifacts
- [ ] T006 Update `src/training/steps.py` and `src/utils/config.py` to thread resolved granularity sequences through validation, adaptive sampling, and standalone granularity selection

---

## Phase 3: User Story 1 - Define custom granularity sets (Priority: P1)

**Goal**: Allow explicit ordered custom granularities and keep the resolved config as the source of truth.

**Independent Test**: Resolve `tests/fixtures/explicit_granularity_smoke.yaml`, confirm a five-label `0.2/0.4/0.6/0.8/1.0` layout is saved in the resolved config, and complete a smoke training run with the custom labels.

### Verification for User Story 1

- [ ] T007 [P] [US1] Add explicit-mode resolution assertions in `tests/test_config.py` for a five-label `0.2/0.4/0.6/0.8/1.0` layout
- [ ] T008 [P] [US1] Add a five-granularity smoke run in `tests/test_training_smoke.py` using `tests/fixtures/explicit_granularity_smoke.yaml`

### Implementation for User Story 1

- [ ] T009 [US1] Extend `src/utils/config.py` to accept `model.granularity_mode=explicit`, preserve the provided label order, and write resolved `model.granularities` and `model.granularity_prefixes`
- [ ] T010 [US1] Update `src/training/steps.py` and `src/training/run.py` so training, validation, checkpointing, reporting, and adaptive sampling read the resolved explicit label list instead of hard-coded canonical labels

**Checkpoint**: Explicit custom granularities should now be runnable and visible in resolved artifacts.

---

## Phase 4: User Story 2 - Preserve canonical compatibility (Priority: P2)

**Goal**: Keep existing canonical `s/m/l/xl` configs and pilot runners working without behavior changes.

**Independent Test**: Resolve the existing canonical pilot config, confirm the canonical label order and fractions are unchanged, and verify the pilot runner still launches the same canonical baselines.

### Verification for User Story 2

- [ ] T011 [P] [US2] Add canonical compatibility regression coverage in `tests/test_config.py` for unchanged `s/m/l/xl` resolution
- [ ] T012 [P] [US2] Add canonical artifact assertions in `tests/test_dmodel256_pilot.py` to verify saved resolved configs record canonical mode, the canonical label order, and derived widths

### Implementation for User Story 2

- [ ] T013 [P] [US2] Update `scripts/run_dmodel256_pilot.sh` to remove hard-coded standalone `s/m/l/xl` handling and derive allowed baselines from the resolved config
- [ ] T014 [P] [US2] Update `scripts/slurm_dmodel256_pilot.sh` and `scripts/queue_dmodel256_pilot.py` to mirror the resolved-granularity handling used by `scripts/run_dmodel256_pilot.sh`
- [ ] T015 [US2] Refresh `configs/dmodel256_pilot_comparison.yaml` comments and `scripts/run_dmodel256_pilot.sh` usage text so operators see the resolved-granularity contract

**Checkpoint**: Canonical runs and pilot launchers should still behave exactly as before.

---

## Phase 5: User Story 3 - Reject malformed or incompatible layouts (Priority: P3)

**Goal**: Fail fast with clear validation errors when explicit granularities are malformed or incompatible.

**Independent Test**: Try non-increasing fractions, non-positive fractions, duplicate labels, missing full-width final fractions, and concat-alignment failures, then confirm each one fails before training begins with a clear error.

### Verification for User Story 3

- [ ] T016 [P] [US3] Add malformed explicit-layout regression tests in `tests/test_config.py` for non-increasing, non-positive, and duplicate label cases
- [ ] T017 [P] [US3] Add concat-alignment failure coverage in `tests/test_config.py` and `tests/test_training_smoke.py` for explicit fractions that violate block alignment
- [ ] T018 [P] [US3] Add positive aligned-concat regression coverage in `tests/test_config.py` and `tests/test_training_smoke.py` for an explicit layout that satisfies block alignment

### Implementation for User Story 3

- [ ] T019 [US3] Harden `src/utils/config.py` validation messages for invalid final widths, missing explicit labels, and invalid `run.granularity` requests

**Checkpoint**: Malformed or incompatible layouts should now fail before any training work starts.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final cleanup and validation across the whole feature.

- [ ] T020 [P] Update `specs/009-configurable-granularity-modes/quickstart.md` commands or comments if validation paths change, and align any lingering references to canonical-only label assumptions in `scripts/` help text
- [ ] T021 [P] Run the targeted checks from `specs/009-configurable-granularity-modes/quickstart.md` and fix any remaining regressions in config resolution, scripts, or tests

---

## Dependencies & Execution Order

### Phase Dependencies

- Phase 1 has no dependencies and can start immediately.
- Phase 2 depends on Phase 1 and blocks all user-story work.
- Phases 3 to 5 depend on Phase 2.
- Phase 6 depends on the user-story phases being complete.

### User Story Dependencies

- User Story 1 can start after Phase 2 and is the MVP slice.
- User Story 2 can start after Phase 2 and should remain compatible with User Story 1 output.
- User Story 3 can start after Phase 2 and is independent of the other two stories.

### Within Each User Story

- Verification tasks can run in parallel where marked `[P]`.
- Configuration and plumbing should land before artifact or smoke-run updates.
- Story completion should be validated before starting the next priority story.

---

## Parallel Execution Examples

### User Story 1

```text
Task: T007 Add explicit-mode resolution assertions in tests/test_config.py
Task: T008 Add a five-granularity smoke run in tests/test_training_smoke.py
```

### User Story 2

```text
Task: T013 Update scripts/run_dmodel256_pilot.sh
Task: T014 Update scripts/slurm_dmodel256_pilot.sh and scripts/queue_dmodel256_pilot.py
```

### User Story 3

```text
Task: T016 Add malformed explicit-layout regression tests in tests/test_config.py
Task: T017 Add concat-alignment failure coverage in tests/test_config.py and tests/test_training_smoke.py
```

---

## Implementation Strategy

### MVP First

1. Complete Phase 1 and Phase 2.
2. Deliver User Story 1 so explicit custom granularities can run end to end.
3. Validate with the smoke test before touching broader compatibility work.

### Incremental Delivery

1. Add the shared resolved-granularity plumbing.
2. Enable explicit custom layouts and validate them.
3. Preserve canonical pilot and standalone behavior.
4. Add the malformed-layout failures and finish with a validation sweep.

### Notes

- `[P]` tasks can be done in parallel when they touch different files.
- Each user story has its own verification tasks so it can be validated independently.
- Keep the resolved config as the source of truth throughout implementation.
