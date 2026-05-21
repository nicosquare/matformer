# Tasks: Cat Llama Granularity Pipeline

**Input**: Design documents from `/specs/002-cat-llama/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/, quickstart.md

**Tests/Verification**: Include focused smoke checks for config resolution, variant selection, artifact writing, comparison labeling, and invalid-variant rejection.

**Organization**: Tasks are grouped by user story so each story can be implemented and tested independently.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Experiment Structure)

**Purpose**: Make the default model variant explicit in source configs so the new selector has a visible baseline.

- [X] T001 [P] Add `model.variant: matformer_llama` to the shared experiment configs that drive model construction in `configs/debug_matrix.yaml` and `configs/dmodel256_pilot_comparison.yaml`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Add shared config and artifact plumbing needed by every user story.

**Checkpoint**: Foundation ready - user story implementation can now begin.

- [X] T002 [P] Add `model.variant` parsing and default-resolution support in `utils/config.py`, including the canonical variant list for `matformer_llama` and `cat_llama`
- [X] T003 [P] Extend `utils/metrics.py` and `training/run.py` so resolved `config.json` and `run_summary.json` persist `model.variant` alongside the existing topology and granularity labels
- [X] T004 [P] Add baseline smoke coverage in `tests/test_config.py` and `tests/test_artifacts.py` for default variant resolution and artifact presence

---

## Phase 3: User Story 1 - Select Cat Llama Family (Priority: P1)

**Goal**: Run the existing experiment entry point with `cat_llama` selected through config override while keeping the baseline path unchanged.

**Independent Test**: Run a debug nested experiment with `--override model.variant=cat_llama` and confirm the resolved run summary records `cat_llama`.

### Implementation for User Story 1

- [X] T005 [P] [US1] Update `training/run.py` to choose `CatLlamaMLP` only when the resolved variant is `cat_llama`, and keep the default `ModifiedLlamaForCausalLM` path for `matformer_llama`
- [X] T006 [US1] Update the legacy direct execution path in `train.py` so it respects the same `model.variant` default and stays consistent with the baseline behavior
- [X] T007 [US1] Add a config-driven smoke test in `tests/test_training_smoke.py` that runs a debug nested job with `--override model.variant=cat_llama` and asserts the selected variant reaches the resolved run summary

---

## Phase 4: User Story 2 - Compare Families Consistently (Priority: P2)

**Goal**: Keep baseline and cat-llama runs directly comparable through the same structured outputs.

**Independent Test**: Run one baseline job and one cat-llama job with the same seed and dataset settings, then confirm the output schema stays aligned and the variant labels distinguish the runs.

### Implementation for User Story 2

- [X] T008 [P] [US2] Keep comparison outputs stable in `utils/metrics.py` and ensure `model.variant` is visible in the resolved summary fields used to compare baseline and cat runs
- [X] T009 [US2] Add comparison assertions in `tests/test_pilot_comparison.py` or `tests/test_artifacts.py` that baseline and cat runs share the same artifact schema while remaining distinguishable by variant metadata

---

## Phase 5: User Story 3 - Fail Fast on Invalid Selection (Priority: P3)

**Goal**: Reject unsupported model variants before training starts.

**Independent Test**: Launch config resolution with an unsupported `model.variant` value and confirm it fails immediately with a clear config error.

### Implementation for User Story 3

- [X] T010 [P] [US3] Harden `utils/config.py` to reject unsupported `model.variant` values before training starts and to surface a clear config error
- [X] T011 [US3] Add a negative-path test in `tests/test_config.py` that validates an invalid `model.variant` override fails fast with the expected error

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Finish docs alignment and run the focused validation path.

- [X] T012 [P] Update `specs/002-cat-llama/quickstart.md`, `specs/002-cat-llama/contracts/cli.md`, `specs/002-cat-llama/contracts/experiment-config.md`, and `specs/002-cat-llama/contracts/run-artifacts.md` if any implemented field names or command examples need to be kept in sync
- [X] T013 [P] Run the focused smoke checks documented in `specs/002-cat-llama/quickstart.md` against `tests/test_config.py`, `tests/test_training_smoke.py`, and `tests/test_artifacts.py` to confirm the variant-specific path works end to end

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - blocks all user stories
- **User Stories (Phase 3+)**: Depend on Foundational completion
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Starts after Foundational; no dependency on later stories
- **User Story 2 (P2)**: Starts after Foundational; uses the same resolved variant metadata as US1
- **User Story 3 (P3)**: Starts after Foundational; independent negative-path validation

### Within Each User Story

- Config and validation changes before end-to-end smoke checks
- Model wiring before comparison assertions
- Artifact labeling before direct side-by-side comparison tests
- Negative-path validation before polish

## Parallel Opportunities

- `T001`, `T002`, and `T004` can run in parallel with different file sets once the feature docs are stable
- `T005` and `T006` can run in parallel if model wiring and legacy-path cleanup are split cleanly
- `T008` and `T009` can run in parallel because one updates artifact plumbing while the other adds comparison assertions
- `T010` and `T011` can run in parallel once the shared validation shape is in place
- `T012` and `T013` can run in parallel with late-stage verification

## Parallel Example: User Story 1

```bash
Task: "Update `training/run.py` variant selection for cat_llama"
Task: "Update `train.py` legacy direct execution path"
Task: "Add cat_llama smoke coverage in `tests/test_training_smoke.py`"
```

## Parallel Example: User Story 2

```bash
Task: "Keep comparison outputs stable in `utils/metrics.py`"
Task: "Add schema/metadata comparison assertions in `tests/test_pilot_comparison.py`"
```

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. Stop and validate the cat-llama path independently

### Incremental Delivery

1. Setup + Foundational -> shared baseline plumbing
2. User Story 1 -> selectable cat-llama path through the existing entry point
3. User Story 2 -> comparison-ready outputs and labels
4. User Story 3 -> fast failure on unsupported variants
5. Polish -> docs alignment and focused smoke verification

### Suggested MVP Scope

- User Story 1 only: explicit `cat_llama` selection through the existing config-driven experiment path, with baseline behavior preserved
