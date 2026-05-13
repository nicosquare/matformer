# Tasks: MatFormer Language Model Reproduction

**Input**: Design documents from `/specs/001-matformer-lm-reproduction/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests/Verification**: Include focused smoke checks for config resolution,
FFN prefix slicing, artifact writing, non-embedding parameter counting, and
debug run wiring. These checks target research failure modes rather than broad
production coverage.

**Organization**: Tasks are grouped by user story so each phase produces an
independently testable research increment.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel when it touches different files and has no
  dependency on incomplete tasks.
- **[Story]**: Which user story the task belongs to.
- Every task includes exact file paths.

## Phase 1: Setup (Shared Experiment Structure)

**Purpose**: Create the shallow research layout, configuration stubs, and
dependency documentation needed by all later phases.

- [X] T001 Create shallow project directories `configs/`, `training/`, `evaluation/`, `utils/`, `scripts/`, `tests/`, and `outputs/`
- [X] T002 Add package marker files `training/__init__.py`, `evaluation/__init__.py`, and `utils/__init__.py`
- [X] T003 [P] Create dependency specification in `requirements.txt` for torch, transformers, datasets, pyyaml, pandas, matplotlib, pytest, and lm-eval
- [X] T004 [P] Create debug matrix config skeleton in `configs/debug_matrix.yaml`
- [X] T005 [P] Create 78M reduced-token pilot config skeleton in `configs/78m_reduced_pilot.yaml`
- [X] T006 [P] Create consistency evaluation config skeleton in `configs/consistency.yaml`
- [X] T007 [P] Create speculative evaluation config skeleton in `configs/speculative.yaml`
- [X] T008 [P] Update environment setup notes in `README.md`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build shared config, metric, model-size, dataset, and training
plumbing that every user story needs.

**CRITICAL**: No user story work can begin until this phase is complete.

- [X] T009 Implement YAML loading, CLI override merging, and resolved config writing in `utils/config.py`
- [X] T010 [P] Add config validation smoke checks in `tests/test_config.py`
- [X] T011 Implement CSV/JSON artifact writers for config, metrics, summaries, and failed-run summaries in `utils/metrics.py`
- [X] T012 [P] Add artifact writer smoke checks in `tests/test_artifacts.py`
- [X] T013 Implement non-embedding parameter counting in `utils/model_size.py`
- [X] T014 [P] Add non-embedding parameter counting checks in `tests/test_model_size.py`
- [X] T015 Update canonical MatFormer granularity definitions and FFN prefix metadata in `modified_llama.py`
- [X] T016 [P] Add FFN prefix ordering and tensor-shape checks in `tests/test_matformer_prefixes.py`
- [X] T017 Implement dataset loading and preprocessing helpers in `training/data.py`
- [X] T018 Implement shared validation loss and perplexity evaluation helpers in `evaluation/validation.py`
- [ ] T019 Implement shared training loop with config-driven output artifacts in `training/run.py`
- [ ] T020 Update `train.py` to accept `--config` and `--run-id` while preserving visible training flow
- [ ] T021 Implement run summary and baseline mismatch helpers in `training/baselines.py`
- [ ] T022 Implement CSV-driven plot generation for loss, perplexity, and consistency in `scripts/make_figures.py`

**Checkpoint**: Configs resolve, MatFormer prefixes are testable, one run can
write config/metrics/summary artifacts, and plots read structured CSV inputs.

---

## Phase 3: User Story 1 - Validate Nested MatFormer Training (Priority: P1)

**Goal**: Run a debug-size nested MatFormer proof-of-concept that evaluates
S/M/L/XL granularities, extracts metadata, writes structured outputs, and
includes at least one matched standalone comparison.

**Independent Test**: Run the debug nested configuration on a tiny dataset and
confirm saved config, validation loss, perplexity, extraction metadata, and at
least one baseline comparison artifact.

### Verification for User Story 1

- [ ] T023 [P] [US1] Add debug nested config resolution test in `tests/test_debug_matrix.py`
- [ ] T024 [P] [US1] Add tiny nested training smoke test with mocked or tiny data in `tests/test_training_smoke.py`
- [ ] T025 [P] [US1] Add extraction metadata artifact check in `tests/test_artifacts.py`

### Implementation for User Story 1

- [ ] T026 [US1] Implement all-granularity MatFormer loss accumulation in `training/run.py`
- [ ] T027 [US1] Implement nested granularity evaluation and extraction metadata writing in `training/run.py`
- [ ] T028 [US1] Add debug nested run values to `configs/debug_matrix.yaml`
- [ ] T029 [US1] Implement one matched standalone debug baseline path in `training/baselines.py`
- [ ] T030 [US1] Add nested-plus-one-baseline execution to `scripts/run_debug_matrix.sh`
- [ ] T031 [US1] Generate debug metrics and scaling summary rows from nested and one baseline outputs in `utils/metrics.py`
- [ ] T032 [US1] Document the P1 debug validation command in `specs/001-matformer-lm-reproduction/quickstart.md`

**Checkpoint**: User Story 1 is independently complete when
`scripts/run_debug_matrix.sh` can produce one nested debug run, one matched
standalone baseline comparison, `metrics.csv`, `scaling_results.csv`, and
`run_summary.json`.

---

## Phase 4: User Story 2 - Compare Against Standalone Baselines (Priority: P2)

**Goal**: Complete matched S/M/L/XL standalone baseline coverage for the
debug-size matrix and prepare the first paper-aligned 78M reduced-token pilot
labeling path.

**Independent Test**: Confirm S, M, L, and XL standalone runs exist for the
debug-size matrix and that 78M reduced-token pilot configs label token-budget
completion correctly.

### Verification for User Story 2

- [ ] T033 [P] [US2] Add standalone granularity config validation checks in `tests/test_config.py`
- [ ] T034 [P] [US2] Add baseline matching validation checks in `tests/test_baseline_matching.py`
- [ ] T035 [P] [US2] Add 78M completion label checks in `tests/test_config.py`

### Implementation for User Story 2

- [ ] T036 [US2] Implement standalone fixed-width model configuration path in `training/baselines.py`
- [ ] T037 [US2] Add standalone S, M, L, and XL entries to `configs/debug_matrix.yaml`
- [ ] T038 [US2] Extend `scripts/run_debug_matrix.sh` to run nested plus S/M/L/XL standalone debug matrix
- [ ] T039 [US2] Implement baseline match records with mismatch notes in `training/baselines.py`
- [ ] T040 [US2] Emit matched baseline rows with non-embedding parameters in `utils/metrics.py`
- [ ] T041 [US2] Add paper-aligned 78M reduced-token pilot values to `configs/78m_reduced_pilot.yaml`
- [ ] T042 [US2] Implement `scripts/run_78m_pilot.sh` with reduced-token pilot labeling
- [ ] T043 [US2] Update 78M pilot instructions in `specs/001-matformer-lm-reproduction/quickstart.md`

**Checkpoint**: User Story 2 is independently complete when debug-size
S/M/L/XL nested and standalone comparisons are present and 78M reduced-token
pilot output labels are correct.

---

## Phase 5: User Story 3 - Reproduce Scaling and Downstream Trends (Priority: P3)

**Goal**: Generate scaling reports and minimal downstream evaluation results
for nested and standalone runs.

**Independent Test**: Complete at least one medium-scale or reduced-token
scaling pass and generate loss/perplexity/average-accuracy summaries versus
non-embedding parameter count.

### Verification for User Story 3

- [ ] T044 [P] [US3] Add scaling result schema checks in `tests/test_artifacts.py`
- [ ] T045 [P] [US3] Add downstream task result schema checks in `tests/test_downstream.py`

### Implementation for User Story 3

- [ ] T046 [US3] Implement scaling summary aggregation in `evaluation/validation.py`
- [ ] T047 [US3] Implement minimal downstream suite adapter for HellaSwag, PIQA, ARC-Challenge, BoolQ, WinoGrande, and OpenBookQA in `evaluation/downstream.py`
- [ ] T048 [US3] Add downstream suite config values to `configs/78m_reduced_pilot.yaml`
- [ ] T049 [US3] Extend `scripts/make_figures.py` to generate loss_vs_size, ppl_vs_size, and accuracy_vs_size plots from `scaling_results.csv`
- [ ] T050 [US3] Add medium trend reporting helper in `scripts/make_figures.py`
- [ ] T051 [US3] Document downstream and scaling commands in `specs/001-matformer-lm-reproduction/quickstart.md`

**Checkpoint**: User Story 3 is independently complete when scaling CSV rows,
minimal downstream task rows, and figure artifacts can be generated from
structured outputs.

---

## Phase 6: User Story 4 - Measure Consistency and Elastic Behavior (Priority: P4)

**Goal**: Measure prediction alignment between nested granularities and record
mix-and-match layer-granularity patterns.

**Independent Test**: Evaluate at least one smaller/larger nested pair and one
heterogeneous layer pattern, then write `consistency_results.csv`.

### Verification for User Story 4

- [ ] T052 [P] [US4] Add token-level agreement metric checks in `tests/test_consistency.py`
- [ ] T053 [P] [US4] Add mix-and-match pattern validation checks in `tests/test_consistency.py`

### Implementation for User Story 4

- [ ] T054 [US4] Implement token-level argmax agreement in `evaluation/consistency.py`
- [ ] T055 [US4] Implement optional top-k overlap or KL placeholder output fields in `evaluation/consistency.py`
- [ ] T056 [US4] Implement mix-and-match layer granularity configuration in `modified_llama.py`
- [ ] T057 [US4] Add consistency and mix-and-match config values to `configs/consistency.yaml`
- [ ] T058 [US4] Write `consistency_results.csv` rows through `utils/metrics.py`
- [ ] T059 [US4] Extend `scripts/make_figures.py` to generate consistency_vs_size plots from `consistency_results.csv`

**Checkpoint**: User Story 4 is independently complete when consistency and
mix-and-match evaluations produce comparable nested and standalone alignment
artifacts.

---

## Phase 7: User Story 5 - Evaluate Speculative Decoding Alignment (Priority: P5)

**Goal**: Compare nested draft/verifier pairs against standalone draft/verifier
pairs and report acceptance, rollback, throughput, and latency.

**Independent Test**: Run one nested draft/verifier pair and one standalone
draft/verifier pair on the same prompt set and compare speculative metrics.

### Verification for User Story 5

- [ ] T060 [P] [US5] Add speculative result schema checks in `tests/test_speculative.py`
- [ ] T061 [P] [US5] Add prompt-set pairing checks in `tests/test_speculative.py`

### Implementation for User Story 5

- [ ] T062 [US5] Implement draft/verifier pair loading for nested and standalone models in `evaluation/speculative.py`
- [ ] T063 [US5] Implement acceptance rate and rollback frequency measurement in `evaluation/speculative.py`
- [ ] T064 [US5] Implement throughput and latency measurement in `evaluation/speculative.py`
- [ ] T065 [US5] Add speculative prompt-set and pair config values to `configs/speculative.yaml`
- [ ] T066 [US5] Write speculative metrics into `task_results.csv` or `run_summary.json` through `utils/metrics.py`
- [ ] T067 [US5] Document speculative evaluation commands in `specs/001-matformer-lm-reproduction/quickstart.md`

**Checkpoint**: User Story 5 is independently complete when nested and
standalone speculative decoding comparisons write acceptance, rollback,
throughput, and latency artifacts.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Tighten documentation, reproducibility, and artifact quality across
all completed stories.

- [ ] T068 Run focused smoke checks from `specs/001-matformer-lm-reproduction/quickstart.md`
- [ ] T069 Update run output examples and caveats in `README.md`
- [ ] T070 Verify all generated plots are reproducible from CSV inputs in `scripts/make_figures.py`
- [ ] T071 Remove unnecessary abstraction or unused helpers from `training/`, `evaluation/`, and `utils/`
- [ ] T072 Confirm no required metrics are terminal-only by inspecting `outputs/` and `utils/metrics.py`
- [ ] T073 Update final phase coverage notes in `specs/001-matformer-lm-reproduction/quickstart.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup; blocks all user stories.
- **US1 (Phase 3)**: Depends on Foundational.
- **US2 (Phase 4)**: Depends on US1 for shared nested run and at least one baseline path.
- **US3 (Phase 5)**: Depends on US2 for matched baseline matrix and 78M pilot labeling.
- **US4 (Phase 6)**: Depends on US2 for extracted nested and standalone comparisons.
- **US5 (Phase 7)**: Depends on US4 for alignment artifacts and model-pair conventions.
- **Polish (Phase 8)**: Depends on completed target stories.

### User Story Dependencies

- **US1**: MVP; validates nested training and one baseline comparison.
- **US2**: Extends US1 to full debug S/M/L/XL standalone matrix and 78M pilot labeling.
- **US3**: Adds scaling and downstream trend reporting after baseline matrix exists.
- **US4**: Adds consistency and mix-and-match evaluation after matched runs exist.
- **US5**: Adds speculative decoding after model-pair comparison conventions exist.

### Within Each User Story

- Verification tasks can run before or alongside implementation when they clarify expected artifacts.
- Config tasks precede scripts that consume those configs.
- Metric and artifact tasks precede plots and reports.
- Story checkpoint must pass before treating the next dependent story as complete.

---

## Parallel Opportunities

- Setup config skeletons T003-T008 can run in parallel.
- Foundational tests T010, T012, T014, and T016 can run in parallel after their target APIs are sketched.
- US1 verification tasks T023-T025 can run in parallel.
- US2 verification tasks T033-T035 can run in parallel.
- US3 verification tasks T044-T045 can run in parallel.
- US4 verification tasks T052-T053 can run in parallel.
- US5 verification tasks T060-T061 can run in parallel.
- Evaluation modules `evaluation/downstream.py`, `evaluation/consistency.py`, and `evaluation/speculative.py` are independent after foundational artifacts exist.

---

## Parallel Example: User Story 1

```bash
Task: "T023 Add debug nested config resolution test in tests/test_debug_matrix.py"
Task: "T024 Add tiny nested training smoke test with mocked or tiny data in tests/test_training_smoke.py"
Task: "T025 Add extraction metadata artifact check in tests/test_artifacts.py"
```

## Parallel Example: User Story 2

```bash
Task: "T033 Add standalone granularity config validation checks in tests/test_config.py"
Task: "T034 Add baseline matching validation checks in tests/test_baseline_matching.py"
Task: "T035 Add 78M completion label checks in tests/test_config.py"
```

## Parallel Example: User Story 3

```bash
Task: "T044 Add scaling result schema checks in tests/test_artifacts.py"
Task: "T045 Add downstream task result schema checks in tests/test_downstream.py"
```

## Parallel Example: User Story 4

```bash
Task: "T052 Add token-level agreement metric checks in tests/test_consistency.py"
Task: "T053 Add mix-and-match pattern validation checks in tests/test_consistency.py"
```

## Parallel Example: User Story 5

```bash
Task: "T060 Add speculative result schema checks in tests/test_speculative.py"
Task: "T061 Add prompt-set pairing checks in tests/test_speculative.py"
```

---

## Implementation Strategy

### MVP First (US1 Only)

1. Complete Phase 1 setup.
2. Complete Phase 2 foundational config, metrics, model-size, data, and training plumbing.
3. Complete Phase 3 User Story 1.
4. Stop and validate: run debug nested training plus one matched baseline comparison.

### Incremental Delivery

1. US1 validates the visible nested training flow and first comparison.
2. US2 completes the debug-size S/M/L/XL baseline matrix and 78M pilot label path.
3. US3 adds scaling/downstream reporting from structured artifacts.
4. US4 adds consistency and elastic behavior analysis.
5. US5 adds speculative decoding alignment.

### Validation Gates

- Before US1 completion: `metrics.csv`, `scaling_results.csv`, and `run_summary.json` exist for nested and one baseline run.
- Before US2 completion: S/M/L/XL debug standalone baselines exist and 78M reduced-token pilot labels are correct.
- Before US3 completion: plots derive from CSV files only.
- Before US4 completion: consistency metrics distinguish nested and standalone sources.
- Before US5 completion: speculative metrics include acceptance, rollback, throughput, and latency.
