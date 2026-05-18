# Tasks: MatFormer Language Model Reproduction

**Input**: Design documents from `/specs/001-matformer-lm-reproduction/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests/Verification**: Include focused smoke checks for config resolution,
FFN prefix slicing, artifact writing, disaggregated parameter counting,
checkpoint status/path reporting, pilot comparison rows, and debug run wiring.
These checks target research failure modes rather than broad production
coverage.

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
- [X] T005 [P] Create d_model=256 pilot comparison config skeleton in `configs/dmodel256_pilot_comparison.yaml`
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
- [X] T019 Implement shared training loop with config-driven output artifacts in `training/run.py`
- [X] T020 Update `train.py` to accept `--config` and `--run-id` while preserving visible training flow
- [X] T021 Implement run summary and baseline mismatch helpers in `training/baselines.py`
- [X] T022 Implement CSV-driven plot generation for loss, perplexity, and consistency in `scripts/make_figures.py`

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

- [X] T023 [P] [US1] Add debug nested config resolution test in `tests/test_debug_matrix.py`
- [X] T024 [P] [US1] Add tiny nested training smoke test with mocked or tiny data in `tests/test_training_smoke.py`
- [X] T025 [P] [US1] Add extraction metadata artifact check in `tests/test_artifacts.py`

### Implementation for User Story 1

- [X] T026 [US1] Implement all-granularity MatFormer loss accumulation in `training/run.py`
- [X] T027 [US1] Implement nested granularity evaluation and extraction metadata writing in `training/run.py`
- [X] T028 [US1] Add debug nested run values to `configs/debug_matrix.yaml`
- [X] T029 [US1] Implement one matched standalone debug baseline path in `training/baselines.py`
- [X] T030 [US1] Add nested-plus-one-baseline execution to `scripts/run_debug_matrix.sh`
- [X] T031 [US1] Generate debug metrics and scaling summary rows from nested and one baseline outputs in `utils/metrics.py`
- [X] T032 [US1] Document the P1 debug validation command in `specs/001-matformer-lm-reproduction/quickstart.md`

**Checkpoint**: User Story 1 is independently complete when
`scripts/run_debug_matrix.sh` can produce one nested debug run, one matched
standalone baseline comparison, `metrics.csv`, `scaling_results.csv`, and
`run_summary.json`.

---

## Phase 3.5: Output Storage Configuration (Cross-Cutting Blocker)

**Purpose**: Update the design artifacts and shared run plumbing so every
future phase can write experiment outputs outside the repository filesystem and
use external cache locations when local space or inodes are constrained.

**Independent Test**: Run a debug matrix command with a custom output root and
confirm nested and standalone run artifacts are created under
`<output_root>/<run_id>/`, while no required run artifact is written under the
repository `outputs/` directory.

- [X] T074 Update storage, configuration, and cache constraints in `specs/001-matformer-lm-reproduction/plan.md`
- [X] T075 [P] Update `run.output_root`, derived `run.output_dir`, explicit output-dir override, and external cache expectations in `specs/001-matformer-lm-reproduction/contracts/experiment-config.md`
- [X] T076 [P] Update `OUTPUT_ROOT`, output-root argument behavior, and `<output_root>/<run_id>` artifact layout in `specs/001-matformer-lm-reproduction/contracts/cli.md` and `specs/001-matformer-lm-reproduction/contracts/run-artifacts.md`
- [X] T077 [P] Add output root and external cache fields or notes to `specs/001-matformer-lm-reproduction/data-model.md` and `specs/001-matformer-lm-reproduction/quickstart.md`
- [X] T078 [P] Add output root config resolution checks for matrix runs, single-run configs, explicit output-dir overrides, and unwritable roots in `tests/test_config.py`
- [X] T079 [P] Add runner smoke checks for `OUTPUT_ROOT` propagation and argument forwarding in `tests/test_debug_matrix.py`
- [X] T080 Implement configurable output-root resolution, default `outputs/`, explicit `run.output_dir` escape hatch, and early writable-root validation in `utils/config.py`
- [X] T081 Implement `OUTPUT_ROOT`, `--output-root`, and `--output-dir` support in `train.py`, `training/baselines.py`, and `scripts/run_debug_matrix.sh`
- [X] T082 Replace hardcoded output directories with output-root-compatible config values in `configs/dmodel256_pilot_comparison.yaml`, `configs/consistency.yaml`, and `configs/speculative.yaml`
- [X] T083 Document external output and Hugging Face cache environment variables in `README.md` and `specs/001-matformer-lm-reproduction/quickstart.md`
- [X] T084 Add external-output artifact smoke coverage proving required run artifacts stay under the configured root in `tests/test_training_smoke.py`

**Checkpoint**: Phase 4 may start only after runner commands and configs can
redirect outputs and caches away from the repository filesystem.

---

## Phase 4: User Story 2 - Compare Against Standalone Baselines (Priority: P2)

**Goal**: Complete matched S/M/L/XL standalone baseline coverage for the
debug-size matrix and prepare the d_model=256 MatFormer-Llama/SwiGLU pilot
shape and token-budget path.

**Independent Test**: Confirm S, M, L, and XL standalone runs exist for the
debug-size matrix and that d_model=256 pilot configs label token-budget
completion for the initial reduced-token pilot path.

### Verification for User Story 2

- [X] T033 [P] [US2] Add standalone granularity config validation checks in `tests/test_config.py`
- [X] T034 [P] [US2] Add baseline matching validation checks in `tests/test_baseline_matching.py`
- [X] T035 [P] [US2] Add existing reduced-token pilot completion-label checks in `tests/test_config.py`

### Implementation for User Story 2

- [X] T036 [US2] Implement standalone fixed-width model configuration path in `training/baselines.py`
- [X] T037 [US2] Add standalone S, M, L, and XL entries to `configs/debug_matrix.yaml`
- [X] T038 [US2] Extend `scripts/run_debug_matrix.sh` to run nested plus S/M/L/XL standalone debug matrix
- [X] T039 [US2] Implement baseline match records with mismatch notes in `training/baselines.py`
- [X] T040 [US2] Emit matched baseline rows with non-embedding parameters in `utils/metrics.py`
- [X] T041 [US2] Add initial d_model=256 reduced-token pilot shape and token-budget values to `configs/dmodel256_pilot_comparison.yaml`
- [X] T042 [US2] Implement `scripts/run_dmodel256_pilot.sh` with reduced-token pilot labeling
- [X] T043 [US2] Update d_model=256 pilot instructions in `specs/001-matformer-lm-reproduction/quickstart.md`

**Checkpoint**: User Story 2 is independently complete when debug-size
S/M/L/XL nested and standalone comparisons are present and d_model=256
reduced-token pilot completion labels are correct.

---

## Phase 4.5: Token-Budget-Derived Training Length (US2 Hardening)

**Purpose**: Make `training.token_budget` authoritative for budgeted training
runs by deriving planned training length from batch size, context length, and
effective distributed world size instead of relying on manually chosen
`max_steps` values.

**Independent Test**: Resolve the d_model=256 pilot config and confirm it records
`expected_tokens_per_step`, `derived_max_steps`, `effective_world_size`,
`token_budget`, and output labels consistently; run a small mocked or tiny-data
training smoke test and confirm the summary reports `tokens_seen` and a
deterministic `stop_reason`.

### Verification for Token-Budget-Derived Training Length

- [X] T085 [P] [US2] Add derived training length config checks for default world size and `WORLD_SIZE` handling in `tests/test_config.py`
- [X] T086 [P] [US2] Add run summary schema checks for `expected_tokens_per_step`, `derived_max_steps`, `effective_world_size`, and `stop_reason` in `tests/test_artifacts.py`
- [X] T087 [P] [US2] Add token-budget stop behavior smoke coverage with mocked or tiny data in `tests/test_training_smoke.py`

### Implementation for Token-Budget-Derived Training Length

- [X] T088 [P] [US2] Update derived training length fields and validation rules in `specs/001-matformer-lm-reproduction/data-model.md` and `specs/001-matformer-lm-reproduction/contracts/experiment-config.md`
- [X] T089 [P] [US2] Update run artifact expectations for budget-derived fields in `specs/001-matformer-lm-reproduction/contracts/run-artifacts.md`
- [X] T090 [US2] Implement effective world size and derived max-step resolution in `utils/config.py`
- [X] T091 [US2] Update budgeted training loop stopping and `stop_reason` calculation in `training/run.py`
- [X] T092 [US2] Emit token-budget-derived summary fields through `utils/metrics.py`
- [X] T093 [US2] Update d_model=256 pilot config and quickstart guidance for token-budget-derived step counts in `configs/dmodel256_pilot_comparison.yaml` and `specs/001-matformer-lm-reproduction/quickstart.md`

**Checkpoint**: Budgeted runs are ready for d_model=256 pilot execution only
when the resolved config and run summary expose derived training length,
effective world size, actual tokens seen, and stop reason.

---

## Phase 4.6: Distributed Pilot Execution and Runtime Observability (US2 Hardening)

**Purpose**: Prepare the d_model=256 reduced-token pilot for single-node multi-GPU
Slurm execution through the config-driven training path and add durable runtime
observability for long-running jobs.

**Independent Test**: Submit or dry-run a short single-node multi-GPU
d_model=256 pilot job and confirm the Slurm wrapper launches one process per
GPU, the resolved config records the active distributed world size and
budget-derived step count, shared artifacts are written only by rank 0, and
heartbeat stdout plus JSONL events identify the current stage and progress.

### Verification for Distributed Pilot Execution and Runtime Observability

- [X] T094 [P] [US2] Add single-node multi-GPU Slurm launcher command checks in `tests/test_dmodel256_pilot.py`
- [X] T095 [P] [US2] Add config-driven distributed/FSDP training smoke checks in `tests/test_training_smoke.py`
- [X] T096 [P] [US2] Add rank-0-only shared artifact write checks in `tests/test_artifacts.py`
- [X] T097 [P] [US2] Add heartbeat JSONL schema, stdout line, and cadence checks in `tests/test_heartbeats.py`

### Implementation for Distributed Pilot Execution and Runtime Observability

- [X] T098 [P] [US2] Update single-node multi-GPU Slurm and heartbeat artifact expectations in `specs/001-matformer-lm-reproduction/contracts/cli.md` and `specs/001-matformer-lm-reproduction/contracts/run-artifacts.md`
- [X] T099 [P] [US2] Implement heartbeat JSONL and stdout event helpers in `utils/heartbeats.py`
- [X] T100 [P] [US2] Implement distributed runtime helpers for rank, local rank, world size, rank-0 checks, and barriers in `training/distributed.py`
- [X] T101 [US2] Wire config-driven distributed device selection, dataloading, and FSDP model wrapping in `training/run.py`
- [X] T102 [US2] Integrate effective `WORLD_SIZE` budget resolution with distributed launch metadata in `utils/config.py` and `training/run.py`
- [X] T103 [US2] Gate resolved config, metrics, summary, checkpoint, and heartbeat shared writes to rank 0 in `training/run.py` and `utils/metrics.py`
- [X] T104 [US2] Update `scripts/slurm_dmodel256_pilot.sh` for single-node multi-GPU resource requests and one config-driven process per GPU
- [X] T105 [US2] Instrument tokenizer loading, dataset loading/preprocessing, model initialization, FSDP wrapping, training, validation, checkpointing, and artifact-writing stages in `training/run.py`
- [X] T106 [US2] Document distributed d_model=256 pilot queueing and heartbeat inspection in `specs/001-matformer-lm-reproduction/quickstart.md` and `README.md`

**Checkpoint**: Phase 4.6 is complete when the d_model=256 pilot has a
single-node multi-GPU Slurm path, config-driven FSDP execution, rank-safe
artifact writes, and heartbeat observability suitable for scheduler logs.

---

## Phase 4.7: Pilot Terminology, Parameter Reporting, Checkpointing, and Comparison Scope (US2 Hardening)

**Purpose**: Align the existing pilot implementation with the clarified
d_model=256 MatFormer-Llama/SwiGLU scope before downstream phases consume pilot
artifacts.

**Independent Test**: Run a capped pilot comparison workflow and confirm
`nested-random`, `nested-all`, and standalone rows expose actual parameter
counts, sampling mode, token budget, effective world size, checkpoint
status/path, and mismatch notes.

### Verification for Pilot Terminology, Parameter Reporting, Checkpointing, and Comparison Scope

- [X] T107 [P] [US2] Add d_model=256 pilot terminology, `model_shape_label`, `table_reference_label`, and explicit shape-field config checks in `tests/test_config.py`
- [X] T108 [P] [US2] Add disaggregated parameter count, LM-head counting convention, and unavailable-component reason checks in `tests/test_model_size.py`
- [X] T109 [P] [US2] Add best-eval selection across multiple validation points plus final and no-checkpoint run summary checks in `tests/test_artifacts.py`
- [X] T110 [P] [US2] Add `nested-random`, `nested-all`, standalone, and omitted-standalone pilot comparison row checks with `run_status=omitted`, `omit_reason`, `model_family=standalone`, `granularity`, `sampling_mode=standalone`, token budget, effective world size when known, unavailable checkpoint path/status, and mismatch notes in `tests/test_pilot_comparison.py`
- [X] T111 [P] [US2] Add d_model=256 runner and Slurm wrapper checks in `tests/test_dmodel256_pilot.py`

### Implementation for Pilot Terminology, Parameter Reporting, Checkpointing, and Comparison Scope

- [X] T112 [US2] Add `model_shape_label`, `table_reference_label`, `sampling_mode`, explicit shape fields, and mismatch notes to `configs/dmodel256_pilot_comparison.yaml`
- [X] T113 [US2] Update `utils/config.py` to accept d_model=256 pilot labels while preserving MatLM table-row token-budget validation
- [X] T114 [US2] Extend parameter reporting in `utils/model_size.py` to emit total, embedding, LM-head, non-embedding, FFN, attention, and other non-embedding counts with explicit null/unavailable status and reason when optional components cannot be computed
- [X] T115 [US2] Emit actual implementation counts, LM-head counting convention, and actual-vs-MatLM-table mismatch notes through resolved configs, `run_summary.json`, `scaling_results.csv`, and pilot comparison rows via `utils/metrics.py` and `utils/config.py`
- [X] T116 [US2] Implement rank-0-safe best-eval checkpoint selection by validation loss or perplexity across multiple eval points plus final and no-checkpoint status handling in `training/run.py`
- [X] T117 [US2] Record checkpoint status, checkpoint path, checkpoint selection metric, and validation-disabled final/no-checkpoint status in `run_summary.json` through `utils/metrics.py`
- [X] T118 [US2] Add default pilot comparison orchestration and omitted-baseline row emission for `nested-random`, `nested-all`, and standalone S/M/L/XL runs with `run_status=omitted`, `omit_reason`, null/unavailable checkpoint fields, and mismatch notes in `scripts/run_dmodel256_pilot.sh`
- [X] T119 [US2] Update d_model=256 runner scripts in `scripts/run_dmodel256_pilot.sh` and `scripts/slurm_dmodel256_pilot.sh` for the preferred config name
- [X] T120 [US2] Update d_model=256 pilot comparison guidance in `specs/001-matformer-lm-reproduction/quickstart.md` and `README.md`
- [X] T121 [US2] Verify `requirements.txt` remains sufficient for rebuilding the `elasticnn` dependency environment

**Checkpoint**: Phase 4.7 is complete when d_model=256 pilot comparison
artifacts expose sampling mode, actual parameter counts, LM-head convention,
token budget, effective world size, checkpoint status/path, and mismatch notes
before downstream, consistency, or speculative evaluation tasks consume them.

---

## Phase 5: User Story 3 - Reproduce Scaling and Downstream Trends (Priority: P3)

**Goal**: Generate scaling reports and minimal downstream evaluation results
for nested and standalone runs.

**Independent Test**: Complete at least one medium-scale or reduced-token
scaling pass and generate loss/perplexity/average-accuracy summaries versus
non-embedding parameter count.

### Verification for User Story 3

- [X] T044 [P] [US3] Add scaling result schema checks in `tests/test_artifacts.py`
- [X] T045 [P] [US3] Add downstream task result schema checks in `tests/test_downstream.py`

### Implementation for User Story 3

- [X] T046 [US3] Implement scaling summary aggregation in `evaluation/validation.py`
- [X] T047 [US3] Implement minimal downstream suite adapter for HellaSwag, PIQA, ARC-Challenge, BoolQ, WinoGrande, and OpenBookQA in `evaluation/downstream.py`
- [X] T048 [US3] Add downstream suite config values to `configs/dmodel256_pilot_comparison.yaml`
- [X] T049 [US3] Extend `scripts/make_figures.py` to generate loss_vs_size, ppl_vs_size, and accuracy_vs_size plots from `scaling_results.csv`
- [X] T050 [US3] Add medium trend reporting helper in `scripts/make_figures.py`
- [X] T051 [US3] Document downstream and scaling commands in `specs/001-matformer-lm-reproduction/quickstart.md`

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

- [X] T052 [P] [US4] Add token-level agreement metric checks in `tests/test_consistency.py`
- [X] T053 [P] [US4] Add mix-and-match pattern validation checks in `tests/test_consistency.py`

### Implementation for User Story 4

- [X] T054 [US4] Implement token-level argmax agreement in `evaluation/consistency.py`
- [X] T055 [US4] Implement top-k overlap output fields and explicit deferred-metric notes for KL divergence in `evaluation/consistency.py`
- [X] T056 [US4] Implement mix-and-match layer granularity configuration in `modified_llama.py`
- [X] T057 [US4] Add consistency and mix-and-match config values to `configs/consistency.yaml`
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
- **Output Storage (Phase 3.5)**: Depends on US1; blocks US2 and all larger runs.
- **US2 (Phase 4)**: Depends on US1 and Output Storage for shared nested run, at least one baseline path, and external output control.
- **Token Budget Hardening (Phase 4.5)**: Depends on US2 and blocks real budgeted d_model=256 pilot execution.
- **Distributed Pilot Observability (Phase 4.6)**: Depends on Token Budget Hardening and blocks real single-node multi-GPU d_model=256 pilot execution.
- **Pilot Alignment (Phase 4.7)**: Depends on Distributed Pilot Observability and blocks US3, US4, and US5 until corrected pilot artifacts expose sampling mode, parameter counts, checkpoint status/path, and mismatch notes.
- **US3 (Phase 5)**: Depends on Phase 4.7-corrected pilot comparison artifacts for matched or explicitly omitted baseline rows, d_model=256 pilot labeling, budget-derived run lengths, parameter-count components, checkpoint status/path, and pilot execution artifacts.
- **US4 (Phase 6)**: Depends on Phase 4.7-corrected extracted nested and standalone comparisons with explicit sampling-mode labels and omitted-baseline markers when compute limits apply.
- **US5 (Phase 7)**: Depends on US4 plus Phase 4.7 best-eval/final checkpoint availability and missing-checkpoint conventions for alignment artifacts and model-pair comparisons.
- **Polish (Phase 8)**: Depends on completed target stories.

### User Story Dependencies

- **US1**: MVP; validates nested training and one baseline comparison.
- **Output Storage**: Cross-cutting blocker that preserves repository disk and inode capacity before larger Phase 4+ runs.
- **US2**: Extends US1 to full debug S/M/L/XL standalone matrix and d_model=256 pilot labeling.
- **Token Budget Hardening**: Clarifies US2 budgeted-run semantics before real d_model=256 or larger scaling runs.
- **Distributed Pilot Observability**: Hardens US2 execution for single-node multi-GPU Slurm pilot runs and long-running runtime inspection.
- **Pilot Alignment**: Corrects US2 pilot terminology, comparison scope, parameter reporting, and checkpoint persistence before later evaluation phases.
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
- Output storage contract updates T075-T077 can run in parallel after T074 is understood.
- Output storage tests T078-T079 can run in parallel before T080-T081 implementation.
- US2 verification tasks T033-T035 can run in parallel.
- Token budget documentation tasks T088-T089 can run in parallel with verification tasks T085-T087.
- Distributed pilot verification tasks T094-T097 can run in parallel.
- Distributed pilot helper tasks T099-T100 can run in parallel after T098 clarifies contracts.
- Pilot alignment verification tasks T107-T111 can run in parallel.
- Pilot alignment parameter reporting, checkpoint, and comparison-runner tasks T114-T118 can run in parallel after T112-T113 establish config labels.
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
Task: "T035 Add existing reduced-token pilot completion-label checks in tests/test_config.py"
```

## Parallel Example: Token-Budget-Derived Training Length

```bash
Task: "T085 Add derived training length config checks for default world size and WORLD_SIZE handling in tests/test_config.py"
Task: "T086 Add run summary schema checks for expected_tokens_per_step, derived_max_steps, effective_world_size, and stop_reason in tests/test_artifacts.py"
Task: "T087 Add token-budget stop behavior smoke coverage with mocked or tiny data in tests/test_training_smoke.py"
```

## Parallel Example: Distributed Pilot Execution and Runtime Observability

```bash
Task: "T094 Add single-node multi-GPU Slurm launcher command checks in tests/test_dmodel256_pilot.py"
Task: "T095 Add config-driven distributed/FSDP training smoke checks in tests/test_training_smoke.py"
Task: "T096 Add rank-0-only shared artifact write checks in tests/test_artifacts.py"
Task: "T097 Add heartbeat JSONL schema, stdout line, and cadence checks in tests/test_heartbeats.py"
```

## Parallel Example: Pilot Terminology, Parameter Reporting, Checkpointing, and Comparison Scope

```bash
Task: "T107 Add d_model=256 pilot terminology, model_shape_label, table_reference_label, and explicit shape-field config checks in tests/test_config.py"
Task: "T108 Add disaggregated parameter count, LM-head convention, and unavailable-component reason checks in tests/test_model_size.py"
Task: "T109 Add best-eval selection across multiple validation points plus final and no-checkpoint summary checks in tests/test_artifacts.py"
Task: "T110 Add nested-random, nested-all, standalone, and omitted-standalone pilot comparison row checks in tests/test_pilot_comparison.py"
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
4. Complete Phase 3.5 output storage configuration before any larger run.
5. Stop and validate: run debug nested training plus one matched baseline comparison with a custom output root.

### Incremental Delivery

1. US1 validates the visible nested training flow and first comparison.
2. Output Storage makes every later runner safe for filesystems with restricted space or inodes.
3. US2 completes the debug-size S/M/L/XL baseline matrix and d_model=256 pilot label path.
4. Token Budget Hardening makes `training.token_budget` authoritative before real d_model=256 or larger budgeted runs.
5. Distributed Pilot Observability prepares the d_model=256 pilot for single-node multi-GPU Slurm execution with rank-safe artifacts and heartbeat logs.
6. Pilot Alignment adds corrected terminology, comparison modes, parameter reporting, and checkpoint persistence before downstream work.
7. US3 adds scaling/downstream reporting from structured artifacts.
8. US4 adds consistency and elastic behavior analysis.
9. US5 adds speculative decoding alignment.

### Validation Gates

- Before US1 completion: `metrics.csv`, `scaling_results.csv`, and `run_summary.json` exist for nested and one baseline run.
- Before US2 start: custom output root runs place required artifacts under `<output_root>/<run_id>/` and avoid repository `outputs/`.
- Before US2 completion: S/M/L/XL debug standalone baselines exist and initial d_model=256 reduced-token pilot completion labels are correct.
- Before budgeted d_model=256 execution: resolved configs and summaries expose derived max steps, expected tokens per step, effective world size, tokens seen, and stop reason.
- Before distributed d_model=256 pilot execution: Slurm launcher checks, config-driven FSDP smoke coverage, rank-0-only artifact writes, and heartbeat JSONL/stdout checks pass.
- Before Phase 5: d_model=256 pilot comparison artifacts expose sampling mode, actual parameter counts, LM-head convention, token budget, effective world size, checkpoint status/path, and mismatch notes.
- Before US3 completion: plots derive from CSV files only.
- Before US4 completion: consistency metrics distinguish nested and standalone sources.
- Before US5 completion: speculative metrics include acceptance, rollback, throughput, and latency.
