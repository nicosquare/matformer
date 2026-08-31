# Tasks: Per-Width Optimizer State

**Input**: Design documents from `/specs/012-per-width-optimizer-state/`
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests/Verification**: Focused tests are required because the specification defines exact optimizer ownership, failure, compatibility, experiment-control, and resume behavior.

**Organization**: Tasks are grouped by user story so each increment has an explicit independent test and can be reviewed against its own acceptance scenarios.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel with other tasks in the same phase once its prerequisites are complete because it changes a different file and has no unresolved dependency
- **[Story]**: Maps the task to User Story 1, 2, 3, or 4
- Every task names the exact repository file it changes or verifies

## Phase 1: Setup (Shared Experiment Structure)

**Purpose**: Add the reusable fixture that all optimizer-scope stories use without changing the historical shared-mode fixtures.

- [X] T001 Create a two-width CPU smoke configuration with AdamW/SGD override support, balanced global actions, and explicit shared/per-granularity scope inputs in `tests/fixtures/per_granularity_optimizer_smoke.yaml`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish backward-compatible configuration and identity contracts before any optimizer lifecycle changes.

**CRITICAL**: No user story implementation begins until configuration omission still resolves to historical shared behavior and the intervention identity can be represented without mutating the existing optimizer mapping.

- [X] T002 [P] Add failing tests for the omitted-scope shared default, accepted dotted overrides, preserved `training.optimizer == {name, kwargs}` shape, ordered global widths, and fixed global scheduler-clock identity in `tests/test_config.py`
- [X] T003 [P] Add failing preflight-output tests that require scope, ordered widths, optimizer contract, scheduler contract, sampling policy, data roles, and run budget to be visible in `tests/test_train_cli.py`
- [X] T004 Implement normalized `optimizer_state_contract` resolution, scope aliases, ordered-width canonicalization, and the historical shared default without altering the resolved optimizer mapping in `src/utils/config.py`
- [X] T005 Extend configuration-only and normal CLI preflight rendering to expose the normalized optimizer-state contract and honor dotted scope/clock overrides in `train.py`
- [X] T006 [P] Add failing tests for full run identity including scope and paired-control identity excluding only scope while retaining optimizer, scheduler, topology, data-role, budget, cadence, seed, and action-policy fields in `tests/test_reproducibility.py`
- [X] T007 Implement distinct full-run and paired-control signatures with historical missing-scope normalization to shared in `src/utils/reproducibility.py`

**Checkpoint**: Existing configurations resolve identically except for additive shared-state identity, and subsequent stories can rely on one normalized configuration contract.

---

## Phase 3: User Story 1 - Isolate optimizer histories by width (Priority: P1) MVP

**Goal**: Maintain one optimizer state machine per ordered global width over the same model parameters while exactly one selected owner commits each global step under one global learning-rate clock.

**Independent Test**: Alternate two widths with deliberately different gradients on a shared parameter under AdamW and SGD; verify both update the same model, their moments/ages diverge, wider-only parameters and all non-selected optimizer states remain bitwise unchanged, one width owns an entire accumulation window, and a pre-commit failure advances no state or successful-update count.

### Verification for User Story 1

- [X] T008 [P] [US1] Add focused AdamW and SGD tests for ordered optimizer construction, shared parameter identity, distinct shared-parameter histories, no state creation or weight decay on inactive parameters, present-zero-gradient semantics, and bitwise-frozen non-selected state in `tests/test_per_granularity_optimizer.py`
- [X] T009 [P] [US1] Add matched-step tests proving every width optimizer receives the same globally scheduled learning rate while local moments and bias-correction ages advance only on selection in `tests/test_global_sampling_windows.py`
- [X] T010 [P] [US1] Add accumulation and failure tests proving one action owns all microbatches, stale gradients cannot cross owners, skipped/pre-commit-failed attempts do not advance state, and a successful commit increments exactly one owner in `tests/test_accumulation.py`
- [X] T011 [P] [US1] Add normal-entrypoint smoke coverage for both supported optimizer families and for unchanged omitted-scope shared behavior in `tests/test_training_smoke.py`

### Implementation for User Story 1

- [X] T012 [US1] Implement the ordered `PerGranularityOptimizerCollection`, width entries, parameter-presence semantics, compact update accounting, and scalar-carrier `GlobalSchedulerClock` in `src/training/optimizer_state.py`
- [X] T013 [US1] Integrate single-owner accumulation windows, owner-only clipping/step/zeroing, global scheduler commits, learning-rate fan-out, and pre-commit cleanup at the irreversible optimizer-return boundary in `src/training/steps.py`
- [X] T014 [US1] Construct the shared optimizer unchanged or the ordered per-granularity collection plus one global clock, and thread that runtime through the existing train loop in `src/training/run.py`
- [X] T015 [US1] Reuse the same one-width ownership lifecycle during warmup and reject any warmup step that would apply multiple widths in `src/training/warmup.py`
- [X] T016 [US1] Complete CPU acceptance coverage for alternating widths, narrow-width inactive tensors, skipped commits, and shared-mode non-regression through the normal loop in `tests/test_training_smoke.py`

**Checkpoint**: User Story 1 is a usable MVP; optimizer isolation can be demonstrated without experiment reporting or exact-resume support.

---

## Phase 4: User Story 2 - Run a controlled shared-versus-per-width comparison (Priority: P2)

**Goal**: Run paired TinyStories-Instruct arms that differ only in optimizer-state scope and produce structured scientific outcomes plus operational-cost measurements.

**Independent Test**: Run paired short jobs from the same initialization, action sequence, batches, learning-rate sequence, token budget, and evaluation cadence; verify exact balance for `H=1`, only scope differs in the paired identity, update counts reconcile, resource costs are present, and reports label pilot/confirmatory evidence correctly.

### Verification for User Story 2

- [X] T017 [P] [US2] Add exact-resolution tests for the frozen TinyStories-Instruct model, four widths, balanced-cycle `H=1`, optimizer/scheduler values, fixed data roles, seeds, pilot budget, and confirmation budget in `tests/test_config.py`
- [X] T018 [P] [US2] Add artifact tests for per-record scope/selected-width fields, per-width successful-update/exposure reconciliation, attempted-versus-committed failures, wall time, peak accelerator memory, and resumable-checkpoint size in `tests/test_artifacts.py`
- [X] T019 [P] [US2] Add freeze/report tests for six-run manifest completeness, paired-signature matching, endpoint tables, resource columns, diagnostic/confirmatory claim labels, immutable manifests, and sealed holdout handling in `tests/test_reporting.py`
- [X] T020 [P] [US2] Add a paired short-run smoke test that compares action IDs, batch provenance, global learning rates, evaluation cadence, balanced per-width counts, and the single allowed scope difference in `tests/test_training_smoke.py`

### Implementation for User Story 2

- [X] T021 [P] [US2] Add the frozen width-250/500/750/1000 TinyStories-Instruct recipe with `d=64`, four layers/heads, context 128, vocabulary 2048, AdamW `lr=.008`, global cosine warmup 64, batch 64, correction none, seeds 42/43/44, and pilot/confirmation overrides in `configs/controlled_exps/tinystories_instruct_per_width_optimizers.yaml`
- [X] T022 [US2] Extend structured metric schemas and run-summary accumulation with scope, selected width, global scheduler position, attempted/committed status, per-width updates/exposures, wall time, peak memory, and checkpoint bytes in `src/utils/metrics.py`
- [X] T023 [US2] Emit ownership and commit-accounting fields from every ordinary or failed training attempt without changing sampling behavior or evaluation cadence in `src/training/steps.py`
- [X] T024 [US2] Finalize reconciled ownership counts and collect wall-time, peak-memory, and terminal resumable-checkpoint measurements in the structured run summary in `src/training/run.py`
- [X] T025 [P] [US2] Teach public reporting models to expose scope, ordered-width outcomes, primary/secondary endpoints, resource costs, and evidence labels while treating missing historical scope as shared in `src/evaluation/reporting.py`
- [X] T026 [P] [US2] Parse and validate resolved configs, metric streams, run summaries, checkpoint metadata, and explicit run directories needed for paired comparison in `src/evaluation/reporting_io.py`
- [X] T027 [US2] Implement paired-control validation, uniform mean loss, worst-width loss, all per-width outcomes, seed aggregation, resource summaries, and guarded matched-compute claims in `src/evaluation/reporting_impl.py`
- [X] T028 [US2] Implement the `freeze` subcommand to accept only explicit run directories, validate exactly six pilot or confirmation runs, record hashes/provenance, and write an immutable paired manifest in `scripts/analyze_tinystories_per_width_optimizer.py`
- [X] T029 [US2] Implement the `report` subcommand to consume only a frozen manifest, prohibit implicit holdout evaluation, and write machine-readable JSON/CSV endpoint and resource tables in `scripts/analyze_tinystories_per_width_optimizer.py`
- [X] T030 [US2] Document corpus audit, six-arm preflight, pilot execution, manifest freezing, holdout choice, confirmation execution, endpoint interpretation, and cost-reporting rules in `docs/tinystories-per-width-optimizer-experiment.md`

**Checkpoint**: User Stories 1 and 2 produce a controlled, auditable paired experiment whose scientific and operational results do not depend on terminal logs.

---

## Phase 5: User Story 3 - Configure only valid ownership modes (Priority: P3)

**Goal**: Accept only single-process policies that resolve exactly one complete global width per optimizer step from at least two unique widths, with actionable preflight failures for every invalid ownership topology.

**Independent Test**: Preflight deterministic, stochastic, and adaptive global policies successfully; reject standalone, nested-all, per-block, adaptive-per-block, distributed multi-process, empty/multi-width actions, one unique width, multi-width warmup, unknown scopes, and non-global clocks before model or optimizer mutation; verify omitted scope remains shared.

### Verification for User Story 3

- [X] T031 [P] [US3] Add a configuration eligibility matrix covering every accepted global policy and every rejected topology, action cardinality, width cardinality, scope, scheduler clock, and warmup combination in `tests/test_config.py`
- [X] T032 [P] [US3] Add CLI tests that require attributable preflight errors before training setup and successful resolved-identity output for each eligible policy in `tests/test_train_cli.py`
- [X] T033 [P] [US3] Add tests rejecting per-granularity scope whenever distributed world size exceeds one while preserving existing distributed shared-mode behavior in `tests/test_distributed_sampling.py`

### Implementation for User Story 3

- [X] T034 [US3] Implement complete topology, unique-width, action-cardinality, scope, scheduler-clock, warmup, and single-process eligibility validation in `src/utils/config.py`
- [X] T035 [US3] Run all eligibility checks during CLI preflight before model, optimizer, data-loader, tracker, or output mutation and emit attributable failures in `train.py`
- [X] T036 [US3] Add runtime assertions that a per-granularity commit resolves one known global-width label and never silently remaps an empty, multiple, or unknown action in `src/training/optimizer_state.py`
- [X] T037 [US3] Enforce single-width warmup ownership at runtime as a defense against configurations that bypass normal preflight in `src/training/warmup.py`
- [X] T038 [US3] Preserve historical missing-scope artifact interpretation as shared while rejecting explicit unknown scopes in reporting loaders in `src/evaluation/reporting_io.py`

**Checkpoint**: Invalid ownership modes fail before mutation, valid global policies remain usable, and the historical shared mode stays compatible.

---

## Phase 6: User Story 4 - Resume and audit exact optimizer ownership (Priority: P4)

**Goal**: Save versioned, purpose-aware training checkpoints that restore every width-local optimizer history, the global scheduler clock, sampling state, and ownership accounting exactly.

**Independent Test**: Compare uninterrupted training with resumes both inside and at a width-sampling boundary; require matching subsequent actions, parameters, optimizer/scheduler states, metrics, and counts, and reject scope, family/hyperparameter, clock, ordered-width, topology, version, missing/extra/reordered/malformed/non-finite state, and model-only checkpoint mismatches before mutation.

### Verification for User Story 4

- [X] T039 [P] [US4] Add exact-resume tests for inside-window and action-boundary interruption plus mutation-free rejection of every required optimizer, scope, scheduler, width-order, topology, version, shape, and finiteness mismatch in `tests/test_per_granularity_optimizer_resume.py`
- [X] T040 [P] [US4] Add checkpoint artifact tests distinguishing resumable-training from model-only evaluation payloads and requiring complete ordered optimizer states, clock state, counts, active owner, purpose, scope, and version in `tests/test_artifacts.py`
- [X] T041 [P] [US4] Add a normal CPU CLI resume smoke test that compares later action records, learning rates, reconciled counts, metrics, and final parameters with an uninterrupted run in `tests/test_train_cli.py`

### Implementation for User Story 4

- [X] T042 [US4] Implement complete ordered optimizer-collection serialization, finiteness/shape validation, active-owner state, global-clock state, and mutation-free staged restore in `src/training/optimizer_state.py`
- [X] T043 [US4] Add versioned checkpoint-purpose metadata and distinct resumable-training versus model-only schemas, preserving historical shared resumable payloads only where unambiguous, in `src/training/checkpointing.py`
- [X] T044 [US4] Validate the full resume contract before loading model or optimizer state, restore sampling/RNG/collection/clock/accounting state in dependency order, and reject cross-scope conversion in `src/training/run.py`
- [X] T045 [US4] Persist and restore accumulation-window owner identity and keep pre-commit failures out of optimizer, scheduler, exposure, and committed-step accounting in `src/training/steps.py`
- [X] T046 [US4] Reconcile restored and newly appended attempted steps, committed steps, scheduler position, optimizer updates, and width exposures without duplicate metric rows in `src/utils/metrics.py`
- [X] T047 [US4] Complete normal-entrypoint CPU acceptance coverage for resumable terminal checkpoints, model-only rejection, post-commit fatal behavior, and exact continuation in `tests/test_training_smoke.py`
- [X] T048 [US4] Add resume and checkpoint-purpose audit fields to structured reporting so incomplete or non-resumable runs cannot enter a paired manifest in `src/evaluation/reporting_io.py`

**Checkpoint**: All four stories are functional; per-width ownership can be stopped, resumed, and independently audited without reconstructing state from terminal output.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Validate the complete feature against its executable quickstart and compatibility envelope.

- [X] T049 [P] Run the focused config, optimizer-isolation, accumulation, scheduler, artifact, resume, reporting, distributed, and CPU smoke commands and correct any stale command or expected artifact descriptions in `specs/012-per-width-optimizer-state/quickstart.md`
- [X] T050 Run the full `pytest -q` compatibility suite, resolve feature-caused regressions without weakening assertions, and record the verified compatibility command in `specs/012-per-width-optimizer-state/quickstart.md`
- [X] T051 Audit the final configuration keys, submitted commands, pilot/confirmation budgets, sealed-holdout workflow, artifact paths, and evidence labels against implemented behavior in `docs/tinystories-per-width-optimizer-experiment.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: Starts immediately.
- **Foundational (Phase 2)**: Depends on T001 and blocks all user-story implementation.
- **User Story 1 (Phase 3)**: Depends on Phase 2 and is the MVP runtime capability.
- **User Story 2 (Phase 4)**: Depends on User Story 1 because its paired experiment exercises the isolated optimizer runtime.
- **User Story 3 (Phase 5)**: Its configuration and CLI work can start after Phase 2; T036-T037 should integrate after the User Story 1 lifecycle interfaces stabilize.
- **User Story 4 (Phase 6)**: Depends on User Story 1 and the finalized User Story 3 ownership contract; it does not depend on completing the paired report.
- **Polish (Phase 7)**: Depends on every story selected for delivery; the full compatibility run follows all focused checks.

### User Story Dependency Graph

```text
Setup -> Foundation -> US1 (MVP) -> US2
                         |          |
Foundation -> US3 -------+-----> US4
                                    |
                          All selected stories -> Polish
```

### Within Each User Story

- Write and observe the story's focused verification failures before implementing its lifecycle or artifact changes.
- Stabilize configuration and data contracts before wiring training or reporting consumers.
- Implement core state transitions before checkpoint, summary, and analysis integration.
- Run the independent test at the story checkpoint before beginning the next dependent story.
- Keep shared-mode compatibility assertions active throughout every phase.

### Parallel Opportunities

- T002, T003, and T006 can be authored in parallel; T004 then unblocks T005 and T007.
- T008-T011 can be authored in parallel before T012-T015 implement the US1 runtime.
- T017-T020 can be authored in parallel, and T021, T025, and T026 touch independent experiment/reporting files.
- T031-T033 can be authored in parallel; the US3 configuration/CLI path can proceed alongside US2 reporting after the foundation is stable.
- T039-T041 can be authored in parallel; checkpoint schema work in T043 can proceed alongside collection serialization in T042 once their payload contract is agreed.
- T049 can run independently of the documentation audit in T051 after implementation is complete.

---

## Parallel Examples

### User Story 1

```text
T008: tests/test_per_granularity_optimizer.py
T009: tests/test_global_sampling_windows.py
T010: tests/test_accumulation.py
T011: tests/test_training_smoke.py
```

After those tests define the contract, T014 (`src/training/run.py`) and T015 (`src/training/warmup.py`) can be prepared in parallel once T012's runtime interface is stable, with T013 providing their step lifecycle integration.

### User Story 2

```text
T017: tests/test_config.py
T018: tests/test_artifacts.py
T019: tests/test_reporting.py
T020: tests/test_training_smoke.py
T021: configs/controlled_exps/tinystories_instruct_per_width_optimizers.yaml
```

Once metric fields are stable, T025 and T026 can implement public reporting models and artifact parsing in parallel before T027 aggregates endpoints.

### User Story 3

```text
T031: tests/test_config.py
T032: tests/test_train_cli.py
T033: tests/test_distributed_sampling.py
```

### User Story 4

```text
T039: tests/test_per_granularity_optimizer_resume.py
T040: tests/test_artifacts.py
T041: tests/test_train_cli.py
```

After the tests define the payload, T042 and T043 can proceed in parallel, followed by T044-T046 integration and T047 acceptance verification.

---

## Implementation Strategy

### MVP First: User Story 1

1. Complete T001-T007 to establish compatible configuration and identity.
2. Complete T008-T016 to implement isolated optimizer histories and one global schedule.
3. Stop and run the User Story 1 independent test across AdamW, SGD, accumulation, failure, and shared compatibility.
4. Treat this checkpoint as the smallest usable feature increment; experiment automation and exact resume can follow separately.

### Incremental Delivery

1. **Foundation**: historical configs stay shared and new scope identity is inspectable.
2. **US1**: per-width optimizer state works through the normal training loop.
3. **US2**: paired research runs and reports become reproducible and auditable.
4. **US3**: every unsupported ownership topology fails before mutation.
5. **US4**: exact resume and checkpoint auditability complete the production workflow.
6. **Polish**: focused and full-suite verification confirm the final compatibility envelope.

### Verification Gates

- Do not claim US1 complete unless non-selected states are bitwise unchanged and both optimizer families pass.
- Do not claim US2 complete unless six-arm preflight, paired signature, update reconciliation, endpoint tables, and cost fields all pass.
- Do not claim US3 complete unless all specified invalid modes fail before setup while omitted scope stays shared.
- Do not claim US4 complete unless boundary and inside-window resumes match uninterrupted training and all mismatch loads are mutation-free.

## Notes

- `[P]` indicates file-level parallelism, not permission to skip stated phase prerequisites.
- Tests intentionally precede implementation within each story where they define exact state or artifact semantics.
- The frozen pilot uses 87,132 global steps and 21,783 updates per width; confirmation uses 261,396 global steps and 65,349 updates per width.
- Model-only evaluation checkpoints remain valid evaluation artifacts but never satisfy resumable-training tasks.
- Commit after each task or cohesive task group, keeping user-story checkpoints independently reviewable.
