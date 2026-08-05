# Tasks: Probabilistic Adaptive Granularity

**Input**: Design documents from `/specs/010-probabilistic-adaptive-granularity/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests/Verification**: The specification requires controlled mathematical, data-isolation, boundary/resume, artifact, and compatibility verification. Test tasks therefore precede implementation tasks within each user story.

**Organization**: Tasks are grouped by user story so Bayesian global adaptation is deliverable as the MVP before additive per-block adaptation and provenance-safe comparison are completed.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel after its phase prerequisites because it changes different files and does not depend on another incomplete task in the same group
- **[Story]**: Maps implementation and verification to User Story 1, 2, or 3
- Every task names the exact repository file or files it changes or verifies

## Phase 1: Setup (Shared Experiment Structure)

**Purpose**: Add opt-in, deterministic experiment inputs without changing the default pilot matrix

- [ ] T001 [P] Add the valid Bayesian global smoke configuration with explicit prior, noise, controller-panel, and final-holdout inputs in tests/fixtures/probabilistic_adaptive_global_smoke.yaml
- [ ] T002 [P] Add the valid Bayesian additive per-block smoke configuration using the same controller and reward contract in tests/fixtures/probabilistic_adaptive_per_block_smoke.yaml
- [ ] T003 [P] Add an old-shaped Thompson configuration fixture that must produce a migration error in tests/fixtures/legacy_thompson_config.yaml

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish strategy resolution, deterministic role isolation, and reusable controller evaluation before either Bayesian scope is wired into training

**CRITICAL**: No user story implementation begins until configuration and four-role data provenance are valid and existing non-Bayesian paths remain isolated.

- [ ] T004 [P] Add configuration-contract tests for adaptive scopes, the Thompson migration error, the 50-step default, covariance/noise validation, fixed zero costs, arbitrary granularity labels, and unchanged UCB resolution in tests/test_config.py
- [ ] T005 [P] Add deterministic four-role split tests covering exact 128/512 counts, all six empty intersections, stable manifests/hashes, insufficient usable data, overlap rejection, and ordinary-validation preservation in tests/test_data_validation.py
- [ ] T006 [P] Add independent controller-panel, final-holdout, and posterior-sampling seed-stream reproducibility tests in tests/test_reproducibility.py
- [ ] T007 Implement strategy-specific Bayesian configuration normalization and preflight validation while preserving the legacy UCB configuration path in src/utils/config.py
- [ ] T008 [P] Add named deterministic seed streams and include their resolved provenance in reproducibility signatures in src/utils/reproducibility.py
- [ ] T009 Carry stable source-example identities through preprocessing and implement the Bayesian-only ordinary-validation/controller/final-holdout/training partition plus role and parent manifests in src/training/data.py
- [ ] T010 [P] Expose a fixed-panel objective that evaluates every resolved global granularity in deterministic order using finite target-token-weighted loss semantics in src/evaluation/validation.py
- [ ] T011 Integrate pre-optimizer role creation, disjointness validation, manifest writes, resolved-config rewrite, and resume hash checks without constructing a final-holdout evaluator in src/training/run.py

**Checkpoint**: Bayesian configuration and four pairwise-disjoint data roles can be resolved and audited before any optimizer update, while UCB and nonadaptive runs retain their existing two-role flow.

---

## Phase 3: User Story 1 - Learn Global Training Utility from Heldout Rewards (Priority: P1) MVP

**Goal**: Select one global granularity per fixed decision window using a genuine Gaussian posterior Thompson controller updated only from the fixed controller-panel objective.

**Independent Test**: Run a controlled global experiment and a matched inside-window resume, then verify initial and completed boundary objectives, exact per-step rewards, posterior updates, sampled actions, persisted state, non-finite failures, and fresh/resumed equivalence without enabling per-block adaptation.

### Verification for User Story 1

- [ ] T012 [P] [US1] Add closed-form tests for global contrast features, identity prediction with zero and positive process noise, Gaussian conditioning, covariance validation, posterior sampling, deterministic ties, arbitrary labels, and the one-arm case in tests/test_probabilistic_controller.py
- [ ] T013 [P] [US1] Add lifecycle and resume tests for the initial boundary, exact and inside-window checkpoints, incomplete termination, missing/incompatible Bayesian state, manifest mismatch, exact manifest/window/RNG/sample-count/action equality, fresh/resumed objective/reward/posterior equality at `rtol=1e-6`, `atol=1e-8`, and a concise resume log containing source checkpoint, restored phase, window index, progress, and current action in tests/test_probabilistic_controller_resume.py
- [ ] T014 [P] [US1] Add distributed tests proving the fixed 128-example controller panel is partitioned without duplication or omission, target-token totals match single-process evaluation, ordered component losses and the reduced objective match within `rtol=1e-6`, `atol=1e-8`, rank zero owns posterior sampling/update, every rank receives identical controller state/action, shared artifacts are rank-zero-only, and nonzero ranks do not duplicate controller lifecycle logs in tests/test_distributed.py
- [ ] T015 [P] [US1] Add checkpoint and structured-artifact tests for every Bayesian phase, full posterior/RNG/provenance state, transactional failure records, controller JSONL events, and summaries in tests/test_artifacts.py
- [ ] T016 [P] [US1] Add a short adaptive-global integration test asserting the initial boundary emits no reward, each boundary invokes the controller evaluator exactly once, every resolved granularity is evaluated in order, the objective is the uniform mean of target-token-weighted losses, reward is `(pre_objective-post_objective)/h`, training-batch loss cannot affect reward, controller metrics cannot affect checkpoint selection, one action owns exactly each complete window, and rank-zero initial/completed logs contain method, scope, boundary step, window index, action, objectives, reward when available, prediction error, and uncertainty summary in tests/test_training_smoke.py

### Implementation for User Story 1

- [ ] T017 [US1] Implement the versioned float64 Gaussian belief, sum-to-zero global feature schema, finite/PSD validation, identity prediction, conditioning equations, dedicated deterministic sampling, and ordered tie-breaking in src/training/probabilistic_controller.py
- [ ] T018 [P] [US1] Apply the selected global action unchanged to every transformer block and count only successful optimizer updates toward its window in src/training/steps.py
- [ ] T019 [P] [US1] Persist and validate the Bayesian checkpoint schema, resolved prior mean/covariance, observation-noise variance, process-noise covariance, identity transition, intercept-only context, zero compute/switch costs, feature schema/version/hash, window phase/progress, objectives, posterior/predictive state, controller RNG/sample count, journal position, and resume provenance in src/training/checkpointing.py
- [ ] T020 [P] [US1] Add rank-zero controller-state validation and broadcast helpers using the existing distributed primitives in src/training/distributed.py
- [ ] T021 [P] [US1] Add append-only controller boundary/failure journal writing, summary generation, action-frequency counts, uncertainty summaries, compact ordinary metric fields, and compact controller lifecycle log formatting in src/utils/metrics.py and src/utils/artifact_io.py
- [ ] T022 [US1] Integrate the post-warmup initial objective, prediction/sample, active-window, completed-boundary transaction, reused objective baseline, next-action sequence, and rank-zero initial/completed-boundary console records in src/training/run.py
- [ ] T023 [US1] Enforce transactional non-finite/evaluation failures and terminal-incomplete handling without posterior updates or unused final action samples, and emit rank-zero incomplete/failure records without printing full posterior vectors or covariance matrices in src/training/probabilistic_controller.py and src/training/run.py
- [ ] T024 [US1] Run the focused global controller, resume, distributed, artifact, smoke, and captured-console-log verification suites in tests/test_probabilistic_controller.py, tests/test_probabilistic_controller_resume.py, tests/test_distributed.py, tests/test_artifacts.py, and tests/test_training_smoke.py

**Checkpoint**: User Story 1 is a complete Bayesian global MVP with fixed heldout rewards, genuine posterior Thompson selection, attributable failure, structured audit artifacts, and boundary-exact resume.

---

## Phase 4: User Story 2 - Learn Additive Per-Block Preferences (Priority: P2)

**Goal**: Reuse the P1 controller with identifiable additive block/granularity effects and select complete profiles without enumerating `|G|^B` arms or duplicating scalar rewards across chosen pairs.

**Independent Test**: Use at least two blocks and two granularities with controlled complete-profile rewards, then verify distinct inferred block/granularity preferences, retained uncertainty, sampled additive maximization, fixed profiles, one-block/one-label behavior, and no complete-profile enumeration.

### Verification for User Story 2

- [ ] T025 [P] [US2] Add controlled additive-feature and posterior tests for dimension `1+B(|G|-1)`, coefficient identifiability, divergent block preferences, arbitrary labels, one block, one granularity, deterministic ties, and non-duplicated reward conditioning in tests/test_probabilistic_controller.py
- [ ] T026 [P] [US2] Add an adaptive-per-block smoke and resume test that checks fixed complete profiles, shared P1 reward semantics, additive provenance, and absence of an enumerated profile table in tests/test_training_smoke.py and tests/test_probabilistic_controller_resume.py

### Implementation for User Story 2

- [ ] T027 [US2] Extend the feature schema and action optimizer with per-block sum-to-zero contrasts, stable coefficient identities, schema hashing, and decomposed `O(B|G|)` sampled profile selection in src/training/probabilistic_controller.py
- [ ] T028 [P] [US2] Apply and preserve one resolved granularity per transformer block throughout each decision window using the existing MatFormer profile wiring in src/training/steps.py
- [ ] T029 [P] [US2] Extend Bayesian checkpoint and controller-summary validation with additive coefficient identities, per-block/granularity frequencies, and effect uncertainty in src/training/checkpointing.py and src/utils/metrics.py
- [ ] T030 [US2] Run controlled additive, per-block smoke, and resume verification in tests/test_probabilistic_controller.py, tests/test_training_smoke.py, and tests/test_probabilistic_controller_resume.py

**Checkpoint**: User Story 2 independently demonstrates structured additive learning from complete-profile rewards while retaining the P1 controller protocol and persistence behavior.

---

## Phase 5: User Story 3 - Preserve Comparison and Audit Integrity (Priority: P3)

**Goal**: Make new Bayesian global/per-block runs, retained UCB, random baselines, and historical heuristic Thompson artifacts unambiguous, and expose the untouched final holdout only through post-training comparison.

**Independent Test**: Resolve valid and invalid Thompson configs, inspect new and historical artifact fixtures, run UCB/random/nested-all/standalone regressions, and invoke final-holdout evaluation only after completion while verifying it cannot affect controller state or checkpoint selection.

### Verification for User Story 3

- [ ] T031 [P] [US3] Add reporting tests for new Bayesian global/per-block labels, historical heuristic Thompson fallback identity, and unchanged UCB label/style behavior in tests/test_reporting.py
- [ ] T032 [P] [US3] Add end-to-end provenance assertions for controller/data/window/posterior/sampling/resume fields and historical checkpoint rejection in tests/test_artifacts.py
- [ ] T033 [P] [US3] Add post-training final-holdout tests for completion checks, manifest/checkpoint compatibility, deterministic all-granularity loss, explicit-checkpoint fallback, separate result hashing, and non-mutation of training artifacts in tests/test_phase2_finalize.py
- [ ] T034 [US3] Add regression tests proving old Thompson is not selectable; UCB behavior/state/resume is unchanged; and random global/per-block, nested-all, standalone, granularity resolution/model wiring, gradient-membership correction, baseline matching, nonadaptive checkpointing, and default-pilot behavior do not drift in tests/test_adaptive_sampler.py, tests/test_config.py, tests/test_training_smoke.py, tests/test_matformer_prefixes.py, tests/test_baseline_matching.py, tests/test_artifacts.py, and tests/test_pilot_comparison.py

### Implementation for User Story 3

- [ ] T035 [US3] Emit and verify method family/version/scope/feature provenance, resolved prior mean/covariance, observation-noise variance, process-noise covariance, identity transition, intercept-only context, zero compute/switch costs, and controller artifact links across config.json, checkpoints, controller_metrics.jsonl, controller_summary.json, ordinary metrics, and run_summary.json in src/utils/config.py, src/training/checkpointing.py, src/utils/metrics.py, and src/training/run.py
- [ ] T036 [P] [US3] Classify and style Bayesian global, Bayesian per-block, historical heuristic Thompson, and unchanged UCB artifacts from explicit provenance in src/evaluation/reporting.py, src/evaluation/reporting_io.py, and src/evaluation/reporting_styles.py
- [ ] T037 [P] [US3] Mirror provenance-based classifications in the compatibility reporting path without reinterpreting historical artifacts in src/evaluation/reporting_impl.py
- [ ] T038 [P] [US3] Remove legacy pseudo-Thompson from new-training dispatch while retaining its historical data interpretation and leaving UCB scoring/update/state behavior untouched in src/models/adaptive_sampler.py and src/training/steps.py
- [ ] T039 [P] [US3] Implement completed-run and manifest verification plus deterministic target-token-weighted final-holdout comparison without controller or checkpoint-selection mutation in src/evaluation/final_holdout.py
- [ ] T040 [US3] Add the post-training-only final comparison command with required run directory, optional explicit checkpoint, attributable exits, and final_holdout_results.json output in scripts/evaluate_final_holdout.py
- [ ] T041 [P] [US3] Add an explicit opt-in Bayesian pilot configuration while keeping the existing default comparison queue unchanged in configs/probabilistic_adaptive_granularity_smoke.yaml and scripts/queue_dmodel256_pilot.py
- [ ] T042 [US3] Reconcile completion/failure summaries and artifact hashes so controller_summary.json, run_summary.json, and final_holdout_results.json remain separately auditable in src/training/run.py and src/evaluation/final_holdout.py
- [ ] T043 [US3] Run reporting, final-holdout, migration, UCB, random-mode, model-wiring, correction, baseline-matching, nonadaptive-checkpoint, and pilot-compatibility verification in tests/test_reporting.py, tests/test_phase2_finalize.py, tests/test_artifacts.py, tests/test_adaptive_sampler.py, tests/test_config.py, tests/test_training_smoke.py, tests/test_matformer_prefixes.py, tests/test_baseline_matching.py, and tests/test_pilot_comparison.py

**Checkpoint**: All three user stories are independently verifiable, historical pseudo-Thompson remains reportable but not runnable, UCB remains unchanged, and the final holdout stays outside training decisions.

---

## Phase 6: Polish & Cross-Cutting Validation

**Purpose**: Exercise the complete method contract and confirm compatibility after all desired stories are present

- [ ] T044 Run the full focused Bayesian test set and resolve failures in tests/test_probabilistic_controller.py, tests/test_probabilistic_controller_resume.py, tests/test_data_validation.py, tests/test_config.py, tests/test_reproducibility.py, tests/test_distributed.py, tests/test_artifacts.py, tests/test_reporting.py, tests/test_phase2_finalize.py, and tests/test_training_smoke.py
- [ ] T045 Run the baseline compatibility matrix for UCB, random modes, nested-all, standalone, granularity resolution/model wiring, gradient-membership correction, baseline matching, nonadaptive checkpointing, reporting, and the default pilot queue, and resolve behavior drift in tests/test_adaptive_sampler.py, tests/test_config.py, tests/test_artifacts.py, tests/test_reporting.py, tests/test_training_smoke.py, tests/test_matformer_prefixes.py, tests/test_baseline_matching.py, and tests/test_pilot_comparison.py
- [ ] T046 Execute the global, resume, per-block, provenance, and final-holdout workflow and reconcile any documentation mismatch in specs/010-probabilistic-adaptive-granularity/quickstart.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; fixture files can be authored in parallel.
- **Foundational (Phase 2)**: Depends on Setup and blocks all user-story implementation.
- **User Story 1 (Phase 3)**: Depends on Foundational and is the minimum viable feature.
- **User Story 2 (Phase 4)**: Depends on the validated P1 controller because it extends the same posterior, reward, boundary, and resume machinery.
- **User Story 3 (Phase 5)**: Depends on completed P2 so provenance and dispatch changes do not race with additive per-block changes in shared training, checkpoint, and metrics files.
- **Polish (Phase 6)**: Depends on every user story selected for delivery.

### User Story Dependency Graph

```text
Setup -> Foundation -> US1 (Bayesian global MVP)
                         -> US2 (additive per-block)
                         -> US3 (provenance/comparison)
                         -> Polish
```

### Within Each User Story

- Add controlled verification cases before or alongside implementation so equations, split identities, and resume phases are executable contracts.
- Resolve feature schemas and controller state before training-loop integration.
- Integrate action application and distributed agreement before committing boundary artifacts.
- Complete checkpoint/resume and structured artifacts before declaring a story independently testable.
- Run the story-specific test group at its checkpoint before starting the next priority.

### Parallel Opportunities

- T001, T002, and T003 can proceed together because they create separate fixtures.
- T004, T005, and T006 can proceed together; T008 and T010 can also proceed independently after their expected contracts are understood.
- T012 through T016 can be authored together in distinct test files.
- After T017, T018 through T021 can proceed together in separate implementation files before T022 integrates them.
- T025 and T026 can be authored together; after T027, T028 and T029 can proceed together.
- After P2 is complete, T031 through T033 can be authored together; T034 follows T032 because both extend tests/test_artifacts.py. T036 through T039 plus T041 can proceed in parallel after T035 establishes explicit identity fields.
- Cross-story work proceeds in priority order `US1 -> US2 -> US3`; parallel markers apply only within the active phase and its stated prerequisites.

---

## Parallel Example: User Story 1

```text
Task T012: Write Gaussian/global feature tests in tests/test_probabilistic_controller.py
Task T013: Write boundary/resume tests in tests/test_probabilistic_controller_resume.py
Task T014: Write rank-zero agreement tests in tests/test_distributed.py
Task T015: Write controller artifact tests in tests/test_artifacts.py
Task T016: Write adaptive-global integration tests in tests/test_training_smoke.py

After T017:
Task T018: Wire fixed global actions in src/training/steps.py
Task T019: Extend Bayesian checkpoints in src/training/checkpointing.py
Task T020: Add controller broadcasts in src/training/distributed.py
Task T021: Add controller journal/summary writers in src/utils/metrics.py and src/utils/artifact_io.py
```

## Parallel Example: User Story 2

```text
Task T025: Write additive mathematics tests in tests/test_probabilistic_controller.py
Task T026: Write per-block integration/resume tests in tests/test_training_smoke.py and tests/test_probabilistic_controller_resume.py

After T027:
Task T028: Wire complete profiles in src/training/steps.py
Task T029: Extend additive checkpoint/summary fields in src/training/checkpointing.py and src/utils/metrics.py
```

## Parallel Example: User Story 3

```text
Task T031: Write reporting identity tests in tests/test_reporting.py
Task T032: Write audit/resume rejection tests in tests/test_artifacts.py
Task T033: Write final-holdout lifecycle tests in tests/test_phase2_finalize.py

After T032:
Task T034: Write baseline compatibility tests in tests/test_adaptive_sampler.py, tests/test_config.py, tests/test_training_smoke.py, tests/test_matformer_prefixes.py, tests/test_baseline_matching.py, tests/test_artifacts.py, and tests/test_pilot_comparison.py

After T035:
Task T036: Update modular reporting in src/evaluation/reporting.py, src/evaluation/reporting_io.py, and src/evaluation/reporting_styles.py
Task T037: Update compatibility reporting in src/evaluation/reporting_impl.py
Task T038: Isolate UCB and remove runnable pseudo-Thompson in src/models/adaptive_sampler.py and src/training/steps.py
Task T039: Implement final comparison in src/evaluation/final_holdout.py
Task T041: Add opt-in pilot surfaces in configs/probabilistic_adaptive_granularity_smoke.yaml and scripts/queue_dmodel256_pilot.py
```

---

## Implementation Strategy

### MVP First: User Story 1

1. Complete Setup and Foundational phases.
2. Implement and validate only Bayesian global adaptation in User Story 1.
3. Stop at T024 and demonstrate fixed-panel rewards, posterior response, genuine Thompson choices, and exact resume.
4. Do not require per-block adaptation or reporting polish to assess the controller dataset and delayed-reward method.

### Incremental Delivery

1. **Foundation**: Strategy migration safety plus deterministic four-role data isolation.
2. **P1**: Bayesian global controller, boundary state, artifacts, and resume become the MVP.
3. **P2**: Additive per-block effects reuse P1 without changing its reward protocol.
4. **P3**: Complete method provenance, historical reporting, UCB regression safety, opt-in pilots, and post-training final comparison.
5. **Polish**: Run focused and baseline matrices, then execute the documented workflow end to end.

### Research Safety Rules

- Do not silently fill Bayesian priors or noise for old Thompson configurations.
- Do not route any new Thompson run through the heuristic reward-mean update.
- Do not change UCB configuration, reward, selection, checkpoint, resume, label, or style semantics.
- Do not consume controller or final-holdout examples in another role.
- Do not update a posterior from training-batch loss, a failed boundary, or an incomplete terminal window.
- Do not evaluate the final holdout from training or use it to select a checkpoint.
- Do not claim scientific superiority from controlled unit tests or smoke runs.

## Notes

- `[P]` tasks modify separate files or can be authored independently after the stated prerequisite.
- `[US1]`, `[US2]`, and `[US3]` provide traceability to the prioritized specification stories.
- Controller matrices and vectors belong in the boundary journal, summary, and checkpoints rather than every ordinary metrics row.
- Preserve existing dirty-worktree changes and commit only intentional logical groups.
