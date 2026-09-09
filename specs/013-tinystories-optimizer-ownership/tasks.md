# Tasks: TinyStories Optimizer Ownership Comparison

**Input**: Design documents in `specs/013-tinystories-optimizer-ownership/`: `spec.md`, `plan.md`, `research.md`, `data-model.md`, `inspection.md`, `quickstart.md`, and all four `contracts/*.md`.

**Prerequisites**: Read the spec and plan before implementation; use `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python` (Python >=3.12) and the existing pinned environment. No new runtime dependency is planned.

**Tests/Verification**: Explicitly required by FR-028 and SC-001–008. Write the focused story tests before their implementation and confirm that the relevant missing behavior fails. Use small real models and synthetic epoch/report fixtures. Full campaign training is a separate researcher action; implementation verification includes only short diagnostic budgets and keeps final holdout sealed.

**Organization**: Five story phases follow spec order: US1/US2/US3 are P1; US4/US5 are P2. Each phase has an independently runnable acceptance check after its prerequisites. Shared topology and identity helpers are foundational because both CPU preflight and runtime need them.

## Format and paths

- Tasks use `- [ ] Tnnn [P?] [USn?] Description with exact file paths`.
- `[P]` identifies work that can overlap the explicitly named same-stage tasks in different files, once prerequisites are complete. It does not waive phase dependencies.
- All file paths are relative to the repository root. New files are explicitly described as created; existing helpers should be extended or reused.
- Keep ownership in `src/training/optimizer_state.py`, update sequencing in `src/training/steps.py`, and the fixed campaign workflow in `src/evaluation/optimizer_ownership.py` with one thin CLI. Do not add a second trainer or optimizer registry.

## Phase 1: Setup (Shared Experiment Structure)

**Purpose**: Establish the implementation record and the one new campaign module without changing runtime behavior.

- [X] T001 Create `docs/tinystories-optimizer-ownership-experiment.md` with the nine-arm protocol, pinned environment, artifact paths, requirement-to-verification outline, and the separate boundaries for diagnostic checks, later full training, and sealed holdout.
- [X] T002 Create `src/evaluation/optimizer_ownership.py` with the fixed arm/width definitions, exact active counts 115264/164416/213568/262720, schema-version constants, and plain campaign/run/endpoint record conventions from `specs/013-tinystories-optimizer-ownership/data-model.md`; reuse existing hashing and atomic artifact utilities.

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Supply stable identities and static model support shared by preflight, ownership, restore, and reports.

- [X] T003 [P] Add campaign-only schema-1 scientific contract serialization and canonical hashing in `src/utils/reproducibility.py`, covering representation, scope, clipping, initialization, data, budgets, evaluation, and count convention; add legacy-signature stability and changed-control hash checks in `tests/test_reproducibility.py` without injecting new defaults into historical signature inputs.
- [X] T004 [P] Expose or reuse explicit quarter parameter metadata in `src/models/ffn.py` for gate/up/down blocks and segment biases, distinguishing the common down bias and preserving both forward paths and full-shaped slicing gradients.
- [X] T005 Build stable ordered parameter descriptors and a reusable five-way concat partition in `src/training/optimizer_state.py` using T004 metadata: canonical names/tied aliases, shapes/dtypes, physical gradient support by width, quarter/common membership, and complete identity-deduplicated coverage; reject unequal quarters, overlap, or missing trainable parameters without allocating optimizer state.

**Checkpoint**: Identity and topology helpers are ready. T005 depends on T004; T003 can overlap T004–T005. All story implementation depends on this phase.

## Phase 3: User Story 1 — Establish a valid nine-run comparison (Priority: P1) — MVP

**Goal**: Materialize and audit exactly four fresh dense standalones and S1/S2/C1/C2/C3 at seed 42 without training.

**Independent Test**: Run nine-arm preflight with an audited input fixture and then the pinned prepared corpus. Verify exact dimensions/counts, fresh identities, 87132/348528-step horizons, one/four fixed-set epochs, matching first-epoch and elastic action/order digests; change one scientific control at a time and require an identified rejection.

### Verification

- [X] T006 [P] [US1] Add ownership/clipping eligibility and config-only CLI tests in `tests/test_config.py` and `tests/test_train_cli.py`: accept supported C3 and historical defaults; reject unsupported optimizers/topologies/distribution, slicing/standalone/nested-all/per-layer actions, corrections/warmup, conflicting caps, missing/extra owners, and nonfinite/nonpositive thresholds.
- [X] T007 [P] [US1] Create `tests/test_optimizer_ownership_campaign.py` with nine-arm expansion, pinned audit/alignment, dense/elastic shape/count, fresh seed-construction and identity tests; reject changed controls/hashes/tail/horizons and occupied/historical identities, verify complete expected action/epoch digests without balancing, and assert preflight never trains, evaluates holdout, creates run directories, or publishes success after a partial failure.

### Implementation

- [X] T008 [US1] Extend `src/utils/config.py` to resolve `per_ffn_block`, `scheduler_clock=global_step`, and explicit global/per-owner L2 clipping with all five C3 caps 1.0; enforce the supported single-process AdamW concat/global-action eligibility matrix, retain existing shared/per-width and non-campaign SGD behavior, and preserve the historical normalized optimizer mapping.
- [X] T009 [P] [US1] Create `configs/controlled_exps/tinystories_instruct_optimizer_ownership.yaml` with schema 1, campaign ID, common controls, exactly nine overrides and every pinned identity from `specs/013-tinystories-optimizer-ownership/inspection.md`; encode dense source-width metadata, elastic-only sampling fields, seed 42, d64/l4/h4/context128/vocab2048, initializer .02, AdamW .008/(.9,.95)/1e-8/.1, batch64/accumulation1/bf16, no corrections or pre-nested warmup, cosine warmup64/full horizons, validation every64, and repeat-epoch budgets.
- [X] T010 [P] [US1] Implement fixed common-plus-arm expansion and full allowed-difference validation in `src/evaluation/optimizer_ownership.py` through `resolve_run_config`; reject extra/missing arms, altered scientific controls and stale identities, retain fresh normal-constructor seed provenance/code/dependency versions, and attach T003 contract hashes to executable trainer configs and manifest records.
- [X] T011 [US1] Add pinned corpus/tokenizer/shard/order/role auditing and budget validation in `src/evaluation/optimizer_ownership.py`, reusing prepared-corpus audit helpers: require 5576491 available versus 5576448 designated permutation entries, fixed excluded 43 sequences/5504 tokens, 713785344 tokens/87132 updates per epoch, four disjoint roles, and CPU model/partition checks via `src/training/modeling.py` and `src/utils/model_size.py` for FFNs 64/128/192/256 and exact matching active counts.
- [X] T012 [US1] Generate full deterministic expected action and epoch-order digests in `src/evaluation/optimizer_ownership.py` using isolated existing action RNG and `src/training/packed_corpus.py` repeat sampler logic; verify the common first epoch across nine and all four orders/348528 uniform replacement actions across five, with one global action per update and no forced balance or rotating exclusions.
- [X] T013 [US1] Create the `preflight` command in `scripts/analyze_tinystories_optimizer_ownership.py` and its publication helper in `src/evaluation/optimizer_ownership.py` with the five required CLI options from `contracts/cli-entrypoints.md`; stage and publish `preflight.json`, `campaign_manifest.json`, and nine `configs/<arm>.yaml` only after full validation, fail nonzero with named discrepancies, and reserve unoccupied run identities without creating their run directories.
- [X] T014 [US1] Extend `train.py` config-only `--preflight` JSON with resolved ownership, clipping, and campaign identity while preserving legacy output behavior and distinguishing this check from full campaign audit/model/trace validation.
- [X] T015 [US1] Run `tests/test_optimizer_ownership_campaign.py`, the affected `tests/test_config.py` and `tests/test_train_cli.py` cases, then the read-only audit and nine-run preflight commands in `specs/013-tinystories-optimizer-ownership/quickstart.md` using fresh scratch outputs; record resulting manifest paths, counts/digests and named rejection evidence in `docs/tinystories-optimizer-ownership-experiment.md` without launching training.

**Checkpoint**: US1 is the preflight-only MVP. Its C3 topology check needs no optimizer step; passing preflight does not assert runtime implementation is complete.

## Phase 4: User Story 2 — Execute the intended ownership and clipping semantics (Priority: P1)

**Goal**: Execute real slicing/concat diagnostics with ordinary AdamW histories, disjoint C3 owners, independent clipping, and one global learning-rate clock.

**Independent Test**: Use wider-then-narrower and alternating-width real-model sequences to inspect tensors, lazy histories, parameter/owner counters and norms. With identical concat initial values/actions/data/rates, test-only globally clipped C3 must match C1 within existing tolerances; campaign C3 independently clips each active owner at 1.0.

### Verification

- [X] T016 [P] [US2] Create `tests/test_optimizer_ownership.py` with actual `ModifiedLlamaMLP`/`CatLlamaMLP` model paths covering S1 residual tail momentum/decay/full-shaped counters, S2 selected-history isolation/full-shaped allocation/zero never-used tail moments, concat inactive `grad is None` and exact unchanged weights/state/counters, active present-zero AdamW behavior, C2 lazy 4/3/2/1 quarter histories/common four, and C3 complete tied/bias ownership.
- [X] T017 [P] [US2] Extend `tests/test_per_granularity_optimizer.py` with global-clock integration checks for C3: ordered active quarter/common calls once, identical current LR across active and inactive owners, one scheduler advance per complete update, continuous full-horizon schedules, and unchanged shared/per-width clock behavior.
- [X] T018 [US2] Add clipping independence, active-zero/inactive-null observations, no second global rescale, sqrt(2)–sqrt(5) combined bounds, and matched-state C1 versus test-only globally clipped C3 numerical equivalence tests in `tests/test_optimizer_ownership.py`; verify diagnostics do not consume action RNG or change measured gradients beyond the intended clip.

### Implementation

- [X] T019 [US2] Implement `BlockOptimizerCollection` in `src/training/optimizer_state.py` over T005's disjoint O-A/O-B/O-C/O-D/O-common parameters, with ordered active-owner lookup for g250/g500/g750/g1000, lazy AdamW state, explicit successful-call accounting and topology serialization; retain overlapping per-granularity histories over shared weights and existing single shared optimizers.
- [X] T020 [US2] Extend `GlobalSchedulerClock` integration in `src/training/optimizer_state.py` to synchronize every C3 owner initially and after each global advance, using the existing scheduler formula/full assigned horizon and never owner-local counters as time.
- [X] T021 [US2] Wire C3 optimizer/clock construction into `src/training/steps.py` and runtime restoration checks in `src/training/run.py`; preserve shared/per-granularity construction, clear gradients across the whole model to None each update, and retain exactly one global action, batch, forward and backward in campaign mode.
- [X] T022 [US2] Implement explicit clipping and detached observations in `src/training/steps.py`: one global cap for global arms, separate joint-owner L2 caps for C3, existing stabilization, no subsequent rescale, finite loss/gradient/norm and synchronized-LR checks, C1's single applied coefficient, C3 independent coefficients, inactive null fields, active-zero coefficient 1, and combined disjoint norms.
- [X] T023 [US2] Implement the complete C3 update sequence in `src/training/steps.py`: mark mutation-in-flight before the first optimizer call, step active quarters in A/B/C/D order then common, advance/synchronize the global clock once, reconcile width/quarter/owner/token/cursor counts and only then clear the flag and publish committed observations; stage successful owner returns locally until complete accounting.
- [X] T024 [US2] Add an internal diagnostic-only clipping override to C3 construction in `src/training/optimizer_state.py` and `src/training/steps.py` for the T018 matched-state global-clipping test; keep the normal campaign C3 contract restricted to per-owner caps and retain the preflight rejection case in `tests/test_optimizer_ownership_campaign.py`.
- [X] T025 [US2] Run `tests/test_optimizer_ownership.py` and `tests/test_per_granularity_optimizer.py`, inspect full tensor/history/clock comparisons, and record semantic evidence and numerical tolerances in `docs/tinystories-optimizer-ownership-experiment.md`.

**Checkpoint**: Short model-based ownership/clipping diagnostics pass. Durable failure recovery is delivered in US3 before interrupted training is considered supported.

## Phase 5: User Story 3 — Resume without altering the experiment (Priority: P1)

**Goal**: Restore every campaign arm exactly at a committed boundary and preserve prior durable state and attempt costs when a multi-owner update fails.

**Independent Test**: Compare uninterrupted and resumed short runs before, at, and after a synthetic epoch boundary for all nine arms. Actions/batches/integer counts must match exactly, numerical state must meet existing tolerances, invalid payloads must leave every live component unchanged, and injected first/middle/last-owner or later clock/accounting failures must leave the prior durable checkpoint hash unchanged.

### Verification

- [X] T026 [P] [US3] Create `tests/test_optimizer_ownership_resume.py` covering all-arm exact action/batch/model/history/LR/cursor parity before/at/after epoch boundaries, unexposed lazy absence, missing required or impossible extra state, wrong ordered mappings/shapes/dtypes/counters/optimizer kwargs, cross-arm/representation/scope/model-only payloads, malformed model/RNG/scheduler/data/accounting, and mutation-free rejection or whole-bundle rollback on installation failure.
- [X] T027 [P] [US3] Add C3 failure-injection cases in `tests/test_optimizer_ownership.py` for mutation-then-raise in first/middle/last owner and scheduler/accounting failures; test pre-mutation transaction restoration, poisoned-state abort, gating of normal/exception/signal/finalization saves, prior durable checkpoint hash preservation and failure-stage/returned-owner records without per-step full-state snapshots.
- [X] T028 [US3] Add continuation resource and scientific-row reconciliation cases in `tests/test_optimizer_ownership_resume.py`: multiple attempts, replay after an older checkpoint, repeated ledger observations, hard-kill incomplete measurements, sum-of-unique-attempt durations/max peaks, no watermark double counting, and removal or segregation of non-durable trace/metric rows.

### Implementation

- [X] T029 [US3] Implement exposure-derived required-history validation for every campaign scope in `src/training/optimizer_state.py`: standalone/S1 steps T, S2 all physical parameters ng, C1 quarter/common aq/T, C2 supported quarter/common ng and excluded quarter absent, C3 quarter/common aq/T; validate exact names/IDs/groups/kwargs, component shapes/dtypes/finiteness, nonnegative second moments and integral expected counters while preserving true lazy absence and full slicing state.
- [X] T030 [US3] Add `optimizer_ownership_checkpoint_schema_version=1` and full campaign/run/topology/clipping/budget/epoch/accounting/resource-watermark fields in `src/training/checkpointing.py`; require resumable purpose for all new campaign arms, version changed collection layouts explicitly, and keep historical non-campaign loading under its existing compatibility path without cross-arm migration.
- [X] T031 [US3] Stage whole-payload validation before live mutation in `src/training/checkpointing.py`, checking model, T029 histories, clock/horizon/LRs, RNG/action ordinal, sampler/membership/cursor, exposures/tokens/epochs and metric/resource watermarks with temporary local RNG/sampler objects; install only after all checks pass and add a one-time whole-bundle restore guard for unexpected installation failures.
- [X] T032 [US3] Complete mutation-failure handling in `src/training/steps.py` and all resumable publication paths in `src/training/checkpointing.py` and `src/training/run.py`: restore transactions only before mutation, poison and abort after mutation starts, reject unsafe normal/exception/signal/finalization saves, and record pending step/stage/active and returned owners while identifying the last durable checkpoint.
- [X] T033 [US3] Extend batch/action provenance and restored-boundary reconciliation in `src/training/data.py`, `src/training/steps.py`, and `src/training/checkpointing.py` using existing `src/training/packed_corpus.py` repeat-sampler state; retain continuous streams and fixed membership across epochs, exactly one/four terminal epochs, reproduce ordered batches from cursor/order identities, and segregate non-durable scientific rows before replay.
- [X] T034 [US3] Implement a versioned atomic `resource_attempts.json` ledger in `src/training/run.py` using existing artifact IO, keyed by unique attempt ID with latest observation sequence/duration/peaks/source-checkpoint/attempted work/status/completeness; sum unique latest durations, take maximum peaks, exclude queue downtime, retain failed/replayed costs and mark unfinalized hard-kill attempts incomplete.
- [X] T035 [US3] Wire ledger observations into existing flush/heartbeat/checkpoint/normal/failure boundaries in `src/training/run.py`, `src/training/steps.py`, and `src/training/checkpointing.py`, preserving the ledger outside rollbackable scientific state and validating checkpoint watermarks without re-adding prior cost; reconcile final T, selections, activations, owner calls, tokens, epochs and scheduler position.
- [X] T036 [US3] Run `tests/test_optimizer_ownership_resume.py`, T027 failure cases in `tests/test_optimizer_ownership.py`, and `tests/test_per_granularity_optimizer_resume.py`; record exact trace parity, malformed-load non-mutation, failure durability and cumulative-resource evidence in `docs/tinystories-optimizer-ownership-experiment.md`.

**Checkpoint**: Interrupted diagnostics can safely resume from durable boundaries; completed model state and consumed attempt costs remain distinct and auditable.

## Phase 6: User Story 4 — Audit outcomes, clipping, and resource costs (Priority: P2)

**Goal**: Produce inspectable saved traces, scalar summaries, clipping/resource measurements, terminal ordinary-validation provenance and individual diagnostic plots.

**Independent Test**: Inspect short-run artifacts for every arm without rerunning training, reconcile observations to committed updates, compare measured optimizer storage with actual tensors, and recover a missing terminal sidecar from the same durable checkpoint with zero further optimizer updates.

### Verification

- [X] T037 [P] [US4] Create `tests/test_optimizer_ownership_reporting.py` with per-run artifact fixtures checking schema/provenance, committed traces/counts/epochs, C1/C3 clipping frequencies with active denominators, actual owner/component/dtype storage and lazy absence, resource completeness labels, and individual loss/perplexity/resource plot series.
- [X] T038 [P] [US4] Extend `tests/test_optimizer_ownership_resume.py` with terminal checkpoint/evaluation/sidecar failure injection and repeated completion tests: weighted causal-target aggregation and exact counts, no sidecar for failed evaluation/checkpoint publication, recovery from the same terminal hash with zero owner steps, idempotent reuse of a valid sidecar, and recovery costs retained in the attempt ledger.

### Implementation

- [X] T039 [US4] Extend `src/utils/metrics.py` and `src/training/run.py` with versioned scalar/summary fields and atomic or durable append handling for `optimizer_ownership_trace.jsonl` and C1/C3 `optimizer_ownership_clipping.jsonl`; persist committed action/batch digests/cursors, selected width, active owners, integer clocks/counts, packed tokens, immutable controls/provenance and per-group observations from US2 while keeping full parameter states in checkpoints.
- [X] T040 [US4] Add non-allocating optimizer-state measurement in `src/training/optimizer_state.py` at report/checkpoint boundaries, grouped by owner/component/dtype using actual `numel * element_size`, with separate counters; verify post-exposure moment expectations 2(F+R), 2(4F+4R), and 2(2.5F+4R) for F=196608/R=328256 while measuring actual allocations rather than assuming bf16 state.
- [X] T041 [US4] Integrate resource and exposure summaries in `src/training/run.py` and `src/utils/metrics.py`: separate actual width selections from quarter activation and labeled expectations, persistent storage from temporary concat observations or method-labeled estimates, allocated/reserved peaks, wall time, useful committed versus attempted throughput, checkpoint bytes/hash and attempt completeness; never represent unsupported device metrics as zero or FFN-only 37.5% savings as total-memory savings.
- [X] T042 [US4] Extend terminal results from `src/evaluation/validation.py` with examples, valid causal-target counts, ordinary-validation manifest/protocol identity and exact active counts via `src/utils/model_size.py`, preserving `target_token_weighted_causal_shift_float64`, perplexity `exp(loss)`, four elastic widths and dense source-width labels without averaging batch perplexities.
- [X] T043 [US4] Publish immutable schema-1 `terminal_validation_results.json` from `src/training/run.py`, bound to the durable terminal checkpoint SHA256 and full campaign/budget/evaluation/count contract with a content hash excluding itself; support completion-only recovery and idempotent reuse with unchanged checkpoint identity/no optimizer update, and emit no valid sidecar when checkpoint publication or evaluation fails.
- [X] T044 [US4] Implement per-run loss/perplexity trajectories, resource plots and width/group clipping summaries in `src/evaluation/optimizer_ownership.py` from saved artifacts, validating trace/accounting agreement and active denominators; retain distinct expected/random exposure labels and clearly marked incomplete resource measurements.
- [X] T045 [US4] Run the per-run cases in `tests/test_optimizer_ownership_reporting.py` and terminal-recovery cases in `tests/test_optimizer_ownership_resume.py`; inspect generated scalar/sidecar/plot artifacts and record their paths and reconciliation evidence in `docs/tinystories-optimizer-ownership-experiment.md`.

**Checkpoint**: Completed/resumed diagnostics produce auditable records and terminal ordinary-validation sidecars that downstream reporting can consume without another training update.

## Phase 7: User Story 5 — Produce the complete terminal endpoint comparison (Priority: P2)

**Goal**: Freeze nine valid terminals and export exactly 24 ordinary-validation endpoints plus two combined figures in PNG and PDF.

**Independent Test**: Use a complete nine-run/24-endpoint controlled fixture. Verify source identities, CSV/JSON equivalence and plotted data, then reject missing/duplicate/stale/nonterminal/mixed-role/nonfinite/count/target/budget/control mismatches. Explicit partial mode may omit points but must still reject invalid supplied evidence and visibly label every output.

### Verification

- [X] T046 [P] [US5] Extend `tests/test_optimizer_ownership_reporting.py` with complete frozen-source/24-endpoint fixtures, CSV/JSON row parity and both PNG/PDF figures; test best/epoch-one/trailing-mean substitution, mixed roles, wrong evaluated targets/counts/budgets/controls, duplicate/missing/nonfinite endpoints, changed source hashes, explicit partial labeling, ignored `final_holdout_results.json`, five separate four-point curves, four unconnected standalone markers and no uncertainty bars.
- [X] T047 [P] [US5] Extend `tests/test_optimizer_ownership_campaign.py` with `freeze`/`report` CLI tests for mutually exclusive `--run-root` and repeated `--run-dir`, manifest-based identities, nonzero named failures, no implicit evaluation/training, staged success publication, and independent `--allow-partial` opt-in for both freezing and consuming a partial manifest.

### Implementation

- [X] T048 [US5] Implement strict freeze validation in `src/evaluation/optimizer_ownership.py`: require nine distinct preflight-bound fresh run IDs, durable terminal checkpoint/sidecar/summary hashes, exact horizons/epochs/tokens, full common controls and allowed arm differences, and runtime trace equality against expected digests across elastic actions/all four epochs and all nine first epochs; atomically emit `frozen_manifest.json` with source hashes only after validation.
- [X] T049 [US5] Implement endpoint reading and revalidation in `src/evaluation/optimizer_ownership.py` for exactly five-by-four elastic plus four dense `(arm,width)` records, uniform ordinary-validation role/manifest/protocol/target counts, finite loss and `exp(loss)` perplexity, exact active counts and assigned/actual budgets/checkpoint provenance; never select best checkpoints, trailing metrics, or holdout-preferred data.
- [X] T050 [US5] Add explicit partial freeze/report behavior in `src/evaluation/optimizer_ownership.py`: permit only missing arms/points on opt-in, enumerate omissions, reject malformed supplied endpoints, require opt-in again when reading a partial manifest, and visibly mark manifests/tables/reports/figures partial without inventing points.
- [X] T051 [US5] Export matching `optimizer_ownership_endpoints.csv` and `.json` plus `comparison_report.json` from `src/evaluation/optimizer_ownership.py`, including representation/scope/clipping, run/seed/width/dimension/count, loss/perplexity/targets, actual/assigned budgets, checkpoint/evaluation identities, exposure and resource fields; express all six EX-012 comparisons descriptively for seed 42 with explicit initialization-by-seed and aggregate-token versus compute distinctions.
- [X] T052 [US5] Generate `optimizer_ownership_perplexity_vs_non_embedding_parameters` and `optimizer_ownership_loss_vs_non_embedding_parameters` in PNG and PDF in `src/evaluation/optimizer_ownership.py`: exact integer x values, five consistently colored connected elastic series, four disconnected standalone markers, retained coincident labels, no seed bars, and dataset/seed/role/count/terminal/one-pass-versus-four-pass budget annotations; retain T044 individual plots.
- [X] T053 [US5] Wire `freeze` and `report` subcommands in `scripts/analyze_tinystories_optimizer_ownership.py` to T048–T052 with all options in `specs/013-tinystories-optimizer-ownership/contracts/cli-entrypoints.md`, named nonzero failures and saved-artifact-only execution, preserving `scripts/analyze_tinystories_per_width_optimizer.py` historical semantics.
- [X] T054 [US5] Run `tests/test_optimizer_ownership_reporting.py` and `tests/test_optimizer_ownership_campaign.py`, inspect fixture exports/figure series and four combined image files, and record complete/partial/rejection evidence in `docs/tinystories-optimizer-ownership-experiment.md` without claiming full-budget campaign results.

**Checkpoint**: Complete reporting is verified on controlled fixtures; real campaign figures are produced only after a separately requested full campaign supplies nine terminal runs.

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Verify compatibility and document the implemented workflow and its measured limits.

- [X] T055 [P] Finalize `docs/tinystories-optimizer-ownership-experiment.md` and `specs/013-tinystories-optimizer-ownership/quickstart.md` with actual preflight/individual training/resume/completion-only recovery/freeze/report commands, artifacts, failure recovery, resource caveats and the six comparison interpretations; state that full campaign launch and future uniform holdout evaluation require separate researcher requests.
- [X] T056 [P] Run the four focused ownership suites and compatibility suites listed in `specs/013-tinystories-optimizer-ownership/quickstart.md`, plus `tests/test_reproducibility.py`, `tests/test_distributed.py`, `tests/test_distributed_sampling.py`, and `tests/test_global_sampling_windows.py`; use `tests/test_optimizer_ownership_campaign.py` to retain a historical Feature 12 analyzer fixture check, and record commands/results or environment limitations in `specs/013-tinystories-optimizer-ownership/verification.md` for shared/per-width defaults/checkpoints, distributed shared behavior, sampling, repeat epochs, counts, CLI and reporting.
- [X] T057 Run a short single-GPU bf16 real-model diagnostic through `tests/test_optimizer_ownership.py` and `tests/test_optimizer_ownership_resume.py` with test-only budgets (add device-parametrized cases if needed), checking finite clipping/state, one-clock behavior, exact action/batch resume and measured allocated/reserved peaks; record device, precision, commands, results and unavailable-GPU limitations in `specs/013-tinystories-optimizer-ownership/verification.md`, keeping full campaign configs immutable.
- [X] T058 Reconcile FR-001–028, EX-001–013 and SC-001–008 against implemented checks and saved evidence in `docs/tinystories-optimizer-ownership-experiment.md` and `specs/013-tinystories-optimizer-ownership/verification.md`; distinguish diagnostic/fixture evidence from future full-budget observations, inspect the final diff for historical behavior changes or new hot-loop snapshots, and update completion checkboxes in `specs/013-tinystories-optimizer-ownership/tasks.md` only for work actually verified.

**Phase 8 result (2026-09-09)**: T055–T058 verified as documented in
[verification.md](verification.md). GPU cases are implemented but skipped because
CUDA is unavailable, including outside the sandbox; T057 uses its explicit
unavailable-GPU reporting path. No GPU measurement or full-budget outcome is
claimed. Full campaign launch and future uniform holdout evaluation remain
separately requested researcher work.

## Dependencies & Execution Order

### Story completion graph

```text
Setup (T001–T002)
  -> Foundation (T003 || T004 -> T005)
  -> US1: preflight MVP (T006–T015)
  -> US2: ownership/clipping (T016–T025)
  -> US3: exact resume/failure durability (T026–T036)
  -> US4: audit/terminal artifacts (T037–T045)
  -> US5: complete comparison (T046–T054)
  -> Polish and final verification (T055–T058)
```

US1 is independently deliverable after the foundation because it only constructs and inspects models. US2 consumes US1's normalized config and identities. US3 requires US2's owner/clock/accounting runtime. US4 consumes US3's durable-boundary and attempt-ledger guarantees. US5 consumes US1's campaign identities and US4's terminal/artifact schemas. These are real data-flow dependencies; test fixtures permit independent acceptance runs but do not remove implementation prerequisites.

### Within-phase ordering

- Write story verification first; run failing cases before implementing their behavior and rerun them at the story checkpoint.
- Unmarked implementation tasks run in listed order. T003 and T004 are independent; T005 needs T004. US1 config schema T008 precedes the parallel YAML/expansion work T009/T010; T011–T013 require both. T014 then exposes the finished resolved contract.
- Keep modifications to `src/training/optimizer_state.py`, `src/training/steps.py`, `src/training/checkpointing.py`, `src/training/run.py`, and `src/evaluation/optimizer_ownership.py` sequential within their dependent task chains.
- T018 follows T016 because both edit one test file; T028 follows T026; T024 follows clipping/runtime implementation. T038 follows US3's resume tests. Later story tasks extending earlier test files wait for the earlier phase to finish.
- T055 and T056 may overlap using separate documentation/evidence files. T057 follows CPU/compatibility verification and T058 reconciles all results. After a fix, rerun affected checks; broaden only for a new failure or unresolved compatibility concern.

## Parallel Execution Examples

These examples describe possible implementation scheduling, not instructions to launch agents or campaign jobs.

### User Story 1

```text
After foundation:
  T006: config/CLI tests in tests/test_config.py and tests/test_train_cli.py
  T007: campaign tests in tests/test_optimizer_ownership_campaign.py
After T008:
  T009: campaign YAML in configs/controlled_exps/tinystories_instruct_optimizer_ownership.yaml
  T010: expansion/controls in src/evaluation/optimizer_ownership.py
Join both before T011.
```

### User Story 2

```text
After US1:
  T016: real-model semantics in tests/test_optimizer_ownership.py
  T017: scheduler integration in tests/test_per_granularity_optimizer.py
Join before T018 and sequential ownership/runtime implementation.
```

### User Story 3

```text
After US2:
  T026: full restore tests in tests/test_optimizer_ownership_resume.py
  T027: owner-failure tests in tests/test_optimizer_ownership.py
T028 follows T026; join verification work before T029–T035 implementation.
```

### User Story 4

```text
After US3:
  T037: saved artifact tests in tests/test_optimizer_ownership_reporting.py
  T038: terminal recovery tests in tests/test_optimizer_ownership_resume.py
Join before T039–T044 implementation.
```

### User Story 5

```text
After US4:
  T046: endpoint/figure tests in tests/test_optimizer_ownership_reporting.py
  T047: freeze/report CLI tests in tests/test_optimizer_ownership_campaign.py
Join before sequential T048–T053 campaign module/CLI work.
```

## Requirement Coverage

| Requirement group | Primary tasks and evidence |
| --- | --- |
| FR-001–005, FR-015; EX-001–009; SC-001–002 | T003–T015 matrix, config, audit, identity, model counts and expected traces; T033/T035 runtime completion reconciliation |
| FR-006–011; SC-003 | T004–T005 topology; T016–T021 model/history diagnostics and collection; T029 required-state rules |
| FR-012–014, FR-017, FR-019, FR-021; EX-011 | T017–T024 clipping/clock/commit semantics; T027/T032 failures; T035/T039/T041/T044 accounting and summaries |
| FR-016–020; SC-004–005 | T026–T036 exact resume, staged validation, poisoned saves and cumulative attempt costs |
| FR-018, FR-020–023, FR-025; EX-004, EX-010, EX-013; SC-005, SC-007 | T037–T045 audit artifacts, measured storage, target-weighted terminal sidecars and individual plots |
| FR-022, FR-024–027; EX-005, EX-011–013; SC-006–008 | T046–T054 strict freeze, 24 endpoints, tables, two figures, partial labeling and interpretation |
| FR-028; SC-001–008 verification scope | Story checkpoint tasks T015/T025/T036/T045/T054 and final T055–T058; full-budget scientific results remain future campaign work |

## Implementation Strategy

1. **MVP first**: Complete Setup, Foundation and US1. Demonstrate valid nine-run materialization and mismatch rejection without training; stop at the preflight deliverable if only the MVP is being implemented.
2. **Add runtime semantics**: Complete US2 with actual slicing/concat diagnostics. Confirm both intended differences and test-only C1/C3 plumbing equivalence.
3. **Make continuation reliable**: Complete US3 before relying on interrupted diagnostics. Require exact action/batch identity, valid lazy absence, complete restore validation and prior-checkpoint durability.
4. **Make results inspectable**: Complete US4, including completion-only terminal-sidecar recovery and measurements across failed/resumed attempts.
5. **Deliver comparison tooling**: Complete US5 against controlled nine-run/24-endpoint fixtures, then compatibility/GPU verification and the runbook. Fixture plots prove tooling, not experimental outcomes.
6. **Later researcher execution**: A separate full-campaign request runs the immutable manifest configs through the existing trainer. Only completed terminals supply actual comparison outcomes; holdout remains sealed unless separately requested under its freeze/uniform-evaluation protocol.

## Task Summary

| Phase | Tasks |
| --- | ---: |
| Setup | 2 |
| Foundational | 3 |
| US1 (P1, MVP) | 10 |
| US2 (P1) | 10 |
| US3 (P1) | 11 |
| US4 (P2) | 9 |
| US5 (P2) | 9 |
| Polish | 4 |
| **Total** | **58** |

All tasks start unchecked. There are 16 `[P]` tasks in eight disjoint-file scheduling pairs, including at least one pair for each story. Implementation has not been performed by task generation.
