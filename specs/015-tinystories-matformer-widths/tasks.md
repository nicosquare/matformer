# Tasks: TinyStories Optimizer Ownership with MatFormer Widths

**Input**: Design documents in `specs/015-tinystories-matformer-widths/`: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, and `quickstart.md`.
**Prerequisites**: Read those documents and `.specify/memory/constitution.md` before implementation.
**Status**: Closed by user with partial results on 2026-09-22. Implementation, reporting prerequisites, operational CPU/GPU gates, and T034–T036 are complete. Four fresh standalones and S1/S2 passed strict terminal validation; their selected comparison contains 12 new plus four historical endpoints. C1–C3 and full-campaign reporting are cancelled. No further execution or monitoring is authorized. T037/T045/T052 remain unchecked and explicitly cancelled, not pending work or false completion. T056/T057 are completed for the reduced closeout scope. See verification.md and the runbook for evidence.

**Scope amendment**: The user's “finalize this feature as it is” instruction supersedes the remaining execution/report dependencies. An unchecked task prefixed **CANCELLED** is closed without successful execution. Original task descriptions and full-campaign acceptance thresholds are retained for audit. No active tasks remain.

**Tests/Verification**: Explicitly required by FR-017 and SC-001–010. Use the existing trainer and real model/update/restore paths, focused CPU fixtures, and separately authorized Slurm GPU probes. Fixture success establishes implementation readiness, not production completion.

**Organization**: Setup, shared foundations, then the five stories in specification priority order. Execution tasks remain in their owning story and have explicit forward dependencies on reporting and regression checks. The execution order below resolves these dependencies; phase placement does not authorize running production early.

## Format and path conventions

- Tasks use `- [ ] Tnnn [P?] [USn?] Description with file path`.
- `[P]` identifies a task that can run alongside another task in a different file once its stated prerequisites pass. It does not authorize concurrent edits to shared files.
- All source/document paths are relative to the repository root. Use `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python` and `OMP_NUM_THREADS=1` for CPU verification.
- For later authorized operations, `MW_ROOT` denotes a newly validated, distinct campaign root; the proposed path is `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1`. Paths beginning `$MW_ROOT/` name future evidence, not artifacts created by task generation.
- This command authorizes writing this task list. Later implementation and GPU/execution authorization come from the conversation, as specified in `spec.md`; historical launch approvals and passed gates do not supply it. Do not create or reserve the external campaign root during task generation.

## Phase 1: Setup (Shared Experiment Structure)

**Purpose**: Establish the runbook and evidence record without changing scientific controls or creating production artifacts.

- [X] T001 [P] Create `docs/tinystories-matformer-widths-experiment.md` from the planned CLI contract, documenting the nine-arm matrix, inherited controls/data, fresh artifact-root layout, standalone barrier, authorization boundaries, Slurm exclusions/limits, and separate diagnostic/production/new-report/combined-report statuses; mark every unexecuted stage pending.
- [X] T002 [P] Create `specs/015-tinystories-matformer-widths/verification.md` with FR/SC acceptance sections, exact command/environment/source/config-hash fields, CPU/GPU results and skips, job/attempt provenance, artifact references, and explicit pending states for actual model counts, data/reference audits, gates, and terminal results.

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish shared schema and topology contracts. Complete this phase before story implementation.

- [X] T003 Add explicit schema-specific width, arm, and common-control selectors in `src/evaluation/optimizer_ownership.py`: schema 4 selects campaign `tinystories-optimizer-ownership-matformer-widths-v1`, ST-g125/ST-g250/ST-g500/ST-g1000/S1/S2/C1/C2/C3, prefixes 32/64/128/256 and counts 90,688/115,264/164,416/262,720; reject unknown schemas and keep schemas 1–3 constants/defaults unchanged.
- [X] T004 Define new-only `campaign_schema_version=4`, ordered `width_grid`, and contiguous `block_boundaries` A=[0,32), B=[32,64), C=[64,128), D=[128,256) in campaign/run scientific-contract construction in `src/evaluation/optimizer_ownership.py`; retain the existing scientific-contract schema-1 serializer and include the extra fields in its existing hash without adding defaults to legacy contracts.
- [X] T005 Add compatibility fixtures in `tests/test_matformer_widths_campaign.py` that pin legacy schema 1–3 resolutions/signatures, distinguish schema-qualified repeated arm labels, reject unknown versions, and prove new topology metadata changes scientific identity without modifying historical inputs; verify the foundational selectors and serialization checks.

**Checkpoint**: New and historical physical grids have distinct identities and legacy behavior is pinned.

## Phase 3: User Story 1 — Validate the new nine-run protocol (Priority: P1) — MVP

**Goal**: Prepare a strict nine-run campaign and validate actual geometry, controls, budgets, data, and deterministic expected traces without training.

**Independent test**: Construct all nine real definitions; inspect every dense size and elastic width; verify the expected counts and full action/batch digests; mutate individual controls, data identities, topology, initialization, and reservations to obtain explicit rejection. No GPU training is needed.

### Tests

- [X] T006 [P] [US1] Add schema-4 expansion/preflight acceptance and rejection cases in `tests/test_matformer_widths_campaign.py` for all nine runs, source versus dense-local fractions, actual slicing/concat/dense dimensions and counts, exact controls/budgets, uniform replacement sampling, first-epoch/four-epoch traces, role isolation and fixed excluded tail, occupied identities, and historical/diagnostic initialization; retain legacy unequal-quarter rejection tests.

### Implementation

- [X] T007 [P] [US1] Create `configs/controlled_exps/tinystories_instruct_matformer_widths.yaml` with exactly the five top-level keys `schema_version`, `campaign_id`, `common`, `arms`, `expected_data`; copy all original uncorrected pinned controls/data hashes, set explicit .125/.25/.5/1 prefixes, seed 42, four one-epoch dense arms and five four-epoch elastic arms, global replacement H=1, and no correction/warmup/scaling changes.
- [X] T008 [US1] Extend expansion and materialized-config validation in `src/evaluation/optimizer_ownership.py` to resolve arms by schema plus label, insert `run.campaign_schema_version=4` before resolution, preserve dense source fractions with local fraction 1, produce `<campaign_id>-<arm_id>-s42` identities, and bind resolved controls/grid/boundaries consistently into contracts; keep fresh construction separate from own-run continuation.
- [X] T009 [US1] Add exact schema-4 eligibility checks in `src/utils/config.py` for marker/contract/model agreement, dense source-width mapping, four-prefix elastic geometry, declared representation/ownership/clipping, global uniform replacement sampling and correction mode none; reject partial/mismatched metadata and preserve default equal-quarter restrictions for other configurations.
- [X] T010 [US1] Extend real-model preflight inspection in `src/evaluation/optimizer_ownership.py` to instantiate all nine runs, verify all active dimensions and non-embedding counts, inspect 32/32/64/128 concat shapes across four layers, and reject discrepancies rather than substituting nominal fractions or changing expected counts.
- [X] T011 [US1] Make expected action/epoch trace generation and data/control checks grid-aware in `src/evaluation/optimizer_ownership.py`: verify 5,576,448 designated sequences plus the same excluded 43, 87,132 updates/713,785,344 tokens per epoch, elastic 348,528-update horizons, all nine first-epoch batch digests and five full elastic action/batch digests; audit disjoint roles without evaluating the sealed holdout or forcing balanced draws.
- [X] T012 [US1] Integrate strict schema-4 preflight publication and exclusive fresh-run reservation through `src/evaluation/optimizer_ownership.py` and `scripts/analyze_tinystories_optimizer_ownership.py`, preserving existing flags and publishing resolved configs, model counts, expected traces, environment/source/config/data provenance and success only after all checks pass.
- [X] T013 [US1] Run US1 CPU acceptance and relevant legacy campaign/config checks from `tests/test_matformer_widths_campaign.py`, `tests/test_optimizer_ownership_campaign.py`, and `tests/test_config.py`; record exact results and any still-pending real-input audit in `specs/015-tinystories-matformer-widths/verification.md` without reserving a production root unless execution preparation is authorized.

**Checkpoint**: SC-001 can be demonstrated by CPU evidence. This is the MVP; no training or production launch is needed.

## Phase 4: User Story 2 — Verify ownership and clipping with unequal blocks (Priority: P1)

**Goal**: Preserve each arm's update semantics and produce support-derived exposure, storage, and real clipping evidence for the new grid.

**Independent test**: Exercise every width and wider-then-narrower transitions for all five elastic arms; compare weights, gradients, moments/counters, owner coverage, clipping coefficients and one scheduler advance. Read actual C1/C3 sidecars and measured allocations. Strict terminal-reader integration completes with T040/T044.

### Tests

- [X] T014 [P] [US2] Add real-model ownership/update/clipping tests in `tests/test_matformer_widths_campaign.py` covering S1 residual momentum/decay, S2 selected full-shaped histories and zero unused tails, concat absent inactive gradients versus active zero gradients, C2 lazy 4/3/2/1 block and four common histories after real exposure, tied/common-bias coverage, exactly five C3 owners, independent cap-1 coefficients, step ordering/current LR and one global scheduler advance.
- [X] T015 [P] [US2] Extend `tests/test_metrics_compact_accounting.py` with contract-aware g125 acceptance, undeclared-width rejection, compact-accounting restore compatibility, and bounded state growth; preserve legacy accepted labels and counter behavior.

### Implementation

- [X] T016 [US2] Thread optional validated topology through concat partition/runtime construction in `src/training/optimizer_state.py`; validate each layer's gate/up/down and optional bias shapes against declared dimensions, contiguous prefix support, exactly five disjoint owners with tied-object deduplication, and support-driven lazy history requirements while retaining equal-quarter validation for callers without topology.
- [X] T017 [US2] Pass the validated layout into C3 and C1 diagnostic groups in `src/training/steps.py`; retain one global cap for S1/S2/C1/C2/dense, independent L2 cap 1 for each active C3 owner, ordered block/common steps and one scheduler advance, and emit committed active/null clipping observations without a second global clip or block-size scaling.
- [X] T018 [US2] Make compact attempt-width validation in `src/utils/metrics.py` consume the validated campaign grid, accept g125 only where declared, preserve semantic C1/C3 clipping eligibility and legacy restore behavior, and keep attempt accounting bounded rather than storing growing identifier histories.
- [X] T019 [US2] Make `src/training/run.py` summaries grid/support-aware for explicit standalone source widths, actual selections/block activations/owner calls/tokens, scalar metrics and scheduler position; use actual owner/component/dtype allocations and `2*(4*F_A+3*F_B+2*F_C+F_D+4*R)` for fully exposed C2 moment expectations, separating counters, temporaries, device peaks, checkpoint bytes and throughput.
- [X] T020 [US2] Extend observation/clipping inspection in `src/evaluation/optimizer_ownership.py` to use selected physical widths and unequal blocks, require semantic C1/C3 sidecars for new identities, and reconcile active flags, pre/post norms, caps/coefficients, combined norms, committed counts and metrics/summary references; reject omitted or inconsistent evidence.
- [X] T021 [US2] Run the ownership, all-width clipping/sidecar and compact-accounting acceptance cases in `tests/test_matformer_widths_campaign.py` and `tests/test_metrics_compact_accounting.py`; record measured versus expected allocations and observed versus expected exposure in `specs/015-tinystories-matformer-widths/verification.md`, keeping GPU-only cases explicitly pending.

**Checkpoint**: CPU semantics satisfy SC-003; statistical exposure expectations remain expectations, not sampling quotas.

## Phase 5: User Story 3 — Complete standalones before elastic production and resume safely (Priority: P1)

**Goal**: Preserve whole-bundle continuation/failure safety and enforce source-bound readiness, validated standalone-first scheduling, unique attempts, and complete resource accounting.

**Independent test**: Compare uninterrupted/resumed short runs for all nine definitions immediately before, at, and after synthetic epoch boundaries; inject malformed restore and post-mutation failures; exercise barrier/restart/live-limit/attempt reconciliation with fixtures and mocked Slurm. Scheduling tests must not submit real jobs.

### Tests

- [X] T022 [P] [US3] Add resume/failure tests in `tests/test_matformer_widths_campaign.py` for all nine definitions and before/at/after epoch boundaries, exact subsequent actions/batches and inherited numerical tolerances, full S2/lazy C2 histories, identity/grid/boundary/probability/horizon/cursor/accounting mutations, missing/extra/nonfinite state, no live mutation on rejection, installation rollback, owner/scheduler/accounting failure durability, C1/C3 sidecars across resume, and zero-step terminal-sidecar recovery.
- [X] T023 [P] [US3] Create `tests/test_matformer_widths_queue.py` with mocked preflight/gate/barrier/Slurm fixtures covering stale source/config evidence, reservation adoption, incomplete/tampered/diagnostic standalone rejection, worker-entry revalidation, unrelated user jobs and stricter limits, duplicate locks/intents, delayed squeue/sacct visibility, uncertain submissions, own-run continuation, unique attempt costs, unknown hard-kill measurements and restart-safe report statuses; use strict terminal fixtures once T040 is available.

### Implementation and fixture verification

- [X] T024 [US3] Extend new-campaign identity and topology validation in `src/training/checkpointing.py` to stage the complete model/optimizer/scheduler/RNG/sampler/cursor/metrics/resource bundle before installation, preserve whole-bundle rollback, validate required versus absent lazy histories and physical descriptors, and reject model-only/cross-arm/old-campaign/malformed payloads without changing live state.
- [X] T025 [US3] Preserve and verify the complete-update safety boundary in `src/training/steps.py`, `src/training/checkpointing.py`, and `src/training/run.py`: mark unsafe before owner mutation; poison/abort on owner, scheduler or accounting failure; block normal/exception/signal/finalization partial saves; reconcile replayed scientific suffixes to the last durable boundary; recover a terminal sidecar with zero extra steps and unchanged terminal checkpoint identity.
- [X] T026 [US3] Link `ResourceAttemptLedger` observations in `src/training/run.py` to run, monotonic launch-attempt ID, Slurm job ID and process UUID; aggregate the latest duration once per unique attempt and maximum peaks, retain attempted/replayed/failed work and explicit incomplete observations, and keep overlapping scheduler allocation time separate from process time.
- [X] T027 [US3] Create CPU/GPU diagnostic modes in `scripts/preflight_tinystories_matformer_widths.py`: CPU freezes an immutable snapshot and executes acceptance/regression checks against it; both gates bind exact source/config-set/preflight hashes, environment and commands/results; GPU requires sbatch and records job/hardware plus all nine real-shape batch-64/context-128 bf16 checks retaining full horizons, separate all-width semantic probes and synthetic epoch-boundary/failure probes with explicit overrides/identities; skips cannot pass GPU readiness. Gate execution depends on T044, T051 and T053.
- [X] T028 [US3] Implement `prepare --campaign-root --cpu-evidence [--reference-manifest]` in `scripts/run_tinystories_matformer_widths.py` to verify tested snapshot/config/preflight bindings, adopt the matching exclusive reservation, establish source/campaign/diagnostics/launchers/logs/runs/reports paths and a durable launch plan, and record unavailable history separately without relabeling old evidence or submitting jobs.
- [X] T029 [US3] Implement a durable standalone barrier in `scripts/run_tinystories_matformer_widths.py` using T040's strict selected-run terminal reader: publish `launchers/standalone-barrier.json` only after all four fresh one-epoch terminals validate, bind campaign/source/preflight/run-contract/checkpoint/sidecar/supporting hashes, revalidate after restart and at elastic worker entry, and block new admission/report an error if previously valid evidence changes.
- [X] T030 [US3] Implement `queue --campaign-root [--once]` in `scripts/run_tinystories_matformer_widths.py` with queue/writer locks, monotonic durable intents before sbatch, reconciliation against squeue/sacct/worker records, blocked uncertain submissions, live user-wide two-running/four-submitted or stricter association/QoS ceilings, unrelated-job counting, exclusions `gpu-[05,50,51,54]`, one GPU/process, conservative admission when running limits cannot be guaranteed, and per-run-horizon progress/ETA.
- [X] T031 [US3] Implement `worker --campaign-root --arm --attempt-id` in `scripts/run_tinystories_matformer_widths.py` to verify its intent, gates, snapshot/config hashes, locks and stage barrier before invoking the existing trainer; distinguish fresh unoccupied state, valid own-checkpoint continuation, zero-step completion-only recovery and blocked ambiguous/invalid state, preserving costs/logs for every attempt and never restarting fresh in an occupied identity.
- [X] T032 [US3] Implement restart-safe `report --campaign-root [--reference-manifest]` in `scripts/run_tinystories_matformer_widths.py` using T043/T050's analyzer interfaces; validate existing output manifests/hashes before reuse, persist independent production/new-report/combined-report statuses with outstanding reasons, retain valid new output when history fails, return nonzero for an outstanding requested combined deliverable, and submit no GPU work.
- [X] T033 [US3] Run the resume, failure, barrier, gate-provenance, queue/worker and report-recovery fixtures in `tests/test_matformer_widths_campaign.py` and `tests/test_matformer_widths_queue.py` after T044/T051; record results in `specs/015-tinystories-matformer-widths/verification.md`, including unchanged-state rejection, durable-boundary preservation and zero duplicate submissions under restart.

### Separately authorized execution — do not run in numeric phase order

**Prerequisites**: T033, T044, T051, T053–T055 and later conversation execution authorization. Diagnostics and production readiness remain distinct from authorization. Use the exact quickstart commands after verifying their implemented interfaces.

- [X] T034 [US3] After execution preparation is authorized, audit the pinned real corpus/tokenizer and unused `MW_ROOT`, run campaign preflight, snapshot-bound CPU diagnostics and prepare through `scripts/preflight_tinystories_matformer_widths.py` and `scripts/run_tinystories_matformer_widths.py`; inspect `$MW_ROOT/campaign/campaign_manifest.json`, `$MW_ROOT/campaign/preflight.json` and `$MW_ROOT/diagnostics/cpu-gate.json`, recording real counts/data hashes/reservation and snapshot evidence in `specs/015-tinystories-matformer-widths/verification.md`.
- [X] T035 [US3] After GPU diagnostic authorization and T034, submit the implemented GPU diagnostics from `scripts/preflight_tinystories_matformer_widths.py` through sbatch with live limits and exclusions enforced; retain commands, job IDs, overrides, numerical comparisons, clipping artifacts and failure probes under `$MW_ROOT/diagnostics/`, and validate `$MW_ROOT/diagnostics/gpu-gate.json` against the exact CPU snapshot/config/preflight before declaring readiness.
- [X] T036 [US3] After production authorization and passed T035 gates, use `scripts/run_tinystories_matformer_widths.py` to complete only the four fresh standalones at 87,132 updates and 713,785,344 tokens each; reconcile each continuation from its own durable state, validate all terminal/ordinary-evaluation/resource evidence, and publish/revalidate `$MW_ROOT/launchers/standalone-barrier.json` before admitting any elastic production job.
- [ ] T037 **CANCELLED by user, 2026-09-22; original scope not achieved.** [US3] After T036's validated barrier, use `scripts/run_tinystories_matformer_widths.py` to complete all five fresh elastic arms at 348,528 updates and 2,855,141,376 tokens each; reconcile restarts/attempt costs, verify nine first-epoch and five full elastic traces, actual exposure and C1/C3 clipping coverage, and record nine-run totals of 17,130,848,256 assigned tokens plus separate replay/diagnostic costs in `specs/015-tinystories-matformer-widths/verification.md`.

**Checkpoint**: T033 establishes CPU continuation/scheduling behavior. SC-002/004 require the applicable CPU/GPU evidence; SC-005/006 require T036/T037's actual complete production artifacts.

## Phase 6: User Story 4 — Audit the 24 new terminal endpoints (Priority: P2)

**Goal**: Strictly freeze and publish the new campaign independently of historical availability.

**Independent test**: Supply nine full-budget terminal fixtures yielding four standalone plus twenty elastic endpoints; compare parsed CSV/JSON, follow provenance and per-run diagnostics, and reject missing/duplicate/corrupt/nonterminal inputs without replacing a valid report.

### Tests

- [X] T038 [P] [US4] Create new-report fixtures and acceptance tests in `tests/test_matformer_widths_reporting.py` for exactly nine runs/24 endpoints, dense source fractions, semantic CSV/JSON equality, provenance and real C1/C3 clipping-reader integration; reject missing/duplicate widths, wrong counts/budgets, stale hashes, nonfinite/inconsistent loss-perplexity, wrong role/manifest/target counts, best/early/trailing inputs, incomplete clipping and invalid traces, while preserving a prior valid publication.

### Implementation

- [X] T039 [P] [US4] Extend endpoint records and table identity in `src/evaluation/optimizer_ownership.py` to key by campaign/run/source fraction/active dimension and retain group, canonical arm, seed, exact count/convention, actual/assigned budgets, checkpoint/evaluation/config/source provenance, exposure/clipping/resource fields and explicit not-applicable nulls; preserve standalone source widths despite local dense fraction 1.
- [X] T040 [US4] Extend strict selected-run terminal inspection in `src/evaluation/optimizer_ownership.py` for schema 4, checking full assigned budgets, frozen checkpoint and ordinary-validation hashes/identities, inherited target-token-weighted causal loss with 285 sequences/36,195 targets, finite exp(loss) perplexity, counts/grid, committed traces, real C1/C3 sidecars/references and resource completeness disclosures; retain legacy full-campaign strictness and expose the reader for the standalone barrier.
- [X] T041 [US4] Extend schema-4 freeze/new-report publication in `src/evaluation/optimizer_ownership.py` to require all nine runs and exactly 24 endpoints, write matching `endpoints.csv`/`endpoints.json` using stable nested-value serialization, and atomically publish manifests binding input/output hashes, endpoint counts and completion status; partial output must never satisfy a complete report or barrier.
- [X] T042 [US4] Make per-run scalar training/validation trajectories, exposure/clipping and resource diagnostics in `src/evaluation/optimizer_ownership.py` use the selected physical grid and explicit source widths; generate diagnostics for all nine runs from saved artifacts, disclose incomplete resources, and keep new-only figures distinct from combined-comparison completion.
- [X] T043 [US4] Extend `preflight`, `freeze` and `report` schema dispatch in `scripts/analyze_tinystories_optimizer_ownership.py` without changing legacy flags/defaults; enforce freeze's exactly-one run-root/run-dir input rule, return nonzero with discrepant file/identity on invalid input, and keep reporting free of training or holdout evaluation.
- [X] T044 [US4] Run US4 fixtures in `tests/test_matformer_widths_reporting.py` and actual short-run C1/C3 reader integration from `tests/test_matformer_widths_campaign.py`; record matching 24-row exports, nine diagnostic sets, corruption rejection and atomic publication checks in `specs/015-tinystories-matformer-widths/verification.md`. This fixture gate must pass before T034–T037.

### Saved production evidence

- [ ] T045 **CANCELLED by user, 2026-09-22; original scope not achieved.** [US4] After T037, freeze nine actual full-budget terminals and publish `$MW_ROOT/reports/frozen/frozen_manifest.json`, `$MW_ROOT/reports/new/endpoints.csv`, `$MW_ROOT/reports/new/endpoints.json` and all nine per-run diagnostics via `scripts/analyze_tinystories_optimizer_ownership.py`; validate 24 endpoints and separate new-report completion from historical availability in `specs/015-tinystories-matformer-widths/verification.md`.

**Checkpoint**: T044 establishes implementation acceptance; T045 establishes actual 24-endpoint completion even if the combined comparison remains outstanding.

## Phase 7: User Story 5 — Compare new results with historical standalones (Priority: P2)

**Goal**: Publish 28 distinct endpoints and four combined figure files while preserving repeated measurements at shared sizes.

**Independent test**: Combine 24 valid new fixture points with exactly four schema-1 standalone terminals; inspect row identity, input rejection, five connected four-point elastic curves plus eight disconnected standalone markers, legends, annotations, and exact coincident coordinates.

### Tests

- [X] T046 [P] [US5] Add combined-report fixtures to `tests/test_matformer_widths_reporting.py` for exactly 28 endpoints/thirteen runs, independent old/new records at dimensions 64/128/256, historical g750=.75/192/213,568, four-only historical selection even when unrelated old elastic files are absent, manifest-integrity rejection, missing/incompatible/corrupt selected-history failures, preserved new output, CSV/JSON parity and structural PNG/PDF figure checks with coincident markers at unchanged coordinates.

### Implementation

- [X] T047 [P] [US5] Add selected historical-source validation in `src/evaluation/optimizer_ownership.py` that verifies the original schema-1 frozen manifest/preflight integrity and exactly ST-g250/ST-g500/ST-g750/ST-g1000 terminal source/checkpoint/evaluation/budget/count evidence, excludes historical elastic/corrected/inverse-membership endpoints, reads history without rewriting it, and retains existing full-campaign callers' strict checks.
- [X] T048 [US5] Implement combined export/manifest publication in `src/evaluation/optimizer_ownership.py` for exactly 24 new plus four historical rows, preserving physical/run identities and matching `combined_endpoints.csv`/`combined_endpoints.json`; unavailable/incompatible history records combined-outstanding, invalid present history fails explicitly, and no partial output replaces a valid complete comparison or hides new-report success.
- [X] T049 [US5] Implement `loss_vs_parameters` and `perplexity_vs_parameters` PNG/PDF figures and descriptive interpretation in `src/evaluation/optimizer_ownership.py`: five consistently colored connected four-point curves, four fresh and four historical disconnected markers, concentric distinguishable overlaps at exact counts, required standalone legend labels and seed/dataset/ordinary-validation/count/terminal/budget annotations; compare primarily with fresh baselines and separate history/representation from C3 clipping/block-size effects without across-seed or equal-compute claims.
- [X] T050 [US5] Add `report-matformer-widths --manifest --reference-manifest --output-dir` to `scripts/analyze_tinystories_optimizer_ownership.py`, call strict combined validation/publication, identify invalid input in nonzero errors and leave existing comparison interfaces intact; unavailable required history returns nonzero.
- [X] T051 [US5] Run combined table/figure/input-validation and historical compatibility fixtures in `tests/test_matformer_widths_reporting.py`; record exact 28-row identity preservation, four figure outputs, five curves/eight standalone markers, missing-history status and unchanged legacy reader behavior in `specs/015-tinystories-matformer-widths/verification.md`. This fixture gate must pass before production.

### Saved production evidence

- [ ] T052 **CANCELLED by user, 2026-09-22; original scope not achieved.** [US5] After T045, revalidate the four selected originals from `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/reports/complete-20260910T181951Z/frozen/frozen_manifest.json` and publish `$MW_ROOT/reports/combined/combined_endpoints.csv`, `$MW_ROOT/reports/combined/combined_endpoints.json`, `$MW_ROOT/reports/combined/loss_vs_parameters.png`, `$MW_ROOT/reports/combined/loss_vs_parameters.pdf`, `$MW_ROOT/reports/combined/perplexity_vs_parameters.png` and `$MW_ROOT/reports/combined/perplexity_vs_parameters.pdf`; inspect the figures and record hashes/results in `specs/015-tinystories-matformer-widths/verification.md`, leaving this task incomplete with an explicit reason if required history is unavailable/invalid.

**Checkpoint**: T051 establishes reporting behavior with fixtures; only T052 establishes the requested complete historical comparison.

## Phase 8: Polish & Cross-Cutting Concerns

### Before execution admission

- [X] T053 Run the three new suites `tests/test_matformer_widths_campaign.py`, `tests/test_matformer_widths_reporting.py`, and `tests/test_matformer_widths_queue.py` plus the complete relevant ownership/config/checkpoint/metrics/reporting/model/data/CLI regression command listed in `specs/015-tinystories-matformer-widths/quickstart.md`; record commands, pass/fail/skip counts, environment and tested hashes in `specs/015-tinystories-matformer-widths/verification.md`, preserving schemas 1–3 signatures and distinguishing CUDA skips from GPU passes.
- [X] T054 [P] Reconcile implemented commands, failure/restart procedures, exact gate invalidation rules, source snapshot use, full horizons, reference-outstanding behavior and current pending/completed states in `docs/tinystories-matformer-widths-experiment.md` and `specs/015-tinystories-matformer-widths/quickstart.md`; remove planning-only interface claims only where implementation has actually been verified.
- [X] T055 [P] Review implementation against `.specify/memory/constitution.md` and all FR-001–026/SC-001–010, recording evidence gaps in `specs/015-tinystories-matformer-widths/verification.md`; confirm reuse of the existing trainer/serializer/helpers, no generic registry/scheduler framework or per-update model copies, bounded metrics, strict historical compatibility and zero holdout evaluations before execution admission.

### After saved production/report evidence

- [X] T056 **Rescoped at user closeout, 2026-09-22.** Document the saved four-standalone/S1/S2 comparison and four historical baselines in `docs/tinystories-matformer-widths-experiment.md`; retain the 16-endpoint selected report, describe exposure and measured costs, and explicitly exclude incomplete C1–C3 and causal/across-seed claims. The original all-arm interpretation after T045/T052 is cancelled.
- [X] T057 Reconcile checkboxes, final statuses, evidence and artifact links in this file, `verification.md`, `spec.md`, `plan.md`, `quickstart.md` and the runbook. Distinguish passed CPU/GPU readiness, six completed production runs, cancelled nine-run completion and unproduced 24/28-endpoint reports; retain original unmet SCs and record the user's reduced scope.

## Dependencies & Execution Order

### Story dependency graph

```text
Setup T001–T002 → Foundations T003–T005 → US1 T006–T013 (MVP)
                                              ↓
                                      US2 T014–T021
                                         ↙         ↘
               US3 restore/operations             US4 fixture implementation
               T022–T028, T030–T031                 T038–T044
                         ↑                              ↓
                         └─ T040 → T029 barrier      US5 T046–T051
                                                        ↓
                           T043 + T050 → T032 → T033
                                                        ↓
                                          T053 → T054–T055
                                                        ↓
                    later authorization → T034 → T035 → T036 → T037
                                                        ↓
                                     T045 (24 actual endpoints)
                                                        ↓
                              valid history → T052 (28 actual endpoints)
                                                        ↓
                                                  T056 → T057
```

T056/T057 also record the actual outstanding combined status if T052 cannot complete; they must not mark T052 or combined SCs complete in that case.

### Detailed ordering and shared-file constraints

- T003–T005 are sequential; no story implementation begins before they pass. T006 and T007 may be developed together, then T008–T013 integrate them.
- US2 depends on US1's validated contracts. T014/T015 are independent test-file edits; T016–T020 integrate sequentially before T021.
- US3 continuation work T022/T024/T025/T026 depends on US2. T023 may be written with T022; queue fixtures use mocked dependencies until the strict terminal reader and report interfaces are ready.
- US4 depends on US1/US2 contracts and artifacts, but its fixture implementation does not require production or the US3 launcher. T038/T039 may start together; T040 then supplies T029's strict barrier reader. Finish T041–T044 before completing T029's integration acceptance.
- US5 fixture implementation depends on US4; T046/T047 may start together. T048–T051 are sequential. T032 depends on both T043 and T050, and T033 depends on T032/T044/T051. This is a one-way implementation dependency, not a production/reporting cycle.
- T027 can be implemented before fixture completion, but cannot produce passed gate evidence until its required suites exist and pass. T028 depends on T027's evidence contract; T029 depends on T028/T040; T030/T031 require the implemented barrier before final integration. Never enable elastic admission using a placeholder reader.
- T053 follows all implementation/fixture tasks through T033/T044/T051; T054/T055 can then proceed concurrently in distinct files. T034 requires all three and execution-preparation authorization; T035 additionally requires GPU diagnostic authorization, and T036/T037 require production authorization. Source/config changes after a gate invalidate it and require verification of the changed snapshot.
- T036 must finish all four strictly validated standalones before T037 can submit any full-budget elastic. Separate short diagnostic probes do not satisfy this barrier.
- T045 follows T037. T052 follows T045 and revalidation of all four historical sources. Missing history blocks T052 alone, preserving T045's valid result.
- Edits to `src/evaluation/optimizer_ownership.py`, `src/training/run.py`, `tests/test_matformer_widths_campaign.py`, `tests/test_matformer_widths_reporting.py`, the analyzer CLI or `verification.md` must be serialized whenever tasks share that file. Explicit task dependencies take precedence over numeric phase placement.

## Parallel execution examples by story

These are work packages, not instructions to spawn agents or run GPUs. Each pair starts only after its preceding phase prerequisites pass.

| Story | Concurrent tasks | Why independent |
| --- | --- | --- |
| US1 | T006: campaign tests; T007: schema-4 YAML | Separate files using the same fixed design contract; integrate with T008–T013 afterward. |
| US2 | T014: real-model ownership tests; T015: compact-accounting tests | Different test files; neither edits the other's fixtures. |
| US3 | T022: resume/failure tests; T023: queue/restart tests | Campaign tests and mocked scheduler tests occupy distinct files. |
| US4 | T038: terminal/report tests; T039: endpoint records | Separate test and implementation files; strict-reader integration follows. |
| US5 | T046: combined-report tests; T047: selected-history reader | Separate test and implementation files; complete exports/figures afterward. |

Other opportunities: T001/T002 establish separate documents, and T054/T055 update separate documentation/evidence files after regression checks. No parallel production admission may bypass the standalone barrier or live Slurm limits.

## Requirement coverage

| Requirements | Primary tasks |
| --- | --- |
| FR-001–002: distinct fresh schema/matrix and compatibility | T003–T009, T012–T013, T024, T031, T053 |
| FR-003: standalone-first barrier | T023, T029–T031, T033, T036–T037 |
| FR-004–006: sampling, deterministic data, grid/boundaries | T006–T011, T014–T020, T022, T037 |
| FR-007–010: histories, absent gradients, owners/clipping | T014, T016–T017, T020–T022, T035 |
| FR-011: strict preflight | T006–T013, T034 |
| FR-012–013: exact restore and complete-update failure boundary | T022, T024–T025, T031, T033, T035 |
| FR-014–016: traces, clipping, storage and attempt resources | T015, T018–T023, T026, T033, T037, T039–T045 |
| FR-017–018: evidence gates and authorized operations | T023, T027–T037, T044, T051, T053–T055 |
| FR-019–020, FR-022: terminal-only 24/28 exports | T025, T032, T038–T045, T046, T048, T050–T052 |
| FR-021: four-only strict historical reuse | T046–T048, T051–T052 |
| FR-023–025: figures, repeated points and interpretation | T046, T049, T051–T052, T056 |
| FR-026: evidence-aligned documentation and completion | T001–T002, T013, T021, T033–T037, T044–T045, T051–T057 |

## Implementation Strategy

### MVP first

Complete T001–T013 and validate US1 independently. Deliver a CPU-verifiable nine-run protocol with strict geometry, scientific identity and inherited controls. Do not equate this MVP with optimizer acceptance, GPU readiness or campaign completion.

### Incremental delivery

1. Add US2's real-update semantics and bounded metrics, then US3's complete-state continuation checks.
2. Build US4/US5 strict readers and fixture reports before operational admission; use the same reader for the standalone barrier.
3. Integrate snapshot-bound gates, queue/worker reconciliation and restart-safe reporting; pass the required CPU compatibility suites.
4. After the applicable later authorization, prepare the real inputs, run Slurm diagnostics, validate four complete standalones, and then complete five elastics using only own-run continuation.
5. Freeze and publish 24 actual endpoints independently; publish the 28-point comparison only after all four originals validate. Interpret saved evidence and reconcile task/runbook statuses.

### Completion rules

Each story has a fixture/CPU acceptance boundary and, where applicable, a separate actual-evidence task. A passing mocked scheduler is not evidence of production stage order; a passed GPU probe is not full-budget completion; a 24-row fixture is not a completed new report. Preserve these distinctions when checking tasks and evaluating SC-001–010.
