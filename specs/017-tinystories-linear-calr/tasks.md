# Tasks: TinyStories Linear S1/S2 CaLR Comparison

**Input**: Design artifacts in `specs/017-tinystories-linear-calr/`: spec.md, plan.md, research.md, data-model.md, contracts/, quickstart.md.
**Prerequisites**: Approved design; Python `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python`; existing PyTorch/Transformers/NumPy/PyYAML/Matplotlib/pytest environment, with no new packages.
**Organization**: Shared prerequisites followed by US1 (P1), US2 (P1), US3 (P2), and cross-cutting acceptance. Verification is explicitly required by EX-004/005.

## Format and execution conventions

Each task uses `- [ ] Tnnn [P?] [USn?] Description with exact paths`. `[P]` means independent files within the stated dependency wave, not permission to ignore prerequisites. Paths are repository-relative unless absolute. Keep the existing trainer and shallow structure; do not build a scheduler registry or second campaign framework.

This file plans work; generating it does not implement, reserve an external output root, submit jobs, train, commit or publish results. External root preparation and GPU execution tasks require their later authorizations. A missing authorization leaves those tasks pending. Tests use temporary fixture roots and mocked Slurm until execution is authorized. Historical references remain read-only throughout.

## Phase 1: Setup

**Purpose**: Establish compatibility evidence and implementation boundaries without altering historical results.

- [ ] T001 Record baseline commands, installed versions, repository/source identity and relevant historical regression outcomes in specs/017-tinystories-linear-calr/verification.md; run existing optimizer ownership, continuation, reporting, queue, metrics and reproducibility tests with the specified Python and distinguish pass/fail/skips.
- [ ] T002 Capture schema 1–5 resolved configuration/signature and S1/S2/C4 schedule/restore invariants in tests/test_linear_calr_campaign.py using tests/fixtures/optimizer_ownership_legacy_signatures.json without regenerating historical expected signatures; document the tested baseline in specs/017-tinystories-linear-calr/verification.md.

## Phase 2: Foundational (Blocking prerequisites)

**Purpose**: Provide new-only identity and schedule primitives shared by the stories.

- [ ] T003 Add immutable version-1 schedule-contract parsing/validation and semantic slicing/shared-or-per-granularity eligibility in src/utils/config.py; bind grid, counts/definition, policies/exponents/bounds, peak/warmup/horizon, effective-rate policy, ownership and run identity, reject malformed contracts, and leave old resolution/serialization branches unchanged.
- [ ] T004 Extend src/utils/reproducibility.py with new-contract-only scientific/optimizer/schedule hash inputs while preserving old signatures and initialization/action/data stream derivation independently of new arm/run names.
- [ ] T005 Implement pure validated warmup-polynomial rates and complexity-log exponents in src/training/schedules.py: integer p in [0,T], 0<W<T, positive counts, distinct CaLR extrema, finite positive gamma, exact zero at T, and explicit analytic-versus-applied evidence names; nominal gamma is always one.

**Checkpoint**: T001–T005 complete before story implementation. No historical signature or restore semantics may change.

## Phase 3: User Story 1 — Prepare a scientifically matched comparison (P1, MVP)

**Goal**: Resolve and audit exactly four matched definitions without training.
**Independent test**: CPU fixture preparation proves exact controls, actual-model counts, analytic rates, fresh equal weights and complete matching action/data streams; undeclared differences fail specifically.

### Verification

- [ ] T006 [P] [US1] Add four-arm/schema/count/budget tests and negative closed-difference audits in tests/test_linear_calr_campaign.py; cover extra arms/seeds, geometric grids, correction, corpus/clipping/probability changes, counterpart differences, schedule pairs, ownership pairs and historical signature identity.
- [ ] T007 [P] [US1] Add analytic schedule tests in tests/test_linear_calr_schedule.py for all widths at 0/63/64/65, midpoint decay and T−1/T; check intermediate exponents, finite nonnegative full-horizon rates, invalid positions/horizons/counts and exact anchor values from contracts/schedule-and-update.md.

### Implementation

- [ ] T008 [P] [US1] Create configs/controlled_exps/tinystories_instruct_linear_calr.yaml with schema 6, campaign tinystories-linear-s1-s2-calr-v1, exactly S1-linear-poly/S1-linear-CaLR/S2-linear-poly/S2-linear-CaLR, seed 42, all inherited fixed controls, .008/64/348528 schedule and eight explicit saved reference mappings from spec.md.
- [ ] T009 [US1] Extend src/evaluation/optimizer_ownership.py to expand schema 6 into arm-qualified run IDs, fresh normal-constructor configs and fixed budgets; implement separate enumerated resolved-field allowlists for counterpart, schedule-policy and ownership comparisons, including consequential hashes without wildcard exclusions.
- [ ] T010 [US1] Validate actual-model identity-deduplicated active trainable counts including embeddings/head in src/evaluation/optimizer_ownership.py against 377408/426560/475712/524864 and separate reporting counts 115264/164416/213568/262720; resolve FFN 64/128/192/256 and formula-derived gamma maps without substituting width fractions or reporting counts.
- [ ] T011 [US1] Add inherited tokenizer/corpus/manifest/control checks and analytic audit output in src/evaluation/optimizer_ownership.py; verify 5576448 designated sequences/epoch, excluded tail 43, ordinary validation only, 87132 updates/epoch, 8192 tokens/update and exact run/campaign totals, leaving controller/final holdout unused.
- [ ] T012 [US1] Extend tests/test_linear_calr_campaign.py to compare fresh state tensors and complete deterministic width/action and data-cursor evidence across all four arms and original matching-scope counterparts; exercise stream equivalence through the declared horizon using bounded-memory iteration rather than training or storing all batches.
- [ ] T013 [US1] Expose CPU campaign definition/preflight audits through scripts/analyze_tinystories_optimizer_ownership.py using schema-6 resolution, explicit reference selection and analytic labels; preserve existing CLI operations and make failed control/count/stream checks return nonzero.

**Checkpoint**: T006–T013 pass; US1 is the MVP. This proves definitions, not real GPU readiness or completed training. Requirements: FR-001–004/007–009, EX-001/004, SC-001.

## Phase 4: User Story 2 — Execute and continue faithful training (P1)

**Goal**: Apply selected-width rates faithfully, restore own-arm state safely, and gate operations with durable evidence.
**Independent test**: CPU actual-update/manual-reference, continuation/failure and mocked-queue suites pass for every arm. Later authorized real-shape BF16 diagnostics establish GPU readiness separately; production establishes exact terminal budgets.

### Verification

- [ ] T014 [P] [US2] Extend tests/test_linear_calr_schedule.py with actual all-group AdamW/manual-reference updates using populated moments and decay, mixed widths, p=0 zero-LR first-step semantics, unselected S2 histories, shared parameter identities, no multiplier accumulation and inherited full-shaped zero-gradient tail behavior.
- [ ] T015 [P] [US2] Add own-arm uninterrupted-versus-resume cases in tests/test_linear_calr_resume.py before/at/after warmup, representative decay, every 87132-update epoch boundary and terminal T; use state-seeded probes where appropriate, inherited rtol=1e-6/atol=1e-7 or stricter existing assertions, and compare weights, histories, counters, RNGs, cursors, rates and accounting without claiming full training.
- [ ] T016 [US2] Add pre-mutation rejection and failure-injection tests in tests/test_linear_calr_resume.py for cross-arm/grid/scope/count/gamma/horizon/malformed bundles, optimizer/rate-restoration/scheduler/accounting failures, poisoned save refusal, last durable checkpoint preservation and terminal-output recovery with zero optimizer steps.
- [ ] T017 [P] [US2] Add mocked lifecycle/admission tests in tests/test_linear_calr_queue.py for absent authorization, missing/failed/stale CPU/GPU gates, CPU fallback, changed snapshots/configs, occupied roots, duplicate/uncertain submissions, user-wide/stricter limits, wrong-arm continuation and worker-versus-job reconciliation.

### Runtime implementation

- [ ] T018 [US2] Integrate the exact nominal gamma=1 lambda into src/training/optimizer_state.py using existing GlobalSchedulerClock and unchanged collection layouts; keep all S2 group rates synchronized at durable boundaries and preserve legacy scalar/C4 clocks.
- [ ] T019 [US2] Implement the new-contract-only temporary all-group LR transaction in src/training/steps.py for S1 and S2: validate nominal rates, assign/capture selected effective rates, enter the existing unsafe boundary, step one owner, restore each nominal group in finally before advancing/synchronizing, and poison post-mutation failures without clearing unsafe state or publishing success.
- [ ] T020 [US2] Stream committed applied-rate/action/data evidence and resource/exposure watermarks in src/utils/metrics.py with bounded memory; include width/owner/count/gamma/global p/all-group LR, distinguish nominal clock fields, and separate validation and failed-attempt costs without changing legacy metric signatures.
- [ ] T021 [US2] Extend src/training/checkpointing.py with new-contract-only whole-bundle validation before install: immutable hashes/identity, exact nominal reconstruction, separate last applied record at step−1, group/history counters, RNG/cursor/accounting and trace watermarks; reject update at T and preserve all old restore branches.
- [ ] T022 [US2] Reconcile new contracts, committed summaries and terminal recovery in src/training/run.py; at full budget recreate missing ordinary-validation/terminal outputs from the own durable checkpoint with zero extra training, preserving unsafe-state refusal and last valid durable state.

### Operational implementation

- [ ] T023 [US2] Implement snapshot and CPU modes in scripts/preflight_tinystories_linear_calr.py with immutable source/config hashes, full-horizon analytic and actual-update/restore gates for all arms, commands/outcomes/log hashes and retained failed records; use fixture roots until external preparation is authorized.
- [ ] T024 [US2] Implement real-shape CUDA-required gpu diagnostics in scripts/preflight_tinystories_linear_calr.py for every arm at d64/l4/h4, batch 64/context 128, actual BF16, every width, nonzero moments, schedule/continuation boundaries and throughput/optimizer-state/CUDA-peak evidence bound to the same CPU snapshot/config; reject CPU fallback and label seeded probes accurately.
- [ ] T025 [US2] Implement atomic prepare and report dispatch in scripts/run_tinystories_linear_calr.py using existing operational helpers; require tested snapshot and CPU evidence, reserve only fresh identities, generate four resolved runs and independent readiness/submission/execution/terminal/comparison states, and never write to saved references.
- [ ] T026 [US2] Implement queue/worker admission and durable attempt reconciliation in scripts/run_tinystories_linear_calr.py; require separate durable explicit execution authorization plus matching passing CPU/all-arm GPU gates, reuse scripts/train_cuda_required.py and existing locks/intents/ledgers, enforce cscc-gpu-p/cscc-gpu-qos, one GPU/no requeue, gpu-[05,50,51,54] exclusions and user-wide two-running/four-submitted or stricter limits, reconciling uncertain submission before retry.
- [ ] T027 [US2] Implement own-arm continuation, scheduler/worker completion reconciliation and exact terminal accounting in scripts/run_tinystories_linear_calr.py; require 348528 updates/2855141376 training tokens per arm and 1394112/11420565504 campaign totals, preserve failed attempt costs and do not equate job disappearance with completion.
- [ ] T028 [US2] Run focused CPU schedule/resume/queue checks and relevant historical S1/S2/C4 optimizer/checkpoint/metrics/reproducibility regressions; record exact commands, results, limitations and source/config bindings in specs/017-tinystories-linear-calr/verification.md, fixing regressions before readiness use.

### Later authorized execution — leave pending until its gate is satisfied

- [ ] T029 [US2] After external preparation authorization, use scripts/preflight_tinystories_linear_calr.py snapshot/cpu and scripts/run_tinystories_linear_calr.py prepare to reserve and audit /nfs-stor/ivo.navarrete/results/elasticnn/tinystories-linear-s1-s2-calr-v1; retain bound readiness records and reference discovery without submitting GPU jobs.
- [ ] T030 [US2] After explicit diagnostic execution authorization, submit snapshot-bound scripts/preflight_tinystories_linear_calr.py gpu through the existing Slurm admission helpers; collect passing real-shape BF16 evidence for all four arms under the declared live limits and record commands/failures/device tolerances in specs/017-tinystories-linear-calr/verification.md.
- [ ] T031 [US2] After separate production authorization, T030 passing gates and T034–T041 reporting-fixture checks, use scripts/run_tinystories_linear_calr.py queue/worker from the tested snapshot to fresh-start exactly four arms and own-state continuations only; reconcile all attempts and full-budget checkpoints/ordinary endpoints before recording terminal completion in specs/017-tinystories-linear-calr/verification.md.

**Checkpoint**: Code/CPU correctness is T014–T028; actual execution is T029–T031. Missing authorization or GPU skips cannot count as readiness or SC-004 completion. Requirements: FR-003–006/008/010–012/014, EX-001–005, SC-002–004/008.

## Phase 5: User Story 3 — Interpret all widths against validated references (P2)

**Goal**: Validate historical terminals read-only and publish correct endpoint, comparison and measured-figure evidence.
**Independent test**: Provenance-complete fixture reports contain 36 endpoints, 48 required pairs, four interactions and correctly separated seven-series panels; absent terminal/provenance/trajectory evidence remains explicitly incomplete.

### Verification — complete before production

- [ ] T032 [P] [US3] Add native/legacy reference and terminal-admission fixtures in tests/test_linear_calr_reporting.py covering eight saved runs/20 reference endpoints, actual saved schemas, full checkpoint/budget/ordinary-manifest/hash proof, missing artifacts, intermediate/best checkpoints, stale dormant legacy counters, immutable references and exp(loss) consistency.
- [ ] T033 [US3] Add report arithmetic/figure/status tests in tests/test_linear_calr_reporting.py for 36 unique endpoints, 48 required pair directions/ratios/gaps, four interactions, seven exact labels, primary .008 versus supplemental .004 separation, unconnected standalones at update 87132, measured-only LR/progress, PNG/PDF/manifest binding and atomic complete publication with nonzero incomplete/error status.

### Implementation

- [ ] T034 [US3] Add native read-only terminal/reference admission in src/evaluation/optimizer_ownership.py for four new arms, original S1/S2 and four standalones; validate actual identities/configs, complete saved checkpoints/budgets, ordinary role/manifests/aggregation, counts, finite endpoints and hashes rather than trusting summaries or aggregate tables.
- [ ] T035 [US3] Add a saved-schema-specific read-only adapter in src/evaluation/optimizer_ownership.py for supplemental S1/S2 .004 using validated checks from scripts/plot_tinystories_s1_peak_lr.py and scripts/plot_tinystories_s2_peak_lr.py; require actual terminal checkpoint/config/ordinary-metric provenance, disregard unmaintained legacy counters and never synthesize newer manifests or missing measured traces.
- [ ] T036 [US3] Build schema-6 endpoints.csv, paired_differences.csv and interactions.csv in src/evaluation/optimizer_ownership.py with all data-model provenance/count/schedule/budget fields, 36 unique rows, required 8/8/16/16 comparison families and four signed interactions; put optional .004 comparisons in a separate explicitly labeled family/table.
- [ ] T037 [US3] Implement primary/supplemental PNG/PDF terminal loss/perplexity plots in scripts/analyze_tinystories_optimizer_ownership.py against active non-embedding counts using exactly standalone-cosine/S1-cosine/S2-cosine/S1-polynomial/S2-polynomial/S1-carl/S2-carl; use separate elastic width curves, unconnected standalone scatter, consistent styles and captions declaring peak settings/warmup/seed/budgets.
- [ ] T038 [US3] Implement recorded per-width validation-progress and actually applied LR plots in scripts/analyze_tinystories_optimizer_ownership.py with separate primary/supplemental baseline panels, standalone terminal points only at 87132, measured elastic evidence only and disclosed missing traces; retain technical CaLR naming outside the carl plotting alias and never reconstruct observed LR from schedules or next-update fields.
- [ ] T039 [US3] Add report-linear-calr CLI and atomic report publication in scripts/analyze_tinystories_optimizer_ownership.py; bind tables/figures through a plot-source manifest to source/config/checkpoint/evaluation/metric hashes, expose independent new-terminal/reference/trajectory/comparison states, return nonzero on incomplete/error and preserve valid new evidence.
- [ ] T040 [US3] Add descriptive all-width findings generation in src/evaluation/optimizer_ownership.py for matched CaLR/poly/cosine, both ownership scopes/interactions, tuned S1 .004 and S2 .008 practical references, g1000 standalone gaps and smaller-width costs; explain one seed, unequal horizons/exposure and cumulative LR/AdamW-decay changes without guaranteed improvement or mechanism claims.
- [ ] T041 [US3] Run provenance-complete reporting fixtures and existing historical reporting regressions, verify required counts/arithmetic/plot sources and capture commands/source/config bindings and limitations in specs/017-tinystories-linear-calr/verification.md before production admission.
- [ ] T042 [US3] After T031 full-budget completion, use scripts/run_tinystories_linear_calr.py report to revalidate all eight historical runs read-only and publish the real 36 endpoints/48 pairs/four interactions and measured PNG/PDF artifacts; write evidence-based all-width findings and independent incomplete statuses to specs/017-tinystories-linear-calr/experiment-report.md if any reference or trajectory lacks proof.

**Checkpoint**: T032–T041 are independently executable with fixtures before training. T042 requires real saved evidence and cannot claim comparison acceptance when references/trajectories are missing. Requirements: FR-013–019, EX-003/006, SC-005–007.

## Phase 6: Polish and cross-cutting acceptance

- [ ] T043 [P] Update specs/017-tinystories-linear-calr/quickstart.md with implemented CLI arguments, exact CPU verification and snapshot/prepare/diagnostic/queue/report commands, independent authorization boundaries and incomplete-evidence behavior; replace proposed-command language only for implemented operations.
- [ ] T044 Run the complete focused five-file CPU suite and relevant historical schema/config/S1/S2/C4/continuation/accounting/reporting/queue regressions with OMP_NUM_THREADS=1 and the specified Python; validate quickstart commands in temporary roots without GPU submission and record final commands/pass/fail/skips and tested source/config hashes in specs/017-tinystories-linear-calr/verification.md.
- [ ] T045 Reconcile every FR-001–019, EX-001–006 and SC-001–008 against saved evidence in specs/017-tinystories-linear-calr/experiment-report.md; verify exact totals, all widths, failed-attempt costs, terminal hashes and all figure/table provenance, and leave unmet real-execution/reference/trajectory criteria incomplete rather than treating code tests as final acceptance.

## Dependencies and execution order

```text
Setup T001–T002 → Foundation T003–T005 → US1 T006–T013 (MVP)
                                         ├→ US2 code/CPU T014–T028
                                         └→ US3 fixtures/reporting T032–T041
US2 CPU + external preparation authorization → T029
T029 + diagnostic authorization → T030 (all-arm CUDA readiness)
T030 + T032–T041 + production authorization → T031 (four full runs)
T031 + validated references/trajectories → T042 (real reporting)
Implemented code → T043–T044; all required real evidence → T045
```

US2 consumes US1's resolved contracts. US3 fixtures consume US1 definitions but do not depend on production or US2 GPU execution; actual reporting requires new terminals and validated references. US2 and US3 both edit src/evaluation/optimizer_ownership.py or scripts/analyze_tinystories_optimizer_ownership.py indirectly through integration: serialize changes to shared files. Phase order is story priority, not a requirement to run production before writing reports. T031 explicitly waits for reporting fixture readiness.

Within US1: T006/T007 verification definitions precede implementation; T008 and T009 can be prepared concurrently after that wave, then T010–T013 integrate. Within US2: verification definitions precede runtime T018–T022; T023 consumes runtime/US1 audits, T024 adds GPU gate, T025–T027 integrate operations and T028 validates. T016 follows T015 in the same test file. Within US3: T033 follows T032 in the same file; T034–T036 share the campaign module, T037–T039 share the analysis script and run sequentially; T040 then T041 closes fixture readiness.

## Parallel execution examples

- **US1**: After foundation, T006 (`tests/test_linear_calr_campaign.py`) and T007 (`tests/test_linear_calr_schedule.py`) can be written together. After those definitions, T008 YAML and T009 campaign-module expansion use different files; align their declared contract before integration.
- **US2**: After US1, T014 schedule tests, T015 resume tests and T017 mocked queue tests use different files and can proceed together. T016 follows T015. Runtime integration remains sequential because rate/clock/checkpoint evidence must agree.
- **US3**: After US1, T032 reporting fixtures can proceed while US2 test tasks run. After T039 publication wiring, T040 campaign findings and T043 quickstart documentation use different files; T041/T044 validate the integrated result. Reporting test tasks within one file run sequentially.

`[P]` tasks: T006, T007, T008, T014, T015, T017, T032, T043. Parallel examples describe available work, not instructions to spawn agents.

## Implementation strategy

1. Deliver the US1 MVP first: exact definitions, closed audits, actual-model counts and deterministic streams; validate without training.
2. Deliver US2 runtime and CPU safety checks, then source-bound operational gates. In parallel where files permit, deliver US3 provenance-complete fixtures and reports before any production.
3. Complete documentation and integrated CPU regressions before freezing the tested snapshot. Source/config changes invalidate old gates and require new matching evidence.
4. With later authorizations, prepare the external root, perform all-arm real-shape CUDA diagnostics, then admit exactly four fresh production runs under live limits. Resume only own durable state.
5. Publish measured results only after terminal/reference validation. Missing evidence preserves valid new results and leaves comparison acceptance pending; full delivery requires T045 and all SC-001–008 regardless of quality ordering.

## Coverage and task totals

45 tasks: setup 2, foundation 3, US1 8, US2 18, US3 11, polish 3. US1 covers matched definitions/analytic preparation; US2 covers applied updates, continuation, poisoning, accounting, gates and execution; US3 covers read-only references, arithmetic, figures and interpretation. The later execution tasks are T029–T031 and T042; they remain separate from code/fixture completion.
