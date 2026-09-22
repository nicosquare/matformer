# Tasks: TinyStories S1 Fourfold LR Warmup

**Input**: Design documents in `specs/016-tinystories-s1-warmup/`: `spec.md`, `plan.md`, `research.md`, `data-model.md`, `contracts/`, and `quickstart.md`.
**Status**: Phases 1–4 complete (T001–T035); CPU preparation, runtime/continuation, admission and historical compatibility pass against fixtures. See [verification.md](verification.md). Reporting implementation (phase 5), real input/terminal audits, readiness, production, and comparison remain pending.
**Tests/Verification**: Explicitly required by FR-009 and the story acceptance scenarios. Add focused contract, runtime, reporting, and queue checks before their corresponding implementation; reuse existing fixtures and inherited tolerances.
**Organization**: Setup and compatibility prerequisites, then US1 (P1), US2 (P1), US3 (P2), then cross-cutting verification and execution. Story checkpoints establish independently testable software increments; real terminal and scientific completion require the final phase.

## Format and paths

- Checklist format: `- [ ] Tnnn [P?] [USn?] Action with file path`.
- `[P]` marks independent files that can be worked on together once their stated prerequisites pass. It does not override phase dependencies or permit concurrent edits to shared modules.
- Source and documentation paths are relative to the repository root. Extend existing modules; introduce no trainer, scheduler, model, or generic campaign framework.
- Interpreter: `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python`; no additional dependencies.
- Proposed fresh result root (`WARMUP_ROOT`): `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-s1-warmup-v1`. Recheck uniqueness during preparation; creating a task does not reserve it.
- Linear reference: `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1`; Geometric reference: `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1`. Historical inputs are read-only.
- Subsequent implementation instructions authorize software work. T053 requires subsequent GPU-diagnostic authorization; T054 requires production authorization. Prior campaigns' approvals do not apply, and passing a gate does not itself grant authorization (FR-019).

## Phase 1: Setup (Shared Experiment Structure)

**Purpose**: Establish evidence records and explicit scope without creating production artifacts.

- [X] T001 Create `specs/016-tinystories-s1-warmup/verification.md` with separate pending statuses for implementation checks, input/reference audits, CPU gate, GPU readiness, each terminal, new-only report, endpoint comparison, and early/report completion; record interpreter/dependency versions, source revision, commands, hashes, pass/fail/skip counts, and evidence links as later tasks execute.
- [X] T002 [P] Create `docs/tinystories-s1-warmup-experiment.md` with the exact two-arm matrix, controls/budgets, read-only reference selections, proposed artifact layout, CLI contracts, and separate implementation/GPU/production authorization boundaries from `specs/016-tinystories-s1-warmup/quickstart.md`.

## Phase 2: Foundational (Blocking Compatibility Prerequisites)

**Purpose**: Capture historical behavior before changing resolution or serialization. Complete both tasks before any story implementation.

- [X] T003 Capture pre-change resolved configuration and scientific-contract signatures for all schema-4 arms in `tests/fixtures/s1_warmup_schema4_signatures.json`, using the unchanged resolver; retain `tests/fixtures/optimizer_ownership_legacy_signatures.json` for schemas 1–3 and record the capture command/source hash in `specs/016-tinystories-s1-warmup/verification.md`.
- [X] T004 Add and run compatibility assertions in `tests/test_s1_warmup_campaign.py` for schema-1–4 signatures, omitted-arm selector defaults, old serialized field sets, and strict rejection of historical warmup 256; use the two signature fixtures and record the passing pre-change baseline in `specs/016-tinystories-s1-warmup/verification.md`.

**Checkpoint**: Historical identities and 64-update rules have a reproducible baseline. Do not regenerate golden signatures to accommodate a regression.

## Phase 3: User Story 1 — Establish a Controlled Warmup Comparison (Priority: P1) — MVP

**Goal**: Prepare exactly two fresh, counterpart-matched S1 protocols and auditable input/reference evidence without training.
**Independent test**: Expand both runs against saved-counterpart fixtures; verify physical counts, full four-epoch streams, and the closed warmup-only difference audit. Reject altered controls, identities, and invalid selected references. No production job is needed.
**Coverage**: FR-001–006, FR-012; SC-001 and historical-preservation portions of SC-008.

### Verification for User Story 1

- [X] T005 [P] [US1] Add protocol/preflight acceptance cases in `tests/test_s1_warmup_campaign.py` for exactly two arms, required schema-5 arm-qualified selectors, both physical grids/counts, fresh seed-42 construction, closed control differences, full budgets/streams, atomic publication, and rejection of extra arms, wrong warmup/LR/horizon/seeds/sampling/clipping/data/precision, enabled pre-nested warmup, reference initialization, and occupied identities.
- [X] T006 [P] [US1] Add selected-reference contract cases in `tests/test_s1_warmup_reporting.py` for ten selected terminals yielding sixteen historical endpoints, distinct shared-size standalones, closed partial Feature 015 acceptance, historical-generation device evidence, source immutability, and rejection of archived CPU attempts, cancelled/unselected arms, incomplete budgets, stale hashes, and invalid ordinary-validation provenance.

### Implementation for User Story 1

- [X] T007 [P] [US1] Add `configs/controlled_exps/tinystories_instruct_s1_warmup.yaml` with exact top-level keys `schema_version`, `campaign_id`, `common`, `arms`, `expected_data`, and `references`; declare schema 5, campaign `tinystories-optimizer-ownership-s1-warmup-v1`, arms `S1-linear-w256`/`S1-geometric-w256`, warmup 256, inherited controls/data hashes, and the pinned counterpart config hashes and five-arm selections from `specs/016-tinystories-s1-warmup/research.md`.
- [X] T008 [US1] Extend arm definitions, `campaign_widths`, `campaign_common`, `campaign_topology`, expansion, and manifest construction in `src/evaluation/optimizer_ownership.py` with strict per-arm schema-5 grids and `<campaign_id>-<arm_id>-s42` run IDs; reject missing/unknown arms and leave schemas 1–4 and `PINNED_COMMON` unchanged.
- [X] T009 [US1] Update schema-5 output eligibility/topology dispatch in `src/utils/config.py` and protocol-bound materialized/budget validation in `src/evaluation/optimizer_ownership.py`; accept 256 only for the fixed new identities, reject schema-5 warmup 64, and validate fully merged controls rather than allowing a caller-supplied warmup bypass.
- [X] T010 [US1] Bind grid/intervention/reference identities only into new scientific contracts in `src/evaluation/optimizer_ownership.py`, using the existing extra-field serializer in `src/utils/reproducibility.py`; preserve old serialization and version-1 initialization/action/data seed derivation independently of operational run names.
- [X] T011 [US1] Implement the saved-counterpart control-difference audit in `src/evaluation/optimizer_ownership.py` with explicit allowed field paths for warmup, consequential schedule values, and necessary identity/output/provenance metadata; publish resolved projections, file/contract hashes, actual differences, and named failures without ignoring whole scientific-control sections.
- [X] T012 [US1] Extend model/data preflight and expected traces in `src/evaluation/optimizer_ownership.py` to measure active counts excluding embeddings/LM head, verify corpus/tokenizer identities and disjoint roles, 5,576,448 designated sequences plus 43 excluded sequences, and exact own-counterpart action/batch equality across 348,528 updates/four epochs; preserve `randrange(4)` replacement sampling and report expected counts without enforcing quotas.
- [X] T013 [US1] Implement strict ten-terminal reference selection in `src/evaluation/optimizer_ownership.py` through `inspect_selected_terminals`, reusing or extracting the read-only device checks in `scripts/plot_tinystories_standalones.py`; validate saved controls, terminal hashes/steps/budgets/traces, physical counts, ordinary-validation membership/targets, and applicable job/precision/device evidence without requiring cancelled Feature 015 arms or modifying historical files.
- [X] T014 [US1] Integrate schema-5 `preflight` and its required `--linear-reference-root`/`--geometric-reference-root` flags in `scripts/analyze_tinystories_optimizer_ownership.py`; atomically publish configs, audits, per-arm traces, and new reference-selection records only after all checks pass, rejecting incompatible roots while allowing verification of an identical prepared identity.
- [X] T015 [US1] Add bounded expected-schedule export in `src/evaluation/optimizer_ownership.py` for positions 0–348,528 with applied-update mapping, warmup/horizon, dependency/source identity, and full-array hash; use the inherited cosine formula, require identical new-run schedules, and represent terminal position 348,528 with no subsequent applied update.
- [X] T016 [US1] Run the US1 protocol/reference acceptance cases and historical signature checks in `tests/test_s1_warmup_campaign.py` and `tests/test_s1_warmup_reporting.py`; record commands/results and fixture-versus-real-input limitations in `specs/016-tinystories-s1-warmup/verification.md` without declaring real input audits or execution readiness.

**Checkpoint**: The CPU-only preparation MVP works against fixtures and preserves historical behavior. Real input/reference certification occurs at T051.

## Phase 4: User Story 2 — Verify and Complete the Longer Schedule Safely (Priority: P1)

**Goal**: Make existing S1 training, continuation, readiness, and two-run admission honor the new protocol with durable execution evidence.
**Independent test**: Compare full schedules and actual short interrupted/uninterrupted executions for both grids, reject invalid restores before mutation, inject partial-update failures, and exercise queue/CUDA failures using fixtures. Real GPU and full-budget acceptance remain T053–T054.
**Coverage**: FR-002–004, FR-007–011, FR-013; SC-002–004 and operational portions of SC-008.

### Verification for User Story 2

- [X] T017 [P] [US2] Add both-grid schedule/application cases in `tests/test_s1_warmup_campaign.py` covering every expected position, actual pre-optimizer LR around updates 64/65 and 256/257, zero initial LR, peak first applied at update 257, terminal stored LR zero, final applied LR, one scheduler advance per committed update, and continuous real-horizon epoch transitions.
- [X] T018 [US2] Add both-grid all-width and resume cases in `tests/test_s1_warmup_campaign.py` covering shared full-tensor AdamW inactive-tail momentum/decay, global L2 cap 1, exact action/batch suffixes, and complete model/optimizer/scheduler/RNG/cursor/accounting agreement before/at/after warmup 256 and epoch boundaries 87,132/174,264/261,396; reuse inherited numerical tolerances and label state-seeded probes as synthetic-history diagnostics.
- [X] T019 [US2] Add restore/failure cases in `tests/test_s1_warmup_campaign.py` for cross-run/grid, original warmup-64, model-only, malformed/non-finite bundles, rejection before live mutation, optimizer/scheduler/accounting failure durability, and terminal-only output recovery at step 348,528 with zero extra updates.
- [X] T020 [P] [US2] Add lifecycle/CLI acceptance cases in `tests/test_s1_warmup_queue.py` for snapshot and gate bindings, stale/failed/skipped gates, absent CUDA/non-bf16 execution, exact two-arm admission, occupied identities, duplicate workers, uncertain submissions, delayed accounting, user-wide limits including unrelated jobs, matching job/worker success, replay resource reconciliation, and restart/terminal-only recovery.

### Implementation for User Story 2

- [X] T021 [US2] Make compact accounting in `src/utils/metrics.py` resolve schema-5 support by arm, retain measured pre-optimizer applied LR and committed-step provenance, and preserve bounded streaming action/exposure/resource accounting without changing legacy metrics behavior.
- [X] T022 [US2] Update `src/training/run.py` to carry arm/grid/protocol identities through summaries, counts, exposure, checkpoints, and four-width ordinary-validation endpoints; preserve validation every 64 updates plus terminal evaluation, target-token-weighted loss, `exp(loss)` perplexity, and sealed controller/final roles.
- [X] T023 [US2] Audit and update literal S1-name checks in `src/training/steps.py` to use declared slicing/shared-history semantics for the new arms; retain the current scheduler, capture applied LR before optimizer execution, and preserve optimizer-then-scheduler complete-update ordering and finite all-width behavior.
- [X] T024 [US2] Extend whole-bundle identity/restore validation in `src/training/checkpointing.py` to bind the new grid/protocol/warmup/horizon through existing contracts; validate complete state before installation, preserve poisoned-state checkpoint rejection after partial failure, and use terminal-only recovery in `src/training/run.py` without extending the budget.
- [X] T025 [US2] Implement immutable executable-source snapshot creation and CPU gate generation in `scripts/preflight_tinystories_s1_warmup.py`; test the snapshot itself and bind recipe/config-set/manifest/contracts, reference/control/trace audits, commands/logs, and pass/fail/skip results, invalidating changed bindings rather than relabeling old evidence.
- [X] T026 [US2] Implement the `gpu --campaign-root` diagnostic in `scripts/preflight_tinystories_s1_warmup.py` for both grids at d64/l4/h4, batch 64, context 128, actual CUDA bf16, and all widths; retain the real 348,528 horizon for boundary/resume/failure probes, bind passed CPU evidence, record allocation/device/precision coverage, and prohibit mandatory skips from passing the gate.
- [X] T027 [US2] Implement `prepare` in `scripts/run_tinystories_s1_warmup.py` to verify the unique reservation, source/config/reference/CPU-gate bindings and publish an exact two-arm launch plan; reuse pure lock/reservation helpers while omitting the historical nine-arm policy and standalone barrier.
- [X] T028 [US2] Implement `queue [--once]` in `scripts/run_tinystories_s1_warmup.py` with live user-wide two-running/four-submitted ceilings or stricter Slurm limits, one process/GPU per job, and excluded nodes `gpu-[05,50,51,54]`; account for unrelated jobs and active reservations so newly pending work cannot later exceed the running ceiling.
- [X] T029 [US2] Add durable submission intents, monotonic attempt IDs/UUIDs, scheduler queue/accounting reconciliation, queue/per-run locks, and validated own-checkpoint continuation in `scripts/run_tinystories_s1_warmup.py`; keep ambiguous submissions pending and block occupied runs lacking valid continuation instead of duplicating or silently restarting them.
- [X] T030 [US2] Implement the sbatch-only `worker` in `scripts/run_tinystories_s1_warmup.py` using frozen `scripts/train_cuda_required.py`; revalidate intent/gates/configs/continuation before training, require usable CUDA and actual bf16/device evidence, and fail without CPU fallback or alternative arms.
- [X] T031 [US2] Integrate `ResourceAttemptLedger`, matching worker exit/Slurm success, source/config/job identities, full terminal/evaluation/trace checks, and independent stage statuses in `scripts/run_tinystories_s1_warmup.py`; reconcile committed progress separately from failed/replayed costs, disclose missing resource measurements, and retain pending completion while accounting evidence is delayed.
- [X] T032 [US2] Add readiness-diagnostic sbatch submission support to `scripts/preflight_tinystories_s1_warmup.py`, reusing the launcher's pure live-limit/reservation policy; execute the frozen GPU entry point with one GPU/process and excluded nodes, persist intent/job evidence, and keep diagnostic artifacts and costs separate from production.
- [X] T033 [US2] Integrate own-run terminal-only recovery into `scripts/run_tinystories_s1_warmup.py`, allowing missing terminal evaluation/summary outputs to be regenerated from a validated full-horizon checkpoint without increasing committed steps or weakening GPU/job/worker provenance requirements.
- [X] T034 [US2] Run schedule, all-width, restore/failure, accounting, and queue acceptance tests in `tests/test_s1_warmup_campaign.py` and `tests/test_s1_warmup_queue.py`; record inherited numerical tolerances, full-horizon versus state-seeded coverage, and outstanding GPU checks in `specs/016-tinystories-s1-warmup/verification.md`.
- [X] T035 [US2] Update the exact snapshot, CPU/GPU diagnostic, prepare/queue/worker, continuation, pending-accounting, and terminal-recovery procedures in `docs/tinystories-s1-warmup-experiment.md`; make their authorization prerequisites and evidence locations concrete before execution.

**Checkpoint**: CPU runtime and admission checks pass. This does not establish GPU readiness or two-run terminal completion; those need authorized real execution after reporting fixtures pass.

## Phase 5: User Story 3 — Assess the Intervention from Saved Evidence (Priority: P2)

**Goal**: Independently preserve eight new endpoints, then publish a strict 24-endpoint comparison with eight differences, early observations, sixteen figure files, and descriptive findings.
**Independent test**: Saved-result fixtures produce exactly 12 endpoints per grid, correct paired differences/standalone gaps, inspectable early data, and exact plotted series. Invalid inputs must prevent complete status while preserving valid new-only evidence.
**Coverage**: FR-011–019; SC-005–008.

### Verification for User Story 3

- [ ] T036 [P] [US3] Extend `tests/test_s1_warmup_reporting.py` with new-freeze/comparison fixtures for 8/24 endpoint cardinalities, grid-qualified uniqueness, CSV/JSON parity, required provenance/budget/resource fields, eight signed differences and both standalone gaps, target-weighted loss/perplexity, exact parameter coordinates/series, and sixteen PNG/PDF figure outputs.
- [ ] T037 [P] [US3] Add analyzer and launcher report integration cases in `tests/test_s1_warmup_queue.py` for `freeze`, `report-s1-warmup`, and report-only recovery; require nonzero complete-report failure for invalid/missing references or required early evidence, preserve valid new-only outputs, and prove reporting invokes no training/submission or historical mutation.
- [ ] T038 [US3] Add early-evidence and rejection cases in `tests/test_s1_warmup_reporting.py` for committed replay resolution, ambiguous duplicate rows, missing observations, recorded versus reconstructed LR, no invented step-zero loss, end steps below 1,024, wrong-device/non-finite/nonterminal/stale/duplicate/budget-inconsistent/evaluation-incompatible inputs, and sources changing before publication.

### Implementation for User Story 3

- [ ] T039 [US3] Extend terminal inspection, endpoint extraction, and `freeze_campaign` in `src/evaluation/optimizer_ownership.py` for the two schema-5 arms and eight new full-budget ordinary-validation endpoints; bind checkpoint/evaluation/execution identities and keep new-only publication independent of subsequent historical comparison failures.
- [ ] T040 [US3] Implement strict comparison assembly in `src/evaluation/optimizer_ownership.py` by revalidating the selected ten historical terminals and pairing by grid/campaign/run/physical width; emit exactly 24 unique endpoints with all fields in `specs/016-tinystories-s1-warmup/contracts/reporting.md`, retaining separate same-size standalones and rejecting best/early/trailing estimates.
- [ ] T041 [US3] Implement matching `endpoints.csv`/`endpoints.json` and eight-row `paired_deltas.csv`/`paired_deltas.json` exports in `src/evaluation/optimizer_ownership.py`; use new-minus-original loss/perplexity and both original/new S1-minus-own-grid-standalone gaps, preserving lossless logical parity and explicit endpoint links.
- [ ] T042 [US3] Implement provenance-aware early scalar extraction in `src/evaluation/optimizer_ownership.py` for raw recorded training loss/applied LR and available per-width validation loss over a common absolute-update window starting at 0 and ending at least at 1,024; resolve committed attempts, export `early_metrics.csv`/`early_metrics.json`, and allow only explicitly labeled schedule-derived LR reconstruction from validated controls.
- [ ] T043 [US3] Implement Linear/Geometric endpoint figures in `src/evaluation/optimizer_ownership.py`: loss and perplexity against exact active non-embedding counts, each in PNG/PDF, with four disconnected standalone markers and two connected four-point S1 curves labeled 64/256-update warmup; annotate dataset, seed, ordinary validation, terminal selection, count convention, and unequal epoch budgets without jitter/deduplication/error bars.
- [ ] T044 [US3] Implement both grids' early LR/loss PNG/PDF figures in `src/evaluation/optimizer_ownership.py` with matching x limits, markers at updates 64 and 256, separate training/validation panels, physical-width labels, raw data by default, and explicit cadence/gaps/smoothing/reconstruction annotations; never fabricate losses or join unexplained missing segments as measured data.
- [ ] T045 [US3] Implement atomic comparison publication, `comparison_report.json`, and `findings.md` in `src/evaluation/optimizer_ownership.py`; recheck source hashes, record independent endpoint/early/overall statuses and produced file hashes, report numerical direction/magnitude and standalone gaps at all eight widths, and disclose descriptive seed-42/resource limitations without requiring improvement or inferring causation/significance/equal compute.
- [ ] T046 [US3] Add `report-s1-warmup --manifest --linear-reference-root --geometric-reference-root --output-dir [--early-end-step 1024]` and schema-5 freeze behavior in `scripts/analyze_tinystories_optimizer_ownership.py`; preserve unrelated command behavior, reject early-end-step below 1,024, and expose specific incomplete/invalid reasons with nonzero complete-report status.
- [ ] T047 [US3] Implement `report --campaign-root` in `scripts/run_tinystories_s1_warmup.py` using prepared reference mappings and the frozen source; recover new freeze/report outputs before strict comparison, retain valid new artifacts on missing historical/early evidence, and never submit or train from this command.
- [ ] T048 [US3] Run saved-artifact reporting and report-only integration checks in `tests/test_s1_warmup_reporting.py` and `tests/test_s1_warmup_queue.py`, inspect fixture figures/coordinates and scalar exports, and record results in `specs/016-tinystories-s1-warmup/verification.md`; these checks must pass before production admission.

**Checkpoint**: Reporting is proven with fixtures before expensive execution. Real endpoint/figure/findings completion remains pending until T055.

## Phase 6: Polish, Cross-Cutting Verification, and Authorized Execution

**Purpose**: Validate compatibility, bind readiness to final source, then complete the scientific deliverables only within subsequent authorization. T051–T052 depend on real-input access; missing references are a reported blocker, not permission to replace or retrain them.

- [ ] T049 Run the three focused `tests/test_s1_warmup_*.py` suites and relevant existing `tests/test_config.py`, `tests/test_optimizer_ownership.py`, `tests/test_optimizer_ownership_campaign.py`, `tests/test_optimizer_ownership_resume.py`, `tests/test_optimizer_ownership_reporting.py`, `tests/test_optimizer_ownership_corrections.py`, `tests/test_matformer_widths_campaign.py`, `tests/test_matformer_widths_reporting.py`, `tests/test_matformer_widths_queue.py`, `tests/test_matformer_resource_reconciliation.py`, and `tests/test_metrics_compact_accounting.py`; record exact selection/results and schema-1–4 signature preservation in `specs/016-tinystories-s1-warmup/verification.md` and fix failures before snapshot/gating.
- [ ] T050 Validate implemented CLI help and local temporary-fixture commands against `specs/016-tinystories-s1-warmup/quickstart.md` and `docs/tinystories-s1-warmup-experiment.md`; replace proposed-command descriptions with verified behavior, document incomplete outcomes accurately, and ensure no example bypasses authorization, live limits, or CUDA-required entry.
- [ ] T051 Execute real-input schema-5 preflight using `scripts/analyze_tinystories_optimizer_ownership.py` and `configs/controlled_exps/tinystories_instruct_s1_warmup.yaml` with the corpus/tokenizer/reference paths in `specs/016-tinystories-s1-warmup/quickstart.md`; recheck/reserve the fresh root only after successful controls/models/data/role/ten-terminal audits, publish under that root, and record all audit hashes and historical-source preservation in `specs/016-tinystories-s1-warmup/verification.md`.
- [ ] T052 Run snapshot-bound `cpu` through `scripts/preflight_tinystories_s1_warmup.py` and `prepare` through the frozen `scripts/run_tinystories_s1_warmup.py`; validate final source/config/reference bindings and record gate/plan hashes in `specs/016-tinystories-s1-warmup/verification.md`, leaving GPU readiness pending and rerunning affected gates if bindings change.
- [ ] T053 After subsequent GPU-diagnostic authorization, submit the frozen `scripts/preflight_tinystories_s1_warmup.py` diagnostic through its checked sbatch path; require both real-shape bf16 grids, all widths, warmup/epoch/terminal boundary and continuation/failure coverage, zero mandatory GPU skips, matching CPU/source/config bindings and job/device evidence, and record readiness plus measured diagnostic costs in `specs/016-tinystories-s1-warmup/verification.md`.
- [ ] T054 After subsequent production authorization and passed gates, run the frozen `scripts/run_tinystories_s1_warmup.py queue` for exactly the two fresh S1 arms, continuing only each run's own durable state; reconcile completion at four epochs/348,528 updates/2,855,141,376 tokens per run, exact counterpart traces, finite four-width terminal validation, attempt resources, and successful matching GPU/job/worker evidence in `specs/016-tinystories-s1-warmup/verification.md`.
- [ ] T055 Run the frozen `scripts/run_tinystories_s1_warmup.py report` on real saved artifacts, revalidate references, and inspect eight new endpoints, 24 comparison endpoints, eight paired deltas/standalone gaps, raw early data and all sixteen figure files; record report hashes and all eight observed effects in `docs/tinystories-s1-warmup-experiment.md`, preserving new-only results and marking comparison/early completion outstanding if required evidence is missing.
- [ ] T056 Reconcile SC-001–008 against saved evidence in `specs/016-tinystories-s1-warmup/verification.md`, `specs/016-tinystories-s1-warmup/plan.md`, `specs/016-tinystories-s1-warmup/quickstart.md`, `docs/tinystories-s1-warmup-experiment.md`, and `specs/016-tinystories-s1-warmup/tasks.md`; check off only completed work, distinguish all readiness/result stages, and confirm zero historical retraining/writes, extra production arms, or final-holdout evaluation.

## Dependencies and Execution Order

```text
Setup T001–T002
  -> historical baseline T003–T004
  -> US1 T005–T016 (preparation MVP)
  -> US2 T017–T035 (runtime/readiness/admission software)
  -> US3 T036–T048 (saved-artifact report software)
  -> T049–T050 (regressions and executable runbook)
  -> T051 (real inputs/references and fresh root)
  -> T052 (final source snapshot, CPU gate, prepared plan)
  -> T053 (authorized GPU gate)
  -> T054 (authorized two-run production)
  -> T055 (real comparison)
  -> T056 (evidence-backed acceptance)
```

### Story and task dependencies

- US1 is independently testable after the compatibility foundation. T005, T006, and T007 can proceed together; T008–T015 integrate in order, and T016 is the checkpoint. T010 must preserve T004's golden identities.
- US2 uses US1's resolved protocols and identities. T017–T019 share one test file and run sequentially; T020 is independent test authoring. Runtime integration T021–T024 precedes snapshot/probe implementation T025–T026. Queue/worker work T027–T031 precedes diagnostic submission reuse T032 and recovery integration T033. T034 validates the increment.
- US3 uses US1 reference selection and US2 execution/terminal evidence contracts. Its T036 and T037 fixture work can proceed together; T038 extends T036. T039–T045 share the campaign module and run sequentially, followed by analyzer/launcher wiring and T048 verification. Saved fixtures make US3 testable without live production.
- Complete US3 software/fixtures before production, even though US2 owns the training story. T053–T055 finish acceptance across both stories; successful software tests alone cannot satisfy SC-004–007.
- Missing or invalid initial references block T051 preparation. A later loss of references cannot erase valid new terminals; T055 preserves new-only outputs and records incomplete comparison. Do not reopen cancelled arms or train replacements.
- Any source/config/reference-binding change after T052 requires the relevant fresh CPU/GPU evidence before admission. Do not modify a tested executable snapshot in place.

### Parallel opportunities and examples

Examples describe independent work; this task-generation command does not spawn workers or launch jobs.

| Scope | Work that can proceed together | Prerequisite |
| --- | --- | --- |
| Setup | T001 evidence record and T002 runbook skeleton | Design documents loaded |
| US1 | T005 protocol tests, T006 reference tests, T007 fixed recipe | T004 baseline passed |
| US2 | T017–T019 runtime tests in sequence alongside T020 queue tests | US1 checkpoint passed |
| US3 | T036 report fixtures alongside T037 CLI/report integration fixtures | US2 checkpoint passed |

The shared `src/evaluation/optimizer_ownership.py`, launcher, and campaign test files serialize their edits. Do not mark tasks on those same files as parallel simply because they serve different stories. Authorized GPU jobs also remain subject to live user-wide capacity, regardless of these software examples.

## Implementation Strategy

1. **MVP first**: Complete T001–T016 and validate the two fixed definitions, exact own-counterpart streams, selected-reference contract, strict difference audit, and unchanged historical identities. This delivers controlled CPU preparation without training.
2. **Add safe execution**: Complete US2 and validate schedules, applied LR, complete-state continuation, failure durability, snapshot/gates, and two-only admission on fixtures. Preserve the existing scheduler and training primitives.
3. **Prove analysis before production**: Complete US3 using saved-result fixtures, including negative cases and the actual requested figure structure. Keep new-only, endpoint comparison, and early/report completion independent.
4. **Bind real readiness**: Run regressions and real audits, freeze final executable sources, and obtain evidence from CPU and subsequently authorized real-shape GPU diagnostics. Short/state-seeded diagnostics do not represent full training.
5. **Deliver the experiment**: With production authorization, complete exactly two runs, compare immutable saved results, and publish all endpoint/early evidence and descriptive findings. Completion depends on valid artifacts, not whether longer warmup improves results.

## Requirement and Acceptance Coverage

| Requirement group | Tasks | Acceptance evidence |
| --- | --- | --- |
| FR-001–006, FR-012 | T003–T016, T051 | Two fixed protocols, actual counts/data/roles, closed control differences, own-counterpart traces, ten selected terminals; SC-001/008 |
| FR-007–009 | T015, T017–T026, T034, T049, T052–T053 | Full arrays and applied LR, exact resume streams, inherited numerical tolerance, invalid-state/failure durability, bound CPU/GPU gates; SC-002/003 |
| FR-010–011 | T020–T035, T051–T054 | Two-only sbatch admission, actual CUDA bf16, complete source/job/worker evidence, durable attempts and reconciled resources; SC-004/008 |
| FR-013–015 | T022, T036–T041, T045–T048, T054–T055 | Two terminals/eight new endpoints, strict 24-row tables/eight deltas, ordinary-validation provenance; SC-004/005 |
| FR-016–018 | T036–T038, T042–T048, T055 | Sixteen figure files, inspectable early scalars, eight-width numerical findings and limitations; SC-006/007 |
| FR-019 and historical compatibility | T001–T004, T016, T034–T035, T048–T056 | Separate evidence-backed statuses, explicit authorization stages, preserved legacy contracts and historical artifacts; SC-008 |

**Task inventory**: 56 tasks: setup 2, foundational 2, US1 12, US2 19, US3 13, final cross-cutting 8. Eight tasks carry `[P]`; prerequisites above define the safe parallel groups. Tasks are checked only when their stated work and evidence are complete.
