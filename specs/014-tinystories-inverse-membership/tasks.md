# Tasks: TinyStories Inverse-Membership Sampling Comparison

**Input**: spec.md, plan.md, research.md, data-model.md, contracts/ and quickstart.md.
**Tests**: Explicitly required by the experiment protocol. CPU/GPU gates precede production.

## Phase 1: Setup

- [x] T001 Verify existing environment, ignore rules and fresh campaign paths; record source/reference inspection in specs/014-tinystories-inverse-membership/verification.md (FR-001/012/019).

## Phase 2: Foundation

- [x] T002 Define schema-3 arms and explicit sampling metadata in src/evaluation/optimizer_ownership.py, retaining strict schema-1/2 controls and historical hashes (FR-001/005, EX-001).

## Phase 3: US1 — Validate the sampling comparison (P1)

Independent test: exact five-arm recipe/model counts, weighted mapping and deterministic traces; reject changed controls before training.

- [x] T003 [US1] Add five-arm expansion/rejection, real count and categorical-boundary/seeded-trace cases in tests/test_inverse_membership_sampling.py (FR-002–005/009, EX-001–005).
- [x] T004 [US1] Add configs/controlled_exps/tinystories_instruct_inverse_membership.yaml and fixed-global C3 eligibility in src/utils/config.py (FR-002/003/009).
- [x] T005 [US1] Integrate strict materialized sampling metadata, expected weighted traces and preflight validation in src/evaluation/optimizer_ownership.py (FR-004/005/009, EX-002–005).

## Phase 4: US2 — Preserve optimizer behavior and continuation (P1)

Independent test: all five real model paths, wider/narrower updates, exact action/batch resume across epoch boundaries, pre-mutation mismatch and unsafe-save rejection.

- [x] T006 [US2] Add five-arm real-update, resume, categorical-state corruption and C3 failure cases in tests/test_inverse_membership_sampling.py (FR-006–010).
- [x] T007 [US2] Integrate policy-bound compact state and sampled probability through selection/commit in src/training/steps.py and src/training/checkpointing.py (FR-004–008/014).
- [x] T008 [US2] Add weighted ownership checkpoint replay and compatibility handling in src/training/checkpointing.py; run five-arm semantics/resume and historical sampling/ownership suites (FR-005–010).

## Phase 5: US4 — Deliver validated comparisons (P2; tooling precedes production)

Independent test: complete 20/44-point fixtures with wrong/missing provenance rejection, styles, full-range progress and paired deltas.

- [x] T009 [US4] Add new/comparison report, source mismatch, count and progress cases in tests/test_inverse_membership_reporting.py (FR-015–018, EX-006/007).
- [x] T010 [US4] Add strict 20/44 endpoint reporting, policy labels, epoch-only cross-policy pairing, deltas, exposures/resources/clipping and five-arm figures in src/evaluation/optimizer_ownership.py and scripts/analyze_tinystories_optimizer_ownership.py (FR-014–018).

## Phase 6: US3 — Execute the authorized five-run campaign (P1; all earlier verification gates required)

Independent test: queue/intent/writer safety fixtures, five real-control GPU diagnostics and exact terminal budget checks.

- [x] T011 [US3] Add limit, duplicate-intent, source/gate and completion handling checks in tests/test_inverse_membership_queue.py (FR-011–013/019).
- [x] T012 [US3] Implement isolated prepare/queue/worker/monitor flow in scripts/run_tinystories_inverse_membership.py with fresh root, hashes, stricter live limits, one writer and reconciliation (FR-011–013/019).
- [x] T013 [P] [US3] Add real-shape bf16 five-arm action/ownership/finite/save-restore/timing diagnostics in scripts/preflight_tinystories_inverse_membership.py (FR-010).
- [x] T014 [US3] Run focused CPU/compatibility suites, immutable-input preflight and historical evidence audit; freeze source/configs and record gate evidence beneath campaign-root/diagnostics (FR-009/010/012, EX-002–006).
- [x] T015 [US3] Submit sbatch diagnostics for all five arms, inspect complete results and bind passed GPU gate to source hashes under campaign-root/diagnostics (FR-010/011/019).
- [ ] T016 [US3] Submit and monitor five full-budget runs, queue fifth on capacity, reconcile attempts/resources and validate all terminals under campaign-root/runs and launchers (FR-008/011–015, EX-004).
- [ ] T017 [US3] Freeze terminals and generate 20/44 endpoints, five PNG/PDF figure stems and paired interpretation under campaign-root/reports; identify unavailable references explicitly (FR-014–019, EX-006/007).

## Phase 7: Polish and reconciliation

- [ ] T018 Finalize docs/tinystories-inverse-membership-experiment.md and specs/014-tinystories-inverse-membership/verification.md with actual commands, analysis, tests, job IDs, outcomes and requirement reconciliation (FR-019, SC-001–007).

## Dependencies and execution strategy

T001 → T002 → T003/T004/T005 → T006/T007/T008 → T009/T010 →
T011/T012/T013 → T014 → T015 → T016 → T017 → T018.
Tests are written before the corresponding implementation. Shared-file changes
are sequential. US4 tooling is implemented before production despite its P2
priority so expensive runs have a verified reporting path. US3 execution depends
on all CPU/GPU gates; US4 final reporting depends on terminal US3 outputs.
MVP is US1's reproducible five-arm preflight; the authorized objective includes
all phases, not just MVP. Mark tasks complete only after their evidence exists.

Parallel opportunities: US1 independent external data reads can accompany local
fixture inspection; US2 corruption-case design can accompany model-case design;
US4 plotting fixtures and provenance fixture design are independent; US3 T013
can be developed alongside T012 because it owns a distinct script. These examples
do not override task prerequisites or permit concurrent edits to shared files.
