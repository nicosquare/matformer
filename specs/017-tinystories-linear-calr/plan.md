# Implementation Plan: TinyStories Linear S1/S2 CaLR Comparison

**Branch**: `017-tinystories-linear-calr` | **Date**: 2026-10-01 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification and `notes/tinystories_calr_speckit_prompt_2026-10-01.md`.
**Status**: Planning complete; implementation, readiness and execution remain pending.

## Summary

Extend the existing optimizer-ownership experiment with schema 6 and exactly four linear arms: S1/S2 × uniform polynomial/CaLR. Keep peak LR 0.008, warmup 64, initialization, data/actions and nonschedule controls matched to original S1/S2. Use one global nominal polynomial clock and a temporary selected-width LR on every group of the stepped optimizer. Restore nominal rates before advancing the clock, preserving S2 synchronization at every durable boundary and all historical scalar schedule paths.

Reuse the existing trainer, failure boundary, checkpoints, accounting, ordinary validation and Slurm admission helpers. Produce 16 new endpoints, validate 20 historical endpoints read-only, and publish 36 rows, 48 required pairs, four interactions and measured figures. Legacy supplemental references require their actual saved schema's provenance adapter; missing proof leaves comparison incomplete.

## Technical Context

**Language/Version**: Python 3.12.13; `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python`.
**Primary Dependencies**: Installed PyTorch 2.11.0+cu128, Transformers 5.8.0, NumPy 2.4.3, PyYAML 6.0.3, Matplotlib 3.10.9, pytest 9.0.3; no new packages.
**Storage**: Existing packed mmap corpus; YAML configuration; JSON/JSONL/CSV evidence; PyTorch resumable checkpoints; PNG/PDF figures.
**Testing**: Focused pytest CPU schedules, actual AdamW updates, continuation, reporting and queue fixtures; later authorized real-shape CUDA BF16 diagnostics for all four arms; historical S1/S2/C4 regressions.
**Target Platform**: Linux/Slurm, one process/GPU per run; CPU preparation and reporting.
**Project Type**: Controlled research experiment extending the current pipeline.
**Experiment Scope**: Four seed-42 linear slicing arms; shared versus per-granularity AdamW histories; exponent 1 versus complexity-derived exponents; no correction.
**Datasets/Data Assumptions**: Inherit original audited TinyStories-Instruct tokenizer, packed corpus and manifests; 5,576,448 designated sequences per epoch, fixed excluded tail 43; ordinary validation only, controller reserved and final holdout sealed. Revalidate identities during preparation.
**Configuration Inputs**: Schema-6 fixed recipe; linear FFN prefixes 64/128/192/256; LR .008, warmup 64, T=348528; complexity definition/counts and exponent bounds .5/2; inherited controls/data hashes and explicit counterpart mappings.
**Experiment Outputs**: Config difference audits, analytic schedules, measured LR/action/data evidence, scalar trajectories, exposure/resources, four terminal checkpoints, 36 endpoints, 48 pairs, four interactions and PNG/PDF comparisons.
**Reproducibility Notes**: Fresh normal constructor at seed 42, initializer .02; names must not change dedicated initialization/action/data streams; source/config hashes, own-arm continuation, immutable reference provenance. Inherit CPU continuation tensor tolerances rtol=1e-6/atol=1e-7; preserve stricter existing assertions where applicable and document device-specific checks.
**Performance Goals**: Measure throughput, optimizer-state memory, CUDA peaks and attempt costs; no promised quality improvement or speedup; stream accounting with bounded memory.
**Constraints**: GPU diagnostics/production require later explicit execution authorization; cscc-gpu-p / cscc-gpu-qos; exclude gpu-[05,50,51,54]; user-wide at most two running GPU jobs/four submitted or stricter live limits; no historical writes/retraining.
**Scale/Scope**: d64/l4/h4, FFN 256, context 128, vocab 2048, batch 64; four × 348528 updates × 8192 tokens = 1394112 updates and 11420565504 packed tokens. Standalone references have 87132 updates each.

Environment versions and proposed-root absence were checked on 2026-10-01. Reference discovery/config inspection is distinct from full terminal validation. No GPU work, full checkpoint audit, corpus reaudit or output reservation is part of this invocation. Design unknowns are resolved in [research.md](research.md); acceptance dependencies remain explicit.

## Constitution Check

| Principle | Before research | After design | Basis |
| --- | --- | --- | --- |
| I. Research code first | PASS | PASS | Four visible arms and existing trainer. |
| II. Simplicity/local reasoning | PASS | PASS | Pure formula plus narrow temporary LR transaction. |
| III. Explicit experiment flow | PASS | PASS | Fixed recipe and focused launcher; no registry or second framework. |
| IV. Minimal abstraction/validation | PASS | PASS | Validation prevents confounds, poisoned saves and duplicate jobs. |
| V. Configuration/reproducibility | PASS | PASS | New-only versioned contract; unchanged historical serialization and streams. |
| VI. Useful outputs/logging | PASS | PASS | Structured measured evidence, terminal tables and figures. |
| Shallow organization | PASS | PASS | Current modules plus focused scripts/config/tests. |

No gate failures or exceptions before research or after design. Full optimizer copies per update, generic schedulers/campaign frameworks and speculative registries are unnecessary.

## Project Structure

### Documentation (this feature)

```text
specs/017-tinystories-linear-calr/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── contracts/
    ├── schedule-and-update.md
    ├── campaign-and-lifecycle.md
    └── reporting.md
```

`tasks.md` is generated by the subsequent tasks command. Implementation will add verification and an experiment report; this invocation does not create them.

### Source Code (repository root)

| Path | Responsibility |
| --- | --- |
| `configs/controlled_exps/tinystories_instruct_linear_calr.yaml` | Fixed schema-6 four-arm recipe, controls and reference mappings. |
| `src/evaluation/optimizer_ownership.py` | Schema expansion, factor/counterpart audits, complexity tables, reference selection, terminal freeze and comparison. |
| `src/utils/config.py` | Explicit new schedule resolution, semantic S1/S2 eligibility, immutable scientific/optimizer contracts. |
| `src/training/schedules.py` | Pure validated polynomial/CaLR formula and new-only metric fields. |
| `src/training/optimizer_state.py` | Reuse GlobalSchedulerClock with exact nominal gamma=1 lambda; preserve collection state layout and synchronized boundary rates. |
| `src/training/steps.py` | Same nominal clock construction for S1/S2; temporary all-group effective LR, actual-rate capture, finally restoration and existing commit boundary. |
| `src/training/checkpointing.py` | New-contract nominal reconstruction and applied-evidence checks before whole-bundle install; preserve old branches. |
| `src/training/run.py` | Reconcile new contract, scalar/evidence summaries and terminal recovery. |
| `src/utils/reproducibility.py`, `src/utils/metrics.py` | New-only hash fields and compact evidence; unchanged old signatures, deterministic streams and bounded accounting. |
| `scripts/analyze_tinystories_optimizer_ownership.py` | New campaign preflight/freeze and `report-linear-calr` operation. |
| `scripts/run_tinystories_linear_calr.py` | Prepare/queue/worker/report with four-only admission and durable attempts. |
| `scripts/preflight_tinystories_linear_calr.py` | Snapshot-bound CPU and later CUDA gates, full-horizon probes. |
| `scripts/train_cuda_required.py` | Existing required CUDA/BF16 entry point. |
| `scripts/plot_tinystories_standalones.py`, `scripts/plot_tinystories_s1_peak_lr.py`, `scripts/plot_tinystories_s2_peak_lr.py` | Reuse read-only terminal/legacy validation and recorded curves where appropriate. |
| `tests/test_linear_calr_campaign.py`, `tests/test_linear_calr_schedule.py` | Counts, factor audits, analytic and actual updates, streams and legacy identity. |
| `tests/test_linear_calr_resume.py`, `tests/test_linear_calr_reporting.py`, `tests/test_linear_calr_queue.py` | Restore/failure boundaries, provenance/counts/plots, gate/admission/recovery. |

**Structure Decision**: Keep scientific policy in the campaign module and runtime mechanics in current training modules. Use an optional immutable schedule contract only for the new treatment. Preserve original scalar clocks and collection synchronization outside the narrow optimizer call. A new width-clock class and preparation-width state would add restore complexity without scientific benefit.

## Phase 0: Research

[research.md](research.md) records repository/source inspection, two read-only research agents, installed versions, paper context, schedule integration and legacy reference dependencies. All design questions have decisions and alternatives. Availability and terminal proof must be rechecked during preparation rather than inferred from existing tables.

## Phase 1: Design and Contracts

Campaign ID `tinystories-linear-s1-s2-calr-v1`; arm-qualified run IDs `<campaign_id>-<arm_id>-s42`. Proposed root `/nfs-stor/ivo.navarrete/results/elasticnn/tinystories-linear-s1-s2-calr-v1` was absent during planning; no reservation was made.

[data-model.md](data-model.md) defines identities, evidence and transitions. Contracts specify [schedule/update](contracts/schedule-and-update.md), [campaign/lifecycle](contracts/campaign-and-lifecycle.md) and [reporting](contracts/reporting.md). [quickstart.md](quickstart.md) documents planned commands and authorization boundaries. Update root AGENTS.md's plan reference using explicit marker replacement: no update-agent-context script exists in this checkout. Post-design gates pass.

## Phase 2: Implementation and Verification Sequence

1. **US1 / FR-001–004,007–009**: Capture historical signatures; add four schema-6 arms and separate schedule/ownership/counterpart closed difference audits. Validate actual-model complexity and inherited corpus/control identities. Compare fresh weights and complete deterministic action/data sequences.
2. **US2 / FR-003–006,011**: Add formula/nominal lambda and narrow temporary-rate path. Test all widths at 0/63/64/65, representative decay and T−1/T; full-horizon analytic rates; all-group AdamW reference steps with populated moments; zero-rate first-step semantics; unselected S2 histories; slicing tail behavior.
3. **US2 / FR-011, EX-004**: Test uninterrupted versus continuation before/at/after warmup, decay and every epoch boundary; malformed/cross-arm contracts rejected before mutation; optimizer/scheduler/accounting failures poison live state; zero-update terminal output recovery. State-seeded boundary probes must not claim completed training.
4. **US3 / FR-013–019**: Build provenance-complete report fixtures before production. Validate native and legacy reference adapters; 36 endpoints/48 pairs/four interactions; primary/supplemental seven-series plots and measured-only progress/LR. Missing evidence must yield specific incomplete states.
5. **US2 / FR-010,012 / EX-005**: Snapshot/config-bound CPU gate, four-arm launcher and attempt ledger. Test stale/failed/missing gates, CPU fallback rejection, occupied roots, duplicate/uncertain submission, user-wide queue limits and job/worker reconciliation. No production submission without execution authorization.
6. **Compatibility**: Run focused tests and relevant config, optimizer/history, checkpoint, metrics, reproducibility, reporting and queue regressions, including historical S1/S2/C4 and schemas 1–5. Record commands, source/config hashes and pass/fail/skips; GPU skips are not readiness.
7. **Later authorized CUDA readiness**: Through sbatch, diagnose every arm at real shape/batch/context with actual BF16, all widths, nonzero moments, schedule/continuation boundaries and resource evidence. Bind passing GPU and CPU gates to identical tested source/configs.
8. **Later authorized production and acceptance**: Admit at most two running/four submitted; fresh-start exactly four arms, own-state continuations only. Finish exact budgets, terminal ordinary validation and evidence reconciliation. Read-only validate eight historical runs; publish endpoint/pair/interaction tables, PNG/PDF figures and descriptive findings. Missing references preserve new results but prevent comparison acceptance.

Planning completion establishes neither GPU readiness nor SC-004 terminal completion. Overall delivery requires all SC-001–008 with real saved evidence, including every width and all failed attempts.

## Complexity Tracking

No constitution violations. Temporary LR restoration avoids changing S2 ownership or serialized optimizer layouts. New-only schedule validation and legacy terminal adapters are necessary to prevent incorrect rates and unsupported provenance claims.
