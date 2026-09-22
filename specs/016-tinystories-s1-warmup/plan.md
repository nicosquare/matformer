# Implementation Plan: TinyStories S1 Fourfold LR Warmup

**Branch**: `016-tinystories-s1-warmup` | **Date**: 2026-09-22 | **Spec**: [spec.md](spec.md)

**Input**: Feature 016 and [source request](../../notes/tinystories_s1_warmup_speckit_prompt_2026-09-22.md).

**Status**: Planning and phases 1–5 implementation complete; CPU preparation/runtime/admission/reporting and compatibility verification is recorded in [verification.md](verification.md). Final cross-cutting verification, real input audits, readiness, production, and real comparison remain pending.

## Summary

Extend the existing optimizer-ownership pipeline with campaign schema 5 and exactly two fresh S1 runs, Linear and Geometric, with 256-update LR warmup inside the unchanged 348,528-update horizon. Preserve counterpart controls and random streams. Add per-run grid selection, strict counterpart-difference checks, schedule evidence, and a two-arm launcher using the CUDA-required trainer and existing Slurm safeguards.

Revalidate eight selected standalone and two original S1 terminals read-only. Export 24 endpoints, eight paired differences, standalone gaps, endpoint plots, and early LR/loss comparisons. Select valid terminals from the closed Feature 015 campaign without requiring cancelled arms. Preserve schemas 1–4, their strict 64-update warmup, and historical hashes.

## Technical Context

**Language/Version**: Python 3.12.13; `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python`

**Primary Dependencies**: Installed PyTorch 2.11.0+cu128, Transformers 5.8.0, NumPy 2.4.3, PyYAML 6.0.3, Matplotlib 3.10.9, pytest 9.0.3; no additions

**Storage**: Existing packed mmap corpus; YAML controls; JSON/JSONL/CSV evidence; resumable PyTorch checkpoints; PNG/PDF figures

**Testing**: Focused CPU protocol/schedule/restore/report/queue checks; later authorized real-shape CUDA bf16 diagnostics for both grids; historical signature and relevant runtime regressions

**Target Platform**: Linux/Slurm; one process and one GPU per training run; CPU preflight/reporting

**Project Type**: Controlled research experiment extension using existing trainer and saved-artifact comparison

**Experiment Scope**: Two seed-42 slicing/shared-AdamW S1 runs; uniform global replacement H=1; no correction; only warmup changes scientifically

**Datasets/Data Assumptions**: Audited TinyStories-Instruct tokenizer and four-role corpus; 5,576,448 designated sequences/epoch, fixed excluded tail of 43; ordinary validation only; controller reserved and final holdout sealed

**Configuration Inputs**: Schema-5 fixed two-arm recipe; explicit grids/reference mappings; inherited control/data hashes; 256 warmup updates, LR .008, source/config-bound evidence

**Experiment Outputs**: Config difference audit, full schedule/action/batch evidence, scalar trajectories, exposure/resources, two terminal checkpoints, eight new endpoints, 24 comparison endpoints, eight differences, figures and descriptive report

**Reproducibility Notes**: Fresh normal constructor at seed 42; preserve versioned initialization/action/data streams independently of names; exact counterpart action/batch equality; inherited numerical restore tolerances; frozen terminal/evaluation identities

**Performance Goals**: Measure throughput, attempt costs, optimizer state and CUDA peaks; bounded streaming accounting; no promised accuracy or resource improvement

**Constraints**: Subsequent authorization for GPU work; sbatch excluding `gpu-[05,50,51,54]`; user-wide two-running/four-submitted ceilings or stricter live limits; fresh artifact root and no reference retraining/writes

**Scale/Scope**: d64/l4/h4, context 128, vocabulary 2048, full FFN 256; two × four epochs × 87,132 updates; 697,056 assigned updates and 5,710,282,752 tokens total

Versions and saved S1 config hashes were inspected during planning. No new corpus audit, complete terminal audit, training, or readiness checks were performed. Design unknowns are resolved in [research.md](research.md).

## Constitution Check

| Principle | Before research | After design | Basis |
| --- | --- | --- | --- |
| I. Research code first | PASS | PASS | Existing trainer, scheduler, checkpoint and report paths; two explicit arms. |
| II. Simplicity/local reasoning | PASS | PASS | Arm-qualified grids and visible warmup-only difference audit. |
| III. Explicit experiment flow | PASS | PASS | Fixed YAML and focused launcher; no registry or alternative trainer. |
| IV. Minimal abstraction/validation | PASS | PASS | Checks address scientific confounds, invalid restore, duplicate jobs and false completion. |
| V. Configuration/reproducibility | PASS | PASS | New-only identity metadata, unchanged old serialization, counterpart streams and immutable evidence. |
| VI. Useful outputs/logging | PASS | PASS | Structured schedules, measured scalars, tables, resources and figures. |
| Shallow organization | PASS | PASS | Existing modules plus focused recipe, scripts and acceptance suites. |

No gate failure or constitution exception. Reuse pure operational helpers and artifact publication where appropriate; keep feature-specific admission/reporting policy visible. Do not build a generic campaign framework.

## Project Structure

### Documentation (this feature)

```text
specs/016-tinystories-s1-warmup/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── contracts/
    ├── protocol-and-schedule.md
    ├── lifecycle-and-cli.md
    └── reporting.md
```

Subsequent stages add `tasks.md`, `verification.md`, and `docs/tinystories-s1-warmup-experiment.md`; these are not delivered by this planning invocation.

### Source Code (repository root)

| Path | Planned responsibility |
| --- | --- |
| `configs/controlled_exps/tinystories_instruct_s1_warmup.yaml` | Exactly two schema-5 arms, inherited controls/hashes, explicit reference mappings. |
| `src/evaluation/optimizer_ownership.py` | Schema/arm grid selectors, expansion, warmup-only differences, schedule/counterpart traces, selected references, freeze and report. |
| `src/utils/config.py` | New arm output eligibility and schema-5 topology dispatch; resolved-control validation without relaxing old rules. |
| `src/utils/reproducibility.py` | Reuse stream derivation and extra-field contract hashing; preserve old serializer outputs. |
| `src/utils/metrics.py` | Arm-qualified grid validation in compact accounting; preserve recorded applied LR. |
| `src/training/run.py` | Grid-aware summaries/counts/endpoints and identity metadata; existing terminal-only recovery. |
| `src/training/steps.py` | Reuse current scheduler, applied-LR capture and complete-update ordering; verify semantic S1 eligibility for new names. |
| `src/training/checkpointing.py` | Bind new protocol/grid/schedule through existing whole-bundle restore validation. |
| `scripts/analyze_tinystories_optimizer_ownership.py` | Schema-5 preflight/freeze and `report-s1-warmup`. |
| `scripts/run_tinystories_s1_warmup.py` | Two-arm prepare/queue/worker/report; durable attempts, CUDA entry and independent stage status. |
| `scripts/preflight_tinystories_s1_warmup.py` | Immutable source snapshot and CPU/GPU gates for both grids and real-horizon boundary probes. |
| `scripts/train_cuda_required.py` | Reuse explicit CUDA bf16 entry point. |
| `scripts/plot_tinystories_standalones.py` | Reuse/extract read-only selected-run device validation while preserving its reports. |
| `tests/test_s1_warmup_campaign.py` | Protocol/counts, schedule, streams, actual updates, resume/failure and legacy identity tests. |
| `tests/test_s1_warmup_reporting.py` | Selected references, 8/24 endpoints, deltas, early scalars, plots and rejection cases. |
| `tests/test_s1_warmup_queue.py` | Two-only admission, stale gates, limits, duplicates, device evidence and recovery. |

**Structure Decision**: Keep scientific policy in the existing campaign module and training primitives in their current modules. Extend width/common/topology selectors with an optional arm argument required for schema 5; retain original defaults/results for schemas 1–4. A schema-5 campaign has no single width grid. Resolve new S1 labels through declared representation/ownership rather than literal-name checks. No trainer, scheduler or model fork is needed.

## Phase 0: Research

[research.md](research.md) resolves schema design, grids, control differences, schedule indexing, random streams, restore, historical selection, early scalars and operations. Two read-only research agents inspected independent source areas as directed by the skill. Installed dependency source establishes the scheduler convention for this environment.

## Phase 1: Design and Contracts

Use campaign ID `tinystories-optimizer-ownership-s1-warmup-v1`, schema 5, arms `S1-linear-w256` and `S1-geometric-w256`, and run IDs `<campaign_id>-<arm_id>-s42`. Proposed fresh root: `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-s1-warmup-v1`. It was absent during planning; preflight must recheck/reserve it.

[data-model.md](data-model.md) defines identities/evidence/state transitions. Contracts cover [protocol and schedule](contracts/protocol-and-schedule.md), [lifecycle and CLI](contracts/lifecycle-and-cli.md), and [reporting](contracts/reporting.md). [quickstart.md](quickstart.md) documents proposed commands and subsequent authorization boundaries.

Update root `AGENTS.md` to this plan. No update-agent-context script exists in this checkout; use the skill's explicit marker replacement. Post-design constitution gates pass, with no unresolved clarification.

## Phase 2: Implementation and Verification Sequence

This sequence informs later task generation; this command creates no tasks or implementation.

1. **US1 / FR-001–006, 012**: Add schema-5 arms/selectors, config and counterpart-difference checks. Audit actual counts, data, selected references and full four-epoch expected traces. Preserve schemas 1–4 and capture their pre-change signatures, including schema 4.
2. **US2 / FR-002, 007–009**: Verify all LR positions 0–348,528 and actual applied LR around both warmup boundaries. Exercise all-width S1 history/clipping, resume before/at/after warmup and each epoch boundary, rejection before mutation, optimizer/scheduler/accounting failure durability and zero-update terminal recovery. Distinguish state-seeded boundary probes from full training.
3. **US3 / FR-011–018**: Complete reporting fixtures before production: eight new endpoints, strict 24-row comparison, eight deltas/standalone gaps, exact-count figures, measured early LR/training loss/per-width validation, missing-evidence rejection and explicit incomplete statuses.
4. **US2 / FR-009–011**: Implement snapshot-bound gates and two-run launcher. Reuse reservation/lock/intents/continuation and explicit CUDA entry. Verify failed/stale gates, GPU absence, occupied identities, uncertain submissions, live user-wide limits, matching job/worker success and restart recovery. No standalone stage or cancelled-arm admission.
5. **FR-005, 009, 019 / compatibility**: Run focused CPU and relevant legacy config/campaign/metrics/checkpoint/report/queue regressions. Record hashes, commands, pass/fail/skip counts and limitations. A skipped GPU check cannot establish readiness.
6. **Subsequently authorized GPU readiness**: Through sbatch, verify both grids at d64/l4/h4, batch 64, context 128 and actual bf16, all widths, finite updates, full-horizon schedule and continuation boundaries. Bind gate evidence to tested source/configs and CPU gate; keep probes separate from production.
7. **Subsequently authorized production / FR-010, 013**: Fresh-start exactly two runs; continue only own durable state. Finish four epochs and terminal ordinary validation; freeze eight new endpoints with actual execution evidence and reconciled traces/resources.
8. **US3 / FR-014–019**: Revalidate ten historical terminals, publish 24 endpoints/eight differences, requested PNG/PDF views and eight-width descriptive findings. Align runbook/tasks/verification with saved evidence.

Implementation readiness is distinct from SC-004 terminal completion and SC-005–007 comparison completion. Missing references preserve valid new-run evidence and leave comparison completion outstanding. Overall acceptance requires SC-001–008, including real terminal results and figures.

## Complexity Tracking

No constitution violations. Arm-qualified grids prevent silent mislabeling of traces and counts. Source/config bindings and durable attempts reuse safeguards against known invalid CPU results and duplicate work. No per-update whole-state copies or growing in-memory attempt history.
