# Implementation Plan: TinyStories Optimizer Ownership with MatFormer Widths

**Branch**: `015-tinystories-matformer-widths` | **Date**: 2026-09-21 | **Spec**: [spec.md](spec.md)
**Input**: Feature 015 specification and [source request](../../notes/tinystories_matformer_widths_speckit_prompt_2026-09-21.md).
**Status**: Planning complete. Implementation, diagnostics, and production execution remain future work requiring their respective conversation authorization.

## Summary

Add campaign schema 4 to the existing optimizer-ownership pipeline for four fresh
dense standalones at FFN dimensions 32/64/128/256 followed by five fresh elastic
S1/S2/C1/C2/C3 runs. Reuse explicit FFN prefixes and uniform global replacement
sampling. Admit 32/32/64/128 concat blocks through an explicit campaign topology
path, retaining five C3 owners, current AdamW semantics, independent cap-1 clipping,
and the complete-update checkpoint boundary.

Make campaign accounting, compact metrics, terminal validation and plotting use
the selected physical grid. Keep schemas 1–3 and their signatures strict. Add a
campaign launcher with a durable, validated standalone-first barrier and exact
source/config evidence gates. Export 24 new endpoints independently, then 28
combined endpoints using only four revalidated historical standalones. Preserve
both standalone measurements at shared sizes.

## Technical Context

**Language/Version**: Python 3.12.13; `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python`  
**Primary Dependencies**: Installed PyTorch 2.11.0+cu128, Transformers 5.8.0, NumPy 2.4.3, PyYAML 6.0.3, Matplotlib 3.10.9, pytest 9.0.3; existing repository dependencies, no additions  
**Storage**: Packed-mmap corpus; YAML controls; JSON/JSONL/CSV manifests, traces and metrics; resumable PyTorch checkpoints; PNG/PDF figures  
**Testing**: Focused CPU real-model/update/resume/failure checks, scheduling fixtures and 24/28 endpoint fixtures; separately authorized sbatch bf16 diagnostics at real shape/batch, plus epoch-boundary resume probes  
**Target Platform**: Linux/Slurm; one process and one GPU per training run; CPU preflight/reporting  
**Project Type**: Controlled research training and saved-artifact comparison pipeline  
**Experiment Scope**: New width grid under original uncorrected ownership/clipping arms; uniform global replacement at H=1; four mandatory fresh dense baselines  
**Datasets/Data Assumptions**: Audited TinyStories-Instruct tokenizer and four-role corpus; 5,576,448 designated sequences/epoch, fixed 43-sequence excluded tail; ordinary validation, controller reserved, final holdout sealed  
**Configuration Inputs**: Schema-4 fixed nine-arm recipe, explicit grid/boundaries, inherited controls/data hashes; source-bound readiness evidence and distinct run identities  
**Experiment Outputs**: Resolved controls, expected/committed traces, exposure/clipping/resources, nine terminal checkpoints, per-run diagnostics, 24/28-row CSV/JSON tables, combined loss/perplexity figures in PNG/PDF  
**Reproducibility Notes**: Seed 42 normal fresh construction; independent action/data streams; exact action/batch continuation and inherited numerical tolerances; frozen terminal ordinary-validation identities; no cross-model initial-tensor equality assumption  
**Performance Goals**: Measure steady-state throughput, attempt costs, actual state storage and device peaks; keep compact accounting bounded; no promised accuracy, memory or speed improvement  
**Constraints**: Validated standalones before elastic admission; all GPU work via sbatch excluding gpu-[05,50,51]; user-wide two-running/four-submitted ceilings or stricter live limits; distinct artifact root; preserve prior campaigns  
**Scale/Scope**: d64/l4/h4, context 128, vocabulary 2048, full elastic FFN 256; four × 87,132 updates plus five × 348,528 updates; 17,130,848,256 assigned tokens excluding diagnostics/replay

Environment versions were inspected through package metadata during planning.
Actual new model counts, corpus/reference validity, tests, and readiness gates
are not claimed as verified. Design choices are resolved; runtime validation
remains implementation/execution acceptance work.

## Constitution Check

| Principle | Before research | After design | Evidence |
| --- | --- | --- | --- |
| I. Research code first | PASS | PASS | Extend current trainer/campaign module; no training framework. |
| II. Simplicity/local reasoning | PASS | PASS | Explicit schema branches, physical boundaries, visible stage barrier. |
| III. Explicit experiment flow | PASS | PASS | Fixed YAML matrix and ordinary train.py path; no optimizer registry. |
| IV. Minimal abstraction/validation | PASS | PASS | Checks target scientific mismatches, partial updates, duplicate writers and false completion. |
| V. Configuration/reproducibility | PASS | PASS | New-only topology metadata, seeds, hashes, full horizons and durable identity; legacy signatures retained. |
| VI. Useful outputs/logging | PASS | PASS | Structured traces/metrics/resources, strict exports, figures and evidence-driven runbook. |
| Shallow organization | PASS | PASS | Existing modules plus focused launcher/preflight scripts and acceptance suites. |

No gate failure or constitution exception. Reuse locking, snapshot and artifact
publication helpers where their assumptions fit; keep stage policy visible in
the new launcher. Do not build a generic scheduling framework.

## Project Structure

### Documentation (this feature)

```text
specs/015-tinystories-matformer-widths/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── campaign-and-topology.md
│   ├── lifecycle-and-operations.md
│   └── reporting-and-cli.md
└── tasks.md                 # subsequent /speckit-tasks output, not created here
```

Implementation will add `docs/tinystories-matformer-widths-experiment.md` and
`specs/015-tinystories-matformer-widths/verification.md` with actual evidence.

### Source Code (repository root)

| Path | Planned responsibility |
| --- | --- |
| `configs/controlled_exps/tinystories_instruct_matformer_widths.yaml` | New schema-4 recipe, original controls and data hashes. |
| `src/evaluation/optimizer_ownership.py` | Schema-specific grids/arms, expansion/preflight, strict selected-run terminal validation, 24/28 exports/figures. |
| `src/utils/config.py` | Explicit new-campaign topology eligibility and materialized control consistency; old restrictions retained. |
| `src/training/optimizer_state.py` | Partition validation against actual dimensions, existing support-driven history/storage logic. |
| `src/training/steps.py` | Pass topology to C3/C1 clipping groups; preserve commit/failure order. |
| `src/training/checkpointing.py` | Carry/validate new identity and boundaries through whole-bundle restore. |
| `src/utils/metrics.py` | Contract-aware g125 acceptance in compact attempt accounting; semantic clipping predicates retained. |
| `src/training/run.py` | Grid-aware summaries, exposure/storage estimates, explicit source-width endpoints; resource attempts linked to jobs. |
| `scripts/analyze_tinystories_optimizer_ownership.py` | Extend preflight/freeze/report; add report-matformer-widths. |
| `scripts/run_tinystories_matformer_widths.py` | New prepare/queue/worker/report flow, standalone barrier, continuation reconciliation. |
| `scripts/preflight_tinystories_matformer_widths.py` | CPU/GPU readiness evidence with exact source/config bindings. |
| `tests/test_matformer_widths_campaign.py` | Definitions, geometry/counts, histories, clipping, resume/failure and metrics integration. |
| `tests/test_matformer_widths_reporting.py` | Strict 24/28 endpoints, historical selection, repeated-size markers and compatibility. |
| `tests/test_matformer_widths_queue.py` | Barrier, live limits, restart/attempt reconciliation, gate provenance and report recovery. |

**Structure Decision**: Existing granularity/FFN, dense construction, data sampler,
validation, scientific hashing and optimizer storage helpers supply the required
primitives. Exercise them with the new grid; avoid parallel model, trainer,
sampler or reporting frameworks. Add small explicit grid/arm selectors in the
current campaign module instead of mutating legacy constants.

## Phase 0: Research

[research.md](research.md) resolves geometry eligibility, schema/identity,
accounting, storage, scheduling, evidence gates and historical-only reporting.
Three read-only research agents inspected topology/restore, reporting/metrics,
and operations. Primary MatFormer sources confirm the intended FFN grid. Local
inspection identifies equal-quarter guards despite general FFN support, and a
g125 restriction in compact attempt IDs. The historical literal-arm clipping
bug is already fixed; retain that fix and test new identities.

## Phase 1: Design and Contracts

[data-model.md](data-model.md) defines identities, boundaries, owner histories,
barrier/evidence records, attempts and endpoints. Contracts specify
[campaign/topology](contracts/campaign-and-topology.md),
[lifecycle/operations](contracts/lifecycle-and-operations.md), and
[reporting/CLI](contracts/reporting-and-cli.md).
[quickstart.md](quickstart.md) describes future verification and execution.

Use campaign ID `tinystories-optimizer-ownership-matformer-widths-v1`, recipe
schema 4, and run IDs `<campaign_id>-<arm_id>-s42`. Add explicit schema/topology
metadata only to new configurations/contracts. Resolve arms by campaign schema
plus short label, since S1 and ST-g250 also exist historically. Scientific
contract schema 1 retains its existing serializer: extra-field support hashes
new topology fields without changing historical inputs.

Default partition validation stays equal-quarter. Schema 4 opts into one exact
MatFormer grid; validate derived block shapes and support before training. Keep
legacy serialized `quarter` names where necessary, documenting that A/B/C/D
refer to declared incremental blocks. No broad topology relaxation.

Update the root `AGENTS.md` reference to this plan. This checkout has no
update-agent-context script; use the skill's explicit marker replacement.

## Phase 2: Implementation and Verification Sequence

This sequence informs the later task-generation workflow; this command does not
create `tasks.md`.

1. **US1 / FR-001–006, 011**: Add schema-4 definitions/topology opt-in; preserve
   legacy resolution/hashes. Verify nine actual models/counts, unequal blocks,
   full expected traces, pinned controls and rejection cases.
2. **US2 / FR-007–010, 014–016**: Extend partition/clipping diagnostics and
   grid-aware compact accounting/summaries. Exercise every width, wider/narrower
   transitions, tied/bias coverage, active zero gradients, lazy histories, real
   C1/C3 sidecars and measured storage.
3. **US3 / FR-012–013**: Verify whole-bundle restore for all nine definitions
   before/at/after epoch boundaries, numerical state and exact action/batch
   equivalence, malformed-state rejection, installation rollback, and owner/
   scheduler/accounting failure durability. Test zero-step terminal recovery.
4. **US4–5 / FR-019–025**: Complete reporting before production: strict 24/28
   fixtures, four-only historical selection, coincident standalone preservation,
   missing/corrupt input rejection, figure structure and per-run diagnostics.
5. **US3 / FR-003, 016–018**: Implement hash-bound evidence generation,
   standalone barrier, queue/worker restart checks, unique attempts, uncertain
   submission reconciliation, snapshots and resource completeness.
6. **FR-017, 026 / compatibility**: Run focused new suites and relevant ownership,
   config/checkpoint/metrics/reporting regressions for schemas 1–3. Save commands,
   counts and hashes; skipped GPU cases do not establish GPU readiness.
7. **Separately authorized GPU verification/execution**: Run real-shape bf16
   diagnostics and controlled epoch-boundary resume/failure probes through sbatch.
   Freeze passed gates; complete/validate four standalones, then admit five
   elastics, continuing each only from its own durable state.
8. **Evidence completion / FR-026**: Freeze nine full terminals; publish 24 new
   endpoints and per-run diagnostics; revalidate four historical standalones and
   publish 28-point comparison if valid. Interpret descriptively and reconcile
   tasks/runbook/verification from saved artifacts.

Production acceptance requires SC-001–010, including real terminal evidence;
implementation diagnostics alone do not satisfy SC-005–009. Missing historical
references permit new-report completion but leave combined completion outstanding.

## Complexity Tracking

No constitution violations. Schema-specific grids and a durable standalone
barrier prevent physical-width mislabeling and premature admission. Existing
full-state restore guards stay confined to restore; no per-update model/optimizer
copies or growing attempt-ID histories are introduced.
