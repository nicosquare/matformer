# Implementation Plan: TinyStories Optimizer Ownership Comparison

**Branch**: `013-tinystories-optimizer-ownership` | **Date**: 2026-09-09 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/013-tinystories-optimizer-ownership/spec.md`

## Summary

Extend the existing trainer for the nine-run seed-42 comparison: four fresh dense
standalones and five elastic arms S1/S2/C1/C2/C3. Reuse current slicing,
concatenation, shared/per-width optimizer state, deterministic repeated epochs,
global action sampling and ordinary validation. Add a focused five-owner C3
collection with separate clipping and one global scheduler advance per complete
update. Strengthen campaign checkpoint validation to preserve required histories
and valid lazy absence, and prevent publishing partial multi-owner updates.

Add one fixed-matrix campaign YAML and one preflight/freeze/report script. Record
clipping, exposure, cumulative continuation costs and terminal ordinary-validation
provenance. Export 24 terminal endpoints and two combined PNG/PDF figures at exact
active non-embedding counts. Preserve historical Feature 12 behavior/artifacts.
Planning does not launch the campaign or evaluate the sealed holdout.

## Technical Context

**Language/Version**: Python >=3.12; existing `elasticnn` environment for checks  
**Primary Dependencies**: Repository pins: PyTorch 2.11.0 CUDA 12.8, Transformers 5.8.0, datasets 4.8.5, PyYAML 6.0.3, NumPy 2.4.3, pandas 3.0.2, Matplotlib 3.10.9, pytest 9.0.3; no new runtime dependency  
**Storage**: YAML configs; packed-mmap corpus; CSV metrics, JSON/JSONL provenance and summaries; PyTorch checkpoints; PNG/PDF figures  
**Testing**: pytest CPU real-model semantics, epoch/action traces, checkpoint rejection/failure injection, campaign/endpoint fixtures; short GPU bf16 verification during implementation  
**Target Platform**: Linux, one process on one GPU for campaign; CPU preflight/model checks and focused tests  
**Project Type**: Research training pipeline and fixed controlled-experiment analysis  
**Experiment Scope**: FFN representation, AdamW history ownership and clipping; existing objective and sampling policy  
**Datasets/Data Assumptions**: Audited TinyStories-Instruct four-role prepared corpus, immutable tokenizer/manifests/order, fixed aligned epoch set, ordinary validation and sealed final holdout  
**Configuration Inputs**: Common-controls/nine-arm YAML materialized into normal trainer YAMLs; new `training.optimizer.state_scope=per_ffn_block` and explicit clipping contract; shared/per_granularity defaults preserved  
**Experiment Outputs**: Preflight manifest, configs, action/batch traces and digests, metrics, owner/clipping counts, attempt resource ledger, terminal resumable checkpoints, terminal-validation sidecars, individual plots, 24-row CSV/JSON table and two combined PNG/PDF figures  
**Reproducibility Notes**: Seed 42 via normal initialization; no cross-shape initial-value equality claim; existing isolated action/data RNG, full schedule horizons, full scientific-contract hashes, immutable terminal hashes, exact action/batch resume traces  
**Performance Goals**: One forward/backward per update; no per-step full-state snapshots; measure storage, throughput, wall time and peaks rather than promise quality or speed gains  
**Constraints**: Disjoint C3 owners, cap 1.0 per active owner, one global LR clock; lazy C2 state; full-tensor S1/S2 semantics; no distributed block/per-width campaign, balancing, extracted initialization or automatic training  
**Scale/Scope**: d64/l4/h4, context 128, vocab 2048, FFN 256 in four quarters; 9 runs/24 endpoints; standalone 87,132 steps and 713,785,344 tokens each; elastic 348,528 steps and 2,855,141,376 tokens each; aggregate 17,130,848,256 tokens

All technical context is resolved. The system default Python is 3.10; use
`/home/ivo.navarrete/.conda/envs/elasticnn/bin/python` or activate that environment.

## Constitution Check

*GATE: Passed before Phase 0 and re-evaluated after Phase 1.*

| Principle | Initial gate | Design evidence |
| --- | --- | --- |
| I. Research code first | PASS | Existing trainer and fixed-matrix script, no service framework. |
| II. Simplicity/local reasoning | PASS | Five owners in the existing optimizer module; visible update order. |
| III. Explicit experiment flow | PASS | Interventions explicit in YAML/loop; no registry or second trainer. |
| IV. Minimal abstraction/validation | PASS | Partition, checkpoint and endpoint checks prevent silent scientific mismatches. |
| V. Configuration/reproducibility | PASS | Controls, seed, aligned membership, exclusions, histories and provenance persisted. |
| VI. Useful outputs/logging | PASS | Structured traces, metrics, summaries, durable state and diagnostic/endpoint plots. |
| Shallow organization | PASS | Existing modules, one campaign module/script, focused tests. |

No gate violation needs an exception. Strict validation is scoped to the new
campaign's scientific boundaries and newly supported owner scope.

## Project Structure

### Documentation (this feature)

```text
specs/013-tinystories-optimizer-ownership/
├── spec.md
├── inspection.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── configuration-and-campaign.md
│   ├── optimizer-lifecycle-and-checkpoint.md
│   ├── artifacts-and-comparison.md
│   └── cli-entrypoints.md
└── tasks.md                         # subsequent /speckit-tasks output
```

### Source Code (repository root)

```text
train.py                             # resolved owner/clipping preflight fields
configs/controlled_exps/
└── tinystories_instruct_optimizer_ownership.yaml  # new fixed campaign
scripts/
└── analyze_tinystories_optimizer_ownership.py     # new CLI
docs/tinystories-optimizer-ownership-experiment.md # new runbook
src/
├── models/ffn.py                     # quarter metadata; preserve forwards
├── training/
│   ├── optimizer_state.py           # five owners, topology, lazy validation, clock
│   ├── steps.py                     # clipping and logical commit lifecycle
│   ├── checkpointing.py             # staged campaign restore, unsafe-save gate
│   ├── run.py                       # wiring, terminal sidecar, attempt resources
│   ├── data.py                      # epoch/batch provenance and reconciliation
│   ├── packed_corpus.py             # reuse fixed-set repeat sampler
│   └── modeling.py                  # reuse dense/nested construction
├── utils/
│   ├── config.py                    # scope/clipping eligibility
│   ├── reproducibility.py           # campaign identity, preserve legacy hashes
│   ├── metrics.py                   # owner/clipping/resource fields
│   └── model_size.py                # reuse exact active-count convention
└── evaluation/
    ├── validation.py                # weighted evaluation/terminal provenance
    └── optimizer_ownership.py       # new fixed-campaign validation/export/plots
tests/
├── test_optimizer_ownership.py      # new real-model ownership/clipping tests
├── test_optimizer_ownership_resume.py
├── test_optimizer_ownership_campaign.py
├── test_optimizer_ownership_reporting.py
└── existing config/CLI/model-size/packed-corpus/optimizer/artifact/report tests
```

**Structure Decision**: Keep mechanics in `optimizer_state.py` and the update
sequence in `steps.py`. The campaign module supplies fixed-protocol expansion,
validation and reporting; the script is its thin CLI. Reuse audit, serialization,
metrics and validation primitives. Preserve the old analyzer's selection rules.

## Phase 0: Research

[research.md](research.md) records decisions, rationale, alternatives, local
source evidence and version-specific PyTorch references. Key resolved findings:

1. Existing FFNs already implement required zero-present/absent behavior.
2. C3 disjoint owners need a separate collection from overlapping width histories.
3. Current lazy-state validation allows missing required histories; whole restore
   validation must precede model installation.
4. Mutation-started and complete logical commit need distinct states.
5. Epoch machinery is reusable; campaign preflight must enforce the pinned audit.
6. The old analyzer prefers holdout or trailing means; this campaign requires
   terminal ordinary validation bound to durable checkpoint identity.
7. Resource totals need an attempt ledger beyond rollbackable checkpoint progress.

CPU construction using the actual model and count helper confirmed:

| Width | FFN dimension | Active non-embedding count: slicing, concat and standalone |
| --- | ---: | ---: |
| g250 | 64 | 115,264 |
| g500 | 128 | 164,416 |
| g750 | 192 | 213,568 |
| g1000 | 256 | 262,720 |

Full physical count is 524,864: F=196,608 quarter-owned FFN parameters and
R=328,256 common parameters including embeddings/head. These are planning checks,
not evidence of successful campaign preflight or training.

## Phase 1: Design and Contracts

- [data-model.md](data-model.md): campaign/run, ownership, histories, data cursor,
  logical update, checkpoint, resource attempt and endpoints/report.
- [Configuration and campaign](contracts/configuration-and-campaign.md): exact
  matrix, eligibility, normalized inputs, identities, audit and traces.
- [Optimizer lifecycle and checkpoint](contracts/optimizer-lifecycle-and-checkpoint.md):
  owner support, clipping, counters, staged restore and failure durability.
- [Artifacts and comparison](contracts/artifacts-and-comparison.md): traces,
  memory, clipping, continuation, terminal evaluation and figure contracts.
- [CLI entrypoints](contracts/cli-entrypoints.md): preflight/materialization,
  existing trainer, terminal freeze, complete/explicit-partial report.
- [quickstart.md](quickstart.md): ordered verification and future commands.

Update `AGENTS.md` between its existing markers to reference this plan. There is
no agent-context update script in this checkout; use the skill's direct marker
replacement instruction.

### Post-design constitution re-check

All gates remain PASS. The extra collection is justified by disjoint ownership;
the new reader by incompatible historical endpoint rules. The one-time restore
guard sits outside the hot loop and protects against partial installation. The
simple attempt ledger prevents rolled-back progress from erasing consumed costs.
There is no new framework or unresolved gate.

## Phase 2: Implementation Sequence

This sequence defines task-generation boundaries; `/speckit-tasks` creates the
ordered task list later. This command does not implement runtime code.

1. **Configuration/matrix/identity (US1; FR-001–005,015; EX-001–009)**: scope and
   clipping gates, common/arm YAML expansion, pinned audit, CPU model counts,
   deterministic expected data/action traces and fresh-identity validation.
2. **Ownership/lazy histories (US2; FR-006–011)**: disjoint C3 owners, common/tied/
   bias coverage, support descriptors, state expectations; real-FFN diagnostics.
3. **Clipping/complete update (US2/3; FR-012–014,017,019,021)**: active owner
   clipping/steps, global clock, separate counts and unsafe-save flag. Verify
   matched-state C1/C3 global-clipping diagnostic and owner-failure handling.
4. **Exact resume (US3; FR-016–020)**: versioned full-payload staging, lazy
   validation, restore guard, epoch/action reconciliation and attempt-cost ledger.
5. **Audit/terminal artifacts (US4; FR-018–023,025)**: compact traces, clipping
   summaries, measured storage/costs, terminal-validation sidecars, individual plots.
6. **Complete comparison (US5; FR-024,026–027; EX-010–013)**: freeze nine terminals,
   validate 24 endpoints/full controls, export table/figures and interpretation.
7. **Verification/runbook (FR-028; SC-001–008)**: focused CPU and compatibility
   suites, short GPU bf16 diagnostics when available, explicit future launch and
   reporting commands. Full campaign execution is a separate researcher action.

## Verification Strategy

| Boundary | Required evidence |
| --- | --- |
| Preflight | Nine arms accepted; changed controls/hashes/tail/horizon rejected; real shapes/counts and fresh identities. |
| Data/actions | First-epoch equality across nine; four epoch/action trace equality across five; independent uniform replacement without balance enforcement. |
| S1/S2 | Wider then narrower tail momentum/decay; full-sized state; selected-width history isolation. |
| C1/C2 | Inactive quarters unchanged with absent gradients; C2 4/3/2/1 histories only after exposure. |
| C3 | Disjoint tied/bias coverage; selected owners/common once and clock once; independent clipping and sqrt(N) combined bound. |
| Diagnostic | Test-only C3 global clipping matches C1 with identical concat starting values, actions/data/rates. |
| Resume | Exact action/batch traces before/at/after epoch boundary; numerical state parity; malformed/cross-arm/model-only payloads rejected before mutation. |
| Failure | First/middle/last owner or later scheduler/accounting failure aborts; prior durable checkpoint unchanged. |
| Resources | Durations sum unique attempts, peaks take max; no double counting; actual tensor bytes and separate temporary/device peaks. |
| Endpoints | Nine terminals/24 rows; reject best/nonterminal/trailing-mean, mixed roles, nonfinite/duplicate/missing inputs, wrong counts/targets/budgets/hashes; sidecar failure recovers from the same terminal without another update. |
| Figures | Two stems, PNG/PDF each; five four-point curves and four disconnected markers; exact x, consistent colors/annotations, no seed bars. |
| Compatibility | Shared/per-width defaults/checkpoints, distributed shared mode, old analyzer, reporting, global sampling and repeat sampler. |

Use existing numerical tolerances; require exact actions, batches, integer counts,
inactive unchanged tensors and identity hashes. Do not assert initial tensor
identity across campaign shapes. Planning validation covers documents/links and
budget arithmetic; runtime implementation tests are future work.

## Complexity Tracking

No constitution violations require justification.

## Authorized correction extension — 2026-09-10

FR-029–035 / SC-009–011 extend the completed original campaign. No new feature,
branch, corpus, dependencies or trainer. The original matrix and hashes remain
strict schema 1; a separate schema-2 six-arm recipe selects fixed corrected concat
arms. Reuse expansion/audit/terminal readers with explicit schema-specific arm
lists, never process-global patching or relabeling historical inputs.

Execution order: update spec and resolve clarification (done from user inputs),
contract/design, tasks, read-only consistency analysis, CPU implementation tests,
GPU preflight, source snapshot and six-run launch, monitoring, terminal reporting.
Remove only C3's correction restriction. Capture active non-unit FFN parameter
values once before C3's owner loop, apply LMC once after all owners return, then
advance the global clock. Keep the unsafe flag set through correction/accounting;
only affected parameter values are copied, never optimizer states or histories.
C1/C2 retain their helper and selected optimizer. GMC remains backward hooks.

Correction contract v1 records mode, four trained widths, membership counts,
factors, scope, bias/common treatment, order and decay/moment semantics. Add it
only to new campaign contracts, preserving legacy hashes. Existing whole-contract
resume validation binds it. Scalar correction context and base LR remain separate.

Verification uses real resolved full-model C1/C2/C3 paths, explicit per-parameter
LR reference optimizers, all widths after full-width exposure, bias/decay cases,
GMC/clipping/moment checks, exact resume and correction failure injection. CPU
fixtures may shorten budgets; production never does. GPU diagnostics use sbatch,
all six combinations, own directories, checkpoint/restore and steady-state timing.
Compare current none/GMC/LMC diagnostic throughput with identical short controls.

Operational tooling: one campaign-local locked submission helper, atomic intent
and Slurm-name reconciliation, owner writer locks, at most four submitted GPU jobs
and two running including other user jobs, excluded nodes gpu-[05,50,51]. All
artifacts including source snapshots/logs/configs/evidence below the requested
campaign root. Use immutable fresh identities, independent seed/data streams, and
original terminal references. No old helper is restarted. Produce strict frozen
terminal evidence and 40-row combined reports with three PNG/PDF figure stems.

Constitution re-check: all six principles PASS. Explicit schema branch and a small
campaign helper are justified by preserving original scientific validation and
restart-safe resource limits. No generic service/registry or full-state rollback.
