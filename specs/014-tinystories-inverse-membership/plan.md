# Implementation Plan: TinyStories Inverse-Membership Sampling Comparison

**Branch**: `014-tinystories-inverse-membership` | **Date**: 2026-09-11 | **Spec**: [spec.md](spec.md)
**Input**: Feature 014 specification and authorized end-to-end campaign.

## Summary

Add an explicit schema-3 five-arm recipe to existing optimizer-ownership campaign
expansion/validation. Reuse fixed-global categorical selection, adding compact
policy-bound selection accounting for this campaign and weighted checkpoint
replay. Admit fixed-global C3 with its existing disjoint ownership and failure
boundary. Preserve schema-1 uniform and schema-2 correction controls and hashes.
Extend saved-artifact reporting for 20 new and 44 combined terminal endpoints,
paired policy deltas and five-arm full-range progress. Use a campaign-specific
launcher/diagnostic script, reusing proven operational helpers where safe.

## Technical Context

**Language/Version**: Python 3.12 in `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python`  
**Primary Dependencies**: Existing pinned PyTorch, Transformers, NumPy, PyYAML, Matplotlib, pytest; no added dependency  
**Storage**: Existing packed corpus, YAML controls, JSON/JSONL/CSV evidence, resumable checkpoints, PNG/PDF  
**Testing**: Focused CPU real-model/update/resume/failure, campaign/report/queue fixtures; sbatch bf16 five-arm smoke/resume at real controls  
**Target Platform**: Linux, Slurm, one process on one GPU per run  
**Project Type**: Controlled research training and comparison pipeline  
**Experiment Scope**: Only categorical width probabilities change within each original uncorrected arm  
**Datasets/Data Assumptions**: Existing audited four-role TinyStories-Instruct corpus; fixed designated set/excluded tail and ordinary validation; holdout sealed  
**Configuration Inputs**: Distinct schema-3 recipe, fixed_global distribution .12/.16/.24/.48, five arm overrides, pinned controls/hashes  
**Experiment Outputs**: Five complete runs; 20/44 endpoint exports; five PNG/PDF figure stems; exposure, clipping, resources and 20 paired deltas  
**Reproducibility Notes**: Normal seed-42 initialization, independent action/data streams, exact actions/batches, full-horizon continuation and frozen terminal identities  
**Performance Goals**: Measure steady-state throughput and total costs; no promised accuracy/speed improvement  
**Constraints**: All GPU work sbatch; excluded gpu-[05,50,51]; two running/four submitted user-wide or stricter; fresh artifact root; no reference retraining  
**Scale/Scope**: Five × 348,528 updates / 2,855,141,376 tokens, four epochs; total 14,275,706,880 tokens

## Constitution Check

Pre-research and post-design: all six principles PASS. Research flow stays visible
in existing trainer/campaign functions, no registry or second trainer. The compact
state is needed to prevent scientifically incorrect ownership counts and resume.
Schema-specific validation prevents silent policy mixing. Plain evidence and
figures satisfy reproducibility and useful-output requirements. A focused launcher
is justified by fixed identities, admission limits and duplicate prevention.

## Project Structure

### Documentation (this feature)

`specs/014-tinystories-inverse-membership/` contains spec, checklist, plan,
research, data-model, contracts, quickstart, tasks, analysis and verification.
`docs/tinystories-inverse-membership-experiment.md` is the execution runbook.

### Source Code (repository root)

- `configs/controlled_exps/tinystories_instruct_inverse_membership.yaml`: schema-3 recipe.
- `src/evaluation/optimizer_ownership.py`: five-arm definitions, sampling contract,
  trace generation, strict policy validation, endpoint/progress/delta reports.
- `src/utils/config.py`: fixed-global C3 eligibility with unchanged other constraints.
- `src/training/steps.py`: compact action-state selection/commit integration.
- `src/training/checkpointing.py`: campaign fixed-global state, policy-aware replay.
- `src/training/run.py`: policy-correct exposure expectations in run summaries.
- `scripts/analyze_tinystories_optimizer_ownership.py`: IM comparison command.
- `scripts/preflight_tinystories_inverse_membership.py`: real-control bf16 diagnostics.
- `scripts/run_tinystories_inverse_membership.py`: campaign-local prepare/queue/worker/report flow.
- `tests/test_inverse_membership_sampling.py`: campaign, sampler/model/resume/failure acceptance.
- `tests/test_inverse_membership_reporting.py`: export/progress/provenance acceptance.
- `tests/test_inverse_membership_queue.py`: admission and restart safety.

**Structure Decision**: Keep existing module ownership and simple explicit schema
branches. Reuse owner stepping, state validation, audit, freeze, metrics and
resource helpers. Isolate new operational identity from old campaign helpers.

## Phase 0: Research

[research.md](research.md) records inspected local evidence and decisions. No new
third-party technology or unresolved scientific choice requires external research.
Fixed-global already uses isolated `random.choices`; C3 rejects its config;
compact ownership accounting and restore replay currently assume uniform global.
Historical common inputs stay pinned, but their obsolete launch statements do not
apply to this explicitly authorized new campaign.

## Phase 1: Design and Contracts

See [data-model.md](data-model.md), [sampling/resume contract](contracts/sampling-and-resume.md),
[campaign/report/operations contract](contracts/campaign-and-reporting.md) and
[quickstart.md](quickstart.md). Update only AGENTS.md's current plan reference.

New schema-3 arm IDs S1-IM/S2-IM/C1-IM/C2-IM/C3-IM carry reference_arm_id and
sampling_policy. Recipe uses fixed_global plus explicit probability mapping;
uniform-only schedule/interval raw keys are omitted, resolved cadence remains
random replacement H=1. Sampling metadata includes ordered widths, membership,
probabilities, mode/policy/version/cadence and derived action seed. RNG state stays
in the checkpoint RNG payload. Whole-contract comparison precedes mutation.

Extend compact window-state applicability only for ownership fixed-global runs;
legacy generic fixed-global checkpoints retain their existing absent-state path.
New fixed-global state records policy/distribution explicitly; existing uniform
state encoding remains unchanged. Select through the existing weighted primitive,
commit counts only after complete updates, replay weighted selections on restore.

Freeze/report new terminals with existing strict controls and terminal readers.
Compare historical inputs under schema 1; compare epoch traces (not action traces)
between policies. Annotate policy in exports without rewriting historical files.
New reporting completes independently if historical evidence is unavailable;
comparison failure is explicit and does not fabricate endpoints.

## Phase 2: Implementation and Verification Sequence

1. US1: recipe/schema/contract, fixed-global eligibility, exact probability and
   expected action traces; reject altered controls; inspect real model counts.
2. US2: compact state/selection/commit/replay, all five real model paths including
   wider/narrower and epoch boundaries; mismatch rejection and partial-C3 failure.
3. US4 tooling before production: 20/44 endpoints, fixed policy labels, deltas,
   consistent colors/styles and full-range five-arm progress; provenance fixtures.
4. US3: fresh root, CPU evidence, audit/preflight and historical reference audit;
   snapshot source/configs; short GPU diagnostics for all five; bind gates to hashes.
5. Submit/monitor five full-budget runs within live limits; reconcile continuation
   attempts; freeze terminal ordinary-validation evidence; export and interpret.
6. Final regression and documentation evidence, requirement/task reconciliation.

Independent checks: exact categorical boundary mapping; identical five action
traces without finite-frequency assertions; wider/narrower gradient/history/state;
exact action/batch resume and numerical tolerances, malformed state unchanged;
failed owner cannot overwrite checkpoint; schema 1/2 remain strict; missing or
wrong endpoint identity fails; user-wide admission and pending intent reconciliation.

## Complexity Tracking

No constitution violations. Compact campaign state and explicit schema 3 are the
smallest changes that preserve historical semantics while making exposure/resume
correct. No per-step model/optimizer snapshots, dynamic plugin system or trainer fork.
