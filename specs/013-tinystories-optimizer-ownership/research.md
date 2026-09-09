# Research: TinyStories Optimizer Ownership Comparison

**Date**: 2026-09-09 | **Spec**: [spec.md](spec.md)

## Scope and evidence

The specification settles all scientific choices: seed-only initialization,
threshold 1.0 per C3 owner, and ordinary validation. Research concerns integration
with the current repository, not reconsideration of those choices. Two parallel
read-only research tasks inspected ownership/checkpoints and campaign/reporting.
The previous [corpus audit](inspection.md) supplies pinned identity evidence;
implementation preflight must repeat the audit before accepting a campaign.

Dependencies are the versions pinned in `requirements.txt`, not newer releases.
No new runtime dependency or full training launch is needed for this design.

## R1 — Preserve the existing representations

**Decision**: Reuse `ModifiedLlamaMLP.forward` and `CatLlamaMLP.forward` in
`src/models/ffn.py`. Use `build_model` in `src/training/modeling.py` for fresh dense
standalones and nested models. Every run initializes normally with seed 42.

**Rationale**: Slice backward already reaches full physical tensors; concatenation
only connects active quarter parameters. AdamW distinguishes present zero
gradients from absent gradients, so clearing gradients to `None` preserves the
intended inactive-block semantics. This behavior is documented by
[PyTorch 2.11 AdamW](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html).
Full-shaped sliced state and decay must remain ordinary optimizer behavior.

**Alternatives considered**: Canonical initialization copying would contradict
the settled campaign protocol; compact slicing moments or tail restoration would
change S1/S2. Copying values is only a short C1/C3 test control.

## R2 — Add one explicit five-owner runtime

**Decision**: Add `BlockOptimizerCollection` beside
`PerGranularityOptimizerCollection` in `src/training/optimizer_state.py`.
Expose it through the new scope `per_ffn_block`, restricted to the supported
four-quarter, concatenation, single-process AdamW path with global width actions.
Keep ordinary shared and per-granularity construction intact.

**Rationale**: C3 needs disjoint parameter ownership whereas per-width optimizers
intentionally overlap. Build O-A/B/C/D from gate/up/down block parameters and
segment biases at each layer. O-common is the identity-deduplicated remainder,
including common output bias and tied embedding/head parameters. Model-based
coverage checks catch silent omissions without a generic optimizer registry.

**Alternatives considered**: Forcing disjoint owners into the existing collection
would violate its identical-parameter-order invariant. One optimizer with five
groups obscures the requested independently activated owner calls.

## R3 — Keep clipping explicit and observable

**Decision**: Resolve a separate clipping contract. C1 clips the full active
gradient vector once; C3 clips each active owner's joint L2 norm independently
at 1.0. Record group norms before/after, applied coefficients, active flags,
combined norms, and width-conditioned clipping frequencies.

**Rationale**: `clip_grad_norm_` treats its iterable as one gradient vector and
modifies gradients in place; call it once per intended group, using the existing
stabilization convention. See
[PyTorch 2.11 clipping documentation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.utils.clip_grad_norm_.html).
Diagnostics read detached values and reuse measured norms where possible; they
must not trigger another rescale or RNG draw. The disjoint-group combined cap
is sqrt(N), not a cap on AdamW parameter updates.

**Alternatives considered**: Element clamps, per-tensor caps, and a second global
clip each implement a different intervention. The C3/global test mode is internal
to diagnostics and is rejected by campaign preflight.

## R4 — Reuse the action and schedule streams

**Decision**: Use existing global `random_with_replacement`, H=1, uniform sampling
and isolated action/data RNG streams. Reuse `GlobalSchedulerClock` for C3, with
the complete run horizon; synchronize all owners before the first update and
after each clock advance and resume.

**Rationale**: `src/training/steps.py` already selects once per accumulation
window. Existing clock code fans out the current rate without duplicating the
schedule formula. The five elastic arms share 348,528-step schedules; standalones
share 87,132-step schedules. Differences between horizon groups are intended.

**Alternatives considered**: Feature 12 balanced cycles, owner-local clocks,
forced first exposure, or schedule restarts at epoch boundaries change controls.

## R5 — Treat mutation and commit as separate events

**Decision**: Mark an update unsafe for checkpointing before the first owner
step. Execute active C3 owners in O-A/B/C/D/common order, then one scheduler
advance and reconciled accounting. Only then mark a complete committed boundary.
Any failure after mutation begins aborts and retains the prior durable checkpoint.

**Rationale**: The existing `optimizer_committed` flag only records successful
return from one optimizer. An optimizer can mutate and raise; C3 can also fail
after an earlier owner returned. Restoring RNG/accounting alone cannot repair
weights. The unsafe flag must survive exception handling and gate every save path.

**Alternatives considered**: Per-step full-state snapshots are explicitly
excluded. Continuing with partially updated weights invalidates the experiment.

## R6 — Validate required lazy state, not just existing state

**Decision**: Add a versioned campaign resume contract with ordered parameter
descriptors, representation, owners/support, clipping, provenance, budgets,
exposures, and scheduler/data/RNG identities. Infer required AdamW components and
parameter counters from committed exposure and static parameter support.

**Rationale**: Current `PerGranularityOptimizerCollection.validate_state_dict`
checks only entries present in the payload; missing previously required moments
can pass. C2 must preserve truly unexposed absence while rejecting lost required
state. Shared campaign arms also need full validation. See the exact counter
rules in [the runtime contract](contracts/optimizer-lifecycle-and-checkpoint.md).

**Alternatives considered**: Eager allocation destroys C2's storage semantics;
requiring state for every registered parameter rejects valid lazy checkpoints;
trusting optimizer numeric IDs alone does not prove parameter identity.

## R7 — Validate the entire restore before installation

**Decision**: Stage model, optimizer, scheduler, action/RNG, sampler, metrics,
and accounting validation before live mutation. A one-time restore guard covers
unexpected installation failures; it does not run in the per-step path.

**Rationale**: `load_checkpoint_state` currently installs model state before all
other components. Component-local rollback is insufficient to meet whole-run
mutation-free rejection. Historical non-campaign payloads retain their existing
compatibility path; they cannot masquerade as exact campaign resumes.

**Alternatives considered**: Optimizer-only validation or cross-arm migration
silently changes the experiment. A second permanently resident model is unnecessary.

## R8 — Pin the aligned data contract

**Decision**: Reuse the repeat-epoch sampler's fixed prefix of the stored
permutation: 5,576,448 designated sequences, 43 excluded, 87,132 updates per epoch.
Validate all hashes in `inspection.md`, generate deterministic order/action
digests without forward/backward, and record actual runtime batch/action traces.

**Rationale**: `src/training/packed_corpus.py`, `data.py`, and config resolution
already support repeated deterministic epochs. Each standalone shares elastic
epoch one; elastic runs share all four. Draw counts are random, so selected-width
exposure and block activation are separate counters.

**Alternatives considered**: Rotating the excluded tail, minimum-token validation
alone, balancing action counts, or reusing a historical one-pass run violates
the protocol. Re-auditing is a preflight requirement, not evidence of training.

## R9 — Use one dedicated campaign workflow

**Decision**: Add one explicit campaign YAML with common controls and nine arm
overrides, and one `scripts/analyze_tinystories_optimizer_ownership.py` entrypoint
for preflight/materialization, freeze, and report. Materialize ordinary trainer YAMLs;
training continues through `train.py`.

**Rationale**: The existing per-width analyzer requires three seeds and a distinct
holdout protocol. A local fixed-matrix validator compares actual controls and
intended differences rather than weakening that historical analyzer or trusting
its paired-control hash. Full campaign/run contracts get their own hashes.

**Alternatives considered**: Nine copied recipes invite drift. A generic sweep
engine, new trainer, or experiment database adds unnecessary machinery.

## R10 — Bind ordinary-validation endpoints to terminal checkpoints

**Decision**: Extend terminal ordinary-validation artifacts with evaluated target
counts, evaluation protocol, active counts and checkpoint hash. Freeze nine
terminal identities and consume their 24 saved evaluations for reports.

**Rationale**: Existing validation computes target-weighted causal loss and
`exp(loss)`; `src/utils/model_size.py` counts active non-embedding parameters.
Preserving those helpers avoids alternative count/loss definitions. A terminal
sidecar must be written only when both evaluation and durable checkpoint agree
on the same committed model state.

**Alternatives considered**: Best-validation/epoch-one checkpoints and averaged
batch perplexities are invalid endpoints. Plotting does not evaluate holdout or
silently fall back to another checkpoint.

## R11 — Preserve resource costs across attempts

**Decision**: Extend resource accounting with a durable per-attempt ledger and
checkpoint watermark. Sum each unique attempt's latest elapsed duration and take
the maximum peak, including work lost since the resumed checkpoint. Measure
optimizer tensors by owner/component/dtype using actual sizes.

**Rationale**: `run.py` currently subtracts the latest process start time and
resets CUDA peaks. Checkpoint-only totals omit failed work after that checkpoint.
The ledger stores operational costs separately from replayable scientific state.
Report allocated/reserved GPU peaks separately from persistent state and temporary
concatenation buffers. Unfinalized abrupt-kill attempts have incomplete cost
measurement and must be flagged rather than reported as exact complete totals.

**Alternatives considered**: bf16-derived byte estimates, summing repeated ledger
snapshots, or overwriting totals at resume misreports costs. Per-step full-state
inspection is unnecessary; sample allocations at reporting/checkpoint boundaries.

## R12 — Verify semantics through model paths

**Decision**: Use focused CPU real-model diagnostics, restore/failure injection,
small repeat-epoch fixtures, complete campaign/report fixtures, and a later short
GPU bf16 check. Keep training launches outside planning.

**Rationale**: A toy optimizer alone cannot prove sliced gradients, concat lazy
absence, tied ownership, dense matching, or terminal provenance. Existing pytest,
Matplotlib, and artifact tools suffice; no additional library is needed.

**Alternatives considered**: Full-budget runs are too expensive for correctness
tests; memorized expected plots do not test missing/mixed endpoints or count
conventions. Verify the plotted series data and PNG/PDF exports directly.

All technical unknowns are resolved. Implementation verification remains future
work and is not claimed by this research record.
