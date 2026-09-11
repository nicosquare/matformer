# Optimizer Lifecycle and Checkpoint Contract

## Construction and support

S1/C1/standalones retain one ordinary optimizer. S2/C2 retain
`PerGranularityOptimizerCollection` with four histories over the same ordered
model parameters. C3 adds `BlockOptimizerCollection` in `optimizer_state.py` with
ordered O-A/O-B/O-C/O-D/O-common optimizers over disjoint physical parameters.

Collect C3 quarter owners from every CatLlamaMLP's corresponding gate/up/down
ParameterLists, including segment biases. O-common is the identity-deduplicated
trainable remainder, including common FFN down bias, attention, norms, embeddings
and LM head. Tied objects appear once. Validate equal quarters, no overlap and
complete coverage before running. Reuse the same quarter/common grouping for
C1 clipping diagnostics without changing C1 optimizer ownership.

All gradients are cleared to absent once across the entire model. Inactive
concat quarters remain `grad is None`; active present zero gradients retain
ordinary AdamW updates. Sliced tensor gradients have full physical shape; do not
mask momentum/decay after stepping or compact their state.

## Active owners and committed accounting

Let n250/n500/n750/n1000 be committed width selections and T their sum. Define
aA=T, aB=n500+n750+n1000, aC=n750+n1000, aD=n1000. These are measured counts,
not target counts. Common parameters participate T times.

| Scope | Owner calls for selected width | Sum of successful owner calls |
| --- | --- | --- |
| shared | shared once | T |
| per_granularity | selected width once | T |
| per_ffn_block | active quarter prefix, then O-common, each once | T+aA+aB+aC+aD |

Each C3 owner's count equals its activation count; common count equals T.
Scheduler position equals T regardless of owner count. Standalone shared count
equals its own committed steps. Failed partial owner calls belong in attempt
failure records, not committed scientific counts.

## Explicit update sequence in train_for_steps

1. Capture existing pre-update transactional RNG/data/accounting state. Select one
   global action through the existing stream and determine active owners.
2. Clear all model gradients to None. Consume one batch and perform one ordinary
   forward/backward (campaign accumulation=1); no controller or clipping RNG.
3. Measure detached group pre-norms. Global arms apply the existing single global
   L2 cap. C3 applies L2 cap 1.0 to each active owner's combined parameter gradients
   with the existing numerical stabilization. Measure post-norms and record the
   actual coefficient used; no second rescale.
4. Verify finite loss/gradients/norms and synchronized current global LR before
   mutation. Set `update_in_flight=true`/unsafe-to-checkpoint before calling any
   optimizer. An optimizer may mutate before raising.
5. Shared/per-width arms step once. C3 steps active quarters in A/B/C/D order,
   then common once. Record returned owners locally until the update completes.
6. Advance the global scheduler once and synchronize all collection rates,
   including inactive owners. Account width/quarter/owner counts, tokens, data
   cursor and committed global step as one complete update.
7. Reconcile clocks/counts; clear the unsafe flag and publish the committed trace.
   Existing evaluation/checkpoint callbacks may now observe a complete boundary.

GlobalSchedulerClock remains the sole collection clock and uses each run's full
budget. Initial and resumed rates are synchronized before any owner step. No LR,
RNG, optimizer or data-membership reset occurs at epoch boundaries.

## Clipping semantics and diagnostic

C1 group pre/post observations describe its single applied global coefficient;
group thresholds are not independently applied. C3 coefficients depend only on
their owner's joint gradient norm and cap. Inactive observations have an explicit
inactive flag/null norm, not a fabricated clipping event. Active zero norms have
coefficient 1. Combined norms are the L2 combination of disjoint group norms.
C3's combined cap is sqrt(2) at g250 through sqrt(5) at g1000; these are gradient
bounds, not bounds on AdamW parameter updates.

A short internal test copies the same concat model state to C1 and C3, uses the
same data/actions/rates and temporarily selects global clipping for C3. Compare
all parameters and history components within existing tolerances. The nine
campaign initializations do not use this copying control.

## Failure boundary

Before mutation starts, restore existing transaction state and clear gradients.
After mutation starts, any owner, scheduler or accounting failure poisons the
live state, aborts training and preserves the previous durable checkpoint. Do
not restore just metadata and claim model rollback. Record pending step, failure
stage, active/returned owners and attempt cost without serializing partial state.
Every normal, exception, signal and finalization resumable-save path must reject
an unsafe/poisoned state. No per-update full model/optimizer snapshots are added.

Failures after a reconciled commit but during artifact IO follow existing durable
artifact policy; only a fully reconciled state is eligible for a later save.
Recovery always identifies the last successfully published durable checkpoint.

## Campaign resume schema and required histories

Add `optimizer_ownership_checkpoint_schema_version=1`, full campaign/run contract,
representation and ordered parameter descriptors, owner topology/support,
clipping, budget/epoch identity and resource-ledger watermark to resumable payloads.
Keep existing model, optimizer, scheduler, RNG, sampler and accounting surfaces.
Increment the relevant collection schema if its serialized layout changes; read
historical payloads only through their existing explicit compatibility path.
All new campaign arms, including shared standalones/S1/C1, require the campaign
schema. No model-only, cross-arm, representation or cross-scope migration.

For the fixed all-trainable campaign graph, required AdamW parameter steps are:

| Arm/history | Physical parameter | Required step |
| --- | --- | --- |
| Standalone/shared | all | committed standalone steps |
| S1/shared | all full tensors | T |
| S2/width g | all full tensors | ng |
| C1/shared | quarter q / common | aq / T |
| C2/width g | active quarter q / common | ng / ng |
| C2/width g | quarter excluded by g | 0 (state absent) |
| C3/quarter owner | that quarter | aq |
| C3/common | common | T |

Required step zero means no allocated state; a positive step requires `step`,
`exp_avg`, `exp_avg_sq` and any configured optional components. Check exact ordered
name-to-ID mapping, optimizer groups/kwargs, full moment shapes, expected state
dtypes, finite values, nonnegative second moments and integral counters equal to
exposure. Reject missing or impossible extra state; validation must not allocate
lazy state. S2 never-used tail moments remain zero while its parameter step ng
and full-shaped allocation are valid. C2 histories after every width is selected
are A/B/C/D = 4/3/2/1, common = 4.

## Whole-bundle restore

Stage validation without changing live model, RNG, optimizer, scheduler, sampler
or metrics. Check purpose/schema/identity, model keys/shapes/dtypes/finiteness,
ordered topology/groups and complete required histories, scheduler state/rates/
position/horizon, action RNG and ordinal, batch cursor/epoch/membership/budgets,
integer exposures/tokens, and metric/resource watermark consistency. Use local
temporary RNG/sampler objects to validate encoded state, not the live streams.

After all checks pass, install the bundle. A one-time restore snapshot/guard
restores every affected live component if installation unexpectedly raises;
component-local optimizer rollback alone is insufficient. This memory cost is
at resume only. Then synchronize LRs and reconcile T, exposure, cursor and clock.
Discard/segregate non-durable scientific trace rows past the restored boundary,
while preserving all attempt-cost and failure records.

Historical non-campaign state remains readable under its historical rules and
cannot be relabeled as an exact nine-run campaign checkpoint. Test both valid
lazy absence and lost required state before/at/after epoch boundaries.

## Correction extension lifecycle

Corrected backward hooks execute before the arm's clipping. For LMC capture only
active non-unit FFN blocks before stepping. C1/C2 use their single selected AdamW;
C3 steps its active owner prefix and common, then applies each captured correction
once before the single scheduler advance. Keep mutation-started/unsafe state set
until correction and all bookkeeping reconcile. A correction failure is fatal,
like any owner/clock/accounting failure, and cannot publish resumable partial state.
Full campaign contract validation rejects changed/missing correction metadata;
original contracts are unchanged. See [membership correction](membership-correction.md).
