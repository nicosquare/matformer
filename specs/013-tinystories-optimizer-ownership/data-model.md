# Data Model: TinyStories Optimizer Ownership Comparison

Proposed additions below describe implementation contracts, not currently
available runtime fields. Existing compatible artifact fields are reused.

## Campaign and run

`CampaignContract` (schema 1) contains campaign ID, seed 42, common scientific
controls, exactly nine ordered arm definitions, pinned corpus/tokenizer/role
hashes, designated membership/alignment, schedule horizons, evaluation protocol,
count convention, and per-arm resolved contract hashes.

`CampaignRun` contains run/campaign/arm IDs, resolved trainer config,
representation (`dense`, `slicing`, `concat`), ownership (`shared`,
`per_granularity`, `per_ffn_block`), clipping, physical/source width metadata,
initialization provenance, assigned updates/tokens/epochs, output path and full
scientific identity. Provenance records seed, initializer range, normal
constructor and code/dependency versions, without asserting equal initial tensors.

Lifecycle: `defined -> preflight_passed -> running -> completed -> frozen`.
Failed attempts may resume the same run from a durable boundary. Changing arm,
budget, initialization or historical identity creates a new run, not continuation.

## Width and physical parameter topology

`Width` contains label, source fraction, active FFN dimension, active non-embedding
count and active quarters. Elastic widths g250/g500/g750/g1000 resolve to dimensions
64/128/192/256. Dense models retain source labels while their physical FFN is
already sized and their local active fraction is 1.0.

`ParameterDescriptor` contains canonical name, tied aliases, shape, dtype,
trainable flag, scalar count, component, quarter ID if any, and widths whose
backward produces a gradient for the physical tensor. Descriptors deduplicate
physical identity; saved optimizer IDs map through this stable ordered list.
Sliced tensors have physical gradient support at every width.

## Owner and history

`OptimizerOwner` contains ordered ID, parameter descriptors, activation support,
optimizer family/kwargs, state dictionary, successful calls, current LR and
clipping group. Shared/per-width parameter sets overlap intentionally; C3 sets
form a complete disjoint partition.

`History`, keyed by `(owner_id, parameter_name)`, contains AdamW `step`, `exp_avg`,
`exp_avg_sq` and configured optional components, with measured dtype/shape/bytes.
Lazy absence means no state entry, not allocated zero tensors. Sliced moments
retain full physical shape. Required state exists iff committed owner exposure
and physical parameter support prove at least one gradient-bearing update.
The [runtime contract](contracts/optimizer-lifecycle-and-checkpoint.md) gives exact
counter rules for each arm.

## Clipping

`ClippingContract` (schema 1) contains mode `global|per_owner`, norm type 2, global
cap or ordered owner caps, stabilization convention and topology hash. Campaign
caps are 1.0; C3 explicitly records the cap for all five owners.

`ClippingObservation` contains run/attempt/update/width, ordered active flags,
group pre/post norms, applied coefficients/caps, combined pre/post norms and the
global coefficient when applicable. Inactive groups have null values and
`active=false`; active zero-norm groups have coefficient 1. Clipping frequency
uses active observations for the width/group as its denominator.

## Data contract and cursor

`EpochDataContract` pins corpus/tokenizer/roles, stored permutation, available
5,576,491 sequences, designated 5,576,448, excluded tail 43, context 128, batch 64,
accumulation/world size 1, designated-set hash, epoch-order version and seed 42.
Membership is the stored permutation prefix.

`DataCursor` reuses repeat-sampler state: epoch, next batch offset, completed
batches, membership and budget identity. Every 87,132 updates advance an epoch
without resetting streams, membership or clocks.

`UpdateTrace` contains committed step, action ordinal/width, epoch, batch offset,
ordered sample IDs or losslessly reproducible cursor plus order identity, batch
digest, action digest and actual packed tokens. Evaluations separately record
valid causal targets. One compact record per committed update is sufficient;
attempt failures are separate. Resume reconciles non-durable scientific records
before appending replayed updates.

## Logical update and clock

`UpdateAccounting` contains attempts, committed global updates, width selections,
quarter activations, successful owner calls, tokens, complete epochs, cursor and
scheduler position. Attempts across retries are distinguished from committed work.

Transitions: `prepared -> backward_complete -> clipped -> mutation_started ->
owners_complete -> clock_and_accounting_complete -> committed`.
Pre-mutation failures can restore transactional action/data state. Failure after
mutation starts poisons the live state and aborts; no resumable save is permitted.
The next attempt restores the prior durable checkpoint.

`GlobalClock` contains scheduler configuration/horizon, committed position,
current/last-used LRs, scheduler state and existing scalar carrier for collections.
Owner calls and local parameter steps are not its timebase.

## Resumable checkpoint

Campaign payloads include `optimizer_ownership_checkpoint_schema_version=1`
alongside existing outer purpose/version fields, campaign/run contracts, model,
ordered owners/history/topology, clipping, clock, RNG/actions, sampler/epochs/budget,
metrics/accounting, and resource-ledger watermark. Any changed collection layout
also gets an explicit collection schema version; historical readers do not guess.

Only `resumable_training` purpose at a reconciled committed boundary permits exact campaign
resume. Model-only exports cannot satisfy it. Validate all components before
installation. A one-time whole-bundle restore guard covers installation failure.

## Resource attempt

`ResourceAttempt` contains unique attempt ID, run ID, source durable checkpoint
hash/step, start timestamp, latest elapsed duration, peak allocated/reserved GPU
bytes (or separately labeled CPU metric), attempted work, observation sequence,
status and measurement-completeness flag.

The ledger retains the latest measurement per unique attempt. Total runtime is
their sum and peak is their maximum. It remains independent of rollbackable
scientific state; a checkpoint watermark is not an amount to add again.
Unfinalized abrupt-kill attempts retain known costs and an incomplete flag.

`OptimizerStorageMeasurement` contains step, owner/component, dtype, elements,
bytes and separate counter totals. Temporary concatenation storage is reported
separately with measurement/estimate method. Compute bf16 does not set state dtype.

## Endpoint and report

`TerminalValidationResults` (schema 1) binds ordinary-validation results to the
durable terminal checkpoint hash/path, actual and assigned step/tokens/epochs,
role/manifest/protocol, evaluated examples/causal targets, count convention and
ordered width endpoints. Its content hash excludes its own hash field. Evaluation
and checkpoint must describe the same committed model state.

`Endpoint` key is `(campaign_id, arm_id, run_id, width)`. Fields include
representation/ownership/clipping, exact active count, loss/perplexity, evaluation
and checkpoint identities, targets, actual/assigned budgets, initialization/data
provenance, realized exposure and resources. Loss and `exp(loss)` must be finite;
counts agree with the matching standalone and stored definition.

`FrozenCampaignManifest` binds nine unique terminal identities and sidecar hashes
to preflight. `ComparisonReport` contains 24 endpoints, complete/partial status,
explicit omissions if requested, table/figure paths and descriptive interpretation.
Completeness is 5 elastic arms × 4 widths + 4 standalones. No mixed roles, trailing
means, substituted points, across-seed bars or automatic holdout evaluation.

## Correction extension entities

Campaign schema 2 has six corrected concat ArmDefinitions with correction_mode
and reference_arm_id. A correction contract (schema 1) is an additional field in
the new scientific contract only. Existing whole-contract hashes bind checkpoint
compatibility. Submission attempts record campaign/arm/run, source hashes, Slurm
ID/name/state, resources and durable progress. Six corrected frozen runs plus
seven original reference runs supply 40 endpoints without altering either source.
