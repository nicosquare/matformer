# Optimizer Lifecycle and Checkpoint Contract

## Construction

### Shared scope

Use the existing optimizer and scheduler construction unchanged. One AdamW or
SGD owns all updates and one scheduler is bound to it.

### Per-granularity scope

1. Read the ordered labels from the resolved optimizer state contract.
2. Construct one optimizer per label with identical family, kwargs, resolved
   learning rate, and references to the same ordered model parameters.
3. Construct one global scheduler clock using the existing scheduler family and
   resolved horizon.
4. Copy the clock's initialized learning rates to every width optimizer.
5. Initialize every successful-update count to zero and last owner to null.

The scheduler clock's scalar carrier is not a model optimizer, owns no model
parameter, and is excluded from width-optimizer counts and checkpoints. Only
its scheduler state, position, and current rates are persisted.

## Explicit Optimizer-Window Order

For every accumulation window:

1. Snapshot existing action RNG, controller, run, journal, and data-cursor state.
2. Select one action through the existing global action path.
3. Require exactly one ordered global label and select its width optimizer.
4. Clear shared model gradients with absent-gradient semantics.
5. Run every microbatch in the window with the same action.
6. Aggregate target-weighted gradients through the existing backward path.
7. Apply existing clipping and correction semantics.
8. Verify that all width optimizer param groups agree with the global current
   learning rate and capture the rate used by this commit.
9. Call only the selected width optimizer's `step`.
10. Mark the optimizer commit irreversible after the call returns.
11. Advance the global scheduler clock exactly once and synchronize the next
    rate to all width optimizers.
12. Increment only the selected width update count.
13. Commit existing global action exposure, data cursor, tokens, and metrics.
14. Clear gradients before the next window or on exit.

Shared scope follows the existing equivalent order without collection-specific
selection or synchronization.

## Gradient and Parameter Semantics

- All width optimizers reference the same model tensors.
- A parameter active for the selected width receives the ordinary accumulated
  gradient and ordinary AdamW/SGD semantics, including the present-zero case.
- A parameter inactive for the selected width has `grad is None` and receives
  no parameter update, weight decay, state creation, moment change, or counter
  change.
- Non-selected optimizer parameter-local states are bitwise unchanged.
- The only allowed non-selected param-group change is the globally synchronized
  learning rate.
- Gradients are never retained for reuse by another owner.

## Warmup

Existing `full_only` and `balanced_global` pre-nested warmups use the normal
training loop with one forced global label per optimizer step. The forced label
owns that step and its width optimizer advances. Warmup still advances the one
global scheduler. Any future warmup action that contains zero or multiple
global labels is rejected for per-granularity scope.

## Failure Semantics

### Pre-commit failure

If failure occurs before the selected optimizer returns successfully:

- restore existing RNG, action/controller state, run state, and packed-data
  cursor snapshots;
- leave every optimizer state and the global scheduler unchanged;
- increment no update count or exposure;
- clear gradients;
- record the selected label, failing stage, and `committed: false`.

### Post-commit fatal failure

If the selected optimizer returned but scheduler synchronization or required
accounting fails:

- do not pretend the step was rolled back;
- stop the run with explicit `post_commit_fatal` provenance;
- do not write a new checkpoint advertised as reconciled or resumable;
- retain the previous durable latest checkpoint for deterministic recovery.

The implementation prevalidates scheduler and accounting structures before the
optimizer call so the post-commit path contains only small deterministic
operations. Full per-step model/optimizer snapshots are explicitly excluded.

## Completion Reconciliation

Before a completed summary or terminal resumable checkpoint is accepted:

- total width updates equal committed global steps;
- per-width updates equal existing per-width global action exposures;
- scheduler position equals committed global steps;
- all width optimizer learning rates equal the global clock's rates;
- balanced-cycle completion satisfies its existing exact exposure rules;
- no optimizer window or ownership commit is in flight.

## Checkpoint Purpose

Every checkpoint has one explicit purpose:

- `resumable_training`: latest, continuation, terminal fixed-budget, and other
  checkpoints that contain complete continuation state;
- `model_only_evaluation`: best-evaluation artifacts that intentionally omit
  optimizer/scheduler continuation.

Training resume rejects `model_only_evaluation` before loading the model.

## Optimizer State Payload

Shared resumable checkpoints retain the historical `optimizer_state_dict` and
add explicit ownership metadata. Per-granularity resumable checkpoints contain:

```text
optimizer_state_contract:
  schema_version: 1
  state_scope: per_granularity
  scheduler_clock: global_step
  ordered_granularities: [...]
  optimizer_name: ...
  optimizer_kwargs: ...
  scheduler_contract: ...
  topology_identity: ...

optimizer_state_collection:
  schema_version: 1
  ordered_entries:
    - granularity: g250
      state_dict: ...
    - granularity: g500
      state_dict: ...
  successful_update_counts: ...
  total_successful_updates: ...
  last_active_granularity: ...
  current_learning_rates: ...

scheduler_state_dict: ...
global_sampling_state: ...
```

The ordered entry sequence, not just label membership, is authoritative.

## Historical Compatibility

- A checkpoint without optimizer scope metadata means historical `shared`.
- Historical shared checkpoints resume only into resolved shared scope.
- Shared state is never cloned into per-granularity entries.
- Per-granularity states are never merged into shared state.
- Shared checkpoints and shared runtime behavior retain their existing
  distributed and compatibility paths.

## Mutation-Free Resume Validation

Before loading any model, optimizer, scheduler, RNG, or data-cursor state:

1. Require `resumable_training` purpose, treating only compatible historical
   continuation artifacts as resumable shared checkpoints.
2. Validate scope, collection version, scheduler clock, optimizer family and
   normalized kwargs, scheduler contract, topology identity, and ordered labels.
3. Require exactly one unique ordered optimizer entry per label.
4. Validate parameter-group structure and recursively reject non-finite tensor
   or scalar optimizer values.
5. Validate nonnegative counts, total count, global steps, schedule position,
   and per-label exposure reconciliation.
6. Validate all existing configuration, model, dataset role, sampler, RNG,
   journal, warmup, and controller continuation contracts.
7. Only then load model and optimizer states, load the global scheduler state,
   install its saved current rates on every optimizer, and recheck invariants.

Any missing, extra, duplicate, reordered, malformed, non-finite, or mismatched
state fails without partial loading or automatic repair.

## Exact Resume Scenarios

- Inside a random or balanced global interval.
- Exactly at a balanced-cycle boundary.
- During either supported one-owner pre-nested warmup.
- Before the first use of a wider-only parameter in one optimizer.
- With AdamW and SGD.

Under existing deterministic runtime conditions, resumed execution must match
subsequent actions exactly and match batches, learning rates, optimizer states,
scheduler state, metrics, counts, and final parameters within the repository's
established tolerance.
