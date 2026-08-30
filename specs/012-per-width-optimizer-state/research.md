# Research: Per-Width Optimizer State

## Decision 1: Preserve the existing optimizer configuration shape

**Decision**: Accept `training.optimizer.state_scope` and
`training.optimizer.scheduler_clock` as user-facing inputs, default them to
`shared` and `global_step`, then normalize them into an explicit sibling
`training.optimizer_state_contract`. Keep the resolved `training.optimizer`
mapping limited to its historical `{name, kwargs}` shape.

**Rationale**: Optimizer resolution currently rebuilds that exact mapping, and
the comparison-control signature hashes it. Keeping the shape stable preserves
historical shared-run identity while the new contract makes scope and clock
visible in preflight, resolved config, checkpoints, metrics, and summaries.

**Alternatives considered**:

- Retain scope and clock inside the resolved optimizer mapping: rejected because
  it changes existing shared comparison signatures and compatibility checks.
- Add unrelated top-level configuration keys: rejected because the requested
  user-facing form correctly groups the concepts with optimizer configuration.

## Decision 2: Use one focused ordered optimizer collection

**Decision**: Add `src/training/optimizer_state.py` with a small
per-granularity optimizer collection. It constructs one AdamW or SGD instance
per ordered resolved label, with every instance referencing the same model
parameters. The existing single-optimizer construction path remains unchanged
for shared scope.

**Rationale**: The collection owns only the experiment-specific behavior that
would otherwise be repeated across the training loop and checkpoint code:
ordered selection, one-owner validation, learning-rate synchronization,
successful-update counts, last owner, and state serialization. It is not a
generic optimizer registry or training strategy framework.

**Alternatives considered**:

- Copy the model or parameters per width: rejected because weights must remain
  shared and the experiment varies optimizer history only.
- Store loose optimizer mappings directly in the main loop: rejected because
  checkpoint order, validation, and count reconciliation would be scattered.
- Wrap all shared and per-granularity optimizers in a generic runtime framework:
  rejected as unnecessary abstraction; shared runs retain their current path.

## Decision 3: Let gradient presence govern parameter-local state

**Decision**: Every width optimizer is created over the complete shared
parameter set. Gradients are cleared with `set_to_none=True`. During a selected
width step, ordinary AdamW or SGD behavior applies to every parameter whose
gradient is present; parameters outside that width have no gradient and acquire
no state, weight decay, counter change, or parameter update.

**Rationale**: This preserves existing optimizer semantics, naturally gives
shared prefix parameters distinct histories by width, and prevents narrow
steps from touching wider-only parameters. A present numerically zero gradient
remains ordinary optimizer input; only absence denotes inactivity.

**Alternatives considered**:

- Materialize only active parameter subsets for each step: rejected because it
  duplicates model layout logic and makes shared-parameter state harder to
  audit.
- Filter numerically zero gradients: rejected because it changes AdamW/SGD
  semantics and confuses mathematical cancellation with structural inactivity.

## Decision 4: Keep action selection once per accumulation window

**Decision**: Reuse `_select_optimizer_window_action` before the accumulated
forward/backward window. Resolve the one global label from that action, select
its optimizer, and reuse both action and optimizer for every microbatch in the
window.

**Rationale**: `train_for_steps` already selects one action before grouping and
processing all microbatches. The new mode only inserts an explicit owner
selection after the existing action decision; it does not add a second sampler
or change batches, losses, clipping, autocast, correction, or data cursors.

**Alternatives considered**:

- Select an optimizer per microbatch: rejected because accumulated gradients
  would have multiple owners.
- Derive the optimizer from the step number: rejected because adaptive actions
  cannot be reconstructed from the step alone.

## Decision 5: Use one explicit global scheduler clock

**Decision**: Shared scope keeps the existing optimizer-bound scheduler. In
per-granularity scope, use a focused global scheduler-clock object backed by a
non-model scalar carrier solely to run the repository's existing scheduler
implementation. After construction, after each global scheduler step, and
after resume, copy the clock's learning rate to every width optimizer.

**Rationale**: This reuses cosine, constant, and warmup-stable-decay behavior
without reimplementing their mathematics. Binding the scheduler to a real
width optimizer would couple warnings and bookkeeping to whether that width was
selected; one scheduler per width would create redundant or width-local state.
The carrier owns no model parameters or optimizer moments and is not counted as
a width optimizer.

**Alternatives considered**:

- One scheduler per width: rejected because unequal width exposure risks local
  schedule clocks and contradicts the single global scheduler contract.
- Bind one scheduler to the first width optimizer: rejected because scheduler
  step ordering becomes implicitly coupled to that width's selection history.
- Reimplement scheduler formulas as scalar functions: rejected because it
  duplicates supported scheduler behavior and raises drift risk.

## Decision 6: Define the irreversible optimizer return as the commit boundary

**Decision**: The visible update order is: select one action and owner, clear
gradients, run the full accumulation window, clip, validate the common learning
rate, call only the selected optimizer, mark the optimizer commit after that
call returns, advance the global scheduler once, synchronize learning rates,
then increment the selected update count and existing sampling exposure.

**Rationale**: Existing code already snapshots RNG, run/controller state, and
the packed-data cursor and restores them for pre-commit failures. Snapshotting
all model and optimizer tensors every step would be prohibitively expensive.
A failure after the optimizer returns is classified as post-commit fatal; no
new checkpoint is advertised as reconciled, and recovery uses the previous
durable checkpoint.

**Alternatives considered**:

- Treat optimizer and scheduler calls as one rollback transaction: rejected
  because full per-step tensor snapshots would invalidate experiment cost.
- Count the action before optimizer return: rejected because failed attempts
  would inflate exposure and optimizer histories.

## Decision 7: Reuse existing global sampling state and RNG

**Decision**: Add no optimizer-specific random stream. Reuse existing fixed,
random, balanced-cycle, and adaptive global action state. Persist optimizer
update counts separately and require them to match the committed global action
exposures for every label.

**Rationale**: Global action selection is already deterministic, versioned, and
checkpointed, including balanced-cycle permutation, position, progress, and
exposures. Optimizer ownership is a deterministic consequence of the selected
action and must not alter paired action traces.

**Alternatives considered**:

- Maintain another ownership RNG: rejected as an unintended paired-arm
  difference.
- Replace balanced-cycle state: rejected because existing tests already prove
  exact equal exposure, rollback, and boundary resume.

## Decision 8: Add a versioned, purpose-aware checkpoint contract

**Decision**: Resumable checkpoints include explicit checkpoint purpose and an
optimizer-state contract containing schema version, scope, global clock,
ordered labels, optimizer and scheduler contracts, topology identity, current
learning rate, per-width counts, last owner, total commits, and an ordered list
of complete width optimizer states. Shared checkpoints retain the historical
single `optimizer_state_dict`. Missing scope metadata is interpreted as shared
only.

**Rationale**: Count or mapping inference cannot detect reordered, missing, or
fabricated states. Complete compatibility must be validated before loading the
model, optimizer, scheduler, or RNG. Model-only best-evaluation checkpoints
remain valid for evaluation but are explicitly rejected for training resume.

**Alternatives considered**:

- Clone a historical shared state into every width: rejected because it
  fabricates independent histories.
- Infer resumability from the presence of optimizer data: rejected because
  explicit artifact purpose is safer and easier to report.
- Store an unordered mapping only: rejected because label order is part of the
  scientific and resume identity.

## Decision 9: Keep paired controls separate from intervention identity

**Decision**: The full run/checkpoint identity includes optimizer scope. The
arm-invariant paired-control signature explicitly hashes optimizer name and
kwargs but excludes only state scope. Paired validation requires that signature
to match while requiring one `shared` and one `per_granularity` arm and checking
action, learning-rate, batch/data-role, budget, evaluation, and terminal-rule
identity directly.

**Rationale**: A single signature cannot simultaneously distinguish resumable
state scope and certify that scope is the only intended experimental
difference. The explicit split makes both purposes auditable.

**Alternatives considered**:

- Include scope in the paired-control signature: rejected because valid paired
  arms would never match.
- Remove scope from all identities: rejected because cross-scope resume would
  become unsafe.

## Decision 10: Extend existing compact artifacts

**Decision**: Add optimizer scope, scheduler clock, active owner, collection
version, and cumulative width update counts to ordinary metric rows. Add the
same final accounting plus training wall time, peak accelerator memory, and
resumable checkpoint path, size, and hash to run summaries. Full optimizer
states remain only in resumable checkpoints.

**Rationale**: The current metrics already record selected granularity,
learning rate, scheduler position, wall time, peak memory, and sampling
exposures. Small ownership fields complete the audit contract without writing
large state digests per step.

**Alternatives considered**:

- Store optimizer-moment digests on every row: rejected because of artifact
  volume and limited diagnostic value.
- Calculate all resource values only in the comparison script: rejected because
  each run must be independently auditable.

## Decision 11: Freeze and analyze the paired experiment explicitly

**Decision**: Add the frozen controlled recipe expected by the runbook and a
dedicated `scripts/analyze_tinystories_per_width_optimizer.py` with `freeze` and
`report` phases. `freeze` validates six completed runs and records terminal
checkpoint hashes, controls, endpoints, and holdout status. `report` validates
final-holdout artifacts when present and emits paired JSON/CSV outcomes with an
explicit diagnostic, confirmatory, or descriptive claim status.

**Rationale**: Existing final-holdout evaluation already produces per-width
loss, perplexity, a uniform average, checkpoint hash, and role provenance. The
portfolio analyzer assumes five checkpoints per seed and cannot represent this
two-arm terminal-checkpoint comparison cleanly.

**Alternatives considered**:

- Extend the portfolio catch-up analyzer: rejected because its selection and
  checkpoint cardinality contracts are scientifically different.
- Depend on manual tables only: rejected because the spec requires structured,
  reproducible comparison inputs and outputs.

## Decision 12: Limit the initial feature to one process

**Decision**: Reject per-granularity scope unless distributed strategy is
disabled and effective and expected world size are one. Shared scope retains
all existing distributed behavior.

**Rationale**: The clarified feature is single-process only, and its frozen
recipe uses one process. Early rejection avoids an unverified optimizer
collection checkpoint path under sharding while preserving current distributed
training unchanged.

**Alternatives considered**:

- Conditionally enable FSDP during planning: rejected by the clarification.
- Reject distributed training globally: rejected because only the opt-in
  per-granularity mode is restricted.

## Resolved Technical Baseline

- Python 3.12, PyTorch 2.11, Transformers 5.8, PyYAML, NumPy, pandas, and pytest.
- Existing YAML/dotted override resolution, structured CSV/JSON artifacts, and
  PyTorch checkpoint files.
- Existing prepared TinyStories-Instruct four-role corpus and deterministic
  repeat-epoch iteration.
- Existing balanced-cycle H=1 action state, post-training final-holdout
  evaluator, monitoring, and reporting surfaces.
- No unresolved `NEEDS CLARIFICATION` items remain.
