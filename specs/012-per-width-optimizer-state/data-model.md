# Data Model: Per-Width Optimizer State

## 1. Resolved Optimizer State Contract

Represents the normalized experiment choice used by runtime, checkpointing,
metrics, and reporting.

### Fields

- `schema_version`: positive integer; initial version is `1`.
- `state_scope`: `shared` or `per_granularity`.
- `scheduler_clock`: fixed value `global_step`.
- `ordered_granularities`: ordered, unique global labels.
- `optimizer_name`: `adamw` or `sgd`.
- `optimizer_kwargs`: fully resolved hyperparameters.
- `base_learning_rate`: configured base value.
- `resolved_learning_rate`: learning rate after existing scale rules.
- `scheduler_contract`: resolved global scheduler identity and horizon.
- `single_process_required`: true only for `per_granularity`.
- `topology_identity`: stable model family, variant, dimensions, and resolved
  granularity-layout identity used for resume checks.

### Validation

- Missing input scope resolves to `shared`; missing clock resolves to
  `global_step`.
- Per-granularity scope requires at least two unique resolved global labels and
  exactly one global label per optimizer step.
- Per-granularity scope requires one effective process and rejects standalone,
  nested-all, per-block, and adaptive-per-block modes.
- User-facing scope and clock inputs are consumed during resolution; the
  historical resolved `training.optimizer` mapping remains `{name, kwargs}`.

## 2. Per-Granularity Optimizer Collection

Runtime owner of the width-local optimizer state machines. All entries reference
the same model parameters.

### Fields

- `schema_version`: collection state version, initially `1`.
- `contract`: the resolved optimizer state contract.
- `entries`: ordered list of width optimizer entries.
- `successful_update_counts`: mapping with exactly one nonnegative integer per
  ordered label.
- `total_successful_updates`: sum of all width counts.
- `last_active_granularity`: null before the first commit, then an ordered label.
- `current_learning_rates`: ordered learning rates shared by every entry.

### Relationships

- Contains exactly one **Width Optimizer Entry** per resolved label.
- Uses one **Global Scheduler Clock** shared by the collection.
- Counts must reconcile with **Global Sampling State** and the durable metrics
  journal.

### Invariants

- Every entry points at the same ordered model parameter objects; no model
  parameter is copied.
- Only the selected entry may change model-related optimizer state during one
  commit.
- All entries have identical current learning rates after construction, commit,
  checkpoint save, and resume.
- Non-selected entry moments and counters are bitwise unchanged; synchronized
  param-group learning rate is the intentional global-clock exception.

## 3. Width Optimizer Entry

One optimizer state machine associated with one global granularity.

### Fields

- `granularity`: unique label; position must match the resolved ordered labels.
- `optimizer_state_dict`: complete parameter groups and parameter-local state.
- `successful_updates`: nonnegative count for this owner.
- `state_digest`: optional checkpoint-time integrity digest.

### Parameter-state rules

- Shared-prefix parameters may have different moments and counters in every
  entry.
- Wider-only parameters remain absent from a narrower entry's parameter-local
  state until that parameter receives a gradient during a step owned by that
  entry.
- A parameter with an absent gradient is not updated or decayed.
- A present zero-valued gradient follows ordinary AdamW or SGD semantics.
- All tensor state values must be finite when a resumable checkpoint is loaded.

## 4. Global Scheduler Clock

The single schedule position applied to shared and per-granularity modes.

### Fields

- `clock`: fixed value `global_step`.
- `scheduler_name`: resolved scheduler family.
- `scheduler_contract`: warmup, horizon, and scheduler-specific inputs.
- `state_dict`: one scheduler state.
- `position`: committed global optimizer-step position.
- `current_learning_rates`: rates prepared for the next commit.
- `last_committed_learning_rate`: rate used by the most recent commit.

### Invariants

- Position advances exactly once after a selected optimizer returns
  successfully.
- Width exposure does not affect schedule position.
- Every width optimizer receives the same current rate.
- Matched shared and per-granularity runs use the same committed rate at every
  global step.

## 5. Optimizer Commit Accounting

Compact durable state describing optimizer ownership without serializing
moments into ordinary metrics.

### Fields

- `pending_step`: next global step being attempted.
- `selected_granularity`: sole optimizer owner for the accumulation window.
- `optimizer_state_scope`: resolved state scope.
- `scheduler_clock`: `global_step`.
- `committed`: whether the selected optimizer returned successfully and the
  global schedule/accounting block completed.
- `successful_update_counts`: cumulative counts after a commit.
- `total_successful_updates`: cumulative global commits.
- `global_sampling_exposure_counts`: existing action exposures.
- `scheduler_position`: global schedule position associated with the row.
- `failure_stage`: null for a commit, otherwise the attributable pre-commit or
  post-commit stage.

### Reconciliation

- For every completed run,
  `sum(successful_update_counts) == total_successful_updates == steps_completed`.
- Each label's update count equals its global action exposure.
- A pre-commit failure changes none of these values.
- A post-commit fatal failure cannot create a newly advertised reconciled
  resumable checkpoint; recovery uses the prior durable boundary.

## 6. Resumable Training Checkpoint

A checkpoint whose purpose explicitly allows exact training continuation.

### Fields

- `checkpoint_kind`: `resumable_training`.
- `checkpoint_schema_version`: existing checkpoint format identity plus new
  optimizer ownership version.
- `optimizer_state_contract`: complete resolved scope/clock/optimizer/topology
  identity.
- `optimizer_state_dict`: historical single state for shared scope.
- `optimizer_state_collection`: complete per-granularity collection for
  per-granularity scope; absent for shared scope.
- `scheduler_state_dict`: one global scheduler state.
- `global_sampling_state`: existing action RNG/schedule/exposure continuation.
- `run_state`: committed steps, tokens, data cursor, metrics journal, warmup,
  controllers, and artifact continuation.
- `current_learning_rates`: saved rates used to resynchronize on load.

### Load validation order

1. Verify checkpoint purpose before any runtime mutation.
2. Interpret missing optimizer scope as historical shared state only.
3. Validate schema, exact scope/clock, optimizer family and kwargs, scheduler
   contract, topology identity, ordered labels, state count/order, finiteness,
   and accounting.
4. Validate data, sampling, controller, RNG, journal, and existing
   reproducibility contracts.
5. Load model and optimizer state.
6. Load the single scheduler state and resynchronize all learning rates.
7. Recheck scheduler position, counts, and exposures before training.

### Model-only checkpoint

A best-evaluation artifact has `checkpoint_kind: model_only_evaluation`, omits
complete optimizer/scheduler continuation state, and is never accepted by the
training resume path.

## 7. Optimizer Ownership Metric Fields

Fields added to each ordinary training row:

- `optimizer_state_scope`
- `optimizer_scheduler_clock`
- `optimizer_owner_granularity`
- `optimizer_collection_schema_version`
- `optimizer_total_successful_updates`
- `optimizer_update_counts`

Existing selected granularity, learning rate, scheduler position, action window,
exposure, wall time, and peak memory fields remain authoritative and are not
duplicated.

## 8. Optimizer Ownership Run Summary

Completion and failure summary extension.

### Fields

- resolved scope/clock/collection version;
- ordered granularities and final update counts;
- total successful optimizer updates and final global scheduler position;
- reconciliation status and any failure stage;
- terminal training wall-clock seconds;
- peak accelerator memory bytes;
- resumable checkpoint path, SHA-256, and installed byte size;
- full run identity and arm-invariant paired-control signature.

## 9. Paired Comparison Manifest

Frozen input to the paired analysis and holdout workflow.

### Fields

- `schema_version`
- `experiment_phase`: `pilot` or `confirmation`.
- `claim_status`: `diagnostic`, `confirmatory`, or
  `descriptive_after_holdout_open`.
- `holdout_status`: sealed/opened state and timestamp/provenance.
- `paired_control_signature`: scope-invariant control identity.
- `terminal_checkpoint_rule`: exact terminal fixed-budget checkpoint.
- `budget_tokens` and `global_steps`.
- `primary_endpoint` and ordered `secondary_endpoints`.
- `seeds`: exactly 42, 43, and 44 for the frozen experiment.
- `pairs`: one shared and one per-granularity run per seed.
- Per run: run directory, scope, resolved config hash, role hashes, action-trace
  hash, learning-rate-trace hash, terminal checkpoint path/hash/size, committed
  steps, per-width counts, wall time, and peak memory.
- `matched_compute_claim`: false unless a separately validated compute contract
  permits it.

### Validation

- Exactly one arm of each scope exists per required seed.
- Only state scope differs between paired resolved scientific controls.
- Actions, batches/data identities, learning rates, budgets, cadence, ordered
  widths, and terminal rules match within each seed pair.
- Pilot manifests are diagnostic.
- Confirmation is claim-eligible only with all three seeds and a holdout that
  remained sealed through the pilot decision.

## 10. Paired Comparison Result

Structured JSON and CSV output derived from the frozen manifest and, when
authorized, final-holdout results.

### Fields

- per-seed candidate-minus-shared uniform mean final-holdout loss;
- per-width final-holdout loss and perplexity for both arms and paired
  differences;
- worst-width final-holdout loss;
- trailing-five ordinary-validation means;
- matched-token convergence observations;
- optimizer counts and reconciliation status;
- wall time, peak memory, checkpoint size, and paired cost deltas;
- aggregate descriptive summaries across seeds;
- claim status, holdout status, provenance hashes, and matched-compute flag.
