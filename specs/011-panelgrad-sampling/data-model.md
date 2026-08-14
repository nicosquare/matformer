# Data Model: PanelGrad Sampling

## Entities

### PanelGradConfiguration

Resolved scientific inputs and fixed method identity.

- **method_family**: `panelgrad_gradient_rms`
- **method_version**: `1`
- **strategy**: `panelgrad`
- **scope**: `global`
- **ordered_granularities**: nonempty, unique resolved labels
- **refresh_interval_steps**: positive integer `H`, default `50`
- **eta**: finite positive scalar, default `1e-12`
- **temperature**: finite positive scalar `T`, default `1.0`
- **epsilon**: finite scalar in `[0,1]`, default `0.1`
- **relative_tolerance**: fixed `1e-6`
- **absolute_tolerance**: fixed `1e-8`
- **score_definition**: raw aggregate-controller-gradient RMS over controlled FFN support
- **probability_definition**: powered normalized score plus uniform mixture
- **controller_panel_contract**: fixed existing 128-example controller role
- **final_holdout_contract**: fixed existing 512-example training-inaccessible role

Validation rules:

- PanelGrad is valid only with `adaptive_global` and `nested-random`.
- Any supplied fixed identity, scope, score, or tolerance field must match this contract.
- The four configurable values resolve before model construction and are saved.
- Thompson posterior/reset inputs, UCB controls, per-block scope, inverse weighting, and cost correction are incompatible.

### ControlledFFNSupport

Defines the exact parameter coordinates measured for one global granularity.

- **granularity**: resolved label
- **variant**: `slicing` or `concat`
- **layer_count**: number of controlled transformer FFN layers
- **prefix_widths or selected_block_counts**: resolved layout for every layer
- **included_parameter_families**: gate weight, up weight, down weight, and selected gate/up biases when present
- **excluded_parameter_families**: shared down bias, embeddings, attention, normalization, language-model head, and every other granularity-independent parameter
- **unique_trainable_scalar_count**: `N_g`
- **support_schema_version/hash**: compatibility identity derived from variant, layout, trainability, and resolved granularities

Rules:

- `N_g` is determined from resolved model layout, not observed gradient sparsity.
- Selected zero-valued gradients remain in the count.
- Shared parameter storage is counted once.
- The support is resolved before distributed wrapping and must match the support used to extract gradients.

### ControllerDataIdentity

References the existing four-role partition.

- **data_roles_manifest_hash**
- **optimizer_training_manifest_hash**
- **controller_manifest_hash**
- **ordinary_validation_manifest_hash**
- **final_holdout_manifest_hash**
- **controller_example_count**: `128`
- **controller_target_count**: valid causal targets used by a refresh
- **selection seeds and source provenance**: inherited from existing role manifests

The four roles remain pairwise disjoint. PanelGrad consumes only the controller role for decisions and never consumes the final holdout during training.

### GranularityMeasurement

One completed measurement for one granularity at one refresh.

- **granularity**
- **controlled_parameter_count**: `N_g`
- **gradient_squared_norm**: float64 sum over the controlled support
- **gradient_norm**: `sqrt(gradient_squared_norm)`
- **gradient_rms_score**: `I_g = gradient_norm / sqrt(N_g)`
- **aggregate_controller_loss**: finite target-token-weighted loss used for differentiation
- **evaluation_examples/target_tokens/batches**
- **microbatch_aggregation**: `target_token_weighted_aggregate_gradient`
- **gradient_semantics**: raw, pre-membership-correction, pre-clipping, pre-optimizer
- **distributed_semantics**: full global FFN support under synchronized FSDP backward

Validation rules:

- Loss, squared norm, norm, and score are finite and nonnegative.
- The controller example and target totals are consistent across granularities.
- Every configured granularity appears exactly once in resolved order.

### ProbabilitySnapshot

The complete contemporaneous policy state from one refresh.

- **refresh_index**
- **boundary_step**: completed optimizer step before the next action
- **measurements**: ordered `GranularityMeasurement` list
- **q**: ordered normalized powered-score vector
- **p**: ordered final categorical vector
- **entropy**, **minimum_probability**, **maximum_probability**
- **eta**, **temperature**, **epsilon**
- **measurement_duration_seconds**, **controller_backward_count**, and cumulative cost
- **controller/support/config hashes**

Rules:

- `q` and `p` align exactly with `ordered_granularities`.
- Values are finite and nonnegative and sums equal one within `rtol=1e-6`, `atol=1e-8`.
- A snapshot becomes active only after every granularity measurement and probability check succeeds.
- The snapshot remains immutable for its interval.

### CategoricalSamplingState

Owns only PanelGrad action randomness and exposure accounting.

- **seed_stream_name**: `panelgrad_sampling`
- **resolved_seed**
- **generator_state**: serializable CPU generator state
- **sample_count**
- **exposure_counts**: one nonnegative count per granularity
- **last_sampled_granularity**
- **last_sampled_probability**
- **pending_sample**: ephemeral draw awaiting optimizer commit, never counted as exposure

Resume rule:

- Restoring the same snapshot and generator state reproduces the next categorical action exactly.
- A failed optimizer attempt restores the pre-draw generator/sample state and does not increment exposure.

### PanelGradLifecycleState

Tracks refresh ownership and exact continuation.

- **phase**: lifecycle state below
- **refresh_index**
- **last_refresh_step**
- **next_refresh_step**
- **completed_steps_since_refresh**: integer in `[0,H]`
- **active_probability_snapshot**: required while an interval is active
- **sampling_state**
- **last_committed_action**
- **terminal_reason/progress**
- **failure stage/message**, when failed

Lifecycle states:

1. `initial_refresh_pending`: no PanelGrad action or exposure exists; optional warmup may run without changing this state.
2. `active_interval`: a complete probability snapshot exists and progress is in `[0,H-1]`; the next step samples from its `p`.
3. `refresh_pending`: exactly `H` successful PanelGrad steps have completed; no next action may be sampled until refresh succeeds.
4. `terminal_partial`: training ended with progress in `[1,H-1]`; no extra refresh occurs.
5. `terminal_complete`: training ended at progress `H` (or with no remaining budget at an initial boundary); no unused refresh occurs.
6. `failed`: refresh, probability validation, distributed agreement, journal commit, or state validation failed; no fallback action is allowed.

Transitions:

- `initial_refresh_pending -> active_interval` after the first complete refresh, which occurs only after optional warmup and only if training continues.
- `active_interval -> active_interval` after each successful step with progress `<H`.
- `active_interval -> refresh_pending` after the `H`th successful step.
- `refresh_pending -> active_interval` after a complete refresh before the next action.
- Any live phase may become its appropriate terminal phase when the run budget is exhausted.
- Refresh failure enters `failed` without partially replacing the last complete snapshot.

### PanelGradAction

One action drawn for one optimizer window.

- **kind**: `global`
- **granularity**: one resolved label
- **probability**: corresponding element of active `p`
- **refresh_index and boundary_step**
- **sample_index**

The internal training action remains `{"kind":"global","granularities":[g]}` and is reused unchanged across all gradient-accumulation microsteps for that optimizer update.

### PanelGradRefreshEvent

Append-only audit record for a refresh or refresh failure.

- **schema/method version and run identity**
- **event_type**: `panelgrad_refresh_completed` or `panelgrad_refresh_failed`
- **boundary/refresh/interval state**
- **ordered measurements, q, p, and probability diagnostics**
- **controller/support/config hashes**
- **measurement cost and cumulative cost**
- **resume provenance**
- **failure stage/message and previous valid snapshot hash**, for failures

Completed events are committed transactionally with the active snapshot. A failure event states that no new distribution or action was committed.

### PanelGradRunSummary

Aggregated audit state for a completed or failed run.

- method/config/data/support provenance
- refresh count and final probability snapshot
- per-granularity exposure counts/fractions
- probability entropy/minimum/maximum history summaries
- total controller examples, target tokens, backward evaluations, duration, and refresh cost
- warmup policy/progress and first adaptive boundary
- terminal interval progress/status
- resume count/source and journal path/hash
- failure summary, when applicable

## Relationships

- One `PanelGradConfiguration`, one `ControllerDataIdentity`, and one controlled support per granularity resolve before adaptive training.
- Each completed refresh owns exactly one measurement per granularity and produces one immutable `ProbabilitySnapshot`.
- One `PanelGradLifecycleState` references the active snapshot and one `CategoricalSamplingState`.
- Every successful adaptive optimizer step commits one `PanelGradAction` and increments exactly one exposure count.
- `PanelGradRefreshEvent` records snapshot transitions; ordinary metrics record actions; `PanelGradRunSummary` aggregates both without replacing them.

## Persisted Artifacts

- `config.json`: resolved PanelGrad identity/defaults, support counts/hash, seeds, and role hashes
- existing four role manifests and `data_roles_manifest.json`
- `controller_metrics.jsonl`: PanelGrad refresh, warmup, terminal, and failure events keyed by explicit family/version
- `controller_summary.json`: `PanelGradRunSummary`
- `metrics.csv`: compact current action/probability, exposure, refresh, interval, and method fields
- `run_summary.json`: controller artifact paths/hashes, final summary, and cumulative measurement cost
- checkpoints: complete `panelgrad_state` beside existing model/optimizer/scheduler/training/RNG state
