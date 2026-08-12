# Data Model: Probabilistic Adaptive Granularity

## Entities

### BayesianAdaptiveConfiguration

Represents the resolved research inputs for a genuine Bayesian Thompson run.

- **method_family**: stable discriminator for the Gaussian Bayesian controller
- **method_version**: controller-state and artifact schema version
- **strategy**: `thompson`
- **scope**: `global` or `per_block`, derived from the explicit sampling mode
- **feature_model**: `arms` for global or `additive` for per-block
- **context_model**: `intercept_only`
- **transition_model**: `identity`
- **decision_interval_steps**: positive optimizer-step count; defaults to 50
- **prior_mean_input**: finite scalar or coefficient-length vector
- **prior_covariance_input**: finite nonnegative scalar, diagonal vector, or dense covariance
- **observation_noise_variance**: finite strictly positive scalar
- **process_noise_covariance_input**: finite nonnegative scalar, diagonal vector, or dense covariance
- **compute_weight**: resolved constant `0.0`
- **switch_weight**: resolved constant `0.0`
- **controller_panel_contract**: fixed 128-example, uniform-objective data role
- **final_holdout_contract**: fixed 512-example, training-inaccessible data role

Validation rules:

- Every required probabilistic and data-role input must be present before training.
- The decision interval must be a positive integer.
- Mean values must be finite.
- Covariance inputs must resolve to finite symmetric positive-semidefinite matrices of the feature dimension.
- Observation-noise variance must be finite and greater than zero.
- Controller examples, objective weighting, fixed-manifest flags, final-holdout examples, and training-evaluation policy must match the fixed experiment contract.
- Legacy heuristic exploration, decay, or reward-penalty inputs cannot substitute for Bayesian inputs.

### AdaptiveExperimentIdentity

Represents the method identity used to prevent historical/new run conflation.

- **requested_sampling_mode**: original user-facing mode
- **resolved_sampling_mode**: `adaptive_global` or `adaptive_per_block` for Bayesian runs
- **resolved_scope**: `global` or `per_block`
- **resolved_strategy**: `thompson` or retained `ucb`
- **method_family**: Bayesian controller, legacy heuristic UCB, legacy heuristic Thompson, random, nested-all, or standalone
- **method_version**: present for new Bayesian artifacts; absent on historical heuristic artifacts
- **classification_source**: explicit provenance fields or historical fallback rule

Identity rules:

- A new run with `strategy=thompson` must have the Bayesian method discriminator and version.
- A historical adaptive-per-block artifact with `strategy=thompson` and no Bayesian discriminator is classified as legacy heuristic Thompson.
- `adaptive_per_block + ucb` retains its existing identity and labels.
- No historical artifact is rewritten to acquire new provenance.

### StableExampleIdentity

Represents one usable tokenized example independently of its assigned role.

- **dataset_name**: configured dataset source
- **dataset_config_name**: optional source configuration
- **source_split**: configured source split
- **source_dataset_fingerprint**: source identity before role assignment
- **source_row_identity**: stable original row index or equivalent source identifier
- **tokenization_identity**: tokenizer, context length, padding/truncation policy, and preprocessing version
- **valid_target_count**: number of causal prediction targets after masking

Uniqueness rule:

- The combination of source fingerprint, source row identity, and tokenization identity uniquely identifies a role-eligible example.

### DataRoleManifest

Represents the selected examples and provenance for one of four disjoint roles.

- **role**: `optimizer_training`, `controller`, `ordinary_validation`, or `final_holdout`
- **source_provenance**: dataset and preprocessing identity shared by all roles
- **selection_seed**: resolved named seed for the role, or training remainder provenance
- **selection_algorithm**: deterministic without-replacement algorithm and version
- **ordered_example_identities**: stable identities in deterministic consumption order
- **example_count**: number of selected identities
- **example_identity_hash**: stable hash of ordered identities
- **manifest_hash**: stable hash of the complete manifest
- **usable_target_policy**: rule requiring causal prediction targets

Validation rules:

- Controller count is exactly 128.
- Final-holdout count is exactly 512.
- Ordinary-validation count matches its existing configured request.
- Optimizer training contains at least one usable example.
- Pairwise identity intersections among all role manifests are empty.

### DataRolePartition

Represents the complete four-way split and its audit proof.

- **split_version**: deterministic partition contract version
- **source_pool_hash**: stable hash of the usable source identity pool
- **ordinary_validation_manifest_hash**: ordinary-validation role link
- **controller_manifest_hash**: controller role link
- **final_holdout_manifest_hash**: final role link
- **optimizer_training_manifest_hash**: training role link
- **parent_manifest_hash**: stable hash of the full split description
- **pairwise_intersection_counts**: six role-pair counts, all required to be zero
- **selection_order**: ordinary validation, controller, final holdout, training remainder

### ActionFeatureSchema

Represents the deterministic mapping from an action to the Gaussian coefficient space.

- **scope**: `global` or `per_block`
- **ordered_granularities**: complete resolved label list
- **block_count**: positive transformer-block count
- **encoding**: intercept plus deterministic sum-to-zero contrasts
- **contrast_basis**: saved basis derived from the ordered labels
- **coefficient_names**: ordered, human-readable coefficient identities
- **dimension**: `|G|` for global or `1 + B(|G|-1)` for per-block
- **tie_order**: lexicographic action order induced by block order and resolved granularity order
- **schema_hash**: stable hash used for resume compatibility

Validation rules:

- Granularity labels are nonempty, ordered, and unique.
- One granularity yields an intercept-only dimension of one.
- Per-block schema supports one or more blocks.
- The contrast basis is finite, deterministic, full-rank in the sum-to-zero subspace, and identical after resume.

### AdaptiveAction

Represents the action held fixed throughout one decision window.

- **scope**: `global` or `per_block`
- **global_granularity**: populated only for global scope
- **block_granularities**: ordered length-`B` profile for per-block scope
- **feature_vector**: action encoded by the `ActionFeatureSchema`
- **sampled_predicted_reward**: sampled linear reward used for selection
- **tie_resolution**: whether and how deterministic tie-breaking was applied
- **selection_round**: controller round that owns the action

Validation rules:

- Every selected label belongs to the resolved ordered granularity set.
- Global action repeats one label for every block.
- Per-block action has exactly one label per transformer block.
- The feature-vector dimension and schema hash match controller state.

### GaussianBeliefState

Represents the explicit reward-model belief at a controller round.

- **round_index**: nonnegative observation/selection round
- **posterior_mean**: finite coefficient vector after the latest completed observation
- **posterior_covariance**: finite symmetric positive-semidefinite covariance after conditioning
- **predictive_mean**: finite mean after identity transition and before observation
- **predictive_covariance**: posterior covariance plus process covariance
- **observation_noise_variance**: saved scalar
- **process_noise_covariance**: saved resolved matrix
- **transition_matrix_provenance**: explicit identity transition
- **feature_schema_hash**: compatibility link
- **last_prediction_step**: boundary step where the predictive belief was formed
- **last_update_step**: boundary step of the latest completed observation, if any

State rules:

- Prediction leaves the mean unchanged and adds process covariance.
- Conditioning uses exactly one completed window observation.
- A failed or incomplete window cannot change posterior state.
- Non-finite or invalid belief state cannot be committed or checkpointed as valid.

### PosteriorSamplingState

Represents randomness dedicated only to coefficient sampling.

- **seed_stream_name**: stable controller sampling stream
- **resolved_seed**: deterministic seed provenance
- **generator_state**: serializable controller-local state
- **sample_count**: number of coefficient samples already consumed
- **factorization_contract**: deterministic symmetric covariance-factor method and tolerance

Resume rule:

- Restoring this entity with the same belief and schema must reproduce the next coefficient sample and action.

### ControllerObjective

Represents one boundary evaluation of the fixed controller panel.

- **boundary_step**: optimizer step at evaluation
- **ordered_component_losses**: one finite target-token-weighted loss per resolved global granularity
- **ordered_granularities**: evaluation order
- **uniform_objective**: arithmetic mean of component losses
- **evaluation_example_count**: fixed controller panel count
- **evaluation_target_tokens**: aggregate valid targets
- **aggregation_method**: stable target-token-weighted causal loss contract
- **controller_manifest_hash**: data identity link
- **evaluation_status**: complete or failed

Validation rules:

- Every resolved granularity is evaluated exactly once per conceptual boundary.
- Component losses and uniform objective must be finite.
- Controller data never appear under ordinary-validation or final-holdout identities.

### DecisionWindowState

Represents delayed reward ownership and exact resume phase.

- **phase**: lifecycle state listed below
- **window_index**: nonnegative controller window index
- **window_length**: resolved positive `h`
- **boundary_step**: optimizer step at window start
- **current_action**: selected global or per-block action, when active
- **completed_optimizer_steps**: integer in `[0, h]`
- **pre_window_objective**: controller objective at window start
- **boundary_evaluation_status**: not-started, pending, complete, or failed
- **terminal_status**: continuing, complete-boundary, incomplete, or failed

Lifecycle states:

1. `initial_objective_pending`: manifests and prior exist, but no controller objective or action exists.
2. `ready_for_action`: a finite baseline objective and posterior exist; identity prediction and Thompson sampling may occur.
3. `active_window`: one action owns `0..h-1` completed optimizer steps.
4. `boundary_evaluation_pending`: the `h`th update completed; reward/update/action work has not fully committed.
5. `terminal_incomplete`: training ended with fewer than `h` owned steps; no observation is emitted.
6. `failed`: controller evaluation or validation failed; no partial posterior or next action is valid.

Transition rules:

- Warmup checkpoints retain `initial_objective_pending`; warmup steps do not increment the Bayesian window.
- `initial_objective_pending -> ready_for_action` only after the initial objective is finite and saved.
- `ready_for_action -> active_window` only after a valid predictive belief and action sample.
- `active_window -> boundary_evaluation_pending` exactly when completed optimizer steps reach `h`.
- `boundary_evaluation_pending -> ready_for_action` only after objective, reward, posterior update, and boundary record commit; if more training remains, action selection follows.
- `active_window -> terminal_incomplete` when training ends early.
- Any controller validation/evaluation failure enters `failed` without posterior conditioning.

### PreNestedWarmupState

Represents the pre-controller schedule and exact continuation point.

- **policy**: legacy `full_only` or `balanced_global`
- **requested/completed steps**: configured duration and successful optimizer updates
- **action interval**: fixed optimizer-step width of each global-action window
- **schedule seed/hash/list**: independent named seed provenance and immutable actions
- **current window index/offset**: exact position, including inside-window checkpoints
- **per-granularity counts**: complete scheduled windows per label
- **controller-start step**: intended first fixed-panel baseline step
- **completion state**: completion step/reason or terminal-incomplete reason

Each balanced pass is one seeded permutation of all resolved granularities.
Resume validates every identity and progress field against the checkpoint step.
The model, optimizer, scheduler, dataloader, global step, and token counters
continue across windows. No controller state other than journal commit
provenance changes during this phase.

### ControllerObservation

Represents the complete scientific record for one finished decision window.

- **window_index**: owned decision window
- **action**: action used for exactly `h` updates
- **feature_vector**: action features used by the reward model
- **pre_window_objective**: baseline at window start
- **post_window_objective**: boundary objective after `h` updates
- **reward**: `(pre - post) / h`
- **predicted_reward**: `x^T m^-`
- **prediction_error**: reward minus predicted reward
- **gain_vector**: Gaussian conditioning gain
- **predictive_mean/covariance**: state used for action and likelihood
- **posterior_mean/covariance**: state after conditioning
- **boundary_step_start/end**: exact optimizer-step ownership
- **controller_manifest_hash**: fixed panel provenance
- **sampling_provenance**: coefficient-sample round and state link
- **status**: committed or failed

Validation rules:

- A committed observation owns exactly `h` optimizer steps under one action.
- One boundary objective is shared as the preceding post-window and next pre-window value.
- A failed record contains attributable diagnostics but no committed posterior state.

### ControllerRunSummary

Represents completed-run controller audit data.

- **method identity and probabilistic inputs**
- **feature schema and resolved granularities**
- **data-role manifest paths and hashes**
- **decision interval and boundary counts**
- **action frequencies and optional action entropy**
- **posterior uncertainty summaries by arm or block/granularity effect**
- **controller evaluation counts and target-token totals**
- **terminal-window status and partial progress**
- **resume count, source checkpoint, and compatibility results**
- **controller boundary journal path and hash**
- **failure summary**, if applicable

### FinalHoldoutComparison

Represents a post-training-only evaluation.

- **run_id**: completed training run
- **checkpoint_path**: ordinary-validation-selected checkpoint or explicit comparison checkpoint
- **checkpoint_selection_provenance**: proves the final holdout did not select the checkpoint
- **final_holdout_manifest_hash**: verified data identity
- **ordered_per_granularity_losses**: target-token-weighted results
- **uniform_average_loss**: optional family-level comparison summary
- **evaluation_timestamp/provenance**: post-training invocation data
- **result_path and result_hash**: separate final-comparison artifact

Validation rules:

- Training status must already be complete.
- The final manifest must match saved run provenance.
- Evaluation cannot update controller state, checkpoint selection, configuration, or training summary decisions.

## Relationships

- One `BayesianAdaptiveConfiguration` and one `AdaptiveExperimentIdentity` resolve before a Bayesian run starts.
- One `DataRolePartition` owns exactly four `DataRoleManifest` entities over one stable identity pool.
- One `ActionFeatureSchema` resolves from scope, ordered granularities, and block count.
- One `GaussianBeliefState`, one `PosteriorSamplingState`, and one `DecisionWindowState` form the live Bayesian controller checkpoint state.
- Each committed `ControllerObservation` links one action, two consecutive controller objectives, and one predictive-to-posterior belief update.
- A post-window `ControllerObjective` becomes the next window's pre-window objective without reevaluation.
- `ControllerRunSummary` aggregates controller observations but does not replace their boundary journal.
- `FinalHoldoutComparison` consumes the saved final manifest only after training and remains independent of controller and ordinary validation records.

## Persisted Artifacts

- `config.json`: resolved adaptive identity, configuration, feature schema summary, and all role hashes
- `data_roles_manifest.json`: parent partition contract and overlap audit
- `training_manifest.json`: optimizer-training selected identities and hash
- `controller_manifest.json`: fixed 128-example panel, seed, provenance, and hash
- `validation_manifest.json`: existing ordinary-validation compatibility artifact with stable identities/hash
- `final_holdout_manifest.json`: untouched 512-example role, independent seed, provenance, and hash
- `controller_metrics.jsonl`: one structured event per initial boundary, completed observation, incomplete terminal window, or attributable failure
- `controller_summary.json`: action, uncertainty, evaluation, resume, and terminal summaries
- `metrics.csv`: compact method/scope/window/controller scalar fields alongside existing metrics
- `run_summary.json`: paths, hashes, method identity, and completed controller summary
- checkpoints: complete Bayesian live state plus existing model/optimizer/scheduler/training state
- `final_holdout_results.json`: separate post-training comparison output

## Episodic Reset Extension

`EpisodicResetState` represents the optional global-only episode and
acquisition contract:

- **contract**: enabled flag, K, `full_prior` or `acquisition_only` policy,
  balanced-global acquisition, pass count, and root schedule seed
- **controller start**: optimizer step after optional warmup; episode timing is
  relative to this step
- **live episode**: index, start/end, offset, episode seed, schedule/hash,
  acquisition progress/counts, selection source, and current action
- **reset history**: actual full-prior reset count/steps and completed episode
  archives containing the conditioned boundary posterior and selected policy
- **statistics**: total, forced-acquisition, and Thompson-only frequencies and
  entropy

Forced observations use the ordinary Gaussian likelihood update but never draw
from the posterior-sampling generator. At a completed boundary followed by
more training, `full_prior` restores the configured prior exactly;
`acquisition_only` leaves the conditioned posterior unchanged. Neither policy
resets the model, optimizer, scheduler, scaler, dataloader, counters, panel, or
Thompson generator.
