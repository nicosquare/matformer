# Controller Boundaries and Artifacts Contract

## Boundary Processing Order

### Fresh run after optional pre-nested warmup

1. Verify the four role manifests and hashes.
2. Evaluate the controller objective at the current optimizer step.
3. Save the initial boundary record with no action reward.
4. Form the predictive belief with identity transition and process covariance.
5. Draw one coefficient sample from controller-local random state.
6. Select one action with deterministic tie-breaking.
7. Enter an active window with progress zero and the saved objective baseline.

### Active window

For each optimizer step:

1. Restore/configure the saved action.
2. Perform exactly one optimizer update governed by that action.
3. Increment window progress only after the update succeeds.
4. If progress is less than `h`, emit ordinary metrics/checkpoints with the
   unchanged active-window controller state.
5. If progress equals `h`, mark boundary evaluation pending and complete the
   boundary transaction before generic post-step checkpoint writing.

### Completed boundary transaction

1. Evaluate all resolved global granularities on the fixed controller panel in
   resolved order.
2. Validate every component and the uniform objective as finite.
3. Compute `reward = (pre_objective - post_objective) / h`.
4. Validate reward, predictive state, gain, and posterior state.
5. Append one committed boundary observation.
6. Make the post-window objective the next baseline without reevaluation.
7. If more optimizer steps remain, predict and sample the next action; otherwise
   retain controller-local random state before that unused sample.

The update is transactional: a failure before step 5 leaves no committed
posterior update or next action. A failure event records the attributable stage.

## Checkpoint Phase Contract

Every Bayesian checkpoint includes:

- method family/version, strategy, scope, and feature-schema hash;
- resolved ordered granularities and block count;
- normalized prior/noise/identity-transition provenance;
- all four role manifest hashes and the parent split hash;
- posterior and, when applicable, predictive mean/covariance;
- controller-local generator state and sample count;
- phase, window index, boundary step, action, action features, `h`, and progress;
- pre-window objective and component-loss provenance;
- last committed boundary-journal position/hash;
- resume count, source checkpoint, and compatibility results.

Phase-specific requirements:

| Phase | Required state |
|---|---|
| `initial_objective_pending` | Prior, manifests, no action/reward |
| `ready_for_action` | Finite baseline and posterior; no active action |
| `active_window` | Current action, baseline, progress in `[0,h-1]` |
| `boundary_evaluation_pending` | Action and progress `h`; no partially valid update |
| `terminal_incomplete` | Action, baseline, progress `<h`, no observation |
| `failed` | Last valid state plus attributable failure record |

Resume rejects missing or incompatible method version, strategy, scope,
granularity order, feature schema, probabilistic inputs, role hashes, phase,
belief, or sampling state. A legacy heuristic Thompson checkpoint is never
coerced into this schema. UCB continues through the existing legacy checkpoint
contract.

## `controller_metrics.jsonl`

The controller journal is append-only and contains one JSON object per event.

### Common fields

- `schema_version`
- `run_id`
- `event_type`: `initial_boundary`, `completed_window`, `terminal_incomplete`,
  or `controller_failure`
- `method_family`, `method_version`, `strategy`, `scope`
- `ordered_granularities`, `feature_schema_hash`
- `controller_manifest_hash`, `data_roles_manifest_hash`
- `boundary_step`, `window_index`, `decision_interval_steps`
- `resume_count`, `resume_source_checkpoint`

### Initial boundary fields

- ordered per-granularity controller losses
- uniform controller objective
- predictive mean/covariance used for first selection
- selected action, feature vector, sampled predicted reward
- posterior-sampling provenance and deterministic tie outcome

No reward or posterior conditioning fields are present.

### Completed-window fields

- window start/end steps and exact completed-step count
- action and feature vector
- pre/post objectives and ordered post-objective component losses
- reward, predicted reward, and prediction error
- predictive mean/covariance, gain, posterior mean/covariance
- controller-local sampling provenance before/after any next selection
- cumulative action frequencies and uncertainty summary

### Terminal-incomplete fields

- active action, window start, progress `<h`, and pre-window objective
- explicit `observation_emitted: false`
- unchanged posterior and controller-local sampling state summary

### Failure fields

- failing stage and error category
- last valid phase, belief hash, and journal position
- explicit `posterior_updated: false` and `new_action_selected: false`
- offending non-finite/compatibility field when safely representable

## `controller_summary.json`

The summary contains:

- full resolved controller and data-role provenance;
- completed observation count and controller evaluation count;
- per-action frequencies for global scope or per-block/granularity frequencies
  for additive scope;
- final posterior mean/covariance and interpretable uncertainty summaries;
- boundary objective/reward/prediction-error summaries;
- terminal-window status and partial progress;
- controller journal path/hash;
- resume provenance and any failure summary.

## Ordinary Metrics and Run Summary

Ordinary `metrics.csv` rows receive compact scalar/identity fields only:

- Bayesian method family/version, strategy, scope;
- current action summary, window index/progress, boundary step;
- latest completed controller objective/reward/prediction error when available;
- controller and final-holdout manifest hashes;
- controller journal and summary paths.

Full vectors and matrices remain in the controller journal, summary, and
checkpoints. Controller rows never use `split=validation` and never enter best
checkpoint selection.

`run_summary.json` includes the final controller summary and artifact paths.
The resolved `config.json` is rewritten immediately after role manifests are
created so a later failure still leaves complete data provenance.

## Artifact Identity and Historical Reporting

- New Bayesian global label: `probabilistic_global_thompson`.
- New Bayesian per-block label: `probabilistic_per_block_thompson`.
- Historical artifacts with `adaptive_per_block + thompson` but no Bayesian
  method discriminator retain the legacy `adaptive_per_block_thompson`
  identity and are displayed as legacy heuristic Thompson.
- Existing `adaptive_per_block_ucb` labels, styles, checkpoint semantics, and
  reports remain unchanged.
- Reporting classification must agree across modular reporting helpers and the
  active compatibility implementation; strategy name alone is insufficient.

## Final Holdout Contract

The 512-example final manifest is written during setup, but no final loader or
evaluation result is consumed by the training/controller workflow.

A post-training final comparison:

1. verifies completed training status and all saved final-manifest provenance;
2. uses the ordinary-validation-selected checkpoint, or requires an explicit
   checkpoint when no such selection exists;
3. evaluates every resolved global granularity in deterministic order using
   target-token-weighted validation semantics;
4. writes `final_holdout_results.json` separately;
5. never mutates controller state, checkpoint selection, hyperparameters, or
   the historical training/controller journal.

