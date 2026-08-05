# Research: Probabilistic Adaptive Granularity

## Decision 1: Reuse `thompson` for Bayesian control and isolate UCB

- **Decision**: Add `adaptive_global` as a model granularity-sampling mode, keep `adaptive_per_block`, and dispatch by both mode and strategy. `thompson` in either adaptive mode uses the new Bayesian controller. `adaptive_per_block + ucb` continues through the existing heuristic state, reward, checkpoint, and reporting path without semantic changes. `adaptive_global + ucb` is invalid.
- **Rationale**: `global` already means random global selection under `nested-random`, so overloading it would make experiments ambiguous. The existing `adaptive_per_block` surface must remain for UCB compatibility, while the accepted clarification requires the old pseudo-Thompson runtime to be replaced rather than retained. Explicit dispatch before state construction prevents Bayesian state from being coerced into the legacy per-block statistics.
- **Alternatives considered**:
  - Add a generic `adaptive` mode plus a separate scope field. Rejected because it would disturb the established `adaptive_per_block + ucb` surface and introduce conflicting sources of scope truth.
  - Add a new `bayesian_ts` strategy. Rejected because the clarification explicitly assigns genuine Bayesian behavior to `thompson`.
  - Keep the heuristic Thompson path as a selectable baseline. Rejected because the clarification replaces that path; only its historical artifacts remain reportable.

## Decision 2: Require a Bayesian controller mapping and reject legacy Thompson configuration

- **Decision**: A run resolving to `thompson`, including one selected through the previous implicit default, must contain a dedicated Bayesian controller configuration with prior mean, prior covariance, observation-noise variance, process-noise covariance, and the required controller/final-holdout sections. The decision interval may be omitted only because the method-defined default is 50 optimizer steps. Legacy exploration-scale, decay-rate, and reward-penalty fields do not satisfy this contract and are not consumed by Bayesian Thompson.
- **Rationale**: The current resolver silently supplies heuristic Thompson defaults. Reusing those values would silently reinterpret old experiments and contradict the migration clarification. A dedicated mapping keeps new probability-model inputs separate from UCB's existing flat heuristic controls and lets configuration fail before model training.
- **Alternatives considered**:
  - Auto-fill Bayesian priors and noise values for old configs. Rejected because the chosen migration policy requires an attributable error instead of guessed experimental parameters.
  - Reuse heuristic exploration/decay fields as approximate Bayesian inputs. Rejected because they have different mathematical meanings.

## Decision 3: Use one deterministic four-role partition for Bayesian runs

- **Decision**: Partition the stable usable-example identity pool without replacement. Select ordinary validation first with the existing validation seed and algorithm, then select the 128-example controller panel from the remainder with a new controller seed stream, then select the 512-example final holdout from the remainder with a distinct final-holdout seed stream; all remaining examples form optimizer training. Persist source identities and full manifests for all four roles, plus role hashes and a parent split hash.
- **Rationale**: Selecting ordinary validation first preserves its established identity for a matching Bayesian config. Sampling later roles only from the remainder proves pairwise disjointness by construction. Named seed streams fit the existing reproducibility contract without coupling role selection to training or posterior randomness. Stable source identity should combine dataset fingerprint and original source row identity rather than rely only on post-shuffle positions.
- **Alternatives considered**:
  - Independently sample all roles and reject overlaps. Rejected because it wastes examples and creates avoidable failure cases.
  - Select the final holdout first. Rejected because it would unnecessarily change which examples ordinary validation receives relative to current behavior.
  - Apply the four-role split to UCB and nonadaptive runs. Rejected because UCB and existing baselines must remain behaviorally unchanged.

## Decision 4: Reuse stable validation loss for the controller objective

- **Decision**: Build a fixed, deterministic controller loader and reuse the existing per-granularity validation evaluation in resolved granularity order. Compute the controller objective as the arithmetic mean of the resulting target-token-weighted losses. Add strict finite checks for every per-granularity component, the aggregate objective, and the derived reward. Controller records use a distinct split/event identity and never enter ordinary checkpoint selection.
- **Rationale**: The repository evaluator already provides evaluation mode, no-gradient execution, causal target counting, float64 target-token-weighted aggregation, distributed reduction, deterministic order supplied by the caller, and restoration of the active granularity profile. Reuse keeps the scientific semantics identical while a separate identity prevents controller observations from contaminating ordinary validation.
- **Alternatives considered**:
  - Implement a second loss aggregator. Rejected because duplicated loss semantics would risk controller/validation drift.
  - Use training-batch loss or a rotating controller batch. Rejected by the experimental method.

## Decision 5: Use an identifiable contrast feature schema for both scopes

- **Decision**: Represent global arms with an intercept plus deterministic sum-to-zero contrasts over the ordered granularity set. Represent per-block actions with one intercept plus the same contrast basis independently for each block. The coefficient dimension is `|G|` for global scope and `1 + B(|G|-1)` for per-block scope. The ordered basis and coefficient names are saved as feature-schema provenance.
- **Rationale**: Sum-to-zero contrasts distinguish all arms, remain identifiable, handle one granularity and one block naturally, and avoid privileging a reference label. The additive per-block sampled optimum decomposes across blocks, so the complete profile can be chosen without enumerating `|G|^B` profiles. A shared intercept is the specified intercept-only context; no progress or learning-rate features are introduced.
- **Alternatives considered**:
  - Full block/granularity one-hot features. Rejected because the additive design is rank-deficient without an explicit constraint.
  - Reference-label coding. Rejected because it can make prior uncertainty depend on which arbitrary label is chosen as the reference.
  - Adjacent-block interactions. Rejected as follow-up work outside this feature.

## Decision 6: Resolve explicit Gaussian inputs to dense float64 state

- **Decision**: Accept explicit scalar, diagonal, or dense prior/process covariance inputs and scalar or coefficient-length prior means, resolve them against the feature dimension, and save the resolved vector and matrices. Maintain controller mathematics in CPU float64. Symmetrize covariance after prediction and conditioning, validate finiteness and positive semidefiniteness with a documented tolerance, and obtain Gaussian samples by transforming dedicated-generator standard-normal draws through a deterministic symmetric covariance factor. Materially negative eigenvalues fail; only roundoff-scale negatives may clamp to zero.
- **Rationale**: Controller dimensions are small (`|G|` or `1 + B(|G|-1)`), so dense state is the clearest research representation. CPU float64 separates controller numerics from mixed-precision model training. A symmetric factor supports valid degenerate covariance, including zero process noise, without adding unrequested jitter that would change the probability model.
- **Alternatives considered**:
  - Cholesky with automatic jitter. Rejected because jitter silently changes zero or low-rank covariance.
  - GPU or training-dtype posterior state. Rejected because device topology and mixed precision would make controller reproducibility harder to audit.
  - A diagonal-only posterior. Rejected because Gaussian conditioning creates cross-covariances and the specification requires the explicit posterior covariance.

## Decision 7: Use an explicit transactional decision-window state machine

- **Decision**: Persist phases for `initial_objective_pending`, `ready_for_action`, `active_window`, `boundary_evaluation_pending`, `terminal_incomplete`, and `failed`. Bayesian control starts after any pre-nested warmup. The initial objective is evaluated once, prediction and sampling select an action, every optimizer step in the window reuses that action, and the `h`th update enters boundary processing before generic metrics/checkpoint writing. A boundary commits objective, reward, posterior update, and—only if more training remains—the next sampled action after every value validates successfully.
- **Rationale**: Optimizer step modulo `h` cannot distinguish a checkpoint taken before versus after objective evaluation, posterior update, or action selection. Explicit phase and transactional commit prevent duplicate or skipped observations and ensure incomplete terminal windows never become rewards. Warmup uses a forced largest granularity and therefore cannot be credited to a Bayesian action.
- **Alternatives considered**:
  - Infer window phase from optimizer step. Rejected because exact-boundary resumes would be ambiguous.
  - Select a new action before conditioning on the preceding reward. Rejected because it violates the stated posterior Thompson sequence.
  - Include warmup steps in the first action window. Rejected because those steps are not governed by that action.

## Decision 8: Make rank zero authoritative for distributed controller state

- **Decision**: Reuse distributed validation reduction on the partitioned controller panel. Rank zero owns the controller-local random generator, posterior transition/update, and action selection, then broadcasts the validated controller state and chosen action to every rank. All ranks apply the same action for the window; shared artifacts remain rank-zero writes.
- **Rationale**: The training stack already supports FSDP, distributed validation reduction, object broadcast, and rank-zero artifacts. Central controller authority avoids per-rank RNG or numerical divergence while preserving the repository's existing single-node distributed training surface.
- **Alternatives considered**:
  - Sample independently but deterministically on every rank. Rejected because unrelated RNG consumption or numerical drift could split actions.
  - Restrict Bayesian runs to one process. Rejected because the existing training path already has the primitives needed to preserve its distributed execution model.

## Decision 9: Extend checkpoints with a separate Bayesian schema

- **Decision**: Store a Bayesian method/version discriminator, scope, feature schema, resolved ordered granularities, probabilistic inputs, role hashes, decision-window phase/action/progress/baseline, predictive and posterior belief, controller-local sampling state, boundary journal position, and resume provenance in every Bayesian checkpoint. Validate the complete schema before restoring. Legacy heuristic Thompson checkpoints fail migration; UCB continues through its existing validator and state schema unchanged.
- **Rationale**: The existing checkpoint already persists model, optimizer, scheduler, training position, global RNG state, and heuristic adaptive fields. A separate controller schema prevents old `mean_reward/count` statistics from being mistaken for a Gaussian posterior and provides enough phase information for exact-boundary and inside-window continuation.
- **Alternatives considered**:
  - Retrofit the legacy adaptive state object. Rejected because shared coercion would put UCB compatibility and Thompson migration safety at risk.
  - Restore a missing posterior from config. Rejected because it would discard observations and break resume equivalence.

## Decision 10: Use dedicated controller artifacts and explicit method provenance

- **Decision**: Append full boundary events to `controller_metrics.jsonl`, write a completion/failure aggregate to `controller_summary.json`, retain compact method/scope/window fields in ordinary metrics, and include controller paths and summaries in `run_summary.json`. Write the resolved config again immediately after data-role manifests are established. New artifacts carry a Bayesian controller family/version; historical `thompson` artifacts without that discriminator are classified as legacy heuristic Thompson. Preserve the existing UCB label and style keys.
- **Rationale**: Posterior vectors and matrices do not fit cleanly in the fixed ordinary metrics CSV and should not be duplicated on every training row. A boundary journal gives each observation clear ownership and supports recovery/audit. The strategy name alone cannot distinguish old and new Thompson results.
- **Alternatives considered**:
  - Repurpose the existing `adaptive_per_block_thompson` reporting label. Rejected because it would make historical figures scientifically ambiguous.
  - Rewrite historical artifacts with migration metadata. Rejected because existing artifacts must remain interpretable without migration.
  - Put full posterior state into every ordinary metrics row. Rejected because it bloats artifacts and obscures boundary semantics.

## Decision 11: Keep the final holdout outside training and checkpoint selection

- **Decision**: Reserve and manifest the final holdout during data setup but do not construct or invoke its evaluator during training. Provide a separate post-training final-comparison entrypoint that verifies run completion and the saved manifest, selects the already ordinary-validation-chosen checkpoint (or requires an explicit checkpoint when no such selection exists), evaluates all resolved global granularities in deterministic order, and writes separate final-holdout results without modifying training artifacts or controller state.
- **Rationale**: A separate entrypoint creates a visible lifecycle boundary and prevents accidental consultation during controller updates, monitoring, or checkpoint selection. Reusing the per-granularity evaluator gives final results the same target-token-weighted semantics while keeping them distinct from controller and ordinary validation data.
- **Alternatives considered**:
  - Automatically evaluate the final holdout inside the training loop at intervals. Rejected because it violates the untouched-holdout contract.
  - Automatically evaluate at completion before writing the run summary. Rejected because it makes final comparison look like another training-time stage and increases the chance that results affect later decisions.

## Decision 12: Validate in staged, opt-in experiments

- **Decision**: Implement and validate Bayesian global control first, then additive per-block control, then provenance/reporting compatibility. Add focused posterior, split, boundary, resume, distributed, and artifact tests. Keep the current default pilot matrix unchanged and add opt-in Bayesian global/per-block smoke fixtures or overrides. Preserve all UCB regression expectations while replacing old Thompson execution tests with historical-reporting fixtures and Bayesian posterior tests.
- **Rationale**: The global action space isolates delayed reward and posterior correctness before structured credit assignment. Avoiding an automatic expansion of the expensive pilot matrix preserves comparison cost and baseline behavior. Controlled tests establish method correctness; smoke tests establish integration only.
- **Alternatives considered**:
  - Implement both scopes in one undifferentiated change. Rejected because failures would be harder to attribute.
  - Add Bayesian modes to every default pilot submission. Rejected because it changes baseline compute and queue behavior without explicit opt-in.

