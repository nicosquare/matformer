# Feature Specification: Probabilistic Adaptive Granularity

**Feature Branch**: `010-probabilistic-adaptive-granularity`  
**Created**: 2026-08-05  
**Status**: Draft  
**Input**: User description: "Implement probabilistic adaptive granularity sampling for MatFormer training, grounded in the resolved experimental decisions in the 2026-08-05 discussion note, with Bayesian global adaptation first, additive per-block adaptation second, and provenance-safe baseline comparison."

## Clarifications

### Session 2026-08-05

- Q: When an existing configuration selects `thompson`, how should replacement work? → A: Reuse `thompson` for the genuine Bayesian controller; configurations missing required Bayesian inputs fail with a migration error, while UCB remains untouched.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Learn global training utility from heldout rewards (Priority: P1)

As a researcher, I can select probabilistic global adaptation so one resolved granularity is held fixed across all transformer blocks for a decision window, and future selections learn from the resulting improvement on a fixed controller panel rather than from training-batch losses.

**Why this priority**: The global action space is the minimum viable experiment for validating the data separation, delayed-reward protocol, Gaussian posterior, Thompson selection, checkpoint persistence, and deterministic resume behavior before introducing block-level credit assignment.

**Independent Test**: Run a controlled global experiment with a fixed controller panel and known reward sequence, then verify boundary objectives, per-step rewards, posterior updates, action selections, saved state, and fresh-versus-resumed equivalence without enabling per-block adaptation.

**Acceptance Scenarios**:

1. **Given** four pairwise-disjoint data roles and a valid probabilistic global configuration, **When** training begins, **Then** the initial controller objective is evaluated on the fixed 128-example controller panel over every resolved global granularity in deterministic order before any reward exists.
2. **Given** a selected global action and a resolved window length of `h` optimizer steps, **When** exactly `h` steps complete, **Then** the action has remained fixed, the controller objective is evaluated once at the new boundary, and the action receives reward `(pre-window objective - post-window objective) / h`.
3. **Given** a controlled reward sequence, **When** completed windows update the Gaussian belief, **Then** both posterior expectations and uncertainty respond to the observations and subsequent posterior samples can change the selected arm.
4. **Given** equivalent initial state, data manifests, configuration, seeds, software, hardware, distributed topology, and deterministic runtime settings, **When** one execution is uninterrupted and another resumes from a checkpoint inside a decision window, **Then** their manifests, controller window state, posterior-sampling state, sample count, and subsequent sampled actions match exactly, while their completed-window objectives, rewards, posterior means, and posterior covariances match with relative tolerance `1e-6` and absolute tolerance `1e-8`.
5. **Given** a run whose controller objective or reward is non-finite or cannot be evaluated, **When** the boundary is processed, **Then** no posterior update or new action selection occurs and the run fails with an error attributable to controller evaluation.

---

### User Story 2 - Learn additive per-block preferences (Priority: P2)

As a researcher, I can select probabilistic per-block adaptation so each decision-window action contains one resolved granularity per transformer block and the controller infers additive block/granularity contributions from complete-profile rewards without enumerating every possible profile.

**Why this priority**: This extends the validated controller and reward protocol to the scientifically important structured action space while avoiding the invalid practice of assigning one complete-profile reward independently to every selected block action.

**Independent Test**: Use a small controlled model with at least two blocks and at least two resolved granularities, provide rewards that favor different block/granularity effects, and verify that the inferred effects diverge, uncertainty is retained appropriately, and selected complete profiles maximize sampled additive reward without exhaustive profile enumeration.

**Acceptance Scenarios**:

1. **Given** a valid per-block probabilistic run, **When** a decision is made, **Then** one granularity is selected for each transformer block and the complete profile remains fixed for the full decision window.
2. **Given** repeated observations of varied complete profiles, **When** the additive posterior is updated, **Then** evidence can produce different preferences for different block/granularity effects without recording the full scalar reward as the independent reward mean for every selected pair.
3. **Given** `B` blocks and an ordered resolved set `G`, **When** a per-block action is selected, **Then** selection uses identifiable additive block/granularity effects and does not create or search an independent arm for each of the `|G|^B` complete profiles.
4. **Given** one transformer block, one resolved granularity, or arbitrary noncanonical granularity labels, **When** per-block adaptation runs, **Then** its action, posterior, provenance, and resume behavior remain valid without assuming four canonical labels.

---

### User Story 3 - Preserve comparison and audit integrity (Priority: P3)

As a researcher, I can distinguish current Bayesian Thompson, random global, random per-block, retained UCB, and historical legacy heuristic Thompson runs from their saved artifacts, while the new Bayesian controller replaces the old pseudo-Thompson training path and UCB retains its prior behavior.

**Why this priority**: Adaptive results are scientifically useful only when the method, data roles, posterior state, and resume history are auditable, older heuristic Thompson artifacts are never mistaken for Bayesian Thompson sampling, and UCB comparisons remain stable.

**Independent Test**: Run new `thompson` configurations with and without the required Bayesian inputs, inspect new Bayesian, retained UCB, random, nested-all, standalone, and historical legacy Thompson artifacts, and verify replacement, migration failure, provenance distinction, and UCB compatibility independently.

**Acceptance Scenarios**:

1. **Given** a new training configuration selecting `thompson` with all required probabilistic inputs, **When** the strategy resolves, **Then** it uses the genuine Bayesian controller and never the legacy heuristic Thompson behavior.
2. **Given** a configuration selecting `thompson` without the required probabilistic inputs, **When** configuration is resolved, **Then** it fails before training with an attributable migration error rather than receiving inferred defaults or falling back to the legacy heuristic.
3. **Given** a historical pseudo-Thompson artifact, **When** it is loaded for reporting, **Then** it remains identifiable as legacy heuristic Thompson and is not relabeled as a Bayesian posterior run.
4. **Given** an existing UCB configuration, checkpoint, or reporting path, **When** it is used after this feature, **Then** its behavior and interpretation remain unchanged.
5. **Given** a probabilistic adaptive checkpoint or completed run, **When** a researcher audits it, **Then** the controller data, final-holdout data, window state, posterior evolution, sampling randomness, action frequencies, uncertainty, and resume provenance required to reproduce future choices are present in structured artifacts.
6. **Given** a reserved 512-example final holdout, **When** training, controller updates, checkpoint selection, monitoring, or hyperparameter selection occurs, **Then** the holdout is neither evaluated nor consulted and remains available only for final run comparison.

### Edge Cases

- At the initial boundary, the controller objective is saved before an action window begins; no reward or posterior observation exists yet, but the predictive belief and sampled action are reproducible.
- If the available examples cannot provide 128 controller examples, 512 final-holdout examples, the requested ordinary-validation set, and at least one optimizer-training example as pairwise-disjoint roles, the run fails before training.
- Explicit resolved granularity sets may contain arbitrary valid labels and counts; no probabilistic behavior assumes the canonical `s`, `m`, `l`, `xl` set or exactly four arms.
- With one resolved granularity, that sole action is selected deterministically while boundary evaluation, posterior learning, persistence, and provenance still occur.
- With one transformer block, global and per-block scopes remain distinct requested experiment modes even though their realized profiles may coincide.
- A checkpoint taken exactly at a controller boundary records whether the preceding window has been evaluated and whether the next action has been selected, so resume neither duplicates nor skips an objective, reward, transition, update, or sample.
- A checkpoint taken inside a window preserves the current action, completed optimizer steps within the window, pre-window objective, posterior state, and sampling state; resume completes the same window before reward or reselection.
- A run ending after fewer than `h` steps in its current window records an incomplete window and emits no completed observation or posterior update for it.
- Missing, malformed, non-finite, dimensionally incompatible, or configuration-incompatible posterior state on resume causes a clear failure rather than posterior reinitialization.
- Non-finite controller losses, aggregate objectives, rewards, posterior means, or covariance values cause an attributable failure and cannot update the posterior.
- Zero process noise preserves posterior uncertainty between observations except for conditioning; positive process noise restores uncertainty through the identity transition without shrinking expected coefficients toward zero.
- Ties between candidate sampled rewards are resolved deterministically from the resolved granularity order and transformer-block order, and the rule is recorded as provenance.
- A controller-panel or final-holdout manifest or hash mismatch on resume causes failure before further training or evaluation.
- Any overlap or attempted reuse among optimizer training, controller, ordinary validation, and final-holdout examples is rejected before training when detectable and before the affected role is consumed otherwise.
- A controller evaluation failure after a completed training window leaves that window without a posterior observation and does not silently select a replacement action.
- A legacy heuristic Thompson checkpoint cannot resume as a Bayesian Thompson run because it lacks compatible controller state; the resume attempt fails with a migration error while the artifact remains available for historical reporting.

## Requirements *(mandatory)*

### Functional Requirements

#### Strategy and scope

- **FR-001**: Researchers MUST be able to select probabilistic adaptation without changing code and MUST explicitly select either global or per-block adaptive scope.
- **FR-002**: The system MUST validate the resolved sampling mode, adaptive strategy, adaptive scope, granularity set, data-role configuration, probabilistic parameters, decision-window length, and resume compatibility before training wherever the required information is available.
- **FR-003**: Invalid, unsupported, or ambiguous strategy and scope combinations MUST fail clearly rather than falling back to another sampling behavior.
- **FR-004**: Global probabilistic actions MUST consist of one resolved granularity applied to every transformer block.
- **FR-005**: Per-block probabilistic actions MUST consist of one resolved granularity for each transformer block and MUST model complete-profile reward through additive block/granularity contributions.
- **FR-006**: Both probabilistic scopes MUST consume the same controller objective, boundary-to-boundary reward protocol, Gaussian belief semantics, identity transition, and persistence contract.
- **FR-007**: The probabilistic action space MUST derive from the complete resolved ordered granularity set and MUST support any valid label and count without a hard-coded four-granularity assumption.
- **FR-008**: If sampled candidate actions have equal predicted reward, the system MUST select one deterministically according to a documented order derived from the resolved granularity order and, for profiles, transformer-block order.

#### Data roles and objective

- **FR-009**: Every probabilistic adaptive run MUST establish four pairwise-disjoint example roles before training: optimizer training, controller panel, ordinary validation, and final holdout.
- **FR-010**: The controller panel MUST contain exactly 128 examples selected deterministically before training and MUST remain unchanged at every controller boundary in the run.
- **FR-011**: The final holdout MUST contain exactly 512 examples selected deterministically before training using a seed independent from the controller-panel seed.
- **FR-012**: The controller panel MUST be used only for adaptive controller objectives and rewards; it MUST NOT contribute optimizer updates, ordinary validation, checkpoint selection, final evaluation, or any other role.
- **FR-013**: The final holdout MUST NOT be evaluated or consulted during training and MUST NOT influence controller updates, ordinary monitoring, checkpoint selection, hyperparameter selection, or other during-training decisions; it MAY be evaluated only for final run comparison.
- **FR-014**: Ordinary validation data MUST retain its existing checkpoint-selection and monitoring role and MUST NOT overlap optimizer-training, controller, or final-holdout examples.
- **FR-015**: Before training, the system MUST fail if the available data cannot satisfy the requested four disjoint roles while leaving at least one usable optimizer-training example.
- **FR-016**: Each role MUST have a saved selected-example manifest, source provenance, deterministic selection seed, and stable hash sufficient to detect overlap and resume mismatch; roles that inherit an existing deterministic selection MUST still record equivalent provenance.
- **FR-017**: At every controller boundary, the controller objective MUST be the uniform average of the controller-panel loss across all resolved granularities used globally:

  $$
  J_{H_c}(\theta)
  =\frac{1}{|\mathcal G|}
  \sum_{g\in\mathcal G}
  L_{H_c}\left(\theta;(g,\ldots,g)\right).
  $$

- **FR-018**: The controller objective MUST use the repository's stable target-token-weighted validation-loss semantics, fixed evaluation behavior, and deterministic resolved-granularity order; it MUST NOT include training-batch loss.
- **FR-019**: A controller evaluation failure or non-finite component loss or aggregate objective MUST produce a clear attributable error before any posterior update or subsequent action selection.

#### Decision windows and delayed reward

- **FR-020**: The decision-window length `h` MUST be an explicit positive number of optimizer steps and MUST resolve to 50 steps when omitted.
- **FR-021**: The selected global granularity or per-block profile MUST remain fixed for every optimizer step in its decision window.
- **FR-022**: The initial controller boundary MUST evaluate and save the controller objective before the first selected action begins and MUST NOT emit a reward.
- **FR-023**: At each subsequent completed boundary, the system MUST evaluate the controller objective once and assign the preceding action the per-optimizer-step reward

  $$
  r_t=\frac{J_{H_c}(\theta_t)-J_{H_c}(\theta_{t+1})}{h}.
  $$

- **FR-024**: The post-window objective from a completed boundary MUST become the pre-window objective for the next action, so the controller performs only one objective evaluation per boundary.
- **FR-025**: A completed controller observation MUST own exactly `h` optimizer steps taken under one unchanged action.
- **FR-026**: An incomplete terminal window MUST be recorded as incomplete and MUST NOT emit a reward, condition the posterior, or be reported as a completed controller observation.
- **FR-027**: A non-finite reward MUST produce a clear attributable error and MUST NOT update the posterior or trigger selection of a new action.
- **FR-028**: Checkpoints created inside a decision window MUST preserve the selected action, boundary step, number of completed steps in the window, pre-window objective, all controller belief state, and posterior-sampling random state or deterministic equivalent.
- **FR-029**: A resumed run inside a window MUST finish the same window before evaluating its reward or selecting another action.
- **FR-030**: Checkpoints created exactly at a boundary MUST preserve enough phase state to ensure a resumed run neither repeats nor skips boundary evaluation, reward emission, posterior transition, posterior update, or action selection.

#### Probabilistic learning and selection

- **FR-031**: The adaptive controller MUST represent reward with the Gaussian linear observation model

  $$
  r_t=x_t^\top\beta_t+\epsilon_t,
  \qquad \epsilon_t\sim\mathcal N(0,\sigma^2),
  $$

  and MUST maintain an explicit Gaussian belief with posterior mean and covariance over `beta`.
- **FR-032**: The initial context MUST be intercept-only, `z_t = [1]`; training progress, learning rate, controller slopes, action interactions, and learned switching features MUST NOT affect the starting feature's reward model or choices.
- **FR-033**: Before every controller round, the belief MUST use the identity state transition

  $$
  F=I,\qquad m_t^-=m_{t-1},\qquad V_t^-=V_{t-1}+Q,
  $$

  so process noise restores uncertainty without shrinking expected coefficients toward zero.
- **FR-034**: Prior mean, prior covariance, observation-noise variance, and process-noise covariance MUST be explicit reproducible run inputs and MUST be saved with identity-transition provenance.
- **FR-035**: All prior means MUST be finite; prior and process covariances MUST be finite, dimensionally valid, symmetric, and positive semidefinite; observation-noise variance MUST be finite and strictly positive; zero and positive valid process-noise covariances MUST both be supported.
- **FR-036**: Before action selection, the controller MUST draw a coefficient sample from the predictive Gaussian belief and select a genuine posterior Thompson action:

  $$
  \widetilde\beta_t\sim\mathcal N(m_t^-,V_t^-),
  \qquad
  A_t=\arg\max_{A\in\mathcal A}x(A)^\top\widetilde\beta_t.
  $$

- **FR-037**: After a finite completed-window reward, the controller MUST condition the predictive Gaussian belief on that observation according to

  $$
  k_t=\frac{V_t^-x_t}{x_t^\top V_t^-x_t+\sigma^2},
  $$

  $$
  m_t=m_t^-+k_t(r_t-x_t^\top m_t^-),
  $$

  $$
  V_t=V_t^--k_t x_t^\top V_t^-.
  $$

- **FR-038**: Each completed observation MUST record its action features, predicted reward, reward prediction error `r_t - x_t^T m_t^-`, predictive mean and covariance, and posterior mean and covariance after conditioning.
- **FR-039**: Non-finite predictive or posterior means or covariances, invalid covariance state, or a non-finite update intermediate MUST fail clearly and MUST NOT become persisted valid controller state.
- **FR-040**: Global action features MUST distinguish every resolved granularity arm so controlled rewards can change the expected reward and uncertainty of the corresponding arms.
- **FR-041**: Per-block action features MUST use an identifiable additive parameterization of block/granularity contributions, including a defined reference or equivalent constraint where necessary.
- **FR-042**: Per-block conditioning MUST infer additive effects from varied complete-profile observations and MUST NOT assign the full scalar reward independently as the reward mean of every selected block/granularity pair.
- **FR-043**: Per-block Thompson selection MUST maximize the sampled additive complete-profile reward without enumerating or maintaining an independent posterior for all `|G|^B` possible profiles.
- **FR-044**: Compute and switching costs MUST both be exactly zero; the controller MUST NOT apply a Hamming-distance profile-change penalty, an implicit size penalty, a learned switching feature, or a hard compute budget.

#### Persistence, audit, and compatibility

- **FR-045**: Every probabilistic adaptive checkpoint and completed-run artifact MUST record the resolved adaptive strategy and scope, resolved ordered granularity set, window length, current action, boundary step, partial-window progress, and window completion state.
- **FR-046**: Every probabilistic adaptive checkpoint and completed-run artifact MUST include controller and final-holdout manifests, seeds, source provenance, and hashes, plus enough corresponding training and ordinary-validation manifest provenance to audit pairwise disjointness.
- **FR-047**: Structured stepwise Bayesian Thompson controller records MUST include boundary objectives before and after each completed window, emitted reward, prediction error, action, posterior mean and covariance before and after update, and applicable sampling-seed or random-state provenance.
- **FR-048**: Bayesian Thompson completed-run summaries MUST include action frequencies, uncertainty summaries, controller-evaluation counts, incomplete terminal-window status when applicable, and resume provenance; these data MUST be structured rather than available only in console logs.
- **FR-049**: Bayesian Thompson resume MUST verify the strategy, scope, ordered granularity set, feature dimensions and identification, probabilistic inputs, window length, all data manifests and hashes, controller phase, and posterior-sampling state before continuing.
- **FR-050**: Missing or incompatible posterior, window, manifest, or sampling state on a Bayesian Thompson resume MUST fail clearly rather than silently reinitialize, reinterpret, or partially restore the controller.
- **FR-051**: Given matching initial state, data manifests, configuration, seeds, software, hardware, distributed topology, and deterministic runtime settings, fresh and resumed Bayesian Thompson executions MUST match manifests, controller window state, posterior-sampling state, sample count, and sampled actions exactly; controller objectives, rewards, posterior means, and posterior covariances MUST match with relative tolerance `1e-6` and absolute tolerance `1e-8`. Cross-hardware or cross-runtime equivalence is outside this acceptance contract.
- **FR-052**: Saved artifacts MUST unambiguously distinguish current Bayesian Thompson global and per-block runs, random global runs, random per-block runs, retained heuristic UCB runs, and historical legacy heuristic Thompson runs.
- **FR-053**: For new training, the existing `thompson` strategy selection MUST resolve exclusively to the genuine Bayesian controller defined by this specification; the legacy heuristic Thompson implementation MUST NOT remain selectable as a training strategy.
- **FR-054**: A configuration selecting `thompson`, explicitly or through the previous default, MUST provide all required Bayesian prior, noise, data-role, scope, and controller inputs; if any required input is absent or incompatible, resolution MUST fail before training with an attributable migration error rather than infer new defaults or fall back to legacy behavior.
- **FR-055**: Historical heuristic Thompson artifacts MUST remain readable and clearly identified as legacy non-Bayesian results without migration, but their checkpoints MUST NOT resume as Bayesian Thompson runs or be treated as compatible posterior state.
- **FR-056**: Existing UCB configuration, selection behavior, state updates, checkpoint and resume semantics, artifact identity, and reporting MUST remain behaviorally unchanged.
- **FR-057**: Existing model wiring, granularity resolution, correction behavior, random global sampling, random per-block sampling, nested-all behavior, standalone behavior, reporting outside the replaced Thompson path, and non-adaptive checkpoint semantics MUST remain behaviorally unchanged.

### Research & Experiment Requirements

- **EX-001**: The feature MUST be delivered as two independently testable experimental stages: P1 Bayesian global adaptation first and P2 Bayesian additive per-block adaptation reusing the validated P1 controller and reward protocol.
- **EX-002**: P1 validation MUST establish data-role disjointness, fixed-panel objective values, delayed reward alignment, posterior learning, posterior Thompson selection, persistence, and boundary-safe resume behavior before P2 is required for minimum viability.
- **EX-003**: P2 validation MUST demonstrate structured credit assignment through an additive model rather than independent complete-profile arms or duplicated scalar rewards.
- **EX-004**: P3 provenance validation MUST compare current Bayesian Thompson global and per-block artifacts, random global and per-block artifacts, retained UCB artifacts, and historical legacy heuristic Thompson artifacts and demonstrate unambiguous experimental identity.
- **EX-005**: Controller evaluation MUST cover every resolved granularity uniformly on the same panel at each boundary, including when only one granularity is resolved.
- **EX-006**: Controlled posterior tests MUST include reward sequences that change posterior means, reduce or restore relevant uncertainty as expected, and alter posterior Thompson preferences.
- **EX-007**: Controlled per-block tests MUST include at least two independently identifiable block/granularity effects with different learned preferences.
- **EX-008**: Resume validation MUST cover checkpoints exactly at a controller boundary and strictly inside a controller window, as well as a run ending with an incomplete window.
- **EX-009**: Validation MUST cover zero process noise, positive process noise, deterministic ties, non-finite inputs and observations, incompatible posterior state, manifest mismatch, and insufficient disjoint data.
- **EX-010**: Final comparisons MAY evaluate the untouched final holdout only after training-time decisions are complete and MUST report that evaluation separately from controller and ordinary-validation results.
- **EX-011**: Smoke tests MUST be treated as method validation only and MUST NOT be used to claim scientific superiority.

### Scope Boundaries

This feature includes the fixed controller and final-holdout protocol, Bayesian global arms, Bayesian additive per-block effects, decision-window reward attribution, deterministic posterior Thompson selection, adaptive persistence and resume, structured controller artifacts, and provenance-safe baseline comparison.

The following are excluded: adjacent-block or higher-order action interactions; progress, learning-rate, loss-slope, neural, or other learned context beyond an intercept; full reinforcement learning, policy networks, replay buffers, or credit across multiple changing actions; compute or switching penalties, learned switching effects, or hard compute budgets; fresh or rotating controller panels; training-batch-loss rewards; deployment routing; scientific-superiority claims from smoke tests; and redesign of MatFormer FFN mathematics, correction mechanisms, or unrelated training modes.

### Key Entities

- **Adaptive Experiment Identity**: The resolved strategy, scope, and method provenance that distinguish current Bayesian Thompson global and per-block runs, random global and per-block runs, retained UCB runs, and historical legacy heuristic Thompson artifacts.
- **Resolved Granularity Set**: The ordered collection of valid granularity labels that defines global arms, per-block choices, objective evaluation order, deterministic tie-breaking, and posterior feature dimensions.
- **Data-Role Manifest**: The selected examples, deterministic seed, source provenance, and stable hash for optimizer training, controller, ordinary validation, or final holdout, with relationships proving pairwise disjointness.
- **Controller Boundary**: A reproducible point before or after a decision window containing the optimizer step, controller objective, evaluation status, and relationship to the preceding and next action.
- **Decision Window**: Exactly `h` optimizer steps assigned to one unchanged global or per-block action, or an explicitly incomplete terminal segment that produces no observation.
- **Controller Observation**: A completed window's action features, pre- and post-window objectives, normalized reward, predicted reward, prediction error, and link to its posterior transition and update.
- **Gaussian Belief**: The prior, predictive, and posterior means and covariances over reward coefficients, including observation noise, process noise, identity-transition provenance, validity state, and update history.
- **Posterior Sampling State**: The random state or equivalent deterministic seed provenance needed to reproduce coefficient samples, tie resolution, and subsequent actions.
- **Adaptive Checkpoint State**: The controller boundary, current action, partial-window progress, baseline objective, belief, sampling state, manifests, and compatibility data required for exact continuation.
- **Controller Metrics Summary**: Structured boundary records, action frequencies, uncertainty summaries, evaluation counts, terminal-window status, and final provenance for audit and comparison.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In validated probabilistic adaptive runs, 100% of emitted rewards are derived only from the fixed 128-example controller panel and the uniform objective over every resolved global granularity; no emitted reward includes a training-batch loss.
- **SC-002**: In 100% of validated Bayesian Thompson runs, optimizer-training, controller, ordinary-validation, and final-holdout manifests are pairwise disjoint, and insufficient data is rejected before training.
- **SC-003**: In 100% of completed controller observations, exactly the resolved positive decision-window length belongs to one unchanged action, and omission of the setting resolves to 50 optimizer steps.
- **SC-004**: Controlled finite reward sequences produce the mathematically expected change in posterior mean and covariance, and at least one controlled sequence changes a subsequent posterior Thompson preference relative to its initial preference.
- **SC-005**: A controlled additive per-block experiment learns different posterior preferences for at least two block/granularity effects without storing the complete scalar observation as each selected pair's independent reward mean.
- **SC-006**: For 100% of fresh and checkpoint-resumed validation cases matched on initial state, manifests, configuration, seeds, software, hardware, distributed topology, and deterministic runtime settings, manifests, controller window state, posterior-sampling state, sample count, and subsequent actions match exactly, while completed-window objectives, rewards, posterior means, and posterior covariances match with relative tolerance `1e-6` and absolute tolerance `1e-8`.
- **SC-007**: 100% of completed probabilistic adaptive runs record the controller, posterior, decision-window, data-manifest, strategy, sampling-randomness, and resume provenance required to audit observations and reproduce future choices.
- **SC-008**: In 100% of validated Bayesian Thompson training executions, the fixed 512-example final holdout is never evaluated or consulted before final comparison and remains available for that comparison.
- **SC-009**: In 100% of comparison checks, current Bayesian Thompson global and per-block, random global, random per-block, retained UCB, historical legacy heuristic Thompson, nested-all, and standalone runs remain behaviorally appropriate and distinguishable from structured artifacts; no new run executes the legacy heuristic Thompson path.
- **SC-010**: Across all validation cases for one or many arbitrary granularity labels and one or many transformer blocks, 100% of selected actions, posterior dimensions, deterministic ties, persistence records, and resumes resolve without relying on a four-label assumption.
- **SC-011**: In 100% of injected Bayesian Thompson controller-evaluation, non-finite-value, posterior-incompatibility, manifest-mismatch, and data-overlap failures, the run stops with an attributable error before an invalid posterior update or further action selection.
- **SC-012**: Every probabilistic adaptive controller boundary performs exactly one controller-objective evaluation, with the completed post-window value reused as the next pre-window baseline.

## Assumptions

- The discussion note's **Resolved experimental decisions** section is authoritative where earlier exploratory alternatives differ: uniform objective weights, 50-step default windows, a fixed 128-example controller panel, an additive per-block model, zero compute and switching costs, and a fixed 512-example final holdout.
- Existing dataset preparation can expose stable example identities and source provenance sufficient to create, hash, compare, and restore deterministic pairwise-disjoint manifests.
- Existing stable validation behavior defines target-token-weighted loss aggregation. Matched fresh/resume verification uses exact equality for manifests, discrete controller state, posterior-sampling state, sample count, and actions, and uses relative tolerance `1e-6` plus absolute tolerance `1e-8` for controller objectives, rewards, posterior means, and posterior covariances on the same documented software, hardware, distributed topology, and deterministic runtime settings; cross-hardware or cross-runtime equivalence is not assumed.
- The ordinary-validation set size remains governed by the existing experiment configuration; regardless of that requested size, the four roles plus at least one optimizer-training example must fit.
- Controller-panel, final-holdout, and posterior-sampling seeds come from independent named seed streams derived deterministically from the saved run root seed; every resolved seed and stream identity is saved before it is consumed.
- A positive-semidefinite covariance is considered a valid Gaussian belief or process distribution, including a degenerate zero process covariance; observation noise remains strictly positive.
- Posterior transition, conditioning, and sampling occur in one documented boundary order that is saved explicitly, making initial and exact-boundary resume behavior unambiguous.
- Final comparison is a post-training research action and does not retroactively affect the completed training run, controller, checkpoint choice, or hyperparameter decisions.
- Historical heuristic Thompson results remain available for reporting but not for new training or Bayesian resume; UCB, random, nested-all, standalone, correction, model-wiring, and unrelated reporting behavior remain covered by baseline regression expectations and available for comparison.
