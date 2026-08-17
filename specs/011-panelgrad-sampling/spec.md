# Feature Specification: PanelGrad Sampling

**Feature Branch**: `011-panelgrad-sampling`
**Created**: 2026-08-14
**Status**: Draft
**Input**: User description: "Add PanelGrad, a global elastic-training policy that periodically measures every granularity's heldout gradient RMS and samples one complete granularity per training step from the resulting categorical distribution."

## Clarifications

### Session 2026-08-14

- Q: How should the controller heldout set produce each granularity's gradient RMS score? → A: Compute the gradient of the aggregate target-token-weighted controller loss, then calculate its RMS.
- Q: Which gradient should determine each PanelGrad importance score? → A: Use the raw controller-loss gradient before membership correction.
- Q: In which model mode should PanelGrad measure controller gradients? → A: Use evaluation mode with gradients enabled, then restore the prior training mode.
- Q: Which parameters define the gradient vector and normalization count for a granularity? → A: Use only the parameters selected by the resolved granularity definitions across the FFN layers; exclude embeddings, attention, and all other granularity-independent parameters.
- Q: What defaults should PanelGrad use when its policy parameters are omitted? → A: Use `H=50`, `eta=1e-12`, `T=1`, and `epsilon=0.1`, while keeping all four configurable.
- Q: What numerical tolerance should PanelGrad use for gradient-score and probability-vector equivalence? → A: Use relative tolerance `1e-6` and absolute tolerance `1e-8`.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Sample global granularities from current importance (Priority: P1)

As a researcher, I can use PanelGrad to measure every complete global granularity at a common decision boundary and convert those comparable measurements into the distribution used to choose one granularity at each subsequent training step.

**Why this priority**: This is the complete scientific behavior under study. It tests whether whole-granularity gradient importance can guide elastic training without introducing architecture search or asynchronous score estimates.

**Independent Test**: Run a controlled experiment with known per-granularity gradients and sampling randomness, then verify the measured scores, normalized probabilities, and sequence of global actions.

**Acceptance Scenarios**:

1. **Given** a resolved set of global granularities and no adaptive warmup, **When** PanelGrad starts, **Then** it measures every granularity on the same controller panel and model state before selecting the first training action.
2. **Given** finite gradient RMS scores, **When** a refresh completes, **Then** the derived `q` and exploration-adjusted `p` vectors contain valid probabilities summing to one.
3. **Given** a frozen `p` and a decision interval of `H`, **When** the next `H` optimizer steps execute, **Then** each step independently samples exactly one complete global granularity from `Categorical(p)` and trains it through the ordinary training path.
4. **Given** `H` completed PanelGrad optimizer steps since the previous refresh, **When** the next action is required, **Then** all granularities are remeasured before that action is sampled.

---

### User Story 2 - Preserve data and optimization separation (Priority: P2)

As a researcher, I can trust that PanelGrad uses the existing heldout controller role only to make sampling decisions and that its measurement gradients never become training updates or contaminate final evaluation.

**Why this priority**: The experiment is interpretable only if controller measurements, optimizer training, ordinary validation, and final evaluation retain distinct roles.

**Independent Test**: Compare model, optimizer, scheduler, gradient, training-randomness, and data-role state immediately before and after a refresh, and verify that only PanelGrad measurement state changes.

**Acceptance Scenarios**:

1. **Given** a PanelGrad refresh, **When** all granularities have been measured, **Then** no model parameter, optimizer state, scheduler state, or training statistic has changed and no measurement gradient remains accumulated.
2. **Given** the existing controller, training, ordinary-validation, and final-holdout roles, **When** PanelGrad runs, **Then** only controller data influences its scores and the final holdout is never consulted during training.
3. **Given** an optional balanced-global warmup, **When** warmup finishes, **Then** PanelGrad performs a fresh full-panel measurement before its first adaptive action.
4. **Given** a run using any existing sampling strategy, **When** PanelGrad is added but not selected, **Then** the run's action selection and training behavior remain unchanged.

---

### User Story 3 - Reproduce and audit PanelGrad decisions (Priority: P3)

As a researcher, I can resume and inspect a PanelGrad run with enough state and diagnostics to reproduce its refresh boundaries, categorical samples, and granularity exposures.

**Why this priority**: Sampling-policy comparisons are useful only when their decisions, data provenance, randomness, and additional measurement cost are auditable.

**Independent Test**: Compare uninterrupted and checkpoint-resumed runs under equivalent deterministic conditions, and inspect their structured artifacts for identical boundaries, distributions, and subsequent actions.

**Acceptance Scenarios**:

1. **Given** equivalent initial state and deterministic runtime conditions, **When** one run is uninterrupted and another resumes inside a PanelGrad interval, **Then** both produce the same next refresh boundary and subsequent sampled actions.
2. **Given** a distributed run, **When** an action is sampled, **Then** all workers train the same complete global granularity and agree on the score and probability vectors.
3. **Given** a completed PanelGrad run, **When** its artifacts are inspected, **Then** every refresh and training action can be associated with its scores, probabilities, controller-panel provenance, exposure history, and measurement cost.

### Edge Cases

- With one resolved granularity, `q` and `p` are exactly `[1]`, while measurement and provenance behavior remain active.
- If every gradient RMS score is zero, positive `eta` yields a uniform `q` rather than an undefined distribution.
- With `epsilon = 1`, training samples uniformly regardless of the measured scores; with `epsilon = 0`, `p` equals `q`.
- A linear epsilon schedule may increase or decrease; it uses its start at the initial refresh, its endpoint at the duration, and clamps there beyond the duration.
- Warmup, failed optimizer attempts, and failed refreshes do not advance epsilon schedule progress.
- Invalid `H`, `eta`, `T`, or `epsilon`, an empty granularity set, or a granularity with zero active trainable scalars causes failure before affected training begins.
- A non-finite controller loss, gradient, score, or probability causes an attributable refresh failure and no new action is sampled.
- A refresh failure does not partially replace the last complete score or probability state.
- A run ending before the current interval reaches `H` steps records a partial interval without performing an unnecessary refresh.
- A checkpoint exactly at a refresh boundary records whether the refresh and next categorical sample have occurred so resume neither duplicates nor skips either event.
- Controller-panel provenance mismatch on resume causes failure before further training.
- Parameters selected by a resolved granularity definition are counted once in `N_g`; zero-valued gradients on those parameters remain part of the support rather than changing `N_g` by batch. Parameters outside the granularity-controlled FFN support are excluded.

## Requirements *(mandatory)*

### Functional Requirements

#### Method identity and scope

- **FR-001**: Researchers MUST be able to select PanelGrad as a distinct global-granularity sampling method through experiment configuration.
- **FR-002**: PanelGrad actions MUST always be one complete resolved global granularity applied consistently across the model; per-block, per-layer, per-projection, and mixed-profile choices MUST NOT be created.
- **FR-003**: PanelGrad MUST have explicit method identity and configuration provenance distinct from Thompson sampling, UCB, random sampling, and other existing strategies.
- **FR-004**: Selecting PanelGrad MUST NOT introduce a generic policy framework or alter the semantics of unselected sampling strategies.

#### Controller measurement

- **FR-005**: PanelGrad MUST reuse the dedicated, deterministic heldout controller-panel role established for adaptive training and MUST preserve its manifest and provenance.
- **FR-006**: Controller examples MUST remain disjoint from optimizer-training, ordinary-validation, and final-holdout examples; the final holdout MUST NOT be evaluated or consulted during PanelGrad training.
- **FR-007**: At each refresh, PanelGrad MUST measure every resolved global granularity against the same complete controller panel, current model parameters, and controller-loss definition in evaluation mode with gradient tracking enabled, then restore the model's prior training mode.
- **FR-008**: For each granularity `g`, PanelGrad MUST obtain the raw gradient of one aggregate target-token-weighted controller-panel loss with respect to only the parameters selected by the resolved granularity definition across all controlled FFN layers, before gradient clipping, optimizer transformations, or training-only gradient correction; the result MUST be invariant to controller microbatch partitioning within relative tolerance `1e-6` and absolute tolerance `1e-8`.
- **FR-009**: PanelGrad MUST calculate the complete granularity-controlled score

  $$
  I_g = \frac{\lVert d_g \rVert_2}{\sqrt{N_g}},
  $$

  where `d_g` contains the gradients for the complete trainable FFN parameter support selected by granularity `g` and `N_g` is the number of unique trainable scalars in that same support.
- **FR-010**: The definition of `N_g` MUST come from the resolved granularity definitions and remain stable across controller batches: selected parameter storage is counted once, selected scalars with numerical zero gradients remain counted, and embeddings, attention parameters, frozen parameters, and every other scalar outside the granularity-controlled FFN support are excluded.
- **FR-011**: Full-panel measurement MUST NOT update model parameters, optimizer state, scheduler state, mixed-precision state, training metrics, or the ordinary training-data position, and MUST leave no accumulated measurement gradients.
- **FR-012**: PanelGrad measurement MUST NOT perturb the random sequence used by ordinary training actions or training-batch computation beyond its explicitly separate PanelGrad sampling stream.

#### Probability construction and action lifecycle

- **FR-013**: PanelGrad MUST expose a positive integer refresh interval `H`, positive `eta`, positive temperature `T`, and either a scalar exploration mixture `epsilon` in `[0,1]` or a mutually exclusive linear `epsilon_schedule` with endpoints in `[0,1]` and positive integer duration; omitted values MUST resolve to `H=50`, `eta=1e-12`, `T=1`, and fixed `epsilon=0.1`, and invalid resolved values MUST fail before training.
- **FR-014**: At each refresh, PanelGrad MUST convert the complete contemporaneous score vector into

  $$
  q_g = \frac{(I_g+\eta)^{1/T}}
  {\sum_h (I_h+\eta)^{1/T}},
  \qquad
  p_g = (1-\epsilon)q_g + \frac{\epsilon}{K}.
  $$

- **FR-015**: PanelGrad MUST validate that `q` and `p` are finite, nonnegative vectors summing to one within relative tolerance `1e-6` and absolute tolerance `1e-8` before using them for sampling.
- **FR-016**: The `p` vector MUST be the categorical action distribution; `q` is an intermediate normalized score distribution and MUST NOT bypass the configured exploration mixture.
- **FR-017**: PanelGrad MUST evaluate scheduled epsilon only at refresh boundaries from the number of committed PanelGrad optimizer steps, then freeze epsilon, scores, `q`, and `p` for the next `H` completed PanelGrad optimizer steps.
- **FR-018**: At each ordinary PanelGrad optimizer step, the method MUST independently draw one action from `Categorical(p)` and train exactly that complete global granularity using the existing forward/backward and optimizer behavior.
- **FR-019**: PanelGrad MUST refresh before its first adaptive action and before the first action following each group of `H` completed PanelGrad optimizer steps.
- **FR-020**: An optional existing balanced-global warmup MAY precede PanelGrad; warmup steps MUST NOT consume the PanelGrad interval, and a full refresh MUST occur after warmup before adaptive sampling.
- **FR-021**: PanelGrad MUST reuse the existing completed-step scheduling semantics, global-action representation, and ordinary selected-action training path while keeping the lifecycle visibly ordered as refresh-if-due, sample, train, and record completion.
- **FR-022**: PanelGrad MUST NOT use Thompson posterior resets, delayed-reward acquisition episodes, fixed-action-per-window behavior, EMA, EXP3, or another partial-feedback estimator.
- **FR-023**: The selected training loss MUST be used without inverse-probability weighting or cost-aware score correction.
- **FR-024**: Measurement gradients MUST NOT be reused for the selected training update, including when the selected action matches a granularity just measured.

#### Distributed behavior, persistence, and diagnostics

- **FR-025**: In distributed execution, gradient squared norms and parameter counts MUST represent the complete distributed granularity-controlled FFN support, and all workers MUST agree on `I`, `q`, and `p` within relative tolerance `1e-6` and absolute tolerance `1e-8`.
- **FR-026**: Rank zero MUST own the authoritative categorical sampling stream and synchronize the selected global action so every worker trains the same action.
- **FR-027**: PanelGrad checkpoints MUST preserve the resolved epsilon schedule, current snapshot epsilon and schedule step, scores, `q`, `p`, categorical RNG state, completed steps since refresh, next refresh boundary, exposure counts, lifecycle phase, method parameters, and controller-panel provenance. Fixed-epsilon version-1 state MAY migrate to the current schema only for a fixed-policy resume.
- **FR-028**: Resume MUST reject missing, malformed, non-finite, method-incompatible, granularity-incompatible, or controller-provenance-incompatible PanelGrad state rather than silently reinitializing it.
- **FR-029**: Under equivalent deterministic runtime conditions, resume MUST neither repeat nor skip a refresh or action sample and MUST reproduce the subsequent discrete action sequence.
- **FR-030**: Every refresh record MUST include its completed-step boundary, active epsilon, epsilon schedule step, controller-panel identity, per-granularity active count, gradient norm, gradient RMS, `q`, `p`, entropy, probability extrema, duration or equivalent measurement-cost value, and success or failure state.
- **FR-031**: Every PanelGrad training-step record MUST identify the sampled granularity, its probability, cumulative per-granularity exposure, and whether balanced warmup or adaptive PanelGrad selection was active.
- **FR-032**: Shared artifacts MUST be written once by the authoritative process while remaining sufficient to audit distributed agreement.

### Research & Experiment Requirements

- **EX-001**: Every PanelGrad run MUST save the resolved granularity set, `H`, `eta`, `T`, fixed epsilon or schedule type/start/end/duration, seeds, method identity, data-role provenance, and numerical-tolerance assumptions.
- **EX-002**: Every PanelGrad run MUST write refresh and action diagnostics to structured artifacts rather than terminal output alone.
- **EX-003**: The first PanelGrad evaluation MUST compare against uniform global sampling at matched optimizer steps or target tokens.
- **EX-004**: PanelGrad's controller-measurement work MUST be reported separately so matched-step results are not misrepresented as matched-compute results.
- **EX-005**: The first evaluation MUST use ordinary selected-loss training and MUST NOT combine PanelGrad with EMA, EXP3, per-block decisions, inverse-probability weighting, or compute-aware sampling.

### Key Entities

- **PanelGrad configuration**: Method identity, ordered granularity set, refresh interval, score-normalization parameters, exploration mixture, seed identity, and optional warmup configuration.
- **Controller panel**: Fixed heldout examples and the manifest, selection provenance, and stable identity proving separation from other data roles.
- **Refresh state**: Boundary step, per-granularity controlled FFN support and gradient measurements, `q`, final categorical distribution `p`, measurement cost, and lifecycle phase.
- **Sampling state**: Authoritative categorical random state, current frozen distribution, interval progress, and cumulative granularity exposures.
- **PanelGrad action**: One complete global granularity selected from `Categorical(p)` for one ordinary optimizer step.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In controlled score cases, 100% of refreshes produce the expected `q` and `p` within relative tolerance `1e-6` and absolute tolerance `1e-8`, with every probability finite and nonnegative and each vector summing to one within the same tolerance.
- **SC-002**: In lifecycle tests, 100% of adaptive optimizer steps train exactly one complete global granularity sampled from the currently frozen `p`, and refreshes occur at exactly the configured completed-step boundaries.
- **SC-003**: Across measurement-isolation tests, 100% of refreshes leave model parameters, optimizer state, scheduler state, training-data position, and accumulated training gradients unchanged.
- **SC-004**: Under equivalent deterministic conditions, uninterrupted and resumed executions have identical controller-panel identity, boundary sequence, exposure counts, and subsequent sampled actions, with distributions equal within relative tolerance `1e-6` and absolute tolerance `1e-8`.
- **SC-005**: In distributed tests, 100% of workers agree on each selected action and on each refresh result within relative tolerance `1e-6` and absolute tolerance `1e-8`, and only one shared copy of each PanelGrad artifact is produced.
- **SC-006**: Existing non-PanelGrad strategy regression scenarios pass without changes to their expected sampling behavior.
- **SC-007**: Every completed research run produces a resolved configuration, controller-panel provenance, structured refresh diagnostics, structured action/exposure records, checkpointable PanelGrad state, and separately reported measurement cost.

## Assumptions

- The existing adaptive-training data split provides a fixed 128-example controller panel, ordinary validation data, and a reserved 512-example final holdout with pairwise-disjoint manifests; PanelGrad reuses these roles rather than defining new splits.
- PanelGrad reuses the existing completed-step interval concept; its documented omitted-value defaults are `H=50`, `eta=1e-12`, `T=1`, and `epsilon=0.1`, and every resolved value is saved with the run.
- Gradient RMS over the granularity-controlled FFN parameters is intended as an importance signal for the complete global granularity, not a guarantee of optimal sampling or estimator variance.
- The probability mapping is a deliberately simple first baseline: it preserves score order, supports temperature control, and becomes a valid categorical distribution through explicit normalization.
- A permanent nonzero exploration floor requires `epsilon > 0`; `epsilon = 0` remains valid for controlled ablations but provides no floor beyond positive `eta`.
- PanelGrad uses its own deterministic random stream so controller measurement and categorical action sampling do not disturb ordinary training randomness.
- Exact numerical equality across different hardware or distributed topologies is not assumed; exact-resume comparisons use the same software, hardware, topology, seeds, and deterministic runtime settings.
