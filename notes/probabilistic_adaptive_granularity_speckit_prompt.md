# Spec Kit Prompt: Bayesian Adaptive Granularity Sampling

Suggested short name: bayesian-adaptive-sampling

Use the following as the feature description passed to /speckit-specify.

---

Implement probabilistic adaptive granularity sampling for MatFormer training.
Ground the specification in
/l/users/nicolas.avila/dev/references/matformer/notes/probabilistic_adaptive_granularity_discussion_2026-08-05.md.
Treat that note, especially its Resolved experimental decisions section, as the
source of truth. This feature introduces an explicit probabilistic controller
as the successor to reward-insensitive adaptive behavior while preserving
existing baselines for comparison.

## Problem

The current adaptive per-block path is heuristic rather than Bayesian. It
compares consecutive training-batch losses, confounds batch difficulty with
profile utility, assigns one scalar reward independently to every selected
block action, and penalizes complete-profile changes without evidence that
switching is harmful. Its current default reward-mean update can also leave
future decisions completely unaffected by observed rewards.

Researchers need an adaptive sampler that:

- learns an explicit probability distribution over expected rewards;
- updates that distribution from heldout reward observations;
- measures the training utility of a selected action rather than its immediate
  training-batch loss;
- supports both whole-model and per-block adaptive actions;
- remains reproducible, resumable, inspectable, and distinguishable from the
  existing heuristic and random baselines.

## Scope and priority

Structure the feature as two independently testable stages within one
specification.

### P1: Bayesian global adaptation

Provide an adaptive global scope whose action is one resolved granularity
applied to every transformer block. This is the minimum viable feature and must
validate the controller dataset, delayed reward, posterior learning, Thompson
selection, persistence, and resume behavior before per-block adaptation is
required.

### P2: Bayesian additive per-block adaptation

Reuse the same controller and reward protocol with an action containing one
resolved granularity per transformer block. Model the expected complete-profile
reward through additive block/granularity contributions. Do not enumerate every
possible complete profile and do not assign the full scalar reward separately
to each selected block/granularity pair.

### P3: Research provenance and comparison safety

Make probabilistic global, probabilistic per-block, random global, random
per-block, and the existing heuristic adaptive behavior unambiguously
distinguishable from saved artifacts. Preserve existing baseline behavior and
do not silently reinterpret old run configurations or artifacts as genuine
Bayesian Thompson sampling.

## Controller data and objective

Create four pairwise-disjoint data roles:

- optimizer training data;
- a controller panel used only for adaptive rewards;
- ordinary validation data used for existing checkpoint selection and
  monitoring;
- an untouched final holdout used only for final comparisons.

Every adaptive run must use the same deterministic controller panel of 128
examples at every decision boundary. Save its seed, selected-example manifest,
source provenance, and stable hash. No controller example may be used for
optimizer updates, ordinary validation, or final evaluation.

Reserve a separate deterministic final holdout of 512 examples before training.
Save its independent seed, manifest, source provenance, and stable hash. It
must never influence controller updates, checkpoint selection, hyperparameter
selection, or other during-training decisions. Evaluate it only as part of
final run comparison.

Fail before training when the available dataset cannot provide the requested
pairwise-disjoint sets plus at least one usable training example.

For controller parameters theta, resolved granularity set G, and fixed
controller panel H_c, define the controller objective as the uniform average
heldout loss over every global granularity:

$$
J_{H_c}(\theta)
=\frac{1}{|\mathcal G|}
\sum_{g\in\mathcal G}
L_{H_c}\left(\theta;(g,\ldots,g)\right).
$$

The objective must use the repository's stable target-token-weighted validation
loss semantics and evaluate every resolved granularity in deterministic order.
It must not use training-batch loss.

## Decision windows and reward

The decision-window length must be a configurable positive number of optimizer
steps, with 50 steps as the default. Let the resolved window length be h. Keep
the selected global granularity or per-block profile fixed throughout the
window.

At the initial boundary, evaluate and save the controller objective. At every
subsequent completed boundary, evaluate it again and reward the action from the
preceding window:

$$
r_t
=\frac{J_{H_c}(\theta_t)-J_{H_c}(\theta_{t+1})}{h}.
$$

The post-window objective is the pre-window baseline for the next decision, so
only one controller evaluation is needed per boundary.

Checkpoints created inside a decision window must preserve the selected action,
the number of completed steps in that window, its pre-window controller
objective, and all controller state. A resumed run must finish the same window
before producing its reward or selecting another action. A run that terminates
with an incomplete window must not silently treat it as a completed
observation.

Controller evaluation failure or a non-finite objective/reward must not update
the posterior and must fail with a clear, attributable error.

## Probabilistic learning behavior

Use a Gaussian Bayesian linear reward model:

$$
r_t=x_t^\top\beta_t+\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,\sigma^2).
$$

The initial context is intercept-only, z_t = [1]. Training progress,
learning-rate context, controller slopes, action interactions, and learned
switching features are follow-up work outside this starting feature.

Maintain an explicit Gaussian belief over beta with posterior mean and
covariance. Before each controller round, use an identity state transition:

$$
F=I,
\qquad
m_t^-=m_{t-1},
\qquad
V_t^-=V_{t-1}+Q.
$$

Process noise Q expresses nonstationarity by restoring uncertainty without
shrinking expected coefficients toward zero. Prior parameters, observation
noise, and process noise must be explicit, valid, reproducible run inputs and
must be saved in provenance. The specification should require finite values
and covariance/noise settings that define valid probability distributions,
while leaving source-code organization and numerical linear-algebra choices to
planning.

Select actions using genuine posterior Thompson sampling:

$$
\widetilde\beta_t\sim\mathcal N(m_t^-,V_t^-),
\qquad
A_t=\arg\max_{A\in\mathcal A}x(A)^\top\widetilde\beta_t.
$$

After observing reward r_t, condition the Gaussian belief on that observation:

$$
k_t
=\frac{V_t^-x_t}
{x_t^\top V_t^-x_t+\sigma^2},
$$

$$
m_t=m_t^-+k_t(r_t-x_t^\top m_t^-),
$$

$$
V_t=V_t^--k_t x_t^\top V_t^-.
$$

For global adaptation, the action features distinguish the resolved
granularity arms. For per-block adaptation, use identifiable additive
block/granularity features so evidence is inferred from varied complete
profiles. The behavior must support arbitrary valid resolved granularity labels
and counts rather than restoring a hard-coded four-granularity assumption.

Both compute cost and switching cost are exactly zero in this feature:

$$
\lambda_{\mathrm{compute}}
=\lambda_{\mathrm{switch}}
=0.
$$

Do not apply the current Hamming-distance profile-change penalty and do not add
an implicit size penalty.

## Configuration, state, and artifacts

Researchers must be able to select probabilistic adaptation without changing
code and must explicitly choose global or per-block scope. Invalid scope,
strategy, data-split, prior, noise, interval, or resume combinations must fail
before training where possible.

Every adaptive checkpoint and completed-run artifact must contain enough
information to reproduce and audit future choices, including:

- resolved adaptive scope and strategy;
- resolved ordered granularity set;
- controller and final-holdout manifests and hashes;
- decision-window length, current action, boundary step, and partial-window
  progress;
- controller objective before and after completed windows;
- emitted reward and reward prediction error;
- posterior mean and covariance before and after updates;
- prior, observation-noise, process-noise, and identity-transition provenance;
- posterior-sampling random state or equivalent deterministic seed provenance;
- action frequencies and uncertainty summaries;
- resume provenance.

Stepwise controller metrics and completed-run summaries must be available in
structured artifacts, not only console logs.

Given the same initial state, data manifests, configuration, and seeds, a fresh
or resumed run must reproduce the same controller boundary values, posterior
updates, and sampled actions, subject only to existing documented
reproducibility limits.

## Compatibility

Preserve existing model wiring, granularity resolution, correction behavior,
random global sampling, random per-block sampling, nested-all behavior,
standalone behavior, reporting, and non-adaptive checkpoint semantics.

Keep the existing heuristic adaptive implementation available as an explicitly
identified comparison baseline. Never label its Gaussian score perturbations
as a Bayesian posterior. Existing artifacts must remain interpretable without
migration.

## Required edge cases

Cover at least:

- the initial boundary before any reward exists;
- fewer available examples than required for disjoint data roles;
- arbitrary explicit granularity sets and labels;
- one granularity only;
- one transformer block only;
- checkpoints and resumes exactly at and inside controller boundaries;
- a run ending in an incomplete decision window;
- missing or incompatible posterior state on resume;
- non-finite controller losses, rewards, means, or covariance values;
- zero process noise and positive process noise;
- deterministic ties between candidate actions;
- controller or final-holdout manifest mismatch on resume;
- accidental attempts to use controller or final examples for another role.

## Out of scope

Exclude:

- adjacent-block or higher-order action interactions;
- learned progress, learning-rate, loss-slope, or neural context models;
- full reinforcement learning, policy networks, replay buffers, and delayed
  credit across multiple different actions;
- compute penalties, switching penalties, learned switching effects, or hard
  compute budgets;
- fresh or rotating controller panels;
- using training-batch losses as controller rewards;
- deployment routing;
- claiming scientific superiority from smoke tests alone;
- redesigning MatFormer FFN mathematics, correction mechanisms, or unrelated
  training modes.

## Measurable outcomes

The resulting specification must make the following outcomes independently
verifiable:

- 100% of adaptive rewards in validated runs are derived only from the fixed
  128-example controller panel and the uniform all-granularity objective.
- 100% of controller, ordinary-validation, final-holdout, and training manifests
  are pairwise disjoint in validated runs.
- 100% of completed controller observations own exactly the resolved positive
  decision-window length, which defaults to 50 optimizer steps when omitted.
- Controlled reward sequences change posterior means and uncertainty, and
  posterior Thompson choices respond to those changes.
- A controlled additive per-block experiment can learn different preferences
  for at least two block/granularity effects without assigning the same reward
  mean independently to every chosen pair.
- Fresh and checkpoint-resumed executions with matching inputs produce matching
  completed-window rewards, posterior state, and subsequent actions.
- Every completed adaptive run records the full controller, posterior,
  decision-window, data-manifest, and strategy provenance needed for audit and
  resume.
- The untouched 512-example final holdout is never evaluated or consulted
  during training and is available for final comparison.
- Existing random, heuristic, nested-all, and standalone baselines remain
  behaviorally unchanged and distinguishable from the new probabilistic modes.

Do not prescribe programming-language constructs, module names, class names,
or file-level implementation details in the specification. Express the
mathematical behavior above as research requirements because it defines the
experimental method. Record reasonable assumptions instead of introducing
unnecessary clarification markers.

---
