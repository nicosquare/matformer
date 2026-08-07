# Episodic Reset for Probabilistic Adaptive Granularity

Date: 2026-08-07

Status: design proposal; not implemented or experimentally validated.

## Motivation

The current Bayesian controller uses a random-walk transition

$$
V_t^- = V_{t-1} + Q
$$

to prevent old observations from becoming permanently certain. Calibration
experiments show that fixed $Q$ is also compensating for the changing training
reward scale. Its useful value is therefore coupled to the observation
variance, scheduler phase, warmup policy, and training horizon.

An episodic controller replaces continuous covariance injection with an
explicit memory horizon. Evidence is accumulated without process noise for a
fixed episode and then deliberately discarded at a reproducible boundary.
This does not claim that the environment is stationary; it makes the chosen
stationarity interval visible and auditable.

## Proposed method

Start with global adaptation only. Preserve the existing 500-step balanced
pre-adaptive warmup and 50-step controller decision interval. After warmup:

1. Divide training into controller episodes of $K$ optimizer steps.
2. Set $Q=0$ within every episode.
3. At an episode boundary, finish and journal the preceding controller
   observation, then reset the Gaussian posterior to the configured prior.
4. Run one seeded permutation of all granularities as forced global actions.
   These are controller observations: each completed window updates the fresh
   posterior even though Thompson sampling did not choose its action.
5. Use ordinary Thompson sampling for the remaining windows in the episode.
6. Repeat until the token or step budget terminates training.

The initial reference should use

$$
h=50,
\qquad
K=1000,
\qquad
|\mathcal G|=5.
$$

Each episode then contains 20 windows: five balanced acquisition windows and
15 Thompson-selected windows. The reset interval must be an integer multiple
of $h$ and should contain at least two complete passes over the action set so
that forced acquisition does not consume the entire episode.

## Reset semantics

The first experiment should use a **full controller reset**:

$$
m \leftarrow m_0,
\qquad
V \leftarrow V_0.
$$

Resetting only the covariance while retaining the posterior mean is a distinct
method. It preserves possibly stale action preferences and should be evaluated
later, not silently mixed into the first experiment.

The reset affects only controller belief. It must not reset or reconstruct the
language model, optimizer, scheduler, scaler, dataloader, global step, token
counters, controller panel, data manifests, or controller random generator.
The fixed-panel objective already evaluated at the reset boundary becomes the
baseline for the first acquisition window, so no redundant panel evaluation is
needed.

At a boundary shared by a completed window and a reset, operations occur in
this order:

1. evaluate the fixed controller panel;
2. finish the preceding reward and posterior update;
3. journal the completed episode;
4. reset the posterior;
5. initialize the seeded balanced acquisition schedule;
6. select the first forced action.

This ordering preserves observation accounting even though the just-updated
posterior is immediately archived and replaced.

## Relationship to balanced warmup

The initial 500-step warmup and the per-episode acquisition pass have different
contracts:

| Property | Pre-adaptive warmup | Episode acquisition pass |
| --- | --- | --- |
| Trains the model | Yes | Yes |
| Action source | Seeded balanced schedule | Seeded balanced schedule |
| Controller panel evaluated | No | At every boundary |
| Observation emitted | No | Yes |
| Posterior updated | No | Yes, with $Q=0$ |
| Counts as adaptive action frequency | No | Report separately |

The acquisition pass guarantees recent evidence for every granularity after a
reset. It also prevents a broad reset prior from immediately starving an
action due only to the first random samples.

## Configuration sketch

```yaml
training:
  pre_nested_warmup:
    enabled: true
    duration: 500
    unit: steps
    policy: balanced_global
    action_interval_steps: 50

model:
  adaptive_controller:
    decision_interval_steps: 50
    process_noise_covariance: 0.0
    reset:
      enabled: true
      interval_steps: 1000
      policy: full_prior
      acquisition_policy: balanced_global
      acquisition_passes: 1
```

The final names should follow the existing configuration conventions. A scalar
zero process covariance must resolve to a correctly sized zero matrix for both
global and per-block feature schemas.

## Reproducibility and persistence

Add a named seed stream such as `controller_reset_schedule`. Derive one local
permutation for every episode; do not reuse or rewind the Thompson generator.
Persist and validate:

- reset policy and interval;
- episode index, start step, intended end step, and within-episode offset;
- reset count and exact reset steps;
- acquisition schedule, seed, hash, current window, and action counts;
- posterior snapshot immediately before each reset;
- fresh prior state immediately after each reset;
- whether the current action is forced or Thompson-selected;
- all existing controller phase, RNG, observation, and window state.

Resume inside an acquisition or adaptive window must exactly reproduce the
uninterrupted episode, next reset, future permutations, and Thompson actions.

Suggested journal events are:

- `controller_episode_initialized`;
- `controller_posterior_reset`;
- `controller_acquisition_window_completed`;
- `controller_acquisition_completed`;
- `controller_episode_completed`;
- `controller_episode_terminal_incomplete`.

Summaries should report forced and Thompson action frequencies separately.
Both kinds are valid controller observations, but only Thompson-selected
windows measure the learned policy's action frequency.

## Computational cost

The posterior reset and permutation are negligible dense operations on the
existing small controller state. No extra fixed-panel evaluation is required
when resets align with ordinary controller boundaries. The method therefore
adds no forward-pass overhead relative to the current post-warmup controller.

Forced acquisition changes which subnetworks receive training updates, but it
does not add optimizer steps. Its cost is an exploration-allocation tradeoff,
not additional computation.

## Initial experiment

Use seed 42 and compare four otherwise matched 20M-token runs:

1. static posterior: $Q=0$, no reset;
2. episodic reset with $K=500$ steps;
3. episodic reset with $K=1000$ steps;
4. episodic reset with $K=2000$ steps.

Keep the 500-step balanced warmup, $h=50$, prior, observation variance, data
manifests, optimizer, and scheduler fixed. The current fixed-$Q$ balanced run
is a historical reference rather than another required submission.

The primary diagnostics are:

- ordinary validation loss at every granularity;
- action starvation and normalized entropy, separating acquisition from
  Thompson windows;
- action frequencies by episode and training phase;
- innovation bias and deviation within each episode;
- posterior action-score separation immediately before reset;
- the fraction of each episode spent in forced acquisition;
- exact reset and checkpoint-resume reproducibility.

Replicate seeds 43 and 44 only if one interval improves stability or action
selection without making the method effectively equivalent to uniform blocked
training.

## Expected tradeoffs

Benefits:

- eliminates process-noise covariance and its continuous interaction with
  posterior contraction;
- makes controller memory an explicit number of optimizer steps;
- prevents arbitrarily old evidence from dominating late training;
- requires no model of the learning curve and no additional panel compute;
- provides simple episode-level artifacts and ablations.

Risks:

- replaces $Q$ with a reset-interval choice rather than eliminating all
  hyperparameters;
- discards useful action evidence at every full reset;
- creates discontinuous policy uncertainty at episode boundaries;
- can spend too much training on forced acquisition when $K$ is short;
- still depends on the prior covariance and observation variance;
- assumes action effects are useful within an episode even if the common reward
  scale continues changing inside it.

Periodic reset is therefore best treated as a simpler nonstationarity model,
not a parameter-free controller. Its value is that its failure mode and memory
horizon are substantially easier to interpret than a fixed $Q$.

## Acceptance criteria for an implementation

- No process covariance is added between observations.
- Resets occur only at complete controller boundaries.
- The boundary evaluation is reused rather than repeated.
- Every acquisition pass selects each granularity exactly once per pass.
- Acquisition observations update the posterior but are distinguishable from
  Thompson selections in artifacts and statistics.
- Model-training state and counters remain continuous across every reset.
- Fresh and mid-window-resumed runs produce identical reset schedules,
  posterior transitions, and subsequent Thompson actions.
- Distributed ranks agree on reset state and action; only rank zero writes the
  shared journal.
