# Episodic Reset for Probabilistic Adaptive Granularity

Date: 2026-08-07

Status: implemented; three-seed pre-experiments completed; $K=2000$ selected
for confirmatory final-holdout evaluation.

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

## Pre-Experiment 1: seed-42 reset-interval sweep

Use the existing d_model=256 Slurm entrypoint. These jobs retain the tuned
20M-token calibration setup: concat + GMC, seed 42, 400 scheduler-warmup
steps, the 500-step balanced pre-adaptive warmup, AdamW through the `adam`
preset, learning rate 0.001, the five explicit granularities, $h=50$,
$V_0=\operatorname{diag}(10^{-4},10^{-6},10^{-6},10^{-6},10^{-6})$,
and observation variance $10^{-7}$. The only controller-method change from
that setup is $Q=0$ plus the reset policy and interval. Request exactly one
GPU because reset-enabled Bayesian Thompson remains single-process only.

Set the shared paths and explicit granularity map once from the repository
root:

```bash
export OUT=/nfs-stor/nicolas.avila/results/elasticnn
repo=/l/users/nicolas.avila/dev/references/matformer
script="$repo/scripts/slurm_dmodel256_pilot.sh"
labels='[micro,small,medium,large,full]'
prefixes='{micro: 0.125, small: 0.25, medium: 0.5, large: 0.75, full: 1.0}'

cd "$repo"
mkdir -p logs
```

First submit the matched $Q=0$ no-reset control:

```bash
sbatch --gres=gpu:1 --time=04:00:00 "$script" \
  --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random \
  --run-id d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-noreset-pc1p0em6-s42 \
  --override run.seed=42 \
  --override training.token_budget=20000000 \
  --override training.warmup_steps=400 \
  --override training.pre_nested_warmup.enabled=true \
  --override training.pre_nested_warmup.duration=500 \
  --override training.pre_nested_warmup.unit=steps \
  --override training.pre_nested_warmup.policy=balanced_global \
  --override training.pre_nested_warmup.action_interval_steps=50 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override dataset.sample_limit=100000 \
  --override model.variant=concat \
  --override model.correction_mode=gmc \
  --override model.membership_correction=true \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=adaptive_global \
  --override model.adaptive_sampler_strategy=thompson \
  --override model.adaptive_controller.preset=bayesian_thompson \
  --override model.adaptive_controller.decision_interval_steps=50 \
  --override 'model.adaptive_controller.prior_mean=[0.0,0.0,0.0,0.0,0.0]' \
  --override 'model.adaptive_controller.prior_covariance=[1.0e-4,1.0e-6,1.0e-6,1.0e-6,1.0e-6]' \
  --override model.adaptive_controller.observation_noise_variance=1.0e-7 \
  --override model.adaptive_controller.process_noise_covariance=0.0 \
  --override model.adaptive_controller.reset.enabled=false
```

Submit the full-prior reset variant with $K=500$:

```bash
sbatch --gres=gpu:1 --time=04:00:00 "$script" \
  --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random \
  --run-id d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-k500-pc1p0em6-s42 \
  --override run.seed=42 \
  --override training.token_budget=20000000 \
  --override training.warmup_steps=400 \
  --override training.pre_nested_warmup.enabled=true \
  --override training.pre_nested_warmup.duration=500 \
  --override training.pre_nested_warmup.unit=steps \
  --override training.pre_nested_warmup.policy=balanced_global \
  --override training.pre_nested_warmup.action_interval_steps=50 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override dataset.sample_limit=100000 \
  --override model.variant=concat \
  --override model.correction_mode=gmc \
  --override model.membership_correction=true \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=adaptive_global \
  --override model.adaptive_sampler_strategy=thompson \
  --override model.adaptive_controller.preset=bayesian_thompson \
  --override model.adaptive_controller.decision_interval_steps=50 \
  --override 'model.adaptive_controller.prior_mean=[0.0,0.0,0.0,0.0,0.0]' \
  --override 'model.adaptive_controller.prior_covariance=[1.0e-4,1.0e-6,1.0e-6,1.0e-6,1.0e-6]' \
  --override model.adaptive_controller.observation_noise_variance=1.0e-7 \
  --override model.adaptive_controller.process_noise_covariance=0.0 \
  --override model.adaptive_controller.reset.enabled=true \
  --override model.adaptive_controller.reset.interval_steps=500 \
  --override model.adaptive_controller.reset.policy=full_prior \
  --override model.adaptive_controller.reset.acquisition_policy=balanced_global \
  --override model.adaptive_controller.reset.acquisition_passes=1
```

Submit the full-prior reset variant with $K=1000$:

```bash
sbatch --gres=gpu:1 --time=04:00:00 "$script" \
  --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random \
  --run-id d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-k1000-pc1p0em6-s42 \
  --override run.seed=42 \
  --override training.token_budget=20000000 \
  --override training.warmup_steps=400 \
  --override training.pre_nested_warmup.enabled=true \
  --override training.pre_nested_warmup.duration=500 \
  --override training.pre_nested_warmup.unit=steps \
  --override training.pre_nested_warmup.policy=balanced_global \
  --override training.pre_nested_warmup.action_interval_steps=50 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override dataset.sample_limit=100000 \
  --override model.variant=concat \
  --override model.correction_mode=gmc \
  --override model.membership_correction=true \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=adaptive_global \
  --override model.adaptive_sampler_strategy=thompson \
  --override model.adaptive_controller.preset=bayesian_thompson \
  --override model.adaptive_controller.decision_interval_steps=50 \
  --override 'model.adaptive_controller.prior_mean=[0.0,0.0,0.0,0.0,0.0]' \
  --override 'model.adaptive_controller.prior_covariance=[1.0e-4,1.0e-6,1.0e-6,1.0e-6,1.0e-6]' \
  --override model.adaptive_controller.observation_noise_variance=1.0e-7 \
  --override model.adaptive_controller.process_noise_covariance=0.0 \
  --override model.adaptive_controller.reset.enabled=true \
  --override model.adaptive_controller.reset.interval_steps=1000 \
  --override model.adaptive_controller.reset.policy=full_prior \
  --override model.adaptive_controller.reset.acquisition_policy=balanced_global \
  --override model.adaptive_controller.reset.acquisition_passes=1
```

Submit the full-prior reset variant with $K=2000$:

```bash
sbatch --gres=gpu:1 --time=04:00:00 "$script" \
  --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random \
  --run-id d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-k2000-pc1p0em6-s42 \
  --override run.seed=42 \
  --override training.token_budget=20000000 \
  --override training.warmup_steps=400 \
  --override training.pre_nested_warmup.enabled=true \
  --override training.pre_nested_warmup.duration=500 \
  --override training.pre_nested_warmup.unit=steps \
  --override training.pre_nested_warmup.policy=balanced_global \
  --override training.pre_nested_warmup.action_interval_steps=50 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override dataset.sample_limit=100000 \
  --override model.variant=concat \
  --override model.correction_mode=gmc \
  --override model.membership_correction=true \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=adaptive_global \
  --override model.adaptive_sampler_strategy=thompson \
  --override model.adaptive_controller.preset=bayesian_thompson \
  --override model.adaptive_controller.decision_interval_steps=50 \
  --override 'model.adaptive_controller.prior_mean=[0.0,0.0,0.0,0.0,0.0]' \
  --override 'model.adaptive_controller.prior_covariance=[1.0e-4,1.0e-6,1.0e-6,1.0e-6,1.0e-6]' \
  --override model.adaptive_controller.observation_noise_variance=1.0e-7 \
  --override model.adaptive_controller.process_noise_covariance=0.0 \
  --override model.adaptive_controller.reset.enabled=true \
  --override model.adaptive_controller.reset.interval_steps=2000 \
  --override model.adaptive_controller.reset.policy=full_prior \
  --override model.adaptive_controller.reset.acquisition_policy=balanced_global \
  --override model.adaptive_controller.reset.acquisition_passes=1
```

The launcher prints the resolved preflight before starting each job. Confirm
that the reset runs resolve as `adaptive_global + thompson`, process covariance
zero, reset enabled with the requested interval, and one acquisition pass.
Re-submitting the same command and run ID resumes from that run's latest
checkpoint because continuation remains enabled in the d_model=256 pilot
configuration.

## Initial experiment results

Date analyzed: 2026-08-10

Results directory:
`/nfs-stor/nicolas.avila/results/elasticnn/matformer_llama_148m_20m_tokens`

### Run integrity and comparison controls

All four seed-42 jobs completed from scratch without a resume or recorded
failure. Each job reached 4,883 optimizer steps, exactly 20,000,000 planned
tokens, and 10,576,344 non-padding content tokens. Ordinary validation selected
the final step for every run, so checkpoint timing does not confound the final
comparison.

The optimizer-training, ordinary-validation, controller-panel, and final-
holdout manifest hashes match across all four jobs. The ten-window balanced
warmup schedule also matches exactly, selects every granularity twice, leaves
the prior untouched, and produces the same initial controller objective,

$$
J_{500}=6.4112006864.
$$

Every run records 88 controller-panel evaluations, 87 complete controller
observations, and one terminal 33-step partial window with no observation. The
reset jobs therefore add no controller-panel evaluations. Their boundary
journals record the required order

$$
\text{completed window}
\rightarrow
\text{episode completed}
\rightarrow
\text{posterior reset}
\rightarrow
\text{next episode initialized}.
$$

The observed reset counts are eight, four, and two for $K=500$, $K=1000$,
and $K=2000$, respectively. Every completed episode contains one forced use of
each granularity, and every reset restores the configured zero mean and exact
diagonal prior covariance. Total wall-clock times are 2,191--2,208 seconds
(36.5--36.8 minutes), a spread below 0.8%; no reset-related runtime overhead is
measurable at this resolution.

The results below use the 512-example ordinary-validation panel. The untouched
final holdout has been reserved consistently but has not yet been evaluated;
there is no `final_holdout_results.json` in these run directories.

### Final ordinary-validation loss

For the five granularities $\mathcal G$, define the primary uniform aggregate
as

$$
\overline{L}
=
\frac{1}{|\mathcal G|}
\sum_{g\in\mathcal G} L_g,
\qquad
|\mathcal G|=5,
$$

and compare each reset interval with the matched $Q=0$ no-reset control using

$$
\Delta\overline{L}_K
=
\overline{L}_K-\overline{L}_{\mathrm{no\ reset}}.
$$

Lower values are better.

| Controller | Micro | Small | Medium | Large | Full | $\overline{L}$ | $\Delta\overline{L}$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| $Q=0$, no reset | 5.243622 | **5.168346** | **5.163630** | 5.162705 | 5.164691 | 5.180599 | -- |
| Reset, $K=500$ | **5.218363** | 5.181292 | 5.170834 | 5.167893 | 5.167898 | 5.181256 | +0.000657 |
| Reset, $K=1000$ | 5.223222 | 5.179508 | 5.166702 | 5.164188 | 5.164302 | 5.179584 | -0.001015 |
| Reset, $K=2000$ | 5.227641 | 5.179323 | 5.163740 | **5.161442** | **5.162078** | **5.178845** | **-0.001754** |

$K=2000$ has the lowest seed-42 uniform mean, followed by $K=1000$.
Nevertheless, the reductions relative to no reset are only 0.034% and 0.020%,
respectively. One deterministic training seed supplies no estimate of
between-seed variability, so these differences are not evidence that reset
improves expected validation loss.

All reset variants improve the micro loss by 0.0160--0.0253 but worsen the
small loss by 0.0110--0.0129. More frequent acquisition increasingly shifts
training exposure toward micro and away from the small-heavy no-reset policy.
For $K=500$, the micro improvement is offset by worse medium, large, and full
losses, leaving its uniform mean 0.000657 worse than no reset. $K=2000$
retains most of the micro improvement while avoiding the degradation at large
and full.

The final controller-panel objective at step 4,850 tells the same aggregate
story: 5.147893 for no reset, 5.147580 for $K=500$, 5.146902 for $K=1000$,
and 5.145528 for $K=2000$. This agreement is useful, but it is not an
independent replication because the controller and ordinary-validation panels
observe the same trained model.

### Forced acquisition and learned-policy allocation

For action counts $n_g$ over a selected subset of windows, report normalized
entropy

$$
H_{\mathrm{norm}}
=
-\frac{1}{\log |\mathcal G|}
\sum_{g\in\mathcal G}
\frac{n_g}{N}
\log\!\left(\frac{n_g}{N}\right).
$$

$H_{\mathrm{norm}}=1$ is uniform. Counts below are ordered as micro / small /
medium / large / full. The Thompson-only columns measure the learned policy;
the total columns include forced acquisition.

| Controller | Forced windows | Thompson-only counts | Thompson $H_{\mathrm{norm}}$ | Total counts | Total $H_{\mathrm{norm}}$ |
| --- | ---: | --- | ---: | --- | ---: |
| $Q=0$, no reset | 0 / 87 (0.0%) | 11 / 45 / 9 / 12 / 10 | 0.844 | 11 / 45 / 9 / 12 / 10 | 0.844 |
| Reset, $K=500$ | 45 / 87 (51.7%) | 11 / 13 / 9 / 1 / 8 | 0.900 | 20 / 22 / 18 / 10 / 17 | 0.981 |
| Reset, $K=1000$ | 25 / 87 (28.7%) | 12 / 17 / 16 / 8 / 9 | **0.973** | 17 / 22 / 21 / 13 / 14 | **0.987** |
| Reset, $K=2000$ | 15 / 87 (17.2%) | 13 / 16 / 23 / 9 / 11 | 0.966 | 16 / 19 / 26 / 12 / 14 | 0.977 |

The no-reset controller selects small in 45 of 87 windows, or 51.7% of all
adaptive windows. Both $K=1000$ and $K=2000$ materially improve
Thompson-only diversity without relying primarily on forced actions. They also
avoid Thompson-only action starvation: their least-selected actions still
receive eight and nine windows.

$K=500$ is qualitatively different. In a complete episode, the acquisition
fraction is

$$
f_{\mathrm{acq}}
=
\frac{|\mathcal G|}{K/h}
=
\frac{250}{K}.
$$

It is therefore 50%, 25%, and 12.5% for $K=500$, $K=1000$, and $K=2000$.
The observed fractions are slightly higher because the run ends seven complete
windows into a new episode, after all five acquisition windows have already
run. At $K=500$, more than half of all observed windows are forced and the
Thompson subset selects large only once. Its high total entropy therefore
overstates learned-policy balance and makes this interval close to uniform
blocked training for much of the run.

### Predictive calibration

For completed window $t$ with action feature $x_t$, predictive coefficient
mean $m_t^-$, predictive covariance $V_t^-$, observed per-step reward $r_t$,
and observation variance $\sigma^2$, define the standardized innovation

$$
z_t
=
\frac{r_t-x_t^\top m_t^-}
{\sqrt{x_t^\top V_t^-x_t+\sigma^2}}.
$$

A calibrated Gaussian observation model should have innovation mean near zero,
standard deviation near one, and approximately 95% of values in
$[-1.96,1.96]$.

| Controller | All-window mean $z$ | All-window sd $z$ | Thompson mean $z$ | Thompson sd $z$ | Thompson 95% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| $Q=0$, no reset | -0.785 | 0.454 | -0.785 | 0.454 | 98.9% |
| Reset, $K=500$ | **-0.115** | 0.401 | **-0.221** | 0.537 | 100.0% |
| Reset, $K=1000$ | -0.235 | 0.483 | -0.333 | 0.534 | 98.4% |
| Reset, $K=2000$ | -0.421 | 0.532 | -0.509 | 0.539 | 98.6% |

The no-reset posterior retains an increasingly stale positive reward level as
training improvements slow, so observed rewards are systematically below its
predictions. Resetting shortens this lag: innovation bias moves monotonically
toward zero as $K$ decreases. This is direct evidence that the explicit memory
horizon addresses the nonstationary common reward level that motivated the
method.

It does not solve calibration. Every Thompson-only standard deviation is near
0.54 or lower and coverage is 98.4--100%, so the predictive distribution
remains over-dispersed. Forced-acquisition innovations have means between
-0.015 and 0.007 and standard deviations of only 0.155--0.181. The freshly
restored broad prior makes those early observations look artificially small in
standardized units. Pooling forced and Thompson windows therefore makes reset
calibration appear better than the learned-policy subset warrants.

The first post-warmup episode also remains the hardest phase. Its
Thompson-only mean innovation is -1.139, -0.887, and -0.861 for $K=500$,
$K=1000$, and $K=2000$. Later reset episodes are much closer to zero. Reset
helps after it has a boundary at which to discard stale evidence, but it does
not by itself model the rapid reward change immediately after adaptive control
starts.

### Posterior action separation

For completed episode $e$, measure the largest standardized difference between
two posterior action scores immediately before reset as

$$
S_e
=
\max_{g\ne g'}
\frac{
\left|\left(x_g-x_{g'}\right)^\top m_e\right|
}{
\sqrt{
\left(x_g-x_{g'}\right)^\top
V_e
\left(x_g-x_{g'}\right)
}
}.
$$

The intercept cancels in the feature difference, so $S_e$ measures separation
among granularity effects rather than confidence in the common reward level.

| Controller | Completed episodes | Mean $S_e$ | Range of $S_e$ |
| --- | ---: | ---: | ---: |
| Reset, $K=500$ | 8 | 0.701 | 0.128--2.283 |
| Reset, $K=1000$ | 4 | 0.697 | 0.249--1.357 |
| Reset, $K=2000$ | 2 | **0.872** | 0.637--1.107 |

Only one completed $K=500$ episode exceeds 1.96 standard deviations; the
other reset episodes do not strongly distinguish any action pair. At the end
of the terminal partial episode, the largest separation is only 0.028, 0.012,
and 0.042 for $K=500$, $K=1000$, and $K=2000$, because each run has recently
reset and observed only seven complete windows. The no-reset posterior ends at
0.932 standard deviations. These values do not support a stable, confidently
learned granularity preference.

### Interpretation and next experiment

The implementation-level hypothesis is supported. Episodes occur at the
configured controller-relative steps, acquisition covers every granularity,
forced observations condition the posterior, resets restore the exact prior,
the terminal episode is archived as incomplete, and panel-evaluation cost is
unchanged.

The scientific result is narrower:

- Reset clearly reduces stale-intercept prediction bias and prevents the
  no-reset policy from concentrating half its windows on small.
- $K=1000$ produces the highest Thompson-only action entropy, while $K=2000$
  has the lowest acquisition tax and the best seed-42 validation mean.
- $K=500$ spends too much of the run in forced acquisition, still starves large
  in its Thompson subset, and does not improve the uniform validation mean.
- None of the variants yields a calibrated predictive distribution or a
  consistently well-separated posterior action ranking.
- The no-reset comparison changes both posterior memory and acquisition
  exposure: it has no periodic forced passes. Validation differences therefore
  cannot be attributed to posterior reset alone.

The best follow-up is to replicate the no-reset, $K=1000$, and $K=2000$
conditions with seeds 43 and 44. Treat $K=2000$ as the primary candidate
because it combines the best current uniform loss with only 17.2% observed
forced acquisition, and retain $K=1000$ as the shorter-memory bracket because
its innovation bias and Thompson entropy are better. Do not promote $K=500$
to the replication stage unless it is intentionally retained as a high-
acquisition ablation.

After selecting an interval from training seeds without consulting the final
holdout, run the separate final-holdout evaluator once for the selected
configuration and matched control. A future blocked-uniform or periodic-
acquisition-without-reset control would be needed to isolate posterior memory
reset from the training-exposure effect of forced acquisition.

## Pre-Experiment 2: seed replication

This stage adds seeds 43 and 44 for the three conditions retained after
Pre-Experiment 1:

1. $Q=0$ with no reset;
2. full-prior reset with $K=1000$;
3. full-prior reset with $K=2000$.

It deliberately omits $K=500$. That interval spends more than half of the
observed adaptive windows in forced acquisition, starves large in its
Thompson-only subset, and did not improve the seed-42 uniform validation mean.

The commands below submit six independent one-GPU jobs through the same
d_model=256 Slurm entrypoint. Within each seed, all three jobs preserve the
20M-token concat + GMC configuration, data-role split seeds, balanced warmup,
optimizer, scheduler, prior, observation variance, and controller decision
interval. Only the reset contract changes.

Run the complete block from the repository root:

```bash
set -euo pipefail

export OUT=/nfs-stor/nicolas.avila/results/elasticnn
repo=/l/users/nicolas.avila/dev/references/matformer
script="$repo/scripts/slurm_dmodel256_pilot.sh"
labels='[micro,small,medium,large,full]'
prefixes='{micro: 0.125, small: 0.25, medium: 0.5, large: 0.75, full: 1.0}'
output_group=matformer_llama_148m_20m_tokens
status_python="${PYTHON_BIN:-/home/nicolas.avila/.conda/envs/elasticnn/bin/python}"
max_in_flight=4
heartbeat_active_seconds=900
submitted=0
slurm_user=$(id -un)
active_job_names=$(squeue -h -u "$slurm_user" -o '%j')

cd "$repo"
mkdir -p logs

run_is_completed() {
  local run_dir=$1
  local summary_path="$run_dir/run_summary.json"

  [[ -f "$summary_path" ]] || return 1
  "$status_python" -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as summary_file:
    summary = json.load(summary_file)

status = summary.get("status")
if status is None:
    status = summary.get("continuation_state", {}).get("status")
raise SystemExit(0 if status == "completed" else 1)
' "$summary_path"
}

run_is_active() {
  local run_id=$1
  local run_dir=$2
  local active_job_name

  while IFS= read -r active_job_name
  do
    if [[ "$active_job_name" == "$run_id" ]]; then
      return 0
    fi
  done <<< "$active_job_names"

  local heartbeat_path="$run_dir/heartbeats.jsonl"
  [[ -s "$heartbeat_path" ]] || return 1
  "$status_python" -c '
import json
import sys
from datetime import datetime, timezone

heartbeat_path = sys.argv[1]
maximum_age_seconds = float(sys.argv[2])
with open(heartbeat_path, encoding="utf-8") as heartbeat_file:
    records = [line for line in heartbeat_file if line.strip()]
if not records:
    raise SystemExit(1)

timestamp = json.loads(records[-1]).get("timestamp")
if not timestamp:
    raise SystemExit(1)
observed_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
age_seconds = (datetime.now(timezone.utc) - observed_at).total_seconds()
raise SystemExit(0 if 0 <= age_seconds <= maximum_age_seconds else 1)
' "$heartbeat_path" "$heartbeat_active_seconds"
}

submit_if_needed() {
  local run_id=$1
  shift
  local run_dir="$OUT/$output_group/$run_id"

  if run_is_completed "$run_dir"; then
    printf 'SKIP completed: %s\n' "$run_id"
    return 0
  fi

  if run_is_active "$run_id" "$run_dir"; then
    printf 'SKIP queued/running: %s\n' "$run_id"
    return 0
  fi

  if (( active_count + submitted >= max_in_flight )); then
    printf 'DEFER batch limit reached: %s\n' "$run_id"
    return 0
  fi

  sbatch --job-name="$run_id" --gres=gpu:1 --time=04:00:00 "$script" \
    --repo-root "$repo" --output-root "$OUT" \
    --mode nested-random \
    --run-id "$run_id" \
    "${common_args[@]}" \
    "$@"
  submitted=$((submitted + 1))
  active_job_names+=$'\n'"$run_id"
}

candidate_run_ids=(
  d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-noreset-pc1p0em6-s43
  d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-k1000-pc1p0em6-s43
  d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-k2000-pc1p0em6-s43
  d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-noreset-pc1p0em6-s44
  d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-k1000-pc1p0em6-s44
  d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-k2000-pc1p0em6-s44
)

active_count=0
for candidate_run_id in "${candidate_run_ids[@]}"
do
  candidate_run_dir="$OUT/$output_group/$candidate_run_id"
  if ! run_is_completed "$candidate_run_dir" \
    && run_is_active "$candidate_run_id" "$candidate_run_dir"
  then
    active_count=$((active_count + 1))
  fi
done
available_slots=$((max_in_flight - active_count))
if (( available_slots < 0 )); then
  available_slots=0
fi
printf 'Recognized %d queued/running experiment jobs; %d slots available.\n' \
  "$active_count" "$available_slots"

for seed in 43 44
do
  common_args=(
    --override "run.seed=$seed"
    --override training.token_budget=20000000
    --override training.warmup_steps=400
    --override training.pre_nested_warmup.enabled=true
    --override training.pre_nested_warmup.duration=500
    --override training.pre_nested_warmup.unit=steps
    --override training.pre_nested_warmup.policy=balanced_global
    --override training.pre_nested_warmup.action_interval_steps=50
    --override training.optimizer.preset=adam
    --override training.learning_rate=0.001
    --override training.learning_rate_scale_rule=none
    --override dataset.sample_limit=100000
    --override model.variant=concat
    --override model.correction_mode=gmc
    --override model.membership_correction=true
    --override model.granularity_mode=explicit
    --override "model.granularities=$labels"
    --override "model.granularity_prefixes=$prefixes"
    --override model.granularity_sampling_mode=adaptive_global
    --override model.adaptive_sampler_strategy=thompson
    --override model.adaptive_controller.preset=bayesian_thompson
    --override model.adaptive_controller.decision_interval_steps=50
    --override 'model.adaptive_controller.prior_mean=[0.0,0.0,0.0,0.0,0.0]'
    --override 'model.adaptive_controller.prior_covariance=[1.0e-4,1.0e-6,1.0e-6,1.0e-6,1.0e-6]'
    --override model.adaptive_controller.observation_noise_variance=1.0e-7
    --override model.adaptive_controller.process_noise_covariance=0.0
  )

  submit_if_needed \
    "d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-noreset-pc1p0em6-s${seed}" \
    --override model.adaptive_controller.reset.enabled=false

  submit_if_needed \
    "d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-k1000-pc1p0em6-s${seed}" \
    --override model.adaptive_controller.reset.enabled=true \
    --override model.adaptive_controller.reset.interval_steps=1000 \
    --override model.adaptive_controller.reset.policy=full_prior \
    --override model.adaptive_controller.reset.acquisition_policy=balanced_global \
    --override model.adaptive_controller.reset.acquisition_passes=1

  submit_if_needed \
    "d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-k2000-pc1p0em6-s${seed}" \
    --override model.adaptive_controller.reset.enabled=true \
    --override model.adaptive_controller.reset.interval_steps=2000 \
    --override model.adaptive_controller.reset.policy=full_prior \
    --override model.adaptive_controller.reset.acquisition_policy=balanced_global \
    --override model.adaptive_controller.reset.acquisition_passes=1
done
```

The loop considers these six run IDs in order and submits at most four that are
not already complete:

```text
d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-noreset-pc1p0em6-s43
d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-k1000-pc1p0em6-s43
d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-k2000-pc1p0em6-s43
d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-noreset-pc1p0em6-s44
d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-k1000-pc1p0em6-s44
d256-ag-cgmc-bw500-reset20m-h50-r1p0em7-q0-k2000-pc1p0em6-s44
```

On a clean destination, the first invocation submits the three seed-43 jobs and
the seed-44 no-reset job, then prints `DEFER` for the remaining two. After that
batch completes, run the same block again. It reads each destination
`run_summary.json`, prints `SKIP` for completed runs, and submits the two
remaining seed-44 reset jobs. A run directory without a completed summary is
eligible for resubmission once it is absent from the live queue and its latest
heartbeat is stale, allowing the launcher's normal continuation logic to
recover an interrupted job.

The loop also sets each new Slurm job name to its exact run ID and checks the
current user's live `squeue` job names. For jobs submitted earlier with the
generic `matformer-dmodel256` name, it treats a destination heartbeat from the
last 15 minutes as evidence that the run is active. It prints
`SKIP queued/running` in either case.

Already active experiment jobs count toward `max_in_flight=4`. The in-memory
queue snapshot is updated after each successful submission, so an invocation
cannot exceed four recognized queued/running experiment jobs or submit a
duplicate. An incomplete destination whose heartbeat is older than 15 minutes
and whose run ID is absent from `squeue` remains eligible for checkpoint
continuation.

For each seed $s$, compute the paired uniform-validation difference

$$
\Delta_{K,s}
=
\overline{L}_{K,s}
-
\overline{L}_{\mathrm{no\ reset},s}.
$$

Select an interval using the three-seed ordinary-validation losses,
Thompson-only action entropy and minimum action count, forced-acquisition
fraction, Thompson-only innovation statistics, and episode-level posterior
separation. Prefer consistent paired effects over the smallest single-seed
loss. Do not run the final-holdout evaluator during this selection stage.

After freezing the choice between $K=1000$ and $K=2000$, evaluate the
untouched final holdout for the selected interval and the matched no-reset
control across seeds 42--44. Those final-holdout results are confirmatory and
must not be used to switch intervals afterward.

### Pre-Experiment 2 results

Date analyzed: 2026-08-10

All six seed-43/44 jobs completed from scratch without a resume or controller
failure. Each reached 4,883 optimizer steps and exactly 20M planned tokens,
selected the final step as its ordinary-validation checkpoint, emitted 87
complete controller observations from 88 panel evaluations, and recorded the
terminal 33-step window as incomplete. Within each seed, the optimizer-
training, ordinary-validation, controller-panel, and final-holdout manifest
hashes match across no reset, $K=1000$, and $K=2000$. The comparison therefore
retains the intended paired design.

No final holdout has been evaluated. Interval selection below uses only
ordinary validation and controller diagnostics.

#### Paired ordinary-validation result

The seed-level uniform losses and paired differences are:

| Seed | No reset | $K=1000$ | $\Delta_{1000,s}$ | $K=2000$ | $\Delta_{2000,s}$ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 5.180599 | 5.179584 | -0.001015 | **5.178845** | **-0.001754** |
| 43 | 5.200443 | **5.193429** | **-0.007014** | 5.195721 | -0.004722 |
| 44 | **5.211679** | 5.213356 | +0.001678 | 5.211890 | +0.000211 |
| Three-seed mean | 5.197573 | **5.195456** | **-0.002117** | 5.195485 | -0.002088 |

Both intervals improve the matched control for seeds 42 and 43 and regress
slightly for seed 44. Their three-seed means are effectively tied: $K=1000$
is lower than $K=2000$ by only

$$
5.195485-5.195456=0.000029.
$$

The paired differences summarize as

$$
\begin{aligned}
\operatorname{mean}(\Delta_{1000,s})
&=-0.002117,
&\operatorname{sd}(\Delta_{1000,s})
&=0.004449,\\
\operatorname{mean}(\Delta_{2000,s})
&=-0.002088,
&\operatorname{sd}(\Delta_{2000,s})
&=0.002483.
\end{aligned}
$$

With only three seeds, the two-sided 95% paired $t$ intervals are wide:
$[-0.01317,0.00894]$ for $K=1000$ and $[-0.00826,0.00408]$ for $K=2000$.
They include zero and should be treated as descriptive uncertainty, not proof
of equivalence or absence of an effect. $K=2000$ nevertheless has the more
stable paired response.

#### The aggregate gain is concentrated at micro

Three-seed mean losses by granularity are:

| Controller | Micro | Small | Medium | Large | Full | Uniform mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No reset | 5.278705 | **5.188435** | **5.174667** | **5.172415** | **5.173645** | 5.197573 |
| $K=1000$ | **5.257780** | 5.191645 | 5.178250 | 5.174791 | 5.174817 | **5.195456** |
| $K=2000$ | 5.263610 | 5.188489 | 5.176885 | 5.173901 | 5.174542 | 5.195485 |

Relative to no reset, the mean changes are:

| Controller | $\Delta$ micro | $\Delta$ small | $\Delta$ medium | $\Delta$ large | $\Delta$ full |
| --- | ---: | ---: | ---: | ---: | ---: |
| $K=1000$ | -0.020925 | +0.003210 | +0.003583 | +0.002376 | +0.001172 |
| $K=2000$ | -0.015095 | +0.000054 | +0.002217 | +0.001485 | +0.000897 |

Thus, both uniform-mean improvements are driven by better micro loss. Averaged
over small, medium, large, and full, $K=1000$ is 0.002585 worse than no reset
and $K=2000$ is 0.001163 worse. This is compatible with forced acquisition
redistributing training exposure toward micro. It is a valid effect under the
predeclared uniform-granularity objective, but it is not broad improvement at
every model size.

#### Learned-policy behavior across seeds

Counts below pool Thompson-selected windows over seeds 42--44 and remain
ordered as micro / small / medium / large / full. Entropies, innovations, and
episode separations are means of the three seed-level statistics.

| Controller | Forced fraction | Pooled Thompson counts | Mean Thompson $H_{\mathrm{norm}}$ | Mean Thompson $z$ | Mean Thompson sd($z$) | Mean $S_e$ |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| No reset | 0.0% | 31 / 86 / 53 / 50 / 41 | 0.920 | -0.755 | 0.511 | -- |
| $K=1000$ | 28.7% | 22 / 60 / 37 / 37 / 30 | **0.942** | **-0.365** | 0.593 | 0.770 |
| $K=2000$ | **17.2%** | 27 / 75 / 45 / 41 / 28 | 0.924 | -0.504 | **0.603** | **0.918** |

The seed-42 conclusion that reset improves learned-policy diversity weakens
after replication. No-reset Thompson entropy ranges from 0.844 to 0.996 across
seeds; its strong seed-42 concentration on small is not universal. $K=1000$
has the highest mean entropy and the smallest entropy variation across seeds.
$K=2000$ is close to the no-reset mean and concentrates 33 of 72 seed-43
Thompson windows on small. Forced acquisition guarantees balanced total
exposure, but it does not guarantee a balanced learned policy.

The calibration result does replicate. For every seed, reset moves the
negative Thompson innovation mean toward zero. The three-seed mean changes
from -0.755 without reset to -0.365 at $K=1000$ and -0.504 at $K=2000$.
Thompson innovation standard deviation also moves toward one, from 0.511 to
0.593 and 0.603. Both reset variants therefore track the decaying reward level
better than the static posterior.

Neither is calibrated: mean 95% coverage remains 97.3% for $K=1000$ and
97.2% for $K=2000$, and standard deviations remain far below one. Posterior
action separation also stays weak. The mean completed-episode $S_e$ is 0.770
and 0.918, while the largest value in any seed is only 1.547 for $K=1000$ and
1.151 for $K=2000$. No replicated episode reaches 1.96 standard deviations.

#### Interval selection

Freeze $K=2000$ for the confirmatory comparison. $K=1000$ and $K=2000$ have
indistinguishable three-seed mean loss at the observed precision, but
$K=2000$:

- wins the direct interval comparison in seeds 42 and 44;
- has a smaller sample standard deviation of its paired loss effect;
- spends 17.2% rather than 28.7% of observed windows in forced acquisition;
- produces slightly stronger episode-level action separation;
- retains the replicated reduction in stale-reward innovation bias.

$K=1000$ remains the better choice only if innovation-mean correction or
Thompson entropy is assigned priority over acquisition cost. Those were
diagnostics rather than the primary performance objective, so they do not
justify switching away from $K=2000$ after observing the near-tied losses.

The next authorized use of the held-out data is a confirmatory evaluation of
$K=2000$ and no reset for seeds 42--44. Report all six final-holdout results,
including the paired uniform-mean differences, without reopening the interval
choice. Even a favorable holdout result should be described as preliminary:
the three-seed ordinary-validation gain is small, not statistically resolved,
and concentrated at micro.

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
