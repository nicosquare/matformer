# Probabilistic Adaptive Granularity: Fixed-Q Calibration Experiments

Date range: 2026-08-06 to 2026-08-07

Status: historical experimental record. Manual fixed-$Q$ calibration is paused
after submission batch 4; command blocks for completed or superseded batches
must not be resubmitted blindly.

This record was moved out of the main
[probabilistic-controller discussion](probabilistic_adaptive_granularity_discussion_2026-08-05.md)
to keep the design rationale concise. It preserves the 100M-token diagnosis,
all four 20M-token command batches, their results, and the reasoning that led
from covariance tuning to balanced warmup and then away from further one-axis
$Q$ sweeps.

The current alternative under consideration is the
[episodic-reset proposal](probabilistic_adaptive_granularity_reset_method_proposal_2026-08-07.md).

## Experiment notes from the 100M-token sweep

Date analyzed: 2026-08-06

Results directory:
`/nfs-stor/nicolas.avila/results/elasticnn/matformer_llama_148m_100m_tokens`

The main performance objective for the next experiments is to make adaptive
training approach the quality of `nested-all`, while retaining a meaningful
compute advantage. The completed 100M-token runs identify posterior calibration
as the immediate bottleneck. The present Bayesian controller does not yet learn
a useful action ranking: its uncertainty is many orders of magnitude larger
than the reward differences it is trying to distinguish.

### Current performance comparison

The most relevant final scaling losses are:

| Training method | Mean loss over five granularities | Full loss | Observed runtime |
| --- | ---: | ---: | ---: |
| `nested-all`, slicing, GMC | **4.4332** | **4.3722** | not recorded here |
| `nested-all`, concat, GMC | 4.4379 | 4.3769 | 5.61 h |
| adaptive-global Thompson, concat, GMC | **4.4782** | **4.4449** | 2.82 h |
| random-global, concat, GMC | 4.4986 | 4.4665 | 1.61 h |
| adaptive-per-block Thompson, slicing, GMC | 4.4974 | 4.4570 | not recorded here |

Thus the best completed adaptive run is currently global Thompson sampling with
the concat variant and GMC. Relative to the matching random-global baseline,
it improves mean loss by approximately

$$
4.4986-4.4782=0.0204.
$$

It remains approximately

$$
4.4782-4.4379=0.0403
$$

behind the matching concat-GMC `nested-all` run. Compared with the best
`nested-all` result in the directory, the gap is approximately $0.0450$.

These comparisons currently come from one training seed and should be treated
as directional until replicated. In particular, the observed adaptive
improvement cannot yet be attributed to posterior learning because the adaptive
action schedule is almost uniform and differs structurally from the ordinary
random baseline by holding each action fixed for 50 steps.

### The current controller is effectively random

For global adaptation there are 488 completed decision windows. In the best
adaptive run, the action counts in the order

$$
(	ext{micro},\text{small},\text{medium},\text{large},\text{full})
$$

are

$$
(93,104,110,88,93).
$$

This is close to a uniform allocation. More importantly, across the five
completed global adaptive experiments, between 485 and 488 of the 488 selected
actions are identical to the concat-GMC action sequence. Changing model variant
or correction mode barely changes controller choices. The posterior-sampling
random stream, rather than learned reward differences, is therefore determining
almost the entire schedule.

The scale mismatch explains this behavior. The current preset resolves to

$$
V_0=I,
\qquad
\sigma^2=10^{-2},
\qquad
Q=10^{-4}I.
$$

These are covariance values. Their corresponding standard deviations are

$$
\sqrt{\operatorname{diag}(V_0)}=1,
\qquad
\sigma=0.1,
\qquad
\sqrt{\operatorname{diag}(Q)}=0.01.
$$

By comparison, the measured reward standard deviation over the complete run is
approximately

$$
\operatorname{sd}(r_t)\approx 2.1\times 10^{-3},
$$

and it falls to approximately

$$
\operatorname{sd}(r_t)\approx 2.1\times 10^{-5}
$$

in the last quarter of training. Injecting process uncertainty with standard
deviation $0.01$ every 50 steps is therefore extremely large relative to the
late-training reward signal. Similarly, an observation standard deviation of
$0.1$ says that individual rewards are far less informative than their observed
scale supports.

At the final boundary of the concat-GMC adaptive-global run, the posterior mean
action scores are approximately

$$
\begin{aligned}
\widehat\mu_{\mathrm{micro}} &= 6.69\times10^{-6},\\
\widehat\mu_{\mathrm{small}} &= 1.24\times10^{-6},\\
\widehat\mu_{\mathrm{medium}} &= 6.78\times10^{-7},\\
\widehat\mu_{\mathrm{large}} &= 7.61\times10^{-7},\\
\widehat\mu_{\mathrm{full}} &= 2.22\times10^{-7}.
\end{aligned}
$$

The largest difference between these means is only about $6.5\times10^{-6}$.
However, the posterior standard deviation of a pairwise action-score difference
is between $0.059$ and $0.077$. The uncertainty relevant to choosing an action
is consequently roughly four orders of magnitude larger than the estimated
ranking differences. Continued near-uniform Thompson exploration is the
expected result of this posterior, not a sampling bug.

### Highest-priority knob: observation-noise variance

In the observation model

$$
r_t=x_t^\top\beta_t+\epsilon_t,
\qquad
\epsilon_t\sim\mathcal{N}(0,\sigma^2),
$$

`observation_noise_variance` is $\sigma^2$, not $\sigma$. Lowering it makes an
observed reward more informative. The first sweep should use

$$
\sigma^2\in\{10^{-8},10^{-7},10^{-6}\}.
$$

An offline replay of the recorded concat-GMC action and reward trace favored
$\sigma^2\approx10^{-8}$ within a coarse one-step predictive-likelihood grid.
This is a center for the next sweep, not evidence that $10^{-8}$ maximizes final
language-model performance.

The observation variance should describe reward variation not captured by the
action and state model. It should not be set to the total variation of the
learning curve. Overall reward drift belongs in the intercept dynamics or in an
explicit training-progress context.

### Use different process variances for the intercept and contrasts

The overall rate of language-model improvement changes quickly during
training, whereas relative granularity effects should usually change more
slowly. A scalar process covariance forces these two phenomena to use the same
drift scale. Instead, for global adaptation use

$$
Q=\operatorname{diag}
\left(
q_{\mathrm{intercept}},
q_{\mathrm{contrast}},
q_{\mathrm{contrast}},
q_{\mathrm{contrast}},
q_{\mathrm{contrast}}
\right).
$$

The next sweep should include

$$
q_{\mathrm{intercept}}\in\{10^{-8},10^{-7}\},
\qquad
q_{\mathrm{contrast}}\in\{0,10^{-10},10^{-9}\}.
$$

The interpretations are:

- $q_{\mathrm{contrast}}=0$ assumes a stationary action ranking and lets
  evidence accumulate throughout training;
- $q_{\mathrm{contrast}}=10^{-10}$ permits slow evolution of the ranking;
- $q_{\mathrm{contrast}}=10^{-9}$ is a more reactive alternative;
- the larger intercept variance lets the common reward level track the
  changing learning curve without forcing the contrast coefficients to forget
  equally quickly.

The offline replay's best coarse predictive setting was approximately

$$
\sigma^2=10^{-8},
\qquad
q_{\mathrm{intercept}}=10^{-8},
\qquad
q_{\mathrm{contrast}}=10^{-9}.
$$

Again, offline replay evaluates reward prediction under the already-recorded
action sequence. A real training run is required because changing the posterior
also changes future actions and hence the model trajectory.

### Calibrate the prior covariance in reward units

A neutral prior should retain zero contrast means:

$$
m_0=
\begin{bmatrix}
0&0&0&0&0
\end{bmatrix}^\top.
$$

The common intercept does not affect which action maximizes a sampled score,
whereas nonzero contrast means would encode a prior granularity preference. A
useful initial diagonal covariance is

$$
V_0=\operatorname{diag}
\left(10^{-4},10^{-6},10^{-6},10^{-6},10^{-6}\right).
$$

This gives the intercept standard deviation $10^{-2}$, which can cover the
large early reward level, and gives each contrast standard deviation $10^{-3}$.
The latter remains exploratory but is far closer to plausible action effects
than the current standard deviation of one. Contrast prior variance $10^{-8}$
should also be tested as a more conservative initialization.

The configuration system accepts a scalar, a diagonal vector, or a complete
covariance matrix. A diagonal vector is sufficient for this experiment. The
coefficient order for global adaptation is the intercept followed by the four
saved Helmert contrast coefficients.

One concrete starting configuration is:

```yaml
model:
  granularity_sampling_mode: adaptive_global
  adaptive_sampler_strategy: thompson
  adaptive_controller:
    preset: bayesian_thompson
    decision_interval_steps: 50
    prior_mean: [0.0, 0.0, 0.0, 0.0, 0.0]
    prior_covariance: [1.0e-4, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6]
    observation_noise_variance: 1.0e-8
    process_noise_covariance: [1.0e-8, 1.0e-9, 1.0e-9, 1.0e-9, 1.0e-9]
```

All covariance entries in this configuration are variances, not standard
deviations.

### Decision interval and controller cost

The current interval $h=50$ produces 488 completed reward observations and 489
controller-panel evaluations. In the concat-GMC runs, observed runtimes are
approximately:

$$
\begin{array}{l|r}
\text{method} & \text{runtime}\\
\hline
\text{random-global} & 1.61\ \mathrm{h}\\
\text{adaptive-global},\ h=50 & 2.82\ \mathrm{h}\\
\text{nested-all} & 5.61\ \mathrm{h}.
\end{array}
$$

Thus controller evaluation currently adds about 1.21 hours relative to the
random-global run. Once posterior scale is fixed at $h=50$, sweep

$$
h\in\{50,100,200\}.
$$

A larger interval has three effects:

1. it gives an action more optimizer steps over which to produce a measurable
   objective change;
2. it reduces controller-evaluation overhead;
3. it gives fewer posterior observations and makes the policy less responsive.

The five-dimensional global controller should still have enough observations
at $h=100$ or $h=200$. The 65-dimensional per-block controller is much more
data-limited, so increasing its interval before adding stronger structure or
shrinkage is not recommended.

If fixed-panel evaluation error dominates, a useful initial variance scaling is

$$
\sigma_h^2
\approx
\sigma_{50}^2\left(\frac{50}{h}\right)^2,
$$

because reward divides a boundary-objective difference by $h$. If coefficient
drift accumulates independently per optimizer step, an initial process scaling
is

$$
Q_h\approx Q_{50}\frac{h}{50}.
$$

These relations are initialization heuristics. Changing $h$ also changes action
persistence and the model trajectory, so the resulting innovations must be
checked directly.

### Scope, MatFormer variant, and correction mode

Global adaptation should remain the primary development target. It has only
five coefficients, whereas the additive per-block controller has

$$
1+B(G-1)=1+16(5-1)=65
$$

coefficients but still receives only one scalar reward per window. Completed
per-block runs retain nearly uniform action counts in every block and have mean
final coefficient standard deviation around $0.071$. Their results do not yet
show successful block-level credit assignment.

Among the completed runs:

- concat + GMC is the best adaptive-global configuration;
- GMC improves concat adaptive-global mean loss by about $0.0055$ relative to
  no correction;
- concat + LMC is substantially worse, with mean loss $4.5639$ rather than
  $4.4782$ for concat + GMC;
- slicing + GMC is the best `nested-all` configuration, but current
  adaptive-global slicing is about $0.0076$ worse than adaptive-global concat.

Therefore, controller calibration should use global + concat + GMC. Per-block
adaptation should be revisited only after the global posterior learns a clear,
reproducible nonuniform policy. A successful per-block model may require
hierarchical shrinkage or shared layer effects rather than 64 independently
drifting contrast coefficients.

### Reward nonstationarity needs an explicit follow-up

The reward standard deviation changes by roughly two orders of magnitude over
training: it is about $4\times10^{-3}$ in the first quarter and about
$2\times10^{-5}$ in the final quarter. A single stationary observation
variance and intercept-only context must accommodate both regimes.

If diagonal $Q$ is insufficient, the next method-level experiments should be:

- delay adaptive exploitation until the 2,000-step learning-rate warmup is
  complete, using a balanced schedule to collect initial observations;
- add normalized training progress or current learning-rate ratio as context;
- interact that context with the action features, rather than merely appending
  a context-only feature;
- normalize reward or innovation scale using only information available before
  the next action is chosen;
- consider phase-dependent observation and process covariance.

The first-quarter reward trend can otherwise be misattributed to whichever
granularity happened to be selected during a sharp change in the learning
curve.

### Why matching `nested-all` may require more than posterior tuning

At every optimizer step, `nested-all` evaluates all five granularities on the
same batch and averages their gradients. The adaptive method evaluates one
granularity. Uniform one-action sampling can be an unbiased stochastic estimate
of the mean nested-all gradient, but it has substantially higher variance across
granularities. The two methods are token-matched but not compute-matched.

Two follow-up comparisons are therefore important:

1. **Compute-matched longer training.** The current adaptive-global run takes
   about half the wall time of `nested-all`. A roughly 200M-token adaptive run
   is consequently a useful comparison after reducing controller overhead.
2. **Hybrid or sandwich sampling.** Always train the micro and full
   granularities and let the controller choose one intermediate granularity.
   Averaging these three losses would cost less than five-granularity
   `nested-all`, protect both endpoints, and reduce the variance of the
   one-profile gradient estimator.

Hybrid sampling changes the implemented one-action contract and should be
reported as a separate method, not as a hidden hyperparameter change.

### Ordered next experiments

The recommended experiment order is:

1. Use global + concat + GMC and keep $h=50$ while calibrating the posterior.
2. Run short 10M--20M-token jobs with
   $\sigma^2\in\{10^{-8},10^{-7}\}$,
   $q_{\mathrm{intercept}}\in\{10^{-8},10^{-7}\}$, and
   $q_{\mathrm{contrast}}\in\{0,10^{-10},10^{-9}\}$.
3. Include both contrast prior variances $10^{-6}$ and $10^{-8}$ for the most
   promising $(\sigma^2,Q)$ settings.
4. Add a balanced, nonadaptive baseline that holds each action for 50 steps.
   This isolates posterior learning from the effect of temporally blocked
   sampling.
5. Reject configurations whose action-score uncertainty remains many orders of
   magnitude above posterior mean differences, or whose policy collapses before
   collecting adequate evidence.
6. For the best calibrated posterior, sweep $h\in\{50,100,200\}$.
7. Run the selected 100M-token configuration with at least three training and
   posterior-sampling seeds.
8. Compare the winner at matched tokens and at approximately matched wall time,
   including a longer adaptive run.
9. Only then revisit per-block adaptation, progress interactions, or a hybrid
   multi-granularity action.

### Commands for the next calibration experiments

These are explicit submission commands, following the format in
`notes/dmodel256_explicit_granularity_commands.md`. The 12-run calibration
grid is divided into three submission batches of four jobs. Submit only one
batch at a time. Once a batch completes, inspect its results before submitting
the next batch.

All runs use global adaptation, concat + GMC, 20M tokens, a 400-step scheduler
warmup, \(h=50\), contrast prior variance \(10^{-6}\), and root seed 42. Only
the observation variance, intercept process variance, and contrast process
variance change.

```bash
export OUT=/nfs-stor/nicolas.avila/results/elasticnn
repo=/l/users/nicolas.avila/dev/references/matformer
script="$repo/scripts/slurm_dmodel256_pilot.sh"
labels='[micro,small,medium,large,full]'
prefixes='{micro: 0.125, small: 0.25, medium: 0.5, large: 0.75, full: 1.0}'

cd "$repo"
mkdir -p logs
```

#### Submission batch 1: central calibration settings

This first batch contains the offline-replay center and the most informative
nearby alternatives.

```bash
sbatch --gres=gpu:1 --time=04:00:00 "$script" \
  --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random \
  --run-id d256-ag-cgmc-cal20m-h50-r1p0em8-qi1p0em8-qc0-pc1p0em6-s42 \
  --override run.seed=42 \
  --override training.token_budget=20000000 \
  --override training.warmup_steps=400 \
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
  --override model.adaptive_controller.observation_noise_variance=1.0e-8 \
  --override 'model.adaptive_controller.process_noise_covariance=[1.0e-8,0,0,0,0]'
```

```bash
sbatch --gres=gpu:1 --time=04:00:00 "$script" \
  --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random \
  --run-id d256-ag-cgmc-cal20m-h50-r1p0em8-qi1p0em8-qc1p0em10-pc1p0em6-s42 \
  --override run.seed=42 \
  --override training.token_budget=20000000 \
  --override training.warmup_steps=400 \
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
  --override model.adaptive_controller.observation_noise_variance=1.0e-8 \
  --override 'model.adaptive_controller.process_noise_covariance=[1.0e-8,1.0e-10,1.0e-10,1.0e-10,1.0e-10]'
```

```bash
sbatch --gres=gpu:1 --time=04:00:00 "$script" \
  --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random \
  --run-id d256-ag-cgmc-cal20m-h50-r1p0em8-qi1p0em8-qc1p0em9-pc1p0em6-s42 \
  --override run.seed=42 \
  --override training.token_budget=20000000 \
  --override training.warmup_steps=400 \
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
  --override model.adaptive_controller.observation_noise_variance=1.0e-8 \
  --override 'model.adaptive_controller.process_noise_covariance=[1.0e-8,1.0e-9,1.0e-9,1.0e-9,1.0e-9]'
```

```bash
sbatch --gres=gpu:1 --time=04:00:00 "$script" \
  --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random \
  --run-id d256-ag-cgmc-cal20m-h50-r1p0em7-qi1p0em8-qc1p0em9-pc1p0em6-s42 \
  --override run.seed=42 \
  --override training.token_budget=20000000 \
  --override training.warmup_steps=400 \
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
  --override 'model.adaptive_controller.process_noise_covariance=[1.0e-8,1.0e-9,1.0e-9,1.0e-9,1.0e-9]'
```

Wait for these four jobs to finish before submitting batch 2. We should analyze
this batch first because it already tests stationary versus drifting contrasts
and the main observation-variance alternative.

##### Results and interpretation of submission batch 1

Date analyzed: 2026-08-06

Results directory:
`/nfs-stor/nicolas.avila/results/elasticnn/calibration/matformer_llama_148m_20m_tokens`

The first batch gives a decisive negative result for
$\sigma^2=10^{-8}$ and a qualified positive result for
$\sigma^2=10^{-7}$. The four completed runs are:

| Controller setting | Mean loss | Micro loss | Full loss | Action counts |
| --- | ---: | ---: | ---: | --- |
| $\sigma^2=10^{-7}$, $q_I=10^{-8}$, $q_C=10^{-9}$ | **5.1908** | 5.2301 | 5.1769 | 19 / 24 / 32 / 1 / 21 |
| $\sigma^2=10^{-8}$, $q_I=10^{-8}$, $q_C=0$ | 5.8799 | 8.0327 | **5.1050** | 0 / 1 / 96 / 0 / 0 |
| $\sigma^2=10^{-8}$, $q_I=10^{-8}$, $q_C=10^{-10}$ | 5.8799 | 8.0327 | **5.1050** | 0 / 1 / 96 / 0 / 0 |
| $\sigma^2=10^{-8}$, $q_I=10^{-8}$, $q_C=10^{-9}$ | 5.8799 | 8.0327 | **5.1050** | 0 / 1 / 96 / 0 / 0 |

Here $q_I$ is the intercept process variance and $q_C$ is the process
variance assigned to each Helmert contrast. Action counts are ordered as

$$
(\text{micro},\text{small},\text{medium},\text{large},\text{full}).
$$

###### Why $\sigma^2=10^{-8}$ must be rejected

All three $\sigma^2=10^{-8}$ runs produce exactly the same scaling losses and
almost exactly the same deterministic training trajectory. The controller
selects `small` for its first window and `medium` for the remaining 96 windows.
Changing $q_C$ from zero to $10^{-10}$ or $10^{-9}$ does not change this
outcome.

This is premature posterior collapse. Training almost exclusively with the
medium granularity gives relatively strong medium, large, and full losses, but
it leaves the micro and small subnetworks badly undertrained. In particular,
micro loss rises to 8.0327. The attractive full loss of 5.1050 therefore does
not make these configurations successful: the configured controller objective
is the uniform mean across all five granularities, for which their mean loss is
5.8799.

The normalized entropy of the complete action distribution is only about

$$
\frac{-\sum_g p_g\log p_g}{\log 5}=0.036.
$$

The controller has effectively committed to one action after two
observations. At this observation variance, the small contrast-process values
in this batch cannot restore meaningful exploration once an action has been
assigned an unfavorable early coefficient.

###### What improved with $\sigma^2=10^{-7}$

Increasing the observation variance by a factor of ten prevents the immediate
medium-only collapse. The resulting action counts are

$$
(19,24,32,1,21),
$$

with normalized action entropy approximately

$$
0.876.
$$

All five subnetworks remain usable and mean loss improves from 5.8799 to
5.1908. The policy also changes through training rather than becoming fixed
after the first two windows. This makes $\sigma^2=10^{-7}$ the only viable
starting point from batch 1.

It is not yet a properly calibrated final controller. Its action frequencies
alone can make it look healthy, but one action was still rejected for a reason
that is almost certainly spurious.

###### Early learning-curve drift is being mistaken for an action effect

The first several controller rewards change extremely quickly:

$$
\begin{array}{c|c|r|r|r}
\text{window} & \text{action} & r_t & \widehat r_t & r_t-\widehat r_t\\
\hline
0 & \text{small}  & 0.024343 & 0        & 0.024343\\
1 & \text{medium} & 0.056713 & 0.024077 & 0.032636\\
2 & \text{medium} & 0.021117 & 0.055230 & -0.034113\\
3 & \text{medium} & 0.004332 & 0.037656 & -0.033324\\
4 & \text{medium} & 0.003503 & 0.024862 & -0.021359\\
5 & \text{large}  & 0.003349 & 0.017714 & -0.014365.
\end{array}
$$

Large is sampled only in window 5. By then the common rate of improvement has
fallen sharply, but the intercept-only reward model predicts 0.017714. The
large negative innovation is partly assigned to the large action contrasts.
The controller subsequently never selects large again. Its final inferred
large reward is approximately

$$
\widehat\mu_{\mathrm{large}}=-4.994\times10^{-3},
$$

approximately ten posterior standard deviations below competing action
scores. One observation during a rapidly changing learning-curve phase has
therefore produced an implausibly confident long-lived rejection.

This behavior is visible in standardized innovations. For the
$\sigma^2=10^{-7}$ run, their mean and standard deviation by training quarter
are approximately:

$$
\begin{array}{c|r|r}
\text{quarter} & \operatorname{mean}(z_t) & \operatorname{sd}(z_t)\\
\hline
1 & -14.41 & 25.40\\
2 & -0.18 & 0.53\\
3 & -0.06 & 0.36\\
4 & -0.06 & 0.14.
\end{array}
$$

The model is drastically overconfident during the early reward transition,
while the same fixed observation variance is conservative later. This shows
that observation noise alone cannot model both phases. The intercept process
variance $q_I=10^{-8}$ is too small to track the early common reward drift, so
the posterior uses action contrasts to explain a change that is largely caused
by ordinary training progress.

###### Revised next four experiments

Do **not** submit the original batch-2 commands below. Batch 1 makes additional
$\sigma^2=10^{-8}$ runs with $q_I\leq10^{-7}$ low priority, and reducing
$q_C$ below $10^{-9}$ is unlikely to help an action recover after early
misattribution.

The next four experiments should instead keep the 20M-token budget, $h=50$,
contrast prior variance $10^{-6}$, and root seed 42, while testing:

1. $\sigma^2=10^{-7}$, $q_I=10^{-6}$, $q_C=10^{-9}$;
2. $\sigma^2=10^{-7}$, $q_I=10^{-5}$, $q_C=10^{-9}$;
3. $\sigma^2=10^{-6}$, $q_I=10^{-5}$, $q_C=10^{-9}$;
4. $\sigma^2=10^{-7}$, $q_I=10^{-5}$, $q_C=10^{-8}$.

This targeted batch asks four separate questions:

- Does a 100-times larger intercept process variance absorb enough of the
  common learning-curve drift?
- Does a 1,000-times larger intercept process variance prevent early action
  poisoning?
- Does a larger observation variance provide additional protection from early
  overconfidence?
- Does a larger contrast process variance allow an incorrectly rejected action
  to recover?

If these configurations still acquire persistent action preferences from the
first few rapidly changing windows, further tuning of fixed covariances is not
the right next step. The method should then add an explicit balanced controller
burn-in, delay adaptive exploitation, or model training progress in the reward
features.

The original batch-2 commands have been replaced below by these four targeted
experiments. The original batch-3 commands remain only as a record of the
initial grid and must not be submitted before analyzing the revised batch 2.

#### Submission batch 2: revised early-drift calibration

Submit these four jobs together after batch 1. They replace the original
batch-2 grid and specifically test whether a more responsive intercept, a
larger observation variance, or faster contrast recovery prevents early
learning-curve drift from poisoning an action estimate.

```bash
sbatch --gres=gpu:1 --time=04:00:00 "$script" \
  --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random \
  --run-id d256-ag-cgmc-cal20m-h50-r1p0em7-qi1p0em6-qc1p0em9-pc1p0em6-s42 \
  --override run.seed=42 \
  --override training.token_budget=20000000 \
  --override training.warmup_steps=400 \
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
  --override 'model.adaptive_controller.process_noise_covariance=[1.0e-6,1.0e-9,1.0e-9,1.0e-9,1.0e-9]'
```

```bash
sbatch --gres=gpu:1 --time=04:00:00 "$script" \
  --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random \
  --run-id d256-ag-cgmc-cal20m-h50-r1p0em7-qi1p0em5-qc1p0em9-pc1p0em6-s42 \
  --override run.seed=42 \
  --override training.token_budget=20000000 \
  --override training.warmup_steps=400 \
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
  --override 'model.adaptive_controller.process_noise_covariance=[1.0e-5,1.0e-9,1.0e-9,1.0e-9,1.0e-9]'
```

```bash
sbatch --gres=gpu:1 --time=04:00:00 "$script" \
  --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random \
  --run-id d256-ag-cgmc-cal20m-h50-r1p0em6-qi1p0em5-qc1p0em9-pc1p0em6-s42 \
  --override run.seed=42 \
  --override training.token_budget=20000000 \
  --override training.warmup_steps=400 \
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
  --override model.adaptive_controller.observation_noise_variance=1.0e-6 \
  --override 'model.adaptive_controller.process_noise_covariance=[1.0e-5,1.0e-9,1.0e-9,1.0e-9,1.0e-9]'
```

```bash
sbatch --gres=gpu:1 --time=04:00:00 "$script" \
  --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random \
  --run-id d256-ag-cgmc-cal20m-h50-r1p0em7-qi1p0em5-qc1p0em8-pc1p0em6-s42 \
  --override run.seed=42 \
  --override training.token_budget=20000000 \
  --override training.warmup_steps=400 \
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
  --override 'model.adaptive_controller.process_noise_covariance=[1.0e-5,1.0e-8,1.0e-8,1.0e-8,1.0e-8]'
```

Wait for all four revised batch-2 jobs to finish before selecting any
configuration or submitting a later batch. Compare their early action
sequences, first-quarter standardized innovations, final action entropy,
pairwise score uncertainty, and mean scaling loss.

##### Results and interpretation of submission batch 2

Date analyzed: 2026-08-07

Results directory:
`/nfs-stor/nicolas.avila/results/elasticnn/calibration/matformer_llama_148m_20m_tokens`

Batch 2 identifies the intercept process variance as the critical calibration
knob. Increasing it from $10^{-8}$ to $10^{-6}$ is insufficient, but
$q_I=10^{-5}$ prevents the catastrophic medium-only failure and brings the
overall standardized-innovation distribution close to a useful scale.

The completed batch-2 results are:

| Controller setting | Mean loss | Micro | Small | Medium | Large | Full | Action counts | Entropy | $\operatorname{sd}(z_t)$ | 95% coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| $\sigma^2=10^{-7}$, $q_I=10^{-6}$, $q_C=10^{-9}$ | 5.8799 | 8.0327 | 6.0541 | 5.1035 | 5.1043 | 5.1050 | 0 / 1 / 96 / 0 / 0 | 0.036 | 5.17 | 0.649 |
| $\sigma^2=10^{-7}$, $q_I=10^{-5}$, $q_C=10^{-9}$ | 5.1942 | 5.2082 | 5.2044 | 5.1856 | 5.1860 | 5.1868 | 37 / 1 / 52 / 5 / 2 | 0.610 | 1.65 | 0.948 |
| $\sigma^2=10^{-6}$, $q_I=10^{-5}$, $q_C=10^{-9}$ | 5.1917 | 5.2130 | 5.2059 | 5.1796 | 5.1796 | 5.1802 | 33 / 1 / 54 / 6 / 3 | 0.634 | **1.52** | **0.959** |
| $\sigma^2=10^{-7}$, $q_I=10^{-5}$, $q_C=10^{-8}$ | **5.1903** | 5.2157 | 5.2042 | 5.1780 | 5.1760 | 5.1774 | 31 / 2 / 49 / 8 / 7 | **0.736** | 1.65 | 0.948 |

Action counts are again ordered as

$$
(\text{micro},\text{small},\text{medium},\text{large},\text{full}).
$$

The 95% coverage column is the empirical fraction of standardized innovations
inside $[-1.96,1.96]$. It should not be interpreted without the phase-wise
diagnostics below, because a small number of large early errors coexist with
very conservative predictions late in training.

###### $q_I=10^{-6}$ is still below the required scale

The $q_I=10^{-6}$ run exactly reproduces the failed medium-only trajectory:
it chooses `small` once and `medium` for the remaining 96 windows. Its scaling
losses are identical to the failed batch-1 runs. Increasing $q_I$ from
$10^{-8}$ to $10^{-6}$ therefore does not cross the threshold required to
absorb the rapid early change in the common reward level.

Its standardized-innovation standard deviation falls from 13.90 in the viable
batch-1 reference to 5.17, but this is not enough to prevent premature action
collapse. Innovation calibration is useful only if it changes posterior credit
assignment before an arm is eliminated.

###### $q_I=10^{-5}$ fixes the largest calibration error

All three $q_I=10^{-5}$ runs avoid the catastrophic failure. Relative to the
batch-1 viable reference

$$
(\sigma^2,q_I,q_C)=(10^{-7},10^{-8},10^{-9}),
$$

their standardized-innovation spread improves from

$$
\operatorname{sd}(z_t)=13.90
$$

to between 1.52 and 1.65. Their overall 95% predictive coverage is between
0.948 and 0.959, close to the nominal value. This is strong evidence that the
controller needed a much more responsive intercept state rather than only a
larger scalar observation variance.

The corresponding per-round intercept drift standard deviation is

$$
\sqrt{q_I}=\sqrt{10^{-5}}\approx3.16\times10^{-3}.
$$

This is large enough to absorb a meaningful fraction of the initial reward
transition without forcing the action contrasts to explain all of it. Because
the intercept contributes equally to every candidate action score, increasing
its variance does not directly favor a granularity.

###### Effect of observation variance at $q_I=10^{-5}$

With $q_C=10^{-9}$ fixed, increasing observation variance from $10^{-7}$ to
$10^{-6}$ changes mean loss from 5.1942 to 5.1917 and standardized-innovation
spread from 1.65 to 1.52. This is a small calibration improvement, but it does
not broaden the action distribution much: action entropy changes only from
0.610 to 0.634.

The loss difference is only 0.0025 in a one-seed comparison, so it is not
evidence that $\sigma^2=10^{-6}$ is intrinsically better. Its main benefit is
slightly safer predictive coverage. The larger observation variance is also
more conservative than necessary after the initial phase.

###### Effect of increasing contrast process variance

At $\sigma^2=10^{-7}$ and $q_I=10^{-5}$, increasing $q_C$ from $10^{-9}$ to
$10^{-8}$ produces the broadest action coverage among the calibrated batch-2
runs:

$$
(37,1,52,5,2)
\longrightarrow
(31,2,49,8,7).
$$

Normalized entropy increases from 0.610 to 0.736. Large and full are revisited
more often, and small is revisited once after the first quarter. Mean loss also
improves from 5.1942 to 5.1903, the best value in batch 2, although the
difference is too small to establish a performance ranking from one seed.

The final posterior under $q_C=10^{-8}$ has approximate action-score means and
standard deviations

$$
\begin{array}{c|r|r}
\text{action} & \widehat\mu_g & \operatorname{sd}(\mu_g)\\
\hline
\text{micro}  &  6.77\times10^{-7} & 3.15\times10^{-4}\\
\text{small}  & -2.58\times10^{-3} & 1.51\times10^{-3}\\
\text{medium} & -4.51\times10^{-5} & 9.28\times10^{-4}\\
\text{large}  & -3.71\times10^{-4} & 1.05\times10^{-3}\\
\text{full}   & -5.44\times10^{-4} & 1.13\times10^{-3}.
\end{array}
$$

No pairwise mean difference exceeds approximately 1.75 posterior standard
deviations. This is much healthier than the batch-1 viable run, where one early
large observation produced a difference of approximately 10.2 standard
deviations and eliminated the action. The $q_C=10^{-8}$ posterior can recover
from early evidence rather than treating it as permanent.

###### Remaining phase-wise failure

Despite the improved overall calibration, all three $q_I=10^{-5}$ runs have
nearly the same first-quarter policy. Their first 24 windows contain

$$
(0,1,22,0,1)
$$

micro, small, medium, large, and full selections. The first 20 actions are also
identical across these runs: one small action, three medium actions, one full
action, and then medium for the remainder of that prefix. The hyperparameter
changes mainly affect whether actions recover later; they do not prevent the
controller from concentrating before receiving representative evidence about
all five actions.

For the provisional best $q_C=10^{-8}$ run, phase-wise diagnostics are:

$$
\begin{array}{c|ccccc|r|r}
 & \text{micro} & \text{small} & \text{medium} & \text{large} & \text{full}
 & \operatorname{mean}(z_t) & \operatorname{sd}(z_t)\\
\hline
Q_1 & 0  & 1 & 22 & 0 & 1 & -0.26 & 3.28\\
Q_2 & 10 & 0 & 12 & 1 & 1 &  0.03 & 0.67\\
Q_3 & 8  & 1 & 9  & 3 & 3 &  0.00 & 0.29\\
Q_4 & 13 & 0 & 6  & 4 & 2 &  0.00 & 0.11.
\end{array}
$$

The controller remains overconfident in the first quarter and becomes highly
conservative later. Overall coverage near 95% hides this phase mismatch. A
single stationary observation variance cannot simultaneously match the rapid
initial reward transition and the much smaller late-training innovations.

###### Provisional selected configuration

Among the configurations implemented so far, the best reference for further
work is

$$
\sigma^2=10^{-7},
\qquad
Q=\operatorname{diag}
\left(10^{-5},10^{-8},10^{-8},10^{-8},10^{-8}\right),
$$

with

$$
V_0=\operatorname{diag}
\left(10^{-4},10^{-6},10^{-6},10^{-6},10^{-6}\right),
\qquad h=50.
$$

It has the lowest mean loss in batch 2, the highest entropy among the calibrated
$q_I=10^{-5}$ runs, broader late action coverage, near-nominal overall
predictive coverage, and no posterior action separation above two standard
deviations. The performance difference is not yet statistically established;
this selection is based on the joint controller diagnostics, not only its
0.001--0.004 loss advantage.

###### Consequence for the next experiment

Do not submit the original batch-3 covariance grid. The stopping condition
defined after batch 1 has been reached: persistent initial action concentration
remains even after the overall innovation scale is repaired. Another broad
sweep of fixed $\sigma^2$, $q_I$, and $q_C$ would mainly adjust how quickly
actions recover from biased early evidence, rather than remove its source.

The next method-level experiment should add an explicit controller-start or
balanced-burn-in contract. A concrete initial design is:

1. During scheduler warmup, train with a deterministic balanced global
   schedule rather than Thompson exploitation.
2. Ensure every granularity is selected at least twice. With five actions and
   $h=50$, ten windows correspond to 500 optimizer steps, close to the 400-step
   scheduler warmup in these 20M-token runs.
3. At the controller start boundary, establish a fresh fixed-panel objective
   baseline and initialize or reset the Bayesian posterior so that the rapid
   initial learning transition is not encoded as a lasting action contrast.
4. Start Thompson sampling with the provisional selected covariance values.
5. Save the burn-in schedule, controller start step, posterior-reset event, and
   first adaptive window in the controller artifacts.

Resetting the posterior deliberately discards early controller rewards. Those
rewards describe a strongly nonstationary phase and have already been shown to
produce misleading action contrasts. If retaining them is desired instead,
the reward model needs an explicit progress or learning-rate interaction; a
balanced action order alone does not remove time confounding.

Before promoting the provisional configuration to a 100M-token result, compare
the burn-in version against the current no-burn-in version and replicate the
comparison with additional root seeds. If implementation of burn-in is
deferred, the $q_C=10^{-8}$ configuration is the safest current choice, but it
should still be described as vulnerable to first-quarter action concentration.

#### Submission batch 3: balanced pre-adaptive warmup

This batch replaces the superseded covariance grid. It holds the provisional
selected controller configuration fixed and tests the new 500-step balanced
pre-adaptive warmup. The seed-42 warmup run pairs with the already completed
batch-2 no-warmup run
`d256-ag-cgmc-cal20m-h50-r1p0em7-qi1p0em5-qc1p0em8-pc1p0em6-s42`.
Seeds 43 and 44 each receive a fresh matched no-warmup/warmup pair.

Every balanced warmup uses ten 50-step global-action windows. The five
granularities therefore appear exactly twice before the controller evaluates
its first fixed-panel baseline at step 500. Warmup actions must not count as
controller observations or adaptive action selections, and the posterior must
still equal its configured prior at that boundary.

```bash
sbatch --gres=gpu:1 --time=04:00:00 "$script" \
  --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random \
  --run-id d256-ag-cgmc-bw500-20m-h50-r1p0em7-qi1p0em5-qc1p0em8-pc1p0em6-s42 \
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
  --override 'model.adaptive_controller.process_noise_covariance=[1.0e-5,1.0e-8,1.0e-8,1.0e-8,1.0e-8]'
```

```bash
sbatch --gres=gpu:1 --time=04:00:00 "$script" \
  --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random \
  --run-id d256-ag-cgmc-nowarm-20m-h50-r1p0em7-qi1p0em5-qc1p0em8-pc1p0em6-s43 \
  --override run.seed=43 \
  --override training.token_budget=20000000 \
  --override training.warmup_steps=400 \
  --override training.pre_nested_warmup.enabled=false \
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
  --override 'model.adaptive_controller.process_noise_covariance=[1.0e-5,1.0e-8,1.0e-8,1.0e-8,1.0e-8]'
```

```bash
sbatch --gres=gpu:1 --time=04:00:00 "$script" \
  --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random \
  --run-id d256-ag-cgmc-bw500-20m-h50-r1p0em7-qi1p0em5-qc1p0em8-pc1p0em6-s43 \
  --override run.seed=43 \
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
  --override 'model.adaptive_controller.process_noise_covariance=[1.0e-5,1.0e-8,1.0e-8,1.0e-8,1.0e-8]'
```

```bash
sbatch --gres=gpu:1 --time=04:00:00 "$script" \
  --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random \
  --run-id d256-ag-cgmc-nowarm-20m-h50-r1p0em7-qi1p0em5-qc1p0em8-pc1p0em6-s44 \
  --override run.seed=44 \
  --override training.token_budget=20000000 \
  --override training.warmup_steps=400 \
  --override training.pre_nested_warmup.enabled=false \
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
  --override 'model.adaptive_controller.process_noise_covariance=[1.0e-5,1.0e-8,1.0e-8,1.0e-8,1.0e-8]'
```

```bash
sbatch --gres=gpu:1 --time=04:00:00 "$script" \
  --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random \
  --run-id d256-ag-cgmc-bw500-20m-h50-r1p0em7-qi1p0em5-qc1p0em8-pc1p0em6-s44 \
  --override run.seed=44 \
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
  --override 'model.adaptive_controller.process_noise_covariance=[1.0e-5,1.0e-8,1.0e-8,1.0e-8,1.0e-8]'
```

##### Results and interpretation of submission batch 3

Date analyzed: 2026-08-07

Results directory:
`/nfs-stor/nicolas.avila/results/elasticnn/calibration/matformer_llama_148m_20m_tokens`

All five new jobs completed 4,883 optimizer steps and exactly 20M tokens. The
completed seed-42 batch-2 run supplies the sixth row and the no-warmup member
of that pair. Within each seed, the ordinary-validation manifest hashes match,
so each warmup comparison uses the same evaluation examples.

The final scaling losses are:

| Policy | Seed | Mean loss | Micro | Small | Medium | Large | Full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No warmup | 42 | 5.1903 | 5.2157 | 5.2042 | 5.1780 | 5.1760 | 5.1774 |
| Balanced 500 | 42 | **5.1792** | 5.2311 | 5.1814 | 5.1646 | 5.1598 | 5.1590 |
| No warmup | 43 | **5.1713** | 5.2608 | 5.1542 | 5.1475 | 5.1464 | 5.1475 |
| Balanced 500 | 43 | 5.1978 | 5.2725 | 5.1924 | 5.1750 | 5.1733 | 5.1760 |
| No warmup | 44 | 5.4489 | 6.5955 | 5.1704 | 5.1610 | 5.1592 | 5.1583 |
| Balanced 500 | 44 | **5.2144** | 5.2470 | 5.2169 | 5.2050 | 5.2015 | 5.2018 |

Mean loss is the uniform mean of the five reported granularity losses. The
adaptive action counts below exclude the ten scheduled warmup windows, as
required by the artifact contract. Counts are ordered as

$$
(\text{micro},\text{small},\text{medium},\text{large},\text{full}).
$$

| Policy | Seed | Adaptive action counts | Entropy | $\operatorname{sd}(z_t)$ | 95% coverage | Largest final pair separation |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| No warmup | 42 | 31 / 2 / 49 / 8 / 7 | 0.736 | 1.650 | 0.948 | 1.75 sd |
| Balanced 500 | 42 | 17 / 19 / 17 / 14 / 20 | **0.996** | 0.062 | 1.000 | 0.12 sd |
| No warmup | 43 | 10 / 48 / 21 / 17 / 1 | 0.787 | 1.842 | 0.928 | 2.20 sd |
| Balanced 500 | 43 | 10 / 22 / 22 / 22 / 11 | **0.965** | 0.093 | 1.000 | 0.11 sd |
| No warmup | 44 | 1 / 36 / 12 / 34 / 14 | 0.820 | 2.106 | 0.845 | 2.28 sd |
| Balanced 500 | 44 | 20 / 19 / 14 / 17 / 17 | **0.996** | 0.095 | 1.000 | 0.08 sd |

Here the standardized innovation is

$$
z_t=
\frac{r_t-x_t^\top m_t^-}
{\sqrt{x_t^\top V_t^-x_t+\sigma^2}},
$$

and pair separation is the largest absolute posterior difference between two
action scores divided by its posterior standard deviation.

###### Warmup lifecycle audit

All three balanced runs satisfy the intended lifecycle contract:

- the ten warmup windows are contiguous 50-step windows covering steps
  0--500;
- every scheduled action is global and is repeated across all 16 transformer
  blocks;
- each granularity has exactly two completed warmup windows;
- all warmup events record `posterior_updated: false`;
- the initial fixed-panel boundary is at step 500 and references the warmup
  schedule hash;
- `prior_untouched` is true at that boundary, and the first completed adaptive
  window ends at step 550;
- controller summaries contain 87 observations and 88 evaluations, compared
  with 97 observations and 98 evaluations without warmup. The ten warmup
  windows do not contaminate either count or the adaptive action frequencies.

The independently seeded schedules are:

- seed 42: schedule seed `7932627976006192913`, schedule
  `full, large, medium, micro, small, medium, small, micro, full, large`, hash
  `b3c5ccaa791be7ed9d3e93d4c3d9fe472a9dc29f254d9c045f87c5d4b2f52488`;
- seed 43: schedule seed `9086074912419706454`, schedule
  `small, micro, large, full, medium, micro, small, medium, full, large`, hash
  `b694ad242355dbaa20bfa067b1bf3c2a97fd2eb91f4764606ec64b482685ac7c`;
- seed 44: schedule seed `7060123897221414229`, schedule
  `small, medium, large, full, micro, micro, medium, large, full, small`, hash
  `add53403e51ea260eaea0364779c2d7f78eba6ac32311eccb67a6dc29620fb66`.

The first Thompson action is also identical within each paired seed despite
moving from step 0 to step 500: `small` for seed 42, `full` for seed 43, and
`micro` for seed 44. This is consistent with the posterior and controller RNG
remaining unused during warmup.

###### Balanced warmup removes the early action poisoning

The first-quarter adaptive policies make the effect unambiguous:

| Policy | Seed | First-quarter action counts | $\operatorname{sd}(z_t)$ in first quarter |
| --- | ---: | --- | ---: |
| No warmup | 42 | 0 / 1 / 22 / 0 / 1 | 3.280 |
| Balanced 500 | 42 | 3 / 7 / 5 / 4 / 2 | 0.098 |
| No warmup | 43 | 0 / 14 / 7 / 2 / 1 | 3.730 |
| Balanced 500 | 43 | 2 / 5 / 5 / 6 / 3 | 0.152 |
| No warmup | 44 | 1 / 0 / 1 / 22 / 0 | 4.152 |
| Balanced 500 | 44 | 2 / 6 / 4 / 3 / 6 | 0.165 |

The no-warmup runs still concentrate on whichever action receives favorable
credit during the rapid initial transition: medium for seed 42, small for seed
43, and large for seed 44. In contrast, every balanced run explores all five
actions immediately and retains high action entropy through the complete
adaptive phase. Mean normalized entropy rises from 0.781 without warmup to
0.985 with warmup, and its across-seed standard deviation falls from 0.042 to
0.018.

Seed 44 demonstrates the practical value of this protection. Its no-warmup
controller selects `micro` only once, and the final micro loss diverges to
6.5955 while the other four losses remain near 5.16--5.17. Balanced warmup
prevents that starvation and reduces micro loss by 1.3485. The warmup method
therefore removes the catastrophic tail behavior seen when an action is
effectively eliminated before representative evidence is available.

###### The apparent mean-loss gain is not yet a general performance gain

The paired warmup-minus-no-warmup mean-loss differences are

$$
(-0.0111,\ +0.0266,\ -0.2345)
$$

for seeds 42, 43, and 44. Their mean is $-0.0730$, but their sample standard
deviation is 0.1411 and the average is dominated by the seed-44 micro failure.
Across only small, medium, large, and full, warmup changes mean loss by
$-0.0177$, $+0.0303$, and $+0.0441$ in the three seeds. Thus seed 42 improves,
while seeds 43 and 44 become worse on those four granularities.

Across seeds, warmup reduces the standard deviation of uniform mean loss from
0.1551 to 0.0176. That is evidence for robustness, not yet for better expected
performance. With only three paired seeds and one severe no-warmup outlier,
these results do not establish that balanced warmup lowers ordinary scaling
loss on average.

###### The selected covariance is now much too conservative

Warmup also changes the relevant calibration regime. Without warmup, the raw
innovation standard deviations are 0.0060--0.0073 because the controller sees
the initial rapid objective change. After starting at step 500, they fall to
0.00024--0.00033. The median predictive standard deviation nevertheless stays
near 0.00337 because the inherited process covariance still uses
$q_I=10^{-5}$.

Consequently, the balanced runs have standardized-innovation deviations of
only 0.062--0.095 and 100% empirical 95% coverage. This is severe
over-dispersion, not improved nominal calibration. Final action-score mean
differences are at most 0.12 posterior standard deviations in all three runs,
so Thompson sampling remains nearly uniform because its uncertainty is much
larger than the evidence observed after warmup.

The covariance selected in batch 2 was calibrated to absorb the very early
learning transient. Balanced warmup successfully removes that transient from
the controller data, so retaining $q_I=10^{-5}$ solves a problem that the
controller no longer observes and prevents useful posterior contraction.

###### Consequence for the next experiment

Balanced pre-adaptive warmup should be retained: it satisfies the lifecycle
contract, removes seed-dependent first-quarter collapse, and prevents action
starvation. The current covariance values should not yet be promoted to a
100M-token result, however.

The next short calibration batch should hold the 500-step balanced schedule,
$h=50$, and $\sigma^2=10^{-7}$ fixed while reducing the intercept process
variance below $10^{-5}$. The purpose is to calibrate the posterior to the
post-warmup innovation scale, not to revisit the discarded early transient.
Contrast process variance should be changed only after finding an intercept
scale that brings phase-wise standardized innovations materially closer to
unit scale without recreating action collapse. A blocked-uniform control at the
same 50-step interval remains useful for separating the benefit of balanced
training exposure from posterior-guided action selection.

#### Submission batch 4: post-warmup process-noise recalibration

This batch recalibrates the fixed process covariance in the regime created by
balanced warmup. Hold the 500-step balanced schedule, $h=50$,
$\sigma^2=10^{-7}$, $q_C=10^{-8}$, and
$V_0=\operatorname{diag}(10^{-4},10^{-6},10^{-6},10^{-6},10^{-6})$
fixed. Sweep only

$$
q_I\in\{0,10^{-8},10^{-7},10^{-6}\},
\qquad
Q=\operatorname{diag}(q_I,10^{-8},10^{-8},10^{-8},10^{-8}).
$$

The completed balanced seed-42 run with $q_I=10^{-5}$ is the reference and
must not be rerun. Submit these four seed-42 jobs first; inspect their
phase-wise standardized innovations and action trajectories before replicating
the best candidates or changing $q_C$.

```bash
export OUT=/nfs-stor/nicolas.avila/results/elasticnn
repo=/l/users/nicolas.avila/dev/references/matformer
script="$repo/scripts/slurm_dmodel256_pilot.sh"
labels='[micro,small,medium,large,full]'
prefixes='{micro: 0.125, small: 0.25, medium: 0.5, large: 0.75, full: 1.0}'

cd "$repo"
mkdir -p logs

for qi_spec in \
  'qi0:0' \
  'qi1p0em8:1.0e-8' \
  'qi1p0em7:1.0e-7' \
  'qi1p0em6:1.0e-6'
do
  qi_label=${qi_spec%%:*}
  qi_value=${qi_spec#*:}

  sbatch --gres=gpu:1 --time=04:00:00 "$script" \
    --repo-root "$repo" --output-root "$OUT" \
    --mode nested-random \
    --run-id "d256-ag-cgmc-bw500-qrecal20m-h50-r1p0em7-${qi_label}-qc1p0em8-pc1p0em6-s42" \
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
    --override "model.adaptive_controller.process_noise_covariance=[${qi_value},1.0e-8,1.0e-8,1.0e-8,1.0e-8]"
done
```

##### Results and interpretation of submission batch 4

Date analyzed: 2026-08-07

Results directory:
`/nfs-stor/nicolas.avila/results/elasticnn/calibration/matformer_llama_148m_20m_tokens`

All four new jobs completed 4,883 optimizer steps and exactly 20M tokens. They
share the seed-42 data manifests, balanced schedule, schedule hash, and first
adaptive action with the $q_I=10^{-5}$ reference. Every run has ten balanced
warmup windows, two selections per granularity, an untouched prior through step
500, 87 completed controller observations, 88 controller evaluations, and no
posterior update during warmup. The terminal 33-step partial window emits no
observation.

The final ordinary scaling losses are:

| $q_I$ | Mean loss | Micro | Small | Medium | Large | Full |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| $0$ | 5.1779 | 5.2277 | 5.1772 | 5.1641 | 5.1603 | 5.1604 |
| $10^{-8}$ | **5.1766** | 5.2308 | 5.1751 | 5.1619 | 5.1577 | 5.1577 |
| $10^{-7}$ | 5.1787 | 5.2242 | 5.1786 | 5.1663 | 5.1623 | 5.1622 |
| $10^{-6}$ | 5.1804 | 5.2215 | 5.1819 | 5.1688 | 5.1648 | 5.1648 |
| $10^{-5}$ reference | 5.1792 | 5.2311 | 5.1814 | 5.1646 | 5.1598 | 5.1590 |

The controller diagnostics are:

| $q_I$ | Adaptive action counts | Entropy | $\operatorname{mean}(z_t)$ | $\operatorname{sd}(z_t)$ | RMS $z_t$ | Median predictive sd | 95% coverage | Largest final pair separation |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| $0$ | 16 / 25 / 14 / 16 / 16 | 0.986 | -0.561 | **0.404** | 0.690 | $4.20\times10^{-4}$ | 1.000 | 0.42 sd |
| $10^{-8}$ | 15 / 26 / 15 / 13 / 18 | 0.980 | -0.111 | 0.330 | 0.347 | $4.73\times10^{-4}$ | 1.000 | 0.06 sd |
| $10^{-7}$ | 18 / 25 / 15 / 12 / 17 | 0.982 | -0.034 | 0.279 | 0.280 | $6.42\times10^{-4}$ | 1.000 | 0.08 sd |
| $10^{-6}$ | 20 / 21 / 16 / 13 / 17 | 0.991 | -0.012 | 0.174 | 0.174 | $1.240\times10^{-3}$ | 1.000 | 0.09 sd |
| $10^{-5}$ reference | 17 / 19 / 17 / 14 / 20 | 0.996 | -0.003 | 0.062 | 0.061 | $3.366\times10^{-3}$ | 1.000 | 0.12 sd |

Counts are ordered as micro / small / medium / large / full and exclude the
warmup schedule. All settings retain high action entropy and select every
granularity at least 12 times, so reducing $q_I$ does not recreate the
pre-warmup action-starvation failure on this seed. The loss spread is only
0.0037, and the lowest mean at $q_I=10^{-8}$ is not evidence of a performance
difference from one seed.

###### Recalibrating $q_I$ alone is insufficient

Lowering $q_I$ monotonically reduces predictive uncertainty and moves
$\operatorname{sd}(z_t)$ toward one, but no candidate reaches the calibration
criterion. Even at $q_I=0$, $\operatorname{sd}(z_t)=0.404$ and empirical 95%
coverage remains 100%. The phase-wise deviations make the residual mismatch
visible:

| $q_I$ | Quarter 1 | Quarter 2 | Quarter 3 | Quarter 4 |
| ---: | ---: | ---: | ---: | ---: |
| $0$ | 0.593 | 0.390 | 0.311 | 0.263 |
| $10^{-8}$ | 0.490 | 0.325 | 0.224 | 0.107 |
| $10^{-7}$ | 0.378 | 0.344 | 0.207 | 0.088 |
| $10^{-6}$ | 0.257 | 0.201 | 0.128 | 0.056 |
| $10^{-5}$ reference | 0.098 | 0.048 | 0.060 | 0.022 |

The raw innovation standard deviations across the five runs are only
$2.17\times10^{-4}$--$2.60\times10^{-4}$. In contrast, the observation-noise
standard deviation alone is

$$
\sqrt{\sigma^2}=\sqrt{10^{-7}}=3.16\times10^{-4}.
$$

Thus, even after setting all process uncertainty to zero, the fixed
observation model already has a variance floor above the observed post-warmup
residual scale. No nonnegative $Q$ can repair this. The retained
$q_C=10^{-8}$ and posterior state uncertainty add further predictive variance.

$q_I=0$ is not a satisfactory workaround. Its standardized innovations have a
persistent negative mean in every quarter (-0.539, -0.502, -0.536, and
-0.664), showing that a frozen common intercept cannot track the changing
reward level. Its larger RMS statistic is therefore partly systematic lag,
not well-calibrated random variation. A small positive intercept process
variance remains necessary.

###### Consequence for the next calibration batch

Do not replicate this grid across seeds yet: none of its candidates satisfies
the calibration target. Retain $q_I=10^{-8}$ as the next center because it has
the lowest seed-42 mean loss, avoids the persistent intercept bias of $q_I=0$,
and preserves broad action coverage. This is a provisional calibration choice,
not a selected final configuration.

The next short seed-42 grid should jointly test

$$
\sigma^2\in\{10^{-8},3\times10^{-8}\},
\qquad
q_C\in\{0,10^{-9},10^{-8}\},
\qquad
q_I=10^{-8},
$$

with balanced warmup and all other settings unchanged. This separates the
observation floor from contrast drift while keeping enough common process
noise to track reward level. Select candidates using phase-wise innovation
mean and deviation, coverage, action starvation, and final action-score
separation; replicate seeds 43 and 44 only after this six-run grid identifies
a plausible calibration region. As before, the current controller keeps $Q$
fixed during each run rather than updating it online.
