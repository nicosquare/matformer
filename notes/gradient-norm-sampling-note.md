# Gradient-Norm Granularity Sampling

**Status**: exploratory mathematical note  
**Scope**: standalone adaptive granularity sampling for MatFormer training  
**Motivation**: adapt the useful gradient-importance idea from
[PA&DA](https://arxiv.org/abs/2302.14772) to nested MatFormer subnetworks without
assuming that the complete-subnetwork gradient norm is comparable across widths.

## Question

Given that only the sampled granularity produces an observed gradient at each
step, how should training update and normalize a sampling distribution over
all granularities? The gradient signal should also avoid an automatic
preference for wider subnetworks merely because they contain more active
parameters.

This note treats gradient-based sampling as its own training strategy. It does
not assume a held-out reward model or another adaptive controller.

## Notation

Let the ordered granularities be

$$
\mathcal{G} = \{g_1, \ldots, g_K\},
$$

from narrowest to widest. Let $A_g$ be the set of parameters active under a
global granularity $g$, so the slicing construction gives

$$
A_{g_1} \subset A_{g_2} \subset \cdots \subset A_{g_K}.
$$

For one training example or minibatch $z$, define

$$
L_g(\theta; z)
$$

as the loss under granularity $g$, and

$$
d_g = \nabla_{\theta_{A_g}} L_g(\theta; z)
$$

as its gradient on the active parameters. When needed, $d_g$ can be embedded
in the complete supernetwork parameter space by filling inactive coordinates
with zero.

## Why the complete $L_2$ norm is size-biased

The direct PA&DA-style score would be

$$
I_g^{\mathrm{total}} = \lVert d_g \rVert_2.
$$

Suppose, only to expose the dimensional effect, that the active gradient
coordinates have the same second moment:

$$
\mathbb{E}[d_{g,i}^2] = \sigma^2.
$$

If $N_g = |A_g|$, then

$$
\mathbb{E}\left[\lVert d_g \rVert_2^2\right]
= N_g \sigma^2.
$$

Consequently, the typical total norm scales approximately as

$$
\lVert d_g \rVert_2 \propto \sqrt{N_g}.
$$

A wider subnetwork can therefore receive a larger importance score even when
its typical gradient per active parameter is identical. Normalizing the scores
across granularities,

$$
p_g = \frac{I_g}{\sum_h I_h},
$$

does not remove this effect; it only turns the already size-biased scores into
probabilities.

## Gradient signal supplied to a policy

The score is an observation available to a sampling policy, not itself a
sampling method. The unit of importance is the complete global granularity,
not an individual block, projection, or parameter ring.

### Primary signal: active-parameter gradient RMS

Use one scalar for the gradient of the entire active subnetwork:

$$
I_g^{\mathrm{rms}}
= \frac{\lVert d_g \rVert_2}{\sqrt{N_g}}
= \sqrt{\frac{1}{N_g}\sum_{i\in A_g}d_{g,i}^2}.
$$

Under the equal-coordinate-second-moment example above,
$\mathbb{E}[(I_g^{\mathrm{rms}})^2]=\sigma^2$, independent of $N_g$. The score
therefore measures average gradient energy per active parameter while keeping
the complete granularity as the action and measurement unit.

Different granularities intentionally use different active supports: those
supports define the models whose importance is being compared. The RMS
normalization removes the leading dimensional effect, but cannot guarantee
that differences in parameter type, scale, or optimization dynamics are
irrelevant. Those are empirical properties to diagnose rather than reasons to
split the score into architectural components.

### Diagnostic signals

The following scores remain useful for testing whether the primary signal
removes width bias, but need not each define a policy:

- **Complete gradient norm**:
  $I_g^{\mathrm{total}}=\lVert d_g\rVert_2$. This is the uncorrected reference
  expected to favor wider subnetworks.
- **Active relative gradient**:
  $I_g^{\mathrm{relative}}
  =\lVert d_g\rVert_2/(\lVert\theta_{A_g}\rVert_2+\varepsilon)$. This is a
  dimensionless whole-granularity diagnostic. Per-coordinate division by
  $|\theta_i|$ should be avoided because near-zero parameters are unstable.

The multiplication by learning rate is unnecessary for comparing actions at a
single step because it is common to every action. With AdamW, however, raw
gradient scores do not equal eventual preconditioned update magnitudes. An
optimizer-aware whole-granularity score is a possible later experiment.

## Partial-feedback problem

At an ordinary optimizer step, only $I_{G_t,t}$ is observed. It is therefore
impossible to compute a fresh normalization such as

$$
p_{g,t}=\frac{I_{g,t}}{\sum_h I_{h,t}}
$$

without evaluating every granularity. A one-arm policy must instead normalize
stored estimates or bandit weights. These values were measured at different
model states and are not contemporaneous scores.

## Candidate sampling policies

Partial-feedback policies should begin with equal weights or a balanced
warmup. Every adaptive policy should retain a permanent probability floor so
that every complete granularity continues receiving training.

### 1. Full-panel reference policy

On every decision step, compute $I_{g,t}$ for every granularity on the same
minibatch, normalize the resulting vector, and sample one granularity for the
optimizer update. This is the cleanest comparison because all scores share the
same parameters and data, but it requires approximately $K$ gradient
evaluations per decision. The unselected evaluations must remain
measurement-only so that they do not silently change the training objective.

This policy is primarily an oracle-style reference for cheaper policies.

### 2. Selective-EMA softmax policy

Maintain a per-granularity exponential moving average and update only the
sampled arm:

$$
S_{g,t}
= \begin{cases}
  \beta S_{g,t-1} + (1-\beta) I_{g,t}, & g = G_t, \\
  S_{g,t-1}, & g \ne G_t.
  \end{cases}
$$

Map the stored estimates through a temperature-controlled softmax:

$$
q_{g,t}
= \frac{\exp(\log(S_{g,t}+\eta)/T)}
       {\sum_h \exp(\log(S_{h,t}+\eta)/T)}.
$$

Equivalently, $q_g \propto (S_g+\eta)^{1/T}$. Here:

- $T>1$ flattens the distribution;
- $T<1$ sharpens it;
- $\eta>0$ makes zero or initially small estimates safe.

Mix the learned distribution with a permanent uniform floor:

$$
p_{g,t}
= (1-\epsilon)q_{g,t} + \frac{\epsilon}{K},
\qquad 0 < \epsilon \le 1.
$$

This floor is necessary because unselected arms have stale score estimates and
some granularities otherwise risk receiving no training. The policy is cheap
and simple, but frequently sampled arms are refreshed more often while rarely
sampled arms can remain stale. Its probabilities normalize asynchronous
estimates, not current scores.

### 3. Importance-weighted bandit policy

A bandit policy can account explicitly for observing arm $g$ with probability
$p_{g,t}$. For example, first transform the nonnegative score into a bounded
signal $r_{g,t}\in[0,1]$ using a declared scale and clipping rule. EXP3 forms
the estimate

$$
\widehat r_{g,t}
= \frac{\mathbb{1}[G_t=g]r_{G_t,t}}{p_{g,t}}
$$

and updates positive arm weights by

$$
w_{g,t+1}=w_{g,t}\exp(\lambda\widehat r_{g,t}),
$$

followed by normalization with a uniform exploration mixture. The estimator
corrects unequal observation probabilities in expectation and requires only
the selected score. Its main risks are high variance when $p_{g,t}$ is small
and sensitivity to the score-to-reward scale. Because training changes both
the model and future scores, standard fixed-reward regret interpretations do
not directly establish better model training.

### 4. Periodically calibrated hybrid policy

Every $H$ steps, evaluate every granularity on the same minibatch and refresh
all estimates together:

$$
S_{g,t}\leftarrow I_{g,t}
\qquad\text{for every }g\in\mathcal G.
$$

Between calibrations, use the selective-EMA update and softmax policy above.
This periodically restores a contemporaneous cross-granularity scale while
avoiding full-panel evaluation at every step. It introduces extra gradient
evaluations and a calibration interval $H$; as in the full-panel reference,
unselected calibration gradients are measurement-only.

## Optional exposure and compute terms

Gradient importance and sampling coverage are distinct concerns. Track the
exposure of each complete granularity through step $t$ as

$$
E_{g,t} = \sum_{s=1}^{t} \mathbb{1}[G_s=g].
$$

Coverage should be protected by balanced warmup and the uniform sampling floor
rather than folded into the importance signal.

If optimization is constrained by wall-clock compute rather than optimizer
steps or tokens, a cost-aware score is possible:

$$
\widetilde{S}_g
= \frac{S_g}{\operatorname{cost}(g)^\gamma},
\qquad \gamma \ge 0.
$$

- $\gamma=0$: prioritize gradient signal per optimizer step;
- $\gamma=1$: approximately prioritize gradient signal per unit compute;
- intermediate values trade off the two.

Cost correction changes the scientific objective and must be explicit in
method provenance.

## Does the sampled loss need inverse-probability weighting?

This is separate from both the gradient signal and the policy used to update
sampling probabilities.

Suppose the target training objective is the uniform global-granularity loss

$$
J(\theta)
= \frac{1}{K}\sum_{g=1}^{K}
  \mathbb{E}_z[L_g(\theta;z)].
$$

If $G \sim p$ and the selected gradient is used without reweighting, then

$$
\mathbb{E}[d_G]
= \sum_g p_g\,\mathbb{E}[d_g],
$$

which generally differs from $\nabla J$. This is an intentional adaptive
curriculum: high-score granularities affect the learned parameters more often.

If preserving the uniform objective is required, use

$$
\widehat{d}
= \frac{1}{Kp_G}d_G,
$$

for which

$$
\mathbb{E}[\widehat{d}]
= \frac{1}{K}\sum_g \mathbb{E}[d_g].
$$

The variance-minimizing sampling distribution for this estimator is linked to
the complete transformed gradient norm, not necessarily to the proposed
active-gradient RMS. Therefore:

- active-gradient RMS plus no inverse weighting is best described as a
  gradient-priority curriculum;
- active-gradient RMS plus inverse weighting preserves the uniform objective but
  is a heuristic proposal distribution rather than the exact variance-optimal
  one;
- complete-gradient norm plus inverse weighting is closest to classical
  importance sampling, but retains the width concern motivating this note.

The first pilot should make this distinction explicit rather than calling both
variants the same method.

If inverse weighting is used, the observed gradient norm is also multiplied by
$w_G=1/(Kp_G)$. Because norms are homogeneous,

$$
\lVert w_G d_G \rVert_2 = w_G\lVert d_G \rVert_2,
$$

so the unweighted score can be recovered by dividing the measured norm by
$w_G$, provided no clipping has yet occurred.

## Interaction with existing gradient correction and clipping

The sampling score should be measured at a precisely defined point in the
optimizer transaction:

1. finish gradient accumulation for the selected action;
2. apply the existing gradient-membership correction semantics;
3. measure the score before global norm clipping;
4. clip the gradient if configured;
5. perform the optimizer update.

Measuring before clipping is necessary because clipping intentionally removes
the magnitude differences used by the sampler. Measuring after membership
correction makes the score describe the gradient direction actually presented
to the optimizer. A raw, pre-correction norm can also be logged diagnostically.

For slicing with gradient-membership correction, the initial score should use
the corrected gradients over the complete active support. For concat with
local post-step correction, the relationship between gradient norm and actual
update is more complicated; the first pilot should therefore use the global
slicing path.

In distributed training, squared norms must be reduced across all parameter
shards before taking the square root. The measured quantity and the sampler
state must be identical on every rank, with one rank authoritative for the
sampling draw.

## Global granularity is the action

The sampler chooses one complete global granularity per optimizer step and
assigns it one scalar importance observation. It does not estimate importance
for blocks, projections, or parameter rings. Per-block decisions would require
architectural credit assignment and define a different problem closer to
neural architecture search; they are outside the scope of this proposal.

## Proposed first pilot

Use global, per-optimizer-step granularity sampling with a balanced warmup and
hold the observation signal fixed at

$$
I_g = I_g^{\mathrm{rms}}.
$$

Initially keep $\gamma=0$ and use the ordinary selected loss so policy effects
are not confounded with inverse-probability weighting. Compare:

- uniform global sampling;
- the selective-EMA softmax policy;
- the periodically calibrated hybrid for a small set of $H$ values;
- a short full-panel run as the reference for tracking error and behavior.

Evaluate EXP3 only after specifying and validating a fixed bounded
score-to-reward transformation. In a separate experiment, compare ordinary
selected loss (`gradient_priority_global`) with inverse-probability-weighted
loss (`gradient_proposal_uniform_objective_global`). The latter is not
`optimal_importance_sampling`, because active-gradient RMS is not proven to
minimize variance of the complete gradient estimator.

Whenever a granularity is evaluated, also log its complete gradient norm,
active parameter count, and whole-granularity relative-gradient norm. These
diagnose the RMS normalization without multiplying the main policy variants.

## Required diagnostics

For every optimizer step, record:

- selected granularity and its sampling probability;
- policy family, warmup state, and whether the step is a calibration step;
- complete pre-correction and post-correction gradient norms, where available;
- active-parameter count and active gradient RMS;
- optional whole-granularity relative-gradient norm;
- current estimator or bandit weight for every granularity and the age of each
  arm's last observation;
- sampling entropy and minimum/maximum probability;
- global clipping threshold and returned pre-clipping norm;
- clipping coefficient or whether clipping occurred;
- selected loss and valid-target count;
- optional estimated FLOPs or measured step time;
- cumulative exposure count for every granularity.

These diagnostics should determine whether the full norm is mostly a proxy for
width, how far asynchronous estimates drift from full-panel measurements, and
whether the sampler collapses onto a small subset of actions.

## Hypotheses to test

1. **Width bias**: the complete gradient norm is strongly monotonic with active
   parameter count or granularity width.
2. **RMS correction**: active gradient RMS has materially less correlation with
   width than the complete norm.
3. **Granularity-level signal**: active gradient RMS is stable enough across
   minibatches to provide useful evidence about complete-granularity
   importance.
4. **Asynchronous-estimate error**: selective-EMA estimates are less accurate
   for rarely sampled or long-unobserved granularities.
5. **Calibration benefit**: periodic full-panel calibration reduces that
   error and prevents policy decisions from being dominated by stale scores.
6. **Useful adaptation**: gradient-driven sampling improves uniform validation
   or final-holdout results at matched tokens or compute relative to uniform
   global sampling.
7. **Coverage safety**: balanced warmup plus the uniform floor prevents any
   granularity from becoming effectively untrained.
8. **Objective distinction**: the adaptive-curriculum and inverse-weighted
   variants behave differently enough that they must not be pooled under one
   method label.

## Open mathematical decisions

1. Which partial-feedback policy best approximates the full-panel reference at
   acceptable compute: selective EMA, calibrated EMA, or EXP3?
2. What calibration interval $H$ gives a useful freshness/compute tradeoff?
3. What probability floor and temperature prevent stale-score feedback and
   policy collapse without suppressing adaptation?
4. What fixed bounded transformation makes the gradient signal suitable for
   EXP3 without using unobserved cross-arm normalization?
5. Is the desired target a uniform all-granularity objective, or is changing
   training exposure according to estimated need acceptable?
6. Should $N_g$ count every unique trainable active scalar, and how should
   tied or currently gradient-free parameters be handled?
7. Does division by $\sqrt{N_g}$ adequately correct width when the active
   support contains parameter families with different gradient scales?
8. Is the post-membership-correction gradient the intended object, or should
   correction and sampling be derived jointly?
9. Does an optimizer-aware proposed-update norm predict useful training better
   than active-gradient RMS under AdamW?
10. Should action probabilities account for FLOPs or wall-clock cost?
11. What measure of subsequent learning progress should be used to validate
   that the chosen gradient score represents useful training need rather than
   noisy or conflicting gradients?

Until these questions are resolved, the proposal is a family of
partial-feedback sampling policies driven by a motivated gradient signal. A
one-arm policy normalizes asynchronous estimates or bandit weights, not a
fresh vector of cross-granularity scores.
