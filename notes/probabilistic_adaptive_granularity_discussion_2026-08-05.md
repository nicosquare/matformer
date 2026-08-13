# Probabilistic Adaptive Granularity Sampling

Date: 2026-08-05

This note records the design and implementation rationale for replacing the
legacy `adaptive_per_block` heuristic with a probabilistic controller that
learns from reward. It also treats global adaptation as a first-class case.

## Short conclusion

The implemented initial design is a **Bayesian structured bandit** whose reward
is measured on a fixed controller holdout, not on consecutive training batches.
The same controller supports two action spaces:

- **global adaptive**: choose one granularity for the whole model;
- **per-block adaptive**: choose a vector containing one granularity per block.

For the per-block case, an additive Bayesian linear model is a good first
version. Adjacent-block interaction terms can be added later if the additive
model is too restrictive. This lets the data learn whether coherent or abrupt
profiles are useful. A profile-change penalty should not be present by default.

The implemented controller remains useful as a reproducible experimental
baseline, and balanced pre-adaptive warmup remains the selected start contract.
Fixed process-noise calibration has not produced meaningful posterior action
separation, however. Further manual $Q$ tuning is paused while an episodic
posterior-reset method with $Q=0$ is evaluated as the simpler treatment of
nonstationarity.

## What the legacy approach did

Let the profile selected at training step $t$ be

$$
P_t = (P_{t,1}, \ldots, P_{t,B}),
$$

where $B$ is the number of transformer blocks. The legacy code computed a
normalized Hamming distance from the preceding profile:

$$
c_t = \frac{1}{B}\sum_{b=1}^{B}
\mathbf{1}[P_{t,b} \ne P_{t-1,b}],
$$

and constructs the reward

$$
r_t = L_{t-1}^{\mathrm{train}} - L_t^{\mathrm{train}} - \lambda c_t.
$$

There are several distinct problems here.

1. $L_{t-1}^{\mathrm{train}}$ and $L_t^{\mathrm{train}}$ usually come from
   different batches. Their difference therefore includes batch difficulty.
2. The losses can also be produced by different profiles. The difference mixes
   profile quality with training progress.
3. $L_t^{\mathrm{train}}$ is measured before the optimizer update caused by
   step $t$. Assigning this reward to $P_t$ does not measure how useful it
   was to train with $P_t$.
4. The same scalar reward is assigned to every selected block/granularity pair.
   This provides no principled credit assignment among blocks.
5. The change cost $c_t$ assumes, rather than discovers, that moving to a
   different complete profile is harmful.
6. The legacy "Thompson" implementation was a deterministic Gaussian
   perturbation of running scores, not sampling from an explicit Bayesian
   posterior.
7. With the legacy default `adaptive_sampler_decay_rate: 0.0`, the running
   reward means never change. In that configuration, reward cannot affect later
   choices at all.

The legacy field `correction_penalty` was specifically a profile-change penalty;
it was not a measurement of the model's membership or LMC correction.

## First decide what adaptation is trying to optimize

There are two plausible goals, and their rewards are different.

### Goal A: choose the profile that currently predicts best

A reward such as

$$
r_t = -L_H(\theta_t; P_t)
$$

learns which profile has the best heldout loss at the current parameters
$\theta_t$. This is useful for deployment routing, but it does not answer which
profile is most useful for training. It will also tend to prefer the largest or
otherwise easiest profile unless compute and quality targets are explicit.

### Goal B: choose the profile that produces the most useful training update

For adaptive training, the more relevant reward is the improvement in a fixed
heldout objective after training with a selected profile. This is the goal
recommended here.

The distinction is important: **profile quality** and **training utility** are
not the same latent quantity.

## A fixed controller holdout

Introduce a controller dataset $H_c$ that is disjoint from the optimizer's
training examples. Define a stable controller objective

$$
J_{H_c}(\theta)
= \sum_{g \in \mathcal{G}} w_g
L_{H_c}\!\left(\theta; (g,\ldots,g)\right),
\qquad \sum_g w_g = 1.
$$

Here $\mathcal{G}$ is the resolved granularity set. Evaluating the canonical
global subnetworks gives a stable and affordable measure of whether an update
helps the MatFormer family as a whole. The weights $w_g$ encode the intended
deployment mixture. Uniform weights are a reasonable experimental default, but
they are still an explicit objective choice.

An adaptive decision should cover a short window of $h$ optimizer steps:

1. At controller round $t$, select profile $A_t$.
2. Keep $A_t$ fixed for $h$ training steps.
3. Evaluate $J_{H_c}(\theta_{t+1})$ at the next boundary.
4. Attribute the boundary-to-boundary improvement to $A_t$.

The reward is

$$
r_t =
\frac{J_{H_c}(\theta_t)-J_{H_c}(\theta_{t+1})}{h}.
$$

This reward is evaluated only on $H_c$. Training data affects the parameter
update, as it must, but no training-batch loss appears in the reward.

Only one controller evaluation per boundary is needed: the post-evaluation for
round $t$ is the pre-evaluation for round $t+1$. Holding the profile fixed
for the decision window makes the delayed reward attributable to one action.
Changing the action every optimizer step and emitting one reward every few
steps would reintroduce ambiguous credit assignment.

### Data-split caution

Once $H_c$ updates the adaptive policy, it is no longer an unbiased validation
set. Conceptually it becomes meta-training data. The clean split is pairwise
disjoint. If
$\mathcal{S}=\{D_{\mathrm{train}},H_c,H_v,H_{\mathrm{final}}\}$, then

$$
S_i\cap S_j=\varnothing
\qquad\text{for all distinct }S_i,S_j\in\mathcal{S}.
$$

The roles are:

- $D_{\mathrm{train}}$: optimizer updates;
- $H_c$: adaptive-controller rewards;
- $H_v$: checkpoint selection and ordinary validation;
- $H_{\mathrm{final}}$: untouched final reporting, if available.

At minimum, the controller set and final comparison set must be separate.
Repeatedly observing one fixed holdout can overfit the controller to that
holdout even though model gradients never touch it.

For reproducibility, the controller set should have a saved manifest and hash,
fixed tokenization, fixed evaluation order, `model.eval()` behavior, and
target-token-weighted loss aggregation. If a full pass is too expensive, use a
fixed deterministic panel rather than drawing a fresh training-like batch.

## One probabilistic model for both scopes

Let an action be a profile

$$
A_t=(a_{t,1},\ldots,a_{t,B}),
\qquad a_{t,b}\in\mathcal{G}.
$$

The two action spaces are

$$
\mathcal{A}_{\mathrm{global}}
=\{(g,\ldots,g):g\in\mathcal{G}\},
$$

and

$$
\mathcal{A}_{\mathrm{per-block}}=\mathcal{G}^{B}.
$$

The global action space has only $|\mathcal{G}|$ arms and is the simplest
place to validate the reward protocol. The per-block space has
$|\mathcal{G}|^B$ profiles, so treating every complete profile as an
independent arm is impossible.

### Bayesian linear reward model

Represent a profile with features $x(A_t, z_t)$, where $z_t$ can include a
small amount of training context such as normalized progress. Model reward as

$$
r_t = x(A_t,z_t)^\top\beta_t + \epsilon_t,
\qquad \epsilon_t\sim\mathcal{N}(0,\sigma^2).
$$

Use a Gaussian belief over the coefficients:

$$
\beta_t\mid\mathcal{D}_{1:t-1}
\sim \mathcal{N}(m_t^-,V_t^-).
$$

Thompson sampling then has its literal Bayesian meaning:

$$
\widetilde\beta_t\sim\mathcal{N}(m_t^-,V_t^-),
$$

$$
A_t = \arg\max_{A\in\mathcal{A}}
x(A,z_t)^\top\widetilde\beta_t.
$$

After observing $r_t$, the Gaussian posterior update is

$$
k_t = \frac{V_t^-x_t}
{\sigma^2+x_t^\top V_t^-x_t},
$$

$$
m_t = m_t^- + k_t(r_t-x_t^\top m_t^-),
$$

$$
V_t = V_t^- - k_t x_t^\top V_t^-.
$$

These equations make it explicit how a reward changes both the expected value
and uncertainty of later actions.

### What should \(z_t\) contain?

The context \(z_t\) contains information known **before** action \(A_t\) is
selected. It lets the reward model express that an action can be useful at one
part of training and less useful at another.

The implemented initial model is deliberately intercept-only:

$$
z_t=[1].
$$

Equivalently, the controller uses the action feature map \(\phi(A_t)\) itself:

$$
x(A_t,z_t)=\phi(A_t).
$$

Here **intercept-only** refers only to training context. The action map still
contains the Helmert granularity contrasts described below. Nonstationarity is
handled initially through the process covariance \(Q\), rather than through
progress-dependent coefficients.

Possible future context variables, all known before the decision, include:

- normalized token progress;
- the current learning-rate ratio;
- the preceding controller objective \(J_{H_c}(\theta_t)\);
- a trailing controller-objective slope;
- a warmup indicator;
- the previous action, when studying learned switching effects.

Do not include the current training-batch loss, current batch identity, or
anything observed after choosing \(A_t\). Those either bring training data back
into the controller or leak the outcome into its predictors.

The context must interact with the action encoding. Let \(\phi(A)\) encode a
global arm or per-block profile. If a future model uses context beyond the
intercept, a suitable construction is

$$
x(A_t,z_t)=\phi(A_t)\otimes z_t,
$$

where \(\otimes\) is the Kronecker product. This creates, for each action
feature, a base coefficient and context-dependent coefficients. For example,
with \(z_t=[1,p_t]^\top\),

$$
\mathbb{E}[r_t\mid A_t,z_t]
= \phi(A_t)^\top\beta^{(0)}
+p_t\,\phi(A_t)^\top\beta^{(p)}.
$$

Merely concatenating \(\phi(A_t)\) and \(z_t\) would give context-only terms
that are identical for every candidate action. Those terms could predict the
overall reward trend, but would not change which action is selected.
Progress or learning-rate interactions should be added only after validating
the current reward and posterior pipeline. If both are nearly deterministic
functions of the same training clock, use only one to avoid redundant features.

### Meaning of \(m_t^-\) and \(V_t^-\)

At the end of controller round \(t-1\), after observing its reward, suppose the
posterior is

$$
\beta_{t-1}\mid\mathcal{D}_{1:t-1}
\sim\mathcal{N}(m_{t-1},V_{t-1}).
$$

The dynamic model predicts the coefficients for the next round:

$$
\beta_t=F\beta_{t-1}+\xi_t,
\qquad
\xi_t\sim\mathcal{N}(0,Q).
$$

\(F\) is the reward-model **state-transition matrix**. If \(\beta_t\) has
\(d\) coefficients, then \(F\in\mathbb{R}^{d\times d}\). It specifies how the
expected coefficients change from one controller round to the next before a
new reward is observed. It is part of the bandit state model, not part of the
language model.

For this application, the recommended initial choice is

$$
F=I_d.
$$

This says that, absent new evidence, the best prediction for the next
coefficient vector is the current one:

$$
\mathbb{E}[\beta_t\mid\beta_{t-1}]=\beta_{t-1}.
$$

Nonstationarity is then expressed by \(Q\), which increases uncertainty about
that prediction. More complicated choices are possible but are not needed
initially. For example, \(F=\rho I_d\), with \(0<\rho<1\), shrinks coefficients
toward zero between rounds. A better explicit mean-reverting model would be

$$
\beta_t-\bar\beta
=F(\beta_{t-1}-\bar\beta)+\xi_t,
$$

which shrinks toward a chosen long-run mean \(\bar\beta\), rather than
implicitly toward zero. Such behavior introduces an additional assumption and
should only be added if experiments show that a random walk is inadequate.

Therefore, before selecting \(A_t\) and before observing \(r_t\),

$$
\beta_t\mid\mathcal{D}_{1:t-1}
\sim\mathcal{N}(m_t^-,V_t^-),
$$

with

$$
m_t^-=F m_{t-1},
\qquad
V_t^-=F V_{t-1}F^\top+Q.
$$

The superscript minus means **predicted or pre-observation**. After observing
\(r_t\), the updated posterior parameters are written without the minus:
\(m_t,V_t\).

The simplest random-walk model uses \(F=I\), giving

$$
m_t^-=m_{t-1},
\qquad
V_t^-=V_{t-1}+Q.
$$

If the coefficients are assumed stationary, use \(F=I\) and \(Q=0\). Then the
next prior is exactly the previous posterior.

### Deriving the Gaussian posterior update

Yes: the update follows directly from conditioning a joint Gaussian.

Before observing \(r_t\), assume

$$
\beta_t\sim\mathcal{N}(m_t^-,V_t^-)
$$

and the scalar linear observation model

$$
r_t=x_t^\top\beta_t+\epsilon_t,
\qquad
\epsilon_t\sim\mathcal{N}(0,\sigma^2),
$$

with \(\epsilon_t\) independent of \(\beta_t\). Linear transformations of
Gaussian variables remain Gaussian, so

$$
\begin{bmatrix}
\beta_t\\
r_t
\end{bmatrix}
\sim
\mathcal{N}\left(
\begin{bmatrix}
m_t^-\\
x_t^\top m_t^-
\end{bmatrix},
\begin{bmatrix}
V_t^- & V_t^-x_t\\
x_t^\top V_t^- & x_t^\top V_t^-x_t+\sigma^2
\end{bmatrix}
\right).
$$

For jointly Gaussian variables \(a,b\),

$$
\mathbb{E}[a\mid b]
=\mu_a+\Sigma_{ab}\Sigma_{bb}^{-1}(b-\mu_b),
$$

$$
\operatorname{Cov}(a\mid b)
=\Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}.
$$

Substituting \(a=\beta_t\) and \(b=r_t\) gives

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

The scalar

$$
r_t-x_t^\top m_t^-
$$

is the reward prediction error or innovation. The vector \(k_t\), commonly
called the Kalman gain, controls how strongly that error changes each
coefficient. The update becomes smaller when observation noise \(\sigma^2\) is
large, and larger in coefficient directions where prior uncertainty \(V_t^-\)
is large.

The same posterior can also be derived by multiplying the Gaussian prior by the
Gaussian likelihood and completing the square. The joint-Gaussian conditional
view makes the origin of the Kalman-form equations especially transparent.

### Nonstationarity during training

Reward behavior changes as the language model trains, so a stationary posterior
is not quite right. A dynamic Bayesian model can express drift directly:

$$
\beta_t = \beta_{t-1}+\xi_t,
\qquad \xi_t\sim\mathcal{N}(0,Q),
$$

which gives the predictive belief

$$
m_t^- = m_{t-1},
\qquad
V_t^- = V_{t-1}+Q.
$$

The process covariance $Q$ controls how quickly old evidence becomes
uncertain again. This is a probabilistic replacement for an ad hoc decay of a
reward mean. A diagonal $Q$ is enough initially.

## Feature structure and the role of the Helmert contrast matrix

The controller receives categorical actions such as `s`, `m`, `l`, and `xl`,
but its linear Gaussian reward model requires a numerical feature vector. The
Helmert **contrast** matrix provides a deterministic coordinate vector for each
granularity. It is part of the adaptive controller; it does not change the
transformer or the representation of tokens inside the language model.

Let the resolved, ordered granularity set be

$$
\mathcal{G}=(g_1,\ldots,g_G).
$$

The implementation constructs

$$
C\in\mathbb{R}^{G\times(G-1)}
$$

with entries

$$
C_{i,j}=
\begin{cases}
1/\sqrt{j(j+1)}, & i\leq j,\\
-j/\sqrt{j(j+1)}, & i=j+1,\\
0, & i>j+1,
\end{cases}
$$

for $i\in\{1,\ldots,G\}$ and $j\in\{1,\ldots,G-1\}$. Let $c_g^\top$
denote the row associated with granularity $g$. That row is the numerical
encoding of $g$ used by the reward model.

### Worked example with three granularities

For

$$
\mathcal{G}=(s,m,l),
$$

the matrix is

$$
C=
\begin{bmatrix}
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}}\\[2mm]
-\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}}\\[2mm]
0 & -\frac{2}{\sqrt{6}}
\end{bmatrix}.
$$

Its rows are

$$
c_s=
\begin{bmatrix}
1/\sqrt{2}\\
1/\sqrt{6}
\end{bmatrix},
\qquad
c_m=
\begin{bmatrix}
-1/\sqrt{2}\\
1/\sqrt{6}
\end{bmatrix},
\qquad
c_l=
\begin{bmatrix}
0\\
-2/\sqrt{6}
\end{bmatrix}.
$$

For global adaptation, the feature vector for choosing $g$ is

$$
x_{\mathrm{global}}(g)=
\begin{bmatrix}
1\\
c_g
\end{bmatrix}.
$$

If the global coefficient vector is

$$
\beta=
\begin{bmatrix}
\beta_0\\
\beta_c
\end{bmatrix},
\qquad \beta_c\in\mathbb{R}^{G-1},
$$

then the expected reward of each granularity is

$$
\mu_g=\mathbb{E}[r\mid g]
=x_{\mathrm{global}}(g)^\top\beta
=\beta_0+c_g^\top\beta_c.
$$

For the three-granularity example, this gives

$$
\mu_s=\beta_0+c_s^\top\beta_c,
\qquad
\mu_m=\beta_0+c_m^\top\beta_c,
\qquad
\mu_l=\beta_0+c_l^\top\beta_c.
$$

Because the contrast rows sum to zero,

$$
c_s+c_m+c_l=0,
$$

their mean expected reward is

$$
\frac{\mu_s+\mu_m+\mu_l}{3}=\beta_0.
$$

The intercept $\beta_0$ therefore represents the average reward across
granularities. The remaining coefficients describe deviations from that
average.

### Why use only $G-1$ contrasts?

There are $G$ granularity-specific expected rewards, but once their common mean
is represented by the intercept, only $G-1$ independent differences remain.
Using an intercept together with $G$ one-hot indicators would be redundant
because

$$
1=\sum_{g\in\mathcal{G}}\mathbf{1}[a=g].
$$

For example, adding a constant to the intercept and subtracting the same
constant from every one-hot coefficient leaves every prediction unchanged.
The data therefore cannot identify a unique coefficient vector. An intercept
plus $G-1$ full-rank contrasts has exactly $G$ identifiable parameters and can
still represent an arbitrary expected reward for every granularity.

Reference-category coding would also remove the redundancy, but it would make
one arbitrarily chosen granularity the zero-vector baseline. The sum-to-zero
Helmert encoding instead gives the intercept the directly useful meaning of an
average and does not privilege a reference label.

### Why must the contrasts be orthonormal?

The Helmert matrix satisfies

$$
C^\top\mathbf{1}=0
$$

and

$$
C^\top C=I_{G-1}.
$$

The first property is the sum-to-zero condition: contrast directions cannot
change the common intercept. The second says that the contrast columns are
mutually perpendicular and have unit length. Consequently, the coefficients
use comparable numerical scales and no contrast direction is duplicated by
another.

This matters for a Bayesian controller. With a spherical prior such as

$$
\beta_c\sim\mathcal{N}(0,\tau^2 I),
$$

orthonormal features give equal prior variance to independent directions of
granularity variation. Without normalization, one coefficient could appear
more or less uncertain merely because its feature column was scaled
differently. Orthonormal coding also improves the numerical interpretation of
the posterior covariance and its updates.

### Per-block credit assignment

For a per-block profile

$$
A=(a_1,\ldots,a_B),
$$

the feature vector concatenates one contrast row per transformer block:

$$
x_{\mathrm{block}}(A)=
\begin{bmatrix}
1\\
c_{a_1}\\
\vdots\\
c_{a_B}
\end{bmatrix}
\in\mathbb{R}^{1+B(G-1)}.
$$

Partition the controller coefficient vector as

$$
\beta=
\begin{bmatrix}
\beta_0\\
\beta_1\\
\vdots\\
\beta_B
\end{bmatrix},
\qquad
\beta_b\in\mathbb{R}^{G-1}.
$$

The additive per-block reward model is then

$$
\mathbb{E}[r\mid A]
=x_{\mathrm{block}}(A)^\top\beta
=\beta_0+\sum_{b=1}^{B}c_{a_b}^\top\beta_b.
$$

Here $\beta_b$ represents the relative effects of the granularities at block
$b$. The controller observes only one scalar reward for the complete profile;
it does not copy that scalar into a separate reward for every selected
block/granularity pair. Bayesian regression instead infers the additive block
effects from rewards observed across varied profiles and retains uncertainty in
directions for which evidence is weak.

For a sampled coefficient vector $\widetilde\beta$, additivity makes the best
profile separable:

$$
a_b=\arg\max_{g\in\mathcal{G}}
c_g^\top\widetilde\beta_b.
$$

The controller can therefore choose the exact maximizing profile in
$O(BG)$ work rather than enumerating all $G^B$ profiles.

Finally, the Helmert matrix does not assume that the labels form a numerical
scale or that neighboring sizes have similar effects. The granularities remain
categorical. Their saved order determines a reproducible contrast basis and
tie-breaking order, not an ordinal reward relationship.

### Learning profile structure instead of penalizing change

If independent block effects are insufficient, add adjacent-block interactions:

$$
\mathbb{E}[r\mid A]
= \beta_0
+ \sum_{b=1}^{B}u_b(a_b)
+ \sum_{b=1}^{B-1}v_b(a_b,a_{b+1}).
$$

The terms $v_b$ let rewards reveal whether neighboring choices work well
together. They can learn a preference for smooth profiles, abrupt profiles, or
neither. No assumption about complete-profile changes is required.

For a sampled set of coefficients, the best profile can still be found exactly
with dynamic programming:

$$
D_1(g)=u_1(g),
$$

$$
D_b(g)=u_b(g)+
\max_{h\in\mathcal{G}}\left[D_{b-1}(h)+v_{b-1}(h,g)\right].
$$

This costs $O(B|\mathcal{G}|^2)$, rather than enumerating
$|\mathcal{G}|^B$ profiles.

Changing from $A_{t-1}$ to $A_t$ should have zero cost by default. If there
is a real operational switching cost, place that measured cost explicitly in
the selection objective. If we merely suspect that switching hurts training,
include switch indicators as uncertain model features and learn their
coefficient:

$$
x_{b}^{\mathrm{switch}}(A_t,A_{t-1})
=\mathbf{1}[a_{t,b}\ne a_{t-1,b}].
$$

That turns the current assumption into a testable hypothesis.

## Compute or size preferences must be explicit

Do not quietly replace the profile-change penalty with a size penalty. If the
research objective includes compute, define it separately, for example

$$
U_t(A)=\mathbb{E}[r_t\mid A]
-\lambda_{\mathrm{compute}} C(A),
$$

or treat compute as a hard budget. Setting
$\lambda_{\mathrm{compute}}=0$ asks only which sampling action most improves
the controller objective. Sweeping this coefficient traces a quality/compute
frontier.

## Recommended sequence of experiments

### Stage 1: validate reward alignment with global adaptation

Use the $|\mathcal{G}|$-arm global action space, a fixed controller holdout,
and a small number of optimizer steps per decision window. Compare:

- the current uniform global random baseline;
- balanced round-robin global sampling;
- Bayesian global Thompson sampling;
- a greedy posterior-mean policy after an initial balanced exploration phase.

This isolates the reward and posterior machinery before per-block credit
assignment is introduced.

### Stage 2: additive per-block model

Use additive per-block Helmert contrast features only. Confirm that posterior
means actually move after rewards, posterior uncertainty contracts for observed
directions, and restored checkpoints reproduce the same future posterior and
sampling RNG state.

### Stage 3: learned interactions

Add adjacent-block interactions only if heldout reward prediction or final
performance shows that the additive model is inadequate. Compare the learned
interaction model against the additive one and against the old Hamming-penalty
heuristic.

### Essential evaluation

Track at least:

- controller objective and untouched validation/final objective;
- reward prediction error and posterior predictive intervals;
- action frequencies and entropy over time;
- posterior uncertainty by block and granularity;
- total controller-evaluation cost;
- global and per-block performance at matched training tokens and wall time;
- controller-to-final generalization gap, to detect controller-set overfitting.

## Current configuration surface

The implemented global Bayesian controller is configured explicitly as:

```yaml
model:
  granularity_sampling_mode: adaptive_global
  adaptive_sampler_strategy: thompson
  adaptive_controller:
    decision_interval_steps: 50
    prior_mean: 0.0
    prior_covariance: 1.0
    observation_noise_variance: 0.01
    process_noise_covariance: 0.0001

evaluation:
  adaptive_controller:
    enabled: true
    source: configured_dataset_split
    examples: 128
    objective_weights: uniform
    fixed_manifest: true
  final_holdout:
    enabled: true
    source: configured_dataset_split
    examples: 512
    fixed_manifest: true
    evaluate_during_training: false
```

For additive per-block control, only
`granularity_sampling_mode: adaptive_per_block` changes. Scalar prior means
expand to the full coefficient vector, while scalar prior and process
covariances expand to isotropic dense matrices. Thus the preset above resolves
to

$$
m_0=\mathbf{0},\qquad
V_0=I_d,\qquad
\sigma^2=10^{-2},\qquad
Q=10^{-4}I_d.
$$

The method also fixes the context model to intercept-only, the state transition
to identity, and both compute and switching weights to zero. These resolved
choices, along with scope, feature schema, reward data, and decision interval,
are saved in artifacts.

## Resolved experimental decisions

The initial experiment will use the following contract:

1. **Controller objective:** uniform average loss across every resolved
   granularity:

   $$
   J_{H_c}(\theta)
   =\frac{1}{|\mathcal{G}|}
   \sum_{g\in\mathcal{G}}
   L_{H_c}\!\left(\theta;(g,\ldots,g)\right).
   $$

2. **Decision interval:** one selected global granularity or per-block profile
   remains active for a configurable positive number of optimizer steps before
   its reward is observed. The default interval is 50 steps.
3. **Controller data:** every boundary evaluation uses the same deterministic
   panel of 128 examples, with a saved seed, manifest, and hash.
4. **Initial per-block feature model:** additive block contributions only, with
   no adjacent-block interactions.
5. **Costs:** both compute and switching costs are exactly zero:

   $$
   \lambda_{\mathrm{compute}}
   =\lambda_{\mathrm{switch}}
   =0.
   $$

6. **Final evaluation:** reserve a separate fixed set of 512 examples before
   training. Give it its own seed, manifest, and hash. It must never contribute
   controller rewards, checkpoint selection, or hyperparameter choices and is
   evaluated only for final comparisons.

These are initial experimental decisions, not permanent constraints. In
particular, interaction features and explicit compute costs should be added
only as controlled follow-up ablations.

## Current recommendation

Retain **Bayesian global adaptation**, the boundary-to-boundary fixed-holdout
reward, and the 500-step balanced pre-adaptive warmup as the comparison
baseline. Pause further fixed-$Q$ sweeps: the calibration record shows that
process noise, observation noise, and training progress are coupled, while the
tested posteriors still do not learn a useful action ranking. The next
method-level experiment should test episodic posterior reset with $Q=0$ before
expanding additive per-block control. Treat artifacts from the superseded
pre-Bayesian adaptive implementation as a legacy heuristic baseline, not as
Bayesian Thompson sampling.

## Questions and clarifications

### What is $Q$?

The controller coefficients $\beta_t$ describe the training utility of the
available actions at controller round $t$. These are controller variables, not
parameters of the language model. Since the utility of a granularity may change
as language-model training progresses, the controller uses a random-walk state
model:

$$
\beta_t=\beta_{t-1}+\xi_t,
\qquad
\xi_t\sim\mathcal{N}(0,Q).
$$

The random vector $\xi_t$ represents the unknown change in the reward
coefficients between two controller rounds. Its covariance

$$
Q\succeq 0
$$

is called the **process-noise covariance**. It encodes how much coefficient
drift the controller considers plausible between rounds.

Suppose that, after observing the reward from round $t-1$, the posterior is

$$
\beta_{t-1}\mid\mathcal{D}_{1:t-1}
\sim\mathcal{N}(m_{t-1},V_{t-1}).
$$

Before choosing the action for round $t$, the identity transition gives

$$
m_t^-=m_{t-1},
\qquad
V_t^-=V_{t-1}+Q.
$$

Therefore, $Q$ does not move the predictive mean. It increases predictive
uncertainty because older observations may be less informative at a later
stage of language-model training.

If

$$
Q=0,
$$

the controller assumes that action utilities are stationary. Evidence never
becomes less reliable merely because another controller round has passed. If

$$
Q=qI,
$$

each coefficient is allowed to drift independently with variance $q$ per
controller round. A small $q$ preserves old information for longer, while a
large $q$ restores uncertainty quickly and causes continued exploration. If
$q$ is too large, the controller repeatedly forgets useful evidence.

A dense $Q$ could describe correlated coefficient drift. The initial model can
instead use a scalar multiple of the identity or a diagonal matrix, which is
easier to interpret. The units of entries in $Q$ are squared reward units. In
this experiment the reward is

$$
r_t=
\frac{J_{H_c}(\theta_t)-J_{H_c}(\theta_{t+1})}{h},
$$

so $Q$ must be calibrated to the scale of the per-step controller-objective
improvements, which may be small.

### How is the Gaussian prior over $\beta$ initialized?

Before any completed controller window is observed, the controller represents
its uncertainty about the coefficient vector with

$$
\beta_0\sim\mathcal{N}(m_0,V_0).
$$

The implementation requires the prior mean $m_0$ and covariance $V_0$ to be
specified explicitly. It does not silently infer them from the legacy adaptive
sampler settings.

A simple neutral prior is

$$
m_0=\mathbf{0},
\qquad
V_0=\tau^2 I.
$$

The zero mean expresses no initial evidence that one granularity is better than
another. The covariance $\tau^2I$ gives the independent coefficient directions
the same initial variance. With this prior, the expected reward of every action
is initially zero:

$$
\mathbb{E}[r\mid A]
=x(A)^\top m_0
=0.
$$

This does not claim that the true reward is zero. It only says that the
controller has not yet observed evidence about its sign or about relative
action utility.

For the additive per-block model, partition the prior mean in the same way as
the coefficient vector:

$$
m_0=
\begin{bmatrix}
m_{0,\mathrm{intercept}}\\
m_{0,1}\\
\vdots\\
m_{0,B}
\end{bmatrix},
\qquad
m_{0,b}\in\mathbb{R}^{G-1}.
$$

The neutral choice sets the intercept and all block-contrast means to zero. If
reliable pilot experiments estimate an average per-step improvement $\bar r$,
a more informative mean could be

$$
m_0=
\begin{bmatrix}
\bar r\\
0\\
\vdots\\
0
\end{bmatrix}.
$$

This initializes the intercept to the expected general training improvement
while retaining no initial preference among granularities.

The prior covariance controls initial exploration. With

$$
V_0=\tau^2 I,
$$

a small $\tau^2$ expresses strong confidence that granularity effects are near
zero. A large $\tau^2$ allows large differences among actions and produces
stronger initial exploration. The intercept and action contrasts can also use
different prior variances:

$$
V_0=
\operatorname{diag}
\left(
\tau_{\mathrm{intercept}}^2,
\tau_{\mathrm{contrast}}^2,
\ldots,
\tau_{\mathrm{contrast}}^2
\right).
$$

A practical calibration procedure is:

1. Run a short random or balanced round-robin pilot.
2. Measure the scale and variability of the window rewards $r_t$.
3. Set the prior standard deviations to the scale of plausible coefficient
   effects.
4. Set the observation-noise variance $\sigma^2$ from reward variation not
   explained by the action features.
5. Choose $Q$ smaller than $V_0$ unless action utility is expected to change
   very quickly during training.

Although the controller panel is deterministic, $\sigma^2$ can still represent
optimizer stochasticity, imperfect additive credit assignment, and other
reward variation not captured by the linear feature model.

### Are the $\beta$ coefficients sampled from the same Gaussian?

All coefficients are drawn together as one vector from a single joint
multivariate Gaussian predictive belief:

$$
\widetilde\beta_t
\sim\mathcal{N}(m_t^-,V_t^-).
$$

For the per-block model, one complete draw has the partition

$$
\widetilde\beta_t=
\begin{bmatrix}
\widetilde\beta_{0,t}\\
\widetilde\beta_{1,t}\\
\vdots\\
\widetilde\beta_{B,t}
\end{bmatrix},
\qquad
\widetilde\beta_{b,t}\in\mathbb{R}^{G-1}.
$$

The same complete draw is used to select every block action:

$$
a_{t,b}
=\arg\max_{g\in\mathcal{G}}
c_g^\top\widetilde\beta_{b,t}.
$$

Thus the coefficients come from the same **joint belief**, but they are not
necessarily independent or identically distributed. Different entries may
have different means and variances. Off-diagonal entries of $V_t^-$ represent
correlations between coefficients.

Even if $V_0$ and $Q$ are initially diagonal, observing one scalar reward for a
complete profile generally creates posterior correlations:

$$
V_t
=V_t^-
-\frac{V_t^-x_tx_t^\top V_t^-}
{\sigma^2+x_t^\top V_t^-x_t}.
$$

This happens because one profile reward contains information about the
intercept and several selected block effects simultaneously.

### Is the sampled coefficient vector used in the posterior update?

No. The Thompson sample $\widetilde\beta_t$ is temporary and is used only to
select an action. The controller first forms the predictive belief and samples
from it:

$$
\underbrace{\mathcal{N}(m_{t-1},V_{t-1})}_{\text{previous posterior}}
\xrightarrow{\,+Q\,}
\underbrace{\mathcal{N}(m_t^-,V_t^-)}_{\text{predictive belief}}
\xrightarrow{\text{sample}}
\widetilde\beta_t
\xrightarrow{\text{maximize}}
A_t.
$$

After the model trains under $A_t$ for the complete decision window, the
controller observes $r_t$ and conditions the predictive Gaussian itself:

$$
\mathcal{N}(m_t^-,V_t^-)
\xrightarrow{\text{observe }(x_t,r_t)}
\mathcal{N}(m_t,V_t).
$$

The sampled $\widetilde\beta_t$ is not treated as the true coefficient vector
and is not inserted into the conditioning equations. Its sole purpose is to
turn posterior uncertainty into randomized action selection, thereby balancing
exploration and exploitation.

## Experimental evidence and current direction

The complete 100M-token diagnosis and four 20M-token calibration batches have
been moved to the
[fixed-$Q$ calibration record](probabilistic_adaptive_granularity_q_calibration_2026-08-07.md).
That note preserves the exact commands, results, covariance diagnostics, and
superseded recommendations.

The experiments established five durable conclusions:

1. The original broad prior and covariance made Thompson selection effectively
   random because uncertainty dominated the observed reward differences.
2. Before balanced warmup, a larger intercept process variance absorbed the
   early learning transient but allowed the first actions to poison later
   credit assignment.
3. A 500-step balanced global warmup prevents that early action starvation and
   reduces across-seed failure risk without consulting or updating the
   controller.
4. Once the controller starts at step 500, the formerly selected
   $q_I=10^{-5}$ is severely over-dispersed because the controller no longer
   observes the initial transient.
5. Sweeping $q_I$ down to zero cannot calibrate the post-warmup controller while
   $\sigma^2=10^{-7}$ and $q_C=10^{-8}$ remain fixed. At $q_I=0$, the
   intercept also develops persistent negative prediction bias.

Balanced pre-adaptive warmup remains the selected controller-start contract.
Manual fixed-$Q$ tuning is paused: the latest experiment shows that $Q$,
observation noise, and unmodelled training progress are coupled, while none of
the tested configurations produces meaningful posterior action separation.

The primary new direction is the
[episodic-reset proposal](probabilistic_adaptive_granularity_reset_method_proposal_2026-08-07.md):
set $Q=0$, accumulate evidence within an explicit controller episode, reset at
auditable boundaries, and reacquire balanced evidence after each reset. The
empirical-Bayes alternative below is retained as a parked design, not the
current implementation plan.

## Alternative: empirical-Bayes initialization from balanced warmup

Instead of manually recalibrating $V_0$ and $Q$ for every training regime, the
balanced warmup could collect the measurements needed to initialize them. This
would be a new controller contract, not the behavior evaluated in batch 3.

Evaluate the complete fixed controller panel at steps

$$
0,50,\ldots,500,
$$

while retaining the deterministic balanced training schedule and disabling
Thompson selection. The eleven panel evaluations define ten action-labelled
rewards. Fit a regularized model

$$
r_t=f(p_t)+c(a_t)^\top\beta_c+\epsilon_t,
$$

where $p_t$ is training progress, $f$ contains the common intercept and
learning trend, and $c(a_t)$ is the contrast-only row of the existing Helmert
encoding. Separating $f(p_t)$ from the contrasts prevents the initial learning
curve from being attributed to whichever action happens to occur first without
duplicating the controller intercept.

At step 500, initialize the controller as follows:

- use the fitted common reward level at step 500 and the action-contrast
  coefficients for the posterior mean, with strong contrast shrinkage toward
  zero because there are only two warmup observations per action;
- derive $V_0$ from the regularized coefficient-estimation covariance, subject
  to eigenvalue floors and ceilings;
- initialize the observation variance from a robust residual-variance
  estimate;
- parameterize process noise with one learnable scale rather than attempting
  to estimate five independent values from ten rewards:

  $$
  Q_0=s_0\operatorname{diag}(\rho,1,1,1,1).
  $$

Here $s_0$ is initialized from residual drift and $\rho$ controls the fixed
ratio between common intercept drift and contrast drift. Both require bounded
defaults when the warmup estimate is weak.

After warmup, update the single scale slowly from standardized innovations:

$$
\log s_{t+1}
=
\operatorname{clip}
\left(
\log s_t+\eta(z_t^2-1),
\log s_{\min},
\log s_{\max}
\right).
$$

Large innovations increase process uncertainty; persistently small innovations
decrease it. A small adaptation rate $\eta$, a delayed start, and multiplicative
change limits are necessary to prevent one outlier from destabilizing the
controller.

This design would replace repeated absolute-scale sweeps with per-run
initialization plus bounded online correction. Its costs are ten additional
fixed-panel evaluations, a more complex checkpoint schema, and the loss of the
current `prior_untouched` warmup invariant. The fitted progress model,
regularization rule, covariance bounds, adaptive scale, and all warmup panel
measurements would need to be persisted for exact resume and auditability.
