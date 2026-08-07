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

Start with **Bayesian global adaptation** and the boundary-to-boundary fixed
holdout reward. It is the cleanest test of whether adaptive sampling improves
training. Once that works, reuse the controller and posterior update for an
additive per-block action model. Treat artifacts from the superseded
pre-Bayesian adaptive implementation as a legacy heuristic baseline, not as
Bayesian Thompson sampling. The current probabilistic controller has no
profile-change penalty in its default objective.

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

###### Alternative: empirical-Bayes initialization from balanced warmup

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
