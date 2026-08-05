# Probabilistic Adaptive Granularity Sampling

Date: 2026-08-05

This note records a design discussion about replacing the current
`adaptive_per_block` heuristic with a probabilistic controller that learns from
reward. It also treats global adaptation as a first-class case.

## Short conclusion

The most promising next design is a **Bayesian structured bandit** whose reward
is measured on a fixed controller holdout, not on consecutive training batches.
The same controller can support two action spaces:

- **global adaptive**: choose one granularity for the whole model;
- **per-block adaptive**: choose a vector containing one granularity per block.

For the per-block case, an additive Bayesian linear model is a good first
version. Adjacent-block interaction terms can be added later if the additive
model is too restrictive. This lets the data learn whether coherent or abrupt
profiles are useful. A profile-change penalty should not be present by default.

## What the current approach actually does

Let the profile selected at training step $t$ be

$$
P_t = (P_{t,1}, \ldots, P_{t,B}),
$$

where $B$ is the number of transformer blocks. The current code computes a
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
6. The current "Thompson" implementation is a deterministic Gaussian
   perturbation of running scores, not sampling from an explicit Bayesian
   posterior.
7. With the current default `adaptive_sampler_decay_rate: 0.0`, the running
   reward means never change. In that configuration, reward cannot affect later
   choices at all.

The field currently called `correction_penalty` is specifically a profile-change
penalty; it is not a measurement of the model's membership or LMC correction.

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

For a first implementation, use only deterministic, low-dimensional context:

$$
z_t =
\begin{bmatrix}
1 \\
p_t \\
\ell_t
\end{bmatrix},
\qquad
p_t=\frac{\text{tokens seen at }t}{\text{token budget}},
\qquad
\ell_t=\frac{\text{current learning rate}}{\text{initial learning rate}}.
$$

The leading \(1\) is an intercept. Both progress and learning rate are known
before sampling and do not depend on the current training batch. If the
learning-rate schedule is nearly a deterministic function of progress, use
only one of them to avoid redundant features.

Possible later additions, all computed only from information available before
the decision, include:

- the preceding controller objective \(J_{H_c}(\theta_t)\);
- a trailing controller-objective slope;
- a warmup indicator;
- the previous action, when studying learned switching effects.

Do not include the current training-batch loss, current batch identity, or
anything observed after choosing \(A_t\). Those either bring training data back
into the controller or leak the outcome into its predictors.

The context must interact with the action encoding. Let \(\phi(A)\) encode a
global arm or per-block profile. A simple construction is

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

There is also a reasonable simpler first experiment: set \(z_t=[1]\) and rely
on the dynamic process noise \(Q\) to handle nonstationarity. Add progress
interactions only after the reward and posterior pipeline is validated.

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

## Feature structure for per-block credit assignment

The smallest useful feature map has one action indicator per block:

$$
x_{b,g}(A)=\mathbf{1}[a_b=g].
$$

Then

$$
\mathbb{E}[r\mid A]
= \beta_0 + \sum_{b=1}^{B}\beta_{b,a_b}.
$$

Unlike the current update, this does not claim that every selected block action
received the complete scalar reward. Bayesian regression infers block effects
from many varied complete profiles and retains uncertainty when evidence is
weak.

The one-hot design needs an identifiable parameterization: use one reference
granularity per block or sum-to-zero contrast coding. An initial balanced
exploration phase should cover every `(block, granularity)` combination.

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

Use block/granularity indicators only. Confirm that posterior means actually
move after rewards, posterior uncertainty contracts for observed directions,
and restored checkpoints reproduce the same future posterior and sampling RNG
state.

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

## Suggested conceptual configuration

The exact names can follow the repository's compatibility rules, but the design
should expose these concepts explicitly:

```yaml
model:
  granularity_sampling_mode: adaptive
  adaptive_scope: global          # global | per_block
  adaptive_strategy: bayesian_ts
  adaptive_feature_model: additive # arms | additive | adjacent_pairwise
  adaptive_decision_interval_steps: 50
  adaptive_observation_noise: 0.01
  adaptive_process_noise: 0.0001
  adaptive_compute_weight: 0.0
  adaptive_switch_feature: false

evaluation:
  adaptive_controller:
    enabled: true
    examples: 128
    objective_weights: uniform
    fixed_manifest: true
  final_holdout:
    examples: 512
    fixed_manifest: true
    evaluate_during_training: false
```

For backward compatibility, `adaptive_global` and `adaptive_per_block` may be
clearer surface names than changing the existing mode hierarchy immediately.
The important point is that scope, reward set, posterior model, and decision
interval are independently visible in artifacts.

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
   remains active for 50 optimizer steps before its reward is observed.
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
additive per-block action model. Treat the existing adaptive implementation as
a heuristic baseline, not as Thompson sampling, and remove the profile-change
penalty from the default objective.
