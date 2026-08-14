# Gradient-Norm Granularity Sampling

**Status**: exploratory mathematical note  
**Scope**: standalone adaptive granularity sampling for MatFormer training  
**Motivation**: adapt the useful gradient-importance idea from
[PA&DA](https://arxiv.org/abs/2302.14772) to nested MatFormer subnetworks without
assuming that the complete-subnetwork gradient norm is comparable across widths.

## Question

Can training allocate granularity samples using a gradient-based importance
score, while avoiding an automatic preference for wider subnetworks merely
because they contain more active parameters?

This note treats gradient-based sampling as its own training strategy. It does
not assume a held-out reward model or another adaptive controller.

## Notation

Let the ordered granularities be

\[
\mathcal{G} = \{g_1, \ldots, g_K\},
\]

from narrowest to widest. Let \(A_g\) be the set of parameters active under a
global granularity \(g\), so the slicing construction gives

\[
A_{g_1} \subset A_{g_2} \subset \cdots \subset A_{g_K}.
\]

For one training example or minibatch \(z\), define

\[
L_g(\theta; z)
\]

as the loss under granularity \(g\), and

\[
d_g = \nabla_{\theta_{A_g}} L_g(\theta; z)
\]

as its gradient on the active parameters. When needed, \(d_g\) can be embedded
in the complete supernetwork parameter space by filling inactive coordinates
with zero.

For each transformer block \(b\), let \(C_b\) denote a fixed shared-core
parameter set. The natural first choice is the FFN slice belonging to the
narrowest granularity, separated into gate, up, and down projection groups.
Every global granularity activates exactly the same \(C_b\).

## Why the complete \(L_2\) norm is size-biased

The direct PA&DA-style score would be

\[
I_g^{\mathrm{total}} = \lVert d_g \rVert_2.
\]

Suppose, only to expose the dimensional effect, that the active gradient
coordinates have the same second moment:

\[
\mathbb{E}[d_{g,i}^2] = \sigma^2.
\]

If \(N_g = |A_g|\), then

\[
\mathbb{E}\left[\lVert d_g \rVert_2^2\right]
= N_g \sigma^2.
\]

Consequently, the typical total norm scales approximately as

\[
\lVert d_g \rVert_2 \propto \sqrt{N_g}.
\]

A wider subnetwork can therefore receive a larger importance score even when
its typical gradient per active parameter is identical. Normalizing the scores
across granularities,

\[
p_g = \frac{I_g}{\sum_h I_h},
\]

does not remove this effect; it only turns the already size-biased scores into
probabilities.

## Candidate size-aware scores

### 1. Active-parameter gradient RMS

The simplest dimensional correction is

\[
I_g^{\mathrm{rms}}
= \frac{\lVert d_g \rVert_2}{\sqrt{N_g}}
= \sqrt{\frac{1}{N_g}\sum_{i \in A_g} d_{g,i}^2}.
\]

Under the equal-coordinate-second-moment example above,
\(\mathbb{E}[(I_g^{\mathrm{rms}})^2] = \sigma^2\), independent of subnetwork
size.

Advantages:

- cheap to compute;
- removes the leading \(\sqrt{N_g}\) dimensional effect;
- retains information from the entire active subnetwork.

Limitations:

- different granularities are still measured on different parameter sets;
- newly activated outer parameters can dilute or inflate the score;
- large parameter groups can dominate unless aggregation is performed by
  layer or parameter group.

A safer implementation would calculate an RMS for each layer/projection group
and average those group scores rather than concatenate every active parameter.

### 2. Active relative-gradient norm

A dimensionless alternative is

\[
I_g^{\mathrm{relative}}
= \frac{\lVert d_g \rVert_2}
       {\lVert \theta_{A_g} \rVert_2 + \varepsilon}.
\]

This approximates the size of the gradient relative to the current parameter
scale. It is less sensitive to layer scale, but the numerator and denominator
still cover different supports for different granularities.

Per-coordinate division by \(|\theta_i|\) should be avoided because parameters
near zero can create unstable scores. Ratios should instead be formed at the
matrix or parameter-group level.

### 3. Shared-core gradient RMS

To compare all granularities in exactly the same vector space, use only the
fixed parameter intersection:

\[
C = \bigcap_{g \in \mathcal{G}} A_g.
\]

The corresponding score is

\[
I_g^{\mathrm{core\text{-}rms}}
= \frac{\lVert \nabla_{\theta_C} L_g \rVert_2}{\sqrt{|C|}}.
\]

For MatFormer, a more interpretable implementation is to form the score from
the narrowest FFN slice in every block rather than from every globally shared
parameter. This focuses the metric on shared MatFormer capacity and prevents
embeddings or other large common tensors from dominating it.

Advantages:

- identical dimensional support for every granularity;
- no direct width advantage;
- measures how strongly each path wants to change the contested shared
  capacity.

Limitation:

- does not measure the training need of outer FFN rings that exist only in
  wider granularities.

### 4. Shared-core relative-gradient score

The current leading candidate combines common support with relative scaling.
Let \(M_b\) be the gate, up, and down matrices restricted to the smallest FFN
slice in block \(b\). Define

\[
R_{b,m,g}
= \frac{\lVert \nabla_{M_{b,m}} L_g \rVert_F}
       {\lVert M_{b,m} \rVert_F + \varepsilon},
\]

where \(m \in \{\mathrm{gate},\mathrm{up},\mathrm{down}\}\). Aggregate with
equal weight across blocks and projection groups:

\[
I_g^{\mathrm{core\text{-}relative}}
= \frac{1}{3B}
  \sum_{b=1}^{B}
  \sum_m R_{b,m,g}.
\]

Equal group averaging is intentional: concatenating the tensors would allow
the largest matrix or layer to dominate. This score is dimensionless and uses
the same tensors for every granularity.

The multiplication by learning rate is unnecessary for comparing actions at a
single step because it is common to every action. With AdamW, however, raw
gradient ratios do not equal the eventual preconditioned update ratios. An
optimizer-aware version is an optional later experiment, not the initial
metric.

## Sampling policy

Only the selected granularity produces an observed score at a step. Maintain a
per-granularity exponential moving average:

\[
S_{g,t}
= \begin{cases}
  \beta S_{g,t-1} + (1-\beta) I_{g,t}, & g = G_t, \\
  S_{g,t-1}, & g \ne G_t.
  \end{cases}
\]

A balanced warmup should observe every granularity before these estimates are
used for sampling.

To avoid treating score magnitude as an uncontrolled probability scale, map
the scores through a temperature-controlled softmax:

\[
q_{g,t}
= \frac{\exp(\log(S_{g,t}+\eta)/T)}
       {\sum_h \exp(\log(S_{h,t}+\eta)/T)}.
\]

Equivalently, \(q_g \propto (S_g+\eta)^{1/T}\). Here:

- \(T>1\) flattens the distribution;
- \(T<1\) sharpens it;
- \(\eta>0\) makes zero or initially small estimates safe.

Mix the learned distribution with a permanent uniform floor:

\[
p_{g,t}
= (1-\epsilon)q_{g,t} + \frac{\epsilon}{K},
\qquad 0 < \epsilon \le 1.
\]

This floor is necessary because unselected arms have stale score estimates and
outer FFN rings otherwise risk receiving no training.

## Optional exposure and compute terms

The gradient score and the coverage of outer parameters are distinct concerns.
Let ring \(r\) be the incremental parameter set added between two consecutive
granularities, and define its exposure through step \(t\) as

\[
E_{r,t} = \sum_{s=1}^{t} \mathbb{1}[r \subseteq A_{G_s}].
\]

Initially, coverage should be protected by balanced warmup and the uniform
sampling floor rather than folded into the importance metric. If diagnostics
show that outer rings remain materially undertrained, an explicit coverage
factor can be tested as a separate ablation.

If optimization is constrained by wall-clock compute rather than optimizer
steps or tokens, a cost-aware score is possible:

\[
\widetilde{S}_g
= \frac{S_g}{\operatorname{cost}(g)^\gamma},
\qquad \gamma \ge 0.
\]

- \(\gamma=0\): prioritize gradient signal per optimizer step;
- \(\gamma=1\): approximately prioritize gradient signal per unit compute;
- intermediate values trade off the two.

Cost correction changes the scientific objective and must be explicit in
method provenance.

## Does the sampled loss need inverse-probability weighting?

This is a separate decision from choosing the importance score.

Suppose the target training objective is the uniform global-granularity loss

\[
J(\theta)
= \frac{1}{K}\sum_{g=1}^{K}
  \mathbb{E}_z[L_g(\theta;z)].
\]

If \(G \sim p\) and the selected gradient is used without reweighting, then

\[
\mathbb{E}[d_G]
= \sum_g p_g\,\mathbb{E}[d_g],
\]

which generally differs from \(\nabla J\). This is an intentional adaptive
curriculum: high-score granularities affect the learned parameters more often.

If preserving the uniform objective is required, use

\[
\widehat{d}
= \frac{1}{Kp_G}d_G,
\]

for which

\[
\mathbb{E}[\widehat{d}]
= \frac{1}{K}\sum_g \mathbb{E}[d_g].
\]

The variance-minimizing sampling distribution for this estimator is linked to
the complete transformed gradient norm, not necessarily to the proposed
shared-core score. Therefore:

- shared-core score plus no inverse weighting is best described as a
  gradient-priority curriculum;
- shared-core score plus inverse weighting preserves the uniform objective but
  is a heuristic proposal distribution rather than the exact variance-optimal
  one;
- complete-gradient norm plus inverse weighting is closest to classical
  importance sampling, but retains the width concern motivating this note.

The first pilot should make this distinction explicit rather than calling both
variants the same method.

If inverse weighting is used, the observed gradient norm is also multiplied by
\(w_G=1/(Kp_G)\). Because norms are homogeneous,

\[
\lVert w_G d_G \rVert_2 = w_G\lVert d_G \rVert_2,
\]

so the unweighted score can be recovered by dividing the measured norm by
\(w_G\), provided no clipping has yet occurred.

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
the corrected FFN gradients. For concat with local post-step correction, the
relationship between gradient norm and actual update is more complicated; the
first pilot should therefore use the global slicing path.

In distributed training, squared norms must be reduced across all parameter
shards before taking the square root. The measured quantity and the sampler
state must be identical on every rank, with one rank authoritative for the
sampling draw.

## Why global sampling should come first

For a per-block action

\[
a = (g_1,\ldots,g_B),
\]

one could maintain a score \(S_{b,g}\) and sample each block independently.
However, the observed gradient at block \(b\) depends on the complete profile,
not only on \(g_b\). Updating \(S_{b,g_b}\) therefore treats a confounded
profile-dependent observation as local evidence.

There is an additional problem if the target is a uniform distribution over
profiles. Under independent block sampling,

\[
p(a) = \prod_{b=1}^{B}p_{b,g_b},
\]

and exact inverse-probability weights involve \(1/p(a)\). This product can be
extremely large even when every local probability is reasonable.

The first mathematical and empirical validation should therefore use one
global granularity per optimizer step. Per-block sampling requires its own
credit-assignment and weighting design.

## Proposed first pilot

Use global, per-optimizer-step granularity sampling with a balanced warmup.
During the pilot, compute and log all three main candidate scores without
initially assuming which is best:

1. complete pre-clipping gradient norm;
2. layer/group-averaged active-parameter gradient RMS;
3. shared-core relative-gradient score.

Recommended primary sampling signal:

\[
I_g = I_g^{\mathrm{core\text{-}relative}}.
\]

Recommended probability mapping:

\[
p_g
= (1-\epsilon)
  \operatorname{softmax}_g\left(\frac{\log(S_g+\eta)}{T}\right)
  + \frac{\epsilon}{K}.
\]

Initially keep \(\gamma=0\) for no compute penalty. Compare two clearly named
training variants:

- `gradient_priority_global`: ordinary selected loss, intentionally adaptive
  training exposure;
- `gradient_proposal_uniform_objective_global`: inverse-probability-weighted
  selected loss, preserving the uniform objective.

The second name is intentionally not `optimal_importance_sampling`, because a
shared-core score is not proven to minimize the variance of the complete
gradient estimator.

## Required diagnostics

For every optimizer step, record:

- selected granularity and its sampling probability;
- complete pre-correction and post-correction gradient norms, where available;
- active-parameter count and active gradient RMS;
- shared-core score by block and projection group plus its aggregate;
- EMA score for every granularity;
- sampling entropy and minimum/maximum probability;
- global clipping threshold and returned pre-clipping norm;
- clipping coefficient or whether clipping occurred;
- selected loss and valid-target count;
- optional estimated FLOPs or measured step time;
- cumulative action and incremental-ring exposure counts.

These diagnostics should determine whether the full norm is mostly a proxy for
width, whether the proposed normalized scores remove that relationship, and
whether the sampler collapses onto a small subset of actions.

## Hypotheses to test

1. **Width bias**: the complete gradient norm is strongly monotonic with active
   parameter count or granularity width.
2. **RMS correction**: active gradient RMS has materially less correlation with
   width than the complete norm.
3. **Common-support comparison**: the shared-core relative-gradient score is
   not structurally monotonic with granularity width.
4. **Useful adaptation**: score-based sampling improves uniform validation or
   final-holdout results at matched tokens or compute relative to uniform
   global sampling.
5. **Coverage safety**: balanced warmup plus the uniform floor prevents outer
   rings from becoming undertrained.
6. **Objective distinction**: the adaptive-curriculum and inverse-weighted
   variants behave differently enough that they must not be pooled under one
   method label.

## Open mathematical decisions

1. Is the desired target a uniform all-granularity objective, or is changing
   training exposure according to estimated need acceptable?
2. Should the common support contain only the smallest FFN slice, or also
   attention, embeddings, normalization, and output parameters?
3. Should shared-core aggregation use an arithmetic mean, geometric mean,
   median, or a scale-weighted mean across blocks and projections?
4. Is the post-membership-correction gradient the intended object, or should
   correction and sampling be derived jointly?
5. Does an optimizer-aware proposed-update norm predict useful training better
   than the raw-gradient relative norm under AdamW?
6. What uniform floor and temperature prevent stale-score feedback and policy
   collapse without suppressing adaptation?
7. Should action probabilities account for FLOPs or wall-clock cost?
8. What measure of subsequent learning progress should be used to validate
   that the chosen gradient score represents useful training need rather than
   noisy or conflicting gradients?

Until these questions are resolved, the shared-core relative-gradient score is
a motivated adaptive priority heuristic, not a claim of a variance-optimal
importance distribution.
