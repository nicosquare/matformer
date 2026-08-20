# H-window gradient-interference diagnostic

**Status**: active diagnostic proposal; not implemented  
**Date**: 2026-08-20  
**Scope**: uniform global sampling with granularity-persistence windows
$H\in\{1,5,25,50\}$ for the four- and eight-granularity slicing experiments

## Motivation

The current experiments show a repeatable, non-monotonic dependence on the
uniform sampling window:

- $H=5$ performs worse than per-update uniform sampling ($H=1$);
- $H=25$ improves all evaluated granularities;
- $H=50$ improves them further in both the four- and eight-granularity grids.

One possible mechanism is gradient interference between nested granularities.
Holding one sampled granularity for several optimizer updates may reduce
conflicting changes in gradient direction. The diagnostic proposed here tests
that idea while keeping measurement separate from the training policy.

This note is exploratory. A positive result would support the interference
hypothesis, but terminal gradient alignment alone would not establish that
reduced interference caused the training improvement.

## Notation

Let the ordered granularities be

$$
\mathcal{G}=\{G_1,\ldots,G_K\},
$$

from narrowest to widest. Let

$$
L_i(\theta;\mathcal D)
$$

be the target-token-weighted loss of granularity $G_i$ on a fixed diagnostic
panel $\mathcal D$, and let

$$
g_i=\nabla_\theta L_i(\theta;\mathcal D)
$$

be its raw aggregate gradient at one common model checkpoint.

Let $S_i$ denote the granularity-controlled FFN parameter coordinates active
under $G_i$. In the slicing model these supports are nested:

$$
S_1\subset S_2\subset\cdots\subset S_K.
$$

For a pair $(i,j)$, define the common controlled support as

$$
S_{ij}=S_i\cap S_j.
$$

For nested slicing supports, $S_{ij}$ is exactly the smaller granularity's
controlled FFN prefix.

## Primary pairwise measurements

### Shared-support cosine similarity

The primary measurement is

$$
c_{ij}=
\frac{
  g_i[S_{ij}]^\top g_j[S_{ij}]
}{
  \left\lVert g_i[S_{ij}]\right\rVert_2
  \left\lVert g_j[S_{ij}]\right\rVert_2
}.
$$

Both gradients must be restricted to exactly the same coordinates in the
numerator and denominator. Normalizing with each granularity's complete
gradient would confound direction with the larger model's exclusive
parameters.

Interpretation:

- $c_{ij}>0$: the two losses locally prefer compatible changes on their
  shared FFN parameters;
- $c_{ij}\approx0$: their shared-support gradient directions are nearly
  orthogonal;
- $c_{ij}<0$: a gradient step for one granularity is locally harmful to the
  other on the shared coordinates.

### Shared-support dot product

Cosine discards gradient magnitude. Also record

$$
d_{ij}=g_i[S_{ij}]^\top g_j[S_{ij}].
$$

Under a sufficiently small raw-gradient descent step for $G_i$,

$$
\theta'=\theta-\eta g_i,
$$

the first-order change in the other loss is approximately

$$
L_j(\theta')-L_j(\theta)
\approx-\eta d_{ij}.
$$

Thus, a negative $d_{ij}$ indicates first-order interference, while its
magnitude indicates how consequential the interaction may be. This is still
only a raw-gradient approximation: clipping, Adam moments, Adam's second
moment, and weight decay change the actual optimizer update.

### Required sufficient statistics

For every pair, retain:

$$
d_{ij},\qquad
\left\lVert g_i[S_{ij}]\right\rVert_2,\qquad
\left\lVert g_j[S_{ij}]\right\rVert_2,\qquad
c_{ij},\qquad
|S_{ij}|.
$$

Also retain the aggregate diagnostic loss and complete controlled-FFN gradient
norm for every granularity. The gradient vectors themselves need not be saved.

## Parameter-group choice

The primary diagnostic should use **shared granularity-controlled FFN
parameters**. This isolates the part of the network whose active support
changes with granularity and follows the slicing model's exact FFN-prefix
layout.

An optional sensitivity analysis can measure always-shared parameters:

- attention;
- normalization;
- embeddings;
- language-model head;
- other granularity-independent parameters.

These groups should be reported separately. Combining them with the controlled
FFN into one vector risks allowing the large embedding or language-model-head
tensors to dominate the cosine and hide FFN-specific conflict.

Layerwise dot and squared-norm contributions are also useful. Global cosine
must be reconstructed by summing dot products and squared norms across layers,
not by averaging layerwise cosines.

## Measurement semantics

The clean measurement contract is:

- one identical fixed gradient-probe panel for every granularity and H run;
- one common model checkpoint for all pairwise measurements in a snapshot;
- target-token-weighted aggregate loss and gradient;
- evaluation mode with autograd;
- raw gradients before membership correction, clipping, optimizer
  preconditioning, or inverse-probability weighting;
- ordinary RNG, model mode, granularity state, and empty gradients restored
  after measurement;
- no optimizer, scheduler, dataloader-cursor, policy, or exposure update.

The probe data must not be consumed by any H-window optimizer update. It should
be fixed once and reused across $H=1,5,25,50$ so changes in cosine reflect the
model checkpoints rather than different examples. The existing prepared
gradient-probe subset can serve this purpose for the completed uniform-H runs
because those runs never train on it.

The final holdout should not be used as the diagnostic panel.

## Feasibility and expected cost

The current model and data pipeline already exposes the ingredients needed by
an independent diagnostic:

- fixed-panel materialization;
- one full gradient evaluation per granularity;
- exact controlled-FFN support extraction;
- raw-gradient semantics;
- state and RNG restoration;
- distributed-safe layerwise extraction.

The available fixed probe panel contains 77 packed sequences, 78,771 causal
targets, and 10 batches. Observed cost for one full snapshot is:

| Granularity grid | Backward evaluations | Approximate duration |
| --- | ---: | ---: |
| Eight widths | 80 | 15.4 seconds |
| Four widths | 40 | 7.8 seconds |

The controlled gradient vectors require approximately 226 MB of temporary
FP32 CPU storage for the eight-width grid and 126 MB for the four-width grid.
This stores only the coordinates active for each granularity. Pairwise dot
products can be accumulated as each new gradient is transferred to CPU, after
which all vectors are discarded.

The pairwise matrix has only

$$
\frac{K(K-1)}{2}
$$

distinct off-diagonal entries: 28 for eight widths and 6 for four widths.
Calculating these entries requires no additional forward or backward passes
after the $K$ gradients have been measured.

## What the current action histories already show

The observed distance between distinct consecutive decisions is effectively
unchanged across H:

- eight-width mean distance: approximately $3.02$ to $3.04$;
- four-width mean distance: approximately $1.64$ to $1.68$.

The number of distinct switches changes substantially:

| $H$ | Eight-width switches | Four-width switches |
| ---: | ---: | ---: |
| 1 | 10,661 | 9,201 |
| 5 | 2,142 | 1,844 |
| 25 | 424 | 368 |
| 50 | 206 | 180 |

Therefore, $H=5$ did not perform worse because its uniform schedule happened
to make larger granularity jumps. A simple claim that fewer switches always
reduce interference would predict monotonic improvement from $H=1$ to $H=50$,
which the results contradict. The diagnostic must be able to distinguish
gradient geometry from optimizer-state dynamics.

## Working explanation for why H=1 can beat H=5

The statement that persistence reduces interference is insufficient by
itself. If reducing the number of switches were the only effect, performance
should improve monotonically as H increases. The observed dip at $H=5$
instead suggests two competing regimes.

### H=1 as optimization of a mixed objective

Define the uniformly averaged loss and gradient as

$$
\bar L(\theta)
=\frac{1}{K}\sum_{i=1}^{K}L_i(\theta),
$$

and

$$
\bar g(\theta)
=\nabla_\theta\bar L(\theta)
=\frac{1}{K}\sum_{i=1}^{K}g_i(\theta).
$$

Under per-update independent uniform sampling,

$$
\mathbb{E}_{G_t}\left[g_{G_t}(\theta)\right]
=\bar g(\theta).
$$

Ignoring model drift over a short interval, H=1 therefore gives an unbiased
stochastic-gradient estimate of the uniformly averaged objective. Adam's
moments repeatedly see gradients from different granularities and can remain
close to a mixture of their directions rather than specializing strongly to
one granularity.

### Partial moment adaptation under a short block

Adam's first moment follows

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t.
$$

If the gradient is approximated as a constant $g_i$ for H consecutive updates
of granularity $G_i$, then

$$
m_{t+H}
=\beta_1^Hm_t+(1-\beta_1^H)g_i.
$$

The coefficient

$$
1-\beta_1^H
$$

is the fraction of the old first-moment state replaced by the current
granularity's direction during the window. For the experiment's
$\beta_1=0.9$:

| H | $1-\beta_1^H$ |
| ---: | ---: |
| 1 | 0.100 |
| 5 | 0.410 |
| 25 | 0.928 |
| 50 | 0.995 |

Adam's second moment follows the elementwise recurrence

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^{\odot2},
$$

where $g_t^{\odot2}$ denotes the elementwise squared gradient. Under the same
constant-gradient approximation,

$$
v_{t+H}
=\beta_2^Hv_t+(1-\beta_2^H)g_i^{\odot2}.
$$

For $\beta_2=0.95$:

| H | $1-\beta_2^H$ |
| ---: | ---: |
| 1 | 0.050 |
| 5 | 0.226 |
| 25 | 0.723 |
| 50 | 0.923 |

The corresponding rough memory scales are

$$
\tau_1\approx\frac{1}{1-\beta_1}=10,
\qquad
\tau_2\approx\frac{1}{1-\beta_2}=20
$$

optimizer updates. H=5 is shorter than both scales. It can move the first
moment substantially toward the current granularity while leaving both moments
partially adapted when the next switch occurs.

This motivates three candidate regimes:

1. **H=1, interleaved mixing.** Each granularity only nudges the moments before
   another granularity is sampled. The optimizer remains closer to the mixed
   objective $\bar L$.
2. **H=5, partial adaptation.** The moments move away from the mixed objective,
   but the window ends before a stable granularity-specific regime is reached.
   The next granularity inherits a direction and scale biased toward the
   previous one.
3. **H=25/50, coherent adaptation.** The moments have time to become mostly
   consistent with the held granularity. A switching transient remains, but it
   occupies a smaller fraction of the window and can be amortized by many
   subsequent coherent updates.

The working decomposition is therefore

$$
\text{net H effect}
=\underbrace{\text{loss of interleaved gradient mixing}}_{
  \text{can hurt short windows}
}
+\underbrace{\text{coherent within-granularity optimization}}_{
  \text{can help long windows}
}.
$$

H=5 may incur the first term without receiving enough of the second. This is a
falsifiable working hypothesis, not yet an explanation established by the
existing artifacts.

## Isolated H=1 versus H=5 causal experiment

A controlled short experiment is the most direct way to test the partial
adaptation hypothesis. Branch all conditions from exactly the same:

- model parameters;
- Adam first- and second-moment state;
- scheduler state and learning rate;
- fixed training-batch collection;
- per-granularity exposure counts;
- gradient-probe panel.

Construct one event collection containing the same pairs

$$
(G_i, B_{i,r}),
$$

where $B_{i,r}$ is a training batch assigned to granularity $G_i$. Reorder the
same events into:

- an interleaved schedule approximating H=1;
- five-update blocks approximating H=5.

Every granularity must receive the same batches and the same number of updates
in both conditions. Only temporal ordering changes. Several deterministic
block-order permutations should be used so one favorable order is not mistaken
for the H effect.

For eight granularities, 2,400 updates contain six complete balanced H=50
cycles and approximately 20M tokens. This is a reasonable initial horizon
because the existing H trajectories begin separating during the first
2,000–5,000 updates. The same controlled replay can include H=25 and H=50,
but the primary unresolved comparison is H=1 versus H=5.

### Within-window quantities

For H=5, index the update position inside a window by

$$
r\in\{1,2,3,4,5\}.
$$

Track alignment between the carried first moment and the active gradient:

$$
a_t^{\mathrm{active}}
=\frac{m_t[S_{G_t}]^\top g_{G_t}[S_{G_t}]}
{\lVert m_t[S_{G_t}]\rVert_2
 \lVert g_{G_t}[S_{G_t}]\rVert_2}.
$$

At a switch from $G_i$ to $G_j$, track incoming alignment on their common
support:

$$
a_{i\rightarrow j,t}^{\mathrm{incoming}}
=\frac{m_t[S_{ij}]^\top g_j[S_{ij}]}
{\lVert m_t[S_{ij}]\rVert_2
 \lVert g_j[S_{ij}]\rVert_2}.
$$

Positive incoming alignment means descent based on the carried first moment is
locally helpful to $G_j$; negative alignment indicates a harmful inherited
direction.

Measure cross-granularity loss transfer over one block as

$$
T_{i\rightarrow j}(H)
=L_j\!\left(\theta_{\mathrm{after}\;i^H}\right)
-L_j\!\left(\theta_{\mathrm{before}\;i^H}\right).
$$

Interpretation:

- $T_{i\rightarrow j}(H)<0$: the $G_i$ block helped $G_j$;
- $T_{i\rightarrow j}(H)>0$: the block caused forgetting or interference for
  $G_j$.

Let $u_t$ denote the effective Adam update direction after moment
preconditioning. Update-direction volatility can be summarized as

$$
V_t=\lVert u_t-u_{t-1}\rVert_2.
$$

Track how many updates after a switch are required for probe loss, incoming
alignment, and update volatility to return to their within-window stable
levels. If this recovery requires roughly 10--20 updates, H=5 necessarily ends
before recovery while H=25/50 can spend part of each window in the recovered
regime.

### Falsifiable predictions

The partial-adaptation explanation predicts that:

1. $a_t^{\mathrm{active}}$ improves from window positions 1 through 5, showing
   that the optimizer is moving toward the held granularity.
2. $a_{i\rightarrow j,t}^{\mathrm{incoming}}$ drops at H=5 switches because
   the incoming granularity inherits a moment biased toward $G_i$.
3. H=5 windows end before alignment and update scale stabilize.
4. H=1 remains better aligned with $\bar g$, even if it is less aligned with
   whichever single granularity happens to be active.
5. H=25/50 experience a boundary shock but amortize it over enough stable
   updates to outperform H=1.

If these patterns do not appear under matched data and exposure, optimizer
partial adaptation is not a sufficient explanation for the H=5 result.

### Optimizer interventions

Two small isolated interventions can distinguish optimizer memory from generic
architecture-gradient autocorrelation:

1. **Faster moments:** repeat H=5 with smaller $\beta_1$ and $\beta_2$, so the
   moment state adapts within five updates.
2. **Boundary reset:** reset selected shared moment coordinates at H=5 window
   boundaries while leaving model parameters and the action schedule unchanged.

If either intervention closes the H=1/H=5 gap and improves incoming alignment,
stale or partially adapted moments are implicated. If H=5 remains worse, the
more likely cause is correlated architecture-gradient noise or an ordering
interaction not mediated primarily by Adam memory.

## Measurement schedule

### Existing completed runs

The completed H runs retain final/best/latest checkpoints but not model states
at historical validation milestones. They can support a terminal
cross-sectional feasibility study:

1. load the terminal selected checkpoint for each of $H=1,5,25,50$;
2. measure the complete shared-FFN pairwise matrix on the fixed gradient-probe
   panel;
3. compare both the four- and eight-width grids.

This should take only a few minutes including checkpoint loading. It cannot
determine whether any final alignment difference is a cause or consequence of
the training result.

### Prospective runs

For future experiments, use a sparse set of common milestones:

1. initialization;
2. step 1,000, at the warmup boundary;
3. 25% of the token budget;
4. 50% of the token budget;
5. 75% of the token budget;
6. completion.

Six eight-width snapshots add roughly 90 seconds of observed measurement time
per run. Measuring at every H-window boundary is unnecessary and would make
measurement cost depend strongly on H.

Record the active window index, committed progress, held granularity, and next
decision state at each snapshot. Milestones should remain fixed by step or
token count rather than being selected because a particular transition looked
interesting.

## Candidate summaries

### Similarity versus granularity distance

Let

$$
\Delta_{ij}=|i-j|.
$$

For each distance $\delta$, calculate

$$
\bar c_{\delta}
=\frac{1}{N_\delta}
\sum_{i<j:\,\Delta_{ij}=\delta}c_{ij},
$$

where $N_\delta$ is the number of pairs at that distance. Report the number of
pairs and their range or panel-shard uncertainty; pairs are structured
architectural comparisons, not independent experimental replicates.

### Negative-interference summaries

Useful secondary scalars include

$$
\text{negative-pair fraction}
=\frac{2}{K(K-1)}\sum_{i<j}\mathbb{1}[d_{ij}<0],
$$

and

$$
\text{mean negative magnitude}
=\frac{2}{K(K-1)}\sum_{i<j}\max(0,-d_{ij}).
$$

Also report nearest-neighbor and farthest-pair values separately. A single
off-diagonal mean can hide one severe conflicting pair.

### Consecutive sampled granularities

Consecutive-action summaries must condition on an actual change:

$$
G_t\ne G_{t-1}.
$$

Including within-window repeats would inject many diagonal similarities of
one into large-H runs and create a trivial, misleading improvement.

At a fixed diagnostic snapshot, map observed distinct decision transitions to
their corresponding pairwise cosine and dot product. Report:

- mean cosine conditional on switching;
- fraction of negative switch pairs;
- transition-weighted negative dot magnitude;
- optionally, an interference burden per optimizer update:

$$
\mathcal I_H
=\Pr(G_t\ne G_{t-1})
\;\mathbb{E}\!\left[
  \max(0,-d_{G_{t-1},G_t})
  \mid G_t\ne G_{t-1}
\right].
$$

$\mathcal I_H$ must be labeled as a derived quantity because part of its H
dependence is mechanically determined by switch frequency.

## Visualization proposal

### 1. Pairwise heatmaps

Use small-multiple $K\times K$ heatmaps for $H=1,5,25,50$ at one common
milestone:

- fixed granularity order;
- one shared diverging scale from $-1$ to $1$;
- only one matrix triangle;
- annotations for negative or near-zero cells;
- no per-panel color normalization.

This is the primary view for identifying specific conflicting pairs.

### 2. Similarity versus distance

Plot $\bar c_\delta$ against $\delta=|i-j|$ with one H line per run and pair
ranges or panel-shard uncertainty. This tests whether more separated widths
have systematically less compatible gradients.

### 3. Temporal H summary

Across the sparse milestones, plot:

- mean off-diagonal cosine;
- negative-pair fraction;
- mean negative dot magnitude;
- nearest-neighbor and farthest-pair cosine.

These summaries can be aligned with validation-loss differences, but should
not replace the pairwise heatmaps.

### 4. Transition-conditioned distribution

Use an ECDF, compact violin, or interval plot for cosine values associated with
distinct observed decision transitions. Do not include same-granularity
within-window updates.

## Full optimizer-aware follow-up

The first-moment alignments above are inexpensive but do not include Adam's
second-moment preconditioner. A more faithful diagnostic can construct a
virtual update for counterfactual granularity $G_i$ without changing the saved
optimizer state:

$$
m_i'=\beta_1m_t+(1-\beta_1)g_i,
$$

$$
v_i'=\beta_2v_t+(1-\beta_2)g_i^{\odot2},
$$

and, omitting bias-correction indices for compactness,

$$
u_i=
\frac{m_i'}{\sqrt{v_i'}+\epsilon}
+\lambda\theta,
$$

where $\epsilon$ is Adam's numerical stabilizer and $\lambda$ is the AdamW
weight-decay coefficient. The corresponding first-order effect on another
granularity is

$$
P_{i\rightarrow j}
=g_j[S_{ij}]^\top u_i[S_{ij}].
$$

Because the optimizer applies $-u_i$, a negative $P_{i\rightarrow j}$ implies
that the virtual $G_i$ update would locally increase $L_j$. Comparing
$P_{i\rightarrow j}$ with the raw dot $d_{ij}$ separates conflict already
present in the gradients from conflict introduced or amplified by the saved
optimizer state.

This calculation is more complex than raw cosine and should follow rather than
replace the shared-gradient baseline. It is particularly valuable if pairwise
raw-gradient geometry looks similar across H while first-moment alignment and
validation outcomes differ.

## Existing artifacts versus additions

The H pipeline already preserves most of the experimental evidence needed by
this diagnostic. Those artifacts should be reused rather than duplicated.

### Already available; do not introduce again

| Needed information | Existing source |
| --- | --- |
| Resolved H, granularities, seeds, model, training budget, and code/data identity | `config.json` and the existing manifests |
| Committed action at every update | `metrics.csv` |
| Window index, committed progress, total successful updates, and exposure counts | `metrics.csv` and `heartbeats.jsonl` |
| Final H-window continuation state and exposure totals | `run_summary.json` |
| Probe-data identity and role separation | existing data-role and fixed-probe manifests |
| Validation trajectory and selected checkpoint | `metrics.csv`, validation manifest, and `run_summary.json` |
| Final evaluation | existing final-holdout manifest and results |
| Resumable model, optimizer, scheduler, RNG, data-cursor, and H-window state at a saved point | existing continuation-checkpoint schema |

Consequently, this proposal does **not** require a second action log, a second
exposure summary, a new probe manifest, a new configuration/provenance file, or
a new validation/holdout artifact.

### Addition 1: gradient-interference measurements

The one genuinely new scientific artifact is
`gradient_interference.jsonl`. It stores quantities that the current pipeline
does not calculate and that cannot be inferred from losses or action history.
Use one atomic record per measured checkpoint with:

- checkpoint step and hash plus the fixed-probe and controlled-support hashes;
- ordered per-granularity probe losses and controlled-FFN gradient norms;
- for every pair, the shared-parameter count, dot product, two restricted
  norms, cosine, and granularity distance;
- optional layerwise dot and squared-norm contributions if layerwise analysis
  is enabled;
- measurement batch/target/backward counts and elapsed time.

H, tokens, window state, actions, exposures, and run identity should be joined
from the existing config and metrics whenever possible. Only compact keys
needed to verify the join belong in the new record. Raw gradient vectors do not
need to be retained.

### Addition 2: immutable milestone retention

This is not a new checkpoint schema. It is only a new retention policy for
additional instances of the existing full continuation checkpoint. The current
rolling `latest.pt` and `latest.prev.pt` preserve resumability near the end but
overwrite earlier trajectory states.

If an H run must be repeated, retain immutable full-state checkpoints at the
common milestones already proposed:

1. initialization;
2. warmup completion;
3. 25% of the token budget;
4. 50% of the token budget;
5. 75% of the token budget;
6. completion.

Names such as `diagnostic_step_003052.pt` are sufficient. The payload remains
the existing continuation payload. These retained states are the protection
against future checkpoint-level diagnostics that have not yet been designed.
At the observed size of approximately 1.77 GB per continuation checkpoint, six
states cost about 10.6 GB per run.

### Addition 3: derived comparison outputs

`scripts/make_figures.py` should derive, rather than train-time code should
store:

- a tidy pair table with one row per H, milestone, and granularity pair;
- the common H-subfigure heatmap;
- similarity-versus-distance and temporal summary figures.

The table and figures are reporting products. They are regenerated from
`gradient_interference.jsonl` plus existing run artifacts and are not part of
the continuation contract.

### Not proposed yet: a transition trace

Immediate pre-switch/post-switch Adam alignment cannot be reconstructed at
every boundary from sparse checkpoints. A sampled online transition trace
would therefore be a genuinely new artifact, but its event definition and
sampling rule should be added only if we decide that transition-local Adam
behavior is a required question for the rerun. It is not necessary for the
first pairwise-gradient diagnostic and should not be introduced speculatively.

## Minimal recommended sequence

1. Implement and validate `gradient_interference.jsonl` on an existing terminal
   checkpoint.
2. Add its common-H plots and pair table to `scripts/make_figures.py`.
3. Before any full H rerun, add immutable milestone retention using the current
   continuation-checkpoint payload.
4. Decide separately whether the H=1 versus H=5 question justifies the only
   other possible new artifact: a sampled transition-local trace.
5. Keep terminal alignment labeled as correlational until temporal evidence or
   an isolated intervention establishes causality.
