# Full-Width Prefix Transformer: Exploratory Design Note

**Status**: exploratory architecture and training note  
**Scope**: a true width-nested decoder-only Transformer, distinct from the
existing FFN-only MatFormer implementation  
**Question**: can one train a single Transformer whose extracted *whole-model*
width prefixes approach independently trained models of the same widths?

## Motivation

The current MatFormer model changes only the FFN intermediate width.  The
token embeddings, residual stream, normalization parameters, attention
projections, attention-head count, and LM head stay at the full model width.
This is faithful to the primary MatFormer construction, which keeps
$d_{\mathrm{model}}$ fixed and nests FFN neurons; the paper mentions head and
embedding nesting as possible extensions but does not make them its main
decoder experiment [MatFormer, Section 3](https://arxiv.org/html/2310.07707v2).

Consequently, an FFN-only submodel is not a conventional smaller Transformer:
it retains the full embedding and attention pathway.  A full-width prefix
model tests whether good elastic behavior persists when all of those components
scale down together.

This is a new experimental family, not a small extension of the current
`slicing` or `concat` variants.

## Model definition

Let the maximum hidden width be $D$, let

$$
D_1 < D_2 < \cdots < D_K = D
$$

be a fixed, ordered width ladder, and let $F_g$ be the corresponding FFN
intermediate width for submodel $g$.  Usually, choose

$$
F_g = r D_g,
$$

with one fixed FFN expansion ratio $r$, rounded only in a declared,
architecture-valid way.

Every extracted model uses the first $D_g$ hidden coordinates.  If $[a:b]$
denotes the usual half-open slice, then all widths are prefixes of the same
maximum-width parameter tensors.

### Embedding, residual stream, normalization, and LM head

For vocabulary size $V$, input embedding $E \in \mathbb{R}^{V\times D}$,
untied output head $W_{\mathrm{lm}}\in\mathbb{R}^{V\times D}$, and an RMSNorm
scale vector $\gamma\in\mathbb{R}^D$:

$$
E_g = E[:, :D_g],
\qquad
\gamma_g = \gamma[:D_g],
\qquad
W_{\mathrm{lm},g} = W_{\mathrm{lm}}[:, :D_g].
$$

Thus, all residual activations in submodel $g$ lie in
$\mathbb{R}^{D_g}$ and its logits remain in $\mathbb{R}^{V}$:

$$
\ell_g = h_g W_{\mathrm{lm},g}^{\mathsf T}.
$$

Learned position embeddings, if used, are sliced in their hidden dimension in
the same way.  RoPE has no learned hidden-width tensor, but its head geometry
must be valid at every width.

### Attention

For a standard dense-attention block, the query, key, value, and output
projections are all principal submatrices:

$$
W^{(g)}_Q = W_Q[:D_g,:D_g],
\quad
W^{(g)}_K = W_K[:D_g,:D_g],
\quad
W^{(g)}_V = W_V[:D_g,:D_g],
\quad
W^{(g)}_O = W_O[:D_g,:D_g].
$$

Use complete heads, never fractions of a head.  Choose a fixed head dimension
$d_h$ and an integer head count

$$
H_g = D_g / d_h.
$$

The submodel then uses the first $H_g$ heads.  This preserves the rotary
pairing and the usual $1/\sqrt{d_h}$ attention scale.  With grouped-query
attention, choose integer $H_{\mathrm{kv},g}$ values that preserve the
declared query-to-KV-head ratio at every width.

The width ladder must therefore be selected from feasible
$(D_g,H_g,H_{\mathrm{kv},g},F_g)$ tuples; existing FFN labels such as `g125`
should not be reused mechanically unless their new whole-model shapes are
declared explicitly.

### SwiGLU FFN

For Llama-style gate/up/down projections, with maximum-width tensors

$$
W_{\mathrm{gate}}, W_{\mathrm{up}} \in \mathbb{R}^{F\times D},
\qquad
W_{\mathrm{down}}\in\mathbb{R}^{D\times F},
$$

the width-$g$ FFN is

$$
W^{(g)}_{\mathrm{gate}} = W_{\mathrm{gate}}[:F_g,:D_g],
\qquad
W^{(g)}_{\mathrm{up}} = W_{\mathrm{up}}[:F_g,:D_g],
$$

$$
W^{(g)}_{\mathrm{down}} = W_{\mathrm{down}}[:D_g,:F_g],
$$

and

$$
\operatorname{FFN}_g(x) =
\left[\operatorname{SiLU}\!\left(x{W^{(g)}_{\mathrm{gate}}}^{\mathsf T}\right)
\odot x{W^{(g)}_{\mathrm{up}}}^{\mathsf T}\right]
{W^{(g)}_{\mathrm{down}}}^{\mathsf T}.
$$

Biases, if enabled, are sliced on their corresponding output dimension.

## What this does and does not make reusable

It is possible to avoid *reconstructing* matrices at runtime: a training
forward can operate on prefix views, or on a block layout designed to feed
kernel-friendly contiguous tensors.  Materializing a compact standalone
checkpoint at export time is harmless and should be lossless.

It is **not** possible to obtain the narrow model's activations from one
ordinary wide-model forward.  In a dense wide layer, its low-index output
channels include contributions from high-index input channels:

$$
(Wx)_{:D_g}
= W[:D_g,:D_g]x_{:D_g}
 + W[:D_g,D_g:]x_{D_g:}.
$$

### Block-matrix derivation

At one width boundary, partition a wide hidden state and its dense weight
matrix into narrow-prefix and additional coordinates:

$$
x_{\mathrm{wide}}
= \begin{bmatrix}x_{\mathrm{small}}\\x_{\mathrm{extra}}\end{bmatrix},
\qquad
W_{\mathrm{wide}}
= \begin{bmatrix}A & B\\C & D\end{bmatrix}.
$$

The standalone narrow layer has only the prefix input and prefix weight:

$$
y_{\mathrm{small}} = A x_{\mathrm{small}}.
$$

By contrast, the first narrow-width coordinates of the ordinary wide-layer
output are

$$
(y_{\mathrm{wide}})_{:D_g}
= A x_{\mathrm{small}} + B x_{\mathrm{extra}}.
$$

The extra term is generally nonzero.  For example,

$$
x_{\mathrm{small}}=1,
\qquad x_{\mathrm{extra}}=3,
\qquad
W_{\mathrm{wide}}=
\begin{bmatrix}2 & 5\\ \cdot & \cdot\end{bmatrix}
$$

gives

$$
y_{\mathrm{small}}=2\cdot1=2,
\qquad
(y_{\mathrm{wide}})_{:D_g}=2\cdot1+5\cdot3=17.
$$

Thus `wide_output[..., :D_g]` is generally *not* the result produced by the
narrow prefix model.  The discrepancy exists from the token embedding onward:
the wide model's extra embedding coordinates are nonzero and can influence its
prefix outputs through every dense projection.

The narrow model has only the first term.  Reusing the wide forward would
therefore change the narrow model unless all such cross-width blocks were
constrained to zero:

$$
W_{\mathrm{wide}}=
\begin{bmatrix}A & 0\\C & D\end{bmatrix}.
$$

This is a block-lower-triangular restriction.  The added coordinates can read
the shared core through $C$, but they cannot feed information back into it.  It
is a legitimate, but substantially more constrained, wide-model architecture.

### Normalization makes ordinary reuse fail even under that restriction

The block-triangular condition alone is insufficient in a standard
Transformer.  RMSNorm, for example, uses a statistic over its whole input
width.  Ignoring its learned scale for brevity, its narrow and wide prefix
normalizations are respectively

$$
\operatorname{RMSNorm}_{\mathrm{small}}(x_{\mathrm{small}})
= \frac{x_{\mathrm{small}}}
{\sqrt{D_g^{-1}\sum_{i=1}^{D_g}x_i^2+\varepsilon}},
$$

and

$$
\left(\operatorname{RMSNorm}_{\mathrm{wide}}(x_{\mathrm{wide}})\right)_{:D_g}
= \frac{x_{\mathrm{small}}}
{\sqrt{D^{-1}\left(\sum_{i=1}^{D_g}x_i^2+
\sum_{i=D_g+1}^{D}x_i^2\right)+\varepsilon}}.
$$

These are different whenever the extra coordinates contribute nonzero energy.
LayerNorm has the same issue through its full-width mean and variance.  A
prefix-invariant architecture would need explicitly width-specific or
blockwise normalization in addition to the projection constraints.

Accordingly, the claim is not that full-width prefixes require copying or
reconstructing matrices.  A selected width may use tensor views such as
`W[:D_g, :D_g]` and `x[..., :D_g]`, and a compact standalone checkpoint can be
materialized later.  Rather, each selected width needs its own forward
computation.  Multiple widths may be batched or their gradients accumulated,
but they cannot be recovered by slicing ordinary wide-forward activations.

This is the major trade-off relative to
[Matryoshka Language Model Suites](https://arxiv.org/html/2608.09703v1).  That
approach nests complete *stacks of layers*: a smaller exit is followed by a
norm-matched junction that concatenates fresh embedding dimensions before the
new, wider layers.  It thereby obtains all exits on one forward path and can
distill from the largest exit at low additional cost.  Its smaller logits are
emitted before the extra dimensions and later layers are introduced, so it
does not need those later computations to reproduce the smaller model.  It
does not need in-place prefix KQV matrices.

## Topology constraint

All Transformer layers in one extracted prefix model must use the same global
width $D_g$.  A residual add is undefined across different hidden widths.

Therefore, the present MatFormer-style per-layer Mix'n'Match topology should
not be inherited by this model family.  Changing widths between layers
requires an explicit projection or junction; that is a separate architecture
choice and moves the design toward the layer-stacked Matryoshka approach.

## Training objectives and sampling

Let $L_g(\theta; z)$ be the token loss of width $g$ on minibatch $z$.  The
most direct suite objective is a weighted average:

$$
J(\theta;z) = \sum_{g=1}^{K}\alpha_g L_g(\theta;z),
\qquad
\alpha_g\geq 0,\quad \sum_g\alpha_g=1.
$$

### Nested-all

Run every width on the same minibatch, accumulate the weighted losses, and
take one optimizer step:

$$
\nabla J = \sum_g \alpha_g \nabla L_g.
$$

This is the cleanest initial diagnostic because every width sees every
minibatch and can share the standalone schedule horizon $B$.  It requires
one forward/backward execution per width; whole-width prefix models do not
have the Matryoshka-suite forward reuse described above.

### One width sampled per optimizer step

Sample $G_t\sim p$ and train only $L_{G_t}$.  Under uniform sampling,

$$
\mathbb{E}_{G_t\sim\mathrm{Uniform}(\mathcal{G})}
\left[\nabla L_{G_t}\right]
= \nabla\left(\frac{1}{K}\sum_gL_g\right).
$$

Thus, unweighted uniform sampling is already an unbiased estimator of the
uniform-average objective up to the constant convention in the learning rate.
For a target objective $J$ under arbitrary sampling probabilities, use the
importance-weighted estimator

$$
\widehat{\nabla J}
= \frac{\alpha_{G_t}}{p_{G_t}}\nabla L_{G_t}.
$$

The latter can have large variance for rarely sampled widths, so it is not an
automatic default.

### Exposure is not a bug to correct away

Let $A_g$ be the active parameter support of width $g$, with

$$
A_1\subset A_2\subset\cdots\subset A_K.
$$

The shared core belongs to every $A_g$ and should receive gradients from every
loss in the stated suite objective.  Outer rings belong to fewer widths and
are correctly updated less often.  A blanket membership correction changes
the optimization objective; it should be disabled for the first principled
baseline and treated only as an explicit ablation.

AdamW adds a practical complication: parameter rings have different rates of
nonzero gradients and therefore different optimizer-state histories.  This is
part of the method, not a numerical nuisance.  Log update counts and the first
and second moments by width-membership ring.

Separate optimizers per granularity are possible as an exploratory method, but
shared parameters would then carry multiple momentum histories.  The resulting
parameter update depends on optimizer ordering and is no longer ordinary AdamW
on one declared objective.  It should not be the baseline.

### Scheduler alignment

For $K$ uniformly sampled widths over $KB$ global tokens, each width receives
approximately $B$ active tokens, but a single global cosine schedule evolves
over $KB$, not over each width's own $B$ exposures.  This makes a direct
comparison to standalone models with a cosine horizon $B$ ambiguous.

The first diagnostic should therefore use nested-all for $B$ tokens, with the
same schedule horizon as every standalone.  Sampling experiments should report
both global progress and each width's active-token exposure and LR phase.

## Distillation

Largest-to-smallest online distillation is not free for a prefix supernet.
The teacher logits are available only after also executing the largest-width
forward.  If every width is evaluated on a minibatch, a natural objective is

$$
L_g^{\mathrm{total}}
= (1-\lambda_g)L_g^{\mathrm{CE}}
 + \lambda_g
\operatorname{CE}\!\left(
\operatorname{stopgrad}(\operatorname{softmax}(\ell_K/T)),
\operatorname{softmax}(\ell_g/T)
\right).
$$

Run no-distillation first.  Add this as a separately budgeted nested-all
ablation after basic equivalence and optimization behavior are understood.
The Matryoshka-suite paper finds its junction and distillation both important
for small-model performance, but those conclusions belong to that distinct
architecture rather than transferring automatically here.

## Fair comparison and accounting

Every standalone reference must use the exact extracted architecture:

- vocabulary and tokenizer;
- $D_g$, $H_g$, $H_{\mathrm{kv},g}$, $F_g$, layers, positional encoding, and
  normalization type;
- corpus, data order policy, optimizer, initialization distribution, and
  scheduler definition.

The implementation should prove, at initialization and after an optimizer
update, that a materialized width-$g$ checkpoint has identical logits and
gradients to the in-place extracted prefix model.

Because width now changes embeddings, attention, KV cache, and FFN, optimizer
tokens alone are insufficient as a cost claim.  Report at least:

- active parameter count and checkpoint size for each width;
- theoretical training FLOPs/token and measured wall time/memory;
- global optimizer tokens and per-width active tokens;
- total suite FLOPs relative to the corresponding standalone portfolio;
- per-width validation PPL gaps, worst-width gap, and cross-width agreement.

## Recommended staged experiment

1. **Correctness only**: implement a separate `full_width_prefix` model family
   with global width selection.  Do not modify the existing FFN-only
   `slicing`/`concat` contracts.
2. **Shape sanity**: choose a small feasible width ladder from
   $(D_g,H_g,H_{\mathrm{kv},g},F_g)$ tuples, rather than reinterpreting legacy
   FFN granularity labels implicitly.
3. **Standalone suite**: train one ordinary model per width with the exact
   extracted shape.
4. **Nested-all, $B$ tokens**: use uniform loss weights and one shared AdamW
   optimizer.  This answers whether whole-model nesting can match at all,
   without scheduler or sampling-exposure ambiguity.
5. **Uniform global sampling**: run the fixed aggregate-budget version,
   logging active tokens, LR phase, parameter-ring optimizer diagnostics, and
   actual FLOPs.
6. **Only then ablate**: non-uniform sampling, loss weights, membership
   correction, separate optimizer states, and largest-to-smallest
   distillation.

## Interpretation

Success would support a stronger claim than FFN-only MatFormer: a single
weight tensor can host a genuinely smaller family of Transformer checkpoints.
Failure is equally informative.  If FFN-only nesting works but full-width
prefix nesting does not, the likely conclusion is that the stable shared
embedding/attention representation is an essential part of the observed
elastic behavior, rather than nested parameter sharing alone being sufficient.
