# TinyStories optimizer ownership: results and the largest-standalone gap

Date: 2026-09-10

Status: analysis of the completed seed-42 campaign. Proposed follow-up experiments
below have not been run and are not launch authorization. The
[feature plan](../specs/013-tinystories-optimizer-ownership/plan.md) and saved run
configurations remain authoritative for implementation and scientific controls.

## Main finding

The central concern is that **none of the five elastic methods reaches the
largest standalone's validation quality, despite receiving four times its total
training-token budget**. Small differences among optimizer methods are secondary
to this failure to preserve full-width quality while learning smaller subnetworks.

The implementation audits found no evidence that the methods accidentally
collapsed into the same implementation. A leading hypothesis is interference
between the objectives of different widths sharing the same weights. This is a
hypothesis supported by the experimental structure, not a demonstrated causal
explanation of the gap.

## 1. Observed results

All nine runs completed. These are terminal **ordinary-validation** losses at
the assigned budgets, not best-checkpoint or holdout results. The sealed holdout
was not evaluated.

| Method | g250 | g500 | g750 | g1000 |
| --- | ---: | ---: | ---: | ---: |
| Standalone at the matching width | 2.062977 | 1.991942 | 1.943363 | 1.909057 |
| S1: slicing, shared AdamW | 2.068157 | 1.995913 | 1.966166 | 1.953872 |
| S2: slicing, per-width AdamW | 2.067589 | 1.991578 | 1.963845 | 1.951732 |
| C1: concat, shared AdamW | 2.073095 | 1.998167 | 1.968448 | 1.954377 |
| C2: concat, per-width AdamW | 2.071721 | 1.994960 | 1.963783 | 1.951738 |
| C3: concat, disjoint owners and separate clipping | 2.077839 | 1.999872 | 1.968143 | 1.954962 |

The best-to-worst elastic perplexity spreads are approximately **1.03%, 0.83%,
0.47%, and 0.32%**, respectively. Perplexity ratios are computed as
`exp(loss_difference)`.

At g1000, the best elastic loss is 1.951732 versus 1.909057 for the standalone:
approximately **4.36% higher perplexity**. At g750, the corresponding gap is
approximately **2.06%**. These gaps are larger than the variation among elastic
methods at those widths.

S2 beats S1 at every width, and C2 beats C1 at every width. This is compatible
with a modest benefit from per-width optimizer state, but one seed does not
establish reliability or statistical significance. Similar aggregate losses
also do not establish identical predictions or equivalent performance per story.

## 2. What the four-times budget provides

Each standalone trained for 87,132 updates and 713,785,344 packed tokens. Each
elastic run trained for 348,528 updates and 2,855,141,376 packed tokens. Thus each
elastic run matches the aggregate token budget of the four standalones.

The elastic methods sampled one width per update, uniformly with replacement.
Their realized selections were identical across methods:

| Selected width | Updates |
| --- | ---: |
| g250 | 86,898 |
| g500 | 87,221 |
| g750 | 87,337 |
| g1000 | 87,072 |

Consequently, the full-width network was selected approximately as often as the
largest standalone was trained. The remaining updates trained smaller widths.

| Parameter group | Updates on which it participates in the forward/backward computation |
| --- | ---: |
| Common layers and FFN quarter A | 348,528 |
| FFN quarter B | 261,630 |
| FFN quarter C | 174,409 |
| FFN quarter D | 87,072 |
| Every parameter of the largest standalone | 87,132 |

These are **active gradient exposures**, not universally the number of weight
mutations or AdamW counter increments. In slicing, a full tensor has a gradient
with zeros outside the active prefix; inactive entries can still change through
weight decay and, where populated, residual momentum. In concat, inactive blocks
have no gradient and their optimizer state does not advance.

Similar selection counts do not imply identical example coverage. A standalone
sees the optimizer dataset once; a width sampled during four elastic passes may
see the same training position multiple times and miss another position entirely.
The campaign matches total tokens, not per-width example coverage or FLOPs.

This allocation explains why four times the total budget is not equivalent to
four times the full-width training. It does not remove the practical concern:
the extra elastic training still failed to recover the largest standalone's quality.

## 3. Why the elastic methods may perform similarly

### Same forward-function family

Vanilla slicing and concat implement the same SwiGLU computation at matched
weights and width. Slicing selects rows/columns of full projection matrices;
concat assembles the selected independently registered blocks. Concat changes
parameter storage and inactive-parameter optimization, without adding capacity
or a different forward-function family.

### Separate optimizer state still acts on shared model weights

S2/C2 maintain separate AdamW statistics for each width, not separate copies of
the model. All widths still modify the same attention, embeddings, and overlapping
FFN parameters. Separate optimizer state therefore does not eliminate the
compromises required by shared weights.

Approximately 62.5% of the physical parameters lie outside the FFN quarters and
participate at every width. All methods also share the same action sequence,
data order, token budget, and learning-rate schedule. These common conditions
plausibly contribute to similar curves; parameter count alone does not identify
which components dominate learning.

### C3 mainly changes clipping relative to C1

Disjoint AdamW optimizers are equivalent to shared AdamW when the corresponding
parameters receive identical gradients, learning rates, and counter advances.
The matched-state diagnostic verifies this with global clipping. In the actual
campaign, C3's substantive distinction from C1 is separate clipping per owner.

The distinction was active: C1 globally clipped on 88,581 updates, while C3
clipped its common group on 81,869 updates, quarter A on 10 updates, and quarters
B/C/D on none. These records rule out a silent fallback to identical clipping,
but do not prove why the resulting quality differences are small.

## 4. Leading hypothesis for the largest-standalone gap

Uniform width sampling targets the expected loss

\[
J(\theta)=\tfrac14\left(L_{250}(\theta)+L_{500}(\theta)
                         +L_{750}(\theta)+L_{1000}(\theta)\right).
\]

Improving this average does not guarantee matching the minimum achievable
full-width loss. For example, a g250 update modifies common layers and the first
FFN quarter without adapting the remaining quarters to those changes. A subsequent
full-width update must use those changed shared components.

This creates an opportunity for conflicting updates between widths. All five
methods retain that shared-weight objective, so changing optimizer ownership may
leave the main source of the full-width gap intact.

The current data does not separate this hypothesis from alternatives such as
insufficient full-width exposure, repeated-data coverage differences, optimization
or schedule choices, and initialization/run variation. A larger gap to the
standalone is evidence to investigate width interference, not proof of it.

## 5. Scheduler and interpretation of the loss curves

All elastic methods used linear warmup over 64 updates to learning rate 0.008,
followed by cosine decay to zero at update 348,528. The clock advances once per
global update and synchronizes the width/owner optimizers.

At update 10,000 the learning rate is approximately 0.007984. There is no scheduler
boundary there. The plot beginning at 10,000 is an additional zoom to prevent the
initial sharp loss drop from compressing the later curves; the full-progress plot
retains all recorded measurements. Neither plot applies smoothing.

The declining learning rate may contribute to the later loss improvement, but
the curves alone cannot separate its effect from additional training. Equal
token budgets across protocols also do not imply an identical effective schedule
for every physical parameter or every width-specific optimizer.

## 6. Implementation validation completed

- Checked the actual runtime source hashes, saved configs, terminal checkpoints,
  optimizer state allocation, and per-parameter counters for all five elastic methods.
- Verified correct width selection and gradient routing through every layer.
- Confirmed different final weights across all pairs of elastic methods.
- Confirmed identical sampled-action and four-epoch data-order digests across methods.
- Ownership/campaign/resume tests: **255 passed, 28 GPU-only skipped**.
- Vanilla FFN/model-size tests: **56 passed**.
- Independent vanilla comparisons: **32 FFN cases** and **8 full-transformer
  cases**, covering all campaign widths. Forward results matched dense references
  exactly; gradient differences were zero or float64 rounding below `6e-17`.
- Mixed-width SGD preserved slicing/concat equivalence; AdamW diverged after a
  narrow update as expected from inactive-parameter semantics.

Fresh validation probes ran on CPU, with membership correction disabled as in
the campaign. These checks found no implementation mismatch; they are not a
claim that every possible GPU/kernel/configuration behavior has been exhaustively
validated.

## 7. Proposed diagnostic: introduce smaller widths from a trained full model

The next diagnostic discussed is to begin from the trained largest standalone,
which already achieves the target quality, and introduce smaller-width updates
while tracking ordinary-validation loss at every width.

Use a matched full-width-only continuation as a control. Both branches should
start from the same weights, use a nonzero continuation learning-rate schedule,
and have an explicitly matched optimizer-state policy and data order. Simply
resuming the completed zero-learning-rate schedule would not test the hypothesis.
For concat, any dense-to-block conversion must preserve the starting full-width
function and follow a documented optimizer-state mapping or reset policy.

If adding smaller-width updates degrades full-width quality relative to the
continuation control, that would provide more direct evidence of interference.
The original standalone loss anchors whether the full-width target is preserved.
Warm-start behavior would still not, by itself, prove the cause of the gap in
the original from-scratch campaign or establish that the trade-off is unavoidable.

Additional seeds and paired per-story validation comparisons would help assess
the reliability and distribution of the smaller method-to-method differences.
These are proposed follow-ups only; no new run or evaluation is authorized by
this note.

## Evidence and artifacts

All campaign results remain under
`/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1`.

- [Complete report and endpoint tables](/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/reports/complete-20260910T181951Z/report)
- [Four-panel elastic loss progress and source CSV](/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/reports/elastic-loss-progress-20260910T184040Z)
- [Optimizer-ownership implementation audit](/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/diagnostics/implementation-audit-20260910/report.md)
- [Vanilla slicing/concat audit](/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/diagnostics/vanilla-ffn-audit-20260910/report.md)
- [Original experiment design](tinystories_optimizer_ownership_experiments.md)

Runtime measurements require separate care: S1/S2 cumulative costs include
original and fixed-code attempts plus replayed work. They should not be interpreted
as measurements of fixed-code throughput alone.
