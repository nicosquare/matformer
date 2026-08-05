# Checkpoint-selection methodology discussion

Date: 2026-08-05

Status: discussion note; no methodology or plotting change has been approved or
implemented from this note.

## Context

After correcting validation construction and aggregation, the standalone
`d_model=256` runs still show `large` slightly outperforming `full`. The runs
are directly comparable: they use the same deterministic validation holdout,
comparison-control signature, valid-target count, training budget, content
tokens, seed, and training-batch order.

Relevant standalone results:

| Granularity | Effective width | Parameters | Final validation loss | Best validation loss | Trailing-five mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| micro | 128 | 136,847,616 | 4.553348 | 4.553319 | 4.554784 |
| small | 256 | 138,420,480 | 4.522071 | 4.522071 | 4.523761 |
| medium | 512 | 141,566,208 | 4.452773 | 4.452770 | 4.454418 |
| large | 768 | 144,711,936 | 4.386518 | 4.386518 | 4.388216 |
| full | 1024 | 147,857,664 | 4.389051 | 4.389043 | 4.390809 |

The full-minus-large difference is approximately `+0.002533` using final
loss and `+0.002525` using best loss. Therefore, choosing best instead of
final does not change this particular ordering.

The existing size-comparison figures read the `loss` and `perplexity` columns
from `scaling_results.csv`. Those columns currently contain the final
completion-validation values. Best validation values are recorded separately
but are not used in the size plots. Validation-over-token figures instead plot
the complete validation history.

## Concern with final loss as the universal primary metric

The current methodology treats final completed-model loss as primary because
the experiments have a fixed token budget. That interpretation is not the only
reasonable one.

A fixed budget can be understood as a maximum search budget. Under that view,
the objective is to produce the best checkpoint observed within the allowed
budget, rather than to require deployment of the last checkpoint. All
standalone runs receive the same total training budget, validation cadence,
number of checkpoint-selection opportunities, validation holdout, and other
comparison controls. Selecting the best observed validation checkpoint is
therefore a natural part of the training protocol.

Repeatedly selecting the minimum validation loss still introduces selection
bias. Without an untouched language-model test split, it must be labeled
"best validation loss within budget," not test performance. However, the
selection pressure is comparable across standalone runs when evaluation
cadence and run length are identical.

## Proposed distinction between standalone and elastic models

Standalone and elastic training produce different deployable objects, so they
do not necessarily require the same checkpoint-selection rule.

### Standalone runs

Each standalone run produces one independent model. Its primary result can be
the best validation checkpoint observed within the budget:

```text
best_standalone_loss[g] = min over evaluation steps t of validation_loss[g, t]
```

Final loss and trailing-window statistics should remain supplementary measures
of endpoint behavior and stability.

### Elastic runs

An elastic run produces one shared model whose parameters support multiple
granularities. Independently minimizing each granularity's validation loss
will generally select a different checkpoint for each granularity:

```text
oracle_loss[g] = min over evaluation steps t of validation_loss[g, t]
```

Plotting all of those minima together creates a curve assembled from multiple
checkpoints. It does not describe one deployable elastic model. This curve is
still useful as an oracle envelope, but it should not be the primary elastic
result.

The primary elastic result should evaluate every granularity at one shared
checkpoint. Select that checkpoint using a predefined aggregate validation
objective:

```text
aggregate_loss[t] = sum over granularities g of weight[g] * validation_loss[g, t]
selected_step = argmin over t of aggregate_loss[t]
reported_elastic_loss[g] = validation_loss[g, selected_step]
```

This yields one coherent checkpoint and one deployable elastic scaling curve.

## Candidate aggregate objectives for elastic selection

The aggregation rule remains unresolved. Candidates include:

1. Uniform mean validation NLL

   Give every configured granularity equal weight. This is the simplest
   default when all granularities are equally important.

2. Deployment- or sampling-weighted mean

   Use the expected deployment frequency, or possibly the configured training
   sampling distribution, as `weight[g]`. This directly optimizes expected
   deployed loss under that distribution.

3. Worst-case loss

   Select the checkpoint minimizing the maximum validation loss across
   granularities. This protects the weakest granularity but will usually be
   dominated by the smallest model.

4. Normalized regret or another multi-objective criterion

   Balance granularities relative to their attainable baselines rather than
   raw NLL. This may be useful, but it is more complex and could introduce
   dependence on standalone comparison results.

The tentative default is the uniform mean validation NLL unless a clear
deployment distribution justifies different weights.

Per-block adaptive configurations introduce an additional problem because the
space of patterns may be combinatorial. Their shared-checkpoint objective would
need a fixed evaluation suite or a deterministic expectation over a predefined
pattern distribution rather than opportunistically selected patterns.

## Proposed reporting layers

For clarity, keep three distinct elastic summaries:

1. Shared selected checkpoint -- primary

   Report every granularity at the single checkpoint selected by the aggregate
   validation objective.

2. Per-granularity oracle -- supplementary

   Report the independently best observed loss for each granularity, explicitly
   labeled as an oracle envelope assembled from different checkpoints.

3. Final checkpoint -- supplementary

   Report all granularities at the completion checkpoint to expose endpoint
   behavior and late-training degradation.

For standalone runs:

1. Best checkpoint within budget -- proposed primary.
2. Final checkpoint -- supplementary.
3. Trailing-window statistics -- supplementary stability summary.

This asymmetry reflects deployment reality: standalone training creates
separate models that can each use their own selected checkpoint, while elastic
training creates one shared model that should normally use one shared
checkpoint.

## Proposed comparison and plotting interpretation

The primary elasticity gap for granularity `g` would be:

```text
elasticity_gap[g] =
    elastic_validation_loss[g, shared_selected_step]
    - standalone_best_validation_loss[g]
```

This compares the best independently trained model at that width with the
corresponding submodel from one operational elastic checkpoint. The gap then
includes the cost of requiring all granularities to coexist in one model and
one checkpoint.

Potential plotting convention:

- Standalone markers use `best_validation_loss`.
- Primary elastic curves use all granularities from the shared aggregate-selected
  checkpoint.
- A faint or dashed elastic oracle curve shows per-granularity minima.
- Final-checkpoint values and trailing statistics remain in tables or secondary
  plots.
- Axis labels should say exactly which metric is shown, for example "Best
  validation loss within budget" or "Validation loss at shared selected
  checkpoint," rather than the ambiguous label "Loss."

## Open questions

- Which aggregate objective and weights should select the shared elastic
  checkpoint?
- Should the aggregate selection rule match the training sampling distribution
  or a deployment distribution?
- How should per-block adaptive patterns be represented by a fixed evaluation
  suite?
- Should downstream benchmarks run on the shared selected elastic checkpoint,
  on final checkpoints, or on both?
- Should `checkpoint_path` in `scaling_results.csv` be split into explicit
  final and best/shared-selected checkpoint fields to avoid mixing a final loss
  with a best-checkpoint path?
- Which figure names, legends, and table columns need to change if best
  standalone validation becomes primary?

No decision in this note changes the existing outputs. The current pipeline
continues to use final validation loss as the primary size-plot value until the
selection and reporting protocol is explicitly revised.
