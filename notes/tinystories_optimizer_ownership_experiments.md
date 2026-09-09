# TinyStories-Instruct: slicing, concatenation, and optimizer ownership

Date: 2026-09-09

Status: experiment design and input for a future Spec Kit feature. This note
records the agreed series and its technical motivation. It does not imply that
all arms or the reporting workflow are implemented, and it does not authorize
launching experiments. Existing feature specifications remain authoritative
for current runtime behavior.

## 1. Objective and agreed scope

Understand how AdamW history sharing, FFN parameter representation, and global
versus blockwise gradient clipping affect learning in a nested MatFormer.
Measure optimization outcomes as well as memory and execution costs; this is
not solely a memory experiment.

The agreed series contains nine fresh training runs:

- Four independent standalone models, one per width.
- Two nested slicing settings, S1 and S2.
- Three nested concatenation settings, C1, C2, and C3.

Use seed **42 only**. Each standalone receives **one pass over the prepared
optimizer-training dataset**; each elastic setting (S1, S2, C1, C2, C3) receives
**four passes over that same dataset**. Here "pass" means an epoch, not a
minibatch. Each elastic run therefore matches the aggregate token budget of
the four standalone references combined.

All nested settings select one global width per update, independently and
uniformly with replacement. There is no balanced cycle, enforced equal
exposure, or adaptive width selection in this series.

The four widths are `g250`, `g500`, `g750`, and `g1000`, corresponding to 25%,
50%, 75%, and 100% of the full FFN intermediate width. These percentages do not
describe the fraction of total model parameters: attention, embeddings, and
other common components remain present at every width.

## 2. Experiment matrix

### Slicing combinations

| ID | Representation | AdamW ownership | Principal behavior |
| --- | --- | --- | --- |
| S1 | Slicing | One optimizer over the shared model | Active prefixes share histories across widths; unused tensor tails can receive residual-momentum updates and weight decay. |
| S2 | Slicing | One optimizer per width over the same full parameter tensors | Each width retains its own histories; unused tails have zero moments for that width but still receive weight decay. |

S2 uses the existing full-sized moment allocation. Compact moment storage for
slicing was discussed as a possible implementation variant; it is not an
additional arm in this nine-run series. It must not silently replace S2 or
change its unused-tail decay behavior.

### Concatenation combinations

| ID | Representation | AdamW ownership | Principal behavior |
| --- | --- | --- | --- |
| C1 | Concatenation | One optimizer over the shared model | Each physical parameter block has one history shared among widths that activate it. Inactive blocks are skipped. |
| C2 | Concatenation | One optimizer per width over the same block parameters | Each width has independent histories for its active blocks; unused blocks acquire no state in that width's optimizer. |
| C3 | Concatenation | One optimizer per FFN block ownership group, plus an owner for common parameters | Clip each active ownership group independently before stepping its optimizer; histories remain shared across widths that use each block. |

### Standalone references

| ID | Trained width | Training mode |
| --- | --- | --- |
| ST-g250 | g250 | Independent fixed-width dense model, one AdamW optimizer |
| ST-g500 | g500 | Independent fixed-width dense model, one AdamW optimizer |
| ST-g750 | g750 | Independent fixed-width dense model, one AdamW optimizer |
| ST-g1000 | g1000 | Independent fixed-width dense model, one AdamW optimizer |

Each standalone model trains its width on every update and owns its own model
weights. The four standalone runs are not a jointly trained model or a sixth
nested setting. Their matching attention, normalization, and FFN dimensions
must equal the corresponding active nested subnetwork.

## 3. AdamW semantics being compared

For a parameter tensor that has a gradient, AdamW maintains first and second
moments and a parameter-local update counter:

```text
m <- beta1 * m + (1 - beta1) * gradient
v <- beta2 * v + (1 - beta2) * gradient^2
parameter <- (1 - learning_rate * weight_decay) * parameter
             - learning_rate * bias_corrected_m
               / (sqrt(bias_corrected_v) + epsilon)
```

Weight decay is decoupled: it does not enter these gradient moment estimates.
Adam counters belong to parameter tensors within an optimizer, not individual
tensor entries. A zero-valued gradient is different from `grad=None`.

### S1: shared AdamW with slicing

The forward pass selects rows of the gate/up projections and columns of the
down projection from full-sized parameter tensors. Backward produces full-sized
gradients with zero entries outside the selected prefix. AdamW processes those
full tensors.

On overlapping prefixes, the moments mix gradients from different widths.
Every selected width advances the shared FFN tensors' counters, including when
some coordinates have zero gradient. A tail coordinate can retain nonzero
moments from a previous wider step and move during a later narrow step through
both residual momentum and decoupled decay.

### S2: width-owned AdamW with slicing

Each width's optimizer sees gradients only from that width, while all optimizers
modify the same model tensors. An optimizer's counters advance only when its
width is selected. Its parameter-local states remain frozen between selections,
although other widths continue to change the shared weights.

For these fixed-width, fresh runs, an optimizer never receives a nonzero
gradient outside its own FFN prefix. Those tail moments therefore remain zero.
However, its full tensor still has a gradient, so ordinary AdamW decays the
unused tail when that optimizer steps. Preserve this behavior in S2.

Width-owned state also applies to common parameters such as attention, input
embeddings, and the output head; it is not restricted to FFN weights.

### C1 and C2: concatenated parameter blocks

For these four fractions, each transformer layer has four FFN segments:

| Selected width | Segments used in every layer |
| --- | --- |
| g250 | A |
| g500 | A, B |
| g750 | A, B, C |
| g1000 | A, B, C, D |

Each segment consists of separate gate, up, and down parameter tensors. The
forward pass concatenates the required segments. With gradients cleared using
`set_to_none=True`, excluded segments have `grad=None`; AdamW skips their
weights, decay, moments, and counters completely.

In C1, A's history receives gradients from all four widths; B's from g500,
g750, and g1000; C's from g750 and g1000; and D's only from g1000. Their counters
advance according to block activation, even though the optimizer object is
shared.

In C2, each width has its own histories for the blocks it activates. A has four
width-owned histories, B has three, C has two, and D has one. A non-selected
optimizer's moments remain frozen. The selected optimizer does not create state
for blocks outside its width, even if a different optimizer previously used
those blocks. Common non-FFN parameters still have one history per width.

### C3: block-owned AdamW with separate clipping per block

Retain the five-optimizer grouping: group the A segments from all transformer
layers into optimizer A, and similarly for B, C, and D. Each group owns the
corresponding gate/up/down tensors and any segment-specific biases. This gives
the four FFN optimizers for g1000. It does not mean assigning all FFN weights
from a transformer layer to a width or creating separate model replicas.

A fifth optimizer owns all remaining parameters, including attention,
normalization, embeddings, the output head, and any common FFN output bias.
Every parameter must have exactly one owner. A g1000 update calls all four FFN
optimizers and the common-parameter optimizer; a g250 update calls optimizer A
and the common-parameter optimizer.

| Selected width | Independent clipping groups and optimizers that step |
| --- | --- |
| g250 | Common, A |
| g500 | Common, A, B |
| g750 | Common, A, B, C |
| g1000 | Common, A, B, C, D |

Use one forward/backward pass through the selected width. Then calculate the
L2 gradient norm independently within each active ownership group and clip that
group before its AdamW update. Here a "block" means one of the A/B/C/D segment
groups across all layers, not an individual projection tensor or an entire
transformer layer. The common parameters form their own fifth clipping group.
Inactive groups remain untouched with absent gradients.

Working threshold default: carry over the existing value **1.0 per active
group**. Separate clipping is the agreed intervention; this numerical threshold
is an inherited default to make the proposal concrete, not a separately tuned
or user-selected value. Record the explicit threshold for every group in the
resolved configuration.

For a group with gradient vector `g` and threshold `c`, apply the same scaling
factor to all gradients within that group:

```text
group_norm = ||g||_2
group_clip_coefficient = min(1, c / (group_norm + epsilon_clip))
g <- group_clip_coefficient * g
```

Use the existing clipping primitive and its numerical stabilization convention.
All clipping operates on gradients before they enter AdamW's moments. Do not
clip each scalar independently or apply a second global clipping pass afterward.
After clipping, step the active owners using identical AdamW hyperparameters and
the same current global learning rate. Advance the global scheduler once after
the complete training update, not once per optimizer call. There are no
block-local learning-rate schedules or additional gradient corrections.

Dividing parameters among optimizer objects alone would reproduce C1
mathematically; the distinct
intervention is now the clipping rule. A large gradient in one group no longer
forces the other groups' gradients to be rescaled. Histories still mix gradients
from all widths that activate a block, unlike C2's separate per-width histories.

The clipping budgets are intentionally different under the working default.
C1 clips the entire gradient vector to 1.0. In C3, `n` disjoint active groups
each capped at 1.0 can have a combined norm up to `sqrt(n)`: `sqrt(2)` for g250
and `sqrt(5)` for g1000. Thus C1 versus C3 tests global versus independent group
clipping at a common nominal threshold, not a matched total gradient-norm
budget. This bound concerns gradients, not AdamW's resulting parameter-update
norm. Log the realized scaling so outcomes can be interpreted accordingly.

## 4. Budget, sampling, and matched controls

### One pass per standalone; four passes per elastic setting

The inherited TinyStories-Instruct recipe uses the aligned optimizer-training
budget `B = 713,785,344` tokens. With 64 sequences per update and context length
128, one pass is `87,132` global updates. Apply these budgets without early
stopping:

| Runs | Passes per run | Tokens per run | Global updates per run |
| --- | --- | --- | --- |
| Each of ST-g250, ST-g500, ST-g750, ST-g1000 | 1 | 713,785,344 (`B`) | 87,132 |
| Each of S1, S2, C1, C2, C3 | 4 | 2,855,141,376 (`4B`) | 348,528 |
| Four standalone references combined | 4 in aggregate | 2,855,141,376 (`4B`) | 348,528 in aggregate |

The nine-run campaign uses `24B` training tokens in total: `4B` for the
standalone panel and `5 * 4B` for the elastic settings. The equality of interest
is one elastic run versus the complete standalone panel, not one elastic run
versus one standalone run.

Before implementation freezes these numbers, verify them against the actual
prepared optimizer-role manifest and packed-data iteration contract. A pass
means visiting every designated packed optimizer-training example once in its
deterministic epoch order; it excludes validation, controller, and final-holdout
data. Standalones stop after the first epoch. Elastic runs use the existing
`repeat_epochs` / `deterministic_per_epoch` iteration for four complete epochs,
with the same epoch-specific orders across elastic arms. Record any packing or
alignment truncation explicitly. Repetition is intentional across the four
elastic epochs, not an arbitrary adjustment to reach a stale token count.

For nested runs, each update draws one of the four widths with probability
0.25 and uses it globally across all transformer layers. Set the existing
global sampling interval to one and use `random_with_replacement`. Share the
same dedicated action RNG stream and resulting action sequence across S1, S2,
C1, C2, and C3. Keep this action stream continuous across all four epochs. Each
width has 87,132 expected selected updates across the elastic run, corresponding
to `B` expected selected-width training tokens. Actual counts must be measured
and are not required to equal that expectation.

This matches each standalone's selected-width training budget in expectation,
not its exact samples or realized exposure. A width is not guaranteed to see
each example exactly once: random selection across four epochs can repeat some
examples for that width and omit others. Smaller FFN supports additionally
participate in wider subnetworks. For concatenation the expected A/B/C/D block
activation counts are 348,528 / 261,396 / 174,264 / 87,132 respectively; the
common parameters are active on every elastic update. Report selected-width
exposure and block activation separately. Token-budget equality to the complete
standalone panel does not establish equal wall time, memory, or measured compute.

### Inherited recipe defaults

Unless subsequently revised, carry forward these controls from the Feature 12
recipe, rather than tune them separately per arm:

| Control | Value |
| --- | --- |
| Dataset | Prepared TinyStories-Instruct, fixed disjoint data roles |
| Seed / data seed | 42 / 42 |
| Hidden dimension / layers / attention heads | 64 / 4 / 4 |
| Context length / vocabulary | 128 / 2,048 |
| Width fractions | 0.25, 0.50, 0.75, 1.00 |
| AdamW learning rate | 0.008, no world-size scaling |
| AdamW betas / epsilon / weight decay | (0.9, 0.95) / 1e-8 / 0.1 |
| Schedule | Global cosine over each run's full budget: 87,132 updates for standalone, 348,528 for elastic; 64 global warmup updates |
| Batch / accumulation | 64 sequences / 1 microbatch per update |
| Gradient clipping: standalone, S1, S2, C1, C2 | Global norm 1.0, once before stepping any owner |
| Gradient clipping: C3 | Independent L2 norm clipping for each active A/B/C/D/common group; working default 1.0 per group |
| Precision / execution | bf16, one process on one GPU |
| Correction | None; no membership correction or block-specific LR correction |
| Pre-nested sampling warmup | Disabled |
| Ordinary validation | Every 64 updates and at completion |

All runs share the tokenizer, first-epoch training sequence order, batch
boundaries, evaluation samples, and evaluation cadence. The five elastic arms
also share the remaining three epoch orders and the complete global
learning-rate trace. The four standalone models share their shorter trace.
With cosine decay resolved over different horizons, standalone and elastic
learning rates need not match at the same absolute update number. Record both
scheduler contracts explicitly. Do not reset the scheduler or optimizer state
at epoch boundaries or stop learning-rate decay at the end of the first elastic
epoch.

Losses are ordinary causal language-model losses without width-dependent
weighting. All runs start fresh with their declared full scheduler horizon;
do not continue an earlier one-pass elastic checkpoint into this four-pass
comparison under a changed schedule contract.

Proposed initialization matching rule: create one canonical untrained dense
initialization under seed 42. Map those values into each nested representation
and extract matching untrained prefixes for the standalone models, which then
train independently. Verify actual tensor equality after mapping; merely
passing the same seed to constructors of different shapes is insufficient.
Record this rule as an explicit implementation choice before launching.

## 5. Required comparisons and interpretation

| Comparison | Question |
| --- | --- |
| S1 vs S2 | What does isolating width histories change when all FFN optimizers operate on full sliced tensors? |
| C1 vs C2 | What does isolating width histories change when unused FFN blocks are skipped entirely? |
| S1 vs C1 | How does splitting FFN parameter tensors change shared-AdamW behavior, including block ages, inactive momentum, and decay? |
| S2 vs C2 | How does representation change width-owned AdamW behavior, especially unused-tail decay and state allocation? |
| C1 vs C3 | What changes when clipping is independent per active block/common group instead of global, including the different total gradient-norm budget? |
| Every nested setting vs the four matching standalones | Can one elastic model trained for `4B` match the independently trained widths, each trained for `B`, at the same aggregate token budget as the standalone panel? |

Representation comparisons change optimizer semantics as well as storage.
Do not describe S1/C1 or S2/C2 as pure memory ablations. C1/C3 is a clipping
comparison, not a test of independent width histories or an expected endpoint
equivalence. Separate histories also do not remove interactions through shared
weights. With one seed, all outcome claims are descriptive; there are no
across-seed confidence intervals or claims of general superiority.

## 6. Evaluation and checkpoint endpoints

The combined comparisons use each run's **terminal checkpoint after its full
assigned budget**: four passes (`4B`) for each elastic setting and one pass
(`B`) for each standalone. Evaluate each nested terminal model at all four
widths and each standalone terminal model at its own width. Do not substitute
independently selected best-validation checkpoints or first-epoch elastic
checkpoints. Keep intermediate and best-validation metrics available as
separately labeled individual-run diagnostics.

The evaluation split was not chosen explicitly in the discussion. Proposed
default: use the same fixed ordinary-validation split for both combined plots,
leaving the final holdout sealed. If final-holdout comparisons are subsequently
requested, freeze the nine terminal checkpoint identities first and explicitly
evaluate all 24 setting/width endpoints on the same fixed holdout. Never mix
validation and holdout endpoints in one comparison or implicitly evaluate the
holdout while plotting.

Use the existing target-token-weighted causal language-model loss aggregation.
Perplexity is `exp(aggregated_loss)`, not an average of per-batch perplexities.
Persist the split, manifest hash, evaluation target-token count, aggregation
convention, and checkpoint hash with every endpoint.

## 7. Artifacts and individual-run diagnostics

Reuse existing structured outputs and extend them where the new ownership
setting requires additional fields. Save:

- Resolved configuration, experiment/arm ID, representation, ownership mode,
  seed streams, initialization mapping/hash, ordered widths, runtime versions,
  hardware, precision, and optimizer/scheduler contracts.
- Clipping scope, exact group membership, and per-group thresholds. Include
  this contract in run identity and checkpoint compatibility checks.
- Dataset, tokenizer, packed-order, and evaluation-role identities and hashes.
- Per-update selected width, action ID, batch provenance, training loss,
  learning rate used, cumulative training tokens, attempted/committed status,
  epoch index, assigned budget/pass count, and global scheduler position.
- Realized width-selection counts and, for concatenation, block-activation
  counts. Identify optimizer calls separately from global training updates in
  C3; several calls still constitute one training update.
- For C1 and C3, record compact per-update A/B/C/D/common gradient norms before
  and after clipping, the applied coefficients, and active-group flags, plus
  the combined gradient norm. In C1 the applied coefficient is global; in C3
  it is group-specific. Summarize clipping frequency by group and selected
  width, distinguishing inactive groups from active groups with zero gradients.
- Per-width validation loss/perplexity over training, active total and
  non-embedding parameter counts, and terminal checkpoint identity.
- Compact optimizer counter summaries by owner and parameter component, plus
  actual allocated state bytes by owner/component. Keep full moments in
  resumable checkpoints rather than duplicating them in ordinary metric rows.
- Training wall time, throughput, peak allocated accelerator memory, and
  checkpoint bytes. Accumulate elapsed time and peak measurements across
  continuation attempts so resumed runs retain complete resource accounting.
- A complete terminal resumable checkpoint and summary that reconcile global
  steps, tokens, selected widths, block activations, and scheduler position.

Individual-run plots should retain training/validation trajectories over
training tokens, per-width evaluation curves for nested runs, sampling exposure,
clipping norms/coefficients and frequency for C1/C3, and optimizer-state/resource
diagnostics. These complement the two combined
endpoint plots; they do not replace them.

## 8. Combined plots: exact agreed presentation

Produce **two separate comparison figures**, both containing all settings:

1. Active non-embedding parameter count on the x-axis; perplexity on the y-axis.
2. Active non-embedding parameter count on the x-axis; loss on the y-axis.

Use the repository's `non_embedding_parameters` convention from
`src/utils/model_size.py`: active total parameters minus input-embedding and
LM-output-head parameters. Attention, normalization, and active FFN parameters
remain included. This is an inference-subnetwork count, not optimizer-state
memory or the full stored nested model size. Calculate actual counts rather
than multiplying the full model count by the width fraction.

For **each of S1, S2, C1, C2, and C3**, plot four markers and connect them into
one curve, ordered by active parameter count. Keep setting colors/styles
consistent between the loss and perplexity figures. Label C1 as shared AdamW
with global clipping and C3 as block-owned AdamW with separate group clipping;
preserve them as distinct series even if their points overlap.

For **standalone**, plot four individual markers with **no connecting line**.
They represent four independently trained models. Use a recognizable standalone
marker style and label the widths without depicting a jointly trained curve.

Each figure therefore contains five nested curves and four standalone points:
24 measured endpoints in total. Use all endpoints from the same evaluation
split and the appropriate terminal checkpoints. Label the dataset, seed 42,
evaluation split, and checkpoint rule. State **elastic: four passes (`4B`) per
curve; standalone: one pass (`B`) per point, `4B` for the complete panel**.
Do not label the points as equal per-run training budgets. Do not put training
tokens on the x-axis of these combined plots, fit artificial points, or add
across-seed uncertainty bands.

Suggested output names:

- `optimizer_ownership_perplexity_vs_non_embedding_parameters.png`
- `optimizer_ownership_loss_vs_non_embedding_parameters.png`
- Matching PDF exports for sharing.
- `optimizer_ownership_endpoints.csv` and `.json`, containing all 24 endpoints,
  the exact x/y values, representation/ownership and clipping identifiers,
  provenance, assigned token budgets/pass counts, realized exposure, and resource
  measurements.

A missing or invalid endpoint must not silently produce a figure advertised as
the complete nine-run comparison. An explicitly labeled partial diagnostic may
be provided separately. Reject mixed splits, wrong budgets, duplicate/missing
arms, non-finite values, and incompatible parameter-count conventions.

## 9. Memory expectations to measure

Let `F` be the number of full-width FFN weight scalars and `R` the number of
remaining model scalars used by every width. For ordinary AdamW without AMSGrad,
the following counts describe persistent first/second moment elements after all
widths have been exercised; scalar counters and temporary buffers are excluded:

| Setting | Moment elements |
| --- | --- |
| S1 | `2 * (F + R)` |
| S2 | `2 * (4F + 4R)` |
| C1 | `2 * (F + R)` |
| C2 | `2 * (2.5F + 4R)` |
| C3 | `2 * (F + R)` |

Thus C2 saves 37.5% of S2's FFN moment storage, not 37.5% of total training
memory. Common-component histories remain replicated per width. Convert
elements to bytes using the actual state dtype; bf16 computation does not by
itself establish the optimizer-state dtype. Concatenation introduces temporary
assembled tensors, so measure peak memory and wall time rather than infer them
from persistent moment counts. Plot parameter counts separately from memory.

## 10. Implementation and verification boundaries

Feature 12 provides shared/per-width ownership, but its frozen recipe and
analyzer assume a different six-run, three-seed balanced-cycle protocol. This
series needs explicit nine-run identities and reporting support. C3 also needs
a block-owned runtime, separate group clipping, and a checkpoint schema that
records ownership and clipping contracts; do not present an invented
configuration value as an already supported option.

Before executing the campaign, verify the actual slicing and concatenation
modules, not only toy independent scalar parameters:

- S1 preserves wider-gradient momentum and applies decay on unused slices.
- S2 has independent width histories and zero unused-tail moments while
  retaining ordinary full-tensor decay.
- C1/C2/C3 leave inactive blocks' weights, moments, and counters unchanged.
- C2 allocates state only for each width's active blocks, with no stale gradient
  leakage or eager full-state allocation during checkpoint restoration.
- As a short implementation check, the C3 optimizer partition with clipping
  temporarily configured to match C1 globally agrees with C1 under matched
  initialization, data, actions, learning rates, and update order. This check
  is not another full-budget experiment.
- In the actual C3 arm, verify independent coefficients and bounded post-clip
  norms for each active group, with no additional global rescaling. A large
  gradient in one group must not change another group's clipping coefficient
  when that other group's input gradients and threshold are fixed. Verify
  common parameters are clipped and updated exactly once. Do not require
  C1/C3 parameter trajectories or final metrics to agree under different clipping.
- Across all nested arms, selected actions and batches match exactly; width
  frequencies are measured rather than forced to balance.
- Standalone subnetworks match the intended dimensions and active parameter
  counts; each standalone completes exactly one pass and each elastic arm
  completes exactly four passes. Check data-cursor and scheduler continuity
  through epoch boundaries, with no reset or accidental fifth epoch.
- Resume restores owner identity, all required moments/counters, RNG, data
  cursor, and the single global clock. Reject malformed or cross-arm state.
- If C3 fails after any owner has updated parameters, stop and recover from
  the previous durable checkpoint; do not save a partially updated model as a
  completed, resumable global step.
- The endpoint table contains exactly 24 valid rows, and both combined figures
  show five nested curves with disconnected standalone markers.

Implementation must compare full resolved scientific controls, including model
shape, representation, correction, clipping, data identities, and initialization.
Do not assume the current paired-control hash alone proves that only the intended
arm differences exist.

Before a future specification is frozen, settle the proposed initialization
mapping, the inherited C3 threshold choice, and the evaluation split. The
agreed nine arms, seed, uniform sampling, one-pass standalone/four-pass elastic
budgets, terminal checkpoint rule, C3's separate clipping per ownership group,
and standalone-marker presentation should be preserved.

## Repository references

- [Feature 12 plan](../specs/012-per-width-optimizer-state/plan.md)
- [Existing TinyStories-Instruct recipe](../configs/controlled_exps/tinystories_instruct_per_width_optimizers.yaml)
- [Slicing and concatenation FFNs](../src/models/ffn.py)
- [Current per-width optimizer collection](../src/training/optimizer_state.py)
- [Training update lifecycle](../src/training/steps.py)
- [Parameter-count convention](../src/utils/model_size.py)
- [Validation aggregation](../src/evaluation/validation.py)

The previously located eight standalone FineWeb runs are historical results
from a different dataset and model. They do not replace the four matched
TinyStories-Instruct standalone references in this series.
