# Spec Kit prompt: TinyStories optimizer ownership comparison

Date: 2026-09-09

Status: proposed feature input, not an implemented specification or permission
to launch training. Companion design:
[experiment technicalities and comparisons](tinystories_optimizer_ownership_experiments.md).
This is a new experiment series extending Feature 12; it does not replace the
historical Feature 12 prompt or retrospectively change existing runs.

Suggested invocation:

```text
/speckit-specify Use notes/tinystories_optimizer_ownership_speckit_prompt.md as the feature description. The feature request begins at "Feature request". Preserve the agreed experiment matrix and budget correction; flag the explicitly proposed defaults for clarification before freezing the specification.
```

## Feature request

Specify support for a controlled TinyStories-Instruct experiment series that
separates the effects of FFN parameter representation, AdamW state ownership,
and gradient clipping in nested MatFormer training. Measure learning outcomes
as well as optimizer memory and execution costs. This is not merely a memory
optimization feature.

Read the companion design and `specs/012-per-width-optimizer-state/plan.md` for
context. Inspect existing model, optimizer, training, evaluation, and reporting
paths before deciding what needs extending. Reuse the existing trainer and
artifact conventions. Feature 12's six-run, three-seed, balanced-cycle recipe
and analyzer are not the protocol for this new series.

### 1. Fixed experiment matrix

Use seed **42 only**, with nine fresh training runs:

| ID | Model representation | AdamW ownership | Clipping | Dataset passes |
| --- | --- | --- | --- | --- |
| ST-g250 | Independent dense 25% FFN width | One optimizer | Global norm | 1 |
| ST-g500 | Independent dense 50% FFN width | One optimizer | Global norm | 1 |
| ST-g750 | Independent dense 75% FFN width | One optimizer | Global norm | 1 |
| ST-g1000 | Independent dense 100% FFN width | One optimizer | Global norm | 1 |
| S1 | Nested slicing | One shared optimizer | Global norm | 4 |
| S2 | Nested slicing | One optimizer per width | Global norm | 4 |
| C1 | Nested concatenation | One shared optimizer | Global norm | 4 |
| C2 | Nested concatenation | One optimizer per width | Global norm | 4 |
| C3 | Nested concatenation | One optimizer per block group, plus common owner | Separate norm per active ownership group | 4 |

Widths `g250`, `g500`, `g750`, and `g1000` mean 25%, 50%, 75%, and 100%
of the full FFN intermediate dimension, not percentages of total parameters.
Attention and other common dimensions remain fixed. Standalone models have
their own independently trained weights and physically appropriate dense FFNs;
they are not checkpoints extracted from an already trained nested model.

All elastic runs use one global width for all layers per complete optimizer
update (`H=1`), sampled independently with replacement from probabilities
`[0.25, 0.25, 0.25, 0.25]`. Reuse the same action RNG stream and exact sampled
width sequence across S1/S2/C1/C2/C3. Do not use round-robin, balanced cycles,
forced equal counts, adaptive sampling, or multiple width losses per update.

### 2. Budget and data contract

The latest budget decision supersedes the earlier one-pass-for-every-run
proposal. A "pass" means a complete epoch over optimizer-training data, not a
minibatch. Each elastic run gets four passes, matching the aggregate token
budget of the four standalone runs combined; each standalone gets one pass.

For the existing prepared corpus and inherited batch recipe:

| Quantity | Each standalone | Each elastic setting |
| --- | --- | --- |
| Passes | 1 | 4 |
| Optimizer-training tokens | 713,785,344 | 2,855,141,376 |
| Tokens per update | 8,192 | 8,192 |
| Updates | 87,132 | 348,528 |

Audit the actual prepared manifest and packing alignment before freezing these
counts; reject discrepancies rather than silently truncating a pass or using a
stale token budget. Each pass visits the designated packed training examples
once. Preserve the existing disjoint training, ordinary-validation, controller,
and sealed-final-holdout roles. Never train on evaluation roles.

Use identical batch sequences and deterministic per-epoch data orders across
the elastic arms, with the same first epoch for the standalones. Preserve RNG,
data cursor, and scheduler continuity across all four epochs; no resets at an
epoch boundary. Use fresh identities, not extensions of old one-pass elastic
runs with a changed scheduler horizon.

Each width has 87,132 expected selections in an elastic run, but actual counts
are random and must be reported, not corrected. Expected token exposure per
width equals one pass in volume, not guaranteed coverage of every example:
individual examples may be selected repeatedly or never for a given width.
Smaller shared FFN blocks also participate in wider selections.

This is matched aggregate training-token budget, not matched FLOPs, wall time,
or per-run budget. The whole nine-run campaign uses 24 single-pass budgets.

### 3. AdamW and representation semantics

Keep shared model weights in every elastic arm. Multiple optimizers own
different state histories, not replicas of model weights. Use ordinary AdamW
semantics, including the distinction between a present zero gradient and
`grad=None`; do not silently introduce a different optimizer algorithm.

- **S1:** Slice full-sized FFN parameter tensors in the forward pass. Backward
  produces full-shaped gradients with zeros outside the selected prefix.
  Shared moments mix gradients from different widths. Unused tensor tails can
  still change through residual momentum and decoupled weight decay, and the
  parameter tensor's Adam counter advances.
- **S2:** Maintain four independent AdamW histories over the same full-sized
  model tensors, including common parameters. Step only the selected width's
  optimizer. Its never-used FFN tails retain zero moments in a fresh run, but
  ordinary full-tensor weight decay still applies on its steps. Retain the
  full-sized moment allocation; compact slicing state is not this arm.
- **C1:** Represent FFN quarters as independent trainable tensors, concatenating
  only active quarters for computation. Each physical parameter has one shared
  history. Inactive blocks must have `grad=None` and receive no update, decay,
  moment change, or counter increment.
- **C2:** Use four width-owned AdamW instances over shared block parameters and
  common parameters. State is allocated lazily only where that width produces
  gradients. A/B/C/D have respectively 4/3/2/1 width-specific histories after
  all widths have been selected; common parameters have four histories. Do not
  eagerly materialize inactive state during construction or resume.
- **C3:** Use the ownership and clipping contract below. Its state partition
  alone does not isolate histories by width; block histories remain shared
  across every width that activates them.

Clear gradients with absent-gradient semantics between updates so inactive
blocks cannot inherit stale gradients. A present zero gradient on an active
parameter must retain ordinary AdamW behavior.

### 4. C3 ownership and separate clipping

Define exactly five disjoint ownership groups covering all trainable parameters:

- `O-A`: first FFN quarter across all transformer layers.
- `O-B`: second FFN quarter across all transformer layers.
- `O-C`: third FFN quarter across all transformer layers.
- `O-D`: fourth FFN quarter across all transformer layers.
- `O-common`: attention, norms, input embeddings, LM output head, and any
  common FFN output bias. Deduplicate tied parameters if applicable.

Each FFN group includes its gate/up/down parameter blocks and any
segment-specific biases. These are four cross-layer FFN groups, not four
optimizers for every transformer layer. The fifth optimizer owns common
parameters; it is not another width or a duplicate optimizer over FFN blocks.

| Selected width | Active owners |
| --- | --- |
| g250 | O-common, O-A |
| g500 | O-common, O-A, O-B |
| g750 | O-common, O-A, O-B, O-C |
| g1000 | O-common, O-A, O-B, O-C, O-D |

After one forward/backward, independently clip each active group's joint L2
gradient norm, then step each active optimizer exactly once. Common parameters
are clipped and updated once. No second global clipping operation is allowed.
Inactive owners do not advance. Advance one global learning-rate scheduler
exactly once after the complete training update, not once per optimizer call.
All owners use the current global learning rate; no block-local schedules.

The inherited global threshold is `1.0`. The proposed C3 default is also `1.0`
**per active group**, pending explicit confirmation. Norm clipping scales all
gradients in a group together when their combined norm exceeds the threshold;
it does not clamp individual gradient elements above 1.

C3 intentionally changes clipping relative to C1. With threshold 1 for each of
N active disjoint groups, the combined norm can reach sqrt(N), unlike C1's
global cap of 1. For g250/g1000 these bounds are sqrt(2)/sqrt(5). Document this
difference; do not claim equal effective clipping budgets or pure history
isolation. There is no extra full-budget C3-equivalence arm.

### 5. Matched recipe and explicit proposed defaults

Inherit the controlled TinyStories-Instruct recipe where not changed above:

- Model dimension 64, four layers, four attention heads, context length 128,
  vocabulary size 2,048, initializer standard deviation 0.02; resolve the full
  FFN dimension and exact quarter dimensions from the model configuration.
- AdamW learning rate 0.008, betas (0.9, 0.95), epsilon 1e-8, weight decay 0.1.
- Batch size 64 sequences, accumulation 1, bf16, one process on one GPU,
  no learning-rate world-size scaling.
- Cosine schedule with 64 warmup updates. Set each run's schedule horizon to
  its own complete budget: 87,132 standalone or 348,528 elastic updates.
  Elastic LR traces must match each other. Standalone and elastic cosine
  traces need not match at equal absolute steps because their horizons differ.
- Same tokenizer, corpus manifests, data seed 42, and evaluation policy.
  Disable membership/LMC/GMC correction and pre-nested width warmup.
- Ordinary validation every 64 updates and at the terminal budget.

The following are **proposed defaults, not previously settled user choices**.
Surface them during specification clarification before freezing the protocol:

1. Initialize a canonical untrained dense model with seed 42, map exactly the
   same initial values to slicing and concatenation representations, and
   extract matching untrained prefixes for independent standalone training.
   Merely reseeding differently shaped constructors is not proof of matching.
2. Use C3 threshold 1.0 independently for each active ownership group.
3. Use ordinary validation for the combined terminal endpoint plots, keeping
   final holdout sealed. If final holdout is explicitly selected instead,
   freeze all nine terminal checkpoints first, evaluate all 24 endpoints under
   one common protocol, and never mix validation and holdout endpoints.

### 6. Metrics, artifacts, and required figures

Retain per-run resolved configurations, ordinary metric CSVs, summaries,
resumable checkpoints, individual loss/perplexity trajectories, and resource
plots. Identify representation, ownership, clipping policy, run/seed, data and
initialization provenance, selected widths, epochs, actual tokens, update counts,
and global scheduler position. Record width selections and block activations
so random exposure can be audited. Keep full states in checkpoints, not CSVs.

Measure optimizer-state bytes by owner using actual tensor dtypes, peak memory,
throughput, wall time, and checkpoint size. Preserve cumulative runtime and
maximum peak-memory accounting across resumes. Distinguish persistent state
from temporary concatenation buffers. For full FFN parameter count F and common
count R, expected ordinary AdamW moment element totals after exposure are:

- S1/C1/C3: `2 * (F + R)`.
- S2: `2 * (4F + 4R)`.
- C2: `2 * (2.5F + 4R)`.

These estimates exclude counters and temporary storage. C2's theoretical 37.5%
reduction versus S2 concerns FFN moments only, not total training memory.

For C1 and C3, log per-group pre/post-clipping norms, coefficients, active
flags, combined norms, and clipping frequency by width and group. C1 applies
one global coefficient; C3 applies independent coefficients. Diagnostics must
not change the gradients or action stream.

Produce exactly these two **combined endpoint comparisons**, in PNG and PDF:

1. Active non-embedding parameter count on x; perplexity on y.
2. Active non-embedding parameter count on x; language-model loss on y.

Each figure contains five elastic curves (S1/S2/C1/C2/C3), each connecting its
four width endpoints. Show the four standalone results as **isolated markers
without any connecting curve**, because they are independently trained models.
Keep setting colors consistent across both figures. No across-seed error bars
are justified by this one-seed campaign.

Use exact active `non_embedding_parameters` according to
`src/utils/model_size.py`: exclude input embeddings and the LM output head;
include active FFN, attention, norms, and other counted common parameters.
Do not use total stored nested-model size, optimizer-state size, width labels,
or width fraction multiplied by total model size as x coordinates.

All endpoints use the same evaluation data and terminal checkpoint after each
run's assigned budget: one pass standalone, four passes elastic. Do not select
best-validation checkpoints or substitute first-epoch elastic results. Loss is
target-token-weighted causal loss; perplexity is exp(aggregated loss), not an
average of batch perplexities. Annotate dataset, seed, evaluation role, count
definition, and unequal per-run but matched panel-level token budgets.

Export one auditable 24-row endpoint CSV/JSON: 20 elastic endpoints plus four
standalones. Include settings, widths, exact x values, loss/perplexity, evaluated
token counts, training budgets, evaluation identity, and checkpoint hashes.
Suggested artifact stems:

- `optimizer_ownership_perplexity_vs_non_embedding_parameters`
- `optimizer_ownership_loss_vs_non_embedding_parameters`
- `optimizer_ownership_endpoints`

A complete report must validate all nine runs and 24 endpoints. Reject
incomplete/mismatched inputs or explicitly label a separately requested partial
report; never silently present it as complete. Plotting must not automatically
open the sealed holdout.

### 7. Comparison questions and interpretation

- S1 versus S2: width-specific versus shared histories under slicing.
- C1 versus C2: width-specific versus shared histories under concatenation.
- S1 versus C1: representation under shared ownership, including inactive-tail
  momentum/decay and parameter-counter semantics; not a pure storage change.
- S2 versus C2: representation under per-width ownership, including inactive-tail
  decay and moment allocation; not a pure memory-only intervention.
- C1 versus C3: global versus separate group clipping, acknowledging the
  different combined norm cap; block ownership alone is not width isolation.
- Every elastic width versus its matching standalone: quality at the same
  active parameter count, with one four-pass elastic run compared against the
  aggregate budget of the four one-pass standalone references.

Report observed effects descriptively for seed 42. Do not claim multi-seed
robustness, exact equal width exposure, or equal compute without evidence.

### 8. Runtime, resume, and verification requirements

Provide explicit configuration, nine-run preflight, execution instructions,
and a dedicated campaign comparison workflow. Inspect supported configuration
names rather than treating a proposed block-owned optimizer option as already
implemented. Preserve existing default behavior and historical artifacts;
avoid a second trainer or general-purpose optimizer registry.

Version checkpoints with representation, owner/group mapping, clipping policy,
all required moments and parameter-local counters, the one global scheduler,
RNG/action state, and data cursor. Validate shape, identity, required/absent
state, and cross-arm compatibility before mutating a live model on load.
Model-only checkpoints are not exact-resume inputs. Inactive lazy state must
remain distinguishable from missing required state.

Treat C3's collection of active optimizer steps as one logical training update.
If any owner fails after another has updated, abort and recover from the prior
durable checkpoint; never save the partial update as a committed resumable
step. Do not introduce per-step full-state snapshots merely for rollback.

Acceptance tests must exercise actual slicing/concatenation model paths:

1. Correct dense standalone shapes and matching active parameter counts.
2. S1 unused-tail momentum/decay; S2 isolated histories, zero never-used-tail
   moments, and retained full-tensor decay.
3. C1/C2/C3 inactive blocks retain weights, moments, and counters; C2 has only
   the expected lazily allocated width/block histories, including after resume.
4. Disjoint and complete C3 groups; active owners and common owner step once;
   one scheduler advance per complete update. Separate clipping respects each
   group cap without an extra global rescale. Changing one group's input norm
   does not change another group's coefficient when its input is held fixed.
5. A short diagnostic test with C3 temporarily using C1's global clipping
   reproduces C1 within numerical tolerance under matched controls. This tests
   partition plumbing, not expected equivalence of the actual C3 experiment.
6. Identical elastic width/action and batch traces, truly with-replacement
   sampling, measured counts without enforced balancing, and correct epoch
   boundaries: one standalone pass versus four elastic passes.
7. Uninterrupted versus resumed action/data traces and model/optimizer/scheduler
   state agree within the project's determinism tolerances, including around
   epoch boundaries. Malformed and incompatible checkpoints fail safely.
8. Preflight resolves all nine budgets and complete scientific controls; full
   report validation checks those controls rather than trusting an existing
   paired-control hash that may omit new representation or clipping fields.
9. Endpoint export has 24 valid rows, exact active counts, consistent evaluation
   provenance, and the two required figures with disconnected standalone points.

Out of scope: additional seeds, adaptive/nonuniform sampling, forced balancing,
compact slicing moments, per-width model replicas, extra clipping-ablation
training arms, block-local LR schedules, distributed block/per-width execution,
and reuse of unmatched historical FineWeb standalones as these references.
Specification and implementation preparation must not launch the campaign.
