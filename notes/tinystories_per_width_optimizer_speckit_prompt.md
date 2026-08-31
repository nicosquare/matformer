# Spec Kit Prompt: Per-Width Optimizer State

Suggested short name: `per-width-optimizer-state`

Use the following as the feature description passed to `/speckit-specify`.

---

Add an opt-in experimental training mode that evaluates whether a nested
MatFormer benefits from a separate optimizer instance for every resolved global
width. The experiment must compare this mode with the existing single shared
optimizer while leaving model weights shared across widths.

## Research question

The current global-width training path uses one optimizer state across all
sampled widths. Gradients from different widths therefore contribute to the
same optimizer moments on shared parameters. Researchers need to determine
whether independent optimizer histories reduce harmful cross-width interaction
or instead slow useful learning on shared parameters.

The primary hypothesis is that independent width-specific optimizer state can
improve the uniform mean and worst-width TinyStories-Instruct performance at a
fixed optimizer-token budget. The null hypothesis is that it provides no
improvement over the shared-optimizer baseline.

## Required optimizer semantics

Support two explicit and distinguishable state scopes:

- `shared`: preserve the existing behavior in which one optimizer serves every
  sampled width;
- `per_granularity`: maintain one independent optimizer state machine for every
  resolved global granularity.

In the per-granularity mode, all optimizer instances operate on the same shared
model parameters. They do not create separate models or separate parameter
copies. When a global width is selected for an optimizer step:

- exactly that width's optimizer instance performs the update;
- its state advances only for parameters that receive gradients under the
  selected width;
- every non-selected optimizer and all of its moments and counters remain
  unchanged;
- inactive parameters receive no update;
- gradients from the completed step cannot leak into a later step or another
  optimizer instance.

For nested slicing widths, shared prefix parameters consequently have distinct
optimizer histories for each width. Wider-only parameters may be known to every
optimizer instance but must not acquire state or advance under a narrower
forward/backward pass when they receive no gradient.

The feature must work with the repository's supported AdamW and SGD optimizer
families. It must preserve their configured hyperparameters, weight decay,
gradient clipping, mixed-precision behavior, and ordinary update semantics.

## Scheduler and warmup semantics

Use one global optimizer-step scheduler clock in both state scopes. The learning
rate applied at global step t must be the same in the shared and
per-granularity arms when all other inputs match. Every width-specific optimizer
must therefore use the learning rate assigned to the current global step,
whether or not that optimizer was selected on preceding steps.

Only the selected optimizer's internal optimizer state advances. Adam bias-
correction counters remain width-specific because they are optimizer state;
the learning-rate schedule, warmup progress, and decay horizon remain global.
Do not introduce width-local learning-rate schedules in this feature.

The normal configured global warmup remains valid. Pre-nested warmup modes that
apply several widths in one optimizer step are out of scope and must be rejected
with per-granularity optimizer state unless the specification can establish an
unambiguous single-optimizer owner.

## Eligible training modes

Per-granularity optimizer state is valid only when every optimizer step resolves
to exactly one global granularity. It must support deterministic or stochastic
global-width sampling, including the existing uniform global H=1 path and any
adaptive global policy that still selects exactly one width per update.

Reject per-granularity optimizer state for:

- standalone training;
- nested-all training;
- per-block or adaptive-per-block actions;
- any action containing zero or more than one global width;
- fewer than two unique resolved granularities.

The shared state scope must remain the default so old configurations and runs
retain their behavior.

## Configuration and identity

Researchers must be able to select the optimizer state scope from configuration
and dotted command-line overrides without changing source code. Prefer the
user-facing form:

```yaml
training:
  optimizer:
    state_scope: shared  # or per_granularity
    scheduler_clock: global_step
```

If planning identifies a better configuration location, preserve the same two
explicit state scopes and the fixed global scheduler-clock meaning. Unknown
values and incompatible combinations must fail before training.

Resolved configuration, monitoring identity, metrics, checkpoints, run
summaries, and reporting inputs must distinguish shared from per-granularity
optimizer state. Existing artifacts without the field must continue to mean
the historical shared behavior.

## Checkpoint and exact-resume contract

Latest and terminal checkpoints for the new mode must contain:

- the ordered resolved granularity labels;
- one complete optimizer state per granularity;
- the global scheduler state and clock position;
- per-granularity successful optimizer-update counts;
- the last active optimizer identity, if required for exact continuation;
- an explicit versioned optimizer-state-scope identity.

A matching resume inside any width-sampling interval must reproduce the same
subsequent width actions, parameter updates, learning rates, optimizer states,
metrics, and final parameters as an uninterrupted run, subject only to existing
documented reproducibility limits.

Resume must fail rather than silently reinitialize or merge state when the
checkpoint differs in state scope, optimizer family or hyperparameters,
scheduler contract, ordered widths, model topology, or required optimizer
state. A shared-optimizer checkpoint is not a valid per-granularity checkpoint,
and the reverse is also invalid.

Best-evaluation checkpoints that intentionally contain only model state may
remain model-only. Any checkpoint advertised as resumable must preserve the
complete optimizer collection.

## Metrics and auditability

Every ordinary training row must identify the selected global width and the
resolved optimizer state scope. Runs using per-granularity state must expose
compact successful-update counts for every width in structured artifacts.

At completion, verify that:

- the sum of per-width optimizer updates equals the number of committed global
  optimizer steps;
- the update counts agree with the persisted global-width exposure counts;
- non-selected optimizers did not advance;
- all optimizer instances followed the same global scheduler position.

Report checkpoint size, peak accelerator memory, and training wall time so the
cost of additional optimizer state is visible. Do not claim matched compute
unless the reported measurements support that claim.

## TinyStories-Instruct controlled comparison

Provide a dedicated opt-in TinyStories-Instruct experiment recipe and an
operational runbook for a paired comparison with:

- d_model 64, four transformer layers, four attention heads;
- context length 128 and vocabulary size 2048;
- slicing widths g250, g500, g750, and g1000;
- correction mode none;
- global balanced-cycle sampling with H=1;
- AdamW with learning rate 0.008, betas (0.9, 0.95), epsilon 1e-8, and weight
  decay 0.1;
- cosine scheduling and 64 global warmup steps;
- batch size 64, one process, and no gradient accumulation;
- fixed prepared TinyStories-Instruct data roles and strict reproducibility;
- seeds 42, 43, and 44.

The baseline and candidate for a seed must use identical model initialization,
optimizer-training batches, ordinary-validation/controller/final-holdout
manifests, global-width action sequence, number of steps, token budget,
learning rates, and evaluation cadence. The only intended difference is
optimizer state scope.

Use balanced-cycle H=1 so every complete four-step cycle selects every width
exactly once. The one-epoch pilot budget is 713,785,344 optimizer tokens or
87,132 global steps, giving exactly 21,783 updates per width. The optional
three-epoch confirmation budget is 2,141,356,032 optimizer tokens or 261,396
global steps, giving exactly 65,349 updates per width.

The comparison uses terminal fixed-budget checkpoints, not independently
selected best checkpoints. Use ordinary validation for pilot trajectory and
stability decisions. Keep the final holdout sealed through the pilot whenever a
three-epoch confirmation may be run. The untouched final holdout may be opened
for the confirmatory comparison only after the arms, seeds, budgets,
terminal-checkpoint rule, and analysis endpoints are frozen. If researchers
choose to open it for the pilot, any later result on the same holdout is
descriptive rather than a new confirmatory claim.

Primary outcome:

- paired seed-level difference in the uniform mean final-holdout loss across
  g250, g500, g750, and g1000.

Secondary outcomes:

- final-holdout loss and perplexity for every width;
- worst-width final-holdout loss;
- trailing-five ordinary-validation mean per width;
- convergence trajectories at matched optimizer tokens;
- optimizer-update counts, wall time, peak memory, and checkpoint size.

The pilot is diagnostic and must not be described as evidence of general
superiority. A confirmatory claim requires all predefined seeds, the frozen
three-epoch protocol, and a holdout that was not inspected during the pilot.

## Required acceptance scenarios

Cover at least:

- two widths producing different gradients on the same shared parameter and
  retaining independent optimizer moments;
- a narrow-width step leaving wider-only parameters and non-selected optimizer
  state unchanged;
- alternating widths under AdamW and SGD;
- identical global learning rates between shared and per-granularity arms;
- gradient accumulation resolving one optimizer owner for the complete window;
- failure before an optimizer commit leaving all optimizer states unchanged;
- checkpoint/resume inside and at a balanced-cycle boundary;
- rejection of missing, extra, reordered, or mismatched optimizer states;
- rejection of standalone, nested-all, and per-block use;
- unchanged behavior and artifact compatibility when state scope is omitted;
- completion accounting that matches optimizer steps to width exposures;
- a short CPU end-to-end run through the normal training entrypoint.

## Out of scope

Exclude:

- separate model weights or model replicas per width;
- disjoint parameter partitions assigned permanently to different optimizers;
- width-local learning-rate schedules or warmup clocks;
- different optimizer algorithms or hyperparameters by width;
- nested-all and per-block optimizer ownership;
- changing granularity sampling probabilities as part of this feature;
- changing MatFormer FFN mathematics or correction behavior;
- distributed multi-process support unless planning can preserve exact optimizer
  collection checkpoint semantics without expanding the initial experiment;
- scientific superiority claims based only on smoke tests or one seed.

## Measurable success criteria

- In every per-granularity run, exactly one optimizer instance advances per
  committed global optimizer step.
- Non-selected optimizer states remain bitwise unchanged across a successful
  step in focused deterministic tests.
- Shared and per-granularity paired runs use identical global learning rates,
  batches, and width actions at every matched step.
- Per-width update counts sum exactly to total committed steps and match width
  exposure counts.
- Matching uninterrupted and resumed executions produce matching subsequent
  actions, optimizer states, scheduler state, and model parameters.
- Invalid topology, action, state-scope, and resume combinations fail before
  silently changing training behavior.
- Existing runs that omit optimizer state scope resolve to shared state and
  remain behaviorally unchanged.
- The TinyStories-Instruct recipe preflights all six one-epoch paired runs and
  records enough provenance to audit the final comparison.

Do not prescribe programming-language classes, module names, or source-file
organization in the specification. Preserve the experimental semantics and
observable contracts above; defer implementation structure to planning.

---
