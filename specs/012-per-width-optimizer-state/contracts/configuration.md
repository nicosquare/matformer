# Configuration Contract: Per-Width Optimizer State

## User-Facing Optimizer Inputs

```yaml
training:
  optimizer:
    name: adamw
    state_scope: shared
    scheduler_clock: global_step
    kwargs:
      betas: [0.9, 0.95]
      eps: 1.0e-8
      weight_decay: 0.1
```

Accepted values:

- `state_scope`: `shared` or `per_granularity`; omitted means `shared`.
- `scheduler_clock`: only `global_step`; omitted means `global_step`.

AdamW and SGD retain their existing supported kwargs and normalization. Scope
does not permit different optimizer families or hyperparameters by width.

## Resolved Shape and Compatibility

Resolution consumes the two new user-facing fields and writes:

```yaml
training:
  optimizer:
    name: adamw
    kwargs: { ... }
  optimizer_name: adamw
  optimizer_kwargs: { ... }
  optimizer_state_scope: shared
  optimizer_scheduler_clock: global_step
  optimizer_state_contract:
    schema_version: 1
    state_scope: shared
    scheduler_clock: global_step
    ordered_granularities: [g250, g500, g750, g1000]
    optimizer_name: adamw
    optimizer_kwargs: { ... }
    scheduler_contract: { ... }
    single_process_required: false
```

The resolved `training.optimizer` mapping deliberately retains its historical
`{name, kwargs}` shape. Artifact readers default a missing scope to `shared` and
a missing clock to `global_step`.

## Eligibility Matrix

| Run condition | `shared` | `per_granularity` |
|---|---:|---:|
| Nested global, fixed-global, or adaptive-global action with at least two widths | Valid | Valid |
| Global balanced-cycle sampling | Valid | Valid |
| Full-only or balanced-global pre-nested warmup, one width per step | Valid | Valid |
| Standalone model family | Valid | Rejected |
| Nested-all action | Valid | Rejected |
| Per-block or adaptive-per-block action | Valid | Rejected |
| Empty or multi-width global action | Existing behavior | Rejected before commit |
| One unique resolved width | Existing behavior | Rejected |
| Distributed multi-process execution | Existing behavior | Rejected |
| Non-global scheduler clock | Rejected | Rejected |

The resolver rejects duplicate labels, duplicate effective widths, malformed
ordered widths, and unknown values before training.

## Scheduler Contract

- `global_step` is the only supported clock.
- Warmup, stable, decay, and cooldown positions use committed global optimizer
  steps, not width exposure.
- Bias-correction ages, momentum, and other optimizer-internal counters remain
  local to each width optimizer.
- The learning rate used at matched step `t` is equal between shared and
  per-granularity arms.
- All width optimizers expose the same current param-group rates even when only
  one width's moments advance.

## Preflight Output

`python train.py --config ... --preflight` adds:

- requested and resolved optimizer state scope;
- scheduler clock;
- optimizer collection schema version;
- ordered optimizer-owner labels;
- optimizer family and normalized kwargs;
- resolved global scheduler contract and learning rate;
- ownership eligibility and single-process requirement;
- arm identity and paired-control signature.

Preflight fails before dataset loading or model mutation for invalid topology,
scope, clock, action cardinality, warmup ownership, or distributed settings.

## Paired Identity Rules

- Full run and checkpoint identity includes state scope.
- The paired-control signature contains optimizer name and kwargs, but excludes
  only state scope.
- Scope is stored separately as the intended intervention.
- Within a seed pair, preflight must show equal model initialization, role
  identities, optimizer family/kwargs, scheduler, learning rates, batches,
  ordered widths, action schedule, budget, and evaluation cadence.

## Frozen TinyStories-Instruct Recipe

`configs/controlled_exps/tinystories_instruct_per_width_optimizers.yaml` is
opt-in and defaults to `shared`. It fixes:

- prepared `roneneldan/TinyStoriesInstruct` four-role corpus;
- slicing model, hidden size 64, four layers, four heads, context 128,
  vocabulary 2048;
- ordered widths `g250`, `g500`, `g750`, `g1000` and correction `none`;
- nested global balanced-cycle sampling with interval one;
- AdamW 0.008, betas 0.9/0.95, epsilon 1e-8, decay 0.1;
- cosine scheduling and 64 global warmup steps;
- batch 64, accumulation one, BF16, one process;
- strict reproducibility, deterministic repeat-epoch optimizer data, ordinary
  validation every 64 steps, fixed controller role, and sealed 512-example
  final holdout;
- pilot budget 713,785,344 tokens (87,132 steps; 21,783 per width) by default;
- seeds and confirmation budget supplied only through explicit overrides.

The candidate arm changes only `training.optimizer.state_scope` to
`per_granularity`. The optional confirmation starts fresh with budget
2,141,356,032 tokens (261,396 steps; 65,349 per width).
