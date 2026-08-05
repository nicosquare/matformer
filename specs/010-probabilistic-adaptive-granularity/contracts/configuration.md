# Configuration Contract: Probabilistic Adaptive Granularity

## Surface

Bayesian Thompson uses the existing YAML configuration and repeatable
`--override dotted.path=value` interface. It does not add dedicated CLI flags.

### Bayesian global example

```yaml
model:
  granularity_sampling_mode: adaptive_global
  adaptive_sampler_strategy: thompson
  adaptive_controller:
    decision_interval_steps: 50
    prior_mean: 0.0
    prior_covariance: 1.0
    observation_noise_variance: 0.01
    process_noise_covariance: 0.0001

evaluation:
  adaptive_controller:
    enabled: true
    source: configured_dataset_split
    examples: 128
    objective_weights: uniform
    fixed_manifest: true
  final_holdout:
    enabled: true
    source: configured_dataset_split
    examples: 512
    fixed_manifest: true
    evaluate_during_training: false
```

For additive per-block Bayesian sampling, change only:

```yaml
model:
  granularity_sampling_mode: adaptive_per_block
  adaptive_sampler_strategy: thompson
```

The controller mapping remains required and has the same reward/data semantics.

## Strategy Resolution Matrix

| Resolved sampling mode | Strategy | Result |
|---|---|---|
| `adaptive_global` | `thompson` | Valid Bayesian global controller |
| `adaptive_global` | `ucb` | Invalid before training |
| `adaptive_per_block` | `thompson` | Valid Bayesian additive per-block controller |
| `adaptive_per_block` | `ucb` | Existing heuristic UCB behavior, unchanged |
| `global` | none | Existing random global behavior, unchanged |
| `per_block` | none | Existing random per-block behavior, unchanged |
| `nested-all` / standalone resolution | none | Existing behavior, unchanged |

If `adaptive_sampler_strategy` is omitted for an adaptive mode, the previous
`thompson` default still identifies the requested strategy, but the complete
Bayesian controller/data configuration remains mandatory. The resolver must not
synthesize missing probabilistic inputs.

## Bayesian Controller Fields

### `model.adaptive_controller.decision_interval_steps`

- Positive integer optimizer-step count.
- Optional only because the experiment contract defines the default as 50.
- Counts optimizer updates governed by the selected action; pre-nested warmup
  steps do not count.

### `model.adaptive_controller.prior_mean`

- Required.
- A finite scalar, expanded deterministically to all coefficient positions, or
  a finite list whose length matches the resolved feature dimension.
- The expanded vector is saved in resolved configuration and controller
  provenance.

### `model.adaptive_controller.prior_covariance`

- Required.
- A finite nonnegative scalar interpreted as an isotropic covariance, a finite
  nonnegative coefficient-length list interpreted as a diagonal covariance, or
  a dense coefficient-dimension square matrix.
- Dense inputs must be symmetric and positive semidefinite.
- The resolved dense matrix is saved.

### `model.adaptive_controller.observation_noise_variance`

- Required finite scalar strictly greater than zero.
- Represents `sigma^2`, not a standard deviation.

### `model.adaptive_controller.process_noise_covariance`

- Required.
- Accepts the same scalar, diagonal, or dense forms as prior covariance.
- Zero is valid and resolves to a zero covariance.
- The resolved dense matrix is saved as `Q`.

## Fixed Method Fields

The resolver writes the following values for Bayesian runs; users cannot select
different values in this feature:

| Resolved field | Global | Per-block |
|---|---|---|
| `scope` | `global` | `per_block` |
| `feature_model` | `arms` | `additive` |
| `context_model` | `intercept_only` | `intercept_only` |
| `transition_model` | `identity` | `identity` |
| `compute_weight` | `0.0` | `0.0` |
| `switch_weight` | `0.0` | `0.0` |
| `method_family` | Bayesian Gaussian linear Thompson | Bayesian Gaussian linear Thompson |
| `method_version` | current controller schema version | current controller schema version |

Any supplied nonzero compute/switch cost or non-intercept context option is
rejected as outside the current method.

## Controller Data Fields

`evaluation.adaptive_controller` is required for Bayesian Thompson and must
resolve exactly to:

- `enabled: true`
- `source: configured_dataset_split`
- `examples: 128`
- `objective_weights: uniform`
- `fixed_manifest: true`

The controller selection seed is derived from the run root seed through a new
named seed stream and is written to resolved configuration and the manifest.

`evaluation.final_holdout` is required for Bayesian Thompson and must resolve
exactly to:

- `enabled: true`
- `source: configured_dataset_split`
- `examples: 512`
- `fixed_manifest: true`
- `evaluate_during_training: false`

Its seed uses a separate named stream from controller, ordinary validation, and
posterior sampling.

Ordinary `evaluation.validation` remains configured through its existing
surface and retains checkpoint-selection semantics.

## Migration and Conflict Rules

- `thompson` without the complete Bayesian controller and data sections fails
  before training with a migration-specific error.
- Legacy fields `adaptive_sampler_exploration_scale`,
  `adaptive_sampler_decay_rate`, and
  `adaptive_sampler_reward_penalty_weight` do not satisfy Bayesian inputs. A
  Thompson configuration that attempts to mix legacy and Bayesian controller
  controls fails rather than ignoring either set.
- `adaptive_per_block + ucb` continues to accept and resolve the current legacy
  UCB controls exactly as before.
- A Bayesian mode is valid only for `nested-random` runs and resolves the legacy
  `training.granularity_sampling` compatibility alias to `random`.
- Bayesian modes require a nonempty resolved granularity set and positive model
  block count.
- Invalid prior/noise shapes or values, data-role sizes, role overlaps, or
  insufficient usable examples fail before any optimizer update.

## Preflight Contract

`python train.py --config ... --preflight` prints, for Bayesian runs:

- requested/resolved sampling mode, strategy, scope, method family/version;
- resolved ordered granularities and coefficient dimension;
- decision interval and fixed cost/context/transition fields;
- normalized prior/noise inputs;
- controller and final-holdout role contracts and seed provenance;
- ordinary reproducibility controls already printed today.

Manifest hashes are marked pending during config-only preflight because they
cannot exist until the dataset is loaded and partitioned.

