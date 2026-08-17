# Configuration Contract: PanelGrad Sampling

## Surface

PanelGrad uses the existing YAML configuration and repeatable dotted `--override` interface. It adds no dedicated CLI flags.

```yaml
model:
  granularity_sampling_mode: adaptive_global
  adaptive_sampler_strategy: panelgrad
  panelgrad:
    refresh_interval_steps: 50
    eta: 1.0e-12
    temperature: 1.0
    epsilon: 0.1

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

## Strategy Resolution

| Sampling mode | Strategy | Result |
|---|---|---|
| `adaptive_global` | `panelgrad` | Valid PanelGrad run |
| `adaptive_per_block` | `panelgrad` | Invalid before training |
| `global` / `per_block` | `panelgrad` | Invalid before training |
| `adaptive_global` | `thompson` | Existing Bayesian global behavior |
| `adaptive_per_block` | `thompson` | Existing Bayesian additive behavior |
| `adaptive_per_block` | `ucb` | Existing UCB behavior |
| Existing nonadaptive modes | existing strategy | Unchanged |

PanelGrad requires a nested model and resolves the run mode to `nested-random`. The legacy `training.granularity_sampling` compatibility value remains `random`.

## PanelGrad Fields

### `model.panelgrad.refresh_interval_steps`

- Positive integer completed PanelGrad optimizer steps.
- Defaults to `50`.
- Does not count balanced warmup or failed optimizer attempts.
- The next refresh occurs before the first action after the interval, never as an unused terminal operation.

### `model.panelgrad.eta`

- Finite scalar strictly greater than zero.
- Defaults to `1e-12`.
- Added to every nonnegative RMS score before powered normalization.

### `model.panelgrad.temperature`

- Finite scalar strictly greater than zero.
- Defaults to `1.0`.
- `1.0` gives direct proportional normalization; larger values flatten and smaller values sharpen.

### `model.panelgrad.epsilon`

- Finite scalar in `[0,1]`.
- Defaults to `0.1`.
- `0` is a controlled no-uniform-mixture ablation; `1` is uniform global sampling after measurement.

### `model.panelgrad.epsilon_schedule`

- Optional alternative to scalar `epsilon`; specifying both is invalid.
- The supported form is `type: linear` with finite `start` and `end` in `[0,1]` and positive integer `duration_steps`.
- At a refresh, schedule step `s` is the number of committed PanelGrad optimizer steps before that refresh. Warmup and failed optimizer attempts do not count.
- The active value is `start + (end - start) * min(s / duration_steps, 1)` and clamps to `end` after the duration.
- The first adaptive refresh uses `start`; epsilon and the resulting `p` remain frozen for the complete `H`-step interval.

Example:

```yaml
model:
  panelgrad:
    epsilon_schedule:
      type: linear
      start: 0.5
      end: 0.1
      duration_steps: 24415
```

## Fixed Resolved Fields

The resolver records and rejects conflicting supplied values for:

- `method_family: panelgrad_gradient_rms`
- `method_version: 1`
- `scope: global`
- `score: raw_aggregate_controller_gradient_rms`
- `support: granularity_controlled_ffn`
- `probability_mapping: powered_score_uniform_mixture`
- `action_distribution: categorical`
- `relative_tolerance: 1e-6`
- `absolute_tolerance: 1e-8`
- `inverse_probability_weighting: false`
- `compute_correction: false`

Resolved configuration also records ordered granularities, support counts/hash, controller/final role contracts, and `panelgrad_sampling` seed provenance.

## Data Roles

PanelGrad reuses the existing exact contracts:

- controller: fixed 128 examples from the configured source, deterministic manifest, used only for PanelGrad decisions;
- final holdout: fixed 512 examples, never evaluated during training;
- ordinary validation: unchanged monitoring/checkpoint-selection role;
- optimizer training: every remaining usable example.

All four roles remain pairwise disjoint and must have compatible saved manifests on resume. Raw-dataset PanelGrad configs must trigger this split; prepared packed-mmap runs reuse their existing role manifests.

The inherited `objective_weights: uniform` value remains part of the shared controller-role contract. PanelGrad does not form the TS uniform scalar objective; it measures every granularity separately on the same role.

## Balanced Warmup

`training.pre_nested_warmup.policy: balanced_global` is valid for PanelGrad. Its existing forced global schedule, state, seed, hash, passes, and resume rules remain unchanged. When `action_interval_steps` is omitted, it defaults to PanelGrad `refresh_interval_steps`.

Warmup steps do not consume PanelGrad exposure or interval progress. If training continues, the first refresh occurs at the warmup completion step. TS posterior reset and acquisition configuration is invalid and unused for PanelGrad.

## Conflict and Validation Rules

- `model.panelgrad` outside `adaptive_global + panelgrad` fails rather than being ignored.
- `adaptive_global + panelgrad` with Bayesian posterior/reset inputs or legacy UCB controls fails as mixed strategy configuration.
- Empty/duplicate granularities, zero controlled FFN support, non-finite policy values, and invalid ranges fail before adaptive training.
- Distributed PanelGrad requires current FSDP `use_orig_params: true` and rejects CPU parameter/gradient offload until an exact supported measurement path exists.
- Controller/final role configuration must match the existing fixed contracts.
- Resume requires exact method version, granularity order, scalar or scheduled epsilon policy, support hash/counts, data hashes, and seed identity. Version-1 fixed-epsilon PanelGrad state is migrated only for fixed-policy resumes.

## Preflight Contract

`python train.py --config ... --preflight` prints:

- requested/resolved sampling mode and strategy;
- PanelGrad family/version/scope;
- ordered granularities and per-granularity controlled FFN counts when available;
- `H`, `eta`, `T`, the fixed epsilon or schedule type/start/end/duration, and tolerance defaults/overrides;
- probability, loss, gradient, and support semantics;
- controller/final role contracts and named seed provenance;
- whether balanced warmup is active and its resolved interval.

Dataset-derived manifest hashes and model-derived support hashes may be marked pending until their inputs are available, then are written to resolved `config.json` before the first adaptive action.
