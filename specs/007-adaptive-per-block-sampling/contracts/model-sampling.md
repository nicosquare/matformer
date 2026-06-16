# Contract: Model Sampling Configuration

This contract defines the config surface for MatFormer sampling modes.

## Required Fields

- `run.sampling_mode`
  - Type: string
  - Allowed values: `nested-random`, `nested-all`, `standalone`
- `model.granularity_sampling_mode`
  - Type: string
  - Allowed values: `global`, `per_block`, `adaptive_per_block`
- `model.correction_mode`
  - Type: string
  - Allowed values: `none`, `gmc`, `lmc`
- `model.membership_correction`
  - Type: boolean
- `model.granularities`
  - Type: ordered list of `s`, `m`, `l`, `xl`

## Adaptive Controls

The adaptive path is only valid when `model.granularity_sampling_mode` is
`adaptive_per_block`.

- `model.adaptive_sampler_strategy`
  - Type: string
  - Allowed values: `thompson`, `ucb`
  - Default: `thompson`
- `model.adaptive_sampler_exploration_scale`
  - Type: number
  - Default: implementation-defined, but must be explicit in resolved config
- `model.adaptive_sampler_decay_rate`
  - Type: number
  - Default: implementation-defined, but must be explicit in resolved config
- `model.adaptive_sampler_reward_penalty_weight`
  - Type: number
  - Purpose: balances loss improvement against correction penalty

## Validation Rules

- `adaptive_per_block` must only resolve under `nested-random`.
- `nested-all` must resolve to the global path.
- `per_block` must preserve the current random baseline.
- invalid mode combinations must fail before training starts.
- the resolved config must keep the legacy alias behavior (`training.granularity_sampling`)
  stable for existing debug and pilot configs.

## Expected Runtime Behavior

- `global`
  - the current whole-model path
  - all layers follow the same granularity decision
- `per_block`
  - the current random per-block baseline
  - one granularity is selected independently for each transformer block
- `adaptive_per_block`
  - the same per-block selection granularity as the baseline
  - selections are biased by the adaptive sampler state and strategy

## Compatibility Notes

- Existing `nested-random + global` and `nested-random + per_block` runs must
  remain behaviorally unchanged apart from explicit config and provenance
  fields.
- `training.granularity_sampling` remains a compatibility alias and not the
  source of truth.
