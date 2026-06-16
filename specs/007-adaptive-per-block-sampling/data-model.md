# Data Model: Adaptive Per-Block Sampling

## Sampling Mode

Represents the resolved model-level sampling mode.

- Fields:
  - `run.sampling_mode`: `nested-random`, `nested-all`, or `standalone`
  - `model.granularity_sampling_mode`: `global`, `per_block`, or
    `adaptive_per_block`
  - `training.granularity_sampling`: compatibility alias (`all` or `random`)
- Validation:
  - `adaptive_per_block` is valid only when `run.sampling_mode == nested-random`
  - `nested-all` and `standalone` must not resolve to adaptive sampling
  - invalid mode pairings must fail before training begins
- Relationships:
  - Drives `GranularityPattern`
  - Determines `CorrectionContext`
  - Selects the adaptive sampler path in training

## Sampler Strategy

Names the adaptive decision rule.

- Fields:
  - `strategy_name`: `thompson` or `ucb`
  - `exploration_scale`: numeric
  - `decay_rate`: numeric
  - `reward_weight` or equivalent correction penalty weight
- Validation:
  - strategy must be an explicit config choice
  - default strategy is `thompson`
  - parameters must be non-negative and finite
- Relationships:
  - Used by `AdaptiveSamplerState`
  - Recorded in `RunProvenanceRecord`

## Adaptive Sampler State

Persistent state for adaptive per-block selection.

- Fields:
  - `phase`
  - `step`
  - `epoch`
  - `exploration_scale`
  - `decay_rate`
  - `stats[block][granularity].mean_reward`
  - `stats[block][granularity].count`
  - `stats[block][granularity].last_seen_step`
  - optional recent-loss and correction summaries
- Validation:
  - block count must match the model
  - granularity keys must be one of `s`, `m`, `l`, `xl`
  - resumed runs must restore compatible state or fail clearly
- State transitions:
  - `fresh` -> `updated` after the first sampled step
  - `updated` -> `decayed` as exploration and history evolve
  - `decayed` -> `resumed` when loaded from checkpoint or artifact state
- Relationships:
  - Produces `RewardRecord`
  - Updates `GranularityPattern`
  - Written to run artifacts for resume support

## Reward Record

Captures the scalar signal used to update the sampler.

- Fields:
  - `previous_loss`
  - `current_loss`
  - `loss_improvement`
  - `correction_penalty`
  - `reward`
  - `phase`
  - `step`
  - `epoch`
- Validation:
  - reward must be derived from loss improvement minus a normalized correction
    penalty
  - the correction penalty must be scaled to the same rough magnitude as the
    loss delta before combination
- Relationships:
  - Feeds the sampler state update
  - Can be summarized in `metrics.csv` and `run_summary.json`

## Granularity Pattern

Describes the selected pattern for one run step.

- Fields:
  - `pattern_type`: `single`, `per_block`, or `all_granularities`
  - `selected_granularities`: ordered tuple/list of `s`, `m`, `l`, `xl`
  - `layer_count`
  - `repeatable_source`
- Validation:
  - each granularity must be valid
  - `per_block` patterns must have one choice per transformer block
  - `single` patterns must remain behaviorally equivalent to the current global
    path
- Relationships:
  - Derived from `Sampling Mode` and the adaptive sampler
  - Stored in artifacts for interpretation

## Run Provenance Record

Summarizes how a run should be interpreted and resumed.

- Fields:
  - resolved run mode
  - resolved sampling mode
  - selected sampler strategy
  - runtime granularity pattern summary
  - correction context
  - reward summary
  - correction-penalty summary
  - resumable sampler state
- Validation:
  - must distinguish `global`, random `per_block`, and `adaptive_per_block`
    without logs
  - must survive resume and artifact comparison
- Relationships:
  - Written to `config.json` and `run_summary.json`
  - Read back by downstream validation and resume checks
