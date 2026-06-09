# Granularity Sampling Discussion

Date: 2026-06-09

## Context

The current pipeline samples a single granularity per data pass and trains that
granularity globally for the step. This keeps the training loop simple and the
metrics easy to interpret, but it does not explore whether more local or
adaptive sampling can improve optimization.

## Ideas Discussed

### 1. Per-block granularity sampling

Sample a granularity independently per transformer block, then enable GMC/LMC
according to the sampled per-block pattern.

Assessment:
- Best match to the existing codebase.
- `ModifiedLlamaForCausalLM` already exposes `configure_layer_granularities(...)`.
- The training loop only needs a policy layer for producing a per-layer pattern.
- Correction logic must become pattern-aware if the selected pattern changes
  every step.

### 2. Active-learning style adaptive sampling

Use observed training outcomes to update a sampling process, e.g. Gaussian or
Bayesian-style selection over granularities.

Assessment:
- Better framed as a contextual bandit or Thompson-sampling policy than
  classical active learning.
- Feasible if the action space is small and discrete.
- More practical than full RL, but still requires stateful sampler bookkeeping.

### 3. Online RL policy

Learn a policy over training to choose granularities.

Assessment:
- Highest complexity.
- Requires policy learning, reward design, checkpointing for policy state, and
  careful handling of nonstationarity.
- Better suited for a later research phase.

### 4. Q-learning with replay buffer

Learn a Q-function with warmup and replay buffer to choose granularities per
step.

Assessment:
- More structured than generic policy RL.
- Still much heavier than the current training pipeline.
- Requires explicit replay, target updates, exploration schedule, and a stable
  discrete action space.

## Recommendation

Start with option 1.

Reasoning:
- It is closest to the current implementation seams.
- It produces an immediate signal on whether local granularity variation helps.
- It keeps the door open for later bandit or RL approaches without locking the
  code into a heavier abstraction too early.

## Additional Precisations

- Reshape `modified_llama.py` into a clearer model module so the model logic is
  easier to extend and reason about as behavior becomes more complex.
- Rule out distributed training for this effort for now.
  - The current distributed path already introduces extra complexity.
  - The first implementation should stay in the single-process/single-rank
    path so the behavior is easier to validate.

## Implementation Direction For Option 1

1. Add a config surface for a per-block granularity policy.
2. Extend the training step so it can request either:
   - one global granularity, or
   - a list of granularities, one per layer.
3. Make GMC/LMC correction depend on the sampled pattern, not only the static
   configured granularity list.
4. Log the sampled pattern in run artifacts and metrics so comparisons remain
   interpretable.

## Open Questions

- Should the first version sample layers independently, or should it sample a
  structured pattern such as contiguous blocks?
- Should validation remain fixed to canonical granularities while training uses
  sampled per-layer patterns?
- Should the correction logic operate at the layer level, or should it be
  aggregated over the whole sampled pattern?
