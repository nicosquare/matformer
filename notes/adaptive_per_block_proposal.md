# Adaptive Per-Block Sampling Proposal

Date: 2026-06-15

This note is the theoretical follow-up to
[granularity_sampling_discussion_2026-06-09.md](/home/nicolas.avila/dev/references/matformer/notes/granularity_sampling_discussion_2026-06-09.md).
It focuses only on the next feature idea: making `nested-random + per_block`
stateful and time-aware instead of purely random.

## Goal

Add a new config-controlled operation mode that keeps the existing random
`nested-random + per_block` path intact, while introducing an adaptive
`per_block` variant that learns which granularities are promising for each
block over the course of training.

The first version should behave like a contextual bandit, not full RL.

## Core Idea

At each training iteration:

1. Build a small state summary for each transformer block.
2. Choose one granularity per block from the discrete action set.
3. Run the forward/backward pass with that sampled pattern.
4. Compute a scalar reward from the step outcome.
5. Update the sampler state so future choices are biased by recent evidence.

The sampler only decides the per-block granularity pattern. It does not
replace the model, optimizer, correction logic, or the existing random
`per_block` baseline.

## State

Keep the state small and cheap.

Per block and granularity:

```text
stats[b][g] = {
  mean_reward,
  count,
  last_seen_step
}
```

Global sampler state:

```text
sampler_state = {
  phase,
  step,
  epoch,
  exploration_scale,
  decay_rate
}
```

Optional context features:

- recent loss slope
- recent correction magnitude
- warmup completion ratio

## Action Space

The initial action space stays discrete:

- `s`
- `m`
- `l`
- `xl`

The first version should stay at the per-block choice level. Structured patterns
can come later if needed.

## Reward

Use a simple scalar reward:

```text
reward_t = (loss_{t-1} - loss_t) - λ * correction_cost_t
```

Where:

- `loss_{t-1}` is the previous step loss
- `loss_t` is the current step loss
- `correction_cost_t` is a normalized scalar penalty
- `λ` is a tunable weight

Preferred first-pass correction cost:

```text
correction_cost_t = mean_block_correction_norm / mean_activation_norm
```

Cheaper fallback:

```text
correction_cost_t = 1 if correction was applied else 0
```

The norm-based version is better because it distinguishes light correction from
heavy correction.

## Selection Rules

### Thompson-Style Sampling

Score each action as a noisy draw around its historical mean:

```text
score[b][g] = mean_reward[b][g] + noise(scale = f(count[b][g], phase))
```

This keeps exploration alive even when one choice already looks good.

### UCB-Style Sampling

Score each action using an explicit exploration bonus:

```text
score[b][g] = mean_reward[b][g] + exploration_scale * sqrt(log(step + 1) / (count[b][g] + 1))
```

This prefers uncertain actions more directly.

### Pattern Choice

For each block:

```text
pattern[b] = argmax_g score[b][g]
```

## Update Rule

Use decay so stale history does not dominate forever.

```text
mean_reward[b][g] <- (1 - decay_rate) * mean_reward[b][g] + decay_rate * reward_t
count[b][g] <- count[b][g] + 1
last_seen_step[b][g] <- step
```

Apply the update to each chosen `(b, g)` pair in the sampled pattern.

This makes the sampler nonstationary by design. Early training and late training
can end up preferring different granularities.

## Phase Awareness

The sampler should be aware of training phase:

- `warmup`
- `mid_train`
- `late_train`

Two simple ways to do that:

- maintain separate statistics per phase
- include `phase` as context in the scoring rule

One practical schedule:

```text
exploration_weight = base_weight * decay(epoch)
```

with exploration decaying over time.

## Toy Numerical Example

Assume a model with 3 transformer blocks and 4 choices per block.

Initial statistics:

- all `(block, granularity)` pairs start at `mean_reward = 0.0`
- all counts start at `0`

Use:

```text
reward = previous_loss - current_loss - correction_penalty
```

Iteration 1:

- pattern: `block 0 -> s`, `block 1 -> m`, `block 2 -> l`
- previous loss: `10.0`
- current loss: `9.4`
- correction penalty: `0.1`
- reward: `0.5`

The selected pairs each get updated toward `0.5`.

Iteration 2:

- pattern: `block 0 -> s`, `block 1 -> m`, `block 2 -> xl`
- previous loss: `9.4`
- current loss: `9.0`
- correction penalty: `0.2`
- reward: `0.2`

Now the sampler has a small amount of evidence that:

- `s` and `m` were useful for the first two blocks in this toy trace
- `xl` was not disastrous for block 2

That is the basic feedback loop: choose a pattern, observe reward, bias the
future pattern.

## Thompson Example

Suppose the observed means after the toy iterations are:

- `(block 0, s)` -> `0.5`
- `(block 1, m)` -> `0.5`
- `(block 2, l)` -> `0.5`
- `(block 2, xl)` -> `0.2`

Thompson-style sampling adds noise around those means.

For block 0, a noisy draw might produce:

- `s` -> `0.52`
- `m` -> `0.31`
- `l` -> `-0.05`
- `xl` -> `0.08`

It chooses `s`.

For block 2:

- `l` -> `0.47`
- `xl` -> `0.29`
- `m` -> `0.15`
- `s` -> `-0.01`

It chooses `l`.

The point is not the exact noise distribution. The point is that the sampler
can still explore weaker actions occasionally while preferring historically
better ones.

## UCB Example

Use:

```text
ucb_score = mean_reward + exploration_bonus
exploration_bonus = sqrt(2 * ln(total_steps) / count)
```

Assume `total_steps = 3`.

For `(block 0, s)`:

- `mean_reward = 0.5`
- `count = 1`
- `exploration_bonus ~= 1.48`
- `ucb_score ~= 1.98`

For `(block 0, m)`:

- `mean_reward = 0.0`
- `count = 0`
- treat as unexplored and give it a large bonus, for example `+2.0`
- `ucb_score = 2.0`

This can still choose a not-yet-tried action even when its mean reward is
currently lower.

## Checkpoint Fields

To resume a run without losing the sampler behavior, checkpoint:

```text
{
  stats,
  phase,
  step,
  epoch,
  exploration_scale,
  decay_rate
}
```

If the sampler state is not checkpointed, a resumed run will behave like a
fresh run and forget its policy history.

## Logging Fields

Log the following for interpretability:

- sampled per-block pattern
- reward
- correction cost
- phase
- sampler summary statistics, such as top preferred granularities per block

## Minimal Implementation Boundary

The first implementation does not need:

- a learned embedding for block state
- a neural policy network
- a replay buffer
- delayed credit assignment across multiple steps

It only needs:

- persistent block/action statistics
- a phase-aware scoring rule
- a stable reward update

## Suggested First Slice

1. Add an `adaptive_per_block` sampling submode alongside the existing
   `global` and `per_block` choices under `nested-random`.
2. Reuse the existing per-block wiring and correction plumbing.
3. Add a minimal bandit state object keyed by block index and granularity.
4. Update the policy once per iteration using a simple scalar reward.
5. Log the sampled pattern and policy state summary in run artifacts.
6. Validate that the adaptive path still preserves the existing global and
   random per-block behaviors when disabled.

## Why This Is A Good Next Step

- It uses the per-block machinery already in place.
- It gives us an experimental signal without full RL complexity.
- It keeps the action space discrete and interpretable.
- It preserves the existing mode surface instead of rewriting it.
