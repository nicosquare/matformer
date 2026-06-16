# Research: Adaptive Per-Block Sampling

## Decision 1: Keep `run.sampling_mode` as the top-level mode selector

- Decision: Preserve `run.sampling_mode` as `nested-random`, `nested-all`, or
  `standalone`, and keep `model.granularity_sampling_mode` as the explicit
  model-level selector.
- Rationale: The repository already resolves runtime provenance from those two
  fields, so extending the existing split keeps configuration readable and keeps
  the legacy `random`/`all` alias behavior intact.
- Alternatives considered:
  - Collapsing everything into `run.sampling_mode`.
  - Introducing a generic sampler registry.
  - Reusing `training.granularity_sampling` as the source of truth.

## Decision 2: Add `adaptive_per_block` as a model-level mode only

- Decision: Allow `adaptive_per_block` only when the resolved run mode is
  `nested-random`.
- Rationale: The feature is explicitly an additive refinement of the current
  nested-random per-block path. Keeping it nested-only avoids ambiguity with
  `nested-all` and `standalone` runs and matches the validation rule in the
  spec.
- Alternatives considered:
  - Allowing `adaptive_per_block` for standalone runs.
  - Permitting it under `nested-all` as a synonym.

## Decision 3: Use a contextual bandit rather than full RL

- Decision: Implement the adaptive sampler as a lightweight contextual bandit.
- Rationale: The proposal note and feature spec both prefer a small state
  update loop that can be inspected and resumed without a separate policy or
  value network.
- Alternatives considered:
  - Full reinforcement learning with a learned policy network.
  - A fixed schedule with no online feedback.

## Decision 4: Track small, explicit sampler state

- Decision: Persist per-block, per-granularity statistics plus a compact global
  state containing phase, step, epoch, exploration scale, and decay rate.
- Rationale: This is enough to make the sampler nonstationary while keeping the
  state inspectable in saved artifacts and cheap to serialize.
- Alternatives considered:
  - Storing only aggregate counts.
  - Keeping the state entirely in memory and recomputing from logs.

## Decision 5: Make strategy selection explicit

- Decision: Support two named adaptive strategies, `thompson` and `ucb`, with
  `thompson` as the default.
- Rationale: The feature spec requires at least Thompson-style and UCB-style
  sampling, and the proposal note already frames them as the two concrete
  starting options.
- Alternatives considered:
  - A single implicit strategy.
  - Adding more strategies before the state and resume path are stable.

## Decision 6: Normalize the correction penalty before reward calculation

- Decision: Compute reward as loss improvement minus a normalized correction
  penalty, where the penalty is scaled to the same rough magnitude as the
  improvement signal before the two are combined.
- Rationale: This keeps the reward numerically interpretable and avoids the
  correction term overwhelming or disappearing relative to the loss delta.
- Alternatives considered:
  - Using raw correction magnitude directly.
  - Using a binary correction flag only.

## Decision 7: Persist provenance in artifacts rather than logs

- Decision: Write the resolved mode, strategy, runtime granularity pattern,
  reward summary, correction-penalty summary, and sampler state into resolved
  config and run summary artifacts, and keep stepwise metrics CSV rows aligned
  with those values.
- Rationale: The repository’s research workflow already treats JSON and CSV
  outputs as the audit trail. Artifact persistence makes the adaptive path
  comparable without terminal logs.
- Alternatives considered:
  - Recording the state only in console output.
  - Splitting the provenance across unrelated files with no single summary.

## Decision 8: Validate the explicit nested-random matrix before training

- Decision: Cover `nested-random + global`, `nested-random + per_block`, and
  `nested-random + adaptive_per_block` in config-resolution and smoke tests,
  plus invalid combinations that should fail before training starts.
- Rationale: The main risk is silent mode drift. The validation matrix should
  prove that the new mode is explicit, resumable, and distinguishable from the
  existing baseline.
- Alternatives considered:
  - Testing only the new adaptive mode.
  - Deferring invalid-mode checks until the training loop begins.
