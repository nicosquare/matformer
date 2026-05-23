# Research: Cat Llama Granularity Pipeline

## Decision: Use a config-level model variant selector

**Decision**: Add a single `model.variant` config value with canonical values
`matformer_llama` and `cat_llama`, defaulting to `matformer_llama`.

**Rationale**: The feature is a comparison-oriented variant of the same
experiment path, so the selector should be visible in configuration rather than
hidden in a separate script or implicit code branch.

**Alternatives considered**:
- Reuse `run.model_family`: rejected because that field already means
  `nested` vs `standalone` topology in the current configs.
- Add a separate command-line flag only: rejected because the project already
  treats YAML config plus overrides as the source of truth for research runs.
- Create a second entry script: rejected because it would split the shared
  experiment path and make comparisons harder to reason about.

## Decision: Keep `ModifiedLlamaForCausalLM` as the shared construction point

**Decision**: Continue building the model through
`ModifiedLlamaForCausalLM(config, mlp_cls=...)` and select the MLP class from
the resolved config.

**Rationale**: The code already exposes the right injection point, so the new
variant can reuse the existing training and artifact flow without introducing a
dispatch layer.

**Alternatives considered**:
- Duplicate model-building logic in `train.py`: rejected because it would split
  the model path and complicate debugging.
- Move variant logic into a registry: rejected because the repository favors
  direct, local reasoning over a generalized factory layer.

## Decision: Preserve the current artifact schema and add variant labels

**Decision**: Keep the existing CSV/JSON artifact categories and add the model
variant to resolved config and run-summary data so `cat_llama` and
`matformer_llama` remain directly comparable.

**Rationale**: The experiment needs a like-for-like comparison, not a new
reporting format.

**Alternatives considered**:
- Emit a new artifact family for `cat_llama`: rejected because it would make
  comparisons and downstream plotting more cumbersome.
- Omit variant labels from outputs: rejected because runs would be harder to
  audit after the fact.

## Decision: Leave granularity assembly differences isolated to the variant

**Decision**: Treat concatenation-based granularity assembly as the only
behavioral difference between the two model variants.

**Rationale**: The feature request is specifically about comparing granularity
construction methods, so the rest of the training flow should stay unchanged.

**Alternatives considered**:
- Change topology or evaluation behavior at the same time: rejected because it
  would confound the comparison.
- Introduce new experiment phases: rejected because the feature should stay
  concise and route through the existing experiments.

## Decision: Resolve warmup from the training step budget

**Decision**: Support both `training.warmup_ratio` and `training.warmup_steps`,
with `warmup_steps` taking precedence when both are present. Resolve ratio-based
warmup against the final `training.max_steps` value derived from
`training.token_budget` and `effective_world_size`.

**Rationale**: Research runs often need either exact control for tiny debug jobs
or a portable fraction of total training length for scaled runs. Tying warmup to
the final step budget keeps the policy stable when world size changes.

**Alternatives considered**:
- Only absolute warmup steps: rejected because it makes scaled runs harder to
  compare.
- Only warmup ratio: rejected because some debug jobs need exact step counts.
- Ratio of token budget directly: rejected because the actual step count depends
  on the resolved batch/global-world-size combination.

## Decision: Make learning-rate scaling explicit and batch-aware

**Decision**: Keep `training.learning_rate` as the base learning rate and scale
it from the resolved global batch size using a config-controlled
`training.learning_rate_scale_rule` with `none`, `linear`, and `sqrt` as valid
values.

**Rationale**: The debug investigation showed that distributed behavior changes
meaningfully as the effective batch size changes. Making the scale rule explicit
keeps those changes auditable and makes distributed runs easier to compare.

**Alternatives considered**:
- No LR scaling: rejected because it hides world-size sensitivity.
- Implicit scaling only: rejected because it makes the effective learning rate
  hard to reconstruct from the artifacts.
- Token-budget-only scaling: rejected because the meaningful control variable is
  the resolved global batch size.

## Decision: Select optimizers from config with AdamW as the default

**Decision**: Add `training.optimizer.name` and `training.optimizer.kwargs`,
default `name` to `adamw`, and allow `sgd` as a supported alternative for
debugging and FSDP comparisons.

**Rationale**: The diagnostic work points to AdamW sensitivity under FSDP, so
the plan needs a config-visible way to switch optimizers without changing the
run path.

**Alternatives considered**:
- Hard-code AdamW only: rejected because it blocks the debugging path.
- Add a custom optimizer registry: rejected because it would add indirection
  without helping the immediate comparison workflow.
- Separate code paths per optimizer: rejected because it would make the
  comparison surface harder to reason about.

## Decision: Record resolved schedule and optimizer state in artifacts

**Decision**: Persist the resolved schedule policy, resolved learning rate,
warmup policy, optimizer name, and optimizer kwargs in `config.json` and
`run_summary.json`.

**Rationale**: Distributed experiments need enough metadata to reconstruct why
two runs differed, especially when the same model variant behaves differently
under different optimizer and warmup settings.

**Alternatives considered**:
- Store only the raw author-written config: rejected because the effective run
  behavior would still be hidden.
- Store schedule only in terminal logs: rejected because logs are not a durable
  comparison surface.
