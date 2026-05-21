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
