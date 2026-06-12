# Research: Granularity Operation Modes

## 1. Canonical Run-Mode Surface

- **Decision**: Keep `run.sampling_mode` as the top-level runtime selector with
  the explicit values `nested-random`, `nested-all`, and `standalone`, while
  `model.granularity_sampling_mode` continues to represent the model-level
  granularity submode with `global` and `per_block`.
- **Rationale**: The repository already distinguishes top-level run mode from
  model-level granularity selection in config resolution. Preserving that split
  keeps the feature explicit without inventing a new config hierarchy.
- **Alternatives considered**:
  - Collapsing run mode and sampling submode into one field. Rejected because it
    would blur the difference between `nested-all`, `standalone`, and the
    `nested-random` submodes.
  - Making an alternate sampling field the main selector. Rejected because it
    would split the configuration contract away from the canonical mode fields.

## 3. Model Module Split

- **Decision**: Keep the shallow `models/` package as the model-facing boundary
  and let it own granularity metadata, FFN layouts, correction helpers, and
  wiring helpers.
- **Rationale**: The repository already separates those concerns into
  `models/granularity.py`, `models/correction.py`, `models/ffn.py`, and
  `models/wiring.py`. That structure keeps the behavior inspectable without
  introducing a deeper abstraction layer.
- **Alternatives considered**:
  - Reintroducing a monolithic model file. Rejected because the current split is
    already clearer and easier to test.
  - Adding a broader factory/registry system. Rejected because the constitution
    favors simple, local reasoning for research code.

## 4. Correction Behavior by Mode

- **Decision**: Preserve the existing whole-model correction behavior for the
  global path, and only derive local GMC/LMC behavior when
  `model.granularity_sampling_mode=per_block`.
- **Rationale**: The feature must not change the behavior of the current global
  path. Tying local correction to the per-block mode makes parity tests for the
  existing path straightforward.
- **Alternatives considered**:
  - Enabling local correction for all modes. Rejected because it would change
    the semantics of the existing global and nested-all runs.
  - Treating correction as a separate independent mode. Rejected because the
    requested behavior is explicitly coupled to the sampling pattern.

## 4. Metadata and Artifact Provenance

- **Decision**: Save `run.sampling_mode`, `model.granularity_sampling_mode`,
  the correction context, and a compact `granularity_pattern_summary` in config
  and run-summary artifacts.
- **Rationale**: The repository already uses `config.json`, `run_summary.json`,
  and CSV metrics as the persistent audit trail. Recording the resolved mode
  and pattern there lets downstream analysis reconstruct a run without logs.
- **Alternatives considered**:
  - Recording only the resolved mode and omitting the runtime pattern summary.
  Rejected because per-block and nested-all runs need the pattern to be
  interpretable later.
  - Storing provenance only in console output. Rejected because logs are not a
    durable experiment record.

## 5. Validation Strategy

- **Decision**: Use focused config-resolution tests, artifact serialization
  tests, model-wiring tests, and smoke runs for the debug matrix and d_model=256
  pilot configurations.
- **Rationale**: This covers the likely failure modes: wrong mode resolution,
  invalid correction activation, and missing artifact provenance.
- **Alternatives considered**:
  - Relying only on end-to-end training runs. Rejected because they are too
    expensive for an iterative refactor.
  - Relying only on unit tests. Rejected because the feature also changes run
    orchestration and saved metadata.
