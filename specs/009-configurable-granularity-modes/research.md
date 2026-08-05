# Research: Configurable Granularity Modes

## Decision 1: Granularity mode lives in configuration, with canonical default behavior

- **Decision**: Use an explicit granularity mode field in configuration with `canonical` and `explicit` values, while preserving canonical behavior when the field is omitted.
- **Rationale**: The spec requires backward compatibility, and the existing code already resolves model metadata from config before training. A config-first mode switch keeps the behavior obvious and avoids a second pathway in the runtime.
- **Alternatives considered**:
  - Treat explicit granularities as the only mode and infer canonical behavior from four labels. Rejected because it makes the backward-compatibility contract implicit.
  - Add a separate resolver object or registry. Rejected because it adds abstraction without reducing complexity.

## Decision 2: Resolve granularities before downstream validation

- **Decision**: Extend the existing config-resolution path so the resolved config contains the ordered granularity list, the prefix-fraction map, and any derived widths before validation and output naming.
- **Rationale**: The current system already relies on resolved config values for output groups, monitoring, and checkpointing. Making granularity resolution part of that same step preserves local reasoning and keeps the resolved config as the single source of truth.
- **Alternatives considered**:
  - Resolve granularities lazily in training or model code. Rejected because it would split authority across multiple layers and make validation harder.
  - Require users to provide the resolved layout manually. Rejected because it duplicates work and increases configuration drift.

## Decision 3: Canonical mode remains a first-class compatibility path

- **Decision**: Keep the current `s`, `m`, `l`, `xl` contract as the canonical mode and preserve existing canonical prefix fractions and run behavior.
- **Rationale**: The spec explicitly requires backward compatibility, and the current tests and pilot scripts still assume canonical runs exist. Preserving this path avoids breaking older configs.
- **Alternatives considered**:
  - Replace canonical with explicit-only configuration. Rejected because it would break existing experiments and validation expectations.
  - Deprecate canonical immediately. Rejected because the feature non-goal is to remove canonical support.

## Decision 4: Explicit layouts must satisfy the same width rules the model already expects

- **Decision**: Validate explicit prefix fractions by checking positivity, strict increase, final full-width resolution, and variant-specific concat alignment before training begins.
- **Rationale**: The model math already depends on consistent nested widths. Validation should protect the existing implementation rather than changing the slicing or concat logic.
- **Alternatives considered**:
  - Auto-adjust fractions to fit alignment. Rejected because it would hide configuration intent and produce surprising runs.
  - Change the underlying FFN math to accept arbitrary misalignment. Rejected because the spec says not to redesign the math unless required.

## Decision 5: CLI and pilot entrypoints should consume resolved configuration, not hard-coded labels

- **Decision**: Update the pilot scripts and training wrappers so they either read the resolved granularity list from config or accept the new mode switch, instead of enumerating `s`, `m`, `l`, and `xl` directly.
- **Rationale**: The user-visible contract includes CLI and pilot runners, so they must follow the same resolved source of truth as the training loop.
- **Alternatives considered**:
  - Leave scripts hard-coded and ask users to update them manually. Rejected because it would undermine the feature and keep the canonical labels embedded in the surface area.
  - Introduce a new script family just for explicit layouts. Rejected because it would fragment the interface and increase maintenance.
