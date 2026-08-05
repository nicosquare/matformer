# Feature Specification: Configurable Granularity Modes

**Feature Branch**: `[009-configurable-granularity-modes]`  
**Created**: 2026-07-13  
**Status**: Draft  
**Input**: User description: "Feature: configurable granularity modes for MatFormer

I want to replace the hard-coded S/M/L/XL granularity contract with a configurable granularity mode system.

Current behavior:
- Granularities are effectively fixed to s, m, l, xl in config validation, CLI scripts, and documentation.
- The model code already works from ordered granularity prefixes, but the public config and validation layers still assume the canonical four labels.

Desired behavior:
- Add a config switch that selects how granularities are defined.
- Support at least two modes:
  - explicit: the config provides the exact ordered granularity labels and their prefix fractions
  - canonical: preserve the current s/m/l/xl behavior for backward compatibility
- Optionally support a proportional mode if it can be derived cleanly from config, but do not require it if explicit mode already solves the problem.

Functional requirements:
- In explicit mode, the config can define any ordered list of granularities, such as five levels with fractions [0.2, 0.4, 0.6, 0.8, 1.0].
- The resolved config must contain both:
  - model.granularities: the ordered granularity labels
  - model.granularity_prefixes: the resolved prefix fractions keyed by label
- Validation must ensure:
  - granularity labels are non-empty and ordered
  - prefix fractions are numeric, strictly increasing, and positive
  - the final granularity resolves to the full intermediate width
  - the resolved widths are valid for the selected model variant
- Existing canonical configs using s/m/l/xl must continue to work without behavior changes.
- The resolved config must remain the single source of truth after config resolution.
- The training loop, validation, checkpointing, reporting, and adaptive sampling code must use the resolved granularities list, not hard-coded labels.
- CLI scripts and pilot runners must stop assuming only s/m/l/xl and instead consume the resolved configuration or accept the new mode switch.
- Standalone runs must validate run.granularity against the resolved granularity list.
- For concat models, explicit granularity fractions must either satisfy the existing block-alignment constraints or fail fast with a clear validation error.

Non-goals:
- Do not change the underlying FFN slicing/concat math unless needed for validation compatibility.
- Do not redesign the training pipeline or experiment artifacts beyond granularity configuration.
- Do not remove canonical mode support."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Define custom granularity sets (Priority: P1)

As a researcher, I can define an explicit ordered granularity set with custom labels and prefix fractions so that I can run experiments with more than the canonical four levels.

**Why this priority**: This is the new capability the feature exists to unlock, and it is the only way to support arbitrary granularity counts.

**Independent Test**: Resolve a config that defines five ordered granularities with fractions `[0.2, 0.4, 0.6, 0.8, 1.0]`, then run training and confirm the resolved config and run artifacts use the custom labels and fractions.

**Acceptance Scenarios**:

1. **Given** an explicit-mode config with five ordered granularity labels and matching fractions, **When** the config is resolved, **Then** the resolved model config contains the ordered labels, the resolved prefix fractions keyed by label, and the derived widths for each granularity.
2. **Given** an explicit-mode config with valid custom labels, **When** training, validation, checkpointing, reporting, and adaptive sampling run, **Then** each component uses the resolved granularity list rather than a fixed label set.
3. **Given** a standalone run that selects one of the resolved explicit labels, **When** validation begins, **Then** the run is accepted only if the requested granularity appears in the resolved granularity list.

---

### User Story 2 - Preserve canonical compatibility (Priority: P2)

As a maintainer, I can keep using existing canonical S/M/L/XL configs without changing behavior so that older experiments and pilot scripts continue to work.

**Why this priority**: Backward compatibility protects the existing experiment surface and avoids breaking current configs while the new mode is introduced.

**Independent Test**: Resolve an existing canonical config that relies on the current four labels, then verify it still resolves to the same labels and fractions and continues to run successfully.

**Acceptance Scenarios**:

1. **Given** an existing config that uses the canonical mode, **When** it is resolved, **Then** it produces the current ordered labels `s`, `m`, `l`, `xl` with the current prefix fractions and no behavior change.
2. **Given** canonical model and run settings, **When** the training pipeline runs end to end, **Then** the resolved config remains the single source of truth and downstream stages do not need any special-case handling for legacy labels.
3. **Given** a saved resolved canonical config, **When** it is inspected later, **Then** it clearly records that canonical mode was selected and which labels and prefix fractions were used.

---

### User Story 3 - Reject malformed or incompatible layouts (Priority: P3)

As a researcher or operator, I get a clear failure when a granularity layout is malformed or incompatible so that I do not waste time on runs that cannot execute correctly.

**Why this priority**: Fast, clear validation prevents invalid experiments from reaching training and makes it obvious how to fix the configuration.

**Independent Test**: Try invalid explicit layouts, concat-incompatible fractions, and invalid standalone granularity requests, then confirm each fails before training starts with an error that names the problem.

**Acceptance Scenarios**:

1. **Given** explicit fractions that are not strictly increasing or contain non-positive values, **When** the config is resolved, **Then** validation fails with a message that identifies the malformed granularity definition.
2. **Given** an explicit layout whose last granularity does not resolve to the full intermediate width, **When** the config is resolved, **Then** validation fails before training begins.
3. **Given** a concat model whose explicit fractions do not satisfy the existing block-alignment constraints, **When** the config is resolved, **Then** validation fails fast with a clear alignment error.

### Edge Cases

- A custom explicit label may be longer or shorter than the canonical names as long as it is non-empty and ordered.
- A resolved explicit layout may contain more or fewer granularities than the canonical four, including the possibility of a single valid granularity.
- If the same prefix fraction is repeated or a later fraction rounds to a width that does not grow, validation must fail.
- A standalone run must reject a requested granularity even if it is spelled correctly when that label is absent from the resolved list.
- Concat-specific alignment failures must be reported before any training step is attempted.
- Canonical configs that omit the new mode switch should continue to behave exactly as they do today.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST expose an explicit granularity mode switch with at least `canonical` and `explicit` values.
- **FR-002**: Existing canonical configurations MUST continue to resolve to the current `s`, `m`, `l`, `xl` granularity order and current prefix fractions without observable behavior changes.
- **FR-003**: In explicit mode, the system MUST accept any non-empty ordered list of non-empty granularity labels paired with prefix fractions.
- **FR-004**: The resolved configuration MUST contain `model.granularities` as the authoritative ordered label list and `model.granularity_prefixes` as the authoritative label-to-fraction mapping.
- **FR-005**: Configuration resolution MUST happen before downstream validation and before any training, reporting, checkpointing, sampling, or CLI/pilot derivation uses the granularity set.
- **FR-006**: Granularity labels MUST be unique, non-empty, and preserved in the order provided by the resolved configuration.
- **FR-007**: Prefix fractions MUST be numeric, positive, and strictly increasing when read in the configured granularity order.
- **FR-008**: The final explicit granularity MUST resolve to the full intermediate width for the selected model variant.
- **FR-009**: Every resolved granularity width MUST be valid for the selected model variant, and invalid widths MUST fail validation before training begins.
- **FR-010**: For concat models, explicit granularity layouts MUST either satisfy the existing block-alignment constraints or fail fast with a clear validation error.
- **FR-011**: All training, validation, checkpointing, reporting, adaptive sampling, and monitoring behavior MUST derive active granularities from the resolved configuration rather than from hard-coded canonical labels.
- **FR-012**: CLI scripts and pilot runners MUST consume the resolved granularity configuration or accept the new mode switch so that custom ordered granularities do not require script edits.
- **FR-013**: Standalone runs MUST validate `run.granularity` against the resolved granularity list and reject any requested label that is not present.
- **FR-014**: Saved resolved configuration and run artifacts MUST record the selected granularity mode, the ordered labels, the resolved prefix fractions, and the derived widths.
- **FR-015**: Validation errors for malformed or incompatible granularities MUST identify the offending field or constraint clearly enough for a user to fix the config without inspecting internal code.

### Key Entities *(include if feature involves data)*

- **Granularity Mode**: The selected rule set that defines whether the configuration uses the canonical labels or an explicit custom ordering.
- **Resolved Granularity Set**: The ordered list of granularity labels plus their resolved prefix fractions and derived widths after configuration resolution.
- **Run Granularity Constraint**: The validation rule that determines whether a requested standalone granularity or model layout is compatible with the resolved set.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A five-granularity explicit config with fractions `[0.2, 0.4, 0.6, 0.8, 1.0]` resolves successfully and completes at least one smoke training run without manual code changes.
- **SC-002**: 100% of checked canonical configs still resolve to the same `s`, `m`, `l`, `xl` set and continue to run successfully.
- **SC-003**: 100% of malformed granularity definitions in the validation suite fail before training begins.
- **SC-004**: 100% of completed runs save the selected granularity mode, ordered labels, resolved prefix fractions, and derived widths in the resolved configuration or run artifacts.
- **SC-005**: 100% of standalone run requests are validated against the resolved granularity list rather than an assumed canonical label set.
- **SC-006**: 100% of concat configurations that violate the existing block-alignment constraints fail fast with a clear validation error.

## Assumptions

- Canonical mode remains the default path for existing configs that do not specify a granularity mode.
- Explicit labels are arbitrary non-empty strings, and the configured order is authoritative.
- The existing FFN slicing and concat math remains unchanged; only validation and configuration resolution are adjusted.
- Concat alignment checks reuse the current block-boundary rules already implied by the model variant.
- A proportional mode is optional and not required for this feature to be considered complete.
