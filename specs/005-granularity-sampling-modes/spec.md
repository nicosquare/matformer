# Feature Specification: Granularity Sampling Modes

**Feature Branch**: `005-granularity-sampling-modes`  
**Created**: 2026-06-09  
**Status**: Draft  
**Input**: User description: "Refactor the MatFormer model code into a clearer model-module structure and extend the granularity sampling surface so it supports multiple sampling modes. The current global granularity behavior should not be treated only as backward compatibility. It should become one explicit configuration of the sampling surface, alongside new per-layer sampling. In other words, the sampling API should be expanded so the existing behavior is preserved as a named/global mode, and the new behavior is introduced as an additional mode. The feature should support at least these sampling modes: global sampling: one granularity is chosen for the entire forward pass, matching the current behavior; per-layer sampling: one granularity is chosen per transformer block / FFN layer. The first implementation should be single-process only. Distributed training is out of scope for now. The model code should be reorganized so the current modified_llama.py logic becomes easier to understand and extend as the model behavior grows. The refactor should make the model responsibilities clearer, especially around: granularity metadata; FFN implementations; correction logic; model wiring / assembly. The correction logic should also be extended in the same spirit as the sampling surface. GMC/LMC behavior should be preserved, but adapted to the selected sampling mode: global sampling should continue to use the existing global correction behavior; per-layer sampling should derive a local GMC/LMC behavior from the sampled per-layer pattern; local GMC/LMC should only be activated when per-layer sampling is enabled. The goal is to keep the current behavior available as a first-class configuration while introducing per-layer sampling as a new experimental option. The implementation should remain configurable, explicit, and easy to extend for future sampling strategies."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Preserve the current global path (Priority: P1)

As a researcher, I can select a named global sampling mode and get the same whole-model granularity behavior that the project already supports today, without relying on implicit fallback behavior or legacy sweep names.

**Why this priority**: This preserves current experimental parity while making the existing behavior an explicit, supported configuration.

**Independent Test**: Configure a run for global sampling, execute a single forward pass, and verify that one granularity choice governs the entire pass and the correction behavior matches the current global path.

**Acceptance Scenarios**:

1. **Given** a single-process run with global sampling selected, **When** the model executes a forward pass, **Then** one granularity selection applies to the entire pass and the correction behavior follows the current global pattern.
2. **Given** an existing experiment that uses the legacy `training.granularity_sampling` field, **When** it is resolved through the compatibility alias, **Then** it maps to the canonical sampling mode and records both the requested alias and resolved mode in run metadata.

---

### User Story 2 - Sample granularity per layer (Priority: P2)

As a researcher, I can enable per-layer sampling to study how different transformer blocks behave when each block receives its own granularity choice.

**Why this priority**: This is the new experimental capability introduced by the feature and is the main reason for expanding the sampling surface.

**Independent Test**: Configure a multi-block model for per-layer sampling, execute a single forward pass, and verify that each block receives its own sampled granularity and that local correction is derived from that pattern.

**Acceptance Scenarios**:

1. **Given** per-layer sampling on a model with multiple transformer blocks, **When** the model executes a forward pass, **Then** each block receives an independently sampled granularity choice from the configured set.
2. **Given** per-layer sampling is enabled, **When** correction logic is applied, **Then** the model uses a local GMC/LMC interpretation derived from the sampled per-layer pattern.
3. **Given** global sampling is selected, **When** correction logic is applied, **Then** local GMC/LMC is not activated.

---

### User Story 3 - Keep model responsibilities clear (Priority: P3)

As a maintainer, I can reason about granularity metadata, FFN behavior, correction behavior, and model assembly as distinct responsibilities so that future sampling strategies can be added with less risk.

**Why this priority**: The refactor is meant to make the model easier to understand and extend, which reduces regression risk for future work.

**Independent Test**: Validate each model concern with targeted tests or focused configuration checks, without needing to exercise the full training workflow for every concern.

**Acceptance Scenarios**:

1. **Given** a change limited to granularity metadata, **When** it is validated, **Then** the FFN implementations and model wiring do not need to change for the check to pass.
2. **Given** a new sampling strategy is considered in the future, **When** the current structure is reviewed, **Then** global and per-layer behavior remain distinct and reusable.

### Edge Cases

- A model with only one transformer block still uses the selected sampling mode consistently and does not require special-case behavior.
- Per-layer sampling may produce the same granularity choice for every block in a given pass; that outcome still counts as per-layer sampling because the choices were made independently.
- Local correction must not appear in global mode, even if the sampled granularity value happens to match across layers.
- The feature remains single-process only; distributed execution is out of scope and does not need to be validated for this release.
- Legacy `training.granularity_sampling` inputs resolve through a compatibility alias rather than creating a separate execution path.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The model MUST expose sampling mode as an explicit configuration with at least `global` and `per_block` options.
- **FR-002**: Global sampling MUST select one granularity for the entire forward pass and apply it consistently to every eligible layer in that pass.
- **FR-003**: Per-layer sampling MUST select one granularity independently for each transformer block or FFN layer during a single forward pass.
- **FR-004**: The current global granularity behavior MUST remain available as a named, explicit mode rather than only as an implicit fallback.
- **FR-005**: Global sampling MUST preserve the current global correction behavior, including the existing GMC/LMC interpretation for that mode.
- **FR-006**: When per-layer sampling is enabled, the model MUST derive a local GMC/LMC interpretation from the sampled per-layer pattern.
- **FR-007**: Local GMC/LMC MUST be active only when per-layer sampling is selected and MUST not be applied when global sampling is selected.
- **FR-008**: Granularity metadata, FFN behavior, correction behavior, and model assembly MUST be separable enough that each concern can be validated independently.
- **FR-009**: The feature MUST run correctly in single-process mode for both supported sampling modes.
- **FR-010**: Each completed run MUST record the selected sampling mode and a summary of the resulting granularity choice pattern in saved run metadata.
- **FR-011**: The model MUST accept legacy `training.granularity_sampling` inputs as compatibility aliases and resolve them to the canonical sampling mode used by the rest of the system.
- **FR-012**: When a legacy alias is supplied, the system MUST record both the requested alias and the resolved canonical sampling mode in saved run metadata.

### Research & Experiment Requirements

- **EX-001**: The experiment surface MUST let a researcher choose global sampling or per-layer sampling without changing code.
- **EX-002**: Validation MUST include at least one regression case for the current global behavior and one case for per-layer sampling.
- **EX-003**: Saved run metadata MUST preserve the chosen sampling mode and the corresponding correction mode for later comparison.
- **EX-004**: The refactored structure SHOULD keep the four responsibilities named in the feature description distinct enough that future sampling modes can be added without rewriting the existing global or per-layer behavior.

### Key Entities *(include if data involved)*

- **Sampling Mode**: The explicit choice that determines whether one granularity is shared across the full pass or sampled separately per layer.
- **Granularity Pattern**: The resulting set of granularity choices made for a run, either a single global choice or a per-layer pattern.
- **Correction Behavior**: The selected GMC/LMC interpretation associated with the sampling mode and the sampled granularity pattern.
- **Model Responsibility Boundary**: The separation between granularity metadata, FFN behavior, correction behavior, and model assembly.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Global sampling reproduces the current whole-model granularity behavior in 100% of the checked regression runs.
- **SC-002**: Per-layer sampling produces independently sampled granularity choices for at least 2 transformer blocks in a multi-block validation run whenever the configured granularity space allows variation.
- **SC-003**: Local correction appears only in per-layer mode and never in global mode across 100% of the validated scenarios.
- **SC-004**: A single configuration change can switch between global and per-layer sampling without changing any other experiment setting.
- **SC-005**: 100% of completed runs record the chosen sampling mode and a granularity-pattern summary in saved metadata.
- **SC-006**: Validation artifacts cover both supported sampling modes and the corrected behavior for each mode before the feature is considered ready for planning.
- **SC-007**: 100% of supported legacy alias inputs resolve to the intended canonical sampling mode and are reflected in saved metadata.

## Assumptions

- Single-process execution is the only supported runtime for this feature release.
- Global sampling remains the explicit baseline configuration for parity and regression testing.
- Per-layer sampling uses the same underlying granularity choices that are already available to the current model behavior.
- No additional correction family beyond the current GMC/LMC behavior is introduced in this iteration.
- Existing run metadata can be extended to store the requested legacy alias, the resolved canonical sampling mode, and granularity-pattern summaries.
