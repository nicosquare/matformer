# Feature Specification: Experiment Config Resolution

**Feature Branch**: `004-experiment-config-resolution`  
**Created**: 2026-06-03  
**Status**: Draft  
**Input**: User description: "Add LMC for concat runs, fix shared artifact folder resolution for standalone family runs, and add modular config presets starting with optimizer presets."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Select Correction Mode for Concat Runs (Priority: P1)

As a researcher, I want concat runs to support `none`, `gmc`, and `lmc` correction modes so that I can compare the existing gradient correction behavior with the new learning-rate-based correction behavior without changing the rest of the experiment setup.

**Why this priority**: This is the only new training behavior in the feature set, and it is the main experiment-facing change for concat runs.

**Independent Test**: Run a synthetic concat model with known membership counts, execute one optimizer step under each correction mode, and verify that only `lmc` changes the effective learning rate while leaving gradients and optimizer state untouched.

**Acceptance Scenarios**:

1. **Given** a concat run with active granularities `s`, `m`, and `l`, **When** `correction_mode=lmc`, **Then** each concat block uses `effective_lr = base_lr * (total_active_losses / membership_count)` during the optimizer step.
2. **Given** the same concat run, **When** `correction_mode=gmc`, **Then** the existing gradient-membership correction behavior remains in effect and the learning rate is not block-specific.
3. **Given** the same concat run, **When** `correction_mode=none`, **Then** no membership correction is applied.
4. **Given** a slicing-model run, **When** `correction_mode=lmc`, **Then** the run is rejected with a clear message that LMC is only supported for concat runs in v1.

---

### User Story 2 - Resolve Runs Into Shared Family Artifact Folders (Priority: P2)

As a researcher, I want standalone `s`, `m`, and `l` runs to resolve into the shared family folder for the largest model size in the comparison set so that plots and comparisons can be generated directly without copying or renaming artifacts.

**Why this priority**: This removes the manual post-processing step that currently makes standalone runs awkward to compare with the rest of the experiment family.

**Independent Test**: Resolve the same family config for standalone `s`, `m`, and `l` runs, confirm that they share the same folder key, and verify that downstream figure generation can read that folder directly.

**Acceptance Scenarios**:

1. **Given** a family config whose resolved granularity list includes `s`, `m`, `l`, and `xl`, **When** standalone `s`, `m`, and `l` runs are resolved, **Then** they all use the shared family folder associated with the largest configured size in that family.
2. **Given** the same resolved run config, **When** it is resolved again, **Then** the folder name and artifact paths are identical.
3. **Given** a saved run summary, **When** the run is inspected later, **Then** it shows both the active model size and the family folder rule that was used to place the run.
4. **Given** the downstream figure workflow, **When** it is pointed at the resolved family folder, **Then** it can consume the run outputs directly without manual copying or renaming.

---

### User Story 3 - Reuse Config Sections With Presets (Priority: P3)

As a researcher, I want recurring config blocks such as optimizer settings to be defined once and referenced by name so that I can keep experiment configs shorter, more consistent, and easier to update.

**Why this priority**: This improves maintainability and reduces duplication, but it does not change the experiment results as directly as correction behavior or artifact placement.

**Independent Test**: Resolve a config that selects an optimizer preset and overrides one nested field, then verify that the preset values remain in effect except for the explicit override.

**Acceptance Scenarios**:

1. **Given** `training.optimizer.preset: adam`, **When** the config is resolved, **Then** the current recommended Adam settings are applied.
2. **Given** `training.optimizer.preset: adam` and `training.optimizer.learning_rate: 2e-4`, **When** the config is resolved, **Then** the explicit learning rate overrides the preset value while the remaining preset values stay in effect.
3. **Given** an invalid preset name, **When** the config is resolved, **Then** validation fails before training starts and reports that the preset name is unknown.
4. **Given** future preset-enabled sections, **When** they are added later, **Then** they can follow the same section-scoped preset pattern without changing the training workflow that consumes the resolved config.

### Edge Cases

- What happens when `correction_mode=lmc` is selected for a non-concat model path? The run should fail fast rather than silently falling back.
- What happens when legacy membership-correction settings and the new correction-mode field disagree? The resolved config should reject the conflict instead of guessing.
- What happens when a standalone run belongs to the same comparison family but uses a smaller active size? It should still resolve to the shared family folder, while the active size remains visible in metadata.
- What happens when the family comparison set does not include `xl`? The folder key should use the largest granularity actually present in the resolved family list.
- What happens when a preset supplies nested defaults and the config overrides only one nested field? The override should merge into the preset rather than replacing the whole section.
- What happens when a preset name is missing or duplicated? Validation should fail with a clear message before any training work starts.
- What happens when figure generation sees historical runs from the old folder naming scheme? Those runs remain readable by their saved metadata, while new runs use the resolved family folder naming.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The experiment configuration MUST support a single resolved correction mode with the values `none`, `gmc`, and `lmc`.
- **FR-002**: The correction mode MUST be recorded in the resolved config and saved run summary for every run.
- **FR-003**: LMC MUST reuse the same membership-count logic already used for GMC scale derivation.
- **FR-004**: For concat runs, LMC MUST compute a correction multiplier from the membership count and apply it only at optimizer-step time, using the scheduler-provided base learning rate as the starting point.
- **FR-005**: LMC MUST leave gradients unchanged and MUST NOT modify optimizer state or moment buffers.
- **FR-006**: LMC MUST be limited to the concat model path in v1; the slicing path MUST remain unchanged unless shared plumbing is explicitly required by a later design.
- **FR-007**: GMC and LMC MUST be mutually exclusive in resolved config and runtime behavior.
- **FR-008**: The resolved config MUST make it clear which correction mode was selected and which model path it applies to.
- **FR-009**: Artifact and output naming MUST be resolved from the largest configured family size in the comparison family, while preserving the existing family and token-budget components of the folder key.
- **FR-010**: Standalone `s`, `m`, and `l` runs that belong to the same comparison family MUST resolve into the same shared family folder when their family definition is otherwise identical.
- **FR-011**: The resolved family folder rule MUST be deterministic and MUST be visible in saved run metadata.
- **FR-012**: All run outputs that are consumed for later comparison, including config snapshots, run summaries, metrics, checkpoints, scaling outputs, and figure inputs, MUST be written under the resolved family folder.
- **FR-013**: The config system MUST support section-scoped named presets, with optimizer presets introduced first.
- **FR-014**: Optimizer presets MUST be defined under a preset registry for the optimizer section, and the optimizer section MUST select a preset by name using `training.optimizer.preset`.
- **FR-015**: Preset resolution MUST happen during config loading before the final resolved config is used by training.
- **FR-016**: Preset merge precedence MUST be base defaults first, then preset values, then explicit config values, then CLI overrides.
- **FR-017**: Preset resolution MUST deep-merge nested mappings so that explicit overrides replace only the targeted nested fields.
- **FR-018**: Only one preset MAY be selected for a given section in v1.
- **FR-019**: Invalid preset names and conflicting preset combinations MUST fail validation with clear, user-facing messages.
- **FR-020**: The resolved config and run summary MUST record both the selected preset name and the final merged values that were applied.
- **FR-021**: Preset definitions MUST remain configuration-driven so that adding a new preset does not require changing the training workflow that consumes the resolved config.

### Research & Experiment Requirements

- **EX-001**: A synthetic concat test MUST demonstrate that `lmc` changes block update magnitudes according to membership count while gradients and optimizer state remain unchanged.
- **EX-002**: At least one real concat experiment configuration MUST run with `lmc` enabled and produce the expected per-block effective learning-rate behavior in saved metadata or test assertions.
- **EX-003**: A standalone family test MUST show that `s`, `m`, and `l` runs resolve into the same shared family folder when they belong to the same comparison family.
- **EX-004**: A figure-generation smoke test MUST be able to read the new shared family folder structure without manual copying, renaming, or post-processing.
- **EX-005**: An optimizer preset test MUST verify that a named preset resolves to the expected default values and that a partial override only changes the intended nested field.
- **EX-006**: Every completed run MUST save enough metadata to explain the selected correction mode, the folder-resolution rule, and any preset selected for the run.

### Key Entities *(include if feature involves data)*

- **Correction Mode**: The resolved membership-correction behavior for a run, with the allowed values `none`, `gmc`, and `lmc`.
- **Concat Block**: A unit of concat-model parameters that receives a block-specific effective learning rate when LMC is active.
- **Family Artifact Group**: The shared output-folder identity used to keep related standalone and family runs together for comparison.
- **Preset Registry Entry**: A named reusable config block that can be selected from a section and merged into the final resolved config.
- **Resolved Run Metadata**: The saved config and run summary fields that explain which correction mode, folder rule, and preset values were used.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In validation testing, 100% of concat runs with `correction_mode=lmc` use block-specific effective learning rates derived from membership counts, while gradients and optimizer state remain unchanged.
- **SC-002**: In validation testing, 100% of standalone `s`, `m`, and `l` runs that belong to the same family resolve to the same shared folder across reruns.
- **SC-003**: In validation testing, downstream figure generation can consume the new shared family folder directly with zero manual copying or renaming steps.
- **SC-004**: In validation testing, 100% of runs that select a valid optimizer preset and override one nested field resolve to the expected merged config values and record the preset name in saved metadata.
- **SC-005**: In validation testing, invalid correction modes, unsupported correction-mode/model-path combinations, and invalid preset names fail before training starts with a clear error message.
- **SC-006**: Every completed run exposes correction mode, active model size, family folder resolution, and preset provenance in both the resolved config and run summary.

## Assumptions

- The existing membership-count logic used for GMC scale derivation is the source of truth for LMC.
- LMC is scoped to the concat model path in v1, and the slicing path remains unchanged unless a later design introduces shared plumbing.
- The family comparison folder is derived from the resolved ordered granularity list already present in the config, with the largest configured family size serving as the shared folder key.
- The existing token-budget component of the folder key remains in place; only the size component changes when a family comparison folder is resolved.
- Preset names are section-scoped and configuration-driven, with optimizer presets stored as separate YAML registry files under `configs/presets/` and selected via `training.optimizer.preset`.
- The current output file formats remain unchanged; the feature changes where they are rooted and what provenance they record, not the contents of the metrics or checkpoint files themselves.
- Presets are selected one at a time per section in v1, and the optimizer preset registry is the first supported preset namespace.
