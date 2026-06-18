# Feature Specification: Source Layout Housekeeping

**Feature Branch**: `008-src-layout-housekeeping`  
**Created**: 2026-06-18  
**Status**: Draft  
**Input**: User description: "Move all working code into src/, keep files under 500 lines where practical, and split the large training and figure-generation scripts into focused modules."

## Clarifications

### Session 2026-06-18

- Q: Which production areas are in scope for the housekeeping pass? → A: Limit this pass to `src/` relocation, thin wrappers, `training/run.py`, `scripts/make_figures.py`, `config`, and `utils/metrics.py`.
- Q: Which package layout should the `src/` migration use? → A: Preserve the current package names under `src/` and keep imports stable.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Find the real code in one place (Priority: P1)

As a contributor, I can find importable production code under a single source root, and the root of the repository only contains thin entrypoints and non-code assets.

**Why this priority**: A consistent source layout is the foundation for understanding the codebase and for future refactors.

**Independent Test**: Inspect the repository layout and verify that production imports resolve from the source package layout rather than from scattered root-level modules.

**Acceptance Scenarios**:

1. **Given** a fresh checkout, **When** a developer looks for importable production code, **Then** it is organized under `src/` rather than split across multiple root-level module trees.
2. **Given** the main CLI entrypoints, **When** they are opened, **Then** they delegate to package code instead of containing large amounts of business logic.
3. **Given** the repository root, **When** a developer scans it, **Then** they can distinguish entrypoints and support files from the actual implementation modules.

---

### User Story 2 - Understand and modify training flow in smaller pieces (Priority: P2)

As a maintainer, I can read the training pipeline without navigating a single monolithic file, and the training responsibilities are separated into smaller modules with clear names.

**Why this priority**: `training/run.py` is currently the main readability bottleneck and needs the first structural split.

**Independent Test**: Change a training concern such as checkpointing, sampler state, warmup, or dataloader orchestration and confirm the relevant code lives in a focused module rather than a giant all-in-one script.

**Acceptance Scenarios**:

1. **Given** the training pipeline code, **When** a maintainer looks for checkpoint or continuation logic, **Then** it is isolated from unrelated plotting, configuration, and CLI wiring.
2. **Given** the training pipeline code, **When** a maintainer looks for sampler or warmup logic, **Then** it is isolated from the top-level run orchestration.
3. **Given** the training pipeline code, **When** a maintainer updates one responsibility, **Then** the change does not require editing a multi-thousand-line file for unrelated concerns.

4. **Given** the shared configuration helpers, **When** a maintainer needs to adjust config resolution or config-driven layout decisions, **Then** that code is organized as part of the same housekeeping effort instead of remaining a separate oversized module.

---

### User Story 3 - Understand and modify figure generation in smaller pieces (Priority: P3)

As a maintainer, I can update plotting and report generation without editing a single oversized script, and the figure generator is broken into reusable helpers by concern.

**Why this priority**: `scripts/make_figures.py` is the other major readability bottleneck and is best handled as a focused follow-up to the training split.

**Independent Test**: Update one figure family or report path and confirm the relevant logic lives in a dedicated helper module rather than in a single large script.

**Acceptance Scenarios**:

1. **Given** the figure-generation code, **When** a maintainer inspects the plotting logic, **Then** CSV loading, metadata enrichment, plotting, and report writing are separated.
2. **Given** the figure-generation command, **When** it is run on existing artifacts, **Then** it produces the same class of outputs as before the refactor.
3. **Given** the figure-generation code, **When** a maintainer adds or changes one plot family, **Then** they can do so without touching unrelated plot families.

### Edge Cases

- Root-level wrappers must continue to work when invoked exactly as they are today from the repository root.
- Existing tests and scripts that import the current package names must keep working after the source relocation.
- Files that are already small enough should not be split just to satisfy the line-count goal.
- If a module is still above the target line count because it contains a single cohesive responsibility, the refactor should leave a short rationale and defer further splitting.
- Generated artifacts, logs, notebooks, and reference material must remain outside the source-layout move.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST place all importable production code under `src/` while preserving the current package names.
- **FR-002**: The system MUST keep root-level production entrypoints as thin wrappers around package code.
- **FR-003**: The system MUST preserve the current command-line interfaces for the training and figure-generation workflows.
- **FR-004**: The system MUST preserve the current artifact formats, output locations, and naming conventions used by the training and reporting workflows.
- **FR-005**: The system MUST split the training pipeline into smaller modules organized by responsibility, with `training/run.py` no longer acting as a single monolithic implementation file.
- **FR-006**: The system MUST split the figure-generation workflow into smaller modules organized by responsibility, with `scripts/make_figures.py` no longer acting as a single monolithic implementation file.
- **FR-007**: The system MUST keep the behavior of existing training and figure-generation workflows unchanged for representative runs except for the improved code organization.
- **FR-008**: The system MUST keep production modules in the refactor scope at or below 500 lines unless a documented exception is unavoidable, including shared config and metrics modules.
- **FR-009**: The system MUST preserve import compatibility for existing tests and internal modules that rely on the current package names.
- **FR-010**: The system MUST keep repository-specific non-source assets, including notebooks, logs, outputs, paper assets, and reference material, out of the source-layout move.
- **FR-011**: The system MUST add or update tests that verify the source layout, wrapper entrypoints, and unchanged representative behavior.

### Research & Experiment Requirements *(include for experiment-facing changes)*

- **EX-001**: The training workflow MUST remain runnable through the existing training command path after the refactor.
- **EX-002**: The figure-generation workflow MUST remain runnable through the existing figure-generation command path after the refactor.
- **EX-003**: Representative training and reporting runs MUST continue to produce the same classes of outputs and artifacts as before the refactor.
- **EX-004**: The refactor SHOULD reduce the number of places a contributor needs to inspect to understand training or plotting behavior.
- **EX-005**: The refactor SHOULD make it possible to locate each major responsibility in a focused module instead of a large all-purpose file, including shared configuration and metrics logic.

### Key Entities *(include if feature involves data)*

- **Source Package**: The importable production modules that live under `src/` after the refactor.
- **Entrypoint Wrapper**: A root-level script that forwards control to the package implementation while keeping the command interface stable.
- **Runtime Module**: A focused code unit that owns one training responsibility such as orchestration, checkpointing, or dataloading.
- **Figure Module**: A focused code unit that owns one plotting or report-generation responsibility.
- **Config Module**: A focused code unit that owns one shared configuration responsibility such as resolution, validation, or derived settings.
- **Metrics Module**: A focused code unit that owns one shared metrics or artifact-summary responsibility.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of importable production code is reachable from `src/` after the refactor.
- **SC-002**: 100% of root-level production entrypoints are thin wrappers.
- **SC-003**: `training/run.py`, `scripts/make_figures.py`, and the in-scope shared configuration and metrics modules are each decomposed into smaller focused modules, and no refactored production file in scope exceeds 500 lines.
- **SC-004**: A representative training run still resolves configuration, starts successfully, and writes the expected training artifacts after the layout change.
- **SC-005**: A representative figure-generation run still reads existing artifacts and writes the expected figures and reports after the layout change.
- **SC-006**: Existing tests that cover training, reporting, and import paths continue to pass after the refactor.

## Assumptions

- The current package names should remain stable unless a compatibility-preserving wrapper is clearly simpler.
- The refactor should prioritize the training pipeline and figure-generation workflow before any secondary cleanup.
- The line-count target is a strong preference, but the change should preserve behavior over forcing unnecessary splitting.
- Existing notebooks, logs, generated outputs, and reference material are out of scope for the source relocation.
- A thin wrapper at the repository root is acceptable if it preserves the current command-line interface.
