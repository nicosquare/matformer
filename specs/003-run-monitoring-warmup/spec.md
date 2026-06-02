# Feature Specification: Long Run Support

**Feature Branch**: `003-run-monitoring-warmup`  
**Created**: 2026-06-01  
**Status**: Draft  
**Input**: User description: "Add support for long-running runs that can survive slurm time limits, report granularity losses to wandb, and include a configurable warmup phase before nested block slicing/splitting"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Continue Long Runs Across Scheduler Limits (Priority: P1)

As a researcher, I want a run that can continue after a scheduler time limit so that increasing the token budget does not force me to restart the experiment from scratch.

**Why this priority**: Long runs are blocked if they cannot survive a single scheduler window, so this capability is the foundation for the rest of the feature.

**Independent Test**: Start a run whose budget exceeds one scheduler allocation, interrupt it at the scheduler limit, relaunch the same resolved run, and confirm it resumes from the latest saved progress instead of repeating completed work.

**Acceptance Scenarios**:

1. **Given** a run reaches the scheduler time limit before finishing, **When** the same run is launched again with the same resolved configuration, **Then** it resumes from the latest saved progress and continues toward the original budget.
2. **Given** a run completes before any scheduler interruption, **When** the run is inspected afterward, **Then** it is recorded as a normal completed run with no continuation step required.

---

### User Story 2 - Monitor Losses by Granularity in Weights & Biases (Priority: P2)

As a researcher, I want the run monitoring view to show the same per-run loss-by-granularity information that I inspect during training so that I can follow progress without leaving the experiment dashboard.

**Why this priority**: Monitoring is useful only if it reflects the same per-run comparison surface used for debugging and later analysis.

**Independent Test**: Run one nested experiment and one standalone experiment with monitoring enabled, then verify that the dashboard shows the expected loss series for each topology over training steps or epochs.

**Acceptance Scenarios**:

1. **Given** a nested run with multiple active granularities, **When** monitoring is enabled, **Then** the dashboard shows a separate loss series for each active granularity.
2. **Given** a standalone run with one active granularity, **When** monitoring is enabled, **Then** the dashboard shows only the standalone loss series and no empty granularity series.

---

### User Story 3 - Warm Up Before Nested Splitting (Priority: P3)

As a researcher, I want a short configurable warmup phase before nested block slicing or splitting begins so that nested training starts from better weights.

**Why this priority**: This improves the quality of the nested phase, but it is only valuable after long-run continuation and monitoring are in place.

**Independent Test**: Run a nested experiment with warmup enabled, verify that the warmup stage completes first, and confirm that the nested phase starts from the warmed state rather than a fresh initialization.

**Acceptance Scenarios**:

1. **Given** a nested run with warmup enabled, **When** the warmup duration is reached, **Then** the run transitions into nested training from the warmed state.
2. **Given** a run with warmup disabled, **When** the run starts, **Then** it proceeds directly to the configured training path without a warmup stage.
3. **Given** a warmup configuration that is longer than the remaining budget, **When** the run ends before nested training begins, **Then** the run finishes cleanly and records that nested training was not reached.

### Edge Cases

- What happens when a run is interrupted during warmup rather than during nested training?
- How does the system behave when monitoring is enabled but only one granularity is active?
- What happens when a warmup is configured in epochs for one run and in optimization steps for another?
- How is a resumed run labeled when it required multiple scheduler allocations to finish?
- What happens when a warmup override is provided for a standalone run?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The feature MUST behave consistently across all supported run entry points, so the same resolved configuration produces the same long-run, monitoring, and warmup behavior regardless of how the run is launched.
- **FR-002**: The run configuration MUST support a continuation-capable mode for runs whose token budget is expected to exceed a single scheduler allocation.
- **FR-003**: When a run is interrupted by the scheduler time limit, relaunching the same resolved run MUST resume from the latest saved progress rather than restarting completed work.
- **FR-004**: The saved run summary MUST record whether the run completed in one allocation or across multiple allocations, and MUST preserve the final continuation state.
- **FR-005**: When monitoring is enabled, the system MUST publish run progress and loss metrics to Weights & Biases.
- **FR-006**: The monitored loss view MUST expose the same per-run loss-by-granularity series written to the run's metrics artifacts.
- **FR-007**: Nested runs MUST report one loss series per active granularity.
- **FR-008**: Standalone runs MUST report only the active standalone loss series and MUST NOT emit empty placeholder series for inactive granularities.
- **FR-009**: The configuration MUST support a warmup phase that runs before nested block slicing or splitting begins.
- **FR-010**: The warmup configuration MUST support enabling or disabling the warmup phase, selecting its duration, and choosing its duration unit.
- **FR-011**: The warmup configuration MUST support at least two duration units: epochs and optimization steps.
- **FR-012**: The warmup configuration MUST apply only to nested runs, and standalone runs MUST bypass warmup even when warmup is configured.
- **FR-013**: After warmup completes, nested training MUST begin from the warmed state rather than from a fresh initialization.
- **FR-014**: Resolved run artifacts MUST record the warmup settings, the resolved warmup duration, and whether the run reached the nested phase.

### Research & Experiment Requirements

- **EX-001**: A long-running validation case MUST be able to resume after a scheduler cutoff and complete without redoing completed training work.
- **EX-002**: A monitored nested run MUST expose the same granularity loss series in the dashboard that the run's training trace uses for analysis.
- **EX-003**: A monitored standalone run MUST expose only the standalone loss series and remain directly comparable to the nested loss view where applicable.
- **EX-004**: A warmup-enabled nested run MUST persist the resolved warmup plan and the transition into nested training in its saved artifacts.
- **EX-005**: The feature MUST remain usable from every supported run entry point without forcing a different launch workflow.

### Key Entities *(include if feature involves data)*

- **Run Continuation**: The saved state that allows a run to resume after a scheduler interruption without losing completed progress.
- **Monitoring Series**: The set of loss signals that appear in the experiment dashboard for nested and standalone runs.
- **Warmup Policy**: The resolved warmup settings that determine whether warmup runs and how long it runs for nested training.
- **Granularity Loss View**: The comparison surface that determines which loss series should be visible for a given run topology.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In validation testing, 100% of interrupted long-run cases resume from the latest saved progress when relaunched with the same resolved run.
- **SC-002**: For every monitored nested run, the dashboard contains one loss series per active granularity; for every monitored standalone run, it contains only the active standalone series.
- **SC-003**: In validation testing, 100% of warmup-enabled nested runs record a warmup completion point before the nested phase begins.
- **SC-004**: Users can enable long-run continuation, monitoring, and warmup from the existing run entry points without changing the launch path.
- **SC-005**: Completed runs that require more than one scheduler allocation remain comparable to single-allocation runs using the same saved run summary and dashboard loss series.

## Assumptions

- Scheduler time limits or preemption can interrupt a run, and relaunching the same resolved run is the expected recovery path.
- Weights & Biases is the standard monitoring destination for this feature.
- The loss-by-granularity definitions used during a run are the source of truth for what should appear in monitoring.
- Warmup is intended only for nested runs; standalone runs bypass warmup even if a warmup override is present.
- Existing datasets, evaluation logic, and model choices remain unchanged.
