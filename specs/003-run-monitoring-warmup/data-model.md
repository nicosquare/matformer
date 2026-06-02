# Data Model: Long Run Support

## RunContinuationState

Represents the saved state needed to resume a run after a scheduler
interruption.

**Fields**
- `run_id`: Stable run identifier.
- `output_dir`: Resolved artifact directory for the run.
- `latest_checkpoint_path`: Location of the latest checkpoint available for
  resumption.
- `last_completed_step`: Highest fully completed training step.
- `tokens_seen`: Number of tokens consumed when the run last paused.
- `status`: `fresh`, `resumed`, `completed`, or `failed`.
- `resume_count`: Number of times the same run has been relaunched.

**Relationships**
- Attached to one `RunSummary`.
- Used by the shared training entry point to decide whether to resume or start
  from scratch.

**Validation Rules**
- `latest_checkpoint_path` must match the resolved output directory for the run.
- `resume_count` must be zero for a fresh run and increase only when the same
  run is relaunched after interruption.
- A resumed run must preserve the original `run_id`.

## MonitoringSeries

Represents one scalar series visible in the live monitoring dashboard.

**Fields**
- `series_name`: Dashboard label for the scalar series.
- `split`: Training or validation split.
- `granularity`: Active granularity represented by the series.
- `topology`: `nested` or `standalone`.
- `metric_name`: Usually `loss`, but may include other scalar metrics that
  belong to the same run view.
- `step`: Training step at which the measurement was emitted.
- `value`: Numeric metric value.

**Relationships**
- Derived from the same metric rows that populate `metrics.csv`.
- Grouped using the same nested-versus-standalone rules as the run's training
  trace and saved metrics.

**Validation Rules**
- Nested runs must produce one loss series per active granularity.
- Standalone runs must produce only the active standalone loss series.
- Empty placeholder series must not be emitted.

## WarmupPolicy

Represents the explicit pre-nested warmup configuration and its resolved state.

**Fields**
- `enabled`: Whether the warmup phase runs.
- `duration`: Numeric length of the warmup.
- `unit`: `epochs` or `steps`.
- `completed`: Whether warmup finished before the run ended.
- `completion_step`: Step or epoch at which warmup completed.
- `transition_reason`: Why the run moved from warmup into the nested phase.

**Relationships**
- Attached to `RunConfiguration` and copied into `RunSummary`.
- Controls the transition into the nested training phase.

**Validation Rules**
- `duration` must be positive when warmup is enabled.
- `unit` must be one of the supported duration units.
- Warmup applies only to nested runs; standalone runs bypass it.
- Warmup completion must be recorded before nested training begins.

## RunSummary

Represents the saved record of what ran, what resumed, and what artifacts were
produced.

**Fields**
- `run_id`
- `model_family`
- `model_variant`
- `continuation_state`
- `monitoring_enabled`
- `warmup_policy`
- `warmup_completion_step`
- `warmup_completed`
- `latest_checkpoint_path`
- `status`
- `output_dir`
- `metrics_path`
- `scaling_results_path`
- `checkpoint_status`
- `best_checkpoint_path`

**Relationships**
- Summarizes one `RunConfiguration`.
- Links the filesystem artifacts with the live monitoring run.

**Validation Rules**
- The summary must distinguish a fresh run from a resumed run.
- The summary must preserve the latest checkpoint pointer even when the run
  ends early.
- Warmup metadata must remain present even when warmup is disabled.
