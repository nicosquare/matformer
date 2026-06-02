# Contract: Monitoring Output

The live dashboard is an external view of the same run metrics written to the
filesystem artifacts.

## Series Mapping

- Nested runs must log one loss series per active granularity.
- Standalone runs must log only the active standalone loss series.
- Series names must be readable as a single-run loss trace over training steps
  or epochs and should preserve the granularity labels used in the metrics
  artifacts.
- Validation-loss series should use the same granularity labels as the training
  loss series when they are emitted.

## Run Lifecycle Metadata

- Resumed runs must continue the same logical monitoring run rather than start
  a fresh dashboard entry for the same `run_id`.
- Monitoring should include enough run metadata to identify the topology,
  granularity, and warmup state.
- Disabling monitoring must not change the filesystem artifacts or the
  training outcome.

## Validation Rules

- Empty series must not be emitted for inactive granularities.
- The dashboard must remain useful for both nested and standalone runs without
  requiring a separate visualization script.
- Live monitoring must remain consistent with the saved CSV artifacts for the
  same run.
