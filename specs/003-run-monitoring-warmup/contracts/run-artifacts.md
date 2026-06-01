# Contract: Run Artifacts

Completed runs continue to write structured artifacts under the configured
output root. The long-run support feature adds continuation and warmup metadata
but keeps the same output shape.

## Directory Layout

```text
<output_root>/<output_group>/<run_id>/
├── config.json
├── run_summary.json
├── metrics.csv
├── scaling_results.csv
├── checkpoints/
└── heartbeats.jsonl
```

## Validation Rules

- `config.json` and `run_summary.json` must record whether the run was fresh or
  resumed.
- `config.json` and `run_summary.json` must record the latest checkpoint path,
  continuation status, warmup policy, warmup completion state, and monitoring
  enablement.
- `metrics.csv` must continue to hold the same per-granularity scalar rows that
  the live dashboard uses.
- `heartbeats.jsonl` must continue to record stage transitions so preemption and
  warmup progress remain auditable.
- Shared artifacts must continue to be written under the configured output root
  only.
