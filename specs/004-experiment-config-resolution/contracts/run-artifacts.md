# Contract: Run Artifacts

Completed runs continue to write structured artifacts under the configured
output root. This feature changes the resolved folder key and the provenance
recorded in saved metadata, but not the file types themselves.

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

- `output_group` must resolve from the largest configured family size in the
  family, while preserving the family and token-budget components.
- Standalone `s`, `m`, and `l` runs that belong to the same family must share
  the same resolved folder key.
- `config.json` and `run_summary.json` must record the selected correction
  mode, the resolved family-folder rule, and the selected preset provenance.
- `metrics.csv` and `scaling_results.csv` must remain readable from the shared
  family folder without manual copying or renaming.
- `heartbeats.jsonl` must continue to record stage transitions so the run can
  be audited after a scheduler interruption.

## Comparison Workflow

- Figure generation should be able to scan the shared family folder directly.
- Downstream analysis should be able to distinguish active run size from family
  folder identity using saved metadata alone.
