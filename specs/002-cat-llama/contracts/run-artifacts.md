# Contract: Run Artifacts

Completed runs continue to write structured artifacts under the configured
output root. The cat-llama feature adds variant labels but keeps the same output
shape.

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

- `config.json` and `run_summary.json` must record the selected model variant.
- `metrics.csv` and `scaling_results.csv` must remain comparable across
  `matformer_llama` and `cat_llama` runs.
- Checkpoint status and checkpoint path fields must remain present in the run
  summary even when no checkpoint is written.
- Shared artifacts must continue to be written under the configured output
  root only.
