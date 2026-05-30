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
- `config.json` and `run_summary.json` must also record `base_learning_rate`,
  `learning_rate_scale_rule`, `learning_rate_scale_factor`,
  `resolved_learning_rate`, `warmup_ratio`, `warmup_steps`,
  `resolved_warmup_steps`, `gradient_clip_norm`, `optimizer_name`, and
  `optimizer_kwargs`.
- The resolved config must preserve the base learning rate, the scale rule, the
  scale factor, the warmup ratio, the resolved learning rate, and the resolved
  warmup step count so distributed runs remain auditable.
- `metrics.csv` and `scaling_results.csv` must remain comparable across
  `matformer_llama` and `cat_llama` runs.
- Checkpoint status and checkpoint path fields must remain present in the run
  summary even when no checkpoint is written.
- Shared artifacts must continue to be written under the configured output
  root only.
