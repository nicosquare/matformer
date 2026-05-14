# Contract: Run Artifacts

Each completed run writes artifacts under `<output_root>/<run_id>/`.
`output_root` defaults to repository-local `outputs/`, but may point outside
the repository filesystem.

## Directory Layout

```text
<output_root>/<run_id>/
├── config.json
├── run_summary.json
├── metrics.csv
├── task_results.csv
├── scaling_results.csv
├── consistency_results.csv
├── plots/
│   ├── loss_vs_size.png
│   ├── ppl_vs_size.png
│   └── consistency_vs_size.png
└── checkpoints/
    └── <checkpoint files>
```

Only artifacts relevant to the run phase are required. For example, a pure
validation run may omit `task_results.csv` and `consistency_results.csv`.

No required run artifact may be written under repository `outputs/` when the
researcher configures a different output root. Generated figure directories
should also live under the configured root unless the researcher explicitly
chooses another path.

## `metrics.csv`

Required columns:

```text
run_id,step,split,model_family,model_size_label,granularity,loss,perplexity,tokens_seen,wall_clock_seconds,tokens_per_second,peak_memory_bytes
```

## `task_results.csv`

Required columns:

```text
run_id,suite_id,task,model_family,model_size_label,granularity,metric_name,metric_value
```

## `scaling_results.csv`

Required columns:

```text
comparison_id,run_id,model_family,model_size_label,completion_label,granularity,total_parameters,embedding_parameters,lm_head_parameters,non_embedding_parameters,loss,perplexity,average_downstream_accuracy
```

## `consistency_results.csv`

Required columns:

```text
comparison_id,small_run_id,large_run_id,small_granularity,large_granularity,metric_name,metric_value,sample_count
```

## `run_summary.json`

Required fields:

```json
{
  "run_id": "debug-nested-001",
  "phase_id": "debug_matrix",
  "model_family": "nested",
  "model_size_label": "debug",
  "completion_label": "debug",
  "dataset_name": "tiny-stories",
  "dataset_split": "train",
  "token_budget": 1000000,
  "expected_tokens_per_step": 2048,
  "derived_max_steps": 489,
  "effective_world_size": 1,
  "tokens_seen": 1000000,
  "stop_reason": "token_budget_reached",
  "seed": 42,
  "status": "completed",
  "output_root": "/mnt/experiments/matformer",
  "output_dir": "/mnt/experiments/matformer/debug-nested-001",
  "paper_aligned": false,
  "notes": []
}
```

`expected_tokens_per_step`, `derived_max_steps`, and `effective_world_size` are
copied from the resolved `config.json`. `tokens_seen` and `stop_reason` are
runtime outcomes written by the training loop.

## Validation Rules

- Required scalar metrics must appear in CSV or JSON artifacts, not only logs.
- `config.json`, CSV metrics, summaries, checkpoints, and generated plots must
  be rooted under the configured output root unless explicitly overridden.
- Plot files must list their source CSV files in `run_summary.json` or a report.
- Any baseline mismatch must be recorded in `run_summary.json`.
- `expected_tokens_per_step`, `derived_max_steps`, and `effective_world_size`
  must match the resolved `config.json` for the run.
- `tokens_seen` must report actual non-padding training tokens observed by the
  loop, which may meet or slightly exceed `token_budget` because stopping occurs
  at batch boundaries.
- `stop_reason` must make budget completion explicit. Allowed values are
  `not_started`, `token_budget_reached`,
  `max_steps_reached_before_token_budget`, and `failed`.
- A failed run may omit metrics, but must write a `run_summary.json` with
  `status=failed` and a short failure note when possible.
