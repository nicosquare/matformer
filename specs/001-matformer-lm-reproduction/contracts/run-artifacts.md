# Contract: Run Artifacts

Each completed run writes artifacts under `outputs/<run_id>/`.

## Directory Layout

```text
outputs/<run_id>/
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
  "tokens_seen": 1000000,
  "seed": 42,
  "status": "completed",
  "paper_aligned": false,
  "notes": []
}
```

## Validation Rules

- Required scalar metrics must appear in CSV or JSON artifacts, not only logs.
- Plot files must list their source CSV files in `run_summary.json` or a report.
- Any baseline mismatch must be recorded in `run_summary.json`.
- A failed run may omit metrics, but must write a `run_summary.json` with
  `status=failed` and a short failure note when possible.
