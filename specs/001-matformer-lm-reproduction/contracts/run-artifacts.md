# Contract: Run Artifacts

Each completed run writes artifacts under `<output_root>/<run_id>/`.
`output_root` defaults to repository-local `outputs/`, but may point outside
the repository filesystem.

## Directory Layout

```text
<output_root>/<run_id>/
├── config.json
├── run_summary.json
├── heartbeats.jsonl
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
Long-running Slurm jobs must write `heartbeats.jsonl`; short local smoke runs
may omit it when heartbeat logging is disabled.

No required run artifact may be written under repository `outputs/` when the
researcher configures a different output root. Generated figure directories
should also live under the configured root unless the researcher explicitly
chooses another path.

For distributed runs, shared artifacts are written only by rank 0. This covers
`config.json`, CSV metrics, summaries, checkpoints, generated figures, and
`heartbeats.jsonl`. Nonzero ranks may emit process-local diagnostics to stdout
or stderr, but they must not race to write shared run artifacts.

## `metrics.csv`

Required columns:

```text
run_id,step,split,model_family,model_size_label,model_shape_label,sampling_mode,granularity,loss,perplexity,tokens_seen,content_tokens_seen,wall_clock_seconds,tokens_per_second,peak_memory_bytes
```

## `task_results.csv`

Required columns:

```text
run_id,suite_id,task,model_family,model_size_label,model_shape_label,sampling_mode,granularity,metric_name,metric_value
```

## `scaling_results.csv`

Required columns:

```text
comparison_id,run_id,model_family,model_size_label,model_shape_label,sampling_mode,completion_label,granularity,d_model,num_layers,num_attention_heads,context_length,vocab_size_assumption,token_budget,effective_world_size,total_parameters,embedding_parameters,lm_head_parameters,non_embedding_parameters,ffn_parameters,attention_parameters,other_non_embedding_parameters,lm_head_counting,checkpoint_path,loss,perplexity,average_downstream_accuracy
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
  "sampling_mode": "nested-all",
  "model_shape_label": "debug",
  "completion_label": "debug",
  "d_model": 256,
  "num_layers": 4,
  "num_attention_heads": 4,
  "context_length": 256,
  "vocab_size_assumption": 32000,
  "dataset_name": "tiny-stories",
  "dataset_split": "train",
  "token_budget": 1000000,
  "expected_tokens_per_step": 2048,
  "derived_max_steps": 489,
  "effective_world_size": 1,
  "tokens_seen": 1000000,
  "content_tokens_seen": 925000,
  "stop_reason": "token_budget_reached",
  "seed": 42,
  "status": "completed",
  "output_root": "/mnt/experiments/matformer",
  "output_dir": "/mnt/experiments/matformer/debug-nested-001",
  "parameter_counts": {
    "total_parameters": 123456,
    "embedding_parameters": 32000,
    "lm_head_parameters": 32000,
    "non_embedding_parameters": 59456,
    "ffn_parameters": 32768,
    "attention_parameters": 16384,
    "other_non_embedding_parameters": 10304,
    "lm_head_counting": "separately_counted"
  },
  "checkpoint_status": "best_eval",
  "best_checkpoint_path": "/mnt/experiments/matformer/debug-nested-001/checkpoints/best_eval.pt",
  "final_checkpoint_path": null,
  "checkpoint_metric": "validation_loss"
}
```

`expected_tokens_per_step`, `derived_max_steps`, and `effective_world_size` are
copied from the resolved `config.json`. `tokens_seen`, `content_tokens_seen`,
and `stop_reason` are runtime outcomes written by the training loop.
`tokens_seen` is the global budget counter based on planned token slots across
the effective world size. `content_tokens_seen` is the global non-padding
training-token counter observed from attention masks.

Distributed run summaries must also include active distributed context when a
run is launched with more than one process:

```json
{
  "distributed_strategy": "fsdp",
  "distributed_rank": 0,
  "distributed_local_rank": 0,
  "distributed_world_size": 2
}
```

These fields describe the writer process. Shared summaries are written by rank
0, so `distributed_rank` is expected to be `0` for shared `run_summary.json`.

## `heartbeats.jsonl`

Each line is one JSON object. Required fields:

```json
{
  "event_type": "heartbeat",
  "run_id": "dmodel256-pilot-comparison-001",
  "stage": "training",
  "rank": 0,
  "world_size": 2,
  "timestamp": "2026-05-15T12:00:00Z",
  "elapsed_seconds": 60.0,
  "step": 10,
  "derived_max_steps": 100,
  "tokens_seen": 81920,
  "content_tokens_seen": 74250,
  "token_budget": 1000000,
  "latest_loss": 1.25,
  "tokens_per_second": 512.0,
  "peak_gpu_memory_bytes": 123456,
  "eta_seconds": 120.0
}
```

`event_type` may be `stage_start`, `stage_complete`, or `heartbeat`.
Step-related and training-metric fields may be null for non-step stages such as
tokenizer loading, dataset loading, preprocessing, model initialization, FSDP
wrapping, validation, checkpointing, and artifact writing. Heartbeats emit when
either the configured step interval or elapsed-time interval is reached.

## Validation Rules

- Required scalar metrics must appear in CSV or JSON artifacts, not only logs.
- `config.json`, CSV metrics, summaries, checkpoints, and generated plots must
  be rooted under the configured output root unless explicitly overridden.
- Distributed shared artifacts must be written only by rank 0.
- Slurm heartbeat output must be available both as readable stdout lines and as
  `heartbeats.jsonl` under the run output directory.
- Plot files must list their source CSV files in `run_summary.json` or a report.
- Pilot artifacts must expose actual implementation counts for total,
  embedding, LM-head, non-embedding, and FFN parameters, plus attention and
  other non-embedding parameters when feasible.
- Pilot artifacts must state whether the LM head is tied, untied, excluded, or
  separately counted.
- Pilot comparison artifacts must expose model family, granularity, sampling
  mode, token budget, effective world size, and checkpoint path when available.
- `expected_tokens_per_step`, `derived_max_steps`, and `effective_world_size`
  must match the resolved `config.json` for the run.
- `tokens_seen` must report the global budget counter used for stopping. It is
  derived from planned token slots, effective world size, and completed steps,
  and should reach `token_budget` when the budget-derived step count is
  completed.
- `content_tokens_seen` must report global non-padding training tokens observed
  by the loop. This value may be lower than `tokens_seen` for padded datasets.
- `stop_reason` must make budget completion explicit. Allowed values are
  `not_started`, `token_budget_reached`,
  `max_steps_reached_before_token_budget`, and `failed`.
- A failed run may omit metrics, but must write a `run_summary.json` with
  `status=failed` and a short failure note when possible.
- Pilot runs with validation enabled must write a rank-0-safe best-eval
  checkpoint under `checkpoints/` and reference it from `run_summary.json`.
- Pilot runs with validation disabled must record `checkpoint_status=final` or
  `checkpoint_status=none` in `run_summary.json`.
