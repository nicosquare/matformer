# Contract: Experiment Configuration

The reproduction uses simple YAML input configs. Each run saves a resolved
`config.json` with the same fields under `<output_root>/<run_id>/`, defaulting
to `outputs/<run_id>/`.

## Required Top-Level Fields

```yaml
run:
  run_id: debug-nested-001
  phase_id: debug_matrix
  model_family: nested
  model_size_label: debug
  completion_label: debug
  seed: 42
  output_root: outputs
  output_dir: outputs/debug-nested-001

model:
  base_model_name: debug-llama
  paper_aligned: false
  num_layers: 4
  num_attention_heads: 4
  hidden_size: 256
  intermediate_size: 1024
  context_length: 256
  vocab_size_assumption: 32000
  granularities: [s, m, l, xl]

training:
  token_budget: 1000000
  max_steps: 1000
  batch_size_per_process: 8
  learning_rate: 0.0001
  warmup_steps: 50
  eval_interval: 100
  mixed_precision: bf16
  activation_checkpointing: true

dataset:
  dataset_name: tiny-stories
  dataset_split: train
  dataset_phase: debug
  sample_limit: 10000
  preprocessing_notes: truncate_or_pad_to_context_length

outputs:
  save_config: true
  save_metrics_csv: true
  save_run_summary_json: true
  save_checkpoints: true
  make_plots: true

evaluation:
  validation: true
  downstream_suite: []
  consistency: false
  speculative: false
```

`run.output_dir` is derived as `<run.output_root>/<run.run_id>` unless an
explicit per-run output directory is provided. The derived value is saved in
the resolved `config.json`.

## Granularity Values

`granularities` uses canonical lowercase values: `s`, `m`, `l`, `xl`.

The corresponding display labels are `S`, `M`, `L`, `XL`.

## Completion Labels

- `debug`: Debug-size workflow validation.
- `reduced-token-pilot`: Paper-aligned model shape with less than the paper
  training-token budget.
- `paper-budget-complete`: Paper-aligned model shape with the paper
  training-token budget.

## Output Root and Cache Paths

- `run.output_root` is the preferred storage field and defaults to `outputs`.
- `run.output_dir` is normally derived from `run.output_root` and `run.run_id`.
- A direct `run.output_dir` value is an explicit escape hatch for one-off runs.
- Matrix configs should use one `run.output_root` so nested and standalone runs
  resolve to sibling directories.
- Researcher-facing runners should also accept `OUTPUT_ROOT` and output-root
  command arguments that map to `run.output_root`.
- Hugging Face model and dataset caches are controlled outside this YAML with
  `HF_HOME`, `HF_DATASETS_CACHE`, and `TRANSFORMERS_CACHE`.

## Validation Rules

- If neither `run.output_root` nor `run.output_dir` is set, `run.output_root`
  defaults to `outputs`.
- Unless explicitly overridden, `run.output_dir` resolves to
  `<run.output_root>/<run.run_id>`.
- `run.run_id` must match the final path segment of `run.output_dir`.
- The resolved output root must be created when missing and must be writable
  before training starts.
- `run.model_family=standalone` requires exactly one granularity.
- `run.model_family=nested` may include multiple granularities.
- `model.paper_aligned=true` requires 16 layers, 16 heads, context length 1024,
  and 256k vocabulary assumption unless the run is explicitly labeled
  non-paper-aligned in `run_summary.json`.
- `model_size_label=78m` and `training.token_budget < 10000000000` requires
  `completion_label=reduced-token-pilot`.
- `model_size_label=78m` and `training.token_budget = 10000000000` requires
  `completion_label=paper-budget-complete`.
- Every run must write a resolved `config.json`.
