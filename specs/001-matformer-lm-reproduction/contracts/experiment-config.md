# Contract: Experiment Configuration

The reproduction uses simple YAML input configs. Each run saves a resolved
`config.json` with the same fields under `<output_root>/<run_id>/`, defaulting
to `outputs/<run_id>/`.

## Author-Written Source YAML

Source YAML files contain hand-authored experiment inputs only. Researchers
must not manually set `training.effective_world_size`,
`training.expected_tokens_per_step`, `training.derived_max_steps`, or the
resolved effective `training.max_steps`. When `run.sampling_mode` is present,
researchers also should not manually set `training.granularity_sampling`; those
fields are produced during config resolution and saved in `config.json`.

```yaml
run:
  run_id: debug-nested-001
  phase_id: debug_matrix
  model_family: nested
  sampling_mode: nested-all
  model_shape_label: debug
  completion_label: debug
  seed: 42
  output_root: outputs

model:
  base_model_name: debug-llama
  d_model: 256
  num_layers: 4
  num_attention_heads: 4
  intermediate_size: 1024
  context_length: 256
  vocab_size_assumption: 32000
  granularities: [s, m, l, xl]
  granularity_prefixes:
    s: 0.125
    m: 0.25
    l: 0.5
    xl: 1.0

training:
  token_budget: 1000000
  batch_size_per_process: 8
  max_steps_cap: null
  learning_rate: 0.0001
  warmup_steps: 50
  eval_interval: 100
  mixed_precision: bf16
  activation_checkpointing: true

parameter_reporting:
  lm_head_counting: separately_counted
  include_attention_parameters_when_feasible: true

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
  checkpoint_policy: best_eval_when_validation
  make_plots: true

evaluation:
  validation: true
  downstream_suite: []
  consistency: false
  speculative: false
```

`training.max_steps_cap` is optional. It is only a visible safety cap for
budgeted runs; omitting it means the resolver uses the token-budget-derived
step count.

## Resolved `config.json`

Resolved configs contain the source inputs plus defaults and derived fields.
Each run saves the resolved `config.json` under `<output_root>/<run_id>/`.
Budget-derived fields appear here, not in source YAML:

```yaml
run:
  run_id: debug-nested-001
  phase_id: debug_matrix
  model_family: nested
  sampling_mode: nested-all
  model_shape_label: debug
  completion_label: debug
  seed: 42
  output_root: outputs
  output_dir: outputs/debug-nested-001
  explicit_output_dir: false

training:
  token_budget: 1000000
  batch_size_per_process: 8
  effective_world_size: 1
  expected_tokens_per_step: 2048
  derived_max_steps: 489
  max_steps: 489
  max_steps_cap: null
  granularity_sampling: all

parameter_counts:
  total_parameters: 123456
  embedding_parameters: 32000
  lm_head_parameters: 32000
  non_embedding_parameters: 59456
  ffn_parameters: 32768
  attention_parameters: 16384
  other_non_embedding_parameters: 10304
  lm_head_counting: separately_counted
```

`run.output_dir` is derived as `<run.output_root>/<run.run_id>` unless an
explicit per-run output directory is provided. The derived value is saved in
the resolved `config.json`.

`training.token_budget` is the source of truth for budgeted run length. The
resolver writes `training.effective_world_size`,
`training.expected_tokens_per_step`, `training.derived_max_steps`, and the
effective `training.max_steps` into `config.json`. `training.max_steps` is the
step count used by the training loop, derived from token budget unless an
explicit safety cap is modeled with `training.max_steps_cap`.

`run.sampling_mode` is the comparison-facing source of truth. The resolver
derives internal `training.granularity_sampling=random` for `nested-random`,
`training.granularity_sampling=all` for `nested-all`, and
`training.granularity_sampling=all` for standalone baselines. Explicit
contradictory `training.granularity_sampling` overrides are rejected.

The derived `training.granularity_sampling` controls how nested MatFormer
subnetworks are trained. `random` samples one configured granularity per batch,
matching the original `train.py` behavior. `all` evaluates all configured
granularities on the same batch and averages their losses; this is useful for
debug and ablation runs but is not the original pilot training rule.

## Granularity Values

`granularities` uses canonical lowercase values: `s`, `m`, `l`, `xl`.

The corresponding display labels are `S`, `M`, `L`, `XL`.

## Completion Labels

- `debug`: Debug-size workflow validation.
- `reduced-token-pilot`: d_model=256 pilot run with less than the full
  10B-token budget.
- `full-token-budget`: d_model=256 pilot run using the full 10B-token budget.

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
- `run.sampling_mode` must be one of `nested-random`, `nested-all`, or
  `standalone` and must be consistent with `run.model_family`.
- `model_shape_label=dmodel256` and `training.token_budget < 10000000000`
  requires
  `completion_label=reduced-token-pilot`.
- `model_shape_label=dmodel256` and `training.token_budget = 10000000000`
  requires `completion_label=full-token-budget`.
- Pilot resolved configs must expose `d_model`, layer count, attention-head
  count, context length, vocabulary-size assumption, token budget, and
  granularity prefixes.
- Pilot resolved configs and run summaries must include `parameter_counts`
  fields for total, embedding, LM-head, non-embedding, and FFN parameters, plus
  attention and other non-embedding parameters when feasible.
- `parameter_counts.lm_head_counting` must state whether the LM head is tied,
  untied, excluded, or separately counted.
- `training.token_budget`, `training.batch_size_per_process`, and
  `model.context_length` must be positive integers for budgeted runs.
- `training.effective_world_size` must resolve from the active distributed
  `WORLD_SIZE` when distributed training is launched, otherwise 1. It must not
  be inferred from visible or allocated GPU count.
- `training.expected_tokens_per_step` must equal the product of
  `training.batch_size_per_process`, `model.context_length`, and
  `training.effective_world_size`.
- `training.derived_max_steps` must equal
  `ceil(training.token_budget / training.expected_tokens_per_step)`.
- `training.max_steps` in the resolved config must be the effective planned
  step count for the run. For budgeted runs it is derived from
  `training.token_budget`; any early safety cap must be explicit and visible in
  the resolved training section.
- Every run must write a resolved `config.json`.
- Pilot runs with validation enabled must use
  `outputs.checkpoint_policy=best_eval_when_validation` or an equivalent policy
  and record the best-eval checkpoint path in `run_summary.json`.
- Pilot runs without validation must record final-checkpoint or no-checkpoint
  status in `run_summary.json`.
