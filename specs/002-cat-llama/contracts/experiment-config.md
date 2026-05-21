# Contract: Experiment Configuration

The source experiment config stays YAML-based. The new model variant is exposed
as a config value so the cat-llama path can be selected without a separate
entry point.

## Author-Written Source YAML

```yaml
run:
  run_id: debug-nested-001
  phase_id: debug_matrix
  model_family: nested
  seed: 42
  output_root: outputs

model:
  base_model_name: debug-llama
  variant: cat_llama
  num_layers: 2
  num_attention_heads: 4
  hidden_size: 128
  intermediate_size: 512
  context_length: 64
  vocab_size_assumption: 32000
  granularities: [s, m, l, xl]

training:
  token_budget: 8192
  batch_size_per_process: 2
  learning_rate: 0.0003
  warmup_steps: 0
  eval_interval: 2
  mixed_precision: none
  activation_checkpointing: false
  granularity_sampling: all

dataset:
  dataset_name: roneneldan/TinyStories
  dataset_split: train
  dataset_phase: debug
  sample_limit: 16
  preprocessing_notes: truncate_or_pad_to_context_length_64_debug_samples
```

## Validation Rules

- `model.variant` must be one of `matformer_llama` or `cat_llama`.
- The default variant must remain `matformer_llama` when no override is
  provided.
- `model.variant` must not change the nested/standalone topology fields.
- Resolved configs must persist the selected variant in `config.json`.
- Config overrides must be sufficient to switch variants without changing the
  command path.
