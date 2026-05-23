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
  learning_rate_scale_rule: linear
  warmup_ratio: 0.03
  optimizer:
    name: adamw
    kwargs:
      betas: [0.9, 0.95]
      eps: 1.0e-8
      weight_decay: 0.1
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
- `training.learning_rate` is the author-written base learning rate and
  `training.learning_rate_scale_rule` must be one of `none`, `linear`, or
  `sqrt`.
- Learning-rate scaling must be resolved from the global batch size implied by
  `batch_size_per_process` and `effective_world_size`, then written back to the
  resolved config and run summary.
- `training.warmup_ratio` may be used on its own or with
  `training.warmup_steps`; if both are present, `warmup_steps` takes precedence.
- Warmup must be resolved against `training.max_steps`, not manually derived
  from `token_budget`.
- `training.optimizer.name` must be `adamw` or `sgd`.
- `training.optimizer.kwargs` must support at minimum AdamW `betas`, `eps`,
  and `weight_decay`, and SGD `momentum`, `dampening`, `nesterov`, and
  `weight_decay`.
- The resolved config and run summary must persist
  `base_learning_rate`, `learning_rate_scale_rule`,
  `learning_rate_scale_factor`, `resolved_learning_rate`, `warmup_ratio`,
  `warmup_steps`, `resolved_warmup_steps`, `optimizer_name`, and
  `optimizer_kwargs`.
