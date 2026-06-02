# Contract: Experiment Configuration

The source experiment config remains YAML-based. Long-run continuation,
monitoring, and warmup are exposed as configuration values so the feature can
be enabled without a separate entry point.

## Author-Written Source YAML

```yaml
run:
  run_id: debug-nested-001
  phase_id: debug_matrix
  model_family: nested
  seed: 42
  output_root: outputs
  continuation:
    enabled: true

model:
  base_model_name: debug-llama
  variant: matformer_llama
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
  gradient_clip_norm: 1.0
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
  pre_nested_warmup:
    enabled: true
    duration: 2
    unit: epochs

monitoring:
  enabled: true
  backend: wandb
  project: debug-matrix
  entity: research-team
  group: debug-nested-001
  job_type: train
  name: debug-nested-001
  tags: [debug, nested]
  notes: long-run smoke
  mode: online
  log_loss_by_granularity: true
  log_validation_loss: true
  log_stage_events: true

dataset:
  dataset_name: roneneldan/TinyStories
  dataset_split: train
  dataset_phase: debug
  sample_limit: 16
  preprocessing_notes: truncate_or_pad_to_context_length_64_debug_samples
```

## Validation Rules

- `run.continuation.enabled` must default to `false` when omitted and must not
  change the launch path unless explicitly enabled.
- Continuation must preserve the original `run_id` and resolved output
  directory.
- `monitoring.enabled` must be optional; disabling it must not change the saved
  CSV/JSON artifact schema.
- `monitoring.project`, `monitoring.entity`, `monitoring.group`,
  `monitoring.job_type`, `monitoring.name`, `monitoring.tags`, `monitoring.notes`,
  and `monitoring.mode` must be optional and may default from the resolved run
  metadata when omitted.
- `training.pre_nested_warmup.enabled` must be optional and default to `false`
  when omitted.
- `training.pre_nested_warmup.duration` must be positive when warmup is
  enabled.
- `training.pre_nested_warmup.unit` must be one of `epochs` or `steps`.
- Warmup must apply only to nested runs; standalone runs must bypass it.
- Warmup must be resolved before the nested phase starts, not after the run has
  already entered the nested block-splitting path.
- The resolved config must persist the continuation, monitoring, and warmup
  settings used for the run.
