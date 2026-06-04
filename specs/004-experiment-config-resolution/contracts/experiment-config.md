# Contract: Experiment Configuration Resolution

The experiment config remains YAML-based. This feature adds an explicit
correction mode, a shared-family folder rule, and named presets for reusable
sections.

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
  correction_mode: lmc
  membership_correction: true
  d_model: 128
  num_layers: 2
  num_attention_heads: 4
  context_length: 64
  vocab_size_assumption: 32000
  granularities: [s, m, l, xl]

training:
  token_budget: 8192
  batch_size_per_process: 2
  learning_rate: 0.0003
  optimizer:
    preset: adam
  scheduler:
    name: cosine

dataset:
  dataset_name: roneneldan/TinyStories
  dataset_split: train
  dataset_phase: debug
  sample_limit: 16
  preprocessing_notes: truncate_or_pad_to_context_length_64_debug_samples
```

## Preset Registry YAML

```yaml
# configs/presets/optimizer/adam.yaml
name: adamw
kwargs:
  betas: [0.9, 0.95]
  eps: 1.0e-8
  weight_decay: 0.1
```

```yaml
# configs/presets/optimizer/sgd.yaml
name: sgd
kwargs:
  momentum: 0.9
  dampening: 0.0
  nesterov: false
  weight_decay: 0.1
```

## Resolved Configuration Expectations

- `model.correction_mode` must resolve to one of `none`, `gmc`, or `lmc`.
- `model.correction_mode=lmc` must be rejected for non-concat runs.
- `model.membership_correction` must be the only accepted boolean correction
  input.
- `training.optimizer.preset` must select a named entry from
  `configs/presets/`.
- Explicit `training.optimizer` fields must override the preset values.
- The resolved config must record the preset name, the merged optimizer values,
  the resolved correction mode, and the family-folder rule.
- The resolved config must expose the shared output-group key used for run
  artifacts and figure generation.

## Validation Rules

- Unknown preset names must fail before training starts.
- Conflicting correction-mode inputs must fail before training starts.
- Nested preset values must merge deeply so partial overrides are preserved.
- The resolved output-group key must be deterministic for the same family
  definition.
- Preset registry files must be traceable from the resolved config.
