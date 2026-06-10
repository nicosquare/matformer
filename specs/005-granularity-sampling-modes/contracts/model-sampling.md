# Contract: Model Sampling Surface

This contract describes the configuration and runtime behavior that the model
sampling surface must expose after the refactor.

## Configuration Contract

- `model.granularity_sampling_mode`
  - Required values: `global`, `per_layer`
  - `global` means one granularity is selected for the entire forward pass.
  - `per_layer` means one granularity is selected per transformer block.
- `training.granularity_sampling`
  - Legacy compatibility input that resolves to the canonical mode.
  - `all` resolves to `global`.
  - `random` resolves to `per_layer`.
- `model.correction_mode`
  - Required values: `none`, `gmc`, `lmc`
  - The selected correction mode must remain explicit in config and saved
    artifacts.

## Runtime Behavior Contract

- Global mode must preserve the existing whole-model granularity behavior.
- Per-layer mode must produce a block-wise granularity pattern for each forward
  pass.
- Local GMC/LMC may be derived only when per-layer mode is active.
- The model must continue to run in single-process mode without distributed
  coordination for this feature.
- Legacy alias values must resolve before runtime behavior is selected.

## Validation Expectations

- The global path must be regression-tested against the current behavior.
- The per-layer path must be tested for independent layer selection and local
  correction activation.
- Invalid sampling-mode values must fail before training starts.
