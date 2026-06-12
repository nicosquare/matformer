# Contract: Model Sampling Surface

This contract describes the configuration and runtime behavior that the model
sampling surface must expose after the run-mode refactor.

## Configuration Contract

- `run.sampling_mode`
  - Required values: `nested-random`, `nested-all`, `standalone`
  - Selects the top-level run behavior.
- `model.granularity_sampling_mode`
  - Required values: `global`, `per_block`
  - `global` means the current whole-model path.
  - `per_block` means one granularity is selected independently per
    transformer block.
- `model.correction_mode`
  - Required values: `none`, `gmc`, `lmc`
  - The selected correction mode must remain explicit in config and saved
    artifacts.

## Runtime Behavior Contract

- `nested-random + global` must select one granularity for the entire forward
  pass on a given iteration.
- `nested-random + per_block` must select one granularity per transformer block
  on a given iteration.
- `nested-all` must evaluate every configured granularity on every iteration
  and optimize the mean of the per-granularity losses.
- `standalone` must keep one fixed granularity for the full run.
- Local GMC/LMC may be derived only when the active sampling pattern is
  per-block.
- The model must continue to run in single-process mode without distributed
  coordination for this feature.

## Validation Expectations

- The canonical global path must remain regression-tested against the current
  behavior.
- The per-block path must be tested for independent block selection and local
  correction activation.
- `nested-all` must be tested for exhaustive evaluation and mean-loss
  aggregation.
- `standalone` must be tested for fixed granularity selection.
- Invalid sampling-mode or correction-mode combinations must fail before
  training starts.
