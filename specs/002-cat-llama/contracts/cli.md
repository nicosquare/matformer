# Contract: Command Interface

The researcher-facing entry point remains the existing training command. No new
script is required for the cat-llama variant.

## Run One Experiment

```bash
python train.py --config configs/debug_matrix.yaml --run-id debug-nested-001 --output-root /mnt/experiments/matformer --override model.variant=cat_llama
```

Required behavior:
- Resolve the existing config-driven training path.
- Accept `--override model.variant=cat_llama` as the selector for the new
  variant.
- Leave the default behavior unchanged when the override is omitted.
- Write the same run artifacts as the baseline path.
- Accept training overrides for `training.learning_rate_scale_rule`,
  `training.warmup_ratio`, `training.warmup_steps`, and
  `training.optimizer.{name,kwargs}` on the same command path so schedule and
  optimizer debugging do not require a separate script.
