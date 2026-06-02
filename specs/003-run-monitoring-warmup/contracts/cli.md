# Contract: Command Interface

The repository keeps the same researcher-facing entry points for long runs. No
new launcher script is required.

## Run a Config-Driven Experiment

```bash
python train.py \
  --config configs/debug_matrix.yaml \
  --run-id debug-nested-001 \
  --output-root /mnt/experiments/matformer \
  --override run.continuation.enabled=true \
  --override monitoring.enabled=true \
  --override monitoring.backend=wandb \
  --override training.pre_nested_warmup.enabled=true \
  --override training.pre_nested_warmup.duration=2 \
  --override training.pre_nested_warmup.unit=epochs
```

Required behavior:
- Use the existing config-driven training path.
- Resume a relaunched run from the same resolved output directory when
  continuation is enabled and prior checkpoints exist.
- Keep the default behavior unchanged when continuation and monitoring are not
  enabled.
- Apply warmup only to nested runs; standalone runs bypass warmup.
- Accept the same overrides from the Slurm wrappers and direct `train.py`
  launches.
- Preserve the same run entry point for nested and standalone experiments.

## Run a Slurm-Backed Experiment

```bash
sbatch scripts/slurm_debug_matrix.sh \
  --output-root /mnt/experiments/matformer \
  --override run.continuation.enabled=true \
  --override monitoring.enabled=true \
  --override monitoring.backend=wandb \
  --override training.pre_nested_warmup.enabled=true \
  --override training.pre_nested_warmup.duration=2 \
  --override training.pre_nested_warmup.unit=epochs
```

Required behavior:
- Pass the same config overrides through the Slurm wrapper.
- Keep the command path identical to the non-Slurm path except for scheduler
  submission.
- Resume the same run directory when a job is relaunched after preemption.
- Standalone jobs bypass warmup even if a warmup override is supplied.
