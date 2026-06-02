# Quickstart: Long Run Support

This quickstart describes the intended validation path for long-run
continuation, W&B monitoring, and the pre-nested warmup phase.

## 1. Prepare Environment

Use the same Python environment as the existing MatFormer experiments. If you
plan to enable live monitoring, make sure the W&B credentials for your account
or project are already configured.

Set an explicit output root when the local filesystem is constrained:

```bash
export OUTPUT_ROOT=/mnt/experiments/matformer
```

## 2. Run Focused Smoke Checks

```bash
python -m pytest tests/test_config.py tests/test_training_smoke.py tests/test_artifacts.py
```

Expected result:
- Config resolution accepts continuation, monitoring, and nested-only warmup
  overrides.
- A resumed run keeps the same `run_id` and output directory.
- The monitored nested path records the same per-run granularity series that
  appear in the training trace over steps or epochs.
- The warmup state is visible in the resolved summary artifacts.

## 3. Run a Baseline Nested Job

```bash
python train.py \
  --config configs/debug_matrix.yaml \
  --run-id debug-nested-001 \
  --output-root "$OUTPUT_ROOT"
```

Expected result:
- The run follows the existing MatFormer path.
- Resolved artifacts do not require continuation or W&B monitoring.

## 4. Run the Continuation and Monitoring Path

```bash
python train.py \
  --config configs/debug_matrix.yaml \
  --run-id debug-nested-001 \
  --output-root "$OUTPUT_ROOT" \
  --override run.continuation.enabled=true \
  --override monitoring.enabled=true \
  --override monitoring.backend=wandb \
  --override training.pre_nested_warmup.enabled=true \
  --override training.pre_nested_warmup.duration=2 \
  --override training.pre_nested_warmup.unit=epochs
```

Expected result:
- The same run can be relaunched after interruption without changing the
  command path.
- `run_summary.json` records the continuation state, warmup state, and
  monitoring enablement.
- W&B shows the loss series grouped by granularity for the single run.

### Warmup-Only Nested Example

Use the warmup knobs without enabling continuation when you just want to
verify the pre-nested transition on a nested run:

```bash
python train.py \
  --config configs/debug_matrix.yaml \
  --run-id debug-nested-001 \
  --output-root "$OUTPUT_ROOT" \
  --override training.pre_nested_warmup.enabled=true \
  --override training.pre_nested_warmup.duration=1 \
  --override training.pre_nested_warmup.unit=steps
```

Expected result:
- The nested run records a completed warmup before the main nested phase.
- Standalone runs still bypass warmup even if these overrides are supplied.
- The saved warmup metadata in `config.json` and `run_summary.json` stays
  explicit about the duration, unit, and transition reason.

## 5. Inspect Continuation Artifacts

After a relaunch, inspect the output directory:

```bash
tail -n 20 "$OUTPUT_ROOT"/<output_group>/<run_id>/heartbeats.jsonl
```

Expected result:
- Stage transitions include the warmup phase and the later nested-training
  phase.
- The latest checkpoint path in `run_summary.json` matches the resumed run.

## 6. Run the Same Path Under Slurm

```bash
sbatch scripts/slurm_debug_matrix.sh \
  --output-root "$OUTPUT_ROOT" \
  --override run.continuation.enabled=true \
  --override monitoring.enabled=true \
  --override monitoring.backend=wandb \
  --override training.pre_nested_warmup.enabled=true \
  --override training.pre_nested_warmup.duration=2 \
  --override training.pre_nested_warmup.unit=epochs
```

Expected result:
- The same configuration works from the Slurm wrapper.
- A job preempted by the scheduler can be relaunched to continue from the same
  run directory.
- Standalone jobs bypass warmup even if a warmup override is supplied.
