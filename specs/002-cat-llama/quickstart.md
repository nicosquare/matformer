# Quickstart: Cat Llama Granularity Pipeline

This quickstart describes the intended validation path for the cat-llama
variant. It assumes the existing research environment and run-artifact layout.

## 1. Prepare Environment

Use the same Python environment as the existing MatFormer experiments.
Set an explicit output root when the local filesystem is constrained:

```bash
export OUTPUT_ROOT=/mnt/experiments/matformer
```

## 2. Run Focused Smoke Checks

```bash
python -m pytest tests/test_config.py tests/test_training_smoke.py tests/test_artifacts.py
```

Expected result:
- Config resolution accepts the new model variant override.
- Existing MatFormer runs still resolve unchanged when no override is used.
- The config-driven nested smoke path records `cat_llama` in `run_summary.json`.
- Config and artifact files can be written under the configured output root.

## 3. Run the Baseline Path

```bash
python train.py --config configs/debug_matrix.yaml --run-id debug-nested-001 --output-root "$OUTPUT_ROOT"
```

Expected result:
- The run follows the existing MatFormer path.
- Resolved artifacts record the default model variant.

## 4. Run the Cat Llama Path

```bash
python train.py --config configs/debug_matrix.yaml --run-id debug-nested-001 --output-root "$OUTPUT_ROOT" --override model.variant=cat_llama
```

Expected result:
- The same experiment path runs with the cat-llama variant selected through a
  configuration override.
- `config.json` and `run_summary.json` record the selected variant.
- Comparison artifacts remain in the same schema as the baseline run.
