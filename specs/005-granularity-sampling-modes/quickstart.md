# Quickstart: Granularity Sampling Modes

This quickstart validates the two supported sampling modes and the associated
correction behavior for the MatFormer model refactor.

## 1. Prepare the Environment

Use the same Python environment as the existing MatFormer experiments. Set an
explicit output root if you want to keep the run artifacts outside the default
`outputs/` tree:

```bash
export OUTPUT_ROOT=/mnt/experiments/matformer
```

## 2. Run Focused Validation Tests

```bash
python -m pytest tests/test_config.py tests/test_matformer_prefixes.py tests/test_training_smoke.py
```

Expected result:
- The config surface accepts the new model-level sampling mode.
- Granularity metadata remains consistent after the refactor.
- The current global path still behaves like the existing implementation.

## 3. Run a Global-Sampling Smoke Example

Use the debug matrix and force the explicit global path:

```bash
python train.py \
  --config configs/debug_matrix.yaml \
  --run-id debug-nested-001 \
  --output-root "$OUTPUT_ROOT" \
  --override model.granularity_sampling_mode=global
```

Expected result:
- One granularity selection governs the whole forward pass.
- The correction path matches the existing global behavior.
- Saved run metadata records the selected mode and the resulting pattern.

## 4. Run a Per-Layer Sampling Smoke Example

Use the same debug matrix and switch only the sampling mode:

```bash
python train.py \
  --config configs/debug_matrix.yaml \
  --run-id debug-nested-001 \
  --output-root "$OUTPUT_ROOT" \
  --override model.granularity_sampling_mode=per_block
```

Expected result:
- Each transformer block receives its own granularity choice.
- Local GMC/LMC is derived from the sampled per-layer pattern.
- Global correction is not activated in this mode.

## 5. Inspect the Saved Metadata

Open the saved `config.json` and `run_summary.json` for one run from each mode.

Expected result:
- `model.granularity_sampling_mode` is recorded explicitly.
- `model.correction_mode` is recorded explicitly.
- The saved summary includes a readable granularity-pattern description for the
  run.

## 6. Validate the Legacy Alias Path

Run a config that still uses the legacy sweep field and confirm it resolves into
the canonical mode:

```bash
python train.py \
  --config configs/debug_matrix.yaml \
  --run-id debug-nested-001 \
  --output-root "$OUTPUT_ROOT" \
  --override training.granularity_sampling=all
```

Expected result:
- The legacy field resolves through the compatibility alias.
- The resolved canonical mode is recorded in saved metadata alongside the
  requested alias.
- Downstream behavior uses the canonical model-level mode only.

## 7. Optional Follow-Up

After the smoke tests, run the broader debug matrix and d_model=256 pilot
configs with the same mode override to confirm the refactor does not change the
existing experiment flow.
