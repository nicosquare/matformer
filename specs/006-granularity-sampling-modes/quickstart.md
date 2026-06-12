# Quickstart: Granularity Operation Modes

This quickstart validates the canonical run modes and the nested-random
sampling submodes for the MatFormer training flow.

## 1. Prepare the Environment

Use the same Python environment as the existing MatFormer experiments. Set an
explicit output root if you want to keep the run artifacts outside the default
`outputs/` tree:

```bash
export OUTPUT_ROOT=/mnt/experiments/matformer
```

## 2. Run Focused Validation Tests

```bash
python -m pytest \
  tests/test_config.py \
  tests/test_matformer_prefixes.py \
  tests/test_artifacts.py \
  tests/test_training_smoke.py
```

Expected result:
- Config resolution accepts the canonical run modes.
- Granularity metadata remains consistent after the refactor.
- Artifact serialization records the resolved mode and pattern provenance.

## 3. Run a Nested-Random Global Smoke Example

Use the debug matrix and force the explicit global path:

```bash
python train.py \
  --config configs/debug_matrix.yaml \
  --run-id debug-nested-001 \
  --output-root "$OUTPUT_ROOT" \
  --override run.sampling_mode=nested-random \
  --override model.granularity_sampling_mode=global
```

Expected result:
- One granularity selection governs the whole forward pass for each iteration.
- The correction path matches the existing global behavior.
- Saved run metadata records the resolved run mode and the resulting pattern.

## 4. Run a Nested-Random Per-Layer Smoke Example

Use the same debug matrix and switch only the nested-random submode:

```bash
python train.py \
  --config configs/debug_matrix.yaml \
  --run-id debug-nested-001 \
  --output-root "$OUTPUT_ROOT" \
  --override run.sampling_mode=nested-random \
  --override model.granularity_sampling_mode=per_block
```

Expected result:
- Each transformer block receives its own granularity choice.
- Local GMC/LMC is derived from the sampled per-block pattern.
- Global correction is not activated in this mode.

## 5. Run a Nested-All Smoke Example

```bash
python train.py \
  --config configs/debug_matrix.yaml \
  --run-id debug-nested-001 \
  --output-root "$OUTPUT_ROOT" \
  --override run.sampling_mode=nested-all
```

Expected result:
- Every configured granularity is evaluated on every iteration.
- The training objective is the mean of the per-granularity losses.
- The saved pattern summary shows the full evaluated granularity set.

## 6. Run a Standalone Smoke Example

```bash
python train.py \
  --config configs/debug_matrix.yaml \
  --run-id debug-standalone-m-001 \
  --output-root "$OUTPUT_ROOT" \
  --override run.sampling_mode=standalone \
  --override run.model_family=standalone \
  --override run.granularity=m
```

Expected result:
- The chosen granularity remains fixed for the full run.
- The saved artifacts identify the standalone mode without consulting logs.
- The run stays within the supported fixed granularities `s`, `m`, `l`, and
  `xl`.

## 7. Inspect the Saved Metadata

Open the saved `config.json` and `run_summary.json` for one run from each mode.

Expected result:
- `run.sampling_mode` is recorded explicitly.
- `model.granularity_sampling_mode` is recorded explicitly.
- `model.correction_mode` is recorded explicitly.
- The saved summary includes a readable granularity-pattern description for the
  run.

## 8. Optional Follow-Up

After the smoke tests, run the broader debug matrix and d_model=256 pilot
configs with the same mode overrides to confirm the refactor does not change
the existing experiment flow.
