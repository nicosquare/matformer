# Quickstart: Experiment Config Resolution

This quickstart validates the three parts of the feature: concat LMC, shared
family-folder resolution, and config presets loaded from separate YAML files
under `configs/presets/`.

## 1. Prepare the Environment

Use the same Python environment as the existing MatFormer experiments. If you
plan to inspect live monitoring or write outputs outside the default root, set
an explicit output directory first:

```bash
export OUTPUT_ROOT=/mnt/experiments/matformer
```

## 2. Run Focused Validation Tests

```bash
python -m pytest tests/test_config.py tests/test_artifacts.py tests/test_training_smoke.py
```

Expected result:
- Config resolution accepts `correction_mode`, shared-family folder rules, and
  optimizer presets.
- Saved config and summary artifacts include the selected correction mode and
  preset provenance.
- Artifact placement stays deterministic across reruns.
- Preset definitions are loaded from `configs/presets/` rather than inline in
  the experiment config.

## 3. Run a Concat LMC Smoke Example

Use a concat-capable run and select the new correction mode:

```bash
python train.py \
  --config configs/debug_matrix.yaml \
  --run-id debug-nested-001 \
  --output-root "$OUTPUT_ROOT" \
  --override model.variant=cat_llama \
  --override model.correction_mode=lmc \
  --override training.optimizer.preset=adam
```

Expected result:
- The run resolves with `correction_mode=lmc`.
- Concat blocks use block-specific effective learning rates.
- Gradients and optimizer moments remain unchanged by LMC.
- The saved run metadata records the selected preset and family-folder rule.

## 4. Validate Shared Family Folder Resolution

Run the same comparison family for standalone `s`, `m`, and `l` cases and
confirm that they resolve into the same shared folder key:

```bash
python train.py --config configs/debug_matrix.yaml --run-id debug-standalone-s-001 --output-root "$OUTPUT_ROOT"
python train.py --config configs/debug_matrix.yaml --run-id debug-standalone-m-001 --output-root "$OUTPUT_ROOT"
python train.py --config configs/debug_matrix.yaml --run-id debug-standalone-l-001 --output-root "$OUTPUT_ROOT"
```

Expected result:
- Each run resolves to the same family folder for later comparison.
- The active size remains visible in `config.json` and `run_summary.json`.
- Figure generation can read the folder directly without copying files.

## 5. Generate Figures from the Shared Folder

```bash
python scripts/make_figures.py --input "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/figures"
```

Expected result:
- The script discovers the saved CSV artifacts under the shared family folder.
- No manual renaming or post-processing is needed before plotting.

## 6. Inspect the Saved Metadata

Open the saved `config.json` and `run_summary.json` for one completed run.

Expected result:
- `correction_mode` is recorded explicitly.
- `model.membership_correction` is recorded explicitly.
- The resolved family-folder rule is recorded explicitly.
- The selected preset and final merged values are recorded explicitly.
