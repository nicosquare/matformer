# Quickstart: Source Layout Housekeeping

## 1. Install the project in editable mode

```bash
python -m pip install -e .
```

## 2. Verify package imports resolve from `src/`

```bash
python - <<'PY'
import src.models as models
import src.training as training
import src.evaluation as evaluation
import src.utils as utils
print(models.__file__)
print(training.__file__)
print(evaluation.__file__)
print(utils.__file__)
PY
```

## 3. Run the training wrapper

```bash
python train.py \
  --config configs/debug_matrix.yaml \
  --run-id debug-nested-001 \
  --override training.max_steps_cap=1
```

If the workspace cannot reach Hugging Face, use cached assets instead:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_CACHE=/path/to/writable/hf-datasets-cache \
python train.py \
  --config configs/debug_matrix.yaml \
  --run-id debug-nested-001 \
  --override training.max_steps_cap=1
```

The training code now falls back to a local tokenizer snapshot when the hub is
unavailable. The dataset cache still needs to be writable because the
preprocessing step writes temporary files into the active Hugging Face cache
root.

## 4. Run the figure-generation wrapper

```bash
python scripts/make_figures.py \
  --input outputs \
  --output outputs/figures \
  --no-refresh-counts
```

## 5. Run focused regression tests

```bash
pytest \
  tests/test_config.py \
  tests/test_training_smoke.py \
  tests/test_artifacts.py \
  tests/test_model_size.py \
  tests/test_monitoring.py
```

## Success Check

- Root-level wrappers still execute from the repository root.
- Imports resolve through the `src/` layout.
- Representative training and figure-generation smoke runs still write the expected artifacts.
- In restricted environments, the training smoke can run from local Hugging
  Face caches without changing repository code.
