# Quickstart: Source Layout Housekeeping

## 1. Install the project in editable mode

```bash
python -m pip install -e .
```

## 2. Verify package imports resolve from `src/`

```bash
python - <<'PY'
import models
import training
import evaluation
import utils
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
