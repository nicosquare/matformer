# Quickstart: MatFormer Language Model Reproduction

This quickstart describes the intended validation path for the planned
implementation. Commands may require dependency setup before they run in the
current environment.

## 1. Prepare Environment

The default `python3` available during planning is Python 3.12.3, but it does
not currently include the ML dependencies imported by `train.py`.

Install or activate an environment with:
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- pandas or standard CSV/JSON support
- matplotlib
- pytest for focused smoke checks
- EleutherAI LM Evaluation Harness before downstream evaluation

## 2. Run Focused Smoke Checks

```bash
pytest tests/test_config.py tests/test_matformer_prefixes.py tests/test_artifacts.py
```

Expected result:
- Configs resolve to explicit experiment concepts.
- S/M/L/XL prefixes are valid and ordered.
- Metrics/config artifacts can be written to `outputs/<run_id>/`.

## 3. Run P1 Debug Validation

```bash
bash scripts/run_debug_matrix.sh
```

Expected result:
- One debug-size nested run evaluates S, M, L, and XL.
- One matched debug-size standalone baseline is produced. The default Phase 3
  baseline is `s`; set `BASELINE_GRANULARITY=m`, `l`, or `xl` to run a
  different single baseline.
- `metrics.csv`, `scaling_results.csv`, and `run_summary.json` are written for
  the nested run and the matched standalone baseline under `outputs/`.
- The nested `run_summary.json` records the baseline match and any mismatch
  notes.

Useful local override example:

```bash
PYTHON_BIN=/home/nicolas.avila/.conda/envs/elasticnn/bin/python \
  bash scripts/run_debug_matrix.sh --override training.max_steps=1
```

The full S/M/L/XL standalone debug matrix is a Phase 4 extension.

## 4. Inspect Debug Outputs

```bash
python scripts/make_figures.py --input outputs --output outputs/figures
```

Check:
- No required metric appears only in terminal logs.
- Every plot can be traced to CSV inputs.
- Baseline matches expose dataset, token budget, and model-size labels.

## 5. Run 78M Reduced-Token Pilot

```bash
bash scripts/run_78m_pilot.sh
```

Expected result:
- Architecture constants are paper-aligned.
- The run is labeled `reduced-token-pilot` unless it uses the 10B token budget.
- Outputs record actual tokens seen and target token budget.

## 6. Add Downstream Evaluation

After the debug matrix and 78M pilot artifact flow are stable, run the minimal
downstream suite:
- HellaSwag
- PIQA
- ARC-Challenge
- BoolQ
- WinoGrande
- OpenBookQA

Expected result:
- `task_results.csv` records per-task metrics.
- `scaling_results.csv` records average downstream accuracy.

## 7. Add Consistency and Speculative Evaluations

Run consistency before speculative decoding:

```bash
python -m evaluation.consistency --config configs/consistency.yaml
python -m evaluation.speculative --config configs/speculative.yaml
```

Expected result:
- `consistency_results.csv` records nested-vs-standalone alignment metrics.
- Speculative evaluation reports acceptance rate, rollback frequency,
  throughput, and latency.
