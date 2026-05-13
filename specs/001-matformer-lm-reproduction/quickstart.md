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

## 3. Run Debug Matrix

```bash
bash scripts/run_debug_matrix.sh
```

Expected result:
- One debug-size nested run evaluates S, M, L, and XL.
- Four matched debug-size standalone baselines are produced.
- `metrics.csv`, `scaling_results.csv`, `run_summary.json`, and plots are
  written under `outputs/`.

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
