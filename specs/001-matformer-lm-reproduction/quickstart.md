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

For machines with restricted repository or home filesystem space, place run
artifacts and Hugging Face caches on a larger filesystem before launching
experiments:

```bash
export OUTPUT_ROOT=/mnt/experiments/matformer
export HF_HOME=/mnt/experiments/hf-cache
export HF_DATASETS_CACHE=/mnt/experiments/hf-cache/datasets
export TRANSFORMERS_CACHE=/mnt/experiments/hf-cache/transformers
```

`OUTPUT_ROOT` is the common runner path for matrix-style commands. Single-run
commands can also pass `--output-root "$OUTPUT_ROOT"` directly. Use
`--output-dir` only when one run needs an explicit directory that does not
follow `<output_root>/<run_id>`.

## 2. Run Focused Smoke Checks

```bash
pytest tests/test_config.py tests/test_matformer_prefixes.py tests/test_artifacts.py
```

Expected result:
- Configs resolve to explicit experiment concepts.
- S/M/L/XL prefixes are valid and ordered.
- Metrics/config artifacts can be written to `<output_root>/<run_id>/`.

## 3. Run P1 Debug Validation

```bash
bash scripts/run_debug_matrix.sh
```

Expected result:
- One debug-size nested run evaluates S, M, L, and XL.
- Matched debug-size standalone baselines are produced for S, M, L, and XL by
  default.
- `metrics.csv`, `scaling_results.csv`, and `run_summary.json` are written for
  the nested run and matched standalone baselines under the configured
  output root.
- The nested `run_summary.json` records the baseline match and any mismatch
  notes.

Useful local override example:

```bash
PYTHON_BIN=/home/nicolas.avila/.conda/envs/elasticnn/bin/python \
  OUTPUT_ROOT=/mnt/experiments/matformer \
  bash scripts/run_debug_matrix.sh --override training.max_steps=1
```

On a Slurm GPU partition, queue the same validation instead of running it on
the login node:

```bash
sbatch scripts/slurm_debug_matrix.sh \
  --output-root /mnt/experiments/matformer \
  --override training.max_steps=1
```

The Slurm launcher defaults to the `elasticnn` conda environment's Python at
`$HOME/.conda/envs/elasticnn/bin/python`, requests one GPU, and forwards extra
arguments to `scripts/run_debug_matrix.sh`. Submit it with `sbatch`; direct
`bash scripts/slurm_debug_matrix.sh ...` execution is rejected outside a Slurm
allocation. Override scheduler resources at submission time when needed, for
example `sbatch --time=01:00:00 --mem=32G ...`.

To queue only part of the standalone debug matrix during scheduler debugging,
pass an explicit baseline set:

```bash
sbatch scripts/slurm_debug_matrix.sh \
  --output-root /mnt/experiments/matformer \
  --baseline-granularities "s m" \
  --override training.max_steps=1
```

Equivalent config override form:

```bash
bash scripts/run_debug_matrix.sh \
  --override run.output_root=/mnt/experiments/matformer
```

## 4. Inspect Debug Outputs

```bash
python scripts/make_figures.py --input "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/figures"
```

Check:
- No required metric appears only in terminal logs.
- Every plot can be traced to CSV inputs.
- Baseline matches expose dataset, token budget, and model-size labels.

## 5. Run 78M Reduced-Token Pilot

```bash
PYTHON_BIN=/home/nicolas.avila/.conda/envs/elasticnn/bin/python \
  OUTPUT_ROOT=/mnt/experiments/matformer \
  bash scripts/run_78m_pilot.sh
```

Expected result:
- Architecture constants are paper-aligned.
- The run is labeled `reduced-token-pilot` unless it uses the 10B token budget.
- Outputs record actual tokens seen and target token budget.

Queue this on a GPU node rather than the login node. For a short scheduler and
artifact-path check, keep the 78M config but override the step count:

```bash
PYTHON_BIN=/home/nicolas.avila/.conda/envs/elasticnn/bin/python \
  OUTPUT_ROOT=/mnt/experiments/matformer \
  bash scripts/run_78m_pilot.sh --override training.max_steps=1
```

The default run id is `78m-reduced-pilot-001`, so artifacts resolve under
`<OUTPUT_ROOT>/78m-reduced-pilot-001/`. Use `--output-root` for an explicit
root, `--output-dir` for a one-off directory, and
`--run-id 78m-reduced-pilot-001` when validating the runner contract manually.

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
OUTPUT_ROOT=/mnt/experiments/matformer \
  python -m evaluation.consistency --config configs/consistency.yaml
OUTPUT_ROOT=/mnt/experiments/matformer \
  python -m evaluation.speculative --config configs/speculative.yaml
```

Expected result:
- `consistency_results.csv` records nested-vs-standalone alignment metrics.
- Speculative evaluation reports acceptance rate, rollback frequency,
  throughput, and latency.
