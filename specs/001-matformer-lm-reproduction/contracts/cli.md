# Contract: Command Interfaces

These commands describe the intended researcher-facing interfaces. Names may be
implemented as scripts or Python module entry points, but the arguments and
outputs must remain visible and easy to trace.

## Run One Experiment

```bash
python train.py --config configs/debug_matrix.yaml --run-id debug-nested-001 --output-root /mnt/experiments/matformer
```

Required behavior:
- Resolve config and CLI overrides.
- Accept `--output-root` and resolve artifacts under `<output_root>/<run_id>/`.
- Accept `--output-dir` as an explicit one-run output directory override.
- Save `<output_root>/<run_id>/config.json` unless `--output-dir` is used.
- Train or evaluate the requested model family and granularity set.
- Write relevant CSV/JSON artifacts.

## Run Debug Matrix

```bash
OUTPUT_ROOT=/mnt/experiments/matformer bash scripts/run_debug_matrix.sh
```

Required behavior:
- Run one debug-size nested S/M/L/XL experiment.
- Run matched standalone S, M, L, and XL baselines.
- Accept `OUTPUT_ROOT` or output-root arguments and place all matrix run
  directories under `<output_root>/<run_id>/`.
- Write comparison artifacts and plots derived from CSV files.

## Queue Debug Matrix On Slurm

```bash
sbatch scripts/slurm_debug_matrix.sh --output-root /mnt/experiments/matformer
```

Required behavior:
- Request one GPU when submitted with `sbatch`.
- Refuse direct local execution outside a Slurm allocation.
- Use the `elasticnn` conda environment by default unless `PYTHON_BIN` or
  `--python-bin` is provided.
- Accept `--output-root`, `--baseline-granularity`, `--nested-run-id`, and
  `--config`.
- Forward remaining runner arguments, including `--override`, to
  `scripts/run_debug_matrix.sh`.

## Run 78M Reduced-Token Pilot

```bash
OUTPUT_ROOT=/mnt/experiments/matformer bash scripts/run_78m_pilot.sh
```

Required behavior:
- Use paper-aligned architecture constants.
- Label the run `reduced-token-pilot` unless the token budget is 10B.
- Write model-size and token-budget labels into summaries.
- Accept the same output-root controls as the debug matrix runner.

## Queue 78M Reduced-Token Pilot On Slurm

```bash
sbatch scripts/slurm_78m_pilot.sh --output-root /mnt/experiments/matformer
```

Required behavior:
- Request one GPU when submitted with `sbatch`.
- Refuse direct local execution outside a Slurm allocation.
- Use the `elasticnn` conda environment by default unless `PYTHON_BIN` or
  `--python-bin` is provided.
- Accept `--output-root`, `--run-id`, and `--config`.
- Forward remaining runner arguments, including `--override`, to
  `scripts/run_78m_pilot.sh`.

## Generate Figures

```bash
python scripts/make_figures.py --input /mnt/experiments/matformer --output /mnt/experiments/matformer/figures
```

Required behavior:
- Read `metrics.csv`, `scaling_results.csv`, and `consistency_results.csv`.
- Generate plots from structured artifacts only.
- Never require terminal logs as plot inputs.

## Run Consistency Evaluation

```bash
OUTPUT_ROOT=/mnt/experiments/matformer python -m evaluation.consistency --config configs/consistency.yaml
```

Required behavior:
- Compare smaller and larger granularities on the same samples.
- Write `consistency_results.csv`.

## Run Speculative Evaluation

```bash
OUTPUT_ROOT=/mnt/experiments/matformer python -m evaluation.speculative --config configs/speculative.yaml
```

Required behavior:
- Compare nested draft/verifier pairs and standalone draft/verifier pairs.
- Report acceptance rate, rollback frequency, throughput, and latency.

## External Cache Control

Runner commands do not own Hugging Face cache placement directly. Researchers
should set cache environment variables before running commands when repository
or home filesystems have limited space or inodes:

```bash
export HF_HOME=/mnt/experiments/hf-cache
export HF_DATASETS_CACHE=/mnt/experiments/hf-cache/datasets
export TRANSFORMERS_CACHE=/mnt/experiments/hf-cache/transformers
```
