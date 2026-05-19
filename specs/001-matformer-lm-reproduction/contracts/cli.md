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
- Accept `--output-root` and resolve artifacts under
  `<output_root>/<output_group>/<run_id>/`.
- Accept `--output-dir` as an explicit one-run output directory override.
- Save `<output_root>/<output_group>/<run_id>/config.json` unless
  `--output-dir` is used.
- Train or evaluate the requested training topology, sampling mode, and granularity
  set.
- Write relevant CSV/JSON artifacts.

## Run Debug Matrix

```bash
OUTPUT_ROOT=/mnt/experiments/matformer bash scripts/run_debug_matrix.sh
```

Required behavior:
- Run one debug-size nested S/M/L/XL experiment.
- Run matched standalone S, M, L, and XL baselines.
- Accept `OUTPUT_ROOT` or output-root arguments and place all matrix run
  directories under `<output_root>/<output_group>/<run_id>/`.
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

## Run d_model=256 Pilot Comparison

```bash
OUTPUT_ROOT=/mnt/experiments/matformer bash scripts/run_dmodel256_pilot.sh
```

Required behavior:
- Default to the pilot comparison workflow rather than a single nested run.
- Run or schedule `nested-random`, `nested-all`, and standalone S/M/L/XL rows
  where compute allows.
- Allow smoke/debug invocations that select one mode through an explicit mode
  argument.
- Label the pilot with explicit d_model=256 shape fields and sampling mode.
- Derive `completion_label` automatically as `debug` or `run`.
- Derive `model_family_slug`, `model_size_slug`, `token_budget_slug`, and
  `output_group` automatically in the resolved config.
- Write explicit shape fields, token-budget labels, sampling mode, actual
  parameter counts, and LM-head counting convention into summaries.
- Accept the same output-root controls as the debug matrix runner.

## Queue d_model=256 Pilot Comparison On Slurm

```bash
sbatch scripts/slurm_dmodel256_pilot.sh --output-root /mnt/experiments/matformer
```

Required behavior:
- Request one node with multiple GPUs when submitted with `sbatch`.
- Refuse direct local execution outside a Slurm allocation.
- Use the `elasticnn` conda environment by default unless `PYTHON_BIN` or
  `--python-bin` is provided.
- Accept `--output-root`, `--run-id`, `--config`, `--mode`, and baseline
  selection arguments.
- Launch one config-driven training process per allocated GPU through
  `torchrun` or `python -m torch.distributed.run`.
- The launcher may use Slurm allocation variables, including
  `CUDA_VISIBLE_DEVICES` when that is how the cluster exposes assigned GPUs, to
  choose `--nproc_per_node`.
- After launch, use the active distributed `WORLD_SIZE` set by `torchrun`, not
  `CUDA_VISIBLE_DEVICES` or available GPU count directly, when the resolved
  config derives token-budget training length.
- Use FSDP for the d_model=256 pilot model when distributed execution is
  enabled.
- Write shared artifacts such as `config.json`, `metrics.csv`,
  `run_summary.json`, checkpoints, and `heartbeats.jsonl` from rank 0 only.
- Save a rank-0-safe best-eval checkpoint when validation is enabled and record
  checkpoint status/path in `run_summary.json`.
- Default Slurm output to heartbeat lines instead of tqdm-style progress bars.
- Forward remaining runner arguments, including `--override`, to the
  config-driven d_model=256 pilot entry point.

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
