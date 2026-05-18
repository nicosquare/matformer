# Quickstart: MatFormer Language Model Reproduction

This quickstart describes the intended validation path for the planned
implementation. Commands may require dependency setup before they run in the
current environment.

## 1. Prepare Environment

Use the `elasticnn` conda environment as the working experiment environment.
The Slurm wrappers default to `$HOME/.conda/envs/elasticnn/bin/python`, and
local commands can use the same interpreter through `PYTHON_BIN`.

Keep `requirements.txt` updated as the portable dependency manifest for
rebuilding a compatible environment when `elasticnn` is unavailable. It should
cover:
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PyYAML
- pandas or standard CSV/JSON support
- matplotlib
- pytest for focused smoke checks
- EleutherAI LM Evaluation Harness before downstream evaluation

Example local interpreter selection:

```bash
export PYTHON_BIN="$HOME/.conda/envs/elasticnn/bin/python"
```

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
"${PYTHON_BIN:-python}" -m pytest tests/test_config.py tests/test_matformer_prefixes.py tests/test_artifacts.py
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
  bash scripts/run_debug_matrix.sh --override training.max_steps_cap=1
```

On a Slurm GPU partition, queue the same validation instead of running it on
the login node:

```bash
sbatch scripts/slurm_debug_matrix.sh \
  --output-root /mnt/experiments/matformer \
  --override training.max_steps_cap=1
```

The Slurm launcher defaults to the `elasticnn` conda environment's Python at
`$HOME/.conda/envs/elasticnn/bin/python`, requests one GPU, and forwards extra
arguments to `scripts/run_debug_matrix.sh`. Submit it with `sbatch`; direct
`bash scripts/slurm_debug_matrix.sh ...` execution is rejected outside a Slurm
allocation. Override scheduler resources at submission time when needed, for
example `sbatch --time=01:00:00 --mem=32G ...`.

Training length is derived from `training.token_budget`,
`training.batch_size_per_process`, model context length, and the active
distributed `WORLD_SIZE`. Use `training.max_steps_cap` only for short smoke
checks that intentionally stop before the derived token-budget length.

To queue only part of the standalone debug matrix during scheduler debugging,
pass an explicit baseline set:

```bash
sbatch scripts/slurm_debug_matrix.sh \
  --output-root /mnt/experiments/matformer \
  --baseline-granularities "s m" \
  --override training.max_steps_cap=1
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
- Baseline matches expose dataset, token budget, model shape labels, sampling
  mode, parameter counts, and mismatch notes.

## 5. Run d_model=256 Pilot Comparison

```bash
PYTHON_BIN=/home/nicolas.avila/.conda/envs/elasticnn/bin/python \
  OUTPUT_ROOT=/mnt/experiments/matformer \
  bash scripts/run_dmodel256_pilot.sh
```

Expected result:
- The pilot is labeled as d_model=256 MatFormer-Llama/SwiGLU with optional
  `table_reference_label=matlm_78m`, not as an exact MatLM-paper reproduction.
- The dataset resolves to `HuggingFaceFW/fineweb` with the `sample-10BT`
  configuration.
- Runs are labeled `reduced-token-pilot` unless they use the MatLM table-row
  10B-token budget reference.
- The resolved config records `effective_world_size`,
  `expected_tokens_per_step`, `derived_max_steps`, and the effective
  `max_steps`.
- The default comparison includes `nested-random`, `nested-all`, and standalone
  S/M/L/XL rows. Standalone rows are emitted as explicit omitted rows by
  default for capped comparison runs unless full standalone baselines are
  requested.
- Outputs record actual tokens seen, target token budget, `stop_reason`,
  sampling mode, actual parameter-count components, LM-head counting
  convention, checkpoint status/path, and mismatch notes.

Queue this on a GPU node rather than the login node. For a short scheduler and
artifact-path check, keep the d_model=256 pilot config but cap the derived step
count:

```bash
sbatch --time=01:00:00 --mem=64G scripts/slurm_dmodel256_pilot.sh \
  --output-root /mnt/experiments/matformer \
  --override training.max_steps_cap=1
```

For the full reduced-token pilot, omit the cap and use the Slurm script's
default resource request unless the cluster queue requires overrides:

```bash
sbatch scripts/slurm_dmodel256_pilot.sh \
  --output-root /mnt/experiments/matformer
```

The default comparison run id prefix is `dmodel256-pilot-comparison`. The
runner launches `nested-random` and `nested-all`, then records standalone S/M/L/XL
as omitted comparison rows when compute is capped. Those omitted rows include
`run_status=omitted`, an `omit_reason`, `checkpoint_status=unavailable`, a null
checkpoint path, and mismatch notes so downstream phases can distinguish
planned-but-not-run baselines from missing metadata.

Use `--output-root` for an explicit root, pass
`--override training.max_steps_cap=...` only for intentionally short runs, and
use `--mode nested-random`, `--mode nested-all`, or
`--mode standalone --granularity <s|m|l|xl>` only for selected smoke/debug
runs. The resolver derives internal `training.granularity_sampling` and
standalone `model.granularities` from those public fields. Example single-mode
checks:

```bash
bash scripts/run_dmodel256_pilot.sh \
  --mode nested-all \
  --run-id dmodel256-nested-all-001 \
  --output-root /mnt/experiments/matformer \
  --override training.max_steps_cap=1

bash scripts/run_dmodel256_pilot.sh \
  --mode standalone \
  --granularity m \
  --run-id dmodel256-standalone-m-001 \
  --output-root /mnt/experiments/matformer \
  --override training.max_steps_cap=1
```

When compute is available for independent baselines, request real standalone
training instead of omitted rows:

```bash
RUN_STANDALONE_BASELINES=1 \
  bash scripts/run_dmodel256_pilot.sh \
  --output-root /mnt/experiments/matformer
```

The Slurm launcher requests one node and multiple GPUs by default. It starts
one config-driven training process per allocated GPU with
`python -m torch.distributed.run`, then the training process derives
`effective_world_size` from the active distributed `WORLD_SIZE`. Do not set
`training.effective_world_size`, `training.expected_tokens_per_step`, or
`training.derived_max_steps` in source YAML. Those fields are written to the
resolved `config.json`.

On clusters where the assigned GPUs are exposed through
`CUDA_VISIBLE_DEVICES`, keep the Slurm GPU request and the number of visible
devices aligned. To queue fewer GPUs than the script default, override the
resource request at submission time:

```bash
sbatch --gres=gpu:2 --time=04:00:00 scripts/slurm_dmodel256_pilot.sh \
  --output-root /mnt/experiments/matformer \
  --override training.max_steps_cap=10
```

Submit the launcher with `sbatch`; do not run it directly on a login node.
The script defaults to the `elasticnn` conda environment at
`$HOME/.conda/envs/elasticnn/bin/python`. Pass `--python-bin` only when using a
different environment.

Slurm stdout and stderr default to `logs/matformer_dmodel256_<jobid>.out` and
`logs/matformer_dmodel256_<jobid>.err` under the repository root. If that
directory is not writable on your cluster, create a writable scheduler-log
directory and override the Slurm paths before the script name:

```bash
mkdir -p /mnt/experiments/matformer/slurm
sbatch \
  --output=/mnt/experiments/matformer/slurm/dmodel256_%j.out \
  --error=/mnt/experiments/matformer/slurm/dmodel256_%j.err \
  scripts/slurm_dmodel256_pilot.sh \
  --output-root /mnt/experiments/matformer \
  --override training.max_steps_cap=1
```

While the job is queued or running, use Slurm and the scheduler logs for coarse
status:

```bash
squeue -j <jobid>
tail -f /mnt/experiments/matformer/slurm/dmodel256_<jobid>.out
```

Rank 0 also writes durable heartbeat events to:

```text
/mnt/experiments/matformer/<run_id>/heartbeats.jsonl
```

Inspect that file to see stage starts, stage completions, and training
progress:

```bash
tail -n 20 /mnt/experiments/matformer/<run_id>/heartbeats.jsonl
```

Expected heartbeat stages include `artifact_writing`, `model_initialization`,
`fsdp_wrapping`, `tokenizer_loading`, `dataset_loading_preprocessing`,
`dataloader_creation`, `training`, `validation`, and `checkpointing` when those
stages occur. Training heartbeats include `step`, `derived_max_steps`,
`tokens_seen`, `content_tokens_seen`, `token_budget`, `latest_loss`,
`tokens_per_second`, `peak_gpu_memory_bytes`, and `eta_seconds`.
`tokens_seen` is the global budget-token counter; `content_tokens_seen` is the
global non-padding token counter. Nonzero ranks may emit process diagnostics to
stdout or stderr, but shared artifacts and `heartbeats.jsonl` are rank-0-only.

## 6. Add Downstream Evaluation

After the debug matrix and d_model=256 pilot comparison artifacts are stable,
run the minimal downstream suite:
- HellaSwag
- PIQA
- ARC-Challenge
- BoolQ
- WinoGrande
- OpenBookQA

The preferred downstream tool is EleutherAI LM Evaluation Harness. Use the
saved best-eval or final checkpoint path recorded in `run_summary.json` when
one is available:

```bash
"${PYTHON_BIN:-python}" -m evaluation.downstream \
  --config configs/dmodel256_pilot_comparison.yaml \
  --run-id dmodel256-nested-random-001 \
  --checkpoint-path "$OUTPUT_ROOT/dmodel256-nested-random-001/checkpoints/best_eval_step_<step>.pt" \
  --granularity xl
```

If `lm_eval` was already run separately, convert its JSON output into this
project's `task_results.csv` schema without relaunching evaluation:

```bash
"${PYTHON_BIN:-python}" -m evaluation.downstream \
  --config configs/dmodel256_pilot_comparison.yaml \
  --run-id dmodel256-nested-random-001 \
  --output-dir "$OUTPUT_ROOT/dmodel256-nested-random-001" \
  --results-json "$OUTPUT_ROOT/dmodel256-nested-random-001/lm_eval_results.json" \
  --granularity xl
```

After downstream task rows exist for the runs you want to compare, regenerate
scaling figures and the medium trend report from the structured artifacts:

```bash
"${PYTHON_BIN:-python}" scripts/make_figures.py \
  --input "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/figures"
```

Expected result:
- `task_results.csv` records per-task metrics.
- `scaling_results.csv` records loss, perplexity, parameter counts, sampling
  mode, checkpoint path, and mismatch notes for each plotted point.
- `scripts/make_figures.py` reads `scaling_results.csv` and `task_results.csv`
  to generate `loss_vs_size.png`, `ppl_vs_size.png`, `accuracy_vs_size.png`,
  and `medium_trend_report.md`.

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
- `speculative-001/task_results.csv` records one row per pair/metric for
  `acceptance_rate`, `rollback_frequency`, `throughput`, and `latency`.
- The default speculative config uses the inline debug prompt set in
  `configs/speculative.yaml`; replace `prompt_set.prompts` or set
  `prompt_set.path` to evaluate a different prompt source.
