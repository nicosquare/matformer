# MatFormer: Nested Transformer for Elastic Inference

This repository provides a public reproduction and open-source implementation of the [MatFormer](https://nips.cc/virtual/2024/poster/94199)'s language modeling experiments (MatLM). It includes the essential building blocks and code required to reproduce the results presented in the paper.

## Features
- Simplified implementation of MatFormer for language modeling tasks.
- Open-source release for community use and further research.
- Reproducibility: Includes key components to replicate the experiments.

## Environment Setup

This repository uses a lightweight Python research stack. Install the baseline
dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

GPU-enabled PyTorch builds may require a platform-specific install command from
the PyTorch project before installing the remaining packages.

## Running LM Pre-training Jobs

To run the training script, execute:

```bash
python train.py
```

For multi-GPU training with FSDP memory sharding, launch with one process per GPU:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 train.py
```

`--batch-size` is per process/GPU. The script automatically enables FSDP when
launched with `torch.distributed.run`.

## Config-Driven Experiment Outputs

Spec Kit experiment configs write resolved configs, metrics, summaries, and
checkpoints under `<output_root>/<run_id>/`. The default output root is
`outputs/`, but larger runs should use a filesystem with enough space and
inodes:

```bash
export OUTPUT_ROOT=/mnt/experiments/matformer
python train.py \
  --config configs/debug_matrix.yaml \
  --run-id debug-nested-001 \
  --output-root "$OUTPUT_ROOT"
```

The debug matrix runner also reads `OUTPUT_ROOT`:

```bash
OUTPUT_ROOT=/mnt/experiments/matformer bash scripts/run_debug_matrix.sh
```

For GPU clusters, submit the same Phase 3 validation through Slurm from the
repository root:

```bash
sbatch scripts/slurm_debug_matrix.sh \
  --output-root /mnt/experiments/matformer \
  --baseline-granularity s
```

The Slurm launcher uses the `elasticnn` conda environment by default through
`$HOME/.conda/envs/elasticnn/bin/python`. Pass `--python-bin` or set
`PYTHON_BIN` if the environment lives elsewhere. Submit it with `sbatch`; it
refuses direct `bash` execution outside a Slurm allocation.

The d_model=256 reduced-token pilot comparison has the same Slurm wrapper pattern:

```bash
sbatch scripts/slurm_dmodel256_pilot.sh \
  --output-root /mnt/experiments/matformer
```

For a short scheduler smoke check, cap the derived token-budget step count:

```bash
sbatch --time=01:00:00 scripts/slurm_dmodel256_pilot.sh \
  --output-root /mnt/experiments/matformer \
  --override training.max_steps_cap=1
```

The d_model=256 Slurm launcher requests one node with multiple GPUs and
launches one config-driven process per allocated GPU through
`python -m torch.distributed.run`. The launcher can use Slurm GPU allocation
variables or `CUDA_VISIBLE_DEVICES` to choose `--nproc_per_node`, but the
resolved training length uses the active distributed `WORLD_SIZE` inside the
training process. Do not hand-maintain `training.effective_world_size`,
`training.expected_tokens_per_step`, or `training.derived_max_steps` in source
configs.

The d_model=256 pilot comparison defaults to
`training.granularity_sampling=random`, which matches the original training
script by sampling one MatFormer granularity per batch. Use
`--override training.granularity_sampling=all` only for the ablation path that
evaluates all configured granularities on each batch and averages losses.

To request a different single-node GPU count, override the Slurm resource
request at submission time:

```bash
sbatch --gres=gpu:2 scripts/slurm_dmodel256_pilot.sh \
  --output-root /mnt/experiments/matformer \
  --override training.max_steps_cap=10
```

Submit the launcher with `sbatch`; it is not intended for direct execution on a
login node. Scheduler stdout/stderr default to `logs/matformer_dmodel256_%j.out` and
`logs/matformer_dmodel256_%j.err`. If repository-local `logs/` is not writable,
override those paths before the script name:

```bash
mkdir -p /mnt/experiments/matformer/slurm
sbatch \
  --output=/mnt/experiments/matformer/slurm/dmodel256_%j.out \
  --error=/mnt/experiments/matformer/slurm/dmodel256_%j.err \
  scripts/slurm_dmodel256_pilot.sh \
  --output-root /mnt/experiments/matformer
```

During the job, check scheduler status and heartbeat progress:

```bash
squeue -j <jobid>
tail -f /mnt/experiments/matformer/slurm/dmodel256_<jobid>.out
tail -n 20 /mnt/experiments/matformer/dmodel256-pilot-comparison-001/heartbeats.jsonl
```

`heartbeats.jsonl` is written by rank 0 under the run output directory and
records stage start/completion events plus training progress such as step,
token budget, loss, throughput, peak GPU memory, and ETA.

Use `--output-dir` only for a one-off explicit run directory. For shared cache
pressure, place Hugging Face caches outside the repository before launching
jobs:

```bash
export HF_HOME=/mnt/experiments/hf-cache
export HF_DATASETS_CACHE=/mnt/experiments/hf-cache/datasets
export TRANSFORMERS_CACHE=/mnt/experiments/hf-cache/transformers
```
