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

The 78M reduced-token pilot has the same Slurm wrapper pattern:

```bash
sbatch scripts/slurm_78m_pilot.sh \
  --output-root /mnt/experiments/matformer
```

For a short scheduler smoke check, cap the derived token-budget step count:

```bash
sbatch --time=01:00:00 scripts/slurm_78m_pilot.sh \
  --output-root /mnt/experiments/matformer \
  --override training.max_steps_cap=1
```

Use `--output-dir` only for a one-off explicit run directory. For shared cache
pressure, place Hugging Face caches outside the repository before launching
jobs:

```bash
export HF_HOME=/mnt/experiments/hf-cache
export HF_DATASETS_CACHE=/mnt/experiments/hf-cache/datasets
export TRANSFORMERS_CACHE=/mnt/experiments/hf-cache/transformers
```
