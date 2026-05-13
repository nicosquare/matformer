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
