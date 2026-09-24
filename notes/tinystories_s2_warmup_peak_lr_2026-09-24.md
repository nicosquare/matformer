# TinyStories S2 warmup and peak-LR follow-up

## Protocol

Fresh seed-42 S2 runs use slicing with one AdamW history per selected width,
uniform global replacement sampling, global scheduler clock, bf16, global L2
clip 1, the existing audited corpus/tokenizer and ordinary validation, and
the same 348,528-update / 2,855,141,376-token four-epoch budget. The completed
Linear and Geometric S2 runs at peak LR 0.008 and 64 global warmup updates
are read-only controls.

| Grid | Run | Peak LR | Warmup updates |
| --- | --- | ---: | ---: |
| Linear | S2-linear-w256 | 0.008 | 256 |
| Geometric | S2-geometric-w256 | 0.008 | 256 |
| Linear | S2-linear-lr0004 | 0.004 | 64 |
| Geometric | S2-geometric-lr0004 | 0.004 | 64 |
| Linear | S2-linear-lr0012 | 0.012 | 64 |
| Geometric | S2-geometric-lr0012 | 0.012 | 64 |

All six configs were derived from the corresponding saved S2 configuration.
Resolved model, data, evaluation and output controls match the reference;
within each branch, only the declared LR or warmup changes scientifically.
The frozen trainer is the previously used source snapshot under
`optimizer-ownership-s1-warmup-v1/source`. Fresh outputs are under
`/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-s2-warmup-peak-lr-v1`.
`campaign/manifest.json` pins source and config hashes; `campaign/configs.sha256`
is checked in each allocation before training.

## Slurm submission

The live user ceiling was two running GPU jobs and four submitted jobs. One
array submission, parent job **275468**, schedules three tasks with `%2`, each
running two independent S2 experiments sequentially on its GPU. Thus six
training runs occupy at most two GPUs concurrently and three submitted Slurm
tasks. Each task has one GPU, four CPUs, 16 GiB and a 24-hour limit. Excluded
nodes are `gpu-[05,50,51,54]`.

| Array task | First run | Second run | Startup state |
| --- | --- | --- | --- |
| 0 | S2-linear-w256 | S2-linear-lr0004 | Running as job 275469 on gpu-02 |
| 1 | S2-geometric-w256 | S2-geometric-lr0004 | Running as job 275470 on gpu-02 |
| 2 | S2-linear-lr0012 | S2-geometric-lr0012 | Pending behind the array's two-task limit |

Tasks 0 and 1 were assigned different GPU IDs, 1 and 3, on gpu-02. Both
CUDA entries passed `nvidia-smi`, resolved bf16 and per-width optimizer state,
and committed at least 32 real updates with 466,390,528 recorded peak CUDA
bytes. The first 32 width choices and batch provenance match their respective
historical S2 controls. The launcher runs the second configuration even if the
first fails, records both return codes in `launchers/pair-*.txt`, and marks the
Slurm task failed if either run fails. Logs and CUDA entry evidence live under
`logs/` and `launchers/`. Completion and terminal comparisons remain pending.
