# TinyStories S1 peak learning-rate sweep

## Question and observed results

Does changing peak learning rate improve the terminal ordinary-validation loss
of uniform-sampling S1 while retaining the standalone runs' 64-update warmup?

The completed four-epoch, seed-42 S1 results at peak LR 0.008 motivate a sweep.
The 256-update warmup changed terminal loss relative to the 64-update S1 as
follows (positive means worse):

| Grid | Width order | 256 minus 64 warmup loss, in width order |
| --- | --- | --- |
| Linear | g250, g500, g750, g1000 | +0.011792, +0.013590, +0.013429, +0.012838 |
| Geometric | g125, g250, g500, g1000 | -0.003527, -0.001183, -0.002500, -0.003605 |

Every 64-update S1 endpoint remains above its matching one-epoch standalone.
The four Linear gaps are +0.005180, +0.003972, +0.022803, +0.044816;
the Geometric gaps are +0.022596, +0.011273, +0.039369, +0.087138.
These comparisons describe the saved endpoints; they do not establish which
direction to change peak LR.

Sources: `terminal_validation_results.json` in the S1 and standalone run
directories under the existing
`optimizer-ownership-v1`, `optimizer-ownership-matformer-widths-v1`, and
`optimizer-ownership-s1-warmup-v1` result roots. Each selected S1 terminal
records 348,528 committed updates and four ordinary-validation endpoints.

## Proposed runs

Use fresh, separately identified runs for each grid and new peak LR:

| Grid | Peak LR | Warmup updates | Assigned updates | Role |
| --- | ---: | ---: | ---: | --- |
| Linear | 0.004 | 64 | 348,528 | lower LR candidate |
| Linear | 0.012 | 64 | 348,528 | higher LR candidate |
| Geometric | 0.004 | 64 | 348,528 | lower LR candidate |
| Geometric | 0.012 | 64 | 348,528 | higher LR candidate |

The existing 0.008/64-update Linear and Geometric S1 runs supply the center
control; they require no retraining. Values 0.004 and 0.012 bracket that
control without assuming that either grid has the same optimum. All four new
runs use the original respective S1 grid, seed and version-1 random streams,
fresh model/optimizer state, uniform global replacement sampling, shared
AdamW, bf16, unchanged corpus and validation manifests, four epochs, and the
same cosine terminal horizon. Keep pre-nested warmup and LR scaling disabled.
The first applied peak LR is at update 65 under the inherited scheduler
indexing convention. Do not initialize from the 0.008 or 256-warmup checkpoints.

Changing peak LR is the sole scientific intervention relative to the matching
0.008/64-update S1. Keep the full 348,528-update budget, or a truncated run
would mix LR selection with a changed training horizon. Four new runs assign
1,394,112 updates and 11,420,565,504 training tokens in total. The two
completed 0.008 runs already provide the center measurements.

## Readout and decision rule

Compare terminal ordinary-validation loss and perplexity for all four widths
within each grid. Report each candidate's signed change from the matching
0.008/64-update S1 and its gap to the matching standalone. Show early loss and
applied LR traces to identify instability, and report the full-width result
separately from the narrower widths. Retain the final checkpoint as the
terminal selection; a transient best checkpoint is diagnostic only.

Treat each grid independently. A candidate is an across-width improvement
only if all four terminal losses decrease; otherwise report the tradeoff by
width. The sweep gives a bracket, not a guarantee of an optimum. If the lower
candidate wins, a later sweep can probe below 0.004; if the higher candidate
wins, probe above 0.012. If each helps different widths, select LR by the
intended width objective before further runs.

## Execution boundary

Prepare unique campaign/run IDs, resolved config differences, source snapshot,
resume identity, and a saved-artifact comparison before submitting GPU jobs.
Use the fresh campaign root
`/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-s1-peak-lr-v1`.
It was absent when this proposal was written; reserve it during preflight.
Put each training run in its own directory:

```text
optimizer-ownership-s1-peak-lr-v1/
  campaign/                    # resolved configs and protocol evidence
  diagnostics/                 # source-bound readiness evidence
  launchers/                   # job and attempt records
  logs/                        # scheduler output
  runs/
    S1-linear-lr0004/
    S1-linear-lr0012/
    S1-geometric-lr0004/
    S1-geometric-lr0012/
  reports/                     # comparison with existing 0.008 and standalones
```

Here `lr0004` means 0.004 and `lr0012` means 0.012. The run directories
contain their own checkpoints, metrics and terminal validation. Keep existing
result roots read-only. Recheck live Slurm limits and source-bound CPU/GPU
readiness when execution is prepared.

## Submission and startup, 2026-09-23

The four resolved configs were published under `campaign/configs/`, with their
SHA-256 hashes in `campaign/manifest.json`. Each was derived from its grid's
completed 0.008/64-update S1 config. A read-only preflight confirmed identical
model, data, evaluation, outputs and training sections except the peak LR;
each resolves to 348,528 updates, 64 warmup steps, shared AdamW and bf16.
Training uses the previously tested immutable source snapshot at
`optimizer-ownership-s1-warmup-v1/source` through
`scripts/train_cuda_required.py`. The new configs use fresh run IDs and output
paths. They omit the historical ownership-report contract, so their terminal
readout should use each run's saved validation metrics and scaling results.

| Run | Slurm job | Initial state | Node / dependency |
| --- | ---: | --- | --- |
| S1-linear-lr0004 | 274754 | running | gpu-03 |
| S1-linear-lr0012 | 274755 | running | gpu-08 |
| S1-geometric-lr0004 | 274757 | pending | afterok:274754:274755 |
| S1-geometric-lr0012 | 274758 | pending | afterok:274754:274755 |

Both running jobs recorded successful `nvidia-smi`, their assigned
`CUDA_VISIBLE_DEVICES`, resolved bf16, at least 1,440 committed training
updates and 453,779,968 peak CUDA bytes in training metrics. The first 100
Linear 0.004 width choices and batch provenance match the historical S1
reference. The Geometric jobs can start only after both Linear jobs complete
successfully, preserving the two-running-job ceiling. Submission and CUDA
entry evidence live in `launchers/`; scheduler output is in `logs/`.

## Linear terminal figures

Jobs 274754 and 274755 completed with Slurm exit `0:0`, 348,528 committed
updates each, full token budgets, and terminal ordinary validation. The two
Geometric jobs were running when the Linear plots were produced.

| Width | Standalone 0.008 (1 epoch) | S1 0.004 | S1 0.008 | S1 0.012 |
| --- | ---: | ---: | ---: | ---: |
| g250 | 2.062977 | 2.062966 | 2.068157 | 2.080816 |
| g500 | 1.991942 | 1.991542 | 1.995913 | 2.007175 |
| g750 | 1.943363 | 1.962110 | 1.966166 | 1.977576 |
| g1000 | 1.909057 | 1.951018 | 1.953872 | 1.963961 |

The S1 0.004 run improves on the S1 0.008 control at every width by
0.002854–0.005191 loss. S1 0.012 worsens all four by 0.010089–0.012659.
At g250 the 0.004 run and standalone differ by only 0.000011 in loss;
the wider S1 points still trail their standalones.

`reports/linear/` contains terminal loss and perplexity versus active
non-embedding parameters, raw validation loss versus updates, and recorded
early LR/training loss, each in PNG and PDF. It also contains `endpoints.csv`,
`endpoints.json`, and `plot_sources.json`. Figures use the final ordinary
validation values, not the best-checkpoint fields from the generic
`scaling_results.csv`. The reusable plotting command is
`scripts/plot_tinystories_s1_peak_lr.py --grid linear --output-dir <destination>`.

## Cosine scheduler audit

`reports/scheduler_audit.json` compares recorded pre-optimizer LR at every
committed update to the intended schedule: for scheduler position `p=s-1`,
`peak*p/64` before position 64, then
`peak*(1+cos(pi*(p-64)/(348528-64)))/2` through the fixed horizon.
It also checks consecutive steps, the recorded scheduler positions and
warmup field, and the terminal checkpoint state for completed runs.

Both Linear jobs matched at all 348,528 updates, with maximum absolute LR
error below 1.1e-16 and zero rows above 1e-12. Update 1 applied zero LR;
update 65 first applied the full peak (0.004 or 0.012). The final applied
LR was approximately 8.13e-14 or 2.44e-13 at update 348,528, respectively;
both terminal checkpoints store scheduler position 348,528 and next LR zero.
The running Geometric jobs matched over the available committed prefixes:
179,360 updates at 0.004 and 185,856 at 0.012, each with the same 64 warmup
positions and no mismatch above 1e-12. Their terminal schedule positions
remain to be checked after completion.

## Geometric terminal figures, 2026-09-24

Jobs 274757 and 274758 completed with Slurm exit `0:0` and 348,528
committed updates each. Their final ordinary-validation losses are:

| Width | Standalone 0.008 (1 epoch) | S1 0.004 | S1 0.008 | S1 0.012 |
| --- | ---: | ---: | ---: | ---: |
| g125 | 2.105963 | 2.127988 | 2.128560 | 2.122445 |
| g250 | 2.062977 | 2.072754 | 2.074250 | 2.068214 |
| g500 | 1.991942 | 2.030020 | 2.031311 | 2.024198 |
| g1000 | 1.909057 | 1.990702 | 1.996195 | 1.990808 |

S1 0.012 has the lowest terminal loss at g125, g250 and g500; S1 0.004
is lower by 0.000105 at g1000. Both candidates improve on S1 0.008 at all
four widths while remaining above the matching standalone endpoints.
`reports/geometric/` contains loss and perplexity versus active parameters,
raw validation trajectories, and early applied-LR/training-loss figures,
each in PNG and PDF, plus endpoint tables and plotted-source hashes.

The saved `reports/scheduler_audit.json` now covers all four complete runs.
Each has exactly 348,528 consecutive committed LR observations, maximum
absolute difference from the declared schedule below 1.1e-16, no row above
1e-12, and a terminal checkpoint at scheduler position 348,528 with stored
LR zero.
