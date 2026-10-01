# TinyStories C4 selected-block experiment

Status on 2026-09-26: both seed-42 BF16 runs completed 348,528 updates and
2,855,141,376 tokens. The dependent report completed with 16 endpoint rows,
eight differences, and six figures in both PNG and PDF. Saved config, summary,
and terminal checkpoint hashes were rechecked against the report evidence.

## Protocol

Each global update samples one of four widths uniformly with replacement. The
full selected prefix participates in forward and backward, while only its newly
introduced FFN block and the common owner receive an AdamW step. Earlier block
gradients are cleared before one joint global L2 clip at 1.0. A single cosine
clock advances after the logical update. C4 keeps the historical batch order,
seed streams, 64-update warmup, four-epoch budget, and ordinary-validation
manifest. The grids are linear `g250/g500/g750/g1000` and geometric
`g125/g250/g500/g1000`.

The fresh campaign root is
`/nfs-stor/ivo.navarrete/results/elasticnn/tinystories-optimizer-ownership-c4-selected-block-v1`.
The tested source is under `source/`; the resolved YAMLs are in
`source/configs/controlled_exps/`. The original standalone runs are read only.

## Submission and evidence

| Purpose | Slurm job | Output |
| --- | --- | --- |
| Final real-shape BF16 gate | 278692, completed | `diagnostics/gpu-gate.json` |
| C4 linear | 278693, completed | `runs/tinystories-optimizer-ownership-c4-selected-block-v1-C4-linear-s42/` |
| C4 geometric | 278694, completed | `runs/tinystories-optimizer-ownership-c4-selected-block-v1-C4-geometric-s42/` |
| Terminal comparison | 278706, completed | `report/` |

The first two diagnostic attempts, 278688 and 278689, failed before completing
the BF16 gate. Their logs and the first tested source revision are retained.
The final gate ran both grids at the production model shape, BF16, batch 64,
context 128, and checked eight committed updates per grid, selected owner calls,
joint clipping, unchanged unselected parameters, and checkpoint state loading.

The CPU regression suite passed with 645 tests and 32 expected GPU skips. The
focused C4 and compact-accounting suite passed with 36 tests. The first 64
committed updates on each grid matched historical width-position and batch
streams. See `diagnostics/action-data-audit.json` and
`diagnostics/control-difference-audit.json` for the saved audits.

CUDA entry records are in `diagnostics/cuda-entry-{linear,geometric}.json`.
Both production jobs produced BF16 metrics with committed updates. Their latest
checkpoints passed CPU inspection of model shape, owner histories, scheduler
position, data and RNG provenance. The action/data audit is limited to the first
64 updates; full-run accounting is checked at terminal reporting.

## Terminal report and recovery

The CPU report job completed after both production jobs.
It validated each terminal checkpoint and matched all eight C4 endpoints against
the corresponding standalone width, active non-embedding parameter count,
ordinary-validation manifest, and evaluation role, then published 16 endpoint
rows, eight differences, and PNG/PDF plots. C4 endpoints are connected;
standalones are separate points. Each standalone appears only once, at 87,132
updates, in validation-progress plots. The report is descriptive for seed 42.
Equal expected FFN block owner calls do not imply equal training-example
coverage across widths.

### Terminal loss gaps

Positive values mean C4 has higher ordinary-validation loss. Neither C4 run
matched or improved its standalone endpoint at any width.

| Grid | Width | C4 loss | Standalone loss | C4 minus standalone |
| --- | --- | ---: | ---: | ---: |
| Linear | g250 | 2.069516 | 2.062977 | +0.006539 |
| Linear | g500 | 2.020171 | 1.991942 | +0.028229 |
| Linear | g750 | 2.000646 | 1.943363 | +0.057283 |
| Linear | g1000 | 1.988650 | 1.909057 | +0.079593 |
| Geometric | g125 | 2.116458 | 2.105963 | +0.010495 |
| Geometric | g250 | 2.075836 | 2.062977 | +0.012859 |
| Geometric | g500 | 2.040597 | 1.991942 | +0.048655 |
| Geometric | g1000 | 2.008321 | 1.909057 | +0.099264 |

At g1000, C4 perplexity is 7.305662 (linear) and 7.450795 (geometric),
versus standalone 6.746721. The respective perplexity gaps are +0.558941
and +0.704074. These are descriptive seed-42 comparisons.

### Comparison including S1

The additional `report-with-s1/` figures include the original matching S1 runs
with 64-update warmup from `optimizer-ownership-v1/runs/S1` (linear) and
`optimizer-ownership-matformer-widths-v1/runs/S1` (geometric). S1 and C4 each
completed 348,528 updates. The S1 terminal checkpoint hashes, width counts,
ordinary-validation role, and manifest are checked before plotting.

Generate these figures with
`python scripts/report_tinystories_c4_selected_block.py --include-s1`.
The directory contains PNG/PDF endpoint and progress plots, 24 endpoint rows,
and C4-minus-S1 differences. S1 uses a dashed curve; standalones remain scatter
points, appearing only at 87,132 updates in the progress figures.

The recovery command is
`python scripts/run_tinystories_c4_selected_block.py resume`. It does nothing
while either run remains live. After a failed, timed-out, or cancelled attempt,
it validates only that arm's own durable BF16 checkpoint before resubmission.
If either original training job fails, the report dependency must be updated to
the successful continuation job IDs before terminal reporting.
