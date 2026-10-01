# TinyStories C4 separate correction campaign

Four fresh seed-42 runs compare GMC only and LMC only on the linear and
geometric grids. The campaign identity and output root are
`tinystories-optimizer-ownership-c4-separate-corrections-v1` under
`/nfs-stor/ivo.navarrete/results/elasticnn/`. Historical campaigns are read-only.

The selected-width prefix participates in forward/backward. Only its newly
introduced FFN block and common owner step. Earlier gradients are cleared
before one joint L2 cap of 1.0. Each owner retains its own AdamW history, and
one global cosine clock advances after the complete update.

Correction version 1 is explicit in `training.c4_correction`, the optimizer
checkpoint contract, and the optimizer collection state. Its configured forward
membership factors are A=1, B=4/3, C=2, D=4, common=1 on both grids. C4 already
has equal expected block update counts. These factors test forward membership
correction, not compensation for unequal C4 optimizer selection frequencies.

GMC only uses the existing backward hooks before joint clipping, with nominal
LR for all owners. LMC only resolves `model.correction_mode: none` (no gradient
hooks) and applies the factor to the selected owner's LR inside AdamW. This
scales adaptive update and decoupled decay once, without scaling moments or
counters. The LR is restored in a `finally` block before another owner or the
scheduler runs. Common uses nominal LR. No post-step delta multiplier is used.
Legacy `correction_mode: lmc` remains combined GMC plus post-step LMC; historical
identities, signatures and restore schemas are preserved.

Controls remain four epochs, 348,528 updates, 2,855,141,376 tokens, batch 64,
context 128, d64/l4/h4/vocab2048, BF16, seed 42, fresh normal initialization,
peak nominal LR .008, 64-step warmup, AdamW (.9,.95), epsilon 1e-8, decay .1.
LMC effective block peak LRs are .008, .010666666666666666, .016, .032.

Prepare and gate with:

```bash
python scripts/run_tinystories_c4_separate_corrections.py prepare
python scripts/run_tinystories_c4_separate_corrections.py diagnostic
python scripts/run_tinystories_c4_separate_corrections.py submit
```

Preparation freezes source/config SHA256 identities and runs CPU regressions,
read-only terminal inspections, control differences and action/data audits.
The BF16 gate verifies all four arms at real shape and exercises real-corpus
trainer checkpoints around warmup steps 63–65, with the production horizon
retained. Boundary schedule probes at later positions are state-seeded tests,
not full-budget training evidence. Failed logs are retained. Production requires
matching successful CPU and CUDA gates. Queue admission uses the existing
Slurm partition/QoS, exclusion `gpu-[05,50-51,54]`, and verified user-wide
ceilings of two running and four submitted jobs (or stricter live limits).

`status` reads recorded intents and the live queue. `resume` admits only an
arm's own validated durable BF16 checkpoint after its previous attempt has
stopped; it avoids duplicate live jobs. Owner-step or post-mutation failures
poison the logical update and cannot publish it as a successful checkpoint.

Per-training-row `correction_context` records nominal/effective owner LRs,
factors, versioned correction semantics and joint clipping evidence. The final
successful worker runs the terminal report automatically when all four summaries
are completed. Missing terminals leave `diagnostics/comparison-status.json`
incomplete; no intermediate checkpoint substitutes for a terminal.

The report publishes 32 grid-qualified endpoints, 16 corrected-minus-uncorrected
and 16 corrected-minus-standalone comparisons for loss and perplexity, hashes,
and PNG/PDF plots. Endpoint C4 curves are connected and standalones are separate
scatter points. Progress plots use recorded C4 trajectories and one standalone
point at 87,132 updates, with LMC labeled “LMC only, no GMC”. C1–C3 and S1 are
excluded. These are descriptive seed-42 results, not multi-seed robustness.

## Launch evidence on 2026-10-01

The immutable snapshot passed 791 CPU tests with 38 expected GPU skips.
CUDA gate 289806 passed on gpu-06, including four CUDA isolation/manual-AdamW
tests, all four real-shape BF16 arms, and real-corpus checkpoint resume at
steps 63–65. The initial local CPU gate lacked captured Git provenance; its
source and failure logs are preserved in
`diagnostics/revisions/initial-local-cpu-failure/`. The corrected gate captures
`source-provenance.json` and passed all regressions.

| Arm | Slurm job | State at launch verification |
| --- | --- | --- |
| C4-linear-GMC | 289809 | Running on gpu-06, committing BF16 updates |
| C4-linear-LMC-only | 289810 | Running on gpu-53, committing BF16 updates |
| C4-geometric-GMC | 289811 | Queued under the two-running limit |
| C4-geometric-LMC-only | 289812 | Queued under the two-running limit |

At update 65 (`g1000`), GMC nominal and effective D/common LRs were .008.
LMC-only nominal LR was .008, effective D LR .032, and common LR .008.
The two running arms' own checkpoints were inspected for committed step,
correction metadata, owner histories/counters, scheduler and reproducibility
contracts. Their first 64 actual width choices and batch provenance match the
saved uncorrected C4 run. See `diagnostics/production-entry-audit.json`.
The geometric arms have readiness evidence but no production-entry evidence
until Slurm schedules them. Durable CUDA admission is recorded in
`diagnostics/gpu-gate-admission.json` for later recovery.

All runs and the 32-endpoint terminal comparison remain incomplete at this
launch verification. The final successful worker automatically publishes the
comparison after all four full-budget summaries complete.

## Linear comparison with S1

After both linear correction arms completed, publish their ordinary-validation
terminals with original S1 (64-update warmup), uncorrected C4 and four individual
standalone endpoints using:

```bash
python scripts/report_tinystories_c4_separate_corrections.py --grid linear --include-s1
```

The separate `report-linear-with-s1/` directory contains 20 endpoints, 24 paired
differences, and loss, perplexity and validation-progress figures in PNG/PDF.
Only completed selected grids are required. The campaign's frozen production
source and its original four-arm reporting contract are preserved.

At g1000, loss is 1.989702 for GMC only, 1.986851 for LMC only,
1.988650 for uncorrected C4, 1.953872 for S1, and 1.909057 for standalone.
LMC only slightly improves uncorrected C4 but remains above S1 and standalone.
At g250, LMC only reaches a slightly lower loss than standalone (2.062595 versus
2.062977). These remain descriptive seed-42 results.

## Geometric runs stopped by user

On 2026-10-01, the user cancelled geometric GMC job 289811 and geometric
LMC-only job 289812 to avoid further compute. Slurm confirmed both CANCELLED
and neither remained in the queue. Their existing checkpoints and logs are
retained. Production intents record `cancelled_by_user`; the launcher skips
these arms during recovery. The geometric arms have no full-budget terminals
and the four-arm campaign comparison remains incomplete. The completed linear
comparison with S1 is preserved in `report-linear-with-s1/`.
