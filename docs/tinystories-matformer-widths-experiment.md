# TinyStories MatFormer-width optimizer-ownership experiment

Campaign: `tinystories-optimizer-ownership-matformer-widths-v1`, recipe schema 4.
The current authorization covers implementation phases 1–4 (T001–T021).
Operational preparation, GPU diagnostics and production remain pending separate
authorization. Historical launch approvals do not apply to this campaign.

The [feature contracts](../specs/015-tinystories-matformer-widths/contracts/campaign-and-topology.md)
define the protocol; the [verification record](../specs/015-tinystories-matformer-widths/verification.md)
records actual evidence. Schema-4 expansion/preflight and optimizer semantics
are implemented and verified with CPU fixtures. Launcher, readiness-gate and
reporting commands below remain planned interfaces for later phases.

## Matrix and scientific controls

| Arm | Representation | AdamW history | L2 clipping cap | FFN dimension | Epochs |
| --- | --- | --- | --- | ---: | ---: |
| ST-g125 | Dense | Shared | Global 1 | 32 | 1 |
| ST-g250 | Dense | Shared | Global 1 | 64 | 1 |
| ST-g500 | Dense | Shared | Global 1 | 128 | 1 |
| ST-g1000 | Dense | Shared | Global 1 | 256 | 1 |
| S1 | Slicing | Shared | Global 1 | 256 | 4 |
| S2 | Slicing | Per width | Global 1 | 256 | 4 |
| C1 | Concat | Shared | Global 1 | 256 | 4 |
| C2 | Concat | Per width, lazy | Global 1 | 256 | 4 |
| C3 | Concat | Per block and common | Independently 1 per active owner | 256 | 4 |

Widths g125/g250/g500/g1000 mean FFN source fractions .125/.25/.5/1,
dimensions 32/64/128/256, and expected active non-embedding parameter counts
90,688/115,264/164,416/262,720. Actual CPU model checks confirm these counts. Dense models
retain source fractions while their local active fraction is 1. Concat boundaries
are A=[0,32), B=[32,64), C=[64,128), D=[128,256); C3 has five owners including
common parameters. Historical g750 remains dimension 192/count 213,568.

Use the original uncorrected controls: d64/l4/h4, context 128, vocabulary 2048,
initializer .02, seed 42, AdamW LR .008, betas (.9,.95), epsilon 1e-8, decay .1,
batch 64, accumulation 1, bf16 and one process/GPU. Cosine uses 64 warmup updates
and the full assigned horizon. LR scaling, pre-nested warmup and corrections are
disabled. Every elastic update draws one global width independently with
replacement at H=1 and uniform probabilities [.25]*4. Preserve independent action
and data streams; expected exposure is not a quota.

Retain all `PINNED_COMMON` controls except the declared width grid and all
`PINNED_DATA` hashes in `src/evaluation/optimizer_ownership.py`. Candidate inputs:

- Corpus: `/nfs-stor/ivo.navarrete/matformer-corpora/tinystories-instruct-packed-full-v1`.
- Tokenizer: `/nfs-stor/ivo.navarrete/matformer-tokenizers/tinystories-instruct-sentencepiece-bpe-2k-v1`.
- Historical reference: `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/reports/complete-20260910T181951Z/frozen/frozen_manifest.json`.

These inputs require fresh audits. An epoch has 5,576,448 designated sequences,
87,132 updates and 713,785,344 tokens; the same 43 excluded sequences never rotate
in. Each elastic receives 348,528 updates/2,855,141,376 tokens. All nine runs total
17,130,848,256 assigned tokens, excluding diagnostics/replay. Ordinary validation
runs every 64 updates and at completion, with target-token-weighted causal loss
and exp(loss) perplexity (expected 285 sequences/36,195 targets). Controller data
never guides training/selection, and final holdout remains sealed.

## Planned operations and artifact layout

The proposed fresh root is
`/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1`.
Its availability has not been audited or reserved. Future layout:

```text
source/       immutable tested snapshot and source hashes
campaign/     campaign_manifest.json, preflight.json, nine resolved configs
diagnostics/  cpu-gate.json, gpu-gate.json, distinct diagnostic runs
launchers/    intents, status.json, standalone-barrier.json, locks
logs/         per-job and per-attempt logs
runs/         <arm>/ controls, metrics, traces, checkpoints and terminal sidecars
reports/      frozen/, new/, combined/ and per-run diagnostics
```

Use `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python` and
`OMP_NUM_THREADS=1` for CPU verification. See the
[quickstart](../specs/015-tinystories-matformer-widths/quickstart.md) for planned
commands and the complete regression suite. The required order is:

1. Analyzer `preflight --campaign --prepared-corpus-dir --tokenizer-dir
   --output-dir --run-output-root`: audit inputs, nine actual models, full traces
   and fresh identities; publish configs and an exclusive reservation.
2. `preflight_tinystories_matformer_widths.py --mode cpu --campaign-root`: freeze
   a snapshot, test it and bind evidence to exact source/config/preflight hashes.
3. `run_tinystories_matformer_widths.py prepare --campaign-root --cpu-evidence
   [--reference-manifest]`: verify evidence, adopt its matching reservation and
   record the launch plan. No submissions.
4. In an authorized sbatch job, run the diagnostic script with `--mode gpu` and
   `--campaign-root`: real-shape batch-64/context-128 bf16 probes for all nine
   definitions, all-width semantics and separately labeled synthetic epoch-boundary
   resume/failure probes. CUDA skips do not pass this gate.
5. Launcher `queue --campaign-root [--once]`: reconcile intents/jobs and admit
   standalones. Internal `worker --campaign-root --arm --attempt-id` revalidates
   its own intent, gates, locks, source/config and stage evidence.
6. Strictly validate all four fresh full-budget standalone terminals before
   publishing `launchers/standalone-barrier.json` and admitting any elastic.
   Revalidate checkpoint/sidecar/supporting hashes on restart and worker entry.
   Scheduler completion or file existence alone cannot satisfy the barrier.
7. Analyzer `freeze --campaign-manifest --run-root --output-dir`, then `report
   --manifest --output-dir`, publish 24 new endpoints. `report-matformer-widths
   --manifest --reference-manifest --output-dir` adds only four revalidated
   historical standalones for 28 endpoints. Launcher `report --campaign-root
   [--reference-manifest]` will perform restart-safe finalization without GPU jobs.

All GPU work uses sbatch, excludes `gpu-[05,50,51]` and obeys live user-wide
two-running/four-submitted ceilings or stricter association/QoS limits. Count
unrelated jobs and preserve them. Defer admission if the running ceiling cannot
be guaranteed. Persist monotonic attempt intents before sbatch; uncertain
submissions require reconciliation against squeue/sacct/worker evidence before
retry. One writer holds each run lock.

Continue only from a valid checkpoint of the same run. Occupied invalid or
ambiguous state blocks fresh restart. Partial updates cannot replace the last
durable complete checkpoint. Terminal-sidecar recovery takes zero extra updates.
Retain failed/replayed attempt costs, sum each unique attempt's latest duration,
take maximum measured peaks, and disclose unknown hard-kill observations.
Keep overlapping scheduler allocation time separate from process time.

Source/config changes invalidate affected readiness evidence. Preparation cannot
relabel old evidence with current hashes. Diagnostics cannot initialize or satisfy
production terminals. Fresh IDs are `<campaign_id>-<arm>-s42`.

## Deliverables and current status

| Stage | Status | Required evidence |
| --- | --- | --- |
| Phase 1 setup | Complete | This runbook and verification record |
| Phase 2 foundations | Complete | Selectors, topology identity and all 20 legacy signatures verified; 269 passed, 6 CUDA-only skips across focused/regression checks |
| Phase 3 protocol/preflight | Complete on CPU | Nine actual models/counts, full expected traces and rejection/publication fixtures |
| Phase 4 ownership/accounting | Complete on CPU | All-width updates, C1/C3 sidecars, physical allocations and bounded g125 accounting; final regression 1,062 passed |
| Full CPU readiness | Pending | All new acceptance/regression suites, actual models and audited inputs |
| GPU diagnostics | Pending | Authorized sbatch results and valid bound GPU gate |
| Nine-run production | Pending | Four validated standalone terminals followed by five elastic terminals |
| New report | Pending | Nine validated runs, matching 24-row endpoints.csv/json and diagnostics |
| Combined report | Pending | Four valid historical standalones, matching 28-row combined_endpoints.csv/json |

Combined figures are `loss_vs_parameters.{png,pdf}` and
`perplexity_vs_parameters.{png,pdf}`. Each has five connected four-point elastic
curves and eight disconnected standalone markers. Preserve both measurements at
64/128/256, distinguish coincident markers at exact coordinates, and use legends
`Standalone — historical grid` / `Standalone — MatFormer grid`.

New-report success remains valid when history is unavailable; combined completion
stays outstanding with an explicit reason and a nonzero requested-comparison
result. Validate saved manifests/hashes before reusing outputs. Interpret only
saved seed-42 results, primarily against fresh baselines; distinguish C3 clipping
and changed block sizes from representation/history effects. Equal tokens do not
establish equal compute, runtime or direct width exposure.
