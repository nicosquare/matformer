# TinyStories MatFormer-width optimizer-ownership experiment

Campaign: `tinystories-optimizer-ownership-matformer-widths-v1`, recipe schema 4.
The 2026-09-21 continuation explicitly authorizes operational preparation, GPU
diagnostics and all Slurm submissions/resumptions necessary for T034–T037.
Implementation and reporting prerequisites are already available. Actual reports
T045/T052 remain separate pending tasks.

The requested interim comparison of both standalone grids is available in
`reports/standalone-comparison-20260922-v2/` under the production campaign root:
separate `loss_vs_parameters.{png,pdf}` and `perplexity_vs_parameters.{png,pdf}`,
endpoint JSON/CSV and a provenance manifest.
All eight terminal endpoints were strictly revalidated. Shared sizes have
identical saved loss/perplexity. The legends use Linear (blue triangles) and
Geometric (orange triangles); coincident points use one split-color triangle. This
standalone-only plot leaves the full T045/T052 reports pending; reproduction and
hash evidence are in the verification record.

The [feature contracts](../specs/015-tinystories-matformer-widths/contracts/campaign-and-topology.md)
define the protocol; the [verification record](../specs/015-tinystories-matformer-widths/verification.md)
records actual evidence. Schema-4 expansion/preflight and optimizer semantics
are implemented and verified with CPU fixtures. The launcher, diagnostic modes and
24/28-endpoint reporting interfaces are implemented; their current CPU verification
is recorded separately from operational readiness below.

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

The real-input preflight audit passed on 2026-09-21; see the verification ledger. An epoch has 5,576,448 designated sequences,
87,132 updates and 713,785,344 tokens; the same 43 excluded sequences never rotate
in. Each elastic receives 348,528 updates/2,855,141,376 tokens. All nine runs total
17,130,848,256 assigned tokens, excluding diagnostics/replay. Ordinary validation
runs every 64 updates and at completion, with target-token-weighted causal loss
and exp(loss) perplexity (expected 285 sequences/36,195 targets). Controller data
never guides training/selection, and final holdout remains sealed.

## Operations and artifact layout

The proposed fresh root is
`/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1`.
It was verified unused and reserved by the successful real-input preflight on
2026-09-21. Campaign layout:

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
[quickstart](../specs/015-tinystories-matformer-widths/quickstart.md) for implemented
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
   [--reference-manifest]` performs restart-safe finalization without GPU jobs.

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


The CPU diagnostic freezes runtime modules, scripts, recipes and tests before
running the complete acceptance/regression command from `source/`. Its gate binds
source, config-set, campaign manifest, preflight, environment, command, JUnit and
log hashes. The read-only snapshot includes provenance for operation outside Git.
Preparation adopts the exact preflight reservation and never changes tested hashes.

The GPU command must run from that snapshot in an sbatch allocation. It checks
live limits and retains scheduler allocation evidence. All nine real-corpus probes
use batch 64/context 128/bf16 and their full assigned scheduler horizons, stopping
at four updates and resuming to eight. Only diagnostic run identity and output
path differ; the local diagnostic adapter validates every original scientific
control. Separate forced-width semantic probes and synthetic two-update epochs
exercise numerical resume at steps 1/2/3, malformed bundles and partial failures.
These are diagnostic overrides, never production terminals or full-budget evidence.

On restart, unknown scheduler states or uncertain submissions block admission;
`squeue`, `sacct` and durable worker evidence must identify a unique attempt.
A confirmed ended attempt with its valid own checkpoint can receive the next
monotonic attempt ID. An occupied output without a checkpoint cannot start fresh.
`launchers/admission-error.json` records blocked reconciliation. Per-attempt worker
records and the resource ledger retain process UUID, job and launch identity.
A process killed before its first measurement has null observations and explicit
incomplete costs; recorded totals are lower bounds. Scheduler allocation seconds
remain in submission accounting, separate from overlapping process measurements.

`report` revalidates frozen inputs and published output hashes before reuse.
It persists independent `production`, `new_report` and `combined_report` statuses
in `launchers/completion.json`. Missing history returns nonzero after preserving
the valid new report. A changed prior publication fails explicitly; it is never
silently overwritten or accepted from directory existence.

## Deliverables and current status

| Stage | Status | Required evidence |
| --- | --- | --- |
| Phase 1 setup | Complete | This runbook and verification record |
| Phase 2 foundations | Complete | Selectors, topology identity and all 20 legacy signatures verified; 269 passed, 6 CUDA-only skips across focused/regression checks |
| Phase 3 protocol/preflight | Complete on CPU | Nine actual models/counts, full expected traces and rejection/publication fixtures |
| Phase 4 ownership/accounting | Complete on CPU | All-width updates, C1/C3 sidecars, physical allocations and bounded g125 accounting; final regression 1,062 passed |
| Phase 5 and reporting implementation | Complete on CPU | Restore/failure, gate/barrier/queue/worker fixtures and 24/28 exports; final regression 1,302 passed, 39 GPU-only skips, 1 existing expected failure |
| Full snapshot-bound CPU readiness | T034 complete | Audited inputs; immutable source CPU gate: 1,307 passed, 39 GPU skips, 1 existing xfail; prepared reservation |
| GPU diagnostics | T035 complete | Job 271285 COMPLETED/0:0 on A100 gpu-52; nine real-shape probes + 118 GPU tests, no skips; all gate/result hashes validated |
| Nine-run production | T036 complete; T037 elastic admission underway | Four standalones COMPLETED/0:0 and strictly validated; barrier published/revalidated; five elastic terminals pending |
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

Operational correction: CPU/GPU pytest checks use a writable diagnostic workspace
whose source/config/test entries link to the immutable snapshot. Default fixture
outputs stay outside the snapshot. On hosts lacking `sacctmgr` or an idle user's
QoS usage row, the launcher compiles the small read-only
`scripts/matformer_slurm_limits.c` query with installed GCC/Slurm headers and
reads configured QoS and user/ancestor association limits directly from the
controller. Missing query support fails admission. No limits are inferred from
another user's usage. The initial failed snapshot/checks are preserved under
`diagnostics/superseded-before-operational-fixes` and their `cpu-*` directories.

Diagnostic job **271285** (`mw-v1-gpu-diagnostic-a1`) completed successfully;
T036 standalone admission follows its fully validated GPU gate. Its command, bindings and scheduler evidence
are in `launchers/diagnostic-submission-1.json`; logs are
`logs/diagnostic-a1-271285.{out,err}`. Set
`SLURM_CONF=$MW_ROOT/launchers/slurm-client.conf` when continuing on this host:
the campaign-local file corrects only the accounting endpoint from controller-
local `localhost` to `ciai-head`. The original export and both hashes are saved;
server policy is unchanged. The verification record contains exact scheduler,
gate-validation and next queue commands. T035 passed actual GPU validation;
T036 still requires four validated standalones/barrier and T037 the five elastics
and reconciled traces/costs. T045/T052 actual reports remain independently pending.

Initial production handoff (superseded by the 2026-09-22 recovery below): **271319 (ST-g125)** and **271320 (ST-g250)** are
RUNNING on gpu-52; **271321 (ST-g500)** and **271322 (ST-g1000)** are PENDING.
All four are attempt 1, fresh own identities, 87,132 assigned updates, one GPU/task,
24-hour allocation limits, with the required exclusions. Live user-wide count:
two running/four submitted. See `launchers/submissions.json`,
`launchers/handoff-standalones.json`, `logs/<arm>-a1-<jobid>.{out,err}` and
`runs/<arm>/`. The barrier is absent and no elastic was admitted. After completion
notification, query actual scheduler status/artifacts and run the snapshot queue
with `--once`; it handles valid own-checkpoint continuations and strictly validates
all four terminals before barrier publication/elastic admission. Existing
execution authorization persists. T036/T037 and actual report T045/T052 remain
incomplete; holdout evaluation count remains zero.

On the user's subsequent request, an automatic background checker is now running
on **ciai-login-1**, PID **1407294**. It invokes the verified snapshot's continuous
queue and checks again 30 seconds after each completed cycle. The first live cycle
passed without duplicate submissions. It automatically admits remaining elastic
jobs as capacity opens **after all four standalones pass the strict barrier**,
and resumes only valid own-run checkpoints. Invalid/uncertain evidence stops it
with a logged error. It exits when the queue has validated all nine terminals;
actual reports and final evidence/task reconciliation remain separate.

Monitor `launchers/background-checker.json`, `launchers/status.json`,
`launchers/submissions.json` and `logs/background-checker.log`. The wrapper is
`launchers/background-checker.py`, outside the immutable source snapshot. It is
detached from the chat session, but is not a reboot service. The verification
record documents its exact provenance and safe stop/manual-takeover instructions.
Do not start a competing queue while this checker holds the queue lock.

## 2026-09-22 validation and recovery

All four standalone jobs 271319–271322 completed with ExitCode 0:0 and their full
87,132-update/713,785,344-token budgets. The first checker stopped when the strict
terminal reader detected stale elapsed-time summaries: the trainer's `finally`
block recorded another completion observation after summary publication.
Final ledger durations exceeded the summaries by 12–18 ms; counts and peaks
agreed. No elastic had been admitted at that failure.

Original summaries and full checker failure records are preserved. A constrained
CPU artifact helper now reconciles only elapsed time and derived throughput from
the authoritative final ledger after successful scheduler/worker outcomes; it
requires staged and post-publication acceptance by the unchanged strict terminal
reader. Four actual terminals passed. The helper's 14 regression tests passed.
No checkpoint, evaluation, trace, training source/config or readiness gate changed;
no training/evaluation rerun occurred. See the verification record for hashes,
measurements and `launchers/resource-reconciliations/` for sealed receipts.

**T036 is complete:** `launchers/standalone-barrier.json` is published and
revalidated, content hash
`f30938b189747b1672fac07335ca2ab4b01d71317be34854e48527fe6da8d378`.
A detached checker, PID **1212882** on ciai-login-1, reconciles completed resource
records before each verified snapshot queue cycle and admits T037 elastics within
live user-wide limits. It still stops on invalid/uncertain evidence and never
relaxes strict validation. Current jobs are in `launchers/submissions.json` and
`launchers/status.json`; its log is `logs/background-checker.log`. The first
checker is archived at `launchers/background-checker-failure-20260922/`.
T037 and actual reports T045/T052 remain incomplete; holdout remains sealed.

Current T037 jobs: **S1 272204** and **S2 272205** RUNNING on gpu-54;
**C1 272207** and **C2 272208** PENDING. **C3** awaits a submitted-job slot and
will be admitted automatically. The first restarted checker cycle completed
successfully, with two running/four submitted user jobs. Full details are in
`launchers/handoff-elastics-20260922.json` and the verification record. T036 is
checked complete; T037 and actual reporting T045/T052 remain unchecked.
