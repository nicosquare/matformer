# TinyStories S1 Fourfold LR Warmup

## Current scope

Phases 3–4 implement schema-5 preparation, grid-aware runtime accounting and
terminal metadata, immutable source/CPU/GPU diagnostics, and the two-arm launcher.
CPU fixtures verify protocol, continuation, failure durability and admission.
Reporting implementation (phase 5), real input/reference audits, source-bound
readiness and production remain pending. See independent stage
statuses and test evidence in [verification.md](../specs/016-tinystories-s1-warmup/verification.md).

## Fixed experiment

Campaign `tinystories-optimizer-ownership-s1-warmup-v1`, schema 5, has exactly
two fresh seed-42 runs. Run IDs are `<campaign_id>-<arm_id>-s42`.

| Arm / grid | Ordered widths | FFN dimensions | Active non-embedding parameters |
| --- | --- | --- | --- |
| S1-linear-w256 / Linear | g250, g500, g750, g1000 | 64, 128, 192, 256 | 115264, 164416, 213568, 262720 |
| S1-geometric-w256 / Geometric | g125, g250, g500, g1000 | 32, 64, 128, 256 | 90688, 115264, 164416, 262720 |

Counts exclude input embeddings and the LM head and are verified against
fresh constructed CPU models in the phase-3 tests. Both runs use d64/l4/h4, full FFN256,
context128, vocabulary2048 and initializer .02. Use fresh normal construction,
slicing, shared AdamW history, betas (.9,.95), epsilon 1e-8, weight decay .1,
global L2 clipping cap1, batch64, accumulation1 and bf16. Sampling is one uniform
global draw with replacement per update (H=1), with correction none. Pre-nested
warmup, LR scaling and activation checkpointing are disabled. Preserve version-1
initialization/action/data streams and each run's exact own-counterpart traces.

Only LR warmup changes scientifically: 64 to 256 updates, inside the unchanged
cosine horizon with peak LR .008. Each run has four epochs, 348,528 updates and
2,855,141,376 training tokens; together, 697,056 updates and 5,710,282,752 tokens.
One epoch is 87,132 updates / 713,785,344 tokens. Each update consumes 8,192 tokens.
Use 5,576,448 designated sequences per epoch and exclude the fixed tail of 43.
Validate every 64 updates and at completion on ordinary validation only;
controller and final-holdout roles remain sealed.

Scheduler position p is applied at update p+1. Position0/update1 has LR0;
position256 is peak and is first applied at update257. Position348528 stores
terminal LR0 and has no next applied update. Expected schedule values are not
measured execution evidence. See the [schedule contract](../specs/016-tinystories-s1-warmup/contracts/protocol-and-schedule.md).

## Read-only historical selection

| Grid | Reference root | Selected arms |
| --- | --- | --- |
| Linear (schema 1) | `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1` | ST-g250, ST-g500, ST-g750, ST-g1000, S1 |
| Geometric (schema 4) | `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1` | ST-g125, ST-g250, ST-g500, ST-g1000, S1 |

The saved `campaign/configs/S1.yaml` hashes pinned during planning are:

- Linear: `ef16f8a6cb883abe6782b9201f32b6d137b324f4d0cd5ac112f66b6fb57ac3fa`.
- Geometric: `781254c7bbd8a7fdc48ba4c09bdc940dd44e132c73d1fe9d8b9209796d94a6c8`.

Revalidate ten selected terminals, their controls, full budgets, traces,
ordinary-validation provenance and applicable device/job evidence. Geometric
S1 attempt2/job272716 is a planning observation requiring revalidation; archived
CPU attempt1 is invalid. Older Linear evidence uses its applicable generation
of launcher/job/precision/resource records. Do not require cancelled Feature015
arms, rewrite historical files, retrain references or merge distinct same-size
standalones. New selection records belong under the new root. Standalones have
one epoch each; both historical and new S1 runs have four.

## Artifacts and CLI

Proposed fresh root (not reserved):
`/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-s1-warmup-v1`.

```text
<root>/
  campaign/          # manifest, configs, control differences, schedules/traces
  campaign/references/ # selection records, published atomically with preflight
  source/            # immutable tested executable snapshot
  diagnostics/       # source manifest, CPU/GPU gates, probe evidence
  launchers/         # plan, submission intents, attempts, device/worker status
  logs/
  runs/S1-linear-w256/
  runs/S1-geometric-w256/
  reports/new/       # eight endpoints and frozen manifest
  reports/comparison/ # 24 endpoints, deltas, early data, figures, findings
```

Use `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python` from the repository
root. Preflight, diagnostics and prepare/queue/worker are implemented. The
reporting interfaces in the last three rows remain phase-5 work:

| Entry point | Arguments | Contract |
| --- | --- | --- |
| `scripts/analyze_tinystories_optimizer_ownership.py` | `preflight --campaign RECIPE --prepared-corpus-dir CORPUS --tokenizer-dir TOKENIZER --output-dir ROOT/campaign --run-output-root ROOT/runs --linear-reference-root LINEAR --geometric-reference-root GEOMETRIC` | Audit before atomic publication; reject occupied conflicting identities |
| `scripts/preflight_tinystories_s1_warmup.py` | `cpu --campaign-root ROOT` | Freeze and test exact source/configs; publish CPU gate |
| Same | `submit-gpu --campaign-root ROOT` | Explicit checked diagnostic submission; persist intent before sbatch |
| Same | `gpu --campaign-root ROOT` | Internal sbatch-only real-shape bf16 diagnostic worker for both grids |
| `scripts/run_tinystories_s1_warmup.py` | `prepare --campaign-root ROOT --cpu-evidence ROOT/diagnostics/cpu-gate.json` | Verify reservation and bindings; publish two-arm plan |
| Same | `queue --campaign-root ROOT [--once]` | Verify gates, reconcile attempts, enforce live limits before submission |
| Same | `worker --campaign-root ROOT --arm ARM --attempt-id N` | Internal sbatch worker; CUDA-required trainer and own continuation only |
| Analyzer | `freeze --campaign-manifest ROOT/campaign/campaign_manifest.json --run-root ROOT/runs --output-dir DEST` | Validate two full terminals and eight new endpoints |
| Analyzer | `report-s1-warmup --manifest FROZEN --linear-reference-root LINEAR --geometric-reference-root GEOMETRIC --output-dir DEST [--early-end-step 1024]` | Strict saved-artifact comparison; early end step >=1024 |
| Launcher | `report --campaign-root ROOT` | New-only freeze and comparison recovery without training/submission |

Recipe: `configs/controlled_exps/tinystories_instruct_s1_warmup.yaml`.
Corpus: `/nfs-stor/ivo.navarrete/matformer-corpora/tinystories-instruct-packed-full-v1`.
Tokenizer: `/nfs-stor/ivo.navarrete/matformer-tokenizers/tinystories-instruct-sentencepiece-bpe-2k-v1`.
The quickstart provides the proposed full command sequence with these paths.
After preparation, use the frozen `ROOT/source/scripts/` entry points. Changed
source/config/reference bindings invalidate gates. Missing required evidence
returns an explicit nonzero failure; queue capacity records a pending wait.

## Authorization, execution and recovery

The current request authorizes phase 4 (T017–T035) software and fixture checks,
building on phases 1–3. Real-input preparation remains T051–T052. GPU
readiness T053 requires subsequent diagnostic authorization; production T054
requires subsequent production authorization and passed CPU/GPU gates. Earlier
campaign approvals and successful tests do not authorize these stages.

All GPU work uses checked sbatch admission, one process/GPU, excluded nodes
`gpu-[05,50,51,54]`, and user-wide ceilings of two running/four submitted jobs
or stricter live limits. Include unrelated jobs and active reservations; do not
bypass admission with a bare sbatch example. Mandatory GPU skips cannot pass
readiness. Real d64/l4/h4, batch64, context128 bf16 diagnostics must cover both
grids, all widths and schedule/continuation/failure boundaries.

Re-running the same queue reconciles durable submission intents and accounting.
An uncertain submission blocks duplication. Locks protect queue/per-run writers;
monotonic attempt IDs/UUIDs and checkpoint hashes identify continuation. Resume
only the run's own validated complete bundle. Occupied identities without valid
continuation require reconciliation. Full-horizon recovery regenerates missing
terminal outputs with zero extra updates. Terminal completion requires matching
successful worker and Slurm evidence as well as checkpoints/evaluation/traces;
delayed accounting remains pending. Record failed/replayed resource costs
separately from committed progress and disclose missing measurements.

## Report acceptance

Preserve eight valid new endpoints independently. Complete comparison requires
24 endpoints (12 per grid), eight new-minus-old loss/perplexity differences and
both S1-minus-own-grid-standalone gaps in CSV/JSON. Key by grid, campaign, run and
physical width. Use terminal target-token-weighted ordinary-validation loss and
exp(loss), never best checkpoints or trailing averages.

For each grid, publish loss/parameter and perplexity/parameter figures plus
early LR/loss figures, each in PNG/PDF: sixteen files total. Endpoint plots use
four disconnected standalone markers and two four-point S1 curves at exact
parameter counts. Early views cover at least updates0–1024 and mark64/256;
retain raw recorded training LR/loss and per-width validation observations,
separate training/validation panels, provenance and gaps. Never invent step-zero
loss; schedule-derived LR must be labeled reconstructed. Ambiguous replay rows
fail validation. Findings describe all eight effects and standalone gaps as
seed-42 observations, without significance or equal-compute claims.

Missing references or early evidence leave comparison/report completion pending
while preserving valid new-only results. The [report contract](../specs/016-tinystories-s1-warmup/contracts/reporting.md)
defines exact filenames, fields, formulas and rejection cases.

## Implemented CPU preparation details

Preflight requires both reference-root flags for schema 5. It checks all inputs
before creating the new campaign artifacts, then publishes configs, schedules,
traces, control differences and `campaign/references/selection.json` together.
The reference-selection record stays inside `campaign/` to preserve one atomic
directory publication. It reserves the two run IDs with the existing sibling
`.runs.optimizer-ownership-reservation.json` mechanism. It does not create run
directories or launch training.

An identical prepared identity can be verified again without rewriting files;
changed sources/configs/references, occupied run directories or unrelated root
contents are rejected. Expected schedules contain every position 0–348528 and
separately identify the applied update (blank at the terminal position).
Historical job checks use read-only `sacct`; scheduler access is mocked in tests.
Older saved configs receive only explicitly enumerated, disabled diagnostic
metadata allowances. Runtime-added observations do not replace saved controls.


## Phase-4 operational procedures

These commands are implemented and checked with CPU fixtures. They are not
records of running the real experiment. Set `WARMUP_ROOT` and `WARMUP_PYTHON`
as in the [quickstart](../specs/016-tinystories-s1-warmup/quickstart.md).

After the separately scheduled real preflight at T051:

```bash
"$WARMUP_PYTHON" scripts/preflight_tinystories_s1_warmup.py cpu --campaign-root "$WARMUP_ROOT"
"$WARMUP_PYTHON" "$WARMUP_ROOT/source/scripts/run_tinystories_s1_warmup.py" prepare \
  --campaign-root "$WARMUP_ROOT" --cpu-evidence "$WARMUP_ROOT/diagnostics/cpu-gate.json"
```

The CPU command copies `src/`, `scripts/`, `configs/`, `tests/` and root Python
build/entry files into an immutable, permission-read-only `source/` directory.
It records every file in `diagnostics/source-manifest.json`, then invokes the
frozen diagnostic entry. Pytest runs from a writable workspace of symlinks to
that source, with CUDA hidden. Results, commands, JUnit counts and log hashes
are saved in `diagnostics/cpu-<uuid>/`; the sealed canonical result is
`diagnostics/cpu-gate.json`. A failed run publishes failed evidence. An existing
snapshot must match its recorded bytes; it is never overwritten or relabeled.

Bindings cover recipe, executable config set, manifest, scientific contracts,
control/trace/schedule artifacts, selected historical sources and executable
snapshot. `prepare` verifies the preflight reservation and refuses occupied
runs; repeating an identical prepare verifies the existing plan. The checkout's
prepare convenience command forwards to the frozen executable. Queue and
workers must be invoked from that snapshot.

After T053 diagnostic authorization, use the checked submission path:

```bash
"$WARMUP_PYTHON" "$WARMUP_ROOT/source/scripts/preflight_tinystories_s1_warmup.py" submit-gpu \
  --campaign-root "$WARMUP_ROOT"
```

Re-run this same command to reconcile an outstanding diagnostic. It does not
submit again while queue/accounting visibility is uncertain. Evidence lives in
`diagnostics/submissions.json`, `diagnostics/worker-<intent_uuid>.json`,
`diagnostics/gpu-<uuid>/` and `diagnostics/gpu-gate.json`; diagnostic Slurm logs
are `logs/diagnostic-a<N>-<job>.out/.err`. Diagnostic runs stay under their own
GPU artifact directory, including resource ledgers, and never under production
`runs/`.

The GPU entry checks both canonical configs at real d64/l4/h4, batch64,
context128 and bf16, observes CUDA bf16 autocast in actual forwards, and executes
eight real-corpus updates per grid with restore at step4. It also runs the
runtime acceptance matrix with real-shape synthetic batches on CUDA: actual
updates through258, all widths, warmup/epoch/terminal continuation, invalid
restores, partial failures and terminal-only trainer reentry. Late-boundary
histories are explicitly state-seeded, not evidence of prior training. A GPU
skip cannot pass; the gate additionally requires matching successful diagnostic
worker and Slurm accounting. Until accounting arrives, readiness is pending.

Phase4's CPU gate sets `reporting_fixture_status=pending`. Production admission
explicitly rejects that status even if runtime diagnostics pass. Phase5 must
add and pass the reporting fixture gate before any production launch; refresh
the tested snapshot and affected gates after implementation changes. Do not
manually change a gate to bypass this prerequisite.

After phase5, all readiness evidence, and T054 production authorization:

```bash
"$WARMUP_PYTHON" "$WARMUP_ROOT/source/scripts/run_tinystories_s1_warmup.py" queue \
  --campaign-root "$WARMUP_ROOT" --once
```

Omit `--once` for reconciliation every30 seconds. Queue/diagnostic admission
share `launchers/admission.lock`; queue and each arm have separate writer locks.
Every pending or uncertain job reserves a running slot, including unrelated
jobs and delayed-visibility submission IDs. This can deliberately admit fewer
than four pending jobs to keep the running ceiling safe.

`launchers/submissions.json` is durable before sbatch, and records monotonic
attempt numbers, UUIDs, source/gate bindings, command and own-checkpoint hash.
Worker records are `launchers/worker-<arm>-<N>.json`; explicit CUDA entry evidence
is `launchers/cuda-entry-<arm>-<N>.json`. `launchers/status.json` keeps readiness,
production, per-terminal, new-report, comparison and early-report stages
separate. Admission errors are saved in `launchers/admission-error.json`.

Restart the same queue to continue from a validated own `checkpoints/latest.pt`.
An occupied identity with no valid checkpoint is blocked. A full-horizon
checkpoint selects `completion_only`, invoking the same CUDA-required trainer
to restore the complete bundle and recover terminal validation/summary without
another optimizer or scheduler update. A sidecar alone cannot complete a run:
require successful matching job/worker, bf16 config, CUDA allocator evidence,
valid complete bundle, four endpoints and reconciled full action/batch traces.
Delayed accounting yields pending status and never grants an automatic retry.

`resource_attempts.json` retains failed/replayed process costs independently of
checkpoint progress. Worker/queue records disclose missing observations and
report Slurm allocation seconds separately, since these overlap process time.
A lost worker can remain an explicitly incomplete cost observation; it does not
silently become zero cost or invalidate earlier valid committed work.
