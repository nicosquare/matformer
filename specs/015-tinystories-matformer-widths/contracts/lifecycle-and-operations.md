# Optimizer Lifecycle, Readiness and Operations Contract

## Complete updates and observations

Reuse the existing trainer, optimizer collections and global scheduler. For each
elastic update: select one global width, clear all gradients to absent, consume
one batch, perform one causal forward/backward, measure/apply clipping, step
owners, advance the scheduler once, reconcile accounting, then publish committed
observations. Width and data RNG streams remain independent.

S1/S2/C1/C2/standalones use one global L2 cap 1.0. C3 clips each active owner's
combined gradients independently to 1.0, steps active blocks in A/B/C/D order and
then common once at the same current global LR. Synchronize all collection rates
from the one global clock. No second global clip or block-size cap scaling.

C1/C3 sidecars use semantic representation/scope eligibility, not literal run or
arm names. Pass the new validated topology into C1 group construction. Record
active flags, pre/post norms, coefficients/caps, combined norms and selected
width for every committed update. Inactive groups have null measurements; active
zero gradients have coefficient 1 and normal AdamW behavior. C1 group observations
describe its single applied global coefficient. Summary/metrics references and
committed counts must agree with the sidecar and strict terminal reader.

Compact metric attempt accounting accepts exactly the validated campaign widths,
including g125. Preserve existing schema/counter behavior for legacy restores;
do not accumulate an unbounded list of attempt identifiers. Derive expected and
observed selections/block activations/storage from actual grid/support. Persist
measured allocations by owner/component/dtype, separating moments, counters,
temporary estimates, device peaks and checkpoint bytes. Unequal C2 moment
expectations use `2*(4*F_A+3*F_B+2*F_C+F_D+4*R)` after full width exposure.

## Restore and failure safety

Before mutating live state, validate checkpoint purpose/schema; full campaign,
run and arm identity; explicit fractions/dimensions/boundaries; topology and
ordered parameter descriptors; optimizer groups/kwargs and numerical states;
required versus lazy histories; clipping; exact probabilities/RNG stream; full
scheduler horizon/position/rates; data membership/epoch cursor; and committed
counts/metrics/resource watermark. Use temporary replay RNG/sampler objects.
Installation retains the existing whole-bundle rollback guard.

Retain full-shaped S2 state, zero never-used tail moments and support-driven lazy
C2 absence. Reject model-only, cross-arm, old-campaign, changed-grid, wrong-boundary,
nonfinite, missing-required and impossible-extra history payloads before mutation.
Wrong identity cannot be repaired by replacing labels or hashes in saved data.

Set update-in-flight before the first potentially mutating optimizer call. Owner,
scheduler or accounting failure after mutation poisons the attempt and aborts;
normal, exception, signal and finalization save paths all reject partial state.
Preserve the last durable complete checkpoint. Replayed scientific trace suffixes
are reconciled to that boundary while failed/replayed resource costs remain.

At full budget, a durable terminal checkpoint with a missing sidecar enters
completion-only recovery: restore that exact state, complete ordinary validation
and sidecar publication, and take zero further optimizer steps. Never replace it
with best/trailing metrics or a changed terminal checkpoint identity.

## Evidence gates

CPU and GPU gates record status, tested source snapshot hash, exact config hashes,
preflight manifest hash, environment, commands/results, probe identities and
overrides; GPU evidence also records Slurm job/hardware and real bf16 execution.
Source hash covers runtime modules, entry scripts, tests and recipes used by the
checks. Mutable logs/evidence are excluded from the snapshot's own hash.

The CPU diagnostic command creates an immutable source snapshot after successful
campaign preflight, executes checks against that snapshot and writes its evidence.
Preflight already creates the exclusive run reservation. Preparation verifies and
adopts that matching reservation; it does not reserve the same identities again.
`prepare` verifies that evidence and all config/preflight/source bindings; it does
not manufacture readiness by attaching a current hash to old results. GPU probes
execute the same snapshot. Any relevant source/config change invalidates affected
gates and requires re-verification. Production configs keep full horizons.

Required coverage:

- All nine definitions/counts, exact uniform actions and deterministic epoch
  batches, strict controls/data identities, incompatible/occupied identity cases.
- Every elastic width, wider-then-narrower behavior, tied/common-bias coverage,
  lazy histories, active zero gradients, independent clipping and scheduler calls.
- Exact actions/batches and numerical uninterrupted/resumed comparison at positions
  before, at and after epoch boundaries; malformed-state and partial-update cases.
- Real C1/C3 sidecars across resume, metric/summary references, strict terminal
  acceptance and missing/inconsistent sidecar rejection.
- Standalone barrier and duplicate-writer/submission failures, attempt costs,
  24/28 reporting fixtures, and schemas 1–3 compatibility.

GPU verification has two explicitly labeled parts. Short real-shape/batch/bf16
checks use inherited controls, all four dense sizes and all five elastic arms,
and retain full assigned scheduler horizons. Controlled small-epoch fixtures
exercise boundary resume/failure through the real trainer/sampler and compare
with uninterrupted numerical state. Record their synthetic membership/budget
overrides; they are not evidence of full-budget production or audited real-corpus
completion. Separate semantic probes ensure all widths are exercised without
forcing production sampling balance. Diagnostic outputs/identities cannot satisfy
production terminal/barrier checks. A skipped CUDA case is not a passing GPU gate.

## Artifact root and stage barrier

Proposed future root:
`/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1`.
Require fresh availability when execution preparation begins. Keep `source/`,
`campaign/` (manifest/configs), `diagnostics/`, `launchers/`, `logs/`, `runs/` and
`reports/` beneath it. Planning does not create or reserve this external root.

Stage 1 admits only four fresh standalones after CPU/GPU gates pass. For each,
invoke the shared strict terminal reader: run/contract identity, counts, full
one-epoch budget, checkpoint/ordinary-evaluation hashes, data/target identities,
complete traces and applicable resources must validate. Slurm COMPLETED and file
existence alone are insufficient.

Publish `launchers/standalone-barrier.json` only after all four pass. Bind its
content hash to the campaign/preflight/source identities, four run contracts,
terminal checkpoint/sidecar hashes and supporting source records. Queue restart
and each elastic worker entry revalidate the same terminal inputs. Tampered,
missing, incomplete or diagnostic terminals block elastic admission. Do not
submit full-budget elastic jobs before the barrier. A subsequently invalidated
barrier blocks new starts and records an error for already-running work; never
silently certify completion from a cached boolean.

## Admission, reconciliation and cost accounting

All GPU work, diagnostics included, uses sbatch, one GPU/process, exclusions
`gpu-[05,50,51]`, and live user-wide limits bounded by two running/four submitted.
Query enforced association/QoS limits before admission and honor stricter positive
limits. Count unrelated user jobs; preserve their jobs and helper processes.
If shared scheduler enforcement cannot guarantee the running ceiling, defer new
admission conservatively. No shell-held login-node GPU workload.

Use queue and per-run writer locks, durable unique submission intents written
before sbatch, and monotonic per-run launch-attempt IDs. On restart reconcile
intents with squeue, sacct and worker records. Delayed visibility or uncertain
submission never justifies a duplicate. Worker verifies gate, source/config and
stage/barrier identity before invoking the existing trainer.

For interrupted runs, distinguish fresh unoccupied output, valid own-checkpoint
resume, completion-only recovery, and blocked invalid/ambiguous state. Never start
fresh in an occupied identity. Failed attempts preserve diagnostics and costs;
successful reconciliation can admit the next own-run continuation attempt within
the same authorized campaign. Unresolved failures stop affected admission.

Link each `ResourceAttemptLedger` process UUID to run/launch-attempt/job ID.
Aggregate each unique attempt's latest measured elapsed time once, take maximum
allocated/reserved peaks, and include failed/replayed work. Scheduler allocation
time and process time overlap; retain separately, never add as disjoint costs.
Hard-kill missing observations remain incomplete/unknown rather than zero.
ETA/progress uses each run's assigned horizon, including 87,132-step standalones.

Finalization is restart-safe: validate any existing frozen/report manifests and
hashes before reuse. Track production, new-report and combined-report status
independently; a report directory alone is not success. Save job/attempt evidence,
outstanding reasons and actual verification/runbook updates.
