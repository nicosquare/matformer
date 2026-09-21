# Verification evidence — Feature 014

## Setup and design

2026-09-11: Created new branch 014-tinystories-inverse-membership. Initial Git
status contained only the untracked source prompt; preserved it. Existing Python
ignore rules cover generated/cache artifacts. No dependency or ignore change needed.
Proposed campaign root was absent; live squeue was empty at initial inspection.
Specification checklist 16/16 passed; consistency analysis has 100% FR coverage.
Historical runbook contains stale completion statements; saved evidence will govern.

Implementation and runtime checks are recorded below as they complete. No
production or GPU success is claimed by this initial record.

## CPU implementation checks

- New sampling/report acceptance: 62 passed. Covers five-arm counts and rejection,
  categorical boundary mapping, all five real AdamW model paths at each width,
  exact packed-epoch resume, before-mutation corruption rejection, C3 partial
  failure durability, 20/44-point terminal exports and figure/progress structure.
- Feature/global-window/reproducibility suite: 180 passed, two existing dependency
  deprecation warnings, 38.41 seconds.
- Historical compatibility suite: 720 passed, 34 GPU-only skipped, one expected
  failure, two dependency deprecation warnings, 153.35 seconds. Files:
  test_inverse_membership_queue.py, test_optimizer_ownership_campaign.py,
  test_optimizer_ownership_reporting.py, test_optimizer_ownership_corrections.py,
  test_optimizer_ownership.py, test_optimizer_ownership_resume.py, test_config.py,
  test_training_smoke.py. These suite counts overlap; they are not unique totals.
- Queue tests after prepare-command integration: 22 passed, 0.18 seconds.
- `git diff --check` passed. Initial tests failed on missing recipe, output-arm
  eligibility and missing fixed-global accounting, establishing the tested gaps.

## Input and historical evidence

Full preflight passed: five real CPU model paths match exact counts
115264/164416/213568/262720. Corpus/tokenizer audit retains all pinned identities,
89 shards, designated 5,576,448 sequences and fixed excluded tail of 43.
All five seeded action traces match, SHA256
`b08f27dd5aae359e5c8e5a58a6dca0faeeb920f5ecffdb576aeea3f53f0ed99d`.
Realized selections: g250 41581, g500 55948, g750 83767, g1000 167232.
These are deterministic realized draws, not enforced expected frequencies.

Original frozen reference validation passed under the strict schema-1 reader:
nine runs, 24 endpoints, 58 source records. Reference:
`/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/reports/complete-20260910T181951Z/frozen/frozen_manifest.json`.
No historical artifact was modified or holdout evaluated.

Live pre-diagnostic scheduler check confirmed enforced associations/limits/qos,
MaxJobsPU=2 and MaxSubmitJobsPU=4 for cscc-gpu-qos, with no active user jobs.
Admission repeats this check; it is not a permanent queue reservation.

## Final source and diagnostic submission

Final reporting checks: 69 passed (64.84 seconds); final queue checks 22 passed.
Refreshed preflight manifest hash:
`97e677df3ceca4d59e3e82f68f400c749bf8fe6c945b43fd72b22ccbb62712cd`.
Preliminary preflight is preserved under diagnostics/preflight-before-final-reporting.

Snapshot: campaign-root/source (176 files). Full snapshot hash:
`c8e47f25571ad27422adfeab28922b65c1f3306f706f8ec3a3be5225851bd3da`.
CPU command/results and historical reference audit are saved under diagnostics.

Submitted diagnostic job 229045, one GPU, 16G memory, one-hour limit, required
node exclusions, cscc-gpu-p/cscc-gpu-qos. Intent and exact command are recorded in
launchers/diagnostic-submission.json. This is submission evidence, not GPU success.

## GPU gate — passed

Slurm job 229045 completed successfully (ExitCode 0:0), NVIDIA A100-SXM4-40GB,
gpu-03, runtime 71 seconds. All five arms ran 192 updates, restored their own
checkpoint, then reached 256. Finite training/validation losses and parameters,
matching actual seeded action digests, width/owner counts and global clock passed.
All five selected g250/g500/g750/g1000 25/42/69/120 times. C3 owner calls were
A=256, B=231, C=189, D=120, common=256. No holdout evaluation.

Recent update median seconds: S1 .029632; S2 .030201; C1 .032826; C2 .033115;
C3 .036191. These exclude startup, warmup and post-validation intervals and are
short diagnostics, not measured full-production wall times.

Gate: campaign-root/diagnostics/gpu-gate.json, bound to the validated snapshot.
All diagnostic configs/checkpoints/metrics/attempt ledgers are retained in
diagnostics/gpu-runtime-229045. Production queue was started after this success.

## Production execution — active, results pending

Submitted after both gates passed:

| Arm | Slurm job |
| --- | --- |
| S1-IM | 229058 |
| S2-IM | 229059 |
| C1-IM | 229060 |
| C2-IM | 229061 |
| C3-IM | Awaiting a user-wide admission slot |

All four submitted jobs were pending Priority at the latest check. Scheduler
estimates were 2026-09-11 16:31:06, 18:20:46, 20:10:00 and 20:43:24 in cluster
local time, respectively; estimates may change and are not promised starts.
No production completion or terminal result is claimed yet.

The campaign monitor runs detached as PID 2917977, parent PID 1 and its own
session, verified with ps outside the tool sandbox. Its status heartbeat and
four existing submission IDs were reconciled after detaching, without duplicate
submissions. The previous interactive monitor was replaced; no GPU job or
unrelated campaign helper was stopped. Durable process record:
`launchers/monitor-process.json`; progress: `launchers/status.json`; log:
`logs/queue.log`; exact job intents/commands: `launchers/submissions.json`.

The monitor automatically admits C3 when capacity opens, tracks progress,
checkpoints/utilization/costs, freezes completed terminals, produces the validated
20-point new report and attempts the 44-point historical comparison. It records
`launchers/completion.json` only after reporting, or explicitly marks a historical
comparison outstanding. Failed attempts stop admission for checkpoint/resource
reconciliation rather than silently restarting from scratch. T016–T018 remain
open until actual production completion, interpretation and final reconciliation.

## Logging audit and pre-start correction — 2026-09-11

The user requested verification of the historical logging slowdown. Both the
repository and the originally queued source contained commit 3459890's compact
optimizer-attempt accounting. The logging, rollback and checkpoint modules matched
the pinned snapshot byte for byte. All 43 focused compact-accounting/history tests
passed against that snapshot. The five actual GPU diagnostic checkpoints also
contained schema-2 counters and a single last-attempt marker, with no attempt-ID
history.

A CPU benchmark retained the actual diagnostic validation summaries and advanced
the accumulator through the full 348,528-update horizon. It measured accumulator
update, journal state publication and the metrics portion of rollback copying:

| Through update | Samples | Median milliseconds | Serialized state bytes |
| --- | --- | --- | --- |
| 2,256 | 2,000 | 3.621 | 171,367 |
| 100,000 | 2,000 | 3.612 | 171,379 |
| 348,528 | 2,000 | 3.620 | 171,383 |

This demonstrates bounded history-related work, not zero logging overhead. The
benchmark excludes filesystem I/O, model compute and other rollback fields; it
keeps the diagnostic's validation summaries fixed. Scripts and machine-readable
measurements are saved in campaign-root/diagnostics/logging-audit.

The audit also found a separate missing-artifact bug: clipping sidecars and metric
paths were gated on literal C1/C3 arm IDs, so C1-IM/C3-IM did not write them even
though the final reader requires them. Both gates now use concat representation
and optimizer scope, matching the run summary and audit contract. Five real-model
tests cover sidecars, active quarters, exact committed steps and strict saved-trace
auditing across checkpoint resume. The GPU diagnostic now requires 256 clipping
observations for both relevant arms and verifies compact metrics checkpoint state.
Final regression: 199 passed, two existing dependency warnings, 59.25 seconds.

All production jobs were PENDING with zero runtime when the correction began.
Stopped the old monitor and held the jobs before replacing any runtime artifact.
Archived the previous snapshot, campaign, reservation, gates and launch records
under diagnostics/before-logging-audit, then reran preflight and prepare. Updated
snapshot hash: `e355e8c4174aa3dcb55764cb5adedf7831caf741da146939685d4495752ca335`.
The snapshot includes the tested working changes; provenance records the working
tree diff and source hashes relative to commit a77d3c2.

Replacement GPU diagnostic: job 229101, ten-minute limit, same required resources
and exclusions. To respect four submitted jobs, cancelled only the never-started
C2 job 229061; its pre/post scheduler records are saved in logging-audit/deferred-c2.json.
S1/S2/C1 retain IDs 229058/229059/229060 and remain held until the new GPU gate
passes. C2 and C3 will then be admitted within the same user-wide limits.
Detached monitor PID 3120525 waits for successful diagnostic completion, checks
both gates against the new source hash, records before/after timing, releases the
held jobs and resumes the original five-arm queue/report workflow. Its process
and heartbeat were verified. Diagnostic failure leaves production held and records
logging-audit/release-error.json. At this audit entry the diagnostic was still
pending; fresh GPU timing and production throughput are not yet claimed.

## User-requested fresh resubmission — 2026-09-11

The user subsequently requested cancellation of all Slurm submissions and a fresh
launch. Stopped monitor 3120525 and verified cancellation of every then-current
job: 229058, 229059, 229060 and 229101. All were still pending with zero runtime;
the production output directory did not exist. Prior launch records and scheduler
cancellation evidence are preserved in
diagnostics/fresh-resubmission-20260911T104825Z. The earlier cancelled C2 job
229061 remains documented in the logging audit.

Reused the unchanged, hash-verified corrected source and passed CPU evidence.
Submitted fresh diagnostic 229115 with the required exclusions and ten-minute
limit. New detached monitor 3222762, launchers/fresh_campaign_monitor.py, waits for
this exact diagnostic's successful scheduler completion and matching GPU gate,
then invokes the original queue to submit all five production runs from step zero.
There are no retained or held production submissions in this fresh launch. The
two-running/four-submitted limits still apply; production submission follows GPU
validation. Current evidence is in launchers/diagnostic-submission.json,
launchers/monitor-process.json and launchers/status.json; monitor output is in
logs/fresh-queue.log. Failure is recorded in launchers/fresh-monitor-error.json.

## Diagnostic config-default correction — 2026-09-11

Job 229115 ran on gpu-03 from 15:22:41 to 15:23:26 cluster local time and failed
with ExitCode 1:0. The new diagnostic clipping predicate incorrectly indexed
optimizer_state_scope in the raw executable config, where that resolved setting
is absent. It now reads the resolved config. S1-IM, S2-IM and C1-IM had each
reached checkpoint step 256 before the diagnostic check failed; C2/C3 were not
reached. No production jobs were submitted.

CPU inspection verified all five resolved configurations, all three saved
checkpoints and traces, compact metrics state, and all 256 C1 clipping records
including active-quarter identity and finite norms. Only the diagnostic script
changed relative to the previous pinned snapshot; production source is identical.
Previous source and launch evidence were archived under
diagnostics/diagnostic-config-fix-20260911T112757Z. Updated snapshot hash:
`80bccba4340f063b9b00dcb7522e8e54ecfdf0b762ef38bfec9f93bdf2d0b3ff`.

Submitted replacement diagnostic 229177 and restarted the detached monitor as
PID 4068500. Fresh production remains gated on successful completion of this
exact diagnostic and matching source hashes. The monitor now publishes a failed
status as well as an error artifact if it exits on an error, avoiding a stale
RUNNING diagnostic status. GPU success for 229177 is not yet claimed here.

### Replacement diagnostic passed and production admitted

Job 229177 subsequently COMPLETED with ExitCode 0:0, 67 seconds, gpu-03,
NVIDIA A100-SXM4-40GB (15:28:12–15:29:19 cluster local time). All five arms
passed the 192-to-256 interruption/resume diagnostic and compact accounting checks.
C1-IM and C3-IM each saved and validated exactly 256 clipping records.
Recent update median seconds: S1 .029928; S2 .030161; C1 .033497; C2 .032990;
C3 .035415. Compared with the original same-device diagnostic, the largest
increase is approximately 2.0% for C1; C3 is approximately 2.1% faster. These
short-run measurements show no substantial slowdown and are not a guarantee
against future filesystem or scheduler variability.

The restarted monitor verified the completed job and matching CPU/GPU source
hashes, then submitted fresh production: S1-IM 229183, S2-IM 229184, C1-IM
229185, C2-IM 229186. All four were pending Priority at the admission check.
C3-IM awaits the four-submitted limit and is admitted automatically when a slot
opens. Durable success: launchers/fresh-gate-passed.json and diagnostics/gpu-gate.json;
production IDs: launchers/submissions.json. No production terminal results yet.
