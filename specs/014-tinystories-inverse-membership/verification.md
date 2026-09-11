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
