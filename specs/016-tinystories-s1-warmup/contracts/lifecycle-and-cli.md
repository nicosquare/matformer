# Contract: Lifecycle, Evidence and CLI

Phases 4–5 implement preparation, diagnostics, prepare/queue/worker and saved-artifact reporting. Existing unrelated command behavior remains unchanged.

## Preparation and identities

Proposed result root: `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-s1-warmup-v1`.

```text
<root>/
├── campaign/                 # manifest, preflight, configs, control differences, schedules/traces
├── references/               # new read-only selection records pointing to original files
├── source/                   # immutable tested executable snapshot
├── diagnostics/              # source manifest, CPU/GPU gates, probe artifacts
├── launchers/                # prepared plan, intents, attempts, worker/device/status records
├── logs/
├── runs/S1-linear-w256/
├── runs/S1-geometric-w256/
└── reports/
    ├── new/                  # eight new terminal endpoints and frozen manifest
    └── comparison/           # 24 endpoints, deltas, early data, figures and findings
```

Reject occupied/newly conflicting roots or run identities before publication. Repeated preparation can verify an identical already-prepared identity, but cannot adopt unrelated files. Preflight publishes atomically only after controls/data/models/traces and required selected-reference checks pass. Freeze source and exact executable config hashes before readiness tests. Reference files remain read-only; selection copies contain provenance records, not rewritten source results.

## CLI surface

| Entry point | Command and required arguments | Result |
| --- | --- | --- |
| `scripts/analyze_tinystories_optimizer_ownership.py` | `preflight --campaign RECIPE --prepared-corpus-dir CORPUS --tokenizer-dir TOKENIZER --output-dir ROOT/campaign --run-output-root ROOT/runs --linear-reference-root LINEAR --geometric-reference-root GEOMETRIC` | Strict schema 5 expansion/audit, expected schedules/traces and new reference selections. Reference flags required only for schema 5. |
| `scripts/preflight_tinystories_s1_warmup.py` | `cpu --campaign-root ROOT` | Freeze source if absent; test exact snapshot/configs and publish passed/failed CPU gate. |
| Same | `submit-gpu --campaign-root ROOT` | Explicitly authorized diagnostic submission through shared live-limit/intent policy. |
| Same | `gpu --campaign-root ROOT` | Internal worker under authorized sbatch allocation only; both real-shape bf16 grids, matching CPU evidence. |
| `scripts/run_tinystories_s1_warmup.py` | `prepare --campaign-root ROOT --cpu-evidence PATH` | Validate reservation, source/config/reference bindings and CPU gate; publish launch plan. No submission. |
| Same | `queue --campaign-root ROOT [--once]` | Reconcile attempts, verify CPU/GPU gates, inspect live limits, submit only eligible declared runs. |
| Same | `worker --campaign-root ROOT --arm ARM --attempt-id N` | Internal sbatch-only unique writer; validate intent/gates/own continuation; invoke CUDA-required trainer. |
| Analyzer | `freeze --campaign-manifest ROOT/campaign/campaign_manifest.json --run-root ROOT/runs --output-dir DEST` | Strict two-run terminals, eight new endpoints; existing `--allow-partial` never means complete. |
| Analyzer | `report-s1-warmup --manifest FROZEN --linear-reference-root LINEAR --geometric-reference-root GEOMETRIC --output-dir DEST [--early-end-step 1024]` | Revalidate selection/device/scalars; publish complete comparison or explicit incomplete evidence. End step must be >=1024. |
| Launcher | `report --campaign-root ROOT` | Recover new freeze/report and comparison outputs using prepared mappings, without training or submitting. |

Commands return nonzero on invalid/missing required inputs and explain the specific failure. A complete-report request failing required references/early evidence cannot return a complete status. Valid new artifacts and structured diagnostics remain available. Queue capacity causes a recorded wait (or pending result for `--once`), not a false failure or unauthorized alternate execution.

## Readiness and admission

CPU/GPU gates include source-file map/hash (including actual launchers/trainer), recipe/config-set/manifest/contract hashes, reference/control/trace audit bindings, test command/log hashes and pass/fail/skip results. GPU gate additionally binds the passed CPU gate, both grid names, real d64/l4/h4/batch64/context128 probes, actual CUDA bf16, allocation/job/device and executed all-width/continuation checks. Skipped GPU cases do not pass. Changed executable source/config/reference bindings stale the gates and require fresh evidence rather than relabeling old results.

Queue and worker revalidate gates. Production begins only with subsequent user authorization, correct readiness, unique reservation and usable CUDA. No standalone training barrier is needed: there are no new standalones. Required reference/control audits still precede admission. Subsequent missing historical inputs prevent comparison completion and must not erase already-valid new terminals.

All GPU work uses sbatch, one training process and one GPU per run, excluding `gpu-[05,50,51,54]`. Enforce user-wide at most two running/four submitted jobs or stricter live Slurm policy; count unrelated jobs and preserve them. The launcher must ensure its submissions cannot exceed the running ceiling, through verified Slurm enforcement or conservative admission with active reservations. Do not rely solely on a check that current running jobs are already greater than the ceiling. No automatic CPU fallback or extra arms.

Persist a unique submission intent before sbatch. Reconcile active queue and accounting by durable identity; ambiguous/missing scheduler response remains uncertain until resolved. Do not infer permission to resubmit from an empty queue. Lock queue and per-run writers. Each process has a monotonic attempt ID/UUID and validated own continuation hash. An occupied run without a valid own checkpoint is blocked for reconciliation, not silently restarted.

## Completion, resources and recovery

New terminal completion requires full-budget checkpoint/evaluation/trace validation, actual CUDA bf16 evidence, matching source/config/job/worker identities, successful worker exit and successful scheduler accounting. A terminal sidecar alone is insufficient. If Slurm accounting is delayed, retain pending/uncertain status until evidence arrives.

Use existing `ResourceAttemptLedger` accounting for attempted work, elapsed time, peaks and interrupted/replayed work; disclose missing observations. Restore committed progress from the durable bundle independently of consumed attempt resources. Failed update state cannot overwrite that bundle. Terminal-only recovery may regenerate missing evaluation/report outputs without increasing 348528 committed steps.

Track planning, implementation checks, CPU gate, GPU readiness, each new terminal, new-only report, endpoint comparison and early/report completion separately. A source snapshot or fixtures never establish production completion. Planning creates no production root or job. Later implementation and execution follow the user's subsequent instructions; prior campaigns' launch approvals do not carry over.
