# Quickstart: Linear S1/S2 CaLR

Phase 4 implements the four-arm runtime, snapshot/CPU checks, CUDA diagnostic worker and gated launcher. T029–T031 execution is authorized; external preparation and CUDA diagnostics are recorded in verification.md. Reporting fixtures remain pending and block production. CPU tests do not establish GPU readiness.

## 1. Generate tasks and implement

Run `/speckit-tasks` using this plan, then authorize implementation. Implement all four arms together, preserve legacy signatures, and complete reporting fixtures before production. Use `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python`; no dependency additions are planned.

Focused CPU verification after implementation:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest   tests/test_linear_calr_campaign.py tests/test_linear_calr_schedule.py   tests/test_linear_calr_resume.py tests/test_linear_calr_reporting.py   tests/test_linear_calr_queue.py -q -rs --tb=short
```

Also run relevant historical config, S1/S2/C4 optimizer/restore, metrics, reporting and queue regressions. Record source/config hashes, outcomes and limitations. Old passing results and GPU skips do not establish new readiness.

## 2. Snapshot and CPU preparation

After external preparation authorization, set `TASK_CORPUS_DIR` and `TASK_TOKENIZER_DIR` to the inherited audited artifact directories. Implemented commands are:

```bash
TASK_ROOT=/nfs-stor/ivo.navarrete/results/elasticnn/tinystories-linear-s1-s2-calr-v1
PYTHON_BIN=/home/ivo.navarrete/.conda/envs/elasticnn/bin/python
"$PYTHON_BIN" scripts/preflight_tinystories_linear_calr.py snapshot --campaign-root "$TASK_ROOT" \
  --prepared-corpus-dir "$TASK_CORPUS_DIR" --tokenizer-dir "$TASK_TOKENIZER_DIR" \
  --reference-root /nfs-stor/ivo.navarrete/results/elasticnn
"$PYTHON_BIN" "$TASK_ROOT/source/scripts/preflight_tinystories_linear_calr.py" cpu --campaign-root "$TASK_ROOT"
"$PYTHON_BIN" "$TASK_ROOT/source/scripts/run_tinystories_linear_calr.py" prepare   --campaign-root "$TASK_ROOT" --cpu-evidence "$TASK_ROOT/diagnostics/cpu-gate.json"
```

Atomic reservation rejects occupied identities. Preparation verifies model counts, exact controls/differences, full-horizon analytic schedules, fresh initialization/streams and reference discovery. Saved references are always read-only. Actual checkpoint/evaluation proof is needed before report acceptance; absence of newer artifacts on .004 runs requires the legacy adapter, not reconstructed evidence.

## 3. Later authorized GPU readiness and production

Explicit execution authorization is recorded separately for diagnostic sbatch jobs and production. The implemented submission uses existing operational helpers with partition cscc-gpu-p, QoS cscc-gpu-qos and exclusions gpu-[05,50,51,54]. Use the verified accounting client environment for submission and reconciliation:

```bash
export SLURM_CONF=/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1/launchers/slurm-client.conf
```

After diagnostic authorization is durably recorded, submit from the tested snapshot with `preflight_tinystories_linear_calr.py submit-gpu --campaign-root ROOT`. The GPU worker mode is `preflight_tinystories_linear_calr.py gpu --campaign-root ROOT`; run from the immutable snapshot on the allocated GPU. Cover all four arms at batch 64/context 128 with actual BF16 and bind passing evidence to CPU/source/config identities.

After separate production authorization, passing CPU/all-arm GPU gates and Phase 5 reporting-fixture acceptance, the implemented admission command is:

```bash
"$PYTHON_BIN" "$TASK_ROOT/source/scripts/run_tinystories_linear_calr.py" queue   --campaign-root "$TASK_ROOT" --once
```

Queue checks user-wide two-running/four-submitted ceilings and stricter live limits, duplicate/uncertain submissions and own-run continuation. Repeat admission as needed; no manual bypass of gates. Terminal recovery at full budget performs zero additional training updates.

## 4. Report and acceptance

```bash
"$PYTHON_BIN" "$TASK_ROOT/source/scripts/run_tinystories_linear_calr.py" report --campaign-root "$TASK_ROOT"
```

Require four complete new terminals, eight validated references, 36 endpoint rows, 48 required differences and four interactions. Check primary/supplemental seven-label panels, PNG/PDF figure families, recorded per-width validation and actual applied rates, resource/provenance manifests and all-width findings. Missing terminal/provenance/trajectory evidence is incomplete; partial runs, analytic schedules and old report tables cannot replace it. Full delivery is SC-001–008, independent of whether CaLR improves quality.


## Phase 4 gate records

Snapshot inputs live in `diagnostics/inputs.json`; frozen source hashes in `diagnostics/source-manifest.json`. CPU and GPU gate records are `diagnostics/cpu-gate.json` and `diagnostics/gpu-gate.json`, with retained per-attempt records/logs under `diagnostics/`. Changed source, configurations, inputs, audits or recorded logs invalidate their bindings. `snapshot` requires a fresh root; repeat use validates the existing immutable source. `cpu` runs full corpus/control/model/action/data/schedule audits and CPU tests from those frozen bytes; it never submits a job.

A subsequent explicit user instruction is recorded separately in `authorizations/diagnostic.json` or `authorizations/production.json`. Each record contains `purpose`, `authorized: true`, the verbatim `user_instruction`, `recorded_at`, and the current `bindings(ROOT)` dictionary; use the launcher's `sealed()` helper for its `content_hash`. Passing readiness or a CLI flag does not supply authorization. The T029–T031 authorization has supplied both records for the current snapshot.

Phase 4's CPU gate keeps `reporting_fixture_status: pending` until Phase 5 provides its executed reporting-fixture proof. `report` dispatches the Phase 5 `report-linear-calr` operation and records its return code/incomplete status; that analyzer operation remains a Phase 5 task. Consult tasks.md and verification.md for the actual T029–T031 outcomes.
