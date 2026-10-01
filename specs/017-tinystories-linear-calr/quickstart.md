# Quickstart: Linear S1/S2 CaLR

Planning artifacts describe proposed implementation. The new recipe/scripts below do not exist yet. This invocation performs no training, readiness run, output reservation, commit or push.

## 1. Generate tasks and implement

Run `/speckit-tasks` using this plan, then authorize implementation. Implement all four arms together, preserve legacy signatures, and complete reporting fixtures before production. Use `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python`; no dependency additions are planned.

Focused CPU verification after implementation:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest   tests/test_linear_calr_campaign.py tests/test_linear_calr_schedule.py   tests/test_linear_calr_resume.py tests/test_linear_calr_reporting.py   tests/test_linear_calr_queue.py -q -rs --tb=short
```

Also run relevant historical config, S1/S2/C4 optimizer/restore, metrics, reporting and queue regressions. Record source/config hashes, outcomes and limitations. Old passing results and GPU skips do not establish new readiness.

## 2. Snapshot and CPU preparation

After implementation and authorization to prepare the results root, planned commands are:

```bash
TASK_ROOT=/nfs-stor/ivo.navarrete/results/elasticnn/tinystories-linear-s1-s2-calr-v1
PYTHON_BIN=/home/ivo.navarrete/.conda/envs/elasticnn/bin/python
"$PYTHON_BIN" scripts/preflight_tinystories_linear_calr.py snapshot --campaign-root "$TASK_ROOT"
"$PYTHON_BIN" "$TASK_ROOT/source/scripts/preflight_tinystories_linear_calr.py" cpu --campaign-root "$TASK_ROOT"
"$PYTHON_BIN" "$TASK_ROOT/source/scripts/run_tinystories_linear_calr.py" prepare   --campaign-root "$TASK_ROOT" --cpu-evidence "$TASK_ROOT/readiness/cpu.json"
```

Atomic reservation rejects occupied identities. Preparation verifies model counts, exact controls/differences, full-horizon analytic schedules, fresh initialization/streams and reference discovery. Saved references are always read-only. Actual checkpoint/evaluation proof is needed before report acceptance; absence of newer artifacts on .004 runs requires the legacy adapter, not reconstructed evidence.

## 3. Later authorized GPU readiness and production

Only a later explicit execution instruction authorizes diagnostic sbatch jobs and production. Implement the diagnostic submission through existing operational helpers with partition cscc-gpu-p, QoS cscc-gpu-qos and exclusions gpu-[05,50,51,54]. The GPU worker mode is `preflight_tinystories_linear_calr.py gpu --campaign-root ROOT`; run from the immutable snapshot on the allocated GPU. Cover all four arms at batch 64/context 128 with actual BF16 and bind passing evidence to CPU/source/config identities.

After separate production authorization and passing gates, planned admission command:

```bash
"$PYTHON_BIN" "$TASK_ROOT/source/scripts/run_tinystories_linear_calr.py" queue   --campaign-root "$TASK_ROOT" --once
```

Queue checks user-wide two-running/four-submitted ceilings and stricter live limits, duplicate/uncertain submissions and own-run continuation. Repeat admission as needed; no manual bypass of gates. Terminal recovery at full budget performs zero additional training updates.

## 4. Report and acceptance

```bash
"$PYTHON_BIN" "$TASK_ROOT/source/scripts/run_tinystories_linear_calr.py" report --campaign-root "$TASK_ROOT"
```

Require four complete new terminals, eight validated references, 36 endpoint rows, 48 required differences and four interactions. Check primary/supplemental seven-label panels, PNG/PDF figure families, recorded per-width validation and actual applied rates, resource/provenance manifests and all-width findings. Missing terminal/provenance/trajectory evidence is incomplete; partial runs, analytic schedules and old report tables cannot replace it. Full delivery is SC-001–008, independent of whether CaLR improves quality.
