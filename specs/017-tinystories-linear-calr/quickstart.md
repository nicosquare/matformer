# Quickstart: Linear S1/S2 CaLR

The four-arm runtime, immutable preparation, CPU/CUDA diagnostics, gated Slurm launcher and reporting CLI are implemented. Production and the real comparison are complete; see [experiment-report.md](experiment-report.md) and [verification.md](verification.md). The commands below describe the implemented lifecycle, not a request to rerun this completed campaign. Continuous monitoring remains stopped.

## CPU verification

Use the installed interpreter; no new dependencies are required:

```bash
PYTHON_BIN=/home/ivo.navarrete/.conda/envs/elasticnn/bin/python
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -m pytest \
  tests/test_linear_calr_campaign.py tests/test_linear_calr_schedule.py \
  tests/test_linear_calr_resume.py tests/test_linear_calr_reporting.py \
  tests/test_linear_calr_queue.py -q -rs --tb=short -p no:cacheprovider
```

The exact historical regression command and final source/config hashes are recorded in verification.md. CPU fixtures use temporary roots and mocked scheduler calls. CPU passes and CUDA skips establish CPU correctness only; all-arm, real-shape CUDA BF16 readiness is a separate bound gate.

## Snapshot, audit and prepare

External preparation requires its own authorization. Supply the inherited audited packed corpus/tokenizer paths; choose a fresh campaign root. The completed canonical root below is occupied and must not be reused for a new reservation.

```bash
TASK_ROOT=/nfs-stor/ivo.navarrete/results/elasticnn/tinystories-linear-s1-s2-calr-v1
PYTHON_BIN=/home/ivo.navarrete/.conda/envs/elasticnn/bin/python
# Set TASK_CORPUS_DIR and TASK_TOKENIZER_DIR to the audited artifact directories.
"$PYTHON_BIN" scripts/preflight_tinystories_linear_calr.py snapshot \
  --campaign-root "$TASK_ROOT" --prepared-corpus-dir "$TASK_CORPUS_DIR" \
  --tokenizer-dir "$TASK_TOKENIZER_DIR" \
  --reference-root /nfs-stor/ivo.navarrete/results/elasticnn
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" \
  "$TASK_ROOT/source/scripts/preflight_tinystories_linear_calr.py" cpu \
  --campaign-root "$TASK_ROOT"
"$PYTHON_BIN" "$TASK_ROOT/source/scripts/run_tinystories_linear_calr.py" prepare \
  --campaign-root "$TASK_ROOT" --cpu-evidence "$TASK_ROOT/diagnostics/cpu-gate.json"
```

`snapshot` creates immutable source/config bytes and `diagnostics/inputs.json` / `source-manifest.json`; repeated snapshot use validates existing identities. `cpu` performs corpus/control/model/count, fresh-weight, complete action/data-stream and analytic schedule audits and executes the frozen CPU suite. `prepare` requires the passing bound CPU record and reserves exactly four resolved runs. None of these operations submits GPU work. Occupied/conflicting identities and modified snapshot/config/input/evidence bindings fail. Saved references remain read-only; discovery alone does not prove terminal provenance.

## Diagnostic and production admission

Diagnostic and production execution require separate durable explicit user authorizations. Records live in `authorizations/diagnostic.json` and `authorizations/production.json`, with `purpose`, `authorized: true`, verbatim `user_instruction`, `recorded_at`, current launcher `bindings(TASK_ROOT)` and a `sealed()` content hash. A passing gate or CLI flag does not grant authorization. This completed campaign already retains both records; source changes require new matching evidence, without relabeling old gates.

Use the verified scheduler client configuration:

```bash
export SLURM_CONF=/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1/launchers/slurm-client.conf
# Only after diagnostic authorization:
"$PYTHON_BIN" "$TASK_ROOT/source/scripts/preflight_tinystories_linear_calr.py" \
  submit-gpu --campaign-root "$TASK_ROOT"
# Only after production authorization, passing CPU/reporting fixtures and all-arm GPU gates:
"$PYTHON_BIN" "$TASK_ROOT/source/scripts/run_tinystories_linear_calr.py" \
  queue --campaign-root "$TASK_ROOT" --once
```

`submit-gpu` admits/reconciles the diagnostic job; its allocated worker runs `preflight_tinystories_linear_calr.py gpu --campaign-root "$TASK_ROOT"`. Diagnostics cover every arm at d64/l4/h4, batch 64/context 128, actual BF16, all widths, nonzero moments, boundary updates/own continuation and resource measurements. CPU fallback is rejected. Records/logs, including failures, remain under `diagnostics/`; CPU/GPU gates must bind identical tested source/config identities and successful worker/Slurm evidence.

`queue` without `--once` repeats admission/reconciliation until completion. It uses cscc-gpu-p/cscc-gpu-qos, one GPU, no requeue, exclusions gpu-[05,50,51,54], user-wide ceilings of two running/four submitted GPU jobs and stricter live limits. Durable locks/intents prevent duplicate or uncertain resubmission. It launches the frozen worker as:

```bash
"$PYTHON_BIN" "$TASK_ROOT/source/scripts/run_tinystories_linear_calr.py" worker \
  --campaign-root "$TASK_ROOT" --arm S1-linear-poly --attempt-id 1
```

That entry requires its Slurm allocation and revalidates gates; it is not a login-node training command. Legal arms are S1-linear-poly, S1-linear-CaLR, S2-linear-poly and S2-linear-CaLR. Continuation uses only the arm's own durable state. Terminal recovery at the full budget adds zero training updates. Job disappearance alone never proves completion.

## Reporting and incomplete evidence

The launcher accepts `--report-source` for a separate immutable reporting snapshot, preserving production bindings. The completed campaign uses its corrected final reporting snapshot:

```bash
TASK_REPORT_SOURCE="$TASK_ROOT/reports/reporting-source-final-20261002"
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" \
  "$TASK_REPORT_SOURCE/scripts/run_tinystories_linear_calr.py" report \
  --campaign-root "$TASK_ROOT" --report-source "$TASK_REPORT_SOURCE"
```

Direct reporting is also implemented; supply a fresh output directory:

```bash
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" \
  "$TASK_REPORT_SOURCE/scripts/analyze_tinystories_optimizer_ownership.py" report-linear-calr \
  --campaign-manifest "$TASK_ROOT/campaign/campaign_manifest.json" \
  --run-root "$TASK_ROOT/runs" --reference-root /nfs-stor/ivo.navarrete/results/elasticnn \
  --output-dir "$TASK_ROOT/reports/comparison-new"
```

Complete publication requires four full-budget new terminals and eight validated historical runs: 36 endpoints, 48 required pairs, four interactions, 16 separately labeled supplemental .004 pairs and 16 PNG/PDF artifacts. The plot-source manifest binds tables/figures to source/config/checkpoint/evaluation/metric hashes. Primary .008 and supplemental .004 panels stay separate. Standalone progress contains terminal points only at update 87132. Applied LR plots sample recorded first 65, every 128th global commit and last commit per width after validating the complete trace; analytic schedules never substitute for observed rates.

Missing terminal/reference/provenance/trajectory evidence returns nonzero (direct incomplete publication: exit 1), retains admitted evidence and exposes independent new-terminal/reference/trajectory/comparison states. Partial runs, summaries and older aggregate tables cannot replace proof. Full acceptance covers SC-001–008 regardless of whether CaLR improves loss.
