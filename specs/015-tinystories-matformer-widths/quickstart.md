# Quickstart: Feature 015

## Final disposition — 2026-09-22

The user requested finalization as-is and no further experiments. This feature
is **closed with partial experimental results**: implementation and readiness
checks are complete; four fresh standalones and S1/S2 passed full-budget terminal
validation. C1/C2 were cancelled partway through and C3 before starting. The
background checker is stopped. Earlier execution authorization is superseded;
do not submit, resume, or restart monitoring for this campaign.

The retained report has **12 new endpoints plus four historical standalone
endpoints (16 total)**. T037, T045 and T052 are cancelled, not successfully
completed. The original nine-run and 24/28-endpoint acceptance criteria remain
unmet; they are retained below as the original protocol, not future work.
T056/T057 are completed against this explicitly reduced closeout scope. See
[the runbook](../../docs/tinystories-matformer-widths-experiment.md) and
[verification.md](verification.md) for results, costs and artifact provenance.

The commands below document the implemented interfaces and historical execution
procedure. Queue, worker, diagnostic and monitor commands are not instructions
to restart this closed campaign. Saved reports remain available for inspection.

## 1. Implement and verify the scientific changes

Follow [plan.md](plan.md) and [contracts](contracts/campaign-and-topology.md).
Use the pinned interpreter from the repository root. Once implemented, run the
focused CPU acceptance suites:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest \
  tests/test_matformer_widths_campaign.py \
  tests/test_matformer_widths_reporting.py \
  tests/test_matformer_widths_queue.py -q -rs --tb=short
```

Run relevant regression suites for the touched boundaries:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest \
  tests/test_optimizer_ownership.py tests/test_optimizer_ownership_resume.py \
  tests/test_optimizer_ownership_campaign.py tests/test_optimizer_ownership_reporting.py \
  tests/test_optimizer_ownership_corrections.py \
  tests/test_inverse_membership_sampling.py tests/test_inverse_membership_reporting.py \
  tests/test_inverse_membership_queue.py tests/test_config.py \
  tests/test_per_granularity_optimizer.py tests/test_per_granularity_optimizer_resume.py \
  tests/test_metrics_compact_accounting.py tests/test_metrics_history_performance.py \
  tests/test_global_sampling_windows.py tests/test_reproducibility.py \
  tests/test_model_size.py tests/test_packed_corpus.py \
  tests/test_train_cli.py tests/test_training_smoke.py -q -rs --tb=short
```

Record exact results and source hashes. GPU-only skips remain pending GPU checks.
Successful CPU checks must include real new model counts, all-width updates,
compact g125 accounting, clipping sidecars, restore/failure, barrier and 24/28
fixture acceptance. Do not weaken existing equal-quarter negative tests to make
the new layout pass; use its explicit campaign path.

## 2. Prepare an authorized fresh campaign

The proposed root must be unused by any prior campaign. These paths are inherited
input candidates to audit, not newly validated data. The historical reference is
read-only. Set task-specific variables:

```bash
MW_PY=/home/ivo.navarrete/.conda/envs/elasticnn/bin/python
MW_ROOT=/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1
MW_CORPUS=/nfs-stor/ivo.navarrete/matformer-corpora/tinystories-instruct-packed-full-v1
MW_TOKENIZER=/nfs-stor/ivo.navarrete/matformer-tokenizers/tinystories-instruct-sentencepiece-bpe-2k-v1
MW_REFERENCE=/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/reports/complete-20260910T181951Z/frozen/frozen_manifest.json

"$MW_PY" scripts/analyze_tinystories_optimizer_ownership.py preflight \
  --campaign configs/controlled_exps/tinystories_instruct_matformer_widths.yaml \
  --prepared-corpus-dir "$MW_CORPUS" --tokenizer-dir "$MW_TOKENIZER" \
  --output-dir "$MW_ROOT/campaign" --run-output-root "$MW_ROOT/runs"

"$MW_PY" scripts/preflight_tinystories_matformer_widths.py \
  --mode cpu --campaign-root "$MW_ROOT"

"$MW_PY" "$MW_ROOT/source/scripts/run_tinystories_matformer_widths.py" prepare \
  --campaign-root "$MW_ROOT" \
  --cpu-evidence "$MW_ROOT/diagnostics/cpu-gate.json" \
  --reference-manifest "$MW_REFERENCE"
```

Preflight verifies nine definitions, dimensions/counts, pinned corpus/tokenizer and
role hashes, designated membership/excluded tail, budgets and full deterministic
traces. The CPU runner snapshots source and executes acceptance checks against
that snapshot through a separate writable diagnostic workspace. Preparation
verifies existing evidence rather than replacing its hashes. All three commands are CPU preparation; none submit training.

Inspect `campaign/campaign_manifest.json`, `campaign/preflight.json`, nine configs,
the source snapshot manifest and `diagnostics/cpu-gate.json`. Fresh runs cannot
reuse trained historical/diagnostic state. A missing old reference affects only
the later combined report, not the new campaign's independent validity.

## 3. Run authorized GPU diagnostics through Slurm

Check live user-wide limits before this submission. The launcher/preflight
contract requires the same check for automated submissions; this manual example
must also be admitted within two running/four submitted or stricter live limits.
Use an allowed partition/QoS confirmed at execution time; the inherited candidates
are cscc-gpu-p/cscc-gpu-qos. After preparation creates diagnostics/log directories:

```bash
sbatch --partition=cscc-gpu-p --qos=cscc-gpu-qos \
  --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:30:00 \
  --exclude='gpu-[05,50,51,54]' \
  --chdir="$MW_ROOT/source" \
  --output="$MW_ROOT/logs/diagnostic-%j.out" \
  --error="$MW_ROOT/logs/diagnostic-%j.err" \
  --wrap="$MW_PY scripts/preflight_tinystories_matformer_widths.py --mode gpu --campaign-root $MW_ROOT"
```

Retain the submission command and job ID. Resource requests are initial diagnostic
requests to validate during implementation, not promised runtime requirements.
The runner stops each of the nine real-shape batch-64/context-128/bf16 probes
at four updates and resumes to eight while retaining full scheduler horizons.
It also runs
explicit small-epoch uninterrupted/resumed probes, semantic coverage of every
elastic width, clipping sidecars, invalid restores and partial failures. Distinct
probe identities and overrides stay under diagnostics. Readiness requires a
successful job and valid `diagnostics/gpu-gate.json` bound to the CPU snapshot,
config set and preflight manifest. A job being submitted is not a passed gate.

## 4. Execute the staged campaign after gates pass

From the verified snapshot, use the queue command; `--once` performs one
admission/reconciliation cycle, while omission continues the campaign monitor:

For the existing campaign's accepted CUDA-required revision (2026-09-22), use
`$MW_ROOT/launchers/cuda-required-v1/scripts/run_tinystories_matformer_widths.py`
in place of the original snapshot launcher below. Its separate CPU/GPU gates
must pass. The active checker handles this queue; do not launch a competing
queue while it owns `launchers/background-checker.lock`. Actual GPU evidence
is in `launchers/gpu-verification.json`, and the current checker log is
`logs/background-checker-cuda-required.log`. The original launcher is retained
for historical evidence and lacks the new gpu-54 exclusion/CUDA requirement.

```bash
"$MW_PY" "$MW_ROOT/source/scripts/run_tinystories_matformer_widths.py" queue \
  --campaign-root "$MW_ROOT" --once
```

Only standalones are initially eligible. Elastic admission waits for strict
validation of all four full-budget standalone terminals and publication of
`launchers/standalone-barrier.json`. Worker entry revalidates those inputs.
Monitor `launchers/status.json`, submission intents, attempt/resource ledgers and
logs. Queue restarts reconcile existing jobs and run locks; they never blindly
resubmit. Interrupted work resumes its own checkpoint, and terminal-sidecar
recovery takes no additional update. Invalid/ambiguous state is reported for
reconciliation without a fresh restart in the occupied identity.

## 5. Freeze and report saved evidence

The operational `report` mode may perform this sequence restart-safely. The
explicit analyzer interfaces are:

```bash
"$MW_PY" "$MW_ROOT/source/scripts/analyze_tinystories_optimizer_ownership.py" freeze \
  --campaign-manifest "$MW_ROOT/campaign/campaign_manifest.json" \
  --run-root "$MW_ROOT/runs" --output-dir "$MW_ROOT/reports/frozen"

"$MW_PY" "$MW_ROOT/source/scripts/analyze_tinystories_optimizer_ownership.py" report \
  --manifest "$MW_ROOT/reports/frozen/frozen_manifest.json" \
  --output-dir "$MW_ROOT/reports/new"

"$MW_PY" "$MW_ROOT/source/scripts/analyze_tinystories_optimizer_ownership.py" report-matformer-widths \
  --manifest "$MW_ROOT/reports/frozen/frozen_manifest.json" \
  --reference-manifest "$MW_REFERENCE" \
  --output-dir "$MW_ROOT/reports/combined"
```

Run these against the verified source snapshot. Expected results are 24 new rows,
28 combined rows, matching CSV/JSON, nine per-run diagnostics and combined
`loss_vs_parameters.{png,pdf}` / `perplexity_vs_parameters.{png,pdf}`. Each combined
figure has five four-point curves and eight disconnected standalone markers;
overlapping old/new points remain visible at exact coordinates. Historical g750
stays dimension 192. Invalid or missing present inputs cannot yield a complete
comparison. When history is unavailable, preserve the new report and record
combined completion as outstanding.

Finally update the future tasks, verification record and runbook from saved
results. Distinguish diagnostics, nine-run production completion, 24-point report
completion and 28-point combined completion. Interpret seed-42 observations
descriptively, including exposure/clipping/resources and fresh versus historical
baseline context; never infer equal compute from equal assigned tokens.
