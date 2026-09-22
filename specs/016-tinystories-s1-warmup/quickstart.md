# Quickstart: S1 Warmup Extension

Phases 3–4 implement schema-5 preflight, runtime/continuation checks, snapshot-bound diagnostics and prepare/queue/worker, verified against fixtures (see [verification.md](verification.md)). Reporting remains phase-5 work; production is blocked until its fixture gate passes. Real corpus/terminal audits, GPU diagnostics and production have not been performed. Subsequent user instructions authorize those stages; earlier campaigns' approvals do not apply.

## 1. Implement and verify locally

Generate tasks from [plan.md](plan.md), then implement the fixed protocol and focused checks. Use the pinned interpreter from the repository root. Planned focused command:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_s1_warmup_campaign.py tests/test_s1_warmup_reporting.py tests/test_s1_warmup_queue.py -q -rs --tb=short
```

Also run relevant existing ownership/matformer campaign, resume, config, compact accounting, reporting and queue regressions. Capture schemas 1–4 signatures before changes, since the existing legacy signature fixture covers only 1–3. Verify old warmup 256 rejection, new warmup 64 rejection, both complete LR arrays and actual LR indexing, own-counterpart traces, all widths, failure durability, terminal-only recovery and strict report rejection. Record commands/source/config hashes and outcomes in later `verification.md` and runbook. GPU skips are not readiness.

## 2. Audit real inputs and references

Proposed commands below use the real paths already saved in reference configs; revalidate them. Shell variables have task-specific names. The preflight checks/creates only the fresh result root after validation and never modifies references.

```bash
WARMUP_ROOT=/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-s1-warmup-v1
WARMUP_LINEAR_REF=/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1
WARMUP_GEOMETRIC_REF=/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1
WARMUP_PYTHON=/home/ivo.navarrete/.conda/envs/elasticnn/bin/python

"$WARMUP_PYTHON" scripts/analyze_tinystories_optimizer_ownership.py preflight \
  --campaign configs/controlled_exps/tinystories_instruct_s1_warmup.yaml \
  --prepared-corpus-dir /nfs-stor/ivo.navarrete/matformer-corpora/tinystories-instruct-packed-full-v1 \
  --tokenizer-dir /nfs-stor/ivo.navarrete/matformer-tokenizers/tinystories-instruct-sentencepiece-bpe-2k-v1 \
  --output-dir "$WARMUP_ROOT/campaign" --run-output-root "$WARMUP_ROOT/runs" \
  --linear-reference-root "$WARMUP_LINEAR_REF" \
  --geometric-reference-root "$WARMUP_GEOMETRIC_REF"

"$WARMUP_PYTHON" scripts/preflight_tinystories_s1_warmup.py cpu --campaign-root "$WARMUP_ROOT"
"$WARMUP_PYTHON" scripts/run_tinystories_s1_warmup.py prepare \
  --campaign-root "$WARMUP_ROOT" --cpu-evidence "$WARMUP_ROOT/diagnostics/cpu-gate.json"
```

Review the exact two configs and warmup-only differences, real parameter counts, data/role hashes, 43 excluded sequences, full expected schedules/traces, selected historical controls/terminals and source snapshot. In particular, position 256 is peak, applied first at update 257; final stored position 348528 is zero. Validating each historical S1 means its own grid and valid GPU attempt, not stale campaign-level status.

## 3. Subsequently authorized GPU readiness and execution

After diagnostic authorization, submit with `"$WARMUP_PYTHON" "$WARMUP_ROOT/source/scripts/preflight_tinystories_s1_warmup.py" submit-gpu --campaign-root "$WARMUP_ROOT"`. This invokes the frozen `source/scripts/preflight_tinystories_s1_warmup.py gpu --campaign-root ROOT` entry point, one process/GPU, excluded nodes `gpu-[05,50,51,54]`, and current user-wide limits. Do not bypass live admission by blindly executing a sample sbatch command. Diagnostics use real d64/l4/h4, batch64, context128 and bf16 for both grids, all widths and boundary/resume/failure checks. Save actual device/allocation/job evidence, zero mandatory GPU skips and exact CPU/source/config bindings.

Once execution is authorized and both gates pass, the queue entry is:

```bash
"$WARMUP_PYTHON" "$WARMUP_ROOT/source/scripts/run_tinystories_s1_warmup.py" queue \
  --campaign-root "$WARMUP_ROOT"
```

Only `S1-linear-w256` and `S1-geometric-w256` may run. Require fresh initial starts, explicit CUDA bf16 and maximum two running/four submitted jobs user-wide (or stricter live policy). Preserve unrelated jobs. Restart the same queue for reconciliation; uncertain submissions block duplicates. Continue only each run's own complete checkpoint. Full-horizon recovery completes missing outputs without new training updates.

## 4. Freeze and compare saved results

After phase-5 reporting implementation, use the frozen source and prepared mappings:

```bash
"$WARMUP_PYTHON" "$WARMUP_ROOT/source/scripts/run_tinystories_s1_warmup.py" report \
  --campaign-root "$WARMUP_ROOT"
```

The launcher freezes two full-budget terminals and eight new endpoints independently, then revalidates selected historical inputs and generates the comparison. Lower-level proposed analyzer commands are specified in the [CLI contract](contracts/lifecycle-and-cli.md).

Expected completion evidence:

- Each new run: 348528 updates, 2855141376 tokens, four finite terminal ordinary-validation endpoints, exact counterpart traces and matching successful GPU/job/worker evidence.
- New report: eight endpoints; comparison: 24 endpoints (12 per grid), eight paired differences and matching standalone gaps in CSV/JSON.
- Each grid:loss/parameter and perplexity/parameter views plus early LR/loss views in PNG/PDF;16 figure files total, inspectable scalar data and marked 64/256 boundaries.
- Findings:all eight effects, early observations, standalone gaps and limitations; consistent stage statuses, no final-holdout evaluation or historical retraining.

If references or required early evidence are missing/invalid, preserve valid new outputs and record comparison completion as outstanding. Do not fabricate losses, substitute partial/best checkpoints, launch replacements or reopen cancelled Feature 015 arms.
