# Run the Per-Width Optimizer-State Experiments

This is the copy-paste operator guide for the TinyStories-Instruct comparison
between a shared optimizer state and one optimizer state per global width. The
scientific protocol and endpoint definitions remain authoritative in
[tinystories-per-width-optimizer-experiment.md](./tinystories-per-width-optimizer-experiment.md).

Run every command from the repository root. The experiment has two phases:

- pilot: 713,785,344 tokens, 87,132 global steps, 21,783 updates per width;
- confirmation: 2,141,356,032 tokens, 261,396 global steps, 65,349 updates per
  width.

Each phase has six independent runs: seeds 42, 43, and 44 for both `shared`
and `per_granularity` optimizer-state scopes.

## 1. Set up the environment and paths

Activate the validated environment and select the prepared
TinyStories-Instruct profile:

```bash
conda activate elasticnn
# If the prepared artifacts are not under the profile's default NFS roots,
# export MATFORMER_TOKENIZER_ROOT and MATFORMER_CORPUS_ROOT here first.
export TINYSTORIES_PROFILE=instruct
source scripts/select_tinystories_profile.sh
test -d "$TOKENIZER"
test -d "$CORPUS"
```

Choose an experiment root on durable storage. Pilot and confirmation must use
different roots and run IDs:

```bash
export OPTIMIZER_STATE_EXPERIMENT_ROOT="$MATFORMER_EXPERIMENT_ROOT/per-width-optimizer-state-v1"
export PILOT_ROOT="$OPTIMIZER_STATE_EXPERIMENT_ROOT/pilot"
export CONFIRM_ROOT="$OPTIMIZER_STATE_EXPERIMENT_ROOT/confirmation"
mkdir -p "$PILOT_ROOT" "$CONFIRM_ROOT" logs
```

Set a Slurm wall time appropriate for the cluster. The launcher already
requests one GPU, four CPUs, and 16 GiB of memory; command-line `sbatch`
options can override its defaults.

```bash
export OPTIMIZER_STATE_WALLTIME="1-00:00:00"
```

Confirm that the resolved executables and paths are correct:

```bash
"$PYTHON_BIN" --version
printf 'python=%s\ntokenizer=%s\ncorpus=%s\npilot=%s\nconfirmation=%s\n' \
  "$PYTHON_BIN" "$TOKENIZER" "$CORPUS" "$PILOT_ROOT" "$CONFIRM_ROOT"
```

## 2. Audit the prepared corpus

Run this once before either phase:

```bash
"$PYTHON_BIN" scripts/audit_prepared_corpus.py \
  --prepared-corpus-dir "$CORPUS" \
  --prepared-tokenizer-dir "$TOKENIZER" \
  --required-vocab-size 2048 \
  --minimum-training-tokens 713785344
```

Do not proceed if the audit fails. Keep the optimizer-training, controller,
ordinary-validation, and final-holdout manifests unchanged for the entire
experiment.

## 3. Define reusable commands

Paste these Bash functions into the active shell. They generate distinct pilot
and confirmation run IDs and keep each output directory aligned with its run
ID, as required by configuration validation.

```bash
optimizer_state_run_id() {
  local phase="$1"
  local scope="$2"
  local seed="$3"
  local scope_slug="${scope//_/-}"
  printf 'optimizer-state-%s-%s-s%s' "$phase" "$scope_slug" "$seed"
}

preflight_optimizer_state_arm() {
  local phase="$1"
  local root="$2"
  local budget="$3"
  local scope="$4"
  local seed="$5"
  local pilot_holdout_opened="$6"
  local run_id
  run_id="$(optimizer_state_run_id "$phase" "$scope" "$seed")"

  mkdir -p "$root/preflight"
  "$PYTHON_BIN" train.py \
    --config configs/controlled_exps/tinystories_instruct_per_width_optimizers.yaml \
    --override "run.seed=$seed" \
    --override "run.run_id=$run_id" \
    --override "model.tokenizer_dir=$TOKENIZER" \
    --override "dataset.prepared_corpus_dir=$CORPUS" \
    --override "training.token_budget=$budget" \
    --override "training.optimizer.state_scope=$scope" \
    --override "controlled_experiment.holdout_opened_during_pilot=$pilot_holdout_opened" \
    --output-dir "$root/$run_id" \
    --preflight | tee "$root/preflight/$run_id.json"
  test "${PIPESTATUS[0]}" -eq 0
}

submit_optimizer_state_arm() {
  local phase="$1"
  local root="$2"
  local budget="$3"
  local scope="$4"
  local seed="$5"
  local pilot_holdout_opened="$6"
  local run_id
  run_id="$(optimizer_state_run_id "$phase" "$scope" "$seed")"

  sbatch \
    --job-name="$run_id" \
    --time="$OPTIMIZER_STATE_WALLTIME" \
    scripts/slurm_tinystories_controlled.sh \
    --python-bin "$PYTHON_BIN" \
    --config configs/controlled_exps/tinystories_instruct_per_width_optimizers.yaml \
    --override "run.seed=$seed" \
    --override "run.run_id=$run_id" \
    --override "model.tokenizer_dir=$TOKENIZER" \
    --override "dataset.prepared_corpus_dir=$CORPUS" \
    --override "training.token_budget=$budget" \
    --override "training.optimizer.state_scope=$scope" \
    --override "controlled_experiment.holdout_opened_during_pilot=$pilot_holdout_opened" \
    --output-dir "$root/$run_id"
}

optimizer_state_run_args() {
  local phase="$1"
  local root="$2"
  local seed scope run_id
  for seed in 42 43 44; do
    for scope in shared per_granularity; do
      run_id="$(optimizer_state_run_id "$phase" "$scope" "$seed")"
      printf '%s\n' --run-dir "$root/$run_id"
    done
  done
}

check_optimizer_state_preflights() {
  local phase="$1"
  local root="$2"
  local expected_steps="$3"
  local expected_updates="$4"

  "$PYTHON_BIN" - "$phase" "$root/preflight" "$expected_steps" "$expected_updates" <<'PY'
import json
import sys
from pathlib import Path

phase = sys.argv[1]
root = Path(sys.argv[2])
expected_steps = int(sys.argv[3])
expected_updates = int(sys.argv[4])
widths = ["g250", "g500", "g750", "g1000"]
for seed in (42, 43, 44):
    records = {}
    for scope, slug in (("shared", "shared"), ("per_granularity", "per-granularity")):
        path = root / f"optimizer-state-{phase}-{slug}-s{seed}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["optimizer_state_scope"] == scope
        assert record["optimizer_scheduler_clock"] == "global_step"
        assert record["ordered_granularities"] == widths
        assert record["global_sampling_schedule"] == "balanced_cycle"
        assert record["global_sampling_interval_steps"] == 1
        assert record["optimizer"]["name"] == "adamw"
        assert record["learning_rate"] == 0.008
        assert record["resolved_warmup_steps"] == 64
        assert record["derived_max_steps"] == expected_steps
        assert expected_steps // len(widths) == expected_updates
        assert record["effective_world_size"] == 1
        records[scope] = record
    assert records["shared"]["paired_control_signature"] == records["per_granularity"]["paired_control_signature"]
    print(f"seed {seed}: paired controls match; {expected_updates:,} updates per width")
PY
}

check_optimizer_state_runs() {
  local phase="$1"
  local root="$2"
  local expected_steps="$3"
  local expected_updates="$4"

  "$PYTHON_BIN" - "$phase" "$root" "$expected_steps" "$expected_updates" <<'PY'
import json
import sys
from pathlib import Path

phase = sys.argv[1]
root = Path(sys.argv[2])
expected_steps = int(sys.argv[3])
expected_updates = int(sys.argv[4])
expected = {width: expected_updates for width in ("g250", "g500", "g750", "g1000")}
for seed in (42, 43, 44):
    for scope, slug in (("shared", "shared"), ("per_granularity", "per-granularity")):
        run_id = f"optimizer-state-{phase}-{slug}-s{seed}"
        run_dir = root / run_id
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "completed"
        assert summary["optimizer_state_scope"] == scope
        assert summary["committed_optimizer_steps"] == expected_steps
        assert summary["optimizer_successful_update_counts"] == expected
        assert summary["optimizer_exposure_counts"] == expected
        assert summary["optimizer_accounting_reconciled"] is True
        assert summary["terminal_checkpoint_purpose"] == "resumable_training"
        assert (run_dir / "checkpoints" / "latest.pt").is_file()
        print(f"complete: {run_id}")
PY
}
```

The submission helper is also the resume command. If a job stops, submit the
identical function call again. Never submit the same run ID concurrently.

## 4. Preflight all six pilot arms

Keep the final holdout sealed during the pilot if confirmation remains an
option:

```bash
export PILOT_HOLDOUT_OPENED=false

for seed in 42 43 44; do
  for scope in shared per_granularity; do
    preflight_optimizer_state_arm \
      pilot "$PILOT_ROOT" 713785344 "$scope" "$seed" \
      "$PILOT_HOLDOUT_OPENED"
  done
done
```

Validate the saved preflight records and paired signatures:

```bash
check_optimizer_state_preflights pilot "$PILOT_ROOT" 87132 21783
```

Do not submit the pilot unless all six preflights and all three paired-signature
checks pass.

## 5. Submit and monitor the pilot

```bash
for seed in 42 43 44; do
  for scope in shared per_granularity; do
    submit_optimizer_state_arm \
      pilot "$PILOT_ROOT" 713785344 "$scope" "$seed" \
      "$PILOT_HOLDOUT_OPENED"
  done
done
```

Monitor jobs and launcher logs with:

```bash
squeue -u "$(id -un)"
ls -lt logs/tinystories_controlled_*.out | head
```

For an interrupted arm, rerun only its identical submission, for example:

```bash
submit_optimizer_state_arm \
  pilot "$PILOT_ROOT" 713785344 per_granularity 42 \
  "$PILOT_HOLDOUT_OPENED"
```

## 6. Check that all pilot runs completed

```bash
check_optimizer_state_runs pilot "$PILOT_ROOT" 87132 21783
```

## 7. Freeze the pilot and create the diagnostic report

Freeze before opening the final holdout:

```bash
mapfile -t PILOT_RUN_ARGS < <(
  optimizer_state_run_args pilot "$PILOT_ROOT"
)

"$PYTHON_BIN" scripts/analyze_tinystories_per_width_optimizer.py freeze \
  --phase pilot \
  "${PILOT_RUN_ARGS[@]}" \
  --output-dir "$PILOT_ROOT/analysis"

"$PYTHON_BIN" scripts/analyze_tinystories_per_width_optimizer.py report \
  --manifest "$PILOT_ROOT/analysis/optimizer_state_manifest.json" \
  --output-dir "$PILOT_ROOT/analysis"
```

At this point the report is diagnostic and uses ordinary validation plus
resource measurements. Inspect:

```bash
"$PYTHON_BIN" -m json.tool \
  "$PILOT_ROOT/analysis/optimizer_state_comparison.json" | less
```

## 8. Choose the holdout path

Make one decision and record it before evaluating the final holdout.

### Path A: finish with pilot-only diagnostic evidence

If no confirmation will be run, evaluate the six frozen pilot endpoints and
rerun the report. The evidence label remains `diagnostic`.

```bash
PILOT_HOLDOUT_ARGS=()
for seed in 42 43 44; do
  for scope in shared per_granularity; do
    run_id="$(optimizer_state_run_id pilot "$scope" "$seed")"
    PILOT_HOLDOUT_ARGS+=(--final-holdout-only "$PILOT_ROOT/$run_id")
  done
done

sbatch \
  --job-name=optimizer-state-pilot-holdout \
  --time="$OPTIMIZER_STATE_WALLTIME" \
  scripts/slurm_tinystories_controlled.sh \
  --python-bin "$PYTHON_BIN" \
  "${PILOT_HOLDOUT_ARGS[@]}"
```

After the holdout job finishes:

```bash
"$PYTHON_BIN" scripts/analyze_tinystories_per_width_optimizer.py report \
  --manifest "$PILOT_ROOT/analysis/optimizer_state_manifest.json" \
  --output-dir "$PILOT_ROOT/analysis"
```

### Path B: run a fresh confirmation

Do not evaluate the pilot holdout. Keep:

```bash
export PILOT_HOLDOUT_OPENED=false
```

If the pilot holdout was already opened, set the value to `true` before every
confirmation preflight and submission. The resulting evidence will be labeled
`descriptive_after_holdout_open`, not confirmatory.

Preflight the fresh confirmation runs:

```bash
for seed in 42 43 44; do
  for scope in shared per_granularity; do
    preflight_optimizer_state_arm \
      confirmation "$CONFIRM_ROOT" 2141356032 "$scope" "$seed" \
      "$PILOT_HOLDOUT_OPENED"
  done
done
```

Check all confirmation preflights, then submit:

```bash
check_optimizer_state_preflights confirmation "$CONFIRM_ROOT" 261396 65349
```

```bash
for seed in 42 43 44; do
  for scope in shared per_granularity; do
    submit_optimizer_state_arm \
      confirmation "$CONFIRM_ROOT" 2141356032 "$scope" "$seed" \
      "$PILOT_HOLDOUT_OPENED"
  done
done
```

Do not copy or continue pilot checkpoints into the confirmation root.

## 9. Freeze and evaluate confirmation

After all confirmation summaries report `completed`, freeze the six fresh
runs before final-holdout evaluation:

```bash
check_optimizer_state_runs confirmation "$CONFIRM_ROOT" 261396 65349

mapfile -t CONFIRM_RUN_ARGS < <(
  optimizer_state_run_args confirmation "$CONFIRM_ROOT"
)

"$PYTHON_BIN" scripts/analyze_tinystories_per_width_optimizer.py freeze \
  --phase confirmation \
  "${CONFIRM_RUN_ARGS[@]}" \
  --output-dir "$CONFIRM_ROOT/analysis"
```

Evaluate the six frozen terminal checkpoints:

```bash
CONFIRM_HOLDOUT_ARGS=()
for seed in 42 43 44; do
  for scope in shared per_granularity; do
    run_id="$(optimizer_state_run_id confirmation "$scope" "$seed")"
    CONFIRM_HOLDOUT_ARGS+=(--final-holdout-only "$CONFIRM_ROOT/$run_id")
  done
done

sbatch \
  --job-name=optimizer-state-confirmation-holdout \
  --time="$OPTIMIZER_STATE_WALLTIME" \
  scripts/slurm_tinystories_controlled.sh \
  --python-bin "$PYTHON_BIN" \
  "${CONFIRM_HOLDOUT_ARGS[@]}"
```

After all six `final_holdout_results.json` files exist, create the final report:

```bash
"$PYTHON_BIN" scripts/analyze_tinystories_per_width_optimizer.py report \
  --manifest "$CONFIRM_ROOT/analysis/optimizer_state_manifest.json" \
  --output-dir "$CONFIRM_ROOT/analysis"
```

A complete confirmation with `PILOT_HOLDOUT_OPENED=false` is eligible for the
`confirmatory` evidence label. The analyzer reports outcomes and eligibility;
it does not declare scientific superiority.

## 10. Expected artifacts

Every completed run directory should contain:

```text
config.json
metrics.csv
run_summary.json
checkpoints/latest.pt
final_holdout_results.json       # only after explicit holdout evaluation
```

Each analysis directory should contain:

```text
optimizer_state_manifest.json
optimizer_state_comparison.json
optimizer_state_comparison.csv
```

The primary endpoint is the seed-level `per_granularity - shared` difference
in uniform mean final-holdout loss. Always retain the signed per-seed values,
per-width outcomes, worst-width loss, update/exposure reconciliation, wall
time, peak accelerator memory, and checkpoint size.

## 11. Safety rules

- Never run the same run ID concurrently.
- Resume only by resubmitting the identical command and output directory.
- Never resume confirmation from a pilot checkpoint.
- Freeze a manifest before evaluating the final holdout.
- Do not change tokenizer, corpus, widths, scheduler, optimizer kwargs, budget,
  evaluation cadence, or action policy between paired arms.
- The only paired-arm intervention is
  `training.optimizer.state_scope=shared|per_granularity`.
- Keep all run directories and frozen manifests immutable after analysis.
