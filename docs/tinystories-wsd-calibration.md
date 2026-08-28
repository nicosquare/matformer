# TinyStories-Instruct ratio-WSD calibration

This optional seed-42 workflow calibrates standalone d64/l4 g1000 training with
Warmup–Stable–Decay (WSD). It does not change the primary cosine plateau and
H=1 catch-up experiment, freeze standalone targets, evaluate the final holdout,
or support an elastic-training claim.

The stable learning-rate screen uses one aligned epoch. Each run warms up for
64 updates, holds its candidate LR, then spends the final 10% of its resolved
horizon on one cosine cooldown to zero. The selected LR is the lowest
ordinary-validation g1000 loss observed at or after the cooldown boundary. A
new three-epoch run then applies the same ratio policy to its longer horizon.

## 1. Resolve the shared data and isolated roots

Run from the repository root in the training environment:

```bash
export TINYSTORIES_PROFILE=instruct
source scripts/select_tinystories_profile.sh

export BASE=configs/controlled_exps/tinystories_instruct_wsd_calibration.yaml
export SEED="${SEED:-42}"
export EPOCH_TOKENS=713785344
export SCREEN_BUDGET=$EPOCH_TOKENS
export CALIBRATION_BUDGET=$((3 * EPOCH_TOKENS))
export WSD_ROOT="$MATFORMER_EXPERIMENT_ROOT/tinystories-instruct-wsd-calibration-v1"
export WSD_SCREEN_ROOT="$WSD_ROOT/standalone-1epoch-lr-screen-s$SEED"
export WSD_FULL_ROOT="$WSD_ROOT/standalone-3epoch-calibration-s$SEED"
export WSD_SELECTION_ROOT="$WSD_ROOT/analysis/lr-selection"
export WSD_CALIBRATION_ROOT="$WSD_ROOT/analysis/three-epoch-calibration"
export WSD_FIGURES_ROOT="$WSD_ROOT/figures/three-epoch-calibration"

test "$SEED" -eq 42
test -r "$BASE"
test -r "$TOKENIZER/tokenizer_manifest.json"
test -r "$CORPUS/corpus_manifest.json"
mkdir -p "$WSD_SCREEN_ROOT" "$WSD_FULL_ROOT" "$WSD_SELECTION_ROOT" "$WSD_CALIBRATION_ROOT" "$WSD_FIGURES_ROOT" logs
```

The screen and full-run roots must stay separate. A change to the budget, LR,
ratio, warmup, shape, corpus, or topology is a new scheduler contract and is
resume-incompatible.

## 2. Preflight the four one-epoch candidates

```bash
for LR in 0.002 0.004 0.006 0.008; do
  LR_SLUG="${LR/./p}"
  PREFLIGHT_FILE="$WSD_ROOT/preflight-screen-lr-$LR_SLUG.json"
  "$PYTHON_BIN" train.py \
    --config "$BASE" \
    --output-root "$WSD_SCREEN_ROOT/lr-$LR_SLUG" \
    --override "run.run_id=tiny-instruct-wsd-screen-1epoch-d64-l4-g1000-lr${LR_SLUG}-s$SEED" \
    --override "model.tokenizer_dir=$TOKENIZER" \
    --override "dataset.prepared_corpus_dir=$CORPUS" \
    --override "training.token_budget=$SCREEN_BUDGET" \
    --override "training.learning_rate=$LR" \
    --override "run.seed=$SEED" \
    --preflight > "$PREFLIGHT_FILE"
  test "$(jq -r '.derived_max_steps' "$PREFLIGHT_FILE")" -eq 87132
  test "$(jq -r '.scheduler_contract.decay_steps' "$PREFLIGHT_FILE")" -eq 8714
  test "$(jq -r '.scheduler_contract.stable_steps' "$PREFLIGHT_FILE")" -eq 78354
  test "$(jq -r '.scheduler_contract.cooldown_start_step' "$PREFLIGHT_FILE")" -eq 78418
done
```

Every preflight must also show `decay_ratio=0.1`, `warmup_type=linear`,
`decay_type=cosine`, `min_lr_ratio=0.0`, and `num_cycles=0.5`.

## 3. Launch the one-epoch LR screen

```bash
for LR in 0.002 0.004 0.006 0.008; do
  LR_SLUG="${LR/./p}"
  sbatch \
    --job-name="tiny-wsd-screen-lr$LR_SLUG-s$SEED" \
    --time=24:00:00 \
    scripts/slurm_tinystories_controlled.sh \
    --python-bin "$PYTHON_BIN" \
    --config "$BASE" \
    --output-root "$WSD_SCREEN_ROOT/lr-$LR_SLUG" \
    --override "run.run_id=tiny-instruct-wsd-screen-1epoch-d64-l4-g1000-lr${LR_SLUG}-s$SEED" \
    --override "model.tokenizer_dir=$TOKENIZER" \
    --override "dataset.prepared_corpus_dir=$CORPUS" \
    --override "training.token_budget=$SCREEN_BUDGET" \
    --override "training.learning_rate=$LR" \
    --override "run.seed=$SEED"
done
```

After all four jobs complete, freeze the LR selection provenance:

```bash
"$PYTHON_BIN" scripts/analyze_tinystories_plateau_catchup.py select-standalone-wsd-lr \
  --runs-root "$WSD_SCREEN_ROOT" \
  --output-dir "$WSD_SELECTION_ROOT"

export WSD_SELECTION="$WSD_SELECTION_ROOT/standalone_wsd_lr_selection.json"
export WSD_LR="$(jq -r '.selected_learning_rate' "$WSD_SELECTION")"
export WSD_SELECTION_HASH="$(jq -r '.manifest_hash' "$WSD_SELECTION")"
test -n "$WSD_LR"
test -n "$WSD_SELECTION_HASH"
test -r "$WSD_SELECTION_ROOT/standalone_wsd_lr_candidates.csv"
test -r "$WSD_SELECTION_ROOT/standalone_wsd_lr_selection.png"
```

The JSON manifest is immutable: rerunning the command with different candidate
provenance fails instead of replacing the selection.

## 4. Preflight and launch one fresh three-epoch calibration

Do not continue the selected one-epoch screen directory. Create a distinct run
whose three-epoch WSD boundaries are fixed before launch:

```bash
export WSD_FULL_RUN_ID="tiny-instruct-wsd-calibration-3epochs-d64-l4-g1000-lr${WSD_LR/./p}-s$SEED"

export PREFLIGHT_FILE="$WSD_ROOT/preflight-calibration-3epochs.json"
"$PYTHON_BIN" train.py \
  --config "$BASE" \
  --output-root "$WSD_FULL_ROOT" \
  --override "run.run_id=$WSD_FULL_RUN_ID" \
  --override "model.tokenizer_dir=$TOKENIZER" \
  --override "dataset.prepared_corpus_dir=$CORPUS" \
  --override "training.token_budget=$CALIBRATION_BUDGET" \
  --override "training.learning_rate=$WSD_LR" \
  --override "controlled_experiment.selection_report_hash=$WSD_SELECTION_HASH" \
  --override "run.seed=$SEED" \
  --preflight > "$PREFLIGHT_FILE"

test "$(jq -r '.derived_max_steps' "$PREFLIGHT_FILE")" -eq 261396
test "$(jq -r '.scheduler_contract.decay_steps' "$PREFLIGHT_FILE")" -eq 26140
test "$(jq -r '.scheduler_contract.stable_steps' "$PREFLIGHT_FILE")" -eq 235192
test "$(jq -r '.scheduler_contract.cooldown_start_step' "$PREFLIGHT_FILE")" -eq 235256

sbatch \
  --job-name="tiny-wsd-calibration-3ep-s$SEED" \
  --time=36:00:00 \
  scripts/slurm_tinystories_controlled.sh \
  --python-bin "$PYTHON_BIN" \
  --config "$BASE" \
  --output-root "$WSD_FULL_ROOT" \
  --override "run.run_id=$WSD_FULL_RUN_ID" \
  --override "model.tokenizer_dir=$TOKENIZER" \
  --override "dataset.prepared_corpus_dir=$CORPUS" \
  --override "training.token_budget=$CALIBRATION_BUDGET" \
  --override "training.learning_rate=$WSD_LR" \
  --override "controlled_experiment.selection_report_hash=$WSD_SELECTION_HASH" \
  --override "run.seed=$SEED"
```

An interrupted three-epoch job may be resubmitted with the identical command
and run ID. It may not load a one-epoch checkpoint or a checkpoint with a
different WSD contract.

## 5. Analyze the calibration and generate standard figures

The trainer inserts an output-group directory below `$WSD_FULL_ROOT`. Resolve
the one completed run directory and analyze it:

```bash
export WSD_FULL_RUN_DIR="$(find "$WSD_FULL_ROOT" -type f -name run_summary.json -print -quit | xargs dirname)"
test -n "$WSD_FULL_RUN_DIR"

"$PYTHON_BIN" scripts/analyze_tinystories_plateau_catchup.py wsd-calibration \
  --run-dir "$WSD_FULL_RUN_DIR" \
  --wsd-selection "$WSD_SELECTION" \
  --output-dir "$WSD_CALIBRATION_ROOT"

test -r "$WSD_CALIBRATION_ROOT/wsd_calibration_report.json"
test -r "$WSD_CALIBRATION_ROOT/wsd_calibration_validation.csv"
test -r "$WSD_CALIBRATION_ROOT/wsd_calibration.png"
test ! -e "$WSD_CALIBRATION_ROOT/frozen_standalone_targets.json"

"$PYTHON_BIN" scripts/make_figures.py \
  --input "$WSD_FULL_ROOT" \
  --output "$WSD_FIGURES_ROOT" \
  --skip-final-holdout \
  --validation-loss-log-y

test -r "$WSD_FIGURES_ROOT/learning_rate_schedule.png"
test -r "$WSD_FIGURES_ROOT/validation_loss_over_tokens_granularity_comparison.png"
```

Inspect stable quarter-epoch improvements, stable and cooldown best losses,
cooldown final loss and gain, the best-checkpoint SHA256, phase boundaries, and
the loss/LR plot. `--skip-final-holdout` is required because this calibration
intentionally generates ordinary-validation figures without requiring a sealed
final-holdout result. This calibration ends with those diagnostics; it does not
freeze a target, launch seeds 43/44, convert to elastic WSD, make a catch-up
claim, or open the final holdout.
