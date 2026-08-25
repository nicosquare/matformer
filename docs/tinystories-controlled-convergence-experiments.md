# TinyStories frozen-recipe elastic experiments

This guide runs standalone-width and elastic-training comparisons with the
converged TinyStories recipe. The recipe is frozen in
`configs/controlled_exps/tinystories_controlled_convergence.yaml`: learning
rate `3e-3`, cosine decay, 64 warmup updates, and 33,554,432 tokens over 4,096
one-GPU optimizer updates. `TOKEN_BUDGET` retains that selected horizon by
default and is the only recipe field varied by longer-horizon comparisons;
do not override the optimizer, learning rate, scheduler, warmup, or batch
geometry.

The immutable tokenizer and packed corpus are prerequisites, not routine
steps. Section 11 documents their one-time preparation. Section 12 records the
completed hyperparameter-selection process that produced the frozen recipe.
On a first setup, export the path variables at the start of section 1, run
section 11, and then return to section 1 for the manifest and budget checks.

The main comparison asks how elastic training changes each width relative to
independent standalone training. It covers the default eight-width grid and a
matched four-width subset, uniform global sampling with several hold windows,
Thompson sampling, PanelGrad, and optional joint-training and balanced-exposure
references. The uniform runs can also opt into the same fixed-probe
gradient-interference diagnostic used by the larger setup.

## 1. Set the shared contract

### 1.1 Choose the budget first

Run from the repository root, activate the validated environment, and choose
exactly one horizon before defining any setup, policy, or submission arrays:

| Horizon | Exact tokens | Optimizer updates | Command |
| --- | ---: | ---: | --- |
| 1x (default) | 33,554,432 | 4,096 | `export TOKEN_BUDGET=33554432` |
| 2x | 67,108,864 | 8,192 | `export TOKEN_BUDGET=67108864` |
| 4x | 134,217,728 | 16,384 | `export TOKEN_BUDGET=134217728` |
| 8x | 268,435,456 | 32,768 | `export TOKEN_BUDGET=268435456` |

For the 2x campaign requested here, start with:

```bash
conda activate elasticnn
export TOKEN_BUDGET=67108864
```

If `TOKEN_BUDGET` is unset, setup falls back to the 1x value. An explicitly
exported value is preserved, so do not rely on the fallback when switching an
existing shell from one campaign to another.

### 1.2 Resolve and validate the selected budget

After choosing the budget, run this entire block:

```bash

tinystories_setup() {
  export PYTHON_BIN="$(command -v python)"
  export EXPERIMENT_SEED=42
  export WALLTIME="00:30:00"

  if [[ -x "$PYTHON_BIN" ]]; then
    :
  else
    echo "The active environment does not provide python" >&2
    return 2
  fi

  export NFS_USER_ROOT="${NFS_USER_ROOT:-/nfs-stor/$USER}"
  export MATFORMER_TOKENIZER_ROOT="${MATFORMER_TOKENIZER_ROOT:-$NFS_USER_ROOT/matformer-tokenizers}"
  export MATFORMER_CORPUS_ROOT="${MATFORMER_CORPUS_ROOT:-$NFS_USER_ROOT/matformer-corpora}"
  export MATFORMER_EXPERIMENT_ROOT="${MATFORMER_EXPERIMENT_ROOT:-$NFS_USER_ROOT/results/elasticnn}"

  export TINYSTORIES_TOKENIZER_NAME="${TINYSTORIES_TOKENIZER_NAME:-tinystories-sentencepiece-bpe-2k-v1}"
  export TINYSTORIES_CORPUS_NAME="${TINYSTORIES_CORPUS_NAME:-tinystories-packed-full-v1}"
  export TINYSTORIES_EXPERIMENT_NAME="${TINYSTORIES_EXPERIMENT_NAME:-tinystories-frozen-elastic-v2}"

  # Fall back to the frozen 1x horizon only when no budget was selected above.
  export TOKEN_BUDGET="${TOKEN_BUDGET:-33554432}"
  export TOKENS_PER_STEP=8192

  if [[ "$TOKEN_BUDGET" =~ ^[0-9]+$ ]]; then
    :
  else
    echo "TOKEN_BUDGET must be a positive integer" >&2
    return 2
  fi
  if (( TOKEN_BUDGET > 0 && TOKEN_BUDGET % TOKENS_PER_STEP == 0 )); then
    :
  else
    echo "TOKEN_BUDGET must be positive and divisible by $TOKENS_PER_STEP" >&2
    return 2
  fi

  export DERIVED_MAX_STEPS=$((TOKEN_BUDGET / TOKENS_PER_STEP))
  export BUDGET_RUN_SLUG="${DERIVED_MAX_STEPS}steps"
  if TOKEN_BUDGET_GROUP="$("$PYTHON_BIN" -c \
    'from src.utils.model_size import derive_token_budget_slug; import sys; print(derive_token_budget_slug(int(sys.argv[1])))' \
    "$TOKEN_BUDGET")"; then
    export TOKEN_BUDGET_GROUP
  else
    echo "Could not derive the token-budget output slug" >&2
    return 2
  fi

  export TOKENIZER="$MATFORMER_TOKENIZER_ROOT/$TINYSTORIES_TOKENIZER_NAME"
  export CORPUS="$MATFORMER_CORPUS_ROOT/$TINYSTORIES_CORPUS_NAME"
  export BASE=configs/controlled_exps/tinystories_controlled_convergence.yaml
  export OUT_8G="$MATFORMER_EXPERIMENT_ROOT/${TINYSTORIES_EXPERIMENT_NAME}-${BUDGET_RUN_SLUG}-8g"
  export OUT_4G="$MATFORMER_EXPERIMENT_ROOT/${TINYSTORIES_EXPERIMENT_NAME}-${BUDGET_RUN_SLUG}-4g"
  export SLURM_LOG_ROOT=./logs

  if mkdir -p "$OUT_8G" "$OUT_4G" "$SLURM_LOG_ROOT"; then
    :
  else
    echo "Could not create the experiment or log directories" >&2
    return 2
  fi
  if [[ -w "$OUT_8G" && -w "$OUT_4G" ]]; then
    :
  else
    echo "The experiment output directories are not writable" >&2
    return 2
  fi
  if [[ -r "$TOKENIZER/tokenizer_manifest.json" && -r "$CORPUS/corpus_manifest.json" ]]; then
    :
  else
    echo "Prepared artifacts are missing; run section 11, then run tinystories_setup again" >&2
    return 2
  fi

  if CORPUS_CONTRACT="$("$PYTHON_BIN" -c \
    'import json, pathlib, sys; m=json.loads((pathlib.Path(sys.argv[1]) / "corpus_manifest.json").read_text()); print(m["available_optimizer_token_count"], m["source"]["termination"])' \
    "$CORPUS")"; then
    :
  else
    echo "Could not read the prepared corpus manifest" >&2
    return 2
  fi
  read -r AVAILABLE_CORPUS_TOKENS CORPUS_TERMINATION <<<"$CORPUS_CONTRACT"
  if [[ "$CORPUS_TERMINATION" == source_exhausted ]]; then
    :
  else
    echo "Expected a full source-exhausted corpus; found: $CORPUS_TERMINATION" >&2
    return 2
  fi

  export MAX_ALIGNED_TOKEN_BUDGET=$((AVAILABLE_CORPUS_TOKENS / TOKENS_PER_STEP * TOKENS_PER_STEP))
  export MAX_ALIGNED_STEPS=$((MAX_ALIGNED_TOKEN_BUDGET / TOKENS_PER_STEP))
  if (( TOKEN_BUDGET <= AVAILABLE_CORPUS_TOKENS )); then
    :
  else
    echo "TOKEN_BUDGET=$TOKEN_BUDGET exceeds corpus capacity=$AVAILABLE_CORPUS_TOKENS" >&2
    return 2
  fi

  printf 'tokens=%s optimizer_steps=%s corpus_capacity=%s max_aligned_tokens=%s max_aligned_steps=%s\n' \
    "$TOKEN_BUDGET" "$DERIVED_MAX_STEPS" "$AVAILABLE_CORPUS_TOKENS" \
    "$MAX_ALIGNED_TOKEN_BUDGET" "$MAX_ALIGNED_STEPS"
}

tinystories_setup
```

For 2x, do not continue unless the final line reports:

```text
tokens=67108864 optimizer_steps=8192
```

To train on the largest complete optimizer-step prefix, set `TOKEN_BUDGET` to
the printed `max_aligned_tokens` value and rerun every block in this section.
The raw corpus capacity may contain a final partial 8,192-token update and must
not be used directly unless it is already aligned.

The setup function returns on validation failures instead of exiting the
interactive shell. If it reports missing prepared artifacts, run section 11
and then invoke `tinystories_setup` again.

The model is a four-layer Llama decoder with `d_model=128`, four heads,
context length 128, a 2,048-token vocabulary, and a 512-unit full-width SwiGLU
FFN. Batch size 64 gives 8,192 tokens per update. The default configuration is
the eight-width grid; the four-width grid is an evenly spaced subset:

```bash
COMMON_OVERRIDES=(
  --override "run.seed=$EXPERIMENT_SEED"
  --override "model.tokenizer_dir=$TOKENIZER"
  --override "dataset.prepared_corpus_dir=$CORPUS"
  --override "training.token_budget=$TOKEN_BUDGET"
)

EIGHT_GRANULARITY_OVERRIDES=(
  "${COMMON_OVERRIDES[@]}"
)

FOUR_GRANULARITY_OVERRIDES=(
  "${COMMON_OVERRIDES[@]}"
  --override 'model.granularities=[g250,g500,g750,g1000]'
  --override 'model.granularity_prefixes={g250: 0.25, g500: 0.50, g750: 0.75, g1000: 1.00}'
)

SBATCH=(
  sbatch
  --output="$SLURM_LOG_ROOT/%x_%j.out"
  --error="$SLURM_LOG_ROOT/%x_%j.err"
)
if [[ -n "${SLURM_EXCLUDE:-}" ]]; then
  SBATCH+=(--exclude="$SLURM_EXCLUDE")
fi
```

### 1.3 Required sequence after changing the budget

Changing `TOKEN_BUDGET` in an existing shell does not rewrite arrays that were
already defined. Use this order every time:

1. Export the new `TOKEN_BUDGET` from section 1.1.
2. Run `tinystories_setup` from section 1.2 and verify its reported tokens and
   optimizer steps.
3. Rerun the `COMMON_OVERRIDES`, granularity, and `SBATCH` block immediately
   above so it captures the new budget and output roots.
4. Rerun all of section 2 so budget-dependent policies and diagnostic arrays
   are rebuilt.
5. Run the section 3 preflights, redefine the section 4 submission helper, and
   only then submit a campaign from section 5 onward.

The resulting FFN prefix widths are `64, 128, 192, 256, 320, 384, 448,
512` for eight granularities and `128, 256, 384, 512` for four. Both scopes
retain the same data roles, initialization seed, optimizer, scheduler, selected
token budget, and validation cadence. Changing `TOKEN_BUDGET` creates a new
cosine schedule and budget-isolated output roots, so every such run starts from
initialization rather than resuming a completed shorter schedule.

## 2. Define the adaptive policies

Run this section after both section 1 blocks every time the budget changes.
Use the established 25-update controller cadence. Uniform `H=25` is the
no-controller temporal-batching control for both adaptive policies.

```bash
export THOMPSON_CONTROLLER='{"preset":"bayesian_thompson","decision_interval_steps":25,"prior_mean":0.0,"prior_covariance":1.0,"observation_noise_variance":0.01,"process_noise_covariance":0.0001,"reset":{"enabled":false}}'
export PANELGRAD_RMS='{"importance_metric":"gradient_rms","refresh_interval_steps":25,"eta":1.0e-12,"temperature":1.0,"epsilon":0.1}'
export PANELGRAD_L2="{\"importance_metric\":\"gradient_l2\",\"refresh_interval_steps\":25,\"eta\":1.0e-12,\"temperature\":1.0,\"epsilon_schedule\":{\"type\":\"linear\",\"start\":0.5,\"end\":0.1,\"duration_steps\":$DERIVED_MAX_STEPS}}"
```

PanelGrad performs controller-gradient measurements in addition to the normal
training update, so compare its validation quality together with its extra
wall-clock cost.

### Optional gradient-interference diagnostic

The uniform-global runs can measure raw-gradient compatibility between every
pair of nested widths on the fixed controller probe. It is disabled by default:

```bash
# Change to 1 before running preflights/submissions for a diagnostic campaign.
export ENABLE_GRADIENT_INTERFERENCE=0

GRADIENT_INTERFERENCE_OVERRIDES=(
  --override evaluation.gradient_interference.enabled=false
)
GRADIENT_INTERFERENCE_RUN_SLUG=""
if [[ "${ENABLE_GRADIENT_INTERFERENCE:-0}" == 1 ]]; then
  GRADIENT_INTERFERENCE_OVERRIDES=(
    --override evaluation.gradient_interference.enabled=true
    --override 'evaluation.gradient_interference.trajectory_fractions=[0.0,0.25,0.5,0.75,1.0]'
    --override evaluation.gradient_interference.include_warmup_completion=true
    --override evaluation.gradient_interference.layerwise=true
  )
  GRADIENT_INTERFERENCE_RUN_SLUG="-gradient-interference"
fi
```

The disabled form deliberately contains an explicit override instead of being
an empty array. This prevents a stale empty scalar with the same name from
being forwarded to `train.py` as a literal empty argument.

The separate run slug prevents a diagnostic run from colliding with an
ordinary run's immutable continuation state. Apply this array only to uniform
global `nested-random` jobs, including the balanced screen; it is intentionally
incompatible with Thompson, PanelGrad, standalone, and nested-all.

The resolved milestones are the deduplicated union of step `0`, the 25%, 50%,
75%, and 100% trajectory fractions of `DERIVED_MAX_STEPS`, and warmup
completion at step 64. At the default 4,096-update horizon these are steps
`0, 64, 1024, 2048, 3072, 4096`. With the 128-example probe and batch size 64,
each snapshot costs 16 backward evaluations for eight widths or eight for four
widths. The diagnostic journal records the measured cost separately from
training.

## 3. Run representative preflights

Verify the eight-width uniform contract:

```bash
"$PYTHON_BIN" train.py \
  --config "$BASE" \
  --output-root "$OUT_8G" \
  --override "run.run_id=preflight-tiny-${BUDGET_RUN_SLUG}-8g-uniform-h25${GRADIENT_INTERFERENCE_RUN_SLUG}-s$EXPERIMENT_SEED" \
  "${EIGHT_GRANULARITY_OVERRIDES[@]}" \
  "${GRADIENT_INTERFERENCE_OVERRIDES[@]}" \
  --override run.model_family=nested \
  --override run.sampling_mode=nested-random \
  --override run.granularity=null \
  --override model.granularity_sampling_mode=global \
  --override model.global_sampling_schedule=random_with_replacement \
  --override model.global_sampling_interval_steps=25 \
  --preflight
```

Verify the four-width override and the policies whose configuration differs:

```bash
"$PYTHON_BIN" train.py \
  --config "$BASE" \
  --output-root "$OUT_4G" \
  --override "run.run_id=preflight-tiny-${BUDGET_RUN_SLUG}-4g-uniform-h25${GRADIENT_INTERFERENCE_RUN_SLUG}-s$EXPERIMENT_SEED" \
  "${FOUR_GRANULARITY_OVERRIDES[@]}" \
  "${GRADIENT_INTERFERENCE_OVERRIDES[@]}" \
  --override run.model_family=nested \
  --override run.sampling_mode=nested-random \
  --override run.granularity=null \
  --override model.granularity_sampling_mode=global \
  --override model.global_sampling_schedule=random_with_replacement \
  --override model.global_sampling_interval_steps=25 \
  --preflight

"$PYTHON_BIN" train.py \
  --config "$BASE" \
  --output-root "$OUT_8G" \
  --override "run.run_id=preflight-tiny-${BUDGET_RUN_SLUG}-8g-thompson-s$EXPERIMENT_SEED" \
  "${EIGHT_GRANULARITY_OVERRIDES[@]}" \
  --override run.model_family=nested \
  --override run.sampling_mode=nested-random \
  --override run.granularity=null \
  --override model.granularity_sampling_mode=adaptive_global \
  --override model.adaptive_sampler_strategy=thompson \
  --override "model.adaptive_controller=$THOMPSON_CONTROLLER" \
  --preflight

"$PYTHON_BIN" train.py \
  --config "$BASE" \
  --output-root "$OUT_8G" \
  --override "run.run_id=preflight-tiny-${BUDGET_RUN_SLUG}-8g-panelgrad-l2-s$EXPERIMENT_SEED" \
  "${EIGHT_GRANULARITY_OVERRIDES[@]}" \
  --override run.model_family=nested \
  --override run.sampling_mode=nested-random \
  --override run.granularity=null \
  --override model.granularity_sampling_mode=adaptive_global \
  --override model.adaptive_sampler_strategy=panelgrad \
  --override "model.panelgrad=$PANELGRAD_L2" \
  --preflight
```

Every preflight must report one process/GPU, 8,192 tokens per update,
`TOKEN_BUDGET` total tokens, `DERIVED_MAX_STEPS` updates, 64 warmup updates,
and the expected ordered width grid. It must also preserve the frozen learning
rate and cosine scheduler. The PanelGrad L2 epsilon duration must equal
`DERIVED_MAX_STEPS`.

## 4. Define the submission helper

The helper chooses the output root and width grid while keeping the run mode
explicit at each call site:

```bash
submit_tinystories_run() {
  local scope="$1"
  local job_name="$2"
  local run_id="$3"
  shift 3

  local output_root
  local -a scope_overrides
  case "$scope" in
    8g)
      output_root="$OUT_8G"
      scope_overrides=("${EIGHT_GRANULARITY_OVERRIDES[@]}")
      ;;
    4g)
      output_root="$OUT_4G"
      scope_overrides=("${FOUR_GRANULARITY_OVERRIDES[@]}")
      ;;
    *)
      echo "scope must be 8g or 4g" >&2
      return 2
      ;;
  esac

  if [[ "$run_id" == *"-$BUDGET_RUN_SLUG-$scope-"* ]]; then
    :
  else
    echo "run ID must contain -$BUDGET_RUN_SLUG-$scope-; got: $run_id" >&2
    return 2
  fi

  "${SBATCH[@]}" \
    --time="$WALLTIME" \
    --job-name="$job_name" \
    scripts/slurm_tinystories_controlled.sh \
    --python-bin "$PYTHON_BIN" \
    --config "$BASE" \
    --output-root "$output_root" \
    "${scope_overrides[@]}" \
    --override "run.run_id=$run_id" \
    "$@"
}
```

The launcher streams unbuffered progress to `./logs`. If a job is interrupted,
resubmit the identical helper call and run ID; continuation restores its
checkpoint, scheduler, RNG streams, sampling-policy state, exposure counts,
and packed-corpus cursor. Never submit one run ID concurrently. To select a
different budget, follow the complete section 1.3 sequence before defining
this helper.

## 5. Submit the uniform-window sweep

Run the same `H=1,5,25,50` sweep for both grids:

```bash
for SCOPE in 8g 4g; do
  for H in 1 5 25 50; do
    submit_tinystories_run \
      "$SCOPE" \
      "tiny-$BUDGET_RUN_SLUG-$SCOPE-uniform-h$H$GRADIENT_INTERFERENCE_RUN_SLUG" \
      "tiny-$BUDGET_RUN_SLUG-$SCOPE-uniform-h$H$GRADIENT_INTERFERENCE_RUN_SLUG-s$EXPERIMENT_SEED" \
      "${GRADIENT_INTERFERENCE_OVERRIDES[@]}" \
      --override run.model_family=nested \
      --override run.sampling_mode=nested-random \
      --override run.granularity=null \
      --override model.granularity_sampling_mode=global \
      --override model.global_sampling_schedule=random_with_replacement \
      --override "model.global_sampling_interval_steps=$H"
  done
done
```

`H=1` redraws a uniformly sampled global width after every successful update.
For `H>1`, the selected width is held for `H` updates before an independent
uniform redraw; adjacent windows may select the same width.

## 6. Submit Thompson and PanelGrad comparisons

Submit the same three adaptive policies for both grids:

```bash
for SCOPE in 8g 4g; do
  submit_tinystories_run \
    "$SCOPE" \
    "tiny-$BUDGET_RUN_SLUG-$SCOPE-thompson" \
    "tiny-$BUDGET_RUN_SLUG-$SCOPE-thompson-h25-s$EXPERIMENT_SEED" \
    --override run.model_family=nested \
    --override run.sampling_mode=nested-random \
    --override run.granularity=null \
    --override model.granularity_sampling_mode=adaptive_global \
    --override model.adaptive_sampler_strategy=thompson \
    --override "model.adaptive_controller=$THOMPSON_CONTROLLER"

  submit_tinystories_run \
    "$SCOPE" \
    "tiny-$BUDGET_RUN_SLUG-$SCOPE-panelgrad-rms" \
    "tiny-$BUDGET_RUN_SLUG-$SCOPE-panelgrad-rms-eps0p1-s$EXPERIMENT_SEED" \
    --override run.model_family=nested \
    --override run.sampling_mode=nested-random \
    --override run.granularity=null \
    --override model.granularity_sampling_mode=adaptive_global \
    --override model.adaptive_sampler_strategy=panelgrad \
    --override "model.panelgrad=$PANELGRAD_RMS"

  submit_tinystories_run \
    "$SCOPE" \
    "tiny-$BUDGET_RUN_SLUG-$SCOPE-panelgrad-l2" \
    "tiny-$BUDGET_RUN_SLUG-$SCOPE-panelgrad-l2-eps0p5-to0p1-s$EXPERIMENT_SEED" \
    --override run.model_family=nested \
    --override run.sampling_mode=nested-random \
    --override run.granularity=null \
    --override model.granularity_sampling_mode=adaptive_global \
    --override model.adaptive_sampler_strategy=panelgrad \
    --override "model.panelgrad=$PANELGRAD_L2"
done
```

Compare Thompson and PanelGrad against uniform `H=25` to separate controller
adaptation from simply holding one global width for 25 updates.

## 7. Submit standalone references

Run every standalone width needed for the complete nested-versus-standalone
curves:

```bash
for GRANULARITY in g125 g250 g375 g500 g625 g750 g875 g1000; do
  submit_tinystories_run \
    8g \
    "tiny-$BUDGET_RUN_SLUG-8g-standalone-$GRANULARITY" \
    "tiny-$BUDGET_RUN_SLUG-8g-standalone-$GRANULARITY-s$EXPERIMENT_SEED" \
    --override run.model_family=standalone \
    --override run.sampling_mode=standalone \
    --override "run.granularity=$GRANULARITY"
done

for GRANULARITY in g250 g500 g750 g1000; do
  submit_tinystories_run \
    4g \
    "tiny-$BUDGET_RUN_SLUG-4g-standalone-$GRANULARITY" \
    "tiny-$BUDGET_RUN_SLUG-4g-standalone-$GRANULARITY-s$EXPERIMENT_SEED" \
    --override run.model_family=standalone \
    --override run.sampling_mode=standalone \
    --override "run.granularity=$GRANULARITY"
done
```

The overlapping four-grid standalone jobs are intentionally isolated in the
four-grid campaign root so each report is self-contained. If compute is
constrained, begin with the smallest, middle, and full widths, but all widths
are required for the complete quality-versus-size curve.

## 8. Optional joint and balanced references

Nested-all backpropagates every configured width at each update. It has the
same data and optimizer-step horizon, but substantially more compute per update,
so treat it as a high-compute reference rather than a matched-compute policy:

```bash
for SCOPE in 8g 4g; do
  submit_tinystories_run \
    "$SCOPE" \
    "tiny-$BUDGET_RUN_SLUG-$SCOPE-nested-all" \
    "tiny-$BUDGET_RUN_SLUG-$SCOPE-nested-all-s$EXPERIMENT_SEED" \
    --override run.model_family=nested \
    --override run.sampling_mode=nested-all \
    --override run.granularity=null
done
```

The balanced screen isolates the effect of holding a width from random
selected-label imbalance. The suggested 1x, 2x, 4x, and 8x budgets all let
`H=1` and `H=64` finish complete cycles for four and eight widths. Validate
that property before submitting an arbitrary exact budget:

```bash
if (( DERIVED_MAX_STEPS % (8 * 64) == 0 )); then
  for SCOPE in 8g 4g; do
    for H in 1 64; do
      submit_tinystories_run \
        "$SCOPE" \
        "tiny-$BUDGET_RUN_SLUG-$SCOPE-balanced-h$H$GRADIENT_INTERFERENCE_RUN_SLUG" \
        "tiny-$BUDGET_RUN_SLUG-$SCOPE-balanced-h$H$GRADIENT_INTERFERENCE_RUN_SLUG-s$EXPERIMENT_SEED" \
        "${GRADIENT_INTERFERENCE_OVERRIDES[@]}" \
        --override run.model_family=nested \
        --override run.sampling_mode=nested-random \
        --override run.granularity=null \
        --override model.granularity_sampling_mode=global \
        --override model.global_sampling_schedule=balanced_cycle \
        --override "model.global_sampling_interval_steps=$H"
    done
  done
else
  echo "Balanced H=64 requires DERIVED_MAX_STEPS divisible by 512; nothing submitted" >&2
fi
```

At completion, each selected label must have `DERIVED_MAX_STEPS / 8` updates
in the eight-width runs and `DERIVED_MAX_STEPS / 4` in the four-width runs.
Equal selected-label exposure does not equal equal parameter exposure: the
smaller shared prefixes also participate when a larger width is selected.

## 9. Monitor, resume, and verify

```bash
squeue --me
tail -f "logs/tiny-$BUDGET_RUN_SLUG-8g-uniform-h25_<job-id>.out"
tail -f "logs/tiny-$BUDGET_RUN_SLUG-8g-uniform-h25_<job-id>.err"
```

For a representative completed run:

```bash
export RUN_ID="tiny-$BUDGET_RUN_SLUG-8g-uniform-h25${GRADIENT_INTERFERENCE_RUN_SLUG}-s$EXPERIMENT_SEED"
export GROUP_DIR_8G="$OUT_8G/matformer_llama_2m_$TOKEN_BUDGET_GROUP"
export RUN_DIR="$GROUP_DIR_8G/$RUN_ID"

"$PYTHON_BIN" - "$RUN_DIR" "$TOKEN_BUDGET" "$DERIVED_MAX_STEPS" <<'PY'
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
expected_tokens = int(sys.argv[2])
expected_steps = int(sys.argv[3])
config = json.loads((run_dir / "config.json").read_text())
summary = json.loads((run_dir / "run_summary.json").read_text())

assert config["controlled_experiment"]["recipe_status"] == "frozen"
assert config["training"]["token_budget"] == expected_tokens
assert config["training"]["derived_max_steps"] == expected_steps
assert config["training"]["resolved_learning_rate"] == 3e-3
assert config["training"]["scheduler_name"] == "cosine"
assert config["training"]["resolved_warmup_steps"] == 64
assert config["training"]["effective_world_size"] == 1
assert config["training"]["expected_tokens_per_step"] == 8_192
assert summary["status"] == "completed"
assert summary["tokens_seen"] == expected_tokens
assert not summary.get("unresolved_artifact_failures")

print("verified", summary["run_id"], summary["tokens_seen"], "tokens")
PY
```

Within each width grid, require identical optimizer-training,
ordinary-validation, controller, and final-holdout role hashes before comparing
policies. A run is not comparable if it changes the frozen recipe, seed, data,
batch geometry, or selected token/optimizer-step horizon.

## 10. Evaluate and compare the completed campaigns

Declare the comparison set using ordinary validation before exposing the
sealed final holdout. Then submit final-holdout evaluation for every completed
run that belongs to that declared set:

```bash
export GROUP_DIR_8G="$OUT_8G/matformer_llama_2m_$TOKEN_BUDGET_GROUP"
export GROUP_DIR_4G="$OUT_4G/matformer_llama_2m_$TOKEN_BUDGET_GROUP"

for GROUP_DIR in "$GROUP_DIR_8G" "$GROUP_DIR_4G"; do
  find "$GROUP_DIR" -mindepth 1 -maxdepth 1 -type d -print0 |
    while IFS= read -r -d '' RUN_DIR; do
      [[ -f "$RUN_DIR/run_summary.json" ]] || continue
      [[ -f "$RUN_DIR/final_holdout_results.json" ]] && continue
      STATUS="$("$PYTHON_BIN" -c \
        'import json, pathlib, sys; print(json.loads((pathlib.Path(sys.argv[1]) / "run_summary.json").read_text())["status"])' \
        "$RUN_DIR")"
      [[ "$STATUS" == completed ]] || continue
      RUN_ID="$(basename "$RUN_DIR")"
      "${SBATCH[@]}" \
        --job-name="holdout-$RUN_ID" \
        scripts/slurm_tinystories_controlled.sh \
        --python-bin "$PYTHON_BIN" \
        --final-holdout-only "$RUN_DIR"
    done
done
```

Generate one figure set per width grid:

```bash
"$PYTHON_BIN" scripts/make_figures.py \
  --input "$GROUP_DIR_8G" \
  --output "$GROUP_DIR_8G/figures" \
  --variant slicing \
  --correction none

"$PYTHON_BIN" scripts/make_figures.py \
  --input "$GROUP_DIR_4G" \
  --output "$GROUP_DIR_4G/figures" \
  --variant slicing \
  --correction none
```

Use ordinary validation trajectories to study when gaps appear and the sealed
final holdout for the final reported comparisons. For each width grid, compare:

- elastic `H=1` against `H=5`, `H=25`, and `H=50`;
- `H=25` against Thompson and both PanelGrad policies;
- each width from every elastic policy against its standalone counterpart;
- the same policies on four versus eight granularities;
- optionally, IID against balanced exposure and matched-compute against
  nested-all, with nested-all's extra compute reported explicitly;
- for diagnostic runs, the pairwise gradient-cosine trajectories together
  with their measured backward-evaluation and wall-clock cost.

## 11. One-time full-corpus preparation

Skip this section when both manifests tested in section 1 already exist. The
preparer downloads the pinned `roneneldan/TinyStories` train and validation
splits through the Hugging Face `datasets` interface; no direct dataset URL is
required.

```bash
export HF_HOME="${HF_HOME:-$NFS_USER_ROOT/huggingface}"
mkdir -p "$HF_HOME" "$MATFORMER_TOKENIZER_ROOT" "$MATFORMER_CORPUS_ROOT"

"$PYTHON_BIN" scripts/prepare_tinystories.py \
  --tokenizer-dir "$TOKENIZER" \
  --corpus-dir "$CORPUS" \
  --optimizer-token-count all \
  --tokenization-workers 4 \
  --progress-interval-seconds 60

"$PYTHON_BIN" scripts/audit_prepared_corpus.py \
  --prepared-corpus-dir "$CORPUS" \
  --prepared-tokenizer-dir "$TOKENIZER" \
  --minimum-training-tokens "$TOKEN_BUDGET" \
  --required-vocab-size 2048

"$PYTHON_BIN" - "$CORPUS/corpus_manifest.json" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert manifest["source"]["termination"] == "source_exhausted"
assert manifest["source"]["source_exhausted"] is True
assert "optimizer_token_limit" not in manifest
print(
    "full corpus verified:",
    manifest["available_optimizer_token_count"],
    "optimizer tokens",
)
PY
```

`all` is also the preparer's default. It consumes the source to exhaustion and
writes every complete 128-token optimizer sequence available after the fixed
reserved roles; it does not impose a training-token cap. The new
`tinystories-packed-full-v1` directory keeps the old capped 33M artifact
immutable.

The command emits periodic, unbuffered tokenizer/corpus progress. A matching
completed tokenizer or source-exhausted corpus is reused. A capped corpus is
not accepted as a match for this request. An interrupted corpus preparation
resumes from its preparation checkpoint; rerunning the same command does not
start a valid completed artifact from scratch.

Blank source rows are skipped deterministically while each retained story
keeps its physical split-row identity. The manifested roles are:

- first 128 non-empty train stories: controller;
- all remaining non-empty train stories: optimizer, packed until the source is
  exhausted;
- first 128 non-empty validation stories: ordinary validation;
- next 512 non-empty validation stories: sealed final holdout.

Each role is packed independently with EOS separators. The first 50,000
optimizer-eligible train stories train the tokenizer and remain in the
optimizer corpus; tokenizer preparation introduces no extra source documents.

## 12. Completed recipe-selection process

The frozen recipe came from six independent dense full-width runs over
`learning_rate in {3e-4, 1e-3, 3e-3}` and `scheduler in {cosine,
constant_with_warmup}` at 16,777,216 tokens (2,048 updates). Ordinary
validation did not establish convergence for the globally best stable run, so
the two best fallback candidates were rerun from initialization at 33,554,432
tokens (4,096 updates), with the cosine schedule refit to the longer horizon.

The analyzer selected:

- run: `tinystories-dense-lr3e-3-schedcosine-4096-s42`;
- best/final ordinary-validation loss: `1.832638272675135` at step 4,096;
- recipe: learning rate `3e-3`, cosine decay, 33,554,432 tokens;
- selection report hash:
  `ecf84f1131b57255e945c10b51599bd1e84dfc872d48e65bc7a4818fe92c1c69`.

That provenance is recorded in the frozen config. The calibration does not
need to be rerun for the comparison campaign. For audit or reconstruction, the
original grid can be expressed from the frozen config by explicitly restoring
the calibration fields. Exact reconstruction uses the immutable capped corpus
from that historical campaign, not the new source-exhausted corpus:

```bash
export TINYSTORIES_CALIBRATION_NAME="${TINYSTORIES_CALIBRATION_NAME:-tinystories-controlled-convergence-v1}"
export TINYSTORIES_CALIBRATION_CORPUS_NAME="${TINYSTORIES_CALIBRATION_CORPUS_NAME:-tinystories-packed-33m-v1}"
export CALIBRATION_CORPUS="$MATFORMER_CORPUS_ROOT/$TINYSTORIES_CALIBRATION_CORPUS_NAME"
export CALIBRATION_OUT="$MATFORMER_EXPERIMENT_ROOT/$TINYSTORIES_CALIBRATION_NAME"
mkdir -p "$CALIBRATION_OUT"
test -r "$CALIBRATION_CORPUS/corpus_manifest.json"

CALIBRATION_OVERRIDES=(
  "${COMMON_OVERRIDES[@]}"
  --override "run.output_root=$CALIBRATION_OUT"
  --override "dataset.prepared_corpus_dir=$CALIBRATION_CORPUS"
  --override controlled_experiment.recipe_status=calibration
  --override controlled_experiment.recipe_source_run_id=null
  --override controlled_experiment.selection_report_hash=null
  --override training.token_budget=16777216
)

for LR in 3e-4 1e-3 3e-3; do
  for SCHEDULER in cosine constant_with_warmup; do
    RUN_ID="tinystories-dense-lr${LR}-sched${SCHEDULER}-2048-s$EXPERIMENT_SEED"
    "${SBATCH[@]}" \
      --job-name="tiny-cal-${LR}-${SCHEDULER}" \
      scripts/slurm_tinystories_controlled.sh \
      --python-bin "$PYTHON_BIN" \
      --config "$BASE" \
      "${CALIBRATION_OVERRIDES[@]}" \
      --override "run.run_id=$RUN_ID" \
      --override "training.learning_rate=$LR" \
      --override "training.scheduler.name=$SCHEDULER"
  done
done
```

Each run has its own checkpoint and exact-resume state. Analyze only ordinary
validation:

```bash
"$PYTHON_BIN" scripts/analyze_tinystories_convergence.py \
  --runs-root "$CALIBRATION_OUT" \
  --output-dir "$CALIBRATION_OUT/convergence-analysis"
```

The selection contract ignores evaluations before step 512, requires five
final evaluations without a new 0.5% relative best, and uses a 0.1% relative
loss tie before wall time. The sealed final holdout does not participate in
recipe selection.
