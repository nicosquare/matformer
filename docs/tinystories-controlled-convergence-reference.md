# TinyStories controlled-convergence historical and optional reference

> Historical/optional material preserved from the broad campaign guide. The
> primary plateau-to-catch-up workflow now lives in
> `docs/tinystories-controlled-convergence-experiments.md`.
> To reproduce the historical broad d128 campaign after selecting the Instruct
> profile, explicitly set
> `BASE=configs/controlled_exps/tinystories_controlled_convergence.yaml`; the
> current profile selector intentionally points Instruct at the d64/l4 plateau
> calibration config.

This guide supports two deliberately separate data profiles:

- `stories` reproduces standalone-width and elastic comparisons with the
  frozen ordinary TinyStories recipe in
  `configs/controlled_exps/tinystories_controlled_convergence.yaml`;
- `instruct` searches for a new model capacity and training recipe on the
  separately prepared TinyStories-Instruct corpus and tokenizer.

Choose the profile once in section 1.1. The selector binds the dataset,
tokenizer, corpus, provenance fields, and output identity as one unit. Do not
mix individual paths between profiles. The ordinary TinyStories recipe uses
learning rate `3e-3`, cosine decay, 64 warmup updates, and 33,554,432 tokens
over 4,096 one-GPU optimizer updates. Those values are only starting
candidates—not frozen evidence—for `instruct`.

The immutable tokenizers and packed corpora are prerequisites, not routine
steps. Section 11 documents their one-time preparation. Section 12 defines the
capacity-aware selection needed to find a unique-data plateau before testing
elastic catch-up. On a first setup, run the profile selector in section 1.1,
prepare its artifacts in the matching section 11 subsection, and then run the
rest of section 1.

The main comparison asks how elastic training changes each width relative to
independent standalone training. It covers the default eight-width grid and a
matched four-width subset, uniform global sampling with several hold windows,
Thompson sampling, PanelGrad, and optional joint-training and balanced-exposure
references. The uniform runs can also opt into the same fixed-probe
gradient-interference diagnostic used by the larger setup.

## 1. Set the shared contract

### 1.1 Choose the dataset and tokenizer profile

Run from the repository root, activate the validated environment, and select
exactly one profile. For the new recipe search, use `instruct`:

```bash
conda activate elasticnn
export TINYSTORIES_PROFILE=instruct  # use stories only for the frozen baseline
source scripts/select_tinystories_profile.sh
```

This is the only dataset/tokenizer choice in the workflow. Switching profiles
requires rerunning all of section 1 so no shell arrays retain the old profile.
The selector must be sourced—not executed—because the later commands use its
exported contract. It prints the resolved dataset, tokenizer, and corpus for a
visual check.

### 1.2 Choose the budget and optional learning rate

Choose exactly one horizon before defining any policy or submission arrays:

| Horizon | Exact tokens | Optimizer updates | Command |
| --- | ---: | ---: | --- |
| 1x (default) | 33,554,432 | 4,096 | `export TOKEN_BUDGET=33554432` |
| 2x | 67,108,864 | 8,192 | `export TOKEN_BUDGET=67108864` |
| 4x | 134,217,728 | 16,384 | `export TOKEN_BUDGET=134217728` |
| 8x | 268,435,456 | 32,768 | `export TOKEN_BUDGET=268435456` |

For the 2x campaign requested here, start with:

```bash
export TOKEN_BUDGET=67108864
export LEARNING_RATE=0.003
```

For `stories`, `0.003` is the selected frozen learning rate. For `instruct`, it
is the initial calibration candidate. To test another learning rate while
keeping the same token horizon, change only `LEARNING_RATE`, for example:

```bash
export TOKEN_BUDGET=268435456
export LEARNING_RATE=0.0045
```

If either value is unset, setup falls back to the 1x horizon or `0.003`,
respectively. Export both explicitly when switching campaigns.

### 1.3 Resolve and validate the selection

After choosing the budget and learning rate, run this entire block:

```bash

tinystories_setup() {
  export EXPERIMENT_SEED=42
  export WALLTIME="00:30:00"

  if [[ -x "${PYTHON_BIN:-}" ]]; then
    :
  else
    echo "Run the profile selector in section 1.1 first" >&2
    return 2
  fi
  if [[ -n "${TINYSTORIES_PROFILE:-}" && -n "${TOKENIZER:-}" && \
        -n "${CORPUS:-}" && -n "${DATASET_NAME:-}" ]]; then
    :
  else
    echo "The dataset/tokenizer profile is incomplete; rerun section 1.1" >&2
    return 2
  fi

  # Use the historical starting values only when none were selected above.
  export TOKEN_BUDGET="${TOKEN_BUDGET:-33554432}"
  export FROZEN_LEARNING_RATE=0.003
  export LEARNING_RATE="${LEARNING_RATE:-$FROZEN_LEARNING_RATE}"
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

  if LEARNING_RATE_CONTRACT="$("$PYTHON_BIN" -c \
    'import math, sys; value = float(sys.argv[1]); (math.isfinite(value) and value > 0.0) or sys.exit("LEARNING_RATE must be finite and positive"); canonical = format(value, ".12g"); print(canonical, "lr" + canonical.replace("-", "m").replace("+", "p").replace(".", "p"))' \
    "$LEARNING_RATE")"; then
    read -r LEARNING_RATE LEARNING_RATE_SLUG <<<"$LEARNING_RATE_CONTRACT"
    export LEARNING_RATE LEARNING_RATE_SLUG
  else
    return 2
  fi

  export LEARNING_RATE_OUTPUT_SUFFIX=""
  export SELECTED_RECIPE_STATUS="$PROFILE_RECIPE_STATUS"
  if [[ "$SELECTED_RECIPE_STATUS" == calibration || \
        "$LEARNING_RATE" != "$FROZEN_LEARNING_RATE" ]]; then
    export LEARNING_RATE_OUTPUT_SUFFIX="-$LEARNING_RATE_SLUG"
    export SELECTED_RECIPE_STATUS=calibration
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

  export OUT_8G="$MATFORMER_EXPERIMENT_ROOT/${EXPERIMENT_NAME}${LEARNING_RATE_OUTPUT_SUFFIX}-${BUDGET_RUN_SLUG}-8g"
  export OUT_4G="$MATFORMER_EXPERIMENT_ROOT/${EXPERIMENT_NAME}${LEARNING_RATE_OUTPUT_SUFFIX}-${BUDGET_RUN_SLUG}-4g"
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
    'import json, pathlib, sys; m=json.loads((pathlib.Path(sys.argv[1]) / "corpus_manifest.json").read_text()); s=m["source"]; t=m["tokenizer"]; expected=dict(zip(("dataset_name", "dataset_config_name", "split"), sys.argv[2:5])); bad=[key for key, value in expected.items() if s.get(key) != value]; bad += (["tokenizer"] if t.get("name") != sys.argv[5] else []); bad and sys.exit("Selected profile does not match corpus manifest: " + ", ".join(bad)); print(m["available_optimizer_token_count"], s["termination"])' \
    "$CORPUS" "$DATASET_NAME" "$DATASET_CONFIG_NAME" "$DATASET_SPLIT" \
    "$TOKENIZER_NAME")"; then
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

  printf 'profile=%s tokens=%s optimizer_steps=%s learning_rate=%s recipe_status=%s corpus_capacity=%s max_aligned_tokens=%s max_aligned_steps=%s\n' \
    "$TINYSTORIES_PROFILE" "$TOKEN_BUDGET" "$DERIVED_MAX_STEPS" "$LEARNING_RATE" \
    "$SELECTED_RECIPE_STATUS" "$AVAILABLE_CORPUS_TOKENS" \
    "$MAX_ALIGNED_TOKEN_BUDGET" "$MAX_ALIGNED_STEPS"
}

tinystories_setup
```

For the `instruct` 2x calibration, do not continue unless the final line starts
with:

```text
profile=instruct tokens=67108864 optimizer_steps=8192 learning_rate=0.003 recipe_status=calibration
```

The `stories` profile reports `recipe_status=frozen` only at `0.003`; every
`instruct` run is a calibration until section 12 freezes a selected recipe.
Calibration output roots include the profile and learning-rate slug, preventing
either profile or learning rate from resuming another run.

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
DATA_PROFILE_OVERRIDES=(
  --override "run.phase_id=$EXPERIMENT_PHASE"
  --override "run.model_shape_label=${PROFILE_SLUG}-d128-l4"
  --override "model.tokenizer_dir=$TOKENIZER"
  --override "dataset.prepared_corpus_dir=$CORPUS"
  --override "dataset.dataset_name=$DATASET_NAME"
  --override "dataset.dataset_config_name=$DATASET_CONFIG_NAME"
  --override "dataset.dataset_split=$DATASET_SPLIT"
  --override "dataset.dataset_phase=$DATASET_PHASE"
  --override "dataset.preprocessing_notes=$PREPROCESSING_NOTES"
  --override "monitoring.project=$EXPERIMENT_PHASE"
)

COMMON_OVERRIDES=(
  --override "run.seed=$EXPERIMENT_SEED"
  "${DATA_PROFILE_OVERRIDES[@]}"
  --override "training.token_budget=$TOKEN_BUDGET"
  --override "training.learning_rate=$LEARNING_RATE"
)

if [[ "$SELECTED_RECIPE_STATUS" == calibration ]]; then
  COMMON_OVERRIDES+=(
    --override controlled_experiment.recipe_status=calibration
    --override controlled_experiment.recipe_source_run_id=null
    --override controlled_experiment.selection_report_hash=null
  )
fi

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

### 1.4 Required sequence after changing the profile, budget, or learning rate

Changing the profile, `TOKEN_BUDGET`, or `LEARNING_RATE` in an existing shell
does not rewrite arrays that were already defined. Use this order every time:

1. Run the profile selector in section 1.1.
2. Export `TOKEN_BUDGET` and `LEARNING_RATE` in section 1.2.
3. Run `tinystories_setup` from section 1.3 and verify its profile, tokens,
   optimizer steps, learning rate, and recipe status.
4. Rerun the `COMMON_OVERRIDES`, granularity, and `SBATCH` block immediately
   above so it captures the new budget, learning rate, and output roots.
5. Rerun all of section 2 so budget-dependent policies and diagnostic arrays
   are rebuilt.
6. Run the section 3 preflights, redefine the section 4 submission helper, and
   only then submit a campaign from section 5 onward.

The resulting FFN prefix widths are `64, 128, 192, 256, 320, 384, 448,
512` for eight granularities and `128, 256, 384, 512` for four. Both scopes
retain the same data roles, initialization seed, optimizer, scheduler, selected
token budget, learning rate, and validation cadence. Changing `TOKEN_BUDGET`
creates a new cosine schedule and budget-isolated output roots. Changing
`LEARNING_RATE` creates learning-rate-isolated output roots. Changing the
profile changes the dataset, tokenizer, provenance, phase, run IDs, and output
roots together. Every such run starts from initialization rather than resuming
an incompatible completed run.

## 2. Define the adaptive policies

Run this section after section 1 every time the profile, budget, or learning
rate changes.
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
  --override "run.run_id=preflight-${RUN_PREFIX}-${BUDGET_RUN_SLUG}-8g-uniform-h25${GRADIENT_INTERFERENCE_RUN_SLUG}-s$EXPERIMENT_SEED" \
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
  --override "run.run_id=preflight-${RUN_PREFIX}-${BUDGET_RUN_SLUG}-4g-uniform-h25${GRADIENT_INTERFERENCE_RUN_SLUG}-s$EXPERIMENT_SEED" \
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
  --override "run.run_id=preflight-${RUN_PREFIX}-${BUDGET_RUN_SLUG}-8g-thompson-s$EXPERIMENT_SEED" \
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
  --override "run.run_id=preflight-${RUN_PREFIX}-${BUDGET_RUN_SLUG}-8g-panelgrad-l2-s$EXPERIMENT_SEED" \
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
the selected `LEARNING_RATE`, and the expected ordered width grid. It must also
preserve the cosine scheduler. The PanelGrad L2 epsilon duration must equal
`DERIVED_MAX_STEPS`. A non-default learning rate must resolve as a calibration
run under the learning-rate-suffixed output root printed by setup.

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
different profile, budget, or learning rate, follow the complete section 1.4 sequence
before defining this helper.

## 5. Submit the uniform-window sweep

Run the same `H=1,5,25,50` sweep for both grids:

```bash
for SCOPE in 8g 4g; do
  for H in 1 5 25 50; do
    submit_tinystories_run \
      "$SCOPE" \
      "$RUN_PREFIX-$BUDGET_RUN_SLUG-$SCOPE-uniform-h$H$GRADIENT_INTERFERENCE_RUN_SLUG" \
      "$RUN_PREFIX-$BUDGET_RUN_SLUG-$SCOPE-uniform-h$H$GRADIENT_INTERFERENCE_RUN_SLUG-s$EXPERIMENT_SEED" \
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
    "$RUN_PREFIX-$BUDGET_RUN_SLUG-$SCOPE-thompson" \
    "$RUN_PREFIX-$BUDGET_RUN_SLUG-$SCOPE-thompson-h25-s$EXPERIMENT_SEED" \
    --override run.model_family=nested \
    --override run.sampling_mode=nested-random \
    --override run.granularity=null \
    --override model.granularity_sampling_mode=adaptive_global \
    --override model.adaptive_sampler_strategy=thompson \
    --override "model.adaptive_controller=$THOMPSON_CONTROLLER"

  submit_tinystories_run \
    "$SCOPE" \
    "$RUN_PREFIX-$BUDGET_RUN_SLUG-$SCOPE-panelgrad-rms" \
    "$RUN_PREFIX-$BUDGET_RUN_SLUG-$SCOPE-panelgrad-rms-eps0p1-s$EXPERIMENT_SEED" \
    --override run.model_family=nested \
    --override run.sampling_mode=nested-random \
    --override run.granularity=null \
    --override model.granularity_sampling_mode=adaptive_global \
    --override model.adaptive_sampler_strategy=panelgrad \
    --override "model.panelgrad=$PANELGRAD_RMS"

  submit_tinystories_run \
    "$SCOPE" \
    "$RUN_PREFIX-$BUDGET_RUN_SLUG-$SCOPE-panelgrad-l2" \
    "$RUN_PREFIX-$BUDGET_RUN_SLUG-$SCOPE-panelgrad-l2-eps0p5-to0p1-s$EXPERIMENT_SEED" \
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
    "$RUN_PREFIX-$BUDGET_RUN_SLUG-8g-standalone-$GRANULARITY" \
    "$RUN_PREFIX-$BUDGET_RUN_SLUG-8g-standalone-$GRANULARITY-s$EXPERIMENT_SEED" \
    --override run.model_family=standalone \
    --override run.sampling_mode=standalone \
    --override "run.granularity=$GRANULARITY"
done

for GRANULARITY in g250 g500 g750 g1000; do
  submit_tinystories_run \
    4g \
    "$RUN_PREFIX-$BUDGET_RUN_SLUG-4g-standalone-$GRANULARITY" \
    "$RUN_PREFIX-$BUDGET_RUN_SLUG-4g-standalone-$GRANULARITY-s$EXPERIMENT_SEED" \
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
    "$RUN_PREFIX-$BUDGET_RUN_SLUG-$SCOPE-nested-all" \
    "$RUN_PREFIX-$BUDGET_RUN_SLUG-$SCOPE-nested-all-s$EXPERIMENT_SEED" \
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
        "$RUN_PREFIX-$BUDGET_RUN_SLUG-$SCOPE-balanced-h$H$GRADIENT_INTERFERENCE_RUN_SLUG" \
        "$RUN_PREFIX-$BUDGET_RUN_SLUG-$SCOPE-balanced-h$H$GRADIENT_INTERFERENCE_RUN_SLUG-s$EXPERIMENT_SEED" \
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
tail -f "logs/$RUN_PREFIX-$BUDGET_RUN_SLUG-8g-uniform-h25_<job-id>.out"
tail -f "logs/$RUN_PREFIX-$BUDGET_RUN_SLUG-8g-uniform-h25_<job-id>.err"
```

For a representative completed run:

```bash
export RUN_ID="$RUN_PREFIX-$BUDGET_RUN_SLUG-8g-uniform-h25${GRADIENT_INTERFERENCE_RUN_SLUG}-s$EXPERIMENT_SEED"
export GROUP_DIR_8G="$OUT_8G/matformer_llama_2m_$TOKEN_BUDGET_GROUP"
export RUN_DIR="$GROUP_DIR_8G/$RUN_ID"

"$PYTHON_BIN" - "$RUN_DIR" "$TOKEN_BUDGET" "$DERIVED_MAX_STEPS" \
  "$LEARNING_RATE" "$SELECTED_RECIPE_STATUS" <<'PY'
import json
import math
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
expected_tokens = int(sys.argv[2])
expected_steps = int(sys.argv[3])
expected_learning_rate = float(sys.argv[4])
expected_recipe_status = sys.argv[5]
config = json.loads((run_dir / "config.json").read_text())
summary = json.loads((run_dir / "run_summary.json").read_text())

assert config["controlled_experiment"]["recipe_status"] == expected_recipe_status
assert config["training"]["token_budget"] == expected_tokens
assert config["training"]["derived_max_steps"] == expected_steps
assert math.isclose(
    config["training"]["resolved_learning_rate"],
    expected_learning_rate,
    rel_tol=0.0,
    abs_tol=1e-15,
)
assert config["training"]["scheduler_name"] == "cosine"
assert config["training"]["resolved_warmup_steps"] == 64
assert config["training"]["effective_world_size"] == 1
assert config["training"]["expected_tokens_per_step"] == 8_192
assert summary["status"] == "completed"
assert summary["tokens_seen"] == expected_tokens
assert not summary.get("unresolved_artifact_failures")

print(
    "verified",
    summary["run_id"],
    summary["tokens_seen"],
    "tokens at learning rate",
    expected_learning_rate,
)
PY
```

Within each width grid, require identical optimizer-training,
ordinary-validation, controller, and final-holdout role hashes before comparing
policies. Compare only runs from the same learning-rate-isolated root. A run is
not comparable if it changes the selected recipe, seed, data, batch geometry,
learning rate, or token/optimizer-step horizon within that comparison set.

## 10. Evaluate and compare the completed campaigns

Declare the comparison set using ordinary validation before exposing the
sealed final holdout. Then collect every completed run in that declared set and
submit one finalization allocation. The allocation evaluates the runs
sequentially, so the entire final holdout consumes one Slurm job slot rather
than one slot per run:

```bash
export GROUP_DIR_8G="$OUT_8G/matformer_llama_2m_$TOKEN_BUDGET_GROUP"
export GROUP_DIR_4G="$OUT_4G/matformer_llama_2m_$TOKEN_BUDGET_GROUP"
export FINALIZATION_WALLTIME="${FINALIZATION_WALLTIME:-02:00:00}"

FINAL_HOLDOUT_ARGS=()

for GROUP_DIR in "$GROUP_DIR_8G" "$GROUP_DIR_4G"; do
  if [[ -d "$GROUP_DIR" ]]; then
    while IFS= read -r -d '' RUN_DIR; do
      [[ -f "$RUN_DIR/run_summary.json" ]] || continue
      STATUS="$("$PYTHON_BIN" -c \
        'import json, pathlib, sys; print(json.loads((pathlib.Path(sys.argv[1]) / "run_summary.json").read_text())["status"])' \
        "$RUN_DIR")"
      [[ "$STATUS" == completed ]] || continue
      FINAL_HOLDOUT_ARGS+=(--final-holdout-only "$RUN_DIR")
    done < <(find "$GROUP_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
  fi
done

if (( ${#FINAL_HOLDOUT_ARGS[@]} > 0 )); then
  "${SBATCH[@]}" \
    --time="$FINALIZATION_WALLTIME" \
    --job-name="$RUN_PREFIX-$BUDGET_RUN_SLUG-final-holdouts" \
    scripts/slurm_tinystories_controlled.sh \
    --python-bin "$PYTHON_BIN" \
    "${FINAL_HOLDOUT_ARGS[@]}"
else
  echo "No completed runs are waiting for final-holdout evaluation"
fi
```

The finalization launcher uses `--skip-existing` for each run. If one run
fails or the allocation reaches its wall-time limit, rerun this block: existing
valid results are skipped and only unfinished runs consume evaluation time.

Wait for the single finalization job to complete successfully, then generate
one figure set per width grid:

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

`make_figures.py` now keeps the two metric sources separate:

- `ppl_vs_size*.png` and the validation trajectories use ordinary validation;
- `final_holdout_ppl_vs_size.png` uses only `final_holdout_results.json`, with
  perplexity on the primary axis and loss on the secondary axis.

The final-holdout figures are all-or-nothing. Figure generation fails clearly
if a completed holdout-enabled run is missing its result, if a result is
malformed, or if the selected runs mix data-role hashes, holdout hashes, token
budgets, seeds, or nested granularity grids. This prevents a partial final plot
from being mistaken for the declared comparison set.

Use ordinary validation trajectories to study when gaps appear and use the
sealed final-holdout figures for the final reported comparisons. For each width
grid, compare:

- elastic `H=1` against `H=5`, `H=25`, and `H=50`;
- `H=25` against Thompson and both PanelGrad policies;
- each width from every elastic policy against its standalone counterpart;
- the same policies on four versus eight granularities;
- optionally, IID against balanced exposure and matched-compute against
  nested-all, with nested-all's extra compute reported explicitly;
- for diagnostic runs, the pairwise gradient-cosine trajectories together
  with their measured backward-evaluation and wall-clock cost.

## 11. One-time full-corpus preparation

Run section 1.1 first, then run only the subsection matching the selected
profile. Both subsections use the `TOKENIZER` and `CORPUS` paths bound by that
selector.

### 11.1 TinyStories

Use this subsection only for `TINYSTORIES_PROFILE=stories` and skip it when
both manifests tested in section 1 already exist. The
preparer downloads the pinned `roneneldan/TinyStories` train and validation
splits through the Hugging Face `datasets` interface; no direct dataset URL is
required.

```bash
if [[ "${TINYSTORIES_PROFILE:-}" != stories ]]; then
  echo "Section 11.1 requires TINYSTORIES_PROFILE=stories" >&2
else
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
    --minimum-training-tokens 1 \
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
fi
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

### 11.2 TinyStories-Instruct packed artifacts

Use this subsection only for `TINYSTORIES_PROFILE=instruct`.
TinyStories-Instruct has a separate tokenizer and corpus contract. The command
uses the paths selected in section 1.1, keeps Hugging Face downloads under the
same NFS root, and leaves the TinyStories artifacts above untouched:

```bash
set -euo pipefail

if [[ "${TINYSTORIES_PROFILE:-}" != instruct ]]; then
  echo "Section 11.2 requires TINYSTORIES_PROFILE=instruct" >&2
else

export HF_HOME="${HF_HOME:-$NFS_USER_ROOT/huggingface}"
export TINYSTORIES_FULL_CORPUS="$MATFORMER_CORPUS_ROOT/tinystories-packed-full-v1"

mkdir -p \
  "$HF_HOME" \
  "$MATFORMER_TOKENIZER_ROOT" \
  "$MATFORMER_CORPUS_ROOT" \
  ./logs

"$PYTHON_BIN" scripts/prepare_tinystories_instruct.py \
  --tokenizer-dir "$TOKENIZER" \
  --corpus-dir "$CORPUS" \
  --optimizer-token-count all \
  --tokenization-workers 4 \
  --progress-interval-seconds 60

"$PYTHON_BIN" scripts/audit_prepared_corpus.py \
  --prepared-corpus-dir "$CORPUS" \
  --prepared-tokenizer-dir "$TOKENIZER" \
  --minimum-training-tokens 1 \
  --required-vocab-size 2048

"$PYTHON_BIN" - \
  "$TOKENIZER/tokenizer_manifest.json" \
  "$CORPUS/corpus_manifest.json" \
  "$TINYSTORIES_FULL_CORPUS/corpus_manifest.json" <<'PY'
import json
import pathlib
import sys

tokenizer = json.loads(pathlib.Path(sys.argv[1]).read_text())
instruct = json.loads(pathlib.Path(sys.argv[2]).read_text())
tinystories = json.loads(pathlib.Path(sys.argv[3]).read_text())

assert tokenizer["tokenizer_name"] == (
    "tinystories-instruct-sentencepiece-bpe-2k-v1"
)
assert tokenizer["vocab_size"] == 2_048
assert instruct["corpus_name"] == "tinystories-instruct-packed-full-v1"
assert instruct["source"]["termination"] == "source_exhausted"
assert instruct["source"]["source_exhausted"] is True
assert "optimizer_token_limit" not in instruct
assert instruct["reserved_role_counts"] == {
    "ordinary_validation": 128,
    "controller": 128,
    "final_holdout": 512,
}
assert all(
    count == 0
    for count in instruct["reserved_pairwise_intersection_counts"].values()
)
assert instruct["available_optimizer_token_count"] > (
    tinystories["available_optimizer_token_count"]
)

tokens_per_optimizer_step = 8_192
available = instruct["available_optimizer_token_count"]
maximum_aligned_budget = available // tokens_per_optimizer_step
maximum_aligned_budget *= tokens_per_optimizer_step
assert maximum_aligned_budget > 0
assert maximum_aligned_budget <= available
assert available - maximum_aligned_budget < tokens_per_optimizer_step
print(
    "TinyStories-Instruct verified:",
    f"available_optimizer_tokens={available}",
    f"maximum_aligned_budget={maximum_aligned_budget}",
    f"maximum_aligned_steps={maximum_aligned_budget // tokens_per_optimizer_step}",
)
PY
fi
```

The preparer is pinned to `roneneldan/TinyStoriesInstruct` revision
`ee050ed1f8720795be342921335e821856a2b42e`. It assembles physical `datasets`
rows through an exact `<|endoftext|>` delimiter, removes that source delimiter,
and then lets the packer add one EOS. It preserves the complete instruction
record—including `Features`, `Words`, `Summary`, `Story`, blank lines, and
internal newlines—rather than extracting the story alone. Roles count assembled
records, not physical rows: the first 128 train records are controller data,
the remaining train records are optimizer data, and validation records 0-127
and 128-639 are ordinary validation and sealed final holdout, respectively.
The first 50,000 optimizer records train the new 2,048-piece tokenizer and also
remain in the optimizer corpus.

The pinned train blob itself ends with one malformed 11-row fragment at
physical rows `21755670-21755680`; its final row stops mid-word and it has no
delimiter. The adapter emits a warning and excludes only that exact known range
and content hash. It still rejects every other nonblank unterminated fragment,
so the packed source consists exclusively of complete delimited records. This
exception does not invalidate or restart a preparation checkpoint created by
the same command.

`all` is the default and must report `source_exhausted=true`; it packs every
complete optimizer record once, without a token cap or source repetition. The
verification block also reports the largest budget aligned to the experiment's
8,192-token optimizer-step geometry and confirms that this corpus has more
optimizer tokens than the full TinyStories corpus.

Preparation is safe to resume by rerunning the identical command. During an
interrupted build it replays assembled source records to the committed
checkpoint with visible `resume_replay` progress, then continues packing. After
completion, rerun the same command once more and require its JSON summary to
report `tokenizer_status=already_prepared` and
`corpus_status=already_prepared`; matching artifacts are checksum-verified and
reused rather than rebuilt. A changed dataset revision, parser/content policy,
tokenizer identity, shard geometry, or optimizer-token request is rejected as
incompatible with either partial or completed artifacts.

Do not run `scripts/prepare_tinystories.py` with only `--dataset` changed. That
command intentionally retains the ordinary TinyStories one-row-per-story
adapter and tokenizer identity, so overriding only its dataset would split
TinyStories-Instruct records across physical lines and violate this contract.
Use `scripts/prepare_tinystories_instruct.py` exactly as shown above.
The `set -euo pipefail` at the start of the command block also prevents the
audit and manifest checks from running after a failed preparation.

## 12. Capacity-aware recipe reselection

For the new dataset, complete section 1 with
`TINYSTORIES_PROFILE=instruct` before running this section. Every command below
uses that selected profile; the profile slug isolates all calibration output
and run identities. Use `stories` only when intentionally repeating the
ordinary TinyStories capacity study.

The original d128 recipe is a valid historical baseline, but the 8x to 12x
results show that its standalone full-width model is still improving. More
elastic training therefore does not yet answer the intended catch-up question:
the standalone target is moving at the same time. Use this section to select a
smaller model whose standalone target plateaus within one pass over the packed
corpus, then freeze that target before increasing the elastic budget.

Catch-up is a hypothesis, not an assumption. A shared nested tensor may retain
a positive gap from independently optimized standalone tensors even at a long
horizon. The experiment must permit either conclusion: catch-up within a
measured budget, or no catch-up within the available unique-data budget.

### 12.1 Prospective selection contract

Use only ordinary validation for every decision in this section. Although the
pipeline materializes final-holdout results when a run completes, do not inspect
or use them until the model size, standalone recipes, elastic recipes, budgets,
and catch-up rule have all been declared.

The operational plateau tolerance is a 0.5% perplexity change:

```text
PLATEAU_LOSS_TOLERANCE = log(1.005) = 0.0049875415
```

For two successive budgets, let `improvement = earlier_best_validation_loss -
later_best_validation_loss`. A model is a plateau candidate only when the runs
are stable, `0 <= improvement < PLATEAU_LOSS_TOLERANCE`, and the same condition
holds for the next budget interval. A worse later result is not automatically
a plateau; inspect stability and repeat it before selection. Five trailing
ordinary-validation evaluations without a new 0.5% relative best remain useful
within-run evidence, but do not replace this cross-budget test.

Keep the following fixed during the capacity screen:

- four decoder layers, four attention heads, context length 128, vocabulary
  size 2,048, seed 42, optimizer, cosine schedule, warmup, and batch geometry;
- learning rate `0.003`, so model capacity is the only intentional change;
- the full, source-exhausted packed corpus, without wrapping or repeating it;
- standalone `g1000` training only.

Screen these shapes in this order:

| Shape | `d_model` | Full FFN width | Total parameters | Non-embedding parameters | Eight-grid FFN widths |
| --- | ---: | ---: | ---: | ---: | --- |
| d96/l4 (preferred) | 96 | 384 | 983,904 | 590,688 | 48, 96, 144, 192, 240, 288, 336, 384 |
| d64/l4 (fallback) | 64 | 256 | 524,864 | 262,720 | 32, 64, 96, 128, 160, 192, 224, 256 |

The existing d128/l4 full model has 1,574,016 total parameters and 1,049,728
non-embedding parameters. Prefer d96 if both candidates plateau: it changes
capacity less and preserves a stronger standalone target.

### 12.2 Submit the four-job capacity screen

First run sections 1.1-1.3 so `PYTHON_BIN`, `TOKENIZER`, `CORPUS`,
`MATFORMER_EXPERIMENT_ROOT`, `SBATCH`, and the artifact checks are available.
Then define this isolated calibration root and helper:

```bash
export CAPACITY_SELECTION_NAME="${PROFILE_SLUG}-capacity-selection-v1"
export CAPACITY_SELECTION_ROOT="$MATFORMER_EXPERIMENT_ROOT/$CAPACITY_SELECTION_NAME"
mkdir -p "$CAPACITY_SELECTION_ROOT"

submit_capacity_candidate() {
  local d_model="$1"
  local token_budget="$2"
  local learning_rate="$3"
  local num_layers="${4:-4}"
  local granularity="${5:-g1000}"
  local steps
  local lr_slug
  local shape
  local output_root
  local run_id

  if [[ "$d_model" == 96 || "$d_model" == 64 ]]; then
    :
  else
    echo "d_model must be 96 or 64" >&2
    return 2
  fi
  if [[ "$num_layers" == 4 || "$num_layers" == 3 ]]; then
    :
  else
    echo "num_layers must be 4 or 3" >&2
    return 2
  fi
  if [[ "$token_budget" =~ ^[0-9]+$ ]] && \
     (( token_budget > 0 && token_budget % TOKENS_PER_STEP == 0 )); then
    :
  else
    echo "token budget must be positive and divisible by $TOKENS_PER_STEP" >&2
    return 2
  fi
  if (( token_budget <= MAX_ALIGNED_TOKEN_BUDGET )); then
    :
  else
    echo "token budget exceeds the unique-data limit: $MAX_ALIGNED_TOKEN_BUDGET" >&2
    return 2
  fi

  steps=$((token_budget / TOKENS_PER_STEP))
  if lr_slug="$("$PYTHON_BIN" -c \
    'import math, sys; v=float(sys.argv[1]); (math.isfinite(v) and v > 0) or sys.exit(2); print("lr" + format(v, ".12g").replace("-", "m").replace("+", "p").replace(".", "p"))' \
    "$learning_rate")"; then
    :
  else
    echo "learning rate must be finite and positive" >&2
    return 2
  fi

  shape="${PROFILE_SLUG}-d${d_model}-l${num_layers}"
  output_root="$CAPACITY_SELECTION_ROOT/${shape}-${steps}steps-${lr_slug}"
  run_id="capacity-${shape}-${steps}steps-${granularity}-${lr_slug}-s${EXPERIMENT_SEED}"
  mkdir -p "$output_root"

  "${SBATCH[@]}" \
    --time="$WALLTIME" \
    --job-name="$RUN_PREFIX-cap-d${d_model}-l${num_layers}-${steps}-${lr_slug}" \
    scripts/slurm_tinystories_controlled.sh \
    --python-bin "$PYTHON_BIN" \
    --config "$BASE" \
    --output-root "$output_root" \
    "${DATA_PROFILE_OVERRIDES[@]}" \
    --override "run.run_id=$run_id" \
    --override "run.seed=$EXPERIMENT_SEED" \
    --override "run.model_shape_label=$shape" \
    --override run.model_family=standalone \
    --override run.sampling_mode=standalone \
    --override "run.granularity=$granularity" \
    --override "model.d_model=$d_model" \
    --override "model.num_layers=$num_layers" \
    --override "training.token_budget=$token_budget" \
    --override "training.learning_rate=$learning_rate" \
    --override controlled_experiment.recipe_status=calibration \
    --override controlled_experiment.recipe_source_run_id=null \
    --override controlled_experiment.selection_report_hash=null
}
```

Run one representative preflight for each shape before consuming the four job
slots:

```bash
for D_MODEL in 96 64; do
  "$PYTHON_BIN" train.py \
    --config "$BASE" \
    --output-root "$CAPACITY_SELECTION_ROOT/preflight-d$D_MODEL" \
    "${DATA_PROFILE_OVERRIDES[@]}" \
    --override "run.run_id=preflight-capacity-${PROFILE_SLUG}-d${D_MODEL}-l4-g1000" \
    --override "run.seed=$EXPERIMENT_SEED" \
    --override "run.model_shape_label=${PROFILE_SLUG}-d${D_MODEL}-l4" \
    --override run.model_family=standalone \
    --override run.sampling_mode=standalone \
    --override run.granularity=g1000 \
    --override "model.d_model=$D_MODEL" \
    --override model.num_layers=4 \
    --override training.token_budget=134217728 \
    --override training.learning_rate=0.003 \
    --override controlled_experiment.recipe_status=calibration \
    --override controlled_experiment.recipe_source_run_id=null \
    --override controlled_experiment.selection_report_hash=null \
    --preflight
done
```

Submit exactly the first four jobs together:

```bash
for D_MODEL in 96 64; do
  submit_capacity_candidate "$D_MODEL" 134217728 0.003
  submit_capacity_candidate "$D_MODEL" 268435456 0.003
done
```

The helper derives 16,384 and 32,768 updates rather than accepting a separate,
possibly inconsistent step count. Every shape, horizon, and learning rate has
an isolated output root and run ID. Resubmit the identical call after an
interruption to resume it; never submit the same run ID concurrently. Its
optional fourth argument is `num_layers` and defaults to 4; its optional fifth
argument is the standalone granularity and defaults to `g1000`.

### 12.3 Analyze the screen and confirm the capacity choice

After all four jobs complete, generate ordinary-validation diagnostics:

```bash
"$PYTHON_BIN" scripts/analyze_tinystories_convergence.py \
  --runs-root "$CAPACITY_SELECTION_ROOT" \
  --output-dir "$CAPACITY_SELECTION_ROOT/convergence-analysis"
```

Use `convergence-analysis/run_comparison.csv`, not the analyzer's global
`winner`, for the cross-budget test. The latter was designed to select an LR
and scheduler for one fixed model shape; it does not encode this capacity rule.
Run this comparison after each analysis refresh:

```bash
"$PYTHON_BIN" - \
  "$CAPACITY_SELECTION_ROOT/convergence-analysis/run_comparison.csv" \
  "$PROFILE_SLUG" <<'PY'
import csv
import math
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
profile_slug = re.escape(sys.argv[2])
pattern = re.compile(
    rf"capacity-{profile_slug}-d(?P<d_model>64|96)-"
    r"l(?P<num_layers>3|4)-"
    r"(?P<steps>\d+)steps-g1000-lr0p003-s\d+$"
)
runs = {}
with path.open(newline="", encoding="utf-8") as source:
    for row in csv.DictReader(source):
        match = pattern.fullmatch(row["run_id"])
        if match is None:
            continue
        if row["stable"].lower() != "true":
            print("UNSTABLE", row["run_id"], row["rejection_reasons"])
            continue
        runs[
            (
                int(match["d_model"]),
                int(match["num_layers"]),
                int(match["steps"]),
            )
        ] = float(
            row["best_validation_loss"]
        )

tolerance = math.log(1.005)
shapes = sorted({(d_model, num_layers) for d_model, num_layers, _ in runs})
for d_model, num_layers in shapes:
    steps = sorted(
        step
        for candidate_d_model, candidate_num_layers, step in runs
        if (candidate_d_model, candidate_num_layers) == (d_model, num_layers)
    )
    for earlier_steps, later_steps in zip(steps, steps[1:]):
        earlier = runs[(d_model, num_layers, earlier_steps)]
        later = runs[(d_model, num_layers, later_steps)]
        improvement = earlier - later
        if 0.0 <= improvement < tolerance:
            decision = "PLATEAU"
        elif improvement >= tolerance:
            decision = "IMPROVING"
        else:
            decision = "REGRESSED; inspect or repeat"
        print(
            f"d{d_model}/l{num_layers} {earlier_steps}->{later_steps}: "
            f"loss improvement={improvement:.8f}, "
            f"perplexity improvement={math.expm1(improvement):.4%}, "
            f"{decision}"
        )
PY
```

If d96 passes the 4x to 8x threshold, confirm it at 12x. Otherwise confirm d64
if it passes. For example:

```bash
export SELECTED_D_MODEL=96
export SELECTED_NUM_LAYERS=4
submit_capacity_candidate \
  "$SELECTED_D_MODEL" 402653184 0.003 "$SELECTED_NUM_LAYERS"
```

Regenerate the diagnostics and require the 8x to 12x improvement to be below
the same tolerance before declaring a plateau. If neither model passes the
first interval, set `SELECTED_D_MODEL=64` and
`SELECTED_NUM_LAYERS=4`, then continue the lower-capacity candidate at 12x
before considering the complete unique-data prefix. Use at most the limit
reported by section 1:

```bash
submit_capacity_candidate \
  "$SELECTED_D_MODEL" "$MAX_ALIGNED_TOKEN_BUDGET" 0.003 "$SELECTED_NUM_LAYERS"
```

Do not silently wrap the corpus. If the selected four-layer shape is still
improving at the unique-data limit, record that the l4 stage did not find a
one-pass plateau and proceed to the depth refinement in section 12.4.
Repeated-data training remains a separate experiment with a declared
epoch/repetition policy and overfitting checks.

### 12.4 Refine depth if the four-layer models do not plateau

Use this refinement only after the d64/l4 candidate reaches the unique-data
limit without passing the plateau contract. Removing one transformer block
preserves the attention dimensions and FFN granularity ladder while reducing
the number of nested FFNs in which elastic gradients can interfere. It is a
more conservative refinement than reducing `d_model` below 64.

| Shape | Total parameters | Non-embedding parameters | Full FFN width | Eight-grid FFN widths |
| --- | ---: | ---: | ---: | --- |
| d96/l3 | 836,256 | 443,040 | 384 | 48, 96, 144, 192, 240, 288, 336, 384 |
| d64/l3 | 459,200 | 197,056 | 256 | 32, 64, 96, 128, 160, 192, 224, 256 |

This removes 25% of the non-embedding parameters but only about 15% of d96's
or 12.5% of d64's total parameters because the input embedding and untied LM
head are unchanged. Treat depth as the only new capacity variable: retain
standalone `g1000`, LR `0.003`, seed 42, and every data/training control from
section 12.1.

Preflight both three-layer shapes:

```bash
for D_MODEL in 96 64; do
  "$PYTHON_BIN" train.py \
    --config "$BASE" \
    --output-root "$CAPACITY_SELECTION_ROOT/preflight-d${D_MODEL}-l3" \
    "${DATA_PROFILE_OVERRIDES[@]}" \
    --override "run.run_id=preflight-capacity-${PROFILE_SLUG}-d${D_MODEL}-l3-g1000" \
    --override "run.seed=$EXPERIMENT_SEED" \
    --override "run.model_shape_label=${PROFILE_SLUG}-d${D_MODEL}-l3" \
    --override run.model_family=standalone \
    --override run.sampling_mode=standalone \
    --override run.granularity=g1000 \
    --override "model.d_model=$D_MODEL" \
    --override model.num_layers=3 \
    --override training.token_budget=134217728 \
    --override training.learning_rate=0.003 \
    --override controlled_experiment.recipe_status=calibration \
    --override controlled_experiment.recipe_source_run_id=null \
    --override controlled_experiment.selection_report_hash=null \
    --preflight
done
```

Then use all four available job slots for the paired screen:

```bash
for D_MODEL in 96 64; do
  submit_capacity_candidate "$D_MODEL" 134217728 0.003 3
  submit_capacity_candidate "$D_MODEL" 268435456 0.003 3
done
```

Regenerate `convergence-analysis` and rerun the generalized comparison block
from section 12.3; it discovers both l3 and l4 horizons from their run IDs. If
both l3 shapes pass 4x to 8x, prefer d96/l3 because it preserves the stronger
standalone target. Confirm the selected shape at 12x:

```bash
export SELECTED_D_MODEL=96
export SELECTED_NUM_LAYERS=3
submit_capacity_candidate \
  "$SELECTED_D_MODEL" 402653184 0.003 "$SELECTED_NUM_LAYERS"
```

Require the same threshold again from 8x to 12x. If neither l3 shape passes the
first interval, continue d64/l3 at 12x. Run it at the maximum aligned
unique-data budget only if 8x to 12x passes and a second successive plateau
interval is still needed:

```bash
export SELECTED_D_MODEL=64
export SELECTED_NUM_LAYERS=3
submit_capacity_candidate \
  "$SELECTED_D_MODEL" 402653184 0.003 "$SELECTED_NUM_LAYERS"

# Run only after analyzing the 12x result.
submit_capacity_candidate \
  "$SELECTED_D_MODEL" "$MAX_ALIGNED_TOKEN_BUDGET" 0.003 "$SELECTED_NUM_LAYERS"
```

Do not assume that fewer layers guarantee saturation: d64/l4 continued to
benefit from tokens, and shrinking d_model from 96 to 64 did not reduce its 4x
to 8x improvement. If d64/l3 also reaches the corpus limit without two
successive plateau intervals, close the unique-data refinement as “no
one-pass plateau found.” Decide explicitly between a fixed full-corpus
standalone reference and a new controlled-repetition experiment rather than
continuing unplanned architecture changes.

### 12.5 Tune standalone and elastic optimization separately

Once a shape has passed both cross-budget intervals, keep its `d_model`,
`num_layers`, and confirmed plateau horizon fixed. Tune the full-width
standalone anchor first. Reuse its `0.003` capacity run and submit the two new
initial candidates:

```bash
# Set this to the horizon that actually passed the plateau contract.
export SELECTED_PLATEAU_TOKEN_BUDGET=402653184
submit_capacity_candidate \
  "$SELECTED_D_MODEL" "$SELECTED_PLATEAU_TOKEN_BUDGET" 0.0045 \
  "$SELECTED_NUM_LAYERS"
submit_capacity_candidate \
  "$SELECTED_D_MODEL" "$SELECTED_PLATEAU_TOKEN_BUDGET" 0.006 \
  "$SELECTED_NUM_LAYERS"
```

Select on best ordinary-validation loss. If an edge wins, extend the grid one
point past that edge before freezing it: test `0.002` when `0.003` wins, or a
higher stable value when `0.006` wins. A learning rate is not selected merely
because its final loss is close; its completed run must also be stable and have
an available ordinary-validation best checkpoint.

Next train every standalone width at increasing unique-data budgets until each
passes the same two-interval plateau test. Freeze the best ordinary-validation
loss and checkpoint independently for every width. A full-width plateau alone
must not be projected onto smaller widths.

Elastic training gets its own LR calibration because the d128 results show
that `0.0045` slightly helped standalone training but did not close the elastic
gap and worsened the eight-granularity elastic result. Start the elastic grid
at `{0.002, 0.0025, 0.003, 0.0035}`. For each LR, use the normal section 1
profile and learning-rate isolation, run uniform `H=25` for both four and eight
granularities, and add these three overrides to every preflight and submission:

```bash
--override "model.d_model=$SELECTED_D_MODEL" \
--override "model.num_layers=$SELECTED_NUM_LAYERS" \
--override "run.model_shape_label=${PROFILE_SLUG}-d${SELECTED_D_MODEL}-l${SELECTED_NUM_LAYERS}"
```

Submit the LR grid in batches that respect the four-job queue limit. Select the
elastic LR by ordinary-validation loss across all widths in the intended grid,
not only by `g1000` and not by an unweighted average that can hide a badly
undertrained width. If separate four-grid and eight-grid LRs win, retain both
as separate recipes instead of forcing a post-hoc common winner.

### 12.6 Freeze targets, then measure catch-up

This historical workflow proposed creating a separate Instruct
`tinystories_instruct_capacity_converged.yaml` recipe after selection. That
recipe was never created and must not be inferred from the old instructions;
the profile selector now points at the real calibration recipe until plateau
evidence exists. Do not change `tinystories_controlled_convergence.yaml`, which
is the immutable ordinary-TinyStories d128 baseline. A future frozen recipe
must record at least the selected model shape,
standalone and elastic learning rates, scheduler, warmup, batch geometry,
plateau budgets, seed, selection-report hash, and source run IDs. Give each
recipe a new experiment name so it cannot resume any d128 run.

Then increase only the elastic token budget and compare each width with its
already frozen standalone target. A width has caught up when

```text
elastic_best_validation_loss - standalone_frozen_best_validation_loss
    <= log(1.005)
```

Require the condition independently at every reported width. Report the first
elastic token budget that satisfies it, the elastic-to-standalone token ratio,
and uncertainty across additional seeds before making a general claim. If the
unique-data limit is reached first, report a censored result: “no catch-up
within the available unique-data budget,” together with the remaining gap.

Only after these declarations may section 10's sealed final-holdout results be
opened for the one-time unbiased comparison. The holdout does not choose a
model, learning rate, checkpoint, horizon, or catch-up threshold.

### 12.7 Historical d128 recipe provenance

The current frozen recipe came from six independent d128 full-width runs over
`learning_rate in {3e-4, 1e-3, 3e-3}` and `scheduler in {cosine,
constant_with_warmup}` at 16,777,216 tokens. The two best fallback candidates
were rerun from initialization at 33,554,432 tokens with the schedule refit to
the longer horizon. The analyzer selected:

- run `tinystories-dense-lr3e-3-schedcosine-4096-s42`;
- best/final ordinary-validation loss `1.832638272675135` at step 4,096;
- learning rate `3e-3`, cosine decay, and 33,554,432 tokens;
- report hash
  `ecf84f1131b57255e945c10b51599bd1e84dfc872d48e65bc7a4818fe92c1c69`.

That provenance remains recorded in the frozen config. It is historical
evidence, not evidence that the d128 standalone model has reached its capacity
plateau on the full corpus.

## 13. Small amortized-budget pilot

Use this pilot to preview a different question from section 12: can one elastic
training run produce four useful widths with less aggregate training than four
independent models? It does **not** test whether dense training has plateaued,
and a positive pilot result is not a converged-recipe claim.

The pilot uses the TinyStories-Instruct profile, d64/l4, and ordinary
validation only. Its deliberately small, exactly aligned budgets are:

| Training program | Runs | Tokens per run | Updates per run | Aggregate tokens |
| --- | ---: | ---: | ---: | ---: |
| Four-width elastic | 1 | 32,768,000 (`B`) | 4,000 | 32,768,000 |
| Standalone references | 4 | 24,576,000 (`0.75B`) | 3,000 | 98,304,000 (`3B`) |

This is an amortization comparison, not equal aggregate compute. The
standalone group receives three times as many optimizer tokens as the elastic
run. Report measured aggregate single-GPU time as well, because widths have
different costs and token counts are not FLOPs. The pilot uses the same
`0.008` learning rate on both sides to isolate the workflow; a full experiment
must tune standalone and elastic optimization separately. Start elastic
training with balanced `H=1`, which changes width every update without random
label-count imbalance. Treat `H=25` as a later temporal-hold comparison.

### 13.1 Configure and preflight the isolated pilot

Run this block from the repository root. It is self-contained and does not
reuse the budget-dependent arrays from sections 1-8:

```bash
conda activate elasticnn
export TINYSTORIES_PROFILE=instruct
source scripts/select_tinystories_profile.sh

export EXPERIMENT_SEED=42
export PILOT_ELASTIC_TOKEN_BUDGET=32768000
export PILOT_STANDALONE_TOKEN_BUDGET=24576000
export PILOT_ELASTIC_STEPS=4000
export PILOT_STANDALONE_STEPS=3000
export PILOT_LEARNING_RATE=0.008
export PILOT_WALLTIME=00:30:00
export PILOT_NAME=tinystories-instruct-amortized-budget-pilot-v1
export PILOT_ROOT="$MATFORMER_EXPERIMENT_ROOT/$PILOT_NAME"
export PILOT_RUN_PREFIX=amortized-pilot-tinystories-instruct-d64-l4
export PILOT_LOG_ROOT=./logs

if (( PILOT_ELASTIC_TOKEN_BUDGET != PILOT_ELASTIC_STEPS * 8192 || \
      PILOT_STANDALONE_TOKEN_BUDGET != PILOT_STANDALONE_STEPS * 8192 || \
      4 * PILOT_STANDALONE_TOKEN_BUDGET != \
        3 * PILOT_ELASTIC_TOKEN_BUDGET || \
      PILOT_ELASTIC_STEPS % 4 != 0 )); then
  echo "Invalid pilot budget geometry" >&2
  false
fi

if [[ ! -r "$TOKENIZER/tokenizer_manifest.json" || \
      ! -r "$CORPUS/corpus_manifest.json" ]]; then
  echo "Prepare and audit the instruct artifacts with section 11.2 first" >&2
  false
fi

mkdir -p "$PILOT_ROOT" "$PILOT_LOG_ROOT"

PILOT_COMMON_OVERRIDES=(
  --override "run.phase_id=tinystories_instruct_amortized_budget_pilot"
  --override "run.seed=$EXPERIMENT_SEED"
  --override "run.model_shape_label=tinystories-instruct-d64-l4"
  --override model.d_model=64
  --override model.num_layers=4
  --override "model.tokenizer_dir=$TOKENIZER"
  --override 'model.granularities=[g250,g500,g750,g1000]'
  --override 'model.granularity_prefixes={g250: 0.25, g500: 0.50, g750: 0.75, g1000: 1.00}'
  --override "dataset.prepared_corpus_dir=$CORPUS"
  --override "dataset.dataset_name=$DATASET_NAME"
  --override "dataset.dataset_config_name=$DATASET_CONFIG_NAME"
  --override "dataset.dataset_split=$DATASET_SPLIT"
  --override "dataset.dataset_phase=$DATASET_PHASE"
  --override "dataset.preprocessing_notes=$PREPROCESSING_NOTES"
  --override "training.learning_rate=$PILOT_LEARNING_RATE"
  --override controlled_experiment.recipe_status=calibration
  --override controlled_experiment.recipe_source_run_id=null
  --override controlled_experiment.selection_report_hash=null
  --override monitoring.project=tinystories_instruct_amortized_budget_pilot
)

SBATCH=(
  sbatch
  --output="$PILOT_LOG_ROOT/%x_%j.out"
  --error="$PILOT_LOG_ROOT/%x_%j.err"
)
if [[ -n "${SLURM_EXCLUDE:-}" ]]; then
  SBATCH+=(--exclude="$SLURM_EXCLUDE")
fi

"$PYTHON_BIN" train.py \
  --config "$BASE" \
  --output-root "$PILOT_ROOT" \
  "${PILOT_COMMON_OVERRIDES[@]}" \
  --override "run.run_id=preflight-$PILOT_RUN_PREFIX-$PILOT_STANDALONE_STEPS-steps-4g-standalone-g1000-s$EXPERIMENT_SEED" \
  --override "training.token_budget=$PILOT_STANDALONE_TOKEN_BUDGET" \
  --override run.model_family=standalone \
  --override run.sampling_mode=standalone \
  --override run.granularity=g1000 \
  --preflight

"$PYTHON_BIN" train.py \
  --config "$BASE" \
  --output-root "$PILOT_ROOT" \
  "${PILOT_COMMON_OVERRIDES[@]}" \
  --override "run.run_id=preflight-$PILOT_RUN_PREFIX-$PILOT_ELASTIC_STEPS-steps-4g-elastic-balanced-h1-s$EXPERIMENT_SEED" \
  --override "training.token_budget=$PILOT_ELASTIC_TOKEN_BUDGET" \
  --override run.model_family=nested \
  --override run.sampling_mode=nested-random \
  --override run.granularity=null \
  --override model.granularity_sampling_mode=global \
  --override model.global_sampling_schedule=balanced_cycle \
  --override model.global_sampling_interval_steps=1 \
  --preflight
```

Require the standalone preflight to report only `g1000` and 3,000 updates, and
the elastic preflight to report the four ordered granularities, balanced-cycle
schedule version 1, interval 1, and 4,000 updates. Both must report 8,192
tokens per update, learning rate `0.008`, cosine decay, 64 warmup updates, and
seed 42. The elastic budget is divisible by the four-label cycle, so every
label will be selected for exactly 1,000 updates. Prefix parameters still
receive additional membership exposure when larger widths are selected.

### 13.2 Submit four standalone references and the H=1 elastic run

Define the pilot-specific launcher:

```bash
submit_amortized_pilot() {
  local job_name="$1"
  local run_id="$2"
  local token_budget="$3"
  shift 3

  "${SBATCH[@]}" \
    --time="$PILOT_WALLTIME" \
    --job-name="$job_name" \
    scripts/slurm_tinystories_controlled.sh \
    --python-bin "$PYTHON_BIN" \
    --config "$BASE" \
    --output-root "$PILOT_ROOT" \
    "${PILOT_COMMON_OVERRIDES[@]}" \
    --override "run.run_id=$run_id" \
    --override "training.token_budget=$token_budget" \
    "$@"
}
```

Submit the four standalone jobs as the first wave:

```bash
for GRANULARITY in g250 g500 g750 g1000; do
  submit_amortized_pilot \
    "$PILOT_RUN_PREFIX-$PILOT_STANDALONE_STEPS-steps-standalone-$GRANULARITY" \
    "$PILOT_RUN_PREFIX-$PILOT_STANDALONE_STEPS-steps-4g-standalone-$GRANULARITY-s$EXPERIMENT_SEED" \
    "$PILOT_STANDALONE_TOKEN_BUDGET" \
    --override run.model_family=standalone \
    --override run.sampling_mode=standalone \
    --override "run.granularity=$GRANULARITY"
done
```

After a job slot becomes available, submit the elastic run:

```bash
submit_amortized_pilot \
  "$PILOT_RUN_PREFIX-$PILOT_ELASTIC_STEPS-steps-elastic-balanced-h1" \
  "$PILOT_RUN_PREFIX-$PILOT_ELASTIC_STEPS-steps-4g-elastic-balanced-h1-s$EXPERIMENT_SEED" \
  "$PILOT_ELASTIC_TOKEN_BUDGET" \
  --override run.model_family=nested \
  --override run.sampling_mode=nested-random \
  --override run.granularity=null \
  --override model.granularity_sampling_mode=global \
  --override model.global_sampling_schedule=balanced_cycle \
  --override model.global_sampling_interval_steps=1
```

Rerun an identical call to resume an interrupted job. Never submit the same
run ID concurrently. Do not run final-holdout finalization for this pilot.
Analyze H=1 first. If a temporal-hold comparison is useful afterward, submit
H=25 under its own identity; it is a sixth run and is not part of the primary
five-run audit:

```bash
submit_amortized_pilot \
  "$PILOT_RUN_PREFIX-$PILOT_ELASTIC_STEPS-steps-elastic-balanced-h25" \
  "$PILOT_RUN_PREFIX-$PILOT_ELASTIC_STEPS-steps-4g-elastic-balanced-h25-s$EXPERIMENT_SEED" \
  "$PILOT_ELASTIC_TOKEN_BUDGET" \
  --override run.model_family=nested \
  --override run.sampling_mode=nested-random \
  --override run.granularity=null \
  --override model.granularity_sampling_mode=global \
  --override model.global_sampling_schedule=balanced_cycle \
  --override model.global_sampling_interval_steps=25
```

### 13.3 Compare the completed pilot on ordinary validation

The elastic comparison uses all widths from its single final model state. Each
standalone target may use its own best ordinary-validation checkpoint. This is
stricter than choosing a different elastic checkpoint for each width and
preserves the claim that one elastic model supplies all four widths.

Run this audit after all five jobs complete:

```bash
"$PYTHON_BIN" - "$PILOT_ROOT" "$PILOT_RUN_PREFIX" <<'PY'
import csv
import json
import math
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
prefix = sys.argv[2]
widths = ("g250", "g500", "g750", "g1000")
standalone_budget = 24_576_000
elastic_budget = 32_768_000
loss_tolerance = math.log(1.005)

runs = {}
for summary_path in root.rglob("run_summary.json"):
    summary = json.loads(summary_path.read_text())
    run_id = summary["run_id"]
    if not run_id.startswith(prefix):
        continue
    run_dir = summary_path.parent
    config = json.loads((run_dir / "config.json").read_text())
    assert summary["status"] == "completed", (run_id, summary["status"])
    assert config["model"]["d_model"] == 64
    assert config["model"]["num_layers"] == 4
    assert config["comparison_control_inputs"]["root_seed"] == 42
    runs[run_id] = (run_dir, summary, config)

standalones = {}
elastic = None
for run_id, payload in runs.items():
    match = re.search(r"-4g-standalone-(g\d+)-s42$", run_id)
    if match:
        standalones[match.group(1)] = payload
    elif run_id.endswith("-4g-elastic-balanced-h1-s42"):
        elastic = payload

assert set(standalones) == set(widths), sorted(standalones)
assert elastic is not None
all_runs = [standalones[width] for width in widths] + [elastic]

for field in (
    "corpus_hash",
    "data_roles_manifest_hash",
    "validation_manifest_hash",
    "controller_manifest_hash",
    "final_holdout_manifest_hash",
):
    values = {summary[field] for _, summary, _ in all_runs}
    assert len(values) == 1, (field, values)

for _, _, config in standalones.values():
    assert config["training"]["token_budget"] == standalone_budget
assert elastic[2]["training"]["token_budget"] == elastic_budget
assert elastic[1]["global_sampling_state"]["exposure_counts"] == {
    width: 1_000 for width in widths
}

def scaling_rows(run_dir):
    with (run_dir / "scaling_results.csv").open(newline="") as source:
        return {row["granularity"]: row for row in csv.DictReader(source)}

elastic_rows = scaling_rows(elastic[0])
all_pass = True
print("width,standalone_best,elastic_final,loss_gap,ppl_gap,within_0.5pct")
for width in widths:
    standalone_row = scaling_rows(standalones[width][0])[width]
    standalone_loss = float(standalone_row["best_validation_loss"])
    elastic_loss = float(elastic_rows[width]["final_validation_loss"])
    gap = elastic_loss - standalone_loss
    passed = gap <= loss_tolerance
    all_pass &= passed
    print(
        f"{width},{standalone_loss:.8f},{elastic_loss:.8f},"
        f"{gap:.8f},{math.expm1(gap):.4%},{passed}"
    )

standalone_seconds = sum(run[1]["wall_clock_seconds"] for run in standalones.values())
elastic_seconds = elastic[1]["wall_clock_seconds"]
print(f"aggregate_token_ratio={4 * standalone_budget / elastic_budget:.3f}x")
print(f"observed_single_gpu_time_ratio={standalone_seconds / elastic_seconds:.3f}x")
print(f"all_widths_within_0.5pct={all_pass}")
PY
```

Treat the output as a pipeline and effect-size check. Success requires every
width to be within 0.5% perplexity of its standalone reference; an average may
not hide a failing width. Before making a research claim, tune the two sides
separately, replace the pilot budgets with the declared large budgets, repeat
the frozen protocol across additional seeds, report estimated FLOPs or
aggregate GPU-hours, and only then perform one sealed final-holdout evaluation.
