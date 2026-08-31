# TinyStories-Instruct amortized portfolio catch-up

This workflow compares acquisition of four independent standalone models with
one four-width elastic model. It is separate from the historical plateau and
WSD campaigns: no plateau, convergence-window, or one-standalone advantage is
claimed.

The fixed budgets are:

- `B = 713785344` optimizer tokens (`87132` aligned steps);
- standalone portfolio acquisition: `4B = 2855141376` tokens;
- full elastic candidate: `3B = 2141356032` tokens;
- realized frozen-recipe spend: `3B/4B = 75%`.

Both roles use the same fixed optimizer recipe, including LR `0.008`, AdamW
settings, warmup, batch construction, and cosine schedule family. The cosine
horizon follows each declared budget (`B` or `3B`); there is no LR screen or
separate tuning cost.

## Shell requirement

Every command block in this guide is executable in **zsh**. Run the one-time
environment block and all later stages in the same zsh session so its arrays,
functions, and paths remain available. No Bash subshell or shell switch is
required.

## Choose the MatFormer granularities

Copy and paste this before the environment block to use
`{1/8, 1/4, 1/2, 1}` instead of the default quartile geometry:

```bash
export USE_MATFORMER_GRANULARITIES=1
```

## One-time environment and all paths

Run this block once in every new shell. It defines every training, analysis,
holdout, and figure path used by the complete workflow. It creates only
`logs/`, which Slurm needs when a job is submitted; every artifact-producing
command creates its own destination only when it actually has work to write.

```bash
export TINYSTORIES_PROFILE=instruct
source scripts/select_tinystories_profile.sh

export BASE=configs/controlled_exps/tinystories_instruct_portfolio_catchup.yaml
export USE_MATFORMER_GRANULARITIES="${USE_MATFORMER_GRANULARITIES:-0}"

case "$USE_MATFORMER_GRANULARITIES" in
  0)
    export PORTFOLIO_ROOT="$MATFORMER_EXPERIMENT_ROOT/tinystories-instruct-portfolio-catchup-v1"
    export PORTFOLIO_COMPARISON_GROUP=tinystories_instruct_portfolio_catchup_v1
    PORTFOLIO_WIDTHS=(g250 g500 g750 g1000)
    GRANULARITY_OVERRIDES=()
    ANALYZER_PROFILE_ARGS=()
    ;;
  1)
    export PORTFOLIO_ROOT="$MATFORMER_EXPERIMENT_ROOT/tinystories-instruct-portfolio-catchup-matformer-granularities-v1"
    export PORTFOLIO_COMPARISON_GROUP=tinystories_instruct_portfolio_catchup_matformer_granularities_v1
    PORTFOLIO_WIDTHS=(g125 g250 g500 g1000)
    GRANULARITY_OVERRIDES=(
      --override "controlled_experiment.comparison_group_id=$PORTFOLIO_COMPARISON_GROUP"
      --override 'controlled_experiment.portfolio_catchup.granularities=[g125, g250, g500, g1000]'
      --override 'model.granularities=[g125, g250, g500, g1000]'
      --override 'model.granularity_prefixes={g125: 0.125, g250: 0.25, g500: 0.50, g1000: 1.00}'
    )
    ANALYZER_PROFILE_ARGS=(--granularity-profile matformer)
    ;;
  *)
    echo "USE_MATFORMER_GRANULARITIES must be 0 or 1" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

export REFERENCE_ROOT="$PORTFOLIO_ROOT/reference-runs"
export CANDIDATE_ROOT="$PORTFOLIO_ROOT/elastic-candidate-runs"
export ANALYSIS_ROOT="$PORTFOLIO_ROOT/analysis"
export HOLDOUT_ROOT="$PORTFOLIO_ROOT/final-holdout-analysis"
export FIGURES_ROOT="$PORTFOLIO_ROOT/figures"
export REFERENCE_ANALYSIS="$ANALYSIS_ROOT/references"
export TARGET_MANIFEST="$REFERENCE_ANALYSIS/standalone_portfolio_targets.json"

export CATCHUP_ANALYSIS="$ANALYSIS_ROOT/portfolio-catchup"
export CATCHUP_REPORT="$CATCHUP_ANALYSIS/portfolio_catchup_report.json"
export HOLDOUT_SELECTION="$CATCHUP_ANALYSIS/final_holdout_selection_manifest.json"

export EXTENSION_ROOT="$PORTFOLIO_ROOT/diagnostic-extension"
export UNIFORM_4B_ROOT="$EXTENSION_ROOT/uniform-h1-4b-runs"
export NESTED_ALL_ROOT="$EXTENSION_ROOT/nested-all-b-runs"
export NESTED_ALL_4B_ROOT="$EXTENSION_ROOT/nested-all-4b-runs"
export CONCAT_4B_ROOT="$EXTENSION_ROOT/concat-uniform-h1-4b-runs"
export EXTENSION_ANALYSIS_ROOT="$EXTENSION_ROOT/analysis"
export EXTENSION_HOLDOUT_ROOT="$EXTENSION_ROOT/final-holdout-analysis"
export EXTENSION_FIGURES_ROOT="$EXTENSION_ROOT/figures"

export UNIFORM_4B_ANALYSIS="$EXTENSION_ANALYSIS_ROOT/uniform-h1-4b"
export NESTED_ALL_ANALYSIS="$EXTENSION_ANALYSIS_ROOT/nested-all-b"
export NESTED_ALL_4B_ANALYSIS="$EXTENSION_ANALYSIS_ROOT/nested-all-4b"
export CONCAT_4B_ANALYSIS="$EXTENSION_ANALYSIS_ROOT/concat-uniform-h1-4b"

PORTFOLIO_ARMS=(
  uniform_h1_3b
  uniform_h1_4b
  nested_all_b
  nested_all_4b
  concat_uniform_h1_4b
)

arm_run_root() {
  case "$1" in
    uniform_h1_3b) echo "$CANDIDATE_ROOT" ;;
    uniform_h1_4b) echo "$UNIFORM_4B_ROOT" ;;
    nested_all_b) echo "$NESTED_ALL_ROOT" ;;
    nested_all_4b) echo "$NESTED_ALL_4B_ROOT" ;;
    concat_uniform_h1_4b) echo "$CONCAT_4B_ROOT" ;;
    *) return 2 ;;
  esac
}

arm_analysis_dir() {
  case "$1" in
    uniform_h1_3b) echo "$CATCHUP_ANALYSIS" ;;
    uniform_h1_4b) echo "$UNIFORM_4B_ANALYSIS" ;;
    nested_all_b) echo "$NESTED_ALL_ANALYSIS" ;;
    nested_all_4b) echo "$NESTED_ALL_4B_ANALYSIS" ;;
    concat_uniform_h1_4b) echo "$CONCAT_4B_ANALYSIS" ;;
    *) return 2 ;;
  esac
}

arm_holdout_dir() {
  case "$1" in
    uniform_h1_3b) echo "$HOLDOUT_ROOT" ;;
    uniform_h1_4b) echo "$EXTENSION_HOLDOUT_ROOT/uniform-h1-4b" ;;
    nested_all_b) echo "$EXTENSION_HOLDOUT_ROOT/nested-all-b" ;;
    nested_all_4b) echo "$EXTENSION_HOLDOUT_ROOT/nested-all-4b" ;;
    concat_uniform_h1_4b) echo "$EXTENSION_HOLDOUT_ROOT/concat-uniform-h1-4b" ;;
    *) return 2 ;;
  esac
}

arm_figure_dir() {
  case "$1" in
    uniform_h1_3b) echo "$FIGURES_ROOT" ;;
    uniform_h1_4b) echo "$EXTENSION_FIGURES_ROOT/uniform-h1-4b" ;;
    nested_all_b) echo "$EXTENSION_FIGURES_ROOT/nested-all-b" ;;
    nested_all_4b) echo "$EXTENSION_FIGURES_ROOT/nested-all-4b" ;;
    concat_uniform_h1_4b) echo "$EXTENSION_FIGURES_ROOT/concat-uniform-h1-4b" ;;
    *) return 2 ;;
  esac
}

export B=713785344
export THREE_B=2141356032
export FOUR_B=2855141376

test "$B" -eq 713785344
test "$THREE_B" -eq 2141356032
test "$FOUR_B" -eq 2855141376
test "${#PORTFOLIO_WIDTHS[@]}" -eq 4
if test -r "$TARGET_MANIFEST"; then
  export TARGET_HASH="$(jq -er '.manifest_hash' "$TARGET_MANIFEST")"
fi
mkdir -p logs
```

Leave the variable unset for the existing quartile geometry
`{1/4, 1/2, 3/4, 1}`.

The switch changes the comparison-group contract, every training override, the
reference manifest, and the complete artifact tree. It therefore runs the same
standalone, `3B`, `4B`, nested-all, concat, holdout, and figure workflow without
mixing either geometry's checkpoints or analysis artifacts. Do not change the
flag midway through a workflow. With the flag set, all of those artifacts live
under
`$MATFORMER_EXPERIMENT_ROOT/tinystories-instruct-portfolio-catchup-matformer-granularities-v1`.

Every resubmission must use the same role, run ID, budget, LR, scheduler,
granularity set, corpus, topology, and manifest hashes. Changing one starts a
fresh run rather than continuing an existing checkpoint.

Standalone references launched with the earlier schema-1 contract use the
same scientific recipe and should be allowed to finish. They are accepted by
`freeze-references`. If one of those exact runs must resume from a checkpoint
after this update, append all three compatibility overrides to its original
standalone command:

```bash
--override controlled_experiment.portfolio_catchup.schema_version=1 \
--override controlled_experiment.portfolio_catchup.lr_selection_manifest_path=null \
--override controlled_experiment.portfolio_catchup.lr_selection_manifest_hash=null
```

These overrides are only for an existing schema-1 standalone run. New
standalone references and every elastic candidate use the schema-2 base
configuration.

## Acquire the standalone references

Preflight the complete matrix before submitting it:

```bash
for SEED in 42 43 44; do
  for WIDTH in "${PORTFOLIO_WIDTHS[@]}"; do
    RUN_ID="tiny-instruct-portfolio-${WIDTH}-s${SEED}"
    "$PYTHON_BIN" train.py \
      --config "$BASE" \
      "${GRANULARITY_OVERRIDES[@]}" \
      --output-root "$REFERENCE_ROOT/$WIDTH/s$SEED" \
      --override "run.run_id=$RUN_ID" \
      --override "run.seed=$SEED" \
      --override "run.granularity=$WIDTH" \
      --override "model.tokenizer_dir=$TOKENIZER" \
      --override "dataset.prepared_corpus_dir=$CORPUS" \
      --preflight
  done
done
```

Launch exactly those commands:

```bash
for SEED in 42 43 44; do
  for WIDTH in "${PORTFOLIO_WIDTHS[@]}"; do
    RUN_ID="tiny-instruct-portfolio-${WIDTH}-s${SEED}"
    sbatch \
      --job-name="portfolio-${WIDTH}-s${SEED}" \
      --time=24:00:00 \
      scripts/slurm_tinystories_controlled.sh \
      --python-bin "$PYTHON_BIN" \
      --config "$BASE" \
      "${GRANULARITY_OVERRIDES[@]}" \
      --output-root "$REFERENCE_ROOT/$WIDTH/s$SEED" \
      --override "run.run_id=$RUN_ID" \
      --override "run.seed=$SEED" \
      --override "run.granularity=$WIDTH" \
      --override "model.tokenizer_dir=$TOKENIZER" \
      --override "dataset.prepared_corpus_dir=$CORPUS"
  done
done
```

## Analyze every currently completed result

Run this block after a complete four-width reference panel finishes, then rerun
the same block unchanged whenever candidate arms finish. It ignores missing and
in-progress runs, freezes references only when no target manifest exists, and
analyzes every arm with at least one completed candidate for a target-manifest
seed. No empty analysis directory is created for an unavailable arm.

```bash
completed_runs_under() {
  local ROOT="$1"
  local SUMMARY
  test -d "$ROOT" || return 0
  while IFS= read -r SUMMARY; do
    if jq -e '.status == "completed"' "$SUMMARY" >/dev/null; then
      dirname -- "$SUMMARY"
    fi
  done < <(find "$ROOT" -type f -name run_summary.json -print)
}

if ! test -r "$TARGET_MANIFEST"; then
  REFERENCE_ARGS=()
  for SEED in 42 43 44; do
    PANEL_ARGS=()
    PANEL_READY=1
    for WIDTH in "${PORTFOLIO_WIDTHS[@]}"; do
      MATCHES=()
      while IFS= read -r MATCH; do
        MATCHES+=("$MATCH")
      done < <(completed_runs_under "$REFERENCE_ROOT/$WIDTH/s$SEED" | sort)
      if test "${#MATCHES[@]}" -ne 1; then
        PANEL_READY=0
        if test "${#MATCHES[@]}" -gt 1; then
          echo "Skipping ambiguous reference seed $SEED width $WIDTH" >&2
        fi
        break
      fi
      for MATCH in "${MATCHES[@]}"; do
        PANEL_ARGS+=(--run-dir "$MATCH")
      done
    done
    if test "$PANEL_READY" -eq 1; then
      REFERENCE_ARGS+=("${PANEL_ARGS[@]}")
    fi
  done

  if test "${#REFERENCE_ARGS[@]}" -gt 0; then
    "$PYTHON_BIN" scripts/analyze_tinystories_portfolio_catchup.py \
      freeze-references \
      "${ANALYZER_PROFILE_ARGS[@]}" \
      "${REFERENCE_ARGS[@]}" \
      --output-dir "$REFERENCE_ANALYSIS"
  else
    echo "No complete four-width reference panel is available yet"
  fi
else
  echo "Keeping existing immutable target manifest: $TARGET_MANIFEST"
fi

if test -r "$TARGET_MANIFEST"; then
  export TARGET_HASH="$(jq -er '.manifest_hash' "$TARGET_MANIFEST")"
  TARGET_SEEDS=()
  while IFS= read -r SEED; do
    TARGET_SEEDS+=("$SEED")
  done < <(jq -r '.observed_seeds[]' "$TARGET_MANIFEST")

  for ARM in "${PORTFOLIO_ARMS[@]}"; do
    ARM_RUN_DIR="$(arm_run_root "$ARM")"
    ARM_ANALYSIS="$(arm_analysis_dir "$ARM")"
    CANDIDATE_ARGS=()
    ARM_UNAMBIGUOUS=1
    for SEED in "${TARGET_SEEDS[@]}"; do
      MATCHES=()
      while IFS= read -r MATCH; do
        MATCHES+=("$MATCH")
      done < <(completed_runs_under "$ARM_RUN_DIR/s$SEED" | sort)
      if test "${#MATCHES[@]}" -eq 1; then
        for MATCH in "${MATCHES[@]}"; do
          CANDIDATE_ARGS+=(--run-dir "$MATCH")
        done
      elif test "${#MATCHES[@]}" -gt 1; then
        echo "Skipping arm $ARM: multiple completed runs for seed $SEED" >&2
        ARM_UNAMBIGUOUS=0
      fi
    done

    if test "$ARM_UNAMBIGUOUS" -eq 1 && test "${#CANDIDATE_ARGS[@]}" -gt 0; then
      "$PYTHON_BIN" scripts/analyze_tinystories_portfolio_catchup.py \
        portfolio-catchup \
        "${CANDIDATE_ARGS[@]}" \
        --target-manifest "$TARGET_MANIFEST" \
        --candidate-arm "$ARM" \
        --output-dir "$ARM_ANALYSIS"
      jq '{
        arm: .comparison_arm_id,
        status,
        observed_seeds,
        missing_seeds,
        general_claim: .general_portfolio_catchup_claim
      }' "$ARM_ANALYSIS/portfolio_catchup_report.json"
    else
      echo "No analyzable completed candidates for $ARM"
    fi
  done

  AVAILABLE_REPORTS=()
  for ARM in "${PORTFOLIO_ARMS[@]}"; do
    REPORT="$(arm_analysis_dir "$ARM")/portfolio_catchup_report.json"
    if test -r "$REPORT"; then
      AVAILABLE_REPORTS+=("$REPORT")
    fi
  done
  if test "${#AVAILABLE_REPORTS[@]}" -gt 0; then
    jq -s '[.[] | {
      arm: .comparison_arm_id,
      variant: .candidate_model_variant,
      status,
      observed_seeds,
      optimizer_spend_over_4B: .realized_full_run_spend_over_4B,
      subnetwork_target_tokens: .realized_subnetwork_target_tokens,
      worst_terminal_ppl_deficit_by_seed: [
        .seeds[] | {
          seed,
          value: ([.final_per_width_deficits[].perplexity_deficit] | max)
        }
      ]
    }]' "${AVAILABLE_REPORTS[@]}"
  fi
fi
```

A one-seed freeze records `observed_seeds: [42]`, `missing_seeds: [43, 44]`,
and `status: references_frozen_provisional`. Its target manifest can drive the
matching candidate, holdout, and figures. Because that manifest becomes a
hashed training input, the block deliberately never overwrites it. To freeze a
larger reference set for newly launched candidates, choose a new
`REFERENCE_ANALYSIS` in the one-time path block.

## Run fresh `3B` elastic candidates

The frozen target manifest is an immutable resume input. Catch-up saves the
first fifth-point confirmation checkpoint but never stops the run early. One or
two completed seeds can be analyzed provisionally; all three seeds remain
required for the predefined general claim.

```bash
for SEED in 42 43 44; do
  RUN_ID="tiny-instruct-portfolio-candidate-s${SEED}"
  "$PYTHON_BIN" train.py \
    --config "$BASE" \
    "${GRANULARITY_OVERRIDES[@]}" \
    --output-root "$CANDIDATE_ROOT/s$SEED" \
    --override "run.run_id=$RUN_ID" \
    --override "run.seed=$SEED" \
    --override run.model_family=nested \
    --override run.sampling_mode=nested-random \
    --override run.granularity=null \
    --override controlled_experiment.comparison_role=elastic_candidate \
    --override controlled_experiment.portfolio_catchup.enabled=true \
    --override "controlled_experiment.portfolio_catchup.target_manifest_path=$TARGET_MANIFEST" \
    --override "controlled_experiment.portfolio_catchup.target_manifest_hash=$TARGET_HASH" \
    --override "training.token_budget=$THREE_B" \
    --override training.learning_rate=0.008 \
    --override model.granularity_sampling_mode=global \
    --override model.global_sampling_schedule=random_with_replacement \
    --override model.global_sampling_interval_steps=1 \
    --override "model.tokenizer_dir=$TOKENIZER" \
    --override "dataset.prepared_corpus_dir=$CORPUS" \
    --preflight
done
```

```bash
for SEED in 42 43 44; do
  RUN_ID="tiny-instruct-portfolio-candidate-s${SEED}"
  sbatch \
    --job-name="portfolio-candidate-s${SEED}" \
    --time=36:00:00 \
    scripts/slurm_tinystories_controlled.sh \
    --python-bin "$PYTHON_BIN" \
    --config "$BASE" \
    "${GRANULARITY_OVERRIDES[@]}" \
    --output-root "$CANDIDATE_ROOT/s$SEED" \
    --override "run.run_id=$RUN_ID" \
    --override "run.seed=$SEED" \
    --override run.model_family=nested \
    --override run.sampling_mode=nested-random \
    --override run.granularity=null \
    --override controlled_experiment.comparison_role=elastic_candidate \
    --override controlled_experiment.portfolio_catchup.enabled=true \
    --override "controlled_experiment.portfolio_catchup.target_manifest_path=$TARGET_MANIFEST" \
    --override "controlled_experiment.portfolio_catchup.target_manifest_hash=$TARGET_HASH" \
    --override "training.token_budget=$THREE_B" \
    --override training.learning_rate=0.008 \
    --override model.granularity_sampling_mode=global \
    --override model.global_sampling_schedule=random_with_replacement \
    --override model.global_sampling_interval_steps=1 \
    --override "model.tokenizer_dir=$TOKENIZER" \
    --override "dataset.prepared_corpus_dir=$CORPUS"
done
```

After any `3B` candidate completes, rerun **Analyze every currently completed
result**. A one-seed report selects five checkpoints, a two-seed report selects
ten, and a complete report selects fifteen. Derived reports may be refreshed as
more candidate seeds finish; the frozen target manifest is never changed.

The report uses the fifth simultaneous qualifying validation as `t*`. If an
observed seed is censored, its elastic holdout entry uses the exact terminal
candidate-budget checkpoint and remains diagnostic rather than creating a
catch-up or savings claim.

## Post-hoc diagnostic extension: `4B` uniform H=1 and nested-all

Use this extension after the frozen `3B` uniform-H1 arm is censored. It reuses
the existing 12 standalone runs, their best checkpoints, and the exact frozen
target manifest. Do **not** rerun or refreeze the standalone references.

The three additional arms are:

- `uniform_h1_4b`: a fresh uniform-global, random-with-replacement, `H=1`
  elastic run with a cosine horizon of `4B`;
- `nested_all_b`: a fresh nested-all run with a cosine horizon of `B`, where
  every optimizer update averages the four width losses;
- `nested_all_4b`: the same nested-all procedure continued over a cosine
  horizon of `4B` optimizer/data tokens.

All three arms retain LR `0.008`, AdamW, warmup, batch construction, corpus,
ordinary-validation manifest, targets, tolerance, and five-validation streak.
They use schema 3 so their arm, topology, and budget are immutable resume
inputs. Existing schema-1/2 references and `3B` candidates remain unchanged
and resumable under their original contracts.

This is a post-hoc diagnosis, not a replacement frozen-recipe claim. Even if an
extension arm confirms catch-up, its analyzer report and final holdout remain
claim-ineligible. The already-opened final holdout may be evaluated for
descriptive comparison; a new prospective generalization claim would require a
new untouched holdout protocol fixed before inspecting these arm results.

### Reuse and verify the frozen inputs

Start from the one-time environment above. The manifest hash comes from the
already-frozen artifact rather than a regenerated target set:

```bash
export TARGET_HASH="$(jq -r '.manifest_hash' "$TARGET_MANIFEST")"
test -r "$TARGET_MANIFEST"
test -r "$CATCHUP_REPORT"
test "$(jq -r '.reference_budget_tokens' "$TARGET_MANIFEST")" -eq "$B"
test "$(jq -r '.aggregate_reference_budget_tokens' "$TARGET_MANIFEST")" -eq "$FOUR_B"
test "${#TARGET_HASH}" -eq 64
```

Never continue a `3B` checkpoint into either `4B` arm. Its cosine horizon was
resolved for `3B`, so each `4B` diagnosis must use fresh run IDs and an empty
output root. Likewise, nested-all is a new topology and must start from scratch.

### Preflight and launch fresh uniform-H1 `4B`

```bash
for SEED in 42 43 44; do
  RUN_ID="tiny-instruct-portfolio-uniform-h1-4b-s${SEED}"
  "$PYTHON_BIN" train.py \
    --config "$BASE" \
    "${GRANULARITY_OVERRIDES[@]}" \
    --output-root "$UNIFORM_4B_ROOT/s$SEED" \
    --override "run.run_id=$RUN_ID" \
    --override "run.seed=$SEED" \
    --override run.model_family=nested \
    --override run.sampling_mode=nested-random \
    --override run.granularity=null \
    --override controlled_experiment.comparison_role=elastic_candidate \
    --override controlled_experiment.comparison_arm_id=uniform_h1_4b \
    --override controlled_experiment.portfolio_catchup.schema_version=3 \
    --override controlled_experiment.portfolio_catchup.enabled=true \
    --override "controlled_experiment.portfolio_catchup.elastic_budget_cap_tokens=$FOUR_B" \
    --override "controlled_experiment.portfolio_catchup.target_manifest_path=$TARGET_MANIFEST" \
    --override "controlled_experiment.portfolio_catchup.target_manifest_hash=$TARGET_HASH" \
    --override "training.token_budget=$FOUR_B" \
    --override training.learning_rate=0.008 \
    --override model.granularity_sampling_mode=global \
    --override model.global_sampling_schedule=random_with_replacement \
    --override model.global_sampling_interval_steps=1 \
    --override "model.tokenizer_dir=$TOKENIZER" \
    --override "dataset.prepared_corpus_dir=$CORPUS" \
    --preflight
done
```

```bash
for SEED in 42 43 44; do
  RUN_ID="tiny-instruct-portfolio-uniform-h1-4b-s${SEED}"
  sbatch \
    --job-name="portfolio-uniform-h1-4b-s${SEED}" \
    --time=48:00:00 \
    scripts/slurm_tinystories_controlled.sh \
    --python-bin "$PYTHON_BIN" \
    --config "$BASE" \
    "${GRANULARITY_OVERRIDES[@]}" \
    --output-root "$UNIFORM_4B_ROOT/s$SEED" \
    --override "run.run_id=$RUN_ID" \
    --override "run.seed=$SEED" \
    --override run.model_family=nested \
    --override run.sampling_mode=nested-random \
    --override run.granularity=null \
    --override controlled_experiment.comparison_role=elastic_candidate \
    --override controlled_experiment.comparison_arm_id=uniform_h1_4b \
    --override controlled_experiment.portfolio_catchup.schema_version=3 \
    --override controlled_experiment.portfolio_catchup.enabled=true \
    --override "controlled_experiment.portfolio_catchup.elastic_budget_cap_tokens=$FOUR_B" \
    --override "controlled_experiment.portfolio_catchup.target_manifest_path=$TARGET_MANIFEST" \
    --override "controlled_experiment.portfolio_catchup.target_manifest_hash=$TARGET_HASH" \
    --override "training.token_budget=$FOUR_B" \
    --override training.learning_rate=0.008 \
    --override model.granularity_sampling_mode=global \
    --override model.global_sampling_schedule=random_with_replacement \
    --override model.global_sampling_interval_steps=1 \
    --override "model.tokenizer_dir=$TOKENIZER" \
    --override "dataset.prepared_corpus_dir=$CORPUS"
done
```

### Preflight and launch fresh nested-all `B`

Do not pass `model.global_sampling_*` overrides to nested-all; those controls
belong only to nested-random global sampling. Nested-all evaluates and
backpropagates all four widths on every optimizer update.

```bash
for SEED in 42 43 44; do
  RUN_ID="tiny-instruct-portfolio-nested-all-b-s${SEED}"
  "$PYTHON_BIN" train.py \
    --config "$BASE" \
    "${GRANULARITY_OVERRIDES[@]}" \
    --output-root "$NESTED_ALL_ROOT/s$SEED" \
    --override "run.run_id=$RUN_ID" \
    --override "run.seed=$SEED" \
    --override run.model_family=nested \
    --override run.sampling_mode=nested-all \
    --override run.granularity=null \
    --override controlled_experiment.comparison_role=elastic_candidate \
    --override controlled_experiment.comparison_arm_id=nested_all_b \
    --override controlled_experiment.portfolio_catchup.schema_version=3 \
    --override controlled_experiment.portfolio_catchup.enabled=true \
    --override "controlled_experiment.portfolio_catchup.elastic_budget_cap_tokens=$B" \
    --override "controlled_experiment.portfolio_catchup.target_manifest_path=$TARGET_MANIFEST" \
    --override "controlled_experiment.portfolio_catchup.target_manifest_hash=$TARGET_HASH" \
    --override "training.token_budget=$B" \
    --override training.learning_rate=0.008 \
    --override "model.tokenizer_dir=$TOKENIZER" \
    --override "dataset.prepared_corpus_dir=$CORPUS" \
    --preflight
done
```

```bash
for SEED in 42 43 44; do
  RUN_ID="tiny-instruct-portfolio-nested-all-b-s${SEED}"
  sbatch \
    --job-name="portfolio-nested-all-b-s${SEED}" \
    --time=48:00:00 \
    scripts/slurm_tinystories_controlled.sh \
    --python-bin "$PYTHON_BIN" \
    --config "$BASE" \
    "${GRANULARITY_OVERRIDES[@]}" \
    --output-root "$NESTED_ALL_ROOT/s$SEED" \
    --override "run.run_id=$RUN_ID" \
    --override "run.seed=$SEED" \
    --override run.model_family=nested \
    --override run.sampling_mode=nested-all \
    --override run.granularity=null \
    --override controlled_experiment.comparison_role=elastic_candidate \
    --override controlled_experiment.comparison_arm_id=nested_all_b \
    --override controlled_experiment.portfolio_catchup.schema_version=3 \
    --override controlled_experiment.portfolio_catchup.enabled=true \
    --override "controlled_experiment.portfolio_catchup.elastic_budget_cap_tokens=$B" \
    --override "controlled_experiment.portfolio_catchup.target_manifest_path=$TARGET_MANIFEST" \
    --override "controlled_experiment.portfolio_catchup.target_manifest_hash=$TARGET_HASH" \
    --override "training.token_budget=$B" \
    --override training.learning_rate=0.008 \
    --override "model.tokenizer_dir=$TOKENIZER" \
    --override "dataset.prepared_corpus_dir=$CORPUS"
done
```

`nested_all_b` spends `B` optimizer/data tokens but performs four subnetwork
forward/backward evaluations per optimizer step. The report therefore records
both `realized_full_run_spend_over_4B = 0.25` in optimizer-token units and
`realized_subnetwork_target_tokens = 4B` as a compute-exposure diagnostic. Do
not describe it as a quarter-compute run.

### Preflight and launch fresh nested-all `4B`

This is the full-optimizer-budget nested-all stress test. It uses the same
`4B` optimizer/data-token budget as the standalone portfolio acquisition cost,
but each optimizer step evaluates all four widths. Its aggregate subnetwork
exposure is therefore `16B`, four times the standalone portfolio's aggregate
target-token count. Treat it as a high-compute diagnostic, not a compute-matched
efficiency comparison.

```bash
for SEED in 42 43 44; do
  RUN_ID="tiny-instruct-portfolio-nested-all-4b-s${SEED}"
  "$PYTHON_BIN" train.py \
    --config "$BASE" \
    "${GRANULARITY_OVERRIDES[@]}" \
    --output-root "$NESTED_ALL_4B_ROOT/s$SEED" \
    --override "run.run_id=$RUN_ID" \
    --override "run.seed=$SEED" \
    --override run.model_family=nested \
    --override run.sampling_mode=nested-all \
    --override run.granularity=null \
    --override controlled_experiment.comparison_role=elastic_candidate \
    --override controlled_experiment.comparison_arm_id=nested_all_4b \
    --override controlled_experiment.portfolio_catchup.schema_version=3 \
    --override controlled_experiment.portfolio_catchup.enabled=true \
    --override "controlled_experiment.portfolio_catchup.elastic_budget_cap_tokens=$FOUR_B" \
    --override "controlled_experiment.portfolio_catchup.target_manifest_path=$TARGET_MANIFEST" \
    --override "controlled_experiment.portfolio_catchup.target_manifest_hash=$TARGET_HASH" \
    --override "training.token_budget=$FOUR_B" \
    --override training.learning_rate=0.008 \
    --override "model.tokenizer_dir=$TOKENIZER" \
    --override "dataset.prepared_corpus_dir=$CORPUS" \
    --preflight
done
```

Submit the fresh runs. This arm is approximately four times as expensive as
`nested_all_b`; adjust `--time` to the cluster limit and resubmit the identical
command if checkpoint continuation is required.

```bash
for SEED in 42 43 44; do
  RUN_ID="tiny-instruct-portfolio-nested-all-4b-s${SEED}"
  sbatch \
    --job-name="portfolio-nested-all-4b-s${SEED}" \
    --time=168:00:00 \
    scripts/slurm_tinystories_controlled.sh \
    --python-bin "$PYTHON_BIN" \
    --config "$BASE" \
    "${GRANULARITY_OVERRIDES[@]}" \
    --output-root "$NESTED_ALL_4B_ROOT/s$SEED" \
    --override "run.run_id=$RUN_ID" \
    --override "run.seed=$SEED" \
    --override run.model_family=nested \
    --override run.sampling_mode=nested-all \
    --override run.granularity=null \
    --override controlled_experiment.comparison_role=elastic_candidate \
    --override controlled_experiment.comparison_arm_id=nested_all_4b \
    --override controlled_experiment.portfolio_catchup.schema_version=3 \
    --override controlled_experiment.portfolio_catchup.enabled=true \
    --override "controlled_experiment.portfolio_catchup.elastic_budget_cap_tokens=$FOUR_B" \
    --override "controlled_experiment.portfolio_catchup.target_manifest_path=$TARGET_MANIFEST" \
    --override "controlled_experiment.portfolio_catchup.target_manifest_hash=$TARGET_HASH" \
    --override "training.token_budget=$FOUR_B" \
    --override training.learning_rate=0.008 \
    --override "model.tokenizer_dir=$TOKENIZER" \
    --override "dataset.prepared_corpus_dir=$CORPUS"
done
```

### Interpret the completed extension arms

Rerun **Analyze every currently completed result** after any extension seed
reaches its cap. Each arm is processed separately, so `B`, `3B`, and `4B` runs
are never aggregated as replications.

First interpret `nested_all_b` and `uniform_h1_4b` against the censored
uniform-H1 `3B` result:

| nested-all `B` | uniform-H1 `4B` | Main diagnosis |
|---|---|---|
| passes | passes | Width-exposure dilution is the leading explanation. |
| passes | fails | Sequential sampling or its schedule is the leading issue. |
| fails | passes | Extra sequential updates overcome simultaneous-gradient interference. |
| fails | fails | The shared architecture/optimizer recipe does not meet the frozen tolerance under either reasonable extension. |

Compare `nested_all_4b` directly with `nested_all_b`. If only the `4B` arm
passes, simultaneous all-width training is viable but needs substantially more
optimizer/data exposure than the compute-matched `B` test. If both fail, the
negative result persists even after spending the entire `4B` optimizer-token
portfolio budget—and `16B` subnetwork-target tokens—under nested-all.

“Passes” means five consecutive **joint** validations for all four widths and
all three seeds. Independently passing widths at different validations does not
count.

Extension holdout reports may set `diagnostic_arm_equivalence`, but they always
leave `general_portfolio_equivalence_claim` false.

## Post-hoc optimizer-isolation diagnostic: concat uniform-H1 `4B`

Run this arm after the slicing `uniform_h1_4b` result. It tests whether the
shared sliced FFN tensor interacts adversely with AdamW on steps where an outer
prefix is inactive. With `variant: slicing`, an inactive suffix belongs to a
parameter whose dense gradient exists, so AdamW can still decay the suffix and
advance its optimizer state. With `variant: concat`, each FFN quarter is a
separate parameter block and an unselected outer block has `grad=None`.

This arm changes only the FFN parameter representation. It retains uniform
global random-with-replacement sampling, `H=1`, LR `0.008`, AdamW with weight
decay `0.1`, cosine over `4B`, corpus, topology, targets, validation protocol,
and all four widths. Keep `correction_mode: none`; enabling LMC or GMC would
confound the optimizer-isolation test.

The arm deliberately reuses the frozen slicing standalone targets. The
analyzer permits exactly one reference/candidate provenance difference:
`model_variant` must change from reference `slicing` to candidate `concat`.
Every other provenance field remains exact. The result is post-hoc and
claim-ineligible even if all seeds confirm; do not rerun or refreeze the 12
standalone references.

### Isolate roots and verify frozen inputs

```bash
export TARGET_HASH="$(jq -r '.manifest_hash' "$TARGET_MANIFEST")"
test -r "$TARGET_MANIFEST"
test -r "$UNIFORM_4B_ANALYSIS/portfolio_catchup_report.json"
test "$(jq -r '.shared_provenance.model_variant' "$TARGET_MANIFEST")" = slicing
test "${#TARGET_HASH}" -eq 64
```

Use fresh run IDs and output roots. A slicing checkpoint cannot be resumed into
concat because its parameter layout and optimizer state are different.

### Preflight the three-seed matrix

```bash
for SEED in 42 43 44; do
  RUN_ID="tiny-instruct-portfolio-concat-uniform-h1-4b-s${SEED}"
  "$PYTHON_BIN" train.py \
    --config "$BASE" \
    "${GRANULARITY_OVERRIDES[@]}" \
    --output-root "$CONCAT_4B_ROOT/s$SEED" \
    --override "run.run_id=$RUN_ID" \
    --override "run.seed=$SEED" \
    --override run.model_family=nested \
    --override run.sampling_mode=nested-random \
    --override run.granularity=null \
    --override controlled_experiment.comparison_role=elastic_candidate \
    --override controlled_experiment.comparison_arm_id=concat_uniform_h1_4b \
    --override controlled_experiment.portfolio_catchup.schema_version=3 \
    --override controlled_experiment.portfolio_catchup.enabled=true \
    --override "controlled_experiment.portfolio_catchup.elastic_budget_cap_tokens=$FOUR_B" \
    --override "controlled_experiment.portfolio_catchup.target_manifest_path=$TARGET_MANIFEST" \
    --override "controlled_experiment.portfolio_catchup.target_manifest_hash=$TARGET_HASH" \
    --override "training.token_budget=$FOUR_B" \
    --override training.learning_rate=0.008 \
    --override model.variant=concat \
    --override model.correction_mode=none \
    --override model.granularity_sampling_mode=global \
    --override model.global_sampling_schedule=random_with_replacement \
    --override model.global_sampling_interval_steps=1 \
    --override "model.tokenizer_dir=$TOKENIZER" \
    --override "dataset.prepared_corpus_dir=$CORPUS" \
    --preflight
done
```

### Launch seed 42, then replicate only if warranted

Define the submission once:

```bash
submit_concat_4b_seed() {
  local SEED="$1"
  local RUN_ID="tiny-instruct-portfolio-concat-uniform-h1-4b-s${SEED}"
  sbatch \
    --job-name="portfolio-concat-h1-4b-s${SEED}" \
    --time=48:00:00 \
    scripts/slurm_tinystories_controlled.sh \
    --python-bin "$PYTHON_BIN" \
    --config "$BASE" \
    "${GRANULARITY_OVERRIDES[@]}" \
    --output-root "$CONCAT_4B_ROOT/s$SEED" \
    --override "run.run_id=$RUN_ID" \
    --override "run.seed=$SEED" \
    --override run.model_family=nested \
    --override run.sampling_mode=nested-random \
    --override run.granularity=null \
    --override controlled_experiment.comparison_role=elastic_candidate \
    --override controlled_experiment.comparison_arm_id=concat_uniform_h1_4b \
    --override controlled_experiment.portfolio_catchup.schema_version=3 \
    --override controlled_experiment.portfolio_catchup.enabled=true \
    --override "controlled_experiment.portfolio_catchup.elastic_budget_cap_tokens=$FOUR_B" \
    --override "controlled_experiment.portfolio_catchup.target_manifest_path=$TARGET_MANIFEST" \
    --override "controlled_experiment.portfolio_catchup.target_manifest_hash=$TARGET_HASH" \
    --override "training.token_budget=$FOUR_B" \
    --override training.learning_rate=0.008 \
    --override model.variant=concat \
    --override model.correction_mode=none \
    --override model.granularity_sampling_mode=global \
    --override model.global_sampling_schedule=random_with_replacement \
    --override model.global_sampling_interval_steps=1 \
    --override "model.tokenizer_dir=$TOKENIZER" \
    --override "dataset.prepared_corpus_dir=$CORPUS"
}

submit_concat_4b_seed 42
```

First compare seed 42's validation trajectories with the completed slicing
seed-42 run. If concat materially reduces the g750/g1000 deficits, submit the
remaining replications from the same shell definition:

```bash
submit_concat_4b_seed 43
submit_concat_4b_seed 44
```

Completing only seed 42 is sufficient for a provisional optimizer-mechanism
analysis. The analyzer will emit a five-checkpoint seed-42 holdout manifest and
figures, but it will mark seeds 43 and 44 missing and keep every general claim
false. Rerun **Analyze every currently completed result** after seed 42 and
again after adding seeds 43 and 44; the derived report and selection manifest
will update to the complete matrix.

### Analyze against the same targets

The centralized analysis block compares the concat report with the slicing
`4B` report without aggregating their seeds as replications.

Interpretation is deliberately narrow:

- a large, width-ordered improvement with concat supports the inactive-slice
  AdamW-state/decay mechanism;
- similar slicing and concat trajectories argue against that mechanism being
  the dominant cause;
- worse concat results require checking numerical/layout equivalence before
  drawing an optimizer conclusion.

As with the earlier diagnostic arms, holdout results may establish descriptive
`diagnostic_arm_equivalence` but cannot set a general portfolio-equivalence or
catch-up claim.

## Evaluate and analyze every available holdout

Run this block after the centralized analysis block. For each available
selection manifest it does one of three things: summarizes already-matching
results, leaves an active Slurm allocation alone, or submits one allocation for
the missing or stale results. Rerun it after submitted jobs finish. Arms without
a selection manifest are skipped and do not get a holdout directory.

```bash
selection_results_ready() {
  local SELECTION="$1"
  local RESULT_PATH CHECKPOINT_SHA RUN_ID CHECKPOINT_PATH
  local ENTRY_COUNT=0
  while IFS=$'\t' read -r RESULT_PATH CHECKPOINT_SHA RUN_ID CHECKPOINT_PATH; do
    ENTRY_COUNT=$((ENTRY_COUNT + 1))
    test -r "$RESULT_PATH" || return 1
    jq -e \
      --arg checkpoint_sha "$CHECKPOINT_SHA" \
      --arg run_id "$RUN_ID" \
      --arg checkpoint_path "$CHECKPOINT_PATH" \
      '.checkpoint_sha256 == $checkpoint_sha
       and .run_id == $run_id
       and .checkpoint_path == $checkpoint_path' \
      "$RESULT_PATH" >/dev/null || return 1
  done < <(
    jq -r '.entries[] |
      [.result_path, .checkpoint_sha256, .run_id, .checkpoint_path] | @tsv' \
      "$SELECTION"
  )
  test "$ENTRY_COUNT" -gt 0
}

for ARM in "${PORTFOLIO_ARMS[@]}"; do
  ARM_ANALYSIS="$(arm_analysis_dir "$ARM")"
  ARM_HOLDOUT="$(arm_holdout_dir "$ARM")"
  SELECTION="$ARM_ANALYSIS/final_holdout_selection_manifest.json"
  if ! test -r "$SELECTION"; then
    echo "No holdout selection for $ARM"
    continue
  fi

  if selection_results_ready "$SELECTION"; then
    "$PYTHON_BIN" scripts/analyze_tinystories_portfolio_catchup.py \
      final-holdout \
      --selection-manifest "$SELECTION" \
      --output-dir "$ARM_HOLDOUT"
    jq '{
      arm: .comparison_arm_id,
      status,
      observed_seeds,
      all_pairs_within_tolerance,
      general_claim: .general_portfolio_equivalence_claim
    }' "$ARM_HOLDOUT/portfolio_final_holdout_report.json"
    continue
  fi

  JOB_NAME="portfolio-${ARM//_/-}-holdout"
  PENDING_JOB=""
  if command -v squeue >/dev/null; then
    PENDING_JOB="$(squeue -h -u "$USER" -n "$JOB_NAME" | head -n 1)"
  fi
  if test -n "$PENDING_JOB"; then
    echo "Holdout allocation already active for $ARM"
  else
    sbatch \
      --job-name="$JOB_NAME" \
      --time=24:00:00 \
      scripts/slurm_tinystories_controlled.sh \
      --python-bin "$PYTHON_BIN" \
      --final-holdout-manifest "$SELECTION"
  fi
done
```

The Slurm wrapper uses `--skip-existing`, which hash-checks every requested
checkpoint before reusing a result. A provisional manifest evaluates five or
ten checkpoints; a complete manifest evaluates fifteen. Confirmation-selected
complete reports can support the predefined equivalence claim. Terminal or
post-hoc selections remain descriptive regardless of their holdout outcome.

## Generate figures for every available report

This block generates one manifest-scoped figure set per available analysis
report. It skips unavailable arms and therefore creates no empty figure roots.
Final-holdout panels appear automatically only when every result selected by
that arm's manifest is present and checkpoint-matched.

```bash
for ARM in "${PORTFOLIO_ARMS[@]}"; do
  ARM_ANALYSIS="$(arm_analysis_dir "$ARM")"
  ARM_FIGURES="$(arm_figure_dir "$ARM")"
  REPORT="$ARM_ANALYSIS/portfolio_catchup_report.json"
  if ! test -r "$REPORT"; then
    echo "No figure input for $ARM"
    continue
  fi

  "$PYTHON_BIN" scripts/make_figures.py \
    --input "$PORTFOLIO_ROOT" \
    --comparison-manifest "$REPORT" \
    --output "$ARM_FIGURES"
done
```

Figure membership comes exclusively from each sealed report. Standalone curves
end at `B`; elastic curves end at the arm's actual budget. Titles are derived
from the saved sampling mode, model variant, scheduler, budget, and seed count.
Legends use short role/arm labels, while panel titles identify the active
granularity. `final_holdout_ppl_vs_size.png` is the single final-holdout size
plot; the old duplicate `portfolio_final_holdout_perplexity.png` alias is
removed on regeneration. Without `--comparison-manifest`, figure generation
retains its historical strict same-contract behavior.
