# TinyStories-Instruct portfolio bundle

This workflow first freezes one standalone four-width reference panel, then
adds elastic candidate variants to that same reference lane. Preflights run
directly through Python; every training and final-holdout job runs through
`sbatch`. The portfolio analyzer and figures remain separate follow-up steps.
Each Slurm `.out` log starts with the experiment/run identity and its assigned
node, CUDA visibility, Slurm GPU allocation, CPU/memory allocation, and detected
GPU model and memory.

`B = 713,785,344` is the aligned epoch budget. A reference lane uses `R×B` per
standalone width. Each candidate variant independently uses `X×B`.

## 1. Freeze the standalone reference lane

Run this section first in one **zsh** session. Copy any desired optional line
before the setup block; otherwise it creates the default quartile, three-seed,
`R=1` lane.

```bash
# Optional geometry: {1/8, 1/4, 1/2, 1}; default is {1/4, 1/2, 3/4, 1}.
export PORTFOLIO_GRANULARITY_PROFILE=matformer

# Optional seed subset for an initial panel; default is "42 43 44".
export PORTFOLIO_SEEDS="42"
export PORTFOLIO_SEEDS="42 43"

# Optional standalone horizon; default is 1.
export REFERENCE_B_MULTIPLIER=2

# Optional Slurm exclusions, as a comma-separated list.
export SLURM_EXCLUDE_NODES=node001,node002
```

```bash
source /apps/local/anaconda3.10/etc/profile.d/conda.sh
conda activate elasticnn
export PYTHON_BIN="$(command -v python)"

export TINYSTORIES_PROFILE=instruct
source scripts/select_tinystories_profile.sh
export BASE=configs/controlled_exps/tinystories_instruct_portfolio_catchup.yaml
export PORTFOLIO_GRANULARITY_PROFILE="${PORTFOLIO_GRANULARITY_PROFILE:-quartile}"
export PORTFOLIO_SEEDS="${PORTFOLIO_SEEDS:-42 43 44}"
PORTFOLIO_SEED_LIST=(${=PORTFOLIO_SEEDS})
export REFERENCE_B_MULTIPLIER="${REFERENCE_B_MULTIPLIER:-1}"
export SLURM_EXCLUDE_NODES="${SLURM_EXCLUDE_NODES:-}"
export B=713785344
[[ "$REFERENCE_B_MULTIPLIER" == <-> && "$REFERENCE_B_MULTIPLIER" -gt 0 ]]
export REFERENCE_TOKENS=$(( REFERENCE_B_MULTIPLIER * B ))
SBATCH_EXCLUDE_ARGS=()
if [[ -n "$SLURM_EXCLUDE_NODES" ]]; then
  SBATCH_EXCLUDE_ARGS=(--exclude="$SLURM_EXCLUDE_NODES")
fi

case "$PORTFOLIO_GRANULARITY_PROFILE" in
  quartile)
    export PORTFOLIO_ROOT="$MATFORMER_EXPERIMENT_ROOT/tinystories-instruct-portfolio-bundle-v2"
    export PORTFOLIO_COMPARISON_GROUP=tinystories_instruct_portfolio_catchup_v1
    PORTFOLIO_WIDTHS=(g250 g500 g750 g1000)
    GRANULARITY_OVERRIDES=()
    ANALYZER_PROFILE_ARGS=()
    ;;
  matformer)
    export PORTFOLIO_ROOT="$MATFORMER_EXPERIMENT_ROOT/tinystories-instruct-portfolio-bundle-matformer-v2"
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
  *) print -u2 'PORTFOLIO_GRANULARITY_PROFILE must be quartile or matformer'; return 2 ;;
esac

export LANE_ROOT="$PORTFOLIO_ROOT/r$REFERENCE_B_MULTIPLIER"
export REFERENCE_ROOT="$LANE_ROOT/references"
export REFERENCE_ANALYSIS="$LANE_ROOT/analysis/references"
export TARGET_MANIFEST="$REFERENCE_ANALYSIS/standalone_portfolio_targets.json"
mkdir -p logs

REFERENCE_OVERRIDES=(
  --override "training.token_budget=$REFERENCE_TOKENS"
  --override "controlled_experiment.portfolio_catchup.reference_budget_multiplier=$REFERENCE_B_MULTIPLIER"
  --override "model.tokenizer_dir=$TOKENIZER"
  --override "dataset.prepared_corpus_dir=$CORPUS"
)
```

Run the reference preflights directly:

```bash
for SEED in "${PORTFOLIO_SEED_LIST[@]}"; do
  for WIDTH in "${PORTFOLIO_WIDTHS[@]}"; do
    "$PYTHON_BIN" train.py --config "$BASE" "${GRANULARITY_OVERRIDES[@]}" \
      --output-root "$REFERENCE_ROOT/$WIDTH/s$SEED" \
      --override "run.run_id=tiny-instruct-portfolio-r${REFERENCE_B_MULTIPLIER}-${WIDTH}-s${SEED}" \
      --override "run.seed=$SEED" --override "run.granularity=$WIDTH" \
      "${REFERENCE_OVERRIDES[@]}" --preflight
  done
done
```

After every preflight succeeds, submit the matching reference runs:

```bash
for SEED in "${PORTFOLIO_SEED_LIST[@]}"; do
  for WIDTH in "${PORTFOLIO_WIDTHS[@]}"; do
    sbatch "${SBATCH_EXCLUDE_ARGS[@]}" \
      --job-name="portfolio-r${REFERENCE_B_MULTIPLIER}-${WIDTH}-s${SEED}" \
      --time=24:00:00 scripts/slurm_tinystories_controlled.sh \
      --python-bin "$PYTHON_BIN" --config "$BASE" "${GRANULARITY_OVERRIDES[@]}" \
      --output-root "$REFERENCE_ROOT/$WIDTH/s$SEED" \
      --override "run.run_id=tiny-instruct-portfolio-r${REFERENCE_B_MULTIPLIER}-${WIDTH}-s${SEED}" \
      --override "run.seed=$SEED" --override "run.granularity=$WIDTH" \
      "${REFERENCE_OVERRIDES[@]}"
  done
done
```

Once a complete reference panel is available, freeze its immutable targets:

```bash
"$PYTHON_BIN" scripts/analyze_tinystories_portfolio_catchup.py freeze-references \
  "${ANALYZER_PROFILE_ARGS[@]}" --runs-root "$REFERENCE_ROOT" \
  --output-dir "$REFERENCE_ANALYSIS"
export TARGET_HASH="$(jq -er '.manifest_hash' "$TARGET_MANIFEST")"
```

Changing `REFERENCE_B_MULTIPLIER` starts another `r<R>` lane. Do not mix its
references or target manifest with candidates from another lane.

## 2. Run a reusable elastic variant

Keep the setup shell open after reference freeze. Copy one variant selection
and, if desired, a candidate budget multiplier:

| Variant | Sampling policy | Copy-paste selection |
| --- | --- | --- |
| Uniform H=1 | Uniform IID global width each update | `export VARIANT=uniform_h1` |
| Balanced H=1 | Equal selected-width counts, one update at a time | `export VARIANT=balanced_h1` |
| Balanced H=5 | Equal selected-width counts, five-update blocks | `export VARIANT=balanced_h5` |
| Fixed large | Global `0.20/0.20/0.25/0.35`, smallest to largest | `export VARIANT=fixed_large` |
| Thompson | Global Thompson, 25-update decisions, no reset | `export VARIANT=thompson` |
| PanelGrad L2 | Global L2 PanelGrad, 25-step refresh | `export VARIANT=panelgrad_l2` |
| Nested-all | All four widths per optimizer update | `export VARIANT=nested_all` |

```bash
# Optional candidate horizon; default is X=3.
export CANDIDATE_B_MULTIPLIER=4
```

Define the selected arm and its reusable overrides. No policy-specific
override is copied into an individual seed command.

```bash
export VARIANT="${VARIANT:-uniform_h1}"
export CANDIDATE_B_MULTIPLIER="${CANDIDATE_B_MULTIPLIER:-3}"
[[ "$CANDIDATE_B_MULTIPLIER" == <-> && "$CANDIDATE_B_MULTIPLIER" -gt 0 ]]
export CANDIDATE_TOKENS=$(( CANDIDATE_B_MULTIPLIER * B ))
export CANDIDATE_STEPS=$(( CANDIDATE_TOKENS / 8192 ))
export ARM_ID="${VARIANT}-x${CANDIDATE_B_MULTIPLIER}"
export ARM_ROOT="$LANE_ROOT/candidates/$ARM_ID"
export TARGET_HASH="$(jq -er '.manifest_hash' "$TARGET_MANIFEST")"

case "$VARIANT" in
  uniform_h1)
    VARIANT_OVERRIDES=(--override run.sampling_mode=nested-random --override model.granularity_sampling_mode=global --override model.global_sampling_schedule=random_with_replacement --override model.global_sampling_interval_steps=1) ;;
  balanced_h1|balanced_h5)
    H="${VARIANT#balanced_h}"
    VARIANT_OVERRIDES=(--override run.sampling_mode=nested-random --override model.granularity_sampling_mode=global --override model.global_sampling_schedule=balanced_cycle --override "model.global_sampling_interval_steps=$H") ;;
  fixed_large)
    VARIANT_OVERRIDES=(--override run.sampling_mode=nested-random --override model.granularity_sampling_mode=fixed_global --override "model.global_sampling_distribution={$PORTFOLIO_WIDTHS[1]: 0.20, $PORTFOLIO_WIDTHS[2]: 0.20, $PORTFOLIO_WIDTHS[3]: 0.25, $PORTFOLIO_WIDTHS[4]: 0.35}") ;;
  thompson)
    VARIANT_OVERRIDES=(--override run.sampling_mode=nested-random --override model.granularity_sampling_mode=adaptive_global --override model.adaptive_sampler_strategy=thompson --override 'model.adaptive_controller={preset: bayesian_thompson, decision_interval_steps: 25, prior_mean: 0.0, prior_covariance: 1.0, observation_noise_variance: 0.01, process_noise_covariance: 0.0001, reset: {enabled: false}}') ;;
  panelgrad_l2)
    VARIANT_OVERRIDES=(--override run.sampling_mode=nested-random --override model.granularity_sampling_mode=adaptive_global --override model.adaptive_sampler_strategy=panelgrad --override "model.panelgrad={importance_metric: gradient_l2, refresh_interval_steps: 25, eta: 1.0e-12, temperature: 1.0, epsilon_schedule: {type: linear, start: 0.5, end: 0.1, duration_steps: $CANDIDATE_STEPS}}") ;;
  nested_all) VARIANT_OVERRIDES=(--override run.sampling_mode=nested-all) ;;
  *) print -u2 "Unknown VARIANT=$VARIANT"; return 2 ;;
esac

CANDIDATE_OVERRIDES=(
  --override run.model_family=nested
  --override run.granularity=null
  --override controlled_experiment.comparison_role=elastic_candidate
  --override "controlled_experiment.comparison_arm_id=$ARM_ID"
  --override controlled_experiment.portfolio_catchup.enabled=true
  --override "controlled_experiment.portfolio_catchup.reference_budget_multiplier=$REFERENCE_B_MULTIPLIER"
  --override "controlled_experiment.portfolio_catchup.candidate_budget_multiplier=$CANDIDATE_B_MULTIPLIER"
  --override "controlled_experiment.portfolio_catchup.target_manifest_path=$TARGET_MANIFEST"
  --override "controlled_experiment.portfolio_catchup.target_manifest_hash=$TARGET_HASH"
  --override "training.token_budget=$CANDIDATE_TOKENS"
  --override "model.tokenizer_dir=$TOKENIZER"
  --override "dataset.prepared_corpus_dir=$CORPUS"
  "${VARIANT_OVERRIDES[@]}"
)
```

Run the selected candidate preflight matrix directly:

```bash
for SEED in "${PORTFOLIO_SEED_LIST[@]}"; do
  "$PYTHON_BIN" train.py --config "$BASE" "${GRANULARITY_OVERRIDES[@]}" \
    --output-root "$ARM_ROOT/s$SEED" \
    --override "run.run_id=tiny-instruct-portfolio-${ARM_ID}-s${SEED}" \
    --override "run.seed=$SEED" "${CANDIDATE_OVERRIDES[@]}" --preflight
done
```

After every preflight succeeds, submit the corresponding training matrix:

```bash
for SEED in "${PORTFOLIO_SEED_LIST[@]}"; do
  sbatch "${SBATCH_EXCLUDE_ARGS[@]}" \
    --job-name="portfolio-${ARM_ID}-s${SEED}" --time=48:00:00 \
    scripts/slurm_tinystories_controlled.sh --python-bin "$PYTHON_BIN" \
    --config "$BASE" "${GRANULARITY_OVERRIDES[@]}" \
    --output-root "$ARM_ROOT/s$SEED" \
    --override "run.run_id=tiny-instruct-portfolio-${ARM_ID}-s${SEED}" \
    --override "run.seed=$SEED" "${CANDIDATE_OVERRIDES[@]}"
done
```

## 3. Resume timed-out training jobs

Continuation is automatic: submitting the exact same run ID, output root, and
resolved experiment contract loads that run's `checkpoints/latest.pt`. Do not
pass a checkpoint path. Do not delete, rename, or move the timed-out run
directory, and never submit a resume while its previous job is still running.

After opening a new shell, restore the same optional geometry, reference
multiplier, and node exclusions, then rerun the setup block from section 1. For
an elastic candidate, also select the original `$VARIANT` and
`$CANDIDATE_B_MULTIPLIER` and rerun the **Define the selected arm and its
reusable overrides** block from section 2. These values must match the original
run exactly. For example, job `199775` uses:

```bash
export VARIANT=uniform_h1
export CANDIDATE_B_MULTIPLIER=16
```

### Resume selected candidate seeds

Set only the seeds that are no longer running. The default resume allocation is
48 hours; change it only if the cluster permits a different limit.

```bash
export RESUME_SEEDS="42"
export CANDIDATE_RESUME_TIME="${CANDIDATE_RESUME_TIME:-48:00:00}"
RESUME_SEED_LIST=(${=RESUME_SEEDS})

for SEED in "${RESUME_SEED_LIST[@]}"; do
  RESUME_CHECKPOINTS=("$ARM_ROOT/s$SEED"/**/checkpoints/latest.pt(N))
  if (( ${#RESUME_CHECKPOINTS[@]} != 1 )); then
    print -u2 "Expected one latest.pt for $ARM_ID seed $SEED; found ${#RESUME_CHECKPOINTS[@]}"
    continue
  fi

  CANDIDATE_RESUME_TRAIN_ARGS=(
    --config
    "$BASE"
    "${GRANULARITY_OVERRIDES[@]}"
    --output-root
    "$ARM_ROOT/s$SEED"
    --override
    "run.run_id=tiny-instruct-portfolio-${ARM_ID}-s${SEED}"
    --override
    "run.seed=$SEED"
    "${CANDIDATE_OVERRIDES[@]}"
  )

  if ! "$PYTHON_BIN" train.py "${CANDIDATE_RESUME_TRAIN_ARGS[@]}" --preflight; then
    print -u2 "Resume preflight failed for $ARM_ID seed $SEED"
    continue
  fi

  CANDIDATE_RESUME_SUBMIT=(
    sbatch
    "${SBATCH_EXCLUDE_ARGS[@]}"
    "--job-name=portfolio-${ARM_ID}-s${SEED}"
    "--time=$CANDIDATE_RESUME_TIME"
    ./scripts/slurm_tinystories_controlled.sh
    --python-bin
    "$PYTHON_BIN"
    "${CANDIDATE_RESUME_TRAIN_ARGS[@]}"
  )
  "${CANDIDATE_RESUME_SUBMIT[@]}"
done
```

The new `.out` log must contain `continuation_status=resumed` with a nonzero
`last_completed_step`. A long run can use this same block after every timeout.

### Resume selected standalone references

Use the original reference multiplier and geometry from section 1. Select both
the affected widths and seeds:

```bash
export RESUME_WIDTHS="g250"
export RESUME_SEEDS="42"
export REFERENCE_RESUME_TIME="${REFERENCE_RESUME_TIME:-24:00:00}"
RESUME_WIDTH_LIST=(${=RESUME_WIDTHS})
RESUME_SEED_LIST=(${=RESUME_SEEDS})

for SEED in "${RESUME_SEED_LIST[@]}"; do
  for WIDTH in "${RESUME_WIDTH_LIST[@]}"; do
    RESUME_CHECKPOINTS=("$REFERENCE_ROOT/$WIDTH/s$SEED"/**/checkpoints/latest.pt(N))
    if (( ${#RESUME_CHECKPOINTS[@]} != 1 )); then
      print -u2 "Expected one latest.pt for $WIDTH seed $SEED; found ${#RESUME_CHECKPOINTS[@]}"
      continue
    fi

    REFERENCE_RESUME_TRAIN_ARGS=(
      --config
      "$BASE"
      "${GRANULARITY_OVERRIDES[@]}"
      --output-root
      "$REFERENCE_ROOT/$WIDTH/s$SEED"
      --override
      "run.run_id=tiny-instruct-portfolio-r${REFERENCE_B_MULTIPLIER}-${WIDTH}-s${SEED}"
      --override
      "run.seed=$SEED"
      --override
      "run.granularity=$WIDTH"
      "${REFERENCE_OVERRIDES[@]}"
    )

    if ! "$PYTHON_BIN" train.py "${REFERENCE_RESUME_TRAIN_ARGS[@]}" --preflight; then
      print -u2 "Resume preflight failed for $WIDTH seed $SEED"
      continue
    fi

    REFERENCE_RESUME_SUBMIT=(
      sbatch
      "${SBATCH_EXCLUDE_ARGS[@]}"
      "--job-name=portfolio-r${REFERENCE_B_MULTIPLIER}-${WIDTH}-s${SEED}"
      "--time=$REFERENCE_RESUME_TIME"
      ./scripts/slurm_tinystories_controlled.sh
      --python-bin
      "$PYTHON_BIN"
      "${REFERENCE_RESUME_TRAIN_ARGS[@]}"
    )
    "${REFERENCE_RESUME_SUBMIT[@]}"
  done
done
```

If a run failed before producing `latest.pt`, it is not resumable. Archive its
partial directory and relaunch it as a fresh run instead. Contract mismatches
are rejected rather than silently starting from incompatible state.

## 4. Analyze every completed candidate in the lane

This command reads the candidate result tree and groups completed runs by the
immutable arm saved in each resolved config. It writes one report and one
holdout-selection manifest per discovered arm. It ignores unfinished runs, so
it is safe to repeat as jobs finish. It does not use `$VARIANT`, `$ARM_ID`, or
any other candidate-selection variable; after opening a new shell, rerun only
the setup block in section 1 before this section.

```bash
"$PYTHON_BIN" scripts/analyze_tinystories_portfolio_catchup.py portfolio-catchup \
  --runs-root "$LANE_ROOT/candidates" --target-manifest "$TARGET_MANIFEST" \
  --output-dir "$LANE_ROOT/analysis/candidates"
```

Runs from an older launch that have no saved `comparison_arm_id` cannot safely
be classified after the fact. The analyzer stops and lists them instead of
guessing a policy; relaunch those candidates from section 2 if they are needed.

## 5. Submit holdout jobs for the discovered selections

This submits one bundle-wide `sbatch` job. Inside that allocation, every arm
manifest runs sequentially in a fresh Python process. A failed manifest is
recorded without preventing later arms from running; after attempting all
manifests, the job exits nonzero if any failed. `--skip-existing` reuses valid
results, so resubmitting after a failure or time limit evaluates only missing or
stale checkpoints. Do not submit a second bundle holdout while one is active.

The sequential job also prevents different arms from concurrently writing the
same standalone-reference holdout result. Wait for it to finish before
continuing to section 6.

```bash
HOLDOUT_SELECTIONS=("$LANE_ROOT"/analysis/candidates/*/final_holdout_selection_manifest.json(N))
(( ${#HOLDOUT_SELECTIONS[@]} > 0 )) || { print -u2 'No completed candidate selections found'; return 1; }

HOLDOUT_MANIFEST_ARGS=()
for HOLDOUT_SELECTION in "${HOLDOUT_SELECTIONS[@]}"; do
  HOLDOUT_MANIFEST_ARGS+=(--final-holdout-manifest "$HOLDOUT_SELECTION")
done

HOLDOUT_BUNDLE_SUBMIT=(
  sbatch
  "${SBATCH_EXCLUDE_ARGS[@]}"
  "--job-name=portfolio-r${REFERENCE_B_MULTIPLIER}-holdouts"
  --time=24:00:00
  ./scripts/slurm_tinystories_controlled.sh
  --python-bin
  "$PYTHON_BIN"
  "${HOLDOUT_MANIFEST_ARGS[@]}"
)
"${HOLDOUT_BUNDLE_SUBMIT[@]}"
```

## 6. Report completed holdouts and render figures

After the jobs from section 5 have completed, this processes every discovered
arm. Re-running it is safe after additional holdout jobs finish.

```bash
HOLDOUT_SELECTIONS=("$LANE_ROOT"/analysis/candidates/*/final_holdout_selection_manifest.json(N))
(( ${#HOLDOUT_SELECTIONS[@]} > 0 )) || { print -u2 'No completed candidate selections found'; return 1; }

for HOLDOUT_SELECTION in "${HOLDOUT_SELECTIONS[@]}"; do
  ARM_ANALYSIS="${HOLDOUT_SELECTION:h}"
  ARM_ID="${ARM_ANALYSIS:t}"
  "$PYTHON_BIN" scripts/analyze_tinystories_portfolio_catchup.py final-holdout \
    --selection-manifest "$HOLDOUT_SELECTION" \
    --output-dir "$LANE_ROOT/final-holdout-analysis/$ARM_ID"
  "$PYTHON_BIN" scripts/make_figures.py --input "$LANE_ROOT" \
    --comparison-manifest "$ARM_ANALYSIS/portfolio_catchup_report.json" \
    --output "$LANE_ROOT/figures/$ARM_ID"
done
```

A selected seed subset yields a provisional report. Restore
`PORTFOLIO_SEEDS="42 43 44"` for the complete three-seed comparison. To add
another candidate, select a different `$VARIANT` and/or `X`, rerun this section,
and leave the frozen reference lane unchanged.
