# TinyStories-Instruct plateau and H=1 catch-up experiment

This is the primary operational workflow. Its hypothesis is deliberately
narrow: first establish a repeated-data plateau for standalone d64/l4 g1000,
freeze its per-seed best ordinary-validation targets, then identify when a
four-width elastic model with uniform global `H=1` matches them inside the same
declared optimizer-token budget.

Standalone and elastic catch-up runs use the same three-epoch declared token
budget and therefore the same cosine-schedule horizon. Plateau is an event
detected inside that matched horizon; it is not a shorter standalone budget.

The pass threshold is locked at 0.5% perplexity, represented in loss space as
`log(1.005)`. Standalone plateau detection requires two consecutive
quarter-epoch windows below that improvement, starting only after one complete
epoch, with no later best-loss improvement beyond the same tolerance. Elastic
catch-up requires five consecutive qualifying g1000 ordinary validations. No
smaller-width loss makes a primary selection or pass decision.

Completed d64/l3 evidence fixes the starting optimizer recipe to learning rate
0.008, cosine decay, 64 warmup updates, and 8,192 tokens per update. This
calibration deliberately moves the architecture to d64/l4. The earlier
unique-data run was still improving at its limit; it motivates repeating the
fixed optimizer set, but it is neither l4 plateau evidence nor evidence that a
plateau has already been found.

Historical eight-width/four-width campaigns, Thompson and PanelGrad runs, the
old capacity search, d128 results, preparation details, and optional diagnostics
are preserved in [the controlled-convergence reference](tinystories-controlled-convergence-reference.md).

## 1. Select and verify the Instruct data profile

Run from the repository root in the environment used for training:

```bash
export TINYSTORIES_PROFILE=instruct
source scripts/select_tinystories_profile.sh

test "$BASE" = configs/controlled_exps/tinystories_instruct_plateau.yaml
test -r "$TOKENIZER/tokenizer_manifest.json"
test -r "$CORPUS/corpus_manifest.json"
test "$DATASET_NAME" = roneneldan/TinyStoriesInstruct
```

The immutable corpus contains 5,576,491 optimizer sequences. With one GPU,
batch size 64, accumulation 1, and context 128, the largest complete optimizer-
step prefix is 5,576,448 sequences (713,785,344 tokens); the 43-sequence tail is
always excluded. Budgets are derived only from that aligned epoch:

```bash
export EPOCH_SAMPLES=5576448
export EPOCH_TOKENS=713785344
export TOKENS_PER_UPDATE=8192
export UPDATES_PER_EPOCH=$((EPOCH_TOKENS / TOKENS_PER_UPDATE))
export MATCHED_BUDGET=$((3 * EPOCH_TOKENS))
export ELASTIC_SCREEN_BUDGET=$EPOCH_TOKENS

test "$UPDATES_PER_EPOCH" -eq 87132
test "$MATCHED_BUDGET" -eq 2141356032
```

Use isolated roots and IDs. A changed horizon must start under a fresh run ID
and root: changing `training.token_budget` refits the cosine schedule, so a
longer run is not a compatible continuation of a shorter one.

```bash
export EXPERIMENT_ROOT="$MATFORMER_EXPERIMENT_ROOT/tinystories-instruct-plateau-catchup-v1"
export MATCHED_RUNS_ROOT="$EXPERIMENT_ROOT/matched-repeat-3epochs"
export STANDALONE_ROOT="$MATCHED_RUNS_ROOT/standalone-d64-l4-g1000-lr0p008"
export ELASTIC_SCREEN_ROOT="$EXPERIMENT_ROOT/elastic-repeat-1epoch-4g-h1-lr-screen-s42"
export ELASTIC_CATCHUP_ROOT="$MATCHED_RUNS_ROOT/elastic-4g-h1-matched-seeds"
export ANALYSIS_ROOT="$EXPERIMENT_ROOT/analysis"
export FIGURES_ROOT="$EXPERIMENT_ROOT/figures-matched-3epochs"
export SEED="${SEED:-42}"
mkdir -p "$STANDALONE_ROOT" "$ELASTIC_SCREEN_ROOT" "$ELASTIC_CATCHUP_ROOT" "$ANALYSIS_ROOT" "$FIGURES_ROOT" logs
```

## 2. Preflight and launch a standalone discovery seed

The primary discovery seed defaults to 42. `SEED` is deliberately part of the
run ID and every seed override, so the same block can be reused by exporting a
different seed first. This does not change the locked confirmation set of 42,
43, and 44. Define the ID once:

```bash
export STANDALONE_RUN_ID="tiny-instruct-standalone-repeat-3epochs-d64-l4-g1000-lr0p008-s$SEED"
```

The preflight must report `repeat_epochs`, `deterministic_per_epoch`, exactly
three complete epochs, 5,576,448 aligned samples, 713,785,344 aligned tokens, and
43 excluded samples.

```bash
"$PYTHON_BIN" train.py \
  --config "$BASE" \
  --output-root "$STANDALONE_ROOT" \
  --override "run.run_id=$STANDALONE_RUN_ID" \
  --override "model.tokenizer_dir=$TOKENIZER" \
  --override "dataset.prepared_corpus_dir=$CORPUS" \
  --override "training.token_budget=$MATCHED_BUDGET" \
  --override "run.seed=$SEED" \
  --preflight
```

Launch the exact same contract:

```bash
sbatch \
  --job-name="tiny-instruct-standalone-3ep-s$SEED" \
  --time=36:00:00 \
  scripts/slurm_tinystories_controlled.sh \
  --python-bin "$PYTHON_BIN" \
  --config "$BASE" \
  --output-root "$STANDALONE_ROOT" \
  --override "run.run_id=$STANDALONE_RUN_ID" \
  --override "model.tokenizer_dir=$TOKENIZER" \
  --override "dataset.prepared_corpus_dir=$CORPUS" \
  --override "training.token_budget=$MATCHED_BUDGET" \
  --override "run.seed=$SEED"
```

An interrupted job may be resubmitted only with the identical command and run
ID. Sampler, optimizer, scheduler, model, metrics, and RNG state resume from the
same logical repeated stream.

## 3. Analyze the discovery seed before spending the confirmation budget

```bash
export PLATEAU_ANALYSIS="$ANALYSIS_ROOT/standalone-plateau"

"$PYTHON_BIN" scripts/analyze_tinystories_plateau_catchup.py plateau \
  --runs-root "$STANDALONE_ROOT" \
  --seed "$SEED" \
  --output-dir "$PLATEAU_ANALYSIS"
```

The analyzer searches recursively because the trainer inserts its resolved
`run.output_group` between the supplied output root and run ID. With only the
discovery seed, `plateau_report.json` correctly reports the other confirmation
seeds as missing and does not emit `frozen_standalone_targets.json`. Inspect the
seed's `contract_satisfied` field, `plateau_onset_tokens`,
`plateau_confirmation_tokens`, the window improvements, trailing-five mean,
checkpoint path/hash, and `plateau.png`. Stop if the discovery seed fails.

## 4. Confirm seeds 43 and 44, then freeze standalone targets

```bash
for SEED in 43 44; do
  export SEED
  STANDALONE_RUN_ID="tiny-instruct-standalone-repeat-3epochs-d64-l4-g1000-lr0p008-s$SEED"
  sbatch \
    --job-name="tiny-instruct-standalone-3ep-s$SEED" \
    --time=36:00:00 \
    scripts/slurm_tinystories_controlled.sh \
    --python-bin "$PYTHON_BIN" \
    --config "$BASE" \
    --output-root "$STANDALONE_ROOT" \
    --override "run.run_id=$STANDALONE_RUN_ID" \
    --override "model.tokenizer_dir=$TOKENIZER" \
    --override "dataset.prepared_corpus_dir=$CORPUS" \
    --override "training.token_budget=$MATCHED_BUDGET" \
    --override "run.seed=$SEED"
done
```

After both runs complete:

```bash
"$PYTHON_BIN" scripts/analyze_tinystories_plateau_catchup.py plateau \
  --runs-root "$STANDALONE_ROOT" \
  --output-dir "$PLATEAU_ANALYSIS"

export FROZEN_TARGETS="$PLATEAU_ANALYSIS/frozen_standalone_targets.json"
test -r "$FROZEN_TARGETS"
```

The target manifest appears only when all three completed runs satisfy the
plateau and trailing-stability contract with matching corpus, aligned set,
ordering policy, tokenizer, and model provenance.

## 5. Screen the four H=1 elastic learning rates for one epoch

The elastic grid is exactly `g250,g500,g750,g1000`; uniform global `H=1`
redraws a width after every successful update. The primary LR endpoint is the
best ordinary-validation g1000 loss. Smaller widths are diagnostics only.

```bash
for LR in 0.004 0.006 0.008 0.010; do
  LR_SLUG="${LR/./p}"
  "$PYTHON_BIN" train.py \
    --config "$BASE" \
    --output-root "$ELASTIC_SCREEN_ROOT/lr-$LR_SLUG" \
    --override "run.run_id=tiny-instruct-elastic-repeat-1epoch-4g-h1-lr${LR_SLUG}-s42" \
    --override "model.tokenizer_dir=$TOKENIZER" \
    --override "dataset.prepared_corpus_dir=$CORPUS" \
    --override "training.token_budget=$ELASTIC_SCREEN_BUDGET" \
    --override "training.learning_rate=$LR" \
    --override run.seed=42 \
    --override run.model_family=nested \
    --override run.sampling_mode=nested-random \
    --override run.granularity=null \
    --override model.granularity_sampling_mode=global \
    --override model.global_sampling_schedule=random_with_replacement \
    --override model.global_sampling_interval_steps=1 \
    --preflight
done
```

Launch the four jobs after all preflights pass:

```bash
for LR in 0.004 0.006 0.008 0.010; do
  LR_SLUG="${LR/./p}"
  sbatch \
    --job-name="tiny-instruct-elastic-1ep-h1-lr$LR_SLUG" \
    --time=24:00:00 \
    scripts/slurm_tinystories_controlled.sh \
    --python-bin "$PYTHON_BIN" \
    --config "$BASE" \
    --output-root "$ELASTIC_SCREEN_ROOT/lr-$LR_SLUG" \
    --override "run.run_id=tiny-instruct-elastic-repeat-1epoch-4g-h1-lr${LR_SLUG}-s42" \
    --override "model.tokenizer_dir=$TOKENIZER" \
    --override "dataset.prepared_corpus_dir=$CORPUS" \
    --override "training.token_budget=$ELASTIC_SCREEN_BUDGET" \
    --override "training.learning_rate=$LR" \
    --override run.seed=42 \
    --override run.model_family=nested \
    --override run.sampling_mode=nested-random \
    --override run.granularity=null \
    --override model.granularity_sampling_mode=global \
    --override model.global_sampling_schedule=random_with_replacement \
    --override model.global_sampling_interval_steps=1
done
```

Freeze the winner:

```bash
export LR_SELECTION_ANALYSIS="$ANALYSIS_ROOT/elastic-lr-selection"

"$PYTHON_BIN" scripts/analyze_tinystories_plateau_catchup.py select-elastic-lr \
  --runs-root "$ELASTIC_SCREEN_ROOT" \
  --frozen-targets "$FROZEN_TARGETS" \
  --output-dir "$LR_SELECTION_ANALYSIS"

export ELASTIC_SELECTION="$LR_SELECTION_ANALYSIS/elastic_lr_selection.json"
export ELASTIC_LR="$(jq -r '.selected_learning_rate' "$ELASTIC_SELECTION")"
test -n "$ELASTIC_LR"
```

## 6. Launch three matched-seed H=1 runs for three epochs

These are fresh three-epoch runs. Do not resume the one-epoch LR-screen winner,
because its shorter budget created a different cosine schedule.

```bash
for SEED in 42 43 44; do
  LR_SLUG="${ELASTIC_LR/./p}"
  sbatch \
    --job-name="tiny-instruct-elastic-3ep-h1-s$SEED" \
    --time=36:00:00 \
    scripts/slurm_tinystories_controlled.sh \
    --python-bin "$PYTHON_BIN" \
    --config "$BASE" \
    --output-root "$ELASTIC_CATCHUP_ROOT/lr-$LR_SLUG" \
    --override "run.run_id=tiny-instruct-elastic-repeat-3epochs-4g-h1-lr${LR_SLUG}-s$SEED" \
    --override "model.tokenizer_dir=$TOKENIZER" \
    --override "dataset.prepared_corpus_dir=$CORPUS" \
    --override "training.token_budget=$MATCHED_BUDGET" \
    --override "training.learning_rate=$ELASTIC_LR" \
    --override "run.seed=$SEED" \
    --override run.model_family=nested \
    --override run.sampling_mode=nested-random \
    --override run.granularity=null \
    --override model.granularity_sampling_mode=global \
    --override model.global_sampling_schedule=random_with_replacement \
    --override model.global_sampling_interval_steps=1
done
```

## 7. Measure catch-up before opening the sealed holdout

```bash
export CATCHUP_ANALYSIS="$ANALYSIS_ROOT/elastic-catchup"

"$PYTHON_BIN" scripts/analyze_tinystories_plateau_catchup.py catchup \
  --runs-root "$ELASTIC_CATCHUP_ROOT" \
  --frozen-targets "$FROZEN_TARGETS" \
  --elastic-selection "$ELASTIC_SELECTION" \
  --output-dir "$CATCHUP_ANALYSIS"

test -r "$CATCHUP_ANALYSIS/catchup_report.json"
test -r "$CATCHUP_ANALYSIS/catchup_by_seed.csv"
test -r "$CATCHUP_ANALYSIS/catchup.png"
```

Inspect the signed token deltas, nonnegative additional budgets, both ratios,
and smaller-width diagnostics. If any seed has no five-point streak by three
epochs, the report marks it censored and refuses a general cross-seed claim.

Generate the repository-standard matched curves from only the two three-epoch
families. The one-epoch LR screen is outside `$MATCHED_RUNS_ROOT`, so it cannot
contaminate these plots:

```bash
"$PYTHON_BIN" scripts/make_figures.py \
  --input "$MATCHED_RUNS_ROOT" \
  --output "$FIGURES_ROOT" \
  --validation-loss-log-y

test -r "$FIGURES_ROOT/validation_loss_over_tokens_granularity_comparison.png"
```

The g1000 panel overlays standalone and elastic validation loss against the
same total-token x-axis. The remaining panels show elastic widths
diagnostically. Curves are aggregated across completed seeds by their saved,
seed-independent experiment contracts.

Only after the plateau targets, LR choice, and catch-up report are frozen may
the final holdout be opened. Evaluate the exact run directories recorded in the
two manifests/report; for example:

```bash
mapfile -t ELASTIC_RUN_DIRS < <(jq -r '.seeds[].run_dir' "$CATCHUP_ANALYSIS/catchup_report.json")
mapfile -t STANDALONE_RUN_DIRS < <(jq -r '.targets | to_entries | sort_by(.key | tonumber) | .[].value.run_dir' "$FROZEN_TARGETS")

for RUN_DIR in \
  "${STANDALONE_RUN_DIRS[@]}" \
  "${ELASTIC_RUN_DIRS[@]}"; do
  sbatch scripts/slurm_tinystories_controlled.sh \
    --python-bin "$PYTHON_BIN" \
    --final-holdout-only "$RUN_DIR"
done
```

## 8. Terminal outcomes

The experiment ends in exactly one of these states:

1. **Robust plateau and catch-up**: seeds 42, 43, and 44 all freeze standalone
   targets and all three elastic runs achieve a five-evaluation streak. Report
   per-seed additional budgets plus their mean, median, minimum, and maximum.
2. **Robust plateau, censored H=1**: all standalone targets freeze, but at least
   one elastic seed has no qualifying streak by three epochs. Report censoring;
   make no general catch-up claim.
3. **No robust standalone plateau within three epochs**: seed 42 fails, any
   confirmation seed fails, or cross-seed provenance disagrees. Do not freeze
   targets, screen elastic LR, or launch catch-up runs.

Any proposal to extend either horizon is a new experiment with a fresh output
root and run IDs, because the changed token budget refits cosine decay.
