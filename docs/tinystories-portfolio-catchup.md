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

## Environment and isolated roots

```bash
export TINYSTORIES_PROFILE=instruct
source scripts/select_tinystories_profile.sh

export BASE=configs/controlled_exps/tinystories_instruct_portfolio_catchup.yaml
export PORTFOLIO_ROOT="$MATFORMER_EXPERIMENT_ROOT/tinystories-instruct-portfolio-catchup-v1"
export REFERENCE_ROOT="$PORTFOLIO_ROOT/reference-runs"
export CANDIDATE_ROOT="$PORTFOLIO_ROOT/elastic-candidate-runs"
export ANALYSIS_ROOT="$PORTFOLIO_ROOT/analysis"
export HOLDOUT_ROOT="$PORTFOLIO_ROOT/final-holdout-analysis"
export FIGURES_ROOT="$PORTFOLIO_ROOT/figures"
export B=713785344
export THREE_B=2141356032
export FOUR_B=2855141376

test "$B" -eq 713785344
test "$THREE_B" -eq 2141356032
test "$FOUR_B" -eq 2855141376
mkdir -p "$REFERENCE_ROOT" "$CANDIDATE_ROOT" \
  "$ANALYSIS_ROOT" "$HOLDOUT_ROOT" "$FIGURES_ROOT" logs
```

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

## Acquire the 12 standalone references

Preflight the complete matrix before submitting it:

```bash
for SEED in 42 43 44; do
  for WIDTH in g250 g500 g750 g1000; do
    RUN_ID="tiny-instruct-portfolio-${WIDTH}-s${SEED}"
    "$PYTHON_BIN" train.py \
      --config "$BASE" \
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
  for WIDTH in g250 g500 g750 g1000; do
    RUN_ID="tiny-instruct-portfolio-${WIDTH}-s${SEED}"
    sbatch \
      --job-name="portfolio-${WIDTH}-s${SEED}" \
      --time=24:00:00 \
      scripts/slurm_tinystories_controlled.sh \
      --python-bin "$PYTHON_BIN" \
      --config "$BASE" \
      --output-root "$REFERENCE_ROOT/$WIDTH/s$SEED" \
      --override "run.run_id=$RUN_ID" \
      --override "run.seed=$SEED" \
      --override "run.granularity=$WIDTH" \
      --override "model.tokenizer_dir=$TOKENIZER" \
      --override "dataset.prepared_corpus_dir=$CORPUS"
  done
done
```

After all 12 runs reach `B`, freeze their best ordinary-validation checkpoints:

```bash
export REFERENCE_ANALYSIS="$ANALYSIS_ROOT/references"

"$PYTHON_BIN" scripts/analyze_tinystories_portfolio_catchup.py \
  freeze-references \
  --runs-root "$REFERENCE_ROOT" \
  --output-dir "$REFERENCE_ANALYSIS"

export TARGET_MANIFEST="$REFERENCE_ANALYSIS/standalone_portfolio_targets.json"
export TARGET_HASH="$(jq -r '.manifest_hash' "$TARGET_MANIFEST")"
test -r "$REFERENCE_ANALYSIS/standalone_portfolio_targets.csv"
test -r "$REFERENCE_ANALYSIS/standalone_portfolio_diagnostics.json"
test -r "$REFERENCE_ANALYSIS/standalone_portfolio_diagnostics.png"
```

## Run the three fresh `3B` elastic candidates

The frozen target manifest is an immutable resume input. Catch-up saves the
first fifth-point confirmation checkpoint but never stops the run early.

```bash
for SEED in 42 43 44; do
  RUN_ID="tiny-instruct-portfolio-candidate-s${SEED}"
  "$PYTHON_BIN" train.py \
    --config "$BASE" \
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

## Verify catch-up, final holdout, and figures

```bash
export CATCHUP_ANALYSIS="$ANALYSIS_ROOT/portfolio-catchup"

"$PYTHON_BIN" scripts/analyze_tinystories_portfolio_catchup.py \
  portfolio-catchup \
  --runs-root "$CANDIDATE_ROOT" \
  --target-manifest "$TARGET_MANIFEST" \
  --output-dir "$CATCHUP_ANALYSIS"

export CATCHUP_REPORT="$CATCHUP_ANALYSIS/portfolio_catchup_report.json"
export HOLDOUT_SELECTION="$CATCHUP_ANALYSIS/final_holdout_selection_manifest.json"
test -r "$CATCHUP_ANALYSIS/portfolio_catchup.csv"
test -r "$CATCHUP_ANALYSIS/portfolio_joint_deficit.png"
test -r "$CATCHUP_ANALYSIS/portfolio_per_granularity_deficits.png"
test -r "$HOLDOUT_SELECTION"
```

The report uses the fifth simultaneous qualifying validation as `t*`, reports
`t*/B`, `t*/4B`, required savings, and separately reports the realized `75%`
full-run spend. A censored seed prevents the general claim.

The analyzer always seals an exact 15-checkpoint holdout manifest. Standalone
entries use their ordinary-validation best checkpoints. When all seeds confirm,
elastic entries use their immutable confirmation checkpoints and the holdout is
eligible for the predefined general claim. If any seed is censored, all three
elastic entries instead use their exact terminal `3B` checkpoints. That mode is
diagnostic only: it reports terminal holdout equivalence without creating
`t*`, savings, catch-up, or general portfolio-equivalence claims.

Inspect which selection rule was sealed:

```bash
jq '{
  status: .final_holdout_selection_status,
  mode: .final_holdout_selection_mode,
  claim_eligible: .final_holdout_claim_eligible
}' "$CATCHUP_REPORT"
```

Evaluate exactly the selected checkpoints in one Slurm allocation:

```bash
sbatch \
  --job-name=portfolio-final-holdout \
  --time=24:00:00 \
  scripts/slurm_tinystories_controlled.sh \
  --python-bin "$PYTHON_BIN" \
  --final-holdout-manifest "$HOLDOUT_SELECTION"
```

After all 15 results exist:

```bash
"$PYTHON_BIN" scripts/analyze_tinystories_portfolio_catchup.py \
  final-holdout \
  --selection-manifest "$HOLDOUT_SELECTION" \
  --output-dir "$HOLDOUT_ROOT"

test -r "$HOLDOUT_ROOT/portfolio_final_holdout_report.json"
test -r "$HOLDOUT_ROOT/portfolio_final_holdout.csv"
```

For confirmation-selected manifests, a general portfolio-equivalence claim is
emitted only if all 12 same-seed, same-width comparisons pass the `0.5%`
perplexity tolerance. For censored terminal-`3B` manifests, the report instead
sets `diagnostic_terminal_3B_equivalence`; its
`general_portfolio_equivalence_claim` remains false regardless of the holdout
result.

Generate only the manifest-declared mixed-budget figures:

```bash
"$PYTHON_BIN" scripts/make_figures.py \
  --input "$PORTFOLIO_ROOT" \
  --comparison-manifest "$CATCHUP_REPORT" \
  --output "$FIGURES_ROOT"

test -r "$FIGURES_ROOT/portfolio_validation_loss_over_tokens.png"
test -r "$FIGURES_ROOT/ppl_vs_size.png"
test -r "$FIGURES_ROOT/portfolio_per_granularity_deficits.png"
test -r "$FIGURES_ROOT/portfolio_worst_width_deficit.png"
test -r "$FIGURES_ROOT/learning_rate_schedule.png"
test -r "$FIGURES_ROOT/final_holdout_ppl_vs_size.png"
test -r "$FIGURES_ROOT/portfolio_final_holdout_deficit_vs_size.png"
```

Standalone curves end naturally at `B`; elastic curves continue to `3B`.
Labels include role, active width, budget, cosine scheduler, and seed count.
Target/tolerance bands and joint confirmation markers come from the immutable
comparison report. Both size plots use exact non-embedding parameter counts and
the manifest-selected checkpoints. A censored report remains fully plottable;
its elastic series is labeled as a claim-ineligible terminal-`3B` diagnostic and
has no confirmation marker. `portfolio_final_holdout_perplexity.png` is retained
as a compatibility alias of `final_holdout_ppl_vs_size.png`. Without
`--comparison-manifest`, figure generation retains its historical strict
same-contract behavior.
