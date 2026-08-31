# TinyStories-Instruct Per-Width Optimizer-State Experiment

This runbook defines the frozen shared-versus-per-width optimizer-state study.
The two arms share model weights, batches, global width actions, optimizer
hyperparameters, and one global learning-rate clock. The intervention is only
`training.optimizer.state_scope` (`shared` or `per_granularity`).

For a copy-paste six-arm execution guide, see
[run-per-width-optimizer-experiments.md](./run-per-width-optimizer-experiments.md).

## Corpus audit

Select the prepared instruct profile and audit its four immutable roles before
submitting either arm:

```bash
export TINYSTORIES_PROFILE=instruct
source scripts/select_tinystories_profile.sh
python scripts/audit_prepared_corpus.py \
  --prepared-corpus-dir "$CORPUS" \
  --prepared-tokenizer-dir "$TOKENIZER" \
  --required-vocab-size 2048 \
  --minimum-training-tokens 713785344
```

The optimizer-training, controller, ordinary-validation, and final-holdout
manifest hashes must remain fixed across all runs.

## Pilot preflight and execution

The pilot consists of seeds 42, 43, and 44 in both scopes. For each seed and
scope, use an isolated run ID/output directory and run preflight first:

```bash
python train.py \
  --config configs/controlled_exps/tinystories_instruct_per_width_optimizers.yaml \
  --override run.seed=42 \
  --override run.run_id=optimizer-state-shared-s42 \
  --override model.tokenizer_dir="$TOKENIZER" \
  --override dataset.prepared_corpus_dir="$CORPUS" \
  --override training.token_budget=713785344 \
  --override training.optimizer.state_scope=shared \
  --output-dir "$PILOT_ROOT/optimizer-state-shared-s42" \
  --preflight
```

Repeat for both scopes and all seeds, then remove `--preflight` to execute.
Every preflight must show 87,132 global steps, 21,783 updates per width,
balanced-cycle `H=1`, AdamW 0.008, cosine warmup 64, evaluation every 64 steps,
one process, and matching paired-control signatures within each seed. The
prepared-corpus data seed remains the frozen value 42; `run.seed` supplies the
three model/action seeds.

Submit each preflighted command through the existing one-GPU launcher. This is
the shared-arm seed-42 form; replace the seed, run ID, scope, and output
directory for the other five arms:

```bash
sbatch --job-name=optimizer-state-shared-s42 \
  scripts/slurm_tinystories_controlled.sh \
  --python-bin "$HOME/.conda/envs/elasticnn/bin/python" \
  --config configs/controlled_exps/tinystories_instruct_per_width_optimizers.yaml \
  --override run.seed=42 \
  --override run.run_id=optimizer-state-shared-s42 \
  --override model.tokenizer_dir="$TOKENIZER" \
  --override dataset.prepared_corpus_dir="$CORPUS" \
  --override training.token_budget=713785344 \
  --override training.optimizer.state_scope=shared \
  --output-dir "$PILOT_ROOT/optimizer-state-shared-s42"
```

Resubmit the identical command and output directory after an interruption so
the normal continuation path restores the complete optimizer state. Never run
the same run ID concurrently.

## Freeze and diagnostic report

Freeze only the six explicit completed run directories; discovery by directory
name is intentionally unsupported:

```bash
python scripts/analyze_tinystories_per_width_optimizer.py freeze \
  --phase pilot \
  --run-dir "$PILOT_ROOT/optimizer-state-shared-s42" \
  --run-dir "$PILOT_ROOT/optimizer-state-per-granularity-s42" \
  --run-dir "$PILOT_ROOT/optimizer-state-shared-s43" \
  --run-dir "$PILOT_ROOT/optimizer-state-per-granularity-s43" \
  --run-dir "$PILOT_ROOT/optimizer-state-shared-s44" \
  --run-dir "$PILOT_ROOT/optimizer-state-per-granularity-s44" \
  --output-dir "$PILOT_ROOT/analysis"

python scripts/analyze_tinystories_per_width_optimizer.py report \
  --manifest "$PILOT_ROOT/analysis/optimizer_state_manifest.json" \
  --output-dir "$PILOT_ROOT/analysis"
```

The pilot report is `diagnostic`. It reports every width, uniform mean loss,
worst-width loss, paired deltas, wall time, peak accelerator memory, and
resumable-checkpoint bytes. `matched_compute_claim` remains false: tokens and
steps are matched, but measured operational cost is reported rather than
declared equal.

Each run directory must contain `config.json`, `metrics.csv`,
`run_summary.json`, and the terminal resumable
`checkpoints/latest.pt`. Freeze writes
`analysis/optimizer_state_manifest.json`; report writes
`analysis/optimizer_state_comparison.json` and
`analysis/optimizer_state_comparison.csv`, plus scope-aware validation,
endpoint, and resource figures:

- `analysis/optimizer_state_validation_loss_over_tokens.png`;
- `analysis/optimizer_state_endpoint_by_width.png`;
- `analysis/optimizer_state_resource_costs.png`.

## Holdout and confirmation decision

Keep the 512-example final holdout sealed during pilot analysis if confirmation
is possible. If the pilot holdout is opened, all later reuse is labeled
`descriptive_after_holdout_open`.

Confirmation starts from fresh initial states and run IDs; never continue pilot
checkpoints. Override `training.token_budget=2141356032`. Each confirmation run
must resolve 261,396 global steps and 65,349 updates per width. Freeze the six
confirmation directories before explicitly evaluating each frozen terminal
checkpoint:

```bash
python scripts/analyze_tinystories_per_width_optimizer.py freeze \
  --phase confirmation \
  --run-dir "$CONFIRM_ROOT/optimizer-state-confirmation-shared-s42" \
  --run-dir "$CONFIRM_ROOT/optimizer-state-confirmation-per-granularity-s42" \
  --run-dir "$CONFIRM_ROOT/optimizer-state-confirmation-shared-s43" \
  --run-dir "$CONFIRM_ROOT/optimizer-state-confirmation-per-granularity-s43" \
  --run-dir "$CONFIRM_ROOT/optimizer-state-confirmation-shared-s44" \
  --run-dir "$CONFIRM_ROOT/optimizer-state-confirmation-per-granularity-s44" \
  --output-dir "$CONFIRM_ROOT/analysis"
```

If the pilot holdout was opened, also submit every confirmation arm with
`--override controlled_experiment.holdout_opened_during_pilot=true`; otherwise
the confirmation provenance would falsely claim a sealed pilot holdout.

Evaluate each manifest-recorded terminal checkpoint explicitly:

```bash
python scripts/evaluate_final_holdout.py \
  --run-dir "$RUN_DIR" \
  --checkpoint "$RUN_DIR/checkpoints/latest.pt" \
  --device cuda \
  --skip-existing
```

Rerun `report` after all six explicit holdout results exist. Only a complete
three-seed confirmation with the pilot holdout untouched is labeled
`confirmatory`; incomplete results remain diagnostic.

## Endpoint interpretation

The primary seed-level endpoint is the candidate-minus-shared uniform mean
final-holdout loss. Secondary endpoints are each width's loss/perplexity,
worst-width loss, trailing-five ordinary-validation loss, update/exposure
reconciliation, and the three operational costs. The report preserves signed
seed-level differences and does not infer scientific superiority or invent an
unplanned hypothesis test.
