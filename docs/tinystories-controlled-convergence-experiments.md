# TinyStories controlled-convergence experiments

This experiment first identifies a converged dense pretraining recipe, then
freezes that recipe for later standalone-width and elastic comparisons. All
runs use the normal `train.py --config ... --override ...` pipeline and the
same immutable `packed_mmap` corpus. No story or packed sequence is repeated.

The dense model is a four-layer Llama decoder with `d_model=128`, four heads,
a 512-unit SwiGLU FFN, context length 128, and a 2,048-token vocabulary. The
full-width model has roughly 1.6M parameters and consumes 8,192 tokens per
optimizer update on one GPU.

Run the commands from the repository root in the validated environment:

```bash
conda activate elasticnn
export PYTHON_BIN="$(command -v python)"

export NFS_USER_ROOT="${NFS_USER_ROOT:-/nfs-stor/$USER}"
export MATFORMER_TOKENIZER_ROOT="${MATFORMER_TOKENIZER_ROOT:-$NFS_USER_ROOT/matformer-tokenizers}"
export MATFORMER_CORPUS_ROOT="${MATFORMER_CORPUS_ROOT:-$NFS_USER_ROOT/matformer-corpora}"
export MATFORMER_EXPERIMENT_ROOT="${MATFORMER_EXPERIMENT_ROOT:-$NFS_USER_ROOT/results/elasticnn}"

export TINYSTORIES_TOKENIZER_NAME="${TINYSTORIES_TOKENIZER_NAME:-tinystories-sentencepiece-bpe-2k-v1}"
export TINYSTORIES_CORPUS_NAME="${TINYSTORIES_CORPUS_NAME:-tinystories-packed-33m-v1}"
export TINYSTORIES_EXPERIMENT_NAME="${TINYSTORIES_EXPERIMENT_NAME:-tinystories-controlled-convergence-v1}"
```

Override any of these environment variables before running the guide to use a
different NFS mount, artifact root, or experiment name.

## 1. Prepare TinyStories once

The preparation command downloads the pinned `roneneldan/TinyStories` train
and validation splits through the Hugging Face `datasets` interface. Keep the
download cache and the immutable prepared artifacts on NFS; `$USER` is expanded
by the shell, so these paths do not embed a particular account name.

The tokenizer uses the first 50,000 optimizer-eligible train stories. Those
same stories remain part of the optimizer corpus; preparation fails instead of
publishing if the token cap would exclude any of them. Tokenizer preparation
introduces no extra source documents.

```bash
export HF_HOME="${HF_HOME:-$NFS_USER_ROOT/huggingface}"
export TOKENIZER="$MATFORMER_TOKENIZER_ROOT/$TINYSTORIES_TOKENIZER_NAME"
export CORPUS="$MATFORMER_CORPUS_ROOT/$TINYSTORIES_CORPUS_NAME"
mkdir -p "$HF_HOME" "$MATFORMER_TOKENIZER_ROOT" "$MATFORMER_CORPUS_ROOT"

python scripts/prepare_tinystories.py \
  --tokenizer-dir "$TOKENIZER" \
  --corpus-dir "$CORPUS" \
  --tokenization-workers 4

python scripts/audit_prepared_corpus.py \
  --prepared-corpus-dir "$CORPUS" \
  --prepared-tokenizer-dir "$TOKENIZER" \
  --minimum-training-tokens 33554432 \
  --required-vocab-size 2048
```

Each non-empty Hugging Face `text` row is one complete story. The source has a
small number of blank rows; preparation skips them deterministically while
retaining each story's physical split-row index in its manifested identity.
Preparation preserves source order and assigns roles as follows:

- First 128 non-empty train stories: controller role.
- Remaining non-empty train stories: optimizer role, capped at exactly
  33,554,432 unique tokens (262,144 packed sequences).
- First 128 non-empty validation stories: ordinary validation.
- Next 512 non-empty validation stories: final holdout.

Every role is packed independently with EOS separators. Only excess tokens
after the final optimizer boundary are discarded.

## 2. Preflight the dense grid

Set the central config and artifact roots. Do not change seed, batch size,
warmup, data, or validation cadence across grid jobs.

```bash
export BASE=configs/controlled_exps/tinystories_controlled_convergence.yaml
export OUT="$MATFORMER_EXPERIMENT_ROOT/$TINYSTORIES_EXPERIMENT_NAME"
export SLURM_LOG_ROOT="./logs"
mkdir -p "$OUT" "$SLURM_LOG_ROOT"

COMMON_OVERRIDES=(
  --override "model.tokenizer_dir=$TOKENIZER"
  --override "dataset.prepared_corpus_dir=$CORPUS"
  --override "run.output_root=$OUT"
)
```

Resolve all six configurations without training:

```bash
for LR in 3e-4 1e-3 3e-3; do
  for SCHEDULER in cosine constant_with_warmup; do
    RUN_ID="tinystories-dense-lr${LR}-sched${SCHEDULER}-2048-s42"
    python train.py \
      --config "$BASE" \
      --preflight \
      "${COMMON_OVERRIDES[@]}" \
      --override "run.run_id=$RUN_ID" \
      --override "training.learning_rate=$LR" \
      --override "training.scheduler.name=$SCHEDULER"
  done
done
```

Each preflight must report one GPU, 8,192 tokens/update, 16,777,216 total
tokens, 2,048 updates, 64 warmup updates, standalone `g1000`, and the same
corpus/tokenizer hashes.

## 3. Submit six independent dense jobs

The loop below issues six independent `sbatch` submissions. It is not a job
array or an in-process sweep; each grid point has its own run ID, checkpoint,
resume state, and Slurm lifecycle.

```bash
for LR in 3e-4 1e-3 3e-3; do
  for SCHEDULER in cosine constant_with_warmup; do
    RUN_ID="tinystories-dense-lr${LR}-sched${SCHEDULER}-2048-s42"
    sbatch \
      --output="$SLURM_LOG_ROOT/%x_%j.out" \
      --error="$SLURM_LOG_ROOT/%x_%j.err" \
      scripts/slurm_tinystories_controlled.sh \
      --config "$BASE" \
      "${COMMON_OVERRIDES[@]}" \
      --override "run.run_id=$RUN_ID" \
      --override "training.learning_rate=$LR" \
      --override "training.scheduler.name=$SCHEDULER"
  done
done
```

Resubmit a preempted job with the identical command and run ID. The existing
latest checkpoint and packed sampler cursor provide exact resume. The launcher
does not evaluate final holdout after grid jobs.

## 4. Select a converged recipe

Analyze only ordinary validation:

```bash
python scripts/analyze_tinystories_convergence.py \
  --runs-root "$OUT" \
  --output-dir "$OUT/convergence-analysis"
```

The command writes `selection_report.json` and `run_comparison.csv`. It ignores
evaluations before step 512, requires five final evaluations without a new
0.5% relative best, and rejects incomplete/non-finite runs. It first ranks all
stable runs by best validation loss. A recipe is selected only when the
globally best stable run has converged; runs within 0.1% relative loss are tied
and lower wall time wins. If the globally best stable run has not converged,
the two lowest-loss stable runs become fallback candidates even when a worse
run has plateaued.

If the report status is `fallback_required`, rerun its two
`fallback_candidates` from initialization using their original learning rate
and scheduler, a new `-4096-s42` run ID, and:

```bash
--override training.token_budget=33554432
```

The unchanged seed recreates the initialization. The larger token budget
re-fits the scheduler over 4,096 updates; do not resume a completed 2,048-step
cosine schedule. Re-run the analyzer over the root after both fallback jobs.
If its status remains `fallback_required` or becomes `no_stable_recipe`, stop:
there is no recipe to freeze.

## 5. Freeze and evaluate the winner

For `recipe_selected`, update the same central config with the values in
`frozen_recipe_overrides`:

- Set `training.learning_rate`.
- Set `training.scheduler.name`.
- Set `training.token_budget` to 16,777,216 or 33,554,432 as selected.
- Set `controlled_experiment.recipe_status: frozen`.
- Record `winner.run_id` and `report_hash` in the two provenance fields.

Freeze the complete winning horizon, not its best-checkpoint step. Then expose
the sealed final holdout exactly once for the selected run:

```bash
export SELECTION_REPORT="$OUT/convergence-analysis/selection_report.json"
export WINNER_RUN_DIR="$(
  "$PYTHON_BIN" -c \
    'import json, pathlib, sys; print(json.loads(pathlib.Path(sys.argv[1]).read_text())["winner"]["run_dir"])' \
    "$SELECTION_REPORT"
)"
sbatch \
  --output="$SLURM_LOG_ROOT/%x_%j.out" \
  --error="$SLURM_LOG_ROOT/%x_%j.err" \
  scripts/slurm_tinystories_controlled.sh \
  --final-holdout-only "$WINNER_RUN_DIR"
```

The existing evaluator uses the checkpoint chosen by ordinary validation and
writes `final_holdout_results.json`. Final-holdout values never participate in
recipe selection.

## 6. Later experiments from the frozen config

These commands are templates for the next experimental stage; they are not
part of the dense calibration grid.

Train a standalone granularity by changing only its run identity and width:

```bash
sbatch \
  --output="$SLURM_LOG_ROOT/%x_%j.out" \
  --error="$SLURM_LOG_ROOT/%x_%j.err" \
  scripts/slurm_tinystories_controlled.sh \
  --config "$BASE" \
  "${COMMON_OVERRIDES[@]}" \
  --override run.run_id=tinystories-standalone-g250-s42 \
  --override run.model_family=standalone \
  --override run.sampling_mode=standalone \
  --override run.granularity=g250
```

Use `g125`, `g250`, `g500`, or `g1000` in separate jobs.

Train all four granularities with IID global sampling (`H=1`):

```bash
sbatch \
  --output="$SLURM_LOG_ROOT/%x_%j.out" \
  --error="$SLURM_LOG_ROOT/%x_%j.err" \
  scripts/slurm_tinystories_controlled.sh \
  --config "$BASE" \
  "${COMMON_OVERRIDES[@]}" \
  --override run.run_id=tinystories-elastic-iid-h1-s42 \
  --override run.model_family=nested \
  --override run.sampling_mode=nested-random \
  --override run.granularity=null \
  --override model.granularity_sampling_mode=global \
  --override model.global_sampling_schedule=random_with_replacement \
  --override model.global_sampling_interval_steps=1
```

For windowed global sampling, change `global_sampling_interval_steps` to the
desired `H>1`. To use exact balanced exposure, additionally set:

```bash
--override model.global_sampling_schedule=balanced_cycle
```

Balanced-cycle jobs require the frozen update count to be divisible by `4H`.
Keep data, seed, optimizer, scheduler, warmup, validation cadence, and token
budget inherited from the frozen central config.
