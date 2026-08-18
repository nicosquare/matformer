# 10B unique-token slicing experiments

This is the primary operator guide for the production 10B experiments. Follow
it from a clean checkout of `main`, run every command from the repository root,
and use the prepared tokenizer and corpus locations supplied privately by the
experiment owner. Do not commit credentials, identities, or site-specific
filesystem paths.

Every production run consumes the same deterministic 10B-token prefix of one
immutable FineWeb `sample-100BT` corpus and the same immutable 256,000-entry
FineWeb SentencePiece tokenizer. The production schedule uses one node with
four GPUs and exact checkpoint resume.

## 1. Start from current `main`

```bash
git switch main
git pull --ff-only
git status --short
```

`git status --short` must print nothing before environment setup or submission.

## 2. Create the validated environment

The checked-in environment is intentionally path-neutral. It pins Python and
the packages used by training, FSDP, prepared-corpus loading, validation,
testing, and figure generation. Its pip phase installs the Linux x86-64
PyTorch wheel from the official PyTorch CUDA 12.8 index.

```bash
conda env create --file environment.yml
conda activate elasticnn
python -m pip check
```

Confirm the principal versions:

```bash
python -c 'import datasets, numpy, sentencepiece, torch, transformers; print("torch", torch.__version__, "cuda", torch.version.cuda); print("transformers", transformers.__version__); print("datasets", datasets.__version__); print("numpy", numpy.__version__); print("sentencepiece", sentencepiece.__version__)'
```

The validated values are Python `3.12.13`, PyTorch `2.11.0+cu128`, Transformers
`5.8.0`, Datasets `4.8.5`, NumPy `2.4.3`, and SentencePiece `0.2.1`.

The production configs disable W&B and downstream `lm-eval`; neither package
nor their credentials are required for these runs. Install them separately
only if those optional features are deliberately enabled.

## 3. Configure private runtime values

Copy the ignored environment template and fill in the three required values
using locations supplied out of band:

```bash
cp .env.template .env
```

```dotenv
TOKENIZER=<prepared-tokenizer-directory>
CORPUS=<prepared-corpus-directory>
OUT=<writable-output-root>
# Optional: SLURM_EXCLUDE=gpu-[05],gpu-09
```

Load the values into the submission shell and capture the active Python
executable. The Slurm jobs use this exact environment rather than assuming a
particular home-directory layout.

```bash
set -a
source .env
set +a

export PYTHON_BIN="$(command -v python)"
export BASE=configs/production/slicing_10b_base.yaml
export BAYES=configs/production/slicing_10b_bayesian.yaml

mkdir -p "$OUT" logs
test -r "$TOKENIZER/tokenizer_manifest.json"
test -r "$CORPUS/corpus_manifest.json"
test -w "$OUT"
```

Prepared-corpus training is local-only and does not need a Hugging Face token.
Optional Hugging Face cache and credential fields remain available in `.env`
for workflows that access remote datasets.

Define the arguments shared by every Slurm launch:

```bash
COMMON=(
  --output-root "$OUT"
  --python-bin "$PYTHON_BIN"
  --override "dataset.prepared_corpus_dir=$CORPUS"
  --override "model.tokenizer_dir=$TOKENIZER"
)

SBATCH=(sbatch)
if [[ -n "${SLURM_EXCLUDE:-}" ]]; then
  SBATCH+=(--exclude="$SLURM_EXCLUDE")
fi
```

`SLURM_EXCLUDE` accepts the same comma-separated node list and bracket syntax
as Slurm's `--exclude` option. Leave it unset to make no command-line override.
Define it in `.env` before constructing `SBATCH` so every submission in this
guide uses the same list. This command-line value replaces any
`#SBATCH --exclude` default in the wrapper, so include wrapper defaults in the
list if they must remain excluded. `COMMON` is shell convenience only; each
command below submits exactly one experiment and nothing automatically fans out
the matrix.

## 4. Run both preflights

Preflight reads only manifests and configuration metadata; it does not scan all
prepared shards or start training.

```bash
python train.py \
  --config "$BASE" \
  --output-root "$OUT" \
  --override run.run_id=10b-base-preflight \
  --override "dataset.prepared_corpus_dir=$CORPUS" \
  --override "model.tokenizer_dir=$TOKENIZER" \
  --override training.distributed.expected_world_size=4 \
  --preflight

python train.py \
  --config "$BAYES" \
  --output-root "$OUT" \
  --override run.run_id=10b-bayesian-preflight \
  --override "dataset.prepared_corpus_dir=$CORPUS" \
  --override "model.tokenizer_dir=$TOKENIZER" \
  --override training.distributed.expected_world_size=4 \
  --preflight
```

Both must report:

- `effective_world_size: 4`
- `expected_tokens_per_microstep: 16384`
- `gradient_accumulation_steps: 64`
- `expected_tokens_per_step: 1048576`
- `derived_max_steps: 9537`
- `resolved_warmup_steps: 156`
- the expected private tokenizer manifest hash and corpus hash

The Bayesian preflight must additionally report Thompson sampling, the fixed
128-example controller role, and an 800-update balanced pre-nested warmup.

## 5. Run four-GPU accumulation smokes

Before production, submit one-update smokes for the five distinct control-flow
paths. Do not proceed until all five complete successfully. These jobs exercise
the full 64-microstep accumulation window under four-process FSDP.

```bash
"${SBATCH[@]}" --time=01:00:00 --gres=gpu:4 \
  --job-name=smoke-10b-random-global \
  scripts/slurm_dmodel256_pilot.sh \
  --config "$BASE" --mode nested-random \
  --run-id smoke-10b-random-global \
  "${COMMON[@]}" \
  --override model.granularity_sampling_mode=global \
  --override training.max_steps_cap=1

"${SBATCH[@]}" --time=01:00:00 --gres=gpu:4 \
  --job-name=smoke-10b-nested-all \
  scripts/slurm_dmodel256_pilot.sh \
  --config "$BASE" --mode nested-all \
  --run-id smoke-10b-nested-all \
  "${COMMON[@]}" \
  --override training.max_steps_cap=1

"${SBATCH[@]}" --time=01:00:00 --gres=gpu:4 \
  --job-name=smoke-10b-ts-global \
  scripts/slurm_dmodel256_pilot.sh \
  --config "$BAYES" --mode nested-random \
  --run-id smoke-10b-ts-global \
  "${COMMON[@]}" \
  --override training.max_steps_cap=1

"${SBATCH[@]}" --time=01:00:00 --gres=gpu:4 \
  --job-name=smoke-10b-ts-reset \
  scripts/slurm_dmodel256_pilot.sh \
  --config "$BAYES" --mode nested-random \
  --run-id smoke-10b-ts-reset \
  "${COMMON[@]}" \
  --override training.max_steps_cap=1 \
  --override model.adaptive_controller.reset.enabled=true \
  --override model.adaptive_controller.reset.interval_steps=2000 \
  --override model.adaptive_controller.reset.policy=full_prior \
  --override model.adaptive_controller.reset.acquisition_policy=balanced_global \
  --override model.adaptive_controller.reset.acquisition_passes=1

"${SBATCH[@]}" --time=01:00:00 --gres=gpu:4 \
  --job-name=smoke-10b-ts-acquisition \
  scripts/slurm_dmodel256_pilot.sh \
  --config "$BAYES" --mode nested-random \
  --run-id smoke-10b-ts-acquisition \
  "${COMMON[@]}" \
  --override training.max_steps_cap=1 \
  --override model.adaptive_controller.reset.enabled=true \
  --override model.adaptive_controller.reset.interval_steps=2000 \
  --override model.adaptive_controller.reset.policy=acquisition_only \
  --override model.adaptive_controller.reset.acquisition_policy=balanced_global \
  --override model.adaptive_controller.reset.acquisition_passes=1
```

## 6. Submit production experiments

The production configs default to plain slicing (`correction_mode: none`).
`correction_mode` is the only correction input required on the command line;
the resolver derives the internal membership-correction boolean.

All examples request the launcher's validated four-GPU topology and its default
24-hour wall time. If the site permits a longer allocation, change only the
`sbatch --time` value. Do not change GPU count, accumulation geometry, data
paths, or run ID when continuing an existing run.

### Plain slicing — initial focus

Run the random-global reference first, then the remaining variants.

```bash
# Random global reference
"${SBATCH[@]}" --time=24:00:00 --gres=gpu:4 \
  --job-name=10b-random-global-plain \
  scripts/slurm_dmodel256_pilot.sh \
  --config "$BASE" --mode nested-random \
  --run-id 10b-random-global-plain \
  "${COMMON[@]}" \
  --override model.granularity_sampling_mode=global

# Random independent per-block sampling
"${SBATCH[@]}" --time=24:00:00 --gres=gpu:4 \
  --job-name=10b-random-per-block-plain \
  scripts/slurm_dmodel256_pilot.sh \
  --config "$BASE" --mode nested-random \
  --run-id 10b-random-per-block-plain \
  "${COMMON[@]}" \
  --override model.granularity_sampling_mode=per_block

# Evaluate and train all eight nested granularities per microstep
"${SBATCH[@]}" --time=24:00:00 --gres=gpu:4 \
  --job-name=10b-nested-all-plain \
  scripts/slurm_dmodel256_pilot.sh \
  --config "$BASE" --mode nested-all \
  --run-id 10b-nested-all-plain \
  "${COMMON[@]}"
```

Independent standalone controls:

```bash
"${SBATCH[@]}" --time=24:00:00 --gres=gpu:4 --job-name=10b-standalone-g125 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g125 --run-id 10b-standalone-g125 "${COMMON[@]}"
"${SBATCH[@]}" --time=24:00:00 --gres=gpu:4 --job-name=10b-standalone-g250 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g250 --run-id 10b-standalone-g250 "${COMMON[@]}"
"${SBATCH[@]}" --time=24:00:00 --gres=gpu:4 --job-name=10b-standalone-g375 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g375 --run-id 10b-standalone-g375 "${COMMON[@]}"
"${SBATCH[@]}" --time=24:00:00 --gres=gpu:4 --job-name=10b-standalone-g500 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g500 --run-id 10b-standalone-g500 "${COMMON[@]}"
"${SBATCH[@]}" --time=24:00:00 --gres=gpu:4 --job-name=10b-standalone-g625 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g625 --run-id 10b-standalone-g625 "${COMMON[@]}"
"${SBATCH[@]}" --time=24:00:00 --gres=gpu:4 --job-name=10b-standalone-g750 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g750 --run-id 10b-standalone-g750 "${COMMON[@]}"
"${SBATCH[@]}" --time=24:00:00 --gres=gpu:4 --job-name=10b-standalone-g875 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g875 --run-id 10b-standalone-g875 "${COMMON[@]}"
"${SBATCH[@]}" --time=24:00:00 --gres=gpu:4 --job-name=10b-standalone-g1000 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g1000 --run-id 10b-standalone-g1000 "${COMMON[@]}"
```

Bayesian Thompson-sampling variants:

```bash
# Global and per-block Thompson sampling
"${SBATCH[@]}" --time=24:00:00 --gres=gpu:4 --job-name=10b-ts-global-plain scripts/slurm_dmodel256_pilot.sh --config "$BAYES" --mode nested-random --run-id 10b-ts-global-plain "${COMMON[@]}"
"${SBATCH[@]}" --time=24:00:00 --gres=gpu:4 --job-name=10b-ts-per-block-plain scripts/slurm_dmodel256_pilot.sh --config "$BAYES" --mode nested-random --run-id 10b-ts-per-block-plain "${COMMON[@]}" --override model.granularity_sampling_mode=adaptive_per_block

# Full-prior reset every 2,000 optimizer updates
"${SBATCH[@]}" --time=24:00:00 --gres=gpu:4 --job-name=10b-ts-global-plain-reset-k2000 scripts/slurm_dmodel256_pilot.sh --config "$BAYES" --mode nested-random --run-id 10b-ts-global-plain-reset-k2000 "${COMMON[@]}" --override model.adaptive_controller.reset.enabled=true --override model.adaptive_controller.reset.interval_steps=2000 --override model.adaptive_controller.reset.policy=full_prior --override model.adaptive_controller.reset.acquisition_policy=balanced_global --override model.adaptive_controller.reset.acquisition_passes=1

# Acquisition-only episode every 2,000 optimizer updates
"${SBATCH[@]}" --time=24:00:00 --gres=gpu:4 --job-name=10b-ts-global-plain-acquisition-k2000 scripts/slurm_dmodel256_pilot.sh --config "$BAYES" --mode nested-random --run-id 10b-ts-global-plain-acquisition-k2000 "${COMMON[@]}" --override model.adaptive_controller.reset.enabled=true --override model.adaptive_controller.reset.interval_steps=2000 --override model.adaptive_controller.reset.policy=acquisition_only --override model.adaptive_controller.reset.acquisition_policy=balanced_global --override model.adaptive_controller.reset.acquisition_passes=1
```

### GMC extensions

Run these only after the corresponding plain-slicing variants. GMC is enabled
through one explicit correction override.

```bash
# Random global and random independent per-block
"${SBATCH[@]}" --time=24:00:00 --gres=gpu:4 --job-name=10b-random-global-gmc scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode nested-random --run-id 10b-random-global-gmc "${COMMON[@]}" --override model.granularity_sampling_mode=global --override model.correction_mode=gmc
"${SBATCH[@]}" --time=24:00:00 --gres=gpu:4 --job-name=10b-random-per-block-gmc scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode nested-random --run-id 10b-random-per-block-gmc "${COMMON[@]}" --override model.granularity_sampling_mode=per_block --override model.correction_mode=gmc

# Nested-all
"${SBATCH[@]}" --time=24:00:00 --gres=gpu:4 --job-name=10b-nested-all-gmc scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode nested-all --run-id 10b-nested-all-gmc "${COMMON[@]}" --override model.correction_mode=gmc

# Thompson global and per-block
"${SBATCH[@]}" --time=24:00:00 --gres=gpu:4 --job-name=10b-ts-global-gmc scripts/slurm_dmodel256_pilot.sh --config "$BAYES" --mode nested-random --run-id 10b-ts-global-gmc "${COMMON[@]}" --override model.correction_mode=gmc
"${SBATCH[@]}" --time=24:00:00 --gres=gpu:4 --job-name=10b-ts-per-block-gmc scripts/slurm_dmodel256_pilot.sh --config "$BAYES" --mode nested-random --run-id 10b-ts-per-block-gmc "${COMMON[@]}" --override model.granularity_sampling_mode=adaptive_per_block --override model.correction_mode=gmc

# Thompson episodic variants
"${SBATCH[@]}" --time=24:00:00 --gres=gpu:4 --job-name=10b-ts-global-gmc-reset-k2000 scripts/slurm_dmodel256_pilot.sh --config "$BAYES" --mode nested-random --run-id 10b-ts-global-gmc-reset-k2000 "${COMMON[@]}" --override model.correction_mode=gmc --override model.adaptive_controller.reset.enabled=true --override model.adaptive_controller.reset.interval_steps=2000 --override model.adaptive_controller.reset.policy=full_prior --override model.adaptive_controller.reset.acquisition_policy=balanced_global --override model.adaptive_controller.reset.acquisition_passes=1
"${SBATCH[@]}" --time=24:00:00 --gres=gpu:4 --job-name=10b-ts-global-gmc-acquisition-k2000 scripts/slurm_dmodel256_pilot.sh --config "$BAYES" --mode nested-random --run-id 10b-ts-global-gmc-acquisition-k2000 "${COMMON[@]}" --override model.correction_mode=gmc --override model.adaptive_controller.reset.enabled=true --override model.adaptive_controller.reset.interval_steps=2000 --override model.adaptive_controller.reset.policy=acquisition_only --override model.adaptive_controller.reset.acquisition_policy=balanced_global --override model.adaptive_controller.reset.acquisition_passes=1
```

## 7. Monitor, resume, and verify completion

Track scheduler state and inspect Slurm output from the repository root:

```bash
squeue --me
tail -f logs/matformer_dmodel256_<job-id>.out
tail -f logs/matformer_dmodel256_<job-id>.err
```

Find a run directory and inspect its rank-zero heartbeat journal:

```bash
export RUN_ID="<run-id>"
export RUN_DIR="$(find "$OUT" -type d -name "$RUN_ID" -print -quit)"
test -n "$RUN_DIR"
tail -n 20 "$RUN_DIR/heartbeats.jsonl"
```

The 24-hour allocation may end before a 10B experiment completes. Resume by
submitting the exact same production command with the same run ID. Continuation
is enabled, and the latest checkpoint is saved at each 500M-token validation
boundary. Never run two jobs with the same run ID concurrently. Resume requires
the same four-GPU topology, accumulation geometry, tokenizer, corpus, role
manifests, and sampling policy.

The Slurm wrapper automatically launches final-holdout evaluation after a
training run reaches completion. A complete run must contain
`run_summary.json`, a final checkpoint, and `final_holdout_results.json`.
Validate its summary with:

```bash
python -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); s=json.loads(p.read_text()); assert s["status"] == "completed", s.get("status"); assert s["tokens_seen"] == 10_000_000_000, s.get("tokens_seen"); assert not s.get("unresolved_artifact_failures"), s.get("unresolved_artifact_failures"); print("completed", s["run_id"], s["tokens_seen"], "tokens")' "$RUN_DIR/run_summary.json"
test -r "$RUN_DIR/final_holdout_results.json"
```

## 8. Fixed scientific and data contract

For four GPUs, each rank consumes four 1,024-token sequences per microstep:
16,384 global tokens per microstep, accumulation 64, and 1,048,576 nominal
tokens per optimizer update. The 10B budget resolves to 9,537 updates; the final
window has 48 microsteps and commits exactly 779,264 token IDs. The cosine
scheduler uses 156 warmup updates. Bayesian `h=50`, the 800-update balanced
warmup, and reset `K=2000` are all expressed in optimizer-update units.

Validation occurs whenever committed tokens cross another 500M-token boundary
and at completion. Random global/per-block and Bayesian profiles are sampled
once per optimizer window; nested-all evaluates all eight granularities on each
microstep. Every local mean loss is scaled from the exact valid-target
denominator of the buffered window. FSDP synchronizes the final backward
contribution before clipping, AdamW, correction, scheduler, controller,
metrics, and checkpoint state commit once at the optimizer boundary.

The prepared corpus reserves disjoint ordinary-validation, controller, and
final-holdout roles before optimizer training. Training selects the first
`training.token_budget / 1024` entries of the corpus-owned PCG64 permutation,
so the 10B order is an exact prefix of a future 80B run.

## Appendix: creating prepared inputs

This section is not part of the normal handoff execution path. Use it only when
deliberately building a new immutable tokenizer and full-source corpus. It is
the authoritative preparation and resume procedure; module execution keeps
repository imports valid from a fresh checkout.

### Preparation contract

The tokenizer and corpus are separate immutable artifacts:

1. Train one 256,000-entry SentencePiece BPE tokenizer from a deterministic
   seed-42 shuffle of FineWeb `sample-10BT`. The default tokenizer-training
   contract consumes 5,000,000 source documents.
2. Tokenize and pack the complete seed-42 shuffled FineWeb `sample-100BT`
   source with that tokenizer.
3. Audit the published tokenizer and corpus before any production launch.

Set `TOKENIZER` and `CORPUS` to new, nonexistent output directories supplied
privately. The full corpus is hundreds of gigabytes, so confirm quota and
filesystem capacity before starting. Remote source preparation also requires
working Hugging Face access and cache configuration; unlike training from an
existing prepared corpus, it may require `HF_TOKEN` at the site.

### Train the tokenizer

```bash
python -m scripts.train_fineweb_tokenizer \
  --output-dir "$TOKENIZER"
```

The command writes the tokenizer files and `tokenizer_manifest.json`. It is
safe to run again with identical arguments: an exact existing artifact is
verified and reported as `already_prepared`. A missing, partial, corrupt, or
differently configured artifact fails without being silently replaced.

The tokenizer contract records source dataset/config/split, seed, shuffle
buffer, training document and chunk counts, special tokens, vocabulary size,
and checksums for every published tokenizer file. Treat the finished directory
as read-only input to corpus preparation and all later runs.

### Prepare the full-source corpus

```bash
python -m scripts.prepare_fineweb_corpus \
  --output-dir "$CORPUS" \
  --prepared-tokenizer-dir "$TOKENIZER" \
  --tokenization-workers 8
```

`--tokenization-workers` defaults to `1`. Workers tokenize concurrently through
a bounded thread pool, but results are committed in deterministic source order.
The same value defaults `--source-read-workers`; source readers fetch
independent streaming shards concurrently and merge them back into exact source
order before applying the seed-pinned buffer shuffle. Set the counts separately
when source reading and tokenization need different concurrency:

```bash
python -m scripts.prepare_fineweb_corpus \
  --output-dir "$CORPUS" \
  --prepared-tokenizer-dir "$TOKENIZER" \
  --source-read-workers 4 \
  --tokenization-workers 8 \
  --progress-interval-seconds 15
```

Preparation streams the complete source without padding or truncation, inserts
EOS between documents, and packs contiguous 1,024-token rows into immutable
little-endian uint32 shards. Before optimizer data, it reserves disjoint roles
in this order:

- 512 ordinary-validation documents
- 128 controller documents
- 512 final-holdout documents

Every remaining complete packed sequence belongs to optimizer training. Source
exhaustion is required; the manifest's recorded count, rather than the dataset
name, is authoritative.

Progress is printed to stderr every 60 seconds and whenever a role or shard
completes. `--progress-interval-seconds 15` requests a faster cadence. The only
stdout output is the final JSON summary, whose status is `prepared`,
`resumed_and_prepared`, or `already_prepared`.

### Interrupted preparation and resume

Preparation uses a stable hidden work directory next to `CORPUS`, checkpoints
each completed shard, and publishes the final corpus directory only after every
source document and permutation entry is committed. A lock prevents two
writers from targeting the same output.

Resume uses the exact same command and arguments. Before continuing, it:

1. verifies the progress-envelope hash and preparation identity;
2. rereads and checksums every completed token shard;
3. restores the partially filled shard without retokenizing committed output;
4. deterministically rereads and reshuffles source documents up to the saved
   document position; and
5. continues ordered tokenization from the first uncommitted document.

Shard verification and source replay can both be long. Resume progress reports
replayed/target documents, percentage, throughput, ETA, and active reader
count. Do not delete or rename the hidden work directory, change worker-neutral
scientific arguments, or launch a second writer while recovery is running.

Worker counts affect throughput but not corpus identity. Changes to source,
tokenizer, seed, context length, shuffle buffer, packing version, or shard
capacity are incompatible with an existing partial preparation and fail rather
than mixing artifacts.

### Publish and audit

After source exhaustion, preparation writes one seed-42 PCG64 permutation over
all optimizer sequences as a little-endian uint64 memory map. It then publishes
the schema-v3 manifest last and makes the artifacts read-only.

Run the full integrity audit once before handing the corpus to production:

```bash
python -m scripts.audit_prepared_corpus \
  --prepared-corpus-dir "$CORPUS" \
  --prepared-tokenizer-dir "$TOKENIZER" \
  --minimum-training-tokens 80000000000 \
  --required-vocab-size 256000
```

The audit verifies:

- every tokenizer and packed-shard checksum;
- tokenizer provenance and exact 256,000-entry vocabulary compatibility;
- corpus and role manifest hashes;
- pairwise separation of validation, controller, and final-holdout documents;
- the complete stored permutation, including bounds, uniqueness, and checksum;
- at least 80B optimizer tokens for reuse by both 10B and future 80B runs.

The audit intentionally reads the entire prepared corpus and order artifact, so
it can take significant time and NFS bandwidth. It is separate from ordinary
training startup, which validates metadata and memory-maps only the required
shards.

Re-running preparation against a fully published exact-match corpus also
performs a complete checksum pass and prints `already_prepared`. It does not
reload FineWeb or rebuild the corpus after verification succeeds.

### Reuse guarantee

Each run selects the first `training.token_budget / 1024` entries of the stored
optimizer permutation. Therefore every 10B experiment consumes the same exact
sequence order, and that order is a strict prefix of a later 80B experiment.
Changing the run-level token budget does not rebuild, reshuffle, or modify the
corpus.
