# 10B unique-token slicing experiments

Every run below consumes a deterministic prefix of one immutable, full-source
FineWeb `sample-100BT` corpus and the same immutable 256,000-entry FineWeb
SentencePiece tokenizer. Build the tokenizer once, then prepare all of
`sample-100BT` once. A later 80B run reuses the same artifact and ordering.

The production workflow is:

1. Prepare the complete `sample-100BT` artifact.
2. Audit that it contains at least 80B optimizer tokens.
3. Reuse it for the 10B runs and later 80B runs by changing only the run-level
   `training.token_budget`.

```bash
export TOKENIZER=/nfs-stor/$USER/matformer-tokenizers/fineweb-sp-bpe-256k
export CORPUS=/nfs-stor/$USER/matformer-corpora/fineweb-sample-100bt-sp256k

python scripts/train_fineweb_tokenizer.py \
  --output-dir "$TOKENIZER"

python scripts/prepare_fineweb_corpus.py \
  --output-dir "$CORPUS" \
  --prepared-tokenizer-dir "$TOKENIZER" \
  --tokenization-workers 8

python scripts/audit_prepared_corpus.py \
  --prepared-corpus-dir "$CORPUS" \
  --prepared-tokenizer-dir "$TOKENIZER" \
  --minimum-training-tokens 80000000000 \
  --required-vocab-size 256000
```

Tokenizer training remains a separate immutable input. Corpus preparation
streams the complete seed-42 shuffled source, reserves the first 1,152
validation/controller/final documents, and assigns every remaining complete
packed sequence to optimizer training. The audit must pass before launch. It
verifies all tokenizer, shard, and stored-order checksums, provenance, exact
vocabulary compatibility, role separation, and at least 80B packed optimizer
tokens. The manifest's actual count is authoritative. Preparation tokenizes
without padding or truncation, inserts EOS between source documents, and packs
contiguous 1,024-token rows.

Both preparation commands are safe to repeat with the same arguments. Corpus
preparation uses a stable hidden work directory and checkpoints each completed
shard. An interrupted invocation resumes by replaying only uncommitted source
documents; completed tokenization is retained. If the
output already exists, the tokenizer command verifies its manifest and every
tokenizer-file checksum; the corpus command verifies its manifest, tokenizer
provenance, preparation arguments, and every shard checksum. An exact match
prints `status: already_prepared` and exits before FineWeb is loaded. A partial,
corrupt, or differently configured work state fails without modifying it. The
corpus check reads all packed shards and the order artifact, so it can take a
few minutes, but it avoids the much longer tokenization and packing pass.
Successful first-time and resumed builds report `prepared` and
`resumed_and_prepared`, respectively, together with actual token, document,
shard, elapsed-time, and throughput statistics.
`--tokenization-workers` defaults to `1`; increase it for future corpus builds.
Workers tokenize concurrently through a bounded thread pool while results are
committed in source order, so the setting affects throughput but not corpus
identity. A preparation lock rejects a second writer for the same output.
Preparation prints a progress line to stderr every 60 seconds and whenever a
role or shard completes. Use `--progress-interval-seconds 15` for a faster
cadence; the final JSON summary remains the only stdout output.

After source exhaustion, preparation writes one PCG64 permutation over every
optimizer sequence as a little-endian uint64 memory map and publishes the v3
manifest last. Each run selects the first `training.token_budget / 1024`
entries. Consequently, the 10B sample order is an exact prefix of the future
80B sample order; no corpus rebuild or run-specific reshuffle is involved.

The commands below submit exactly one experiment each. The Slurm wrapper derives
`training.distributed.expected_world_size` from its allocation; the production
schedule uses four GPUs. `COMMON` is shell convenience only—it does not queue or
fan out experiments.

```bash
export OUT=/nfs-stor/$USER/matformer-10b-runs
export BASE=configs/opt-in_exps/slicing_10b_base.yaml
export BAYES=configs/opt-in_exps/slicing_10b_bayesian.yaml
COMMON="--output-root $OUT --override dataset.prepared_corpus_dir=$CORPUS --override model.tokenizer_dir=$TOKENIZER"
```

Both production configs default to plain slicing (`correction_mode: none`).
`correction_mode` is the only correction input needed on the command line;
the resolved `membership_correction` boolean is derived from it. The commands
do not queue or fan out experiments.

## Plain slicing variants — initial focus

Run these variants first. The random-global run is the reference control.

```bash
# Plain slicing: random global reference control
sbatch --job-name=10b-random-global-plain scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode nested-random --run-id 10b-random-global-plain $COMMON --override model.granularity_sampling_mode=global

# Plain slicing: random independent per-block sampling
sbatch --job-name=10b-random-per-block-plain scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode nested-random --run-id 10b-random-per-block-plain $COMMON --override model.granularity_sampling_mode=per_block
```

```bash
# Plain slicing: evaluate and train all eight nested granularities per microstep
sbatch --job-name=10b-nested-all-plain scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode nested-all --run-id 10b-nested-all-plain $COMMON
```

```bash
# Plain slicing: independent standalone controls for every width
sbatch --job-name=10b-standalone-g125 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g125 --run-id 10b-standalone-g125 $COMMON
sbatch --job-name=10b-standalone-g250 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g250 --run-id 10b-standalone-g250 $COMMON
sbatch --job-name=10b-standalone-g375 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g375 --run-id 10b-standalone-g375 $COMMON
sbatch --job-name=10b-standalone-g500 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g500 --run-id 10b-standalone-g500 $COMMON
sbatch --job-name=10b-standalone-g625 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g625 --run-id 10b-standalone-g625 $COMMON
sbatch --job-name=10b-standalone-g750 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g750 --run-id 10b-standalone-g750 $COMMON
sbatch --job-name=10b-standalone-g875 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g875 --run-id 10b-standalone-g875 $COMMON
sbatch --job-name=10b-standalone-g1000 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g1000 --run-id 10b-standalone-g1000 $COMMON
```

```bash
# Plain slicing: Bayesian Thompson sampling variants
sbatch --job-name=10b-ts-global-plain scripts/slurm_dmodel256_pilot.sh --config "$BAYES" --mode nested-random --run-id 10b-ts-global-plain $COMMON
sbatch --job-name=10b-ts-per-block-plain scripts/slurm_dmodel256_pilot.sh --config "$BAYES" --mode nested-random --run-id 10b-ts-per-block-plain $COMMON --override model.granularity_sampling_mode=adaptive_per_block
sbatch --job-name=10b-ts-global-plain-reset-k2000 scripts/slurm_dmodel256_pilot.sh --config "$BAYES" --mode nested-random --run-id 10b-ts-global-plain-reset-k2000 $COMMON --override model.adaptive_controller.reset.enabled=true --override model.adaptive_controller.reset.interval_steps=2000 --override model.adaptive_controller.reset.policy=full_prior --override model.adaptive_controller.reset.acquisition_policy=balanced_global --override model.adaptive_controller.reset.acquisition_passes=1
sbatch --job-name=10b-ts-global-plain-acquisition-k2000 scripts/slurm_dmodel256_pilot.sh --config "$BAYES" --mode nested-random --run-id 10b-ts-global-plain-acquisition-k2000 $COMMON --override model.adaptive_controller.reset.enabled=true --override model.adaptive_controller.reset.interval_steps=2000 --override model.adaptive_controller.reset.policy=acquisition_only --override model.adaptive_controller.reset.acquisition_policy=balanced_global --override model.adaptive_controller.reset.acquisition_passes=1
```

## GMC extensions

Run these after the corresponding plain-slicing variants. GMC is enabled with
one override; `membership_correction=true` is inferred.

```bash
# GMC: random global and random independent per-block sampling
sbatch --job-name=10b-random-global-gmc scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode nested-random --run-id 10b-random-global-gmc $COMMON --override model.granularity_sampling_mode=global --override model.correction_mode=gmc
sbatch --job-name=10b-random-per-block-gmc scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode nested-random --run-id 10b-random-per-block-gmc $COMMON --override model.granularity_sampling_mode=per_block --override model.correction_mode=gmc
```

```bash
# GMC: nested-all
sbatch --job-name=10b-nested-all-gmc scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode nested-all --run-id 10b-nested-all-gmc $COMMON --override model.correction_mode=gmc
```

```bash
# GMC: Bayesian Thompson sampling variants
sbatch --job-name=10b-ts-global-gmc scripts/slurm_dmodel256_pilot.sh --config "$BAYES" --mode nested-random --run-id 10b-ts-global-gmc $COMMON --override model.correction_mode=gmc
sbatch --job-name=10b-ts-per-block-gmc scripts/slurm_dmodel256_pilot.sh --config "$BAYES" --mode nested-random --run-id 10b-ts-per-block-gmc $COMMON --override model.granularity_sampling_mode=adaptive_per_block --override model.correction_mode=gmc
sbatch --job-name=10b-ts-global-gmc-reset-k2000 scripts/slurm_dmodel256_pilot.sh --config "$BAYES" --mode nested-random --run-id 10b-ts-global-gmc-reset-k2000 $COMMON --override model.correction_mode=gmc --override model.adaptive_controller.reset.enabled=true --override model.adaptive_controller.reset.interval_steps=2000 --override model.adaptive_controller.reset.policy=full_prior --override model.adaptive_controller.reset.acquisition_policy=balanced_global --override model.adaptive_controller.reset.acquisition_passes=1
sbatch --job-name=10b-ts-global-gmc-acquisition-k2000 scripts/slurm_dmodel256_pilot.sh --config "$BAYES" --mode nested-random --run-id 10b-ts-global-gmc-acquisition-k2000 $COMMON --override model.correction_mode=gmc --override model.adaptive_controller.reset.enabled=true --override model.adaptive_controller.reset.interval_steps=2000 --override model.adaptive_controller.reset.policy=acquisition_only --override model.adaptive_controller.reset.acquisition_policy=balanced_global --override model.adaptive_controller.reset.acquisition_passes=1
```

For four GPUs the preflight must report four 1,024-token sequences per rank,
16,384 tokens per microstep, accumulation 64, 1,048,576 nominal tokens per
optimizer update, 9,537 updates, and 156 scheduler-warmup updates. The last
optimizer window has 48 microsteps and commits exactly 779,264 token IDs.
Bayesian `h=50`, the 800-update balanced warmup, and reset `K=2000` remain in
optimizer-update units.

Validation is token-cadenced at every crossed 500,000,000 committed token IDs
and at completion. Random global/per-block and Bayesian profiles are sampled
once per optimizer window; nested-all evaluates all eight granularities on
each microstep. Every local mean loss is scaled from the exact valid-target
denominator of the buffered window. FSDP synchronizes only the final backward
contribution, then clipping, AdamW, GMC/LMC, scheduler, controller progress,
metrics, and checkpointing occur once at the optimizer boundary.

Resume requires the same four-rank topology, accumulation geometry, tokenizer,
corpus, roles, permutation, committed sampler cursor, optimizer/microstep
counters, next validation-token threshold, and per-rank RNG provenance.
Before the production launch, run a short four-GPU accumulation smoke for the
reference random-global run, nested-all, Bayesian plain TS, full-prior reset,
and acquisition-only configurations.
