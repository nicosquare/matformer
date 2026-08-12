# 10B unique-token slicing experiments

Every run below consumes the same immutable FineWeb `sample-10BT` corpus. Run
the preparation command once with an immutable tokenizer revision (normally a
Hub commit SHA), audit it, and keep both values unchanged for the full matrix:

```bash
export CORPUS=/nfs-stor/$USER/matformer-corpora/fineweb-sample-10bt-llama
export TOKENIZER_REVISION=<immutable-tokenizer-commit>

python scripts/prepare_fineweb_corpus.py \
  --output-dir "$CORPUS" \
  --tokenizer hf-internal-testing/llama-tokenizer \
  --tokenizer-revision "$TOKENIZER_REVISION"

python scripts/audit_prepared_corpus.py --prepared-corpus-dir "$CORPUS"
```

The audit must pass before launch. It reads every shard, verifies its SHA-256,
checks the fixed roles for overlap, and requires exactly 10,000,000,000 packed
training token IDs. Preparation tokenizes without padding or truncation,
inserts EOS between source documents, and packs contiguous 1024-token rows.

The commands below submit exactly one experiment each. The Slurm wrapper derives
`training.distributed.expected_world_size` from its allocation; the production
schedule uses four GPUs. `COMMON` is shell convenience only—it does not queue or
fan out experiments.

```bash
export OUT=/nfs-stor/$USER/matformer-10b-runs
export BASE=configs/opt-in_exps/slicing_10b_base.yaml
export BAYES=configs/opt-in_exps/slicing_10b_bayesian.yaml
COMMON="--output-root $OUT --override dataset.prepared_corpus_dir=$CORPUS --override model.tokenizer_revision=$TOKENIZER_REVISION"
```

Random nested, one run per command:

```bash
sbatch --job-name=10b-random-global-none scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode nested-random --run-id 10b-random-global-none $COMMON --override model.granularity_sampling_mode=global --override model.correction_mode=none --override model.membership_correction=false
sbatch --job-name=10b-random-global-gmc scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode nested-random --run-id 10b-random-global-gmc $COMMON --override model.granularity_sampling_mode=global --override model.correction_mode=gmc --override model.membership_correction=true
sbatch --job-name=10b-random-per-block-none scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode nested-random --run-id 10b-random-per-block-none $COMMON --override model.granularity_sampling_mode=per_block --override model.correction_mode=none --override model.membership_correction=false
sbatch --job-name=10b-random-per-block-gmc scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode nested-random --run-id 10b-random-per-block-gmc $COMMON --override model.granularity_sampling_mode=per_block --override model.correction_mode=gmc --override model.membership_correction=true
```

Nested-all, one run per command:

```bash
sbatch --job-name=10b-nested-all-none scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode nested-all --run-id 10b-nested-all-none $COMMON --override model.correction_mode=none --override model.membership_correction=false
sbatch --job-name=10b-nested-all-gmc scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode nested-all --run-id 10b-nested-all-gmc $COMMON --override model.correction_mode=gmc --override model.membership_correction=true
```

Standalone, one run per command:

```bash
sbatch --job-name=10b-standalone-g125 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g125 --run-id 10b-standalone-g125 $COMMON
sbatch --job-name=10b-standalone-g250 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g250 --run-id 10b-standalone-g250 $COMMON
sbatch --job-name=10b-standalone-g375 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g375 --run-id 10b-standalone-g375 $COMMON
sbatch --job-name=10b-standalone-g500 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g500 --run-id 10b-standalone-g500 $COMMON
sbatch --job-name=10b-standalone-g625 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g625 --run-id 10b-standalone-g625 $COMMON
sbatch --job-name=10b-standalone-g750 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g750 --run-id 10b-standalone-g750 $COMMON
sbatch --job-name=10b-standalone-g875 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g875 --run-id 10b-standalone-g875 $COMMON
sbatch --job-name=10b-standalone-g1000 scripts/slurm_dmodel256_pilot.sh --config "$BASE" --mode standalone --granularity g1000 --run-id 10b-standalone-g1000 $COMMON
```

Bayesian TS + GMC, one run per command:

```bash
sbatch --job-name=10b-ts-global scripts/slurm_dmodel256_pilot.sh --config "$BAYES" --mode nested-random --run-id 10b-ts-global $COMMON
sbatch --job-name=10b-ts-per-block scripts/slurm_dmodel256_pilot.sh --config "$BAYES" --mode nested-random --run-id 10b-ts-per-block $COMMON --override model.granularity_sampling_mode=adaptive_per_block
sbatch --job-name=10b-ts-global-reset-k2000 scripts/slurm_dmodel256_pilot.sh --config "$BAYES" --mode nested-random --run-id 10b-ts-global-reset-k2000 $COMMON --override model.adaptive_controller.reset.enabled=true --override model.adaptive_controller.reset.interval_steps=2000 --override model.adaptive_controller.reset.policy=full_prior --override model.adaptive_controller.reset.acquisition_policy=balanced_global --override model.adaptive_controller.reset.acquisition_passes=1
sbatch --job-name=10b-ts-global-acquisition-k2000 scripts/slurm_dmodel256_pilot.sh --config "$BAYES" --mode nested-random --run-id 10b-ts-global-acquisition-k2000 $COMMON --override model.adaptive_controller.reset.enabled=true --override model.adaptive_controller.reset.interval_steps=2000 --override model.adaptive_controller.reset.policy=acquisition_only --override model.adaptive_controller.reset.acquisition_policy=balanced_global --override model.adaptive_controller.reset.acquisition_passes=1
```

For four GPUs the preflight must report 16,384 nominal tokens per optimizer
step, 610,352 optimizer steps, and 9,980 scheduler warmup steps. The final
global batch contains nine sequences and is split 3/2/2/2 across ranks; its
local mean losses are weighted by valid-target count before FSDP reduction.
Resume requires the same four-rank topology and the saved corpus, role,
permutation, sampler-cursor, and per-rank RNG provenance.
