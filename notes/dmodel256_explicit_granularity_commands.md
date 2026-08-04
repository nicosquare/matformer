# dmodel256 explicit granularity sbatch commands

Run these one at a time if you have a submission cap. This layout uses an aligned explicit granularity map so the `concat` runs stay valid at `d_model=256`.

All commands select the `adam` optimizer preset. The preset resolves to AdamW,
so do not also set `training.optimizer.name`; preset and direct-name selection
are mutually exclusive.

The pilot config deterministically selects 100,000 FineWeb examples before
tokenization. This covers the 100M-token training budget and 512-example
validation holdout without trying to create a tokenized Arrow cache for all
14.9M rows in `sample-10BT`. The bounded tokenized subset stays in memory to
avoid generated Arrow writes on Lustre; the original FineWeb shards are still
read from the shared Hugging Face cache.

```bash
export OUT=/mnt/experiments/matformer
repo=/home/nicolas.avila/dev/references/matformer
script="$repo/scripts/slurm_dmodel256_pilot.sh"
labels='[micro,small,medium,large,full]'
prefixes='{micro: 0.125, small: 0.25, medium: 0.5, large: 0.75, full: 1.0}'
```

## Nested-random, slicing

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random --run-id dmodel256-explicit-nested-random-slicing-none-global \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=slicing \
  --override model.correction_mode=none \
  --override model.membership_correction=false \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=global
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random --run-id dmodel256-explicit-nested-random-slicing-none-per_block \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=slicing \
  --override model.correction_mode=none \
  --override model.membership_correction=false \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=per_block
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random --run-id dmodel256-explicit-nested-random-slicing-gmc-global \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=slicing \
  --override model.correction_mode=gmc \
  --override model.membership_correction=true \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=global
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random --run-id dmodel256-explicit-nested-random-slicing-gmc-per_block \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=slicing \
  --override model.correction_mode=gmc \
  --override model.membership_correction=true \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=per_block
```

## Nested-random, concat

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random --run-id dmodel256-explicit-nested-random-concat-none-global \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=concat \
  --override model.correction_mode=none \
  --override model.membership_correction=false \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=global
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random --run-id dmodel256-explicit-nested-random-concat-none-per_block \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=concat \
  --override model.correction_mode=none \
  --override model.membership_correction=false \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=per_block
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random --run-id dmodel256-explicit-nested-random-concat-gmc-global \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=concat \
  --override model.correction_mode=gmc \
  --override model.membership_correction=true \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=global
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random --run-id dmodel256-explicit-nested-random-concat-gmc-per_block \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=concat \
  --override model.correction_mode=gmc \
  --override model.membership_correction=true \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=per_block
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random --run-id dmodel256-explicit-nested-random-concat-lmc-global \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=concat \
  --override model.correction_mode=lmc \
  --override model.membership_correction=true \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=global
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random --run-id dmodel256-explicit-nested-random-concat-lmc-per_block \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=concat \
  --override model.correction_mode=lmc \
  --override model.membership_correction=true \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=per_block
```

## Nested-all, slicing

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-all --run-id dmodel256-explicit-nested-all-slicing-none \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=slicing \
  --override model.correction_mode=none \
  --override model.membership_correction=false \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=global
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-all --run-id dmodel256-explicit-nested-all-slicing-gmc \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=slicing \
  --override model.correction_mode=gmc \
  --override model.membership_correction=true \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=global
```

## Nested-all, concat

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-all --run-id dmodel256-explicit-nested-all-concat-none \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=concat \
  --override model.correction_mode=none \
  --override model.membership_correction=false \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=global
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-all --run-id dmodel256-explicit-nested-all-concat-gmc \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=concat \
  --override model.correction_mode=gmc \
  --override model.membership_correction=true \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=global
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-all --run-id dmodel256-explicit-nested-all-concat-lmc \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=concat \
  --override model.correction_mode=lmc \
  --override model.membership_correction=true \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=global
```

## Standalone

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode standalone --run-id dmodel256-explicit-standalone-micro-001 --granularity micro \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.correction_mode=none \
  --override model.membership_correction=false \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes"
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode standalone --run-id dmodel256-explicit-standalone-small-001 --granularity small \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.correction_mode=none \
  --override model.membership_correction=false \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes"
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode standalone --run-id dmodel256-explicit-standalone-medium-001 --granularity medium \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.correction_mode=none \
  --override model.membership_correction=false \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes"
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode standalone --run-id dmodel256-explicit-standalone-large-001 --granularity large \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.correction_mode=none \
  --override model.membership_correction=false \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes"
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode standalone --run-id dmodel256-explicit-standalone-full-001 --granularity full \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.correction_mode=none \
  --override model.membership_correction=false \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes"
```

## Adaptive Thompson

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random --run-id dmodel256-explicit-nested-random-slicing-none-adaptive-thompson \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=slicing \
  --override model.correction_mode=none \
  --override model.membership_correction=false \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=adaptive_per_block \
  --override model.adaptive_sampler_strategy=thompson
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random --run-id dmodel256-explicit-nested-random-slicing-gmc-adaptive-thompson \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=slicing \
  --override model.correction_mode=gmc \
  --override model.membership_correction=true \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=adaptive_per_block \
  --override model.adaptive_sampler_strategy=thompson
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random --run-id dmodel256-explicit-nested-random-concat-none-adaptive-thompson \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=concat \
  --override model.correction_mode=none \
  --override model.membership_correction=false \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=adaptive_per_block \
  --override model.adaptive_sampler_strategy=thompson
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random --run-id dmodel256-explicit-nested-random-concat-gmc-adaptive-thompson \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=concat \
  --override model.correction_mode=gmc \
  --override model.membership_correction=true \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=adaptive_per_block \
  --override model.adaptive_sampler_strategy=thompson
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random --run-id dmodel256-explicit-nested-random-concat-lmc-adaptive-thompson \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=concat \
  --override model.correction_mode=lmc \
  --override model.membership_correction=true \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=adaptive_per_block \
  --override model.adaptive_sampler_strategy=thompson
```

## Adaptive UCB

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random --run-id dmodel256-explicit-nested-random-slicing-none-adaptive-ucb \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=slicing \
  --override model.correction_mode=none \
  --override model.membership_correction=false \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=adaptive_per_block \
  --override model.adaptive_sampler_strategy=ucb
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random --run-id dmodel256-explicit-nested-random-slicing-gmc-adaptive-ucb \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=slicing \
  --override model.correction_mode=gmc \
  --override model.membership_correction=true \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=adaptive_per_block \
  --override model.adaptive_sampler_strategy=ucb
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random --run-id dmodel256-explicit-nested-random-concat-none-adaptive-ucb \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=concat \
  --override model.correction_mode=none \
  --override model.membership_correction=false \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=adaptive_per_block \
  --override model.adaptive_sampler_strategy=ucb
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random --run-id dmodel256-explicit-nested-random-concat-gmc-adaptive-ucb \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=concat \
  --override model.correction_mode=gmc \
  --override model.membership_correction=true \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=adaptive_per_block \
  --override model.adaptive_sampler_strategy=ucb
```

```bash
sbatch --gres=gpu:1 "$script" --repo-root "$repo" --output-root "$OUT" \
  --mode nested-random --run-id dmodel256-explicit-nested-random-concat-lmc-adaptive-ucb \
  --override training.token_budget=100000000 \
  --override training.optimizer.preset=adam \
  --override training.learning_rate=0.001 \
  --override training.learning_rate_scale_rule=none \
  --override model.variant=concat \
  --override model.correction_mode=lmc \
  --override model.membership_correction=true \
  --override model.granularity_mode=explicit \
  --override "model.granularities=$labels" \
  --override "model.granularity_prefixes=$prefixes" \
  --override model.granularity_sampling_mode=adaptive_per_block \
  --override model.adaptive_sampler_strategy=ucb
```
