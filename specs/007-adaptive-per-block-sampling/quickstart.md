# Quickstart: Adaptive Per-Block Sampling

This quickstart assumes the feature implementation is complete.

## 1. Validate the config surface

Run the config tests that cover explicit mode resolution and invalid pairings:

```bash
pytest tests/test_config.py -q
```

## 2. Validate artifact provenance

Run the artifact tests that assert the resolved mode, strategy, pattern
summary, reward summary, and sampler-state fields are written:

```bash
pytest tests/test_artifacts.py -q
```

## 3. Smoke test the canonical modes

Run the nested-random global baseline:

```bash
python train.py \
  --config configs/debug_matrix.yaml \
  --run-id debug-nested-001 \
  --override model.granularity_sampling_mode=global
```

Run the random per-block baseline:

```bash
python train.py \
  --config configs/debug_matrix.yaml \
  --run-id debug-nested-001 \
  --override model.granularity_sampling_mode=per_block
```

Run the adaptive per-block mode:

```bash
python train.py \
  --config configs/debug_matrix.yaml \
  --run-id debug-nested-001 \
  --override model.granularity_sampling_mode=adaptive_per_block \
  --override model.adaptive_sampler_strategy=thompson
```

## 4. Verify resume behavior

Re-run the adaptive command with continuation enabled and confirm that the
sampler resumes from saved state instead of restarting empty:

```bash
python train.py \
  --config configs/debug_matrix.yaml \
  --run-id debug-nested-001 \
  --override run.continuation.enabled=true \
  --override model.granularity_sampling_mode=adaptive_per_block \
  --override model.adaptive_sampler_strategy=thompson
```

## 5. Inspect outputs

Check these files in the run directory:

- `config.json`
- `run_summary.json`
- `metrics.csv`
- `scaling_results.csv`
- checkpoints under `checkpoints/`

The saved artifacts should make the selected mode and strategy obvious without
reading stdout.

## 6. Validate the adaptive artifact trail

Run the adaptive mode once more, then inspect the saved config and summary
without relying on logs:

```bash
python train.py \
  --config configs/debug_matrix.yaml \
  --run-id debug-nested-001 \
  --override model.granularity_sampling_mode=adaptive_per_block \
  --override model.adaptive_sampler_strategy=thompson

jq '.model.granularity_sampling_mode, .model.adaptive_sampler_strategy, .model.granularity_pattern_provenance' \
  outputs/*/debug-nested-001/config.json

jq '.adaptive_sampler_strategy, .adaptive_sampler_state, .adaptive_reward_summary, .adaptive_correction_penalty_summary' \
  outputs/*/debug-nested-001/run_summary.json

head -n 2 outputs/*/debug-nested-001/metrics.csv
```
