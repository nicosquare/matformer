# Quickstart: Configurable Granularity Modes

## Validate Canonical Compatibility

Run the config-resolution tests that cover the canonical path and explicit
custom layouts:

```bash
pytest tests/test_config.py -k granularity
```

## Smoke Test an Explicit Layout

Run a small explicit-mode smoke config after adding the feature fixture for a five-label layout:

```bash
python train.py \
  --config tests/fixtures/explicit_granularity_smoke.yaml \
  --override training.max_steps=1 \
  --override outputs.save_checkpoints=false
```

Expected result:

- the resolved config records `model.granularity_mode=explicit`
- the resolved config records the ordered labels and prefix fractions
- training completes without assuming only `s`, `m`, `l`, and `xl`

## Verify the Pilot Runner

Run the pilot comparison path using the canonical config:

```bash
bash scripts/run_dmodel256_pilot.sh --mode nested-random --config configs/dmodel256_pilot_comparison.yaml
```

Standalone validation should reject any granularity not in the resolved list. For canonical runs, a valid standalone smoke command remains:

```bash
bash scripts/run_dmodel256_pilot.sh --mode standalone --granularity s --config configs/dmodel256_pilot_comparison.yaml
```

The same entrypoint accepts an explicit layout and validates the requested
label against that resolved list:

```bash
bash scripts/run_dmodel256_pilot.sh \
  --mode standalone \
  --granularity medium \
  --config tests/fixtures/explicit_granularity_smoke.yaml \
  -- --override training.max_steps=1 --override outputs.save_checkpoints=false
```

## Inspect Outputs

After a run, confirm the resolved granularity set appears in saved artifacts:

- resolved config JSON
- metrics CSV or JSON
- run summary JSON
- checkpoint metadata when checkpoints are enabled
