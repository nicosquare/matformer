# Quickstart: Per-Width Optimizer State

These are the planned validation and experiment commands for implementation.
The feature remains opt-in and per-granularity scope is single-process only.

## 1. Inspect configuration and ownership validation

```bash
python train.py \
  --config tests/fixtures/per_granularity_optimizer_smoke.yaml \
  --preflight

python train.py \
  --config tests/fixtures/per_granularity_optimizer_smoke.yaml \
  --override training.optimizer.state_scope=shared \
  --preflight
```

Confirm that both resolve the same optimizer family, kwargs, global scheduler,
ordered widths, budget, and paired-control signature, while full identity and
state scope differ.

Run the rejection/default matrix:

```bash
pytest -q tests/test_config.py tests/test_train_cli.py \
  -k "optimizer_state_scope or scheduler_clock"
```

## 2. Validate optimizer isolation

```bash
pytest -q tests/test_per_granularity_optimizer.py
```

Coverage must include:

- AdamW and SGD alternating-width histories;
- different moments for the same shared prefix parameter;
- no state, decay, or update for wider-only parameters on narrow steps;
- exactly one width update count per commit;
- common global learning rates despite unequal width exposure;
- one owner across a complete accumulation window;
- unchanged optimizer state and counts on pre-commit failure.

## 3. Run the normal CPU smoke path

```bash
python train.py \
  --config tests/fixtures/per_granularity_optimizer_smoke.yaml \
  --override run.run_id=per-granularity-optimizer-smoke \
  --output-dir outputs/per-granularity-optimizer-smoke
```

Inspect:

```bash
python -m json.tool outputs/per-granularity-optimizer-smoke/run_summary.json
sed -n '1,10p' outputs/per-granularity-optimizer-smoke/metrics.csv
```

Verify scope, selected owner, global scheduler position, per-width counts,
exposures, wall time, peak memory, and resumable checkpoint size/hash.

## 4. Validate exact resume

```bash
pytest -q tests/test_per_granularity_optimizer_resume.py
```

Compare uninterrupted execution with resume inside a balanced cycle and at an
exact boundary. Actions and counts are exact; batches, learning rates,
optimizer/scheduler states, metrics, and final parameters match within existing
deterministic tolerance. Missing, extra, reordered, non-finite, cross-scope,
wrong-family, wrong-scheduler, and model-only states must fail before mutation.

## 5. Validate artifacts and compatibility

```bash
pytest -q \
  tests/test_global_sampling_windows.py \
  tests/test_accumulation.py \
  tests/test_training_smoke.py \
  tests/test_artifacts.py \
  tests/test_reproducibility.py \
  tests/test_reporting.py \
  tests/test_distributed.py
```

Shared scope, historical scope omission, current sampling/controller methods,
standalone/nested-all behavior, existing schedulers, and distributed shared
training must remain unchanged. Per-granularity distributed configurations must
fail preflight.

## 6. Select and audit TinyStories-Instruct data

```bash
export TINYSTORIES_PROFILE=instruct
source scripts/select_tinystories_profile.sh

python scripts/audit_prepared_corpus.py \
  --prepared-corpus-dir "$CORPUS" \
  --prepared-tokenizer-dir "$TOKENIZER" \
  --required-vocab-size 2048 \
  --minimum-training-tokens 713785344
```

Confirm the fixed optimizer-training, controller, ordinary-validation, and final
holdout role manifests.

## 7. Preflight all pilot arms

Use the six-command loop in
`docs/tinystories-per-width-optimizer-experiment.md`. Each preflight must resolve:

- four ordered widths and balanced-cycle interval one;
- 87,132 steps and 21,783 updates per width;
- AdamW 0.008 and one global cosine/64-warmup clock;
- one process and the requested scope;
- equal paired-control signatures within each seed.

Do not submit until all six preflights and focused tests pass.

## 8. Run and freeze the pilot

Submit using the runbook. After all six runs complete:

```bash
python scripts/analyze_tinystories_per_width_optimizer.py freeze \
  --phase pilot \
  --input-root "$PILOT_ROOT" \
  --output-dir "$PILOT_ROOT/analysis"

python scripts/analyze_tinystories_per_width_optimizer.py report \
  --manifest "$PILOT_ROOT/analysis/optimizer_state_manifest.json" \
  --output-dir "$PILOT_ROOT/analysis"
```

The first report uses ordinary validation and resources while the final holdout
remains sealed.

## 9. Choose holdout or fresh confirmation

If no confirmation will be run, evaluate each frozen pilot terminal checkpoint
explicitly and rerun `report`. The result remains diagnostic.

If confirmation will be run, leave the pilot holdout sealed, freeze the
confirmation protocol, start fresh six-run confirmation jobs, and then freeze:

```bash
python scripts/analyze_tinystories_per_width_optimizer.py freeze \
  --phase confirmation \
  --input-root "$CONFIRM_ROOT" \
  --output-dir "$CONFIRM_ROOT/analysis"
```

Only after that manifest is frozen, evaluate the six explicit terminal
checkpoints with `scripts/evaluate_final_holdout.py`, then run `report`. A
confirmatory label requires all seeds and proof that the holdout was not opened
during the pilot.
