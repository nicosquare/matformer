# Contract: Command Interfaces

These commands describe the intended researcher-facing interfaces. Names may be
implemented as scripts or Python module entry points, but the arguments and
outputs must remain visible and easy to trace.

## Run One Experiment

```bash
python train.py --config configs/debug_matrix.yaml --run-id debug-nested-001
```

Required behavior:
- Resolve config and CLI overrides.
- Save `outputs/<run_id>/config.json`.
- Train or evaluate the requested model family and granularity set.
- Write relevant CSV/JSON artifacts.

## Run Debug Matrix

```bash
bash scripts/run_debug_matrix.sh
```

Required behavior:
- Run one debug-size nested S/M/L/XL experiment.
- Run matched standalone S, M, L, and XL baselines.
- Write comparison artifacts and plots derived from CSV files.

## Run 78M Reduced-Token Pilot

```bash
bash scripts/run_78m_pilot.sh
```

Required behavior:
- Use paper-aligned architecture constants.
- Label the run `reduced-token-pilot` unless the token budget is 10B.
- Write model-size and token-budget labels into summaries.

## Generate Figures

```bash
python scripts/make_figures.py --input outputs --output outputs/figures
```

Required behavior:
- Read `metrics.csv`, `scaling_results.csv`, and `consistency_results.csv`.
- Generate plots from structured artifacts only.
- Never require terminal logs as plot inputs.

## Run Consistency Evaluation

```bash
python -m evaluation.consistency --config configs/consistency.yaml
```

Required behavior:
- Compare smaller and larger granularities on the same samples.
- Write `consistency_results.csv`.

## Run Speculative Evaluation

```bash
python -m evaluation.speculative --config configs/speculative.yaml
```

Required behavior:
- Compare nested draft/verifier pairs and standalone draft/verifier pairs.
- Report acceptance rate, rollback frequency, throughput, and latency.
