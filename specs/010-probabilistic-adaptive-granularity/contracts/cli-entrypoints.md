# CLI Entry Points Contract: Probabilistic Adaptive Granularity

## `train.py`

### Purpose

Resolve and run Bayesian global or additive per-block Thompson training through
the existing config-driven entrypoint while preserving other sampling modes.

### Existing inputs

- `--config PATH`
- `--run-id RUN_ID` optional
- `--output-root PATH` optional
- `--output-dir PATH` optional
- `--preflight` optional
- `--override KEY=VALUE` repeatable

No new dedicated Bayesian flags are added. The configuration contract remains
the single experiment surface.

### Behavior

- `--preflight` validates strategy/scope/probabilistic values and prints their
  resolved provenance without loading the dataset or starting training.
- A Thompson config missing required Bayesian inputs exits before training with
  a migration-specific error.
- Data cardinality, role disjointness, stable identities, and manifest hashes
  are validated after dataset preparation but before any optimizer update.
- `adaptive_global + thompson` and `adaptive_per_block + thompson` enter the new
  boundary controller.
- `adaptive_per_block + ucb` uses the existing UCB runtime unchanged.
- Random global/per-block, nested-all, and standalone behavior remain unchanged.

### Outputs

In addition to existing artifacts, Bayesian runs produce:

- four role manifests plus a parent split manifest;
- `controller_metrics.jsonl`;
- `controller_summary.json`;
- Bayesian controller state in checkpoints;
- resolved method/scope/controller provenance in config, metrics, and run
  summary artifacts.

## `scripts/evaluate_final_holdout.py`

### Purpose

Evaluate the untouched final holdout only after a run is complete.

### Inputs

- `--run-dir PATH` required
- `--checkpoint PATH` optional; when omitted, use the checkpoint previously
  selected by ordinary validation
- `--device DEVICE` optional, following existing evaluation conventions

### Preconditions

- The run summary reports completed training.
- Resolved configuration and final-holdout manifest exist and hashes match.
- The checkpoint belongs to the run and its data/controller provenance is
  compatible.
- If no ordinary-validation-selected checkpoint exists, `--checkpoint` is
  required rather than silently choosing based on final data.

### Behavior

- Reconstruct the final dataset from its saved stable identities.
- Evaluate all resolved global granularities in saved order.
- Use target-token-weighted causal validation semantics.
- Do not load the result into checkpoint selection, controller state, or
  training monitoring.

### Output

- `final_holdout_results.json` in the run directory, containing per-granularity
  loss/perplexity, uniform average loss, checkpoint-selection provenance,
  manifest hash, aggregation semantics, and result hash.

## Pilot and Queue Entry Points

- Existing pilot and queue defaults remain unchanged; no Bayesian runs are
  silently added to the current comparison matrix.
- Bayesian global and per-block pilots are explicit opt-in runs using a
  dedicated smoke/pilot config or forwarded `--override` values.
- Existing UCB commands and labels remain unchanged.

## Exit and Failure Contract

The commands fail with attributable nonzero status for:

- missing Bayesian inputs or invalid covariance/noise;
- invalid strategy/scope combinations;
- insufficient usable examples or role overlap;
- controller/final manifest mismatch;
- non-finite controller objective, reward, mean, or covariance;
- incompatible or legacy heuristic Thompson checkpoint resume;
- attempted final-holdout evaluation before training completion.

No failing controller event silently falls back to random, UCB, legacy
Thompson, or a reinitialized posterior.

