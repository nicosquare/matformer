# CLI Entry Points Contract: PanelGrad Sampling

## `train.py`

PanelGrad uses the existing inputs:

- `--config PATH`
- `--run-id RUN_ID`
- `--output-root PATH` or `--output-dir PATH`
- `--preflight`
- repeatable `--override KEY=VALUE`

No PanelGrad-specific flags are added.

### Preflight

For `adaptive_global + panelgrad`, preflight validates and displays method identity, resolved granularities, policy defaults/overrides, probability and score semantics, data-role contracts, support/count status, seed provenance, warmup compatibility, and distributed constraints. Invalid scope, mixed strategy fields, policy values, or role contracts fail before training.

### Training

- Raw and packed datasets establish the existing controller, optimizer-training, ordinary-validation, and final-holdout roles.
- Optional balanced warmup runs through its existing forced-global path.
- PanelGrad performs the first refresh only if adaptive training will continue.
- Each optimizer step then samples one existing global action from the frozen `p` and uses the ordinary training path.
- Existing Thompson, UCB, random, nested-all, standalone, validation, monitoring, and final-holdout behavior remains unchanged when PanelGrad is not selected.

### Outputs

PanelGrad runs add method-keyed events and state to existing experiment surfaces:

- resolved PanelGrad fields and support counts/hash in `config.json`;
- existing four role manifests;
- PanelGrad events in `controller_metrics.jsonl`;
- PanelGrad aggregation in `controller_summary.json`;
- compact action/progress fields in `metrics.csv`;
- `panelgrad_state` in checkpoints;
- PanelGrad identity, artifact hashes, and measurement cost in `run_summary.json`.

## `scripts/evaluate_final_holdout.py`

The existing post-training command remains unchanged. A completed PanelGrad run may be evaluated only after training, using the saved final-holdout manifest and an ordinary-validation-selected or explicit checkpoint. The result cannot alter PanelGrad state, exposure, configuration, or checkpoint selection.

## Opt-in Experiment Surface

`configs/opt-in_exps/panelgrad_smoke.yaml` provides a small explicit example. Existing pilot and queue defaults do not gain PanelGrad automatically. Overrides may shorten the refresh interval or total step count for smoke validation without changing default method values in other runs.

## Failure Contract

Training exits with an attributable failure for:

- invalid strategy/scope/policy configuration;
- missing, overlapping, or incompatible data-role manifests;
- zero or mismatched controlled FFN support;
- unsupported distributed settings;
- non-finite controller loss, gradient, score, or probability;
- probability vectors outside the fixed tolerance;
- rank disagreement;
- journal commit failure;
- malformed or incompatible PanelGrad checkpoint state.

No failure silently selects from a stale distribution or falls back to another method.
