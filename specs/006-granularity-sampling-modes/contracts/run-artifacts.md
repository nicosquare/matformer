# Contract: Run Artifacts

This contract describes the saved outputs that must explain how a run used the
granularity sampling surface.

## Required Artifacts

- `config.json`
- `run_summary.json`
- `metrics.csv`
- `scaling_results.csv`
- checkpoints when configured
- extraction metadata for nested runs

## Provenance Fields

The saved config, run summary, and metric rows must include:

- `run.sampling_mode`
- `model.granularity_sampling_mode`
- `model.correction_mode`
- a compact description of the selected granularity pattern
- the existing experiment identity fields already used by the repository

## Interpretation Contract

- A saved `nested-random + global` run must be readable without knowing the
  implementation internals.
- A saved `nested-random + per_layer` run must include enough information to
  distinguish the pattern from the global path.
- A saved `nested-all` run must show the evaluated granularity set and the mean
  loss aggregation behavior.
- A saved `standalone` run must show the fixed granularity used for the full
  job.
- Downstream analysis should be able to compare runs without consulting console
  logs.

## Acceptance Contract

- Artifact generation must succeed for all supported run modes.
- The saved metadata must match the resolved runtime configuration.
- The artifact contract is not satisfied if the selected mode is only visible
  in terminal output.
