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

The saved config and run summary must include:

- the requested legacy alias, when one was supplied
- `model.granularity_sampling_mode`
- `model.correction_mode`
- a compact description of the selected granularity pattern
- the existing experiment identity fields already used by the repository

## Interpretation Contract

- A saved global run must be readable without knowing the implementation
  internals.
- A saved per-layer run must include enough information to distinguish the
  pattern from the global path.
- Downstream analysis should be able to compare runs without consulting console
  logs.

## Acceptance Contract

- Artifact generation must succeed for both supported sampling modes.
- The saved metadata must match the resolved runtime configuration.
- The artifact contract is not satisfied if the selected sampling mode is only
  visible in terminal output.
- Legacy alias provenance must be visible without consulting logs.
