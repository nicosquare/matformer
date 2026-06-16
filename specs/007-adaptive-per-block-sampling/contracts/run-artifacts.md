# Contract: Run Artifact Provenance

This contract defines the fields that must be visible in saved artifacts for
adaptive per-block runs.

## Required Artifact Files

- `config.json`
- `run_summary.json`
- `metrics.csv`
- `scaling_results.csv`
- checkpoint files when continuation is enabled
- extraction metadata for nested runs

## `config.json`

Must record:

- resolved `run.sampling_mode`
- resolved `model.granularity_sampling_mode`
- selected adaptive strategy
- resolved granularity pattern provenance
- correction mode and correction membership flags
- any sampler hyperparameters needed to reproduce the run

## `run_summary.json`

Must record:

- resolved run mode
- resolved sampling mode
- selected sampler strategy
- sampled granularity pattern summary
- reward summary
- correction-penalty summary
- resumable sampler state
- correction context
- resolved config and output locations

## `metrics.csv`

Must include per-step or per-evaluation rows with enough information to tell
the run type apart without console logs.

Minimum provenance fields:

- `sampling_mode`
- `resolved_run_mode`
- `resolved_sampling_mode`
- `granularity_sampling_mode`
- `granularity_pattern_summary`
- `correction_context`
- `reward`
- `correction_penalty`
- `sampler_strategy`

## Resume Contract

Adaptive runs must be resumable from saved state.

- If sampler state is present and compatible, resume from that state.
- If sampler state is missing or incompatible, fail clearly before training.
- Resumed runs must preserve the previous phase, step, epoch, and exploration
  settings.

## Distinguishability Contract

An inspector reading artifacts only, not logs, must be able to distinguish:

- `nested-random + global`
- `nested-random + per_block`
- `nested-random + adaptive_per_block`

The artifact contract is satisfied only when the mode, strategy, and sampler
state are explicit enough to tell those cases apart.
