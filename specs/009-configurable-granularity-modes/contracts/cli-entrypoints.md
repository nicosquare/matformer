# CLI Entry Points Contract: Configurable Granularity Modes

## `train.py`

### Purpose

Run a single training job from a YAML config after resolving the granularity mode and the rest of the experiment configuration.

### Inputs

- `--config PATH`
- `--run-id RUN_ID` optional
- `--output-root PATH` optional
- `--output-dir PATH` optional
- `--override KEY=VALUE` repeatable

### Granularity Contract

- The config may specify canonical or explicit granularity mode.
- The training entrypoint must not assume only `s`, `m`, `l`, and `xl`.
- Standalone runs must reject `run.granularity` values that are not in the resolved granularity list.
- The resolved config written for the run must record the selected granularity mode and resolved ordered labels.

## `scripts/run_dmodel256_pilot.sh`

### Purpose

Launch the d_model=256 pilot comparison and forward the chosen mode, granularity, and overrides to `train.py`.

### Inputs

- `--config PATH`
- `--mode comparison|nested-random|nested-all|standalone`
- `--granularity NAME` for standalone mode
- `--run-id RUN_ID` optional
- `--output-root PATH` optional
- `--output-dir PATH` optional
- forwarded training overrides after `--`

### Granularity Contract

- The runner must derive comparison behavior from the resolved configuration surface rather than from a fixed canonical label list.
- The standalone path must validate the requested granularity against the resolved list for the chosen config.
- If the pilot config defines a custom explicit layout, the runner must accept it without code changes.

## `scripts/slurm_dmodel256_pilot.sh`

### Purpose

Submit the same pilot comparison flow through Slurm.

### Inputs

- `--repo-root PATH`
- `--output-root PATH`
- `--run-id RUN_ID`
- `--config PATH`
- `--mode MODE`
- `--granularity NAME`
- `--python-bin PATH`
- forwarded training overrides after `--`

### Granularity Contract

- The Slurm wrapper must mirror the shell runner’s granularity handling.
- It must not encode canonical granularity names as the only valid standalone choices.

## Stable Behavior

- Existing canonical configs continue to work unchanged.
- Explicit custom granularities become a supported path through the same entrypoints.
- Validation failures should surface before training begins and should identify the bad granularity or incompatible layout.
