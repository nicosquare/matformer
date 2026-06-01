# Research: Long Run Support

## Decision: Resume runs from the latest checkpoint in the resolved output directory

**Decision**: Treat the resolved output directory as the continuation anchor and
resume a relaunched run from the latest saved checkpoint and summary metadata in
that directory.

**Rationale**: The repository already groups artifacts by `run_id` under a
stable output directory and already writes summaries, metrics, checkpoints, and
heartbeats there. Resuming from that location keeps continuation visible and
works the same way from every entry point.

**Alternatives considered**:
- Restart from scratch after preemption: rejected because it wastes the token
  budget when a run exceeds one scheduler allocation.
- Track continuation in a separate database: rejected because it adds another
  moving part without improving the experiment path.
- Infer continuation only from job names: rejected because it would be fragile
  and harder to audit than checkpoint-backed state.

## Decision: Log per-run loss series in W&B without coupling to the figure script

**Decision**: Emit W&B scalars from the per-run training loss rows, with one
series per active granularity for nested runs and one standalone series for
standalone runs, but keep that live-monitoring logic separate from
`scripts/make_figures.py`.

**Rationale**: W&B is a single-run dashboard, while the figure script produces
cross-run size/perplexity comparisons. Sharing the same helper would couple two
different analysis surfaces that do not have the same inputs or outputs.

**Alternatives considered**:
- Share a helper with `scripts/make_figures.py`: rejected because the figure
  workflow aggregates across experiments while W&B logs a single experiment at a
  time.
- Log only one aggregate loss series: rejected because it hides the
  granularity-level signal needed for analysis.

## Decision: Make warmup an explicit pre-nested phase for nested runs only

**Decision**: Add a dedicated warmup phase before nested block slicing/splitting
with explicit controls for whether it is enabled, how long it runs, and whether
the duration is measured in epochs or optimization steps. Standalone runs skip
this phase entirely.

**Rationale**: The existing scheduler warmup already covers optimizer ramp-up.
The requested warmup is different: it is meant to improve the starting weights
before the nested phase begins, so it needs its own configuration and summary
state. Standalone runs do not have a nested phase to warm up for.

**Alternatives considered**:
- Reuse scheduler warmup only: rejected because it does not represent the
  requested pre-nested training stage.
- Hard-code warmup to a fixed number of epochs: rejected because debug runs and
  longer Slurm runs need different control granularity.
- Make warmup available to standalone runs: rejected because the use case is
  nested-only and standalone comparison runs should remain untouched.

## Decision: Keep monitoring optional and non-blocking

**Decision**: Treat W&B as an optional monitoring destination. If monitoring is
disabled or the backend is unavailable, the training run should still complete
and write the filesystem artifacts.

**Rationale**: Monitoring should improve observability without blocking the
research run itself.

**Alternatives considered**:
- Make W&B mandatory for every run: rejected because it would create an
  external dependency for a feature that should primarily improve inspection.
- Drop live monitoring entirely: rejected because the feature request explicitly
  asks for dashboard visibility.

## Decision: Persist continuation and warmup state in run summaries

**Decision**: Record the continuation state, latest checkpoint pointer,
warmup-resolution details, and warmup completion status in the saved run
summary so relaunched runs and later analysis can distinguish fresh, resumed,
and completed runs.

**Rationale**: The repository already relies on `run_summary.json` as the
human-readable record of what happened. Adding the new state there keeps the
feature auditable without introducing a new metadata store.

**Alternatives considered**:
- Store the state only in transient logs: rejected because logs are not a
  reliable comparison surface.
- Encode the state only in checkpoint filenames: rejected because that is too
  indirect for later analysis.
