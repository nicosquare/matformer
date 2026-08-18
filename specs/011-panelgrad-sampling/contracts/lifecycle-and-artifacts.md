# Lifecycle and Artifacts Contract: PanelGrad

## Explicit Training-Loop Order

Immediately before the ordinary optimizer-window action is selected:

1. If no PanelGrad method is active, execute the existing selector unchanged.
2. If PanelGrad is active and its phase is refresh-pending, execute one complete refresh transaction.
3. Draw one global label from the active `Categorical(p)` on rank zero.
4. Broadcast and validate the sampled action/state.
5. Convert it to the existing `{"kind":"global","granularities":[g]}` action.
6. Clear gradients and run the ordinary accumulated forward/backward and optimizer path.
7. Only after optimizer and scheduler commit, record the action exposure and interval progress.
8. When progress reaches `H`, mark refresh pending; do not measure until another action is actually required.

The code block is named and commented as the PanelGrad policy decision point. It is not hidden inside the TS post-step boundary callback or a generic policy registry.

## Refresh Transaction

For a refresh at completed step `s`:

1. Snapshot PanelGrad state, generator state, model train/eval and global granularity state, correction-suspension flags, and ordinary RNG state.
2. Verify controller and support hashes, assert that no optimizer window is in flight, and clear obsolete post-step gradients.
3. Enter evaluation mode with autograd and suspend membership correction.
4. For each resolved granularity in order:
   - configure that label globally across all FFN layers;
   - clear measurement gradients;
   - accumulate backward contributions for the full target-token-weighted controller loss;
   - calculate the float64 squared norm over only that granularity's controlled FFN support;
   - combine with the saved `N_g` to calculate norm and RMS;
   - clear its gradients before measuring the next label.
5. Select either each recorded RMS or raw L2 norm according to `importance_metric`, resolve epsilon from committed PanelGrad optimizer-step count `s` (not global/warmup steps), then construct and validate `q`, `p`, entropy, and probability extrema in float64.
6. Restore correction, prior model mode/granularity state, ordinary RNG, and empty gradients in `finally`.
7. Atomically install the complete snapshot, append its event, synchronize state, and enter `active_interval` with progress zero.

The initial refresh uses the schedule start. A complete snapshot stores its active epsilon and schedule step, and both epsilon and `p` remain immutable until the next successful refresh. A failed refresh leaves schedule progress unchanged.

On any failure, restore the previous complete state, leave gradients empty, record an attributable failure, synchronize failure, and stop. PanelGrad never falls back to its old `p`, random sampling, TS, or UCB for another training step.

## Aggregate Gradient Contract

For each controller microbatch `b`, let `n_b` be its valid causal target count and `N` the full fixed-panel target count. Backpropagate `L_{g,b} n_b/N`; the accumulated gradient is therefore

$$
d_g=\nabla_{\theta_g}\left(\frac{\sum_b n_bL_{g,b}}{N}\right).
$$

The default RMS score is

$$
I_g=\frac{\sqrt{\sum_{j\in S_g}d_{g,j}^2}}{\sqrt{|S_g|}},
$$

where `S_g` is the resolved controlled FFN support. Batchwise norms are never averaged. Membership correction, clipping, optimizer transforms, and inverse-probability weights are absent.

With `importance_metric: gradient_l2`, the score is instead

$$
I_g=\sqrt{\sum_{j\in S_g}d_{g,j}^2}.
$$

Both measurement fields remain recorded and the controller measurement path is identical.

## Distributed Measurement Contract

- Every rank iterates the same controller examples in the same batches, so FSDP executes the same number of forward/backward collectives.
- Each rank uses `n_b/N`; because contributions are identical, FSDP averaging preserves the full-panel gradient. No training-style world-size multiplier and no `no_sync` accumulation are used.
- With `use_orig_params: true`, all ranks summon one wrapped decoder layer at a time with gradients, extract that layer's exact controlled support, accumulate squared values in float64, and release the layer before proceeding.
- The implementation never summons the full model. It excludes shared down bias and all non-FFN support.
- Counts are resolved once before sharding and are not all-reduced as if they were rank-local counts.
- Every rank must obtain scores and probabilities equal within `rtol=1e-6`, `atol=1e-8`; rank zero remains authoritative and broadcasts the committed state/action.
- Shared controller artifacts are written only once.

## Sampling and Rollback Contract

- `p`, not `q`, is passed to the categorical draw.
- One draw occurs for each pending optimizer step; the selected label is held only across that step's accumulation microbatches.
- The CPU generator state is snapshotted before the draw.
- A successful optimizer/scheduler commit increments `sample_count`, the selected exposure, and interval progress exactly once.
- Any pre-commit failure restores the generator and PanelGrad state so retry produces the same action; no exposure is recorded.
- At progress `H`, the checkpointable phase is `refresh_pending`. Resume performs the refresh before drawing the next action.
- A run ending at or before a boundary records terminal state and does not perform an unused refresh or draw.

## Checkpoint Contract

Each PanelGrad checkpoint includes:

- method family/version, importance metric, scope, ordered granularities, policy values, and tolerances;
- resolved fixed/linear epsilon schedule plus the active snapshot epsilon and schedule step;
- all controller/data role hashes and controlled-support schema/counts;
- lifecycle phase, refresh index, last/next boundary, and interval progress;
- last complete measurements, `q`, `p`, and cost;
- categorical seed/generator state, sample count, exposures, and last committed action/probability;
- warmup state link;
- controller journal path, committed event count/offset/hash;
- resume count/source and last failure provenance.

Resume validates the complete schema before further measurement or training, including reconstructing `p` from the saved importance scores and snapshot epsilon. State schema version 3 treats versions 1 and 2 as legacy `gradient_rms` checkpoints; version 1 additionally migrates only when the resumed run uses fixed epsilon. Cross-metric resume is rejected. Missing or incompatible scheduled state is never synthesized. Non-PanelGrad checkpoints contain no valid PanelGrad state, and Bayesian/legacy controller state is never coerced.

## `controller_metrics.jsonl`

PanelGrad uses explicit method/version and event types:

- `panelgrad_refresh_completed`: boundary, importance metric and selected importance-score vector, active epsilon, epsilon schedule step, ordered per-granularity support counts/loss/norm/RMS, `q`, `p`, entropy/extrema, controller totals, duration/backward count, cumulative cost, hashes, and resume provenance;
- `panelgrad_refresh_failed`: failing stage/error, boundary, attempted active epsilon/schedule step, previous valid snapshot hash, and explicit no-new-distribution/no-action fields;
- existing balanced-warmup events, carrying `posterior_updated: false` and no PanelGrad exposure;
- `panelgrad_terminal_partial` or `panelgrad_terminal_complete`: last snapshot, progress, exposures, and explicit no-unused-refresh/no-unused-draw fields.

All events are append-only, validated before commit, and written by rank zero.

## Ordinary Metrics and Summary

Each adaptive training row includes compact fields:

- PanelGrad method family/version and phase;
- refresh index, boundary, and interval progress;
- sampled granularity and its `p_g`;
- cumulative exposure counts;
- current entropy and probability extrema;
- controller manifest hash and controller artifact paths.

Full measurement and probability vectors stay in the refresh journal, summary, and checkpoint.

`controller_summary.json` and `run_summary.json` include resolved policy/support/data provenance, resolved epsilon schedule, final active epsilon/schedule step, epsilon history, final snapshot, refresh count, exposure counts/fractions, cumulative controller examples/targets/backward evaluations/duration, terminal/warmup/resume status, journal path/hash, and any failure. Reporting classifies the run from explicit PanelGrad family/version rather than strategy-name inference and includes epsilon schedule identity in plot grouping. Refresh diagnostics plot active epsilon over tokens.

## Final-Holdout and Comparison Contract

The final holdout remains unused during PanelGrad training. Existing post-training final-holdout evaluation may consume its saved manifest after the run completes. Initial PanelGrad results compare with uniform global sampling at matched optimizer steps or target tokens, while separately reporting PanelGrad measurement work; no matched-compute claim is implied unless an experiment explicitly performs that comparison.
