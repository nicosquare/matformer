# Artifact and Comparison Contract

## Per-run artifacts

Retain existing resolved config, metrics CSV, run summary, checkpoints and ordinary
validation rows. Add versioned campaign fields to those surfaces and use simple
sidecars where rows would otherwise become large:

- `optimizer_ownership_trace.jsonl`: committed update, selected width/action
  identity, epoch/batch cursor and digest, actual packed tokens, active owners,
  counts and scheduler position. Cursor plus immutable order must reproduce the
  exact ordered batch; one record per update, no full tensor state.
- `optimizer_ownership_clipping.jsonl`: C1/C3 per-update group active flags,
  pre/post norms, coefficients/caps and combined norms. Accumulated summaries
  include clipping frequencies by selected width/group with active denominators.
- `resource_attempts.json`: versioned atomic ledger keyed by unique attempt ID,
  latest observation sequence, elapsed time, allocated/reserved peaks, attempted
  work and completion status. Checkpoint references the ledger watermark.
- `terminal_validation_results.json`: immutable ordinary-validation terminal
  endpoint sidecar, specified below.

Existing scalar rows and summaries reference sidecars and their identities.
Full optimizer/model tensors remain in checkpoints. Per-step CSV may contain
compact counts/selected owner data, not a repeated full parameter-name partition.

## Exposure, epochs and resource accounting

Reconcile committed width selections, quarter activations, owner calls, global
steps, sampler cursor, epochs, actual packed tokens and scheduler position.
Elastic updates total 348528; standalone 87132. Final cursor is exactly four/one
complete designated epochs. Selected-width counts remain random; expected 87132
per elastic width and A/B/C/D activations 348528/261396/174264/87132 are labeled
expectations, not enforced totals or example coverage guarantees.

At report/checkpoint boundaries measure optimizer tensor elements and bytes via
actual `numel * element_size`, grouped by owner/component/dtype. Report moments
and counters separately; don't eagerly touch missing state. After all widths
are exposed compare first/second moment totals against:

| Arms | Expected moment elements |
| --- | --- |
| S1/C1/C3 | 2(F+R) |
| S2 | 2(4F+4R) |
| C2 | 2(2.5F+4R) |

Current bias-free model construction gives F=196608, R=328256, producing
1049728 / 4198912 / 3609088 moment elements respectively. Still measure actual
allocations; don't derive state dtype from bf16 compute. C2 saves 37.5% of FFN
moments relative to S2, not 37.5% of total training memory.

Record device peak allocated/reserved memory, persistent state, temporary concat
buffer observations or labeled layout-byte estimates, wall time, throughput and
terminal checkpoint size/hash separately. A buffer estimate is not device peak
memory; label method and whether backward temporaries are included. Keep CPU
measurements distinctly labeled and never report unsupported device metrics as 0.

Persist attempt observations at existing metric flush/heartbeat/checkpoint and
normal/failure exit boundaries. Sum each unique attempt's latest elapsed duration;
take max peak across all attempts. Include failed/replayed work after the last
checkpoint, exclude inter-attempt queue downtime, and never add a checkpoint
watermark a second time. Report useful committed tokens/cumulative time separately
from attempted throughput. Unfinalized hard-kill attempts have incomplete costs;
retain known totals and flag completeness rather than inventing unmeasured costs.
Complete resource claims require finalized or externally reconciled attempts.

## Terminal ordinary-validation sidecar

Use `evaluate_validation_per_granularity` and the existing
`target_token_weighted_causal_shift_float64` aggregation. Perplexity is `exp(loss)`
of the aggregated valid causal-target-weighted loss. Record examples/valid target
counts and ordinary-validation manifest/protocol identity. Never average batch
perplexities or use trailing validation means.

After the assigned terminal update, evaluate all four elastic widths or the
standalone source width. Reuse an existing terminal validation only if its exact
committed model identity is known. Publish a durable terminal resumable checkpoint
and bind the sidecar to its hash; no optimizer mutation may intervene. If either
checkpoint publication or evaluation fails, no valid terminal sidecar is emitted.

If the durable terminal checkpoint exists but sidecar publication failed,
continuation performs completion-only recovery from that exact checkpoint:
reuse or rerun ordinary validation and atomically publish the missing sidecar.
Do not take another optimizer step, replace the terminal checkpoint identity or
select a best-validation checkpoint. Repeating completion after a valid sidecar
exists validates and reuses it. Record recovery attempt costs in the ledger.

Schema 1 includes run/campaign/arm IDs and contract hash, checkpoint path/SHA256,
global step, actual/assigned tokens/epochs/updates, representation/ownership/
clipping and initialization identity, evaluation role `ordinary_validation`,
manifest/protocol/aggregation, examples/targets, count convention and per-width
loss/perplexity/counts. Include a canonical content hash excluding its own field.

## Freeze and complete-report validation

Dedicated campaign readers in `src/evaluation/optimizer_ownership.py` validate the
full expected controls and allowed differences. Do not reuse the old analyzer's
holdout preference, five-validation average, three-seed matrix or sum-of-owner-
calls invariant. It stays compatible with historical experiments.

Freeze requires nine distinct completed fresh campaign runs with durable terminal
checkpoints, reconciled summaries and bound ordinary-validation sidecars. Check
hashes, correct horizons, seed/initializer/data/sampler controls, representation,
scope/clipping, count convention, epoch alignment and full runtime trace digests.
Compare all elastic action/batch traces and all first epochs. Emit a frozen
manifest with source file hashes and selected terminal identities; reporting
rechecks these sources to detect later replacement.

Report requires exactly 24 unique `(arm,width)` endpoints: four for each elastic
arm and one for each standalone. All share evaluation manifest/protocol/role and
evaluated targets. Validate finite loss/perplexity, their exponential relation,
exact counts 115264/164416/213568/262720, assigned/actual budgets, terminal hash and
provenance. Reject best/epoch-one/nonterminal checkpoints, missing/duplicate points,
mixed roles, stale hashes, wrong targets or conventions. No fallback or invented
point is allowed. Presence of `final_holdout_results.json` must not affect reading.

Explicit `--allow-partial` permits visibly labeled incomplete diagnostic output,
with missing arms/points and a manifest marked partial. Invalid present endpoints
still fail; a partial report cannot claim the full campaign comparison. It never
opens holdout. Future holdout work requires a separate user request, prior freeze
of all nine terminals and explicit uniform evaluation of all 24 endpoints.

## Tables, figures and interpretation

Export `optimizer_ownership_endpoints.csv` and `.json` with identical 24 endpoint
rows for complete output. Columns/fields include arm/run/seed, representation,
ownership, clipping contract, width/fraction/dimension, exact
`non_embedding_parameters`, loss/perplexity, evaluated targets, assigned/actual
training budgets, checkpoint/evaluation identities, exposure and resource values.
JSON may include a metadata envelope; tabular rows and scalar values match CSV.

Produce exactly two combined endpoint figures, each PNG and PDF:

- `optimizer_ownership_perplexity_vs_non_embedding_parameters`
- `optimizer_ownership_loss_vs_non_embedding_parameters`

Each uses exact integer active counts as x, five consistently colored connected
four-point elastic curves, and four unconnected standalone markers. Coincident
points retain separate labeled series. No seed error bars or uncertainty bands.
Annotate dataset, seed 42, ordinary-validation role, count definition (input
embeddings/head excluded), terminal rule and one-pass standalone/four-pass elastic
budgets. One elastic run matches the aggregate tokens of the four-standalone
panel; do not claim equal per-run compute, time or realized width coverage.

Also retain individual loss/perplexity trajectories and resource plots per run.
Interpret S1/S2 and C1/C2 as within-representation history comparisons; S1/C1 and
S2/C2 as representation comparisons with changed tail/counter/allocation
semantics; C1/C3 as clipping with different combined caps; and all elastic widths
against their matching standalone. Make descriptive seed-42 claims only.
