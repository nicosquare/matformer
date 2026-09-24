# Contract: Saved-Artifact Warmup Comparison

## Selected inputs and validation

| Grid | Root | Selected arms |
| --- | --- | --- |
| Linear | `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1` | ST-g250, ST-g500, ST-g750, ST-g1000, S1 |
| Geometric | `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1` | ST-g125, ST-g250, ST-g500, ST-g1000, S1 |

Read selected terminals directly through strict existing readers, with new selection manifests under the new root. A complete nine-arm historical campaign is not required. Exclude corrected/inverse-membership, S2/C arms, partial/cancelled runs and archived invalid CPU attempts. Geometric S1's valid attempt 2/job 272716 is a planning observation to revalidate, not permanent certification.

Require each selected config/contract, checkpoint hash/kind/step, complete budget/trace, physical support/count, initialization/seed/data controls and ordinary-validation provenance to agree. Apply saved execution evidence appropriate to the historical generation: older Linear runs need their applicable launcher/job and actual precision/device/resource evidence, not nonexistent new-format files. Modern new runs require the full lifecycle contract. Missing applicable device evidence fails selection; a historical exception cannot excuse known CPU execution.

Ordinary validation must use the inherited membership/protocol and target-token-weighted causal aggregation. Perplexity equals exp(aggregated loss). Reject non-finite values, missing/duplicate/stale endpoints, wrong target counts, incompatible evaluation, best/early checkpoints, trailing averages and budget inconsistencies. Do not evaluate reserved controller/final data. Recheck source hashes before atomic publication to reject inputs changed during reading.

## Tables and cardinalities

`reports/new/endpoints.{csv,json}` preserves eight valid new endpoints independently of the full comparison. `reports/comparison/endpoints.{csv,json}` contains exactly 24 unique rows, 12 per grid: four standalone, four original S1/warmup 64, four new S1/warmup 256. Twelve terminal checkpoints contribute these rows: ten historical and two new.

Unique endpoint key is `(grid_id, campaign_id, run_id, ffn_dimension)`, never parameter count alone. Required fields:

- Grid/display label, campaign/run/arm/role/seed, warmup, representation/history/clipping.
- Width label, physical dimension/fraction and exact active non-embedding count excluding input embedding and LM head.
- Assigned/actual epochs, updates and training tokens; terminal step.
- Loss, perplexity, evaluated target count, ordinary-validation role/membership/protocol identity.
- Terminal checkpoint/evaluation paths and hashes, config/contract/source identities, execution evidence references.
- Matching original S1 and own-grid standalone endpoint identities; actual width selections, expected exposure, resource measurements and completeness.

Both S1 roles have four epochs/348528 updates/2855141376 tokens per run. Each standalone has one epoch/87132 updates/713785344 tokens. Shared-size standalone measurements remain distinct even when numerically equal. CSV/JSON contain identical logical values using existing lossless serialization conventions.

`paired_deltas.{csv,json}` contains exactly eight `(grid_id, ffn_dimension)` rows with links to the three matched endpoints:

```text
delta_loss = loss_w256 - loss_w64
delta_perplexity = perplexity_w256 - perplexity_w64
old_standalone_gap_loss = loss_w64 - loss_standalone
new_standalone_gap_loss = loss_w256 - loss_standalone
old_standalone_gap_perplexity = perplexity_w64 - perplexity_standalone
new_standalone_gap_perplexity = perplexity_w256 - perplexity_standalone
```

Negative intervention deltas mean improvement; positive means worsening; zero means unchanged. Do not pair across grids even at identical sizes.

## Endpoint figures

For each `linear` and `geometric`, publish `<grid>_loss_vs_parameters.{png,pdf}` and `<grid>_perplexity_vs_parameters.{png,pdf}`: eight endpoint figure files total. Each view has four disconnected matching standalone markers and two distinct connected four-point S1 curves labeled “64-update warmup” and “256-update warmup”. X coordinates are exact active counts; no averaging, jitter or deduplication. Use consistent colors and recognizable Linear/Geometric labels. Annotate seed 42, TinyStories-Instruct, ordinary validation, terminal selection, count convention and one-epoch standalone versus four-epoch S1 budgets. No across-seed error bars.

## Early metrics and figures

Publish `early_metrics.{csv,json}` plus `<grid>_early_lr.{png,pdf}` and `<grid>_early_loss.{png,pdf}` for both grids: eight additional figure files. Common absolute-update window starts at 0 and covers at least 0–1024; mark 64 and 256 in both grids and retain same x limits. There is no fabricated measured loss at 0.

Preserve recorded training loss and recorded pre-optimizer applied LR at committed updates, and available per-width ordinary-validation losses at their recorded cadence. Early loss figures use separate clearly labeled training and validation panels (validation may be faceted by physical width), with both S1 schedules. A training minibatch loss is not a terminal or validation estimate. Resolve replay rows via committed attempt provenance; conflicting/ambiguous duplicates are invalid evidence.

Each supporting row identifies grid/run, committed absolute update, split/width, metric/value, source path/hash/row or attempt, schedule position convention, and `source_kind=recorded` or `reconstructed_schedule`. When recorded LR is unavailable, reconstruct only from validated saved controls and label it explicitly. Reconstructed LR is not GPU execution evidence. Never reconstruct/interpolate unobserved losses as measurements.

Default is raw unsmoothed data. Any optional smoothed overlay must disclose method/window and retain raw export. Plot missing observations as gaps and disclose them; do not connect across an unexplained missing segment as though measured. Historical observed cadence is legitimate. Missing required training/validation evidence is explicitly listed and prevents declaring the early deliverable complete; valid endpoint results remain publishable.

## Report and failure behavior

`comparison_report.json` records source hashes, selection outcomes, row counts, produced files/hashes, missing/invalid reasons, terminal and endpoint/early/comparison statuses. `findings.md` reports direction and numerical magnitude at all eight widths, both S1 standalone gaps, early schedule/loss behavior and resource limitations. Results are descriptive seed 42 observations; no significance, equal-compute/runtime or causal diagnosis of prior underperformance. Improvement is not required.

Complete output publication is atomic and requires all endpoint/early checks and requested figures. Invalid/missing inputs yield an explicit diagnostic/incomplete record and nonzero complete-report request; never a misleading complete table. Keep independently valid new-only evidence. Checks include cardinality/uniqueness, CSV/JSON parity, formula/delta correctness, exact figure coordinates/series, source provenance, and corrupt/missing/wrong-device/nonterminal fixtures.
