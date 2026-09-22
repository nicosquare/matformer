# Contract: Protocol, Grids and Scheduler

## Fixed new protocol

Campaign schema is 5; campaign ID is `tinystories-optimizer-ownership-s1-warmup-v1`. Only `S1-linear-w256` and `S1-geometric-w256` are accepted. Run IDs are `<campaign_id>-<arm_id>-s42`; output directories are `<fresh-root>/runs/<arm_id>`. Use `grid_id=linear` or `geometric` in new configs/contracts and bind reference S1 identities in intervention metadata.

| Arm | Ordered labels | FFN dimensions | Reference campaign schema |
| --- | --- | --- | --- |
| S1-linear-w256 | g250, g500, g750, g1000 | 64, 128, 192, 256 | 1 |
| S1-geometric-w256 | g125, g250, g500, g1000 | 32, 64, 128, 256 | 4 |

Model and optimizer controls remain EX-002: d64/l4/h4, full FFN256, context128, vocab2048, initializer .02; slicing/shared AdamW, betas (.9,.95), eps 1e-8, weight decay .1; global L2 cap1, batch64, accumulation1, bf16, no LR scaling or activation checkpointing. One uniform global draw with replacement per update (H=1); correction none; disabled pre-nested warmup. No other arm or sampling intervention is accepted.

Each run has exactly four epochs, 348528 updates and 2855141376 training tokens. One update is 8192 tokens; one epoch is 87132 updates/713785344 tokens. Preserve the designated 5576448 sequences and excluded tail of 43, disjoint ordinary/controller/final roles, and validation every 64 updates plus completion. Evaluation target tokens are separate from training tokens.

## Resolution and compatibility

The new recipe has exact top-level keys `schema_version`, `campaign_id`,
`common`, `arms`, `expected_data`, and `references`. `common` is the inherited
Linear common configuration with only `training.warmup_steps=256` changed.
`arms` contains exactly the two declared entries, each with its fixed `grid_id`
and S1 overrides; the Geometric entry additionally replaces the ordered
granularities/prefix map with its declared grid. `references` maps each grid to
historical campaign ID/schema, S1 config SHA-256 and the exact five selected
arms. CLI reference roots locate those identities; they cannot change the
selection or pinned controls. Expansion validates the fully merged result
against the fixed definition and its saved counterpart. Do not copy a saved
resolved contract wholesale into a fresh configuration.

`campaign_arms(5)` returns exactly two definitions. Extend selectors such as `campaign_widths(schema_version, arm_id=None)`, `campaign_common(...)` and `campaign_topology(...)` with strict arm-qualified schema-5 behavior; old omitted-argument calls retain their outputs. Missing/unknown arm for schema 5 fails. New manifests store per-arm grids, not one top-level grid inferred from schema. Validate topology against declared support and actual models at counts 90688/115264/164416/213568/262720 as applicable.

Old `PINNED_COMMON`, schema1–4 resolution and scientific signature serialization remain unchanged. New fields are added only to schema-5 contracts. Schema5 warmup 256 acceptance follows validated campaign/arm identity; passing an arbitrary expected warmup is not an escape hatch. Schema5 with64 and old schemas with256 must both fail.

Compare each resolved new config with the saved resolved counterpart. Implement a closed list of changed field paths, including only:

- `training.warmup_steps`, its resolved warmup fields, and consequential scheduler-specific values.
- Campaign/run/arm/grid/protocol/intervention/reference identities and their hashes.
- Output/config locations, current source provenance and attempt/execution metadata needed for a fresh run.

Enumerate concrete resolver-derived metadata paths in the audit; do not ignore entire model/training/data/evaluation/reproducibility sections. Changes to peak LR, horizon, seeds, data/tokenizer/evaluation membership, clipping, sampling, representation, precision or initialization policy fail. Source changes are provenance, never permission to change scientific behavior.

## Schedule semantics

Let T=348528, peak=.008, w=256 and p be scheduler position:

```text
L(p) = .008 * p / w                              for 0 <= p < w
L(p) = .004 * (1 + cos(pi * (p-w) / (T-w)))      for w <= p <= T
```

The inherited Transformers cosine convention uses half a cycle. Initialization stores p=0 and LR0. One-based update u applies L(u−1); one scheduler advance after the optimizer produces stored p=u. The warmup is inside the budget; cosine spans T−w=348272 positions to the existing terminal endpoint.

| Position p | New L(p) | Meaning |
| --- | ---: | --- |
| 0 | 0 | Initialization; applied at update 1 |
| 63 | .00196875 | Applied at update 64 |
| 64 | .002 | Applied at update 65 |
| 65 | .00203125 | Applied at update 66 |
| 255 | .00796875 | Applied at update 256 |
| 256 | .008 | First peak, applied at update 257 |
| 257 | approximately .00799999999983726 | Applied at update 258 |
| 348527 | approximately 1.6273915548481455e-13 | Last applied LR at update 348528 |
| 348528 | 0 | Terminal stored LR, no next training update |

Original w64 first applies peak at update 65. Preserve that historical behavior. Export all 348529 positions with `scheduler_position`, `learning_rate`, `applied_update` (null at terminal), warmup and horizon. Both new arrays and hashes agree. Boundary evidence records actual pre-optimizer LR and post-commit position separately; reconstructed values never count as measured execution.

## Streams and continuation

Keep `derive_seed/seed_for` root42/stream/version1 unchanged. Preserve `random.Random(action_seed).randrange(4)` for uniform actions and the packed sampler's deterministic data-seed 42 epoch ordering. Each new labeled action trace must exactly match its own counterpart across all 348528 updates. Batch traces must match each own counterpart over all four epochs. No requirement equates Linear and Geometric labeled action hashes or forces expected selection counts.

Fresh construction uses the unchanged model-initialization policy; no reference weights are loaded. Own-run resume validates full contract/run/grid/warmup/horizon, model, moments/counters, scheduler/rates, randomness, sampler cursor and accounting before live mutation. Reject cross-run/grid, old 64-update warmup, model-only, malformed and non-finite bundles. Failure after optimizer/scheduler/accounting mutation cannot publish a new durable checkpoint. Full-horizon recovery recreates missing terminal outputs without an update.

Acceptance covers positions/updates around 64 and 256, before/at/after epoch boundaries 87132/174264/261396, terminal 348528, all widths, shared inactive-tail momentum/decay, clipping, exact stream suffixes and inherited numerical tolerances. State-seeded short probes retain real horizon and are labeled diagnostics; actual committed full-budget evidence remains required.
