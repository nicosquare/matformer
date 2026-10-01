# Data Model: Linear S1/S2 CaLR

## Campaign and run

Campaign fields: schema_version=6, campaign_id, fresh root, four ordered arms, seed=42, source snapshot/hash, recipe/resolved hashes, inherited control/data identities, reference mappings, assigned totals. One campaign has four runs; each run has arm_id/run_id, slicing representation, optimizer_state_scope (`shared` or `per_granularity`), exponent_policy (`uniform` or `complexity_log`), output identity, schedule contract and independent statuses.

Exactly `S1-linear-poly`, `S1-linear-CaLR`, `S2-linear-poly`, `S2-linear-CaLR` are legal. Budget per run is 348528 updates/2855141376 packed tokens; campaign totals are 1394112/11420565504. Initialization/action/data streams are inherited and independent of names.

## Schedule contract and width complexity

Contract version 1 includes family=`warmup_polynomial`, position convention=`pre_update_zero_based`, peak=.008, warmup=64, horizon=348528, exponent_policy, complexity_definition, ordered width→FFN/count/reporting-count/exponent maps, gamma_min=.5/gamma_max=2 for CaLR, nominal_exponent=1, and effective_rate_policy=`temporary_all_groups_restore_nominal_v1`. Hash immutable contents. Uniform arms store all gamma=1; bounds must not silently affect them.

Width rows are g250/g500/g750/g1000 with FFNs 64/128/192/256; counts/exponents are in research.md. Positive counts, distinct extrema for CaLR, exact grid/model agreement, finite positive exponents and 0<W<T are required. Reporting counts never substitute for complexity.

## Committed update

Fields: run/arm/attempt identity, one-based committed step k, pre-update position p=k−1, selected width, owner, complexity, exponent, actual all-group applied LRs (one equal finite nonnegative value), action ordinal, batch/cursor/hash evidence, 8192 packed tokens, width exposure and resource watermark. One successful update advances the global clock once; only selected S2 history advances. Failed attempts are separate records.

Nominal current/previous clock rates remain distinct from measured applied rates. At commit step n, nominal prepared position is n; last applied record is position n−1. At n=0 there is no last-applied record. At n=T all nominal rates are zero and no next update exists. Compact checkpoint evidence includes the last applied record/hash and trace watermark; full records stream to JSONL rather than accumulating in memory.

## Durable checkpoint

Own-run identity, scientific/schedule/optimizer contracts and hashes; full weights; shared history or four width histories over identical parameter objects; nominal clock state; last applied record; width/owner counters; action/data/global RNG; sampler cursor; epoch/update/token totals; metrics and resource watermarks. A checkpoint is eligible only at a reconciled safe boundary. Reject cross-arm or changed contract before live mutation. Post-mutation failure poisons live state; latest durable checkpoint stays authoritative.

## Reference terminal

Fields: discovery path, actual run/config identity and hashes, source identity, saved schema/adapter, schedule/peak/warmup, grid/counts, expected/actual budgets, terminal checkpoint hash/step, ordinary role/manifests/aggregation, endpoint/evaluation-source hashes and available trajectory paths. Eight references contribute 20 endpoints. States: discovered → inspected → validated, or missing/invalid. Legacy counters that were not maintained cannot prove exposure. No adapter writes into references.

## Comparison artifact

Endpoint primary key: run identity + width + terminal evaluation identity. Fields: loss, perplexity, both counts, grid/FFN, technical schedule/policy/exponent, scope, peak/warmup, seed, budgets and provenance. Pair key identifies comparison family + left/right endpoints; fields include delta_loss, perplexity_ratio and relative_gap. Interaction key is width and the four new endpoints. Plot-source manifest binds measured data and every figure to validated identities.

## Readiness and attempts

Readiness includes source/config/snapshot hashes, CPU gate, all-four-arm CUDA gate, real shape and BF16 proof, commands/exits/pass/fail/skips and log hashes. Stale source/config invalidates readiness. Diagnostic and production attempts have separate purpose, durable ID, submission intent/job/worker status, durable checkpoint and cost records.

Run lifecycle: unprepared → prepared → ready → submission_pending → submitted → running → terminal_pending → terminal_validated. Failed/uncertain submissions and failed execution remain explicit; own durable state can support a new continuation attempt. Readiness, execution, new-terminal and comparison status remain independent. Completion requires both successful worker and matching scheduler evidence, not job disappearance. Missing terminal outputs at full budget transition through recovery with zero training updates. Comparison: pending → references_validated + four_new_terminals_validated → published → accepted; missing evidence remains incomplete.
