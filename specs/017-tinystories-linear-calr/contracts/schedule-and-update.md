# Schedule, Update and Continuation Contract

## Pure schedule

Input: immutable version-1 contract, declared width and integer global p in [0,T]. Reject unknown widths, nonintegral/out-of-domain positions, invalid horizons/counts/exponents. For p<W, LR=.008*p/64. Otherwise LR=.008*(1−(p−64)/(348528−64))**gamma_w. At p=T return exactly zero. Updates are admitted only for p<T. CaLR gamma is `2−1.5*log(C_w/C_min)/log(C_max/C_min)`; uniform gamma=1.

Required analytic anchors: p=0 → 0; p=63 → .007875; p=64 → .008; midpoint decay → .004 uniform, .002 g250 CaLR, .005656854249492381 g1000 CaLR; terminal zero. Include p=65 and T−1 checks, and all intermediate widths. Analytic output is labeled analytic, never observed.

## Nominal/effective rate transaction

1. Validate global position and nominal gamma=1 schedule versus every group; collection rates remain synchronized. Select width via inherited action stream, use the existing active slicing forward/backward and joint clipping cap 1.
2. Compute selected effective LR directly from peak and position. Preserve each stepped group's nominal LR, assign effective LR to every group including common parameters. Never change `initial_lr`, `base_lrs`, weight decay or parameter ownership. Validate effective groups against the selected contract, then capture actual rates/width/count/exponent/owner/position.
3. Enter existing unsafe mutation boundary before AdamW; step shared or selected width optimizer once. Other S2 histories remain unchanged and all optimizers reference one model's parameters. Preserve full-shaped zero-gradient tails and their AdamW behavior.
4. In `finally`, restore nominal group rates on success or error before any clock advance, synchronization or save. Assert restored collection/group rates. Restoring rates does not undo model/history mutation and never clears poisoning.
5. On success advance nominal clock once, synchronize S2, reconcile all counts/cursor/accounting and last actual record, clear unsafe state and publish committed evidence. Post-mutation optimizer, rate-restoration, scheduler or accounting failure aborts with poisoned state and failed attempt; no successful row or resumable partial checkpoint.

Do not bypass synchronization wholesale. Limit effective-rate validation to the explicit new-contract interval. Legacy update paths and C4 ownership/rate corrections remain unchanged. In-memory state is bounded; no whole-model/history copies per update.

## Restore

New-only immutable contracts bind grid, counts/definition, policy/bounds/exponents, effective-rate policy, peak/warmup/horizon, scope, seed/streams and run identity. Validate the entire saved bundle before installing anything. Reconstruct nominal LambdaLR fields with the exact new lambda; validate last actual record at step−1 separately from nominal `last_committed_learning_rates`. Current stored groups at step n are nominal rates at n. Match all model descriptors, histories/counters, RNG/cursor, exposure/tokens and evidence watermarks.

At step zero no applied record exists; at T there is no update. Terminal recovery recreates outputs from full-budget own-arm checkpoint and ordinary validation without any optimizer step. Preserve historical schedule reconstruction and serialized layouts for old contracts.

Verification includes actual all-group AdamW reference updates with nonzero moments/decay, mixed widths, unselected histories, warmup/decay/epoch continuation, malformed contract rejection before mutation and injected failures after optimizer mutation. CPU tensor continuation uses inherited rtol=1e-6/atol=1e-7 unless an existing case is stricter. Real-shape BF16 checks require later GPU authorization and recorded device-specific results.
