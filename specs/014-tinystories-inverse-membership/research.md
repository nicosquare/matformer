# Research decisions — 2026-09-11

## Sampling integration

Decision: reuse `select_random_granularity_index` with isolated `random.choices`
and the ordered .12/.16/.24/.48 weights; retain original randrange for uniform.
Rationale: steps.py already implements fixed-global draws, independent of model
initialization/data. Expected traces and checkpoint replay must use the same
primitive. Alternatives: a new sampler, forced balance or inverse-width weights
would change the requested experiment.

Decision: add compact state to ownership fixed-global runs, with explicit policy
metadata and H=1. Rationale: steps.py selection currently bypasses uniform window
state; ownership accounting reads its exposure_counts, so merely admitting C3
would produce incorrect counts. checkpointing.py ownership restore replays
randrange and needs a weighted branch. Preserve legacy generic fixed-global
absence and uniform state encoding. No new full-state rollback is required.

Decision: narrowly admit fixed_global C3 using resolved random replacement H=1;
omit raw uniform-only schedule/interval settings. Rationale: config.py rejects
explicit uniform-window keys on fixed_global but resolves default cadence.
All other C3 topology/optimizer/single-process/warmup requirements remain.

## Campaign and reporting

Decision: schema 3, exact five IM arms, explicit sampling contract in existing
scientific contract. Keep schema 1 nine arms and schema 2 six corrections strict.
Rationale: optimizer_ownership.py currently hardcodes uniform probabilities and
trace expectations. Existing whole-contract validation already protects restore.
Alternatives: relabeling original runs or loosening old validation would undermine
historical provenance.

Decision: compare only epoch/batch digests across policies; match actions within
the five new arms. Add 20/44 endpoint paths and five-arm progress styling.
Rationale: correction comparisons require identical actions and skip slicing
progress; those rules do not apply to this sampling intervention. Reuse strict
terminal source readers under each campaign's schema and preserve reference files.

## Execution and references

Decision: fresh inverse-membership-sampling-v1 artifact root, gated CPU/GPU source
hashes, new submission identity, existing queue/accounting primitives where safe.
Rationale: old helper is hardcoded to six correction arms/root; it must never be
restarted for this feature. Inspect stricter limits rather than requiring exact
2/4 values. Live queue read was empty on 2026-09-11; this is not launch-time evidence.

Decision: validate historical saved manifests/terminals before comparison. The
runbook contains outdated claims that full training has not run, while the dated
results note records nine completed terminals. Actual saved evidence governs.
The proposed new root was absent on initial inspection. Diagnostics and production
are already authorized, conditional on successful validation; holdout stays sealed.
