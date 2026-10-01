# Research: TinyStories Linear S1/S2 CaLR

Date: 2026-10-01. Read-only research; no training or full checkpoint validation.

## Evidence and boundaries

Inspected Feature 013 lifecycle/checkpoint contracts and verification, Feature 016 plan/contracts/verification, the source request and design note, and current schedules, optimizer-state, update, checkpoint and campaign code. Two research agents independently inspected runtime integration and campaign/reference paths. Installed metadata confirms Python 3.12.13, torch 2.11.0+cu128, transformers 5.8.0, numpy 2.4.3, PyYAML 6.0.3, matplotlib 3.10.9 and pytest 9.0.3. Proposed output root was absent; recheck atomically at preparation.

The motivating [author paper](https://arxiv.org/html/2503.10740v1) describes complexity-dependent decay and separate momentum for CNN NAS. This feature's AdamW histories, fixed exponent bounds and warmup are declared ElasticNN adaptations. Paper results do not establish language-model improvement. The feature specification supplies the exact experiment formula and controls.

## 1. Schedule integration

**Decision**: Use a custom exact gamma=1 polynomial LambdaLR as the nominal global clock for both policies. For CaLR only, compute the selected-width effective rate directly from peak/position/exponent, temporarily assign every group in the stepped optimizer, capture the actual rate and step, then restore the original nominal group rates in `finally` before clock advance/synchronization. Uniform polynomial follows the same validated contract with exponent one.

**Rationale**: `PerGranularityOptimizerCollection` enforces synchronized rates across owners; `GlobalSchedulerClock` is a scalar carrier scheduler. Current training steps the selected optimizer directly. The collection invariant can remain intact before/after the narrow temporary interval. S1's raw LambdaLR also needs restoration before advancement. An exact custom lambda avoids library polynomial defaults/minimum LR and base-rate restrictions.

**Alternatives considered**: A width-aware clock broadcasting effective rates to every owner introduces preparation-width checkpoint state and changes rate reconstruction. Multiplying the current group rate accumulates stale scaling and fails at zero; disabling collection validation weakens safeguards. Independent S2 clocks change the experiment.

The global clock's `last_committed_learning_rates` remains nominal reference evidence. A separately versioned applied-update record stores actual width/rates/position. Never relabel the nominal field as observed CaLR LR. Keep `initial_lr` and scheduler `base_lrs` unchanged. Restore all group values even on exceptions; restoration is not weight/history rollback and does not clear poisoning.

## 2. Complexity and exponents

**Decision**: Define `active_trainable_scalars_including_embeddings_head_tied_once_v1`; count identity-deduplicated trainable common parameters plus active slicing entries. Store separate reporting counts excluding embeddings/head. Validate against actual model descriptors during preflight.

| Width | FFN | Complexity | Reporting count | CaLR exponent |
| --- | ---: | ---: | ---: | ---: |
| g250 | 64 | 377408 | 115264 | 2.0 |
| g500 | 128 | 426560 | 164416 | 1.4432006280490555 |
| g750 | 192 | 475712 | 213568 | 0.9471931630317243 |
| g1000 | 256 | 524864 | 262720 | 0.5 |

**Rationale**: The current campaign reporting convention excludes input embeddings/LM head; it cannot silently become schedule complexity. The formula includes shared parameters and tied identities once.
**Alternatives considered**: Full storage count, width fraction, FFN-only or reporting counts change the treatment. Model parameter copies are unnecessary; use existing physical descriptors/prefix geometry.

## 3. Configuration, factor audits and streams

**Decision**: Add campaign schema 6 and schedule contract version 1, with four exact arm IDs and explicit semantic representation/ownership. Resolve declared fields into new-only scientific/optimizer/schedule identities. Preserve old schema expansion, field serialization and signature fixtures.
**Rationale**: Current branches include literal S1/S2 eligibility and fixed campaign schedules; new names must enter the same fatal-update boundary and all historical behavior must remain readable. Compare each arm to original matching-scope LR .008 cosine config, each exponent-policy pair, and each ownership pair through separate closed path allowlists. Consequential identity differences are allowed explicitly, not by broad wildcard.
**Alternatives considered**: Renaming saved arms, reusing schema 5 or modifying global serializer defaults risks historical signatures. Stream seeds derived from new run names break matching; retain the original stream policy independently of identity.

## 4. Checkpoint validation and failure safety

**Decision**: Extend `_validate_ownership_payload` only under the new schedule contract. Reconstruct nominal exact-zero gamma=1 rates without mutation and separately validate last actual width/owner/exponent/rates at p=step−1. Bind contract hash, complexity and exponent maps, ownership, grid and run identity before installing model/optimizer/RNG/cursor/accounting. Keep current collection layout and clock fields where their nominal meaning is unchanged.
**Rationale**: The validator currently reconstructs private carrier lambda rates and assumes current/previous values are scalar. Applied rates require a distinct record. The existing whole-bundle validation and poisoned-state refusal already prevent post-mutation partial saves.
**Alternatives considered**: Per-step full optimizer/model copies are too costly; restoring LR alone does not undo AdamW. Cross-arm checkpoint migration would violate fresh/own-arm semantics. No additional update is allowed for terminal recovery.

## 5. Lifecycle and queue

**Decision**: Reuse pure Feature 016/ownership operational helpers for locks, intents, immutable source snapshots, attempt ledgers, CUDA-required entry and live Slurm limits; expose a focused four-arm prepare/queue/worker/report launcher.
**Rationale**: Partition `cscc-gpu-p`, QoS `cscc-gpu-qos`, exclusions and user-wide limits are already visible in launchers. Queue authorization and passing source/config-bound CPU plus all-arm real-shape GPU evidence are separate gates. Snapshot creation and reservation must remain explicit; no GPU submission during planning.
**Alternatives considered**: A generic campaign engine obscures a fixed experiment. Four independent unrestricted sbatch calls miss duplicate/uncertain outcomes and user-wide limits. Earlier passing gates do not bind changed scheduler source.

## 6. Reference provenance and reports

**Decision**: Select eight saved runs read-only: original S1/S2 at .008, supplemental S1/S2 at .004 and four matching standalones. Use native ownership terminal validation where available and a separately tested legacy adapter based on actual saved config, terminal checkpoint, completed budget, terminal ordinary-evaluation metrics and manifests. Missing/inconsistent proof remains an explicit incomplete dependency; do not synthesize newer manifests or rewrite references.
**Rationale**: Discovery found the original six runs have newer campaign/terminal/trace artifacts; both .004 runs have config/summary/metrics but lack those newer artifacts. A uniform native validator would reject valid legacy layouts or tempt fabricated evidence. A summary alone is insufficient; dormant legacy ownership counters must not be treated as measured counts.
**Alternatives considered**: Aggregate report tables or design-note numbers do not establish terminal provenance. Retraining is outside scope. Replacing missing observed LR/progress with analytic curves falsely implies execution.

Publish 36 endpoints, eight CaLR−poly, eight S2−S1, sixteen new−matching-scope .008 cosine and sixteen new−standalone pairs; four interactions. Supplemental .004 comparisons may be additional distinctly labeled rows. Primary/supplemental panels have seven exact labels with consistent styles; standalone points remain unconnected. Interpretation must cover every width, practical references, g1000 gap, exposure/horizon differences and cumulative LR/AdamW decay; no one-seed robustness claim.

## Resolved questions versus remaining acceptance work

No unresolved implementation-design clarification remains. Actual-model counts, full corpus hashes, saved terminal checkpoint integrity, observed trajectory availability, output-root reservation, CPU/GPU readiness and production results are future verification work. Their absence now must not be represented as passing acceptance evidence.

## Inspected saved reference identities

Actual config SHA-256 values reported by read-only discovery:

| Reference | Config SHA-256 |
| --- | --- |
| Original S1 .008 | `64c3f6c4309b114776a93a58f9b794be7d77e7902ccdcef53eb004802ef799e3` |
| Original S2 .008 | `3d7eb97093c9f6175ae4cc600bcb539481ebf2f3edd655d7e5fd85f6e6a7488a` |
| Supplemental S1 .004 | `f15871ca6b6b0c4743595276adcdebfa43c55a570bafccc8ff00904977ddd32a` |
| Supplemental S2 .004 | `4d045660a93ad7ea200ce548381f6496622955b485b37c40528086039d6f3cf3` |

All four inspected configs share train manifest `c06270adebde4c5456517a5bcf266b2e4520587fb928541333009735ee2e60dd` and ordinary-validation manifest `0c1beea552f54941e397d2442de736b1586e0292f6b1271b62d27ad782627856`, AdamW betas .9/.95, epsilon 1e-8, decay .1, BF16, accumulation 1 and global cap 1. Supplemental scopes are shared/per_granularity with .004/64. This is not the full closed-difference or terminal audit.

Both supplemental latest checkpoints exist; summaries pin terminal paths/hashes and resumable purpose. Existing `scripts/plot_tinystories_s2_peak_lr.py::candidate_endpoints` checks completed summaries, committed counts, aggregation, checkpoint hashes, scaling endpoints and validation tokens/manifests. `scripts/plot_tinystories_s1_peak_lr.py::candidate_rows` and `read_curves` supply terminal measured-metric consistency and curve checks. Extend those checks in the dedicated adapter with actual checkpoint metadata and complete identity/control validation. Tuned S1 has stale continuation global-clock/optimizer-total fields despite a completed summary; never use these dormant fields as scientific proof or impose the new restore contract on this legacy reference.

Current early-observation helpers can reconstruct missing historical LR values. They must not be reused unchanged for measured-LR plots: retain actual recorded rates only, with absent evidence explicitly disclosed.
