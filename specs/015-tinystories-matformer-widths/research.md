# Research: MatFormer Width Campaign

**Date**: 2026-09-21. Evidence: read-only source/document inspection and environment
metadata. No Feature 015 model-count preflight, training tests, corpus audit,
historical terminal validation, or GPU work was performed.

## 1. Width interpretation and FFN support

**Decision**: Fractions [.125, .25, .5, 1] at full FFN 256 give prefixes
32/64/128/256 and incremental blocks 32/32/64/128. Reuse explicit-prefix models
and the ordinary dense standalone constructor.

**Rationale**: The authors' [implementation](https://github.com/devvrit/matformer/blob/main/modified_llama.py)
sets these scale factors; [paper section 4.1](https://papers.nips.cc/paper_files/paper/2024/file/fe066022bab2a6c6a3c57032a1623c70-Paper-Conference.pdf)
specifies FFN-to-model ratios .5/1/2/4. These establish the grid, not this campaign's
training protocol. Locally, `get_concat_block_metadata` in
`src/models/granularity.py` takes successive prefix differences; `CatLlamaMLP`
in `src/models/ffn.py` allocates variable-sized blocks and describes physical
parameter support. Existing base-block alignment admits this grid.

**Alternatives considered**: Total parameter fractions misinterpret the request.
A new FFN implementation is unnecessary; splitting C/D into extra clipped owners
changes C3. Required counts 90,688/115,264/164,416/262,720 remain expectations until
actual dense/slicing/concat preflight verifies them.

## 2. Campaign schema and identity

**Decision**: Add recipe schema 4 and campaign
`tinystories-optimizer-ownership-matformer-widths-v1`. Retain short arm names,
resolving them within the campaign schema. Add new-only version/grid/boundary
metadata to scientific contracts while retaining the schema-1 serializer.

**Rationale**: `campaign_arms`, `expand_campaign`, and
`validate_materialized_config` in `src/evaluation/optimizer_ownership.py` select
among schemas 1–3; the last currently searches arms by label alone. Shared labels
now require schema-qualified selection. `build_optimizer_ownership_signature` in
`src/utils/reproducibility.py` already retains/hashes extra fields. The proposed
name satisfies existing campaign-ID validation. Global replacement of `WIDTHS`,
`WIDTH_LABELS` or pinned common controls would corrupt historical behavior.

**Alternatives considered**: Schema 1 cannot distinguish physical grids; schema 3
implies inverse-membership sampling. Renaming all short arms is unnecessary when
full identities and schema dispatch are explicit.

## 3. Topology eligibility

**Decision**: Add a schema-4 opt-in before config resolution; thread its validated
layout into partition construction. Default callers retain equal-quarter
validation. Check the exact prefixes and each component's actual block shape.

**Rationale**: Blockers are `config._validate_optimizer_state_eligibility`,
`optimizer_state._validate_concat_quarters`/`build_concat_parameter_partition`,
and the equal-grid C1 clipping-group predicate in `steps.train_for_steps`.
The expander currently attaches the scientific contract after resolution; late
contract metadata alone cannot admit C3. Set `run.campaign_schema_version=4`
during expansion and check it with exact grid/arm/sampling/correction/ownership
controls, then bind the resolved marker to the complete contract. Do not infer
eligibility from run-name prefixes. Existing physical descriptors already
deduplicate ties, include segment biases and classify common output bias.
Retain serialized `quarter_id` and activation keys as incremental-block IDs.

**Alternatives considered**: Globally allowing unequal prefixes relaxes old
contracts. A topology registry adds indirection without an experimental need.

## 4. Sampling, metrics and clipping

**Decision**: Keep `global`/`random_with_replacement`/H=1 and the isolated
`randrange` selection stream. Derive traces, observation validation, summary
exposure and compact attempt-label validation from the selected grid.

**Rationale**: `expected_action_trace`, `inspect_run_observations`,
`_accumulate_clipping` and report helpers use legacy constants.
`run.build_ownership_run_summary` imports old labels for exposure and temporary
storage estimates. `MetricsAccumulator._attempt_ordinal` explicitly accepts
only g250/g500/g750/g1000; g125 needs contract-aware validation that preserves
bounded compact accounting and old checkpoint compatibility.

The historical literal-arm clipping omission is already fixed in
`optimizer_ownership_metric_fields`, `append_optimizer_ownership_observation`,
the summary and `_inspect_terminal_run`: concat plus non-per-granularity scope
requires sidecars. Preserve this predicate and test new identities across resume,
metrics references, terminal reading and missing-sidecar rejection. The remaining
equal-grid C1 group construction still needs decision 3.

**Alternatives considered**: Equal weights in `fixed_global` use a different RNG
primitive. Forced balance and eager lazy-history allocation violate the protocol.

## 5. Storage and restore

**Decision**: Reuse actual allocation measurement and support-driven history
validation. Hash explicit physical boundaries with the full run identity;
preserve staged whole-bundle restore, installation rollback and poisoned-save
rejection.

**Rationale**: `validate_campaign_optimizer`, `validate_adamw_history`, and
`measure_optimizer_storage` already use physical support and actual tensor
elements/dtypes. For FFN block parameter counts F_A–F_D and common count R, C2
moments after every width appears are
`2*(4*F_A + 3*F_B + 2*F_C + F_D + 4*R)`. With bias-free shapes the weighted FFN
term is 1.875F; the old 2.5F assumes equal quarters. Counters, temporaries and
device peaks remain separate measurements.

`checkpointing._ownership_identity`, `_validate_ownership_payload`, and
`_load_ownership_checkpoint` stage identity, descriptors, numerical state,
required histories, clocks, RNG, data and metrics validation. `steps.py` marks
updates unsafe before owner mutation; `assert_checkpoint_safe` rejects partial
publication. Extend inputs/coverage without another checkpoint implementation.

**Alternatives considered**: A global checkpoint schema bump without layout
changes needlessly migrates history. Per-step model/optimizer copies are
unnecessary: abort after partial mutation and resume the last durable update.

## 6. Standalone barrier and attempts

**Decision**: Use a focused campaign launcher with strict standalone terminal
validation before elastic submission and at worker entry. Persist a barrier bound
to all four terminal identities. Track unique launch attempts and own-checkpoint
resume/completion-only decisions; reuse operational primitives where valid.

**Rationale**: `run_tinystories_inverse_membership.py` supplies snapshot/lock,
intent and live-limit patterns, but its queue treats Slurm completion plus sidecar
existence as completion until finalization. That cannot establish this barrier.
Its attempt=1 and fixed 348,528-step ETA assumptions also need replacement.
Reconcile ambiguous submission using queue/accounting/worker evidence before
retry. Validate report manifests rather than trusting directory existence.

`run.ResourceAttemptLedger` sums each process UUID's latest duration, takes max
peaks and marks unfinished costs incomplete. Link launch/job identities without
double-counting overlapping scheduler and process time.

**Alternatives considered**: Scheduler dependencies alone do not establish valid
terminal evidence. Copying a third entire queue retains obsolete assumptions;
a generic job framework is excessive. Use an explicit stage loop and narrowly
extract repeated operations only when needed.

## 7. Readiness evidence and diagnostics

**Decision**: Gate evidence must originate from the exact tested snapshot,
config set and preflight manifest; preparation verifies it. Separate short
real-control diagnostics from controlled small-epoch deterministic probes.

**Rationale**: Existing IM `prepare` adds the current snapshot hash to passed CPU
evidence, which cannot prove that snapshot was tested. Its 192→256 GPU run is
useful smoke/resume evidence but does not cross an epoch boundary or compare
numerical state against uninterrupted execution. Cover before/at/after synthetic
epoch boundaries through the real sampler/trainer, plus real-shape/batch/bf16
checks retaining production scheduler horizons. Cover all nine model definitions
and every elastic width. Forced actions are confined to separate semantic probes;
diagnostic identities/data overrides cannot satisfy production barriers.

**Alternatives considered**: A full production epoch solely to test a boundary
wastes resources. Relabeling old gates or counting CUDA skips as passes does not
establish readiness. All GPU work requires later authorization and sbatch.

## 8. Historical selection and reports

**Decision**: Extend freeze/report to 24 new endpoints, then add
`report-matformer-widths` for 28. Validate original frozen-manifest/preflight
integrity and only the four required standalone source sets using strict original
terminal rules. Preserve existing callers' whole-campaign validation.

**Rationale**: `_validated_comparison_sources` currently requires historical
elastic artifacts too. `_terminal_endpoints`, `_endpoint_table`, figures and
interpretations assume the old grid. Schema-aware records and keys
`(campaign_id, run_id, fraction, dimension)` preserve shared-size repeats.
Historical g750 remains dimension 192/count 213,568. Filled/open concentric
markers distinguish coincident groups without moving exact coordinates.

**Alternatives considered**: Arm/width-only or count-only indexing collapses
measurements. Historical elastic/IM/corrected curves are outside scope. Missing
references leave the new report intact and combined comparison outstanding;
invalid present inputs must not be silently skipped.

## Resolution and evidence limits

All design unknowns are resolved from the specification, primary width sources
and local code. Pinned interpreter/package metadata matches the prior environment.
Root availability, actual model counts, input hashes, Slurm limits and historical
terminals remain runtime validation gates. Prior verification records establish
neither new readiness nor authorization. No dependency or scientific clarification
is needed.
