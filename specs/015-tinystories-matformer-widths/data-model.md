# Data Model: MatFormer Width Campaign

Records are plain JSON-compatible mappings, YAML controls, CSV endpoint rows and
existing checkpoint dictionaries. These are design contracts, not saved results.
Legacy serialized fields containing `quarter` continue to name A/B/C/D; their
physical meaning is the campaign's declared incremental boundaries.

## Campaign and run

| Entity | Fields and relationships | Validation |
| --- | --- | --- |
| Campaign | `schema_version=4`, `campaign_id`, common controls, nine arms, ordered width grid, pinned data/evaluation, per-run contract hashes, source snapshot/config hashes, artifact root | Exact matrix; fresh independent identities; no changes to schemas 1–3. |
| Run | Campaign/run/arm IDs, stage, seed 42, representation, optimizer scope, clipping, physical dimension, optional source width, endpoint widths, assigned epochs/updates/tokens, config/output paths | Exactly four standalone and five elastic definitions; fresh initialization or own-run continuation only. |
| Scientific contract | Existing `schema_version=1` and required sections; new-only `campaign_schema_version=4`, `width_grid`, `block_boundaries`; complete resolved controls | Existing stable serialization; all extra fields hashed; marker, grid and resolved model agree. Historical contracts gain no defaults. |

Campaign ID: `tinystories-optimizer-ownership-matformer-widths-v1`.
Run ID: `<campaign_id>-<arm_id>-s42`. Short arm labels are ST-g125/ST-g250/
ST-g500/ST-g1000/S1/S2/C1/C2/C3; they are not globally unique.

Each standalone has 1 epoch, 87,132 updates, 713,785,344 tokens. Each elastic has
4 epochs, 348,528 updates, 2,855,141,376 tokens. Total assigned tokens:
17,130,848,256. Stage and report completion are separate records, not deductions
from a directory name or scheduler exit.

## Width and incremental block

| Label | Source fraction | Active FFN dimension | Expected active non-embedding parameters | Blocks |
| --- | ---: | ---: | ---: | --- |
| g125 | .125 | 32 | 90,688 | A |
| g250 | .25 | 64 | 115,264 | A/B |
| g500 | .50 | 128 | 164,416 | A/B/C |
| g1000 | 1 | 256 | 262,720 | A/B/C/D |

`width_grid` is an ordered list of label, source fraction, dimension, expected
count and active-block IDs. Preserve source fractions on dense standalone rows
even though the resolved local dense fraction is 1. Historical grid records are
separate: g250/g500/g750/g1000 at 64/128/192/256, with g750 count 213,568.

`block_boundaries` records `(id, start, end, dimension, supported_widths)`:
A=[0,32), B=[32,64), C=[64,128), D=[128,256). Boundaries are contiguous, strictly
increasing, span 256, and match actual per-layer tensors. Membership counts
4/3/2/1 derive from support; expected uniform activation probabilities are
1/.75/.5/.25. No equality of block parameter counts is assumed.

## Parameter descriptors, owners and histories

Reuse physical descriptors: canonical name, tied aliases, shape, dtype,
trainability, scalar count, component, block ID, and gradient-support widths.
Each C3 owner has ID O-A/O-B/O-C/O-D/O-common, ordered parameter descriptors,
active widths and cap 1.0. Owners partition all trainable parameter objects once;
the common owner includes tied embeddings/head as applicable and common FFN
output bias. S2/C2 histories share model parameter objects across widths.

Let `n[w]` be committed selections, `T=sum(n)`, and
`a[b]=sum(n[w] for w supporting b)`. Required AdamW parameter counters are:

| Scope/representation | Parameter support | Counter/history requirement |
| --- | --- | --- |
| Dense/shared | All | T |
| S1/shared slicing | Full physical tensors | T |
| S2/per-width slicing | All full tensors in history w | n[w]; never-used tail moments zero |
| C1/shared concat | Block b / common | a[b] / T |
| C2/per-width concat | Block active at w / common | n[w] / n[w] |
| C2/per-width concat | Block excluded at w | Absent history |
| C3/block owners | Block b / common | a[b] / T |

Required positive counters require finite moments with exact shapes/dtypes;
required zero counters retain absent lazy state. No manufactured exposure or
allocation to satisfy expected multiplicities. After all widths have appeared,
C2 block histories number 4/3/2/1 and common histories number four.

## Data, sampling and evaluation

Retain tokenizer/corpus/order hashes and role manifest hashes from original pinned
controls. Designated membership is 5,576,448 sequences, 87,132 complete batches of
64 context-128 sequences, with the same 43 excluded sequences every epoch.
Deterministic epoch order and independent action/data seed streams remain fixed.

The elastic sampling record contains ordered widths, probabilities [.25]*4,
global mode, replacement schedule, H=1, derived action seed and no correction.
Standalone sampling has no stochastic action seed. All nine share first-epoch
batches; five elastics share complete four-epoch action/batch traces.

Ordinary evaluation uses the inherited manifest/protocol and
`target_token_weighted_causal_shift_float64`: expected 285 packed sequences and
36,195 causal target tokens, to be revalidated against pinned inputs. Perplexity
is exp(aggregated loss). The controller role stays unused for decisions; final
holdout has zero evaluations.

## Checkpoint and committed observations

Checkpoint records retain existing purpose/schema, model/optimizer/scheduler,
RNG, sampler/cursor, metrics and resource watermark, plus complete scientific
identity and physical descriptors. New topology fields travel in the hashed
contract and must agree with runtime descriptors; an old contract cannot become
a new-grid continuation by relabeling widths.

Committed update records include ordinal, width, batch/epoch identity, tokens,
width/block/owner counts and scheduler position. One record per complete update;
replayed suffixes are reconciled to the restored boundary. Clipping sidecars for
C1/C3 contain width, active flags, measured pre/post norms, coefficients/caps,
combined norms and observation count; metrics/summary paths bind this evidence.
Inactive groups use explicit inactivity/nulls, not fabricated zero observations.

Live update states: `safe → in_flight → safe` only after optimizer, scheduler and
accounting all commit; post-mutation failure transitions to `poisoned → aborted`.
Neither in-flight nor poisoned state can publish a checkpoint. Restore validates
the entire bundle before installation and rolls back installation failures.

## Readiness, barrier and launch attempts

| Record | Required fields | Invariant |
| --- | --- | --- |
| Readiness evidence | Gate kind/status, exact source snapshot hash, config-set hashes, preflight manifest hash, environment, commands/results, diagnostic overrides/identities, GPU job/hardware when applicable | Produced against the tested inputs; prepare cannot relabel stale evidence. |
| Standalone barrier | Campaign/preflight/source identity; four run/contract IDs; terminal checkpoint, sidecar and supporting-source hashes; validation results; content hash | Exactly the four fresh full-budget standalone terminals; revalidate after restart and at elastic worker entry. |
| Submission intent | Unique launch-attempt ID, run, stage, job name, command, config/source hashes, decision, status, optional scheduler job ID | Persist before sbatch; uncertainty blocks duplicate submission. |
| Attempt resource record | Process UUID, run, launch-attempt/job ID, monotonic observation, elapsed/attempted/replayed work, peaks, terminal/failure status, completeness | Sum latest unique-attempt elapsed values; max peaks; unknown measurements remain unknown. |

Campaign execution states:
`defined → prepared → diagnostics_passed → standalones_active → standalones_validated
→ elastics_active → terminals_validated`. A failed or uncertain attempt enters
`reconciliation_required`; it does not reset the run. Resume decisions are
`fresh`, `resume_from_valid_own_checkpoint`, `completion_only`, or `blocked`.
Diagnostics occupy separate run identities and cannot advance production stages.

## Endpoint and report

Endpoint primary key is `(campaign_id, run_id, source_fraction,
active_ffn_dimension)`. Retain width label as metadata, not the physical identity.
Fields include group (`elastic`, `standalone_matformer`, `standalone_historical`),
canonical arm, seed, exact count/convention, loss/perplexity, actual/assigned
epochs/updates/tokens, terminal checkpoint path/hash/step, evaluation manifest/
protocol/role/target counts, config/source hashes, exposure, clipping and resource
references/values. Not-applicable fields are null with a clear applicability
meaning; missing required evidence is an error.

New reports require nine runs and 24 endpoints. Combined reports require 28
endpoints and thirteen runs, including eight distinct standalone records. No
deduplication at dimensions 64/128/256. CSV nested fields use the existing stable
JSON serialization, allowing parsed CSV/JSON semantic equality.

Maintain separate `production_status`, `new_report_status`, and
`combined_report_status`, with artifact hashes and explicit outstanding/error
reasons. New-report completion survives unavailable/incompatible history;
combined completion requires all four valid old standalones. Directory existence
does not establish completion. Publication is staged/atomic and resume revalidates
existing manifests before reuse.
