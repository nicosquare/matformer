# Campaign and Topology Contract

## Fixed recipe and identities

Create `configs/controlled_exps/tinystories_instruct_matformer_widths.yaml` with
the existing five top-level keys: `schema_version: 4`, `campaign_id`, `common`,
`arms`, `expected_data`. Campaign ID is
`tinystories-optimizer-ownership-matformer-widths-v1`. Use normal trainer YAMLs
materialized by the existing expander; no second configuration engine.

| Arm | Resolved representation | AdamW history scope | Clip | FFN dimension | Epochs |
| --- | --- | --- | --- | ---: | ---: |
| ST-g125 | dense | shared | global 1 | 32 | 1 |
| ST-g250 | dense | shared | global 1 | 64 | 1 |
| ST-g500 | dense | shared | global 1 | 128 | 1 |
| ST-g1000 | dense | shared | global 1 | 256 | 1 |
| S1 | slicing | shared | global 1 | 256 | 4 |
| S2 | slicing | per_granularity | global 1 | 256 | 4 |
| C1 | concat | shared | global 1 | 256 | 4 |
| C2 | concat | per_granularity | global 1 | 256 | 4 |
| C3 | concat | per_ffn_block | independently 1 per active owner | 256 | 4 |

Dense is a resolved representation of the ordinary standalone constructor, not
a new `model.variant` value. Retain original standalone source-width conversion
and its local fraction 1. Run IDs are `<campaign_id>-<arm>-s42`; output directories
are distinct and reserved before production. All nine initialize normally from
seed 42. No historical/diagnostic trained state initializes a production run.

New common model controls have `granularity_mode: explicit`, ordered labels
g125/g250/g500/g1000 and prefix mapping .125/.25/.5/1. Model dimension 64, four
layers/heads, context 128, vocabulary 2048 and initializer .02 remain unchanged.
Keep original AdamW .008/(.9,.95)/1e-8/.1, batch 64, accumulation 1, bf16, single
process, LR scaling none, cosine with 64 warmup updates and full assigned horizon.
Pre-nested warmup and membership corrections stay disabled. Validate every
inherited common control and pinned data hash, not a hand-picked subset.

Elastic overrides retain `run.sampling_mode: nested-random`,
`model.granularity_sampling_mode: global`,
`global_sampling_schedule: random_with_replacement`, and
`global_sampling_interval_steps: 1`. Derived probabilities are exactly [.25]*4;
do not switch to fixed_global. Standalone overrides omit elastic-only sampling
fields. Use independent existing RNG streams; no balancing, holding or weighting.

## Schema-scoped resolution and topology admission

Add explicit schema-specific grid/arm/common-control selectors in
`src/evaluation/optimizer_ownership.py`. Schemas 1–3 retain current constants,
default behavior and hashes. A selector rejects unknown versions. Arm lookup is
by schema and label; matching the short label alone is insufficient.

The expander places `run.campaign_schema_version=4` into new raw configs before
`resolve_run_config`. Elastic validation requires the exact four-prefix new grid
and declared arm/ownership/sampling/correction controls. Standalone validation
instead checks the schema-4 source-width mapping, matching dense dimension and
resolved local fraction 1; it must not require four local widths on a dense model.
The marker is not a general unequal-topology flag. After resolution, include `campaign_schema_version=4`,
`width_grid` and `block_boundaries` in the schema-1 scientific contract and require
the marker/config/contract to agree. Do not stamp this field into old configs or
infer it from IDs. Missing/mismatched schema metadata in new materialized configs
fails validation. Recipe and run metadata must agree on campaign and schema.

Pass the validated topology explicitly to `build_concat_parameter_partition` and
C1 diagnostics. An omitted topology preserves current equal-quarter validation.
The new layout is exactly A=[0,32), B=[32,64), C=[64,128), D=[128,256), with active
prefix support. Validate gate/up shapes `(block_dimension,64)`, down shapes
`(64,block_dimension)`, optional block biases `(block_dimension,)`, and common
down bias `(64,)` across all four layers. Reject absent/extra blocks, changed
prefix order, shape mismatch, mixed layouts and noncontiguous support.

C3 owns O-A/O-B/O-C/O-D/O-common, deduplicating tied objects and covering every
trainable parameter exactly once. Do not divide C/D or scale their caps. S1/S2
retain full-tensor gradients/state; C1/C2/C3 retain absent inactive-block gradients.
Active present zero gradients retain AdamW behavior. Per-width histories operate
on the shared model parameters, with C2 lazy multiplicity 4/3/2/1 and common 4
only after every width has actually appeared.

## Counts, budgets and data

CPU preflight constructs all nine real models; inspect every elastic width and
all four dense sizes. Require active non-embedding counts
90,688/115,264/164,416/262,720, excluding embeddings and LM head while retaining
other common parameters. Preserve physical source fractions even for dense local
fraction 1. Fail discrepancies instead of replacing expectations.

Retain original audited tokenizer, corpus, role manifests and packing/order.
One epoch is 5,576,448 designated sequences, 87,132 updates, 713,785,344 tokens;
the fixed excluded tail is 43 sequences. Standalone budgets are one epoch;
elastic budgets are four, 348,528 updates and 2,855,141,376 tokens. No changed
horizon, epoch-reset scheduler, rotated exclusion or max-step cap is permitted.

Preflight generates full expected action/data digests without training: all five
elastic action sequences match; all nine first epochs and all five four-epoch
batch sequences match. Expectations of 87,132 selections per width and block
activations 348,528/261,396/174,264/87,132 are statistical expectations, not quotas.

Ordinary validation runs every 64 updates and at completion under inherited
target-token weighting. Audit disjoint roles; controller data never guides
training/selection and final holdout remains sealed.

## Acceptance boundary

Reject altered matrices/controls/budgets, wrong actual counts or boundaries,
nonuniform probabilities, corrections, earlier-campaign/diagnostic initialization,
and occupied or incompatible production identities. Preflight publishes success
only after every check, with configs, source/environment provenance, counts and
expected traces. It does no training or evaluation. Re-entry for continuation
uses a separate validated own-run path, never bypasses fresh-identity preflight.
