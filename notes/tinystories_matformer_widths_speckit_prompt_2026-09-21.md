# Spec Kit prompt: Feature 015 — TinyStories optimizer ownership with original MatFormer widths

Date: 2026-09-21

Use this document as the feature description for **Feature 015**, extending the
uncorrected optimizer-ownership comparison from Feature 013. Preserve Features
013 and 014, their scientific contracts, and their saved experiments.

Suggested invocation:

```text
/speckit-specify Use notes/tinystories_matformer_widths_speckit_prompt_2026-09-21.md as the Feature 015 description. Preserve the standalone-first execution order, original MatFormer width grid, uniform width sampling, nine fresh runs, and 28-endpoint historical-standalone comparison. Respect the authorization boundary below. The feature request begins at "Feature request".
```

## Feature request

Repeat the original uncorrected S1/S2/C1/C2/C3 comparison on TinyStories-Instruct
using the original MatFormer FFN widths: **12.5%, 25%, 50%, and 100%** of the
full intermediate dimension. First train four fresh standalone baselines at
these widths. Then train the five elastic arms at the same new widths. Produce
comparison plots containing all nine new experiments and the four saved
standalone baselines from Feature 013's equally spaced width grid.

The intervention is **the width grid**, not the width-selection distribution.
The new widths are nonuniformly spaced; elastic sampling remains uniform across
the four choices. Do not introduce Feature 014's inverse-membership rule.

### Authorization and workflow boundary

The current user approval covers writing this prompt in `notes/` only. Do not
interpret this document, or historical launch authorizations from Features 013
and 014, as approval to implement or launch Feature 015 now. The user requested
discussion before further work. A later invocation of the specification workflow
authorizes that requested stage; carry out subsequent implementation and GPU
execution only when authorized in the conversation.

The protocol below defines the intended experiment and its acceptance criteria.
Preparing specifications, plans, tasks, or launch commands does not itself
constitute an experiment result. Keep proposed work distinct from verified work.

### Required context and validated width reference

Read the repository's `AGENTS.md`, the Feature 013 specification, plan,
contracts and verification records, and:

- `configs/controlled_exps/tinystories_instruct_optimizer_ownership.yaml`
- `docs/tinystories-optimizer-ownership-experiment.md`
- `specs/014-tinystories-inverse-membership/plan.md` and `verification.md`
- `src/models/granularity.py` and `src/models/ffn.py`
- The ownership, configuration, checkpoint, metrics, and report paths used by
  Features 013 and 014.

The authors' public implementation explicitly defines
`scale_factors = [1/8, 1/4, 1/2, 1]` for S/M/L/XL:
[modified_llama.py](https://github.com/devvrit/matformer/blob/main/modified_llama.py).
The [paper, section 4.1](https://papers.nips.cc/paper_files/paper/2024/file/fe066022bab2a6c6a3c57032a1623c70-Paper-Conference.pdf)
expresses these as FFN-to-model-dimension ratios 0.5, 1, 2, and 4. These describe
the FFN intermediate dimension, not fractions of total model parameters.

| Grid | Ordered width labels | FFN fractions | Dimensions at full FFN 256 |
| --- | --- | --- | --- |
| Feature 013 historical standalones | g250, g500, g750, g1000 | .25, .50, .75, 1.00 | 64, 128, 192, 256 |
| Feature 015 new runs | g125, g250, g500, g1000 | .125, .25, .50, 1.00 | 32, 64, 128, 256 |

Use explicit fractions and dimensions in campaign metadata. Do not relabel
historical g750 results as a new width or confuse ordered width position with
physical dimension.

### Nine fresh runs and required execution order

All new runs use seed **42**, normal fresh model construction, and independent
run identities. Never initialize a new run from a trained historical or
diagnostic checkpoint.

**Stage 1: four fresh standalone baselines.** Train independent dense models
with FFN dimensions 32, 64, 128, and 256, each for one designated training epoch.
The new 64/128/256 runs are required even though matching historical standalone
sizes exist. Do not replace them with reused results, copies, or relabeled
historical runs.

**Stage 2: five fresh elastic runs.** Begin full-budget elastic production only
after all four new standalones finish and their terminal artifacts pass
validation. Short pre-production diagnostics may exercise all arm types.

| Arm | FFN representation | AdamW history ownership | L2 gradient clipping |
| --- | --- | --- | --- |
| S1 | Slicing | Shared | Global cap 1.0 |
| S2 | Slicing | Per width (`per_granularity`) | Global cap 1.0 |
| C1 | Concat | Shared | Global cap 1.0 |
| C2 | Concat | Per width (`per_granularity`) | Global cap 1.0 |
| C3 | Concat | Per incremental FFN block plus common owner (`per_ffn_block`) | Cap 1.0 independently per active owner |

Each elastic run trains for four complete designated epochs. Give the campaign
and all nine runs fresh, unambiguous identities while retaining S1/S2/C1/C2/C3
as the recognizable arm names in reports. Multiple optimizer histories operate
on shared model weights; they do not create independent width-specific models.

### Uniform sampling and unequal incremental blocks

Use `granularity_sampling_mode=global`, independent uniform draws with
replacement, and interval **H=1**. Select one width globally across all layers
per complete optimizer update, with one forward/backward and one loss:

```text
ordered widths = [g125, g250, g500, g1000]
probabilities  = [0.25, 0.25, 0.25, 0.25]
```

All five elastic arms must use the same isolated action RNG initialization and
exact selected-width sequence. Keep data ordering independent of action RNG.
Do not force balanced counts, hold widths, use adaptive selection, or apply
inverse-probability loss weighting. Set `correction_mode=none` throughout;
GMC, LMC, and inverse-membership sampling are outside this feature.

The new concatenated incremental blocks are:

| Block | FFN coordinate interval | Block dimension | Active widths |
| --- | --- | ---: | --- |
| A | [0, 32) | 32 | g125, g250, g500, g1000 |
| B | [32, 64) | 32 | g250, g500, g1000 |
| C | [64, 128) | 64 | g500, g1000 |
| D | [128, 256) | 128 | g1000 |

These are four unequal blocks, not four equal quarters. Preserve one C3 owner
per incremental block and one common owner; do not split the larger increments
into additional independently clipped owners. The five owner caps remain 1.0,
without scaling by block size and without a second global clipping pass.

Generalize the current equal-quarter topology restrictions and any associated
shape/count assumptions only as needed for this explicit new campaign. Derive
block support, optimizer allocations, exposure, and checkpoint identities from
the actual boundaries. Verify slicing and concat realize the same active FFN
dimensions at every width. Keep old campaign topology and validation intact.

Under uniform sampling, expected block activation probabilities remain
1.00/.75/.50/.25; common parameters participate every update. Record actual
counts separately from expectations. Preserve slicing's full-tensor AdamW
momentum/decay behavior and concat's absent-gradient behavior for inactive
blocks. C2 retains lazy histories with block membership counts 4/3/2/1 and four
common histories after every width has been selected. Preserve one global
scheduler advance per complete update and the existing C3 partial-failure
boundary.

### Inherited scientific controls and budgets

Verify controls against the original Feature 013 resolved configurations and
audited data; change only the declared width grid and its necessary topology,
identity, count, and reporting metadata:

- d_model 64, four layers, four attention heads, context 128, vocabulary 2,048,
  full elastic FFN 256, initializer standard deviation 0.02.
- AdamW LR 0.008, betas (0.9, 0.95), epsilon 1e-8, weight decay 0.1; cosine
  schedule with 64 warmup steps and each run's full assigned horizon.
- Batch 64, gradient accumulation 1, bf16, one process on one GPU; no LR scaling,
  pre-nested width warmup, or membership correction.
- Same tokenizer, audited four-role corpus, designated training membership,
  fixed excluded tail, packing, and deterministic epoch-order policy.
- One designated epoch: 5,576,448 packed sequences, **87,132 updates** and
  **713,785,344 training tokens**; preserve the excluded 43-sequence tail.
- Each standalone: one epoch at that exact budget. Each elastic: four epochs,
  **348,528 updates** and **2,855,141,376 training tokens**.
- All nine new runs share the first-epoch batch sequence. All five elastic
  runs share all four epoch sequences, with continuous optimizer, scheduler,
  data-cursor, and RNG state across boundaries.
- Ordinary validation every 64 updates and at completion, with inherited
  evaluation membership and target-token-weighted causal loss. Keep final
  holdout sealed and controller data out of training and sampling decisions.

Expected active non-embedding counts for new dimensions 32/64/128/256 are
**90,688 / 115,264 / 164,416 / 262,720** under the existing count convention,
excluding embeddings and LM head. Confirm these on actual models in preflight,
including each dense standalone. Retain the historical dimension-192 point at
**213,568** parameters. Count checks must detect discrepancies rather than
silently substituting nominal FFN fractions for parameter counts.

Matching seeds do not establish identical initial tensors across model sizes
or representations. Equal aggregate training tokens do not establish equal
compute, runtime, or direct exposure of individual widths.

### Historical standalone references

Reuse only the four original uncorrected standalone terminals as historical
comparison inputs. The known Feature 013 frozen reference is:

```text
/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/reports/complete-20260910T181951Z/frozen/frozen_manifest.json
```

Revalidate saved identities, hashes, actual FFN dimensions, parameter-count
convention, one-epoch budgets, and evaluation protocol before using these
endpoints. Preserve historical files. Missing or incompatible references must
be reported explicitly, not silently omitted or replaced.

The requested combined plots do not include historical S1/S2/C1/C2/C3 curves,
Feature 013 GMC/LMC runs, or Feature 014 inverse-membership runs. Historical
standalones are an additional reference group, not substitutes for fresh ones.

### Verification and execution requirements

Reuse the existing trainer, ownership collections, checkpointing, metrics,
preflight, terminal readers, and plotting machinery. Introduce a distinct
campaign contract so the new width grid cannot be confused with schemas and
checkpoints from earlier campaigns. Avoid a second trainer or optimizer registry.

Before production, verify:

- Exact nine-run matrix, ordered widths, uniform sampling, controls, budgets,
  and actual model counts; reject incompatible or occupied identities.
- Real unequal-block model/update semantics for all five elastic arms,
  including wider-then-narrower updates, inactive gradients, lazy histories,
  owner coverage, independent clipping, and global scheduler accounting.
- Exact action/batch resume and numerical state agreement within existing
  tolerances for all new arm types, including epoch boundaries. Reject wrong
  width grids, block boundaries, ownership, probabilities, and malformed state
  before live mutation. Partial C3 updates must not overwrite durable state.
- Real clipping sidecars for new C1/C3 identities, their metrics/summary paths,
  committed observation counts, and terminal-reader acceptance. Explicitly
  guard against the historical corrected-arm clipping-log omission.
- New report fixtures with 24 endpoints and combined fixtures with 28; reject
  missing runs, duplicate identities, wrong widths/counts, stale hashes,
  mismatched evaluation roles, and premature endpoints.
- Restart-safe scheduling, one writer per run, and the standalone-to-elastic
  production barrier. Resume from each run's own valid checkpoint and retain
  failed/replayed work in resource accounting.
- Historical Feature 013 and 014 configuration, topology, sampling, checkpoint,
  and report compatibility. Existing scientific contracts must remain strict.

Use the pinned Python environment at
`/home/ivo.navarrete/.conda/envs/elasticnn/bin/python`. Retain short real-shape
bf16 GPU diagnostics, including save/restore and clipping artifact checks, with
passed CPU/GPU evidence bound to source/config hashes before production.

When execution is authorized, use a fresh campaign root, separate from both
prior campaigns, for snapshots, configs, diagnostics, launch records, logs,
checkpoints, and reports. Submit every GPU workload through `sbatch`, exclude
`gpu-[05,50,51]`, and observe the existing user-wide ceiling of two running and
four submitted jobs or stricter live limits. Preserve unrelated jobs/helpers.
Record job IDs, attempt costs, checkpoints, throughput, and peak resources.

### Required reports and comparison plots

Use terminal checkpoints after the complete assigned budgets. Freeze checkpoint
and ordinary-validation identities before reporting. Perplexity is exp of the
aggregated causal validation loss; do not use best checkpoints, early elastic
endpoints, or trailing averages as terminal results.

Deliver CSV/JSON tables for:

1. **24 new endpoints:** four fresh standalones plus four widths for each of the
   five new elastic arms.
2. **28 combined endpoints:** all 24 new endpoints plus the four historical
   standalone results at dimensions 64/128/192/256.

Key rows by campaign/run identity and physical width, not just dimension. The
old and new standalone results at dimensions 64/128/256 are distinct records
and must not be deduplicated. Include group, arm, seed, width fraction, FFN
dimension, exact count, loss/perplexity, actual/assigned budgets, evaluation and
checkpoint provenance, exposure, clipping, and resource fields as applicable.

Generate combined **loss versus active non-embedding parameters** and
**perplexity versus active non-embedding parameters**, each in PNG and PDF:

- Five connected, consistently colored elastic curves for the new S1/S2/C1/C2/C3
  results, each with four endpoints at dimensions 32/64/128/256.
- Four disconnected historical standalone markers at 64/128/192/256, and four
  disconnected fresh standalone markers at 32/64/128/256.
- Distinguishable marker styles and clear legend entries for `Standalone —
  historical grid` and `Standalone — MatFormer grid`. Preserve both records
  and make coincident markers identifiable, for example with filled/open
  markers; do not jitter the exact parameter-count coordinates or invent gaps.
- Explicit seed, dataset, ordinary-validation role, count convention, and
  one-epoch standalone versus four-epoch elastic budget annotations.

Retain standard per-run training/validation and resource diagnostics. Interpret
the new elastic curves against the fresh standalones, with historical
standalones providing context and repeat measurements at the three shared
sizes. Keep seed-42 conclusions descriptive, with no across-seed error bars or
significance claims. Distinguish representation/history effects from C3's
independent clipping and the altered block sizes.

New-campaign reporting may complete independently with 24 validated endpoints
if historical references are unavailable, but the requested combined comparison
is complete only with all 28 valid endpoints. Record outstanding work honestly.
Update the feature's tasks, verification record, and runbook from actual saved
evidence so production/report completion does not remain stale in documentation.
