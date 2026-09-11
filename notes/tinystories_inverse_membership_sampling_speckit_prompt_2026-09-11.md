# Spec Kit prompt: TinyStories inverse-membership sampling comparison

Date: 2026-09-11

Use this as the feature description for a **new Spec Kit feature** following
Feature 013. Determine the next available feature number from the repository.
Keep Feature 013's specifications and completed campaign identities intact.

Suggested invocation:

```text
/speckit-specify Use notes/tinystories_inverse_membership_sampling_speckit_prompt_2026-09-11.md as the feature description. Preserve the five-arm matrix, fixed probabilities, inherited controls, and execution authorization. The feature request begins at "Feature request".
```

## Feature request

Evaluate S1, S2, C1, C2, and C3 on TinyStories-Instruct using fixed nonuniform
global width sampling with inverse granularity-membership weighting. Compare
their learning, width/block exposure, optimizer behavior, and resource costs
against the completed uncorrected uniform-sampling counterparts from Feature 013.

The primary intervention is the sampling distribution. Inherit the original
uncorrected arm definitions and set `correction_mode=none` throughout. GMC/LMC
combinations, additional seeds, adaptive policies, and hold-interval sweeps are
outside this five-run campaign.

### Workflow and authorization

Create the new specification, clarify material unresolved scientific choices,
produce its plan/contracts and tasks, analyze their consistency, then implement,
test, run short GPU diagnostics, launch the full campaign, monitor it, and report
the completed results. Apply the relevant Spec Kit skills at each stage.

**The user explicitly authorizes submitting sbatch files for this experiment.**
This includes short GPU diagnostics and the five full-budget production runs
after validation passes. Carry this authorization through subsequent workflow
stages; no separate launch confirmation is needed. The prompt-generation step
itself only writes this document. Resolve routine implementation choices using
the repository and this protocol.

Inspect Git status and preserve unrelated work. Reuse the trainer, sampler,
ownership machinery, checkpointing, and reporting primitives. Give this campaign
fresh identities and directories, preserving historical artifacts and validation
rules. If historical documentation conflicts with saved evidence, inspect the
actual configurations, manifests, and terminal artifacts before drawing conclusions.

### Required context

Read:

- `AGENTS.md`
- `specs/013-tinystories-optimizer-ownership/plan.md`, `spec.md`, and contracts
- `configs/controlled_exps/tinystories_instruct_optimizer_ownership.yaml`
- `docs/tinystories-optimizer-ownership-experiment.md`
- `notes/tinystories_optimizer_ownership_results_analysis_2026-09-10.md`
- The fixed-global inverse-membership examples in `docs/100m-slicing-experiments.md`
- `docs/tinystories-portfolio-catchup.md`

Inspect the current fixed-global sampling and C3 configuration/stepping paths;
do not assume their combination is already supported by campaign validation.

### Five fresh runs

Use seed **42** and four complete designated training epochs for every run.

| Arm | FFN representation | AdamW state ownership | L2 gradient clipping |
| --- | --- | --- | --- |
| S1-IM | Slicing | Shared | Global cap 1.0 |
| S2-IM | Slicing | Per width (`per_granularity`) | Global cap 1.0 |
| C1-IM | Concat | Shared | Global cap 1.0 |
| C2-IM | Concat | Per width (`per_granularity`) | Global cap 1.0 |
| C3-IM | Concat | Per FFN block plus common owner (`per_ffn_block`) | Cap 1.0 independently per active owner |

IM denotes the fixed inverse-membership sampling policy. All arms share model
weights across their widths; per-width ownership means separate optimizer
histories, not independent models. C3 retains disjoint owners O-A/O-B/O-C/O-D
and O-common, clips and steps only active owners, and advances one global
scheduler once per complete update. No second global clipping follows its
per-owner clipping.

Preserve slicing's full-tensor zero-gradient/momentum/decay semantics and
concat's absent-gradient behavior for inactive blocks. Preserve C2 lazy
allocation and exact resume. Start every new run from its normal fresh seed-42
initialization, not a trained baseline checkpoint. Matching seeds do not imply
identical initial tensor values across representations.

### Exact sampling contract

The ordered widths are `g250`, `g500`, `g750`, and `g1000`, corresponding to
FFN fractions 0.25, 0.50, 0.75, and 1.00. Their newly introduced quarter blocks
have membership counts **[4, 3, 2, 1]** in this configured width set.

Define the fixed width probabilities by

```text
m = [4, 3, 2, 1]
p_i = (1 / m_i) / sum_j(1 / m_j)
p = [3/25, 4/25, 6/25, 12/25] = [0.12, 0.16, 0.24, 0.48]
```

This is inverse membership of the incremental blocks associated with the width
labels; it is not inverse width fraction or inverse parameter count.

Use the existing `fixed_global` categorical sampler, one independent draw with
replacement per complete update (`H=1`), applied globally across all layers:

```yaml
model:
  granularity_sampling_mode: fixed_global
  global_sampling_distribution:
    g250: 0.12
    g500: 0.16
    g750: 0.24
    g1000: 0.48
```

Resolve the remaining sampler settings against the actual configuration schema.
Use the same isolated action RNG initialization and exactly the same selected
width sequence across all five new arms. Keep data RNG independent. Do not force
counts to match expected proportions, balance cycles, hold widths for multiple
updates, or introduce inverse-probability loss weighting.

The new action sequences must match each other; they are not expected to match
the historical uniform action sequences. Their batch sequences must match each
other and the original deterministic data-order contract.

Report both actual width selections and actual block activations. Expected
quarter activation probabilities are the tail sums of p:

```text
                 A      B      C      D
Fixed IM       1.00   0.88   0.72   0.48
Uniform        1.00   0.75   0.50   0.25
```

O-common participates in every update. These are exposure expectations, not
guaranteed counts or claims of equal parameter exposure. In slicing, an inactive
tail can still change through full-tensor AdamW behavior; distinguish gradient
support exposure from actual optimizer updates.

### Inherited controls and budget

Verify these against the original resolved configurations and prepared manifests:

- Model: d_model 64, four layers, four attention heads, context 128, vocabulary
  2,048, full FFN dimension 256, initializer standard deviation 0.02.
- Active FFN dimensions: 64/128/192/256. Exact active non-embedding parameter
  counts: **115,264 / 164,416 / 213,568 / 262,720**, excluding embeddings and LM head.
- AdamW: LR 0.008, betas (0.9, 0.95), epsilon 1e-8, weight decay 0.1; original
  cosine schedule, 64 LR warmup steps, full production schedule horizon.
- Batch 64, accumulation 1, bf16, one process on one GPU, no LR scaling;
  ordinary causal LM loss, no membership correction, no pre-nested width warmup.
- Same audited four-role corpus, tokenizer, stored permutation, fixed excluded
  tail, and deterministic per-epoch ordering as the original campaign.
- Each designated epoch contains 5,576,448 packed sequences, with the same
  excluded 43-sequence tail. Each update accounts for 8,192 packed training tokens.
- **348,528 updates and 2,855,141,376 training tokens per run**, over four epochs.
  Five new runs total **1,742,640 updates and 14,275,706,880 training tokens**.
- Preserve scheduler, data cursor, RNG, and optimizer continuity across epochs.
- Ordinary validation every 64 updates and at completion, using the inherited
  evaluation membership and target-token-weighted loss aggregation.

Re-audit immutable input identities before production. Keep the final holdout
sealed and controller data out of training and sampling decisions.

Reuse the five original uncorrected uniform runs and four original standalone
terminals as historical references after validating their identities, budgets,
and evaluation protocol. No baseline retraining is included. Report any missing
or incompatible reference explicitly; do not fabricate or silently substitute it.

### Implementation and acceptance checks

Introduce a distinct five-arm campaign recipe and explicit sampling contract.
Bind ordered widths, probabilities, draw cadence, RNG state, and policy identity
into configuration, provenance, checkpoints, and reporting. Preserve strict
validation of historical uniform and correction campaigns.

Extend only eligibility and integration paths needed for fixed-global sampling
with S1/S2/C1/C2/C3. Verify actual execution through real model paths, especially
C3; changing a validation gate alone is insufficient.

Before full training, verify:

- Five-arm preflight accepts the exact matrix and rejects changed probabilities,
  missing widths, mismatched budgets, or incompatible scientific controls.
- The production sampler produces identical deterministic traces across all five
  arms. Check categorical mapping against the specified probabilities without
  requiring finite samples to have exact expected frequencies.
- Real optimizer paths retain inactive-block, history, clipping, and scheduler
  semantics under fixed-global selection, including wider-then-narrower updates.
- Interrupted and uninterrupted runs match action/batch traces and numerical
  state within existing tolerances, including around epoch boundaries; sampling
  contract mismatches are rejected before live-state mutation.
- C3 failures cannot publish partially completed owner updates as resumable state.
- New reporting correctly validates terminal provenance, policy labels, and
  endpoint counts while existing campaign tests continue to pass.

Use `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python`. Run focused CPU checks
and short bf16 GPU diagnostics through `sbatch` for all five arms, in separate
diagnostic directories. Confirm finite loss, actual action selection, save/restore,
ownership accounting, and measured steady-state throughput before production.

### Slurm execution and storage

Use a fresh campaign directory, proposed as:

```text
/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/campaigns/inverse-membership-sampling-v1
```

Check for conflicts before using it. Keep all experiment artifacts, source
snapshots, materialized configs, diagnostics, sbatch files, logs, submission
records, checkpoints, and reports beneath the chosen campaign root. Repository
source code, specifications, tests, and documentation stay in the repository.

- Submit every GPU workload through `sbatch`; exclude `gpu-[05,50,51]`.
- Retain the prior operational ceiling of two running and four submitted GPU
  jobs across the user, including other campaigns; inspect live limits and queue
  state and obey stricter applicable limits.
- Queue the fifth run as capacity becomes available. Use restart-safe submission
  tracking and one writer per run; prevent duplicate jobs and avoid restarting
  unrelated campaign helpers.
- Record source revision/snapshot hashes, scientific identities, Slurm IDs,
  continuation attempts, cumulative runtime, and peak resources.
- Resume each interrupted new run only from its own valid checkpoint. Reconcile
  consumed work beyond the last durable checkpoint in resource accounting.
- Monitor progress, checkpoint health, recent throughput, utilization, and
  estimated completion. Continue through terminal validation and reporting.

### Evaluation and deliverables

Evaluate all four widths from each terminal checkpoint after its complete
assigned budget. Use ordinary validation, target-token-weighted causal loss,
and perplexity equal to the exponential of aggregated loss. Freeze terminal
checkpoint and evaluation identities before assembling endpoint comparisons.
Do not substitute best checkpoints, trailing means, or early endpoints.

Deliver:

1. A new feature specification, plan/contracts, tasks, consistency analysis,
   implementation, runbook, and CPU/GPU verification evidence.
2. Five completed runs with validated terminal checkpoints, action/batch traces,
   exposure/clipping summaries, and measured resource accounting.
3. CSV/JSON tables with **20 new endpoints**, and a validated comparison table
   with **44 endpoints**: 20 fixed-IM, 20 original uncorrected uniform, and four
   original standalones. Mark historical references and preserve their provenance.
4. Loss and perplexity versus exact active non-embedding parameter counts as
   PNG/PDF. Include a five-curve new-campaign view and a uniform-versus-IM
   comparison with consistent arm colors and distinct policy line styles.
   Display standalone references as disconnected brown triangles under one
   `Standalone` legend entry.
5. A four-panel ordinary-validation loss-progress figure, one panel per width,
   over the full recorded update range, comparing each arm's fixed-IM and
   uniform runs; export PNG/PDF.
6. Per-arm/per-width fixed-IM minus uniform loss/perplexity deltas, width/block
   exposure comparisons, clipping diagnostics, optimizer-state bytes, peak
   memory, wall time, throughput, and checkpoint sizes.
7. A concise interpretation distinguishing sampling effects within each arm
   from representation/ownership/clipping differences between arms. Equal
   training-token budgets do not imply equal FLOPs or runtime: IM samples wider
   networks more often. Keep seed-42 findings descriptive, with no across-seed
   significance claims or error bars.

Required new-campaign reporting must validate all five runs and 20 endpoints.
The complete historical comparison must also validate its 44 endpoints; if
historical references are unavailable, deliver the validated new-campaign results
and explicitly identify the outstanding comparison rather than labeling it complete.
