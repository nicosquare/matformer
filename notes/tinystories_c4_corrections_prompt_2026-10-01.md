# Fresh-chat prompt: TinyStories C4 separate GMC and LMC experiments

Work in `/home/ivo.navarrete/ElasticNN/matformer`. Read `AGENTS.md`, `notes/tinystories_c4_selected_block_prompt_2026-09-26.md`, the relevant TinyStories optimizer-ownership implementation, and the completed C4 campaign artifacts before changing anything.

Implement and run **exactly four new C4 elastic experiments** to evaluate gradient membership correction (GMC) and learning rate membership correction (LMC) **separately**, on both the linear and geometric width grids:

| New arm | Grid | Gradient correction | Learning rate correction |
| --- | --- | --- | --- |
| C4-linear-GMC | Linear | Enabled | Disabled |
| C4-linear-LMC-only | Linear | Disabled | Enabled |
| C4-geometric-GMC | Geometric | Enabled | Disabled |
| C4-geometric-LMC-only | Geometric | Disabled | Enabled |

The purpose is to determine whether either isolated correction improves the completed uncorrected C4 results and reaches the **matching standalone ordinary-validation result at each width**, especially `g1000`. Use the saved uncorrected C4 runs and standalones as read-only references. Do not retrain them or add C1–C3 or S1 curves to these comparisons.

## C4 update rule

Preserve the completed C4 selected-block protocol. Each update samples exactly one global width. The forward pass uses all FFN blocks in that width's prefix, but the optimizer updates **only the newly introduced block for that width plus the common parameters**:

| Width position | FFN blocks used in forward pass | FFN block updated | Other parameters updated |
| --- | --- | --- | --- |
| First | A | A | Common |
| Second | A + B | B | Common |
| Third | A + B + C | C | Common |
| Fourth | A + B + C + D | D | Common |

The linear grid is `g250/g500/g750/g1000`, with A/B/C/D each adding 64 FFN units. The geometric grid is `g125/g250/g500/g1000`, with additions of 32/32/64/128 units. The full FFN dimension is 256. Retain the existing disjoint ownership partition, including embeddings/head, attention, norms, and any common FFN bias in the common owner.

Do not detach earlier blocks from the graph. Gradients on earlier active blocks may be computed, but those blocks must receive **no AdamW step, weight decay, momentum/counter advance, or parameter change**. Clear their gradients before clipping or explicitly exclude them. Step the selected block owner and common owner exactly once per committed update. Keep one AdamW history per FFN block and one for common parameters.

Advance the **single global cosine scheduler once** after the complete logical update. C4 owner calls must equal the corresponding width-selection counts; common owner calls must equal total committed updates. Preserve and validate owner mapping, correction semantics and factors, per-owner state/counters, width-selection counts, scheduler, RNG, and data cursor across checkpoint/resume.

## Separate correction semantics

Reuse the existing configured membership factors on both grids:

| Block | Configured forward membership count | Correction factor |
| --- | ---: | ---: |
| A | 4 | 1 |
| B | 3 | 4/3 |
| C | 2 | 2 |
| D | 1 | 4 |

These factors depend on configured forward membership, not the physical block dimension or the number of widths sampled in the current update. Do not replace them with a uniform factor of four. Under C4, all four blocks already have equal expected optimizer selection frequency. Describe these runs as tests of the existing membership factors applied to C4, not compensation for unequal C4 block update counts.

**GMC-only:** Apply the existing block gradient multiplication during backward, before clipping. Only the selected block's corrected gradients and common gradients participate in clipping and stepping. Common gradients are not membership-scaled. All owners retain the same nominal scheduler LR. Do not multiply the resulting parameter change after AdamW or apply block LR multipliers.

**LMC-only:** Leave gradients unchanged by membership correction. Apply the selected FFN block's factor **only to its learning rate for the AdamW step**. Common parameters use the nominal global LR. For nominal scheduled LR `eta`, selected block B uses `eta * 4/3`, C uses `eta * 2`, and D uses `eta * 4`; A uses `eta`. Scale the LR for the whole selected block's AdamW update, including the decoupled weight-decay term. Do not scale gradients, AdamW moments, or counters. Do not additionally multiply the completed parameter change, which would apply the correction twice. Use the existing LMC factors and scientific intent, while making this isolated LR behavior explicit.

The current legacy `correction_mode: lmc` also enables GMC hooks through `membership_correction`, and scales the parameter change after AdamW. **Do not simply enable that legacy mode for these new LMC-only arms.** Introduce explicit, versioned correction semantics for the new campaign so LMC-only has no gradient correction and records nominal and effective applied LRs. Adapt scheduler synchronization and checkpoint validation to distinguish the global nominal clock from the selected block's effective step LR. Avoid compounding multipliers or leaving a temporary effective LR in the next update's nominal scheduler state.

Preserve historical configs, source snapshots, artifacts, and restore behavior. Existing completed `C1-LMC`, `C2-LMC`, and `C3-LMC` runs are combined GMC-plus-LMC results; do not relabel them as LMC-only or use them as isolated LMC evidence. The completed uncorrected C4 runs are unaffected. Document the new semantics without rewriting historical experiment identities or signatures.

## Gradient clipping

Retain **one global L2 norm cap of 1.0** over the selected FFN block plus common parameters jointly. Compute and apply one coefficient before either optimizer step. Earlier active blocks must not enter the norm. Do not use separate per-owner caps.

For GMC-only, clipping sees the membership-scaled selected block gradients and unchanged common gradients. For LMC-only, clipping sees raw selected block and common gradients; the LR multiplier is applied only at the optimizer step. Make this order explicit in resolved configs and saved evidence.

## Training controls and references

- Uniform independent width draws with replacement, probability `0.25` per width, one width per global update. Preserve the original deterministic batch order and dedicated action/data RNG streams. No coverage balancing, inverse-membership sampling, or epoch-varying probabilities.
- Seed 42, fresh normal model initialization for all four arms, TinyStories-Instruct prepared corpus, `d_model=64`, four layers, four heads, context length 128, vocab 2048, BF16, batch size 64, one process/GPU. Do not warm-start from completed C4 checkpoints.
- Four epochs, `87,132` global updates per epoch and `348,528` total; `2,855,141,376` tokens per run. Nominal peak LR `0.008`, 64-step warmup, global cosine schedule, AdamW betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`. LMC-only selected-block peak LRs are `0.008`, `0.010666666666666666`, `0.016`, and `0.032`; common peak LR remains `0.008`. Do not retune these controls in this comparison.
- Use `configs/controlled_exps/tinystories_instruct_c4_selected_block_linear.yaml` and `configs/controlled_exps/tinystories_instruct_c4_selected_block_geometric.yaml` as starting controls. Preserve ordinary validation, its aggregation and manifests, and the final holdout seal.
- Give the new arms fresh identities and a fresh campaign root, suggested `/nfs-stor/ivo.navarrete/results/elasticnn/tinystories-optimizer-ownership-c4-separate-corrections-v1`, after checking it is free. The existing launcher hardcodes the old C4 identity and output root; adapt it or create a focused launcher for these four arms so it cannot submit into the completed campaign.
- Read uncorrected C4 terminals from `/nfs-stor/ivo.navarrete/results/elasticnn/tinystories-optimizer-ownership-c4-selected-block-v1/runs/tinystories-optimizer-ownership-c4-selected-block-v1-C4-{linear,geometric}-s42/`. Use their `348,528`-update terminal ordinary-validation endpoints, not their best intermediate checkpoints.
- Read standalone terminals from `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/runs/ST-*` for linear and `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1/runs/ST-*` for geometric. Standalones have one epoch and `87,132` updates. Match every reference by grid, width, exact active non-embedding parameter count, validation manifest, aggregation, and evaluation role. Never write to reference runs.

## Readiness and execution

Before production submission, inspect the model gradient hooks, config resolution, training update order, optimizer ownership, clipping, global scheduler, checkpoint/resume, accounting, and reporting paths. Run focused regressions for the new correction isolation and historical behavior. Record immutable source/config identities, resolved controls, and action/data audits under the new campaign root.

Use real concat models to verify every width on both grids. Show that GMC-only applies the intended gradient factor and leaves effective LR nominal, while LMC-only has no gradient multiplication and applies exactly one selected-block LR factor. Verify the joint clipping coefficient, unchanged unselected parameters and AdamW histories, unchanged common correction treatment, selected owner counters, and one global scheduler advance. Compare an LMC-only step against a manual AdamW step using the multiplied LR from identical parameters, gradients, and optimizer state, including weight decay and nonzero moment histories.

Exercise checkpoint/resume around warmup boundaries and representative scheduler positions. Verify resumed updates match uninterrupted updates, correction metadata rejects incompatible restores before mutation, and failures during owner steps or after mutation cannot be recorded as successful complete updates. Check that effective LRs do not accumulate factors across consecutive selections or after resume.

Perform a real-shape CUDA BF16 readiness gate for all four arms at batch 64 and context 128. CPU-only training is not a valid production result. Preserve gate failures and logs; do not submit production until the corrected source/config identities pass readiness.

Once implementation and readiness checks pass, **submit all four new C4 jobs**, with at most two running concurrently and within the existing user-wide submission limits. Follow the existing Slurm partition/QoS and exclusion rule `gpu-[05,50-51,54]`. Confirm each job entered CUDA BF16 and is committing updates. Report job IDs, output directories, measured nominal/effective LRs, and operational issues. Continue/recover only the corresponding new arm from its own durable checkpoint; avoid duplicate submissions or writes to historical campaigns.

## Terminal outputs and interpretation

When all four runs finish, validate their terminal checkpoints, full budgets, owner counters, scheduler positions, correction metadata, and ordinary-validation identities. Produce machine-readable endpoint and difference tables, source/config/checkpoint hashes, and PNG/PDF loss and perplexity figures under the new campaign root.

Publish **32 grid-qualified endpoint rows**: 16 new corrected C4 endpoints, eight saved uncorrected C4 endpoints, and eight matching standalone endpoints. Publish 16 corrected-minus-uncorrected C4 comparisons and 16 corrected-minus-standalone comparisons for both loss and perplexity. Report each width's gaps, especially `g1000`, and whether either correction reaches or improves its standalone terminal result.

For each grid, draw GMC-only, LMC-only, and uncorrected C4 width endpoints as separately labeled connected curves. Draw standalones as **separate scatter points with no connecting line**. In validation-loss-versus-update plots, use recorded C4 validation trajectories; each standalone contributes only its single terminal point at `87,132` updates. Never draw a standalone horizontal line or implied trajectory. Label LMC curves explicitly as **LMC only, no GMC**.

Distinguish equal expected C4 FFN-block update counts from unequal forward/example coverage across widths, and distinguish nominal global LR from LMC effective block LR. These are descriptive seed-42 results, not evidence of multi-seed robustness. If an arm fails or lacks valid terminal evidence, report it as incomplete rather than substituting an intermediate checkpoint or claiming the four-arm comparison is complete.
