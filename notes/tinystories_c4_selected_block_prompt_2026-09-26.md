# Fresh-chat prompt: TinyStories C4 selected-block experiment

Work in `/home/ivo.navarrete/ElasticNN/matformer`. Read `AGENTS.md` and the relevant existing TinyStories optimizer-ownership implementation and saved campaign artifacts before changing anything.

Implement and run **exactly two new C4 elastic experiments**, one on the linear width grid and one on the geometric width grid. C4 uses the concatenated FFN representation. Its purpose is to see whether either C4 run reaches the **matching standalone ordinary-validation result at each width**. Compare C4 only with standalones in the final figures and tables; do not add C1–C3 curves to this comparison or retrain standalone models.

## C4 update rule

Each update samples exactly one global width. The forward pass uses all FFN blocks in that width's prefix, but the optimizer updates **only the newly introduced block for that width plus the common parameters**:

| Width position | FFN blocks used in forward pass | FFN block updated | Other parameters updated |
| --- | --- | --- | --- |
| First | A | A | Common |
| Second | A + B | B | Common |
| Third | A + B + C | C | Common |
| Fourth | A + B + C + D | D | Common |

For the **linear** grid, the width labels are `g250`, `g500`, `g750`, `g1000`; A/B/C/D each add one quarter of the full FFN width. For the **geometric** grid, they are `g125`, `g250`, `g500`, `g1000`; the successive additions are 12.5%, 12.5%, 25%, and 50%. The common owner includes embeddings/head, attention, norms, and any common FFN bias, following the existing concatenation ownership partition. Forward/backward must preserve the full selected-width computation: do not detach earlier blocks from the graph. Gradients on earlier active blocks may be computed, but those blocks must receive **no AdamW step, weight decay, momentum/counter advance, or parameter change** on this update. Clear their gradients before clipping or otherwise exclude them explicitly.

Use one AdamW history per disjoint FFN block and one for the common parameters, or an equivalent implementation with the same state semantics. On each committed update, step the selected block owner and common owner exactly once. Advance the **single global cosine LR scheduler once** after the complete logical update; all owners use the same current global LR. Checkpoint and resume must preserve and validate the C4 owner mapping, per-owner state/counters, width-selection counts, scheduler, RNG and data cursor. Update the existing accounting assumptions that currently infer C3's prefix-owner activation from `per_ffn_block`; C4's expected FFN owner calls equal the corresponding width-selection counts, while common calls equal total committed updates.

## Gradient clipping

Use **one global L2 norm cap of 1.0**, as in C1/C2 and the standalones, over the gradients of the **selected FFN block plus common parameters jointly**. Compute one coefficient and apply it to those gradients before their optimizer steps. Do not use C3's separate per-owner caps. Gradients of earlier blocks that participate in the forward pass but will not be updated must **not** enter this norm. In particular, a `g1000` update clips D + common together and steps only D + common. Make this rule explicit in the resolved config and saved evidence; the existing `per_ffn_block` configuration currently requires per-owner clipping and must not silently retain that C3 restriction or semantics.

## Training controls and references

- Uniform independent width draws with replacement, probability `0.25` for each width, one width per optimizer update. Keep the original deterministic batch order and dedicated action/data RNG streams. No coverage balancing or epoch-varying probability schedule.
- Seed 42, fresh normal model initialization, TinyStories-Instruct prepared corpus, `d_model=64`, four layers, four heads, context length 128, vocab 2048, bf16, batch size 64, one process/GPU.
- Four epochs, `87,132` global updates per epoch and `348,528` total; unchanged `2,855,141,376` token budget. Peak LR `0.008`, 64-step warmup, global cosine schedule, AdamW betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`. Retain the existing ordinary-validation protocol and final holdout seal.
- Use the existing linear and geometric concatenation configs as the starting controls: `configs/controlled_exps/tinystories_instruct_optimizer_ownership.yaml` and `configs/controlled_exps/tinystories_instruct_matformer_widths.yaml`. Resolve all scientific controls explicitly; give C4 fresh run identities and a fresh output root, suggested `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-c4-selected-block-v1` after checking it is free.
- Read standalone terminals from `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/runs/ST-*` for linear, and `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1/runs/ST-*` for geometric. Standalones have one epoch and `87,132` updates. Match endpoints by width, exact active non-embedding parameter count, validation manifest, and evaluation role. Never write to reference runs.

## Execution and outputs

Before production submission, inspect the relevant training, optimizer, clipping, checkpoint, metrics, and reporting paths. Make C4 a distinct, resumable protocol without changing historical C1–C3 behavior or signatures. Check the resolved configs, exact owner/gradient/clipping/update semantics, complete-update failure behavior, and schedule/accounting at representative widths and resume boundaries. Record the source/config identity and action/data audit in the campaign output. Do not treat CPU-only training as a valid production result.

Once the implementation and readiness checks pass, **submit the two C4 jobs** to CUDA GPUs, at most two concurrently. Follow the existing Slurm partition/QoS and exclusion rule `gpu-[05,50-51,54]`; confirm each job entered CUDA and is committing updates. Report job IDs, output directories, and any operational issue. Continue/recover only the corresponding C4 run if interrupted; avoid duplicate submissions.

When both runs finish, validate terminal checkpoints and produce PNG/PDF ordinary-validation loss and perplexity comparisons, plus machine-readable endpoint and difference tables and source hashes, under the new campaign root. Draw the C4 width endpoints as a connected curve and the standalone endpoints as **separate scatter points with no connecting line**. In any validation-loss-versus-update plot, each standalone contributes only its single terminal point at `87,132` updates; never draw a standalone horizontal line or implied trajectory. Report each C4 width's gap to its matching standalone, especially `g1000`, and clearly distinguish equal expected FFN-block update counts from equal training-example coverage. These are descriptive seed-42 results, not evidence of multi-seed robustness.
