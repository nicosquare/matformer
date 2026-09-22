# Spec Kit prompt: TinyStories S1 warmup extension

Date: 2026-09-22

Suggested invocation:

```text
/speckit-specify Use notes/tinystories_s1_warmup_speckit_prompt_2026-09-22.md as the feature description. Extend the existing TinyStories experiment workflow with exactly two S1 runs testing fourfold LR warmup.
```

## Feature request

Extend the existing TinyStories optimizer-ownership experiments, following the
workflow of Features 014 and 015 and reusing their training, validation,
checkpointing, and reporting infrastructure. This is an extension of the
existing comparison, not an independent experiment framework.

Test whether longer learning-rate warmup improves S1 results with two fresh
seed-42 runs:

| Run | FFN dimensions | Existing warmup | New warmup |
| --- | --- | --- | --- |
| Linear S1 | 64, 128, 192, 256 | 64 updates | 256 updates |
| Geometric S1 | 32, 64, 128, 256 | 64 updates | 256 updates |

Both saved S1 configurations use an absolute 64-update LR warmup, peak LR
0.008, and cosine scheduling. Their four-epoch budget did not scale warmup
automatically. Multiply warmup by four while retaining **four epochs,
348,528 total updates, and 2,855,141,376 tokens** per run. Warmup is included
in this budget. Keep cosine decay over the remaining updates to the same
terminal horizon; keep `pre_nested_warmup` disabled.

Change only LR warmup relative to each original uniform-sampling S1 counterpart:
slicing, shared AdamW history, global clipping cap 1.0, uniform global width
sampling with replacement at H=1, no corrections, and all model, data,
optimizer, seed, precision, and evaluation controls remain inherited. Preserve
the respective counterpart's deterministic action and data streams. Start from
fresh initialization with distinct run identities and artifact paths.

Reuse and validate the existing standalone and 64-update-warmup S1 results;
do not retrain them or add other arms. Reference campaigns are:

- Linear: `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1`
- Geometric: `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1`

Read `AGENTS.md`, the relevant Feature 013–015 specifications and verification
records, and the saved reference configurations. Extend campaign validation to
accept the new warmup explicitly while preserving historical contracts.

Verify the resolved LR schedule and resume continuity before production;
inherit the existing GPU execution checks and cluster limits. Completion requires
both full-budget terminal checkpoints and ordinary-validation loss/perplexity
at every width. Produce comparison tables and plots showing existing
standalones, original S1, and longer-warmup S1 for each grid, plus early LR/loss
curves. Report whether the intervention helped without assuming warmup caused
the earlier underperformance.

Current authorization covers this prompt only. A later Spec Kit invocation
authorizes its requested stage; implementation and GPU execution follow the
user's subsequent instructions.
