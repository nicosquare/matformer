# Configuration and Campaign Contract

## Inputs and fixed matrix

Add `configs/controlled_exps/tinystories_instruct_optimizer_ownership.yaml` with
`schema_version: 1`, `campaign_id`, `common` (ordinary trainer sections), `arms`
(exactly nine explicit override mappings), and `expected_data` (pinned audit).
The campaign entrypoint applies common controls then the named arm's overrides,
materializes ordinary YAML and uses `resolve_run_config`. It is a fixed campaign
expansion, not a new trainer config format or general sweep engine.

| Arm | Family/variant | Scope | Clip mode | Physical FFN | Epochs/steps |
| --- | --- | --- | --- | ---: | --- |
| ST-g250 | standalone/dense | shared | global | 64 | 1 / 87,132 |
| ST-g500 | standalone/dense | shared | global | 128 | 1 / 87,132 |
| ST-g750 | standalone/dense | shared | global | 192 | 1 / 87,132 |
| ST-g1000 | standalone/dense | shared | global | 256 | 1 / 87,132 |
| S1 | nested/slicing | shared | global | 256 | 4 / 348,528 |
| S2 | nested/slicing | per_granularity | global | 256 | 4 / 348,528 |
| C1 | nested/concat | shared | global | 256 | 4 / 348,528 |
| C2 | nested/concat | per_granularity | global | 256 | 4 / 348,528 |
| C3 | nested/concat | per_ffn_block | per_owner | 256 | 4 / 348,528 |

Dense is the resolved representation; the existing standalone constructor uses
ordinary `LlamaForCausalLM`, not a new `model.variant=dense` value. Use the current
standalone source-width resolution and retain source metadata. Elastic-only
sampling fields belong in elastic arm overrides; config validation rejects them
on standalones.

## Matched controls

- Seed/data seed 42; normal fresh initialization, initializer standard deviation
  .02; d_model 64, layers 4, attention heads 4, context 128, vocab 2048.
- Elastic fractions .25/.50/.75/1.00 at g250/g500/g750/g1000. Use
  `run.sampling_mode=nested-random`, `model.granularity_sampling_mode=global`,
  `model.global_sampling_schedule=random_with_replacement`, H=1. Existing uniform
  action selection resolves probabilities [.25,.25,.25,.25]. No forced exposure.
- AdamW LR .008, betas [.9,.95], eps 1e-8, decay .1. Batch 64, accumulation 1,
  bf16, distributed strategy none/world size 1, LR scale rule none.
- Ordinary single causal loss; correction mode none with membership/LMC/GMC off;
  pre-nested width warmup disabled. Scheduler warmup is still 64 updates.
- Cosine over the full assigned horizon. Ordinary validation every 64 updates
  and at terminal completion. Holdout configured/separated but never evaluated
  during training or reporting. Controller role remains reserved, not an adaptive
  action signal in this campaign.
- `dataset.optimizer_iteration.mode=repeat_epochs` and
  `epoch_order=deterministic_per_epoch` for all nine runs.
- Token budgets: 713785344 standalone; 2855141376 elastic. No max-step cap that
  shortens the run. Each update counts 8192 packed training tokens.

## New ownership and clipping inputs

Extend `training.optimizer.state_scope` with `per_ffn_block`; retain
`shared|per_granularity` and `scheduler_clock=global_step`. Resolve the new scope
only for AdamW, nested concat, four equal quarter widths, global random single
actions, single process and disabled pre-nested warmup/corrections. Reject slicing,
standalone, nested-all, per-layer width sampling, unsupported topology and multi-
process configurations before model training. Other historical scopes retain
their existing eligibility matrices, including SGD support outside this campaign.

```yaml
training:
  optimizer:
    name: adamw
    state_scope: per_ffn_block
    scheduler_clock: global_step
  gradient_clipping:
    mode: per_owner
    norm_type: 2
    owner_max_norms:
      O-A: 1.0
      O-B: 1.0
      O-C: 1.0
      O-D: 1.0
      O-common: 1.0
```

Global mode reuses `training.gradient_clip_norm=1.0`. Omitted clipping mapping
means historical global behavior. Reject conflicting thresholds, absent/extra
owner keys, nonfinite/nonpositive caps or per_owner mode without C3 ownership.
The resolved contract records the applied mode, norm, ordered caps, stabilization
and topology. C3 with global clipping is a test-only construction; it is never
accepted as the campaign's C3 arm.

Preserve the historical normalized optimizer mapping. New campaign records carry
an `optimizer_ownership_contract` schema 1 that includes representation, ownership,
clipping, initialization/data, budget, evaluation and count conventions. Do not
add defaults to legacy signature inputs in a way that changes historical hashes.

## Audit, identity and preflight

Pin every identity recorded in [inspection.md](../inspection.md), including corpus,
optimizer-training/order/tokenizer and three evaluation-role manifests. Audit
shards/order and disjoint roles. Confirm 5,576,491 available sequences versus
5,576,448 designated stored-permutation entries, excluded tail 43/5504 tokens,
713,785,344 designated tokens and 87,132 complete updates per epoch. Reject any
identity/alignment difference instead of adopting newly computed budgets.

CPU-construct each model with its normal seed path, inspect dense/quarter shapes,
deduplicated ownership and exact `non_embedding_parameters`. Expected values:
115264/164416/213568/262720 for all matching representations. Count convention
excludes input embeddings and LM head, not all common parameters.

Generate full expected action-sequence and epoch-order digests using isolated
existing RNG/sampler implementations without training. Check first epoch matches
across nine and all four match across five; preserve all runtime trace evidence
for later verification. No probabilistic test requires exact width balance.

Write the full allowed-difference matrix and all resolved controls, not only a
paired-control hash. Campaign-wide controls match across nine; arm-local expected
representation/width/scope/clipping/horizon differences match the matrix exactly.
Each run gets a fresh unique run ID and unoccupied output identity. Resume later
uses that same identity and exact checkpoint contract. Never accept historical
Feature 12 runs or relabel an earlier short-horizon run.

Save code revision, dependency versions, constructor and seed provenance. Record
that initialization is matched by seed, not exact cross-model tensors. The first
preflight materializes artifacts only and is not authorization to train.
