# Repository inspection and corpus audit

**Date**: 2026-09-09  
**Purpose**: Record evidence used to draft the specification; this is not an implementation plan or a completed campaign preflight.

## Existing behavior inspected

| Area | Evidence | Implication for planning |
| --- | --- | --- |
| Prior feature | [Feature 12 plan](../012-per-width-optimizer-state/plan.md) and [recipe](../../configs/controlled_exps/tinystories_instruct_per_width_optimizers.yaml) | Reuse trainer/artifact conventions, but the three-seed balanced-cycle campaign is distinct. |
| Model representations | [FFN models](../../src/models/ffn.py), sliced forward and concatenated active block assembly | Both representations exist. Verify real-model zero-present versus absent-gradient behavior in the new acceptance tests. |
| Dense models | [Model construction](../../src/training/modeling.py) and [baseline matching](../../src/training/baselines.py) | Existing standalone support is relevant. The user selected seed-42 initialization through normal model construction; canonical mapping and exact cross-run initial-value equality are not required. |
| Optimizer ownership | [Optimizer state](../../src/training/optimizer_state.py) and [config](../../src/utils/config.py) | Currently accepted state scopes are `shared` and `per_granularity`. Width optimizers share ordered model parameters. Do not present block-owned C3 as an existing supported scope. |
| Update lifecycle | [Training steps](../../src/training/steps.py) | Current path clears gradients to absent, applies global clipping, steps one selected optimizer, then the scheduler. C3 needs five-owner activation, separate clipping, and failure handling for partial owner updates. |
| Checkpoints | [Checkpointing](../../src/training/checkpointing.py) and [optimizer validation](../../src/training/optimizer_state.py) | Existing scope-aware resume provides a starting point; representation/group/clipping compatibility and lazy state must be checked for the new campaign. |
| Epochs | [Config alignment](../../src/utils/config.py), [data construction](../../src/training/data.py), and [repeating sampler](../../src/training/packed_corpus.py) | The existing repeated-epoch contract fixes an aligned prefix of the stored permutation, then reorders that same set deterministically per epoch. It does not rotate excluded tail examples. |
| Evaluation/counts | [Validation](../../src/evaluation/validation.py) and [model size](../../src/utils/model_size.py) | Loss is weighted by valid causal target count; perplexity is its exponential. Active non-embedding counts exclude input embeddings and LM head. |
| Comparison | [Existing analyzer](../../scripts/analyze_tinystories_per_width_optimizer.py) | Its frozen manifest requires seeds 42/43/44. New nine-run validation must cover the complete controls and intended differences rather than reuse its protocol or trust its paired-control signature alone. |

## Prepared corpus audit

Ran this read-only audit successfully; no model training or holdout model evaluation ran:

```bash
python scripts/audit_prepared_corpus.py \
  --prepared-corpus-dir /nfs-stor/ivo.navarrete/matformer-corpora/tinystories-instruct-packed-full-v1 \
  --prepared-tokenizer-dir /nfs-stor/ivo.navarrete/matformer-tokenizers/tinystories-instruct-sentencepiece-bpe-2k-v1 \
  --required-vocab-size 2048 \
  --minimum-training-tokens 713785344
```

The audit checks shard integrity and stored ordering; the minimum-token check alone does not prove an exact campaign epoch budget. The separate alignment calculation below establishes the relationship to the requested budget.

| Audit item | Result |
| --- | --- |
| Status | passed |
| Corpus | tinystories-instruct-packed-full-v1 |
| Schema | 3 |
| Source | exhausted |
| Verified shards | 89 |
| Verified stored order entries | 5,576,491 |
| Available training sequences | 5,576,491 |
| Available training tokens | 713,790,848 |
| Tokenizer vocabulary | 2,048 |
| Reserved ordinary-validation/controller/final-holdout document counts | 128 / 128 / 512 |
| Reserved pairwise intersections | all zero |
| Optimizer-training source documents | 2,476,404 |

Identity evidence:

```text
corpus_hash=e2eff35bc7078f4f4c4d618638988b3b82e96e7c20e40175ec8d272c5370efc9
optimizer_training_manifest_hash=c06270adebde4c5456517a5bcf266b2e4520587fb928541333009735ee2e60dd
training_order_sha256=d94ed9514a314a404fc420385a4b0f3a317774707ca3a212100aa7d4f35badbc
tokenizer_manifest_hash=93e6bc7a94df82ca043519a71d5178bb645846ecd553f2302244b72c954e3819
tokenizer_model_sha256=6ae42a09bb5dc007267f11bfd7b7b4fefd006c960c5954a4a45ce7b89ace7713
ordinary_validation_manifest_hash=0c1beea552f54941e397d2442de736b1586e0292f6b1271b62d27ad782627856
controller_manifest_hash=69b039d6f6cec565e9efb576080b3e91d652c63290ebdea6f376fd133b6bd987
final_holdout_manifest_hash=e20160ab1d7781b4fe7f24a78ed547fce0ef3b7b1f0cd66eaf2f84dd89113076
```

## Alignment resolution

The existing resolver uses complete optimizer-update alignment. With one process, batch 64, accumulation 1, and context 128:

```text
available sequences                  = 5,576,491
aligned epoch sequences              = floor(5,576,491 / 64) * 64 = 5,576,448
fixed excluded tail                  = 43 sequences = 5,504 tokens
aligned epoch tokens                 = 5,576,448 * 128 = 713,785,344
updates per aligned epoch            = 5,576,448 / 64 = 87,132
four aligned epochs                  = 2,855,141,376 tokens = 348,528 updates
nine-run aggregate (24 epoch budgets) = 17,130,848,256 tokens
```

The designated epoch set is the prefix of the immutable stored permutation, not a raw sequential prefix of corpus shards. Every later epoch reorders precisely those same 5,576,448 entries. This explains the available-token difference without revising the requested budget or silently omitting examples from the designated epoch. The specification explicitly discloses the inherited exclusion and rejects any changed identity/alignment in future preflight.

Model configuration inspection also establishes `DEFAULT_FFN_MULTIPLIER = 4`; dimension 64 therefore resolves full intermediate size 256, with active intermediate sizes 64/128/192/256. Actual active parameter counts still require model-based preflight; no total-size scaling estimate is substituted for them.

## Remaining boundaries

- All protocol decisions are settled: seed 42 is sufficient without exact weight matching across campaign representations/shapes; C3 uses threshold 1.0 per active ownership group; all 24 combined endpoints use ordinary validation, keeping final holdout sealed.
- The read-only corpus audit passed. No claim is made that C3, the nine-run configurations, or the new comparison workflow already pass runtime tests.
- Feature preparation does not launch training or explicitly evaluate final holdout models.
