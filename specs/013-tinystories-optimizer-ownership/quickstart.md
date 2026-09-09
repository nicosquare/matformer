# Quickstart: TinyStories Optimizer Ownership Comparison

This is an implementation and later execution guide. New files/commands named
below are planned interfaces; they are not implemented by `/speckit-plan`.
Planning performed no training or final-holdout model evaluation.

## 1. Environment and implementation checks

From the repository root, use the validated environment:

```bash
conda activate elasticnn
python --version
```

Alternatively use `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python` directly.
The repository requires Python >=3.12; system Python 3.10 is not suitable. Reuse
the pinned `requirements.txt` environment rather than upgrading dependencies.

After implementing the design, run the focused suites:

```bash
python -m pytest tests/test_optimizer_ownership.py tests/test_optimizer_ownership_resume.py tests/test_optimizer_ownership_campaign.py tests/test_optimizer_ownership_reporting.py
python -m pytest tests/test_per_granularity_optimizer.py tests/test_per_granularity_optimizer_resume.py tests/test_config.py tests/test_train_cli.py tests/test_model_size.py tests/test_packed_corpus.py tests/test_training_smoke.py tests/test_artifacts.py tests/test_reporting.py
```

The new tests use small real models and synthetic repeat-epoch fixtures, not the
full campaign. Cover wider-then-narrower slicing/concat behavior, C2 lazy state,
C3 tied/bias partitioning and clipping independence, matched-state C1/C3 test-only
global clipping, exact resume before/at/after epoch boundaries, and poisoned-owner
failure durability. Report fixtures check all 24 endpoints and PNG/PDF structure.
Implementation should also run a short one-GPU bf16 diagnostic under test budgets;
CPU checks do not establish CUDA precision or resource behavior.

## 2. Re-audit prepared inputs

```bash
python scripts/audit_prepared_corpus.py \
  --prepared-corpus-dir /nfs-stor/ivo.navarrete/matformer-corpora/tinystories-instruct-packed-full-v1 \
  --prepared-tokenizer-dir /nfs-stor/ivo.navarrete/matformer-tokenizers/tinystories-instruct-sentencepiece-bpe-2k-v1 \
  --required-vocab-size 2048 \
  --minimum-training-tokens 713785344
```

The historical [audit evidence](inspection.md) passed. This command verifies
integrity; minimum tokens alone do not prove the exact campaign budget. Campaign
preflight must pin every hash, the designated 5,576,448 sequences and excluded 43.

## 3. Materialize and validate all nine runs

```bash
python scripts/analyze_tinystories_optimizer_ownership.py preflight \
  --campaign configs/controlled_exps/tinystories_instruct_optimizer_ownership.yaml \
  --prepared-corpus-dir /nfs-stor/ivo.navarrete/matformer-corpora/tinystories-instruct-packed-full-v1 \
  --tokenizer-dir /nfs-stor/ivo.navarrete/matformer-tokenizers/tinystories-instruct-sentencepiece-bpe-2k-v1 \
  --output-dir /scratch/ivo.navarrete/tmp/optimizer-ownership-campaign \
  --run-output-root /scratch/ivo.navarrete/tmp/optimizer-ownership-runs
```

Inspect `preflight.json`, `campaign_manifest.json` and nine `configs/*.yaml`:

- Four standalone horizons 87132; five elastic horizons 348528. Every run uses
  seed 42, 8192 tokens/update and cosine warmup 64.
- Active counts 115264/164416/213568/262720 match across model representations.
- S1/S2 slicing; C1/C2/C3 concat; C3 five disjoint owners/caps 1.0; all corrections
  and pre-nested warmup disabled.
- All first-epoch digests match, all elastic four-epoch/action digests match;
  action draws use uniform replacement without balancing.
- Fresh run identities, immutable data/role hashes and ordinary-validation-only
  terminal endpoint policy are recorded. No model training has started.

## 4. Later execution and resume

Full execution requires a separate researcher request after implementation and
preflight validation. Use the existing trainer once per selected arm, with the
config/output paths recorded by the manifest; see the
[CLI contract](contracts/cli-entrypoints.md). Do not extend a historical run,
change its schedule horizon, or extract standalone initialization.

Use the same run identity/config/output directory for continuation. A terminal
standalone has one complete epoch; an elastic run has four. Verify durable
checkpoint identity and exact action/batch/clock reconciliation. C3 failure after
an owner mutation must abort and recover from the previous durable checkpoint.
Resource summaries retain all continuation attempt costs, including replayed work.

## 5. Freeze and generate the comparison

After the nine runs complete their assigned budgets:

```bash
python scripts/analyze_tinystories_optimizer_ownership.py freeze \
  --campaign-manifest /scratch/ivo.navarrete/tmp/optimizer-ownership-campaign/campaign_manifest.json \
  --run-root /scratch/ivo.navarrete/tmp/optimizer-ownership-runs \
  --output-dir /scratch/ivo.navarrete/tmp/optimizer-ownership-frozen
python scripts/analyze_tinystories_optimizer_ownership.py report \
  --manifest /scratch/ivo.navarrete/tmp/optimizer-ownership-frozen/frozen_manifest.json \
  --output-dir /scratch/ivo.navarrete/tmp/optimizer-ownership-report
```

Expect `optimizer_ownership_endpoints.csv` and `.json` with 24 identical rows,
`comparison_report.json`, individual trajectory/resource plots, and these two
combined figure stems in both PNG and PDF:

- `optimizer_ownership_perplexity_vs_non_embedding_parameters`
- `optimizer_ownership_loss_vs_non_embedding_parameters`

Both combined figures have five four-point elastic curves plus four disconnected
standalone markers. All endpoints are terminal ordinary validation at exact active
non-embedding counts. Reports must reject incomplete/mixed/stale inputs by default;
`--allow-partial` is only for separately requested, visibly labeled diagnostics.
The final holdout remains sealed throughout this workflow.

## Planning validation record

Read-only source inspection and CPU model construction confirmed reusable FFN
paths and exact parameter counts. Budget arithmetic and document consistency are
checked as planning validation. Runtime ownership, resume, campaign and reporting
tests are implementation deliverables, not completed results of this plan.
