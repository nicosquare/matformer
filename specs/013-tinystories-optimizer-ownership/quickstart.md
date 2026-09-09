# Quickstart: TinyStories Optimizer Ownership Comparison

The preflight, trainer continuation, terminal recovery, freeze and report paths
are implemented. See [verification.md](verification.md) for test results and
environment limitations. Verification uses short models and synthetic reports;
full-budget campaign outcomes remain future work. Full campaign launch and future
uniform holdout evaluation each require a separate researcher request.

## 1. Environment and implementation checks

From the repository root, use the validated environment:

```bash
conda activate elasticnn
python --version
```

Alternatively use `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python` directly.
The repository requires Python >=3.12; system Python 3.10 is not suitable. Reuse
the pinned `requirements.txt` environment rather than upgrading dependencies.

Run the focused and compatibility suites with one CPU thread per process:

```bash
OMP_NUM_THREADS=1 python -m pytest tests/test_optimizer_ownership.py tests/test_optimizer_ownership_resume.py tests/test_optimizer_ownership_campaign.py tests/test_optimizer_ownership_reporting.py -q -rs
OMP_NUM_THREADS=1 python -m pytest tests/test_per_granularity_optimizer.py tests/test_per_granularity_optimizer_resume.py tests/test_config.py tests/test_train_cli.py tests/test_model_size.py tests/test_packed_corpus.py tests/test_training_smoke.py tests/test_artifacts.py tests/test_reporting.py tests/test_reproducibility.py tests/test_distributed.py tests/test_distributed_sampling.py tests/test_global_sampling_windows.py -q -rs
```

The new tests use small real models and synthetic repeat-epoch fixtures, not the
full campaign. Cover wider-then-narrower slicing/concat behavior, C2 lazy state,
C3 tied/bias partitioning and clipping independence, matched-state C1/C3 test-only
global clipping, exact resume before/at/after epoch boundaries, and poisoned-owner
failure durability. Report fixtures check all 24 endpoints and PNG/PDF structure.
The campaign suite retains Feature 12's six-run/three-seed analyzer fixture,
including its trailing-five validation and saved-holdout endpoint policies.
These synthetic files do not open the real sealed holdout. Gloo integration tests
need local loopback sockets. On a machine with one visible native-bf16 CUDA GPU,
run the eight-update runtime and all-nine-arm resume diagnostics:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m pytest \
  tests/test_optimizer_ownership.py::test_real_trainer_orders_owner_calls_then_clock_and_publishes_complete_updates \
  tests/test_optimizer_ownership_resume.py::test_repeating_packed_sampler_exact_batches_actions_rng_and_state \
  -k cuda -q -rs --basetemp=/scratch/ivo.navarrete/tmp/optimizer-ownership-gpu-check
```

The tests assert bf16 LM-head output, finite weights/histories/clipping, owner
order and one clock advance, exact resumed actions/batches/state around epoch
boundaries, and positive measured CUDA allocated/reserved peaks. The runtime case
saves `cuda_bf16_diagnostic.json` and a resource ledger beneath the test directory.
CUDA cases explicitly skip without a compatible device; skips do not establish
GPU behavior. Use a fresh `--basetemp` to retain old evidence: pytest clears a
reused base directory. These tests never edit the full campaign YAML.

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

Use unused preflight and run-root paths: preflight reserves the nine identities
in a sibling reservation JSON without creating run directories. Earlier real
preflight evidence and its full action/epoch digests are linked in
[the runbook](../../docs/tinystories-optimizer-ownership-experiment.md#phase-3-verification--2026-09-09).
Generate a new manifest for the final code revision before an authorized launch.

## 4. Later execution and resume

Full execution requires a separate researcher request after verification and
preflight validation. Config-only inspection is safe before that request:

```bash
python train.py \
  --config /scratch/ivo.navarrete/tmp/optimizer-ownership-campaign/configs/C3.yaml \
  --preflight
```

After an explicit launch request, execute one selected arm using its immutable
manifest paths. For example, C3:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python train.py \
  --config /scratch/ivo.navarrete/tmp/optimizer-ownership-campaign/configs/C3.yaml \
  --output-dir /scratch/ivo.navarrete/tmp/optimizer-ownership-runs/C3
```

For an individual standalone, use `configs/ST-g250.yaml` and the recorded
`optimizer-ownership-runs/ST-g250` output; the other arm names work likewise.
There is no implicit nine-run launcher. Do not extend a historical run, change
the full schedule horizon, or extract standalone initialization.

Use the same run identity/config/output directory for continuation. A terminal
standalone has one complete epoch; an elastic run has four. Verify durable
checkpoint identity and exact action/batch/clock reconciliation. C3 failure after
an owner mutation must abort and recover from the previous durable checkpoint.
Resource summaries retain all continuation attempt costs, including replayed work.

**Resume and completion-only recovery use exactly the same trainer command above.**
The materialized config enables continuation. An interrupted run loads its latest
durable compatible checkpoint and reconciles scientific rows beyond that boundary.
Keep `resource_attempts.json`, failure records and the original config with the
run; their failed/replayed work belongs in cumulative costs. Do not save or reuse
the poisoned in-memory state after an owner, clock or accounting failure.

If the assigned terminal checkpoint exists but `terminal_validation_results.json`
is missing because evaluation or publication failed, the same command restores
that terminal, performs ordinary validation as needed and publishes the sidecar
with **zero further optimizer steps and the same checkpoint SHA256**. Repeating
completion validates and reuses an existing valid sidecar. A malformed or
incompatible checkpoint/sidecar is an error, not a reason to start fresh in the
occupied directory. Restore an intact matching artifact from backup when needed.

Inspect `config.json`, `metrics.csv`, `run_summary.json`, `checkpoints/latest.pt`,
`optimizer_ownership_trace.jsonl`, C1/C3 `optimizer_ownership_clipping.jsonl`,
`resource_attempts.json` and the terminal sidecar before freezing. Ordinary
validation uses 285 packed sequences / 36,195 causal targets from the pinned
128 source documents. Training token counts include all 128 packed tokens.

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
The final holdout remains sealed throughout this workflow. As an alternative to
`--run-root`, repeat `--run-dir` once per supplied run; these options are mutually
exclusive and saved identities determine the arms. Use unused freeze/report
output directories. Reports revalidate frozen source hashes; another continuation
or recovery changes resources/summary and requires a new freeze after completion.

For a separately requested partial diagnostic, add `--allow-partial` to **both**
freeze and report. Missing arms/points are listed and every output is labeled
partial; malformed supplied evidence still fails.

## 6. Read the comparisons and resource limits

| Comparison | Interpretation |
| --- | --- |
| S1 / S2 | Shared versus width-owned AdamW histories within slicing |
| C1 / C2 | Shared versus width-owned histories within concat, retaining lazy absent state |
| S1 / C1 | Representation change under shared histories; inactive-tail momentum/decay and counters also change |
| S2 / C2 | Representation change under width-owned histories; inactive-tail behavior and allocated histories also change |
| C1 / C3 | Global cap 1 versus separate owner caps 1; combined gradient cap grows from sqrt(2) to sqrt(5), with histories shared across activating widths |
| Each elastic width / matching standalone | Equal active non-embedding count; fresh seed-42 construction, one versus four per-run epochs |

All findings are descriptive for seed 42. Matching initialization seeds does not
establish equal tensors across shapes. One elastic run matches the aggregate
tokens of the four standalones, without matching per-run FLOPs, wall time or
realized width exposure. Expected selections/quarter activations are not targets.

Actual optimizer-state bytes use measured tensor dtypes. C2's theoretical 37.5%
FFN-moment saving relative to S2 is not a total-memory saving. Persistent state,
temporary concat layout estimates and CUDA allocated/reserved peaks are separate.
The ledger sums the latest duration per unique attempt and takes maximum peaks,
including failed/replayed work but excluding queue downtime. Unfinalized hard-kill
attempts have incomplete measurements; null device metrics are not zero usage.

Full-budget losses, timings, memory peaks and the six measured scientific
comparisons remain to be collected by the separately requested campaign. See
[verification.md](verification.md) for the evidence that validates the tooling.
