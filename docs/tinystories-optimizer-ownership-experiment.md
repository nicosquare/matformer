# TinyStories optimizer ownership experiment

## Implementation status and scope

Phase 1 establishes the experiment record and fixed definitions in
[`src/evaluation/optimizer_ownership.py`](../src/evaluation/optimizer_ownership.py).
Campaign preflight, C3 runtime support, exact campaign resume, terminal sidecars,
and comparison commands are later implementation tasks. This record does not
claim that those interfaces work or that campaign results exist.

The authoritative protocol is the [specification](../specs/013-tinystories-optimizer-ownership/spec.md),
with the [plan](../specs/013-tinystories-optimizer-ownership/plan.md),
[data model](../specs/013-tinystories-optimizer-ownership/data-model.md), and
[task list](../specs/013-tinystories-optimizer-ownership/tasks.md).
Historical Feature 12 configurations, artifacts, and analyzer behavior remain
independent of this new seed-42 campaign.

## Fixed nine-arm protocol

| Arm | Representation | AdamW state scope | L2 clipping | Physical FFN | Epochs | Updates | Training tokens |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| ST-g250 | dense | shared | global 1.0 | 64 | 1 | 87,132 | 713,785,344 |
| ST-g500 | dense | shared | global 1.0 | 128 | 1 | 87,132 | 713,785,344 |
| ST-g750 | dense | shared | global 1.0 | 192 | 1 | 87,132 | 713,785,344 |
| ST-g1000 | dense | shared | global 1.0 | 256 | 1 | 87,132 | 713,785,344 |
| S1 | slicing | shared | global 1.0 | 256 | 4 | 348,528 | 2,855,141,376 |
| S2 | slicing | per_granularity | global 1.0 | 256 | 4 | 348,528 | 2,855,141,376 |
| C1 | concat | shared | global 1.0 | 256 | 4 | 348,528 | 2,855,141,376 |
| C2 | concat | per_granularity | global 1.0 | 256 | 4 | 348,528 | 2,855,141,376 |
| C3 | concat | per_ffn_block | 1.0 per active owner | 256 | 4 | 348,528 | 2,855,141,376 |

All runs initialize freshly through the normal constructor with seed 42 and
initializer standard deviation 0.02. Matching seeds do not imply equal initial
tensors across representations or shapes. Dense is a resolved representation,
not a new trainer `model.variant`; standalones keep their source width label and
use local active fraction 1.0.

| Source width | Fraction | Active FFN dimension | Active non-embedding parameters | Active quarters |
| --- | ---: | ---: | ---: | --- |
| g250 | 0.25 | 64 | 115,264 | A |
| g500 | 0.50 | 128 | 164,416 | A, B |
| g750 | 0.75 | 192 | 213,568 | A, B, C |
| g1000 | 1.00 | 256 | 262,720 | A, B, C, D |

Counts exclude input embeddings and the LM head and match the corresponding
dense standalone. CPU model construction must verify these exact counts during
preflight using the existing model-size helper.

Common controls: d_model 64, four layers, four attention heads, context 128,
vocabulary 2048; AdamW learning rate 0.008, betas (0.9, 0.95), epsilon 1e-8,
weight decay 0.1; batch 64, accumulation 1, bf16, one process on one GPU, no LR
scaling. Use the ordinary causal loss with corrections and pre-nested width
warmup disabled. Cosine scheduling has 64 warmup updates and each run's full
assigned horizon. The global scheduler advances once per complete update.

Elastic arms draw one global width per update uniformly with replacement
(`nested-random`, global sampling, `random_with_replacement`, H=1). Their action
streams must match exactly; counts are measured, never forced to balance.
C3 owns disjoint O-A/O-B/O-C/O-D/O-common parameter sets and steps the active
quarter prefix followed by common. Separate caps have a combined gradient bound
from sqrt(2) through sqrt(5); no second global rescaling is applied.

## Environment and immutable inputs

Use `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python` or activate `elasticnn`.
The repository requires Python >=3.12; system Python 3.10 is unsuitable.
Reuse [`requirements.txt`](../requirements.txt): PyTorch 2.11.0 CUDA 12.8,
Transformers 5.8.0, datasets 4.8.5, PyYAML 6.0.3, NumPy 2.4.3, pandas 3.0.2,
Matplotlib 3.10.9, and pytest 9.0.3. No new runtime dependency is needed.

Prepared corpus:
`/nfs-stor/ivo.navarrete/matformer-corpora/tinystories-instruct-packed-full-v1`.
Tokenizer:
`/nfs-stor/ivo.navarrete/matformer-tokenizers/tinystories-instruct-sentencepiece-bpe-2k-v1`.
Pin every corpus, tokenizer, order, and role hash from the
[inspection record](../specs/013-tinystories-optimizer-ownership/inspection.md).
Its previous successful audit is historical evidence; campaign preflight must
repeat the read-only audit and reject any identity or alignment mismatch.

Each epoch uses the same 5,576,448-entry prefix of the immutable stored
permutation out of 5,576,491 available sequences. The fixed excluded tail is
43 sequences / 5,504 tokens. At 8,192 packed tokens per update, an epoch is
87,132 updates / 713,785,344 tokens. All nine runs share epoch one; the five
elastic runs share four deterministic epoch orders without resetting RNGs or
clocks. Total campaign training is 17,130,848,256 tokens. One elastic run matches
the aggregate token budget of the four standalones, without implying equal
per-run compute, runtime, or realized width coverage.

Optimizer training, ordinary validation, controller, and final holdout have
disjoint reserved roles. Controller data does not guide actions. Ordinary
validation runs every 64 updates and at completion, using valid causal-target
weighted loss and perplexity equal to its exponential.

## Artifact locations and record conventions

The following paths are planned outputs, not artifacts produced by Phase 1.
The future campaign input is
`configs/controlled_exps/tinystories_instruct_optimizer_ownership.yaml`, and the
future thin CLI is `scripts/analyze_tinystories_optimizer_ownership.py`.
The [CLI contract](../specs/013-tinystories-optimizer-ownership/contracts/cli-entrypoints.md)
and [quickstart](../specs/013-tinystories-optimizer-ownership/quickstart.md)
define later commands and required arguments.

| Location | Planned artifacts |
| --- | --- |
| `/scratch/ivo.navarrete/tmp/optimizer-ownership-campaign/` | `preflight.json`, `campaign_manifest.json`, nine `configs/<arm>.yaml` |
| `/scratch/ivo.navarrete/tmp/optimizer-ownership-runs/<arm>/` | Existing resolved config, metrics CSV, summary and resumable checkpoints; `optimizer_ownership_trace.jsonl`, `optimizer_ownership_clipping.jsonl`, `resource_attempts.json`, `terminal_validation_results.json` |
| `/scratch/ivo.navarrete/tmp/optimizer-ownership-frozen/` | `frozen_manifest.json` binding nine terminal identities and sidecar hashes |
| `/scratch/ivo.navarrete/tmp/optimizer-ownership-report/` | `comparison_report.json`, `optimizer_ownership_endpoints.csv` and `.json`, individual trajectory/resource plots, combined figures |

Records use plain JSON-compatible dictionaries, explicit schema versions,
ordered arm/width definitions, string paths, and full campaign/run identities.
Reuse `src.utils.reproducibility.stable_hash` for canonical JSON hashes and
`src.utils.metrics.write_json_artifact` for atomic JSON publication. Future
scientific contract serialization must preserve historical signature inputs.
Terminal sidecar content hashes exclude their own hash field.

The 24 endpoint keys are `(campaign_id, arm_id, run_id, width)`: four widths for
each elastic arm and one source width for each standalone. Each records exact
active counts, terminal checkpoint and ordinary-validation identities, evaluated
targets, loss/perplexity, actual/assigned budgets, provenance, exposure, and
resources. Freeze/report validates complete controls and durable terminals;
best checkpoints, trailing means, substituted endpoints, and mixed roles fail.
Partial diagnostics require explicit opt-in and visible missing-point labels.

Two combined figure stems are required in both PNG and PDF:

- `optimizer_ownership_perplexity_vs_non_embedding_parameters`
- `optimizer_ownership_loss_vs_non_embedding_parameters`

Each has five connected four-point elastic series and four disconnected
standalone markers at the exact integer counts, with consistent colors and no
seed error bars. Interpret S1/S2 and C1/C2 as history comparisons, S1/C1 and S2/C2
as representation comparisons, C1/C3 as a clipping comparison with different
combined caps, and each elastic width against its matching standalone.

## Requirement-to-verification outline

All runtime evidence below is pending later phases. Tests use small real models
or controlled fixtures and write their outcomes into this record.

| Requirements | Tasks | Required evidence |
| --- | --- | --- |
| FR-001–005,015; EX-001–009; SC-001–002 | T003–T015 | Config/CLI and campaign tests: exact matrix, fresh identities, pinned audit/alignment, model counts, deterministic action/order digests, named changed-control rejections |
| FR-006–011; SC-003 | T004–T005, T016–T021, T029 | Real slicing/concat gradients, tail updates, lazy C2 histories, complete tied/bias C3 partition, required counters |
| FR-012–014,017,019,021; EX-011 | T017–T024, T027, T032, T035 | Independent caps, active-zero/inactive-null observations, one global clock, C1/C3 test-only global-clipping parity, complete commit boundary |
| FR-016–020; SC-004–005 | T026–T036 | Exact action/batch resume around epoch boundaries; malformed restore leaves live state unchanged; owner/clock/accounting failures preserve the prior durable checkpoint; cumulative attempt costs |
| FR-018,020–023,025; EX-004,010,013; SC-005,007 | T037–T045 | Reconciled artifacts, actual state tensor bytes/dtypes, separate temporary/device peaks, terminal sidecar recovery with zero further updates, individual plots |
| FR-022,024–027; EX-005,011–013; SC-006–008 | T046–T054 | Nine terminals / 24 valid endpoints, identical CSV/JSON rows, two PNG/PDF figures, complete/partial rejection fixtures and descriptive seed-42 interpretation |
| FR-028; SC-001–008 | T055–T058 | Focused CPU and historical compatibility suites plus short GPU bf16 diagnostics; recorded commands, tolerances and environment limitations |

Resource verification sums the latest elapsed time per unique attempt and takes
the maximum peak across attempts, retaining failed/replayed work and flagging
incomplete hard-kill measurements. Checkpoint watermarks are not added twice.
After optimizer mutation begins, a failed multi-owner update must abort and
must never replace the prior durable checkpoint with partial state.

## Execution boundaries

1. **Implementation diagnostics:** setup checks, CPU fixtures and later short GPU
   bf16 checks establish tooling semantics. Read-only corpus audit and preflight
   can materialize configs without training or creating run directories.
2. **Later full campaign:** a separate researcher request follows implementation
   verification and successful preflight. Each fresh immutable arm uses the
   existing trainer. Continuation keeps its identity and full schedule horizon;
   a completed run missing a terminal sidecar performs completion-only recovery.
3. **Sealed holdout:** implementation, training, freeze, and reporting use no
   holdout model evaluation. Any future holdout comparison requires a separate
   request, prior freeze of all nine terminal checkpoints, and uniform evaluation
   of all 24 endpoints.

## Phase 1 verification record

On 2026-09-09, setup checks passed under the pinned environment's Python 3.12.13:
the ordered nine arms, exact four active counts, 24 endpoint slots, per-arm
updates/tokens/epochs, and aggregate 17,130,848,256-token budget agree. A temporary
JSON artifact round-trip preserved the canonical hash, used the existing atomic
writer and hash helper directly, and left no temporary artifact behind. All
local Markdown links in this runbook resolve. The existing `.gitignore` covers
the Python environment, generated outputs, caches, and editor files.

```bash
/home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_src_layout.py tests/test_reproducibility.py tests/test_artifacts.py -q
```

Result: **83 passed**, with two SWIG import deprecation warnings, in 8.65 seconds.
`git diff --check` also passed. T001 and T002 are complete; later tasks remain
unchecked. No runtime ownership test, campaign preflight, training, or holdout
evaluation is claimed by the setup phase.
