# TinyStories optimizer ownership experiment

## Implementation status and scope

Phase 1 establishes the experiment record and fixed definitions in
[`src/evaluation/optimizer_ownership.py`](../src/evaluation/optimizer_ownership.py).
Phase 2 supplies campaign-only scientific hashing, physical FFN metadata and a
validated static five-owner concat partition (T003–T005).
Phase 3 implements nine-arm preflight, fixed-control and identity validation,
C3 config eligibility, CPU model/partition inspection, and full expected traces
(T006–T015). Phases 4–5 implement C3 stepping/clipping and durable exact resume
(T016–T036). Phase 6 adds committed trace/clipping artifacts, measured storage and
resource summaries, immutable terminal ordinary-validation sidecars and per-run
plots (T037–T045). Phase 7 implements strict freeze, 24-endpoint export and
complete/partial comparison reports (T046–T054). Phase 8 finalizes commands and
compatibility evidence, with native-CUDA bf16 tests that explicitly skip when no
GPU is available (T055–T058). See the final
[verification record](../specs/013-tinystories-optimizer-ownership/verification.md)
for CPU results and the successful Slurm GPU follow-up (job 220964). No full-budget
campaign results exist; evidence comes from short diagnostics and controlled
fixtures. Full campaign launch and future uniform holdout evaluation each
require a separate researcher request.

The schema-1 `build_optimizer_ownership_signature` helper in
`src/utils/reproducibility.py` accepts explicit resolved contract sections and
returns `(hash, JSON-compatible inputs)`. Required sections cover campaign/run/arm
identity, representation, state scope, clipping, initialization, model, optimizer,
sampling, data, budget, evaluation and count convention. It hashes every supplied
control, normalizes tuples to JSON arrays and rejects missing sections, unsupported
schema versions, non-string keys and nonfinite/nonserializable values. Preflight
supplies and validates the fixed scientific values; this helper does not resolve
trainer defaults. Historical paired/full signature functions remain unchanged.

`physical_parameter_metadata()` on both FFNs describes whole physical tensors,
concat segment indices and gradient presence by width, including the common down
bias. `build_parameter_descriptors(model, ordered_widths=...)` retains canonical
names, tied aliases, shapes/dtypes, trainability, scalar counts and quarter/support
metadata in model registration order. `build_concat_parameter_partition` uses
those descriptors to return ordered O-A/O-B/O-C/O-D/O-common groups without
constructing optimizers. It rejects unequal quarters, invalid block shapes,
unclassified FFN parameters, conflicting tied ownership and mixed FFN layouts.
Frozen parameters remain described and are excluded from owner groups.

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

The implemented campaign input is
`configs/controlled_exps/tinystories_instruct_optimizer_ownership.yaml`, and its
thin CLI is `scripts/analyze_tinystories_optimizer_ownership.py`. The example
paths below describe a future authorized campaign; actual diagnostic artifact
paths appear in the dated evidence sections.
The [CLI contract](../specs/013-tinystories-optimizer-ownership/contracts/cli-entrypoints.md)
and [quickstart](../specs/013-tinystories-optimizer-ownership/quickstart.md)
define later commands and required arguments.

| Location | Artifacts |
| --- | --- |
| `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/campaign/` | `preflight.json`, `campaign_manifest.json`, nine `configs/<arm>.yaml` |
| `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/runs/<arm>/` | Existing resolved config, metrics CSV, summary and resumable checkpoints; `optimizer_ownership_trace.jsonl`, `optimizer_ownership_clipping.jsonl`, `resource_attempts.json`, `terminal_validation_results.json` |
| `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/frozen/` | `frozen_manifest.json` binding nine terminal identities and sidecar hashes |
| `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/report/` | `comparison_report.json`, `optimizer_ownership_endpoints.csv` and `.json`, individual trajectory/resource plots, combined figures |

Records use plain JSON-compatible dictionaries, explicit schema versions,
ordered arm/width definitions, string paths, and full campaign/run identities.
The implementation uses `src.utils.reproducibility.stable_hash` for canonical
JSON hashes and `src.utils.metrics.write_json_artifact` for atomic JSON
publication. Campaign serialization preserves historical signature inputs.
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

The table maps the full feature's required evidence to the implemented checks.
The dated sections below and the final
[requirement reconciliation](../specs/013-tinystories-optimizer-ownership/verification.md#requirement-reconciliation)
distinguish real input audits, model diagnostics and synthetic endpoint fixtures
from future full-budget observations. GPU results require a compatible device.

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

## Phase 2 verification (2026-09-09)

Both new helper suites initially failed on the missing APIs before implementation.
Final CPU verification used the pinned environment:

```bash
/home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest \
  tests/test_reproducibility.py tests/test_optimizer_ownership.py \
  tests/test_matformer_prefixes.py tests/test_model_size.py \
  tests/test_per_granularity_optimizer.py tests/test_per_granularity_optimizer_resume.py -q
```

Result: **186 passed**, with two dependency SWIG deprecation warnings.
`git diff --check` also passed. Evidence includes canonical JSON/hash stability,
changed scientific controls, pinned legacy signatures, exact real-model gradient
presence, unchanged FFN forward/backward values, frozen/tied/bias coverage,
malformed partition rejection and no optimizer construction by static helpers.
The d64/l4/v2048/FFN256 fixture has 49,152 parameters per quarter owner and
328,256 common parameters (524,864 physical parameters total).

These are foundation and compatibility checks. Later optimizer/clipping tests
will extend `tests/test_optimizer_ownership.py`; T016 and subsequent tasks remain
incomplete. No full campaign or sealed-holdout evaluation was run, and Phase 2
does not establish C3 runtime or GPU bf16 behavior.

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


## Phase 3 verification — 2026-09-09

T006–T015 are implemented and verified. The focused missing-behavior tests first
failed for the unsupported scope, missing campaign recipe, and stale materialized
controls. Final verification used the pinned environment:

```bash
/home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest \
  tests/test_optimizer_ownership_campaign.py tests/test_config.py \
  tests/test_train_cli.py tests/test_reproducibility.py -q
```

Result: **343 passed**, with two existing SWIG deprecation warnings. These tests
cover the supported C3 config and five owner caps; unsupported scope/topology/
action/distribution/correction/warmup and invalid clipping rejection; legacy
optimizer mappings/signatures; fixed common and arm controls; complete expected
RNG/sampler digests; exact real-model counts; normal seed construction and RNG
restoration; fresh model parameter objects; and staged publication with injected
model/publication failures. A saved trainer config rejects changed scientific
controls or a stale contract hash. Small CPU fixtures in the existing CLI and
reproducibility suites remain diagnostic tests, not campaign runs.

The read-only audit command from the quickstart passed again: all **89 shards**,
**5,576,491** stored ordering entries, source exhaustion, tokenizer identity and
all reserved-role intersections were verified. It retained all eight pinned hashes
from `inspection.md`. The designated epoch contains **5,576,448 sequences**,
excluding the same **43 sequences / 5,504 tokens**. The full audit result is
embedded in the published `preflight.json`.

The final campaign command was:

```bash
/home/ivo.navarrete/.conda/envs/elasticnn/bin/python \
  scripts/analyze_tinystories_optimizer_ownership.py preflight \
  --campaign configs/controlled_exps/tinystories_instruct_optimizer_ownership.yaml \
  --prepared-corpus-dir /nfs-stor/ivo.navarrete/matformer-corpora/tinystories-instruct-packed-full-v1 \
  --tokenizer-dir /nfs-stor/ivo.navarrete/matformer-tokenizers/tinystories-instruct-sentencepiece-bpe-2k-v1 \
  --output-dir /scratch/ivo.navarrete/tmp/optimizer-ownership-phase3-verified-20260909 \
  --run-output-root /scratch/ivo.navarrete/tmp/optimizer-ownership-phase3-verified-runs-20260909
```

Artifacts:

- Manifest: `/scratch/ivo.navarrete/tmp/optimizer-ownership-phase3-verified-20260909/campaign_manifest.json`
- Audit and digest report: `/scratch/ivo.navarrete/tmp/optimizer-ownership-phase3-verified-20260909/preflight.json`
- Nine ordinary trainer YAMLs: the same directory's `configs/`.
- Identity reservation: `/scratch/ivo.navarrete/tmp/.optimizer-ownership-phase3-verified-runs-20260909.optimizer-ownership-reservation.json`.

The reservation is an exclusive sibling sidecar and records the manifest hash
and all nine run IDs. Preflight rejects occupied arm paths or an existing
reservation. It does not create the run root or any arm directory. On a failed
publication it removes its reservation and staged output. Use fresh output/root
paths for another preflight; campaign IDs should distinguish separately intended
experiments. The manifest stores every resolved control, the allowed-difference
matrix, constructor/seed provenance, code revision and source-file checksums
(including uncommitted implementation), dependency versions, model descriptors,
owner topology and per-run scientific hashes.

CPU checks verified dense FFN dimensions **64/128/192/256** and matching active
non-embedding counts **115264/164416/213568/262720** across representations.
Each concat quarter owner contains **49152** parameter elements across layers;
common contains **328256**, for **524864** total physical parameters. The clipping
contract binds the inspected parameter topology. The four standalones resolve to
**87132 updates / 713785344 tokens / one epoch**; the five elastic arms resolve to
**348528 updates / 2855141376 tokens / four epochs**. No optimizer is constructed
by campaign preflight and no forward/backward or model evaluation occurs.

All nine first epochs match, and all five elastic four-epoch streams match.
Digests encode flattened ordered sequence IDs as little-endian uint64; fixed batch
size 64 makes every batch recoverable from that order. Each later epoch uses the
existing repeat sampler over precisely the same stored-permutation prefix.

| Epoch (one-based) | SHA256 |
| --- | --- |
| 1 | `b12bb2e42d4f6625f48c68dd39a872a962ffa01cbbfc87c0edbe47368d387f27` |
| 2 | `0359d3e543bbb65fe0cb401f872a8c28fdd10d68a387157d324a52097e4b5cd4` |
| 3 | `32b4df4d8a74bc017076c3a20135649ab2e933b53ee0b9cfa1f86b08fb7d949c` |
| 4 | `7e36c5b85d9ba4a92c94837224635689a47a0deefa4a5376d18c8ed4f03c78fb` |

Fixed designated-set hash: `1329fd4243b12fad836f2ba1328b4bd679450d516ae643ca8473ca57c953f0a2`.

Complete elastic action SHA256: `275c4fd957d103c609b3cb1ae9e7d1f37635e58e6eca92994261ddb5f0346b26`.
Encoding is one ASCII width label plus LF per update, generated with an isolated
`granularity_selection` RNG using the trainer's uniform `randrange(4)` path.
The actual expected trace contains g250=86898,
g500=87221, g750=87337,
g1000=87072 selections, totaling 348528. The statistical
expectation of 87132 per width is recorded separately; counts are not balanced.
These are precomputed traces, not observations of completed training.

Manifest hash: `0e1499f79ebc31ae79e62f0db1d51874f6071745e2bf4f2118e948305a77d8ed`.

The following config-only CLI also passed and reported C3 ownership, per-owner
L2 clipping, campaign identity and the bound scientific contract:

```bash
/home/ivo.navarrete/.conda/envs/elasticnn/bin/python train.py \
  --config /scratch/ivo.navarrete/tmp/optimizer-ownership-phase3-verified-20260909/configs/C3.yaml \
  --preflight
```

Named CLI rejection evidence is saved in
`/scratch/ivo.navarrete/tmp/optimizer-ownership-phase3-rejections-20260909.json`:
changed learning rate, stale corpus hash, reserved run identities and a stale
materialized contract hash all fail nonzero. Unit tests additionally reject
changed tail alignment, partial/wrong horizons, missing/extra arms, historical
run IDs, changed clipping and injected partial failures without success output.

**Boundary:** no campaign training or sealed-holdout model evaluation ran. The
published records explicitly set `training_started=false`,
`holdout_evaluated=false`, and `runtime_ownership_verified=false`. The Phase 3 records alone do not verify optimizer stepping or clipping; see the
Phase 4 evidence below. Later phases add exact campaign resume, terminal
sidecars and freeze/report commands.

## Phase 4 — ownership and clipping diagnostics (2026-09-09)

T016–T025 are complete. `BlockOptimizerCollection` constructs five disjoint lazy
AdamW owners from the existing concat partition, including segment biases, common
down biases and identity-deduplicated tied parameters. Its serialized state records
ordered descriptors, support, histories, width selections and successful owner
counts under an explicit block-collection version. The global clock synchronizes
all five owners at construction and after each complete update.

The existing trainer clears all model gradients to None, applies the resolved
clipping contract and steps active quarters in A/B/C/D order followed by common.
It marks mutation in flight before the first owner call, stages returned owners
locally, advances the clock once and reconciles committed exposure/token/cursor
accounting before publishing `last_clipping_observation` and clearing the flag.
Nonfinite loss/gradients/norms and unsynchronized owner rates fail before stepping.
Clipping observations contain detached pre/post norms, coefficients, inactive
null fields and combined disjoint norms. Per-owner clipping has no later global
rescale; the global arms retain a single global coefficient. Other supported
concat layouts retain ordinary global clipping without a four-quarter partition.

Verification used the pinned Python 3.12 `elasticnn` environment on CPU, with
small real `ModifiedLlamaMLP`/`CatLlamaMLP` modules and a two-layer Llama trainer:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest \
  tests/test_optimizer_ownership.py tests/test_per_granularity_optimizer.py -q
```

Result: **50 passed**. The original static ownership/topology checks remain in
place. New semantic checks establish:

- S1 wider-then-narrower backward produces full-shaped gradients with zero tails;
  tail momentum decays by beta1 while ordinary AdamW momentum/decay still changes
  tail weights and increments the full physical parameter counter.
- S2 retains isolated selected-width histories with full-sized moments and zero
  never-exposed tail moments. Nonselected histories remain bitwise unchanged.
- C1/C2/C3 inactive concat quarters retain absent gradients and bitwise unchanged
  weights/history/counters. Active present-zero gradients still perform ordinary
  AdamW updates. C2 allocates quarter histories in multiplicities 4/3/2/1 and four
  common histories after each width has been selected once.
- C3's quarter/common calls after one selection of every width are 4/3/2/1/4,
  while the global scheduler advances four times. Tied parameters appear once;
  bias ownership spans both real FFN layers.
- Independent cap-1 clipping reaches combined norms sqrt(2), sqrt(3), sqrt(4)
  and sqrt(5), checked to absolute tolerance **2e-6**. Changing common gradients
  does not change the A-owner coefficient. Active-zero coefficients equal 1;
  inactive observations have null norms/coefficient/cap. Observations consume no
  Python or PyTorch RNG draws.
- The internal `_diagnostic_global_clip=True` construction matches C1 parameters
  and every allocated AdamW history component after eight alternating-width
  updates, with **rtol=1e-6, atol=1e-7**. Both start from copied concat tensors and
  use identical data, actions and cosine rates. Campaign preflight continues to
  reject globally clipped C3; this override is not a YAML setting.
- An eight-update real-Llama loop spans four synthetic two-batch epochs, with
  exactly eight forwards. Every owner call observes the in-flight flag, every
  callback sees a reconciled commit, token counts equal 8 times committed steps,
  and batch/epoch cursors, width exposures and owner counts agree. Initial warmup
  rates and the full-horizon cosine schedule match shared optimizer scheduling;
  inactive owner rates stay synchronized and all rates reach zero at the horizon.

Compatibility checks:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest \
  tests/test_training_smoke.py tests/test_config.py tests/test_train_cli.py \
  tests/test_optimizer_ownership_campaign.py \
  tests/test_per_granularity_optimizer_resume.py \
  tests/test_global_sampling_windows.py -q
```

Result: **386 passed, 1 expected failure**. Output was saved to
`/tmp/optimizer-ownership-phase4-compatibility.log`. Only existing SWIG import
deprecation warnings were emitted. `git diff --check` passed. The existing
`.gitignore` already covers the Python environment, artifacts and editor files;
no additional tool-specific ignore files were needed.

**Scope boundary:** these are short CPU diagnostic results, not campaign quality
or performance outcomes. Full campaign training and sealed-holdout evaluation
were not launched. Phase 5 still owns campaign checkpoint loading, whole-bundle
restore validation, all-save-path unsafe-state gating, failure durability and
resource-attempt accounting. This phase does not establish interrupted C3 resume
support. Persistent clipping reports and terminal outputs remain in later phases;
GPU bf16 validation remains T057.

## Phase 5 verification — exact continuation and failure durability (2026-09-09)

Implemented T026–T036. Campaign resumable checkpoints now carry
`optimizer_ownership_checkpoint_schema_version=1`, the full scientific contract,
ordered parameter descriptors, clipping/budget/epoch identities, reconciled
width/quarter/owner accounting and a resource-ledger watermark. C3's explicitly
versioned collection has a complete validated loader. Historical shared and
per-width checkpoints retain their existing compatibility path.

AdamW validation derives required parameter counters from committed exposure:
full tensors for slicing, support-dependent histories for concat, and disjoint
quarter/common counts for C3. It rejects lost histories and impossible allocated
histories, reordered IDs/owners, changed kwargs, wrong shapes/dtypes, nonfinite
moments, negative second moments and nonintegral or incorrect counters. True
unexposed state remains absent. Model tensors, tied aliases, scheduler formula
and full horizon, RNG payloads, seeded action ordinal, sampler membership/cursor,
metrics and ledger watermarks are checked before installing anything. A snapshot
of the whole live bundle is taken only at resume to recover from unexpected
installation failure; no model or optimizer snapshot was added to the hot loop.

The update boundary remains unsafe from immediately before the first optimizer
call through scheduler and accounting completion. A mutation-then-raise failure
poisons the live state and aborts. All resumable save entry points reject it;
the operational failure record identifies the pending update, failure stage,
active/returned owners and last durable checkpoint. Pre-mutation failures restore
the existing RNG/data/accounting transaction and clear gradients.

Repeat-sampler provenance now includes total cursor, epoch position, fixed-set
hash, order policy and data seed. Campaign epoch/batch accounting uses the
sampler's logical position, including exactly completed terminal epochs. The
raw-loader epoch normalization bug found at resume was fixed. Creating a campaign
DataLoader iterator preserves the model RNG, so reopening the loader cannot add
an extra model RNG draw. Resume segregates non-durable JSONL trace/clipping rows;
the existing metrics journal repairs CSV rows beyond the restored boundary.

`resource_attempts.json` is an atomic schema-1 ledger with a unique ID per attempt,
source checkpoint hash/step, timestamps, latest observation sequence/duration,
attempted updates, allocated/reserved peaks and status/completeness. Observations
occur at start, metric cadence, heartbeat, checkpoint, completion and failure
boundaries. Durations sum once per attempt and peaks take the maximum; checkpoint
watermarks only reference observations and never add costs again. Replayed work
and failed attempts remain in the ledger. An unfinalized attempt is explicitly
incomplete, and unavailable accelerator measurements are null. Campaign summaries
use cumulative observed wall time and peaks.

Verification command:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest \
  tests/test_optimizer_ownership_resume.py tests/test_optimizer_ownership.py \
  tests/test_per_granularity_optimizer_resume.py tests/test_per_granularity_optimizer.py \
  tests/test_training_smoke.py tests/test_artifacts.py tests/test_reporting.py \
  tests/test_packed_corpus.py tests/test_global_sampling_windows.py -q --tb=short
```

Result: **513 passed, 1 expected failure** in 63.95 seconds. Log:
`/tmp/optimizer-ownership-phase5-verification.log`. The focused ownership/resume
and historical resume suites separately passed **233 tests**. Only existing
SWIG deprecation warnings were emitted. `git diff --check` and Python compilation
also passed. After the final placement of the pre-mutation clock-rate check,
**86 targeted failure/rejection tests passed**; their log is
`/tmp/optimizer-ownership-phase5-failure-check.log`.

Evidence includes 54 all-arm round trips before/at/after a synthetic two-update
epoch boundary: 27 raw-loader cases and 27 real repeat-sampler cases. Packed cases
compare actual ordered input batches, actions, final RNG, model tensors, AdamW
histories, scheduler state and sampler cursor exactly (`torch.equal`, not a relaxed
numerical tolerance). Additional cases verify fresh lazy checkpoints, corrupted
payload non-mutation, whole-bundle rollback after a scheduler installation failure,
and first/middle/last owner plus scheduler/accounting failures preserving the
previous durable checkpoint SHA256. Trainer-level checks verify a failed second
update retains checkpoint step 1 and attempt costs; reopening a completed run
records a second attempt with zero additional optimizer updates. Ledger fixtures
verify stale/repeated observation handling, older-checkpoint replay, null CPU
accelerator metrics and incomplete hard-kill measurements.

These are short CPU diagnostics with synthetic budgets, not full campaign
results. No full campaign or sealed-holdout evaluation was launched. Persistent
clipping reports, terminal-validation sidecars, endpoint reporting and GPU bf16
verification remain in their later task phases.

## Phase 6 verification — audit artifacts and terminal recovery

Phase 6 implements T037–T045. Campaign training durably appends one schema-1
`optimizer_ownership_trace.jsonl` record per committed update, including the
run/contract/attempt identity, selected action and digest, packed sample IDs and
batch digest, reproducible cursor/order identity, tokens, exposures, owner calls
and scheduler position. C1/C3 also append `optimizer_ownership_clipping.jsonl`
with the applied coefficients and separate active/null group observations.
The existing restore path segregates rows beyond the durable checkpoint.
CSV fields reference these artifacts without repeating the parameter partition.

`measure_optimizer_storage` reads existing state dictionaries without allocating
missing histories. Checkpoints and final summaries contain measured elements and
bytes by owner/component/dtype, with counters separate from moments. Four real
forward/backward passes on the pinned d64/l4/vocab2048 model, one at each width,
verified these allocated moment totals (F=196608, R=328256):

| Arms | Moment elements | Measured float32 moment bytes |
| --- | ---: | ---: |
| S1/C1/C3 | 1,049,728 | 4,198,912 |
| S2 | 4,198,912 | 16,795,648 |
| C2 | 3,609,088 | 14,436,352 |

These are persistent moment allocations, excluding counters, model weights and
other training memory. The 37.5% C2 saving is confined to FFN moments relative to
S2. Summary resource fields retain cumulative attempt duration, attempted work,
useful committed and attempted throughput, and separate allocated/reserved CUDA
peaks. Unsupported CPU accelerator metrics remain null. Temporary concat storage
is explicitly an active parameter-layout byte estimate excluding backward
workspaces, not a device peak. Realized width selections and quarter activations
are separate from labeled uniform-replacement expectations.

Terminal completion publishes or reuses the same durable resumable checkpoint,
checks its model against the live evaluated model, and writes immutable
`terminal_validation_results.json`. The canonical content hash excludes itself.
The sidecar records campaign/arm/run and full contract identity, checkpoint
SHA256/size, actual and assigned updates/tokens/epochs, representation/ownership/
clipping/initialization, ordinary-validation manifest and protocol, target-weighted
causal loss, exp(loss), evaluated examples/targets and exact active non-embedding
counts. Dense endpoints retain their source width; elastic terminals contain all
four widths. No best/trailing/holdout substitution is performed.

Injected checkpoint, evaluation and sidecar-publication failures leave no valid
sidecar. Recovery bypasses training, validates/reuses or reevaluates ordinary
validation, and retains the terminal checkpoint SHA256. Repeated completion
reuses a valid sidecar without evaluation or optimizer calls. A replaced
checkpoint model and changed sidecar identity are rejected. Recovery attempt
costs remain in `resource_attempts.json`. The weighted-loss test uses batches
with five and seven valid shifted targets and verifies `(5*loss1+7*loss2)/12`,
12 targets and two examples across every arm.

`report_run_artifacts(run_dir, output_dir)` in
`src/evaluation/optimizer_ownership.py` reads saved metrics, summary and trace
sidecars, reconciles actions/counts/tokens/cursors and C1/C3 group observations,
and writes `run_diagnostics.json` plus loss, perplexity and resource figures in
both PNG and PDF. Clipping frequencies use only active width/group observations;
unobserved/inactive denominators yield null frequencies. Resource figures retain
incomplete-measurement labels and distinguish storage, exposure, peaks, runtime
and throughput. The phase-7 CLI will invoke this saved-artifact reader.

Validation command (pinned CPU environment):

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest \
  tests/test_optimizer_ownership_reporting.py tests/test_optimizer_ownership_resume.py \
  tests/test_optimizer_ownership.py tests/test_per_granularity_optimizer.py \
  tests/test_per_granularity_optimizer_resume.py tests/test_data_validation.py \
  tests/test_artifacts.py tests/test_reporting.py tests/test_training_smoke.py \
  -q --disable-warnings --tb=short
```

Result: **510 passed, 1 existing expected failure** in 68.15 seconds. Log:
`/tmp/optimizer-ownership-phase6-verification.log`. Python compilation and
`git diff --check` also pass. An isolated evidence run exposed a test dependency
on determinism settings established by earlier trainer tests; the shared
campaign fixture now establishes strict determinism explicitly.

The isolated phase-6 evidence command is:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest \
  tests/test_optimizer_ownership_reporting.py tests/test_optimizer_ownership_resume.py \
  -k 'terminal or storage_reads or committed_saved or clipping_active or post_exposure' \
  --basetemp=/scratch/ivo.navarrete/tmp/optimizer-ownership-phase6-verified-20260909 \
  -q --disable-warnings --tb=short
```

Result: **39 passed, 163 deselected** in 13.24 seconds, independently of the
broader suite. Log: `/tmp/optimizer-ownership-phase6-artifacts-verified.log`.
Saved evidence root: `/scratch/ivo.navarrete/tmp/optimizer-ownership-phase6-verified-20260909`:

- `test_committed_saved_trace_an0` through `...an8`: nine-arm committed trace fixtures.
- `test_terminal_validation_count0` through `...count8`: nine-arm terminal checkpoint/sidecar fixtures.
- `test_clipping_active_denominat0/plots` and `...denominat1/plots`: C1/C3 controlled loss/perplexity series, active clipping denominators and PNG/PDF fixtures (the tests subsequently corrupt their source trace to verify rejection).
- `test_runtime_terminal_sidecar_0/per-granularity-optimizer-smoke-001`: intact trainer output after one failed terminal-publication attempt and two completion-only invocations; three finalized attempts, eight total attempted updates, unchanged checkpoint hash, valid sidecar and scalar journal.
- `runtime_plots`: six PNG/PDF figures and `run_diagnostics.json` generated afterward from that intact saved trainer output, with no model evaluation. The resource figure was visually inspected.

Use a fresh `--basetemp` for another run; pytest clears an existing base directory.
The diagnostic models use synthetic eight-update budgets, small deterministic
batches and two-update epoch fixtures. They do not establish full-budget
scientific outcomes or CUDA bf16 behavior. No full campaign or sealed-holdout
model evaluation was launched.

## Phase 7 — Frozen terminal comparison (T046–T054)

The campaign CLI now implements `freeze` and `report`. Freeze validates the saved
preflight matrix and explicit scientific controls, nine unique run identities,
completed summaries, resumable terminal checkpoint metadata and SHA256, bound
ordinary-validation sidecars, and runtime action/epoch digests. Reports recheck
all frozen source hashes and terminal evidence before publishing. Neither command
constructs or evaluates a model, launches training, or reads holdout results.
The historical per-width analyzer is unchanged.

After separately requested full training has supplied the nine terminals:

```bash
python scripts/analyze_tinystories_optimizer_ownership.py freeze \
  --campaign-manifest /nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/campaign/campaign_manifest.json \
  --run-root /nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/runs \
  --output-dir /nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/frozen
python scripts/analyze_tinystories_optimizer_ownership.py report \
  --manifest /nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/frozen/frozen_manifest.json \
  --output-dir /nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/report
```

`--run-dir` may be repeated instead of `--run-root`; saved identities determine
the arms. Both commands require unused output directories and publish their
staged outputs only after validation/export succeeds. A changed checkpoint,
summary, endpoint sidecar, trace, clipping file, metrics file or resource ledger
invalidates the freeze. A ledger appearing after freezing is also rejected.
Freeze again into a fresh directory after an intentional source change.

Explicit diagnostic subsets require `--allow-partial` independently on both
commands. Missing arms/widths are enumerated; supplied malformed evidence still
fails. Partial status appears in each CSV row, JSON exports, comparison and
individual reports, and the combined and individual figure titles. No missing
point is invented. A complete report contains exactly 24 endpoints, matching CSV
and JSON rows, six descriptive comparisons, nine individual diagnostic reports
with loss/perplexity/resource plots, and the two prescribed combined figure stems
in both PNG and PDF. Combined figures retain five separate four-point curves and
four disconnected standalone markers even for coincident measurements.

The pinned ordinary-validation corpus manifest contains **285 packed sequences**
(36,480 packed tokens) from **128 source documents**. Its context-128 batches
therefore supply **36,195 valid causal targets**. A read-only inspection of
`/nfs-stor/ivo.navarrete/matformer-corpora/tinystories-instruct-packed-full-v1/corpus_manifest.json`
confirmed these counts for role hash
`0c1beea552f54941e397d2442de736b1586e0292f6b1271b62d27ad782627856`.
Endpoint checks distinguish evaluated sequences from source documents and retain
`target_token_weighted_causal_shift_float64` with perplexity equal to exp(loss).

Interpretation is restricted to descriptive seed-42 observations:

- S1/S2 and C1/C2 compare history ownership within each representation.
- S1/C1 and S2/C2 also change inactive-tail momentum/decay, counters or lazy
  allocations when changing representation.
- C1/C3 compares global clipping at 1 with independent owner clipping at 1.
  Combined gradient bounds range from sqrt(2) to sqrt(5); they are not AdamW
  parameter-update bounds. C3 histories are shared across activating widths.
- Each elastic width is compared with its fresh matching standalone at the same
  active non-embedding count. Initialization is matched by the normal constructor
  seed, not cross-model tensor identity. One elastic run matches the aggregate
  training tokens of all four standalones; per-run compute, time and realized
  width exposure are not matched. Resource completeness flags remain visible.

Phase-7 acceptance uses synthetic terminal metadata at the pinned full horizons
and controlled bulk-trace digests, plus separate real eight-update model runs for
all nine arms. The latter exercise the terminal reader, real checkpoint loading,
streamed trace/accounting validation and ordinary evaluation without stubbing
those readers. Fixed scale/count expectations alone are reduced for that test.
Per-width/block collection checkpoints and shared checkpoints are both covered.
The initial missing-command acceptance checks failed as expected; subsequent
checks include rehashed control changes, wrong targets/counts/budgets/roles,
duplicate/missing/nonfinite endpoints, nonterminal/model-only checkpoint
substitution, changed action/epoch digests, frozen source replacement, and
publication/plot failure cleanup. A poisoned holdout JSON fixture remains unread.

These artifacts establish reporting correctness, not full-budget campaign
outcomes. Final compatibility results and GPU availability are recorded in
Phase 8 below. Full campaign execution and future uniform holdout evaluation
remain separately requested work.

Validation command (pinned environment):

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest \
  tests/test_optimizer_ownership_reporting.py tests/test_optimizer_ownership_campaign.py \
  -q --disable-warnings --tb=short \
  --basetemp=/scratch/ivo.navarrete/tmp/optimizer-ownership-phase7-final-20260909
```

Result: **99 passed**, two dependency deprecation warnings, in 53.20 seconds.
Log: `/tmp/optimizer-ownership-phase7-final.log`. Python compilation and
`git diff --check` pass. Use a new `--basetemp` when retaining this evidence;
pytest clears a reused base directory.

Saved evidence under
`/scratch/ivo.navarrete/tmp/optimizer-ownership-phase7-final-20260909`:

- `test_complete_freeze_tables_an0/frozen/frozen_manifest.json` and `.../report/`:
  complete synthetic 24-row CSV/JSON exports, nine individual reports, all four
  combined PNG/PDF files. Plot-data checks assert nine labeled series, five
  connected four-point curves, four unconnected markers, exact integer counts,
  and no error-bar containers. The combined loss PNG was visually inspected;
  all fixture values intentionally coincide while their labels remain separate.
- `test_partial_requires_two_opt_0/report/`: three valid C3 endpoints, 21 explicit
  omissions, partial tables/reports and labeled figures.
- `test_freeze_report_cli_saved_a0/report/`: eight endpoints from repeated C1/C3
  `--run-dir` inputs with independent freeze/report partial opt-ins.
- `test_terminal_reader_consumes_0` through `..._8`: real nine-arm diagnostic
  checkpoints, traces and terminal sidecars accepted by the terminal reader.

The complete fixtures intentionally use synthetic checkpoint tensors and bulk
trace digests. Their apparent full horizons and coincident losses are test
inputs, not measured campaign learning or resource outcomes. The separate real
runtime cases verify schema integration at eight updates, not full horizons.

## Phase 8 — Final workflow and compatibility evidence (T055–T058)

The [quickstart](../specs/013-tinystories-optimizer-ownership/quickstart.md) now
documents the implemented read-only audit/preflight, individual training,
continuation, completion-only recovery, freeze and report commands. For example,
after a separate researcher launch request and a fresh final-revision preflight:

```bash
export OO_BASE=/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1
mkdir -p "$OO_BASE/logs"
sbatch --exclude='gpu-[05,50,51]' --time=24:00:00 \
  --output="$OO_BASE/logs/C3-%j.out" \
  --error="$OO_BASE/logs/C3-%j.err" \
  scripts/slurm_tinystories_controlled.sh \
  --python-bin /home/ivo.navarrete/.conda/envs/elasticnn/bin/python \
  --config /nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/campaign/configs/C3.yaml \
  --output-dir /nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/runs/C3
```

Use this same command and immutable paths to resume an interrupted C3 run, or
recover a missing terminal sidecar from its existing terminal checkpoint. The
materialized config already enables continuation; no new resume flag or shortened
budget is needed. For another individual arm, select its manifest-recorded config
and output path, such as `ST-g250.yaml` and `optimizer-ownership-runs/ST-g250`.
Adding `--preflight` performs config-only inspection without training.

After an owner/clock/accounting mutation failure, discard the live poisoned state
and restart from the prior durable checkpoint. Preserve `resource_attempts.json`
and failure records so replayed work remains charged. Recovery at terminal budget
takes zero optimizer steps and preserves checkpoint SHA256; a valid sidecar is
reused. An invalid checkpoint or sidecar fails explicitly. Do not change the
horizon, cross-load arms, or restart fresh in an occupied run directory.

Freeze only after all intended continuation/recovery finishes. Report rechecks
the saved source hashes; changed summaries or resource ledgers require another
freeze into a new output directory. Both freeze and report require explicit
`--allow-partial` for a separately requested incomplete diagnostic. The six
comparison interpretations and resource caveats are tabulated in the
[quickstart](../specs/013-tinystories-optimizer-ownership/quickstart.md#6-read-the-comparisons-and-resource-limits).

The [final verification record](../specs/013-tinystories-optimizer-ownership/verification.md)
contains every requested test command, saved evidence paths and a complete
FR-001–028 / EX-001–013 / SC-001–008 reconciliation. The initial requested suites
produced **976 passes and one existing expected failure**; four Gloo cases failed
because sandbox sockets were prohibited, then **all four passed outside the
sandbox**. The final affected ownership suites produced **322 passes and 28 CUDA
skips**. The added Feature 12 fixture preserves six-run/three-seed reporting,
trailing-five validation means and preference for saved synthetic holdout results.

The initial login-node checks could not access CUDA. The subsequent Slurm GPU
follow-up passed **28 tests with no skips** on an NVIDIA A100-SXM4-40GB in job
**220964** on `gpu-52`, with exit code `0:0`. The test fixture now configures strict
determinism before CUDA initialization and verifies the settings when reusing the
CUDA context. Production determinism checks remain unchanged. The runtime test
measured 69,053,952 allocated and 90,177,536 reserved peak bytes for its eight-update
small model; these are diagnostic measurements, not full-campaign memory estimates.

All future campaign and diagnostic outputs belong under
`/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1`. GPU submissions
exclude `gpu-[05,50,51]` and use `sbatch`. The reusable
[`slurm_optimizer_ownership_gpu_check.sh`](../scripts/slurm_optimizer_ownership_gpu_check.sh)
launcher writes directly to the shared `diagnostics/` directory. Job 220964 was
already running on allowed node gpu-52 when the output-root correction arrived;
its completed evidence was copied to that directory without rewriting embedded
original paths. The verification record documents the failed attempts as well.

Phase 8 changes tests/documentation only; the campaign YAML, historical analyzer,
legacy hash inputs and training hot loop are unchanged. The restore-only full-
bundle guard remains outside per-step execution. Short real-model and synthetic
report tests verify tooling; full-budget learning/resource results and uniform
holdout evaluation remain separately requested researcher work.
