# Feature 015 verification record

## Scope and status

Authorized work: the 2026-09-21 continuation explicitly authorizes T034–T037,
including real-input preparation, GPU diagnostics, all production submissions and
valid own-run resumptions. Existing uncommitted implementation changes are
preserved. The current operational ledger below supersedes earlier pending-
authorization statements, which remain historical evidence. The real-input
preflight and corrected snapshot CPU checks passed, and preparation completed
(T034). GPU diagnostic job 271285 completed successfully on gpu-52; all 50 bound result
artifacts and the gate revalidated (T035 complete). Four T036 standalones are
submitted: 271319/271320 running, 271321/271322 pending at handoff. T036/T037
completion remains pending. The holdout remains sealed.

| Evidence stage | Status | Outstanding work |
| --- | --- | --- |
| Setup documents T001–T002 | Complete | Reconcile as later evidence arrives |
| Foundations T003–T005 | Complete | Focused CPU evidence below; end-to-end schema-4 resolution verified in phase 3 |
| Actual nine-model counts | CPU verified | Real-input campaign preflight remains T034 |
| Phase 3 protocol/preflight T006–T013 | Complete on CPU | Real-input audit remains T034 |
| Phase 4 ownership/accounting T014–T021 | Complete on CPU | GPU diagnostics pending; terminal-reader integration now has CPU fixtures |
| Phase 5 implementation T022–T033 | Complete on CPU | Real execution remains T034–T037 |
| Reporting prerequisites T038–T044/T046–T051 | Complete on CPU | Actual reports remain T045/T052 |
| Real corpus/tokenizer/role audit | Passed | T034 complete; real preflight ledger below |
| Historical four-terminal audit | Pending | T052 |
| Full snapshot-bound CPU gate | Passed; T034 complete | 1,307 passed, 39 GPU-only skips, 1 existing xfail; preparation validated |
| GPU gate | Passed; T035 complete | Job 271285 COMPLETED/0:0; nine real-shape probes and 118 GPU tests, zero skips; all hashes revalidated |
| Standalone barrier | Pending | Four validated production terminals |
| Nine-run production | T036 standalones submitted; T036/T037 incomplete | Jobs 271319–271322; no barrier or elastic admission |
| 24-endpoint new report | Pending | T045 |
| 28-endpoint combined report | Pending | T052 |

## Command and provenance ledger

For each check record date, exact command/cwd, interpreter and dependency versions,
source revision and file SHA-256s, recipe/config-set hashes, preflight hash,
stdout/artifact references, exit code and pass/fail/skip counts. Fixture hashes
must be identified separately from audited production config hashes.

- Cwd: `/home/ivo.navarrete/ElasticNN/matformer`.
- Interpreter: `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python`.
- CPU environment: `OMP_NUM_THREADS=1`.
- Baseline source revision for phases 1–2: `82fbc273879952b179a9eafa73782d7c45d5284a`; phase 5 baseline: `707096307a14e3c7235932a68ae94cb833f057f7`. Phase-specific tested file hashes are recorded below; current changes are uncommitted.
- Runtime inspected: Python 3.12.13 (conda-forge), PyTorch 2.11.0+cu128, Transformers 5.8.0, NumPy 2.4.3, PyYAML 6.0.3, pytest 9.0.3; Linux x86_64/glibc 2.39.
- Schema-4 recipe hash: recorded below. Production config-set/preflight hashes remain pending T034.
- Production snapshot hash: pending; local CPU checks are not the readiness gate.
- Foundation CPU results: 269 passed, 6 CUDA-only skips, exit code 0; later phase results follow below.
- GPU results: not run; GPU readiness pending (no skipped check counts as a pass).

For future GPU/production attempts record run ID, monotonic launch-attempt ID,
process UUID, sbatch command/job ID, hardware, start/end/exit status, source/config/
gate/barrier hashes, checkpoint and terminal sidecar references, measured elapsed
time/peaks and observation completeness. No attempts exist for this work.

Expected future artifact references are under the proposed root in the
[runbook](../../docs/tinystories-matformer-widths-experiment.md):
`campaign/{campaign_manifest,preflight}.json`, `diagnostics/{cpu,gpu}-gate.json`,
`launchers/standalone-barrier.json`, per-run traces/checkpoints/terminal sidecars,
`reports/new/endpoints.{csv,json}` and
`reports/combined/combined_endpoints.{csv,json}`. All remain pending.

## Functional requirement acceptance

Full campaign acceptance remains pending. CPU fixtures cover definitions, topology,
update semantics, continuation, clipping, accounting, operations and reporting.
Real-input preparation, GPU gates, production and actual reports remain pending.

| Requirement | Required evidence | Status |
| --- | --- | --- |
| FR-001 | Distinct nine-arm schema; unchanged legacy contracts and runtime | CPU definitions and legacy signatures verified; production pending |
| FR-002 | Nine fresh seed-42 initializations, own-run continuation only | CPU fresh construction and exact own-run continuation verified; production pending |
| FR-003 | Strict standalone-first barrier including restart/worker entry | Strict four-terminal, restart, tamper and worker-entry fixtures verified; actual barrier pending |
| FR-004 | Uniform global replacement H=1, unweighted causal updates | CPU uniform replacement H=1 verified |
| FR-005 | Identical action and designated epoch-batch traces | Full expected synthetic-order digests verified; real-input audit pending |
| FR-006 | Hashed physical grid/boundaries and actual model support | CPU geometry, counts, support and hashed topology verified |
| FR-007 | S1/S2 full-tensor history/update semantics | CPU all-width/full-history semantics verified |
| FR-008 | Inactive concat absence and C2 lazy support multiplicities | CPU absent/zero gradients and lazy histories verified |
| FR-009 | Five disjoint C3 owners including ties/common bias | CPU five owners, tied objects and optional biases verified |
| FR-010 | Real cap-1 clipping and single scheduler advance | CPU real clipping/order/clock verified; GPU pending |
| FR-011 | Strict controls/data/count/budget/identity preflight | Fixture preflight/publication/rejections verified; real audit pending |
| FR-012 | Complete restore validation before live mutation | CPU nine-arm boundary/mutation/rollback fixtures; GPU pending |
| FR-013 | Epoch-boundary equivalence and durable partial-failure rejection | CPU before/at/after epoch, owner/scheduler/accounting failure and zero-step recovery fixtures |
| FR-014 | Actual configs/provenance/traces/scalars/diagnostics | CPU configs/provenance/trace/scalar/diagnostic fixtures verified; production artifacts pending |
| FR-015 | Real C1/C3 sidecars and strict reader rejection cases | Real short C1/C3 sidecars pass strict terminal reader; missing/incomplete clips rejected |
| FR-016 | Measured unequal-block allocations and complete attempt resources | CPU allocation/exposure and unique-attempt/incomplete-resource fixtures verified; GPU/production measurements pending |
| FR-017 | Source/config-bound CPU/GPU readiness and compatibility | Snapshot/config binding and legacy compatibility fixtures verified; actual CPU/GPU gates pending |
| FR-018 | Authorized fresh root, sbatch limits, restart-safe admission | Reservation, live-limit, intent/reconciliation and worker fixtures verified; actual execution pending |
| FR-019 | Full-budget terminal ordinary validation and zero-step recovery | Full-budget metadata fixtures and real zero-step recovery verified; actual full-budget terminals pending |
| FR-020 | Matching 24/28 CSV/JSON with physical/run identity | 24/28-row CSV/JSON parity and physical/run identities verified with fixtures; actual exports pending |
| FR-021 | Four-only historical standalone revalidation | Four-only historical fixture selection and strict rejection verified; real historical audit pending |
| FR-022 | Strict missing/corrupt/nonterminal input rejection | Missing/corrupt/nonterminal rejection and atomic publication fixtures verified |
| FR-023 | Two combined figures in PNG/PDF | Four fixture PNG/PDF outputs and curve/marker structure verified; actual figures pending |
| FR-024 | Exact coincident coordinates and distinguishable standalone legends | Exact coincident fixture coordinates, concentric markers and legends verified |
| FR-025 | Descriptive saved-results interpretation | Required report annotations and interpretation implemented; saved production comparison pending |
| FR-026 | Evidence-aligned tasks/runbook/statuses | Implementation tasks/runbook reconciled with CPU evidence; operational completion pending |

## Success criterion acceptance

| Criterion | Required evidence | Status |
| --- | --- | --- |
| SC-001 | Nine actual definitions/counts and rejection cases | CPU acceptance verified; real-input audit pending |
| SC-002 | Zero production elastic starts before validated standalones | Admission/barrier fixtures verified; actual production ordering pending |
| SC-003 | All-width ownership/history/clipping/update semantics | CPU acceptance verified; GPU verification pending |
| SC-004 | Exact future actions/batches, numerical resume, atomic rejection | CPU nine-arm actions/batches/numerical resume and rejection verified; GPU probes pending |
| SC-005 | Nine full-budget terminals and reconciled complete traces | Pending |
| SC-006 | Traceable run artifacts, resource disclosures and C1/C3 clips | Resource/trace/clipping fixture disclosures verified; actual complete artifacts pending |
| SC-007 | 24/28 actual valid endpoints and corruption fixtures | 24/28-row corruption/identity fixtures verified; actual endpoints pending |
| SC-008 | Four combined figure files with five curves/eight markers | Four fixture files with five curves/eight markers verified; actual figures pending |
| SC-009 | Saved-artifact scientific comparison and interpretation | Pending |
| SC-010 | Full legacy compatibility, sealed holdout and evidence-aligned status | Relevant legacy CPU regressions verified; holdout sealed; actual campaign completion pending |

## Foundation verification results

Executed from the recorded cwd on 2026-09-21:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_matformer_widths_campaign.py tests/test_optimizer_ownership_campaign.py tests/test_optimizer_ownership_corrections.py tests/test_inverse_membership_sampling.py tests/test_reproducibility.py -q -rs --tb=short
```

Result: **269 passed, 6 skipped, 2 warnings in 46.20s**, exit code 0.
All six skips originate at `tests/test_optimizer_ownership.py:467` (imported by
the correction suite): CUDA unavailable; bf16 diagnostics require one GPU.
Warnings concern SWIG type metadata. No GPU gate has passed.

The new suite contains 24 foundation cases. It verifies the nine-arm declarations,
widths/count expectations, budgets, inherited controls, schema-qualified repeated
labels, unknown/noninteger-schema rejection, ordered boundary/support metadata,
detached construction and topology-sensitive schema-1 serialization. Schema-4
contract tests use declared-control fixtures, not fully resolved runtime configs.

Before source edits, all 20 legacy scientific signatures (nine schema-1, six
schema-2 and five schema-3 arms) were captured through the real resolver at the
baseline revision. The fixture substitutes external corpus/tokenizer audit IO
and constant source provenance; no real-input audit is claimed. Re-expansion after
the edits matches every stored hash and adds no new fields to historical inputs.
The fixture is `tests/fixtures/optimizer_ownership_legacy_signatures.json`.
Related regression suites exercise existing real CPU models, updates, resume,
preflight/report publication and reproducibility; they do not establish new-grid
runtime acceptance.

`git diff --check` passed. Relative links in the two new documents resolve.
`.gitignore` already covers essential Python, environment, build and editor files;
no other tool-specific ignore file was indicated by project setup.

| Task | Evidence | Status |
| --- | --- | --- |
| T001 | Runbook: matrix, controls/data, future layout/interfaces, stage barrier, authorization, Slurm limits, separate statuses | Complete |
| T002 | This FR/SC record, command/environment/hash fields, explicit pending gates/audits/terminals and attempt provenance | Complete |
| T003 | Schema-specific selectors and declared nine-arm grid; selector/rejection/legacy fixtures pass | Complete |
| T004 | New-only topology fields in campaign manifest and run contract construction; existing serializer unchanged and hashes tested | Complete |
| T005 | 24 foundation cases plus related regressions; all 20 pre-edit legacy signatures retained | Complete |

Schema-4 recipe creation, resolver marker/eligibility integration and materialized
config validation are T007–T009 and remain pending. New actual model counts,
training semantics, complete data traces and all reporting/launch gates remain
pending their tasks. This foundation verification is not full CPU readiness.

### Tested file SHA-256s

These identify the changed source/tests, unchanged serializer and legacy recipes
used here. The baseline revision plus these local file hashes identifies this
foundation change; it is not a production source-snapshot or config-set gate.

| File | SHA-256 |
| --- | --- |

| `src/evaluation/optimizer_ownership.py` | `928267fb5df112ab9e23ce236fc8a429368f91be590a21a74d2ee22ca4f6e08a` |
| `src/utils/reproducibility.py` | `b54771768da5a9ba7d093d73828e71bdfa943c2fbad55fd3f1613b1be094ecc1` |
| `tests/test_matformer_widths_campaign.py` | `8d10e978fd913f4594eec4d5ca0d69490d2db9e51b07e0e93870fde49e0ae017` |
| `tests/fixtures/optimizer_ownership_legacy_signatures.json` | `8a3d67fac6f7bc266866bc24b597de71dadb7b5dc1d0af0f54c914ce74201252` |
| `configs/controlled_exps/tinystories_instruct_optimizer_ownership.yaml` | `4a3218bd745c74a0c8aed3610d1d2b09dec889bdafaf751781c2f4eb099c0bea` |
| `configs/controlled_exps/tinystories_instruct_optimizer_ownership_corrections.yaml` | `278671bf2a666623b38f91c1e41cebe483773f37e3dda2c298fde3365c2c659a` |
| `configs/controlled_exps/tinystories_instruct_inverse_membership.yaml` | `f4c964cee967531cfab51447ec10fb0b0e843a76902c944ebb7827b32c8822f6` |


## Phase 3 CPU acceptance (2026-09-21)

T006–T013 complete. Command from the repository root:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_matformer_widths_campaign.py tests/test_optimizer_ownership_campaign.py tests/test_config.py -k 'not all_width_histories and not unequal_partition' -q -rs --tb=short
```

Result: **305 passed, 8 deselected, 2 warnings in 40.13s**, exit 0. The
8 deselections were phase-4 tests under development. Warnings concern SWIG
metadata. Actual CPU constructions verify all nine models, every endpoint count,
and physical concat owner sizes 24,576/24,576/49,152/98,304 with common 328,256.
Counts exclude embedding/head as specified. Full 348,528-action digests use the
ordinary isolated randrange stream. Real full-size repeating samplers use a
synthetic 5,576,491-entry uint64 permutation: all nine first epochs and five full
four-epoch traces agree, with 5,576,448 fixed designated sequences per epoch.
This is fixture evidence, not a real corpus/tokenizer audit.

Rejection checks cover controls, budgets, grid/marker/boundaries, fresh constructor
identity, source widths, data hashes, role overlap, occupied paths/reservations,
and failed publication. Existing CLI flags dispatch schema 4 without a new CLI.
Preflight publishes nine reloadable configs and reserves only temporary fixture
paths after all checks pass. No external campaign root was reserved.

## Phase 4 CPU ownership and accounting evidence

The real-model probes retain d64/l4 and FFN 256 with the schema-4 prefixes.
Short trainer fixtures explicitly use CPU float32, batch 1, eight-token contexts,
eight updates and a short diagnostic scheduler. They are temporary fixtures, not
production configs, full-horizon readiness checks or GPU evidence.

- All five elastic arms exercise every width, including full-width followed by
  g125. S1's zero tail retains momentum and decay; S2 retains isolated full-shaped
  histories and zero never-used tails across gate/up/down tensors in all layers.
- Inactive concat parameters and histories remain unchanged. Present zero
  gradients still advance AdamW. C2 lazily creates exactly 4/3/2/1 block histories
  and four common histories after every width appears.
- C3 has five disjoint owners, including tied-object and optional-bias coverage.
  Actual updates use current scheduler rates, ordered active block/common calls,
  and one global scheduler advance. All widths receive independent owner caps of
  1; the common gradient cannot alter another owner's coefficient. C1 retains
  its single global clip.
- Real C1/C3 trainer observations are written to JSONL and cross-checked against
  scalar metrics and summaries. Missing references/records, changed active flags,
  coefficients, combined norms and metric references are rejected. Strict
  full-budget terminal-reader integration remains T040/T044.
- The compact accumulator accepts g125 only with the matching schema-4 topology,
  rejects g750 for that grid, and retains bounded schema-2 state through 20,000
  attempts. Real S2/C3 checkpoint round-trips restore compact accounting. Full
  nine-arm epoch-boundary/failure acceptance remains phase 5.

For one actual update at each width, the probes measure the following moment
allocations in float32. Counts include embeddings/head because these are optimizer
storage measurements, distinct from the non-embedding comparison convention.

| Arm | Measured and support-derived moment elements | Measured moment bytes |
| --- | ---: | ---: |
| S1 | 1,049,728 | 4,198,912 |
| S2 | 4,198,912 | 16,795,648 |
| C1 | 1,049,728 | 4,198,912 |
| C2 | 3,363,328 | 13,453,312 |
| C3 | 1,049,728 | 4,198,912 |

C2's expectation is
`2*(4*24576 + 3*24576 + 2*49152 + 98304 + 4*328256)`.
Summaries also compute expectations from widths actually exposed, so unallocated
lazy histories are not reported as measured storage. Component/owner/dtype tensor
measurements and scalar counters remain separate from concat buffer estimates,
CUDA peaks, checkpoint bytes and throughput. CPU runs do not measure CUDA peaks.

The all-width semantic sequence has observed selections 1/1/1/1 and block calls
4/3/2/1, with four common calls. The eight-step real C1/C3 sidecar probe uses the
ordinary replacement primitive with diagnostic RNG seed 4: observed width counts
3/2/1/2, block activations 8/5/3/2, versus statistical expectations 2/2/2/2 and
8/6/4/2. This is not a production seed override or a balancing policy; full
seed-42 campaign expected traces are checked separately in phase 3.

The initial ownership regression command was:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_matformer_widths_campaign.py tests/test_metrics_compact_accounting.py tests/test_optimizer_ownership.py tests/test_optimizer_ownership_resume.py tests/test_per_granularity_optimizer.py -q -rs --tb=short
```

Result: **327 passed, 28 skipped, 2 warnings in 56.30s**, exit 0. Skips are
CUDA-only diagnostics at `tests/test_optimizer_ownership.py:467`; GPU readiness
remains pending. Additional rejection checks and the broader final command are
recorded below.


## Final phase 3–4 regression and provenance

Executed on 2026-09-21 from the recorded repository root, against the local
working tree based on `82fbc273879952b179a9eafa73782d7c45d5284a`:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_matformer_widths_campaign.py tests/test_metrics_compact_accounting.py tests/test_optimizer_ownership.py tests/test_optimizer_ownership_resume.py tests/test_optimizer_ownership_campaign.py tests/test_optimizer_ownership_reporting.py tests/test_optimizer_ownership_corrections.py tests/test_inverse_membership_sampling.py tests/test_inverse_membership_reporting.py tests/test_config.py tests/test_per_granularity_optimizer.py tests/test_per_granularity_optimizer_resume.py tests/test_metrics_history_performance.py tests/test_global_sampling_windows.py tests/test_reproducibility.py tests/test_model_size.py tests/test_packed_corpus.py tests/test_train_cli.py tests/test_training_smoke.py -q -rs --tb=short
```

Result: **1,062 passed, 34 skipped, 1 xfailed, 2 warnings in 166.36s**, exit 0.
All 34 skips are CUDA-only cases at `tests/test_optimizer_ownership.py:467`.
The pre-existing expected failure is
`tests/test_training_smoke.py::test_interrupted_and_relaunched_run_preserves_the_same_output_dir`
(existing marker: run-resumption wiring not implemented in that test).
Both warnings concern SWIG type metadata. No unexpected failure occurred.
Local command output: `/tmp/mw-final-validation.txt`; this record preserves its
result independently of temporary test-artifact retention. `git diff --check`
and Python AST parsing passed.

All T006–T021 tasks are complete at their CPU/fixture boundaries. T012 reuses the
existing analyzer preflight dispatch/flags; no CLI change was necessary. T018
also threads the validated campaign contract through the two compact-metrics
checkpoint restore call sites without changing the checkpoint schema. T016's
static layout validator is shared with T010 real-model preflight.

The final run retains all 20 legacy scientific signatures and the old unequal-
quarter rejection checks, and exercises historical reporting, optimizer restore,
metrics performance, global sampling, model sizes, corpus, CLI and training
regressions. This does not complete the later T053 gate: new queue/report suites,
real-input audit, source-snapshot binding and GPU diagnostics remain pending.
At that phase-3/4 checkpoint, phases 5–8, production, and the 24/28-endpoint
actual reports were unexecuted. Current phase-5 implementation evidence follows.

### Phase 3–4 tested file identities

These SHA-256s identify local implementation/test inputs, not a production
snapshot or audited config-set/preflight binding. Fixture output paths vary by
pytest temporary directory; no production config-set/preflight hash is claimed.

| File | SHA-256 |
| --- | --- |
| `configs/controlled_exps/tinystories_instruct_matformer_widths.yaml` | `e8a6774d002bfe983e308a4b319c077bbf4f2ba6e1908182fa5108afca0816d4` |
| `src/evaluation/optimizer_ownership.py` | `345b7fb97debd80a024cd76e759f551a9f9e944b963e3bc582b037d6aeb7683c` |
| `src/utils/config.py` | `56f3bd163cc252f229ffe7dc92c2bb6b60a590fc0282a0d6cf5ced4c8e16b9eb` |
| `src/training/optimizer_state.py` | `28f15b5fea923c9db4423343f46990ff4d8ae7bcb7094e639e4b9456d656618e` |
| `src/training/steps.py` | `f03088a2c9fefad8e7ba1c487c604254f3a3492c343f87aa7e1f31ba3695fa40` |
| `src/utils/metrics.py` | `b3943732405137cb4fa332e37fd97576d5af45d22c88c0f296bf87ab90412298` |
| `src/training/run.py` | `de31c55d32b38c062234cbf9bbdb7b5853b715b2e2e616fbb0ad5efb743288bc` |
| `src/training/checkpointing.py` | `bccfd97f920c0dda3f5249759ca0cb693767d4cf94570ffdf6ccd7bffad99982` |
| `src/utils/reproducibility.py` | `b54771768da5a9ba7d093d73828e71bdfa943c2fbad55fd3f1613b1be094ecc1` |
| `scripts/analyze_tinystories_optimizer_ownership.py` | `8c00fa0f945e20c07b477fd04b432eb64857f388201be262bccba391fc6ed9d2` |
| `tests/test_matformer_widths_campaign.py` | `93a90f46d3626f33ea6d5bb68be1713464878fed72a28ef9ef2b013722324b36` |
| `tests/test_metrics_compact_accounting.py` | `63f1c5cc33e3fc5c5e7de9a1c0f4a52ae6342cd1bb8dfa4feb984c80aae0758a` |
| `tests/fixtures/optimizer_ownership_legacy_signatures.json` | `8a3d67fac6f7bc266866bc24b597de71dadb7b5dc1d0af0f54c914ce74201252` |

Canonical recipe mapping hash (`stable_hash`): `9eb79a4cb5d4673709d80d4d13474cd73824d0981c011a01aef1522e820882ce`.


## Phase 5 implementation and required reporting prerequisites (2026-09-21)

Baseline revision: `707096307a14e3c7235932a68ae94cb833f057f7`. This invocation
implements T022–T033 and the reporting prerequisites required by T029/T032/T033.
All execution is local CPU testing with temporary fixtures. T034–T037, T045 and
T052 remain pending; no external campaign root, real-data audit, historical audit,
Slurm submission, production terminal or holdout evaluation was performed.

The resume matrix uses all nine real d64/l4 model geometries with eight-token,
batch-one CPU float32 diagnostic runs. The repeat sampler has two updates per
synthetic epoch and a fixed excluded tail. Checkpoints at steps 1/2/3 reproduce
future selected actions, actual sample IDs, RNG and model/optimizer/scheduler
state. S2/C2 histories retain full shape and lazy support. Corruption checks cover
identity/grid/boundary/probability/horizon/cursor/accounting, missing or extra
histories, nonfinite/dtype/shape/counter state and installation rollback. These
are diagnostic budgets and identities, not full production horizons.

C1/C3 resumed traces and clipping match uninterrupted observations after discarded
scientific suffixes are reconciled. Their `batch_indices` field is a local loader
enumeration and restarts; absolute cursors, epochs, sample IDs, actions and clipping
are compared exactly. Real short C1/C3 checkpoints, ordinary validation, scalar
metrics, trace and clipping sidecars reach the strict terminal reader with explicit
8-update/64-token and 1-example/7-target diagnostic overrides. Missing clipping
is rejected. Full-budget metadata fixtures separately require 285 examples and
36,195 targets and the assigned 87,132/348,528-update horizons.

Three runtime defects were exposed and corrected:

- Completion-only sidecar reuse compared in-memory topology tuples directly with
  JSON lists. It now compares their canonical scientific serialization, retaining
  the same terminal checkpoint hash through repeated zero-step recovery.
- A successful-update callback ran before scalar accounting; interruption could
  add a fictitious next-step failure row. Campaign callbacks now follow committed
  scalar accounting, and post-commit exceptions do not fabricate another attempt.
  Owner/scheduler/accounting failures still poison partial updates and preserve
  the previous durable checkpoint across all save paths.
- Scientific-observation, scalar-journal or resource-accounting write failures
  after optimizer mutation now also poison the boundary until committed scalar
  observations exist. Three injected IO failures verify that the previous
  durable checkpoint remains byte-identical.

The launcher verifies preflight, exact materialized configurations, immutable
source manifests and sealed CPU/GPU evidence. CPU diagnostics copy source before
executing the acceptance/regression command in that snapshot, recording JUnit/log
hashes. Snapshot-fixture tests substitute the expensive acceptance runner while
checking its immutable inputs and executing a real provenance import outside Git;
a separate subprocess fixture verifies the complete pytest command and result
capture. These tests do not claim a production CPU gate.

Preparation adopts the exact exclusive reservation and records unavailable history
separately. The barrier uses the shared strict terminal reader and binds all four
standalone source sets to campaign/source/preflight/contracts. Queue and worker
revalidate it, reject changed evidence, hold exclusive locks, and preserve durable
monotonic submission intents. Mocked Slurm tests cover unrelated jobs, tighter
limits, delayed visibility, unknown states, ambiguous names and no duplicate
submission after restart. CPU own-checkpoint inspection validates the full bundle
and seeded action ordinal without creating a CUDA context; the actual trainer
still validates GPU RNG bytes/topology before installation.

Resource ledger records bind run, launch attempt, job and process UUID. A worker
reserves an unknown observation before the trainer starts. Latest observations
are summed once per process, peaks use maxima, and killed/unobserved attempts
remain incomplete with lower-bound totals. Allocation time is retained separately
in scheduler accounting and is never added to overlapping process time.

Reporting fixtures validate exactly 24 new endpoints and 28 combined endpoints,
matching CSV/JSON cells, nine diagnostic sets, and independent repeated-size
standalones. The selected historical reader requires only the four original
standalones; unrelated historical elastic files are deliberately removed. The
historical g750 endpoint remains .75/192/213,568. Structural figure checks require
five four-point curves and eight disconnected markers at exact counts. Publication
failure/corruption tests retain existing valid outputs; missing history preserves
the new report and returns nonzero with combined status outstanding.

The implemented GPU runner remains **unexecuted**. It requires an sbatch job,
checks live limits, uses the tested snapshot, and records allocation/hardware.
Nine real-corpus batch-64/context-128 bf16 probes stop at four updates and resume
to eight without shortening scheduler horizons. A narrow local diagnostic adapter
permits only a distinct run ID/path while revalidating original scientific controls;
production validators remain strict. Separate forced-width semantic probes and
synthetic epoch/malformed/partial-failure tests execute in the same GPU process.
Any skipped GPU check fails readiness. No GPU result is inferred from CPU tests.

Constitution review: the existing trainer, optimizer collections, sampler,
serializer, metrics journal and terminal reader remain authoritative. The launcher
has explicit campaign stages and direct Slurm calls, with no scheduling framework
or optimizer registry. Whole-state copies occur only during restore or test probes,
never per optimizer update. Compact attempt accounting stays bounded. All new
reporting reads saved ordinary-validation evidence; final holdout remains sealed.

### Phase 5 commands and results

The complete final result follows the supplemental checks below.

Supplemental focused checks during integration:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_matformer_widths_campaign.py tests/test_matformer_widths_reporting.py tests/test_matformer_widths_queue.py tests/test_optimizer_ownership_reporting.py tests/test_inverse_membership_reporting.py -q -rs --tb=short
```

Result: **349 passed, 5 skipped, 2 warnings in 189.35s**, exit 0.
The skips are the separately authorized GPU semantic probes. This preceded the
last resource-disclosure and accounting-IO checks; the complete final command
below supersedes it. Log: `/tmp/mw-phase5-final-focused.txt`.

The last targeted IO durability check was:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_matformer_widths_campaign.py -k accounting_io -q --tb=short
```

Result: **3 passed, 206 deselected, 2 warnings in 7.05s**, exit 0.
Log: `/tmp/mw-phase5-io2.txt`. A generated combined fixture PNG was visually
inspected: concentric fresh/historical markers remain visible at identical
coordinates, with all five arm legends and the required ordinary-validation,
seed, count-convention and budget annotations. All fixture losses are synthetic.

### Final tested inputs

SHA-256 values captured before the final regression command and revalidated after
source edits ended. These identify local implementation/fixture inputs, not a
production snapshot or materialized campaign config set.

| File | SHA-256 |
| --- | --- |
| `configs/controlled_exps/tinystories_instruct_matformer_widths.yaml` | `e8a6774d002bfe983e308a4b319c077bbf4f2ba6e1908182fa5108afca0816d4` |
| `scripts/run_tinystories_matformer_widths.py` | `17fc3ffd171fd17d85c17123961178ca56688ba06fa866ad6f810e604e23e5ca` |
| `scripts/preflight_tinystories_matformer_widths.py` | `c115e642ac74f817da606da90123d05228633520912c1e19385e75ed186af459` |
| `scripts/analyze_tinystories_optimizer_ownership.py` | `9a7731292e5f8d96a43cfe9e11291b67ee939db22d136d77c5d1269d5facbeb1` |
| `src/evaluation/optimizer_ownership.py` | `0dc833c73992438ddb3d5a4ecc6de13d8f3cece5d91bdd34b3951a64d27f1458` |
| `src/training/checkpointing.py` | `ab9c8a7f3753edae0acd83ca390def5a0c7c626a17c986c86473f0b8707e26d4` |
| `src/training/run.py` | `9448834e735cf079e644ad0fd12dc923ba3420319ae4730b787e5e21e59239a6` |
| `src/training/steps.py` | `22029cc8cd604f7b209cd2c8d4bad2b9f89b08b39c3426f39bf0d59fdc11dab4` |
| `tests/test_matformer_widths_campaign.py` | `7998b52b679abc4ea259cc69e5c4654d85b3f75aec6c4122400300eea4884d7a` |
| `tests/test_matformer_widths_queue.py` | `68b22b138fef6991796e095e0c26d1021d8c35c8191a7a330337997707a7ba01` |
| `tests/test_matformer_widths_reporting.py` | `78f70c2a003e2f77cd6f4b02c3ea9cf61199c6533f9ffa7de8bb84d1d57ca60e` |
| `tests/test_optimizer_ownership_reporting.py` | `5341e2c991f85d38082cdd4abafd82738b72be74ef31aa93fd66826d1d3be5de` |

### Final regression result

Executed from the recorded repository cwd with the pinned Python environment:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_matformer_widths_campaign.py tests/test_matformer_widths_reporting.py tests/test_matformer_widths_queue.py tests/test_optimizer_ownership.py tests/test_optimizer_ownership_resume.py tests/test_optimizer_ownership_campaign.py tests/test_optimizer_ownership_reporting.py tests/test_optimizer_ownership_corrections.py tests/test_inverse_membership_sampling.py tests/test_inverse_membership_reporting.py tests/test_inverse_membership_queue.py tests/test_config.py tests/test_per_granularity_optimizer.py tests/test_per_granularity_optimizer_resume.py tests/test_metrics_compact_accounting.py tests/test_metrics_history_performance.py tests/test_global_sampling_windows.py tests/test_reproducibility.py tests/test_model_size.py tests/test_packed_corpus.py tests/test_train_cli.py tests/test_training_smoke.py -q -rs --tb=short
```

Result: **1,302 passed, 39 skipped, 1 xfailed, 2 warnings in 286.40s**, exit 0.
Log: `/tmp/mw-phase5-complete-validation.txt` (temporary local evidence).
Five skips are the new separately authorized GPU semantic probes; 34 are existing
CUDA/bf16 ownership checks. The expected failure is the pre-existing general
`test_interrupted_and_relaunched_run_preserves_the_same_output_dir` marker in
`tests/test_training_smoke.py`; all nine-arm campaign resume checks passed.
Warnings concern third-party SWIG types. This command combines all three new
suites and every relevant regression suite listed in quickstart.md.

An earlier broad run caught a legacy corrected-report CSV naming regression;
that path was restored and the complete final command above passes. CLI help for
the diagnostic runner, operational runner and new analyzer command succeeds.
`git diff --check` passes. All twelve recipe/source/test hashes above match the
files used for this final command.

| Tasks | Implementation acceptance | Status |
| --- | --- | --- |
| T022–T026 | Nine-arm restore/failure/rollback/zero-step recovery and attempt-resource tests | Complete on CPU |
| T027–T033 | Snapshot gates, preparation, strict barrier, queue/worker and report-recovery tests | Complete on CPU; actual operational gates not run |
| T038–T044 | Strict 24-row exports, nine diagnostics, C1/C3 integration and corruption/atomicity tests | Complete on CPU fixtures |
| T046–T051 | Four-only history, 28-row identities, four figures and input/legacy compatibility tests | Complete on CPU fixtures |
| T053–T055 | Full regression, reconciled interfaces/runbook and constitution/FR/SC review | Complete for implementation admission |
| T034–T037 | Real-input preparation, CPU/GPU gates, standalone-first production | Pending separate execution authorization |
| T045/T052 | Actual 24/28-endpoint production reports | Pending production and selected-history evidence |
| T056–T057 | Saved-result interpretation and final campaign reconciliation | Pending actual artifacts |


## Authorized operational continuation (2026-09-21)

Current authorization covers all T034–T037 execution preparation, diagnostics,
submissions and own-run resumptions. It persists across job-completion handoffs.
No further authorization is required for those actions. T045/T052 are distinct
actual reporting tasks and remain pending.

### Real-input preflight

The proposed root was absent before this invocation and is now exclusively
reserved:
`/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1`.
All earlier campaign roots and artifacts are preserved. Command (repository cwd,
pinned interpreter, `OMP_NUM_THREADS=1`), exit 0:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python scripts/analyze_tinystories_optimizer_ownership.py preflight --campaign configs/controlled_exps/tinystories_instruct_matformer_widths.yaml --prepared-corpus-dir /nfs-stor/ivo.navarrete/matformer-corpora/tinystories-instruct-packed-full-v1 --tokenizer-dir /nfs-stor/ivo.navarrete/matformer-tokenizers/tinystories-instruct-sentencepiece-bpe-2k-v1 --output-dir /nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1/campaign --run-output-root /nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1/runs
```

The audit verified all 89 shards and 5,576,491 training-order entries, designated
membership 5,576,448 with the fixed excluded tail of 43, vocabulary 2048, and
all pinned corpus/tokenizer/order/four-role hashes. Reserved-role pairwise
intersections are zero; source role counts are optimizer training 2,476,404,
ordinary validation 128, controller 128, final holdout 512. These are source
document counts, distinct from packed evaluation sequence counts. All nine actual
model definitions match the required 90,688/115,264/164,416/262,720 endpoint counts.
Expected first-epoch batch hashes agree across all nine; full elastic action and
four-epoch batch hashes agree across all five. `holdout_evaluated=false` and
`training_started=false` in preflight; the audit does not evaluate the holdout.

| Artifact/identity | SHA-256 |
| --- | --- |
| `campaign/campaign_manifest.json` file | `a12fcb2be0450be59a38af185ec56a27b74efcef26edbdfb193952f9977a403e` |
| Manifest content hash | `bdba3e1cedd6bdc3996425b64da97c71a17d8832c4406fcc3f41c7c37ce48b27` |
| `campaign/preflight.json` file | `158b90b3bbf62d6ed5ba0b1dc3d9b37b2ed7b1a1892a6da70b740b92e3df4de8` |
| Corrected source set | `146282e8ed36a4799b7a583731649d6c3d6a2585e699e902d18733aa10a0f1a8` |

### Operational defects found before admission

The initial actual snapshot check failed because fixtures created default
`outputs/` in the immutable source directory. It was interrupted after 356
failures, 210 passes and 33 skips (193.11s); none establish readiness. Its log,
JUnit and failed gate remain in
`diagnostics/cpu-c24ed7db7b9241bfb5e55ef676874a34/`. A second check accidentally
started against the obsolete snapshot after NFS rejected an archive rename;
it was stopped after 38 passes (11.66s), with all evidence retained in
`diagnostics/cpu-df970a4fac9f423db9fb8ff6d7e6db41/`. The obsolete snapshot and its
manifest are retained in `diagnostics/superseded-before-operational-fixes/`.
Only its top directory write bit was temporarily enabled for the NFS rename,
then restored; source file contents were not rewritten.

The corrected runner uses a writable diagnostic workspace with links to immutable
snapshot inputs, an explicit snapshot PYTHONPATH, and separate outputs. The
source hash is revalidated after checks. The live scheduler initially had no
user QoS usage row, and this host lacks `sacctmgr`; the launcher now falls back
to a small read-only query compiled against installed Slurm headers/libslurm.
It reads configured QoS limits and all applicable user/ancestor association
limits, rejecting missing/unknown enforcement or ancestry. This changes no
scientific controls, checkpoint semantics, or training source.

Targeted operational regression: `OMP_NUM_THREADS=1
/home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest
tests/test_matformer_widths_queue.py -k 'live_scheduler or idle_user or
pytest_evidence' -q --tb=short`: **10 passed, 42 deselected**, exit 0, 7.34s.
The corrected full snapshot CPU check is tracked below when complete.

Live Slurm 25.05.6 audit: `AccountingStorageEnforce=associations,limits,nosteps,qos`;
`cscc-gpu-p` permits `cscc-gpu-qos`; its configured ceilings are MaxJobsPU=2 and
MaxSubmitJobsPU=4. User association 476 → account 411 → root 1 adds no stricter
finite job/submission limits. The user-wide queue was empty. QoS MaxWallPJ is
2880 minutes; requests retain 30 minutes for diagnostics and 24 hours for
production. All submissions must re-query these values and exclude gpu-[05,50,51],
with one task and one GPU. The fallback was checked against the live controller
successfully before snapshot freezing. No scheduler limits were changed.

### Passed CPU gate and preparation (T034 complete)

The corrected snapshot CPU command, exit 0:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python scripts/preflight_tinystories_matformer_widths.py --mode cpu --campaign-root /nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1
```

Result: **1,307 passed, 39 GPU-only skips, 1 existing xfailed, 2 warnings in
276.16s**, exit 0. JUnit represents the expected failure as a skipped case, so the
gate records 1,347 tests and 40 skipped with zero failures/errors. This is CPU
readiness only. The gate records the exact complete acceptance/regression argv,
snapshot source path, writable workspace cwd, environment versions and result
hashes. Passed evidence: `diagnostics/cpu-abfb2f8137524a5b9c80d1a402858b7d/`.

| Artifact/identity | SHA-256 |
| --- | --- |
| Config set | `3355637f58d6455a33d5a25e074ffc23917d3149226da021a841ac44d6834f4a` |
| `diagnostics/cpu-gate.json` file | `fb0902458a768534947dfd4b2c8c0dc79e9743088b9691b887e87d8b9c72d348` |
| CPU gate content hash | `740c7f3274ddb076ea9e2a7586b452b87f17082411f7608bdc3d9dc62997e2fa` |
| Passed `pytest.log` | `29d0b8e5e17abcb8c0d9a694f2ed8e04f2430a74dec535b3bf36c231c2d7215c` |
| Passed `pytest.xml` | `766908820a518ddfa73a213c52903a050edccc928c95f7abf63240b781f4f86e` |
| `launchers/plan.json` file | `aa38163ac73294e18052d2bf0ce4596de02c4dc7a076748be086b146f264195c` |
| Launch plan content hash | `c39d817e353ea1012a125d599f8f1262ca591385432be0b7fe3fe18daa730c05` |

Preparation used the snapshot CLI, exit 0:

```bash
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python /nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1/source/scripts/run_tinystories_matformer_widths.py prepare --campaign-root /nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1 --cpu-evidence /nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1/diagnostics/cpu-gate.json --reference-manifest /nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/reports/complete-20260910T181951Z/frozen/frozen_manifest.json
```

It validated the reservation, source/config/preflight binding and executed CPU
evidence. The historical reference is `available_unvalidated`; no T052 historical
terminal audit is implied. Production runs remain fresh and empty. Filesystem
availability at preparation was 88 TiB on NFS and 1.1 TiB on `/tmp` (filesystem
free space, not a quota guarantee).

### GPU diagnostic submission record (initial observation; superseded below)

**Job 271285**, name `mw-v1-gpu-diagnostic-a1`, is confirmed **RUNNING on gpu-52**.
Slurm reports submission/start 2026-09-21T18:29:59, one node/task/GPU, four CPUs,
16 GiB, 30-minute time limit, `cscc-gpu-p` / `cscc-gpu-qos`, and exclusions
`gpu-[05,50-51]`. The recorded observation had elapsed 15 seconds. This is not
GPU readiness or production completion. No production jobs or barrier exist.

Before sbatch, the initial accounting query failed with connection refused at
`localhost:6819`; **no intent or sbatch had been issued**. The controller's
exported config pointed accounting at its own localhost. A campaign-local
`launchers/slurm-client.conf` changes only `AccountingStorageHost` to `ciai-head`,
whose authenticated accounting endpoint is reachable. `sacct` then succeeded,
confirming no earlier job with this diagnostic name. No system/server config or
scheduler policy was changed. Original export, corrected file and hashes are in
`launchers/slurm-controller-export.conf` and `launchers/slurm-client-config.json`.
**Set SLURM_CONF to this campaign-local file for subsequent accounting and queue
commands from this host.** It was exported to the diagnostic submission.

The reviewed submission driver is `launchers/submit-diagnostic-1.py` (outside the
immutable training snapshot). It validates the prepared CPU gate, re-queries live
limits/user-wide queue, checks historical duplicate identity, writes its durable
intent before sbatch, and records the returned ID. Do not rerun it as a retry.
The durable record `launchers/diagnostic-submission-1.json` contains exact command,
source/config/preflight and CPU-gate bindings, live admission limits, job ID,
`scontrol` allocation and `sacct` observation. Exact sbatch command:

```bash
sbatch --parsable --job-name=mw-v1-gpu-diagnostic-a1 --partition=cscc-gpu-p --qos=cscc-gpu-qos --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:30:00 '--exclude=gpu-[05,50,51]' --no-requeue --chdir=/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1/source --output=/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1/logs/diagnostic-a1-%j.out --error=/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1/logs/diagnostic-a1-%j.err '--wrap=env OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR=/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1/diagnostics/matplotlib /home/ivo.navarrete/.conda/envs/elasticnn/bin/python /nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1/source/scripts/preflight_tinystories_matformer_widths.py --mode gpu --campaign-root /nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1'
```

Artifacts under the campaign root:

- `logs/diagnostic-a1-271285.out` and `.err`: scheduler process logs.
- `diagnostics/gpu-*/`: distinct real-shape probes, checkpoints, real-shapes.json,
  semantic/boundary/failure pytest log and JUnit, and attempt gate.
- `diagnostics/gpu-gate.json`: canonical result when the runner publishes it;
  absent at the recorded RUNNING observation.
- `campaign/`, `source/`, `diagnostics/cpu-gate.json`, `launchers/plan.json`:
  the prepared and immutable inputs. Do not alter or relabel their bindings.

On the user's job-completion notification, continue under the existing execution
authorization. Notification alone is not evidence of success. Exact next steps:

```bash
export MW_ROOT=/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1
export MW_PY=/home/ivo.navarrete/.conda/envs/elasticnn/bin/python
export SLURM_CONF="$MW_ROOT/launchers/slurm-client.conf"
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
squeue --noheader --user=ivo.navarrete --format='%i|%j|%T|%q'
sacct -X --noheader --parsable2 --jobs=271285 --format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList
"$MW_PY" - <<'PYVERIFY'
import os, sys
from pathlib import Path
root = Path(os.environ['MW_ROOT'])
sys.path.insert(0, str(root/'source'))
from scripts import run_tinystories_matformer_widths as ops
ops.verify_plan(root)  # Rechecks CPU and GPU gates, result hashes and bindings.
print('Exact snapshot/config/preflight CPU and GPU gates validate')
PYVERIFY
```

Require actual terminal scheduler success, no skipped GPU checks, all nine
real-shape bf16 probes, and valid hashed results before checking T035. Inspect
logs and attempt artifacts even if Slurm reports COMPLETED. On failure, preserve
the diagnostic intent/attempt, determine the cause, reconcile the known job, and
create a distinct diagnostic attempt only after valid repair/reverification;
source changes require a new snapshot and readiness gates, never hash relabeling.

After T035 passes, run the verified snapshot queue once:

```bash
"$MW_PY" "$MW_ROOT/source/scripts/run_tinystories_matformer_widths.py" queue --campaign-root "$MW_ROOT" --once
```

Record all resulting standalone IDs and stage evidence, then hand off if waiting.
Reconcile actual scheduler states and valid own-run checkpoints on each return.
T036 requires four strictly validated 87,132-update terminals and a published,
revalidated `launchers/standalone-barrier.json`. Only then may the queue admit
T037's five 348,528-update elastics. T037 additionally requires complete trace,
exposure, C1/C3 clipping and attempt-cost reconciliation, with 17,130,848,256
assigned production tokens distinguished from diagnostics/replay. At this initial observation T035–T037 were unchecked. The completed diagnostic
and subsequent production handoff below supersede it; T045/T052 remain separate.

### Actual GPU verification (T035 complete)

The diagnostic finished during preparation of the initial handoff. Verified
`sacct` result: `271285|COMPLETED|0:0|86|gpu-52` (86 allocation seconds).
Hardware: **NVIDIA A100-SXM4-40GB**. The gate records **nine real-corpus,
batch-64/context-128 bf16 probes**, each stopped at four updates and resumed from
its own diagnostic checkpoint to eight, preserving its full scheduler horizon.
All nine original definitions are present. Separate semantic, synthetic epoch-
boundary, malformed restore and partial-failure pytest checks: **118 passed,
91 deselected, zero skips, one pytest import-rewrite warning, 47.24s**, exit 0.
The deliberate deselections are the GPU producer's documented test selection;
no selected GPU check skipped. Actual outputs are under
`diagnostics/gpu-c22ac3b6faba4c8f843ea6fb5b446f9a/`.

`verify_plan(root)` and `verify_gate(root, 'gpu')` succeeded against the exact
prepared snapshot/config-set/preflight and CPU gate, including all **50 hashed
result references**, no skipped checks, all nine arms, job identity and hardware.
`holdout_evaluated=false`. `launchers/diagnostic-submission-1.json` now records
completed status, final scheduler accounting and the verified GPU gate hash.

- GPU gate file SHA-256:
  `0c7c1e7cb8581c248662c7da15f5b7fb02c300727b4f310758a10da1e2d09b1e`.
- GPU gate content hash:
  `683e5f6e35d205e4740dfc2da7b362ca46dbedac112140824cb2fe7a738bdd29`.
- All diagnostic checkpoints remain separate from fresh production identities.

The authorized snapshot `queue --once` command admitted only the four
standalones, with live user-wide limits and durable intents, then exited 0. Production completion
is not implied by passed diagnostics; the final job handoff is recorded below.

### Current handoff: standalone production submitted (T036 incomplete)

This is the **current stage**, superseding the initial diagnostic handoff above.
T034 and T035 are complete. The snapshot queue admission cycle exited 0:

```bash
SLURM_CONF=/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1/launchers/slurm-client.conf OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python /nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1/source/scripts/run_tinystories_matformer_widths.py queue --campaign-root /nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1 --once
```

| Arm | Job ID | Attempt | Verified scheduler state | Node |
| --- | --- | --- | --- | --- |
| ST-g125 | 271319 | 1 | RUNNING | gpu-52 |
| ST-g250 | 271320 | 1 | RUNNING | gpu-52 |
| ST-g500 | 271321 | 1 | PENDING | Not assigned |
| ST-g1000 | 271322 | 1 | PENDING | Not assigned |

Each run is fresh, seed 42, with its full one-epoch assignment of 87,132 updates /
713,785,344 tokens. Each job requests one node/task/GPU, four CPUs, 16 GiB,
24 hours, cscc-gpu-p/cscc-gpu-qos, no automatic requeue and exclusions
gpu-[05,50,51]. The two running jobs occupy distinct single-GPU allocations on
the same allowed node. At the recorded live check, exactly two user jobs were
running and four submitted; no stricter finite association ceiling applied.
The running jobs' stderr files were empty. No terminal success is inferred.

Durable records and artifacts under `$MW_ROOT`:

- `launchers/submissions.json`: all four unique intents, exact sbatch commands,
  fresh continuation decisions, attempt IDs, job IDs and CPU/GPU/source bindings.
- `launchers/status.json`: completed arms empty and assigned-horizon progress.
- `launchers/handoff-standalones.json`: live queue, limits, accounting rows and
  observation timestamp at this handoff.
- `launchers/worker-<arm>-1.json`: worker process/job identity and eventual result.
- `logs/<arm>-a1-<jobid>.{out,err}`: each Slurm attempt's logs.
- `runs/<arm>/`: own checkpoints, heartbeats, scalar metrics, traces,
  resource_attempts.json, and eventual terminal sidecars.
- `launchers/standalone-barrier.json` is **absent**. No elastic job is submitted.

**Exact next steps after the user's completion notification:**

```bash
export MW_ROOT=/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1
export MW_PY=/home/ivo.navarrete/.conda/envs/elasticnn/bin/python
export SLURM_CONF="$MW_ROOT/launchers/slurm-client.conf"
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
squeue --noheader --user=ivo.navarrete --format='%i|%j|%T|%q'
sacct -X --noheader --parsable2 --jobs=271319,271320,271321,271322 --format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList
"$MW_PY" "$MW_ROOT/source/scripts/run_tinystories_matformer_widths.py" queue --campaign-root "$MW_ROOT" --once
```

First inspect actual scheduler outcomes, stdout/stderr, worker records and saved
artifacts. The queue revalidates exact gates and reconciles known intents; it
strictly inspects any terminal before accepting completion and validates an
own-run checkpoint before retry. An uncertain submission, ambiguous/invalid
state, or ended attempt without a durable checkpoint must be reconciled explicitly;
do not bypass it with a fresh restart. Preserve failed/replayed resource costs.

When all four complete terminals validate, this same queue cycle publishes and
revalidates the standalone barrier before admitting eligible elastic jobs under
live limits. Inspect and record the barrier and all four terminal/ordinary-
evaluation/trace/resource evidence before marking T036 complete. T037 remains
pending until all five full-budget elastic terminals and actual trace/exposure/
C1-C3 clipping/attempt-cost totals reconcile. Record new IDs/artifact locations
and another handoff whenever further progress awaits jobs.

The current turn stops while the four standalone jobs run/queue. **T036 and T037
remain unchecked. T045 and T052 actual 24/28-endpoint reporting remain separate,
pending production; no report or phase-5 completion is claimed.** Existing
uncommitted work and historical artifacts are preserved. Zero holdout evaluations.

Final startup observation: both running standalone logs reached committed step
1,580/87,132 (12,943,360 assigned tokens) with empty stderr. This confirms actual
training began, not terminal completion. No elastic admission occurred.
