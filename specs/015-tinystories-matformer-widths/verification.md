# Feature 015 verification record

## Scope and status

Authorized work: implementation phases 1–4, T001–T021, on 2026-09-21.
The current invocation adds phases 3–4 to the existing foundations.
No external campaign root is created/reserved by this work. No GPU jobs,
production runs or holdout evaluations are performed. Actual model counts are
verified on CPU. Pinned input hashes still require the separately pending
real-input preflight audit.

| Evidence stage | Status | Outstanding work |
| --- | --- | --- |
| Setup documents T001–T002 | Complete | Reconcile as later evidence arrives |
| Foundations T003–T005 | Complete | Focused CPU evidence below; end-to-end schema-4 resolution verified in phase 3 |
| Actual nine-model counts | CPU verified | Real-input campaign preflight remains T034 |
| Phase 3 protocol/preflight T006–T013 | Complete on CPU | Real-input audit remains T034 |
| Phase 4 ownership/accounting T014–T021 | Complete on CPU | GPU diagnostics and terminal-reader integration remain pending |
| Real corpus/tokenizer/role audit | Pending | T034 |
| Historical four-terminal audit | Pending | T052 |
| Full snapshot-bound CPU gate | Pending | T027/T034/T053 |
| GPU gate | Pending, not run | T035 and separate authorization |
| Standalone barrier | Pending | Four validated production terminals |
| Nine-run production | Pending | T036–T037 |
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
- Baseline source revision: `82fbc273879952b179a9eafa73782d7c45d5284a`; local implementation changes are uncommitted. Tested file hashes are recorded below.
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

Full campaign acceptance remains pending. Phases 3–4 establish CPU evidence for
definitions, topology, update semantics, clipping, and compact accounting;
production, GPU gates, and subsequent stories remain pending.

| Requirement | Required evidence | Status |
| --- | --- | --- |
| FR-001 | Distinct nine-arm schema; unchanged legacy contracts and runtime | CPU definitions and legacy signatures verified; production pending |
| FR-002 | Nine fresh seed-42 initializations, own-run continuation only | Fresh CPU construction verified; full continuation acceptance remains phase 5 |
| FR-003 | Strict standalone-first barrier including restart/worker entry | Pending |
| FR-004 | Uniform global replacement H=1, unweighted causal updates | CPU uniform replacement H=1 verified |
| FR-005 | Identical action and designated epoch-batch traces | Full expected synthetic-order digests verified; real-input audit pending |
| FR-006 | Hashed physical grid/boundaries and actual model support | CPU geometry, counts, support and hashed topology verified |
| FR-007 | S1/S2 full-tensor history/update semantics | CPU all-width/full-history semantics verified |
| FR-008 | Inactive concat absence and C2 lazy support multiplicities | CPU absent/zero gradients and lazy histories verified |
| FR-009 | Five disjoint C3 owners including ties/common bias | CPU five owners, tied objects and optional biases verified |
| FR-010 | Real cap-1 clipping and single scheduler advance | CPU real clipping/order/clock verified; GPU pending |
| FR-011 | Strict controls/data/count/budget/identity preflight | Fixture preflight/publication/rejections verified; real audit pending |
| FR-012 | Complete restore validation before live mutation | Pending |
| FR-013 | Epoch-boundary equivalence and durable partial-failure rejection | Pending |
| FR-014 | Actual configs/provenance/traces/scalars/diagnostics | Pending |
| FR-015 | Real C1/C3 sidecars and strict reader rejection cases | CPU C1/C3 sidecars and inspection verified; terminal-reader integration remains T040/T044 |
| FR-016 | Measured unequal-block allocations and complete attempt resources | CPU allocations/exposure verified; GPU peaks and production attempt costs pending |
| FR-017 | Source/config-bound CPU/GPU readiness and compatibility | Pending; legacy fixtures in scope |
| FR-018 | Authorized fresh root, sbatch limits, restart-safe admission | Pending |
| FR-019 | Full-budget terminal ordinary validation and zero-step recovery | Pending |
| FR-020 | Matching 24/28 CSV/JSON with physical/run identity | Pending |
| FR-021 | Four-only historical standalone revalidation | Pending |
| FR-022 | Strict missing/corrupt/nonterminal input rejection | Pending |
| FR-023 | Two combined figures in PNG/PDF | Pending |
| FR-024 | Exact coincident coordinates and distinguishable standalone legends | Pending |
| FR-025 | Descriptive saved-results interpretation | Pending |
| FR-026 | Evidence-aligned tasks/runbook/statuses | Initial record complete; later evidence pending |

## Success criterion acceptance

| Criterion | Required evidence | Status |
| --- | --- | --- |
| SC-001 | Nine actual definitions/counts and rejection cases | CPU acceptance verified; real-input audit pending |
| SC-002 | Zero production elastic starts before validated standalones | Pending |
| SC-003 | All-width ownership/history/clipping/update semantics | CPU acceptance verified; GPU verification pending |
| SC-004 | Exact future actions/batches, numerical resume, atomic rejection | Pending |
| SC-005 | Nine full-budget terminals and reconciled complete traces | Pending |
| SC-006 | Traceable run artifacts, resource disclosures and C1/C3 clips | Pending |
| SC-007 | 24/28 actual valid endpoints and corruption fixtures | Pending |
| SC-008 | Four combined figure files with five curves/eight markers | Pending |
| SC-009 | Saved-artifact scientific comparison and interpretation | Pending |
| SC-010 | Full legacy compatibility, sealed holdout and evidence-aligned status | Pending; signature fixtures in scope |

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
Phases 5–8, production, and the 24/28-endpoint actual reports are unexecuted.

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
