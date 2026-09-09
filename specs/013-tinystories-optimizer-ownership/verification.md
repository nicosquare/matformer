# Phase 8 verification — 2026-09-09

**GPU follow-up passed:** Slurm job 220964 on gpu-52 completed all 28 CUDA bf16
checks with no skips. See [the follow-up record](#slurm-gpu-follow-up--2026-09-09)
for the test setup correction, measurements and shared evidence location. The
initial login-node limitations recorded below are retained as historical evidence.

Scope: T055–T058, on branch `013-tinystories-optimizer-ownership`, based on
`341e5de` plus the Phase 8 test/documentation changes. Full campaign configurations
and runtime implementation are unchanged. No full-budget training or real
holdout model evaluation was performed.

## Environment and commands

Python: `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python`, version 3.12.13.
Installed versions: PyTorch 2.11.0+cu128, CUDA runtime 12.8, Transformers 5.8.0,
datasets 4.8.5, PyYAML 6.0.3, NumPy 2.4.3, pandas 3.0.2, Matplotlib 3.10.9,
pytest 9.0.3. No dependency changes. `.gitignore` already covers the actual Python
project, generated outputs and editor files; no additional ignore file applies.

The complete requested suite command was:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest \
  tests/test_optimizer_ownership.py tests/test_optimizer_ownership_resume.py \
  tests/test_optimizer_ownership_campaign.py tests/test_optimizer_ownership_reporting.py \
  tests/test_per_granularity_optimizer.py tests/test_per_granularity_optimizer_resume.py \
  tests/test_config.py tests/test_train_cli.py tests/test_model_size.py \
  tests/test_packed_corpus.py tests/test_training_smoke.py tests/test_artifacts.py \
  tests/test_reporting.py tests/test_reproducibility.py tests/test_distributed.py \
  tests/test_distributed_sampling.py tests/test_global_sampling_windows.py \
  -q --disable-warnings --tb=short \
  --basetemp=/scratch/ivo.navarrete/tmp/optimizer-ownership-phase8-cpu-20260909
```

Initial result: **976 passed, 1 xfailed, 4 failed**, two dependency warnings,
141.42 seconds. Log: `/tmp/optimizer-ownership-phase8-cpu.log`. All four failures
were Gloo process-group initialization: the sandbox denied loopback sockets with
`Operation not permitted`. They occurred before distributed assertions ran.
The existing non-strict expected failure is
`test_interrupted_and_relaunched_run_preserves_the_same_output_dir` in
`tests/test_training_smoke.py`; it is not introduced by this feature.

The four blocked cases were rerun outside the sandbox:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest \
  tests/test_distributed.py -k real_two_process -q --disable-warnings --tb=short \
  --basetemp=/scratch/ivo.navarrete/tmp/optimizer-ownership-phase8-gloo-20260909
```

Result: **4 passed, 22 deselected**, 30.43 seconds.
Log: `/tmp/optimizer-ownership-phase8-gloo.log`. Controller commit, terminal
validation consensus and successful/failing gradient-journal publication agree
across two real Gloo processes. The original suite therefore has **980 passing
cases and one existing expected failure** across the initial run and retry.

After adding the legacy analyzer regression and device variants, rerun all four
affected ownership suites:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest \
  tests/test_optimizer_ownership.py tests/test_optimizer_ownership_resume.py \
  tests/test_optimizer_ownership_campaign.py tests/test_optimizer_ownership_reporting.py \
  -q -rs --disable-warnings --tb=short \
  --basetemp=/scratch/ivo.navarrete/tmp/optimizer-ownership-phase8-focused-20260909
```

Final focused result: **322 passed, 28 skipped**, two dependency warnings, in
85.31 seconds. Log: `/tmp/optimizer-ownership-phase8-focused.log`. All skips are
explicit CUDA-unavailable cases. The preliminary targeted
run passed **30 cases**, with **28 CUDA skips**, in 14.19 seconds; log:
`/tmp/optimizer-ownership-phase8-device.log`. It directly verifies the two new
historical endpoint cases and all CPU packed-resume cases.

## GPU diagnostic and measured limits

Both sandboxed and outside-sandbox probes returned `torch.cuda.is_available() ==
False` and `torch.cuda.device_count() == 0`. NVML could not initialize; no
`/dev/nvidia*` nodes were visible and `nvidia-smi` was unavailable. No GPU model,
bf16 result or allocated/reserved GPU peak can be reported from this environment.

The following runnable test variants use the existing real-model trainer and
eight-update synthetic budgets. Their CUDA cases skip explicitly when CUDA or
bf16 support is absent:

- `test_real_trainer_orders_owner_calls_then_clock_and_publishes_complete_updates[cuda]`
  resolves bf16 through the normal runtime helper, constructs the model on device
  before optimizers, verifies bf16 LM-head output, ordered active owner calls,
  one clock advance, finite weights/histories/group norms, separate caps and the
  combined sqrt(N) bound. It connects the real resource observer, checks eight
  attempted updates and finalized measurements, asserts positive allocated and
  reserved CUDA peaks, and saves `cuda_bf16_diagnostic.json` plus the ledger.
- `test_repeating_packed_sampler_exact_batches_actions_rng_and_state[cuda-*]`
  covers all nine arms before, at and after a two-update epoch boundary. It
  compares uninterrupted versus resumed batches, actions, RNG, model, optimizer,
  scheduler and sampler state. CPU versions use exact tensor equality; the CUDA
  variants retain those checks rather than weakening tolerances without evidence.

The [quickstart GPU command](quickstart.md#1-environment-and-implementation-checks)
selects these 28 cases on one visible GPU. **T057's unavailable-GPU reporting path
is complete; successful GPU execution remains an environment-dependent follow-up.**
No GPU performance or memory claim is supported by skipped tests.

## Saved evidence and interpretation limits

The real Phase 3 `preflight.json` and `campaign_manifest.json` still exist at
`/scratch/ivo.navarrete/tmp/optimizer-ownership-phase3-verified-20260909/`.
Their audited counts, hashes, exclusion and full precomputed action/epoch digests
are recorded in the [runbook](../../docs/tinystories-optimizer-ownership-experiment.md#phase-3-verification--2026-09-09).
This is prior preflight evidence, not a newly generated final-revision manifest.
Generate a fresh manifest for the final implementation before authorized training.

The final focused `--basetemp` contains real short-run checkpoints, terminal
sidecars, trace/clipping/ledger records, complete and partial synthetic report
fixtures, and the historical Feature 12 reports. Important fixture directories:

- `test_complete_freeze_tables_an0`: complete frozen manifest, matching 24-row
  CSV/JSON tables, nine individual reports and four combined PNG/PDF files.
- `test_partial_requires_two_opt_0`: visibly partial three-endpoint C3 output
  with 21 omissions and independent freeze/report opt-ins.
- `test_terminal_reader_consumes_*`: all nine real eight-update arm artifacts
  accepted by the strict terminal reader with reduced test-only expectations.
- `test_historical_feature12_anal*`: six-run seed-42/43/44 reports retaining the
  old trailing-five ordinary-validation mean and preference for saved synthetic
  holdout results. No model evaluation or actual sealed holdout data is used.

Synthetic full-horizon metadata and coincident endpoint losses validate export
and rejection logic; they do not measure campaign learning. Short real models
establish runtime semantics at diagnostic budgets only. CPU measurements establish
actual state tensor storage and ledger arithmetic, not GPU peaks or campaign
throughput. The 37.5% C2 saving concerns FFN moments only. Resource totals sum
unique attempts' latest durations and take maximum peaks; hard-kill costs may be
incomplete. Seed matching does not establish cross-shape tensor identity or
multi-seed robustness, and aggregate token matching is not compute matching.

## Requirement reconciliation

All IDs below refer to [spec.md](spec.md). “Verified” means implemented behavior
tested at the indicated scale, not a completed full-budget scientific outcome.
The runbook's earlier dated records retain detailed numerical and artifact evidence.

| Requirements | Implemented checks and evidence | Remaining scientific observation |
| --- | --- | --- |
| FR-001, FR-002; EX-001, EX-002, EX-005 | Campaign expansion, horizon/alignment/control rejections; actual pinned audit and nine fresh definitions in Phase 3 | Nine full-budget terminal runs and aggregate consumed costs |
| FR-003, FR-004; EX-004 | Full precomputed uniform action/fixed-membership epoch digests; runtime-RNG/repeat-sampler fixture comparisons; exact resumed traces; global sampling suites | Matching full runtime streams and realized selections/quarter activations |
| FR-005; EX-003 | Corpus/tokenizer/role/hash/tail rejection tests; real 89-shard audit, designated 5,576,448 samples and fixed 43 exclusion | Re-audit before final-revision launch |
| FR-006, FR-007, FR-008 | S1 residual tail momentum/decay/counters; S2 full allocation and selected-history isolation; real FFN paths | Full-run learning/resource differences |
| FR-009, FR-010 | Absent concat quarters remain bitwise unchanged; present-zero AdamW updates; C2 4/3/2/1 lazy histories and restored absence | Full-run learning/resource differences |
| FR-011 | Complete disjoint five-owner topology, tied aliases, segment/common biases, malformed topology rejection, no eager state allocation | None for static ownership coverage |
| FR-012, FR-013; EX-011 | Independent active-owner clipping, zero/inactive cases, sqrt(2)–sqrt(5) bounds at absolute tolerance 2e-6; config gate for all five caps; C1/test-only global-C3 history/parameter parity at rtol 1e-6, atol 1e-7 | Full-run clipping frequencies; CUDA execution unavailable |
| FR-014; EX-008 | Ordered owner calls and one global clock, inactive LR synchronization, shared/per-width scheduler compatibility, complete schedule horizons | Full-run LR/clock reconciliation |
| FR-015; EX-006, EX-007, EX-009 | Normal seed-constructor/RNG isolation, real dimensions/counts, full matched recipe and changed-control rejection, no cross-shape initial equality claim | CUDA bf16 execution and completed seed-42 outcomes |
| FR-016 | All-arm staged restore, required/missing/impossible histories, model/RNG/clock/sampler/accounting corruption rejection, whole-bundle installation rollback | GPU restore execution unavailable |
| FR-017 | First/middle/last owner and later clock/accounting mutation failures poison state; all save paths preserve prior checkpoint hash; pre-mutation transaction recovery | No additional full-run evidence needed for injected failure paths |
| FR-018, FR-019 | Committed trace/scalar/summary reconciliation; separate selections/quarter activations/owner calls; nine real diagnostic artifact readers | Full-budget artifacts and final count reconciliation |
| FR-020; EX-010 | Actual dtype/component/owner tensor bytes, exact post-exposure moments, ledger replay/dedup/max/incomplete tests, terminal checkpoint size/hash; temporary buffers labeled estimates | GPU allocated/reserved peaks, full-run time and throughput |
| FR-021 | C1 global coefficient/C3 separate coefficients, active denominators and null inactive observations, clipping plots, RNG unchanged | Full-budget width-conditioned clipping statistics |
| FR-022, FR-023 | Weighted causal-target terminal evaluation, examples/targets, source-width labels, durable hash binding, zero-update recovery and idempotent reuse; poisoned holdout fixture ignored | Nine full-budget ordinary-validation sidecars; future holdout requires separate request |
| FR-024, FR-025, FR-026; EX-013 | Exact active counts 115264/164416/213568/262720, CSV/JSON parity, five connected four-point series and four disconnected markers, coincident labels, no seed bars; individual plots and four combined files | Actual 24 terminal measurements and figures |
| FR-027 | Full contract/budget/role/target/hash/source validation; duplicate/missing/nonfinite/stale/nonterminal rejection; atomic staging and explicit twice-opted-in partial labeling | Complete real frozen manifest |
| FR-028 | Four focused suites plus named compatibility suites, Feature 12 fixture, real Gloo retry, short GPU variants with explicit environment skips | CUDA execution only; full campaign is outside implementation scope |
| EX-012 | Six descriptive comparison records and documented intervention caveats: S1/S2, C1/C2, S1/C1, S2/C2, C1/C3, elastic/matching standalone | Measured learning/resource interpretations from actual campaign results |
| SC-001 | Valid nine-arm preflight and named invalid-control rejection verified | Fresh final-revision preflight before launch |
| SC-002 | Exact assigned budgets, full expected digests, short runtime/repeat-epoch reconciliation verified | Actual one/four-epoch completion and full runtime equality |
| SC-003 | All five elastic model/history/clipping semantics and C3 clock verified on CPU | CUDA bf16 execution unavailable |
| SC-004 | Exact CPU batch/action/RNG/model/history/scheduler resume and malformed-load non-mutation verified | CUDA counterpart unavailable |
| SC-005 | Nine-arm short-run artifacts, failure costs, terminal recovery and resource completeness verified | Full-budget artifacts and device measurements |
| SC-006, SC-007 | Complete 24-endpoint fixture, four combined files, source traceability, strict complete/partial rejection verified | Actual frozen campaign and 24 measurements |
| SC-008 | Six comparisons, descriptive seed-42 language, token-versus-compute/resource caveats verified on fixtures | Measured full-budget scientific answers |

## Final review

Phase 8 changes only tests and documentation. Historical analyzer code, legacy
signature functions and the immutable campaign YAML are unchanged. Compatibility
tests cover omitted shared defaults, per-width checkpoints, distributed shared
resolution, global sampling, fixed repeat epochs, model counts, CLI and reports.

Source review confirms that campaign updates snapshot transaction metadata/RNG/
sampler state, not full model or optimizer tensors. Full-bundle copies occur in
`_load_ownership_checkpoint` only at restore. Existing concat LMC
snapshots are correction-specific and the campaign disables those corrections.
No new hot-loop snapshot, runtime abstraction or dependency was added in Phase 8.

After the last test-only resource-observer wiring change, the affected runtime
case was checked separately:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest \
  tests/test_optimizer_ownership.py::test_real_trainer_orders_owner_calls_then_clock_and_publishes_complete_updates \
  -q -rs --disable-warnings --tb=short \
  --basetemp=/scratch/ivo.navarrete/tmp/optimizer-ownership-phase8-runtime-20260909
```

Result: **1 passed, 1 CUDA skip**, two dependency warnings, 6.19 seconds; log:
`/tmp/optimizer-ownership-phase8-runtime.log`. The CPU case exercised the real
observer and verified eight attempted updates plus finalized ledger accounting.

`git diff --check`, Python compilation of all three modified test files, local
file links in the runbook/quickstart/verification record, and explicit presence
of every FR/EX/SC identifier in the reconciliation all passed. Saved fixture
directory names were checked against the final focused output. GPU execution
and full-budget scientific observations remain explicitly unclaimed.

## Slurm GPU follow-up — 2026-09-09

User execution requirements: all outputs under `/nfs-stor/ivo.navarrete/results`,
and exclude `gpu-05`, `gpu-50`, `gpu-51`. The campaign root is
`/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1`; diagnostic
logs, XML and fixture artifacts are in its `diagnostics/` subdirectory. The
repository's existing `cscc-gpu-p` partition / `cscc-gpu-qos` QoS is retained.

| Job | Node | Result |
| --- | --- | --- |
| 220959 | gpu-51 | Failed in 8 seconds; logs targeted login-node-local scratch and were not accessible from the login node afterward |
| 220961 | gpu-51 | Shared logs captured CUDA driver initialization failure before pytest |
| 220962 | gpu-52 | A100 bf16 runtime/clipping test passed; 27 resume cases failed at the late strict-determinism setup guard |
| 220964 | gpu-52 | **28 passed**, 222 deselected, two dependency deprecation warnings, **18.22 seconds**; Slurm **COMPLETED**, exit **0:0**, allocation elapsed **24 seconds** |

The resume fixture previously called `configure_strict_determinism` after
`runtime_fixture` had initialized CUDA through capability checks/peak reset/model
placement. It also tried to configure determinism again for each restored bundle
in the same CUDA process. The shared test fixture now applies the normal strict
recipe before the first CUDA operation and verifies every strict setting on
context reuse. The late resume-fixture call is removed. A CPU regression test
clears the workspace setting and checks that determinism is established before
the device capability call. Production guards, training code, numeric equality
assertions and campaign configs are unchanged.

CPU regression command:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest \
  tests/test_optimizer_ownership.py tests/test_optimizer_ownership_resume.py \
  tests/test_optimizer_ownership_reporting.py -q -rs --tb=short
```

Result: **290 passed, 28 CUDA-unavailable skips**, two dependency warnings,
70.85 seconds. Log: `diagnostics/ownership-cuda-order-regression.log` under the
shared campaign root. The separate GPU job executes those 28 CUDA cases with no
skips. Actual CUDA checks retain exact model/history/RNG/action/batch comparisons
across all nine arms and all three synthetic epoch-boundary positions.

Job 220964 used an NVIDIA A100-SXM4-40GB, driver 570.195.03, PyTorch 2.11.0+cu128,
and verified `torch.bfloat16` LM-head output. Its eight-update C3 diagnostic
recorded **69,053,952 peak allocated bytes**, **90,177,536 peak reserved bytes**,
eight attempted updates, one finalized attempt and 1.1056 seconds of observer
elapsed time. These small-model diagnostics do not estimate full-campaign peaks
or throughput. The GPU case proves one-clock/owner order, finite independent
clipping and state, and positive measured allocated/reserved peaks.

Preserved files under the shared `diagnostics/` directory:

- `gpu-check-220964.out`, `.err`, `.xml`, and `gpu-verification-220964.json`.
- `gpu-tests-220964/test_real_trainer_orders_owner0/cuda_bf16_diagnostic.json`
  and its resource ledger; all 27 real resume fixture directories.
- Logs from failed jobs 220961 and 220962, plus the preliminary CPU regression.

The successful log SHA256 is
`b3a474eca247c07cef4ea2bd1a2f41cda749b624bd7df312a266cfeee9248386`.
It records base revision `4746bd1` and the uncommitted fixed test file hashes:
`da36fc42d1581f6347ac09adf8ea690d3501945b3e828150934814b482662ae0`
(`test_optimizer_ownership.py`) and
`bf3259089a9361f76815955af4e3cdc4fb2d53da6398b467b985baaa5f8daeed`
(`test_optimizer_ownership_resume.py`). Job 220964 was already running on allowed
node gpu-52 with the earlier output paths when the user restated the storage and
exclusion requirements. Evidence was copied intact after completion; original
embedded paths and source copies were retained for provenance. Subsequent jobs
use the corrected shared paths and all three exclusions directly.

Reusable submission, from the repository root:

```bash
export OO_BASE=/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1
mkdir -p "$OO_BASE/diagnostics"
sbatch scripts/slurm_optimizer_ownership_gpu_check.sh
```

The launcher requests one GPU, four CPUs, 16 GiB and 30 minutes, retains Slurm's
GPU visibility assignment and writes job-specific artifacts. This successful
follow-up closes the GPU-execution limitations for T057, FR-028 and SC-003/004 in
the earlier tables. Actual full-budget campaign observations remain future work;
no campaign training or sealed holdout evaluation was launched.
