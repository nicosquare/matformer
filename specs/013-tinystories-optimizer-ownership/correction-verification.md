# Correction extension verification — 2026-09-10

Campaign root: `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/campaigns/concat-gmc-lmc-v1`.
All paths below are relative to that root unless explicitly repository paths.
Source base: `cbcd567`; additional working-source hashes are pinned in the campaign
manifest and launch plan. Original results, configs and scientific hashes remain
unchanged. No holdout was evaluated.

## CPU evidence

- Initial real-model regression reproduced the C3 eligibility rejection after C1
  and C2 passed. The subsequent reference test covers all 6 × 4 widths × 2 bias
  cases (48), starts full-width to populate moments, and compares actual training
  paths to independently scaled gradients, independently clipped groups and
  explicit per-parameter LR AdamW. Both decay and common bias are covered.
- `diagnostics/cpu-05.log` / `.xml`: **758 passed, 34 CUDA skips, one existing
  expected failure** across correction, LMC, ownership/resume/campaign/reporting,
  config, training smoke, per-width optimizer/resume and CLI tests.
- New report acceptance exercises strict schema-2 freeze, schema-1 references,
  40 matching CSV/JSON rows, nine concat curves, four standalone points, full
  progress coverage, six PNG/PDF outputs and rejection of replaced frozen sources.
- `diagnostics/cpu-queue-07.log`: **4 passed**, verifying two-job admission,
  other-user-job accounting, restart deduplication and stale GPU gate rejection.
- Intermediate failures are retained: resume fixture needed the independent
  training RNG seed; new arm names needed output-path acceptance and clipping
  artifact discovery. Test-only minimal contracts exposed a summary fixture
  dependency; the final implementation derives logging scope from resolved
  model/training controls. `cpu-final-11` reruns all affected cases on that fix.
- C3 failure injection covers all five owners, partial LMC correction, scheduler
  and accounting; unsafe saves preserve the previous checkpoint hash. Six-arm
  resume equivalence and correction-contract mismatch rejection are included.
- `git diff --check` and Python syntax checks passed before source freezing.

Use `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python` with
`OMP_NUM_THREADS=1`; JUnit records preserve the exact test cases and results.
GPU skips are not treated as evidence of GPU acceptance.

## Original references and paired controls

`diagnostics/reference-validation.json` records a strict read-only validation of
all nine original frozen terminal sources (24 endpoints) and saved resolved C1/
C2/C3 configs. Model, tokenizer/corpus, optimizer/scheduler, precision, batches,
budget and validation controls match the extension. The six expected action and
four-epoch order digests exactly match their corresponding original arms.
Initial campaign audit is preserved under `diagnostics/campaign-before-final-source`;
the final preflight refreshes only source provenance before any training launch.
The initial audit's embedded paths are historical and intentionally not rewritten.

## GPU and production gates

Pending final CPU completion and sbatch GPU preflight. No production submission
is authorized by a partial or failed gate. GPU evidence will identify actual
job/node/device, all six corrected and three none diagnostic measurements,
192→256 checkpoint restore, applied factors and recent throughput. Full runs and
terminal reports remain incomplete until explicitly recorded below.


Final affected CPU gate: `diagnostics/cpu-final-11.log` / `.xml`, **322 passed,
33 CUDA skips** in 90.15 seconds, including corrected comparison and partial-LMC
failure injection. The machine-readable `diagnostics/cpu-gate.json` binds this,
the 758-pass broad run and four queue checks to runtime source hash
`2333416e555d2018fe075543ffcaabff000c0f405688cdd9bd0ba49ba152897f`.
`source-v1/` contains the pinned source, tests, recipes and scripts; all 161 file
hashes plus base commit are in `launchers/source-provenance.json` and `plan.json`.
Final campaign preflight passed with matching source hashes. GPU preflight job
227577 submitted through sbatch with one GPU and required exclusions; acceptance
and production launch remain pending its result.


GPU preflight 227577 on gpu-03: **6 CUDA resume tests passed, zero skips**.
The real-shape harness then interrupted before metrics publication; its expected
step-192 checkpoint was not durable, and the save guard preserved the prior
checkpoint. This was a diagnostic stop-location error, not accepted GPU runtime
evidence. Logs and scheduler failure/50-second allocation record are retained.
The harness now stops after the normal validation checkpoint save. `source-v2/`
pins this change plus a tested scontrol/durable-worker accounting fallback because
the cluster's sacct database refused connections. Training runtime source/hash is
unchanged. `diagnostics/cpu-queue-12.log`: **6 passed**. Retry job **227584** runs
all GPU checks again; production remains gated.


GPU acceptance **227584**: COMPLETED / 0:0 on gpu-03, NVIDIA A100-SXM4-40GB.
All six CUDA exact-resume tests passed with zero skips. All nine real-shape
256-update diagnostics (none C1/C2/C3 plus six corrected) passed finite-loss/
parameter checks, actual factors, once-only LMC, checkpoint save and restored
192→256 progress. Full production horizons were retained in diagnostic configs.
Evidence: `diagnostics/gpu-gate.json`, `gpu-runtime-227584/measurements.json`,
`gpu-tests-227584.xml`, logs and `launchers/gpu-preflight-227584-slurm.txt`.

Recent median update seconds: C1-GMC .03350, C1-LMC .03383, C2-GMC .03395,
C2-LMC .03433, C3-GMC .03388, C3-LMC .03437. Same-job none C1/C2/C3 measured
.04723/.03237/.03421. Fixed sequential diagnostic ordering and short windows
limit causal overhead inference; C1's apparent speedup is not a reliable gain.
These are new measurements, not historical full-run throughput.

Production launcher started after both gates passed. `source-v3/` adds only
explicit no-requeue submission to the tested operational helper; training runtime
hash remains unchanged. Six queue/accounting tests pass in `cpu-queue-13`.
`launchers/plan.json` records final source/config hashes, gate identity and reference
manifest. `queue.pid`, `submissions.json`, `status.json`, `worker-<job>.json`, raw
Slurm records and GPU utilization logs preserve attempts and monitoring evidence.
The helper will validate/freeze/report after all six complete; production outcomes
are not yet claimed.

Initial production jobs: C1-GMC **227607** on gpu-03; C1-LMC **227608** on gpu-07.
Both are fresh and have durable checkpoints. An early 1024-update actual trace
prefix matches between them and original C1 (`diagnostics/production-prefix-check.json`).
At roughly 3200 updates recent intervals were 36.6/37.9 ms, implying about
3.5/3.6 hours remaining for these two at that moment; estimates are not completion
claims. The other four remain in the admission queue.

A monitor-only restart selected each job's allocated GPU instead of the last
GPU in node-wide inventory. `source-v4/` pins the tested operational change (7
queue/monitor tests passed in `cpu-queue-14`); source-v3 C1 trainers were not
restarted. All versions retain the identical CPU/GPU-validated training runtime.
`launchers/monitor-restart.json` and each submission's command preserve this.

On 2026-09-11 the user requested early submission of the next experiments.
The healthy helper had deliberately limited submissions to two. Controller
inspection verified `AccountingStorageEnforce=associations,limits,nosteps,qos`
and this user's `cscc-gpu-qos` limits `MaxJobsPU=2`, `MaxSubmitJobsPU=4`.
`source-v5/` now keeps four jobs submitted without dependencies and lets Slurm
enforce the two-running cap. It refreshes user-wide queue state before each
submission, counts newly accepted jobs even if squeue visibility lags, and defers
admission while another QoS is active outside the shared concurrency cap.
The required controller limits are rechecked before each new submission.

`diagnostics/cpu-queue-15.log` / `.xml`: **15 passed**, covering four-job
admission, slot refill, restart deduplication, delayed visibility, foreign-QoS
admission deferral, scheduler-limit rejection, accounting and GPU monitoring.
`launchers/monitor-restart-v5.json` records the helper-only restart, source hashes
and unchanged validated training runtime; neither C1 trainer was restarted.
Helper PID **2454479** submitted C2-GMC **227651** and C2-LMC **227652**.
Slurm confirmed two C1 jobs RUNNING and these two C2 jobs PENDING with
`QOSMaxJobsPerUserLimit`, no dependencies. Both C1 checkpoints were healthy near
12,800 updates. C3 jobs will be submitted as slots open, maintaining four total
submitted jobs. Early submission does not guarantee a GPU allocation time.

### C1 terminal audit — 2026-09-11

Both C1 jobs completed with Slurm exit 0:0: C1-GMC 227607 allocated 13,320
seconds; C1-LMC 227608 allocated 13,299 seconds. Each saved terminal evidence for
four epochs, 348,528 updates and 2,855,141,376 training tokens, with one fresh
attempt and no failed optimizer updates. The helper advanced C2 and submitted
C3-GMC 227930 / C3-LMC 227931 within the user-wide queue limits.

**Full artifact acceptance is blocked by missing clipping logs.** The actual
writer and scalar path field in `src/utils/metrics.py` still gate on arm names
`('C1', 'C3')`, excluding corrected identities. The earlier change to run-summary
artifact discovery did not fix these writer gates. Thus both C1 summaries name
`optimizer_ownership_clipping.jsonl`, but neither file exists. The same writer
gate also excludes corrected C3. Earlier CPU/GPU acceptance did not detect this
artifact omission; it does not establish complete campaign artifact acceptance.
Historical per-update clipping norms cannot be reconstructed from terminal
weights. Preserve this missing evidence explicitly; do not fabricate logs or
weaken the strict campaign freeze/report contract.

`diagnostics/validate_c1_terminals.py` records a separate read-only audit. Strict
validation must retain the missing-file failure. Supplemental validation checks
terminal ordinary-validation identities/hashes, finite checkpoint tensors and
AdamW moments, per-parameter counters against exposure, full action/epoch traces,
metrics and resource attempts. It excludes clipping only from a temporary
in-memory summary for the separate trace audit, never from saved run evidence.
The machine-readable outcome and command log are
`diagnostics/c1-terminal-validation.json` and `.log`. No training or holdout
evaluation is part of this audit. T071 remains incomplete.

The supplemental audit passed for both runs: all checkpoint model/optimizer
tensors finite, all 75 parameter-state shapes and exposure-dependent AdamW step
counters valid, and all 348,528 committed actions and four epoch-order digests
match preflight and frozen original C1. Terminal checkpoint hashes, all eight
ordinary-validation endpoints, finite saved metric losses and completed single
attempt resource ledgers passed. Strict artifact validation failed exactly on
the absent clipping file in each run.

| Width | Original C1 loss | C1-GMC loss | C1-LMC loss |
| --- | ---: | ---: | ---: |
| g250 | 2.073095 | 2.078093 | 2.071540 |
| g500 | 1.998167 | 2.004669 | 1.998847 |
| g750 | 1.968448 | 1.973308 | 1.966235 |
| g1000 | 1.954377 | 1.958090 | 1.954590 |

These terminal seed-42 observations show slightly higher loss with GMC at all
four widths and mixed, small LMC differences. They do not establish significance
across seeds or explain clipping behavior, which lacks the required log evidence.

### Requested C1 plots — 2026-09-11

`scripts/plot_tinystories_c1_variants.py` generated the explicitly scoped result
comparison at campaign `reports/c1-variants-20260911/`: loss/perplexity versus
active non-embedding parameters, four-panel full validation progress and a
separate epochs 3–4 detail, all as PNG/PDF. All 12 endpoint values match validated
terminal sidecars. Each of the 12 progress curves retains 5,446 unsmoothed ordinary
validation observations from update 64 through 348,528; its final loss matches
the terminal endpoint. Source hashes were checked before publication, and all
four PNGs were visually inspected. The directory includes CSV/JSON endpoints,
progress CSV, source provenance, the exact script and output hashes.

This is an authorized C1 result comparison with the clipping gap documented in
README/report metadata. It does not invoke or weaken strict full-campaign
freeze/report validation. Saved experiment artifacts remain unchanged, and no
training or holdout evaluation was executed.

The user identified the missing standalone reference points in that first C1
plot set. The corrected version is at
`reports/c1-variants-with-standalones-20260911/`, preserving the first version.
It includes all four original frozen standalone terminal endpoints as unconnected
brown triangles with one `Standalone` legend entry in the loss/perplexity figures.
The original shortened budget footnote states one epoch per standalone and four
per elastic run. All 16 endpoints were validated against their terminal sources;
the 12 C1 endpoints and progress CSV are unchanged. Both endpoint PNGs were
visually inspected; PNG/PDF files, tables, source hashes and exact plotting script
are retained in the new directory. C1 progress panels continue to compare the
three elastic trajectories over their full four-epoch horizon.

### C2 terminal validation and plots — 2026-09-11

C2-GMC **227651** and C2-LMC **227652** completed with Slurm exit **0:0**.
Each reached four complete epochs, 348,528 updates and 2,855,141,376 tokens in
one fresh attempt, with no failed optimizer updates or unresolved artifact errors.
Slurm allocations were 14,242 / 15,053 seconds; recorded training-attempt times
were 14,226.85 / 15,037.18 seconds, respectively.

`scripts/validate_tinystories_c2_terminals.py` passed the existing strict terminal
artifact reader for both runs. A snapshot containing these two complete runs is
at campaign `diagnostics/c2-terminal-freeze-20260911/frozen_manifest.json`.
Its campaign status is explicitly partial because it contains only the two C2
arms. C2's per-width contract does not require a separate clipping sidecar.
The C1/C3 clipping-artifact issue does not prevent C2 terminal acceptance.

`diagnostics/c2-terminal-validation.json` and `.log` record the complete outcome:
all eight terminal ordinary-validation endpoints and checkpoint identities
passed; all 348,528 action records and four epoch-order digests match preflight
and frozen original C2. All 759 saved model/optimizer tensors per run are finite.
The four AdamW histories contain exactly 39/51/63/75 parameter states for
g250/g500/g750/g1000; inactive blocks have no width-specific history, and each
present state's counter matches its width's observed update count. All moment
shapes, nonnegative second moments, scheduler positions and data cursors passed.
Each run's 370,312 recorded metric losses are finite. Completed resource ledgers
and worker exit codes were reconciled. No holdout evaluation was executed.

| Width | Original C2 loss | C2-GMC loss | C2-LMC loss |
| --- | ---: | ---: | ---: |
| g250 | 2.071721 | 2.072127 | 2.075571 |
| g500 | 1.994960 | 1.994998 | 2.002812 |
| g750 | 1.963783 | 1.963244 | 1.973272 |
| g1000 | 1.951738 | 1.948205 | 1.962822 |

These paired seed-42 endpoints show nearly unchanged GMC losses at the two
smaller widths and lower losses at the two larger widths; LMC losses are higher
at all four widths. This is descriptive evidence without multi-seed significance.

The plotting implementation is now `scripts/plot_tinystories_concat_variants.py`
with `--family C1|C2`; the old C1 entrypoint remains a compatible wrapper.
The C2 command writes a distinct `reports/c2-variants-with-standalones-20260911/`
directory containing loss/perplexity versus active non-embedding parameters,
full unsmoothed four-panel validation progress and a separate epochs 3–4 detail,
all as PNG/PDF, with endpoint and progress tables. The endpoint plots include
all four original standalone references as brown triangles with one legend entry
and the original shortened budget footnote. Full six-arm acceptance remains
incomplete; these C2 results do not waive the other arms' artifact requirements.

C2 plot export completed successfully: all 16 terminal rows match their validated
sources, and each of the 12 progress curves contains 5,446 recorded observations
from update 64 through 348,528 with a matching terminal loss. All four PNGs were
visually inspected; all eight PNG/PDF exports are present. The report directory
archives both exact scripts and `output-validation.json` with file hashes.

### C3 terminal audit and requested plots — 2026-09-11

C3-GMC **227930** and C3-LMC **227931** completed with Slurm exit **0:0**.
Both reached four epochs, 348,528 updates and 2,855,141,376 training tokens in
one fresh attempt without failed optimizer updates or unresolved artifact errors.
Allocated times were 13,737 / 14,158 seconds; measured training-attempt times
were 13,710.77 / 14,136.78 seconds. All six production jobs are now complete,
and the user has no jobs remaining in squeue. The helper's subsequent strict
campaign freeze stopped on C1-GMC's absent clipping log, as expected from the
documented writer defect. This is a reporting failure, not a failed training job.

`scripts/validate_tinystories_c3_terminals.py` recorded the separate C3 audit at
campaign `diagnostics/c3-terminal-validation.json` and `.log`. Both corrected C3
runs also lack their required clipping logs, so strict artifact validation retains
that failure. Checkpoint, endpoint and trace checks passed: all eight terminal
ordinary-validation endpoints match their contracts, all 300 saved model/optimizer
tensors per run are finite, and five disjoint owners cover all 75 parameter states
(12 each for A/B/C/D and 27 common). Owner counters are
348,528 / 261,630 / 174,409 / 87,072 / 348,528, matching actual width exposure.
Moment shapes, nonnegative second moments, global scheduler/data cursors and
resource ledgers passed. Full action and four epoch-order digests match preflight
and frozen original C3. All 370,312 recorded metric losses per run are finite.
The trace-only audit excludes the absent clipping file only in an in-memory
summary copy; saved artifacts and strict reporting requirements are preserved.

| Width | Original C3 loss | C3-GMC loss | C3-LMC loss |
| --- | ---: | ---: | ---: |
| g250 | 2.077839 | 2.076538 | 2.066974 |
| g500 | 1.999872 | 2.000256 | 1.994106 |
| g750 | 1.968143 | 1.968223 | 1.963359 |
| g1000 | 1.954962 | 1.954004 | 1.951429 |

LMC has lower terminal loss at all four widths in this paired seed-42 comparison;
GMC differences are small and mixed. No multi-seed significance is claimed.

The existing plotting script now accepts `--family C3`. Its distinct output at
`reports/c3-variants-with-standalones-20260911/` includes loss/perplexity endpoint
plots with four original standalone brown triangles, one Standalone legend entry
and the shortened budget footnote; full unsmoothed four-panel validation progress;
and a separate epochs 3–4 detail. PNG/PDF exports and CSV/JSON tables use the
audited saved results and explicitly retain the clipping-artifact limitation.
No training or holdout evaluation is performed to generate these plots.

C3 exports completed and were checked: 16 terminal rows, 12 progress curves with
5,446 observations each from update 64 through 348,528, and all eight PNG/PDF
files. The last progress losses match the terminal endpoints. All four PNGs were
visually inspected. Exact audit/plotting scripts and `output-validation.json`
with file hashes are archived alongside the figures and tables.

### Combined C1/C2/C3 plot set — 2026-09-11

At the user's request, `scripts/plot_tinystories_all_concat.py` combined the three
validated family report tables and curves at
`reports/all-c-variants-20260911/`. This includes all nine concat variants and
the four standalone terminal references: **40 endpoint rows**, with each
standalone included once. Color identifies C1/C2/C3; solid/dashed/dotted lines
identify none/GMC/LMC. Standalones remain unconnected brown triangles with one
legend entry, and the original shortened budget footnote is retained.

Loss/perplexity endpoint plots, full four-panel validation progress and a separate
epochs 3–4 detail are exported in PNG/PDF. The progress table contains **36 curves**
with **5,446 unsmoothed observations each**; last values match terminal endpoints.
Input table hashes and underlying source hashes were checked before/after export.
All four PNGs were visually inspected; all eight exports, CSV/JSON tables, exact
script, source provenance and output hashes are retained. Earlier reports and
run artifacts remain unchanged. This completes the requested combined quality
plots; strict full-campaign acceptance still records the missing corrected
C1/C3 clipping evidence and has not been weakened or marked passed.
