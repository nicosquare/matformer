# Verification: TinyStories S1 Fourfold LR Warmup

## Scope and stage status

The current authorization covers phase 4 (T017–T035), building on phases 1–3
and their historical compatibility and CPU preparation baseline. It creates no production
root, reservation, production source snapshot, real-input readiness gate, or
training job. Snapshot/gate/submission artifacts created by tests are temporary fixtures. Historical
results remain read-only. Fixture metadata is not an audit of real inputs.

| Stage | Status | Evidence / remaining work |
| --- | --- | --- |
| Setup and compatibility foundation | Passed | T001–T004; 20 focused compatibility tests passed |
| Phase-3 CPU preparation | Passed against fixtures | T005–T016; 72 distinct focused cases validated |
| Phase-4 runtime/admission | Passed against CPU fixtures | T017–T035; 1149 broad regression passes; final runtime60/queue54 passes |
| Full implementation checks | Pending | Reporting phase 5 and final T049–T050 remain |
| Real input audit | Pending | Corpus/tokenizer/role and trace checks at T051 |
| Selected historical reference audit | Pending | Ten terminals at T051; planning observations are not certification |
| Snapshot-bound CPU gate | Pending | T052; local compatibility tests are not this gate |
| GPU readiness | Pending | T053; requires subsequent diagnostic authorization |
| S1-linear-w256 terminal | Pending | T054; requires production authorization and passed gates |
| S1-geometric-w256 terminal | Pending | T054; requires production authorization and passed gates |
| New-only report | Pending | Eight valid new endpoints at T055 |
| Endpoint comparison | Pending | 24 endpoints and eight paired differences at T055 |
| Early metrics and report completion | Pending | Raw early evidence, sixteen figure files, findings at T055 |
| Overall SC-001–008 acceptance | Pending | Reconcile saved evidence at T056 |

## Environment and source identity

Observed on 2026-09-22 from the repository root, using
`/home/ivo.navarrete/.conda/envs/elasticnn/bin/python` and
`importlib.metadata.version` (no package installation):

| Component | Observed version |
| --- | --- |
| Python | 3.12.13 (conda-forge; GCC 14.4.0) |
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.8.0 |
| NumPy | 2.4.3 |
| PyYAML | 6.0.3 |
| Matplotlib | 3.10.9 |
| pytest | 9.0.3 |

Pre-change revision: `f309fe22e98ad0e78698ba963911b2f84df729ff`
(`git rev-parse HEAD`). The working tree was clean at entry. Runtime resolution
and serialization source files were unchanged during phases 1–2. Phase 3
changes the resolver/campaign module while retaining the serializer and golden
signatures.

| Baseline source / retained fixture | SHA-256 |
| --- | --- |
| `src/evaluation/optimizer_ownership.py` | `0dc833c73992438ddb3d5a4ecc6de13d8f3cece5d91bdd34b3951a64d27f1458` |
| `src/utils/config.py` | `56f3bd163cc252f229ffe7dc92c2bb6b60a590fc0282a0d6cf5ced4c8e16b9eb` |
| `src/utils/reproducibility.py` | `b54771768da5a9ba7d093d73828e71bdfa943c2fbad55fd3f1613b1be094ecc1` |
| `tests/test_optimizer_ownership_campaign.py` | `80e6fc0bdfebd01df25ed77897d771173d2367d23a434085dda15ce3a2fb40d5` |
| `tests/fixtures/optimizer_ownership_legacy_signatures.json` | `8a3d67fac6f7bc266866bc24b597de71dadb7b5dc1d0af0f54c914ce74201252` |

The existing Python/universal ignore rules were verified. No additional ignore
file or dependency is required. Requirements checklist: 16/16 complete.

## Evidence ledger

Capture commands, fixture hashes and test outcomes are recorded here as the
foundation tasks execute. Future gates must retain their own source/config
bindings, commands/log hashes, exit status, passed/failed/skipped counts and
artifact links; this record cannot substitute for those artifacts.

### T001–T002: setup complete

Created this stage ledger and the [experiment runbook](../../docs/tinystories-s1-warmup-experiment.md).
Reviewed the two-arm matrix, controls, budgets, ten read-only selections,
artifact layout, proposed CLI and authorization boundaries against the design
contracts. These documentation checks are not CPU/GPU readiness.

### T003: pre-change schema-4 capture complete

Captured all nine arms through the unchanged `expand_campaign` and
`resolve_run_config`, using the existing `audited_inputs` fixture to stub external
corpus/tokenizer reads only. No model/training/data audit was executed. The
fixture records source revision/file hashes, fixed initialization provenance,
common/grid/topology hashes, full resolved-config hashes and scientific-contract
hashes, plus exact top-level config/contract field sets.

Exactly four path values (`run.output_dir`, `run.output_root`,
`model.tokenizer_dir`, `dataset.prepared_corpus_dir`) are normalized to
`<fixture>` locations. No scientific fields are removed. Fixed provenance is
`{"code_revision": "s1-warmup-schema4-fixture", "working_tree_dirty": false}`;
actual capture-source hashes are separately recorded. Schemas 1–3 retain the
original fixture and its own fixed provenance verbatim.

Evidence: [schema-4 signatures](../../tests/fixtures/s1_warmup_schema4_signatures.json).
SHA-256: `657031baaf15b720605259799b6ba9b89fcb7e2d721e4fd6b7df3030b1100b19`.

One-time capture command (run at the source revision above, from the repo root;
exclusive creation deliberately refuses to overwrite an existing golden):

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python - <<'PY'
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest
import yaml

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path('tests').resolve()))
from test_optimizer_ownership_campaign import audited_inputs, expand
from src.evaluation import optimizer_ownership as campaign
from src.utils.reproducibility import stable_hash

recipe_path = Path('configs/controlled_exps/tinystories_instruct_matformer_widths.yaml')
provenance = {'code_revision': 's1-warmup-schema4-fixture', 'working_tree_dirty': False}
source_paths = [
    'src/evaluation/optimizer_ownership.py', 'src/utils/config.py',
    'src/utils/reproducibility.py', 'tests/test_optimizer_ownership_campaign.py',
    'tests/fixtures/optimizer_ownership_legacy_signatures.json', str(recipe_path),
]
result = {
    'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'source_files_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths},
    'provenance': provenance,
    'normalized_paths': {
        'run.output_dir': '<fixture>/runs/<arm_id>',
        'run.output_root': '<fixture>/runs',
        'model.tokenizer_dir': '<fixture>/tokenizer',
        'dataset.prepared_corpus_dir': '<fixture>/corpus',
    },
    'common_hash': stable_hash(campaign.campaign_common(4)),
    'widths_hash': stable_hash(campaign.campaign_widths(4)),
    'topology_hash': stable_hash(campaign.campaign_topology(4)),
    'arms': {},
}
with tempfile.TemporaryDirectory(prefix='s1-warmup-baseline-') as directory, pytest.MonkeyPatch.context() as patch:
    root = Path(directory)
    audited_inputs.__wrapped__(root, patch)
    patch.setattr(campaign, '_provenance', lambda: copy.deepcopy(provenance))
    for run in expand(root, yaml.safe_load(recipe_path.read_text())):
        config = copy.deepcopy(run['resolved_config'])
        for field, value in result['normalized_paths'].items():
            section, key = field.split('.')
            config[section][key] = value.replace('<arm_id>', run['arm_id'])
        result['arms'][run['arm_id']] = {
            'contract_hash': run['contract_hash'],
            'resolved_config_hash': stable_hash(config),
            'contract_fields': sorted(run['optimizer_ownership_contract']),
            'resolved_top_level_fields': sorted(config),
        }
output = Path('tests/fixtures/s1_warmup_schema4_signatures.json')
# A baseline is captured once from unchanged code, never refreshed by tests.
with output.open('x') as stream:
    stream.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(f'{len(result["arms"])} arms: {output}')
print('SHA-256:', hashlib.sha256(output.read_bytes()).hexdigest())
PY
```

The command was executed via `/tmp/capture_s1_warmup_schema4.py`; the inline
body above is retained so reproduction does not depend on that temporary file.
Never regenerate these goldens to accommodate a compatibility regression.

### T004: historical compatibility baseline passed

[test_s1_warmup_campaign.py](../../tests/test_s1_warmup_campaign.py) covers all
29 historical arms (schema1:9, schema2:6, schema3:5, schema4:9). It checks:

- Exact scientific-contract signatures for schemas 1–4, and complete resolved
  config signatures for all nine schema-4 arms with only four paths normalized.
- Omitted-arm selector outputs for schemas 1–4 and completely omitted defaults.
- Exact contract field sets, detached serialization and JSON/YAML round trips.
  Only schema2 includes `correction`; schema4 alone includes topology fields.
- Warmup256 rejection in common/individual-arm recipe overrides, direct budget
  checks, materialized config validation and ordinary YAML resolution. Rehashed
  mutated contracts still fail the strict 64-update rule.
- No output run directories created during expansion or rejection checks.

Final focused command:

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_s1_warmup_campaign.py -q -rs --tb=short
```

Result: exit0, **20 passed, 0 failed, 0 skipped**, 10.35 seconds. The first
iteration had 19 passed/1 failed because the test incorrectly expected a
`correction` field in schema3; an intermediate assertion also incorrectly
expected it for schema3 concat arms. Inspection confirmed no schema3 arm has
that field. Only the new test expectation was corrected; runtime and golden
signatures were unchanged.

Source/recipe checks: all six SHA-256 bindings in the schema4 fixture still
match, including the retained schemas1–3 fixture. Documentation relative links
resolve. Full feature implementation and all real-input/GPU/result stages remain
pending. These tests use synthetic metadata, not historical checkpoint reads,
actual corpus audits, real schedule execution or GPU diagnostics.

Final test source SHA-256: `8501fb2f7045b0463ac51dcdfcd89ca45b7cba41e1544a1521ccaa554e9a2a19`.

Additional existing-regression command (started before the final field-set
assertion correction and allowed to finish):

```bash
OMP_NUM_THREADS=1 /home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_s1_warmup_campaign.py tests/test_matformer_widths_campaign.py -q -rs --tb=short
```

Raw result: exit1, **223 passed, 1 failed, 5 skipped**, 332.80 seconds. The sole
failure was the intermediate schema3 field-set assertion documented above,
subsequently corrected and verified by the final 20/20 focused run. The existing
`test_matformer_widths_campaign.py` portion had **204 passed, 0 failed, 5 skipped**.
All five skips are explicitly separately authorized sbatch GPU semantic probes;
none establishes GPU readiness. Two collected SWIG deprecation warnings were
reported, plus a `swigvarlink` shutdown warning. No runtime-source change was
needed. Across the final focused run and the existing-regression portion, all
224 selected CPU cases passed.

Final repository checks: `git diff --check` passed; new-file whitespace and
relative documentation links passed; exactly T001–T004 are checked in tasks.md.
The optional `after_implement` Git commit hook was offered but not executed.

### T005–T016: phase-3 CPU preparation

Implemented the fixed schema-5 recipe, required arm-qualified selectors, strict
merged and materialized protocol validation, new-only grid/intervention/reference
contracts, closed counterpart control audit, and full expected schedule export.
Historical `PINNED_COMMON`, serializer and seed derivation remain unchanged.
Both grids use fresh CPU model construction for physical count checks.

Selected-reference validation inspects exactly ten terminals/sixteen historical
endpoints and their saved config, checkpoint, evaluation, trace, precision and
applicable launcher/job/resource records. It reads no cancelled-arm artifacts.
Modern submissions bind manifest/config hashes, successful worker/process/job
identity, measured CUDA allocation and (for replacement S1) CUDA-entry evidence.
Legacy Linear and earlier Geometric standalones use their generation's evidence;
read-only scheduler accounting still must show successful completion.

The analyzer preflight adds both reference-root flags, rejects occupied roots,
compares each new run's complete expected traces to its own selected S1, and
publishes configs/audits/traces/schedules/reference selections atomically. The
selection record is `campaign/references/selection.json`, inside the same atomic
directory transaction rather than a separately published root-level directory.
Repeated preparation verifies an identical untouched identity and file/source
hashes without rewriting it. Publication failures remove the new reservation.

Test scope and limitations:

- Full 348529-position schedules for both grids are streamed and compared with
  the installed Transformers cosine function. Hashes use ordered little-endian
  float64 values; the terminal has no next applied update. This is expected
  schedule evidence, not actual optimizer execution (phase 4).
- Both full four-epoch batch streams use 5576448 designated sequences and a
  43-sequence excluded tail from a synthetic 5576491-element initial order.
  All 348528 actions use the actual isolated `randrange(4)` primitive. Own-pair
  traces agree; realized counts are not forced to their 87132 expectations.
- Reference fixtures use synthetic checkpoint payloads and stub bulk terminal
  trace IO and Slurm accounting. They validate budgets/metadata/publication,
  not real training or actual device execution. Source immutability is checked.
- Read-only inspection of existing saved config/launcher/resource formats
  informed generation compatibility. No full real input/reference audit,
  checkpoint certification, production reservation, readiness gate or job was
  performed. No historical file was modified.
- The closed audit explicitly lists consequential scheduler/disabled diagnostic
  milestone paths. Older Linear's absent sign-dynamics record is accepted only
  as the current exact disabled defaults, enumerated by leaf path. No complete
  scientific-control section is ignored.

Initial test development observed the expected missing-schema failures with all
20 historical compatibility tests passing. Intermediate green runs: 61 passed,
then 68 passed. A later run exposed the warmup diagnostic being reported before
its primary invalid control; validation was reordered for the correct named
failure. Another intermediate failure correctly detected source edits made
while a repeated-preflight test was running; final tests run against fixed source.

Final validation commands (repository root, `OMP_NUM_THREADS=1`):

```bash
/home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_s1_warmup_campaign.py tests/test_s1_warmup_reporting.py -q -rs --tb=short
```

Result: exit 0, **70 passed, 0 failed, 0 skipped**, 92.57 seconds. Includes all
20 schema-1–4 compatibility cases; neither golden fixture was regenerated.
The subsequent targeted run below adds two new cases and changes the existing
atomic-publication case to enter through the actual analyzer CLI.

```bash
/home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_config.py tests/test_optimizer_ownership_campaign.py tests/test_matformer_widths_campaign.py -q -rs --tb=short
```

Initial result: exit 1, **466 passed, 1 failed, 5 skipped**, 90.00 seconds. The
sole failure was the old unknown-schema test still treating integer 5 as
unsupported. Its unsupported-version parameter was changed to 6, keeping the
noninteger/unknown-version checks. No historical signature or scientific
expectation was changed. The five skips are separately authorized sbatch GPU
semantic probes and establish no GPU readiness.

```bash
/home/ivo.navarrete/.conda/envs/elasticnn/bin/python -m pytest tests/test_s1_warmup_reporting.py::test_preflight_atomic_publication_and_identical_verification tests/test_s1_warmup_reporting.py::test_reference_changed_during_read_is_rejected tests/test_s1_warmup_reporting.py::test_cli_requires_both_reference_flags tests/test_matformer_widths_campaign.py::test_unknown_or_noninteger_schemas_fail_all_selectors -q -rs --tb=short
```

Result: exit 0, **12 passed, 0 failed, 0 skipped**, 18.11 seconds. This verifies
the actual CLI publication path, a source changing during reference inspection,
missing required flags, and all nine corrected unknown/noninteger schema cases.
Across these runs, **72 distinct phase-3/compatibility tests** and **467 existing
CPU regression cases** pass; five existing GPU cases remain skipped. The two
SWIG import deprecation warnings and shutdown warning are unchanged.

CLI `preflight --help` displays both new reference flags. `git diff --check`
passes. T001–T016 alone are checked; later phases remain pending. The optional
Git hooks were surfaced but not executed. Working changes remain uncommitted.

Final implementation and verification bindings (SHA-256):

| File | SHA-256 |
| --- | --- |

| `src/evaluation/optimizer_ownership.py` | `58ea35226f71144957ec527ab8163601282c652699b60f79704fa975d0c52f79` |
| `src/utils/config.py` | `1b3390d5c5d2817fc041341bbb6babcaf970b33d831a10b5485a86d7166f5707` |
| `src/utils/reproducibility.py` | `b54771768da5a9ba7d093d73828e71bdfa943c2fbad55fd3f1613b1be094ecc1` |
| `scripts/analyze_tinystories_optimizer_ownership.py` | `8aaf1dd13bdcb9c4e6f34b541eae8b65b3497eb7cc1fc0bfdda63d92bd1bedb7` |
| `configs/controlled_exps/tinystories_instruct_s1_warmup.yaml` | `133ba407d683e803c023a301680e78e9675bb4e269f436a77b38fbf82d858b6b` |
| `tests/test_s1_warmup_campaign.py` | `913bf8026cf4626ee91dd41e2a68741dcb6815264068ffd53e3d647391a571ec` |
| `tests/test_s1_warmup_reporting.py` | `4b3c13ca0134160a3e5c9acb08b6572d8a2dfa89916aad6126103c55bc02f12b` |
| `tests/test_matformer_widths_campaign.py` | `70fbfc88e4852dd4506e19c9fbb22925cca20c3e65b05d3a5e6e213f464d67ab` |
| `tests/fixtures/optimizer_ownership_legacy_signatures.json` | `8a3d67fac6f7bc266866bc24b597de71dadb7b5dc1d0af0f54c914ce74201252` |
| `tests/fixtures/s1_warmup_schema4_signatures.json` | `657031baaf15b720605259799b6ba9b89fcb7e2d721e4fd6b7df3030b1100b19` |


## Phase 4: runtime, readiness software and two-arm lifecycle

T017–T035 are complete. T034 records the final test outcomes below.
The working tree already contained the phase1–3 changes on entry; they were
preserved. No external historical input was modified and no sbatch command,
GPU workload or production training was executed. Slurm/accounting and device
admission cases use fixtures. The Python/universal ignore rules remain adequate;
no dependencies or ignore patterns were added.

### Runtime evidence

- The existing scheduler is checked at all348529 positions for both grids.
  Actual training captures pre-optimizer LR for updates1–258, including64/65,
  256/257 and the first peak at257; CSV values must equal the captured rates.
  Each committed update advances the scheduler exactly once. The final-update
  probe verifies positive applied LR at348528 and stored terminal LR zero.
- All four physical widths use shared full-tensor AdamW history and global L2
  clipping cap1. Inactive tails retain decaying momentum and change beyond
  pure weight decay. Summary and compact-accounting support are arm-qualified.
- Resume probes retain horizon348528 and epochs at87132/174264/261396. Boundary
  positions255/256/257 and each epoch minus1/exact/plus1 compare model,
  optimizer, scheduler, RNG, exact batch/action suffixes, cursor, counters and
  compact metrics. Late-boundary histories are explicitly synthetic and
  state-seeded; only their short suffixes actually execute.
- Cross-run/grid, warmup64, horizon, model-only, non-finite optimizer history,
  scheduler, cursor, RNG and metrics corruption are rejected before live
  mutation. Optimizer, scheduler and accounting failure injections poison the
  live state and preserve the durable checkpoint for all save reasons.
- At348528, both direct terminal recovery and full trainer re-entry evaluate
  four ordinary-validation endpoints without another update. Two full trainer
  re-entries record zero attempted updates and preserve checkpoint bytes.

CPU diagnostics use d64/l4/h4/fullFFN256 with synthetic batch1/context8/vocab32;
GPU-mode fixtures use batch64/context128/vocab2048 and bf16. Resume state and
stream checks require exact equality. Inactive-tail arithmetic retains
rtol1e-6/atol1e-7; clipping uses the inherited1.000001 bound. Independent
analytic LR comparison uses rel_tol1e-12/abs_tol1e-18 solely for floating-point
operation-order rounding; measured-versus-applied rates are exact. No inherited
model/optimizer resume tolerance was relaxed.

`steps.py` had no literal S1-name eligibility checks to change: declared slicing,
shared optimizer scope and global clipping already select the right semantics.
`checkpointing.py` already binds the entire scientific contract, descriptors,
scheduler and budget before installation, including the new grid/intervention
fields. Those existing paths were retained and exercised rather than forked.
The runtime changes are limited to arm-qualified selectors and schema5 metadata.
Terminal/trace inspection also resolves the arm's grid and validates its metrics
and resource disclosure for launcher completion.

### Readiness and lifecycle evidence

The new diagnostic script freezes executable sources, re-enters that snapshot,
runs CPU checks with CUDA hidden, and records sealed command/log/JUnit/input
bindings. Its GPU path requires a durable sbatch intent, one usable GPU and
actual bf16 forwards for both real-shape grids; it retains the full horizon,
requires all runtime boundary/failure tests and cannot pass skipped checks.
Actual GPU execution remains T053; implementation tests are not GPU readiness.

The launcher verifies preflight reservation and snapshot/config/recipe/audit/
reference bindings, permits exactly the two declared arms, reserves running
capacity for pending/uncertain/unrelated jobs, and uses the inherited live
Slurm-limit queries and required exclusions. It persists intents before sbatch,
locks writers, validates own continuation, and requires matching successful
Slurm accounting, worker, CUDA entry, actual bf16/allocator evidence and complete
terminal/trace state. Worker success cannot replace delayed accounting.
Missing terminal outputs, including after recorded completion, require a
validated full-horizon checkpoint and a new completion-only attempt.

Resource ledgers retain failed/replayed costs separately from durable progress,
and worker/queue status discloses unobserved attempts and separate overlapping
allocation seconds. CPU/GPU/terminal/new-report/comparison/early-report statuses
remain distinct. The phase4 CPU gate sets `reporting_fixture_status=pending`;
production admission requires phase5's reporting fixture acceptance. No manually
relabelled gate or earlier campaign approval can bypass that prerequisite.

### Validation record

Final results and exact commands are saved in [phase4-cpu-tests.json](evidence/phase4-cpu-tests.json), including source/recipe/golden hashes and log/JUnit hashes. CLI help
for both new scripts exited0. `git diff --check` passed. The runbook documents
snapshot, CPU/GPU commands, prepare/queue/worker, continuation, delayed accounting,
resource disclosure and completion-only recovery, with the remaining execution
authorizations explicit.


| Check | Passed | Failed | Skipped | Result |
| --- | ---: | ---: | ---: | --- |
| Fourteen focused/relevant regression suites | 1149 | 0 | 39 | Exit0, 355.45s |
| Final runtime matrix (`-k runtime`) | 60 | 0 | 0 | Exit0, 29.57s; 53 deselected |
| Final launcher/readiness matrix | 54 | 0 | 0 | Exit0, 9.31s |

The broad run includes all three feature suites plus config, ownership runtime,
campaign, resume, reporting, corrections, MatFormer campaign/reporting/queue,
resource reconciliation and compact metrics. Its39 skips are34 CUDA-unavailable
cases and5 separately authorized GPU semantic probes. Only existing SWIG
module deprecation warnings remain. The runtime/queue totals overlap the broad
selection; they are not additional distinct-test totals.

Logs: [broad regressions](evidence/phase4-regressions.txt),
[final runtime](evidence/phase4-runtime-final.txt),
[final queue](evidence/phase4-queue-final.txt). The broad run preceded the final
missing-output recovery refinement and two stronger runtime assertions; the
final60/54-case reruns cover those final source versions. Shared runtime sources
were unchanged after the broad run started. Golden signatures were preserved.

During TDD the new tests first exposed the missing arm-qualified selector and
missing launcher modules. Subsequent fixture corrections removed stale prefix
metadata, synchronized diagnostic resolved scheduler fields, normalized JSON
sequence types, and excluded test-spy attributes from state comparisons.
Independent cosine operation ordering differed by a rounding bit; measured LR
comparisons remain exact. None of these changes relaxed production contracts.
The final test results above supersede those intermediate failures.

The exact CLI help commands also passed:

```bash
/home/ivo.navarrete/.conda/envs/elasticnn/bin/python scripts/run_tinystories_s1_warmup.py --help
/home/ivo.navarrete/.conda/envs/elasticnn/bin/python scripts/preflight_tinystories_s1_warmup.py --help
```

All phase4 task boxes are checked. Phase5 and T049–T056 remain unchecked even
where this work exercised overlapping regressions. A real snapshot-bound CPU
gate remains T052; GPU readiness, production, real terminal acceptance and
reports remain outstanding. The optional Git commit hook was not executed.
