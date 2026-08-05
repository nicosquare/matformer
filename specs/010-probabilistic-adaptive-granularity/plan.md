# Implementation Plan: Probabilistic Adaptive Granularity

**Branch**: `010-probabilistic-adaptive-granularity` | **Date**: 2026-08-05 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/010-probabilistic-adaptive-granularity/spec.md`

**Note**: This plan implements genuine Bayesian Thompson sampling in two stages. It replaces the legacy pseudo-Thompson runtime, preserves historical artifacts for reporting, and leaves UCB behavior untouched.

## Summary

Add a fixed-panel, delayed-reward Bayesian controller for MatFormer granularity training. P1 adds an `adaptive_global` scope and validates deterministic four-role data separation, the uniform all-granularity controller objective, Gaussian prediction/conditioning, posterior Thompson selection, decision-window persistence, distributed agreement, and exact resume. P2 reuses that controller with identifiable additive block/granularity contrasts and profile selection without complete-profile enumeration. P3 adds explicit method/version provenance so new Bayesian global/per-block runs, retained UCB, random baselines, and historical heuristic Thompson artifacts remain unambiguous.

Implementation stays in the current config/data/training/evaluation/checkpoint/reporting flow. A focused controller module owns the small dense float64 Gaussian state and action features; the training loop calls its boundary operations directly. Existing target-token-weighted validation, seed streams, object broadcast, checkpoint machinery, metrics/run summaries, and artifact I/O are extended rather than replaced.

## Technical Context

**Language/Version**: Python 3.12  
**Primary Dependencies**: PyTorch, transformers, datasets, PyYAML, NumPy, pandas, pytest; no new runtime dependency  
**Storage**: YAML inputs; JSON manifests/summaries; JSONL controller boundary journal; CSV ordinary metrics/scaling results; PyTorch checkpoints under existing run output directories  
**Testing**: pytest unit tests for Gaussian math/features/config/data; CPU training and resume integration tests; distributed gloo/FSDP-compatible controller agreement tests where current test infrastructure permits; existing compatibility suite  
**Target Platform**: Linux research workstation or single-node CPU/GPU cluster, including the repository's current multi-process FSDP execution path  
**Project Type**: research model-training pipeline with post-training evaluation/reporting  
**Experiment Scope**: fixed controller/final data roles, Bayesian global arms, Bayesian additive per-block effects, delayed reward windows, posterior persistence/resume, final-holdout comparison, and provenance-safe reporting  
**Datasets/Data Assumptions**: the configured causal-language-model dataset tokenizes one source row into one fixed-length example with stable source identity; role selection operates only on examples with at least one valid target; ordinary validation retains its configured size  
**Configuration Inputs**: existing YAML plus dotted CLI overrides; `adaptive_global` or `adaptive_per_block`, `thompson`, required Bayesian prior/covariance/noise mapping, fixed controller/final role sections, and optional decision interval defaulting to 50  
**Experiment Outputs**: resolved config; parent and four role manifests/hashes; `controller_metrics.jsonl`; `controller_summary.json`; compact controller fields in metrics; complete Bayesian checkpoint state; run summary; optional post-training `final_holdout_results.json`  
**Reproducibility Notes**: strict existing runtime settings; independent named seed streams derived from the saved run root seed for controller split, final split, and posterior sampling; controller-local random state; stable feature/split/method versions; rank-zero controller authority; complete boundary phase persisted; all role hashes included in resume compatibility; matched fresh/resume tests require exact discrete controller, sampling, manifest, and action state plus `rtol=1e-6`, `atol=1e-8` for objective, reward, and posterior values on the same software/hardware/runtime topology  
**Performance Goals**: one conceptual controller evaluation per boundary; exactly `|G|` fixed-panel subnetwork evaluations every `h` optimizer steps; `O(|G|)` global selection; `O(B|G|)` additive per-block selection; no `|G|^B` enumeration; dense controller state limited to dimension `|G|` or `1+B(|G|-1)`  
**Constraints**: fixed 128-example controller and 512-example final holdout; uniform global-granularity objective; intercept-only context; identity transition; finite valid Gaussian inputs; zero compute/switch costs; no training-loss reward; no partial observation for an incomplete window; no legacy Thompson resume; UCB semantics unchanged  
**Scale/Scope**: arbitrary nonempty ordered granularity labels/counts, one or more transformer blocks, configurable positive decision interval, fresh and exact/inside-boundary resumes, existing single-node distributed world sizes, opt-in smoke/pilot runs

## Constitution Check

*GATE: Evaluated before research and re-evaluated after Phase 1 design.*

- **Research code first**: Pass. The controller is a focused experiment module with equations, feature construction, and state transitions visible; no service or production framework is introduced.
- **Simplicity and local reasoning**: Pass. Dense float64 vectors/matrices and an explicit small phase machine are easier to inspect than a general bandit framework. Tensor/matrix dimensions are named and saved.
- **Explicit experiment flow**: Pass. Data reservation, initial objective, fixed action window, boundary evaluation, posterior update, and next selection remain direct calls in the existing orchestration/training flow.
- **Minimal abstraction and validation**: Pass. A separate Bayesian state schema is necessary to avoid corrupt legacy resume and UCB drift. Strict finite/covariance/manifest checks protect against silent experimental invalidity.
- **Transparent configuration and reproducibility**: Pass. Required probabilistic values map directly to the research equations; method/split/feature versions, seeds, manifests, posterior state, window phase, and controller RNG are persisted.
- **Useful outputs and logging**: Pass. Full controller state is stored in structured artifacts, while concise rank-zero console records expose controller initialization, completed boundaries, resume transitions, incomplete terminal windows, and attributable failures without dumping posterior vectors or covariance matrices.
- **Shallow organization**: Pass. Changes stay in existing `src/training`, `src/evaluation`, `src/utils`, `scripts`, `configs`, and `tests` directories with one focused controller module and one final-evaluation module.

No constitution gate is violated.

## Project Structure

### Documentation (this feature)

```text
specs/010-probabilistic-adaptive-granularity/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── configuration.md
│   ├── controller-boundaries-and-artifacts.md
│   └── cli-entrypoints.md
└── tasks.md                              # created later by /speckit-tasks
```

### Source Code (repository root)

```text
train.py                                  # preflight and existing training CLI
configs/                                  # opt-in Bayesian smoke/pilot config
scripts/
├── evaluate_final_holdout.py             # post-training-only final comparison
└── queue_dmodel256_pilot.py              # explicit opt-in only; defaults unchanged
src/
├── training/
│   ├── probabilistic_controller.py       # Gaussian features/state/predict/update/sample
│   ├── data.py                           # four-role split and manifests
│   ├── steps.py                          # visible decision-window integration
│   ├── run.py                            # setup, artifacts, warmup boundary, orchestration
│   ├── checkpointing.py                  # Bayesian schema and strict resume
│   └── distributed.py                    # existing reduction/broadcast helpers reused
├── evaluation/
│   ├── validation.py                     # stable controller objective primitive
│   ├── final_holdout.py                  # post-training manifest reconstruction/evaluation
│   ├── reporting.py
│   ├── reporting_io.py
│   ├── reporting_styles.py
│   └── reporting_impl.py                 # keep compatibility path classification aligned
├── models/
│   ├── adaptive_sampler.py               # retained UCB runtime; pseudo-Thompson not selectable
│   └── granularity.py                    # resolved ordered labels reused
└── utils/
    ├── config.py                         # strategy-specific resolution/migration validation
    ├── reproducibility.py                # new independent seed streams/signatures
    └── metrics.py                        # compact fields and controller summary paths
tests/
├── fixtures/
│   ├── probabilistic_adaptive_global_smoke.yaml
│   └── probabilistic_adaptive_per_block_smoke.yaml
├── test_probabilistic_controller.py
├── test_probabilistic_controller_resume.py
├── test_data_validation.py
├── test_config.py
├── test_distributed.py
├── test_artifacts.py
├── test_reporting.py
├── test_phase2_finalize.py
└── test_training_smoke.py
```

**Structure Decision**: Keep the repository's existing shallow structure. Add one direct training controller module because Bayesian state/equations are incompatible with the retained heuristic UCB state, and one evaluation module/script because the final holdout must remain physically outside training-time decision flow. Extend existing files for orchestration, persistence, and reporting instead of creating registries or a new package hierarchy.

## Phase 0: Research Decisions

Research is complete in [research.md](./research.md). The decisions are:

1. Add `adaptive_global`; route `thompson` to Bayesian control in both adaptive scopes and leave `adaptive_per_block + ucb` on the unchanged legacy path.
2. Require a dedicated Bayesian controller mapping and fail old-shaped Thompson configurations rather than synthesizing priors/noise.
3. Build a deterministic without-replacement four-role partition while preserving the existing ordinary-validation selection first.
4. Reuse ordered target-token-weighted validation for the controller objective with strict finite checks and distinct identity.
5. Use intercept plus deterministic sum-to-zero contrasts: dimension `|G|` globally and `1+B(|G|-1)` per block.
6. Resolve explicit Gaussian inputs to dense CPU float64 state and sample through a deterministic symmetric covariance factor.
7. Persist an explicit transactional decision-window phase machine beginning after optional pre-nested warmup.
8. Make rank zero authoritative for posterior randomness/update and broadcast validated action/state in distributed runs.
9. Store a separate versioned Bayesian checkpoint schema and reject legacy heuristic Thompson resume.
10. Journal full controller boundary events separately and classify method identity from explicit provenance, not strategy name alone.
11. Reserve final data during setup but evaluate it only through a separate post-training comparison entrypoint.
12. Validate global first, additive per-block second, provenance third, without expanding the default pilot queue.

All technical unknowns are resolved; no `NEEDS CLARIFICATION` markers remain.

## Phase 1: Design and Contracts

Design artifacts are complete:

- [data-model.md](./data-model.md) defines configuration, experiment identity, stable example identities, four manifests, feature schema, Gaussian belief, posterior sampling, decision windows, boundary observations, summaries, and final comparison state.
- [contracts/configuration.md](./contracts/configuration.md) defines the YAML/override surface, strategy matrix, required Bayesian values, fixed method values, data roles, migration failures, and preflight output.
- [contracts/controller-boundaries-and-artifacts.md](./contracts/controller-boundaries-and-artifacts.md) defines boundary order, transactional commit, checkpoint phases, controller journal/summary schemas, historical identity, and final-holdout separation.
- [contracts/cli-entrypoints.md](./contracts/cli-entrypoints.md) defines `train.py`, the post-training final-holdout command, opt-in pilot behavior, outputs, and attributable failures.
- [quickstart.md](./quickstart.md) defines validation commands for migration, math, data roles, global/per-block smoke, resume, distributed agreement, reporting, final comparison, and compatibility.

### Post-design constitution re-check

- The design remains direct: no registry, policy network, replay buffer, or generic controller framework was introduced.
- Added validation is limited to failures that would invalidate rewards, posterior state, data isolation, or resume provenance.
- High-dimensional controller records use a dedicated JSONL journal rather than expanding every ordinary metrics row.
- The final-holdout entrypoint is separate specifically to make the no-training-consultation constraint visible and testable.
- UCB stays isolated on its current schema and execution path, avoiding a broad adaptive refactor.

All constitution gates continue to pass after design.

## Implementation Sequence

### P1: Bayesian global adaptation

1. Extend configuration resolution with `adaptive_global`, strategy-specific validation, required Bayesian/data mappings, fixed resolved method values, and migration errors; add new seed streams and preflight fields.
2. Carry stable source identities through preprocessing, build the Bayesian-only four-role split, validate usable counts/disjointness, write all manifests before optimizer updates, and update reproducibility signatures.
3. Add the versioned Gaussian controller state, global contrast schema, exact predict/update/sample/tie operations, finite/PSD validation, and controller-local random state.
4. Reuse validation evaluation for the controller objective and integrate the explicit window state machine after pre-nested warmup, with rank-zero authority and broadcast in distributed runs.
5. Extend checkpoints/resume, controller journaling, summaries, compact metrics, config rewrite timing, concise rank-zero controller lifecycle logs, and failure records; process completed boundaries before post-step checkpoints.
6. Verify mathematical updates, data isolation, initial/completed/incomplete windows, finite failures, exact/inside-boundary resume, distributed agreement, arbitrary labels, one arm, and legacy migration rejection.

### P2: Bayesian additive per-block adaptation

1. Extend the feature schema with one sum-to-zero contrast block per transformer block and persist coefficient identities/schema hash.
2. Select the maximizing label independently per block from one sampled coefficient vector, preserving deterministic ties and avoiding complete-profile enumeration.
3. Reuse the P1 controller objective, state machine, checkpoint, distributed, and journal paths without a second controller implementation; reuse the shared final-holdout reservation and manifest provenance without evaluating that holdout.
4. Verify controlled recovery of distinct block/granularity preferences, one-block behavior, arbitrary labels, uncertainty behavior, and the absence of duplicated scalar pair rewards.

### P3: Provenance and comparison safety

1. Add explicit Bayesian family/version/scope classification to metrics, summaries, checkpoints, and both modular and compatibility reporting helpers.
2. Reserve the existing `adaptive_per_block_thompson` classification for historical heuristic artifacts lacking Bayesian provenance; add distinct Bayesian global/per-block labels.
3. Keep UCB config/state/reward/resume/labels/styles unchanged and preserve random, nested-all, standalone, correction, monitoring, and nonadaptive checkpoint regressions.
4. Implement the separate post-training final-holdout evaluator and result artifact, requiring completed training and ordinary-validation-selected or explicitly supplied checkpoint provenance, without mutating controller or training decisions.
5. Add opt-in Bayesian smoke/pilot surfaces without changing the default comparison queue or making scientific-superiority claims.

## Verification Strategy

- **Config/migration**: matrix tests for both Bayesian scopes, invalid UCB global, missing/mixed legacy Thompson inputs, valid covariance forms, decision interval default, and arbitrary labels.
- **Mathematics**: closed-form controlled Gaussian predictions/updates, Q=0/Q>0, degenerate covariance, invalid numerical state, deterministic samples/ties, and feature dimension/identifiability.
- **Data**: fixed counts, stable source identities, all six empty intersections, role/source hashes, insufficient/zero-target examples, resume manifest mismatch, and no final evaluation call during training.
- **Boundary integration**: initial objective, ordered all-granularity component losses, uniform averaging, exclusion of training-batch loss, exactly `h` owned steps, exactly one evaluation per boundary, action fixedness, transactional failure, exact/inside-window checkpoints, incomplete terminal window, warmup exclusion, exact discrete fresh/resume equivalence, and numerical equivalence at `rtol=1e-6`, `atol=1e-8`.
- **Distributed**: fixed controller-panel partitioning without duplication or omission, target-token totals consistent with single-process evaluation, identical reduced controller objective within the specified tolerance, rank-zero sample/update, broadcast action/state, and rank-zero-only shared artifacts.
- **Logging**: rank-zero-only records for initial boundaries, completed windows, resumes, incomplete termination, and failures; required scalar and identity fields remain readable while posterior vectors and covariance matrices remain in structured artifacts.
- **Artifacts/reporting**: JSONL events and summary schema, full checkpoint state, compact metric fields, method labels, historical legacy fixture, non-resumable old Thompson, and unchanged UCB output.
- **Compatibility**: existing random global/per-block, nested-all, standalone, granularity resolution and model wiring, gradient-membership correction, baseline matching, monitoring, pilot matrix, and nonadaptive checkpoint suites.

## Complexity Tracking

No constitution violations require justification. The dense controller covariance and explicit phase state are the smallest representations that satisfy the specified Gaussian posterior and exact-resume method; they do not introduce a general framework.
