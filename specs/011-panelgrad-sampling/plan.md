# Implementation Plan: PanelGrad Sampling

**Branch**: `011-panelgrad-sampling` | **Date**: 2026-08-14 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/011-panelgrad-sampling/spec.md`

**Note**: This plan implements only PanelGrad. It reuses stable controller-data and global-training infrastructure without creating a generic sampling-policy framework or changing Thompson, UCB, random, nested-all, or standalone semantics.

## Summary

Add PanelGrad as `adaptive_global + panelgrad`. At the first adaptive action and every 50 completed PanelGrad steps by default, measure each resolved global granularity's raw aggregate controller-loss gradient over only its granularity-controlled FFN parameters. Convert the contemporaneous RMS scores into `q`, mix with the uniform exploration floor to obtain `p`, freeze `p` for the interval, and draw one global action from `Categorical(p)` before every optimizer step.

Implementation keeps the research flow visible: immediately before ordinary action selection, one PanelGrad block refreshes if due and samples a global action; immediately after a successful optimizer commit, it records exposure and interval progress. A focused `src/training/panelgrad.py` owns measurement math, categorical RNG, and compact state. Existing controller/final-holdout roles, balanced warmup, global action shape, forward/backward path, distributed primitives, checkpoint flow, and controller artifact names are extended directly.

## Technical Context

**Language/Version**: Python 3.12
**Primary Dependencies**: PyTorch, transformers, datasets, PyYAML, NumPy, pandas, pytest; no new runtime dependency
**Storage**: YAML configuration; JSON manifests/config/summary; append-only JSONL controller events; CSV ordinary metrics; PyTorch checkpoints
**Testing**: pytest unit, CPU integration, deterministic resume, artifact, compatibility, and distributed gloo/FSDP tests
**Target Platform**: Linux research workstation or single-node CPU/GPU cluster; current single-process and 1-4 process FSDP paths
**Project Type**: Research language-model training pipeline
**Experiment Scope**: One global elastic-training sampling method; slicing and concat MatFormer FFNs; no per-block policy or alternate estimator
**Datasets/Data Assumptions**: Reuse the fixed 128-example controller role, ordinary validation, optimizer training, and reserved 512-example final holdout from the existing deterministic four-role split; support raw tokenized and prepared packed-mmap data
**Configuration Inputs**: Existing YAML and dotted overrides; `adaptive_global`, `panelgrad`, resolved granularities, optional balanced warmup, and `model.panelgrad.{refresh_interval_steps,eta,temperature,epsilon}`
**Experiment Outputs**: Resolved config and role manifests; `controller_metrics.jsonl`; `controller_summary.json`; compact PanelGrad fields in `metrics.csv`; PanelGrad checkpoint state; run summary; opt-in PanelGrad and matched uniform-baseline configurations; structured `outputs/panelgrad-comparison/comparison.json` comparison results
**Reproducibility Notes**: Existing strict runtime and role seeds; new independent `panelgrad_sampling` stream; rank-zero categorical authority; complete refresh phase and RNG state persisted; same-topology comparisons use `rtol=1e-6`, `atol=1e-8` for numeric state and exact discrete actions
**Performance Goals**: One ordinary training forward/backward per optimizer step; exactly `K` full controller-panel gradient measurements per refresh; `O(K)` probability construction and sampling; report controller examples, target tokens, backward count, duration, and cumulative refresh cost separately
**Constraints**: Raw pre-correction gradients; evaluation mode with autograd; granularity-controlled FFN support only; one granularity measured at a time; no full-model parameter summon; no measurement optimizer effects; no inverse weighting or compute correction
**Scale/Scope**: Arbitrary nonempty ordered granularity labels, current slicing/concat layouts and model sizes, fixed 128-example controller set, positive configurable refresh interval, exact inside-interval/boundary resume, and supported distributed world sizes

## Constitution Check

*GATE: Evaluated before research and re-evaluated after Phase 1 design.*

- **Research code first**: Pass. PanelGrad is one opt-in experiment with its equations, support definition, refresh, and sampling sequence visible; no service or production framework is introduced.
- **Simplicity and local reasoning**: Pass. One focused module owns the small method state and math, while the training loop contains a plainly commented refresh/sample block and a post-commit record call.
- **Explicit experiment flow**: Pass. Warmup, refresh, categorical action, ordinary training, successful-step accounting, and terminal handling remain direct calls in the existing orchestration and loop.
- **Minimal abstraction and validation**: Pass. New validation is limited to invalid scientific inputs, non-finite measurement/probability state, data/checkpoint incompatibility, and tensor-support mistakes that could silently invalidate results.
- **Transparent configuration and reproducibility**: Pass. All policy values map directly to the equations; defaults, role hashes, support counts, RNG, lifecycle phase, and resume state are saved.
- **Useful outputs and logging**: Pass. Full refresh records use the existing controller JSONL/summary surfaces; ordinary rows contain only compact action/progress diagnostics; measurement cost is explicit.
- **Shallow organization**: Pass. One new training module, one opt-in config, focused tests, and direct extensions to existing files preserve the repository layout.

No constitution gate is violated.

## Project Structure

### Documentation (this feature)

```text
specs/011-panelgrad-sampling/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── configuration.md
│   ├── lifecycle-and-artifacts.md
│   └── cli-entrypoints.md
└── tasks.md                         # created later by /speckit-tasks
```

### Source Code (repository root)

```text
train.py
configs/opt-in_exps/
├── panelgrad_smoke.yaml             # explicit opt-in experiment
└── panelgrad_uniform_baseline.yaml  # matched uniform-global comparison
outputs/
└── panelgrad-comparison/
    └── comparison.json              # structured matched-budget results
src/
├── models/
│   └── ffn.py                       # controlled FFN support/count and correction suspension
├── training/
│   ├── panelgrad.py                 # measurement, probability math, RNG, state, validation
│   ├── data.py                      # existing four-role split activated for PanelGrad
│   ├── run.py                       # setup, replicated panel loader, artifact orchestration
│   ├── steps.py                     # explicit pre-action refresh/sample and post-commit record
│   ├── warmup.py                    # existing balanced-global lifecycle reused
│   ├── checkpointing.py             # dedicated PanelGrad schema and resume validation
│   └── distributed.py               # existing reduction/broadcast/FSDP primitives reused
├── evaluation/
│   ├── reporting.py
│   ├── reporting_io.py
│   └── reporting_impl.py            # explicit PanelGrad provenance/classification
└── utils/
    ├── config.py                    # strategy/config resolution and preflight
    ├── reproducibility.py           # panelgrad_sampling seed stream and signatures
    └── metrics.py                   # refresh journal, summary, compact columns
tests/
├── fixtures/
│   └── panelgrad_smoke.yaml
├── test_panelgrad.py                # score/support/probability/state unit tests
├── test_panelgrad_resume.py
├── test_training_smoke.py
├── test_matformer_prefixes.py
├── test_data_validation.py
├── test_distributed.py
├── test_config.py
├── test_artifacts.py
└── test_reporting.py
```

**Structure Decision**: Keep all method-specific mathematics and state in one shallow `panelgrad.py`. Extend the current training loop directly instead of adding a strategy registry or adapting the Bayesian controller. Reuse the established role manifests and controller artifact filenames because one run selects only one adaptive method; distinguish contents through explicit PanelGrad family/version provenance.

## Phase 0: Research Decisions

Research is complete in [research.md](./research.md). The principal decisions are:

1. Resolve PanelGrad as `adaptive_global + panelgrad` with its own `model.panelgrad` mapping and state schema.
2. Reuse the existing four disjoint data roles and final-holdout separation; use a PanelGrad-specific replicated controller loader under FSDP to keep backward collectives aligned.
3. Differentiate one target-token-weighted aggregate controller loss per granularity and never average microbatch gradient norms.
4. Define both `d_g` and `N_g` from the resolved FFN prefix/block support only; exclude shared down bias and all non-controlled parameters.
5. Suspend membership-correction hooks only inside the measurement context, then restore correction, RNG, model mode, granularity state, and empty gradients in `finally`.
6. Use float64 score/probability arithmetic and a stable equivalent of the specified power normalization; rank zero samples with a dedicated CPU generator and broadcasts the action/state.
7. Refresh before action selection, sample once per optimizer step, and advance exposure/interval state only after a successful optimizer commit.
8. Store PanelGrad state separately from Bayesian posterior and legacy UCB state; validate exact config, roles, support, phase, journal, and RNG on resume.
9. Reuse `controller_metrics.jsonl` and `controller_summary.json` with PanelGrad event schemas and explicit provenance; keep large vectors out of ordinary metrics rows.
10. Validate the method against uniform global sampling at matched steps/tokens while reporting full-panel measurement cost separately.

All technical unknowns are resolved; no `NEEDS CLARIFICATION` markers remain.

## Phase 1: Design and Contracts

Design artifacts are complete:

- [data-model.md](./data-model.md) defines configuration, controlled FFN support, refresh measurements, probability snapshots, sampling state, lifecycle state, actions, events, and summaries.
- [contracts/configuration.md](./contracts/configuration.md) defines the YAML/override surface, strategy matrix, defaults, data roles, warmup reuse, validation, and preflight identity.
- [contracts/lifecycle-and-artifacts.md](./contracts/lifecycle-and-artifacts.md) defines aggregate gradient measurement, FSDP behavior, refresh/action ordering, rollback, checkpoints, and structured artifacts.
- [contracts/cli-entrypoints.md](./contracts/cli-entrypoints.md) defines existing training, preflight, final-holdout, and opt-in experiment behavior.
- [quickstart.md](./quickstart.md) defines focused validation commands and the initial comparison workflow.

### Post-design constitution re-check

- The design keeps one visible method block in `steps.py`; it does not introduce a registry, generic policy interface, or reuse the TS posterior state machine.
- The focused PanelGrad module is justified by transactional RNG/checkpoint state and gradient-support mathematics that would obscure the ordinary training loop if inlined.
- The only model-level extension exposes existing FFN layout facts and a scoped correction suspension; it does not change ordinary forward or training behavior.
- Replicating the small fixed controller panel under FSDP deliberately trades extra refresh compute for matched collectives and simple, auditable semantics; the cost is recorded.
- Data, config, state, event, and reporting identities remain explicit and versioned. Existing methods are protected by compatibility tests.

All constitution gates continue to pass after design.

## Implementation Sequence

1. **Resolve identity and data**: add the PanelGrad strategy/config mapping and defaults; activate the existing four-role split, manifests, seed provenance, preflight, comparison signature, and balanced warmup for PanelGrad without changing TS/UCB resolution.
2. **Expose exact FFN support**: add slicing/concat support descriptors and counts that exclude granularity-independent parameters; make existing correction hooks honor a scoped suspension flag; test counts, slices, zero gradients, and restoration.
3. **Implement PanelGrad math/state**: add aggregate panel-gradient measurement, float64 RMS and stable `q/p` construction, phase transitions, dedicated categorical generator, transaction snapshots, exposure accounting, and strict state validation.
4. **Integrate the explicit loop lifecycle**: build/restore PanelGrad around warmup; in `train_for_steps`, refresh and sample immediately before existing action selection, reuse the global action and forward/backward path, then record only successful commits; synchronize rank-zero state/action.
5. **Persist and report**: extend checkpoint save/load, controller event journal/summary, compact metrics, run summary, method classification, failure records, and exact resume reconciliation.
6. **Verify the experiment**: add unit, smoke, warmup, isolation, failure rollback, exact-boundary/inside-interval resume, FSDP agreement, artifact, reporting, and existing-strategy compatibility tests; add one opt-in smoke config and document the matched-step uniform comparison.

## Verification Strategy

- **Configuration**: valid `adaptive_global + panelgrad`; invalid scope/strategy combinations; defaults and overrides; exact role contracts; balanced warmup interval fallback; unchanged TS/UCB/random resolution.
- **Support and math**: slicing and concat controlled support; arbitrary labels/counts; shared-bias exclusion; stable `N_g`; aggregate-gradient microbatch invariance; raw correction bypass; all-zero scores; one arm; `epsilon` 0/1; temperature; finite and normalization checks.
- **Isolation**: measurement preserves parameters, optimizer, scheduler, mixed-precision state, training-data cursor, prior model/granularity mode, and ordinary RNG; gradients are `None` afterward on success and failure.
- **Lifecycle**: initial/post-warmup refresh; frozen `p` for exactly `H` successful steps; one categorical action per step; failed update rolls back sample RNG and does not increment exposure; terminal exact/partial intervals cause no unused refresh.
- **Resume**: inside interval, refresh-pending boundary, warmup, and terminal states; exact action sequence and exposures; numeric state within `rtol=1e-6`, `atol=1e-8`; reject config, support, method, granularity, RNG, journal, and manifest mismatches.
- **Distributed**: replicated fixed-panel batching; matched backward counts; per-layer FSDP full-gradient support extraction without full-model summon; rank agreement on scores/probabilities; rank-zero draw/broadcast; one shared artifact copy.
- **Artifacts/reporting**: transactional refresh/failure events, summary hash/provenance, compact per-step probability/exposure fields, separate measurement cost, explicit PanelGrad classification, untouched final holdout.
- **Compatibility**: current Thompson fixed-window behavior, UCB state/reward, random global/per-block, nested-all, standalone, correction, checkpointing, monitoring, and default pilot queue remain unchanged.

## Complexity Tracking

No constitution violations require justification.
