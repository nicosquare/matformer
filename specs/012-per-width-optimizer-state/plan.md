# Implementation Plan: Per-Width Optimizer State

**Branch**: `012-per-width-optimizer-state` | **Date**: 2026-08-31 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/012-per-width-optimizer-state/spec.md`

**Note**: This plan implements one opt-in optimizer-state intervention over the
existing global-width training path. It keeps model weights, actions, data,
losses, and the global learning-rate schedule shared and changes only whether
optimizer moments/counters are shared or isolated by resolved global width.

## Summary

Add `shared` (default) and `per_granularity` optimizer-state scopes to the
existing AdamW/SGD configuration. Per-granularity scope creates one ordered
optimizer state machine per resolved global width over the same model
parameters, selects exactly the optimizer owned by the existing global action
once per accumulation window, and advances one global scheduler clock for every
committed step. A focused optimizer-state module owns the collection, global
learning-rate synchronization, counts, and versioned state; the main loop keeps
the action → backward → selected optimizer → global scheduler → accounting flow
explicit.

Extend resumable checkpoints with a purpose-aware, strictly validated ordered
optimizer collection while treating historical scope omission as shared only.
Add compact ownership fields and resource costs to existing metrics/summaries,
provide the frozen TinyStories-Instruct paired recipe, and add a dedicated
freeze/report analyzer for terminal-checkpoint comparisons. Per-granularity
scope is single-process only; all existing shared and distributed paths remain
unchanged.

## Technical Context

**Language/Version**: Python 3.12  
**Primary Dependencies**: PyTorch 2.11, Transformers 5.8, datasets, PyYAML, NumPy, pandas, pytest; no new runtime dependency  
**Storage**: YAML resolved configuration; CSV ordinary metrics; JSON run summaries, manifests, final-holdout results, and paired reports; PyTorch resumable/model-only checkpoints  
**Testing**: pytest unit, configuration/preflight, CPU end-to-end, deterministic resume, artifact/reporting, balanced-cycle, accumulation, compatibility, and distributed-rejection tests  
**Target Platform**: Linux research workstation or single-GPU cluster; per-granularity scope requires one process, while shared scope retains current CPU/GPU and FSDP support  
**Project Type**: Research language-model training pipeline and controlled experiment tooling  
**Experiment Scope**: Paired optimizer-state ablation over existing nested global-width sampling; no model-weight, sampling-policy, objective, correction, or scheduler-policy change  
**Datasets/Data Assumptions**: Reuse the prepared TinyStories-Instruct deterministic four-role corpus, immutable manifests, packed-mmap optimizer data, ordinary validation, fixed controller role, and sealed final holdout  
**Configuration Inputs**: Existing YAML and dotted overrides; `training.optimizer.state_scope=shared|per_granularity`, fixed `training.optimizer.scheduler_clock=global_step`, existing optimizer/scheduler/global sampling/warmup fields  
**Experiment Outputs**: Resolved scope contract; compact per-step ownership/count fields; run summary resource and reconciliation fields; complete resumable optimizer collection; frozen six-run manifest; paired JSON/CSV analysis; existing explicit final-holdout results  
**Reproducibility Notes**: No new RNG; reuse global action/controller/data RNG and balanced-cycle state; full run/checkpoint identity includes scope, paired-control identity excludes only scope; exact ordered state and action/count reconciliation on resume  
**Performance Goals**: Preserve one ordinary forward/backward and one model optimizer step per global step; add `O(K)` optimizer state memory/checkpoint storage for `K` widths; report rather than cap wall time, peak memory, and checkpoint-size overhead  
**Constraints**: One global owner per accumulation window; shared model tensors; absent-gradient inactivity; one global scheduler clock; no cross-scope state migration; no distributed per-granularity execution; no per-step full model/optimizer rollback snapshots  
**Scale/Scope**: General ordered global width sets of at least two labels for AdamW/SGD; frozen four-width d64/l4 recipe; six pilot runs at 87,132 steps and optional six fresh confirmation runs at 261,396 steps

No technical context item remains unresolved.

## Constitution Check

*GATE: Evaluated before Phase 0 research and re-evaluated after Phase 1 design.*

- **Research code first**: Pass. The change is one explicit experiment and a
  small analysis tool; it does not introduce a service layer or production
  framework.
- **Simplicity and local reasoning**: Pass. One focused optimizer-state module
  owns collection mechanics, while the training loop shows the selected owner
  and commit order directly.
- **Explicit experiment flow**: Pass. Existing action selection, accumulation,
  clipping, optimizer commit, scheduler advance, exposure, checkpoint, and
  reporting paths are extended in place rather than hidden behind a registry.
- **Minimal abstraction and validation**: Pass. The new collection and global
  clock exist because loose mappings would scatter ordering/checkpoint logic.
  Validation is limited to combinations or state mismatches that could silently
  invalidate experiments.
- **Transparent configuration and reproducibility**: Pass. Scope and clock map
  directly to research concepts; the default is historical shared behavior;
  configs, ordered states, controls, manifests, counts, and checkpoint hashes
  are recorded.
- **Useful outputs and logging**: Pass. Existing CSV/JSON/checkpoint surfaces
  gain compact ownership and resource fields, and the paired analyzer emits
  structured manifest and result artifacts.
- **Shallow organization**: Pass. One new training module, one controlled config,
  one analysis script, focused tests, and direct extensions to current files
  preserve the repository layout.

No constitution gate is violated.

## Project Structure

### Documentation (this feature)

```text
specs/012-per-width-optimizer-state/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── configuration.md
│   ├── optimizer-lifecycle-and-checkpoint.md
│   ├── experiment-and-analysis.md
│   └── cli-entrypoints.md
└── tasks.md                              # created later by /speckit-tasks
```

### Source Code (repository root)

```text
train.py                                  # preflight scope/clock contract
configs/controlled_exps/
└── tinystories_instruct_per_width_optimizers.yaml
docs/
└── tinystories-per-width-optimizer-experiment.md
scripts/
└── analyze_tinystories_per_width_optimizer.py
src/
├── training/
│   ├── optimizer_state.py                # ordered collection, global clock, state validation
│   ├── steps.py                          # explicit action owner and commit lifecycle
│   ├── run.py                            # build/restore, completion reconciliation, summaries
│   ├── warmup.py                         # existing one-owner forced actions reused
│   └── checkpointing.py                  # purpose, collection save/load, historical compatibility
├── utils/
│   ├── config.py                         # scope/clock defaults and eligibility matrix
│   ├── reproducibility.py                # full identity vs paired-control identity
│   └── metrics.py                        # compact ownership/resource fields
└── evaluation/
    ├── reporting.py
    ├── reporting_io.py
    └── reporting_impl.py                 # scope-aware grouping and historical defaults
tests/
├── fixtures/
│   └── per_granularity_optimizer_smoke.yaml
├── test_per_granularity_optimizer.py
├── test_per_granularity_optimizer_resume.py
├── test_config.py
├── test_train_cli.py
├── test_training_smoke.py
├── test_global_sampling_windows.py
├── test_accumulation.py
├── test_artifacts.py
├── test_reproducibility.py
├── test_reporting.py
└── test_distributed.py
```

**Structure Decision**: Keep optimizer-state-specific runtime and schema logic
in one shallow `optimizer_state.py`. Extend the current training, checkpoint,
metrics, and reporting modules directly. Do not create a generic optimization
strategy registry, a second trainer, or a new evaluation framework. The paired
analysis script is separate because existing portfolio analysis has a different
five-checkpoint-per-seed selection contract.

## Phase 0: Research Decisions

Research is complete in [research.md](./research.md). Principal decisions:

1. Accept nested scope/clock inputs but normalize them outside the historical
   resolved optimizer mapping to preserve shared signatures and checkpoints.
2. Keep shared construction unchanged; use one focused ordered optimizer
   collection for per-granularity state over the same model tensors.
3. Use absent gradients as the inactivity boundary and preserve ordinary
   present-gradient AdamW/SGD behavior.
4. Reuse the existing once-per-accumulation-window global action as the sole
   optimizer owner.
5. Use one explicit global scheduler clock and synchronize its current learning
   rate to every width optimizer.
6. Treat successful return from the selected optimizer as the irreversible
   commit boundary; pre-commit failures roll back existing transactional state,
   and post-commit failures retain the prior durable checkpoint.
7. Reuse current RNG and global/balanced/adaptive sampling state; add no
   optimizer ownership RNG.
8. Add versioned, purpose-aware ordered checkpoint state and validate it fully
   before any load mutation; historical omission means shared only.
9. Separate full intervention identity from an arm-invariant paired-control
   signature that excludes only state scope.
10. Extend existing compact metrics/run summaries with ownership, count,
    reconciliation, and resource fields.
11. Add the frozen recipe and a dedicated freeze/report analyzer while reusing
    the explicit terminal final-holdout evaluator.
12. Reject per-granularity scope under multi-process distributed execution.

All technical unknowns are resolved; no `NEEDS CLARIFICATION` markers remain.

## Phase 1: Design and Contracts

Design artifacts are complete:

- [data-model.md](./data-model.md) defines the resolved scope contract, ordered
  optimizer collection, width entries, global clock, commit accounting,
  checkpoint purpose/payload, metric/summary extensions, and paired artifacts.
- [contracts/configuration.md](./contracts/configuration.md) defines inputs,
  defaults, normalized compatibility shape, eligibility matrix, preflight,
  paired identity, and exact controlled recipe.
- [contracts/optimizer-lifecycle-and-checkpoint.md](./contracts/optimizer-lifecycle-and-checkpoint.md)
  defines construction, accumulation ownership, gradient semantics, commit and
  failure order, completion reconciliation, purpose-aware checkpointing,
  mutation-free resume validation, and historical behavior.
- [contracts/experiment-and-analysis.md](./contracts/experiment-and-analysis.md)
  defines paired controls, budgets, terminal checkpoint selection, manifest
  freeze, holdout policy, endpoints, claim labels, and structured outputs.
- [contracts/cli-entrypoints.md](./contracts/cli-entrypoints.md) defines normal
  train/preflight, corpus audit, explicit final-holdout evaluation, and new
  freeze/report commands.
- [quickstart.md](./quickstart.md) orders focused tests, CPU smoke/resume,
  TinyStories profile/audit, six-run preflight, pilot freeze, and optional fresh
  confirmation.

### Post-design constitution re-check

- The shared optimizer/scheduler path remains direct and unchanged unless the
  resolved scope opts into the collection.
- The per-granularity commit stays visible in `train_for_steps`; the focused
  module does not choose actions or hide training flow.
- The global scheduler carrier is explicitly documented and owns no model state;
  it avoids duplicating scheduler mathematics or creating width-local clocks.
- Strict checkpoint validation is justified because scope/order/state mismatch
  would silently change the scientific intervention.
- The extra metrics are compact; full states remain only in resumable
  checkpoints, and resource overhead is reported.
- The analysis script freezes existing run artifacts and does not introduce a
  general experiment database or reporting framework.

All constitution gates continue to pass after design.

## Implementation Sequence

1. **Resolve configuration and identity**: parse scope/clock inputs, preserve
   the historical optimizer mapping, create the resolved optimizer-state
   contract, add eligibility/distributed gates, expose preflight identity, and
   split full versus paired-control signatures.
2. **Implement optimizer collection and global clock**: build ordered AdamW/SGD
   instances over shared parameters, select one owner, synchronize global rates,
   track counts/last owner, and serialize/validate the collection.
3. **Integrate the explicit training lifecycle**: resolve the owner immediately
   after existing action selection, reuse it for all accumulation microbatches,
   commit only that optimizer, advance the global clock once, and update counts
   beside existing exposure accounting; reuse both warmup policies.
4. **Persist exact resumable state**: add checkpoint purpose, scope metadata,
   ordered optimizer entries, current rates/counts, mutation-free validation,
   historical shared interpretation, model-only rejection, and post-commit
   failure durability rules.
5. **Extend audit artifacts**: add compact metric columns, final summary counts,
   scheduler reconciliation, terminal wall time/peak memory/checkpoint
   path-size-hash, monitoring identity, and scope-aware reporting defaults.
6. **Add the controlled experiment**: create the exact four-width
   TinyStories-Instruct recipe, keep shared as default, and align the operational
   runbook with resolved names and analyzer workflow.
7. **Implement paired freeze/report**: validate six runs and immutable terminal
   artifacts, write the frozen manifest, consume explicit holdout results, emit
   primary/secondary/resource JSON and CSV, and enforce diagnostic/confirmatory
   labels.
8. **Verify and protect compatibility**: add focused collection/resume tests and
   extend config, CLI, smoke, accumulation, balanced-cycle, artifact,
   reproducibility, reporting, and distributed suites.

## Verification Strategy

- **Configuration/defaults**: omitted scope/clock resolve to historical shared
  and global-step behavior; valid global modes pass; standalone, nested-all,
  per-block, one-width, invalid clock, and distributed per-granularity fail.
- **AdamW/SGD isolation**: different shared-parameter gradients create distinct
  moments/counters; only selected state changes; wider-only parameters remain
  untouched and absent under narrow steps.
- **Global schedule**: cosine, constant, and warmup-stable-decay traces match the
  shared arm step by step; unequal width exposure does not alter rate; all width
  param groups synchronize after build, commit, and resume.
- **Accumulation/warmup**: one action and owner span each complete window;
  `full_only` and balanced-global warmup actions advance their sole owner;
  future ambiguous warmups reject.
- **Failure**: failures before optimizer return preserve all optimizer states,
  scheduler, counts, exposures, action RNG, and data cursor and clear gradients;
  post-commit failures never overwrite the previous reconciled checkpoint.
- **Resume**: uninterrupted and resumed execution inside and at balanced-cycle
  boundaries match actions exactly and optimizer/scheduler/model numeric state
  within existing tolerance; missing, extra, reordered, non-finite, wrong scope,
  family, kwargs, scheduler, topology, and model-only states reject before load.
- **Accounting/artifacts**: every training row identifies scope and owner;
  adjacent rows show one count advance; completion reconciles updates,
  exposures, global steps, and scheduler position; summaries expose terminal
  wall time, peak memory, and resumable checkpoint size/hash.
- **Controlled recipe**: all six pilot preflights resolve exactly 87,132 steps,
  21,783 updates per width, one process, matched controls, and the requested
  scope; confirmation resolves 261,396 and 65,349 from fresh identities.
- **Analysis/holdout**: freeze rejects any paired mismatch; report validates
  hashes and endpoints, never opens the holdout, labels pilot diagnostic, and
  permits confirmatory status only for the complete untouched-holdout protocol.
- **Compatibility**: existing shared optimizers/checkpoints/signatures,
  Thompson/UCB/PanelGrad/random action policies, nested-all/standalone,
  correction, monitoring, final-holdout, distributed shared training, and
  reporting retain expected behavior.

## Complexity Tracking

No constitution violations require justification.
