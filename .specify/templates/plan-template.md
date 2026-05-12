# Implementation Plan: [FEATURE]

**Branch**: `[###-feature-name]` | **Date**: [DATE] | **Spec**: [link]
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

[Extract from feature spec: primary requirement + technical approach from research]

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. Keep details explicit and close to the experiment concepts.
-->

**Language/Version**: [e.g., Python 3.11, Swift 5.9, Rust 1.75 or NEEDS CLARIFICATION]  
**Primary Dependencies**: [e.g., FastAPI, UIKit, LLVM or NEEDS CLARIFICATION]  
**Storage**: [if applicable, e.g., PostgreSQL, CoreData, files or N/A]  
**Testing**: [e.g., pytest, XCTest, cargo test or NEEDS CLARIFICATION]  
**Target Platform**: [e.g., Linux server, iOS 15+, WASM or NEEDS CLARIFICATION]  
**Project Type**: [e.g., research script/model change/training pipeline/evaluation tool or NEEDS CLARIFICATION]  
**Experiment Scope**: [model variant, baseline, ablation, data change, evaluation change, or N/A]  
**Datasets/Data Assumptions**: [datasets, preprocessing, synthetic data, or NEEDS CLARIFICATION]  
**Configuration Inputs**: [YAML/dataclass/dict/CLI args and key experiment parameters]  
**Experiment Outputs**: [metrics CSV/JSON, plots, checkpoints, logs, or N/A]  
**Reproducibility Notes**: [seeds, checkpoint policy, config saving, environment assumptions]  
**Performance Goals**: [research-specific, e.g., target perplexity, training throughput, memory budget or N/A]  
**Constraints**: [domain-specific, e.g., GPU memory, runtime budget, dataset access or NEEDS CLARIFICATION]  
**Scale/Scope**: [domain-specific, e.g., model sizes, token counts, run count, GPUs or NEEDS CLARIFICATION]

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Research code first**: Experiment flow remains visible and editable; any
  production-style framework, service layer, or heavy indirection is justified.
- **Simplicity and local reasoning**: Logic is straightforward, names are
  descriptive, tensor shapes are explicit where relevant, and behavior can be
  traced from entry point to outputs.
- **Minimal abstraction and validation**: New abstractions, factories,
  registries, wrappers, or broad validation are included only when a direct
  implementation would make debugging harder.
- **Transparent configuration and reproducibility**: Config values map to
  experiment concepts; configs, seeds, dataset assumptions, and checkpoints are
  recorded as needed.
- **Useful outputs and logging**: Experiment-facing work writes structured
  metrics and summaries, with plots/checkpoints when relevant. Metrics do not
  live only in terminal logs.
- **Shallow organization**: New files use obvious locations and avoid deep
  nesting unless the plan explains why it improves research iteration.

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., models/, training/, evaluation/, configs/). The delivered plan must
  not include Option labels.
-->

```text
# [REMOVE IF UNUSED] Minimal flat research layout
train.py
modified_llama.py
configs/
outputs/

# [REMOVE IF UNUSED] Shallow research layout
models/
experiments/
training/
evaluation/
utils/
configs/
scripts/
outputs/
tests/
```

**Structure Decision**: [Document the selected structure and reference the real
directories captured above]

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
