<!--
Sync Impact Report
Version change: unratified template -> 1.0.0
Modified principles:
- Template Principle 1 -> I. Research Code First
- Template Principle 2 -> II. Simplicity and Local Reasoning
- Template Principle 3 -> III. Explicit Experiment Flow
- Template Principle 4 -> IV. Minimal Abstraction and Validation
- Template Principle 5 -> V. Transparent Configuration and Reproducibility
- Added -> VI. Useful Outputs and Logging
Added sections:
- Research Code Constraints
- Development Workflow
Removed sections:
- Placeholder Section 2
- Placeholder Section 3
Templates requiring updates:
- .specify/templates/plan-template.md: updated
- .specify/templates/spec-template.md: updated
- .specify/templates/tasks-template.md: updated
- .specify/templates/commands/*.md: not present
Runtime guidance:
- README.md: reviewed, no conflicting guidance found
- AGENTS.md: reviewed, no conflicting guidance found
Follow-up TODOs: none
-->
# MatFormer Research Code Constitution

## Core Principles

### I. Research Code First
This repository is research code for MatFormer language modeling experiments.
Changes MUST optimize for clarity, iteration speed, experimental flexibility,
and ease of modification. Production-software patterns are not goals unless a
specific experiment requires them and the plan documents why.

Rationale: a researcher needs to be able to open a file, understand the logic
quickly, modify behavior safely, and trace an experiment end-to-end without
navigating a complex framework.

### II. Simplicity and Local Reasoning
Implementations MUST use straightforward control flow, descriptive names, short
functions, and explicit tensor-shape reasoning where shapes matter. Code SHOULD
keep related experiment logic close together when that improves readability.

Avoid clever compression, hidden side effects, unnecessary metaprogramming,
deep inheritance, premature optimization, and indirection that makes behavior
harder to trace.

Rationale: research code is read and changed repeatedly while ideas are still
moving. Local reasoning is more valuable than polished architecture.

### III. Explicit Experiment Flow
Experiment entry points, training loops, model variants, evaluation logic, and
ablations MUST remain visible and editable. New methods, baselines, and
architecture changes SHOULD be easy to compare without introducing registries
or dispatch layers that obscure what runs.

Duplicating one-off logic is acceptable when it keeps an experiment easier to
read than an abstraction would. Abstractions MUST emerge from repeated, stable
patterns rather than speculative reuse.

Rationale: experiment code needs to make the research path clear enough to inspect,
debug, and modify during active iteration.

### IV. Minimal Abstraction and Validation
The codebase MUST avoid public-API style defensive programming. Add argument
validation, type enforcement, runtime checks, or wrappers only when silent
failures are likely, tensor shapes are ambiguous, or debugging would otherwise
be difficult.

Generic factories, deep module hierarchies, and reusable frameworks MUST be
justified in the implementation plan when a direct implementation would work.

Rationale: excessive structure slows experiments and hides the behavior that
researchers need to change.

### V. Transparent Configuration and Reproducibility
Configurations MUST map directly to experiment concepts and avoid hidden
defaults. Prefer YAML, dataclasses, or simple dictionaries. Dynamic config
mutation, implicit magic, and complex registries require explicit justification.

Experiments MUST save the configuration used, log seeds when set, record dataset
identity or preprocessing assumptions, and write checkpoints when needed to
reproduce or inspect results.

Rationale: reproducibility matters, but it needs to support understanding rather
than bury experiment behavior under configuration machinery.

### VI. Useful Outputs and Logging
Every experiment MUST produce structured outputs that can be analyzed without
rerunning the experiment. At minimum, experiment outputs MUST include the
configuration used, scalar metrics, and a simple summary file such as CSV or
JSON. Plots, figures, and checkpoints SHOULD be produced when they are relevant
to comparison or later inspection.

Logging MUST help debugging and analysis through readable console output and
clear metrics. Avoid complicated callback systems or logging infrastructure
unless they make an active experiment easier to inspect.

Rationale: terminal logs alone are not enough for comparing runs, plotting
results, or auditing what happened after an experiment finishes.

## Research Code Constraints

Project structure SHOULD stay shallow and obvious. Prefer directories such as
`models/`, `experiments/`, `training/`, `evaluation/`, `utils/`, `configs/`,
and `scripts/` when the repository grows beyond a few files. Deeply nested
packages are allowed only when they simplify navigation for active experiments.

Comments SHOULD explain research motivation, mathematical intent, tensor-shape
assumptions, and non-obvious implementation choices. Comments MUST NOT merely
restate the code.

Generated experiment outputs SHOULD use simple, inspectable formats such as
CSV, JSON, PNG, PDF, and standard checkpoint files. Metrics MUST NOT live only
in terminal output.

## Development Workflow

Plans and reviews MUST check that new work preserves research-code readability,
keeps experiment flow traceable, and avoids unnecessary abstraction. When a
change adds complexity, the plan MUST explain why the simpler direct approach
is insufficient.

Feature specs for experiment-facing work MUST describe expected outputs,
configuration inputs, datasets or data assumptions, and reproducibility
information. Task lists MUST include implementation and verification work for
those outputs when relevant.

Validation SHOULD be focused on the failure modes most likely to waste research
time: incorrect tensor shapes, broken training or evaluation flow, missing
outputs, unreproducible configs, and unclear experiment comparisons.

## Governance

This constitution supersedes conflicting local conventions for Spec Kit plans,
specifications, and tasks in this repository. Amendments MUST be documented in
`.specify/memory/constitution.md` with a Sync Impact Report that lists affected
principles, templates, runtime guidance, and deferred follow-ups.

Versioning follows semantic versioning:
- MAJOR: removes or redefines core governance or principles in a backward
  incompatible way.
- MINOR: adds a principle, adds a governance section, or materially expands
  required research-code guidance.
- PATCH: clarifies wording, fixes typos, or makes non-semantic refinements.

Constitution compliance MUST be reviewed during `/speckit-plan` and rechecked
when tasks are generated or implementation choices introduce new complexity.
Any violation MUST be recorded with the reason it is needed and the simpler
alternative that was rejected.

**Version**: 1.0.0 | **Ratified**: 2026-05-12 | **Last Amended**: 2026-05-12
