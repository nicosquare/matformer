# Research: Source Layout Housekeeping

## Decision 1: Use a `src/` layout with editable installation support

- **Decision**: Add minimal packaging metadata so the repository imports from `src/` in a normal developer setup.
- **Rationale**: It makes import resolution match the final source layout and avoids ad hoc `PYTHONPATH` or `sys.path` hacks.
- **Alternatives considered**:
  - Set `PYTHONPATH=src` in scripts and tests. Rejected because it hides packaging mistakes and makes the developer workflow fragile.
  - Inject `src/` into `sys.path` in multiple entrypoints. Rejected because it spreads path management across the codebase.

## Decision 2: Preserve the current package names

- **Decision**: Keep `models`, `training`, `evaluation`, and `utils` as the importable package names under `src/`.
- **Rationale**: This minimizes churn in the existing tests and internal imports while still giving the repo a clean source root.
- **Alternatives considered**:
  - Introduce a new top-level namespace. Rejected because it would force a broad rename without solving the real readability problem.
  - Keep code at the repository root. Rejected because it preserves the current layout confusion.

## Decision 3: Split by responsibility, not by artificial layering

- **Decision**: Decompose the large modules into focused helpers around training orchestration, checkpointing, warmup, config resolution, metrics I/O, and figure/report generation.
- **Rationale**: These are the actual seams already visible in the codebase and they match the housekeeping goal of keeping files under 500 lines.
- **Alternatives considered**:
  - Create a generic framework layer or registry. Rejected because it adds indirection that is not needed for this research code.
  - Leave the oversized modules intact and rely on comments. Rejected because it does not materially improve readability.

## Decision 4: Keep wrapper entrypoints thin and stable

- **Decision**: Preserve `train.py` and `scripts/make_figures.py` as command-compatible wrappers.
- **Rationale**: Existing commands continue to work while the implementation moves into importable modules under `src/`.
- **Alternatives considered**:
  - Replace the entrypoints with new command names. Rejected because it breaks current workflows for no gain.

## Decision 5: Put reusable report logic in `src/evaluation/`

- **Decision**: Move figure-generation helpers into importable reporting modules under `src/evaluation/`.
- **Rationale**: The existing `evaluation/` package already owns experiment analysis and validation logic, so reporting is a natural extension of that area.
- **Alternatives considered**:
  - Create a new `reporting/` package. Rejected because it adds another top-level concept without a strong need.
  - Leave helper logic in `scripts/make_figures.py`. Rejected because it keeps the script monolithic.
