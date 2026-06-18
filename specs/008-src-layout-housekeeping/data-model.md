# Data Model: Source Layout Housekeeping

## Entities

### Source Package

- **Purpose**: The importable production code that lives under `src/`.
- **Key attributes**:
  - Package name
  - On-disk module path
  - Responsibility
- **Relationships**:
  - Contains runtime modules, config helpers, metrics helpers, and reporting helpers.
  - Is referenced by thin root-level wrappers and by tests.

### Entrypoint Wrapper

- **Purpose**: A root-level script that preserves the current command interface while delegating to packaged code.
- **Key attributes**:
  - Script path
  - Delegated module/function
  - CLI arguments passed through unchanged
- **Relationships**:
  - Imports from the source package.
  - Must remain behavior-compatible with current commands.

### Runtime Module

- **Purpose**: A focused code unit that owns one training responsibility.
- **Key attributes**:
  - Module name
  - Responsibility boundary
  - Dependencies on training data, distributed runtime, or checkpointing
- **Relationships**:
  - Used by the training orchestration module.
  - Should not contain unrelated plotting or config logic.

### Figure Module

- **Purpose**: A focused code unit that owns one reporting or figure-generation responsibility.
- **Key attributes**:
  - Module name
  - Supported figure family or report
  - Dependencies on artifact loading and formatting
- **Relationships**:
  - Used by the figure-generation wrapper.
  - Should be reusable from tests and future reporting scripts.

### Config Module

- **Purpose**: A focused code unit that owns shared configuration resolution or validation logic.
- **Key attributes**:
  - Resolved config fields
  - Validation rules
  - Derived settings
- **Relationships**:
  - Used by training, evaluation, and artifact-writing code.
  - Must preserve existing configuration semantics.

### Metrics Module

- **Purpose**: A focused code unit that owns metrics serialization and artifact-summary logic.
- **Key attributes**:
  - Metrics row schema
  - Summary fields
  - CSV/JSON output helpers
- **Relationships**:
  - Used by training and reporting code.
  - Must preserve the current output formats.

## Validation Rules

- Root-level wrappers must not become the primary home for implementation logic.
- The current package names must remain stable so existing imports continue to resolve.
- Modules in the refactor scope should stay at or below 500 lines unless a small exception is clearly justified.
- Shared config and metrics responsibilities should be separated into smaller focused files where that improves readability.

## State Transitions

- **Monolith** -> **Focused module set**: large files are split by responsibility.
- **Root import tree** -> **`src/` package tree**: importable code is relocated without changing public package names.
- **Script implementation** -> **wrapper + package module**: CLI scripts delegate to importable code.
