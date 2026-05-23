# Feature Specification: Cat Llama Granularity Pipeline

**Feature Branch**: `002-cat-llama`  
**Created**: 2026-05-21  
**Status**: Draft  
**Input**: User description: "I introduced an alternative method to create granularities through concatenation instead of slicing in '/home/nicolas.avila/dev/references/matformer/modified_llama.py:187' this is a method I want to integrate into our pipeline to compare agains our previous results. We can call this family of models cat_llama, it works exactly as matformer_llama with the exception that we manage different the granularities. I want this spec to be a concise one that allows me to run this version of the ModifiedLlamaForCausalLM (taking mlp_cls as an argument) through the same path as our previous experiments (maybe overriding an argument in the configs)"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Select Cat Llama Family (Priority: P1)

As a researcher, I want to select the `cat_llama` family through the same experiment entry point used for `matformer_llama` so I can run comparable experiments without changing the surrounding workflow.

**Why this priority**: This is the minimum change needed to compare the concatenation-based granularity variant against the existing baseline.

**Independent Test**: Run one experiment with `model.variant` set to `cat_llama` and confirm it completes through the existing pipeline path and records the selected variant in the resolved run metadata.

**Acceptance Scenarios**:

1. **Given** an existing MatFormer experiment configuration, **When** the researcher switches `model.variant` to `cat_llama`, **Then** the run uses the cat llama variant and otherwise follows the same experiment path.
2. **Given** no `model.variant` override, **When** the researcher launches the standard experiment, **Then** the default behavior remains the existing `matformer_llama` variant.

---

### User Story 2 - Compare Families Consistently (Priority: P2)

As a researcher, I want both `matformer_llama` and `cat_llama` runs to produce the same kind of outputs so I can compare results directly.

**Why this priority**: Comparison value depends on matching run structure, labels, and artifacts across both families.

**Independent Test**: Run one baseline-family experiment and one cat-llama experiment with the same dataset, seed, and reporting settings, then verify the output schema and comparison artifacts are aligned.

**Acceptance Scenarios**:

1. **Given** two runs with identical training and evaluation settings except for `model.variant`, **When** both complete, **Then** their summaries and metrics can be compared side by side.
2. **Given** a completed `cat_llama` run, **When** the researcher inspects the artifacts, **Then** the variant choice and granularity mode are clearly labeled.

---

### User Story 3 - Fail Fast on Invalid Selection (Priority: P3)

As a researcher, I want invalid model variants to fail early with a clear error so I do not waste time on unusable runs.

**Why this priority**: This protects experiment time and keeps comparisons trustworthy.

**Independent Test**: Launch the pipeline with an unsupported `model.variant` value and confirm it stops before training starts with a clear validation error.

**Acceptance Scenarios**:

1. **Given** an unsupported `model.variant` value, **When** the experiment is resolved, **Then** the run is rejected before execution begins.
2. **Given** a valid `model.variant` value, **When** the experiment is resolved, **Then** the run proceeds without requiring a separate script or alternate workflow.

### Edge Cases

- What happens when the family selector is omitted and the researcher expects the legacy path?
- How does the pipeline report a family selection that is valid in name but not supported by the current experiment configuration?
- What happens when two comparable runs differ only in the granularity construction method?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The experiment configuration MUST expose a `model.variant` selector that can choose between the existing `matformer_llama` variant and the new `cat_llama` variant, while `run.model_family` remains the topology selector for `nested` and `standalone`.
- **FR-002**: The experiment pipeline MUST construct the language model through the same run path for both families.
- **FR-003**: The selected variant MUST determine which MLP variant is used during model creation, while leaving the rest of the experiment settings unchanged.
- **FR-004**: `cat_llama` MUST preserve the existing experiment behavior except for granularity management based on concatenation rather than slicing.
- **FR-005**: Resolved run metadata and artifacts MUST record the selected variant and the granularity mode used for the run.
- **FR-006**: Existing `matformer_llama` runs MUST continue to work when the new family selector is not provided.
- **FR-007**: Invalid family selections MUST fail before training starts and provide a clear user-facing error.

### Research & Experiment Requirements

- **EX-001**: A single configuration override MUST be sufficient to switch between `matformer_llama` and `cat_llama`.
- **EX-002**: The experiment output for both families MUST use the same artifact categories needed for comparison.
- **EX-003**: The completed run MUST save the resolved configuration used for the selected family.
- **EX-004**: The completed run MUST label comparison outputs so a reviewer can tell which variant produced each result.
- **EX-005**: The feature MUST preserve the existing baseline path so previous experiments remain reproducible.
- **EX-006**: The resolved run configuration MUST record the learning-rate scale rule, resolved learning rate, warmup policy, and optimizer settings used for the run so distributed debugging remains auditable.

### Key Entities *(include if feature involves data)*

- **Model Variant**: The experiment-level choice between the existing matformer variant and the concatenation-based cat llama variant.
- **Run Configuration**: The resolved experiment inputs that control family selection, granularity handling, and reporting.
- **Run Summary**: The saved record that identifies the selected family and supports later comparison.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A researcher can switch from `matformer_llama` to `cat_llama` with one configuration change and complete the same experiment entry point.
- **SC-002**: 100% of completed runs record the selected variant and granularity mode in the resolved configuration or run summary.
- **SC-003**: At least one `cat_llama` run and one `matformer_llama` run can be produced with the same reporting structure and compared directly.
- **SC-004**: Runs started without the new selector continue to resolve to the existing `matformer_llama` behavior in validation testing.
- **SC-005**: Invalid family values fail before training begins in 100% of validation cases.

## Assumptions

- The existing experiment entry point remains the primary way to launch runs.
- `cat_llama` is a comparison-oriented variant, not a new training objective.
- The only intended behavioral difference is how granularities are assembled and managed.
- Existing datasets, metrics, and artifact formats remain valid for both families.
