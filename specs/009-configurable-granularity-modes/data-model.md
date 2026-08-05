# Data Model: Configurable Granularity Modes

## Entities

### GranularityModeDefinition

Represents the user-facing rule set that defines how a model’s granularities are specified.

- **mode**: `canonical` or `explicit`
- **ordered_labels**: ordered list of granularity names
- **prefix_fractions**: mapping from label to prefix fraction
- **source**: config section where the mode was declared
- **variant**: selected model variant, used for validation

### ResolvedGranularitySet

Represents the authoritative post-resolution granularity layout used by the rest of the system.

- **granularities**: ordered list of labels
- **granularity_prefixes**: resolved label-to-fraction map
- **prefix_widths**: derived integer prefix widths in the same order
- **intermediate_size**: model intermediate width used for validation
- **variant**: `slicing` or `concat`
- **alignment_status**: valid or invalid for the selected variant

### StandaloneGranularityRequest

Represents a standalone run’s requested granularity and its validation against the resolved set.

- **requested_label**: `run.granularity`
- **resolved_label**: the label chosen after validation
- **is_valid**: whether the request exists in the resolved set
- **error_reason**: explanation when the request is not valid

### ConcatAlignmentProfile

Represents the block-alignment check used for concat variants.

- **base_block_width**: smallest resolved prefix width
- **prefix_widths**: ordered cumulative widths for the configured granularities
- **block_widths**: per-step differences between adjacent prefix widths
- **alignment_valid**: whether all widths satisfy existing block alignment
- **failure_reason**: clear message when alignment is invalid

## Relationships

- `GranularityModeDefinition` resolves into exactly one `ResolvedGranularitySet`.
- `ResolvedGranularitySet` feeds training, validation, checkpointing, reporting, and adaptive sampling.
- `StandaloneGranularityRequest` is validated against `ResolvedGranularitySet`.
- `ConcatAlignmentProfile` is derived from `ResolvedGranularitySet` when the model variant is `concat`.

## Validation Rules

- Granularity labels must be non-empty and unique.
- Ordered labels must be preserved in the resolved configuration.
- Prefix fractions must be numeric, positive, and strictly increasing in order.
- The last prefix fraction must resolve to full intermediate width.
- Resolved widths must remain valid for the selected model variant.
- Concat layouts must either satisfy existing block alignment or fail before training begins.
- Standalone requests must match one of the resolved labels exactly.

## Derived Outputs

- Resolved config fields:
  - `model.granularity_mode`
  - `model.granularities`
  - `model.granularity_prefixes`
- Validation-time artifacts:
  - per-granularity widths
  - concat alignment status
  - clear failure messages when incompatible
