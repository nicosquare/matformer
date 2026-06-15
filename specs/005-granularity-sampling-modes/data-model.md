# Data Model: Granularity Sampling Modes

## Sampling Mode

Represents the explicit model-level choice for how granularity is selected.

**Fields**
- `mode`: Either `global` or `per_block`.
- `scope`: Always model-wide for this feature; describes whether one granularity
  applies to the full pass or to each transformer block.
- `source`: The resolved config field that selected the mode.
- `requested_alias`: Optional legacy input such as `training.granularity_sampling`.

**Relationships**
- Attached to one resolved run config.
- Consumed by model assembly and correction logic.

**Validation Rules**
- `mode` must be one of `global` or `per_block`.
- `global` must preserve the current whole-model behavior.
- `per_block` must enable per-block sampling and local correction derivation.
- A legacy alias, when present, must resolve into one of the canonical modes
  before any model or correction logic runs.

## Legacy Sampling Alias

Represents the compatibility layer that maps old config names into the new
canonical sampling mode.

**Fields**
- `source_field`: The legacy config field name, currently
  `training.granularity_sampling`.
- `source_value`: The provided legacy value such as `all` or `random`.
- `resolved_mode`: The canonical sampling mode used downstream.

**Relationships**
- Attached to the resolved config and copied into run metadata.
- Consumed only during config resolution.

**Validation Rules**
- `all` must resolve to `global`.
- `random` must resolve to `per_block`.
- Downstream components must consume only `resolved_mode`, not the legacy alias.

## Granularity Pattern

Represents the granularity choices actually used during a forward pass.

**Fields**
- `pattern_type`: `single` for global sampling or `per_block` for block-wise
  sampling.
- `selected_granularities`: The chosen granularity for the pass or the ordered
  list of per-layer choices.
- `layer_count`: Number of transformer blocks covered by the pattern.
- `repeatable_source`: The configuration inputs that can reproduce the pattern
  for a rerun.

**Relationships**
- Created by the model at runtime.
- Saved into run metadata for later inspection.

**Validation Rules**
- Global sampling must produce a single choice for the entire pass.
- Per-layer sampling must produce one choice per transformer block.
- A per-layer pattern may repeat the same granularity across blocks, but the
  choices must still be recorded as independently selected.

## Correction Context

Represents the correction behavior that accompanies a sampling decision.

**Fields**
- `correction_mode`: The configured correction family (`none`, `gmc`, or
  `lmc`).
- `sampling_mode`: The active sampling mode that determines whether the
  correction is global or local in interpretation.
- `local_correction_active`: Boolean flag indicating whether local GMC/LMC is
  allowed.
- `derived_membership_pattern`: The layer-wise membership pattern used to
  derive local correction.

**Relationships**
- Consumed by FFN correction helpers.
- Saved with the run summary so the correction choice can be reconstructed.

**Validation Rules**
- Local correction may be active only when `sampling_mode=per_block`.
- Global sampling must use the existing global correction behavior.
- A per-layer pattern must drive the local correction interpretation.

## FFN Module Profile

Represents the common FFN metadata needed by both MatFormer FFN variants.

**Fields**
- `variant`: Dense-prefix or concat-block FFN variant.
- `granularity_metadata`: Canonical display names and prefix fractions.
- `prefix_metadata`: Width information for each configured granularity.
- `block_metadata`: Block layout used by the concat variant.

**Relationships**
- Shared by the common granularity helpers and the model wiring layer.
- Used to build the per-layer sampling and correction logic.

**Validation Rules**
- Metadata must remain consistent across the dense-prefix and concat variants.
- Prefix widths must remain strictly increasing in configured granularity order.
- Block metadata must match the available blocks for the configured FFN
  variant.

## Model Wiring State

Represents the assembled model and the per-layer selections currently in use.

**Fields**
- `layer_modules`: Ordered list of FFN-bearing transformer blocks.
- `current_sampling_mode`: Active model-level sampling mode.
- `current_granularity_pattern`: The latest pattern chosen for the forward pass.
- `compatibility_shim`: Legacy entry points retained for current call sites.

**Relationships**
- Owned by the model assembly layer.
- Updated when the caller configures a new granularity decision.

**Validation Rules**
- The wiring layer must apply the selected sampling mode consistently across
  all eligible layers.
- The compatibility shim must preserve existing call sites until migration is
  complete.
