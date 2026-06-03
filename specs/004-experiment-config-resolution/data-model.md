# Data Model: Experiment Config Resolution

## Resolved Correction Mode

Represents the membership-correction behavior selected for a run.

**Fields**
- `requested_mode`: The mode selected in config or via override.
- `resolved_mode`: The validated mode used by training.
- `model_path`: The model family path the mode applies to.
- `legacy_mode_source`: Whether the value came from a legacy boolean setting.
- `conflict_reason`: Explanation when requested settings disagree.

**Relationships**
- Attached to one resolved run config and copied into the run summary.
- Consumed by the training loop when deciding whether to apply GMC, LMC, or no
  correction.

**Validation Rules**
- `resolved_mode` must be one of `none`, `gmc`, or `lmc`.
- `lmc` must be rejected for non-concat runs.
- A legacy gradient-correction input and an explicit correction-mode input must
  not disagree.

## Concat Block Membership Plan

Represents the block-level membership metadata used to derive correction values
for concat runs.

**Fields**
- `block_name`: Stable block label such as `block_1`.
- `display_name`: Human-readable label such as `B1`.
- `membership_count`: Number of active granularities that include the block.
- `total_active_losses`: Total number of active granularities in the run.
- `correction_multiplier`: Derived multiplier used for LMC.
- `base_learning_rate`: Scheduler-provided base learning rate.
- `effective_learning_rate`: Learning rate after the correction multiplier is
  applied.

**Relationships**
- Derived from the concat block metadata already carried on the model.
- Used only by concat runs.

**Validation Rules**
- `correction_multiplier` must be `total_active_losses / membership_count`
  when `membership_count` is positive.
- `effective_learning_rate` must be computed at update time, not stored as a
  mutated base rate.
- `membership_count` may be zero only for inactive blocks, which must not
  receive a training update.

## Artifact Family Key

Represents the shared comparison-folder identity used to group related runs.

**Fields**
- `model_family_slug`: Stable family identifier.
- `family_size_slug`: Size component derived from the largest configured family
  size in the family.
- `token_budget_slug`: Budget component preserved from the current output key.
- `output_group`: Final resolved folder key.
- `active_size_label`: Active run size such as `s`, `m`, or `l`.
- `family_resolution_rule`: Human-readable explanation of how the folder was
  chosen.

**Relationships**
- Attached to the resolved run config.
- Used by config artifact writers, metrics writers, checkpoint writers, and
  figure generation.

**Validation Rules**
- `output_group` must be deterministic for the same resolved family config.
- The active size label must remain visible even when the folder key uses the
  family maximum.
- The size component must be derived from the family definition, not from the
  standalone parameter count.

## Preset Definition

Represents a reusable named config block for a single section.

**Fields**
- `section_name`: Section the preset belongs to, such as `optimizer`.
- `preset_name`: Named preset such as `adam`.
- `base_values`: Default section values before preset application.
- `preset_values`: Values defined by the preset.
- `merged_values`: Final values after preset, explicit config values, and CLI
  overrides are merged.
- `provenance`: Metadata that records how the final values were resolved.

**Relationships**
- Selected by one section at a time.
- Merged into the resolved config during config loading.

**Validation Rules**
- Exactly one preset may be selected for a section in v1.
- Invalid preset names must fail before training starts.
- Nested mappings must merge deeply so partial overrides only replace targeted
  fields.

## Resolved Run Metadata

Represents the saved audit trail for one completed or interrupted run.

**Fields**
- `run_id`
- `correction_mode`
- `family_resolution_rule`
- `output_group`
- `preset_selections`
- `config_path`
- `run_summary_path`

**Relationships**
- Stored in `config.json` and `run_summary.json`.
- Used by figure generation and later comparisons to interpret the saved
  outputs.

**Validation Rules**
- The saved metadata must be enough to explain the selected correction mode,
  family folder, and preset values without reading console logs.
- Saved metadata must match the resolved configuration exactly.
