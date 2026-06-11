# Data Model: Granularity Operation Modes

## Run Mode

Represents the canonical top-level mode for a training run.

**Fields**
- `name`: One of `nested-random`, `nested-all`, or `standalone`.
- `model_family`: The resolved run family used by the training pipeline
  (`nested` or `standalone`).
- `sampling_mode`: The run-level selector written to config and summaries.

**Relationships**
- Owns the choice of how the rest of the run should interpret granularity.
- Feeds the model-level sampling submode and the correction context.

**Validation Rules**
- `name` must be one of the three canonical values.
- `nested-random` must support `global` and `per_layer` model sampling.
- `nested-all` must evaluate all configured granularities and must not enable
  per-layer random sampling.
- `standalone` must remain fixed to one granularity for the entire run.

## Sampling Submode

Represents the model-level choice used within `nested-random`.

**Fields**
- `mode`: One of `global` or `per_layer`.
- `source`: The resolved config field that selected the submode.

**Relationships**
- Attached to one resolved run config and one run provenance record.
- Consumed by model wiring and correction logic.

**Validation Rules**
- `global` must select one granularity for the full model on a given iteration.
- `per_layer` must select one granularity per transformer block.

## Granularity Pattern

Represents the granularity choices actually used during a forward pass or
evaluation pass.

**Fields**
- `pattern_type`: `single`, `per_layer`, or `all_granularities`.
- `selected_granularities`: The chosen granularity or ordered sequence of
  per-layer or per-evaluation choices.
- `layer_count`: Number of transformer blocks covered by the pattern.
- `repeatable_source`: The config inputs needed to reconstruct the pattern.

**Relationships**
- Created by the model or evaluation loop at runtime.
- Saved into run metadata and summaries for later inspection.

**Validation Rules**
- `nested-random + global` must produce a single shared selection per
  iteration.
- `nested-random + per_layer` must produce one selection per transformer
  block.
- `nested-all` must record the full set of evaluated granularities.
- `standalone` must record the fixed granularity used for the whole run.

## Correction Context

Represents the correction behavior that accompanies a sampling decision.

**Fields**
- `correction_mode`: One of `none`, `gmc`, or `lmc`.
- `sampling_mode`: The active run or sampling mode used to interpret
  correction.
- `local_correction_active`: Boolean flag indicating whether local GMC/LMC is
  active.
- `derived_membership_pattern`: The layer-wise pattern used to derive local
  correction when that mode is active.

**Relationships**
- Consumed by FFN correction helpers and written into run summaries.
- Derived from both the resolved config and the runtime granularity pattern.

**Validation Rules**
- Local correction may be active only when the run is using per-layer
  sampling.
- Global, nested-all, and standalone paths must preserve the current whole-
  model correction behavior.
- A per-layer pattern must drive the local correction interpretation.

## Run Provenance Record

Represents the saved metadata needed to reconstruct a run without logs.

**Fields**
- `resolved_run_mode`: The canonical top-level run mode.
- `resolved_sampling_mode`: The canonical model-level sampling mode.
- `correction_mode`: The resolved correction family.
- `granularity_pattern_summary`: A compact summary of the runtime pattern.

**Relationships**
- Written to `config.json`, `run_summary.json`, and metrics rows.
- Consumed by downstream analysis and regression tests.

**Validation Rules**
- All accepted runs must be reconstructable from saved artifacts alone.
- The provenance record must distinguish global, per-layer, nested-all, and
  standalone executions.
- The provenance record must match the resolved config exactly.

## FFN Module Profile

Represents the common FFN metadata needed by both MatFormer FFN variants.

**Public terminology**
- `slicing`: the existing FFN path that narrows the active granularity by
  slicing the shared FFN layout.
- `concat`: the existing FFN path that composes granularity-specific blocks by
  concatenation.

**Fields**
- `variant`: Canonical public variant label, one of `slicing` or `concat`.
- `granularity_metadata`: Canonical display names and prefix fractions.
- `prefix_metadata`: Width information for each configured granularity.
- `block_metadata`: Block layout used by the `concat` variant.

**Relationships**
- Shared by the granularity helpers and the model wiring layer.
- Used to build the per-layer sampling and correction logic.

**Validation Rules**
- Metadata must remain consistent across the slicing and concat variants.
- Prefix widths must remain strictly increasing in configured granularity order.
- Block metadata must match the available blocks for the configured FFN
  variant.

## Model Wiring State

Represents the assembled model and the per-layer selections currently in use.

**Fields**
- `layer_modules`: Ordered list of FFN-bearing transformer blocks.
- `current_run_mode`: Active top-level run mode.
- `current_sampling_mode`: Active model-level sampling mode.
- `current_granularity_pattern`: The latest pattern chosen for the forward or
  evaluation pass.

**Relationships**
- Owned by the model assembly layer.
- Updated when the caller configures a new granularity decision.

**Validation Rules**
- The wiring layer must apply the selected mode consistently across all
  eligible layers.
