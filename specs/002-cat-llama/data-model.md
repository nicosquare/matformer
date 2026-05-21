# Data Model: Cat Llama Granularity Pipeline

## ModelVariant

Represents the architecture variant used to build the nested Llama model.

**Fields**
- `variant_name`: Canonical name such as `matformer_llama` or `cat_llama`.
- `default_variant`: The fallback variant when no override is present.
- `construction_note`: Short description of how the variant is assembled.
- `applies_to`: Which experiment topologies may use the variant.

**Relationships**
- Selected by `RunConfiguration`.
- Recorded in `RunSummary` and structured artifacts.

**Validation Rules**
- `variant_name` must be one of the supported canonical values.
- The default must remain `matformer_llama`.
- The variant must not change the nested/standalone topology rules.

## GranularityStrategy

Represents how the model constructs granularity prefixes.

**Fields**
- `strategy_name`: `slicing` or `concatenation`.
- `variant_name`: The associated model variant.
- `granularity_order`: Ordered set of `s`, `m`, `l`, `xl`.
- `selection_note`: Human-readable explanation of the difference.

**Relationships**
- Selected by `ModelVariant`.
- Reflected in resolved config and run summaries.

**Validation Rules**
- `cat_llama` must use `concatenation`.
- `matformer_llama` must continue to use the existing slicing path.

## RunConfiguration

Represents the resolved experiment inputs for one run.

**Fields**
- `run_id`
- `phase_id`
- `model_family` topology, such as `nested` or `standalone`
- `model_variant`
- `granularities`
- `output_root`
- `seed`
- `token_budget`

**Relationships**
- Contains one `ModelVariant`.
- Produces one `RunSummary`.

**Validation Rules**
- The variant must be explicit in the resolved config.
- Missing variant input must resolve to `matformer_llama`.
- `run.model_family` must remain the topology selector and must not be reused for variant selection.
- The topology and variant must be consistent with the selected experiment.

## RunSummary

Represents the saved record of what ran and what artifacts were produced.

**Fields**
- `run_id`
- `model_family`
- `model_variant`
- `status`
- `output_dir`
- `metrics_path`
- `scaling_results_path`
- `checkpoint_status`
- `best_checkpoint_path`

**Relationships**
- Summarizes one `RunConfiguration`.
- Links to the output artifact set.

**Validation Rules**
- The summary must echo the resolved variant.
- Comparison artifacts must remain distinguishable by variant.
- Checkpoint fields must remain present even when no checkpoint is saved.
