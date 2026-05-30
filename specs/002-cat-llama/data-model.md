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
- `effective_world_size`
- `granularities`
- `output_root`
- `seed`
- `token_budget`
- `max_steps`
- `learning_rate`
- `learning_rate_scale_rule`
- `warmup_ratio`
- `warmup_steps`
- `optimizer`

**Relationships**
- Contains one `ModelVariant`.
- Produces one `RunSummary`.

**Validation Rules**
- The variant must be explicit in the resolved config.
- Missing variant input must resolve to `matformer_llama`.
- `run.model_family` must remain the topology selector and must not be reused for variant selection.
- The topology and variant must be consistent with the selected experiment.

## TrainingSchedule

Represents the resolved learning-rate and warmup policy for a run.

**Fields**
- `token_budget`
- `batch_size_per_process`
- `context_length`
- `effective_world_size`
- `expected_tokens_per_step`
- `derived_max_steps`
- `max_steps`
- `base_learning_rate`
- `learning_rate_scale_rule`
- `learning_rate_scale_factor`
- `resolved_learning_rate`
- `warmup_ratio`
- `warmup_steps`
- `resolved_warmup_steps`

**Relationships**
- Selected by `RunConfiguration`.
- Recorded in `RunSummary` and structured artifacts.

**Validation Rules**
- `learning_rate_scale_rule` must be one of `none`, `linear`, or `sqrt`.
- The resolved learning rate must be derived from the base learning rate and
  the resolved global batch size.
- `resolved_warmup_steps` must be derived from `max_steps` when warmup is
  configured as a ratio.
- `learning_rate_scale_factor` must reflect the global-batch-size scaling that
  was applied to `base_learning_rate`.

## OptimizerConfig

Represents the optimizer selected for the run.

**Fields**
- `name`
- `kwargs`

**Relationships**
- Selected by `RunConfiguration`.
- Recorded in `RunSummary` and structured artifacts.

**Validation Rules**
- `name` must be one of the supported optimizer names.
- `kwargs` must accept the minimum supported parameter set for the selected
  optimizer.

## RunSummary

Represents the saved record of what ran and what artifacts were produced.

**Fields**
- `run_id`
- `model_family`
- `model_variant`
- `base_learning_rate`
- `learning_rate_scale_rule`
- `learning_rate_scale_factor`
- `resolved_learning_rate`
- `warmup_ratio`
- `resolved_warmup_steps`
- `optimizer_name`
- `optimizer_kwargs`
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
