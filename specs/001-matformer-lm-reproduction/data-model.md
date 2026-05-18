# Data Model: MatFormer Language Model Workflow

## ReproductionPhase

Represents a planned scale of work.

**Fields**
- `phase_id`: Stable label such as `debug_matrix`,
  `dmodel256_pilot_comparison`, `medium_trends`, `consistency`, or
  `speculative`.
- `description`: Human-readable purpose.
- `required_model_families`: `nested`, `standalone`, or both.
- `required_granularities`: List of `s`, `m`, `l`, `xl`.
- `required_sampling_modes`: List such as `nested-random`, `nested-all`, and
  `standalone`.
- `dataset_plan`: Dataset identities and splits.
- `completion_criteria`: Required artifacts and comparisons.

**Relationships**
- Owns many `TrainingRun` records.
- Produces many `MetricsArtifact` and `FigureArtifact` records.

**Validation Rules**
- `debug_matrix` requires nested and standalone coverage for all four
  granularities.
- `dmodel256_pilot_comparison` must identify whether each run is a
  reduced-token pilot or uses the full 10B-token budget.
- `dmodel256_pilot_comparison` defaults to `nested-random`, `nested-all`, and
  standalone S/M/L/XL rows where compute allows; omitted rows must be explicit.

## ModelFamily

Groups runs by model type.

**Values**
- `nested`: MatFormer-Llama/SwiGLU model that can expose S/M/L/XL submodels by
  FFN prefix.
- `standalone`: Independently trained fixed-width Transformer baseline.

**Validation Rules**
- `standalone` runs have exactly one granularity.
- `nested` runs may evaluate one or more granularities from the same checkpoint.

## SamplingMode

Represents the training or comparison mode for a run.

**Values**
- `nested-random`: Nested MatFormer training samples one configured granularity
  per batch or step, matching the original `train.py` behavior.
- `nested-all`: Nested MatFormer training evaluates all configured
  granularities on each batch and averages their losses.
- `standalone`: Fixed-width independent baseline training for one granularity.

**Validation Rules**
- `nested-random` and `nested-all` require `model_family=nested`.
- `standalone` requires `model_family=standalone`.
- Metrics, summaries, scaling rows, and reports must expose the sampling mode.

## Granularity

Represents an FFN expansion ratio and prefix rule.

**Fields**
- `name`: One of `s`, `m`, `l`, `xl`.
- `display_name`: One of `S`, `M`, `L`, `XL`.
- `ffn_ratio`: One of `0.5`, `1`, `2`, `4`.
- `full_intermediate_fraction`: Fraction of full XL intermediate width:
  `0.125`, `0.25`, `0.5`, `1.0`.

**Validation Rules**
- Smaller granularities must be strict prefixes of larger granularities.
- Comparison labels must use the same canonical names across configs, metrics,
  and reports.

## ModelShapeTarget

Represents debug, pilot, or later scaling shape targets.

**Fields**
- `model_shape_label`: Stable implementation label such as `debug` or
  `dmodel256`.
- `d_model`: Transformer hidden size.
- `num_layers`: Transformer layer count.
- `num_attention_heads`: Attention-head count.
- `context_length`: Maximum context length.
- `vocab_size_assumption`: Vocabulary-size assumption.
- `granularity_prefixes`: Ordered mapping of S/M/L/XL FFN prefix widths or
  fractions.
- `training_token_budget`: Planned token budget.
- `completion_label`: `debug`, `reduced-token-pilot`, or
  `full-token-budget`.

**Validation Rules**
- Pilot artifacts must preserve explicit shape fields rather than relying on a
  single model-size label.
- `model_shape_label=dmodel256` with fewer than the full 10B-token budget is
  `reduced-token-pilot`.
- `model_shape_label=dmodel256` with the full 10B-token budget uses
  `completion_label=full-token-budget`.

## DatasetPlan

Captures dataset identity and preprocessing assumptions.

**Fields**
- `dataset_name`: Public dataset identifier.
- `dataset_config_name`: Optional public dataset configuration name, such as
  `sample-10BT` for FineWeb.
- `dataset_split`: Split name.
- `dataset_phase`: `debug`, `medium`, or `large`.
- `sample_limit`: Optional example count.
- `preprocessing_notes`: Tokenization/truncation/shuffling assumptions.

**Validation Rules**
- Dataset identity and preprocessing notes are required for every run.
- Reports must not compare runs with different dataset plans unless the
  difference is visible in the summary.

## TrainingRun

Represents one train/eval execution.

**Fields**
- `run_id`: Unique run directory name.
- `phase_id`: Related `ReproductionPhase`.
- `model_family`: `nested` or `standalone`.
- `sampling_mode`: Related `SamplingMode`.
- `model_shape_target`: Related `ModelShapeTarget`.
- `granularity`: Required for standalone; optional or list-valued for nested.
- `seed`: Optional integer.
- `config_path`: Saved resolved config path.
- `output_root`: Root directory for generated run artifacts; defaults to
  `outputs` and may point outside the repository filesystem.
- `output_dir`: Run artifact directory.
- `explicit_output_dir`: Optional Boolean indicating that `output_dir` was
  provided directly rather than derived from `output_root` and `run_id`.
- `token_budget`: Authoritative planned training-token budget for budgeted
  runs.
- `batch_size_per_process`: Number of examples processed by each data-parallel
  process per training step.
- `effective_world_size`: Active data-parallel process count used for budget
  planning; defaults to 1 unless distributed training sets `WORLD_SIZE`.
- `expected_tokens_per_step`: Planned tokens per optimizer step, derived from
  batch size, context length, and effective world size.
- `derived_max_steps`: Planned step count derived from the token budget.
- `max_steps`: Resolved effective step count used by the training loop. For
  budgeted runs this is derived from `token_budget`, not manually chosen.
- `granularity_sampling`: Nested-training policy. `random` samples one
  configured granularity per batch to match the original `train.py` behavior;
  `all` evaluates all configured granularities on each batch and averages their
  losses for debug or ablation runs.
- `parameter_report_id`: Related `ParameterCountReport`.
- `tokens_seen`: Actual non-padding training tokens observed by the run.
- `stop_reason`: Reason training stopped, such as `not_started`,
  `token_budget_reached`, `max_steps_reached_before_token_budget`, or `failed`.
- `checkpoint_status`: `best_eval`, `final`, `none`, or `unavailable`.
- `best_checkpoint_path`: Optional best-eval checkpoint path.
- `final_checkpoint_path`: Optional final checkpoint path.
- `checkpoint_metric`: Validation metric used for best-eval selection, such as
  `validation_loss` or `perplexity`.
- `status`: `planned`, `running`, `completed`, `failed`, or `superseded`.

**Relationships**
- Belongs to one `ReproductionPhase`.
- Uses one `DatasetPlan`.
- Uses one `StorageEnvironment`.
- Has one `ParameterCountReport` once the model is materialized.
- May have one `CheckpointArtifact`.
- Produces many `MetricsArtifact` records.
- May participate in one or more `BaselineMatch` records.

**State Transitions**
- `planned -> running -> completed`
- `planned -> running -> failed`
- `completed -> superseded` when a corrected run replaces it.

**Validation Rules**
- When `explicit_output_dir` is false or absent, `output_dir` is
  `<output_root>/<run_id>`.
- The resolved output root must be writable before training starts.
- A custom output root must keep required artifacts out of repository
  `outputs/` unless the researcher explicitly chooses that location.
- `effective_world_size` must come from the active distributed `WORLD_SIZE`
  when distributed training is launched, otherwise 1. It must not be inferred
  from the number of visible or allocated GPUs.
- `expected_tokens_per_step` must equal
  `batch_size_per_process * context_length * effective_world_size`.
- `derived_max_steps` must equal
  `ceil(token_budget / expected_tokens_per_step)`.
- Resolved configs and run summaries must expose the derived budget fields so
  reduced-token pilots cannot be confused with table-budget reference runs.
- `sampling_mode` must be one of `nested-random`, `nested-all`, or
  `standalone` and must be consistent with `model_family`.
- Pilot runs with validation enabled must save a best-eval checkpoint or record
  why no best-eval checkpoint was produced.
- Pilot run summaries must expose checkpoint status and path fields even when
  no checkpoint exists.

## ParameterCountReport

Captures actual implementation parameter counts.

**Fields**
- `parameter_report_id`: Unique id.
- `run_id`: Related training run.
- `total_parameters`: Actual implementation total.
- `embedding_parameters`: Token embedding parameters.
- `lm_head_parameters`: Language-model-head parameters.
- `non_embedding_parameters`: Total excluding token embeddings and LM-head or
  output embedding parameters.
- `ffn_parameters`: FFN parameters for the relevant model or granularity.
- `attention_parameters`: Optional attention parameter count when feasible.
- `other_non_embedding_parameters`: Optional remaining non-embedding count when
  feasible.
- `lm_head_counting`: `tied`, `untied`, `excluded`, or `separately_counted`.

**Validation Rules**
- Actual implementation count fields are required for pilot resolved configs,
  run summaries, scaling rows, and comparison artifacts.
- `lm_head_counting` is required whenever any total or non-embedding count is
  reported.

## CheckpointArtifact

Represents a saved or explicitly omitted model state for reuse.

**Fields**
- `checkpoint_id`: Unique id.
- `run_id`: Related training run.
- `checkpoint_status`: `best_eval`, `final`, `none`, or `unavailable`.
- `checkpoint_path`: Path under the run output directory when available.
- `selection_metric`: Validation metric used for best-eval selection.
- `selection_metric_value`: Numeric metric value when available.
- `written_by_rank`: Distributed rank that wrote the shared checkpoint.

**Validation Rules**
- Pilot best-eval checkpoints are written only by rank 0 under distributed/FSDP
  execution.
- `run_summary.json` must reference the checkpoint path when one exists.
- Runs with validation disabled must record final-checkpoint or no-checkpoint
  status.

## StorageEnvironment

Captures filesystem and cache placement for an experiment run.

**Fields**
- `output_root`: Root directory for generated run artifacts.
- `output_dir`: Resolved directory for a single run.
- `hf_home`: Optional Hugging Face home/cache root from `HF_HOME`.
- `hf_datasets_cache`: Optional dataset cache from `HF_DATASETS_CACHE`.
- `transformers_cache`: Optional model cache from `TRANSFORMERS_CACHE`.

**Validation Rules**
- Missing output roots are created when possible.
- Unwritable output roots fail before training or evaluation starts.
- External caches are documented as environment variables and are not embedded
  in run comparison semantics.

## BaselineMatch

Pairs an extracted nested submodel with a standalone baseline.

**Fields**
- `match_id`: Unique comparison id.
- `nested_run_id`: Source nested run.
- `standalone_run_id`: Source standalone run.
- `granularity`: `s`, `m`, `l`, or `xl`.
- `nested_sampling_mode`: `nested-random` or `nested-all`.
- `non_embedding_parameters_nested`: Numeric count.
- `non_embedding_parameters_standalone`: Numeric count.
- `nested_parameter_report_id`: Related nested `ParameterCountReport`.
- `standalone_parameter_report_id`: Related standalone `ParameterCountReport`.
- `match_notes`: Any known mismatch or caveat.

**Validation Rules**
- Dataset, tokenizer assumption, token budget, and architecture phase must
  match unless the mismatch is recorded.
- Every reported nested comparison either has a `BaselineMatch` or is marked
  baseline missing.
- Pilot comparison rows must expose model family, granularity, sampling mode,
  token budget, effective world size, checkpoint path when available, and
  mismatch notes.

## EvaluationSuite

Defines validation, downstream, consistency, or speculative evaluations.

**Fields**
- `suite_id`: Stable suite name.
- `suite_type`: `validation`, `downstream`, `consistency`, `mix-and-match`, or
  `speculative`.
- `tasks`: Task names or metric names.
- `prompt_set`: Optional prompt source for speculative decoding.
- `required_metrics`: Expected metric fields.

**Validation Rules**
- The minimal downstream suite contains HellaSwag, PIQA, ARC-Challenge, BoolQ,
  WinoGrande, and OpenBookQA.
- Speculative suites must include acceptance rate, rollback frequency,
  throughput, and latency.

## MetricsArtifact

Structured metrics written by runs and evaluations.

**Fields**
- `artifact_path`: CSV or JSON file path.
- `artifact_type`: `metrics`, `task_results`, `scaling_results`,
  `consistency_results`, or `run_summary`.
- `run_id`: Related run when applicable.
- `schema_version`: Artifact schema version.
- `created_at`: Timestamp.

**Validation Rules**
- Required metrics must be written to CSV or JSON, not only terminal logs.
- Artifact paths for a run must be rooted under that run's resolved
  `output_dir` unless the researcher explicitly overrides a figure/report path.
- Plotting scripts read these artifacts as their source of truth.

## FigureArtifact

Plots or reports generated from structured metrics.

**Fields**
- `figure_path`: PNG, PDF, or markdown report path.
- `source_artifacts`: List of CSV/JSON inputs.
- `figure_type`: `loss_vs_size`, `ppl_vs_size`, `accuracy_vs_size`,
  `consistency_vs_size`, or `efficiency`.

**Validation Rules**
- Every figure must be reproducible from listed source artifacts.
- Figure paths should live under the configured output root unless explicitly
  overridden.
