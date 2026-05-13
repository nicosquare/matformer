# Data Model: MatFormer Language Model Reproduction

## ReproductionPhase

Represents a planned scale of work.

**Fields**
- `phase_id`: Stable label such as `debug_matrix`, `78m_pilot`,
  `medium_trends`, `consistency`, or `speculative`.
- `description`: Human-readable purpose.
- `required_model_families`: `nested`, `standalone`, or both.
- `required_granularities`: List of `s`, `m`, `l`, `xl`.
- `dataset_plan`: Dataset identities and splits.
- `completion_criteria`: Required artifacts and comparisons.

**Relationships**
- Owns many `TrainingRun` records.
- Produces many `MetricsArtifact` and `FigureArtifact` records.

**Validation Rules**
- `debug_matrix` requires nested and standalone coverage for all four
  granularities.
- `78m_pilot` must identify whether it is reduced-token or paper-budget
  complete.

## ModelFamily

Groups runs by model type.

**Values**
- `nested`: MatFormer model that can expose S/M/L/XL submodels by FFN prefix.
- `standalone`: Independently trained fixed-width Transformer baseline.

**Validation Rules**
- `standalone` runs have exactly one granularity.
- `nested` runs may evaluate one or more granularities from the same checkpoint.

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

## ModelSizeTarget

Represents debug or paper-aligned model-size targets.

**Fields**
- `size_label`: `debug`, `78m`, `180m`, `310m`, `463m`, or `850m`.
- `paper_aligned`: Boolean.
- `num_layers`: Transformer layer count.
- `num_attention_heads`: Attention-head count.
- `context_length`: Maximum context length.
- `vocab_size_assumption`: Vocabulary-size assumption.
- `training_token_budget`: Planned token budget.
- `completion_label`: `debug`, `reduced-token-pilot`, or
  `paper-budget-complete`.

**Validation Rules**
- Paper-aligned runs preserve 16 layers, 16 heads, context length 1024, and
  256k vocabulary assumption unless explicitly labeled non-paper-aligned.
- `78m` with fewer than 10B training tokens is `reduced-token-pilot`.
- `78m` with 10B training tokens is `paper-budget-complete`.

## DatasetPlan

Captures dataset identity and preprocessing assumptions.

**Fields**
- `dataset_name`: Public dataset identifier.
- `dataset_split`: Split name.
- `dataset_phase`: `debug`, `medium`, or `large`.
- `sample_limit`: Optional example count.
- `token_budget`: Planned training-token budget.
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
- `model_size_target`: Related `ModelSizeTarget`.
- `granularity`: Required for standalone; optional or list-valued for nested.
- `seed`: Optional integer.
- `config_path`: Saved resolved config path.
- `output_dir`: Run artifact directory.
- `checkpoint_path`: Optional checkpoint path.
- `status`: `planned`, `running`, `completed`, `failed`, or `superseded`.

**Relationships**
- Belongs to one `ReproductionPhase`.
- Uses one `DatasetPlan`.
- Produces many `MetricsArtifact` records.
- May participate in one or more `BaselineMatch` records.

**State Transitions**
- `planned -> running -> completed`
- `planned -> running -> failed`
- `completed -> superseded` when a corrected run replaces it.

## BaselineMatch

Pairs an extracted nested submodel with a standalone baseline.

**Fields**
- `match_id`: Unique comparison id.
- `nested_run_id`: Source nested run.
- `standalone_run_id`: Source standalone run.
- `granularity`: `s`, `m`, `l`, or `xl`.
- `non_embedding_parameters_nested`: Numeric count.
- `non_embedding_parameters_standalone`: Numeric count.
- `match_notes`: Any known mismatch or caveat.

**Validation Rules**
- Dataset, tokenizer assumption, token budget, and architecture phase must
  match unless the mismatch is recorded.
- Every reported nested comparison either has a `BaselineMatch` or is marked
  baseline missing.

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
