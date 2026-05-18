# Research: MatFormer Language Model Workflow

## Decision: Keep the implementation Llama-based and local to the repo

**Decision**: Continue using the existing Llama-style model path in
`modified_llama.py` and `train.py`, with MatFormer behavior implemented by
explicit FFN prefix slicing.

**Rationale**: The current repository already contains a compact
`ModifiedLlamaMLP` that slices `gate_proj`, `up_proj`, and `down_proj`. This is
the simplest path to inspect tensor shapes, compare S/M/L/XL granularities, and
avoid a second model framework.

**Alternatives considered**:
- From-scratch Transformer: easier to make tiny, but less aligned with the
  current code and paper-style decoder-only setup.
- Full external training framework: more features, but too much hidden control
  flow for research iteration.

## Decision: Use a debug-size matrix before the d_model=256 pilot comparison

**Decision**: First implement a debug-size S/M/L/XL nested and standalone
matrix, then run a d_model=256 MatFormer-Llama/SwiGLU pilot comparison
inspired by the MatLM 78M table row.

**Rationale**: The debug matrix catches FFN prefix, config, artifact, and
baseline-matching bugs before expensive runs. It also proves the central
nested-versus-standalone comparison across all granularities.

**Alternatives considered**:
- Start directly with the d_model=256 pilot: closer to the target pilot, but
  slow and risky before the workflow is proven.
- Use only one standalone baseline: faster, but it leaves the central
  granularity comparison incomplete before scaling.

## Decision: Frame the pilot as an explicit d_model=256 workflow

**Decision**: The pilot is labeled by explicit d_model=256 shape fields,
sampling mode, and token-budget completion labels. Runs with fewer than the
full 10B-token budget are `reduced-token-pilot`; runs using that budget are
`full-token-budget`.

**Rationale**: The implementation should describe what it actually runs. The
shape fields and parameter counts are enough to identify the pilot without
carrying paper-reference metadata through configs and artifacts.

**Alternatives considered**:
- Keep calling the pilot "78M": compact, but it hides actual implementation
  parameter counts and counting conventions.
- Require full 10B before any pilot run: too strict for iterative planning and
  checkpoint/artifact validation.

## Decision: Preserve explicit shape fields and implementation counts

**Decision**: Debug runs may shrink architecture constants. Pilot and scaling
artifacts preserve explicit fields for d_model, layer count, attention-head
count, context length, vocabulary-size assumption, token budget, and
granularity prefixes, plus implementation parameter counts.

**Rationale**: The debug stage exists to test workflow correctness. The
pilot stage exists to support trend checks and later evaluation, so it must
make its actual shape inspectable.

**Alternatives considered**:
- Preserve all constants in debug runs: too expensive for iteration.
- Allow proportional scaling without explicit shape reporting: risks unclear
  comparison labels.

## Decision: Use YAML configs with resolved JSON snapshots

**Decision**: Author experiment inputs as simple YAML files and save a resolved
`config.json` under each `outputs/<run_id>/`.

**Rationale**: YAML is readable for experiments, while JSON snapshots are easy
to diff, parse, and attach to summaries. The config maps directly to research
concepts: phase, model family, granularity, model size, token budget, dataset,
seed, output policy, and evaluation suite.

**Alternatives considered**:
- CLI-only arguments: quick but error-prone for repeated matrices.
- Dynamic config systems: unnecessary for this repository and contrary to the
  constitution.

## Decision: Make CSV/JSON artifacts the source of truth

**Decision**: Every run writes scalar metrics to CSV/JSON. Plots and reports are
derived from those artifacts, not terminal logs.

**Rationale**: The reproduction depends on comparing runs across nested and
standalone families. Structured artifacts make comparisons and figure
generation repeatable.

**Alternatives considered**:
- Terminal logs only: insufficient for later analysis.
- Heavy experiment trackers: useful later, but unnecessary for the first
  implementation and less inspectable than local files.

## Decision: Report disaggregated implementation parameter counts

**Decision**: Report actual implementation counts for
`total_parameters`, `embedding_parameters`, `lm_head_parameters`,
`non_embedding_parameters`, and `ffn_parameters`; when feasible also report
`attention_parameters` and `other_non_embedding_parameters`. Reports must state
whether the LM head is tied, untied, excluded, or separately counted.

**Rationale**: The spec requires Figure 2-style reporting, and embedding
parameters dominate smaller models while being mostly unaffected by FFN nesting.
The clarified pilot needs disaggregated implementation counts because FFN size
changes and LM-head counting materially affect comparisons.

**Alternatives considered**:
- Total parameters only: easier but hides the exact mismatch motivating Phase
  4.7.
- FFN-only parameters: useful but too narrow for model-size comparisons.

## Decision: Default the pilot runner to comparison mode

**Decision**: The d_model=256 pilot runner defaults to a comparison workflow
with `nested-random`, `nested-all`, and standalone S/M/L/XL baselines where
compute allows. `nested-random` is the primary nested mode because it matches
the original `train.py` sampling behavior; `nested-all` remains an ablation for
the existing all-at-once averaged-loss behavior.

**Rationale**: The central question is whether jointly trained nested
granularities are competitive with independently trained baselines. A single
nested run cannot answer that question, and unlabeled sampling modes would make
later downstream or consistency results ambiguous.

**Alternatives considered**:
- Keep a single nested pilot by default: cheaper, but it defers the central
  comparison and risks misleading early reports.
- Use only `nested-all`: preserves existing behavior but does not match the
  original random-granularity training rule.

## Decision: Save best-eval pilot checkpoints when validation is enabled

**Decision**: Pilot runs with validation enabled save a rank-0-safe best-eval
checkpoint under the run output directory and record its path and selection
metric in `run_summary.json`. Runs without validation either save a final
checkpoint or record that no best-eval checkpoint was produced.

**Rationale**: Downstream, consistency, and speculative decoding phases should
reuse trained pilot models instead of requiring a rerun. Recording the
checkpoint status also prevents comparison rows from implying that evaluation
can proceed when no usable model state exists.

**Alternatives considered**:
- Save no pilot checkpoint by default: cheaper on storage but blocks later
  evaluation reuse.
- Save every checkpoint: unnecessary storage pressure for a repository already
  designed around redirectable output roots.

## Decision: Start downstream evaluation with the minimal six-task suite

**Decision**: Use the minimal suite from the spec for early downstream trend
checks: HellaSwag, PIQA, ARC-Challenge, BoolQ, WinoGrande, and OpenBookQA.

**Rationale**: It is representative enough for trend checks while keeping the
first downstream phase bounded.

**Alternatives considered**:
- Full paper-style task suite immediately: too much scope before validating the
  training and baseline matrix.
- Validation perplexity only: misses the spec's downstream trend requirement.

## Decision: Use token-level agreement first for consistency

**Decision**: Implement token-level argmax agreement as the first consistency
metric, then add KL divergence or top-k overlap after the basic artifact flow is
stable.

**Rationale**: Token-level agreement is simple, interpretable, and directly
tests smaller/larger prediction alignment.

**Alternatives considered**:
- KL divergence first: more informative but requires careful numerical handling.
- Speculative acceptance only: useful, but it depends on a later decoding
  evaluation path.

## Decision: Treat speculative decoding as a later evaluation phase

**Decision**: Keep speculative decoding out of the first implementation slice.
Plan it after validation, baseline, scaling, and consistency artifacts exist.

**Rationale**: Draft/verifier comparisons need reliable extracted submodels,
standalone baselines, and prompt/result artifact schemas first.

**Alternatives considered**:
- Implement speculative decoding during P1: too much risk before the core
  training and comparison loop is validated.

## Decision: Use focused smoke tests instead of broad framework testing

**Decision**: Add focused tests for config resolution, FFN prefix slicing,
artifact writing, non-embedding parameter counting, and tiny run wiring.

**Rationale**: These tests target the failure modes most likely to waste
research time without turning the repository into production software.

**Alternatives considered**:
- Large test suite around every helper: unnecessary overhead.
- No tests: risky because shape and artifact failures are easy to miss until
  after long runs.
