# Research: MatFormer Language Model Reproduction

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

## Decision: Use a debug-size matrix before paper-aligned 78M work

**Decision**: First implement a debug-size S/M/L/XL nested and standalone
matrix, then treat 78M as the first paper-aligned scaling point.

**Rationale**: The debug matrix catches FFN prefix, config, artifact, and
baseline-matching bugs before expensive runs. It also proves the central
nested-versus-standalone comparison across all granularities.

**Alternatives considered**:
- Start directly with 78M: closer to the paper, but slow and risky before the
  workflow is proven.
- Use only one standalone baseline: faster, but it leaves the central
  granularity comparison incomplete before scaling.

## Decision: Label 78M reduced-token pilots separately from 78M/10B complete

**Decision**: A 78M run with fewer than 10B training tokens is a
`reduced-token-pilot`. A 78M run trained with the 10B budget is
`paper-budget-complete`.

**Rationale**: This avoids overstating reproduction progress while still
allowing practical pilots under limited compute.

**Alternatives considered**:
- Count any 78M run as paper-aligned: too easy to misread reduced-token trends.
- Require full 10B before any 78M label: too strict for iterative planning.

## Decision: Preserve paper architecture only for paper-aligned runs

**Decision**: Debug runs may shrink architecture constants. Paper-aligned runs
preserve 16 layers, 16 heads, context length 1024, and the 256k vocabulary
assumption unless explicitly labeled non-paper-aligned.

**Rationale**: The debug stage exists to test workflow correctness. The
paper-aligned stage exists to support trend reproduction claims, so it must
carry the paper constants.

**Alternatives considered**:
- Preserve all constants in debug runs: too expensive for iteration.
- Allow proportional scaling in all phases: risks unclear reproduction labels.

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

## Decision: Use non-embedding parameters for Figure 2-style x-axis

**Decision**: Report non-embedding parameters as
`total_parameters - embedding_parameters - lm_head_parameters`.

**Rationale**: The spec requires Figure 2-style reporting, and embedding
parameters dominate smaller models while being mostly unaffected by FFN nesting.

**Alternatives considered**:
- Total parameters: easier but less faithful to the paper's comparison.
- FFN-only parameters: too narrow for model-size comparisons.

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
