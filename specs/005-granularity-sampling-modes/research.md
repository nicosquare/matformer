# Research: Granularity Sampling Modes

## 1. Sampling-Mode Configuration Surface

- **Decision**: Introduce a dedicated model-level `granularity_sampling_mode`
  with explicit values `global` and `per_layer`, and treat the legacy
  `training.granularity_sampling` field as a compatibility alias that resolves
  into the canonical mode.
- **Rationale**: The repository already uses `training.granularity_sampling`
  to drive the old sweep behavior, but the feature goal is to make the model
  sampling mode explicit. A compatibility alias keeps old configs usable while
  making the new canonical mode the only behavior downstream code needs to
  understand.
- **Alternatives considered**:
  - Overloading `training.granularity_sampling` for model sampling without a
    compatibility layer. Rejected because it would create a harder migration
    path for existing configs.
  - Making per-layer behavior implicit. Rejected because the feature goal is to
    make the current global behavior a first-class, named mode.

## 2. Legacy Alias Resolution

- **Decision**: Resolve legacy `training.granularity_sampling` inputs into the
  canonical sampling mode and record both the requested alias and resolved mode
  in run metadata.
- **Rationale**: The migration is easier to understand if there is a single
  canonical mode in the runtime, while the alias remains visible in artifacts
  for debugging and provenance.
- **Alternatives considered**:
  - Dropping the legacy field immediately. Rejected because it makes migration
    unnecessarily abrupt for existing configs.
  - Keeping the legacy field as a second independent control surface. Rejected
    because it would recreate the ambiguity the feature is trying to remove.

## 3. Model Module Split

- **Decision**: Move the shared logic out of `modified_llama.py` into a shallow
  `models/` package with four responsibilities: granularity metadata,
  FFN implementations, correction logic, and model wiring/assembly.
- **Rationale**: The current file already mixes metadata helpers, dense MLP
  variants, concat block logic, and model replacement logic. A shallow package
  keeps those concerns close enough to inspect while making future changes less
  risky.
- **Alternatives considered**:
  - Keeping the monolith and only adding comments. Rejected because the file is
    already carrying multiple responsibilities that will grow further with
    per-layer sampling.
  - Introducing a deep package hierarchy. Rejected because the constitution
    favors shallow, obvious organization for research code.

## 4. Correction Behavior by Sampling Mode

- **Decision**: Preserve the existing global correction path for
  `granularity_sampling_mode=global`, and derive local GMC/LMC behavior from
  the sampled per-layer pattern when `granularity_sampling_mode=per_layer`.
- **Rationale**: The feature explicitly asks for the current behavior to remain
  first-class while adding a new experimental option. Local correction should
  be tied to the new mode so that global parity remains easy to test.
- **Alternatives considered**:
  - Applying local correction in both modes. Rejected because it would change
    the existing global semantics and make regression testing harder.
  - Creating a separate correction mode independent of sampling mode. Rejected
    because the requested behavior is mode-dependent rather than a third
    standalone family.

## 5. Metadata and Artifact Provenance

- **Decision**: Record the sampling mode, the selected correction mode, and a
  compact granularity-pattern summary in saved run metadata and summaries.
- **Rationale**: The repository already treats `config.json`, `run_summary.json`,
  and CSV metrics as the audit trail for a run. Adding the sampling-mode
  provenance there makes comparisons possible without reading logs.
- **Alternatives considered**:
  - Recording only the selected mode and omitting the pattern summary.
    Rejected because per-layer behavior needs the sampled pattern to be
    interpretable later.
  - Storing provenance only in console output. Rejected because the outputs
    must be durable and machine-readable.

## 6. Validation Strategy

- **Decision**: Validate with focused unit tests for granularity metadata and
  correction helpers, plus smoke tests on the debug matrix for both sampling
  modes.
- **Rationale**: This is the smallest test set that can detect the likely
  regressions: wrong sampling semantics, broken correction activation, and
  accidental changes to the current global path.
- **Alternatives considered**:
  - Relying only on end-to-end training runs. Rejected because they are too
    slow for iterative model refactors.
  - Relying only on unit tests. Rejected because the feature also changes run
    wiring and saved metadata.
