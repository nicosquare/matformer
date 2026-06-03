# Research: Experiment Config Resolution

## 1. Correction Mode for Concat Runs

- **Decision**: Add a resolved `correction_mode` with values `none`, `gmc`,
  and `lmc`, keep LMC scoped to the concat model path, and apply it during the
  optimizer step rather than by mutating gradients.
- **Rationale**: The existing code already computes concat membership counts
  and exposes concat-specific block metadata. That makes the optimizer-step
  seam the clearest place to apply a learning-rate multiplier without changing
  gradient accumulation or optimizer moment state.
- **Alternatives considered**:
  - Reusing the existing gradient hook path for LMC. Rejected because LMC is
    supposed to leave gradients untouched.
  - Introducing a shared abstraction that would also refactor slicing. Rejected
    because the feature can stay local to concat runs for v1.

## 2. Shared Family Folder Resolution

- **Decision**: Keep the existing family and token-budget components of the
  output group, but resolve the size component from the largest configured
  model size in the family rather than from the active standalone size.
- **Rationale**: The project already stores folder identity in resolved config
  and uses it to place all outputs. Resolving the folder from family metadata
  keeps standalone `s`, `m`, and `l` runs comparable without a manual rename
  step and keeps the rule deterministic across reruns.
- **Alternatives considered**:
  - Keying folders from exact standalone parameter counts. Rejected because it
    fragments otherwise comparable family runs.
  - Adding a separate manual post-processing step for figure generation.
    Rejected because it creates a brittle workflow outside the run metadata.
  - Moving to a dedicated family-level override. Rejected because the resolved
    granularity list already carries the necessary comparison context.

## 3. Config Presets

- **Decision**: Use section-scoped, config-driven presets stored under
  `presets.<section>.<name>` and selected with a section field such as
  `training.optimizer.preset`.
- **Rationale**: This matches the current config-style of explicit YAML values
  and works cleanly with the existing component-merge pattern already used for
  optimizer and scheduler kwargs.
- **Alternatives considered**:
  - A single global preset registry. Rejected because it blurs section
    ownership and makes nested defaults harder to reason about.
  - Hardcoding optimizer defaults in the training loop. Rejected because the
    feature is specifically about reusable config, not implicit behavior.
  - Allowing preset composition in v1. Rejected because one preset per section
    keeps resolution obvious and avoids precedence ambiguity.

## 4. Provenance in Resolved Artifacts

- **Decision**: Record the selected correction mode, folder-resolution rule,
  and preset provenance in both `config.json` and `run_summary.json`.
- **Rationale**: The repository already treats these files as the audit trail
  for a run. Keeping the provenance there makes comparison and debugging
  possible without inspecting logs.
- **Alternatives considered**:
  - Logging the values only in console output. Rejected because terminal logs
    are not durable analysis artifacts.
  - Storing provenance only in one file. Rejected because the spec requires the
    saved config and run summary to both explain what ran.

## 5. Validation Scope

- **Decision**: Validate with one synthetic concat test, one real concat smoke
  test, config-resolution tests, artifact-placement tests, and a figure
  generation smoke test.
- **Rationale**: This covers the failure modes most likely to waste research
  time: wrong correction semantics, broken config merging, wrong output folder
  placement, and figure workflows that cannot consume the new structure.
- **Alternatives considered**:
  - Relying only on unit tests. Rejected because the feature changes saved
    artifact organization as well as runtime behavior.
  - Relying only on end-to-end training. Rejected because the synthetic concat
    case needs a precise check on gradients and optimizer state.
