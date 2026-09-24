# Feature Specification: TinyStories S1 Fourfold LR Warmup

**Feature Branch**: `016-tinystories-s1-warmup`  
**Created**: 2026-09-22  
**Status**: Draft — validated for planning  
**Input**: [Source request](../../notes/tinystories_s1_warmup_speckit_prompt_2026-09-22.md): extend the existing TinyStories comparison with exactly two fresh S1 runs testing 256-update learning-rate warmup.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Establish a controlled warmup comparison (Priority: P1)

A researcher prepares one Linear S1 and one Geometric S1 run and verifies that each changes only learning-rate warmup relative to its original uniform-sampling S1 counterpart. This makes the comparison interpretable without repeating existing baselines.

**Why this priority**: An unintended change to sampling, training budget, or model controls would confound the intervention.

**Independent Test**: Compare the two proposed resolved protocols against saved counterpart configurations and known action/data traces, without launching training. Exercise incorrect-control and historical-reference cases.

**Acceptance Scenarios**:

1. **Given** the original Linear and Geometric S1 configurations, **When** the extension is prepared, **Then** exactly two fresh seed-42 S1 runs use their respective original grids, 256 warmup updates, peak LR 0.008, and 348,528 total updates including warmup.
2. **Given** distinct new run identities, **When** deterministic action and batch streams are resolved, **Then** each matches its own original S1 counterpart over all four epochs; identity changes do not reseed those streams.
3. **Given** an extra arm, changed sampling, enabled pre-nested warmup, altered horizon, historical initialization checkpoint, or occupied output path, **When** validation runs, **Then** the discrepancy is identified and the proposed run is rejected.
4. **Given** the eight existing standalones and two original S1 runs, **When** references are selected, **Then** their saved controls, complete terminals, and ordinary-validation provenance are checked without retraining or modifying them; invalid CPU attempts and cancelled arms are excluded.
5. **Given** historical campaign inputs, **When** the extension's validation becomes available, **Then** those inputs retain their original identities and strict 64-update-warmup rules.

---

### User Story 2 - Verify and complete the longer schedule safely (Priority: P1)

A researcher verifies the resolved schedule and interruption behavior before production, then, once execution is authorized, completes both runs with reliable continuation and evidence of actual GPU execution.

**Why this priority**: Warmup must affect the intended updates, and interruptions must not reset or extend the schedule.

**Independent Test**: Inspect the complete resolved schedule and compare interrupted versus uninterrupted short executions around warmup and epoch boundaries. Exercise incompatible restores, partial-update failures, and unavailable-GPU cases.

**Acceptance Scenarios**:

1. **Given** the fixed full horizon, **When** schedule values are inspected at initialization, around updates 64 and 256, and at the terminal boundary, **Then** the inherited warmup convention uses 256 updates and cosine decay spans the remaining 348,272 updates to the original terminal horizon and endpoint.
2. **Given** checkpoints before, at, and after warmup completion or an epoch transition, **When** each new run resumes, **Then** subsequent actions and batches match uninterrupted execution exactly, LR positions agree, and numerical state agrees within inherited tolerances.
3. **Given** a checkpoint from another grid, original 64-update S1, or incompatible new run, **When** continuation is requested, **Then** it is rejected before live state changes. A failed incomplete update cannot replace the last durable complete-update checkpoint.
4. **Given** execution authorization and passed checks bound to the intended source and controls, **When** production runs, **Then** both runs start fresh, use verified GPU bf16 execution under inherited cluster limits, and reach four epochs, 348,528 updates, and 2,855,141,376 tokens each.
5. **Given** no usable GPU, failed readiness evidence, uncertain prior submission, or exhausted user-wide capacity, **When** execution is attempted, **Then** it cannot silently fall back to CPU or create duplicate work; admission waits or fails with the reason recorded.
6. **Given** a valid terminal checkpoint with missing completion outputs, **When** recovery is performed, **Then** outputs can be completed without another training update or extending the schedule.

---

### User Story 3 - Assess the intervention from saved evidence (Priority: P2)

A researcher compares existing standalones, original S1, and longer-warmup S1 within each grid, reviews early learning-rate and loss behavior, and reports where results improved or worsened.

**Why this priority**: The scientific deliverable is an auditable comparison, regardless of whether longer warmup improves loss.

**Independent Test**: Use valid saved-result fixtures to produce 12 endpoints per grid and eight paired warmup deltas, inspect figures and early curves, and verify rejection of incomplete or incompatible comparison inputs.

**Acceptance Scenarios**:

1. **Given** two full-budget new terminals and validated historical references, **When** comparisons are exported, **Then** each grid has four standalone, four original S1, and four longer-warmup S1 endpoints, with matching loss/perplexity tables and plots.
2. **Given** a new endpoint, **When** a reviewer follows its provenance, **Then** the exact physical width, active parameter count, warmup duration, budget, terminal checkpoint, evaluation identity, and matching historical endpoint are identifiable.
3. **Given** recorded early metrics for each S1 pair, **When** early curves are produced, **Then** LR and loss use the same absolute-update range covering both warmup boundaries and subsequent decay; annotations distinguish recorded and reconstructed LR and any smoothing or missing observations.
4. **Given** a missing reference, duplicate endpoint, incompatible evaluation, invalid device evidence, non-finite metric, or premature/best checkpoint, **When** a complete comparison is requested, **Then** it is refused with the specific problem identified. Valid new-run evidence remains available with comparison completion explicitly outstanding.
5. **Given** eight paired differences, **When** findings are written, **Then** the report states the direction and magnitude at every width and the remaining gap to matching standalones, without requiring improvement or claiming that warmup caused earlier underperformance.

### Edge Cases

- The two grids share three physical widths but have different sampling support; shared dimensions do not make their runs or standalone measurements interchangeable.
- A width can occur less often than expected under replacement sampling. Preserve the counterpart's sequence rather than balancing counts.
- Warmup is an absolute update count inside the budget, not an epoch multiplier, extra phase, or full-width-only stage.
- Different scheduler indexing conventions can hide an off-by-one warmup shift; verify actual applied LR and the recorded scheduler position at both sides of the boundary.
- Restoring after warmup or across epochs must not restart LR, action randomness, data ordering, or shared optimizer history.
- Historical campaign directories may contain invalid CPU attempts, cancelled arms, and stale status summaries. Only validated selected terminals qualify.
- Historical early metrics may be sparse or absent. Do not invent losses, interpolate unobserved values as measurements, or retrain references to fill gaps; disclose unavailable evidence and any resulting incomplete deliverable.
- A terminal evaluation and a trailing validation average are distinct; only the terminal ordinary-validation measurement is an endpoint.
- Coincident standalone points retain separate campaign identities and exact parameter coordinates.
- Missing resource measurements after a failed attempt are disclosed; replayed work is not counted as additional committed training progress.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Extend the existing TinyStories experiment workflow with exactly two fresh S1 production runs, one per grid in EX-001. Reuse existing training, validation, checkpointing, and reporting capabilities; add no standalone retraining, other arms, or independent experiment framework.
- **FR-002**: Change each counterpart's absolute LR warmup from 64 to 256 updates, retain peak LR 0.008 and inherited cosine scheduling through update 348,528, and include warmup within the four-epoch budget. Preserve the original terminal LR convention and keep `pre_nested_warmup` disabled.
- **FR-003**: Preserve each counterpart's slicing representation, shared full-tensor AdamW history and inactive-tail momentum/decay behavior, ordinary causal loss, global L2 clipping cap 1.0, uniform global sampling with replacement at H=1, and correction mode none. All other scientific controls remain inherited as specified in EX-002–004.
- **FR-004**: Use normal fresh seed-42 initialization and unique new identities and artifact paths. Preserve each counterpart's initialization policy and independent action/data seed streams; require exact counterpart action and batch sequence equality. Do not initialize from any trained reference checkpoint or derive new random streams from new run names.
- **FR-005**: Accept 256-update warmup explicitly only for the new declared protocol. Validate resolved controls and their differences against the appropriate reference, permitting only the warmup intervention, consequential schedule values, and necessary identity/provenance/output metadata. Preserve historical validation contracts and hashes; do not globally relax warmup validation.
- **FR-006**: Before production, validate both grids, actual active parameter counts, full budgets, corpus/tokenizer identities, role separation, excluded tail, and reference compatibility. Reject unexpected arms, altered scientific controls, mismatched seeds/traces, occupied or incompatible identities, and shortened or extended horizons.
- **FR-007**: Verify the full resolved LR sequence and actual applied LR around the original and new warmup boundaries, using the inherited scheduler indexing convention. Verify one scheduler advance per committed update, continuous epoch transitions, and the unchanged total horizon. Both new runs MUST have the same LR sequence.
- **FR-008**: Continuation MUST restore the interrupted new run's complete model, shared optimizer, scheduler, randomness, data cursor, and accounting state. Reject cross-run, cross-grid, 64-update-warmup, malformed, and model-only checkpoints before live mutation. Preserve the complete-update durability boundary after optimizer, scheduler, or accounting failures, and permit terminal output recovery without extra training.
- **FR-009**: Before production, pass focused CPU checks and short real-shape bf16 GPU diagnostics for both grids, bound to the intended source/configuration evidence. Cover schedule resolution/application, all widths, finite updates, warmup and epoch-boundary resume, exact action/batch continuity, numerical agreement within inherited tolerances, invalid-state rejection, failure durability, and historical compatibility. Skipped GPU checks do not establish readiness.
- **FR-010**: When GPU execution is subsequently authorized, use sbatch, one process and one GPU per run, exclude `gpu-[05,50,51,54]`, and respect at most two running/four submitted jobs user-wide or stricter live limits. Require usable CUDA, actual bf16 execution, device/allocation evidence, and matching job/worker success; fail instead of silently using CPU. Preserve restart-safe admission, one writer per run, duplicate prevention, and unrelated jobs/helpers.
- **FR-011**: Keep new experiment artifacts under a fresh root separate from both references. Save resolved controls, source/config/data identities, schedule and action/batch evidence, scalar trajectories, exposure, resumable and terminal checkpoints, job/attempt records, summaries, and inherited resource diagnostics. Reconcile committed progress separately from failed/replayed work and disclose incomplete resource evidence.
- **FR-012**: Revalidate only the selected eight standalone and two original uniform S1 references from EX-005. Check saved configurations, checkpoint/evaluation identities, budgets, physical widths/counts, data/evaluation compatibility, and applicable completion/device evidence. Preserve historical files and exclude corrected, inverse-membership, S2/C-arm, partial, cancelled, and invalid CPU results.
- **FR-013**: Evaluate each new full-budget terminal at all four widths using inherited ordinary-validation membership and target-token-weighted causal loss; perplexity MUST equal the exponential of aggregated loss. Freeze checkpoint and evaluation provenance for reporting. Do not substitute best checkpoints, early endpoints, or trailing averages; keep controller data reserved and final holdout sealed.
- **FR-014**: Export matching CSV/JSON comparison tables with exactly 24 unique endpoints: 12 per grid, comprising four existing standalones, four original S1 points, and four new S1 points. Include grid, campaign/run, seed, warmup, physical dimension/fraction, exact active non-embedding count, assigned/actual budget, terminal step, loss/perplexity, evaluated target count, checkpoint/evaluation provenance, and relevant exposure/resources. Preserve separate records at coincident dimensions.
- **FR-015**: Export eight paired differences, one per grid/width, for new-minus-original S1 loss and perplexity; negative means improvement. Include each S1 result's difference from its own grid's matching standalone. Validate references and endpoints before declaring the comparison complete; identify missing, duplicate, stale, non-finite, nonterminal, budget-inconsistent, or evaluation-incompatible inputs explicitly.
- **FR-016**: Produce loss-versus-active-parameters and perplexity-versus-active-parameters plots for each grid in PNG/PDF. Each grid view MUST show disconnected matching standalone markers and two distinct connected S1 curves labeled by 64- versus 256-update warmup. Use exact active counts excluding input embeddings and LM head, recognizable Linear/Geometric labels, and annotations for seed, dataset, ordinary validation, terminal selection, and one-epoch standalone versus four-epoch S1 budgets.
- **FR-017**: Produce early LR and loss comparisons for each S1 pair in PNG/PDF with inspectable supporting scalar data. Use a shared absolute-update window beginning at the start of training, covering at least updates 0–1,024, and mark updates 64 and 256. Include recorded training loss and available per-width ordinary-validation loss, distinguish their meanings, disclose smoothing and gaps, and label any LR reconstructed from validated saved schedule controls. Do not represent reconstruction as measured execution.
- **FR-018**: Report the direction and magnitude of the intervention's observed effects at all eight matched endpoints and discuss early behavior and standalone gaps. Treat seed-42 results as descriptive; make no across-seed significance, equal-runtime/compute, or causal diagnosis of earlier underperformance. Improvement is not a completion requirement.
- **FR-019**: Keep later plans, tasks, verification, and runbook statuses aligned with saved evidence. Distinguish specification readiness, implementation checks, GPU readiness, two-run terminal completion, and comparison completion. This invocation authorizes specification work only; implementation and GPU execution follow subsequent user instructions, without inheriting prior campaigns' launch approvals.

### Research & Experiment Requirements *(include for experiment-facing changes)*

- **EX-001 — Fixed intervention and budgets**: Use exactly this seed-42 production matrix. Labels below describe scientific roles; unique operational identifiers are a planning choice.

| New run | Ordered FFN dimensions | Original warmup | New warmup | Epochs | Total updates | Training tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Linear S1, longer warmup | 64, 128, 192, 256 | 64 | 256 | 4 | 348,528 | 2,855,141,376 |
| Geometric S1, longer warmup | 32, 64, 128, 256 | 64 | 256 | 4 | 348,528 | 2,855,141,376 |

Combined assigned production is 697,056 updates and 5,710,282,752 tokens. Diagnostic and replay costs are additional measured work, not changes to assigned budgets. Warmup occupies the first 256 updates; the remaining schedule spans 348,272 updates.

- **EX-002 — Matched controls**: Inherit model dimension 64, four layers, four attention heads, context 128, vocabulary 2,048, full FFN dimension 256, initializer standard deviation .02; AdamW betas (.9, .95), epsilon 1e-8, weight decay .1; batch 64, accumulation 1, bf16, no LR scaling, and no activation checkpointing. Preserve probability .25 for each ordered width with an independent global draw each update. Active non-embedding count expectations are below and MUST be verified against the actual models.

| FFN dimension | Fraction of full FFN | Active non-embedding parameters | Grid membership |
| --- | ---: | ---: | --- |
| 32 | .125 | 90,688 | Geometric |
| 64 | .25 | 115,264 | Both |
| 128 | .50 | 164,416 | Both |
| 192 | .75 | 213,568 | Linear |
| 256 | 1.00 | 262,720 | Both |

- **EX-003 — Data and reproducibility**: Preserve the audited TinyStories-Instruct tokenizer and four-role packed corpus, data seed 42, designated 5,576,448 sequences per epoch, deterministic epoch ordering, and fixed 43-sequence excluded tail. Each complete update accounts for 8,192 training tokens; each epoch is 87,132 updates and 713,785,344 tokens. Reserved evaluation roles never enter training. Preserve counterpart-derived action and initialization seeds and independent randomness; equal seeds do not justify unsupported cross-model tensor-equality claims.
- **EX-004 — Evaluation and exposure**: Preserve ordinary validation every 64 updates and at completion, including membership, protocol, and causal-target weighting. Record training-token and evaluation-target counts separately. Report actual width selections alongside expected 87,132 selections per width, without enforcing expected counts. Action and batch evidence must reconcile all four epochs against the respective original S1.
- **EX-005 — Historical sources**: Use the reference roots below. Each contributes its own four one-epoch standalones (87,132 updates and 713,785,344 tokens each) and original four-epoch, 64-update-warmup uniform S1. Shared-size standalones remain distinct measurements.

| Grid | Reference root | Selected arms |
| --- | --- | --- |
| Linear | `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1` | ST-g250, ST-g500, ST-g750, ST-g1000, S1 |
| Geometric | `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-matformer-widths-v1` | ST-g125, ST-g250, ST-g500, ST-g1000, S1 |

Read-only specification inspection found 64 resolved warmup updates, LR .008, cosine scheduling, bf16, disabled pre-nested warmup, and a 348,528-update horizon in both `campaign/configs/S1.yaml` and `runs/S1/config.json`. The saved S1 configuration file hashes at inspection were:

| Grid | `campaign/configs/S1.yaml` SHA-256 |
| --- | --- |
| Linear | `ef16f8a6cb883abe6782b9201f32b6d137b324f4d0cd5ac112f66b6fb57ac3fa` |
| Geometric | `781254c7bbd8a7fdc48ba4c09bdc940dd44e132c73d1fe9d8b9209796d94a6c8` |

These observations establish specification inputs, not fresh terminal-validation or execution-readiness evidence. Feature 015's final verification records a completed valid GPU S1 and a closed partial campaign; reuse that selected result without reopening its cancelled work.

### Key Entities *(include if feature involves data)*

- **Warmup extension**: Exactly two fresh run identities, their reference mappings, declared intervention, readiness evidence, and separate training/report completion states.
- **Run protocol**: Grid, seed streams, model/optimizer/data/evaluation controls, 256-update warmup, peak LR, full schedule horizon, and assigned budget.
- **Historical reference**: Immutable selected standalone or original S1 result, saved configuration, scientific identity, terminal checkpoint, evaluation, and applicable execution evidence.
- **Continuation checkpoint**: Complete committed state and protocol identity sufficient to resume LR, action/data streams, shared history, and accounting without repeating warmup.
- **Endpoint**: Grid/run/physical-width identity, terminal loss/perplexity, exact active count, budgets, and evaluation/checkpoint provenance.
- **Comparison evidence**: Twenty-four endpoint records, eight paired warmup deltas, standalone gaps, early LR/loss observations, figures, and descriptive findings.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A researcher can validate exactly two fresh run definitions; all scientific controls match their counterparts except 64-to-256 warmup and its resulting schedule. All defined altered-control and incompatible-identity acceptance cases are rejected.
- **SC-002**: Both complete schedule records use 256 warmup updates inside 348,528 total updates, agree with each other, and retain the original peak and terminal horizon. Boundary checks demonstrate that actual applied LR matches the declared schedule.
- **SC-003**: For both grids, interrupted/resumed verification preserves 100% of subsequent actions and batches and numerical agreement within inherited tolerances before/at/after warmup and epoch boundaries; invalid restore and partial-update cases preserve the required live/durable state boundaries.
- **SC-004**: Exactly two new validated full-budget terminal checkpoints account for 697,056 committed updates and 5,710,282,752 tokens in total. Each has four finite ordinary-validation loss/perplexity endpoints and verified execution provenance; all counterpart action/batch traces match.
- **SC-005**: A complete comparison contains exactly 24 unique traceable endpoints, 12 per grid, and eight paired warmup differences, with consistent tabular exports. Every defined incomplete or incompatible input case prevents a false complete-report status.
- **SC-006**: Both grids have loss and perplexity endpoint views and early LR/loss comparisons in both requested formats. A reviewer can identify both warmup boundaries, compare all three endpoint groups, and trace displayed observations to saved evidence.
- **SC-007**: The written assessment states improvement, worsening, or no change with numerical deltas at all eight matched widths, addresses standalone gaps and early behavior, and identifies missing evidence or interpretive limitations without requiring a positive outcome.
- **SC-008**: Existing reference artifacts and historical contracts remain intact; zero historical production runs are retrained, zero extra arms are launched, and zero final-holdout evaluations occur. Stage completion statements agree with saved evidence and authorization.

## Assumptions

- Intended users are research practitioners and reviewers. Scientific vocabulary, fixed optimizer/model controls, output formats, and execution constraints are user-required protocol details; software architecture and operational command design belong in planning.
- The source request governs this feature. Feature 013–015 specifications and verification records supply inherited behavior and operational lessons, not new execution authorization: [013 specification](../013-tinystories-optimizer-ownership/spec.md), [013 verification](../013-tinystories-optimizer-ownership/verification.md), [014 specification](../014-tinystories-inverse-membership/spec.md), [014 verification](../014-tinystories-inverse-membership/verification.md), [015 specification](../015-tinystories-matformer-widths/spec.md), [015 verification](../015-tinystories-matformer-widths/verification.md).
- Each grid uses its own existing standalone panel as the primary baseline. No averaging or deduplication across grids is implied by shared physical widths.
- The early comparison window defaults to updates 0–1,024, four times the new warmup duration; it may be extended while retaining the shared starting range. Historical losses are used at their recorded cadence, without new reference training or invented observations.
- Existing numerical tolerances, scheduler indexing, terminal recovery, and resource-accounting conventions apply. No accuracy, speed, memory, or statistical-significance improvement is promised.
- Access to the audited corpus/tokenizer and selected reference artifacts is required for later validation. This specification inspects saved controls but does not certify terminal hashes, data audits, schedule execution, or new GPU readiness.
- Additional seeds, extra warmup settings, alternative schedulers, corrections, inverse-membership/adaptive sampling, changed data, and restarting Feature 015's cancelled arms are outside scope. Missing references leave comparison completion outstanding rather than authorizing replacements.
- The exact new artifact root and operational identifiers are planning choices subject to uniqueness, preservation, and counterpart-trace requirements. The specification is ready for `/speckit-plan`; later stages follow subsequent user instructions.
