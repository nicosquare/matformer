# Feature Specification: TinyStories Optimizer Ownership with MatFormer Widths

**Feature Branch**: `015-tinystories-matformer-widths`  
**Created**: 2026-09-21  
**Status**: Closed by user with partial experimental results — 2026-09-22
**Input**: [Feature 015 request](../../notes/tinystories_matformer_widths_speckit_prompt_2026-09-21.md), beginning at “Feature request”.

Researchers need to repeat the original uncorrected Feature 013 optimizer-ownership comparison with FFN widths 12.5%, 25%, 50%, and 100%. A width specifies the feed-forward network's intermediate dimension, not a fraction of total model parameters. The comparison uses four fresh dense standalone models followed by five fresh elastic models, which share weights across widths. Optimizer histories are gradient moments and update counters; separate histories do not imply separate model weights.

The scientific intervention is the width grid. Width selection remains uniform. Historical standalone results provide additional context and repeated measurements at shared sizes; they do not replace fresh baselines. This invocation authorizes specification work only. Implementation and GPU execution require subsequent conversation authorization; prior Feature 013/014 launch approvals do not carry over.

## Final disposition — 2026-09-22

The user requested finalization as-is and no further experiments. This feature
is **closed with partial experimental results**: implementation and readiness
checks are complete; four fresh standalones and S1/S2 passed full-budget terminal
validation. C1/C2 were cancelled partway through and C3 before starting. The
background checker is stopped. Earlier execution authorization is superseded;
do not submit, resume, or restart monitoring for this campaign.

The retained report has **12 new endpoints plus four historical standalone
endpoints (16 total)**. T037, T045 and T052 are cancelled, not successfully
completed. The original nine-run and 24/28-endpoint acceptance criteria remain
unmet; they are retained below as the original protocol, not future work.
T056/T057 are completed against this explicitly reduced closeout scope. See
[the runbook](../../docs/tinystories-matformer-widths-experiment.md) and
[verification.md](verification.md) for results, costs and artifact provenance.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Validate the new nine-run protocol (Priority: P1)

A researcher verifies the new widths, baseline models, sampling, and inherited scientific controls before committing training resources.

**Why this priority**: Incorrect width geometry or inherited controls would invalidate every subsequent result.

**Independent Test**: Validate the nine-run description and inspect actual model dimensions/counts without training; alter individual controls to verify explicit rejection.

**Acceptance Scenarios**:

1. **Given** the audited Feature 013 corpus and original uncorrected controls, **When** the new campaign is prepared, **Then** it contains four fresh standalones at dimensions 32/64/128/256 and five fresh S1/S2/C1/C2/C3 elastic runs, all seed 42, with the exact assigned budgets below.
2. **Given** the ordered new widths, **When** each real model is inspected, **Then** slicing, concat, and matching dense models realize the same dimensions and active non-embedding counts; concat blocks have dimensions 32/32/64/128.
3. **Given** the five elastic definitions, **When** action and batch traces are compared, **Then** all five have identical uniform replacement action sequences and four-epoch batch sequences; all nine share the first epoch's batches.
4. **Given** altered controls, wrong dimensions/counts, inverse-membership probabilities, an occupied run identity, or a historical checkpoint proposed as initialization, **When** preflight runs, **Then** it identifies the discrepancy and refuses the campaign.

---

### User Story 2 - Verify ownership and clipping with unequal blocks (Priority: P1)

A researcher uses short real-model diagnostics to establish that the new block geometry preserves each arm's declared weight, history, and clipping behavior.

**Why this priority**: Correct labels and memory estimates cannot establish that an experiment performs the intended updates.

**Independent Test**: Exercise every width in each elastic arm, including wider-then-narrower updates, and inspect weights, gradients, histories, owner coverage, clipping, and scheduler advances.

**Acceptance Scenarios**:

1. **Given** S1 or S2 after a wider update, **When** a narrower update runs, **Then** slicing retains full-tensor AdamW momentum/decay semantics; S2 updates only the selected width's history and retains full-shaped state.
2. **Given** C1/C2/C3 after wider exposure, **When** a narrower width excludes a block, **Then** its gradient is absent and its weights, moments, and counters do not change; an active present zero gradient still receives ordinary AdamW behavior.
3. **Given** C2 after every width has been selected, **When** histories are counted, **Then** A/B/C/D have 4/3/2/1 histories and common parameters have four, with never-active width/block histories absent.
4. **Given** C3 at any width, **When** an update completes, **Then** the active block owners and common owner each step once, each uses independent L2 clipping at 1.0, and the global scheduler advances once. Changing another owner's norm does not alter a group's clipping coefficient.
5. **Given** new C1/C3 run identities, **When** saved clipping evidence is read through the terminal reader, **Then** committed observation counts, active groups, measured norms, coefficients, and metrics/summary references agree. An omitted clipping artifact causes rejection.

---

### User Story 3 - Complete standalones before elastic production and resume safely (Priority: P1)

Once execution is separately authorized, a researcher completes the four fresh baselines, validates their terminal artifacts, and then runs the five elastic arms with reliable continuation and resource accounting.

**Why this priority**: The requested stage order and exact continuation are essential scientific and operational constraints.

**Independent Test**: Exercise the scheduling barrier with saved completion fixtures, then compare interrupted and uninterrupted short runs across epoch boundaries and inject restore/update failures.

**Acceptance Scenarios**:

1. **Given** any fresh standalone is incomplete or its terminal evidence is invalid, **When** elastic production admission is attempted, **Then** it is blocked. Short pre-production diagnostics may exercise all arms without satisfying the production barrier.
2. **Given** all four standalone terminals pass validation and execution is authorized, **When** elastic production is admitted, **Then** each of the five fresh runs receives its full four-epoch horizon and normal seed-42 initialization.
3. **Given** any new run interrupted within or around an epoch boundary, **When** it resumes from its own valid checkpoint, **Then** future actions/batches match uninterrupted execution exactly and numerical state agrees within existing project tolerances.
4. **Given** changed widths, boundaries, ownership, sampling probabilities, identity, or malformed state, **When** restore is attempted, **Then** it fails before changing live state. A C3 partial update cannot overwrite the previous durable checkpoint.
5. **Given** a restarted scheduler or interrupted run, **When** work is reconciled, **Then** no duplicate writer or submission is created, live user-wide limits are respected, and failed/replayed attempt costs remain recorded.

---

### User Story 4 - Audit the 24 new terminal endpoints (Priority: P2)

A researcher inspects a complete new-campaign result using saved evidence, independently of historical-reference availability.

**Why this priority**: New results must remain auditable and usable even when historical inputs cannot be validated.

**Independent Test**: Supply nine valid terminal fixtures yielding 24 endpoints, inspect exports and per-run diagnostics, and reject corrupted or incomplete present inputs.

**Acceptance Scenarios**:

1. **Given** nine completed new runs, **When** new-campaign reporting runs, **Then** matching CSV/JSON exports contain exactly four standalone endpoints and twenty elastic endpoints, all from full-budget terminal ordinary validation.
2. **Given** a reported point, **When** a reviewer follows saved provenance, **Then** its run, physical width, count, budget, evaluation identity, checkpoint, exposure, clipping, and resources can be established without retraining.
3. **Given** missing/duplicate endpoints, wrong counts, stale hashes, mismatched evaluation roles, non-finite results, or premature/best checkpoints, **When** a complete report is requested, **Then** it is rejected with the invalid input identified.
4. **Given** completed new results and unavailable or incompatible historical references, **When** reporting completes, **Then** the 24-endpoint result is available and the 28-endpoint comparison is explicitly outstanding. Documentation reflects saved evidence rather than proposed work.

---

### User Story 5 - Compare new results with historical standalones (Priority: P2)

A researcher reviews loss and perplexity against exact active parameter counts, comparing new elastic arms with fresh baselines and four preserved historical standalone measurements.

**Why this priority**: This is the requested combined scientific deliverable and must preserve repeated measurements at shared sizes.

**Independent Test**: Combine valid new and historical fixtures into 28 distinct endpoints and inspect table identity, plot structure, legends, and interpretation.

**Acceptance Scenarios**:

1. **Given** 24 valid new endpoints and four revalidated original standalone terminals, **When** combined reporting runs, **Then** it exports 28 distinct records, including both old and new standalone results at dimensions 64/128/256 and the historical dimension-192 point.
2. **Given** these endpoints, **When** loss and perplexity figures are generated, **Then** each contains five connected four-point elastic curves, four disconnected fresh standalone markers, and four disconnected historical standalone markers, with exact count coordinates and distinguishable coincident points.
3. **Given** a historical g750 endpoint, **When** it is displayed, **Then** it remains dimension 192 with count 213,568; it is not relabeled as a new width. Historical elastic, GMC/LMC, and inverse-membership runs do not enter the comparison.
4. **Given** the seed-42 report, **When** conclusions are written, **Then** they compare new elastic arms primarily with fresh baselines, distinguish ownership/representation from C3 clipping and changed block sizes, and do not claim across-seed significance or equal compute from equal tokens.

### Edge Cases

- Short uniform random sequences can omit or repeatedly select widths. Never force balance or allocate lazy histories solely to satisfy an expected count.
- Equal ordered positions across campaigns can denote different physical widths. Identity must include the actual fraction, dimension, and campaign/run.
- An unequal block may contain more parameters but still has one C3 owner and cap 1.0. Splitting blocks C/D into extra clipped owners changes the experiment and must fail validation.
- Tied parameters belong to exactly one C3 owner; any common FFN output bias belongs to the common owner.
- The fixed excluded 43-sequence tail remains excluded in every epoch; unexpected membership or alignment changes must fail preflight.
- Missing moments are valid only where exposure proves the history was never required. Lost required state and impossible extra state must fail restore.
- A failure after any C3 owner mutates state aborts the attempt without publishing partial progress. Recovery uses the last durable complete update.
- A run may reach its terminal checkpoint before its evaluation sidecar is published. Completion recovery must retain that terminal identity and take no extra training step.
- Valid new results cannot hide missing historical references, and invalid present historical records cannot be silently skipped.
- Coincident standalone markers must remain identifiable without moving parameter-count coordinates or fabricating outcome differences.
- Incomplete hard-kill resource measurements remain explicitly incomplete; unknown peaks or costs must not be reported as zero.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Define one distinct Feature 015 campaign contract with exactly nine fresh, independent run identities and the matrix in EX-001. Preserve Features 013/014 configurations, topology restrictions, sampling, checkpoint compatibility, report validation, and saved artifacts. Extend existing training, ownership, checkpoint, metrics, preflight, terminal-reading, and plotting capabilities; do not introduce a second trainer or optimizer registry.
- **FR-002**: Initialize every new run normally at seed 42 from untrained weights. Fresh standalone runs at dimensions 64/128/256 remain mandatory despite matching historical sizes. Historical or diagnostic trained checkpoints MUST NOT initialize production; continuation uses only a run's own valid checkpoint.
- **FR-003**: Enforce the standalone-first barrier: all four fresh one-epoch standalone terminals MUST pass validation before any full-budget elastic production begins. Short diagnostics are separate from production and do not satisfy this barrier.
- **FR-004**: Every elastic update MUST select one global width across all layers using independent uniform draws with replacement, probabilities [.25, .25, .25, .25], and interval H=1. Perform one forward/backward and one ordinary causal loss, without balancing, holding widths, adaptive selection, or inverse-probability loss weighting. Preserve global sampling mode and correction mode none.
- **FR-005**: All five elastic runs MUST have the same isolated action RNG initialization and exact action sequence, independent of data ordering. All nine runs MUST share first-epoch batches; all five elastic runs MUST share all four deterministic epoch sequences. Optimizer, scheduler, data-cursor, and RNG state remain continuous across epochs.
- **FR-006**: Record ordered labels, physical fractions/dimensions, and actual incremental boundaries from EX-002. Derive block support, optimizer allocation expectations, exposure, and checkpoint identity from those boundaries. Slicing and concat MUST realize the same active dimension at every width; extending eligibility MUST NOT relax prior campaign contracts.
- **FR-007**: S1 MUST retain shared full-tensor AdamW histories. S2 MUST retain four width-specific histories over the same full model tensors, stepping only the selected history. Preserve sliced full-shaped gradients, residual-momentum/full-tensor decay behavior, and zero never-used tail moments in a fresh width history.
- **FR-008**: C1/C2/C3 MUST retain absent gradients and unchanged weights, moments, and counters for inactive concat blocks. Active present zero gradients retain ordinary AdamW behavior. C1 shares histories; C2 allocates width-specific histories lazily, yielding block multiplicities 4/3/2/1 and four common histories after all widths have been selected. Histories operate on shared model weights.
- **FR-009**: C3 MUST have exactly five disjoint owners: one for each incremental FFN block across all layers, including its segment biases, and one for the common remainder, including attention, norms, embeddings, LM head, and any common FFN output bias. Cover every trainable parameter exactly once and deduplicate tied parameters.
- **FR-010**: S1/S2/C1/C2 and standalones MUST use global L2 clipping cap 1.0. C3 MUST independently clip each active owner's combined gradients to cap 1.0, without scaling caps by block size, splitting owners, or applying a second global clip. Step active blocks in order and common once, using the same current global learning rate; advance the global scheduler once per complete update.
- **FR-011**: Preflight MUST reject wrong matrices, controls, budgets, widths, actual counts, probabilities, data identities, or occupied/incompatible run identities. Confirm actual counts on each standalone and all elastic representations instead of substituting nominal width fractions for parameter counts.
- **FR-012**: Exact continuation MUST validate the complete campaign/run identity, width grid, boundaries, ownership, clipping, probabilities, required/lazy histories, numerical state, scheduler horizon, RNG, data cursor, and accounting before live mutation. Reject malformed, model-only, cross-arm, and earlier-campaign checkpoints as new-campaign continuations.
- **FR-013**: Preserve the complete-update failure boundary. Partial C3 owner steps, scheduler failures, or accounting failures after mutation MUST abort and leave the previous durable checkpoint intact; no exception/finalization path may publish the partial state. Resume MUST reproduce actions/batches exactly and numerical state within existing tolerances, including epoch transitions.
- **FR-014**: Retain inspectable resolved controls, source/config hashes, initialization/data provenance, committed action/batch traces, width/block/owner counts, scheduler position, scalar training/validation metrics, summaries, checkpoints, and standard per-run trajectory/resource diagnostics.
- **FR-015**: C1/C3 new identities MUST produce real clipping sidecars and valid metrics/summary references. Record active flags, pre/post norms, applied coefficients/caps, combined norms, and committed observation counts. Terminal readers MUST reject missing or inconsistent clipping evidence, guarding against the earlier arm-name-related omission.
- **FR-016**: Resource reporting MUST distinguish actual optimizer allocations by owner/component/dtype, counters, temporary storage, device peaks, checkpoint size, elapsed execution, and throughput. Derive unequal-block expectations from actual parameter support; do not reuse equal-quarter storage formulas. Retain job IDs and all continuation/failure attempts, sum unique attempt durations, take maximum peaks, include failed/replayed work, and disclose incomplete measurements.
- **FR-017**: Production MUST be gated on passed CPU checks and short real-shape bf16 GPU diagnostics bound to source/config hashes. Evidence MUST cover all new arm types, unequal-block update behavior, exact resume around epoch boundaries, clipping artifacts/readers, invalid-state rejection, partial C3 failure, scheduling/barrier behavior, 24/28-endpoint fixtures, and Feature 013/014 compatibility.
- **FR-018**: When execution is authorized, use a fresh campaign artifact root separate from both prior campaigns, containing snapshots, configurations, diagnostics, launches, logs, checkpoints, and reports. GPU work MUST use sbatch, exclude gpu-[05,50,51,54], and respect user-wide limits of two running/four submitted jobs or stricter live limits. Scheduling MUST be restart-safe with one writer per run and preserve unrelated jobs/helpers.
- **FR-019**: Report only full-budget terminal checkpoints, with checkpoint and ordinary-validation identities frozen before reporting. Use the inherited target-token-weighted causal validation loss and perplexity equal to exp(aggregated loss). Best checkpoints, early elastic endpoints, and trailing averages MUST NOT substitute for terminal results. Completion-only recovery MUST take no additional training step.
- **FR-020**: Export matching CSV/JSON tables of exactly 24 new endpoints and, when historical references validate, exactly 28 combined endpoints. Key each row by campaign/run identity and physical width; retain separate fresh/historical records at shared dimensions. Include group, arm, seed, fraction, dimension, exact active count, loss/perplexity, actual/assigned budgets, evaluation/checkpoint provenance, exposure, clipping, and resource fields as applicable.
- **FR-021**: Reuse only the four original uncorrected Feature 013 standalone terminals at dimensions 64/128/192/256. Revalidate their saved identities/hashes, dimensions, count convention, one-epoch budgets, and evaluation protocol without rewriting historical files. Missing or incompatible references MUST be explicitly reported; no relabeling, omission, or replacement may produce a claimed complete comparison.
- **FR-022**: Complete-report validation MUST reject missing runs/endpoints, duplicate identities, wrong widths/counts/budgets, stale hashes, non-finite or inconsistent loss/perplexity, mismatched evaluation roles/target counts, and nonterminal inputs. A validated 24-endpoint new report may complete independently, but combined comparison completion requires all 28 valid endpoints.
- **FR-023**: Produce combined loss-versus-active-non-embedding-parameters and perplexity-versus-active-non-embedding-parameters figures, each in PNG and PDF. Each MUST contain five consistently colored connected elastic curves at dimensions 32/64/128/256, four disconnected historical standalone markers at 64/128/192/256, and four disconnected fresh standalone markers at 32/64/128/256.
- **FR-024**: Distinguish standalone groups with marker styles and legend entries `Standalone — historical grid` and `Standalone — MatFormer grid`. Make coincident points identifiable without jittering exact parameter-count coordinates or inventing gaps. Annotate seed, dataset, ordinary-validation role, count convention, terminal selection, and one-epoch standalone versus four-epoch elastic budgets.
- **FR-025**: Interpretation MUST compare new elastic curves primarily with fresh standalones, use historical standalones as context/repeat measurements, and separate representation/history effects from independent C3 clipping and changed block sizes. Keep seed-42 conclusions descriptive, without across-seed error bars, significance claims, identical-initial-tensor assumptions, or equal-compute/runtime/exposure claims based on token totals.
- **FR-026**: Keep feature tasks, verification records, and runbook aligned with actual saved evidence as later stages proceed. Clearly distinguish proposed work, passed diagnostics, production completion, new-report completion, and combined-report completion. Specification/planning artifacts alone MUST NOT be presented as experiment results or launch authorization.

### Research & Experiment Requirements *(include for experiment-facing changes)*

- **EX-001 — Fixed fresh-run matrix**: Use the following seed-42 matrix. Arm labels remain recognizable in reports; full identities are unique to Feature 015.

| Run/arm | Representation | AdamW history ownership | L2 clipping | Epochs |
| --- | --- | --- | --- | ---: |
| ST-g125 | Independent dense FFN 32 | Shared | Global 1.0 | 1 |
| ST-g250 | Independent dense FFN 64 | Shared | Global 1.0 | 1 |
| ST-g500 | Independent dense FFN 128 | Shared | Global 1.0 | 1 |
| ST-g1000 | Independent dense FFN 256 | Shared | Global 1.0 | 1 |
| S1 | Slicing | Shared | Global 1.0 | 4 |
| S2 | Slicing | Per width | Global 1.0 | 4 |
| C1 | Concat | Shared | Global 1.0 | 4 |
| C2 | Concat | Per width | Global 1.0 | 4 |
| C3 | Concat | Per incremental block plus common | Independently 1.0 per active owner | 4 |

- **EX-002 — Width and block contract**: Full elastic FFN dimension is 256. Ordered new widths and required counts are below. Counts exclude input embeddings and LM head, retaining other common parameters. Counts are required preflight expectations, not newly verified measurements at specification time.

| Label | FFN fraction | Active FFN dimension | Active non-embedding parameters | Active blocks |
| --- | ---: | ---: | ---: | --- |
| g125 | .125 | 32 | 90,688 | A |
| g250 | .25 | 64 | 115,264 | A/B |
| g500 | .50 | 128 | 164,416 | A/B/C |
| g1000 | 1.00 | 256 | 262,720 | A/B/C/D |

| Block | FFN coordinates | Dimension | Width membership count | Expected activation probability |
| --- | --- | ---: | ---: | ---: |
| A | [0, 32) | 32 | 4 | 1.00 |
| B | [32, 64) | 32 | 3 | .75 |
| C | [64, 128) | 64 | 2 | .50 |
| D | [128, 256) | 128 | 1 | .25 |

Common parameters participate every update. Historical g750 remains fraction .75, dimension 192, and count 213,568; historical g250/g500/g1000 retain their physical dimensions and counts.

- **EX-003 — Inherited controls**: Verify against original uncorrected Feature 013 resolved configurations and audited data. Preserve model dimension 64, four layers, four attention heads, context 128, vocabulary 2,048, initializer standard deviation .02; AdamW learning rate .008, betas (.9, .95), epsilon 1e-8, decay .1; batch 64, accumulation 1, bf16, one process/GPU, cosine schedule with 64 warmup updates over each run's full assigned horizon. Disable learning-rate scaling, pre-nested width warmup, and all membership corrections. Changes are limited to the declared width grid and necessary topology, identity, count, and reporting metadata.
- **EX-004 — Data and budgets**: Reuse the same tokenizer, audited four-role corpus, packing, designated membership, fixed excluded 43-sequence tail, and deterministic epoch-order policy. Each designated epoch contains 5,576,448 packed sequences, 87,132 complete updates, and 713,785,344 training tokens. Excluded sequences never rotate into later epochs. Reject discrepancies instead of adjusting budgets.

| Budget | Each standalone | Each elastic run |
| --- | ---: | ---: |
| Designated epochs | 1 | 4 |
| Complete updates | 87,132 | 348,528 |
| Training tokens | 713,785,344 | 2,855,141,376 |

The nine-run assigned total is 17,130,848,256 training tokens, excluding diagnostics/replayed work. One elastic run matches the aggregate assigned tokens of the four fresh standalones; this does not establish equal compute, time, or direct width exposure.

- **EX-005 — Evaluation isolation**: Preserve disjoint training, ordinary validation, controller, and sealed final-holdout roles. Ordinary validation runs every 64 updates and at completion with inherited evaluation membership and causal target-token weighting. Controller data MUST NOT enter training or selection decisions; final holdout remains sealed throughout this feature.
- **EX-006 — Exposure**: Record measured selections, block activations, owner calls, tokens, and scheduler position separately from expectations. Expected selections are 87,132 per width; expected A/B/C/D activations are 348,528/261,396/174,264/87,132 per elastic run. Expectations are not quotas or guarantees that a width sees every example. All committed accounting MUST reconcile with actual traces.
- **EX-007 — Historical reference**: The known reference is `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/reports/complete-20260910T181951Z/frozen/frozen_manifest.json`. Its availability and compatibility require revalidation before use. The combined report includes only its four original standalone terminals, excluding historical elastic curves, corrected GMC/LMC arms, and Feature 014 inverse-membership runs.
- **EX-008 — Reproducibility and evidence**: Retain the existing pinned environment at `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python` for subsequent verification. Preserve resolved per-run controls, dataset/preprocessing assumptions, source/config hashes, and machine-readable scalar results. Bound checks and short diagnostics establish readiness only; full-budget results require actual terminal evidence.

### Key Entities *(include if feature involves data)*

- **Campaign**: Distinct identity, nine fresh runs, scientific contract, stage barrier, source/config provenance, and separate production/new-report/combined-report completion states.
- **Run**: One fresh standalone or elastic arm with seed, representation, history ownership, clipping, assigned budget, continuation attempts, and terminal artifacts.
- **Width and incremental block**: Ordered label, physical fraction/dimension, exact active count, coordinate boundaries, and supported owners; historical and new grid membership remain explicit.
- **Optimizer owner/history**: Shared, width-specific, or block/common responsibility over model weights, with lazy state, counters, activation history, and clipping policy.
- **Data/evaluation contract**: Tokenizer/corpus identities, disjoint roles, designated membership/excluded tail, epoch orders, and evaluation target/protocol identity.
- **Checkpoint and attempt**: Durable complete-update state bound to the run contract, plus recorded successful/failed work and measured resource costs across continuation.
- **Endpoint**: Campaign/run/physical-width identity, terminal checkpoint/evaluation provenance, exact count, loss/perplexity, budgets, and supporting diagnostics.
- **Comparison report**: Validated 24- or 28-record export, complete-input status, plot series/group membership, and descriptive scientific interpretation.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A researcher can validate exactly nine fresh definitions at the four requested dimensions; all acceptance cases with altered controls, occupied identities, wrong counts, or historical initialization are rejected with identified reasons.
- **SC-002**: Scheduling evidence shows zero full-budget elastic starts before all four new standalone terminals pass validation, including after scheduler restart.
- **SC-003**: All five elastic arms satisfy their declared update/history/clipping semantics at all four widths; C3 has exactly five disjoint owners and one scheduler advance per complete update, and inactive concat blocks show zero changes.
- **SC-004**: Interrupted/resumed checks for all new run types preserve 100% of subsequent actions/batches and numerical agreement within inherited tolerances around epoch boundaries; every incompatible/malformed restore case leaves live state unchanged and every partial C3 failure preserves durable state.
- **SC-005**: Nine completed runs reconcile to the exact assigned epochs, updates, and tokens. All nine first-epoch batch traces match, and all five elastic full action/batch traces match, without enforcing balanced selection counts.
- **SC-006**: Each new run supplies traceable controls, scalar trajectories, exposure, resource accounting, and a durable terminal checkpoint. New C1/C3 clipping evidence covers every committed update and passes terminal validation.
- **SC-007**: A complete new report contains exactly 24 valid endpoints; a complete combined report contains exactly 28, including eight separate standalone records. Every record is traceable to saved terminal/evaluation provenance and all defined corrupt/missing-input cases prevent false completion.
- **SC-008**: Both requested combined figures are delivered in two formats, for four files total. Each contains five four-point elastic curves and eight disconnected standalone markers with exact parameter counts, identifiable overlaps, group legends, and budget/evaluation annotations.
- **SC-009**: A reviewer can assess new elastic results against fresh baselines, historical repeat measurements at three shared sizes, and measured exposure/clipping/resource differences using saved artifacts alone. Interpretation addresses these comparisons without across-seed significance or unsupported compute equivalence.
- **SC-010**: Compatibility checks retain Feature 013/014 scientific contracts and historical artifact identities; the final holdout receives zero evaluations. Documentation completion states match actual saved evidence, including an explicit outstanding combined comparison whenever references are unavailable.

## Assumptions

- Intended users are research practitioners and reviewers. Scientific dimensions, optimizer semantics, precision, output formats, and execution constraints are user-specified protocol requirements; software design and command interfaces belong in planning.
- The source prompt is authoritative for Feature 015. Existing Feature 013/014 documents provide context; their old launch approvals and stale completion statements are not current authorization or new-campaign evidence.
- Existing audited corpus/tokenizer and original standalone references are dependencies to revalidate. This specification does not assert fresh artifact validation or successful new model-count checks.
- Existing numerical determinism tolerances, clipping stabilization, and terminal recovery conventions apply. No learning-quality, speed, or memory improvement is promised as an acceptance condition.
- The exact new campaign root, version identifier, and operational command names are planning choices subject to fresh identities and the explicit compatibility requirements.
- Reuse existing capabilities and narrowly extend eligibility for this campaign. No additional seeds, extra production arms, corrected losses/updates, inverse-membership/adaptive sampling, forced balanced draws, per-width model replicas, or distributed campaign execution are in scope.
- Specification validation establishes readiness for `/speckit-plan`. Later implementation and GPU execution require separate authorization in the conversation.
