# Feature Specification: TinyStories Optimizer Ownership Comparison

**Feature Branch**: `013-tinystories-optimizer-ownership`  
**Created**: 2026-09-09  
**Status**: Original campaign completed; correction extension authorized 2026-09-10
**Input**: [Feature request](../../notes/tinystories_optimizer_ownership_speckit_prompt.md), beginning at “Feature request”; [companion design](../../notes/tinystories_optimizer_ownership_experiments.md).

This feature enables researchers to distinguish learning effects and resource costs of feed-forward network (FFN) parameter representation, AdamW history ownership, and gradient clipping in TinyStories-Instruct. An optimizer history consists of gradient moments and parameter-local update counters; multiple histories in an elastic model still update the same shared weights. A width specifies the active FFN intermediate dimension, while attention and other common dimensions stay fixed.

This is a new experiment series extending Feature 12. Its nine-run protocol supersedes neither historical specifications nor existing runs. Preparing the specification or implementation does not authorize launching training.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Establish a valid nine-run comparison (Priority: P1)

A researcher prepares nine fresh runs and verifies that their data, starting conditions, widths, sampling, and training budgets support the intended comparisons before spending training resources.

**Why this priority**: A mismatch in budgets or scientific controls invalidates every downstream comparison.

**Independent Test**: Validate a complete campaign description against the audited corpus without training, and deliberately change one control at a time to verify rejection.

**Acceptance Scenarios**:

1. **Given** the audited prepared corpus and settled protocol decisions, **When** the researcher validates the nine-run matrix, **Then** it resolves four independent one-pass standalones and five four-pass elastic settings at seed 42, with the exact budgets and controls below.
2. **Given** the first deterministic epoch order, **When** data traces are compared, **Then** all nine runs share that first epoch and all five elastic runs share the remaining three orders; each designated example occurs once per epoch.
3. **Given** five matched elastic configurations, **When** their action traces are generated, **Then** they use exactly the same sequence of independent uniform draws with replacement, one global width per update, without balancing counts.
4. **Given** a stale manifest, different alignment, changed clipping policy, incorrect seed, wrong model dimensions, or reused historical identity, **When** campaign preflight runs, **Then** it identifies the discrepancy and rejects the campaign.
5. **Given** independent standalone models and matching elastic widths, **When** model dimensions and active counts are inspected, **Then** the standalones have physically appropriate dense FFNs, equal active parameter counts, and independently trainable weights.
6. **Given** any of the nine fresh runs, **When** its initialization contract is inspected, **Then** it uses seed 42 and the declared initializer settings through its normal model construction path; exact initial-value equality across representations or widths is not required.

---

### User Story 2 - Execute the intended ownership and clipping semantics (Priority: P1)

A researcher can run a short controlled diagnostic through the actual slicing and concatenation models and establish what changes when optimizer histories or clipping groups change.

**Why this priority**: The experiment must measure the declared interventions; memory savings alone do not establish correct behavior.

**Independent Test**: Use matched short update sequences on real FFN model paths and inspect active/inactive parameters, histories, counters, and clipping records.

**Acceptance Scenarios**:

1. **Given** an S1 wider update followed by a narrower update, **When** an unused tail is inspected, **Then** its full-shaped zero gradient retains ordinary residual-momentum and decoupled-decay behavior, and its tensor counter advances.
2. **Given** a fresh S2 run, **When** widths alternate, **Then** only the selected width's history changes, its never-used tail moments stay zero, full-sized moments remain allocated, and full-tensor decay still applies on its steps.
3. **Given** C1, C2, or C3 with a previously active wider block, **When** a narrower update excludes that block, **Then** its gradient is absent and its weights, moments, and counters are unchanged. An active present zero gradient remains subject to ordinary AdamW behavior.
4. **Given** a C2 run that has selected every width, **When** histories are counted, **Then** blocks A/B/C/D have respectively 4/3/2/1 histories and common parameters have four; never-active width/block histories remain absent, including after resume.
5. **Given** C3's complete disjoint ownership partition, **When** any width is selected, **Then** exactly its active owners step once, common parameters are clipped and stepped once, and the global scheduler advances once after the complete update.
6. **Given** two active C3 groups, **When** only one group's input norm changes, **Then** the other group's clipping coefficient stays unchanged; every active group respects its chosen cap with no subsequent global rescale.
7. **Given** a short C3 diagnostic temporarily using C1's global clipping, **When** initialization, data, actions, rates, and update order match, **Then** it agrees with C1 within the project's numerical tolerance. Actual C3 with separate clipping is not required to match C1.

---

### User Story 3 - Resume without altering the experiment (Priority: P1)

A researcher resumes an interrupted run with continuous action, data, optimizer, learning-rate, and resource accounting, including across epoch boundaries.

**Why this priority**: Interrupted runs must remain scientifically comparable to uninterrupted runs.

**Independent Test**: Compare uninterrupted and interrupted/resumed short runs inside an epoch and at epoch boundaries, and inject malformed checkpoints and owner-step failures.

**Acceptance Scenarios**:

1. **Given** a durable checkpoint for any arm, **When** it resumes, **Then** actions and batches match the uninterrupted trace exactly, numerical model/history/scheduler state agrees within existing determinism tolerances, and inactive lazy state stays absent.
2. **Given** a malformed, incompatible, or model-only checkpoint, **When** exact resume is requested, **Then** validation rejects it before changing the live model or training state.
3. **Given** C3 where one owner has updated and another fails, **When** failure handling runs, **Then** training aborts, the partial update is not saved as a committed resumable step, and recovery uses the prior durable checkpoint.
4. **Given** a run resumed more than once, **When** its terminal summary is produced, **Then** elapsed runtime is cumulative and reported peak memory is the maximum across continuation attempts.
5. **Given** a standalone or elastic run at its assigned endpoint, **When** completion is reconciled, **Then** it has exactly one or four complete designated epochs, respectively, with no partial final epoch, extra epoch, or epoch-boundary clock reset.

---

### User Story 4 - Audit outcomes, clipping, and resource costs (Priority: P2)

A researcher can inspect learning trajectories, realized width exposure, clipping behavior, and memory/execution costs using saved artifacts without rerunning training.

**Why this priority**: These records explain observed quality differences and distinguish persistent optimizer storage from total resource costs.

**Independent Test**: Inspect short-run artifacts from each arm, verify accounting against update traces, and produce individual diagnostic plots.

**Acceptance Scenarios**:

1. **Given** a completed or resumed diagnostic run, **When** its artifacts are inspected, **Then** resolved controls, provenance, action/batch traces, epochs, actual tokens, owner counts, scheduler position, scalar metrics, summary, and resumable state are available and reconcile.
2. **Given** C1 and C3 runs, **When** clipping records are compared, **Then** pre/post norms, coefficients, active flags, combined norms, and clipping frequencies by width/group expose one global coefficient for C1 and independent coefficients for C3 without changing gradients or sampling.
3. **Given** measured optimizer state, **When** memory accounting is reported, **Then** actual dtypes and owner/component allocations determine bytes; counters, temporary concatenation storage, peak memory, throughput, wall time, and checkpoint size remain distinguishable.
4. **Given** randomly unequal width counts, **When** exposure is reported, **Then** selected-width counts and block activations are separate, measured values are retained, and expected equal exposure is not described as guaranteed example coverage.

---

### User Story 5 - Produce the complete terminal endpoint comparison (Priority: P2)

A researcher creates two auditable combined figures that compare quality at equal active non-embedding parameter counts using all terminal endpoints under one evaluation protocol.

**Why this priority**: The campaign's deliverable is an interpretable learning comparison supported by the nine runs, not merely a set of memory measurements.

**Independent Test**: Supply a complete controlled fixture with nine run identities and 24 endpoints; verify exports and figure structure, then reject missing, duplicated, or mismatched inputs.

**Acceptance Scenarios**:

1. **Given** nine valid terminal checkpoints after their full assigned budgets, **When** comparison reporting runs, **Then** it exports 24 endpoints: four widths for each of S1/S2/C1/C2/C3 and one endpoint for each standalone.
2. **Given** those endpoints under the chosen common evaluation role, **When** figures are produced, **Then** both loss and perplexity figures contain five connected elastic curves and four disconnected standalone markers, with exact active non-embedding counts on the horizontal axis and consistent setting colors.
3. **Given** a best-validation checkpoint, first-epoch elastic checkpoint, mixed evaluation role, wrong budget, duplicate/missing endpoint, non-finite value, or incompatible count convention, **When** a complete report is requested, **Then** it is rejected. A partial report requires a separate explicit request and visible partial labeling.
4. **Given** the sealed final holdout, **When** the campaign's combined figures are produced, **Then** all 24 endpoints use ordinary validation and the holdout remains sealed. A separately requested future holdout comparison requires freezing all nine terminal identities before explicitly evaluating all 24 endpoints under the same protocol; it must not mix evaluation roles.
5. **Given** one-seed results, **When** the report interprets them, **Then** it distinguishes representation, history ownership, and clipping interventions and makes descriptive seed-42 claims without across-seed error bars or unsupported equal-compute claims.

### Edge Cases

- The prepared role contains 43 more packed sequences than fit complete updates: disclose the fixed excluded tail and require exact traversal of the designated aligned set; reject any unexpected manifest/alignment change.
- A width may be absent or repeated in a short random trace; never force selection counts to balance or allocate unused C2 state to make diagnostics look complete.
- A zero-valued active gradient must remain different from an absent inactive gradient; stale gradients must not activate an excluded block.
- Tied embeddings/output-head parameters must have exactly one C3 owner, and any common FFN output bias must not be duplicated among quarter owners.
- Invalid quarter dimensions or incomplete/overlapping owner coverage must fail preflight before training.
- Epoch boundaries and interruptions must not reset RNG streams, data orders, histories, learning rates, or global update accounting.
- A failure during a multi-owner update must never publish a partially updated checkpoint as resumable progress.
- A lazily absent state is valid only when exposure proves it was not yet required; missing previously required moments or counters must fail resume validation.
- Non-finite loss/perplexity or inconsistent evaluated target-token counts/provenance must prevent a complete comparison.
- Identical-looking endpoints remain distinct labeled series; no invented points or statistical uncertainty bands are added.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Provide explicit campaign configuration, a nine-run preflight, execution instructions, and a dedicated comparison workflow using the existing trainer and artifact conventions. Preserve existing default behavior and historical artifacts; do not create a second trainer or a general-purpose optimizer registry.
- **FR-002**: Support exactly the fresh runs in the experiment matrix below, at seed 42 only. Widths mean fractions of the full FFN intermediate dimension, with fixed attention/common dimensions. Standalones MUST be independent dense models trained from untrained starting weights, not extracted trained nested checkpoints.
- **FR-003**: Every elastic update MUST use one global width across all layers, one forward/backward, and one loss. Draw widths independently with replacement with probabilities [0.25, 0.25, 0.25, 0.25] and sampling interval H=1. All elastic arms MUST share the same action RNG stream and complete sampled sequence.
- **FR-004**: All nine runs MUST share the same first-epoch batch sequence. Elastic arms MUST share all four deterministic epoch orders. Each epoch visits the designated fixed aligned training set once, with continuous RNG, data cursor, optimizer history, and scheduler state across boundaries.
- **FR-005**: Preflight MUST audit corpus/tokenizer manifests, role separation, packing, batch alignment, exclusions, and exact budgets. The available role and the designated aligned set MUST be reported separately. Reject discrepancies rather than silently altering the budget or truncating the designated epoch.
- **FR-006**: All elastic arms MUST use shared model weights; multiple optimizers own distinct histories, not weight replicas. Preserve ordinary AdamW semantics, including zero-present versus absent gradients, parameter-local counters, and decoupled weight decay.
- **FR-007**: S1 MUST use full-sized sliced FFN tensors and one shared history. Inactive coordinates receive zero entries in full-shaped gradients; residual moments and full-tensor decay may change unused tails, and tensor counters advance.
- **FR-008**: S2 MUST maintain four independent width-owned histories over the same full-sized model tensors, including common parameters. Only the selected optimizer steps; its never-used tails retain zero moments from fresh initialization, while full-tensor decay and full-sized moment allocation are retained.
- **FR-009**: C1/C2/C3 MUST use independent quarter parameter tensors and concatenate only active quarters. Clear gradients to absent between updates. Inactive blocks MUST have absent gradients and receive no weight update, decay, moment change, or counter advance. Active present zero gradients MUST retain ordinary AdamW behavior.
- **FR-010**: C1 MUST maintain one history per physical parameter, shared by widths that activate it. C2 MUST maintain four width-owned histories over shared block/common parameters, allocating state only where that owner has received a gradient. After all widths are selected, A/B/C/D MUST have 4/3/2/1 histories respectively, and common parameters four. Construction and resume MUST preserve lazy absence.
- **FR-011**: C3 MUST partition all trainable parameters into exactly five disjoint owners: O-A/O-B/O-C/O-D own corresponding FFN quarters across all layers, including gate/up/down blocks and segment biases; O-common owns attention, norms, embeddings, LM head, and any common FFN output bias. Deduplicate tied parameters; no parameter may have two owners or no owner.
- **FR-012**: C3 MUST activate O-common plus A, A/B, A/B/C, or A/B/C/D for g250/g500/g750/g1000 respectively. After one backward, independently clip each active owner's joint L2 gradient norm, scaling the entire group's gradients together, then step each active owner once. Do not clamp individual elements or apply a second global clip. Inactive owners MUST not advance.
- **FR-013**: C3 MUST use an L2 gradient-norm clipping threshold of 1.0 independently for every active ownership group: O-A, O-B, O-C, O-D, and O-common. Record the threshold for every owner in the resolved configuration.
- **FR-014**: Every complete update MUST advance one global scheduler exactly once, after all required owner steps. All owners use the same current global learning rate, with no width-local or block-local schedules. Use each run's complete assigned budget as its schedule horizon.
- **FR-015**: Each run MUST initialize a fresh model using seed 42, the declared initializer settings, and its normal model construction path. Record the seed and initialization provenance. Canonical dense-model copying, cross-representation value mapping, standalone extraction from a canonical initialization, and exact cross-run initial-value equality are not required. Reports MUST describe initialization as matched by seed, without claiming identical initial weights across model shapes or representations.
- **FR-016**: Version resumable checkpoints with representation, owner/group identity, clipping contract, all required moments/counters, global scheduler, RNG/action state, data cursor, epochs, and budget identity. Validate shape, identity, required/absent state, and cross-arm compatibility before mutating live state. Model-only checkpoints MUST be rejected for exact resume.
- **FR-017**: Treat C3 owner steps as one logical update. If any owner fails after another has updated, abort and recover from the prior durable checkpoint; never save that partial update as committed progress. Do not introduce per-step full-state snapshots for rollback.
- **FR-018**: Save resolved configurations, ordinary scalar metrics, summaries, resumable checkpoints, individual loss/perplexity trajectories, and resource plots. Records MUST identify representation, ownership, clipping, run/seed, initialization/data provenance, widths/actions, batches, epochs, actual tokens, attempted/committed updates, owner counts, and scheduler position. Full states belong in checkpoints, not metric rows.
- **FR-019**: Record actual width selections and block activations separately. Reconcile selected widths, owner calls, global updates, epochs, tokens, and scheduler position at completion; multiple C3 owner calls MUST not count as multiple global updates.
- **FR-020**: Measure actual optimizer-state bytes by owner/component and dtype, peak memory, throughput, wall time, and checkpoint size. Preserve cumulative runtime and maximum peak-memory accounting across resumes, and distinguish persistent state from temporary concatenation buffers.
- **FR-021**: For C1/C3, log per-group pre/post-clipping norms, coefficients, active flags, combined norms, and clipping frequencies by selected width and group. C1 uses one global coefficient; C3 uses independent group coefficients. Diagnostics MUST not alter gradients or action streams.
- **FR-022**: Both combined figures MUST use ordinary validation for all 24 endpoints, with identical evaluation data/protocol and terminal checkpoints after assigned budgets. Keep final holdout sealed. Any future final-holdout comparison requires a separate explicit request, prior freezing of all nine terminal checkpoint identities, and explicit evaluation of all 24 endpoints under one common holdout protocol; never mix ordinary-validation and holdout endpoints.
- **FR-023**: Evaluate each elastic terminal model at all four widths and each standalone at its own width. Use target-token-weighted causal loss and perplexity equal to the exponential of aggregated loss, not averaged batch perplexities. Record evaluated token counts and evaluation/checkpoint identities.
- **FR-024**: Produce exactly two combined endpoint figures, each in PNG and PDF: active non-embedding parameter count versus perplexity, and that same count versus language-model loss. Each MUST show five elastic curves connecting four measured width endpoints, plus four standalone markers with no connecting line, and consistent setting colors across figures. No across-seed error bars are permitted.
- **FR-025**: Use exact active `non_embedding_parameters` under the repository convention: exclude input embeddings and the LM output head; include active FFN, attention, norms, and other counted common parameters. Do not substitute stored nested-model size, optimizer memory, width labels, or width fraction times total size.
- **FR-026**: Export the same auditable 24-row endpoint table as CSV and JSON, containing setting/representation/ownership/clipping, width, exact active count, loss/perplexity, evaluated tokens, assigned and actual training budgets, evaluation identity, checkpoint hash, exposure, and resource measurements. Annotate both figures with dataset, seed, evaluation role, count definition, terminal rule, and per-run versus panel-level budgets.
- **FR-027**: Complete reporting MUST validate all nine runs and 24 endpoints against the full scientific controls, including intended arm differences. Reject missing/duplicate/non-finite or mismatched inputs. Do not rely solely on a historical paired-control hash that omits representation or clipping. Partial reports require a separate explicit request and clear labels; plotting MUST never automatically evaluate sealed holdout data.
- **FR-028**: Verify the semantics above through actual slicing/concatenation model paths, dense standalone shape/count checks, randomized trace checks, clipping independence, a short C3-with-global-clipping diagnostic against C1, exact action/data resume comparisons, numerical state comparisons under existing tolerances, malformed-load rejection, multi-owner failure handling, nine-run preflight, and 24-endpoint figure/export checks. This preparation MUST NOT launch the full campaign.

### Research & Experiment Requirements *(include for experiment-facing changes)*

#### Fixed experiment matrix

- **EX-001**: The campaign MUST contain exactly these nine fresh seed-42 runs:

| ID | Representation | AdamW ownership | Clipping | Passes |
| --- | --- | --- | --- | --- |
| ST-g250 | Independent dense 25% FFN | One optimizer | Global L2 norm 1.0 | 1 |
| ST-g500 | Independent dense 50% FFN | One optimizer | Global L2 norm 1.0 | 1 |
| ST-g750 | Independent dense 75% FFN | One optimizer | Global L2 norm 1.0 | 1 |
| ST-g1000 | Independent dense 100% FFN | One optimizer | Global L2 norm 1.0 | 1 |
| S1 | Nested slicing | Shared | Global L2 norm 1.0 | 4 |
| S2 | Nested slicing | Per width | Global L2 norm 1.0 | 4 |
| C1 | Nested concatenation | Shared | Global L2 norm 1.0 | 4 |
| C2 | Nested concatenation | Per width | Global L2 norm 1.0 | 4 |
| C3 | Nested concatenation | Four cross-layer quarter owners plus common | Separate L2 norm 1.0 per active owner | 4 |

#### Audited budget and data contract

- **EX-002**: Preserve one pass per standalone and four passes per elastic run. A pass is a complete epoch over the designated aligned optimizer-training set, not a minibatch. With the inherited complete-update alignment, use:

| Quantity | Each standalone | Each elastic run |
| --- | --- | --- |
| Complete passes | 1 | 4 |
| Training tokens | 713,785,344 | 2,855,141,376 |
| Tokens per update | 8,192 | 8,192 |
| Global updates | 87,132 | 348,528 |

The prepared corpus audit on 2026-09-09 passed for 89 shards and the stored permutation. The optimizer role contains 5,576,491 sequences, or 713,790,848 tokens. The existing complete-update alignment designates the first 5,576,448 entries of the stored training permutation and excludes its fixed final 43 sequences (5,504 tokens). Each epoch reorders the same designated set; excluded sequences do not rotate into later epochs. This explicit inherited alignment yields exactly the budgets above, with no partial designated epoch. Preflight MUST confirm the same counts, exclusion, corpus identity, and order identity before accepting these budgets. [Audit evidence](inspection.md).

- **EX-003**: Keep training, ordinary validation, controller, and sealed final holdout disjoint. Evaluation roles MUST never enter training. Preserve tokenizer, manifests, data seed 42, and deterministic epoch orders. Training-token accounting includes the 128-token packed sequences; evaluated target-token counts use causal prediction targets and are recorded separately.
- **EX-004**: Report expected selections of 87,132 per width per elastic run alongside actual random counts. Expected selected-width token volume equals one pass, but does not imply every example appears once for that width. Smaller blocks participate in wider selections; expected A/B/C/D activations are 348,528/261,396/174,264/87,132, with common parameters active on every update. Do not enforce these expectations.
- **EX-005**: Label budgets as matched aggregate training tokens: one elastic run equals the complete four-standalone panel. The full campaign consumes 24 one-pass budgets (17,130,848,256 training tokens). Do not claim equal per-run budget, FLOPs, wall time, or exact width exposure. Runs MUST start with fresh identities and full schedule horizons, not extend prior one-pass elastic runs.

#### Matched recipe

- **EX-006**: Use model dimension 64, four layers, four attention heads, context 128, vocabulary 2,048, initializer standard deviation 0.02, and FFN fractions 0.25/0.50/0.75/1.00. The current model configuration resolves full FFN dimension 256 and active widths 64/128/192/256; preflight MUST verify exact resolved dimensions and active parameter counts for all representations and dense standalones.
- **EX-007**: Use AdamW learning rate 0.008, betas (0.9, 0.95), epsilon 1e-8, weight decay 0.1; batch 64, accumulation 1, bf16, one process on one GPU, and no learning-rate world-size scaling. Use ordinary causal language-model loss without width weighting. Disable membership/LMC/GMC correction and pre-nested width warmup.
- **EX-008**: Use cosine scheduling with 64 warmup updates and each run's full horizon: 87,132 standalone or 348,528 elastic. Elastic learning-rate traces MUST match each other; standalone traces MUST match each other. Traces between the two horizon groups need not match at equal absolute steps. Run ordinary validation every 64 updates and at terminal completion.
- **EX-009**: The resolved protocol uses seed-42 initialization through normal model construction (FR-015), C3's threshold of 1.0 per active group (FR-013), and ordinary validation for all combined endpoints with final holdout sealed (FR-022). All three choices are confirmed by the user. The short C1/C3 plumbing diagnostic still uses matched starting values as a test control; it does not impose canonical initialization mapping on the nine campaign runs.

#### Resource expectations and interpretation

- **EX-010**: Compare measured ordinary AdamW first/second moment element totals after every width is exposed against these expectations, where F is full-width quarter-owned FFN parameter count and R is common parameter count (including any common FFN output bias). Include segment-specific biases in F where present; counters and temporary buffers are excluded from the estimates.

| Setting | Expected persistent moment elements |
| --- | --- |
| S1 / C1 / C3 | 2 × (F + R) |
| S2 | 2 × (4F + 4R) |
| C2 | 2 × (2.5F + 4R) |

C2's theoretical 37.5% reduction relative to S2 applies only to FFN moments, not total training memory. Convert measured elements to bytes using actual state dtypes; bf16 compute alone does not determine state dtype.

- **EX-011**: Explain that separate clipping intentionally changes C3 relative to C1. With threshold 1.0 per group, N active disjoint groups allow combined gradient norm up to √N: √2 at g250 and √5 at g1000, versus C1's global cap 1.0. These bounds concern gradients, not AdamW parameter-update norms. C3 block histories remain shared across widths activating them.
- **EX-012**: Report S1/S2 and C1/C2 as history-ownership comparisons within each representation; S1/C1 and S2/C2 as representation comparisons that also change inactive-tail momentum/decay, counters, or allocation; C1/C3 as global versus separate clipping with different combined caps. Compare each elastic endpoint with the matching standalone at equal active parameter count. Findings MUST be descriptive for seed 42, without claims of multi-seed robustness or pure memory/history isolation unsupported by the intervention.
- **EX-013**: Retain individual trajectory/resource figures in addition to the two combined endpoint comparisons. Use artifact stems `optimizer_ownership_perplexity_vs_non_embedding_parameters`, `optimizer_ownership_loss_vs_non_embedding_parameters`, and `optimizer_ownership_endpoints` for the respective exports.

### Key Entities *(include if feature involves data)*

- **Campaign**: Nine fresh run identities, seed, fixed matrix, settled scientific controls, corpus/initialization identities, assigned budgets, and completion validation.
- **Run**: One independently executed arm with model representation, history ownership, clipping policy, width configuration, schedule horizon, continuation history, and saved outputs.
- **Width and active subnetwork**: FFN prefix fraction and resolved dimension; exact active non-embedding count; selected-width exposure distinct from block participation.
- **Optimizer owner**: Shared, width-specific, or quarter/common ownership identity, covered parameters, allocated histories, counters, activation count, current global rate, and clipping group.
- **Data contract**: Immutable four-role corpus identity, stored permutation, fixed aligned epoch membership, explicit exclusions, epoch orders, and current data cursor.
- **Training update**: One selected width and batch, gradient/clipping diagnostics, all required owner steps, committed status, token increment, and one global scheduler advance.
- **Resumable checkpoint**: Versioned compatible model/history/clock/RNG/data state and provenance for a complete committed update; distinct from a model-only checkpoint.
- **Endpoint**: One terminal run/width evaluation with exact active count, loss/perplexity, target-token count, evaluation identity, training budget, and checkpoint hash.
- **Comparison report**: Validated 24-row table, two combined figures in two formats, resource/exposure evidence, and interpretation of intended intervention differences.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A researcher can validate all nine intended run definitions before training; every deliberately mismatched scientific control in the acceptance cases is rejected with an identified reason.
- **SC-002**: Complete runs reconcile to exactly one or four designated epochs and 87,132 or 348,528 global updates. The five elastic action and batch traces match exactly, and measured width/activation counts reconcile without enforced balancing.
- **SC-003**: All five elastic settings satisfy their declared active/inactive weight, history, and clipping behavior in the model-based diagnostics; inactive concatenation blocks show zero changes, and every complete C3 update records one global clock advance.
- **SC-004**: Interrupted/resumed validation traces, including epoch-boundary cases, preserve 100% of actions and batches and agree numerically within existing project tolerances; every malformed/incompatible checkpoint case is rejected before live-state mutation.
- **SC-005**: Every completed run supplies inspectable scientific controls, scalar trajectories, exposure, resource measurements, a reconciled summary, and a durable terminal checkpoint; resumed accounting includes all continuation attempts.
- **SC-006**: A complete report contains exactly 24 valid endpoints and two combined figures, each with five four-point elastic curves and four disconnected standalone points. All endpoints use one evaluation role and the correct terminal budget, with exact matching active counts for corresponding widths.
- **SC-007**: A researcher can trace every plotted point to its run, budget, evaluated data, count definition, and checkpoint identity using saved artifacts alone; incomplete or mismatched comparisons cannot be presented as complete.
- **SC-008**: The report answers all six comparison questions in EX-012 using measured learning and resource outcomes, explicitly distinguishes token matching from compute matching, and makes no across-seed claims from this single-seed campaign.

## Assumptions

- Target users are researchers and reviewers of controlled language-model experiments. Scientific terms and exact numerical controls are necessary requirements; implementation interfaces and code design belong in planning.
- The supplied prompt and subsequent user clarifications are authoritative where the companion note or Feature 12 differs. The user selected seed-only initialization in place of the proposed canonical mapping. Feature 12's three-seed six-run balanced-cycle recipe/analyzer is not the protocol for this series.
- “Pass” inherits the repository's explicitly recorded complete-update-aligned epoch membership. It is not a claim that all available packed sequences are consumed: the fixed 43-sequence exclusion is disclosed in EX-002 and must be revalidated, not silently recomputed against a changed corpus.
- The currently prepared corpus and tokenizer remain available for planning and preflight. The read-only audit establishes integrity and alignment, not successful implementation of the nine arms or authorization to train.
- Existing numerical determinism tolerances and clipping stabilization conventions apply. Learning superiority or a particular speed/memory improvement is not guaranteed by feature acceptance.
- Excluded scope: additional seeds, adaptive/nonuniform sampling, balanced cycles or forced equal counts, multiple width losses per update, compact slicing moments, per-width model replicas, extra full-budget clipping-equivalence arms, block-local learning-rate schedules, distributed block/per-width execution, unmatched historical FineWeb standalones, and automatically launching the campaign.


## Correction comparison extension (2026-09-10)

The [continuation request](../../notes/tinystories_gmc_lmc_speckit_prompt_2026-09-10.md)
authorizes implementation, validation, six fresh runs and reporting in this same
feature/branch. The original nine-arm requirements above remain the uncorrected
protocol; statements about future launch there describe that original workflow.
No completed artifact is relabeled, overwritten or retrained.

### User Story 6 — Compare concat membership corrections (Priority: P1)

A researcher compares C1/C2/C3 with GMC and LMC against the completed uncorrected
concat and standalone references, with matched seed, data/actions and budgets.

- **FR-029**: Materialize exactly C1-GMC, C1-LMC, C2-GMC, C2-LMC, C3-GMC,
  C3-LMC with fresh distinct identities. Each retains its original arm's controls,
  seed-42 normal fresh initialization, four trained labels, uniform single-width
  replacement sampling, independent action/data RNG, 348528 updates and
  2855141376 tokens, full cosine horizon and ordinary-validation policy.
- **FR-030**: GMC applies trained-width-count/membership-count factors
  [1, 4/3, 2, 4] to FFN block weights and block-local biases before clipping.
  LMC retains those hooks and additionally scales the entire completed AdamW
  parameter change, including decay, by those factors after stepping. Common
  parameters, including shared down bias, have no direct multiplier. Moments and
  counters are not directly rescaled. Inactive blocks with absent gradients retain
  weights/state/counters; active blocks receive each intervention once.
- **FR-031**: C1/C2 preserve global cap 1.0 and shared/per-width histories;
  C3 preserves five disjoint owners and cap 1.0 per active owner. Apply correction
  within the complete-update failure boundary, before the single global clock
  advance; owner, correction and bookkeeping failures abort without publishing
  partially corrected resumable state. Record base LR separately from factors.
- **FR-032**: Bind the explicit versioned correction contract to configuration,
  checkpoints and provenance. Reject correction mismatches before restore. Preserve
  original strict nine-arm validation, historical hashes and slicing behavior.
  Keep compact accounting; measure correction overhead without full optimizer
  snapshots or growing per-update history work.
- **FR-033**: Before launch, pass CPU real-model checks in all six arms (each width,
  independent explicit-LR AdamW reference with decay/biases, hooks/clipping, moments,
  inactive tails, once-only C3/clock, resume, mismatch and failure injection).
  Then pass sbatch GPU preflight for all six, including finite losses, factors,
  save/restore progress and recent steady-state throughput. Failed gates block launch.
- **FR-034**: Keep every experimental/operational artifact beneath
  `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/campaigns/concat-gmc-lmc-v1`
  after checking for conflicts. Run GPU work only through sbatch, exclude
  gpu-[05,50,51], respect two running/four submitted GPU jobs across the user,
  use one writer/process/GPU per run and restart-safe submission records. Resume
  only each new run's own checkpoint; reconcile rolled-back work and resources.
  Record source revision/hashes, configs, identities, jobs and attempts; monitor
  recent progress/utilization/checkpoint health and estimated completion.
- **FR-035**: Complete all six terminals, validate action/batch digests against
  original saved artifacts, and export CSV/JSON for 40 endpoints (24 corrected,
  12 uncorrected concat, four standalone). Produce loss/perplexity versus active
  non-embedding parameters in PNG/PDF, brown standalone triangles with one
  `Standalone` legend entry, exact axis labels and the shortened original footnote.
  Produce a full-range four-panel ordinary-validation loss-progress plot, one
  width per panel. Interpret quality, exposure, clipping and measured runtime as
  paired seed-42 observations, without across-seed significance or holdout evaluation.

Acceptance: the six-arm matrix rejects any changed scientific control; actual
single-width updates match independent corrected AdamW references; interrupted
runs match uninterrupted runs and reject mismatched contracts; all six full-budget
terminals yield 40 validated endpoints and all requested figures. These extend
US1–US5 without replacing their original acceptance cases.

### Clarifications — 2026-09-10

The supplied continuation request resolves all material choices: LMC is combined
GMC plus full-change scaling (including decay), numerator four even for one sampled
width, C3 independent clipping, seed-only fresh initialization, original horizon
and ordinary validation. All four custom labels are trained, so the separately
noted subset-label issue is outside scope. No scientific question remains open.

### Additional success criteria

- **SC-009**: All six resolved paths pass CPU and GPU numerical/lifecycle gates
  before any production submission.
- **SC-010**: Six complete, unique full-budget runs retain paired action/batch
  digests and valid terminal/resume/resource provenance, with zero holdout evaluation.
- **SC-011**: Saved-artifact reporting exports 40 endpoints, two endpoint figures
  and one four-panel full-progress figure, each as PNG/PDF, with measured caveats.
