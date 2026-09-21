# Feature Specification: TinyStories Inverse-Membership Sampling Comparison

**Feature Branch**: `014-tinystories-inverse-membership`  
**Created**: 2026-09-11  
**Status**: Specification validated  
**Input**: [Feature request](../../notes/tinystories_inverse_membership_sampling_speckit_prompt_2026-09-11.md), beginning at “Feature request”, and explicit diagnostic/production submission authorization.

Researchers need to measure how a fixed change in width-selection probabilities changes learning, exposure and cost in five existing optimizer-ownership settings. A width selects a prefix of a shared feed-forward network (FFN); a quarter block is the newly included part at each successive width. Optimizer ownership determines which gradient histories are maintained over shared weights, not how many independent models exist.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Establish the five-run sampling comparison (Priority: P1)

A researcher validates five fresh seed-42 experiments whose only change from their respective uncorrected historical counterparts is the width-selection policy.

**Why this priority**: Incorrect probabilities, data or budgets invalidate the comparison before training begins.

**Independent Test**: Accept the exact five-arm recipe and reject changed controls; compare deterministic action and batch traces without full training.

**Acceptance Scenarios**:

1. **Given** the five arms and audited inputs, **When** preflight resolves the campaign, **Then** every arm has the matrix, probabilities, fresh identity, model counts and four-epoch budget specified below.
2. **Given** the production selection mechanism, **When** all five action streams are compared, **Then** they are identical independent categorical draws with replacement, one global width per complete update, with the stated probability mapping.
3. **Given** the same data contract, **When** the five batch streams are compared, **Then** they match each other and the inherited deterministic epoch orders, independently of action selection.
4. **Given** changed probabilities, missing/reordered widths, corrections, shortened budgets, occupied run identities or incompatible manifests, **When** preflight runs, **Then** it rejects the mismatch and identifies its cause.

---

### User Story 2 - Preserve real optimizer behavior and exact continuation (Priority: P1)

A researcher verifies that nonuniform sampling preserves each arm's declared gradient, history, clipping and continuation behavior through actual model updates.

**Why this priority**: Accepting a configuration alone cannot establish execution correctness.

**Independent Test**: Exercise all five model paths, including wider-then-narrower selections, interrupted execution, epoch boundaries and failed owner updates.

**Acceptance Scenarios**:

1. **Given** slicing after a wider update, **When** a narrow width is selected, **Then** full-tensor zero-gradient, residual-momentum and decay behavior remains unchanged; per-width history advances only for the selected width.
2. **Given** concat after a wider update, **When** a narrower width excludes a block, **Then** that block's gradient is absent and weights, moments and counters remain unchanged; C2 preserves lazy histories through resume.
3. **Given** C3, **When** a complete update executes, **Then** only the active quarter owners and common owner clip and step, each once with its own cap, and one global scheduler advance follows, with no second global clip.
4. **Given** a compatible interrupted run, **When** restored inside or around an epoch boundary, **Then** actions and batches match uninterrupted execution exactly and numerical state matches within established project tolerances.
5. **Given** a sampling-contract mismatch or failure during a C3 multi-owner update, **When** restoration or checkpoint publication is attempted, **Then** incompatible state is rejected before live-state mutation and partial updates cannot become resumable checkpoints.

---

### User Story 3 - Complete the authorized campaign with auditable costs (Priority: P1)

A researcher obtains five complete runs after CPU verification and short GPU diagnostics while respecting shared cluster limits and preserving all historical work.

**Why this priority**: The requested experiment requires completed, trustworthy results and controlled use of shared resources.

**Independent Test**: Check submission reconciliation and limit handling using queue fixtures; verify diagnostic evidence before production eligibility and reconcile a resumed run's progress and costs.

**Acceptance Scenarios**:

1. **Given** successful CPU checks, **When** short diagnostics run for all five arms, **Then** finite loss, actual width selection, save/restore, ownership accounting and measured steady-state throughput are recorded separately from production.
2. **Given** all validation gates pass, **When** production is submitted, **Then** all five runs are eventually queued within applicable limits, without duplicate jobs or another launch confirmation.
3. **Given** interruptions or a restarted submission helper, **When** execution continues, **Then** each run has one writer, resumes only its own valid state, and retains attempt-level costs including consumed work beyond the last durable checkpoint.
4. **Given** five finished runs, **When** completion is checked, **Then** each has exactly 348,528 complete updates, four designated epochs, a validated terminal checkpoint and ordinary-validation results at all four widths.

---

### User Story 4 - Compare terminal quality, exposure and resources (Priority: P2)

A researcher inspects the new results and their historical counterparts using reproducible tables and figures, with a clear distinction between sampling effects and differences among optimizer arms.

**Why this priority**: Saved results must answer the research question without repeating training.

**Independent Test**: Validate complete endpoint fixtures and reject incorrect identities, missing/duplicate points, nonterminal checkpoints and mixed evaluation roles; inspect figure structure and paired deltas.

**Acceptance Scenarios**:

1. **Given** five validated new terminals, **When** the new-campaign report is generated, **Then** it contains exactly 20 endpoints with frozen checkpoint and evaluation identities.
2. **Given** compatible historical references, **When** the complete comparison is generated, **Then** it contains exactly 44 endpoints: 20 fixed-IM, 20 original uncorrected uniform and four original standalone endpoints, clearly labeled with preserved provenance.
3. **Given** missing or incompatible historical references, **When** reporting runs, **Then** validated new-campaign results remain deliverable and the outstanding comparison is explicitly identified; no reference is invented or substituted.
4. **Given** valid ordinary-validation histories, **When** figures and deltas are exported, **Then** loss/perplexity endpoints and four full-range width-specific progress panels use consistent arm colors, distinct policy styles and the specified standalone markers, with all requested exposure/resource comparisons.
5. **Given** the single-seed comparison, **When** findings are interpreted, **Then** they distinguish within-arm sampling changes from representation/history/clipping changes and make no equal-compute or across-seed significance claims.

### Edge Cases

- A short random sample can omit widths or have unequal counts; neither balancing nor eager allocation of unused histories is allowed.
- Sampling inverse block membership is different from inverse width fraction or parameter count; those alternatives must fail campaign validation.
- The fixed 43-sequence excluded tail must not rotate into subsequent epochs or be silently changed.
- Active zero gradients differ from absent gradients. Inactive sliced coordinates can change through optimizer history or decay even without gradient support.
- Epoch transitions and checkpoint recovery must not reset scheduler, action/data randomness or histories.
- A failure after one C3 owner mutates state must leave the previous durable checkpoint intact.
- Resource records after a hard kill may be incomplete; identify missing measurements instead of inventing costs.
- Other campaigns consume the same user-wide job capacity; wait for capacity and reconcile jobs before submitting again.
- A terminal checkpoint with missing evaluation output requires completion recovery without another training update.
- Stale historical documentation cannot override actual saved controls, manifests or terminal evidence.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Create a separate feature and five-run campaign with fresh identities and directories. Preserve Feature 013 specifications, artifacts, strict uniform/correction validation and unrelated work. Reuse the existing training and reporting capabilities.
- **FR-002**: Implement the exact five-arm matrix below, with ordinary causal language-model loss and correction mode `none`. All widths within an arm share model weights. Start every arm from its normal fresh seed-42 initialization; no trained-baseline warm start or claim of identical cross-representation initial values.
- **FR-003**: Select ordered widths g250/g500/g750/g1000 with probabilities [3/25, 4/25, 6/25, 12/25], derived by normalizing reciprocals of incremental-block membership counts [4,3,2,1]. Use one independent categorical draw with replacement per complete update, applied globally across all layers (H=1).
- **FR-004**: All five runs MUST have identical isolated action-RNG initialization and selected-width sequences. Data randomness remains independent; batch sequences match each other and the historical deterministic data-order contract. New action sequences need not match historical uniform sequences. Do not enforce counts, balance cycles, hold widths or weight losses inversely by selection probability.
- **FR-005**: Bind ordered widths, probabilities, cadence, policy identity and action RNG state into saved controls, provenance, checkpoints and reporting. Reject mismatches before mutating live state. Historical identities and hashes remain valid under their original rules.
- **FR-006**: Preserve S1/S2 full-tensor slicing gradients, momentum and decay. Preserve C1/C2/C3 absent-gradient semantics for inactive blocks. S2/C2 keep separate width histories over shared parameters; only selected history advances. Preserve C2 lazy allocation and exact continuation.
- **FR-007**: C3 has disjoint O-A/O-B/O-C/O-D/O-common ownership covering each trainable parameter exactly once, including tied/common parameters. Clip and step only the active quarter prefix plus common, independently at L2 cap 1.0, with no second global clip. Every complete update advances the common scheduler once; partial owner updates cannot be published as resumable progress.
- **FR-008**: Preserve numerical model/history/scheduler state within existing tolerances and exact action/batch traces through interruptions and epoch boundaries. Recover only from the interrupted run's own compatible durable checkpoint; retain continuity of epochs, data cursor and randomness.
- **FR-009**: Preflight MUST re-audit immutable data/tokenizer identities, role separation, ordering, exclusions, model counts, budgets and full scientific controls, accepting exactly five arms and rejecting discrepancies. Check real model execution, especially C3, rather than configuration eligibility alone.
- **FR-010**: Before production, pass focused CPU checks and short bf16 GPU diagnostics for all five arms. Evidence must cover probability mapping and equal action traces, wider/narrower optimizer semantics, clipping/clock accounting, finite loss, save/restore, pre-mutation mismatch rejection, C3 partial-failure safety and measured steady-state throughput. Existing campaign tests must continue to pass.
- **FR-011**: Submit every GPU workload through sbatch, exclude `gpu-[05,50,51]`, and inspect live limits/queue state. Respect at most two running and four submitted GPU jobs across the user, including other campaigns, or stricter applicable limits. Submit the fifth run when capacity permits; use restart-safe records, one writer per run and duplicate prevention without restarting unrelated helpers.
- **FR-012**: Keep all experiment artifacts, snapshots, materialized configurations, diagnostics, job files/logs/records, checkpoints and reports under one fresh campaign root after checking conflicts. Repository source, tests, specifications and documentation stay in the repository. Record source revision and snapshot hashes, scientific identities, job IDs and continuation attempts.
- **FR-013**: Monitor progress, checkpoint health, recent throughput, utilization and estimated completion through terminal validation and reporting. Measure cumulative runtime, peak resources, optimizer-state bytes, checkpoint sizes and attempted versus committed work, including replay after interruptions. Flag unreconciled resource evidence.
- **FR-014**: Record actual width selections and block activations separately and reconcile them with committed updates and owner calls. Distinguish gradient-support exposure from actual optimizer updates, especially for slicing. Common parameters participate in every update.
- **FR-015**: Evaluate all four widths at each full-budget terminal using ordinary-validation membership, target-token-weighted causal loss and perplexity equal to its exponential. Freeze terminal checkpoint and evaluation identities before comparison. Do not substitute best checkpoints, trailing means or early endpoints; keep the final holdout sealed and controller data out of training/sampling decisions.
- **FR-016**: Validate and export the 20-endpoint new-campaign table and, when compatible references are available, the 44-endpoint combined table as CSV/JSON. Validate all required runs, budgets, policy labels, counts, finite values, checkpoint/evaluation provenance and unique endpoints. Report missing/incompatible references explicitly without fabrication, retraining or silent substitution.
- **FR-017**: Export loss and perplexity versus exact active non-embedding counts as PNG/PDF for both a five-curve new-campaign view and a uniform-versus-IM comparison. Use consistent arm colors and distinct policy line styles; historical standalones are disconnected brown triangles under one `Standalone` legend entry.
- **FR-018**: Export a four-panel ordinary-validation loss-progress figure, one panel per width across the full recorded update range, comparing each arm's fixed-IM and uniform histories, as PNG/PDF. Export per-arm/per-width fixed-IM minus uniform loss/perplexity deltas and exposure, clipping, optimizer-state bytes, peak-memory, wall-time, throughput and checkpoint-size comparisons.
- **FR-019**: Deliver specification, plan/contracts, tasks, consistency analysis, implementation, runbook and CPU/GPU evidence alongside run/results artifacts. Carry the user's sbatch authorization through all stages: diagnostics follow CPU validation and production follows all diagnostic gates, without a separate launch confirmation.

### Research & Experiment Requirements *(include for experiment-facing changes)*

- **EX-001**: Use exactly the following fresh seed-42, four-epoch arms:

| Arm | FFN representation | AdamW history ownership | L2 clipping |
| --- | --- | --- | --- |
| S1-IM | Slicing | Shared | Global 1.0 |
| S2-IM | Slicing | Per width (`per_granularity`) | Global 1.0 |
| C1-IM | Concat | Shared | Global 1.0 |
| C2-IM | Concat | Per width (`per_granularity`) | Global 1.0 |
| C3-IM | Concat | Per FFN quarter plus common (`per_ffn_block`) | Independently 1.0 per active owner |

- **EX-002**: Inherit and verify model dimension 64, four layers, four attention heads, context 128, vocabulary 2,048, full FFN dimension 256 and initializer standard deviation 0.02. Width fractions .25/.50/.75/1.00 give FFN dimensions 64/128/192/256 and exact active non-embedding counts 115,264/164,416/213,568/262,720, excluding input embeddings and LM head.
- **EX-003**: Inherit AdamW LR .008, betas (.9,.95), epsilon 1e-8, weight decay .1; original cosine schedule with 64 LR warmup updates over the full production horizon; batch 64, accumulation 1, bf16, one process/GPU, no LR scaling, no membership correction or pre-nested width warmup.
- **EX-004**: Use the audited four-role corpus, tokenizer, stored permutation and deterministic epoch ordering. Each epoch visits the same 5,576,448 designated packed sequences with the fixed 43-sequence excluded tail. Each update accounts for 8,192 packed training tokens. Every run consumes 348,528 updates and 2,855,141,376 tokens over four epochs; five runs consume 1,742,640 updates and 14,275,706,880 tokens. Validate every 64 updates and at completion; training tokens and evaluated causal-target tokens are distinct counts.
- **EX-005**: Report expected block activation probabilities A/B/C/D of 1.00/.88/.72/.48 under fixed IM and 1.00/.75/.50/.25 under uniform, alongside actual selections and activations. These are expectations, not guaranteed counts or equal parameter/example exposure.
- **EX-006**: Reuse only the validated original five uncorrected uniform elastic terminals and four original standalone terminals as historical references. Elastic references have four-epoch budgets; standalones retain their original one-epoch budgets. No reference retraining belongs to this campaign.
- **EX-007**: Interpret paired policy changes within each arm separately from representation, history and clipping differences between arms. Equal training-token budgets do not imply equal FLOPs or runtime because IM chooses wider networks more often. Keep seed-42 findings descriptive without across-seed significance claims or error bars; improved loss is not an acceptance requirement.

### Key Entities *(include if feature involves data)*

- **Campaign and run**: Five fresh identities, arm matrix, scientific controls, source evidence, budget and lifecycle from validated configuration through terminal reporting.
- **Sampling contract**: Ordered widths, incremental memberships, normalized probabilities, global replacement cadence, policy identity and reproducible action state.
- **Data contract**: Corpus/tokenizer/role identities, designated epoch membership, fixed exclusion, order and current cursor.
- **Optimizer owner**: Shared, width or quarter/common history with covered parameters, activation count, clipping policy and common learning-rate clock.
- **Checkpoint and attempt**: Complete durable training state and provenance, distinct from execution costs that may extend past durable progress.
- **Endpoint and comparison**: Terminal run/width result with count, budgets, evaluation/checkpoint identity, policy label, metrics and historical-reference status.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All five intended configurations pass preflight; every tested deviation in matrix, probability, data, budget or scientific controls is rejected with an identified reason.
- **SC-002**: All five arms pass CPU and short GPU acceptance checks before production; resumed action and batch traces match uninterrupted traces 100%, with numerical state within project tolerances and no publishable partial owner update.
- **SC-003**: Five runs complete exactly four designated epochs each, totaling 1,742,640 updates and 14,275,706,880 training tokens; their action and batch streams match across arms without forced frequency balancing.
- **SC-004**: Every completed run supplies traceable controls, scalar histories, terminal checkpoint, all four terminal evaluations, exposure/clipping summaries and measured attempt/resource evidence; no GPU submission violates the applicable user-wide ceiling or duplicates a run writer.
- **SC-005**: Saved artifacts support exactly 20 validated new endpoints and, with available compatible references, exactly 44 validated combined endpoints. Every plotted endpoint traces to its budget, count and frozen evaluation/checkpoint identity; unavailable comparison inputs remain explicitly outstanding.
- **SC-006**: The deliverables include two new-campaign endpoint figures, two comparative endpoint figures and one full-range four-panel progress figure, each in PNG/PDF, plus 20 paired loss/perplexity delta records when all references validate.
- **SC-007**: The report answers the within-arm sampling question and documents cross-arm ownership/clipping differences using measured exposure and resource evidence, without claiming equal compute, across-seed significance or holdout results.

## Assumptions

- The intended users are experiment researchers; exact scientific terms and numerical controls are requirements. Code structure and interface design belong in the plan/contracts.
- The supplied notes and current authorization govern this new campaign. Historical “separate launch request” statements do not revoke this authorization. No unresolved scientific choice requires clarification.
- Proposed artifact root: `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/campaigns/inverse-membership-sampling-v1`; use a fresh alternative if occupied, preserving existing content.
- Prepared inputs and historical artifacts are expected to remain readable; actual evidence must establish compatibility before production/comparison. Missing references do not justify inventing results or calling the full comparison complete.
- Reuse the existing fixed-global sampler and established numerical tolerances. Short diagnostics may use reduced work in separate identities/directories while retaining the production schedule horizon; production budgets remain fixed.
- Excluded: GMC/LMC combinations, additional seeds, adaptive policies, hold-interval sweeps, forced balancing, inverse-probability loss weighting, baseline retraining, warm starts and sealed-holdout evaluation.
