# Feature Specification: TinyStories Linear S1/S2 CaLR Comparison

**Feature Branch**: `017-tinystories-linear-calr`  
**Created**: 2026-10-01  
**Status**: Draft — validated for planning  
**Input**: User description: “Use notes/tinystories_calr_speckit_prompt_2026-10-01.md as the feature description.”

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Prepare a scientifically matched comparison (Priority: P1)

As an experiment researcher, I want exactly four linear-width runs that vary optimizer-history ownership and decay-exponent policy independently, so I can distinguish the CaLR effect from changing the decay family or separating optimizer histories.

**Why this priority**: Unmatched controls would prevent interpreting the research question.

**Independent Test**: Resolve the four experiment definitions and audit their controls, initialization, deterministic action/data streams, complexity counts, and analytic schedules without training.

**Acceptance Scenarios**:

1. **Given** the original S1/S2 controls, **When** preparing the campaign, **Then** exactly the four declared arms share peak LR 0.008, warmup 64, linear widths, seed, initialization and action/data streams, with only their declared experimental factors differing.
2. **Given** the active model at each width, **When** validating complexity and schedules, **Then** counts match the table below, all warmups agree, uniform exponents equal one, and CaLR exponents follow the declared formula.
3. **Given** a changed corpus, clipping rule, sampling probability, or undeclared arm, **When** auditing definitions, **Then** preparation fails with the mismatched control identified.

---

### User Story 2 - Execute and continue faithful training (Priority: P1)

As an experiment operator, I want each selected width to receive its declared LR while preserving S1/S2 update behavior and reliable continuation, so completed evidence represents the intended experiment.

**Why this priority**: A correct schedule definition is insufficient if actual updates or restored histories differ.

**Independent Test**: Run focused mixed-width update and continuation diagnostics for each arm, including warmup, decay, epoch and terminal boundaries; validate real-shape GPU readiness separately before production.

**Acceptance Scenarios**:

1. **Given** a selected width and global position, **When** a successful update commits, **Then** every parameter group of the selected optimizer uses that width's scheduled LR and the global clock advances once; S2 advances only the selected history.
2. **Given** an own-arm checkpoint, **When** continuing around warmup, decay or epoch boundaries, **Then** weights, histories, counters, applied LRs, actions, data cursor and accounting match uninterrupted execution within inherited numerical tolerances.
3. **Given** an incompatible checkpoint, **When** attempting restore, **Then** rejection occurs before live state changes; a failure after optimizer mutation cannot publish partial state as successful or resumable.
4. **Given** a full-budget durable checkpoint with missing terminal outputs, **When** recovering, **Then** outputs are recreated with zero additional training updates.
5. **Given** missing execution authorization or failed/stale readiness evidence, **When** considering production submission, **Then** no GPU job is submitted.

---

### User Story 3 - Interpret all deployed widths against validated references (Priority: P2)

As a researcher, I want terminal comparisons and measured trajectories for all four widths, so I can assess full-width improvement, smaller-width costs, and interaction with optimizer history.

**Why this priority**: Absolute quality and matched comparisons determine usefulness; loss ordering alone does not.

**Independent Test**: Use provenance-complete reporting fixtures to verify endpoint counts, comparison arithmetic, baseline separation, plot labels and incomplete-evidence handling without production training.

**Acceptance Scenarios**:

1. **Given** four valid new terminals and eight validated saved reference runs, **When** reporting, **Then** the report contains 36 endpoint rows, all required paired comparisons and four interaction contrasts.
2. **Given** cosine references at two peak LRs, **When** plotting comparisons, **Then** primary and supplemental panels keep them separate and each panel shows the seven specified series with explicit budgets and LR settings.
3. **Given** an intermediate checkpoint, absent reference provenance or missing measured trajectory, **When** generating results, **Then** it is not substituted for terminal or observed evidence and the affected completion status remains incomplete.
4. **Given** the completed results, **When** interpreting them, **Then** findings state the CaLR effect at every width and optimizer scope, the g1000 standalone gap, smaller-width costs and the limitations of one seed.

### Edge Cases

- Positions 0, 63, 64, 65, T−1 and T distinguish applied-update rates from stored terminal rates; no update is permitted at T.
- Invalid horizons, positions, nonpositive complexity counts, equal minimum/maximum complexity, or mismatched count definitions must fail validation rather than silently change the experiment.
- A mixed-width sequence must not reuse the preceding width's LR or accumulate multipliers; zero LR on update 1 still follows the inherited successful-update convention.
- An S2 history not yet selected remains unadvanced; zero gradients outside a sliced prefix do not imply frozen weights.
- Cross-arm, wrong-grid, changed-exponent, wrong-history-scope and malformed checkpoints fail before mutation.
- Failure during optimizer mutation, scheduler advancement or accounting preserves the last valid durable checkpoint and records the failed attempt.
- Occupied campaign roots, duplicate submissions, uncertain submission outcomes and stricter live queue limits prevent conflicting admission.
- Missing saved references leave valid new evidence intact but comparison acceptance incomplete; reconstructed LR curves never replace measured applied rates.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The campaign MUST contain exactly `S1-linear-poly`, `S1-linear-CaLR`, `S2-linear-poly`, and `S2-linear-CaLR`. All use g250/g500/g750/g1000 with FFN prefixes 64/128/192/256 in every layer. Geometric runs, new standalones, additional seeds, LR sweeps, correction runs and restarting stopped geometric runs are outside scope.
- **FR-002**: Within each optimizer scope, only exponent policy and consequential metadata MUST differ; within each schedule policy, only optimizer-history ownership and consequential metadata MUST differ. All arms MUST share fresh normal initialization and identical deterministic action/data streams. They MUST NOT warm-start from reference weights.
- **FR-003**: The global schedule MUST follow the exact definition below, with one-based update k applying position p=k−1 and exactly one advance after each successful committed update. S2 width-local optimizer counters MUST NOT serve as LR clocks.
- **FR-004**: CaLR complexity MUST count active trainable scalar parameters including embeddings/output head, counting tied identities once and only active FFN entries. Counts MUST be validated against the actual model; width fractions, FFN-only counts, non-embedding counts and membership frequency MUST NOT replace this definition.
- **FR-005**: Each update MUST sample one global width uniformly and independently with replacement, use its entire active slicing prefix, apply one joint global L2 clipping cap of 1.0 and perform one AdamW step. S1 MUST use one shared model/history; S2 MUST use four distinct moment/counter histories referencing the same model parameter objects and step only the selected history, including common parameters.
- **FR-006**: The selected width's LR MUST apply to every group of the stepped optimizer, including common parameters and decoupled weight decay. Width-conditioned rates MUST have an explicit validated contract compatible with synchronization and checkpoint checks; no stale rate may leak into the next update. Existing full-shaped zero-gradient tails and their AdamW decay/momentum behavior MUST be preserved. No GMC/LMC, gradient rescaling, post-step delta scaling, detached earlier blocks, tail masks or C4 ownership policy may be added.
- **FR-007**: All nonschedule controls MUST match the saved original S1/S2 counterparts at LR 0.008 / warmup 64: seed 42; initializer range 0.02; hidden size 64, four layers/heads, full FFN 256, vocabulary 2048, context 128; batch 64, accumulation 1, one process/GPU, BF16, no activation checkpointing; AdamW betas (0.9, 0.95), epsilon 1e−8, decay 0.1; probability 0.25 per width and sampling interval 1; correction none, no pre-nested warmup or automatic LR scaling. Corpus/tokenizer/manifests, initialization/random-stream policy, batch order, ordinary-validation cadence and aggregation MUST be inherited. Controller data and sealed final holdout MUST remain unused.
- **FR-008**: Each arm MUST train four epochs of 87132 updates, totaling 348528 updates and 2855141376 packed tokens at 8192 tokens/update. Warmup MUST be inside this budget. Assigned campaign totals MUST reconcile to 1394112 updates and 11420565504 tokens; validation tokens and failed-attempt costs MUST be accounted separately.
- **FR-009**: New-versus-original-counterpart differences MUST use a closed allowlist of schedule definition/consequential resolved fields and fresh identity/output/provenance fields. Schedule pairs and ownership pairs MUST be audited separately. Historical configurations, signatures, source snapshots, checkpoints and restore semantics, including S1/S2/C4, MUST remain valid and unchanged.
- **FR-010**: Preparation MUST use a fresh campaign/run identity and an available fresh output root, prevent historical overwrites and duplicate submissions, and distinguish readiness, submission, execution and terminal-completion states. Existing training, evaluation, accounting and reporting workflow MUST be reused without a second training framework or generic NAS system.
- **FR-011**: Continuation MUST restore weights, all appropriate histories/counters, schedule definition/position, width counts, action/data RNGs, sampler cursor and accounting. Incompatible ownership, grids, complexity definitions, exponent policies, warmup/horizons or cross-arm state MUST be rejected before live mutation. Post-mutation failures MUST NOT publish successful-update evidence or resumable partial state. Terminal recovery MUST require no extra training.
- **FR-012**: Production MUST require explicit later execution authorization and passing source/config-bound focused correctness and real-shape CUDA BF16 readiness evidence for all four arms at batch 64/context 128. Submission MUST obey the existing partition/QoS, user-wide limits of at most two running GPU jobs and four submitted jobs (or stricter live limits), and exclusions `gpu-[05,50,51,54]`. Each continuation MUST use only its own durable checkpoint.
- **FR-013**: Saved references MUST be inspected read-only and accepted only after validating actual identities/configs, terminal budgets/checkpoints, ordinary-validation role/manifests/aggregation and active parameter counts. Best or intermediate checkpoints and aggregate tables alone MUST NOT establish terminal provenance. No reference may be retrained or rewritten.
- **FR-014**: Terminal acceptance MUST require all four new full-budget checkpoints and ordinary-validation loss/perplexity at all four widths. Failed or partial arms MUST remain incomplete.
- **FR-015**: Reporting MUST publish 36 endpoint rows: 16 new, 16 saved S1/S2 cosine and four standalone. Each row MUST carry loss, perplexity, grid/width, parameter counts, schedule/exponent metadata, peak LR, budgets and provenance.
- **FR-016**: Reports MUST provide eight CaLR-minus-polynomial pairs, eight S2-minus-S1 pairs, 16 new-minus-matching-scope-cosine pairs at LR 0.008 and 16 new-minus-standalone pairs, with loss differences and perplexity ratios/relative gaps. Four loss interactions MUST equal `(S2_CaLR − S2_poly) − (S1_CaLR − S1_poly)`, one per width. Additional comparisons MUST be labeled distinctly.
- **FR-017**: Reports MUST publish PNG/PDF terminal loss/perplexity versus active non-embedding parameters, recorded validation-loss progress and actually applied LR versus global updates. Primary panels MUST use original cosine S1/S2 at 0.008; supplemental companion panels MUST use cosine S1/S2 at 0.004, retaining new arms and standalones at 0.008. Captions/titles MUST state peak LRs, warmup, seed and budgets; colors/markers MUST remain consistent and readable.
- **FR-018**: Each loss/perplexity panel MUST contain exactly the legend labels `standalone-cosine`, `S1-cosine`, `S2-cosine`, `S1-polynomial`, `S2-polynomial`, `S1-carl`, `S2-carl`, one series per label. Elastic methods MUST be separate width curves; standalones MUST be unconnected scatter points. Progress plots MUST preserve labels and baseline separation, use recorded elastic trajectories, and show standalone terminal points only at update 87132. LR plots MUST use available elastic applied-rate evidence only. The plotting alias `carl` MUST retain technical name CaLR in definitions, tables and provenance.
- **FR-019**: Interpretation MUST report CaLR against matched polynomial and matching-scope cosine at every width, S1 versus S2 for each schedule and their interaction, including g1000 standalone gaps and smaller-width costs. Tuned S1-cosine at 0.004 and S2-cosine at 0.008 MUST remain visible practical references. Findings MUST be descriptive seed-42 results, without guaranteed improvement/parity, statistical robustness, proof of fixed gradient interference or exact transfer of the paper's mechanism.

### Research & Experiment Requirements

- **EX-001**: Resolved experiment records and checkpoint identities MUST version and preserve factors, ownership, complexity definition/counts, exponent bounds/values, warmup/horizon, controls, seed and immutable source/config provenance.
- **EX-002**: Every committed update MUST record actually applied pre-optimizer LR, width, complexity, exponent, optimizer owner and global position. Scientific traces MUST reconcile with action/data evidence, width exposure, budgets and durable checkpoint boundaries.
- **EX-003**: Every run MUST retain structured scalar metrics, recorded validation trajectories, source/config/checkpoint hashes, throughput, optimizer-state memory, CUDA peak memory and attempt-cost evidence. Analytic schedule audits MUST be labeled separately from measured execution.
- **EX-004**: Focused verification MUST cover all widths at position 0, warmup boundaries, representative decay and T−1/T; finite nonnegative rates; actual all-group AdamW steps against manual references with nonzero moments; mixed-width rate/history behavior; continuation around warmup, decay and epoch boundaries; exact counterpart action/data streams; fresh initialization; and historical S1/S2/C4 compatibility.
- **EX-005**: Verification MUST record commands, outcomes, limitations and source/config bindings, preserve failed readiness records, and require corrected passing identities for production. CPU-only or shortened-horizon diagnostics MUST NOT establish production GPU execution or full-budget completion.
- **EX-006**: Saved cosine references MUST be labeled as external comparisons. The 0.008 originals are primary matched-peak references; the 0.004 runs are supplemental and do not isolate either experimental factor. Reports MUST explain different standalone/elastic horizons, aggregate versus selected-width exposure, and that CaLR also changes cumulative LR and AdamW decay.

### Declared Experiment Definitions

These equations and fixed controls define the scientific treatment, not a choice of software implementation.

Let T=348528, W=64 and eta_peak=0.008. The global position starts at zero:

```text
eta_w(p) = eta_peak * p / W                         for 0 <= p < W
u(p)     = (p - W) / (T - W)                       for W <= p <= T
eta_w(p) = eta_peak * (1 - u(p)) ** gamma_w         for W <= p <= T
```

Uniform-polynomial arms use gamma_w=1 for every width. CaLR uses fixed gamma_min=0.5 and gamma_max=2.0:

```text
gamma_w = gamma_max - (gamma_max - gamma_min)
          * log(C_w / C_min) / log(C_max / C_min)
```

| Width | Active FFN dimension | Complexity C_w (including embeddings/head) | Active non-embedding parameters for reporting |
| --- | ---: | ---: | ---: |
| g250 | 64 | 377408 | 115264 |
| g500 | 128 | 426560 | 164416 |
| g750 | 192 | 475712 | 213568 |
| g1000 | 256 | 524864 | 262720 |

Thus CaLR endpoint exponents are 2 and 0.5, with intermediate values derived from the formula. Update 1 applies zero; update 64 uses position 63; update 65 applies peak at position 64. Last update uses T−1; stored terminal position T has zero LR and no associated training update. At decay progress 0.5, uniform LR is 0.004, CaLR g250 is 0.002 and CaLR g1000 is approximately 0.005656854249. Invalid positions/horizons must be rejected; any defensive clamp must not extend training. Bounds must not be tuned using validation during this feature.

### Reference Selection

All paths below are read-only discovery locations; actual saved identity and terminal provenance must be verified before comparison:

| Reference | Run location under `/nfs-stor/ivo.navarrete/results/elasticnn/` | Peak LR / warmup |
| --- | --- | --- |
| Original S1 cosine | `optimizer-ownership-v1/runs/S1/` | 0.008 / 64 |
| Original S2 cosine | `optimizer-ownership-v1/runs/S2/` | 0.008 / 64 |
| Tuned S1 cosine | `optimizer-ownership-s1-peak-lr-v1/runs/S1-linear-lr0004/` | 0.004 / 64 |
| Supplemental S2 cosine | `optimizer-ownership-s2-warmup-peak-lr-v1/runs/S2-linear-lr0004/` | 0.004 / 64 |
| Four matching standalones | `optimizer-ownership-v1/runs/ST-{g250,g500,g750,g1000}/` | 0.008 / 64 |

Each standalone has 87132 updates; each elastic reference has 348528. Equal peak LR does not imply equal schedules or parameter exposure.

### Key Entities

- **Campaign**: Four-arm matched comparison, fresh identity/root, assigned budget and independent lifecycle statuses.
- **Arm/run**: Ownership and exponent policy, fixed controls, width grid, resolved identity and deterministic streams.
- **Width complexity**: Active parameter count, reporting count and derived exponent for each subnet.
- **Committed update**: Selected width/owner, applied LR/global position, data/action evidence, exposure and tokens.
- **Durable checkpoint**: Own-run model/history state, schedule contract, randomness, cursor and reconciled accounting.
- **Reference terminal**: Read-only saved result with validated identity, budget, evaluation provenance and hashes.
- **Comparison artifact**: Endpoints, paired differences, interactions, measured trajectories, figures and interpretation.
- **Readiness/attempt record**: Tested identities, diagnostic or production status, failure evidence and resource costs.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Researchers can validate exactly four arms with zero undeclared scientific differences, all four complexity counts correct, and identical initial weights/action/data streams across the matched design and applicable counterparts.
- **SC-002**: All four arms pass focused schedule/update checks at every declared boundary and mixed-width checks, with zero rate/history leakage or global-clock disagreement and continuation matching within inherited numerical tolerances.
- **SC-003**: Before production, all four arms have passing real-shape readiness evidence bound to their tested identities; no unauthorized, duplicate or queue-limit-violating GPU submission occurs.
- **SC-004**: All four arms complete exactly 348528 updates and 2855141376 packed tokens each, with four valid terminal checkpoints and 16 ordinary-validation endpoints; totals equal 1394112 updates and 11420565504 tokens.
- **SC-005**: Researchers receive exactly 36 provenance-validated endpoint rows, 48 required paired comparisons and four width-specific interaction contrasts, all arithmetically consistent with endpoint losses/perplexities.
- **SC-006**: All requested figure families are available in both PNG and PDF; primary/supplemental loss and perplexity panels each contain the seven exact labels, while progress/LR displays contain no invented trajectories or merged baseline settings.
- **SC-007**: A researcher can determine the direction and magnitude of CaLR's effect at all four widths under both ownership scopes, assess the ownership interaction and full-width standalone gap, and identify any smaller-width cost using published evidence alone.
- **SC-008**: Every completed arm includes reconciled update/token/action/data, applied-LR, provenance and resource evidence; historical references remain unmodified and all rejected/failed attempts remain distinguishable from completion.

## Assumptions

- The requesting user is the experiment researcher/operator; this invocation creates a specification and branch only. Planning, tasks, implementation, GPU diagnostics/production, commits and pushes require subsequent instructions as applicable.
- The supplied feature prompt takes precedence over the earlier note's optional two-arm pilot suggestion: exactly four arms are required.
- Existing Feature 013/016 contracts and verification records supply baseline behavior and numerical tolerances; earlier verification results are context, not readiness evidence for this new treatment.
- Saved references and inherited corpus/tokenizer artifacts are dependencies whose identities and availability must be verified during planning/preparation. This specification does not claim a fresh terminal audit.
- Proposed fresh campaign root is `/nfs-stor/ivo.navarrete/results/elasticnn/tinystories-linear-s1-s2-calr-v1`; availability must be checked before reservation. No external root is reserved by this invocation.
- The CaLR bounds are a fixed mild ElasticNN setting, not optimized hyperparameters. S2 separates both AdamW moments/counters and is an adaptation rather than an exact reproduction of the motivating paper's SGD mechanism.
- Successful feature delivery means correct complete evidence and interpretable results; improvement over polynomial, cosine or standalone is a hypothesis rather than an acceptance prerequisite.
- Runtime integration choices, detailed schemas and exact operational commands are deferred to implementation planning after inspection of actual runtime code and saved resolved configurations.
