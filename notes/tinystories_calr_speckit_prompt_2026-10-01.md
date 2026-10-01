# Spec Kit prompt: TinyStories linear S1/S2 CaLR comparison

Date: 2026-10-01.

Suggested invocation:

```text
/speckit-specify Use notes/tinystories_calr_speckit_prompt_2026-10-01.md as the feature description. Extend the existing TinyStories workflow with exactly four linear experiments comparing S1/S2 optimizer histories and uniform-polynomial/CaLR schedules in a matched 2×2 design.
```

## Feature request and scientific question

Extend the existing ElasticNN TinyStories optimizer-ownership workflow to test whether conditioning LR decay on subnet complexity improves the quality of a shared-weight model, especially its full-width gap to standalone training.

Read `AGENTS.md`, `specs/016-tinystories-s1-warmup/plan.md`, the relevant Feature 013/016 contracts and verification records, and `notes/subnet_aware_calr_elasticnn_2026-10-01.md`. The last note explains the paper, current S1/S2 implementation, completed results, and the proposed adaptation. Inspect saved reference configurations and actual runtime code before defining implementation details.

Source paper: Jeon et al., [Subnet-Aware Dynamic Supernet Training for Neural Architecture Search, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Jeon_Subnet-Aware_Dynamic_Supernet_Training_for_Neural_Architecture_Search_CVPR_2025_paper.pdf), with [readable preprint and supplement](https://arxiv.org/html/2503.10740v1). The paper's CaLR uses a polynomial schedule with an exponent derived from log parameter count. Its separate momentum mechanism motivates S2. This feature evaluates both factors and their interaction; S2 separates both AdamW moments and counters, so the combined arm is an ElasticNN adaptation rather than an exact reproduction of the paper's SGD method.

Fix peak LR 0.008 and warmup 64 for all four new arms, matching the original S1/S2 cosine baselines and matching standalone models. This avoids introducing a peak-LR change into the main scheduler/optimizer comparison. Completed linear S1 results at LR 0.004 were better than those at 0.008, but that tuned configuration is a supplemental practical reference, not the scientific counterpart for this feature. Do not assume separated histories are necessary or sufficient for CaLR to help.

## Exactly four new arms

| Arm | Representation / optimizer state | Peak LR | Warmup | Decay |
| --- | --- | ---: | ---: | --- |
| S1-linear-poly | Slicing / shared AdamW | 0.008 | 64 | Uniform polynomial, exponent 1 at every width |
| S1-linear-CaLR | Slicing / shared AdamW | 0.008 | 64 | Width-dependent polynomial, exponents from active parameter counts |
| S2-linear-poly | Slicing / per-width AdamW | 0.008 | 64 | Uniform polynomial, exponent 1 at every width |
| S2-linear-CaLR | Slicing / per-width AdamW | 0.008 | 64 | Width-dependent polynomial, exponents from active parameter counts |

All four arms use the linear grid `g250/g500/g750/g1000`, with active FFN dimensions `64/128/192/256` in every layer. Within each optimizer-state scope, only the exponent policy differs scientifically. Within each schedule policy, only the optimizer-state scope differs scientifically. All four share fresh initialization and the same deterministic action and data streams.

The uniform polynomial controls distinguish the effect of width dependence from the change from the existing cosine schedule to polynomial decay. Saved cosine runs are external references, not substitutes for these matched controls. Do not add geometric arms, new standalone training, additional seeds, LR sweeps, or correction runs to this feature. Do not restart previously stopped geometric runs.

## Exact schedule semantics

Use one global scheduler position `p`, initialized to zero. One-based training update `k` applies the LR calculated at `p = k - 1`. Advance the scheduler exactly once after a successful optimizer update. Stored terminal position is `T`; there is no training update at that position.

Fix `T = 348528`, `W = 64`, and `eta_peak = 0.008`. Warmup is included in the total budget, and is independent of width:

```text
eta_w(p) = eta_peak * p / W                         for 0 <= p < W
u(p)     = (p - W) / (T - W)                       for W <= p <= T
eta_w(p) = eta_peak * (1 - u(p)) ** gamma_w         for W <= p <= T
```

Preserve the existing scheduler convention: update 1 applies LR zero, update 64 applies the rate at position 63, and update 65 applies peak LR at position 64. All widths have zero stored LR at terminal position `T`; the last actual update uses position `T - 1`. Do not change this convention to make the first update nonzero. Reject invalid horizons/positions or clamp only defensively outside the defined domain, without silently extending training.

For both `*-linear-poly` arms, set every `gamma_w = 1`.

For both `*-linear-CaLR` arms, fix `gamma_min = 0.5`, `gamma_max = 2.0`, and compute:

```text
gamma_w = gamma_max - (gamma_max - gamma_min)
          * log(C_w / C_min) / log(C_max / C_min)
```

These bounds are a declared mild ElasticNN comparison setting, not a claim of optimal hyperparameters or an exact reproduction of the paper's experimental choices. Do not tune them from validation during this feature.

Define `C_w` as active trainable scalar parameters in the selected subnet, **including embeddings and output head**, with tied parameter identities counted once. Count the active FFN prefix, not the full allocated slicing tensors. Validate the following counts from the actual model:

| Width | Active trainable parameters, `C_w` | Active non-embedding parameters for reporting |
| --- | ---: | ---: |
| g250 | 377408 | 115264 |
| g500 | 426560 | 164416 |
| g750 | 475712 | 213568 |
| g1000 | 524864 | 262720 |

Thus `gamma_g250 = 2`, `gamma_g1000 = 0.5`, and intermediate exponents come from the formula. Do not substitute width fractions, FFN-only counts, non-embedding counts, or membership frequencies. Store the complexity definition, counts, bounds, exponents, warmup, and horizon in versioned resolved configuration and checkpoint identities.

At normalized decay progress 0.5, the uniform-polynomial LR is 0.004 at every width; CaLR gives g250 LR 0.002 and g1000 LR approximately 0.005656854249. These are analytic schedule checks, not observations of completed training.

## Update rule and unchanged training controls

Preserve the existing implementations:

- S1: one shared model and one shared AdamW history.
- S2: one shared model and four AdamW optimizers referencing exactly the same parameter objects; distinct first/second moments and counters per width, including common parameters. Step only the selected width's optimizer; other histories do not advance or decay on that update. Do not create separate model copies.

Both use one global width sampled uniformly with replacement per update; slicing forward/backward through its entire active prefix; one joint global L2 clipping cap of 1.0; one AdamW step. S2's selected-width AdamW counters are separate from the global LR clock. Do not use per-width update counts as scheduler positions.

Apply the selected width's scheduled LR uniformly to **every optimizer parameter group**, including common parameters. This is a width-conditioned schedule, not selected-block-only stepping or a membership LR multiplier. No GMC, LMC, gradient rescaling, post-step delta scaling, detached earlier blocks, or C4 ownership policy.

Preserve slicing inactivity behavior: full-shaped FFN gradients contain zero entries outside the active prefix. Existing AdamW weight decay and, where populated in the stepped history, moments can still change inactive entries. Do not introduce tail masks or change that behavior in the scheduler experiment.

Use saved resolved original S1 and S2 configurations at peak LR 0.008 / warmup 64 under `optimizer-ownership-v1/runs/{S1,S2}/` as the respective scientific counterparts. Confirm their nonschedule/nonownership controls match. Inherit corpus/tokenizer/manifests, model initialization and dedicated random streams, validation cadence and aggregation, batch order, and controls:

- Seed 42; fresh normal initialization, initializer range 0.02; no warm-start from reference weights.
- Hidden size 64, four layers, four heads, full FFN 256, vocab 2048, context 128.
- Batch 64, accumulation 1, one process/GPU, BF16, no activation checkpointing.
- Four epochs, 87132 updates per epoch, 348528 total updates and 2855141376 packed training tokens per run; 8192 tokens per update.
- AdamW betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`.
- Uniform independent width draws with replacement, probability 0.25 per width, global sampling interval 1. No coverage balancing or changes to selection probabilities.
- Correction mode none, no pre-nested warmup, no automatic LR scaling.
- Preserve ordinary validation and the distinct controller/final roles. Do not consume reserved controller data or open the sealed final holdout.

Audit new-versus-counterpart config differences with an explicit allowlist: schedule type/definition and consequential resolved fields, plus new identity/output/provenance metadata. Audit schedule pairs and ownership pairs separately, allowing only their declared factor and consequential metadata. Preserve historical configurations, signatures, source snapshots, checkpoints, and restore behavior.

## Reuse, identities, and continuation

Reuse the existing trainer, configuration resolution, scheduler/update path, checkpointing, accounting, evaluation, and report conventions. Do not create a second training framework or a generic NAS system.

Use fresh campaign/run identities and a fresh root; suggested root:

```text
/nfs-stor/ivo.navarrete/results/elasticnn/tinystories-linear-s1-s2-calr-v1
```

Check that the root is available before reservation. Preparation/resume must not overwrite historical references or submit duplicates. Preserve immutable source/config provenance and distinct readiness, submission, execution, and terminal-completion statuses.

The global clock must retain a single position while computing the selected width's effective LR. Log the actually applied pre-optimizer LR, selected width, complexity, exponent, optimizer owner, and global position. Current S2 enforces synchronized base rates across width optimizers: explicitly integrate width-conditioned effective rates with those checks rather than disabling validation wholesale. Any temporary optimizer LR must be restored before synchronization/checkpointing and must not persist incorrectly into the next update. Alternatively introduce a versioned per-width rate contract with explicit validation. The implementation plan should choose a small integration with the existing scheduler/update path.

Resume must restore model weights, shared or all four width-specific AdamW histories/counters as appropriate, schedule definition and position, width-selection counts, action/data RNGs, sampler cursor, and accounting. Reject incompatible optimizer-state scopes, width grids, complexity definitions, exponent policies, warmup/horizons, or cross-arm checkpoints before live mutation. Failures after an optimizer mutation must not publish an apparently complete successful update or resumable partially mutated state. Full-budget recovery must recreate missing terminal outputs without additional training.

## Read-only references

Select terminal ordinary-validation results, not best checkpoints:

1. Tuned cosine S1, peak LR 0.004 / warmup 64:
   `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-s1-peak-lr-v1/runs/S1-linear-lr0004/`.
2. Cosine S2, peak LR 0.004 / warmup 64:
   `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-s2-warmup-peak-lr-v1/runs/S2-linear-lr0004/`.
3. Original cosine S1, peak LR 0.008 / warmup 64:
   `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/runs/S1/`.
4. Original cosine S2, peak LR 0.008 / warmup 64:
   `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/runs/S2/`.
5. Matching standalone terminals:
   `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/runs/ST-{g250,g500,g750,g1000}/`.

Validate actual run identities/configs, terminal budgets/checkpoints, validation role/manifests/aggregation, and exact active parameter counts. References are read-only; do not retrain or rewrite them. Saved aggregate CSVs are useful discovery sources but do not replace terminal provenance checks.

Label the different reference peak LRs visibly. The original cosine S1/S2 baselines and standalones at 0.008 are the primary references; saved cosine runs at 0.004 are supplemental and do not isolate either factor against the new arms. The four new arms provide the matched ownership and exponent-policy comparisons. Standalones have 87132 updates each; elastic runs have 348528. Matching peak LR does not equate their full schedules or parameter exposure: explain aggregate versus selected-width budgets and different schedule horizons.

## Verification required before production

Plan focused checks that establish scientific correctness:

- Analytic schedules for all four widths at position 0, around the warmup boundary, representative decay positions, and positions `T - 1` / `T`; finite nonnegative rates, identical warmup, correct endpoint exponents, and one global tick per committed update.
- Actual optimizer-step LR matches the selected width's schedule, including common parameters and decoupled weight decay. Compare selected steps with manual AdamW references from identical parameters, gradients, and nonzero moment histories.
- A forced mixed-width sequence does not accumulate LR factors, reuse a previous width's rate, or alter S1/S2 update semantics/clipping/tail behavior. In S2 verify identical shared parameter identities, selected-history changes only, width-local counters, common-parameter histories, and one global scheduler tick.
- Resume versus uninterrupted execution around warmup, representative decay and epoch boundaries; matching weights, optimizer states/counters, applied-LR sequences, width choices, and data cursor within inherited numerical tolerances.
- Historical S1/S2/C4 scheduler and checkpoint behavior remains valid; schedule metadata cannot silently reinterpret old checkpoints.
- Exact counterpart action/data stream comparisons and fresh-initialization evidence.
- A real-shape CUDA BF16 readiness gate for all four new arms at batch 64/context 128. CPU checks or synthetic shortened horizons do not establish production GPU execution or full-budget completion.

Avoid running every unrelated repository test repeatedly. Use focused regression gates and record source/config hashes, commands, outcomes, and limitations. Keep failed readiness evidence; production must bind to corrected, passing source/config identities.

After explicit later execution authorization, use existing Slurm safeguards: at most two running GPU jobs and four submitted jobs user-wide, or stricter live limits, existing partition/QoS, and node exclusions `gpu-[05,50,51,54]`. Confirm CUDA BF16 and committed updates. Continue only each new arm from its own checkpoint. This prompt does not authorize GPU submission now.

## Terminal deliverables and acceptance

Completion requires all four new arms' valid full-budget terminal checkpoints and ordinary-validation results at all four widths. The assigned new budget is 1394112 updates / 11420565504 packed tokens total. Treat failed or partial arms as incomplete; never substitute an intermediate checkpoint.

Publish under the fresh campaign root:

- 36 endpoint rows: 16 new endpoints, 16 saved S1/S2 cosine reference endpoints, and four standalone endpoints. Include loss, perplexity, grid/width, parameter counts, schedule/exponent metadata, peak LR, budgets, and provenance.
- Eight paired CaLR-minus-uniform-polynomial comparisons (four per optimizer scope), eight paired S2-minus-S1 comparisons (four per schedule policy), 16 new-minus-matching-scope-cosine comparisons at LR 0.008, and 16 new-minus-standalone comparisons. Include loss differences and perplexity ratios/relative gaps. Also report the four per-width interaction contrasts in loss: `(S2_CaLR − S2_poly) − (S1_CaLR − S1_poly)`. Other reference comparisons may be included with clear labels.
- PNG/PDF terminal loss and perplexity versus active non-embedding parameters, recorded validation-loss trajectories, and actually applied LR versus global updates. Avoid claiming reconstructed schedules are measured execution evidence.
- In each loss/perplexity comparison panel, show the matching standalone models, saved cosine S1 and S2 baselines, and all four new runs together. Use exactly these legend labels:

  | Series | Legend label |
  | --- | --- |
  | Matching standalone models, trained with cosine | `standalone-cosine` |
  | Saved S1 cosine baseline | `S1-cosine` |
  | Saved S2 cosine baseline | `S2-cosine` |
  | New S1 uniform-polynomial run | `S1-polynomial` |
  | New S2 uniform-polynomial run | `S2-polynomial` |
  | New S1 CaLR run | `S1-carl` |
  | New S2 CaLR run | `S2-carl` |

  `carl` is the user-requested plotting alias for CaLR; preserve the correct technical name `CaLR` and its schedule definition in configs, tables, and provenance. Do not interpret it as another scheduler.

- Use a primary panel with original cosine S1/S2 references at LR 0.008, matching both the new runs' and standalones' peak LR. Provide a supplemental companion panel with cosine S1/S2 references at LR 0.004, retaining the same four new runs at 0.008 and standalone points at 0.008. State baseline and new-run peak LRs, warmup, seed, and budgets in titles/captions. Keep the two cosine reference settings separate so there is only one series per legend label in a panel; do not merge their endpoints or histories. Use consistent colors/markers across panels, with S1/S2 and schedule distinctions readable.
- Plot each elastic method as a separately labeled width curve; standalone endpoints are scatter points without a connecting line. In progress plots, use the same seven-series labels and baseline-panel separation, with recorded elastic trajectories. Each standalone contributes only its terminal point at update 87132, with no invented horizontal line or trajectory. LR plots show only available elastic applied-LR evidence; standalone endpoint data must not be presented as an observed LR trajectory.
- Source/config/checkpoint hashes, validated reference identities, action/data evidence, schedule audit, update/token reconciliation, throughput, optimizer-state memory, and CUDA peak memory.

Report whether CaLR improves over its matched polynomial control and over the matching optimizer scope's cosine reference at each width. Compare S1 and S2 under each schedule, and state whether the observed CaLR effect differs by optimizer history. Keep tuned cosine S1 at 0.004 and cosine S2 at 0.008 visible as practical performance references. Highlight the g1000 standalone gap and any cost to smaller widths. Give descriptive seed-42 findings rather than claims of statistical robustness or guaranteed standalone parity.

CaLR changes cumulative LR and AdamW decay as well as width-dependent allocation. Do not claim a gain proves gradient interference was fixed or that the paper's mechanism transfers unchanged to language models. Loss ranking alone is not sufficient: all four deployed widths' absolute quality matters.

## Authorization boundary

The present request authorizes preparing this prompt only. A later `/speckit-specify` invocation authorizes specification; subsequent user instructions authorize planning, tasks, implementation, or execution as requested. Do not create the feature branch/specification, implement CaLR, launch jobs, commit, or push merely because this prompt exists.
