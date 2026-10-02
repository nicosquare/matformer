# TinyStories linear S1/S2 polynomial/CaLR experiment

Phases 5 and 6 are complete. All four new arms reached 348528 optimizer updates and 2855141376 training tokens each: 1394112 updates and 11420565504 tokens in total. Slurm, worker, CUDA BF16, resource, own-checkpoint, sampler and committed trace checks passed. All eight historical runs were revalidated read-only, contributing 20 reference endpoints.

The real report has **36 unique endpoints, 48 required directed comparisons, four interactions, 16 separate supplemental .004 comparisons and 16 PNG/PDF figures**. New-terminal, reference, trajectory and comparison states are all complete. Every published table/figure hash matches its plot-source manifest; endpoint perplexities, all pair directions/ratios/gaps and all four interactions were independently recomputed from exported CSV files.

[Terminal loss plot](evidence/phase5-final-comparison/primary_loss.png) · [Perplexity plot](evidence/phase5-final-comparison/primary_perplexity.png) · [Measured validation curves](evidence/phase5-final-comparison/primary_validation_progress.png) · [Measured applied LR](evidence/phase5-final-comparison/primary_applied_lr.png) · [Tuned-baseline comparison](evidence/phase5-final-comparison/supplemental_loss.png).

Compared with polynomial, S1 CaLR increases loss at all widths. S2 CaLR increases loss at g250/g500/g750, with especially large gaps of +0.368228 and +0.119148 at the two smallest widths; it decreases g1000 loss by 0.005925. Every new endpoint remains above its same-scope .008 cosine baseline. These are descriptive results from one seed.

## Terminal ordinary-validation loss

Saved cosine runs are external comparisons. Primary cosine peaks are .008; new arms and standalones also use .008. Lower loss is better.

| Width | Standalone | S1 cosine | S2 cosine | S1 poly | S1 CaLR | S2 poly | S2 CaLR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| g250 | 2.062977 | 2.068157 | 2.067589 | 2.071964 | 2.081180 | 2.076825 | 2.445053 |
| g500 | 1.991942 | 1.995913 | 1.991578 | 2.000365 | 2.007385 | 2.000802 | 2.119950 |
| g750 | 1.943363 | 1.966166 | 1.963845 | 1.973704 | 1.977642 | 1.972476 | 1.985223 |
| g1000 | 1.909057 | 1.953872 | 1.951732 | 1.962806 | 1.968072 | 1.962459 | 1.956534 |

## All-width findings

Seed-42 descriptive ordinary-validation results. Negative loss differences favor the left endpoint.

g250:
S1: CaLR−polynomial +0.009216; CaLR−cosine .008 +0.013023; polynomial−cosine .008 +0.003807; CaLR−practical cosine peak 0.004 +0.018214; polynomial−practical cosine +0.008998; CaLR−standalone +0.018203; polynomial−standalone +0.008987.
S2: CaLR−polynomial +0.368228; CaLR−cosine .008 +0.377464; polynomial−cosine .008 +0.009236; CaLR−practical cosine peak 0.008 +0.377464; polynomial−practical cosine +0.009236; CaLR−standalone +0.382075; polynomial−standalone +0.013848.
Ownership S2−S1: +0.004861, +0.363873; interaction +0.359012.

g500:
S1: CaLR−polynomial +0.007020; CaLR−cosine .008 +0.011472; polynomial−cosine .008 +0.004452; CaLR−practical cosine peak 0.004 +0.015843; polynomial−practical cosine +0.008823; CaLR−standalone +0.015443; polynomial−standalone +0.008423.
S2: CaLR−polynomial +0.119148; CaLR−cosine .008 +0.128371; polynomial−cosine .008 +0.009224; CaLR−practical cosine peak 0.008 +0.128371; polynomial−practical cosine +0.009224; CaLR−standalone +0.128008; polynomial−standalone +0.008860.
Ownership S2−S1: +0.000437, +0.112564; interaction +0.112128.

g750:
S1: CaLR−polynomial +0.003938; CaLR−cosine .008 +0.011475; polynomial−cosine .008 +0.007537; CaLR−practical cosine peak 0.004 +0.015532; polynomial−practical cosine +0.011594; CaLR−standalone +0.034278; polynomial−standalone +0.030340.
S2: CaLR−polynomial +0.012748; CaLR−cosine .008 +0.021378; polynomial−cosine .008 +0.008631; CaLR−practical cosine peak 0.008 +0.021378; polynomial−practical cosine +0.008631; CaLR−standalone +0.041860; polynomial−standalone +0.029113.
Ownership S2−S1: -0.001228, +0.007582; interaction +0.008809.

g1000:
S1: CaLR−polynomial +0.005266; CaLR−cosine .008 +0.014199; polynomial−cosine .008 +0.008934; CaLR−practical cosine peak 0.004 +0.017054; polynomial−practical cosine +0.011788; CaLR−standalone +0.059015; polynomial−standalone +0.053750.
S2: CaLR−polynomial -0.005925; CaLR−cosine .008 +0.004802; polynomial−cosine .008 +0.010727; CaLR−practical cosine peak 0.008 +0.004802; polynomial−practical cosine +0.010727; CaLR−standalone +0.047478; polynomial−standalone +0.053403.
Ownership S2−S1: -0.000347, -0.011537; interaction -0.011191.

The g1000 standalone gaps above describe full-width parity; the smaller-width gaps describe elastic sharing costs. Elastic runs have 348528 global updates versus 87132 standalone updates; selected-width exposure differs from global exposure. Changing exponents changes cumulative LR and AdamW decay as well as update timing. One seed cannot establish robustness, guaranteed improvement, resolved interference or transfer of the CNN paper mechanism.

## Evidence and practical limits

The canonical report is `/nfs-stor/ivo.navarrete/results/elasticnn/tinystories-linear-s1-s2-calr-v1/reports/comparison`; a reviewable copy is [saved here](evidence/phase5-final-comparison/comparison_report.json). [Plot sources](evidence/phase5-final-comparison/plot_sources.json) bind actual configs, checkpoints, ordinary evaluations, recorded metrics and reporter code; [reporting snapshot manifest](evidence/phase5-final-comparison/reporting-source-manifest.json) binds the separate corrected reporting snapshot. [Production terminal proof](evidence/phase5-production-terminals.json) binds scheduler/worker/resource evidence and the unchanged production source/config/CPU/GPU gates.

All curves are measured; no observed LR was reconstructed analytically. LR exports retain the first65, every128th global commit and last per width after validating the entire trace. Standalone progress plots contain only the terminal point at87132. Primary .008 and supplemental .004 baselines occupy separate panels. Historical files were not modified or retrained.

Each production arm has one successful attempt and zero failed production-process seconds. Ordinary-validation and update costs overlap process time, which overlaps Slurm allocation; they must not be summed. The earlier failed GPU diagnostic's18 allocation seconds remain separately documented in verification.md. Reporting-only source fixes do not relabel or replace production readiness. Continuous monitoring remains stopped.

T031 and T042 are complete. Phase 6 reconciles acceptance below; final CPU results and command bindings are in verification.md.


## Phase 6 acceptance reconciliation — 2026-10-02

All 19 functional requirements, six experiment requirements and eight success criteria have saved evidence. The [final audit](evidence/phase6-acceptance-audit.json) independently rehashes every table/figure output and the report/terminal source artifacts at their recorded paths, including the four terminal checkpoints and historical references. It recomputes exp(loss), all 48 required and 16 supplemental pair differences/ratios/gaps, all four interactions, unique endpoint counts, per-arm budgets, exposure sums and terminal trace watermarks. All checks pass; no real-execution/reference/trajectory criterion remains incomplete.

| Criterion | Verified behavior | Evidence | Status |
| --- | --- | --- | --- |
| FR-001 | Four exact arms, four linear prefixes, one seed | campaign tests; production terminal proof | PASS |
| FR-002 | Closed factor audits; fresh equal weights and full deterministic streams | campaign tests; saved CPU gate | PASS |
| FR-003 | Pre-update global clock and exact schedule boundaries | schedule/resume tests; committed trace/checkpoint hashes | PASS |
| FR-004 | Actual-model complexity 377408/426560/475712/524864 | campaign/foundation tests; endpoint count fields | PASS |
| FR-005 | Uniform global selection, shared versus selected per-width histories and joint clipping | update/resume tests; terminal selection/owner counts | PASS |
| FR-006 | All-group temporary LR, restored nominal rates and inherited tails/decay | manual AdamW/schedule tests; full applied-trace validation | PASS |
| FR-007 | Matched inherited controls/corpus/manifests; ordinary validation only | closed campaign audits; CPU gate; endpoint manifests | PASS |
| FR-008 | 348528 updates / 2855141376 tokens per arm; exact campaign totals | production terminals; phase6 audit; separate cost records | PASS |
| FR-009 | Enumerated counterpart/schedule/ownership differences; schemas 1–5 and S1/S2/C4 compatible | campaign/foundation and historical regression suites | PASS |
| FR-010 | Fresh immutable root, four runs, durable intents and independent lifecycle states | queue fixtures; saved preparation/submission/terminal evidence | PASS |
| FR-011 | Own-state restoration, pre-mutation rejection, poisoned-save refusal and zero-step recovery | resume/queue suites; validated production checkpoints | PASS |
| FR-012 | Separate authorization, matching CPU/all-arm BF16 gates and enforced Slurm admission | saved authorizations/gates; production bindings; queue fixtures | PASS |
| FR-013 | Eight historical runs admitted read-only with native/legacy terminal proof | saved report source hashes; reporting tests; phase6 source rehash | PASS |
| FR-014 | Four complete checkpoints and 16 ordinary-validation endpoints | production terminals; phase6 checkpoint/evaluation rehash | PASS |
| FR-015 | 36 unique endpoints with schedule/count/budget/provenance fields | endpoints.csv; phase6 audit | PASS |
| FR-016 | 8/8/16/16 required pairs and four signed interactions; supplemental family separate | paired_differences.csv; interactions.csv; independent phase6 arithmetic | PASS |
| FR-017 | Eight primary/supplemental figure families in PNG/PDF; measured progress/LR | 16 figure hashes; plot-source manifest; reporting fixtures | PASS |
| FR-018 | Seven exact labels; separate elastic curves and unconnected standalone terminal points | plot-source labels and measured exports; reporting fixtures | PASS |
| FR-019 | Every-width matched/practical baselines, ownership interactions and standalone gaps | All-width findings above; findings.md; reporting fixtures | PASS |
| EX-001 | Versioned factors/contracts and immutable identities | resolved configs; checkpoints; production/report source manifests | PASS |
| EX-002 | All commits bind width/owner/count/gamma/p/LR/action/data and durable watermark | validated trace hashes; endpoint watermarks/counts; phase6 rehash | PASS |
| EX-003 | Scalar/measured trajectories, provenance, throughput, optimizer/CUDA memory and costs | metrics/report exports; real-shape GPU gate; resource ledgers | PASS |
| EX-004 | All boundaries/widths, nonzero-moment manual updates, streams, continuation and compatibility | focused/historical CPU suites; saved 151-case CUDA suite | PASS |
| EX-005 | Commands/results/bindings and retained failed readiness; real execution distinct | verification.md; CPU/GPU gate records; retained readiness roots | PASS |
| EX-006 | External primary .008 / supplemental .004; unequal horizons/exposure and cumulative decay disclosed | Evidence and practical limits; all-width findings; separate plot panels | PASS |
| SC-001 | Four definitions, zero undeclared differences, counts and stream equality | campaign tests; full CPU preflight gate | PASS |
| SC-002 | Boundary/mixed-width updates and faithful continuation | focused CPU tests; saved CUDA boundary checks | PASS |
| SC-003 | All-arm source-bound readiness before authorized limited submissions | production bindings; saved GPU gate/intents; queue tests | PASS |
| SC-004 | Four exact terminals, 16 endpoints and exact aggregate budgets | production terminal proof; phase6 totals/checkpoint rehash | PASS |
| SC-005 | 36 endpoints, 48 comparisons, four consistent interactions | independent phase6 CSV arithmetic | PASS |
| SC-006 | All PNG/PDF families and exact labels, no invented trajectories | figure/source hashes; recorded exports; reporting fixtures | PASS |
| SC-007 | Effect direction/magnitude, interactions and every-width standalone cost visible | All-width findings and published comparison tables | PASS |
| SC-008 | Reconciled budgets/action/data/LR/resources; references immutable; failures separate | production proof; full-trace validation; source rehash; retained diagnostic failure | PASS |

The common evidence is the [production terminal proof](evidence/phase5-production-terminals.json), [saved comparison](evidence/phase5-final-comparison/comparison_report.json), [plot sources](evidence/phase5-final-comparison/plot_sources.json), [final CPU gate](evidence/phase5-final-cpu-gate.json), [all-arm GPU gate](evidence/phase5-final-gpu-gate.json), and Phase 6 logs/bindings in verification.md. This reconciliation combines those previously validated real artifacts with final CPU regressions; CPU tests alone do not establish terminal or GPU acceptance.

Production costs retain one successful attempt per arm, zero failed production-process seconds, no incomplete/unobserved attempts, ordinary-validation costs separately, and the earlier failed diagnostic's 18 allocation seconds. Costs overlap process/allocation time and are not summed. The audit reads saved evidence and does not rerun training or restart monitoring. Historical source hashes match the published admission records. The one-seed limitations and recorded-LR plot sampling remain unchanged.
