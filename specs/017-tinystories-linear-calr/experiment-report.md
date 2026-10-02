# TinyStories linear S1/S2 polynomial/CaLR experiment

Phase 5 is complete. All four new arms reached 348528 optimizer updates and 2855141376 training tokens each: 1394112 updates and 11420565504 tokens in total. Slurm, worker, CUDA BF16, resource, own-checkpoint, sampler and committed trace checks passed. All eight historical runs were revalidated read-only, contributing 20 reference endpoints.

The real report has **36 unique endpoints, 48 required directed comparisons, four interactions, 16 separate supplemental .004 comparisons and 16 PNG/PDF figures**. New-terminal, reference, trajectory and comparison states are all complete. Every published table/figure hash matches its plot-source manifest; endpoint perplexities, all pair directions/ratios/gaps and all four interactions were independently recomputed from exported CSV files.

[Terminal loss plot](evidence/phase5-final-comparison/primary_loss.png) · [Perplexity plot](evidence/phase5-final-comparison/primary_perplexity.png) · [Measured validation curves](evidence/phase5-final-comparison/primary_validation_progress.png) · [Measured applied LR](evidence/phase5-final-comparison/primary_applied_lr.png) · [Tuned-baseline comparison](evidence/phase5-final-comparison/supplemental_loss.png).

Compared with polynomial, S1 CaLR increases loss at all widths. S2 CaLR increases loss at g250/g500/g750, with especially large gaps of +0.368228 and +0.119148 at the two smallest widths; it decreases g1000 loss by 0.005925. Every new endpoint remains above its same-scope .008 cosine baseline. These are descriptive results from one seed.

## Terminal ordinary-validation loss

Primary cosine peaks are .008; new arms and standalones also use .008. Lower loss is better.

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

T031 and T042 are complete. Phase6 tasks T043–T045 remain pending and are outside this Phase5 finalization.
