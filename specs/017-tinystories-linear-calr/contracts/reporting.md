# Reporting Contract

## Reference admission and endpoints

Read-only discovery locations and LR settings are specified in spec.md. Native terminal artifacts and legacy adapters must validate actual config/run identities, saved terminal checkpoints, complete budgets, ordinary-validation role/manifests/aggregation, active counts and hashes. Best/intermediate checkpoints, aggregate tables and absent newer artifacts do not establish provenance. Adapters may read legacy config, continuation, terminal checkpoint and per-step ordinary metrics; they may not invent trace or inactive-counter evidence. Missing proof yields explicit incomplete status without rewriting or retraining references.

Publish `endpoints.csv` with exactly 36 unique terminal rows: 16 new, 16 cosine (S1/S2 × two peak settings × four widths), four standalone. Columns include actual run/checkpoint/evaluation hashes, grid/width/FFN, active total and non-embedding counts, loss/perplexity, technical schedule/exponent policy/gamma, scope, peak/warmup/horizon, actual updates/tokens and ordinary-validation provenance. Require finite loss/perplexity and agreement of perplexity with exp(loss) within documented serialization precision.

## Comparisons

`paired_differences.csv` has 48 required rows: eight CaLR−poly; eight S2−S1 (both policies); sixteen new−same-scope cosine .008; sixteen new−matching standalone. Each has left/right endpoint identities, delta_loss=left−right, perplexity_ratio=left/right and relative_gap=ratio−1. Supplemental .004 comparisons, if published, go in a distinct labeled family/table and do not change the required 48.

`interactions.csv` has four rows, one per width: `(S2_CaLR−S2_poly)−(S1_CaLR−S1_poly)` in loss units, with all four identities. No cross-peak baseline enters this factor interaction.

## Figures and measured progress

Publish PNG/PDF for primary and supplemental terminal loss/perplexity versus active non-embedding parameters; recorded validation-loss versus global updates at every width; actually applied LR versus global pre-update position/update index. A four-width subplot layout is acceptable. Separate primary/supplemental progress panels too; LR panels use only available measured elastic evidence and distinguish width-conditioned traces.

Each endpoint and per-width loss/perplexity comparison panel uses exactly `standalone-cosine`, `S1-cosine`, `S2-cosine`, `S1-polynomial`, `S2-polynomial`, `S1-carl`, `S2-carl`. Primary cosine S1/S2 use .008, supplemental use .004; new and standalone remain .008. Keep method colors/markers consistent across widths and settings. Elastic endpoints are width curves; standalone endpoints are unconnected scatter points. Progress elastic lines use recorded ordinary-validation trajectories; standalone points occur only at update 87132. Missing trajectories remain missing and affected completion is incomplete. Do not draw invented horizontal standalone trajectories.

LR curves use measured applied evidence; never draw analytic reconstructed schedules as observed data. Absent historical applied records are disclosed and omitted, rather than inferred from stored next-update rates. Titles/captions state peaks, warmup=64, seed=42, elastic/standalone budgets and selected-width versus global exposure. The plot alias `carl` maps to technical CaLR in configuration, tables and provenance.

## Publication and findings

A plot-source manifest binds figures/tables to source/config/checkpoint/evaluation/metric hashes and validated references; publish only reconciled complete sets atomically. Expose independent new-terminal, reference, trajectory and comparison statuses. Reporting errors return nonzero and preserve valid new evidence; partial results are clearly labeled.

Interpret all widths for both scopes: CaLR versus polynomial and matching cosine, S1 versus S2, interactions, tuned S1 .004 and S2 .008 practical references, g1000 standalone gap and smaller-width costs. Explain elastic/standalone horizons, shared exposure and that cumulative LR/AdamW decay also change. Results are descriptive seed-42 observations with no guarantee of parity, robustness, fixed interference or exact paper mechanism transfer.
