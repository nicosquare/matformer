# S1 interim comparison, 2026-10-02

Both S1 production jobs completed their exact 348,528-update / 2,855,141,376-token budgets, with successful Slurm and matching worker/CUDA/resource evidence. Selected native terminal/config/model/optimizer/RNG/sampler/trace validation and the legacy tuned-baseline adapter passed. All plots use recorded ordinary-validation measurements.

At peak LR .008, terminal loss and perplexity rank standalone, S1 cosine, S1 polynomial, S1 CaLR from lowest to highest at every width. This is one seed (42); it does not establish statistical significance. Each standalone has 87,132 updates / 713,785,344 tokens, while each elastic run has four times that total budget distributed across widths.

Primary figures use the original .008 cosine baseline; supplemental figures substitute the historical tuned .004 S1 cosine baseline. Standalones are plotted as terminal points on progress plots.

## Reporting correction

The original report helper omitted the live sampler required for packed checkpoint validation. The corrected helper stages it from a deep copy of the saved runtime configuration, retaining the admitted comparison signature. The queue continuation helper also reconstructs a differing comparison signature; this plot generator uses the strict saved-config terminal validator for completion evidence. Frozen production sources and historical results were left unchanged. The reporting suite passes 56 tests (see ../s1-interim-reporting-tests.txt).

This is an interim S1 report. S2 completion and the full campaign report remain separate. S2 was already running when review was requested; the user elected to let both jobs finish. Continuous monitoring remains stopped.
