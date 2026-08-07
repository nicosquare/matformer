# Probabilistic Adaptive Granularity: Balanced Warmup Notes

Balanced pre-adaptive warmup separates early optimization from
posterior-guided selection. It creates no controller observations: the fixed
panel, Thompson sampler, reward model, and posterior remain unused until the
full schedule completes.

The reference uses five granularities, 500 optimizer steps, and 50 steps per
global-action window. Two independently permuted complete passes produce
exactly two windows per granularity. The resolved seed, full schedule, schedule
hash, progress, and action counts are checkpointed so an interruption inside a
window resumes identically.

`full_only` remains the default and retains the historical largest-subnetwork
warmup. `balanced_global` is restricted to Bayesian adaptive global or
per-block experiments; for per-block adaptation, each warmup action is still a
single granularity repeated across every transformer block.

Warmup journal events are audit records, not adaptive-controller data. Reports
exclude them from observation/evaluation counts, posterior diagnostics, and
adaptive action-frequency statistics. A terminally incomplete warmup never
initializes the controller.
