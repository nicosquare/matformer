# Campaign, reporting and operations contract

Use existing analyzer preflight/freeze/report commands with the new schema-3
recipe; add `report-inverse-membership --manifest NEW_FROZEN --reference-manifest
ORIGINAL_FROZEN --output-dir OUTPUT`. Paths must be fresh. New-campaign report
requires five terminals/20 endpoints independently; comparison requires original
schema-1 nine terminals and new schema-3 five terminals, yielding 44 points.
Every source is revalidated against its frozen identity and own scientific controls.

Between policies validate matching epoch traces, immutable corpus/evaluation
identities and matching elastic budgets, not matching action traces. Within IM
require equal complete action/batch traces, exact weighted sampler expectations
and observed exposure reconciliation. Every endpoint is terminal ordinary
validation at full budget with target-token-weighted causal loss, exp(loss), and
exact active counts 115264/164416/213568/262720. No model evaluation or holdout
access occurs during reporting.

Export policy-labeled CSV/JSON endpoints, 20 IM-minus-uniform deltas, selected-width
and block exposure comparisons/expectations, clipping diagnostics, measured state
bytes, peak memory, wall time, throughput and checkpoint sizes. New and comparison
views each have loss/perplexity PNG/PDF; comparison has a full recorded range
four-panel progress PNG/PDF including S1/S2/C1/C2/C3. Same arm colors, uniform solid
and IM dashed; standalones disconnected brown triangles with one legend entry.
Missing historical evidence is explicit, leaving new results valid but not calling
the 44-point comparison complete. Descriptive seed-42 interpretation only.

All experimental files live beneath a fresh campaign root, including TMPDIR,
snapshots, materialized configs, diagnostics, job files, logs, submission records,
checkpoints and reports. Default proposed root is the one in the spec. Source,
tests and docs stay in repo. Use immutable snapshots and hash-bound CPU/GPU gates.

Every GPU workload uses sbatch, one process/GPU, exclude gpu-[05,50,51]. Inspect
scheduler enforcement and live user-wide jobs at admission; max two running/four
submitted or stricter. Count other campaigns and do not restart their helpers.
Atomic submission intents, scheduler/history reconciliation and per-run writer
locks prevent duplicates. Queue remaining runs as capacity permits. Interrupted
attempts require own-checkpoint validation and resource reconciliation before
continuation. Monitor recent timing, utilization and checkpoint health. Freeze
all five terminals and generate new report even if historical comparison is
unavailable. Authorization is already granted; failed validation gates block
production until repaired, not until another user launch confirmation.
