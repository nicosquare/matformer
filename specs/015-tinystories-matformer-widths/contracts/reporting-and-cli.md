# Terminal Reporting and CLI Contract

**Campaign closed by user on 2026-09-22.** This contract retains the original
implemented protocol. C1–C3 execution and complete 24/28-endpoint publication
were cancelled; the 16-endpoint selected report is the retained closeout output.
No completion threshold is relaxed and no further execution is authorized.
See [verification.md](../verification.md).

## Terminal input and endpoint validation

Extend existing `preflight`, `freeze` and `report` to dispatch schema 4 explicitly.
Preserve old command defaults and schemas 1–3 behavior. All reporting reads saved
evidence without model training or holdout evaluation.

Require full assigned terminal checkpoints and frozen ordinary-validation
identities. Validate config/source/checkpoint/sidecar hashes, exact campaign/run/
physical width identity, actual/assigned budgets, counts, evaluation role/protocol/
manifest, 285 sequences and 36,195 causal targets under the pinned protocol,
finite loss and perplexity equal to exp(aggregated target-weighted loss). Reject
best, early, trailing-average and mismatched terminal inputs. C1/C3 require real
clipping sidecars and consistent committed counts/summary/metrics references.
The terminal reader also validates traces and resource completeness disclosures.

New frozen/report output contains exactly nine runs and 24 endpoints: four fresh
standalones and twenty elastic width points. Combined output has 28 endpoints:
all new points plus the four original Feature 013 standalones. Keys include
campaign/run and physical fraction/dimension, preserving old/new repeated
measurements at dimensions 64/128/256. Historical g750 stays .75/192/213,568.

Revalidate original frozen-manifest and preflight integrity. Select exactly
ST-g250/ST-g500/ST-g750/ST-g1000 from schema 1 and validate each selected run's
source files, checkpoint, full one-epoch terminal and ordinary-evaluation evidence.
Unrelated historical elastic artifacts are not required by this new comparison;
manifest integrity still is. Existing full-campaign comparison callers retain
their original strict validation. Never import corrected or inverse-membership
endpoints, relabel history, or rewrite original files.

Missing/incompatible history leaves validated new output available and records
the combined comparison as outstanding with an identified reason. An invalid
present historical record fails combined reporting; it cannot be silently
dropped. Complete publication requires all 28 records. No partial figure/table
may be labeled as the complete comparison.

## Tables and figures

Export `endpoints.csv` and `endpoints.json` for the new report; export
`combined_endpoints.csv` and `combined_endpoints.json` for the combined report.
Rows carry campaign/run, group, canonical arm, seed, label/fraction/dimension,
exact active count/convention, loss/perplexity, assigned/actual budgets, terminal
checkpoint/evaluation/source/config provenance, exposure, clipping and resources
as applicable. Use existing stable JSON encoding for nested CSV values. Parsed
CSV and JSON must match, including row identity and null applicability.

Combined figure stems are `loss_vs_parameters` and `perplexity_vs_parameters`,
each saved as PNG and PDF. Both contain:

- Five connected, consistently colored four-point curves for new S1/S2/C1/C2/C3
  at exact counts 90,688/115,264/164,416/262,720.
- Four disconnected fresh standalone markers at these counts and four disconnected
  historical markers at 115,264/164,416/213,568/262,720.
- Distinct open/filled marker shapes/sizes that remain visible concentrically at
  exact coincident coordinates; no count jitter or fabricated outcome differences.
- Legends `Standalone — historical grid` and `Standalone — MatFormer grid`, plus
  all elastic arms; seed 42, dataset/ordinary-validation role, count convention,
  terminal selection and one-epoch versus four-epoch budget annotations.

Reuse existing per-run scalar training/validation, exposure/clipping and resource
diagnostic reports with grid-aware labels. New-only reporting may produce its own
24-point figures; it must not imply the requested combined figures are complete.
Report manifests bind output hashes, endpoint counts, input provenance and status.
Publish atomically; rejection must not replace a previously valid report.

Interpret new elastic results primarily against fresh dense baselines. Historical
standalones provide context and seed-42 repeat measurements at three sizes.
Separate representation/history effects from C3's independent clipping and changed
block sizes. Do not infer across-seed significance, identical initial tensors,
equal compute/runtime, or equal direct width exposure from equal tokens.

## Planned command interfaces

The new interfaces below are implementation contracts, not commands already
available. Existing analyzer commands retain their current flags.

| Command | Inputs | Behavior/output |
| --- | --- | --- |
| `analyze_tinystories_optimizer_ownership.py preflight` | Required `--campaign`, `--prepared-corpus-dir`, `--tokenizer-dir`, `--output-dir`, `--run-output-root` | Validate schema-4 controls/data/real models/traces; materialize campaign_manifest.json, preflight.json and configs. No training. |
| `preflight_tinystories_matformer_widths.py --mode cpu` | Required `--campaign-root` | Read successful preflight; freeze immutable source snapshot, run CPU acceptance/compatibility checks against it, write diagnostics/cpu-gate.json with exact bindings. |
| `run_tinystories_matformer_widths.py prepare` | Required `--campaign-root`, `--cpu-evidence`; optional `--reference-manifest` | Verify snapshot/evidence/configs and adopt the matching preflight reservation; create launch plan. Does not submit training; reference unavailability is recorded separately. |
| `preflight_tinystories_matformer_widths.py --mode gpu` | Required `--campaign-root`; must execute in an authorized sbatch job | Run real-shape bf16 and explicit boundary probes from snapshot; write diagnostics/gpu-gate.json and retain all outputs/attempts. |
| `run_tinystories_matformer_widths.py queue` | Required `--campaign-root`; optional `--once` | Verify gates; reconcile intents/attempts, check live limits, admit standalones then validated-barrier elastics. --once performs one reconciliation/admission cycle. |
| `run_tinystories_matformer_widths.py worker` | Required `--campaign-root`, `--arm`, `--attempt-id` | Internal sbatch entry; verify own durable intent, gates, locks, source and stage barrier; fresh/resume/completion-only existing trainer invocation. |
| `analyze_tinystories_optimizer_ownership.py freeze` | Required `--campaign-manifest`, `--output-dir`; exactly one of `--run-root` or repeated `--run-dir` | Strictly validate/freeze nine schema-4 terminals and write frozen_manifest.json. |
| `analyze_tinystories_optimizer_ownership.py report` | Required `--manifest`, `--output-dir` | Validate schema-4 frozen input, publish 24-row exports/figures and per-run diagnostics. |
| `analyze_tinystories_optimizer_ownership.py report-matformer-widths` | Required `--manifest`, `--reference-manifest`, `--output-dir` | Validate 24 new and four historical points; publish 28-row exports and four combined figure files. |
| `run_tinystories_matformer_widths.py report` | Required `--campaign-root`; optional `--reference-manifest` overriding a recorded reference | Resume strict freeze/new-report/combined-report finalization without submitting GPU work; keep completion statuses separate. |

Analyzer errors use nonzero status and identify the discrepant file/identity.
`report-matformer-widths` returns nonzero on unavailable/invalid required history.
The operational report command persists new-report success and combined-outstanding
status but returns nonzero if the requested combined deliverable is outstanding.
Partial analyzer output, if requested through existing flags, cannot satisfy a
production barrier, complete report or readiness claim.

Queue and worker run only under later conversation execution authorization;
command availability, old launch records and passed diagnostics are not such
authorization. Keep this boundary in the runbook rather than copying historical
authorization dates into executable defaults.
