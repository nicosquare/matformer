# TinyStories inverse-membership experiment

Feature [014](../specs/014-tinystories-inverse-membership/spec.md) compares fresh
S1-IM/S2-IM/C1-IM/C2-IM/C3-IM with the original uncorrected Feature 013 runs.
Widths g250/g500/g750/g1000 have probabilities .12/.16/.24/.48, from inverse
incremental-block memberships 4/3/2/1. Every update draws independently with
replacement and uses one width globally. Correction stays none. Seed 42, normal
fresh initialization, data order, optimizer/clipping controls and four-epoch
budgets are inherited. Each run has 348,528 updates / 2,855,141,376 tokens.

All GPU jobs are authorized by the user after validation. Submit through sbatch,
exclude gpu-[05,50,51], observe two running/four submitted user-wide or stricter
limits, and preserve unrelated campaign helpers. The final holdout stays sealed.

## Artifacts and verification

Campaign root:
`/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/campaigns/inverse-membership-sampling-v1`.
All runtime artifacts live below it; code/specifications/tests/docs remain in repo.
Use `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python` throughout.

- `campaign/`: audited preflight, manifest and five ordinary trainer configurations.
- `source/`: immutable source snapshot; hashes in `launchers/plan.json`.
- `diagnostics/`: CPU evidence, GPU smoke/resume and timing, hash-bound gates.
- `launchers/`: reviewable sbatch scripts, queue/writer locks, job intents/IDs,
  scheduler/accounting observations, progress and completion state.
- `runs/<arm>/`: checkpoints, action/batch/clipping histories, metrics, resource
  attempt ledger and terminal ordinary-validation evidence.
- `reports/new-campaign/`: 20 endpoints and loss/perplexity PNG/PDF.
- `reports/comparison/`: 44 endpoints, 20 paired deltas, exposure/clipping/resource
  comparisons, loss/perplexity and full-range four-panel progress PNG/PDF.

Record actual test results and job IDs in the feature's
[verification record](../specs/014-tinystories-inverse-membership/verification.md).
GPU diagnostics deliberately stop after durable checkpoints at updates 192 and
256, restore at 192, and retain the production schedule horizon. They compare
actual action digests to the production sampler, confirm finite losses and owner
counts, and measure update timing after warmup. They use separate identities and
never become production starting weights.

## Commands

Set `IM_ROOT` to the campaign root and `IM_PYTHON` to the Python path above.
Preflight uses `scripts/analyze_tinystories_optimizer_ownership.py preflight`
with the Feature 014 recipe, prepared corpus/tokenizer paths, output-dir
`$IM_ROOT/campaign` and run-output-root `$IM_ROOT/runs`. It audits immutable
inputs and reserves fresh identities without training. It rejects an occupied
output; inspect conflicts rather than overwrite existing evidence.

After CPU tests pass, save the exact command/results as JSON with status passed,
then snapshot and create job files:

```bash
"$IM_PYTHON" scripts/run_tinystories_inverse_membership.py prepare \
  --campaign-root "$IM_ROOT" --cpu-evidence "$IM_ROOT/diagnostics/cpu-evidence.json" \
  --reference-manifest /nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/reports/complete-20260910T181951Z/frozen/frozen_manifest.json
```

Inspect live Slurm limits and queue before diagnostic submission. Request one GPU
on cscc-gpu-p with cscc-gpu-qos, exclude gpu-[05,50,51], direct stdout/stderr below
`$IM_ROOT/logs`, and submit `$IM_ROOT/launchers/diagnostic.sbatch`. Its complete
five-arm success writes `diagnostics/gpu-gate.json` bound to snapshot hashes.
No production job is admitted until both source-matched gates pass.

```bash
"$IM_PYTHON" "$IM_ROOT/source/scripts/run_tinystories_inverse_membership.py" queue \
  --campaign-root "$IM_ROOT"
```

The queue tracks atomic submission intents and distinct `oo-im-v1-*` names,
submits at most the applicable global limit, and queues the fifth arm as capacity
opens. A per-arm writer lock guards each worker. `--once` performs one admission/
monitor cycle. An ambiguous intent or failed attempt requires reconciliation
before resubmission; never delete intent records or silently restart from scratch.
Continuation must use the same arm/config/full horizon and its own valid durable
checkpoint. Retain consumed work beyond the restored checkpoint in attempt costs.

On completion the helper freezes all five terminals, writes the new report and
attempts the historical comparison. If historical evidence is missing/incompatible,
`launchers/completion.json` identifies the outstanding comparison and the valid
new report remains available. No fabricated or replacement references are allowed.
Reporting can also be invoked explicitly via helper mode `report`.

## Interpretation

Compare fixed-IM minus uniform within each arm. Across arms, slicing and concat
have different inactive-tail momentum/decay behavior, per-width optimizers have
separate histories over shared weights, and C3 clips disjoint groups independently.
Actual selections and block activations are distinct. Expected A/B/C/D support
probabilities are 1/.88/.72/.48 under IM versus 1/.75/.50/.25 under uniform;
these are not enforced counts or guarantees of equal exposure.

Endpoints use target-token-weighted causal ordinary-validation loss and exp(loss)
at the full terminal budget, never best checkpoints or trailing means. Exact
active non-embedding counts are 115264/164416/213568/262720. Historical standalones
retain their one-epoch budgets and appear as disconnected brown triangles.
Equal training tokens do not imply equal compute/runtime; IM chooses wider
networks more often. Seed-42 findings are descriptive, without across-seed error
bars or significance claims. Inspect resource-ledger completeness and replay
costs before interpreting runtime differences.

## Current execution record — 2026-09-11

CPU gates and five-arm GPU diagnostic job 229045 passed. Production jobs
229058/229059/229060/229061 cover S1-IM/S2-IM/C1-IM/C2-IM and were pending
scheduler priority at the latest check. C3-IM is waiting for admission capacity.
Detached monitor PID 2917977 records live state in `launchers/status.json` and
will perform terminal freeze/reporting after completion. Results are pending;
see the verification record and campaign artifacts for subsequent state.
