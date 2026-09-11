# TinyStories GMC/LMC comparison — new-chat prompt

Work in `/home/ivo.navarrete/ElasticNN/matformer`.

Continue feature 013, `tinystories-optimizer-ownership`, using its existing
Spec Kit documents to implement and execute the comparison of concat methods
C1, C2, and C3 with gradient membership correction (GMC) and learning-rate
membership correction (LMC).

## Workflow and authorization

Extend the existing Spec Kit artifacts in this order:

1. Read and update `specs/013-tinystories-optimizer-ownership/spec.md` to include
   this correction comparison within the existing feature.
2. Clarify unresolved scientific choices that materially affect the comparison.
3. Update the existing `plan.md` and relevant contracts.
4. Extend the existing `tasks.md` with actionable tasks, preserving completed
   tasks and their status.
5. Analyze consistency across the specification, plan, and tasks.
6. Implement, test, and perform a short GPU preflight.
7. Launch and monitor the six experiments after the acceptance checks pass.
8. Generate the comparison reports.

This prompt authorizes the implementation, preflight, and six-run launch after
successful validation. Proceed through the workflow without asking for repeated
approval of these actions. Resolve routine implementation choices independently.
Report any material scientific ambiguity before making a choice that changes the
experiment defined below.

Keep this work under `specs/013-tinystories-optimizer-ownership/`. Do not create
a new feature number, feature directory, or feature branch, or invoke a workflow
that scaffolds a new feature. Apply relevant Spec Kit skills to the existing
artifacts; update them directly where a skill would otherwise create a feature.
Keep the original uncorrected campaign requirements explicit while extending
the documents for the correction comparison.

Assign distinct experiment identities and output directories to the six new
runs so the completed results and their scientific provenance remain intact.
Inspect current Git status, branch, and history before making changes; preserve
unrelated work.

## Required context

First read:

- `AGENTS.md`
- `specs/013-tinystories-optimizer-ownership/plan.md`
- The relevant specification and contracts under that feature directory
- `notes/gradient_membership_correction_adamw_analysis_2026-09-10.md`
- `notes/tinystories_optimizer_ownership_results_analysis_2026-09-10.md`
- `configs/controlled_exps/tinystories_instruct_optimizer_ownership.yaml`
- `docs/tinystories-optimizer-ownership-experiment.md`

The completed original campaign contains four standalones and S1/S2/C1/C2/C3.
Its elastic arms used `correction_mode=none`. Use the existing uncorrected
C1/C2/C3 and standalone results as references; do not retrain those baselines.

The LMC numerator fix was committed as:

```text
cbcd567 fix: align LMC normalization with configured trained widths
```

Its recorded CPU verification was 396 passed, 28 skipped, and one expected
failure. Verify the current checkout and rerun checks appropriate to new changes;
do not assume the historical test result establishes C3 correction support.

## Experiment matrix

Run six fresh experiments:

| Run | Representation | AdamW state ownership | Clipping | Correction |
| --- | --- | --- | --- | --- |
| C1-GMC | Concat | Shared | Global, cap 1.0 | GMC |
| C1-LMC | Concat | Shared | Global, cap 1.0 | LMC |
| C2-GMC | Concat | Per width | Global, cap 1.0 | GMC |
| C2-LMC | Concat | Per width | Global, cap 1.0 | LMC |
| C3-GMC | Concat | Per FFN block plus common owner | Per owner, cap 1.0 | GMC |
| C3-LMC | Concat | Per FFN block plus common owner | Per owner, cap 1.0 | LMC |

For C3, preserve the five disjoint owners: O-A, O-B, O-C, O-D, and O-common.
For C2, preserve the original meaning of per-width optimizer histories over
shared model weights.

All six runs start from fresh initialization using the original procedure and
seed 42. Do not initialize them from completed or partially trained checkpoints.
An interruption of one of these new runs should resume its own valid checkpoint.

Match each original arm's controls except for the specified correction:

- Four widths: g250, g500, g750, g1000, with FFN fractions 0.25, 0.5, 0.75, 1.0.
- One sampled width per update, uniform random sampling with replacement and
  the original independent action/data RNG behavior.
- Original model, tokenizer, prepared corpus, partition and manifest hashes,
  data order, batch size, precision, AdamW hyperparameters, and validation policy.
- Original learning-rate schedule, warmup, and full training horizon. Advance
  the global scheduler once per completed update regardless of active owners.
- 348,528 updates and 2,855,141,376 training tokens per elastic run.

Verify these controls against the saved resolved configurations and artifacts.
Preserve paired action and batch sequences and verify their digests. Do not
introduce additional budget balancing, adaptive sampling, or hyperparameter tuning.

## Correction semantics

Specify and test the following behavior explicitly:

1. GMC scales gradients by the configured trained-width count divided by the
   block's membership count. For all four widths the factors are
   **[1, 4/3, 2, 4]**.
2. LMC uses those same configured membership factors to scale the completed
   AdamW parameter change. The numerator remains four when only one width is
   sampled in an update.
3. Preserve the existing LMC mode: it also enables GMC gradient hooks. Its order
   is gradient correction, the arm's clipping operation, AdamW, then parameter
   change correction. Describe it as this combined intervention in the report;
   do not silently turn it into an LR-only ablation.
4. LMC scales the entire AdamW parameter change, including weight decay. It
   must not directly rescale optimizer moments or step counters.
5. Apply correction to the relevant FFN block weights and block-local biases.
   Common parameters, including any shared down-projection bias, receive no
   direct membership multiplier.
6. Inactive concat blocks with absent gradients retain their weights, moments,
   and step counters. Each active block receives its correction exactly once.
7. Preserve the distinction between the base scheduled LR and the effective
   block LR. Record the correction factors alongside the base rate.

The experiment covers concat only. Do not remove existing slicing GMC support
or change slicing behavior as part of this work.

The analysis note also records a separate custom-label subset membership issue.
This campaign trains all four labels. Verify its actual metadata and avoid
expanding this task into unrelated subset support.

## Required implementation work

Audit the real training paths rather than assuming that setting
`correction_mode` is sufficient.

Two C3 gaps were identified in the inspected checkout:

- `src/utils/config.py` requires `correction_mode=none` for `per_ffn_block`.
- The `BlockOptimizerCollection` branch in `src/training/steps.py` steps owners
  directly and bypasses the existing LMC application helper.

Extend C3 eligibility and integrate correction into its update lifecycle. Simply
removing the configuration restriction is insufficient. Preserve disjoint owner
coverage, common-owner behavior, active-owner stepping, clipping order, and one
global scheduler advance. Preserve fatal handling of partial updates and prevent
saving or publishing a partially corrected state as a resumable checkpoint.

Verify the corresponding C1 and C2 paths, including selected-width optimizer
state and effective update factors.

Extend feature 013's campaign tooling with a separate six-arm campaign
configuration and an explicit correction contract in its existing contracts.
Update configuration validation, checkpoint compatibility, provenance, and
reporting as needed. Preserve strict validation for the original campaign;
do not relabel old artifacts to make them fit the extended protocol.

Retain the existing compact optimizer accounting and performance fixes. Do not
reintroduce growing per-update history sorting/copying or full optimizer-state
snapshots. Measure any overhead introduced by correction handling.

## Acceptance checks before full training

Use `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python` for CPU checks.

Provide meaningful regression coverage for:

- Resolved configuration and real-model correction execution in all six arms.
- The four single-width selections and their expected membership factors.
- Actual AdamW parameter changes against an independent reference with explicit
  block learning rates, including nonzero weight decay and bias handling.
- GMC hooks, clipping order, optimizer moments, and inactive tails after prior
  full-width training steps.
- C3 applying each correction once, stepping only the intended owners, and
  advancing the scheduler once.
- Checkpoint/resume equivalence, correction-contract mismatches, and C3 failures
  during owner steps, correction, or subsequent update bookkeeping.
- Existing uncorrected ownership behavior and original reporting compatibility.

After CPU checks pass, perform a short GPU preflight through `sbatch`, covering
all ownership/correction combinations. Keep preflight artifacts separate from
production run directories. Verify finite losses, actual correction factors,
checkpoint saves, restored progress, and recent steady-state throughput.

Do not launch the full campaign if a required correction is rejected, bypassed,
applied twice, or produces invalid state. Resolve failures and record evidence
that the preflight passed before submission.

## Storage, submission, and monitoring

All experimental results and operational artifacts must remain beneath:

```text
/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1
```

Create a distinct campaign subdirectory, for example `campaigns/concat-gmc-lmc-v1`,
after checking for existing contents. Never overwrite a conflicting campaign.
Store its run directories, materialized configs, diagnostics, launchers, logs,
submission records, checkpoints, and reports beneath that new directory.
Source code, specifications, tests, and documentation belong in the repository.

- Run every GPU workload through `sbatch` and exclude `gpu-[05,50,51]`.
- Use one training process on one GPU per run.
- Respect observed limits of two running and four submitted GPU jobs, including
  other jobs that count toward the same user limits.
- Inspect current Slurm state and existing submission helpers before launching
  a helper for this campaign. Do not reactivate old campaign submissions.
- Ensure only one process writes each run directory. Make submission tracking
  resilient to helper restarts and prevent duplicate jobs.
- Record the source commit and any additional source hashes, resolved config,
  scientific identity, Slurm job IDs, attempts, and consumed resources.
- On interruption, reconcile resource accounting and records beyond the last
  durable checkpoint before resuming that run.
- Monitor progress, recent update time, GPU utilization, checkpoint health, and
  estimated finish times. Distinguish startup/validation time from steady-state
  training and new measurements from historical runtime measurements.

## Evaluation and deliverables

Do not evaluate the sealed holdout. Use the original ordinary-validation role,
protocol, and terminal-checkpoint rules for endpoint comparisons.

Deliver:

1. Updated feature 013 specification, plan, tasks, contracts, consistency
   analysis, and runbook.
2. The implementation and regression checks, with CPU/GPU validation evidence.
3. Six completed runs with validated terminal checkpoints and resource records.
4. CSV/JSON endpoint tables comparing the six corrected runs with the completed
   uncorrected C1/C2/C3 and standalone references.
5. Loss and perplexity versus active non-embedding parameters, exported as PNG
   and PDF. Keep standalone points as brown triangles with one legend entry,
   `Standalone`. Use `Loss` or `Perplexity` for the y-axis and
   `Active non-embedding parameters` for the x-axis. Preserve the shortened
   footnote, without restoring the removed second sentence.
6. A four-panel ordinary-validation loss-progress figure, one panel per width,
   comparing the corrected methods and their uncorrected concat baselines over
   the full recorded update range. Label methods and correction modes clearly.
7. A concise interpretation of quality, exposure, clipping, and runtime. Treat
   these as paired seed-42 observations; do not claim significance across seeds.

Keep progress updates concise. When reporting readiness or completion, state
what has actually been verified and any remaining limitations.
