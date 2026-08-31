# Experiment and Analysis Contract

## Controlled Arms

For each seed, run exactly two arms from the dedicated controlled recipe:

- baseline: `state_scope=shared`;
- candidate: `state_scope=per_granularity`.

The only intended resolved scientific-control difference is state scope. Both
arms use one shared model parameter set.

## Frozen Controls

Each pair must agree on:

- seed and model-initialization identity;
- tokenizer, prepared corpus, optimizer-training role, ordinary validation,
  controller role, and final-holdout hashes;
- model topology and ordered widths;
- optimizer family, kwargs, base/resolved learning rate;
- one global scheduler contract and complete learning-rate trace;
- balanced-cycle action state and complete width-action trace;
- optimizer-training sampler identity and batch/data cursor sequence;
- token budget, committed steps, evaluation cadence, and terminal-checkpoint
  rule;
- mixed precision, correction, accumulation, and one-process topology.

The arm-invariant paired-control signature excludes only state scope. Full run
and checkpoint identities retain scope.

## Budgets and Exposure

| Phase | Tokens | Global steps | Updates per width | Seeds |
|---|---:|---:|---:|---|
| Pilot | 713,785,344 | 87,132 | 21,783 | 42, 43, 44 |
| Confirmation | 2,141,356,032 | 261,396 | 65,349 | 42, 43, 44 |

Balanced-cycle interval one must complete exact four-width cycles at both
budgets. Confirmation uses fresh run IDs and initial states; pilot checkpoints
are never continued.

## Terminal Checkpoint Rule

- The scientific endpoint is the exact terminal fixed-budget resumable
  checkpoint, normally `checkpoints/latest.pt` saved on completion.
- Independently selected best-evaluation checkpoints are not comparison
  endpoints.
- The frozen manifest records the terminal checkpoint path, SHA-256, byte size,
  committed tokens and steps, scope, and checkpoint purpose.
- Explicit final-holdout evaluation always receives the frozen checkpoint path.

## Freeze Phase

`scripts/analyze_tinystories_per_width_optimizer.py freeze` reads six completed
run directories and writes an atomic comparison manifest. It must reject the
freeze if:

- a seed or scope is missing or duplicated;
- any run is incomplete or its budget/step/count reconciliation fails;
- terminal checkpoints are missing, model-only, or not at the declared budget;
- paired control signatures differ;
- ordered widths, role hashes, action traces, batch/sampler identity, learning
  rates, cadence, or terminal rule differ within a pair;
- any resolved difference other than state scope is found;
- update counts fail to match width exposures.

The manifest freezes endpoint definitions, checkpoint hashes, holdout status,
and claim status before optional final-holdout evaluation.

## Holdout Policy

- Ordinary validation is available for pilot trajectory and stability analysis.
- If a confirmation may be run, the final holdout remains sealed throughout
  the pilot and confirmation decision.
- Opening the holdout for pilot evaluation permanently marks later reuse of the
  same examples as descriptive.
- A confirmatory result requires all three confirmation seeds, the frozen
  three-epoch protocol, and a holdout not inspected during the pilot.
- The analyzer never evaluates the holdout implicitly. Evaluation remains an
  explicit call to `scripts/evaluate_final_holdout.py` with the frozen terminal
  checkpoint.

## Report Phase

`scripts/analyze_tinystories_per_width_optimizer.py report` reads only the
frozen manifest and referenced immutable artifacts. It validates all hashes
before emitting:

- `optimizer_state_comparison.json`: full structured provenance and results;
- `optimizer_state_comparison.csv`: tidy seed/width/endpoint/resource rows.

If final-holdout results are absent, the report is an ordinary-validation and
resource diagnostic only. If present, their checkpoint and role hashes must
match the frozen manifest.

## Endpoints

### Primary

For each seed:

```text
per_granularity uniform mean final-holdout loss
- shared uniform mean final-holdout loss
```

The report preserves the signed seed-level differences and descriptive
aggregate summaries; it does not invent an unplanned hypothesis test.

### Secondary

- final-holdout loss and perplexity for each width and paired differences;
- worst-width final-holdout loss;
- trailing-five ordinary-validation mean for each width;
- matched-token convergence trajectories;
- per-width optimizer updates and exposure reconciliation;
- training wall time, peak accelerator memory, terminal checkpoint size, and
  paired cost deltas.

## Claim Labels

- `diagnostic`: pilot or incomplete-seed evidence.
- `confirmatory`: all frozen confirmation requirements and untouched-holdout
  conditions satisfied.
- `descriptive_after_holdout_open`: the same holdout was previously inspected.

One seed or smoke results are never labeled confirmatory. Scientific
superiority is not inferred by the tool; it reports the frozen outcomes and
eligibility status.

## Compute Claims

The comparison is matched by optimizer tokens and global steps. Every report
sets `matched_compute_claim=false` unless a separate, explicit compute budget
contract includes measured wall time and all relevant work. The additional
optimizer state cost remains visible through wall time, memory, and checkpoint
size.

## Per-Run Artifact Contract

Every completed run provides:

- resolved configuration and paired/full identity;
- ordinary metrics with selected width, owner scope, global learning rate,
  scheduler position, update counts, exposures, wall time, and peak memory;
- run summary with reconciled final counts and resource values;
- resumable terminal checkpoint with explicit purpose and scope;
- fixed data-role manifests and hashes;
- post-training final-holdout result only when explicitly requested.

## Failure Behavior

- Freeze and report are read-only with respect to run directories.
- Invalid or incomplete inputs fail without emitting a success manifest/report.
- Atomic output replacement prevents a partial manifest from being mistaken for
  a frozen protocol.
- `--skip-existing` final-holdout behavior is accepted only when the existing
  result validates against the exact checkpoint and final-role hashes.
