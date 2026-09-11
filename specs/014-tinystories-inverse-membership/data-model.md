# Feature 014 data model

- Campaign schema 3: fresh campaign_id, exactly five ordered IM arm definitions,
  unchanged pinned common/data controls, expected per-arm action/epoch traces,
  source/config hashes, run contract hashes. State: defined → preflight passed →
  CPU/GPU validated → submitted/running → terminal → frozen → reported.
- Run: original representation/history/clipping fields, IM arm_id,
  reference_arm_id, sampling_policy=fixed_inverse_membership, seed 42, four-epoch
  budget, unique output path. Continuation retains this identity.
- Sampling contract: version, ordered widths, membership_counts=[4,3,2,1],
  probabilities=[.12,.16,.24,.48], global scope, with replacement, cadence one,
  action seed and no loss weighting. Stored in full scientific contract.
- Compact selection state: committed total/exposure/token counts, held width and
  one-update window cursor, plus fixed policy and ordered probability metadata.
  Selection does not commit counts; successful complete owner update does.
- Checkpoint: unchanged campaign complete-update schema and full contract; compact
  state plus dedicated RNG state must replay to the committed ordinal. Invalid
  state is rejected before installation. Partial owner updates are poisoned.
- Submission attempt: atomic intent/name/id, arm and attempt, snapshot/config hashes,
  queue/accounting and writer identity. An ambiguous intent is reconciled before
  another submission. Resource attempts preserve consumed work beyond rollback.
- Endpoint: run/arm/width, policy, reference status, exact count, loss/perplexity,
  ordinary-validation target count/identity, terminal checkpoint hash and budgets,
  exposure/clipping/storage/resource observations. Unique 20 new or 44 combined.
- Comparison: separately validated new and historical source sets, 20 paired
  deltas, full progress coverage, expectation versus observed exposure, figures;
  historical failure leaves new report valid and comparison explicitly outstanding.
