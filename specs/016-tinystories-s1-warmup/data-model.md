# Data Model: S1 Warmup Extension

Records use existing YAML/JSON/JSONL/CSV/checkpoint conventions. New schemas extend explicit records, without a database or registry.

## Extension and run protocol

**Extension** fields: `schema_version=5`, campaign ID, exact two-arm list, reference mapping, inherited `expected_data`, run root, source/config hashes, per-run expected traces, readiness/report states. Primary identity is campaign ID plus immutable manifest hash. A campaign stores grids per arm, never one ambiguous global grid.

**RunProtocol** fields: campaign/run/arm IDs, `grid_id` (`linear` or `geometric`), role S1, seed 42, ordered widths/fractions/counts, slicing representation, shared history, global L2 clip 1, correction none, uniform replacement H=1, model/data/optimizer/evaluation controls, initialization/action/data stream identities, warmup 256, peak .008, total updates 348528, four epochs, 2855141376 tokens, reference S1 identity, scientific contract/hash and output path.

| Grid | Arm | FFN dimensions | Exact active non-embedding counts |
| --- | --- | --- | --- |
| Linear | S1-linear-w256 | 64, 128, 192, 256 | 115264, 164416, 213568, 262720 |
| Geometric | S1-geometric-w256 | 32, 64, 128, 256 | 90688, 115264, 164416, 262720 |

Each run belongs to one extension and references exactly one historical S1. Counts exclude input embedding and LM head and must be measured against real constructed models. Full tensor storage is distinct from active count. Validate exactly the prescribed arms, streams, 256 warmup, budget and width support; reject occupied/incompatible identities. Scientific contract schema/serializer remains the existing one; new fields are hashed only for new inputs.

## Control comparison and schedule evidence

**ControlDifferenceAudit** contains old/new config paths and hashes, resolved control projections, explicit allowed changed paths, actual differences, old/new contract identities, result and reasons. The allowable scientific difference is warmup and scheduler values mathematically implied by warmup. Identity/output/source/reference metadata is separately enumerated. Unknown differences fail.

**ScheduleEvidence** contains run/grid, scheduler family, peak, warmup, horizon, indexing convention, expected full-array hash, installed dependency/source identity, and boundary observations. The expected table has 348529 rows for positions 0..348528. Position p is applied by update p+1 only when p<T; terminal p=T is an endpoint with no subsequent applied update. Observed rows link applied LR to committed update, stored position and checkpoint/metric source.

**TraceEvidence** contains action seed/primitive, ordered support, ordinal/count/digest, four epoch batch digests/cursors, exclusions, and corresponding historical digests. All 348528 actions and four epochs must reconcile per counterpart. Expected selections 87132 per width are statistical expectations, not enforced quotas.

## Historical reference selection

**HistoricalReference** fields: grid, role (`standalone` or `s1_warmup64`), original campaign/arm/run, source manifest/config/hash, contract, own budget, physical widths/counts, terminal checkpoint/hash/step, ordinary-validation identity/target count, trace evidence, applicable execution provenance and selection status.

Exactly ten distinct selected terminals contribute 16 historical endpoints: eight standalone checkpoints yield one each, two S1 checkpoints yield four each. Shared physical sizes retain separate grid/campaign records. Selection states are `unvalidated`, `valid`, `missing`, `invalid`; validation reasons and source hashes are explicit. Historical files are immutable; a stale selection is revalidated, never silently trusted.

## Checkpoint, attempt and readiness

**ContinuationCheckpoint** reuses existing complete ownership bundle: model tensors, shared AdamW state/counters, scheduler/rates, RNG/action state, packed sampler cursor/epochs, compact accounting, protocol/run identity and durable step. No model-only or cross-run checkpoint can satisfy this entity. Step must be in 0..348528; scheduler position, tokens (8192 per committed update), action count and cursor agree. Non-finite/malformed/incompatible state is rejected before installation.

**ReadinessEvidence** fields: CPU/GPU mode, exact source manifest including executed entry points, recipe/config-set/protocol hashes, reference/control/trace audit hashes, command/log/artifact hashes, test counts/exit status, all-width/boundary probe coverage, CPU gate dependency, and GPU job/device/allocation/precision evidence when applicable. Status is `pending`, `passed`, `failed`, or `stale`. A skipped mandatory GPU check cannot produce `passed`.

**LaunchAttempt** fields: run, monotonic attempt ID, durable submission intent/name, Slurm ID, process UUID, gate/source/config hashes, continuation checkpoint/hash, timestamps, scheduler/worker result, device/bf16 evidence, resource observations and completeness. Attempts link many-to-one to a run; only one writer is admitted. Committed progress excludes failed/replayed work; resource cost includes it where measured and explicitly discloses gaps.

Transitions: reserved → prepared → eligible → submitting → submitted/running → terminal-validated. An uncertain submission blocks another submission until reconciliation. A failed/interrupted attempt may become retryable only after validating its own durable checkpoint. Full-horizon checkpoints without outputs enter completion-only recovery; they execute no further training update. Readiness invalidation removes eligibility. Terminal status requires checkpoint, ordinary evaluation and successful execution evidence, not sidecar presence alone.

## Endpoints, deltas and trajectories

**Endpoint** key: `(grid_id, campaign_id, run_id, ffn_dimension)`. Fields: role, seed, warmup, width label/fraction/dimension, exact active count, assigned/actual epochs/updates/tokens, terminal step, target-weighted loss, exp(loss), evaluated target count, ordinary-validation protocol/membership, checkpoint/evaluation/source hashes, actual/expected exposure and resources. Require finite values and full terminal budgets (standalone 87132; S1 348528).

**PairedDelta** key: `(grid_id, ffn_dimension)`. Links one warmup-256, one warmup-64 and one matching standalone endpoint. Fields include new-minus-old loss/perplexity and both S1-minus-standalone gaps. Exactly eight rows. Negative delta means improvement; positive worsening; zero no change.

**EarlyObservation** key includes grid/run/attempt, split, physical width where applicable, committed absolute update and metric. Fields include loss/LR, applied-versus-stored schedule position, source file/hash/row, measured/reconstructed kind, gap and smoothing annotations. Resolve replay rows against committed attempt evidence; ambiguous/conflicting duplicates fail. Retain raw values. Training minibatch loss is distinct from per-width validation loss. No step-zero measured loss is fabricated.

**ComparisonEvidence** links all endpoints, deltas, early data, figure paths/hashes and findings. Exactly 24 endpoints, 12 per grid, are required for complete comparison. Independent states record planning, implementation checks, GPU readiness, new terminals, endpoint comparison and early/report deliverables. Valid eight-endpoint new output survives missing references; missing required early evidence prevents overall report completion even if endpoint tables are complete.
