# Research: TinyStories S1 Fourfold LR Warmup

All design unknowns are resolved. These findings establish planning inputs, not new execution readiness or complete historical-terminal certification.

## 1. Protocol and grid identity

**Decision:** Add schema 5 with two declared arms, `S1-linear-w256` and `S1-geometric-w256`. Require arm identity for new width/common/topology selectors; add new-only grid/intervention/reference fields to contracts. Retain old serialized forms exactly.

**Rationale:** `src/evaluation/optimizer_ownership.py` currently selects one grid per schema. `validate_run_budget` requires 64 warmup updates; `validate_materialized_config` explicitly accepts only schema 4 when a schema marker exists. `src/utils/config.py` also has arm output-path eligibility and `_validate_matformer_campaign_topology` guards. `src/utils/metrics.py` and `src/training/run.py` resolve widths by schema. All these paths need explicit schema-5/arm dispatch. Warmup 256 is accepted only after validating the fixed protocol; `PINNED_COMMON` stays unchanged.

**Alternatives considered:** Globally relaxing warmup validation undermines old protocols; relabeling new runs as old campaigns collides with identities; two separate campaign frameworks obscure the exact two-run invariant.

## 2. Scientific controls and random streams

**Decision:** Expand fixed controls through the ordinary resolver, then compare to each saved counterpart using a closed path-level difference list. Preserve ordinary fresh initialization and existing random primitives. New operational names never seed randomness.

**Rationale:** `src/utils/reproducibility.py::derive_seed/seed_for` uses root seed, stream name and version, without run ID. Uniform action generation uses isolated `random.Random(...).randrange(4)`; replacing it with equally weighted `choices` changes the trace. Packed batches use the saved initial permutation, then `deterministic_epoch_positions` with data seed 42, `SeedSequence([42, epoch_index, 1])` and PCG64. They do not use a generic loader seed. `build_expected_traces` currently requires all elastic trace dictionaries equal; schema 5 must compare each action stream to its own counterpart and verify batch streams separately.

Saved S1 YAML hashes still matched the spec during planning on 2026-09-22:

| Grid | `campaign/configs/S1.yaml` SHA-256 |
| --- | --- |
| Linear | `ef16f8a6cb883abe6782b9201f32b6d137b324f4d0cd5ac112f66b6fb57ac3fa` |
| Geometric | `781254c7bbd8a7fdc48ba4c09bdc940dd44e132c73d1fe9d8b9209796d94a6c8` |

Corpus: `/nfs-stor/ivo.navarrete/matformer-corpora/tinystories-instruct-packed-full-v1`. Tokenizer: `/nfs-stor/ivo.navarrete/matformer-tokenizers/tinystories-instruct-sentencepiece-bpe-2k-v1`. A new audit must verify these inputs and all spec hashes.

**Alternatives considered:** Loading trained weights violates fresh initialization; run-ID seeding breaks pairing; comparing cross-grid labeled action hashes is incorrect; balancing width counts changes replacement sampling.

## 3. Scheduler indexing

**Decision:** Keep the installed Transformers cosine scheduler and optimizer-then-scheduler order. Export positions 0–348,528, with explicit mapping to one-based applied updates. Verify expected and actual LR at both boundaries and the terminal horizon.

**Rationale:** `src/training/steps.py::build_optimizer_and_scheduler`, LR capture before optimizer execution, and its subsequent `scheduler.step()` establish the convention. `src/training/schedules.py` and installed Transformers `optimization.py` confirm it. Position zero has LR zero; update u applies position u−1, then checkpoint position becomes u. Update 256 applies .00796875; update 257 first applies peak .008. Final stored LR at position 348,528 is zero; last applied LR is approximately 1.6273915548481455e-13. Both new runs have identical full schedules. See [precise formula](contracts/protocol-and-schedule.md).

**Alternatives considered:** A new scheduler or changed step order adds a scientific intervention; extending the horizon violates the budget; calling post-step LR the applied value hides an off-by-one difference.

## 4. Continuation and failed updates

**Decision:** Reuse `_ownership_identity`, `_validate_ownership_payload`, `_validate_ownership_action_rng`, `_load_ownership_checkpoint` and `assert_checkpoint_safe` in `src/training/checkpointing.py`. Bind new protocol metadata through the existing full contract. Preserve complete-update durability and terminal-only recovery.

**Rationale:** Existing validation covers model, moments, counters, global scheduler/rates, RNG/action state, packed cursor and accounting before live installation. Partial optimizer/scheduler/accounting failure poisons the live bundle so it cannot overwrite the durable checkpoint. Reuse concrete inherited numerical comparison helpers (existing float-state checks include rtol 1e-6/atol 1e-7); record any inherited dtype-specific tolerance, never loosen it merely to pass. Actions, batches, counters and scheduler positions remain exact. State-seeded probes near real epoch boundaries are valid diagnostics when their synthetic history is disclosed; they are not full-budget evidence.

**Alternatives considered:** Model-only restart loses optimizer/stream state; per-update full-state rollback is unnecessary overhead; shortened schedules alone fail to verify real boundary indexing.

## 5. Selected historical references and early curves

**Decision:** Use `inspect_selected_terminals` and applicable read-only device checks for five selected runs per grid. Write new selection manifests only under the new root. Keep old artifacts immutable.

**Rationale:** The strict reader delegates to `_inspect_terminal_run`, `_terminal_endpoints` and trace validation. The full `report_matformer_widths_comparison` requires all nine geometric runs, so it cannot directly serve the closed partial Feature 015 campaign. `scripts/plot_tinystories_standalones.py` already validates selected geometric elastic execution using job/worker/CUDA-entry/resource records. Do not use monitoring helpers that can issue `scancel` as report validators.

Geometric S1 metadata identifies valid attempt 2, job 272716, separate from invalid archived CPU attempt 1. Linear S1 predates modern CUDA-entry files; inspect its applicable saved job/launcher evidence, resolved bf16 and measured CUDA resource records. Missing a newer-format file alone must not reject an older valid run, and a summary claiming completion alone is insufficient.

Each selected historical S1 `metrics.csv` has recorded training LR and loss at updates 1–1,024 and 16 ordinary-validation observations per width through update 1,024. Update 1 records LR zero. There is no measured step-zero loss. These bounded reads establish availability only; full terminal/checkpoint/trace and scalar provenance validation remains future work.

**Alternatives considered:** Require cancelled arms to finish (out of scope); trust stale status summaries (admits invalid attempts); reconstruct losses or retrain references (not permitted); invent a measured step-zero observation (false evidence).

## 6. Operational reuse and evidence

**Decision:** Use a focused two-arm launcher with existing lock, reservation, intent reconciliation, snapshot, continuation and resource patterns. Require explicit CUDA bf16 entry and matching job/worker success for new terminal completion.

**Rationale:** `scripts/run_tinystories_matformer_widths.py` hardcodes schema 4, nine arms and a standalone barrier. Reuse pure helpers, not that admission policy. Its sidecar-based completion alone is insufficient for this spec. `scripts/train_cuda_required.py` enforces sbatch, allowed host, one visible CUDA device and bf16; runtime evidence must also establish actual execution. Freeze the exact launcher/trainer revision before gating. Preserve uncertain submissions until reconciled instead of duplicating them.

Proposed root `/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-s1-warmup-v1` was absent at inspection. No external directory, reservation, job, or readiness gate was created.

**Alternatives considered:** Reusing nine-arm admission launches wrong work; trainer CPU fallback repeats a known failure; a generic orchestration framework adds unnecessary indirection; sidecar or exit status alone lacks required evidence.

## 7. Reports and dependencies

**Decision:** Extend the current analyzer with a focused warmup report and keep installed dependencies. Reuse `_endpoint_table`, `endpoint_csv_value`, `_source_record`, `_check_sources` and `_publish_directory` where their assumptions fit. Produce eight new endpoints independently and a strict 24-endpoint comparison with eight paired deltas.

**Rationale:** Pair identity must include grid and physical width so shared dimensions never merge distinct baselines. Early data is available and should remain raw by default. Reconstructed LR is a separately labeled fallback from validated controls; loss is never reconstructed. Matplotlib already supports required PNG/PDF figures.

**Alternatives considered:** Deduplicating by parameter count destroys provenance; best/trailing losses are not terminal endpoints; new plotting/orchestration packages provide no needed capability.
