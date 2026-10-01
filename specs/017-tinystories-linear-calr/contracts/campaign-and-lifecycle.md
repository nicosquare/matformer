# Campaign and Lifecycle Contract

## Configuration admission

Campaign schema 6 admits exactly four seed-42 slicing runs, linear widths, scopes and exponent policies declared in spec.md. Reject any extra arm/seed, changed width, correction or control. Match original saved S1/S2 at .008/64. Audit three independent relations: counterpart (only schedule and fresh identity/provenance), schedule pair (only exponent policy/consequences), ownership pair (only state scope/consequences). Implement enumerated resolved-field allowlists, including explicit hashes/derived fields, rather than wildcard exclusions. Verify same fresh weights/action/data streams independently of arm names.

Validate actual-model complexity plus inherited tokenizer/corpus/manifests, controls and budgets before readiness. Historical schemas/config hashes and S1/S2/C4 restores stay unchanged.

## Proposed CLI

`preflight_tinystories_linear_calr.py` modes: `snapshot --campaign-root ROOT`, `cpu --campaign-root ROOT`, `gpu --campaign-root ROOT`. Snapshot creates immutable tested source/config identity; CPU produces bound evidence; gpu is a CUDA-required diagnostic worker invoked only by a later authorized sbatch submission. Implement documented arguments consistently; commands are proposed until implementation.

`run_tinystories_linear_calr.py` modes: `prepare --campaign-root ROOT --cpu-evidence FILE`, `queue --campaign-root ROOT [--once]`, `worker --campaign-root ROOT --arm ARM --attempt-id ID`, `report --campaign-root ROOT`. Execute preparation/queue/worker/report from tested snapshot. CPU preparation never submits GPU work. Queue requires a durable record of explicit subsequent execution authorization plus valid CPU/GPU gates. No command flag or passing test substitutes for user authorization.

`analyze_tinystories_optimizer_ownership.py` gains `report-linear-calr` with explicit new root and selected reference mappings, preserving existing operations. Workers use existing train_cuda_required entry, not an alternate trainer.

## Operations

Suggested fresh root is `/nfs-stor/ivo.navarrete/results/elasticnn/tinystories-linear-s1-s2-calr-v1`; atomically reserve before writes and reject conflicting identities. Planning discovered absence only. Read-only reference access never writes configuration, manifests, metrics or checkpoints.

Reuse lock/submission-intent/attempt helpers, reconcile uncertain sbatch outcomes before retry, and prevent duplicate run submissions. sbatch uses cscc-gpu-p / cscc-gpu-qos, one GPU, no requeue, exclusions gpu-[05,50,51,54]. Query all user jobs and stricter live partition/QoS constraints: at most two running GPU/four submitted jobs. Resume only own latest valid durable checkpoint and source/config contract. Preserve failed attempts and readiness records.

GPU readiness must cover all four arms at d64/l4/h4, batch 64/context 128, all widths and actual BF16, update/restore boundaries, resource evidence and source/config bindings matching CPU gates. CPU-only and shortened-horizon results are not production readiness. Readiness and production authorization remain separate.

Terminal acceptance: each run exactly 348528 updates/2855141376 training tokens, valid full-budget checkpoint, four ordinary-validation endpoints and reconciled measured traces/resources; campaign totals exactly 1394112/11420565504. Validation and failed-attempt costs are separate. Successful job status alone cannot establish a valid terminal. Missing reference evidence does not erase valid new terminals, but prevents full comparison completion.
