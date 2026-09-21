# Ordered execution

1. Use `/home/ivo.navarrete/.conda/envs/elasticnn/bin/python` for focused tests:
   `-m pytest tests/test_inverse_membership_sampling.py tests/test_inverse_membership_reporting.py tests/test_inverse_membership_queue.py`.
   Run existing ownership/correction/config/resume/sampling compatibility suites.
2. Select an unused campaign root. Validate historical frozen references, inspect
   current scheduler limits and queue, audit immutable corpus/tokenizer inputs.
3. Run `scripts/analyze_tinystories_optimizer_ownership.py preflight` with
   `--campaign configs/controlled_exps/tinystories_instruct_inverse_membership.yaml`,
   `--prepared-corpus-dir /nfs-stor/ivo.navarrete/matformer-corpora/tinystories-instruct-packed-full-v1`,
   `--tokenizer-dir /nfs-stor/ivo.navarrete/matformer-tokenizers/tinystories-instruct-sentencepiece-bpe-2k-v1`,
   and separate campaign-root `campaign`/`runs` paths for output-dir/run-output-root.
4. Snapshot validated sources under campaign root; bind CPU evidence, preflight
   and all configs to their hashes. Run `scripts/preflight_tinystories_inverse_membership.py`
   through sbatch at real shape/batch/data with fresh diagnostic identities.
   Pass all five finite/action/ownership/resume/throughput checks before production.
5. Use `scripts/run_tinystories_inverse_membership.py` with the recorded campaign
   plan to submit/monitor five arms, respecting live global limits. Preserve
   diagnostics, continuations and all costs. No separate launch permission needed.
6. Use analyzer `freeze` then `report` for the five full terminals (20 endpoints).
   Use `report-inverse-membership` with the validated historical frozen manifest
   for 44 endpoints and comparisons. Report unavailable history as outstanding.
7. Record actual commands, job IDs, gate results, artifacts and interpretation in
   verification/runbook. Never report fixture or short diagnostic results as
   completed production. Never evaluate the sealed holdout.
