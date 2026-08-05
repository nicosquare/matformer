# Quickstart: Probabilistic Adaptive Granularity

Commands below describe the expected workflow after implementation. Bayesian
fixtures are opt-in and do not expand the default pilot queue.

## 1. Inspect Bayesian global resolution

```bash
python train.py \
  --config tests/fixtures/probabilistic_adaptive_global_smoke.yaml \
  --preflight
```

Confirm preflight reports:

- `granularity_sampling_mode=adaptive_global`;
- `adaptive_sampler_strategy=thompson`;
- resolved scope `global` and Bayesian method version;
- ordered resolved granularities and global feature dimension;
- explicit prior, observation-noise, and process-noise inputs;
- a positive decision interval, defaulting to 50 only when omitted;
- fixed 128-example controller and 512-example final-holdout contracts;
- zero compute and switching costs.

## 2. Verify migration failure and unchanged UCB

An old-shaped Thompson configuration must not run under guessed defaults:

```bash
python train.py \
  --config tests/fixtures/legacy_thompson_config.yaml \
  --preflight
```

Expected result: a migration-specific error listing missing Bayesian controller
or data-role inputs.

The explicit UCB regression remains valid:

```bash
pytest tests/test_config.py tests/test_adaptive_sampler.py -k ucb
```

Expected result: existing UCB config resolution, scores, reward updates,
checkpoint fields, and reporting labels remain unchanged.

## 3. Validate Gaussian mathematics and feature schemas

```bash
pytest tests/test_probabilistic_controller.py
```

Coverage should include:

- identity prediction with zero and positive process covariance;
- exact Gaussian mean/covariance conditioning from controlled rewards;
- finite, symmetry, dimension, and positive-semidefinite validation;
- global contrast arms and additive per-block contrasts;
- arbitrary labels/counts, one granularity, and one transformer block;
- deterministic ties and dedicated posterior-sampling state;
- per-block preference learning without duplicated scalar reward means.

## 4. Validate four disjoint data roles

```bash
pytest tests/test_data_validation.py -k "controller or final_holdout or four_role"
```

Confirm artifacts prove:

- exactly 128 controller examples;
- exactly 512 final examples;
- the requested ordinary-validation count;
- at least one usable optimizer-training example;
- six empty pairwise intersections;
- deterministic source identities, seeds, role hashes, and parent split hash;
- attributable failure when the source pool is too small.

## 5. Run a short Bayesian global smoke

Use a small fixture whose decision interval is two optimizer steps:

```bash
python train.py \
  --config tests/fixtures/probabilistic_adaptive_global_smoke.yaml \
  --override model.adaptive_controller.decision_interval_steps=2 \
  --override training.max_steps=4 \
  --override run.run_id=bayesian-global-smoke
```

Inspect:

```bash
python -m json.tool outputs/bayesian-global-smoke/controller_manifest.json
python -m json.tool outputs/bayesian-global-smoke/final_holdout_manifest.json
python -m json.tool outputs/bayesian-global-smoke/controller_summary.json
sed -n '1,10p' outputs/bayesian-global-smoke/controller_metrics.jsonl
```

Expected boundary sequence:

- one initial controller evaluation and no initial reward;
- one unchanged global action for each two-step window;
- one post-window evaluation reused as the next baseline;
- one committed posterior observation per completed window;
- Bayesian method/scope and role hashes in config, metrics, checkpoints, and
  run summary.

## 6. Validate exact resume behavior

```bash
pytest tests/test_probabilistic_controller_resume.py -k "inside_window or exact_boundary or incomplete"
```

On the same software, hardware, distributed topology, and deterministic runtime
settings, matched fresh/resumed cases must reproduce manifests, window state,
posterior-sampling state, sample count, and subsequent actions exactly.
Completed-window objectives, rewards, posterior means, and posterior
covariances must match with `rtol=1e-6` and `atol=1e-8`. A terminal short window
must record `observation_emitted=false`.

Also validate migration and mismatch failures:

```bash
pytest tests/test_probabilistic_controller_resume.py -k "legacy or manifest_mismatch or incompatible"
```

## 7. Run additive per-block smoke

```bash
python train.py \
  --config tests/fixtures/probabilistic_adaptive_per_block_smoke.yaml \
  --override model.adaptive_controller.decision_interval_steps=2 \
  --override training.max_steps=6 \
  --override run.run_id=bayesian-per-block-smoke
```

Confirm the saved feature schema has dimension `1 + B(|G|-1)`, complete
profiles remain fixed within windows, and posterior summaries can distinguish
at least two block/granularity effects without complete-profile enumeration.

## 8. Check distributed controller agreement

```bash
pytest tests/test_distributed.py -k probabilistic_controller
```

Expected result: controller loss is reduced across fixed nonduplicated
partitions, rank zero owns posterior sampling/update, and every rank receives
the same action and controller state.

## 9. Verify reporting provenance

```bash
pytest tests/test_reporting.py tests/test_phase2_finalize.py -k "probabilistic or legacy_thompson or ucb"
```

Expected classifications:

- new global: `probabilistic_global_thompson`;
- new per-block: `probabilistic_per_block_thompson`;
- historical pseudo-Thompson without Bayesian provenance:
  `adaptive_per_block_thompson` displayed as legacy heuristic Thompson;
- UCB: existing `adaptive_per_block_ucb` identity and style unchanged.

## 10. Evaluate the final holdout after training

Do not run this command until the training run is complete and checkpoint
selection has finished:

```bash
python scripts/evaluate_final_holdout.py \
  --run-dir outputs/bayesian-global-smoke
```

If the run has no ordinary-validation-selected checkpoint, provide one
explicitly:

```bash
python scripts/evaluate_final_holdout.py \
  --run-dir outputs/bayesian-global-smoke \
  --checkpoint outputs/bayesian-global-smoke/checkpoints/latest.pt
```

Inspect the separate result:

```bash
python -m json.tool outputs/bayesian-global-smoke/final_holdout_results.json
```

The result must record the fixed final manifest hash, checkpoint-selection
provenance, ordered per-granularity target-token-weighted losses, and uniform
average. It must not modify controller observations or checkpoint selection.

## 11. Run the compatibility matrix

```bash
pytest \
  tests/test_adaptive_sampler.py \
  tests/test_config.py \
  tests/test_artifacts.py \
  tests/test_reporting.py \
  tests/test_training_smoke.py \
  tests/test_pilot_comparison.py
```

Random global, random per-block, UCB, nested-all, standalone, correction,
nonadaptive checkpointing, and the default pilot queue must remain unchanged.
