# Quickstart: Probabilistic Adaptive Granularity

Commands below describe the expected workflow after implementation. Bayesian
fixtures are opt-in and do not expand the default pilot queue.

Commands that load a tokenizer or dataset require Hugging Face network access
or an already populated, writable cache. Use the controlled workflow in
section 12 when validating in an offline or restricted environment.

## 1. Inspect Bayesian global resolution

```bash
python train.py \
  --config tests/fixtures/probabilistic_adaptive_global_smoke.yaml \
  --preflight
```

The preflight output summarizes reproducibility and comparison-control inputs.
To inspect the complete normalized Bayesian method mapping without loading a
dataset, run:

```bash
python -c '
import json
from src.utils.config import resolve_run_config

config = resolve_run_config(
    "tests/fixtures/probabilistic_adaptive_global_smoke.yaml"
)
print(json.dumps({
    "granularity_sampling_mode": config["model"]["granularity_sampling_mode"],
    "adaptive_sampler_strategy": config["model"]["adaptive_sampler_strategy"],
    "adaptive_controller": config["model"]["adaptive_controller"],
}, indent=2, sort_keys=True))
'
```

Confirm the resolved mapping contains:

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

Expected result for this fixture: a migration-specific error identifying the
legacy adaptive-sampler fields that cannot be mixed with the Bayesian
controller. Other incomplete Thompson configurations report their missing
Bayesian controller or data-role inputs.

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
  --override run.run_id=bayesian-global-smoke \
  --output-dir outputs/bayesian-global-smoke
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
  --override run.run_id=bayesian-per-block-smoke \
  --output-dir outputs/bayesian-per-block-smoke
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
provenance, ordered per-granularity target-token-weighted losses, uniform
average, independent result hash, and the verified run-summary,
controller-summary, and controller-journal hashes. It must not modify
controller observations or checkpoint selection.

## 11. Run the compatibility matrix

```bash
pytest \
  tests/test_adaptive_sampler.py \
  tests/test_config.py \
  tests/test_artifacts.py \
  tests/test_reporting.py \
  tests/test_training_smoke.py \
  tests/test_matformer_prefixes.py \
  tests/test_baseline_matching.py \
  tests/test_pilot_comparison.py
```

Random global, random per-block, UCB, nested-all, standalone, correction,
nonadaptive checkpointing, and the default pilot queue must remain unchanged.

## 12. Run the controlled end-to-end workflow

When live Hugging Face data access is unavailable, run the deterministic
workflow coverage that exercises global boundaries, exact resume, additive
per-block profiles, provenance classification, and post-training final-holdout
evaluation without external downloads:

```bash
pytest -q \
  tests/test_training_smoke.py::test_probabilistic_adaptive_global_boundary_reward_action_and_logs \
  tests/test_probabilistic_controller_resume.py::test_fresh_and_resumed_controller_match_from_inside_window_and_exact_boundary \
  tests/test_training_smoke.py::test_probabilistic_adaptive_per_block_uses_fixed_profiles_and_shared_rewards \
  tests/test_artifacts.py::test_probabilistic_artifacts_preserve_end_to_end_controller_provenance \
  tests/test_reporting.py::test_reporting_uses_explicit_provenance_to_distinguish_bayesian_and_legacy_thompson \
  tests/test_phase2_finalize.py::test_final_holdout_requires_completed_run_and_resolves_checkpoint_fallback \
  tests/test_phase2_finalize.py::test_final_holdout_rejects_manifest_and_checkpoint_provenance_mismatches \
  tests/test_phase2_finalize.py::test_final_holdout_evaluates_all_granularities_deterministically_and_is_non_mutating
```

This controlled workflow verifies method behavior and artifact boundaries. It
does not replace a live dataset-backed smoke run when network/cache access is
available.

## 13. Generate controller granularity timelines

The existing figure command discovers Bayesian controller journals
automatically:

```bash
python scripts/make_figures.py --input outputs --output outputs/figures
```

Each valid Bayesian global or per-block run with at least one confirmed window
adds:

```text
selected_granularity_over_tokens_<run-id>.png
selected_granularity_share_over_tokens_<run-id>.png
```

The original heatmap retains one row for a global controller's `all blocks`
action or 1-based transformer-block rows for a per-block controller. The new
figure instead gives every saved granularity its own panel. Its vertical axis
is the fraction of transformer blocks directly assigned to that granularity
in each confirmed controller window, so global actions are binary 0/1 traces
and per-block actions can lie anywhere from 0 to 1. These are realized
selection shares, not posterior probabilities.

Both views use planned tokens clipped to the configured token budget and
follow the saved `ordered_granularities`. Completed windows and explicit
terminal partial windows are shown, while warmup and uncommitted active-window
regions remain blank. A malformed run emits a warning and does not prevent
figures for other runs.

## 14. Run the balanced 500-step pre-adaptive reference

Resolve the complete five-granularity reference without loading data:

```bash
python train.py \
  --config configs/opt-in_exps/probabilistic_balanced_warmup_500.yaml \
  --preflight
```

The resolved warmup has ten 50-step windows. Each granularity appears exactly
twice, while its position in each complete pass is determined by the
independent `pre_nested_warmup_schedule` seed. During steps 1–500 the fixed
controller panel is not evaluated and the posterior remains the configured
prior. At step 500, the initial fixed-panel baseline and first Thompson action
are recorded.

For the controlled lifecycle, interruption, and terminal-budget checks:

```bash
pytest -q \
  tests/test_training_smoke.py::test_balanced_global_warmup_precedes_controller_for_both_adaptive_scopes \
  tests/test_training_smoke.py::test_balanced_global_warmup_resume_inside_window_matches_uninterrupted_run \
  tests/test_training_smoke.py::test_balanced_global_warmup_terminal_budget_exhaustion_never_starts_controller
```

## 15. Run the opt-in episodic-reset comparison

The matched seed-42 comparison manifest is
`configs/opt-in_exps/probabilistic_global_reset_seed42.yaml`. It records the common base
config and dotted overrides for Q=0 with no reset, then reset K=500, K=1000,
and K=2000. All variants retain h=50, the 500-step balanced warmup, prior,
observation variance, role manifests, optimizer, scheduler, and training
budget. It is not part of the default pilot queue.

Validate reset behavior and continuation without external data access:

```bash
pytest -q \
  tests/test_config.py -k probabilistic_reset \
  tests/test_reproducibility.py -k controller_reset \
  tests/test_probabilistic_controller.py -k reset \
  tests/test_probabilistic_controller_resume.py -k reset \
  tests/test_artifacts.py -k 'same_boundary or separates_forced' \
  tests/test_reporting.py -k explicit_provenance
```

To run the same episodic acquisition schedule without discarding the learned
posterior, retain the reset-enabled interval contract and override only the
policy:

```bash
python train.py \
  --config tests/fixtures/probabilistic_adaptive_global_smoke.yaml \
  --override model.adaptive_controller.process_noise_covariance=0.0 \
  --override model.adaptive_controller.reset.enabled=true \
  --override model.adaptive_controller.reset.interval_steps=12 \
  --override model.adaptive_controller.reset.policy=acquisition_only \
  --preflight
```

This variant is reported as
`probabilistic_global_thompson_acquisition_only`; `reset_count` remains zero
because no posterior restoration occurs.

## 16. Compare Bayesian global TS variants in size plots

The existing figure command keeps the persisted reporting identities above,
but `loss_vs_size` and `ppl_vs_size` place all Bayesian global Thompson
variants in one panel. The panel title carries the shared method context; each
legend entry contains only the contract fields needed to distinguish its
curve, such as `No reset · Q=1e−10`, `Full-prior · K=2k`, or
`Acquisition-only · K=2k`.

Rows with an identical seed-independent experiment contract are aggregated at
each model size. The line is the seed mean, the translucent envelope is the
seed minimum–maximum range, and `n=<count> seeds` is added to the legend when
more than one seed contributes. A historical artifact that lacks the metadata
needed to prove contract equivalence remains isolated by run identity instead
of being silently merged. Empty panels are omitted, and all displayed panels
retain a common y-axis range.

## 17. Inspect contract-safe validation and saturation diagnostics

Generate the default completed-run-only validation figures and global-action
diagnostics with the existing command:

```bash
python scripts/make_figures.py --input outputs --output outputs/figures
```

`run_summary.json` must report `status: completed` for a run to enter a
scientific validation aggregate. Matching seeds are averaged only within the
same seed-independent saved-config contract at identical total-training-token
checkpoints; the translucent envelope is the per-checkpoint seed minimum and
maximum. Historical artifacts without enough saved configuration to prove
equivalence remain isolated by run identity. The per-granularity panels follow
the configured micro-to-full order, share y limits, and label the horizontal
axis `Total training tokens`.

To inspect unfinished artifacts for debugging, opt in explicitly:

```bash
python scripts/make_figures.py \
  --input outputs \
  --output outputs/figures-with-incomplete \
  --include-incomplete-validation-traces
```

Each unfinished or summary-missing run is then a separate dashed curve labeled
with run ID, status, and token progress. It never enters a completed seed
aggregate.

For each completed `global` or Bayesian `adaptive_global` method, figure
generation also writes:

```text
validation_loss_over_selected_exposure_<method>.png
validation_marginal_utility_over_tokens_<method>.png
validation_marginal_utility_ranking.md
```

Direct selected exposure is reconstructed from successive planned-token
increments in streamed training rows, including balanced warmup actions.
Repeated resume rows are deduplicated; a conflicting or non-monotonic history
is skipped with a warning. Loss-versus-exposure seed curves are interpolated
on a deterministic 100-point grid only over common exposure support.

Marginal utility is the negative OLS slope of ordinary-validation loss against
direct selected exposure in millions of tokens, using the latest five distinct
exposure observations and requiring at least three. Negative values remain in
the figures and ranking as degradation. Treat the ranking as evidence about
saturation, not as a binary saturation decision or controller action. Direct
exposure records the selected global action; MatFormer parameter sharing can
still improve one granularity while another action is selected.

## 18. Filter generated figures by model variant and correction

To produce a directory containing only uncorrected slicing results, use:

```bash
python scripts/make_figures.py \
  --input outputs \
  --output outputs/figures-slicing-uncorrected \
  --variant slicing \
  --correction none
```

`--variant` accepts `slicing` or `concat`, and `--correction` accepts `none`,
`gmc`, or `lmc`. Each option is repeatable to form an inclusive set; omitting
an option leaves that dimension unfiltered. The filters apply consistently to
size, validation, saturation, consistency, and both controller-policy views.
Figures with no matching rows are omitted.
