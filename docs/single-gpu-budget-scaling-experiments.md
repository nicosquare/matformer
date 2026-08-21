# Variable-budget single-GPU slicing experiments

This guide reproduces the prepared-corpus slicing comparison at a nominal
`XXXM` token budget selected through `BUDGET_M`. It is intended to answer two
questions without changing the one-GPU optimizer geometry:

1. Does holding a uniformly sampled global granularity for several updates
   remain beneficial as the training budget grows?
2. Does the nested model approach or fall farther behind independently trained
   standalone widths?

The core comparison is uniform global sampling with `H=1`, `H=5`, `H=25`, and
`H=50`. Global Thompson sampling and both established PanelGrad configurations
are included as comparison policies. Commands are provided for the production
eight-width grid and the optional four-width grid.

Token budget changes do not change model memory use. Every command requests
one GPU, but larger budgets may require resubmitting an interrupted job because
of the Slurm wall-time limit. Resume only with the identical run ID, budget,
configuration, seed, and one-GPU topology.

## 1. Set the budget and shared contract

Run all commands from the repository root. Activate the validated environment
and load the private tokenizer, corpus, and output paths:

```bash
conda activate elasticnn
set -a
source .env
set +a

export PYTHON_BIN="$(command -v python)"
export BASE=configs/production/slicing_100m_prepared.yaml
export EXPERIMENT_SEED=42

# Change this one variable to select the nominal budget in millions of tokens.
export BUDGET_M=200

# Adjust the allocation limit for the local cluster. Training still uses one GPU.
export WALLTIME=24:00:00

# Use new, budget-isolated roots. Do not mix these runs with the 100M suite.
export OUT_8G="/nfs-stor/$USER/results/elasticnn/slicing-${BUDGET_M}m-prepared-8g-v1"
export OUT_4G="/nfs-stor/$USER/results/elasticnn/slicing-${BUDGET_M}m-prepared-4g-v1"
```

Convert the nominal budget to the largest complete one-GPU optimizer-step
budget at 8,192 tokens per update. This produces the existing exact 100M value
of 99,999,744 tokens when `BUDGET_M=100`.

```bash
case "$BUDGET_M" in
  [1-9]|[1-9][0-9]*) ;;
  *) echo "BUDGET_M must be a positive integer" >&2; exit 2 ;;
esac

export TOKENS_PER_STEP=8192
export REQUESTED_TOKEN_BUDGET=$((BUDGET_M * 1000000))
export TOKEN_BUDGET=$((REQUESTED_TOKEN_BUDGET / TOKENS_PER_STEP * TOKENS_PER_STEP))
export DERIVED_MAX_STEPS=$((TOKEN_BUDGET / TOKENS_PER_STEP))

printf 'Nominal=%sM exact_tokens=%s optimizer_steps=%s\n' \
  "$BUDGET_M" "$TOKEN_BUDGET" "$DERIVED_MAX_STEPS"
```

The packed corpus must contain at least `TOKEN_BUDGET` optimizer-training
tokens. Preflight will reject a budget beyond the prepared corpus. Keep batch
size eight, accumulation one, and expected world size one: changing any of
them changes the step count and no longer reproduces this comparison.

Create the output roots and shared override arrays:

```bash
mkdir -p "$OUT_8G" "$OUT_4G" logs
test -r "$TOKENIZER/tokenizer_manifest.json"
test -r "$CORPUS/corpus_manifest.json"
test -w "$OUT_8G"
test -w "$OUT_4G"

COMMON_OVERRIDES=(
  --override "run.seed=$EXPERIMENT_SEED"
  --override "dataset.prepared_corpus_dir=$CORPUS"
  --override "model.tokenizer_dir=$TOKENIZER"
)

EIGHT_GRANULARITY_OVERRIDES=(
  "${COMMON_OVERRIDES[@]}"
)

FOUR_GRANULARITY_OVERRIDES=(
  "${COMMON_OVERRIDES[@]}"
  --override 'model.granularities=[g250,g500,g750,g1000]'
  --override 'model.granularity_prefixes={g250: 0.25, g500: 0.50, g750: 0.75, g1000: 1.00}'
)

SBATCH=(sbatch)
if [[ -n "${SLURM_EXCLUDE:-}" ]]; then
  SBATCH+=(--exclude="$SLURM_EXCLUDE")
fi
```

The base configuration retains AdamW, learning rate `1e-3`, 1,000 warmup
updates, and cosine decay. Because `training.token_budget` determines the
training horizon, cosine decay now ends at `DERIVED_MAX_STEPS`. This is a new
longer-horizon run from initialization, not a continuation of a completed
100M cosine schedule.

## 2. Define the policy configurations

Thompson and both PanelGrad policies retain their existing 25-update decision
or refresh interval. The scheduled PanelGrad configuration must use the
variable `DERIVED_MAX_STEPS`; hard-coding 12,207 would finish its epsilon
schedule at 100M even in a longer run.

```bash
export THOMPSON_CONTROLLER='{"preset":"bayesian_thompson","decision_interval_steps":25,"prior_mean":0.0,"prior_covariance":1.0,"observation_noise_variance":0.01,"process_noise_covariance":0.0001,"reset":{"enabled":false}}'

export PANELGRAD_RMS='{"importance_metric":"gradient_rms","refresh_interval_steps":25,"eta":1.0e-12,"temperature":1.0,"epsilon":0.1}'

export PANELGRAD_L2="{\"importance_metric\":\"gradient_l2\",\"refresh_interval_steps\":25,\"eta\":1.0e-12,\"temperature\":1.0,\"epsilon_schedule\":{\"type\":\"linear\",\"start\":0.5,\"end\":0.1,\"duration_steps\":$((BUDGET_M * 1000000 / 8192))}}"
```

PanelGrad performs controller-gradient measurements in addition to ordinary
training. It remains a one-GPU run, but its wall-clock cost is higher than
uniform global or Thompson sampling.

### Optional gradient-interference diagnostic

The uniform-global H-window runs can additionally measure raw-gradient
compatibility between every pair of nested granularities on the fixed
controller probe. The diagnostic is disabled by default. Set the opt-in flag
before defining the following arrays to enable it:

```bash
# Change to 1 for the diagnostic sweep; leave at 0 for ordinary runs.
export ENABLE_GRADIENT_INTERFERENCE=0

GRADIENT_INTERFERENCE_OVERRIDES=()
GRADIENT_INTERFERENCE_RUN_SLUG=""
if [[ "${ENABLE_GRADIENT_INTERFERENCE:-0}" == 1 ]]; then
  GRADIENT_INTERFERENCE_OVERRIDES=(
    --override 'evaluation.gradient_interference.enabled=true'
    --override 'evaluation.gradient_interference.trajectory_fractions=[0.0,0.25,0.5,0.75,1.0]'
    --override 'evaluation.gradient_interference.include_warmup_completion=true'
    --override 'evaluation.gradient_interference.layerwise=true'
  )
  GRADIENT_INTERFERENCE_RUN_SLUG="-gradient-interference"
fi
```

The separate run slug prevents an enabled diagnostic run from colliding with
or attempting to resume an existing ordinary run with a different immutable
configuration. For stronger isolation, also use diagnostic-specific `OUT_8G`
and `OUT_4G` roots.

Apply `GRADIENT_INTERFERENCE_OVERRIDES` only to `nested-random` runs with
uniform global sampling. It is intentionally incompatible with Thompson,
PanelGrad, standalone, nested-all, fixed-global, adaptive, and per-block runs,
so the comparison-policy and reference commands below do not include this
array. The production base already supplies the required fixed four-role
partition, enabled 128-example controller probe, and fixed manifest.

Milestones are successful optimizer-update counts. They are the sorted,
deduplicated union of `ceil(fraction * DERIVED_MAX_STEPS)` and the resolved
warmup-completion step. For `BUDGET_M=20`, the exact budget is 19,996,672
tokens and 2,441 updates, giving diagnostic steps
`0, 611, 1000, 1221, 1831, 2441`.

Each eight-width snapshot evaluates the complete fixed probe once per width.
With 128 packed probe sequences and batch size eight, this is 128 backward
evaluations per snapshot, or 768 across the six 20M snapshots. The journal
records actual batches, targets, backward evaluations, and duration so this
cost can be reported separately from training.

## 3. Run preflights

First verify the eight-width uniform-window contract:

```bash
"$PYTHON_BIN" train.py \
  --config "$BASE" \
  --output-root "$OUT_8G" \
  --override "run.run_id=preflight-${BUDGET_M}m-8g-uniform-window25${GRADIENT_INTERFERENCE_RUN_SLUG}-s$EXPERIMENT_SEED" \
  "${EIGHT_GRANULARITY_OVERRIDES[@]}" \
  "${GRADIENT_INTERFERENCE_OVERRIDES[@]}" \
  --override "training.token_budget=$((BUDGET_M * 1000000 / 8192 * 8192))" \
  --override "model.granularity_sampling_mode=global" \
  --override "model.global_sampling_interval_steps=25" \
  --preflight
```

Then verify the policies whose configuration changes with the budget:

```bash
"$PYTHON_BIN" train.py \
  --config "$BASE" \
  --output-root "$OUT_8G" \
  --override "run.run_id=preflight-${BUDGET_M}m-8g-ts-global-s$EXPERIMENT_SEED" \
  "${EIGHT_GRANULARITY_OVERRIDES[@]}" \
  --override "training.token_budget=$((BUDGET_M * 1000000 / 8192 * 8192))" \
  --override "model.granularity_sampling_mode=adaptive_global" \
  --override "model.adaptive_sampler_strategy=thompson" \
  --override "model.adaptive_controller=$THOMPSON_CONTROLLER" \
  --preflight

"$PYTHON_BIN" train.py \
  --config "$BASE" \
  --output-root "$OUT_8G" \
  --override "run.run_id=preflight-${BUDGET_M}m-8g-panelgrad-l2-s$EXPERIMENT_SEED" \
  "${EIGHT_GRANULARITY_OVERRIDES[@]}" \
  --override "training.token_budget=$((BUDGET_M * 1000000 / 8192 * 8192))" \
  --override "model.granularity_sampling_mode=adaptive_global" \
  --override "model.adaptive_sampler_strategy=panelgrad" \
  --override "model.panelgrad=$PANELGRAD_L2" \
  --preflight
```

Finally verify the four-width override independently:

```bash
"$PYTHON_BIN" train.py \
  --config "$BASE" \
  --output-root "$OUT_4G" \
  --override "run.run_id=preflight-${BUDGET_M}m-4g-uniform-window25${GRADIENT_INTERFERENCE_RUN_SLUG}-s$EXPERIMENT_SEED" \
  "${FOUR_GRANULARITY_OVERRIDES[@]}" \
  "${GRADIENT_INTERFERENCE_OVERRIDES[@]}" \
  --override "training.token_budget=$((BUDGET_M * 1000000 / 8192 * 8192))" \
  --override "model.granularity_sampling_mode=global" \
  --override "model.global_sampling_interval_steps=25" \
  --preflight
```

Every preflight must report:

- `effective_world_size: 1`
- `expected_tokens_per_step: 8192`
- `token_budget: $TOKEN_BUDGET`
- `derived_max_steps: $DERIVED_MAX_STEPS`
- `resolved_warmup_steps: 1000`
- eight or four ordered granularities, as appropriate
- the expected uniform, Thompson, or PanelGrad policy identity

The L2 PanelGrad preflight must additionally report an epsilon-schedule
duration equal to `DERIVED_MAX_STEPS`.

When the optional diagnostic is enabled, each uniform-window preflight must
also report `gradient_interference.enabled: true`, the expected resolved
milestones and reasons, layerwise measurement enabled, and the fixed raw
pre-correction/pre-clipping gradient semantics. A diagnostic-enabled preflight
for an incompatible policy must fail rather than silently disable measurement.

## 4. Define the one-GPU submission helper

The helper selects the correct output root and granularity overrides while
always requesting exactly one GPU:

```bash
submit_budget_run() {
  local scope="$1"
  local job_name="$2"
  local mode="$3"
  local run_id="$4"
  shift 4

  case "${BUDGET_M:-}" in
    [1-9]|[1-9][0-9]*)
      ;;
    *)
      echo "Export BUDGET_M as a positive integer before submitting" >&2
      return 2
      ;;
  esac

  local exact_token_budget=$((BUDGET_M * 1000000 / 8192 * 8192))
  if [[ "$run_id" != "${BUDGET_M}m-"* ]]; then
    echo "run ID must begin with ${BUDGET_M}m-; got: $run_id" >&2
    return 2
  fi

  local output_root
  local -a scope_overrides
  case "$scope" in
    8g)
      output_root="$OUT_8G"
      scope_overrides=("${EIGHT_GRANULARITY_OVERRIDES[@]}")
      ;;
    4g)
      output_root="$OUT_4G"
      scope_overrides=("${FOUR_GRANULARITY_OVERRIDES[@]}")
      ;;
    *)
      echo "scope must be 8g or 4g" >&2
      return 2
      ;;
  esac

  "${SBATCH[@]}" --time="$WALLTIME" --gres=gpu:1 \
    --job-name="$job_name" \
    scripts/slurm_dmodel256_pilot.sh \
    --mode "$mode" \
    --run-id "$run_id" \
    --config "$BASE" \
    --output-root "$output_root" \
    --python-bin "$PYTHON_BIN" \
    "${scope_overrides[@]}" \
    --override "training.token_budget=$exact_token_budget" \
    "$@"
}
```

The helper intentionally derives `training.token_budget` again for every
submission instead of trusting an inherited `TOKEN_BUDGET`. It also rejects a
missing budget or malformed run ID before calling `sbatch`. Consequently,
`export BUDGET_M=200` is the only budget input required at submission time.

## 5. Submit the uniform-window sweep

Submit the eight-width H sweep:

```bash
for H in 1 5 25 50; do
  if [[ "$H" == 1 ]]; then
    POLICY_ID=random-global
  else
    POLICY_ID="uniform-window$H"
  fi

  submit_budget_run 8g "${BUDGET_M}m-8g-$POLICY_ID$GRADIENT_INTERFERENCE_RUN_SLUG" nested-random \
    "${BUDGET_M}m-prepared-slicing-$POLICY_ID$GRADIENT_INTERFERENCE_RUN_SLUG-s$EXPERIMENT_SEED" \
    "${GRADIENT_INTERFERENCE_OVERRIDES[@]}" \
    --override "model.granularity_sampling_mode=global" \
    --override "model.global_sampling_interval_steps=$H"
done
```

Submit the matched four-width sweep:

```bash
for H in 1 5 25 50; do
  if [[ "$H" == 1 ]]; then
    POLICY_ID=random-global
  else
    POLICY_ID="uniform-window$H"
  fi

  submit_budget_run 4g "${BUDGET_M}m-4g-$POLICY_ID$GRADIENT_INTERFERENCE_RUN_SLUG" nested-random \
    "${BUDGET_M}m-prepared-slicing-4g-$POLICY_ID$GRADIENT_INTERFERENCE_RUN_SLUG-s$EXPERIMENT_SEED" \
    "${GRADIENT_INTERFERENCE_OVERRIDES[@]}" \
    --override "model.granularity_sampling_mode=global" \
    --override "model.global_sampling_interval_steps=$H"
done
```

`H=1` preserves ordinary per-update uniform global sampling. For `H>1`, a
new independent uniform action is drawn at successful optimizer updates
1, `H+1`, `2H+1`, and so on. The number of decisions is
`ceil(DERIVED_MAX_STEPS / H)`; adjacent windows can select the same width.

## 6. Submit Thompson and PanelGrad comparisons

Submit the three comparison policies for the eight-width grid:

```bash
submit_budget_run 8g "${BUDGET_M}m-8g-ts-global" nested-random \
  "${BUDGET_M}m-prepared-slicing-ts-global-s$EXPERIMENT_SEED" \
  --override "model.granularity_sampling_mode=adaptive_global" \
  --override "model.adaptive_sampler_strategy=thompson" \
  --override "model.adaptive_controller=$THOMPSON_CONTROLLER"

submit_budget_run 8g "${BUDGET_M}m-8g-panelgrad-rms" nested-random \
  "${BUDGET_M}m-prepared-slicing-panelgrad-rms-eps0p1-s$EXPERIMENT_SEED" \
  --override "model.granularity_sampling_mode=adaptive_global" \
  --override "model.adaptive_sampler_strategy=panelgrad" \
  --override "model.panelgrad=$PANELGRAD_RMS"

submit_budget_run 8g "${BUDGET_M}m-8g-panelgrad-l2" nested-random \
  "${BUDGET_M}m-prepared-slicing-panelgrad-l2-eps0p5-to0p1-s$EXPERIMENT_SEED" \
  --override "model.granularity_sampling_mode=adaptive_global" \
  --override "model.adaptive_sampler_strategy=panelgrad" \
  --override "model.panelgrad=$PANELGRAD_L2"
```

Submit the same comparison policies for the four-width grid:

```bash
submit_budget_run 4g "${BUDGET_M}m-4g-ts-global" nested-random \
  "${BUDGET_M}m-prepared-slicing-4g-ts-global-s$EXPERIMENT_SEED" \
  --override "model.granularity_sampling_mode=adaptive_global" \
  --override "model.adaptive_sampler_strategy=thompson" \
  --override "model.adaptive_controller=$THOMPSON_CONTROLLER"

submit_budget_run 4g "${BUDGET_M}m-4g-panelgrad-rms" nested-random \
  "${BUDGET_M}m-prepared-slicing-4g-panelgrad-rms-eps0p1-s$EXPERIMENT_SEED" \
  --override "model.granularity_sampling_mode=adaptive_global" \
  --override "model.adaptive_sampler_strategy=panelgrad" \
  --override "model.panelgrad=$PANELGRAD_RMS"

submit_budget_run 4g "${BUDGET_M}m-4g-panelgrad-l2" nested-random \
  "${BUDGET_M}m-prepared-slicing-4g-panelgrad-l2-eps0p5-to0p1-s$EXPERIMENT_SEED" \
  --override "model.granularity_sampling_mode=adaptive_global" \
  --override "model.adaptive_sampler_strategy=panelgrad" \
  --override "model.panelgrad=$PANELGRAD_L2"
```

Thompson evaluates its controller panel every 25 successful updates.
PanelGrad refreshes controller-gradient measurements at the same cadence.
Uniform H=25 is therefore the no-controller temporal-batching control.

## 7. Submit standalone references

For the complete eight-width comparison, submit one independent run per
width:

```bash
for GRANULARITY in g125 g250 g375 g500 g625 g750 g875 g1000; do
  submit_budget_run 8g "${BUDGET_M}m-8g-standalone-$GRANULARITY" standalone \
    "${BUDGET_M}m-prepared-slicing-standalone-$GRANULARITY-s$EXPERIMENT_SEED" \
    --granularity "$GRANULARITY"
done
```

For the isolated four-width comparison:

```bash
for GRANULARITY in g250 g500 g750 g1000; do
  submit_budget_run 4g "${BUDGET_M}m-4g-standalone-$GRANULARITY" standalone \
    "${BUDGET_M}m-prepared-slicing-4g-standalone-$GRANULARITY-s$EXPERIMENT_SEED" \
    --granularity "$GRANULARITY"
done
```

If compute is constrained and the immediate question is how the standalone
gap evolves, start with the smallest, a middle, and the full width:

- eight widths: `g125`, `g500`, and `g1000`
- four widths: `g250`, `g500`, and `g1000`

The complete standalone set is required for a full perplexity-versus-size
curve.

## 8. Optional joint-training references

Nested-all is substantially more compute-intensive per optimizer update, but
it uses the same one-GPU memory contract and can be included as a high-compute
reference:

```bash
submit_budget_run 8g "${BUDGET_M}m-8g-nested-all" nested-all \
  "${BUDGET_M}m-prepared-slicing-nested-all-s$EXPERIMENT_SEED"

submit_budget_run 4g "${BUDGET_M}m-4g-nested-all" nested-all \
  "${BUDGET_M}m-prepared-slicing-4g-nested-all-s$EXPERIMENT_SEED"
```

Do not use nested-all as a matched-compute substitute for the global sampling
policies: it evaluates and backpropagates all configured widths each update.

## 9. Monitor, resume, and verify

```bash
squeue --me
tail -f logs/matformer_dmodel256_<job-id>.out
tail -f logs/matformer_dmodel256_<job-id>.err
```

If a job reaches its wall-time limit, resubmit its exact `submit_budget_run`
command. The checkpoint continuation preserves optimizer, scheduler, policy,
RNG, exposure, and successful-update state. Never submit the same run ID
concurrently.

After a run finishes, resolve its directory and verify the variable budget:

```bash
export RUN_ID="${BUDGET_M}m-prepared-slicing-uniform-window25${GRADIENT_INTERFERENCE_RUN_SLUG}-s$EXPERIMENT_SEED"
export RUN_DIR="$(find "$OUT_8G" -type d -name "$RUN_ID" -print -quit)"
test -n "$RUN_DIR"

"$PYTHON_BIN" - "$RUN_DIR" "$TOKEN_BUDGET" "$DERIVED_MAX_STEPS" <<'PY'
import csv
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
expected_tokens = int(sys.argv[2])
expected_steps = int(sys.argv[3])

config = json.loads((run_dir / "config.json").read_text())
summary = json.loads((run_dir / "run_summary.json").read_text())
rows = [
    row
    for row in csv.DictReader((run_dir / "metrics.csv").open())
    if row["split"] == "train"
]

assert config["training"]["effective_world_size"] == 1
assert config["training"]["expected_tokens_per_step"] == 8192
assert config["training"]["token_budget"] == expected_tokens
assert config["training"]["derived_max_steps"] == expected_steps
assert config["training"]["resolved_warmup_steps"] == 1000
assert config["model"]["global_sampling_interval_steps"] == 25
assert summary["status"] == "completed"
assert summary["tokens_seen"] == expected_tokens
assert len(rows) == expected_steps

state = summary["global_sampling_state"]
assert state["total_successful_updates"] == expected_steps
assert sum(state["exposure_counts"].values()) == expected_steps
decision_count = (expected_steps + 24) // 25
assert state["window_index"] == decision_count - 1
assert state["successful_updates_in_window"] == (expected_steps - 1) % 25 + 1
assert not summary.get("unresolved_artifact_failures")
assert (run_dir / "final_holdout_results.json").is_file()

diagnostic = config["evaluation"]["gradient_interference"]
if diagnostic["enabled"]:
    expected_diagnostic_steps = diagnostic["resolved_steps"]
    journal_path = run_dir / "gradient_interference.jsonl"
    records = [
        json.loads(line)
        for line in journal_path.read_text().splitlines()
        if line.strip()
    ]
    assert [record["step"] for record in records] == expected_diagnostic_steps
    assert summary["gradient_interference_snapshot_count"] == len(
        expected_diagnostic_steps
    )
    assert (
        summary["gradient_interference_measured_steps"]
        == expected_diagnostic_steps
    )
    assert (
        summary["gradient_interference_expected_steps"]
        == expected_diagnostic_steps
    )
    assert summary["gradient_interference_measurement_cost"][
        "backward_evaluations"
    ] > 0

print(
    "verified",
    summary["run_id"],
    expected_tokens,
    "tokens,",
    expected_steps,
    "updates, and",
    decision_count,
    "H=25 decisions",
)
PY
```

Within a budget and width grid, require identical optimizer-training,
ordinary-validation, controller, and final-holdout role hashes before comparing
policies. Across different budgets, the optimizer-training manifest is expected
to differ because it contains a different number of packed training rows; the
validation, controller, and final-holdout roles must remain fixed.

`scripts/make_figures.py` automatically discovers completed runs whose saved
`config.json` enables the diagnostic. It writes one grouped H-policy comparison
per compatible contract as
`gradient_interference_cosine_trajectories__<contract>.png` and one six-panel
milestone matrix per run as
`gradient_interference_cosine_heatmaps__<run-id>.png`. The ordinary figure
commands below need no additional flag; `--variant`, `--correction`, and `--dpi`
also apply to these diagnostic figures.

Reporting silently ignores disabled runs, stray journals attached to disabled
configs, and incomplete runs. A completed enabled run must have all resolved
milestones in `gradient_interference.jsonl` plus a completed `run_summary.json`
with matching coverage and journal hash. Run, fixed-probe, controlled-support,
and diagnostic-contract identities must agree across the saved config, summary,
and every snapshot. Invalid completed artifacts fail figure generation instead
of producing a partial or misleading comparison.

Use distinct run IDs for ordinary and diagnostic runs so their checkpoints and
continuation state cannot collide. Separate output roots are useful for
operational isolation, but are not required: related runs may share a campaign
root when they should be discovered by the same figure-generation command.

## 10. Generate budget-isolated figures

The run directory family uses the rounded budget slug generated by the
configuration system. For integer `BUDGET_M`, it is `${BUDGET_M}m_tokens`:

```bash
export GROUP_DIR_8G="$OUT_8G/matformer_llama_148m_${BUDGET_M}m_tokens"
export GROUP_DIR_4G="$OUT_4G/matformer_llama_148m_${BUDGET_M}m_tokens"

"$PYTHON_BIN" scripts/make_figures.py \
  --input "$GROUP_DIR_8G" \
  --output "$GROUP_DIR_8G/figures" \
  --variant slicing \
  --correction none

"$PYTHON_BIN" scripts/make_figures.py \
  --input "$GROUP_DIR_4G" \
  --output "$GROUP_DIR_4G/figures" \
  --variant slicing \
  --correction none
```

Use the reserved `final_holdout_results.json` values for final policy and
standalone comparisons. Use ordinary validation checkpoints to study when the
H-policy and standalone gaps grow, shrink, or cross during training.

For each width grid, compare:

- H=1 versus H=5, H=25, and H=50 at matched tokens
- H=25 versus Thompson to separate temporal batching from adaptation
- the best uniform H versus PanelGrad, reporting controller-measurement cost
- every nested width versus its same-budget standalone run

Do not compare the endpoint of a new longer cosine schedule with a resumed
100M checkpoint as though they were the same training schedule. To measure how
gaps evolve with tokens, compare checkpoints from policies trained with the
same longer horizon and optimizer contract.

## 11. Balanced-exposure follow-up screen

This follow-up isolates temporal specialization from unequal sampled-width
exposure. Under `balanced_cycle`, every width receives the same number of
selected optimizer updates, while H changes only how long consecutive updates
remain on one width. If H=50 changes the endpoint or the gradient-interference
trajectory relative to balanced H=1, that is evidence consistent with temporal
clustering affecting specialization rather than ordinary multinomial exposure
noise. It is not by itself proof of a causal mechanism.

Equal selected-width exposure does not equal equal parameter exposure. Nested
widths share prefixes, so parameters in smaller prefixes still participate in
updates selected for larger widths. The balanced contract removes imbalance in
the selected labels; it cannot and should not remove this unavoidable nested
shared-parameter exposure.

Put this screen in the existing per-grid campaign roots alongside the IID
uniform-window experiments. The balanced run IDs below create distinct run
directories, so checkpoints and continuation state remain isolated while one
figure-generation pass can discover both schedules:

```bash
test -d "$OUT_8G" && test -w "$OUT_8G"
test -d "$OUT_4G" && test -w "$OUT_4G"
```

The ordinary `scripts/make_figures.py` commands in section 10 read these shared
roots. They discover IID and balanced histories together for direct comparison,
while classifying balanced curves as `balanced_global_h<H>` and placing them in
the separate `balanced_global_window` panel. Balanced H=1 is the dotted
reference and H is ordered numerically. Gradient-interference trajectories
group balanced H values together but never share a contract with IID
uniform-global histories.

Run only the one-seed H=1/H=50 screen on the eight- and four-width grids. The
exact budget is 2,400 optimizer updates at 8,192 tokens per update:

```bash
export BALANCED_TOKEN_BUDGET=19660800
export BALANCED_TRAJECTORY='[0.0,0.3333333333333333,0.6666666666666666,1.0]'

for H in 1 50; do
  "${SBATCH[@]}" --time="$WALLTIME" --gres=gpu:1 \
    --job-name="balanced-8g-h$H" scripts/slurm_dmodel256_pilot.sh \
    --mode nested-random \
    --run-id "balanced-8g-h$H-s$EXPERIMENT_SEED" \
    --config "$BASE" --output-root "$OUT_8G" --python-bin "$PYTHON_BIN" \
    "${EIGHT_GRANULARITY_OVERRIDES[@]}" \
    --override "training.token_budget=$BALANCED_TOKEN_BUDGET" \
    --override "model.granularity_sampling_mode=global" \
    --override "model.global_sampling_schedule=balanced_cycle" \
    --override "model.global_sampling_interval_steps=$H" \
    --override "training.pre_nested_warmup.enabled=false" \
    --override "evaluation.gradient_interference.enabled=true" \
    --override "evaluation.gradient_interference.trajectory_fractions=$BALANCED_TRAJECTORY" \
    --override "evaluation.gradient_interference.include_warmup_completion=false"

  "${SBATCH[@]}" --time="$WALLTIME" --gres=gpu:1 \
    --job-name="balanced-4g-h$H" scripts/slurm_dmodel256_pilot.sh \
    --mode nested-random \
    --run-id "balanced-4g-h$H-s$EXPERIMENT_SEED" \
    --config "$BASE" --output-root "$OUT_4G" --python-bin "$PYTHON_BIN" \
    "${FOUR_GRANULARITY_OVERRIDES[@]}" \
    --override "training.token_budget=$BALANCED_TOKEN_BUDGET" \
    --override "model.granularity_sampling_mode=global" \
    --override "model.global_sampling_schedule=balanced_cycle" \
    --override "model.global_sampling_interval_steps=$H" \
    --override "training.pre_nested_warmup.enabled=false" \
    --override "evaluation.gradient_interference.enabled=true" \
    --override "evaluation.gradient_interference.trajectory_fractions=$BALANCED_TRAJECTORY" \
    --override "evaluation.gradient_interference.include_warmup_completion=false"
done
```

The resolved diagnostic milestones are steps `0, 800, 1600, 2400`; the
ordinary 1,000-step optimizer warmup remains configured, but it is deliberately
not added as a diagnostic milestone because it is not a complete H=50 cycle.
At completion, require 300 selected updates per width on the eight-width grid
and 600 per width on the four-width grid.

The primary interference contrasts are endpoint cosine `g125–g1000` for eight
widths and `g250–g1000` for four widths. The practical outcome is untouched
final-holdout loss. Do not add seeds, blockwise figures, or H=5/H=25 runs for
the initial screen. Only if an endpoint result is interesting should the same
contract be extended with H=5 and H=25; those follow-ups use the existing loop
with `for H in 5 25` and no other changes.
