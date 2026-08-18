# 100M prepared-corpus slicing experiments

This is the operator guide for the clean 100M slicing comparison. It reuses
the immutable tokenizer and packed FineWeb `sample-100BT` corpus prepared for
the 10B experiments and uses a one-GPU batch-eight update geometry throughout
the new comparison. Run every command from a clean checkout of `main` at the
repository root.

The purpose of this suite is to replace the earlier mixed-contract figure with
runs that all use the same optimizer-training order, ordinary-validation role,
128-example controller role, and 512-example final holdout. Use a dedicated
output root so the new runs are never aggregated with the legacy 100M runs.

The primary seed-42 matrix is intentionally small:

| Group | Runs |
|---|---|
| Baselines | random global, nested-all, eight standalones |
| Thompson sampling | global TS without resets |
| PanelGrad | gradient RMS with fixed epsilon 0.1; gradient L2 with epsilon 0.5 to 0.1 |

Random per-block sampling is retained as an optional diagnostic. Per-block TS,
TS resets, PanelGrad temperatures 2 and 4, and the other epsilon ablations are
not part of the primary reproduction because they did not improve the earlier
100M comparison.

## 1. Start from current `main`

```bash
git switch main
git pull --ff-only
git status --short
```

`git status --short` must print nothing before environment setup or
submission.

## 2. Activate the validated environment

Create the environment if it is not already available:

```bash
conda env create --file environment.yml
conda activate elasticnn
python -m pip check
```

For an existing environment:

```bash
conda activate elasticnn
python -m pip check
```

## 3. Configure private paths and the shared contract

Use the same prepared tokenizer and corpus as the 10B experiments. Set `OUT`
to a new, dedicated directory, not the root containing the legacy 100M runs.

```bash
set -a
source .env
set +a

export PYTHON_BIN="$(command -v python)"
export BASE=configs/production/slicing_100m_prepared.yaml
export EXPERIMENT_SEED=42

# Example only: choose a new path supplied for this comparison.
# export OUT="/nfs-stor/$USER/results/elasticnn/slicing-100m-prepared-v1"

mkdir -p "$OUT" logs
test -r "$TOKENIZER/tokenizer_manifest.json"
test -r "$CORPUS/corpus_manifest.json"
test -w "$OUT"
```

Define the exact scientific overrides shared by every run:

```bash
CONFIG_OVERRIDES=(
  --override "run.seed=$EXPERIMENT_SEED"
  --override "dataset.prepared_corpus_dir=$CORPUS"
  --override "model.tokenizer_dir=$TOKENIZER"
)

SLURM_COMMON=(
  --config "$BASE"
  --output-root "$OUT"
  --python-bin "$PYTHON_BIN"
  "${CONFIG_OVERRIDES[@]}"
)
```

The nominal 100M budget is exactly `99,999,744` token IDs. Packed training
requires complete 1,024-token rows, and this is the largest valid packed budget
below 100,000,000. It contains 97,656 packed sequences and, at eight sequences
per optimizer step, resolves to exactly 12,207 optimizer updates. Relative to
the earlier batch-four runs, all step-based schedules below are halved so their
cadence in processed tokens remains unchanged.

The production base supplies the eight slicing widths used by the 10B suite:
`g125`, `g250`, `g375`, `g500`, `g625`, `g750`, `g875`, and `g1000`.

## 4. Run policy preflights

Run these locally before submitting any GPU jobs.

### Random-global baseline

```bash
python train.py \
  --config "$BASE" \
  --output-root "$OUT" \
  --override "run.run_id=preflight-100m-random-global-s$EXPERIMENT_SEED" \
  "${CONFIG_OVERRIDES[@]}" \
  --override "model.granularity_sampling_mode=global" \
  --preflight
```

### Global Thompson sampling

```bash
python train.py \
  --config "$BASE" \
  --output-root "$OUT" \
  --override "run.run_id=preflight-100m-ts-global-s$EXPERIMENT_SEED" \
  "${CONFIG_OVERRIDES[@]}" \
  --override "model.granularity_sampling_mode=adaptive_global" \
  --override "model.adaptive_sampler_strategy=thompson" \
  --override 'model.adaptive_controller={"preset":"bayesian_thompson","decision_interval_steps":25,"prior_mean":0.0,"prior_covariance":1.0,"observation_noise_variance":0.01,"process_noise_covariance":0.0001,"reset":{"enabled":false}}' \
  --preflight
```

### PanelGrad raw L2 with scheduled epsilon

```bash
python train.py \
  --config "$BASE" \
  --output-root "$OUT" \
  --override "run.run_id=preflight-100m-panelgrad-l2-s$EXPERIMENT_SEED" \
  "${CONFIG_OVERRIDES[@]}" \
  --override "model.granularity_sampling_mode=adaptive_global" \
  --override "model.adaptive_sampler_strategy=panelgrad" \
  --override 'model.panelgrad={"importance_metric":"gradient_l2","refresh_interval_steps":25,"eta":1.0e-12,"temperature":1.0,"epsilon_schedule":{"type":"linear","start":0.5,"end":0.1,"duration_steps":12207}}' \
  --preflight
```

Every preflight must report:

- `effective_world_size: 1`
- `expected_tokens_per_microstep: 8192`
- `gradient_accumulation_steps: 1`
- `expected_tokens_per_step: 8192`
- `token_budget: 99999744`
- `derived_max_steps: 12207`
- `resolved_warmup_steps: 1000`
- the eight ordered production granularities
- identical corpus, optimizer-training, validation, controller, and final-holdout hashes

The TS preflight must additionally report global Thompson sampling with
`h=25` and no pre-nested warmup. The PanelGrad preflight must report
`panelgrad_gradient_l2` and a linear epsilon schedule with duration 12,207.
That resolved lifecycle produces 489 refreshes, including the initial
step-zero boundary and the last active boundary at step 12,200.

## 5. Run one-update smokes

These four jobs cover the distinct training paths used by the primary matrix.
Do not submit the full runs until they complete successfully.

```bash
sbatch --time=01:00:00 --gres=gpu:1 \
  --job-name=smoke-100m-random-global \
  scripts/slurm_dmodel256_pilot.sh \
  --mode nested-random \
  --run-id "smoke-100m-random-global-s$EXPERIMENT_SEED" \
  "${SLURM_COMMON[@]}" \
  --override "model.granularity_sampling_mode=global" \
  --override "training.max_steps_cap=1"

sbatch --time=01:00:00 --gres=gpu:1 \
  --job-name=smoke-100m-nested-all \
  scripts/slurm_dmodel256_pilot.sh \
  --mode nested-all \
  --run-id "smoke-100m-nested-all-s$EXPERIMENT_SEED" \
  "${SLURM_COMMON[@]}" \
  --override "training.max_steps_cap=1"

sbatch --time=01:00:00 --gres=gpu:1 \
  --job-name=smoke-100m-ts-global \
  scripts/slurm_dmodel256_pilot.sh \
  --mode nested-random \
  --run-id "smoke-100m-ts-global-s$EXPERIMENT_SEED" \
  "${SLURM_COMMON[@]}" \
  --override "model.granularity_sampling_mode=adaptive_global" \
  --override "model.adaptive_sampler_strategy=thompson" \
  --override 'model.adaptive_controller={"preset":"bayesian_thompson","decision_interval_steps":25,"prior_mean":0.0,"prior_covariance":1.0,"observation_noise_variance":0.01,"process_noise_covariance":0.0001,"reset":{"enabled":false}}' \
  --override "training.max_steps_cap=1"

sbatch --time=01:00:00 --gres=gpu:1 \
  --job-name=smoke-100m-panelgrad-l2 \
  scripts/slurm_dmodel256_pilot.sh \
  --mode nested-random \
  --run-id "smoke-100m-panelgrad-l2-s$EXPERIMENT_SEED" \
  "${SLURM_COMMON[@]}" \
  --override "model.granularity_sampling_mode=adaptive_global" \
  --override "model.adaptive_sampler_strategy=panelgrad" \
  --override 'model.panelgrad={"importance_metric":"gradient_l2","refresh_interval_steps":25,"eta":1.0e-12,"temperature":1.0,"epsilon_schedule":{"type":"linear","start":0.5,"end":0.1,"duration_steps":12207}}' \
  --override "training.max_steps_cap=1"
```

## 6. Submit the primary seed-42 matrix

Every command below requests one GPU and the default 24-hour allocation. Each
command submits exactly one run.

### 6.1 Matched baselines

Random global is the primary low-cost baseline:

```bash
sbatch --time=24:00:00 --gres=gpu:1 \
  --job-name=100m-random-global \
  scripts/slurm_dmodel256_pilot.sh \
  --mode nested-random \
  --run-id "100m-prepared-slicing-random-global-s$EXPERIMENT_SEED" \
  "${SLURM_COMMON[@]}" \
  --override "model.granularity_sampling_mode=global"
```

Nested-all is the high-compute joint-training reference:

```bash
sbatch --time=24:00:00 --gres=gpu:1 \
  --job-name=100m-nested-all \
  scripts/slurm_dmodel256_pilot.sh \
  --mode nested-all \
  --run-id "100m-prepared-slicing-nested-all-s$EXPERIMENT_SEED" \
  "${SLURM_COMMON[@]}"
```

Independent standalone controls:

```bash
sbatch --time=24:00:00 --gres=gpu:1 --job-name=100m-standalone-g125  scripts/slurm_dmodel256_pilot.sh --mode standalone --granularity g125  --run-id "100m-prepared-slicing-standalone-g125-s$EXPERIMENT_SEED"  "${SLURM_COMMON[@]}"
sbatch --time=24:00:00 --gres=gpu:1 --job-name=100m-standalone-g250  scripts/slurm_dmodel256_pilot.sh --mode standalone --granularity g250  --run-id "100m-prepared-slicing-standalone-g250-s$EXPERIMENT_SEED"  "${SLURM_COMMON[@]}"
sbatch --time=24:00:00 --gres=gpu:1 --job-name=100m-standalone-g375  scripts/slurm_dmodel256_pilot.sh --mode standalone --granularity g375  --run-id "100m-prepared-slicing-standalone-g375-s$EXPERIMENT_SEED"  "${SLURM_COMMON[@]}"
sbatch --time=24:00:00 --gres=gpu:1 --job-name=100m-standalone-g500  scripts/slurm_dmodel256_pilot.sh --mode standalone --granularity g500  --run-id "100m-prepared-slicing-standalone-g500-s$EXPERIMENT_SEED"  "${SLURM_COMMON[@]}"
sbatch --time=24:00:00 --gres=gpu:1 --job-name=100m-standalone-g625  scripts/slurm_dmodel256_pilot.sh --mode standalone --granularity g625  --run-id "100m-prepared-slicing-standalone-g625-s$EXPERIMENT_SEED"  "${SLURM_COMMON[@]}"
sbatch --time=24:00:00 --gres=gpu:1 --job-name=100m-standalone-g750  scripts/slurm_dmodel256_pilot.sh --mode standalone --granularity g750  --run-id "100m-prepared-slicing-standalone-g750-s$EXPERIMENT_SEED"  "${SLURM_COMMON[@]}"
sbatch --time=24:00:00 --gres=gpu:1 --job-name=100m-standalone-g875  scripts/slurm_dmodel256_pilot.sh --mode standalone --granularity g875  --run-id "100m-prepared-slicing-standalone-g875-s$EXPERIMENT_SEED"  "${SLURM_COMMON[@]}"
sbatch --time=24:00:00 --gres=gpu:1 --job-name=100m-standalone-g1000 scripts/slurm_dmodel256_pilot.sh --mode standalone --granularity g1000 --run-id "100m-prepared-slicing-standalone-g1000-s$EXPERIMENT_SEED" "${SLURM_COMMON[@]}"
```

Optional random per-block diagnostic:

```bash
sbatch --time=24:00:00 --gres=gpu:1 \
  --job-name=100m-random-per-block \
  scripts/slurm_dmodel256_pilot.sh \
  --mode nested-random \
  --run-id "100m-prepared-slicing-random-per-block-s$EXPERIMENT_SEED" \
  "${SLURM_COMMON[@]}" \
  --override "model.granularity_sampling_mode=per_block"
```

### 6.2 Thompson reproduction

Reproduce only global Thompson sampling without resets. This was the strongest
TS path in the earlier plain-slicing 100M comparison. The explicit values
below reproduce that controller profile while using the new prepared-corpus
contract and eight production widths.

```bash
sbatch --time=24:00:00 --gres=gpu:1 \
  --job-name=100m-ts-global \
  scripts/slurm_dmodel256_pilot.sh \
  --mode nested-random \
  --run-id "100m-prepared-slicing-ts-global-s$EXPERIMENT_SEED" \
  "${SLURM_COMMON[@]}" \
  --override "model.granularity_sampling_mode=adaptive_global" \
  --override "model.adaptive_sampler_strategy=thompson" \
  --override 'model.adaptive_controller={"preset":"bayesian_thompson","decision_interval_steps":25,"prior_mean":0.0,"prior_covariance":1.0,"observation_noise_variance":0.01,"process_noise_covariance":0.0001,"reset":{"enabled":false}}'
```

### 6.3 PanelGrad reproductions

RMS with fixed epsilon 0.1 had the best uniform-average final-holdout loss
among the earlier PanelGrad runs and favored the smallest widths:

```bash
sbatch --time=24:00:00 --gres=gpu:1 \
  --job-name=100m-panelgrad-rms \
  scripts/slurm_dmodel256_pilot.sh \
  --mode nested-random \
  --run-id "100m-prepared-slicing-panelgrad-rms-eps0p1-s$EXPERIMENT_SEED" \
  "${SLURM_COMMON[@]}" \
  --override "model.granularity_sampling_mode=adaptive_global" \
  --override "model.adaptive_sampler_strategy=panelgrad" \
  --override 'model.panelgrad={"importance_metric":"gradient_rms","refresh_interval_steps":25,"eta":1.0e-12,"temperature":1.0,"epsilon":0.1}'
```

Raw L2 with a linear epsilon schedule gave the strongest medium-to-full-width
PanelGrad curve:

```bash
sbatch --time=24:00:00 --gres=gpu:1 \
  --job-name=100m-panelgrad-l2 \
  scripts/slurm_dmodel256_pilot.sh \
  --mode nested-random \
  --run-id "100m-prepared-slicing-panelgrad-l2-eps0p5-to0p1-s$EXPERIMENT_SEED" \
  "${SLURM_COMMON[@]}" \
  --override "model.granularity_sampling_mode=adaptive_global" \
  --override "model.adaptive_sampler_strategy=panelgrad" \
  --override 'model.panelgrad={"importance_metric":"gradient_l2","refresh_interval_steps":25,"eta":1.0e-12,"temperature":1.0,"epsilon_schedule":{"type":"linear","start":0.5,"end":0.1,"duration_steps":12207}}'
```

With eight granularities and a 128-example controller panel, each PanelGrad
refresh performs 128 controller backward evaluations at batch size eight.
There are 489 refreshes, so each complete PanelGrad run records 62,592
measurement backward evaluations in addition to its 12,207 ordinary training
updates.

## 7. Replicate only the decision-critical runs

Seed 42 establishes the clean curve. If it completes correctly, repeat random
global, global TS, RMS PanelGrad, and L2 PanelGrad with seeds 43 and 44 before
drawing a method conclusion. Do not initially replicate nested-all or all
standalones.

For each replication, start a fresh shell, set the seed before defining
`CONFIG_OVERRIDES`, and use the same commands:

```bash
export EXPERIMENT_SEED=43
```

Then repeat with:

```bash
export EXPERIMENT_SEED=44
```

The prepared corpus keeps `dataset.data_seed=42`, so all seeds consume the same
optimizer sequence order and role manifests. `run.seed` changes model
initialization and policy RNG streams. Every replication has a distinct run ID
through its `-s43` or `-s44` suffix.

## 8. Monitor, resume, and verify completion

```bash
squeue --me
tail -f logs/matformer_dmodel256_<job-id>.out
tail -f logs/matformer_dmodel256_<job-id>.err
```

Resolve one run directory and inspect its heartbeat journal:

```bash
export RUN_ID="100m-prepared-slicing-random-global-s42"
export RUN_DIR="$(find "$OUT" -type d -name "$RUN_ID" -print -quit)"
test -n "$RUN_DIR"
tail -n 20 "$RUN_DIR/heartbeats.jsonl"
```

Resume an interrupted run by resubmitting its exact command with the same run
ID, seed, tokenizer, corpus, and one-GPU topology. Never run two jobs with the
same run ID concurrently.

The Slurm wrapper launches final-holdout evaluation automatically after
training completes. Verify each run with:

```bash
python -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); s=json.loads(p.read_text()); assert s["status"] == "completed", s.get("status"); assert s["tokens_seen"] == 99_999_744, s.get("tokens_seen"); assert not s.get("unresolved_artifact_failures"), s.get("unresolved_artifact_failures"); print("completed", s["run_id"], s["tokens_seen"], "tokens")' "$RUN_DIR/run_summary.json"
test -r "$RUN_DIR/final_holdout_results.json"
```

Before comparing metrics, confirm that all runs report identical data hashes:

```bash
python - "$OUT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
fields = (
    "optimizer_training_manifest_hash",
    "validation_manifest_hash",
    "controller_manifest_hash",
    "final_holdout_manifest_hash",
)
rows = []
for path in sorted(root.rglob("run_summary.json")):
    summary = json.loads(path.read_text())
    if str(summary.get("run_id", "")).startswith("100m-prepared-slicing-"):
        rows.append((summary["run_id"], tuple(summary.get(field) for field in fields)))
assert rows, "no completed 100M prepared slicing runs found"
reference = rows[0][1]
for run_id, hashes in rows:
    assert hashes == reference, (run_id, hashes, reference)
print("matched data contract for", len(rows), "runs")
PY
```

## 9. Generate isolated figures

Because `OUT` is dedicated to this suite, figure discovery cannot mix in the
legacy 100M artifacts.

```bash
export GROUP_DIR="$OUT/matformer_llama_148m_100m_tokens"

python scripts/make_figures.py \
  --input "$GROUP_DIR" \
  --output "$GROUP_DIR/figures" \
  --variant slicing \
  --correction none
```

Use `final_holdout_results.json` as the primary method-selection surface.
Ordinary validation selects checkpoints and is useful for learning curves, but
it must not replace the reserved final holdout in the final comparison.

## 10. Fixed scientific contract

- Architecture: d-model 256, 16 layers, 16 attention heads, context 1,024.
- Variant: slicing only, without GMC or LMC correction.
- Widths: the eight production prefixes from 0.125 through 1.0.
- Data: immutable prepared FineWeb `sample-100BT` tokenizer and packed corpus.
- Budget: 99,999,744 exact packed token IDs, 12,207 optimizer updates.
- Optimizer geometry: one GPU, batch size 8, accumulation 1.
- Scheduler: cosine with 1,000 warmup updates.
- Validation: every 250 updates and at completion.
- Roles: fixed optimizer-training, 512-example validation, 128-example
  controller, and 512-example final holdout for every method.
- Continuation: exact checkpoint resume with unchanged run identity and
  topology.

Do not change the tokenizer, corpus, role contract, width grid, token budget,
optimizer geometry, or correction mode within this comparison. Any such
change starts a new experiment family and requires a new output root.
