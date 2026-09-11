# TinyStories-Instruct Sign-Dynamics Runbook

This runbook launches only the elastic sign-dynamics campaign. Defaults are
budget `4B`, seed `42`, long interval `H=548`, shared AdamW state, BF16, and a
24-hour Slurm allocation.

## 1. Environment and prepared data

From the repository root, activate the training environment and source the
same TinyStories-Instruct profile used by the existing campaigns. The profile
defaults to `/nfs-stor/$USER/results/elasticnn` for results and resolves the
prepared tokenizer and packed corpus beneath the shared NFS roots:

```bash
source /apps/local/anaconda3.10/etc/profile.d/conda.sh
conda activate elasticnn
export PYTHON_BIN="$(command -v python)"

export TINYSTORIES_PROFILE=instruct
source scripts/select_tinystories_profile.sh

export SIGN_DYNAMICS_TOKENIZER_DIR="${SIGN_DYNAMICS_TOKENIZER_DIR:-$TOKENIZER}"
export SIGN_DYNAMICS_CORPUS_DIR="${SIGN_DYNAMICS_CORPUS_DIR:-$CORPUS}"

mkdir -p "$MATFORMER_EXPERIMENT_ROOT"
test -r "$SIGN_DYNAMICS_TOKENIZER_DIR/tokenizer_manifest.json"
test -r "$SIGN_DYNAMICS_CORPUS_DIR/corpus_manifest.json"
test -w "$MATFORMER_EXPERIMENT_ROOT"
```

With the profile defaults, the resolved locations are:

```text
/nfs-stor/$USER/results/elasticnn
/nfs-stor/$USER/matformer-tokenizers/tinystories-instruct-sentencepiece-bpe-2k-v1
/nfs-stor/$USER/matformer-corpora/tinystories-instruct-packed-full-v1
```

Only if either manifest check fails, prepare the deterministic artifacts at
those same resolved locations:

```bash
mkdir -p "$MATFORMER_TOKENIZER_ROOT" "$MATFORMER_CORPUS_ROOT"

"$PYTHON_BIN" scripts/prepare_tinystories_instruct.py \
  --tokenizer-dir "$SIGN_DYNAMICS_TOKENIZER_DIR" \
  --corpus-dir "$SIGN_DYNAMICS_CORPUS_DIR" \
  --optimizer-token-count all \
  --tokenization-workers 4
```

Preparation may reuse already matching manifests. Do not edit the generated
tokenizer or corpus manifests between preflight and training.

## 2. Select the matrix

Set the experiment root and any desired overrides. Comma-separated arms use
stable aliases; the launcher expands `hlong` to the configured numeric value.

```bash
# Keep the shared profile result root unless it was explicitly overridden.
export MATFORMER_EXPERIMENT_ROOT="${MATFORMER_EXPERIMENT_ROOT:-/nfs-stor/$USER/results/elasticnn}"
export SIGN_DYNAMICS_B_MULTIPLIER="${SIGN_DYNAMICS_B_MULTIPLIER:-4}"
export SIGN_DYNAMICS_LONG_H="${SIGN_DYNAMICS_LONG_H:-548}"
export SIGN_DYNAMICS_SEEDS="${SIGN_DYNAMICS_SEEDS:-42}"
export SIGN_DYNAMICS_ARMS="${SIGN_DYNAMICS_ARMS:-uniform_h1,uniform_hlong,balanced_h1,balanced_hlong,fixed_inverse_membership,thompson}"
export SIGN_DYNAMICS_NODE_EXCLUSIONS="${SIGN_DYNAMICS_NODE_EXCLUSIONS-gpu-[05,50,51]}"
export SIGN_DYNAMICS_SLURM_TIME="${SIGN_DYNAMICS_SLURM_TIME:-24:00:00}"

mkdir -p "$MATFORMER_EXPERIMENT_ROOT" logs
printf 'results=%s\ntokenizer=%s\ncorpus=%s\nexclude=%s\nslurm_time=%s\n' \
  "$MATFORMER_EXPERIMENT_ROOT" \
  "$SIGN_DYNAMICS_TOKENIZER_DIR" \
  "$SIGN_DYNAMICS_CORPUS_DIR" \
  "${SIGN_DYNAMICS_NODE_EXCLUSIONS:-none}" \
  "$SIGN_DYNAMICS_SLURM_TIME"
```

The six default arms are uniform `H=1`, uniform long-H, balanced `H=1`,
balanced long-H, fixed probabilities `[0.12, 0.16, 0.24, 0.48]`, and Thompson
with a 25-step decision interval. To run a subset, change only
`SIGN_DYNAMICS_ARMS`; for example:

```bash
export SIGN_DYNAMICS_ARMS="uniform_h1,balanced_hlong"
```

The output tree is
`$MATFORMER_EXPERIMENT_ROOT/tinystories-instruct-sign-dynamics-v1/x<MULTIPLIER>/runs/<arm>/s<seed>`.

## 3. Preflight every selected arm and seed

The launcher expands the same matrix for preflight and submission:

```bash
"$PYTHON_BIN" scripts/run_tinystories_sign_dynamics.py preflight \
  --python-bin "$PYTHON_BIN"
```

Every JSON response must show the expected token budget, global-step count,
shared optimizer state, BF16 precision, selected sampling policy, and exact
`evaluation.sign_dynamics` contract. Do not submit a partial matrix after a
preflight failure.

## 4. Submit through Slurm

```bash
"$PYTHON_BIN" scripts/run_tinystories_sign_dynamics.py submit \
  --python-bin "$PYTHON_BIN"
```

`SIGN_DYNAMICS_NODE_EXCLUSIONS` defaults to `gpu-[05,50,51]` and is forwarded
as one Slurm `--exclude` argument. Set it explicitly to an empty string before
submission to disable exclusions. `SIGN_DYNAMICS_SLURM_TIME` defaults to
`24:00:00`; the launcher passes it as an `sbatch --time` override, replacing
the shared wrapper's 30-minute directive. Use `--dry-run` to print all `sbatch`
commands without submitting them.

## 5. Resume timed-out runs

Confirm the earlier allocation is no longer running, restore exactly the same
environment variables from sections 1–2, narrow `SIGN_DYNAMICS_SEEDS` or
`SIGN_DYNAMICS_ARMS` to the affected jobs if desired, and run:

```bash
"$PYTHON_BIN" scripts/run_tinystories_sign_dynamics.py preflight \
  --python-bin "$PYTHON_BIN"
"$PYTHON_BIN" scripts/run_tinystories_sign_dynamics.py resume \
  --python-bin "$PYTHON_BIN"
```

The unchanged output path makes continuation load `checkpoints/latest.pt`.
Resume validates the support and diagnostic identities, restores all
per-coordinate state, truncates any uncheckpointed journal tail, and rejects
missing historical snapshots.

## 6. Verify exact diagnostic coverage

After a run completes, point `RUN_DIR` to it and require one JSONL event per
committed step plus all declared snapshot identities:

```bash
export RUN_DIR="$MATFORMER_EXPERIMENT_ROOT/tinystories-instruct-sign-dynamics-v1/x${SIGN_DYNAMICS_B_MULTIPLIER}/runs/uniform_h1/s42"

"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
summary = json.loads((run_dir / "run_summary.json").read_text())
records = [json.loads(line) for line in (run_dir / "sign_dynamics.jsonl").read_text().splitlines()]
assert summary["status"] == "completed"
assert len(records) == summary["steps_completed"]
assert [row["step"] for row in records] == list(range(1, len(records) + 1))
assert summary["sign_dynamics_measured_steps"] == summary["steps_completed"]
assert summary["sign_dynamics_step_coverage_complete"] is True
for identity in summary["sign_dynamics_snapshot_identities"].values():
    assert pathlib.Path(identity["path"]).is_file()
print("exact step and milestone coverage verified")
PY
```

## 7. Analyze and generate figures

Analyze one multiplier directory (or the campaign root to group several
multipliers) with a configurable number of time bins:

```bash
export CAMPAIGN_ROOT="$MATFORMER_EXPERIMENT_ROOT/tinystories-instruct-sign-dynamics-v1/x${SIGN_DYNAMICS_B_MULTIPLIER}"
export SIGN_DYNAMICS_ANALYSIS_WORKERS="${SIGN_DYNAMICS_ANALYSIS_WORKERS:-8}"

"$PYTHON_BIN" scripts/analyze_tinystories_sign_dynamics.py \
  --campaign-root "$CAMPAIGN_ROOT" \
  --analysis-dir "$CAMPAIGN_ROOT/analysis" \
  --figures-dir "$CAMPAIGN_ROOT/figures" \
  --time-bins 20 \
  --workers "$SIGN_DYNAMICS_ANALYSIS_WORKERS"
```

The analyzer uses only completed sign-dynamics runs, rejects mixed scientific
contracts, and labels incomplete or single-seed panels provisional. Its primary
analysis joins each ordinary-validation checkpoint to exact sign dynamics over
the preceding validation interval and the cumulative bands used by that model
width. It writes:

- `validation_sign_dynamics.csv`, containing validation improvement, raw and
  robust flips, update-normalized flips, learning rate, relative-update RMS,
  and support exposure for each interval, width, and hysteresis threshold;
- `performance_sign_correlations.csv`, containing concurrent and next-interval
  Pearson, Spearman, and control-adjusted correlations for early, middle, late,
  and post-10% training phases; and
- `run_performance_dynamics.csv`, containing terminal performance, early sign
  dynamics, halfway terminal-sign alignment, and terminal-sign acquisition.

The six `performance_*.png` files plus `terminal_performance_dynamics.png` are
the seven primary figures. They test correlation with validation performance;
they do not rank arms. The original action/band, off-support, and H-window
figures remain secondary diagnostic outputs. Terminal associations are plotted
in separate width panels because nested widths are not independent experimental
replicates. Additional arms and seeds provide the independent variation needed
for cross-run correlation estimates.

Time-bin aggregation runs across the requested CPU workers, while progress
messages identify long journal-read, validation-join, snapshot, table-write,
and plotting phases.

## 8. Add seeds or budget multipliers

No command restructuring is needed. Set a new comma-separated seed list or
positive integer multiplier and repeat sections 3–7:

```bash
export SIGN_DYNAMICS_SEEDS="43,44"
export SIGN_DYNAMICS_B_MULTIPLIER=8

"$PYTHON_BIN" scripts/run_tinystories_sign_dynamics.py preflight --python-bin "$PYTHON_BIN"
"$PYTHON_BIN" scripts/run_tinystories_sign_dynamics.py submit --python-bin "$PYTHON_BIN"
```

These commands create independent `x8` paths and never mutate `x4` results.

This workflow intentionally does not run portfolio reference freezing,
standalone reference training, target-manifest generation, candidate discovery,
catch-up qualification, or portfolio/final-holdout commands. Ordinary
validation remains enabled; the sealed final holdout is not consumed.
