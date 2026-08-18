# Quickstart: PanelGrad Sampling

These commands were verified against the implementation. PanelGrad remains opt-in and does not expand the default pilot queue.

PanelGrad defaults to the existing RMS importance score. To opt into the raw
aggregate L2 norm without adding any controller evaluations:

```yaml
model:
  panelgrad:
    importance_metric: gradient_l2
```

Supported values are `gradient_rms` and `gradient_l2`. Checkpoint resumes must
use the same metric. State schema versions 1 and 2 are legacy RMS states and
migrate only to `gradient_rms`; RMS and L2 checkpoints are intentionally not
cross-compatible because they contain metric-specific probability snapshots.

## 1. Inspect resolved configuration

```bash
python train.py \
  --config tests/fixtures/panelgrad_smoke.yaml \
  --preflight
```

Confirm:

- `granularity_sampling_mode=adaptive_global` and strategy `panelgrad`;
- method family/version and global scope;
- defaults `H=50`, `eta=1e-12`, `T=1`, `epsilon=0.1` unless overridden;
- the selected raw aggregate-controller-gradient RMS or L2 importance over granularity-controlled FFN support;
- fixed controller/final-holdout contracts and `panelgrad_sampling` seed stream.

To opt into a refresh-boundary linear epsilon schedule, replace scalar `epsilon`
in a PanelGrad YAML with:

```yaml
epsilon_schedule:
  type: linear
  start: 0.5
  end: 0.1
  duration_steps: 24415
```

The initial refresh uses `0.5`; later refreshes interpolate from committed
PanelGrad optimizer steps, excluding warmup and failed attempts.

Inspect normalized fields without loading data:

```bash
python -c '
import json
from src.utils.config import resolve_run_config

c = resolve_run_config("tests/fixtures/panelgrad_smoke.yaml")
print(json.dumps({
    "mode": c["model"]["granularity_sampling_mode"],
    "strategy": c["model"]["adaptive_sampler_strategy"],
    "panelgrad": c["model"]["panelgrad"],
}, indent=2, sort_keys=True))
'
```

## 2. Validate score and categorical mathematics

```bash
pytest -q tests/test_panelgrad.py -k "score or probability or categorical or state"
```

Coverage includes one granularity, all-zero scores, temperature, `epsilon` at 0 and 1, invalid/non-finite inputs, sum-to-one tolerance, deterministic draws, exposure counts, and transaction rollback.

## 3. Validate controlled FFN support

```bash
pytest -q tests/test_matformer_prefixes.py tests/test_panelgrad.py -k "panelgrad or controlled_support or correction_suspension"
```

Confirm for slicing and concat:

- `N_g` follows resolved prefix/block definitions;
- gate/up/down controlled coordinates are included;
- shared down bias, embeddings, attention, and other common parameters are excluded;
- zero gradients do not change `N_g`;
- membership correction is bypassed only during measurement and restored afterward.

## 4. Run a short PanelGrad smoke

```bash
python train.py \
  --config configs/opt-in_exps/panelgrad_smoke.yaml \
  --override model.panelgrad.refresh_interval_steps=2 \
  --override training.max_steps=5 \
  --override run.run_id=panelgrad-smoke \
  --output-dir outputs/panelgrad-smoke
```

Expected lifecycle:

- refresh at completed step 0;
- one independent categorical global action at steps 1 and 2 using the first frozen `p`;
- refresh before the step-3 action;
- another refresh before step 5;
- no unused terminal refresh after step 5.

Inspect:

```bash
python -m json.tool outputs/panelgrad-smoke/controller_summary.json
sed -n '1,10p' outputs/panelgrad-smoke/controller_metrics.jsonl
sed -n '1,10p' outputs/panelgrad-smoke/metrics.csv
```

Verify refresh records contain active epsilon/schedule progress, ordered `N_g`, gradient norm/RMS, `q`, `p`, entropy/extrema, controller totals, and measurement cost. Training rows remain compact and contain the sampled granularity, its probability, exposure counts, and interval progress.

## 5. Validate measurement isolation and boundaries

```bash
pytest -q tests/test_training_smoke.py tests/test_panelgrad.py -k "panelgrad and (isolation or boundary or warmup or failure)"
```

The tests must show that measurement changes no model/optimizer/scheduler/training-data state, restores model/correction/granularity/RNG state, leaves gradients empty, counts only successful optimizer steps, and refreshes immediately after optional balanced warmup.

## 6. Validate exact resume

```bash
pytest -q tests/test_panelgrad_resume.py
```

Compare uninterrupted and resumed runs inside an interval and at `refresh_pending`. Subsequent actions and exposures are exact; scores and probabilities match at `rtol=1e-6`, `atol=1e-8`. Config, support, role, journal, RNG, and method mismatches must fail rather than reinitialize.

## 7. Validate distributed agreement

```bash
pytest -q tests/test_distributed.py -k panelgrad
```

Confirm replicated controller batches produce matched FSDP backward counts, per-layer full-gradient support never summons the full model, every rank agrees numerically, rank zero owns categorical sampling, all ranks train the same action, and only rank zero writes shared artifacts.

## 8. Run compatibility coverage

```bash
pytest -q \
  tests/test_config.py \
  tests/test_adaptive_sampler.py \
  tests/test_probabilistic_controller.py \
  tests/test_probabilistic_controller_resume.py \
  tests/test_training_smoke.py \
  tests/test_artifacts.py \
  tests/test_reporting.py \
  tests/test_data_validation.py
```

Existing Thompson, UCB, random global/per-block, nested-all, standalone, checkpoint, reporting, and data-role behavior must remain unchanged when PanelGrad is not selected.

## 9. Run the first comparison

Preflight both opt-in inputs:

```bash
python train.py --config configs/opt-in_exps/panelgrad_smoke.yaml --preflight
python train.py --config configs/opt-in_exps/panelgrad_uniform_baseline.yaml --preflight
```

Run the deterministic synthetic-data smoke and matched three-step comparison:

```bash
python -m scripts.run_panelgrad_comparison \
  --output-root outputs/panelgrad-comparison
python -m json.tool outputs/panelgrad-comparison/comparison.json
```

The runner resolves the two opt-in configurations with the same small model, four data-role manifests, root seed, optimizer, scheduler, validation settings, and exactly three completed optimizer steps. It fails if any role-manifest hash differs. Its explicit selection rule uses each final in-memory model after step 3, then evaluates the untouched final holdout once. The resulting artifact reports:

- per-granularity validation/final-holdout results;
- sampled exposure counts and the final sampling probabilities;
- training steps/tokens and wall time;
- PanelGrad refresh count, controller backward evaluations/examples/tokens, and cumulative measurement duration.

Treat this as matched-step or matched-token evidence. Do not call it matched-compute unless the separate measurement work is included in the comparison budget.
