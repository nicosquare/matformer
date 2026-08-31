# CLI Entrypoints Contract

## Training and Preflight

The existing entrypoint remains authoritative:

```bash
python train.py \
  --config configs/controlled_exps/tinystories_instruct_per_width_optimizers.yaml \
  --override training.optimizer.state_scope=per_granularity \
  --override training.optimizer.scheduler_clock=global_step \
  --preflight
```

Without `--preflight`, the same command follows the normal training path. No
separate per-width trainer is introduced.

### Contract

- Dotted overrides accept both new optimizer fields.
- Preflight prints the resolved optimizer state contract and paired-control
  identity.
- Invalid topology, action scope, width cardinality, scheduler clock, warmup,
  or distributed configuration exits nonzero before training.
- Omitted state scope preserves shared behavior.

## TinyStories-Instruct Profile and Corpus Audit

Reuse the existing profile and preparation tools:

```bash
export TINYSTORIES_PROFILE=instruct
source scripts/select_tinystories_profile.sh

python scripts/audit_prepared_corpus.py \
  --prepared-corpus-dir "$CORPUS" \
  --prepared-tokenizer-dir "$TOKENIZER" \
  --required-vocab-size 2048 \
  --minimum-training-tokens 713785344
```

The controlled recipe requires the fixed prepared TinyStories-Instruct role
manifests. The raw dataset path is not used for submitted paired runs.

## Final-Holdout Evaluation

Reuse the existing explicit evaluator:

```bash
python scripts/evaluate_final_holdout.py \
  --run-dir "$RUN_DIR" \
  --checkpoint "$RUN_DIR/checkpoints/latest.pt" \
  --device cuda \
  --skip-existing
```

### Contract

- The checkpoint argument is the frozen terminal fixed-budget checkpoint.
- The evaluator validates checkpoint and final-role provenance and evaluates all
  four widths.
- It writes `final_holdout_results.json` with checkpoint hash, per-width loss
  and perplexity, and uniform average loss.
- `--skip-existing` skips only a result that validates against the same inputs.

## Freeze a Paired Analysis Manifest

New entrypoint:

```bash
python scripts/analyze_tinystories_per_width_optimizer.py freeze \
  --phase pilot \
  --input-root "$PILOT_ROOT" \
  --output-dir "$PILOT_ROOT/analysis"
```

Confirmation uses `--phase confirmation` and its isolated root.

### Freeze contract

- Locate exactly one completed `shared` and `per_granularity` run for seeds 42,
  43, and 44 using resolved run artifacts, not directory naming alone.
- Accept an explicit repeated `--run-dir` override for nonstandard layouts.
- Verify matched controls, exact action and learning-rate traces, role hashes,
  budgets, counts, and terminal checkpoint purpose/hash.
- Refuse to overwrite a different frozen manifest without an explicit new
  output directory.
- Write a versioned JSON manifest atomically and print its path and hash.

## Produce the Paired Report

```bash
python scripts/analyze_tinystories_per_width_optimizer.py report \
  --manifest "$PILOT_ROOT/analysis/optimizer_state_manifest.json" \
  --output-dir "$PILOT_ROOT/analysis"
```

### Report contract

- Validate the frozen manifest and every referenced artifact hash.
- Produce ordinary-validation/resource diagnostics when holdout results are
  absent.
- Include final-holdout endpoints only when all referenced results validate.
- Emit JSON and CSV with scope, seed, width, endpoints, paired deltas, resource
  costs, holdout status, and claim status.
- Never open or evaluate the final holdout implicitly.

## Submitted Training Commands

The existing Slurm wrapper remains unchanged. The runbook supplies the base
config, isolated output root, run ID, seed, budget, state scope, global scheduler
clock, tokenizer, and corpus through its current argument/override forwarding.

Resubmission uses the exact original command and output directory so the normal
latest-checkpoint continuation policy restores the complete optimizer
collection.
