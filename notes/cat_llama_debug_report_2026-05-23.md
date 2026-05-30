# Cat Llama Debugging Report

Date: 2026-05-23

## Goal

Investigate why `cat_llama` underperformed `matformer_llama`, especially in
distributed/FSDP runs, and verify whether gradient membership correction (GMC)
was being applied.

## Main Findings

1. `cat_llama` and `matformer_llama` are effectively equivalent on 1 GPU at
   baseline settings, and `cat_llama` can outperform the slicing version at a
   higher learning rate.
2. The large degradation appears when moving to 4-GPU FSDP runs, not in the
   plain non-distributed path.
3. GMC is present in resolved configs and saved run configs, but it has almost
   no practical effect under AdamW.
4. `use_orig_params=True` in FSDP did not resolve the cat vs dense gap.
5. The strongest remaining explanation is AdamW sensitivity to parameterization
   under FSDP, not a forward mismatch.

## Code Changes Made During Debugging

### Membership correction

- Added configurable `model.gradient_membership_correction`.
- Extended GMC to:
  - `CatLlamaMLP` with block-level hooks.
  - `ModifiedLlamaMLP` with slice-segment scaling.

### FSDP wrapper

- Updated config-driven FSDP wrapper in `training/distributed.py` to use
  `use_orig_params=True`.
- Result: this did not materially change the dense vs cat behavior in the
  distributed diagnostic.

### Diagnostic script

- Added `scripts/diagnose_fsdp_mlp_equivalence.py`.
- It compares one `nested-all` step between `ModifiedLlamaMLP` and
  `CatLlamaMLP` from matched initialization and reports:
  - loss difference
  - per-granularity output difference
  - reconstructed gradient difference before optimizer step
  - reconstructed parameter difference after optimizer step

## Experiment 1: Full 1B-token pilot comparison

Location:
`/nfs-stor/nicolas.avila/results/matformer/matformer_llama_148m_1b_tokens`

### Nested-all

`matformer_llama`

- `s`: loss `1.7582518607`, ppl `5.8022853206`
- `m`: loss `1.7220680267`, ppl `5.5960893680`
- `l`: loss `1.6995684206`, ppl `5.4715854588`
- `xl`: loss `1.6775543839`, ppl `5.3524499136`

`cat_llama`

- `s`: loss `1.8817423135`, ppl `6.5649330751`
- `m`: loss `1.8824448884`, ppl `6.5695470526`
- `l`: loss `1.8857127279`, ppl `6.5910503939`
- `xl`: loss `1.8893649578`, ppl `6.6151664371`

### Nested-random

`matformer_llama`

- `s`: loss `1.8047418445`, ppl `6.0784020732`
- `m`: loss `1.7761931866`, ppl `5.9073254747`
- `l`: loss `1.7566232234`, ppl `5.7928431932`
- `xl`: loss `1.7495296896`, ppl `5.7518968626`

`cat_llama`

- `s`: loss `1.9069436491`, ppl `6.7324804983`
- `m`: loss `1.8987729698`, ppl `6.6776956793`
- `l`: loss `1.9035996199`, ppl `6.7100044882`
- `xl`: loss `1.9071858972`, ppl `6.7341116270`

### Conclusion

The degradation was large enough that it was unlikely to be explained only by
membership correction or late-training dynamics.

## Experiment 2: Short 4-GPU nested-all comparison, 200 steps, batch size 4

Location:
`/nfs-stor/nicolas.avila/results/matformer/compare_short/matformer_llama_148m_100m_tokens`

This was used to check whether the gap appears early and whether GMC changes
anything.

### Initial 4-GPU short runs

`cmp-matformer-na-bs4-200`

- `s`: loss `5.9416668415`, ppl `380.5687489401`
- `m`: loss `5.9416203499`, ppl `380.5510560926`
- `l`: loss `5.9419463873`, ppl `380.6751502007`
- `xl`: loss `5.9395329952`, ppl `379.7575395357`

`cmp-cat-na-bs4-200`

- `s`: loss `8.6197519302`, ppl `5540.0119137578`
- `m`: loss `8.3917510509`, ppl `4410.5340061899`
- `l`: loss `8.1300125122`, ppl `3394.8420418421`
- `xl`: loss `8.2291573286`, ppl `3748.6735208161`

`cmp-matformer-na-bs4-200-gmc`

- `s`: loss `5.9415638447`, ppl `380.5295535854`
- `m`: loss `5.9415031672`, ppl `380.5064646931`
- `l`: loss `5.9418666363`, ppl `380.6447921818`
- `xl`: loss `5.9394506216`, ppl `379.7262588212`

`cmp-cat-na-bs4-200-gmc`

- `s`: loss `8.6197531223`, ppl `5540.0185179706`
- `m`: loss `8.3917508125`, ppl `4410.5329546368`
- `l`: loss `8.1300131083`, ppl `3394.8440653262`
- `xl`: loss `8.2291570902`, ppl `3748.6726270628`

### Conclusion

- `cat_llama` was already much worse by step 200.
- GMC had effectively no observable effect.

## Experiment 3: Verify whether GMC override reached saved configs

Observed:

- `model.gradient_membership_correction=true` was present in the saved
  `config.json`.
- `training.batch_size_per_process=4` was also present.
- The custom `run.run_id` passed through `--override` did not survive the Slurm
  wrapper. The wrapper reused the default run id for the selected mode.

### Conclusion

The GMC field was being resolved and written to config. The lack of effect was
not caused by missing config plumbing.

## Experiment 4: 1-GPU short nested-all comparison, 200 steps, batch size 4

The 1-GPU jobs overwrote the same `compare_short` run ids. The saved configs
showed:

- `distributed.enabled = false`
- `strategy = none`
- `world_size = 1`

### 1-GPU results

`cmp-matformer-na-bs4-200`

- `s`: loss `6.1156811714`, ppl `452.9044478871`
- `m`: loss `6.1142940521`, ppl `452.2766509048`
- `l`: loss `6.1149659157`, ppl `452.5806212056`
- `xl`: loss `6.1119823456`, ppl `451.2323275613`

`cmp-matformer-na-bs4-200-gmc`

- `s`: loss `6.1155247688`, ppl `452.8336179985`
- `m`: loss `6.1142396927`, ppl `452.2520660693`
- `l`: loss `6.1149845123`, ppl `452.5890377668`
- `xl`: loss `6.1119427681`, ppl `451.2144692744`

`cmp-cat-na-bs4-200`

- `s`: loss `6.1156306267`, ppl `452.8815565286`
- `m`: loss `6.1143064499`, ppl `452.2822581597`
- `l`: loss `6.1149582863`, ppl `452.5771683026`
- `xl`: loss `6.1119680405`, ppl `451.2258726772`

`cmp-cat-na-bs4-200-gmc`

- `s`: loss `6.1156520844`, ppl `452.8912744168`
- `m`: loss `6.1142883301`, ppl `452.2740629644`
- `l`: loss `6.1149773598`, ppl `452.5858006094`
- `xl`: loss `6.1119775772`, ppl `451.2301759230`

### Conclusion

- The catastrophic `cat_llama` gap disappears on 1 GPU.
- `cat_llama` and `matformer_llama` are nearly identical without FSDP.
- GMC still has almost no effect.

This strongly localized the main issue to the distributed/FSDP path.

## Experiment 4.5: 1-GPU learning-rate sweep at 2e-4

Location:
`/nfs-stor/nicolas.avila/results/matformer/matformer_llama_148m_100m_tokens_lr_exps/matformer_llama_148m_100m_tokens_lr_0.0002`

These runs keep the single-GPU setup but raise the learning rate from the
baseline 1e-4 to 2e-4.

### Results

`matformer_llama`

- best validation loss: `2.0664424896`
- best checkpoint granularity: `l`

`cat_llama`

- best validation loss: `2.0389015675`
- best checkpoint granularity: `xl`

### Conclusion

- At `lr=2e-4`, `cat_llama` outperformed `matformer_llama` in the single-GPU
  run.
- This makes the earlier cat-vs-slice gap look more hyperparameter-sensitive
  than architectural.
- The open question is therefore not whether cat can work, but why it degrades
  much more sharply than the slicing version when moving from single GPU to
  FSDP.

## Experiment 5: FSDP equivalence diagnostic

Script:
`scripts/diagnose_fsdp_mlp_equivalence.py`

Purpose:

- Build `ModifiedLlamaMLP` and `CatLlamaMLP` from matched initialization.
- Run one `nested-all` forward/backward/step on the same batch.
- Compare:
  - `loss_diff`
  - `output_max_abs_diff`
  - `grad_max_abs_diff`
  - `state_max_abs_diff`

### 5.1 1 GPU, no FSDP, AdamW

From `logs/diag_mlp_1gpu_adamw_77979.out`

- `loss_diff = 0.0`
- `output_max_abs_diff = 0.0` for all granularities
- `grad_max_abs_diff`
  - `down_proj.weight`: `0.0159822199`
  - `gate_proj.weight`: `0.0182928909`
  - `up_proj.weight`: `0.0173670687`
- `state_max_abs_diff`
  - `down_proj.weight`: `2.85e-07`
  - `gate_proj.weight`: `8.90e-05`
  - `up_proj.weight`: `1.91e-04`

### 5.2 1 GPU, no FSDP, SGD

From `logs/diag_mlp_1gpu_sgd_77981.out`

- `loss_diff = 0.0`
- `output_max_abs_diff = 0.0`
- `grad_max_abs_diff`
  - `down_proj.weight`: `0.0159822199`
  - `gate_proj.weight`: `0.0182928909`
  - `up_proj.weight`: `0.0173670687`
- `state_max_abs_diff`
  - `down_proj.weight`: `1.60e-05`
  - `gate_proj.weight`: `1.83e-05`
  - `up_proj.weight`: `1.74e-05`

### Interpretation

Without FSDP:

- Dense and cat have identical forward behavior.
- Raw gradients are not identical.
- One optimizer step still lands them very close.

So the two parameterizations are not gradient-identical, but they remain
practically equivalent in the simple non-distributed setting.

### 5.3 4 GPU, FSDP, AdamW, `use_orig_params=True`

From `logs/diag_mlp_fsdp_orig_adamw_77997.out`

- `loss_diff = 0.0`
- `output_max_abs_diff = 0.0`
- `grad_max_abs_diff`
  - `gate_proj.weight`: `null`
- `state_max_abs_diff`
  - `down_proj.weight`: `0.0010012463`
  - `gate_proj.weight`: `0.0010025054`
  - `up_proj.weight`: `0.0010024905`

### 5.4 4 GPU, FSDP, AdamW, `use_orig_params=False`

From `logs/diag_mlp_fsdp_flat_adamw_77998.out`

- `loss_diff = 0.0`
- `output_max_abs_diff = 0.0`
- `grad_max_abs_diff = {}`
- `state_max_abs_diff`
  - `down_proj.weight`: `0.0010012463`
  - `gate_proj.weight`: `0.0010025054`
  - `up_proj.weight`: `0.0010024905`

### 5.5 4 GPU, FSDP, SGD, `use_orig_params=True`

From `logs/diag_mlp_fsdp_orig_sgd_78000.out`

- `loss_diff = 0.0`
- `output_max_abs_diff = 0.0`
- `grad_max_abs_diff`
  - `gate_proj.weight`: `null`
- `state_max_abs_diff`
  - `down_proj.weight`: `1.56e-05`
  - `gate_proj.weight`: `1.29e-05`
  - `up_proj.weight`: `9.76e-06`

### 5.6 4 GPU, FSDP, SGD, `use_orig_params=False`

From `logs/diag_mlp_fsdp_flat_sgd_78001.out`

- `loss_diff = 0.0`
- `output_max_abs_diff = 0.0`
- `grad_max_abs_diff = {}`
- `state_max_abs_diff`
  - `down_proj.weight`: `1.56e-05`
  - `gate_proj.weight`: `1.29e-05`
  - `up_proj.weight`: `9.76e-06`

### Diagnostic conclusions

1. The forward pass is not the problem.
   - `loss_diff = 0`
   - `output_max_abs_diff = 0`
   in every diagnostic.

2. The dense and cat parameterizations already produce different raw gradients
   on 1 GPU.

3. `use_orig_params=True` vs `False` does not materially change the post-step
   divergence in the diagnostic.

4. The largest post-step mismatch appears specifically with AdamW under FSDP:
   about `1e-3`.

5. With SGD under FSDP, the post-step mismatch shrinks back to about `1e-5`,
   similar to the non-distributed case.

## Interpretation

The collected evidence points to:

- `cat_llama` forward semantics are correct.
- `cat_llama` can match or exceed `matformer_llama` on 1 GPU once the
  learning rate is adjusted.
- GMC is not the main issue.
- `use_orig_params=True` is not the main issue.
- The main confound is the single-GPU to FSDP degradation, especially under
  AdamW.

This means the concat factorization is not a neutral refactor from the point of
view of AdamW in distributed training. Even when the function class and forward
outputs match, the optimizer-state dynamics and/or FSDP interaction differ
enough to produce the observed distributed training gap.

## Practical Conclusions

1. The remaining comparison to explain is not cat vs slice in the abstract, but
   single-GPU versus FSDP behavior for `cat_llama`.

2. If the goal is to compare functionally equivalent models, 1-GPU results show
   that the implementation is basically fine outside the distributed optimizer
   path.

3. For further debugging, the next most useful experiments are:
   - short distributed training with SGD instead of AdamW
   - short distributed training with different optimizer settings
   - inspect AdamW state evolution on corresponding dense vs cat parameters
   - compare the same cat run at 1 GPU and 4 GPU with identical seeds and
     learning rate

## Bottom Line

The investigation does not support “cat_llama is fundamentally wrong.”
Instead, it supports:

`cat_llama` can be competitive or better on 1 GPU at a tuned learning rate, but
it degrades much more sharply than `matformer_llama` when moved into FSDP,
especially under AdamW.
