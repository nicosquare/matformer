# Gradient and learning-rate membership corrections with AdamW

Date: 2026-09-10

This note collects the discussion of gradient membership correction (GMC),
learning-rate membership correction (LMC), and the discrepancy found in the
original single-width LMC path. It uses ordinary text, numerical examples, and
code excerpts. It does not require a mathematical renderer.

The original equations are also available as a [rendered image](/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/reports/gradient-membership-explanation-20260910/adamw_gradient_scaling.png) and [PDF](/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-v1/reports/gradient-membership-explanation-20260910/adamw_gradient_scaling.pdf).

## Implementation correction (2026-09-10)

The numerator mismatch is now fixed in the working tree. LMC reuses the model's
configured GMC membership factors, instead of calculating a new numerator from
the number of losses evaluated in the current update:

```python
scales = module.gradient_membership_correction_scales
```

With all four widths configured, both single-width and all-width updates now use
these factors on active blocks:

| FFN quarter | Before fix: single-width LMC | Corrected LMC, either selection mode |
| --- | ---: | ---: |
| A | 0.25 | 1 |
| B | About 0.333 | About 1.333 |
| C | 0.5 | 2 |
| D | 1 | 4 |

LMC reuses GMC factors for trained-width subsets as well: the canonical subset
m/xl uses factors 1, 1, 2, 2. Inactive concat blocks remain unchanged. The correction
continues to scale the whole AdamW parameter change, including weight decay;
existing GMC hooks and clipping behavior remain in place.

The sections below retain the diagnosis of the **pre-fix implementation**.
Historical code excerpts and line numbers describe the checkout before this fix.
No correction training experiment has been launched.

## The main idea

**Making every gradient four times larger usually does not make AdamW's weight changes four times larger.** AdamW adjusts for the size of the gradients it has been receiving.

## A multiply-by-four example

AdamW maintains two running averages for each weight:

1. An average of recent gradients, which includes their direction.
2. An average of recent squared gradients, which measures their size.

To calculate a weight change, AdamW divides the first average by the square root of the second average, with a small stability constant added to the denominator. It then applies the learning rate.

Suppose we multiply every gradient by four, starting from the first update:

| Quantity | What happens |
| --- | --- |
| Gradient | Becomes four times larger |
| Average gradient | Becomes four times larger |
| Average squared gradient | Becomes sixteen times larger |
| Square root of that second average | Becomes four times larger |
| Division used to calculate the update | Four times larger divided by four times larger |
| Final gradient-driven weight change | Approximately unchanged |

**The extra factor appears both above and below the division, so it cancels.**

Ignoring the small stability constant and rounding, this cancellation is exact when the positive multiplier stays fixed, optimizer averages start at zero, and clipping or other processing does not break that fixed relationship. The comparison assumes matching initial weights, learning rates, and parameter-activation patterns. AdamW's separate weight-decay operation is unchanged if its learning rate and decay setting stay unchanged.

## What this means for our membership correction

The first FFN quarter is used by all four widths. The last quarter is used only by the largest width. The correction tries to compensate by multiplying their gradients differently:

| Quarter | Widths that use it | Gradient scaling |
| --- | ---: | --- |
| A | 4 | Unchanged |
| B | 3 | About 1.33 times larger |
| C | 2 | Twice as large |
| D | 1 | Four times larger |

These are fixed multipliers for a fixed set of trained widths. AdamW adjusts each individual weight separately, so giving different quarters different multipliers does not avoid the cancellation.

**The intended amplification is therefore largely lost.** Quarter D still sees fewer batches, and multiplying its gradients by four does not create four times as many learning opportunities.

## Why enabling the correction can still change a real run

- **Gradient clipping:** our correction runs before clipping. Increasing some gradients can make clipping stronger, including for other parameters. The effective scaling can then vary from one update to the next, so the simple cancellation no longer applies.
- **Turning it on midway:** the running averages still contain earlier, unscaled gradients. They need time to adapt, producing a temporary difference.
- **Very small gradients:** AdamW adds a small stability constant, called epsilon, to the denominator. This prevents perfect cancellation and can matter when gradients are tiny.
- **Changing multipliers:** the argument assumes the same positive multiplier for each weight throughout its history. A multiplier that changes over time is a different intervention.
- **Rounding:** computer arithmetic introduces small numerical differences.

Thus, the claim is that fixed gradient scaling is not a reliable way to multiply AdamW's updates. It is not a claim that every practical run must remain identical when correction is enabled.

## What would actually make the update larger?

Let AdamW calculate its normalized gradient-driven weight change first. Then multiply that calculated change by the desired factor.

For example, if AdamW proposes moving a weight by 0.001, multiplying that proposed change by four gives a change of 0.004. The multiplier comes after the normalization, so it survives.

Increasing the learning rate also increases the update. However, in standard AdamW it also increases weight decay. Scaling only the gradient-driven change and increasing the entire learning rate therefore have different effects unless decay is adjusted deliberately.

Neither option replaces missing examples or guarantees that competing widths stop interfering with one another.

## Learning-rate correction as compensation for update frequency

Under uniform sampling of one of four nested widths, the quarters participate
with different frequencies. One proposed correction keeps quarter A at the base
learning rate and increases the rates for less frequently active quarters:

| Quarter | Expected fraction of updates where active | Proposed learning-rate multiplier | Effective peak rate if the base peak is 0.008 |
| --- | ---: | ---: | ---: |
| A | 100% | 1 | 0.008 |
| B | 75% | About 1.333 | About 0.010667 |
| C | 50% | 2 | 0.016 |
| D | 25% | 4 | 0.032 |

Unlike fixed gradient scaling, these factors survive AdamW's normalization.
Holding the current gradients and optimizer state fixed, doubling the learning
rate doubles that step's gradient-driven parameter change. It does not imply
that an entire training trajectory remains a scaled copy: changed weights lead
to different future gradients and optimizer statistics.

This balances **activation frequency multiplied by learning rate**, assuming
uniform width sampling. It does not establish equal actual update magnitudes or
equal learning across quarters. For nonuniform sampling, activation probabilities
would need to be computed from the sampling distribution rather than just the
number of widths containing each quarter.

There are several limits to the budget-compensation intuition:

- One large step is not four successive small steps. Successive steps see new
  batches and recompute gradients after the parameters change.
- A rarely active concat block still sees fewer examples and advances its AdamW
  statistics less often. Increasing the learning rate does not change that clock.
- Larger steps may overshoot. A fourfold multiplier raises the peak effective
  rate from 0.008 to 0.032 in this example; the benefit and stability are empirical
  questions.
- Standard AdamW learning-rate scaling changes both the adaptive update and weight
  decay. A correction intended to affect only the adaptive part must handle decay
  separately.
- Correcting update size does not directly resolve conflicting objectives among
  widths sharing weights.

Thus LMC has a mechanism that survives normalization, but remains a heuristic
for compensating exposure. Its success at matching the largest standalone is not
guaranteed.

## What the existing LMC code actually does

The current implementation does not directly change an optimizer parameter
group's learning rate. It saves each affected concat block before the step,
executes the optimizer, and then scales the resulting parameter change.

In [steps.py](../src/training/steps.py), `_maybe_apply_concat_lmc_optimizer_step`
at lines 297–309 calls:

```python
snapshots = _capture_concat_lmc_snapshots(model)
optimizer.step()
_apply_concat_lmc_corrections(snapshots)
```

The application helper at lines 289–294 performs:

```python
base_delta = pre_step_value - param.data
param.data.copy_(pre_step_value - (base_delta * scale))
```

In words: measure how far the optimizer moved the parameter, multiply that
distance, and place the parameter at the adjusted position.

For ordinary AdamW, this scales the **whole parameter change, including weight
decay**. It is equivalent to a block-specific learning-rate multiplier for that
step, up to numerical rounding, given the same incoming gradients and optimizer
state. It does not directly rescale the stored AdamW moments. Subsequent moments
can nevertheless differ because the adjusted weights produce different gradients.

The helper applies to the concat gate, up, and down weight blocks and their
block-local gate/up biases. The common output bias and non-FFN parameters receive
no direct LMC multiplier. They can still be affected indirectly through clipping
and the changed model trajectory.

## Pre-fix single-width discrepancy: the same variable name had two meanings

**Before the fix, GMC and LMC used different numerators in their membership factors.** Both were
called `total_losses` in the code, but they were calculated in different places.

### GMC counts the configured trained widths

In [granularity.py](../src/models/granularity.py),
`get_concat_gradient_membership_correction_scales_from_metadata`, lines 349–364:

```python
total_losses = len(granularities)
scales = []
for membership_count in membership_counts:
    if membership_count == 0:
        scales.append(1.0)
        continue
    scales.append(total_losses / membership_count)
```

The model passes its configured trained widths into this calculation. With all
four widths configured, the numerator is **four**, even when the training step
selects only one width. The slicing metadata calculation uses the same principle.

In [ffn.py](../src/models/ffn.py), lines 614–621, the concat hook applies the
result to the gradient:

```python
param.register_hook(
    lambda grad, scale=scale: self._scale_block_grad(grad, scale)
)

def _scale_block_grad(self, grad, scale):
    if not self._membership_correction_is_active():
        return grad
    return grad * scale
```

### Before the fix, LMC received the number of width losses evaluated in the current update

In [steps.py](../src/training/steps.py), `_forward_backward_microbatch` has two
relevant return paths. For the all-width path, lines 718–740:

```python
if action["kind"] == "nested_all":
    total_losses = len(selected)
    # Evaluate and backpropagate each selected width.
    # ...
    return metric_data, total_losses
```

The single-width path ends at lines 763–764 with:

```python
(outputs.loss * scale).backward()
return metric_data, 1
```

The training loop assigns this return value to `total_losses` at line 1080 and
passes it to `_maybe_apply_concat_lmc_optimizer_step` at lines 1191–1195.

Inside `_capture_concat_lmc_snapshots`, lines 254–258:

```python
counts = list(getattr(module, "gradient_membership_counts", []))
scales = [
    (float(total_losses) / float(count)) if int(count) > 0 else 1.0
    for count in counts
]
```

With one sampled width, the numerator is **one**. The denominator still counts
membership across **all configured trained widths**: four, three, two, and one.

### Resulting factors

| Quarter | Membership count | GMC gradient factor | Pre-fix LMC parameter-change factor: one width per update | Pre-fix LMC parameter-change factor: all four widths per update |
| --- | ---: | ---: | ---: | ---: |
| A | 4 | 1 | 0.25 | 1 |
| B | 3 | About 1.333 | About 0.333 | About 1.333 |
| C | 2 | 2 | 0.5 | 2 |
| D | 1 | 4 | 1 | 4 |

For inactive concat blocks, no gradient means no AdamW parameter change in the
first place. The table describes the factor applied to a change when one occurs.

**If the intention is to apply GMC's same membership factors after AdamW
normalization, the single-width LMC implementation does not implement that
intention.** The numerator would need to use the same configured trained-width
set used for the membership counts, rather than the number evaluated that step.

The factors coincide when all four widths are evaluated together, which can hide
the discrepancy. AdamW does not require these numerators to differ; this behavior
comes from the implementation's choice of normalization.

## Boosting rare quarters versus slowing frequent quarters

The pre-fix single-width LMC leaves quarter D at the base step size and reduces
the others. The proposed inverse-activation-frequency correction leaves quarter
A at the base size and boosts the others.

These normalizations preserve the same relative ratios between FFN quarters,
but every pre-fix single-width LMC factor is four times smaller than its
corresponding GMC-style factor. Since common parameters retain their base update,
the two choices also create different FFN-to-common update ratios.

Under the simplified frequency-times-rate accounting:

- The boost-rare-quarters choice gives each quarter the equivalent of the full
  base-rate opportunity per global update, in expectation.
- The slow-frequent-quarters choice gives each quarter one quarter of that
  opportunity, in expectation.
- Common parameters are active every update and are not directly slowed by LMC.

These are distinct interventions, not interchangeable ways to describe one
experiment. Slowing frequent quarters might reduce disruptive changes; boosting
rare quarters might help them adapt faster. Neither benefit has been established
by this code inspection. The authorized fix now implements the stated goal of
using the same factors as GMC after normalization.

## Existing LMC also enables the GMC hooks

In [config.py](../src/utils/config.py), lines 1733–1746, an explicit correction
mode sets:

```python
resolved_mode = _normalize_correction_mode(requested_mode)
membership_correction = resolved_mode != "none"
```

Therefore selecting `lmc` sets `membership_correction` to true.
[modeling.py](../src/training/modeling.py), lines 63–68, passes that flag as
`gradient_membership_correction_enabled` when constructing the FFNs.

The existing mode consequently combines:

1. Fixed membership scaling of gradients during backward.
2. Gradient clipping, which sees those corrected gradients.
3. The optimizer step.
4. Membership scaling of the completed parameter change.

It is not an isolated learning-rate-only intervention. Even if AdamW mostly
cancels the fixed gradient factors, the clipping interactions discussed above
remain relevant. A comparison intended to isolate post-normalization correction
should explicitly decide whether gradient correction is disabled or retained.

## Original verification and test coverage limitation

A CPU inspection instantiated the actual concat model through the resolved
`lmc` configuration and called the snapshot helper with one and four losses.
It confirmed:

```text
LMC membership correction enabled: True
Membership counts: [4, 3, 2, 1]
Gradient hook factors: [1, 1.3333333333333333, 2, 4]
Parameter-change factors with one loss: [0.25, 0.3333333333333333, 0.5, 1]
Parameter-change factors with four losses: [1, 1.3333333333333333, 2, 4]
```

Before the fix, the test
`test_concat_lmc_applies_block_specific_effective_learning_rates_without_changing_gradients_or_optimizer_state`
in [test_training_smoke.py](../tests/test_training_smoke.py) checked the larger
factors using a toy model. Its `_run_concat_lmc_case` helper explicitly forced
all four widths to be selected. Consequently, that test did not establish that
the same factors were used in single-width training. Its unchanged-gradient
assertion also did not demonstrate that the real model's resolved LMC mode
disables GMC hooks.

That initial discussion did not change training code. The subsequent authorized
fix adds single-width cases for all four widths to the training-loop regression.
Before the code change, those four cases failed while the all-width case passed.

A new [real-model regression](../tests/test_learning_rate_membership_correction.py)
compares LMC against explicit per-block AdamW learning rates over three updates.
It covers single-width and all-width selection, canonical trained-width subsets,
biases, nonzero weight decay, unchanged optimizer moments/counters, and inactive tails
after an earlier full-width update. Both models retain the real GMC hooks.

Post-fix CPU verification: **396 passed, 28 skipped, 1 expected failure** across
the LMC regression, training smoke, MatFormer prefix, model-size, optimizer
ownership, resume, and campaign tests. GPU execution was disabled for this check.
`git diff --check` also passed.

A separate existing metadata limitation was uncovered while constructing these
tests: with all four custom labels g250/g500/g750/g1000 in the model geometry but
only g500/g1000 passed as `trained_granularities`, the shared membership helper
uses positions within that subset instead of the full geometry. This gives
counts 2, 1, 0, 0 instead of 2, 2, 1, 1. The numerator fix deliberately reuses
GMC's factors; it does not repair that separate denominator/mapping issue. The
canonical m/xl subset is covered by the tests and uses the correct geometry.
The completed campaign trained all four labels, so this subset limitation does
not affect that configuration.

## Connection to the largest-standalone gap

The completed elastic methods did not reach the largest standalone despite four
times its total token budget. The full width was selected only about as often as
the largest standalone was trained; the other updates trained smaller widths.

GMC's fixed gradient amplification is largely normalized away by AdamW. LMC can
change effective update sizes. Its pre-fix single-width normalization slowed
frequent FFN quarters; the corrected factors boost rarely active quarters. Neither
operation adds missing examples or directly removes competing demands on shared weights.

The LMC discrepancy cannot explain the completed campaign's observed gap:
**all five elastic arms used correction mode `none`**. It matters when interpreting
the correction mechanism and planning a separate correction comparison.

## Scope and references

The completed optimizer-ownership campaign had membership correction disabled.
This note records mathematical reasoning, code inspection, a small CPU
configuration/helper probe, and the subsequent numerator fix with regression
tests. It does not report a new correction training experiment, authorize
launching jobs, or evaluate the sealed holdout. Source
line numbers refer to the checkout inspected on the date above.

- [AdamW algorithm documentation for PyTorch 2.11](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html)
- [Membership scales](../src/models/granularity.py)
- [Gradient correction hooks](../src/models/ffn.py)
- [Clipping and optimizer ordering](../src/training/steps.py)
- [Campaign results analysis](tinystories_optimizer_ownership_results_analysis_2026-09-10.md)
