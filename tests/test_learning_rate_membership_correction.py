"""LMC must match explicit per-block AdamW learning rates on real concat FFNs."""

import pytest
import torch
from transformers import LlamaConfig

from src.models.ffn import CatLlamaMLP
from src.training.steps import _maybe_apply_concat_lmc_optimizer_step


WIDTHS = ("g250", "g500", "g750", "g1000")


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "trained_widths, selected_widths, expected_scales",
    [
        (WIDTHS, (width,), (1.0, 4 / 3, 2.0, 4.0)) for width in WIDTHS
    ] + [
        (WIDTHS, WIDTHS, (1.0, 4 / 3, 2.0, 4.0)),
        (("m", "xl"), ("m",), (1.0, 1.0, 2.0, 2.0)),
        (("m", "xl"), ("m", "xl"), (1.0, 1.0, 2.0, 2.0)),
    ],
)
def test_lmc_matches_per_block_adamw_rates(
    bias, trained_widths, selected_widths, expected_scales,
):
    config = LlamaConfig(
        hidden_size=8, intermediate_size=16, num_attention_heads=2,
        num_key_value_heads=2, mlp_bias=bias,
    )
    widths = ("s", "m", "l", "xl") if "m" in trained_widths else WIDTHS
    config.granularities = list(widths)
    config.granularity_prefixes = dict(zip(widths, (0.25, 0.5, 0.75, 1.0)))
    torch.manual_seed(42)
    actual = CatLlamaMLP(
        config, trained_granularities=trained_widths,
        gradient_membership_correction_enabled=True,
    ).double()
    # Construct separately: deepcopy does not preserve tensor backward hooks.
    reference = CatLlamaMLP(
        config, trained_granularities=trained_widths,
        gradient_membership_correction_enabled=True,
    ).double()
    reference.load_state_dict(actual.state_dict())
    rate = 0.01
    kwargs = dict(betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    optimizer = torch.optim.AdamW(actual.parameters(), lr=rate, **kwargs)
    reference_optimizer = torch.optim.AdamW([
        {
            "params": [parameter],
            "lr": rate * (expected_scales[int(name.rsplit(".", 1)[1])]
                          if "_blocks." in name else 1.0),
        }
        for name, parameter in reference.named_parameters()
    ], lr=rate, **kwargs)

    # Start wide to populate every block's moments, then exercise the requested
    # selection twice. Narrow steps must leave previously used tails untouched.
    for selected in [(widths[-1],), selected_widths, selected_widths]:
        inputs = torch.randn(2, 3, 8, dtype=torch.float64)
        before = {name: p.detach().clone() for name, p in actual.named_parameters()}
        for model, opt in [(actual, optimizer), (reference, reference_optimizer)]:
            opt.zero_grad(set_to_none=True)
            for width in selected:
                model.configure_subnetwork(width)
                (model(inputs).square().mean() / len(selected)).backward()

        _maybe_apply_concat_lmc_optimizer_step(
            {"model": {"correction_mode": "lmc"}}, actual, optimizer,
        )
        reference_optimizer.step()

        max_block = max(widths.index(width) for width in selected)
        for (name, parameter), (reference_name, expected) in zip(
            actual.named_parameters(), reference.named_parameters(), strict=True,
        ):
            assert name == reference_name
            torch.testing.assert_close(parameter, expected, rtol=1e-10, atol=1e-12)
            if "_blocks." in name and int(name.rsplit(".", 1)[1]) > max_block:
                assert parameter.grad is None
                assert torch.equal(parameter, before[name])
            # Post-step rescaling must preserve Adam's moments and counters,
            # including the histories of inactive blocks and the shared bias.
            assert optimizer.state[parameter].keys() == reference_optimizer.state[expected].keys()
            for key, value in optimizer.state[parameter].items():
                torch.testing.assert_close(
                    value, reference_optimizer.state[expected][key],
                    rtol=1e-10, atol=1e-12,
                )
