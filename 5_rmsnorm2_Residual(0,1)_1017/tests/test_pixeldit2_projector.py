"""Regression checks against the adjacent official PixelDiT2 source tree."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from pixeldit2_projector import TimestepProjection


@pytest.fixture
def upstream_projector(monkeypatch):
    source = Path(__file__).resolve().parents[2] / "PixelDiT2"
    if not (source / "pixdit_core" / "grounding.py").is_file():
        pytest.skip("Official PixelDiT2 checkout is required for parity checks")
    monkeypatch.syspath_prepend(str(source))
    from pixdit_core.grounding import TimestepProjection as OfficialProjection
    return OfficialProjection


def test_b_projector_initialization_matches_official(upstream_projector):
    kwargs = dict(in_dim=768, out_dim=768, hidden_size=768, num_heads=12)
    torch.manual_seed(123)
    actual = TimestepProjection(**kwargs)
    torch.manual_seed(123)
    expected = upstream_projector(**kwargs)
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)
    assert len(actual.blocks) == 1
    assert not hasattr(actual, "input_norm")


@pytest.mark.parametrize("grid_hw,num_blocks", [((2, 3), 1), ((3, 2), 2)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward_and_gradients_match_official(upstream_projector, grid_hw, num_blocks, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for BF16 autocast parity")
    torch.manual_seed(321)
    kwargs = dict(in_dim=24, out_dim=32, hidden_size=48, num_heads=3, num_blocks=num_blocks)
    expected = upstream_projector(**kwargs)
    # Open the initially zero gates so attention, RoPE and time conditioning
    # contribute to both output and gradients, as they do after training starts.
    for block in expected.blocks:
        nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.1)
        nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.1)
    actual = TimestepProjection(**kwargs)
    actual.load_state_dict(expected.state_dict(), strict=True)
    actual, expected = actual.to(device), expected.to(device)
    tokens = torch.randn(3, grid_hw[0] * grid_hw[1], 24, device=device)
    timesteps = torch.tensor([0.0, 0.4, 1.0], device=device)
    a_tokens, e_tokens = tokens.clone().requires_grad_(), tokens.clone().requires_grad_()
    a_t, e_t = timesteps.clone().requires_grad_(), timesteps.clone().requires_grad_()
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        a_out = actual(a_tokens, a_t, grid_hw)
        e_out = expected(e_tokens, e_t, grid_hw)
    torch.testing.assert_close(a_out, e_out, rtol=0, atol=0)
    a_out.float().square().mean().backward()
    e_out.float().square().mean().backward()
    torch.testing.assert_close(a_tokens.grad, e_tokens.grad)
    torch.testing.assert_close(a_t.grad, e_t.grad)
    assert a_tokens.grad.abs().sum() > 0
    assert a_t.grad.abs().sum() > 0
    for name, param in actual.named_parameters():
        reference = dict(expected.named_parameters())[name]
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        torch.testing.assert_close(param.grad, reference.grad)
