import pytest
import torch

from util.model_util import RMSNorm


def test_affine_free_has_no_parameters_buffers_or_checkpoint_entries():
    norm = RMSNorm(768, elementwise_affine=False)
    assert norm.weight is None
    assert list(norm.parameters()) == []
    assert list(norm.buffers()) == []
    assert dict(norm.state_dict()) == {}


@pytest.mark.parametrize("width", [3, 768])
def test_affine_free_normalizes_only_the_last_dimension(width):
    tokens = torch.linspace(0.5, 2.0, 2 * 4 * width).reshape(2, 4, width)
    norm = RMSNorm(width, elementwise_affine=False)
    actual = norm(tokens)
    mean_square = tokens.square().mean(dim=-1, keepdim=True)
    expected = tokens * torch.rsqrt(mean_square + 1e-6)
    torch.testing.assert_close(actual, expected)
    expected_l2 = torch.sqrt(width * mean_square.squeeze(-1) / (mean_square.squeeze(-1) + 1e-6))
    torch.testing.assert_close(actual.norm(dim=-1), expected_l2)

    changed = tokens.clone()
    changed[1] *= 100
    changed[0, 1:] *= 0.001
    torch.testing.assert_close(norm(changed)[0, 0], actual[0, 0], rtol=0, atol=0)


def test_reused_and_separate_stateless_instances_are_equivalent():
    dino = torch.arange(1, 25, dtype=torch.float32).reshape(2, 4, 3)
    jit = dino.flip(-1) * 10
    shared = RMSNorm(3, elementwise_affine=False)
    dino_norm = RMSNorm(3, elementwise_affine=False)
    jit_norm = RMSNorm(3, elementwise_affine=False)
    torch.testing.assert_close(shared(dino), dino_norm(dino), rtol=0, atol=0)
    torch.testing.assert_close(shared(jit), jit_norm(jit), rtol=0, atol=0)


def test_parameter_free_normalization_still_backpropagates_to_input():
    tokens = torch.arange(1, 25, dtype=torch.float32).reshape(2, 4, 3).requires_grad_()
    norm = RMSNorm(3, elementwise_affine=False)
    norm(tokens).sum().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    assert tokens.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_affine_free_preserves_dtype_and_handles_zero_or_small_tokens(dtype):
    tokens = torch.tensor([[[0.0, 0.0, 0.0], [1e-5, 2e-5, 3e-5], [0.5, 1.0, 2.0]]], dtype=dtype)
    norm = RMSNorm(3, elementwise_affine=False)
    actual = norm(tokens)
    fp32_tokens = tokens.float()
    expected = (fp32_tokens * torch.rsqrt(fp32_tokens.square().mean(-1, keepdim=True) + 1e-6)).to(dtype)
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual[0, 0].count_nonzero() == 0
    assert actual.float().norm(dim=-1).max() <= 3**0.5 + 0.01


def test_affine_default_and_learned_internal_gains_are_unchanged():
    norm = RMSNorm(3)
    assert norm.weight.requires_grad
    assert sum(parameter.numel() for parameter in norm.parameters()) == 3
    assert set(norm.state_dict()) == {"weight"}
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.5, 1.0, 2.0]))
    tokens = torch.tensor([[[1.0, 2.0, 3.0]]])
    expected = tokens * torch.rsqrt(tokens.square().mean(-1, keepdim=True) + 1e-6) * norm.weight
    torch.testing.assert_close(norm(tokens), expected)
