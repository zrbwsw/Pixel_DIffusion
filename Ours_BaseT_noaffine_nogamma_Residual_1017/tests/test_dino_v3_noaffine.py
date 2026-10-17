"""Regression tests for RAEv2-style final LN, without loading DINO weights."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import dino_v3_lora
from dino_encoder import build_encoder
from dino_lora import LoRALinear
from dino_v3_lora import DINOv3PatchTokenEncoder


class _FakeAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self.proj((q + k + v) / 3)


class _FakeMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _FakeBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = _FakeAttention(dim)
        self.mlp = _FakeMLP(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _FakeDINOv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 8
        self.patch_size = 2
        self.patch_embed = nn.Conv2d(3, 8, kernel_size=2, stride=2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 8))
        self.blocks = nn.ModuleList(_FakeBlock(8) for _ in range(12))
        self.norm = nn.LayerNorm(8)
        with torch.no_grad():
            self.norm.weight.copy_(torch.linspace(0.4, 1.8, 8))
            self.norm.bias.copy_(torch.linspace(-0.5, 0.7, 8))
        self.intermediate_calls = []

    def hidden_states(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        states = []
        for block in self.blocks:
            x = block(x)
            states.append(x)
        return states

    def get_intermediate_layers(
        self, x, n=1, reshape=False, return_class_token=False, norm=True,
    ):
        self.intermediate_calls.append((n, reshape, return_class_token, norm))
        indices = range(len(self.blocks) - n, len(self.blocks)) if isinstance(n, int) else n
        states = self.hidden_states(x)
        return tuple(self.norm(states[i]) if norm else states[i] for i in indices)


@pytest.fixture
def fake_model(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261017)
        model = _FakeDINOv3()
    monkeypatch.setattr(dino_v3_lora, "load_dino_v3", lambda **_: model)
    return model


def _assert_no_final_affine(encoder):
    norm = encoder.model.norm
    assert type(norm) is nn.LayerNorm
    assert norm.normalized_shape == (8,)
    assert norm.eps == 1e-5
    assert norm.elementwise_affine is False
    assert norm.weight is None and norm.bias is None
    assert list(norm.parameters()) == []
    assert list(norm.buffers()) == []
    assert not any(key.startswith("model.norm.") for key in encoder.state_dict())


@pytest.mark.parametrize("train_norm", [False, True])
def test_final_ln_removes_pretrained_affine_without_touching_block_norms(fake_model, train_norm):
    old_final_norm = fake_model.norm
    block_norms = [norm for block in fake_model.blocks for norm in (block.norm1, block.norm2)]
    old_values = [(norm.weight.detach().clone(), norm.bias.detach().clone()) for norm in block_norms]
    encoder = DINOv3PatchTokenEncoder(use_lora=True, lora_r=2, train_norm=train_norm)

    _assert_no_final_affine(encoder)
    assert encoder.model.norm is not old_final_norm
    for block, original_norms in zip(encoder.model.blocks, zip(block_norms[::2], block_norms[1::2])):
        assert block.norm1 is original_norms[0]
        assert block.norm2 is original_norms[1]
    for norm, (weight, bias) in zip(block_norms, old_values):
        assert norm.elementwise_affine is True
        torch.testing.assert_close(norm.weight, weight)
        torch.testing.assert_close(norm.bias, bias)
        assert norm.weight.requires_grad is False
        assert norm.bias.requires_grad is False
    assert "model.blocks.0.norm1.weight" in encoder.state_dict()
    assert "model.blocks.0.norm2.bias" in encoder.state_dict()


@pytest.mark.parametrize("role", ["student", "teacher"])
def test_build_encoder_applies_noaffine_to_both_roles_and_preserves_trainability(fake_model, role):
    args = SimpleNamespace(
        dino_model_name="dinov3_vitb16", dino_aggregation="mls", dino_layers="default",
        dino_lora_r=2, dino_lora_alpha=2.0, dino_train_patch_embed=True,
    )
    encoder = build_encoder(args, role=role)

    _assert_no_final_affine(encoder)
    assert encoder.model.cls_token.requires_grad is False
    if role == "teacher":
        assert encoder.aggregation == "none"
        assert encoder._layer_indices is None
        assert encoder.use_lora is False
        assert encoder.num_lora_modules == 0
        assert not any(parameter.requires_grad for parameter in encoder.parameters())
        assert not any(isinstance(module, LoRALinear) for module in encoder.modules())
    else:
        assert encoder.aggregation == "mls"
        assert encoder._layer_indices == [2, 5, 8, 11]
        assert encoder.use_lora is True
        assert encoder.num_lora_modules == 12 * 4
        for name, parameter in encoder.named_parameters():
            expected_trainable = name.startswith("model.patch_embed.") or "lora_" in name
            assert parameter.requires_grad == expected_trainable, name


@pytest.mark.parametrize("aggregation", ["none", "mean", "mls"])
def test_forward_is_noaffine_layernorm_then_unchanged_aggregation(fake_model, aggregation):
    indices = None if aggregation == "none" else [2, 5, 8, 11]
    old_weight = fake_model.norm.weight.detach().clone()
    old_bias = fake_model.norm.bias.detach().clone()
    encoder = DINOv3PatchTokenEncoder(aggregation=aggregation, layer_indices=indices)
    images = torch.linspace(-1, 1, 2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
    states = fake_model.hidden_states(images)
    normalized = [F.layer_norm(state, (8,), weight=None, bias=None, eps=1e-5) for state in states]
    expected_last = normalized[-1]
    if indices is None:
        expected = expected_last
    else:
        expected = torch.stack([normalized[i] for i in indices]).mean(dim=0)
        if aggregation == "mls":
            expected = expected + normalized[indices[-1]].mean(dim=1, keepdim=True)

    tokens, last_raw = encoder.forward_with_last_raw(images)
    assert tokens.shape == last_raw.shape == (2, 4, 8)
    torch.testing.assert_close(tokens, expected)
    torch.testing.assert_close(last_raw, expected_last)
    torch.testing.assert_close(encoder(images), expected)
    assert all(call[1:] == (False, False, True) for call in fake_model.intermediate_calls)
    affine_last = F.layer_norm(states[-1], (8,), old_weight, old_bias, eps=1e-5)
    assert not torch.allclose(last_raw, affine_last)


def test_nonfinal_mls_selection_still_returns_noaffine_actual_last_layer(fake_model):
    encoder = DINOv3PatchTokenEncoder(aggregation="mls", layer_indices=[2, 5, 8])
    images = torch.linspace(-1, 1, 3 * 4 * 4).reshape(1, 3, 4, 4)
    states = fake_model.hidden_states(images)
    normalized = [F.layer_norm(state, (8,), eps=1e-5) for state in states]

    tokens, last_raw = encoder.forward_with_last_raw(images)
    expected = torch.stack([normalized[i] for i in [2, 5, 8]]).mean(dim=0)
    expected = expected + normalized[8].mean(dim=1, keepdim=True)
    torch.testing.assert_close(tokens, expected)
    torch.testing.assert_close(last_raw, normalized[-1])
    assert [call[0] for call in fake_model.intermediate_calls] == [[2, 5, 8], 1]
    assert all(call[1:] == (False, False, True) for call in fake_model.intermediate_calls)


def test_noaffine_student_preserves_patch_embed_and_lora_gradients(fake_model):
    encoder = DINOv3PatchTokenEncoder(
        use_lora=True, lora_r=2, lora_alpha=2.0, train_patch_embed=True,
        aggregation="mls", layer_indices=[2, 5, 8, 11],
    )
    images = torch.linspace(-1, 1, 2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
    tokens = encoder(images)
    probe = torch.linspace(-0.7, 1.3, tokens.numel()).reshape_as(tokens)
    (tokens * probe).sum().backward()

    patch_grad = encoder.model.patch_embed.weight.grad
    assert patch_grad is not None and torch.isfinite(patch_grad).all()
    assert patch_grad.abs().sum() > 0
    lora_modules = [module for module in encoder.modules() if isinstance(module, LoRALinear)]
    assert len(lora_modules) == 12 * 4
    for module in lora_modules:
        # B starts at zero, so A can legitimately have a zero first-step gradient.
        assert module.lora_A.grad is not None and torch.isfinite(module.lora_A.grad).all()
        assert module.lora_B.grad is not None and torch.isfinite(module.lora_B.grad).all()
        assert module.lora_B.grad.abs().sum() > 0
        assert module.base.weight.grad is None
        assert module.base.bias.grad is None
    assert all(parameter.grad is None for block in encoder.model.blocks for parameter in block.norm1.parameters())
    _assert_no_final_affine(encoder)
