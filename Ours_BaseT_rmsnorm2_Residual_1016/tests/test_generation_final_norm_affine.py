"""Generation-only final-LN ablation must not change the frozen teacher."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import dino_v3_lora
from dino_encoder import build_encoder


class _TinyDINOv3(nn.Module):
    """Expose DINO's shared output-LN contract with nonidentity pretrained affine."""

    def __init__(self):
        super().__init__()
        self.embed_dim = 4
        self.patch_size = 16
        self.patch_embed = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([nn.LayerNorm(4), nn.LayerNorm(4)])
        self.norm = nn.LayerNorm(4, eps=3e-4)
        with torch.no_grad():
            self.norm.weight.copy_(torch.tensor([0.5, 2.0, 3.0, 4.0]))
            self.norm.bias.copy_(torch.tensor([0.2, -0.3, 0.4, -0.5]))

    def raw_layers(self, x):
        first = self.blocks[0](self.patch_embed(x))
        second = self.blocks[1](first.square() + x)
        return first, second

    def get_intermediate_layers(self, x, n, *, reshape, return_class_token, norm):
        assert not reshape and not return_class_token
        layers = self.raw_layers(x)
        indices = range(len(layers) - n, len(layers)) if isinstance(n, int) else n
        return tuple(self.norm(layers[i]) if norm else layers[i] for i in indices)


@pytest.fixture
def pretrained(monkeypatch):
    torch.manual_seed(17)
    model = _TinyDINOv3().double().eval()
    monkeypatch.setattr(dino_v3_lora, "load_dino_v3", lambda **_: deepcopy(model))
    return model


def _args(**overrides):
    values = dict(dino_aggregation="mls", dino_layers="0,1")
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("affine", [False, True])
def test_generation_option_leaves_teacher_weights_and_outputs_unchanged(pretrained, affine):
    teacher = build_encoder(
        _args(dino_generation_final_norm_affine=affine), role="teacher"
    )
    assert teacher.aggregation == "none"
    assert teacher._layer_indices is None
    assert not teacher.use_lora
    assert teacher.num_lora_modules == 0
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    assert teacher.model.norm.elementwise_affine
    assert teacher.model.state_dict().keys() == pretrained.state_dict().keys()
    for name, expected in pretrained.state_dict().items():
        torch.testing.assert_close(teacher.model.state_dict()[name], expected, rtol=0, atol=0)
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    expected = pretrained.get_intermediate_layers(
        x, n=1, reshape=False, return_class_token=False, norm=True
    )[0]
    torch.testing.assert_close(teacher(x), expected, rtol=0, atol=0)


def test_student_default_retains_affine_and_checkpoint_layout(pretrained):
    default = build_encoder(_args(), role="student")
    explicit = build_encoder(_args(dino_generation_final_norm_affine=True), role="student")
    assert default.model.norm.elementwise_affine
    assert default.model.state_dict().keys() == pretrained.state_dict().keys()
    for name, expected in pretrained.state_dict().items():
        torch.testing.assert_close(default.model.state_dict()[name], expected, rtol=0, atol=0)
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    torch.testing.assert_close(default(x), explicit(x), rtol=0, atol=0)


def test_student_noaffine_removes_only_final_ln_parameters_and_normalizes_mls(pretrained):
    student = build_encoder(
        _args(dino_generation_final_norm_affine=False), role="student"
    )
    final_norm = student.model.norm
    assert not final_norm.elementwise_affine
    assert final_norm.weight is None and final_norm.bias is None
    assert final_norm.normalized_shape == pretrained.norm.normalized_shape
    assert final_norm.eps == pretrained.norm.eps
    assert final_norm.training == pretrained.norm.training
    assert student.use_lora and student.aggregation == "mls"
    assert student.model.patch_embed.weight.requires_grad
    assert set(pretrained.state_dict()) - set(student.model.state_dict()) == {
        "norm.weight", "norm.bias"
    }
    for name, actual in student.model.state_dict().items():
        torch.testing.assert_close(actual, pretrained.state_dict()[name], rtol=0, atol=0)
    assert all(block.elementwise_affine for block in student.model.blocks)

    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    raw_layers = pretrained.raw_layers(x)
    normalized = [F.layer_norm(raw, (4,), eps=pretrained.norm.eps) for raw in raw_layers]
    expected_mls = (normalized[0] + normalized[1]) / 2 + normalized[1].mean(1, keepdim=True)
    mls, last = student.forward_with_last_raw(x)
    assert mls.dtype == x.dtype and mls.device == x.device
    torch.testing.assert_close(mls, expected_mls)
    torch.testing.assert_close(last, normalized[-1])
    baseline = build_encoder(_args(), role="student")
    assert not torch.allclose(mls, baseline(x))
    mls.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert student.model.patch_embed.weight.grad is not None
