"""iREPA alignment features stay independent of generation LoRA/MLS/no-affine."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torchvision import transforms

import dino_alignment_encoder
import dino_encoder
from dino_alignment_encoder import DINOv3AlignmentEncoder
from dino_encoder import build_alignment_encoder
from dino_lora import LoRALinear


class _FakePretrainedDINOv3(nn.Module):
    """Small official-shaped backbone with intentionally nontrivial LN affine."""

    def __init__(self):
        super().__init__()
        self.embed_dim = 8
        self.patch_size = 2
        self.patch_embed = nn.Conv2d(3, 8, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(8, eps=1e-5)
        self.cls_token = nn.Parameter(torch.linspace(-0.8, 0.8, 8).reshape(1, 1, 8))
        with torch.no_grad():
            self.norm.weight.copy_(torch.linspace(0.4, 1.8, 8))
            self.norm.bias.copy_(torch.linspace(-0.5, 0.7, 8))
        self.forward_inputs = []
        self.last_output = None

    def forward_features(self, images):
        self.forward_inputs.append(images)
        patch = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls = self.cls_token[:, 0].expand(images.shape[0], -1)
        self.last_output = {
            "x_norm_clstoken": self.norm(cls),
            "x_norm_patchtokens": self.norm(patch),
            "x_prenorm": patch,
        }
        return self.last_output

    def get_intermediate_layers(self, *args, **kwargs):
        raise AssertionError("iREPA alignment must not take the generation MLS path")


@pytest.fixture
def pretrained_backbone(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261017)
        model = _FakePretrainedDINOv3()
    monkeypatch.setattr(dino_alignment_encoder, "load_dino_v3", lambda **kwargs: model)
    return model


def test_alignment_keeps_exact_pretrained_parameters_and_norm(pretrained_backbone):
    original_norm = pretrained_backbone.norm
    original_weight = original_norm.weight
    original_bias = original_norm.bias
    original_values = {name: value.detach().clone() for name, value in pretrained_backbone.state_dict().items()}

    encoder = DINOv3AlignmentEncoder(resolution=4)

    assert encoder.model is pretrained_backbone
    assert encoder.model.norm is original_norm
    assert encoder.model.norm.weight is original_weight
    assert encoder.model.norm.bias is original_bias
    assert encoder.model.norm.elementwise_affine is True
    assert encoder.model.norm.eps == 1e-5
    assert encoder.embed_dim == 8
    assert encoder.patch_size == 2
    assert encoder.resolution == 4
    assert not encoder.training
    assert all(not module.training for module in encoder.model.modules())
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert not any(isinstance(module, LoRALinear) for module in encoder.modules())
    assert not hasattr(encoder, "aggregation")
    assert not hasattr(encoder, "_layer_indices")
    assert set(pretrained_backbone.state_dict()) == set(original_values)
    for name, value in pretrained_backbone.state_dict().items():
        torch.testing.assert_close(value, original_values[name], rtol=0, atol=0)
    assert "model.norm.weight" in encoder.state_dict()
    assert "model.norm.bias" in encoder.state_dict()


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.uint8])
def test_alignment_raw255_preprocess_matches_irepa_order(pretrained_backbone, input_dtype):
    encoder = DINOv3AlignmentEncoder(resolution=4)
    raw = torch.arange(2 * 3 * 6 * 8).reshape(2, 3, 6, 8).remainder(256).to(input_dtype)
    # iREPA: divide raw pixels by 255, square resize with antialias, ImageNet normalize.
    expected = raw / 255.0
    expected = transforms.Resize((4, 4), antialias=True)(expected)
    expected = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    )(expected)

    actual = encoder.preprocess(raw)
    assert actual.shape == (2, 3, 4, 4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert pretrained_backbone.forward_inputs == []


def test_alignment_forward_uses_official_last_features_without_mls(pretrained_backbone):
    encoder = DINOv3AlignmentEncoder(resolution=4)
    normalized = torch.linspace(-2.0, 2.0, 2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
    expected = pretrained_backbone.forward_features(normalized)
    pretrained_backbone.forward_inputs.clear()

    features = encoder.forward_features(normalized)
    assert set(features) == {"x_norm_clstoken", "x_norm_patchtokens"}
    assert features["x_norm_clstoken"] is pretrained_backbone.last_output["x_norm_clstoken"]
    assert features["x_norm_patchtokens"] is pretrained_backbone.last_output["x_norm_patchtokens"]
    assert pretrained_backbone.forward_inputs == [normalized]
    torch.testing.assert_close(features["x_norm_clstoken"], expected["x_norm_clstoken"])
    torch.testing.assert_close(features["x_norm_patchtokens"], expected["x_norm_patchtokens"])

    tokens = encoder(normalized)
    assert tokens.shape == (2, 4, 8)
    assert pretrained_backbone.forward_inputs[-1] is normalized
    torch.testing.assert_close(tokens, expected["x_norm_patchtokens"])
    nonaffine = nn.functional.layer_norm(expected["x_prenorm"], (8,), eps=1e-5)
    assert not torch.allclose(tokens, nonaffine)
    assert tokens.requires_grad is False
    assert all(parameter.grad is None for parameter in encoder.parameters())


@pytest.mark.parametrize("generation_variant", ["mls_lora", "invalid_generation_options"])
def test_alignment_builder_ignores_generation_configuration(monkeypatch, generation_variant):
    load_calls = []

    def load_model(**kwargs):
        load_calls.append(kwargs)
        return _FakePretrainedDINOv3()

    def reject_generation_construction(*args, **kwargs):
        raise AssertionError("alignment builder must not instantiate the generation wrapper")

    monkeypatch.setattr(dino_alignment_encoder, "load_dino_v3", load_model)
    monkeypatch.setattr(dino_encoder, "DINOv3PatchTokenEncoder", reject_generation_construction)
    args = SimpleNamespace(
        dino_model_name="dinov3_vitb16", dino_weights="checkpoint.pth", img_size=4,
        dino_aggregation="mls", dino_layers="default", dino_lora_r=64,
        dino_lora_alpha=64.0, dino_lora_dropout=0.3,
        dino_lora_target_modules="qkv,proj,fc1,fc2", dino_train_patch_embed=True,
        dino_input_preprocess="none",
    )
    if generation_variant == "invalid_generation_options":
        args.dino_aggregation = "invalid"
        args.dino_layers = "invalid"
        args.dino_lora_r = -99
        args.dino_lora_target_modules = "invalid"
        args.dino_train_patch_embed = False

    encoder = build_alignment_encoder(args)

    assert isinstance(encoder, DINOv3AlignmentEncoder)
    assert encoder.resolution == 4
    assert load_calls == [{"model_name": "dinov3_vitb16", "weights_path": "checkpoint.pth"}]
    assert encoder.model.norm.weight is not None
    assert encoder.model.norm.bias is not None
    assert not any(isinstance(module, LoRALinear) for module in encoder.modules())
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_alignment_builder_default_resolution_is_256(pretrained_backbone):
    encoder = build_alignment_encoder(SimpleNamespace())
    assert encoder.resolution == 256
