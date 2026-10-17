import inspect

import torch
import torch.nn as nn

import dino_v3_lora
from dino_v3_lora import DINOv3PatchTokenEncoder


class _FakeDINOv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 8
        self.patch_size = 16
        self.blocks = nn.ModuleList()
        self.patch_embed = nn.Conv2d(3, 8, kernel_size=16, stride=16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 8))
        self.norm = nn.LayerNorm(8)


def test_dinov3_exposes_only_real_trainability_controls(monkeypatch):
    model = _FakeDINOv3()
    monkeypatch.setattr(dino_v3_lora, "load_dino_v3", lambda **_: model)

    encoder = DINOv3PatchTokenEncoder(
        use_lora=True,
        train_patch_embed=True,
        train_cls_token=False,
        train_norm=False,
    )

    assert all(parameter.requires_grad for parameter in encoder.model.patch_embed.parameters())
    assert encoder.model.cls_token.requires_grad is False
    assert all(not parameter.requires_grad for parameter in encoder.model.norm.parameters())
    assert not any("pos_embed" in name for name, _ in encoder.named_parameters())

    signature = inspect.signature(DINOv3PatchTokenEncoder)
    assert "train_pos_embed" not in signature.parameters
    assert "use_time_token" not in signature.parameters
