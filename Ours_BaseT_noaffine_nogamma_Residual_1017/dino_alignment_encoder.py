"""Frozen DINOv3 features for iREPA regularization, independent of generation.

Follow iREPA/jit/vision_encoder.py::DINOv3Encoder: standard image preprocessing
and model.forward_features(), retaining the pretrained final LayerNorm affine.
Only the portable pretrained loader is shared with the generation branch;
this class never constructs its LoRA/MLS/no-affine wrapper.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from dino_v3_lora import load_dino_v3


class DINOv3AlignmentEncoder(nn.Module):
    """Original DINOv3 last-layer features used only as a regularization target.

    ``preprocess`` accepts images in [0,255]; ``forward`` accepts its normalized
    output. Final LN gamma/beta and all pretrained weights remain untouched.
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        weights_path: Optional[str] = None,
        resolution: int = 256,
    ):
        super().__init__()
        self.model = load_dino_v3(model_name=model_name, weights_path=weights_path)
        self.embed_dim = int(self.model.embed_dim)
        self.patch_size = int(self.model.patch_size)
        self.resolution = resolution
        # Keep the legacy loader module's import-time logging out of CLI preflight.
        from jit.models.dinov3_loader import make_dinov3_transform

        self._preprocess = make_dinov3_transform(resize_size=resolution)
        # iREPA evaluates this encoder outside the optimizer under no_grad.
        # Explicit freezing additionally guards against accidental optimization.
        self.model.requires_grad_(False)
        self.eval()

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        return self._preprocess(images)

    def forward_features(self, normalized_images: torch.Tensor) -> dict:
        output = self.model.forward_features(normalized_images)
        return {
            "x_norm_clstoken": output.get("x_norm_clstoken"),
            "x_norm_patchtokens": output.get("x_norm_patchtokens"),
        }

    def forward(self, normalized_images: torch.Tensor) -> torch.Tensor:
        return self.forward_features(normalized_images)["x_norm_patchtokens"]
