"""独立的 iREPA frozen teacher，不依赖生成侧 DINO-LoRA 包装。

沿用 iREPA 的 forward_features 接口，返回最后层 normalized CLS / patch tokens。
输入已在 _encode_teacher_target 中固定做 ImageNet 标准化；这里不重复预处理。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from dinov3_backbone import load_dino_v3


class FrozenDINOv3Encoder(nn.Module):
    """只接收 backbone 标识；构建后全部冻结，并始终处于 eval 模式。"""

    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        weights_path: Optional[str] = None,
    ):
        super().__init__()
        self.model = load_dino_v3(model_name=model_name, weights_path=weights_path)
        self.embed_dim = int(getattr(self.model, "embed_dim", getattr(self.model, "num_features", 768)))
        self.requires_grad_(False)
        self.eval()

    @property
    def patch_size(self) -> int:
        return self.model.patch_size

    def train(self, mode: bool = True) -> "FrozenDINOv3Encoder":
        # DINOv3 的 RoPE 在 train 模式下含随机坐标变换；冻结梯度并不足够。
        super().train(False)
        return self

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # no_grad 的目标仍可参与对齐 loss 的反传；不要改成 inference_mode。
        features = self.model.forward_features(x)
        return {
            "x_norm_clstoken": features["x_norm_clstoken"],
            "x_norm_patchtokens": features["x_norm_patchtokens"],
        }
