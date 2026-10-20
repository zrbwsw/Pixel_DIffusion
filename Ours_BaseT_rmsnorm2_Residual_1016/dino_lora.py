"""DINOv3 Transformer block 使用的轻量 LoRA 注入工具。

当前 class-to-image 主线只在 DINOv3 的 attention/MLP 线性层上训练低秩
增量。这个文件刻意不包含模型下载、DINOv1 兼容或扩散逻辑，阅读时只需关注：

* ``LoRALinear`` 冻结原始 ``nn.Linear``，学习 ``B @ A``；
* ``inject_dino_lora`` 遍历 ViT blocks，并按名称替换 qkv/proj/fc1/fc2。

以 DINOv3-B/16、rank=64 为例，每个 block 的目标层保持原有输入
输出维度，因而不会改变 patch token 的 ``[B, 256, 768]`` 形状。
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """冻结原线性层，仅训练秩为 ``r`` 的低秩残差。"""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_layer)}")
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")

        self.base = base_layer
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for param in self.base.parameters():
            param.requires_grad_(False)

        # A: [r, in_features]，B: [out_features, r]。B 从零初始化，保证注入
        # LoRA 后第一次 forward 与预训练 DINOv3 完全一致。
        self.lora_A = nn.Parameter(torch.empty(self.r, self.base.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.base.out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 的最后一维为 in_features；其余 batch/token 维原样保留。
        base_out = self.base(x)
        lora_out = F.linear(
            self.dropout(x),
            self.lora_A.to(dtype=x.dtype, device=x.device),
        )
        lora_out = F.linear(
            lora_out,
            self.lora_B.to(dtype=x.dtype, device=x.device),
        )
        return base_out + lora_out * self.scaling

    # DINOv3 attention 会读取 qkv.in_features 等属性；代理这些属性可让包装
    # 对模型其余代码透明。
    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias


def _parse_csv(spec: str | Iterable[str]) -> set[str]:
    if isinstance(spec, str):
        return {part.strip() for part in spec.split(",") if part.strip()}
    return {str(part).strip() for part in spec if str(part).strip()}


def _wrap_child_linear(
    parent: nn.Module,
    name: str,
    r: int,
    alpha: float,
    dropout: float,
) -> int:
    if not hasattr(parent, name):
        return 0
    child = getattr(parent, name)
    if not isinstance(child, nn.Linear):
        return 0
    setattr(parent, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
    return 1


def inject_dino_lora(
    model: nn.Module,
    r: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: str | Iterable[str] = "qkv,proj,fc1,fc2",
) -> int:
    """向 DINOv3 ViT blocks 注入 LoRA，并返回实际替换的线性层数量。"""

    targets = _parse_csv(target_modules)
    count = 0
    for block in getattr(model, "blocks", []):
        if "qkv" in targets and hasattr(block, "attn"):
            count += _wrap_child_linear(block.attn, "qkv", r, alpha, dropout)
        if "proj" in targets and hasattr(block, "attn"):
            count += _wrap_child_linear(block.attn, "proj", r, alpha, dropout)
        if "fc1" in targets and hasattr(block, "mlp"):
            count += _wrap_child_linear(block.mlp, "fc1", r, alpha, dropout)
        if "fc2" in targets and hasattr(block, "mlp"):
            count += _wrap_child_linear(block.mlp, "fc2", r, alpha, dropout)
    return count
