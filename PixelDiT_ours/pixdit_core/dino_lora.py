"""给 DINOv3 注入 LoRA，让预训练层适应扩散输入而不全量微调。

每个被包装的 Linear 计算 y = frozen_linear(x) + (alpha/r) * B(A(x))。
A [r,I] 把输入降到低秩 r，B [O,r] 再投回输出宽度；只训练 A/B，
原权重保持冻结。B 从零初始化，因此注入初期不改变预训练层输出。
"""

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """在冻结 Linear 旁增加低秩残差，保持输入输出接口完全不变。"""

    def __init__(self, base_layer: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base_layer  # 原预训练 Linear，保留其权重和输出维度
        self.r = int(r)  # LoRA 秩，远小于输入/输出宽度
        self.scaling = float(alpha) / float(r)  # 控制低秩增量相对主线的强度
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.base.requires_grad_(False)  # 明确冻结原层，优化器只接收新建的 LoRA 参数
        self.lora_A = nn.Parameter(torch.empty(self.r, self.base.in_features))  # [r,I]，降维投影
        self.lora_B = nn.Parameter(torch.zeros(self.base.out_features, self.r))  # [O,r]，零初始化使初始残差为0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算冻结主线与 LoRA 残差，保持输入的 batch/token 维。"""
        base_out = self.base(x)  # 预训练主线：x [...,I] → base_out [...,O]
        lora_out = F.linear(self.dropout(x), self.lora_A.to(dtype=x.dtype))  # 低秩降维：[...,I] → [...,r]
        lora_out = F.linear(lora_out, self.lora_B.to(dtype=x.dtype))  # 低秩恢复：[...,r] → [...,O]
        return base_out + lora_out * self.scaling  # 主线 + alpha/r 倍增量，最终 [...,O]

    # @property 只把原 Linear 的接口透传给 DINO block，不参与前向计算。
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


def inject_dino_lora(
    model: nn.Module,
    r: int,
    alpha: float,
    dropout: float,
    target_modules: str | Iterable[str] = "qkv,proj,fc1,fc2",
) -> int:
    """按名称替换 DINO blocks 中的线性层，返回实际注入数量。

    qkv/proj 属于注意力，fc1/fc2 属于 MLP；默认每个 block 注入四层，
    因此 12 层 DINO-B 共48个 LoRA 包装。未选中的层保持原实现。
    """
    targets = (
        {part.strip() for part in target_modules.split(",") if part.strip()}
        if isinstance(target_modules, str)
        else set(target_modules)
    )
    count = 0
    for block in model.blocks:
        for parent, name in ((block.attn, "qkv"), (block.attn, "proj"), (block.mlp, "fc1"), (block.mlp, "fc2")):
            if name not in targets:
                continue
            base_layer = getattr(parent, name)
            setattr(parent, name, LoRALinear(base_layer, r, alpha, dropout))  # 用“冻结原层+LoRA”替换原 Linear，张量形状不变
            count += 1
    return count
