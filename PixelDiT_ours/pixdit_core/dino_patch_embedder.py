"""DINO 语义 patchify：把 DINO 特征变成 PixelDiT 可用的输入 token。

数据流：带噪图像 → DINOv3+LoRA/选层聚合 → PgT(t) → RMSNorm。
PgT 同时完成 D 到 C 的维度适配、时间条件注入和 token 间信息变换；
输出再与原生 patchify 按 f(t)/(1-f(t)) 加权。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pixdit_core.dino_encoder import DINOv3LoRAEncoder
from pixdit_core.grounding import TimestepProjection
from pixdit_core.modules import RMSNorm


class DINOPatchEmbedder(nn.Module):
    """构建 DINO 生成分支：y 给 backbone，原始 t 给 PgT。

    hidden_size/C 是 PixelDiT patch 级宽度；encoder.embed_dim/D 是 DINO
    特征宽度。y 是否真正注入由 label_cond 控制，t 不使用 f(t) 变换。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        patch_size: int,
        model_name: str,
        weights_path: str | None,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        lora_target_modules,
        train_patch_embed: bool,
        train_final_norm: bool,
        final_norm_affine: bool,
        aggregation: str,
        layers,
        label_cond: bool,
        num_classes: int,
    ):
        super().__init__()
        self.patch_size = int(patch_size)  # PixelDiT 的 patch 边长，用于确定目标 token 网格
        self.encoder = DINOv3LoRAEncoder(
            model_name=model_name,
            weights_path=weights_path,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            train_patch_embed=train_patch_embed,
            train_final_norm=train_final_norm,
            final_norm_affine=final_norm_affine,
            aggregation=aggregation,
            layers=layers,
            label_cond=label_cond,
            num_classes=num_classes,
        )  # 提取 [B,Nd,D] 的 DINO 语义 token，可选注入 y
        self.projector = TimestepProjection(
            in_dim=self.encoder.embed_dim,
            out_dim=hidden_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
        )  # PgT：D→C 投影，并用原始 t 做 adaLN 和二维 RoPE 注意力
        self.norm = RMSNorm(hidden_size, eps=1e-6)  # 与原生 patchify 加权前，统一 DINO 路的通道尺度

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """把带噪图像变成与原生 patchify 完全同形的 [B,N,C] token。"""
        batch, _, height, width = x.shape  # x [B,3,H,W]
        tokens = self.encoder(x, y)  # DINO+可选 y：[B,Nd,D]，Nd 是 DINO 网格的 token 数
        grid_hw = (height // self.patch_size, width // self.patch_size)  # PixelDiT 目标网格 (h,w)，N=h*w
        dino_grid_hw = (height // self.encoder.patch_size, width // self.encoder.patch_size)  # DINO 原始网格 (hd,wd)
        if dino_grid_hw != grid_hw:
            tokens = tokens.transpose(1, 2).reshape(batch, self.encoder.embed_dim, *dino_grid_hw)  # [B,Nd,D] → [B,D,hd,wd]，恢复空间网格
            tokens = F.interpolate(tokens, size=grid_hw, mode="bilinear", align_corners=False)  # 双线性对齐目标网格：[B,D,hd,wd] → [B,D,h,w]
            tokens = tokens.flatten(2).transpose(1, 2)  # [B,D,h,w] → [B,N,D]，与 PixelDiT patch 一一对应
        tokens = self.projector(
            tokens.to(dtype=self.projector.input_proj.weight.dtype),  # PgT 输入 [B,N,D]，对齐投影权重 dtype
            t.reshape(-1),  # 原始扩散时间 [B]，供 PgT 时间条件使用
            grid_hw,  # 给 PgT 的二维 RoPE 提供目标网格形状
        )  # PgT 完成 D→C 和时间条件变换：[B,N,C]
        return self.norm(tokens)  # 融合前最后一道 RMSNorm：[B,N,C]
