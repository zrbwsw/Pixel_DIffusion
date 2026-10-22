"""生成侧 DINOv3：加载预训练权重 → 注入 LoRA → final LN → 选层聚合。

默认 B/16 输入256×256，输出 [B,256,768] patch tokens，不含 CLS/register。
MLS 沿用 RAEv2 的 DINOv3MultiLayerSimpleAddEncoder；teacher 在 irepa_teacher.py。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from dino_lora import inject_dino_lora
# 保留资源预检使用的加载函数别名。
from dinov3_backbone import load_dino_v3, _resolve_repo, _resolve_weights


# RAEv2 默认选层，索引从0开始。
RAEV2_DEFAULT_LAYERS = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [5, 11, 17, 23],
}


def parse_layer_spec(spec: Optional[str], total_blocks: int) -> Optional[List[int]]:
    """支持 last、last-4 / last4、2,5,8,11；default 由工厂替换为默认选层。"""
    if spec is None:
        return None
    spec = spec.strip().lower()
    if spec in ("", "none", "default"):
        return None
    if spec == "last":
        return [total_blocks - 1]
    if spec.startswith("last"):
        k = int(spec.replace("last-", "").replace("last", ""))
        return list(range(total_blocks - k, total_blocks))
    return [int(part) for part in spec.replace(".", ",").split(",") if part.strip()]


class DINOv3PatchTokenEncoder(nn.Module):
    """先冻结 backbone，再启用 LoRA 及指定参数；forward 返回 normalized patch tokens。"""

    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        weights_path: Optional[str] = None,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_target_modules: str = "qkv,proj,fc1,fc2",
        train_patch_embed: bool = True,
        train_cls_token: bool = False,
        train_norm: bool = False,
        aggregation: str = "none",
        layer_indices: Optional[Sequence[int]] = None,
        final_norm_affine: bool = True,
    ):
        super().__init__()
        self.model = load_dino_v3(model_name=model_name, weights_path=weights_path)
        self.embed_dim = self.model.embed_dim
        self.final_norm_affine = final_norm_affine
        if not final_norm_affine:
            # NoAffine 只移除输出 LN 的 gamma/beta，保留归一化和 eps。
            norm = self.model.norm
            self.model.norm = nn.LayerNorm(
                norm.normalized_shape,
                eps=norm.eps,
                elementwise_affine=False,   #这里移除了
                device=norm.weight.device,
                dtype=norm.weight.dtype,
            ).train(norm.training)

        self.use_lora = use_lora
        self.aggregation = aggregation
        self.num_lora_modules = 0
        self._layer_indices = list(layer_indices) if layer_indices is not None else None
        self._total_blocks = len(self.model.blocks)

        # 原始 backbone 全冻结；不启用 LoRA 时，到这里即完成冻结设置。
        self.model.requires_grad_(False)
        if use_lora:
            # qkv/proj/fc1/fc2：原 Linear 冻结，仅训练 LoRA A/B。
            self.num_lora_modules = inject_dino_lora(
                self.model,
                r=lora_r, alpha=lora_alpha, dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            # 额外可训项：patch embed、CLS token、输出 LN 的 gamma/beta。
            # 工厂默认冻结 CLS 和输出 LN；block 内部 norm 始终冻结。
            self.model.patch_embed.requires_grad_(train_patch_embed)
            self.model.cls_token.requires_grad_(train_cls_token)
            self.model.norm.requires_grad_(train_norm)

    @property
    def patch_size(self) -> int:
        return self.model.patch_size

    def forward_patch_tokens(self, x: torch.Tensor, return_last_raw: bool = False):
        """none 取单层，mean 取层均值，MLS 再加最后一个选中层的空间均值。
        last_raw 指 backbone 最后一层经过 LN、尚未聚合的 patch tokens。
        """
        if self.aggregation == "none" and self._layer_indices is None:
            patch_tokens = self.model.get_intermediate_layers(
                x, n=1, reshape=False, return_class_token=False, norm=True,
            )[0]
            return (patch_tokens, patch_tokens) if return_last_raw else patch_tokens

        # 每个选中层共用输出 LN；outputs 中每项都是 [B,N,D]。
        n_arg = self._layer_indices if self._layer_indices is not None else 1
        outputs = self.model.get_intermediate_layers(
            x, n=n_arg, reshape=False, return_class_token=False, norm=True,
        )
        if self._layer_indices and self._layer_indices[-1] == self._total_blocks - 1:
            last_raw = outputs[-1]
        else:
            # 保持原有额外前向及调用顺序，避免改变训练态的随机数消耗。
            last_raw = self.model.get_intermediate_layers(
                x, n=1, reshape=False, return_class_token=False, norm=True,
            )[0]

        patch_tokens = torch.stack(outputs, dim=0).mean(dim=0)
        if self.aggregation == "mls":
            # MLS = 选中层均值 + 最后一个选中层的空间均值广播。
            patch_tokens = patch_tokens + outputs[-1].mean(dim=1, keepdim=True)
        return (patch_tokens, last_raw) if return_last_raw else patch_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_patch_tokens(x)

    def forward_with_last_raw(self, x: torch.Tensor):
        return self.forward_patch_tokens(x, return_last_raw=True)
