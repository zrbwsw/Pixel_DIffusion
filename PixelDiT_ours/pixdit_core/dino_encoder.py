"""生成侧 DINOv3 编码器：提取可与 PixelDiT patchify 融合的语义 token。

主 backbone 来自预训练 DINOv3 并默认冻结，只训练 LoRA、可选的 patch
embedding 和 y 条件嵌入。forward 先做 final LN，再按配置聚合多个层；
它不负责 PixelDiT 的扩散损失，也不是 REPA 的 frozen DINOv2 teacher。
"""

import torch
import torch.nn as nn

from pixdit_core.dino_lora import inject_dino_lora
from pixdit_core.dinov3_backbone import load_dino_v3
from pixdit_core.modules import ClassEmbedder

# 层号从0开始：B/16 的 [2,5,8,11] 对应第3/6/9/12个 block；
# L/16 的 [5,11,17,23] 对应第6/12/18/24个 block。
_DEFAULT_LAYERS = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [5, 11, 17, 23],
}


def _resolve_layers(model_name: str, spec, total_blocks: int):
    """把 YAML 的层描述解析为 block 索引；None 在 forward 中表示最后层。

    default 使用上面的模型默认选层；last 取最后层；last-N 取末 N 层；
    也支持 "2,5,8,11" 或直接的整数索引列表。
    """
    if spec is None:
        return None
    if not isinstance(spec, str):
        return list(spec)
    spec = spec.strip().lower()
    if spec in ("", "none"):
        return None
    if spec == "default":
        return list(_DEFAULT_LAYERS[model_name])
    if spec == "last":
        return [total_blocks - 1]
    if spec.startswith("last"):
        count = int(spec.replace("last-", "").replace("last", ""))
        return list(range(total_blocks - count, total_blocks))
    return [int(part) for part in spec.replace(".", ",").split(",") if part.strip()]


class ConditionalPatchEmbed(nn.Module):
    """在原 DINO patch embedding 后注入本次前向的类别条件。

    用注册的 nn.Module 包装原层，而不是闭包替换 forward，保证 Lightning
    深拷贝出的 EMA 模型有独立的条件状态，避免两套模型互相写入条件。
    """

    def __init__(self, patch_embed: nn.Module):
        super().__init__()
        self.original = patch_embed  # 保留原 DINO patch embedding 的结构和参数
        self._condition = None  # 仅保存本次 forward 的临时条件，不是可学习参数

    def set_condition(self, condition) -> None:
        """设置 [B,1,1,D] 条件；None 表示本次前向不注入。"""
        self._condition = condition

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """先执行原 patch embedding，再把 y 条件广播加到每个 patch token。"""
        tokens = self.original(x)  # 图像 [B,3,H,W] → patch 网格特征 [B,h,w,D]
        if self._condition is not None:
            tokens = tokens + self._condition.to(device=tokens.device, dtype=tokens.dtype)  # [B,h,w,D] + 广播的 [B,1,1,D]
        return tokens  # [B,h,w,D]，随后才进入 DINO blocks


class DINOv3LoRAEncoder(nn.Module):
    """构建生成侧 DINO，控制 LoRA、可训练参数、y 条件和选层聚合。

    label_cond=True 时，y 在 patch embedding 后、blocks 前注入；y 是训练
    时已经 drop 的标签，因此与 PixelDiT 的 CFG null 类1000语义一致。
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        weights_path: str | None = None,
        lora_rank: int = 64,
        lora_alpha: float = 64.0,
        lora_dropout: float = 0.0,
        lora_target_modules="qkv,proj,fc1,fc2",
        train_patch_embed: bool = True,
        train_final_norm: bool = False,
        final_norm_affine: bool = True,
        aggregation: str = "mls",
        layers="default",
        label_cond: bool = False,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.model = load_dino_v3(model_name, weights_path)  # 严格加载预训练 DINOv3 backbone
        self.embed_dim = self.model.embed_dim  # DINO 特征宽度 D，B/16 为768，L/16 为1024
        self.patch_size = self.model.patch_size  # DINO patch 边长，当前支持的模型均为16
        self.aggregation = aggregation
        self.layer_indices = _resolve_layers(model_name.lower(), layers, len(self.model.blocks))
        self.label_cond = bool(label_cond)
        if not final_norm_affine:
            # 只移除 final LN 的 gamma/beta，保留归一化本身和 eps。
            norm = self.model.norm
            self.model.norm = nn.LayerNorm(
                norm.normalized_shape, eps=norm.eps, elementwise_affine=False,
                device=norm.weight.device, dtype=norm.weight.dtype,
            ).train(norm.training)

        self.model.requires_grad_(False)  # 先冻结整个预训练 backbone，再单独开放增量参数
        self.num_lora_modules = inject_dino_lora(
            self.model, r=lora_rank, alpha=lora_alpha, dropout=lora_dropout,
            target_modules=lora_target_modules,
        )  # 在指定 attention/MLP 线性层加入可训练 LoRA，原权重仍冻结
        self.model.patch_embed.requires_grad_(train_patch_embed)  # 配置允许时，让 DINO 输入投影适应带噪图像
        self.model.norm.requires_grad_(train_final_norm)  # 默认 final LN 也冻结；block 内 norm 始终冻结
        if self.label_cond:
            self.model.patch_embed = ConditionalPatchEmbed(self.model.patch_embed)  # 包装后可在 blocks 前注入 y
            self.y_embedder = ClassEmbedder(num_classes + 1, self.embed_dim)  # 类别/null id → [B,D]，独立于 PixelDiT 主 y 条件

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """提取带噪图像的 DINO patch token，并做 final LN 和选层聚合。

        x [B,3,H,W] 是不额外归一化的带噪输入；y [B] 可选。返回 [B,N,D]，
        不包含 CLS/register token。MLS=选层均值+最后选层的空间均值。
        """
        if self.label_cond:
            condition = self.y_embedder(y).reshape(x.shape[0], 1, 1, self.embed_dim)  # y [B] → [B,1,1,D]，供 patch 网格广播
            self.model.patch_embed.set_condition(condition)  # 在 patch embedding 后、blocks 前生效
        try:
            n = 1 if self.layer_indices is None else self.layer_indices  # None 表示只取最后一层；否则取配置层列表
            outputs = self.model.get_intermediate_layers(
                x, n=n, reshape=False, return_class_token=False, norm=True,
            )  # 每层均先过 final LN，再去掉 CLS/register：[L 个 [B,N,D]]
            tokens = torch.stack(outputs, dim=0).mean(dim=0)  # 多层特征逐位置求均值：[L,B,N,D] → [B,N,D]
            if self.aggregation == "mls":
                tokens = tokens + outputs[-1].mean(dim=1, keepdim=True)  # 再加最后选层的全图语义 [B,1,D]，广播到 N 个 patch
            return tokens  # [B,N,D]
        finally:
            if self.label_cond:
                self.model.patch_embed.set_condition(None)  # 清除临时条件，防止下次前向或 EMA 深拷贝时残留
