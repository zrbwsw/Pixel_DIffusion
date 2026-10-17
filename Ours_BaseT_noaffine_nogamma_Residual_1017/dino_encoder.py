"""按功能独立构建生成空间编码器与 iREPA 正则化编码器。

J18-new-B 使用 DINOv3-B/16。两条独立路径均输入
``[B, 3, 256, 256]``，patch size 16 产生 ``16x16=256`` 个 768 维 token。

* generation：塑造生成空间；LoRA、可训练 patch embedding、无仿射最终 LN、可选 MLS。
* alignment：提供 iREPA 正则化目标；原始预训练 LN、无 LoRA/MLS、冻结 raw-last。

两者只共享模型类型和预训练加载工具，不共享模型实例，也不是蒸馏关系。

DINOv3 使用由坐标生成的 RoPE，不存在可训练 absolute position embedding；当前
接口因此只暴露 patch embedding 的 train/freeze 选择。

当前活动代码只支持 DINOv3。历史 DINOv1 构建器已移入 ``code_archive``，避免
阅读当前 class-to-image 数据流时还要追踪另一套模型加载逻辑。
"""

from __future__ import annotations

import torch.nn as nn

from dino_alignment_encoder import DINOv3AlignmentEncoder
from dino_v3_lora import (
    DINOv3PatchTokenEncoder,
    RAEV2_DEFAULT_LAYERS,
    parse_layer_spec,
)


_DINOV3_NAMES = {"dinov3_vitb16", "dinov3_vitl16", "dinov3_vits16", "dinov3"}
_DINOV3_TOTAL_BLOCKS = {
    "dinov3_vits16": 12,
    "dinov3_vitb16": 12,
    "dinov3_vitl16": 24,
}


def _validate_model_name(model_name: str) -> str:
    name = model_name.lower()
    if name in _DINOV3_NAMES or name.startswith("dinov3"):
        return name
    raise ValueError(
        f"Current class-to-image code supports DINOv3 only; got {model_name!r}. "
        f"Known names: {sorted(_DINOV3_NAMES)}"
    )


def _resolve_dinov3_layers(args, total_blocks_hint: int = 12):
    """解析当前 encoder 的聚合类型和 0-based block 索引。"""

    aggregation = str(getattr(args, "dino_aggregation", "none")).lower()
    layers_spec = getattr(args, "dino_layers", None)
    if layers_spec is not None and str(layers_spec).strip() == "":
        layers_spec = None

    model_name = getattr(args, "dino_model_name", "dinov3_vitb16")
    if isinstance(layers_spec, str) and layers_spec.strip().lower() == "default":
        return aggregation, RAEV2_DEFAULT_LAYERS.get(model_name, [2, 5, 8, 11])

    total_blocks_hint = _DINOV3_TOTAL_BLOCKS.get(model_name, total_blocks_hint)
    indices = parse_layer_spec(layers_spec, total_blocks_hint) if layers_spec else None
    return aggregation, indices


def build_generation_encoder(args) -> nn.Module:
    """生成路径：LoRA + RAEv2 风格无 affine 最终 LN / 可选 MLS。"""
    model_name = _validate_model_name(
        getattr(args, "dino_model_name", "dinov3_vitb16")
    )
    weights = getattr(args, "dino_weights", None) or None
    aggregation, layer_indices = _resolve_dinov3_layers(args)
    return DINOv3PatchTokenEncoder(
        model_name=model_name,
        weights_path=weights,
        aggregation=aggregation,
        layer_indices=layer_indices,
        use_lora=True,
        lora_r=int(getattr(args, "dino_lora_r", 16)),
        lora_alpha=float(getattr(args, "dino_lora_alpha", 16.0)),
        lora_dropout=float(getattr(args, "dino_lora_dropout", 0.0)),
        lora_target_modules=getattr(
            args, "dino_lora_target_modules", "qkv,proj,fc1,fc2"
        ),
        train_patch_embed=bool(getattr(args, "dino_train_patch_embed", True)),
        train_cls_token=False,
        train_norm=False,
    )


def build_alignment_encoder(args) -> nn.Module:
    """iREPA 路径：独立加载未改 final LN 的预训练 DINOv3。

    不读取生成路径的 LoRA、MLS 或输入标准化设置。
    """
    return DINOv3AlignmentEncoder(
        model_name=_validate_model_name(getattr(args, "dino_model_name", "dinov3_vitb16")),
        weights_path=getattr(args, "dino_weights", None) or None,
        resolution=int(getattr(args, "img_size", 256)),
    )
