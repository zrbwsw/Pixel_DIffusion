"""按配置分别构建 iREPA teacher 和生成侧 DINO+LoRA student。

teacher：干净图像 → 冻结 DINO → 最后一层 normalized patch tokens。
student：带噪图像 → DINO+LoRA → 选层 / MLS → 生成侧 token。
"""

from __future__ import annotations
import torch.nn as nn
from irepa_teacher import FrozenDINOv3Encoder

_DINOV3_TOTAL_BLOCKS = {
    "dinov3_vits16": 12,
    "dinov3_vitb16": 12,
    "dinov3_vitl16": 24,
}

_DINOV3_EMBED_DIMS = {
    "dinov3_vits16": 384,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
}


def dinov3_embed_dim(model_name: str) -> int:
    """不加载模型直接查特征维；noirepa 模式下用它代替 teacher.embed_dim。"""
    return _DINOV3_EMBED_DIMS[model_name.lower()]

def _backbone_kwargs(args) -> dict:
    # 共享模型名和权重路径；两条路径各自加载独立实例。
    return dict(model_name=args.dino_model_name.lower(), weights_path=args.dino_weights or None, )


def _resolve_student_layers(args, total_blocks_hint: int = 12):
    """将配置中的层名转为从0开始的 block 索引。"""
    from dino_v3_lora import RAEV2_DEFAULT_LAYERS, parse_layer_spec

    aggregation = args.dino_aggregation.lower()
    layers_spec = args.dino_layers
    model_name = args.dino_model_name
    if layers_spec and layers_spec.strip().lower() == "default":
        return aggregation, RAEV2_DEFAULT_LAYERS.get(model_name, [2, 5, 8, 11])

    total_blocks = _DINOV3_TOTAL_BLOCKS.get(model_name, total_blocks_hint)
    return aggregation, parse_layer_spec(layers_spec, total_blocks)


def build_irepa_teacher(args) -> nn.Module:
    """只读取 backbone 标识；teacher 内部冻结并 eval，不读取 student 开关。"""
    return FrozenDINOv3Encoder(**_backbone_kwargs(args))


def build_dino_student(args) -> nn.Module:
    """启用 LoRA；按配置训练 patch embed 和 final LN，CLS token 保持冻结。"""
    # 延迟导入，单独构建 teacher 时不加载 LoRA 包装。
    from dino_v3_lora import DINOv3PatchTokenEncoder

    aggregation, layer_indices = _resolve_student_layers(args)
    return DINOv3PatchTokenEncoder(
        **_backbone_kwargs(args),
        aggregation=aggregation,
        layer_indices=layer_indices,
        use_lora=True,
        lora_r=args.dino_lora_r,
        lora_alpha=args.dino_lora_alpha,
        lora_dropout=args.dino_lora_dropout,
        lora_target_modules=args.dino_lora_target_modules,
        train_patch_embed=args.dino_train_patch_embed,
        train_cls_token=False,
        train_norm=args.dino_train_final_norm,
        final_norm_affine=args.dino_generation_final_norm_affine,
        mls_layer_norms=getattr(args, "dino_mls_layer_norms", False),
        time_cond=args.dino_student_time_cond,
        label_cond=args.dino_student_label_cond,
        num_classes=args.class_num,
    )
