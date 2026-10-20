"""DINOv3 student/teacher 构建与按角色配置。

默认配置使用 DINOv3-B/16。student 和 teacher 均输入
``[B, 3, 256, 256]``，patch size 16 产生 ``16x16=256`` 个 768 维 token。
关键差异由 ``role`` 决定：

* student：LoRA-r64，预训练 patch embedding 可选择 fine-tune；
* teacher：无 LoRA、完全冻结；
* student 可取多层 MLS，teacher 始终使用 raw last。
* student 可关闭最终 LN 的 affine；teacher 始终保留预训练 affine。
* dino_train_final_norm 仅控制 student 最终 LN gamma/beta 是否训练，默认冻结。

DINOv3 使用由坐标生成的 RoPE，不存在可训练 absolute position embedding；当前
接口提供 patch embedding 与生成侧最终 LN 的 train/freeze 选择。

当前活动代码只支持 DINOv3。历史 DINOv1 构建器已移入 ``code_archive``，避免
阅读当前 class-to-image 数据流时还要追踪另一套模型加载逻辑。
"""

from __future__ import annotations

import torch.nn as nn

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


def _resolve_dinov3_layers_for_role(args, role: str, total_blocks_hint: int = 12):
    """按 teacher/student 角色解析聚合配置。

    Teacher 始终返回 raw-last；student 按配置聚合多层 token。
    """

    if role == "teacher":
        return "none", None
    return _resolve_dinov3_layers(args, total_blocks_hint)


def build_encoder(args, *, role: str) -> nn.Module:
    """构建 frozen teacher 或带 LoRA 的 trainable student。"""

    if role not in ("teacher", "student"):
        raise ValueError(f"role must be 'teacher' or 'student', got {role!r}")

    train_final_norm = bool(getattr(args, "dino_train_final_norm", False))
    if train_final_norm:
        if not bool(getattr(args, "dino_generation_final_norm_affine", True)):
            raise ValueError("dino_train_final_norm requires dino_generation_final_norm_affine=true")
        if getattr(args, "dino_jit_input_mode", "replace") == "native":
            raise ValueError("dino_train_final_norm requires an active generation DINO path")

    model_name = _validate_model_name(
        getattr(args, "dino_model_name", "dinov3_vitb16")
    )
    weights = getattr(args, "dino_weights", None) or None
    aggregation, layer_indices = _resolve_dinov3_layers_for_role(args, role)

    common_kwargs = dict(
        model_name=model_name,
        weights_path=weights,
        aggregation=aggregation,
        layer_indices=layer_indices,
    )
    if role == "teacher":
        return DINOv3PatchTokenEncoder(
            **common_kwargs,
            use_lora=False,
            train_patch_embed=False,
            train_cls_token=False,
            train_norm=False,
        )

    return DINOv3PatchTokenEncoder(
        **common_kwargs,
        use_lora=True,
        lora_r=int(getattr(args, "dino_lora_r", 16)),
        lora_alpha=float(getattr(args, "dino_lora_alpha", 16.0)),
        lora_dropout=float(getattr(args, "dino_lora_dropout", 0.0)),
        lora_target_modules=getattr(
            args, "dino_lora_target_modules", "qkv,proj,fc1,fc2"
        ),
        train_patch_embed=bool(getattr(args, "dino_train_patch_embed", True)),
        train_cls_token=False,
        train_norm=train_final_norm,
        final_norm_affine=bool(
            getattr(args, "dino_generation_final_norm_affine", True)
        ),
    )
