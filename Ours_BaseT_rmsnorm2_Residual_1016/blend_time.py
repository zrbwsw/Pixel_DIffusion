"""只变换 DINO/JiT 融合权重中的 t，不变换扩散时间或模型的时间条件。"""

import math

import torch


def sigmoid_blend_time(t: torch.Tensor, key: float = 100.0) -> torch.Tensor:
    """f(t) = sigmoid(key * (t - 0.1))；key 是固定超参数，不是可学习参数。

    FP16/BF16 输入先转 FP32 计算，避免在陡峭的过渡区提前舍入。
    不修改输入；由调用方在用于 token 加权时转换到 token 的 dtype。
    """
    compute_t = t.float() if t.dtype in (torch.float16, torch.bfloat16) else t
    return torch.sigmoid(key * (compute_t - 0.1))


def validate_blend_time(transform: str, key: float, input_mode: str) -> None:
    """供模型构建和资源预检共用；禁止在非融合模式下静默忽略变换。"""
    if transform not in {"linear", "sigmoid"}:
        raise ValueError("dino_time_blend_transform must be linear or sigmoid")
    if not math.isfinite(key):
        raise ValueError("dino_time_blend_key must be finite")
    if transform != "linear" and input_mode != "time_blend":
        raise ValueError("dino_time_blend_transform=sigmoid requires time_blend")
