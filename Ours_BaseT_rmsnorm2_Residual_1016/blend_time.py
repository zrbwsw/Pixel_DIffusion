"""只变换 DINO/JiT 融合权重中的 t，不变换扩散时间或模型的时间条件。"""

import math

import torch


def sigmoid_blend_time(t: torch.Tensor, key: float = 100.0, a: float = 0.1) -> torch.Tensor:
    """f(t) = sigmoid(key * (t - a))；key 控制斜率，a 控制中心点，均不学习。

    FP16/BF16 输入先转 FP32 计算，避免在陡峭的过渡区提前舍入。
    不修改输入；由调用方在用于 token 加权时转换到 token 的 dtype。
    """
    compute_t = t.float() if t.dtype in (torch.float16, torch.bfloat16) else t
    return torch.sigmoid(key * (compute_t - a))


def validate_blend_time(transform: str, key: float, input_mode: str, a: float = 0.1) -> None:
    """启动前预检；禁止在非融合模式下静默忽略变换。"""
    if transform not in {"linear", "sigmoid"}:
        raise ValueError("dino_time_blend_transform must be linear or sigmoid")
    if not math.isfinite(key):
        raise ValueError("dino_time_blend_key must be finite")
    if not math.isfinite(a):
        raise ValueError("dino_time_blend_a must be finite")
    if transform != "linear" and input_mode != "time_blend":
        raise ValueError("dino_time_blend_transform=sigmoid requires time_blend")
