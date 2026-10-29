"""只变换 DINO/JiT 融合权重中的 t，不变换扩散时间或模型的时间条件。"""

import math

import torch

BLEND_TIME_TRANSFORMS = ("linear", "smooth", "smoother", "sigmoid")


def _compute_t(t: torch.Tensor) -> torch.Tensor:
    # FP16/BF16 先转 FP32，避免过渡区提前舍入；linear 原样返回以保持历史行为。
    return t.float() if t.dtype in (torch.float16, torch.bfloat16) else t


def sigmoid_blend_time(t: torch.Tensor, key: float = 100.0, a: float = 0.1) -> torch.Tensor:
    """归一化 sigmoid：key/a 不学习，并保证 f(0)=0、f(1)=1。"""
    compute_t = _compute_t(t)
    low = torch.sigmoid(torch.tensor(-key * a, device=compute_t.device, dtype=compute_t.dtype))
    high = torch.sigmoid(torch.tensor(key * (1.0 - a), device=compute_t.device, dtype=compute_t.dtype))
    return (torch.sigmoid(key * (compute_t - a)) - low) / (high - low)


def apply_blend_time(
    t: torch.Tensor,
    transform: str = "linear",
    key: float = 100.0,
    a: float = 0.1,
) -> torch.Tensor:
    """按配置返回融合权重 f(t)；t 不被原地修改。"""
    if transform == "linear":
        return t
    compute_t = _compute_t(t)
    if transform == "smooth":
        return compute_t.square() * (3.0 - 2.0 * compute_t)  # 3t²-2t³
    if transform == "smoother":
        return compute_t.pow(3) * (6.0 * compute_t.square() - 15.0 * compute_t + 10.0)  # 6t⁵-15t⁴+10t³
    if transform == "sigmoid":
        return sigmoid_blend_time(compute_t, key=key, a=a)
    raise ValueError(f"unknown dino_time_blend_transform: {transform}")


def validate_blend_time(transform: str, key: float, input_mode: str, a: float = 0.1) -> None:
    """启动前预检；禁止在非融合模式下静默忽略变换。"""
    if transform not in BLEND_TIME_TRANSFORMS:
        raise ValueError(f"dino_time_blend_transform must be one of {BLEND_TIME_TRANSFORMS}")
    if not math.isfinite(key):
        raise ValueError("dino_time_blend_key must be finite")
    if not math.isfinite(a):
        raise ValueError("dino_time_blend_a must be finite")
    if transform != "linear" and input_mode != "time_blend":
        raise ValueError(f"dino_time_blend_transform={transform} requires time_blend")
    if transform == "sigmoid":
        if key <= 0:
            raise ValueError("dino_time_blend_key must be positive for sigmoid")
        if not 0.0 < a < 1.0:
            raise ValueError("dino_time_blend_a must be in (0, 1) for sigmoid")
