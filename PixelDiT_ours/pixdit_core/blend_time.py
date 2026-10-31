"""计算两路 patchify 的时间融合权重，不改变扩散过程本身。

PixDiT 中 t=0 表示纯噪声、t=1 表示干净图像；本函数产生 raw
patchify 的权重 f(t)。默认 (DINO,raw)=(1-f(t),f(t))，reverse 时
在调用处交换为 (f(t),1-f(t))；扩散过程仍使用原始 t。
"""

import torch


def apply_blend_time(
    t: torch.Tensor,
    transform: str = "smoother",
    key: float = 20.0,
    a: float = 0.1,
) -> torch.Tensor:
    """将 t [B] 变为原生 patchify 的融合权重 f(t) [B]。

    linear/smooth/smoother 是不同平滑程度的多项式；sigmoid 的 key=k
    控制过渡斜率，a 是未归一化 sigmoid 的中心。四种形式都满足端点
    f(0)=0、f(1)=1；a≠0.5 时，归一化 sigmoid 的 f(a) 不严格等于0.5。
    """
    compute_t = t.float() if t.dtype in (torch.float16, torch.bfloat16) else t  # 半精度先转 FP32，减少过渡区舍入误差；[B]
    if transform == "linear":
        return t  # f(t)=t，最简单的线性权重：[B]
    if transform == "smooth":
        return compute_t.square() * (3.0 - 2.0 * compute_t)  # 三次 smoothstep，端点一阶导为0：[B]
    if transform == "smoother":
        return compute_t.pow(3) * (6.0 * compute_t.square() - 15.0 * compute_t + 10.0)  # 五次 smootherstep，端点一/二阶导均为0：[B]
    if transform == "sigmoid":
        low = torch.sigmoid(torch.tensor(-key * a, device=compute_t.device, dtype=compute_t.dtype))  # 未归一化 sigmoid 在 t=0 的值 σ(-ka)
        high = torch.sigmoid(torch.tensor(key * (1.0 - a), device=compute_t.device, dtype=compute_t.dtype))  # 在 t=1 的值 σ(k(1-a))
        return (torch.sigmoid(key * (compute_t - a)) - low) / (high - low)  # 减 low 再除以 high-low，强制端点归一化：[B]
    raise ValueError(f"Unknown dino_time_blend_transform: {transform}")
