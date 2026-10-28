"""REPA/iREPA 使用的 projection loss。

JiT + REPA/iREPA 的数据流：
1. engine_jit.py 用 frozen vision encoder 从 clean image 提取 patch tokens。
   这些 target tokens 会作为 `zs` 传入本文件。
2. model_jit.py 把 noisy image 送进 JiT，在 `encoder_depth` 层取 hidden tokens，
   再经过 trainable projector 映射到 target encoder 的特征维度。
   这些 projected tokens 会作为 `zs_tilde` 传入本文件。
3. projection_loss.py 对 `zs` 和 `zs_tilde` 做逐 patch 对齐。

张量约定：
- B: batch size。
- T: spatial patch tokens 数量，例如 16 * 16 = 256。
- D: feature dimension，例如 DINOv2-B 的 D = 768。
- `zs`: frozen target visual features，shape [B, T, D]。
- `zs_tilde`: projected JiT features，shape [B, T, D]。

两个归一化要区分：
- 本文件的 cosine loss 使用 F.normalize(..., dim=-1)。
  这是对每个 patch token 的 feature 维度 D 做 L2 normalization。
- spnorm.py 使用 mean/std(dim=1)。
  这是对每个 channel D，在 spatial tokens T 上做 z-score。
"""

import inspect  # 用每个 loss class 的 __init__ 签名过滤 kwargs。
import torch
import torch.nn.functional as F  # cosine loss 里用它做 L2 feature normalization。
from typing import Callable, Dict

# =========================================
# 注册表 / 工厂函数 / kwargs 过滤
# =========================================

_LOSS_REGISTRY: Dict[str, Callable[..., "ProjectionLoss"]] = {}

def register_loss(name: str):
    """把 ProjectionLoss 子类注册到字符串名字下。"""
    def deco(cls):
        # 保存 class，后面 `make_projection_loss(name)` 可以按名字实例化。
        _LOSS_REGISTRY[name] = cls
        # 把名字挂到 class 上，方便 debug / introspection。
        cls.__loss_name__ = name
        return cls
    return deco

def available_losses():
    """返回当前已经注册的所有 loss 名字。"""
    return sorted(_LOSS_REGISTRY.keys())

def _apply_aliases(cls, kwargs: dict) -> dict:
    # 每个 class 可以选择定义参数别名，例如 {"temperature": "tau", "t": "tau"}。
    aliases = getattr(cls, "KWARG_ALIASES", None) or {}
    # 复制一份 kwargs，避免别名替换时改动调用方传入的 dict。
    out = dict(kwargs)
    for a, target in aliases.items():
        # 例如 class 声明 {"temperature": "tau"} 时，两个名字都可以接收。
        if a in out and target not in out:
            out[target] = out.pop(a)
    return out

def make_projection_loss(name: str, strict: bool = False, **kwargs) -> "ProjectionLoss":
    """根据注册名构造 projection loss 对象。"""
    if name not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {available_losses()}")
    # 取出注册过的 class，例如 name="cosine" -> CosineProjectionLoss。
    cls = _LOSS_REGISTRY[name]
    # 先处理用户传入的别名，再检查构造函数签名。
    kw = _apply_aliases(cls, kwargs)
    # 读取构造函数参数，这样可以安全忽略无关 kwargs。
    sig = inspect.signature(cls.__init__)
    # 只保留当前 loss class 真正接收的 kwargs。
    valid = {k: v for k, v in kw.items() if k in sig.parameters}
    # 记录被忽略的 kwargs；strict=True 时会报错。
    unused = {k: v for k, v in kw.items() if k not in sig.parameters}
    if strict and unused:
        raise TypeError(f"Unused kwargs for loss '{name}': {sorted(unused)}")
    # 实例化真正的 loss 对象。
    return cls(**valid)

# =========================================
# 基类
# =========================================

class ProjectionLoss:
    """Projection loss 基类。

    所有 projection loss 都实现：
        __call__(zs, zs_tilde, zs_tilde_original=None, **kwargs)

    要求的 shape：
        zs:       [B, T, D]，frozen target encoder patch tokens。
        zs_tilde: [B, T, D]，projected JiT patch tokens。
    """
    def _check(self, zs, zs_tilde):
        # 两个输入都必须是 rank-3 patch-token 张量：batch、token、channel。
        if zs.ndim != 3 or zs_tilde.ndim != 3:
            raise ValueError(f"zs and zs_tilde must be [B,T,D]; got {zs.shape=} {zs_tilde.shape=}")
        # cosine / L2 这种逐 patch loss 要求 token 一一对应。
        if zs.shape != zs_tilde.shape:
            raise ValueError(f"Shape mismatch: {zs.shape=} vs {zs_tilde.shape=}")

    def __call__(self, zs, zs_tilde, **kwargs):
        # 子类必须实现具体计算。
        raise NotImplementedError

# =========================================
# Cosine loss
# =========================================

@register_loss("cosine")
class CosineProjectionLoss(ProjectionLoss):
    """逐 patch negative cosine similarity。

    这是本地 JiT 代码里 REPA/iREPA 默认的 alignment loss。
    它通过最小化 `-cosine` 来最大化 cosine similarity。
    """

    # 这个 loss 当前没有超参数。除非 strict=True，否则额外 kwargs 会被
    # make_projection_loss 忽略。
    def __init__(self, **kwargs):
        pass

    def __call__(self, zs, zs_tilde, zs_tilde_original=None, **kwargs):
        """计算 mean negative cosine similarity。

        参数：
            zs:
                vision encoder 产生的 frozen target tokens，shape [B, T, D]。
                典型来源：DINOv2 clean-image patch tokens。
            zs_tilde:
                JiT hidden tokens 经过 trainable projector 后的结果，shape [B, T, D]。
            zs_tilde_original:
                projector 之前的 JiT tokens。这里不用，但保留给 gram matrix
                或其他需要 original hidden states 的 loss。

        返回：
            scalar loss。因为这里是 `-cosine`，所以数值可以是负的。
        """
        # 检查两个输入都是 [B,T,D]，并且 shape 完全一致。
        self._check(zs, zs_tilde)

        # 对每个 patch token vector 在 feature/channel 维度做 normalization。
        # dim=-1 表示 D，不是 T：
        #   归一化前：zs[b, t, :] 可以有任意 L2 norm
        #   归一化后：||zs[b, t, :]||_2 = 1
        # shape 仍然是 [B,T,D]。
        zs = F.normalize(zs, dim=-1)

        # 对 projected JiT tokens 做同样的 per-token feature-vector normalization。
        # 这样会去掉 magnitude mismatch，只保留方向信息。
        # shape 仍然是 [B,T,D]。
        zs_tilde = F.normalize(zs_tilde, dim=-1)

        # normalize 后，patch-wise cosine similarity 就是 dot product。
        # 先在 D 上逐元素相乘，再对 D 求和。
        # 输出 shape 是 [B,T]，每个 image patch token 一个 cosine 值。
        cos_sim = (zs * zs_tilde).sum(dim=-1)

        # REPA/iREPA 最小化 negative cosine，所以对齐越好，这个值越负。
        # 例如某个 token 的 cosine=0.7，则该 token 的 loss=-0.7。
        loss = -cos_sim

        # 对 batch B 和 spatial tokens T 同时取平均，得到一个 scalar。
        return loss.mean()
