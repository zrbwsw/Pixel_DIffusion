"""frozen vision-encoder features 的 spatial normalization。

这个模块在 `projection_loss.py` 之前使用。

输入约定：
- feat 的 shape 是 [B, T, D]。
- B 是 batch size。
- T 是 spatial patch tokens 数量，例如 16 * 16 = 256。
- D 是 feature/channel dimension，例如 768。

关键维度：
- spatial_zscore 在 dim=1 上归一化，也就是在 T 个 patch tokens 上归一化。
- 对每张图 b、每个 feature channel d，计算所有 spatial token positions t 上的 mean/std。
- 这和 projection_loss.py 的 cosine normalization 不同；后者使用 dim=-1，
  即对每个 token 的 feature dimension D 做归一化。
"""
import torch


ALL_SPNORM_METHODS = ["none", "zscore"]


def spatial_zscore(feat: torch.Tensor, alpha: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """
    沿 spatial/token 维度做 z-score normalization。

    参数：
        feat: (B, T, D) patch tokens
        alpha: spatial mean subtraction 的缩放系数，默认 1.0
        eps: 数值稳定用的小常数

    返回：
        归一化后的 features，shape 仍是 (B, T, D)
    """
    # mean 的 shape 是 [B, 1, D]。
    # 对每张图、每个 feature channel，在所有 patch tokens T 上取平均。
    mean = feat.mean(dim=1, keepdim=True)

    # std 的 shape 是 [B, 1, D]。
    # 它衡量每个 channel 在 spatial patch tokens 上的变化幅度。
    std = feat.std(dim=1, keepdim=True)

    # 减去带 alpha 缩放的 spatial mean，再除以 spatial std。
    # shape 仍然是 [B, T, D]。
    return (feat - alpha * mean) / (std + eps)


class SpatialNormalization:
    """
    spatial normalization 的简单 wrapper，用于兼容不同配置。

    当前只支持 "none" 和 "zscore"。
    """
    def __init__(self, method: str, *, eps: float = 1e-6):
        # 只有 ALL_SPNORM_METHODS 里的 method 是合法配置。
        assert method in ALL_SPNORM_METHODS, f"Invalid method: {method}. Must be one of {ALL_SPNORM_METHODS}"
        self.method = method
        self.eps = eps

    def __call__(self, feat: torch.Tensor, **kwargs) -> torch.Tensor:
        # vanilla REPA 使用 no-op，不做 spatial normalization。
        if self.method == "none":
            return feat

        # iREPA 使用 zscore；论文/代码里 `zscore_alpha` 通常在 0.6--0.8。
        alpha = kwargs.get('zscore_alpha', 1.0)
        return spatial_zscore(feat, alpha=alpha, eps=self.eps)
