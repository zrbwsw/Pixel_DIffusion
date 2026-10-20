"""REPA/iREPA alignment projectors。

``conv`` 是 iREPA 的空间 2D projector；``mlp`` 是 REPA 的逐 token
``D_in -> D_hidden -> D_hidden -> D_out`` projector。两者都保持
``[B,T,*]`` 的 batch/token 轴，并且在 J18-new 中只服务 alignment 分支。
"""

import torch.nn as nn
import math


ALL_PROJECTION_LAYER_TYPES = ["mlp", "linear", "conv"]

#  Raw REPA projector
def build_mlp(hidden_size, projector_dim, z_dim, **kwargs):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


class ProjectionLayer(nn.Module):
    """统一包装逐 token MLP/Linear 与二维 Conv projector。"""
    def __init__(self, projection_layer_type="mlp", **kwargs):
        super().__init__()
        assert projection_layer_type in ALL_PROJECTION_LAYER_TYPES, f"Unsupported projection layer type: {projection_layer_type}. Must be one of {ALL_PROJECTION_LAYER_TYPES}"
        # self.kwargs = kwargs
        self.projection_layer_type = projection_layer_type 
        self.build_projection_layer(projection_layer_type, **kwargs)

    def build_projection_layer(self, projection_layer_type, **kwargs):
        if projection_layer_type == "mlp":
            self.projection_layer = build_mlp(**kwargs)
        elif projection_layer_type == "linear":
            in_dim  = kwargs.pop("hidden_size")
            out_dim = kwargs.pop("z_dim")
            self.projection_layer = nn.Linear(in_dim, out_dim)
        elif projection_layer_type == "conv": # iREPA
            in_ch  = kwargs.pop("hidden_size")
            out_ch = kwargs.pop("z_dim")
            kernel_size = kwargs.pop("proj_kwargs_kernel_size")
            padding = kernel_size // 2 # to keep spatial dimension
            self.projection_layer = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding)
        else:
            raise ValueError(f"Unsupported projection layer type: {projection_layer_type}")

    def forward(self, x, hw: tuple[int, int] | None = None):
        """投影 token，并保持 batch/token 轴语义。

        Args:
            x: ``[B,T,D]``；J18-new 为 ``[B,256,768]``。
            hw: 非方形网格可显式传 ``(H,W)``；当前 256/16 使用自动推导的 16x16。
        """
        B, T, D = x.shape
        if self.projection_layer_type in ("mlp", "linear"):
            x_ = self.projection_layer(x.reshape(B * T, D))
            return x_.reshape(B, T, -1)

        elif self.projection_layer_type == "conv":
            if hw is None:
                H = W = int(math.isqrt(T))
                assert H * W == T, f"conv projector needs square grid or pass hw; got T={T}"
            else:
                H, W = hw
                assert H * W == T, f"Provided hw={hw} but T={T}"

            # J18-new: [B,256,768] -> [B,768,16,16]。
            x_ = x.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()
            y  = self.projection_layer(x_)                  # [B,768,16,16]
            y  = y.permute(0, 2, 3, 1).contiguous()         # [B, H, W, z_dim]
            return y.reshape(B, T, -1)
