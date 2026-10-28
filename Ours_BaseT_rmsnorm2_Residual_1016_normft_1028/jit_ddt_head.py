"""DDT 像素输出头，参考 RAEv2/src/stage2/models/DDT.py，复用 JiT 的基础层。

带噪图像 → patch embed → DDT blocks → 像素 patch。
JiT 输出 → 可选时间融合 → 投影，作为每个 token 的 AdaLN 条件。
输出 [B,N,patch_size²×3]，由去噪器 unpatchify 还原图像。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 iREPA/JiT 的 attention、MLP 和 patch embedding。
from model_jit import Attention, SwiGLUFFN, BottleneckPatchEmbed
from util.model_util import RMSNorm, VisionRotaryEmbeddingFast


def modulate_pertoken(x, shift, scale):
    """逐 token 调制；x、shift、scale 均为 [B,N,C]。"""
    return x * (1 + scale) + shift


class SimplePatchEmbed(nn.Module):
    """单层卷积：图像 [B,3,H,W] → patch tokens [B,N,C]。"""

    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class DDTHeadBlock(nn.Module):
    """由 JiT token 提供 AdaLN 条件，依次执行 attention 和 MLP 残差。"""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, z_t, rope):
        # 条件生成 attention/MLP 各自的 shift、scale 和残差 gate。
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(z_t).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate_pertoken(self.norm1(x), shift_msa, scale_msa), rope=rope)
        x = x + gate_mlp * self.mlp(modulate_pertoken(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DDTHeadFinalLayer(nn.Module):
    """AdaLN → Linear，将每个 token 映射为一个像素 patch。"""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, z_t):
        shift, scale = self.adaLN_modulation(z_t).chunk(2, dim=-1)
        x = modulate_pertoken(self.norm_final(x), shift, scale)
        return self.linear(x)


class JiTDDTHead(nn.Module):
    """可训练的像素预测分支；二维 RoPE 固定，不添加绝对位置编码。"""

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        in_channels: int = 3,
        bottleneck_dim: int = 128,
        input_size: int = 256,
        depth: int = 2,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        cond_hidden_size: int | None = None,
        time_reinject: bool = False,
        embed_type: str = "bottleneck",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.time_reinject = time_reinject
        self.num_heads = num_heads
        self.embed_type = embed_type
        cond_hidden_size = hidden_size if cond_hidden_size is None else cond_hidden_size

        # 图像入口：bottleneck 为 3→128→head_dim，patchembed 为单层 3→head_dim。
        if self.embed_type == "bottleneck":
            self.x_embedder = BottleneckPatchEmbed(
                input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True
            )
        elif self.embed_type == "patchembed":
            self.x_embedder = SimplePatchEmbed(
                input_size, patch_size, in_channels, hidden_size, bias=True
            )
        # 仅宽度不同时投影条件；如 JiT-B 的768维 → DDT的2048维。
        self.z_proj = (
            nn.Linear(cond_hidden_size, hidden_size)
            if cond_hidden_size != hidden_size
            else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [DDTHeadBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = DDTHeadFinalLayer(hidden_size, patch_size, in_channels)
        # head 独立的二维 RoPE，没有类别 token。
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0,
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 两种 patch embedding 都用 Xavier 权重、零偏置；保留初始化顺序。
        if self.embed_type == "bottleneck":
            w1 = self.x_embedder.proj1.weight.data
            nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            w2 = self.x_embedder.proj2.weight.data
            nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj2.bias, 0)
        else:  # 单层 patch embedding
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            if self.x_embedder.proj.bias is not None:
                nn.init.constant_(self.x_embedder.proj.bias, 0)

        # AdaLN 置零，初始 DDT blocks 为恒等残差。
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        # 输出投影置零，整个 head 的初始预测为0。
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _build_condition(self, z_t, t_emb=None):
        """先在 JiT 宽度上加时间条件，经 SiLU 后再投影到 DDT 宽度。"""
        if not self.time_reinject:
            return self.z_proj(z_t)
        base_with_time = z_t + t_emb.to(device=z_t.device, dtype=z_t.dtype).unsqueeze(1)
        return self.z_proj(F.silu(base_with_time))

    def forward(self, noisy_image, z_t, t_emb=None):
        """带噪图像走 head 分支；JiT 输出只提供调制条件，不与 head token 拼接。"""
        z_cond = self._build_condition(z_t, t_emb)

        x = self.x_embedder(noisy_image)
        for block in self.blocks:
            x = block(x, z_cond, self.feat_rope)
        x = self.final_layer(x, z_cond)
        return x
