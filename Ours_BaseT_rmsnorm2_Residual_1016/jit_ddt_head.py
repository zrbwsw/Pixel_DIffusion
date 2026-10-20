"""可选的 DDT-style 双流像素预测头。

Reference impl: RAEv2/src/stage2/models/DDT.py (DDTDecoderBlock, DDTFinalLayer)
and jit/model_jit.py (BottleneckPatchEmbed, Attention, SwiGLUFFN, FinalLayer).

以 JiT-B/16 主干、2048 维预测头为例：

* base stream：JiT-B 输出 ``z_t [B,256,768]``；
* head stream：noisy image ``[B,3,256,256]`` 经独立 bottleneck patch embed 得到
  ``[B,256,2048]``；
* condition：``z_t`` 线性投影到 ``[B,256,2048]``，只通过 per-token AdaLN
  调制 head stream，不与其拼接；
* 输出：每个 patch 预测 ``16*16*3=768`` 个值，得到 ``[B,256,768]``，随后由
  JiT.unpatchify 还原 ``[B,3,256,256]``。

The caller (DINOJiTDenoiser) keeps JiT's unpatchify, so the head returns patch tokens
of shape [B, N, patch_size**2 * out_channels].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse JiT's own building blocks so the head matches the trunk exactly.
from model_jit import Attention, SwiGLUFFN, BottleneckPatchEmbed
from util.model_util import RMSNorm, VisionRotaryEmbeddingFast


def modulate_pertoken(x, shift, scale):
    """逐 token AdaLN；三项都是 [B,N,C]，区别于 JiT trunk 的全局 [B,C]。"""
    return x * (1 + scale) + shift


class SimplePatchEmbed(nn.Module):
    """RAEv2-style single-conv patch embedding.

    `Conv2d(in_chans, embed_dim, kernel_size=patch, stride=patch)` — one stage,
    no PCA-like bottleneck. Matches RAEv2 `timm.PatchEmbed` semantics. This
    alternative to JiT's two-stage `BottleneckPatchEmbed` isolates the effect
    of the intermediate channel bottleneck on head capacity.
    """

    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        return self.proj(x).flatten(2).transpose(1, 2)


class DDTHeadBlock(nn.Module):
    """使用 per-token condition 的 JiTBlock（对应 RAEv2 DDTDecoderBlock）。

    JiT trunk 用全局 ``c [B,C]`` 在 token 轴广播；这里 ``z_t [B,N,C]`` 已经为
    每个空间位置提供独立 condition，所以六组 AdaLN 参数无需 unsqueeze。
    """

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
        # 2048 维预测头示例：x/z_t 均为 [B,256,2048]；16 heads，每头 128 维。
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(z_t).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate_pertoken(self.norm1(x), shift_msa, scale_msa), rope=rope)
        x = x + gate_mlp * self.mlp(modulate_pertoken(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DDTHeadFinalLayer(nn.Module):
    """逐 token AdaLN 输出层：[B,256,2048] -> [B,256,768]。"""

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
    """用于 JiT pixel diffusion 的 DDT head（RAEv2-style，仅用 RoPE）。

    head stream 不添加绝对位置编码，位置完全由 head-local 2D RoPE 提供。这样宽度
    2048、16 heads（head_dim=128）可以使用与自身匹配的 RoPE，而不会误用 JiT-B
    trunk 的 head_dim=64 RoPE。

    Args:
        hidden_size: head working width (e.g. 768 or 2048).
        patch_size: JiT patch size (16). Output is patch_size**2 * in_channels per token.
        in_channels: image channels (3).
        bottleneck_dim: BottleneckPatchEmbed pca_dim (128 for B/16). Ignored when embed_type='patchembed'.
        input_size: image side (256).
        depth: number of head decoder blocks (default 2).
        num_heads: head attention heads (e.g. 12 at width 768, or 16 at width 2048).
        cond_hidden_size: dim of z_t coming from JiT (default = hidden_size -> Identity proj).
        time_reinject: if True, use the RAEv2 post-base ordering
            ``z_cond = proj(SiLU(z_t + t_emb))``; otherwise use
            ``z_cond = proj(z_t)``. This option does not add time tokens to the
            JiT sequence.
        embed_type: 'bottleneck' (JiT-native 3->128->hidden two-stage) or
                    'patchembed' (RAEv2-style single Conv2d 3->hidden).
    """

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
        self.time_reinject = bool(time_reinject)
        self.num_heads = num_heads
        self.embed_type = str(embed_type)
        cond_hidden_size = hidden_size if cond_hidden_size is None else cond_hidden_size

        # 2048 维 bottleneck 示例：3 -> 128 -> 2048。对 256/P16，输出为
        # [B,256,2048]；patchembed 分支用于比较单层卷积与两阶段嵌入。
        if self.embed_type == "bottleneck":
            self.x_embedder = BottleneckPatchEmbed(
                input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True
            )
        elif self.embed_type == "patchembed":
            self.x_embedder = SimplePatchEmbed(
                input_size, patch_size, in_channels, hidden_size, bias=True
            )
        else:
            raise ValueError(
                f"Unknown embed_type={self.embed_type!r}; expected 'bottleneck' or 'patchembed'."
            )
        # 2048 维预测头将 JiT condition [B,256,768] 投影到 [B,256,2048]。
        self.z_proj = (
            nn.Linear(cond_hidden_size, hidden_size)
            if cond_hidden_size != hidden_size
            else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [DDTHeadBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = DDTHeadFinalLayer(hidden_size, patch_size, in_channels)
        # 16x16 空间网格的 head-local 2D RoPE；没有 class token 前缀。
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

        # Patch embedder init (both variants get xavier on conv weights, bias=0).
        if self.embed_type == "bottleneck":
            w1 = self.x_embedder.proj1.weight.data
            nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            w2 = self.x_embedder.proj2.weight.data
            nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj2.bias, 0)
        else:  # patchembed (single Conv2d)
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            if self.x_embedder.proj.bias is not None:
                nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Zero-out adaLN modulations (decoder blocks + final layer) so the head
        # starts as a near-identity/zero residual, matching JiT/RAEv2 init.
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        # Zero-out output projection so initial head output is 0 (stable warm start).
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _build_condition(self, z_t, t_emb=None):
        """Build the per-token head condition without changing JiT tokens.

        The RAEv2-style branch deliberately adds timestep information at the
        JiT/base width before the base-to-head projector:

            [B,N,C_base] + [B,1,C_base] -> SiLU -> [B,N,C_head].

        This differs from projecting ``z_t`` and ``t_emb`` independently before
        the nonlinearity, and exactly preserves the ``projector(SiLU(base+t))``
        ordering used at RAEv2's base/head boundary.
        """
        if not self.time_reinject:
            return self.z_proj(z_t)
        if t_emb is None:
            raise ValueError("t_emb is required when DDTHead time_reinject=True.")
        if t_emb.ndim != 2 or t_emb.shape[0] != z_t.shape[0] or t_emb.shape[-1] != z_t.shape[-1]:
            raise ValueError(
                "RAEv2-style DDTHead timestep fusion expects "
                f"z_t=[B,N,C] and t_emb=[B,C], got {tuple(z_t.shape)} and {tuple(t_emb.shape)}."
            )
        base_with_time = z_t + t_emb.to(device=z_t.device, dtype=z_t.dtype).unsqueeze(1)
        return self.z_proj(F.silu(base_with_time))

    def forward(self, noisy_image, z_t, t_emb=None):
        """运行 head stream。

        ``noisy_image [B,3,256,256]`` 使用扩散像素范围，不是 DINO 标准化值；
        ``z_t [B,256,768]`` 来自 JiT base stream。默认关闭
        ``time_reinject``；RAEv2-style 消融开启后计算
        ``z_cond = z_proj(SiLU(z_t + t_emb[:,None]))``。返回
        ``[B,256,768]`` patch prediction。

        head-local RoPE 提供位置，head stream 不添加 absolute pos_embed。
        """
        z_cond = self._build_condition(z_t, t_emb)

        x = self.x_embedder(noisy_image)  # [B,3,256,256] -> [B,256,2048]
        for block in self.blocks:
            x = block(x, z_cond, self.feat_rope)
        x = self.final_layer(x, z_cond)
        return x
