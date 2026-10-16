# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
"""原始 JiT Transformer 定义，以及 J18-new 复用的 trunk building blocks。

标准 ``JiT.forward`` 是 image -> BottleneckPatchEmbed -> JiT blocks ->
FinalLayer。J18-new 不直接调用这个 forward，而是在
``dino_jit_denoiser._jit_forward_from_tokens`` 中接收预先构造的空间 token。历史
J18-new 直接使用 DINOv3-B MLS token ``[B,256,768]``；Residual 变体则在进入
blocks 前计算 native PatchEmbed + t-gated DINO residual。类别/时间 embedding、
32 个 in-context class token、RoPE 和 AdaLN 仍完全复用本文件。

形状里的 ``B`` 是单个 DDP rank 的本地 batch，不是有效全局 batch 256。
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm
from projectors import ProjectionLayer


def modulate(x, shift, scale):
    """全局 AdaLN：x [B,N,C]，shift/scale [B,C]，在 N 个 token 上广播。"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BottleneckPatchEmbed(nn.Module):
    """JiT 原生两阶段图像 patch embedding。

    对 256x256、patch16：``[B,3,256,256] -> [B,128,16,16] ->
    [B,768,16,16] -> [B,256,768]``（JiT-B）。历史 J18-new 的 JiT trunk 输入
    来自 DINO，因此该模块被冻结并绕过；Residual 变体重新启用它作为稳定的
    noisy-pixel 主路。DDTHead 另有一个同类 embedder，把 noisy image 嵌入宽度
    2048 的 head stream，两者参数不共享。
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        # B 是本 rank batch；C=3，H=W=256。flatten(2) 把 16x16 网格展成 N=256。
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """把每个样本的标量 t ``[B]`` 映射为全局 condition ``[B,C]``。

    JiT-B 中 C=768。J18-new 的 t 是干净图像权重，取值 0--1；该 MLP 不要求
    t 是离散 diffusion step，因此可直接处理连续 rectified-flow 时间。
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # [B] -> sinusoidal [B,256] -> MLP [B,hidden_size]
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """类别 embedding：0--999 是 ImageNet class，1000 是 CFG null class。

    label dropout 本身在 denoiser 中完成；这里仅执行 ``[B] -> [B,768]`` lookup。
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    """显式 attention；Q/K matmul 强制 FP32，输出再与 value 相乘。

    JiT-B 的 q/k/v 为 ``[B,12,N,64]``。插入 class token 前 N=256，之后 N=288。
    外层训练使用 BF16 autocast，但 logits 的矩阵乘法和 softmax 在这里以 FP32
    计算，以降低注意力数值误差。
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).cuda()

    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Attention(nn.Module):
    """带 QK RMSNorm 与二维 RoPE 的多头自注意力，输入输出均为 [B,N,C]。"""
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        # JiT-B: C=768, heads=12, head_dim=64。
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    """SwiGLU 前馈层；只改变最后一维的内部宽度，输出仍为 [B,N,C]。"""
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """原始 JiT 像素输出层。

    JiT-B/16 会把 ``[B,256,768]`` 映射成每 patch 768 个 RGB 像素值。
    J18-new 启用 DDTHead 后本层被冻结且不参与 forward。
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    """一个 AdaLN-gated Transformer block，序列形状在 block 内保持不变。

    全局 condition ``c=t_emb+y_emb`` 为 ``[B,768]``，产生 attention/MLP 各自的
    shift、scale 和 residual gate；这些参数在 token 维广播。
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x,  c, feat_rope=None):
        # 六组调制量都是 [B,C]；x 是 [B,N,C]。
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class JiT(nn.Module):
    """Just image Transformer 主干。

    J18-new 使用的 ``JiT-B/16`` 固定配置为：depth=12、hidden=768、heads=12、
    patch=16、32 个 class token 在 block index 4 前插入。DINOv3-B 的输出宽度
    也是 768，因此无需额外维度适配即可替换原生 patch token。
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        encoder_depth=8,
        z_dims=[],
        projector_dim=2048,
        projection_layer_type="mlp",
        proj_kwargs_kernel_size=3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes

        self.encoder_depth = encoder_depth
        self.z_dims = z_dims
        self.projector_dim = projector_dim
        self.proj_kwargs_kernel_size = proj_kwargs_kernel_size
        self.projection_layer_type = projection_layer_type

        # 两个全局条件均为 [B,hidden]，在 forward 中相加成 c。
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # 原始 JiT 图像入口。J18-new trunk 不调用它，但 DDTHead 使用同样的设计。
        self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)

        # 256/16=16，因此 num_patches=256；固定二维 sin-cos APE 为 [1,256,768]。
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # 32 个 class token 共用 y_emb，但各自叠加可学习位置 [1,32,768]。
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        # RoPE 不含 class token版本处理 N=256；incontext 版本把前 32 个位置设为
        # 不旋转，空间 token 仍按 16x16 二维网格旋转。
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len
        )

        # JiT-B 共 12 层；中间半段按配置启用 attention/projection dropout。
        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        if len(z_dims) > 0:
            self.projectors = nn.ModuleList([
                ProjectionLayer(projection_layer_type, hidden_size=hidden_size, z_dim=z_dim, projector_dim=projector_dim, proj_kwargs_kernel_size=proj_kwargs_kernel_size) for z_dim in z_dims
            ])
        else:
            self.projectors = None

        # 原始输出头；J18-new 用 DDTHead 替代后会冻结本层。
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        """把 patch-major 像素还原成 NCHW 图像。

        J18-new: ``[B,256,768] -> [B,16,16,16,16,3] -> [B,3,256,256]``。
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """原始 JiT forward；用于 baseline，不是 J18-new 的实际入口。

        ``x [B,3,256,256]``、``t [B]``、``y [B]``。J18-new 从
        ``DINOJiTDenoiser._jit_forward_from_tokens`` 复刻 block 循环，以便直接接收
        DINO token 并把 block 输出交给 DDTHead。
        """
        # [B] -> [B,768]，相加得到所有 block 共用的全局 AdaLN condition。
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        # baseline: [B,3,256,256] -> [B,256,768]，再加固定 APE。
        x = self.x_embedder(x)
        x += self.pos_embed

        zs_tilde = None
        zs_tilde_original = None
        for i, block in enumerate(self.blocks):
            # in-context
            if self.in_context_len > 0 and i == self.in_context_start:
                # JiT-B 在第 5 个 block 前：[B,256,768] -> [B,288,768]。
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens += self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            x = block(x, c, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext)

            if self.projectors is not None and (i + 1) == self.encoder_depth:
                # REPA only aligns the patch tokens, skip the in-context tokens in the front
                if i < self.in_context_start:
                    x_ = x
                else:
                    x_ = x[:, self.in_context_len:]
                zs_tilde = [projector(x_) for projector in self.projectors]
                zs_tilde_original = [x_.clone() for _ in self.projectors]

        # 去掉前 32 个非空间 token，恢复 [B,256,768]。
        x = x[:, self.in_context_len:]

        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)

        return output, zs_tilde, zs_tilde_original


def JiT_B_16(**kwargs):
    """J18-new B-scale trunk：12 层、768 宽度、12 heads、16x16 patch。"""
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=16, **kwargs)

def JiT_B_32(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=32, **kwargs)

def JiT_L_16(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=16, **kwargs)

def JiT_L_32(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=32, **kwargs)

def JiT_H_16(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=16, **kwargs)

def JiT_H_32(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=32, **kwargs)


JiT_models = {
    'JiT-B/16': JiT_B_16,
    'JiT-B/32': JiT_B_32,
    'JiT-L/16': JiT_L_16,
    'JiT-L/32': JiT_L_32,
    'JiT-H/16': JiT_H_16,
    'JiT-H/32': JiT_H_32,
}
