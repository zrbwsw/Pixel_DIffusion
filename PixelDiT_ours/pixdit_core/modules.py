"""PixelDiT v1 与 DINO 融合分支共用的基础组件。

包含两类位置编码（像素绝对 sin-cos、注意力 RoPE）、时间/类别条件嵌入、
RMSNorm、注意力、前馈网络及原生 patch/pixel 输出层。这里只定义通用计算；
具体的双尺度主干在 pixeldit_c2i.py，DINO 支路在 dino_*.py。
"""

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def apply_adaln(x, shift, scale):
    """执行条件仿射调制：x*(1+scale)+shift，三者末维均为 C 并可广播。"""
    return x * (1 + scale) + shift  # x/shift/scale [B,N,C] 或 [B,1,C]，输出形状不变


class TimestepConditioner(nn.Module):
    """把连续扩散时间 t 变成可用于 adaLN/PgT 的全局条件向量。

    先用多频 sin-cos 表示 t，再用两层 MLP 学习非线性时间特征；
    输入 t [B]，输出时间嵌入 [B,C]。
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )  # [B,256] → [B,C] → [B,C]
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10):
        """把标量时间展开成不同频率的 cos/sin，输出 [B,dim]。"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )  # [dim//2]，覆盖不同时间尺度
        args = t[..., None].float() * freqs[None, ...]  # [B,1] × [1,dim//2] → [B,dim//2]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B,dim]
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)  # 奇数维时补一个0通道
        return embedding

    def forward(self, t):
        """将 [B] 的连续时间映射为 [B,C]，并对齐 MLP 参数 dtype。"""
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)  # [B] → [B,256]
        mlp_dtype = next(self.mlp.parameters()).dtype
        if t_freq.dtype != mlp_dtype:
            t_freq = t_freq.to(mlp_dtype)
        t_emb = self.mlp(t_freq)  # [B,256] → [B,C]
        return t_emb


class ClassEmbedder(nn.Module):
    """把整数类别 id 查表为 C 维条件；num_classes 已包含 CFG null 类。"""

    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)  # [num_classes,C] 可学习类别向量表
        self.num_classes = num_classes

    def forward(self, labels):
        """labels [B] → 每个样本一个类别向量 [B,C]。"""
        embeddings = self.embedding_table(labels)  # [B] → [B,C]
        return embeddings


class RMSNorm(nn.Module):
    """在每个 token 的 C 维内部做均方根归一化，并学习逐通道增益。

    与 LayerNorm 不同，它不减均值；主要作用是控制 token 尺度，而不是
    改变 token 数或空间结构。方差用 FP32 计算以减少混合精度误差。
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 可学习逐通道增益 [C]，初始全1
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """输入 [...,C]，输出同形；归一化只发生在最后的通道维。"""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)  # [...,C]，FP32 计算均方
        variance = hidden_states.pow(2).mean(-1, keepdim=True)  # 每个 token 一个均方值 [...,1]
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)  #  RMS≈1，形状不变
        return self.weight * hidden_states.to(input_dtype)  # 乘可学习增益并恢复原 dtype


class FeedForward(nn.Module):
    """逐 token 的 SwiGLU 前馈网络：silu(w1(x)) * w3(x) 后再投回原宽度。"""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, hidden_multiple: int = 0):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)  # SwiGLU 常用的约 2/3 隐藏宽度，保持输入输出均为 dim
        hidden_multiple = int(hidden_multiple)
        if hidden_multiple > 0:
            hidden_dim = ((hidden_dim + hidden_multiple - 1) // hidden_multiple) * hidden_multiple
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # 门控分支 [...,dim] → [...,hidden]
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # 内容分支，与 w1 同形
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # 融合后投回 [...,dim]
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        """对 [...,dim] 的每个 token 独立计算，输出仍为 [...,dim]。"""
        hidden = torch.nn.functional.silu(self.w1(x)) * self.w3(x)  # 两路逐元素相乘形成门控：[...,hidden]
        x = self.w2(self.ffn_dropout(hidden))  # [...,hidden] → [...,dim]
        return x


def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float = 10000.0, scale=16.0):
    """预计算二维 RoPE 复数频率表，供注意力 q/k 注入空间位置。

    dim 是单个 attention head 的宽度；返回 [H*W,dim//2] 的复数张量。
    前一半频率对应 x 坐标，后一半对应 y 坐标，不包含可学习参数。
    """
    x_pos = torch.linspace(0, scale, width)  # [W]，横向坐标
    y_pos = torch.linspace(0, scale, height)  # [H]，纵向坐标
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)  # [H*W]
    x_pos = x_pos.reshape(-1)  # [H*W]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))  # [dim//4]，不同频率的旋转周期
    x_freqs = torch.outer(x_pos, freqs).float()  # [H*W,dim//4]，每个位置的 x 相位
    y_freqs = torch.outer(y_pos, freqs).float()  # [H*W,dim//4]，每个位置的 y 相位
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # 单位复数 exp(i*x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # 单位复数 exp(i*y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)  # [H*W,dim//4,2]
    freqs_cis = freqs_cis.reshape(height * width, -1)  # [H*W,dim//2]
    return freqs_cis

def precompute_freqs_cis_ex2d(dim: int, height: int, width:int, theta: float = 10000.0, scale=1.0):
    if isinstance(scale, float):
        scale = (scale, scale)
    x_pos = torch.linspace(0, width*scale[0], width)
    y_pos = torch.linspace(0, height*scale[1], height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)) # Hc/4
    x_freqs = torch.outer(x_pos, freqs).float() # N Hc/4
    y_freqs = torch.outer(y_pos, freqs).float() # N Hc/4
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1) # N,Hc/4,2
    freqs_cis = freqs_cis.reshape(height*width, -1)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """把 q/k 的相邻通道两两视为复数并乘位置相位，输入输出形状相同。"""
    freqs_cis = freqs_cis[None, :, None, :]  # [N,dim//2] → [1,N,1,dim//2]，广播 batch/head
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # [B,N,heads,head_dim] → 复数 [B,N,heads,head_dim//2]
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # 同形，只旋转 q/k，不处理 v
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)  # 相位旋转后恢复实数通道 [B,N,heads,head_dim]
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)  # 同形
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryAttention(nn.Module):
    """多头自注意力：QK RMSNorm + 二维 RoPE + 缩放点积注意力。

    输入 x [B,N,C] 是一组 patch token；pos 是预计算的 [N,head_dim//2]
    复数位置表。输出仍 [B,N,C]，用于交换 token 之间的信息。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = RMSNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 一次生成 q/k/v：[B,N,C] → [B,N,3C]
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()  # 每个 head 内归一化 q，稳定注意力尺度
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()  # 同样归一化 k
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # 合并多个 head 后再投回 C 维
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, pos, mask) -> torch.Tensor:
        """对 N 个 token 做全连接注意力，mask 可屏蔽不允许访问的位置。"""
        B, N, C = x.shape  # x [B,N,C]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)  # [3,B,N,heads,head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各 [B,N,heads,head_dim]
        q = self.q_norm(q)  # [B,N,heads,head_dim]，QK 归一化不改变形状
        k = self.k_norm(k)  # [B,N,heads,head_dim]
        q, k = apply_rotary_emb(q, k, freqs_cis=pos)  # 给 q/k 注入二维位置；v 不加 RoPE
        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)  # [B,heads,N,head_dim]
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()  # [B,heads,N,head_dim]
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()  # [B,heads,N,head_dim]

        x = scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)  # q·k 决定信息权重，再对 v 加权求和：[B,heads,N,head_dim]

        x = x.transpose(1, 2).reshape(B, N, C)  # 合并所有 head，恢复 [B,N,C]
        x = self.proj(x)  # 跨 head 混合特征：[B,N,C]
        x = self.proj_drop(x)  # 输出 dropout，形状不变
        return x  # [B,N,C]


class MLP(nn.Module):
    """普通两层 GELU 前馈网络，对每个像素/token 独立做非线性变换。"""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)  # [...,dim] → [...,hidden_dim]
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)  # 再投回 [...,dim]
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """保持 [...,dim] 形状，只增加逐位置的通道非线性。"""
        x = self.fc1(x)  # [...,dim] → [...,hidden_dim]
        x = self.act(x)  # GELU 非线性，形状不变
        x = self.drop(x)
        x = self.fc2(x)  # [...,hidden_dim] → [...,dim]
        x = self.drop(x)
        return x


class FinalLayer(nn.Module):
    """PixelDiT v1 的输出头：RMSNorm 后把每个像素 token 映射为 RGB。"""

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6)  # 输出前统一像素特征尺度
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)  # [...,pixel_C] → [...,3]

    def forward(self, x):
        """输入 [B*N,P²,pixel_C]，输出每个像素的 RGB [B*N,P²,3]。"""
        x = self.norm(x)  # [B*N,P²,pixel_C]
        x = self.linear(x)  # [B*N,P²,3]
        return x


class PatchTokenEmbedder(nn.Module):
    """原生 patchify：把每个 patch 的像素向量线性映射为语义 token。

    输入 [B,N,in_chans]，in_chans 通常是 3*P²；输出 [B,N,embed_dim]。
    这里不提取 DINO 特征，是时间加权中的 PixelDiT 原生分支。
    """

    def __init__(
            self,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer = None,
            bias: bool = True,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)  # 每个 patch 独立投影，不与其他 patch 混合
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """输入 [B,N,in_chans]，输出同 token 数的 [B,N,embed_dim]。"""
        x = self.proj(x)  # [B,N,in_chans] → [B,N,embed_dim]
        x = self.norm(x)  # 可选归一化，默认 Identity，形状不变
        return x


class ModulatedDiTBlock(nn.Module):
    """带全局条件的 DiT block，也是 PgT 内部使用的 token 变换单元。

    x [B,N,C] 为 token，c [B,1,C] 为时间等全局条件；先通过 adaLN
    生成六组调制参数，再做 RoPE 注意力和 SwiGLU 两条残差路径。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_hidden_multiple: int = 0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = RotaryAttention(
            hidden_size, num_heads=num_heads, qkv_bias=False,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )  # 跨 token 交换信息，形状保持 [B,N,C]
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(
            hidden_size, int(hidden_size * mlp_ratio),
            drop=proj_drop, hidden_multiple=mlp_hidden_multiple,
        )  # 逐 token 做 SwiGLU 非线性变换
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )  # c [B,1,C] → 六组调制参数 [B,1,6C]

    def forward(self, x, c, pos, mask=None):
        """用全局条件调制注意力/MLP 残差，输入输出均为 [B,N,C]。"""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)  # 各 [B,1,C]
        x = x + gate_msa * self.attn(apply_adaln(self.norm1(x), shift_msa, scale_msa), pos, mask=mask)  # 条件注意力残差：[B,N,C]
        x = x + gate_mlp * self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))  # 条件前馈残差：[B,N,C]
        return x


class PatchFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        if shift.dim() == 2:
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)
        return self.linear(apply_adaln(self.norm(x), shift, scale))

