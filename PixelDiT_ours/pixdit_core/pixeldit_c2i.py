"""PixelDiT v1 的 patch/pixel 双尺度生成主干。

patch 级先用 token 表示整幅图的语义和全局关系；pixel 级再恢复每个
patch 内部的纹理与像素细节。本目录仅把 patch 级输入改成原生 patchify
与 DINO+LoRA 的时间加权，后续 DiT/PiT 计算仍遵循上游 PixelDiT。
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from pixdit_core.blend_time import apply_blend_time
from pixdit_core.dino_patch_embedder import DINOPatchEmbedder
from pixdit_core.modules import (
    ClassEmbedder,
    FinalLayer,
    FeedForward,
    MLP,
    PatchTokenEmbedder,
    RMSNorm,
    RotaryAttention,
    TimestepConditioner,
    apply_adaln,
    get_2d_sincos_pos_embed,
    get_2d_sincos_pos_embed_from_grid,
    precompute_freqs_cis_2d,
)


class AugmentedDiTBlock(nn.Module):
    """patch 级 Transformer block，用全局注意力交换不同空间区域的信息。

    x 是 [B,N,C] 的 patch token；c 是 [B,1,C] 的时间+类别全局条件。
    条件和 RMSNorm 一起产生 adaLN 的 shift/scale/gate，控制两条残差路径。
    """

    def __init__(self, hidden_size, groups, mlp_ratio=4.0, adaLN_modulation=None):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)  # 注意力前的 token 内归一化，不改变 [B,N,C]
        self.attn = RotaryAttention(hidden_size, num_heads=groups, qkv_bias=False)  # 所有 patch 间做 RoPE 全局注意力
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)  # 前馈网络前的归一化
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)  # 对每个 patch 独立做 SwiGLU 非线性变换
        self.adaLN_modulation = adaLN_modulation if adaLN_modulation is not None else nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )  # 全局条件 [B,1,C] → 六组调制参数 [B,1,6C]

    def forward(self, x, c, pos, mask=None):
        """对 patch token 依次执行条件注意力残差和条件前馈残差。"""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)  # 每项 [B,1,C]，分别调制 attention/MLP
        x = x + gate_msa * self.attn(apply_adaln(self.norm1(x), shift_msa, scale_msa), pos, mask=mask)  # 跨 patch 汇聚语义；输出仍 [B,N,C]
        x = x + gate_mlp * self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))  # 逐 token 非线性细化；[B,N,C]
        return x  # [B,N,C]


class PixelTokenEmbedder(nn.Module):
    """把每个像素变成 pixel token，供后续 PiT 恢复 patch 内细节。

    它与 patch 级 s_embedder 不同：这里先做逐像素线性映射和绝对位置编码，
    再把每个 patch 内的 P² 个像素组成一个序列，输出 [B*N,P²,pixel_C]。
    """

    def __init__(self, in_channels: int, hidden_size_output: int, use_pixel_abs_pos: bool = True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_size_output = int(hidden_size_output)  # 每个像素的通道数 pixel_C
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.proj = nn.Linear(self.in_channels, self.hidden_size_output, bias=True)  # 单个像素 RGB → pixel_C，不混合相邻像素
        self._pos_cache = dict()  # 按图像分辨率缓存固定位置编码，避免每次前向重算

    def _fetch_pixel_pos_image(self, height: int, width: int, device, dtype):
        """返回整图的二维 sin-cos 位置编码，形状 [H*W,pixel_C]。"""
        if height == width:
            key = ("image", height, width)
            if key in self._pos_cache:
                pe = self._pos_cache[key]
                return pe.to(device=device, dtype=dtype)
            pos_np = get_2d_sincos_pos_embed(self.hidden_size_output, height)  # 正方形图像直接由边长生成 [H*W,pixel_C]
            pos = torch.from_numpy(pos_np).to(device=device, dtype=dtype)
            self._pos_cache[key] = pos
            return pos
        else:
            key = ("image", height, width)
            if key in self._pos_cache:
                pe = self._pos_cache[key]
                return pe.to(device=device, dtype=dtype)
            grid_h = np.arange(height, dtype=np.float32)  # 非正方形图像分别构造行/列坐标
            grid_w = np.arange(width, dtype=np.float32)
            grid = np.meshgrid(grid_w, grid_h)
            grid = np.stack(grid, axis=0).reshape(2, 1, height, width)  # 两个坐标通道 [2,1,H,W]
            pos_np = get_2d_sincos_pos_embed_from_grid(self.hidden_size_output, grid)  # [H*W,pixel_C]
            pos = torch.from_numpy(pos_np).to(device=device, dtype=dtype)
            self._pos_cache[key] = pos
            return pos

    def forward(self, inputs: torch.Tensor, img_height: int = None, img_width: int = None, patch_size: int = None):
        """将 [B,3,H,W] 图像整理为每个 patch 一段的像素序列。"""
        if inputs.dim() != 4:
            raise ValueError("PixelTokenEmbedder expects inputs of shape [B,C,H,W]")
        assert img_height is not None and img_width is not None and patch_size is not None
        B, C, H, W = inputs.shape  # inputs [B,C,H,W]
        assert H == img_height and W == img_width
        assert (H % patch_size == 0) and (W % patch_size == 0)
        Hs, Ws = H // patch_size, W // patch_size  # patch 网格高宽；N=Hs*Ws
        P2 = patch_size * patch_size  # 每个 patch 的像素数 P²
        x = inputs.permute(0, 2, 3, 1).contiguous()  # [B,C,H,W] → [B,H,W,C]，把 RGB 放到最后逐像素投影
        x = self.proj(x)  # 每个像素独立投影到 pixel_C；[B,H,W,pixel_C]
        if self.use_pixel_abs_pos:
            pos_full = self._fetch_pixel_pos_image(H, W, inputs.device, inputs.dtype)  # 固定绝对位置编码 [H*W,pixel_C]
            pos_full = pos_full.view(H, W, self.hidden_size_output)  # 恢复图像网格 [H,W,pixel_C]
            x = x + pos_full.unsqueeze(0)  # 给每个像素加入其绝对位置；[B,H,W,pixel_C]
        x = x.view(B, Hs, patch_size, Ws, patch_size, self.hidden_size_output)  # 拆出 patch 网格及 patch 内像素坐标
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B,Hs,Ws,P,P,pixel_C]，先枚举 patch 再枚举内部像素
        x = x.view(B * Hs * Ws, P2, self.hidden_size_output)  # [B*N,P²,pixel_C]，每个 patch 一段像素序列
        return x


class PiTBlock(nn.Module):
    """patch 内像素级 block：压缩做全局注意力，再展开恢复像素细节。

    输入 x 为 [B*N,P²,pixel_C]。每个 patch 的 P² 个像素先压成一个
    注意力 token，与全图其他 patch 做全局注意力，随后展开回 P² 个像素。
    patch 级语义 s_cond 同时生成每个像素的 adaLN 调制参数。
    """

    def __init__(
        self,
        pixel_hidden_size: int,
        patch_hidden_size: int,
        patch_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_hidden_size: Optional[int] = None,
        attn_num_heads: Optional[int] = None,
        rope_fn=None,
        adaln_post_modulation: bool = False,
    ):
        super().__init__()
        self.pixel_dim = int(pixel_hidden_size)  # 单个像素 token 的通道数
        self.context_dim = int(patch_hidden_size)  # patch 级语义条件 s_cond 的通道数
        self.patch_size = int(patch_size)
        self.attn_dim = int(attn_hidden_size) if attn_hidden_size is not None else self.context_dim
        self.num_heads = int(attn_num_heads) if attn_num_heads is not None else int(num_heads)
        assert (
            self.attn_dim % self.num_heads == 0
        ), "pixel attention hidden size must be divisible by pixel num_heads"
        p2 = self.patch_size * self.patch_size
        self.compress_to_attn = nn.Linear(p2 * self.pixel_dim, self.attn_dim, bias=True)  # 整个 patch 的像素信息压成一个注意力 token
        self.expand_from_attn = nn.Linear(self.attn_dim, p2 * self.pixel_dim, bias=True)  # 注意力结果重新展开到 patch 内所有像素
        self.norm1 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.attn = RotaryAttention(self.attn_dim, num_heads=self.num_heads, qkv_bias=False)  # 在 N 个 patch 表示之间做全局注意力
        self.norm2 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.mlp = MLP(self.pixel_dim, mlp_ratio=mlp_ratio, drop=0.0)  # 对每个像素 token 独立执行前馈网络
        self.adaln_post_modulation = bool(adaln_post_modulation)
        n_mod = 4 if self.adaln_post_modulation else 6
        self.adaLN_modulation = nn.Sequential(nn.Linear(self.context_dim, n_mod * self.pixel_dim * p2, bias=True))  # patch 条件 → patch 内每个像素的调制参数
        self._pos_cache = dict()
        self._rope_fn = rope_fn if rope_fn is not None else precompute_freqs_cis_2d

    def _fetch_pos(self, height: int, width: int, device):
        """生成 patch 网格的二维 RoPE，形状 [N,head_dim//2]。"""
        key = (height, width)
        if key in self._pos_cache:
            return self._pos_cache[key].to(device)
        pos = self._rope_fn(self.attn_dim // self.num_heads, height, width).to(device)
        self._pos_cache[key] = pos
        return pos

    def forward(self, x: torch.Tensor, s_cond: torch.Tensor, image_height: int, image_width: int, patch_size: int, mask=None) -> torch.Tensor:
        """用 patch 级语义调制像素序列，并通过压缩/展开交换全局信息。"""
        BL, P2, C = x.shape  # x [B*N,P²,pixel_C]；BL=B*N
        if C != self.pixel_dim:
            raise ValueError(f"PiTBlock expected pixel_dim={self.pixel_dim}, got {C}")
        assert (image_height % patch_size == 0) and (image_width % patch_size == 0)
        Hs, Ws = image_height // patch_size, image_width // patch_size
        L = Hs * Ws  # L=N，即每幅图的 patch 数
        B = BL // L
        n_mod = 4 if self.adaln_post_modulation else 6
        cond_params = self.adaLN_modulation(s_cond).view(BL, P2, n_mod * self.pixel_dim)  # [B*N,C] → [B*N,P²,n_mod*pixel_C]，每个像素都有条件
        if self.adaln_post_modulation:
            scale1, shift1, scale2, shift2 = torch.chunk(cond_params, 4, dim=-1)  # 各 [B*N,P²,pixel_C]，在残差输出端调制
            x_norm = self.norm1(x)  # [B*N,P²,pixel_C]
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(cond_params, 6, dim=-1)  # 各 [B*N,P²,pixel_C]
            x_norm = apply_adaln(self.norm1(x), shift_msa, scale_msa)  # 注意力前先做逐像素条件仿射
        x_flat = x_norm.view(BL, P2 * self.pixel_dim)  # 一个 patch 内所有像素展平为 [B*N,P²*pixel_C]
        x_comp = self.compress_to_attn(x_flat).view(B, L, self.attn_dim)  # 每个 patch 压成一个 token：[B,N,attn_dim]
        pos_comp = self._fetch_pos(Hs, Ws, x.device)  # patch 网格 RoPE [N,head_dim//2]
        attn_out = self.attn(x_comp, pos_comp, mask)  # 全图 patch 间交换信息，形状保持 [B,N,attn_dim]
        attn_flat = self.expand_from_attn(attn_out.view(B * L, self.attn_dim))  # 每个 patch 表示展开为 [B*N,P²*pixel_C]
        attn_exp = attn_flat.view(BL, P2, self.pixel_dim)  # 恢复 patch 内像素序列 [B*N,P²,pixel_C]
        if self.adaln_post_modulation:
            x = x + attn_exp * (1 + scale1) + shift1  # 在注意力残差输出上做逐像素 scale/shift
            mlp_out = self.mlp(self.norm2(x))  # 每个像素独立前馈，形状不变
            x = x + mlp_out * (1 + scale2) + shift2  # 在 MLP 残差输出上再次调制
        else:
            x = x + gate_msa * attn_exp  # gate 控制全局注意力残差强度，输出 [B*N,P²,pixel_C]
            mlp_out = self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))  # 先条件调制，再逐像素前馈
            x = x + gate_mlp * mlp_out  # gate 控制 MLP 残差，保持原形状
        return x  # [B*N,P²,pixel_C]


class PixDiT(nn.Module):
    """PixelDiT v1 主干：patch 级理解全图，pixel 级重建局部细节。

    hidden_size/C 是 patch 级语义宽度，pixel_hidden_size/pixel_C 是像素 token
    宽度；patch_depth 和 pixel_depth 分别控制两级 block 数。我们的 DINO 参数
    只改变 patch 级输入表示，不改变 PiT 解码、扩散目标或采样接口。
    """

    def __init__(
        self,
        in_channels=4,
        num_groups=12,
        hidden_size=1152,
        pixel_hidden_size=64,
        patch_depth=18,
        pixel_depth=4,
        patch_size=2,
        num_classes=1000,
        use_pixel_abs_pos=True,
        pit_adaln_post_modulation=False,
        dino_model_name="dinov3_vitb16",
        dino_weights=None,
        dino_lora_rank=64,
        dino_lora_alpha=64.0,
        dino_lora_dropout=0.0,
        dino_lora_target_modules="qkv,proj,fc1,fc2",
        dino_train_patch_embed=True,
        dino_train_final_norm=False,
        dino_generation_final_norm_affine=True,
        dino_aggregation="mls",
        dino_layers="default",
        dino_label_cond=False,
        dino_time_blend_transform="smoother",
        dino_time_blend_key=20.0,
        dino_time_blend_a=0.1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.num_groups = int(num_groups)
        self.patch_depth = int(patch_depth)
        self.pixel_depth = int(pixel_depth)
        self.patch_size = int(patch_size)
        self.pixel_hidden_size = int(pixel_hidden_size)
        self.num_classes = int(num_classes)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.pit_adaln_post_modulation = bool(pit_adaln_post_modulation)
        self.dino_time_blend_transform = dino_time_blend_transform
        self.dino_time_blend_key = float(dino_time_blend_key)
        self.dino_time_blend_a = float(dino_time_blend_a)
        if self.pixel_depth <= 0:
            raise ValueError("PixDiT expects pixel_depth > 0 to preserve the dual-level pipeline")

        # 两条输入支路：原 PixelDiT patchify，以及我们的 DINO 语义 patchify。
        self.pixel_embedder = PixelTokenEmbedder(self.in_channels, self.pixel_hidden_size, use_pixel_abs_pos=self.use_pixel_abs_pos)  # 原 pixel 级输入，重建 patch 内像素
        self.s_embedder = PatchTokenEmbedder(self.in_channels * self.patch_size ** 2, self.hidden_size, bias=True)  # 原生 patch 展平向量 → C 维语义 token
        self.dino_patch_embedder = DINOPatchEmbedder(
            hidden_size=self.hidden_size,
            num_heads=self.num_groups,
            patch_size=self.patch_size,
            model_name=dino_model_name,
            weights_path=dino_weights,
            lora_rank=dino_lora_rank,
            lora_alpha=dino_lora_alpha,
            lora_dropout=dino_lora_dropout,
            lora_target_modules=dino_lora_target_modules,
            train_patch_embed=dino_train_patch_embed,
            train_final_norm=dino_train_final_norm,
            final_norm_affine=dino_generation_final_norm_affine,
            aggregation=dino_aggregation,
            layers=dino_layers,
            label_cond=dino_label_cond,
            num_classes=self.num_classes,
        )  # DINO+LoRA/聚合 → PgT(t) → C 维语义 token
        self.patch_pre_blend_norm = RMSNorm(self.hidden_size, eps=1e-6)  # 与 DINO 路融合前，先统一原生 patch token 的尺度
        self.t_embedder = TimestepConditioner(self.hidden_size)  # 连续扩散时间 → 全局条件 C
        self.y_embedder = ClassEmbedder(self.num_classes + 1, self.hidden_size)  # 类别及 CFG null 类1000 → 全局条件 C

        self.final_layer = FinalLayer(self.pixel_hidden_size, self.out_channels)  # 每个像素 token → 该像素的 RGB，而非整张 patch 一次输出
        self.patch_blocks = nn.ModuleList(
            [AugmentedDiTBlock(self.hidden_size, self.num_groups) for _ in range(self.patch_depth)]
        )  # 逐级建模全图 patch 间的语义关系
        self.pixel_blocks = nn.ModuleList(
            [
                PiTBlock(
                    self.pixel_hidden_size,
                    self.hidden_size,
                    patch_size=self.patch_size,
                    num_heads=self.num_groups,
                    mlp_ratio=4.0,
                    adaln_post_modulation=self.pit_adaln_post_modulation,
                )
                for _ in range(self.pixel_depth)
            ]
        )  # 在 patch 语义条件引导下恢复局部像素细节
        self.initialize_weights()
        self.precompute_pos = dict()  # 按 patch 网格缓存固定 RoPE，训练和采样共用

    def fetch_pos(self, height, width, device):
        """返回 patch 网格的二维旋转位置编码 [N,head_dim//2]。"""
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def initialize_weights(self):
        """保留 PixelDiT 初始化：patch 投影 Xavier，输出和 adaLN 残差门控置零。"""
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))  # 原生 patchify 投影保持 Xavier 初始化
        nn.init.constant_(self.s_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)  # 主 PixelDiT 的类别嵌入；与 DINO 路独立
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.final_layer.linear.weight)  # 初始 RGB 预测为0，稳定扩散训练起点
        nn.init.zeros_(self.final_layer.linear.bias)
        for block in self.patch_blocks:
            nn.init.zeros_(block.adaLN_modulation[0].weight)  # patch block 初始残差门控为0
            nn.init.zeros_(block.adaLN_modulation[0].bias)
        for block in self.pixel_blocks:
            nn.init.zeros_(block.adaLN_modulation[0].weight)  # pixel block 初始残差门控为0
            nn.init.zeros_(block.adaLN_modulation[0].bias)

    def forward(self, x, t, y, s=None, mask=None):
        """从带噪图像预测 velocity 所需的网络输出。

        x [B,3,H,W] 是 x_t；t [B] 为干净图像权重（0=噪声，1=干净图）；
        y [B] 是已执行 label drop 的类别，null 类为1000。若外部传入 s，
        则把它当作已计算的 patch 条件，跳过下面的 patchify 和 patch blocks。
        """
        B, _, H, W = x.shape  # x [B,3,H,W]
        pos = self.fetch_pos(H // self.patch_size, W // self.patch_size, x.device)  # patch 网格 RoPE [N,head_dim//2]，供全局语义注意力定位
        x_patches = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)  # 非重叠切块并展开像素通道：[B,N,3P²]
        t_emb = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)  # 原始扩散时间映射为全局条件：[B] → [B,1,C]
        y_emb = self.y_embedder(y).view(B, 1, self.hidden_size)  # 类别/null 标签映射为全局条件：[B] → [B,1,C]
        c = nn.functional.silu(t_emb + y_emb)  # 原 PixelDiT 主条件，供所有 patch block 的 adaLN 使用：[B,1,C]
        if s is None:
            # 两路都产生 [B,N,C]；f(t) 是原生 patchify 权重，t=1 时只用原生分支。
            pixel_tokens = self.patch_pre_blend_norm(self.s_embedder(x_patches))  # 原生 patch 向量投影并统一尺度：[B,N,3P²] → [B,N,C]
            dino_tokens = self.dino_patch_embedder(x, t, y)  # DINO 提取语义，PgT 注入原始 t；y 可选注入 DINO backbone：[B,N,C]
            blend_t = apply_blend_time(
                t.reshape(-1),  # 融合只读取原始 t：[B]，不改变扩散时间条件
                transform=self.dino_time_blend_transform,
                key=self.dino_time_blend_key,
                a=self.dino_time_blend_a,
            )  # 原生 patchify 的时间权重 f(t)：[B]
            pixel_weight = blend_t.to(device=pixel_tokens.device, dtype=pixel_tokens.dtype).view(B, 1, 1)  # [B] → [B,1,1]，广播到 N 和 C
            s = pixel_weight * pixel_tokens + (1.0 - pixel_weight) * dino_tokens  # f(t)*原生 + (1-f(t))*DINO；[B,N,C]
            for block in self.patch_blocks:
                s = block(s, c, pos, mask)  # 全局 patch 注意力逐层提炼语义；s [B,N,C]，c [B,1,C]
            s = nn.functional.silu(t_emb + s)  # 按上游设计再次加入时间条件，供 pixel block 使用：[B,N,C]
        batch_size, length, _ = s.shape  # s [B,N,C]，N 是每幅图的 patch 数
        s_cond = s.view(batch_size * length, self.hidden_size)  # 每个 patch 一份语义条件：[B,N,C] → [B*N,C]
        x_pixels = self.pixel_embedder(x, img_height=H, img_width=W, patch_size=self.patch_size)  # 原带噪图像的像素序列：[B*N,P²,pixel_C]
        for blk in self.pixel_blocks:
            x_pixels = blk(x_pixels, s_cond, H, W, self.patch_size, mask)  # 用 patch 语义引导 patch 内像素重建，形状不变
        x_pixels = self.final_layer(x_pixels)  # 每个像素 token 预测 RGB：[B*N,P²,pixel_C] → [B*N,P²,3]
        C_out = self.out_channels
        P2 = self.patch_size * self.patch_size
        x_pixels = x_pixels.view(B, length, P2, C_out).permute(0, 3, 2, 1).contiguous()  # [B,N,P²,3] → [B,3,P²,N]，为 fold 整理通道
        x_pixels = x_pixels.view(B, C_out * P2, length)  # [B,3,P²,N] → [B,3P²,N]
        x_img = torch.nn.functional.fold(x_pixels, (H, W), kernel_size=self.patch_size, stride=self.patch_size)  # 将各 patch 的像素拼回 [B,3,H,W]
        return x_img  # 网络预测的 velocity 张量，形状与带噪输入一致：[B,3,H,W]
