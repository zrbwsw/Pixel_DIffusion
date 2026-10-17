"""DINOv3-LoRA + JiT + DDTHead class-to-image 核心模型。

这个文件把四个模块接成一条可训练的数据流：

1. 带噪图像 ``z`` 进入 DINOv3-B/16-LoRA student；
2. BaseT 路径由 student MLS token 替换 JiT patch token；residual 路径
   保留 JiT 原生 PatchEmbed，并叠加由 ``t`` 门控的 DINO representation residual；
   time_blend 路径以 ``1-t``/``t`` 融合原生 token 与 timestep-projector
   输出；从 JiT 指定 block 的空间输出经独立 projector 对齐 frozen teacher；
3. JiT-B 的空间输出交给 DDTHead，或通过原生 final_layer 直接预测像素；
4. 输出 ``x_pred``，再换算成 rectified-flow velocity 计算去噪损失。

J18-new B-scale 的固定形状如下。``B`` 是每个 DDP rank 的本地 batch；有效全局
batch 256 在两卡 H100 上对应 ``B=128``，在四卡上通常对应 ``B=64``。

* image / noise / z / x_pred: ``[B, 3, 256, 256]``
* DINOv3-B patch grid: ``16 x 16``，因此 ``N=256``
* DINOv3-B raw-last、MLS、teacher token: ``[B, 256, 768]``
* JiT-B 空间 token: ``[B, 256, 768]``；插入 32 个类别 token 后为
  ``[B, 288, 768]``
* DDTHead condition: ``[B, 256, 768] -> [B, 256, 2048]``
* DDTHead patch prediction: ``[B, 256, 16*16*3=768]``

当前 ``Ours-J18-new-drop0.1-l4-B`` 配置使用一基 ``jit_alignment_depth=4``：
捕获 JiT block index 3 的输出，经 ``3x3 ConvProj`` 对齐 teacher。这个对齐
projector 不参与 JiT 输入融合；带噪 student 的输入可使用固定 ImageNet 标准化
或与扩散时间相关的解析矩标准化，clean teacher 始终使用固定标准化。
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_jit import JiT_models
from pixeldit2_projector import TimestepProjection
from projectors import ProjectionLayer
from util.model_util import RMSNorm

from dino_encoder import build_encoder
from jit_ddt_head import JiTDDTHead


class DINOResidualAdapter(nn.Module):
    """Map DINO patch features into a JiT-compatible residual stream.

    The final projection is zero-initialized, so a residual experiment starts
    exactly from the native JiT patch-token path instead of abruptly replacing
    its input distribution with pretrained-representation features.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(tokens))))


class DINOJiTDenoiser(nn.Module):
    """可切换 image-token source 和 DDT/native 输出头的 JiT 像素去噪器。

    frozen teacher 被有意放在普通 Python list 中，而不是注册为 ``nn.Module``
    子模块。这样 DDP、优化器、EMA 和 checkpoint 只处理可训练 student/JiT/head，
    teacher 仅在需要 target 时手动搬到设备。
    """

    def __init__(self, args):
        super().__init__()
        # J18-new 使用 JiT-B/16：12 blocks、hidden=768、12 heads、patch=16。
        # 旧 replace 路径绕过并冻结原生 x_embedder；其他路径保留并训练它。
        self.dino_jit_input_mode = str(
            getattr(args, "dino_jit_input_mode", "replace")
        ).lower()
        self.dino_residual_gate = str(
            getattr(args, "dino_residual_gate", "none")
        ).lower()
        self.dino_residual_hidden_dim = int(
            getattr(args, "dino_residual_hidden_dim", 0) or 0
        )
        self.dino_residual_drop_on_label_drop = bool(
            getattr(args, "dino_residual_drop_on_label_drop", False)
        )
        self.dino_input_projector_type = str(
            getattr(args, "dino_input_projector", "none")
        ).lower()
        self.dino_grounding_drop_on_label_drop = bool(
            getattr(args, "dino_grounding_drop_on_label_drop", False)
        )
        self.dino_jit_pre_blend_rmsnorm = bool(
            getattr(args, "dino_jit_pre_blend_rmsnorm", False)
        )
        self.dino_preprocess_input = bool(getattr(args, "dino_preprocess_input", True))
        self.dino_input_normalization = str(
            getattr(args, "dino_input_normalization", "standard")
        ).lower()
        if self.dino_input_normalization not in {"standard", "t_moment"}:
            raise ValueError(
                "dino_input_normalization must be standard or t_moment."
            )
        if self.dino_input_normalization == "t_moment" and not self.dino_preprocess_input:
            raise ValueError(
                "t_moment normalization requires dino_preprocess_input=true."
            )
        valid_input_modes = {"native", "replace", "gated_residual", "time_blend"}
        if self.dino_jit_input_mode not in valid_input_modes:
            raise ValueError(
                f"Unknown dino_jit_input_mode={self.dino_jit_input_mode}. "
                f"Expected one of {sorted(valid_input_modes)}"
            )
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            z_dims=[],
        )
        if self.dino_jit_input_mode == "replace":
            for param in self.net.x_embedder.parameters():
                param.requires_grad_(False)

        # 这些参数共同定义 rectified-flow 路径。这里的 t 是“干净图像权重”：
        # z_t = t*x + (1-t)*noise，所以 t=0 是纯噪声、t=1 是干净图像；采样方向
        # 也是 0 -> 1。该约定与把 t 定义为噪声权重的代码正好相反。
        self.img_size = args.img_size
        self.num_classes = args.class_num
        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

        # Alignment ConvProj is separate from the optional JiT-input projector.
        self.dino_aggregation = str(getattr(args, "dino_aggregation", "none")).lower()
        self.dino_layers = getattr(args, "dino_layers", None)
        self.jit_add_pos_embed = bool(getattr(args, "jit_add_pos_embed_to_dino_tokens", True))
        self.dino_zscore_alpha = float(args.dino_zscore_alpha)
        self.jit_alignment_depth = int(args.jit_alignment_depth)
        if self.dino_input_projector_type not in {"none", "pixeldit2"}:
            raise ValueError("dino_input_projector must be none or pixeldit2.")
        if self.dino_jit_input_mode != "time_blend" and (
            self.dino_input_projector_type != "none"
            or self.dino_grounding_drop_on_label_drop
        ):
            raise ValueError("DINO input projector and grounding dropout require time_blend.")
        if self.dino_jit_input_mode == "gated_residual":
            if self.dino_residual_gate not in {"clean_linear", "constant_one"}:
                raise ValueError(
                    "gated_residual requires dino_residual_gate=clean_linear "
                    "(g(t)=t) or constant_one (g(t)=1)."
                )
            if self.dino_residual_hidden_dim <= 0:
                raise ValueError(
                    "dino_residual_hidden_dim must be positive for gated_residual."
                )
            if not self.jit_add_pos_embed:
                raise ValueError(
                    "gated_residual requires jit_add_pos_embed_to_dino_tokens=true "
                    "so the fused native JiT spatial stream receives its fixed APE."
                )
        else:
            if (
                self.dino_residual_gate != "none"
                or self.dino_residual_hidden_dim != 0
                or self.dino_residual_drop_on_label_drop
            ):
                raise ValueError(
                    "Residual gate/hidden settings are unused when "
                    "dino_jit_input_mode is not gated_residual; use gate=none, hidden_dim=0, "
                    "and drop_on_label_drop=false."
                )
            if self.dino_jit_input_mode == "time_blend" and not self.jit_add_pos_embed:
                raise ValueError("time_blend requires jit_add_pos_embed_to_dino_tokens=true.")
            if self.dino_jit_input_mode == "native" and not self.jit_add_pos_embed:
                raise ValueError("native JiT+iREPA requires jit_add_pos_embed_to_dino_tokens=true.")
        layers_spec = str(self.dino_layers).strip().lower() if self.dino_layers is not None else ""
        if self.dino_aggregation == "none" and layers_spec not in {"", "none"}:
            raise ValueError("dino_layers is unused when dino_aggregation=none.")
        if self.dino_aggregation != "none" and layers_spec in {"", "none"}:
            raise ValueError("dino_layers is required for student aggregation.")
        if not 1 <= self.jit_alignment_depth <= len(self.net.blocks):
            raise ValueError(
                f"jit_alignment_depth={self.jit_alignment_depth} is invalid for "
                f"{len(self.net.blocks)} JiT blocks."
            )
        if self.dino_zscore_alpha < 0:
            raise ValueError("dino_zscore_alpha must be non-negative.")

        teacher = build_encoder(args, role="teacher")
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        self._teacher_box = [teacher]
        self._teacher_device: Optional[torch.device] = None

        # The native iREPA ablation has no noisy DINO student at all. DINOv3 is
        # used only as the frozen clean-image teacher for block-4 alignment.
        self.dino_student: Optional[nn.Module] = None
        if self.dino_jit_input_mode != "native":
            self.dino_student = build_encoder(args, role="student")
        dino_feature_dim = int(teacher.embed_dim)
        self.dino_student_projector = ProjectionLayer(
            "conv",
            hidden_size=self.net.hidden_size,
            z_dim=dino_feature_dim,
            proj_kwargs_kernel_size=int(args.dino_projector_kernel_size),
        )
        self.dino_residual_adapter: Optional[nn.Module] = None
        if self.dino_jit_input_mode == "gated_residual":
            self.dino_residual_adapter = DINOResidualAdapter(
                input_dim=dino_feature_dim,
                hidden_dim=self.dino_residual_hidden_dim,
                output_dim=self.net.hidden_size,
            )
        self.dino_input_projector: Optional[nn.Module] = None
        if self.dino_input_projector_type == "pixeldit2":
            self.dino_input_projector = TimestepProjection(
                in_dim=dino_feature_dim,
                out_dim=self.net.hidden_size,
                hidden_size=self.net.hidden_size,
                num_heads=self.net.num_heads,
            )
        # Stateless, affine-free normalization over each token's hidden dimension.
        # Reusing the module shares no learned gains or input statistics.
        self.dino_jit_pre_blend_norm = (
            RMSNorm(self.net.hidden_size, eps=1e-6, elementwise_affine=False)
            if self.dino_jit_pre_blend_rmsnorm
            else nn.Identity()
        )
        if self.dino_student is not None and int(self.dino_student.embed_dim) != self.net.hidden_size:
            raise ValueError("DINO student token width must match JiT hidden size.")
        if int(teacher.embed_dim) != dino_feature_dim:
            raise ValueError("DINO teacher and alignment projector widths must match.")

        self.use_ddt_head = bool(getattr(args, "use_ddt_head", True))
        self.ddt_head: Optional[nn.Module] = None
        if self.use_ddt_head:
            self.ddt_head_dim = int(getattr(args, "ddt_head_dim", 0)) or self.net.hidden_size
            self.ddt_head_num_heads = int(getattr(args, "ddt_head_num_heads", 0))
            if self.ddt_head_num_heads <= 0:
                self.ddt_head_num_heads = (
                    self.net.num_heads if self.ddt_head_dim % self.net.num_heads == 0 else 16
                )
            if self.ddt_head_dim % self.ddt_head_num_heads != 0:
                raise ValueError("ddt_head_dim must be divisible by ddt_head_num_heads.")
            self.ddt_head = JiTDDTHead(
                hidden_size=self.ddt_head_dim,
                patch_size=self.net.patch_size,
                in_channels=3,
                bottleneck_dim=int(getattr(args, "ddt_head_bottleneck_dim", 128)),
                input_size=self.img_size,
                depth=int(getattr(args, "ddt_head_depth", 2)),
                num_heads=self.ddt_head_num_heads,
                mlp_ratio=4.0,
                cond_hidden_size=self.net.hidden_size,
                time_reinject=True,
                embed_type="bottleneck",
            )
        # Preserve legacy DDT checkpoints; train the native head only when used.
        for param in self.net.final_layer.parameters():
            param.requires_grad_(not self.use_ddt_head)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("dino_mean", mean, persistent=False)
        self.register_buffer("dino_std", std, persistent=False)
        self.register_buffer("dino_mean_pm1", 2.0 * mean - 1.0, persistent=False)
        self.register_buffer("dino_std_pm1", 2.0 * std, persistent=False)

    @property
    def teacher(self) -> nn.Module:
        # Teacher 与 student 都由 DINOv3PatchTokenEncoder 构建，并共享
        # forward_patch_tokens 接口；teacher 本身不注册进 checkpoint。
        if not self._teacher_box:
            raise RuntimeError("This DINOJiTDenoiser was constructed without a teacher.")
        return self._teacher_box[0]

    def _ensure_teacher_device(self, device: torch.device) -> None:
        if self._teacher_device != device:
            self.teacher.to(device)
            self._teacher_device = device
        self.teacher.eval()

    def _preprocess_for_dino(self, x: torch.Tensor) -> torch.Tensor:
        """把扩散像素域 ``[-1,1]`` 转成 DINO ImageNet 标准化输入。

        输入/输出形状均为 ``[B,3,256,256]``。``x.float()`` 保证归一化统计在
        float32 中完成；外层 BF16 autocast 仍会让 DINO 的适合算子以 BF16 执行。
        """
        x01 = (x.float() + 1.0) * 0.5
        mean = self.dino_mean.to(device=x.device, dtype=x01.dtype)
        std = self.dino_std.to(device=x.device, dtype=x01.dtype)
        return (x01 - mean) / std

    def drop_labels_with_mask(
        self, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Drop labels and return the exact per-sample mask used for dropout."""
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        dropped = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return dropped, drop

    def drop_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """按训练 dropout 概率把类别替换成 null id=1000，为 CFG 学无条件分支。"""
        dropped, _ = self.drop_labels_with_mask(labels)
        return dropped

    def sample_t(self, n: int, device=None) -> torch.Tensor:
        """从 logit-normal 采样每个样本的干净权重，返回 ``[B]``。"""
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def _preprocess_noisy_student_for_dino(
        self, noisy_z: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Normalize ``z_t`` with the analytic moments of the diffusion path.

        In the ``[-1, 1]`` domain, the clean ImageNet image moments are
        ``mu_x=2*mu_01-1`` and ``sigma_x=2*sigma_01``. For
        ``z_t=t*x+(1-t)*epsilon`` with independent ``x`` and ``epsilon``,
        use ``mu_t=t*mu_x`` and
        ``sigma_t^2=t^2*sigma_x^2+(1-t)^2*noise_scale^2``.
        """
        if noisy_z.ndim != 4 or noisy_z.shape[1] != 3:
            raise ValueError(
                "noisy_z must have shape [B,3,H,W], got "
                f"{tuple(noisy_z.shape)}."
            )
        batch_size = noisy_z.shape[0]
        if timesteps.numel() != batch_size:
            raise ValueError(
                "timesteps must contain exactly one scalar per noisy image; "
                f"got shape {tuple(timesteps.shape)} for batch {batch_size}."
            )
        z_float = noisy_z.float()
        t = timesteps.to(device=noisy_z.device, dtype=z_float.dtype).reshape(
            batch_size, 1, 1, 1
        )
        mean_x = self.dino_mean_pm1.to(device=noisy_z.device, dtype=z_float.dtype)
        std_x = self.dino_std_pm1.to(device=noisy_z.device, dtype=z_float.dtype)
        mean_t = t * mean_x
        variance_t = (
            t.square() * std_x.square()
            + (1.0 - t).square() * float(self.noise_scale) ** 2
        )
        std_t = variance_t.sqrt().clamp_min(1e-6)
        return (z_float - mean_t) / std_t

    def _encode_noisy_student(
        self, noisy_z: torch.Tensor, timesteps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.dino_student is None:
            raise RuntimeError("The native iREPA mode has no noisy DINO student.")
        input_normalization = getattr(self, "dino_input_normalization", "standard")
        if input_normalization == "t_moment":
            if timesteps is None:
                raise ValueError("t_moment normalization requires diffusion timesteps.")
            student_input = self._preprocess_noisy_student_for_dino(noisy_z, timesteps)
        elif self.dino_preprocess_input:
            student_input = self._preprocess_for_dino(noisy_z)
        else:
            student_input = noisy_z
        return self.dino_student(student_input)

    def _encode_dino_pair(
        self,
        clean_x: torch.Tensor,
        noisy_z: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        student_tokens = (
            None
            if self.dino_jit_input_mode == "native"
            else self._encode_noisy_student(noisy_z, timesteps)
        )
        return student_tokens, self._encode_teacher_target(clean_x)

    def _encode_teacher_target(self, clean_x: torch.Tensor) -> torch.Tensor:
        self._ensure_teacher_device(clean_x.device)
        with torch.no_grad():
            return self.teacher(self._preprocess_for_dino(clean_x)).detach()

    def _project_student_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.dino_student_projector(tokens)

    def _compose_jit_input_tokens(
        self,
        noisy_image: torch.Tensor,
        dino_tokens: Optional[torch.Tensor],
        timesteps: torch.Tensor,
        label_drop_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        """Build the spatial JiT input before its fixed positional embedding.

        ``native`` uses only ``net.x_embedder(noisy_image)`` and is the pure
        JiT+iREPA ablation; its DINO teacher is used only by the alignment loss.

        ``replace`` preserves all historical experiments. ``gated_residual``
        computes, for B-scale ImageNet-256,

        ``native [B,256,768] + g(t) [B,1,1] * residual [B,256,768]``.

        ``time_blend`` instead computes ``(1-t)*native + t*projected_dino``.

        If ``dino_residual_drop_on_label_drop`` is enabled, the residual gate
        is zeroed for dropped labels. In time_blend, optional grounding dropout
        instead restores the full native JiT tokens for those labels. Training
        passes the actual drop mask; CFG sampling derives it from the null class.

        The alignment-only ConvProj is deliberately absent from this method.
        JiT's fixed APE is added exactly once later in
        :meth:`_jit_forward_from_tokens`.
        """
        if self.dino_jit_input_mode == "native":
            native_tokens = self.net.x_embedder(noisy_image)
            return native_tokens, {"native_tokens": native_tokens}

        if dino_tokens is None:
            raise RuntimeError(
                f"dino_jit_input_mode={self.dino_jit_input_mode} requires noisy DINO tokens."
            )

        if self.dino_jit_input_mode == "replace":
            return dino_tokens, None

        if self.dino_jit_input_mode == "time_blend":
            native_tokens = self.net.x_embedder(noisy_image)
            projected_tokens = dino_tokens
            if self.dino_input_projector is not None:
                # PixelDiT2 embeds clean-image t independently of JiT and uses
                # its own 2D RoPE for the spatial patch grid.
                grid_hw = tuple(size // self.net.patch_size for size in noisy_image.shape[-2:])
                projected_tokens = self.dino_input_projector(
                    dino_tokens.to(dtype=self.dino_input_projector.input_proj.weight.dtype),
                    timesteps.flatten(), grid_hw,
                )
            if native_tokens.shape != projected_tokens.shape:
                raise ValueError(
                    "Native JiT and projected DINO token shapes must match, got "
                    f"{tuple(native_tokens.shape)} and {tuple(projected_tokens.shape)}."
                )
            native_tokens = self.dino_jit_pre_blend_norm(native_tokens)
            projected_tokens = self.dino_jit_pre_blend_norm(projected_tokens)
            t = timesteps.flatten().to(device=native_tokens.device, dtype=native_tokens.dtype)
            if t.numel() != native_tokens.shape[0]:
                raise ValueError("Expected one blend timestep per sample.")
            t = t.view(-1, 1, 1)
            jit_weighted = (1.0 - t) * native_tokens
            dino_weighted = t * projected_tokens
            grounding_drop_mask = torch.zeros(
                t.shape[0], device=t.device, dtype=torch.bool,
            )
            if self.dino_grounding_drop_on_label_drop and label_drop_mask is not None:
                grounding_drop_mask = label_drop_mask.flatten().to(
                    device=t.device, dtype=torch.bool,
                )
                if grounding_drop_mask.numel() != t.shape[0]:
                    raise ValueError("Expected one label-drop mask per sample.")
                mask = grounding_drop_mask.view(-1, 1, 1)
                jit_weighted = torch.where(mask, native_tokens, jit_weighted)
                dino_weighted = dino_weighted.masked_fill(mask, 0)
            fused_tokens = jit_weighted + dino_weighted
            return fused_tokens, {
                "native_tokens": native_tokens,
                "dino_projected_tokens": projected_tokens,
                "jit_weighted_tokens": jit_weighted,
                "dino_weighted_tokens": dino_weighted,
                "grounding_drop_mask": grounding_drop_mask,
            }

        if self.dino_jit_input_mode != "gated_residual":
            raise AssertionError(
                f"Unhandled dino_jit_input_mode={self.dino_jit_input_mode}"
            )
        if self.dino_residual_adapter is None:
            raise RuntimeError("gated_residual was configured without an adapter.")

        native_tokens = self.net.x_embedder(noisy_image)
        residual_tokens = self.dino_residual_adapter(dino_tokens)
        if native_tokens.shape != residual_tokens.shape:
            raise ValueError(
                "Native JiT and DINO residual token shapes must match, got "
                f"{tuple(native_tokens.shape)} and {tuple(residual_tokens.shape)}."
            )

        gate = timesteps.flatten()
        if gate.numel() != native_tokens.shape[0]:
            raise ValueError(
                f"Expected one residual gate per sample, got {gate.numel()} for "
                f"batch size {native_tokens.shape[0]}."
            )
        gate = gate.to(device=residual_tokens.device, dtype=residual_tokens.dtype)
        if self.dino_residual_gate == "clean_linear":
            # t is the clean-image weight: t=0 is pure noise and t=1 is clean.
            gate = gate
        elif self.dino_residual_gate == "constant_one":
            gate = torch.ones_like(gate)
        else:
            raise AssertionError(
                f"Unhandled dino_residual_gate={self.dino_residual_gate}"
            )

        drop_residual_on_label_drop = bool(
            getattr(self, "dino_residual_drop_on_label_drop", False)
        )
        if drop_residual_on_label_drop and label_drop_mask is not None:
            label_drop_mask = label_drop_mask.flatten().to(
                device=gate.device, dtype=torch.bool
            )
            if label_drop_mask.numel() != gate.numel():
                raise ValueError(
                    "Expected one label-drop mask value per sample, got "
                    f"{label_drop_mask.numel()} for batch size {gate.numel()}."
                )
            gate = gate.masked_fill(label_drop_mask, 0.0)
        elif label_drop_mask is None:
            label_drop_mask = torch.zeros(
                gate.shape[0], device=gate.device, dtype=torch.bool
            )
        gate = gate.view(-1, 1, 1)
        gated_residual = gate * residual_tokens
        fused_tokens = native_tokens + gated_residual
        return fused_tokens, {
            "native_tokens": native_tokens,
            "residual_tokens": residual_tokens,
            "gated_residual": gated_residual,
            "gate": gate,
            "label_drop_mask": label_drop_mask.view(-1, 1, 1),
        }

    def _spatial_normalize_target(self, target_tokens: torch.Tensor) -> torch.Tensor:
        target_tokens = target_tokens.float()
        mean = target_tokens.mean(dim=1, keepdim=True)
        std = target_tokens.std(dim=1, keepdim=True)
        return (target_tokens - self.dino_zscore_alpha * mean) / (std + 1e-6)

    def _alignment_loss(
        self,
        aligned_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aligned_tokens = aligned_tokens.float()
        target_tokens = target_tokens.float()
        target_loss_tokens = self._spatial_normalize_target(target_tokens)

        raw_mse = (aligned_tokens - target_tokens).square().mean()
        raw_cos_per_token = F.cosine_similarity(aligned_tokens, target_tokens, dim=-1)
        loss_cos_per_token = F.cosine_similarity(
            aligned_tokens, target_loss_tokens, dim=-1
        )
        raw_cos = raw_cos_per_token.mean()
        loss_cos = loss_cos_per_token.mean()
        loss = -loss_cos

        metrics = {
            "raw_mse": raw_mse,
            "raw_cosine": raw_cos,
            "raw_cosine_std": raw_cos_per_token.std(unbiased=False),
            "loss_cosine": loss_cos,
            "loss_cosine_std": loss_cos_per_token.std(unbiased=False),
        }
        return loss, metrics

    def _jit_forward_from_tokens(
        self,
        patch_tokens: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
        noisy_image: Optional[torch.Tensor] = None,
        alignment_depth: Optional[int] = None,
    ):
        """从图像 patch token 执行 JiT trunk，用所选输出头还原图像。

        J18-new 主路径的形状变化：

        * ``patch_tokens``: ``[B,256,768]``；来自 DINO 或 JiT 原生 x_embedder；
        * ``t_emb/y_emb/c``: 均为 ``[B,768]``；c 用于每个 JiT block 的全局 AdaLN；
        * block 0--3: ``[B,256,768]``；block 4 前插入 32 个 class token；
        * block 4--11: ``[B,288,768]``；结束后移除前 32 个 token；
        * 空间输出: ``[B,256,768]``，送往 DDTHead 或原生 final_layer；
        * 返回 ``x_pred``: ``[B,3,256,256]``。

        ``alignment_depth`` 使用与原版 JiT+iREPA 相同的一基语义：值为 4 时
        捕获第 4 个 block（实际索引 3）的输出。
        """
        net = self.net
        t_emb = net.t_embedder(timesteps)
        y_emb = net.y_embedder(labels)
        c = t_emb + y_emb

        x = patch_tokens
        if self.jit_add_pos_embed:
            x = x + net.pos_embed.to(device=x.device, dtype=x.dtype)

        class_tokens_inserted = False
        alignment_tokens: Optional[torch.Tensor] = None

        for i, block in enumerate(net.blocks):
            if net.in_context_len > 0 and i == net.in_context_start:
                # JiT-B 在 block index 4 前插入 32 个相同类别 embedding 加可学习位置：
                # [B,256,768] -> [B,32+256,768]。DINO 自身没有类别输入，类别信息
                # 从这里和全局 AdaLN condition c 两处进入生成网络。
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, net.in_context_len, 1)
                in_context_tokens = in_context_tokens + net.in_context_posemb.to(device=x.device, dtype=x.dtype)
                x = torch.cat([in_context_tokens.to(dtype=x.dtype), x], dim=1)
                class_tokens_inserted = True
            rope = net.feat_rope if i < net.in_context_start else net.feat_rope_incontext
            x = block(x, c, rope)
            # 与原版 JiT+iREPA 一致：alignment_depth 从 1 开始计数。
            # JiT-B 取 alignment_depth=4 时，此处 i=3，class tokens 尚未插入。
            if alignment_depth is not None and (i + 1) == alignment_depth:
                alignment_tokens = x[:, net.in_context_len:] if class_tokens_inserted else x

        if class_tokens_inserted:
            x = x[:, net.in_context_len:]
        if self.use_ddt_head:
            patches = self.ddt_head(noisy_image, x, t_emb=t_emb)
        else:
            patches = net.final_layer(x, c)
        output = net.unpatchify(patches, net.patch_size)

        if alignment_depth is None:
            return output
        if alignment_tokens is None:
            raise RuntimeError(
                f"JiT alignment depth {alignment_depth} was not reached."
            )
        return output, alignment_tokens

    def _predict_clean_from_noisy(
        self,
        noisy_z: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """采样时执行 noisy image-token source -> JiT -> head，不运行 teacher。"""
        student_tokens = (
            None
            if self.dino_jit_input_mode == "native"
            else self._encode_noisy_student(noisy_z, timesteps)
        )
        label_drop_mask = (
            labels == self.num_classes
            if (self.dino_residual_drop_on_label_drop or self.dino_grounding_drop_on_label_drop)
            else None
        )
        jit_tokens, _ = self._compose_jit_input_tokens(
            noisy_z, student_tokens, timesteps, label_drop_mask=label_drop_mask,
        )
        x_pred = self._jit_forward_from_tokens(jit_tokens, timesteps.flatten(), labels, noisy_image=noisy_z)
        return x_pred, jit_tokens

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        """执行一个 J18-new 训练 step 并返回标量总损失与诊断指标。

        Args:
            x: ``float32 [B,3,256,256]``，范围 ``[-1,1]``。
            labels: ``int64 [B]``，ImageNet 类别 0--999。
        """
        # 训练时随机把 label 置为 1000；评估时保留调用方给定标签。
        if self.training:
            labels_dropped, label_drop_mask = self.drop_labels_with_mask(labels)
        else:
            labels_dropped = labels
            label_drop_mask = (
                labels == self.num_classes
                if (self.dino_residual_drop_on_label_drop or self.dino_grounding_drop_on_label_drop)
                else None
            )

        # t: [B,1,1,1]，在每个样本的 C/H/W 维广播。
        # x 是数据，e 是高斯噪声，z 沿直线路径从 t=0 的 e 走到 t=1 的 x。
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale
        z = t * x + (1.0 - t) * e
        # 理论上 v=(x-z)/(1-t)=x-e；clamp 只防止 t 接近 1 时数值除零。
        v = (x - z) / (1.0 - t).clamp_min(self.t_eps)

        residual_diagnostics: Optional[dict[str, torch.Tensor]] = None
        student_tokens, target_tokens = self._encode_dino_pair(x, z, t)
        jit_tokens, residual_diagnostics = self._compose_jit_input_tokens(
            z, student_tokens, t, label_drop_mask=label_drop_mask,
        )
        x_pred, projector_input_tokens = self._jit_forward_from_tokens(
            jit_tokens, t.flatten(), labels_dropped, noisy_image=z,
            alignment_depth=self.jit_alignment_depth,
        )
        align_tokens = self._project_student_tokens(projector_input_tokens)
        projector_output_tokens = align_tokens
        # 模型显式预测 clean x，再用同一解析关系转换为 velocity 监督。
        v_pred = (x_pred - z) / (1.0 - t).clamp_min(self.t_eps)

        # 每张图先在 C/H/W 上平均，再在本地 batch 上平均；DDP backward 会对
        # 各 rank 梯度求平均，因此最终对应有效全局 batch 256。
        denoise_per_sample = ((v - v_pred) ** 2).mean(dim=(1, 2, 3))
        denoise_loss = denoise_per_sample.mean()
        alignment_loss, alignment_metrics = self._alignment_loss(align_tokens, target_tokens)
        total_loss = denoise_loss + 0.5 * alignment_loss

        with torch.no_grad():
            projector_output_tokens_f = projector_output_tokens.detach().float()
            jit_tokens_f = jit_tokens.detach().float()
            projector_output_norm = projector_output_tokens_f.norm(dim=-1).mean()
            jit_norm = jit_tokens_f.norm(dim=-1).mean()
            projector_input_tokens_f = projector_input_tokens.detach().float()
            projector_input_norm = projector_input_tokens_f.norm(dim=-1).mean()
            if projector_input_tokens_f.shape == projector_output_tokens_f.shape:
                projector_delta_norm = (
                    projector_output_tokens_f - projector_input_tokens_f
                ).norm(dim=-1).mean()
            else:
                projector_delta_norm = torch.zeros((), device=x.device)
            x_mse = ((x_pred - x) ** 2).mean(dim=(1, 2, 3)).mean()

            loss_dict: dict[str, float] = {
                "total_loss": total_loss.detach().item(),
                "denoise_loss": denoise_loss.detach().item(),
                "x_mse": x_mse.detach().item(),
                "alignment_projector_delta_norm": projector_delta_norm.item(),
                "alignment_projector_input_token_norm": projector_input_norm.item(),
                "alignment_projected_token_norm": projector_output_norm.item(),
                "alignment_projector_output_input_norm_ratio": (
                    projector_output_norm / projector_input_norm.clamp_min(1e-12)
                ).item(),
                "jit_input_token_norm": jit_norm.item(),
                "t_mean": t.detach().float().mean().item(),
                "t_min": t.detach().float().min().item(),
                "t_max": t.detach().float().max().item(),
            }
            if student_tokens is not None:
                student_tokens_f = student_tokens.detach().float()
                loss_dict["dino_student_token_norm"] = student_tokens_f.norm(dim=-1).mean().item()
            if self.dino_jit_input_mode == "native" and residual_diagnostics is not None:
                loss_dict["native_jit_patch_token_norm"] = residual_diagnostics[
                    "native_tokens"
                ].detach().float().norm(dim=-1).mean().item()
            if residual_diagnostics is not None and self.dino_jit_input_mode == "gated_residual":
                native_tokens_f = residual_diagnostics["native_tokens"].detach().float()
                residual_tokens_f = residual_diagnostics["residual_tokens"].detach().float()
                gated_residual_f = residual_diagnostics["gated_residual"].detach().float()
                residual_gate_f = residual_diagnostics["gate"].detach().float()
                native_norm = native_tokens_f.norm(dim=-1).mean()
                residual_norm = residual_tokens_f.norm(dim=-1).mean()
                gated_residual_norm = gated_residual_f.norm(dim=-1).mean()
                loss_dict.update(
                    {
                        "native_jit_patch_token_norm": native_norm.item(),
                        "dino_residual_pre_gate_norm": residual_norm.item(),
                        "dino_residual_post_gate_norm": gated_residual_norm.item(),
                        "dino_residual_to_native_norm_ratio": (
                            gated_residual_norm / native_norm.clamp_min(1e-12)
                        ).item(),
                        "dino_residual_gate_mean": residual_gate_f.mean().item(),
                        "dino_residual_gate_min": residual_gate_f.min().item(),
                        "dino_residual_gate_max": residual_gate_f.max().item(),
                        "dino_residual_drop_fraction": residual_diagnostics[
                            "label_drop_mask"
                        ].detach().float().mean().item(),
                    }
                )
            elif residual_diagnostics is not None and self.dino_jit_input_mode == "time_blend":
                native_norm = residual_diagnostics["native_tokens"].detach().float().norm(dim=-1).mean()
                projected_norm = residual_diagnostics["dino_projected_tokens"].detach().float().norm(dim=-1).mean()
                jit_weighted_norm = residual_diagnostics["jit_weighted_tokens"].detach().float().norm(dim=-1).mean()
                dino_weighted_norm = residual_diagnostics["dino_weighted_tokens"].detach().float().norm(dim=-1).mean()
                loss_dict.update(
                    {
                        "native_jit_patch_token_norm": native_norm.item(),
                        "dino_projected_token_norm": projected_norm.item(),
                        "jit_weighted_token_norm": jit_weighted_norm.item(),
                        "dino_weighted_token_norm": dino_weighted_norm.item(),
                        "jit_blended_token_norm": jit_norm.item(),
                        "dino_projected_to_native_norm_ratio": (
                            projected_norm / native_norm.clamp_min(1e-12)
                        ).item(),
                        "dino_weighted_to_jit_weighted_norm_ratio": (
                            dino_weighted_norm / jit_weighted_norm.clamp_min(1e-12)
                        ).item(),
                        "dino_grounding_drop_fraction": residual_diagnostics[
                            "grounding_drop_mask"
                        ].detach().float().mean().item(),
                    }
                )
            align_tokens_f = align_tokens.detach().float()
            target_tokens_f = target_tokens.detach().float()
            target_objective_tokens_f = self._spatial_normalize_target(
                target_tokens_f
            )
            align_norm = align_tokens_f.norm(dim=-1).mean()
            target_norm = target_tokens_f.norm(dim=-1).mean()
            target_objective_norm = target_objective_tokens_f.norm(dim=-1).mean()
            pre_raw_mse = F.mse_loss(
                projector_input_tokens_f, target_tokens_f
            )
            pre_raw_cos_per_token = F.cosine_similarity(
                projector_input_tokens_f, target_tokens_f, dim=-1
            )
            pre_objective_cos_per_token = F.cosine_similarity(
                projector_input_tokens_f, target_objective_tokens_f, dim=-1
            )
            pre_raw_cos = pre_raw_cos_per_token.mean()
            pre_objective_cos = pre_objective_cos_per_token.mean()
            post_raw_cos = alignment_metrics["raw_cosine"].detach()
            post_objective_cos = alignment_metrics["loss_cosine"].detach()
            post_raw_mse = alignment_metrics["raw_mse"].detach()
            loss_dict.update(
                {
                    "alignment_pre_projector_raw_mse": pre_raw_mse.item(),
                    "alignment_raw_mse": post_raw_mse.item(),
                    "alignment_projector_raw_mse_reduction": (
                        pre_raw_mse - post_raw_mse
                    ).item(),
                    "alignment_loss": alignment_loss.detach().item(),
                    "alignment_loss_weighted": (
                        0.5 * alignment_loss
                    ).detach().item(),
                    "alignment_pre_projector_raw_cosine": pre_raw_cos.item(),
                    "alignment_raw_cosine": post_raw_cos.item(),
                    "alignment_projector_raw_cosine_gain": (
                        post_raw_cos - pre_raw_cos
                    ).item(),
                    "alignment_pre_projector_raw_cosine_std": (
                        pre_raw_cos_per_token.std(unbiased=False).item()
                    ),
                    "alignment_raw_cosine_std": alignment_metrics[
                        "raw_cosine_std"
                    ].detach().item(),
                    "alignment_pre_projector_objective_cosine": (
                        pre_objective_cos.item()
                    ),
                    "alignment_objective_cosine": post_objective_cos.item(),
                    "alignment_projector_objective_cosine_gain": (
                        post_objective_cos - pre_objective_cos
                    ).item(),
                    "alignment_pre_projector_objective_cosine_std": (
                        pre_objective_cos_per_token.std(unbiased=False).item()
                    ),
                    "alignment_objective_cosine_std": alignment_metrics[
                        "loss_cosine_std"
                    ].detach().item(),
                    "alignment_token_norm": align_norm.item(),
                    "alignment_target_token_norm": target_norm.item(),
                    "alignment_objective_target_token_norm": (
                        target_objective_norm.item()
                    ),
                    "alignment_token_norm_ratio": (
                        align_norm / target_norm.clamp_min(1e-12)
                    ).item(),
                }
            )
        return total_loss, loss_dict

    @torch.no_grad()
    def generate(self, labels: torch.Tensor) -> torch.Tensor:
        """从纯噪声 ``t=0`` 积分到图像 ``t=1``，返回 ``[B_gen,3,256,256]``。"""
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device)
        timesteps = timesteps.view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "flow_dpm_solver":
            return self._flow_dpm_solver_sample(z, timesteps, labels)
        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError(f"Unknown sampling method: {self.method}")

        for i in range(self.steps - 1):
            z = stepper(z, timesteps[i], timesteps[i + 1], labels)
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算采样 velocity；CFG 开启时把 cond/uncond 拼成 2B 一次前向。

        所以 ``gen_bsz`` 是每个 rank 的最终图像数，而 CFG 活跃区间内 DINO/JiT
        实际接收 ``2*gen_bsz``。这也是设置大 generation batch 时显存翻倍的来源。
        """
        is_guidance_active = self.cfg_scale != 1.0
        low, high = self.cfg_interval
        if is_guidance_active and ((t[0] >= high) or (low != 0 and t[0] <= low)):
            is_guidance_active = False

        if not is_guidance_active:
            x_cond, _ = self._predict_clean_from_noisy(z, t, labels)
            return (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        z_in = torch.cat([z, z])
        t_in = torch.cat([t, t])
        null_labels = torch.full_like(labels, self.num_classes)
        y_in = torch.cat([labels, null_labels])
        x_out, _ = self._predict_clean_from_noisy(z_in, t_in, y_in)
        x_cond, x_uncond = x_out.chunk(2)
        x_cfg = x_uncond + self.cfg_scale * (x_cond - x_uncond)
        return (x_cfg - z) / (1.0 - t).clamp_min(self.t_eps)

    @torch.no_grad()
    def _euler_step(self, z: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        v_pred = self._forward_sample(z, t, labels)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def _heun_step(self, z: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        v_pred_t = self._forward_sample(z, t, labels)
        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)
        return z + (t_next - t) * 0.5 * (v_pred_t + v_pred_t_next)

    @torch.no_grad()
    def _flow_dpm_solver_sample(
        self,
        z: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """PixelDiT-style FlowDPMSolver for rectified-flow velocity models.

        This is the class-conditional PixelDiT sampler specialized to our
        existing velocity interface: first step Euler, then AB2 updates.
        With ``steps=N`` it uses N model evaluations, matching PixelDiT's
        FlowDPMSolver-N NFE convention.
        """
        v_prev = None
        for i in range(self.steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t
            v = self._forward_sample(z, t, labels)
            if v_prev is None:
                z = z + dt * v
            else:
                z = z + dt * (1.5 * v - 0.5 * v_prev)
            v_prev = v
        return z

    @torch.no_grad()
    def update_ema(self) -> None:
        if self.ema_params1 is None or self.ema_params2 is None:
            self.ema_params1 = copy.deepcopy(list(self.parameters()))
            self.ema_params2 = copy.deepcopy(list(self.parameters()))
        source_params = list(self.parameters())
        for target, source in zip(self.ema_params1, source_params):
            target.detach().mul_(self.ema_decay1).add_(source, alpha=1.0 - self.ema_decay1)
        for target, source in zip(self.ema_params2, source_params):
            target.detach().mul_(self.ema_decay2).add_(source, alpha=1.0 - self.ema_decay2)
