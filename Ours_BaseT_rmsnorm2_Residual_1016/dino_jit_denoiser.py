"""DINO+LoRA 与 JiT 的融合、iREPA 对齐、训练和采样。

当前 reverse 路径：带噪图像 → DINO+LoRA / JiT patch embed → 两路 RMSNorm
→ (1-t)×DINO + t×JiT → JiT blocks → 像素预测。
干净图像 → 冻结 teacher；JiT 第4层 → ConvProj，与 teacher 做对齐。
B/16、256×256 输入对应 [B,256,768] patch tokens。
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

from dino_encoder import build_dino_student, build_irepa_teacher
from jit_ddt_head import JiTDDTHead
from blend_time import sigmoid_blend_time


class DINOJiTDenoiser(nn.Module):
    """沿用 iREPA 的加噪、velocity loss 和 CFG，增加 DINO 生成分支与 token 融合。"""

    def __init__(self, args):
        super().__init__()
        # 直接使用 YAML 配置；默认 JiT-B/16，12 blocks，hidden=768。
        self.dino_jit_input_mode = args.dino_jit_input_mode
        self.dino_input_projector_type = args.dino_input_projector
        self.dino_grounding_drop_on_label_drop = args.dino_grounding_drop_on_label_drop
        self.dino_time_blend_reverse = args.dino_time_blend_reverse
        self.dino_time_blend_transform = args.dino_time_blend_transform
        self.dino_time_blend_key = args.dino_time_blend_key
        self.dino_time_blend_a = args.dino_time_blend_a
        self.dino_jit_pre_blend_rmsnorm = args.dino_jit_pre_blend_rmsnorm
        self.dino_preprocess_input = args.dino_preprocess_input

        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            z_dims=[],
        )
        # 两种模式都使用并训练 JiT 原生 patch embedding。

        # 沿用 iREPA：z_t=t*x+(1-t)*noise；t=0 为噪声，t=1 为干净图像。
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

        # 对齐用 ConvProj 与可选的 JiT 输入 PgT 相互独立。
        self.dino_aggregation = args.dino_aggregation
        self.dino_layers = args.dino_layers
        self.jit_add_pos_embed = args.jit_add_pos_embed_to_dino_tokens
        self.dino_zscore_alpha = args.dino_zscore_alpha
        self.jit_alignment_depth = args.jit_alignment_depth

        teacher = build_irepa_teacher(args)
        # teacher 自身负责冻结和 eval；普通 list 保持它不进入 checkpoint / EMA。
        self._teacher_box = [teacher]
        self._teacher_device: Optional[torch.device] = None

        # native 仅 JiT+iREPA；time_blend 增加生成侧 DINO+LoRA。
        self.dino_student: Optional[nn.Module] = None
        if self.dino_jit_input_mode == "time_blend":
            self.dino_student = build_dino_student(args)
        dino_feature_dim = int(teacher.embed_dim)
        self.dino_student_projector = ProjectionLayer(  # irepa convproj
            "conv",                                     # 类型：3x3 卷积投影（而非 MLP）
            hidden_size=self.net.hidden_size,           # 输入：JiT 中间层特征（B/16 为 768）
            z_dim=dino_feature_dim,                     # 输出：teacher 特征维度（DINOv3-B 为 768）
            proj_kwargs_kernel_size=int(args.dino_projector_kernel_size),
        )
        self.dino_input_projector: Optional[nn.Module] = None
        if self.dino_input_projector_type == "pixeldit2":
            self.dino_input_projector = TimestepProjection(
                in_dim=dino_feature_dim,
                out_dim=self.net.hidden_size,
                hidden_size=self.net.hidden_size,
                num_heads=self.net.num_heads,
            )
        # 两路分别做 token 内的 RMSNorm，各自学习 gamma。
        self.jit_pre_blend_norm = (
            RMSNorm(self.net.hidden_size, eps=1e-6)
            if self.dino_jit_pre_blend_rmsnorm
            else nn.Identity()
        )
        self.dino_pre_blend_norm = (
            RMSNorm(self.net.hidden_size, eps=1e-6)
            if self.dino_jit_pre_blend_rmsnorm
            else nn.Identity()
        )

        self.use_ddt_head = args.use_ddt_head
        self.ddt_head: Optional[nn.Module] = None
        if self.use_ddt_head:
            # 配置默认 0：跟随 JiT 主干宽度/头数（768/12，天然整除）。
            self.ddt_head_dim = args.ddt_head_dim or self.net.hidden_size
            self.ddt_head_num_heads = args.ddt_head_num_heads or self.net.num_heads
            self.ddt_head = JiTDDTHead(
                hidden_size=self.ddt_head_dim,
                patch_size=self.net.patch_size,
                in_channels=3,
                bottleneck_dim=args.ddt_head_bottleneck_dim,
                input_size=self.img_size,
                depth=args.ddt_head_depth,
                num_heads=self.ddt_head_num_heads,
                mlp_ratio=4.0,
                cond_hidden_size=self.net.hidden_size,
                time_reinject=True,
                embed_type="bottleneck",
            )
        # 启用 DDTHead 时冻结 JiT 原生输出头；保留模块以兼容旧 checkpoint。
        for param in self.net.final_layer.parameters():
            param.requires_grad_(not self.use_ddt_head)

        # ImageNet mean/std 写死给 iREPA teacher；pm1 版本是 [-1,1] 域统计，
        # 供 student 可选的 t 相关解析 norm 使用。
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("dino_mean", mean, persistent=False)
        self.register_buffer("dino_std", std, persistent=False)
        self.register_buffer("dino_mean_pm1", 2.0 * mean - 1.0, persistent=False)
        self.register_buffer("dino_std_pm1", 2.0 * std, persistent=False)

    @property   # 使用一个list封装teacher,装进_teacher_box，这个函数是交给外部访问的，既能像正常子模块一样用 model.teacher，又让 PyTorch 的参数管理体系完全忽略它（冻结、不进 checkpoint/EMA/DDP 同步）。
    def teacher(self) -> nn.Module:
        # 独立的 FrozenDINOv3Encoder；只提供冻结的 iREPA 对齐目标。
        return self._teacher_box[0]

    def _ensure_teacher_device(self, device: torch.device) -> None:
        if self._teacher_device != device:
            self.teacher.to(device)
            self._teacher_device = device
        self.teacher.eval()

    def drop_labels_with_mask(self, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """随机将类别置为 null id=1000，同时返回 DropRes 使用的掩码。"""
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        dropped = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return dropped, drop

    def drop_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """返回 CFG 训练用的类别标签。"""
        dropped, _ = self.drop_labels_with_mask(labels)
        return dropped

    def sample_t(self, n: int, device=None) -> torch.Tensor:
        """从 logit-normal 采样干净图像权重 t，形状 [B]。"""
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def _preprocess_noisy_student_for_dino(self, noisy_z: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """student 可选的 t 相关解析 norm：按加噪路径 z_t=t·x+(1-t)·ε 的解析
        均值/方差标准化，mu_t=t·mu_x；var_t=t²·std_x²+(1-t)²·noise_scale²。
        """
        z_float = noisy_z.float()
        t = timesteps.to(device=noisy_z.device, dtype=z_float.dtype).reshape(-1, 1, 1, 1)
        mean_t = t * self.dino_mean_pm1.to(device=noisy_z.device, dtype=z_float.dtype)
        std_t = (
            t.square() * self.dino_std_pm1.to(device=noisy_z.device, dtype=z_float.dtype).square()
            + (1.0 - t).square() * float(self.noise_scale) ** 2
        ).sqrt().clamp_min(1e-6)
        return (z_float - mean_t) / std_t

    def _encode_noisy_student(self, noisy_z: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """带噪图像 → DINO+LoRA/MLS token；dino_preprocess_input 控制是否做 t 相关解析 norm。"""
        student_input = (
            self._preprocess_noisy_student_for_dino(noisy_z, timesteps)
            if self.dino_preprocess_input
            else noisy_z
        )
        return self.dino_student(student_input)

    def _encode_dino_pair(self, clean_x: torch.Tensor, noisy_z: torch.Tensor, timesteps: torch.Tensor) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """一次取出两条 DINO 通路的 token：

        - student_tokens：带噪图像 → DINO+LoRA（可训练），供 JiT 输入融合；
          native 模式没有这条支路，返回 None；
        - target_tokens：干净图像 → frozen iREPA teacher，作对齐目标。
        """
        student_tokens = (
            None
            if self.dino_jit_input_mode == "native"
            else self._encode_noisy_student(noisy_z, timesteps)
        )
        return student_tokens, self._encode_teacher_target(clean_x)

    def _encode_teacher_target(self, clean_x: torch.Tensor) -> torch.Tensor:
        """跟随 iREPA 源码：干净图像 [-1,1] → [0,1] → 固定 ImageNet 标准化 → frozen teacher。"""
        self._ensure_teacher_device(clean_x.device)
        x01 = (clean_x.float() + 1.0) * 0.5
        mean = self.dino_mean.to(device=clean_x.device, dtype=x01.dtype)
        std = self.dino_std.to(device=clean_x.device, dtype=x01.dtype)
        with torch.no_grad():
            features = self.teacher.forward_features((x01 - mean) / std)
        return features["x_norm_patchtokens"]

    def _project_student_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        # 此处 student 指 JiT 第4层特征，经过 ConvProj 后与 teacher 对齐。
        return self.dino_student_projector(tokens)

    def _compose_jit_input_tokens(
        self,
        noisy_image: torch.Tensor,
        dino_tokens: Optional[torch.Tensor],
        timesteps: torch.Tensor,
        label_drop_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """native 直接使用 JiT token；time_blend 将两路 token 按时间融合。"""
        native_tokens = self.net.x_embedder(noisy_image)
        if self.dino_jit_input_mode == "native":
            return native_tokens, {"native_tokens": native_tokens}

        # time_blend：可选 PgT / RMSNorm，再按时间加权；位置编码稍后添加。
        projected_tokens = dino_tokens
        if self.dino_input_projector is not None:
            # PgT 将 DINO token 投影到 JiT 输入空间，并注入原始时间 t。
            grid_hw = tuple(size // self.net.patch_size for size in noisy_image.shape[-2:])  # noisy_image [B,3,H,W] → token 网格 (H/16, W/16)，256px 下为 (16,16)
            projected_tokens = self.dino_input_projector(
                dino_tokens.to(dtype=self.dino_input_projector.input_proj.weight.dtype),  # [B,N,768]（N=grid_h·grid_w=256，DINOv3-B 特征维），对齐 PgT 权重 dtype
                timesteps.flatten(), grid_hw,  # 原始时间 t: [B]；grid_hw 供 PgT 内部 2D RoPE 定位
            )  # 输出 [B,N,768]（JiT hidden_size），与 native_tokens 同形
        # 过各自的可学习rmsnrom
        native_tokens = self.jit_pre_blend_norm(native_tokens)
        projected_tokens = self.dino_pre_blend_norm(projected_tokens)
        # sigmoid 仅变换融合系数；PgT、JiT 和扩散过程仍使用原始 t。
        blend_t = timesteps.flatten()
        if self.dino_time_blend_transform == "sigmoid":
            blend_t = sigmoid_blend_time(
                blend_t, key=self.dino_time_blend_key, a=self.dino_time_blend_a,
            )
        t = blend_t.to(device=native_tokens.device, dtype=native_tokens.dtype)
        t = t.view(-1, 1, 1)
        reverse = self.dino_time_blend_reverse
        jit_weight = t if reverse else (1.0 - t)
        dino_weight = (1.0 - t) if reverse else t
        # 零权重也保留两路计算图，使 DDP 的参数参与情况一致。
        jit_weighted = jit_weight * native_tokens
        dino_weighted = dino_weight * projected_tokens
        grounding_drop_mask = torch.zeros(
            t.shape[0], device=t.device, dtype=torch.bool,
        )
        if self.dino_grounding_drop_on_label_drop and label_drop_mask is not None:  #DropRes逻辑，true则DINO path token随着label一同drop
            grounding_drop_mask = label_drop_mask.flatten().to(device=t.device, dtype=torch.bool,)
            mask = grounding_drop_mask.view(-1, 1, 1)   # [B] → [B,1,1]，便于对 token 维广播
            jit_weighted = torch.where(mask, native_tokens, jit_weighted)   # 被 drop 的样本：直接用未加权的 JiT token
            dino_weighted = dino_weighted.masked_fill(mask, 0)  # 被 drop 的样本：DINO 路贡献清零
        fused_tokens = jit_weighted + dino_weighted
        return fused_tokens, {
            "native_tokens": native_tokens,
            "dino_projected_tokens": projected_tokens,
            "jit_weighted_tokens": jit_weighted,
            "dino_weighted_tokens": dino_weighted,
            "grounding_drop_mask": grounding_drop_mask,
        }

    def _spatial_normalize_target(self, target_tokens: torch.Tensor) -> torch.Tensor:
        # iREPA 空间 z-score：沿 token 维标准化，alpha 默认0.8。
        target_tokens = target_tokens.float()
        mean = target_tokens.mean(dim=1, keepdim=True)
        std = target_tokens.std(dim=1, keepdim=True)
        return (target_tokens - self.dino_zscore_alpha * mean) / (std + 1e-6)

    def _alignment_loss(
        self,
        aligned_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """投影后的 JiT token 与空间标准化 teacher 做负 cosine 对齐。"""
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
        # 沿用 JiT blocks；取第 alignment_depth 层做对齐，末层还原图像。alignment_depth=4 对应 blocks[3]；采样时为 None，仅返回图像。
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
                # JiT-B 在第5个 block 前插入32个类别 token：[B,256,768] → [B,288,768]。
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, net.in_context_len, 1)
                in_context_tokens = in_context_tokens + net.in_context_posemb.to(device=x.device, dtype=x.dtype)
                x = torch.cat([in_context_tokens.to(dtype=x.dtype), x], dim=1)
                class_tokens_inserted = True
            rope = net.feat_rope if i < net.in_context_start else net.feat_rope_incontext
            x = block(x, c, rope)
            # iREPA 默认取第4层输出；对齐只使用空间 token。
            if alignment_depth is not None and (i + 1) == alignment_depth:
                alignment_tokens = x[:, net.in_context_len:] if class_tokens_inserted else x

        if class_tokens_inserted:
            x = x[:, net.in_context_len:]
        if self.use_ddt_head:
            patches = self.ddt_head(noisy_image, x, t_emb=t_emb)
        else:
            patches = net.final_layer(x, c)
        output = net.unpatchify(patches, net.patch_size)

        if alignment_depth is None: #　alignment_depth=4 对应 blocks[3]；采样时为 None，仅返回图像
            return output
        return output, alignment_tokens

    def _predict_clean_from_noisy(
        self,
        noisy_z: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """采样前向：带噪图像 → 两路 token → JiT → 像素预测，不计算 teacher。"""
        student_tokens = (
            None
            if self.dino_jit_input_mode == "native"
            else self._encode_noisy_student(noisy_z, timesteps)
        )
        label_drop_mask = (
            labels == self.num_classes
            if self.dino_grounding_drop_on_label_drop
            else None
        )
        jit_tokens, _ = self._compose_jit_input_tokens(
            noisy_z, student_tokens, timesteps, label_drop_mask=label_drop_mask,
        )
        x_pred = self._jit_forward_from_tokens(jit_tokens, timesteps.flatten(), labels, noisy_image=noisy_z)
        return x_pred, jit_tokens

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        """训练：x 为 [-1,1] 图像 [B,3,H,W]，labels 为类别 [B]。"""
        # 训练时随机把 label 置为 1000；评估时保留调用方给定标签。
        if self.training:
            labels_dropped, label_drop_mask = self.drop_labels_with_mask(labels)
        else:
            labels_dropped = labels
            label_drop_mask = (
                labels == self.num_classes
                if self.dino_grounding_drop_on_label_drop
                else None
            )

        # 按样本采样 t 并加噪；[B,1,1,1] 广播到图像的 C/H/W。
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale
        z = t * x + (1.0 - t) * e
        # 用同一 t_eps 计算目标和预测 velocity，保持 iREPA 的损失定义。
        v = (x - z) / (1.0 - t).clamp_min(self.t_eps)

        student_tokens, target_tokens = self._encode_dino_pair(x, z, t)
        jit_tokens, input_diagnostics = self._compose_jit_input_tokens(
            z, student_tokens, t, label_drop_mask=label_drop_mask,
        )
        x_pred, projector_input_tokens = self._jit_forward_from_tokens(
            jit_tokens, t.flatten(), labels_dropped, noisy_image=z,
            alignment_depth=self.jit_alignment_depth,
        )
        align_tokens = self._project_student_tokens(projector_input_tokens)
        # 模型显式预测 clean x，再用同一解析关系转换为 velocity 监督。
        v_pred = (x_pred - z) / (1.0 - t).clamp_min(self.t_eps)

        # velocity MSE + 0.5 × iREPA 对齐损失。
        denoise_per_sample = ((v - v_pred) ** 2).mean(dim=(1, 2, 3))
        denoise_loss = denoise_per_sample.mean()
        alignment_loss, alignment_metrics = self._alignment_loss(align_tokens, target_tokens)
        total_loss = denoise_loss + 0.5 * alignment_loss

        loss_dict = self._training_metrics(
            x=x, x_pred=x_pred, t=t, total_loss=total_loss,
            denoise_loss=denoise_loss, alignment_loss=alignment_loss,
            student_tokens=student_tokens, jit_tokens=jit_tokens,
            input_diagnostics=input_diagnostics,
            projector_input_tokens=projector_input_tokens, align_tokens=align_tokens,
            target_tokens=target_tokens, alignment_metrics=alignment_metrics,
        )
        return total_loss, loss_dict

    @torch.no_grad()
    def generate(self, labels: torch.Tensor) -> torch.Tensor:
        """从 t=0 的噪声积分到 t=1 的图像。"""
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device)
        timesteps = timesteps.view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "flow_dpm_solver":
            return self._flow_dpm_solver_sample(z, timesteps, labels)
        stepper = {"euler": self._euler_step, "heun": self._heun_step}[self.method]

        # 最后一步固定 Euler，避免在 t=1 额外求一次 velocity。
        for i in range(self.steps - 1):
            z = stepper(z, timesteps[i], timesteps[i + 1], labels)
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算采样 velocity；CFG 区间内将条件/无条件样本拼成 2B 一次前向。"""
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
        """FlowDPM：第一步 Euler，后续 AB2；N 步使用 N 次模型前向。"""
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
        """沿用两组 EMA；teacher 未注册在 parameters 中。"""
        if self.ema_params1 is None or self.ema_params2 is None:
            self.ema_params1 = copy.deepcopy(list(self.parameters()))
            self.ema_params2 = copy.deepcopy(list(self.parameters()))
        source_params = list(self.parameters())
        for target, source in zip(self.ema_params1, source_params):
            target.detach().mul_(self.ema_decay1).add_(source, alpha=1.0 - self.ema_decay1)
        for target, source in zip(self.ema_params2, source_params):
            target.detach().mul_(self.ema_decay2).add_(source, alpha=1.0 - self.ema_decay2)

    @torch.no_grad()
    def _training_metrics(
        self, *, x, x_pred, t, total_loss, denoise_loss, alignment_loss,
        student_tokens, jit_tokens, input_diagnostics,
        projector_input_tokens, align_tokens, target_tokens, alignment_metrics,
    ):
        """仅记录 loss、token 范数和对齐统计，不参与反向传播。"""
        projector_output_tokens_f = align_tokens.detach().float()
        jit_tokens_f = jit_tokens.detach().float()
        projector_output_norm = projector_output_tokens_f.norm(dim=-1).mean()
        jit_norm = jit_tokens_f.norm(dim=-1).mean()
        projector_input_tokens_f = projector_input_tokens.detach().float()
        projector_input_norm = projector_input_tokens_f.norm(dim=-1).mean()
        projector_delta_norm = (
            projector_output_tokens_f - projector_input_tokens_f
        ).norm(dim=-1).mean()
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
        if self.dino_jit_input_mode == "native":
            loss_dict["native_jit_patch_token_norm"] = input_diagnostics[
                "native_tokens"
            ].detach().float().norm(dim=-1).mean().item()
        elif self.dino_jit_input_mode == "time_blend":
            native_norm = input_diagnostics["native_tokens"].detach().float().norm(dim=-1).mean()
            projected_norm = input_diagnostics["dino_projected_tokens"].detach().float().norm(dim=-1).mean()
            jit_weighted_norm = input_diagnostics["jit_weighted_tokens"].detach().float().norm(dim=-1).mean()
            dino_weighted_norm = input_diagnostics["dino_weighted_tokens"].detach().float().norm(dim=-1).mean()
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
                    "dino_grounding_drop_fraction": input_diagnostics[
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
        return loss_dict
