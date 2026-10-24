"""四层双路径 DINO/JiT denoiser。

前四层分别处理 JiT 和 DINO token；DINO 第四层先做 iREPA 对齐，
再按时间加权相加，最后接回 JiT 的 class token、后续 blocks 和输出头。
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn

from blend_time import sigmoid_blend_time
from dino_jit_denoiser import DINOJiTDenoiser


class FourDiTResidualDenoiser(DINOJiTDenoiser):
    """在原 denoiser 上增加 post-block 双路径输入模式。"""

    def __init__(self, args):
        self.dual_path_mode = args.dino_jit_input_mode == "post_block_time_blend"
        requested_mode = args.dino_jit_input_mode

        # 复用父类的 DINO student、teacher、投影器和训练超参数构建逻辑。
        # 父类只识别 native/time_blend，这里临时映射为 time_blend 以构建 student。
        if self.dual_path_mode:
            args.dino_jit_input_mode = "time_blend"
        super().__init__(args)
        args.dino_jit_input_mode = requested_mode

        self.dual_path_depth = int(getattr(args, "dual_path_depth", 4))  # 双路径各跑前 4 层
        self.requested_dino_jit_input_mode = requested_mode
        if self.dual_path_mode:
            # JiT 原生 blocks[0:4] 是 JiT path；DINO path 使用同结构的独立副本。
            # deepcopy 保持初始化一致，但参数和梯度完全不共享。
            self.dino_prefix_blocks = nn.ModuleList(
                copy.deepcopy(list(self.net.blocks[: self.dual_path_depth]))
            )
            # 父类的诊断逻辑沿用 time_blend 分支；实际前向由本类接管。
            self.dino_jit_input_mode = "time_blend"
        else:
            self.dino_prefix_blocks = None

    def _run_prefix_blocks(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
        blocks,
    ) -> torch.Tensor:
        """一条路径独立经过前四层，只处理空间 patch token。"""
        x = tokens  # [B,256,768]（256 = 16×16 patch 网格）
        if self.jit_add_pos_embed:
            x = x + self.net.pos_embed.to(device=x.device, dtype=x.dtype)  # 两路都加 JiT 位置编码
        for block in blocks:
            x = block(x, condition, self.net.feat_rope)  # 前缀段无 in-context token，用 feat_rope
        return x

    def _post_block_tokens(
        self,
        noisy_image: torch.Tensor,
        dino_tokens: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
        label_drop_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """两路各跑四层，DINO 路输出单独保留用于 iREPA 对齐。"""
        native_tokens = self.net.x_embedder(noisy_image)  # [B,3,256,256] → [B,256,768]
        projected_tokens = dino_tokens  # student 输出 [B,256,768]
        if self.dino_input_projector is not None:
            grid_hw = tuple(
                size // self.net.patch_size for size in noisy_image.shape[-2:]
            )  # token 网格 (16,16)
            projected_tokens = self.dino_input_projector(
                dino_tokens.to(dtype=self.dino_input_projector.input_proj.weight.dtype),
                timesteps.flatten(),  # PgT 注入原始 t
                grid_hw,
            )  # → [B,256,768]，与 JiT token 同形

        t_flat = timesteps.flatten()  # [B]，原始扩散时间
        condition = self.net.t_embedder(t_flat) + self.net.y_embedder(labels)  # [B,768]，两路共用
        jit_block_tokens = self._run_prefix_blocks(
            native_tokens, condition, self.net.blocks[: self.dual_path_depth]
        )  # JiT 路第4层输出 [B,256,768]
        dino_block_tokens = self._run_prefix_blocks(
            projected_tokens, condition, self.dino_prefix_blocks
        )  # DINO 路第4层原始输出 [B,256,768]；对齐用这个（不过 RMSNorm）

        # 第4层输出是实际融合输入；两路先各自 RMSNorm，再计算融合权重。
        jit_prefix = self.jit_pre_blend_norm(jit_block_tokens)
        dino_prefix = self.dino_pre_blend_norm(dino_block_tokens)

        # 加权被放到 iREPA 对齐之后；扩散和 PgT 仍使用原始 t。
        blend_t = t_flat
        if self.dino_time_blend_transform == "sigmoid":
            blend_t = sigmoid_blend_time(
                blend_t,
                key=self.dino_time_blend_key,
                a=self.dino_time_blend_a,
            )
        blend_t = blend_t.to(device=jit_prefix.device, dtype=jit_prefix.dtype)
        blend_t = blend_t.view(-1, 1, 1)  # [B] → [B,1,1]，对每个 token 广播
        if self.dino_time_blend_reverse:
            jit_weight, dino_weight = blend_t, 1.0 - blend_t  # reverse：t 越大 JiT 权重越大
        else:
            jit_weight, dino_weight = 1.0 - blend_t, blend_t

        jit_weighted = jit_weight * jit_prefix
        dino_weighted = dino_weight * dino_prefix
        grounding_drop_mask = torch.zeros(
            blend_t.shape[0], device=blend_t.device, dtype=torch.bool
        )
        if self.dino_grounding_drop_on_label_drop and label_drop_mask is not None:
            grounding_drop_mask = label_drop_mask.flatten().to(
                device=blend_t.device, dtype=torch.bool
            )
            mask = grounding_drop_mask.view(-1, 1, 1)
            jit_weighted = torch.where(mask, jit_prefix, jit_weighted)
            dino_weighted = dino_weighted.masked_fill(mask, 0)

        fused_tokens = jit_weighted + dino_weighted  # [B,256,768]
        diagnostics = {
            "native_tokens": jit_prefix,
            "dino_projected_tokens": dino_prefix,
            "jit_weighted_tokens": jit_weighted,
            "dino_weighted_tokens": dino_weighted,
            "grounding_drop_mask": grounding_drop_mask,
        }
        # 对齐使用 DINO 第4层原始输出（未过 RMSNorm），RMSNorm 只作用于融合分支。
        return fused_tokens, dino_block_tokens, diagnostics

    def _run_shared_tail(
        self,
        fused_tokens: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
        noisy_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """融合后插入 class token，执行剩余 JiT blocks 和输出头。"""
        net = self.net
        t_emb = net.t_embedder(timesteps.flatten())  # [B,768]
        y_emb = net.y_embedder(labels)  # [B,768]
        condition = t_emb + y_emb
        x = fused_tokens  # [B,256,768]
        class_tokens_inserted = False

        for i in range(self.dual_path_depth, len(net.blocks)):
            if net.in_context_len > 0 and i == net.in_context_start:
                # 第4层插入32个类别 in-context token：[B,256,768] → [B,288,768]。
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, net.in_context_len, 1)
                in_context_tokens = in_context_tokens + net.in_context_posemb.to(
                    device=x.device, dtype=x.dtype
                )
                x = torch.cat([in_context_tokens, x], dim=1)
                class_tokens_inserted = True
            rope = net.feat_rope if i < net.in_context_start else net.feat_rope_incontext  # 288 长度用后者
            x = net.blocks[i](x, condition, rope)

        if class_tokens_inserted:
            x = x[:, net.in_context_len :]  # 剥掉 in-context token → [B,256,768]
        if self.use_ddt_head:
            patches = self.ddt_head(noisy_image, x, t_emb=t_emb)
        else:
            patches = net.final_layer(x, condition)
        return net.unpatchify(patches, net.patch_size)

    def _predict_clean_from_noisy(
        self,
        noisy_z: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.dual_path_mode:
            return super()._predict_clean_from_noisy(noisy_z, timesteps, labels)

        student_tokens = self._encode_noisy_student(noisy_z, timesteps, labels)
        label_drop_mask = (
            labels == self.num_classes
            if self.dino_grounding_drop_on_label_drop
            else None
        )
        fused_tokens, _, _ = self._post_block_tokens(
            noisy_z, student_tokens, timesteps, labels, label_drop_mask
        )
        return (
            self._run_shared_tail(fused_tokens, timesteps, labels, noisy_image=noisy_z),
            fused_tokens,
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        if not self.dual_path_mode:
            return super().forward(x, labels)

        if self.training:
            labels_dropped, label_drop_mask = self.drop_labels_with_mask(labels)
        else:
            labels_dropped = labels
            label_drop_mask = (
                labels == self.num_classes
                if self.dino_grounding_drop_on_label_drop
                else None
            )

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        noise = torch.randn_like(x) * self.noise_scale
        z = t * x + (1.0 - t) * noise
        velocity = (x - z) / (1.0 - t).clamp_min(self.t_eps)

        student_tokens, target_tokens = self._encode_dino_pair(
            x, z, t, labels_dropped
        )
        fused_tokens, dino_prefix, input_diagnostics = self._post_block_tokens(
            z, student_tokens, t, labels_dropped, label_drop_mask
        )
        x_pred = self._run_shared_tail(
            fused_tokens, t, labels_dropped, noisy_image=z
        )
        aligned_tokens = self._project_student_tokens(dino_prefix) if self.irepa_alignment else None
        # 注意：此处的 dino_prefix 是 _post_block_tokens 返回的 DINO 路第4层原始输出（未过 RMSNorm）。

        velocity_pred = (x_pred - z) / (1.0 - t).clamp_min(self.t_eps)
        denoise_loss = ((velocity - velocity_pred) ** 2).mean(dim=(1, 2, 3)).mean()
        if self.irepa_alignment:
            alignment_loss, alignment_metrics = self._alignment_loss(
                aligned_tokens, target_tokens
            )
            total_loss = denoise_loss + 0.5 * alignment_loss
        else:
            alignment_loss, alignment_metrics = None, None
            total_loss = denoise_loss

        loss_dict = self._training_metrics(
            x=x,
            x_pred=x_pred,
            t=t,
            total_loss=total_loss,
            denoise_loss=denoise_loss,
            alignment_loss=alignment_loss,
            student_tokens=student_tokens,
            jit_tokens=fused_tokens,
            input_diagnostics=input_diagnostics,
            projector_input_tokens=dino_prefix if self.irepa_alignment else None,
            align_tokens=aligned_tokens,
            target_tokens=target_tokens,
            alignment_metrics=alignment_metrics,
        )
        return total_loss, loss_dict
