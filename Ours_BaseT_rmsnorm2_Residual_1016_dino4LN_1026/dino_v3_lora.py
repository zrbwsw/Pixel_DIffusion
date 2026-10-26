"""生成侧 DINOv3：加载预训练权重 → 注入 LoRA → 各层 LN → MLS 聚合。

默认 B/16 输入256×256，输出 [B,256,768] patch tokens，不含 CLS/register。
MLS 沿用 RAEv2 的 DINOv3MultiLayerSimpleAddEncoder；teacher 在 irepa_teacher.py。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from dino_lora import inject_dino_lora
# 保留资源预检使用的加载函数别名。
from dinov3_backbone import load_dino_v3, _resolve_repo, _resolve_weights


# RAEv2 默认选层，索引从0开始。
RAEV2_DEFAULT_LAYERS = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [5, 11, 17, 23],
}


def parse_layer_spec(spec: Optional[str], total_blocks: int) -> Optional[List[int]]:
    """支持 last、last-4 / last4、2,5,8,11；default 由工厂替换为默认选层。"""
    if spec is None:
        return None
    spec = spec.strip().lower()
    if spec in ("", "none", "default"):
        return None
    if spec == "last":
        return [total_blocks - 1]
    if spec.startswith("last"):
        k = int(spec.replace("last-", "").replace("last", ""))
        return list(range(total_blocks - k, total_blocks))
    return [int(part) for part in spec.replace(".", ",").split(",") if part.strip()]


class DINOv3PatchTokenEncoder(nn.Module):
    """先冻结 backbone，再启用 LoRA 及指定参数；forward 返回 normalized patch tokens。

    可选条件注入（time_cond/label_cond）：patch_embed 之后、进 block 之前，
    给每个 patch token 加上 JiT 同款的 t_embedder(t) / y_embedder(y)，
    让 LoRA 能感知扩散时间和类别。
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        weights_path: Optional[str] = None,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_target_modules: str = "qkv,proj,fc1,fc2",
        train_patch_embed: bool = True,
        train_cls_token: bool = False,
        train_norm: bool = False,
        aggregation: str = "none",
        layer_indices: Optional[Sequence[int]] = None,
        final_norm_affine: bool = True,
        time_cond: bool = False,
        label_cond: bool = False,
        num_classes: int = 1000,
        mls_layer_norms: bool = False,
    ):
        super().__init__()
        self.model = load_dino_v3(model_name=model_name, weights_path=weights_path)
        self.embed_dim = self.model.embed_dim
        self.final_norm_affine = final_norm_affine
        if not final_norm_affine:
            # NoAffine 只移除输出 LN 的 gamma/beta，保留归一化和 eps。
            norm = self.model.norm
            self.model.norm = nn.LayerNorm(
                norm.normalized_shape,
                eps=norm.eps,
                elementwise_affine=False,   #这里移除了
                device=norm.weight.device,
                dtype=norm.weight.dtype,
            ).train(norm.training)

        self.use_lora = use_lora
        self.aggregation = aggregation
        self.num_lora_modules = 0
        self._layer_indices = list(layer_indices) if layer_indices is not None else None
        self._total_blocks = len(self.model.blocks)
        self.mls_layer_norms = mls_layer_norms

        # 原始 backbone 全冻结；不启用 LoRA 时，到这里即完成冻结设置。
        self.model.requires_grad_(False)
        if use_lora:
            # qkv/proj/fc1/fc2：原 Linear 冻结，仅训练 LoRA A/B。
            self.num_lora_modules = inject_dino_lora(
                self.model,
                r=lora_r, alpha=lora_alpha, dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            # 额外可训项：patch embed、CLS token、输出 LN 的 gamma/beta。
            # 工厂默认冻结 CLS 和输出 LN；block 内部 norm 始终冻结。
            self.model.patch_embed.requires_grad_(train_patch_embed)
            self.model.cls_token.requires_grad_(train_cls_token)
            self.model.norm.requires_grad_(train_norm)

        if self.mls_layer_norms:
            # 中间层各自学习 LN（gamma=1、beta=0）；最后层沿用预训练 model.norm。
            norm = self.model.norm
            for index in self._layer_indices:
                if index != self._total_blocks - 1:
                    self.model.add_module(f"norm_{index}", nn.LayerNorm(
                        norm.normalized_shape, eps=norm.eps,
                        device=norm.weight.device, dtype=norm.weight.dtype,
                    ))

        # 可选的 t/y 条件注入：与 JiT 相同的 Timestep/LabelEmbedder，维度对齐 backbone。
        # 延迟导入，仅启用时才要求 jit/ 在 PYTHONPATH 上。
        self.time_cond = time_cond
        self.label_cond = label_cond
        if time_cond or label_cond:
            from model_jit import LabelEmbedder, TimestepEmbedder
            if time_cond:
                self.cond_t_embedder = TimestepEmbedder(self.embed_dim)
            if label_cond:
                self.cond_y_embedder = LabelEmbedder(num_classes, self.embed_dim)
            self._wrap_patch_embed_cond()
        self._current_cond: Optional[torch.Tensor] = None

    def _compute_cond(self, t: Optional[torch.Tensor], y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """t: [B]、y: [B]（null 类为 num_classes）→ [B,1,1,D] 加性条件；都关则 None。"""
        cond = None
        if self.time_cond:
            cond = self.cond_t_embedder(t.reshape(-1).float()).reshape(-1, 1, 1, self.embed_dim)
        if self.label_cond:
            y_emb = self.cond_y_embedder(y).reshape(-1, 1, 1, self.embed_dim)
            cond = y_emb if cond is None else cond + y_emb
        return cond

    def _wrap_patch_embed_cond(self) -> None:
        """包装 patch_embed.forward：输出 [B,H,W,D] 上广播加条件；模块结构/state_dict 不变。"""
        orig_forward = self.model.patch_embed.forward

        def patched_forward(x, *args, **kwargs):
            tokens = orig_forward(x, *args, **kwargs)
            if self._current_cond is not None:
                tokens = tokens + self._current_cond.to(dtype=tokens.dtype)
            return tokens

        self.model.patch_embed.forward = patched_forward

    @property
    def patch_size(self) -> int:
        return self.model.patch_size

    def forward_patch_tokens(self, x: torch.Tensor, t: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None, return_last_raw: bool = False):
        """none 取单层，mean 取层均值，MLS 再加最后一个选中层的空间均值。
        last_raw 指 backbone 最后一层经过 LN、尚未聚合的 patch tokens。
        t/y 为可选条件输入，仅在 time_cond/label_cond 启用时使用。
        """
        self._current_cond = self._compute_cond(t, y)
        try:
            if self.aggregation == "none" and self._layer_indices is None:
                patch_tokens = self.model.get_intermediate_layers(
                    x, n=1, reshape=False, return_class_token=False, norm=True,
                )[0]
                return (patch_tokens, patch_tokens) if return_last_raw else patch_tokens

            # 独立 LN 模式先取未经过 final LN 的 token，避免重复归一化。
            n_arg = self._layer_indices if self._layer_indices is not None else 1
            outputs = self.model.get_intermediate_layers(
                x, n=n_arg, reshape=False, return_class_token=False,
                norm=not self.mls_layer_norms,
            )
            if self.mls_layer_norms:
                # 此时 outputs 是选中层尚未经过输出 LN 的 patch tokens。
                # 默认依次对应 [2,5,8,11]；每项形状 [B,N,D]，B/16 的 N=256、D=768。
                # zip 把层索引和对应 tokens 配成一组，逐层应用 LN 后重新组成 tuple。
                outputs = tuple(
                    # B/16 共12个 block，末层索引为11：使用保留预训练参数的 model.norm。
                    # 其余层读取初始化时创建的独立 LN，例如 getattr(model, "norm_2")。
                    # getattr 只查找模块，不会创建新 LN；末尾 (tokens) 执行该 LN 的 forward。
                    (self.model.norm if index == self._total_blocks - 1
                     else getattr(self.model, f"norm_{index}"))(tokens)
                    for index, tokens in zip(self._layer_indices, outputs)
                )
            # 到这里，outputs 中每项都已完成对应的输出 LN，尚未做 MLS 聚合。
            # 如果选层列表的末项就是 backbone 最后层，直接复用它，不再运行一遍 DINO。
            if self._layer_indices and self._layer_indices[-1] == self._total_blocks - 1:
                # last_raw 的 raw 指“尚未聚合”，不是“尚未 LN”。这里只引用最后层张量，
                # 不再归一化或求均值；它也不是 lastmean，后者在下面的 MLS 加法中计算。
                last_raw = outputs[-1]
            else:
                # 其他取层配置沿用原有兼容分支，单独获取 backbone 真正的最后层。
                # n=1 表示只取最后一个 block；norm=True 表示经过 model.norm；
                # reshape=False 保持 [B,N,D]，return_class_token=False 不返回 CLS。
                # 返回值是长度为1的 tuple，[0] 取出该层的 patch tokens。
                # 默认 [2,5,8,11] 不走此分支；未指定层列表时这里可能重复一次前向。
                # 保持原有额外前向及调用顺序，避免改变训练态的随机数消耗。
                last_raw = self.model.get_intermediate_layers(
                    x, n=1, reshape=False, return_class_token=False, norm=True,
                )[0]

            patch_tokens = torch.stack(outputs, dim=0).mean(dim=0)
            if self.aggregation == "mls":
                # MLS = 选中层均值 + 最后一个选中层的空间均值广播。
                patch_tokens = patch_tokens + outputs[-1].mean(dim=1, keepdim=True)
            return (patch_tokens, last_raw) if return_last_raw else patch_tokens
        finally:
            self._current_cond = None

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward_patch_tokens(x, t=t, y=y)

    def forward_with_last_raw(self, x: torch.Tensor, t: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None):
        return self.forward_patch_tokens(x, t=t, y=y, return_last_raw=True)
