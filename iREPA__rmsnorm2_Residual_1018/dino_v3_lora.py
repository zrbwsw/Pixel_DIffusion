"""DINOv3 patch-token wrapper：预训练加载、LoRA 注入与多层聚合。

J18-new 使用 DINOv3-B/16：12 blocks、patch=16、embed_dim=768。对 256x256 图像，
patch embedding 产生 16x16 网格，即每层 patch token 为 ``[B,256,768]``。

student 对 block [2,5,8,11] 使用 RAEv2 MLS：选中层逐元素平均，再加最后一个
选中层的空间均值广播；teacher 因 student-only 配置保持 raw last layer。

Reference:
  - RAEv2/src/encoders/vision_encoder.py :: DINOv3MultiLayerSimpleAddEncoder
  - SSL_code/dinov3 hubconf (torch.hub source)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from dino_lora import inject_dino_lora
from resources import resource_root


THIS_DIR = Path(__file__).resolve().parent

_DINOV3_WEIGHT_FILENAMES = {
    "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}

# Where to find the DINOv3 source repo (torch.hub.load source=local).
_DINOV3_REPO_CANDIDATES = [
    THIS_DIR / "SSL_code" / "dinov3",
    Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov3_main",
]


# RAEv2's defaults — keep in sync if upstream changes.
RAEV2_DEFAULT_LAYERS = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [5, 11, 17, 23],
}


def _normalize_name(model_name: str) -> str:
    aliases = {
        "dinov3": "dinov3_vitb16",
        "dinov3_vitb": "dinov3_vitb16",
        "dinov3_vitb/16": "dinov3_vitb16",
        "dinov3_b": "dinov3_vitb16",
    }
    return aliases.get(model_name, model_name)


def _resolve_weights(model_name: str, weights_path: Optional[str]) -> Path:
    if weights_path:
        path = Path(weights_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"DINOv3 weights not found: {path}")
        return path
    filename = _DINOV3_WEIGHT_FILENAMES.get(_normalize_name(model_name))
    if filename is None:
        filename = _DINOV3_WEIGHT_FILENAMES["dinov3_vitb16"]
    candidates = [resource_root("PRETRAIN_MODELS_ROOT") / filename]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"DINOv3 weights not found in default locations: {candidates}. "
        "Set PRETRAIN_MODELS_ROOT or pass --dino_weights explicitly."
    )


def _resolve_repo() -> Path:
    env_override = os.environ.get("DINOV3_REPO")
    candidates: List[Path] = []
    if env_override:
        candidates.append(Path(env_override).expanduser().resolve())
    candidates.extend(_DINOV3_REPO_CANDIDATES)
    for cand in candidates:
        if cand.is_dir() and (cand / "hubconf.py").is_file():
            return cand
    raise FileNotFoundError(
        "DINOv3 source repo (with hubconf.py) not found. Tried: "
        f"{candidates}. Set DINOV3_REPO env var to override."
    )


def load_dino_v3(model_name: str = "dinov3_vitb16",
                 weights_path: Optional[str] = None) -> nn.Module:
    """Load DINOv3 via torch.hub from a local source tree.

    ``SSL_code/dinov3`` 使用标准 ``dinov3.*`` 包导入。这里只把本地源码根目录
    加入 ``sys.path``，不再伪造 EPG 命名空间。
    """
    model_name = _normalize_name(model_name)
    repo = _resolve_repo()
    weights = _resolve_weights(model_name, weights_path)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return torch.hub.load(
        str(repo), model_name,
        source="local", pretrained=True, weights=str(weights),
    )


def parse_layer_spec(spec: Optional[str], total_blocks: int) -> Optional[List[int]]:
    """Parse `--dino_layers` strings into explicit block indices.

    Supported forms:
      - None / ""           → None  (caller uses last layer only)
      - "last"              → [total_blocks - 1]
      - "last-k" (k int)    → last k blocks
      - "2,5,8,11" or "2.5.8.11" → explicit 0-based indices
      - "default"           → caller substitutes RAEV2_DEFAULT_LAYERS

    Indices are 0-based and must satisfy 0 <= i < total_blocks.
    """
    if spec is None or spec == "" or spec.lower() == "none":
        return None
    spec = spec.strip().lower()
    if spec == "last":
        return [total_blocks - 1]
    if spec.startswith("last-") or spec.startswith("last"):
        # last-4, last4
        k_str = spec.replace("last-", "").replace("last", "")
        k = int(k_str)
        return list(range(total_blocks - k, total_blocks))
    if spec == "default":
        return None  # caller substitutes
    parts = [p for p in spec.replace(".", ",").split(",") if p.strip()]
    indices = [int(p) for p in parts]
    for i in indices:
        if not (0 <= i < total_blocks):
            raise ValueError(f"layer index {i} out of range [0, {total_blocks})")
    return indices


class DINOv3PatchTokenEncoder(nn.Module):
    """DINOv3-B/16 wrapper returning normalized patch tokens (16x16 grid for 256 input).

    Public API:
      - .embed_dim      : int   (768 for B/16)
      - .patch_size     : int   (16)
      - .use_lora       : bool
      - .num_lora_modules
      - forward(x) -> [B, num_patches, embed_dim]

    聚合相关扩展：
      - aggregation : 'none' | 'mean' | 'mls'    (mls = RAEv2 mean+last_mean style)
      - layer_indices : None | List[int]          (None = last layer only)
    """

    SUPPORTED_AGGREGATIONS = {"none", "mean", "mls"}

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
    ):
        super().__init__()
        if aggregation not in self.SUPPORTED_AGGREGATIONS:
            raise ValueError(f"Unknown aggregation={aggregation}. "
                             f"Supported: {self.SUPPORTED_AGGREGATIONS}")
        self.model = load_dino_v3(model_name=model_name, weights_path=weights_path)
        self.embed_dim = int(getattr(self.model, "embed_dim",
                                     getattr(self.model, "num_features", 768)))
        self.final_norm_affine = bool(final_norm_affine)
        if not self.final_norm_affine:
            # Replace only the output LN after loading the pretrained state.
            # get_intermediate_layers(norm=True) also uses this LN for MLS.
            # build_encoder applies this option only to the generation student;
            # the frozen teacher keeps its original wrapper and pretrained LN.
            norm = self.model.norm
            if not isinstance(norm, nn.LayerNorm):
                raise TypeError(
                    "Removing final LN affine requires nn.LayerNorm; "
                    f"got {type(norm).__name__}"
                )
            reference = next(norm.parameters(), None)
            if reference is None:
                reference = next(self.model.parameters(), None)
            factory_kwargs = (
                {"device": reference.device, "dtype": reference.dtype}
                if reference is not None else {}
            )
            self.model.norm = nn.LayerNorm(
                norm.normalized_shape,
                eps=norm.eps,
                elementwise_affine=False,
                **factory_kwargs,
            ).train(norm.training)
        self.use_lora = use_lora
        self.aggregation = aggregation
        self.num_lora_modules = 0

        # DINOv3-B 共 12 个 block。J18-new 的 default 会在上层解析成
        # [2,5,8,11]；索引均从 0 开始。
        total_blocks = len(self.model.blocks)
        if layer_indices is None:
            self._layer_indices = None      # caller wants last layer only
        else:
            self._layer_indices = list(layer_indices)
            for i in self._layer_indices:
                if not (0 <= i < total_blocks):
                    raise ValueError(f"layer index {i} out of [0, {total_blocks})")
        self._total_blocks = total_blocks

        # 先冻结整套预训练 DINOv3。student 随后注入 LoRA，并可选择 fine-tune
        # 预训练 PatchEmbed Conv2d；teacher 保持全部 requires_grad=False。
        # DINOv3 的空间位置信息来自按坐标计算的 RoPE，不存在可解冻的 pos_embed
        # 参数，因此这里有意不提供 train_pos_embed 配置。
        for param in self.model.parameters():
            param.requires_grad_(False)

        if use_lora:
            self.num_lora_modules = inject_dino_lora(
                self.model,
                r=lora_r, alpha=lora_alpha, dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            self._set_optional_trainable("patch_embed", train_patch_embed)
            if hasattr(self.model, "cls_token") and self.model.cls_token is not None:
                self.model.cls_token.requires_grad_(train_cls_token)
            self._set_optional_trainable("norm", train_norm)

    def _set_optional_trainable(self, module_name: str, trainable: bool) -> None:
        module = getattr(self.model, module_name, None)
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad_(trainable)

    @property
    def patch_size(self) -> int:
        ps = int(getattr(self.model, "patch_size", 16))
        return ps

    def forward_patch_tokens(
        self,
        x: torch.Tensor,
        return_last_raw: bool = False,
    ):
        """返回 ``[B, num_patches, embed_dim]`` 的 normalized patch token。

        在 J18-new B/16、256x256 下，所有输出均为 ``[B,256,768]``：

        * ``aggregation='none'``：标准最后一层 normalized token；
        * ``mean/mls``：通过 ``get_intermediate_layers(..., norm=True)`` 取多层；
        * ``return_last_raw=True``：除 MLS 外额外返回未聚合的最后一层，供
          alignment-only ConvProj 使用。

        这里不会返回 CLS/register token；上层各 patch 可以和 teacher 的同一空间
        位置一一计算 cosine loss。
        """
        # Single-layer fast path：只返回 DINOv3 最后一层 normalized patch tokens。
        if self.aggregation == "none" and self._layer_indices is None:
            outputs = self.model.get_intermediate_layers(
                x, n=1, reshape=False, return_class_token=False, norm=True,
            )
            patch_tokens = outputs[0]
            if return_last_raw:
                return patch_tokens, patch_tokens
            return patch_tokens

        # 多层路径：J18-new n_arg=[2,5,8,11]，outputs 是 4 个
        # [B,256,768] Tensor；梯度会从 MLS 和 raw-last 两条支路共同回到 student。
        n_arg = self._layer_indices if self._layer_indices is not None else 1
        outputs = self.model.get_intermediate_layers(
            x, n=n_arg, reshape=False, return_class_token=False, norm=True,
        )
        if (
            self._layer_indices is not None
            and len(self._layer_indices) > 0
            and self._layer_indices[-1] == self._total_blocks - 1
        ):
            last_raw = outputs[-1]
        else:
            last_raw = self.model.get_intermediate_layers(
                x, n=1, reshape=False, return_class_token=False, norm=True,
            )[0]
        # outputs is a tuple of K tensors, one per selected layer.
        # Each tensor has shape [B, N, D]:
        #   B = batch size, N = number of patch tokens, D = DINO feature dim.
        if self.aggregation in ("none", "mean"):
            # Stack to [K, B, N, D], then average over dim=0 (the layer axis).
            # Result stays [B, N, D].
            patch_tokens = torch.stack(outputs, dim=0).mean(dim=0)
        elif self.aggregation == "mls":
            # RAEv2-style DINOv3 MLS:
            #   1) layer mean: [K, B, N, D] -> mean over dim=0 -> [B, N, D]
            #   2) final_mean: last selected layer [B, N, D] -> mean over
            #      dim=1 (patch/token axis) -> [B, 1, D]
            #   3) broadcast-add [B, 1, D] onto [B, N, D].
            patch_tokens = torch.stack(outputs, dim=0).mean(dim=0)
            final_mean = outputs[-1].mean(dim=1, keepdim=True)
            patch_tokens = patch_tokens + final_mean
        else:
            raise RuntimeError(f"unreachable: aggregation={self.aggregation}")
        if return_last_raw:
            return patch_tokens, last_raw
        return patch_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_patch_tokens(x)

    def forward_with_last_raw(self, x: torch.Tensor):
        return self.forward_patch_tokens(x, return_last_raw=True)
