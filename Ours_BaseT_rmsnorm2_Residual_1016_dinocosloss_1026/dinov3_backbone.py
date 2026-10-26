"""DINOv3 本地 backbone 加载；不包含 LoRA、聚合或冻结策略。

teacher 和 student 共用资源解析，但每次加载各自独立的模型实例。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from jit.resource_paths import resource_root


THIS_DIR = Path(__file__).resolve().parent

_DINOV3_WEIGHT_FILENAMES = {
    "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}

_DINOV3_REPO_CANDIDATES = [
    THIS_DIR / "SSL_code" / "dinov3",
    Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov3_main",
]


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

    ``SSL_code/dinov3`` 使用标准 ``dinov3.*`` 包导入。这里只把本地源码根目录加入 ``sys.path``。
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
