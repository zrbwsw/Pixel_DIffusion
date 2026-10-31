"""离线构建 DINOv3 backbone：使用随目录官方源码和本地预训练权重。

这里只负责模型结构与权重加载，不处理 LoRA、条件注入或聚合；这些由
dino_encoder.py 负责。GPU 训练不依赖临时下载或 GitHub/HuggingFace。
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
DINOV3_SOURCE = PROJECT_ROOT / "third_party" / "dinov3"  # 随方法保存的官方 DINOv3 运行源码及许可证

_WEIGHT_FILENAMES = {
    "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}


def load_dino_v3(model_name: str, weights_path: Optional[str] = None) -> nn.Module:
    """按模型名创建官方结构，再严格恢复本地 state_dict。

    显式 weights_path 优先；否则从 PRETRAIN_MODELS_ROOT 或主仓库
    pretrain_models/ 查找。strict=True 会阻止缺层或名称不匹配静默通过。
    """
    model_name = model_name.lower()  # 配置统一使用 dinov3_vit{s,b,l}16 形式
    if weights_path:
        weights = Path(os.path.expandvars(weights_path)).expanduser().resolve()  # 支持环境变量和 ~ 的显式路径
    else:
        root = Path(os.environ.get("PRETRAIN_MODELS_ROOT") or REPOSITORY_ROOT / "pretrain_models")
        weights = root.expanduser().resolve() / _WEIGHT_FILENAMES[model_name]
    if not weights.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights}")

    if str(DINOV3_SOURCE) not in sys.path:
        sys.path.insert(0, str(DINOV3_SOURCE))  # 让官方 hub.backbones 从随目录源码导入
    from dinov3.hub import backbones

    # pretrained=False 只表示不让构造函数联网下载；随后立即加载本地权重。
    model = getattr(backbones, model_name)(pretrained=False, weights=str(weights))
    state_dict = torch.load(weights, map_location="cpu", weights_only=True)  # 在 CPU 读取，避免加载时占用额外 GPU 显存
    model.load_state_dict(state_dict, strict=True)  # 预训练参数完整覆盖结构，之后才会注入 LoRA
    return model
