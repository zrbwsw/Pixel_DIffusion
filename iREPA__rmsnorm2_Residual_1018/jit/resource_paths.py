"""Local resources for the H-cluster JiT+iREPA baseline (no downloads)."""

import os
from pathlib import Path

IREPA_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = IREPA_ROOT.parent


def resource_path(env_name, default):
    return Path(os.environ.get(env_name) or default).expanduser().resolve()


def imagenet_root():
    return resource_path("IMAGENET_ROOT", "/mnt/shared-storage-user/datafactory-share/gushishen/imagenet")


def dino_assets(model_config="b", repo=None, weights=None):
    filenames = {"b": "dino_vitbase16_pretrain.pth", "s": "dino_deitsmall16_pretrain.pth"}
    if model_config not in filenames:
        raise ValueError(f"Unsupported DINOv1 model: {model_config}")
    repo = Path(repo).expanduser().resolve() if repo else resource_path("DINO_REPO", IREPA_ROOT / "third_party/dino")
    root = resource_path("PRETRAIN_MODELS_ROOT", PROJECT_ROOT / "pretrain_models")
    weights = Path(weights).expanduser().resolve() if weights else resource_path("DINOV1_WEIGHTS", root / filenames[model_config])
    for path in (repo / "hubconf.py", repo / "vision_transformer.py", weights):
        if not path.is_file():
            raise FileNotFoundError(f"Missing local DINOv1 resource: {path}. Set --dino_repo/--dino_weights or DINO_REPO/DINOV1_WEIGHTS.")
    return repo, weights


def evaluation_assets(img_size):
    if img_size not in (256, 512):
        raise ValueError(f"No JiT FID statistics configured for {img_size}px")
    root = resource_path("FID_STATS_ROOT", PROJECT_ROOT / "fid_stats")
    stats = root / f"jit_in{img_size}_stats.npz"
    weights = root / "weights-inception-2015-12-05-6726825d.pth"
    for path in (stats, weights):
        if not path.is_file():
            raise FileNotFoundError(f"Missing local evaluation resource: {path}. Set FID_STATS_ROOT.")
    return stats, weights
