"""Shared resource defaults for training, sampling and evaluation."""

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESOURCE_ROOTS = {
    "PRETRAIN_MODELS_ROOT": REPOSITORY_ROOT / "pretrain_models",
    "FID_STATS_ROOT": REPOSITORY_ROOT / "fid_stats",
    "NANOGEN_EVALS_ROOT": REPOSITORY_ROOT / "nanogen-evals",
    "IMAGENET_ROOT": Path("/mnt/shared-storage-user/datafactory-share/gushishen/imagenet"),
}


def resource_root(name: str) -> Path:
    """Honor a nonempty environment override, independent of the caller's cwd."""
    return Path(os.environ.get(name) or DEFAULT_RESOURCE_ROOTS[name]).expanduser().resolve()
