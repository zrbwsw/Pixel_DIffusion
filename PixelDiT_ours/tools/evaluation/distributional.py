"""PixelDiT adapter for the vendored RAEv2 FID/FDR6 evaluator.

The evaluator itself is kept unchanged under ``evaluation/vendor``. This
module handles PixelDiT's image representation and local cache defaults.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .assets import check_assets
_FDR6_COMPONENTS = (
    "fdr_inception", "fdr_convnext", "fdr_dinov2_cls",
    "fdr_mae_cls", "fdr_siglip_cls", "fdr_clip_cls",
)


def compute_metrics(
    images: np.ndarray | str,
    metrics: Sequence[str] = ("fid", "inception_score", "fdr6"),
    *,
    device: str = "cuda",
    batch_size: int = 64,
    stats_dir: Optional[str] = None,
    feature_cache_dir: Optional[str] = None,
    fid_reference: Optional[str] = None,
    cache_root: Optional[str] = None,
) -> dict[str, float]:
    """Compute FDR6 (and optionally FID/IS) with local RAEv2 assets.

    ``images`` must be a uint8 RGB NHWC NumPy array or an NPZ containing
    ``arr_0``.  ``stats_dir`` points directly at the seven reference NPZ files;
    it is used through ``NANOGEN_EVALS_STATS_DIR`` by the vendored package.
    """
    assets = check_assets(metrics, cache_root=cache_root, stats_dir=stats_dir, fid_reference=fid_reference)
    missing_assets = [item["path"] for item in assets["assets"] if not item["available"]]
    if missing_assets:
        raise FileNotFoundError("Offline evaluation assets are missing:\n" + "\n".join(missing_assets))

    # NANOGEN_EVALS_CACHE_DIR belongs to statistics downloads. Feature-cache
    # configuration is passed separately and must not redirect that cache.
    cache_path = str(Path(feature_cache_dir).expanduser()) if feature_cache_dir else None

    from .vendor.fd_evaluator import compute_metrics as _compute_metrics

    requested = [str(metric) for metric in metrics]
    kwargs = {
        "images": images,
        "metrics": requested,
        "device": device,
        "batch_size": batch_size,
        "feature_cache_dir": cache_path,
        "feature_cache_key": None,
        "reference_feature_cache_key": None,
        "verbose": True,
    }
    if fid_reference:
        kwargs["fid_reference"] = fid_reference
    result = {str(key): float(value) for key, value in _compute_metrics(**kwargs).items()}
    required = set(requested)
    if "fdr6" in required:
        required.update(_FDR6_COMPONENTS)
    missing = sorted(required - result.keys())
    if missing:
        raise RuntimeError(f"Evaluator did not return required metrics: {missing}")
    invalid = sorted(key for key, value in result.items() if not math.isfinite(value))
    if invalid:
        raise RuntimeError(f"Evaluator returned non-finite metrics: {invalid}")
    return result


__all__ = ["compute_metrics"]
