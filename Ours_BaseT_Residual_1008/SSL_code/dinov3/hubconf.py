# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from dinov3.hub.backbones import (
    dinov3_convnext_base,
    dinov3_convnext_large,
    dinov3_convnext_small,
    dinov3_convnext_tiny,
    dinov3_vit7b16,
    dinov3_vitb16,
    dinov3_vith16plus,
    dinov3_vitl16,
    dinov3_vitl16plus,
    dinov3_vits16,
    dinov3_vits16plus,
)


def _optional_dependency_stub(name, error):
    def _stub(*args, **kwargs):
        raise ImportError(f"{name} requires optional DINOv3 dependencies that are not available: {error}") from error

    return _stub


try:
    from dinov3.hub.classifiers import dinov3_vit7b16_lc
except ImportError as _error:
    dinov3_vit7b16_lc = _optional_dependency_stub("dinov3_vit7b16_lc", _error)

try:
    from dinov3.hub.depthers import dinov3_vit7b16_dd, dinov3_vitl16_chmv2
except ImportError as _error:
    dinov3_vit7b16_dd = _optional_dependency_stub("dinov3_vit7b16_dd", _error)
    dinov3_vitl16_chmv2 = _optional_dependency_stub("dinov3_vitl16_chmv2", _error)

try:
    from dinov3.hub.detectors import dinov3_vit7b16_de
except ImportError as _error:
    dinov3_vit7b16_de = _optional_dependency_stub("dinov3_vit7b16_de", _error)

try:
    from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
except ImportError as _error:
    dinov3_vitl16_dinotxt_tet1280d20h24l = _optional_dependency_stub(
        "dinov3_vitl16_dinotxt_tet1280d20h24l", _error
    )

try:
    from dinov3.hub.segmentors import dinov3_vit7b16_ms
except ImportError as _error:
    dinov3_vit7b16_ms = _optional_dependency_stub("dinov3_vit7b16_ms", _error)

dependencies = ["torch", "numpy"]
