"""Load the pinned, unmodified implementation used by the rmsnorm2 reference run."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
REFERENCE_ROOT = PROJECT_ROOT / "rmsnorm2_reference"


def verify_reference():
    manifest = json.loads((PROJECT_ROOT / "rmsnorm2_reference_manifest.json").read_text())
    for group, root in (("reference_files", REFERENCE_ROOT), ("shared_dinov3_files", PROJECT_ROOT)):
        for relative, expected in manifest[group].items():
            path = root / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Pinned rmsnorm2 source changed: {path}")
    return {"source_commit": manifest["source_commit"],
            "reference_files": len(manifest["reference_files"]),
            "shared_dinov3_files": len(manifest["shared_dinov3_files"])}


def load_reference():
    verify_reference()
    repository = PROJECT_ROOT.parent
    for key, relative in (("PRETRAIN_MODELS_ROOT", "pretrain_models"),
                          ("FID_STATS_ROOT", "fid_stats"),
                          ("NANOGEN_EVALS_ROOT", "nanogen-evals")):
        os.environ.setdefault(key, str(repository / relative))
    os.environ.setdefault("IMAGENET_ROOT", "/mnt/shared-storage-user/datafactory-share/gushishen/imagenet")
    os.environ.setdefault("DINOV3_REPO", str(PROJECT_ROOT / "SSL_code/dinov3"))
    for name in ("main_jit", "model_jit", "engine_jit", "main_dino_jit", "dino_jit_denoiser",
                 "dino_encoder", "dino_v3_lora", "config_utils", "resource_paths"):
        module = sys.modules.get(name)
        if module is not None and not Path(module.__file__).resolve().is_relative_to(REFERENCE_ROOT):
            raise RuntimeError(f"Conflicting legacy module {name}; run the rmsnorm2 entrypoint in a fresh process")
    for path in (REFERENCE_ROOT, REFERENCE_ROOT / "jit"):
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    return importlib.import_module("main_dino_jit")
