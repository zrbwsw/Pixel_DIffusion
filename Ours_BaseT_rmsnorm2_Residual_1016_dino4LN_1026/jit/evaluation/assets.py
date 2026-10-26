"""Local-only reference statistics and extractor weight caches."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

# evaluation/ is nested below jit/ in the clean project; nanogen-evals is its
# sibling under Pixel_Diffusion.
DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[3] / "nanogen-evals"
STATS_REPO = "datasets--nanovisionx--nanogen-evals-stats"
INCEPTION_WEIGHTS = "weights-inception-2015-12-05-6726825d.pth"


def resolve_cache_root(cache_root=None):
    return Path(cache_root or os.environ.get("NANOGEN_EVALS_ROOT") or DEFAULT_CACHE_ROOT).expanduser().resolve()


def inception_weights_path(cache_root=None):
    """Share online/offline weights without configuring offline reference stats."""
    return resolve_cache_root(cache_root) / "torch/hub/checkpoints" / INCEPTION_WEIGHTS


def configure_cache(cache_root=None, stats_dir=None):
    root = resolve_cache_root(cache_root)
    stats = stats_dir or os.environ.get("NANOGEN_EVALS_STATS_DIR")
    if stats:
        stats = Path(stats).expanduser().resolve()
    else:
        repo = root / "stats" / STATS_REPO
        ref = repo / "refs" / "main"
        ref_name = ref.read_text().strip() if ref.is_file() else ""
        candidate = repo / "snapshots" / ref_name if ref_name else None
        if candidate is not None and candidate.is_dir():
            stats = candidate
        else:
            snapshots = sorted((repo / "snapshots").glob("*"))
            if len(snapshots) != 1:
                raise FileNotFoundError(f"Cannot resolve nanogen reference snapshot under {repo}; pass --stats-dir")
            stats = snapshots[0]
    if not stats.is_dir():
        raise FileNotFoundError(f"Reference statistics directory not found: {stats}")
    # Set before importing timm/torch-fidelity: their loaders use these caches.
    os.environ.update(
        NANOGEN_EVALS_ROOT=str(root),
        NANOGEN_EVALS_STATS_DIR=str(stats),
        NANOGEN_EVALS_CACHE_DIR=str(root / "stats"),
        HF_HOME=str(root / "huggingface"),
        HF_HUB_CACHE=str(root / "huggingface" / "hub"),
        HUGGINGFACE_HUB_CACHE=str(root / "huggingface" / "hub"),
        TORCH_HOME=str(root / "torch"),
        HF_HUB_OFFLINE="1",
    )
    import torch
    from huggingface_hub import constants

    # Also support callers that imported these libraries before this adapter.
    constants.HF_HOME = os.environ["HF_HOME"]
    constants.HF_HUB_CACHE = constants.HUGGINGFACE_HUB_CACHE = os.environ["HF_HUB_CACHE"]
    constants.HF_HUB_OFFLINE = True
    torch.hub.set_dir(str(root / "torch" / "hub"))
    return root, stats


def check_assets(metrics, *, cache_root=None, stats_dir=None, fid_reference=None):
    """Inspect local files only; never download or instantiate an extractor."""
    root, stats = configure_cache(cache_root, stats_dir)
    catalogue = yaml.safe_load((Path(__file__).parent / "vendor/fd_evaluator/catalogue.yaml").read_text())
    names = catalogue["bundles"]["fdr6"] if "fdr6" in metrics else ["inception"]
    references = {catalogue["extractors"][name]["default_reference"] for name in names} if "fdr6" in metrics else set()
    if "fid" in metrics:
        references.add(fid_reference or "imagenet_256_fid_stats")
    entries = []
    for name in sorted(references):
        path = stats / catalogue["references"][name] if name in catalogue["references"] else Path(name)
        entries.append({"kind": "statistics", "name": name, "path": str(path), "available": path.is_file() and path.stat().st_size > 0})
    for name in names:
        spec = catalogue["extractors"][name]
        if name == "inception":
            path = inception_weights_path(root)
            available = path.is_file() and path.stat().st_size > 0
        else:
            from huggingface_hub import try_to_load_from_cache
            from timm.models import get_pretrained_cfg

            cfg = get_pretrained_cfg(spec["model"])
            repo = cfg.hf_hub_id
            filename = cfg.hf_hub_filename or "pytorch_model.bin"
            candidates = ["model.safetensors", filename] if filename == "pytorch_model.bin" else [filename]
            cached = [try_to_load_from_cache(repo, candidate, cache_dir=str(root / "huggingface/hub")) for candidate in candidates]
            found = next((p for p in cached if isinstance(p, str) and Path(p).is_file() and Path(p).stat().st_size > 0), None)
            path = found or f"{root}/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/<revision>/{filename}"
            available = found is not None
        entries.append({"kind": "weights", "name": name, "path": str(path), "available": available})
    return {"cache_root": str(root), "stats_dir": str(stats), "ready": all(item["available"] for item in entries), "assets": entries}
