"""Offline FDR6 evaluation for JiT-generated NPZ files or PNG directories.

Run from the Ours_BaseT_Residual_1008 project root with PYTHONPATH=.:jit:

    python -m evaluation.evaluate_fdr6 --npz samples.npz
    python -m evaluation.evaluate_fdr6 --images path/to/pngs --device cuda

The default metrics are ``fid,inception_score,fdr6``. This command does not initialize
distributed training and never modifies JiT checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .distributional import compute_metrics
from .assets import check_assets


def _load_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        key = "arr_0" if "arr_0" in data else ("images" if "images" in data else None)
        if key is None:
            raise ValueError(f"{path}: expected NPZ key 'arr_0' or 'images'; found {data.files}")
        # NPZ access materializes an independent array, even for uncompressed
        # archives; another copy would duplicate the full image batch in RAM.
        array = data[key]
    return _validate_images(array, str(path))


def _png_paths(directory: Path) -> list[Path]:
    paths = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"} and not p.name.endswith(".tmp.png")]
    if not paths:
        raise ValueError(f"No PNG/JPEG images found in {directory}")
    # JiT names images with zero-padded integer IDs. Numeric ordering also
    # handles non-padded names and avoids lexicographic 1,10,2 ordering.
    def sort_key(path: Path):
        stem = path.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)
    return sorted(paths, key=sort_key)


def _load_images(directory: Path) -> np.ndarray:
    paths = _png_paths(directory)
    with Image.open(paths[0]) as image:
        first = image.convert("RGB")
        height, width = first.height, first.width
        output = np.empty((len(paths), height, width, 3), dtype=np.uint8)
        output[0] = np.asarray(first, dtype=np.uint8)
    for index, path in enumerate(paths[1:], start=1):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if (rgb.width, rgb.height) != (width, height):
                raise ValueError(
                    f"{path}: image size {(rgb.width, rgb.height)} differs from {(width, height)}"
                )
            output[index] = np.asarray(rgb, dtype=np.uint8)
    return _validate_images(output, str(directory))


def _validate_images(array: np.ndarray, source: str) -> np.ndarray:
    if array.dtype != np.uint8 or array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(
            f"{source}: expected uint8 RGB NHWC, got dtype={array.dtype}, shape={array.shape}"
        )
    if array.shape[0] < 2:
        raise ValueError(f"{source}: at least two images are required, got {array.shape[0]}")
    return np.ascontiguousarray(array)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--npz", type=Path, help="NPZ containing uint8 RGB NHWC images")
    source.add_argument("--images", type=Path, help="directory containing PNG/JPEG images")
    parser.add_argument("--metrics", default="fid,inception_score,fdr6", help="comma-separated metrics; default: fid,inception_score,fdr6")
    parser.add_argument("--cache-root", type=Path, default=None, help="shared nanogen-evals root; defaults to the sibling directory of this project")
    parser.add_argument("--check-assets", action="store_true", help="check local reference statistics and weights without loading images or running metrics")
    parser.add_argument("--device", default="cuda", help="torch device passed to fd_evaluator")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stats-dir", type=Path, default=None, help="directory containing reference NPZ files")
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--fid-reference", default=None, help="optional FID reference name or NPZ path")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    if not metrics:
        parser.error("--metrics cannot be empty")
    supported = {"fid", "inception_score", "fdr6"}
    unknown = sorted(set(metrics) - supported)
    if unknown:
        parser.error(f"unsupported metrics: {', '.join(unknown)}; choose from fid,inception_score,fdr6")
    if not (args.npz or args.images or args.check_assets):
        parser.error("provide --npz, --images, or --check-assets")
    assets = check_assets(metrics, cache_root=args.cache_root, stats_dir=args.stats_dir, fid_reference=args.fid_reference)
    if args.check_assets:
        print(json.dumps(assets, indent=2))
        return 0 if assets["ready"] else 1
    if not assets["ready"]:
        parser.error("missing local evaluation assets:\n" + "\n".join(item["path"] for item in assets["assets"] if not item["available"]))
    if args.npz:
        if not args.npz.is_file():
            parser.error(f"NPZ does not exist: {args.npz}")
        images = _load_npz(args.npz)
        source_name = str(args.npz)
    else:
        if not args.images.is_dir():
            parser.error(f"image directory does not exist: {args.images}")
        images = _load_images(args.images)
        source_name = str(args.images)
    if images.shape[1:3] != (256, 256) and ("fdr6" in metrics or ("fid" in metrics and args.fid_reference is None)):
        parser.error("the bundled reference moments are ImageNet-256; FDR6 requires 256px images, and FID at other sizes requires --fid-reference")

    print(f"[JiT FDR6] source={source_name} shape={images.shape} dtype={images.dtype}")
    print(f"[JiT FDR6] metrics={metrics}")
    results = compute_metrics(
        images,
        metrics,
        device=args.device,
        batch_size=args.batch_size,
        stats_dir=str(args.stats_dir) if args.stats_dir else None,
        feature_cache_dir=str(args.feature_cache_dir) if args.feature_cache_dir else None,
        fid_reference=args.fid_reference,
        cache_root=str(args.cache_root) if args.cache_root else None,
    )
    report = {
        "source": source_name,
        "num_images": int(images.shape[0]),
        "image_shape": list(images.shape[1:]),
        "dtype": str(images.dtype),
        "metrics_requested": metrics,
        "metrics": results,
        "stats_dir": os.environ.get("NANOGEN_EVALS_STATS_DIR"),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "cache_root": os.environ.get("NANOGEN_EVALS_ROOT"),
        "fid_reference": args.fid_reference or "imagenet_256_fid_stats",
        "evaluator": "vendored fd-evaluator 0.1.0 (offline; not JiT online IS)",
    }
    report_json = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    print(report_json)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(report_json + "\n")
        print(f"[JiT FDR6] wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
