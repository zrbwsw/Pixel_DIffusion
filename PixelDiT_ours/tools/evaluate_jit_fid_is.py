"""JiT-compatible FID/IS for uint8 RGB NPZ samples, using torch-fidelity directly."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
DEFAULT_STATS = REPOSITORY_ROOT / "fid_stats" / "jit_in256_stats.npz"
DEFAULT_WEIGHTS = REPOSITORY_ROOT / "fid_stats" / "weights-inception-2015-12-05-6726825d.pth"


def load_samples(path: Path) -> np.ndarray:
    with np.load(path) as data:
        key = "arr_0" if "arr_0" in data else ("images" if "images" in data else None)
        if key is None:
            raise ValueError(f"{path}: expected NPZ key 'arr_0' or 'images'; found {data.files}")
        images = data[key]
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"{path}: expected uint8 RGB NHWC, got {images.dtype} {images.shape}")
    if images.shape[0] < 2 or images.shape[1:3] != (256, 256):
        raise ValueError(f"{path}: expected at least two 256x256 images, got {images.shape}")
    return np.ascontiguousarray(images)


def check_assets(stats_path: Path, weights_path: Path) -> dict:
    if not stats_path.is_file():
        raise FileNotFoundError(f"FID statistics not found: {stats_path}")
    if not weights_path.is_file() or weights_path.stat().st_size == 0:
        raise FileNotFoundError(f"Inception weights not found: {weights_path}")
    with np.load(stats_path) as stats:
        if "mu" not in stats or "sigma" not in stats:
            raise ValueError(f"{stats_path}: expected mu/sigma, found {stats.files}")
        if stats["mu"].shape != (2048,) or stats["sigma"].shape != (2048, 2048):
            raise ValueError(f"{stats_path}: invalid mu/sigma shapes")
    return {"fid_statistics": str(stats_path), "inception_weights": str(weights_path), "ready": True}


def extract_features(images: np.ndarray, weights_path: Path, device: str, batch_size: int):
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

    extractor = FeatureExtractorInceptionV3(
        name="inception-v3-compat",
        features_list=["2048", "logits_unbiased"],
        feature_extractor_weights_path=str(weights_path),
    ).to(device).eval()
    features, logits = [], []
    with torch.inference_mode():
        for start in tqdm(range(0, len(images), batch_size), desc="JiT Inception"):
            batch = images[start : start + batch_size]
            batch = torch.from_numpy(np.ascontiguousarray(batch)).permute(0, 3, 1, 2).contiguous()
            feature, logit = extractor(batch.to(device))
            features.append(feature.cpu())
            logits.append(logit.cpu())
    return torch.cat(features), torch.cat(logits)


def evaluate(args) -> dict:
    assets = check_assets(args.fid_stats, args.weights)
    if args.check_assets:
        return assets
    images = load_samples(args.npz)
    features, logits = extract_features(images, args.weights, args.device, args.batch_size)

    from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
    from torch_fidelity.metric_isc import isc_features_to_metric

    generated_stats = fid_features_to_statistics(features)
    with np.load(args.fid_stats) as reference:
        reference_stats = {"mu": reference["mu"], "sigma": reference["sigma"]}
    fid = fid_statistics_to_metric(generated_stats, reference_stats, False)["frechet_inception_distance"]
    inception = isc_features_to_metric(
        logits,
        splits=args.isc_splits,
        shuffle=not args.no_isc_shuffle,
        rng_seed=args.isc_seed,
    )
    metrics = {
        "fid": float(fid),
        "inception_score": float(inception["inception_score_mean"]),
        "inception_score_std": float(inception["inception_score_std"]),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"Non-finite metrics: {metrics}")
    return {
        "source": str(args.npz.resolve()),
        "num_images": int(images.shape[0]),
        "image_shape": list(images.shape[1:]),
        "dtype": str(images.dtype),
        "metrics": metrics,
        **assets,
        "evaluator": "JiT torch-fidelity direct-NPZ implementation",
        "isc_splits": args.isc_splits,
        "isc_shuffle": not args.no_isc_shuffle,
        "isc_seed": args.isc_seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, help="NPZ containing uint8 RGB NHWC images")
    parser.add_argument("--fid-stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--isc-splits", type=int, default=10)
    parser.add_argument("--isc-seed", type=int, default=2020)
    parser.add_argument("--no-isc-shuffle", action="store_true")
    parser.add_argument("--check-assets", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.isc_splits < 1:
        parser.error("batch size and IS splits must be positive")
    if not args.check_assets and args.npz is None:
        parser.error("provide --npz or --check-assets")
    report = evaluate(args)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
