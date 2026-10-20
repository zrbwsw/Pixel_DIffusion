"""Export Ours' numbered PNGs to an atomic, RGB uint8 NHWC NPZ."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image


def export_pngs(image_dir, npz_path, num_images, image_size, metadata=None):
    """Stream arr_0.npy into an uncompressed NPZ without a full image-array copy.

    The final NPZ is the completion marker. Interrupted temporary files can be
    rewritten; a completed NPZ is never overwritten. Only numbered PNGs count.
    """
    image_dir, npz_path = Path(image_dir).resolve(), Path(npz_path).resolve()
    if npz_path.suffix != ".npz":
        raise ValueError("Export path must end in .npz")
    if num_images < 2 or image_size < 1:
        raise ValueError("Expected at least two images and a positive image size")
    if npz_path.exists():
        raise FileExistsError(f"Completed export already exists: {npz_path}")
    paths = [p for p in image_dir.glob("*.png") if not p.name.endswith(".tmp.png")]
    if any(not p.stem.isdigit() for p in paths):
        raise ValueError(f"Expected numbered PNGs only in {image_dir}")
    paths.sort(key=lambda p: int(p.stem))
    if [int(p.stem) for p in paths] != list(range(num_images)):
        raise ValueError(f"Expected exactly PNG IDs 0..{num_images - 1} in {image_dir}")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = npz_path.with_name(npz_path.name + ".incomplete")
    manifest_path = npz_path.with_suffix(".manifest.json")
    manifest_tmp = manifest_path.with_name(manifest_path.name + ".incomplete")
    shape = (num_images, image_size, image_size, 3)
    try:
        # ZIP64 is required for 50k ImageNet-256 images (about 9.16 GiB).
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            with archive.open("arr_0.npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array_header_2_0(stream, {
                    "descr": "|u1", "fortran_order": False, "shape": shape,
                })
                for path in paths:
                    with Image.open(path) as image:
                        if image.mode != "RGB" or image.size != (image_size, image_size):
                            raise ValueError(f"Expected RGB {image_size}x{image_size}: {path}")
                        stream.write(np.asarray(image, dtype=np.uint8).tobytes())
        digest = hashlib.sha256()
        with temporary.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        manifest = {
            "metadata": metadata or {}, "image_dir": str(image_dir),
            "npz_path": str(npz_path), "npz_key": "arr_0", "shape": list(shape),
            "dtype": "uint8", "layout": "NHWC", "color": "RGB",
            "pixel_source": "decoded numbered PNGs, without resizing or normalization",
            "sha256": digest.hexdigest(), "size_bytes": temporary.stat().st_size,
        }
        os.replace(temporary, npz_path)
        # The NPZ is the completion marker. Publish its sidecar only after the
        # data archive is complete, so an interrupted run cannot advertise a
        # finished export.
        manifest_tmp.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        os.replace(manifest_tmp, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="exact numbered-PNG directory")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()
    result = export_pngs(args.images, args.npz, args.num_images, args.image_size)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
