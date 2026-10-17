#!/usr/bin/env python3
"""Prepare official ILSVRC2012 archives as a standard ImageFolder dataset.

This tool performs archive/layout conversion only:

    ILSVRC2012_img_train.tar -> train/<wnid>/*.JPEG
    ILSVRC2012_img_val.tar   -> val/<wnid>/*.JPEG

Validation labels come from the official ILSVRC2012 devkit. Images remain at
their original JPEG resolution. Crop, resize, normalization, and horizontal
flip belong to the training loader, not this one-time archive preparation.

Typical usage:

    python prepare_imagenet.py \
        --archive-dir /path/to/official_archives \
        --output-dir /data0/datasets/imagenet \
        --workers 8

The output is directly compatible with torchvision ImageFolder and RepPixel's
raw ImageNet loader. Interrupted fresh preparations can be resumed because all
temporary state lives under <output-dir>/.prepare_imagenet and completed class
directories are published atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


ARCHIVES = {
    "train": ("ILSVRC2012_img_train.tar", "1d675b47d978889d74fa0da5fadfb00e"),
    "val": ("ILSVRC2012_img_val.tar", "29b22e2961454d5413ddabcf34fc5622"),
    "devkit": ("ILSVRC2012_devkit_t12.tar.gz", "fa75699e90414af021442c21a62c3abf"),
}
EXPECTED_CLASSES = 1000
EXPECTED_IMAGES = {"train": 1_281_167, "val": 50_000}
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}
WNID_RE = re.compile(r"^n\d{8}$")
VAL_NAME_RE = re.compile(r"^ILSVRC2012_val_\d{8}\.JPEG$", re.IGNORECASE)


@dataclass(frozen=True)
class DevkitMetadata:
    wnid_to_classes: dict[str, tuple[str, ...]]
    val_wnids: list[str]

    @property
    def wnids(self) -> list[str]:
        return sorted(self.wnid_to_classes)


def _log(message: str) -> None:
    print(f"[prepare_imagenet] {message}", flush=True)


def _md5(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_archive(archive_dir: Path, split: str, verify_md5: bool) -> Path:
    filename, expected_md5 = ARCHIVES[split]
    path = archive_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing official {split} archive: {path}")
    if verify_md5:
        _log(f"Verifying MD5 for {path.name} ...")
        actual_md5 = _md5(path)
        if actual_md5 != expected_md5:
            raise RuntimeError(
                f"MD5 mismatch for {path}: expected {expected_md5}, got {actual_md5}"
            )
    return path


def _copy_tar_member(tar: tarfile.TarFile, member: tarfile.TarInfo, target: Path) -> None:
    if not member.isfile():
        raise RuntimeError(f"Expected a regular file in archive, got: {member.name}")
    source = tar.extractfile(member)
    if source is None:
        raise RuntimeError(f"Could not read archive member: {member.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.part-{os.getpid()}")
    try:
        with source, temporary.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
        if temporary.stat().st_size != member.size:
            raise RuntimeError(
                f"Short write for {member.name}: expected {member.size}, "
                f"got {temporary.stat().st_size}"
            )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _find_extracted_devkit(search_roots: Iterable[Path]) -> Path | None:
    for root in search_roots:
        candidates = (root, root / "ILSVRC2012_devkit_t12")
        for candidate in candidates:
            data_dir = candidate / "data"
            if (data_dir / "meta.mat").is_file() and (
                data_dir / "ILSVRC2012_validation_ground_truth.txt"
            ).is_file():
                return candidate
    return None


def _extract_devkit_files(devkit_archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    wanted = {
        "meta.mat": destination / "data" / "meta.mat",
        "ILSVRC2012_validation_ground_truth.txt": (
            destination / "data" / "ILSVRC2012_validation_ground_truth.txt"
        ),
    }
    with tarfile.open(devkit_archive, "r:*") as tar:
        found: set[str] = set()
        for member in tar.getmembers():
            basename = Path(member.name).name
            if basename in wanted:
                _copy_tar_member(tar, member, wanted[basename])
                found.add(basename)
    missing = sorted(set(wanted) - found)
    if missing:
        raise RuntimeError(f"Devkit archive is missing required files: {missing}")
    return destination


def _load_devkit(
    archive_dir: Path,
    output_dir: Path,
    staging_dir: Path,
    verify_md5: bool,
) -> DevkitMetadata:
    devkit_root = _find_extracted_devkit((archive_dir, output_dir))
    if devkit_root is None:
        archive = _require_archive(archive_dir, "devkit", verify_md5)
        devkit_root = _extract_devkit_files(
            archive, staging_dir / "ILSVRC2012_devkit_t12"
        )

    try:
        import scipy.io as sio
    except ImportError as error:
        raise RuntimeError(
            "scipy is required to parse ImageNet devkit meta.mat. "
            "Install it with `pip install scipy`."
        ) from error

    meta_path = devkit_root / "data" / "meta.mat"
    ground_truth_path = (
        devkit_root / "data" / "ILSVRC2012_validation_ground_truth.txt"
    )
    meta = sio.loadmat(meta_path, squeeze_me=True)["synsets"]
    num_children = list(zip(*meta))[4]
    leaves = [meta[index] for index, count in enumerate(num_children) if count == 0]
    class_ids, wnids, class_names = list(zip(*leaves))[:3]

    idx_to_wnid = {int(class_id): str(wnid) for class_id, wnid in zip(class_ids, wnids)}
    wnid_to_classes = {
        str(wnid): tuple(str(name).split(", "))
        for wnid, name in zip(wnids, class_names)
    }
    val_class_ids = [int(line) for line in ground_truth_path.read_text().splitlines()]
    val_wnids = [idx_to_wnid[class_id] for class_id in val_class_ids]

    if len(wnid_to_classes) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Devkit contains {len(wnid_to_classes)} leaf classes, "
            f"expected {EXPECTED_CLASSES}"
        )
    if len(val_wnids) != EXPECTED_IMAGES["val"]:
        raise RuntimeError(
            f"Devkit contains {len(val_wnids)} validation labels, "
            f"expected {EXPECTED_IMAGES['val']}"
        )
    if set(val_wnids) != set(wnid_to_classes):
        raise RuntimeError("Validation labels do not cover the 1000 devkit leaf classes")
    return DevkitMetadata(wnid_to_classes=wnid_to_classes, val_wnids=val_wnids)


def _image_files(class_dir: Path) -> list[Path]:
    files: list[Path] = []
    for entry in os.scandir(class_dir):
        path = Path(entry.path)
        if entry.is_dir(follow_symlinks=False):
            raise RuntimeError(f"Unexpected nested directory in ImageFolder class: {path}")
        if entry.is_file(follow_symlinks=False) and path.suffix.lower() in IMAGE_SUFFIXES:
            files.append(path)
        elif not path.name.startswith("."):
            raise RuntimeError(f"Unexpected non-image file in class directory: {path}")
    return files


def validate_split(
    split_root: Path,
    split: str,
    expected_wnids: Sequence[str] | None,
    expected_val_wnids: Sequence[str] | None = None,
) -> dict[str, int]:
    if not split_root.is_dir():
        raise RuntimeError(f"Missing prepared split: {split_root}")
    root_entries = list(os.scandir(split_root))
    class_dirs = {
        entry.name: Path(entry.path)
        for entry in root_entries
        if entry.is_dir(follow_symlinks=False) and WNID_RE.fullmatch(entry.name)
    }
    unexpected = [
        entry.name
        for entry in root_entries
        if not (entry.is_dir(follow_symlinks=False) and WNID_RE.fullmatch(entry.name))
        and not entry.name.startswith(".")
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected entries under {split_root}: {unexpected[:10]}")
    if len(class_dirs) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"{split_root} has {len(class_dirs)} class directories, "
            f"expected {EXPECTED_CLASSES}"
        )
    if expected_wnids is not None and set(class_dirs) != set(expected_wnids):
        missing = sorted(set(expected_wnids) - set(class_dirs))
        extra = sorted(set(class_dirs) - set(expected_wnids))
        raise RuntimeError(
            f"{split} WNID mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )

    per_class: dict[str, int] = {}
    total = 0
    for wnid, class_dir in sorted(class_dirs.items()):
        files = _image_files(class_dir)
        if split == "train":
            bad_names = [path.name for path in files if not path.name.startswith(f"{wnid}_")]
            if bad_names:
                raise RuntimeError(
                    f"Training images under {wnid} have inconsistent names: {bad_names[:5]}"
                )
        else:
            expected_per_class = EXPECTED_IMAGES["val"] // EXPECTED_CLASSES
            if len(files) != expected_per_class:
                raise RuntimeError(
                    f"Validation class {wnid} has {len(files)} images, "
                    f"expected {expected_per_class}"
                )
        per_class[wnid] = len(files)
        total += len(files)

    if total != EXPECTED_IMAGES[split]:
        raise RuntimeError(
            f"{split_root} has {total} images, expected {EXPECTED_IMAGES[split]}"
        )
    if split == "val" and expected_val_wnids is not None:
        if len(expected_val_wnids) != EXPECTED_IMAGES["val"]:
            raise RuntimeError(
                f"Expected {EXPECTED_IMAGES['val']} validation labels, "
                f"got {len(expected_val_wnids)}"
            )
        for index, wnid in enumerate(expected_val_wnids, start=1):
            image = split_root / wnid / f"ILSVRC2012_val_{index:08d}.JPEG"
            if not image.is_file():
                raise RuntimeError(
                    f"Validation label mismatch at index {index}: expected {image}"
                )
    _log(f"Validated {split}: {len(per_class)} classes, {total} images")
    return per_class


def _check_existing_split(
    output_dir: Path,
    split: str,
    expected_wnids: Sequence[str] | None,
    expected_val_wnids: Sequence[str] | None = None,
) -> bool:
    split_root = output_dir / split
    if not split_root.exists():
        return False
    if split_root.is_dir() and not any(split_root.iterdir()):
        _log(f"Existing {split} directory is empty; using it as a fresh destination")
        return False
    try:
        validate_split(
            split_root,
            split,
            expected_wnids,
            expected_val_wnids=expected_val_wnids,
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"Refusing to modify an existing incomplete or invalid split: {split_root}\n"
            "Move/remove it explicitly, or choose a new --output-dir.\n"
            f"Validation error: {error}"
        ) from error
    _log(f"Existing {split} split is complete; skipping extraction")
    return True


def _extract_train_outer(archive: Path, class_tar_root: Path) -> list[str]:
    class_tar_root.mkdir(parents=True, exist_ok=True)
    marker = class_tar_root / ".outer-extraction-complete.json"
    if marker.is_file():
        wnids = json.loads(marker.read_text())["wnids"]
        if len(wnids) != EXPECTED_CLASSES:
            raise RuntimeError(f"Invalid train staging marker: {marker}")
        return list(wnids)

    _log(f"Extracting the {EXPECTED_CLASSES} nested class archives from the train tar ...")
    with tarfile.open(archive, "r:") as tar:
        members = [member for member in tar.getmembers() if member.isfile()]
        by_wnid: dict[str, tarfile.TarInfo] = {}
        for member in members:
            basename = Path(member.name).name
            if not basename.endswith(".tar"):
                raise RuntimeError(f"Unexpected train archive member: {member.name}")
            wnid = Path(basename).stem
            if not WNID_RE.fullmatch(wnid):
                raise RuntimeError(f"Invalid train class archive name: {member.name}")
            if wnid in by_wnid:
                raise RuntimeError(f"Duplicate train class archive: {wnid}")
            by_wnid[wnid] = member
        if len(by_wnid) != EXPECTED_CLASSES:
            raise RuntimeError(
                f"Train tar contains {len(by_wnid)} class archives, "
                f"expected {EXPECTED_CLASSES}"
            )
        for index, (wnid, member) in enumerate(sorted(by_wnid.items()), start=1):
            target = class_tar_root / f"{wnid}.tar"
            if not target.is_file() or target.stat().st_size != member.size:
                _copy_tar_member(tar, member, target)
            if index % 50 == 0 or index == EXPECTED_CLASSES:
                _log(f"Staged train class archives: {index}/{EXPECTED_CLASSES}")

    wnids = sorted(by_wnid)
    marker.write_text(json.dumps({"wnids": wnids}, indent=2) + "\n")
    return wnids


def _extract_one_train_class(class_tar: Path, build_root: Path, wnid: str) -> tuple[str, int, str]:
    target = build_root / wnid
    if target.is_dir():
        count = len(_image_files(target))
        class_tar.unlink(missing_ok=True)
        return wnid, count, "existing"
    if not class_tar.is_file():
        raise RuntimeError(f"Missing staged class archive for {wnid}: {class_tar}")

    temporary = Path(tempfile.mkdtemp(prefix=f".{wnid}.", dir=build_root))
    try:
        with tarfile.open(class_tar, "r:*") as tar:
            members = [member for member in tar.getmembers() if member.isfile()]
            if not members:
                raise RuntimeError(f"Empty train class archive: {class_tar}")
            seen: set[str] = set()
            for member in members:
                basename = Path(member.name).name
                if basename in seen or not basename.startswith(f"{wnid}_"):
                    raise RuntimeError(
                        f"Invalid or duplicate member {member.name} in {class_tar.name}"
                    )
                if Path(basename).suffix.lower() not in IMAGE_SUFFIXES:
                    raise RuntimeError(f"Unexpected non-image member: {member.name}")
                seen.add(basename)
                _copy_tar_member(tar, member, temporary / basename)
        os.replace(temporary, target)
        class_tar.unlink()
        return wnid, len(seen), "extracted"
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def prepare_train(
    archive: Path,
    output_dir: Path,
    staging_dir: Path,
    workers: int,
    expected_wnids: Sequence[str] | None,
) -> dict[str, int]:
    build_root = staging_dir / "train_build"
    class_tar_root = staging_dir / "train_class_archives"
    build_root.mkdir(parents=True, exist_ok=True)
    archive_wnids = _extract_train_outer(archive, class_tar_root)
    if expected_wnids is not None and set(archive_wnids) != set(expected_wnids):
        raise RuntimeError("Train archive classes do not match ImageNet devkit leaf WNIDs")

    _log(f"Expanding train classes with {workers} worker(s) ...")
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _extract_one_train_class,
                class_tar_root / f"{wnid}.tar",
                build_root,
                wnid,
            ): wnid
            for wnid in archive_wnids
        }
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 25 == 0 or completed == len(futures):
                _log(f"Expanded train classes: {completed}/{len(futures)}")

    counts = validate_split(build_root, "train", expected_wnids)
    target = output_dir / "train"
    if target.exists():
        target.rmdir()  # Only an explicitly empty directory is safe to replace.
    os.replace(build_root, target)
    shutil.rmtree(class_tar_root, ignore_errors=True)
    return counts


def prepare_val(
    archive: Path,
    output_dir: Path,
    staging_dir: Path,
    metadata: DevkitMetadata,
) -> dict[str, int]:
    build_root = staging_dir / "val_build"
    build_root.mkdir(parents=True, exist_ok=True)
    for wnid in metadata.wnids:
        (build_root / wnid).mkdir(exist_ok=True)

    _log("Extracting and organizing validation images from official devkit labels ...")
    with tarfile.open(archive, "r:") as tar:
        members = sorted(
            (member for member in tar.getmembers() if member.isfile()),
            key=lambda member: Path(member.name).name,
        )
        if len(members) != EXPECTED_IMAGES["val"]:
            raise RuntimeError(
                f"Validation tar contains {len(members)} images, "
                f"expected {EXPECTED_IMAGES['val']}"
            )
        for index, (member, wnid) in enumerate(
            zip(members, metadata.val_wnids), start=1
        ):
            basename = Path(member.name).name
            if not VAL_NAME_RE.fullmatch(basename):
                raise RuntimeError(f"Unexpected validation image name: {member.name}")
            target = build_root / wnid / basename
            if not target.is_file() or target.stat().st_size != member.size:
                _copy_tar_member(tar, member, target)
            if index % 5000 == 0 or index == len(members):
                _log(f"Extracted validation images: {index}/{len(members)}")

    counts = validate_split(
        build_root,
        "val",
        metadata.wnids,
        expected_val_wnids=metadata.val_wnids,
    )
    target = output_dir / "val"
    if target.exists():
        target.rmdir()
    os.replace(build_root, target)
    return counts


def _write_manifest(
    output_dir: Path,
    metadata: DevkitMetadata,
    split_counts: dict[str, dict[str, int]],
) -> None:
    class_to_idx = {wnid: index for index, wnid in enumerate(metadata.wnids)}
    manifest = {
        "format": "ImageNet-1K raw ImageFolder",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive_preparation": "extract and organize only; no crop, resize, normalization, or flip",
        "training_geometry": {
            "256": "ADM/Dhariwal center_crop_arr(image, 256)",
            "512": "EDM2 center-crop-dhariwal == center_crop_arr(image, 512)",
            "horizontal_flip_probability": 0.0,
        },
        "class_to_idx": class_to_idx,
        "splits": {
            split: {
                "classes": len(counts),
                "images": sum(counts.values()),
            }
            for split, counts in split_counts.items()
        },
        "official_archives": {
            split: {"filename": filename, "md5": md5}
            for split, (filename, md5) in ARCHIVES.items()
        },
    }
    path = output_dir / "imagenet_prepare_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _log(f"Wrote manifest: {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert official ILSVRC2012 archives to raw train/val ImageFolder layout."
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        required=True,
        help="Directory containing official train/val/devkit archives or an extracted devkit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination root that will contain train/ and val/.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val"),
        default=("train", "val"),
        help="Splits to prepare (default: train val).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel workers for expanding the 1000 train class archives.",
    )
    parser.add_argument(
        "--no-verify-md5",
        action="store_true",
        help="Skip official archive MD5 verification (not recommended).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate existing train/val directories without extracting archives.",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep .prepare_imagenet after successful completion.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    archive_dir = args.archive_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    splits = list(dict.fromkeys(args.splits))
    if args.workers < 1:
        raise ValueError(f"--workers must be positive, got {args.workers}")
    if not archive_dir.is_dir():
        raise FileNotFoundError(f"Archive directory does not exist: {archive_dir}")

    if args.check_only:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Prepared dataset root does not exist: {output_dir}")
        # Keep check-only genuinely read-only. If only a devkit tar is available,
        # extract its two metadata files into an OS temporary directory.
        with tempfile.TemporaryDirectory(prefix="prepare-imagenet-devkit-") as tmp:
            metadata = _load_devkit(
                archive_dir=archive_dir,
                output_dir=output_dir,
                staging_dir=Path(tmp),
                verify_md5=not args.no_verify_md5,
            )
            for split in splits:
                validate_split(
                    output_dir / split,
                    split,
                    metadata.wnids,
                    expected_val_wnids=(
                        metadata.val_wnids if split == "val" else None
                    ),
                )
        _log("Check completed successfully; no dataset files were changed")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir / ".prepare_imagenet"
    staging_dir.mkdir(exist_ok=True)
    metadata = _load_devkit(
        archive_dir=archive_dir,
        output_dir=output_dir,
        staging_dir=staging_dir / "devkit",
        verify_md5=not args.no_verify_md5,
    )

    split_counts: dict[str, dict[str, int]] = {}
    for split in splits:
        val_wnids = metadata.val_wnids if split == "val" else None
        if _check_existing_split(
            output_dir,
            split,
            metadata.wnids,
            expected_val_wnids=val_wnids,
        ):
            split_counts[split] = validate_split(
                output_dir / split,
                split,
                metadata.wnids,
                expected_val_wnids=val_wnids,
            )
            continue
        archive = _require_archive(
            archive_dir, split, verify_md5=not args.no_verify_md5
        )
        if split == "train":
            split_counts[split] = prepare_train(
                archive=archive,
                output_dir=output_dir,
                staging_dir=staging_dir,
                workers=args.workers,
                expected_wnids=metadata.wnids,
            )
        else:
            split_counts[split] = prepare_val(
                archive=archive,
                output_dir=output_dir,
                staging_dir=staging_dir,
                metadata=metadata,
            )

    _write_manifest(output_dir, metadata, split_counts)
    if not args.keep_staging:
        shutil.rmtree(staging_dir, ignore_errors=True)
    _log("ImageNet preparation completed successfully")


if __name__ == "__main__":
    main()
