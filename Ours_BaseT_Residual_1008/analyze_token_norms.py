#!/usr/bin/env python3
"""Measure JiT/DINO/projector token norms over fixed images and noise levels."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "jit"))
import main_dino_jit  # noqa: E402
from dino_jit_denoiser import DINOJiTDenoiser  # noqa: E402
from jit.util.crop import center_crop_arr  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = main_dino_jit.add_dino_jit_args(main_dino_jit.main_jit.get_args_parser())
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--t-points", type=int, default=21)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--fixed-input-only", action="store_true")
    return main_dino_jit.main_jit.parse_args_with_config(parser)


def load_images(data_path: str, count: int, seed: int, rank: int, world_size: int, hflip_prob: float) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = ImageFolder(str(Path(data_path) / "train"), transform=None)
    if len(dataset) < count:
        raise ValueError(f"ImageNet val has only {len(dataset)} images, need {count}")
    generator = np.random.default_rng(seed)
    indices = np.sort(generator.choice(len(dataset), size=count, replace=False))
    flip = generator.random(count) < hflip_prob
    images = []
    for sample_id in range(rank, count, world_size):
        index = indices[sample_id]
        image, _ = dataset[int(index)]
        image = center_crop_arr(image.convert("RGB"), 256)
        if flip[sample_id]:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)
        images.append(image)
    ids = torch.arange(count)[rank::world_size]
    return torch.stack(images).float().div(255.0).mul(2.0).sub(1.0), ids


def stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().reshape(-1).cpu()
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "p05": float(torch.quantile(values, 0.05)),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
        "count": int(values.numel()),
    }


def row_for(t: float, sample_ids: torch.Tensor, tensors: dict[str, torch.Tensor]) -> list[dict]:
    rows = []
    for name, value in tensors.items():
        per_sample = value.float().norm(dim=-1).mean(dim=-1)
        summary = stats(per_sample)
        summary.update({"t": t, "tensor": name})
        rows.append(summary)
    for i in range(sample_ids.numel()):
        item = {"sample_id": int(sample_ids[i]), "t": t}
        for name, value in tensors.items():
            item[f"{name}_norm"] = float(value[i].float().norm(dim=-1).mean())
        rows.append(item)
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = DINOJiTDenoiser(args).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    ema = checkpoint["model_ema1"]
    model.ema_params1 = [ema[name].to(device) for name, _ in model.named_parameters()]
    for parameter, ema_parameter in zip(model.parameters(), model.ema_params1):
        parameter.data.copy_(ema_parameter.data)
    del checkpoint, ema

    images, sample_ids = load_images(args.data_path, args.num_samples, args.seed, args.rank, args.world_size, args.hflip_prob)
    noises = torch.stack([
        torch.randn(images[i].shape, generator=torch.Generator().manual_seed(args.seed + 1 + int(sample_id)))
        for i, sample_id in enumerate(sample_ids)
    ]) * float(args.noise_scale)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, sample_rows = [], []
    t_values = torch.linspace(0.0, 1.0, args.t_points)

    if args.fixed_input_only:
        fixed_rows = []
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, len(images), args.batch_size):
                end = min(start + args.batch_size, len(images))
                x = images[start:end].to(device)
                noise = noises[start:end].to(device)
                z_ref = 0.5 * x + 0.5 * noise
                student = model._encode_noisy_student(
                    z_ref, torch.full((end - start,), 0.5, device=device)
                )
                raw_norms = student.float().norm(dim=-1).mean(dim=-1).cpu()
                for t_value in t_values.tolist():
                    tb = torch.full((end - start,), t_value, device=device)
                    projected = model.dino_input_projector(
                        student.to(dtype=model.dino_input_projector.input_proj.weight.dtype),
                        tb, (16, 16),
                    )
                    projected_norms = projected.float().norm(dim=-1).mean(dim=-1).cpu()
                    for i in range(end - start):
                        fixed_rows.append({
                            "sample_id": int(sample_ids[start + i]), "t": t_value,
                            "dino_raw_ref_norm": float(raw_norms[i]),
                            "dino_projected_fixed_z_norm": float(projected_norms[i]),
                        })
                if start % 64 == 0:
                    print(f"fixed-input completed {end}/{len(images)}", flush=True)
        fields = list(fixed_rows[0])
        with (output_dir / f"fixed_input_rank{args.rank}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(fixed_rows)
        return

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for t_value in t_values.tolist():
            for start in range(0, len(images), args.batch_size):
                end = min(start + args.batch_size, len(images))
                x = images[start:end].to(device, non_blocking=True)
                noise = noises[start:end].to(device, non_blocking=True)
                tb = torch.full((end - start,), t_value, device=device)
                z = tb[:, None, None, None] * x + (1.0 - tb[:, None, None, None]) * noise
                student = model._encode_noisy_student(z, tb)
                native = model.net.x_embedder(z)
                projected = model.dino_input_projector(
                    student.to(dtype=model.dino_input_projector.input_proj.weight.dtype),
                    tb,
                    (16, 16),
                )
                jit_weighted = (1.0 - tb[:, None, None]) * native
                dino_weighted = tb[:, None, None] * projected
                fused = jit_weighted + dino_weighted
                tensors = {
                    "dino_raw": student,
                    "jit_native": native,
                    "dino_projected": projected,
                    "jit_weighted": jit_weighted,
                    "dino_weighted": dino_weighted,
                    "fused": fused,
                }
                batch_rows = row_for(t_value, sample_ids[start:end], tensors)
                summary_rows.extend(row for row in batch_rows if "tensor" in row)
                sample_rows.extend(row for row in batch_rows if "sample_id" in row)
            print(f"completed t={t_value:.3f}", flush=True)

    def write_csv(path: Path, rows: list[dict]) -> None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_dir / f"token_norm_summary_rank{args.rank}.csv", summary_rows)
    write_csv(output_dir / f"token_norm_per_sample_rank{args.rank}.csv", sample_rows)
    (output_dir / f"metadata_rank{args.rank}.json").write_text(json.dumps({
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_samples": args.num_samples, "local_samples": len(images),
        "rank": args.rank, "world_size": args.world_size,
        "batch_size": args.batch_size,
        "t_points": args.t_points, "seed": args.seed,
        "noise": "fixed per sample across t", "weights": "model_ema1",
        "data_preprocess": f"ImageNet train center crop and fixed random horizontal flip p={args.hflip_prob}",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
