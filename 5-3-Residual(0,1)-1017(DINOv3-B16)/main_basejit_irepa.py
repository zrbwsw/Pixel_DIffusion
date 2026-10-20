"""Portable entrypoint for the unchanged iREPA trainer and optional input idea.

Only argument/resource plumbing and the model factory live here. Numerical
training, target extraction, losses, optimizer, EMA and sampling live in the
byte-identical bundled iREPA files, including its main() function.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
JIT_ROOT = PROJECT_ROOT / "jit"
sys.path.insert(0, str(JIT_ROOT))

import torch
import main_jit as original_main
from config_utils import parse_args_with_config
from generation_input import build_denoiser
from resource_paths import resource_root


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Original JiT+iREPA with optional input RMSNorm",
        parents=[original_main.get_args_parser()],
    )
    parser.set_defaults(data_path=None, report_to="tensorboard")
    parser.add_argument("--generation_input_mode", choices=["baseline", "jit_rmsnorm", "rmsnorm2"], default="baseline")
    parser.add_argument("--expected_global_batch_size", type=int, default=512)
    parser.add_argument("--dino_weights", default=None)
    parser.add_argument("--dino_lora_r", type=int, default=64)
    parser.add_argument("--dino_lora_alpha", type=float, default=64.0)
    parser.add_argument("--dino_lora_dropout", type=float, default=0.0)
    parser.add_argument("--dino_train_patch_embed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight", action="store_true", help="Check paths/config; do not construct models or train")
    return parser


def parse_args(argv=None):
    args = parse_args_with_config(get_args_parser(), argv)
    if args.data_path is None:
        imagenet = resource_root("IMAGENET_ROOT")
        if "IMAGENET_ROOT" not in os.environ and Path("/data0/datasets/imagenet/train").is_dir():
            imagenet = Path("/data0/datasets/imagenet")
        if imagenet.name != "imagenet":
            raise ValueError("Original iREPA expects <data_path>/imagenet/train; set --data_path to its parent")
        args.data_path = str(imagenet.parent)
    args.data_path = str(Path(args.data_path).expanduser().resolve())
    for key in ("output_dir", "resume"):
        value = getattr(args, key)
        if value:
            path = Path(value).expanduser()
            setattr(args, key, str((PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()))
    if not args.enable_repa:
        raise ValueError("This project's controls require --enable_repa")
    if args.enc_type not in {"dino-vit-b", "dinov3-vit-b16"}:
        raise ValueError("Use DINOv1-B/16 (dino-vit-b) or DINOv3-B/16 (dinov3-vit-b16) for alignment")
    if args.expected_global_batch_size <= 0 or args.batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.sampling_method not in ("heun", "euler"):
        raise ValueError("Use the original iREPA heun/euler sampler")
    return args


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _link_asset(source, destination):
    """Create a cache link without replacing any existing resource."""
    source, destination = Path(source).resolve(), Path(destination)
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source, target_is_directory=source.is_dir())
    except FileExistsError:
        if destination.exists() and destination.samefile(source):
            return
        if source.is_file() and destination.is_file() and _file_hash(source) == _file_hash(destination):
            return
        raise FileExistsError(f"Existing resource differs; resolve explicitly: {destination} -> {source}")


def prepare_runtime_assets(args):
    """Supply original relative paths/torch.hub cache; no encoder monkeypatch."""
    weights_root = resource_root("PRETRAIN_MODELS_ROOT")
    if "PRETRAIN_MODELS_ROOT" not in os.environ and Path("/data0/pretrain_models").is_dir():
        weights_root = Path("/data0/pretrain_models")
    fid_root = resource_root("FID_STATS_ROOT")
    hub = Path(os.environ.get("BASEJIT_TORCH_HUB", PROJECT_ROOT / ".cache/torch/hub")).expanduser().resolve()
    if args.enc_type == "dino-vit-b":
        dino_weight = weights_root / "dino_vitbase16_pretrain.pth"
        _link_asset(PROJECT_ROOT / "SSL_code/dino", hub / "facebookresearch_dino_main")
        _link_asset(dino_weight, hub / "checkpoints/dino_vitbase16_pretrain.pth")
    else:
        # Configure the unchanged upstream DINOv3 loader before it is imported.
        repo = Path(os.environ.get("DINOV3_REPO_DIR") or os.environ.get("DINOV3_REPO") or PROJECT_ROOT / "SSL_code/dinov3").expanduser().resolve()
        checkpoint_dir = Path(os.environ.get("DINOV3_CKPT_DIR") or weights_root).expanduser().resolve()
        dino_weight = checkpoint_dir / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        for required in (repo / "hubconf.py", dino_weight):
            if not required.is_file():
                raise FileNotFoundError(required)
        os.environ["DINOV3_REPO_DIR"] = str(repo)
        os.environ["DINOV3_CKPT_DIR"] = str(checkpoint_dir)
    # The unchanged torch-fidelity and original engine resolve these filenames.
    if args.online_eval or args.evaluate_gen:
        _link_asset(fid_root / f"jit_in{args.img_size}_stats.npz", JIT_ROOT / "fid_stats" / f"jit_in{args.img_size}_stats.npz")
        _link_asset(fid_root / "weights-inception-2015-12-05-6726825d.pth", hub / "checkpoints/weights-inception-2015-12-05-6726825d.pth")
    torch.hub.set_dir(str(hub))
    if args.generation_input_mode == "rmsnorm2":
        weights = Path(args.dino_weights).expanduser() if args.dino_weights else weights_root / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        if not weights.is_file():
            raise FileNotFoundError(weights)
        args.dino_weights = str(weights.resolve())
    return {"hub_dir": str(hub), "alignment_weights": str(dino_weight), "fid_stats_root": str(fid_root)}


def check_runtime(args):
    from scripts.verify_upstream import verify
    upstream = verify(PROJECT_ROOT)
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    global_batch = world_size * args.batch_size
    if global_batch != args.expected_global_batch_size:
        raise ValueError(f"Planned {world_size} x {args.batch_size} = {global_batch}, expected {args.expected_global_batch_size}; adjust GPUs or --batch_size")
    train_dir = Path(args.data_path) / "imagenet/train"
    hf_dir = Path(args.data_path) / "imagenet-latents-images"
    if not train_dir.is_dir():
        raise FileNotFoundError(train_dir)
    if hf_dir.exists():
        raise ValueError(f"Historical run used ImageFolder, but original iREPA would prefer {hf_dir}; use an isolated data parent containing only the imagenet directory")
    assets = prepare_runtime_assets(args)
    return dict(
        generation_input_mode=args.generation_input_mode,
        alignment_encoder=args.enc_type,
        world_size=world_size,
        batch_size_per_gpu=args.batch_size,
        global_batch_size=global_batch,
        effective_lr=args.lr if args.lr is not None else args.blr * global_batch / 256,
        epochs=args.epochs,
        output_dir=args.output_dir,
        resume=args.resume,
        upstream=upstream,
        train_dir=str(train_dir),
        sampling=dict(method=args.sampling_method, steps=args.num_sampling_steps, cfg=args.cfg,
                      interval=[args.interval_min, args.interval_max], gen_bsz=args.gen_bsz,
                      num_images=args.num_images, eval_freq=args.eval_freq),
        **assets,
    )


@contextmanager
def output_lock(args):
    """One rank-0 writer per output; never silently restart a nonempty run."""
    if int(os.environ.get("RANK", "0")) != 0:
        yield
        return
    output = Path(args.output_dir)
    lock_dir = PROJECT_ROOT / ".runtime_locks"
    lock_dir.mkdir(exist_ok=True)
    lock_name = hashlib.sha256(str(output).encode()).hexdigest() + ".lock"
    with (lock_dir / lock_name).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            resume = Path(args.resume)
            checkpoint = resume / "checkpoint-last.pth"
            if resume != output or not checkpoint.is_file():
                raise ValueError("Resume explicitly from this output directory's checkpoint-last.pth")
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            saved = state.get("args")
            if saved is None:
                raise ValueError("Checkpoint lacks configuration; compatibility must be audited before resuming")
            semantic_fields = (
                "generation_input_mode", "model", "img_size", "class_num", "attn_dropout", "proj_dropout",
                "enable_repa", "enc_type", "encoder_depth", "projector_dim", "projection_layer_type",
                "proj_kwargs_kernel_size", "projection_loss_type", "proj_coeff", "spnorm_method",
                "cls_token_weight", "zscore_alpha", "zscore_proj_skip_std", "label_drop_prob",
                "P_mean", "P_std", "t_eps", "noise_scale", "ema_decay1", "ema_decay2",
                "expected_global_batch_size", "blr", "min_lr", "lr_schedule", "warmup_epochs", "weight_decay",
            )
            if args.generation_input_mode == "rmsnorm2":
                semantic_fields += ("dino_lora_r", "dino_lora_alpha", "dino_lora_dropout", "dino_train_patch_embed")
            for name in semantic_fields:
                if getattr(saved, name, None) != getattr(args, name):
                    raise ValueError(f"Resume configuration mismatch: {name}")
            expected_lr = args.lr if args.lr is not None else args.blr * args.expected_global_batch_size / 256
            if getattr(saved, "lr", None) != expected_lr:
                raise ValueError("Resume configuration mismatch: effective learning rate")
            del state
        elif output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output is nonempty; choose a new --output_dir or explicitly --resume: {output}")
        output.mkdir(parents=True, exist_ok=True)
        yield


def main(argv=None):
    args = parse_args(argv)
    report = check_runtime(args)
    if args.preflight:
        print(json.dumps(report, indent=2))
        return
    # The original main owns seed setting, encoder -> model construction,
    # DataLoader, DDP, optimizer, EMA, train_one_epoch and online evaluation.
    # Baseline's factory returns the original class without any extra work.
    original_main.Denoiser = build_denoiser
    with output_lock(args):
        os.chdir(JIT_ROOT)
        original_main.main(args)


if __name__ == "__main__":
    main()
