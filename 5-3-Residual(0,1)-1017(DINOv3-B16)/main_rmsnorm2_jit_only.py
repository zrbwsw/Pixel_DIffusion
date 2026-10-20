"""Single-factor Residual(0,1) control of the completed rmsnorm2 B-80 run."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path

from rmsnorm2_reference_loader import PROJECT_ROOT, REFERENCE_ROOT, load_reference, verify_reference

reference = load_reference()


def get_args_parser():
    parser = reference.add_dino_jit_args(reference.main_jit.get_args_parser())
    parser.add_argument("--dino_generation_path_enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--jit_input_rmsnorm", action=argparse.BooleanOptionalAction, default=True,
                        help="Extra learned RMSNorm after native patch embedding, before APE; internal JiT norms are unchanged")
    parser.add_argument("--preflight", action="store_true")
    return parser


def parse_args(argv=None):
    return reference.main_jit.parse_args_with_config(get_args_parser(), argv)


def runtime_world_size():
    world_size = int(os.environ.get("WORLD_SIZE", "4"))
    if world_size not in (4, 8):
        raise ValueError("This matched recipe requires 4 x 256 or 8 x 128, global batch 1024")
    return world_size


def verify_recipe(args):
    baseline = parse_args([
        "--config", str(REFERENCE_ROOT / "configs/baseline_B80.yaml"),
        "--batch_size", str(1024 // runtime_world_size()), "--dino_generation_path_enabled",
    ])
    allowed = {"config", "exp_name", "output_dir", "resume", "metrics_log_name", "preflight",
               "dino_generation_path_enabled", "jit_input_rmsnorm", "epochs"}
    differences = {key: [getattr(baseline, key), getattr(args, key)]
                   for key in vars(baseline) if getattr(baseline, key) != getattr(args, key)}
    unexpected = set(differences) - allowed
    if unexpected:
        raise ValueError(f"Single-factor recipe differs from rmsnorm2 baseline: {sorted(unexpected)}")
    if args.epochs != baseline.epochs and (not args.resume or args.epochs < baseline.epochs):
        raise ValueError("Changing the training horizon requires continuation from a matched checkpoint")
    if args.dino_generation_path_enabled:
        raise ValueError("This experiment requires --no-dino_generation_path_enabled")
    return differences


def check_runtime(args):
    world_size = runtime_world_size()
    if (args.batch_size != 1024 // world_size or args.accum_iter != 1
            or args.global_batch_size != 1024):
        raise ValueError("This matched recipe requires 4 x 256 or 8 x 128, accum_iter=1, global batch 1024")
    differences = verify_recipe(args)
    from dino_v3_lora import _resolve_repo, _resolve_weights
    from engine_jit import online_evaluation_assets
    data = Path(args.data_path).expanduser().resolve()
    if not (data / "train").is_dir():
        raise FileNotFoundError(f"Baseline ImageFolder data missing: {data / 'train'}")
    output = Path(args.output_dir).expanduser()
    output = (PROJECT_ROOT / output).resolve() if not output.is_absolute() else output.resolve()
    checkpoint = None
    resume_state = None
    if args.resume:
        resume = Path(args.resume).expanduser()
        resume = (PROJECT_ROOT / resume).resolve() if not resume.is_absolute() else resume.resolve()
        if resume != output:
            checkpoint_root = (PROJECT_ROOT / "checkpoints").resolve()
            if resume.parent != checkpoint_root or output.parent != checkpoint_root:
                raise ValueError("A separate continuation output must share this project's checkpoint root")
            if output.name != args.exp_name:
                raise ValueError("Continuation output must match the requested experiment name")
            if output.exists() and any(output.glob("checkpoint-*.pth")):
                raise ValueError("Continuation output already has checkpoints; resume that directory explicitly")
        checkpoint = resume / "checkpoint-last.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        import torch
        state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        saved = state["args"]
        saved = saved if isinstance(saved, dict) else vars(saved)
        saved_world_size = int(saved.get("world_size", 0))
        if (saved_world_size not in (4, 8) or saved.get("accum_iter") != 1
                or saved.get("batch_size", 0) * saved_world_size != 1024):
            raise ValueError("Resume configuration mismatch: saved GPU/batch layout")
        if saved.get("exp_name") != resume.name or args.epochs < saved["epochs"]:
            raise ValueError("Resume experiment identity or training horizon mismatch")
        ignored = {"config", "exp_name", "epochs", "output_dir", "resume", "start_epoch", "preflight", "lr", "world_size",
                   "rank", "local_rank", "gpu", "distributed", "dist_backend", "dist_url", "batch_size"}
        for key, value in vars(args).items():
            # Checkpoints predating this ablation always used the input norm.
            saved_value = saved.get(key, True if key == "jit_input_rmsnorm" else None)
            if key not in ignored and saved_value != value:
                raise ValueError(f"Resume configuration mismatch: {key}")
        if not math.isclose(saved["lr"], 2e-4) or not 0 < state["epoch"] < args.epochs:
            raise ValueError("Invalid or already completed checkpoint")
        for key in ("model", "optimizer", "model_ema1", "model_ema2"):
            if not state.get(key):
                raise ValueError(f"Incomplete resume checkpoint: missing {key}")
        if not state["optimizer"].get("state") or not state["optimizer"].get("param_groups"):
            raise ValueError("Incomplete resume checkpoint: missing AdamW state")
        if not (set(state["model"]) == set(state["model_ema1"]) == set(state["model_ema2"])):
            raise ValueError("Incomplete resume checkpoint: model/EMA keys differ")
        resume_state = {"completed_epochs": state["epoch"], "saved_world_size": saved_world_size,
                        "saved_batch_size_per_gpu": saved["batch_size"],
                        "optimizer_state_entries": len(state["optimizer"]["state"]),
                        "restores": ["model", "optimizer", "model_ema1", "model_ema2"]}
    weights = _resolve_weights(args.dino_model_name, args.dino_weights)
    statistics, inception = online_evaluation_assets(args.img_size)
    return {
        "source": verify_reference(), "allowed_config_differences": differences,
        "world_size": world_size, "batch_size_per_gpu": args.batch_size,
        "global_batch_size": args.global_batch_size, "accum_iter": args.accum_iter,
        "effective_lr": args.lr if args.lr is not None else args.blr * args.global_batch_size / 256,
        "epochs": args.epochs, "warmup_epochs": args.warmup_epochs,
        "data_path": str(data), "dino_source": str(_resolve_repo()), "dino_weights": str(weights),
        "output_dir": str(output), "resume_checkpoint": str(checkpoint) if checkpoint else None,
        "resume_state": resume_state,
        "training_mode": "resume" if checkpoint else "from_scratch",
        "jit_input_rmsnorm": args.jit_input_rmsnorm,
        "generation_dino_forward": False,
        "sampling": {"method": args.sampling_method, "steps": args.num_sampling_steps,
                     "cfg": args.cfg, "interval": [args.interval_min, args.interval_max],
                     "images": args.num_images, "gen_bsz_per_gpu": args.gen_bsz,
                     "eval_freq": args.eval_freq, "ema": args.ema_decay1},
        "online_assets": {"statistics": statistics, "inception": inception},
    }


def main(argv=None):
    args = parse_args(argv)
    report = check_runtime(args)
    if args.preflight:
        print(json.dumps(report, indent=2))
        return
    from rmsnorm2_jit_only import RMSNorm2JitOnlyDenoiser
    # The dataset, training loop, teacher, objectives, optimizer, EMA and online
    # evaluation remain the byte-identical historical rmsnorm2 implementation.
    reference.main_jit.Denoiser = RMSNorm2JitOnlyDenoiser
    reference.main_jit.HFImageDataset = (
        lambda data_dir, split="train", transform=None: reference.FlexibleImageNet256Dataset(
            data_dir, split=split, transform=transform, image_size=args.img_size)
    )
    os.chdir(PROJECT_ROOT)
    if int(os.environ.get("RANK", "0")) == 0:
        output = Path(report["output_dir"])
        lock_dir = PROJECT_ROOT / ".runtime_locks"
        lock_dir.mkdir(exist_ok=True)
        # Keep the file handle alive for the duration of the training function.
        with (lock_dir / (output.name + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            existing = list(output.iterdir()) if output.exists() else []
            metadata = {"train.log", "launcher.log", "run_info.txt", "resource_preflight.json", "config_base.yaml",
                        "source_sha256.txt", "started_at.txt", "train_args.txt"}
            if not args.resume and any(p.name not in metadata for p in existing):
                raise FileExistsError(f"Output already has a run; resume explicitly: {output}")
            output.mkdir(parents=True, exist_ok=True)
            reference.main_jit.main(args)
    else:
        reference.main_jit.main(args)


if __name__ == "__main__":
    main()
