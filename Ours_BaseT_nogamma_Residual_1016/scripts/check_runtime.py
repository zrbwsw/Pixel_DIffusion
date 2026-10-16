"""Read-only resource/configuration checks; never instantiate a model or DDP."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "jit"):
    sys.path.insert(0, str(path))

from main_dino_jit import add_dino_jit_args, main_jit
from jit.config_utils import parse_args_with_config
from dino_v3_lora import _resolve_repo, _resolve_weights
from jit.engine_jit import online_evaluation_assets


def training_args(argv):
    return parse_args_with_config(add_dino_jit_args(main_jit.get_args_parser()), argv)


def inspect_runtime(args, world_size=1, sampling=False):
    if world_size < 1:
        raise ValueError("world_size must be positive")
    weights = _resolve_weights(args.dino_model_name, args.dino_weights)
    if weights.stat().st_size == 0:
        raise ValueError(f"Empty DINO weights: {weights}")
    report = {
        "model": args.model,
        "dino_model": args.dino_model_name,
        "dino_source": str(_resolve_repo()),
        "dino_weights": str(weights),
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "world_size": world_size,
    }
    if sampling:
        return report
    if args.evaluate_gen:
        raise ValueError("Use sample_portable.sh for generation, not train_portable.sh")
    data = Path(args.data_path).expanduser().resolve()
    if not (data / "train").is_dir() and not (data / "imagenet-latents-images").is_dir():
        raise FileNotFoundError(f"ImageNet train/ or imagenet-latents-images/ not found under {data}")
    if args.batch_size < 1 or args.accum_iter < 1 or args.global_batch_size < 0:
        raise ValueError("Batch size and accumulation must be positive; global batch must be nonnegative")
    local_global_batch = args.batch_size * world_size
    accumulation = args.accum_iter
    if args.global_batch_size:
        if args.global_batch_size < local_global_batch or args.global_batch_size % local_global_batch:
            raise ValueError("global_batch_size must be a positive multiple of batch_size * world_size")
        accumulation = args.global_batch_size // local_global_batch
    checkpoint = Path(args.resume).expanduser() / "checkpoint-last.pth" if args.resume else None
    if checkpoint is not None and checkpoint.exists() and checkpoint.stat().st_size == 0:
        raise ValueError(f"Empty resume checkpoint: {checkpoint}")
    report.update(
        data_path=str(data), batch_size_per_gpu=args.batch_size,
        accum_iter=accumulation, global_batch_size=local_global_batch * accumulation,
        resume_checkpoint=str(checkpoint.resolve()) if checkpoint is not None and checkpoint.is_file() else None,
        training_mode="resume" if checkpoint is not None and checkpoint.is_file() else "from_scratch",
    )
    if args.online_eval:
        statistics, inception = online_evaluation_assets(args.img_size)
        report["online_evaluation"] = {"statistics": statistics, "weights": inception}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--sampling", action="store_true")
    options, remaining = parser.parse_known_args()
    try:
        report = inspect_runtime(training_args(remaining), options.world_size, options.sampling)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
