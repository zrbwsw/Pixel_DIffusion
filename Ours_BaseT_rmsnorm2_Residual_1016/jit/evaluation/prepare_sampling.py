"""Prepare a strict Ours sampling YAML from an explicit experiment configuration."""

import argparse
from pathlib import Path

import yaml


def prepare_config(config_path, checkpoint_path, output_dir, *, num_images=50000,
                   gen_bsz=128, cfg=2.5, sampler="flow_dpm_solver", steps=100,
                   interval=(0.0, 1.0), npz_path=None):
    # Ours uses a strict grouped YAML parser. Do not copy arbitrary checkpoint
    # metadata into it, or silently default unknown historical model options.
    from main_dino_jit import add_dino_jit_args, main_jit
    from jit.config_utils import load_grouped_yaml_defaults

    base_parser = main_jit.get_args_parser()
    parser = add_dino_jit_args(base_parser)
    config = load_grouped_yaml_defaults(parser, config_path)
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if checkpoint_path.is_dir():
        checkpoint_path /= "checkpoint-last.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_dir = Path(output_dir).expanduser().resolve()
    image_size = config.get("img_size", parser.get_default("img_size"))
    classes = config.get("class_num", parser.get_default("class_num"))
    if image_size != 256:
        raise ValueError("Bundled FDR6 statistics require ImageNet-256")
    if num_images < 2 or num_images % classes or gen_bsz < 1 or steps < 1:
        raise ValueError("Expected class-balanced num_images >=2, positive gen_bsz and steps")
    if not 0 <= interval[0] <= interval[1] <= 1:
        raise ValueError("CFG interval must satisfy 0 <= min <= max <= 1")
    npz_path = Path(npz_path).expanduser().resolve() if npz_path else output_dir / "samples.npz"
    if npz_path.suffix != ".npz":
        raise ValueError("NPZ destination must end in .npz")
    config.update(
        evaluate_gen=True, online_eval=False,
        skip_online_eval=True, keep_eval_images=True, resume_eval_images=True,
        save_eval_npz=str(npz_path), gen_checkpoint_path=str(checkpoint_path), resume="",
        output_dir=str(output_dir), exp_name=output_dir.name,
        num_images=num_images, gen_bsz=gen_bsz, cfg=cfg, sampling_method=sampler,
        num_sampling_steps=steps, interval_min=interval[0], interval_max=interval[1],
    )
    for key in ("config", "world_size", "local_rank", "dist_url", "dist_on_itp", "start_epoch"):
        config.pop(key, None)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="matching Ours experiment YAML")
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint file or directory with checkpoint-last.pth")
    parser.add_argument("--output-dir", type=Path, required=True, help="one directory per checkpoint and sampler setting")
    parser.add_argument("--npz", type=Path, default=None, help="defaults to output-dir/samples.npz, independent of PNG path")
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument("--gen-bsz", type=int, default=128)
    parser.add_argument("--cfg", type=float, default=2.5)
    parser.add_argument("--sampler", choices=("heun", "euler", "flow_dpm_solver"), default="flow_dpm_solver")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--interval-min", type=float, default=0.0)
    parser.add_argument("--interval-max", type=float, default=1.0)
    args = parser.parse_args()
    config = prepare_config(args.config, args.checkpoint, args.output_dir,
                            num_images=args.num_images, gen_bsz=args.gen_bsz, cfg=args.cfg,
                            sampler=args.sampler, steps=args.steps,
                            interval=(args.interval_min, args.interval_max), npz_path=args.npz)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "sampling_config.yaml"
    if path.exists():
        if yaml.safe_load(path.read_text()) != config:
            raise ValueError(f"Sampling config differs from {path}; use a separate output directory")
    else:
        with path.open("x") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
    print(f"Sampling configuration: {path.resolve()}")


if __name__ == "__main__":
    main()
