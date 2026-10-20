"""Heun-50 EMA1 sampling with the exact trained JiT-only architecture.

Separate entrypoint preserves the source pinned by active Flow-100 queues.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import yaml

from main_rmsnorm2_jit_only import get_args_parser, parse_args, reference
from rmsnorm2_reference_loader import PROJECT_ROOT, verify_reference
from rmsnorm2_jit_only import RMSNorm2JitOnlyDenoiser


def prepare(options):
    checkpoint = options.checkpoint.resolve()
    output = options.output_dir.resolve()
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    saved = state["args"]
    saved = saved if isinstance(saved, dict) else vars(saved)
    if state["epoch"] != 80 or saved.get("dino_generation_path_enabled") is not False:
        raise ValueError("Expected the completed epoch-80 JiT-only checkpoint")
    if saved["model"] != "JiT-B/16" or saved["ema_decay1"] != 0.9999:
        raise ValueError("Unexpected architecture or EMA decay")
    if not 0 <= options.interval_min <= options.interval_max <= 1:
        raise ValueError("Invalid CFG interval")
    if not math.isfinite(options.cfg) or options.cfg < 0:
        raise ValueError("CFG scale must be finite and nonnegative")
    parser = get_args_parser()
    known = {a.dest for a in parser._actions}
    config = {k: v for k, v in saved.items() if k in known}
    for k in ("config", "world_size", "local_rank", "dist_url", "dist_on_itp", "start_epoch"):
        config.pop(k, None)
    config.update(evaluate_gen=True, online_eval=False, preflight=False,
                  skip_online_eval=True, keep_eval_images=True, resume_eval_images=True,
                  save_eval_npz=str(output / "samples.npz"), gen_checkpoint_path=str(checkpoint),
                  resume="", output_dir=str(output), exp_name=output.name, num_images=50000,
                  gen_bsz=256, cfg=options.cfg, sampling_method="heun",
                  num_sampling_steps=50, interval_min=options.interval_min,
                  interval_max=options.interval_max)
    # Preflight writes once; ranks only read this immutable configuration.
    path = output / "sampling_config.yaml"
    if options.preflight:
        output.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if yaml.safe_load(path.read_text()) != config:
                raise ValueError(f"Existing sampling configuration differs: {path}")
        else:
            path.write_text(yaml.safe_dump(config, sort_keys=False))
    if not path.is_file() or yaml.safe_load(path.read_text()) != config:
        raise ValueError("Run --preflight before generation")
    args = parse_args(["--config", str(path)])
    if options.preflight:
        from evaluation.assets import check_assets
        assets = check_assets(["fid", "inception_score", "fdr6"],
                              cache_root=os.environ["NANOGEN_EVALS_ROOT"])
        if not assets["ready"]:
            raise FileNotFoundError("Offline evaluation assets missing")
        # Strict CPU loading verifies both the trained model and EMA architecture
        # without touching GPUs occupied by the current training run.
        model = RMSNorm2JitOnlyDenoiser(args)
        model.load_state_dict(state["model"], strict=True)
        model.load_state_dict(state["model_ema1"], strict=True)
        assert model.dino_student is None and model.dino_input_projector is None
        assert all(name in state["model_ema2"] for name, _ in model.named_parameters())
        report = dict(reference=verify_reference(), checkpoint=str(checkpoint), epoch=80,
                      generation_dino_forward=False, strict_model_and_ema1_load=True,
                      trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      sampler=args.sampling_method, steps=args.num_sampling_steps,
                      cfg=args.cfg, interval=[args.interval_min, args.interval_max],
                      num_images=args.num_images, gen_bsz=args.gen_bsz, ema="ema1",
                      ema_decay=args.ema_decay1)
        (output / "model_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
        (output / "evaluation_assets.json").write_text(json.dumps(assets, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cfg", type=float, default=2.9)
    parser.add_argument("--interval-min", type=float, default=0.1)
    parser.add_argument("--interval-max", type=float, default=1.0)
    parser.add_argument("--preflight", action="store_true")
    options = parser.parse_args()
    args = prepare(options)
    if options.preflight:
        return
    reference.main_jit.Denoiser = RMSNorm2JitOnlyDenoiser
    os.chdir(PROJECT_ROOT)
    reference.main_jit.main(args)


if __name__ == "__main__":
    main()
