#!/usr/bin/env python3
"""Exercise the optional idea through original iREPA DDP/train_one_epoch.

Example: CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc-per-node=1 \
  scripts/smoke_rmsnorm2.py --output tests_results/rmsnorm2_smoke.json

Runs two synthetic uint8 batches of one image through full JiT-B/16 at 256px,
real pretrained DINOv1 alignment, and real DINOv3+LoRA+PgT input. The original
training loop owns BF16, losses, backward, AdamW, LR warmup and both EMA updates.
This checks connectivity, not equivalence to a prior idea model or its FID.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/rmsnorm2_basejit_irepa_dinov1_b80_gbs1024.yaml"


class ScalarRecorder:
    """The original engine's SummaryWriter interface, held only in memory."""

    log_dir = "synthetic-two-batch-smoke"

    def __init__(self):
        self.values = {}

    def add_scalar(self, tag, value, step):
        value = float(value)
        if not math.isfinite(value):
            raise AssertionError(f"Nonfinite scalar: {tag}={value}")
        self.values.setdefault(tag, []).append({"step": int(step), "value": value})


def parameter_category(name):
    if "generation_input.dino." in name:
        return "generation_lora" if ".lora_" in name else "generation_patch_embed"
    if "generation_input.projector." in name:
        return "generation_pgt"
    if "generation_input.patch_norm." in name:
        return "patch_rmsnorm"
    if "generation_input.dino_norm." in name:
        return "dino_rmsnorm"
    if "net.projectors." in name:
        return "alignment_projector"
    return "native_jit"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "tests_results/rmsnorm2_smoke.json")
    parser.add_argument("--execution", choices=("compiled", "eager"), default="compiled")
    options = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != 1 or "LOCAL_RANK" not in os.environ:
        raise SystemExit("Launch with torchrun --standalone --nproc-per-node=1")
    os.environ["TORCHDYNAMO_DISABLE"] = "1" if options.execution == "eager" else "0"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "jit"))

    import numpy as np
    import torch
    from torch.nn.parallel import DistributedDataParallel
    from main_basejit_irepa import parse_args, prepare_runtime_assets
    from generation_input import build_denoiser
    from engine_jit import train_one_epoch
    from spnorm import SpatialNormalization
    from vision_encoder import load_encoders
    import util.misc as misc

    started = time.time()
    config_hash = hashlib.sha256(options.config.read_bytes()).hexdigest()
    args = parse_args(["--config", str(options.config.resolve())])
    if (args.generation_input_mode, args.model, args.img_size, args.enc_type) != (
        "rmsnorm2", "JiT-B/16", 256, "dino-vit-b"
    ):
        raise ValueError("This smoke test requires rmsnorm2, JiT-B/16, 256px and DINOv1-B/16")
    if args.expected_global_batch_size != 1024:
        raise ValueError("Expected the 1024-global-batch idea configuration")
    configured_batch_size = args.batch_size
    configured_global_batch = args.expected_global_batch_size
    configured_lr = args.lr if args.lr is not None else args.blr * configured_global_batch / 256
    # Test-process overrides only; the YAML and production launcher stay intact.
    args.batch_size = 1
    args.log_freq = 1
    args.lr = configured_lr
    assets = prepare_runtime_assets(args)
    misc.init_distributed_mode(args)
    try:
        torch.set_num_threads(2)
        torch.manual_seed(args.seed + misc.get_rank())
        np.random.seed(args.seed + misc.get_rank())
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
        torch._dynamo.config.cache_size_limit = 128
        torch._dynamo.config.optimize_ddp = False
        device = torch.device("cuda", args.gpu)

        # Same construction order as the production/original training entrypoint.
        encoders = load_encoders(args.enc_type, device, args.img_size)
        args.z_dims = [encoder.embed_dim for encoder in encoders]
        model = build_denoiser(args).to(device)
        adapter = model.net.generation_input
        if adapter.dino.num_lora_modules != 48:
            raise AssertionError(f"Expected 48 LoRA modules; got {adapter.dino.num_lora_modules}")
        if adapter.patch_norm.weight.data_ptr() == adapter.dino_norm.weight.data_ptr():
            raise AssertionError("The two RMSNorm gains share storage")
        for norm in (adapter.patch_norm, adapter.dino_norm):
            if not norm.weight.requires_grad or not torch.equal(norm.weight, torch.ones_like(norm.weight)):
                raise AssertionError("Expected independent trainable RMSNorm gains initialized to one")
        if not adapter.dino.model.norm.elementwise_affine:
            raise AssertionError("rmsnorm2 generation DINO should retain pretrained final LN affine")
        if any(parameter.requires_grad for parameter in adapter.dino.model.norm.parameters()):
            raise AssertionError("Generation DINO final LN affine should remain frozen")
        if not encoders[0].model.norm.elementwise_affine:
            raise AssertionError("Original alignment DINOv1 should retain final LN affine")

        ddp = DistributedDataParallel(model, device_ids=[args.gpu])
        groups = misc.add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))
        optimized = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
        trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        if set(optimized) != trainable or len(optimized) != len(trainable):
            raise AssertionError("Optimizer must contain every trainable parameter exactly once")
        model.ema_params1 = copy.deepcopy(list(model.parameters()))
        model.ema_params2 = copy.deepcopy(list(model.parameters()))
        output_records, gradient_records = [], []

        def check_output(module, inputs, output):
            image, projected, original = output
            expected_image = (1, 3, 256, 256)
            expected_tokens = (1, 256, 768)
            if tuple(image.shape) != expected_image:
                raise AssertionError(f"Unexpected predicted image shape {tuple(image.shape)}")
            if tuple(projected[0].shape) != expected_tokens or tuple(original[0].shape) != expected_tokens:
                raise AssertionError("Unexpected alignment token shape")
            if not torch.isfinite(image).all().item() or not torch.isfinite(projected[0]).all().item():
                raise AssertionError("Nonfinite predicted images or projected tokens")
            if adapter._pending_timestep is not None:
                raise AssertionError("Generation hook left a timestep reference after forward")
            output_records.append({
                "image_shape": list(image.shape), "image_dtype": str(image.dtype),
                "alignment_tokens_shape": list(projected[0].shape),
                "alignment_tokens_dtype": str(projected[0].dtype),
                "pending_timestep_cleared": True,
            })

        def check_gradients(optimizer, positional_args, keyword_args):
            counts = {}
            missing, nonfinite = [], []
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                bucket = counts.setdefault(parameter_category(name), {
                    "parameters": 0, "elements": 0, "gradients_present": 0,
                    "nonzero_gradients": 0,
                })
                bucket["parameters"] += 1
                bucket["elements"] += parameter.numel()
                if parameter.grad is None:
                    missing.append(name)
                    continue
                bucket["gradients_present"] += 1
                if not torch.isfinite(parameter.grad).all().item():
                    nonfinite.append(name)
                if torch.count_nonzero(parameter.grad).item():
                    bucket["nonzero_gradients"] += 1
            if missing or nonfinite:
                raise AssertionError(f"Missing gradients: {missing}; nonfinite gradients: {nonfinite}")
            required = {"generation_lora", "generation_patch_embed", "generation_pgt",
                        "patch_rmsnorm", "dino_rmsnorm", "alignment_projector", "native_jit"}
            if set(counts) != required:
                raise AssertionError(f"Missing/extra trainable parameter categories: {set(counts)}")
            # Zero A gradients and zero-gated block gradients are normal at init.
            # Every subsystem must nevertheless have a live nonzero gradient.
            if any(counts[name]["nonzero_gradients"] == 0 for name in required):
                raise AssertionError(f"A required subsystem has no nonzero gradients: {counts}")
            gradient_records.append({"lr": optimizer.param_groups[0]["lr"], "groups": counts})

        output_handle = model.net.register_forward_hook(check_output)
        optimizer_handle = optimizer.register_step_post_hook(check_gradients)
        # Build deterministic raw images without consuming the CUDA RNG stream.
        base = torch.arange(3 * 256 * 256, dtype=torch.int64).reshape(1, 3, 256, 256)
        batches = [
            (((base * 73 + 19 + index * 31) % 256).to(torch.uint8), torch.tensor([index]))
            for index in range(2)
        ]
        writer = ScalarRecorder()
        repa_kwargs = {
            "encoders": encoders, "spnorm": SpatialNormalization(args.spnorm_method),
            "cls_token_weight": args.cls_token_weight, "zscore_alpha": args.zscore_alpha,
            "zscore_proj_skip_std": args.zscore_proj_skip_std,
        }
        train_one_epoch(ddp, model, batches, optimizer, device, 0, repa_kwargs,
                        log_writer=writer, args=args)
        output_handle.remove()
        optimizer_handle.remove()
        if len(output_records) != 2 or len(gradient_records) != 2:
            raise AssertionError("Expected exactly two original-engine training forwards/optimizer updates")
        if len(writer.values.get("train_loss/total_loss", [])) != 2:
            raise AssertionError("Expected two finite total-loss records")
        if adapter._pending_timestep is not None:
            raise AssertionError("Stale generation timestep after training")
        alignment_gradients = sum(parameter.grad is not None for parameter in encoders[0].model.parameters())
        if alignment_gradients:
            raise AssertionError("The independent alignment encoder received gradients")
        if any(int(optimizer.state[parameter]["step"].item()) != 2
               for parameter in model.parameters() if parameter.requires_grad):
            raise AssertionError("A trainable parameter missed an AdamW update")
        head_index = next(index for index, (name, _) in enumerate(model.named_parameters())
                          if name == "net.final_layer.linear.weight")
        head = model.net.final_layer.linear.weight.detach()
        # Init head and the first LR are zero, so the first EMA remains zero.
        # After the second step, the original EMA must equal (1-decay)*head.
        for decay, ema in ((args.ema_decay1, model.ema_params1), (args.ema_decay2, model.ema_params2)):
            torch.testing.assert_close(ema[head_index], head * (1 - decay), rtol=1e-6, atol=1e-12)
            if not torch.isfinite(ema[head_index]).all().item():
                raise AssertionError("Nonfinite EMA")
        if hashlib.sha256(options.config.read_bytes()).hexdigest() != config_hash:
            raise AssertionError("Training config unexpectedly changed during smoke test")
        report = {
            "passed": True, "execution": options.execution, "ddp_world_size": 1,
            "device": torch.cuda.get_device_name(args.gpu), "torch": torch.__version__,
            "config": str(options.config.resolve()), "config_sha256": config_hash,
            "configured_batch_size_per_gpu": configured_batch_size,
            "configured_global_batch_size": configured_global_batch,
            "configured_peak_lr": configured_lr,
            "test_batch_size_per_gpu": 1, "synthetic_batches": 2,
            "engine": "bundled original engine_jit.train_one_epoch",
            "generation_lora_modules": adapter.dino.num_lora_modules,
            "rmsnorm_gains_independent": True, "alignment_parameter_gradients": alignment_gradients,
            "outputs": output_records, "optimizer_steps": gradient_records,
            "scalars": writer.values, "ema_checked_against_update_equation": True,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(args.gpu),
            "elapsed_seconds": time.time() - started, "assets": assets,
            "limitations": [
                "Single-rank DDP connectivity test; no inter-GPU reduction test.",
                "Two deterministic synthetic images, not a training-quality or FID test.",
                "No claim of numeric equivalence between the idea branch and the previous best implementation.",
                "Zero-initialized head, LoRA B and modulation gates are preserved; some gradient tensors are zero.",
            ],
        }
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"passed": True, "output": str(options.output.resolve()),
                          "optimizer_steps": 2, "missing_gradients": 0,
                          "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
