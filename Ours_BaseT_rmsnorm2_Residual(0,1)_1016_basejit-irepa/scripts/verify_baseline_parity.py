#!/usr/bin/env python3
"""Compare the baseline factory to iREPA in two isolated Python processes.

GPU example (one visible GPU, batch 1, no training jobs started):
  CUDA_VISIBLE_DEVICES=0 python scripts/verify_baseline_parity.py --mode cuda
CPU smoke test, with explicit test-only shims for upstream's hardcoded .cuda():
  CUDA_VISIBLE_DEVICES='' python scripts/verify_baseline_parity.py --mode cpu

The CUDA test uses genuine DINOv1-B/16 weights from an existing official torch
hub cache. It never downloads weights or substitutes the generation encoder.
Full tensors are hashed, not just sampled; activations are also saved for numeric
comparison. The nonzero-head probe modifies only the subprocess test model.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_CONFIG = "configs/basejit_irepa_dinov1_b20_gbs512.yaml"
CONTRACT_FIELDS = (
    "model", "img_size", "class_num", "attn_dropout", "proj_dropout",
    "enable_repa", "enc_type", "encoder_depth", "projector_dim",
    "projection_layer_type", "proj_kwargs_kernel_size", "projection_loss_type",
    "proj_coeff", "spnorm_method", "cls_token_weight", "zscore_alpha",
    "zscore_proj_skip_std", "label_drop_prob", "P_mean", "P_std", "t_eps",
    "noise_scale", "ema_decay1", "ema_decay2", "sampling_method",
    "num_sampling_steps", "cfg", "interval_min", "interval_max", "num_images",
    "gen_bsz", "epochs", "warmup_epochs", "batch_size", "blr", "min_lr",
    "lr_schedule", "weight_decay", "seed", "eval_freq", "online_eval",
)


def historical_args():
    """Literal historical settings, parsed by the selected upstream parser."""
    from main_jit import get_args_parser

    return get_args_parser().parse_args([
        "--model", "JiT-B/16", "--img_size", "256", "--enable_repa",
        "--enc_type", "dino-vit-b", "--encoder_depth", "4",
        "--projection_layer_type", "conv", "--proj_kwargs_kernel_size", "3",
        "--projection_loss_type", "cosine", "--proj_coeff", "0.5",
        "--spnorm_method", "zscore", "--zscore_alpha", "0.8",
        "--epochs", "20", "--warmup_epochs", "5", "--batch_size", "128",
        "--blr", "0.00005", "--weight_decay", "0", "--seed", "0",
        "--sampling_method", "heun", "--num_sampling_steps", "50",
        "--cfg", "2.9", "--interval_min", "0.1", "--interval_max", "1.0",
        "--num_images", "50000", "--gen_bsz", "256", "--eval_freq", "20",
        "--online_eval", "--report_to", "tensorboard",
    ])


def tensor_digest(tensor):
    import torch

    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(value.shape), "dtype": str(value.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def state_digest(items):
    """Hash every element of every tensor, including zero/frozen parameters."""
    return {name: None if value is None else tensor_digest(value) for name, value in items}


def require_cached_dino():
    import torch

    cache = Path(torch.hub.get_dir())
    required = (
        cache / "facebookresearch_dino_main" / "hubconf.py",
        cache / "checkpoints" / "dino_vitbase16_pretrain.pth",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Prepopulate the official DINO hub cache or pass --hub-dir; "
            "this parity test does not download models. Missing: " + ", ".join(missing)
        )
    return {str(path.relative_to(cache)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in required}


def run_worker(options):
    # Set before importing torch; both processes use the same execution policy.
    os.environ["TORCHDYNAMO_DISABLE"] = "1" if options.execution == "eager" else "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    root = Path(options.source_root).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "jit"))

    import copy
    import numpy as np
    import torch
    import model_jit
    from denoiser import Denoiser
    from spnorm import SpatialNormalization
    from vision_encoder import DINOEncoder, load_encoders
    import util.misc as misc
    import util.lr_sched as lr_sched

    torch.set_num_threads(2)
    device = torch.device("cuda" if options.mode == "cuda" else "cpu")
    if options.hub_dir:
        torch.hub.set_dir(options.hub_dir)
    if device.type == "cuda":
        torch.cuda.set_device(0)
        # A/B determinism; production cudnn.benchmark=True is not claimed here.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
        torch.use_deterministic_algorithms(True)
    else:
        # Upstream hardcodes .cuda() in RoPE and attention. This is a CPU-only
        # harness adaptation applied identically to both independent processes.
        torch.Tensor.cuda = lambda self, *a, **kw: self

    expected_args = historical_args()
    if options.side == "candidate":
        from main_basejit_irepa import parse_args
        from generation_input import build_denoiser

        args = parse_args(["--config", str(root / HISTORICAL_CONFIG)])
        mismatches = {key: [getattr(expected_args, key), getattr(args, key)]
                      for key in CONTRACT_FIELDS
                      if getattr(expected_args, key) != getattr(args, key)}
        if mismatches:
            raise AssertionError(f"Historical configuration mismatch: {mismatches}")
        factory = build_denoiser
    else:
        args = expected_args
        factory = Denoiser

    contract = {key: getattr(args, key) for key in CONTRACT_FIELDS}
    args.lr = args.blr * 512 / 256  # Historical absolute LR; test batch stays 1.
    report = {
        "side": options.side, "root": str(root), "mode": options.mode,
        "execution": options.execution, "contract": contract,
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                        "cpu_shim": device.type == "cpu", "cudnn_benchmark": False},
        "states": {}, "scalars": {}, "rng": {}, "structure": {},
    }
    activations = {}

    def capture(name, tensor):
        activations[name] = tensor.detach().cpu().clone()

    def capture_rng(name):
        states = [("cpu", torch.get_rng_state())]
        if device.type == "cuda":
            states.append(("cuda", torch.cuda.get_rng_state()))
        report["rng"][name] = state_digest(states)

    def autocast():
        return (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda" else contextlib.nullcontext())

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    capture_rng("seeded")
    if device.type == "cuda":
        report["dino_cache"] = require_cached_dino()
        encoders = load_encoders(args.enc_type, device, args.img_size)
        args.z_dims = [encoder.embed_dim for encoder in encoders]
        report["states"]["dino_weights"] = state_digest(encoders[0].model.state_dict().items())
        report["structure"]["dino_final_norm_affine"] = bool(
            encoders[0].model.norm.elementwise_affine
        )
    else:
        # Tiny CPU regression probe, not evidence of pretrained-DINO/GPU parity.
        args.img_size = 32
        args.z_dims = [24]
        model_jit.JiT_models[args.model] = lambda **kwargs: model_jit.JiT(
            depth=5, hidden_size=48, num_heads=3, bottleneck_dim=8,
            in_context_len=2, in_context_start=4, patch_size=16, **kwargs
        )
        encoders = [DINOEncoder("dino", "vit", "b", device, resolution=32)]
    capture_rng("after_regularization_encoder")

    model = factory(args).to(device)
    if type(model) is not Denoiser:
        raise AssertionError(f"Baseline must be an unwrapped Denoiser, got {type(model)}")
    if type(model.net) is not model_jit.JiT:
        raise AssertionError(f"Baseline must contain the native JiT, got {type(model.net)}")
    hooks_before_probe = sum(len(module._forward_hooks) + len(module._forward_pre_hooks)
                             for module in model.modules())
    if hooks_before_probe:
        raise AssertionError("Baseline has generation-input hooks")
    generation_modules = [name for name in sys.modules
                          if name in {"dino_v3_lora", "dino_jit_denoiser", "dino_encoder",
                                      "pixeldit2_projector", "jit_ddt_head"}]
    if generation_modules:
        raise AssertionError(f"Baseline unexpectedly imported generation code: {generation_modules}")
    report["structure"].update({
        "class": type(model).__name__, "net_class": type(model.net).__name__,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "hooks_before_probe": hooks_before_probe,
        "generation_modules_imported": generation_modules,
        "parameter_names": [name for name, _ in model.named_parameters()],
    })
    report["states"]["initial_model"] = state_digest(model.state_dict().items())
    capture_rng("after_model_construction")
    groups = misc.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))
    report["lr_schedule_probe"] = []
    for fractional_epoch in (0.0, 100 / 2502, 0.5, 1.0, 5.0, 19.99):
        lr_sched.adjust_learning_rate(optimizer, fractional_epoch, args)
        report["lr_schedule_probe"].append(
            [fractional_epoch, [group["lr"] for group in optimizer.param_groups]]
        )
    # Use a nonzero LR for both update probes; historical epoch0/iter0 LR=0
    # would otherwise conceal parameter/EMA-update differences.
    for group in optimizer.param_groups:
        group["lr"] = args.lr
    report["optimizer_probe_lr"] = args.lr
    model.ema_params1 = copy.deepcopy(list(model.parameters()))
    model.ema_params2 = copy.deepcopy(list(model.parameters()))

    # Fixed uint8 input, generated without touching the model's RNG stream.
    size = options.batch_size * 3 * args.img_size * args.img_size
    raw = ((torch.arange(size, dtype=torch.int64) * 73 + 19) % 256).to(torch.uint8)
    raw = raw.reshape(options.batch_size, 3, args.img_size, args.img_size).to(device)
    labels = torch.arange(options.batch_size, device=device, dtype=torch.long) % args.class_num
    capture("raw_uint8", raw)
    with torch.no_grad(), autocast():
        preprocessed = encoders[0].preprocess(raw)
        capture("dino_preprocessed", preprocessed)
        if device.type == "cuda":
            features = encoders[0].forward_features(preprocessed)
        else:
            # No RNG use; nonconstant spatial features exercise unbiased std.
            shape = (options.batch_size, 4, args.z_dims[0])
            values = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
            features = {"x_norm_patchtokens": torch.sin(values / 13), "x_norm_clstoken": None}
        capture("dino_raw_patchtokens", features["x_norm_patchtokens"])
        target = SpatialNormalization(args.spnorm_method)(
            feat=features["x_norm_patchtokens"], cls=features["x_norm_clstoken"],
            cls_weight=args.cls_token_weight, zscore_alpha=args.zscore_alpha,
            zscore_proj_skip_std=args.zscore_proj_skip_std,
        )
        capture("spatially_normalized_target", target)
    x = raw.to(torch.float32).div_(255) * 2.0 - 1.0
    capture("jit_clean_input", x)
    capture_rng("after_preprocessing")

    current_phase = ""
    def capture_net_input(module, inputs):
        capture(current_phase + "/noisy_input", inputs[0])
        capture(current_phase + "/timestep", inputs[1])
        capture(current_phase + "/dropped_labels", inputs[2])

    handles = [
        model.net.register_forward_pre_hook(capture_net_input),
        model.net.x_embedder.register_forward_hook(
            lambda module, inputs, output: capture(current_phase + "/patch_before_ape", output)),
        model.net.blocks[3].register_forward_hook(
            lambda module, inputs, output: capture(current_phase + "/block4_output", output)),
        model.net.blocks[4].register_forward_pre_hook(
            lambda module, inputs: capture(current_phase + "/block5_input", inputs[0])),
        model.net.register_forward_hook(
            lambda module, inputs, output: (
                capture(current_phase + "/predicted_image", output[0]),
                capture(current_phase + "/projected_tokens", output[1][0]),
                capture(current_phase + "/unprojected_tokens", output[2][0]),
            ) and None),
    ]
    for step, phase in enumerate(("native_init", "nonzero_probe")):
        current_phase = phase
        if step:
            # Test-only: make all residual blocks and final head contribute.
            # Same RNG and perturbation in both isolated processes.
            with torch.no_grad():
                model.net.final_layer.linear.weight.normal_(std=0.005)
                model.net.final_layer.linear.bias.normal_(std=0.005)
                model.net.final_layer.adaLN_modulation[-1].weight.normal_(std=0.005)
                for block in model.net.blocks:
                    block.adaLN_modulation[-1].weight.normal_(std=0.005)
            report["states"]["nonzero_probe_model"] = state_digest(model.state_dict().items())
        model.train()
        optimizer.zero_grad()
        capture_rng(phase + "/before_forward")
        with autocast():
            loss, loss_dict = model(x, labels, [target])
        capture(phase + "/loss", loss)
        report["scalars"][phase] = loss_dict
        capture_rng(phase + "/after_forward")
        loss.backward()
        report["states"][phase + "/gradients"] = state_digest(
            (name, parameter.grad) for name, parameter in model.named_parameters()
        )
        if step:
            gradients_to_check = (
                model.net.x_embedder.proj1.weight.grad,
                model.net.blocks[0].attn.qkv.weight.grad,
                model.net.blocks[-1].attn.qkv.weight.grad,
                model.net.projectors[0].projection_layer.weight.grad,
            )
            if not all(gradient is not None and gradient.abs().max().item() > 0
                       for gradient in gradients_to_check):
                raise AssertionError("Nonzero probe failed to exercise trunk/projector gradients")
        capture_rng(phase + "/after_backward")
        optimizer.step()
        model.update_ema()
        report["states"][phase + "/updated_model"] = state_digest(model.state_dict().items())
        report["states"][phase + "/optimizer"] = state_digest(
            (f"{name}/{key}", value)
            for name, parameter in model.named_parameters()
            for key, value in optimizer.state[parameter].items()
            if isinstance(value, torch.Tensor)
        )
        report["states"][phase + "/ema1"] = state_digest(
            (name, value) for (name, _), value in zip(model.named_parameters(), model.ema_params1)
        )
        report["states"][phase + "/ema2"] = state_digest(
            (name, value) for (name, _), value in zip(model.named_parameters(), model.ema_params2)
        )
        capture_rng(phase + "/after_update")
    for handle in handles:
        handle.remove()

    model.eval()
    z = x.clone()
    cfg_batch_sizes = []
    handle = model.net.register_forward_pre_hook(
        lambda module, inputs: cfg_batch_sizes.append(int(inputs[0].shape[0]))
    )
    # Closed boundaries are CFG-off in the original implementation; low=0 is
    # explicitly special-cased. Use nonzero weights so CFG can affect output.
    cases = ((0.0, (0.1, 1.0)), (0.1, (0.1, 1.0)), (0.1001, (0.1, 1.0)),
             (0.5, (0.1, 1.0)), (1.0, (0.1, 1.0)), (0.0, (0.0, 1.0)))
    with torch.no_grad(), autocast():
        for index, (value, interval) in enumerate(cases):
            model.cfg_interval = interval
            t = torch.full((options.batch_size, 1, 1, 1), value, device=device)
            capture(f"sampling/cfg_boundary_{index}", model._forward_sample(z, t, labels))
        model.cfg_interval = (args.interval_min, args.interval_max)
        t = torch.full((options.batch_size, 1, 1, 1), 0.1, device=device)
        t_next = torch.full_like(t, 0.2)
        capture("sampling/heun_step", model._heun_step(z, t, t_next, labels))
        handle.remove()
        # Full 50-step Heun trajectory can be requested; default is a 3-step
        # mechanism probe with the same final-Euler code path.
        model.steps = options.sampling_steps
        capture_rng("before_generation")
        capture("sampling/generated_image", model.generate(labels))
        capture_rng("after_generation")
    report["cfg_forward_batch_sizes"] = cfg_batch_sizes
    expected_batch_sizes = [1, 1, 2, 2, 1, 2, 1, 2]
    if cfg_batch_sizes != [options.batch_size * value for value in expected_batch_sizes]:
        raise AssertionError(f"Unexpected CFG branch boundaries: {cfg_batch_sizes}")
    report["sampling_steps_tested"] = options.sampling_steps
    report["activations"] = state_digest(activations.items())
    if device.type == "cuda":
        expected_dtype = {
            "native_init/patch_before_ape": "torch.bfloat16",
            "native_init/block4_output": "torch.bfloat16",
            "native_init/block5_input": "torch.float32",
        }
        for name, dtype in expected_dtype.items():
            if str(activations[name].dtype) != dtype:
                raise AssertionError(f"Unexpected native residual dtype at {name}")
        report["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(activations, output / f"{options.side}_activations.pt")
    (output / f"{options.side}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{options.side}: completed {options.mode} {options.execution} parity probe", flush=True)


def compare_reports(reference, candidate):
    """Compare all semantic evidence; machine paths/timings are descriptive."""
    differences = []

    def visit(path, left, right):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                if key not in left or key not in right:
                    differences.append({"path": f"{path}/{key}", "reason": "missing key"})
                else:
                    visit(f"{path}/{key}", left[key], right[key])
        elif left != right:
            differences.append({"path": path, "reference": left, "candidate": right})

    for section in ("mode", "execution", "contract", "environment", "states", "scalars",
                    "rng", "structure", "dino_cache", "cfg_forward_batch_sizes",
                    "sampling_steps_tested", "activations", "lr_schedule_probe", "optimizer_probe_lr"):
        visit(section, reference.get(section), candidate.get(section))
    return differences


def run_comparison(options):
    output = Path(options.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = os.environ.copy()
    env.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2", "PYTHONHASHSEED": "0"})
    if options.mode == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    for side, source in (("reference", options.reference_root),
                         ("candidate", options.candidate_root)):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--side", side,
               "--source-root", str(Path(source).resolve()), "--mode", options.mode,
               "--execution", options.execution, "--output", str(output),
               "--batch-size", str(options.batch_size),
               "--sampling-steps", str(options.sampling_steps)]
        if options.hub_dir:
            cmd += ["--hub-dir", str(Path(options.hub_dir).resolve())]
        with (output / f"{side}.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"{side} failed (exit {completed.returncode}); see {output / (side + '.log')}")
        print(f"Completed {side} probe; log: {output / (side + '.log')}", flush=True)
    reference = json.loads((output / "reference.json").read_text())
    candidate = json.loads((output / "candidate.json").read_text())
    differences = compare_reports(reference, candidate)

    import torch

    tensors_ref = torch.load(output / "reference_activations.pt", map_location="cpu", weights_only=True)
    tensors_new = torch.load(output / "candidate_activations.pt", map_location="cpu", weights_only=True)
    numeric = {}
    for name in sorted(set(tensors_ref) & set(tensors_new)):
        left, right = tensors_ref[name], tensors_new[name]
        if left.shape == right.shape:
            delta = (left.double() - right.double()).abs()
            numeric[name] = {
                "exact": bool(left.dtype == right.dtype and torch.equal(left, right)),
                "max_abs": delta.max().item(), "mean_abs": delta.mean().item(),
                "shape": list(left.shape), "reference_dtype": str(left.dtype),
                "candidate_dtype": str(right.dtype),
            }
    report = {
        "passed": not differences, "strict_bitwise": True,
        "mode": options.mode, "execution": options.execution,
        "batch_size": options.batch_size, "sampling_steps": options.sampling_steps,
        "elapsed_seconds": time.time() - started, "differences": differences,
        "tensor_comparison": numeric,
        "limitations": [
            "CPU uses a tiny network, synthetic target, and test-only .cuda() shim."
            if options.mode == "cpu" else "CUDA uses real pretrained DINOv1 and full JiT-B/16 at resolution 256.",
            "Two independent single-process runs; no DDP/all-reduce or data-loader augmentation test.",
            "Deterministic cuDNN policy for reproducible comparison, not production benchmark policy.",
            "Second step perturbs final/adaLN weights only inside test processes to exercise gradients.",
            "Optimizer probes use nonzero historical peak LR; the warmup schedule is compared separately.",
            "The selected execution mode is tested; eager results do not test compiled kernels.",
            "Trajectory length is the recorded sampling_steps; configuration still verifies Heun50.",
        ],
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "differences": len(differences),
                      "activations": len(numeric), "report": str(output / "comparison.json")}))
    if differences:
        raise SystemExit(1)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--execution", choices=("eager", "compiled"), default="eager")
    parser.add_argument("--reference-root", default=str(PROJECT_ROOT.parent / "iREPA"))
    parser.add_argument("--candidate-root", default=str(PROJECT_ROOT))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "tests_results" / "baseline_parity"))
    parser.add_argument("--hub-dir", help="Existing official torch hub cache directory")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sampling-steps", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--side", choices=("reference", "candidate"), help=argparse.SUPPRESS)
    parser.add_argument("--source-root", help=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    options = get_parser().parse_args()
    if options.batch_size < 1 or options.sampling_steps < 2:
        raise SystemExit("batch-size must be >=1; sampling-steps must be >=2")
    if options.mode == "cpu" and options.execution == "compiled":
        raise SystemExit("CPU mode tests only eager execution; use CUDA for compiled parity")
    (run_worker if options.worker else run_comparison)(options)
