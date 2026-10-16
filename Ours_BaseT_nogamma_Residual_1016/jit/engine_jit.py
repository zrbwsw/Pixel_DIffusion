"""JiT/J18-new 共用的单 epoch 训练与 class-conditional 在线评估循环。

模型计算在 CUDA BF16 autocast 下执行；输入归一化、若干 attention/loss 统计会
显式回到 FP32。当前实现未使用 GradScaler，因为 BF16 的指数范围通常不需要
loss scaling。DDP 的 ``no_sync`` 仅在梯度累计的非更新 micro-step 跳过通信。
"""

import math
import sys
import os
import shutil
import csv
import json
from contextlib import nullcontext

import torch
import torch.distributed as dist
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
import copy


def _as_loggable_scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    if isinstance(value, np.generic):
        return float(value)
    if isinstance(value, (float, int)):
        return float(value)
    return value


def _append_jsonl(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _append_csv(path, row):
    """Append a best-effort wide CSV row.

    The matching JSONL file is the authoritative record.  CSV mirrors it for
    quick inspection and expands its header if resumed runs add new metrics.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing_header = None
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_header = reader.fieldnames

    if existing_header is None:
        fieldnames = list(row.keys())
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in fieldnames})
        return

    new_keys = [key for key in row if key not in existing_header]
    if not new_keys:
        # Normal training path: the schema is fixed after the first row, so do
        # an O(1) append instead of rewriting an ever-growing diagnostics file.
        with open(path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=existing_header)
            writer.writerow({key: row.get(key, "") for key in existing_header})
        return

    # Resumed runs may legitimately introduce new metrics. Expand the header
    # once while preserving prior rows, then return to the append-only path.
    with open(path, "r", encoding="utf-8", newline="") as handle:
        existing_rows = list(csv.DictReader(handle))
    fieldnames = list(existing_header) + new_keys
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for existing_row in existing_rows:
            writer.writerow({key: existing_row.get(key, "") for key in fieldnames})
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _append_metric_row(args, filename_prefix, row):
    if not misc.is_main_process() or args is None or not getattr(args, "output_dir", None):
        return
    if not getattr(args, "write_structured_metrics", False):
        return
    output_dir = getattr(args, "output_dir")
    jsonl_path = os.path.join(output_dir, f"{filename_prefix}.jsonl")
    csv_path = os.path.join(output_dir, f"{filename_prefix}.csv")
    _append_jsonl(jsonl_path, row)
    _append_csv(csv_path, row)


def _jit_block_index(name):
    """Return the zero-based JiT block index encoded in a parameter name."""
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "net" and parts[1] == "blocks":
        try:
            return int(parts[2])
        except ValueError:
            return None
    return None


def _iter_named_parameters_by_group(model_without_ddp, group):
    """Yield named parameters for logging-only norm groups.

    Groups are intentionally name-based so this works for both the original JiT
    denoiser and the DINO+JiT wrapper without introducing a hard dependency.
    """
    for name, param in model_without_ddp.named_parameters():
        if not param.requires_grad:
            continue
        is_jit = name.startswith("net.")
        jit_block_index = _jit_block_index(name)
        alignment_depth = getattr(model_without_ddp, "jit_alignment_depth", None)
        is_ddt_head = name.startswith("ddt_head.")
        is_lora = ("lora_A" in name) or ("lora_B" in name)
        is_dino_encoder = name.startswith("dino_student.")
        is_alignment_projector = name.startswith("dino_student_projector.")
        is_input_projector = name.startswith("dino_input_projector.")
        is_residual_adapter = name.startswith("dino_residual_adapter.")
        is_dino_patch_embed = is_dino_encoder and ".patch_embed." in name

        if group == "jit" and is_jit:
            yield name, param
        elif (
            group == "jit_pre_alignment"
            and alignment_depth is not None
            and jit_block_index is not None
            and jit_block_index < int(alignment_depth)
        ):
            yield name, param
        elif (
            group == "jit_post_alignment"
            and alignment_depth is not None
            and jit_block_index is not None
            and jit_block_index >= int(alignment_depth)
        ):
            yield name, param
        elif group == "ddt_head" and is_ddt_head:
            yield name, param
        elif group == "jit_plus_ddt" and (is_jit or is_ddt_head):
            yield name, param
        elif group == "lora" and is_lora:
            yield name, param
        elif group == "dino_non_lora" and is_dino_encoder and not is_lora:
            yield name, param
        elif group == "dino_patch_embed" and is_dino_patch_embed:
            yield name, param
        elif group == "alignment_projector" and is_alignment_projector:
            yield name, param
        elif group == "input_projector" and is_input_projector:
            yield name, param
        elif group == "residual_adapter" and is_residual_adapter:
            yield name, param
        elif group == "total":
            yield name, param


def _norm_from_tensors(tensors):
    total_sq = None
    count = 0
    for tensor in tensors:
        if tensor is None:
            continue
        value = tensor.detach().float().pow(2).sum()
        total_sq = value if total_sq is None else total_sq + value
        count += tensor.numel()
    if total_sq is None:
        return 0.0, 0
    return float(torch.sqrt(total_sq).item()), int(count)


def _param_norm(model_without_ddp, group):
    return _norm_from_tensors(param for _, param in _iter_named_parameters_by_group(model_without_ddp, group))


def _grad_norm(model_without_ddp, group):
    return _norm_from_tensors(
        param.grad for _, param in _iter_named_parameters_by_group(model_without_ddp, group) if param.grad is not None
    )


def _summarize_train_scalars(loss_value, reduced_loss_dict, lr, grad_stats, param_stats):
    """Flatten the key training scalars used by JSONL/CSV logs."""
    return {
        "lr": _as_loggable_scalar(lr),
        "loss": _as_loggable_scalar(loss_value),
        "total_loss": _as_loggable_scalar(reduced_loss_dict.get("total_loss", loss_value)),
        "denoise_loss": _as_loggable_scalar(reduced_loss_dict["denoise_loss"]),
        "alignment_loss": _as_loggable_scalar(reduced_loss_dict["alignment_loss"]),
        "alignment_loss_weighted": _as_loggable_scalar(reduced_loss_dict["alignment_loss_weighted"]),
        "grad_norm": grad_stats["total"][0],
        "grad_norm/jit": grad_stats["jit"][0],
        "grad_norm/jit_pre_alignment": grad_stats["jit_pre_alignment"][0],
        "grad_norm/jit_post_alignment": grad_stats["jit_post_alignment"][0],
        "grad_norm/ddt_head": grad_stats["ddt_head"][0],
        "grad_norm/jit_plus_ddt": grad_stats["jit_plus_ddt"][0],
        "grad_norm/lora": grad_stats["lora"][0],
        "grad_norm/dino_non_lora_trainable": grad_stats["dino_non_lora"][0],
        "grad_norm/dino_patch_embed": grad_stats["dino_patch_embed"][0],
        "grad_norm/alignment_projector": grad_stats["alignment_projector"][0],
        "grad_norm/input_projector": grad_stats["input_projector"][0],
        "grad_norm/residual_adapter": grad_stats["residual_adapter"][0],
        "grad_norm_ratio/alignment_projector_to_jit_pre_alignment": (
            grad_stats["alignment_projector"][0]
            / max(grad_stats["jit_pre_alignment"][0], 1e-12)
        ),
        "grad_param_count/total": grad_stats["total"][1],
        "grad_param_count/jit": grad_stats["jit"][1],
        "grad_param_count/jit_pre_alignment": grad_stats["jit_pre_alignment"][1],
        "grad_param_count/jit_post_alignment": grad_stats["jit_post_alignment"][1],
        "grad_param_count/ddt_head": grad_stats["ddt_head"][1],
        "grad_param_count/jit_plus_ddt": grad_stats["jit_plus_ddt"][1],
        "grad_param_count/lora": grad_stats["lora"][1],
        "grad_param_count/dino_non_lora_trainable": grad_stats["dino_non_lora"][1],
        "grad_param_count/dino_patch_embed": grad_stats["dino_patch_embed"][1],
        "grad_param_count/alignment_projector": grad_stats["alignment_projector"][1],
        "grad_param_count/input_projector": grad_stats["input_projector"][1],
        "grad_param_count/residual_adapter": grad_stats["residual_adapter"][1],
        "param_norm": param_stats["total"][0],
        "param_norm/jit": param_stats["jit"][0],
        "param_norm/jit_pre_alignment": param_stats["jit_pre_alignment"][0],
        "param_norm/jit_post_alignment": param_stats["jit_post_alignment"][0],
        "param_norm/ddt_head": param_stats["ddt_head"][0],
        "param_norm/jit_plus_ddt": param_stats["jit_plus_ddt"][0],
        "param_norm/lora": param_stats["lora"][0],
        "param_norm/dino_non_lora_trainable": param_stats["dino_non_lora"][0],
        "param_norm/dino_patch_embed": param_stats["dino_patch_embed"][0],
        "param_norm/alignment_projector": param_stats["alignment_projector"][0],
        "param_norm/input_projector": param_stats["input_projector"][0],
        "param_norm/residual_adapter": param_stats["residual_adapter"][0],
        "param_count/total": param_stats["total"][1],
        "param_count/jit": param_stats["jit"][1],
        "param_count/jit_pre_alignment": param_stats["jit_pre_alignment"][1],
        "param_count/jit_post_alignment": param_stats["jit_post_alignment"][1],
        "param_count/ddt_head": param_stats["ddt_head"][1],
        "param_count/jit_plus_ddt": param_stats["jit_plus_ddt"][1],
        "param_count/lora": param_stats["lora"][1],
        "param_count/dino_non_lora_trainable": param_stats["dino_non_lora"][1],
        "param_count/dino_patch_embed": param_stats["dino_patch_embed"][1],
        "param_count/alignment_projector": param_stats["alignment_projector"][1],
        "param_count/input_projector": param_stats["input_projector"][1],
        "param_count/residual_adapter": param_stats["residual_adapter"][1],
    }


def train_one_epoch(
    model, model_without_ddp, data_loader, optimizer, device, epoch, args=None
):
    """训练一个 epoch。

    J18-new 两卡 H100 示例：DataLoader 每 rank 给 ``x uint8 [128,3,256,256]``、
    ``labels [128]``；world_size=2、accum_iter=1，因此有效全局 batch=256。
    Teacher target 和 iREPA loss 均在 ``DINOJiTDenoiser.forward`` 内部完成。
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = max(1, int(getattr(args, "log_freq", 20)))

    accum_iter = max(1, int(getattr(args, "accum_iter", 1)))
    metrics_log_freq = getattr(args, "metrics_log_freq", None)
    if metrics_log_freq is None or int(metrics_log_freq) <= 0:
        metrics_log_freq = int(getattr(args, "log_freq", print_freq))
    metrics_log_freq = max(1, int(metrics_log_freq))
    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # x 是 uint8 [B_local,3,256,256]；labels 是 int64 [B_local]。
        x, labels = batch

        # uint8 [0,255] -> float32 [-1,1]。这也是 flow matching 的像素空间。
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0
        labels = labels.to(device, non_blocking=True)

        update_grad = ((data_iter_step + 1) % accum_iter == 0) or ((data_iter_step + 1) == len(data_loader))
        backend_log_due = data_iter_step % int(getattr(args, "log_freq", print_freq)) == 0
        structured_log_due = (
            data_iter_step % metrics_log_freq == 0
            or data_iter_step == len(data_loader) - 1
        )
        diagnostics_due = backend_log_due or structured_log_due
        sync_context = model.no_sync() if (not update_grad and hasattr(model, "no_sync")) else nullcontext()
        with sync_context:
            # 大部分卷积/线性/attention 以 BF16 执行；DINO preprocess、attention
            # logits 和 loss 中显式 .float() 的部分保持 FP32。
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, loss_dict = model(x, labels)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # 每个 micro-step 除以 accum_iter，确保累计后的梯度尺度等价于大 batch。
            (loss / accum_iter).backward()
        # Gradients are logged after backward and before optimizer.step()/zero_grad().
        # Therefore near-zero group norms here are real for the current graph, not a
        # logging artifact from cleared gradients.
        grad_stats = None
        if diagnostics_due:
            grad_stats = {
                "total": _grad_norm(model_without_ddp, "total"),
                "jit": _grad_norm(model_without_ddp, "jit"),
                "jit_pre_alignment": _grad_norm(model_without_ddp, "jit_pre_alignment"),
                "jit_post_alignment": _grad_norm(model_without_ddp, "jit_post_alignment"),
                "ddt_head": _grad_norm(model_without_ddp, "ddt_head"),
                "jit_plus_ddt": _grad_norm(model_without_ddp, "jit_plus_ddt"),
                "lora": _grad_norm(model_without_ddp, "lora"),
                "dino_non_lora": _grad_norm(model_without_ddp, "dino_non_lora"),
                "dino_patch_embed": _grad_norm(model_without_ddp, "dino_patch_embed"),
                "alignment_projector": _grad_norm(model_without_ddp, "alignment_projector"),
                "input_projector": _grad_norm(model_without_ddp, "input_projector"),
                "residual_adapter": _grad_norm(model_without_ddp, "residual_adapter"),
            }
        if update_grad:
            # DDP 已在 backward 同步各 rank 梯度；一次 AdamW step 后更新两套 EMA。
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            model_without_ddp.update_ema()
        param_stats = None
        if diagnostics_due:
            param_stats = {
                "total": _param_norm(model_without_ddp, "total"),
                "jit": _param_norm(model_without_ddp, "jit"),
                "jit_pre_alignment": _param_norm(model_without_ddp, "jit_pre_alignment"),
                "jit_post_alignment": _param_norm(model_without_ddp, "jit_post_alignment"),
                "ddt_head": _param_norm(model_without_ddp, "ddt_head"),
                "jit_plus_ddt": _param_norm(model_without_ddp, "jit_plus_ddt"),
                "lora": _param_norm(model_without_ddp, "lora"),
                "dino_non_lora": _param_norm(model_without_ddp, "dino_non_lora"),
                "dino_patch_embed": _param_norm(model_without_ddp, "dino_patch_embed"),
                "alignment_projector": _param_norm(model_without_ddp, "alignment_projector"),
                "input_projector": _param_norm(model_without_ddp, "input_projector"),
                "residual_adapter": _param_norm(model_without_ddp, "residual_adapter"),
            }

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # loss_value_reduce = misc.all_reduce_mean(loss_value)
        reduced_loss_dict = {}
        for key, value in loss_dict.items():
            reduced_loss_dict[key] = misc.all_reduce_mean(value)  # called on all ranks
        metric_logger.update(**reduced_loss_dict)
        train_scalars = None
        if diagnostics_due:
            assert grad_stats is not None and param_stats is not None
            train_scalars = _summarize_train_scalars(
                loss_value, reduced_loss_dict, lr, grad_stats, param_stats
            )
            metric_logger.update(
                total_loss=train_scalars["total_loss"],
                grad_norm=train_scalars["grad_norm"],
                grad_norm_jit=train_scalars["grad_norm/jit"],
                grad_norm_jit_pre_alignment=train_scalars["grad_norm/jit_pre_alignment"],
                grad_norm_jit_post_alignment=train_scalars["grad_norm/jit_post_alignment"],
                grad_norm_ddt_head=train_scalars["grad_norm/ddt_head"],
                grad_norm_lora=train_scalars["grad_norm/lora"],
                grad_norm_alignment_projector=train_scalars["grad_norm/alignment_projector"],
                grad_norm_input_projector=train_scalars["grad_norm/input_projector"],
                grad_norm_residual_adapter=train_scalars["grad_norm/residual_adapter"],
                param_norm_jit=train_scalars["param_norm/jit"],
                param_norm_ddt_head=train_scalars["param_norm/ddt_head"],
                param_norm_lora=train_scalars["param_norm/lora"],
                param_norm_alignment_projector=train_scalars["param_norm/alignment_projector"],
                param_norm_input_projector=train_scalars["param_norm/input_projector"],
                param_norm_residual_adapter=train_scalars["param_norm/residual_adapter"],
            )


        if structured_log_due:
            assert train_scalars is not None
            world_size = misc.get_world_size()
            local_batch_size = int(x.shape[0])
            global_iter = epoch * len(data_loader) + data_iter_step
            row = {
                "kind": "train",
                "epoch": int(epoch),
                "iter": int(data_iter_step),
                "iters_per_epoch": int(len(data_loader)),
                "global_iter": int(global_iter),
                "optimizer_step": int(global_iter // accum_iter),
                "epoch_float": float(epoch + data_iter_step / len(data_loader)),
                "epoch_1000x": int((data_iter_step / len(data_loader) + epoch) * 1000),
                "local_batch_size": local_batch_size,
                "world_size": int(world_size),
                "accum_iter": int(accum_iter),
                "effective_batch_size": int(local_batch_size * world_size * accum_iter),
                "samples_seen": int((global_iter + 1) * local_batch_size * world_size),
                "max_mem_mb": int(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0,
                "update_grad": bool(update_grad),
            }
            row.update(train_scalars)
            for key in sorted(reduced_loss_dict):
                row[f"loss/{key}"] = _as_loggable_scalar(reduced_loss_dict[key])
            _append_metric_row(args, getattr(args, "metrics_log_name", "train_metrics"), row)



def online_evaluation_assets(img_size):
    """Resolve online-only JiT moments and Inception weights.

    Online training evaluation intentionally uses the shared, lightweight JiT
    reference under ``fid_stats``. The nanogen-evals extractor zoo is reserved
    for standalone offline FID/IS/FDR6 evaluation.
    """
    if img_size not in (256, 512):
        raise NotImplementedError(f"No online FID reference for image size {img_size}")
    root = os.environ.get(
        "FID_STATS_ROOT", "/data0/zhangruobei/Work/Pixel_Diffusion/fid_stats"
    )
    statistics = os.path.join(root, f"jit_in{img_size}_stats.npz")
    weights = os.path.join(root, "weights-inception-2015-12-05-6726825d.pth")
    for path in (statistics, weights):
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise FileNotFoundError(
                f"Online evaluation resource missing or empty: {path}. "
                "FID_STATS_ROOT must contain both JiT statistics and Inception weights."
            )
    return statistics, weights


def _on_eval_rank_zero(action):
    """Broadcast rank-0 evaluation/export errors before peers enter a barrier."""
    error = [None]
    if misc.is_main_process():
        try:
            action()
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"Evaluation failed on rank 0: {error[0]}")


def evaluate(model_without_ddp, args, epoch, batch_size=64):
    """使用 EMA1 生成 class-balanced 50k 图像并计算 FID/IS。

    ``batch_size`` 即每 rank 的 gen_bsz。CFG 活跃时模型内部把 cond/uncond 沿 batch
    拼接，因此实际网络 forward 是 ``2*gen_bsz``。所有 rank 以不同连续索引保存
    PNG，rank 0 在 barrier 后调用 torch-fidelity。
    """

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    if args.num_images < 2 or batch_size < 1 or args.num_images % args.class_num:
        raise ValueError("num_images must be >=2 and class-balanced; gen_bsz must be positive")
    num_steps = math.ceil(args.num_images / (batch_size * world_size))

    generation_root = os.path.join(args.output_dir, "generated_images")
    save_folder = os.path.join(
        generation_root,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    def prepare_directory():
        npz_path = getattr(args, "save_eval_npz", None)
        if npz_path and (not npz_path.endswith(".npz") or os.path.exists(npz_path)):
            raise ValueError(f"save_eval_npz must be a new .npz path: {npz_path}")
        os.makedirs(save_folder, exist_ok=True)
    _on_eval_rank_zero(prepare_directory)
    # All ranks share this directory. The barrier makes it safe to inspect existing
    # PNGs immediately when resuming a previously interrupted evaluation.
    torch.distributed.barrier()

    # 临时把当前参数替换成 EMA1；生成结束后恢复原训练参数。
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    # 50k / 1000 = 每类 50 张，避免类别采样偏差影响 FID/IS。
    class_num = args.class_num
    assert args.num_images % class_num == 0, "Number of images per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.pad(class_label_gen_world, (0, batch_size * world_size))

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        global_start_idx = world_size * batch_size * i
        global_end_idx = min(global_start_idx + world_size * batch_size, args.num_images)
        if getattr(args, "resume_eval_images", False):
            # Rank 0 checks the entire global batch. Only a fully written batch is
            # skipped; a partial batch is regenerated by every rank and overwritten.
            batch_complete = False
            if misc.is_main_process():
                batch_complete = all(
                    os.path.isfile(os.path.join(save_folder, f"{img_id:05d}.png"))
                    and os.path.getsize(os.path.join(save_folder, f"{img_id:05d}.png")) > 0
                    for img_id in range(global_start_idx, global_end_idx)
                )
            completion_flag = torch.tensor(
                [int(batch_complete)], device=next(model_without_ddp.parameters()).device
            )
            torch.distributed.broadcast(completion_flag, src=0)
            if completion_flag.item():
                print(
                    "Skip generation step {}/{}: images {:05d}-{:05d} already complete".format(
                        i, num_steps, global_start_idx, global_end_idx - 1
                    )
                )
                continue

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_gen)

        torch.distributed.barrier()

        # denormalize images
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        # distributed save images
        for b_id in range(sampled_images.size(0)):
            img_id = start_idx + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            image_path = os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5)))
            temporary_path = f"{image_path}.rank{local_rank}.tmp.png"
            if not cv2.imwrite(temporary_path, gen_img):
                raise OSError(f"Failed to write generated image: {temporary_path}")
            os.replace(temporary_path, image_path)

    torch.distributed.barrier()

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

    _on_eval_rank_zero(lambda: _finish_evaluation(
        model_without_ddp, args, epoch, batch_size, save_folder, generation_root
    ))


def _finish_evaluation(model_without_ddp, args, epoch, batch_size, save_folder, generation_root):
    npz_path = getattr(args, "save_eval_npz", None)
    if npz_path:
        from evaluation.artifacts import export_pngs

        export_pngs(save_folder, npz_path, args.num_images, args.img_size, metadata={
            "checkpoint": getattr(args, "sampling_checkpoint", None), "epoch": int(epoch),
            "ema": "ema1", "ema_decay": float(args.ema_decay1),
            "sampler": model_without_ddp.method, "steps": model_without_ddp.steps,
            "cfg": float(model_without_ddp.cfg_scale),
            "cfg_interval": list(model_without_ddp.cfg_interval),
            "seed": int(args.seed), "seed_rule": "seed + rank",
            "gen_bsz": batch_size, "world_size": misc.get_world_size(),
            "class_num": args.class_num, "label_order": "class-major balanced",
            "resume_eval_images": bool(getattr(args, "resume_eval_images", False)),
            "config": {k: v for k, v in vars(args).items() if v is None or isinstance(v, (str, int, float, bool))},
        })
        print("Saved evaluation NPZ:", npz_path)

    if not getattr(args, "skip_online_eval", False):
        fid_statistics_file, weights_path = online_evaluation_assets(args.img_size)
        print("Online FID statistics:", fid_statistics_file)
        print("Online Inception weights:", weights_path)
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            feature_extractor_weights_path=weights_path,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))

        _append_metric_row(
            args,
            getattr(args, "eval_metrics_log_name", "eval_metrics"),
            {
                "kind": "eval",
                "epoch": int(epoch),
                "fid": float(fid),
                "inception_score": float(inception_score),
                "cfg": float(model_without_ddp.cfg_scale),
                "cfg_interval_low": float(model_without_ddp.cfg_interval[0]),
                "cfg_interval_high": float(model_without_ddp.cfg_interval[1]),
                "img_size": int(args.img_size),
                "num_images": int(args.num_images),
                "gen_bsz": int(batch_size),
                "ema": "ema1",
                "ema_decay": float(getattr(args, "ema_decay1", 0.9999)),
                "fid_statistics_file": fid_statistics_file,
                "feature_extractor_weights_path": weights_path,
            },
        )

    else:
        print("Skipped JiT online FID/IS; run evaluation.evaluate_fdr6 offline after sampling exits.")

    if getattr(args, "keep_eval_images", False) or npz_path or getattr(args, "skip_online_eval", False):
        print("Keep generated images at:", save_folder)
    else:
        shutil.rmtree(save_folder)
        try:
            os.rmdir(generation_root)
        except OSError:
            pass
