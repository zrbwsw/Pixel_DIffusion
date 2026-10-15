"""JiT 原始 class-to-image 训练主程序，J18-new 复用其基础设施。

直接运行本文件会构建 ``jit/denoiser.py::Denoiser``。运行
``main_dino_jit.py`` 时，入口会在调用 ``main`` 前把模块级 ``Denoiser`` 替换为
``DINOJiTDenoiser``，并替换数据集类。因此本文件在 J18-new 中仍负责 DDP、
DataLoader、AdamW、学习率、checkpoint、EMA 切换和在线 FID/IS，但不决定新的
模型 forward。

有效全局 batch 计算为 ``batch_size_per_rank * world_size * accum_iter``。以标准
J18-new H100 两卡为例：128 * 2 * 1 = 256；模型单次前向的本地 B 仍是 128。
"""

import argparse
import datetime
import numpy as np
import os
import sys
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
from dataset import HFImageDataset

from util.crop import center_crop_arr
import util.misc as misc

import copy
from engine_jit import train_one_epoch, evaluate, online_evaluation_assets
from config_utils import parse_args_with_config

# 外层 J18-new 入口会在调用 main() 前注入 DINOJiTDenoiser。直接运行原生 JiT
# 时则在 main() 内延迟导入 baseline Denoiser，避免当前主线 import 未使用的
# projection-loss / REPA wrapper。
Denoiser = None

def get_args_parser():
    """Declare training, checkpoint and sampling settings for the DINO-JiT model."""
    parser = argparse.ArgumentParser('JiT', add_help=False)

    # J18-new B-scale: model=JiT-B/16, img_size=256。
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')

    # batch_size 是每个 rank 的本地 batch；global_batch_size 才是有效总 batch。
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Gradient accumulation iterations; overridden by --global_batch_size when set')
    parser.add_argument('--global_batch_size', default=0, type=int,
                        help='Target effective global batch size. If >0, accum_iter is derived as '
                             'global_batch_size / (batch_size * # GPUs).')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling：CFG=1 表示关闭。Heun-50 除最后一步外每步两次网络评估。
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--keep_eval_images', action='store_true',
                        help='Keep generated PNGs after FID/IS evaluation')
    parser.add_argument('--resume_eval_images', action='store_true',
                        help='Resume image generation by skipping complete on-disk batches. '
                             'Incomplete batches are regenerated atomically.')
    parser.add_argument('--save_eval_npz', default=None, type=str,
                        help='Standalone sampling: export numbered PNGs to this explicit .npz path')
    parser.add_argument('--skip_online_eval', action='store_true',
                        help='Standalone sampling only: defer FID/IS/FDR6 to evaluation.evaluate_fdr6')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')
    parser.add_argument('--gen_checkpoint_path', default=None, type=str,
                        help='Path to the checkpoint to generate from')

    # dataset
    parser.add_argument('--data_path', default='../data', type=str,
                        help='Path to the dataset')
    parser.add_argument('--class_num', default=1000, type=int)
    parser.add_argument('--hflip_prob', default=0.0, type=float,
                        help='Training-time random horizontal flip probability. '
                             'The current RAEv2/PixelDiT-aligned protocol uses 0.0; '
                             'legacy JiT/iREPA experiments used 0.5.')

    # checkpointing
    parser.add_argument('--output_dir', default='./exps',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--exp_name', default=None, type=str,
                        help='Experiment name')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--metrics_log_freq', default=None, type=int,
                        help='Frequency (in iterations) for JSONL/CSV metric rows. Defaults to log_freq.')
    parser.add_argument('--metrics_log_name', default='train_metrics', type=str,
                        help='Base filename for structured training metrics under output_dir.')
    parser.add_argument('--eval_metrics_log_name', default='eval_metrics', type=str,
                        help='Base filename for structured online-eval metrics under output_dir.')
    parser.add_argument('--write_structured_metrics', action='store_true',
                        help='Opt in to writing train/eval JSONL and CSV metric files.')

    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # config file (YAML)
    parser.add_argument("--config", type=str, default=None,
        help="Path to a flat or module-grouped YAML experiment config")

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    return parser


def evaluate_checkpoint(model, args, seed):
    # Standalone generation loads model/EMA only, without a dataset or optimizer.
    checkpoint_path = args.gen_checkpoint_path
    if not checkpoint_path and args.resume:
        checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth")
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Sampling requires an existing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    model.load_state_dict(checkpoint['model'], strict=True)
    device = next(model.parameters()).device
    for index in (1, 2):
        ema = checkpoint[f'model_ema{index}']
        setattr(model, f'ema_params{index}', [ema[name].to(device) for name, _ in model.named_parameters()])
    args.start_epoch = int(checkpoint['epoch'])
    stat = os.stat(checkpoint_path)
    args.sampling_checkpoint = {
        "path": os.path.realpath(checkpoint_path), "epoch": args.start_epoch,
        "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
    }
    del checkpoint
    print("Evaluating checkpoint at {} epoch: {}".format(args.start_epoch, checkpoint_path))
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        with torch.no_grad():
            evaluate(model, args, args.start_epoch, batch_size=args.gen_bsz)


def main(args):
    """初始化分布式环境、数据、模型和优化器，并执行训练或纯评估。"""
    if (args.save_eval_npz or args.skip_online_eval) and not args.evaluate_gen:
        raise ValueError("save_eval_npz and skip_online_eval are standalone --evaluate_gen options")
    if (args.online_eval or args.evaluate_gen) and not args.skip_online_eval:
        online_evaluation_assets(args.img_size)
    misc.init_distributed_mode(args)
    if not 0.0 <= args.hflip_prob <= 1.0:
        raise ValueError(f"hflip_prob must be in [0, 1], got {args.hflip_prob}")
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 训练 forward 在 engine_jit 中使用 BF16 autocast；这里同时允许 TF32 加速
    # 未被 autocast 覆盖的 float32 matmul。模型参数和 AdamW master 参数仍是 FP32。
    cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if not args.evaluate_gen:
        try:
            # main_dino_jit replaces HFImageDataset with FlexibleImageNet256Dataset.
            transform_train = None
            if args.hflip_prob > 0.0:
                transform_train = transforms.RandomHorizontalFlip(p=args.hflip_prob)
            dataset_train = HFImageDataset(args.data_path, split="train", transform=transform_train)
        except Exception as e:
            print(f"Error loading HFImageDataset: {e}")
            print("Falling back to ImageFolder")
            transform_ops = [
                transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
            ]
            if args.hflip_prob > 0.0:
                transform_ops.append(transforms.RandomHorizontalFlip(p=args.hflip_prob))
            transform_ops.append(transforms.PILToTensor())
            transform_train = transforms.Compose(transform_ops)
            dataset_train = ImageFolder(os.path.join(args.data_path, "imagenet", "train"), transform=transform_train)

        print(f"Training horizontal flip probability: {args.hflip_prob}")
        print(dataset_train)
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train =", sampler_train)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True
        )

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    if Denoiser is None:
        raise RuntimeError("Use main_dino_jit.py as the training/sampling entry point.")
    denoiser_cls = Denoiser
    model = denoiser_cls(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    if args.evaluate_gen:
        evaluate_checkpoint(model, args, seed)
        return

    world_size = misc.get_world_size()
    local_global_batch = args.batch_size * world_size
    if args.global_batch_size > 0:
        if args.global_batch_size < local_global_batch:
            raise ValueError(
                f"global_batch_size={args.global_batch_size} is smaller than "
                f"batch_size * world_size = {args.batch_size} * {world_size} = {local_global_batch}."
            )
        if args.global_batch_size % local_global_batch != 0:
            raise ValueError(
                f"global_batch_size={args.global_batch_size} must be divisible by "
                f"batch_size * world_size = {local_global_batch}."
            )
        derived_accum_iter = args.global_batch_size // local_global_batch
        if args.accum_iter != derived_accum_iter:
            print(
                "Overriding accum_iter from {} to {} to match global_batch_size={}".format(
                    args.accum_iter, derived_accum_iter, args.global_batch_size
                )
            )
        args.accum_iter = derived_accum_iter

    # 示例：两卡 H100 为 128*2*1=256；四卡本机为 64*4*1=256。
    # DDP 对各 rank 梯度求平均，梯度累计再把多个 micro-batch 合成一次更新。
    eff_batch_size = args.batch_size * world_size * args.accum_iter
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    if args.global_batch_size > 0:
        print("Requested global batch size: %d" % args.global_batch_size)
    print("Effective batch size: %d" % eff_batch_size)
    print("Gradient accumulation iterations: %d" % args.accum_iter)

    # 每个进程持有完整模型和一个本地 batch；DDP 只同步梯度，不拼接前向张量。
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    # AdamW 更新 JiT、DDTHead、DINO LoRA 及显式解冻的 student 参数；frozen
    # teacher 不在 named_parameters 中，也不会进入 optimizer/checkpoint/EMA。
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # checkpoint-last.pth restores training weights, both EMAs and the optimizer.
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        checkpoint_args = checkpoint.get('args')
        if checkpoint_args is not None and not args.evaluate_gen:
            if isinstance(checkpoint_args, dict):
                saved_hflip_prob = checkpoint_args.get('hflip_prob', 0.5)
            else:
                # 所有引入 hflip_prob 参数之前保存的 JiT/iREPA checkpoint 都由
                # 硬编码 RandomHorizontalFlip() 训练，因此缺失字段等价于 0.5。
                saved_hflip_prob = getattr(checkpoint_args, 'hflip_prob', 0.5)
            if abs(float(saved_hflip_prob) - float(args.hflip_prob)) > 1e-12:
                raise ValueError(
                    "Refusing to change training augmentation while resuming: "
                    f"checkpoint hflip_prob={saved_hflip_prob}, current hflip_prob={args.hflip_prob}. "
                    "Pass the checkpoint value explicitly, or start a new experiment from scratch."
                )
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resumed checkpoint from", args.resume)

        # Manually set the start epoch for generation or evaluation for correct saving, but we are not loading the optimizer state
        if args.evaluate_gen:
            args.start_epoch = checkpoint['epoch']

        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not args.evaluate_gen:
            optimizer.load_state_dict(checkpoint['optimizer'])
            # Resume from the next epoch after the one stored in the checkpoint.
            # Here, checkpoint['epoch'] is interpreted as "number of epochs finished".
            args.start_epoch = checkpoint['epoch']
            print("Loaded optimizer & scaler state!")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

    # epoch 是完整遍历一次 ImageNet train split。J18-new 通常每 10 epoch 保存
    # 持久 checkpoint，并按 eval_freq 用 EMA1 生成 50k 图像在线评估。
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, args=args)

        # Save final checkpoint periodically for easier resuming
        if (epoch + 1) % args.save_last_freq == 0 or (epoch + 1) == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=(epoch + 1),
                epoch_name="last"
            )

        # Every save_interval epochs, we save a persistent checkpoint
        if (epoch + 1) % args.save_interval == 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=(epoch + 1)
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and ((epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, args, (epoch + 1), batch_size=args.gen_bsz)
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)

if __name__ == '__main__':
    raise RuntimeError("Use main_dino_jit.py as the training/sampling entry point.")
