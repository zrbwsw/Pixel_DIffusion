#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES
export PRETRAIN_MODELS_ROOT=/data0/pretrain_models
export FID_STATS_ROOT=/data0/zhangruobei/Work/Pixel_Diffusion/fid_stats

cd /data0/zhangruobei/Work/Pixel_Diffusion/Ours_BaseT_Residual_1008
experiment='BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm-NoDDTHead-L-80'
config="configs/${experiment}.yaml"
output_dir="checkpoints/${experiment}"
torchrun=/data0/zhangruobei/anaconda3/envs/irepa/bin/torchrun

[[ -x "$torchrun" && -f "$config" ]] || { echo "Missing torchrun or config" >&2; exit 1; }
[[ -d /data0/datasets/imagenet/train ]] || { echo "Missing ImageNet training data" >&2; exit 1; }
[[ -s "$PRETRAIN_MODELS_ROOT/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" ]] || { echo "Missing DINOv3-L weights" >&2; exit 1; }
[[ -s "$FID_STATS_ROOT/jit_in256_stats.npz" && -s "$FID_STATS_ROOT/weights-inception-2015-12-05-6726825d.pth" ]] || { echo "Missing online JiT FID/IS assets" >&2; exit 1; }

if [[ "${1-}" == "--preflight" ]]; then
    echo "Preflight passed: $experiment on GPUs $CUDA_VISIBLE_DEVICES"
    exit 0
fi
[[ "$#" -eq 0 ]] || { echo "Usage: $0 [--preflight]" >&2; exit 2; }

mkdir -p "$output_dir"
"$torchrun" --standalone --nproc_per_node=8 main_dino_jit.py \
    --config "$config" \
    --batch_size 32 \
    --global_batch_size 256 \
    --accum_iter 1 2>&1 | tee -a "$output_dir/train.log"
