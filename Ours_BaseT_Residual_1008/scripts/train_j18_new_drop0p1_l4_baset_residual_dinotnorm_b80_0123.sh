#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES
export PRETRAIN_MODELS_ROOT=/data0/pretrain_models
export FID_STATS_ROOT=/data0/zhangruobei/Work/Pixel_Diffusion/fid_stats

PROJECT=/data0/zhangruobei/Work/Pixel_Diffusion/Ours_BaseT_Residual_1008
EXPERIMENT=Ours-J18-new-drop0.1-l4-BaseT-Residual-DinoTNorm-B-80
CONFIG="$PROJECT/configs/$EXPERIMENT.yaml"
OUTPUT_DIR="$PROJECT/checkpoints/$EXPERIMENT"
TORCHRUN=/data0/zhangruobei/anaconda3/envs/irepa/bin/torchrun

[[ -x "$TORCHRUN" && -f "$CONFIG" ]] || {
    echo "Missing torchrun or config: $CONFIG" >&2
    exit 1
}
[[ -d /data0/datasets/imagenet/train ]] || {
    echo "Missing ImageNet training data" >&2
    exit 1
}
[[ -s "$PRETRAIN_MODELS_ROOT/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" ]] || {
    echo "Missing DINOv3-B weights" >&2
    exit 1
}
[[ -s "$FID_STATS_ROOT/jit_in256_stats.npz" && -s "$FID_STATS_ROOT/weights-inception-2015-12-05-6726825d.pth" ]] || {
    echo "Missing online JiT FID/IS assets" >&2
    exit 1
}

if [[ "${1-}" == "--preflight" ]]; then
    echo "Preflight passed: $EXPERIMENT on GPUs $CUDA_VISIBLE_DEVICES"
    exit 0
fi
[[ "$#" -eq 0 ]] || { echo "Usage: $0 [--preflight]" >&2; exit 2; }

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT"
"$TORCHRUN" --standalone --nproc_per_node=4 main_dino_jit.py \
    --config "$CONFIG" 2>&1 | tee -a "$OUTPUT_DIR/train.log"
