#!/usr/bin/env bash
set -euo pipefail

project=/data0/zhangruobei/Work/Pixel_Diffusion/Ours_BaseT_Residual_1008
experiment='BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm-NoDDTHead-L-80'
config="$project/configs/${experiment}.yaml"
checkpoint="$project/checkpoints/${experiment}/checkpoint-20.pth"
eval_dir="$project/checkpoints/eval_checkpoint/${experiment}/checkpoint-20_cfg2p3_interval0p0_1p0_flowdpm100_image50000"
sample_script="$project/scripts/sample_and_evaluate_fdr6.sh"
train_script="$project/scripts/train_BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm-NoDDTHead-L-80_01234567.sh"

[[ -x "$sample_script" ]] || { echo "Missing executable sampling script: $sample_script" >&2; exit 1; }
[[ -x "$train_script" ]] || { echo "Missing executable training script: $train_script" >&2; exit 1; }
[[ -f "$config" && -s "$checkpoint" ]] || { echo "Missing config or checkpoint" >&2; exit 1; }

mkdir -p "$eval_dir"
echo "[$(date '+%F %T')] Starting L checkpoint-20 CFG sampling on GPUs 4,5,6,7"
bash "$sample_script" \
    "$config" "$checkpoint" "$eval_dir" \
    --gpus=4,5,6,7 \
    --sampler=flow_dpm_solver \
    --steps=100 \
    --cfg=2.3 \
    --interval-min=0.0 \
    --interval-max=1.0 \
    --num-images=50000 \
    --gen-bsz=32

echo "[$(date '+%F %T')] Sampling and FDR6 evaluation completed; resuming 8-GPU training"
bash "$train_script"
