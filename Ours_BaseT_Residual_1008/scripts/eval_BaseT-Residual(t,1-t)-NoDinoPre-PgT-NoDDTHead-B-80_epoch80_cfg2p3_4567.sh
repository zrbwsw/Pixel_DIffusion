#!/usr/bin/env bash
set -euo pipefail

project=/data0/zhangruobei/Work/Pixel_Diffusion/Ours_BaseT_Residual_1008
experiment='BaseT-Residual(t,1-t)-NoDinoPre-PgT-NoDDTHead-B-80'
config="$project/configs/$experiment.yaml"
checkpoint="$project/checkpoints/$experiment/checkpoint-80.pth"
eval_dir="$project/checkpoints/eval_checkpoint/$experiment/checkpoint-80_cfg2p3_interval0p0_1p0_flowdpm100_image50000"

mkdir -p "$eval_dir"
bash "$project/scripts/sample_and_evaluate_fdr6.sh" \
    "$config" "$checkpoint" "$eval_dir" \
    --gpus=4,5,6,7 --sampler=flow_dpm_solver --steps=100 --cfg=2.3 \
    --interval-min=0.0 --interval-max=1.0 --num-images=50000 --gen-bsz=128 \
    2>&1 | tee -a "$eval_dir/launcher.log"
