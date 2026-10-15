#!/usr/bin/env bash
set -euo pipefail

project=/data0/zhangruobei/Work/Pixel_Diffusion/Ours_BaseT_Residual_1008
experiment=Ours-J18-new-drop0.1-l4-BaseT-Residual-B-80

config="$project/configs/$experiment.yaml"
checkpoint="$project/checkpoints/$experiment/checkpoint-20.pth"
eval_dir="$project/checkpoints/eval_checkpoint/$experiment/checkpoint-20_cfg2p5_interval0p0_1p0_flowdpm100_image50000"

exec bash "$project/scripts/sample_and_evaluate_fdr6.sh" "$config" "$checkpoint" "$eval_dir" "$@"
