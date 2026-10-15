#!/usr/bin/env bash
set -euo pipefail

cd /data0/zhangruobei/Work/Pixel_Diffusion/Ours_BaseT_Residual_1008
python=/data0/zhangruobei/anaconda3/envs/irepa/bin/python
output=analysis/token_norms_80epoch/train_1024
mkdir -p "$output"

pids=()
for rank in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$rank" "$python" -u analyze_token_norms.py \
        --config='configs/BaseT-Residual(t,1-t)-NoDinoPre-PgT-B-80.yaml' \
        --checkpoint='checkpoints/BaseT-Residual(t,1-t)-NoDinoPre-PgT-B-80/checkpoint-80.pth' \
        --output-dir="$output" --data_path=/data0/datasets/imagenet \
        --num-samples=1024 --batch-size=4 --t-points=21 --seed=1234 \
        --rank="$rank" --world-size=4 > "$output/rank${rank}.log" 2>&1 &
    pids+=("$!")
done

failed=0
for rank in 0 1 2 3; do
    if ! wait "${pids[$rank]}"; then
        echo "rank $rank failed; inspect $output/rank${rank}.log" >&2
        failed=1
    fi
done
exit "$failed"
