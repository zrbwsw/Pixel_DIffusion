#!/usr/bin/env bash
set -euo pipefail

# Keep the existing DINOv3/JiT-only 20-epoch experiment, changing only the
# distribution of global batch 512 from 4x128 to 8x64. No training on preflight.
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
export PYTHON_BIN="${PYTHON_BIN:-/data0/zhangruobei/anaconda3/envs/irepa/bin/python}"

exec bash "$project_root/scripts/train_jit_rmsnorm_irepa.sh" \
    --batch_size 64 \
    --expected_global_batch_size 512 \
    --lr 0.0001 \
    --exp_name jit-rmsnorm-basejit-irepa-dinov3-B-20-gbs512-gpus8 \
    --output_dir checkpoints/jit-rmsnorm-basejit-irepa-dinov3-B-20-gbs512-gpus8 \
    "$@"
