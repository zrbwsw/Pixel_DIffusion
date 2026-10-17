#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
experiment='BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80'

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHON_BIN="${PYTHON_BIN:-python}"
export PRETRAIN_MODELS_ROOT="${PRETRAIN_MODELS_ROOT:-$(dirname -- "$project_root")/pretrain_models}"
export FID_STATS_ROOT="${FID_STATS_ROOT:-$(dirname -- "$project_root")/fid_stats}"
export NANOGEN_EVALS_ROOT="${NANOGEN_EVALS_ROOT:-$(dirname -- "$project_root")/nanogen-evals}"
export IMAGENET_ROOT="${IMAGENET_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/imagenet}"
export DINOV3_REPO="$project_root/SSL_code/dinov3"

# Eight H200 GPUs: 128 * 8 = 1024; lr = 5e-5 * 1024 / 256 = 2e-4.
exec bash "$project_root/scripts/train_portable.sh" \
    "$project_root/configs/${experiment}.yaml" \
    --batch_size 128 --global_batch_size 1024 --accum_iter 1 --blr 0.00005 "$@"
