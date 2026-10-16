#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
experiment='BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80'

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHON_BIN="${PYTHON_BIN:-/data0/zhangruobei/anaconda3/envs/irepa/bin/python}"
export PRETRAIN_MODELS_ROOT="${PRETRAIN_MODELS_ROOT:-/data0/pretrain_models}"
export FID_STATS_ROOT="${FID_STATS_ROOT:-$(dirname -- "$project_root")/fid_stats}"
export NANOGEN_EVALS_ROOT="${NANOGEN_EVALS_ROOT:-$(dirname -- "$project_root")/nanogen-evals}"
export IMAGENET_ROOT="${IMAGENET_ROOT:-/data0/datasets/imagenet}"
export DINOV3_REPO="$project_root/SSL_code/dinov3"

exec bash "$project_root/scripts/train_portable.sh" \
    "$project_root/configs/${experiment}.yaml" "$@"
