#!/usr/bin/env bash
# GPU2 四卡：PixelDiT-B + DINOv3-B YCond + 归一化 sigmoid K20、a=0.1。
# 默认 (DINO,raw)=(1-f(t),f(t))；reverse 时为 (f(t),1-f(t))。
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_HOST="${GPU_HOST:-agent-gpu2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE=4
EXPERIMENT='pixeldit_ours_b_dinov3b_ycond_sigmoid_norm_k20_a0.1_bs256'
CONFIG="$PROJECT_ROOT/c2i/configs/pix256_b_dinov3b_ycond_sigmoid_k20_a0.1_norm.yaml"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/train_logs}"
source "$PROJECT_ROOT/scripts/train_env.sh"
setup_training "$@"

# PixelDiT-B 默认 80 epoch：四卡 × 每卡64 = 全局256，有效 LR=1e-4。
run_training "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --node_rank=0 \
    --nproc_per_node="$NPROC_PER_NODE" \
    "$PROJECT_ROOT/c2i/main.py" fit -c "$CONFIG" \
    --trainer.num_nodes=1 --trainer.devices="$NPROC_PER_NODE" \
    --trainer.default_root_dir="$OUTPUT_ROOT" \
    --trainer.logger.init_args.save_dir="$RUN_DIR" \
    --data.train_dataset.init_args.data_dir="$IMAGENET_ROOT"
