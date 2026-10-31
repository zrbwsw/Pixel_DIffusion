#!/usr/bin/env bash
# GPU2 八卡：PixelDiT-XL + DinoYCond + 归一化 sigmoid K20、a=0.1。
# 默认 (DINO,raw)=(1-f(t),f(t))；reverse 时为 (f(t),1-f(t))。
set -euo pipefail

# 1. 路径与资源：开发机无 GPU 时由 run_on_gpu.sh 转到 GPU_HOST。
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_HOST="${GPU_HOST:-agent-gpu2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE=8
EXPERIMENT='pixeldit_ours_xl_dinov3b_ycond_sigmoid_norm_k20_a0p1_bs256'
CONFIG="$PROJECT_ROOT/c2i/configs/pix256_xl_dinov3b_ycond_sigmoid_k20_a0p1_norm.yaml"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/train_logs}"
source "$PROJECT_ROOT/scripts/train_env.sh"
setup_training "$@"

# 2. PixelDiT 默认训练：八卡 × 每卡32 = 全局256，有效 LR=1e-4。
# DINO 细项以 YAML 为准：MLS、y cond、LoRA64、归一化 sigmoid K20/a0.1。
run_training "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --node_rank=0 \
    --nproc_per_node="$NPROC_PER_NODE" \
    "$PROJECT_ROOT/c2i/main.py" fit -c "$CONFIG" \
    --trainer.num_nodes=1 --trainer.devices="$NPROC_PER_NODE" \
    --trainer.default_root_dir="$OUTPUT_ROOT" \
    --trainer.logger.init_args.save_dir="$RUN_DIR" \
    --data.train_dataset.init_args.data_dir="$IMAGENET_ROOT"
