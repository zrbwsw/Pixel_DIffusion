#!/usr/bin/env bash
# GPU2 八卡：XL 从160续训到320 epoch，切换 lr=1e-5、gradient clip=0.5。
# 默认 (DINO,raw)=(1-f(t),f(t))；reverse 时为 (f(t),1-f(t))。
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_HOST="${GPU_HOST:-agent-gpu2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE=8
EXPERIMENT='pixeldit_ours_xl_dinov3b_ycond_sigmoid_norm_k20_a0p1_bs256'
CONFIG="$PROJECT_ROOT/c2i/configs/pix256_xl_dinov3b_ycond_sigmoid_k20_a0p1_norm_320.yaml"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/train_logs}"
source "$PROJECT_ROOT/scripts/train_env.sh"
setup_training "$@"

# 2. 第2阶段：八卡 × 每卡32 = 全局256；恢复时将有效 LR 改为1e-5，裁剪改为0.5。
# DINO 细项以 YAML 为准：MLS、y cond、LoRA64、归一化 sigmoid K20/a0.1。
run_training "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --node_rank=0 \
    --nproc_per_node="$NPROC_PER_NODE" \
    "$PROJECT_ROOT/c2i/main.py" fit -c "$CONFIG" \
    --trainer.num_nodes=1 --trainer.devices="$NPROC_PER_NODE" \
    --trainer.default_root_dir="$OUTPUT_ROOT" \
    --trainer.logger.init_args.save_dir="$RUN_DIR" \
    --data.train_dataset.init_args.data_dir="$IMAGENET_ROOT"
