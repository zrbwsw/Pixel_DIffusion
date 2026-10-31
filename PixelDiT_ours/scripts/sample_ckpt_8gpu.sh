#!/usr/bin/env bash
# 通用 PixelDiT v1 采样：通常只改下面的 CHECKPOINT、SAMPLER、CFG 和 interval。
# 在开发机前台执行；无 GPU 时自动连接 GPU_HOST。--dry-run 只检查命令，不采样。
set -euo pipefail
export TZ="${TZ:-Asia/Shanghai}"

# ======================== 采样配置 ========================
CHECKPOINT="${CHECKPOINT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/PixelDiT_ours/train_logs/exp_pixeldit_ours_xl_dino_ycond_sigmoid_norm_k20_a0p1_bs256/epoch=79-step=400000.ckpt}"
SAMPLER="${SAMPLER:-flow_dpm_solver}"   # flow_dpm_solver、heun 或 euler
CFG="${CFG:-3.25}"
INTERVAL_MIN="${INTERVAL_MIN:-0.1}"
INTERVAL_MAX="${INTERVAL_MAX:-1.0}"
STEPS="${STEPS:-100}"
CONFIG="${CONFIG:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/PixelDiT_ours/c2i/configs/pix256_xl_dino_ycond_sigmoid_k20_a0p1_norm.yaml}"
NUM_IMAGES="${NUM_IMAGES:-50000}"       # 少于1000时上游数据集仍按每类1张生成1000张
PRED_BSZ="${PRED_BSZ:-32}"              # 每张 GPU 的采样 batch size
SEED="${SEED:-5000}"
TIME_SHIFT="${TIME_SHIFT:-1.0}"
EVAL_ORIGINAL_MODEL="${EVAL_ORIGINAL_MODEL:-false}"  # false=EMA；true=当前训练权重
GPU_HOST="${GPU_HOST:-agent-gpu2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
OUTPUT_DIR="${OUTPUT_DIR:-}"             # 留空则按采样参数自动命名
# ==========================================================

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY_ROOT="$(dirname -- "$PROJECT_ROOT")"
TASK_ROOT="$(dirname -- "$REPOSITORY_ROOT")"
PYTHON_BIN="${PYTHON_BIN:-$TASK_ROOT/envs/pixeldit-ours-py310-cu126-20261029/bin/python}"
export PRETRAIN_MODELS_ROOT="${PRETRAIN_MODELS_ROOT:-$REPOSITORY_ROOT/pretrain_models}"

usage() {
    cat <<'EOF'
用法：bash scripts/sample_ckpt_8gpu.sh [--dry-run|--preflight]
通过顶部变量或同名环境变量设置 CHECKPOINT、SAMPLER、CFG、INTERVAL_MIN/MAX。
默认：80 epoch EMA、FlowDPM100、CFG=3.25、interval=[0.1,1.0]、50K 图像。
带 step 的实验 checkpoint 默认输出到实验目录/eval/iter_<step>_<sampler><steps>_<scale>_<interval>/。
其他可选变量：STEPS、CONFIG、NUM_IMAGES、PRED_BSZ、SEED、TIME_SHIFT、OUTPUT_DIR、GPU_HOST。
EOF
}

mode=run
worker=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) mode=dry_run ;;
        --preflight) mode=preflight ;;
        --worker) worker=true ;;
        --help|-h) usage; exit 0 ;;
        *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

case "${SAMPLER,,}" in
    flow_dpm_solver|flowdpm|flow)
        SAMPLER_CLASS='src.diffusion.FlowDPMSolverSampler'; SAMPLER_TAG='flow' ;;
    heun)
        SAMPLER_CLASS='src.diffusion.HeunSampler'; SAMPLER_TAG='heun' ;;
    euler)
        SAMPLER_CLASS='src.diffusion.EulerSampler'; SAMPLER_TAG='euler' ;;
    *) echo "未知 SAMPLER=$SAMPLER；可选 flow_dpm_solver、heun、euler" >&2; exit 2 ;;
esac
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "STEPS 必须是正整数" >&2; exit 2; }
[[ "$NUM_IMAGES" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_IMAGES 必须是正整数" >&2; exit 2; }
[[ "$PRED_BSZ" =~ ^[1-9][0-9]*$ ]] || { echo "PRED_BSZ 必须是正整数" >&2; exit 2; }
IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE="${#GPUS[@]}"

CHECKPOINT="$(realpath -m -- "$CHECKPOINT")"
CONFIG="$(realpath -m -- "$CONFIG")"
[[ -x "$PYTHON_BIN" ]] || { echo "Python 不存在：$PYTHON_BIN" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "配置不存在：$CONFIG" >&2; exit 1; }
EXP_TAG="$("$PYTHON_BIN" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["tags"]["exp"])' "$CONFIG")"
CKPT_TAG="$(basename -- "$CHECKPOINT" .ckpt)"
tag_number() {
    local value="$1"
    if [[ "$value" == *.* ]]; then
        while [[ "$value" == *0 ]]; do value="${value%0}"; done
        value="${value%.}"
    fi
    printf '%s' "$value"
}
SETTING_TAG="${SAMPLER_TAG}${STEPS}_$(tag_number "$CFG")_$(tag_number "$INTERVAL_MIN")_$(tag_number "$INTERVAL_MAX")"
if [[ -z "$OUTPUT_DIR" ]]; then
    if [[ "$CHECKPOINT" == "$PROJECT_ROOT/train_logs/exp_"* && "$CKPT_TAG" =~ step=([0-9]+)$ ]]; then
        OUTPUT_DIR="$(dirname -- "$CHECKPOINT")/eval/iter_${BASH_REMATCH[1]}_${SETTING_TAG}"
    else
        OUTPUT_DIR="$PROJECT_ROOT/eval_checkpoint/${CKPT_TAG}_${SETTING_TAG}_images${NUM_IMAGES}"
    fi
fi
OUTPUT_DIR="$(realpath -m -- "$OUTPUT_DIR")"
NPZ="$OUTPUT_DIR/exp_${EXP_TAG}/val_256_xl/predict/output.npz"

command=(
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --node_rank=0
    --nproc_per_node="$NPROC_PER_NODE" "$PROJECT_ROOT/c2i/main.py" predict -c "$CONFIG"
    --ckpt_path="$CHECKPOINT"
    --trainer.num_nodes=1 --trainer.devices="$NPROC_PER_NODE"
    --trainer.default_root_dir="$OUTPUT_DIR" --trainer.logger=false
    --data.pred_dataset.init_args.max_num_instances="$NUM_IMAGES"
    --data.pred_batch_size="$PRED_BSZ" --data.pred_num_workers=1
    --model.diffusion_sampler.class_path="$SAMPLER_CLASS"
    --model.diffusion_sampler.init_args.num_steps="$STEPS"
    --model.diffusion_sampler.init_args.guidance="$CFG"
    --model.diffusion_sampler.init_args.timeshift="$TIME_SHIFT"
    --model.diffusion_sampler.init_args.guidance_interval_min="$INTERVAL_MIN"
    --model.diffusion_sampler.init_args.guidance_interval_max="$INTERVAL_MAX"
    --model.eval_original_model="$EVAL_ORIGINAL_MODEL"
    --per_run_seed=false --seed_everything="$SEED"
)

if [[ "$mode" == dry_run ]]; then
    printf '%q ' "${command[@]}"; printf '\nNPZ=%q\nOUTPUT_DIR=%q\n' "$NPZ" "$OUTPUT_DIR"
    exit 0
fi

[[ -f "$CHECKPOINT" ]] || { echo "checkpoint 不存在：$CHECKPOINT" >&2; exit 1; }
[[ -f "$PRETRAIN_MODELS_ROOT/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" ]] || {
    echo "缺少 DINOv3-B/16 权重：$PRETRAIN_MODELS_ROOT" >&2; exit 1;
}
CHECKPOINT="$(realpath -- "$CHECKPOINT")"

if [[ "$mode" == preflight ]]; then
    cd -- "$PROJECT_ROOT/c2i"
    PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "${command[@]:7}" --print_config > /dev/null
    printf 'Preflight OK\nCHECKPOINT=%s\nSAMPLER=%s steps=%s CFG=%s interval=[%s,%s]\nNPZ=%s\n' \
        "$CHECKPOINT" "$SAMPLER_CLASS" "$STEPS" "$CFG" "$INTERVAL_MIN" "$INTERVAL_MAX" "$NPZ"
    exit 0
fi

if ! "$worker"; then
    exec bash "$PROJECT_ROOT/scripts/run_on_gpu.sh" "$GPU_HOST" env \
        TZ="$TZ" CHECKPOINT="$CHECKPOINT" CONFIG="$CONFIG" SAMPLER="$SAMPLER" \
        CFG="$CFG" INTERVAL_MIN="$INTERVAL_MIN" INTERVAL_MAX="$INTERVAL_MAX" \
        STEPS="$STEPS" NUM_IMAGES="$NUM_IMAGES" PRED_BSZ="$PRED_BSZ" SEED="$SEED" \
        TIME_SHIFT="$TIME_SHIFT" EVAL_ORIGINAL_MODEL="$EVAL_ORIGINAL_MODEL" \
        GPU_HOST="$GPU_HOST" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        OUTPUT_DIR="$OUTPUT_DIR" PYTHON_BIN="$PYTHON_BIN" PRETRAIN_MODELS_ROOT="$PRETRAIN_MODELS_ROOT" \
        bash "$PROJECT_ROOT/scripts/sample_ckpt_8gpu.sh" --worker
fi

export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" TORCHINDUCTOR_COMPILE_THREADS=2 MAX_JOBS=2

if [[ -f "$NPZ" ]]; then
    if [[ ! -e "$OUTPUT_DIR/samples.npz" && ! -L "$OUTPUT_DIR/samples.npz" ]]; then
        ln -s "$NPZ" "$OUTPUT_DIR/samples.npz"
    fi
    printf '采样已完成\nNPZ=%s\n便捷链接=%s\n' "$NPZ" "$OUTPUT_DIR/samples.npz"
    exit 0
fi
PREDICT_DIR="$(dirname -- "$NPZ")"
if [[ -d "$PREDICT_DIR" ]] && [[ -n "$(ls -A "$PREDICT_DIR")" ]]; then
    echo "检测到未完成采样目录；请换 OUTPUT_DIR 或手动处理：$PREDICT_DIR" >&2
    exit 2
fi
mkdir -p -- "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/.sampling.lock"
flock -n 9 || { echo "该采样已在运行：$OUTPUT_DIR" >&2; exit 1; }
printf '%q ' "${command[@]}" > "$OUTPUT_DIR/sampling_args.txt"
printf '\nNPZ=%s\n' "$NPZ" >> "$OUTPUT_DIR/sampling_args.txt"
cd -- "$PROJECT_ROOT/c2i"
"${command[@]}" 2>&1 | tee -a "$OUTPUT_DIR/sampling.log"
[[ -f "$NPZ" ]] || { echo "采样结束但未找到 NPZ：$NPZ" >&2; exit 1; }
if [[ ! -e "$OUTPUT_DIR/samples.npz" && ! -L "$OUTPUT_DIR/samples.npz" ]]; then
    ln -s "$NPZ" "$OUTPUT_DIR/samples.npz"
fi
printf '采样完成\nNPZ=%s\n便捷链接=%s\n' "$NPZ" "$OUTPUT_DIR/samples.npz"
