#!/usr/bin/env bash
# PixelDiT_ours 训练共用的环境、预检、日志和前台执行逻辑；不创建 tmux。

setup_training() {
    export TZ="${TZ:-Asia/Shanghai}"
    TRAIN_SCRIPT="$(realpath -- "${BASH_SOURCE[1]}")"
    ORIGINAL_ARGS=("$@")
    TRAIN_EXTRA_ARGS=()
    TRAIN_MODE=run
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run) TRAIN_MODE=dry_run ;;
            --preflight) TRAIN_MODE=preflight ;;
            --help|-h)
                echo "用法：bash $TRAIN_SCRIPT [--dry-run|--preflight] [Lightning 参数...]"
                echo "环境：GPU_HOST=agent-gpu|agent-gpu2|local，CUDA_VISIBLE_DEVICES，OUTPUT_ROOT，PYTHON_BIN"
                echo "默认前台执行，请先自行创建同名 tmux；--dry-run 不连接 GPU、不运行模型。"
                exit 0 ;;
            *) TRAIN_EXTRA_ARGS+=("$1") ;;
        esac
        shift
    done

    local task_root
    task_root="$(dirname -- "$(dirname -- "$PROJECT_ROOT")")"
    export PYTHON_BIN="${PYTHON_BIN:-$task_root/envs/pixeldit-ours-py310-cu126-20261029/bin/python}"
    [[ -x "$PYTHON_BIN" ]] || { echo "Python 不存在：$PYTHON_BIN" >&2; return 1; }
    export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
    export REPOSITORY_ROOT="$(dirname -- "$PROJECT_ROOT")"
    export PRETRAIN_MODELS_ROOT="${PRETRAIN_MODELS_ROOT:-$REPOSITORY_ROOT/pretrain_models}"
    export IMAGENET_ROOT="${IMAGENET_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/imagenet}"
    export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/train_logs}"
    PRETRAIN_MODELS_ROOT="$(realpath -m -- "$PRETRAIN_MODELS_ROOT")"
    IMAGENET_ROOT="$(realpath -m -- "$IMAGENET_ROOT")"
    OUTPUT_ROOT="$(realpath -m -- "$OUTPUT_ROOT")"
    RUN_DIR="$OUTPUT_ROOT/exp_$EXPERIMENT"

    export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
    # 按当前实验约定统一使用 W&B offline，不依赖 GPU 节点联网。
    export WANDB_MODE=offline
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" TORCHINDUCTOR_COMPILE_THREADS=2 MAX_JOBS=2

    IFS=, read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
    [[ "${#gpus[@]}" == "$NPROC_PER_NODE" ]] || {
        echo "此配方需要 $NPROC_PER_NODE 张卡，CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
        return 2
    }
    cd -- "$PROJECT_ROOT/c2i"
}

check_training_resources() {
    local required=(
        "$CONFIG"
        "$IMAGENET_ROOT/train"
        "$PRETRAIN_MODELS_ROOT/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        "$PROJECT_ROOT/pretrain_models/torch_hub/facebookresearch_dinov2_main/hubconf.py"
        "$PROJECT_ROOT/pretrain_models/torch_hub/checkpoints/dinov2_vitb14_pretrain.pth"
    )
    local path
    for path in "${required[@]}"; do
        [[ -e "$path" ]] || { echo "缺少资源：$path" >&2; return 1; }
    done
    "$PYTHON_BIN" -c 'import h5py, jsonargparse, lightning, psutil, timm, torch, wandb; print("Python/deps:", lightning.__version__, torch.__version__)'
}

run_training() {
    local command=("$@" "${TRAIN_EXTRA_ARGS[@]}")
    if [[ "$TRAIN_MODE" == dry_run ]]; then
        printf '%q ' "${command[@]}"
        printf '\nRUN_DIR=%q\n' "$RUN_DIR"
        return
    fi

    if [[ "$TRAIN_MODE" == run && "${GPU_HOST:-local}" != local &&
          "${TRAIN_ON_GPU:-0}" != 1 ]] && ! nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
        exec bash "$PROJECT_ROOT/scripts/run_on_gpu.sh" "$GPU_HOST" \
            env TRAIN_ON_GPU=1 PYTHON_BIN="$PYTHON_BIN" \
            CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
            OUTPUT_ROOT="$OUTPUT_ROOT" PRETRAIN_MODELS_ROOT="$PRETRAIN_MODELS_ROOT" \
            IMAGENET_ROOT="$IMAGENET_ROOT" WANDB_MODE="$WANDB_MODE" \
            bash "$TRAIN_SCRIPT" "${ORIGINAL_ARGS[@]}"
    fi

    check_training_resources
    local train_args=() found=false arg
    for arg in "${command[@]}"; do
        if "$found"; then train_args+=("$arg"); fi
        [[ "${arg##*/}" != main.py ]] || found=true
    done
    "$found" || { echo '训练命令缺少 c2i/main.py' >&2; return 2; }
    if [[ "$TRAIN_MODE" == preflight ]]; then
        "$PYTHON_BIN" "$PROJECT_ROOT/c2i/main.py" "${train_args[@]}" --print_config > /dev/null
        printf 'Preflight OK\nCONFIG=%s\nIMAGENET_ROOT=%s\nRUN_DIR=%s\n' "$CONFIG" "$IMAGENET_ROOT" "$RUN_DIR"
        return
    fi

    nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=index,name --format=csv,noheader
    mkdir -p -- "$RUN_DIR"
    exec 9>"$RUN_DIR/.training.lock"
    flock -n 9 || { echo "该输出目录已有训练：$RUN_DIR" >&2; return 1; }
    printf '%q ' "${command[@]}" > "$RUN_DIR/train_args.txt"
    printf '\n' >> "$RUN_DIR/train_args.txt"
    date -Is > "$RUN_DIR/started_at.txt"
    TRAIN_OUTPUT="$RUN_DIR"
    trap 'rc=$?; trap - EXIT; printf "%s\n" "$rc" > "$TRAIN_OUTPUT/exit_code.txt"; date -Is > "$TRAIN_OUTPUT/finished_at.txt"; exit "$rc"' EXIT
    "${command[@]}" 2>&1 | tee -a "$RUN_DIR/train.log"
}
