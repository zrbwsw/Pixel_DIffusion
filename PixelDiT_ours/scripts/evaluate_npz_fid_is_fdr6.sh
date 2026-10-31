#!/usr/bin/env bash
# 离线评测 PixelDiT 采样 NPZ：FID、Inception Score、FDR6。
# 只需设置 NPZ；使用同级 nanogen-evals 的权重和 ImageNet-256 参考统计。
set -euo pipefail
export TZ="${TZ:-Asia/Shanghai}"

# ======================== 评测配置 ========================
NPZ="${NPZ:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/PixelDiT_ours/train_logs/exp_pixeldit_ours_xl_dino_ycond_sigmoid_norm_k20_a0p1_bs256/val_256_xl_online_flow100_scale3.25_interval0_1_1/iter_400000/output.npz}"  # 采样脚本输出的 samples.npz 或 output.npz
METRICS="${METRICS:-fid,inception_score,fdr6}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"
GPU_HOST="${GPU_HOST:-agent-gpu2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUTPUT_JSON="${OUTPUT_JSON:-}"             # 默认按 iter 和采样配置写入实验目录的 eval 分组
SETTING_TAG="${SETTING_TAG:-flow100_3.25_0.1_1}"
# ==========================================================

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY_ROOT="$(dirname -- "$PROJECT_ROOT")"
TASK_ROOT="$(dirname -- "$REPOSITORY_ROOT")"
PYTHON_BIN="${PYTHON_BIN:-$TASK_ROOT/envs/pixeldit-ours-py310-cu126-20261029/bin/python}"
NANOGEN_EVALS_ROOT="${NANOGEN_EVALS_ROOT:-$REPOSITORY_ROOT/nanogen-evals}"

usage() {
    cat <<'EOF'
用法：NPZ=/path/to/samples.npz bash scripts/evaluate_npz_fid_is_fdr6.sh [--dry-run|--preflight]
默认计算 fid,inception_score,fdr6，单卡评测；标准 val 结果写入实验目录/eval/<iter>_<sampler>_<scale>_<interval>_fdr6/offline_metrics.json。
可选环境变量：METRICS、BATCH_SIZE、DEVICE、OUTPUT_JSON、SETTING_TAG、NANOGEN_EVALS_ROOT、GPU_HOST、PYTHON_BIN。
输入要求：NPZ 内含 uint8 RGB NHWC 的 arr_0 或 images，ImageNet-256 使用 256x256 图像。
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
[[ -n "$NPZ" ]] || { echo '必须设置 NPZ=/path/to/samples.npz' >&2; usage >&2; exit 2; }
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo 'BATCH_SIZE 必须是正整数' >&2; exit 2; }
NPZ="$(realpath -m -- "$NPZ")"
NANOGEN_EVALS_ROOT="$(realpath -m -- "$NANOGEN_EVALS_ROOT")"
if [[ "$NPZ" == *"/val_256_xl/iter_"* || "$NPZ" == *"/val_256_xl_online_"*"/iter_"* ]]; then
    iter_tag="$(basename -- "$(dirname -- "$NPZ")")"
    experiment_dir="$(dirname -- "$(dirname -- "$(dirname -- "$NPZ")")")"
    default_output_json="$experiment_dir/eval/${iter_tag}_${SETTING_TAG}_fdr6/offline_metrics.json"
elif [[ "$NPZ" == */val_256_xl/predict/output.npz ]]; then
    sample_output_dir="$(dirname -- "$(dirname -- "$(dirname -- "$(dirname -- "$NPZ")")")")"
    case "$sample_output_dir" in
        *_fdr6) ;;
        *_jit|*_adm) sample_output_dir="${sample_output_dir%_*}_fdr6" ;;
        *) sample_output_dir="${sample_output_dir}_fdr6" ;;
    esac
    default_output_json="$sample_output_dir/offline_metrics.json"
else
    default_output_json="$(dirname -- "$NPZ")/offline_metrics.json"
fi
OUTPUT_JSON="${OUTPUT_JSON:-$default_output_json}"
OUTPUT_JSON="$(realpath -m -- "$OUTPUT_JSON")"
LOG="$(dirname -- "$OUTPUT_JSON")/offline_metrics.log"
command=(
    "$PYTHON_BIN" -m evaluation.evaluate_fdr6 --npz="$NPZ" --metrics="$METRICS"
    --cache-root="$NANOGEN_EVALS_ROOT" --device="$DEVICE" --batch-size="$BATCH_SIZE"
    --output-json="$OUTPUT_JSON"
)

if [[ "$mode" == dry_run ]]; then
    printf '%q ' "${command[@]}"; printf '\nOUTPUT_JSON=%q\n' "$OUTPUT_JSON"
    exit 0
fi
[[ -x "$PYTHON_BIN" ]] || { echo "Python 不存在：$PYTHON_BIN" >&2; exit 1; }
[[ -f "$NPZ" ]] || { echo "NPZ 不存在：$NPZ" >&2; exit 1; }
[[ -d "$NANOGEN_EVALS_ROOT" ]] || { echo "nanogen-evals 不存在：$NANOGEN_EVALS_ROOT" >&2; exit 1; }

export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$PROJECT_ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export NANOGEN_EVALS_ROOT HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

if [[ "$mode" == preflight ]]; then
    "$PYTHON_BIN" -m evaluation.evaluate_fdr6 --check-assets --metrics="$METRICS" \
        --cache-root="$NANOGEN_EVALS_ROOT"
    printf 'Preflight OK\nNPZ=%s\nOUTPUT_JSON=%s\n' "$NPZ" "$OUTPUT_JSON"
    exit 0
fi

if ! "$worker" && [[ "$DEVICE" != cpu ]]; then
    exec bash "$PROJECT_ROOT/scripts/run_on_gpu.sh" "$GPU_HOST" env \
        TZ="$TZ" NPZ="$NPZ" METRICS="$METRICS" BATCH_SIZE="$BATCH_SIZE" DEVICE="$DEVICE" \
        GPU_HOST="$GPU_HOST" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        OUTPUT_JSON="$OUTPUT_JSON" NANOGEN_EVALS_ROOT="$NANOGEN_EVALS_ROOT" PYTHON_BIN="$PYTHON_BIN" \
        bash "$PROJECT_ROOT/scripts/evaluate_npz_fid_is_fdr6.sh" --worker
fi

if [[ -f "$OUTPUT_JSON" ]]; then
    printf '评测结果已存在：%s\n' "$OUTPUT_JSON"
    cat "$OUTPUT_JSON"
    exit 0
fi
mkdir -p -- "$(dirname -- "$OUTPUT_JSON")"
exec 9>"$(dirname -- "$OUTPUT_JSON")/.evaluation.lock"
flock -n 9 || { echo "该评测已在运行：$OUTPUT_JSON" >&2; exit 1; }
printf '%q ' "${command[@]}" > "$(dirname -- "$OUTPUT_JSON")/evaluation_args.txt"
printf '\n' >> "$(dirname -- "$OUTPUT_JSON")/evaluation_args.txt"
"${command[@]}" 2>&1 | tee -a "$LOG"
printf '评测完成\nRESULT=%s\n' "$OUTPUT_JSON"
