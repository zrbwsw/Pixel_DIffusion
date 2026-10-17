#!/usr/bin/env bash
# agent-gpu GPUs 4-7: 256 per GPU x4 = global batch 1024, no accumulation.
set -euo pipefail

mode="${1-}"
[[ "$#" -le 1 ]] || { echo "Pass at most one option" >&2; exit 2; }
case "$mode" in
    ''|--foreground|--preflight) ;;
    --help|-h) echo "Usage: $0 [--preflight | --foreground] (default: detached tmux)"; exit 0 ;;
    *) echo "Unknown option: $mode" >&2; exit 2 ;;
esac
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_root="$(dirname -- "$(dirname -- "$project")")"
script="$(realpath -- "${BASH_SOURCE[0]}")"
experiment='BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80'
config="$project/configs/$experiment.yaml"
output="$project/checkpoints/$experiment"
session=train-rmsnorm2-B80-4567-b256-gbs1024
export PYTHON_BIN="${PYTHON_BIN:-$task_root/envs/irepa-py310-cu126-20261015/bin/python}"
export CUDA_VISIBLE_DEVICES=4,5,6,7
source "$project/scripts/portable_env.sh"
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
export DINOV3_REPO="$project/SSL_code/dinov3"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2 MAX_JOBS=2
export TORCHINDUCTOR_CACHE_DIR=/tmp/agent_z_irepa_rmsnorm2_b256_train
export TRITON_CACHE_DIR=/tmp/agent_z_irepa_rmsnorm2_b256_triton
cd -- "$project"
train_args=(--batch_size 256 --global_batch_size 1024 --accum_iter 1 --blr 0.00005)

if [[ "$mode" == --preflight ]]; then
    exec bash "$project/scripts/train_portable.sh" "$config" "${train_args[@]}" --preflight
fi
if [[ -z "$mode" ]]; then
    command -v tmux >/dev/null
    if tmux has-session -t "=$session" 2>/dev/null; then
        echo "Existing tmux session: $session; inspect it before starting again." >&2; exit 1
    fi
    bash "$project/scripts/train_portable.sh" "$config" "${train_args[@]}" --preflight
    "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==4, "Physical GPUs 4-7 must be available"'
    tmux new-session -d -s "$session" -c "$project" 'bash --noprofile --norc'
    tmux set-window-option -t "$session:0" remain-on-exit on
    printf -v command 'exec env PYTHON_BIN=%q bash %q --foreground' "$PYTHON_BIN" "$script"
    tmux send-keys -t "$session:0.0" -l "$command"
    tmux send-keys -t "$session:0.0" Enter
    printf 'Started %s\nLog: %s/train.log\n' "$session" "$output"
    exit 0
fi

mkdir -p "$output"
exec 9>"$output/.training.lock"
flock -n 9 || { echo "Output already in use: $output" >&2; exit 1; }
date -Is > "$output/started_at.txt"
rm -f -- "$output/finished_at.txt" "$output/exit_code.txt"
printf 'GPUS=4,5,6,7\nBATCH_SIZE=256\nGLOBAL_BATCH_SIZE=1024\nACCUM_ITER=1\nBLR=0.00005\nEFFECTIVE_LR=0.0002\nEPOCHS=80\nTMUX_SESSION=%s\n' "$session" > "$output/run_info.txt"
hostname >> "$output/run_info.txt"
cp -- "$config" "$output/config_base.yaml"
printf '%q ' "$PYTHON_BIN" main_dino_jit.py --config "$config" "${train_args[@]}" > "$output/train_args.txt"
printf '\n' >> "$output/train_args.txt"
set +e
bash "$project/scripts/train_portable.sh" "$config" "${train_args[@]}" 2>&1 | tee -a "$output/launcher.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > "$output/exit_code.txt"
date -Is > "$output/finished_at.txt"
exit "$status"
