#!/usr/bin/env bash
# agent-gpu H200-0029: epoch 80, EMA1, FlowDPM-100, CFG 2.7, 50K images.
set -euo pipefail
mode=tmux
for arg in "$@"; do
    case "$arg" in
        --preflight) mode=preflight ;;
        --foreground) mode=foreground ;;
        *) echo "Usage: $0 [--preflight|--foreground]" >&2; exit 2 ;;
    esac
done
script="$(realpath -- "${BASH_SOURCE[0]}")"
project="$(cd -- "$(dirname -- "$script")/.." && pwd)"
task_root="$(dirname -- "$(dirname -- "$project")")"
experiment='BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80'
checkpoint="$project/checkpoints/$experiment/checkpoint-80.pth"
config="$project/configs/$experiment.yaml"
output="$project/checkpoints/eval_checkpoint/$experiment/checkpoint-80_cfg2p7_interval0p0_1p0_flowdpm100_image50000_gpus4_bsz256"
session=eval-rmsnorm2-B80-cfg2p7-4567-b256
work_dir="$task_root/work/$session"
export PYTHON_BIN="$task_root/envs/irepa-py310-cu126-20261015/bin/python"
export CUDA_VISIBLE_DEVICES=4,5,6,7
source "$project/scripts/portable_env.sh"
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
export DINOV3_REPO="$project/SSL_code/dinov3"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2 MAX_JOBS=2
export TORCHINDUCTOR_CACHE_DIR=/tmp/agent_z_rmsnorm2_e80_cfg2p7_b256
export TRITON_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}_triton"
cd -- "$project"
mkdir -p "$work_dir"

preflight() {
    "$PYTHON_BIN" - "$checkpoint" "$experiment" "$work_dir" <<'PY'
import json, math, sys, torch
from pathlib import Path
path, experiment, work = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
args = vars(checkpoint['args'])
assert checkpoint['epoch'] == 80
expected = dict(exp_name=experiment, model='JiT-B/16', global_batch_size=1024,
                accum_iter=1, dino_model_name='dinov3_vitb16',
                dino_jit_input_mode='time_blend', dino_input_projector='pixeldit2',
                dino_jit_pre_blend_rmsnorm=True, use_ddt_head=False)
for key, value in expected.items():
    assert args[key] == value, (key, args[key], value)
assert math.isclose(args['lr'], 2e-4)
assert math.isclose(args['ema_decay1'], 0.9999)
for key in ('model', 'model_ema1', 'model_ema2'):
    state = checkpoint[key]
    assert state
    assert 'dino_pre_blend_norm.weight' in state
    assert 'jit_pre_blend_norm.weight' in state
report = dict(checkpoint=str(path), epoch=80, training_global_batch=1024,
              training_lr=args['lr'], ema='ema1', ema_decay=args['ema_decay1'],
              size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
(work/'checkpoint_validation.json').write_text(json.dumps(report, indent=2)+'\n')
PY
    bash "$project/scripts/sample_portable.sh" "$config" "$checkpoint" "$output" \
        --preflight --cfg=2.7 --sampler=flow_dpm_solver --steps=100 \
        --interval-min=0 --interval-max=1 --num-images=50000 --gen-bsz=256
    echo 'Preflight passed: rmsnorm2 epoch 80, EMA1, GPUs 4-7, gen_bsz 256.'
}

if [[ "$mode" == preflight ]]; then preflight; exit 0; fi
[[ "$(hostname)" == gpu-lg-cmc-h-h200-0029* ]] || { echo 'Run on agent-gpu (H200-0029).' >&2; exit 1; }
if [[ "$mode" == tmux ]]; then
    if tmux has-session -t "=$session" 2>/dev/null; then
        echo "Session already exists: $session" >&2; exit 1
    fi
    preflight
    nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits > "$work_dir/gpu_before.csv"
    "$PYTHON_BIN" - "$work_dir/gpu_before.csv" <<'PY'
import csv, sys
rows = {int(row[0]): row for row in csv.reader(open(sys.argv[1]), skipinitialspace=True)}
assert all(int(rows[index][1]) >= 65536 for index in (4,5,6,7)), 'Need 64 GiB free per selected GPU'
PY
    tmux new-session -d -s "$session" -c "$project" 'bash --noprofile --norc'
    tmux set-window-option -t "$session:0" remain-on-exit on
    printf -v command 'exec bash %q --foreground' "$script"
    tmux send-keys -t "$session:0.0" -l "$command"
    tmux send-keys -t "$session:0.0" Enter
    printf 'Started tmux: %s\nSampling log: %s/sampling.log\n' "$session" "$output"
    exit 0
fi

mkdir -p "$output"
exec 9>"$output/.sampling.lock"
flock -n 9 || { echo "Output is already in use: $output" >&2; exit 1; }
exec > >(tee -a "$output/launcher.log") 2>&1
finish() {
    rc=$?
    trap - EXIT
    printf '%s\n' "$rc" > "$output/exit_code.txt"
    date -Is > "$output/finished_at.txt"
    exit "$rc"
}
trap finish EXIT
rm -f -- "$output/exit_code.txt" "$output/finished_at.txt"
date -Is > "$output/started_at.txt"
printf 'CHECKPOINT=%s\nGPUS=4,5,6,7\nCFG=2.7\nSAMPLER=flow_dpm_solver\nSTEPS=100\nINTERVAL=[0,1]\nGEN_BSZ=256\nNUM_IMAGES=50000\nEMA=ema1\nEMA_DECAY=0.9999\nSEED=0\nTMUX_SESSION=%s\n' \
    "$checkpoint" "$session" > "$output/run_info.txt"
hostname >> "$output/run_info.txt"
cp -- "$work_dir/checkpoint_validation.json" "$output/checkpoint_validation.json"
sha256sum "$config" "$script" dino_jit_denoiser.py jit/engine_jit.py > "$output/source_sha256.txt"
bash "$project/scripts/sample_portable.sh" "$config" "$checkpoint" "$output" \
    --cfg=2.7 --sampler=flow_dpm_solver --steps=100 --interval-min=0 --interval-max=1 \
    --num-images=50000 --gen-bsz=256
