#!/usr/bin/env bash
# H200 agent-gpu-2: controlled comparison with nogamma's global-batch-1024 run.
# Corrected f8dbfc3: generation-only no-affine LN; frozen iREPA retains affine.
# Default creates a detached tmux session. --preflight does not start training.
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
experiment='BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm_nogamma-NoDDTHead-B-80'
config="$project/configs/$experiment.yaml"
output="$project/checkpoints/$experiment"
session=train-noaffine-nogamma-B80-8gpu-b128-gbs1024
work_dir="$task_root/work/$session"
export PYTHON_BIN="${PYTHON_BIN:-$task_root/envs/irepa-py310-cu126-20261015/bin/python}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source "$project/scripts/portable_env.sh"
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
export DINOV3_REPO="$project/SSL_code/dinov3"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2 MAX_JOBS=2
export TORCHINDUCTOR_CACHE_DIR=/tmp/agent_z_noaffine_nogamma_train
export TRITON_CACHE_DIR=/tmp/agent_z_noaffine_nogamma_triton
training_args=(--batch_size 128 --global_batch_size 1024 --accum_iter 1 --blr 0.00005)
cd -- "$project"
mkdir -p "$work_dir"

preflight() {
    "$PYTHON_BIN" scripts/check_runtime.py --world-size 8 --config "$config" "${training_args[@]}" \
        > "$work_dir/resource_preflight.json"
    "$PYTHON_BIN" - "$config" "$output" "$work_dir/resource_preflight.json" <<'PY'
import json, math, sys
from pathlib import Path
from scripts.check_runtime import training_args
args = training_args(['--config',sys.argv[1],'--batch_size','128',
                      '--global_batch_size','1024','--accum_iter','1','--blr','0.00005'])
report = json.loads(Path(sys.argv[3]).read_text())
expected = dict(model='JiT-B/16', dino_model_name='dinov3_vitb16',
                epochs=80, warmup_epochs=5, lr_schedule='constant',
                dino_jit_input_mode='time_blend', dino_input_projector='pixeldit2',
                dino_jit_pre_blend_rmsnorm=True, use_ddt_head=False,
                online_eval=True, eval_freq=20, sampling_method='heun',
                num_sampling_steps=50, cfg=1.0, gen_bsz=64, num_images=50000,
                ema_decay1=0.9999, ema_decay2=0.9996, seed=0)
for key,value in expected.items():
    if getattr(args,key) != value:
        raise ValueError(f'Expected {key}={value!r}, got {getattr(args,key)!r}')
assert report['world_size'] == 8 and report['batch_size_per_gpu'] == 128
assert report['global_batch_size'] == 1024 and report['accum_iter'] == 1
assert math.isclose(report['effective_lr'],2e-4)
assert Path(report['output_dir']) == Path(sys.argv[2])
if report['resume_checkpoint']:
    assert Path(report['resume_checkpoint']).parent == Path(sys.argv[2])
print(json.dumps(report,indent=2))
PY
    echo "Preflight passed: noaffine nogamma, 8 x 128, global batch 1024, lr 2e-4, EMA1 0.9999."
}

if [[ "$mode" == preflight ]]; then preflight; exit 0; fi
[[ "$(hostname)" == gpu-lg-cmc-h-h200-0257* ]] || { echo 'Run this launcher on agent-gpu-2 (H200-0257).' >&2; exit 1; }
preflight
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader,nounits > "$work_dir/gpu_before.csv"
"$PYTHON_BIN" - "$work_dir/gpu_before.csv" <<'PY'
import csv,sys
rows = list(csv.reader(open(sys.argv[1]),skipinitialspace=True))
assert {int(row[0]) for row in rows} == set(range(8)), 'Expected 8 GPUs'
assert all('H200' in row[1] for row in rows), 'Expected H200 GPUs'
assert all(int(row[2]) >= 65536 for row in rows), 'Need at least 64 GiB free per GPU'
PY

if [[ "$mode" == tmux ]]; then
    command -v tmux >/dev/null
    if tmux has-session -t "=$session" 2>/dev/null; then
        echo "Session already exists: $session" >&2; exit 1
    fi
    tmux new-session -d -s "$session" -c "$project" 'bash --noprofile --norc'
    tmux set-window-option -t "$session:0" remain-on-exit on
    # Pin the resolved interpreter even if the tmux server predates this environment.
    printf -v command 'exec env PYTHON_BIN=%q PRETRAIN_MODELS_ROOT=%q FID_STATS_ROOT=%q NANOGEN_EVALS_ROOT=%q IMAGENET_ROOT=%q bash %q --foreground' \
        "$PYTHON_BIN" "$PRETRAIN_MODELS_ROOT" "$FID_STATS_ROOT" "$NANOGEN_EVALS_ROOT" "$IMAGENET_ROOT" "$script"
    tmux send-keys -t "$session:0.0" -l "$command"
    tmux send-keys -t "$session:0.0" Enter
    printf 'Started tmux: %s\nTraining log: %s/train.log\n' "$session" "$output"
    exit 0
fi

mkdir -p "$output"
exec 9>"$output/.training.lock"
flock -n 9 || { echo "Training output already in use: $output" >&2; exit 1; }
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
cp -- "$config" "$output/config_base.yaml"
cp -- "$work_dir/resource_preflight.json" "$output/resource_preflight.json"
printf 'VARIANT=noaffine_nogamma\nGPUS=%s\nBATCH_SIZE=128\nGLOBAL_BATCH_SIZE=1024\nACCUM_ITER=1\nBLR=0.00005\nEFFECTIVE_LR=0.0002\nEMA1=0.9999\nEMA2=0.9996\nEPOCHS=80\nTMUX_SESSION=%s\n' \
    "$CUDA_VISIBLE_DEVICES" "$session" > "$output/run_info.txt"
hostname >> "$output/run_info.txt"
printf 'UPSTREAM_SOURCE=f8dbfc368f3d8b74193ea19c9ae73e2c0c85fead\nGENERATION=LoRA+MLS+final_LN_noaffine\nALIGNMENT=frozen_pretrained_affine+irepa_preprocess+forward_features\n' >> "$output/run_info.txt"
sha256sum dino_encoder.py dino_alignment_encoder.py dino_v3_lora.py \
    dino_jit_denoiser.py main_dino_jit.py jit/engine_jit.py jit/main_jit.py \
    "$config" "$script" > "$output/source_sha256.txt"
printf '%q ' "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=8 \
    main_dino_jit.py --config "$config" "${training_args[@]}" > "$output/train_args.txt"
printf '\n' >> "$output/train_args.txt"
bash "$project/scripts/train_portable.sh" "$config" "${training_args[@]}" 2>&1 | tee -a "$output/launcher.log"
