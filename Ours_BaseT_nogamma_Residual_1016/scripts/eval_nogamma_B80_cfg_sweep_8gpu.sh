#!/usr/bin/env bash
# Evaluate the completed global-batch-1024, lr-2e-4 run with EMA1.
set -euo pipefail
mode=tmux
gen_bsz=256
for arg in "$@"; do
    case "$arg" in
        --preflight) mode=preflight ;;
        --foreground) mode=foreground ;;
        --gen-bsz=*) gen_bsz="${arg#*=}" ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done
[[ "$gen_bsz" =~ ^[1-9][0-9]*$ ]] || exit 2
task_root=/mnt/shared-storage-user/datafactory-share/gushishen/agent-z
script="$(realpath -- "${BASH_SOURCE[0]}")"
project="$(cd -- "$(dirname -- "$script")/.." && pwd)"
experiment='BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm_nogamma-NoDDTHead-B-80'
checkpoint="$project/checkpoints/$experiment/checkpoint-80.pth"
config="$project/configs/$experiment.yaml"
eval_root="$project/checkpoints/eval_checkpoint/$experiment"
session="eval-nogamma-B80-cfgsweep-8gpu-b${gen_bsz}"
work_dir="$task_root/work/${session}-20261017"
scales=(2.3 2.5 2.7 2.9 3.1)
export PYTHON_BIN="$task_root/envs/irepa-py310-cu126-20261015/bin/python"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source "$project/scripts/portable_env.sh"
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
export DINOV3_REPO="$project/SSL_code/dinov3"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2 MAX_JOBS=2
export TORCHINDUCTOR_CACHE_DIR="/tmp/agent_z_nogamma_e80_sweep_b${gen_bsz}"
export TRITON_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}_triton"
cd -- "$project"
mkdir -p "$work_dir"

output_for() {
    printf '%s/checkpoint-80_cfg%s_interval0p0_1p0_flowdpm100_image50000_gpus8_bsz%s' "$eval_root" "${1/./p}" "$gen_bsz"
}

preflight() {
    "$PYTHON_BIN" scripts/check_runtime.py --sampling --config "$config" > "$work_dir/resource_preflight.json"
    "$PYTHON_BIN" -m evaluation.evaluate_fdr6 --check-assets --metrics=fid,inception_score,fdr6 \
        --cache-root="$NANOGEN_EVALS_ROOT" > "$work_dir/evaluation_assets.json"
    "$PYTHON_BIN" - "$checkpoint" "$experiment" "$work_dir" <<'PY'
import json, math, sys, torch
from pathlib import Path
path, experiment, work = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
assert json.loads((work/'evaluation_assets.json').read_text())['ready']
checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
args = vars(checkpoint['args'])
assert checkpoint['epoch'] == 80
expected = dict(exp_name=experiment, model='JiT-B/16', global_batch_size=1024,
                accum_iter=1, dino_model_name='dinov3_vitb16',
                dino_jit_input_mode='time_blend', dino_input_projector='pixeldit2',
                dino_jit_pre_blend_rmsnorm=True, use_ddt_head=False)
for key, value in expected.items():
    assert args[key] == value, (key, args[key], value)
assert math.isclose(args['lr'], 2e-4) and math.isclose(args['ema_decay1'], 0.9999)
for key in ('model', 'model_ema1', 'model_ema2'):
    assert checkpoint[key] and not any('pre_blend_norm.' in name for name in checkpoint[key])
report = dict(checkpoint=str(path), epoch=80, training_global_batch=1024,
              training_lr=args['lr'], ema='ema1', ema_decay=args['ema_decay1'],
              size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
(work/'checkpoint_validation.json').write_text(json.dumps(report, indent=2)+'\n')
PY
    for scale in "${scales[@]}"; do
        output="$(output_for "$scale")"
        "$PYTHON_BIN" -m evaluation.prepare_sampling --config="$config" --checkpoint="$checkpoint" \
            --output-dir="$output" --cfg="$scale" --sampler=flow_dpm_solver --steps=100 \
            --interval-min=0 --interval-max=1 --num-images=50000 --gen-bsz="$gen_bsz"
        cp -- "$work_dir/evaluation_assets.json" "$output/evaluation_assets.json"
    done
    echo "Preflight passed: nogamma epoch 80; CFG ${scales[*]}; 8 GPUs; gen_bsz=$gen_bsz; EMA1."
}

if [[ "$mode" == preflight ]]; then preflight; exit 0; fi
if [[ "$mode" == tmux ]]; then
    [[ "$(hostname)" == gpu-lg-cmc-h-h200-0257* ]] || { echo 'Run this sweep on agent-gpu-2 (H200-0257).' >&2; exit 1; }
    command -v tmux >/dev/null
    if tmux has-session -t "=$session" 2>/dev/null; then
        echo "Session already exists: $session" >&2; exit 1
    fi
    preflight
    nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits > "$work_dir/gpu_before.csv"
    "$PYTHON_BIN" - "$work_dir/gpu_before.csv" <<'PY'
import csv, sys
rows = list(csv.reader(open(sys.argv[1]), skipinitialspace=True))
assert {int(r[0]) for r in rows} == set(range(8))
assert all(int(r[1]) >= 32768 for r in rows), 'Insufficient free memory for sampling'
PY
    tmux new-session -d -s "$session" -c "$project" 'bash --noprofile --norc'
    tmux set-window-option -t "$session:0" remain-on-exit on
    printf -v command 'exec bash %q --foreground --gen-bsz=%q' "$script" "$gen_bsz"
    tmux send-keys -t "$session:0.0" -l "$command"
    tmux send-keys -t "$session:0.0" Enter
    printf 'Started tmux: %s\nSweep log: %s/sweep.log\n' "$session" "$work_dir"
    exit 0
fi

exec 9>"$work_dir/.sweep.lock"
flock -n 9 || { echo 'This sweep is already running.' >&2; exit 1; }
exec > >(tee -a "$work_dir/sweep.log") 2>&1
finish() {
    rc=$?
    trap - EXIT
    printf '%s\n' "$rc" > "$work_dir/exit_code.txt"
    date -Is > "$work_dir/finished_at.txt"
    exit "$rc"
}
trap finish EXIT
date -Is > "$work_dir/started_at.txt"
printf 'CHECKPOINT=%s\nGPUS=%s\nSCALES=%s\nSAMPLER=flow_dpm_solver\nSTEPS=100\nINTERVAL=[0,1]\nGEN_BSZ=%s\nNUM_IMAGES=50000\nEMA=ema1\nSEED=0\nTMUX_SESSION=%s\n' \
    "$checkpoint" "$CUDA_VISIBLE_DEVICES" "${scales[*]}" "$gen_bsz" "$session" > "$work_dir/run_info.txt"
hostname >> "$work_dir/run_info.txt"
failed=0
for scale in "${scales[@]}"; do
    output="$(output_for "$scale")"
    mkdir -p "$output"
    exec 8>"$output/.sampling.lock"
    if ! flock -n 8; then
        echo "Sampling output is locked: $output" >&2
        failed=1
        exec 8>&-
        continue
    fi
    printf '%s\n' "$scale" > "$work_dir/current_scale.txt"
    echo "[$(date -Is)] Starting CFG $scale on all 8 GPUs"
    date -Is > "$output/started_at.txt"
    set +e
    bash "$project/scripts/sample_portable.sh" "$config" "$checkpoint" "$output" \
        --cfg="$scale" --sampler=flow_dpm_solver --steps=100 --interval-min=0 --interval-max=1 \
        --num-images=50000 --gen-bsz="$gen_bsz" 2>&1 | tee -a "$output/launcher.log"
    status=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "$status" > "$output/exit_code.txt"
    date -Is > "$output/finished_at.txt"
    echo "[$(date -Is)] CFG $scale finished with exit code $status"
    if [[ "$status" -ne 0 ]]; then failed=1; fi
    "$PYTHON_BIN" - "$work_dir" "$eval_root" "$gen_bsz" <<'PY'
import csv, json, sys
from pathlib import Path
work, root, batch = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
with (work/'results.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['cfg', 'state', 'fid', 'inception_score', 'fdr6', 'output_dir'])
    for scale in ('2.3','2.5','2.7','2.9','3.1'):
        folder = root/f'checkpoint-80_cfg{scale.replace(".","p")}_interval0p0_1p0_flowdpm100_image50000_gpus8_bsz{batch}'
        code, metrics = folder/'exit_code.txt', folder/'offline_metrics.json'
        state = 'queued' if not code.exists() else ('complete' if code.read_text().strip() == '0' else 'failed')
        data = json.loads(metrics.read_text())['metrics'] if state == 'complete' and metrics.exists() else {}
        writer.writerow([scale, state, *[data.get(k,'') for k in ('fid','inception_score','fdr6')], str(folder)])
PY
    exec 8>&-
done
printf 'finished\n' > "$work_dir/current_scale.txt"
exit "$failed"
