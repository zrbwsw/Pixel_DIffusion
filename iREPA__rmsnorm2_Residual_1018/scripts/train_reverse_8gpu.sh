#!/usr/bin/env bash
# Two independent ablations of the completed Residual(1-t,t) run.
set -euo pipefail
variant="${1-}"
[[ $# -ge 1 ]] || { echo 'Usage: script {nomls|pmean0} [--preflight|--foreground] [--resume]' >&2; exit 2; }
shift
mode=tmux
resume=false
for option in "$@"; do
    case "$option" in
        --preflight) mode=preflight ;;
        --foreground) mode=foreground ;;
        --resume) resume=true ;;
        *) echo "Unknown option: $option" >&2; exit 2 ;;
    esac
done
case "$variant" in
    nomls)
        tag=NoMLS; target_host=agent-gpu; host_prefix=gpu-lg-cmc-h-h200-0029 ;;
    pmean0)
        tag=PMean0; target_host=gpu-2; host_prefix=gpu-lg-cmc-h-h200-0257 ;;
    *) echo "Unknown ablation: $variant" >&2; exit 2 ;;
esac
experiment="BaseT-Residual(1-t,t)-NoDinoPre-PgT-blend_rmsnorm2-${tag}-NoDDTHead-B-80"
devices=0,1,2,3,4,5,6,7
script="$(realpath -- "${BASH_SOURCE[0]}")"
project="$(cd -- "$(dirname -- "$script")/.." && pwd)"
task_root="$(dirname -- "$(dirname -- "$project")")"
config="$project/configs/reverse_${variant}_b80.yaml"
output="$project/checkpoints/$experiment"
session="train-irepa1018-reverse-${variant}-B80-8gpu-b128-gbs1024"
work_dir="$task_root/work/$session"
export PYTHON_BIN="${PYTHON_BIN:-$task_root/envs/irepa-py310-cu126-20261015/bin/python}"
export CUDA_VISIBLE_DEVICES="$devices"
export PRETRAIN_MODELS_ROOT="${PRETRAIN_MODELS_ROOT:-$(dirname -- "$project")/pretrain_models}"
export FID_STATS_ROOT="${FID_STATS_ROOT:-$(dirname -- "$project")/fid_stats}"
export IMAGENET_ROOT="${IMAGENET_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/imagenet}"
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$project:$project/jit"
export DINOV3_REPO="$project/SSL_code/dinov3"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2 MAX_JOBS=2
export TORCHINDUCTOR_CACHE_DIR="/tmp/agent_z_irepa1018_reverse_${variant}_8gpu_train"
export TRITON_CACHE_DIR="/tmp/agent_z_irepa1018_reverse_${variant}_8gpu_triton"
cd -- "$project"
resume_path=''
if [[ "$resume" == true ]]; then
    resume_path="$output"
fi
train_args=(--batch_size 128 --blr 0.00005 --resume "$resume_path")
preflight() {
    WORLD_SIZE=8 "$PYTHON_BIN" "$project/main.py" --config "$config" --resume "$resume_path" --preflight
}
if [[ "$mode" == preflight ]]; then preflight; exit 0; fi
[[ "$(hostname)" == "$host_prefix"* ]] || { echo "Run $variant on $target_host / $host_prefix (GPUs $devices)." >&2; exit 1; }
if [[ "$resume" == false && -d "$output" ]]; then
    "$PYTHON_BIN" - "$output" <<'PY'
from pathlib import Path
import sys
assert not [p for p in Path(sys.argv[1]).iterdir() if p.name != '.training.lock'], \
    'Output already contains a run; use --resume for this experiment'
PY
fi
mkdir -p "$work_dir"
preflight > "$work_dir/resource_preflight.json"
cat "$work_dir/resource_preflight.json"
"$PYTHON_BIN" - "$devices" <<'PY'
import csv, subprocess, sys
rows = list(csv.reader(subprocess.check_output([
    'nvidia-smi', '--id='+sys.argv[1], '--query-gpu=index,name,memory.free,uuid',
    '--format=csv,noheader,nounits'], text=True).splitlines(), skipinitialspace=True))
assert len(rows) == 8 and all('H200' in r[1] and int(r[2]) >= 100000 for r in rows), \
    'Expected eight H200 GPUs with at least 100000 MiB free each'
selected = {r[3] for r in rows}
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                '--format=csv,noheader'], text=True).splitlines()
assert not selected.intersection(a.strip() for a in apps), 'Selected GPUs already have compute processes'
PY
if [[ "$mode" == tmux ]]; then
    if tmux has-session -t "=$session" 2>/dev/null; then
        dead="$(tmux list-panes -t "=$session" -F '#{pane_dead}')"
        [[ "$resume" == true && "$dead" == 1 ]] || { echo "Existing tmux session: $session" >&2; exit 1; }
        tmux kill-session -t "=$session"
    fi
    tmux new-session -d -s "$session" -c "$project" 'bash --noprofile --norc'
    tmux set-window-option -t "$session:0" remain-on-exit on
    printf -v command 'exec env PYTHON_BIN=%q PRETRAIN_MODELS_ROOT=%q FID_STATS_ROOT=%q IMAGENET_ROOT=%q bash %q %q --foreground' \
        "$PYTHON_BIN" "$PRETRAIN_MODELS_ROOT" "$FID_STATS_ROOT" "$IMAGENET_ROOT" "$script" "$variant"
    [[ "$resume" == false ]] || command+=' --resume'
    tmux send-keys -t "$session:0.0" -l "$command"
    tmux send-keys -t "$session:0.0" Enter
    printf 'Started %s\nLog: %s/train.log\n' "$session" "$work_dir"
    exit 0
fi
# Keep launcher metadata outside the output directory so main.py can enforce
# a fresh output before it creates checkpoints/TensorBoard files.
exec 9>"$work_dir/.launcher.lock"
flock -n 9 || { echo "Launcher already active: $session" >&2; exit 1; }
date -Is > "$work_dir/started_at.txt"
cp -- "$config" "$work_dir/config.yaml"
sha256sum main.py generation_input.py resources.py dino_v3_lora.py dino_lora.py \
    pixeldit2_projector.py jit/*.py upstream_manifest.json "$config" "$script" > "$work_dir/source_sha256.txt"
printf '%q ' "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=8 \
    main.py --config "$config" "${train_args[@]}" > "$work_dir/train_args.txt"
printf '\n' >> "$work_dir/train_args.txt"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=8 \
    main.py --config "$config" "${train_args[@]}" 2>&1 | tee -a "$work_dir/train.log"
