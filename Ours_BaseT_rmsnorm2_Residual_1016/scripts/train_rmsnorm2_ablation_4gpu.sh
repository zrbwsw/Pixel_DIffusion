#!/usr/bin/env bash
# Shared launcher for the three controlled rmsnorm2 ablations.
set -euo pipefail
variant="${1-}"
[[ "$#" -ge 1 ]] || { echo "Usage: $0 {noaffine|reverse|dropres} [--preflight|--foreground] [--resume]" >&2; exit 2; }
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
    noaffine)
        experiment='BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm2-NoAffine-NoDDTHead-B-80'
        devices=4,5,6,7; host_prefix=gpu-lg-cmc-h-h200-0029 ;;
    reverse)
        experiment='BaseT-Residual(1-t,t)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80'
        devices=0,1,2,3; host_prefix=gpu-lg-cmc-h-h200-0257 ;;
    dropres)
        experiment='BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm2-DropRes-NoDDTHead-B-80'
        devices=4,5,6,7; host_prefix=gpu-lg-cmc-h-h200-0257 ;;
    *) echo "Unknown ablation: $variant" >&2; exit 2 ;;
esac
script="$(realpath -- "${BASH_SOURCE[0]}")"
project="$(cd -- "$(dirname -- "$script")/.." && pwd)"
task_root="$(dirname -- "$(dirname -- "$project")")"
config="$project/configs/$experiment.yaml"
output="$project/checkpoints/$experiment"
session="train-rmsnorm2-${variant}-B80-4gpu-b256-gbs1024"
work_dir="$task_root/work/$session"
export PYTHON_BIN="${PYTHON_BIN:-$task_root/envs/irepa-py310-cu126-20261015/bin/python}"
export CUDA_VISIBLE_DEVICES="$devices"
source "$project/scripts/portable_env.sh"
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
export DINOV3_REPO="$project/SSL_code/dinov3"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2 MAX_JOBS=2
export TORCHINDUCTOR_CACHE_DIR="/tmp/agent_z_rmsnorm2_${variant}_train"
export TRITON_CACHE_DIR="/tmp/agent_z_rmsnorm2_${variant}_triton"
cd -- "$project"
resume_path=''
[[ "$resume" == false ]] || resume_path="$output"
train_args=(--batch_size 256 --global_batch_size 1024 --accum_iter 1 --blr 0.00005 --resume "$resume_path")

preflight() {
    "$PYTHON_BIN" - "$variant" "$config" "$output" "$resume_path" <<'PY'
import json, math, sys
from pathlib import Path
from scripts.check_runtime import inspect_runtime, training_args
variant, config, output, resume = sys.argv[1:]
root = Path.cwd()
base = root / 'configs/BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80.yaml'
baseline = training_args(['--config', str(base), '--batch_size', '256'])
args = training_args(['--config', config, '--resume', resume])
control, value = {
    'noaffine': ('dino_generation_final_norm_affine', False),
    'reverse': ('dino_time_blend_reverse', True),
    'dropres': ('dino_grounding_drop_on_label_drop', True),
}[variant]
assert getattr(args, control) is value
changed = {k for k in vars(baseline) if getattr(args, k) != getattr(baseline, k)}
assert changed == {'config', 'exp_name', 'output_dir', 'resume', 'metrics_log_name', control}, changed
report = inspect_runtime(args, world_size=4)
assert report['batch_size_per_gpu'] == 256 and report['global_batch_size'] == 1024
assert report['accum_iter'] == 1 and math.isclose(report['effective_lr'], 2e-4)
assert Path(report['output_dir']) == Path(output)
if resume:
    import torch
    assert report['resume_checkpoint'], 'No checkpoint-last.pth for this ablation'
    checkpoint = torch.load(report['resume_checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    saved = checkpoint['args']
    saved = saved if isinstance(saved, dict) else vars(saved)
    keys = ('model', 'dino_generation_final_norm_affine', 'dino_time_blend_reverse',
            'dino_grounding_drop_on_label_drop', 'dino_jit_pre_blend_rmsnorm',
            'dino_jit_input_mode', 'dino_input_projector', 'dino_preprocess_input',
            'dino_aggregation', 'dino_layers', 'dino_model_name', 'dino_lora_r',
            'dino_lora_alpha', 'use_ddt_head', 'hflip_prob', 'label_drop_prob',
            'global_batch_size', 'accum_iter', 'blr', 'lr_schedule', 'epochs',
            'warmup_epochs', 'ema_decay1', 'ema_decay2')
    for key in keys:
        assert saved.get(key) == getattr(args, key), f'Resume setting mismatch: {key}'
    assert 0 < checkpoint['epoch'] < args.epochs, 'Checkpoint already complete or invalid'
    report['resume_epoch'] = checkpoint['epoch']
else:
    assert report['training_mode'] == 'from_scratch'
report.update(variant=variant, controls={
    key: getattr(args, key) for key in ('dino_generation_final_norm_affine',
                                      'dino_time_blend_reverse', 'dino_grounding_drop_on_label_drop')})
print(json.dumps(report, indent=2))
PY
}
if [[ "$mode" == preflight ]]; then preflight; exit 0; fi
[[ "$(hostname)" == "$host_prefix"* ]] || { echo "Run $variant on $host_prefix ($devices)." >&2; exit 1; }
if [[ "$resume" == false && -d "$output" ]]; then
    "$PYTHON_BIN" - "$output" <<'PY'
from pathlib import Path
import sys
assert not [p for p in Path(sys.argv[1]).iterdir() if p.name != '.training.lock'], \
    'Output already contains a run; use --resume with its checkpoint, or a new experiment'
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
assert len(rows) == 4 and all('H200' in r[1] and int(r[2]) >= 100000 for r in rows)
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
    printf -v command 'exec env PYTHON_BIN=%q PRETRAIN_MODELS_ROOT=%q FID_STATS_ROOT=%q NANOGEN_EVALS_ROOT=%q IMAGENET_ROOT=%q bash %q %q --foreground' \
        "$PYTHON_BIN" "$PRETRAIN_MODELS_ROOT" "$FID_STATS_ROOT" "$NANOGEN_EVALS_ROOT" "$IMAGENET_ROOT" "$script" "$variant"
    [[ "$resume" == false ]] || command+=' --resume'
    tmux send-keys -t "$session:0.0" -l "$command"
    tmux send-keys -t "$session:0.0" Enter
    printf 'Started %s\nLog: %s/train.log\n' "$session" "$output"
    exit 0
fi
mkdir -p "$output"
exec 9>"$output/.training.lock"
flock -n 9 || { echo "Output already in use: $output" >&2; exit 1; }
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$output/exit_code.txt"
    date -Is > "$output/finished_at.txt"
    exit "$status"
}
trap finish EXIT
date -Is > "$output/started_at.txt"
rm -f -- "$output/finished_at.txt" "$output/exit_code.txt"
cp -- "$config" "$output/config_base.yaml"
cp -- "$work_dir/resource_preflight.json" "$output/resource_preflight.json"
printf 'VARIANT=%s\nGPUS=%s\nBATCH_SIZE=256\nGLOBAL_BATCH_SIZE=1024\nACCUM_ITER=1\nEFFECTIVE_LR=0.0002\nEPOCHS=80\nTMUX_SESSION=%s\n' \
    "$variant" "$devices" "$session" > "$output/run_info.txt"
hostname >> "$output/run_info.txt"
sha256sum dino_encoder.py dino_v3_lora.py dino_jit_denoiser.py main_dino_jit.py \
    jit/main_jit.py jit/engine_jit.py "$config" "$script" > "$output/source_sha256.txt"
printf '%q ' "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=4 \
    main_dino_jit.py --config "$config" "${train_args[@]}" > "$output/train_args.txt"
printf '\n' >> "$output/train_args.txt"
bash "$project/scripts/train_portable.sh" "$config" "${train_args[@]}" 2>&1 | tee -a "$output/launcher.log"
