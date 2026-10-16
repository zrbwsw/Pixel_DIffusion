#!/usr/bin/env bash
set -euo pipefail

[[ "$#" -ge 3 ]] || { echo "Usage: $0 CONFIG CHECKPOINT EVAL_DIR [--preflight] [--cfg=2.3 --gen-bsz=16 ...]" >&2; exit 2; }
config="$(realpath -- "$1")"
checkpoint="$(realpath -- "$2")"
eval_dir="$(realpath -m -- "$3")"
shift 3
source "$(dirname -- "${BASH_SOURCE[0]}")/portable_env.sh"
export HF_HUB_OFFLINE=1
preflight=false
sampling_args=(--cfg=2.3 --sampler=flow_dpm_solver --steps=100 --interval-min=0 --interval-max=1 --gen-bsz=16)
for arg in "$@"; do
    case "$arg" in
        --preflight) preflight=true ;;
        --cfg=*|--sampler=*|--steps=*|--gen-bsz=*|--num-images=*|--interval-min=*|--interval-max=*) sampling_args+=("$arg") ;;
        *) echo "Unknown option: $arg (use --name=value; set GPUs with CUDA_VISIBLE_DEVICES)" >&2; exit 2 ;;
    esac
done
cd -- "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/check_runtime.py --sampling --config "$config"
assets="$("$PYTHON_BIN" -m evaluation.evaluate_fdr6 --check-assets --cache-root="$NANOGEN_EVALS_ROOT")"
"$PYTHON_BIN" -m evaluation.prepare_sampling \
    --config="$config" --checkpoint="$checkpoint" --output-dir="$eval_dir" "${sampling_args[@]}"
printf '%s\n' "$assets" > "$eval_dir/evaluation_assets.json"
if [[ "$preflight" == true ]]; then
    echo "Sampling YAML and asset report prepared; no model was loaded or sampled."
    exit 0
fi
if [[ ! -f "$eval_dir/samples.npz" ]]; then
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" \
        main_dino_jit.py --config="$eval_dir/sampling_config.yaml" \
        2>&1 | tee -a "$eval_dir/sampling.log"
else
    echo "Completed NPZ exists; skip generation: $eval_dir/samples.npz"
fi
CUDA_VISIBLE_DEVICES="${gpus[0]}" "$PYTHON_BIN" -m evaluation.evaluate_fdr6 \
    --npz="$eval_dir/samples.npz" --metrics=fid,inception_score,fdr6 \
    --cache-root="$NANOGEN_EVALS_ROOT" --output-json="$eval_dir/offline_metrics.json" \
    2>&1 | tee -a "$eval_dir/offline_metrics.log"
